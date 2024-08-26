import datetime
import os
import time
import warnings
from IPython import display

import presets
import torch
import torch.utils.data
import torchvision
import utils
import numpy as np
import matplotlib.pyplot as plt
from coco_utils import get_coco
from torch import nn
from torch.optim.lr_scheduler import PolynomialLR
from torchvision.transforms import functional as F, InterpolationMode
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import numpy as np

def decode_segmap(image, nc=21):
        label_colors = np.array([(0, 0, 0),  # 0=background
                # 1=ceiling, 2=floor, 3=wall, 4=blockage, 5=bottle
                (128, 0, 0), (255, 255, 255), (128, 128, 0), (255, 0, 0), (128, 0, 128),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]  
            rgb = np.stack([r, g, b], axis=2)
        return rgb

def get_dataset(args, is_train):
    def sbd(*args, **kwargs):
        kwargs.pop("use_v2")
        return torchvision.datasets.SBDataset(*args, mode="segmentation", **kwargs)

    def voc(*args, **kwargs):
        kwargs.pop("use_v2")
        return torchvision.datasets.VOCSegmentation(*args, **kwargs)

    paths = {
        "voc": (args.data_path, voc, 21),
        "voc_aug": (args.data_path, sbd, 21),
        "coco": (args.data_path, get_coco, 3),
    }
    p, ds_fn, num_classes = paths[args.dataset]

    image_set = "train" if is_train else "val"
    ds = ds_fn(p, image_set=image_set, transforms=get_transform(is_train, args), use_v2=args.use_v2)
    return ds, num_classes

def get_transform(is_train, args):
    if is_train:
        return presets.SegmentationPresetTrain(base_size=520, crop_size=480, backend=args.backend, use_v2=args.use_v2)
    elif args.weights and args.test_only:
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()

        def preprocessing(img, target):
            img = trans(img)
            size = F.get_dimensions(img)[1:]
            target = F.resize(target, size, interpolation=InterpolationMode.NEAREST)
            return img, F.pil_to_tensor(target)

        return preprocessing
    else:
        return presets.SegmentationPresetEval(base_size=520, backend=args.backend, use_v2=args.use_v2)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    #header가 Test로 들어가면 Metric_logger.log_every에서 
    header = "Test:"
    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output["out"]

            confmat.update(target.flatten(), output.argmax(1).flatten())
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            num_processed_samples += image.shape[0]

        confmat.reduce_from_all_processes()

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    return confmat

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    header = f"Epoch: [{epoch}]"
    #log_every에서 Epoch별 내용이 모두 출력된다. (log_every)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])


def main(args):
    writer = SummaryWriter('./runs')
    #acc 데이터 저장할 공간.
    acc_data = [[],[],[],[],[]]
    train_acc_data = [[],[],[],[],[]]
        
    if args.backend.lower() != "pil" and not args.use_v2:
        # TODO: Support tensor backend in V1?
        raise ValueError("Use --use-v2 if you want to use the tv_tensor or tensor backend.")
    if args.use_v2 and args.dataset != "coco":
        raise ValueError("v2 is only support supported for coco dataset for now.")

    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        
    #이거 확인해야함 Train은 num_classes가 있는데 Val은 없다
    #여기에서 출력이 되고 안되고가 나오는지도.
    dataset, num_classes = get_dataset(args, is_train=True)
    dataset_test, num_classes_test = get_dataset(args, is_train=False)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=utils.collate_fn,
        drop_last=True,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=utils.collate_fn
    )
    print("데이터 증식 이미지 출력.")
    # Display image and label.
    '''
        def decode_segmap(image, nc=21):
        label_colors = np.array([(0, 0, 0),  # 0=background
                # 1=ceiling, 2=floor, 3=wall, 4=blockage, 5=bottle
                (128, 0, 0), (255, 255, 255), (128, 128, 0), (255, 0, 0), (128, 0, 128),
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]  
            rgb = np.stack([r, g, b], axis=2)
        return rgb
    '''
    print(len(data_loader))
    train_img, train_labels = next(iter(data_loader))
    print(len(train_img))
    print(len(train_labels))
    '''
    print("Point plt")
    print(train_img[0])  
    print("Point plt labels")  
    print(train_labels[0])
    '''
    plt.subplot(121),plt.imshow(to_pil_image(train_img[0]))    
    plt.subplot(122),plt.imshow(decode_segmap(train_labels[0]))
    plt.show()
    print("예제 코드 종료")
    
    model = torchvision.models.get_model(
        args.model,
        weights=args.weights,
        weights_backbone=args.weights_backbone,
        num_classes=num_classes,
        aux_loss=args.aux_loss,
    )
    model.to(device)
    # freeze backbone weights
    for param in model.backbone.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    iters_per_epoch = len(data_loader)
    main_lr_scheduler = PolynomialLR(
        optimizer, total_iters=iters_per_epoch * (args.epochs - args.lr_warmup_epochs), power=0.9
    )

    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=warmup_iters
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        #checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        checkpoint = torch.load(args.resume, map_location="cuda", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.amp:
                scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        
        print(confmat)
        return

    start_time = time.time()
    #plt.figure(figsize=(16,8))
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, scaler)
        
        #loss값 등 출력이 나오는 곳 confmat는 valid data train_confmat는 학습 정확도 등을 출력하고 싶은 느거임.
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        #추가 코드.
        train_confmat = evaluate(model, data_loader, device=device, num_classes=num_classes)
        
        print("검증 데이터 acc 및 IoU")
        print("[_background_, ceiling, floor, wall, blockage]")
        print(confmat)
        
        #utils ConfusionMatrix에서 확인이 필요함.
        print("훈련 데이터 acc 및 IoU")
        print("[_background_, ceiling, floor, wall, blockage]")
        print(train_confmat)
        #각 클래스별로 정확도 뽑기.
        acc_gloval, class_list_acc, ius = confmat.compute()
        class_list_acc = (class_list_acc * 100).tolist()
        
        train_acc_gloval, train_class_list_acc, train_ius = confmat.compute()
        train_class_list_acc = (train_class_list_acc * 100).tolist()
        
        
        # 5 -> 3
        for i in range(3):
            acc_data[i].append(class_list_acc[i])
        
        for i in range(3):
            train_acc_data[i].append(train_class_list_acc[i])    
        
        
        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": args,
        }
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
        utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
        #그래프 그리는 코드 ACC 관련. + pyplot방법
        '''
        if(epoch != 0):
            #2행 1열의 데이터에서 1행으로 설정
            x_arr = np.arange(epoch + 1)
            print('그래프 그리기 x_arr : ', str(x_arr))
            #plt.subplot(2,1,2)
            plt.plot(x_arr, acc_data[0], '-o', label='back ground', color='black')
            plt.plot(x_arr, acc_data[1], '-o', label='ceiling', color='gray')
            plt.plot(x_arr, acc_data[2], '-o', label='chair', color='red')
            plt.plot(x_arr, acc_data[3], '-o', label='door', color='blue')
            plt.plot(x_arr, acc_data[4], '-o', label='floor', color='yellow')
            plt.plot(x_arr, acc_data[5], '-o', label='glassdoor', color='green')
            plt.plot(x_arr, acc_data[6], '-o', label='table', color='magenta')
            plt.plot(x_arr, acc_data[7], '-o', label='wall', color='blueviolet')
            plt.plot(x_arr, acc_data[8], '-o', label='window', color='chocolate')
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            plt.xlabel('Epoch', size=15)
            plt.ylabel('Acc', size=15)
            
            plt.ion()
            plt.show(block=False) #멈춤 방지 계속해서 코드 진행.
            plt.pause(0.1) #렉걸림 방지.
            display.clear_output(wait=True)
        '''
        #값을 추가하는거라 마지막 인덱스만 추가해줘야 함.
        '''
        writer.add_scalars(
            'classes_Val_Acc', {'back_ground':acc_data[0][-1],
                            'ceiling':acc_data[1][-1],
                            'floor':acc_data[2][-1],
                            'wall':acc_data[3][-1],
                            'blockage':acc_data[4][-1],
                            },
            epoch
        )
        '''
        writer.add_scalars(
            'classes_Val_Acc', {'back_ground':acc_data[0][-1],
                            'floor':acc_data[1][-1],
                            'blockage':acc_data[2][-1],
                            },
            epoch
        )
        writer.add_scalars(
            'classes_Train_Acc', {'back_ground':train_acc_data[0][-1],
                            'floor':train_acc_data[1][-1],
                            'blockage':train_acc_data[2][-1],
                            },
            epoch
        )
        
        
    #모델의 구조와 가중치 모두를 저장
    torch.save(model_without_ddp, os.path.join(args.output_dir, "model.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    
    writer.flush()
    writer.close()

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data-path", default="/datasets01/COCO/022719/", type=str, help="dataset path")
    parser.add_argument("--dataset", default="coco", type=str, help="dataset name")
    parser.add_argument("--model", default="fcn_resnet101", type=str, help="model name")
    parser.add_argument("--aux-loss", action="store_true", help="auxiliary loss")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=30, type=int, metavar="N", help="number of total epochs to run")

    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency") #빈도 에포크별로 출력하는 거
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    #resume 요거 잘 살펴보기.
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
