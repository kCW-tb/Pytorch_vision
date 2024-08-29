Segmantic Segmentation 작업만 수행.

#my_train.py 코드 수정

백본의 가중치를 고정시켜 훈련하지 않도록 하는 작업.
```
    for param in model.backbone.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        print(name, param.requires_grad)
```


모델의 구조와 가중치를 모두 저장하는 코드
```
utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
torch.save(model_without_ddp, os.path.join(args.output_dir, "model.pth"))
```
이후 생성되는 pth모델은 잘 사용할 수 있었으나 onnx파일로 바꾸는 과정에서는 기본으로 생성하는 checkpoint.pth만을 사용함.
이유는 onnx로 변환하는 과정에서 필요한 데이터는 가중치와 손실함수 등으로 고정되어있어 모델의 구조까지 저장하는 model.pth는 사용할 수 없었다.


coco_utils.py의 코드를 학습할 데이터에 맞춰 수정.
```
anno_file_template = "annotations.json"

PATHS = {
"train": ("train", os.path.join("train", anno_file_template)),
"val": ("val", os.path.join("val", anno_file_template)),
}
CAT_LIST = [0, 1, 2]
```
위 코드는 background, floor, blockage 3개의 클래스로 이루어진 코드임.
예제에서 coco dataset의 폴더구조와 파일을 labelme에서 만들어주는 데이터셋 폴더 구조에 맞게 수정하는 과정.


v2_extras.py 코드 수정.
```
COCO_TO_VOC_LABEL_MAP = dict(
        zip(
            [ 0, 1, 2],
            range(3),
        )
    )
```
데이터 증식을 수행할 때 클래스의 개수를 일치시켜주는 과정으로 V1을 사용할 때에는 상관 없으나
이후 --use-v2 명령행 인자 사용시 쓰인다.


evaluate함수 수정.
```
def evaluate(model, data_loader, device, num_classes, epoch, cut_train):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: [{epoch}] cut : " + cut_train
    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, 30, header):
            image, target = image.to(device), target.to(device)
            #if문으로 Data_loader가 Val일 때에만 진행
            scaler = torch.cuda.amp.GradScaler() if args.amp else None
            model_without_ddp = model
            params_to_optimize = [
                {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
                {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
            ]
            if args.aux_loss:
                params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
                params_to_optimize.append({"params": params, "lr": args.lr * 10})

            optimizer = torch.optim.SGD(params_to_optimize, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                output = model(image)
                loss = criterion(output, target)
            metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
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
```
본래 evaluate함수는 Acc값을 평가하는 함수로. MatricLogger와 ConfusionMatrix클래스를 오가며 평가한다.
여기에서 loss값을 계산하는 코드를 추가하여 evaluate함수 내부에서도 data_loader에 들어가는 데이터셋에 대하여 loss값을 계산하도록 수정.

evaluate에서 반환되는 confmat객체에 대해 각 클래스 별로 정확도를 뽑아 저장하는 코드.
```
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, scaler)
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, epoch=epoch, cut_train="V")
        train_confmat = evaluate(model, data_loader, device=device, num_classes=num_classes, epoch=epoch, cut_train="T")
        
        
        print("검증 데이터 acc 및 IoU")
        print("[_background_, floor, blockage]")
        print(confmat)

        print("훈련 데이터 acc 및 IoU")
        print("[_background_, floor,  blockage]")
        print(train_confmat)

        acc_gloval, class_list_acc, ius = confmat.compute()
        class_list_acc = (class_list_acc * 100).tolist()
        
        train_acc_gloval, train_class_list_acc, train_ius = train_confmat.compute()
        train_class_list_acc = (train_class_list_acc * 100).tolist()
        
        
        # 5 -> 3
        for i in range(3):
            acc_data[i].append(class_list_acc[i])
        
        for i in range(3):
            train_acc_data[i].append(train_class_list_acc[i])
```
위 코드를 진행하면 acc_data와 train_acc_data 이중리스트 내부에 값이 저장된다.

![image](https://github.com/user-attachments/assets/f9a09362-bc1b-4771-a460-ba6e883c1423)

저장하는 이중리스트 공간은 main함수 내부 초반에 미리 생성해주었다.
사진은 총 5개의 클래스로 이루어진(background포함) 코드에 대해서 생성해주었으나 이후에는 3개의 클래스로 줄여 학습하였다.

