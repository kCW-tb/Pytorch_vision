import torch

def get_modules(use_v2):
    # We need a protected import to avoid the V2 warning in case just V1 is used
    if use_v2:
        print("Use_V2")
        import torchvision.transforms.v2
        import torchvision.tv_tensors
        import v2_extras

        return torchvision.transforms.v2, torchvision.tv_tensors, v2_extras
    else:
        print("Use_V1")
        import transforms
        
        return transforms, None, None

#transforms에 직접적으로 넣는다 segmentation은
class SegmentationPresetTrain:
    def __init__(
        self,
        *,
        base_size,
        crop_size,
        hflip_prob=0.5,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        backend="pil",
        use_v2=False,
    ):
        T, tv_tensors, v2_extras = get_modules(use_v2)
        
        if tv_tensors==None:
            import torchvision.transforms as V1
        
        transforms = []
        backend = backend.lower()
        if backend == "tv_tensor":
            transforms.append(T.ToImage())
        elif backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        #V2를 사용할 때 사용자가 원하는 변환 방식 추가 공간.
        if use_v2:
            #transforms += [T.RandomResize(min_size=int(0.5 * base_size), max_size=int(2.0 * base_size))]
            #transforms += [T.ColorJitter(saturation=0.5, hue=0.5, contrast=0.5)]
            transforms += [T.RandomRotation((-10,10))]
            #GrayScale사용하면 되려 손해?
            #transforms += [T.RandomGrayscale(p=0.2)]
            #transforms += [T.GaussianBlur(3)]
            #transforms += [T.RandomAffine(degrees=10, )]
            
        #V1를 사용할 때 사용자가 원하는 변환 방식 추가 공간.
        # alpha=50.0, sigma=5.0, interpolation=InterpolationMode.BILINEAR, fill=0):
        if not use_v2:
            transforms += [T.RandomRotations(degrees=20)]
            transforms += [T.ColorJitter(brightness=(1.0, 1.2), saturation=(0.5,1.5), hue=(-0.3, 0.5))]
            transforms += [T.RandomGrayscale(0.1)]
            transforms += [T.RandomErasing()]
            #GaussinaBlur와 ElasticTransform은 서로 겹치지 않게.
            transforms += [T.GaussianBlur(kernel_size=(5,8), sigma=(0.1,2.0))]
            #transforms += [T.ElasticTransform(alpha=100.0,sigma=5.0)]
        
        if hflip_prob > 0:
            transforms += [T.RandomHorizontalFlip(hflip_prob)]

        if use_v2:
            # We need a custom pad transform here, since the padding we want to perform here is fundamentally
            # different from the padding in `RandomCrop` if `pad_if_needed=True`.
            transforms += [v2_extras.PadIfSmaller(crop_size, fill={tv_tensors.Mask: 255, "others": 0})]
        
        transforms += [T.RandomCrop(crop_size)]

        if backend == "pil":
            transforms += [T.PILToTensor()]

        if use_v2:
            img_type = tv_tensors.Image if backend == "tv_tensor" else torch.Tensor
            transforms += [
                T.ToDtype(dtype={img_type: torch.float32, tv_tensors.Mask: torch.int64, "others": None}, scale=True)
            ]
        else:
            # No need to explicitly convert masks as they're magically int64 already
            transforms += [T.ToDtype(torch.float, scale=True)]

        transforms += [T.Normalize(mean=mean, std=std)]
        
        if use_v2:
            transforms += [T.ToPureTensor()]
            
        print("Train 데이터 증강 Compose 내역.")
        for t in transforms:
            print(t)
        print()
        self.transforms = T.Compose(transforms)
        
    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(
        self, *, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), backend="pil", use_v2=False
    ):
        T, _, _ = get_modules(use_v2)
        print("SegmentationPresetEval Using V2?? : ", use_v2)
        print("SegmentationPresetEval Using beckend?? : " , backend)
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms += [T.PILToTensor()]
        elif backend == "tv_tensor":
            transforms += [T.ToImage()]
        elif backend != "pil":
            raise ValueError(f"backend can be 'tv_tensor', 'tensor' or 'pil', but got {backend}")

        if use_v2:
            transforms += [T.Resize(size=(base_size, base_size))]
        else:
            transforms += [T.RandomResize(min_size=base_size, max_size=base_size)]

        if backend == "pil":
            # Note: we could just convert to pure tensors even in v2?
            transforms += [T.ToImage() if use_v2 else T.PILToTensor()]

        transforms += [
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
        ]

        if use_v2:
            transforms += [T.ToPureTensor()]
        
        print("Val 데이터 증식")
        for t in transforms:
            print(t)
        print()
        
        self.transforms = T.Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)
