import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size, antialias=True)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ToDtype:
    def __init__(self, dtype, scale=False):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, image, target):
        if not self.scale:
            return image.to(dtype=self.dtype), target
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
    
    
class RandomRotations:
    def __init__(self, degrees=10 , interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0):
        self.degrees = degrees
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill
    
    def __call__(self, image, target):
        if random.random() < 0.5:
            self.degrees = -(self.degrees)
        image = F.rotate(image, self.degrees, self.interpolation, self.expand, self.center, self.fill)
        target = F.rotate(target, self.degrees, self.interpolation, self.expand, self.center, self.fill)
        return image, target
    
    
class ColorJitter:
    def __init__(self, brightness=(0.7, 1.1), hue=(-0.2, 0.2), contrast=(0.5, 1.0), saturation=(0.9, 1.0)):
        self.brightness = brightness
        self.hue = hue
        self.contrast = contrast
        self.saturation = saturation
    
    def chose_number(self, value):
        random_value = random.uniform(value[0], value[1])
        rounded_value = round(random_value, 1)
        return rounded_value
    
    def __call__(self, image, target):
        brightness = self.chose_number(self.brightness)
        saturation = self.chose_number(self.saturation)
        hue = self.chose_number(self.hue)
        contrast = self.chose_number(self.contrast)
        image = F.adjust_brightness(image, brightness)
        image = F.adjust_saturation(image, saturation)
        image = F.adjust_hue(image, hue)
        image = F.adjust_contrast(image, contrast)

        return image, target
    
    
class RandomGrayscale:
    def __init__(self, p=0.1):
        self.p = p
        
    def __call__(self, image, target):
        num_output_channels, _, _ = F.get_dimensions(image)
        if torch.rand(1) < self.p:
            return F.rgb_to_grayscale(image, num_output_channels=num_output_channels), target

        return image, target


class RandomErasing:
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def chose_number(self, value):
        random_value = random.uniform(value[0], value[1])
        rounded_value = round(random_value, 2)
        return rounded_value    

    def __call__(self, image, target):
        scale = self.chose_number(value=self.scale)
        ratio = self.chose_number(value=self.ratio)
        transform = T.ToTensor()
        tensor_image = transform(image)
        #tensor_target = transform(target)
        
        img_c, img_h, img_w = tensor_image.shape
        image_height = img_h
        image_width = img_w
        #height and width
        erase_height = image_height * scale
        erase_width = image_width * scale
        #start x and y
        erase_height = int(erase_height)
        erase_width = int(erase_height)
        if image_width > erase_width:
            x = random.randint(0, image_width - erase_width)
        else:
            x = 0
        if image_height > erase_height:
            y = random.randint(0, image_height - erase_height)
        else:
            y = 0
        if random.random() <= self.p:
            image = transform(image)
            target = transform(target)
            image = F.erase(image, x, y, erase_height, erase_width, self.value)
            target = F.erase(target, x, y, erase_height, erase_width, self.value)
            transform = T.ToPILImage()
            image = transform(image)
            target = transform(target)
        
        return image, target

        
#kernel_size는 튜플로 입력할 것. 고정시킬 값이라면 (5,5)같이 할 것.
class GaussianBlur:
    def __init__(self, kernel_size=(5, 9), sigma=(0.1, 2.0)):
        self.kernel = kernel_size
        self.sigma_min = sigma[0]
        self.sigma_max = sigma[1]
    def get_params(self, min, max):
        return torch.empty(1).uniform_(min, max).item()
    def get_kernel(self, kernel_size):
        return random.randrange(kernel_size[0], kernel_size[1])
    def __call__(self, image, target):
        sigma = self.get_params(self.sigma_min, self.sigma_max)
        kernel_int = self.get_kernel(self.kernel)
        if kernel_int % 2 == 0:
            kernel_int += 1
        image = F.gaussian_blur(image, kernel_int, [sigma, sigma])
        return image, target
    

class ElasticTransform:
    def __init__(self, alpha=50.0, sigma=5, interpolation=InterpolationMode.BILINEAR, fill=0):
        self.alpha = alpha
        self.sigma = sigma
        self.interpolation = interpolation
        self.fill = fill
    def __call__(self, image, target):
        _, height, width = F.get_dimensions(image)
        _, m_height, m_width = F.get_dimensions(target)
        displacement = T.ElasticTransform.get_params(self.alpha, self.sigma, [height, width])
        image = F.elastic_transform(image, displacement, self.interpolation, self.fill)
        m_displacement = T.ElasticTransform.get_params(self.alpha, self.sigma, [m_height, m_width])
        target = F.elastic_transform(target, m_displacement, self.interpolation, self.fill)
        
        return image, target