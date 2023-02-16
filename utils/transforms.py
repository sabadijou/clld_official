from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import random
import torch
import math


class BaseTransform(ABC):
    def __init__(self, prob, mag):
        self.prob = prob
        self.mag = mag

    def __call__(self, img):
        return transforms.RandomApply([self.transform], self.prob)(img)

    @abstractmethod
    def transform(self, img):
        pass


class Solarize(BaseTransform):
    def transform(self, img):
        thres = (1-self.mag) * 255
        return ImageOps.solarize(img, thres)


class GaussianBlur(BaseTransform):
    def transform(self, img):
        kernel_size = self.mag
        gBlur = ImageFilter.GaussianBlur(kernel_size)
        return img.filter(gBlur)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        is_flip = 0
        if torch.rand(1) < self.p:
            is_flip = 1
            return img.transpose(method=Image.FLIP_LEFT_RIGHT), is_flip
        return img, is_flip


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=Image.BILINEAR):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        width, height = img.size
        area = width * height

        for _ in range(10):
            target_area = random.uniform(*scale) *area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height -h)
                j = random.randint(0, width - w)
                return i, j, h, w

        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = img.crop((j, i, j+w, i+h))
        img = img.resize(self.size)
        return img, j, i, w, h
