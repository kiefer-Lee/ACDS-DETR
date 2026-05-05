import random
from typing import Tuple

import torch
import torchvision.transforms.functional as F


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resize_keep_ratio(image, target, short_size: int, max_size: int):
    w, h = image.size
    scale = short_size / min(h, w)
    if max(h, w) * scale > max_size:
        scale = max_size / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    image = F.resize(image, [new_h, new_w])
    if target["boxes"].numel() > 0:
        target["boxes"] = target["boxes"] * torch.tensor([scale, scale, scale, scale], dtype=torch.float32)
        target["area"] = target["area"] * (scale * scale)
    target["size"] = torch.tensor([new_h, new_w], dtype=torch.int64)
    return image, target


def hflip(image, target):
    w, _ = image.size
    image = F.hflip(image)
    boxes = target["boxes"]
    if boxes.numel() > 0:
        x0 = w - boxes[:, 2]
        x1 = w - boxes[:, 0]
        boxes[:, 0] = x0
        boxes[:, 2] = x1
        target["boxes"] = boxes
    return image, target


def to_tensor_and_normalize(image, target):
    image = F.to_tensor(image)
    image = F.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
    return image, target


def apply_transforms(image, target, train: bool, img_size: int, max_size: int):
    image, target = resize_keep_ratio(image, target, img_size, max_size)
    if train and random.random() < 0.5:
        image, target = hflip(image, target)
    return to_tensor_and_normalize(image, target)

