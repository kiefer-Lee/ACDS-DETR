import random

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


def sanitize_boxes(target, min_size=1.0):
    boxes = target["boxes"]
    if boxes.numel() == 0:
        return target
    h, w = target["size"].tolist()
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, w)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, h)
    keep = (boxes[:, 2] - boxes[:, 0] >= min_size) & (boxes[:, 3] - boxes[:, 1] >= min_size)
    target["boxes"] = boxes[keep]
    target["labels"] = target["labels"][keep]
    target["area"] = ((boxes[keep, 2] - boxes[keep, 0]) * (boxes[keep, 3] - boxes[keep, 1])).to(target["area"].dtype)
    target["iscrowd"] = target["iscrowd"][keep]
    return target


def random_zoom_crop(image, target, ratio_range=(0.6, 1.0), min_boxes=1, attempts=10):
    w, h = image.size
    if target["boxes"].numel() == 0:
        return image, target
    for _ in range(attempts):
        ratio = random.uniform(float(ratio_range[0]), float(ratio_range[1]))
        crop_w = max(1, int(round(w * ratio)))
        crop_h = max(1, int(round(h * ratio)))
        if crop_w >= w and crop_h >= h:
            return image, target
        left = random.randint(0, max(0, w - crop_w))
        top = random.randint(0, max(0, h - crop_h))
        boxes = target["boxes"]
        centers_x = (boxes[:, 0] + boxes[:, 2]) * 0.5
        centers_y = (boxes[:, 1] + boxes[:, 3]) * 0.5
        keep = (centers_x >= left) & (centers_x <= left + crop_w) & (centers_y >= top) & (centers_y <= top + crop_h)
        if int(keep.sum()) < min_boxes:
            continue
        nt = dict(target)
        nt["boxes"] = boxes[keep].clone()
        nt["labels"] = target["labels"][keep]
        nt["area"] = target["area"][keep]
        nt["iscrowd"] = target["iscrowd"][keep]
        nt["boxes"][:, 0::2] -= left
        nt["boxes"][:, 1::2] -= top
        nt["size"] = torch.tensor([crop_h, crop_w], dtype=torch.int64)
        cropped = F.crop(image, top, left, crop_h, crop_w)
        return cropped, sanitize_boxes(nt)
    return image, target


def to_tensor_and_normalize(image, target):
    image = F.to_tensor(image)
    image = F.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
    return image, target


def apply_transforms(image, target, train: bool, img_size: int, max_size: int, augment=None):
    augment = augment or {}
    if train and random.random() < float(augment.get("zoom_crop_prob", 0.0)):
        image, target = random_zoom_crop(
            image,
            target,
            ratio_range=augment.get("zoom_crop_ratio", [0.6, 1.0]),
            min_boxes=int(augment.get("zoom_crop_min_boxes", 1)),
            attempts=int(augment.get("zoom_crop_attempts", 10)),
        )
    short_size = img_size
    multi_scale = augment.get("multi_scale", [])
    if train and multi_scale:
        short_size = int(random.choice(multi_scale))
    image, target = resize_keep_ratio(image, target, short_size, max_size)
    if train and random.random() < 0.5:
        image, target = hflip(image, target)
    return to_tensor_and_normalize(image, target)
