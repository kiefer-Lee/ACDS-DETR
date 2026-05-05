from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import apply_transforms


VISDRONE_CLASSES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


class VisDroneDetection(Dataset):
    def __init__(self, root, split="train", img_size=800, max_size=1333, train=True, min_area=1, max_samples=None):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.max_size = max_size
        self.train = train
        self.min_area = min_area
        split_name = "VisDrone2019-DET-train" if split == "train" else "VisDrone2019-DET-val"
        base = self.root / split_name / split_name
        if not base.exists():
            raise FileNotFoundError(f"VisDrone split directory not found: {base}")
        self.image_dir = base / "images"
        self.ann_dir = base / "annotations"
        self.images = sorted(self.image_dir.glob("*.jpg"))
        if max_samples is not None:
            self.images = self.images[: int(max_samples)]
        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def _read_annotation(self, ann_path):
        boxes, labels, areas, iscrowd = [], [], [], []
        if not ann_path.exists():
            return boxes, labels, areas, iscrowd
        with ann_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                x, y, w, h = map(float, parts[:4])
                score = int(float(parts[4]))
                cls = int(float(parts[5]))
                trunc = int(float(parts[6]))
                occ = int(float(parts[7]))
                if score == 0 or cls < 1 or cls > 10 or w * h < self.min_area:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(cls - 1)
                areas.append(w * h)
                iscrowd.append(0)
        return boxes, labels, areas, iscrowd

    def __getitem__(self, idx):
        image_path = self.images[idx]
        ann_path = self.ann_dir / f"{image_path.stem}.txt"
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        boxes, labels, areas, iscrowd = self._read_annotation(ann_path)
        target = {
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            "orig_size": torch.tensor([h, w], dtype=torch.int64),
            "size": torch.tensor([h, w], dtype=torch.int64),
            "file_name": str(image_path),
        }
        image, target = apply_transforms(image, target, self.train, self.img_size, self.max_size)
        return image, target
