from collections import Counter
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


def _resolve_split_base(root: Path, split_name: str) -> Path:
    """Resolve common VisDrone layouts without silently falling back to a bad path."""
    candidates = [
        root / split_name,
        root / split_name / split_name,
        root / split_name.replace("VisDrone2019-DET-", ""),
    ]
    for base in candidates:
        if (base / "images").exists() and (base / "annotations").exists():
            return base
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"VisDrone split directory not found. Tried:\n{tried}")


class VisDroneDetection(Dataset):
    def __init__(
        self,
        root,
        split="train",
        img_size=800,
        max_size=1333,
        train=True,
        min_area=1,
        max_samples=None,
        augment=None,
        strict_bbox=False,
    ):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.max_size = max_size
        self.train = train
        self.min_area = min_area
        self.augment = augment or {}
        self.strict_bbox = strict_bbox
        split_name = "VisDrone2019-DET-train" if split == "train" else "VisDrone2019-DET-val"
        base = _resolve_split_base(self.root, split_name)
        self.image_dir = base / "images"
        self.ann_dir = base / "annotations"
        self.images = sorted(self.image_dir.glob("*.jpg"))
        if max_samples is not None:
            self.images = self.images[: int(max_samples)]
        if len(self.images) == 0:
            raise FileNotFoundError(f"No images found in {self.image_dir}")
        self._diagnostics = None

    def __len__(self):
        return len(self.images)

    def _read_annotation(self, ann_path, image_size):
        boxes, labels, areas, iscrowd = [], [], [], []
        if not ann_path.exists():
            return boxes, labels, areas, iscrowd
        img_w, img_h = image_size
        with ann_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                # VisDrone DET annotations are xywh in absolute pixels:
                # <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,...
                x, y, w, h = map(float, parts[:4])
                score = int(float(parts[4]))
                cls = int(float(parts[5]))
                trunc = int(float(parts[6]))
                occ = int(float(parts[7]))
                if score == 0 or cls < 1 or cls > 10:
                    continue
                if w <= 0 or h <= 0:
                    if self.strict_bbox:
                        raise ValueError(f"Invalid non-positive bbox in {ann_path}: {line.strip()}")
                    continue
                x0, y0, x1, y1 = x, y, x + w, y + h
                cx0, cy0 = max(0.0, x0), max(0.0, y0)
                cx1, cy1 = min(float(img_w), x1), min(float(img_h), y1)
                if cx1 <= cx0 or cy1 <= cy0:
                    if self.strict_bbox:
                        raise ValueError(f"Invalid out-of-image bbox in {ann_path}: {line.strip()}")
                    continue
                area = (cx1 - cx0) * (cy1 - cy0)
                if area < self.min_area:
                    continue
                boxes.append([cx0, cy0, cx1, cy1])
                # Training uses contiguous 0-based labels; COCO export maps back to 1-based.
                labels.append(cls - 1)
                areas.append(area)
                iscrowd.append(0)
        return boxes, labels, areas, iscrowd

    def __getitem__(self, idx):
        image_path = self.images[idx]
        ann_path = self.ann_dir / f"{image_path.stem}.txt"
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        boxes, labels, areas, iscrowd = self._read_annotation(ann_path, (w, h))
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
        image, target = apply_transforms(image, target, self.train, self.img_size, self.max_size, self.augment)
        return image, target

    def diagnostics(self):
        if self._diagnostics is not None:
            return self._diagnostics
        num_empty = 0
        num_boxes = 0
        max_boxes = 0
        small = medium = large = 0
        category_hist = Counter()
        invalid_files = 0
        for image_path in self.images:
            image = Image.open(image_path)
            w, h = image.size
            boxes, labels, areas, _ = self._read_annotation(self.ann_dir / f"{image_path.stem}.txt", (w, h))
            n = len(boxes)
            if n == 0:
                num_empty += 1
            num_boxes += n
            max_boxes = max(max_boxes, n)
            for label, area in zip(labels, areas):
                category_hist[int(label)] += 1
                if area < 32 * 32:
                    small += 1
                elif area < 96 * 96:
                    medium += 1
                else:
                    large += 1
            if not (self.ann_dir / f"{image_path.stem}.txt").exists():
                invalid_files += 1
        self._diagnostics = {
            "split": self.split,
            "images": len(self.images),
            "empty_images": num_empty,
            "missing_annotation_files": invalid_files,
            "boxes": num_boxes,
            "max_boxes_per_image": max_boxes,
            "avg_boxes_per_image": num_boxes / max(1, len(self.images)),
            "small_boxes": small,
            "medium_boxes": medium,
            "large_boxes": large,
            "small_ratio": small / max(1, num_boxes),
            "category_hist_0_based": dict(sorted(category_hist.items())),
            "image_dir": str(self.image_dir),
            "ann_dir": str(self.ann_dir),
        }
        return self._diagnostics
