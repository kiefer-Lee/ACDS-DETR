"""Convert VisDrone DET txt annotations to COCO JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from mmdet_acds.datasets.visdrone_metainfo import VISDRONE_CLASSES


def _resolve_split_base(root: Path, split: str) -> Path:
    split_name = "VisDrone2019-DET-train" if split == "train" else "VisDrone2019-DET-val"
    candidates = [
        root / split_name,
        root / split_name / split_name,
        root / split,
    ]
    for base in candidates:
        if (base / "images").exists() and (base / "annotations").exists():
            return base
    tried = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"VisDrone split directory not found. Tried:\n{tried}")


def convert_split(root: str | Path, split: str, output: str | Path, min_area: float = 1.0) -> None:
    root = Path(root)
    base = _resolve_split_base(root, split)
    image_dir = base / "images"
    ann_dir = base / "annotations"
    images = []
    annotations = []
    ann_id = 1
    for img_id, img_path in enumerate(sorted(image_dir.glob("*.jpg"))):
        with Image.open(img_path) as img:
            w, h = img.size
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        ann_path = ann_dir / f"{img_path.stem}.txt"
        if not ann_path.exists():
            continue
        with ann_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue
                x, y, bw, bh = map(float, parts[:4])
                score = int(float(parts[4]))
                cls = int(float(parts[5]))
                trunc = int(float(parts[6]))
                occ = int(float(parts[7]))
                if score == 0 or cls < 1 or cls > len(VISDRONE_CLASSES) or bw <= 0 or bh <= 0:
                    continue
                x0 = max(0.0, x)
                y0 = max(0.0, y)
                x1 = min(float(w), x + bw)
                y1 = min(float(h), y + bh)
                area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                if area < min_area:
                    continue
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls,
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "area": area,
                        "iscrowd": 0,
                        "truncation": trunc,
                        "occlusion": occ,
                    }
                )
                ann_id += 1

    data = {
        "info": {"description": f"VisDrone {split} converted to COCO"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(VISDRONE_CLASSES)],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-area", type=float, default=1.0)
    args = parser.parse_args()
    convert_split(args.root, args.split, args.output, args.min_area)


if __name__ == "__main__":
    main()

