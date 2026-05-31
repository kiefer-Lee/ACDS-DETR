"""Convert COCO detection annotations to Ultralytics YOLO labels."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


VISDRONE_CLASSES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def convert_split(coco_json: Path, image_root: Path, output_root: Path, split: str, copy_images: bool) -> None:
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data.get("images", [])}
    categories = sorted(data.get("categories", []), key=lambda item: int(item["id"]))
    cat_to_index = {int(cat["id"]): idx for idx, cat in enumerate(categories)}

    label_dir = output_root / "labels" / split
    image_dir = output_root / "images" / split
    label_dir.mkdir(parents=True, exist_ok=True)
    if copy_images:
        image_dir.mkdir(parents=True, exist_ok=True)

    rows_by_image: dict[int, list[str]] = {image_id: [] for image_id in images}
    for ann in data.get("annotations", []):
        image = images.get(int(ann["image_id"]))
        if image is None or int(ann.get("iscrowd", 0)) == 1:
            continue
        x, y, w, h = [float(v) for v in ann["bbox"]]
        img_w = max(float(image["width"]), 1.0)
        img_h = max(float(image["height"]), 1.0)
        cx = (x + w * 0.5) / img_w
        cy = (y + h * 0.5) / img_h
        norm_w = w / img_w
        norm_h = h / img_h
        cls_id = cat_to_index[int(ann["category_id"])]
        rows_by_image[int(ann["image_id"])].append(f"{cls_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")

    for image_id, image in images.items():
        src = image_root / image["file_name"]
        stem = Path(image["file_name"]).stem
        (label_dir / f"{stem}.txt").write_text("\n".join(rows_by_image[image_id]), encoding="utf-8")
        if copy_images:
            dst = image_dir / Path(image["file_name"]).name
            if not dst.exists():
                shutil.copy2(src, dst)


def write_data_yaml(output_root: Path, names: list[str]) -> None:
    names_rows = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(names))
    text = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names_rows}\n"
    )
    (output_root / "visdrone.yaml").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="VisDrone root containing annotations and images.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for Ultralytics dataset.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images into output-root/images/{train,val}.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_img_root = args.data_root / "VisDrone2019-DET-train" / "VisDrone2019-DET-train" / "images"
    val_img_root = args.data_root / "VisDrone2019-DET-val" / "VisDrone2019-DET-val" / "images"
    convert_split(args.data_root / "annotations" / "train.json", train_img_root, args.output_root, "train", args.copy_images)
    convert_split(args.data_root / "annotations" / "val.json", val_img_root, args.output_root, "val", args.copy_images)
    write_data_yaml(args.output_root, VISDRONE_CLASSES)


if __name__ == "__main__":
    main()
