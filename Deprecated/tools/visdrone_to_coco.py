import argparse
import json
from pathlib import Path

from PIL import Image


def convert_split(root, split, output):
    split_name = "VisDrone2019-DET-train" if split == "train" else "VisDrone2019-DET-val"
    base = Path(root) / split_name / split_name
    image_dir = base / "images"
    ann_dir = base / "annotations"
    images, annotations = [], []
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
                if score == 0 or cls < 1 or cls > 10 or bw <= 0 or bh <= 0:
                    continue
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": [x, y, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "truncation": trunc,
                    "occlusion": occ,
                })
                ann_id += 1
    data = {
        "info": {"description": f"VisDrone {split} converted to COCO"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": str(i)} for i in range(1, 11)],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"saved {output}: images={len(images)} annotations={len(annotations)}")


def main():
    parser = argparse.ArgumentParser("Convert VisDrone txt annotations to COCO json")
    parser.add_argument("--root", default="D:/PythonProjects/SOD/Datasets/VisDrone")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_split(args.root, args.split, args.output)


if __name__ == "__main__":
    main()
