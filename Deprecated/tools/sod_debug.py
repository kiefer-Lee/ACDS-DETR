import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


def split_base(root, split):
    name = "VisDrone2019-DET-train" if split == "train" else "VisDrone2019-DET-val"
    base = Path(root) / name / name
    return base / "images", base / "annotations"


def read_visdrone_txt(path, image_size=None, min_area=1.0):
    boxes = []
    img_w, img_h = image_size if image_size else (None, None)
    if not path.exists():
        return boxes
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        x, y, w, h = map(float, parts[:4])
        score = int(float(parts[4]))
        cls = int(float(parts[5]))
        if score == 0 or cls < 1 or cls > 10 or w <= 0 or h <= 0:
            continue
        x0, y0, x1, y1 = x, y, x + w, y + h
        if img_w is not None:
            x0, x1 = max(0.0, x0), min(float(img_w), x1)
            y0, y1 = max(0.0, y0), min(float(img_h), y1)
        bw, bh = max(0.0, x1 - x0), max(0.0, y1 - y0)
        area = bw * bh
        if area < min_area:
            continue
        boxes.append({"box": [x0, y0, x1, y1], "label": cls - 1, "area": area, "line": line_no})
    return boxes


def iou(box, boxes):
    x0 = max(box[0], boxes[0])
    y0 = max(box[1], boxes[1])
    x1 = min(box[2], boxes[2])
    y1 = min(box[3], boxes[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = max(0.0, boxes[2] - boxes[0]) * max(0.0, boxes[3] - boxes[1])
    return inter / max(1e-6, area_a + area_b - inter)


def command_stats(args):
    image_dir, ann_dir = split_base(args.root, args.split)
    buckets = Counter()
    per_image_small = []
    class_counter = Counter()
    invalid_images = 0
    for img_path in sorted(image_dir.glob("*.jpg")):
        with Image.open(img_path) as img:
            boxes = read_visdrone_txt(ann_dir / f"{img_path.stem}.txt", img.size, args.min_area)
        small = 0
        for item in boxes:
            area = item["area"]
            class_counter[item["label"]] += 1
            if area < args.small_area_thr:
                buckets["small"] += 1
                small += 1
            elif area < 96 * 96:
                buckets["medium"] += 1
            else:
                buckets["large"] += 1
        per_image_small.append(small)
        if not boxes:
            invalid_images += 1
    per_image_small.sort()
    total = sum(buckets.values())
    summary = {
        "images": len(per_image_small),
        "empty_or_no_valid_box_images": invalid_images,
        "boxes": total,
        "small": buckets["small"],
        "medium": buckets["medium"],
        "large": buckets["large"],
        "small_ratio": buckets["small"] / max(1, total),
        "small_per_image_p50": percentile(per_image_small, 50),
        "small_per_image_p90": percentile(per_image_small, 90),
        "small_per_image_max": max(per_image_small) if per_image_small else 0,
        "class_counts_0_based": dict(sorted(class_counter.items())),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def percentile(values, pct):
    if not values:
        return 0
    idx = int(round((len(values) - 1) * pct / 100.0))
    return values[max(0, min(len(values) - 1, idx))]


def draw_boxes(img, items, color, label_prefix="", min_score=0.0):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = None
    for item in items:
        if item.get("score", 1.0) < min_score:
            continue
        x0, y0, x1, y1 = item["box"]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        text = f"{label_prefix}{item.get('label', '')}"
        if "score" in item:
            text += f" {item['score']:.2f}"
        draw.text((x0, max(0, y0 - 12)), text, fill=color, font=font)


def command_vis_ann(args):
    image_dir, ann_dir = split_base(args.root, args.split)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in sorted(image_dir.glob("*.jpg")):
        with Image.open(img_path).convert("RGB") as img:
            boxes = read_visdrone_txt(ann_dir / f"{img_path.stem}.txt", img.size, args.min_area)
            if args.small_only:
                boxes = [b for b in boxes if b["area"] < args.small_area_thr]
            if not boxes:
                continue
            draw_boxes(img, boxes, "red", "gt:")
            img.save(out_dir / img_path.name)
            count += 1
            if count >= args.limit:
                break
    print(f"saved {count} images to {out_dir}")


def load_predictions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    by_image = defaultdict(list)
    for item in data:
        box = item.get("bbox", item.get("box"))
        if box is None:
            continue
        if len(box) == 4 and item.get("bbox_format", "xywh") == "xywh":
            x, y, w, h = box
            box = [x, y, x + w, y + h]
        by_image[int(item["image_id"])].append({
            "box": [float(v) for v in box],
            "label": int(item.get("category_id", item.get("label", 1))) - 1,
            "score": float(item.get("score", 1.0)),
        })
    return by_image


def command_vis_pred(args):
    image_dir, ann_dir = split_base(args.root, args.split)
    preds = load_predictions(args.predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, img_path in enumerate(sorted(image_dir.glob("*.jpg"))[: args.limit]):
        with Image.open(img_path).convert("RGB") as img:
            gt = read_visdrone_txt(ann_dir / f"{img_path.stem}.txt", img.size, args.min_area)
            draw_boxes(img, gt, "red", "gt:")
            draw_boxes(img, preds.get(idx, []), "lime", "pd:", args.score_thresh)
            img.save(out_dir / img_path.name)
    print(f"saved overlays to {out_dir}")


def command_fn(args):
    image_dir, ann_dir = split_base(args.root, args.split)
    preds = load_predictions(args.predictions)
    total_small = 0
    missed = []
    for idx, img_path in enumerate(sorted(image_dir.glob("*.jpg"))):
        with Image.open(img_path) as img:
            gt = read_visdrone_txt(ann_dir / f"{img_path.stem}.txt", img.size, args.min_area)
        small_gt = [g for g in gt if g["area"] < args.small_area_thr]
        total_small += len(small_gt)
        image_preds = [p for p in preds.get(idx, []) if p["score"] >= args.score_thresh]
        for g in small_gt:
            matched = any(p["label"] == g["label"] and iou(g["box"], p["box"]) >= args.iou_thr for p in image_preds)
            if not matched:
                missed.append({"image_id": idx, "file_name": img_path.name, **g})
    print(json.dumps({
        "small_gt": total_small,
        "missed_small": len(missed),
        "small_fn_rate": len(missed) / max(1, total_small),
        "examples": missed[: args.limit],
    }, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser("Small-object dataset and prediction diagnostics")
    parser.add_argument("command", choices=["stats", "vis-ann", "vis-pred", "fn"])
    parser.add_argument("--root", default="/data/libaichuan/Projects/SOD/Datasets/VisDrone")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--small-area-thr", type=float, default=32 * 32)
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument("--output-dir", default="outputs/debug_vis")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--small-only", action="store_true")
    parser.add_argument("--predictions", default=None, help="COCO-style detection JSON with image_id/category_id/bbox/score")
    parser.add_argument("--score-thresh", type=float, default=0.03)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    args = parser.parse_args()
    if args.command == "stats":
        command_stats(args)
    elif args.command == "vis-ann":
        command_vis_ann(args)
    elif args.command == "vis-pred":
        command_vis_pred(args)
    elif args.command == "fn":
        command_fn(args)


if __name__ == "__main__":
    main()
