"""Create a stratified mini split for SODA-D COCO annotations."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Path to the SODA-D root.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Annotation split names under the annotation directory.",
    )
    parser.add_argument("--annotation-dir", default="Annotations")
    parser.add_argument("--image-dir", default="Images")
    parser.add_argument("--out-image-dir", default="mimi_images")
    parser.add_argument("--out-annotation-dir", default="mini-annotations")
    parser.add_argument("--fraction", type=float, default=1 / 3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "hardlink", "symlink", "none"],
        default="copy",
        help="How to place selected images into the mini image directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mini annotation files and existing copied images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print sampling statistics; do not write annotations or images.",
    )
    return parser.parse_args()


def image_category_counts(annotations: list[dict]) -> dict[int, Counter]:
    counts: dict[int, Counter] = defaultdict(Counter)
    for ann in annotations:
        if ann.get("ignore", 0):
            continue
        image_id = ann["image_id"]
        category_id = ann["category_id"]
        counts[image_id][category_id] += 1
    return counts


def stratified_select(
    images: list[dict],
    image_counts: dict[int, Counter],
    fraction: float,
    rng: random.Random,
) -> set[int]:
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    image_ids = [img["id"] for img in images]
    target_images = max(1, round(len(image_ids) * fraction))

    category_to_images: dict[int, list[int]] = defaultdict(list)
    for image_id in image_ids:
        for category_id in image_counts.get(image_id, Counter()):
            category_to_images[category_id].append(image_id)

    total_presence = Counter({cat: len(ids) for cat, ids in category_to_images.items()})
    desired_presence = Counter(
        {category_id: max(1, round(count * fraction)) for category_id, count in total_presence.items()}
    )

    selected: set[int] = set()
    selected_presence: Counter = Counter()

    for category_id, candidates in sorted(category_to_images.items(), key=lambda item: len(item[1])):
        needed = desired_presence[category_id] - selected_presence.get(category_id, 0)
        if needed <= 0 or len(selected) >= target_images:
            continue

        shuffled = list(candidates)
        rng.shuffle(shuffled)
        shuffled.sort(
            key=lambda image_id: (
                sum(
                    max(0, desired_presence[cat] - selected_presence.get(cat, 0))
                    for cat in image_counts.get(image_id, Counter())
                ),
                sum(image_counts.get(image_id, Counter()).values()),
            ),
            reverse=True,
        )

        for image_id in shuffled:
            if image_id in selected:
                continue
            selected.add(image_id)
            for cat in image_counts.get(image_id, Counter()):
                selected_presence[cat] += 1
            needed -= 1
            if needed <= 0 or len(selected) >= target_images:
                break

    if len(selected) < target_images:
        remaining = [image_id for image_id in image_ids if image_id not in selected]
        rng.shuffle(remaining)
        selected.update(remaining[: target_images - len(selected)])

    if len(selected) > target_images:
        removable = list(selected)
        rng.shuffle(removable)
        removable.sort(key=lambda image_id: sum(image_counts.get(image_id, Counter()).values()))
        for image_id in removable:
            if len(selected) <= target_images:
                break
            selected.remove(image_id)

    return selected


def category_instance_counts(image_ids: list[int], image_counts: dict[int, Counter]) -> Counter:
    total_counts = Counter()
    for image_id in image_ids:
        total_counts.update(image_counts.get(image_id, Counter()))
    return total_counts


def relative_category_error(full_counts: Counter, mini_counts: Counter) -> dict[int, float]:
    errors = {}
    full_total = max(1, sum(full_counts.values()))
    mini_total = max(1, sum(mini_counts.values()))
    for category_id in sorted(full_counts):
        full_ratio = full_counts[category_id] / full_total
        mini_ratio = mini_counts.get(category_id, 0) / mini_total
        errors[category_id] = mini_ratio - full_ratio
    return errors


def copy_image(src_root: Path, dst_root: Path, file_name: str, mode: str, overwrite: bool) -> None:
    if mode == "none":
        return

    src = src_root / file_name
    dst = dst_root / file_name
    if not src.is_file():
        raise FileNotFoundError(f"Missing image: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if overwrite:
            dst.unlink()
        else:
            return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unknown copy mode: {mode}")


def process_split(args: argparse.Namespace, split: str, rng: random.Random) -> dict:
    data_root = Path(args.data_root)
    ann_path = data_root / args.annotation_dir / f"{split}.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {ann_path}")

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    image_counts = image_category_counts(annotations)
    selected_ids = stratified_select(images, image_counts, args.fraction, rng)

    selected_images = [img for img in images if img["id"] in selected_ids]
    selected_annotations = [ann for ann in annotations if ann["image_id"] in selected_ids]

    full_counts = category_instance_counts([img["id"] for img in images], image_counts)

    mini_counts = Counter()
    for ann in selected_annotations:
        if not ann.get("ignore", 0):
            mini_counts[ann["category_id"]] += 1

    out_data = dict(data)
    out_data["images"] = selected_images
    out_data["annotations"] = selected_annotations

    out_ann_dir = data_root / args.out_annotation_dir
    out_ann_path = out_ann_dir / f"{split}.json"
    out_image_root = data_root / args.out_image_dir
    src_image_root = data_root / args.image_dir

    if not args.dry_run:
        out_ann_dir.mkdir(parents=True, exist_ok=True)
        if out_ann_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output annotation exists: {out_ann_path}")
        out_ann_path.write_text(json.dumps(out_data), encoding="utf-8")
        for img in selected_images:
            copy_image(src_image_root, out_image_root, img["file_name"], args.copy_mode, args.overwrite)

    errors = relative_category_error(full_counts, mini_counts)
    return {
        "split": split,
        "images": len(images),
        "mini_images": len(selected_images),
        "annotations": len(annotations),
        "mini_annotations": len(selected_annotations),
        "category_count_error": errors,
        "output_annotation": str(out_ann_path),
        "output_image_dir": str(out_image_root),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    summaries = [process_split(args, split, rng) for split in args.splits]

    for item in summaries:
        print(
            f"{item['split']}: "
            f"{item['mini_images']}/{item['images']} images, "
            f"{item['mini_annotations']}/{item['annotations']} annotations"
        )
        print(f"  annotation: {item['output_annotation']}")
        print(f"  images: {item['output_image_dir']}")
        if item["category_count_error"]:
            max_abs_error = max(abs(value) for value in item["category_count_error"].values())
            print(f"  max category-ratio error: {max_abs_error:.6f}")


if __name__ == "__main__":
    main()
