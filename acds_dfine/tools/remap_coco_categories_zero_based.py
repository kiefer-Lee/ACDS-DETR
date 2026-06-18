"""Remap COCO category ids to contiguous zero-based ids for D-FINE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ann-in", required=True, help="Input COCO annotation JSON.")
    parser.add_argument("--ann-out", required=True, help="Output remapped COCO annotation JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ann_in = Path(args.ann_in)
    ann_out = Path(args.ann_out)

    data = json.loads(ann_in.read_text(encoding="utf-8"))
    categories = sorted(data.get("categories", []), key=lambda item: item["id"])
    id_map = {cat["id"]: idx for idx, cat in enumerate(categories)}

    if not id_map:
        raise ValueError(f"No categories found in {ann_in}")

    for cat in categories:
        cat["id"] = id_map[cat["id"]]

    for ann in data.get("annotations", []):
        old_id = ann.get("category_id")
        if old_id not in id_map:
            raise ValueError(f"Unknown category_id={old_id} in {ann_in}")
        ann["category_id"] = id_map[old_id]

    data["categories"] = categories
    ann_out.parent.mkdir(parents=True, exist_ok=True)
    ann_out.write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    main()
