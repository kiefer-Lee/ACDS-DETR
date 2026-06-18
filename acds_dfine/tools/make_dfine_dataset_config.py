"""Generate a D-FINE custom dataset YAML from COCO-format paths."""

from __future__ import annotations

import argparse
from pathlib import Path


TEMPLATE = """task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ["bbox"]

num_classes: {num_classes}
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {train_images}
    ann_file: {train_ann}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: {num_workers}
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {val_images}
    ann_file: {val_ann}
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: {num_workers}
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-ann", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-ann", required=True)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = TEMPLATE.format(
        train_images=args.train_images,
        train_ann=args.train_ann,
        val_images=args.val_images,
        val_ann=args.val_ann,
        num_classes=args.num_classes,
        num_workers=args.num_workers,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

