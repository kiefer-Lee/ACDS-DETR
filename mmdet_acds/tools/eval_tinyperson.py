"""Evaluate COCO-format detections with TinyPerson-style size ranges.

The script is intentionally model-agnostic: any detector that can export
COCO detection json can be evaluated with the same annotation file.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


AREA_RANGES = OrderedDict(
    [
        ("all", (0.0, 1.0e10)),
        ("tiny", (2.0**2, 20.0**2)),
        ("tiny1", (2.0**2, 8.0**2)),
        ("tiny2", (8.0**2, 12.0**2)),
        ("tiny3", (12.0**2, 20.0**2)),
        ("small", (20.0**2, 32.0**2)),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TinyPerson-style AP/AR evaluation for COCO-format results."
    )
    parser.add_argument("--ann", required=True, type=Path, help="COCO ground-truth json.")
    parser.add_argument("--det", required=True, type=Path, help="COCO detection result json.")
    parser.add_argument(
        "--max-dets",
        nargs="+",
        type=int,
        default=[100, 300, 500],
        help="COCOeval maxDets. AP uses the largest value.",
    )
    parser.add_argument(
        "--category-mode",
        choices=("keep", "merge"),
        default="keep",
        help="Keep original categories or merge all categories into one person class.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write metric values as json.",
    )
    return parser.parse_args()


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def _merge_categories(ann_file: Path, det_file: Path, tmp_dir: Path) -> tuple[Path, Path]:
    ann = _load_json(ann_file)
    det = _load_json(det_file)

    ann.setdefault("info", {})
    ann.setdefault("licenses", [])
    ann["categories"] = [{"id": 1, "name": "person", "supercategory": "person"}]
    for item in ann.get("annotations", []):
        item["category_id"] = 1
    for item in det:
        item["category_id"] = 1

    merged_ann = tmp_dir / "tinyperson_merged_gt.json"
    merged_det = tmp_dir / "tinyperson_merged_det.json"
    _write_json(merged_ann, ann)
    _write_json(merged_det, det)
    return merged_ann, merged_det


def _idx(values: Iterable[float], target: float) -> int:
    arr = np.asarray(list(values), dtype=float)
    matches = np.where(np.isclose(arr, target))[0]
    if len(matches) == 0:
        raise ValueError(f"Value {target} not found in COCOeval parameter list: {arr}")
    return int(matches[0])


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid))


def _ap(coco_eval: COCOeval, area_label: str, iou: float | None = None) -> float:
    precision = coco_eval.eval["precision"]
    area_idx = coco_eval.params.areaRngLbl.index(area_label)
    max_det_idx = len(coco_eval.params.maxDets) - 1
    if iou is None:
        iou_mask = np.asarray(coco_eval.params.iouThrs) >= 0.50
        values = precision[iou_mask, :, :, area_idx, max_det_idx]
    else:
        values = precision[_idx(coco_eval.params.iouThrs, iou), :, :, area_idx, max_det_idx]
    return _mean_valid(values)


def _ar(coco_eval: COCOeval, area_label: str, iou: float | None = None) -> float:
    recall = coco_eval.eval["recall"]
    area_idx = coco_eval.params.areaRngLbl.index(area_label)
    max_det_idx = len(coco_eval.params.maxDets) - 1
    if iou is None:
        iou_mask = np.asarray(coco_eval.params.iouThrs) >= 0.50
        values = recall[iou_mask, :, area_idx, max_det_idx]
    else:
        values = recall[_idx(coco_eval.params.iouThrs, iou), :, area_idx, max_det_idx]
    return _mean_valid(values)


def _fmt(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value * 100:.2f}"


def evaluate(ann_file: Path, det_file: Path, max_dets: list[int]) -> dict[str, float]:
    coco_gt = COCO(str(ann_file))
    coco_gt.dataset.setdefault("info", {})
    coco_dt = coco_gt.loadRes(str(det_file))

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.iouThrs = np.asarray([0.25, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    coco_eval.params.maxDets = sorted(max_dets)
    coco_eval.params.areaRng = [list(v) for v in AREA_RANGES.values()]
    coco_eval.params.areaRngLbl = list(AREA_RANGES.keys())
    coco_eval.evaluate()
    coco_eval.accumulate()

    metrics: dict[str, float] = {}
    for area in AREA_RANGES:
        metrics[f"AP{area}"] = _ap(coco_eval, area)
        metrics[f"AP{area}_25"] = _ap(coco_eval, area, 0.25)
        metrics[f"AP{area}_50"] = _ap(coco_eval, area, 0.50)
        metrics[f"AP{area}_75"] = _ap(coco_eval, area, 0.75)
        metrics[f"AR{area}"] = _ar(coco_eval, area)
        metrics[f"AR{area}_50"] = _ar(coco_eval, area, 0.50)

    return metrics


def print_table(metrics: dict[str, float], max_dets: list[int], category_mode: str) -> None:
    print("\nTinyPerson-style bbox metrics")
    print(f"category_mode: {category_mode}")
    print(f"maxDets: {sorted(max_dets)} (AP/AR use {max(max_dets)})")
    print("area ranges use bbox area: tiny=[2,20], tiny1=[2,8], tiny2=[8,12], tiny3=[12,20], small=[20,32]")
    print()
    print("| area | AP | AP25 | AP50 | AP75 | AR | AR50 | Miss@50 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for area in AREA_RANGES:
        ar50 = metrics[f"AR{area}_50"]
        miss50 = float("nan") if np.isnan(ar50) else 1.0 - ar50
        print(
            f"| {area} | {_fmt(metrics[f'AP{area}'])} | {_fmt(metrics[f'AP{area}_25'])} | "
            f"{_fmt(metrics[f'AP{area}_50'])} | {_fmt(metrics[f'AP{area}_75'])} | "
            f"{_fmt(metrics[f'AR{area}'])} | {_fmt(ar50)} | {_fmt(miss50)} |"
        )


def main() -> None:
    args = parse_args()
    ann_file = args.ann
    det_file = args.det

    with tempfile.TemporaryDirectory(prefix="tinyperson_eval_") as tmp:
        if args.category_mode == "merge":
            ann_file, det_file = _merge_categories(args.ann, args.det, Path(tmp))

        metrics = evaluate(ann_file, det_file, args.max_dets)

    print_table(metrics, args.max_dets, args.category_mode)

    if args.summary_json is not None:
        payload = {
            "ann": str(args.ann),
            "det": str(args.det),
            "category_mode": args.category_mode,
            "max_dets": sorted(args.max_dets),
            "metrics": metrics,
        }
        _write_json(args.summary_json, payload)
        print(f"\nWrote summary: {args.summary_json}")


if __name__ == "__main__":
    main()
