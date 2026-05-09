import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from datasets import build_dataset
from utils.misc import apply_overrides, load_config


def _box_health(dataset, limit=None):
    total = 0
    bad_order = 0
    out_of_bounds = 0
    tiny = 0
    max_objects = 0
    for idx in range(len(dataset) if limit is None else min(len(dataset), limit)):
        image, target = dataset[idx]
        boxes = target["boxes"]
        h, w = target["size"].tolist()
        total += int(boxes.shape[0])
        max_objects = max(max_objects, int(boxes.shape[0]))
        if boxes.numel() == 0:
            continue
        bad_order += int(((boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])).sum())
        out_of_bounds += int(((boxes[:, 0] < 0) | (boxes[:, 1] < 0) | (boxes[:, 2] > w) | (boxes[:, 3] > h)).sum())
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        tiny += int((area < 16).sum())
    return {
        "checked_images": len(dataset) if limit is None else min(len(dataset), limit),
        "boxes_after_pipeline": total,
        "max_objects_after_pipeline": max_objects,
        "bad_xyxy_order_after_pipeline": bad_order,
        "out_of_bounds_after_pipeline": out_of_bounds,
        "area_lt_16_after_pipeline": tiny,
    }


def main():
    parser = argparse.ArgumentParser("Diagnose ACDS-DETR_pro data and small-object settings")
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_full_small_object.yaml"))
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--json", default=None)
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.opts)
    train_set = build_dataset("train", cfg)
    val_set = build_dataset("val", cfg)
    report = {
        "config": str(args.config),
        "dataset_root": cfg["dataset"]["root"],
        "num_queries": cfg["model"]["num_queries"],
        "num_feature_levels": cfg["model"]["num_feature_levels"],
        "use_p2": cfg["model"].get("use_p2", False),
        "encoder_feature_indices": cfg["model"].get("encoder_feature_indices"),
        "score_thresh": cfg["eval"].get("score_thresh"),
        "max_detections": cfg["eval"].get("max_detections"),
        "min_detections": cfg["eval"].get("min_detections", 0),
        "train_raw": train_set.diagnostics(),
        "val_raw": val_set.diagnostics(),
        "train_pipeline_health": _box_health(train_set, args.limit),
        "val_pipeline_health": _box_health(val_set, args.limit),
    }
    max_objects = max(report["train_raw"]["max_boxes_per_image"], report["val_raw"]["max_boxes_per_image"])
    report["warnings"] = []
    if cfg["model"]["num_queries"] < max_objects:
        report["warnings"].append(f"num_queries={cfg['model']['num_queries']} is lower than max objects per image={max_objects}; dense recall can be capped.")
    if not cfg["model"].get("use_p2", False):
        report["warnings"].append("use_p2=false; stride-4 features are disabled, which often hurts APs/ARs on VisDrone.")
    if cfg["eval"].get("max_detections", 100) < max_objects:
        report["warnings"].append("eval.max_detections is lower than max objects per image; ARs may be underestimated.")
    if cfg["dataset"].get("augment", {}).get("zoom_crop_prob", 0) > 0 and cfg["dataset"].get("augment", {}).get("zoom_crop_min_visibility", 0) < 0.5:
        report["warnings"].append("zoom crop visibility is below 0.5; very small boxes may be heavily truncated.")

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
