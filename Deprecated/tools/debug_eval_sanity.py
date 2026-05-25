import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from datasets import build_dataset, collate_fn
from models import build_model
from utils.box_ops import box_iou
from utils.checkpoint import load_checkpoint
from utils.metrics import postprocess
from utils.misc import apply_overrides, load_config, move_to_device, seed_worker


def summarize_predictions(preds, targets, topk=100):
    total_pred = 0
    total_gt = 0
    score_min, score_max = 1.0, 0.0
    max_iou_any = []
    max_iou_cls = []
    pred_area = []
    gt_area = []
    recall_any = {0.3: [0, 0], 0.5: [0, 0], 0.75: [0, 0]}
    recall_cls = {0.3: [0, 0], 0.5: [0, 0], 0.75: [0, 0]}
    for pred, target in zip(preds, targets):
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]
        gt_boxes = target["boxes"].cpu()
        gt_labels = target["labels"].cpu()
        total_pred += int(boxes.shape[0])
        total_gt += int(gt_boxes.shape[0])
        if scores.numel() > 0:
            score_min = min(score_min, float(scores.min()))
            score_max = max(score_max, float(scores.max()))
            area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            pred_area.extend(area[:topk].tolist())
        if gt_boxes.numel() > 0:
            garea = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
            gt_area.extend(garea.tolist())
        if boxes.numel() == 0 or gt_boxes.numel() == 0:
            max_iou_any.append(0.0)
            max_iou_cls.append(0.0)
            for thr in recall_any:
                recall_any[thr][1] += int(gt_boxes.shape[0])
                recall_cls[thr][1] += int(gt_boxes.shape[0])
            continue
        order = scores.sort(descending=True).indices[:topk]
        pboxes = boxes[order]
        plabels = labels[order]
        ious = box_iou(pboxes, gt_boxes)[0]
        max_iou_any.append(float(ious.max()))
        best_any_per_gt = ious.max(0).values
        cls_best = 0.0
        best_cls_per_gt = torch.zeros(gt_boxes.shape[0])
        for cls in gt_labels.unique():
            pk = plabels == cls
            gk = gt_labels == cls
            if pk.any() and gk.any():
                cls_ious = box_iou(pboxes[pk], gt_boxes[gk])[0]
                cls_best = max(cls_best, float(cls_ious.max()))
                gt_idx = gk.nonzero().flatten()
                best_cls_per_gt[gt_idx] = cls_ious.max(0).values.cpu()
        max_iou_cls.append(cls_best)
        for thr in recall_any:
            recall_any[thr][0] += int((best_any_per_gt.cpu() >= thr).sum())
            recall_any[thr][1] += int(gt_boxes.shape[0])
            recall_cls[thr][0] += int((best_cls_per_gt >= thr).sum())
            recall_cls[thr][1] += int(gt_boxes.shape[0])
    def avg(values):
        return sum(values) / max(1, len(values))
    return {
        "images": len(targets),
        "total_gt": total_gt,
        "total_pred_after_postprocess": total_pred,
        "score_min": 0.0 if total_pred == 0 else score_min,
        "score_max": score_max,
        "avg_max_iou_any_class_topk": avg(max_iou_any),
        "max_iou_any_class_topk": max(max_iou_any) if max_iou_any else 0.0,
        "avg_max_iou_same_class_topk": avg(max_iou_cls),
        "max_iou_same_class_topk": max(max_iou_cls) if max_iou_cls else 0.0,
        "avg_pred_area_topk": avg(pred_area),
        "avg_gt_area": avg(gt_area),
        "topk_recall_any_iou_0.30": recall_any[0.3][0] / max(1, recall_any[0.3][1]),
        "topk_recall_any_iou_0.50": recall_any[0.5][0] / max(1, recall_any[0.5][1]),
        "topk_recall_any_iou_0.75": recall_any[0.75][0] / max(1, recall_any[0.75][1]),
        "topk_recall_same_class_iou_0.30": recall_cls[0.3][0] / max(1, recall_cls[0.3][1]),
        "topk_recall_same_class_iou_0.50": recall_cls[0.5][0] / max(1, recall_cls[0.5][1]),
        "topk_recall_same_class_iou_0.75": recall_cls[0.75][0] / max(1, recall_cls[0.75][1]),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Sanity-check predictions before COCO evaluation")
    parser.add_argument("--config", default=None, help="Defaults to checkpoint_dir/config_resolved.yaml when available.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        resolved = Path(args.checkpoint).resolve().parent / "config_resolved.yaml"
        config_path = str(resolved if resolved.exists() else ROOT / "configs" / "default.yaml")
        print(f"using config: {config_path}")
    cfg = apply_overrides(load_config(config_path), args.opts)
    requested = args.device or cfg.get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and torch.cuda.is_available():
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device(requested)
    else:
        device = torch.device("cpu")

    dataset = build_dataset("val", cfg)
    workers = int(cfg["train"].get("num_workers", 0))
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    model = build_model(cfg).to(device).eval()
    ckpt = load_checkpoint(args.checkpoint, model, map_location=device)
    if args.use_ema and ckpt.get("model_ema") is not None:
        model.load_state_dict(ckpt["model_ema"], strict=False)
        print("loaded model_ema")

    all_preds, all_targets = [], []
    raw_score_max = []
    raw_box_minmax = []
    for batch_idx, (samples, targets) in enumerate(loader):
        if batch_idx >= args.batches:
            break
        samples = move_to_device(samples, device)
        targets_dev = move_to_device(targets, device)
        outputs = model(samples)
        probs = outputs["pred_logits"].softmax(-1)[..., :-1]
        raw_score_max.append(float(probs.max().detach().cpu()))
        raw_box_minmax.append((float(outputs["pred_boxes"].min().detach().cpu()), float(outputs["pred_boxes"].max().detach().cpu())))
        preds = postprocess(
            outputs,
            targets_dev,
            cfg["eval"]["score_thresh"],
            cfg["eval"]["max_detections"],
            cfg["eval"].get("min_detections", 0),
        )
        all_preds.extend(preds)
        all_targets.extend(targets)
    summary = summarize_predictions(all_preds, all_targets, args.topk)
    summary["raw_score_max_avg"] = sum(raw_score_max) / max(1, len(raw_score_max))
    summary["raw_box_min"] = min(x[0] for x in raw_box_minmax) if raw_box_minmax else 0.0
    summary["raw_box_max"] = max(x[1] for x in raw_box_minmax) if raw_box_minmax else 0.0
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
