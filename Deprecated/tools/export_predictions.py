import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from datasets import build_dataset, collate_fn
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.metrics import postprocess
from utils.misc import apply_overrides, load_config, move_to_device, seed_worker


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Export COCO-style detection JSON for debugging")
    parser.add_argument("--config", default=str(ROOT / "configs" / "paper_full_small_object.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.opts)
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
    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    detections = []
    for samples, targets in loader:
        samples = move_to_device(samples, device)
        targets_dev = move_to_device(targets, device)
        outputs = model(samples)
        preds = postprocess(outputs, targets_dev, cfg["eval"]["score_thresh"], cfg["eval"]["max_detections"])
        for pred in preds:
            for box, label, score in zip(pred["boxes"].tolist(), pred["labels"].tolist(), pred["scores"].tolist()):
                x0, y0, x1, y1 = box
                detections.append({
                    "image_id": int(pred["image_id"]),
                    "category_id": int(label) + 1,
                    "bbox": [float(x0), float(y0), float(max(0.0, x1 - x0)), float(max(0.0, y1 - y0))],
                    "score": float(score),
                })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(detections), encoding="utf-8")
    print(f"saved {len(detections)} detections to {out}")


if __name__ == "__main__":
    main()
