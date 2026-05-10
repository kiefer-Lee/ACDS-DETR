import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from datasets import build_dataset, collate_fn
from engine.evaluator import evaluate
from losses import build_criterion
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.logger import setup_logger
from utils.misc import apply_overrides, load_config, seed_worker


def main():
    parser = argparse.ArgumentParser("Evaluate ACDS-DETR")
    parser.add_argument("--config", default=None, help="Defaults to checkpoint_dir/config_resolved.yaml when available.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--use-ema", action="store_true", help="Evaluate model_ema weights saved in checkpoint when available.")
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()
    config_path = args.config
    if config_path is None:
        resolved = Path(args.checkpoint).resolve().parent / "config_resolved.yaml"
        config_path = str(resolved if resolved.exists() else ROOT / "configs" / "acds_detr_r50_visdrone.yaml")
        print(f"using config: {config_path}")
    cfg = apply_overrides(load_config(config_path), args.opts)
    logger = setup_logger()
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
    workers = int(cfg["train"]["num_workers"])
    loader_kwargs = {
        "batch_size": cfg["train"]["batch_size"],
        "shuffle": False,
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg["train"].get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_kwargs)
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg).to(device)
    ckpt = load_checkpoint(args.checkpoint, model, map_location=device)
    if args.use_ema and ckpt.get("model_ema") is not None:
        model.load_state_dict(ckpt["model_ema"], strict=False)
        logger.info("Loaded model_ema weights for evaluation.")
    losses, metrics = evaluate(model, criterion, loader, device, cfg, logger)
    logger.info("final losses=" + str(losses))
    logger.info("final metrics=" + str(metrics))


if __name__ == "__main__":
    main()
