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
from utils.misc import load_config


def main():
    parser = argparse.ArgumentParser("Evaluate ACDS-DETR")
    parser.add_argument("--config", default=str(ROOT / "configs" / "acds_detr_r50_visdrone.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger = setup_logger()
    if torch.cuda.is_available():
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dataset = build_dataset("val", cfg)
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], collate_fn=collate_fn)
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    losses, metrics = evaluate(model, criterion, loader, device, cfg, logger)
    logger.info("final losses=" + str(losses))
    logger.info("final metrics=" + str(metrics))


if __name__ == "__main__":
    main()

