import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from datasets import build_dataset, collate_fn
from engine.evaluator import evaluate
from engine.trainer import train_one_epoch
from losses import build_criterion
from models import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.distributed import cleanup_distributed, init_distributed_mode
from utils.logger import setup_logger
from utils.misc import is_main_process, load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser("Train ACDS-DETR")
    parser.add_argument("--config", default=str(ROOT / "configs" / "acds_detr_r50_visdrone.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gpu", type=int, default=None, help="Single GPU id, e.g. 0 or 1. Ignored under torchrun.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    init_distributed_mode(args)
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.resume:
        cfg["train"]["resume"] = args.resume
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    logger = setup_logger()
    set_seed(cfg["seed"] + getattr(args, "rank", 0))
    if torch.cuda.is_available():
        if args.distributed:
            device = torch.device(f"cuda:{args.local_rank}")
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if is_main_process():
        logger.info(f"Using device: {device}")
    train_set = build_dataset("train", cfg)
    val_set = build_dataset("val", cfg)
    train_sampler = DistributedSampler(train_set, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if args.distributed else None
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg).to(device)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": cfg["train"]["lr_backbone"]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg["train"]["epochs"] // 3), gamma=0.1)
    start_epoch = 0
    if cfg["train"].get("resume"):
        ckpt = load_checkpoint(cfg["train"]["resume"], model, optimizer, scheduler, map_location=device)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank] if device.type == "cuda" else None)
    scaler = torch.amp.GradScaler("cuda") if cfg["train"]["amp"] and device.type == "cuda" else None
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_stats, elapsed = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, cfg, scaler, logger if is_main_process() else None)
        scheduler.step()
        if is_main_process():
            logger.info(f"train epoch={epoch} time={elapsed:.1f}s " + " ".join(f"{k}={v:.4f}" for k, v in train_stats.items() if k.startswith("loss")))
            save_checkpoint(out_dir / "last.pth", model, optimizer, scheduler, epoch, cfg)
        if (epoch + 1) % cfg["train"]["val_freq"] == 0:
            val_losses, val_metrics = evaluate(model, criterion, val_loader, device, cfg, logger if is_main_process() else None)
            if is_main_process():
                save_checkpoint(out_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scheduler, epoch, cfg)
    cleanup_distributed()


if __name__ == "__main__":
    main()
