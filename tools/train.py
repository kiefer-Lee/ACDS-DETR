import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from datasets import build_dataset, collate_fn
from engine.evaluator import evaluate
from engine.trainer import train_one_epoch
from losses import build_criterion
from models import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.distributed import cleanup_distributed, init_distributed_mode
from utils.logger import setup_logger, write_jsonl
from utils.misc import apply_overrides, configure_reproducibility, git_summary, is_main_process, load_config, seed_worker


def parse_args():
    parser = argparse.ArgumentParser("Train ACDS-DETR")
    parser.add_argument("--config", default=str(ROOT / "configs" / "acds_detr_r50_visdrone.yaml"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gpu", type=int, default=None, help="Single GPU id, e.g. 0 or 1. Ignored under torchrun.")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--opts", nargs="*", default=[], help="Override config values, e.g. train.lr=1e-4 acq.enabled=false")
    return parser.parse_args()


def select_device(args, cfg):
    requested = args.device or cfg.get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and torch.cuda.is_available():
        if args.distributed:
            device = torch.device(f"cuda:{args.local_rank}")
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device(requested)
            if device.index is not None:
                torch.cuda.set_device(device.index)
        return device
    return torch.device("cpu")


def make_loader(dataset, cfg, sampler, shuffle, seed):
    workers = int(cfg["train"]["num_workers"])
    kwargs = {
        "batch_size": cfg["train"]["batch_size"],
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(cfg["train"].get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))
    return DataLoader(dataset, **kwargs)


def checkpoint_meta(cfg, args, best_map, best_ap_small):
    return {
        "seed": cfg["seed"],
        "best_map": best_map,
        "best_ap_small": best_ap_small,
        "git": git_summary(ROOT),
        "args": vars(args),
    }


def main():
    args = parse_args()
    init_distributed_mode(args)
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.opts)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.resume:
        cfg["train"]["resume"] = args.resume
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["device"] = args.device
    logger = setup_logger()
    configure_reproducibility(
        cfg["seed"] + getattr(args, "rank", 0),
        deterministic=cfg["train"].get("deterministic", False),
        benchmark=cfg["train"].get("benchmark", False),
    )
    device = select_device(args, cfg)
    if is_main_process():
        logger.info(f"Using device: {device}")
    train_set = build_dataset("train", cfg)
    val_set = build_dataset("val", cfg)
    train_sampler = DistributedSampler(train_set, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if args.distributed else None
    train_loader = make_loader(train_set, cfg, train_sampler, train_sampler is None, cfg["seed"] + getattr(args, "rank", 0))
    val_loader = make_loader(val_set, cfg, val_sampler, False, cfg["seed"] + 1000 + getattr(args, "rank", 0))
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
        best_map = float(ckpt.get("meta", {}).get("best_map", -1.0))
        best_ap_small = float(ckpt.get("meta", {}).get("best_ap_small", -1.0))
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank] if device.type == "cuda" else None)
    scaler = torch.amp.GradScaler("cuda") if cfg["train"]["amp"] and device.type == "cuda" else None
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        with (out_dir / "config_resolved.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    best_map = locals().get("best_map", -1.0)
    best_ap_small = locals().get("best_ap_small", -1.0)
    metrics_summary = {"best_map": best_map, "best_ap_small": best_ap_small, "best_epoch_map": None, "best_epoch_ap_small": None}
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_stats, elapsed = train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, cfg, scaler, logger if is_main_process() else None)
        scheduler.step()
        if is_main_process():
            train_stats["lr"] = optimizer.param_groups[0]["lr"]
            if device.type == "cuda":
                train_stats["max_mem_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            logger.info(f"train epoch={epoch} time={elapsed:.1f}s " + " ".join(f"{k}={v:.4f}" for k, v in train_stats.items() if k.startswith("loss")))
            write_jsonl(out_dir / "train_log.jsonl", {"epoch": epoch, "time": elapsed, **train_stats})
            save_checkpoint(out_dir / "last.pth", model, optimizer, scheduler, epoch, cfg, checkpoint_meta(cfg, args, best_map, best_ap_small))
        if (epoch + 1) % cfg["train"]["val_freq"] == 0:
            val_losses, val_metrics = evaluate(model, criterion, val_loader, device, cfg, logger if is_main_process() else None)
            if is_main_process():
                write_jsonl(out_dir / "val_metrics.jsonl", {"epoch": epoch, **val_losses, **val_metrics})
                save_checkpoint(out_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scheduler, epoch, cfg)
                current_map = float(val_metrics.get("mAP", val_metrics.get("mAP50_95", -1.0)))
                current_ap_small = float(val_metrics.get("AP_small", -1.0))
                if current_map > best_map:
                    best_map = current_map
                    metrics_summary["best_map"] = best_map
                    metrics_summary["best_epoch_map"] = epoch
                    save_checkpoint(out_dir / "best_map.pth", model, optimizer, scheduler, epoch, cfg, checkpoint_meta(cfg, args, best_map, best_ap_small))
                if current_ap_small > best_ap_small:
                    best_ap_small = current_ap_small
                    metrics_summary["best_ap_small"] = best_ap_small
                    metrics_summary["best_epoch_ap_small"] = epoch
                    save_checkpoint(out_dir / "best_ap_small.pth", model, optimizer, scheduler, epoch, cfg, checkpoint_meta(cfg, args, best_map, best_ap_small))
                save_checkpoint(out_dir / "last.pth", model, optimizer, scheduler, epoch, cfg, checkpoint_meta(cfg, args, best_map, best_ap_small))
                with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
                    json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
    cleanup_distributed()


if __name__ == "__main__":
    main()
