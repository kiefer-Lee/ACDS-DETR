from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, cfg, meta=None, model_ema=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "cfg": cfg,
        "meta": meta or {},
    }
    if model_ema is not None:
        state["model_ema"] = model_ema.state_dict()
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
