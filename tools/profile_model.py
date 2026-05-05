import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from models import build_model
from utils.misc import load_config


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Profile ACDS-DETR")
    parser.add_argument("--config", default=str(ROOT / "configs" / "exp_acds_full_stable.yaml"))
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1333)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device("cuda")
    model = build_model(cfg).to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    samples = {
        "tensors": torch.randn(1, 3, args.height, args.width, device=device),
        "mask": torch.zeros(1, args.height, args.width, dtype=torch.bool, device=device),
    }
    for _ in range(args.warmup):
        model(samples)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    for _ in range(args.iters):
        model(samples)
    if device.type == "cuda":
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated(device)
    else:
        mem = 0
    elapsed = time.time() - t0
    fps = args.iters / max(elapsed, 1e-6)
    print(f"Params: {params / 1e6:.3f}M")
    print(f"Trainable params: {trainable / 1e6:.3f}M")
    print("FLOPs/MACs: unavailable without optional profiling dependency")
    print(f"FPS: {fps:.3f}")
    print(f"Max GPU memory: {mem / (1024 ** 2):.1f} MB")


if __name__ == "__main__":
    main()
