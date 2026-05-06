import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset, collate_fn
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.misc import load_config, move_to_device


def denorm_image(tensor):
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Visualize R-SNDS sampling points")
    parser.add_argument("--config", default=str(ROOT / "configs" / "exp_acds_full_stable.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--query", type=int, default=0)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "visualizations" / "sampling.png"))
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("model", {})["return_intermediates"] = True
    device = torch.device("cpu")
    if torch.cuda.is_available():
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            device = torch.device(f"cuda:{args.gpu}")
        else:
            device = torch.device("cuda")
    dataset = build_dataset("val", cfg)
    image, target = dataset[args.index]
    samples, targets = next(iter(DataLoader([(image, target)], batch_size=1, collate_fn=collate_fn)))
    samples = move_to_device(samples, device)
    targets = move_to_device(targets, device)
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    outputs = model(samples)
    locs = outputs["sampling_locations"][args.layer][0, args.query].detach().cpu()
    h, w = targets[0]["size"].tolist()
    pts = locs.reshape(-1, 2) * locs.new_tensor([w, h])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(denorm_image(samples["tensors"][0].cpu()))
    ax.scatter(pts[:, 0], pts[:, 1], s=14, c="red", label="sampling points")
    ref = outputs["reference_points"][args.layer][0, args.query, :2].detach().cpu() * torch.tensor([w, h])
    ax.scatter([ref[0]], [ref[1]], s=50, c="cyan", marker="x", label="reference")
    for box in targets[0]["boxes"].detach().cpu():
        x0, y0, x1, y1 = box.tolist()
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=0.8))
    ax.legend(loc="upper right")
    ax.axis("off")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"saved {args.output}; points={pts.shape[0]}")


if __name__ == "__main__":
    main()
