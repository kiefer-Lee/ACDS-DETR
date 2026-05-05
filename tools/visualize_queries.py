import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets import build_dataset, collate_fn
from losses import build_criterion
from models import build_model
from utils.checkpoint import load_checkpoint
from utils.misc import load_config, move_to_device


def denorm_image(tensor):
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    img = (tensor * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Visualize ACDS-DETR query reference points")
    parser.add_argument("--config", default=str(ROOT / "configs" / "exp_acds_full_stable.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default=str(ROOT / "outputs" / "visualizations" / "queries.png"))
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
    dataset = build_dataset(args.split, cfg)
    image, target = dataset[args.index]
    loader = DataLoader([(image, target)], batch_size=1, collate_fn=collate_fn)
    samples, targets = next(iter(loader))
    samples = move_to_device(samples, device)
    targets = move_to_device(targets, device)
    model = build_model(cfg).to(device)
    criterion = build_criterion(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    outputs = model(samples)
    indices = criterion.matcher(outputs, criterion._prepare_targets(targets, device))
    refs = outputs["reference_points"][-1][0, :, :2].detach().cpu()
    h, w = targets[0]["size"].tolist()
    points = refs * refs.new_tensor([w, h])

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(denorm_image(samples["tensors"][0].cpu()))
    ax.scatter(points[:, 0], points[:, 1], s=8, c="yellow", label="queries")
    if indices and indices[0][0].numel() > 0:
        src_idx, tgt_idx = indices[0]
        areas = targets[0]["area"][tgt_idx].detach().cpu()
        small = areas < cfg["acq"]["small_area_thr"]
        matched = points[src_idx.detach().cpu()]
        ax.scatter(matched[:, 0], matched[:, 1], s=20, facecolors="none", edgecolors="lime", label="matched")
        if small.any():
            small_pts = points[src_idx.detach().cpu()[small]]
            ax.scatter(small_pts[:, 0], small_pts[:, 1], s=36, facecolors="none", edgecolors="red", label="matched small")
    for box in targets[0]["boxes"].detach().cpu():
        x0, y0, x1, y1 = box.tolist()
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white", linewidth=0.8))
    ax.legend(loc="upper right")
    ax.axis("off")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

