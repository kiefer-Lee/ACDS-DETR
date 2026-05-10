import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
import torchvision.transforms.functional as F

from models import build_model
from utils.checkpoint import load_checkpoint
from utils.metrics import postprocess
from utils.misc import load_config


def main():
    parser = argparse.ArgumentParser("Infer with ACDS-DETR")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
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
    model = build_model(cfg).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()
    image = Image.open(args.image).convert("RGB")
    w, h = image.size
    tensor = F.normalize(F.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
    samples = {"tensors": tensor[None], "mask": torch.zeros((1, h, w), dtype=torch.bool, device=device)}
    targets = [{"orig_size": torch.tensor([h, w], device=device), "image_id": torch.tensor([0], device=device)}]
    with torch.no_grad():
        outputs = model(samples)
        results = postprocess(outputs, targets, cfg["eval"]["score_thresh"], cfg["eval"]["max_detections"])[0]
    print(results)


if __name__ == "__main__":
    main()
