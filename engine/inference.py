import torch

from utils.metrics import postprocess
from utils.misc import move_to_device


@torch.no_grad()
def predict(model, samples, targets, device, cfg):
    model.eval()
    samples = move_to_device(samples, device)
    targets = move_to_device(targets, device)
    outputs = model(samples)
    return postprocess(outputs, targets, cfg["eval"]["score_thresh"], cfg["eval"]["max_detections"])

