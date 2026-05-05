import time

import torch

from utils.metrics import DetectionMetrics, postprocess
from utils.misc import SmoothedValue, move_to_device


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, cfg, logger=None):
    model.eval()
    criterion.eval()
    loss_meters = {}
    metrics = DetectionMetrics(
        num_classes=cfg["model"]["num_classes"],
        iou_thresholds=cfg["eval"]["iou_thresholds"],
        small_area_thr=cfg["acq"]["small_area_thr"],
    )
    infer_time = 0.0
    num_images = 0
    for samples, targets in data_loader:
        samples = move_to_device(samples, device)
        targets_dev = move_to_device(targets, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        outputs = model(samples)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time += time.time() - t0
        num_images += samples["tensors"].shape[0]
        loss_dict = criterion(outputs, targets_dev)
        weight_dict = criterion.weight_dict
        total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict["loss"] = total
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                loss_meters.setdefault(k, SmoothedValue()).update(float(v.detach()))
        preds = postprocess(outputs, targets_dev, cfg["eval"]["score_thresh"], cfg["eval"]["max_detections"])
        metrics.update(preds, targets)
    result = metrics.compute()
    result["FPS"] = num_images / max(1e-6, infer_time)
    losses = {k: m.avg for k, m in loss_meters.items()}
    if logger:
        logger.info(
            "val "
            + " ".join(f"{k}={v:.4f}" for k, v in losses.items() if k.startswith("loss") or k == "query_collision_rate")
            + " "
            + " ".join(f"{k}={v:.4f}" for k, v in result.items())
        )
    return losses, result

