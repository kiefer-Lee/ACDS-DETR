import time

import torch

from utils.metrics import DetectionMetrics, postprocess
from utils.coco_eval import coco_evaluate
from utils.distributed import all_gather_object
from utils.misc import is_main_process
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
        dense_small_count_thr=cfg["eval"].get("dense_small_count_thr", 20),
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
        preds = postprocess(
            outputs,
            targets_dev,
            cfg["eval"]["score_thresh"],
            cfg["eval"]["max_detections"],
            cfg["eval"].get("min_detections", 0),
        )
        metrics.update(preds, targets)
    gathered_preds = all_gather_object(metrics.preds)
    gathered_targets = all_gather_object(metrics.targets)
    if is_main_process():
        metrics.preds = [item for part in gathered_preds for item in part]
        metrics.targets = [item for part in gathered_targets for item in part]
        result = None
        if cfg["eval"].get("use_coco_eval", True):
            result = coco_evaluate(metrics.preds, metrics.targets, cfg["model"]["num_classes"], cfg["eval"].get("max_detections", 100))
        if result is None:
            result = metrics.compute()
        else:
            supplemental = metrics.compute()
            for key in ("precision", "recall", "Dense_images", "Dense_AP_small", "Dense_AR_small"):
                result.setdefault(key, supplemental.get(key, 0.0))
    else:
        result = {}
    losses = {k: m.avg for k, m in loss_meters.items()}
    gathered_losses = all_gather_object(losses)
    gathered_speed = all_gather_object({"num_images": num_images, "infer_time": infer_time})
    if is_main_process():
        loss_keys = sorted({k for part in gathered_losses for k in part.keys()})
        losses = {k: sum(float(part.get(k, 0.0)) for part in gathered_losses) / max(1, len(gathered_losses)) for k in loss_keys}
        total_images = sum(part["num_images"] for part in gathered_speed)
        total_time = sum(part["infer_time"] for part in gathered_speed)
        result["FPS"] = total_images / max(1e-6, total_time)
    else:
        losses = {}
        result["FPS"] = 0.0
    if logger:
        logger.info(
            "val "
            + " ".join(f"{k}={v:.4f}" for k, v in losses.items() if k.startswith("loss") or k == "query_collision_rate")
            + " "
            + " ".join(f"{k}={v:.4f}" for k, v in result.items())
        )
    return losses, result
