import math
import time

import torch

from utils.distributed import reduce_dict
from utils.misc import SmoothedValue, move_to_device


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, cfg, scaler=None, logger=None):
    model.train()
    criterion.train()
    meters = {}
    start = time.time()
    print_freq = cfg["train"]["print_freq"]
    for i, (samples, targets) in enumerate(data_loader):
        samples = move_to_device(samples, device)
        targets = move_to_device(targets, device)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(losses).backward()
            if cfg["train"]["clip_max_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_max_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if cfg["train"]["clip_max_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_max_norm"])
            optimizer.step()
        reduced = reduce_dict({k: v.detach() for k, v in loss_dict.items() if torch.is_tensor(v)})
        reduced["loss"] = losses.detach()
        for k, v in reduced.items():
            meters.setdefault(k, SmoothedValue()).update(float(v))
        if logger and i % print_freq == 0:
            msg = f"epoch={epoch} iter={i}/{len(data_loader)} " + " ".join(
                f"{k}={m.avg:.4f}" for k, m in meters.items() if k.startswith("loss") or k == "query_collision_rate"
            )
            logger.info(msg)
        loss_value = float(losses.detach())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss: {loss_value}")
    elapsed = time.time() - start
    return {k: m.avg for k, m in meters.items()}, elapsed
