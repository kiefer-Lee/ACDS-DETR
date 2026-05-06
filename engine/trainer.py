import math
import time
from pathlib import Path

import torch

from utils.distributed import reduce_dict
from utils.misc import SmoothedValue, move_to_device


def _first_nonfinite_grad(model):
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return name
    return None


def _outputs_are_finite(outputs):
    for key in ("pred_logits", "pred_boxes"):
        value = outputs.get(key)
        if value is not None and not torch.isfinite(value).all():
            return False, key
    return True, None


def _save_nan_debug(debug_dir, epoch, iteration, samples, targets, loss_dict, reason):
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    path = Path(debug_dir) / "nan_batch.pth"
    cpu_samples = move_to_device(samples, torch.device("cpu"))
    cpu_targets = move_to_device(targets, torch.device("cpu"))
    cpu_losses = {k: float(v.detach().cpu()) for k, v in loss_dict.items() if torch.is_tensor(v) and v.numel() == 1}
    torch.save({"epoch": epoch, "iteration": iteration, "samples": cpu_samples, "targets": cpu_targets, "losses": cpu_losses, "reason": reason}, path)


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, cfg, scaler=None, logger=None):
    model.train()
    criterion.train()
    meters = {}
    start = time.time()
    end = start
    print_freq = cfg["train"]["print_freq"]
    for i, (samples, targets) in enumerate(data_loader):
        data_time = time.time() - end
        samples = move_to_device(samples, device)
        targets = move_to_device(targets, device)
        with torch.amp.autocast(device_type=device.type, enabled=scaler is not None):
            outputs = model(samples)
            finite_outputs, bad_output = _outputs_are_finite(outputs)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_value = float(losses.detach()) if torch.is_tensor(losses) else float(losses)
        if (not finite_outputs) or (not math.isfinite(loss_value)):
            reason = f"nonfinite_output:{bad_output}" if not finite_outputs else "nonfinite_loss"
            _save_nan_debug(cfg["train"].get("debug_dir", "outputs/debug"), epoch, i, samples, targets, loss_dict, reason)
            if logger:
                logger.info(f"skip non-finite batch epoch={epoch} iter={i} reason={reason} loss={loss_value}")
            if not cfg["train"].get("skip_nonfinite", True):
                raise RuntimeError(f"Non-finite training state: {reason}")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(losses).backward()
            if cfg["train"]["clip_max_norm"] > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_max_norm"])
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            bad_grad = _first_nonfinite_grad(model)
            if bad_grad is not None or not torch.isfinite(grad_norm):
                _save_nan_debug(cfg["train"].get("debug_dir", "outputs/debug"), epoch, i, samples, targets, loss_dict, f"nonfinite_grad:{bad_grad}")
                if logger:
                    logger.info(f"skip non-finite grad epoch={epoch} iter={i} param={bad_grad} grad_norm={float(grad_norm)}")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if cfg["train"]["clip_max_norm"] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip_max_norm"])
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            bad_grad = _first_nonfinite_grad(model)
            if bad_grad is not None or not torch.isfinite(grad_norm):
                _save_nan_debug(cfg["train"].get("debug_dir", "outputs/debug"), epoch, i, samples, targets, loss_dict, f"nonfinite_grad:{bad_grad}")
                if logger:
                    logger.info(f"skip non-finite grad epoch={epoch} iter={i} param={bad_grad} grad_norm={float(grad_norm)}")
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
        reduced = reduce_dict({k: v.detach() for k, v in loss_dict.items() if torch.is_tensor(v)})
        reduced["loss"] = losses.detach()
        reduced["grad_norm"] = grad_norm.detach() if torch.is_tensor(grad_norm) else torch.as_tensor(grad_norm, device=device)
        reduced["data_time"] = torch.as_tensor(data_time, device=device)
        reduced["iter_time"] = torch.as_tensor(time.time() - end, device=device)
        for k, v in reduced.items():
            meters.setdefault(k, SmoothedValue()).update(float(v))
        if logger and i % print_freq == 0:
            msg = f"epoch={epoch} iter={i}/{len(data_loader)} " + " ".join(
                f"{k}={m.avg:.4f}" for k, m in meters.items() if k.startswith("loss") or k in ("query_collision_rate", "grad_norm", "data_time", "iter_time")
            )
            logger.info(msg)
        end = time.time()
    elapsed = time.time() - start
    return {k: m.avg for k, m in meters.items()}, elapsed
