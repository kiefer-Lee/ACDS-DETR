import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def load_config(path):
    import yaml

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "_base_" in cfg:
        base_path = path.parent / cfg.pop("_base_")
        base = load_config(base_path)
        cfg = merge_dict(base, cfg)
    return cfg


def merge_dict(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


class SmoothedValue:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self):
        return self.total / max(1, self.count)

