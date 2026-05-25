import json
import os
import random
import subprocess
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


def configure_reproducibility(seed, deterministic=False, benchmark=False):
    set_seed(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_value(value):
    import yaml

    return yaml.safe_load(value)


def apply_overrides(cfg, overrides):
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        cursor = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = parse_value(value)
    return cfg


def git_summary(root):
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True, stderr=subprocess.DEVNULL).splitlines()
        return {"commit": commit, "dirty_files": len(status)}
    except Exception:
        return {"commit": "unknown", "dirty_files": None}


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
