"""Preflight CUDA memory for UAVDT training batches.

The script loads an MMDetection config, finds dense UAVDT samples from the
configured training COCO annotation file, and runs real ``model.train_step`` on
the selected worst-case batches. It is intended to catch data-triggered OOMs
before a long training run reaches those samples.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check ACDS-DETR UAVDT CUDA memory.")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=None,
        help="MMDetection config path. Defaults to CONFIG from train_uavdt.sh, or acds_detr_r50_uavdt.py.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for the train_step probe.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train_uavdt.sh train_dataloader.batch_size.")
    parser.add_argument("--num-queries", type=int, default=None, help="Override model.num_queries.")
    parser.add_argument("--topk", type=int, default=20, help="Number of dense images to probe.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each probe batch for random augments.")
    parser.add_argument("--lr", type=float, default=1e-6, help="Tiny optimizer LR for train_step.")
    parser.add_argument(
        "--keep-pretrained",
        action="store_true",
        help="Keep backbone init_cfg. By default it is disabled to avoid downloads during probing.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only print dataset density and effective config; do not run CUDA train_step.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        default=None,
        help="Extra MMEngine cfg overrides applied after train_uavdt.sh-compatible defaults.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_default_data_root(root: Path) -> str:
    data_root = Path(os.environ.get("DATA_ROOT", root / "../Datasets/UAVDT"))
    if not data_root.is_absolute():
        data_root = (root / data_root).resolve()
    return str(data_root)


def train_script_defaults(config_arg: Path | None) -> tuple[Path, dict[str, Any], dict[str, str]]:
    root = project_root()
    config = config_arg or Path(os.environ.get("CONFIG", "mmdet_acds/configs/acds_detr_r50_uavdt.py"))
    if not config.is_absolute():
        config = root / config

    data_root = resolve_default_data_root(root)
    train_frame_stride = os.environ.get("TRAIN_FRAME_STRIDE", "6")
    val_frame_stride = os.environ.get("VAL_FRAME_STRIDE", "6")
    train_ann = os.environ.get("TRAIN_ANN", f"annotations/train_stride{train_frame_stride}.json")
    val_ann = os.environ.get("VAL_ANN", f"annotations/val_stride{val_frame_stride}.json")
    train_img_prefix = os.environ.get("TRAIN_IMG_PREFIX", "")
    val_img_prefix = os.environ.get("VAL_IMG_PREFIX", "")
    train_split = os.environ.get("TRAIN_SPLIT", "train")
    val_split = os.environ.get("VAL_SPLIT", "val")

    defaults = {
        "data_root": data_root,
        "train_dataloader.dataset.data_root": data_root,
        "train_dataloader.dataset.ann_file": train_ann,
        "train_dataloader.dataset.data_prefix.img": train_img_prefix,
        "train_dataloader.batch_size": 3,
        "train_dataloader.dataset.metainfo.classes": ("car", "truck", "bus"),
        "val_dataloader.dataset.data_root": data_root,
        "val_dataloader.dataset.ann_file": val_ann,
        "val_dataloader.dataset.data_prefix.img": val_img_prefix,
        "val_dataloader.batch_size": 3,
        "val_dataloader.dataset.metainfo.classes": ("car", "truck", "bus"),
        "test_dataloader.dataset.data_root": data_root,
        "test_dataloader.dataset.ann_file": val_ann,
        "test_dataloader.dataset.data_prefix.img": val_img_prefix,
        "test_dataloader.dataset.metainfo.classes": ("car", "truck", "bus"),
        "val_evaluator.ann_file": str(Path(data_root) / val_ann),
        "test_evaluator.ann_file": str(Path(data_root) / val_ann),
    }
    env_defaults = {
        "DATA_ROOT": data_root,
        "TRAIN_ANN": train_ann,
        "VAL_ANN": val_ann,
        "TRAIN_SPLIT": train_split,
        "VAL_SPLIT": val_split,
        "TRAIN_FRAME_STRIDE": train_frame_stride,
        "VAL_FRAME_STRIDE": val_frame_stride,
    }
    return config.resolve(), defaults, env_defaults


def ensure_annotation(ann_path: Path, data_root: str, split: str, frame_stride: str) -> None:
    if ann_path.exists():
        return
    from mmdet_acds.tools.convert_uavdt_to_coco import convert

    print(f"Generating annotation: {ann_path}")
    convert(
        root=data_root,
        output=ann_path,
        split=split,
        frame_stride=int(frame_stride),
    )


def import_runtime() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
        from mmengine.config import Config
        from mmengine.dataset import pseudo_collate
        from mmengine.optim import OptimWrapper
        from mmengine.registry import init_default_scope
        from mmdet.registry import DATASETS, MODELS

        import mmdet_acds  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Failed to import the MMDetection runtime. Activate the same environment "
            "used for training, then run this script again."
        ) from exc
    return torch, Config, pseudo_collate, OptimWrapper, init_default_scope, (DATASETS, MODELS)


def parse_cfg_options(options: list[str] | None) -> dict[str, Any]:
    if not options:
        return {}
    parsed: dict[str, Any] = {}
    for item in options:
        if "=" not in item:
            raise ValueError(f"Invalid --cfg-options item, expected key=value: {item}")
        key, raw_value = item.split("=", 1)
        parsed[key] = parse_value(raw_value)
    return parsed


def parse_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    return raw_value


def strip_init_cfg(obj: Any) -> None:
    if isinstance(obj, dict):
        obj.pop("init_cfg", None)
        for value in obj.values():
            strip_init_cfg(value)
    elif isinstance(obj, list):
        for value in obj:
            strip_init_cfg(value)


def resolve_path(base_dir: Path, root: str | Path, path: str | Path) -> Path:
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = (base_dir / root_path).resolve()
    out = Path(path)
    if not out.is_absolute():
        out = root_path / out
    return out.resolve()


def load_coco_density(ann_file: Path) -> tuple[dict[int, dict[str, Any]], dict[int, int]]:
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    image_by_id = {int(item["id"]): item for item in data.get("images", [])}
    gt_count_by_id: dict[int, int] = defaultdict(int)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        gt_count_by_id[int(ann["image_id"])] += 1
    return image_by_id, gt_count_by_id


def print_effective_config(cfg: Any) -> None:
    top_level = cfg.get("num_queries", None)
    model_queries = cfg.model.get("num_queries", None)
    batch_size = cfg.train_dataloader.get("batch_size", None)
    ann_file = cfg.train_dataloader.dataset.get("ann_file", None)
    data_root = cfg.train_dataloader.dataset.get("data_root", None)
    print("Effective config")
    print(f"  top-level num_queries: {top_level}")
    print(f"  model.num_queries:     {model_queries}")
    print(f"  train batch_size:      {batch_size}")
    print(f"  train data_root:       {data_root}")
    print(f"  train ann_file:        {ann_file}")
    if top_level is not None and model_queries != top_level:
        print("  WARNING: top-level num_queries is not the value used by the model.")


def select_probe_indices(dataset: Any, image_by_id: dict[int, dict[str, Any]], gt_count_by_id: dict[int, int], topk: int) -> list[int]:
    dataset.full_init()
    dataset_len = len(dataset)
    print(f"MMDetection dataset length: {dataset_len}")
    if dataset_len == 0:
        return []

    candidates = []
    for index in range(dataset_len):
        item = dataset.get_data_info(index)
        img_id = int(item.get("img_id", item.get("id", -1)))
        image = image_by_id.get(img_id, {})
        gt_count = len(item.get("instances", [])) or gt_count_by_id.get(img_id, 0)
        width = int(item.get("width", image.get("width", 0)) or 0)
        height = int(item.get("height", image.get("height", 0)) or 0)
        pixels = width * height
        candidates.append((gt_count, pixels, index, img_id, item.get("img_path", image.get("file_name", ""))))
    candidates.sort(reverse=True)
    print("Dense samples selected for probing")
    for rank, (gt_count, pixels, index, img_id, img_path) in enumerate(candidates[:topk], start=1):
        megapixels = pixels / 1_000_000 if pixels else 0
        print(f"  {rank:02d}. idx={index} img_id={img_id} gt={gt_count} pixels={megapixels:.2f}M path={img_path}")
    return [item[2] for item in candidates[:topk]]


def build_optimizer(torch: Any, model: Any, optim_wrapper_cls: Any, lr: float) -> Any:
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    return optim_wrapper_cls(optimizer=optimizer)


def run_probe(args: argparse.Namespace) -> int:
    cuda_devices = os.environ.get("CUDA_DEVICES")
    if cuda_devices and cuda_devices.upper() != "ALL" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    torch, Config, pseudo_collate, OptimWrapper, init_default_scope, registries = import_runtime()
    DATASETS, MODELS = registries

    config_path, train_defaults, env_defaults = train_script_defaults(args.config)
    print(f"Using config: {config_path}")
    cfg = Config.fromfile(config_path)
    cfg.merge_from_dict(train_defaults)
    extra_options = parse_cfg_options(args.cfg_options)
    if extra_options:
        cfg.merge_from_dict(extra_options)
    if args.batch_size is not None:
        cfg.train_dataloader.batch_size = args.batch_size
    if args.num_queries is not None:
        cfg.model.num_queries = args.num_queries
    if not args.keep_pretrained:
        strip_init_cfg(cfg.model)

    print_effective_config(cfg)

    cfg_dir = config_path.parent
    data_root = cfg.train_dataloader.dataset.get("data_root", "")
    ann_file = cfg.train_dataloader.dataset.get("ann_file", "")
    ann_path = resolve_path(cfg_dir, data_root, ann_file)
    ensure_annotation(
        ann_path=ann_path,
        data_root=env_defaults["DATA_ROOT"],
        split=env_defaults["TRAIN_SPLIT"],
        frame_stride=env_defaults["TRAIN_FRAME_STRIDE"],
    )
    if not ann_path.exists():
        raise FileNotFoundError(f"Training annotation not found: {ann_path}")

    image_by_id, gt_count_by_id = load_coco_density(ann_path)
    max_gt = max(gt_count_by_id.values(), default=0)
    avg_gt = sum(gt_count_by_id.values()) / max(1, len(image_by_id))
    print("Training annotation density")
    print(f"  ann_file:      {ann_path}")
    print(f"  images:        {len(image_by_id)}")
    print(f"  max gt/image:  {max_gt}")
    print(f"  avg gt/image:  {avg_gt:.2f}")

    if args.analyze_only:
        return 0

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("CUDA is not available. Use --analyze-only for CPU-only density checks.")

    init_default_scope("mmdet")
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    indices = select_probe_indices(dataset, image_by_id, gt_count_by_id, args.topk)
    if not indices:
        raise RuntimeError(
            "No probe samples selected. The COCO file was readable, but the built "
            "MMDetection dataset length is 0. Check data_root, ann_file, data_prefix, "
            "metainfo/classes, and filter_cfg."
        )

    model = MODELS.build(cfg.model)
    model.to(args.device)
    model.train()
    optim_wrapper = build_optimizer(torch, model, OptimWrapper, args.lr)

    batch_size = int(cfg.train_dataloader.get("batch_size", 1))
    peak_overall = 0
    failed = False
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        for repeat in range(args.repeat):
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                data = pseudo_collate([dataset[index] for index in batch_indices])
                log_vars = model.train_step(data, optim_wrapper)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                peak_overall = max(peak_overall, peak)
                peak_gb = peak / 1024**3
                loss_text = ", ".join(f"{key}={float(value):.4f}" for key, value in log_vars.items() if "loss" in key)
                print(f"OK batch={batch_indices} repeat={repeat + 1}/{args.repeat} peak={peak_gb:.2f} GiB {loss_text}")
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                failed = True
                torch.cuda.empty_cache()
                print(f"OOM batch={batch_indices} repeat={repeat + 1}/{args.repeat}: {exc}")

    total_gb = torch.cuda.get_device_properties(args.device).total_memory / 1024**3
    peak_gb = peak_overall / 1024**3
    print("CUDA memory summary")
    print(f"  device:       {args.device}")
    print(f"  total memory: {total_gb:.2f} GiB")
    print(f"  peak memory:  {peak_gb:.2f} GiB")
    print(f"  margin:       {total_gb - peak_gb:.2f} GiB")
    return 2 if failed else 0


def main() -> None:
    args = parse_args()
    try:
        raise SystemExit(run_probe(args))
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
