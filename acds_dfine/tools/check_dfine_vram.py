"""VRAM preflight for ACDS-D-FINE training configs.

The script builds the same D-FINE model, criterion, optimizer, and one training
batch used by the bash launcher, then runs a forward/backward/optimizer step to
measure peak CUDA memory.  It intentionally runs a single local process and
simulates the per-GPU batch size derived from total batch size / world size.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _gb(num_bytes: int | float) -> float:
    return float(num_bytes) / 1024**3


def _add_repo_paths(dfine_root: Path, acds_root: Path) -> None:
    for path in (dfine_root, acds_root):
        text = str(path.resolve())
        if text not in sys.path:
            sys.path.insert(0, text)


def _oom_message(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def _configure_probe_scale(loader: Any, mode: str, explicit_size: int | None) -> str:
    collate_fn = getattr(loader, "collate_fn", None)
    if collate_fn is None or not hasattr(collate_fn, "scales"):
        return "unchanged"

    scales = getattr(collate_fn, "scales", None)
    if explicit_size is not None:
        collate_fn.scales = [int(explicit_size)]
        return str(explicit_size)

    if mode == "max" and scales:
        max_size = max(int(s) for s in scales)
        collate_fn.scales = [max_size]
        return str(max_size)

    if mode == "base" and hasattr(collate_fn, "base_size"):
        base_size = int(collate_fn.base_size)
        collate_fn.scales = [base_size]
        return str(base_size)

    return mode


def _move_targets_to_device(targets: list[dict[str, Any]], device: Any) -> list[dict[str, Any]]:
    import torch

    return [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
        for target in targets
    ]


def _annotation_is_valid(annotation: dict[str, Any]) -> bool:
    if int(annotation.get("iscrowd", 0)) != 0:
        return False
    bbox = annotation.get("bbox", None)
    if not bbox or len(bbox) < 4:
        return False
    return float(bbox[2]) > 0 and float(bbox[3]) > 0


def _rank_dense_images(ann_file: str | Path) -> list[dict[str, Any]]:
    with open(ann_file, "r", encoding="utf-8") as handle:
        coco = json.load(handle)

    counts: Counter[int] = Counter()
    for annotation in coco.get("annotations", []):
        if _annotation_is_valid(annotation):
            counts[int(annotation["image_id"])] += 1

    images = {}
    for image in coco.get("images", []):
        image_id = int(image["id"])
        width = float(image.get("width", 0) or 0)
        height = float(image.get("height", 0) or 0)
        images[image_id] = {
            "image_id": image_id,
            "file_name": image.get("file_name", ""),
            "width": width,
            "height": height,
            "area": width * height,
            "box_count": counts.get(image_id, 0),
        }

    ranked = sorted(
        images.values(),
        key=lambda item: (-int(item["box_count"]), -float(item["area"]), int(item["image_id"])),
    )
    return [item for item in ranked if int(item["box_count"]) > 0]


def _dataset_indices_for_dense_batch(
    dataset: Any,
    ann_file: str | Path,
    batch_size: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    if not hasattr(dataset, "ids"):
        raise ValueError("The configured train dataset does not expose COCO image ids.")

    index_by_image_id = {int(image_id): idx for idx, image_id in enumerate(dataset.ids)}
    ranked_images = _rank_dense_images(ann_file)
    selected_indices: list[int] = []
    selected_images: list[dict[str, Any]] = []

    for image_info in ranked_images:
        image_id = int(image_info["image_id"])
        if image_id not in index_by_image_id:
            continue
        selected_indices.append(index_by_image_id[image_id])
        selected_images.append(image_info)
        if len(selected_indices) == batch_size:
            break

    if not selected_indices:
        raise ValueError("No valid annotated image was found for worst-boxes VRAM probing.")

    base_indices = list(selected_indices)
    base_images = list(selected_images)
    while len(selected_indices) < batch_size:
        pos = len(selected_indices) % len(base_indices)
        selected_indices.append(base_indices[pos])
        selected_images.append(base_images[pos])

    return selected_indices, selected_images


def _set_strict_probe_epoch(loader: Any, disable_random_aug: bool) -> None:
    dataset = getattr(loader, "dataset", None)
    collate_fn = getattr(loader, "collate_fn", None)
    if disable_random_aug and hasattr(dataset, "set_epoch"):
        dataset.set_epoch(10**9)
    elif hasattr(dataset, "set_epoch"):
        dataset.set_epoch(0)

    if hasattr(collate_fn, "set_epoch"):
        collate_fn.set_epoch(0)


def _iter_probe_batches(
    loader: Any,
    mode: str,
    ann_file: str | Path,
    batch_size: int,
    steps: int,
) -> tuple[list[Any], list[dict[str, Any]]]:
    if mode == "dataloader":
        batches = []
        for _, batch in zip(range(steps), loader):
            batches.append(batch)
        return batches, []

    selected_indices, selected_images = _dataset_indices_for_dense_batch(
        loader.dataset,
        ann_file,
        batch_size,
    )
    items = [loader.dataset[index] for index in selected_indices]
    batch = loader.collate_fn(items)
    return [batch for _ in range(steps)], selected_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether ACDS-D-FINE training may OOM.")
    parser.add_argument("--dfine-root", required=True, type=Path)
    parser.add_argument("--acds-root", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-img-folder", required=True)
    parser.add_argument("--train-ann-file", required=True)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--total-batch-size", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--use-amp", type=_bool_arg, default=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--probe-mode", choices=["worst_boxes", "dataloader"], default="worst_boxes")
    parser.add_argument("--probe-scale", choices=["max", "base", "random"], default="max")
    parser.add_argument("--probe-size", type=int, default=None)
    parser.add_argument("--strict-disable-random-aug", type=_bool_arg, default=True)
    parser.add_argument("--safety-fraction", type=float, default=0.90)
    parser.add_argument("--reserve-gb", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-epoch", type=int, default=None)
    parser.add_argument("--extra-update", nargs="*", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.world_size <= 0:
        raise ValueError("--world-size must be positive")
    if args.total_batch_size <= 0:
        raise ValueError("--total-batch-size must be positive")
    if args.total_batch_size % args.world_size != 0:
        raise ValueError("--total-batch-size must be divisible by --world-size")

    per_gpu_batch = args.total_batch_size // args.world_size
    _add_repo_paths(args.dfine_root, args.acds_root)

    import torch
    import torch.amp

    from src.core import YAMLConfig, yaml_utils
    from src.misc import dist_utils

    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot run a VRAM preflight.")
        return 3
    if args.device_index >= torch.cuda.device_count():
        print(
            f"Requested cuda:{args.device_index}, but only {torch.cuda.device_count()} visible GPU(s)."
        )
        return 3

    dist_utils.setup_seed(args.seed)
    torch.cuda.set_device(args.device_index)
    device = torch.device(f"cuda:{args.device_index}")

    update_dict = yaml_utils.parse_cli(args.extra_update)
    train_overrides: dict[str, Any] = {
        "total_batch_size": per_gpu_batch,
        "num_workers": 0,
        "dataset": {
            "img_folder": args.train_img_folder,
            "ann_file": args.train_ann_file,
        },
    }
    if args.stop_epoch is not None:
        train_overrides["dataset"] = {
            **train_overrides["dataset"],
            "transforms": {"policy": {"epoch": args.stop_epoch}},
        }
        train_overrides["collate_fn"] = {"stop_epoch": args.stop_epoch}

    update_dict = yaml_utils.merge_dict(
        update_dict,
        {
            "num_classes": args.num_classes,
            "device": str(device),
            "use_amp": bool(args.use_amp),
            "train_dataloader": train_overrides,
            "val_dataloader": {"total_batch_size": 1, "num_workers": 0},
            "output_dir": "./output/vram_preflight",
        },
    )

    print("ACDS-D-FINE VRAM preflight")
    print(f"  config: {args.config}")
    print(f"  total batch size: {args.total_batch_size}")
    print(f"  world size / GPUS: {args.world_size}")
    print(f"  simulated per-GPU batch size: {per_gpu_batch}")
    print(f"  probe mode: {args.probe_mode}")
    print(f"  AMP: {bool(args.use_amp)}")
    print(f"  device: {torch.cuda.get_device_name(device)}")

    free_start, total_mem = torch.cuda.mem_get_info(device)
    print(f"  GPU free before build: {_gb(free_start):.2f} GiB / {_gb(total_mem):.2f} GiB")

    try:
        cfg = YAMLConfig(args.config, **update_dict)
        model = cfg.model.to(device)
        criterion = cfg.criterion.to(device)
        optimizer = cfg.optimizer
        scaler = cfg.scaler if args.use_amp else None
        loader = cfg.train_dataloader
        _set_strict_probe_epoch(loader, args.strict_disable_random_aug)
        scale_desc = _configure_probe_scale(loader, args.probe_scale, args.probe_size)
        probe_batches, selected_images = _iter_probe_batches(
            loader,
            args.probe_mode,
            args.train_ann_file,
            per_gpu_batch,
            args.steps,
        )

        model.train()
        criterion.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        last_loss = None
        for step, (samples, targets) in enumerate(probe_batches):
            optimizer.zero_grad(set_to_none=True)
            samples = samples.to(device, non_blocking=True)
            targets = _move_targets_to_device(targets, device)
            metas = dict(epoch=0, step=step, global_step=step, epoch_step=len(loader))

            if scaler is not None:
                with torch.autocast(device_type="cuda", cache_enabled=True):
                    outputs = model(samples, targets=targets)
                with torch.autocast(device_type="cuda", enabled=False):
                    loss_dict = criterion(outputs, targets, **metas)
                loss = sum(loss_dict.values())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(samples, targets=targets)
                loss_dict = criterion(outputs, targets, **metas)
                loss = sum(loss_dict.values())
                loss.backward()
                optimizer.step()
            torch.cuda.synchronize(device)
            last_loss = float(loss.detach().cpu())

        peak_alloc = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        free_end, _ = torch.cuda.mem_get_info(device)
        allowed = total_mem * args.safety_fraction
        reserve_ok = _gb(free_end) >= args.reserve_gb
        fraction_ok = peak_reserved <= allowed
        ok = reserve_ok and fraction_ok

        print("")
        print("Result")
        print(f"  probe mode: {args.probe_mode}")
        print(f"  probe scale: {scale_desc}")
        if selected_images:
            print("  densest images in simulated batch:")
            for idx, image in enumerate(selected_images[:per_gpu_batch]):
                print(
                    "    "
                    f"{idx:02d}: image_id={int(image['image_id'])}, "
                    f"boxes={int(image['box_count'])}, "
                    f"size={int(image['width'])}x{int(image['height'])}, "
                    f"file={image['file_name']}"
                )
        print(f"  steps: {args.steps}")
        print(f"  last loss: {last_loss:.6f}" if last_loss is not None else "  last loss: n/a")
        print(f"  peak allocated: {_gb(peak_alloc):.2f} GiB")
        print(f"  peak reserved: {_gb(peak_reserved):.2f} GiB")
        print(f"  free after probe: {_gb(free_end):.2f} GiB")
        print(f"  safety fraction limit: {_gb(allowed):.2f} GiB ({args.safety_fraction:.0%})")
        print(f"  reserve requirement: {args.reserve_gb:.2f} GiB")

        if ok:
            print("  verdict: PASS - this parameter set is unlikely to OOM on the probed GPU.")
            return 0

        print("  verdict: WARN - the probe passed, but memory headroom is too small.")
        print("  suggestion: reduce TRAIN_TOTAL_BATCH_SIZE, reduce GPUS per node mismatch,")
        print("              keep USE_AMP=1, or lower multiscale/input size.")
        return 1

    except RuntimeError as exc:
        if _oom_message(exc):
            print("")
            print("Result")
            print("  verdict: OOM - this parameter set ran out of CUDA memory during preflight.")
            print(f"  simulated per-GPU batch size: {per_gpu_batch}")
            print("  suggestion: lower TRAIN_TOTAL_BATCH_SIZE or increase GPUS.")
            print(f"  CUDA message: {exc}")
            return 2
        traceback.print_exc()
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
