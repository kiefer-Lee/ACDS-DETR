"""Convert UAVDT DET/MOT-style txt annotations to COCO JSON.

Expected annotation columns are:

    frame_id, target_id, x, y, width, height, out_of_view, occlusion, category

The category ids are kept as UAVDT's 1-based ids: 1=car, 2=truck, 3=bus.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

from mmdet_acds.datasets.uavdt_metainfo import UAVDT_CLASSES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def _read_sequence_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if name and not name.startswith("#"):
            names.append(Path(name).stem)
    return names


def _extract_frame_id(path: Path) -> int | None:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _find_image_files(seq_dir: Path) -> dict[int, Path]:
    frame_to_paths: dict[int, list[Path]] = defaultdict(list)
    for path in seq_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            frame_id = _extract_frame_id(path)
            if frame_id is not None:
                frame_to_paths[frame_id].append(path)

    frame_to_path = {}
    for frame_id, paths in frame_to_paths.items():
        paths = sorted(paths)
        frame_to_path[frame_id] = paths[0]
    return frame_to_path


def _find_annotation_file(root: Path, seq_dir: Path, seq_name: str) -> Path | None:
    candidates = [
        seq_dir / "gt_whole.txt",
        seq_dir / "gt" / "gt_whole.txt",
        seq_dir / "annotations" / "gt_whole.txt",
        root / "GT" / f"{seq_name}_gt_whole.txt",
        root / "GT" / f"{seq_name}.txt",
        root / "annotations" / f"{seq_name}_gt_whole.txt",
        root / "annotations" / f"{seq_name}.txt",
        root / "annotations" / seq_name / "gt_whole.txt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _discover_sequences(root: Path, split_list: Path | None) -> list[tuple[str, Path, Path]]:
    requested = set(_read_sequence_names(split_list)) if split_list is not None else None
    sequences = []
    for seq_dir in sorted(p for p in root.rglob("*") if p.is_dir()):
        seq_name = seq_dir.name
        if requested is not None and seq_name not in requested:
            continue
        ann_file = _find_annotation_file(root, seq_dir, seq_name)
        if ann_file is None:
            continue
        if not _find_image_files(seq_dir):
            continue
        sequences.append((seq_name, seq_dir, ann_file))

    if requested is not None:
        found = {seq_name for seq_name, _, _ in sequences}
        missing = sorted(requested - found)
        if missing:
            raise FileNotFoundError(f"UAVDT sequences not found or incomplete: {missing}")
    if not sequences:
        raise FileNotFoundError(f"No UAVDT sequences found under {root}")
    return sequences


def _parse_annotation_line(line: str) -> tuple[int, float, float, float, float, int, int, int] | None:
    parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
    if len(parts) < 9:
        return None
    frame_id = int(float(parts[0]))
    x = float(parts[2])
    y = float(parts[3])
    bw = float(parts[4])
    bh = float(parts[5])
    out_of_view = int(float(parts[6]))
    occlusion = int(float(parts[7]))
    category_id = int(float(parts[8]))
    return frame_id, x, y, bw, bh, out_of_view, occlusion, category_id


def convert(
    root: str | Path,
    output: str | Path,
    split_list: str | Path | None = None,
    min_area: float = 1.0,
) -> None:
    root = Path(root)
    split_list_path = Path(split_list) if split_list is not None else None
    sequences = _discover_sequences(root, split_list_path)

    images = []
    annotations = []
    ann_id = 1
    image_id = 0

    for seq_name, seq_dir, ann_file in sequences:
        frame_to_path = _find_image_files(seq_dir)
        frame_to_image_id = {}
        for frame_id, img_path in sorted(frame_to_path.items()):
            with Image.open(img_path) as img:
                width, height = img.size
            rel_name = img_path.relative_to(root).as_posix()
            images.append(
                {
                    "id": image_id,
                    "file_name": rel_name,
                    "width": width,
                    "height": height,
                    "video_id": seq_name,
                    "frame_id": frame_id,
                }
            )
            frame_to_image_id[frame_id] = (image_id, width, height)
            image_id += 1

        for line in ann_file.read_text(encoding="utf-8").splitlines():
            parsed = _parse_annotation_line(line)
            if parsed is None:
                continue
            frame_id, x, y, bw, bh, out_of_view, occlusion, category_id = parsed
            if frame_id not in frame_to_image_id:
                continue
            if category_id < 1 or category_id > len(UAVDT_CLASSES) or bw <= 0 or bh <= 0:
                continue
            img_id, width, height = frame_to_image_id[frame_id]
            x0 = max(0.0, x)
            y0 = max(0.0, y)
            x1 = min(float(width), x + bw)
            y1 = min(float(height), y + bh)
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if area < min_area:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": area,
                    "iscrowd": 0,
                    "out_of_view": out_of_view,
                    "occlusion": occlusion,
                }
            )
            ann_id += 1

    data = {
        "info": {"description": "UAVDT converted to COCO"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(UAVDT_CLASSES)],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(images)} images and {len(annotations)} annotations to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="UAVDT root containing sequence folders.")
    parser.add_argument("--output", required=True, help="Output COCO json path.")
    parser.add_argument("--split-list", help="Optional txt file with one sequence name per line.")
    parser.add_argument("--min-area", type=float, default=1.0)
    args = parser.parse_args()
    convert(args.root, args.output, args.split_list, args.min_area)


if __name__ == "__main__":
    main()

