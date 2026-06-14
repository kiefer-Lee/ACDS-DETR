#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/libaichuan/Projects/SOD/ACDS-DETR}
DATA_ROOT=${DATA_ROOT:-/root/blockdata/Datasets/tiny_set}

CONFIG=${CONFIG:-mmdet_acds/configs/acds_detr_r50_visdrone.py}
MIM_PACKAGE=${MIM_PACKAGE:-mmdet}
GPUS=${GPUS:-1}
SEED=${SEED:-0}
CUDA_DEVICES=${CUDA_DEVICES:-0}

SLICE_TAG=${SLICE_TAG:-sw640_sh512}
TRAIN_ANN_SRC=${TRAIN_ANN_SRC:-annotations/corner/tiny_set_train_${SLICE_TAG}.json}
TRAIN_ANN=${TRAIN_ANN:-annotations/mmdet/tiny_set_train_${SLICE_TAG}_sliced.json}
VAL_ANN=${VAL_ANN:-annotations/tiny_set_test.json}
TRAIN_IMG_PREFIX=${TRAIN_IMG_PREFIX:-train/}
SLICED_TRAIN_IMG_PREFIX=${SLICED_TRAIN_IMG_PREFIX:-train_sliced_${SLICE_TAG}/}
VAL_IMG_PREFIX=${VAL_IMG_PREFIX:-test/}

CONFIG_NAME=$(basename "$CONFIG" .py)
WORK_DIR=${WORK_DIR:-work_dirs/tinyperson/${SLICE_TAG}/${CONFIG_NAME}/seed_${SEED}}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -f "$DATA_ROOT/$TRAIN_ANN_SRC" ]]; then
  echo "Missing sliced train annotation: $DATA_ROOT/$TRAIN_ANN_SRC" >&2
  exit 1
fi

if [[ ! -f "$DATA_ROOT/$VAL_ANN" ]]; then
  echo "Missing val annotation: $DATA_ROOT/$VAL_ANN" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT/$TRAIN_IMG_PREFIX/labeled_images" ]]; then
  echo "Missing train images: $DATA_ROOT/$TRAIN_IMG_PREFIX/labeled_images" >&2
  echo "If needed, run: tar -xzf $DATA_ROOT/train.tar.gz -C $DATA_ROOT" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT/$VAL_IMG_PREFIX/labeled_images" ]]; then
  echo "Missing val images: $DATA_ROOT/$VAL_IMG_PREFIX/labeled_images" >&2
  echo "If needed, run: tar -xzf $DATA_ROOT/test.tar.gz -C $DATA_ROOT" >&2
  exit 1
fi

mkdir -p "$DATA_ROOT/annotations/mmdet" "$DATA_ROOT/$SLICED_TRAIN_IMG_PREFIX"

python - "$DATA_ROOT/$TRAIN_ANN_SRC" "$DATA_ROOT/$TRAIN_ANN" "$DATA_ROOT/$TRAIN_IMG_PREFIX" "$DATA_ROOT/$SLICED_TRAIN_IMG_PREFIX" <<'PY'
import json
import sys
from pathlib import Path
from PIL import Image

src, dst, src_img_root, dst_img_root = map(Path, sys.argv[1:5])
data = json.loads(src.read_text())

for image_info in data.get("images", []):
    corner = image_info.get("corner")
    if corner is None:
        continue

    rel_path = Path(image_info["file_name"])
    source_path = src_img_root / rel_path
    suffix = rel_path.suffix or ".jpg"
    crop_rel_path = rel_path.parent / f"{rel_path.stem}__crop_{int(image_info['id']):06d}{suffix}"
    crop_path = dst_img_root / crop_rel_path

    if not source_path.is_file():
        raise FileNotFoundError(f"Missing source image: {source_path}")

    if not crop_path.is_file():
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        x1, y1, x2, y2 = [int(v) for v in corner]
        with Image.open(source_path) as image:
            crop = image.crop((x1, y1, x2, y2))
            crop.save(crop_path)

    image_info["file_name"] = crop_rel_path.as_posix()
    image_info.pop("corner", None)

data["categories"] = [
    {"id": 1, "name": "sea_person", "supercategory": "person"},
    {"id": 2, "name": "earth_person", "supercategory": "person"},
]

dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(data), encoding="utf-8")
PY

COMMON_CFG_OPTIONS=(
  data_root="$DATA_ROOT"
  train_dataloader.dataset.data_root="$DATA_ROOT"
  train_dataloader.dataset.ann_file="$TRAIN_ANN"
  train_dataloader.dataset.data_prefix.img="$SLICED_TRAIN_IMG_PREFIX"
  'train_dataloader.dataset.metainfo.classes=("sea_person","earth_person")'
  val_dataloader.dataset.data_root="$DATA_ROOT"
  val_dataloader.dataset.ann_file="$VAL_ANN"
  val_dataloader.dataset.data_prefix.img="$VAL_IMG_PREFIX"
  'val_dataloader.dataset.metainfo.classes=("sea_person","earth_person")'
  test_dataloader.dataset.data_root="$DATA_ROOT"
  test_dataloader.dataset.ann_file="$VAL_ANN"
  test_dataloader.dataset.data_prefix.img="$VAL_IMG_PREFIX"
  'test_dataloader.dataset.metainfo.classes=("sea_person","earth_person")'
  val_evaluator.ann_file="$DATA_ROOT/$VAL_ANN"
  test_evaluator.ann_file="$DATA_ROOT/$VAL_ANN"
  randomness.seed="$SEED"
  randomness.deterministic=False
  default_hooks.checkpoint.max_keep_ckpts=2
  default_hooks.checkpoint.save_last=True
  num_classes=2
  model.bbox_head.num_classes=2
)

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" mim train "$MIM_PACKAGE" "$CONFIG" \
  --launcher pytorch \
  --gpus "$GPUS" \
  --work-dir "$WORK_DIR" \
  --cfg-options "${COMMON_CFG_OPTIONS[@]}" "$@"
