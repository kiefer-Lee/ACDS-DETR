#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
DEFAULT_DATA_ROOT="$PROJECT_ROOT/../Datasets/UAVDT"
if [[ -d "$DEFAULT_DATA_ROOT" ]]; then
  DEFAULT_DATA_ROOT=$(cd "$DEFAULT_DATA_ROOT" && pwd)
fi
DATA_ROOT=${DATA_ROOT:-$DEFAULT_DATA_ROOT}

CONFIG=${CONFIG:-mmdet_acds/configs/acds_detr_r50_uavdt.py}
MIM_PACKAGE=${MIM_PACKAGE:-mmdet}
GPUS=${GPUS:-1}
SEED=${SEED:-0}
CUDA_DEVICES=${CUDA_DEVICES:-0}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_SPLIT=${VAL_SPLIT:-val}
TRAIN_FRAME_STRIDE=${TRAIN_FRAME_STRIDE:-6}
VAL_FRAME_STRIDE=${VAL_FRAME_STRIDE:-6}
DATASET_TAG=${DATASET_TAG:-stride${TRAIN_FRAME_STRIDE}}

TRAIN_ANN=${TRAIN_ANN:-annotations/train_stride${TRAIN_FRAME_STRIDE}.json}
VAL_ANN=${VAL_ANN:-annotations/val_stride${VAL_FRAME_STRIDE}.json}
TRAIN_IMG_PREFIX=${TRAIN_IMG_PREFIX:-}
VAL_IMG_PREFIX=${VAL_IMG_PREFIX:-}

CONFIG_NAME=$(basename "$CONFIG" .py)
WORK_DIR=${WORK_DIR:-work_dirs/uavdt/${CONFIG_NAME}/${DATASET_TAG}/seed_${SEED}}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -f "$DATA_ROOT/$TRAIN_ANN" ]]; then
  echo "Generating train annotation: $DATA_ROOT/$TRAIN_ANN"
  python mmdet_acds/tools/convert_uavdt_to_coco.py \
    --root "$DATA_ROOT" \
    --split "$TRAIN_SPLIT" \
    --frame-stride "$TRAIN_FRAME_STRIDE" \
    --output "$DATA_ROOT/$TRAIN_ANN"
fi

if [[ ! -f "$DATA_ROOT/$VAL_ANN" ]]; then
  echo "Generating val annotation: $DATA_ROOT/$VAL_ANN"
  python mmdet_acds/tools/convert_uavdt_to_coco.py \
    --root "$DATA_ROOT" \
    --split "$VAL_SPLIT" \
    --frame-stride "$VAL_FRAME_STRIDE" \
    --output "$DATA_ROOT/$VAL_ANN"
fi

if [[ -n "$TRAIN_IMG_PREFIX" && ! -d "$DATA_ROOT/$TRAIN_IMG_PREFIX" ]]; then
  echo "Missing train images: $DATA_ROOT/$TRAIN_IMG_PREFIX" >&2
  exit 1
fi

if [[ -n "$VAL_IMG_PREFIX" && ! -d "$DATA_ROOT/$VAL_IMG_PREFIX" ]]; then
  echo "Missing val images: $DATA_ROOT/$VAL_IMG_PREFIX" >&2
  exit 1
fi

if [[ "$CUDA_DEVICES" != "all" && "$CUDA_DEVICES" != "ALL" ]]; then
  IFS=',' read -ra VISIBLE_DEVICES <<< "$CUDA_DEVICES"
  if (( ${#VISIBLE_DEVICES[@]} < GPUS )); then
    echo "GPUS=$GPUS but CUDA_DEVICES exposes only ${#VISIBLE_DEVICES[@]} device(s): $CUDA_DEVICES" >&2
    echo "Set GPUS to the number of visible devices, e.g. CUDA_DEVICES=0 GPUS=1 or CUDA_DEVICES=0,1 GPUS=2." >&2
    exit 1
  fi
fi

COMMON_CFG_OPTIONS=(
  data_root="$DATA_ROOT"
  train_dataloader.dataset.data_root="$DATA_ROOT"
  train_dataloader.dataset.ann_file="$TRAIN_ANN"
  train_dataloader.dataset.data_prefix.img="$TRAIN_IMG_PREFIX"
  train_dataloader.batch_size=3
  'train_dataloader.dataset.metainfo.classes=("car","truck","bus")'
  val_dataloader.dataset.data_root="$DATA_ROOT"
  val_dataloader.dataset.ann_file="$VAL_ANN"
  val_dataloader.dataset.data_prefix.img="$VAL_IMG_PREFIX"
  val_dataloader.batch_size=3
  'val_dataloader.dataset.metainfo.classes=("car","truck","bus")'
  test_dataloader.dataset.data_root="$DATA_ROOT"
  test_dataloader.dataset.ann_file="$VAL_ANN"
  test_dataloader.dataset.data_prefix.img="$VAL_IMG_PREFIX"
  'test_dataloader.dataset.metainfo.classes=("car","truck","bus")'
  val_evaluator.ann_file="$DATA_ROOT/$VAL_ANN"
  test_evaluator.ann_file="$DATA_ROOT/$VAL_ANN"
  randomness.seed="$SEED"
  randomness.deterministic=False
  default_hooks.checkpoint.max_keep_ckpts=2
  default_hooks.checkpoint.save_last=True
)

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" mim train "$MIM_PACKAGE" "$CONFIG" \
  --launcher pytorch \
  --gpus "$GPUS" \
  --work-dir "$WORK_DIR" \
  --cfg-options "${COMMON_CFG_OPTIONS[@]}" "$@"
