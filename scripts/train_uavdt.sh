#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/libaichuan/Projects/SOD/ACDS-DETR}
DATA_ROOT=${DATA_ROOT:-/root/blockdata/Datasets/UAVDT}

CONFIG=${CONFIG:-mmdet_acds/configs/acds_detr_r50_uavdt.py}
MIM_PACKAGE=${MIM_PACKAGE:-mmdet}
GPUS=${GPUS:-1}
SEED=${SEED:-0}
CUDA_DEVICES=${CUDA_DEVICES:-0}

TRAIN_ANN=${TRAIN_ANN:-annotations/train.json}
VAL_ANN=${VAL_ANN:-annotations/val.json}
TRAIN_IMG_PREFIX=${TRAIN_IMG_PREFIX:-train/images/}
VAL_IMG_PREFIX=${VAL_IMG_PREFIX:-val/images/}

CONFIG_NAME=$(basename "$CONFIG" .py)
WORK_DIR=${WORK_DIR:-work_dirs/uavdt/${CONFIG_NAME}/seed_${SEED}}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -f "$DATA_ROOT/$TRAIN_ANN" ]]; then
  echo "Missing train annotation: $DATA_ROOT/$TRAIN_ANN" >&2
  exit 1
fi

if [[ ! -f "$DATA_ROOT/$VAL_ANN" ]]; then
  echo "Missing val annotation: $DATA_ROOT/$VAL_ANN" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT/$TRAIN_IMG_PREFIX" ]]; then
  echo "Missing train images: $DATA_ROOT/$TRAIN_IMG_PREFIX" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT/$VAL_IMG_PREFIX" ]]; then
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
  train_dataloader.dataset.data_prefix.img=""
  'train_dataloader.dataset.metainfo.classes=("car","bus","truck")'
  val_dataloader.dataset.data_root="$DATA_ROOT"
  val_dataloader.dataset.ann_file="$VAL_ANN"
  val_dataloader.dataset.data_prefix.img=""
  'val_dataloader.dataset.metainfo.classes=("car","bus","truck")'
  test_dataloader.dataset.data_root="$DATA_ROOT"
  test_dataloader.dataset.ann_file="$VAL_ANN"
  test_dataloader.dataset.data_prefix.img=""
  'test_dataloader.dataset.metainfo.classes=("car","bus","truck")'
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
