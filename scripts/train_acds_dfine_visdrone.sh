#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_SOD_ROOT=/data/libaichuan/Projects/SOD
if [[ -d "$DEFAULT_SOD_ROOT" ]]; then
  SOD_ROOT=${SOD_ROOT:-$DEFAULT_SOD_ROOT}
else
  SOD_ROOT=${SOD_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
fi

ACDS_ROOT=${ACDS_ROOT:-$SOD_ROOT/ACDS-DETR}
DFINE_ROOT=${DFINE_ROOT:-$SOD_ROOT/D-FINE}

if [[ -d "$SOD_ROOT/Dataset" ]]; then
  DEFAULT_DATASET_ROOT="$SOD_ROOT/Dataset"
else
  DEFAULT_DATASET_ROOT="$SOD_ROOT/Datasets"
fi

DATASET_ROOT=${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}
DATA_ROOT=${DATA_ROOT:-$DATASET_ROOT/VisDrone}
CONFIG=${CONFIG:-configs/dfine/custom/acds_dfine_hgnetv2_s_visdrone.yml}

TRAIN_IMG_FOLDER=${TRAIN_IMG_FOLDER:-$DATA_ROOT/VisDrone2019-DET-train/VisDrone2019-DET-train/images}
TRAIN_ANN_FILE=${TRAIN_ANN_FILE:-$DATA_ROOT/annotations/train.json}
VAL_IMG_FOLDER=${VAL_IMG_FOLDER:-$DATA_ROOT/VisDrone2019-DET-val/VisDrone2019-DET-val/images}
VAL_ANN_FILE=${VAL_ANN_FILE:-$DATA_ROOT/annotations/val.json}

NUM_CLASSES=${NUM_CLASSES:-10}
GPUS=${GPUS:-1}
MASTER_PORT=${MASTER_PORT:-7777}
CUDA_DEVICES=${CUDA_DEVICES:-0}
SEED=${SEED:-0}
USE_AMP=${USE_AMP:-1}
TRAIN_TOTAL_BATCH_SIZE=${TRAIN_TOTAL_BATCH_SIZE:-64}
VAL_TOTAL_BATCH_SIZE=${VAL_TOTAL_BATCH_SIZE:-128}
OUTPUT_DIR=${OUTPUT_DIR:-./output/acds_dfine_hgnetv2_s_visdrone}

cd "$DFINE_ROOT"
export PYTHONPATH="$DFINE_ROOT:$ACDS_ROOT:${PYTHONPATH:-}"

echo "SOD root: $SOD_ROOT"
echo "D-FINE root: $DFINE_ROOT"
echo "VisDrone root: $DATA_ROOT"
echo "Train images: $TRAIN_IMG_FOLDER"
echo "Train annotation: $TRAIN_ANN_FILE"
echo "Val images: $VAL_IMG_FOLDER"
echo "Val annotation: $VAL_ANN_FILE"

for path in "$TRAIN_IMG_FOLDER" "$VAL_IMG_FOLDER"; do
  if [[ ! -d "$path" ]]; then
    echo "Missing image folder: $path" >&2
    exit 1
  fi
done

for path in "$TRAIN_ANN_FILE" "$VAL_ANN_FILE"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing annotation file: $path" >&2
    exit 1
  fi
done

ARGS=(
  -c "$CONFIG"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
  -u
  num_classes="$NUM_CLASSES"
  train_dataloader.total_batch_size="$TRAIN_TOTAL_BATCH_SIZE"
  train_dataloader.dataset.img_folder="$TRAIN_IMG_FOLDER"
  train_dataloader.dataset.ann_file="$TRAIN_ANN_FILE"
  val_dataloader.total_batch_size="$VAL_TOTAL_BATCH_SIZE"
  val_dataloader.dataset.img_folder="$VAL_IMG_FOLDER"
  val_dataloader.dataset.ann_file="$VAL_ANN_FILE"
)

if [[ "$USE_AMP" == "1" || "$USE_AMP" == "true" ]]; then
  ARGS+=(--use-amp)
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" torchrun \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS" \
  train.py "${ARGS[@]}" "$@"
