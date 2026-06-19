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

if [[ -d "$SOD_ROOT/Datasets/UAVDT" ]]; then
  DEFAULT_DATASET_ROOT="$SOD_ROOT/Datasets"
elif [[ -d "$SOD_ROOT/Dataset/UAVDT" ]]; then
  DEFAULT_DATASET_ROOT="$SOD_ROOT/Dataset"
else
  DEFAULT_DATASET_ROOT="$SOD_ROOT/Datasets"
fi

DATASET_ROOT=${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}
DATA_ROOT=${DATA_ROOT:-$DATASET_ROOT/UAVDT}
CONFIG=${CONFIG:-configs/dfine/custom/acds_dfine_hgnetv2_s_uavdt.yml}

TRAIN_IMG_FOLDER=${TRAIN_IMG_FOLDER:-$DATA_ROOT}
TRAIN_ANN_FILE=${TRAIN_ANN_FILE:-$DATA_ROOT/annotations/train_stride6.json}
VAL_IMG_FOLDER=${VAL_IMG_FOLDER:-$DATA_ROOT}
VAL_ANN_FILE=${VAL_ANN_FILE:-$DATA_ROOT/annotations/val_stride6.json}
REMAP_CATEGORY_IDS=${REMAP_CATEGORY_IDS:-1}
DFINE_ANN_DIR=${DFINE_ANN_DIR:-$DATA_ROOT/annotations/dfine}

NUM_CLASSES=${NUM_CLASSES:-3}
GPUS=${GPUS:-1}
MASTER_PORT=${MASTER_PORT:-7777}
CUDA_DEVICES=${CUDA_DEVICES:-0}
SEED=${SEED:-0}
USE_AMP=${USE_AMP:-1}
EPOCHS=${EPOCHS:-220}
TRAIN_TOTAL_BATCH_SIZE=${TRAIN_TOTAL_BATCH_SIZE:-64}
VAL_TOTAL_BATCH_SIZE=${VAL_TOTAL_BATCH_SIZE:-128}
TRAIN_NUM_WORKERS=${TRAIN_NUM_WORKERS:-12}
VAL_NUM_WORKERS=${VAL_NUM_WORKERS:-8}
PIN_MEMORY=${PIN_MEMORY:-true}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-true}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
VAL_INTERVAL=${VAL_INTERVAL:-1}
CHECKPOINT_FREQ=${CHECKPOINT_FREQ:-12}
PRINT_FREQ=${PRINT_FREQ:-100}
STOP_EPOCH=${STOP_EPOCH:-}
OUTPUT_DIR=${OUTPUT_DIR:-./output/acds_dfine_hgnetv2_s_uavdt}

cd "$DFINE_ROOT"
export PYTHONPATH="$DFINE_ROOT:$ACDS_ROOT:${PYTHONPATH:-}"

echo "SOD root: $SOD_ROOT"
echo "D-FINE root: $DFINE_ROOT"
echo "UAVDT root: $DATA_ROOT"
echo "Train images root: $TRAIN_IMG_FOLDER"
echo "Train annotation: $TRAIN_ANN_FILE"
echo "Val images root: $VAL_IMG_FOLDER"
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

TRAIN_ANN_FOR_DFINE="$TRAIN_ANN_FILE"
VAL_ANN_FOR_DFINE="$VAL_ANN_FILE"
if [[ "$REMAP_CATEGORY_IDS" == "1" || "$REMAP_CATEGORY_IDS" == "true" ]]; then
  TRAIN_ANN_FOR_DFINE="$DFINE_ANN_DIR/train_stride6_zero_based.json"
  VAL_ANN_FOR_DFINE="$DFINE_ANN_DIR/val_stride6_zero_based.json"
  python "$ACDS_ROOT/acds_dfine/tools/remap_coco_categories_zero_based.py" \
    --ann-in "$TRAIN_ANN_FILE" \
    --ann-out "$TRAIN_ANN_FOR_DFINE"
  python "$ACDS_ROOT/acds_dfine/tools/remap_coco_categories_zero_based.py" \
    --ann-in "$VAL_ANN_FILE" \
    --ann-out "$VAL_ANN_FOR_DFINE"
fi

echo "D-FINE train annotation: $TRAIN_ANN_FOR_DFINE"
echo "D-FINE val annotation: $VAL_ANN_FOR_DFINE"

ARGS=(
  -c "$CONFIG"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
  -u
  num_classes="$NUM_CLASSES"
  epochs="$EPOCHS"
  val_interval="$VAL_INTERVAL"
  checkpoint_freq="$CHECKPOINT_FREQ"
  print_freq="$PRINT_FREQ"
  train_dataloader.total_batch_size="$TRAIN_TOTAL_BATCH_SIZE"
  train_dataloader.num_workers="$TRAIN_NUM_WORKERS"
  train_dataloader.dataset.img_folder="$TRAIN_IMG_FOLDER"
  train_dataloader.dataset.ann_file="$TRAIN_ANN_FOR_DFINE"
  val_dataloader.total_batch_size="$VAL_TOTAL_BATCH_SIZE"
  val_dataloader.num_workers="$VAL_NUM_WORKERS"
  val_dataloader.dataset.img_folder="$VAL_IMG_FOLDER"
  val_dataloader.dataset.ann_file="$VAL_ANN_FOR_DFINE"
)

if (( TRAIN_NUM_WORKERS > 0 )); then
  ARGS+=(
    train_dataloader.pin_memory="$PIN_MEMORY"
    train_dataloader.persistent_workers="$PERSISTENT_WORKERS"
    train_dataloader.prefetch_factor="$PREFETCH_FACTOR"
  )
fi

if (( VAL_NUM_WORKERS > 0 )); then
  ARGS+=(
    val_dataloader.pin_memory="$PIN_MEMORY"
    val_dataloader.persistent_workers="$PERSISTENT_WORKERS"
    val_dataloader.prefetch_factor="$PREFETCH_FACTOR"
  )
fi

if [[ -n "$STOP_EPOCH" ]]; then
  ARGS+=(
    train_dataloader.dataset.transforms.policy.epoch="$STOP_EPOCH"
    train_dataloader.collate_fn.stop_epoch="$STOP_EPOCH"
  )
fi

if [[ "$USE_AMP" == "1" || "$USE_AMP" == "true" ]]; then
  ARGS+=(--use-amp)
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" torchrun \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS" \
  train.py "${ARGS[@]}" "$@"
