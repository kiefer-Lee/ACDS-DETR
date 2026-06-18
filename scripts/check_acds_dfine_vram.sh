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
REMAP_CATEGORY_IDS=${REMAP_CATEGORY_IDS:-1}
DFINE_ANN_DIR=${DFINE_ANN_DIR:-$DATA_ROOT/annotations/dfine}

NUM_CLASSES=${NUM_CLASSES:-10}
GPUS=${GPUS:-1}
CUDA_DEVICES=${CUDA_DEVICES:-0}
SEED=${SEED:-0}
USE_AMP=${USE_AMP:-1}
TRAIN_TOTAL_BATCH_SIZE=${TRAIN_TOTAL_BATCH_SIZE:-64}
STOP_EPOCH=${STOP_EPOCH:-}

VRAM_DEVICE_INDEX=${VRAM_DEVICE_INDEX:-0}
VRAM_STEPS=${VRAM_STEPS:-1}
VRAM_PROBE_SCALE=${VRAM_PROBE_SCALE:-max}
VRAM_PROBE_SIZE=${VRAM_PROBE_SIZE:-}
VRAM_SAFETY_FRACTION=${VRAM_SAFETY_FRACTION:-0.90}
VRAM_RESERVE_GB=${VRAM_RESERVE_GB:-1.0}

cd "$DFINE_ROOT"
export PYTHONPATH="$DFINE_ROOT:$ACDS_ROOT:${PYTHONPATH:-}"

echo "SOD root: $SOD_ROOT"
echo "D-FINE root: $DFINE_ROOT"
echo "VisDrone root: $DATA_ROOT"
echo "Train images: $TRAIN_IMG_FOLDER"
echo "Train annotation: $TRAIN_ANN_FILE"
echo "Config: $CONFIG"
echo "TRAIN_TOTAL_BATCH_SIZE: $TRAIN_TOTAL_BATCH_SIZE"
echo "GPUS: $GPUS"

if [[ ! -d "$TRAIN_IMG_FOLDER" ]]; then
  echo "Missing image folder: $TRAIN_IMG_FOLDER" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_ANN_FILE" ]]; then
  echo "Missing annotation file: $TRAIN_ANN_FILE" >&2
  exit 1
fi

TRAIN_ANN_FOR_DFINE="$TRAIN_ANN_FILE"
if [[ "$REMAP_CATEGORY_IDS" == "1" || "$REMAP_CATEGORY_IDS" == "true" ]]; then
  TRAIN_ANN_FOR_DFINE="$DFINE_ANN_DIR/train_zero_based.json"
  python "$ACDS_ROOT/acds_dfine/tools/remap_coco_categories_zero_based.py" \
    --ann-in "$TRAIN_ANN_FILE" \
    --ann-out "$TRAIN_ANN_FOR_DFINE"
fi

ARGS=(
  --dfine-root "$DFINE_ROOT"
  --acds-root "$ACDS_ROOT"
  --config "$CONFIG"
  --train-img-folder "$TRAIN_IMG_FOLDER"
  --train-ann-file "$TRAIN_ANN_FOR_DFINE"
  --num-classes "$NUM_CLASSES"
  --total-batch-size "$TRAIN_TOTAL_BATCH_SIZE"
  --world-size "$GPUS"
  --use-amp "$USE_AMP"
  --device-index "$VRAM_DEVICE_INDEX"
  --steps "$VRAM_STEPS"
  --probe-scale "$VRAM_PROBE_SCALE"
  --safety-fraction "$VRAM_SAFETY_FRACTION"
  --reserve-gb "$VRAM_RESERVE_GB"
  --seed "$SEED"
)

if [[ -n "$STOP_EPOCH" ]]; then
  ARGS+=(--stop-epoch "$STOP_EPOCH")
fi

if [[ -n "$VRAM_PROBE_SIZE" ]]; then
  ARGS+=(--probe-size "$VRAM_PROBE_SIZE")
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python \
  "$ACDS_ROOT/acds_dfine/tools/check_dfine_vram.py" "${ARGS[@]}" "$@"

