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

TEST_IMG_FOLDER=${TEST_IMG_FOLDER:-${VAL_IMG_FOLDER:-$DATA_ROOT/VisDrone2019-DET-val/VisDrone2019-DET-val/images}}
TEST_ANN_FILE=${TEST_ANN_FILE:-${VAL_ANN_FILE:-$DATA_ROOT/annotations/val.json}}
REMAP_CATEGORY_IDS=${REMAP_CATEGORY_IDS:-1}
DFINE_ANN_DIR=${DFINE_ANN_DIR:-$DATA_ROOT/annotations/dfine}

NUM_CLASSES=${NUM_CLASSES:-10}
GPUS=${GPUS:-1}
MASTER_PORT=${MASTER_PORT:-7777}
CUDA_DEVICES=${CUDA_DEVICES:-0}
VAL_TOTAL_BATCH_SIZE=${VAL_TOTAL_BATCH_SIZE:-10}
MAX_DETS=${MAX_DETS:-500}
OUTPUT_DIR=${OUTPUT_DIR:-./output/acds_dfine_hgnetv2_s_visdrone}
CHECKPOINT=/data/libaichuan/Projects/SOD/D-FINE/output/acds_dfine_hgnetv2_s_visdrone/best_stg1.pth

cd "$DFINE_ROOT"
export PYTHONPATH="$DFINE_ROOT:$ACDS_ROOT:${PYTHONPATH:-}"

echo "SOD root: $SOD_ROOT"
echo "D-FINE root: $DFINE_ROOT"
echo "VisDrone root: $DATA_ROOT"
echo "Test images: $TEST_IMG_FOLDER"
echo "Test annotation: $TEST_ANN_FILE"

if [[ ! -d "$TEST_IMG_FOLDER" ]]; then
  echo "Missing image folder: $TEST_IMG_FOLDER" >&2
  exit 1
fi

if [[ ! -f "$TEST_ANN_FILE" ]]; then
  echo "Missing annotation file: $TEST_ANN_FILE" >&2
  exit 1
fi

TEST_ANN_FOR_DFINE="$TEST_ANN_FILE"
if [[ "$REMAP_CATEGORY_IDS" == "1" || "$REMAP_CATEGORY_IDS" == "true" ]]; then
  TEST_ANN_FOR_DFINE="$DFINE_ANN_DIR/val_zero_based.json"
  python "$ACDS_ROOT/acds_dfine/tools/remap_coco_categories_zero_based.py" \
    --ann-in "$TEST_ANN_FILE" \
    --ann-out "$TEST_ANN_FOR_DFINE"
fi

echo "D-FINE test annotation: $TEST_ANN_FOR_DFINE"

if [[ -z "$CHECKPOINT" ]]; then
  if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "OUTPUT_DIR does not exist: $OUTPUT_DIR" >&2
    echo "Set CHECKPOINT=/path/to/model.pth or OUTPUT_DIR=/path/to/output_dir" >&2
    exit 1
  fi
  CHECKPOINT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*.pth" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
fi

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT:-<empty>}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" torchrun \
  --master_port="$MASTER_PORT" \
  --nproc_per_node="$GPUS" \
  train.py \
  -c "$CONFIG" \
  --test-only \
  -r "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR/test" \
  -u \
  num_classes="$NUM_CLASSES" \
  evaluator.max_dets="[1,10,$MAX_DETS]" \
  val_dataloader.total_batch_size="$VAL_TOTAL_BATCH_SIZE" \
  val_dataloader.dataset.img_folder="$TEST_IMG_FOLDER" \
  val_dataloader.dataset.ann_file="$TEST_ANN_FOR_DFINE" \
  "$@"
