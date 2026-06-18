#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ACDS_ROOT=${ACDS_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
DFINE_ROOT=${DFINE_ROOT:-$(cd "$ACDS_ROOT/../D-FINE" && pwd)}

DATA_ROOT=${DATA_ROOT:-$(cd "$ACDS_ROOT/../Datasets/VisDrone" 2>/dev/null && pwd || echo "$ACDS_ROOT/../Datasets/VisDrone")}
CONFIG=${CONFIG:-configs/dfine/custom/acds_dfine_hgnetv2_s_visdrone.yml}

VAL_IMG_FOLDER=${VAL_IMG_FOLDER:-$DATA_ROOT/images/val}
VAL_ANN_FILE=${VAL_ANN_FILE:-$DATA_ROOT/annotations/instances_val.json}

NUM_CLASSES=${NUM_CLASSES:-10}
GPUS=${GPUS:-1}
MASTER_PORT=${MASTER_PORT:-7777}
CUDA_DEVICES=${CUDA_DEVICES:-0}
VAL_TOTAL_BATCH_SIZE=${VAL_TOTAL_BATCH_SIZE:-128}
OUTPUT_DIR=${OUTPUT_DIR:-./output/acds_dfine_hgnetv2_s_visdrone}
CHECKPOINT=${CHECKPOINT:-}

cd "$DFINE_ROOT"
export PYTHONPATH="$DFINE_ROOT:$ACDS_ROOT:${PYTHONPATH:-}"

if [[ ! -d "$VAL_IMG_FOLDER" ]]; then
  echo "Missing image folder: $VAL_IMG_FOLDER" >&2
  exit 1
fi

if [[ ! -f "$VAL_ANN_FILE" ]]; then
  echo "Missing annotation file: $VAL_ANN_FILE" >&2
  exit 1
fi

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
  val_dataloader.total_batch_size="$VAL_TOTAL_BATCH_SIZE" \
  val_dataloader.dataset.img_folder="$VAL_IMG_FOLDER" \
  val_dataloader.dataset.ann_file="$VAL_ANN_FILE" \
  "$@"
