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
CUDA_DEVICES=${CUDA_DEVICES:-0}
SEED=${SEED:-0}
TEST_SPLIT=${TEST_SPLIT:-val}
TEST_FRAME_STRIDE=${TEST_FRAME_STRIDE:-1}
DATASET_TAG=${DATASET_TAG:-stride3}

TEST_ANN=${TEST_ANN:-annotations/val.json}
TEST_IMG_PREFIX=${TEST_IMG_PREFIX:-}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-1}
TEST_NUM_WORKERS=${TEST_NUM_WORKERS:-4}
EXPORT_JSON=${EXPORT_JSON:-1}

CONFIG_NAME=$(basename "$CONFIG" .py)
WORK_DIR=${WORK_DIR:-work_dirs/uavdt/${CONFIG_NAME}/${DATASET_TAG}/seed_${SEED}}
TEST_WORK_DIR=${TEST_WORK_DIR:-${WORK_DIR}/test}
OUTFILE_PREFIX=${OUTFILE_PREFIX:-${TEST_WORK_DIR}/predictions}
CHECKPOINT=${CHECKPOINT:-}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -f "$DATA_ROOT/$TEST_ANN" ]]; then
  echo "Generating test annotation: $DATA_ROOT/$TEST_ANN"
  python mmdet_acds/tools/convert_uavdt_to_coco.py \
    --root "$DATA_ROOT" \
    --split "$TEST_SPLIT" \
    --frame-stride "$TEST_FRAME_STRIDE" \
    --output "$DATA_ROOT/$TEST_ANN"
fi

if [[ -n "$TEST_IMG_PREFIX" && ! -d "$DATA_ROOT/$TEST_IMG_PREFIX" ]]; then
  echo "Missing test images: $DATA_ROOT/$TEST_IMG_PREFIX" >&2
  exit 1
fi

if [[ "$CUDA_DEVICES" != "all" && "$CUDA_DEVICES" != "ALL" ]]; then
  IFS=',' read -ra VISIBLE_DEVICES <<< "$CUDA_DEVICES"
  if (( ${#VISIBLE_DEVICES[@]} < 1 )); then
    echo "CUDA_DEVICES does not expose any device: $CUDA_DEVICES" >&2
    exit 1
  fi
fi

if [[ -z "$CHECKPOINT" ]]; then
  if [[ ! -d "$WORK_DIR" ]]; then
    echo "WORK_DIR does not exist: $WORK_DIR" >&2
    echo "Set CHECKPOINT=/path/to/model.pth or WORK_DIR=/path/to/work_dir" >&2
    exit 1
  fi
  CHECKPOINT=$(find "$WORK_DIR" -maxdepth 1 -name "*.pth" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
fi

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT:-<empty>}" >&2
  echo "Set CHECKPOINT=/path/to/model.pth" >&2
  exit 1
fi

COMMON_CFG_OPTIONS=(
  data_root="$DATA_ROOT"
  val_dataloader.batch_size="$TEST_BATCH_SIZE"
  val_dataloader.num_workers="$TEST_NUM_WORKERS"
  val_dataloader.dataset.data_root="$DATA_ROOT"
  val_dataloader.dataset.ann_file="$TEST_ANN"
  val_dataloader.dataset.data_prefix.img="$TEST_IMG_PREFIX"
  'val_dataloader.dataset.metainfo.classes=("car","truck","bus")'
  test_dataloader.batch_size="$TEST_BATCH_SIZE"
  test_dataloader.num_workers="$TEST_NUM_WORKERS"
  test_dataloader.dataset.data_root="$DATA_ROOT"
  test_dataloader.dataset.ann_file="$TEST_ANN"
  test_dataloader.dataset.data_prefix.img="$TEST_IMG_PREFIX"
  'test_dataloader.dataset.metainfo.classes=("car","truck","bus")'
  val_evaluator.ann_file="$DATA_ROOT/$TEST_ANN"
  test_evaluator.ann_file="$DATA_ROOT/$TEST_ANN"
)

if [[ "$EXPORT_JSON" == "1" || "$EXPORT_JSON" == "true" ]]; then
  mkdir -p "$(dirname "$OUTFILE_PREFIX")"
  COMMON_CFG_OPTIONS+=(
    val_evaluator.outfile_prefix="$OUTFILE_PREFIX"
    test_evaluator.outfile_prefix="$OUTFILE_PREFIX"
  )
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" mim test "$MIM_PACKAGE" "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --work-dir "$TEST_WORK_DIR" \
  --cfg-options "${COMMON_CFG_OPTIONS[@]}" "$@"
