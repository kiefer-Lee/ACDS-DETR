#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/libaichuan/Projects/SOD/ACDS-DETR}
DATA_ROOT=${DATA_ROOT:-/root/blockdata/Datasets/tiny_set}

CONFIG=${CONFIG:-mmdet_comparison/configs/yolov8_s_visdrone.py}
MIM_PACKAGE=${MIM_PACKAGE:-mmyolo}
CUDA_DEVICES=${CUDA_DEVICES:-0}
SEED=${SEED:-0}

VAL_ANN=${VAL_ANN:-annotations/tiny_set_test.json}
VAL_IMG_PREFIX=${VAL_IMG_PREFIX:-test/}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-1}
TEST_NUM_WORKERS=${TEST_NUM_WORKERS:-4}
EXPORT_JSON=${EXPORT_JSON:-1}

CONFIG_NAME=$(basename "$CONFIG" .py)
WORK_DIR=${WORK_DIR:-work_dirs/tinyperson/yolo/${CONFIG_NAME}/seed_${SEED}}
TEST_WORK_DIR=${TEST_WORK_DIR:-${WORK_DIR}/test}
OUTFILE_PREFIX=${OUTFILE_PREFIX:-${TEST_WORK_DIR}/predictions}
CHECKPOINT=${CHECKPOINT:-}

cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ ! -f "$DATA_ROOT/$VAL_ANN" ]]; then
  echo "Missing val annotation: $DATA_ROOT/$VAL_ANN" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT/$VAL_IMG_PREFIX/labeled_images" ]]; then
  echo "Missing val images: $DATA_ROOT/$VAL_IMG_PREFIX/labeled_images" >&2
  echo "If needed, run: tar -xzf $DATA_ROOT/test.tar.gz -C $DATA_ROOT" >&2
  exit 1
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
  num_classes=2
  'metainfo.classes=("sea_person","earth_person")'
  val_dataloader.batch_size="$TEST_BATCH_SIZE"
  val_dataloader.num_workers="$TEST_NUM_WORKERS"
  val_dataloader.dataset.data_root="$DATA_ROOT"
  val_dataloader.dataset.ann_file="$VAL_ANN"
  val_dataloader.dataset.data_prefix.img="$VAL_IMG_PREFIX"
  'val_dataloader.dataset.metainfo.classes=("sea_person","earth_person")'
  test_dataloader.batch_size="$TEST_BATCH_SIZE"
  test_dataloader.num_workers="$TEST_NUM_WORKERS"
  test_dataloader.dataset.data_root="$DATA_ROOT"
  test_dataloader.dataset.ann_file="$VAL_ANN"
  test_dataloader.dataset.data_prefix.img="$VAL_IMG_PREFIX"
  'test_dataloader.dataset.metainfo.classes=("sea_person","earth_person")'
  val_evaluator.ann_file="$DATA_ROOT/$VAL_ANN"
  test_evaluator.ann_file="$DATA_ROOT/$VAL_ANN"
  model.bbox_head.head_module.num_classes=2
  model.train_cfg.assigner.num_classes=2
)

if [[ "$EXPORT_JSON" == "1" || "$EXPORT_JSON" == "true" ]]; then
  mkdir -p "$(dirname "$OUTFILE_PREFIX")"
  COMMON_CFG_OPTIONS+=(
    val_evaluator.outfile_prefix="$OUTFILE_PREFIX"
    test_evaluator.outfile_prefix="$OUTFILE_PREFIX"
  )
fi

echo "Testing config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Work dir: $TEST_WORK_DIR"
if [[ "$EXPORT_JSON" == "1" || "$EXPORT_JSON" == "true" ]]; then
  echo "Prediction json prefix: $OUTFILE_PREFIX"
fi

CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" mim test "$MIM_PACKAGE" "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --work-dir "$TEST_WORK_DIR" \
  --cfg-options "${COMMON_CFG_OPTIONS[@]}" "$@"
