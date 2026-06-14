#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/data/libaichuan/Projects/SOD/ACDS-DETR}
DATA_ROOT=${DATA_ROOT:-/root/blockdata/Datasets/tiny_set}

ANN=${ANN:-$DATA_ROOT/annotations/tiny_set_test.json}
DET=${DET:-}
MAX_DETS=${MAX_DETS:-100 300 500}
CATEGORY_MODE=${CATEGORY_MODE:-keep}
SUMMARY_JSON=${SUMMARY_JSON:-}

if [[ -z "$DET" && $# -gt 0 && "$1" != --* ]]; then
  DET=$1
  shift
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

if [[ -z "$DET" ]]; then
  echo "Set DET=/path/to/predictions.bbox.json or bbox_predictions.json" >&2
  exit 1
fi

ARGS=(
  --ann "$ANN"
  --det "$DET"
  --category-mode "$CATEGORY_MODE"
  --max-dets $MAX_DETS
)

if [[ -n "$SUMMARY_JSON" ]]; then
  ARGS+=(--summary-json "$SUMMARY_JSON")
fi

python mmdet_acds/tools/eval_tinyperson.py "${ARGS[@]}" "$@"
