#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_DQDETR_CONFIG=mmdet_comparison/configs/dq_detr_r50_visdrone.py

export CONFIG=${CONFIG:-${DQDETR_CONFIG:-$DEFAULT_DQDETR_CONFIG}}

PROJECT_ROOT=${PROJECT_ROOT:-/data/libaichuan/Projects/SOD/ACDS-DETR}
if [[ ! -f "$PROJECT_ROOT/$CONFIG" && ! -f "$CONFIG" ]]; then
  echo "DQ-DETR config not found: $CONFIG" >&2
  echo "Current repository does not contain a dq_detr config." >&2
  echo "Set CONFIG=/path/to/dqdetr_config.py if your DQ-DETR config is elsewhere." >&2
  exit 1
fi

exec bash "$SCRIPT_DIR/train_tinyperson.sh" "$@"
