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
DATA_ROOT=${DATA_ROOT:-$SOD_ROOT/Datasets/SODA-D}

SPLITS=${SPLITS:-"train val test"}
FRACTION=${FRACTION:-0.25}
SEED=${SEED:-0}
COPY_MODE=${COPY_MODE:-copy}
OVERWRITE=${OVERWRITE:-1}
DRY_RUN=${DRY_RUN:-0}

ARGS=(
  --data-root "$DATA_ROOT"
  --splits $SPLITS
  --fraction "$FRACTION"
  --seed "$SEED"
  --copy-mode "$COPY_MODE"
  --out-image-dir "mimi_images"
  --out-annotation-dir "mini-annotations"
)

if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then
  ARGS+=(--overwrite)
fi

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  ARGS+=(--dry-run)
fi

python "$ACDS_ROOT/acds_dfine/tools/stratified_sample_soda_d.py" "${ARGS[@]}" "$@"
