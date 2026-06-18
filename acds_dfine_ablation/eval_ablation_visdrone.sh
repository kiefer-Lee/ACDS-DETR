#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_SOD_ROOT=/data/libaichuan/Projects/SOD
if [[ -d "$DEFAULT_SOD_ROOT" ]]; then
  SOD_ROOT=${SOD_ROOT:-$DEFAULT_SOD_ROOT}
else
  SOD_ROOT=${SOD_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
fi

TEST_SCRIPT=${TEST_SCRIPT:-$SOD_ROOT/ACDS-DETR/scripts/test_acds_dfine_visdrone.sh}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SOD_ROOT/D-FINE/output/ablation}
SEEDS=${SEEDS:-0}
EXPERIMENTS=${EXPERIMENTS:-"00_dfine_baseline 01_acq_only 02_small_matcher_only 03_scale_query_only 04_rsnds_only 05_acq_plus_small_matcher 06_scale_query_plus_rsnds 07_full_acds_dfine"}

for seed in $SEEDS; do
  for exp in $EXPERIMENTS; do
    config="$SCRIPT_DIR/$exp.yml"
    output_dir="$OUTPUT_ROOT/${exp}_seed${seed}"
    if [[ ! -f "$config" ]]; then
      echo "Missing ablation config: $config" >&2
      exit 1
    fi

    echo "Evaluating ACDS-D-FINE ablation: exp=$exp seed=$seed"
    CONFIG="$config" \
    OUTPUT_DIR="$output_dir" \
    bash "$TEST_SCRIPT"
  done
done

