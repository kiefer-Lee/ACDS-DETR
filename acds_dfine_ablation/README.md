# ACDS-D-FINE Ablation Plan

These configs evaluate the ACDS-D-FINE migration on VisDrone with the same
D-FINE-HGNetv2-S backbone, training schedule, dataloader, optimizer, and
official D-FINE training/evaluation entrypoints. Only the ACDS-related knobs
change across runs.

## Experiment Matrix

| ID | Config | ACQ loss | Small-object matcher | Scale-aware query | R-SNDS | Purpose |
| --- | --- | --- | --- | --- | --- | --- |
| 00 | `00_dfine_baseline.yml` | off | off | off | off | Fair VisDrone D-FINE baseline |
| 01 | `01_acq_only.yml` | on | off | off | off | Isolate auxiliary query-collision loss |
| 02 | `02_small_matcher_only.yml` | off | on | off | off | Isolate small-object cost reweighting in Hungarian matching |
| 03 | `03_scale_query_only.yml` | off | off | on | off | Isolate scale-aware query bias |
| 04 | `04_rsnds_only.yml` | off | off | off | on | Isolate reliability-guided deformable sampling |
| 05 | `05_acq_plus_small_matcher.yml` | on | on | off | off | Test assignment/loss-side complementarity |
| 06 | `06_scale_query_plus_rsnds.yml` | off | off | on | on | Test decoder-side complementarity |
| 07 | `07_full_acds_dfine.yml` | on | on | on | on | Full ACDS-D-FINE model |

## Recommended Metrics

Report the official COCO-style metrics from D-FINE's `CocoEvaluator`:

- `AP`, `AP50`, `AP75`
- `APs`, `APm`, `APl`
- `AR1`, `AR10`, `AR100`

For a small-object paper claim, `APs` is the primary metric. Keep `AP` and
`APm/APl` in the table to show whether the small-object gain harms general
detection quality.

## Fairness Controls

Use the same values for all runs unless explicitly stated:

- backbone: HGNetv2-S/B0
- image size and augmentation policy: inherited from D-FINE config
- epochs: 220
- train total batch size: 64
- validation total batch size: 128
- random seed: at least 3 seeds for paper-level reporting
- dataset split and annotation remapping: controlled by
  `ACDS-DETR/scripts/train_acds_dfine_visdrone.sh`

## Run One Experiment

From the server:

```bash
CONFIG=/data/libaichuan/Projects/SOD/ACDS-DETR/acds_dfine_ablation/07_full_acds_dfine.yml \
OUTPUT_DIR=/data/libaichuan/Projects/SOD/D-FINE/output/ablation/07_full_acds_dfine_seed0 \
SEED=0 \
GPUS=1 \
CUDA_DEVICES=0 \
bash /data/libaichuan/Projects/SOD/ACDS-DETR/scripts/train_acds_dfine_visdrone.sh
```

## Run the Full Ablation Set

```bash
GPUS=1 CUDA_DEVICES=0 bash /data/libaichuan/Projects/SOD/ACDS-DETR/acds_dfine_ablation/run_ablation_visdrone.sh
```

Optional overrides:

```bash
SEEDS="0 1 2" \
EXPERIMENTS="00_dfine_baseline 01_acq_only 02_small_matcher_only 03_scale_query_only 04_rsnds_only 05_acq_plus_small_matcher 06_scale_query_plus_rsnds 07_full_acds_dfine" \
EPOCHS=220 \
TRAIN_TOTAL_BATCH_SIZE=64 \
VAL_TOTAL_BATCH_SIZE=128 \
VAL_INTERVAL=1 \
bash /data/libaichuan/Projects/SOD/ACDS-DETR/acds_dfine_ablation/run_ablation_visdrone.sh
```

## Evaluate the Full Ablation Set

After checkpoints are produced, evaluate all runs with the existing D-FINE
test-only path:

```bash
GPUS=1 CUDA_DEVICES=0 bash /data/libaichuan/Projects/SOD/ACDS-DETR/acds_dfine_ablation/eval_ablation_visdrone.sh
```

## Paper-Level Evidence Checklist

This ablation set is necessary but not sufficient by itself. For a convincing
paper table, run at least three seeds, include the unmodified D-FINE VisDrone
baseline, report `APs` improvements, add FPS/params/FLOPs, and show qualitative
examples on dense small objects.
