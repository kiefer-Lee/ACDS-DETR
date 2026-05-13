# Config Set

All YAML files in this directory are complete, standalone configs. There is no `_base_` inheritance.

## Main configs

- `paper_full_small_object.yaml`: paper-oriented full model config. Uses the mature deformable encoder/decoder stack, P2/P3/P4/P5/P6, 1000 queries, ACQ, R-SNDS, EMA and AMP.
- `default.yaml`: single-stage high-performance config for a 24GB RTX 4090. Uses 1024 input, P2/P3/P4/P5/P6, 1000 queries, ACQ, R-SNDS, EMA, AMP, conservative small-object crop.
- `stage1_localization_bootstrap.yaml`: first-stage localization warm-up. ACQ and R-SNDS are disabled so DETR learns reliable same-class IoU before small-object mechanisms are enabled.
- `stage2_acds_full_finetune.yaml`: second-stage full ACDS-DETR fine-tune. Resume from `stage1_localization_bootstrap/best_map.pth`.

## Ablations

- `ablation_baseline_deformable.yaml`: Deformable-DETR-like baseline, 300 queries, no P2, no ACQ, no R-SNDS.
- `ablation_no_p2.yaml`: full method without stride-4 P2.
- `ablation_no_acq.yaml`: full method without assignment-aware collision loss.
- `ablation_no_rsnds.yaml`: full method without reliability-guided scale-normalized sampling.

## Debug

- `smoke_visdrone_mini.yaml`: tiny CPU/GPU smoke test.
