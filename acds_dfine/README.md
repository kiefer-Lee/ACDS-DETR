# ACDS-D-FINE Migration

This folder contains the first implementation layer for migrating ACDS-DETR
from a Deformable DETR baseline to a D-FINE / DEIM-D-FINE baseline.

The intended engineering layout is hybrid:

- keep MMDetection-side dataset conversion, experiment records, and dense small
  object metrics from `mmdet_acds/`;
- train and validate the detector with the official D-FINE framework so the
  baseline, pretrained weights, FDR, and GO-LSD losses stay faithful;
- import `acds_dfine` from a patched D-FINE criterion/decoder to add ACDS
  modules without replacing D-FINE's original losses.

## Implemented Migration Pieces

- `DFineACDSCriterion`: an auxiliary criterion for D-FINE-shaped outputs
  (`pred_logits`, `pred_boxes`, optional `aux_outputs`) that adds:
  - assignment-aware collision decoupling loss;
  - small-object sensitive Hungarian matching;
  - query collision rate logging.
- `ScaleAwareQueryBias`: a small scale-group query bias that can be inserted
  after D-FINE query initialization.
- `rsnds_gamma_for_dfine` and `apply_rsnds_to_sampling_offsets`: patch helpers
  for decoder cross-attention offset scaling.
- config templates under `configs/` for VisDrone/UAVDT-style COCO datasets.

## Where To Patch D-FINE

The sibling `D-FINE/` checkout in this workspace has already been patched for
the first ACDS-D-FINE path:

1. The official D-FINE criterion instantiates `DFineACDSCriterion` and adds its
   returned `loss_acq` to the loss dictionary.  The minimal hook is:

   ```python
   from acds_dfine import add_acds_losses, build_acds_criterion

   self.acds_criterion = build_acds_criterion(cfg.get("ACDS", {}))

   # inside the existing criterion forward/loss call, after D-FINE losses:
   losses = add_acds_losses(losses, outputs, targets, self.acds_criterion)
   ```

2. The D-FINE matcher supports a configurable small-object bbox/GIoU cost gain.
3. The D-FINE decoder computes R-SNDS `gamma` from the previous layer's query,
   boxes, and logits, then applies it to deformable attention sampling offsets
   before sampling locations are formed.
4. Object queries can be wrapped with `ScaleAwareQueryBias` before the first
   decoder layer.
4. Keep D-FINE's FDR and GO-LSD losses enabled; ACDS is an auxiliary
   small-object specialization, not a replacement for the baseline regression
   task.

## Recommended Experiment Order

1. D-FINE-S baseline on VisDrone/UAVDT/TinyPerson COCO annotations.
2. D-FINE-S + `loss_acq` + small-object matching.
3. Add R-SNDS decoder offset scaling.
4. Add scale-aware query bias.
5. Add P2/high-resolution feature path after the loss/query path is stable.
6. Repeat the best recipe with DEIM-D-FINE.

Report `AP`, `AP50`, `AP75`, `APs`, dense/crowded subset AP, FPS, parameter
count, and `query_collision_rate`.

## Official D-FINE Entrypoints

Use the scripts in `ACDS-DETR/scripts/` for this migration path:

```bash
bash scripts/train_acds_dfine_visdrone.sh
bash scripts/test_acds_dfine_visdrone.sh CHECKPOINT=/path/to/model.pth
```

Both scripts call D-FINE's official `train.py` through `torchrun`; the older
MMDetection/MIM scripts remain only for the original ACDS-DETR experiments.
