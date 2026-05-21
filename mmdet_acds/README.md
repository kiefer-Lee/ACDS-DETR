# ACDS-DETR MMDetection Extension

This package contains the MMDetection 3.x migration of ACDS-DETR. It is an
additive plugin under `mmdet_acds/`; the legacy `models/`, `losses/`, and
`engine/` directories are intentionally left untouched.

Minimum intended stack:

- PyTorch with CUDA support for training
- `mmengine>=0.7`
- `mmcv>=2.0`
- `mmdet>=3.0`

Example:

```bash
python tools/train.py mmdet_acds/configs/acds_detr_r50_visdrone.py \
  --cfg-options train_dataloader.batch_size=1 train_cfg.max_iters=20
```

