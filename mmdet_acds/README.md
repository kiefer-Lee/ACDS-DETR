# ACDS-DETR MMDetection Extension

This package contains the MMDetection 3.x implementation of ACDS-DETR.

Active components:

- `models/`: ACDS-Deformable-DETR detector, ACQ loss, R-SNDS attention, and small-object assigner.
- `configs/`: full model, baseline, and ablation configs.
- `datasets/`: VisDrone metainfo.
- `evaluation/`: dense-small-object metric utilities.
- `tools/`: VisDrone-to-COCO conversion helper.

Minimum intended stack:

- PyTorch with CUDA support
- `mmengine>=0.7`
- `mmcv>=2.0`
- `mmdet>=3.0`
- optional but recommended: `openmim`

See `EXPERIMENTS.md` for the complete ablation command set.

