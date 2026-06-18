# ACDS-DETR

This workspace now uses the MMDetection implementation as the active code path.

- Active implementation: `mmdet_acds/`
- D-FINE migration layer: `acds_dfine/`
- Archived legacy prototype: `Deprecated/`
- Experiment commands: `mmdet_acds/EXPERIMENTS.md`

The legacy standalone training framework, YAML configs, and old scripts have
been moved under `Deprecated/` so ablation experiments can be run from one
MMDetection-compatible entry point.

`acds_dfine/` contains the first ACDS-D-FINE migration layer: portable
small-object matching, ACQ loss integration, R-SNDS patch helpers, D-FINE custom
dataset templates, and tests for D-FINE-shaped DETR outputs.

