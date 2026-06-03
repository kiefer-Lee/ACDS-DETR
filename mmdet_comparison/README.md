# MMDetection Comparison Methods

This package contains comparison baselines for the ACDS-DETR experiments.

Implemented methods:

- `DAB-DETR`: uses MMDetection's official DAB-DETR detector and head with the
  same VisDrone data, schedule, runtime, and evaluator settings as ACDS-DETR.
- `DN-DETR`: extends DAB-DETR with DN-style denoising label/box queries and an
  auxiliary denoising reconstruction loss. It does not use DINO two-stage
  proposals, mixed query selection, or deformable attention.
- `DINO-DETR`: uses MMDetection's official DINO detector/head with a ResNet-50
  backbone, 4-scale deformable attention features, two-stage proposals, and
  denoising queries.
- `Faster R-CNN`: uses MMDetection's official Faster R-CNN detector.
- `FCOS`: uses MMDetection's official FCOS detector.
- `YOLOv8`: uses MMYOLO's official YOLOv8 config through the OpenMMLab config
  package mechanism. Install `mmyolo` before running this config.

External comparison entries:

- `external/RT_DETR.md`: RT-DETR is not an official MMDetection mainline
  detector; use the official RT-DETR implementation and keep the comparison
  contract aligned.
- `external/YOLOv11.md`: YOLOv11 is not an official stable MMYOLO model; use
  the official Ultralytics implementation and evaluate with comparable COCO
  metrics.

Run from `ACDS-DETR` with `PYTHONPATH` including the repository root:

```bash
export PYTHONPATH=$PWD:$PYTHONPATH

mim train mmdet mmdet_comparison/configs/dab_detr_r50_visdrone.py \
  --work-dir work_dirs/comparison/dab_detr_r50_visdrone/seed_0

mim train mmdet mmdet_comparison/configs/dn_detr_r50_visdrone.py \
  --work-dir work_dirs/comparison/dn_detr_r50_visdrone/seed_0

mim train mmdet mmdet_comparison/configs/dino_detr_r50_visdrone.py \
  --work-dir work_dirs/comparison/dino_detr_r50_visdrone/seed_0

mim train mmdet mmdet_comparison/configs/faster_rcnn_r50_fpn_visdrone.py \
  --work-dir work_dirs/comparison/faster_rcnn_r50_fpn_visdrone/seed_0

mim train mmdet mmdet_comparison/configs/fcos_r50_fpn_visdrone.py \
  --work-dir work_dirs/comparison/fcos_r50_fpn_visdrone/seed_0

mim train mmyolo mmdet_comparison/configs/yolov8_s_visdrone.py \
  --work-dir work_dirs/comparison/yolov8_s_visdrone/seed_0
```

Use the same `--cfg-options` data-root overrides documented in
`mmdet_acds/EXPERIMENTS.md` when running outside the default VisDrone path.

DINO-DETR full example:

```bash
cd /data/libaichuan/Projects/SOD/ACDS-DETR
export PYTHONPATH=$PWD:$PYTHONPATH

DATA_ROOT=/root/blockdata/Datasets/VisDrone
GPUS=2
COMMON_OPTS="data_root=$DATA_ROOT \
train_dataloader.dataset.data_root=$DATA_ROOT \
val_dataloader.dataset.data_root=$DATA_ROOT \
test_dataloader.dataset.data_root=$DATA_ROOT \
val_evaluator.ann_file=$DATA_ROOT/annotations/val.json \
test_evaluator.ann_file=$DATA_ROOT/annotations/val.json"

CUDA_VISIBLE_DEVICES=0,1 mim train mmdet \
  mmdet_comparison/configs/dino_detr_r50_visdrone.py \
  --launcher pytorch \
  --gpus $GPUS \
  --work-dir work_dirs/comparison/dino_detr_r50_visdrone/seed_0 \
  --cfg-options $COMMON_OPTS randomness.seed=0 randomness.deterministic=False
```
