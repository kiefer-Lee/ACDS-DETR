# MMDetection Ablation Commands

Run commands from the `ACDS-DETR` directory.

```bash
cd /data/libaichuan/Projects/SOD/ACDS-DETR
export PYTHONPATH=$PWD:$PYTHONPATH
DATA_ROOT=/root/blockdata/Datasets/VisDrone
GPUS=2
```

If COCO-format annotations are missing, generate them first:

```bash
python mmdet_acds/tools/convert_visdrone_to_coco.py \
  --root $DATA_ROOT \
  --split train \
  --output $DATA_ROOT/annotations/train.json

python mmdet_acds/tools/convert_visdrone_to_coco.py \
  --root $DATA_ROOT \
  --split val \
  --output $DATA_ROOT/annotations/val.json
```

Common config overrides:

```bash
COMMON_OPTS="data_root=$DATA_ROOT \
train_dataloader.dataset.data_root=$DATA_ROOT \
val_dataloader.dataset.data_root=$DATA_ROOT \
test_dataloader.dataset.data_root=$DATA_ROOT \
val_evaluator.ann_file=$DATA_ROOT/annotations/val.json \
test_evaluator.ann_file=$DATA_ROOT/annotations/val.json"
```

## Main Ablation Set

These are the core paper ablations. Run each config with three seeds.

```bash
CONFIGS=(
  mmdet_acds/configs/ablation_baseline_deformable.py
  mmdet_acds/configs/ablation_p2_only.py
  mmdet_acds/configs/ablation_acq_only.py
  mmdet_acds/configs/ablation_rsnds_only.py
  mmdet_acds/configs/ablation_no_p2.py
  mmdet_acds/configs/ablation_no_acq.py
  mmdet_acds/configs/ablation_no_rsnds.py
  mmdet_acds/configs/ablation_no_scale_query.py
  mmdet_acds/configs/ablation_no_small_assigner.py
  mmdet_acds/configs/acds_detr_r50_visdrone.py
)

for CONFIG in "${CONFIGS[@]}"; do
  NAME=$(basename "$CONFIG" .py)
  for SEED in 0 1 2; do
    WORK_DIR=work_dirs/visdrone_ablation/${NAME}/seed_${SEED}
    CUDA_VISIBLE_DEVICES=0,1 mim train mmdet "$CONFIG" \
      --launcher pytorch \
      --gpus $GPUS \
      --work-dir "$WORK_DIR" \
      --cfg-options $COMMON_OPTS randomness.seed=$SEED randomness.deterministic=False
  done
done
```

If you run from a local MMDetection source checkout instead of OpenMIM:

```bash
MMDET_DIR=/data/libaichuan/Projects/mmdetection
CONFIG=mmdet_acds/configs/acds_detr_r50_visdrone.py
WORK_DIR=work_dirs/visdrone_ablation/acds_detr_r50_visdrone/seed_0

CUDA_VISIBLE_DEVICES=0,1 bash $MMDET_DIR/tools/dist_train.sh "$CONFIG" $GPUS \
  --work-dir "$WORK_DIR" \
  --cfg-options $COMMON_OPTS randomness.seed=0 randomness.deterministic=False
```

## Single-Config Test

Replace `CHECKPOINT` with the best checkpoint saved in the corresponding work
directory.

```bash
CONFIG=mmdet_acds/configs/acds_detr_r50_visdrone.py
CHECKPOINT=work_dirs/visdrone_ablation/acds_detr_r50_visdrone/seed_0/best_coco_bbox_mAP_epoch_*.pth

CUDA_VISIBLE_DEVICES=0 mim test mmdet "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --work-dir work_dirs/visdrone_ablation/acds_detr_r50_visdrone/seed_0/test \
  --cfg-options $COMMON_OPTS
```

## Recommended Comparison Methods

Use MMDetection official configs where available, keeping the same VisDrone
COCO annotations, training schedule, input scale, batch size, and evaluator
settings.

```bash
COMPARE_CONFIGS=(
  configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
  configs/retinanet/retinanet_r50_fpn_1x_coco.py
  configs/fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py
  configs/detr/deformable-detr_r50_16xb2-50e_coco.py
  configs/dino/dino-4scale_r50_8xb2-12e_coco.py
)
```

For these official baselines, override at least:

```bash
--cfg-options \
  train_dataloader.dataset.type=CocoDataset \
  train_dataloader.dataset.data_root=$DATA_ROOT \
  train_dataloader.dataset.ann_file=annotations/train.json \
  train_dataloader.dataset.data_prefix.img=VisDrone2019-DET-train/VisDrone2019-DET-train/images/ \
  val_dataloader.dataset.type=CocoDataset \
  val_dataloader.dataset.data_root=$DATA_ROOT \
  val_dataloader.dataset.ann_file=annotations/val.json \
  val_dataloader.dataset.data_prefix.img=VisDrone2019-DET-val/VisDrone2019-DET-val/images/ \
  test_dataloader.dataset.type=CocoDataset \
  test_dataloader.dataset.data_root=$DATA_ROOT \
  test_dataloader.dataset.ann_file=annotations/val.json \
  test_dataloader.dataset.data_prefix.img=VisDrone2019-DET-val/VisDrone2019-DET-val/images/ \
  val_evaluator.ann_file=$DATA_ROOT/annotations/val.json \
  test_evaluator.ann_file=$DATA_ROOT/annotations/val.json
```

## Reporting Order

Report `mean +- std` over three seeds for:

- `bbox_mAP`, `bbox_mAP_50`, `bbox_mAP_75`
- `bbox_mAP_s`, `bbox_mAP_m`, `bbox_mAP_l`
- `AR@100`, `AR@300`, `AR@500`
- params, FLOPs, FPS or latency, and GPU model

