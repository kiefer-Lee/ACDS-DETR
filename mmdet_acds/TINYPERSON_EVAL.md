# TinyPerson Evaluation

This project keeps the original COCO-style test metrics and adds a shared
offline TinyPerson-style evaluator. Existing checkpoints do not need to be
retrained.

## 1. Export detection json

MMDetection-based models, including ACDS-DETR, DINO, Deformable DETR, and the
YOLO configs under this repo, now export predictions by default when using the
TinyPerson test scripts.

```bash
CHECKPOINT=/path/to/checkpoint.pth \
bash scripts/test_tinyperson.sh
```

The prediction file is written as:

```text
work_dirs/.../test/predictions.bbox.json
```

YOLO-style comparison configs use the same convention:

```bash
CHECKPOINT=/path/to/checkpoint.pth \
bash scripts/test_tinyperson_yolo.sh
```

RT-DETR saves a COCO-format detection file after test:

```bash
cd /data/libaichuan/Projects/SOD/RT-DETR/rtdetr_pytorch
CHECKPOINT=/path/to/checkpoint.pth \
bash scripts/test_tinyperson.sh
```

The prediction file is written as:

```text
output/.../test/bbox_predictions.json
```

RT-DETR's TinyPerson script remaps categories to `0/1`. When evaluating with
`CATEGORY_MODE=keep`, use the generated annotation file:

```text
/root/blockdata/Datasets/tiny_set/annotations/rtdetr/tiny_set_test_rtdetr.json
```

Alternatively, use `CATEGORY_MODE=merge` to evaluate both categories as one
person class.

## 2. Run TinyPerson-style evaluation

Use the same evaluator for every model:

```bash
cd /data/libaichuan/Projects/SOD/ACDS-DETR
DET=work_dirs/tinyperson/sw640_sh512/acds_detr_r50_visdrone/seed_0/test/predictions.bbox.json \
bash scripts/eval_tinyperson.sh
```

The evaluator reports:

- `APtiny`, `APtiny_25`, `APtiny_50`, `APtiny_75`
- `APtiny1_50`, `APtiny2_50`, `APtiny3_50`
- `APsmall_50`
- matching AR values and `Miss@50 = 1 - AR50`

The default size ranges follow the common TinyPerson convention using bbox
area:

```text
tiny  = [2^2, 20^2]
tiny1 = [2^2, 8^2]
tiny2 = [8^2, 12^2]
tiny3 = [12^2, 20^2]
small = [20^2, 32^2]
```

## 3. Useful options

Write machine-readable metrics:

```bash
DET=/path/to/predictions.bbox.json \
SUMMARY_JSON=/path/to/tinyperson_metrics.json \
bash scripts/eval_tinyperson.sh
```

Evaluate all TinyPerson categories as one person class:

```bash
DET=/path/to/predictions.bbox.json \
CATEGORY_MODE=merge \
bash scripts/eval_tinyperson.sh
```

Change `maxDets`:

```bash
DET=/path/to/predictions.bbox.json \
MAX_DETS="100 300 500 1000" \
bash scripts/eval_tinyperson.sh
```

For fair tables, use the same `CATEGORY_MODE` and `MAX_DETS` for every model.
