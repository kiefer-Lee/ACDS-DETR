# RT-DETR Comparison Entry

RT-DETR is not an official MMDetection detector in the OpenMMLab mainline.
Use the official RT-DETR implementation for this comparison and keep the
dataset split, input scale, evaluator, and seeds aligned with the MMDetection
experiments.

Recommended source:

- `lyuwenyu/RT-DETR`: official Paddle/PyTorch RT-DETR implementation.
- Hugging Face `RTDetrForObjectDetection`: official Transformers integration
  for fine-tuning and inference.

Comparison contract:

- Use the existing COCO-format VisDrone annotations:
  `annotations/train.json` and `annotations/val.json`.
- Use the same 10 VisDrone categories listed in
  `mmdet_acds/configs/_base_/visdrone_coco.py`.
- Resize policy should match the ACDS comparison input envelope:
  short/target side `1024`, max side `1600`.
- Report the same metrics as the MMDetection runs:
  `bbox_mAP`, `bbox_mAP_50`, `bbox_mAP_75`, `bbox_mAP_s/m/l`, and
  `AR@100/300/500`.
