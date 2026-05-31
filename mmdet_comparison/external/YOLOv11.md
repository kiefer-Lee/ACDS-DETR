# YOLOv11 Comparison Entry

YOLOv11/YOLO11 is not supported as an official OpenMMLab MMDetection/MMYOLO
model in the stable MMYOLO documentation. Use the official Ultralytics
implementation for this comparison rather than registering a fake MMDetection
model type.

Comparison contract:

- Convert VisDrone annotations to the Ultralytics YOLO dataset format if not
  already available:

  ```bash
  python mmdet_comparison/tools/coco_to_ultralytics_yolo.py \
    --data-root /root/blockdata/Datasets/VisDrone \
    --output-root /root/blockdata/Datasets/VisDrone-ultralytics \
    --copy-images
  ```

- Train with the official Ultralytics package, for example:

  ```bash
  yolo detect train model=yolo11s.pt \
    data=/root/blockdata/Datasets/VisDrone-ultralytics/visdrone.yaml \
    imgsz=1024 epochs=150 batch=3
  ```

- Preserve the same train/val splits used by the MMDetection configs.
- Use the same 10 VisDrone categories and the same seed schedule.
- Set validation image size to match the comparison input scale as closely as
  the Ultralytics trainer allows.
- Export COCO-style predictions or evaluate with COCO metrics so results are
  comparable with `mmdet_acds` and `mmdet_comparison` runs.
