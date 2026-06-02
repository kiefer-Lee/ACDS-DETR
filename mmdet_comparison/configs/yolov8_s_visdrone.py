_base_ = ["mmyolo::yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco.py"]

num_classes = 10
visdrone_classes = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
metainfo = dict(classes=visdrone_classes)
data_root = "/root/blockdata/Datasets/VisDrone/"

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
        train_cfg=dict(assigner=dict(num_classes=num_classes)),
    ),
    test_cfg=dict(max_per_img=500),
)

train_dataloader = dict(
    batch_size=3,
    num_workers=12,
    dataset=dict(
        type="YOLOv5CocoDataset",
        data_root=data_root,
        ann_file="annotations/train.json",
        data_prefix=dict(img="VisDrone2019-DET-train/VisDrone2019-DET-train/images/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
    ),
)

val_dataloader = dict(
    batch_size=3,
    num_workers=12,
    dataset=dict(
        type="YOLOv5CocoDataset",
        data_root=data_root,
        ann_file="annotations/val.json",
        data_prefix=dict(img="VisDrone2019-DET-val/VisDrone2019-DET-val/images/"),
        metainfo=metainfo,
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="mmdet.CocoMetric",
    ann_file=data_root + "annotations/val.json",
    metric="bbox",
    iou_thrs=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    proposal_nums=(100, 300, 500),
)
test_evaluator = val_evaluator
