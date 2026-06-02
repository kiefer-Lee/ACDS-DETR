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
img_scale = (640, 640)
affine_scale = 0.5
max_aspect_ratio = 100
max_epochs = 500
close_mosaic_epochs = 10

pre_transform = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
]

last_transform = [
    dict(type="YOLOv5HSVRandomAug"),
    dict(type="mmdet.RandomFlip", prob=0.5),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "flip", "flip_direction"),
    ),
]

train_pipeline = [
    *pre_transform,
    dict(type="Mosaic", img_scale=img_scale, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type="YOLOv5RandomAffine",
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114),
    ),
    *last_transform,
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type="YOLOv5KeepRatioResize", scale=img_scale),
    dict(type="LetterResize", scale=img_scale, allow_scale_up=True, pad_val=dict(img=114.0)),
    dict(
        type="YOLOv5RandomAffine",
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border_val=(114, 114, 114),
    ),
    *last_transform,
]

model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes),
    ),
    train_cfg=dict(assigner=dict(num_classes=num_classes)),
    test_cfg=dict(max_per_img=500),
)

train_dataloader = dict(
    batch_size=3,
    num_workers=12,
    collate_fn=dict(_delete_=True, type="yolov5_collate", use_ms_training=False),
    dataset=dict(
        type="YOLOv5CocoDataset",
        data_root=data_root,
        ann_file="annotations/train.json",
        data_prefix=dict(img="VisDrone2019-DET-train/VisDrone2019-DET-train/images/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
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
    proposal_nums=(1, 10, 100),
)
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49,
    ),
    dict(
        type="mmdet.PipelineSwitchHook",
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2,
    ),
]
