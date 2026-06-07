_base_ = ["mmyolo::yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py"]

custom_imports = dict(
    imports=["mmdet_comparison.models"],
    allow_failed_imports=False,
)

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
data_root = "D:/PythonProjects/SOD/Datasets/VisDrone/"
img_scale = (640, 640)
affine_scale = 0.5
max_aspect_ratio = 100
max_epochs = 200
close_mosaic_epochs = 10
train_batch_size_per_gpu = 16
train_num_workers = 8
val_batch_size_per_gpu = 1
val_num_workers = 2

deepen_factor = 0.33
widen_factor = 0.25
strides = [4, 8, 16]
last_stage_out_channels = 512
norm_cfg = dict(type="BN", momentum=0.03, eps=0.001)
act_cfg = dict(type="SiLU", inplace=True)

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

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="YOLOv5KeepRatioResize", scale=img_scale),
    dict(type="LetterResize", scale=img_scale, allow_scale_up=False, pad_val=dict(img=114)),
    dict(type="LoadAnnotations", with_bbox=True, _scope_="mmdet"),
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor", "pad_param"),
    ),
]

model = dict(
    backbone=dict(
        _delete_=True,
        type="SkyYOLOBackbone",
        arch="P4",
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(1, 2, 3),
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
    ),
    neck=dict(
        _delete_=True,
        type="LightBiFPN",
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[128, 256, last_stage_out_channels],
        out_channels=[128, 256, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
    ),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes,
            in_channels=[128, 256, last_stage_out_channels],
            widen_factor=widen_factor,
            featmap_strides=strides,
        ),
        prior_generator=dict(strides=strides),
        loss_bbox=dict(
            _delete_=True,
            type="WiseIoULoss",
            version="v3",
            bbox_format="xyxy",
            reduction="sum",
            loss_weight=7.5,
            alpha=1.7,
            delta=2.7,
            momentum=0.01,
        ),
    ),
    train_cfg=dict(assigner=dict(num_classes=num_classes, use_ciou=False)),
    test_cfg=dict(max_per_img=500),
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
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
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        type="YOLOv5CocoDataset",
        data_root=data_root,
        ann_file="annotations/val.json",
        data_prefix=dict(img="VisDrone2019-DET-val/VisDrone2019-DET-val/images/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
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

default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best="auto"),
)

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), 1)],
)
