custom_imports = dict(imports=["mmdet_acds"], allow_failed_imports=False)

data_root = "/data/libaichuan/Projects/SOD/Datasets/VisDrone/"
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

img_size = 1088
max_size = 1856
train_scales = [(max_size, s) for s in [960, 1024, 1088, 1152, 1216]]

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomChoice",
        transforms=[
            [dict(type="RandomChoiceResize", scales=train_scales, keep_ratio=True)],
            [
                dict(type="RandomChoiceResize", scales=train_scales, keep_ratio=True),
                dict(type="RandomCrop", crop_type="relative_range", crop_size=(0.8, 0.8), allow_negative_crop=False),
                dict(type="RandomChoiceResize", scales=train_scales, keep_ratio=True),
            ],
        ],
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(max_size, img_size), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/train.json",
        data_prefix=dict(img="VisDrone2019-DET-train/VisDrone2019-DET-train/images/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
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
    type="CocoMetric",
    ann_file=data_root + "annotations/val.json",
    metric="bbox",
    iou_thrs=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
    proposal_nums=(100, 300, 500),
)
test_evaluator = val_evaluator

