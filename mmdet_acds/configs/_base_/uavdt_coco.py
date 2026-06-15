custom_imports = dict(imports=["mmdet_acds"], allow_failed_imports=False)

data_root = "/root/blockdata/Datasets/UAVDT/"
data_root = data_root if data_root.endswith(("/", "\\")) else data_root + "/"
train_ann_file = "annotations/train.json"
val_ann_file = "annotations/val.json"
train_img_prefix = ""
val_img_prefix = ""
uavdt_classes = ("car", "bus", "truck")
metainfo = dict(classes=uavdt_classes)

img_size = 1024
max_size = 1600
train_scales = [(max_size, img_size)]

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
    batch_size=3,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_img_prefix),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=False, min_size=1),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=3,
    num_workers=12,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=val_ann_file,
        data_prefix=dict(img=val_img_prefix),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + val_ann_file,
    metric="bbox",
    proposal_nums=(100, 300, 500),
)
test_evaluator = val_evaluator
