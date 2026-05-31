_base_ = [
    "../../mmdet_acds/configs/_base_/visdrone_coco.py",
    "../../mmdet_acds/configs/_base_/schedule_acds.py",
    "../../mmdet_acds/configs/_base_/runtime.py",
]

custom_imports = dict(imports=["mmdet_acds", "mmdet_comparison"], allow_failed_imports=False)

num_classes = 10

model = dict(
    type="FCOS",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=5,
        relu_before_extra_convs=True,
    ),
    bbox_head=dict(
        type="FCOSHead",
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="IoULoss", loss_weight=1.0),
        loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0),
    ),
    train_cfg=None,
    test_cfg=dict(nms_pre=1000, min_bbox_size=0, score_thr=0.05, nms=dict(type="nms", iou_threshold=0.6), max_per_img=500),
)
