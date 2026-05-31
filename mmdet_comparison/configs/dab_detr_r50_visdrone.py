_base_ = [
    "../../mmdet_acds/configs/_base_/visdrone_coco.py",
    "../../mmdet_acds/configs/_base_/schedule_acds.py",
    "../../mmdet_acds/configs/_base_/runtime.py",
]

custom_imports = dict(imports=["mmdet_acds", "mmdet_comparison"], allow_failed_imports=False)

embed_dims = 256
num_classes = 10
num_queries = 300

model = dict(
    type="DABDETR",
    num_queries=num_queries,
    with_random_refpoints=False,
    num_patterns=0,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1,
    ),
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[2048],
        kernel_size=1,
        out_channels=embed_dims,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=1,
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=embed_dims, num_heads=8, dropout=0.1, batch_first=True),
            ffn_cfg=dict(embed_dims=embed_dims, feedforward_channels=2048, ffn_drop=0.1),
        ),
    ),
    decoder=dict(
        num_layers=6,
        query_dim=4,
        query_scale_type="cond_elewise",
        with_modulated_hw_attn=True,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=embed_dims,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                batch_first=True,
                cross_attn=False,
            ),
            cross_attn_cfg=dict(
                embed_dims=embed_dims,
                num_heads=8,
                attn_drop=0.0,
                proj_drop=0.0,
                batch_first=True,
                cross_attn=True,
            ),
            ffn_cfg=dict(
                embed_dims=embed_dims,
                feedforward_channels=2048,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type="PReLU"),
            ),
        ),
    ),
    positional_encoding=dict(num_feats=128, temperature=20, normalize=True),
    bbox_head=dict(
        type="DABDETRHead",
        num_classes=num_classes,
        embed_dims=embed_dims,
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ],
        )
    ),
    test_cfg=dict(max_per_img=500),
)
