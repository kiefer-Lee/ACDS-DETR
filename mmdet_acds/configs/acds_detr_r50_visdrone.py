_base_ = ["_base_/visdrone_coco.py", "_base_/schedule_acds.py", "_base_/runtime.py"]

embed_dims = 256
num_classes = 10
num_queries = 1000
num_feature_levels = 5
rsnds_cfg = dict(enabled=True, beta=1.0, gamma_base=1.0, gamma_min=0.35, gamma_max=1.25, reliability="cls_conf")

model = dict(
    type="ACDSDeformableDETR",
    num_queries=num_queries,
    num_feature_levels=num_feature_levels,
    use_p2=True,
    encoder_level_indices=[1, 2, 3, 4],
    scale_aware_query=dict(enabled=True, groups=4, strength=0.35),
    with_box_refine=True,
    as_two_stage=False,
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
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[256, 512, 1024, 2048],
        kernel_size=1,
        out_channels=embed_dims,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=num_feature_levels,
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=embed_dims, num_levels=4, dropout=0.1, batch_first=True),
            ffn_cfg=dict(embed_dims=embed_dims, feedforward_channels=1024, ffn_drop=0.1),
        ),
    ),
    decoder=dict(
        type="ACDSDeformableDetrTransformerDecoder",
        num_layers=6,
        return_intermediate=True,
        post_norm_cfg=None,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=embed_dims, num_heads=8, dropout=0.1, batch_first=True),
            cross_attn_cfg=dict(
                type="RSNDSMultiScaleDeformableAttention",
                embed_dims=embed_dims,
                num_levels=num_feature_levels,
                dropout=0.1,
                batch_first=True,
                rsnds=rsnds_cfg,
            ),
            ffn_cfg=dict(embed_dims=embed_dims, feedforward_channels=1024, ffn_drop=0.1),
        ),
    ),
    positional_encoding=dict(num_feats=128, normalize=True, offset=-0.5),
    bbox_head=dict(
        type="ACDSDeformableDETRHead",
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        acq_apply_last_n_layers=2,
        acq_loss=dict(
            enabled=True,
            loss_weight=0.03,
            small_area_thr=1024,
            topk_unmatched=30,
            delta=0.03,
            sigma=0.06,
            min_score=0.40,
            use_sigmoid_cls=True,
        ),
        small_object_loss_gain=0.35,
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=7.0),
        loss_iou=dict(type="GIoULoss", loss_weight=3.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type="SmallObjectHungarianAssigner",
            small_object_cost_gain=0.20,
            small_area_thr=1024,
            match_costs=[
                dict(type="FocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=7.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=3.0),
            ],
        )
    ),
    test_cfg=dict(max_per_img=500),
)
