_base_ = ["./acds_detr_r50_visdrone.py"]

model = dict(
    num_queries=300,
    num_feature_levels=4,
    use_p2=False,
    encoder_level_indices=None,
    scale_aware_query=dict(enabled=False, groups=4, strength=0.0),
    neck=dict(in_channels=[512, 1024, 2048], num_outs=4),
    backbone=dict(out_indices=(1, 2, 3)),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=4))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=4, rsnds=dict(enabled=False)))),
    bbox_head=dict(
        acq_loss=dict(enabled=False, loss_weight=0.0),
        small_object_loss_gain=0.0,
    ),
    train_cfg=dict(assigner=dict(small_object_cost_gain=0.0)),
)
