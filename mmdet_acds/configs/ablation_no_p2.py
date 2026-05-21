_base_ = ["./acds_detr_r50_visdrone.py"]

model = dict(
    num_feature_levels=4,
    use_p2=False,
    encoder_level_indices=None,
    neck=dict(in_channels=[512, 1024, 2048], num_outs=4),
    backbone=dict(out_indices=(1, 2, 3)),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=4))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=4))),
)
