_base_ = ["./ablation_baseline_deformable.py"]

model = dict(
    num_feature_levels=5,
    use_p2=True,
    encoder_level_indices=[1, 2, 3, 4],
    neck=dict(in_channels=[256, 512, 1024, 2048], num_outs=5),
    backbone=dict(out_indices=(0, 1, 2, 3)),
    encoder=dict(layer_cfg=dict(self_attn_cfg=dict(num_levels=4))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=5))),
)

