_base_ = ["./acds_detr_r50_visdrone.py"]

model = dict(
    encoder=dict(num_layers=3),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(rsnds=dict(enabled=False)))),
)

