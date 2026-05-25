_base_ = ["./acds_detr_r50_visdrone.py"]

model = dict(
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(rsnds=dict(enabled=False)))),
)
