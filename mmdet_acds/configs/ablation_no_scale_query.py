_base_ = ["./acds_detr_r50_visdrone.py"]

model = dict(
    scale_aware_query=dict(enabled=False, groups=4, strength=0.0),
)

