_base_ = ["./acds_detr_r50_visdrone.py"]

model = dict(
    train_cfg=dict(assigner=dict(small_object_cost_gain=0.0)),
)

