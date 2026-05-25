_base_ = ["./acds_detr_r50_visdrone.py"]

model = dict(
    bbox_head=dict(acq_loss=dict(enabled=False, loss_weight=0.0)),
)
