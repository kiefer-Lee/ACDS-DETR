_base_ = ["./acds_detr_r50_visdrone.py"]

model = dict(
    encoder=dict(num_layers=3),
    bbox_head=dict(acq_loss=dict(enabled=False, loss_weight=0.0)),
)

