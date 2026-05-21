_base_ = ["./acds_detr_r50_visdrone.py"]

model = dict(
    encoder=dict(num_layers=3),
    bbox_head=dict(
        acq_loss=dict(enabled=True, loss_weight=0.04),
        small_object_loss_gain=0.60,
    ),
    train_cfg=dict(assigner=dict(small_object_cost_gain=0.35)),
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=30, val_interval=5)
optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale=dict(init_scale=512),
    optimizer=dict(type="AdamW", lr=1.5e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.05, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.0666667)}),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type="MultiStepLR", by_epoch=True, milestones=[20, 27], gamma=0.1),
]
load_from = None
resume = False

