train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=150, val_interval=5)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="AmpOptimWrapper",
    loss_scale=dict(init_scale=512),
    optimizer=dict(type="AdamW", lr=2.5e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.05, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.01)}),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type="MultiStepLR", by_epoch=True, milestones=[105, 135], gamma=0.1),
]

custom_hooks = [
    dict(type="EMAHook", momentum=0.0002, priority=49),
    dict(type="MetricCurveHook", priority=90),
]

