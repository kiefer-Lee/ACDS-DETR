_base_ = ["./acds_detr_r50_visdrone.py"]

model = dict(
    encoder=dict(num_layers=3),
    scale_aware_query=dict(enabled=True, groups=4, strength=0.20),
    bbox_head=dict(
        acq_loss=dict(enabled=False, loss_weight=0.0),
        small_object_loss_gain=0.0,
    ),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(rsnds=dict(enabled=False)))),
    train_cfg=dict(assigner=dict(small_object_cost_gain=0.0)),
)

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=45, val_interval=5)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=4.0e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.05, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.0)}),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type="MultiStepLR", by_epoch=True, milestones=[34, 42], gamma=0.1),
]
custom_hooks = [dict(type="EMAHook", momentum=0.0003, priority=49)]
load_from = None
resume = False

