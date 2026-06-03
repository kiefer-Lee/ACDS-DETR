_base_ = ["./dab_detr_r50_visdrone.py"]

model = dict(
    type="DNDETR",
    dn_cfg=dict(
        num_groups=5,
        label_noise_scale=0.2,
        box_noise_scale=0.4,
        max_dn_queries=300,
    ),
    bbox_head=dict(type="DNDETRHead"),
)

train_dataloader = dict(
    batch_size=6,
    batch_sampler=dict(
        type="DensityAwareBatchSampler",
        dense_threshold=150,
        dense_batch_size=1,
    ),
)
