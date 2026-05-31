_base_ = ["./dab_detr_r50_visdrone.py"]

model = dict(
    type="DNDETR",
    dn_cfg=dict(num_groups=5, label_noise_scale=0.2, box_noise_scale=0.4),
    bbox_head=dict(type="DNDETRHead"),
)
