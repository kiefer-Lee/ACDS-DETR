_base_ = ["./ablation_baseline_deformable.py"]

model = dict(
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(
                rsnds=dict(
                    enabled=True,
                    beta=1.0,
                    gamma_base=1.0,
                    gamma_min=0.35,
                    gamma_max=1.25,
                    reliability="cls_conf",
                )
            )
        )
    ),
)

