_base_ = ["./ablation_baseline_deformable.py"]

model = dict(
    bbox_head=dict(
        acq_apply_last_n_layers=2,
        acq_loss=dict(
            enabled=True,
            loss_weight=0.03,
            small_area_thr=1024,
            topk_unmatched=30,
            delta=0.03,
            sigma=0.06,
            min_score=0.40,
            use_sigmoid_cls=True,
        ),
    ),
)

