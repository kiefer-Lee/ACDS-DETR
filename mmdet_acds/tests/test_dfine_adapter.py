import torch

from acds_dfine.core import (
    DFineACDSCriterion,
    ScaleAwareQueryBias,
    apply_rsnds_to_sampling_offsets,
    small_object_hungarian_indices,
)
from acds_dfine.hooks import add_acds_losses, build_acds_criterion


def test_dfine_acds_criterion_returns_loss_for_collision():
    cls_scores = torch.full((1, 3, 2), -8.0)
    cls_scores[0, :, 0] = torch.tensor([6.0, 5.0, -8.0])
    bbox_preds = torch.tensor([[[0.50, 0.50, 0.05, 0.05], [0.51, 0.50, 0.05, 0.05], [0.10, 0.10, 0.2, 0.2]]])
    outputs = {"pred_logits": cls_scores, "pred_boxes": bbox_preds}
    targets = [dict(boxes=torch.tensor([[48.0, 48.0, 52.0, 52.0]]), labels=torch.tensor([0]), img_shape=(100, 100))]

    criterion = DFineACDSCriterion(
        dict(
            acq_loss_weight=1.0,
            acq_min_score=0.1,
            acq_apply_last_n_layers=1,
            target_box_format="xyxy_abs",
            use_sigmoid_cls=True,
        )
    )
    losses = criterion(outputs, targets)

    assert losses["loss_acq"].item() > 0.0
    assert losses["query_collision_rate"].item() >= 1.0


def test_small_object_hungarian_indices_prefers_small_gt():
    cls_scores = torch.zeros(1, 2, 1)
    bbox_preds = torch.tensor([[[0.50, 0.50, 0.04, 0.04], [0.80, 0.80, 0.20, 0.20]]])
    targets = [
        dict(
            boxes=torch.tensor([[48.0, 48.0, 52.0, 52.0], [70.0, 70.0, 90.0, 90.0]]),
            labels=torch.tensor([0, 0]),
            img_shape=(100, 100),
        )
    ]

    indices = small_object_hungarian_indices(cls_scores, bbox_preds, targets)

    assert indices[0][0].tolist() == [0, 1]
    assert indices[0][1].tolist() == [0, 1]


def test_scale_aware_query_bias_shape_and_grad():
    query = torch.zeros(2, 5, 8, requires_grad=True)
    bias = ScaleAwareQueryBias(embed_dims=8, groups=4, strength=0.35)
    out = bias(query)
    out.sum().backward()

    assert out.shape == query.shape
    assert query.grad is not None
    assert bias.embedding.weight.grad is not None


def test_apply_rsnds_to_sampling_offsets_scales_offsets():
    offsets = torch.ones(1, 2, 1, 3, 4, 2)
    gamma = torch.tensor([[[0.5], [2.0]]])
    scaled = apply_rsnds_to_sampling_offsets(offsets, gamma)

    assert torch.allclose(scaled[0, 0], torch.full_like(scaled[0, 0], 0.5))
    assert torch.allclose(scaled[0, 1], torch.full_like(scaled[0, 1], 2.0))


def test_add_acds_losses_hook_preserves_existing_losses():
    outputs = {
        "pred_logits": torch.tensor([[[6.0, -8.0], [5.0, -8.0], [-8.0, -8.0]]]),
        "pred_boxes": torch.tensor([[[0.50, 0.50, 0.05, 0.05], [0.51, 0.50, 0.05, 0.05], [0.10, 0.10, 0.2, 0.2]]]),
    }
    targets = [dict(boxes=torch.tensor([[48.0, 48.0, 52.0, 52.0]]), labels=torch.tensor([0]), img_shape=(100, 100))]
    criterion = build_acds_criterion(dict(acq_loss_weight=1.0, acq_min_score=0.1))
    losses = {"loss_vfl": torch.tensor(1.0)}

    add_acds_losses(losses, outputs, targets, criterion)

    assert "loss_vfl" in losses
    assert losses["loss_acq"].item() > 0.0
