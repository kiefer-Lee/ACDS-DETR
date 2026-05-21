import torch

from mmdet_acds.models.acq_loss import ACQLoss


def test_acq_loss_disabled_returns_zero_and_grad():
    cls_scores = torch.randn(1, 4, 10, requires_grad=True)
    bbox_preds = torch.rand(1, 4, 4, requires_grad=True)
    gt = [dict(bboxes=torch.tensor([[0.0, 0.0, 10.0, 10.0]]), labels=torch.tensor([0]), img_shape=(100, 100))]
    loss, stats = ACQLoss(enabled=False)(cls_scores, bbox_preds, gt)
    loss.backward()
    assert loss.item() == 0.0
    assert bbox_preds.grad is not None
    assert stats["query_collision_rate"].item() == 0.0


def test_acq_small_thr_same_coordinate_space():
    cls_scores = torch.full((1, 3, 10), -8.0)
    cls_scores[0, :, 0] = torch.tensor([6.0, 5.0, -8.0])
    bbox_preds = torch.tensor([[[0.50, 0.50, 0.05, 0.05], [0.51, 0.50, 0.05, 0.05], [0.10, 0.10, 0.2, 0.2]]])
    gt = [
        dict(
            bboxes=torch.tensor([[48.0, 48.0, 52.0, 52.0]]),
            labels=torch.tensor([0]),
            areas=torch.tensor([16.0]),
            img_shape=(100, 100),
        )
    ]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    loss, stats = ACQLoss(enabled=True, min_score=0.1, use_sigmoid_cls=True)(
        cls_scores, bbox_preds, gt, indices=indices
    )
    assert loss.item() > 0.0
    assert stats["query_collision_rate"].item() >= 1.0

