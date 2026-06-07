import torch

from mmdet_comparison.models.sky_yolo import WiseIoULoss


def test_wise_iou_loss_is_zero_for_identical_boxes():
    loss_fn = WiseIoULoss(version="v3", reduction="sum")
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [2.0, 2.0, 6.0, 6.0]])

    loss = loss_fn(boxes, boxes)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_wise_iou_loss_applies_positive_weights():
    loss_fn = WiseIoULoss(version="v1", reduction="sum", loss_weight=1.0)
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    target = torch.tensor([[0.0, 0.0, 10.0, 10.0], [5.0, 5.0, 15.0, 15.0]])
    weight = torch.tensor([[1.0], [0.0]])

    loss = loss_fn(pred, target, weight=weight)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
