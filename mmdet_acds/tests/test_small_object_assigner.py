import torch

from mmdet_acds.models.small_object_assigner import SmallObjectHungarianAssigner


def test_small_object_assigner_gain():
    assigner = SmallObjectHungarianAssigner(small_object_cost_gain=0.5, small_area_thr=1024)
    cost_bbox = torch.ones(2, 2)
    cost_iou = torch.ones(2, 2) * 2
    gt_bboxes = torch.tensor([[0.0, 0.0, 16.0, 16.0], [0.0, 0.0, 80.0, 80.0]])
    bbox_gain, iou_gain = assigner.apply_small_object_gain(cost_bbox, cost_iou, gt_bboxes)
    assert torch.allclose(bbox_gain[:, 0], torch.full((2,), 1.5))
    assert torch.allclose(iou_gain[:, 0], torch.full((2,), 3.0))
    assert torch.allclose(bbox_gain[:, 1], torch.ones(2))
    assert torch.allclose(iou_gain[:, 1], torch.full((2,), 2.0))

