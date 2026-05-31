import torch

from mmdet_comparison.models.dn_detr_head import format_dn_losses, split_matching_dn_outputs


def test_split_matching_dn_outputs():
    cls_scores = torch.randn(6, 2, 11, 10)
    bbox_preds = torch.rand(6, 2, 11, 4)
    meta = {"num_denoising_queries": 4}

    matching_cls, matching_bbox, dn_cls, dn_bbox = split_matching_dn_outputs(cls_scores, bbox_preds, meta)

    assert matching_cls.shape == (6, 2, 7, 10)
    assert matching_bbox.shape == (6, 2, 7, 4)
    assert dn_cls.shape == (6, 2, 4, 10)
    assert dn_bbox.shape == (6, 2, 4, 4)


def test_format_dn_losses_uses_expected_keys():
    zero = torch.tensor(0.0)
    losses = format_dn_losses(zero, zero, zero)

    assert set(losses) == {"loss_dn_cls", "loss_dn_bbox", "loss_dn_iou"}
