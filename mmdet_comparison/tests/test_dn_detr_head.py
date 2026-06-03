import torch
from torch import nn

from mmdet_comparison.models.dn_detr_head import DNDETRHead, format_dn_losses, split_matching_dn_outputs


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


class ConstantLoss(nn.Module):
    loss_weight = 1.0

    def forward(self, pred, target, avg_factor=1):
        return pred.sum() * 0.0 + target.float().sum() * 0.0 + 1.0


def test_dn_loss_includes_decoder_auxiliary_layers():
    head = DNDETRHead()
    head.loss_cls = ConstantLoss()
    head.loss_bbox = ConstantLoss()
    head.loss_iou = ConstantLoss()
    dn_meta = {
        "target_labels": torch.zeros(2, 3, dtype=torch.long),
        "target_bboxes": torch.full((2, 3, 4), 0.5),
        "target_weights": torch.ones(2, 3),
    }
    cls_scores = torch.randn(3, 2, 3, 10)
    bbox_preds = torch.full((3, 2, 3, 4), 0.5)

    losses = head.loss_dn_by_feat(cls_scores, bbox_preds, dn_meta)

    assert {"loss_dn_cls", "loss_dn_bbox", "loss_dn_iou"}.issubset(losses)
    assert {"d0.loss_dn_cls", "d0.loss_dn_bbox", "d0.loss_dn_iou"}.issubset(losses)
    assert {"d1.loss_dn_cls", "d1.loss_dn_bbox", "d1.loss_dn_iou"}.issubset(losses)
