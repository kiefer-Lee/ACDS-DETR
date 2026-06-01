"""DN-DETR head extensions for DAB-DETR comparisons."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .compat import MODELS

try:  # pragma: no cover - exercised in the target MMDetection environment.
    from mmdet.models.dense_heads import DABDETRHead as _MMDetDABDETRHead
except Exception:  # pragma: no cover
    _MMDetDABDETRHead = nn.Module


def split_matching_dn_outputs(
    all_layers_cls_scores: Tensor,
    all_layers_bbox_preds: Tensor,
    dn_meta: dict[str, Any] | None,
) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
    num_dn = int((dn_meta or {}).get("num_denoising_queries", 0))
    if num_dn <= 0:
        return all_layers_cls_scores, all_layers_bbox_preds, None, None
    dn_cls_scores = all_layers_cls_scores[:, :, :num_dn, :]
    dn_bbox_preds = all_layers_bbox_preds[:, :, :num_dn, :]
    matching_cls_scores = all_layers_cls_scores[:, :, num_dn:, :]
    matching_bbox_preds = all_layers_bbox_preds[:, :, num_dn:, :]
    matching_cls_scores = torch.nan_to_num(matching_cls_scores, nan=0.0, posinf=50.0, neginf=-50.0)
    matching_bbox_preds = torch.nan_to_num(matching_bbox_preds, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    dn_cls_scores = torch.nan_to_num(dn_cls_scores, nan=0.0, posinf=50.0, neginf=-50.0)
    dn_bbox_preds = torch.nan_to_num(dn_bbox_preds, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return matching_cls_scores, matching_bbox_preds, dn_cls_scores, dn_bbox_preds


def format_dn_losses(loss_cls: Tensor, loss_bbox: Tensor, loss_iou: Tensor) -> dict[str, Tensor]:
    return dict(loss_dn_cls=loss_cls, loss_dn_bbox=loss_bbox, loss_dn_iou=loss_iou)


def bbox_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = box_area(boxes1)[:, None] + box_area(boxes2) - inter
    iou = inter / union.clamp(min=1e-6)

    enclosing_lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclosing_rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclosing_wh = (enclosing_rb - enclosing_lt).clamp(min=0)
    enclosing_area = enclosing_wh[:, :, 0] * enclosing_wh[:, :, 1]
    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-6)


@MODELS.register_module()
class DNDETRHead(_MMDetDABDETRHead):
    """DAB-DETR head with an additional denoising reconstruction loss."""

    def loss(
        self,
        hidden_states: Tensor,
        references: list[Tensor],
        batch_data_samples: list[Any],
        dn_meta: dict[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        if _MMDetDABDETRHead is nn.Module:
            raise ImportError("mmdet is required to use DNDETRHead.loss")

        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_gt_instances.append(data_sample.gt_instances)
            batch_img_metas.append(data_sample.metainfo)
        outs = self(hidden_states, references)
        loss_inputs = outs + (batch_gt_instances, batch_img_metas)
        return self.loss_by_feat(*loss_inputs, dn_meta=dn_meta)

    def loss_by_feat(self, *args: Any, dn_meta: dict[str, Any] | None = None, **kwargs: Any) -> dict[str, Tensor]:
        if _MMDetDABDETRHead is nn.Module:
            raise ImportError("mmdet is required to use DNDETRHead.loss_by_feat")

        kwargs = dict(kwargs)
        all_layers_cls_scores = args[0] if args else kwargs.pop("all_layers_cls_scores")
        all_layers_bbox_preds = args[1] if len(args) > 1 else kwargs.pop("all_layers_bbox_preds")
        kwargs.pop("all_layers_cls_scores", None)
        kwargs.pop("all_layers_bbox_preds", None)
        rest_args = args[2:]
        matching_cls, matching_bbox, dn_cls, dn_bbox = split_matching_dn_outputs(
            all_layers_cls_scores, all_layers_bbox_preds, dn_meta
        )
        losses = super().loss_by_feat(matching_cls, matching_bbox, *rest_args, **kwargs)
        if dn_cls is not None and dn_bbox is not None:
            losses.update(self.loss_dn_by_feat(dn_cls, dn_bbox, dn_meta))
        return losses

    def loss_dn_by_feat(self, dn_cls_scores: Tensor, dn_bbox_preds: Tensor, dn_meta: dict[str, Any]) -> dict[str, Tensor]:
        target_weights = dn_meta["target_weights"].to(dn_cls_scores.device)
        target_labels = dn_meta["target_labels"].to(dn_cls_scores.device)
        target_bboxes = dn_meta["target_bboxes"].to(dn_bbox_preds.device)
        valid = target_weights > 0

        if not bool(valid.any()):
            zero = dn_bbox_preds.sum() * 0.0
            return format_dn_losses(zero, zero, zero)

        last_cls = dn_cls_scores[-1]
        last_bbox = dn_bbox_preds[-1]
        num_classes = int(last_cls.shape[-1])
        flat_valid = valid.reshape(-1)
        pred_logits = last_cls.reshape(-1, num_classes)[flat_valid]
        pred_boxes = last_bbox.reshape(-1, 4)[flat_valid]
        labels = target_labels.reshape(-1)[flat_valid].clamp(min=0, max=num_classes - 1)
        boxes = target_bboxes.reshape(-1, 4)[flat_valid]

        avg_factor = max(int(labels.numel()), 1)
        loss_cls = self.loss_cls(pred_logits, labels, avg_factor=avg_factor)
        loss_bbox = self.loss_bbox(pred_boxes, boxes, avg_factor=avg_factor)
        giou = generalized_box_iou(bbox_cxcywh_to_xyxy(pred_boxes), bbox_cxcywh_to_xyxy(boxes)).diag()
        loss_iou = (1.0 - giou).sum() / avg_factor
        loss_iou = loss_iou * float(getattr(self.loss_iou, "loss_weight", 1.0))
        return format_dn_losses(loss_cls, loss_bbox, loss_iou)
