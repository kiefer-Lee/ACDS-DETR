"""ACDS head extensions for MMDetection Deformable DETR."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .acq_loss import ACQLoss
from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .compat import MODELS
from .small_object_assigner import linear_sum_assignment_torch

try:  # pragma: no cover - requires mmdet.
    from mmdet.models.dense_heads import DeformableDETRHead as _MMDetDeformableDETRHead
except Exception:  # pragma: no cover
    _MMDetDeformableDETRHead = nn.Module


def _gt_to_normalized_cxcywh(gt: Any, img_meta: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    bboxes = gt.bboxes.to(device).float()
    labels = gt.labels.to(device).long()
    h, w = img_meta.get("img_shape", img_meta.get("batch_input_shape", (1, 1)))[:2]
    norm = bboxes.new_tensor([w, h, w, h]).clamp(min=1)
    xyxy = (bboxes / norm).clamp(0.0, 1.0)
    cxcywh = torch.stack(
        (
            (xyxy[:, 0] + xyxy[:, 2]) * 0.5,
            (xyxy[:, 1] + xyxy[:, 3]) * 0.5,
            xyxy[:, 2] - xyxy[:, 0],
            xyxy[:, 3] - xyxy[:, 1],
        ),
        dim=-1,
    )
    return cxcywh.clamp(0.0, 1.0), labels


def _class_prob(cls_scores: torch.Tensor, labels: torch.Tensor, use_sigmoid: bool) -> torch.Tensor:
    if use_sigmoid:
        return cls_scores.float().sigmoid()[:, labels]
    return cls_scores.float().softmax(-1)[:, labels]


@MODELS.register_module()
class ACDSDeformableDETRHead(_MMDetDeformableDETRHead):
    """Adds ACQ loss on top of mmdet's DeformableDETRHead."""

    def __init__(
        self,
        *args: Any,
        acq_loss: dict | None = None,
        acq_apply_last_n_layers: int = 1,
        small_object_loss_gain: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        cfg = dict(acq_loss or {})
        cfg.setdefault("use_sigmoid_cls", bool(getattr(getattr(self, "loss_cls", None), "use_sigmoid", True)))
        self.acq_apply_last_n_layers = int(acq_apply_last_n_layers)
        self.small_object_loss_gain = float(small_object_loss_gain)
        self.loss_acq = ACQLoss(**cfg)

    def _hungarian_indices(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        batch_gt_instances: list[Any],
        batch_img_metas: list[dict],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        indices = []
        for b, gt in enumerate(batch_gt_instances):
            gt_boxes, gt_labels = _gt_to_normalized_cxcywh(gt, batch_img_metas[b], bbox_preds.device)
            if gt_boxes.numel() == 0:
                empty = torch.empty(0, dtype=torch.long, device=bbox_preds.device)
                indices.append((empty, empty))
                continue
            use_sigmoid = bool(getattr(getattr(self, "loss_cls", None), "use_sigmoid", True))
            cost_class = -_class_prob(cls_scores[b], gt_labels, use_sigmoid)
            cost_bbox = torch.cdist(bbox_preds[b].float(), gt_boxes.float(), p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(bbox_preds[b].float()),
                box_cxcywh_to_xyxy(gt_boxes.float()),
            )
            cost = 2.0 * cost_class + 5.0 * cost_bbox + 2.0 * cost_giou
            src, tgt = linear_sum_assignment_torch(cost)
            indices.append((src.to(bbox_preds.device), tgt.to(bbox_preds.device)))
        return indices

    def loss_by_feat(self, *args: Any, **kwargs: Any) -> dict:
        if _MMDetDeformableDETRHead is nn.Module:
            raise ImportError("mmdet is required to use ACDSDeformableDETRHead.loss_by_feat")

        losses = super().loss_by_feat(*args, **kwargs)
        all_layers_cls_scores = args[0] if args else kwargs["all_layers_cls_scores"]
        all_layers_bbox_preds = args[1] if len(args) > 1 else kwargs["all_layers_bbox_preds"]
        batch_gt_instances = args[2] if len(args) > 2 else kwargs["batch_gt_instances"]
        batch_img_metas = args[3] if len(args) > 3 else kwargs["batch_img_metas"]
        refs = kwargs.get("all_layers_reference_points")

        n = max(1, min(self.acq_apply_last_n_layers, int(all_layers_cls_scores.shape[0])))
        acq_loss = all_layers_bbox_preds[-1].sum() * 0.0
        qcr_values = []
        for layer_idx in range(all_layers_cls_scores.shape[0] - n, all_layers_cls_scores.shape[0]):
            indices = self._hungarian_indices(
                all_layers_cls_scores[layer_idx],
                all_layers_bbox_preds[layer_idx],
                batch_gt_instances,
                batch_img_metas,
            )
            ref = refs[layer_idx] if refs is not None else None
            layer_acq, stats = self.loss_acq(
                all_layers_cls_scores[layer_idx],
                all_layers_bbox_preds[layer_idx],
                batch_gt_instances,
                indices=indices,
                reference_points=ref,
            )
            acq_loss = acq_loss + layer_acq
            qcr_values.append(stats["query_collision_rate"])
        losses["loss_acq"] = acq_loss / n
        losses["query_collision_rate"] = torch.stack(qcr_values).mean() if qcr_values else acq_loss.detach()
        return losses
