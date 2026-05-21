"""Assignment-aware collision decoupling loss."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from .box_ops import box_cxcywh_to_xyxy, box_iou
from .compat import MODELS


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _indices_from_assign_result(assign_result: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert mmdet AssignResult to DETR-style ``(src_idx, tgt_idx)``."""

    gt_inds = getattr(assign_result, "gt_inds", None)
    if gt_inds is None and isinstance(assign_result, dict):
        gt_inds = assign_result.get("gt_inds")
    if gt_inds is None:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)
    gt_inds = gt_inds.to(device)
    pos = torch.nonzero(gt_inds > 0, as_tuple=False).flatten()
    return pos, gt_inds[pos].long() - 1


@MODELS.register_module()
class ACQLoss(nn.Module):
    """Decouple high-confidence unmatched queries around matched small objects."""

    def __init__(
        self,
        enabled: bool = True,
        small_area_thr: float = 1024.0,
        topk_unmatched: int = 30,
        delta: float = 0.03,
        sigma: float = 0.06,
        min_score: float = 0.40,
        loss_weight: float = 1.0,
        use_sigmoid_cls: bool = True,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.small_area_thr = float(small_area_thr)
        self.topk_unmatched = int(topk_unmatched)
        self.delta = float(delta)
        self.sigma = float(sigma)
        self.min_score = float(min_score)
        self.loss_weight = float(loss_weight)
        self.use_sigmoid_cls = bool(use_sigmoid_cls)

    def forward(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor,
        batch_gt_instances: Sequence[Any],
        assign_results: Sequence[Any] | None = None,
        indices: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        reference_points: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = bbox_preds.device
        if not self.enabled or self.loss_weight == 0:
            return bbox_preds.sum() * 0.0, {"query_collision_rate": bbox_preds.new_tensor(0.0)}

        logits = cls_scores.float()
        boxes = bbox_preds.float().nan_to_num(0.5).clamp(0.0, 1.0)
        refs = reference_points if reference_points is not None else boxes[..., :2]
        refs = refs.float().nan_to_num(0.5).clamp(0.0, 1.0)
        probs = logits.sigmoid() if self.use_sigmoid_cls else logits.softmax(-1)[..., :-1]
        scores = probs.max(-1).values

        if indices is None:
            indices = [
                _indices_from_assign_result(res, device)
                for res in (assign_results or [None] * len(batch_gt_instances))
            ]

        total_loss = boxes.sum() * 0.0
        total_pairs = 0
        total_small = 0
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if tgt_idx.numel() == 0:
                continue
            gt = batch_gt_instances[b]
            areas = _get_field(gt, "areas", None)
            if areas is None:
                areas = _get_field(gt, "area", None)
            if areas is None:
                gt_bboxes = _get_field(gt, "bboxes")
                areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=0) * (
                    gt_bboxes[:, 3] - gt_bboxes[:, 1]
                ).clamp(min=0)
            img_shape = _get_field(gt, "img_shape", None) or _get_field(gt, "ori_shape", None)
            if img_shape is None:
                img_shape = _get_field(gt, "size", None)
            if img_shape is None:
                size_prod = areas.new_tensor(1.0)
                tgt_area = areas.to(device)
                small_thr = areas.new_tensor(self.small_area_thr).to(device)
            else:
                if torch.is_tensor(img_shape):
                    h, w = img_shape[:2].to(device).float()
                else:
                    h, w = float(img_shape[0]), float(img_shape[1])
                    h = areas.new_tensor(h).to(device)
                    w = areas.new_tensor(w).to(device)
                size_prod = (h * w).clamp(min=1)
                tgt_area = areas.to(device) / size_prod
                small_thr = areas.new_tensor(self.small_area_thr).to(device) / size_prod

            small_mask = tgt_area[tgt_idx] < small_thr
            if small_mask.sum() == 0:
                continue
            matched_small_src = src_idx[small_mask]
            matched_small_tgt = tgt_idx[small_mask]
            total_small += int(matched_small_src.numel())

            matched_set = torch.zeros(logits.shape[1], dtype=torch.bool, device=device)
            matched_set[src_idx] = True
            unmatched = torch.nonzero(~matched_set, as_tuple=False).flatten()
            valid_unmatched = unmatched[scores[b, unmatched] > self.min_score]
            if valid_unmatched.numel() > self.topk_unmatched:
                top = scores[b, valid_unmatched].topk(self.topk_unmatched).indices
                valid_unmatched = valid_unmatched[top]
            if valid_unmatched.numel() == 0:
                continue

            dist = torch.cdist(refs[b, valid_unmatched, :2], refs[b, matched_small_src, :2], p=2).nan_to_num(1.0)
            iou = box_iou(
                box_cxcywh_to_xyxy(boxes[b, valid_unmatched]),
                box_cxcywh_to_xyxy(boxes[b, matched_small_src]),
            )[0]
            close = (dist < self.delta) | (iou > 0.3)
            if close.sum() == 0:
                continue
            s_i = scores[b, valid_unmatched][:, None]
            s_j = scores[b, matched_small_src][None, :]
            m_j = torch.exp(-tgt_area[matched_small_tgt][None, :].float() / (small_thr + 1e-6))
            collision = s_i * s_j * torch.exp(-dist / max(self.sigma, 1e-6)) * m_j
            collision = collision.nan_to_num(0.0).clamp(0.0, 1.0)
            total_loss = total_loss + (collision * torch.relu(self.delta - dist))[close].mean()
            total_pairs += int(close.sum().item())

        qcr = bbox_preds.new_tensor(total_pairs / max(1, total_small))
        return self.loss_weight * total_loss / max(1, len(indices)), {"query_collision_rate": qcr}
