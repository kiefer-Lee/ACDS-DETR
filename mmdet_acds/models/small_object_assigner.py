"""Hungarian assigner with small-object cost gain."""

from __future__ import annotations

from typing import Any

import torch

from .compat import TASK_UTILS

try:  # pragma: no cover - requires mmdet.
    from mmdet.models.task_modules.assigners import AssignResult, BaseAssigner
except Exception:  # pragma: no cover
    AssignResult = None

    class BaseAssigner:  # type: ignore[no-redef]
        pass


def linear_sum_assignment_torch(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        return torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)
    except Exception:
        c = cost.detach().cpu()
        rows: list[int] = []
        cols: list[int] = []
        used_r: set[int] = set()
        used_c: set[int] = set()
        n, m = c.shape
        for idx in torch.argsort(c.flatten()).tolist():
            r, col = divmod(idx, m)
            if r not in used_r and col not in used_c:
                rows.append(r)
                cols.append(col)
                used_r.add(r)
                used_c.add(col)
                if len(cols) == min(n, m):
                    break
        return torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)


@TASK_UTILS.register_module()
class SmallObjectHungarianAssigner(BaseAssigner):
    """Apply ``1 + gain`` to bbox/GIoU costs for small ground-truth boxes."""

    def __init__(
        self,
        match_costs: list[dict] | None = None,
        small_object_cost_gain: float = 0.0,
        small_area_thr: float = 1024.0,
    ) -> None:
        self.small_object_cost_gain = float(small_object_cost_gain)
        self.small_area_thr = float(small_area_thr)
        self.match_costs_cfg = match_costs or []
        self.match_costs = []
        if match_costs is not None:
            for cfg in match_costs:
                self.match_costs.append(TASK_UTILS.build(cfg) if hasattr(TASK_UTILS, "build") else cfg)

    def apply_small_object_gain(
        self,
        cost_bbox: torch.Tensor,
        cost_iou: torch.Tensor,
        gt_bboxes: torch.Tensor,
        img_shape: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.small_object_cost_gain <= 0 or gt_bboxes.numel() == 0:
            return cost_bbox, cost_iou
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=0) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1]
        ).clamp(min=0)
        weights = 1.0 + self.small_object_cost_gain * (areas < self.small_area_thr).float()
        return cost_bbox * weights[None, :], cost_iou * weights[None, :]

    def assign(self, pred_instances: Any, gt_instances: Any, img_meta: dict | None = None, **kwargs: Any):
        if AssignResult is None:
            raise ImportError("mmdet is required to use SmallObjectHungarianAssigner.assign")

        num_gts = int(gt_instances.bboxes.size(0))
        num_preds = int(pred_instances.bboxes.size(0))
        gt_inds = pred_instances.bboxes.new_full((num_preds,), 0, dtype=torch.long)
        labels = pred_instances.bboxes.new_full((num_preds,), -1, dtype=torch.long)
        if num_gts == 0 or num_preds == 0:
            return AssignResult(num_gts, gt_inds, None, labels=labels)

        cost_parts = []
        for cost in self.match_costs:
            cost_matrix = cost(pred_instances, gt_instances, img_meta)
            cost_type = cost.__class__.__name__.lower()
            if "bbox" in cost_type or "iou" in cost_type:
                if self.small_object_cost_gain > 0:
                    areas = (gt_instances.bboxes[:, 2] - gt_instances.bboxes[:, 0]).clamp(min=0) * (
                        gt_instances.bboxes[:, 3] - gt_instances.bboxes[:, 1]
                    ).clamp(min=0)
                    weights = 1.0 + self.small_object_cost_gain * (areas < self.small_area_thr).float()
                    cost_matrix = cost_matrix * weights[None, :]
            cost_parts.append(cost_matrix)
        if not cost_parts:
            raise ValueError("SmallObjectHungarianAssigner requires at least one match cost")
        cost = torch.stack(cost_parts).sum(0)

        matched_row_inds, matched_col_inds = linear_sum_assignment_torch(cost)
        matched_row_inds = matched_row_inds.to(gt_inds.device)
        matched_col_inds = matched_col_inds.to(gt_inds.device)
        gt_inds[matched_row_inds] = matched_col_inds + 1
        labels[matched_row_inds] = gt_instances.labels[matched_col_inds]
        return AssignResult(num_gts, gt_inds, None, labels=labels)
