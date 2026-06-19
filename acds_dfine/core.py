"""Portable ACDS components for D-FINE style DETR outputs.

The official D-FINE codebase is not MMDetection-native.  This module keeps the
ACDS pieces that depend only on DETR-shaped tensors so they can be called from
either a patched D-FINE criterion or from small standalone verification tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import nn

from mmdet_acds.models.acq_loss import ACQLoss
from mmdet_acds.models.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from mmdet_acds.models.rsnds_msda import ReliabilityGuidedScaleSampler
from mmdet_acds.models.small_object_assigner import linear_sum_assignment_torch


@dataclass
class DFineACDSConfig:
    """Configuration for the D-FINE ACDS adapter."""

    acq_enabled: bool = True
    acq_loss_weight: float = 0.03
    acq_apply_last_n_layers: int = 2
    acq_topk_unmatched: int = 30
    acq_delta: float = 0.03
    acq_sigma: float = 0.06
    acq_min_score: float = 0.40
    small_area_thr: float = 1024.0
    small_object_cost_gain: float = 0.20
    class_cost: float = 2.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0
    use_sigmoid_cls: bool = True
    target_box_format: str = "xyxy_abs"
    rsnds: dict[str, Any] = field(
        default_factory=lambda: dict(
            enabled=True,
            beta=1.0,
            gamma_base=1.0,
            gamma_min=0.35,
            gamma_max=1.25,
            reliability="cls_conf",
        )
    )


def _target_size(target: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    size = _get_target_field(target, "img_shape", None)
    if size is None:
        size = _get_target_field(target, "orig_size", None)
    if size is None:
        size = _get_target_field(target, "size", None)
    if size is None:
        return torch.ones((), device=device), torch.ones((), device=device)
    if torch.is_tensor(size):
        h, w = size[:2].to(device).float()
    else:
        h = torch.tensor(float(size[0]), device=device)
        w = torch.tensor(float(size[1]), device=device)
    return h.clamp(min=1), w.clamp(min=1)


def _get_target_field(target: Any, name: str, default: Any = None) -> Any:
    if isinstance(target, dict):
        return target.get(name, default)
    return getattr(target, name, default)


def _target_boxes_to_abs_xyxy(
    boxes: torch.Tensor,
    target: Any,
    box_format: str,
) -> torch.Tensor:
    """Convert common D-FINE/MMDet target box layouts to absolute xyxy."""

    boxes = boxes.float()
    if boxes.numel() == 0:
        return boxes.reshape(0, 4)

    device = boxes.device
    h, w = _target_size(target, device)
    fmt = box_format.lower()
    if fmt == "xyxy_abs":
        return boxes
    if fmt == "xywh_abs":
        return torch.stack((boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]), dim=-1)
    if fmt == "cxcywh_norm":
        xyxy = box_cxcywh_to_xyxy(boxes)
        return xyxy * torch.stack((w, h, w, h))
    if fmt == "xyxy_norm":
        return boxes * torch.stack((w, h, w, h))
    raise ValueError(f"Unsupported target_box_format: {box_format}")


def _abs_xyxy_to_norm_cxcywh(boxes: torch.Tensor, target: Any) -> torch.Tensor:
    device = boxes.device
    h, w = _target_size(target, device)
    norm = torch.stack((w, h, w, h))
    xyxy = (boxes / norm).clamp(0.0, 1.0)
    return torch.stack(
        (
            (xyxy[:, 0] + xyxy[:, 2]) * 0.5,
            (xyxy[:, 1] + xyxy[:, 3]) * 0.5,
            (xyxy[:, 2] - xyxy[:, 0]).clamp(min=0.0),
            (xyxy[:, 3] - xyxy[:, 1]).clamp(min=0.0),
        ),
        dim=-1,
    )


def _class_prob(cls_scores: torch.Tensor, labels: torch.Tensor, use_sigmoid: bool) -> torch.Tensor:
    if use_sigmoid:
        return cls_scores.float().sigmoid()[:, labels]
    return cls_scores.float().softmax(-1)[:, labels]


def build_gt_instances_for_acq(
    targets: Sequence[Any],
    *,
    target_box_format: str = "xyxy_abs",
    device: torch.device | None = None,
) -> list[dict[str, torch.Tensor | tuple[int, int]]]:
    """Build the lightweight GT dictionaries consumed by :class:`ACQLoss`."""

    gt_instances = []
    for target in targets:
        boxes = _get_target_field(target, "bboxes", None)
        if boxes is None:
            boxes = _get_target_field(target, "boxes")
        labels = _get_target_field(target, "labels")
        if boxes is None or labels is None:
            raise ValueError("Each target must provide boxes/bboxes and labels")
        if device is not None:
            boxes = boxes.to(device)
            labels = labels.to(device)
        abs_boxes = _target_boxes_to_abs_xyxy(boxes, target, target_box_format)
        areas = (abs_boxes[:, 2] - abs_boxes[:, 0]).clamp(min=0) * (abs_boxes[:, 3] - abs_boxes[:, 1]).clamp(min=0)
        h, w = _target_size(target, abs_boxes.device)
        gt_instances.append(
            dict(
                bboxes=abs_boxes,
                labels=labels.long(),
                areas=areas,
                img_shape=(int(h.item()), int(w.item())),
            )
        )
    return gt_instances


def small_object_hungarian_indices(
    cls_scores: torch.Tensor,
    bbox_preds: torch.Tensor,
    targets: Sequence[Any],
    *,
    target_box_format: str = "xyxy_abs",
    small_object_cost_gain: float = 0.20,
    small_area_thr: float = 1024.0,
    class_cost: float = 2.0,
    bbox_cost: float = 5.0,
    giou_cost: float = 2.0,
    use_sigmoid_cls: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Hungarian matching with extra bbox/GIoU weight for small GT boxes.

    Args:
        cls_scores: ``[bs, num_queries, num_classes]`` logits.
        bbox_preds: ``[bs, num_queries, 4]`` normalized ``cxcywh`` boxes.
        targets: Per-image target dictionaries/objects.
    """

    if cls_scores.ndim != 3 or bbox_preds.ndim != 3:
        raise ValueError("cls_scores and bbox_preds must be [bs, num_queries, ...]")
    if len(targets) != int(cls_scores.shape[0]):
        raise ValueError("targets length must match batch size")

    indices: list[tuple[torch.Tensor, torch.Tensor]] = []
    device = bbox_preds.device
    for b, target in enumerate(targets):
        boxes = _get_target_field(target, "bboxes", None)
        if boxes is None:
            boxes = _get_target_field(target, "boxes")
        labels = _get_target_field(target, "labels")
        if boxes is None or labels is None:
            raise ValueError("Each target must provide boxes/bboxes and labels")
        boxes = boxes.to(device)
        labels = labels.to(device).long()
        if boxes.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            indices.append((empty, empty))
            continue

        abs_boxes = _target_boxes_to_abs_xyxy(boxes, target, target_box_format)
        gt_boxes = _abs_xyxy_to_norm_cxcywh(abs_boxes, target)
        areas = (abs_boxes[:, 2] - abs_boxes[:, 0]).clamp(min=0) * (abs_boxes[:, 3] - abs_boxes[:, 1]).clamp(min=0)
        small_weights = 1.0 + float(small_object_cost_gain) * (areas < float(small_area_thr)).float()

        cost_class = -_class_prob(cls_scores[b], labels, use_sigmoid_cls)
        cost_bbox = torch.cdist(bbox_preds[b].float(), gt_boxes.float(), p=1) * small_weights[None, :]
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(bbox_preds[b].float()),
            box_cxcywh_to_xyxy(gt_boxes.float()),
        ) * small_weights[None, :]
        cost = class_cost * cost_class + bbox_cost * cost_bbox + giou_cost * cost_giou
        src, tgt = linear_sum_assignment_torch(cost)
        indices.append((src.to(device), tgt.to(device)))
    return indices


class DFineACDSCriterion(nn.Module):
    """ACDS auxiliary criterion for D-FINE/DEIM-D-FINE decoder outputs.

    The criterion returns only ACDS-specific losses.  It should be added to the
    official D-FINE loss dictionary instead of replacing D-FINE's FDR/GO-LSD
    losses.
    """

    def __init__(self, config: DFineACDSConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            self.cfg = DFineACDSConfig()
        elif isinstance(config, dict):
            self.cfg = DFineACDSConfig(**config)
        else:
            self.cfg = config
        self.loss_acq = ACQLoss(
            enabled=self.cfg.acq_enabled,
            small_area_thr=self.cfg.small_area_thr,
            topk_unmatched=self.cfg.acq_topk_unmatched,
            delta=self.cfg.acq_delta,
            sigma=self.cfg.acq_sigma,
            min_score=self.cfg.acq_min_score,
            loss_weight=self.cfg.acq_loss_weight,
            use_sigmoid_cls=self.cfg.use_sigmoid_cls,
        )

    def _layers_from_outputs(self, outputs: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
        layers = []
        for aux in outputs.get("aux_outputs", []) or []:
            if "pred_logits" in aux and "pred_boxes" in aux:
                layers.append(aux)
        layers.append(outputs)
        return layers[-max(1, int(self.cfg.acq_apply_last_n_layers)) :]

    def _indices_for_layers(
        self,
        indices: Sequence[tuple[torch.Tensor, torch.Tensor]]
        | Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]]
        | None,
        num_layers: int,
    ) -> list[Sequence[tuple[torch.Tensor, torch.Tensor]] | None]:
        if indices is None:
            return [None] * num_layers
        if len(indices) == 0:
            return [None] * num_layers

        first = indices[0]
        if isinstance(first, tuple):
            return [indices] * num_layers  # type: ignore[list-item]

        layer_indices = list(indices)  # type: ignore[arg-type]
        if len(layer_indices) < num_layers:
            return [None] * (num_layers - len(layer_indices)) + layer_indices
        return layer_indices[-num_layers:]

    def forward(
        self,
        outputs: dict[str, Any],
        targets: Sequence[Any],
        indices: Sequence[tuple[torch.Tensor, torch.Tensor]]
        | Sequence[Sequence[tuple[torch.Tensor, torch.Tensor]]]
        | None = None,
    ) -> dict[str, torch.Tensor]:
        layers = self._layers_from_outputs(outputs)
        if not layers:
            raise ValueError("outputs must contain pred_logits and pred_boxes")

        total = None
        qcr_values = []
        layer_indices = self._indices_for_layers(indices, len(layers))
        for layer, indices_for_layer in zip(layers, layer_indices):
            cls_scores = layer["pred_logits"]
            bbox_preds = layer["pred_boxes"]
            gt_instances = build_gt_instances_for_acq(
                targets,
                target_box_format=self.cfg.target_box_format,
                device=bbox_preds.device,
            )
            if indices_for_layer is None:
                indices_for_layer = small_object_hungarian_indices(
                    cls_scores,
                    bbox_preds,
                    gt_instances,
                    target_box_format="xyxy_abs",
                    small_object_cost_gain=self.cfg.small_object_cost_gain,
                    small_area_thr=self.cfg.small_area_thr,
                    class_cost=self.cfg.class_cost,
                    bbox_cost=self.cfg.bbox_cost,
                    giou_cost=self.cfg.giou_cost,
                    use_sigmoid_cls=self.cfg.use_sigmoid_cls,
                )
            refs = layer.get("reference_points", None)
            loss, stats = self.loss_acq(
                cls_scores,
                bbox_preds,
                gt_instances,
                indices=indices_for_layer,
                reference_points=refs,
            )
            total = loss if total is None else total + loss
            qcr_values.append(stats["query_collision_rate"])

        assert total is not None
        return {
            "loss_acq": total / len(layers),
            "query_collision_rate": torch.stack(qcr_values).mean(),
        }


class ScaleAwareQueryBias(nn.Module):
    """A lightweight scale-group bias for D-FINE object queries."""

    def __init__(self, embed_dims: int = 256, groups: int = 4, strength: float = 0.35) -> None:
        super().__init__()
        self.groups = int(groups)
        self.strength = float(strength)
        self.embedding = nn.Embedding(self.groups, embed_dims)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        if query.ndim != 3:
            raise ValueError("query must be [bs, num_queries, embed_dims]")
        group_ids = torch.arange(query.shape[1], device=query.device) % self.groups
        return query + self.strength * self.embedding(group_ids).unsqueeze(0).to(dtype=query.dtype)


def rsnds_gamma_for_dfine(
    query: torch.Tensor,
    pred_boxes: torch.Tensor | None,
    pred_logits: torch.Tensor | None,
    *,
    embed_dims: int = 256,
    rsnds: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute R-SNDS gamma/rho tensors for a D-FINE decoder layer."""

    sampler = ReliabilityGuidedScaleSampler(hidden_dim=embed_dims, **(rsnds or {})).to(query.device)
    return sampler(query, pred_boxes, pred_logits)


def apply_rsnds_to_sampling_offsets(sampling_offsets: torch.Tensor, gamma: torch.Tensor | None) -> torch.Tensor:
    """Scale deformable attention offsets with a precomputed R-SNDS gamma."""

    if gamma is None:
        return sampling_offsets
    if gamma.shape[-1] == 1:
        gamma = gamma.repeat_interleave(2, dim=-1)
    return sampling_offsets * gamma[:, :, None, None, None, :].to(dtype=sampling_offsets.dtype)
