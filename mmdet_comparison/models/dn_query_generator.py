"""Denoising query generation for DN-DETR style training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def bbox_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    cx = (boxes[..., 0] + boxes[..., 2]) * 0.5
    cy = (boxes[..., 1] + boxes[..., 3]) * 0.5
    w = boxes[..., 2] - boxes[..., 0]
    h = boxes[..., 3] - boxes[..., 1]
    return torch.stack((cx, cy, w, h), dim=-1)


def normalize_bboxes(boxes: Tensor, img_shape: tuple[int, int] | list[int]) -> Tensor:
    h, w = int(img_shape[0]), int(img_shape[1])
    scale = boxes.new_tensor([w, h, w, h]).clamp(min=1)
    return bbox_xyxy_to_cxcywh((boxes.float() / scale).clamp(0.0, 1.0)).clamp(0.0, 1.0)


def get_gt_instances(data_sample: Any) -> Any:
    if isinstance(data_sample, dict):
        return data_sample.get("gt_instances")
    return getattr(data_sample, "gt_instances", None)


def get_img_shape(data_sample: Any) -> tuple[int, int]:
    if isinstance(data_sample, dict):
        metainfo = data_sample.get("metainfo", data_sample)
        shape = metainfo.get("img_shape", metainfo.get("batch_input_shape", (1, 1)))
    else:
        metainfo = getattr(data_sample, "metainfo", {})
        shape = metainfo.get("img_shape", getattr(data_sample, "img_shape", (1, 1)))
    return int(shape[0]), int(shape[1])


def get_labels_and_boxes(data_sample: Any, device: torch.device) -> tuple[Tensor, Tensor]:
    gt_instances = get_gt_instances(data_sample)
    if gt_instances is None:
        empty_labels = torch.empty(0, dtype=torch.long, device=device)
        empty_boxes = torch.empty(0, 4, dtype=torch.float32, device=device)
        return empty_labels, empty_boxes

    labels = getattr(gt_instances, "labels", None)
    bboxes = getattr(gt_instances, "bboxes", None)
    if labels is None or bboxes is None:
        empty_labels = torch.empty(0, dtype=torch.long, device=device)
        empty_boxes = torch.empty(0, 4, dtype=torch.float32, device=device)
        return empty_labels, empty_boxes

    labels = labels.to(device=device, dtype=torch.long)
    boxes = normalize_bboxes(bboxes.to(device=device), get_img_shape(data_sample))
    return labels, boxes


@dataclass
class DNQueryOutput:
    label_query: Tensor
    bbox_query: Tensor
    self_attn_mask: Tensor
    meta: dict[str, Any]


class DNQueryGenerator(nn.Module):
    """Create DN-DETR denoising label and box queries.

    The generated queries are prepended to the normal matching queries. Targets
    are padded to the largest GT count in the batch, then repeated over
    positive/negative copies inside each denoising group.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dims: int,
        num_groups: int = 5,
        label_noise_scale: float = 0.2,
        box_noise_scale: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.embed_dims = int(embed_dims)
        self.num_groups = int(num_groups)
        self.label_noise_scale = float(label_noise_scale)
        self.box_noise_scale = float(box_noise_scale)
        self.label_embedding = nn.Embedding(self.num_classes + 1, self.embed_dims)

    @property
    def copies_per_group(self) -> int:
        return 2

    def forward(
        self,
        batch_data_samples: list[Any],
        num_matching_queries: int,
        device: torch.device | None = None,
    ) -> DNQueryOutput:
        if device is None:
            device = self.label_embedding.weight.device

        batch_size = len(batch_data_samples)
        gt_labels: list[Tensor] = []
        gt_boxes: list[Tensor] = []
        max_gt = 0
        for sample in batch_data_samples:
            labels, boxes = get_labels_and_boxes(sample, device)
            gt_labels.append(labels)
            gt_boxes.append(boxes)
            max_gt = max(max_gt, int(labels.numel()))

        if max_gt == 0 or self.num_groups <= 0:
            label_query = self.label_embedding.weight.new_zeros(batch_size, 0, self.embed_dims)
            bbox_query = label_query.new_zeros(batch_size, 0, 4)
            mask = torch.zeros(num_matching_queries, num_matching_queries, dtype=torch.bool, device=device)
            meta = dict(
                num_denoising_queries=0,
                num_denoising_groups=0,
                num_matching_queries=int(num_matching_queries),
                max_gt=0,
                target_labels=torch.empty(batch_size, 0, dtype=torch.long, device=device),
                target_bboxes=torch.empty(batch_size, 0, 4, dtype=torch.float32, device=device),
                target_weights=torch.empty(batch_size, 0, dtype=torch.float32, device=device),
            )
            return DNQueryOutput(label_query, bbox_query, mask, meta)

        group_size = max_gt * self.copies_per_group
        pad_size = group_size * self.num_groups
        target_labels = torch.full((batch_size, pad_size), self.num_classes, dtype=torch.long, device=device)
        target_bboxes = torch.zeros(batch_size, pad_size, 4, dtype=torch.float32, device=device)
        target_weights = torch.zeros(batch_size, pad_size, dtype=torch.float32, device=device)

        for batch_idx, (labels, boxes) in enumerate(zip(gt_labels, gt_boxes)):
            num_gt = int(labels.numel())
            if num_gt == 0:
                continue
            for group_idx in range(self.num_groups):
                base = group_idx * group_size
                for copy_idx in range(self.copies_per_group):
                    start = base + copy_idx * max_gt
                    end = start + num_gt
                    target_labels[batch_idx, start:end] = labels
                    target_bboxes[batch_idx, start:end] = boxes
                    target_weights[batch_idx, start:end] = 1.0

        noisy_labels = self._apply_label_noise(target_labels, target_weights)
        noisy_boxes = self._apply_box_noise(target_bboxes, target_weights)
        noisy_boxes = self._fill_padding_boxes(noisy_boxes, target_weights)
        label_query = self.label_embedding(noisy_labels)
        bbox_query = inverse_sigmoid(noisy_boxes)
        mask = self._build_self_attn_mask(pad_size, group_size, num_matching_queries, device)
        meta = dict(
            num_denoising_queries=int(pad_size),
            num_denoising_groups=int(self.num_groups),
            num_matching_queries=int(num_matching_queries),
            max_gt=int(max_gt),
            group_size=int(group_size),
            target_labels=target_labels,
            target_bboxes=target_bboxes,
            target_weights=target_weights,
        )
        return DNQueryOutput(label_query, bbox_query, mask, meta)

    def _apply_label_noise(self, labels: Tensor, weights: Tensor) -> Tensor:
        noisy = labels.clone()
        valid = weights > 0
        if self.label_noise_scale <= 0 or not bool(valid.any()):
            return noisy
        chosen = torch.rand_like(weights) < self.label_noise_scale
        chosen = chosen & valid
        random_labels = torch.randint(self.num_classes, labels.shape, device=labels.device)
        noisy[chosen] = random_labels[chosen]
        return noisy

    def _apply_box_noise(self, boxes: Tensor, weights: Tensor) -> Tensor:
        noisy = boxes.clone()
        valid = weights > 0
        if self.box_noise_scale <= 0 or not bool(valid.any()):
            return noisy.clamp(0.0, 1.0)

        wh = boxes[..., 2:].clamp(min=1e-4)
        centers = boxes[..., :2]
        noise = (torch.rand_like(boxes) * 2.0 - 1.0) * self.box_noise_scale
        center_noise = noise[..., :2] * wh
        size_noise = noise[..., 2:] * wh
        noisy_centers = centers + center_noise
        noisy_wh = (wh + size_noise).clamp(min=1e-4, max=1.0)
        noised = torch.cat((noisy_centers, noisy_wh), dim=-1).clamp(0.0, 1.0)
        noisy[valid] = noised[valid]
        return noisy.clamp(0.0, 1.0)

    @staticmethod
    def _fill_padding_boxes(boxes: Tensor, weights: Tensor) -> Tensor:
        """Give padded DN queries a numerically stable anchor box.

        DAB-DETR's decoder divides by reference width/height when modulated
        height-width attention is enabled. Padding slots are ignored by DN loss,
        but they still flow through the decoder, so their reference boxes must
        not have near-zero width/height.
        """

        padded = weights <= 0
        if not bool(padded.any()):
            return boxes
        boxes = boxes.clone()
        default_box = boxes.new_tensor([0.5, 0.5, 0.2, 0.2])
        boxes[padded] = default_box
        return boxes

    def _build_self_attn_mask(
        self,
        pad_size: int,
        group_size: int,
        num_matching_queries: int,
        device: torch.device,
    ) -> Tensor:
        total_queries = pad_size + int(num_matching_queries)
        mask = torch.zeros(total_queries, total_queries, dtype=torch.bool, device=device)
        if pad_size == 0:
            return mask

        mask[pad_size:, :pad_size] = True
        for group_idx in range(self.num_groups):
            start = group_idx * group_size
            end = start + group_size
            mask[start:end, :start] = True
            mask[start:end, end:pad_size] = True
        return mask
