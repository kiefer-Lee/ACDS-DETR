"""R-SNDS attention components for ACDS-DETR."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .compat import MODELS


class ReliabilityGuidedScaleSampler(nn.Module):
    """Compute per-query offset scale ``gamma`` following the legacy formula."""

    def __init__(
        self,
        hidden_dim: int = 256,
        beta: float = 1.0,
        gamma_base: float = 1.0,
        gamma_min: float = 0.35,
        gamma_max: float = 1.25,
        reliability: str = "cls_conf",
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.gamma_base = float(gamma_base)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.reliability = reliability
        self.enabled = bool(enabled)
        self.mlp = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            if reliability == "mlp"
            else None
        )

    def forward(
        self,
        query: torch.Tensor,
        pred_boxes: torch.Tensor | None,
        pred_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled or pred_boxes is None:
            gamma = query.new_full(query.shape[:2] + (1,), self.gamma_base)
            rho = query.new_zeros(query.shape[:2] + (1,))
            return gamma, rho

        pred_boxes_f = pred_boxes.detach().float().nan_to_num(0.5).clamp(0.0, 1.0)
        wh = pred_boxes_f[..., 2:4].clamp(min=1e-4)
        gamma_scale = self.beta * torch.sqrt((wh[..., 0] * wh[..., 1]).clamp(min=1e-8)).unsqueeze(-1)
        gamma_scale = gamma_scale.clamp(self.gamma_min, self.gamma_max)

        if self.reliability == "mlp":
            rho = torch.sigmoid(self.mlp(query.float())) if self.mlp is not None else torch.zeros_like(gamma_scale)
        elif pred_logits is not None:
            prob = pred_logits.detach().float().softmax(-1)[..., :-1]
            rho = prob.max(-1, keepdim=True).values
        else:
            rho = torch.zeros_like(gamma_scale)

        rho = rho.nan_to_num(0.0).clamp(0.0, 1.0)
        gamma = (1.0 - rho) * self.gamma_base + rho * gamma_scale
        gamma = gamma.nan_to_num(self.gamma_base).clamp(self.gamma_min, self.gamma_max).to(dtype=query.dtype)
        return gamma, rho.to(dtype=query.dtype)


try:  # pragma: no cover - depends on mmcv being installed.
    from mmcv.ops import MultiScaleDeformableAttention as _MMCVMSDA
    from mmcv.ops.multi_scale_deform_attn import (
        MultiScaleDeformableAttnFunction,
        multi_scale_deformable_attn_pytorch,
    )
except Exception:  # pragma: no cover
    _MMCVMSDA = None


@MODELS.register_module()
class RSNDSMultiScaleDeformableAttention(_MMCVMSDA if _MMCVMSDA is not None else nn.Module):
    """MMCV MSDA with optional R-SNDS offset scaling.

    The class is intentionally thin: it keeps MMCV's CUDA op and only injects
    ``gamma`` before sampling locations are formed. On an installed mmdet stack,
    decoder code should set ``self.last_gamma`` through ``set_rsnds_context``.
    """

    def __init__(self, *args: Any, rsnds: dict | None = None, embed_dims: int = 256, **kwargs: Any) -> None:
        if _MMCVMSDA is None:
            nn.Module.__init__(self)
            self.embed_dims = int(embed_dims)
        else:
            super().__init__(*args, embed_dims=embed_dims, **kwargs)
        self.sampler = ReliabilityGuidedScaleSampler(hidden_dim=embed_dims, **(rsnds or {}))
        self.last_gamma: torch.Tensor | None = None
        self.last_rho: torch.Tensor | None = None

    def set_rsnds_context(
        self,
        query: torch.Tensor,
        pred_boxes: torch.Tensor | None,
        pred_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, rho = self.sampler(query, pred_boxes, pred_logits)
        self.last_gamma = gamma
        self.last_rho = rho
        return gamma, rho

    def _scale_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        if self.last_gamma is None:
            return offsets
        gamma = self.last_gamma
        if gamma.shape[-1] == 1:
            gamma = gamma.repeat_interleave(2, dim=-1)
        return offsets * gamma[:, :, None, None, None, :]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        identity: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        **kwargs: Any,
    ):  # pragma: no cover - requires mmcv internals.
        if _MMCVMSDA is None:
            raise ImportError("mmcv is required to run RSNDSMultiScaleDeformableAttention.forward")
        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        if spatial_shapes is None or level_start_index is None or reference_points is None:
            raise ValueError("reference_points, spatial_shapes and level_start_index are required")
        if (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() != num_value:
            raise ValueError("spatial_shapes do not match value length")

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        sampling_offsets = self._scale_offsets(sampling_offsets)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError("reference_points last dim must be 2 or 4")

        if value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value,
                spatial_shapes,
                level_start_index,
                sampling_locations.to(value.dtype),
                attention_weights.to(value.dtype),
                self.im2col_step,
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return self.dropout(output) + identity
