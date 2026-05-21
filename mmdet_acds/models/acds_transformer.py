"""Transformer adapters used by the ACDS mmdet detector."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .compat import MODELS
from .rsnds_msda import RSNDSMultiScaleDeformableAttention

try:  # pragma: no cover - requires mmdet.
    from mmcv.cnn import build_norm_layer
    from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
    from mmengine.model import ModuleList
    from mmdet.models.layers.transformer.deformable_detr_layers import (
        DeformableDetrTransformerDecoder as _MMDetDecoder,
        DeformableDetrTransformerDecoderLayer as _MMDetDecoderLayer,
    )
except Exception:  # pragma: no cover
    build_norm_layer = None
    FFN = None
    MultiheadAttention = None
    ModuleList = nn.ModuleList
    _MMDetDecoder = object
    _MMDetDecoderLayer = object


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


class ACDSDeformableDetrTransformerDecoderLayer(_MMDetDecoderLayer):
    """Decoder layer that builds the cross-attention from registry/config."""

    def _init_layers(self) -> None:
        if _MMDetDecoderLayer is object:
            raise ImportError("mmdet is required to use ACDS decoder layer")
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        cross_attn_cfg = dict(self.cross_attn_cfg)
        attn_type = cross_attn_cfg.pop("type", None)
        if attn_type == "RSNDSMultiScaleDeformableAttention":
            self.cross_attn = RSNDSMultiScaleDeformableAttention(**cross_attn_cfg)
        elif attn_type is not None:
            cross_attn_cfg["type"] = attn_type
            self.cross_attn = MODELS.build(cross_attn_cfg)
        else:
            from mmcv.ops import MultiScaleDeformableAttention

            self.cross_attn = MultiScaleDeformableAttention(**cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        self.norms = ModuleList([build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(3)])


@MODELS.register_module()
class ACDSDeformableDetrTransformerDecoder(_MMDetDecoder):
    """Decoder that feeds previous predictions into R-SNDS cross-attention."""

    def _init_layers(self) -> None:
        if _MMDetDecoder is object:
            raise ImportError("mmdet is required to use ACDSDeformableDetrTransformerDecoder")
        self.layers = ModuleList([
            ACDSDeformableDetrTransformerDecoderLayer(**self.layer_cfg)
            for _ in range(self.num_layers)
        ])
        self.embed_dims = self.layers[0].embed_dims
        if self.post_norm_cfg is not None:
            raise ValueError(f"There is not post_norm in {self._get_name()}")

    def forward(
        self,
        query: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        level_start_index: torch.Tensor | None = None,
        valid_ratios: torch.Tensor | None = None,
        reg_branches: torch.nn.ModuleList | None = None,
        cls_branches: torch.nn.ModuleList | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if _MMDetDecoder is object:
            raise ImportError("mmdet is required to use ACDSDeformableDetrTransformerDecoder")
        intermediate = []
        intermediate_reference_points = []
        pred_boxes = None
        pred_logits = None
        query_pos = kwargs.pop("query_pos", None)
        key_pos = kwargs.pop("key_pos", None)
        for lid, layer in enumerate(self.layers):
            if reference_points is None:
                raise ValueError("reference_points is required for deformable decoder")
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat(
                    [valid_ratios, valid_ratios], -1
                )[:, None]
            else:
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            cross_attn = getattr(layer, "cross_attn", None)
            if hasattr(cross_attn, "set_rsnds_context"):
                cross_attn.set_rsnds_context(query, pred_boxes, pred_logits)

            query = layer(
                query,
                key=value,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                reference_points=reference_points_input,
                **kwargs,
            )

            if reg_branches is not None:
                tmp = reg_branches[lid](query)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                    pred_boxes = new_reference_points
                else:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                    pred_boxes = new_reference_points
                reference_points = new_reference_points.detach()
            else:
                pred_boxes = reference_points
            if cls_branches is not None:
                pred_logits = cls_branches[lid](query)
            if self.return_intermediate:
                intermediate.append(query)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        return query.unsqueeze(0), reference_points.unsqueeze(0)
