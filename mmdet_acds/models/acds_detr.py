"""ACDS detector wrapper around MMDetection Deformable DETR."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .compat import MODELS
from .acds_transformer import ACDSDeformableDetrTransformerDecoder

try:  # pragma: no cover - requires mmdet.
    from mmdet.models.detectors import DeformableDETR as _MMDetDeformableDETR
    from mmdet.models.layers import DeformableDetrTransformerEncoder, SinePositionalEncoding
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
    from mmengine.model import xavier_init
    from torch.nn.init import normal_
except Exception:  # pragma: no cover
    _MMDetDeformableDETR = object


def resolve_encoder_level_indices(num_levels: int, indices: list[int] | tuple[int, ...] | None) -> list[int]:
    if indices is None:
        return list(range(num_levels))
    resolved = [int(i) for i in indices if 0 <= int(i) < num_levels]
    return resolved or list(range(num_levels))


def level_token_slices(spatial_shapes: torch.Tensor) -> list[slice]:
    starts = torch.cat([spatial_shapes.new_zeros(1), spatial_shapes.prod(1).cumsum(0)[:-1]])
    slices = []
    for start, shape in zip(starts.tolist(), spatial_shapes.tolist()):
        end = int(start + shape[0] * shape[1])
        slices.append(slice(int(start), end))
    return slices


def replace_encoded_levels(
    original_tokens: torch.Tensor,
    encoded_tokens: torch.Tensor,
    spatial_shapes: torch.Tensor,
    encoded_level_indices: list[int],
) -> torch.Tensor:
    """Merge encoder output back into all decoder levels.

    This is the key P2 decoder-only behavior: unencoded levels keep their
    original projected features, while selected levels are replaced by encoder
    memory.
    """

    merged = original_tokens.clone()
    all_slices = level_token_slices(spatial_shapes)
    cursor = 0
    for level in encoded_level_indices:
        level_slice = all_slices[level]
        width = level_slice.stop - level_slice.start
        merged[:, level_slice, :] = encoded_tokens[:, cursor : cursor + width, :]
        cursor += width
    return merged


@MODELS.register_module()
class ACDSDeformableDETR(_MMDetDeformableDETR):
    """MMDetection Deformable DETR with ACDS-specific switches."""

    def __init__(
        self,
        *args: Any,
        use_p2: bool = False,
        encoder_level_indices: list[int] | None = None,
        scale_aware_query: dict | None = None,
        **kwargs: Any,
    ) -> None:
        if _MMDetDeformableDETR is object:
            raise ImportError("mmdet is required to instantiate ACDSDeformableDETR")
        self.use_p2 = bool(use_p2)
        self.encoder_level_indices = encoder_level_indices
        self.scale_aware_query_cfg = scale_aware_query or {}
        super().__init__(*args, **kwargs)
        if self.scale_aware_query_cfg.get("enabled", False):
            groups = int(self.scale_aware_query_cfg.get("groups", 4))
            embed_dims = getattr(self, "embed_dims", getattr(self, "hidden_dim", 256))
            self.scale_query_embed = torch.nn.Embedding(groups, embed_dims)

    def pre_decoder(self, *args: Any, **kwargs: Any) -> tuple[dict, dict]:
        decoder_inputs_dict, head_inputs_dict = super().pre_decoder(*args, **kwargs)
        if self.scale_aware_query_cfg.get("enabled", False):
            query = decoder_inputs_dict.get("query")
            query_pos = decoder_inputs_dict.get("query_pos")
            base = query_pos if query_pos is not None else query
            if base is not None:
                groups = int(self.scale_aware_query_cfg.get("groups", 4))
                strength = float(self.scale_aware_query_cfg.get("strength", 0.0))
                group_ids = torch.arange(base.shape[1], device=base.device) % groups
                delta = strength * self.scale_query_embed(group_ids).unsqueeze(0)
                if query_pos is not None:
                    decoder_inputs_dict["query_pos"] = query_pos + delta
                else:
                    decoder_inputs_dict["query"] = query + delta
        return decoder_inputs_dict, head_inputs_dict

    def _init_layers(self) -> None:
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        decoder_cfg = dict(self.decoder)
        decoder_cfg.pop("type", None)
        self.decoder = ACDSDeformableDetrTransformerDecoder(**decoder_cfg)
        self.embed_dims = self.encoder.embed_dims
        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims * 2)

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
            self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans_fc = nn.Linear(self.embed_dims * 2, self.embed_dims * 2)
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * 2)
        else:
            self.reference_points_fc = nn.Linear(self.embed_dims, 2)

    def init_weights(self) -> None:
        super().init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if self.as_two_stage:
            nn.init.xavier_uniform_(self.memory_trans_fc.weight)
            nn.init.xavier_uniform_(self.pos_trans_fc.weight)
        else:
            xavier_init(self.reference_points_fc, distribution="uniform", bias=0.0)
        normal_(self.level_embed)

    def forward_encoder(
        self,
        feat: torch.Tensor,
        feat_mask: torch.Tensor | None,
        feat_pos: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
    ) -> dict:
        if self.encoder_level_indices is None:
            return super().forward_encoder(feat, feat_mask, feat_pos, spatial_shapes, level_start_index, valid_ratios)

        num_levels = int(spatial_shapes.shape[0])
        encoded_levels = resolve_encoder_level_indices(num_levels, self.encoder_level_indices)
        slices = level_token_slices(spatial_shapes)

        def select_tokens(x: torch.Tensor | None) -> torch.Tensor | None:
            if x is None:
                return None
            return torch.cat([x[:, slices[i], ...] for i in encoded_levels], dim=1)

        selected_shapes = torch.stack([spatial_shapes[i] for i in encoded_levels])
        selected_level_start_index = torch.cat([selected_shapes.new_zeros(1), selected_shapes.prod(1).cumsum(0)[:-1]])
        encoded_memory = self.encoder(
            query=select_tokens(feat),
            query_pos=select_tokens(feat_pos),
            key_padding_mask=select_tokens(feat_mask),
            spatial_shapes=selected_shapes,
            level_start_index=selected_level_start_index,
            valid_ratios=valid_ratios[:, encoded_levels, :],
        )
        memory = replace_encoded_levels(feat, encoded_memory, spatial_shapes, encoded_levels)
        return dict(memory=memory, memory_mask=feat_mask, spatial_shapes=spatial_shapes)

    def forward_decoder(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor | None,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
    ) -> dict:
        inter_states, inter_references = self.decoder(
            query=query,
            value=memory,
            query_pos=query_pos,
            key_padding_mask=memory_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches if self.with_box_refine else None,
            cls_branches=self.bbox_head.cls_branches,
        )
        return dict(hidden_states=inter_states, references=[reference_points, *inter_references])
