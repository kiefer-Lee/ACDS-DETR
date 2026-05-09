import warnings

import torch
from torch import nn
import torch.nn.functional as F

try:
    from models.ops.functions.ms_deform_attn_func import MSDeformAttnFunction
    HAS_MS_DEFORM_ATTN = True
except Exception as exc:
    MSDeformAttnFunction = None
    HAS_MS_DEFORM_ATTN = False
    _IMPORT_ERROR = exc


class MultiScaleDeformableAttention(nn.Module):
    """Multi-scale deformable attention with official CUDA MSDeformAttn op.

    It keeps the local ACDS-DETR interface unchanged:
    forward(query, srcs, masks, reference_points, gamma=None)
    """

    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4, dropout=0.1, im2col_step=64):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not HAS_MS_DEFORM_ATTN:
            raise ImportError(
                "Official MSDeformAttn CUDA op is unavailable. "
                "Compile it with: cd models/ops && sh make.sh && python test.py"
            ) from _IMPORT_ERROR

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads
        self.im2col_step = im2col_step

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, srcs, masks, reference_points, gamma=None):
        bs, num_queries, _ = query.shape

        value_list = []
        mask_list = []
        spatial_shapes = []
        for src, mask in zip(srcs, masks):
            _, _, h, w = src.shape
            spatial_shapes.append((h, w))
            value_list.append(src.flatten(2).transpose(1, 2))
            mask_list.append(mask.flatten(1))

        value = torch.cat(value_list, dim=1)
        key_padding_mask = torch.cat(mask_list, dim=1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=query.device)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )

        value = self.value_proj(value)
        value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, -1, self.n_heads, self.head_dim)

        offsets = self.sampling_offsets(query).view(
            bs, num_queries, self.n_heads, self.n_levels, self.n_points, 2
        )
        if gamma is not None:
            offsets = offsets * gamma[:, :, None, None, None, :]

        attention_weights = self.attention_weights(query).view(
            bs, num_queries, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_queries, self.n_heads, self.n_levels, self.n_points
        )

        if reference_points.shape[-1] == 2:
            normalizer = torch.stack([spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1)
            sampling_locations = (
                reference_points[:, :, None, None, None, :]
                + offsets / normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, None, None, :2]
                + offsets / self.n_points * reference_points[:, :, None, None, None, 2:] * 0.5
            )
        else:
            raise ValueError("reference_points last dim must be 2 or 4")

        # The official CUDA op requires value, sampling_locations and
        # attention_weights to share the same floating dtype. Under AMP the
        # projected value tensor is fp16 while geometry tensors can remain
        # fp32, which otherwise raises: expected scalar type Half but found Float.
        sampling_locations = sampling_locations.to(dtype=value.dtype)
        attention_weights = attention_weights.to(dtype=value.dtype)

        output = MSDeformAttnFunction.apply(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        output = self.output_proj(output)
        return self.dropout(output), sampling_locations.detach(), attention_weights.detach()
