import torch
from torch import nn
import torch.nn.functional as F


class MultiScaleDeformableAttention(nn.Module):
    """Pure PyTorch multi-scale deformable attention.

    This is slower than the CUDA op used by the official Deformable DETR, but it
    keeps the project portable and makes R-SNDS easy to inspect.
    """

    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, srcs, masks, reference_points, gamma=None):
        bs, q, _ = query.shape
        offsets = self.sampling_offsets(query).view(bs, q, self.n_heads, self.n_levels, self.n_points, 2)
        if gamma is not None:
            offsets = offsets * gamma[:, :, None, None, None, :]
        attn = self.attention_weights(query).view(bs, q, self.n_heads, self.n_levels * self.n_points)
        attn = attn.softmax(-1).view(bs, q, self.n_heads, self.n_levels, self.n_points)
        outputs = query.new_zeros(bs, q, self.n_heads, self.head_dim)
        sampling_locations = []
        for lvl, src in enumerate(srcs):
            _, c, h, w = src.shape
            value = self.value_proj(src.flatten(2).transpose(1, 2)).transpose(1, 2).view(bs, self.n_heads, self.head_dim, h, w)
            norm = query.new_tensor([w, h])
            ref = reference_points
            if ref.shape[-1] == 4:
                ref_xy = ref[..., :2]
                ref_wh = ref[..., 2:].clamp(min=1e-4)
                loc = ref_xy[:, :, None, None, :] + offsets[:, :, :, lvl] / self.n_points * ref_wh[:, :, None, None, :]
            else:
                loc = ref[:, :, None, None, :] + offsets[:, :, :, lvl] / norm
            loc = loc.clamp(0, 1)
            grid = loc.permute(0, 2, 1, 3, 4).reshape(bs * self.n_heads, q * self.n_points, 1, 2)
            grid = grid * 2.0 - 1.0
            value = value.reshape(bs * self.n_heads, self.head_dim, h, w)
            sampled = F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            sampled = sampled.view(bs, self.n_heads, self.head_dim, q, self.n_points).permute(0, 3, 1, 4, 2)
            w_attn = attn[:, :, :, lvl].unsqueeze(-1)
            outputs = outputs + (sampled * w_attn).sum(3)
            sampling_locations.append(loc.detach())
        outputs = outputs.flatten(2)
        outputs = self.output_proj(outputs)
        return self.dropout(outputs), torch.stack(sampling_locations, dim=3), attn.detach()

