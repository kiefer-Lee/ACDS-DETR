import torch
from torch import nn
import torch.nn.functional as F

from .deformable_attention import MultiScaleDeformableAttention
from .heads import inverse_sigmoid
from .sampling_modules import ReliabilityGuidedScaleSampler


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_levels, num_points):
        super().__init__()
        self.self_attn = MultiScaleDeformableAttention(d_model, nhead, num_levels, num_points, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, pos, srcs, masks, reference_points):
        query = src + pos
        src2, _, _ = self.self_attn(query, srcs, masks, reference_points)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_levels, num_points, rsnds_cfg):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = MultiScaleDeformableAttention(d_model, nhead, num_levels, num_points, dropout)
        self.sampler = ReliabilityGuidedScaleSampler(d_model, **rsnds_cfg)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, query_pos, srcs, masks, reference_points, pred_boxes=None, pred_logits=None):
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        gamma, rho = self.sampler(tgt, pred_boxes, pred_logits)
        gamma_vec = gamma.repeat_interleave(2, dim=-1)
        tgt2, sampling_locations, attn = self.cross_attn(tgt + query_pos, srcs, masks, reference_points, gamma_vec)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt, sampling_locations, attn, gamma, rho


class DeformableTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg["hidden_dim"]
        self.d_model = d_model
        self.num_layers = cfg["dec_layers"]
        self.return_intermediates = cfg.get("return_intermediates", False)
        self.encoder_feature_indices = cfg.get("encoder_feature_indices")
        encoder_level_count = len(self._resolve_encoder_indices(cfg["num_feature_levels"]))
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, cfg["nheads"], cfg["dim_feedforward"], cfg["dropout"], encoder_level_count, cfg["num_points"])
            for _ in range(cfg["enc_layers"])
        ])
        rsnds_cfg = {
            "beta": cfg["rsnds"]["beta"],
            "gamma_base": cfg["rsnds"]["gamma_base"],
            "gamma_min": cfg["rsnds"]["gamma_min"],
            "gamma_max": cfg["rsnds"]["gamma_max"],
            "reliability": cfg["rsnds"]["reliability"],
            "enabled": cfg["rsnds"]["enabled"],
        }
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, cfg["nheads"], cfg["dim_feedforward"], cfg["dropout"], cfg["num_feature_levels"], cfg["num_points"], rsnds_cfg)
            for _ in range(cfg["dec_layers"])
        ])
        self.reference_points = nn.Linear(d_model, 2)

    def _resolve_encoder_indices(self, num_levels):
        if self.encoder_feature_indices is None:
            indices = list(range(num_levels))
        else:
            indices = [int(i) for i in self.encoder_feature_indices]
        indices = [i for i in indices if 0 <= i < num_levels]
        return indices or list(range(num_levels))

    @staticmethod
    def _get_valid_ratio(mask):
        _, h, w = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_w.float() / w, valid_h.float() / h], -1)

    def _get_encoder_reference_points(self, masks):
        spatial_shapes = [(m.shape[-2], m.shape[-1]) for m in masks]
        valid_ratios = torch.stack([self._get_valid_ratio(m) for m in masks], 1)
        refs = []
        device = masks[0].device
        for level, (h, w) in enumerate(spatial_shapes):
            y, x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, w, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * h).clamp(min=1e-6)
            ref_x = x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * w).clamp(min=1e-6)
            refs.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(refs, 1)
        return reference_points[:, :, None, :] * valid_ratios[:, None, :, :]

    def forward(self, srcs, masks, pos, query_embed, class_embed, bbox_embed):
        encoder_indices = self._resolve_encoder_indices(len(srcs))
        flat_src, flat_mask, flat_pos = [], [], []
        for i in encoder_indices:
            src, mask, p = srcs[i], masks[i], pos[i]
            flat_src.append(src.flatten(2).transpose(1, 2))
            flat_mask.append(mask.flatten(1))
            flat_pos.append(p.flatten(2).transpose(1, 2))
        memory = torch.cat(flat_src, dim=1)
        memory_mask = torch.cat(flat_mask, dim=1)
        memory_pos = torch.cat(flat_pos, dim=1)
        encoder_srcs = [srcs[i] for i in encoder_indices]
        encoder_masks = [masks[i] for i in encoder_indices]
        encoder_reference_points = self._get_encoder_reference_points(encoder_masks)
        for layer in self.encoder_layers:
            memory = layer(memory, memory_pos, encoder_srcs, encoder_masks, encoder_reference_points)
            memories_for_next = memory.split([s.shape[-2] * s.shape[-1] for s in encoder_srcs], dim=1)
            encoder_srcs = [
                mem.transpose(1, 2).view(src.shape[0], self.d_model, src.shape[-2], src.shape[-1])
                for mem, src in zip(memories_for_next, encoder_srcs)
            ]
        split_sizes = [srcs[i].shape[-2] * srcs[i].shape[-1] for i in encoder_indices]
        memories = memory.split(split_sizes, dim=1)
        mem_srcs = list(srcs)
        for level_idx, mem in zip(encoder_indices, memories):
            src = srcs[level_idx]
            bs, _, h, w = src.shape
            mem_srcs[level_idx] = mem.transpose(1, 2).view(bs, self.d_model, h, w)

        bs = srcs[0].shape[0]
        query_pos = query_embed.unsqueeze(0).repeat(bs, 1, 1)
        tgt = torch.zeros_like(query_pos)
        reference = self.reference_points(query_pos).sigmoid()
        outputs_classes, outputs_coords, hs = [], [], []
        refs, sampling_locations, attention_weights = [], [], []
        pred_boxes = None
        pred_logits = None
        for lid, layer in enumerate(self.decoder_layers):
            tgt, locs, attn, gamma, rho = layer(tgt, query_pos, mem_srcs, masks, reference, pred_boxes, pred_logits)
            pred_logits = class_embed(tgt)
            delta = bbox_embed(tgt)
            if reference.shape[-1] == 2:
                tmp = delta.clone()
                tmp[..., :2] = tmp[..., :2] + inverse_sigmoid(reference)
                pred_boxes = tmp.sigmoid()
            else:
                pred_boxes = (delta + inverse_sigmoid(reference)).sigmoid()
            reference = pred_boxes.detach()
            outputs_classes.append(pred_logits)
            outputs_coords.append(pred_boxes)
            hs.append(tgt)
            refs.append(reference)
            if self.return_intermediates:
                sampling_locations.append(locs)
                attention_weights.append(attn)
        out = {
            "hs": torch.stack(hs),
            "pred_logits": torch.stack(outputs_classes),
            "pred_boxes": torch.stack(outputs_coords),
            "reference_points": refs,
        }
        if self.return_intermediates:
            out["sampling_locations"] = sampling_locations
            out["attention_weights"] = attention_weights
        return out
