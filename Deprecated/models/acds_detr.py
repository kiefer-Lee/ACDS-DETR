import torch
from torch import nn

from .backbone import Backbone
from .heads import MLP
from .transformer import DeformableTransformer


class ACDSDETR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mcfg = cfg["model"]
        self.num_queries = mcfg["num_queries"]
        self.num_classes = mcfg["num_classes"]
        self.backbone = Backbone(
            name=mcfg["backbone"],
            pretrained=mcfg.get("pretrained_backbone", False),
            hidden_dim=mcfg["hidden_dim"],
            num_feature_levels=mcfg["num_feature_levels"],
            pretrained_path=mcfg.get("pretrained_backbone_path"),
            use_p2=mcfg.get("use_p2", False),
            train_backbone_layers=mcfg.get("train_backbone_layers"),
        )
        transformer_cfg = dict(mcfg)
        transformer_cfg["rsnds"] = cfg["rsnds"]
        self.transformer = DeformableTransformer(transformer_cfg)
        hidden_dim = mcfg["hidden_dim"]
        self.query_embed = nn.Embedding(mcfg["num_queries"], hidden_dim)
        qcfg = mcfg.get("scale_aware_query", {})
        self.scale_query_enabled = bool(qcfg.get("enabled", False))
        self.scale_query_groups = int(qcfg.get("groups", 3))
        self.scale_query_strength = float(qcfg.get("strength", 1.0))
        if self.scale_query_enabled:
            self.scale_query_embed = nn.Embedding(self.scale_query_groups, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, mcfg["num_classes"] + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(self.bbox_embed.net[-1].weight, 0)
        nn.init.constant_(self.bbox_embed.net[-1].bias, 0)

    def forward(self, samples):
        srcs, masks, pos = self.backbone(samples)
        query_embed = self.query_embed.weight
        if self.scale_query_enabled:
            group_ids = torch.arange(self.num_queries, device=query_embed.device) % self.scale_query_groups
            query_embed = query_embed + self.scale_query_strength * self.scale_query_embed(group_ids)
        trans_out = self.transformer(srcs, masks, pos, query_embed, self.class_embed, self.bbox_embed)
        logits = trans_out["pred_logits"]
        boxes = trans_out["pred_boxes"]
        out = {
            "pred_logits": logits[-1],
            "pred_boxes": boxes[-1],
            "aux_outputs": [
                {"pred_logits": l, "pred_boxes": b}
                for l, b in zip(logits[:-1], boxes[:-1])
            ],
            "all_pred_logits": logits,
            "all_pred_boxes": boxes,
            "reference_points": trans_out["reference_points"],
        }
        if "sampling_locations" in trans_out:
            out["sampling_locations"] = trans_out["sampling_locations"]
        if "attention_weights" in trans_out:
            out["attention_weights"] = trans_out["attention_weights"]
        return out


def build_model(cfg):
    return ACDSDETR(cfg)
