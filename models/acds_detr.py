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
        )
        transformer_cfg = dict(mcfg)
        transformer_cfg["rsnds"] = cfg["rsnds"]
        self.transformer = DeformableTransformer(transformer_cfg)
        hidden_dim = mcfg["hidden_dim"]
        self.query_embed = nn.Embedding(mcfg["num_queries"], hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, mcfg["num_classes"] + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(self.bbox_embed.net[-1].weight, 0)
        nn.init.constant_(self.bbox_embed.net[-1].bias, 0)

    def forward(self, samples):
        srcs, masks, pos = self.backbone(samples)
        trans_out = self.transformer(srcs, masks, pos, self.query_embed.weight, self.class_embed, self.bbox_embed)
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
