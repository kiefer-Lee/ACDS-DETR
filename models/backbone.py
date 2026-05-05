from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from .position_encoding import PositionEmbeddingSine


class Backbone(nn.Module):
    def __init__(self, name="resnet50", pretrained=False, train_backbone=True, hidden_dim=256, num_feature_levels=4):
        super().__init__()
        if name == "resnet18":
            backbone = torchvision.models.resnet18(weights=None if not pretrained else "DEFAULT")
            channels = [128, 256, 512]
        elif name == "resnet50":
            backbone = torchvision.models.resnet50(weights=None if not pretrained else "DEFAULT")
            channels = [512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        for pname, p in backbone.named_parameters():
            if not train_backbone or ("layer2" not in pname and "layer3" not in pname and "layer4" not in pname):
                p.requires_grad_(False)
        return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.input_proj = nn.ModuleList([nn.Conv2d(c, hidden_dim, 1) for c in channels])
        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 3:
            self.input_proj.append(nn.Conv2d(channels[-1], hidden_dim, 3, stride=2, padding=1))
        self.position_embedding = PositionEmbeddingSine(hidden_dim // 2)

    def forward(self, samples):
        x = samples["tensors"]
        mask = samples["mask"]
        feats = self.body(x)
        out, masks, pos = [], [], []
        last_raw = None
        for i, feat in enumerate(feats.values()):
            last_raw = feat
            proj = self.input_proj[i](feat)
            m = F.interpolate(mask[:, None].float(), size=proj.shape[-2:]).to(torch.bool)[:, 0]
            out.append(proj)
            masks.append(m)
            pos.append(self.position_embedding(m))
        if self.num_feature_levels > 3:
            proj = self.input_proj[3](last_raw)
            m = F.interpolate(mask[:, None].float(), size=proj.shape[-2:]).to(torch.bool)[:, 0]
            out.append(proj)
            masks.append(m)
            pos.append(self.position_embedding(m))
        return out, masks, pos

