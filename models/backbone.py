from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from .position_encoding import PositionEmbeddingSine


class Backbone(nn.Module):
    def __init__(
        self,
        name="resnet50",
        pretrained=False,
        train_backbone=True,
        hidden_dim=256,
        num_feature_levels=4,
        pretrained_path=None,
        use_p2=False,
        train_backbone_layers=None,
    ):
        super().__init__()
        if name == "resnet18":
            backbone = torchvision.models.resnet18(weights=None if not pretrained else "DEFAULT")
            base_channels = {"layer1": 64, "layer2": 128, "layer3": 256, "layer4": 512}
        elif name == "resnet50":
            backbone = torchvision.models.resnet50(weights=None if not pretrained else "DEFAULT")
            base_channels = {"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}
        else:
            raise ValueError(f"Unsupported backbone: {name}")
        if pretrained_path:
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            cleaned = {k.replace("module.", ""): v for k, v in state.items()}
            backbone.load_state_dict(cleaned, strict=False)
        if train_backbone_layers is None:
            train_backbone_layers = ["layer2", "layer3", "layer4"]
        train_backbone_layers = set(train_backbone_layers)
        for pname, p in backbone.named_parameters():
            if not train_backbone or not any(layer in pname for layer in train_backbone_layers):
                p.requires_grad_(False)
        stage_names = ["layer1", "layer2", "layer3", "layer4"] if use_p2 else ["layer2", "layer3", "layer4"]
        if num_feature_levels < len(stage_names):
            stage_names = stage_names[-num_feature_levels:]
        return_layers = {name: str(i) for i, name in enumerate(stage_names)}
        channels = [base_channels[name] for name in stage_names]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.input_proj = nn.ModuleList([nn.Conv2d(c, hidden_dim, 1) for c in channels])
        self.num_feature_levels = num_feature_levels
        self.num_body_levels = len(channels)
        if num_feature_levels > self.num_body_levels:
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
        if self.num_feature_levels > self.num_body_levels:
            proj = self.input_proj[self.num_body_levels](last_raw)
            m = F.interpolate(mask[:, None].float(), size=proj.shape[-2:]).to(torch.bool)[:, 0]
            out.append(proj)
            masks.append(m)
            pos.append(self.position_embedding(m))
        return out, masks, pos
