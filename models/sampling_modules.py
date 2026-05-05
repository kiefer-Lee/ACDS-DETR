import torch
from torch import nn


class ReliabilityGuidedScaleSampler(nn.Module):
    def __init__(
        self,
        hidden_dim,
        beta=1.0,
        gamma_base=1.0,
        gamma_min=0.25,
        gamma_max=1.5,
        reliability="cls_conf",
        enabled=True,
    ):
        super().__init__()
        self.beta = beta
        self.gamma_base = gamma_base
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.reliability = reliability
        self.enabled = enabled
        # Avoid registering unused parameters under DDP when cls_conf reliability is used.
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)) if reliability == "mlp" else None

    def forward(self, query, pred_boxes, pred_logits=None):
        if not self.enabled or pred_boxes is None:
            gamma = query.new_full(query.shape[:2] + (1,), self.gamma_base)
            rho = query.new_zeros(query.shape[:2] + (1,))
            return gamma, rho
        wh = pred_boxes[..., 2:4].clamp(min=1e-4)
        gamma_scale = self.beta * torch.sqrt(wh[..., 0] * wh[..., 1]).unsqueeze(-1)
        gamma_scale = gamma_scale.clamp(self.gamma_min, self.gamma_max)
        if self.reliability == "mlp" or pred_logits is None:
            if self.mlp is None:
                rho = query.new_zeros(query.shape[:2] + (1,))
                gamma = query.new_full(query.shape[:2] + (1,), self.gamma_base)
                return gamma, rho
            rho = torch.sigmoid(self.mlp(query))
        else:
            prob = pred_logits.softmax(-1)[..., :-1]
            rho = prob.max(-1, keepdim=True).values.detach()
        gamma = (1.0 - rho) * self.gamma_base + rho * gamma_scale
        return gamma, rho
