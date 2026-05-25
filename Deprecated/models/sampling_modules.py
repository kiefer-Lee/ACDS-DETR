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
        compute_dtype = torch.float32
        pred_boxes_f = pred_boxes.detach().to(dtype=compute_dtype).nan_to_num(0.5).clamp(0.0, 1.0)
        wh = pred_boxes_f[..., 2:4].clamp(min=1e-4)
        gamma_scale = self.beta * torch.sqrt((wh[..., 0] * wh[..., 1]).clamp(min=1e-8)).unsqueeze(-1)
        gamma_scale = gamma_scale.clamp(self.gamma_min, self.gamma_max)
        if self.reliability == "mlp" or pred_logits is None:
            if self.mlp is None:
                rho = query.new_zeros(query.shape[:2] + (1,))
                gamma = query.new_full(query.shape[:2] + (1,), self.gamma_base)
                return gamma, rho
            rho = torch.sigmoid(self.mlp(query.float()))
        else:
            prob = pred_logits.detach().float().softmax(-1)[..., :-1]
            rho = prob.max(-1, keepdim=True).values.detach()
        rho = rho.nan_to_num(0.0).clamp(0.0, 1.0)
        gamma = (1.0 - rho) * self.gamma_base + rho * gamma_scale
        gamma = gamma.nan_to_num(self.gamma_base).clamp(self.gamma_min, self.gamma_max).to(dtype=query.dtype)
        return gamma, rho.to(dtype=query.dtype)
