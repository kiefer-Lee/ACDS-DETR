import copy

import torch


class ModelEma:
    def __init__(self, model, decay=0.9997):
        self.decay = float(decay)
        src = model.module if hasattr(model, "module") else model
        self.module = copy.deepcopy(src).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = model.module if hasattr(model, "module") else model
        src_state = src.state_dict()
        ema_state = self.module.state_dict()
        for key, ema_value in ema_state.items():
            src_value = src_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(src_value.to(ema_value.device), alpha=1.0 - self.decay)
            else:
                ema_value.copy_(src_value.to(ema_value.device))

    def to(self, device):
        self.module.to(device)
        return self

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict, strict=False)
