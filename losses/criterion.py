import torch
import torch.distributed as dist
from torch import nn

from models.matcher import HungarianMatcher
from utils.box_ops import box_xyxy_to_cxcywh, normalize_boxes_xyxy
from .acq_loss import ACQLoss
from .detr_losses import loss_boxes, loss_labels


class SetCriterion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_classes = cfg["model"]["num_classes"]
        lcfg = cfg["loss"]
        self.matcher = HungarianMatcher(lcfg["cost_class"], lcfg["cost_bbox"], lcfg["cost_giou"])
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = lcfg["eos_coef"]
        self.register_buffer("empty_weight", empty_weight)
        self.weight_dict = {
            "loss_ce": lcfg["weight_class"],
            "loss_bbox": lcfg["weight_bbox"],
            "loss_giou": lcfg["weight_giou"],
            "loss_acq": cfg["acq"]["lambda_acq"],
        }
        self.aux_loss_weight = lcfg.get("aux_loss_weight", 1.0)
        self.aux_loss_layers = lcfg.get("aux_loss_layers", "all")
        self.acq = ACQLoss(cfg["acq"])
        aux_count = max(0, cfg["model"].get("dec_layers", 1) - 1)
        if isinstance(self.aux_loss_layers, int):
            start_layer = max(0, aux_count - self.aux_loss_layers)
        else:
            start_layer = 0
        for i in range(start_layer, aux_count):
            self.weight_dict[f"loss_ce_{i}"] = self.weight_dict["loss_ce"] * self.aux_loss_weight
            self.weight_dict[f"loss_bbox_{i}"] = self.weight_dict["loss_bbox"] * self.aux_loss_weight
            self.weight_dict[f"loss_giou_{i}"] = self.weight_dict["loss_giou"] * self.aux_loss_weight

    def _prepare_targets(self, targets, device):
        out = []
        for t in targets:
            nt = dict(t)
            boxes_xyxy = normalize_boxes_xyxy(nt["boxes"].to(device), nt["size"].to(device))
            nt["boxes_norm_cxcywh"] = box_xyxy_to_cxcywh(boxes_xyxy).clamp(0, 1)
            nt["area_norm"] = nt["area"].to(device) / nt["size"].prod().to(device).float().clamp(min=1)
            out.append(nt)
        return out

    def forward(self, outputs, targets):
        targets = self._prepare_targets(targets, outputs["pred_logits"].device)
        indices = self.matcher(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs["pred_logits"].device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_boxes)
            num_boxes = num_boxes / dist.get_world_size()
        num_boxes = num_boxes.clamp(min=1).item()
        loss_ce = loss_labels(outputs, targets, indices, num_boxes, self.num_classes, self.empty_weight)
        loss_bbox, loss_giou = loss_boxes(outputs, targets, indices, num_boxes)
        loss_dict = {"loss_ce": loss_ce, "loss_bbox": loss_bbox, "loss_giou": loss_giou}
        acq_outputs = dict(outputs)
        acq_loss, acq_stats = self.acq(acq_outputs, targets, indices)
        loss_dict["loss_acq"] = acq_loss
        loss_dict.update(acq_stats)
        aux_outputs = outputs.get("aux_outputs", [])
        if isinstance(self.aux_loss_layers, int):
            start_layer = max(0, len(aux_outputs) - self.aux_loss_layers)
        else:
            start_layer = 0
        for i, aux in enumerate(aux_outputs):
            if i < start_layer:
                continue
            aux_indices = self.matcher(aux, targets)
            l_ce = loss_labels(aux, targets, aux_indices, num_boxes, self.num_classes, self.empty_weight)
            l_bbox, l_giou = loss_boxes(aux, targets, aux_indices, num_boxes)
            loss_dict[f"loss_ce_{i}"] = l_ce
            loss_dict[f"loss_bbox_{i}"] = l_bbox
            loss_dict[f"loss_giou_{i}"] = l_giou
        return loss_dict


def build_criterion(cfg):
    return SetCriterion(cfg)
