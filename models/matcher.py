import torch
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        small_object_cost_gain=0.0,
        small_area_thr=1024,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.small_object_cost_gain = float(small_object_cost_gain)
        self.small_area_thr = float(small_area_thr)

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]
        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes_norm_cxcywh"]
            if tgt_ids.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.int64, device=out_bbox.device), torch.empty(0, dtype=torch.int64, device=out_bbox.device)))
                continue
            cost_class = -out_prob[b][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox))
            if self.small_object_cost_gain > 0 and "area" in targets[b]:
                small = (targets[b]["area"].to(out_bbox.device) < self.small_area_thr).float()
                weights = 1.0 + self.small_object_cost_gain * small
                cost = self.cost_class * cost_class + weights[None, :] * (self.cost_bbox * cost_bbox + self.cost_giou * cost_giou)
            else:
                cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            src, tgt = linear_sum_assignment_torch(cost)
            indices.append((src.to(out_bbox.device), tgt.to(out_bbox.device)))
        return indices


def linear_sum_assignment_torch(cost):
    try:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        return torch.as_tensor(row, dtype=torch.int64), torch.as_tensor(col, dtype=torch.int64)
    except Exception:
        c = cost.detach().cpu()
        rows, cols = [], []
        used_r, used_c = set(), set()
        flat = torch.argsort(c.flatten())
        n, m = c.shape
        for idx in flat.tolist():
            r, col = divmod(idx, m)
            if r not in used_r and col not in used_c:
                rows.append(r)
                cols.append(col)
                used_r.add(r)
                used_c.add(col)
                if len(cols) == min(n, m):
                    break
        return torch.as_tensor(rows, dtype=torch.int64), torch.as_tensor(cols, dtype=torch.int64)
