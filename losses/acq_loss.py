import torch
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, box_iou


class ACQLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enabled = cfg["enabled"]
        self.small_area_thr = cfg["small_area_thr"]
        self.topk_unmatched = cfg["topk_unmatched"]
        self.delta = cfg["delta"]
        self.sigma = cfg["sigma"]
        self.min_score = cfg["min_score"]

    def forward(self, outputs, targets, indices):
        device = outputs["pred_boxes"].device
        if not self.enabled:
            return outputs["pred_boxes"].sum() * 0.0, {"query_collision_rate": torch.tensor(0.0, device=device)}
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        refs = outputs.get("reference_points", None)
        if isinstance(refs, list):
            refs = refs[-1]
        if refs is None:
            refs = boxes[..., :2]
        probs = logits.softmax(-1)[..., :-1]
        scores = probs.max(-1).values
        total_loss = boxes.sum() * 0.0
        total_pairs = 0
        total_small = 0
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if tgt_idx.numel() == 0:
                continue
            tgt_area = targets[b]["area_norm"][tgt_idx]
            small_thr_norm = self.small_area_thr / float(targets[b]["size"].prod().item())
            small_mask = tgt_area < small_thr_norm
            if small_mask.sum() == 0:
                continue
            matched_small_src = src_idx[small_mask]
            total_small += int(matched_small_src.numel())
            matched_set = torch.zeros(logits.shape[1], dtype=torch.bool, device=device)
            matched_set[src_idx] = True
            unmatched = (~matched_set).nonzero().flatten()
            if unmatched.numel() == 0:
                continue
            valid_unmatched = unmatched[scores[b, unmatched] > self.min_score]
            if valid_unmatched.numel() > self.topk_unmatched:
                top = scores[b, valid_unmatched].topk(self.topk_unmatched).indices
                valid_unmatched = valid_unmatched[top]
            if valid_unmatched.numel() == 0:
                continue
            r_i = refs[b, valid_unmatched, :2]
            r_j = refs[b, matched_small_src, :2]
            dist = torch.cdist(r_i, r_j, p=2)
            pred_i = box_cxcywh_to_xyxy(boxes[b, valid_unmatched])
            pred_j = box_cxcywh_to_xyxy(boxes[b, matched_small_src])
            iou = box_iou(pred_i, pred_j)[0]
            close = (dist < self.delta) | (iou > 0.3)
            if close.sum() == 0:
                continue
            s_i = scores[b, valid_unmatched][:, None]
            s_j = scores[b, matched_small_src][None, :]
            m_j = torch.exp(-tgt_area[small_mask][None, :] / (small_thr_norm + 1e-6))
            collision = s_i * s_j * torch.exp(-dist / self.sigma) * m_j
            loss = collision * torch.relu(self.delta - dist)
            total_loss = total_loss + loss[close].mean()
            total_pairs += int(close.sum().item())
        denom = max(1, len(indices))
        qcr = torch.tensor(total_pairs / max(1, total_small), device=device, dtype=torch.float32)
        return total_loss / denom, {"query_collision_rate": qcr}
