import torch
import torch.nn.functional as F

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def get_tgt_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


def loss_labels(outputs, targets, indices, num_boxes, num_classes, empty_weight):
    src_logits = outputs["pred_logits"]
    idx = get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)]) if len(idx[0]) else torch.empty(0, dtype=torch.int64, device=src_logits.device)
    target_classes = torch.full(src_logits.shape[:2], num_classes, dtype=torch.int64, device=src_logits.device)
    if len(idx[0]):
        target_classes[idx] = target_classes_o
    return F.cross_entropy(src_logits.transpose(1, 2), target_classes, empty_weight)


def loss_boxes(outputs, targets, indices, num_boxes, small_object_loss_gain=0.0, small_area_thr=1024):
    idx = get_src_permutation_idx(indices)
    if len(idx[0]) == 0:
        z = outputs["pred_boxes"].sum() * 0.0
        return z, z
    src_boxes = outputs["pred_boxes"][idx]
    target_boxes = torch.cat([t["boxes_norm_cxcywh"][i] for t, (_, i) in zip(targets, indices)], dim=0)
    box_loss = F.l1_loss(src_boxes, target_boxes, reduction="none").sum(-1)
    loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
    if small_object_loss_gain > 0:
        target_areas = torch.cat([t["area"][i].to(src_boxes.device) for t, (_, i) in zip(targets, indices)], dim=0)
        weights = 1.0 + float(small_object_loss_gain) * (target_areas < float(small_area_thr)).float()
        box_loss = box_loss * weights
        loss_giou = loss_giou * weights
    loss_bbox = box_loss.sum() / num_boxes
    loss_giou = loss_giou.sum() / num_boxes
    return loss_bbox, loss_giou
