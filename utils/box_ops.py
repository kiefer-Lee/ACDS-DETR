import torch


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area.clamp(min=1e-6)


def normalize_boxes_xyxy(boxes, size):
    h, w = size
    scale = boxes.new_tensor([w, h, w, h])
    return boxes / scale


def denormalize_boxes_xyxy(boxes, size):
    h, w = size
    scale = boxes.new_tensor([w, h, w, h])
    return boxes * scale

