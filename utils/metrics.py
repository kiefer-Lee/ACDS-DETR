import torch

from .box_ops import box_cxcywh_to_xyxy, box_iou, denormalize_boxes_xyxy


@torch.no_grad()
def postprocess(outputs, targets, score_thresh=0.05, max_detections=100):
    probs = outputs["pred_logits"].softmax(-1)[..., :-1]
    scores, labels = probs.max(-1)
    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"]).clamp(0, 1)
    results = []
    for b in range(boxes.shape[0]):
        keep = scores[b] > score_thresh
        s = scores[b, keep]
        l = labels[b, keep]
        bx = boxes[b, keep]
        if s.numel() > max_detections:
            top = s.topk(max_detections).indices
            s, l, bx = s[top], l[top], bx[top]
        bx = denormalize_boxes_xyxy(bx, targets[b]["orig_size"].to(bx.device))
        results.append({"scores": s.cpu(), "labels": l.cpu(), "boxes": bx.cpu(), "image_id": int(targets[b]["image_id"].item())})
    return results


class DetectionMetrics:
    def __init__(self, num_classes, iou_thresholds=None, small_area_thr=32 * 32):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5 + 0.05 * i for i in range(10)]
        self.small_area_thr = small_area_thr
        self.preds = []
        self.targets = []

    def update(self, preds, targets):
        self.preds.extend(preds)
        for t in targets:
            self.targets.append({
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu(),
                "area": t["area"].cpu(),
                "image_id": int(t["image_id"].item()),
            })

    def compute(self):
        ap_by_thr = []
        ap_small_by_thr = []
        p50, r50 = 0.0, 0.0
        for thr in self.iou_thresholds:
            ap, precision, recall = self._ap_at_iou(thr, small_only=False)
            ap_s, _, _ = self._ap_at_iou(thr, small_only=True)
            ap_by_thr.append(ap)
            ap_small_by_thr.append(ap_s)
            if abs(thr - 0.5) < 1e-6:
                p50, r50 = precision, recall
        return {
            "mAP": float(sum(ap_by_thr) / max(1, len(ap_by_thr))),
            "mAP50_95": float(sum(ap_by_thr) / max(1, len(ap_by_thr))),
            "AP50": float(ap_by_thr[0]) if ap_by_thr else 0.0,
            "AP_small": float(sum(ap_small_by_thr) / max(1, len(ap_small_by_thr))),
            "precision": float(p50),
            "recall": float(r50),
        }

    def _ap_at_iou(self, thr, small_only=False):
        aps = []
        precisions, recalls = [], []
        gt_by_img_cls = {}
        npos = 0
        for t in self.targets:
            boxes, labels, areas = t["boxes"], t["labels"], t["area"]
            if small_only:
                keep = areas < self.small_area_thr
                boxes, labels = boxes[keep], labels[keep]
            for c in range(self.num_classes):
                keep = labels == c
                if keep.any():
                    gt_by_img_cls[(t["image_id"], c)] = {"boxes": boxes[keep], "used": torch.zeros(int(keep.sum()), dtype=torch.bool)}
                    npos += int(keep.sum())
        if npos == 0:
            return 0.0, 0.0, 0.0
        for c in range(self.num_classes):
            cls_preds = []
            for p in self.preds:
                keep = p["labels"] == c
                for s, b in zip(p["scores"][keep], p["boxes"][keep]):
                    cls_preds.append((float(s), p["image_id"], b))
            cls_preds.sort(key=lambda x: x[0], reverse=True)
            tp = torch.zeros(len(cls_preds))
            fp = torch.zeros(len(cls_preds))
            cls_npos = sum(v["boxes"].shape[0] for (img, cc), v in gt_by_img_cls.items() if cc == c)
            if cls_npos == 0:
                continue
            for i, (_, img_id, box) in enumerate(cls_preds):
                entry = gt_by_img_cls.get((img_id, c))
                if entry is None or entry["boxes"].numel() == 0:
                    fp[i] = 1
                    continue
                ious = box_iou(box[None, :], entry["boxes"])[0][0]
                best_iou, best_idx = ious.max(0)
                if best_iou >= thr and not entry["used"][best_idx]:
                    tp[i] = 1
                    entry["used"][best_idx] = True
                else:
                    fp[i] = 1
            if len(cls_preds) == 0:
                aps.append(0.0)
                continue
            tp_cum = torch.cumsum(tp, 0)
            fp_cum = torch.cumsum(fp, 0)
            rec = tp_cum / max(1, cls_npos)
            prec = tp_cum / (tp_cum + fp_cum).clamp(min=1e-6)
            aps.append(voc_ap(rec, prec))
            precisions.append(float(prec[-1]))
            recalls.append(float(rec[-1]))
        return (
            sum(aps) / max(1, len(aps)),
            sum(precisions) / max(1, len(precisions)),
            sum(recalls) / max(1, len(recalls)),
        )


def voc_ap(rec, prec):
    mrec = torch.cat([torch.tensor([0.0]), rec, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), prec, torch.tensor([0.0])])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    idx = (mrec[1:] != mrec[:-1]).nonzero().flatten()
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())

