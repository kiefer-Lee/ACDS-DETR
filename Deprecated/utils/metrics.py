import torch

from .box_ops import box_cxcywh_to_xyxy, box_iou, denormalize_boxes_xyxy


@torch.no_grad()
def postprocess(outputs, targets, score_thresh=0.05, max_detections=100, min_detections=0):
    probs = outputs["pred_logits"].softmax(-1)[..., :-1]
    scores, labels = probs.max(-1)
    boxes = box_cxcywh_to_xyxy(outputs["pred_boxes"]).clamp(0, 1)
    results = []
    for b in range(boxes.shape[0]):
        keep = scores[b] > score_thresh
        if int(keep.sum()) < int(min_detections):
            k = min(scores.shape[1], max(int(min_detections), int(max_detections)))
            keep = torch.zeros_like(scores[b], dtype=torch.bool)
            keep[scores[b].topk(k).indices] = True
        s = scores[b, keep]
        l = labels[b, keep]
        bx = boxes[b, keep]
        if s.numel() > max_detections:
            top = s.topk(max_detections).indices
            s, l, bx = s[top], l[top], bx[top]
        out_size = targets[b].get("size", targets[b]["orig_size"]).to(bx.device)
        bx = denormalize_boxes_xyxy(bx, out_size)
        results.append({"scores": s.cpu(), "labels": l.cpu(), "boxes": bx.cpu(), "image_id": int(targets[b]["image_id"].item())})
    return results


class DetectionMetrics:
    def __init__(self, num_classes, iou_thresholds=None, small_area_thr=32 * 32, dense_small_count_thr=20):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5 + 0.05 * i for i in range(10)]
        self.small_area_thr = small_area_thr
        self.dense_small_count_thr = dense_small_count_thr
        self.preds = []
        self.targets = []

    def update(self, preds, targets):
        self.preds.extend(preds)
        for t in targets:
            self.targets.append({
                "boxes": t["boxes"].cpu(),
                "labels": t["labels"].cpu(),
                "area": t["area"].cpu(),
                "iscrowd": t.get("iscrowd", torch.zeros_like(t["labels"])).cpu(),
                "size": t.get("size", t["orig_size"]).cpu(),
                "orig_size": t["orig_size"].cpu(),
                "image_id": int(t["image_id"].item()),
            })

    def compute(self):
        ap_by_thr = []
        ap_small_by_thr = []
        ap_medium_by_thr = []
        ap_large_by_thr = []
        p50, r50 = 0.0, 0.0
        ap75 = 0.0
        for thr in self.iou_thresholds:
            ap, precision, recall = self._ap_at_iou(thr, small_only=False)
            ap_s, _, _ = self._ap_at_iou(thr, small_only=True)
            ap_m, _, _ = self._ap_at_iou(thr, area_range=(self.small_area_thr, 96 * 96))
            ap_l, _, _ = self._ap_at_iou(thr, area_range=(96 * 96, float("inf")))
            ap_by_thr.append(ap)
            ap_small_by_thr.append(ap_s)
            ap_medium_by_thr.append(ap_m)
            ap_large_by_thr.append(ap_l)
            if abs(thr - 0.5) < 1e-6:
                p50, r50 = precision, recall
            if abs(thr - 0.75) < 1e-6:
                ap75 = ap
        return {
            "mAP": float(sum(ap_by_thr) / max(1, len(ap_by_thr))),
            "mAP50_95": float(sum(ap_by_thr) / max(1, len(ap_by_thr))),
            "AP50": float(ap_by_thr[0]) if ap_by_thr else 0.0,
            "AP75": float(ap75),
            "AP_small": float(sum(ap_small_by_thr) / max(1, len(ap_small_by_thr))),
            "AP_medium": float(sum(ap_medium_by_thr) / max(1, len(ap_medium_by_thr))),
            "AP_large": float(sum(ap_large_by_thr) / max(1, len(ap_large_by_thr))),
            "APs": float(sum(ap_small_by_thr) / max(1, len(ap_small_by_thr))),
            "APm": float(sum(ap_medium_by_thr) / max(1, len(ap_medium_by_thr))),
            "APl": float(sum(ap_large_by_thr) / max(1, len(ap_large_by_thr))),
            "precision": float(p50),
            "recall": float(r50),
            "AR@1": self._average_recall(1),
            "AR@10": self._average_recall(10),
            "AR@100": self._average_recall(100),
            "AR_small": self._average_recall(100, area_range=(0, self.small_area_thr)),
            "AR_medium": self._average_recall(100, area_range=(self.small_area_thr, 96 * 96)),
            "AR_large": self._average_recall(100, area_range=(96 * 96, float("inf"))),
            "ARs": self._average_recall(100, area_range=(0, self.small_area_thr)),
            "ARm": self._average_recall(100, area_range=(self.small_area_thr, 96 * 96)),
            "ARl": self._average_recall(100, area_range=(96 * 96, float("inf"))),
            **self._dense_subset_metrics(),
        }

    def _dense_subset_metrics(self):
        dense_ids = []
        for t in self.targets:
            small_count = int((t["area"] < self.small_area_thr).sum())
            if small_count >= self.dense_small_count_thr:
                dense_ids.append(t["image_id"])
        if not dense_ids:
            return {"Dense_images": 0.0, "Dense_AP_small": 0.0, "Dense_AR_small": 0.0}
        ap_small = [self._ap_at_iou(thr, small_only=True, image_ids=set(dense_ids))[0] for thr in self.iou_thresholds]
        ar_small = self._average_recall(100, area_range=(0, self.small_area_thr), image_ids=set(dense_ids))
        return {
            "Dense_images": float(len(dense_ids)),
            "Dense_AP_small": float(sum(ap_small) / max(1, len(ap_small))),
            "Dense_AR_small": float(ar_small),
        }

    def _ap_at_iou(self, thr, small_only=False, image_ids=None, area_range=None):
        aps = []
        precisions, recalls = [], []
        gt_by_img_cls = {}
        npos = 0
        for t in self.targets:
            if image_ids is not None and t["image_id"] not in image_ids:
                continue
            boxes, labels, areas = t["boxes"], t["labels"], t["area"]
            if small_only:
                area_range = (0, self.small_area_thr)
            if area_range is not None:
                keep = (areas >= area_range[0]) & (areas < area_range[1])
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
                if image_ids is not None and p["image_id"] not in image_ids:
                    continue
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

    def _average_recall(self, max_dets=100, area_range=None, image_ids=None):
        recalls = [self._recall_at_iou(thr, max_dets, area_range, image_ids) for thr in self.iou_thresholds]
        return float(sum(recalls) / max(1, len(recalls)))

    def _recall_at_iou(self, thr, max_dets=100, area_range=None, image_ids=None):
        total_gt = 0
        matched_gt = 0
        preds_by_img = {p["image_id"]: p for p in self.preds if image_ids is None or p["image_id"] in image_ids}
        for t in self.targets:
            if image_ids is not None and t["image_id"] not in image_ids:
                continue
            gt_boxes, gt_labels, gt_areas = t["boxes"], t["labels"], t["area"]
            if area_range is not None:
                keep = (gt_areas >= area_range[0]) & (gt_areas < area_range[1])
                gt_boxes, gt_labels = gt_boxes[keep], gt_labels[keep]
            total_gt += int(gt_boxes.shape[0])
            if gt_boxes.numel() == 0:
                continue
            pred = preds_by_img.get(t["image_id"])
            if pred is None or pred["boxes"].numel() == 0:
                continue
            order = pred["scores"].sort(descending=True).indices[:max_dets]
            pred_boxes = pred["boxes"][order]
            pred_labels = pred["labels"][order]
            used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
            for box, label in zip(pred_boxes, pred_labels):
                keep = gt_labels == label
                if not keep.any():
                    continue
                gt_idx = keep.nonzero().flatten()
                ious = box_iou(box[None, :], gt_boxes[gt_idx])[0][0]
                best_iou, local_idx = ious.max(0)
                real_idx = gt_idx[local_idx]
                if best_iou >= thr and not used[real_idx]:
                    used[real_idx] = True
            matched_gt += int(used.sum())
        return matched_gt / max(1, total_gt)


def voc_ap(rec, prec):
    mrec = torch.cat([torch.tensor([0.0]), rec, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), prec, torch.tensor([0.0])])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    idx = (mrec[1:] != mrec[:-1]).nonzero().flatten()
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())
