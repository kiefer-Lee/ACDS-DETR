import contextlib
import io


def coco_evaluate(preds, targets, num_classes):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception:
        return None

    images, annotations, detections = [], [], []
    ann_id = 1
    for target in targets:
        image_id = int(target["image_id"])
        h, w = target.get("size", target.get("orig_size")).tolist()
        images.append({"id": image_id, "height": int(h), "width": int(w)})
        iscrowd_values = target.get("iscrowd")
        if iscrowd_values is None:
            iscrowd_values = target["labels"].new_zeros(target["labels"].shape)
        for box, label, area, iscrowd in zip(target["boxes"].tolist(), target["labels"].tolist(), target["area"].tolist(), iscrowd_values.tolist()):
            x0, y0, x1, y1 = box
            bw = max(0.0, x1 - x0)
            bh = max(0.0, y1 - y0)
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(label) + 1,
                "bbox": [float(x0), float(y0), float(bw), float(bh)],
                "area": float(area),
                "iscrowd": int(iscrowd),
            })
            ann_id += 1
    for pred in preds:
        image_id = int(pred["image_id"])
        for box, label, score in zip(pred["boxes"].tolist(), pred["labels"].tolist(), pred["scores"].tolist()):
            x0, y0, x1, y1 = box
            detections.append({
                "image_id": image_id,
                "category_id": int(label) + 1,
                "bbox": [float(x0), float(y0), float(max(0.0, x1 - x0)), float(max(0.0, y1 - y0))],
                "score": float(score),
            })
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {"description": "ACDS-DETR in-memory evaluation"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i + 1, "name": str(i + 1)} for i in range(num_classes)],
    }
    if not detections:
        return {"mAP": 0.0, "mAP50_95": 0.0, "AP50": 0.0, "AP75": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0, "AR@1": 0.0, "AR@10": 0.0, "AR@100": 0.0, "AR_small": 0.0, "AR_medium": 0.0, "AR_large": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(detections) if detections else coco_gt.loadRes([])
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = evaluator.stats
    return {
        "mAP": float(stats[0]),
        "mAP50_95": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR@1": float(stats[6]),
        "AR@10": float(stats[7]),
        "AR@100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }
