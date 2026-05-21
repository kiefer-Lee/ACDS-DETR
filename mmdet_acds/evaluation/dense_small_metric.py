"""Dense-small subset metric for VisDrone-style small-object analysis."""

from __future__ import annotations

from typing import Any

from mmdet_acds.models.compat import METRICS

try:  # pragma: no cover - requires mmdet.
    from mmdet.evaluation.metrics import CocoMetric
except Exception:  # pragma: no cover
    CocoMetric = object


@METRICS.register_module()
class DenseSmallCocoMetric(CocoMetric):
    """CocoMetric wrapper carrying dense-small config.

    The first migration version uses standard mmdet COCO evaluation for fair
    comparison. This class records the dense subset threshold so later reports
    can add dense-small summaries without changing config names.
    """

    def __init__(self, dense_small_count_thr: int = 20, small_area_thr: float = 1024.0, **kwargs: Any) -> None:
        if CocoMetric is object:
            self.dense_small_count_thr = int(dense_small_count_thr)
            self.small_area_thr = float(small_area_thr)
            return
        super().__init__(**kwargs)
        self.dense_small_count_thr = int(dense_small_count_thr)
        self.small_area_thr = float(small_area_thr)

