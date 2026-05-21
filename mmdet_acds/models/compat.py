"""Small compatibility helpers for optional MMDetection imports."""

from __future__ import annotations


class _NoopRegistry:
    def register_module(self, *args, **kwargs):
        def decorator(cls):
            return cls

        if args and isinstance(args[0], type):
            return args[0]
        return decorator


try:  # pragma: no cover - exercised when MMDetection is installed.
    from mmdet.registry import DATASETS, METRICS, MODELS, TASK_UTILS, TRANSFORMS
except Exception:  # pragma: no cover - local unit tests can run without mmdet.
    MODELS = TASK_UTILS = DATASETS = TRANSFORMS = METRICS = _NoopRegistry()

