"""Compatibility helpers for optional MMDetection imports."""

from __future__ import annotations


class _NoopRegistry:
    def register_module(self, *args, **kwargs):
        def decorator(cls):
            return cls

        if args and isinstance(args[0], type):
            return args[0]
        return decorator


try:  # pragma: no cover - exercised in the target MMDetection environment.
    from mmdet.registry import DATA_SAMPLERS, MODELS
except Exception:  # pragma: no cover - local tests can run without mmdet.
    DATA_SAMPLERS = _NoopRegistry()
    MODELS = _NoopRegistry()
