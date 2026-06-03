"""Batch sampler that isolates annotation-dense images."""

from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Any

from mmdet_comparison.models.compat import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class DensityAwareBatchSampler:
    """Keep normal images in large batches and put dense images in small ones."""

    def __init__(
        self,
        sampler: Sized,
        batch_size: int,
        drop_last: bool = False,
        dense_threshold: int = 150,
        dense_batch_size: int = 1,
        dataset: Any | None = None,
    ) -> None:
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.dense_threshold = int(dense_threshold)
        self.dense_batch_size = int(dense_batch_size)
        self.dataset = dataset if dataset is not None else getattr(sampler, "dataset", None)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.dense_batch_size <= 0:
            raise ValueError("dense_batch_size must be positive.")
        self._count_cache: dict[int, int] = {}

    def __iter__(self) -> Iterator[list[int]]:
        normal_batch: list[int] = []
        dense_batch: list[int] = []
        for idx in self.sampler:
            idx = int(idx)
            if self._object_count(idx) >= self.dense_threshold:
                if normal_batch:
                    yield normal_batch
                    normal_batch = []
                dense_batch.append(idx)
                if len(dense_batch) == self.dense_batch_size:
                    yield dense_batch
                    dense_batch = []
            else:
                normal_batch.append(idx)
                if len(normal_batch) == self.batch_size:
                    yield normal_batch
                    normal_batch = []
        if dense_batch and not self.drop_last:
            yield dense_batch
        if normal_batch and not self.drop_last:
            yield normal_batch

    def __len__(self) -> int:
        dense = 0
        normal = 0
        for idx in range(len(self.sampler)):
            if self._object_count(idx) >= self.dense_threshold:
                dense += 1
            else:
                normal += 1
        dense_batches = dense // self.dense_batch_size
        normal_batches = normal // self.batch_size
        if not self.drop_last:
            dense_batches += int(dense % self.dense_batch_size > 0)
            normal_batches += int(normal % self.batch_size > 0)
        return dense_batches + normal_batches

    def _object_count(self, idx: int) -> int:
        if idx not in self._count_cache:
            self._count_cache[idx] = self._read_object_count(idx)
        return self._count_cache[idx]

    def _read_object_count(self, idx: int) -> int:
        if self.dataset is None:
            return 0
        data_info = self._get_data_info(idx)
        if isinstance(data_info, dict):
            for key in ("instances", "annotations"):
                value = data_info.get(key)
                if isinstance(value, list):
                    return sum(1 for item in value if not self._is_ignored(item))
            ann_info = data_info.get("ann_info")
            if isinstance(ann_info, dict):
                return self._count_ann_info(ann_info)
        if hasattr(self.dataset, "get_ann_info"):
            return self._count_ann_info(self.dataset.get_ann_info(idx))
        return 0

    def _get_data_info(self, idx: int) -> Any:
        if hasattr(self.dataset, "get_data_info"):
            return self.dataset.get_data_info(idx)
        data_list = getattr(self.dataset, "data_list", None)
        if data_list is not None:
            return data_list[idx]
        return None

    @staticmethod
    def _count_ann_info(ann_info: Any) -> int:
        if not isinstance(ann_info, dict):
            return 0
        bboxes = ann_info.get("bboxes")
        if bboxes is not None:
            return len(bboxes)
        labels = ann_info.get("labels")
        return len(labels) if labels is not None else 0

    @staticmethod
    def _is_ignored(instance: Any) -> bool:
        return isinstance(instance, dict) and bool(instance.get("ignore_flag", instance.get("ignore", False)))
