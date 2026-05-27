"""Save validation metric curves over training epochs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

try:  # pragma: no cover - requires MMEngine at training time.
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS
except Exception:  # pragma: no cover - lets local unit tests import without MMEngine.
    from mmdet_acds.models.compat import _NoopRegistry

    class Hook:  # type: ignore[no-redef]
        pass

    HOOKS = _NoopRegistry()


@HOOKS.register_module()
class MetricCurveHook(Hook):
    """Collect validation metrics and plot their epoch-wise curves.

    The hook writes a CSV after every validation epoch and draws a PNG at the
    end of training. The CSV is still useful if a run is stopped early.
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        out_dir: str = "metric_curves",
        csv_name: str = "metrics_by_epoch.csv",
        figure_name: str = "metrics_by_epoch.png",
        draw: bool = True,
        **kwargs,
    ) -> None:
        self.metrics = metrics or [
            "coco/bbox_mAP",
            "coco/bbox_mAP_50",
            "coco/bbox_mAP_75",
            "coco/bbox_mAP_s",
            "coco/bbox_mAP_m",
            "coco/bbox_mAP_l",
        ]
        self.out_dir = out_dir
        self.csv_name = csv_name
        self.figure_name = figure_name
        self.draw = draw
        self.history: list[dict[str, float | int]] = []

    def after_val_epoch(self, runner, metrics: Mapping[str, float] | None = None) -> None:
        if getattr(runner, "rank", 0) != 0 or not metrics:
            return

        epoch = int(getattr(runner, "epoch", 0)) + 1
        record: dict[str, float | int] = {"epoch": epoch}
        for name in self.metrics:
            value = self._get_metric(metrics, name)
            if value is not None:
                record[name] = value

        if len(record) == 1:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    record[name] = float(value)

        if len(record) == 1:
            return

        self._upsert_record(record)
        self._dump_csv(runner)

    def after_train(self, runner) -> None:
        if getattr(runner, "rank", 0) != 0 or not self.history:
            return

        self._dump_csv(runner)
        if self.draw:
            self._draw_figure(runner)

    def _get_metric(self, metrics: Mapping[str, float], name: str) -> float | None:
        candidates = [name]
        if "/" in name:
            candidates.append(name.split("/", 1)[1])
        else:
            candidates.append(f"coco/{name}")

        for candidate in candidates:
            value = metrics.get(candidate)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _upsert_record(self, record: dict[str, float | int]) -> None:
        for idx, old_record in enumerate(self.history):
            if old_record["epoch"] == record["epoch"]:
                self.history[idx] = record
                return
        self.history.append(record)
        self.history.sort(key=lambda item: int(item["epoch"]))

    def _output_dir(self, runner) -> Path:
        out_dir = Path(getattr(runner, "work_dir", ".")) / self.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _dump_csv(self, runner) -> None:
        keys = ["epoch"]
        for record in self.history:
            for key in record:
                if key not in keys:
                    keys.append(key)

        path = self._output_dir(runner) / self.csv_name
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.history)

    def _draw_figure(self, runner) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - depends on runtime image.
            runner.logger.warning(
                "MetricCurveHook skipped figure drawing because matplotlib "
                f"is unavailable: {exc}"
            )
            return

        epochs = [int(record["epoch"]) for record in self.history]
        metric_names = [key for key in self.history[0] if key != "epoch"]
        if not epochs or not metric_names:
            return

        plt.figure(figsize=(10, 6), dpi=160)
        for name in metric_names:
            values = [record.get(name) for record in self.history]
            valid_points = [(epoch, value) for epoch, value in zip(epochs, values) if value is not None]
            if not valid_points:
                continue
            xs, ys = zip(*valid_points)
            plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label=name)

        plt.xlabel("Epoch")
        plt.ylabel("Metric")
        plt.title("Validation Metrics by Epoch")
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.savefig(self._output_dir(runner) / self.figure_name)
        plt.close()
