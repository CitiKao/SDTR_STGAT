from __future__ import annotations

from typing import Any

import numpy as np


REPORT_TASKS = ("demand", "supply", "gap")
REPORT_METRICS = ("mae", "rmse", "mape")


def _new_bucket() -> dict[str, float]:
    return {"se": 0.0, "ae": 0.0, "ape": 0.0, "count": 0.0, "mape_count": 0.0}


def _finalize_bucket(bucket: dict[str, float]) -> dict[str, float]:
    count = max(float(bucket["count"]), 1.0)
    mse = float(bucket["se"]) / count
    mae = float(bucket["ae"]) / count
    mape_count = max(float(bucket["mape_count"]), 1.0)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mae),
        "mape": float(100.0 * float(bucket["ape"]) / mape_count),
        "count": float(bucket["count"]),
    }


def build_horizon_reports(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return one comparable metric block per requested forecast horizon.

    The horizon labels are prediction-ahead slots, e.g. ``30min`` means the
    model's second 15-minute-ahead prediction when the benchmark slot is 15
    minutes. It is not a 30-minute aggregate bin.
    """

    reports: dict[str, dict[str, Any]] = {}
    report_horizons = metrics.get("report_horizons", {})
    for minute, step in zip(
        report_horizons.get("resolved_minutes", []),
        report_horizons.get("resolved_steps", []),
    ):
        label = f"{int(minute)}min"
        row: dict[str, Any] = {"minutes": int(minute), "step": int(step)}
        for task in REPORT_TASKS:
            task_report = metrics.get(task, {}).get("report", {})
            values = task_report.get(label)
            if not values:
                continue
            row[task] = {
                metric: float(values[metric])
                for metric in REPORT_METRICS
                if metric in values
            }
            if "count" in values:
                row[task]["count"] = float(values["count"])
        if "demand" in row and "supply" in row:
            row["raw_dc"] = float(row["demand"]["rmse"] + row["supply"]["rmse"])
        reports[label] = row
    return reports


def build_horizon_metric_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten horizon reports into rows that are easy to write to CSV."""

    reports = metrics.get("horizon_reports") or build_horizon_reports(metrics)
    rows: list[dict[str, Any]] = []
    for label, horizon_report in reports.items():
        for task in REPORT_TASKS:
            values = horizon_report.get(task)
            if not values:
                continue
            rows.append(
                {
                    "horizon": label,
                    "minutes": int(horizon_report["minutes"]),
                    "step": int(horizon_report["step"]),
                    "target": task,
                    "mae": float(values["mae"]),
                    "rmse": float(values["rmse"]),
                    "mape": float(values["mape"]),
                    "count": float(values.get("count", 0.0)),
                    "raw_dc_horizon": (
                        float(horizon_report["raw_dc"])
                        if "raw_dc" in horizon_report
                        else None
                    ),
                }
            )
    return rows


class DCMetricAccumulator:
    """Streaming raw-scale D/C metric accumulator.

    Expected prediction and target shape: ``(batch, nodes, horizon, 2)`` where
    channel 0 is demand and channel 1 is supply/capacity.
    """

    def __init__(
        self,
        *,
        pred_horizon: int,
        report_horizons: dict[str, Any],
        target_stats: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.pred_horizon = int(pred_horizon)
        self.report_horizons = dict(report_horizons)
        self.target_stats = target_stats or {}
        self._buckets = {
            task: {
                "overall": _new_bucket(),
                "per_step": [_new_bucket() for _ in range(self.pred_horizon)],
            }
            for task in REPORT_TASKS
        }

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if pred.shape != target.shape:
            raise ValueError(f"Prediction shape {pred.shape} must match target shape {target.shape}.")
        if pred.ndim != 4 or pred.shape[-1] != 2:
            raise ValueError("Expected shape (batch, nodes, horizon, 2).")
        if pred.shape[2] != self.pred_horizon:
            raise ValueError(f"Expected horizon {self.pred_horizon}, got {pred.shape[2]}.")

        comparisons = {
            "demand": (pred[..., 0], target[..., 0]),
            "supply": (pred[..., 1], target[..., 1]),
            "gap": (pred[..., 0] - pred[..., 1], target[..., 0] - target[..., 1]),
        }
        for name, (task_pred, task_target) in comparisons.items():
            self._update_task(name, task_pred, task_target)

    def _update_task(self, name: str, pred: np.ndarray, target: np.ndarray) -> None:
        diff = pred - target
        mask = np.isfinite(pred) & np.isfinite(target)
        sq = np.square(diff)
        abs_diff = np.abs(diff)
        overall = self._buckets[name]["overall"]
        overall["se"] += float(sq[mask].sum())
        overall["ae"] += float(abs_diff[mask].sum())
        overall["count"] += float(mask.sum())
        mape_mask = mask & (np.abs(target) > 1e-6)
        overall["ape"] += float((abs_diff[mape_mask] / np.abs(target[mape_mask])).sum())
        overall["mape_count"] += float(mape_mask.sum())

        for step_idx in range(self.pred_horizon):
            step_mask = mask[..., step_idx]
            step_target = target[..., step_idx]
            step_abs = abs_diff[..., step_idx]
            bucket = self._buckets[name]["per_step"][step_idx]
            bucket["se"] += float(sq[..., step_idx][step_mask].sum())
            bucket["ae"] += float(step_abs[step_mask].sum())
            bucket["count"] += float(step_mask.sum())
            step_mape_mask = step_mask & (np.abs(step_target) > 1e-6)
            bucket["ape"] += float((step_abs[step_mape_mask] / np.abs(step_target[step_mape_mask])).sum())
            bucket["mape_count"] += float(step_mape_mask.sum())

    def finalize(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {"report_horizons": self.report_horizons}
        slot_minutes = int(self.report_horizons.get("slot_minutes", 15))
        for task in REPORT_TASKS:
            overall = _finalize_bucket(self._buckets[task]["overall"])
            if task in self.target_stats:
                raw_std = float(self.target_stats[task].get("raw_std", 0.0))
                raw_mean = float(self.target_stats[task].get("raw_mean", 0.0))
                overall["nrmse_std_train"] = (
                    float(overall["rmse"] / raw_std) if raw_std > 1e-12 else None
                )
                overall["nrmse_mean_train"] = (
                    float(overall["rmse"] / raw_mean) if abs(raw_mean) > 1e-12 else None
                )
            per_step: dict[str, dict[str, float]] = {}
            for step_idx, bucket in enumerate(self._buckets[task]["per_step"], start=1):
                per_step[f"step_{step_idx}"] = {
                    "step": step_idx,
                    "minutes": step_idx * slot_minutes,
                    **_finalize_bucket(bucket),
                }

            report: dict[str, dict[str, float]] = {}
            for minute, step in zip(
                self.report_horizons.get("resolved_minutes", []),
                self.report_horizons.get("resolved_steps", []),
            ):
                step_key = f"step_{int(step)}"
                if step_key in per_step:
                    report[f"{int(minute)}min"] = dict(per_step[step_key])
            metrics[task] = {**overall, "per_step": per_step, "report": report}

        metrics["raw_dc"] = float(metrics["demand"]["rmse"] + metrics["supply"]["rmse"])
        if (
            metrics["demand"].get("nrmse_std_train") is not None
            and metrics["supply"].get("nrmse_std_train") is not None
        ):
            metrics["dc_nrmse_std_train"] = float(
                metrics["demand"]["nrmse_std_train"] + metrics["supply"]["nrmse_std_train"]
            )
        metrics["horizon_reports"] = build_horizon_reports(metrics)
        metrics["horizon_table"] = build_horizon_metric_table(metrics)
        return metrics


def evaluate_dc_predictions(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    pred_horizon: int,
    report_horizons: dict[str, Any],
    target_stats: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    accumulator = DCMetricAccumulator(
        pred_horizon=pred_horizon,
        report_horizons=report_horizons,
        target_stats=target_stats,
    )
    accumulator.update(pred, target)
    return accumulator.finalize()
