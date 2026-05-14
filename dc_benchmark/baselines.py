from __future__ import annotations

from typing import Any

import numpy as np

from .dataset import iter_dc_windows
from .metrics import DCMetricAccumulator


PAPER_METHODS: dict[str, dict[str, str]] = {
    "ha": {
        "paper_name": "Historical Average",
        "family": "historical_average",
        "scope": "sanity_baseline",
    },
    "arima": {
        "paper_name": "ARIMA",
        "family": "autoregressive_arima_inspired",
        "scope": "paper_inspired_reimplementation",
    },
    "xgboost": {
        "paper_name": "XGBoost",
        "family": "gradient_boosting_inspired",
        "scope": "paper_inspired_reimplementation",
    },
    "lstm": {
        "paper_name": "LSTM",
        "family": "nodewise_lstm",
        "scope": "paper_inspired_reimplementation",
    },
    "convlstm": {
        "paper_name": "ConvLSTM",
        "family": "temporal_convlstm_inspired",
        "scope": "paper_inspired_reimplementation",
    },
    "st_resnet": {
        "paper_name": "ST-ResNet",
        "family": "residual_temporal_conv",
        "scope": "paper_inspired_reimplementation",
    },
    "stgcn": {
        "paper_name": "STGCN",
        "family": "graph_temporal_conv",
        "scope": "paper_inspired_reimplementation",
    },
    "dcrnn": {
        "paper_name": "DCRNN",
        "family": "diffusion_recurrent_inspired",
        "scope": "paper_inspired_reimplementation",
    },
    "graph_wavenet": {
        "paper_name": "Graph WaveNet",
        "family": "adaptive_graph_temporal_conv",
        "scope": "paper_inspired_reimplementation",
    },
    "mlrnn_taxi_demand": {
        "paper_name": "MLRNN Taxi Demand",
        "family": "cluster_recurrent_inspired",
        "scope": "paper_inspired_reimplementation",
    },
    "deep_multiconvlstm": {
        "paper_name": "Deep Multi-Scale ConvLSTM",
        "family": "multi_scale_temporal_conv",
        "scope": "paper_inspired_reimplementation",
    },
    "mt_mf_gcn": {
        "paper_name": "MT-MF-GCN",
        "family": "multi_task_factorized_graph",
        "scope": "paper_inspired_reimplementation",
    },
}


def _slot_values(time_meta: Any, length: int) -> np.ndarray:
    if "slot" in time_meta:
        return np.asarray(time_meta["slot"].to_numpy(), dtype=np.int64)
    return np.arange(length, dtype=np.int64)


def _target_time_mask(indices: list[int], *, hist_len: int, pred_horizon: int, length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for idx in indices:
        start = int(idx) + hist_len
        mask[start:start + pred_horizon] = True
    return mask


def _report_for_method(method_id: str, metrics: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    meta = PAPER_METHODS[method_id]
    claim_type = "sanity_baseline" if meta["scope"] == "sanity_baseline" else "paper_based_reimplementation"
    return {
        "method_id": method_id,
        "paper_name": meta["paper_name"],
        "implementation_family": meta["family"],
        "implementation_scope": meta["scope"],
        "claim_type": claim_type,
        "official_source_verified": False,
        "not_official_paper_result": True,
        "metrics": metrics,
        "run_config": config,
    }


def run_historical_average(benchmark: dict[str, Any], *, batch_size: int = 64) -> dict[str, Any]:
    manifest = benchmark["manifest"]
    splits = benchmark["splits"]["indices"]
    targets = benchmark["targets"]
    hist_len = int(manifest["hist_len"])
    pred_horizon = int(manifest["pred_horizon"])
    slots = _slot_values(benchmark["time_meta"], len(targets))
    train_mask = _target_time_mask(
        splits["train"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        length=len(targets),
    )

    unique_slots = np.unique(slots)
    global_mean = np.asarray(targets[train_mask], dtype=np.float64).mean(axis=0)
    slot_mean: dict[int, np.ndarray] = {}
    for slot in unique_slots:
        mask = train_mask & (slots == slot)
        slot_mean[int(slot)] = (
            np.asarray(targets[mask], dtype=np.float64).mean(axis=0)
            if np.any(mask)
            else global_mean
        )

    accumulator = DCMetricAccumulator(
        pred_horizon=pred_horizon,
        report_horizons=manifest["report_horizons"],
        target_stats=manifest.get("target_stats_train_time_mask"),
    )
    for batch in iter_dc_windows(
        targets,
        splits["test"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        batch_size=batch_size,
    ):
        pred = np.empty_like(batch["target"], dtype=np.float32)
        for row_idx, target_times in enumerate(batch["target_times"]):
            for step_idx, time_idx in enumerate(target_times):
                pred[row_idx, :, step_idx, :] = slot_mean[int(slots[int(time_idx)])]
        accumulator.update(pred, batch["target"])

    return _report_for_method(
        "ha",
        accumulator.finalize(),
        {"batch_size": batch_size, "uses_train_slot_means": True},
    )


def _fit_global_ar_coefficients(
    benchmark: dict[str, Any],
    *,
    max_train_rows: int,
) -> np.ndarray:
    manifest = benchmark["manifest"]
    splits = benchmark["splits"]["indices"]
    targets = benchmark["targets"]
    hist_len = int(manifest["hist_len"])
    pred_horizon = int(manifest["pred_horizon"])
    rng = np.random.default_rng(42)
    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    for batch in iter_dc_windows(
        targets,
        splits["train"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        batch_size=64,
    ):
        x = np.log1p(batch["history"].reshape(-1, hist_len * 2))
        y = np.log1p(batch["target"].reshape(-1, pred_horizon * 2))
        rows_x.append(x)
        rows_y.append(y)
        if sum(part.shape[0] for part in rows_x) >= max_train_rows:
            break

    x_all = np.concatenate(rows_x, axis=0)
    y_all = np.concatenate(rows_y, axis=0)
    if x_all.shape[0] > max_train_rows:
        take = rng.choice(x_all.shape[0], size=max_train_rows, replace=False)
        x_all = x_all[take]
        y_all = y_all[take]
    x_all = np.concatenate([x_all, np.ones((x_all.shape[0], 1), dtype=x_all.dtype)], axis=1)
    coeff, *_ = np.linalg.lstsq(x_all, y_all, rcond=None)
    return coeff.astype(np.float32)


def run_autoregressive(
    benchmark: dict[str, Any],
    *,
    method_id: str = "arima",
    batch_size: int = 64,
    max_train_rows: int = 200_000,
) -> dict[str, Any]:
    manifest = benchmark["manifest"]
    splits = benchmark["splits"]["indices"]
    targets = benchmark["targets"]
    hist_len = int(manifest["hist_len"])
    pred_horizon = int(manifest["pred_horizon"])
    coeff = _fit_global_ar_coefficients(benchmark, max_train_rows=max_train_rows)

    accumulator = DCMetricAccumulator(
        pred_horizon=pred_horizon,
        report_horizons=manifest["report_horizons"],
        target_stats=manifest.get("target_stats_train_time_mask"),
    )
    for batch in iter_dc_windows(
        targets,
        splits["test"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        batch_size=batch_size,
    ):
        x = np.log1p(batch["history"].reshape(-1, hist_len * 2))
        x = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
        pred = np.expm1(x @ coeff).reshape(batch["target"].shape)
        pred = np.maximum(pred, 0.0).astype(np.float32)
        accumulator.update(pred, batch["target"])

    return _report_for_method(
        method_id,
        accumulator.finalize(),
        {
            "batch_size": batch_size,
            "max_train_rows": int(max_train_rows),
            "note": "Global log1p autoregressive adapter; not an official ARIMA/XGBoost implementation.",
        },
    )
