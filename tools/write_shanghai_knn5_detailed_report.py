from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader import SpatioTemporalDataset, load_nyc_real_graph_features
from dc_benchmark.metrics import DCMetricAccumulator
from predictor_normalization import (
    build_normalization_stats,
    denormalize_count_values,
    normalize_node_features,
    normalize_speed_features,
)
from stgat_model import STGATPredictor
from train_predictor import (
    build_monthly_split_indices,
    build_window_time_mask,
    configure_cuda_runtime,
    filter_split_indices_by_time_mask,
    forward_for_task,
    infer_time_slot_minutes,
    load_observed_time_mask,
    load_time_meta_for_training,
    resolve_device,
    resolve_precision,
    resolve_report_horizons,
)


SELECTED_BENCHMARK_RUN = Path("runs/shanghai_benchmarks_selected_ep150_20260514_073131")
FULL_BENCHMARK_RUN = Path("runs/shanghai_dc_full_20260514_055501/benchmarks")
STGAT_RUN = Path("runs/shanghai_stgat_continue80_20260514_065707/stgat")
STGAT_DATA_DIR = Path("data/shanghai_dc")
OUTPUT_PATH = Path("runs/shanghai_knn5_detailed_ranking.md")


BENCHMARK_SOURCES = [
    (SELECTED_BENCHMARK_RUN / "Deep_MultiConvLSTM_dc_metrics.json", "selected 150-epoch rerun"),
    (SELECTED_BENCHMARK_RUN / "Graph_WaveNet_dc_metrics.json", "selected 150-epoch rerun"),
    (SELECTED_BENCHMARK_RUN / "MT_MF_GCN_dc_metrics.json", "selected 150-epoch rerun"),
    (SELECTED_BENCHMARK_RUN / "STGCN_dc_metrics.json", "selected 150-epoch rerun"),
    (SELECTED_BENCHMARK_RUN / "MLRNN_dc_metrics.json", "selected 150-epoch rerun"),
    (SELECTED_BENCHMARK_RUN / "ST_ResNet_dc_metrics.json", "selected 150-epoch rerun"),
    (FULL_BENCHMARK_RUN / "DCRNN_dc_metrics.json", "full 100-epoch run"),
    (FULL_BENCHMARK_RUN / "HA_dc_metrics.json", "full benchmark run"),
    (FULL_BENCHMARK_RUN / "ConvLSTM_dc_metrics.json", "full 100-epoch run"),
    (FULL_BENCHMARK_RUN / "LSTM_dc_metrics.json", "full 100-epoch run"),
    (FULL_BENCHMARK_RUN / "ARIMA_dc_metrics.json", "full benchmark run"),
    (FULL_BENCHMARK_RUN / "XGBoost_dc_metrics.json", "full benchmark run"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "N/A"
    return f"{number:.{digits}f}"


def metric(metrics: dict[str, Any], task: str, name: str) -> float | None:
    values = metrics.get(task, {})
    if not isinstance(values, dict):
        return None
    value = values.get(name)
    return float(value) if value is not None else None


def horizon_metric(metrics: dict[str, Any], task: str, horizon: str, name: str) -> float | None:
    values = metrics.get(task, {}).get("report", {}).get(horizon, {})
    if not isinstance(values, dict):
        return None
    value = values.get(name)
    return float(value) if value is not None else None


def dc_rmse(metrics: dict[str, Any]) -> float:
    raw_dc = metrics.get("raw_dc")
    if raw_dc is not None:
        return float(raw_dc)
    return float(metric(metrics, "demand", "rmse") or 0.0) + float(metric(metrics, "supply", "rmse") or 0.0)


def dc_mse_sum(metrics: dict[str, Any], horizon: str | None = None) -> float | None:
    if horizon is None:
        d = metric(metrics, "demand", "mse")
        c = metric(metrics, "supply", "mse")
    else:
        d = horizon_metric(metrics, "demand", horizon, "mse")
        c = horizon_metric(metrics, "supply", horizon, "mse")
    if d is None or c is None:
        return None
    return float(d + c)


def dc_mape_avg(metrics: dict[str, Any], horizon: str | None = None) -> float | None:
    if horizon is None:
        d = metric(metrics, "demand", "mape")
        c = metric(metrics, "supply", "mape")
    else:
        d = horizon_metric(metrics, "demand", horizon, "mape")
        c = horizon_metric(metrics, "supply", horizon, "mape")
    if d is None or c is None:
        return None
    return float((d + c) / 2.0)


def dc_rmse_horizon(metrics: dict[str, Any], horizon: str) -> float | None:
    reports = metrics.get("horizon_reports", {})
    if isinstance(reports, dict) and horizon in reports and "raw_dc" in reports[horizon]:
        return float(reports[horizon]["raw_dc"])
    d = horizon_metric(metrics, "demand", horizon, "rmse")
    c = horizon_metric(metrics, "supply", horizon, "rmse")
    if d is None or c is None:
        return None
    return float(d + c)


def run_info(record: dict[str, Any]) -> tuple[str, str, str]:
    config = record.get("run_config") or {}
    completed = config.get("completed_epochs", config.get("epochs", "N/A"))
    best_epoch = config.get("best_epoch", "N/A")
    val_score = config.get("selection_score", "N/A")
    return str(completed), str(best_epoch), fmt(val_score)


def evaluate_stgat_with_mape(device_arg: str) -> dict[str, Any]:
    cache_path = STGAT_RUN / "predictor_test_metrics_with_mape.json"
    meta_path = STGAT_RUN / "stgat_meta.json"
    checkpoint_path = STGAT_RUN / "stgat_best.pt"
    meta = load_json(meta_path)

    device = resolve_device(device_arg)
    configure_cuda_runtime(device)
    precision = resolve_precision(device, "auto")
    amp_enabled = device.type == "cuda" and precision == "bf16"
    amp_dtype = torch.bfloat16 if amp_enabled else None
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    data = load_nyc_real_graph_features(
        STGAT_DATA_DIR,
        max_time_steps=0,
        edge_length_source="osrm",
        add_time_features=bool(meta.get("use_time_features", True)),
    )
    adj = data["adj"]
    edge_index = data["edge_index"]
    edge_lengths = data["edge_lengths"]
    node_feat = data["node_features"]
    edge_speeds = data["edge_speeds"]
    hist_len = int(meta["hist_len"])
    pred_horizon = int(meta["pred_horizon"])
    train_task = str(meta.get("train_task", "dc"))

    time_meta = load_time_meta_for_training(STGAT_DATA_DIR, int(node_feat.shape[0]))
    slot_minutes = infer_time_slot_minutes(time_meta)
    report_horizons = resolve_report_horizons(
        time_slot_minutes=slot_minutes,
        pred_horizon=pred_horizon,
        requested_minutes=[15, 30, 60],
        strict=True,
    )
    split_indices = build_monthly_split_indices(time_meta, hist_len, pred_horizon)
    observed_time_mask = load_observed_time_mask(STGAT_DATA_DIR, int(node_feat.shape[0]))
    split_indices = filter_split_indices_by_time_mask(
        split_indices,
        observed_time_mask,
        hist_len,
        pred_horizon,
    )
    train_time_mask = build_window_time_mask(
        int(node_feat.shape[0]),
        split_indices["train"],
        hist_len,
        pred_horizon,
    )
    normalization_stats = build_normalization_stats(node_feat, edge_speeds, train_time_mask)
    node_feat = normalize_node_features(node_feat, normalization_stats)
    edge_speeds = normalize_speed_features(edge_speeds, normalization_stats, edge_axis=1)

    full_ds = SpatioTemporalDataset(node_feat, edge_speeds, hist_len=hist_len, pred_horizon=pred_horizon)
    test_loader = DataLoader(
        Subset(full_ds, split_indices["test"]),
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = STGATPredictor(
        num_nodes=int(adj.shape[0]),
        edge_index=torch.from_numpy(edge_index),
        edge_lengths=torch.from_numpy(edge_lengths),
        adj_matrix=torch.from_numpy(adj),
        hidden_dim=int(meta["hidden_dim"]),
        num_heads=int(meta["num_heads"]),
        num_st_blocks=int(meta["num_st_blocks"]),
        num_gtcn_layers=int(meta["num_gtcn_layers"]),
        kernel_size=int(meta["kernel_size"]),
        pred_horizon=pred_horizon,
        node_feat_dim=int(node_feat.shape[-1]),
        adaptive_topk=int(meta["adaptive_topk"]),
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    accumulator = DCMetricAccumulator(pred_horizon=pred_horizon, report_horizons=report_horizons)
    with torch.inference_mode():
        for batch in test_loader:
            node_seq = batch["node_seq"].to(device, non_blocking=non_blocking)
            speed_seq = batch["speed_seq"].to(device, non_blocking=non_blocking)
            d_tgt = batch["demand_target"].to(device, non_blocking=non_blocking)
            c_tgt = batch["supply_target"].to(device, non_blocking=non_blocking)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                d_pred, c_pred, _ = forward_for_task(model, node_seq, speed_seq, train_task)
            d_pred_raw = denormalize_count_values(
                d_pred.detach().float().cpu().numpy(),
                normalization_stats,
                task="demand",
            )
            c_pred_raw = denormalize_count_values(
                c_pred.detach().float().cpu().numpy(),
                normalization_stats,
                task="supply",
            )
            d_tgt_raw = denormalize_count_values(
                d_tgt.detach().float().cpu().numpy(),
                normalization_stats,
                task="demand",
            )
            c_tgt_raw = denormalize_count_values(
                c_tgt.detach().float().cpu().numpy(),
                normalization_stats,
                task="supply",
            )
            pred = np.stack([d_pred_raw, c_pred_raw], axis=-1)
            target = np.stack([d_tgt_raw, c_tgt_raw], axis=-1)
            accumulator.update(pred, target)

    metrics = accumulator.finalize()
    base_metrics = load_json(STGAT_RUN / "predictor_test_metrics.json")
    output = {
        "method_id": "stgat_knn5",
        "paper_name": "STGAT (ours, KNN=5)",
        "metrics": metrics,
        "base_predictor_test_metrics": base_metrics,
        "run_config": {
            "completed_epochs": "100 + 80 continued",
            "best_epoch": "continue epoch 51",
            "selection_metric": "val_raw_dc",
            "selection_score": base_metrics.get("selected_checkpoint_metric"),
            "checkpoint_path": str(checkpoint_path),
            "data_dir": str(STGAT_DATA_DIR),
            "graph": "KNN=5",
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    cache_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def load_records(device_arg: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stgat = evaluate_stgat_with_mape(device_arg)
    records.append(
        {
            "method_id": stgat["method_id"],
            "paper_name": stgat["paper_name"],
            "metrics": stgat["metrics"],
            "run_config": stgat["run_config"],
            "source_note": "STGAT KNN5 continued run; MAPE recomputed from best checkpoint",
            "source_file": str(STGAT_RUN / "predictor_test_metrics_with_mape.json"),
        }
    )
    for path, note in BENCHMARK_SOURCES:
        record = load_json(path)
        records.append(
            {
                "method_id": record["method_id"],
                "paper_name": record["paper_name"],
                "metrics": record["metrics"],
                "run_config": record.get("run_config", {}),
                "source_note": note,
                "source_file": str(path),
            }
        )
    return sorted(records, key=lambda item: dc_rmse(item["metrics"]))


def write_table(lines: list[str], headers: list[str], rows: list[list[str]]) -> None:
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")


def build_report(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Shanghai DC KNN5 Detailed Ranking")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("Scope: Shanghai DC KNN=5 only. KNN4/KNN8 validation runs are intentionally excluded.")
    lines.append("")
    lines.append("Definitions:")
    lines.append("- D = demand target; C = supply/capacity target.")
    lines.append("- DC_RMSE is the ranking metric: D_RMSE + C_RMSE.")
    lines.append("- DC_MSE_SUM is D_MSE + C_MSE, included as a detailed scale reference.")
    lines.append("- DC_MAPE_AVG is the average of D_MAPE and C_MAPE; percentages are not summed.")
    lines.append("- MAPE excludes zero-valued targets using abs(target) > 1e-6, matching the benchmark evaluator.")
    lines.append("")

    ranking_rows: list[list[str]] = []
    for rank, record in enumerate(records, start=1):
        metrics = record["metrics"]
        completed, best_epoch, val_score = run_info(record)
        ranking_rows.append(
            [
                str(rank),
                record["paper_name"],
                fmt(dc_rmse(metrics)),
                fmt(metric(metrics, "demand", "rmse")),
                fmt(metric(metrics, "supply", "rmse")),
                fmt(dc_mape_avg(metrics)),
                completed,
                best_epoch,
                val_score,
            ]
        )
    lines.append("## Overall Ranking")
    lines.append("")
    write_table(
        lines,
        ["Rank", "Method", "DC_RMSE", "D_RMSE", "C_RMSE", "DC_MAPE_AVG", "Epochs", "Best Epoch", "Best Val RawDC"],
        ranking_rows,
    )
    lines.append("")

    detail_rows: list[list[str]] = []
    for rank, record in enumerate(records, start=1):
        metrics = record["metrics"]
        detail_rows.append(
            [
                str(rank),
                record["paper_name"],
                fmt(metric(metrics, "demand", "mse")),
                fmt(metric(metrics, "demand", "rmse")),
                fmt(metric(metrics, "demand", "mape")),
                fmt(metric(metrics, "supply", "mse")),
                fmt(metric(metrics, "supply", "rmse")),
                fmt(metric(metrics, "supply", "mape")),
                fmt(dc_mse_sum(metrics)),
                fmt(dc_rmse(metrics)),
                fmt(dc_mape_avg(metrics)),
                fmt(metric(metrics, "gap", "rmse")),
                fmt(metric(metrics, "gap", "mape")),
            ]
        )
    lines.append("## Overall Detailed Metrics")
    lines.append("")
    write_table(
        lines,
        [
            "Rank",
            "Method",
            "D_MSE",
            "D_RMSE",
            "D_MAPE",
            "C_MSE",
            "C_RMSE",
            "C_MAPE",
            "DC_MSE_SUM",
            "DC_RMSE",
            "DC_MAPE_AVG",
            "Gap_RMSE",
            "Gap_MAPE",
        ],
        detail_rows,
    )
    lines.append("")

    lines.append("## Horizon Metrics")
    lines.append("")
    for rank, record in enumerate(records, start=1):
        metrics = record["metrics"]
        lines.append(f"### Rank {rank}: {record['paper_name']}")
        lines.append("")
        horizon_rows: list[list[str]] = []
        for horizon in ("15min", "30min", "60min"):
            horizon_rows.append(
                [
                    horizon,
                    fmt(horizon_metric(metrics, "demand", horizon, "mse")),
                    fmt(horizon_metric(metrics, "demand", horizon, "rmse")),
                    fmt(horizon_metric(metrics, "demand", horizon, "mape")),
                    fmt(horizon_metric(metrics, "supply", horizon, "mse")),
                    fmt(horizon_metric(metrics, "supply", horizon, "rmse")),
                    fmt(horizon_metric(metrics, "supply", horizon, "mape")),
                    fmt(dc_mse_sum(metrics, horizon)),
                    fmt(dc_rmse_horizon(metrics, horizon)),
                    fmt(dc_mape_avg(metrics, horizon)),
                    fmt(horizon_metric(metrics, "gap", horizon, "rmse")),
                    fmt(horizon_metric(metrics, "gap", horizon, "mape")),
                ]
            )
        write_table(
            lines,
            [
                "Horizon",
                "D_MSE",
                "D_RMSE",
                "D_MAPE",
                "C_MSE",
                "C_RMSE",
                "C_MAPE",
                "DC_MSE_SUM",
                "DC_RMSE",
                "DC_MAPE_AVG",
                "Gap_RMSE",
                "Gap_MAPE",
            ],
            horizon_rows,
        )
        lines.append("")

    source_rows: list[list[str]] = []
    for rank, record in enumerate(records, start=1):
        source_rows.append(
            [
                str(rank),
                record["method_id"],
                record["paper_name"],
                record["source_note"],
                record["source_file"],
            ]
        )
    lines.append("## Source Files")
    lines.append("")
    write_table(lines, ["Rank", "Method ID", "Method", "Selected Source", "File"], source_rows)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write detailed Shanghai KNN5 DC ranking report.")
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    records = load_records(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(records), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
