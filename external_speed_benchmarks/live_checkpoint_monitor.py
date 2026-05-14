from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import SpatioTemporalDataset
from external_speed_benchmarks.train_sensor_speed import (
    evaluate_v_loss,
    load_prepared_sensor_dataset,
    resolve_outlier_cleaning,
    resolve_fixed_edge_length_feature_mode,
    resolve_split_boundaries,
)
from predictor_normalization import (
    build_normalization_stats,
    normalize_node_features,
    normalize_speed_features,
)
from stgat_model import STGATPredictor
from train_predictor import (
    build_monthly_split_indices,
    build_window_time_mask,
    evaluate_loader_raw_metrics,
    resolve_report_horizons,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write live validation metrics, including report horizons, from a running checkpoint.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training run directory containing stgat_latest.pt.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="stgat_latest.pt",
        help="Checkpoint file name inside --run-dir.",
    )
    parser.add_argument(
        "--output-name",
        default="live_progress_153060.log",
        help="Output log file name inside --run-dir.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=10.0,
        help="Polling interval in seconds between checkpoint checks.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Evaluation device for the side monitor. Default keeps training GPU free.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
        help="Maximum CPU threads for the side monitor.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Evaluate the latest checkpoint once and then exit.",
    )
    return parser.parse_args()


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric(value: object, *, digits: int = 5) -> str:
    parsed = safe_float(value)
    if parsed is None or not np.isfinite(parsed):
        return "None"
    return f"{parsed:.{digits}f}"


def load_state(path: Path, *, device: torch.device) -> dict | None:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception:
        return None


def build_runtime_context(run_config: dict[str, object], *, device: torch.device) -> dict[str, object]:
    dataset_dir = Path(str(run_config["dataset_dir"]))
    dataset, _, time_meta = load_prepared_sensor_dataset(
        dataset_dir,
        disable_time_features=bool(run_config.get("disable_time_features", False)),
    )

    time_slot_minutes = int(
        round(
            (
                pd.to_datetime(time_meta["timestamp"]).iloc[1]
                - pd.to_datetime(time_meta["timestamp"]).iloc[0]
            ).total_seconds()
            / 60.0
        )
    )
    requested_minutes = [int(value) for value in run_config.get("report_horizons_minutes", [])]
    report_horizons = resolve_report_horizons(
        time_slot_minutes=time_slot_minutes,
        pred_horizon=int(run_config["pred_horizon"]),
        requested_minutes=requested_minutes,
        strict=bool(requested_minutes),
    )

    adjacency = dataset["adj"]
    adjacency_weights = dataset["adjacency_weights"]
    edge_index = dataset["edge_index"]
    edge_lengths = dataset["edge_lengths"]
    node_features = dataset["node_features"]
    speed_values = dataset["edge_speeds"]
    speed_valid_mask = dataset["speed_valid_mask"]
    representation_domain = str(dataset.get("representation_domain", "sensor_node"))
    v_domain = str(run_config.get("v_domain") or ("edge" if representation_domain == "pseudo_edge" else "node"))
    num_nodes = int(adjacency.shape[0])
    num_time_steps = int(speed_values.shape[0])

    split_policy = str(run_config.get("split_policy", "benchmark_contiguous"))
    split_alignment = str(run_config.get("split_alignment", "none"))
    hist_len = int(run_config["hist_len"])
    pred_horizon = int(run_config["pred_horizon"])

    time_meta = time_meta.copy()
    time_meta["date"] = pd.to_datetime(time_meta["date"], errors="coerce")
    if split_policy == "project_monthly":
        split_indices = build_monthly_split_indices(time_meta, hist_len, pred_horizon)
    else:
        train_end, val_end, _ = resolve_split_boundaries(time_meta, alignment=split_alignment)
        from external_speed_benchmarks.train_sensor_speed import build_time_contained_split_indices

        split_indices = build_time_contained_split_indices(
            num_time_steps,
            hist_len=hist_len,
            pred_horizon=pred_horizon,
            train_end=train_end,
            val_end=val_end,
        )

    train_mask = build_window_time_mask(
        num_time_steps,
        split_indices["train"],
        hist_len,
        pred_horizon,
    )
    outlier_summary = run_config.get("outlier_cleaning", {})
    outlier_params = outlier_summary.get("params", {}) if isinstance(outlier_summary, dict) else {}
    cleaned_speed_values, _, _ = resolve_outlier_cleaning(
        speed_values=speed_values,
        speed_valid_mask=speed_valid_mask,
        train_time_mask=train_mask,
        mode=str(outlier_summary.get("method", "none")) if isinstance(outlier_summary, dict) else "none",
        lower_quantile=float(outlier_params.get("lower_quantile", 0.01)),
        upper_quantile=float(outlier_params.get("upper_quantile", 0.99)),
    )
    normalization_stats = build_normalization_stats(
        node_features,
        cleaned_speed_values,
        train_mask,
        speed_valid_mask=speed_valid_mask,
    )
    normalized_node_features = normalize_node_features(node_features, normalization_stats)
    normalized_speed_values = normalize_speed_features(
        cleaned_speed_values,
        normalization_stats,
        edge_axis=1,
        speed_valid_mask=speed_valid_mask,
    )

    use_missing_aware_history = bool(run_config.get("use_speed_history_mask", False))
    use_weighted_fixed_graph = bool(run_config.get("fixed_graph_weighted", False))
    use_fixed_edge_length_feature = bool(
        run_config.get(
            "use_fixed_edge_length_feature",
            resolve_fixed_edge_length_feature_mode(
                dataset.get("dataset_summary", {}),
                representation_domain=representation_domain,
            )[0],
        )
    )

    full_dataset = SpatioTemporalDataset(
        normalized_node_features,
        normalized_speed_values,
        edge_speed_valid_mask=speed_valid_mask,
        edge_speed_history_valid_mask=(
            speed_valid_mask if use_missing_aware_history else None
        ),
        history_imputation_enabled=use_missing_aware_history,
        hist_len=hist_len,
        pred_horizon=pred_horizon,
    )
    val_dataset = Subset(full_dataset, split_indices["val"])
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(run_config.get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = STGATPredictor(
        num_nodes=num_nodes,
        edge_index=torch.from_numpy(edge_index),
        edge_lengths=torch.from_numpy(edge_lengths),
        adj_matrix=torch.from_numpy(adjacency),
        adj_weight_matrix=(
            torch.from_numpy(adjacency_weights) if use_weighted_fixed_graph else None
        ),
        hidden_dim=int(run_config["hidden_dim"]),
        num_heads=int(run_config["num_heads"]),
        num_st_blocks=int(run_config["num_st_blocks"]),
        num_gtcn_layers=int(run_config["num_gtcn_layers"]),
        kernel_size=int(run_config["kernel_size"]),
        pred_horizon=pred_horizon,
        node_feat_dim=int(normalized_node_features.shape[-1]),
        adaptive_topk=int(run_config.get("adaptive_topk", 0)),
        speed_use_adaptive=bool(run_config.get("adaptive_enabled", True)),
        use_speed_history_mask=use_missing_aware_history,
        use_fixed_edge_length_feature=use_fixed_edge_length_feature,
        v_domain=v_domain,
    ).to(device)
    model.eval()

    return {
        "model": model,
        "val_loader": val_loader,
        "normalization_stats": normalization_stats,
        "time_slot_minutes": time_slot_minutes,
        "report_horizons": report_horizons,
        "speed_metric_mask_zeros": bool(not run_config.get("disable_speed_mask_zero", False)),
        "v_loss_name": str(run_config.get("v_loss", "mse")),
        "huber_delta": float(run_config.get("huber_delta", 1.0)),
        "charbonnier_eps": float(run_config.get("charbonnier_eps", 1e-3)),
        "amp_enabled": False,
        "amp_dtype": None,
        "non_blocking": False,
    }


def evaluate_report_metrics(
    state: dict[str, object],
    runtime: dict[str, object],
    *,
    device: torch.device,
) -> dict[str, object]:
    model = runtime["model"]
    model.load_state_dict(state["model_state"])
    model.eval()

    val_loss = evaluate_v_loss(
        model,
        runtime["val_loader"],
        device=device,
        non_blocking=bool(runtime["non_blocking"]),
        amp_enabled=bool(runtime["amp_enabled"]),
        amp_dtype=runtime["amp_dtype"],
        v_loss_name=str(runtime["v_loss_name"]),
        huber_delta=float(runtime["huber_delta"]),
        charbonnier_eps=float(runtime["charbonnier_eps"]),
    )
    raw_metrics = evaluate_loader_raw_metrics(
        model,
        runtime["val_loader"],
        train_task="v",
        device=device,
        non_blocking=bool(runtime["non_blocking"]),
        amp_enabled=bool(runtime["amp_enabled"]),
        amp_dtype=runtime["amp_dtype"],
        normalization_stats=runtime["normalization_stats"],
        time_slot_minutes=int(runtime["time_slot_minutes"]),
        report_horizons=runtime["report_horizons"],
        speed_metric_mask_zeros=bool(runtime["speed_metric_mask_zeros"]),
    )
    speed_metrics = raw_metrics["speed"]
    report = speed_metrics.get("report", {}) if isinstance(speed_metrics, dict) else {}
    return {
        "val_v": float(val_loss),
        "val_raw_speed_rmse": float(speed_metrics["rmse"]),
        "report": report,
    }


def build_log_line(
    *,
    record: dict[str, object],
    evaluated: dict[str, object],
) -> str:
    report = evaluated.get("report", {})
    horizon_15 = report.get("15min", {}).get("rmse") if isinstance(report, dict) else None
    horizon_30 = report.get("30min", {}).get("rmse") if isinstance(report, dict) else None
    horizon_60 = report.get("60min", {}).get("rmse") if isinstance(report, dict) else None
    return (
        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
        f"epoch {int(record.get('epoch', -1))} | "
        f"train_v={format_metric(record.get('train_v'))} | "
        f"val_v={format_metric(record.get('val_v', evaluated.get('val_v')))} | "
        f"val_rmse={format_metric(record.get('val_raw_speed_rmse', evaluated.get('val_raw_speed_rmse')))} | "
        f"15min={format_metric(horizon_15)} | "
        f"30min={format_metric(horizon_30)} | "
        f"60min={format_metric(horizon_60)} | "
        f"best={format_metric(record.get('best_val_raw_speed_rmse'))} | "
        f"lr={format_metric(record.get('lr'), digits=6)} -> {format_metric(record.get('lr_next'), digits=6)} | "
        f"eta={record.get('eta')}"
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    ckpt_path = run_dir / args.checkpoint_name
    out_path = run_dir / args.output_name

    torch.set_num_threads(max(int(args.torch_threads), 1))
    device = torch.device(args.device)

    runtime: dict[str, object] | None = None
    runtime_fingerprint: tuple | None = None
    last_epoch = -1

    while True:
        if not ckpt_path.exists():
            time.sleep(args.poll_seconds)
            continue

        state = load_state(ckpt_path, device=device)
        if state is None:
            time.sleep(args.poll_seconds)
            continue

        history = state.get("history", [])
        if not history:
            time.sleep(args.poll_seconds)
            continue

        record = history[-1]
        epoch = int(record.get("epoch", -1))
        run_config = dict(state.get("run_config", {}))
        if not run_config:
            time.sleep(args.poll_seconds)
            continue

        current_fingerprint = (
            str(run_config.get("dataset_dir")),
            int(run_config.get("hist_len", 0)),
            int(run_config.get("pred_horizon", 0)),
            int(run_config.get("hidden_dim", 0)),
            int(run_config.get("num_heads", 0)),
            int(run_config.get("num_st_blocks", 0)),
            int(run_config.get("num_gtcn_layers", 0)),
            int(run_config.get("kernel_size", 0)),
            str(run_config.get("split_policy")),
            str(run_config.get("split_alignment")),
            str(run_config.get("v_domain")),
            bool(run_config.get("adaptive_enabled", True)),
            bool(run_config.get("fixed_graph_weighted", False)),
            bool(run_config.get("use_fixed_edge_length_feature", False)),
            bool(run_config.get("use_speed_history_mask", False)),
        )
        if runtime is None or current_fingerprint != runtime_fingerprint:
            runtime = build_runtime_context(run_config, device=device)
            runtime_fingerprint = current_fingerprint

        if epoch != last_epoch:
            try:
                evaluated = evaluate_report_metrics(state, runtime, device=device)
                line = build_log_line(record=record, evaluated=evaluated)
                with out_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                last_epoch = epoch
                if args.once:
                    return
            except Exception as exc:
                with out_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] epoch {epoch} | monitor_error={type(exc).__name__}: {exc}\n"
                    )
                if args.once:
                    return

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
