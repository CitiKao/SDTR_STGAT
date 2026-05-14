from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import SpatioTemporalDataset
from external_speed_benchmarks.sensor_dataset_utils import (
    OUTLIER_CLEANING_MODES,
    apply_valid_speed_clip,
    build_cyclical_time_features,
    compute_train_quantile_clip_bounds,
    normalize_representation_domain,
    validate_outlier_cleaning_mode,
)
from predictor_normalization import (
    build_normalization_stats,
    normalize_node_features,
    normalize_speed_features,
    serialize_normalization_stats,
)
from stgat_model import STGATPredictor
from training_runtime_utils import (
    OPTIMIZER_CHOICES,
    V_LOSS_CHOICES,
    build_optimizer,
    masked_regression_loss,
    maybe_apply_linear_warmup,
)
from train_predictor import (
    CALENDAR_SPLIT_STRATEGY,
    build_monthly_split_indices,
    build_window_time_mask,
    configure_cuda_runtime,
    evaluate_loader_raw_metrics,
    parse_report_horizons_minutes,
    resolve_device,
    resolve_num_workers,
    resolve_precision,
    resolve_report_horizons,
)

SPLIT_ALIGNMENT_MODES = ("none", "day", "week", "month")
SPLIT_POLICIES = ("project_monthly", "benchmark_contiguous")
PROJECT_MONTHLY_SPLIT_DESCRIPTION = (
    "Per-month days 1-20 train / 21-24 val / 25+ test with full history+target window containment."
)
RESUME_CONFIG_COMPARABLE_KEYS = (
    "dataset_dir",
    "dataset_name",
    "processed_dataset_fingerprint",
    "seed",
    "hist_len",
    "pred_horizon",
    "report_horizons_minutes",
    "hidden_dim",
    "num_heads",
    "num_st_blocks",
    "num_gtcn_layers",
    "kernel_size",
    "node_feat_dim",
    "adaptive_topk",
    "adaptive_enabled",
    "fixed_graph_weighted",
    "use_fixed_edge_length_feature",
    "fixed_edge_length_feature_mode",
    "use_speed_history_mask",
    "disable_time_features",
    "disable_speed_mask_zero",
    "batch_size",
    "optimizer",
    "lr",
    "weight_decay",
    "warmup_epochs",
    "grad_clip_norm",
    "v_loss",
    "huber_delta",
    "charbonnier_eps",
    "val_interval",
    "split_policy",
    "split_alignment",
    "scheduler_monitor",
    "best_checkpoint_monitor",
    "scheduler_factor",
    "scheduler_patience",
    "scheduler_cooldown",
    "scheduler_min_lr",
    "early_stopping_enabled",
    "early_stop_patience",
    "min_epochs",
    "representation_domain",
    "v_domain",
    "num_speed_items",
    "metric_protocol",
    "preprocessing_variant_id",
)


def load_prepared_sensor_dataset(dataset_dir: str | Path, *, disable_time_features: bool) -> tuple[dict[str, np.ndarray], list[str], pd.DataFrame]:
    dataset_dir = Path(dataset_dir)
    speed_values = np.load(dataset_dir / "speed_values.npy").astype(np.float32)
    speed_valid_mask_path = dataset_dir / "speed_valid_mask.npy"
    if not speed_valid_mask_path.exists():
        raise FileNotFoundError(
            f"{speed_valid_mask_path} is missing. Re-run prepare_dcrnn_sensor_datasets.py to build "
            "the explicit observation mask required for benchmark-safe speed training."
        )
    speed_valid_mask = np.load(speed_valid_mask_path).astype(bool)
    adjacency = np.load(dataset_dir / "adjacency_matrix.npy").astype(np.float32)
    adjacency_weights_path = dataset_dir / "adjacency_weights.npy"
    adjacency_weights = (
        np.load(adjacency_weights_path).astype(np.float32)
        if adjacency_weights_path.exists()
        else adjacency.copy()
    )
    edge_index = np.load(dataset_dir / "edge_index.npy").astype(np.int64)
    edge_lengths = np.load(dataset_dir / "edge_lengths_km.npy").astype(np.float32)
    time_meta = pd.read_csv(dataset_dir / "time_meta.csv")
    dataset_summary_path = dataset_dir / "dataset_summary.json"
    dataset_summary = (
        json.loads(dataset_summary_path.read_text(encoding="utf-8"))
        if dataset_summary_path.exists()
        else {}
    )
    representation_domain = normalize_representation_domain(
        dataset_summary.get("representation_domain", "sensor_node")
    )

    time_feature_names: list[str] = []
    num_graph_nodes = int(adjacency.shape[0])
    node_features = np.zeros((speed_values.shape[0], num_graph_nodes, 2), dtype=np.float32)
    if not disable_time_features:
        time_features, time_feature_names = build_cyclical_time_features(time_meta)
        expanded = np.broadcast_to(
            time_features[:, None, :],
            (speed_values.shape[0], num_graph_nodes, time_features.shape[1]),
        )
        node_features = np.concatenate([node_features, expanded], axis=-1).astype(np.float32)

    return {
        "adj": adjacency,
        "adjacency_weights": adjacency_weights,
        "edge_index": edge_index,
        "edge_lengths": edge_lengths,
        "node_features": node_features,
        "edge_speeds": speed_values,
        "speed_valid_mask": speed_valid_mask,
        "dataset_summary": dataset_summary,
        "representation_domain": representation_domain,
        "source": dataset_dir.name,
    }, time_feature_names, time_meta


def resolve_fixed_edge_length_feature_mode(
    dataset_summary: dict[str, object] | None,
    *,
    representation_domain: str,
) -> tuple[bool, str]:
    summary = dataset_summary if isinstance(dataset_summary, dict) else {}
    graph_source = summary.get("graph_source", {})
    if not isinstance(graph_source, dict):
        graph_source = {}
    distance_semantics = graph_source.get("distance_semantics", {})
    if not isinstance(distance_semantics, dict):
        distance_semantics = {}

    if str(representation_domain) == "pseudo_edge":
        return True, "enabled_pseudo_edge"

    graph_mode = str(graph_source.get("mode", "")).strip().lower()
    edge_length_semantics = str(distance_semantics.get("edge_lengths_km", "")).strip().lower()
    if edge_length_semantics == "disabled_zero_no_custom_length_feature":
        return False, "disabled_zero_no_custom_length_feature"
    if graph_mode == "official_adj_pkl":
        return False, "disabled_for_official_adj_pkl_sensor_node"
    if edge_length_semantics:
        return True, edge_length_semantics
    return True, "enabled_unspecified_legacy"


def build_time_contained_split_indices(
    num_time_steps: int,
    *,
    hist_len: int,
    pred_horizon: int,
    train_end: int | None = None,
    val_end: int | None = None,
) -> dict[str, list[int]]:
    if train_end is None:
        train_end = round(num_time_steps * 0.7)
    if val_end is None:
        val_end = round(num_time_steps * 0.8)
    total_samples = num_time_steps - hist_len - pred_horizon + 1
    window_len = hist_len + pred_horizon
    splits = {"train": [], "val": [], "test": []}
    for idx in range(max(total_samples, 0)):
        end = idx + window_len
        if end <= train_end:
            splits["train"].append(idx)
        elif idx >= train_end and end <= val_end:
            splits["val"].append(idx)
        elif idx >= val_end and end <= num_time_steps:
            splits["test"].append(idx)
    return splits


def validate_split_alignment(mode: str) -> str:
    cleaned = str(mode).strip().lower()
    if cleaned not in SPLIT_ALIGNMENT_MODES:
        raise ValueError(
            f"Unsupported split alignment '{mode}'. Expected one of: {', '.join(SPLIT_ALIGNMENT_MODES)}."
        )
    return cleaned


def is_matching_split_boundary(timestamp: pd.Timestamp, mode: str) -> bool:
    if mode == "none":
        return True
    ts = pd.Timestamp(timestamp)
    is_day_start = bool(
        ts.hour == 0
        and ts.minute == 0
        and ts.second == 0
        and ts.microsecond == 0
        and ts.nanosecond == 0
    )
    if not is_day_start:
        return False
    if mode == "day":
        return True
    if mode == "week":
        return ts.weekday() == 0
    if mode == "month":
        return ts.day == 1
    raise ValueError(f"Unsupported split alignment '{mode}'.")


def align_split_boundary_index(
    timestamps: pd.Series,
    *,
    raw_index: int,
    mode: str,
    lower_bound: int,
    upper_bound: int,
) -> tuple[int, dict[str, object]]:
    cleaned_mode = validate_split_alignment(mode)
    if cleaned_mode == "none":
        clipped_index = int(min(max(raw_index, lower_bound), upper_bound))
        ts = pd.Timestamp(timestamps.iloc[clipped_index])
        return clipped_index, {
            "mode": cleaned_mode,
            "raw_index": int(raw_index),
            "effective_index": clipped_index,
            "shift_steps": int(clipped_index - raw_index),
            "raw_timestamp": ts.isoformat(),
            "effective_timestamp": ts.isoformat(),
            "matched_boundary": True,
            "fallback_used": False,
        }

    candidates = [
        idx
        for idx in range(int(lower_bound), int(upper_bound) + 1)
        if is_matching_split_boundary(pd.Timestamp(timestamps.iloc[idx]), cleaned_mode)
    ]
    clipped_index = int(min(max(raw_index, lower_bound), upper_bound))
    raw_ts = pd.Timestamp(timestamps.iloc[clipped_index])
    if not candidates:
        return clipped_index, {
            "mode": cleaned_mode,
            "raw_index": int(raw_index),
            "effective_index": clipped_index,
            "shift_steps": int(clipped_index - raw_index),
            "raw_timestamp": raw_ts.isoformat(),
            "effective_timestamp": raw_ts.isoformat(),
            "matched_boundary": False,
            "fallback_used": True,
        }

    effective_index = min(
        candidates,
        key=lambda idx: (abs(int(idx) - int(raw_index)), int(idx) < int(raw_index), int(idx)),
    )
    effective_ts = pd.Timestamp(timestamps.iloc[effective_index])
    return int(effective_index), {
        "mode": cleaned_mode,
        "raw_index": int(raw_index),
        "effective_index": int(effective_index),
        "shift_steps": int(effective_index - raw_index),
        "raw_timestamp": raw_ts.isoformat(),
        "effective_timestamp": effective_ts.isoformat(),
        "matched_boundary": True,
        "fallback_used": False,
    }


def build_split_boundary_summary(
    timestamps: pd.Series,
    *,
    alignment: str,
    train_end_raw: int,
    val_end_raw: int,
    train_end: int,
    val_end: int,
) -> dict[str, object]:
    ts = pd.to_datetime(timestamps)

    def summarize_segment(start_idx: int, end_idx: int) -> dict[str, object]:
        segment = ts.iloc[start_idx:end_idx]
        if segment.empty:
            return {
                "start": None,
                "end": None,
                "start_weekday": None,
                "end_weekday": None,
                "months": [],
                "weekday_counts": {},
            }
        return {
            "start": pd.Timestamp(segment.iloc[0]).isoformat(),
            "end": pd.Timestamp(segment.iloc[-1]).isoformat(),
            "start_weekday": str(pd.Timestamp(segment.iloc[0]).day_name()),
            "end_weekday": str(pd.Timestamp(segment.iloc[-1]).day_name()),
            "months": sorted(segment.dt.to_period("M").astype(str).unique().tolist()),
            "weekday_counts": {
                str(key): int(value)
                for key, value in segment.dt.day_name().value_counts().sort_index().items()
            },
        }

    return {
        "mode": "contiguous_ratio_70_10_20",
        "alignment": validate_split_alignment(alignment),
        "raw_boundaries": {
            "train_end_index": int(train_end_raw),
            "val_end_index": int(val_end_raw),
            "train_boundary_timestamp": pd.Timestamp(ts.iloc[min(max(train_end_raw, 0), len(ts) - 1)]).isoformat(),
            "val_boundary_timestamp": pd.Timestamp(ts.iloc[min(max(val_end_raw, 0), len(ts) - 1)]).isoformat(),
        },
        "effective_boundaries": {
            "train_end_index": int(train_end),
            "val_end_index": int(val_end),
            "train_last_timestamp": pd.Timestamp(ts.iloc[max(train_end - 1, 0)]).isoformat(),
            "val_first_timestamp": pd.Timestamp(ts.iloc[min(train_end, len(ts) - 1)]).isoformat(),
            "val_last_timestamp": pd.Timestamp(ts.iloc[max(val_end - 1, 0)]).isoformat(),
            "test_first_timestamp": pd.Timestamp(ts.iloc[min(val_end, len(ts) - 1)]).isoformat(),
        },
        "calendar": {
            "train": summarize_segment(0, train_end),
            "val": summarize_segment(train_end, val_end),
            "test": summarize_segment(val_end, len(ts)),
        },
    }


def resolve_split_boundaries(
    time_meta: pd.DataFrame,
    *,
    alignment: str,
) -> tuple[int, int, dict[str, object]]:
    timestamps = pd.to_datetime(time_meta["timestamp"])
    num_time_steps = int(len(timestamps))
    raw_train_end = round(num_time_steps * 0.7)
    raw_val_end = round(num_time_steps * 0.8)
    cleaned_alignment = validate_split_alignment(alignment)
    train_end, train_alignment = align_split_boundary_index(
        timestamps,
        raw_index=raw_train_end,
        mode=cleaned_alignment,
        lower_bound=1,
        upper_bound=max(num_time_steps - 2, 1),
    )
    val_end, val_alignment = align_split_boundary_index(
        timestamps,
        raw_index=raw_val_end,
        mode=cleaned_alignment,
        lower_bound=min(max(train_end + 1, 1), max(num_time_steps - 1, 1)),
        upper_bound=max(num_time_steps - 1, 1),
    )
    if val_end <= train_end:
        val_end = min(max(train_end + 1, 1), max(num_time_steps - 1, 1))
        val_alignment = {
            **val_alignment,
            "effective_index": int(val_end),
            "effective_timestamp": pd.Timestamp(timestamps.iloc[val_end]).isoformat(),
            "shift_steps": int(val_end - raw_val_end),
            "fallback_used": True,
            "matched_boundary": False,
        }

    summary = build_split_boundary_summary(
        timestamps,
        alignment=cleaned_alignment,
        train_end_raw=raw_train_end,
        val_end_raw=raw_val_end,
        train_end=train_end,
        val_end=val_end,
    )
    summary["alignment_details"] = {
        "train_boundary": train_alignment,
        "val_boundary": val_alignment,
    }
    return int(train_end), int(val_end), summary


def build_history_time_mask(
    num_time_steps: int,
    sample_indices: list[int],
    hist_len: int,
) -> np.ndarray:
    mask = np.zeros(num_time_steps, dtype=bool)
    for idx in sample_indices:
        start = int(idx)
        end = min(start + hist_len, num_time_steps)
        if start < end:
            mask[start:end] = True
    return mask


def build_project_monthly_split_summary(
    time_meta: pd.DataFrame,
    split_indices: dict[str, list[int]],
    *,
    hist_len: int,
    pred_horizon: int,
) -> dict[str, object]:
    ts = pd.to_datetime(time_meta["timestamp"])
    dates = pd.to_datetime(time_meta["date"])
    split_labels = pd.Series(
        np.where(dates.dt.day <= 20, "train", np.where(dates.dt.day <= 24, "val", "test")),
        index=time_meta.index,
        dtype="object",
    )
    window_len = hist_len + pred_horizon

    def summarize_split(name: str) -> dict[str, object]:
        label_mask = split_labels == name
        segment_ts = ts[label_mask]
        indices = split_indices.get(name, [])
        if indices:
            first_window_start = int(indices[0])
            last_window_start = int(indices[-1])
            first_target_idx = first_window_start + hist_len
            last_target_idx = min(last_window_start + window_len - 1, len(ts) - 1)
        else:
            first_window_start = None
            last_window_start = None
            first_target_idx = None
            last_target_idx = None
        return {
            "first_labeled_timestamp": None if segment_ts.empty else pd.Timestamp(segment_ts.iloc[0]).isoformat(),
            "last_labeled_timestamp": None if segment_ts.empty else pd.Timestamp(segment_ts.iloc[-1]).isoformat(),
            "first_window_start_timestamp": None if first_window_start is None else pd.Timestamp(ts.iloc[first_window_start]).isoformat(),
            "last_window_start_timestamp": None if last_window_start is None else pd.Timestamp(ts.iloc[last_window_start]).isoformat(),
            "first_target_timestamp": None if first_target_idx is None else pd.Timestamp(ts.iloc[first_target_idx]).isoformat(),
            "last_target_timestamp": None if last_target_idx is None else pd.Timestamp(ts.iloc[last_target_idx]).isoformat(),
            "months": [] if segment_ts.empty else sorted(segment_ts.dt.to_period("M").astype(str).unique().tolist()),
            "weekday_counts": (
                {}
                if segment_ts.empty
                else {str(key): int(value) for key, value in segment_ts.dt.day_name().value_counts().sort_index().items()}
            ),
        }

    return {
        "mode": "project_monthly",
        "rule": {
            "train": "days 1-20",
            "val": "days 21-24",
            "test": "days 25+",
        },
        "calendar": {
            "train": summarize_split("train"),
            "val": summarize_split("val"),
            "test": summarize_split("test"),
        },
    }


def extract_temporal_context(node_seq: torch.Tensor) -> torch.Tensor | None:
    if node_seq.shape[-1] <= 2:
        return None
    return node_seq[:, 0, :, 2:]


def masked_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    target_mask: torch.Tensor | None,
) -> torch.Tensor:
    return masked_regression_loss(
        prediction,
        target,
        loss_name="mse",
        target_mask=target_mask,
    )


def _format_variant_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def build_preprocessing_variant_id(
    *,
    lower_quantile: float,
    upper_quantile: float,
    metric_protocol: str,
) -> str:
    return (
        f"train_quantile_clip_q{_format_variant_number(lower_quantile)}_"
        f"{_format_variant_number(upper_quantile)}_{metric_protocol}"
    )


def compute_processed_dataset_fingerprint(
    *,
    speed_values: np.ndarray,
    speed_valid_mask: np.ndarray,
    dataset_summary: dict,
) -> str:
    def sanitize_fingerprint_metadata(value: object) -> object:
        if isinstance(value, dict):
            cleaned: dict[str, object] = {}
            for key, item in value.items():
                key_text = str(key)
                if key_text.endswith("_file") or key_text.endswith("_path"):
                    continue
                cleaned[key_text] = sanitize_fingerprint_metadata(item)
            return cleaned
        if isinstance(value, list):
            return [sanitize_fingerprint_metadata(item) for item in value]
        return value

    hasher = hashlib.sha256()
    hasher.update(np.asarray(speed_values, dtype=np.float32).tobytes())
    hasher.update(np.asarray(speed_valid_mask, dtype=np.bool_).tobytes())
    hasher.update(
        json.dumps(
            sanitize_fingerprint_metadata(dataset_summary),
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    )
    return hasher.hexdigest()


def resolve_outlier_cleaning(
    *,
    speed_values: np.ndarray,
    speed_valid_mask: np.ndarray,
    train_time_mask: np.ndarray,
    mode: str,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    cleaned_mode = validate_outlier_cleaning_mode(mode)
    if cleaned_mode == "none":
        return (
            np.asarray(speed_values, dtype=np.float32).copy(),
            np.zeros_like(speed_valid_mask, dtype=bool),
            {
                "enabled": False,
                "method": "none",
                "fit_scope": "none",
                "apply_scope": "none",
                "replace_strategy": "none",
                "params": {},
                "cleaned_points": 0,
                "cleaned_ratio": 0.0,
            },
        )

    bounds = compute_train_quantile_clip_bounds(
        speed_values,
        speed_valid_mask,
        train_time_mask=train_time_mask,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    clipped = apply_valid_speed_clip(
        speed_values,
        speed_valid_mask,
        lower_bounds=bounds["lower"],
        upper_bounds=bounds["upper"],
    )
    summary = dict(clipped["summary"])
    summary.update(
        {
            "enabled": True,
            "method": cleaned_mode,
            "fit_scope": "train_history_only",
            "apply_scope": "all_valid_points",
            "replace_strategy": "clip",
            "params": {
                "lower_quantile": float(lower_quantile),
                "upper_quantile": float(upper_quantile),
            },
            "cleaned_points": int(summary["num_flagged"]),
            "cleaned_ratio": float(summary["flagged_ratio_valid"]),
            "train_valid_counts_min": int(np.min(bounds["valid_counts"])) if bounds["valid_counts"].size else 0,
            "train_valid_counts_max": int(np.max(bounds["valid_counts"])) if bounds["valid_counts"].size else 0,
            "sensors_without_train_valid": int(np.sum(bounds["valid_counts"] == 0)),
        }
    )
    return clipped["cleaned_speed_values"], clipped["outlier_mask"], summary


def is_early_stopping_enabled(early_stop_patience: int) -> bool:
    return int(early_stop_patience) > 0


def should_trigger_early_stop(
    *,
    should_validate: bool,
    epoch: int,
    early_stop_patience: int,
    min_epochs: int,
    validations_without_improvement: int,
) -> bool:
    return bool(
        should_validate
        and is_early_stopping_enabled(early_stop_patience)
        and epoch >= int(min_epochs)
        and int(validations_without_improvement) >= int(early_stop_patience)
    )


def build_training_control_summary(
    *,
    configured_epochs: int,
    history: list[dict],
    early_stop_patience: int,
    min_epochs: int,
    training_end_reason: str,
) -> dict[str, object]:
    completed_epochs = int(history[-1]["epoch"]) if history else 0
    completed_validations = int(sum(1 for record in history if record.get("val_raw_speed_rmse") is not None))
    early_stopping_enabled = is_early_stopping_enabled(early_stop_patience)
    early_stopping_triggered = training_end_reason == "early_stop_patience"
    return {
        "configured_epochs": int(configured_epochs),
        "completed_epochs": completed_epochs,
        "completed_validations": completed_validations,
        "completed_full_budget": completed_epochs >= int(configured_epochs),
        "training_end_reason": str(training_end_reason),
        "early_stopping": {
            "enabled": early_stopping_enabled,
            "patience": int(early_stop_patience),
            "min_epochs": int(min_epochs),
            "triggered": early_stopping_triggered,
        },
    }


def attach_speed_report_metrics_to_record(
    record: dict[str, object],
    speed_metrics: dict[str, object] | None,
    *,
    prefix: str = "val_raw_speed_rmse",
) -> dict[str, object]:
    if not isinstance(speed_metrics, dict):
        return record
    report = speed_metrics.get("report")
    if not isinstance(report, dict):
        return record

    for label, values in report.items():
        if not isinstance(values, dict) or "rmse" not in values:
            continue
        record[f"{prefix}_{str(label).lower()}"] = round(float(values["rmse"]), 5)
    return record


def collect_speed_node_trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    candidates: list[torch.nn.Parameter] = []
    modules_or_params = [
        getattr(base_model, "speed_node_proj", None),
        getattr(base_model, "vn_gtcn_fix", None),
        getattr(base_model, "vn_gat_fix", None),
        getattr(base_model, "speed_head", None),
    ]
    if getattr(base_model, "speed_use_adaptive", False):
        modules_or_params.extend(
            [
                getattr(base_model, "speed_node_emb_src", None),
                getattr(base_model, "speed_node_emb_dst", None),
                getattr(base_model, "vn_gtcn_adp", None),
                getattr(base_model, "vn_gat_adp", None),
                getattr(base_model, "speed_node_fusion", None),
            ]
        )

    seen: set[int] = set()
    for item in modules_or_params:
        if item is None:
            continue
        if isinstance(item, torch.nn.Parameter):
            params = [item]
        else:
            params = list(item.parameters())
        for param in params:
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            candidates.append(param)
    if not candidates:
        raise RuntimeError("No trainable parameters were collected for the node-speed path.")
    return candidates


def collect_speed_edge_trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    candidates: list[torch.nn.Parameter] = []
    modules_or_params = [
        getattr(base_model, "edge_proj", None),
        getattr(base_model, "e_gtcn", None),
        getattr(base_model, "e_gat", None),
        getattr(base_model, "speed_head", None),
    ]
    if getattr(base_model, "speed_use_adaptive", False):
        modules_or_params.extend(
            [
                getattr(base_model, "speed_emb_src", None),
                getattr(base_model, "speed_emb_dst", None),
                getattr(base_model, "e_gtcn_adp", None),
                getattr(base_model, "e_gat_adp", None),
                getattr(base_model, "speed_fusion", None),
            ]
        )

    seen: set[int] = set()
    for item in modules_or_params:
        if item is None:
            continue
        if isinstance(item, torch.nn.Parameter):
            params = [item]
        else:
            params = list(item.parameters())
        for param in params:
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            candidates.append(param)
    if not candidates:
        raise RuntimeError("No trainable parameters were collected for the edge-speed path.")
    return candidates


def evaluate_v_loss(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    non_blocking: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    v_loss_name: str,
    huber_delta: float,
    charbonnier_eps: float,
) -> float:
    total = 0.0
    batches = 0
    with torch.inference_mode():
        for batch in loader:
            node_seq = batch["node_seq"].to(device, non_blocking=non_blocking)
            speed_seq = batch["speed_seq"].to(device, non_blocking=non_blocking)
            speed_history_mask = batch.get("speed_history_mask")
            if speed_history_mask is not None:
                speed_history_mask = speed_history_mask.to(device, non_blocking=non_blocking)
            speed_target = batch["speed_target"].to(device, non_blocking=non_blocking)
            speed_target_mask = batch.get("speed_target_mask")
            if speed_target_mask is not None:
                speed_target_mask = speed_target_mask.to(device, non_blocking=non_blocking)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                speed_pred = model.forward_v(
                    speed_seq,
                    extract_temporal_context(node_seq),
                    speed_history_mask_seq=speed_history_mask,
                )
                loss = masked_regression_loss(
                    speed_pred,
                    speed_target,
                    loss_name=v_loss_name,
                    target_mask=speed_target_mask,
                    huber_delta=huber_delta,
                    charbonnier_eps=charbonnier_eps,
                )
            total += float(loss.item())
            batches += 1
    return total / max(batches, 1)


def build_resume_safe_run_config(run_config: dict) -> dict:
    return {
        key: run_config[key]
        for key in RESUME_CONFIG_COMPARABLE_KEYS
        if key in run_config
    }


def diff_resume_safe_run_config(saved_run_config: dict, expected_run_config: dict) -> dict[str, dict[str, object]]:
    mismatches: dict[str, dict[str, object]] = {}
    for key in sorted(set(saved_run_config).union(expected_run_config)):
        saved_value = saved_run_config.get(key, "__missing__")
        current_value = expected_run_config.get(key, "__missing__")
        if saved_value != current_value:
            mismatches[key] = {
                "saved": saved_value,
                "current": current_value,
            }
    return mismatches


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    history: list[dict],
    best_val_rmse: float,
    best_val_loss: float,
    best_epoch: int,
    validations_without_improvement: int,
    elapsed_offset: float,
    run_config: dict,
) -> None:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "best_val_rmse": best_val_rmse,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "validations_without_improvement": validations_without_improvement,
        "elapsed_offset": float(elapsed_offset),
        "run_config": run_config,
        "resume_safe_run_config": build_resume_safe_run_config(run_config),
    }
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
    expected_run_config: dict,
 ) -> tuple[int, list[dict], float, float, int, int, float]:
    state = torch.load(path, map_location=device, weights_only=False)
    saved_run_config = state.get("run_config")
    if saved_run_config is None:
        raise RuntimeError(
            f"Checkpoint {path} is missing run_config metadata and cannot be resumed safely."
        )
    saved_run_config = dict(saved_run_config)
    saved_resume_safe_run_config = state.get("resume_safe_run_config")
    if saved_resume_safe_run_config is None:
        saved_resume_safe_run_config = build_resume_safe_run_config(saved_run_config)
    expected_resume_safe_run_config = build_resume_safe_run_config(expected_run_config)
    mismatches = diff_resume_safe_run_config(
        dict(saved_resume_safe_run_config),
        expected_resume_safe_run_config,
    )
    if mismatches:
        raise RuntimeError(
            f"Checkpoint config mismatch for {path}. "
            f"Mismatches={mismatches}"
        )
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    history = list(state.get("history", []))
    saved_epoch = int(state["epoch"])
    if history:
        history_epochs = [int(record.get("epoch", -1)) for record in history]
        if history_epochs != sorted(history_epochs) or len(history_epochs) != len(set(history_epochs)):
            raise RuntimeError(f"Checkpoint {path} has non-monotonic or duplicate history epochs.")
        if history_epochs[-1] != saved_epoch:
            raise RuntimeError(
                f"Checkpoint {path} history tail mismatch: last_history_epoch={history_epochs[-1]} "
                f"saved_epoch={saved_epoch}"
            )
    elapsed_offset = state.get("elapsed_offset")
    if elapsed_offset is None:
        elapsed_offset = float(history[-1].get("elapsed", 0.0)) if history else 0.0
    return (
        saved_epoch,
        history,
        float(state.get("best_val_rmse", float("inf"))),
        float(state.get("best_val_loss", float("inf"))),
        int(state.get("best_epoch", 0)),
        int(state.get("validations_without_improvement", 0)),
        float(elapsed_offset),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the STGAT sensor-speed adaptation on METR-LA / PEMS-BAY.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Prepared dataset directory under data/external_datasets/processed.")
    parser.add_argument("--dataset-name", type=str, required=True, help="Display name used in logs and reports.")
    parser.add_argument("--log-dir", type=str, required=True, help="Output run directory.")
    parser.add_argument("--hist-len", type=int, default=12)
    parser.add_argument("--pred-horizon", type=int, default=12)
    parser.add_argument("--report-horizons-minutes", type=str, default="15,30,60")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-st-blocks", type=int, default=2)
    parser.add_argument("--num-gtcn-layers", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--adaptive-topk", type=int, default=16)
    parser.add_argument(
        "--epochs",
        type=int,
        default=180,
        help=(
            "Epoch budget for this invocation. When reusing an existing --log-dir checkpoint, "
            "the run continues for this many additional epochs."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam", choices=list(OPTIMIZER_CHOICES))
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--v-loss", type=str, default="mse", choices=list(V_LOSS_CHOICES))
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--charbonnier-eps", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp32"])
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument(
        "--split-policy",
        type=str,
        default="project_monthly",
        choices=list(SPLIT_POLICIES),
        help="Use the project monthly split by default; benchmark_contiguous keeps the legacy 70/10/20 ratio split.",
    )
    parser.add_argument(
        "--split-alignment",
        type=str,
        default="none",
        choices=list(SPLIT_ALIGNMENT_MODES),
        help="Only used when --split-policy=benchmark_contiguous.",
    )
    parser.add_argument(
        "--scheduler-monitor",
        type=str,
        default="val_rmse",
        choices=["val_loss", "val_rmse"],
        help="Validation metric used for ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--best-checkpoint-monitor",
        type=str,
        default="val_rmse",
        choices=["val_loss", "val_rmse"],
        help="Validation metric used to decide when stgat_best.pt should be updated.",
    )
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--scheduler-cooldown", type=int, default=2)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Number of non-improving validations before stopping early. Set to 0 to disable early stopping.",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=10,
        help="Minimum epochs before early stopping can trigger. Ignored when early stopping is disabled.",
    )
    parser.add_argument(
        "--stop-when-best-val-rmse-below",
        type=float,
        default=None,
        help=(
            "Optional absolute best validation RMSE target. If a validation reaches a strictly "
            "lower best val RMSE than this threshold, training stops gracefully after checkpointing."
        ),
    )
    parser.add_argument("--disable-time-features", action="store_true")
    parser.add_argument("--disable-adaptive", action="store_true")
    parser.add_argument(
        "--disable-weighted-fixed-graph",
        action="store_true",
        help="Fallback to binary fixed adjacency instead of weighted official adjacency.",
    )
    parser.add_argument(
        "--disable-missing-aware-history",
        action="store_true",
        help="Fallback to legacy zero-filled history without causal imputation or history-mask input.",
    )
    parser.add_argument("--disable-speed-mask-zero", action="store_true")
    parser.add_argument(
        "--outlier-cleaning-mode",
        type=str,
        default="train_quantile_clip",
        choices=list(OUTLIER_CLEANING_MODES),
        help="Speed cleaning mode applied before normalization. Defaults to train_quantile_clip.",
    )
    parser.add_argument("--outlier-lower-quantile", type=float, default=0.01)
    parser.add_argument("--outlier-upper-quantile", type=float, default=0.99)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", type=str, default="reduce-overhead", choices=["default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--fresh", action="store_true", help="Ignore any existing latest checkpoint in the log dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cleaned_outlier_mode = validate_outlier_cleaning_mode(args.outlier_cleaning_mode)
    if not 0.0 < args.outlier_lower_quantile < args.outlier_upper_quantile < 1.0:
        raise SystemExit("Outlier quantiles must satisfy 0 < lower < upper < 1.")
    split_policy = str(args.split_policy)
    split_alignment = validate_split_alignment(args.split_alignment)

    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    precision = resolve_precision(device, args.precision)
    amp_enabled = device.type == "cuda" and precision == "bf16"
    amp_dtype = torch.bfloat16 if amp_enabled else None
    num_workers = resolve_num_workers(args.num_workers, device)
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    dataset, time_feature_names, time_meta = load_prepared_sensor_dataset(
        args.dataset_dir,
        disable_time_features=args.disable_time_features,
    )
    report_minutes = parse_report_horizons_minutes(args.report_horizons_minutes)
    time_slot_minutes = int(
        round(
            (
                pd.to_datetime(time_meta["timestamp"]).iloc[1]
                - pd.to_datetime(time_meta["timestamp"]).iloc[0]
            ).total_seconds()
            / 60.0
        )
    )
    report_horizons = resolve_report_horizons(
        time_slot_minutes=time_slot_minutes,
        pred_horizon=args.pred_horizon,
        requested_minutes=report_minutes,
        strict=bool(report_minutes),
    )

    adjacency = dataset["adj"]
    adjacency_weights = dataset["adjacency_weights"]
    edge_index = dataset["edge_index"]
    edge_lengths = dataset["edge_lengths"]
    node_features = dataset["node_features"]
    speed_values = dataset["edge_speeds"]
    speed_valid_mask = dataset["speed_valid_mask"]
    dataset_summary = dataset.get("dataset_summary", {})
    representation_domain = dataset.get("representation_domain", "sensor_node")
    speed_item_count = int(speed_values.shape[1])
    v_domain = "edge" if representation_domain == "pseudo_edge" else "node"
    metric_protocol = "masked_zero_excluded" if not args.disable_speed_mask_zero else "unmasked_all_values"
    processed_dataset_fingerprint = compute_processed_dataset_fingerprint(
        speed_values=speed_values,
        speed_valid_mask=speed_valid_mask,
        dataset_summary=dataset_summary,
    )
    num_nodes = adjacency.shape[0]
    num_time_steps = speed_values.shape[0]
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency_matrix.npy must have shape (N, N).")
    if adjacency_weights.shape != adjacency.shape:
        raise ValueError("adjacency_weights.npy must have the same shape as adjacency_matrix.npy.")
    if speed_values.ndim != 2:
        raise ValueError("speed_values.npy must have shape (T, M).")
    if time_meta.shape[0] != num_time_steps:
        raise ValueError("time_meta.csv length must match the time dimension of speed_values.npy.")
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError("edge_index.npy must have shape (E, 2).")
    if edge_lengths.ndim != 1 or edge_lengths.shape[0] != edge_index.shape[0]:
        raise ValueError("edge_lengths_km.npy must have shape (E,).")
    if edge_index.size > 0:
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes:
            raise ValueError("edge_index.npy contains node ids outside the adjacency range.")
    if representation_domain == "sensor_node":
        if speed_item_count != num_nodes:
            raise ValueError("sensor_node datasets require speed_values.npy second dimension to match adjacency size.")
    elif representation_domain == "pseudo_edge":
        if speed_item_count != edge_index.shape[0]:
            raise ValueError("pseudo_edge datasets require speed_values.npy second dimension to match edge_index.npy rows.")
    else:
        raise ValueError(f"Unsupported representation_domain: {representation_domain}")

    time_meta = time_meta.copy()
    time_meta["date"] = pd.to_datetime(time_meta["date"], errors="coerce")
    if time_meta["date"].isna().any():
        raise ValueError("time_meta.csv must include a valid date column for project monthly splitting.")
    if split_policy == "project_monthly":
        split_indices = build_monthly_split_indices(time_meta, args.hist_len, args.pred_horizon)
        split_summary = build_project_monthly_split_summary(
            time_meta,
            split_indices,
            hist_len=args.hist_len,
            pred_horizon=args.pred_horizon,
        )
    else:
        train_end, val_end, split_summary = resolve_split_boundaries(
            time_meta,
            alignment=split_alignment,
        )
        split_indices = build_time_contained_split_indices(
            num_time_steps,
            hist_len=args.hist_len,
            pred_horizon=args.pred_horizon,
            train_end=train_end,
            val_end=val_end,
        )
    train_window_mask = build_window_time_mask(
        num_time_steps,
        split_indices["train"],
        args.hist_len,
        args.pred_horizon,
    )
    train_history_mask = build_history_time_mask(
        num_time_steps,
        split_indices["train"],
        args.hist_len,
    )
    speed_values, speed_outlier_mask, outlier_cleaning_summary = resolve_outlier_cleaning(
        speed_values=speed_values,
        speed_valid_mask=speed_valid_mask,
        train_time_mask=train_history_mask,
        mode=cleaned_outlier_mode,
        lower_quantile=float(args.outlier_lower_quantile),
        upper_quantile=float(args.outlier_upper_quantile),
    )
    normalization_stats = build_normalization_stats(
        node_features,
        speed_values,
        train_window_mask,
        speed_valid_mask=speed_valid_mask,
    )
    node_features = normalize_node_features(node_features, normalization_stats)
    speed_values = normalize_speed_features(
        speed_values,
        normalization_stats,
        edge_axis=1,
        speed_valid_mask=speed_valid_mask,
    )

    use_weighted_fixed_graph = not args.disable_weighted_fixed_graph
    use_missing_aware_history = not args.disable_missing_aware_history
    use_fixed_edge_length_feature, fixed_edge_length_feature_mode = resolve_fixed_edge_length_feature_mode(
        dataset_summary,
        representation_domain=representation_domain,
    )

    full_dataset = SpatioTemporalDataset(
        node_features,
        speed_values,
        edge_speed_valid_mask=speed_valid_mask,
        edge_speed_history_valid_mask=(
            speed_valid_mask if use_missing_aware_history else None
        ),
        history_imputation_enabled=use_missing_aware_history,
        hist_len=args.hist_len,
        pred_horizon=args.pred_horizon,
    )
    train_dataset = Subset(full_dataset, split_indices["train"])
    val_dataset = Subset(full_dataset, split_indices["val"])
    test_dataset = Subset(full_dataset, split_indices["test"])

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = STGATPredictor(
        num_nodes=num_nodes,
        edge_index=torch.from_numpy(edge_index),
        edge_lengths=torch.from_numpy(edge_lengths),
        adj_matrix=torch.from_numpy(adjacency),
        adj_weight_matrix=(
            torch.from_numpy(adjacency_weights) if use_weighted_fixed_graph else None
        ),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_st_blocks=args.num_st_blocks,
        num_gtcn_layers=args.num_gtcn_layers,
        kernel_size=args.kernel_size,
        pred_horizon=args.pred_horizon,
        node_feat_dim=int(node_features.shape[-1]),
        adaptive_topk=args.adaptive_topk,
        speed_use_adaptive=not args.disable_adaptive,
        use_speed_history_mask=use_missing_aware_history,
        use_fixed_edge_length_feature=use_fixed_edge_length_feature,
        v_domain=v_domain,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=args.compile_mode)

    trainable_parameters = (
        collect_speed_edge_trainable_parameters(model)
        if v_domain == "edge"
        else collect_speed_node_trainable_parameters(model)
    )
    optimizer = build_optimizer(
        trainable_parameters,
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(args.scheduler_factor),
        patience=int(args.scheduler_patience),
        cooldown=int(args.scheduler_cooldown),
        min_lr=float(args.scheduler_min_lr),
    )
    run_config = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "dataset_name": args.dataset_name,
        "processed_dataset_fingerprint": processed_dataset_fingerprint,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "hist_len": int(args.hist_len),
        "pred_horizon": int(args.pred_horizon),
        "report_horizons_minutes": list(report_minutes),
        "hidden_dim": int(args.hidden_dim),
        "num_heads": int(args.num_heads),
        "num_st_blocks": int(args.num_st_blocks),
        "num_gtcn_layers": int(args.num_gtcn_layers),
        "kernel_size": int(args.kernel_size),
        "node_feat_dim": int(node_features.shape[-1]),
        "adaptive_topk": int(args.adaptive_topk),
        "adaptive_enabled": bool(not args.disable_adaptive),
        "fixed_graph_weighted": bool(use_weighted_fixed_graph),
        "use_fixed_edge_length_feature": bool(use_fixed_edge_length_feature),
        "fixed_edge_length_feature_mode": str(fixed_edge_length_feature_mode),
        "use_speed_history_mask": bool(use_missing_aware_history),
        "disable_time_features": bool(args.disable_time_features),
        "disable_speed_mask_zero": bool(args.disable_speed_mask_zero),
        "batch_size": int(args.batch_size),
        "optimizer": str(args.optimizer),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_epochs": int(args.warmup_epochs),
        "grad_clip_norm": float(args.grad_clip_norm),
        "v_loss": str(args.v_loss),
        "huber_delta": float(args.huber_delta),
        "charbonnier_eps": float(args.charbonnier_eps),
        "val_interval": int(args.val_interval),
        "split_policy": split_policy,
        "split_alignment": split_alignment,
        "scheduler_monitor": str(args.scheduler_monitor),
        "best_checkpoint_monitor": str(args.best_checkpoint_monitor),
        "scheduler_factor": float(args.scheduler_factor),
        "scheduler_patience": int(args.scheduler_patience),
        "scheduler_cooldown": int(args.scheduler_cooldown),
        "scheduler_min_lr": float(args.scheduler_min_lr),
        "early_stopping_enabled": bool(is_early_stopping_enabled(args.early_stop_patience)),
        "early_stop_patience": int(args.early_stop_patience),
        "min_epochs": int(args.min_epochs),
        "stop_when_best_val_rmse_below": (
            float(args.stop_when_best_val_rmse_below)
            if args.stop_when_best_val_rmse_below is not None
            else None
        ),
        "graph_source": dataset_summary.get("graph_source", {}),
        "representation_domain": representation_domain,
        "v_domain": v_domain,
        "num_speed_items": speed_item_count,
        "num_trainable_parameters": int(sum(param.numel() for param in trainable_parameters)),
        "metric_protocol": metric_protocol,
        "preprocessing_variant_id": build_preprocessing_variant_id(
            lower_quantile=float(args.outlier_lower_quantile),
            upper_quantile=float(args.outlier_upper_quantile),
            metric_protocol=metric_protocol,
        ),
        "outlier_cleaning": outlier_cleaning_summary,
    }

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = log_dir / "stgat_latest.pt"
    best_ckpt = log_dir / "stgat_best.pt"
    speed_outlier_mask_path = log_dir / "speed_outlier_mask.npy"
    saved_speed_outlier_mask_path = None

    history: list[dict] = []
    best_val_rmse = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    start_epoch = 1
    epoch_offset = 0
    elapsed_offset = 0.0
    validations_without_improvement = 0
    training_end_reason = "epoch_budget_exhausted"
    if latest_ckpt.exists() and not args.fresh:
        (
            completed_epoch,
            history,
            best_val_rmse,
            best_val_loss,
            best_epoch,
            validations_without_improvement,
            elapsed_offset,
        ) = load_checkpoint(
            latest_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            expected_run_config=run_config,
        )
        epoch_offset = completed_epoch
        start_epoch = epoch_offset + 1
        print(
            f"Resuming {args.dataset_name} from epoch {completed_epoch} ({latest_ckpt}) | "
            f"target_epoch={epoch_offset + int(args.epochs)}"
        )

    with (log_dir / "prepared_dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_summary, handle, ensure_ascii=False, indent=2)
    if representation_domain == "pseudo_edge":
        pseudo_edge_summary_path = Path(args.dataset_dir) / "pseudo_edge_summary.json"
        if pseudo_edge_summary_path.exists():
            pseudo_edge_summary = json.loads(pseudo_edge_summary_path.read_text(encoding="utf-8"))
            with (log_dir / "prepared_pseudo_edge_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(pseudo_edge_summary, handle, ensure_ascii=False, indent=2)
    if outlier_cleaning_summary["enabled"]:
        np.save(speed_outlier_mask_path, speed_outlier_mask.astype(np.bool_))
        saved_speed_outlier_mask_path = speed_outlier_mask_path
    elif speed_outlier_mask_path.exists():
        speed_outlier_mask_path.unlink()

    print(
        f"[{args.dataset_name}] representation={representation_domain} | graph_nodes={num_nodes} | "
        f"graph_edges={edge_index.shape[0]} | speed_items={speed_item_count} | "
        f"time_steps={num_time_steps} | slot={time_slot_minutes} min | "
        f"report={report_horizons['resolved_minutes']} min"
    )
    print(
        f"[{args.dataset_name}] train/val/test samples="
        f"{len(train_dataset)}/{len(val_dataset)}/{len(test_dataset)} | "
        f"adaptive={'on' if not args.disable_adaptive else 'off'} | "
        f"device={device} | precision={precision} | num_workers={num_workers}"
    )
    print(
        f"[{args.dataset_name}] graph_source={dataset_summary.get('graph_source', {}).get('mode', 'unknown')} | "
        f"metric_protocol={metric_protocol} | "
        f"variant={run_config['preprocessing_variant_id']} | "
        f"split_policy={split_policy} | "
        f"split_alignment={split_alignment if split_policy == 'benchmark_contiguous' else 'n/a'} | "
        f"fixed_graph={'weighted' if use_weighted_fixed_graph else 'binary'} | "
        f"fixed_edge_length={'on' if use_fixed_edge_length_feature else 'off'} "
        f"({fixed_edge_length_feature_mode}) | "
        f"history_missing={'causal_ffill_plus_mask' if use_missing_aware_history else 'zero_fill_no_mask'} | "
        f"optimizer={args.optimizer} | "
        f"v_loss={args.v_loss} | "
        f"scheduler={args.scheduler_monitor}/patience{args.scheduler_patience}/cooldown{args.scheduler_cooldown}/minlr{args.scheduler_min_lr:g} | "
        f"best_ckpt={args.best_checkpoint_monitor} | "
        f"early_stopping={'on' if run_config['early_stopping_enabled'] else 'off'} "
        f"(patience={args.early_stop_patience}, min_epochs={args.min_epochs}) | "
        f"outlier_cleaning={outlier_cleaning_summary['method']} "
        f"({outlier_cleaning_summary['cleaned_points']} points) | "
        f"trainable_params={sum(param.numel() for param in trainable_parameters):,}"
    )

    started_at = time.time()
    final_epoch = epoch_offset + int(args.epochs)
    if (
        args.stop_when_best_val_rmse_below is not None
        and np.isfinite(best_val_rmse)
        and float(best_val_rmse) < float(args.stop_when_best_val_rmse_below)
    ):
        training_end_reason = "best_val_target_already_reached"
        print(
            f"[{args.dataset_name}] Target already reached by resumed checkpoint: "
            f"best_val_rmse={best_val_rmse:.4f} < {float(args.stop_when_best_val_rmse_below):.4f}. "
            "Skipping further epochs."
        )
    warmup_steps = max(int(args.warmup_epochs), 0) * max(len(train_loader), 1)
    global_step = max(epoch_offset, 0) * max(len(train_loader), 1)
    for epoch in range(start_epoch, final_epoch + 1):
        if training_end_reason == "best_val_target_already_reached":
            break
        model.train()
        train_total = 0.0
        train_batches = 0
        lr_at_epoch_start = float(optimizer.param_groups[0]["lr"])
        for batch in train_loader:
            global_step += 1
            maybe_apply_linear_warmup(
                optimizer,
                base_lr=float(args.lr),
                global_step=global_step,
                warmup_steps=warmup_steps,
            )
            node_seq = batch["node_seq"].to(device, non_blocking=non_blocking)
            speed_seq = batch["speed_seq"].to(device, non_blocking=non_blocking)
            speed_history_mask = batch.get("speed_history_mask")
            if speed_history_mask is not None:
                speed_history_mask = speed_history_mask.to(device, non_blocking=non_blocking)
            speed_target = batch["speed_target"].to(device, non_blocking=non_blocking)
            speed_target_mask = batch.get("speed_target_mask")
            if speed_target_mask is not None:
                speed_target_mask = speed_target_mask.to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                speed_pred = model.forward_v(
                    speed_seq,
                    extract_temporal_context(node_seq),
                    speed_history_mask_seq=speed_history_mask,
                )
                loss = masked_regression_loss(
                    speed_pred,
                    speed_target,
                    loss_name=args.v_loss,
                    target_mask=speed_target_mask,
                    huber_delta=float(args.huber_delta),
                    charbonnier_eps=float(args.charbonnier_eps),
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
            optimizer.step()

            train_total += float(loss.item())
            train_batches += 1

        train_loss = train_total / max(train_batches, 1)
        should_validate = (epoch % max(args.val_interval, 1) == 0) or (epoch == final_epoch)
        val_loss = None
        val_raw_metrics = None
        val_rmse = None
        scheduler_metric = None
        if should_validate:
            model.eval()
            val_loss = evaluate_v_loss(
                model,
                val_loader,
                device=device,
                non_blocking=non_blocking,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                v_loss_name=args.v_loss,
                huber_delta=float(args.huber_delta),
                charbonnier_eps=float(args.charbonnier_eps),
            )
            val_raw_metrics = evaluate_loader_raw_metrics(
                model,
                val_loader,
                train_task="v",
                device=device,
                non_blocking=non_blocking,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                normalization_stats=normalization_stats,
                time_slot_minutes=time_slot_minutes,
                report_horizons=report_horizons,
                speed_metric_mask_zeros=bool(not args.disable_speed_mask_zero),
            )
            val_rmse = float(val_raw_metrics["speed"]["rmse"])
            scheduler_metric = float(val_loss) if args.scheduler_monitor == "val_loss" else float(val_rmse)
            best_checkpoint_metric = float(val_loss) if args.best_checkpoint_monitor == "val_loss" else float(val_rmse)
            previous_best_checkpoint_metric = (
                float(best_val_loss) if args.best_checkpoint_monitor == "val_loss" else float(best_val_rmse)
            )
            scheduler.step(scheduler_metric)
            if best_checkpoint_metric < previous_best_checkpoint_metric:
                best_val_rmse = float(val_rmse)
                best_val_loss = float(val_loss)
                best_epoch = epoch
                validations_without_improvement = 0
                save_checkpoint(
                    best_ckpt,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    history=history,
                    best_val_rmse=best_val_rmse,
                    best_val_loss=best_val_loss,
                    best_epoch=best_epoch,
                    validations_without_improvement=validations_without_improvement,
                    elapsed_offset=elapsed_offset + (time.time() - started_at),
                    run_config=run_config,
                )
            else:
                validations_without_improvement += 1

        elapsed = elapsed_offset + (time.time() - started_at)
        seconds_per_epoch = elapsed / max(epoch, 1)
        remaining = seconds_per_epoch * max(final_epoch - epoch, 0)
        eta = datetime.now().astimezone() + timedelta(seconds=remaining)
        record = {
            "epoch": epoch,
            "lr": lr_at_epoch_start,
            "lr_next": float(optimizer.param_groups[0]["lr"]),
            "elapsed": round(elapsed, 1),
            "train_v": round(train_loss, 5),
            "val_v": round(val_loss, 5) if val_loss is not None else None,
            "val_raw_speed_rmse": round(val_rmse, 5) if val_rmse is not None else None,
            "scheduler_metric_name": str(args.scheduler_monitor) if scheduler_metric is not None else None,
            "scheduler_metric": round(float(scheduler_metric), 5) if scheduler_metric is not None else None,
            "best_checkpoint_metric_name": str(args.best_checkpoint_monitor) if val_loss is not None else None,
            "best_checkpoint_metric": (
                round(float(best_val_loss if args.best_checkpoint_monitor == "val_loss" else best_val_rmse), 5)
                if val_loss is not None
                else None
            ),
            "best_val_raw_speed_rmse": round(best_val_rmse, 5) if np.isfinite(best_val_rmse) else None,
            "eta": eta.isoformat(timespec="seconds"),
        }
        attach_speed_report_metrics_to_record(record, val_raw_metrics.get("speed") if val_raw_metrics is not None else None)
        history.append(record)
        save_checkpoint(
            latest_ckpt,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            history=history,
            best_val_rmse=best_val_rmse,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            validations_without_improvement=validations_without_improvement,
            elapsed_offset=elapsed,
            run_config=run_config,
        )

        if epoch % args.log_interval == 0 or epoch == 1 or should_validate:
            if should_validate and val_raw_metrics is not None:
                report = val_raw_metrics["speed"]["report"]
                report_text = " ".join(
                    f"{label}:RMSE={values['rmse']:.3f}"
                    for label, values in report.items()
                )
                print(
                    f"[{args.dataset_name}][Ep {epoch:>3d}] TrainV={train_loss:.4f} "
                    f"ValV={val_loss:.4f} ValRMSE={val_rmse:.4f} Best={best_val_rmse:.4f} "
                    f"| {report_text} | ETA={eta:%Y-%m-%d %H:%M}"
                )
            else:
                best_text = f"{best_val_rmse:.4f}" if np.isfinite(best_val_rmse) else "nan"
                print(
                    f"[{args.dataset_name}][Ep {epoch:>3d}] TrainV={train_loss:.4f} "
                    f"Best={best_text} "
                    f"| ETA={eta:%Y-%m-%d %H:%M}"
                )

        if should_trigger_early_stop(
            should_validate=should_validate,
            epoch=epoch,
            early_stop_patience=args.early_stop_patience,
            min_epochs=args.min_epochs,
            validations_without_improvement=validations_without_improvement,
        ):
            training_end_reason = "early_stop_patience"
            print(
                f"[{args.dataset_name}] Early stopping at epoch {epoch} after "
                f"{validations_without_improvement} validations without improvement."
            )
            break

        if (
            should_validate
            and args.stop_when_best_val_rmse_below is not None
            and np.isfinite(best_val_rmse)
            and float(best_val_rmse) < float(args.stop_when_best_val_rmse_below)
        ):
            training_end_reason = "best_val_target_reached"
            print(
                f"[{args.dataset_name}] Target reached at epoch {epoch}: "
                f"best_val_rmse={best_val_rmse:.4f} < {float(args.stop_when_best_val_rmse_below):.4f}. "
                "Stopping after checkpoint + metadata update."
            )
            break

    if not best_ckpt.exists():
        raise RuntimeError(f"No best checkpoint was produced for {args.dataset_name}.")

    training_control = build_training_control_summary(
        configured_epochs=final_epoch,
        history=history,
        early_stop_patience=args.early_stop_patience,
        min_epochs=args.min_epochs,
        training_end_reason=training_end_reason,
    )

    best_state = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(best_state["model_state"])
    model.eval()

    test_loss = evaluate_v_loss(
        model,
        test_loader,
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        v_loss_name=args.v_loss,
        huber_delta=float(args.huber_delta),
        charbonnier_eps=float(args.charbonnier_eps),
    )
    test_raw_metrics = evaluate_loader_raw_metrics(
        model,
        test_loader,
        train_task="v",
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        normalization_stats=normalization_stats,
        time_slot_minutes=time_slot_minutes,
        report_horizons=report_horizons,
        speed_metric_mask_zeros=bool(not args.disable_speed_mask_zero),
    )
    val_raw_at_best = None
    val_raw_report = None
    if len(history) > 0:
        model.load_state_dict(best_state["model_state"])
        val_raw_at_best = evaluate_loader_raw_metrics(
            model,
            val_loader,
            train_task="v",
            device=device,
            non_blocking=non_blocking,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            normalization_stats=normalization_stats,
            time_slot_minutes=time_slot_minutes,
            report_horizons=report_horizons,
            speed_metric_mask_zeros=bool(not args.disable_speed_mask_zero),
        )
        val_raw_report = val_raw_at_best["speed"]["report"]

    meta = {
        "dataset_name": args.dataset_name,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "processed_dataset_fingerprint": processed_dataset_fingerprint,
        "seed": int(args.seed),
        "preprocessing_variant_id": run_config["preprocessing_variant_id"],
        "representation_domain": representation_domain,
        "representation_variant_id": dataset_summary.get("representation_variant_id", representation_domain),
        "split_policy": split_policy,
        "split_alignment": split_alignment if split_policy == "benchmark_contiguous" else "n/a",
        "num_sensors": int(dataset_summary.get("num_sensors", speed_item_count)),
        "num_nodes": int(num_nodes),
        "num_graph_nodes": int(num_nodes),
        "num_graph_edges": int(edge_index.shape[0]),
        "num_speed_items": int(speed_item_count),
        "time_steps": int(num_time_steps),
        "hidden_dim": int(args.hidden_dim),
        "training_control": training_control,
        "training_target": {
            "best_val_raw_speed_rmse_below": (
                float(args.stop_when_best_val_rmse_below)
                if args.stop_when_best_val_rmse_below is not None
                else None
            ),
        },
        "resume": {
            "resumed": bool(epoch_offset > 0),
            "resume_from_epoch": int(epoch_offset),
            "requested_additional_epochs": int(args.epochs),
            "target_total_epochs": int(final_epoch),
        },
        "hist_len": int(args.hist_len),
        "pred_horizon": int(args.pred_horizon),
        "time_slot_minutes": int(time_slot_minutes),
        "report_horizons": report_horizons,
        "split_strategy": (
            CALENDAR_SPLIT_STRATEGY
            if split_policy == "project_monthly"
            else (
                "contiguous_time_70_10_20_full_window_containment"
                if split_alignment == "none"
                else f"contiguous_time_70_10_20_full_window_containment_{split_alignment}_aligned"
            )
        ),
        "split_description": (
            PROJECT_MONTHLY_SPLIT_DESCRIPTION
            if split_policy == "project_monthly"
            else "Contiguous 70/10/20 time split with full history+target window containment."
        ),
        "split_summary": split_summary,
        "split_counts": {
            "train": len(train_dataset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "graph_topology": {
            "v_domain": v_domain,
            "adaptive_enabled": not args.disable_adaptive,
            "adaptive_topk": int(args.adaptive_topk),
            "fixed_graph_weighted": bool(use_weighted_fixed_graph),
            "fixed_edge_length_feature_enabled": bool(use_fixed_edge_length_feature),
            "fixed_edge_length_feature_mode": str(fixed_edge_length_feature_mode),
            "history_missing_mode": (
                "causal_ffill_plus_mask" if use_missing_aware_history else "zero_fill_no_mask"
            ),
            "fixed_graph_source": dataset_summary.get("graph_source", {}),
            "adaptive_graph": (
                (model._orig_mod if hasattr(model, "_orig_mod") else model).speed_adaptive_graph_summary()
            ),
        },
        "node_feat_dim": int(node_features.shape[-1]),
        "time_feature_names": time_feature_names,
        "speed_missing_ratio": float(dataset_summary.get("speed_missing_ratio", 0.0)),
        "speed_metric_protocol": metric_protocol,
        "prepared_dataset_summary_snapshot": str((log_dir / "prepared_dataset_summary.json").resolve()),
        "prepared_pseudo_edge_summary_snapshot": (
            str((log_dir / "prepared_pseudo_edge_summary.json").resolve())
            if (log_dir / "prepared_pseudo_edge_summary.json").exists()
            else None
        ),
        "outlier_cleaning": {
            **outlier_cleaning_summary,
            "mask_path": str(saved_speed_outlier_mask_path) if saved_speed_outlier_mask_path is not None else None,
        },
        "benchmark_comparability": {
            "profile": (
                "cleaned_train_quantile_clip"
                if outlier_cleaning_summary["enabled"]
                else ("raw_masked_speed" if metric_protocol == "masked_zero_excluded" else "raw_unmasked_speed")
            ),
            "is_official_like": bool(
                str(dataset_summary.get("graph_source", {}).get("mode", "")).startswith("official_")
                and not outlier_cleaning_summary["enabled"]
                and metric_protocol == "masked_zero_excluded"
                and split_policy == "benchmark_contiguous"
                and split_alignment == "none"
            ),
            "deviations": [
                *(
                    ["train-only quantile clipping applied before normalization"]
                    if outlier_cleaning_summary["enabled"]
                    else []
                ),
                *(
                    ["speed metrics include masked-zero disabling and are not official-like"]
                    if metric_protocol != "masked_zero_excluded"
                    else []
                ),
                *(
                    ["project monthly split used instead of external benchmark contiguous split"]
                    if split_policy == "project_monthly"
                    else []
                ),
                *(
                    [f"validation/test split boundaries aligned to {split_alignment} starts instead of raw ratio boundaries"]
                    if split_policy == "benchmark_contiguous" and split_alignment != "none"
                    else []
                ),
                *(
                    ["experimental pseudo-edge representation used instead of the official sensor-node benchmark target"]
                    if representation_domain == "pseudo_edge"
                    else []
                ),
                "custom STGAT adaptation rather than the original benchmark model",
            ],
        },
        "num_trainable_parameters": int(sum(param.numel() for param in trainable_parameters)),
        "optimizer": {
            "name": str(args.optimizer).capitalize(),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "warmup_epochs": int(args.warmup_epochs),
            "grad_clip_norm": float(args.grad_clip_norm),
            "train_scope": "speed_edge_path_only" if v_domain == "edge" else "speed_node_path_only",
        },
        "loss": {
            "v_loss": str(args.v_loss),
            "huber_delta": float(args.huber_delta),
            "charbonnier_eps": float(args.charbonnier_eps),
        },
        "scheduler": {
            "name": "ReduceLROnPlateau",
            "monitor": str(args.scheduler_monitor),
            "factor": float(args.scheduler_factor),
            "patience": int(args.scheduler_patience),
            "cooldown": int(args.scheduler_cooldown),
            "min_lr": float(args.scheduler_min_lr),
        },
        "normalization": serialize_normalization_stats(normalization_stats),
        "selected_checkpoint": {
            "path": str(best_ckpt),
            "best_epoch": int(best_epoch),
            "monitor": str(args.best_checkpoint_monitor),
            "best_monitor_value": float(best_val_loss if args.best_checkpoint_monitor == "val_loss" else best_val_rmse),
            "best_val_raw_speed_rmse": float(best_val_rmse),
            "best_val_loss": float(best_val_loss),
        },
    }
    with (log_dir / "stgat_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    with (log_dir / "predictor_log.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)

    metrics_payload = {
        "loss_space": "normalized",
        "normalized_loss": {
            "v": float(test_loss),
            "speed": float(test_loss),
        },
        "val_normalized_loss": {
            "v": float(best_val_loss),
            "speed": float(best_val_loss),
        },
        "raw_metrics": {
            "speed": {
                "mse": float(test_raw_metrics["speed"]["mse"]),
                "rmse": float(test_raw_metrics["speed"]["rmse"]),
                "mae": float(test_raw_metrics["speed"]["mae"]),
                "mape": float(test_raw_metrics["speed"]["mape"]),
            }
        },
        "raw_metrics_per_step": {
            "speed": test_raw_metrics["speed"]["per_step"],
        },
        "raw_metrics_report": {
            "speed": test_raw_metrics["speed"]["report"],
        },
        "report_horizons": test_raw_metrics["report_horizons"],
        "val_raw_metrics": (
            {
                "speed": {
                    "mse": None if not np.isfinite(best_val_rmse) else float(val_raw_at_best["speed"]["mse"]),
                    "rmse": None if not np.isfinite(best_val_rmse) else float(val_raw_at_best["speed"]["rmse"]),
                    "mae": None if not np.isfinite(best_val_rmse) else float(val_raw_at_best["speed"]["mae"]),
                    "mape": None if not np.isfinite(best_val_rmse) else float(val_raw_at_best["speed"]["mape"]),
                }
            }
            if val_raw_report is not None
            else None
        ),
        "val_raw_metrics_per_step": (
            {"speed": val_raw_at_best["speed"]["per_step"]}
            if val_raw_report is not None
            else None
        ),
        "val_raw_metrics_report": (
            {"speed": val_raw_report}
            if val_raw_report is not None
            else None
        ),
        "selected_checkpoint": "stgat_best.pt",
        "selected_checkpoint_task": "v",
        "selected_checkpoint_metric_name": (
            "val_loss" if args.best_checkpoint_monitor == "val_loss" else "val_raw_speed_rmse"
        ),
        "selected_checkpoint_metric": float(best_val_loss if args.best_checkpoint_monitor == "val_loss" else best_val_rmse),
        "train_task": "v",
        "split_strategy": meta["split_strategy"],
        "split_description": meta["split_description"],
        "training_control": training_control,
        "speed_use_adaptive": bool(not args.disable_adaptive),
        "fixed_graph_weighted": bool(use_weighted_fixed_graph),
        "use_fixed_edge_length_feature": bool(use_fixed_edge_length_feature),
        "fixed_edge_length_feature_mode": str(fixed_edge_length_feature_mode),
        "use_speed_history_mask": bool(use_missing_aware_history),
        "history_missing_mode": (
            "causal_ffill_plus_mask" if use_missing_aware_history else "zero_fill_no_mask"
        ),
        "representation_domain": representation_domain,
        "representation_variant_id": meta["representation_variant_id"],
        "v_domain": v_domain,
        "speed_topology_mode": "pseudo_edge_graph" if v_domain == "edge" else "sensor_node_graph",
        "speed_metric_protocol": metric_protocol,
        "processed_dataset_fingerprint": processed_dataset_fingerprint,
        "preprocessing_variant_id": run_config["preprocessing_variant_id"],
        "optimizer": meta["optimizer"],
        "scheduler": meta["scheduler"],
        "split_summary": split_summary,
        "outlier_cleaning": {
            **outlier_cleaning_summary,
            "mask_path": str(saved_speed_outlier_mask_path) if saved_speed_outlier_mask_path is not None else None,
        },
        "benchmark_comparability": meta["benchmark_comparability"],
    }
    with (log_dir / "predictor_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)

    report_text = " ".join(
        f"{label}:RMSE={values['rmse']:.3f}"
        for label, values in test_raw_metrics["speed"]["report"].items()
    )
    print(
        f"[{args.dataset_name}] TestV={test_loss:.4f} "
        f"RMSE={test_raw_metrics['speed']['rmse']:.4f} "
        f"MAPE={test_raw_metrics['speed']['mape']:.2f}% | {report_text}"
    )


if __name__ == "__main__":
    main()
