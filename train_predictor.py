"""
train_predictor.py — STGAT 預測模型訓練腳本

訓練流程：
  1. 生成或載入時空資料
  2. 建構 SpatioTemporalDataset 與 DataLoader
  3. 初始化 STGATPredictor
  4. 多任務損失訓練：L = λ₁·L_demand + λ₂·L_supply + λ₃·L_speed
  5. 記錄各項指標並儲存最佳模型

使用方式（需先準備 data/ 真實路網與 build_speed_features 產物）：
    python train_predictor.py
    python train_predictor.py --epochs 100 --device auto --precision bf16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data_loader import SpatioTemporalDataset, load_nyc_real_graph_features
from predictor_normalization import (
    build_normalization_stats,
    denormalize_count_values,
    denormalize_speed_values,
    normalize_node_features,
    normalize_speed_features,
    serialize_normalization_stats,
)
from stgat_model import STGATPredictor


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 CUDA，但目前環境沒有可用 GPU。")
    return device


def resolve_precision(device: torch.device, precision_arg: str) -> str:
    if device.type != "cuda":
        return "fp32"
    if precision_arg == "auto":
        return "bf16"
    return precision_arg


def resolve_num_workers(requested: int, device: torch.device) -> int:
    if requested >= 0:
        return requested
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task:
        try:
            available_cpus = max(int(slurm_cpus_per_task), 1)
        except ValueError:
            available_cpus = os.cpu_count() or 1
    else:
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 1
    if device.type == "cuda":
        return min(4, available_cpus)
    return 0


def configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def load_time_meta_for_training(data_dir: str | Path, num_time_steps: int) -> pd.DataFrame:
    time_meta_path = Path(data_dir) / "time_meta.csv"
    if not time_meta_path.exists():
        raise FileNotFoundError(f"找不到 {time_meta_path}，請先執行 build_speed_features.py")
    time_meta = pd.read_csv(time_meta_path)
    if len(time_meta) < num_time_steps:
        raise ValueError(
            f"time_meta.csv 行數 {len(time_meta)} 少於目前資料時間步 {num_time_steps}"
        )
    time_meta = time_meta.iloc[:num_time_steps].copy()
    time_meta["date"] = pd.to_datetime(time_meta["date"], errors="coerce")
    if time_meta["date"].isna().any():
        raise ValueError("time_meta.csv 的 date 欄位有無法解析的值")
    return time_meta


def assign_calendar_split(time_meta: pd.DataFrame) -> pd.Series:
    day = time_meta["date"].dt.day
    return pd.Series(
        np.where(day <= 20, "train", np.where(day <= 24, "val", "test")),
        index=time_meta.index,
        dtype="object",
    )


CALENDAR_SPLIT_STRATEGY = (
    "per_month_day_1_20_train_21_24_val_25_plus_test_full_window_containment"
)
CALENDAR_SPLIT_DESCRIPTION = (
    "每月 1-20 train / 21-24 val / 25+ test，且整個 history+target window 必須完整落在同一 split"
)


def build_monthly_split_indices(
    time_meta: pd.DataFrame,
    hist_len: int,
    pred_horizon: int,
) -> dict[str, list[int]]:
    total = len(time_meta) - hist_len - pred_horizon + 1
    split_labels = assign_calendar_split(time_meta)
    splits = {"train": [], "val": [], "test": []}
    window_len = hist_len + pred_horizon

    for idx in range(max(total, 0)):
        window_labels = split_labels.iloc[idx: idx + window_len].unique()
        if len(window_labels) != 1:
            continue
        splits[str(window_labels[0])].append(idx)

    return splits


def build_window_time_mask(
    num_time_steps: int,
    sample_indices: list[int],
    hist_len: int,
    pred_horizon: int,
) -> np.ndarray:
    mask = np.zeros(num_time_steps, dtype=bool)
    window_len = hist_len + pred_horizon
    for idx in sample_indices:
        start = int(idx)
        end = min(start + window_len, num_time_steps)
        if start < end:
            mask[start:end] = True
    return mask


def init_loss_dict() -> dict[str, float]:
    return {"dc": 0.0, "v": 0.0, "demand": 0.0, "supply": 0.0, "speed": 0.0}


def skipped_loss_dict() -> dict[str, None]:
    return {"dc": None, "v": None, "demand": None, "supply": None, "speed": None}


def skipped_raw_metric_dict() -> dict[str, float | dict[str, float] | None]:
    return {
        "raw_dc": None,
        "demand": None,
        "supply": None,
        "speed": None,
        "report_horizons": None,
    }


def build_task_losses(
    loss_d: torch.Tensor,
    loss_c: torch.Tensor,
    loss_v: torch.Tensor,
    *,
    lam1: float,
    lam2: float,
    lam3: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dc_loss = lam1 * loss_d + lam2 * loss_c
    v_loss = lam3 * loss_v
    return dc_loss, v_loss


def optimized_tasks_for_mode(train_task: str) -> list[str]:
    if train_task == "dc":
        return ["dc"]
    if train_task == "v":
        return ["v"]
    return ["dc", "v"]


def build_raw_dc_metric(raw_metrics: dict[str, dict[str, float]]) -> float:
    return float(raw_metrics["demand"]["rmse"] + raw_metrics["supply"]["rmse"])


DEFAULT_V_PRED_HORIZON = 4
DEFAULT_NON_V_PRED_HORIZON = 3
DEFAULT_V_REPORT_HORIZONS_MINUTES = (15, 30, 60)


def parse_report_horizons_minutes(raw_value: str) -> list[int]:
    value = raw_value.strip()
    if not value:
        return []

    report_minutes: list[int] = []
    seen: set[int] = set()
    for chunk in value.split(","):
        minute = int(chunk.strip())
        if minute <= 0:
            raise ValueError("report horizon minutes must be positive integers.")
        if minute in seen:
            continue
        seen.add(minute)
        report_minutes.append(minute)
    return report_minutes


def infer_time_slot_minutes(time_meta: pd.DataFrame) -> int:
    if time_meta.empty:
        raise ValueError("time_meta is empty; cannot infer slot length.")
    timestamps = (
        time_meta["date"]
        + pd.to_timedelta(time_meta["hour"], unit="h")
        + pd.to_timedelta(time_meta["minute"], unit="m")
    )
    diff_minutes = timestamps.diff().dt.total_seconds().dropna() / 60.0
    positive_diffs = diff_minutes[diff_minutes > 0]
    if positive_diffs.empty:
        raise ValueError("Unable to infer slot length from time_meta timestamps.")
    slot_minutes = int(round(float(positive_diffs.mode().iloc[0])))
    if slot_minutes <= 0:
        raise ValueError(f"Inferred invalid slot length: {slot_minutes} minutes.")
    return slot_minutes


def resolve_report_horizons(
    *,
    time_slot_minutes: int,
    pred_horizon: int,
    requested_minutes: list[int],
    strict: bool = True,
) -> dict[str, int | list[int]]:
    resolved_steps: list[int] = []
    resolved_minutes: list[int] = []
    missing_minutes: list[int] = []

    for minute in requested_minutes:
        if minute % time_slot_minutes != 0:
            raise ValueError(
                f"Requested report horizon {minute} min is not divisible by "
                f"time_slot_minutes={time_slot_minutes}."
            )
        step = minute // time_slot_minutes
        if step < 1 or step > pred_horizon:
            missing_minutes.append(minute)
            continue
        resolved_steps.append(int(step))
        resolved_minutes.append(int(minute))

    if strict and missing_minutes:
        missing = ", ".join(str(v) for v in missing_minutes)
        raise ValueError(
            "Requested report horizons exceed the available prediction horizon: "
            f"missing [{missing}] min for pred_horizon={pred_horizon} "
            f"with time_slot_minutes={time_slot_minutes}."
        )

    return {
        "slot_minutes": int(time_slot_minutes),
        "pred_horizon": int(pred_horizon),
        "requested_minutes": [int(v) for v in requested_minutes],
        "resolved_minutes": resolved_minutes,
        "resolved_steps": resolved_steps,
        "missing_minutes": missing_minutes,
    }


def init_raw_metric_bucket() -> dict[str, float]:
    return {"se": 0.0, "ae": 0.0, "count": 0.0}


def finalize_raw_metric_bucket(bucket: dict[str, float]) -> dict[str, float]:
    count = max(bucket["count"], 1.0)
    mse_value = bucket["se"] / count
    mae_value = bucket["ae"] / count
    return {
        "mse": float(mse_value),
        "rmse": float(np.sqrt(mse_value)),
        "mae": float(mae_value),
    }


def summarize_report_metrics(
    task_metrics: dict[str, float | dict[str, dict[str, float]] | None] | None,
    *,
    metric_name: str = "rmse",
) -> str:
    if task_metrics is None:
        return ""
    report_metrics = task_metrics.get("report") if isinstance(task_metrics, dict) else None
    if not isinstance(report_metrics, dict):
        return ""

    parts: list[str] = []
    for label, values in report_metrics.items():
        if not isinstance(values, dict) or metric_name not in values:
            continue
        parts.append(f"{label}:{values[metric_name]:.3f}")
    return " ".join(parts)


def build_history_record(
    *,
    epoch: int,
    train_losses: dict[str, float],
    val_losses: dict[str, float | None],
    val_raw_metrics: dict[str, float | dict[str, float] | None],
    lr: float,
    elapsed: float,
    train_task: str,
) -> dict:
    record = {
        "epoch": epoch,
        "lr": lr,
        "elapsed": round(elapsed, 1),
    }
    if train_task != "v":
        record.update(
            {
                "train_dc": round(train_losses["dc"], 5),
                "train_demand": round(train_losses["demand"], 5),
                "train_supply": round(train_losses["supply"], 5),
                "val_dc": round(val_losses["dc"], 5) if val_losses["dc"] is not None else None,
                "val_demand": round(val_losses["demand"], 5) if val_losses["demand"] is not None else None,
                "val_supply": round(val_losses["supply"], 5) if val_losses["supply"] is not None else None,
                "val_raw_dc": round(val_raw_metrics["raw_dc"], 5) if val_raw_metrics["raw_dc"] is not None else None,
                "val_raw_demand_rmse": (
                    round(val_raw_metrics["demand"]["rmse"], 5)
                    if val_raw_metrics["demand"] is not None
                    else None
                ),
                "val_raw_supply_rmse": (
                    round(val_raw_metrics["supply"]["rmse"], 5)
                    if val_raw_metrics["supply"] is not None
                    else None
                ),
            }
        )
    if train_task != "dc":
        record.update(
            {
                "train_v": round(train_losses["v"], 5),
                "train_speed": round(train_losses["speed"], 5),
                "val_v": round(val_losses["v"], 5) if val_losses["v"] is not None else None,
                "val_speed": round(val_losses["speed"], 5) if val_losses["speed"] is not None else None,
                "val_raw_speed_rmse": (
                    round(val_raw_metrics["speed"]["rmse"], 5)
                    if val_raw_metrics["speed"] is not None
                    else None
                ),
            }
        )
        if val_raw_metrics["speed"] is not None:
            report_summary = val_raw_metrics["speed"].get("report", {})
            if isinstance(report_summary, dict):
                for label, values in report_summary.items():
                    if isinstance(values, dict) and "rmse" in values:
                        record[f"val_raw_speed_rmse_{label.lower()}"] = round(values["rmse"], 5)
    return record


def format_banner(*, lam1: float, lam2: float, lam3: float, args: argparse.Namespace, val_interval: int) -> str:
    if args.train_task == "v":
        return (
            f"\n開始訓練 | Epochs={args.epochs} | "
            f"v=({lam3}*V) | val_interval={val_interval} | "
            f"monitor={args.monitor_task} | train_task={args.train_task}"
        )
    if args.train_task == "dc":
        return (
            f"\n開始訓練 | Epochs={args.epochs} | "
            f"dc=({lam1}*D + {lam2}*C) | val_interval={val_interval} | "
            f"monitor={args.monitor_task} | train_task={args.train_task}"
        )
    return (
        f"\n開始訓練 | Epochs={args.epochs} | "
        f"dc=({lam1}*D + {lam2}*C) | v=({lam3}*V) | "
        f"val_interval={val_interval} | monitor={args.monitor_task} | "
        f"train_task={args.train_task}"
    )


def format_val_message(
    *,
    val_losses: dict[str, float | None],
    val_raw_metrics: dict[str, float | dict[str, float] | None],
    train_task: str,
) -> str:
    if train_task == "v":
        if val_losses["v"] is None:
            return "Val=skip"
        if val_raw_metrics["speed"] is None:
            return f"ValV={val_losses['v']:.4f}"
        report_summary = summarize_report_metrics(val_raw_metrics["speed"])
        suffix = f" ValRawV={val_raw_metrics['speed']['rmse']:.3f}"
        if report_summary:
            suffix += f" ({report_summary})"
        return f"ValV={val_losses['v']:.4f}{suffix}"
    if val_losses["dc"] is None:
        return "Val=skip"
    if val_raw_metrics["raw_dc"] is not None:
        return (
            f"ValDC={val_losses['dc']:.4f} ValV={val_losses['v']:.4f} "
            f"ValRawDC={val_raw_metrics['raw_dc']:.3f} "
            f"(D:{val_raw_metrics['demand']['rmse']:.3f} C:{val_raw_metrics['supply']['rmse']:.3f})"
        )
    return f"ValDC={val_losses['dc']:.4f} ValV={val_losses['v']:.4f}"


def filter_normalized_losses(losses: dict[str, float | None], train_task: str) -> dict[str, float | None]:
    if train_task == "v":
        return {
            "v": losses["v"],
            "speed": losses["speed"],
        }
    if train_task == "dc":
        return {
            "dc": losses["dc"],
            "demand": losses["demand"],
            "supply": losses["supply"],
        }
    return losses


def _extract_overall_task_metrics(task_metrics: dict[str, float | dict[str, dict[str, float]]]) -> dict[str, float]:
    return {
        "mse": float(task_metrics["mse"]),
        "rmse": float(task_metrics["rmse"]),
        "mae": float(task_metrics["mae"]),
    }


def filter_raw_metrics(
    metrics: dict[str, dict[str, float | dict[str, dict[str, float]]]],
    train_task: str,
) -> dict[str, dict[str, float]]:
    if train_task == "v":
        return {"speed": _extract_overall_task_metrics(metrics["speed"])}
    if train_task == "dc":
        return {
            "demand": _extract_overall_task_metrics(metrics["demand"]),
            "supply": _extract_overall_task_metrics(metrics["supply"]),
        }
    return {
        "demand": _extract_overall_task_metrics(metrics["demand"]),
        "supply": _extract_overall_task_metrics(metrics["supply"]),
        "speed": _extract_overall_task_metrics(metrics["speed"]),
    }


def filter_raw_metrics_per_step(
    metrics: dict[str, dict[str, float | dict[str, dict[str, float]]]],
    train_task: str,
) -> dict[str, dict[str, dict[str, float]]]:
    if train_task == "v":
        return {"speed": dict(metrics["speed"].get("per_step", {}))}
    if train_task == "dc":
        return {
            "demand": dict(metrics["demand"].get("per_step", {})),
            "supply": dict(metrics["supply"].get("per_step", {})),
        }
    return {
        "demand": dict(metrics["demand"].get("per_step", {})),
        "supply": dict(metrics["supply"].get("per_step", {})),
        "speed": dict(metrics["speed"].get("per_step", {})),
    }


def filter_raw_metrics_report(
    metrics: dict[str, dict[str, float | dict[str, dict[str, float]]]],
    train_task: str,
) -> dict[str, dict[str, dict[str, float]]]:
    if train_task == "v":
        return {"speed": dict(metrics["speed"].get("report", {}))}
    if train_task == "dc":
        return {
            "demand": dict(metrics["demand"].get("report", {})),
            "supply": dict(metrics["supply"].get("report", {})),
        }
    return {
        "demand": dict(metrics["demand"].get("report", {})),
        "supply": dict(metrics["supply"].get("report", {})),
        "speed": dict(metrics["speed"].get("report", {})),
    }


def extract_report_horizons(
    metrics: dict[str, object],
) -> dict[str, int | list[int]] | None:
    report_horizons = metrics.get("report_horizons")
    return dict(report_horizons) if isinstance(report_horizons, dict) else None


def extract_temporal_context(node_seq: torch.Tensor) -> torch.Tensor | None:
    if node_seq.shape[-1] <= 2:
        return None
    return node_seq[:, 0, :, 2:]


def forward_for_task(
    model: nn.Module,
    node_seq: torch.Tensor,
    speed_seq: torch.Tensor,
    train_task: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    if train_task == "v":
        v_pred = model.forward_v(speed_seq, extract_temporal_context(node_seq))
        return None, None, v_pred
    d_pred, c_pred, v_pred = model(node_seq, speed_seq)
    return d_pred, c_pred, v_pred


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    train_task: str,
    device: torch.device,
    non_blocking: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    mse: nn.Module,
    lam1: float,
    lam2: float,
    lam3: float,
) -> dict[str, float]:
    losses = init_loss_dict()
    n_batches = 0

    with torch.inference_mode():
        for batch in loader:
            node_seq = batch["node_seq"].to(device, non_blocking=non_blocking)
            speed_seq = batch["speed_seq"].to(device, non_blocking=non_blocking)
            d_tgt = batch["demand_target"].to(device, non_blocking=non_blocking)
            c_tgt = batch["supply_target"].to(device, non_blocking=non_blocking)
            v_tgt = batch["speed_target"].to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                d_pred, c_pred, v_pred = forward_for_task(model, node_seq, speed_seq, train_task)

                loss_v = mse(v_pred, v_tgt)
                if train_task == "v":
                    zero = loss_v.new_zeros(())
                    loss_d = zero
                    loss_c = zero
                else:
                    loss_d = mse(d_pred, d_tgt)
                    loss_c = mse(c_pred, c_tgt)
                loss_dc, loss_task_v = build_task_losses(
                    loss_d,
                    loss_c,
                    loss_v,
                    lam1=lam1,
                    lam2=lam2,
                    lam3=lam3,
                )

            losses["dc"] += loss_dc.item()
            losses["v"] += loss_task_v.item()
            losses["demand"] += loss_d.item()
            losses["supply"] += loss_c.item()
            losses["speed"] += loss_v.item()
            n_batches += 1

    for key in losses:
        losses[key] /= max(n_batches, 1)
    return losses


def evaluate_loader_raw_metrics(
    model: nn.Module,
    loader: DataLoader,
    *,
    train_task: str,
    device: torch.device,
    non_blocking: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    normalization_stats: dict | None,
    time_slot_minutes: int,
    report_horizons: dict[str, int | list[int]],
) -> dict[str, object]:
    pred_horizon = int(report_horizons["pred_horizon"])
    accum = {
        "demand": {
            "overall": init_raw_metric_bucket(),
            "per_step": [init_raw_metric_bucket() for _ in range(pred_horizon)],
        },
        "supply": {
            "overall": init_raw_metric_bucket(),
            "per_step": [init_raw_metric_bucket() for _ in range(pred_horizon)],
        },
        "speed": {
            "overall": init_raw_metric_bucket(),
            "per_step": [init_raw_metric_bucket() for _ in range(pred_horizon)],
        },
    }

    with torch.inference_mode():
        for batch in loader:
            node_seq = batch["node_seq"].to(device, non_blocking=non_blocking)
            speed_seq = batch["speed_seq"].to(device, non_blocking=non_blocking)
            d_tgt = batch["demand_target"].to(device, non_blocking=non_blocking)
            c_tgt = batch["supply_target"].to(device, non_blocking=non_blocking)
            v_tgt = batch["speed_target"].to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                d_pred, c_pred, v_pred = forward_for_task(model, node_seq, speed_seq, train_task)

            v_pred_raw = denormalize_speed_values(
                v_pred.detach().float().cpu().numpy(),
                normalization_stats,
                edge_axis=1,
            )
            v_tgt_raw = denormalize_speed_values(
                v_tgt.detach().float().cpu().numpy(),
                normalization_stats,
                edge_axis=1,
            )

            comparisons = [("speed", v_pred_raw, v_tgt_raw)]
            if train_task != "v":
                d_pred_raw = denormalize_count_values(
                    d_pred.detach().float().cpu().numpy(),
                    normalization_stats,
                    task="demand",
                )
                d_tgt_raw = denormalize_count_values(
                    d_tgt.detach().float().cpu().numpy(),
                    normalization_stats,
                    task="demand",
                )
                c_pred_raw = denormalize_count_values(
                    c_pred.detach().float().cpu().numpy(),
                    normalization_stats,
                    task="supply",
                )
                c_tgt_raw = denormalize_count_values(
                    c_tgt.detach().float().cpu().numpy(),
                    normalization_stats,
                    task="supply",
                )
                comparisons = [
                    ("demand", d_pred_raw, d_tgt_raw),
                    ("supply", c_pred_raw, c_tgt_raw),
                    ("speed", v_pred_raw, v_tgt_raw),
                ]

            for name, pred_raw, tgt_raw in comparisons:
                diff = pred_raw.astype(np.float64) - tgt_raw.astype(np.float64)
                sq = np.square(diff)
                abs_diff = np.abs(diff)
                accum[name]["overall"]["se"] += float(sq.sum())
                accum[name]["overall"]["ae"] += float(abs_diff.sum())
                accum[name]["overall"]["count"] += float(diff.size)

                for step_idx in range(diff.shape[-1]):
                    step_sq = sq[..., step_idx]
                    step_abs = abs_diff[..., step_idx]
                    accum[name]["per_step"][step_idx]["se"] += float(step_sq.sum())
                    accum[name]["per_step"][step_idx]["ae"] += float(step_abs.sum())
                    accum[name]["per_step"][step_idx]["count"] += float(step_sq.size)

    metrics: dict[str, object] = {
        "report_horizons": dict(report_horizons),
    }
    for name, values in accum.items():
        overall = finalize_raw_metric_bucket(values["overall"])
        per_step: dict[str, dict[str, float]] = {}
        for step_idx, bucket in enumerate(values["per_step"], start=1):
            per_step[f"step_{step_idx}"] = {
                "step": int(step_idx),
                "minutes": int(step_idx * time_slot_minutes),
                **finalize_raw_metric_bucket(bucket),
            }

        report_metrics: dict[str, dict[str, float]] = {}
        resolved_steps = report_horizons.get("resolved_steps", [])
        resolved_minutes = report_horizons.get("resolved_minutes", [])
        if isinstance(resolved_steps, list) and isinstance(resolved_minutes, list):
            for minute, step in zip(resolved_minutes, resolved_steps):
                step_key = f"step_{int(step)}"
                step_metrics = per_step[step_key]
                report_metrics[f"{int(minute)}min"] = {
                    "step": int(step),
                    "minutes": int(minute),
                    "mse": float(step_metrics["mse"]),
                    "rmse": float(step_metrics["rmse"]),
                    "mae": float(step_metrics["mae"]),
                }

        metrics[name] = {
            **overall,
            "per_step": per_step,
            "report": report_metrics,
        }
    return metrics


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.train_task == "dc" and args.monitor_task not in {"dc", "raw_dc"}:
        raise SystemExit("--train-task dc only supports --monitor-task dc/raw_dc.")
    if args.train_task == "v" and args.monitor_task != "v":
        raise SystemExit("--train-task v only supports --monitor-task v.")
    if args.pred_horizon < 1:
        raise SystemExit("--pred-horizon must be >= 1.")
    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    precision = resolve_precision(device, args.precision)
    amp_enabled = device.type == "cuda" and precision == "bf16"
    amp_dtype = torch.bfloat16 if amp_enabled else None
    num_workers = resolve_num_workers(args.num_workers, device)
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    # ── 資料準備 ──
    print("載入紐約真實路網與 build_speed_features 特徵 ...")
    data = load_nyc_real_graph_features(
        args.data_dir,
        max_time_steps=args.max_time_steps,
        edge_length_source=args.edge_length_source,
        add_time_features=not args.disable_time_features,
    )

    adj = data["adj"]
    edge_index = data["edge_index"]
    edge_lengths = data["edge_lengths"]
    node_feat = data["node_features"]
    edge_speeds = data["edge_speeds"]
    time_feature_names = data.get("time_feature_names", [])

    N = adj.shape[0]
    nE = edge_index.shape[0]
    t_steps = int(node_feat.shape[0])
    node_feat_dim = int(node_feat.shape[-1])
    print(f"  節點={N}, 邊={nE}, 時間步={t_steps}, node_feat_dim={node_feat_dim}")
    if time_feature_names:
        print(f"  時間特徵: {', '.join(time_feature_names)}")

    time_meta = load_time_meta_for_training(args.data_dir, t_steps)
    time_slot_minutes = infer_time_slot_minutes(time_meta)
    report_horizons = resolve_report_horizons(
        time_slot_minutes=time_slot_minutes,
        pred_horizon=args.pred_horizon,
        requested_minutes=args.report_horizons_minutes,
        strict=bool(args.report_horizons_minutes),
    )
    if report_horizons["resolved_minutes"]:
        print(
            "  report_horizons="
            f"{report_horizons['resolved_minutes']} min "
            f"(steps={report_horizons['resolved_steps']}, slot={time_slot_minutes} min)"
        )
    split_indices = build_monthly_split_indices(time_meta, args.hist_len, args.pred_horizon)
    train_time_mask = build_window_time_mask(
        t_steps,
        split_indices["train"],
        args.hist_len,
        args.pred_horizon,
    )
    normalization_stats = build_normalization_stats(node_feat, edge_speeds, train_time_mask)
    node_feat = normalize_node_features(node_feat, normalization_stats)
    edge_speeds = normalize_speed_features(edge_speeds, normalization_stats, edge_axis=1)
    print("  正規化: D/C=log1p+zscore, V=per-edge zscore (僅用 train split 統計量)")

    # Dataset
    full_ds = SpatioTemporalDataset(
        node_feat, edge_speeds,
        hist_len=args.hist_len,
        pred_horizon=args.pred_horizon,
    )
    train_ds = Subset(full_ds, split_indices["train"])
    val_ds = Subset(full_ds, split_indices["val"])
    test_ds = Subset(full_ds, split_indices["test"])

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print(f"  切分方式: {CALENDAR_SPLIT_DESCRIPTION}")
    print(
        f"  訓練集={len(train_ds)}, 驗證集={len(val_ds)}, 測試集={len(test_ds)}"
    )
    print(
        f"  裝置={device} | precision={precision} | "
        f"num_workers={num_workers} | pin_memory={pin_memory}"
    )

    # ── 模型 ──
    model = STGATPredictor(
        num_nodes=N,
        edge_index=torch.from_numpy(edge_index),
        edge_lengths=torch.from_numpy(edge_lengths),
        adj_matrix=torch.from_numpy(adj),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_st_blocks=args.num_st_blocks,
        num_gtcn_layers=args.num_gtcn_layers,
        kernel_size=args.kernel_size,
        pred_horizon=args.pred_horizon,
        node_feat_dim=node_feat_dim,
        adaptive_topk=args.adaptive_topk,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型參數量: {total_params:,}")

    if args.compile:
        if hasattr(torch, "compile"):
            print(f"  啟用 torch.compile | mode={args.compile_mode}")
            try:
                model = torch.compile(model, mode=args.compile_mode)
            except Exception as exc:
                print(f"  torch.compile 啟用失敗，改用 eager mode: {exc}")
        else:
            print("  目前 torch 版本不支援 torch.compile，將改用 eager mode")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    mse = nn.MSELoss()

    # ── 訓練 ──
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val_losses = {"dc": float("inf"), "v": float("inf")}
    best_monitor_values = {"dc": float("inf"), "v": float("inf"), "raw_dc": float("inf")}
    selected_val_losses = skipped_loss_dict()
    selected_val_raw_metrics = skipped_raw_metric_dict()
    optimize_dc = args.train_task != "v"
    optimize_v = args.train_task != "dc"
    optimized_tasks = optimized_tasks_for_mode(args.train_task)

    lam1, lam2, lam3 = args.lambda1, args.lambda2, args.lambda3
    val_interval = max(args.val_interval, 1)
    print(format_banner(lam1=lam1, lam2=lam2, lam3=lam3, args=args, val_interval=val_interval))
    print("-" * 70)

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        train_losses = init_loss_dict()
        n_batches = 0

        for batch in train_loader:
            node_seq = batch["node_seq"].to(device, non_blocking=non_blocking)       # (B, N, h, C)
            speed_seq = batch["speed_seq"].to(device, non_blocking=non_blocking)     # (B, |E|, h)
            d_tgt = batch["demand_target"].to(device, non_blocking=non_blocking)
            c_tgt = batch["supply_target"].to(device, non_blocking=non_blocking)
            v_tgt = batch["speed_target"].to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                d_pred, c_pred, v_pred = forward_for_task(model, node_seq, speed_seq, args.train_task)
                loss_v = mse(v_pred, v_tgt)
                if args.train_task == "v":
                    zero = loss_v.new_zeros(())
                    loss_d = zero
                    loss_c = zero
                else:
                    loss_d = mse(d_pred, d_tgt)
                    loss_c = mse(c_pred, c_tgt)
                loss_dc, loss_task_v = build_task_losses(
                    loss_d,
                    loss_c,
                    loss_v,
                    lam1=lam1,
                    lam2=lam2,
                    lam3=lam3,
                )

            optimizer.zero_grad(set_to_none=True)
            if optimize_dc and optimize_v:
                loss_dc.backward(retain_graph=True)
                loss_task_v.backward()
            elif optimize_v:
                loss_task_v.backward()
            else:
                loss_dc.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses["dc"] += loss_dc.item()
            train_losses["v"] += loss_task_v.item()
            train_losses["demand"] += loss_d.item()
            train_losses["supply"] += loss_c.item()
            train_losses["speed"] += loss_v.item()
            n_batches += 1

        for k in train_losses:
            train_losses[k] /= max(n_batches, 1)

        # ── val ──
        should_validate = (epoch % val_interval == 0) or (epoch == args.epochs)
        if should_validate:
            model.eval()
            val_losses = evaluate_loader(
                model,
                val_loader,
                train_task=args.train_task,
                device=device,
                non_blocking=non_blocking,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                mse=mse,
                lam1=lam1,
                lam2=lam2,
                lam3=lam3,
            )
            val_raw_metrics = skipped_raw_metric_dict()
            should_compute_raw_metrics = args.monitor_task == "raw_dc" or args.train_task != "dc"
            if should_compute_raw_metrics:
                raw_metrics = evaluate_loader_raw_metrics(
                    model,
                    val_loader,
                    train_task=args.train_task,
                    device=device,
                    non_blocking=non_blocking,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    normalization_stats=normalization_stats,
                    time_slot_minutes=time_slot_minutes,
                    report_horizons=report_horizons,
                )
                val_raw_metrics = {
                    "raw_dc": (
                        build_raw_dc_metric(raw_metrics)
                        if args.train_task != "v"
                        else None
                    ),
                    "demand": raw_metrics["demand"] if args.train_task != "v" else None,
                    "supply": raw_metrics["supply"] if args.train_task != "v" else None,
                    "speed": raw_metrics["speed"] if args.train_task != "dc" else None,
                    "report_horizons": raw_metrics["report_horizons"],
                }
            if args.monitor_task == "raw_dc":
                monitor_value = val_raw_metrics["raw_dc"]
            else:
                monitor_value = val_losses[args.monitor_task]
            scheduler.step(monitor_value)
        else:
            val_losses = skipped_loss_dict()
            val_raw_metrics = skipped_raw_metric_dict()

        # ── 記錄 ──
        elapsed = time.time() - t0
        record = build_history_record(
            epoch=epoch,
            train_losses=train_losses,
            val_losses=val_losses,
            val_raw_metrics=val_raw_metrics,
            lr=optimizer.param_groups[0]["lr"],
            elapsed=elapsed,
            train_task=args.train_task,
        )
        history.append(record)

        val_msg = format_val_message(
            val_losses=val_losses,
            val_raw_metrics=val_raw_metrics,
            train_task=args.train_task,
        )
        if epoch % args.log_interval == 0 or epoch == 1:
            if args.train_task == "v":
                print(
                    f"[Ep {epoch:>4d}]  "
                    f"TrainV={train_losses['v']:.4f} "
                    f"(S={train_losses['speed']:.3f})  "
                    f"{val_msg}  "
                    f"({elapsed:.0f}s)"
                )
            elif args.train_task == "dc":
                print(
                    f"[Ep {epoch:>4d}]  "
                    f"TrainDC={train_losses['dc']:.4f} "
                    f"(D={train_losses['demand']:.3f} C={train_losses['supply']:.3f})  "
                    f"{val_msg}  "
                    f"({elapsed:.0f}s)"
                )
            else:
                print(
                    f"[Ep {epoch:>4d}]  "
                    f"TrainDC={train_losses['dc']:.4f} "
                    f"TrainV={train_losses['v']:.4f} "
                    f"(D={train_losses['demand']:.3f} C={train_losses['supply']:.3f} S={train_losses['speed']:.3f})  "
                    f"{val_msg}  "
                    f"({elapsed:.0f}s)"
                )

        # ── 儲存最佳 ──
        if optimize_dc and val_losses["dc"] is not None and val_losses["dc"] < best_val_losses["dc"]:
            best_val_losses["dc"] = val_losses["dc"]
            torch.save(model.state_dict(), log_dir / "stgat_best_dc.pt")
            if args.monitor_task == "dc":
                torch.save(model.state_dict(), log_dir / "stgat_best.pt")
                best_monitor_values["dc"] = val_losses["dc"]
                selected_val_losses = dict(val_losses)
                selected_val_raw_metrics = dict(val_raw_metrics)
        if optimize_v and val_losses["v"] is not None and val_losses["v"] < best_val_losses["v"]:
            best_val_losses["v"] = val_losses["v"]
            torch.save(model.state_dict(), log_dir / "stgat_best_v.pt")
            if args.monitor_task == "v":
                torch.save(model.state_dict(), log_dir / "stgat_best.pt")
                best_monitor_values["v"] = val_losses["v"]
                selected_val_losses = dict(val_losses)
                selected_val_raw_metrics = dict(val_raw_metrics)
        if val_raw_metrics["raw_dc"] is not None and val_raw_metrics["raw_dc"] < best_monitor_values["raw_dc"]:
            best_monitor_values["raw_dc"] = val_raw_metrics["raw_dc"]
            torch.save(model.state_dict(), log_dir / "stgat_best_raw_dc.pt")
            if args.monitor_task == "raw_dc":
                torch.save(model.state_dict(), log_dir / "stgat_best.pt")
                selected_val_losses = dict(val_losses)
                selected_val_raw_metrics = dict(val_raw_metrics)

    # ── 結束 ──
    torch.save(model.state_dict(), log_dir / "stgat_final.pt")
    best_monitor_value = best_monitor_values[args.monitor_task]
    if optimize_v:
        print(f"Best Val V = {best_val_losses['v']:.5f}")
    else:
        print("Best Val V = skipped (train_task=dc)")
    if optimize_dc and best_val_losses["dc"] < float("inf"):
        print(f"Best Val DC = {best_val_losses['dc']:.5f}")
    if args.train_task != "v" and best_monitor_values["raw_dc"] < float("inf"):
        print(f"Best Val Raw DC = {best_monitor_values['raw_dc']:.5f}")
    print(f"Default checkpoint is {args.monitor_task}-best: {log_dir / 'stgat_best.pt'}")
    print(f"\n訓練完成 | Best Monitor Metric = {best_monitor_value:.5f}")
    print(f"模型已儲存至 {log_dir / 'stgat_best.pt'}")

    best_state = torch.load(
        log_dir / "stgat_best.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state)
    model.eval()
    test_losses = evaluate_loader(
        model,
        test_loader,
        train_task=args.train_task,
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        mse=mse,
        lam1=lam1,
        lam2=lam2,
        lam3=lam3,
    )
    test_raw_metrics = evaluate_loader_raw_metrics(
        model,
        test_loader,
        train_task=args.train_task,
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        normalization_stats=normalization_stats,
        time_slot_minutes=time_slot_minutes,
        report_horizons=report_horizons,
    )
    if args.train_task == "v":
        print(
            "最終 Test="
            f"V={test_losses['v']:.4f} "
            f"(S={test_losses['speed']:.3f})"
        )
        print(
            "原始尺度 Test RMSE="
            f"V:{test_raw_metrics['speed']['rmse']:.3f}"
        )
        test_report_summary = summarize_report_metrics(test_raw_metrics["speed"])
        if test_report_summary:
            print(f"Paper-aligned speed RMSE={test_report_summary}")
    elif args.train_task == "dc":
        print(
            "最終 Test="
            f"DC={test_losses['dc']:.4f} "
            f"(D={test_losses['demand']:.3f} C={test_losses['supply']:.3f})"
        )
        print(
            "原始尺度 Test RMSE="
            f"D:{test_raw_metrics['demand']['rmse']:.3f} "
            f"C:{test_raw_metrics['supply']['rmse']:.3f}"
        )
    else:
        print(
            "最終 Test="
            f"DC={test_losses['dc']:.4f} "
            f"V={test_losses['v']:.4f} "
            f"(D={test_losses['demand']:.3f} C={test_losses['supply']:.3f} S={test_losses['speed']:.3f})"
        )
        print(
            "原始尺度 Test RMSE="
            f"D:{test_raw_metrics['demand']['rmse']:.3f} "
            f"C:{test_raw_metrics['supply']['rmse']:.3f} "
            f"V:{test_raw_metrics['speed']['rmse']:.3f}"
        )
        test_report_summary = summarize_report_metrics(test_raw_metrics["speed"])
        if test_report_summary:
            print(f"Paper-aligned speed RMSE={test_report_summary}")

    # 儲存訓練資料元資訊（供 pipeline 使用）
    meta = {
        "num_nodes": N,
        "num_edges": nE,
        "edge_index": edge_index.tolist(),
        "edge_lengths": edge_lengths.tolist(),
        "adj": adj.tolist(),
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_st_blocks": args.num_st_blocks,
        "num_gtcn_layers": args.num_gtcn_layers,
        "kernel_size": args.kernel_size,
        "adaptive_topk": args.adaptive_topk,
        "pred_horizon": args.pred_horizon,
        "hist_len": args.hist_len,
        "time_slot_minutes": time_slot_minutes,
        "report_horizons": report_horizons,
        "node_feat_dim": node_feat_dim,
        "use_time_features": bool(time_feature_names),
        "time_feature_names": time_feature_names,
        "loss_space": "normalized",
        "loss_tasks": (
            {"v": {"formula": f"{lam3} * speed"}}
            if args.train_task == "v"
            else {
                "dc": {"formula": f"{lam1} * demand + {lam2} * supply"},
                "v": {"formula": f"{lam3} * speed"},
            }
        ),
        "train_task": args.train_task,
        "optimized_tasks": optimized_tasks,
        "checkpoint_selection": {
            "default_checkpoint": "stgat_best.pt",
            "default_monitor": args.monitor_task,
            "available_checkpoints": (
                {
                    "v": "stgat_best_v.pt",
                }
                if args.train_task == "v"
                else (
                {
                    "dc": "stgat_best_dc.pt",
                    "v": "stgat_best_v.pt",
                    "raw_dc": "stgat_best_raw_dc.pt",
                }
                if optimize_v
                else {
                    "dc": "stgat_best_dc.pt",
                    "raw_dc": "stgat_best_raw_dc.pt",
                }
                )
            ),
        },
        "normalization": serialize_normalization_stats(normalization_stats),
        "split_strategy": CALENDAR_SPLIT_STRATEGY,
        "split_description": CALENDAR_SPLIT_DESCRIPTION,
        "normalization_time_steps": int(train_time_mask.sum()),
        "split_counts": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "data_source": "nyc_real",
        "data_dir": args.data_dir,
    }
    with open(log_dir / "stgat_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(log_dir / "predictor_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"訓練日誌已儲存至 {log_dir / 'predictor_log.json'}")

    with open(log_dir / "predictor_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss_space": "normalized",
                "normalized_loss": filter_normalized_losses(test_losses, args.train_task),
                "val_normalized_loss": filter_normalized_losses(selected_val_losses, args.train_task),
                "raw_metrics": filter_raw_metrics(test_raw_metrics, args.train_task),
                "raw_metrics_per_step": filter_raw_metrics_per_step(test_raw_metrics, args.train_task),
                "raw_metrics_report": filter_raw_metrics_report(test_raw_metrics, args.train_task),
                "report_horizons": extract_report_horizons(test_raw_metrics),
                "val_raw_metrics": (
                    filter_raw_metrics(selected_val_raw_metrics, args.train_task)
                    if selected_val_raw_metrics["speed"] is not None or args.train_task == "dc"
                    else None
                ),
                "val_raw_metrics_per_step": (
                    filter_raw_metrics_per_step(selected_val_raw_metrics, args.train_task)
                    if selected_val_raw_metrics["speed"] is not None or args.train_task == "dc"
                    else None
                ),
                "val_raw_metrics_report": (
                    filter_raw_metrics_report(selected_val_raw_metrics, args.train_task)
                    if selected_val_raw_metrics["speed"] is not None or args.train_task == "dc"
                    else None
                ),
                "selected_checkpoint": "stgat_best.pt",
                "selected_checkpoint_task": args.monitor_task,
                "selected_checkpoint_metric": best_monitor_values[args.monitor_task],
                "train_task": args.train_task,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"測試指標已儲存至 {log_dir / 'predictor_test_metrics.json'}")


# ── CLI ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train STGAT Predictor")

    # 資料
    p.add_argument("--data-dir", type=str, default="data", help="真實資料目錄（adjacency、時序特徵）")
    p.add_argument(
        "--max-time-steps",
        type=int,
        default=0,
        help="截斷時間步（0=全部；除錯或減輕顯存時可設小於完整 T）",
    )
    p.add_argument(
        "--edge-length-source",
        type=str,
        default="osrm",
        choices=["osrm", "centroid"],
        help="邊長來源：osrm 優先讀 edge_lengths_osrm.npy，否則 centroid",
    )
    p.add_argument(
        "--disable-time-features",
        action="store_true",
        help="停用 month / weekday / slot 時間特徵，回到舊版 demand+supply 節點輸入",
    )
    p.add_argument("--hist-len", type=int, default=12, help="歷史窗口 h")
    p.add_argument(
        "--pred-horizon",
        type=int,
        default=None,
        help="Predicted steps p; V-only defaults to 4 for 15/30/60 reporting, other modes default to 3.",
    )
    p.add_argument(
        "--report-horizons-minutes",
        type=str,
        default=None,
        help="Comma-separated report horizons in minutes (for example 15,30,60). Empty disables report extraction.",
    )

    # 模型
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-st-blocks", type=int, default=2)
    p.add_argument(
        "--adaptive-topk",
        type=int,
        default=16,
        help="Top-k neighbors kept by the learned adaptive adjacency; 0 keeps the dense graph.",
    )
    p.add_argument("--num-gtcn-layers", type=int, default=2)
    p.add_argument("--kernel-size", type=int, default=3)

    # 訓練
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda1", type=float, default=1.0, help="需求損失權重")
    p.add_argument("--lambda2", type=float, default=1.0, help="空車損失權重")
    p.add_argument("--lambda3", type=float, default=1.0, help="速度損失權重")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp32"],
        help="CUDA 上預設 auto->bf16；CPU 會自動退回 fp32",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers；-1 代表自動（CUDA 預設最多 8，CPU 預設 0）",
    )
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument(
        "--monitor-task",
        type=str,
        default="raw_dc",
        choices=["dc", "v", "raw_dc"],
        help="Validation metric used for the default scheduler/checkpoint (raw_dc = demand_rmse + supply_rmse).",
    )
    p.add_argument(
        "--train-task",
        type=str,
        default="joint",
        choices=["joint", "dc", "v"],
        help="Optimization mode: update both task heads, only the DC objective, or only the V objective.",
    )
    p.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="每隔多少個 epoch 跑一次 validation；最後一個 epoch 會強制驗證",
    )

    p.add_argument("--compile", action="store_true", help="啟用 torch.compile 加速")
    p.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile 的 mode",
    )

    args = p.parse_args()
    if args.pred_horizon is None:
        args.pred_horizon = (
            DEFAULT_V_PRED_HORIZON
            if args.train_task == "v"
            else DEFAULT_NON_V_PRED_HORIZON
        )

    if args.report_horizons_minutes is None:
        args.report_horizons_minutes = (
            ",".join(str(v) for v in DEFAULT_V_REPORT_HORIZONS_MINUTES)
            if args.train_task == "v"
            else ""
        )
    args.report_horizons_minutes = parse_report_horizons_minutes(args.report_horizons_minutes)
    return args


if __name__ == "__main__":
    train(parse_args())
