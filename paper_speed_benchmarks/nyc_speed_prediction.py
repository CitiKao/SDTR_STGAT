from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import load_nyc_real_graph_features  # noqa: E402
from external_speed_benchmarks.sensor_dataset_utils import (  # noqa: E402
    apply_valid_speed_clip,
    build_cyclical_time_features,
    compute_train_quantile_clip_bounds,
)
from predictor_normalization import (  # noqa: E402
    build_normalization_stats,
    denormalize_speed_values,
    normalize_node_features,
    normalize_speed_features,
    serialize_normalization_stats,
)
from stgat_model import STGATPredictor  # noqa: E402
from train_predictor import (  # noqa: E402
    CALENDAR_SPLIT_DESCRIPTION,
    build_monthly_split_indices,
    build_window_time_mask,
    filter_split_indices_by_time_mask,
    infer_time_slot_minutes,
    load_observed_time_mask,
    load_time_meta_for_training,
    resolve_report_horizons,
)


MODEL_CHOICES = (
    "persistence",
    "temporal_mlp",
    "line_graph_gru",
    "stgat_edge",
    "astgnn_ltd",
    "icst_dnet",
    "3s_tbln",
    "mvstgcn",
    "nexusqn",
    "sgsl_gat_nlstm",
)


@dataclass
class DatasetBundle:
    dataset_name: str
    dataset_format: str
    adj: np.ndarray
    edge_index: np.ndarray
    edge_lengths: np.ndarray
    speed_adjacency: np.ndarray
    node_features: np.ndarray
    edge_speeds: np.ndarray
    speed_valid_mask: np.ndarray | None
    time_meta: pd.DataFrame
    split_indices: dict[str, list[int]]
    normalization_stats: dict[str, Any]
    time_slot_minutes: int
    report_horizons: dict[str, Any]
    time_feature_names: list[str]
    split_description: str
    preprocessing_summary: dict[str, Any]


@dataclass
class MetricBucket:
    se: float = 0.0
    ae: float = 0.0
    ape: float = 0.0
    count: float = 0.0
    mape_count: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def resolve_precision(raw: str, device: torch.device) -> tuple[bool, torch.dtype | None, str]:
    if device.type != "cuda":
        return False, None, "fp32"
    if raw == "auto":
        raw = "bf16"
    if raw == "bf16":
        return True, torch.bfloat16, "bf16"
    return False, None, "fp32"


def parse_minutes(raw: str) -> list[int]:
    if not raw:
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def make_default_log_dir(model_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / "paper_speed_benchmarks" / f"nyc_{model_name}_{stamp}"


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    whole = int(round(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_horizon_metrics(report: dict[str, dict[str, float]]) -> str:
    parts: list[str] = []
    for label in ("15min", "30min", "60min"):
        values = report.get(label)
        if not values:
            continue
        parts.append(
            f"{label}[MSE={values['mse']:.4f} RMSE={values['rmse']:.4f} MAE={values['mae']:.4f}]"
        )
    return " ".join(parts)


def normalize_square_adjacency(
    adjacency: np.ndarray,
    *,
    add_self_loops: bool,
) -> np.ndarray:
    weights = np.asarray(adjacency, dtype=np.float32).copy()
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError(f"speed adjacency must be square, got {weights.shape}")
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights[weights < 0.0] = 0.0
    if add_self_loops:
        diag = np.arange(weights.shape[0])
        weights[diag, diag] = np.maximum(weights[diag, diag], 1.0)
    denom = weights.sum(axis=1, keepdims=True)
    denom[denom <= 0.0] = 1.0
    return (weights / denom).astype(np.float32)


def build_line_graph_adjacency(edge_index: np.ndarray) -> np.ndarray:
    """Dense row-normalized directed line-graph adjacency for edge predictors."""
    edge_index = np.asarray(edge_index, dtype=np.int64)
    num_edges = int(edge_index.shape[0])
    by_src: dict[int, list[int]] = {}
    for edge_id, (src, _dst) in enumerate(edge_index):
        by_src.setdefault(int(src), []).append(int(edge_id))

    adj = np.eye(num_edges, dtype=np.float32)
    for prev_edge, (_src, dst) in enumerate(edge_index):
        for next_edge in by_src.get(int(dst), []):
            adj[int(next_edge), int(prev_edge)] = 1.0

    denom = adj.sum(axis=1, keepdims=True)
    denom[denom <= 0.0] = 1.0
    return (adj / denom).astype(np.float32)


def resolve_speed_adjacency(
    edge_index: np.ndarray | None,
    speed_adjacency: np.ndarray | None,
) -> np.ndarray:
    if speed_adjacency is not None:
        return normalize_square_adjacency(speed_adjacency, add_self_loops=False)
    if edge_index is None:
        raise ValueError("edge_index is required when speed_adjacency is not supplied")
    return build_line_graph_adjacency(edge_index)


def build_masked_normalization_stats(
    node_features: np.ndarray,
    speed_values: np.ndarray,
    train_time_mask: np.ndarray,
    speed_valid_mask: np.ndarray | None,
) -> dict[str, Any]:
    stats = build_normalization_stats(node_features, speed_values, train_time_mask)
    if speed_valid_mask is None:
        return stats

    values = np.asarray(speed_values, dtype=np.float32)
    valid = np.asarray(speed_valid_mask, dtype=bool)
    train_mask = np.asarray(train_time_mask, dtype=bool)
    if valid.shape != values.shape:
        raise ValueError(f"speed_valid_mask {valid.shape} does not match speed_values {values.shape}")

    train_valid = valid & train_mask[:, None]
    global_values = values[train_valid]
    if global_values.size == 0:
        global_values = values[train_mask].reshape(-1)
    global_mean = float(global_values.mean()) if global_values.size else 0.0
    global_std = float(global_values.std()) if global_values.size else 1.0
    if global_std < 1e-6:
        global_std = 1.0

    mean = np.full(values.shape[1], global_mean, dtype=np.float32)
    std = np.full(values.shape[1], global_std, dtype=np.float32)
    for item_idx in range(values.shape[1]):
        item_mask = train_valid[:, item_idx]
        if not np.any(item_mask):
            continue
        observed = values[item_mask, item_idx]
        mean[item_idx] = float(observed.mean())
        item_std = float(observed.std())
        std[item_idx] = item_std if item_std >= 1e-6 else 1.0
    stats["speed"] = {
        "transform": "per_speed_item_zscore_masked_valid_train",
        "mean": mean,
        "std": std,
    }
    return stats


def normalize_speed_values(
    speed_values: np.ndarray,
    normalization_stats: dict[str, Any],
    speed_valid_mask: np.ndarray | None,
) -> np.ndarray:
    if speed_valid_mask is None:
        return normalize_speed_features(speed_values, normalization_stats, edge_axis=1)
    arr = np.asarray(speed_values, dtype=np.float32).copy()
    valid = np.asarray(speed_valid_mask, dtype=bool)
    mean = np.asarray(normalization_stats["speed"]["mean"], dtype=np.float32)
    std = np.asarray(normalization_stats["speed"]["std"], dtype=np.float32)
    arr = (arr - mean.reshape(1, -1)) / std.reshape(1, -1)
    arr[~valid] = 0.0
    return arr.astype(np.float32)


def filter_split_indices_by_target_mask(
    splits: dict[str, list[int]],
    speed_valid_mask: np.ndarray | None,
    hist_len: int,
    pred_horizon: int,
) -> dict[str, list[int]]:
    if speed_valid_mask is None:
        return {name: list(indices) for name, indices in splits.items()}
    valid = np.asarray(speed_valid_mask, dtype=bool)
    filtered: dict[str, list[int]] = {}
    for name, indices in splits.items():
        kept: list[int] = []
        for raw_idx in indices:
            idx = int(raw_idx)
            target_start = idx + int(hist_len)
            target_end = target_start + int(pred_horizon)
            if target_end <= valid.shape[0] and bool(valid[target_start:target_end].any()):
                kept.append(idx)
        filtered[name] = kept
    return filtered


def clean_sensor_speed_values(
    speed_values: np.ndarray,
    speed_valid_mask: np.ndarray,
    train_time_mask: np.ndarray,
    *,
    mode: str,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode == "none":
        return np.asarray(speed_values, dtype=np.float32).copy(), {
            "enabled": False,
            "method": "none",
            "cleaned_points": 0,
            "cleaned_ratio": 0.0,
        }
    if mode != "train_quantile_clip":
        raise ValueError("sensor outlier cleaning must be one of: none, train_quantile_clip")
    bounds = compute_train_quantile_clip_bounds(
        speed_values,
        speed_valid_mask,
        train_time_mask=train_time_mask,
        lower_quantile=float(lower_quantile),
        upper_quantile=float(upper_quantile),
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
            "method": "train_quantile_clip",
            "fit_scope": "train_history_and_target_windows",
            "params": {
                "lower_quantile": float(lower_quantile),
                "upper_quantile": float(upper_quantile),
            },
            "cleaned_points": int(summary["num_flagged"]),
            "cleaned_ratio": float(summary["flagged_ratio_valid"]),
            "train_valid_counts_min": int(np.min(bounds["valid_counts"])) if bounds["valid_counts"].size else 0,
            "train_valid_counts_max": int(np.max(bounds["valid_counts"])) if bounds["valid_counts"].size else 0,
        }
    )
    return np.asarray(clipped["cleaned_speed_values"], dtype=np.float32), summary


class BenchmarkSpatioTemporalDataset(Dataset):
    def __init__(
        self,
        node_features: np.ndarray,
        speed_values: np.ndarray,
        *,
        hist_len: int,
        pred_horizon: int,
        speed_valid_mask: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.node_feat = np.asarray(node_features, dtype=np.float32)
        self.speed_values = np.asarray(speed_values, dtype=np.float32)
        self.speed_valid_mask = None if speed_valid_mask is None else np.asarray(speed_valid_mask, dtype=bool)
        self.h = int(hist_len)
        self.p = int(pred_horizon)
        self.total = int(self.node_feat.shape[0] - self.h - self.p + 1)
        if self.speed_values.shape[0] != self.node_feat.shape[0]:
            raise ValueError("node_features and speed_values must share the same time axis")
        if self.speed_valid_mask is not None and self.speed_valid_mask.shape != self.speed_values.shape:
            raise ValueError("speed_valid_mask must match speed_values shape")

    def __len__(self) -> int:
        return max(self.total, 0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t = int(idx) + self.h
        node_seq = self.node_feat[idx:t]
        speed_seq = self.speed_values[idx:t]
        speed_target = self.speed_values[t:t + self.p].T
        item = {
            "node_seq": torch.from_numpy(node_seq.transpose(1, 0, 2)),
            "speed_seq": torch.from_numpy(speed_seq.T),
            "speed_target": torch.from_numpy(speed_target),
        }
        if self.speed_valid_mask is not None:
            item["speed_history_mask"] = torch.from_numpy(self.speed_valid_mask[idx:t].T)
            item["speed_target_mask"] = torch.from_numpy(self.speed_valid_mask[t:t + self.p].T)
        return item


def _normalize_rows(values: torch.Tensor) -> torch.Tensor:
    denom = values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return values / denom


def _edge_sequence_input(
    node_seq: torch.Tensor,
    speed_seq: torch.Tensor,
    time_feat_dim: int,
) -> torch.Tensor:
    speed_input = speed_seq.unsqueeze(-1)
    if time_feat_dim > 0 and node_seq.shape[-1] > 2:
        time_feat = node_seq[:, 0, :, 2:]
        time_feat = time_feat.unsqueeze(1).expand(-1, speed_seq.shape[1], -1, -1)
        return torch.cat([speed_input, time_feat], dim=-1)
    return speed_input


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class GraphMixLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neigh = torch.einsum("ij,bjd->bid", adjacency.to(h.dtype), h)
        candidate = torch.tanh(self.self_proj(h) + self.neighbor_proj(neigh))
        gate = torch.sigmoid(self.gate(torch.cat([h, neigh], dim=-1)))
        out = gate * candidate + (1.0 - gate) * h
        return self.norm(h + self.dropout(out))


class AdaptiveGraphMixLayer(nn.Module):
    def __init__(
        self,
        *,
        num_edges: int,
        hidden_dim: int,
        rank: int,
        topk: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.src = nn.Parameter(torch.randn(num_edges, rank) * 0.02)
        self.dst = nn.Parameter(torch.randn(num_edges, rank) * 0.02)
        self.topk = int(topk)
        self.mix = GraphMixLayer(hidden_dim, dropout=dropout)

    def adjacency(self) -> torch.Tensor:
        scores = F.relu(self.src @ self.dst.T)
        num_edges = scores.shape[0]
        if 0 < self.topk < num_edges:
            _, idx = scores.topk(self.topk, dim=1)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            scores = scores.masked_fill(~mask, 0.0)
        diag = torch.arange(num_edges, device=scores.device)
        scores[diag, diag] = scores[diag, diag] + 1.0
        return _normalize_rows(scores)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mix(h, self.adjacency())


class CausalGatedTemporalConv(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        padding = (int(kernel_size) - 1) * int(dilation)
        self.filter_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.gate_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm = nn.GroupNorm(1, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        filt = torch.tanh(self.filter_conv(x))
        gate = torch.sigmoid(self.gate_conv(x))
        out = filt * gate
        out = out[..., -residual.shape[-1]:]
        return self.norm(out + residual)


class PersistenceSpeedModel(nn.Module):
    def __init__(self, pred_horizon: int) -> None:
        super().__init__()
        self.pred_horizon = int(pred_horizon)

    def forward(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        del node_seq
        return speed_seq[:, :, -1:].expand(-1, -1, self.pred_horizon)


class TemporalMLPSpeedModel(nn.Module):
    def __init__(
        self,
        *,
        hist_len: int,
        pred_horizon: int,
        time_feat_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = int(hist_len) * (1 + int(time_feat_dim))
        layers: list[nn.Module] = []
        width = int(hidden_dim)
        depth = max(int(num_layers), 1)
        for layer_idx in range(depth):
            layers.append(nn.Linear(input_dim if layer_idx == 0 else width, width))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(width, int(pred_horizon)))
        self.net = nn.Sequential(*layers)
        self.time_feat_dim = int(time_feat_dim)

    def forward(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        batch_size, num_edges, hist_len = speed_seq.shape
        speed_input = speed_seq.unsqueeze(-1)
        if self.time_feat_dim > 0 and node_seq.shape[-1] > 2:
            time_feat = node_seq[:, 0, :, 2:]
            time_feat = time_feat.unsqueeze(1).expand(-1, num_edges, -1, -1)
            model_input = torch.cat([speed_input, time_feat], dim=-1)
        else:
            model_input = speed_input
        flat = model_input.reshape(batch_size * num_edges, hist_len * model_input.shape[-1])
        pred = self.net(flat)
        return pred.reshape(batch_size, num_edges, -1)


class LineGraphGRUSpeedModel(nn.Module):
    def __init__(
        self,
        *,
        edge_index: np.ndarray | None,
        speed_adjacency: np.ndarray | None,
        hist_len: int,
        pred_horizon: int,
        time_feat_dim: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        del hist_len
        self.time_feat_dim = int(time_feat_dim)
        self.gru = nn.GRU(
            input_size=1 + self.time_feat_dim,
            hidden_size=int(hidden_dim),
            batch_first=True,
        )
        self.self_proj = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.neighbor_proj = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.graph_layers = max(int(graph_layers), 1)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(int(hidden_dim), int(pred_horizon))
        self.register_buffer(
            "line_adj",
            torch.from_numpy(resolve_speed_adjacency(edge_index, speed_adjacency)),
        )

    def forward(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        batch_size, num_edges, hist_len = speed_seq.shape
        speed_input = speed_seq.unsqueeze(-1)
        if self.time_feat_dim > 0 and node_seq.shape[-1] > 2:
            time_feat = node_seq[:, 0, :, 2:]
            time_feat = time_feat.unsqueeze(1).expand(-1, num_edges, -1, -1)
            model_input = torch.cat([speed_input, time_feat], dim=-1)
        else:
            model_input = speed_input
        seq = model_input.reshape(batch_size * num_edges, hist_len, model_input.shape[-1])
        _out, hidden = self.gru(seq)
        h = hidden[-1].reshape(batch_size, num_edges, -1)
        for _ in range(self.graph_layers):
            h_neigh = torch.einsum("ij,bjd->bid", self.line_adj.to(h.dtype), h)
            h = F.relu(self.self_proj(h) + self.neighbor_proj(h_neigh))
            h = self.dropout(h)
        return self.head(h)


class ASTGNNLongTermSpeedModel(nn.Module):
    """Paper-inspired long-term attention STGNN adapter for edge speeds."""

    def __init__(
        self,
        *,
        edge_index: np.ndarray | None,
        speed_adjacency: np.ndarray | None,
        hist_len: int,
        pred_horizon: int,
        time_feat_dim: int,
        hidden_dim: int,
        num_heads: int,
        graph_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.time_feat_dim = int(time_feat_dim)
        self.input_proj = nn.Linear(1 + self.time_feat_dim, hidden_dim)
        self.temporal_blocks = nn.ModuleList(
            [
                CausalGatedTemporalConv(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=2 ** layer_idx,
                )
                for layer_idx in range(max(int(graph_layers), 1))
            ]
        )
        self.attn_norm = RMSNorm(hidden_dim)
        self.long_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=max(int(num_heads), 1),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.graph_mix = GraphMixLayer(hidden_dim, dropout=dropout)
        self.head = nn.Linear(hidden_dim, int(pred_horizon))
        self.register_buffer(
            "line_adj",
            torch.from_numpy(resolve_speed_adjacency(edge_index, speed_adjacency)),
        )
        self.hist_len = int(hist_len)

    def forward(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        batch_size, num_edges, hist_len = speed_seq.shape
        x = _edge_sequence_input(node_seq, speed_seq, self.time_feat_dim)
        h = self.input_proj(x).reshape(batch_size * num_edges, hist_len, -1)
        conv = h.transpose(1, 2)
        for block in self.temporal_blocks:
            conv = block(conv)
        h = conv.transpose(1, 2)
        attn_in = self.attn_norm(h)
        attn_out, _ = self.long_attn(attn_in, attn_in, attn_in, need_weights=False)
        h = h + attn_out
        h = h + self.ffn(h)
        h_last = h[:, -1].reshape(batch_size, num_edges, -1)
        h_graph = self.graph_mix(h_last, self.line_adj)
        return self.head(h_graph)


class ICSTDNetSpeedModel(nn.Module):
    """Paper-inspired causal spatio-temporal diffusion adapter."""

    def __init__(
        self,
        *,
        edge_index: np.ndarray | None,
        speed_adjacency: np.ndarray | None,
        hist_len: int,
        pred_horizon: int,
        time_feat_dim: int,
        hidden_dim: int,
        adaptive_topk: int,
        graph_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        del hist_len
        resolved_adj = resolve_speed_adjacency(edge_index, speed_adjacency)
        num_speed_items = int(resolved_adj.shape[0])
        self.time_feat_dim = int(time_feat_dim)
        self.encoder = nn.GRU(1 + self.time_feat_dim, hidden_dim, batch_first=True)
        self.fixed_diffusion = nn.ModuleList(
            [GraphMixLayer(hidden_dim, dropout=dropout) for _ in range(max(int(graph_layers), 1))]
        )
        self.causal_graph = AdaptiveGraphMixLayer(
            num_edges=num_speed_items,
            hidden_dim=hidden_dim,
            rank=max(hidden_dim // 2, 8),
            topk=int(adaptive_topk),
            dropout=dropout,
        )
        self.pattern_gate = nn.Sequential(
            nn.Linear(max(self.time_feat_dim, 1), hidden_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Linear(hidden_dim, int(pred_horizon))
        self.register_buffer(
            "line_adj",
            torch.from_numpy(resolved_adj),
        )

    def forward(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        batch_size, num_edges, hist_len = speed_seq.shape
        x = _edge_sequence_input(node_seq, speed_seq, self.time_feat_dim)
        seq = x.reshape(batch_size * num_edges, hist_len, -1)
        _out, hidden = self.encoder(seq)
        h = hidden[-1].reshape(batch_size, num_edges, -1)
        fixed = h
        for layer in self.fixed_diffusion:
            fixed = layer(fixed, self.line_adj)
        causal = self.causal_graph(h)
        fused = fixed + causal
        if self.time_feat_dim > 0 and node_seq.shape[-1] > 2:
            pattern = node_seq[:, 0, -1, 2:]
        else:
            pattern = speed_seq.new_zeros(batch_size, 1)
        gate = self.pattern_gate(pattern).unsqueeze(1)
        return self.head(fused * gate + h * (1.0 - gate))


class ThreeSTBLNSpeedModel(nn.Module):
    """Paper-inspired self-supervised spatio-temporal bilateral learner."""

    def __init__(
        self,
        *,
        edge_index: np.ndarray | None,
        speed_adjacency: np.ndarray | None,
        hist_len: int,
        pred_horizon: int,
        time_feat_dim: int,
        hidden_dim: int,
        graph_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.time_feat_dim = int(time_feat_dim)
        self.hist_len = int(hist_len)
        half_hidden = max(int(hidden_dim) // 2, 4)
        self.bilateral_gru = nn.GRU(
            1 + self.time_feat_dim,
            half_hidden,
            batch_first=True,
            bidirectional=True,
        )
        encoded_dim = half_hidden * 2
        self.graph_layers = nn.ModuleList(
            [GraphMixLayer(encoded_dim, dropout=dropout) for _ in range(max(int(graph_layers), 1))]
        )
        self.forecast_head = nn.Linear(encoded_dim, int(pred_horizon))
        self.reconstruct_head = nn.Linear(encoded_dim, int(hist_len))
        self.register_buffer(
            "line_adj",
            torch.from_numpy(resolve_speed_adjacency(edge_index, speed_adjacency)),
        )
        self._last_reconstruction: torch.Tensor | None = None

    def forward(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        batch_size, num_edges, hist_len = speed_seq.shape
        x = _edge_sequence_input(node_seq, speed_seq, self.time_feat_dim)
        seq = x.reshape(batch_size * num_edges, hist_len, -1)
        out, hidden = self.bilateral_gru(seq)
        forward_h = hidden[-2]
        backward_h = hidden[-1]
        h = torch.cat([forward_h, backward_h], dim=-1).reshape(batch_size, num_edges, -1)
        for layer in self.graph_layers:
            h = layer(h, self.line_adj)
        self._last_reconstruction = self.reconstruct_head(h)
        return self.forecast_head(h)

    def auxiliary_loss(self, speed_seq: torch.Tensor) -> torch.Tensor:
        if self._last_reconstruction is None:
            return speed_seq.new_zeros(())
        return F.mse_loss(self._last_reconstruction, torch.flip(speed_seq, dims=[-1]))


class MultiViewSTGCNSpeedModel(nn.Module):
    """Paper-inspired multi-view spatial-temporal graph convolution adapter."""

    def __init__(
        self,
        *,
        edge_index: np.ndarray | None,
        speed_adjacency: np.ndarray | None,
        hist_len: int,
        pred_horizon: int,
        time_feat_dim: int,
        hidden_dim: int,
        adaptive_topk: int,
        dropout: float,
    ) -> None:
        super().__init__()
        del hist_len
        line_adj = resolve_speed_adjacency(edge_index, speed_adjacency)
        num_speed_items = int(line_adj.shape[0])
        self.time_feat_dim = int(time_feat_dim)
        self.encoder = nn.GRU(1 + self.time_feat_dim, hidden_dim, batch_first=True)
        self.fixed_view = GraphMixLayer(hidden_dim, dropout=dropout)
        self.reverse_view = GraphMixLayer(hidden_dim, dropout=dropout)
        self.adaptive_view = AdaptiveGraphMixLayer(
            num_edges=num_speed_items,
            hidden_dim=hidden_dim,
            rank=max(hidden_dim // 2, 8),
            topk=int(adaptive_topk),
            dropout=dropout,
        )
        self.view_score = nn.Linear(hidden_dim, 1)
        self.head = nn.Linear(hidden_dim, int(pred_horizon))
        rev_adj = line_adj.T
        rev_adj = rev_adj / np.maximum(rev_adj.sum(axis=1, keepdims=True), 1e-6)
        self.register_buffer("line_adj", torch.from_numpy(line_adj.astype(np.float32)))
        self.register_buffer("reverse_line_adj", torch.from_numpy(rev_adj.astype(np.float32)))

    def forward(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        batch_size, num_edges, hist_len = speed_seq.shape
        x = _edge_sequence_input(node_seq, speed_seq, self.time_feat_dim)
        seq = x.reshape(batch_size * num_edges, hist_len, -1)
        _out, hidden = self.encoder(seq)
        h = hidden[-1].reshape(batch_size, num_edges, -1)
        views = torch.stack(
            [
                self.fixed_view(h, self.line_adj),
                self.reverse_view(h, self.reverse_line_adj),
                self.adaptive_view(h),
            ],
            dim=2,
        )
        weights = torch.softmax(self.view_score(views).squeeze(-1), dim=2).unsqueeze(-1)
        fused = (views * weights).sum(dim=2)
        return self.head(fused)


class NexuSQNSpeedModel(nn.Module):
    """Paper-inspired spatiotemporal MLP-Mixer adapter."""

    def __init__(
        self,
        *,
        num_edges: int,
        hist_len: int,
        pred_horizon: int,
        time_feat_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.time_feat_dim = int(time_feat_dim)
        self.hist_len = int(hist_len)
        self.input_proj = nn.Linear(1 + self.time_feat_dim, hidden_dim)
        self.edge_embedding = nn.Parameter(torch.randn(num_edges, hidden_dim) * 0.02)
        self.time_embedding = nn.Parameter(torch.randn(hist_len, hidden_dim) * 0.02)
        self.time_mixers = nn.ModuleList(
            [nn.Linear(hist_len, hist_len) for _ in range(max(int(num_layers), 1))]
        )
        self.channel_mixers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(max(int(num_layers), 1))
            ]
        )
        self.space_u = nn.Parameter(torch.randn(num_edges, max(hidden_dim // 2, 8)) * 0.02)
        self.space_v = nn.Parameter(torch.randn(num_edges, max(hidden_dim // 2, 8)) * 0.02)
        self.head = nn.Linear(hidden_dim, int(pred_horizon))

    def _space_adj(self) -> torch.Tensor:
        return torch.softmax(self.space_u @ self.space_v.T, dim=-1)

    def forward(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        x = _edge_sequence_input(node_seq, speed_seq, self.time_feat_dim)
        h = self.input_proj(x)
        h = h + self.edge_embedding.unsqueeze(0).unsqueeze(2)
        h = h + self.time_embedding.unsqueeze(0).unsqueeze(0)
        for time_mixer, channel_mixer in zip(self.time_mixers, self.channel_mixers):
            h = h + time_mixer(h.transpose(-1, -2)).transpose(-1, -2)
            h = h + channel_mixer(h)
            h = h + torch.einsum("ij,bjtd->bitd", self._space_adj().to(h.dtype), h)
        pooled = h.mean(dim=2)
        return self.head(pooled)


class SGSLGATnLSTMSpeedModel(nn.Module):
    """Paper-inspired task-oriented spatial graph learning plus GAT-nLSTM adapter."""

    def __init__(
        self,
        *,
        edge_index: np.ndarray | None,
        speed_adjacency: np.ndarray | None,
        hist_len: int,
        pred_horizon: int,
        time_feat_dim: int,
        hidden_dim: int,
        adaptive_topk: int,
        dropout: float,
    ) -> None:
        super().__init__()
        resolved_adj = resolve_speed_adjacency(edge_index, speed_adjacency)
        self.time_feat_dim = int(time_feat_dim)
        self.lstm = nn.LSTM(1 + self.time_feat_dim, hidden_dim, batch_first=True)
        self.spatial_graph = AdaptiveGraphMixLayer(
            num_edges=int(resolved_adj.shape[0]),
            hidden_dim=hidden_dim,
            rank=max(hidden_dim // 2, 8),
            topk=int(adaptive_topk),
            dropout=dropout,
        )
        self.gat_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gat_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gat_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gat_out = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, int(pred_horizon))
        self.hist_len = int(hist_len)

    def forward(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        batch_size, num_edges, hist_len = speed_seq.shape
        x = _edge_sequence_input(node_seq, speed_seq, self.time_feat_dim)
        seq = x.reshape(batch_size * num_edges, hist_len, -1)
        _out, hidden = self.lstm(seq)
        h = hidden[0][-1].reshape(batch_size, num_edges, -1)
        learned = self.spatial_graph(h)
        adj = self.spatial_graph.adjacency().to(h.dtype)
        q = self.gat_query(learned)
        k = self.gat_key(learned)
        v = self.gat_value(learned)
        scores = torch.einsum("bid,bjd->bij", q, k) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(adj.unsqueeze(0) <= 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        h_attn = torch.einsum("bij,bjd->bid", attn, v)
        h = self.norm(learned + self.gat_out(h_attn))
        return self.head(h)


class STGATEdgeSpeedModel(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        edge_index: np.ndarray,
        edge_lengths: np.ndarray,
        adj: np.ndarray,
        hidden_dim: int,
        num_heads: int,
        num_st_blocks: int,
        num_gtcn_layers: int,
        kernel_size: int,
        pred_horizon: int,
        node_feat_dim: int,
        adaptive_topk: int,
    ) -> None:
        super().__init__()
        self.model = STGATPredictor(
            num_nodes=int(num_nodes),
            edge_index=torch.from_numpy(edge_index),
            edge_lengths=torch.from_numpy(edge_lengths),
            adj_matrix=torch.from_numpy(adj),
            hidden_dim=int(hidden_dim),
            num_heads=int(num_heads),
            num_st_blocks=int(num_st_blocks),
            num_gtcn_layers=int(num_gtcn_layers),
            kernel_size=int(kernel_size),
            pred_horizon=int(pred_horizon),
            node_feat_dim=int(node_feat_dim),
            adaptive_topk=int(adaptive_topk),
        )

    def forward(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        temporal_context = node_seq[:, 0, :, 2:] if node_seq.shape[-1] > 2 else None
        return self.model.forward_v(speed_seq, temporal_context)


def build_model(args: argparse.Namespace, bundle: DatasetBundle) -> nn.Module:
    time_feat_dim = max(int(bundle.node_features.shape[-1]) - 2, 0)
    if args.model == "persistence":
        return PersistenceSpeedModel(args.pred_horizon)
    if args.model == "temporal_mlp":
        return TemporalMLPSpeedModel(
            hist_len=args.hist_len,
            pred_horizon=args.pred_horizon,
            time_feat_dim=time_feat_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    if args.model == "line_graph_gru":
        return LineGraphGRUSpeedModel(
            edge_index=bundle.edge_index,
            speed_adjacency=bundle.speed_adjacency,
            hist_len=args.hist_len,
            pred_horizon=args.pred_horizon,
            time_feat_dim=time_feat_dim,
            hidden_dim=args.hidden_dim,
            graph_layers=args.graph_layers,
            dropout=args.dropout,
        )
    if args.model == "astgnn_ltd":
        return ASTGNNLongTermSpeedModel(
            edge_index=bundle.edge_index,
            speed_adjacency=bundle.speed_adjacency,
            hist_len=args.hist_len,
            pred_horizon=args.pred_horizon,
            time_feat_dim=time_feat_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            graph_layers=args.graph_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
    if args.model == "icst_dnet":
        return ICSTDNetSpeedModel(
            edge_index=bundle.edge_index,
            speed_adjacency=bundle.speed_adjacency,
            hist_len=args.hist_len,
            pred_horizon=args.pred_horizon,
            time_feat_dim=time_feat_dim,
            hidden_dim=args.hidden_dim,
            adaptive_topk=args.adaptive_topk,
            graph_layers=args.graph_layers,
            dropout=args.dropout,
        )
    if args.model == "3s_tbln":
        return ThreeSTBLNSpeedModel(
            edge_index=bundle.edge_index,
            speed_adjacency=bundle.speed_adjacency,
            hist_len=args.hist_len,
            pred_horizon=args.pred_horizon,
            time_feat_dim=time_feat_dim,
            hidden_dim=args.hidden_dim,
            graph_layers=args.graph_layers,
            dropout=args.dropout,
        )
    if args.model == "mvstgcn":
        return MultiViewSTGCNSpeedModel(
            edge_index=bundle.edge_index,
            speed_adjacency=bundle.speed_adjacency,
            hist_len=args.hist_len,
            pred_horizon=args.pred_horizon,
            time_feat_dim=time_feat_dim,
            hidden_dim=args.hidden_dim,
            adaptive_topk=args.adaptive_topk,
            dropout=args.dropout,
        )
    if args.model == "nexusqn":
        return NexuSQNSpeedModel(
            num_edges=bundle.edge_speeds.shape[1],
            hist_len=args.hist_len,
            pred_horizon=args.pred_horizon,
            time_feat_dim=time_feat_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
    if args.model == "sgsl_gat_nlstm":
        return SGSLGATnLSTMSpeedModel(
            edge_index=bundle.edge_index,
            speed_adjacency=bundle.speed_adjacency,
            hist_len=args.hist_len,
            pred_horizon=args.pred_horizon,
            time_feat_dim=time_feat_dim,
            hidden_dim=args.hidden_dim,
            adaptive_topk=args.adaptive_topk,
            dropout=args.dropout,
        )
    if args.model == "stgat_edge":
        if bundle.dataset_format != "nyc_edge":
            raise ValueError("stgat_edge is only supported for edge-speed datasets in this benchmark script")
        return STGATEdgeSpeedModel(
            num_nodes=bundle.adj.shape[0],
            edge_index=bundle.edge_index,
            edge_lengths=bundle.edge_lengths,
            adj=bundle.adj,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_st_blocks=args.num_st_blocks,
            num_gtcn_layers=args.num_gtcn_layers,
            kernel_size=args.kernel_size,
            pred_horizon=args.pred_horizon,
            node_feat_dim=bundle.node_features.shape[-1],
            adaptive_topk=args.adaptive_topk,
        )
    raise ValueError(f"Unsupported model: {args.model}")


def infer_dataset_format(data_dir: str | Path, requested: str) -> str:
    if requested != "auto":
        return requested
    root = Path(data_dir)
    if (root / "speed_values.npy").exists():
        return "sensor_node"
    return "nyc_edge"


def load_nyc_dataset_bundle(args: argparse.Namespace) -> DatasetBundle:
    data = load_nyc_real_graph_features(
        args.data_dir,
        max_time_steps=args.max_time_steps,
        edge_length_source=args.edge_length_source,
        add_time_features=not args.disable_time_features,
    )
    node_features = data["node_features"]
    edge_speeds = data["edge_speeds"]
    num_time_steps = int(edge_speeds.shape[0])
    time_meta = load_time_meta_for_training(args.data_dir, num_time_steps)
    time_slot_minutes = infer_time_slot_minutes(time_meta)
    report_horizons = resolve_report_horizons(
        time_slot_minutes=time_slot_minutes,
        pred_horizon=args.pred_horizon,
        requested_minutes=parse_minutes(args.report_horizons_minutes),
        strict=bool(args.report_horizons_minutes),
    )

    split_indices = build_monthly_split_indices(time_meta, args.hist_len, args.pred_horizon)
    observed_mask = load_observed_time_mask(args.data_dir, num_time_steps)
    split_indices = filter_split_indices_by_time_mask(
        split_indices,
        observed_mask,
        args.hist_len,
        args.pred_horizon,
    )
    train_time_mask = build_window_time_mask(
        num_time_steps,
        split_indices["train"],
        args.hist_len,
        args.pred_horizon,
    )
    if not np.any(train_time_mask):
        raise ValueError("Train split is empty; increase --max-time-steps or check time_meta.csv.")
    normalization_stats = build_masked_normalization_stats(
        node_features,
        edge_speeds,
        train_time_mask,
        None,
    )
    node_features = normalize_node_features(node_features, normalization_stats)
    edge_speeds = normalize_speed_values(edge_speeds, normalization_stats, None)

    return DatasetBundle(
        dataset_name=args.dataset_name or "NYC edge speed",
        dataset_format="nyc_edge",
        adj=data["adj"],
        edge_index=data["edge_index"],
        edge_lengths=data["edge_lengths"],
        speed_adjacency=build_line_graph_adjacency(data["edge_index"]),
        node_features=node_features,
        edge_speeds=edge_speeds,
        speed_valid_mask=None,
        time_meta=time_meta,
        split_indices=split_indices,
        normalization_stats=normalization_stats,
        time_slot_minutes=int(time_slot_minutes),
        report_horizons=report_horizons,
        time_feature_names=list(data.get("time_feature_names", [])),
        split_description=CALENDAR_SPLIT_DESCRIPTION,
        preprocessing_summary={"speed_metric_protocol": "unmasked_all_values"},
    )


def load_sensor_dataset_bundle(args: argparse.Namespace) -> DatasetBundle:
    root = Path(args.data_dir)
    speed_path = root / "speed_values.npy"
    valid_path = root / "speed_valid_mask.npy"
    if not speed_path.exists():
        raise FileNotFoundError(f"Missing sensor speed file: {speed_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Missing sensor valid mask: {valid_path}")

    edge_speeds = np.load(speed_path).astype(np.float32)
    speed_valid_mask = np.load(valid_path).astype(bool)
    if edge_speeds.shape != speed_valid_mask.shape:
        raise ValueError(f"speed_values {edge_speeds.shape} and speed_valid_mask {speed_valid_mask.shape} differ")
    if args.max_time_steps > 0:
        t_use = min(int(args.max_time_steps), int(edge_speeds.shape[0]))
        edge_speeds = edge_speeds[:t_use]
        speed_valid_mask = speed_valid_mask[:t_use]

    adj = np.load(root / "adjacency_matrix.npy").astype(np.float32)
    adjacency_weights_path = root / "adjacency_weights.npy"
    adjacency_weights = (
        np.load(adjacency_weights_path).astype(np.float32)
        if adjacency_weights_path.exists()
        else adj.copy()
    )
    edge_index = np.load(root / "edge_index.npy").astype(np.int64)
    edge_lengths_path = root / "edge_lengths_km.npy"
    edge_lengths = (
        np.load(edge_lengths_path).astype(np.float32)
        if edge_lengths_path.exists()
        else np.ones(edge_index.shape[0], dtype=np.float32)
    )
    time_meta = pd.read_csv(root / "time_meta.csv").iloc[: edge_speeds.shape[0]].copy()
    time_meta["date"] = pd.to_datetime(time_meta["date"], errors="raise")
    num_time_steps, num_speed_items = edge_speeds.shape

    node_features = np.zeros((num_time_steps, num_speed_items, 2), dtype=np.float32)
    time_feature_names: list[str] = []
    if not args.disable_time_features:
        time_features, time_feature_names = build_cyclical_time_features(time_meta)
        expanded = np.broadcast_to(
            time_features[:, None, :],
            (num_time_steps, num_speed_items, time_features.shape[1]),
        )
        node_features = np.concatenate([node_features, expanded], axis=-1).astype(np.float32)

    time_slot_minutes = infer_time_slot_minutes(time_meta)
    report_horizons = resolve_report_horizons(
        time_slot_minutes=time_slot_minutes,
        pred_horizon=args.pred_horizon,
        requested_minutes=parse_minutes(args.report_horizons_minutes),
        strict=bool(args.report_horizons_minutes),
    )
    split_indices = build_monthly_split_indices(time_meta, args.hist_len, args.pred_horizon)
    split_indices = filter_split_indices_by_target_mask(
        split_indices,
        speed_valid_mask,
        args.hist_len,
        args.pred_horizon,
    )
    train_time_mask = build_window_time_mask(
        num_time_steps,
        split_indices["train"],
        args.hist_len,
        args.pred_horizon,
    )
    if not np.any(train_time_mask):
        raise ValueError("Train split is empty; increase --max-time-steps or check time_meta.csv.")

    edge_speeds, outlier_summary = clean_sensor_speed_values(
        edge_speeds,
        speed_valid_mask,
        train_time_mask,
        mode=args.sensor_outlier_cleaning,
        lower_quantile=args.outlier_lower_quantile,
        upper_quantile=args.outlier_upper_quantile,
    )
    normalization_stats = build_masked_normalization_stats(
        node_features,
        edge_speeds,
        train_time_mask,
        speed_valid_mask,
    )
    node_features = normalize_node_features(node_features, normalization_stats)
    edge_speeds = normalize_speed_values(edge_speeds, normalization_stats, speed_valid_mask)

    summary_path = root / "dataset_summary.json"
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    dataset_name = args.dataset_name or str(dataset_summary.get("dataset_name") or root.name)

    return DatasetBundle(
        dataset_name=dataset_name,
        dataset_format="sensor_node",
        adj=adj,
        edge_index=edge_index,
        edge_lengths=edge_lengths,
        speed_adjacency=normalize_square_adjacency(adjacency_weights, add_self_loops=True),
        node_features=node_features,
        edge_speeds=edge_speeds,
        speed_valid_mask=speed_valid_mask,
        time_meta=time_meta,
        split_indices=split_indices,
        normalization_stats=normalization_stats,
        time_slot_minutes=int(time_slot_minutes),
        report_horizons=report_horizons,
        time_feature_names=time_feature_names,
        split_description=CALENDAR_SPLIT_DESCRIPTION,
        preprocessing_summary={
            "speed_metric_protocol": "masked_zero_excluded",
            "outlier_cleaning": outlier_summary,
            "source_dataset_summary": dataset_summary,
        },
    )


def load_dataset_bundle(args: argparse.Namespace) -> DatasetBundle:
    args.dataset_format_resolved = infer_dataset_format(args.data_dir, args.dataset_format)
    if args.dataset_format_resolved == "sensor_node":
        return load_sensor_dataset_bundle(args)
    return load_nyc_dataset_bundle(args)


def make_loaders(args: argparse.Namespace, bundle: DatasetBundle) -> dict[str, DataLoader]:
    full_dataset = BenchmarkSpatioTemporalDataset(
        bundle.node_features,
        bundle.edge_speeds,
        hist_len=args.hist_len,
        pred_horizon=args.pred_horizon,
        speed_valid_mask=bundle.speed_valid_mask,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.device_resolved.type == "cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    return {
        "train": DataLoader(
            Subset(full_dataset, bundle.split_indices["train"]),
            shuffle=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            Subset(full_dataset, bundle.split_indices["val"]),
            shuffle=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            Subset(full_dataset, bundle.split_indices["test"]),
            shuffle=False,
            **loader_kwargs,
        ),
    }


def forecast_loss(
    args: argparse.Namespace,
    model: nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
    speed_seq: torch.Tensor | None = None,
    target_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_name = str(args.loss).lower()
    if loss_name == "auto":
        loss_name = "huber" if args.model == "astgnn_ltd" else "mse"
    if loss_name == "huber":
        per_item = F.smooth_l1_loss(
            prediction,
            target,
            beta=float(args.huber_beta),
            reduction="none",
        )
    else:
        per_item = torch.square(prediction - target)
    if target_mask is None:
        loss = per_item.mean()
    else:
        mask = target_mask.to(dtype=prediction.dtype)
        loss = torch.sum(per_item * mask) / torch.clamp(mask.sum(), min=1.0)
    if speed_seq is not None and hasattr(model, "auxiliary_loss"):
        aux = model.auxiliary_loss(speed_seq)
        loss = loss + float(args.aux_loss_weight) * aux
    return loss


def update_bucket(
    bucket: MetricBucket,
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> None:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        diff = diff[valid]
        target = target[valid]
        if diff.size == 0:
            return
    abs_diff = np.abs(diff)
    bucket.se += float(np.square(diff).sum())
    bucket.ae += float(abs_diff.sum())
    bucket.count += float(diff.size)
    mask = np.abs(target) > 1e-6
    if np.any(mask):
        bucket.ape += float((abs_diff[mask] / np.abs(target[mask])).sum())
        bucket.mape_count += float(mask.sum())


def finalize_bucket(bucket: MetricBucket) -> dict[str, float]:
    count = max(float(bucket.count), 1.0)
    mse = float(bucket.se / count)
    mae = float(bucket.ae / count)
    mape_count = max(float(bucket.mape_count), 1.0)
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": mae,
        "mape": float(100.0 * bucket.ape / mape_count),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    args: argparse.Namespace,
    bundle: DatasetBundle,
    max_batches: int = 0,
) -> dict[str, Any]:
    model.eval()
    device = args.device_resolved
    amp_enabled = args.amp_enabled
    amp_dtype = args.amp_dtype
    overall = MetricBucket()
    per_step = [MetricBucket() for _ in range(args.pred_horizon)]
    normalized_loss = 0.0
    num_batches = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            node_seq = batch["node_seq"].to(device, non_blocking=args.non_blocking)
            speed_seq = batch["speed_seq"].to(device, non_blocking=args.non_blocking)
            speed_target = batch["speed_target"].to(device, non_blocking=args.non_blocking)
            speed_target_mask = batch.get("speed_target_mask")
            if speed_target_mask is not None:
                speed_target_mask = speed_target_mask.to(device, non_blocking=args.non_blocking)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                speed_pred = model(node_seq, speed_seq)
                loss = forecast_loss(args, model, speed_pred, speed_target, target_mask=speed_target_mask)
            normalized_loss += float(loss.detach().cpu())
            num_batches += 1

            pred_raw = denormalize_speed_values(
                speed_pred.detach().float().cpu().numpy(),
                bundle.normalization_stats,
                edge_axis=1,
            )
            target_raw = denormalize_speed_values(
                speed_target.detach().float().cpu().numpy(),
                bundle.normalization_stats,
                edge_axis=1,
            )
            mask_np = None if speed_target_mask is None else speed_target_mask.detach().cpu().numpy().astype(bool)
            update_bucket(overall, pred_raw, target_raw, mask_np)
            for step_idx in range(args.pred_horizon):
                step_mask = None if mask_np is None else mask_np[..., step_idx]
                update_bucket(per_step[step_idx], pred_raw[..., step_idx], target_raw[..., step_idx], step_mask)

            if max_batches > 0 and batch_idx >= max_batches:
                break

    per_step_metrics = {}
    for step_idx, bucket in enumerate(per_step, start=1):
        per_step_metrics[f"step_{step_idx}"] = {
            "step": int(step_idx),
            "minutes": int(step_idx * bundle.time_slot_minutes),
            **finalize_bucket(bucket),
        }
    report = {}
    for minute, step in zip(
        bundle.report_horizons.get("resolved_minutes", []),
        bundle.report_horizons.get("resolved_steps", []),
    ):
        report[f"{int(minute)}min"] = dict(per_step_metrics[f"step_{int(step)}"])

    return {
        "normalized_loss": float(normalized_loss / max(num_batches, 1)),
        "raw_metrics": finalize_bucket(overall),
        "raw_metrics_per_step": per_step_metrics,
        "raw_metrics_report": report,
        "num_batches": int(num_batches),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    model.train()
    device = args.device_resolved
    total_loss = 0.0
    num_batches = 0
    for batch_idx, batch in enumerate(loader, start=1):
        node_seq = batch["node_seq"].to(device, non_blocking=args.non_blocking)
        speed_seq = batch["speed_seq"].to(device, non_blocking=args.non_blocking)
        speed_target = batch["speed_target"].to(device, non_blocking=args.non_blocking)
        speed_target_mask = batch.get("speed_target_mask")
        if speed_target_mask is not None:
            speed_target_mask = speed_target_mask.to(device, non_blocking=args.non_blocking)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=args.amp_dtype,
            enabled=args.amp_enabled,
        ):
            speed_pred = model(node_seq, speed_seq)
            loss = forecast_loss(args, model, speed_pred, speed_target, speed_seq, speed_target_mask)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        num_batches += 1
        if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
            break

    return {
        "train_normalized_loss": float(total_loss / max(num_batches, 1)),
        "num_batches": int(num_batches),
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_state_dict_if_requested(
    model: nn.Module,
    checkpoint: str,
    device: torch.device,
) -> dict[str, Any] | None:
    if not checkpoint:
        return None
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(state)!r}")

    if isinstance(model, STGATEdgeSpeedModel):
        try:
            model.model.load_state_dict(state, strict=True)
            return {
                "path": str(path),
                "mode": "strict_inner_model",
                "loaded_keys": int(len(state)),
                "skipped_keys": 0,
            }
        except RuntimeError as exc:
            target = model.model
            strict_error = str(exc)
    else:
        try:
            model.load_state_dict(state, strict=True)
            return {
                "path": str(path),
                "mode": "strict",
                "loaded_keys": int(len(state)),
                "skipped_keys": 0,
            }
        except RuntimeError as exc:
            target = model
            strict_error = str(exc)

    target_state = target.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    if not compatible:
        raise RuntimeError(
            f"Checkpoint {path} is incompatible with {type(model).__name__}; "
            f"strict load error was: {strict_error}"
        )
    incompatible = target.load_state_dict(compatible, strict=False)
    skipped = len(state) - len(compatible)
    print(
        "  checkpoint loaded in compatible mode: "
        f"loaded={len(compatible)} skipped={skipped}"
    )
    return {
        "path": str(path),
        "mode": "compatible_shape_matched",
        "loaded_keys": int(len(compatible)),
        "skipped_keys": int(skipped),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "strict_error": strict_error[:2000],
    }


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    args.device_resolved = resolve_device(args.device)
    args.amp_enabled, args.amp_dtype, args.precision_resolved = resolve_precision(
        args.precision,
        args.device_resolved,
    )
    args.non_blocking = bool(args.device_resolved.type == "cuda")
    args.dataset_format_resolved = infer_dataset_format(args.data_dir, args.dataset_format)
    if args.device_resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if not args.log_dir:
        args.log_dir = str(make_default_log_dir(args.model))
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"{args.dataset_format_resolved} speed benchmark | dataset={args.dataset_name or args.data_dir} | model={args.model}")
    print(f"  log_dir={log_dir}")
    print(f"  device={args.device_resolved} precision={args.precision_resolved}")
    bundle = load_dataset_bundle(args)
    print(
        f"  data: name={bundle.dataset_name} nodes={bundle.adj.shape[0]} graph_edges={bundle.edge_index.shape[0]} "
        f"speed_items={bundle.edge_speeds.shape[1]} "
        f"time_steps={bundle.edge_speeds.shape[0]} node_feat_dim={bundle.node_features.shape[-1]}"
    )
    print(f"  split: {bundle.split_description}")
    print(
        "  windows: "
        f"train={len(bundle.split_indices['train'])} "
        f"val={len(bundle.split_indices['val'])} "
        f"test={len(bundle.split_indices['test'])}"
    )
    if bundle.time_feature_names:
        print(f"  time_features={','.join(bundle.time_feature_names)}")

    loaders = make_loaders(args, bundle)
    model = build_model(args, bundle).to(args.device_resolved)
    checkpoint_load_summary = load_state_dict_if_requested(
        model,
        args.init_checkpoint,
        args.device_resolved,
    )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"  trainable_params={sum(p.numel() for p in trainable_params):,}")

    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_checkpoint = log_dir / "best_model.pt"
    should_train = bool(trainable_params) and not args.eval_only and args.epochs > 0

    if should_train:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        train_start = time.time()
        for epoch in range(1, int(args.epochs) + 1):
            epoch_start = time.time()
            train_record = train_one_epoch(model, loaders["train"], args=args, optimizer=optimizer)
            val_metrics = evaluate(
                model,
                loaders["val"],
                args=args,
                bundle=bundle,
                max_batches=args.max_eval_batches,
            )
            elapsed = time.time() - epoch_start
            record = {
                "epoch": int(epoch),
                "elapsed_sec": round(float(elapsed), 3),
                **train_record,
                "val_normalized_loss": val_metrics["normalized_loss"],
                "val_raw_speed_rmse": val_metrics["raw_metrics"]["rmse"],
                "val_raw_speed_mae": val_metrics["raw_metrics"]["mae"],
            }
            for label, metric in val_metrics["raw_metrics_report"].items():
                record[f"val_raw_speed_rmse_{label.lower()}"] = metric["rmse"]
                record[f"val_raw_speed_mse_{label.lower()}"] = metric["mse"]
                record[f"val_raw_speed_mae_{label.lower()}"] = metric["mae"]
            history.append(record)
            elapsed_total = time.time() - train_start
            progress = epoch / max(int(args.epochs), 1)
            eta = (elapsed_total / max(epoch, 1)) * max(int(args.epochs) - epoch, 0)
            horizon_text = format_horizon_metrics(val_metrics["raw_metrics_report"])
            print(
                f"  epoch {epoch:03d}/{int(args.epochs):03d} "
                f"progress={epoch}/{int(args.epochs)} ({progress * 100:.1f}%) "
                f"elapsed={format_duration(elapsed_total)} eta={format_duration(eta)} "
                f"train={record['train_normalized_loss']:.5f} "
                f"val={record['val_normalized_loss']:.5f} "
                f"val_rmse={record['val_raw_speed_rmse']:.4f} "
                f"{horizon_text}",
                flush=True,
            )
            if val_metrics["normalized_loss"] < best_val_loss:
                best_val_loss = float(val_metrics["normalized_loss"])
                best_epoch = int(epoch)
                torch.save(model.state_dict(), best_checkpoint)
    else:
        print("  training skipped; evaluating the current model.")

    if best_checkpoint.exists() and should_train:
        model.load_state_dict(torch.load(best_checkpoint, map_location=args.device_resolved))

    val_metrics = evaluate(
        model,
        loaders["val"],
        args=args,
        bundle=bundle,
        max_batches=args.max_eval_batches,
    )
    test_metrics = evaluate(
        model,
        loaders["test"],
        args=args,
        bundle=bundle,
        max_batches=args.max_eval_batches,
    )

    if not np.isfinite(best_val_loss):
        best_val_loss = float(val_metrics["normalized_loss"])
        best_epoch = 0

    if not best_checkpoint.exists():
        torch.save(model.state_dict(), best_checkpoint)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "program": "paper_speed_benchmarks/nyc_speed_prediction.py",
        "dataset": bundle.dataset_name,
        "dataset_format": bundle.dataset_format,
        "model": args.model,
        "args": {
            key: value
            for key, value in vars(args).items()
            if key not in {"device_resolved", "amp_dtype"}
        },
        "model_config": {
            "hist_len": int(args.hist_len),
            "pred_horizon": int(args.pred_horizon),
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "graph_layers": int(args.graph_layers),
            "num_heads": int(args.num_heads),
            "num_st_blocks": int(args.num_st_blocks),
            "adaptive_topk": int(args.adaptive_topk),
            "loss": str(args.loss),
            "aux_loss_weight": float(args.aux_loss_weight),
        },
        "data_summary": {
            "num_nodes": int(bundle.adj.shape[0]),
            "num_graph_edges": int(bundle.edge_index.shape[0]),
            "num_speed_items": int(bundle.edge_speeds.shape[1]),
            "num_time_steps": int(bundle.edge_speeds.shape[0]),
            "node_feat_dim": int(bundle.node_features.shape[-1]),
            "time_feature_names": bundle.time_feature_names,
            "time_slot_minutes": int(bundle.time_slot_minutes),
            "split_policy": bundle.split_description,
            "split_window_counts": {
                key: int(len(value)) for key, value in bundle.split_indices.items()
            },
            "preprocessing": bundle.preprocessing_summary,
        },
        "report_horizons": bundle.report_horizons,
        "normalization": serialize_normalization_stats(bundle.normalization_stats),
        "best_epoch": int(best_epoch),
        "best_val_normalized_loss": None if not np.isfinite(best_val_loss) else float(best_val_loss),
        "selected_checkpoint": str(best_checkpoint),
        "init_checkpoint_load": checkpoint_load_summary,
    }

    metrics_payload = {
        "selected_checkpoint": str(best_checkpoint),
        "selected_checkpoint_metric_name": "val_normalized_loss",
        "selected_checkpoint_metric": (
            None if not np.isfinite(best_val_loss) else float(best_val_loss)
        ),
        "validation": val_metrics,
        "test": test_metrics,
        "raw_metrics": {"speed": test_metrics["raw_metrics"]},
        "raw_metrics_per_step": {"speed": test_metrics["raw_metrics_per_step"]},
        "raw_metrics_report": {"speed": test_metrics["raw_metrics_report"]},
        "report_horizons": bundle.report_horizons,
        "speed_metric_protocol": bundle.preprocessing_summary.get("speed_metric_protocol", "unmasked_all_values"),
    }

    save_json(log_dir / "benchmark_meta.json", meta)
    save_json(log_dir / "predictor_log.json", history)
    save_json(log_dir / "predictor_test_metrics.json", metrics_payload)

    report_text = format_horizon_metrics(test_metrics["raw_metrics_report"])
    print(
        "  test: "
        f"mse={test_metrics['raw_metrics']['mse']:.4f} "
        f"rmse={test_metrics['raw_metrics']['rmse']:.4f} "
        f"mae={test_metrics['raw_metrics']['mae']:.4f} "
        f"mape={test_metrics['raw_metrics']['mape']:.2f}% "
        f"{report_text}",
        flush=True,
    )
    print(f"  wrote {log_dir / 'predictor_test_metrics.json'}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train/evaluate graph speed benchmark predictors."
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, default="temporal_mlp")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--dataset-format", choices=["auto", "nyc_edge", "sensor_node"], default="auto")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--edge-length-source", choices=["osrm", "centroid"], default="osrm")
    parser.add_argument("--max-time-steps", type=int, default=0)
    parser.add_argument("--disable-time-features", action="store_true")
    parser.add_argument("--hist-len", type=int, default=14)
    parser.add_argument("--pred-horizon", type=int, default=4)
    parser.add_argument("--report-horizons-minutes", default="15,30,60")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--graph-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-st-blocks", type=int, default=2)
    parser.add_argument("--num-gtcn-layers", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--adaptive-topk", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--loss", choices=["auto", "mse", "huber"], default="auto")
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--aux-loss-weight", type=float, default=0.1)
    parser.add_argument("--sensor-outlier-cleaning", choices=["none", "train_quantile_clip"], default="none")
    parser.add_argument("--outlier-lower-quantile", type=float, default=0.01)
    parser.add_argument("--outlier-upper-quantile", type=float, default=0.99)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp32"], default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
