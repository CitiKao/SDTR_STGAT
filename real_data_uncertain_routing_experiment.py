from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from data_loader import load_nyc_real_graph_features
from ddqn_agent import DoubleDQNAgent
from dispatch import build_dispatch_od_pairs, greedy_dispatch
from graph_env import CityGraph, RoutingEnv, create_graph_from_data, infer_max_neighbors
from predictor_normalization import (
    denormalize_speed_values,
    load_normalization_stats,
    normalize_node_features,
    normalize_speed_features,
)
from stgat_model import (
    GTCN,
    GatedFusion,
    STGATPredictor,
    SparseGATLayer,
    build_line_graph_edge_index,
)
from superzone_graph import aggregate_speed_profile_to_rl_edges, load_superzone_artifacts
from train_predictor import build_monthly_split_indices


@dataclass
class RouteResult:
    reached: bool
    travel_time: float
    travel_dist: float
    reward: float
    steps: int
    path: list[int]
    last_loss: float | None = None


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but no CUDA device is available.")
    return device


def configure_runtime(device: torch.device, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def mean_finite(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def weighted_choice(pairs: list[tuple[int, int, int]], rng: np.random.RandomState) -> tuple[int, int, int]:
    weights = np.asarray([max(int(c), 0) for _, _, c in pairs], dtype=np.float64)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("dispatch pairs have no positive count")
    pick = float(rng.random_sample() * total)
    idx = int(np.searchsorted(np.cumsum(weights), pick, side="right"))
    idx = min(idx, len(pairs) - 1)
    return pairs[idx]


def edge_lookup(edge_index: np.ndarray) -> dict[tuple[int, int], int]:
    return {
        (int(src), int(dst)): int(i)
        for i, (src, dst) in enumerate(np.asarray(edge_index, dtype=np.int32))
    }


def simulate_path_under_real_speeds(
    path: list[int],
    *,
    edge_id: dict[tuple[int, int], int],
    edge_lengths: np.ndarray,
    real_speed_profile: np.ndarray,
    start_slot: int,
    time_slot_duration_hours: float,
    num_time_slots: int,
    alpha: float,
    beta: float,
    rho: float,
) -> RouteResult:
    if len(path) < 2:
        return RouteResult(False, float("inf"), float("inf"), -20.0, 0, path)

    total_time = 0.0
    total_dist = 0.0
    reward = 0.0
    slot = int(np.clip(start_slot, 0, num_time_slots - 1))

    for src, dst in zip(path[:-1], path[1:]):
        idx = edge_id.get((int(src), int(dst)))
        if idx is None:
            return RouteResult(False, float("inf"), float("inf"), -20.0, len(path) - 1, path)
        length = float(edge_lengths[idx])
        speed = float(real_speed_profile[idx, slot])
        travel_time = length / max(speed, 1e-5)
        total_time += travel_time
        total_dist += length
        reward += -alpha * travel_time - beta * length
        elapsed_slots = int(total_time / max(time_slot_duration_hours, 1e-6))
        slot = min(start_slot + elapsed_slots, num_time_slots - 1)

    reward += rho
    return RouteResult(True, total_time, total_dist, reward, len(path) - 1, path)


def dijkstra_result(
    graph: CityGraph,
    *,
    edge_index: np.ndarray,
    edge_lengths: np.ndarray,
    pred_speed_profile: np.ndarray,
    real_speed_profile: np.ndarray,
    origin: int,
    dest: int,
    use_real_speed_for_planning: bool,
    time_slot_duration_hours: float,
    alpha: float,
    beta: float,
    rho: float,
) -> RouteResult:
    graph.update_predicted_speeds(edge_index, pred_speed_profile[:, 0])
    graph.update_real_speeds(edge_index, real_speed_profile[:, 0])
    path, _, _ = graph.dijkstra(
        origin,
        dest,
        use_real_speed=use_real_speed_for_planning,
    )
    if not path:
        return RouteResult(False, float("inf"), float("inf"), -20.0, 0, [])
    return simulate_path_under_real_speeds(
        path,
        edge_id=edge_lookup(edge_index),
        edge_lengths=edge_lengths,
        real_speed_profile=real_speed_profile,
        start_slot=0,
        time_slot_duration_hours=time_slot_duration_hours,
        num_time_slots=real_speed_profile.shape[1],
        alpha=alpha,
        beta=beta,
        rho=rho,
    )


def run_ddqn_episode(
    env: RoutingEnv,
    agent: DoubleDQNAgent,
    *,
    origin: int,
    dest: int,
    train: bool,
) -> RouteResult:
    state, mask, _ = env.reset(origin, dest, time_slot=0)
    done = False
    total_reward = 0.0
    last_loss: float | None = None
    path = [int(origin)]

    while not done:
        action = agent.select_action(state, mask, greedy=not train)
        result = env.step(action)
        if train:
            agent.store_transition(
                state,
                action,
                result.reward,
                result.state,
                result.done,
                mask,
                result.action_mask,
            )
            loss = agent.learn()
            if loss is not None and math.isfinite(float(loss)):
                last_loss = float(loss)
        state = result.state
        mask = result.action_mask
        total_reward += float(result.reward)
        done = bool(result.done)
        path = [int(x) for x in result.info.get("path_history", path)]

    return RouteResult(
        bool(result.info.get("reached_goal", False)),
        float(result.info.get("total_travel_time", float("nan"))),
        float(result.info.get("total_distance", float("nan"))),
        float(total_reward),
        int(result.info.get("steps", 0)),
        path,
        last_loss=last_loss,
    )


class PersistenceSpeedProvider:
    name = "persistence"

    def __init__(self, edge_speeds: np.ndarray, hist_len: int, pred_horizon: int) -> None:
        self.edge_speeds = edge_speeds
        self.hist_len = int(hist_len)
        self.pred_horizon = int(pred_horizon)

    def predict_base_edges(self, window_idx: int) -> np.ndarray:
        observed_t = int(window_idx + self.hist_len - 1)
        last_speed = np.asarray(self.edge_speeds[observed_t], dtype=np.float32)
        return np.repeat(last_speed[:, None], self.pred_horizon, axis=1)

    def predict_base_edges_batch(self, window_indices: list[int] | np.ndarray) -> np.ndarray:
        idx = np.asarray(window_indices, dtype=np.int64)
        observed_t = idx + self.hist_len - 1
        last_speed = np.asarray(self.edge_speeds[observed_t], dtype=np.float32)
        return np.repeat(last_speed[:, :, None], self.pred_horizon, axis=2)


class StgatEdgeSpeedAdapter(torch.nn.Module):
    """Speed-only STGAT adapter for exported NYC V checkpoints."""

    def __init__(
        self,
        *,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
        hidden_dim: int,
        num_heads: int,
        num_st_blocks: int,
        num_gtcn_layers: int,
        kernel_size: int,
        pred_horizon: int,
        time_feat_dim: int,
        edge_input_dim: int,
        speed_use_adaptive: bool,
        speed_adaptive_topk: int,
        adaptive_emb: int = 10,
    ) -> None:
        super().__init__()
        num_edges = int(edge_index.shape[0])
        d_per_head = hidden_dim // num_heads
        if hidden_dim != d_per_head * num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.time_feat_dim = int(time_feat_dim)
        self.edge_input_dim = int(edge_input_dim)
        self.use_edge_domain_length_feature = self.edge_input_dim == self.time_feat_dim + 2
        self.speed_use_adaptive = bool(speed_use_adaptive)
        self.speed_adaptive_topk = int(speed_adaptive_topk)

        self.register_buffer("edge_index", edge_index.long())
        self.register_buffer("edge_lengths", edge_lengths.float())
        self.register_buffer("line_edge_index", build_line_graph_edge_index(edge_index.long()))

        self.edge_proj = torch.nn.Linear(self.edge_input_dim, hidden_dim)
        self.e_gtcn = torch.nn.ModuleList(
            [GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size) for _ in range(num_st_blocks)]
        )
        self.e_gat = torch.nn.ModuleList(
            [SparseGATLayer(hidden_dim, d_per_head, num_heads, concat=True) for _ in range(num_st_blocks)]
        )

        if self.speed_use_adaptive:
            self.speed_emb_src = torch.nn.Parameter(torch.randn(num_edges, adaptive_emb) * 0.1)
            self.speed_emb_dst = torch.nn.Parameter(torch.randn(num_edges, adaptive_emb) * 0.1)
            self.e_gtcn_adp = torch.nn.ModuleList(
                [GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size) for _ in range(num_st_blocks)]
            )
            self.e_gat_adp = torch.nn.ModuleList(
                [SparseGATLayer(hidden_dim, d_per_head, num_heads, concat=True) for _ in range(num_st_blocks)]
            )
            self.speed_fusion = GatedFusion(hidden_dim)

        self.speed_head = torch.nn.Linear(hidden_dim, pred_horizon)

    def _build_edge_input(
        self,
        speed_seq: torch.Tensor,
        temporal_feat_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [speed_seq.unsqueeze(-1)]
        if self.use_edge_domain_length_feature:
            lengths = self.edge_lengths.to(speed_seq.dtype).view(1, -1, 1, 1)
            parts.append(lengths.expand(speed_seq.shape[0], -1, speed_seq.shape[2], -1))
        if self.time_feat_dim > 0:
            if temporal_feat_seq is None:
                temporal_feat_seq = speed_seq.new_zeros(
                    speed_seq.shape[0],
                    speed_seq.shape[2],
                    self.time_feat_dim,
                )
            temporal_feat = temporal_feat_seq.unsqueeze(1).expand(-1, speed_seq.shape[1], -1, -1)
            parts.append(temporal_feat)
        edge_input = torch.cat(parts, dim=-1)
        if edge_input.shape[-1] != self.edge_input_dim:
            raise ValueError(
                f"edge input dim {edge_input.shape[-1]} does not match checkpoint dim {self.edge_input_dim}"
            )
        return edge_input

    def _adaptive_edge_index(self) -> torch.Tensor:
        scores = torch.relu(self.speed_emb_src @ self.speed_emb_dst.T)
        num_edges = scores.shape[0]
        keep_topk = self.speed_adaptive_topk > 0 and self.speed_adaptive_topk < num_edges
        if keep_topk:
            _, topk_idx = scores.topk(self.speed_adaptive_topk, dim=1)
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(1, topk_idx, True)
        else:
            mask = torch.ones_like(scores, dtype=torch.bool)
        diag = torch.arange(num_edges, device=scores.device)
        mask[diag, diag] = True
        recv, send = torch.nonzero(mask, as_tuple=True)
        return torch.stack([recv, send], dim=0).long()

    def _run_edge_path(self, edge_h: torch.Tensor, edge_index: torch.Tensor, *, adaptive: bool) -> torch.Tensor:
        blocks = self.e_gtcn_adp if adaptive else self.e_gtcn
        gats = self.e_gat_adp if adaptive else self.e_gat
        bsz, num_edges, steps, _ = edge_h.shape
        x = edge_h
        for gtcn, gat in zip(blocks, gats):
            x = gtcn(x)
            x_flat = x.permute(0, 2, 1, 3).reshape(bsz * steps, num_edges, -1)
            x_flat = gat(x_flat, edge_index)
            x = x_flat.reshape(bsz, steps, num_edges, -1).permute(0, 2, 1, 3)
        return x[:, :, -1, :]

    def forward_v(
        self,
        speed_seq: torch.Tensor,
        temporal_feat_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_h = self.edge_proj(self._build_edge_input(speed_seq, temporal_feat_seq))
        h_fix = self._run_edge_path(edge_h, self.line_edge_index, adaptive=False)
        if self.speed_use_adaptive:
            h_adp = self._run_edge_path(edge_h, self._adaptive_edge_index(), adaptive=True)
            h = self.speed_fusion(h_fix, h_adp)
        else:
            h = h_fix
        return self.speed_head(h)


class StgatSpeedProvider:
    name = "stgat"

    def __init__(
        self,
        *,
        data_dir: Path,
        checkpoint_dir: Path,
        device: torch.device,
        edge_length_source: str,
        max_time_steps: int = 0,
    ) -> None:
        meta_path = checkpoint_dir / "stgat_meta.json"
        ckpt_path = checkpoint_dir / "stgat_best.pt"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing STGAT meta: {meta_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"missing STGAT checkpoint: {ckpt_path}")

        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.hist_len = int(self.meta["hist_len"])
        self.pred_horizon = int(self.meta["pred_horizon"])
        self.device = device
        self.normalization = load_normalization_stats(self.meta["normalization"])

        data = load_nyc_real_graph_features(
            data_dir,
            max_time_steps=max_time_steps,
            edge_length_source=edge_length_source,
            add_time_features=bool(self.meta.get("use_time_features", False)),
        )
        self.node_features = np.asarray(data["node_features"], dtype=np.float32)
        self.edge_speeds = np.asarray(data["edge_speeds"], dtype=np.float32)
        meta_edge_index = np.asarray(self.meta.get("edge_index", []), dtype=np.int64)
        data_edge_index = np.asarray(data["edge_index"], dtype=np.int64)
        if meta_edge_index.size and not np.array_equal(meta_edge_index, data_edge_index):
            raise ValueError("Current data edge_index does not match the STGAT checkpoint metadata.")
        expected_time_features = list(self.meta.get("time_feature_names", []))
        actual_time_features = list(data.get("time_feature_names", []))
        if expected_time_features != actual_time_features:
            raise ValueError(
                f"Time feature mismatch: checkpoint={expected_time_features}, data={actual_time_features}"
            )
        expected_node_dim = int(self.meta.get("node_feat_dim", self.node_features.shape[-1]))
        if self.node_features.shape[-1] != expected_node_dim:
            raise ValueError(
                f"node feature dim {self.node_features.shape[-1]} does not match checkpoint {expected_node_dim}"
            )
        if self.normalization is not None:
            norm_mean = np.asarray(self.normalization["speed"]["mean"], dtype=np.float32)
            if norm_mean.shape[0] != self.edge_speeds.shape[1]:
                raise ValueError("Speed normalization vector length does not match edge_speeds edge count.")

        try:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=device)
        time_feat_dim = max(int(self.meta.get("node_feat_dim", self.node_features.shape[-1])) - 2, 0)
        edge_input_dim = int(state["edge_proj.weight"].shape[1])
        if "speed_emb_src" in state or bool(self.meta.get("speed_use_adaptive", False)):
            self.model = StgatEdgeSpeedAdapter(
                edge_index=torch.from_numpy(np.asarray(data["edge_index"], dtype=np.int64)),
                edge_lengths=torch.from_numpy(np.asarray(data["edge_lengths"], dtype=np.float32)),
                hidden_dim=int(self.meta["hidden_dim"]),
                num_heads=int(self.meta["num_heads"]),
                num_st_blocks=int(self.meta["num_st_blocks"]),
                num_gtcn_layers=int(self.meta["num_gtcn_layers"]),
                kernel_size=int(self.meta["kernel_size"]),
                pred_horizon=self.pred_horizon,
                time_feat_dim=time_feat_dim,
                edge_input_dim=edge_input_dim,
                speed_use_adaptive="speed_emb_src" in state,
                speed_adaptive_topk=int(self.meta.get("speed_adaptive_topk", self.meta.get("adaptive_topk", 16))),
            ).to(device)
            model_state = self.model.state_dict()
            filtered = {k: v for k, v in state.items() if k in model_state}
            missing = sorted(set(model_state) - set(filtered))
            if missing:
                raise RuntimeError(f"STGAT speed adapter missing checkpoint keys: {missing[:10]}")
            self.model.load_state_dict(filtered, strict=True)
        else:
            self.model = STGATPredictor(
                num_nodes=int(self.meta["num_nodes"]),
                edge_index=torch.from_numpy(np.asarray(data["edge_index"], dtype=np.int64)),
                edge_lengths=torch.from_numpy(np.asarray(data["edge_lengths"], dtype=np.float32)),
                adj_matrix=torch.from_numpy(np.asarray(data["adj"], dtype=np.float32)),
                hidden_dim=int(self.meta["hidden_dim"]),
                num_heads=int(self.meta["num_heads"]),
                num_st_blocks=int(self.meta["num_st_blocks"]),
                num_gtcn_layers=int(self.meta["num_gtcn_layers"]),
                kernel_size=int(self.meta["kernel_size"]),
                pred_horizon=self.pred_horizon,
                node_feat_dim=int(self.meta.get("node_feat_dim", self.node_features.shape[-1])),
                adaptive_topk=int(self.meta.get("adaptive_topk", 16)),
            ).to(device)
            self.model.load_state_dict(state, strict=True)
        self.model.eval()

    def predict_base_edges(self, window_idx: int) -> np.ndarray:
        return self.predict_base_edges_batch([window_idx])[0]

    def predict_base_edges_batch(self, window_indices: list[int] | np.ndarray) -> np.ndarray:
        indices = np.asarray(window_indices, dtype=np.int64)
        h = self.hist_len
        node_seq_raw = np.stack([self.node_features[int(i) : int(i) + h] for i in indices], axis=0)
        speed_seq_raw = np.stack([self.edge_speeds[int(i) : int(i) + h] for i in indices], axis=0)
        node_seq = normalize_node_features(node_seq_raw, self.normalization)
        speed_seq = normalize_speed_features(speed_seq_raw, self.normalization, edge_axis=2)
        node_t = torch.from_numpy(node_seq.transpose(0, 2, 1, 3)).float().to(self.device)
        speed_t = torch.from_numpy(speed_seq.transpose(0, 2, 1)).float().to(self.device)
        temporal = node_t[:, 0, :, 2:] if node_t.shape[-1] > 2 else None
        with torch.no_grad():
            pred_norm = self.model.forward_v(speed_t, temporal).detach().float().cpu().numpy()
        pred = denormalize_speed_values(pred_norm, self.normalization, edge_axis=1)
        return np.maximum(pred.astype(np.float32), 1.0)


def build_provider(
    args: argparse.Namespace,
    *,
    data_dir: Path,
    edge_speeds: np.ndarray,
    device: torch.device,
) -> PersistenceSpeedProvider | StgatSpeedProvider:
    if args.pred_source == "persistence":
        return PersistenceSpeedProvider(edge_speeds, args.hist_len, args.pred_horizon)
    if args.pred_source == "stgat":
        provider = StgatSpeedProvider(
            data_dir=data_dir,
            checkpoint_dir=Path(args.stgat_ckpt_dir),
            device=device,
            edge_length_source=args.edge_length_source,
            max_time_steps=args.max_time_steps,
        )
        if provider.pred_horizon < args.pred_horizon:
            raise ValueError(
                f"STGAT horizon {provider.pred_horizon} is shorter than requested {args.pred_horizon}"
            )
        if provider.hist_len != args.hist_len:
            raise ValueError(
                f"STGAT hist_len {provider.hist_len} differs from --hist-len {args.hist_len}"
            )
        return provider
    raise ValueError(f"unsupported pred source: {args.pred_source}")


def summarize_window(rows: list[dict]) -> dict:
    if not rows:
        return {}
    ddqn_success = np.asarray([r["ddqn_reached"] for r in rows], dtype=np.float32)
    fallback_used = np.asarray([r["fallback_used"] for r in rows], dtype=np.float32)
    pred_times = np.asarray([r["dijkstra_pred_time"] for r in rows], dtype=np.float64)
    real_times = np.asarray([r["dijkstra_real_time"] for r in rows], dtype=np.float64)
    posthoc_times = np.asarray([r["posthoc_fallback_time"] for r in rows], dtype=np.float64)
    ddqn_ratios = [
        float(r["ddqn_time"]) / float(r["dijkstra_pred_time"])
        for r in rows
        if r["ddqn_reached"] and float(r["dijkstra_pred_time"]) > 0
    ]
    hybrid_ratios = [
        float(r["posthoc_fallback_time"]) / float(r["dijkstra_pred_time"])
        for r in rows
        if r["posthoc_fallback_reached"] and float(r["dijkstra_pred_time"]) > 0
    ]
    pred_real_ratios = [
        float(r["dijkstra_pred_time"]) / float(r["dijkstra_real_time"])
        for r in rows
        if float(r["dijkstra_real_time"]) > 0
    ]
    losses = [
        float(r["loss"])
        for r in rows
        if r["loss"] is not None and r["loss"] != "" and math.isfinite(float(r["loss"]))
    ]
    ddqn_win = [
        1.0 if float(r["ddqn_time"]) < float(r["dijkstra_pred_time"]) else 0.0
        for r in rows
        if r["ddqn_reached"] and float(r["dijkstra_pred_time"]) > 0
    ]
    return {
        "ddqn_sr": float(ddqn_success.mean() * 100.0),
        "fallback_rate": float(fallback_used.mean() * 100.0),
        "avg_reward": float(np.mean([float(r["ddqn_reward"]) for r in rows])),
        "avg_loss": float(np.mean(losses)) if losses else float("nan"),
        "ddqn_time_success": mean_finite([r["ddqn_time"] for r in rows if r["ddqn_reached"]]),
        "posthoc_fallback_time": mean_finite(posthoc_times),
        "dijkstra_pred_time": mean_finite(pred_times),
        "dijkstra_real_time": mean_finite(real_times),
        "ddqn_over_pred_success": mean_finite(ddqn_ratios),
        "posthoc_fallback_over_pred": mean_finite(hybrid_ratios),
        "pred_over_real": mean_finite(pred_real_ratios),
        "ddqn_win_rate_success": float(np.mean(ddqn_win) * 100.0) if ddqn_win else float("nan"),
        "avg_dispatch_count": float(np.mean([float(r["dispatch_count"]) for r in rows])),
        "route_filter_match_rate": float(np.mean([float(r["route_filter_matched"]) for r in rows]) * 100.0),
        "pred_real_path_diff_rate": float(np.mean([float(r["dijkstra_pred_path_differs_real"]) for r in rows]) * 100.0),
        "avg_pred_real_regret_ratio": mean_finite([r["pred_real_regret_ratio"] for r in rows]),
        "avg_dijkstra_pred_hops": float(np.mean([float(r["dijkstra_pred_hops"]) for r in rows])),
        "avg_candidate_attempts": float(np.mean([float(r["candidate_attempts"]) for r in rows])),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def route_window_to_profiles(
    *,
    window_idx: int,
    hist_len: int,
    pred_horizon: int,
    edge_speeds: np.ndarray,
    provider: PersistenceSpeedProvider | StgatSpeedProvider,
    artifacts: dict,
) -> tuple[np.ndarray, np.ndarray]:
    target_t = int(window_idx + hist_len)
    base_pred = provider.predict_base_edges(window_idx)[:, :pred_horizon]
    base_real = np.asarray(edge_speeds[target_t : target_t + pred_horizon], dtype=np.float32).T
    pred_rl = aggregate_speed_profile_to_rl_edges(
        base_pred,
        artifacts["rl_edge_speed_mapping_offsets"],
        artifacts["rl_edge_speed_mapping_indices"],
        artifacts["rl_edge_speeds_kmh"],
    )
    real_rl = aggregate_speed_profile_to_rl_edges(
        base_real,
        artifacts["rl_edge_speed_mapping_offsets"],
        artifacts["rl_edge_speed_mapping_indices"],
        artifacts["rl_edge_speeds_kmh"],
    )
    return np.maximum(pred_rl, 1.0), np.maximum(real_rl, 1.0)


def route_windows_to_profiles_batch(
    *,
    window_indices: list[int] | np.ndarray,
    hist_len: int,
    pred_horizon: int,
    edge_speeds: np.ndarray,
    provider: PersistenceSpeedProvider | StgatSpeedProvider,
    artifacts: dict,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    windows = [int(w) for w in window_indices]
    base_pred_batch = provider.predict_base_edges_batch(windows)[:, :, :pred_horizon]
    pred_profiles: list[np.ndarray] = []
    real_profiles: list[np.ndarray] = []
    for batch_idx, window_idx in enumerate(windows):
        target_t = int(window_idx + hist_len)
        base_real = np.asarray(edge_speeds[target_t : target_t + pred_horizon], dtype=np.float32).T
        pred_rl = aggregate_speed_profile_to_rl_edges(
            base_pred_batch[batch_idx],
            artifacts["rl_edge_speed_mapping_offsets"],
            artifacts["rl_edge_speed_mapping_indices"],
            artifacts["rl_edge_speeds_kmh"],
        )
        real_rl = aggregate_speed_profile_to_rl_edges(
            base_real,
            artifacts["rl_edge_speed_mapping_offsets"],
            artifacts["rl_edge_speed_mapping_indices"],
            artifacts["rl_edge_speeds_kmh"],
        )
        pred_profiles.append(np.maximum(pred_rl, 1.0))
        real_profiles.append(np.maximum(real_rl, 1.0))
    return pred_profiles, real_profiles


def sample_dispatch_od(
    *,
    window_idx: int,
    hist_len: int,
    artifacts: dict,
    rng: np.random.RandomState,
    dispatch_source: str,
) -> tuple[int, int, int, int]:
    target_t = int(window_idx + hist_len)
    if dispatch_source == "persistence":
        dispatch_t = target_t - 1
    elif dispatch_source == "real_future_oracle":
        dispatch_t = target_t
    else:
        raise ValueError(f"unsupported dispatch source: {dispatch_source}")

    demand = np.maximum(artifacts["region_demand"][dispatch_t], 0.0)
    supply = np.maximum(artifacts["region_supply"][dispatch_t], 0.0)
    matrix = greedy_dispatch(
        demand,
        supply,
        artifacts["dispatch_duration_hours"],
        skip_unreachable=True,
    )
    pairs = build_dispatch_od_pairs(matrix)
    if not pairs:
        raise ValueError("empty dispatch matrix")
    origin, dest, count = weighted_choice(pairs, rng)
    return int(origin), int(dest), int(count), int(dispatch_t)


def route_pair_passes_filters(
    *,
    pred_result: RouteResult,
    real_result: RouteResult,
    min_pred_hops: int,
    min_pred_distance_km: float,
    require_pred_real_path_diff: bool,
    min_pred_real_regret_ratio: float,
) -> bool:
    if not pred_result.reached or not real_result.reached:
        return False
    if pred_result.steps < int(min_pred_hops):
        return False
    if pred_result.travel_dist < float(min_pred_distance_km):
        return False
    if require_pred_real_path_diff and pred_result.path == real_result.path:
        return False
    if min_pred_real_regret_ratio > 1.0:
        if real_result.travel_time <= 0 or not math.isfinite(real_result.travel_time):
            return False
        if pred_result.travel_time / real_result.travel_time < float(min_pred_real_regret_ratio):
            return False
    return True


def make_route_candidate(
    *,
    window_idx: int,
    origin: int,
    dest: int,
    dispatch_count: int,
    dispatch_t: int,
    pred_profile: np.ndarray,
    real_profile: np.ndarray,
    graph: CityGraph,
    rl_edge_index: np.ndarray,
    rl_edge_lengths: np.ndarray,
    args: argparse.Namespace,
    apply_filters: bool = True,
) -> dict | None:
    pred_result = dijkstra_result(
        graph,
        edge_index=rl_edge_index,
        edge_lengths=rl_edge_lengths,
        pred_speed_profile=pred_profile,
        real_speed_profile=real_profile,
        origin=origin,
        dest=dest,
        use_real_speed_for_planning=False,
        time_slot_duration_hours=args.time_slot_minutes / 60.0,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
    )
    real_result = dijkstra_result(
        graph,
        edge_index=rl_edge_index,
        edge_lengths=rl_edge_lengths,
        pred_speed_profile=pred_profile,
        real_speed_profile=real_profile,
        origin=origin,
        dest=dest,
        use_real_speed_for_planning=True,
        time_slot_duration_hours=args.time_slot_minutes / 60.0,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
    )
    if apply_filters and not route_pair_passes_filters(
        pred_result=pred_result,
        real_result=real_result,
        min_pred_hops=args.min_pred_hops,
        min_pred_distance_km=args.min_pred_distance_km,
        require_pred_real_path_diff=args.require_pred_real_path_diff,
        min_pred_real_regret_ratio=args.min_pred_real_regret_ratio,
    ):
        return None
    return {
        "window_idx": int(window_idx),
        "origin": int(origin),
        "dest": int(dest),
        "dispatch_count": int(dispatch_count),
        "dispatch_t": int(dispatch_t),
        "pred_profile": pred_profile,
        "real_profile": real_profile,
        "pred_result": pred_result,
        "real_result": real_result,
    }


def build_candidate_pool(
    *,
    candidate_windows: np.ndarray,
    edge_speeds: np.ndarray,
    provider: PersistenceSpeedProvider | StgatSpeedProvider,
    artifacts: dict,
    graph: CityGraph,
    rl_edge_index: np.ndarray,
    rl_edge_lengths: np.ndarray,
    rng: np.random.RandomState,
    args: argparse.Namespace,
) -> list[dict]:
    pool: list[dict] = []
    attempts = 0
    start = time.time()
    max_attempts = max(args.candidate_pool_size * args.candidate_pool_attempt_multiplier, args.candidate_pool_size)
    while len(pool) < args.candidate_pool_size and attempts < max_attempts:
        batch_meta: list[tuple[int, int, int, int, int]] = []
        batch_windows: list[int] = []
        for _ in range(args.candidate_build_batch_size):
            attempts += 1
            window_idx = int(rng.choice(candidate_windows))
            try:
                origin, dest, dispatch_count, dispatch_t = sample_dispatch_od(
                    window_idx=window_idx,
                    hist_len=args.hist_len,
                    artifacts=artifacts,
                    rng=rng,
                    dispatch_source=args.dispatch_source,
                )
            except ValueError:
                continue
            batch_windows.append(window_idx)
            batch_meta.append((window_idx, origin, dest, dispatch_count, dispatch_t))
            if attempts >= max_attempts:
                break
        if not batch_windows:
            continue
        pred_profiles, real_profiles = route_windows_to_profiles_batch(
            window_indices=batch_windows,
            hist_len=args.hist_len,
            pred_horizon=args.pred_horizon,
            edge_speeds=edge_speeds,
            provider=provider,
            artifacts=artifacts,
        )
        for meta, pred_profile, real_profile in zip(batch_meta, pred_profiles, real_profiles):
            window_idx, origin, dest, dispatch_count, dispatch_t = meta
            cand = make_route_candidate(
                window_idx=window_idx,
                origin=origin,
                dest=dest,
                dispatch_count=dispatch_count,
                dispatch_t=dispatch_t,
                pred_profile=pred_profile,
                real_profile=real_profile,
                graph=graph,
                rl_edge_index=rl_edge_index,
                rl_edge_lengths=rl_edge_lengths,
                args=args,
            )
            if cand is not None:
                pool.append(cand)
                if len(pool) >= args.candidate_pool_size:
                    break
        if len(pool) and (len(pool) % max(args.candidate_build_log_interval, 1) == 0):
            elapsed = time.time() - start
            print(
                f"candidate_pool={len(pool)}/{args.candidate_pool_size} "
                f"attempts={attempts} accept_rate={len(pool)/max(attempts,1)*100:.2f}% "
                f"elapsed={format_seconds(elapsed)}",
                flush=True,
            )
    if len(pool) < args.candidate_pool_size:
        raise RuntimeError(
            f"Only built {len(pool)} route candidates after {attempts} attempts. "
            "Relax filters or increase --candidate-pool-attempt-multiplier."
        )
    return pool


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    configure_runtime(device, args.seed)
    rng = np.random.RandomState(args.seed)

    artifacts = load_superzone_artifacts(data_dir, args.superzone_dir)
    edge_speeds = np.load(data_dir / "edge_speeds.npy", mmap_mode="r")
    if edge_speeds.shape[0] < edge_speeds.shape[1]:
        edge_speeds = edge_speeds.T
    time_meta = pd.read_csv(data_dir / "time_meta.csv")
    if args.max_time_steps > 0:
        max_t = min(int(args.max_time_steps), edge_speeds.shape[0], len(time_meta))
        edge_speeds = edge_speeds[:max_t]
        time_meta = time_meta.iloc[:max_t].reset_index(drop=True)
    time_meta["date"] = pd.to_datetime(time_meta["date"], errors="raise")
    if len(time_meta) != edge_speeds.shape[0]:
        raise ValueError(
            f"time_meta rows ({len(time_meta)}) must match edge_speeds time steps ({edge_speeds.shape[0]})"
        )

    split_indices = build_monthly_split_indices(time_meta, args.hist_len, args.pred_horizon)
    candidate_windows = np.asarray(split_indices[args.split], dtype=np.int32)
    if candidate_windows.size == 0:
        raise ValueError(f"split {args.split} has no valid windows")

    provider = build_provider(args, data_dir=data_dir, edge_speeds=edge_speeds, device=device)
    rl_edge_index = artifacts["rl_edge_index"].astype(np.int32)
    rl_edge_lengths = artifacts["rl_edge_lengths"].astype(np.float32)
    num_nodes = int(artifacts["region_demand"].shape[1])
    max_neighbors = infer_max_neighbors(rl_edge_index, num_nodes, minimum=args.max_neighbors)
    avg_speeds = np.maximum(artifacts["rl_edge_speeds_kmh"].astype(np.float32), 1.0)
    graph = create_graph_from_data(
        num_nodes,
        rl_edge_index,
        rl_edge_lengths,
        avg_speeds,
        avg_speeds,
        max_neighbors=max_neighbors,
    )
    env = RoutingEnv(
        graph,
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
        rho=args.rho,
        max_steps=args.max_steps,
        num_time_slots=args.pred_horizon,
        time_slot_duration_hours=args.time_slot_minutes / 60.0,
        use_real_speed=True,
        dynamic_edge_index=rl_edge_index,
        dynamic_pred_speeds=np.repeat(avg_speeds[:, None], args.pred_horizon, axis=1),
        dynamic_real_speeds=np.repeat(avg_speeds[:, None], args.pred_horizon, axis=1),
    )
    agent = DoubleDQNAgent(
        num_nodes=num_nodes,
        max_neighbors=max_neighbors,
        state_dim=env.state_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update=args.target_update,
        device=str(device),
    )

    config = vars(args).copy()
    config.update(
        {
            "pred_source_resolved": provider.name,
            "num_nodes": int(num_nodes),
            "num_edges": int(rl_edge_index.shape[0]),
            "state_dim": int(env.state_dim),
            "max_neighbors_resolved": int(max_neighbors),
            "split_window_count": int(candidate_windows.size),
            "device_resolved": str(device),
            "real_speed_source": str(data_dir / "edge_speeds.npy"),
            "dispatch_od_sampling": "count_weighted_from_greedy_dispatch",
            "candidate_pool_enabled": bool(args.candidate_pool_size > 0),
            "posthoc_fallback_note": (
                "PostHocFallback uses completed DDQN realized time/failure to choose Dijkstra-pred; "
                "treat it as a safety diagnostic, not as a deployable online baseline."
            ),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("=" * 88, flush=True)
    print("Real-data uncertain routing DDQN experiment", flush=True)
    print(
        f"split={args.split} windows={candidate_windows.size} pred={provider.name} "
        f"dispatch={args.dispatch_source} nodes={num_nodes} edges={rl_edge_index.shape[0]} "
        f"device={device}",
        flush=True,
    )
    print(
        "Decision information: predicted speeds and dispatch source only; "
        "reward/evaluation: realized held-out edge_speeds.",
        flush=True,
    )
    print(f"run_dir={run_dir}", flush=True)
    print("=" * 88, flush=True)

    rows: list[dict] = []
    window_rows: list[dict] = []
    candidate_pool: list[dict] = []
    if args.candidate_pool_size > 0:
        print(
            f"Building candidate pool: target={args.candidate_pool_size} "
            f"batch={args.candidate_build_batch_size}",
            flush=True,
        )
        candidate_pool = build_candidate_pool(
            candidate_windows=candidate_windows,
            edge_speeds=edge_speeds,
            provider=provider,
            artifacts=artifacts,
            graph=graph,
            rl_edge_index=rl_edge_index,
            rl_edge_lengths=rl_edge_lengths,
            rng=rng,
            args=args,
        )
        print(f"Candidate pool ready: {len(candidate_pool)} routes", flush=True)
    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        if candidate_pool:
            candidate_attempt = 1
            filter_matched = True
            cand = candidate_pool[int(rng.randint(0, len(candidate_pool)))]
            window_idx = int(cand["window_idx"])
            origin = int(cand["origin"])
            dest = int(cand["dest"])
            dispatch_count = int(cand["dispatch_count"])
            dispatch_t = int(cand["dispatch_t"])
            pred_profile = cand["pred_profile"]
            real_profile = cand["real_profile"]
            pred_result = cand["pred_result"]
            real_result = cand["real_result"]
            target_t = int(window_idx + args.hist_len)
        else:
            filter_matched = False
            candidate_attempt = 0
            for candidate_attempt in range(1, args.max_episode_resample + 1):
                window_idx = int(rng.choice(candidate_windows))
                for _ in range(args.max_dispatch_resample):
                    try:
                        origin, dest, dispatch_count, dispatch_t = sample_dispatch_od(
                            window_idx=window_idx,
                            hist_len=args.hist_len,
                            artifacts=artifacts,
                            rng=rng,
                            dispatch_source=args.dispatch_source,
                        )
                        break
                    except ValueError:
                        continue
                else:
                    origin = int(rng.randint(0, num_nodes))
                    dest = int(rng.randint(0, num_nodes - 1))
                    if dest >= origin:
                        dest += 1
                    dispatch_count = 1
                    dispatch_t = int(window_idx + args.hist_len - 1)

                target_t = int(window_idx + args.hist_len)
                pred_profile, real_profile = route_window_to_profiles(
                    window_idx=window_idx,
                    hist_len=args.hist_len,
                    pred_horizon=args.pred_horizon,
                    edge_speeds=edge_speeds,
                    provider=provider,
                    artifacts=artifacts,
                )
                cand = make_route_candidate(
                    window_idx=window_idx,
                    origin=origin,
                    dest=dest,
                    dispatch_count=dispatch_count,
                    dispatch_t=dispatch_t,
                    pred_profile=pred_profile,
                    real_profile=real_profile,
                    graph=graph,
                    rl_edge_index=rl_edge_index,
                    rl_edge_lengths=rl_edge_lengths,
                    args=args,
                )
                if cand is not None:
                    pred_result = cand["pred_result"]
                    real_result = cand["real_result"]
                    filter_matched = True
                    break
            if not filter_matched:
                cand = make_route_candidate(
                    window_idx=window_idx,
                    origin=origin,
                    dest=dest,
                    dispatch_count=dispatch_count,
                    dispatch_t=dispatch_t,
                    pred_profile=pred_profile,
                    real_profile=real_profile,
                    graph=graph,
                    rl_edge_index=rl_edge_index,
                    rl_edge_lengths=rl_edge_lengths,
                    args=args,
                    apply_filters=False,
                )
                if cand is None:
                    raise RuntimeError("Failed to build even an unfiltered route candidate.")
                pred_result = cand["pred_result"]
                real_result = cand["real_result"]

        env.dynamic_pred_speeds = pred_profile
        env.dynamic_real_speeds = real_profile
        ddqn_result = run_ddqn_episode(env, agent, origin=origin, dest=dest, train=True)

        fallback_used = (
            (not ddqn_result.reached)
            or (not math.isfinite(ddqn_result.travel_time))
            or (
                pred_result.reached
                and math.isfinite(pred_result.travel_time)
                and ddqn_result.travel_time > args.fallback_ratio * pred_result.travel_time
            )
        )
        posthoc_fallback = pred_result if fallback_used else ddqn_result
        pred_real_regret_ratio = (
            float(pred_result.travel_time / real_result.travel_time)
            if real_result.reached and real_result.travel_time > 0 and math.isfinite(real_result.travel_time)
            else float("nan")
        )

        meta_row = time_meta.iloc[target_t]
        row = {
            "episode": int(ep),
            "split": args.split,
            "window_idx": int(window_idx),
            "target_time_idx": int(target_t),
            "dispatch_time_idx": int(dispatch_t),
            "date": str(meta_row.get("date", "")),
            "slot": int(meta_row.get("slot", -1)),
            "hour": int(meta_row.get("hour", -1)),
            "origin": int(origin),
            "dest": int(dest),
            "dispatch_count": int(dispatch_count),
            "candidate_attempts": int(candidate_attempt),
            "route_filter_matched": int(filter_matched),
            "ddqn_reached": int(ddqn_result.reached),
            "ddqn_time": ddqn_result.travel_time,
            "ddqn_dist": ddqn_result.travel_dist,
            "ddqn_reward": ddqn_result.reward,
            "ddqn_steps": ddqn_result.steps,
            "dijkstra_pred_reached": int(pred_result.reached),
            "dijkstra_pred_time": pred_result.travel_time,
            "dijkstra_pred_dist": pred_result.travel_dist,
            "dijkstra_pred_hops": pred_result.steps,
            "dijkstra_real_reached": int(real_result.reached),
            "dijkstra_real_time": real_result.travel_time,
            "dijkstra_real_dist": real_result.travel_dist,
            "dijkstra_real_hops": real_result.steps,
            "dijkstra_pred_path_differs_real": int(pred_result.path != real_result.path),
            "pred_real_regret_ratio": pred_real_regret_ratio,
            "fallback_used": int(fallback_used),
            "posthoc_fallback_reached": int(posthoc_fallback.reached),
            "posthoc_fallback_time": posthoc_fallback.travel_time,
            "posthoc_fallback_dist": posthoc_fallback.travel_dist,
            "epsilon": float(agent.epsilon),
            "loss": "" if ddqn_result.last_loss is None else float(ddqn_result.last_loss),
        }
        rows.append(row)

        if ep % args.log_interval == 0 or ep == args.episodes:
            window = rows[-args.log_interval :]
            stats = summarize_window(window)
            elapsed = time.time() - start_time
            progress = ep / max(args.episodes, 1)
            eta = elapsed * (1.0 - progress) / max(progress, 1e-9)
            stats_row = {
                "episode": int(ep),
                "progress": float(progress),
                "elapsed_sec": float(elapsed),
                "eta_sec": float(eta),
                "epsilon": float(agent.epsilon),
                **stats,
            }
            window_rows.append(stats_row)
            write_csv(run_dir / "episode_metrics.csv", rows)
            write_csv(run_dir / "window_summary.csv", window_rows)
            (run_dir / "summary.json").write_text(json.dumps(stats_row, indent=2), encoding="utf-8")
            agent.save(run_dir / "ddqn_latest.pt")
            print(
                f"[{ep:>6d}/{args.episodes:<6d} {progress*100:6.2f}%] "
                f"ETA={format_seconds(eta)} elapsed={format_seconds(elapsed)} "
                f"eps={agent.epsilon:.3f} SR={stats['ddqn_sr']:5.1f}% "
                f"DDQNsucc={stats['ddqn_time_success']:.4f}h "
                f"DijPred={stats['dijkstra_pred_time']:.4f}h "
                f"DijReal={stats['dijkstra_real_time']:.4f}h "
                f"DDQN/PredSucc={stats['ddqn_over_pred_success']:.3f} "
                f"PostHoc/Pred={stats['posthoc_fallback_over_pred']:.3f} "
                f"fallback={stats['fallback_rate']:.1f}% "
                f"filter={stats['route_filter_match_rate']:.1f}% "
                f"pathdiff={stats['pred_real_path_diff_rate']:.1f}% "
                f"win={stats['ddqn_win_rate_success']:.1f}% "
                f"R={stats['avg_reward']:.2f} loss={stats['avg_loss']:.4f}",
                flush=True,
            )

    final_stats = summarize_window(rows)
    agent.save(run_dir / "ddqn_final.pt")
    report = {
        "config": config,
        "final_all_episode_metrics": final_stats,
        "last_window": window_rows[-1] if window_rows else {},
        "outputs": {
            "checkpoint": str(run_dir / "ddqn_final.pt"),
            "episode_metrics": str(run_dir / "episode_metrics.csv"),
            "window_summary": str(run_dir / "window_summary.csv"),
        },
    }
    (run_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 88, flush=True)
    print("Training finished", flush=True)
    print(json.dumps(final_stats, indent=2), flush=True)
    print(f"Saved: {run_dir / 'ddqn_final.pt'}", flush=True)
    print("=" * 88, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train/evaluate a DDQN router on NYC real-data forecast-realization mismatch. "
            "Predicted speeds are generated from persistence or STGAT; realized speeds come "
            "from held-out edge_speeds.npy."
        )
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--superzone-dir", type=str, default="data/superzones_k64")
    p.add_argument("--edge-length-source", type=str, default="osrm", choices=["osrm", "centroid"])
    p.add_argument("--run-dir", type=str, default="runs/real_data_uncertain_routing")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--max-time-steps", type=int, default=0)
    p.add_argument("--hist-len", type=int, default=14)
    p.add_argument("--pred-horizon", type=int, default=4)
    p.add_argument("--time-slot-minutes", type=float, default=15.0)
    p.add_argument("--pred-source", type=str, default="persistence", choices=["persistence", "stgat"])
    p.add_argument(
        "--stgat-ckpt-dir",
        type=str,
        default="runs/final_dc_v_export_20260514_134425/checkpoints/v_nyc",
    )
    p.add_argument(
        "--dispatch-source",
        type=str,
        default="persistence",
        choices=["persistence", "real_future_oracle"],
        help="persistence uses t-1 demand/supply; real_future_oracle uses target-time realized demand/supply and is not causal.",
    )
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-neighbors", type=int, default=10)
    p.add_argument("--max-dispatch-resample", type=int, default=25)
    p.add_argument(
        "--max-episode-resample",
        type=int,
        default=1,
        help="How many route candidates to try per episode when nontrivial-route filters are enabled.",
    )
    p.add_argument(
        "--min-pred-hops",
        type=int,
        default=0,
        help="Require Dijkstra-pred path to have at least this many hops; useful to exclude one-edge trivial OD.",
    )
    p.add_argument(
        "--min-pred-distance-km",
        type=float,
        default=0.0,
        help="Require Dijkstra-pred path distance to be at least this many km.",
    )
    p.add_argument(
        "--require-pred-real-path-diff",
        action="store_true",
        help="Resample until Dijkstra-pred and Dijkstra-real oracle choose different paths, if possible.",
    )
    p.add_argument(
        "--min-pred-real-regret-ratio",
        type=float,
        default=1.0,
        help="Require realized Dijkstra-pred time / realized real-oracle time to be at least this ratio.",
    )
    p.add_argument(
        "--candidate-pool-size",
        type=int,
        default=0,
        help="Prebuild this many filtered route candidates, then train by sampling from the pool.",
    )
    p.add_argument(
        "--candidate-build-batch-size",
        type=int,
        default=32,
        help="Batch size for STGAT/persistence prediction while building the route candidate pool.",
    )
    p.add_argument(
        "--candidate-pool-attempt-multiplier",
        type=int,
        default=200,
        help="Maximum candidate-build attempts equals pool size times this multiplier.",
    )
    p.add_argument(
        "--candidate-build-log-interval",
        type=int,
        default=100,
        help="Print candidate pool progress each time this many accepted candidates are reached.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=2500)
    p.add_argument("--buffer-capacity", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--target-update", type=int, default=100)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--rho", type=float, default=50.0)
    p.add_argument("--fallback-ratio", type=float, default=1.2)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
