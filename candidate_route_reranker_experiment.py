from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from graph_env import create_graph_from_data, infer_max_neighbors
from real_data_uncertain_routing_experiment import (
    RouteResult,
    build_provider,
    configure_runtime,
    dijkstra_result,
    format_seconds,
    mean_finite,
    resolve_device,
    route_windows_to_profiles_batch,
    sample_dispatch_od,
    simulate_path_under_real_speeds,
)
from replay_buffer import ReplayBuffer
from superzone_graph import load_superzone_artifacts
from train_predictor import build_monthly_split_indices


@dataclass
class PathCandidate:
    path: list[int]
    edge_ids: list[int]
    pred_result: RouteResult
    real_result: RouteResult
    static_pred_cost: float
    min_pred_speed: float
    mean_pred_speed: float
    pred_time_last: float
    pred_time_std: float
    pred_speed_std: float
    low_speed_frac: float
    pred_time_trend_ratio: float
    pattern_sim_mean: float
    pattern_sim_max: float
    pattern_ref_speed_mean: float
    pattern_ref_speed_std: float
    pattern_low_speed_frac: float
    pattern_attention_entropy: float
    pattern_route_ref_gap: float
    pattern_topk_edges: list[int]


@dataclass
class RouteChoiceCandidate:
    window_idx: int
    target_t: int
    dispatch_t: int
    origin: int
    dest: int
    dispatch_count: int
    pred_profile: np.ndarray
    real_profile: np.ndarray
    candidates: list[PathCandidate]
    dijkstra_pred: PathCandidate
    candidate_oracle: PathCandidate
    dijkstra_real: RouteResult


class CandidateQNetwork(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        action_dim: int,
        state_dim: int,
        embed_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_nodes, embed_dim)
        input_dim = 2 * embed_dim + (state_dim - 2)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        origin = state[:, 0].long()
        dest = state[:, 1].long()
        cont = state[:, 2:]
        return self.net(torch.cat([self.embed(origin), self.embed(dest), cont], dim=1))


class CandidateDoubleDQNAgent:
    def __init__(
        self,
        *,
        num_nodes: int,
        action_dim: int,
        state_dim: int,
        embed_dim: int,
        hidden_dim: int,
        lr: float,
        gamma: float,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: int,
        buffer_capacity: int,
        batch_size: int,
        target_update: int,
        device: str,
    ) -> None:
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_update = int(target_update)
        self.device = torch.device(device)
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = int(epsilon_decay)
        self.learn_step_count = 0

        self.online_net = CandidateQNetwork(
            num_nodes=num_nodes,
            action_dim=action_dim,
            state_dim=state_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_net = CandidateQNetwork(
            num_nodes=num_nodes,
            action_dim=action_dim,
            state_dim=state_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, state: np.ndarray, action_mask: np.ndarray, *, greedy: bool = False) -> int:
        valid = np.flatnonzero(action_mask > 0)
        if valid.size == 0:
            return 0
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.choice(valid))
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online_net(s).squeeze(0)
            mask = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
            q[mask == 0] = -float("inf")
            return int(q.argmax().item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done, action_mask, next_action_mask)

    def learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size, self.device)
        q_online = self.online_net(batch.states)
        q_sa = q_online.gather(1, batch.actions)
        with torch.no_grad():
            has_next = batch.next_action_masks.sum(dim=1, keepdim=True) > 0
            q_next_online = self.online_net(batch.next_states)
            q_next_online[batch.next_action_masks == 0] = -float("inf")
            best_actions = q_next_online.argmax(dim=1, keepdim=True)
            q_next_target = self.target_net(batch.next_states)
            q_next_val = q_next_target.gather(1, best_actions)
            q_next_val = torch.where(has_next, q_next_val, torch.zeros_like(q_next_val))
            target = batch.rewards + self.gamma * q_next_val * (1.0 - batch.dones)

        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end) * self.learn_step_count / max(self.epsilon_decay, 1),
        )
        self.learn_step_count += 1
        if self.learn_step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return float(loss.item())

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_step_count": self.learn_step_count,
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
            },
            path,
        )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_adj(edge_index: np.ndarray, weights: np.ndarray, num_nodes: int) -> list[list[tuple[int, int, float]]]:
    adj: list[list[tuple[int, int, float]]] = [[] for _ in range(num_nodes)]
    for eid, (src, dst) in enumerate(edge_index.astype(np.int32)):
        adj[int(src)].append((int(dst), int(eid), float(weights[eid])))
    return adj


def shortest_path_with_bans(
    *,
    adj: list[list[tuple[int, int, float]]],
    src: int,
    dst: int,
    banned_edges: set[tuple[int, int]],
    banned_nodes: set[int],
) -> tuple[list[int], list[int], float] | None:
    if src in banned_nodes:
        return None
    dist: dict[int, float] = {src: 0.0}
    prev: dict[int, tuple[int, int]] = {}
    pq: list[tuple[float, int]] = [(0.0, src)]
    while pq:
        cost, node = heapq.heappop(pq)
        if cost > dist.get(node, float("inf")):
            continue
        if node == dst:
            break
        for nb, eid, weight in adj[node]:
            if nb in banned_nodes and nb != dst:
                continue
            if (node, nb) in banned_edges:
                continue
            next_cost = cost + weight
            if next_cost < dist.get(nb, float("inf")):
                dist[nb] = next_cost
                prev[nb] = (node, eid)
                heapq.heappush(pq, (next_cost, nb))
    if dst not in dist:
        return None

    nodes = [dst]
    edge_ids: list[int] = []
    cur = dst
    while cur != src:
        parent, eid = prev[cur]
        edge_ids.append(int(eid))
        nodes.append(parent)
        cur = parent
    nodes.reverse()
    edge_ids.reverse()
    return nodes, edge_ids, float(dist[dst])


def yen_k_shortest_paths(
    *,
    edge_index: np.ndarray,
    weights: np.ndarray,
    num_nodes: int,
    src: int,
    dst: int,
    k: int,
) -> list[tuple[list[int], list[int], float]]:
    adj = build_adj(edge_index, weights, num_nodes)
    first = shortest_path_with_bans(
        adj=adj,
        src=src,
        dst=dst,
        banned_edges=set(),
        banned_nodes=set(),
    )
    if first is None:
        return []

    accepted: list[tuple[list[int], list[int], float]] = [first]
    accepted_keys = {tuple(first[0])}
    heap: list[tuple[float, int, list[int], list[int]]] = []
    heap_keys: set[tuple[int, ...]] = set()
    counter = 0

    for _ in range(1, k):
        base_nodes, base_edges, _ = accepted[-1]
        for spur_idx in range(len(base_nodes) - 1):
            spur_node = base_nodes[spur_idx]
            root_nodes = base_nodes[: spur_idx + 1]
            root_edges = base_edges[:spur_idx]
            banned_edges: set[tuple[int, int]] = set()
            for path_nodes, _, _ in accepted:
                if len(path_nodes) > spur_idx + 1 and path_nodes[: spur_idx + 1] == root_nodes:
                    banned_edges.add((path_nodes[spur_idx], path_nodes[spur_idx + 1]))
            banned_nodes = set(root_nodes[:-1])
            spur = shortest_path_with_bans(
                adj=adj,
                src=spur_node,
                dst=dst,
                banned_edges=banned_edges,
                banned_nodes=banned_nodes,
            )
            if spur is None:
                continue
            spur_nodes, spur_edges, _ = spur
            total_nodes = root_nodes[:-1] + spur_nodes
            total_edges = root_edges + spur_edges
            key = tuple(total_nodes)
            if key in accepted_keys or key in heap_keys:
                continue
            total_cost = float(np.asarray(weights, dtype=np.float64)[total_edges].sum())
            heapq.heappush(heap, (total_cost, counter, total_nodes, total_edges))
            heap_keys.add(key)
            counter += 1

        while heap:
            cost, _, nodes, edge_ids = heapq.heappop(heap)
            key = tuple(nodes)
            heap_keys.discard(key)
            if key not in accepted_keys:
                accepted.append((nodes, edge_ids, float(cost)))
                accepted_keys.add(key)
                break
        else:
            break
    return accepted[:k]


def pattern_topk_attention_summary(
    *,
    pred_profile: np.ndarray,
    route_edge_ids: list[int],
    pattern_topk: int,
    temperature: float,
) -> tuple[float, float, float, float, float, float, float, list[int]]:
    """Find non-contiguous edges with speed-pattern attention similar to a route."""
    if not route_edge_ids:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []
    edge_arr = np.asarray(route_edge_ids, dtype=np.int64)
    speeds = np.maximum(np.asarray(pred_profile, dtype=np.float32), 1e-5)
    route_pattern = speeds[edge_arr].mean(axis=0)

    def normalize_rows(x: np.ndarray) -> np.ndarray:
        centered = x - x.mean(axis=1, keepdims=True)
        scale = x.std(axis=1, keepdims=True) + 1e-6
        return centered / scale

    q = normalize_rows(route_pattern[None, :])[0]
    keys = normalize_rows(speeds)
    pattern_sim = keys @ q / max(float(speeds.shape[1]), 1.0)

    route_level = float(route_pattern.mean())
    edge_level = speeds.mean(axis=1)
    level_sim = 1.0 - np.abs(edge_level - route_level) / 130.0
    scores = 0.75 * pattern_sim + 0.25 * level_sim

    valid = np.ones(scores.shape[0], dtype=bool)
    valid[edge_arr] = False
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, []
    k = min(max(int(pattern_topk), 1), int(valid_idx.size))
    ranked_local = np.argpartition(scores[valid_idx], -k)[-k:]
    top_edges = valid_idx[ranked_local]
    top_edges = top_edges[np.argsort(scores[top_edges])[::-1]]

    top_scores = scores[top_edges].astype(np.float64)
    top_speeds = speeds[top_edges].astype(np.float64)
    temp = max(float(temperature), 1e-4)
    logits = top_scores / temp
    logits = logits - float(np.max(logits))
    probs = np.exp(logits)
    probs = probs / max(float(probs.sum()), 1e-12)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12))) / max(math.log(len(probs)), 1e-12)
    weighted_speed = float(np.sum(probs[:, None] * top_speeds) / max(speeds.shape[1], 1))
    route_ref_gap = (route_level - weighted_speed) / 130.0

    return (
        float(np.mean(top_scores)),
        float(np.max(top_scores)),
        weighted_speed,
        float(np.std(top_speeds)),
        float(np.mean(top_speeds < 20.0)),
        entropy,
        float(route_ref_gap),
        [int(x) for x in top_edges.tolist()],
    )


def evaluate_candidate_path(
    *,
    path: list[int],
    edge_ids: list[int],
    static_pred_cost: float,
    pred_profile: np.ndarray,
    real_profile: np.ndarray,
    edge_index: np.ndarray,
    edge_lengths: np.ndarray,
    args: argparse.Namespace,
) -> PathCandidate:
    edge_id = {(int(src), int(dst)): int(i) for i, (src, dst) in enumerate(edge_index)}
    pred_result = simulate_path_under_real_speeds(
        path,
        edge_id=edge_id,
        edge_lengths=edge_lengths,
        real_speed_profile=pred_profile,
        start_slot=0,
        time_slot_duration_hours=args.time_slot_minutes / 60.0,
        num_time_slots=pred_profile.shape[1],
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
    )
    real_result = simulate_path_under_real_speeds(
        path,
        edge_id=edge_id,
        edge_lengths=edge_lengths,
        real_speed_profile=real_profile,
        start_slot=0,
        time_slot_duration_hours=args.time_slot_minutes / 60.0,
        num_time_slots=real_profile.shape[1],
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
    )
    speeds = pred_profile[np.asarray(edge_ids, dtype=np.int64), 0] if edge_ids else np.asarray([0.0])
    if edge_ids:
        edge_arr = np.asarray(edge_ids, dtype=np.int64)
        lengths = edge_lengths[edge_arr].astype(np.float64)
        speed_profile = np.maximum(pred_profile[edge_arr, :].astype(np.float64), 1e-5)
        slot_times = (lengths[:, None] / speed_profile).sum(axis=0)
        pred_time_last = float(slot_times[-1])
        pred_time_std = float(np.std(slot_times))
        pred_speed_std = float(np.std(speed_profile))
        low_speed_frac = float(np.mean(speed_profile < 20.0))
        pred_time_trend_ratio = float((slot_times[-1] - slot_times[0]) / max(slot_times[0], 1e-6))
        (
            pattern_sim_mean,
            pattern_sim_max,
            pattern_ref_speed_mean,
            pattern_ref_speed_std,
            pattern_low_speed_frac,
            pattern_attention_entropy,
            pattern_route_ref_gap,
            pattern_topk_edges,
        ) = pattern_topk_attention_summary(
            pred_profile=pred_profile,
            route_edge_ids=edge_ids,
            pattern_topk=int(getattr(args, "pattern_topk", 16)),
            temperature=float(getattr(args, "pattern_attention_temperature", 0.2)),
        )
    else:
        pred_time_last = 0.0
        pred_time_std = 0.0
        pred_speed_std = 0.0
        low_speed_frac = 0.0
        pred_time_trend_ratio = 0.0
        pattern_sim_mean = 0.0
        pattern_sim_max = 0.0
        pattern_ref_speed_mean = 0.0
        pattern_ref_speed_std = 0.0
        pattern_low_speed_frac = 0.0
        pattern_attention_entropy = 0.0
        pattern_route_ref_gap = 0.0
        pattern_topk_edges = []
    return PathCandidate(
        path=[int(x) for x in path],
        edge_ids=[int(x) for x in edge_ids],
        pred_result=pred_result,
        real_result=real_result,
        static_pred_cost=float(static_pred_cost),
        min_pred_speed=float(np.min(speeds)),
        mean_pred_speed=float(np.mean(speeds)),
        pred_time_last=pred_time_last,
        pred_time_std=pred_time_std,
        pred_speed_std=pred_speed_std,
        low_speed_frac=low_speed_frac,
        pred_time_trend_ratio=pred_time_trend_ratio,
        pattern_sim_mean=pattern_sim_mean,
        pattern_sim_max=pattern_sim_max,
        pattern_ref_speed_mean=pattern_ref_speed_mean,
        pattern_ref_speed_std=pattern_ref_speed_std,
        pattern_low_speed_frac=pattern_low_speed_frac,
        pattern_attention_entropy=pattern_attention_entropy,
        pattern_route_ref_gap=pattern_route_ref_gap,
        pattern_topk_edges=pattern_topk_edges,
    )


def make_route_choice_candidate(
    *,
    window_idx: int,
    target_t: int,
    dispatch_t: int,
    origin: int,
    dest: int,
    dispatch_count: int,
    pred_profile: np.ndarray,
    real_profile: np.ndarray,
    graph,
    edge_index: np.ndarray,
    edge_lengths: np.ndarray,
    num_nodes: int,
    args: argparse.Namespace,
) -> RouteChoiceCandidate | None:
    weights = edge_lengths.astype(np.float64) / np.maximum(pred_profile[:, 0].astype(np.float64), 1e-5)
    paths = yen_k_shortest_paths(
        edge_index=edge_index,
        weights=weights,
        num_nodes=num_nodes,
        src=origin,
        dst=dest,
        k=args.k_routes,
    )
    if len(paths) < args.min_unique_routes:
        return None

    candidates = [
        evaluate_candidate_path(
            path=nodes,
            edge_ids=edge_ids,
            static_pred_cost=cost,
            pred_profile=pred_profile,
            real_profile=real_profile,
            edge_index=edge_index,
            edge_lengths=edge_lengths,
            args=args,
        )
        for nodes, edge_ids, cost in paths
    ]
    candidates = [c for c in candidates if c.real_result.reached and c.pred_result.reached]
    if len(candidates) < args.min_unique_routes:
        return None

    dijkstra_pred = candidates[0]
    if dijkstra_pred.real_result.steps < args.min_pred_hops:
        return None
    if dijkstra_pred.real_result.travel_dist < args.min_pred_distance_km:
        return None

    candidate_oracle = min(candidates, key=lambda c: c.real_result.travel_time)
    if args.require_candidate_oracle_diff and candidate_oracle.path == dijkstra_pred.path:
        return None
    if args.min_candidate_oracle_improvement_ratio > 1.0:
        baseline = dijkstra_pred.real_result.travel_time
        oracle = candidate_oracle.real_result.travel_time
        if not (math.isfinite(baseline) and math.isfinite(oracle) and oracle > 0.0):
            return None
        if baseline / oracle < args.min_candidate_oracle_improvement_ratio:
            return None

    dijkstra_real = dijkstra_result(
        graph,
        edge_index=edge_index,
        edge_lengths=edge_lengths,
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
    return RouteChoiceCandidate(
        window_idx=int(window_idx),
        target_t=int(target_t),
        dispatch_t=int(dispatch_t),
        origin=int(origin),
        dest=int(dest),
        dispatch_count=int(dispatch_count),
        pred_profile=pred_profile,
        real_profile=real_profile,
        candidates=candidates[: args.k_routes],
        dijkstra_pred=dijkstra_pred,
        candidate_oracle=candidate_oracle,
        dijkstra_real=dijkstra_real,
    )


def build_state_and_mask(
    cand: RouteChoiceCandidate,
    *,
    meta_row: pd.Series,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    k = int(args.k_routes)
    per_route_dim = route_feature_dim(args)
    feats = np.zeros((k, per_route_dim), dtype=np.float32)
    mask = np.zeros(k, dtype=np.float32)
    baseline_pred_time = max(float(cand.dijkstra_pred.pred_result.travel_time), 1e-6)
    for idx, route in enumerate(cand.candidates[:k]):
        pred_time = float(route.pred_result.travel_time)
        pred_dist = float(route.pred_result.travel_dist)
        hops = float(route.pred_result.steps)
        feats[idx, 0] = pred_time / args.feature_time_norm
        feats[idx, 1] = pred_dist / args.feature_distance_norm
        feats[idx, 2] = hops / args.feature_hop_norm
        feats[idx, 3] = route.min_pred_speed / args.feature_speed_norm
        feats[idx, 4] = route.mean_pred_speed / args.feature_speed_norm
        feats[idx, 5] = float(idx) / max(k - 1, 1)
        feats[idx, 6] = pred_time / baseline_pred_time
        if args.feature_set in {"uncertainty", "pattern_topk"}:
            feats[idx, 7] = route.pred_time_last / args.feature_time_norm
            feats[idx, 8] = route.pred_time_std / args.feature_time_norm
            feats[idx, 9] = route.pred_speed_std / args.feature_speed_norm
            feats[idx, 10] = route.low_speed_frac
            feats[idx, 11] = float(np.clip(route.pred_time_trend_ratio, -1.0, 1.0))
        if args.feature_set == "pattern_topk":
            feats[idx, 12] = route.pattern_sim_mean
            feats[idx, 13] = route.pattern_sim_max
            feats[idx, 14] = route.pattern_ref_speed_mean / args.feature_speed_norm
            feats[idx, 15] = route.pattern_ref_speed_std / args.feature_speed_norm
            feats[idx, 16] = route.pattern_low_speed_frac
            feats[idx, 17] = route.pattern_attention_entropy
            feats[idx, 18] = float(np.clip(route.pattern_route_ref_gap, -1.0, 1.0))
        mask[idx] = 1.0

    hour = float(meta_row.get("hour", 0.0)) / 23.0
    slots_per_day = max(int(round((24 * 60) / args.time_slot_minutes)), 1)
    slot = float(meta_row.get("slot", 0.0)) / max(slots_per_day - 1, 1)
    count = math.log1p(max(float(cand.dispatch_count), 0.0)) / args.feature_log_count_norm
    cont = np.concatenate(
        [
            np.asarray([hour, slot, count], dtype=np.float32),
            feats.reshape(-1),
        ]
    )
    state = np.concatenate(
        [
            np.asarray([float(cand.origin), float(cand.dest)], dtype=np.float32),
            cont.astype(np.float32),
        ]
    )
    return state.astype(np.float32), mask


def route_feature_dim(args: argparse.Namespace) -> int:
    if getattr(args, "feature_set", "base") == "pattern_topk":
        return 19
    if getattr(args, "feature_set", "base") == "uncertainty":
        return 12
    return 7


def state_dim_for_args(args: argparse.Namespace) -> int:
    return 2 + 3 + int(args.k_routes) * route_feature_dim(args)


def choose_reward(
    cand: RouteChoiceCandidate,
    chosen: PathCandidate,
    args: argparse.Namespace,
) -> float:
    baseline = float(cand.dijkstra_pred.real_result.travel_time)
    oracle = float(cand.candidate_oracle.real_result.travel_time)
    chosen_time = float(chosen.real_result.travel_time)
    denom = max(baseline, 1e-6)
    improvement = (baseline - chosen_time) / denom
    oracle_regret = (chosen_time - oracle) / denom
    mode = getattr(args, "reward_mode", "shaped")
    if mode == "direct_regret":
        reward = args.reward_scale * improvement
    elif mode == "time_only":
        reward = -args.reward_scale * chosen_time / max(baseline, 1e-6)
    else:
        reward = args.reward_scale * improvement - args.oracle_regret_weight * oracle_regret
        if chosen_time < baseline:
            reward += args.win_bonus
        if chosen.path == cand.candidate_oracle.path:
            reward += args.oracle_bonus
    return float(reward)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {}
    ddqn_times = [r["ddqn_time"] for r in rows]
    pred_times = [r["dijkstra_pred_time"] for r in rows]
    oracle_times = [r["candidate_oracle_time"] for r in rows]
    return {
        "ddqn_time": mean_finite(ddqn_times),
        "dijkstra_pred_time": mean_finite(pred_times),
        "candidate_oracle_time": mean_finite(oracle_times),
        "dijkstra_real_time": mean_finite([r["dijkstra_real_time"] for r in rows]),
        "ddqn_over_pred": mean_finite(
            [
                r["ddqn_time"] / r["dijkstra_pred_time"]
                for r in rows
                if r["dijkstra_pred_time"] > 0 and math.isfinite(r["dijkstra_pred_time"])
            ]
        ),
        "ddqn_over_candidate_oracle": mean_finite(
            [
                r["ddqn_time"] / r["candidate_oracle_time"]
                for r in rows
                if r["candidate_oracle_time"] > 0 and math.isfinite(r["candidate_oracle_time"])
            ]
        ),
        "candidate_oracle_over_pred": mean_finite(
            [
                r["candidate_oracle_time"] / r["dijkstra_pred_time"]
                for r in rows
                if r["dijkstra_pred_time"] > 0 and math.isfinite(r["dijkstra_pred_time"])
            ]
        ),
        "ddqn_win_rate": float(np.mean([float(r["ddqn_win_pred"]) for r in rows]) * 100.0),
        "oracle_pick_rate": float(np.mean([float(r["ddqn_pick_oracle"]) for r in rows]) * 100.0),
        "pred_already_oracle_rate": float(np.mean([float(r["pred_is_candidate_oracle"]) for r in rows]) * 100.0),
        "opportunity_rate": float(np.mean([float(r["has_candidate_opportunity"]) for r in rows]) * 100.0),
        "avg_reward": float(np.mean([float(r["reward"]) for r in rows])),
        "avg_loss": mean_finite([r["loss"] for r in rows if r["loss"] != ""]),
        "avg_action": float(np.mean([float(r["action"]) for r in rows])),
        "avg_unique_routes": float(np.mean([float(r["num_routes"]) for r in rows])),
    }


def build_candidate_pool(
    *,
    candidate_windows: np.ndarray,
    edge_speeds: np.ndarray,
    provider,
    artifacts: dict,
    graph,
    edge_index: np.ndarray,
    edge_lengths: np.ndarray,
    num_nodes: int,
    rng: np.random.RandomState,
    args: argparse.Namespace,
) -> list[RouteChoiceCandidate]:
    pool: list[RouteChoiceCandidate] = []
    attempts = 0
    next_log = max(args.candidate_build_log_interval, 1)
    start = time.time()
    progress_path = Path(args.run_dir) / "candidate_pool_progress.json"
    max_attempts = max(args.candidate_pool_size * args.candidate_pool_attempt_multiplier, args.candidate_pool_size)
    while len(pool) < args.candidate_pool_size and attempts < max_attempts:
        batch_windows: list[int] = []
        batch_meta: list[tuple[int, int, int, int, int, int]] = []
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
                if attempts >= max_attempts:
                    break
                continue
            target_t = int(window_idx + args.hist_len)
            batch_windows.append(window_idx)
            batch_meta.append((window_idx, target_t, dispatch_t, origin, dest, dispatch_count))
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
            window_idx, target_t, dispatch_t, origin, dest, dispatch_count = meta
            cand = make_route_choice_candidate(
                window_idx=window_idx,
                target_t=target_t,
                dispatch_t=dispatch_t,
                origin=origin,
                dest=dest,
                dispatch_count=dispatch_count,
                pred_profile=pred_profile,
                real_profile=real_profile,
                graph=graph,
                edge_index=edge_index,
                edge_lengths=edge_lengths,
                num_nodes=num_nodes,
                args=args,
            )
            if cand is not None:
                pool.append(cand)
                if len(pool) >= next_log:
                    elapsed = time.time() - start
                    progress = len(pool) / max(int(args.candidate_pool_size), 1)
                    eta = elapsed * (1.0 - progress) / max(progress, 1e-9)
                    progress_path.write_text(
                        json.dumps(
                            {
                                "accepted": int(len(pool)),
                                "target": int(args.candidate_pool_size),
                                "attempts": int(attempts),
                                "accept_rate": float(len(pool) / max(attempts, 1)),
                                "elapsed_sec": float(elapsed),
                                "eta_sec": float(eta),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    print(
                        f"candidate_pool={len(pool)}/{args.candidate_pool_size} "
                        f"attempts={attempts} accept_rate={len(pool)/max(attempts,1)*100:.2f}% "
                        f"elapsed={format_seconds(elapsed)} stage_ETA={format_seconds(eta)}",
                        flush=True,
                    )
                    next_log += max(args.candidate_build_log_interval, 1)
                if len(pool) >= args.candidate_pool_size:
                    break
    if len(pool) < args.candidate_pool_size:
        raise RuntimeError(
            f"Only built {len(pool)} route-choice candidates after {attempts} attempts. "
            "Relax filters or increase --candidate-pool-attempt-multiplier."
        )
    return pool


def row_from_choice(
    *,
    episode: int,
    cand: RouteChoiceCandidate,
    action: int,
    reward: float,
    loss: float | None,
    epsilon: float,
    meta_row: pd.Series,
) -> dict:
    chosen = cand.candidates[action]
    baseline = cand.dijkstra_pred.real_result.travel_time
    oracle = cand.candidate_oracle.real_result.travel_time
    pred_is_oracle = int(cand.dijkstra_pred.path == cand.candidate_oracle.path)
    return {
        "episode": int(episode),
        "window_idx": int(cand.window_idx),
        "target_time_idx": int(cand.target_t),
        "dispatch_time_idx": int(cand.dispatch_t),
        "date": str(meta_row.get("date", "")),
        "slot": int(meta_row.get("slot", -1)),
        "hour": int(meta_row.get("hour", -1)),
        "origin": int(cand.origin),
        "dest": int(cand.dest),
        "dispatch_count": int(cand.dispatch_count),
        "num_routes": int(len(cand.candidates)),
        "action": int(action),
        "ddqn_time": float(chosen.real_result.travel_time),
        "ddqn_pred_time": float(chosen.pred_result.travel_time),
        "ddqn_dist": float(chosen.real_result.travel_dist),
        "ddqn_hops": int(chosen.real_result.steps),
        "dijkstra_pred_time": float(baseline),
        "dijkstra_pred_dist": float(cand.dijkstra_pred.real_result.travel_dist),
        "dijkstra_pred_hops": int(cand.dijkstra_pred.real_result.steps),
        "candidate_oracle_time": float(oracle),
        "candidate_oracle_action": int(
            next(i for i, route in enumerate(cand.candidates) if route.path == cand.candidate_oracle.path)
        ),
        "candidate_oracle_dist": float(cand.candidate_oracle.real_result.travel_dist),
        "dijkstra_real_time": float(cand.dijkstra_real.travel_time),
        "dijkstra_real_hops": int(cand.dijkstra_real.steps),
        "ddqn_win_pred": int(chosen.real_result.travel_time < baseline),
        "ddqn_pick_oracle": int(chosen.path == cand.candidate_oracle.path),
        "pred_is_candidate_oracle": pred_is_oracle,
        "has_candidate_opportunity": int(oracle < baseline),
        "candidate_oracle_improvement_ratio": float(baseline / oracle) if oracle > 0 else float("nan"),
        "chosen_pattern_sim_mean": float(chosen.pattern_sim_mean),
        "chosen_pattern_sim_max": float(chosen.pattern_sim_max),
        "chosen_pattern_ref_speed_mean": float(chosen.pattern_ref_speed_mean),
        "chosen_pattern_low_speed_frac": float(chosen.pattern_low_speed_frac),
        "chosen_pattern_attention_entropy": float(chosen.pattern_attention_entropy),
        "chosen_pattern_topk_edges": ";".join(str(x) for x in chosen.pattern_topk_edges[:10]),
        "reward": float(reward),
        "epsilon": float(epsilon),
        "loss": "" if loss is None else float(loss),
    }


def greedy_evaluate(
    *,
    agent: CandidateDoubleDQNAgent,
    pool: list[RouteChoiceCandidate],
    time_meta: pd.DataFrame,
    args: argparse.Namespace,
    limit: int,
) -> dict:
    rows: list[dict] = []
    count = len(pool) if limit <= 0 else min(limit, len(pool))
    for idx, cand in enumerate(pool[:count], start=1):
        state, mask = build_state_and_mask(cand, meta_row=time_meta.iloc[cand.target_t], args=args)
        action = agent.select_action(state, mask, greedy=True)
        reward = choose_reward(cand, cand.candidates[action], args)
        rows.append(
            row_from_choice(
                episode=idx,
                cand=cand,
                action=action,
                reward=reward,
                loss=None,
                epsilon=agent.epsilon,
                meta_row=time_meta.iloc[cand.target_t],
            )
        )
    return summarize(rows)


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

    split_indices = build_monthly_split_indices(time_meta, args.hist_len, args.pred_horizon)
    candidate_windows = np.asarray(split_indices[args.split], dtype=np.int32)
    if candidate_windows.size == 0:
        raise ValueError(f"split {args.split} has no valid windows")

    provider = build_provider(args, data_dir=data_dir, edge_speeds=edge_speeds, device=device)
    edge_index = artifacts["rl_edge_index"].astype(np.int32)
    edge_lengths = artifacts["rl_edge_lengths"].astype(np.float32)
    num_nodes = int(artifacts["region_demand"].shape[1])
    max_neighbors = infer_max_neighbors(edge_index, num_nodes, minimum=args.max_neighbors)
    avg_speeds = np.maximum(artifacts["rl_edge_speeds_kmh"].astype(np.float32), 1.0)
    graph = create_graph_from_data(
        num_nodes,
        edge_index,
        edge_lengths,
        avg_speeds,
        avg_speeds,
        max_neighbors=max_neighbors,
    )

    state_dim = state_dim_for_args(args)
    agent = CandidateDoubleDQNAgent(
        num_nodes=num_nodes,
        action_dim=args.k_routes,
        state_dim=state_dim,
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
            "num_edges": int(edge_index.shape[0]),
            "state_dim": int(state_dim),
            "decision_type": "DDQN reranks K predicted-speed candidate routes",
            "reward_note": "Reward uses realized future edge_speeds; state uses predicted route features only.",
            "candidate_oracle_note": "CandidateOracle is the best realized route among generated K candidates, not deployable.",
            "device_resolved": str(device),
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("=" * 96, flush=True)
    print("Candidate-route DDQN reranker experiment", flush=True)
    print(
        f"split={args.split} windows={candidate_windows.size} pred={provider.name} "
        f"dispatch={args.dispatch_source} K={args.k_routes} nodes={num_nodes} edges={edge_index.shape[0]} "
        f"device={device}",
        flush=True,
    )
    print(
        "Action: choose one route from K predicted-speed candidates. "
        "Reward/evaluation: realized held-out edge_speeds.",
        flush=True,
    )
    print(f"run_dir={run_dir}", flush=True)
    print("=" * 96, flush=True)

    print(
        f"Building route-choice candidate pool: target={args.candidate_pool_size} "
        f"batch={args.candidate_build_batch_size}",
        flush=True,
    )
    pool = build_candidate_pool(
        candidate_windows=candidate_windows,
        edge_speeds=edge_speeds,
        provider=provider,
        artifacts=artifacts,
        graph=graph,
        edge_index=edge_index,
        edge_lengths=edge_lengths,
        num_nodes=num_nodes,
        rng=rng,
        args=args,
    )
    print(f"Candidate pool ready: {len(pool)} route-choice states", flush=True)
    normal_pool: list[RouteChoiceCandidate] = []
    if args.normal_pool_size > 0:
        import copy

        normal_args = copy.copy(args)
        normal_args.candidate_pool_size = int(args.normal_pool_size)
        normal_args.require_candidate_oracle_diff = False
        normal_args.min_candidate_oracle_improvement_ratio = 1.0
        print(
            f"Building normal/non-opportunity candidate pool: target={normal_args.candidate_pool_size}",
            flush=True,
        )
        normal_pool = build_candidate_pool(
            candidate_windows=candidate_windows,
            edge_speeds=edge_speeds,
            provider=provider,
            artifacts=artifacts,
            graph=graph,
            edge_index=edge_index,
            edge_lengths=edge_lengths,
            num_nodes=num_nodes,
            rng=rng,
            args=normal_args,
        )
        print(f"Normal candidate pool ready: {len(normal_pool)} route-choice states", flush=True)

    rows: list[dict] = []
    window_rows: list[dict] = []
    start_time = time.time()
    zero_next_mask = np.zeros(args.k_routes, dtype=np.float32)

    for ep in range(1, args.episodes + 1):
        if normal_pool and rng.random_sample() > float(args.opportunity_sample_ratio):
            cand = normal_pool[int(rng.randint(0, len(normal_pool)))]
        else:
            cand = pool[int(rng.randint(0, len(pool)))]
        meta_row = time_meta.iloc[cand.target_t]
        state, mask = build_state_and_mask(cand, meta_row=meta_row, args=args)
        action = agent.select_action(state, mask, greedy=False)
        action = int(np.clip(action, 0, len(cand.candidates) - 1))
        chosen = cand.candidates[action]
        reward = choose_reward(cand, chosen, args)
        agent.store_transition(state, action, reward, state.copy(), True, mask, zero_next_mask)
        loss = agent.learn()
        row = row_from_choice(
            episode=ep,
            cand=cand,
            action=action,
            reward=reward,
            loss=loss,
            epsilon=agent.epsilon,
            meta_row=meta_row,
        )
        rows.append(row)

        if ep % args.log_interval == 0 or ep == args.episodes:
            window = rows[-args.log_interval :]
            stats = summarize(window)
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
            agent.save(run_dir / "ddqn_reranker_latest.pt")
            print(
                f"[{ep:>6d}/{args.episodes:<6d} {progress*100:6.2f}%] "
                f"ETA={format_seconds(eta)} elapsed={format_seconds(elapsed)} eps={agent.epsilon:.3f} "
                f"DDQN={stats['ddqn_time']:.4f}h PredDij={stats['dijkstra_pred_time']:.4f}h "
                f"CandOracle={stats['candidate_oracle_time']:.4f}h RealDij={stats['dijkstra_real_time']:.4f}h "
                f"DDQN/Pred={stats['ddqn_over_pred']:.3f} "
                f"DDQN/Oracle={stats['ddqn_over_candidate_oracle']:.3f} "
                f"win={stats['ddqn_win_rate']:.1f}% pickOracle={stats['oracle_pick_rate']:.1f}% "
                f"opp={stats['opportunity_rate']:.1f}% predIsOracle={stats['pred_already_oracle_rate']:.1f}% "
                f"R={stats['avg_reward']:.3f} loss={stats['avg_loss']:.4f}",
                flush=True,
            )

    final_stats = summarize(rows)
    greedy_stats = greedy_evaluate(
        agent=agent,
        pool=pool,
        time_meta=time_meta,
        args=args,
        limit=args.greedy_eval_size,
    )
    agent.save(run_dir / "ddqn_reranker_final.pt")
    report = {
        "config": config,
        "final_training_metrics": final_stats,
        "greedy_eval_on_candidate_pool": greedy_stats,
        "last_window": window_rows[-1] if window_rows else {},
        "outputs": {
            "checkpoint": str(run_dir / "ddqn_reranker_final.pt"),
            "episode_metrics": str(run_dir / "episode_metrics.csv"),
            "window_summary": str(run_dir / "window_summary.csv"),
        },
    }
    (run_dir / "final_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=" * 96, flush=True)
    print("Training finished", flush=True)
    print("Final training metrics:", flush=True)
    print(json.dumps(final_stats, indent=2), flush=True)
    print("Greedy eval on candidate pool:", flush=True)
    print(json.dumps(greedy_stats, indent=2), flush=True)
    print(f"Saved: {run_dir / 'ddqn_reranker_final.pt'}", flush=True)
    print("=" * 96, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDQN reranker over K predicted-speed candidate routes.")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--superzone-dir", type=str, default="data/superzones_k64")
    p.add_argument("--edge-length-source", type=str, default="osrm", choices=["osrm", "centroid"])
    p.add_argument("--run-dir", type=str, default="runs/candidate_route_reranker")
    p.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    p.add_argument("--max-time-steps", type=int, default=0)
    p.add_argument("--hist-len", type=int, default=14)
    p.add_argument("--pred-horizon", type=int, default=4)
    p.add_argument("--time-slot-minutes", type=float, default=15.0)
    p.add_argument("--pred-source", type=str, default="stgat", choices=["persistence", "stgat"])
    p.add_argument(
        "--stgat-ckpt-dir",
        type=str,
        default="runs/final_dc_v_export_20260514_134425/checkpoints/v_nyc",
    )
    p.add_argument("--dispatch-source", type=str, default="persistence", choices=["persistence", "real_future_oracle"])
    p.add_argument("--episodes", type=int, default=50000)
    p.add_argument("--log-interval", type=int, default=500)
    p.add_argument("--candidate-pool-size", type=int, default=1000)
    p.add_argument("--candidate-build-batch-size", type=int, default=32)
    p.add_argument("--candidate-pool-attempt-multiplier", type=int, default=400)
    p.add_argument("--candidate-build-log-interval", type=int, default=100)
    p.add_argument("--k-routes", type=int, default=6)
    p.add_argument("--min-unique-routes", type=int, default=3)
    p.add_argument("--min-pred-hops", type=int, default=2)
    p.add_argument("--min-pred-distance-km", type=float, default=5.0)
    p.add_argument("--require-candidate-oracle-diff", action="store_true")
    p.add_argument("--min-candidate-oracle-improvement-ratio", type=float, default=1.0)
    p.add_argument("--greedy-eval-size", type=int, default=1000)
    p.add_argument("--feature-set", type=str, default="base", choices=["base", "uncertainty", "pattern_topk"])
    p.add_argument("--pattern-topk", type=int, default=16)
    p.add_argument("--pattern-attention-temperature", type=float, default=0.2)
    p.add_argument("--reward-mode", type=str, default="shaped", choices=["shaped", "direct_regret", "time_only"])
    p.add_argument("--normal-pool-size", type=int, default=0)
    p.add_argument("--opportunity-sample-ratio", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=62)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max-neighbors", type=int, default=10)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=10000)
    p.add_argument("--buffer-capacity", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-update", type=int, default=200)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--rho", type=float, default=50.0)
    p.add_argument("--reward-scale", type=float, default=10.0)
    p.add_argument("--oracle-regret-weight", type=float, default=5.0)
    p.add_argument("--win-bonus", type=float, default=1.0)
    p.add_argument("--oracle-bonus", type=float, default=1.0)
    p.add_argument("--feature-time-norm", type=float, default=2.0)
    p.add_argument("--feature-distance-norm", type=float, default=20.0)
    p.add_argument("--feature-hop-norm", type=float, default=12.0)
    p.add_argument("--feature-speed-norm", type=float, default=130.0)
    p.add_argument("--feature-log-count-norm", type=float, default=6.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
