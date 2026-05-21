from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from candidate_route_reranker_experiment import (
    CandidateDoubleDQNAgent,
    RouteChoiceCandidate,
    build_candidate_pool,
    build_state_and_mask,
    choose_reward,
    row_from_choice,
    state_dim_for_args,
    summarize,
    write_csv,
)
from evaluate_candidate_route_reranker import load_train_config, value
from graph_env import create_graph_from_data, infer_max_neighbors
from real_data_uncertain_routing_experiment import (
    build_provider,
    configure_runtime,
    format_seconds,
    resolve_device,
)
from superzone_graph import load_superzone_artifacts
from train_predictor import build_monthly_split_indices


@dataclass
class RuntimeBundle:
    data_dir: Path
    superzone_dir: str
    artifacts: dict
    edge_speeds: np.ndarray
    time_meta: pd.DataFrame
    split_indices: dict
    provider: object
    graph: object
    edge_index: np.ndarray
    edge_lengths: np.ndarray
    num_nodes: int
    device: torch.device


@dataclass
class GateRecord:
    cand: RouteChoiceCandidate
    meta_row: pd.Series
    state: np.ndarray
    mask: np.ndarray
    q_values: np.ndarray
    raw_action: int
    feature: np.ndarray
    label: int


class GateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def eval_args_from_config(
    args: argparse.Namespace,
    train_config: dict,
    *,
    run_dir: Path,
    split: str,
    pool_size: int,
) -> argparse.Namespace:
    hist_len = int(value(args, train_config, "hist_len") or 14)
    pred_horizon = int(value(args, train_config, "pred_horizon") or 4)
    eval_args = argparse.Namespace(**train_config)
    eval_args.run_dir = str(run_dir)
    eval_args.split = split
    eval_args.hist_len = hist_len
    eval_args.pred_horizon = pred_horizon
    eval_args.time_slot_minutes = float(value(args, train_config, "time_slot_minutes") or 15.0)
    eval_args.dispatch_source = args.dispatch_source or train_config.get("dispatch_source", "persistence")
    eval_args.candidate_pool_size = int(pool_size)
    eval_args.candidate_build_batch_size = int(args.candidate_build_batch_size)
    eval_args.candidate_build_log_interval = int(args.candidate_build_log_interval)
    eval_args.candidate_pool_attempt_multiplier = int(args.candidate_pool_attempt_multiplier)
    eval_args.k_routes = int(value(args, train_config, "k_routes") or 6)
    eval_args.min_unique_routes = int(value(args, train_config, "min_unique_routes") or 1)
    eval_args.min_pred_hops = int(value(args, train_config, "min_pred_hops") or 0)
    eval_args.min_pred_distance_km = float(value(args, train_config, "min_pred_distance_km") or 0.0)
    eval_args.require_candidate_oracle_diff = bool(
        args.require_candidate_oracle_diff
        if args.require_candidate_oracle_diff is not None
        else train_config.get("require_candidate_oracle_diff", False)
    )
    eval_args.min_candidate_oracle_improvement_ratio = float(
        value(args, train_config, "min_candidate_oracle_improvement_ratio") or 1.0
    )
    eval_args.alpha = float(value(args, train_config, "alpha") or 1.0)
    eval_args.beta = float(value(args, train_config, "beta") or 0.1)
    eval_args.rho = float(value(args, train_config, "rho") or 50.0)
    eval_args.reward_scale = float(value(args, train_config, "reward_scale") or 10.0)
    eval_args.oracle_regret_weight = float(value(args, train_config, "oracle_regret_weight") or 5.0)
    eval_args.win_bonus = float(value(args, train_config, "win_bonus") or 1.0)
    eval_args.oracle_bonus = float(value(args, train_config, "oracle_bonus") or 1.0)
    eval_args.feature_time_norm = float(value(args, train_config, "feature_time_norm") or 2.0)
    eval_args.feature_distance_norm = float(value(args, train_config, "feature_distance_norm") or 20.0)
    eval_args.feature_hop_norm = float(value(args, train_config, "feature_hop_norm") or 12.0)
    eval_args.feature_speed_norm = float(value(args, train_config, "feature_speed_norm") or 130.0)
    eval_args.feature_log_count_norm = float(value(args, train_config, "feature_log_count_norm") or 6.0)
    eval_args.feature_set = args.feature_set or train_config.get("feature_set", "pattern_topk")
    eval_args.reward_mode = args.reward_mode or train_config.get("reward_mode", "shaped")
    eval_args.pattern_topk = int(value(args, train_config, "pattern_topk") or 16)
    eval_args.pattern_attention_temperature = float(
        value(args, train_config, "pattern_attention_temperature") or 0.2
    )
    return eval_args


def build_runtime(args: argparse.Namespace, train_config: dict, device: torch.device) -> RuntimeBundle:
    data_dir = Path(args.data_dir or train_config.get("data_dir", "data"))
    superzone_dir = args.superzone_dir or train_config.get("superzone_dir", "data/superzones_k64")
    hist_len = int(value(args, train_config, "hist_len") or 14)
    pred_horizon = int(value(args, train_config, "pred_horizon") or 4)

    artifacts = load_superzone_artifacts(data_dir, superzone_dir)
    edge_speeds = np.load(data_dir / "edge_speeds.npy", mmap_mode="r")
    if edge_speeds.shape[0] < edge_speeds.shape[1]:
        edge_speeds = edge_speeds.T
    time_meta = pd.read_csv(data_dir / "time_meta.csv")
    if args.max_time_steps and args.max_time_steps > 0:
        max_t = min(int(args.max_time_steps), edge_speeds.shape[0], len(time_meta))
        edge_speeds = edge_speeds[:max_t]
        time_meta = time_meta.iloc[:max_t].reset_index(drop=True)
    time_meta["date"] = pd.to_datetime(time_meta["date"], errors="raise")
    split_indices = build_monthly_split_indices(time_meta, hist_len, pred_horizon)

    provider_args = argparse.Namespace(
        pred_source=args.pred_source or train_config.get("pred_source", "stgat"),
        stgat_ckpt_dir=args.stgat_ckpt_dir or train_config.get("stgat_ckpt_dir"),
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        edge_length_source=args.edge_length_source or train_config.get("edge_length_source", "osrm"),
        max_time_steps=int(args.max_time_steps or train_config.get("max_time_steps", 0) or 0),
    )
    provider = build_provider(provider_args, data_dir=data_dir, edge_speeds=edge_speeds, device=device)

    edge_index = artifacts["rl_edge_index"].astype(np.int32)
    edge_lengths = artifacts["rl_edge_lengths"].astype(np.float32)
    num_nodes = int(artifacts["region_demand"].shape[1])
    max_neighbors = infer_max_neighbors(
        edge_index,
        num_nodes,
        minimum=int(value(args, train_config, "max_neighbors") or 10),
    )
    avg_speeds = np.maximum(artifacts["rl_edge_speeds_kmh"].astype(np.float32), 1.0)
    graph = create_graph_from_data(
        num_nodes,
        edge_index,
        edge_lengths,
        avg_speeds,
        avg_speeds,
        max_neighbors=max_neighbors,
    )
    return RuntimeBundle(
        data_dir=data_dir,
        superzone_dir=superzone_dir,
        artifacts=artifacts,
        edge_speeds=edge_speeds,
        time_meta=time_meta,
        split_indices=split_indices,
        provider=provider,
        graph=graph,
        edge_index=edge_index,
        edge_lengths=edge_lengths,
        num_nodes=num_nodes,
        device=device,
    )


def load_agent(
    args: argparse.Namespace,
    train_config: dict,
    eval_args: argparse.Namespace,
    runtime: RuntimeBundle,
) -> CandidateDoubleDQNAgent:
    state_dim = state_dim_for_args(eval_args)
    agent = CandidateDoubleDQNAgent(
        num_nodes=runtime.num_nodes,
        action_dim=eval_args.k_routes,
        state_dim=state_dim,
        embed_dim=int(value(args, train_config, "embed_dim") or 16),
        hidden_dim=int(value(args, train_config, "hidden_dim") or 128),
        lr=float(value(args, train_config, "lr") or 1e-3),
        gamma=float(value(args, train_config, "gamma") or 0.0),
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1,
        buffer_capacity=1,
        batch_size=1,
        target_update=1,
        device=str(runtime.device),
    )
    ckpt = torch.load(args.checkpoint, map_location=runtime.device, weights_only=False)
    if int(ckpt.get("action_dim", eval_args.k_routes)) != eval_args.k_routes:
        raise ValueError("Checkpoint action_dim does not match k_routes")
    if int(ckpt.get("state_dim", state_dim)) != state_dim:
        raise ValueError("Checkpoint state_dim does not match evaluator state_dim")
    agent.online_net.load_state_dict(ckpt["online_state_dict"])
    agent.target_net.load_state_dict(ckpt["target_state_dict"])
    agent.epsilon = 0.0
    agent.online_net.eval()
    return agent


def masked_q_values(agent: CandidateDoubleDQNAgent, state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        s = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        q = agent.online_net(s).squeeze(0).detach().cpu().numpy().astype(np.float32)
    q = np.where(mask > 0, q, -np.inf).astype(np.float32)
    return q


def softmax_masked(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(q, dtype=np.float32)
    valid = mask > 0
    if not np.any(valid):
        return out
    vals = q[valid].astype(np.float64)
    vals = vals - np.max(vals)
    probs = np.exp(vals)
    probs = probs / max(float(probs.sum()), 1e-12)
    out[valid] = probs.astype(np.float32)
    return out


def gate_feature_vector(
    *,
    state: np.ndarray,
    mask: np.ndarray,
    q_values: np.ndarray,
    raw_action: int,
    num_nodes: int,
) -> np.ndarray:
    state_feat = state.astype(np.float32).copy()
    node_norm = max(float(num_nodes - 1), 1.0)
    state_feat[0] = state_feat[0] / node_norm
    state_feat[1] = state_feat[1] / node_norm

    q_clean = np.where(np.isfinite(q_values), q_values, 0.0).astype(np.float32)
    q0 = float(q_clean[0]) if q_clean.size else 0.0
    q_rel = (q_clean - q0).astype(np.float32)
    probs = softmax_masked(q_values, mask)
    valid_q = q_values[mask > 0]
    if valid_q.size >= 2:
        top = np.sort(valid_q.astype(np.float64))[-2:]
        top2_margin = float(top[-1] - top[-2])
    else:
        top2_margin = 0.0
    best_q = float(q_clean[raw_action]) if 0 <= raw_action < q_clean.size else q0
    extra = np.asarray(
        [
            float(raw_action) / max(float(mask.size - 1), 1.0),
            float(raw_action != 0),
            best_q,
            q0,
            best_q - q0,
            top2_margin,
            float(np.sum(mask > 0)) / max(float(mask.size), 1.0),
            float(np.max(probs)) if probs.size else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([state_feat, q_clean, q_rel, probs, extra]).astype(np.float32)


def gate_label(
    cand: RouteChoiceCandidate,
    raw_action: int,
    min_improvement_ratio: float,
    label_mode: str,
) -> int:
    baseline = float(cand.dijkstra_pred.real_result.travel_time)
    oracle = float(cand.candidate_oracle.real_result.travel_time)
    if not (math.isfinite(baseline) and math.isfinite(oracle) and baseline > 0.0):
        return 0
    if label_mode == "candidate_opportunity":
        return int(oracle < baseline * (1.0 - float(min_improvement_ratio)))
    if raw_action <= 0 or raw_action >= len(cand.candidates):
        return 0
    chosen = float(cand.candidates[raw_action].real_result.travel_time)
    if not math.isfinite(chosen):
        return 0
    return int(chosen < baseline * (1.0 - float(min_improvement_ratio)))


def build_records(
    *,
    pool: list[RouteChoiceCandidate],
    time_meta: pd.DataFrame,
    agent: CandidateDoubleDQNAgent,
    eval_args: argparse.Namespace,
    num_nodes: int,
    min_improvement_ratio: float,
    label_mode: str,
) -> list[GateRecord]:
    records: list[GateRecord] = []
    for cand in pool:
        meta_row = time_meta.iloc[cand.target_t]
        state, mask = build_state_and_mask(cand, meta_row=meta_row, args=eval_args)
        q = masked_q_values(agent, state, mask)
        raw_action = int(np.argmax(q))
        raw_action = int(np.clip(raw_action, 0, len(cand.candidates) - 1))
        feature = gate_feature_vector(
            state=state,
            mask=mask,
            q_values=q,
            raw_action=raw_action,
            num_nodes=num_nodes,
        )
        records.append(
            GateRecord(
                cand=cand,
                meta_row=meta_row,
                state=state,
                mask=mask,
                q_values=q,
                raw_action=raw_action,
                feature=feature,
                label=gate_label(cand, raw_action, min_improvement_ratio, label_mode),
            )
        )
    return records


def rows_for_policy(
    records: list[GateRecord],
    actions: list[int],
    *,
    eval_args: argparse.Namespace,
    prefix: str,
) -> list[dict]:
    rows: list[dict] = []
    for idx, (record, action) in enumerate(zip(records, actions, strict=True), start=1):
        action = int(np.clip(action, 0, len(record.cand.candidates) - 1))
        reward = choose_reward(record.cand, record.cand.candidates[action], eval_args)
        row = row_from_choice(
            episode=idx,
            cand=record.cand,
            action=action,
            reward=reward,
            loss=None,
            epsilon=0.0,
            meta_row=record.meta_row,
        )
        row["policy"] = prefix
        rows.append(row)
    return rows


def policy_stats(
    records: list[GateRecord],
    actions: list[int],
    *,
    eval_args: argparse.Namespace,
    prefix: str,
) -> dict:
    rows = rows_for_policy(records, actions, eval_args=eval_args, prefix=prefix)
    stats = summarize(rows)
    stats["override_rate"] = float(np.mean([int(a != 0) for a in actions]) * 100.0) if actions else 0.0
    stats["bad_gt_1pct"] = bad_override_rate(records, actions, 0.01)
    stats["bad_gt_5pct"] = bad_override_rate(records, actions, 0.05)
    return stats


def bad_override_rate(records: list[GateRecord], actions: list[int], ratio: float) -> float:
    bad = []
    for record, action in zip(records, actions, strict=True):
        baseline = float(record.cand.dijkstra_pred.real_result.travel_time)
        chosen = float(record.cand.candidates[int(action)].real_result.travel_time)
        bad.append(int(chosen > baseline * (1.0 + ratio)))
    return float(np.mean(bad) * 100.0) if bad else 0.0


def train_gate(
    *,
    fit_records: list[GateRecord],
    calib_records: list[GateRecord],
    eval_args: argparse.Namespace,
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> tuple[GateMLP, dict, float, dict]:
    x_fit = np.stack([r.feature for r in fit_records]).astype(np.float32)
    y_fit = np.asarray([r.label for r in fit_records], dtype=np.float32)
    x_calib = np.stack([r.feature for r in calib_records]).astype(np.float32)

    mean = x_fit.mean(axis=0)
    std = x_fit.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x_fit = (x_fit - mean) / std
    x_calib_norm = (x_calib - mean) / std

    model = GateMLP(x_fit.shape[1], int(args.gate_hidden_dim), float(args.gate_dropout)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(args.gate_lr), weight_decay=float(args.gate_weight_decay))
    pos = float(y_fit.sum())
    neg = float(len(y_fit) - pos)
    pos_weight = torch.tensor([min(max(neg / max(pos, 1.0), 1.0), float(args.max_pos_weight))], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    x_tensor = torch.tensor(x_fit, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_fit, dtype=torch.float32, device=device)
    rng = np.random.RandomState(int(args.seed) + 77)

    print(
        f"Training gate: fit={len(fit_records)} positives={int(pos)} "
        f"positive_rate={pos / max(len(fit_records), 1) * 100:.2f}% pos_weight={float(pos_weight.item()):.2f}",
        flush=True,
    )
    start = time.time()
    for epoch in range(1, int(args.gate_epochs) + 1):
        order = rng.permutation(len(fit_records))
        model.train()
        losses = []
        for begin in range(0, len(order), int(args.gate_batch_size)):
            idx = order[begin : begin + int(args.gate_batch_size)]
            logits = model(x_tensor[idx])
            loss = loss_fn(logits, y_tensor[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % int(args.gate_log_interval) == 0 or epoch == int(args.gate_epochs):
            elapsed = time.time() - start
            done = epoch / max(int(args.gate_epochs), 1)
            eta = elapsed / max(done, 1e-6) - elapsed
            with torch.no_grad():
                probs = torch.sigmoid(model(x_tensor)).detach().cpu().numpy()
            print(
                f"gate_epoch={epoch}/{args.gate_epochs} loss={np.mean(losses):.5f} "
                f"prob_mean={float(np.mean(probs)):.4f} elapsed={format_seconds(elapsed)} ETA={format_seconds(eta)}",
                flush=True,
            )

    normalizer = {"mean": mean.tolist(), "std": std.tolist()}
    calib_probs = predict_gate_probs(model, x_calib_norm, device)
    eval_args.gate_threshold_selection = str(args.gate_threshold_selection)
    threshold, sweep = choose_threshold(
        records=calib_records,
        probs=calib_probs,
        eval_args=eval_args,
        candidate_thresholds=int(args.threshold_candidates),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalizer": normalizer,
            "input_dim": int(x_fit.shape[1]),
            "threshold": float(threshold),
            "fit_positive_rate": float(pos / max(len(fit_records), 1)),
            "args": vars(args),
        },
        run_dir / "gate_model.pt",
    )
    (run_dir / "gate_threshold_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    (run_dir / "gate_normalizer.json").write_text(json.dumps(normalizer), encoding="utf-8")
    return model, normalizer, threshold, sweep[0]


def predict_gate_probs(model: GateMLP, x_norm: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for begin in range(0, len(x_norm), 4096):
            batch = torch.tensor(x_norm[begin : begin + 4096], dtype=torch.float32, device=device)
            out.append(torch.sigmoid(model(batch)).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out) if out else np.asarray([], dtype=np.float32)


def normalize_features(records: list[GateRecord], normalizer: dict) -> np.ndarray:
    x = np.stack([r.feature for r in records]).astype(np.float32)
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    std = np.asarray(normalizer["std"], dtype=np.float32)
    return (x - mean) / std


def gated_actions(records: list[GateRecord], probs: np.ndarray, threshold: float) -> list[int]:
    actions = []
    for record, prob in zip(records, probs, strict=True):
        if record.raw_action != 0 and float(prob) >= float(threshold):
            actions.append(int(record.raw_action))
        else:
            actions.append(0)
    return actions


def choose_threshold(
    *,
    records: list[GateRecord],
    probs: np.ndarray,
    eval_args: argparse.Namespace,
    candidate_thresholds: int,
) -> tuple[float, list[dict]]:
    if probs.size == 0:
        return 1.01, []
    quantiles = np.linspace(0.0, 1.0, max(candidate_thresholds, 2))
    thresholds = sorted(set([0.0, 0.5, 1.01] + [float(np.quantile(probs, q)) for q in quantiles]))
    sweep: list[dict] = []
    for threshold in thresholds:
        actions = gated_actions(records, probs, threshold)
        stats = policy_stats(records, actions, eval_args=eval_args, prefix="gate_calibration")
        stats["threshold"] = float(threshold)
        labels = np.asarray([r.label for r in records], dtype=np.float32)
        pred = np.asarray([int(a != 0) for a in actions], dtype=np.float32)
        tp = float(np.sum((labels > 0) & (pred > 0)))
        fp = float(np.sum((labels <= 0) & (pred > 0)))
        fn = float(np.sum((labels > 0) & (pred <= 0)))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        stats["gate_label_precision"] = precision * 100.0
        stats["gate_label_recall"] = recall * 100.0
        stats["gate_label_f1"] = (2.0 * precision * recall / max(precision + recall, 1e-12)) * 100.0
        sweep.append(stats)
    if getattr(eval_args, "gate_threshold_selection", "value") == "f1":
        sweep.sort(
            key=lambda s: (
                -float(s.get("gate_label_f1", 0.0)),
                float(s.get("ddqn_over_pred", float("inf"))),
                float(s.get("bad_gt_1pct", float("inf"))),
            )
        )
    else:
        sweep.sort(
            key=lambda s: (
                float(s.get("ddqn_over_pred", float("inf"))),
                float(s.get("bad_gt_1pct", float("inf"))),
                float(s.get("override_rate", float("inf"))),
            )
        )
    return float(sweep[0]["threshold"]), sweep


def build_pool_for_split(
    *,
    split: str,
    pool_size: int,
    run_dir: Path,
    args: argparse.Namespace,
    train_config: dict,
    runtime: RuntimeBundle,
    seed_offset: int,
) -> tuple[list[RouteChoiceCandidate], argparse.Namespace]:
    split_dir = run_dir / f"pool_{split}_{pool_size}"
    split_dir.mkdir(parents=True, exist_ok=True)
    eval_args = eval_args_from_config(args, train_config, run_dir=split_dir, split=split, pool_size=pool_size)
    candidate_windows = np.asarray(runtime.split_indices[split], dtype=np.int32)
    if candidate_windows.size == 0:
        raise ValueError(f"split {split} has no valid windows")
    rng = np.random.RandomState(int(args.seed) + int(seed_offset))
    print(
        f"Building full-distribution pool: split={split} windows={candidate_windows.size} "
        f"pool={pool_size} require_opportunity={eval_args.require_candidate_oracle_diff}",
        flush=True,
    )
    pool = build_candidate_pool(
        candidate_windows=candidate_windows,
        edge_speeds=runtime.edge_speeds,
        provider=runtime.provider,
        artifacts=runtime.artifacts,
        graph=runtime.graph,
        edge_index=runtime.edge_index,
        edge_lengths=runtime.edge_lengths,
        num_nodes=runtime.num_nodes,
        rng=rng,
        args=eval_args,
    )
    return pool, eval_args


def evaluate_split(
    *,
    split: str,
    records: list[GateRecord],
    probs: np.ndarray,
    threshold: float,
    eval_args: argparse.Namespace,
    run_dir: Path,
) -> dict:
    raw_actions = [r.raw_action for r in records]
    dijkstra_actions = [0 for _ in records]
    gate_actions = gated_actions(records, probs, threshold)
    raw_rows = rows_for_policy(records, raw_actions, eval_args=eval_args, prefix="raw_ddqn")
    dijkstra_rows = rows_for_policy(records, dijkstra_actions, eval_args=eval_args, prefix="dijkstra")
    gate_rows = rows_for_policy(records, gate_actions, eval_args=eval_args, prefix="gated_ddqn")
    write_csv(run_dir / f"{split}_raw_ddqn_episode_metrics.csv", raw_rows)
    write_csv(run_dir / f"{split}_dijkstra_episode_metrics.csv", dijkstra_rows)
    write_csv(run_dir / f"{split}_gated_ddqn_episode_metrics.csv", gate_rows)

    raw_stats = policy_stats(records, raw_actions, eval_args=eval_args, prefix="raw_ddqn")
    dijkstra_stats = policy_stats(records, dijkstra_actions, eval_args=eval_args, prefix="dijkstra")
    gate_stats = policy_stats(records, gate_actions, eval_args=eval_args, prefix="gated_ddqn")
    gate_activation = np.asarray([int(a != 0) for a in gate_actions], dtype=np.float32)
    gate_labels = np.asarray([r.label for r in records], dtype=np.float32)
    raw_override = np.asarray([int(a != 0) for a in raw_actions], dtype=np.float32)
    report = {
        "split": split,
        "threshold": float(threshold),
        "num_samples": int(len(records)),
        "raw_ddqn": raw_stats,
        "dijkstra_default": dijkstra_stats,
        "gated_ddqn": gate_stats,
        "gate_diagnostics": {
            "gate_activation_rate": float(gate_activation.mean() * 100.0) if len(gate_activation) else 0.0,
            "raw_override_rate": float(raw_override.mean() * 100.0) if len(raw_override) else 0.0,
            "gate_positive_rate_after_fact": float(gate_labels.mean() * 100.0) if len(gate_labels) else 0.0,
            "gate_precision_after_fact": float(gate_labels[gate_activation > 0].mean() * 100.0)
            if np.any(gate_activation > 0)
            else 0.0,
            "prob_mean": float(np.mean(probs)) if probs.size else 0.0,
            "prob_p90": float(np.quantile(probs, 0.9)) if probs.size else 0.0,
            "prob_p99": float(np.quantile(probs, 0.99)) if probs.size else 0.0,
        },
        "leakage_note": (
            "Gate decisions use only prediction-time state features, pattern_topk attention features, "
            "and DDQN Q-values. Realized future speeds are used only to train labels on train split "
            "and to score completed validation/test experiments."
        ),
    }
    (run_dir / f"{split}_gate_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{split} gated evaluation:", flush=True)
    print(json.dumps(report["gated_ddqn"], indent=2), flush=True)
    return report


def threshold_sweep_for_records(
    *,
    records: list[GateRecord],
    probs: np.ndarray,
    eval_args: argparse.Namespace,
    thresholds: list[float],
) -> list[dict]:
    labels = np.asarray([r.label for r in records], dtype=np.float32)
    sweep: list[dict] = []
    for threshold in thresholds:
        actions = gated_actions(records, probs, float(threshold))
        stats = policy_stats(records, actions, eval_args=eval_args, prefix="threshold_sweep")
        pred = np.asarray([int(a != 0) for a in actions], dtype=np.float32)
        tp = float(np.sum((labels > 0) & (pred > 0)))
        fp = float(np.sum((labels <= 0) & (pred > 0)))
        fn = float(np.sum((labels > 0) & (pred <= 0)))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        stats["threshold"] = float(threshold)
        stats["gate_label_precision"] = precision * 100.0
        stats["gate_label_recall"] = recall * 100.0
        stats["gate_label_f1"] = (2.0 * precision * recall / max(precision + recall, 1e-12)) * 100.0
        sweep.append(stats)
    sweep.sort(
        key=lambda s: (
            float(s.get("bad_gt_5pct", float("inf"))),
            -float(s.get("gate_label_precision", 0.0)),
            -float(s.get("override_rate", 0.0)),
        )
    )
    return sweep


def run(args: argparse.Namespace) -> None:
    train_config = load_train_config(args.train_config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))
    runtime = build_runtime(args, train_config, device)

    print("=" * 96, flush=True)
    print("Gated pattern_topk candidate-route reranker experiment", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"device={device} feature_set={args.feature_set or train_config.get('feature_set', 'pattern_topk')}", flush=True)
    print("No-leakage decision rule: gate sees predicted state/features/Q-values only.", flush=True)
    print("=" * 96, flush=True)

    agent_eval_args = eval_args_from_config(
        args,
        train_config,
        run_dir=run_dir / "agent_init",
        split="train",
        pool_size=max(1, int(args.train_gate_pool_size)),
    )
    agent = load_agent(args, train_config, agent_eval_args, runtime)
    if args.load_gate_model:
        gate_ckpt = torch.load(args.load_gate_model, map_location=device, weights_only=False)
        saved_args = gate_ckpt.get("args", {})
        gate_model = GateMLP(
            int(gate_ckpt["input_dim"]),
            int(saved_args.get("gate_hidden_dim", args.gate_hidden_dim)),
            float(saved_args.get("gate_dropout", args.gate_dropout)),
        ).to(device)
        gate_model.load_state_dict(gate_ckpt["model_state_dict"])
        normalizer = gate_ckpt["normalizer"]
        threshold = float(args.fixed_gate_threshold) if args.fixed_gate_threshold is not None else float(gate_ckpt["threshold"])
        best_calib = {
            "loaded_gate_model": str(args.load_gate_model),
            "threshold": float(threshold),
            "note": "Gate training was skipped; loaded model/normalizer from disk.",
        }
        print(f"Loaded gate model: {args.load_gate_model}", flush=True)
        print(f"Using gate threshold={threshold:.6f}", flush=True)
    else:
        train_pool, train_eval_args = build_pool_for_split(
            split="train",
            pool_size=int(args.train_gate_pool_size),
            run_dir=run_dir,
            args=args,
            train_config=train_config,
            runtime=runtime,
            seed_offset=0,
        )
        train_records = build_records(
            pool=train_pool,
            time_meta=runtime.time_meta,
            agent=agent,
            eval_args=train_eval_args,
            num_nodes=runtime.num_nodes,
            min_improvement_ratio=float(args.gate_min_improvement_ratio),
            label_mode=str(args.gate_label_mode),
        )
        rng = np.random.RandomState(int(args.seed) + 33)
        order = rng.permutation(len(train_records))
        fit_count = max(1, int(round(len(order) * float(args.gate_fit_fraction))))
        fit_count = min(fit_count, len(order) - 1) if len(order) > 1 else len(order)
        fit_records = [train_records[i] for i in order[:fit_count]]
        calib_records = [train_records[i] for i in order[fit_count:]]
        if not calib_records:
            calib_records = fit_records

        gate_model, normalizer, threshold, best_calib = train_gate(
            fit_records=fit_records,
            calib_records=calib_records,
            eval_args=train_eval_args,
            args=args,
            run_dir=run_dir,
            device=device,
        )
        if args.fixed_gate_threshold is not None:
            threshold = float(args.fixed_gate_threshold)
        print(f"Selected gate threshold={threshold:.6f} from train-calibration", flush=True)
        print(json.dumps(best_calib, indent=2), flush=True)

    reports = {
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "gate_threshold": float(threshold),
        "train_gate_pool_size": int(args.train_gate_pool_size),
        "eval_pool_size": int(args.eval_pool_size),
        "splits": {},
        "leakage_control": {
            "threshold_source": "train split calibration only",
            "gate_input": "prediction-time state + pattern_topk features + DDQN Q-values",
            "excluded_from_gate_input": "real future speeds, candidate_oracle, realized travel times",
            "gate_label_mode": str(args.gate_label_mode),
            "gate_threshold_selection": str(args.gate_threshold_selection),
        },
        "best_calibration": best_calib,
    }
    for idx, split in enumerate(args.eval_splits):
        split_run_dir = run_dir / f"eval_{split}"
        split_run_dir.mkdir(parents=True, exist_ok=True)
        pool, eval_args = build_pool_for_split(
            split=split,
            pool_size=int(args.eval_pool_size),
            run_dir=split_run_dir,
            args=args,
            train_config=train_config,
            runtime=runtime,
            seed_offset=1000 + idx * 1000,
        )
        records = build_records(
            pool=pool,
            time_meta=runtime.time_meta,
            agent=agent,
            eval_args=eval_args,
            num_nodes=runtime.num_nodes,
            min_improvement_ratio=float(args.gate_min_improvement_ratio),
            label_mode=str(args.gate_label_mode),
        )
        x_norm = normalize_features(records, normalizer)
        probs = predict_gate_probs(gate_model, x_norm, device)
        if args.eval_thresholds:
            sweep = threshold_sweep_for_records(
                records=records,
                probs=probs,
                eval_args=eval_args,
                thresholds=[float(x) for x in args.eval_thresholds],
            )
            (split_run_dir / f"{split}_threshold_sweep.json").write_text(
                json.dumps(sweep, indent=2),
                encoding="utf-8",
            )
            print(f"{split} threshold sweep saved: {split_run_dir / f'{split}_threshold_sweep.json'}", flush=True)
        report = evaluate_split(
            split=split,
            records=records,
            probs=probs,
            threshold=threshold,
            eval_args=eval_args,
            run_dir=split_run_dir,
        )
        reports["splits"][split] = report

    (run_dir / "gated_full_distribution_summary.json").write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )
    print("=" * 96, flush=True)
    print(f"Saved summary: {run_dir / 'gated_full_distribution_summary.json'}", flush=True)
    print("=" * 96, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate a no-leakage gate for DDQN route reranking.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--load-gate-model", type=str, default="")
    p.add_argument("--fixed-gate-threshold", type=float, default=None)
    p.add_argument("--eval-thresholds", nargs="*", type=float, default=None)
    p.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--train-gate-pool-size", type=int, default=5000)
    p.add_argument("--eval-pool-size", type=int, default=1000)
    p.add_argument("--candidate-build-batch-size", type=int, default=16)
    p.add_argument("--candidate-build-log-interval", type=int, default=100)
    p.add_argument("--candidate-pool-attempt-multiplier", type=int, default=200)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--superzone-dir", type=str, default=None)
    p.add_argument("--edge-length-source", type=str, default=None, choices=[None, "osrm", "centroid"])
    p.add_argument("--pred-source", type=str, default=None, choices=[None, "persistence", "stgat"])
    p.add_argument("--stgat-ckpt-dir", type=str, default=None)
    p.add_argument("--dispatch-source", type=str, default=None, choices=[None, "persistence", "real_future_oracle"])
    p.add_argument("--max-time-steps", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=121)
    p.add_argument("--hist-len", type=int, default=None)
    p.add_argument("--pred-horizon", type=int, default=None)
    p.add_argument("--k-routes", type=int, default=None)
    p.add_argument("--min-unique-routes", type=int, default=None)
    p.add_argument("--min-pred-hops", type=int, default=None)
    p.add_argument("--min-pred-distance-km", type=float, default=None)
    p.add_argument("--require-candidate-oracle-diff", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-candidate-oracle-improvement-ratio", type=float, default=None)
    p.add_argument("--max-neighbors", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--time-slot-minutes", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--reward-scale", type=float, default=None)
    p.add_argument("--oracle-regret-weight", type=float, default=None)
    p.add_argument("--win-bonus", type=float, default=None)
    p.add_argument("--oracle-bonus", type=float, default=None)
    p.add_argument("--feature-time-norm", type=float, default=None)
    p.add_argument("--feature-distance-norm", type=float, default=None)
    p.add_argument("--feature-hop-norm", type=float, default=None)
    p.add_argument("--feature-speed-norm", type=float, default=None)
    p.add_argument("--feature-log-count-norm", type=float, default=None)
    p.add_argument("--feature-set", type=str, default=None, choices=[None, "base", "uncertainty", "pattern_topk"])
    p.add_argument("--reward-mode", type=str, default=None, choices=[None, "shaped", "direct_regret", "time_only"])
    p.add_argument("--pattern-topk", type=int, default=None)
    p.add_argument("--pattern-attention-temperature", type=float, default=None)
    p.add_argument("--gate-fit-fraction", type=float, default=0.7)
    p.add_argument(
        "--gate-label-mode",
        type=str,
        default="raw_action_win",
        choices=["raw_action_win", "candidate_opportunity"],
    )
    p.add_argument("--gate-min-improvement-ratio", type=float, default=0.0)
    p.add_argument("--gate-threshold-selection", type=str, default="value", choices=["value", "f1"])
    p.add_argument("--gate-hidden-dim", type=int, default=128)
    p.add_argument("--gate-dropout", type=float, default=0.1)
    p.add_argument("--gate-lr", type=float, default=1e-3)
    p.add_argument("--gate-weight-decay", type=float, default=1e-4)
    p.add_argument("--gate-epochs", type=int, default=120)
    p.add_argument("--gate-batch-size", type=int, default=256)
    p.add_argument("--gate-log-interval", type=int, default=20)
    p.add_argument("--max-pos-weight", type=float, default=50.0)
    p.add_argument("--threshold-candidates", type=int, default=101)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
