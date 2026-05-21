from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from candidate_route_reranker_experiment import (
    CandidateDoubleDQNAgent,
    build_candidate_pool,
    build_state_and_mask,
    choose_reward,
    row_from_choice,
    summarize,
    state_dim_for_args,
    write_csv,
)
from graph_env import create_graph_from_data, infer_max_neighbors
from real_data_uncertain_routing_experiment import (
    build_provider,
    configure_runtime,
    resolve_device,
)
from superzone_graph import load_superzone_artifacts
from train_predictor import build_monthly_split_indices


def load_train_config(config_path: str | Path | None) -> dict:
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find training config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def value(args: argparse.Namespace, train_config: dict, name: str):
    raw = getattr(args, name)
    if raw is not None:
        return raw
    return train_config.get(name)


def run(args: argparse.Namespace) -> None:
    train_config = load_train_config(args.train_config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    seed = int(value(args, train_config, "seed") or 123)
    configure_runtime(device, seed)
    rng = np.random.RandomState(seed + int(args.seed_offset))

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
    candidate_windows = np.asarray(split_indices[args.split], dtype=np.int32)
    if candidate_windows.size == 0:
        raise ValueError(f"split {args.split} has no valid windows")

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

    eval_args = argparse.Namespace(**train_config)
    eval_args.run_dir = str(run_dir)
    eval_args.split = args.split
    eval_args.hist_len = hist_len
    eval_args.pred_horizon = pred_horizon
    eval_args.time_slot_minutes = float(value(args, train_config, "time_slot_minutes") or 15.0)
    eval_args.dispatch_source = args.dispatch_source or train_config.get("dispatch_source", "persistence")
    eval_args.candidate_pool_size = int(args.eval_pool_size)
    eval_args.candidate_build_batch_size = int(args.candidate_build_batch_size)
    eval_args.candidate_build_log_interval = int(args.candidate_build_log_interval)
    eval_args.candidate_pool_attempt_multiplier = int(args.candidate_pool_attempt_multiplier)
    eval_args.k_routes = int(value(args, train_config, "k_routes") or 6)
    eval_args.min_unique_routes = int(value(args, train_config, "min_unique_routes") or 3)
    eval_args.min_pred_hops = int(value(args, train_config, "min_pred_hops") or 2)
    eval_args.min_pred_distance_km = float(value(args, train_config, "min_pred_distance_km") or 5.0)
    eval_args.require_candidate_oracle_diff = bool(
        args.require_candidate_oracle_diff
        if args.require_candidate_oracle_diff is not None
        else train_config.get("require_candidate_oracle_diff", True)
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
    eval_args.feature_set = args.feature_set or train_config.get("feature_set", "base")
    eval_args.reward_mode = args.reward_mode or train_config.get("reward_mode", "shaped")
    eval_args.pattern_topk = int(value(args, train_config, "pattern_topk") or 16)
    eval_args.pattern_attention_temperature = float(
        value(args, train_config, "pattern_attention_temperature") or 0.2
    )

    state_dim = state_dim_for_args(eval_args)
    agent = CandidateDoubleDQNAgent(
        num_nodes=num_nodes,
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
        device=str(device),
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if int(ckpt.get("action_dim", eval_args.k_routes)) != eval_args.k_routes:
        raise ValueError("Checkpoint action_dim does not match k_routes")
    if int(ckpt.get("state_dim", state_dim)) != state_dim:
        raise ValueError("Checkpoint state_dim does not match evaluator state_dim")
    agent.online_net.load_state_dict(ckpt["online_state_dict"])
    agent.target_net.load_state_dict(ckpt["target_state_dict"])
    agent.epsilon = 0.0

    print("=" * 96, flush=True)
    print("Unseen candidate-route reranker evaluation", flush=True)
    print(
        f"split={args.split} windows={candidate_windows.size} pool={args.eval_pool_size} "
        f"pred={provider.name} K={eval_args.k_routes} device={device}",
        flush=True,
    )
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print("=" * 96, flush=True)

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
        args=eval_args,
    )
    rows: list[dict] = []
    for idx, cand in enumerate(pool, start=1):
        meta_row = time_meta.iloc[cand.target_t]
        state, mask = build_state_and_mask(cand, meta_row=meta_row, args=eval_args)
        action = agent.select_action(state, mask, greedy=True)
        action = int(np.clip(action, 0, len(cand.candidates) - 1))
        reward = choose_reward(cand, cand.candidates[action], eval_args)
        rows.append(
            row_from_choice(
                episode=idx,
                cand=cand,
                action=action,
                reward=reward,
                loss=None,
                epsilon=0.0,
                meta_row=meta_row,
            )
        )
    stats = summarize(rows)
    write_csv(run_dir / "eval_episode_metrics.csv", rows)
    report = {
        "eval_split": args.split,
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "eval_pool_size": int(args.eval_pool_size),
        "metrics": stats,
        "notes": {
            "unseen": "Evaluation pool is built from a different monthly split and is not used for training.",
            "opportunity_set": "Filters can require CandidateOracle to differ from Dijkstra-pred, so this evaluates correction cases rather than all OD cases.",
        },
    }
    (run_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Evaluation finished", flush=True)
    print(json.dumps(stats, indent=2), flush=True)
    print(f"Saved: {run_dir / 'eval_report.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained DDQN route reranker on unseen split samples.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str, default="")
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--eval-pool-size", type=int, default=300)
    p.add_argument("--candidate-build-batch-size", type=int, default=8)
    p.add_argument("--candidate-build-log-interval", type=int, default=5)
    p.add_argument("--candidate-pool-attempt-multiplier", type=int, default=1200)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--superzone-dir", type=str, default=None)
    p.add_argument("--edge-length-source", type=str, default=None, choices=[None, "osrm", "centroid"])
    p.add_argument("--pred-source", type=str, default=None, choices=[None, "persistence", "stgat"])
    p.add_argument("--stgat-ckpt-dir", type=str, default=None)
    p.add_argument("--dispatch-source", type=str, default=None, choices=[None, "persistence", "real_future_oracle"])
    p.add_argument("--max-time-steps", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seed-offset", type=int, default=1000)
    p.add_argument("--hist-len", type=int, default=None)
    p.add_argument("--pred-horizon", type=int, default=None)
    p.add_argument("--k-routes", type=int, default=None)
    p.add_argument("--min-unique-routes", type=int, default=None)
    p.add_argument("--min-pred-hops", type=int, default=None)
    p.add_argument("--min-pred-distance-km", type=float, default=None)
    p.add_argument("--require-candidate-oracle-diff", action=argparse.BooleanOptionalAction, default=None)
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
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
