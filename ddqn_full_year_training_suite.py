from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from candidate_route_reranker_experiment import (
    CandidateDoubleDQNAgent,
    build_state_and_mask,
    make_route_choice_candidate,
    state_dim_for_args,
)
from evaluate_candidate_route_reranker import load_train_config, value
from gated_candidate_route_reranker_experiment import build_runtime
from rcog_full_year_od_weighted_eval import (
    apply_train_config_defaults,
    configure_eval_args,
    dispatch_tasks_for_split,
    split_task_batches,
)
from rcog_stability_validation import cprint
from real_data_uncertain_routing_experiment import (
    configure_runtime,
    format_seconds,
    resolve_device,
    route_windows_to_profiles_batch,
)


RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RED = "\033[31m"


@dataclass
class RouteRecords:
    split: str
    arrays: dict[str, np.ndarray]
    meta: dict

    @property
    def n(self) -> int:
        return int(self.arrays["weights"].shape[0])


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_records(records: RouteRecords, cache_path: Path, meta_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **records.arrays)
    save_json(meta_path, records.meta)


def load_records(cache_path: Path, meta_path: Path, split: str) -> RouteRecords:
    loaded = np.load(cache_path, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return RouteRecords(split=split, arrays=arrays, meta=meta)


def concat_records(split: str, parts: list[RouteRecords]) -> RouteRecords:
    keys = parts[0].arrays.keys()
    arrays = {key: np.concatenate([p.arrays[key] for p in parts], axis=0) for key in keys}
    meta = {
        "split": split,
        "parts": [p.meta for p in parts],
        "records": int(arrays["weights"].shape[0]),
        "accepted_dispatch_weight": int(np.sum(arrays["weights"])),
        "total_tasks": int(sum(int(p.meta.get("total_tasks", 0)) for p in parts)),
        "total_dispatch_weight": int(sum(int(p.meta.get("total_dispatch_weight", 0)) for p in parts)),
    }
    meta["coverage_by_weight"] = float(meta["accepted_dispatch_weight"] / max(meta["total_dispatch_weight"], 1))
    return RouteRecords(split=split, arrays=arrays, meta=meta)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(values * weights) / denom)


def finite_eta(elapsed: float, done: float, total: float) -> float:
    progress = max(float(done) / max(float(total), 1.0), 1e-9)
    return elapsed * (1.0 - progress) / progress


def route_rewards(action_times: np.ndarray, baseline: float, mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    denom = max(float(baseline), 1e-6)
    rewards = float(args.reward_scale) * (float(baseline) - action_times.astype(np.float64)) / denom
    bad = (action_times > float(baseline) * (1.0 + float(args.bad_delta))) & (mask > 0)
    minor_bad = (action_times > float(baseline) * (1.0 + float(args.minor_bad_delta))) & (mask > 0)
    rewards = rewards - bad.astype(np.float64) * float(args.ddqn_bad_penalty)
    rewards = rewards - minor_bad.astype(np.float64) * float(args.ddqn_minor_bad_penalty)
    rewards[mask <= 0] = -1.0e6
    if mask.size:
        rewards[0] = 0.0
    return rewards.astype(np.float32)


def build_route_records(
    *,
    split: str,
    runtime,
    eval_args: argparse.Namespace,
    args: argparse.Namespace,
    run_dir: Path,
) -> RouteRecords:
    split_dir = run_dir / f"route_records_{split}"
    cache_path = split_dir / "route_records.npz"
    meta_path = split_dir / "route_records_meta.json"
    if cache_path.exists() and meta_path.exists() and not bool(args.rebuild_records):
        records = load_records(cache_path, meta_path, split)
        cprint(GREEN, f"[CACHE] loaded {split}: records={records.n:,} weight={int(np.sum(records.arrays['weights'])):,}")
        return records

    tasks = dispatch_tasks_for_split(
        runtime=runtime,
        split=split,
        hist_len=int(eval_args.hist_len),
        dispatch_source=str(eval_args.dispatch_source),
        max_windows=int(args.max_windows_per_split),
    )
    total_tasks = int(tasks.shape[0])
    total_weight = int(np.sum(tasks[:, 5])) if total_tasks else 0
    cprint(
        CYAN,
        f"[BUILD:{split}] OD-time tasks={total_tasks:,} dispatch_weight={total_weight:,} "
        f"profile_batch={int(args.profile_batch_size)}",
    )

    states: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    action_times: list[np.ndarray] = []
    action_rewards: list[np.ndarray] = []
    pred_action_times: list[np.ndarray] = []
    weights: list[int] = []
    target_actions: list[int] = []
    oracle_actions: list[int] = []
    baseline_times: list[float] = []
    oracle_times: list[float] = []
    dijkstra_real_times: list[float] = []
    opportunity_labels: list[int] = []
    unique_routes: list[int] = []
    origin_arr: list[int] = []
    dest_arr: list[int] = []
    window_arr: list[int] = []
    target_arr: list[int] = []
    dispatch_arr: list[int] = []
    skipped = 0
    skipped_weight = 0
    accepted_weight = 0
    start = time.time()
    next_log = int(args.record_log_interval)
    total_batches = math.ceil(max(len(np.unique(tasks[:, 0])) if total_tasks else 0, 1) / max(int(args.profile_batch_size), 1))

    for batch_idx, (batch_windows, starts, ends) in enumerate(split_task_batches(tasks, int(args.profile_batch_size)), start=1):
        pred_profiles, real_profiles = route_windows_to_profiles_batch(
            window_indices=[int(x) for x in batch_windows],
            hist_len=int(eval_args.hist_len),
            pred_horizon=int(eval_args.pred_horizon),
            edge_speeds=runtime.edge_speeds,
            provider=runtime.provider,
            artifacts=runtime.artifacts,
        )
        for window_idx, start_i, end_i, pred_profile, real_profile in zip(
            batch_windows, starts, ends, pred_profiles, real_profiles, strict=True
        ):
            for task in tasks[int(start_i) : int(end_i)]:
                _, target_t, dispatch_t, origin, dest, count = [int(x) for x in task]
                cand = make_route_choice_candidate(
                    window_idx=int(window_idx),
                    target_t=target_t,
                    dispatch_t=dispatch_t,
                    origin=origin,
                    dest=dest,
                    dispatch_count=count,
                    pred_profile=pred_profile,
                    real_profile=real_profile,
                    graph=runtime.graph,
                    edge_index=runtime.edge_index,
                    edge_lengths=runtime.edge_lengths,
                    num_nodes=runtime.num_nodes,
                    args=eval_args,
                )
                if cand is None:
                    skipped += 1
                    skipped_weight += int(count)
                    continue

                meta_row = runtime.time_meta.iloc[cand.target_t]
                state, mask = build_state_and_mask(cand, meta_row=meta_row, args=eval_args)
                k = int(eval_args.k_routes)
                times = np.full(k, np.inf, dtype=np.float32)
                pred_times = np.full(k, np.inf, dtype=np.float32)
                valid_n = min(len(cand.candidates), k)
                for action_idx, route in enumerate(cand.candidates[:k]):
                    times[action_idx] = float(route.real_result.travel_time)
                    pred_times[action_idx] = float(route.pred_result.travel_time)
                valid_mask = mask > 0
                if not np.any(valid_mask):
                    skipped += 1
                    skipped_weight += int(count)
                    continue
                oracle_action = int(np.argmin(np.where(valid_mask, times, np.inf)))
                baseline = float(cand.dijkstra_pred.real_result.travel_time)
                oracle = float(times[oracle_action])
                opportunity = int(oracle < baseline * (1.0 - float(args.benefit_delta)))
                target_action = oracle_action if opportunity else 0
                rewards = route_rewards(times, baseline, mask, args)

                states.append(state.astype(np.float32))
                masks.append(mask.astype(np.float32))
                action_times.append(times.astype(np.float32))
                action_rewards.append(rewards.astype(np.float32))
                pred_action_times.append(pred_times.astype(np.float32))
                weights.append(int(cand.dispatch_count))
                target_actions.append(int(target_action))
                oracle_actions.append(int(oracle_action))
                baseline_times.append(float(baseline))
                oracle_times.append(float(oracle))
                dijkstra_real_times.append(float(cand.dijkstra_real.travel_time))
                opportunity_labels.append(opportunity)
                unique_routes.append(int(valid_n))
                origin_arr.append(int(cand.origin))
                dest_arr.append(int(cand.dest))
                window_arr.append(int(cand.window_idx))
                target_arr.append(int(cand.target_t))
                dispatch_arr.append(int(cand.dispatch_t))
                accepted_weight += int(cand.dispatch_count)

        processed = len(weights) + skipped
        if processed >= next_log or batch_idx == total_batches:
            elapsed = time.time() - start
            eta = finite_eta(elapsed, processed, total_tasks)
            opp_rate = (
                float(np.average(opportunity_labels, weights=np.asarray(weights, dtype=np.float64)) * 100.0)
                if weights
                else 0.0
            )
            cprint(
                BLUE,
                f"[BUILD:{split}] {processed:,}/{total_tasks:,} records={len(weights):,} skipped={skipped:,} "
                f"weight={accepted_weight:,}/{total_weight:,} opp={opp_rate:5.2f}% "
                f"elapsed={format_seconds(elapsed)} ETA={format_seconds(eta)}",
            )
            next_log += int(args.record_log_interval)

    if not states:
        raise RuntimeError(f"No valid route records built for split={split}")

    arrays = {
        "states": np.stack(states).astype(np.float32),
        "masks": np.stack(masks).astype(np.float32),
        "action_times": np.stack(action_times).astype(np.float32),
        "action_rewards": np.stack(action_rewards).astype(np.float32),
        "pred_action_times": np.stack(pred_action_times).astype(np.float32),
        "weights": np.asarray(weights, dtype=np.float64),
        "target_action": np.asarray(target_actions, dtype=np.int16),
        "oracle_action": np.asarray(oracle_actions, dtype=np.int16),
        "baseline_time": np.asarray(baseline_times, dtype=np.float64),
        "oracle_time": np.asarray(oracle_times, dtype=np.float64),
        "dijkstra_real_time": np.asarray(dijkstra_real_times, dtype=np.float64),
        "opportunity_label": np.asarray(opportunity_labels, dtype=bool),
        "unique_routes": np.asarray(unique_routes, dtype=np.int16),
        "origin": np.asarray(origin_arr, dtype=np.int16),
        "dest": np.asarray(dest_arr, dtype=np.int16),
        "window_idx": np.asarray(window_arr, dtype=np.int32),
        "target_t": np.asarray(target_arr, dtype=np.int32),
        "dispatch_t": np.asarray(dispatch_arr, dtype=np.int32),
    }
    meta = {
        "split": split,
        "total_tasks": total_tasks,
        "total_dispatch_weight": total_weight,
        "records": int(arrays["weights"].shape[0]),
        "accepted_dispatch_weight": int(np.sum(arrays["weights"])),
        "skipped_tasks": int(skipped),
        "skipped_dispatch_weight": int(skipped_weight),
        "coverage_by_pairs": float(arrays["weights"].shape[0] / max(total_tasks, 1)),
        "coverage_by_weight": float(np.sum(arrays["weights"]) / max(total_weight, 1)),
        "elapsed_sec": float(time.time() - start),
        "state_dim": int(arrays["states"].shape[1]),
        "action_dim": int(arrays["masks"].shape[1]),
        "benefit_delta": float(args.benefit_delta),
        "bad_delta": float(args.bad_delta),
        "min_unique_routes": int(eval_args.min_unique_routes),
        "min_pred_hops": int(eval_args.min_pred_hops),
        "min_pred_distance_km": float(eval_args.min_pred_distance_km),
    }
    records = RouteRecords(split=split, arrays=arrays, meta=meta)
    save_records(records, cache_path, meta_path)
    cprint(
        GREEN,
        f"[CACHE] saved {split}: records={records.n:,} weight={int(np.sum(arrays['weights'])):,} "
        f"coverage={meta['coverage_by_weight']*100:.2f}% elapsed={format_seconds(meta['elapsed_sec'])}",
    )
    return records


def create_agent(args: argparse.Namespace, runtime, eval_args: argparse.Namespace, *, epsilon_start: float) -> CandidateDoubleDQNAgent:
    state_dim = state_dim_for_args(eval_args)
    return CandidateDoubleDQNAgent(
        num_nodes=runtime.num_nodes,
        action_dim=int(eval_args.k_routes),
        state_dim=state_dim,
        embed_dim=int(args.embed_dim),
        hidden_dim=int(args.hidden_dim),
        lr=float(args.lr),
        gamma=float(args.gamma),
        epsilon_start=float(epsilon_start),
        epsilon_end=float(args.epsilon_end),
        epsilon_decay=int(args.epsilon_decay),
        buffer_capacity=1,
        batch_size=int(args.ddqn_batch_size),
        target_update=int(args.target_update),
        device=str(runtime.device),
    )


def save_agent_checkpoint(
    path: Path,
    agent: CandidateDoubleDQNAgent,
    *,
    args: argparse.Namespace,
    stage: str,
    step: int,
    metrics: dict,
    extra: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "online_state_dict": agent.online_net.state_dict(),
        "target_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "epsilon": float(agent.epsilon),
        "learn_step_count": int(agent.learn_step_count),
        "action_dim": int(agent.action_dim),
        "state_dim": int(agent.state_dim),
        "stage": stage,
        "step": int(step),
        "metrics": metrics,
        "training_args": vars(args).copy(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_policy_checkpoint(path: Path, agent: CandidateDoubleDQNAgent, *, load_optimizer: bool) -> dict:
    ckpt = torch.load(path, map_location=agent.device, weights_only=False)
    if int(ckpt.get("state_dim", agent.state_dim)) != int(agent.state_dim):
        raise ValueError(f"Checkpoint state_dim {ckpt.get('state_dim')} != agent state_dim {agent.state_dim}")
    if int(ckpt.get("action_dim", agent.action_dim)) != int(agent.action_dim):
        raise ValueError(f"Checkpoint action_dim {ckpt.get('action_dim')} != agent action_dim {agent.action_dim}")
    agent.online_net.load_state_dict(ckpt["online_state_dict"])
    agent.target_net.load_state_dict(ckpt.get("target_state_dict", ckpt["online_state_dict"]))
    if load_optimizer and "optimizer_state_dict" in ckpt:
        agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    agent.epsilon = float(ckpt.get("epsilon", agent.epsilon))
    agent.learn_step_count = int(ckpt.get("learn_step_count", 0))
    return ckpt


def greedy_actions(agent: CandidateDoubleDQNAgent, records: RouteRecords, batch_size: int) -> np.ndarray:
    states = records.arrays["states"]
    masks = records.arrays["masks"]
    out = np.zeros(records.n, dtype=np.int16)
    agent.online_net.eval()
    with torch.no_grad():
        for start in range(0, records.n, int(batch_size)):
            end = min(start + int(batch_size), records.n)
            s = torch.tensor(states[start:end], dtype=torch.float32, device=agent.device)
            m_np = masks[start:end]
            q = agent.online_net(s).detach().cpu().numpy().astype(np.float32)
            q = np.where(m_np > 0, q, -np.inf)
            out[start:end] = np.argmax(q, axis=1).astype(np.int16)
    agent.online_net.train()
    return out


def stats_for_actions(records: RouteRecords, actions: np.ndarray, args: argparse.Namespace) -> dict[str, float]:
    arr = records.arrays
    actions = np.asarray(actions, dtype=np.int64)
    idx = np.arange(records.n)
    w = arr["weights"].astype(np.float64)
    chosen = arr["action_times"][idx, actions].astype(np.float64)
    baseline = arr["baseline_time"].astype(np.float64)
    oracle = arr["oracle_time"].astype(np.float64)
    activated = actions != 0
    opp = arr["opportunity_label"].astype(bool)
    benefit = activated & (chosen < baseline * (1.0 - float(args.benefit_delta)))
    bad1 = chosen > baseline * 1.01
    bad5 = chosen > baseline * (1.0 + float(args.bad_delta))
    denom = max(float(np.sum(w)), 1e-12)
    active_w = w[activated]
    opp_w = w[opp]
    benefit_w = w[benefit]
    total_ratio = float(np.sum(chosen * w) / max(float(np.sum(baseline * w)), 1e-12))
    return {
        "num_records": int(records.n),
        "dispatch_weight": float(np.sum(w)),
        "ddqn_time": weighted_mean(chosen, w),
        "dijkstra_pred_time": weighted_mean(baseline, w),
        "candidate_oracle_time": weighted_mean(oracle, w),
        "dijkstra_real_time": weighted_mean(arr["dijkstra_real_time"].astype(np.float64), w),
        "ddqn_over_pred": weighted_mean(chosen / np.maximum(baseline, 1e-9), w),
        "total_cost_ratio": total_ratio,
        "total_cost_improvement_pct": float((1.0 - total_ratio) * 100.0),
        "ddqn_over_candidate_oracle": weighted_mean(chosen / np.maximum(oracle, 1e-9), w),
        "candidate_oracle_over_pred": weighted_mean(oracle / np.maximum(baseline, 1e-9), w),
        "activation_rate": float(np.sum(active_w) / denom * 100.0),
        "opportunity_rate": float(np.sum(w[opp]) / denom * 100.0),
        "opportunity_precision": float(np.sum(w[activated & opp]) / max(float(np.sum(active_w)), 1e-12) * 100.0) if np.any(activated) else 0.0,
        "opportunity_recall": float(np.sum(w[activated & opp]) / max(float(np.sum(opp_w)), 1e-12) * 100.0) if np.any(opp) else 0.0,
        "benefit_precision": float(np.sum(w[benefit]) / max(float(np.sum(active_w)), 1e-12) * 100.0) if np.any(activated) else 0.0,
        "benefit_recall": float(np.sum(w[benefit]) / max(float(np.sum(benefit_w)), 1e-12) * 100.0) if np.any(benefit) else 0.0,
        "bad_gt_1pct": float(np.sum(w[bad1]) / denom * 100.0),
        "bad_gt_5pct": float(np.sum(w[bad5]) / denom * 100.0),
        "oracle_pick_rate": float(np.sum(w[actions == arr["oracle_action"]]) / denom * 100.0),
        "avg_action": weighted_mean(actions.astype(np.float64), w),
        "avg_unique_routes": weighted_mean(arr["unique_routes"].astype(np.float64), w),
    }


def evaluate_policy(
    *,
    agent: CandidateDoubleDQNAgent,
    records_by_name: dict[str, RouteRecords],
    args: argparse.Namespace,
    names: tuple[str, ...] = ("val",),
) -> dict[str, dict]:
    metrics = {}
    for name in names:
        actions = greedy_actions(agent, records_by_name[name], int(args.eval_batch_size))
        metrics[name] = stats_for_actions(records_by_name[name], actions, args)
    return metrics


def sample_weights(records: RouteRecords, args: argparse.Namespace) -> np.ndarray:
    w = records.arrays["weights"].astype(np.float64)
    if args.sample_weight_mode == "none":
        probs = np.ones_like(w, dtype=np.float64)
    elif args.sample_weight_mode == "sqrt_count":
        probs = np.sqrt(np.maximum(w, 1.0))
    else:
        probs = np.maximum(w, 1.0)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    return probs


def sample_batch_indices(rng: np.random.RandomState, probs: np.ndarray, batch_size: int) -> np.ndarray:
    return rng.choice(probs.shape[0], size=int(batch_size), replace=True, p=probs)


def supervised_loss(
    agent: CandidateDoubleDQNAgent,
    states: torch.Tensor,
    masks: torch.Tensor,
    targets: torch.Tensor,
    reward_targets: torch.Tensor,
    sample_weight: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    q = agent.online_net(states)
    q_masked = q.masked_fill(masks <= 0, -1.0e9)
    ce = nn.functional.cross_entropy(q_masked, targets, reduction="none")
    target_bonus = torch.where(targets > 0, float(args.supervised_positive_weight), 1.0)
    ce_loss = torch.mean(ce * sample_weight * target_bonus)
    valid = masks > 0
    reg = nn.functional.smooth_l1_loss(q[valid], reward_targets[valid], reduction="none")
    reg_weights = sample_weight.unsqueeze(1).expand_as(masks)[valid]
    reg_loss = torch.sum(reg * reg_weights) / torch.clamp(torch.sum(reg_weights), min=1.0)
    loss = ce_loss + float(args.supervised_reward_loss_weight) * reg_loss
    return loss, {"ce_loss": float(ce_loss.item()), "reg_loss": float(reg_loss.item())}


def train_supervised_reranker(
    *,
    agent: CandidateDoubleDQNAgent,
    records_by_name: dict[str, RouteRecords],
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Path, dict]:
    stage_dir = run_dir / "supervised_reranker"
    stage_dir.mkdir(parents=True, exist_ok=True)
    train = records_by_name["train"]
    rng = np.random.RandomState(int(args.seed) + 1001)
    probs = sample_weights(train, args)
    total_steps = int(args.supervised_epochs) * max(1, math.ceil(train.n / int(args.supervised_batch_size)))
    history: list[dict] = []
    best_ratio = float("inf")
    best_path = stage_dir / "supervised_reranker_best.pt"
    latest_path = stage_dir / "supervised_reranker_latest.pt"
    final_path = stage_dir / "supervised_reranker_final.pt"
    start = time.time()
    step = 0
    cprint(MAGENTA, f"[RERANKER] training supervised reranker epochs={args.supervised_epochs} steps={total_steps:,}")

    for epoch in range(1, int(args.supervised_epochs) + 1):
        steps_this_epoch = max(1, math.ceil(train.n / int(args.supervised_batch_size)))
        for _ in range(steps_this_epoch):
            step += 1
            idx = sample_batch_indices(rng, probs, int(args.supervised_batch_size))
            s = torch.tensor(train.arrays["states"][idx], dtype=torch.float32, device=agent.device)
            m = torch.tensor(train.arrays["masks"][idx], dtype=torch.float32, device=agent.device)
            y = torch.tensor(train.arrays["target_action"][idx].astype(np.int64), dtype=torch.long, device=agent.device)
            r = torch.tensor(train.arrays["action_rewards"][idx], dtype=torch.float32, device=agent.device)
            sw_np = np.sqrt(np.maximum(train.arrays["weights"][idx].astype(np.float64), 1.0))
            sw_np = sw_np / max(float(np.mean(sw_np)), 1e-12)
            sw = torch.tensor(sw_np.astype(np.float32), dtype=torch.float32, device=agent.device)
            loss, loss_parts = supervised_loss(agent, s, m, y, r, sw, args)
            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.online_net.parameters(), max_norm=float(args.grad_clip))
            agent.optimizer.step()
            agent.learn_step_count += 1
            if agent.learn_step_count % max(int(args.target_update), 1) == 0:
                agent.target_net.load_state_dict(agent.online_net.state_dict())

            if step % int(args.supervised_log_interval) == 0 or step == total_steps:
                metrics = evaluate_policy(agent=agent, records_by_name=records_by_name, args=args, names=("val",))
                val = metrics["val"]
                elapsed = time.time() - start
                eta = finite_eta(elapsed, step, total_steps)
                row = {
                    "stage": "supervised_reranker",
                    "epoch": int(epoch),
                    "step": int(step),
                    "total_steps": int(total_steps),
                    "elapsed_sec": float(elapsed),
                    "eta_sec": float(eta),
                    "loss": float(loss.item()),
                    **loss_parts,
                    **{f"val_{k}": v for k, v in val.items()},
                }
                history.append(row)
                write_rows(stage_dir / "metrics.csv", history)
                save_agent_checkpoint(latest_path, agent, args=args, stage="supervised_reranker", step=step, metrics=row)
                if val["total_cost_ratio"] < best_ratio:
                    best_ratio = float(val["total_cost_ratio"])
                    save_agent_checkpoint(best_path, agent, args=args, stage="supervised_reranker", step=step, metrics=row)
                color = GREEN if val["total_cost_ratio"] < 1.0 else YELLOW
                cprint(
                    color,
                    f"[RERANKER {step:,}/{total_steps:,}] epoch={epoch}/{args.supervised_epochs} "
                    f"loss={loss.item():.4f} ce={loss_parts['ce_loss']:.4f} reg={loss_parts['reg_loss']:.4f} "
                    f"val_ratio={val['total_cost_ratio']:.6f} imp={val['total_cost_improvement_pct']:.3f}% "
                    f"benP={val['benefit_precision']:.1f}% benR={val['benefit_recall']:.1f}% "
                    f"bad5={val['bad_gt_5pct']:.3f}% act={val['activation_rate']:.2f}% "
                    f"elapsed={format_seconds(elapsed)} reranker_total_ETA={format_seconds(eta)}",
                )

    agent.target_net.load_state_dict(agent.online_net.state_dict())
    final_metrics = evaluate_policy(
        agent=agent,
        records_by_name=records_by_name,
        args=args,
        names=("train", "val", "test", "full"),
    )
    save_agent_checkpoint(final_path, agent, args=args, stage="supervised_reranker", step=step, metrics=final_metrics)
    save_json(stage_dir / "final_summary.json", {"best_checkpoint": str(best_path), "final_checkpoint": str(final_path), "metrics": final_metrics})
    cprint(
        GREEN,
        f"[RERANKER DONE] final_test_ratio={final_metrics['test']['total_cost_ratio']:.6f} "
        f"imp={final_metrics['test']['total_cost_improvement_pct']:.3f}% checkpoint={final_path}",
    )
    return final_path, final_metrics


def ddqn_train_step(
    agent: CandidateDoubleDQNAgent,
    records: RouteRecords,
    idx: np.ndarray,
    episode_seen: int,
    args: argparse.Namespace,
    rng: np.random.RandomState,
) -> tuple[float, float, dict[str, float]]:
    states_np = records.arrays["states"][idx]
    masks_np = records.arrays["masks"][idx]
    rewards_np = records.arrays["action_rewards"][idx]
    states = torch.tensor(states_np, dtype=torch.float32, device=agent.device)
    masks = torch.tensor(masks_np, dtype=torch.float32, device=agent.device)
    rewards = torch.tensor(rewards_np, dtype=torch.float32, device=agent.device)
    q = agent.online_net(states)
    q_masked = q.masked_fill(masks <= 0, -1.0e9)
    greedy = torch.argmax(q_masked, dim=1).detach().cpu().numpy().astype(np.int64)
    eps = max(
        float(args.epsilon_end),
        float(args.epsilon_start)
        - (float(args.epsilon_start) - float(args.epsilon_end)) * float(episode_seen) / max(int(args.epsilon_decay), 1),
    )
    actions_np = greedy.copy()
    explore = rng.random_sample(actions_np.shape[0]) < eps
    if np.any(explore):
        rows = np.flatnonzero(explore)
        for row in rows:
            valid = np.flatnonzero(masks_np[row] > 0)
            actions_np[row] = int(rng.choice(valid)) if valid.size else 0
    actions = torch.tensor(actions_np, dtype=torch.long, device=agent.device).unsqueeze(1)
    q_sa = q.gather(1, actions).squeeze(1)
    target = rewards.gather(1, actions).squeeze(1).detach()
    sw_np = np.sqrt(np.maximum(records.arrays["weights"][idx].astype(np.float64), 1.0))
    sw_np = sw_np / max(float(np.mean(sw_np)), 1e-12)
    sw = torch.tensor(sw_np.astype(np.float32), dtype=torch.float32, device=agent.device)
    loss_vec = nn.functional.smooth_l1_loss(q_sa, target, reduction="none")
    loss = torch.mean(loss_vec * sw)
    agent.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.online_net.parameters(), max_norm=float(args.grad_clip))
    agent.optimizer.step()
    agent.learn_step_count += int(actions_np.shape[0])
    agent.epsilon = float(eps)
    if agent.learn_step_count % max(int(args.target_update), 1) < int(actions_np.shape[0]):
        agent.target_net.load_state_dict(agent.online_net.state_dict())
    chosen_times = records.arrays["action_times"][idx, actions_np].astype(np.float64)
    baseline = records.arrays["baseline_time"][idx].astype(np.float64)
    benefit = np.mean((actions_np != 0) & (chosen_times < baseline * (1.0 - float(args.benefit_delta)))) * 100.0
    bad5 = np.mean((actions_np != 0) & (chosen_times > baseline * (1.0 + float(args.bad_delta)))) * 100.0
    return float(loss.item()), float(eps), {
        "batch_reward": float(np.mean(target.detach().cpu().numpy())),
        "batch_action": float(np.mean(actions_np)),
        "batch_activation": float(np.mean(actions_np != 0) * 100.0),
        "batch_benefit_rate": float(benefit),
        "batch_bad_gt_5pct": float(bad5),
    }


def train_ddqn_stage(
    *,
    stage_name: str,
    agent: CandidateDoubleDQNAgent,
    records_by_name: dict[str, RouteRecords],
    args: argparse.Namespace,
    run_dir: Path,
    total_ddqn_episodes: int,
    ddqn_episode_offset: int,
    episodes: int,
) -> tuple[Path, dict]:
    stage_dir = run_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    train = records_by_name["train"]
    rng = np.random.RandomState(int(args.seed) + (2001 if stage_name == "pure_ddqn_200k" else 3001))
    probs = sample_weights(train, args)
    history: list[dict] = []
    latest_path = stage_dir / f"{stage_name}_latest.pt"
    best_path = stage_dir / f"{stage_name}_best.pt"
    final_path = stage_dir / f"{stage_name}_final.pt"
    best_ratio = float("inf")
    start = time.time()
    stage_seen = 0
    cprint(MAGENTA, f"[{stage_name}] training episodes={episodes:,} batch={args.ddqn_batch_size}")

    while stage_seen < int(episodes):
        batch_n = min(int(args.ddqn_batch_size), int(episodes) - stage_seen)
        idx = sample_batch_indices(rng, probs, batch_n)
        ddqn_seen = int(ddqn_episode_offset) + int(stage_seen)
        loss, eps, batch_stats = ddqn_train_step(agent, train, idx, ddqn_seen, args, rng)
        stage_seen += batch_n

        if stage_seen % int(args.ddqn_log_interval) < batch_n or stage_seen >= int(episodes):
            metrics = evaluate_policy(agent=agent, records_by_name=records_by_name, args=args, names=("val",))
            val = metrics["val"]
            elapsed = time.time() - start
            stage_eta = finite_eta(elapsed, stage_seen, episodes)
            ddqn_done = int(ddqn_episode_offset) + int(stage_seen)
            ddqn_progress_elapsed = elapsed
            if ddqn_episode_offset > 0 and getattr(args, "_pure_ddqn_elapsed_sec", 0.0):
                ddqn_progress_elapsed = float(args._pure_ddqn_elapsed_sec) + elapsed
            ddqn_total_eta = finite_eta(ddqn_progress_elapsed, ddqn_done, total_ddqn_episodes)
            row = {
                "stage": stage_name,
                "episode": int(stage_seen),
                "episodes": int(episodes),
                "ddqn_total_episode": int(ddqn_done),
                "ddqn_total_episodes": int(total_ddqn_episodes),
                "elapsed_sec": float(elapsed),
                "stage_eta_sec": float(stage_eta),
                "ddqn_total_eta_sec": float(ddqn_total_eta),
                "loss": float(loss),
                "epsilon": float(eps),
                **batch_stats,
                **{f"val_{k}": v for k, v in val.items()},
            }
            history.append(row)
            write_rows(stage_dir / "metrics.csv", history)
            save_agent_checkpoint(latest_path, agent, args=args, stage=stage_name, step=stage_seen, metrics=row)
            if val["total_cost_ratio"] < best_ratio:
                best_ratio = float(val["total_cost_ratio"])
                save_agent_checkpoint(best_path, agent, args=args, stage=stage_name, step=stage_seen, metrics=row)
            color = GREEN if val["total_cost_ratio"] < 1.0 else YELLOW
            cprint(
                color,
                f"[{stage_name} {stage_seen:,}/{episodes:,}] ddqn_total={ddqn_done:,}/{total_ddqn_episodes:,} "
                f"eps={eps:.3f} loss={loss:.4f} batchR={batch_stats['batch_reward']:.3f} "
                f"val_ratio={val['total_cost_ratio']:.6f} imp={val['total_cost_improvement_pct']:.3f}% "
                f"benP={val['benefit_precision']:.1f}% benR={val['benefit_recall']:.1f}% "
                f"bad5={val['bad_gt_5pct']:.3f}% act={val['activation_rate']:.2f}% "
                f"stage_ETA={format_seconds(stage_eta)} DDQN_total_ETA={format_seconds(ddqn_total_eta)}",
            )

    final_metrics = evaluate_policy(
        agent=agent,
        records_by_name=records_by_name,
        args=args,
        names=("train", "val", "test", "full"),
    )
    save_agent_checkpoint(final_path, agent, args=args, stage=stage_name, step=stage_seen, metrics=final_metrics)
    save_json(stage_dir / "final_summary.json", {"best_checkpoint": str(best_path), "final_checkpoint": str(final_path), "metrics": final_metrics})
    elapsed = time.time() - start
    if stage_name == "pure_ddqn_200k":
        setattr(args, "_pure_ddqn_elapsed_sec", float(elapsed))
    cprint(
        GREEN,
        f"[{stage_name} DONE] final_test_ratio={final_metrics['test']['total_cost_ratio']:.6f} "
        f"imp={final_metrics['test']['total_cost_improvement_pct']:.3f}% checkpoint={final_path}",
    )
    return final_path, final_metrics


def prepare_records(args: argparse.Namespace, train_config: dict, runtime, run_dir: Path) -> tuple[argparse.Namespace, dict[str, RouteRecords]]:
    train_eval_args = configure_eval_args(args, train_config, run_dir / "config_train", "train")
    val_eval_args = configure_eval_args(args, train_config, run_dir / "config_val", "val")
    test_eval_args = configure_eval_args(args, train_config, run_dir / "config_test", "test")
    eval_args_by_split = {"train": train_eval_args, "val": val_eval_args, "test": test_eval_args}
    records_by_name: dict[str, RouteRecords] = {}
    for split in ("train", "val", "test"):
        records_by_name[split] = build_route_records(
            split=split,
            runtime=runtime,
            eval_args=eval_args_by_split[split],
            args=args,
            run_dir=run_dir,
        )
    records_by_name["full"] = concat_records("full_year_train_val_test", [records_by_name["train"], records_by_name["val"], records_by_name["test"]])
    return train_eval_args, records_by_name


def baseline_report(records_by_name: dict[str, RouteRecords], args: argparse.Namespace) -> dict:
    report = {}
    for name, records in records_by_name.items():
        zeros = np.zeros(records.n, dtype=np.int16)
        oracle = records.arrays["oracle_action"].astype(np.int16)
        target = records.arrays["target_action"].astype(np.int16)
        report[name] = {
            "dijkstra": stats_for_actions(records, zeros, args),
            "candidate_oracle": stats_for_actions(records, oracle, args),
            "supervised_label_target": stats_for_actions(records, target, args),
        }
    return report


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_config = load_train_config(args.train_config)
    apply_train_config_defaults(args, train_config)
    args.embed_dim = int(value(args, train_config, "embed_dim") or args.embed_dim)
    args.hidden_dim = int(value(args, train_config, "hidden_dim") or args.hidden_dim)
    args.lr = float(value(args, train_config, "lr") or args.lr)
    args.gamma = float(value(args, train_config, "gamma") or 0.0)

    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    print(f"{BOLD}{CYAN}{'=' * 118}{RESET}", flush=True)
    cprint(CYAN, "Full-year supervised reranker + pure DDQN + supervised warm-start DDQN suite")
    cprint(CYAN, f"run_dir={run_dir}")
    cprint(CYAN, f"device={device} train_config={args.train_config}")
    cprint(CYAN, f"episodes: pure={args.pure_ddqn_episodes:,} warm={args.warm_ddqn_episodes:,} supervised_epochs={args.supervised_epochs}")
    print(f"{BOLD}{CYAN}{'=' * 118}{RESET}", flush=True)

    runtime = build_runtime(args, train_config, device)
    eval_args, records_by_name = prepare_records(args, train_config, runtime, run_dir)
    records_report = {name: rec.meta for name, rec in records_by_name.items()}
    save_json(run_dir / "records_summary.json", records_report)
    base = baseline_report(records_by_name, args)
    save_json(run_dir / "baseline_reference.json", base)
    for name in ("train", "val", "test", "full"):
        rec = records_by_name[name]
        cprint(
            YELLOW,
            f"[REFERENCE {name}] records={rec.n:,} weight={int(np.sum(rec.arrays['weights'])):,} "
            f"oracle_ratio={base[name]['candidate_oracle']['total_cost_ratio']:.6f} "
            f"target_ratio={base[name]['supervised_label_target']['total_cost_ratio']:.6f} "
            f"opp={base[name]['dijkstra']['opportunity_rate']:.2f}%",
        )

    total_ddqn_episodes = int(args.pure_ddqn_episodes) + int(args.warm_ddqn_episodes)
    supervised_agent = create_agent(args, runtime, eval_args, epsilon_start=0.0)
    supervised_ckpt, supervised_metrics = train_supervised_reranker(
        agent=supervised_agent,
        records_by_name=records_by_name,
        args=args,
        run_dir=run_dir,
    )

    pure_agent = create_agent(args, runtime, eval_args, epsilon_start=float(args.epsilon_start))
    pure_ckpt, pure_metrics = train_ddqn_stage(
        stage_name="pure_ddqn_200k",
        agent=pure_agent,
        records_by_name=records_by_name,
        args=args,
        run_dir=run_dir,
        total_ddqn_episodes=total_ddqn_episodes,
        ddqn_episode_offset=0,
        episodes=int(args.pure_ddqn_episodes),
    )

    warm_agent = create_agent(args, runtime, eval_args, epsilon_start=float(args.epsilon_start))
    load_policy_checkpoint(supervised_ckpt, warm_agent, load_optimizer=False)
    warm_agent.optimizer = optim.Adam(warm_agent.online_net.parameters(), lr=float(args.warm_lr))
    warm_agent.epsilon = float(args.epsilon_start)
    warm_agent.learn_step_count = 0
    warm_ckpt, warm_metrics = train_ddqn_stage(
        stage_name="warm_start_ddqn_200k",
        agent=warm_agent,
        records_by_name=records_by_name,
        args=args,
        run_dir=run_dir,
        total_ddqn_episodes=total_ddqn_episodes,
        ddqn_episode_offset=int(args.pure_ddqn_episodes),
        episodes=int(args.warm_ddqn_episodes),
    )

    summary = {
        "run_dir": str(run_dir),
        "train_config": str(args.train_config),
        "device": str(device),
        "records": records_report,
        "baseline_reference": base,
        "stages": {
            "supervised_reranker": {"checkpoint": str(supervised_ckpt), "metrics": supervised_metrics},
            "pure_ddqn_200k": {"checkpoint": str(pure_ckpt), "metrics": pure_metrics},
            "warm_start_ddqn_200k": {"checkpoint": str(warm_ckpt), "metrics": warm_metrics},
        },
        "outputs": {
            "records_summary": str(run_dir / "records_summary.json"),
            "baseline_reference": str(run_dir / "baseline_reference.json"),
            "summary": str(run_dir / "full_year_training_suite_summary.json"),
        },
    }
    save_json(run_dir / "full_year_training_suite_summary.json", summary)
    print(f"{BOLD}{GREEN}{'=' * 118}{RESET}", flush=True)
    cprint(GREEN, f"Saved summary: {run_dir / 'full_year_training_suite_summary.json'}")
    for stage_name, metrics in [
        ("supervised_reranker", supervised_metrics),
        ("pure_ddqn_200k", pure_metrics),
        ("warm_start_ddqn_200k", warm_metrics),
    ]:
        test = metrics["test"]
        cprint(
            GREEN if test["total_cost_ratio"] < 1.0 else YELLOW,
            f"{stage_name}: test_ratio={test['total_cost_ratio']:.6f} improve={test['total_cost_improvement_pct']:.3f}% "
            f"benP={test['benefit_precision']:.1f}% benR={test['benefit_recall']:.1f}% "
            f"bad5={test['bad_gt_5pct']:.3f}% act={test['activation_rate']:.2f}%",
        )
    print(f"{BOLD}{GREEN}{'=' * 118}{RESET}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-year route-reranker training suite.")
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=231)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--superzone-dir", type=str, default=None)
    p.add_argument("--edge-length-source", type=str, default=None, choices=[None, "osrm", "centroid"])
    p.add_argument("--pred-source", type=str, default=None, choices=[None, "persistence", "stgat"])
    p.add_argument("--stgat-ckpt-dir", type=str, default=None)
    p.add_argument("--dispatch-source", type=str, default=None, choices=[None, "persistence", "real_future_oracle"])
    p.add_argument("--max-time-steps", type=int, default=0)
    p.add_argument("--hist-len", type=int, default=None)
    p.add_argument("--pred-horizon", type=int, default=None)
    p.add_argument("--time-slot-minutes", type=float, default=None)
    p.add_argument("--k-routes", type=int, default=None)
    p.add_argument("--min-unique-routes", type=int, default=1)
    p.add_argument("--min-pred-hops", type=int, default=0)
    p.add_argument("--min-pred-distance-km", type=float, default=0.0)
    p.add_argument("--require-candidate-oracle-diff", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-candidate-oracle-improvement-ratio", type=float, default=1.0)
    p.add_argument("--profile-batch-size", type=int, default=24)
    p.add_argument("--candidate-build-batch-size", type=int, default=24)
    p.add_argument("--candidate-build-log-interval", type=int, default=500)
    p.add_argument("--candidate-pool-attempt-multiplier", type=int, default=1)
    p.add_argument("--record-log-interval", type=int, default=5000)
    p.add_argument("--max-windows-per-split", type=int, default=0)
    p.add_argument("--rebuild-records", action="store_true")
    p.add_argument("--feature-set", type=str, default="base", choices=[None, "base", "uncertainty", "pattern_topk"])
    p.add_argument("--pattern-topk", type=int, default=None)
    p.add_argument("--pattern-attention-temperature", type=float, default=None)
    p.add_argument("--max-neighbors", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warm-lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--reward-mode", type=str, default=None, choices=[None, "shaped", "direct_regret", "time_only"])
    p.add_argument("--reward-scale", type=float, default=None)
    p.add_argument("--oracle-regret-weight", type=float, default=None)
    p.add_argument("--win-bonus", type=float, default=None)
    p.add_argument("--oracle-bonus", type=float, default=None)
    p.add_argument("--feature-time-norm", type=float, default=None)
    p.add_argument("--feature-distance-norm", type=float, default=None)
    p.add_argument("--feature-hop-norm", type=float, default=None)
    p.add_argument("--feature-speed-norm", type=float, default=None)
    p.add_argument("--feature-log-count-norm", type=float, default=None)
    p.add_argument("--benefit-delta", type=float, default=0.005)
    p.add_argument("--bad-delta", type=float, default=0.05)
    p.add_argument("--minor-bad-delta", type=float, default=0.01)
    p.add_argument("--ddqn-bad-penalty", type=float, default=2.5)
    p.add_argument("--ddqn-minor-bad-penalty", type=float, default=0.25)
    p.add_argument("--sample-weight-mode", type=str, default="sqrt_count", choices=["none", "sqrt_count", "count"])
    p.add_argument("--supervised-epochs", type=int, default=20)
    p.add_argument("--supervised-batch-size", type=int, default=1024)
    p.add_argument("--supervised-log-interval", type=int, default=100)
    p.add_argument("--supervised-positive-weight", type=float, default=8.0)
    p.add_argument("--supervised-reward-loss-weight", type=float, default=0.15)
    p.add_argument("--pure-ddqn-episodes", type=int, default=200000)
    p.add_argument("--warm-ddqn-episodes", type=int, default=200000)
    p.add_argument("--ddqn-batch-size", type=int, default=256)
    p.add_argument("--ddqn-log-interval", type=int, default=5000)
    p.add_argument("--eval-batch-size", type=int, default=8192)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=80000)
    p.add_argument("--target-update", type=int, default=5000)
    p.add_argument("--grad-clip", type=float, default=10.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
