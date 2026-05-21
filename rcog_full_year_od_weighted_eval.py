from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from candidate_route_reranker_experiment import (
    CandidateDoubleDQNAgent,
    RouteChoiceCandidate,
    build_state_and_mask,
    choose_reward,
    make_route_choice_candidate,
)
from dispatch import build_dispatch_od_pairs, greedy_dispatch
from evaluate_candidate_route_reranker import load_train_config, value
from gated_candidate_route_reranker_experiment import build_runtime, eval_args_from_config
from no_topk_rcog_gate_experiment import build_agent, q_summary, rcog_feature
from rcog_stability_validation import cprint, parse_seed_list
from real_data_uncertain_routing_experiment import configure_runtime, format_seconds, resolve_device, route_windows_to_profiles_batch


RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RED = "\033[31m"


@dataclass
class CompactRecords:
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


def configure_eval_args(args: argparse.Namespace, train_config: dict, run_dir: Path, split: str) -> argparse.Namespace:
    eval_args = eval_args_from_config(
        args,
        train_config,
        run_dir=run_dir,
        split=split,
        pool_size=0,
    )
    eval_args.feature_set = "base"
    eval_args.k_routes = int(value(args, train_config, "k_routes") or 6)
    eval_args.min_unique_routes = int(args.min_unique_routes)
    eval_args.min_pred_hops = int(args.min_pred_hops)
    eval_args.min_pred_distance_km = float(args.min_pred_distance_km)
    eval_args.require_candidate_oracle_diff = False
    eval_args.min_candidate_oracle_improvement_ratio = 1.0
    return eval_args


def apply_train_config_defaults(args: argparse.Namespace, train_config: dict) -> None:
    args.k_routes = int(value(args, train_config, "k_routes") or 6)
    args.min_unique_routes = int(args.min_unique_routes)
    args.min_pred_hops = int(args.min_pred_hops)
    args.min_pred_distance_km = float(args.min_pred_distance_km)
    args.require_candidate_oracle_diff = False
    args.min_candidate_oracle_improvement_ratio = 1.0
    args.candidate_build_batch_size = int(args.profile_batch_size)
    args.feature_set = "base"
    args.reward_mode = value(args, train_config, "reward_mode") or "direct_regret"
    for name, fallback in [
        ("hist_len", 14),
        ("pred_horizon", 4),
        ("time_slot_minutes", 15.0),
        ("alpha", 1.0),
        ("beta", 0.1),
        ("rho", 50.0),
        ("reward_scale", 10.0),
        ("oracle_regret_weight", 5.0),
        ("win_bonus", 1.0),
        ("oracle_bonus", 1.0),
        ("feature_time_norm", 2.0),
        ("feature_distance_norm", 20.0),
        ("feature_hop_norm", 12.0),
        ("feature_speed_norm", 130.0),
        ("feature_log_count_norm", 6.0),
        ("max_neighbors", 10),
        ("embed_dim", 16),
        ("hidden_dim", 128),
        ("lr", 1e-3),
        ("gamma", 0.0),
    ]:
        if getattr(args, name, None) is None:
            raw = value(args, train_config, name)
            setattr(args, name, fallback if raw is None else raw)


def dispatch_tasks_for_split(
    *,
    runtime,
    split: str,
    hist_len: int,
    dispatch_source: str,
    max_windows: int,
) -> np.ndarray:
    windows = np.asarray(runtime.split_indices[split], dtype=np.int32)
    if max_windows > 0:
        windows = windows[: int(max_windows)]
    tasks: list[tuple[int, int, int, int, int, int]] = []
    demand_all = runtime.artifacts["region_demand"]
    supply_all = runtime.artifacts["region_supply"]
    travel_time = runtime.artifacts["dispatch_duration_hours"]
    for window_idx in windows:
        target_t = int(window_idx) + int(hist_len)
        if dispatch_source == "persistence":
            dispatch_t = target_t - 1
        elif dispatch_source == "real_future_oracle":
            dispatch_t = target_t
        else:
            raise ValueError(f"Unsupported dispatch source: {dispatch_source}")
        demand = np.maximum(demand_all[dispatch_t], 0.0)
        supply = np.maximum(supply_all[dispatch_t], 0.0)
        matrix = greedy_dispatch(demand, supply, travel_time, skip_unreachable=True)
        for origin, dest, count in build_dispatch_od_pairs(matrix):
            if int(count) > 0:
                tasks.append((int(window_idx), int(target_t), int(dispatch_t), int(origin), int(dest), int(count)))
    if not tasks:
        return np.zeros((0, 6), dtype=np.int32)
    return np.asarray(tasks, dtype=np.int32)


def masked_q_values_batch(agent: CandidateDoubleDQNAgent, states: np.ndarray, masks: np.ndarray) -> np.ndarray:
    if states.size == 0:
        return np.zeros((0, masks.shape[1] if masks.ndim == 2 else 0), dtype=np.float32)
    with torch.no_grad():
        s = torch.tensor(states, dtype=torch.float32, device=agent.device)
        q = agent.online_net(s).detach().cpu().numpy().astype(np.float32)
    return np.where(masks > 0, q, -np.inf).astype(np.float32)


def split_task_batches(tasks: np.ndarray, profile_batch_size: int):
    if tasks.size == 0:
        return
    windows = tasks[:, 0]
    unique_windows, start_indices = np.unique(windows, return_index=True)
    order = np.argsort(start_indices)
    unique_windows = unique_windows[order]
    start_indices = start_indices[order]
    end_indices = np.r_[start_indices[1:], len(tasks)]
    for offset in range(0, len(unique_windows), int(profile_batch_size)):
        chunk_windows = unique_windows[offset : offset + int(profile_batch_size)]
        chunk_starts = start_indices[offset : offset + int(profile_batch_size)]
        chunk_ends = end_indices[offset : offset + int(profile_batch_size)]
        yield chunk_windows, chunk_starts, chunk_ends


def load_compact_records(cache_path: Path, meta_path: Path, split: str) -> CompactRecords:
    loaded = np.load(cache_path, allow_pickle=False)
    arrays = {key: loaded[key] for key in loaded.files}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return CompactRecords(split=split, arrays=arrays, meta=meta)


def save_compact_records(records: CompactRecords, cache_path: Path, meta_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **records.arrays)
    meta_path.write_text(json.dumps(records.meta, indent=2), encoding="utf-8")


def build_compact_records(
    *,
    split: str,
    runtime,
    eval_args: argparse.Namespace,
    agent: CandidateDoubleDQNAgent,
    args: argparse.Namespace,
    run_dir: Path,
) -> CompactRecords:
    split_dir = run_dir / f"records_{split}"
    cache_path = split_dir / "compact_records.npz"
    meta_path = split_dir / "compact_records_meta.json"
    if cache_path.exists() and meta_path.exists() and not bool(args.rebuild_records):
        rec = load_compact_records(cache_path, meta_path, split)
        cprint(GREEN, f"[CACHE] loaded {split}: records={rec.n:,} weight={int(np.sum(rec.arrays['weights'])):,}")
        return rec

    split_dir.mkdir(parents=True, exist_ok=True)
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

    features: list[np.ndarray] = []
    weights: list[int] = []
    raw_action: list[int] = []
    raw_time: list[float] = []
    baseline_time: list[float] = []
    oracle_time: list[float] = []
    dijkstra_real_time: list[float] = []
    raw_reward: list[float] = []
    dijkstra_reward: list[float] = []
    raw_pick_oracle: list[int] = []
    dijkstra_pick_oracle: list[int] = []
    pred_is_oracle: list[int] = []
    opportunity_no_delta: list[int] = []
    opportunity_label: list[int] = []
    benefit_label: list[int] = []
    bad_label: list[int] = []
    unique_routes: list[int] = []
    q_margin: list[float] = []
    q_top2_margin: list[float] = []
    q_entropy: list[float] = []
    origin_arr: list[int] = []
    dest_arr: list[int] = []
    window_arr: list[int] = []
    target_arr: list[int] = []
    skipped = 0
    skipped_weight = 0
    accepted_weight = 0
    start = time.time()
    next_log = int(args.record_log_interval)

    for batch_idx, (batch_windows, starts, ends) in enumerate(split_task_batches(tasks, int(args.profile_batch_size)), start=1):
        pred_profiles, real_profiles = route_windows_to_profiles_batch(
            window_indices=[int(x) for x in batch_windows],
            hist_len=int(eval_args.hist_len),
            pred_horizon=int(eval_args.pred_horizon),
            edge_speeds=runtime.edge_speeds,
            provider=runtime.provider,
            artifacts=runtime.artifacts,
        )
        batch_cands: list[RouteChoiceCandidate] = []
        batch_states: list[np.ndarray] = []
        batch_masks: list[np.ndarray] = []
        batch_meta = []
        for window_idx, start_i, end_i, pred_profile, real_profile in zip(batch_windows, starts, ends, pred_profiles, real_profiles, strict=True):
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
                batch_cands.append(cand)
                batch_states.append(state)
                batch_masks.append(mask)
                batch_meta.append(meta_row)
        if batch_cands:
            states = np.stack(batch_states).astype(np.float32)
            masks = np.stack(batch_masks).astype(np.float32)
            q_values_batch = masked_q_values_batch(agent, states, masks)
            for cand, meta_row, state, mask, q_values in zip(batch_cands, batch_meta, batch_states, batch_masks, q_values_batch, strict=True):
                action = int(np.argmax(q_values))
                action = int(np.clip(action, 0, len(cand.candidates) - 1))
                raw_route = cand.candidates[action]
                dij_route = cand.candidates[0]
                baseline = float(cand.dijkstra_pred.real_result.travel_time)
                oracle = float(cand.candidate_oracle.real_result.travel_time)
                raw_t = float(raw_route.real_result.travel_time)
                qs = q_summary(q_values, mask, action)
                features.append(
                    rcog_feature(
                        cand,
                        meta_row,
                        q_values,
                        mask,
                        action,
                        edge_speeds=runtime.edge_speeds,
                        edge_lengths=runtime.edge_lengths,
                        num_nodes=runtime.num_nodes,
                        hist_len=int(eval_args.hist_len),
                        time_slot_minutes=float(eval_args.time_slot_minutes),
                    ).astype(np.float32)
                )
                weights.append(int(cand.dispatch_count))
                raw_action.append(action)
                raw_time.append(raw_t)
                baseline_time.append(baseline)
                oracle_time.append(oracle)
                dijkstra_real_time.append(float(cand.dijkstra_real.travel_time))
                raw_reward.append(float(choose_reward(cand, raw_route, eval_args)))
                dijkstra_reward.append(float(choose_reward(cand, dij_route, eval_args)))
                raw_pick_oracle.append(int(raw_route.path == cand.candidate_oracle.path))
                dijkstra_pick_oracle.append(int(dij_route.path == cand.candidate_oracle.path))
                pred_is_oracle.append(int(cand.dijkstra_pred.path == cand.candidate_oracle.path))
                opportunity_no_delta.append(int(oracle < baseline))
                opportunity_label.append(int(oracle < baseline * (1.0 - float(args.benefit_delta))))
                benefit_label.append(int(action != 0 and raw_t < baseline * (1.0 - float(args.benefit_delta))))
                bad_label.append(int(action != 0 and raw_t > baseline * (1.0 + float(args.bad_delta))))
                unique_routes.append(int(len(cand.candidates)))
                q_margin.append(float(qs["q_margin"]))
                q_top2_margin.append(float(qs["q_top2_margin"]))
                q_entropy.append(float(qs["q_entropy"]))
                origin_arr.append(int(cand.origin))
                dest_arr.append(int(cand.dest))
                window_arr.append(int(cand.window_idx))
                target_arr.append(int(cand.target_t))
                accepted_weight += int(cand.dispatch_count)

        if len(weights) >= next_log or batch_idx == math.ceil(max(len(np.unique(tasks[:, 0])), 1) / max(int(args.profile_batch_size), 1)):
            elapsed = time.time() - start
            progress = (len(weights) + skipped) / max(total_tasks, 1)
            eta = elapsed * (1.0 - progress) / max(progress, 1e-9)
            opp_rate = (np.average(opportunity_label, weights=np.asarray(weights)) * 100.0) if weights else 0.0
            cprint(
                BLUE,
                f"[BUILD:{split}] processed={len(weights)+skipped:,}/{total_tasks:,} "
                f"records={len(weights):,} skipped={skipped:,} weight={accepted_weight:,}/{total_weight:,} "
                f"opp={opp_rate:5.2f}% elapsed={format_seconds(elapsed)} ETA={format_seconds(eta)}",
            )
            next_log += int(args.record_log_interval)

    if not features:
        raise RuntimeError(f"No valid RCOG records built for split={split}")

    arrays = {
        "features": np.stack(features).astype(np.float32),
        "weights": np.asarray(weights, dtype=np.float64),
        "raw_action": np.asarray(raw_action, dtype=np.int16),
        "raw_action_nonzero": (np.asarray(raw_action, dtype=np.int16) != 0),
        "raw_time": np.asarray(raw_time, dtype=np.float64),
        "baseline_time": np.asarray(baseline_time, dtype=np.float64),
        "oracle_time": np.asarray(oracle_time, dtype=np.float64),
        "dijkstra_real_time": np.asarray(dijkstra_real_time, dtype=np.float64),
        "raw_reward": np.asarray(raw_reward, dtype=np.float64),
        "dijkstra_reward": np.asarray(dijkstra_reward, dtype=np.float64),
        "raw_pick_oracle": np.asarray(raw_pick_oracle, dtype=bool),
        "dijkstra_pick_oracle": np.asarray(dijkstra_pick_oracle, dtype=bool),
        "pred_is_oracle": np.asarray(pred_is_oracle, dtype=bool),
        "opportunity_no_delta": np.asarray(opportunity_no_delta, dtype=bool),
        "opportunity_label": np.asarray(opportunity_label, dtype=bool),
        "benefit_label": np.asarray(benefit_label, dtype=bool),
        "bad_label": np.asarray(bad_label, dtype=bool),
        "unique_routes": np.asarray(unique_routes, dtype=np.int16),
        "q_margin": np.asarray(q_margin, dtype=np.float32),
        "q_top2_margin": np.asarray(q_top2_margin, dtype=np.float32),
        "q_entropy": np.asarray(q_entropy, dtype=np.float32),
        "origin": np.asarray(origin_arr, dtype=np.int16),
        "dest": np.asarray(dest_arr, dtype=np.int16),
        "window_idx": np.asarray(window_arr, dtype=np.int32),
        "target_t": np.asarray(target_arr, dtype=np.int32),
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
        "feature_dim": int(arrays["features"].shape[1]),
        "min_unique_routes": int(eval_args.min_unique_routes),
        "min_pred_hops": int(eval_args.min_pred_hops),
        "min_pred_distance_km": float(eval_args.min_pred_distance_km),
    }
    records = CompactRecords(split=split, arrays=arrays, meta=meta)
    save_compact_records(records, cache_path, meta_path)
    cprint(
        GREEN,
        f"[CACHE] saved {split}: records={records.n:,} weight={int(np.sum(arrays['weights'])):,} "
        f"coverage={meta['coverage_by_weight']*100:.2f}% elapsed={format_seconds(meta['elapsed_sec'])}",
    )
    return records


def concat_records(name: str, parts: list[CompactRecords]) -> CompactRecords:
    keys = parts[0].arrays.keys()
    arrays = {key: np.concatenate([p.arrays[key] for p in parts], axis=0) for key in keys}
    meta = {
        "split": name,
        "parts": [p.meta for p in parts],
        "records": int(arrays["weights"].shape[0]),
        "accepted_dispatch_weight": int(np.sum(arrays["weights"])),
        "total_tasks": int(sum(int(p.meta.get("total_tasks", 0)) for p in parts)),
        "total_dispatch_weight": int(sum(int(p.meta.get("total_dispatch_weight", 0)) for p in parts)),
    }
    meta["coverage_by_weight"] = float(meta["accepted_dispatch_weight"] / max(meta["total_dispatch_weight"], 1))
    return CompactRecords(split=name, arrays=arrays, meta=meta)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(values * weights) / denom)


def gate_from_config(records: CompactRecords, probs: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    score = probs["opportunity"] * probs["benefit"] * (1.0 - probs["bad"])
    gate = (
        (score >= float(cfg["score_threshold"]))
        & (probs["opportunity"] >= float(cfg["p_o_min"]))
        & (probs["benefit"] >= float(cfg["p_b_min"]))
        & (probs["bad"] <= float(cfg["p_d_max"]))
        & (records.arrays["q_margin"] >= float(cfg["q_margin_min"]))
    )
    return gate & records.arrays["raw_action_nonzero"]


def stats_for_gate(records: CompactRecords, gate: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    arr = records.arrays
    if mask is None:
        mask = np.ones(records.n, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return {
            "num_records": 0,
            "dispatch_weight": 0.0,
            "ddqn_over_pred": float("nan"),
            "total_cost_ratio": float("nan"),
            "activation_rate": 0.0,
            "opportunity_recall": 0.0,
            "benefit_precision": 0.0,
            "benefit_recall": 0.0,
            "bad_gt_5pct": 0.0,
        }
    w = arr["weights"][mask].astype(np.float64)
    activated_all = np.asarray(gate, dtype=bool) & arr["raw_action_nonzero"]
    activated = activated_all[mask]
    chosen_time = np.where(activated, arr["raw_time"][mask], arr["baseline_time"][mask])
    chosen_reward = np.where(activated, arr["raw_reward"][mask], arr["dijkstra_reward"][mask])
    chosen_action = np.where(activated, arr["raw_action"][mask], 0)
    baseline = arr["baseline_time"][mask]
    oracle = arr["oracle_time"][mask]
    opp = arr["opportunity_label"][mask]
    opp_no_delta = arr["opportunity_no_delta"][mask]
    benefit = arr["benefit_label"][mask]
    bad = arr["bad_label"][mask]
    raw_pick = arr["raw_pick_oracle"][mask]
    dij_pick = arr["dijkstra_pick_oracle"][mask]
    pick_oracle = np.where(activated, raw_pick, dij_pick)
    pred_is_oracle = arr["pred_is_oracle"][mask]
    denom = max(float(np.sum(w)), 1e-12)
    active_w = w[activated]
    opp_w = w[opp]
    benefit_w = w[benefit]
    return {
        "num_records": int(np.sum(mask)),
        "dispatch_weight": float(np.sum(w)),
        "ddqn_time": weighted_mean(chosen_time, w),
        "dijkstra_pred_time": weighted_mean(baseline, w),
        "candidate_oracle_time": weighted_mean(oracle, w),
        "dijkstra_real_time": weighted_mean(arr["dijkstra_real_time"][mask], w),
        "ddqn_over_pred": weighted_mean(chosen_time / np.maximum(baseline, 1e-9), w),
        "total_cost_ratio": float(np.sum(chosen_time * w) / max(float(np.sum(baseline * w)), 1e-12)),
        "ddqn_over_candidate_oracle": weighted_mean(chosen_time / np.maximum(oracle, 1e-9), w),
        "candidate_oracle_over_pred": weighted_mean(oracle / np.maximum(baseline, 1e-9), w),
        "ddqn_win_rate": float(np.sum(w[chosen_time < baseline]) / denom * 100.0),
        "oracle_pick_rate": float(np.sum(w[pick_oracle]) / denom * 100.0),
        "pred_already_oracle_rate": float(np.sum(w[pred_is_oracle]) / denom * 100.0),
        "opportunity_rate": float(np.sum(w[opp_no_delta]) / denom * 100.0),
        "opportunity_label_rate": float(np.sum(w[opp]) / denom * 100.0),
        "avg_reward": weighted_mean(chosen_reward, w),
        "avg_action": weighted_mean(chosen_action.astype(np.float64), w),
        "avg_unique_routes": weighted_mean(arr["unique_routes"][mask].astype(np.float64), w),
        "activation_rate": float(np.sum(active_w) / denom * 100.0),
        "opportunity_precision": float(np.sum(w[activated & opp]) / max(float(np.sum(active_w)), 1e-12) * 100.0) if np.any(activated) else 0.0,
        "opportunity_recall": float(np.sum(w[activated & opp]) / max(float(np.sum(opp_w)), 1e-12) * 100.0) if np.any(opp) else 0.0,
        "benefit_precision": float(np.sum(w[activated & benefit]) / max(float(np.sum(active_w)), 1e-12) * 100.0) if np.any(activated) else 0.0,
        "benefit_recall": float(np.sum(w[activated & benefit]) / max(float(np.sum(benefit_w)), 1e-12) * 100.0) if np.any(benefit) else 0.0,
        "bad_override_precision": float(np.sum(w[activated & bad]) / max(float(np.sum(active_w)), 1e-12) * 100.0) if np.any(activated) else 0.0,
        "bad_gt_1pct": float(np.sum(w[chosen_time > baseline * 1.01]) / denom * 100.0),
        "bad_gt_5pct": float(np.sum(w[chosen_time > baseline * 1.05]) / denom * 100.0),
        "mean_improvement_pct": float((1.0 - weighted_mean(chosen_time / np.maximum(baseline, 1e-9), w)) * 100.0),
        "total_cost_improvement_pct": float((1.0 - (np.sum(chosen_time * w) / max(float(np.sum(baseline * w)), 1e-12))) * 100.0),
    }


def fit_hgb_weighted(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    positive_weight_mult: float,
    negative_weight_mult: float,
    base_weights: np.ndarray | None,
) -> HistGradientBoostingClassifier:
    y = y.astype(np.int32)
    sample_weight = np.ones_like(y, dtype=np.float64)
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos > 0 and neg > 0:
        sample_weight[y == 1] = min(neg / max(pos, 1.0), 50.0) * float(positive_weight_mult)
        sample_weight[y == 0] = float(negative_weight_mult)
    if base_weights is not None:
        bw = np.asarray(base_weights, dtype=np.float64)
        sample_weight = sample_weight * np.maximum(bw, 1.0)
        sample_weight = sample_weight / max(float(np.mean(sample_weight)), 1e-12)
    clf = HistGradientBoostingClassifier(
        max_iter=int(max_iter),
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.01,
        random_state=int(seed),
    )
    clf.fit(x, y, sample_weight=sample_weight)
    return clf


def train_indices_for_seed(seed: int, n: int, args: argparse.Namespace) -> tuple[np.ndarray | None, str]:
    if args.train_resample_mode == "bootstrap":
        rng = np.random.RandomState(int(seed) + 991)
        return rng.randint(0, n, size=n), "bootstrap"
    if args.train_resample_mode == "subsample":
        rng = np.random.RandomState(int(seed) + 991)
        sample_n = max(1, int(round(n * float(args.train_subsample_frac))))
        return rng.choice(n, size=sample_n, replace=False), f"subsample_{float(args.train_subsample_frac):.2f}"
    return None, "full"


def fit_models(records: CompactRecords, seed: int, args: argparse.Namespace, train_indices: np.ndarray | None) -> dict:
    idx = train_indices if train_indices is not None else np.arange(records.n)
    x = records.arrays["features"][idx].astype(np.float32)
    base_weights = records.arrays["weights"][idx] if str(args.train_weight_mode) == "count" else None
    labels = {
        "opportunity": records.arrays["opportunity_label"][idx].astype(np.int32),
        "benefit": records.arrays["benefit_label"][idx].astype(np.int32),
        "bad": records.arrays["bad_label"][idx].astype(np.int32),
    }
    weight_cfg = {
        "opportunity": (float(args.opportunity_positive_weight_mult), float(args.opportunity_negative_weight_mult)),
        "benefit": (float(args.benefit_positive_weight_mult), float(args.benefit_negative_weight_mult)),
        "bad": (float(args.bad_positive_weight_mult), float(args.bad_negative_weight_mult)),
    }
    return {
        name: fit_hgb_weighted(
            x,
            y,
            seed=int(seed) + head_idx * 17,
            max_iter=int(args.model_iter),
            positive_weight_mult=weight_cfg[name][0],
            negative_weight_mult=weight_cfg[name][1],
            base_weights=base_weights,
        )
        for head_idx, (name, y) in enumerate(labels.items())
    }


def predict_prob(clf, x: np.ndarray) -> np.ndarray:
    return clf.predict_proba(x)[:, 1].astype(np.float32)


def make_probs(models: dict, records: CompactRecords) -> dict[str, np.ndarray]:
    x = records.arrays["features"].astype(np.float32)
    return {name: predict_prob(model, x) for name, model in models.items()}


def threshold_grid(score: np.ndarray, q_margin: np.ndarray, dense: bool) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
    if dense:
        score_thresholds = sorted(
            set(
                [0.0]
                + [float(x) for x in np.linspace(0.04, 0.24, 41)]
                + [float(np.quantile(score, q)) for q in np.linspace(0.40, 0.995, 60)]
            )
        )
        p_o_mins = [float(x) for x in np.round(np.arange(0.20, 0.551, 0.05), 2)]
        p_b_mins = [float(x) for x in np.round(np.arange(0.20, 0.601, 0.05), 2)]
        p_d_maxs = [float(x) for x in np.round(np.arange(0.10, 0.351, 0.05), 2)]
    else:
        score_thresholds = sorted(set([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9] + [float(np.quantile(score, q)) for q in np.linspace(0.50, 0.995, 30)]))
        p_o_mins = [0.0, 0.3, 0.5, 0.7]
        p_b_mins = [0.0, 0.3, 0.5, 0.7]
        p_d_maxs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]
    q_thresholds = [-1e9, 0.0]
    finite_q = q_margin[np.isfinite(q_margin)]
    if finite_q.size:
        q_thresholds.extend(float(np.quantile(finite_q, q)) for q in [0.5, 0.7, 0.8, 0.9])
    return score_thresholds, p_o_mins, p_b_mins, p_d_maxs, sorted(set(q_thresholds))


def select_config(records: CompactRecords, probs: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[dict, list[dict]]:
    score = probs["opportunity"] * probs["benefit"] * (1.0 - probs["bad"])
    score_thresholds, p_o_mins, p_b_mins, p_d_maxs, q_thresholds = threshold_grid(
        score,
        records.arrays["q_margin"],
        dense=str(args.threshold_search_grid) == "dense",
    )
    results: list[dict] = []
    total = len(score_thresholds) * len(p_o_mins) * len(p_b_mins) * len(p_d_maxs) * len(q_thresholds)
    done = 0
    start = time.time()
    for st in score_thresholds:
        for po in p_o_mins:
            for pb in p_b_mins:
                for pdmax in p_d_maxs:
                    for qmin in q_thresholds:
                        cfg = {
                            "score_threshold": float(st),
                            "p_o_min": float(po),
                            "p_b_min": float(pb),
                            "p_d_max": float(pdmax),
                            "q_margin_min": float(qmin),
                        }
                        stats = stats_for_gate(records, gate_from_config(records, probs, cfg))
                        results.append({**cfg, **stats})
                        done += 1
        if done and done % max(int(args.sweep_log_interval), 1) == 0:
            progress = done / max(total, 1)
            eta = (time.time() - start) * (1.0 - progress) / max(progress, 1e-9)
            cprint(CYAN, f"[SWEEP] {done:,}/{total:,} elapsed={format_seconds(time.time()-start)} ETA={format_seconds(eta)}")

    def feasible(item: dict, strict: bool) -> bool:
        if float(item["bad_gt_5pct"]) > float(args.max_bad_gt_5pct):
            return False
        if float(item["bad_gt_1pct"]) > float(args.max_bad_gt_1pct):
            return False
        if strict and float(item["activation_rate"]) < float(args.min_activation):
            return False
        if strict and float(item["activation_rate"]) > float(args.max_activation):
            return False
        if strict and float(item["benefit_precision"]) < float(args.min_benefit_precision):
            return False
        if strict and float(item["benefit_recall"]) < float(args.min_benefit_recall):
            return False
        if strict and float(item["opportunity_recall"]) < float(args.min_opportunity_recall):
            return False
        return True

    strict_pool = [x for x in results if feasible(x, True)]
    pool = strict_pool or [x for x in results if feasible(x, False)] or results
    objective = str(args.selection_objective)
    if objective == "precision":
        pool.sort(
            key=lambda x: (
                -float(x.get("benefit_precision", 0.0)),
                -float(x.get("benefit_recall", 0.0)),
                float(x.get("bad_gt_5pct", float("inf"))),
                float(x.get("total_cost_ratio", float("inf"))),
            )
        )
    elif objective == "ratio":
        pool.sort(
            key=lambda x: (
                float(x.get("total_cost_ratio", float("inf"))),
                -float(x.get("benefit_precision", 0.0)),
                -float(x.get("benefit_recall", 0.0)),
                float(x.get("bad_gt_5pct", float("inf"))),
            )
        )
    else:
        pool.sort(
            key=lambda x: (
                -float(x.get("benefit_recall", 0.0)),
                -float(x.get("benefit_precision", 0.0)),
                float(x.get("bad_gt_1pct", float("inf"))),
                float(x.get("bad_gt_5pct", float("inf"))),
                float(x.get("total_cost_ratio", float("inf"))),
            )
        )
    results.sort(key=lambda x: float(x.get("total_cost_ratio", float("inf"))))
    return pool[0], results[: int(args.save_sweep_top_k)]


def bootstrap_ci(records: CompactRecords, gate: np.ndarray, *, seed: int, iters: int) -> dict[str, float]:
    if iters <= 0:
        return {}
    rng = np.random.RandomState(int(seed))
    n = records.n
    ratios = []
    improvements = []
    for _ in range(int(iters)):
        idx = rng.randint(0, n, size=n)
        sub_arrays = {k: v[idx] for k, v in records.arrays.items()}
        sub = CompactRecords(split=f"{records.split}_bootstrap", arrays=sub_arrays, meta={})
        sub_gate = gate[idx]
        stats = stats_for_gate(sub, sub_gate)
        ratios.append(float(stats["total_cost_ratio"]))
        improvements.append(float(stats["total_cost_improvement_pct"]))
    ratios_arr = np.asarray(ratios, dtype=np.float64)
    imp_arr = np.asarray(improvements, dtype=np.float64)
    return {
        "total_cost_ratio": float(np.mean(ratios_arr)),
        "total_cost_ratio_ci95_low": float(np.quantile(ratios_arr, 0.025)),
        "total_cost_ratio_ci95_high": float(np.quantile(ratios_arr, 0.975)),
        "total_cost_improvement_pct": float(np.mean(imp_arr)),
        "total_cost_improvement_pct_ci95_low": float(np.quantile(imp_arr, 0.025)),
        "total_cost_improvement_pct_ci95_high": float(np.quantile(imp_arr, 0.975)),
        "prob_total_cost_ratio_below_1": float(np.mean(ratios_arr < 1.0)),
        "bootstrap_iters": int(iters),
    }


def flatten(prefix: str, values: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in values.items()}


def aggregate_seed_rows(rows: list[dict], target_bad5: float) -> dict:
    def arr(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in rows if key in r and r[key] not in ("", None)], dtype=np.float64)

    ratio = arr("test_total_cost_ratio")
    mean_ratio = arr("test_ddqn_over_pred")
    benp = arr("test_benefit_precision")
    benr = arr("test_benefit_recall")
    bad5 = arr("test_bad_gt_5pct")
    act = arr("test_activation_rate")
    opp_ratio = arr("test_opp_total_cost_ratio")
    full_ratio = arr("full_total_cost_ratio")
    ci_low = arr("ci_total_cost_improvement_pct_ci95_low")
    return {
        "variant": "RCOG",
        "seeds": len(rows),
        "test_total_cost_ratio_mean": float(np.mean(ratio)),
        "test_total_cost_ratio_std": float(np.std(ratio, ddof=1)) if ratio.size > 1 else 0.0,
        "test_mean_ratio_mean": float(np.mean(mean_ratio)),
        "test_total_cost_improvement_pct_mean": float((1.0 - np.mean(ratio)) * 100.0),
        "benefit_precision_mean": float(np.mean(benp)),
        "benefit_recall_mean": float(np.mean(benr)),
        "bad_gt_5pct_mean": float(np.mean(bad5)),
        "bad_gt_5pct_max": float(np.max(bad5)),
        "activation_mean": float(np.mean(act)),
        "opportunity_total_cost_ratio_mean": float(np.mean(opp_ratio)),
        "full_year_total_cost_ratio_mean": float(np.mean(full_ratio)),
        "ci_improvement_low_min": float(np.min(ci_low)) if ci_low.size else float("nan"),
        "safety_pass": bool(np.max(bad5) <= float(target_bad5)),
        "positive_ci_pass": bool(np.min(ci_low) > 0.0) if ci_low.size else False,
    }


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_config = load_train_config(args.train_config)
    apply_train_config_defaults(args, train_config)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))

    print(f"{BOLD}{CYAN}{'=' * 118}{RESET}", flush=True)
    cprint(CYAN, "RCOG full-year OD-time weighted evaluation")
    cprint(CYAN, f"run_dir={run_dir}")
    cprint(CYAN, f"device={device} seeds={args.seeds} train_weight_mode={args.train_weight_mode}")
    cprint(CYAN, f"constraints: cap={args.max_bad_gt_5pct}% act<={args.max_activation}% benP>={args.min_benefit_precision}% bad_weight={args.bad_positive_weight_mult}")
    print(f"{BOLD}{CYAN}{'=' * 118}{RESET}", flush=True)

    runtime = build_runtime(args, train_config, device)
    train_eval_args = configure_eval_args(args, train_config, run_dir / "config_train", "train")
    val_eval_args = configure_eval_args(args, train_config, run_dir / "config_val", "val")
    test_eval_args = configure_eval_args(args, train_config, run_dir / "config_test", "test")
    agent = build_agent(args, train_config, train_eval_args, runtime)

    start_all = time.time()
    train_records = build_compact_records(split="train", runtime=runtime, eval_args=train_eval_args, agent=agent, args=args, run_dir=run_dir)
    val_records = build_compact_records(split="val", runtime=runtime, eval_args=val_eval_args, agent=agent, args=args, run_dir=run_dir)
    test_records = build_compact_records(split="test", runtime=runtime, eval_args=test_eval_args, agent=agent, args=args, run_dir=run_dir)
    full_records = concat_records("full_year_train_val_test", [train_records, val_records, test_records])

    cprint(MAGENTA, "[REFERENCE]")
    for name, rec in [("train", train_records), ("val", val_records), ("test", test_records), ("full", full_records)]:
        raw_gate = rec.arrays["raw_action_nonzero"]
        raw_stats = stats_for_gate(rec, raw_gate)
        cprint(
            YELLOW,
            f"{name:<5} records={rec.n:,} weight={int(np.sum(rec.arrays['weights'])):,} "
            f"raw_total_ratio={raw_stats['total_cost_ratio']:.6f} raw_benP={raw_stats['benefit_precision']:.1f}% "
            f"raw_bad5={raw_stats['bad_gt_5pct']:.2f}%",
        )

    seeds = parse_seed_list(args.seeds)
    seed_rows: list[dict] = []
    reports: list[dict] = []
    model_start_all = time.time()
    total_seed_jobs = len(seeds)

    for job_idx, seed in enumerate(seeds, start=1):
        seed_start = time.time()
        train_indices, train_mode = train_indices_for_seed(seed, train_records.n, args)
        cprint(BLUE, f"[SEED {job_idx}/{total_seed_jobs}] seed={seed} mode={train_mode} fitting RCOG heads")
        models = fit_models(train_records, int(seed), args, train_indices)
        val_probs = make_probs(models, val_records)
        cfg, sweep_top = select_config(val_records, val_probs, args)
        test_probs = make_probs(models, test_records)
        full_probs = make_probs(models, full_records)

        test_gate = gate_from_config(test_records, test_probs, cfg)
        full_gate = gate_from_config(full_records, full_probs, cfg)
        test_stats = stats_for_gate(test_records, test_gate)
        test_opp_stats = stats_for_gate(test_records, test_gate, test_records.arrays["opportunity_label"])
        test_noopp_stats = stats_for_gate(test_records, test_gate, ~test_records.arrays["opportunity_label"])
        full_stats = stats_for_gate(full_records, full_gate)
        full_opp_stats = stats_for_gate(full_records, full_gate, full_records.arrays["opportunity_label"])
        ci = bootstrap_ci(test_records, test_gate, seed=int(seed) + 6007, iters=int(args.bootstrap_iters))
        row = {
            "seed": int(seed),
            "train_mode": train_mode,
            "selected_score_threshold": float(cfg["score_threshold"]),
            "selected_p_o_min": float(cfg["p_o_min"]),
            "selected_p_b_min": float(cfg["p_b_min"]),
            "selected_p_d_max": float(cfg["p_d_max"]),
            "selected_q_margin_min": float(cfg["q_margin_min"]),
            **flatten("val_selected", cfg),
            **flatten("test", test_stats),
            **flatten("test_opp", test_opp_stats),
            **flatten("test_noopp", test_noopp_stats),
            **flatten("full", full_stats),
            **flatten("full_opp", full_opp_stats),
            **flatten("ci", ci),
        }
        seed_rows.append(row)
        report = {
            "seed": int(seed),
            "train_mode": train_mode,
            "selected_config": cfg,
            "test": test_stats,
            "test_opportunity": test_opp_stats,
            "test_no_opportunity": test_noopp_stats,
            "full_year": full_stats,
            "full_year_opportunity": full_opp_stats,
            "bootstrap_ci": ci,
            "sweep_top": sweep_top,
        }
        reports.append(report)
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_rows(run_dir / "seed_metrics_partial.csv", seed_rows)
        progress = job_idx / max(total_seed_jobs, 1)
        elapsed = time.time() - model_start_all
        eta = elapsed * (1.0 - progress) / max(progress, 1e-9)
        cprint(
            GREEN if test_stats["total_cost_ratio"] < 1.0 else YELLOW,
            f"[RESULT seed={seed}] test_total_ratio={test_stats['total_cost_ratio']:.6f} "
            f"test_mean_ratio={test_stats['ddqn_over_pred']:.6f} improve={test_stats['total_cost_improvement_pct']:.3f}% "
            f"benP={test_stats['benefit_precision']:.1f}% benR={test_stats['benefit_recall']:.1f}% "
            f"bad5={test_stats['bad_gt_5pct']:.3f}% act={test_stats['activation_rate']:.2f}% "
            f"opp_ratio={test_opp_stats['total_cost_ratio']:.6f} full_ratio={full_stats['total_cost_ratio']:.6f}",
        )
        cprint(CYAN, f"[TOTAL] seeds={job_idx}/{total_seed_jobs} seed_elapsed={format_seconds(time.time()-seed_start)} total_ETA={format_seconds(eta)}")

    aggregate = aggregate_seed_rows(seed_rows, target_bad5=float(args.target_bad_gt_5pct))
    write_rows(run_dir / "seed_metrics.csv", seed_rows)
    write_rows(run_dir / "aggregate_metrics.csv", [aggregate])
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "seeds": seeds,
        "train_resample_mode": str(args.train_resample_mode),
        "train_weight_mode": str(args.train_weight_mode),
        "selection": {
            "variant_name": "RCOG",
            "max_bad_gt_5pct": float(args.max_bad_gt_5pct),
            "max_bad_gt_1pct": float(args.max_bad_gt_1pct),
            "max_activation": float(args.max_activation),
            "min_activation": float(args.min_activation),
            "min_benefit_precision": float(args.min_benefit_precision),
            "selection_objective": str(args.selection_objective),
            "bad_positive_weight_mult": float(args.bad_positive_weight_mult),
        },
        "records": {
            "train": train_records.meta,
            "val": val_records.meta,
            "test": test_records.meta,
            "full_year": full_records.meta,
        },
        "aggregate": aggregate,
        "reports": reports,
        "outputs": {
            "seed_metrics": str(run_dir / "seed_metrics.csv"),
            "aggregate_metrics": str(run_dir / "aggregate_metrics.csv"),
        },
        "elapsed_sec": float(time.time() - start_all),
    }
    out_path = run_dir / "rcog_full_year_od_weighted_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{BOLD}{GREEN}{'=' * 118}{RESET}", flush=True)
    cprint(GREEN, f"Saved summary: {out_path}")
    cprint(
        GREEN if aggregate["safety_pass"] and aggregate["positive_ci_pass"] else YELLOW,
        f"RCOG 5-seed aggregate | test_total_ratio={aggregate['test_total_cost_ratio_mean']:.6f} "
        f"improve={aggregate['test_total_cost_improvement_pct_mean']:.3f}% "
        f"benP={aggregate['benefit_precision_mean']:.1f}% benR={aggregate['benefit_recall_mean']:.1f}% "
        f"bad5={aggregate['bad_gt_5pct_mean']:.3f}/{aggregate['bad_gt_5pct_max']:.3f}% "
        f"act={aggregate['activation_mean']:.2f}% full_ratio={aggregate['full_year_total_cost_ratio_mean']:.6f}",
    )
    print(f"{BOLD}{GREEN}{'=' * 118}{RESET}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-year OD-time weighted RCOG evaluation.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--seeds", type=str, default="121,122,123,124,125")
    p.add_argument("--train-resample-mode", type=str, default="bootstrap", choices=["full", "bootstrap", "subsample"])
    p.add_argument("--train-subsample-frac", type=float, default=0.8)
    p.add_argument("--train-weight-mode", type=str, default="none", choices=["none", "count"])
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--model-iter", type=int, default=160)
    p.add_argument("--benefit-delta", type=float, default=0.005)
    p.add_argument("--bad-delta", type=float, default=0.05)
    p.add_argument("--max-bad-gt-5pct", type=float, default=0.30)
    p.add_argument("--target-bad-gt-5pct", type=float, default=0.50)
    p.add_argument("--max-bad-gt-1pct", type=float, default=3.0)
    p.add_argument("--min-activation", type=float, default=5.0)
    p.add_argument("--max-activation", type=float, default=11.0)
    p.add_argument("--min-benefit-precision", type=float, default=64.0)
    p.add_argument("--min-benefit-recall", type=float, default=0.0)
    p.add_argument("--min-opportunity-recall", type=float, default=0.0)
    p.add_argument("--selection-objective", type=str, default="balanced", choices=["balanced", "precision", "ratio"])
    p.add_argument("--threshold-search-grid", type=str, default="dense", choices=["coarse", "dense"])
    p.add_argument("--opportunity-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--opportunity-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--benefit-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--benefit-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--bad-positive-weight-mult", type=float, default=2.0)
    p.add_argument("--bad-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--profile-batch-size", type=int, default=24)
    p.add_argument("--candidate-build-batch-size", type=int, default=24)
    p.add_argument("--candidate-build-log-interval", type=int, default=500)
    p.add_argument("--candidate-pool-attempt-multiplier", type=int, default=1)
    p.add_argument("--record-log-interval", type=int, default=5000)
    p.add_argument("--sweep-log-interval", type=int, default=20000)
    p.add_argument("--save-sweep-top-k", type=int, default=100)
    p.add_argument("--max-windows-per-split", type=int, default=0)
    p.add_argument("--rebuild-records", action="store_true")
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
    p.add_argument("--min-unique-routes", type=int, default=1)
    p.add_argument("--min-pred-hops", type=int, default=0)
    p.add_argument("--min-pred-distance-km", type=float, default=0.0)
    p.add_argument("--require-candidate-oracle-diff", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--min-candidate-oracle-improvement-ratio", type=float, default=1.0)
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
    p.add_argument("--reward-mode", type=str, default=None, choices=[None, "shaped", "direct_regret", "time_only"])
    p.add_argument("--oracle-regret-weight", type=float, default=None)
    p.add_argument("--win-bonus", type=float, default=None)
    p.add_argument("--oracle-bonus", type=float, default=None)
    p.add_argument("--feature-time-norm", type=float, default=None)
    p.add_argument("--feature-distance-norm", type=float, default=None)
    p.add_argument("--feature-hop-norm", type=float, default=None)
    p.add_argument("--feature-speed-norm", type=float, default=None)
    p.add_argument("--feature-log-count-norm", type=float, default=None)
    p.add_argument("--feature-set", type=str, default="base", choices=[None, "base", "uncertainty", "pattern_topk"])
    p.add_argument("--pattern-topk", type=int, default=None)
    p.add_argument("--pattern-attention-temperature", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
