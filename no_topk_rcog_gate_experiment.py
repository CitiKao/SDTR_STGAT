from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from candidate_route_reranker_experiment import (
    CandidateDoubleDQNAgent,
    RouteChoiceCandidate,
    build_state_and_mask,
    choose_reward,
    row_from_choice,
    state_dim_for_args,
    summarize,
    write_csv,
)
from evaluate_candidate_route_reranker import load_train_config, value
from gated_candidate_route_reranker_experiment import (
    build_pool_for_split,
    build_runtime,
    eval_args_from_config,
    masked_q_values,
    softmax_masked,
)
from real_data_uncertain_routing_experiment import configure_runtime, format_seconds, resolve_device


@dataclass
class RCOGRecord:
    cand: RouteChoiceCandidate
    meta_row: pd.Series
    state: np.ndarray
    mask: np.ndarray
    q_values: np.ndarray
    raw_action: int
    feature: np.ndarray
    opportunity_label: int
    benefit_label: int
    bad_override_label: int


@dataclass
class SearchArrays:
    raw_action: np.ndarray
    raw_action_nonzero: np.ndarray
    baseline_time: np.ndarray
    raw_time: np.ndarray
    oracle_time: np.ndarray
    dijkstra_real_time: np.ndarray
    raw_reward: np.ndarray
    dijkstra_reward: np.ndarray
    raw_pick_oracle: np.ndarray
    dijkstra_pick_oracle: np.ndarray
    pred_is_oracle: np.ndarray
    opportunity_no_delta: np.ndarray
    opportunity_label: np.ndarray
    benefit_label: np.ndarray
    bad_label: np.ndarray
    unique_routes: np.ndarray


def edge_set(route) -> set[int]:
    return set(int(x) for x in route.edge_ids)


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def route_edge_time_stats(route, pred_profile: np.ndarray, edge_lengths: np.ndarray) -> dict[str, float]:
    if not route.edge_ids:
        return {
            "bottleneck_frac": 0.0,
            "edge_time_cv": 0.0,
            "edge_time_max": 0.0,
            "edge_time_mean": 0.0,
        }
    eids = np.asarray(route.edge_ids, dtype=np.int64)
    speed0 = np.maximum(pred_profile[eids, 0].astype(np.float64), 1e-5)
    lengths = edge_lengths[eids].astype(np.float64)
    edge_times = lengths / speed0
    total = float(np.sum(edge_times))
    return {
        "bottleneck_frac": float(np.max(edge_times) / max(total, 1e-9)),
        "edge_time_cv": float(np.std(edge_times) / max(float(np.mean(edge_times)), 1e-9)),
        "edge_time_max": float(np.max(edge_times)),
        "edge_time_mean": float(np.mean(edge_times)),
    }


def route_recent_speed_stats(
    route,
    edge_speeds: np.ndarray,
    *,
    target_t: int,
    hist_len: int,
) -> dict[str, float]:
    if not route.edge_ids:
        return {
            "recent_speed_mean": 0.0,
            "recent_speed_std": 0.0,
            "recent_speed_delta": 0.0,
            "recent_speed_cv": 0.0,
        }
    start = max(0, int(target_t) - int(hist_len))
    end = max(start + 1, int(target_t))
    eids = np.asarray(route.edge_ids, dtype=np.int64)
    hist = np.asarray(edge_speeds[start:end, :][:, eids], dtype=np.float64)
    mean = float(np.mean(hist))
    std = float(np.std(hist))
    if hist.shape[0] >= 2:
        delta = float(np.mean(hist[-1] - hist[0]))
    else:
        delta = 0.0
    return {
        "recent_speed_mean": mean,
        "recent_speed_std": std,
        "recent_speed_delta": delta,
        "recent_speed_cv": float(std / max(abs(mean), 1e-9)),
    }


def route_persistence_time(
    route,
    edge_speeds: np.ndarray,
    edge_lengths: np.ndarray,
    *,
    target_t: int,
) -> float:
    if not route.edge_ids:
        return 0.0
    hist_t = max(0, int(target_t) - 1)
    eids = np.asarray(route.edge_ids, dtype=np.int64)
    speeds = np.maximum(np.asarray(edge_speeds[hist_t, eids], dtype=np.float64), 1e-5)
    lengths = edge_lengths[eids].astype(np.float64)
    return float(np.sum(lengths / speeds))


def q_summary(q_values: np.ndarray, mask: np.ndarray, raw_action: int) -> dict[str, float]:
    q_clean = np.where(np.isfinite(q_values), q_values, 0.0).astype(np.float64)
    probs = softmax_masked(q_values, mask)
    valid = q_values[mask > 0]
    if valid.size >= 2:
        sorted_q = np.sort(valid.astype(np.float64))
        top2_margin = float(sorted_q[-1] - sorted_q[-2])
    else:
        top2_margin = 0.0
    q0 = float(q_clean[0]) if q_clean.size else 0.0
    qbest = float(q_clean[raw_action]) if 0 <= raw_action < q_clean.size else q0
    entropy = -float(np.sum(probs[probs > 0] * np.log(np.maximum(probs[probs > 0], 1e-12))))
    return {
        "q_best": qbest,
        "q_dijkstra": q0,
        "q_margin": qbest - q0,
        "q_top2_margin": top2_margin,
        "q_best_prob": float(np.max(probs)) if probs.size else 0.0,
        "q_dijkstra_prob": float(probs[0]) if probs.size else 0.0,
        "q_entropy": entropy,
        "raw_action_nonzero": float(raw_action != 0),
    }


def rcog_feature(
    cand: RouteChoiceCandidate,
    meta_row: pd.Series,
    q_values: np.ndarray,
    mask: np.ndarray,
    raw_action: int,
    *,
    edge_speeds: np.ndarray,
    edge_lengths: np.ndarray,
    num_nodes: int,
    hist_len: int,
    time_slot_minutes: float,
) -> np.ndarray:
    k = len(cand.candidates)
    d0 = cand.dijkstra_pred
    raw_route = cand.candidates[int(np.clip(raw_action, 0, k - 1))]
    pred_times = np.asarray([float(r.pred_result.travel_time) for r in cand.candidates], dtype=np.float64)
    pred_dists = np.asarray([float(r.pred_result.travel_dist) for r in cand.candidates], dtype=np.float64)
    pred_hops = np.asarray([float(r.pred_result.steps) for r in cand.candidates], dtype=np.float64)
    t0 = max(float(pred_times[0]), 1e-9)
    rel_times = pred_times / t0
    gaps = rel_times[1:] - 1.0 if rel_times.size > 1 else np.zeros(1, dtype=np.float64)

    d0_edges = edge_set(d0)
    raw_edges = edge_set(raw_route)
    overlaps = np.asarray([jaccard(d0_edges, edge_set(r)) for r in cand.candidates], dtype=np.float64)
    alt_overlaps = overlaps[1:] if overlaps.size > 1 else np.asarray([1.0])
    close_alt = rel_times <= 1.05
    close_alt[0] = False
    close_diversity = float(np.max(1.0 - overlaps[close_alt])) if np.any(close_alt) else 0.0

    d0_edge_stats = route_edge_time_stats(d0, cand.pred_profile, edge_lengths)
    raw_edge_stats = route_edge_time_stats(raw_route, cand.pred_profile, edge_lengths)
    d0_recent = route_recent_speed_stats(d0, edge_speeds, target_t=cand.target_t, hist_len=hist_len)
    raw_recent = route_recent_speed_stats(raw_route, edge_speeds, target_t=cand.target_t, hist_len=hist_len)
    d0_persist_time = route_persistence_time(d0, edge_speeds, edge_lengths, target_t=cand.target_t)
    raw_persist_time = route_persistence_time(raw_route, edge_speeds, edge_lengths, target_t=cand.target_t)
    qstats = q_summary(q_values, mask, raw_action)

    slots_per_day = max(int(round((24 * 60) / max(time_slot_minutes, 1e-9))), 1)
    hour = float(meta_row.get("hour", 0.0))
    slot = float(meta_row.get("slot", 0.0))
    hour_angle = 2.0 * math.pi * hour / 24.0
    slot_angle = 2.0 * math.pi * slot / max(slots_per_day, 1)
    node_norm = max(float(num_nodes - 1), 1.0)

    feature = [
        float(cand.origin) / node_norm,
        float(cand.dest) / node_norm,
        math.sin(hour_angle),
        math.cos(hour_angle),
        math.sin(slot_angle),
        math.cos(slot_angle),
        math.log1p(max(float(cand.dispatch_count), 0.0)) / 6.0,
        t0,
        float(pred_dists[0]),
        float(pred_hops[0]),
        float(gaps[0]) if gaps.size >= 1 else 0.0,
        float(gaps[1]) if gaps.size >= 2 else float(gaps[0]) if gaps.size else 0.0,
        float(np.min(gaps)) if gaps.size else 0.0,
        float(np.mean(gaps)) if gaps.size else 0.0,
        float(np.std(rel_times)),
        float(np.max(rel_times) - np.min(rel_times)),
        float(np.min(alt_overlaps)),
        float(np.mean(alt_overlaps)),
        float(1.0 - jaccard(d0_edges, raw_edges)),
        close_diversity,
        float(d0.min_pred_speed),
        float(d0.mean_pred_speed),
        float(d0.pred_time_std),
        float(d0.pred_speed_std),
        float(d0.low_speed_frac),
        float(d0.pred_time_trend_ratio),
        d0_edge_stats["bottleneck_frac"],
        d0_edge_stats["edge_time_cv"],
        d0_recent["recent_speed_mean"],
        d0_recent["recent_speed_std"],
        d0_recent["recent_speed_delta"],
        d0_recent["recent_speed_cv"],
        float(d0_persist_time / max(float(d0.pred_result.travel_time), 1e-9)),
        float(abs(d0_persist_time - float(d0.pred_result.travel_time)) / max(float(d0.pred_result.travel_time), 1e-9)),
        float(raw_action) / max(float(max(k - 1, 1)), 1.0),
        float(raw_route.pred_result.travel_time / t0),
        float(raw_route.pred_result.travel_dist / max(float(d0.pred_result.travel_dist), 1e-9)),
        float(raw_route.pred_result.steps / max(float(d0.pred_result.steps), 1.0)),
        float(raw_route.min_pred_speed - d0.min_pred_speed) / 130.0,
        float(raw_route.mean_pred_speed - d0.mean_pred_speed) / 130.0,
        float(raw_route.pred_time_std - d0.pred_time_std),
        float(raw_route.low_speed_frac - d0.low_speed_frac),
        raw_edge_stats["bottleneck_frac"] - d0_edge_stats["bottleneck_frac"],
        raw_recent["recent_speed_std"] - d0_recent["recent_speed_std"],
        float(raw_persist_time / max(float(raw_route.pred_result.travel_time), 1e-9)),
        float(abs(raw_persist_time - float(raw_route.pred_result.travel_time)) / max(float(raw_route.pred_result.travel_time), 1e-9)),
        qstats["q_best"],
        qstats["q_dijkstra"],
        qstats["q_margin"],
        qstats["q_top2_margin"],
        qstats["q_best_prob"],
        qstats["q_dijkstra_prob"],
        qstats["q_entropy"],
        qstats["raw_action_nonzero"],
    ]
    return np.nan_to_num(np.asarray(feature, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


def build_agent(args: argparse.Namespace, train_config: dict, eval_args: argparse.Namespace, runtime) -> CandidateDoubleDQNAgent:
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
    if int(ckpt.get("state_dim", state_dim)) != state_dim:
        raise ValueError(f"Checkpoint state_dim {ckpt.get('state_dim')} != eval state_dim {state_dim}")
    agent.online_net.load_state_dict(ckpt["online_state_dict"])
    agent.target_net.load_state_dict(ckpt["target_state_dict"])
    agent.online_net.eval()
    return agent


def build_records(
    pool: list[RouteChoiceCandidate],
    *,
    runtime,
    eval_args: argparse.Namespace,
    agent: CandidateDoubleDQNAgent,
    benefit_delta: float,
    bad_delta: float,
) -> list[RCOGRecord]:
    records: list[RCOGRecord] = []
    for cand in pool:
        meta_row = runtime.time_meta.iloc[cand.target_t]
        state, mask = build_state_and_mask(cand, meta_row=meta_row, args=eval_args)
        q_values = masked_q_values(agent, state, mask)
        raw_action = int(np.argmax(q_values))
        raw_action = int(np.clip(raw_action, 0, len(cand.candidates) - 1))
        baseline = float(cand.dijkstra_pred.real_result.travel_time)
        oracle = float(cand.candidate_oracle.real_result.travel_time)
        raw_time = float(cand.candidates[raw_action].real_result.travel_time)
        feature = rcog_feature(
            cand,
            meta_row,
            q_values,
            mask,
            raw_action,
            edge_speeds=runtime.edge_speeds,
            edge_lengths=runtime.edge_lengths,
            num_nodes=runtime.num_nodes,
            hist_len=int(eval_args.hist_len),
            time_slot_minutes=float(eval_args.time_slot_minutes),
        )
        records.append(
            RCOGRecord(
                cand=cand,
                meta_row=meta_row,
                state=state,
                mask=mask,
                q_values=q_values,
                raw_action=raw_action,
                feature=feature,
                opportunity_label=int(oracle < baseline * (1.0 - benefit_delta)),
                benefit_label=int(raw_action != 0 and raw_time < baseline * (1.0 - benefit_delta)),
                bad_override_label=int(raw_action != 0 and raw_time > baseline * (1.0 + bad_delta)),
            )
        )
    return records


def fit_hgb_classifier(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    positive_weight_mult: float = 1.0,
    negative_weight_mult: float = 1.0,
) -> HistGradientBoostingClassifier:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    weights = np.ones_like(y, dtype=np.float64)
    if pos > 0 and neg > 0:
        weights[y == 1] = min(neg / max(pos, 1.0), 50.0) * float(positive_weight_mult)
        weights[y == 0] = float(negative_weight_mult)
    clf = HistGradientBoostingClassifier(
        max_iter=int(max_iter),
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.01,
        random_state=int(seed),
    )
    clf.fit(x, y, sample_weight=weights)
    return clf


def predict_prob(clf, x: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(x)[:, 1].astype(np.float32)
    return clf.predict(x).astype(np.float32)


def simple_gate_mask(probs: dict[str, np.ndarray], score: np.ndarray, q_margins: np.ndarray, cfg: dict) -> np.ndarray:
    return (
        (score >= float(cfg["score_threshold"]))
        & (probs["opportunity"] >= float(cfg["p_o_min"]))
        & (probs["benefit"] >= float(cfg["p_b_min"]))
        & (probs["bad"] <= float(cfg["p_d_max"]))
        & (q_margins >= float(cfg["q_margin_min"]))
    )


def gate_mask_for_config(records: list[RCOGRecord], probs: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    score = probs["opportunity"] * probs["benefit"] * (1.0 - probs["bad"])
    q_margins = np.asarray([q_summary(r.q_values, r.mask, r.raw_action)["q_margin"] for r in records], dtype=np.float32)
    raw_nonzero = np.asarray([r.raw_action != 0 for r in records], dtype=bool)
    if cfg.get("gate_mode") == "two_stage":
        gate = simple_gate_mask(probs, score, q_margins, cfg["core"]) | simple_gate_mask(probs, score, q_margins, cfg["recovery"])
    else:
        gate = simple_gate_mask(probs, score, q_margins, cfg)
    return gate & raw_nonzero


def actions_for_config(records: list[RCOGRecord], probs: dict[str, np.ndarray], cfg: dict) -> list[int]:
    gate = gate_mask_for_config(records, probs, cfg)
    return [int(rec.raw_action) if bool(activate) else 0 for rec, activate in zip(records, gate, strict=True)]


def rows_for_actions(records: list[RCOGRecord], actions: list[int], eval_args: argparse.Namespace, policy: str) -> list[dict]:
    rows = []
    for idx, (rec, action) in enumerate(zip(records, actions, strict=True), start=1):
        action = int(np.clip(action, 0, len(rec.cand.candidates) - 1))
        reward = choose_reward(rec.cand, rec.cand.candidates[action], eval_args)
        row = row_from_choice(
            episode=idx,
            cand=rec.cand,
            action=action,
            reward=reward,
            loss=None,
            epsilon=0.0,
            meta_row=rec.meta_row,
        )
        row["policy"] = policy
        rows.append(row)
    return rows


def extra_metrics(records: list[RCOGRecord], actions: list[int]) -> dict[str, float]:
    activated = np.asarray([int(a != 0) for a in actions], dtype=bool)
    opp = np.asarray([r.opportunity_label for r in records], dtype=bool)
    benefit = np.asarray([r.benefit_label for r in records], dtype=bool)
    bad = np.asarray([r.bad_override_label for r in records], dtype=bool)
    return {
        "activation_rate": float(np.mean(activated) * 100.0) if len(records) else 0.0,
        "opportunity_precision": float(np.mean(opp[activated]) * 100.0) if np.any(activated) else 0.0,
        "opportunity_recall": float(np.sum(opp & activated) / max(np.sum(opp), 1) * 100.0),
        "benefit_precision": float(np.mean(benefit[activated]) * 100.0) if np.any(activated) else 0.0,
        "benefit_recall": float(np.sum(benefit & activated) / max(np.sum(benefit), 1) * 100.0),
        "bad_override_precision": float(np.mean(bad[activated]) * 100.0) if np.any(activated) else 0.0,
    }


def evaluate_actions(
    records: list[RCOGRecord],
    actions: list[int],
    eval_args: argparse.Namespace,
    *,
    policy: str,
    out_csv: Path | None = None,
) -> dict:
    rows = rows_for_actions(records, actions, eval_args, policy)
    if out_csv is not None:
        write_csv(out_csv, rows)
    stats = summarize(rows)
    stats.update(extra_metrics(records, actions))
    stats["bad_gt_1pct"] = float(
        np.mean(
            [
                rec.cand.candidates[action].real_result.travel_time
                > rec.cand.dijkstra_pred.real_result.travel_time * 1.01
                for rec, action in zip(records, actions, strict=True)
            ]
        )
        * 100.0
    )
    stats["bad_gt_5pct"] = float(
        np.mean(
            [
                rec.cand.candidates[action].real_result.travel_time
                > rec.cand.dijkstra_pred.real_result.travel_time * 1.05
                for rec, action in zip(records, actions, strict=True)
            ]
        )
        * 100.0
    )
    return stats


def make_search_arrays(records: list[RCOGRecord], eval_args: argparse.Namespace) -> SearchArrays:
    raw_action = []
    raw_action_nonzero = []
    baseline_time = []
    raw_time = []
    oracle_time = []
    dijkstra_real_time = []
    raw_reward = []
    dijkstra_reward = []
    raw_pick_oracle = []
    dijkstra_pick_oracle = []
    pred_is_oracle = []
    opportunity_no_delta = []
    opportunity_label = []
    benefit_label = []
    bad_label = []
    unique_routes = []
    for rec in records:
        action = int(np.clip(rec.raw_action, 0, len(rec.cand.candidates) - 1))
        raw_route = rec.cand.candidates[action]
        dij_route = rec.cand.candidates[0]
        baseline = float(rec.cand.dijkstra_pred.real_result.travel_time)
        oracle = float(rec.cand.candidate_oracle.real_result.travel_time)
        raw_action.append(action)
        raw_action_nonzero.append(int(action != 0))
        baseline_time.append(baseline)
        raw_time.append(float(raw_route.real_result.travel_time))
        oracle_time.append(oracle)
        dijkstra_real_time.append(float(rec.cand.dijkstra_real.travel_time))
        raw_reward.append(float(choose_reward(rec.cand, raw_route, eval_args)))
        dijkstra_reward.append(float(choose_reward(rec.cand, dij_route, eval_args)))
        raw_pick_oracle.append(int(raw_route.path == rec.cand.candidate_oracle.path))
        dijkstra_pick_oracle.append(int(dij_route.path == rec.cand.candidate_oracle.path))
        pred_is_oracle.append(int(rec.cand.dijkstra_pred.path == rec.cand.candidate_oracle.path))
        opportunity_no_delta.append(int(oracle < baseline))
        opportunity_label.append(int(rec.opportunity_label))
        benefit_label.append(int(rec.benefit_label))
        bad_label.append(int(rec.bad_override_label))
        unique_routes.append(int(len(rec.cand.candidates)))
    return SearchArrays(
        raw_action=np.asarray(raw_action, dtype=np.int32),
        raw_action_nonzero=np.asarray(raw_action_nonzero, dtype=bool),
        baseline_time=np.asarray(baseline_time, dtype=np.float64),
        raw_time=np.asarray(raw_time, dtype=np.float64),
        oracle_time=np.asarray(oracle_time, dtype=np.float64),
        dijkstra_real_time=np.asarray(dijkstra_real_time, dtype=np.float64),
        raw_reward=np.asarray(raw_reward, dtype=np.float64),
        dijkstra_reward=np.asarray(dijkstra_reward, dtype=np.float64),
        raw_pick_oracle=np.asarray(raw_pick_oracle, dtype=np.float64),
        dijkstra_pick_oracle=np.asarray(dijkstra_pick_oracle, dtype=np.float64),
        pred_is_oracle=np.asarray(pred_is_oracle, dtype=np.float64),
        opportunity_no_delta=np.asarray(opportunity_no_delta, dtype=np.float64),
        opportunity_label=np.asarray(opportunity_label, dtype=bool),
        benefit_label=np.asarray(benefit_label, dtype=bool),
        bad_label=np.asarray(bad_label, dtype=bool),
        unique_routes=np.asarray(unique_routes, dtype=np.float64),
    )


def fast_stats_for_gate(gate: np.ndarray, arrays: SearchArrays) -> dict[str, float]:
    activated = np.asarray(gate, dtype=bool) & arrays.raw_action_nonzero
    chosen_time = np.where(activated, arrays.raw_time, arrays.baseline_time)
    chosen_reward = np.where(activated, arrays.raw_reward, arrays.dijkstra_reward)
    pick_oracle = np.where(activated, arrays.raw_pick_oracle, arrays.dijkstra_pick_oracle)
    actions = np.where(activated, arrays.raw_action, 0)
    baseline = arrays.baseline_time
    oracle = arrays.oracle_time
    return {
        "ddqn_time": float(np.mean(chosen_time)),
        "dijkstra_pred_time": float(np.mean(baseline)),
        "candidate_oracle_time": float(np.mean(oracle)),
        "dijkstra_real_time": float(np.mean(arrays.dijkstra_real_time)),
        "ddqn_over_pred": float(np.mean(chosen_time / np.maximum(baseline, 1e-9))),
        "ddqn_over_candidate_oracle": float(np.mean(chosen_time / np.maximum(oracle, 1e-9))),
        "candidate_oracle_over_pred": float(np.mean(oracle / np.maximum(baseline, 1e-9))),
        "ddqn_win_rate": float(np.mean(chosen_time < baseline) * 100.0),
        "oracle_pick_rate": float(np.mean(pick_oracle) * 100.0),
        "pred_already_oracle_rate": float(np.mean(arrays.pred_is_oracle) * 100.0),
        "opportunity_rate": float(np.mean(arrays.opportunity_no_delta) * 100.0),
        "avg_reward": float(np.mean(chosen_reward)),
        "avg_loss": float("nan"),
        "avg_action": float(np.mean(actions)),
        "avg_unique_routes": float(np.mean(arrays.unique_routes)),
        "activation_rate": float(np.mean(activated) * 100.0),
        "opportunity_precision": float(np.mean(arrays.opportunity_label[activated]) * 100.0) if np.any(activated) else 0.0,
        "opportunity_recall": float(np.sum(arrays.opportunity_label & activated) / max(np.sum(arrays.opportunity_label), 1) * 100.0),
        "benefit_precision": float(np.mean(arrays.benefit_label[activated]) * 100.0) if np.any(activated) else 0.0,
        "benefit_recall": float(np.sum(arrays.benefit_label & activated) / max(np.sum(arrays.benefit_label), 1) * 100.0),
        "bad_override_precision": float(np.mean(arrays.bad_label[activated]) * 100.0) if np.any(activated) else 0.0,
        "bad_gt_1pct": float(np.mean(chosen_time > baseline * 1.01) * 100.0),
        "bad_gt_5pct": float(np.mean(chosen_time > baseline * 1.05) * 100.0),
    }


def make_probs(models: dict, records: list[RCOGRecord]) -> dict[str, np.ndarray]:
    x = np.stack([r.feature for r in records]).astype(np.float32)
    return {name: predict_prob(model, x) for name, model in models.items()}


def select_gate_config(
    records: list[RCOGRecord],
    probs: dict[str, np.ndarray],
    eval_args: argparse.Namespace,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    score = probs["opportunity"] * probs["benefit"] * (1.0 - probs["bad"])
    q_margins = np.asarray([q_summary(r.q_values, r.mask, r.raw_action)["q_margin"] for r in records], dtype=np.float32)
    arrays = make_search_arrays(records, eval_args)
    if str(getattr(args, "threshold_search_grid", "coarse")) == "dense":
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
    nonzero_q = q_margins[np.isfinite(q_margins)]
    if nonzero_q.size:
        q_thresholds.extend(float(np.quantile(nonzero_q, q)) for q in [0.5, 0.7, 0.8, 0.9])
    q_thresholds = sorted(set(q_thresholds))

    results: list[dict] = []
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
                        gate = (
                            (score >= float(st))
                            & (probs["opportunity"] >= float(po))
                            & (probs["benefit"] >= float(pb))
                            & (probs["bad"] <= float(pdmax))
                            & (q_margins >= float(qmin))
                        )
                        stats = fast_stats_for_gate(gate, arrays)
                        item = {**cfg, **stats}
                        results.append(item)

    def feasible(item: dict, strict: bool) -> bool:
        if item["bad_gt_5pct"] > float(args.max_bad_gt_5pct):
            return False
        if item["bad_gt_1pct"] > float(args.max_bad_gt_1pct):
            return False
        if strict and item["activation_rate"] < float(args.min_activation):
            return False
        if strict and item["activation_rate"] > float(args.max_activation):
            return False
        if strict and item["benefit_precision"] < float(args.min_benefit_precision):
            return False
        if strict and item["benefit_recall"] < float(args.min_benefit_recall):
            return False
        min_opp_precision = float(getattr(args, "min_opportunity_precision", 0.0) or 0.0)
        min_opp_recall = float(getattr(args, "min_opportunity_recall", 0.0) or 0.0)
        if strict and min_opp_precision > 0.0 and item["opportunity_precision"] < min_opp_precision:
            return False
        if strict and min_opp_recall > 0.0 and item["opportunity_recall"] < min_opp_recall:
            return False
        return True

    strict_items = [x for x in results if feasible(x, True)]
    pool = strict_items or [x for x in results if feasible(x, False)] or results
    objective = str(getattr(args, "selection_objective", "time"))
    if objective == "two_stage_precision_at_recall":
        def cfg_only(item: dict) -> dict:
            return {
                "score_threshold": float(item["score_threshold"]),
                "p_o_min": float(item["p_o_min"]),
                "p_b_min": float(item["p_b_min"]),
                "p_d_max": float(item["p_d_max"]),
                "q_margin_min": float(item["q_margin_min"]),
            }

        core_candidates = [
            x
            for x in results
            if x["bad_gt_5pct"] <= float(args.max_bad_gt_5pct)
            and x["bad_gt_1pct"] <= float(args.max_bad_gt_1pct)
            and x["activation_rate"] <= float(args.two_stage_core_max_activation)
            and x["benefit_precision"] >= float(args.two_stage_core_min_precision)
        ]
        recovery_candidates = [
            x
            for x in results
            if x["bad_gt_5pct"] <= float(args.two_stage_recovery_max_bad_gt_5pct)
            and x["bad_gt_1pct"] <= float(args.two_stage_recovery_max_bad_gt_1pct)
            and x["activation_rate"] <= float(args.max_activation)
            and x["benefit_recall"] >= float(args.two_stage_recovery_min_recall)
        ]
        core_candidates.sort(
            key=lambda x: (
                -float(x["benefit_precision"]),
                -float(x["benefit_recall"]),
                float(x["bad_gt_1pct"]),
                float(x["activation_rate"]),
            )
        )
        recovery_candidates.sort(
            key=lambda x: (
                -float(x["benefit_recall"]),
                -float(x["benefit_precision"]),
                float(x["bad_gt_1pct"]),
                float(x["activation_rate"]),
            )
        )
        core_candidates = core_candidates[: int(args.two_stage_core_top_k)]
        recovery_candidates = recovery_candidates[: int(args.two_stage_recovery_top_k)]
        two_stage_results: list[dict] = []
        for core in core_candidates:
            core_cfg = cfg_only(core)
            core_gate = simple_gate_mask(probs, score, q_margins, core_cfg)
            for recovery in recovery_candidates:
                recovery_cfg = cfg_only(recovery)
                gate = core_gate | simple_gate_mask(probs, score, q_margins, recovery_cfg)
                stats = fast_stats_for_gate(gate, arrays)
                item = {
                    "gate_mode": "two_stage",
                    "core": core_cfg,
                    "recovery": recovery_cfg,
                    **stats,
                }
                two_stage_results.append(item)
        strict_two = [x for x in two_stage_results if feasible(x, True)]
        two_pool = strict_two or [x for x in two_stage_results if feasible(x, False)] or two_stage_results or pool
        two_pool.sort(
            key=lambda x: (
                -float(x.get("benefit_precision", 0.0)),
                -float(x.get("benefit_recall", 0.0)),
                float(x.get("bad_gt_1pct", float("inf"))),
                float(x.get("bad_gt_5pct", float("inf"))),
                float(x.get("ddqn_over_pred", float("inf"))),
            )
        )
        sweep_out = two_stage_results if bool(getattr(args, "save_full_sweep", False)) else two_pool[:200]
        return two_pool[0], sweep_out

    if objective == "constrained_balanced":
        pool.sort(
            key=lambda x: (
                -float(x.get("benefit_recall", 0.0)),
                -float(x.get("benefit_precision", 0.0)),
                float(x.get("bad_gt_1pct", float("inf"))),
                float(x.get("bad_gt_5pct", float("inf"))),
                float(x.get("ddqn_over_pred", float("inf"))),
            )
        )
    elif objective == "precision_at_recall":
        pool.sort(
            key=lambda x: (
                -float(x.get("benefit_precision", 0.0)),
                -float(x.get("benefit_recall", 0.0)),
                float(x.get("bad_gt_1pct", float("inf"))),
                float(x.get("bad_gt_5pct", float("inf"))),
                float(x.get("ddqn_over_pred", float("inf"))),
            )
        )
    elif objective == "benefit_recall":
        pool.sort(
            key=lambda x: (
                -float(x.get("benefit_recall", 0.0)),
                -float(x.get("benefit_precision", 0.0)),
                float(x.get("bad_gt_5pct", float("inf"))),
                float(x.get("ddqn_over_pred", float("inf"))),
            )
        )
    elif objective == "opportunity_recall":
        pool.sort(
            key=lambda x: (
                -float(x.get("opportunity_recall", 0.0)),
                -float(x.get("opportunity_precision", 0.0)),
                float(x.get("bad_gt_5pct", float("inf"))),
                float(x.get("ddqn_over_pred", float("inf"))),
            )
        )
    elif objective == "opportunity_precision_at_recall":
        pool.sort(
            key=lambda x: (
                -float(x.get("opportunity_precision", 0.0)),
                -float(x.get("opportunity_recall", 0.0)),
                -float(x.get("benefit_precision", 0.0)),
                -float(x.get("benefit_recall", 0.0)),
                float(x.get("bad_gt_1pct", float("inf"))),
                float(x.get("bad_gt_5pct", float("inf"))),
                float(x.get("ddqn_over_pred", float("inf"))),
            )
        )
    elif objective == "opportunity_balanced":
        pool.sort(
            key=lambda x: (
                -(
                    2.0
                    * float(x.get("opportunity_precision", 0.0))
                    * float(x.get("opportunity_recall", 0.0))
                    / max(float(x.get("opportunity_precision", 0.0)) + float(x.get("opportunity_recall", 0.0)), 1e-9)
                ),
                -float(x.get("opportunity_recall", 0.0)),
                -float(x.get("benefit_precision", 0.0)),
                float(x.get("bad_gt_5pct", float("inf"))),
                float(x.get("ddqn_over_pred", float("inf"))),
            )
        )
    else:
        pool.sort(
            key=lambda x: (
                float(x.get("ddqn_over_pred", float("inf"))),
                -float(x.get("benefit_precision", 0.0)),
                float(x.get("bad_gt_5pct", float("inf"))),
            )
        )
    if bool(getattr(args, "save_full_sweep", False)):
        sweep_out = results
    else:
        sweep_out = sorted(results, key=lambda x: float(x.get("ddqn_over_pred", float("inf"))))[:200]
    return pool[0], sweep_out


def run(args: argparse.Namespace) -> None:
    train_config = load_train_config(args.train_config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))
    runtime = build_runtime(args, train_config, device)

    print("=" * 96, flush=True)
    print("No-topk RCOG gate experiment", flush=True)
    print("Features: route margin/diversity/risk, forecast disagreement, recent volatility, DDQN Q confidence.", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"run_dir={run_dir} device={device}", flush=True)
    print("=" * 96, flush=True)

    train_eval_args = eval_args_from_config(
        args,
        train_config,
        run_dir=run_dir / "agent_config",
        split="train",
        pool_size=max(1, int(args.train_pool_size)),
    )
    train_eval_args.feature_set = "base"
    agent = build_agent(args, train_config, train_eval_args, runtime)
    labels = {}
    if args.load_rcog_models:
        model_path = Path(args.load_rcog_models)
        with model_path.open("rb") as f:
            models = pickle.load(f)
        print(f"Loaded RCOG heads from {model_path}", flush=True)
    else:
        train_pool, train_eval_args = build_pool_for_split(
            split="train",
            pool_size=int(args.train_pool_size),
            run_dir=run_dir,
            args=args,
            train_config=train_config,
            runtime=runtime,
            seed_offset=0,
        )
        train_eval_args.feature_set = "base"
        train_records = build_records(
            train_pool,
            runtime=runtime,
            eval_args=train_eval_args,
            agent=agent,
            benefit_delta=float(args.benefit_delta),
            bad_delta=float(args.bad_delta),
        )
        x_train = np.stack([r.feature for r in train_records]).astype(np.float32)
        labels = {
            "opportunity": np.asarray([r.opportunity_label for r in train_records], dtype=np.int32),
            "benefit": np.asarray([r.benefit_label for r in train_records], dtype=np.int32),
            "bad": np.asarray([r.bad_override_label for r in train_records], dtype=np.int32),
        }
        print(
            "Train label rates: "
            + ", ".join(f"{k}={float(v.mean())*100:.2f}%" for k, v in labels.items()),
            flush=True,
        )
        start = time.time()
        weight_cfg = {
            "opportunity": (float(args.opportunity_positive_weight_mult), float(args.opportunity_negative_weight_mult)),
            "benefit": (float(args.benefit_positive_weight_mult), float(args.benefit_negative_weight_mult)),
            "bad": (float(args.bad_positive_weight_mult), float(args.bad_negative_weight_mult)),
        }
        models = {
            name: fit_hgb_classifier(
                x_train,
                y,
                seed=int(args.seed) + idx * 17,
                max_iter=int(args.model_iter),
                positive_weight_mult=weight_cfg[name][0],
                negative_weight_mult=weight_cfg[name][1],
            )
            for idx, (name, y) in enumerate(labels.items())
        }
        print(f"Fitted RCOG heads in {format_seconds(time.time() - start)}", flush=True)
        with (run_dir / "rcog_models.pkl").open("wb") as f:
            pickle.dump(models, f)

    reports = {
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "train_pool_size": int(args.train_pool_size),
        "eval_pool_size": int(args.eval_pool_size),
        "feature_set": "no_topk_rcog",
        "label_rates": {k: float(v.mean()) for k, v in labels.items()},
        "selection_objective": str(args.selection_objective),
        "threshold_search_grid": str(args.threshold_search_grid),
        "weight_multipliers": {
            "opportunity_positive": float(args.opportunity_positive_weight_mult),
            "opportunity_negative": float(args.opportunity_negative_weight_mult),
            "benefit_positive": float(args.benefit_positive_weight_mult),
            "benefit_negative": float(args.benefit_negative_weight_mult),
            "bad_positive": float(args.bad_positive_weight_mult),
            "bad_negative": float(args.bad_negative_weight_mult),
        },
        "splits": {},
        "leakage_note": "Gate features use decision-time predictions/history/Q-values only. Realized future speeds are used only for train labels and final evaluation.",
    }

    val_pool, val_eval_args = build_pool_for_split(
        split="val",
        pool_size=int(args.eval_pool_size),
        run_dir=run_dir / "eval_val",
        args=args,
        train_config=train_config,
        runtime=runtime,
        seed_offset=1000,
    )
    val_eval_args.feature_set = "base"
    val_records = build_records(
        val_pool,
        runtime=runtime,
        eval_args=val_eval_args,
        agent=agent,
        benefit_delta=float(args.benefit_delta),
        bad_delta=float(args.bad_delta),
    )
    val_probs = make_probs(models, val_records)
    best_cfg, val_sweep = select_gate_config(val_records, val_probs, val_eval_args, args)
    (run_dir / "val_rcog_threshold_sweep_top200.json").write_text(json.dumps(val_sweep, indent=2), encoding="utf-8")
    (run_dir / "selected_rcog_config.json").write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
    print("Selected RCOG config on val:", flush=True)
    print(json.dumps(best_cfg, indent=2), flush=True)

    for split, pool, eval_args, seed_offset in [("val", val_pool, val_eval_args, 1000)]:
        probs = val_probs
        records = val_records
        actions = actions_for_config(records, probs, best_cfg)
        raw_actions = [r.raw_action for r in records]
        dij_actions = [0 for _ in records]
        split_dir = run_dir / f"eval_{split}"
        split_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "rcog": evaluate_actions(records, actions, eval_args, policy="rcog", out_csv=split_dir / f"{split}_rcog_episode_metrics.csv"),
            "raw_ddqn": evaluate_actions(records, raw_actions, eval_args, policy="raw_ddqn", out_csv=split_dir / f"{split}_raw_ddqn_episode_metrics.csv"),
            "dijkstra": evaluate_actions(records, dij_actions, eval_args, policy="dijkstra", out_csv=split_dir / f"{split}_dijkstra_episode_metrics.csv"),
        }
        reports["splits"][split] = report
        print(f"{split} RCOG:", json.dumps(report["rcog"], indent=2), flush=True)

    for idx, split in enumerate(args.eval_splits):
        if split == "val":
            continue
        split_dir = run_dir / f"eval_{split}"
        split_dir.mkdir(parents=True, exist_ok=True)
        pool, eval_args = build_pool_for_split(
            split=split,
            pool_size=int(args.eval_pool_size),
            run_dir=split_dir,
            args=args,
            train_config=train_config,
            runtime=runtime,
            seed_offset=2000 + idx * 1000,
        )
        eval_args.feature_set = "base"
        records = build_records(
            pool,
            runtime=runtime,
            eval_args=eval_args,
            agent=agent,
            benefit_delta=float(args.benefit_delta),
            bad_delta=float(args.bad_delta),
        )
        probs = make_probs(models, records)
        actions = actions_for_config(records, probs, best_cfg)
        raw_actions = [r.raw_action for r in records]
        dij_actions = [0 for _ in records]
        report = {
            "rcog": evaluate_actions(records, actions, eval_args, policy="rcog", out_csv=split_dir / f"{split}_rcog_episode_metrics.csv"),
            "raw_ddqn": evaluate_actions(records, raw_actions, eval_args, policy="raw_ddqn", out_csv=split_dir / f"{split}_raw_ddqn_episode_metrics.csv"),
            "dijkstra": evaluate_actions(records, dij_actions, eval_args, policy="dijkstra", out_csv=split_dir / f"{split}_dijkstra_episode_metrics.csv"),
        }
        reports["splits"][split] = report
        print(f"{split} RCOG:", json.dumps(report["rcog"], indent=2), flush=True)

    (run_dir / "rcog_summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Saved summary: {run_dir / 'rcog_summary.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-topk regret-calibrated opportunity gate.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--train-pool-size", type=int, default=2000)
    p.add_argument("--eval-pool-size", type=int, default=1000)
    p.add_argument("--model-iter", type=int, default=160)
    p.add_argument("--load-rcog-models", type=str, default="")
    p.add_argument("--selection-objective", type=str, default="time", choices=["time", "benefit_recall", "opportunity_recall", "opportunity_precision_at_recall", "opportunity_balanced", "constrained_balanced", "precision_at_recall", "two_stage_precision_at_recall"])
    p.add_argument("--threshold-search-grid", type=str, default="coarse", choices=["coarse", "dense"])
    p.add_argument("--save-full-sweep", action="store_true")
    p.add_argument("--two-stage-core-top-k", type=int, default=80)
    p.add_argument("--two-stage-recovery-top-k", type=int, default=160)
    p.add_argument("--two-stage-core-min-precision", type=float, default=75.0)
    p.add_argument("--two-stage-core-max-activation", type=float, default=10.0)
    p.add_argument("--two-stage-recovery-min-recall", type=float, default=55.0)
    p.add_argument("--two-stage-recovery-max-bad-gt-1pct", type=float, default=5.0)
    p.add_argument("--two-stage-recovery-max-bad-gt-5pct", type=float, default=0.8)
    p.add_argument("--opportunity-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--opportunity-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--benefit-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--benefit-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--bad-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--bad-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--benefit-delta", type=float, default=0.005)
    p.add_argument("--bad-delta", type=float, default=0.05)
    p.add_argument("--max-bad-gt-1pct", type=float, default=100.0)
    p.add_argument("--max-bad-gt-5pct", type=float, default=0.5)
    p.add_argument("--min-activation", type=float, default=5.0)
    p.add_argument("--max-activation", type=float, default=12.0)
    p.add_argument("--min-benefit-precision", type=float, default=55.0)
    p.add_argument("--min-benefit-recall", type=float, default=0.0)
    p.add_argument("--min-opportunity-precision", type=float, default=0.0)
    p.add_argument("--min-opportunity-recall", type=float, default=0.0)
    p.add_argument("--candidate-build-batch-size", type=int, default=16)
    p.add_argument("--candidate-build-log-interval", type=int, default=100)
    p.add_argument("--candidate-pool-attempt-multiplier", type=int, default=1200)
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
    p.add_argument("--feature-set", type=str, default="base", choices=[None, "base", "uncertainty", "pattern_topk"])
    p.add_argument("--reward-mode", type=str, default=None, choices=[None, "shaped", "direct_regret", "time_only"])
    p.add_argument("--pattern-topk", type=int, default=None)
    p.add_argument("--pattern-attention-temperature", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
