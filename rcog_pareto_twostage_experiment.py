from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np

from evaluate_candidate_route_reranker import load_train_config
from gated_candidate_route_reranker_experiment import build_runtime
from no_topk_rcog_gate_experiment import (
    build_agent,
    fast_stats_for_gate,
    make_probs,
    make_search_arrays,
    q_summary,
    select_gate_config,
)
from rcog_safety_constraint_experiment import train_indices_for_seed
from rcog_stability_validation import (
    bootstrap_ci,
    cache_path_for,
    cprint,
    fit_gate_models,
    flatten_metrics,
    load_cached_pool,
    load_or_build_records,
    parse_seed_list,
)
from real_data_uncertain_routing_experiment import configure_runtime, format_seconds, resolve_device


RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"


def parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


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


def normalize(values: np.ndarray, *, larger_is_better: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi - lo < 1e-12:
        return np.ones_like(values, dtype=np.float64)
    out = (values - lo) / (hi - lo)
    return out if larger_is_better else 1.0 - out


def simple_gate(
    *,
    probs: dict[str, np.ndarray],
    score: np.ndarray,
    q_margins: np.ndarray,
    raw_nonzero: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    gate = (
        (score >= float(cfg["score_threshold"]))
        & (probs["opportunity"] >= float(cfg["p_o_min"]))
        & (probs["benefit"] >= float(cfg["p_b_min"]))
        & (probs["bad"] <= float(cfg["p_d_max"]))
        & (q_margins >= float(cfg["q_margin_min"]))
    )
    return gate & raw_nonzero


def union_gate(
    *,
    probs: dict[str, np.ndarray],
    score: np.ndarray,
    q_margins: np.ndarray,
    raw_nonzero: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    if cfg.get("gate_mode") == "two_stage":
        return simple_gate(probs=probs, score=score, q_margins=q_margins, raw_nonzero=raw_nonzero, cfg=cfg["core"]) | simple_gate(
            probs=probs,
            score=score,
            q_margins=q_margins,
            raw_nonzero=raw_nonzero,
            cfg=cfg["recovery"],
        )
    return simple_gate(probs=probs, score=score, q_margins=q_margins, raw_nonzero=raw_nonzero, cfg=cfg)


def actions_from_gate(raw_actions: np.ndarray, gate: np.ndarray) -> list[int]:
    return np.where(np.asarray(gate, dtype=bool), raw_actions, 0).astype(np.int32).tolist()


def cfg_only(item: dict) -> dict:
    return {
        "score_threshold": float(item["score_threshold"]),
        "p_o_min": float(item["p_o_min"]),
        "p_b_min": float(item["p_b_min"]),
        "p_d_max": float(item["p_d_max"]),
        "q_margin_min": float(item["q_margin_min"]),
    }


def feasible_items(
    sweep: list[dict],
    *,
    cap: float,
    bad1: float,
    min_activation: float,
    max_activation: float,
    min_precision: float,
) -> list[dict]:
    return [
        item
        for item in sweep
        if float(item.get("bad_gt_5pct", 999.0)) <= float(cap)
        and float(item.get("bad_gt_1pct", 999.0)) <= float(bad1)
        and float(item.get("activation_rate", -1.0)) >= float(min_activation)
        and float(item.get("activation_rate", 999.0)) <= float(max_activation)
        and float(item.get("benefit_precision", 0.0)) >= float(min_precision)
    ]


def non_dominated(items: list[dict], *, max_frontier: int = 3000) -> list[dict]:
    if not items:
        return []
    metrics = np.asarray(
        [
            [
                -float(x.get("ddqn_over_pred", 999.0)),
                -float(x.get("bad_gt_5pct", 999.0)),
                -float(x.get("bad_gt_1pct", 999.0)),
                float(x.get("benefit_precision", 0.0)),
                float(x.get("benefit_recall", 0.0)),
                float(x.get("opportunity_recall", 0.0)),
            ]
            for x in items
        ],
        dtype=np.float64,
    )
    order = np.lexsort(
        (
            -metrics[:, 5],
            -metrics[:, 4],
            -metrics[:, 3],
            -metrics[:, 0],
        )
    )
    frontier_idx: list[int] = []
    frontier_points: list[np.ndarray] = []
    eps = 1e-12
    for idx in order:
        point = metrics[int(idx)]
        if frontier_points:
            f = np.vstack(frontier_points)
            dominated = np.any(np.all(f >= point - eps, axis=1) & np.any(f > point + eps, axis=1))
            if dominated:
                continue
            keep = ~(np.all(point >= f - eps, axis=1) & np.any(point > f + eps, axis=1))
            frontier_idx = [frontier_idx[i] for i, ok in enumerate(keep) if bool(ok)]
            frontier_points = [frontier_points[i] for i, ok in enumerate(keep) if bool(ok)]
        frontier_idx.append(int(idx))
        frontier_points.append(point)
        if len(frontier_idx) > max_frontier:
            break
    return [items[i] for i in frontier_idx]


def select_from_frontier(frontier: list[dict], objective: str) -> dict | None:
    if not frontier:
        return None
    if objective == "pareto_ratio":
        return sorted(
            frontier,
            key=lambda x: (
                float(x.get("ddqn_over_pred", 999.0)),
                -float(x.get("benefit_precision", 0.0)),
                -float(x.get("benefit_recall", 0.0)),
                float(x.get("bad_gt_5pct", 999.0)),
            ),
        )[0]
    if objective == "pareto_precision":
        return sorted(
            frontier,
            key=lambda x: (
                -float(x.get("benefit_precision", 0.0)),
                float(x.get("ddqn_over_pred", 999.0)),
                -float(x.get("benefit_recall", 0.0)),
                float(x.get("bad_gt_5pct", 999.0)),
            ),
        )[0]
    ratios = np.asarray([float(x.get("ddqn_over_pred", 999.0)) for x in frontier])
    bad5 = np.asarray([float(x.get("bad_gt_5pct", 999.0)) for x in frontier])
    benp = np.asarray([float(x.get("benefit_precision", 0.0)) for x in frontier])
    benr = np.asarray([float(x.get("benefit_recall", 0.0)) for x in frontier])
    oppr = np.asarray([float(x.get("opportunity_recall", 0.0)) for x in frontier])
    score = (
        0.35 * normalize(ratios, larger_is_better=False)
        + 0.15 * normalize(bad5, larger_is_better=False)
        + 0.20 * normalize(benp, larger_is_better=True)
        + 0.15 * normalize(benr, larger_is_better=True)
        + 0.15 * normalize(oppr, larger_is_better=True)
    )
    return frontier[int(np.argmax(score))]


def subset_ratio(arrays, gate: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan"), float("nan")
    activated = np.asarray(gate, dtype=bool) & arrays.raw_action_nonzero
    chosen = np.where(activated, arrays.raw_time, arrays.baseline_time)
    ratio = chosen / np.maximum(arrays.baseline_time, 1e-9)
    return float(np.mean(ratio[mask])), float(np.mean(activated[mask]) * 100.0)


def make_result_row(
    *,
    variant: str,
    seed: int,
    train_mode: str,
    cap: float,
    bad_weight: float,
    cfg: dict,
    probs: dict[str, np.ndarray],
    arrays,
    q_margins: np.ndarray,
    raw_nonzero: np.ndarray,
    opp_mask: np.ndarray,
    noopp_mask: np.ndarray,
) -> tuple[dict, np.ndarray]:
    score = probs["opportunity"] * probs["benefit"] * (1.0 - probs["bad"])
    gate = union_gate(probs=probs, score=score, q_margins=q_margins, raw_nonzero=raw_nonzero, cfg=cfg)
    stats = fast_stats_for_gate(gate, arrays)
    opp_ratio, opp_activation = subset_ratio(arrays, gate, opp_mask)
    noopp_ratio, noopp_activation = subset_ratio(arrays, gate, noopp_mask)
    row = {
        "variant": variant,
        "seed": int(seed),
        "train_mode": train_mode,
        "cap": float(cap),
        "bad_weight": float(bad_weight),
        "gate_mode": str(cfg.get("gate_mode", "single")),
        **flatten_metrics("test", stats),
        "opp_ratio": float(opp_ratio),
        "opp_activation": float(opp_activation),
        "noopp_ratio": float(noopp_ratio),
        "noopp_activation": float(noopp_activation),
    }
    return row, gate


def select_two_stage(
    *,
    sweep: list[dict],
    probs: dict[str, np.ndarray],
    arrays,
    q_margins: np.ndarray,
    raw_nonzero: np.ndarray,
    cap: float,
    objective: str,
    args: argparse.Namespace,
) -> dict | None:
    score = probs["opportunity"] * probs["benefit"] * (1.0 - probs["bad"])
    core = [
        item
        for item in sweep
        if float(item.get("bad_gt_5pct", 999.0)) <= float(args.core_bad5)
        and float(item.get("bad_gt_1pct", 999.0)) <= float(args.core_bad1)
        and float(item.get("benefit_precision", 0.0)) >= float(args.core_min_precision)
        and float(item.get("activation_rate", 999.0)) <= float(args.core_max_activation)
        and float(item.get("activation_rate", -1.0)) >= float(args.min_activation)
    ]
    recovery = [
        item
        for item in sweep
        if float(item.get("bad_gt_5pct", 999.0)) <= float(cap)
        and float(item.get("bad_gt_1pct", 999.0)) <= float(args.max_bad_gt_1pct)
        and float(item.get("benefit_precision", 0.0)) >= float(args.recovery_min_precision)
        and float(item.get("opportunity_recall", 0.0)) >= float(args.recovery_min_opportunity_recall)
        and float(item.get("activation_rate", 999.0)) <= float(args.max_activation)
    ]
    core = sorted(
        core,
        key=lambda x: (
            float(x.get("ddqn_over_pred", 999.0)),
            -float(x.get("benefit_precision", 0.0)),
            float(x.get("bad_gt_5pct", 999.0)),
        ),
    )[: int(args.core_top_k)]
    recovery = sorted(
        recovery,
        key=lambda x: (
            -float(x.get("opportunity_recall", 0.0)),
            -float(x.get("benefit_recall", 0.0)),
            float(x.get("bad_gt_5pct", 999.0)),
            float(x.get("ddqn_over_pred", 999.0)),
        ),
    )[: int(args.recovery_top_k)]
    best_cfg = None
    best_stats = None
    best_key = None
    for c in core:
        c_cfg = cfg_only(c)
        c_gate = simple_gate(probs=probs, score=score, q_margins=q_margins, raw_nonzero=raw_nonzero, cfg=c_cfg)
        for r in recovery:
            r_cfg = cfg_only(r)
            gate = c_gate | simple_gate(probs=probs, score=score, q_margins=q_margins, raw_nonzero=raw_nonzero, cfg=r_cfg)
            stats = fast_stats_for_gate(gate, arrays)
            if float(stats["bad_gt_5pct"]) > float(cap):
                continue
            if float(stats["bad_gt_1pct"]) > float(args.max_bad_gt_1pct):
                continue
            if float(stats["activation_rate"]) > float(args.max_activation):
                continue
            if float(stats["benefit_precision"]) < float(args.twostage_min_precision):
                continue
            if objective == "two_stage_ratio":
                key = (
                    float(stats["ddqn_over_pred"]),
                    -float(stats["benefit_recall"]),
                    -float(stats["benefit_precision"]),
                    float(stats["bad_gt_5pct"]),
                )
            else:
                key = (
                    -float(stats["opportunity_recall"]),
                    float(stats["ddqn_over_pred"]),
                    -float(stats["benefit_precision"]),
                    float(stats["bad_gt_5pct"]),
                )
            if best_key is None or key < best_key:
                best_key = key
                best_stats = stats
                best_cfg = {"gate_mode": "two_stage", "core": c_cfg, "recovery": r_cfg, "val_stats": stats}
    return best_cfg


def aggregate(rows: list[dict], *, seed_count: int, target_bad5: float) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    out = []
    for name, group in grouped.items():
        if len(group) < seed_count:
            continue
        def arr(key: str) -> np.ndarray:
            return np.asarray([float(x[key]) for x in group], dtype=np.float64)

        ratio = arr("test_ddqn_over_pred")
        benp = arr("test_benefit_precision")
        benr = arr("test_benefit_recall")
        bad5 = arr("test_bad_gt_5pct")
        oppr = arr("test_opportunity_recall")
        opp_ratio = arr("opp_ratio")
        activation = arr("test_activation_rate")
        ci_low = arr("ci_mean_improvement_pct_ci95_low") if "ci_mean_improvement_pct_ci95_low" in group[0] else np.asarray([])
        out.append(
            {
                "variant": name,
                "seeds": int(len(group)),
                "cap": float(group[0]["cap"]),
                "bad_weight": float(group[0]["bad_weight"]),
                "gate_mode": str(group[0]["gate_mode"]),
                "mean_ratio_mean": float(np.mean(ratio)),
                "mean_ratio_std": float(np.std(ratio, ddof=1)) if ratio.size > 1 else 0.0,
                "benefit_precision_mean": float(np.mean(benp)),
                "benefit_recall_mean": float(np.mean(benr)),
                "bad_gt_5pct_mean": float(np.mean(bad5)),
                "bad_gt_5pct_max": float(np.max(bad5)),
                "opportunity_recall_mean": float(np.mean(oppr)),
                "opportunity_ratio_mean": float(np.mean(opp_ratio)),
                "activation_mean": float(np.mean(activation)),
                "ci_improvement_low_min": float(np.min(ci_low)) if ci_low.size else float("nan"),
                "safety_pass": bool(float(np.max(bad5)) <= float(target_bad5)),
                "positive_ci_pass": bool(np.min(ci_low) > 0.0) if ci_low.size else False,
            }
        )
    out.sort(
        key=lambda x: (
            not bool(x["safety_pass"]),
            not bool(x["positive_ci_pass"]),
            float(x["mean_ratio_mean"]),
            -float(x["opportunity_recall_mean"]),
            float(x["bad_gt_5pct_max"]),
        )
    )
    return out


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records_cache_dir = Path(args.records_cache_run_dir)
    cache_run_dir = Path(args.cache_run_dir)
    train_config = load_train_config(args.train_config)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))

    print(f"{BOLD}{CYAN}{'=' * 112}{RESET}", flush=True)
    cprint(CYAN, "RCOG Pareto + two-stage safety gate experiment")
    cprint(CYAN, f"run_dir={run_dir}")
    print(f"{BOLD}{CYAN}{'=' * 112}{RESET}", flush=True)

    runtime = build_runtime(args, train_config, device)
    train_pool, train_eval_args = load_cached_pool(cache_path_for(cache_run_dir, "train", int(args.train_pool_size)))
    val_pool, val_eval_args = load_cached_pool(cache_path_for(cache_run_dir, "val", int(args.eval_pool_size)))
    test_pool, test_eval_args = load_cached_pool(cache_path_for(cache_run_dir, "test", int(args.eval_pool_size)))
    agent = build_agent(args, train_config, train_eval_args, runtime)
    train_records = load_or_build_records(name="train", pool=train_pool, runtime=runtime, eval_args=train_eval_args, agent=agent, args=args, run_dir=records_cache_dir)
    val_records = load_or_build_records(name="val", pool=val_pool, runtime=runtime, eval_args=val_eval_args, agent=agent, args=args, run_dir=records_cache_dir)
    test_records = load_or_build_records(name="test", pool=test_pool, runtime=runtime, eval_args=test_eval_args, agent=agent, args=args, run_dir=records_cache_dir)

    val_args = copy.copy(args)
    val_args.save_full_sweep = True
    val_args.selection_objective = "time"
    val_args.max_bad_gt_5pct = 100.0
    val_args.max_bad_gt_1pct = 100.0
    val_args.min_benefit_precision = 0.0
    val_args.max_activation = 100.0
    val_args.min_activation = 0.0

    seeds = parse_seed_list(args.seeds)
    bad_weights = parse_float_list(args.bad_weights)
    caps = parse_float_list(args.caps)
    test_arrays = make_search_arrays(test_records, test_eval_args)
    val_arrays = make_search_arrays(val_records, val_eval_args)
    test_q_margins = np.asarray([q_summary(r.q_values, r.mask, r.raw_action)["q_margin"] for r in test_records], dtype=np.float32)
    val_q_margins = np.asarray([q_summary(r.q_values, r.mask, r.raw_action)["q_margin"] for r in val_records], dtype=np.float32)
    raw_actions = test_arrays.raw_action
    opp_mask = test_arrays.opportunity_label
    noopp_mask = ~test_arrays.opportunity_label

    rows: list[dict] = []
    cfg_rows: list[dict] = []
    gate_cache: dict[tuple[str, int], tuple[np.ndarray, dict]] = {}
    total_models = len(seeds) * len(bad_weights)
    done = 0
    start = time.time()

    for seed in seeds:
        train_indices, train_mode = train_indices_for_seed(seed, len(train_records), args)
        for bad_weight in bad_weights:
            done += 1
            model_args = copy.copy(args)
            model_args.bad_positive_weight_mult = float(bad_weight)
            cprint(BLUE, f"[MODEL {done}/{total_models}] seed={seed} mode={train_mode} bad_weight={bad_weight}")
            models = fit_gate_models(train_records, seed=int(seed), args=model_args, train_indices=train_indices)
            val_probs = make_probs(models, val_records)
            test_probs = make_probs(models, test_records)
            _, full_sweep = select_gate_config(val_records, val_probs, val_eval_args, val_args)
            val_score = val_probs["opportunity"] * val_probs["benefit"] * (1.0 - val_probs["bad"])

            for cap in caps:
                feasible = feasible_items(
                    full_sweep,
                    cap=float(cap),
                    bad1=float(args.max_bad_gt_1pct),
                    min_activation=float(args.min_activation),
                    max_activation=float(args.max_activation),
                    min_precision=float(args.pareto_min_precision),
                )
                frontier = non_dominated(feasible, max_frontier=int(args.max_frontier))
                for objective in ["pareto_ratio", "pareto_balanced", "pareto_precision"]:
                    selected = select_from_frontier(frontier, objective)
                    if selected is None:
                        continue
                    cfg = cfg_only(selected)
                    variant = f"{objective}_cap{cap:.2f}_bw{bad_weight:.1f}"
                    row, gate = make_result_row(
                        variant=variant,
                        seed=int(seed),
                        train_mode=train_mode,
                        cap=float(cap),
                        bad_weight=float(bad_weight),
                        cfg=cfg,
                        probs=test_probs,
                        arrays=test_arrays,
                        q_margins=test_q_margins,
                        raw_nonzero=test_arrays.raw_action_nonzero,
                        opp_mask=opp_mask,
                        noopp_mask=noopp_mask,
                    )
                    rows.append(row)
                    cfg_rows.append({"variant": variant, "seed": int(seed), "selected_config": cfg, "val_stats": selected})
                    gate_cache[(variant, int(seed))] = (gate, cfg)

                for objective in ["two_stage_ratio", "two_stage_recall"]:
                    cfg = select_two_stage(
                        sweep=full_sweep,
                        probs=val_probs,
                        arrays=val_arrays,
                        q_margins=val_q_margins,
                        raw_nonzero=val_arrays.raw_action_nonzero,
                        cap=float(cap),
                        objective=objective,
                        args=args,
                    )
                    if cfg is None:
                        continue
                    variant = f"{objective}_cap{cap:.2f}_bw{bad_weight:.1f}"
                    row, gate = make_result_row(
                        variant=variant,
                        seed=int(seed),
                        train_mode=train_mode,
                        cap=float(cap),
                        bad_weight=float(bad_weight),
                        cfg=cfg,
                        probs=test_probs,
                        arrays=test_arrays,
                        q_margins=test_q_margins,
                        raw_nonzero=test_arrays.raw_action_nonzero,
                        opp_mask=opp_mask,
                        noopp_mask=noopp_mask,
                    )
                    rows.append(row)
                    cfg_rows.append({"variant": variant, "seed": int(seed), "selected_config": cfg})
                    gate_cache[(variant, int(seed))] = (gate, cfg)

            elapsed = time.time() - start
            progress = done / max(total_models, 1)
            eta = elapsed * (1.0 - progress) / max(progress, 1e-9)
            cprint(CYAN, f"[MODEL DONE] {done}/{total_models} sweep={len(full_sweep)} rows={len(rows)} elapsed={format_seconds(elapsed)} ETA={format_seconds(eta)}")

    write_rows(run_dir / "pareto_twostage_seed_metrics_raw.csv", rows)
    pre = aggregate(rows, seed_count=len(seeds), target_bad5=float(args.target_bad_gt_5pct))
    top_names = [x["variant"] for x in pre[: int(args.bootstrap_top_k)]]
    cprint(BLUE, f"[BOOTSTRAP] confirming top {len(top_names)} variants")
    for row in rows:
        if row["variant"] not in top_names:
            continue
        gate, _ = gate_cache[(str(row["variant"]), int(row["seed"]))]
        actions = actions_from_gate(raw_actions, gate)
        ci = bootstrap_ci(test_records, actions, seed=int(row["seed"]) + 7079, iters=int(args.bootstrap_iters))
        row.update(flatten_metrics("ci", ci))

    agg = aggregate(rows, seed_count=len(seeds), target_bad5=float(args.target_bad_gt_5pct))
    write_rows(run_dir / "pareto_twostage_seed_metrics.csv", rows)
    write_rows(run_dir / "pareto_twostage_aggregate_metrics.csv", agg)
    (run_dir / "selected_configs.json").write_text(json.dumps(cfg_rows, indent=2), encoding="utf-8")
    summary = {
        "run_dir": str(run_dir),
        "seeds": seeds,
        "bad_weights": bad_weights,
        "caps": caps,
        "target_bad_gt_5pct": float(args.target_bad_gt_5pct),
        "references": {
            "baseline_cap0.50": {
                "mean_ratio": 0.998663,
                "benefit_precision": 57.2,
                "benefit_recall": 65.3,
                "bad_gt_5pct_max": 0.73,
                "opportunity_recall": 56.4,
            },
            "refined_0.30": {
                "mean_ratio": 0.998329,
                "benefit_precision": 64.03,
                "benefit_recall": 61.40,
                "bad_gt_5pct_max": 0.333,
                "opportunity_recall": 52.11,
            },
            "cap0.34_balanced_bw1.5": {
                "mean_ratio": 0.998368,
                "benefit_precision": 63.23,
                "benefit_recall": 62.13,
                "bad_gt_5pct_max": 0.367,
                "opportunity_recall": 52.81,
            },
        },
        "aggregates_ranked": agg[:100],
        "recommendation": agg[0] if agg else {},
        "outputs": {
            "seed_metrics": str(run_dir / "pareto_twostage_seed_metrics.csv"),
            "aggregate_metrics": str(run_dir / "pareto_twostage_aggregate_metrics.csv"),
            "selected_configs": str(run_dir / "selected_configs.json"),
        },
    }
    out_path = run_dir / "rcog_pareto_twostage_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{BOLD}{GREEN}{'=' * 112}{RESET}", flush=True)
    cprint(GREEN, f"Saved summary: {out_path}")
    for row in agg[:20]:
        color = GREEN if row["safety_pass"] and row["positive_ci_pass"] else YELLOW
        cprint(
            color,
            f"{row['variant']:<36} ratio={row['mean_ratio_mean']:.6f} "
            f"benP={row['benefit_precision_mean']:.1f}% benR={row['benefit_recall_mean']:.1f}% "
            f"bad5={row['bad_gt_5pct_mean']:.2f}/{row['bad_gt_5pct_max']:.2f}% "
            f"oppR={row['opportunity_recall_mean']:.1f}% act={row['activation_mean']:.1f}%",
        )
    print(f"{BOLD}{GREEN}{'=' * 112}{RESET}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict Pareto and two-stage safety gate experiments for RCOG-DDQN.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--cache-run-dir", type=str, required=True)
    p.add_argument("--records-cache-run-dir", type=str, required=True)
    p.add_argument("--train-pool-size", type=int, default=10000)
    p.add_argument("--eval-pool-size", type=int, default=3000)
    p.add_argument("--seeds", type=str, default="121,122,123,124,125")
    p.add_argument("--train-resample-mode", type=str, default="bootstrap", choices=["full", "bootstrap", "subsample"])
    p.add_argument("--train-subsample-frac", type=float, default=0.8)
    p.add_argument("--caps", type=str, default="0.30,0.34")
    p.add_argument("--bad-weights", type=str, default="1.0,1.5,2.0")
    p.add_argument("--target-bad-gt-5pct", type=float, default=0.5)
    p.add_argument("--bootstrap-top-k", type=int, default=20)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--max-frontier", type=int, default=3000)
    p.add_argument("--pareto-min-precision", type=float, default=58.0)
    p.add_argument("--core-min-precision", type=float, default=72.0)
    p.add_argument("--core-bad5", type=float, default=0.20)
    p.add_argument("--core-bad1", type=float, default=2.0)
    p.add_argument("--core-max-activation", type=float, default=8.0)
    p.add_argument("--core-top-k", type=int, default=50)
    p.add_argument("--recovery-min-precision", type=float, default=58.0)
    p.add_argument("--recovery-min-opportunity-recall", type=float, default=45.0)
    p.add_argument("--recovery-top-k", type=int, default=120)
    p.add_argument("--twostage-min-precision", type=float, default=60.0)
    p.add_argument("--rebuild-records", action="store_true")
    p.add_argument("--model-iter", type=int, default=160)
    p.add_argument("--threshold-search-grid", type=str, default="dense", choices=["coarse", "dense"])
    p.add_argument("--save-full-sweep", action="store_true")
    p.add_argument("--selection-objective", type=str, default="constrained_balanced")
    p.add_argument("--opportunity-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--opportunity-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--benefit-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--benefit-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--bad-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--bad-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--benefit-delta", type=float, default=0.005)
    p.add_argument("--bad-delta", type=float, default=0.05)
    p.add_argument("--max-bad-gt-1pct", type=float, default=3.0)
    p.add_argument("--max-bad-gt-5pct", type=float, default=0.5)
    p.add_argument("--min-activation", type=float, default=5.0)
    p.add_argument("--max-activation", type=float, default=12.0)
    p.add_argument("--min-benefit-precision", type=float, default=55.0)
    p.add_argument("--min-benefit-recall", type=float, default=0.0)
    p.add_argument("--min-opportunity-precision", type=float, default=0.0)
    p.add_argument("--min-opportunity-recall", type=float, default=0.0)
    p.add_argument("--two-stage-core-top-k", type=int, default=80)
    p.add_argument("--two-stage-recovery-top-k", type=int, default=160)
    p.add_argument("--two-stage-core-min-precision", type=float, default=75.0)
    p.add_argument("--two-stage-core-max-activation", type=float, default=10.0)
    p.add_argument("--two-stage-recovery-min-recall", type=float, default=55.0)
    p.add_argument("--two-stage-recovery-max-bad-gt-1pct", type=float, default=5.0)
    p.add_argument("--two-stage-recovery-max-bad-gt-5pct", type=float, default=0.8)
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
