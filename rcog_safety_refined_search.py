from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np

from evaluate_candidate_route_reranker import load_train_config
from gated_candidate_route_reranker_experiment import build_runtime
from no_topk_rcog_gate_experiment import (
    actions_for_config,
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


def parse_objectives(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


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


def fmt_param(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def variant_name(
    *,
    cap: float,
    max_activation: float,
    min_precision: float,
    min_benefit_recall: float,
    min_opportunity_recall: float,
    objective: str,
    bad_weight: float,
) -> str:
    return (
        f"cap{cap:.2f}_act{fmt_param(max_activation)}_prec{min_precision:.0f}_"
        f"benR{fmt_param(min_benefit_recall)}_oppR{fmt_param(min_opportunity_recall)}_"
        f"{objective}_bw{fmt_param(bad_weight)}"
    )


def gate_from_cfg(
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


def actions_from_gate(raw_actions: np.ndarray, gate: np.ndarray) -> list[int]:
    return np.where(gate, raw_actions, 0).astype(np.int32).tolist()


def chosen_ratio_stats(arrays, gate: np.ndarray, mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return {"ratio": float("nan"), "activation": float("nan"), "opportunity_recall": float("nan")}
    activated = np.asarray(gate, dtype=bool) & arrays.raw_action_nonzero
    chosen = np.where(activated, arrays.raw_time, arrays.baseline_time)
    ratio = chosen / np.maximum(arrays.baseline_time, 1e-9)
    opp_count = int(np.sum(arrays.opportunity_label))
    return {
        "ratio": float(np.mean(ratio[mask])),
        "activation": float(np.mean(activated[mask]) * 100.0),
        "opportunity_recall": float(np.sum(arrays.opportunity_label & activated) / max(opp_count, 1) * 100.0),
    }


def selected_cfg_from_sweep(
    sweep: list[dict],
    *,
    cap: float,
    max_activation: float,
    min_precision: float,
    min_benefit_recall: float,
    min_opportunity_recall: float,
    objective: str,
    min_activation: float,
    max_bad_gt_1pct: float,
) -> dict | None:
    pool = [
        item
        for item in sweep
        if float(item.get("bad_gt_5pct", 999.0)) <= float(cap)
        and float(item.get("bad_gt_1pct", 999.0)) <= float(max_bad_gt_1pct)
        and float(item.get("activation_rate", -1.0)) >= float(min_activation)
        and float(item.get("activation_rate", 999.0)) <= float(max_activation)
        and float(item.get("benefit_precision", 0.0)) >= float(min_precision)
        and float(item.get("benefit_recall", 0.0)) >= float(min_benefit_recall)
        and float(item.get("opportunity_recall", 0.0)) >= float(min_opportunity_recall)
    ]
    if not pool:
        return None
    if objective == "ratio":
        pool.sort(
            key=lambda x: (
                float(x.get("ddqn_over_pred", float("inf"))),
                -float(x.get("benefit_precision", 0.0)),
                -float(x.get("benefit_recall", 0.0)),
                float(x.get("bad_gt_5pct", float("inf"))),
                float(x.get("activation_rate", float("inf"))),
            )
        )
    elif objective == "precision":
        pool.sort(
            key=lambda x: (
                -float(x.get("benefit_precision", 0.0)),
                float(x.get("ddqn_over_pred", float("inf"))),
                -float(x.get("benefit_recall", 0.0)),
                float(x.get("bad_gt_5pct", float("inf"))),
            )
        )
    else:
        pool.sort(
            key=lambda x: (
                -float(x.get("benefit_recall", 0.0)),
                -float(x.get("benefit_precision", 0.0)),
                float(x.get("bad_gt_5pct", float("inf"))),
                float(x.get("ddqn_over_pred", float("inf"))),
            )
        )
    return pool[0]


def aggregate_rows(rows: list[dict], *, seed_count: int, target_bad5: float) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    aggregates = []
    for name, group in grouped.items():
        if len(group) < int(seed_count):
            continue
        def arr(key: str) -> np.ndarray:
            return np.asarray([float(x[key]) for x in group], dtype=np.float64)

        ratio = arr("test_ddqn_over_pred")
        benp = arr("test_benefit_precision")
        benr = arr("test_benefit_recall")
        bad5 = arr("test_bad_gt_5pct")
        act = arr("test_activation_rate")
        opp_ratio = arr("opp_ratio")
        opp_recall = arr("opp_recall")
        noopp_ratio = arr("noopp_ratio")
        ci_low = arr("ci_mean_improvement_pct_ci95_low") if "ci_mean_improvement_pct_ci95_low" in group[0] else np.asarray([])
        first = group[0]
        aggregates.append(
            {
                "variant": name,
                "cap": float(first["cap"]),
                "max_activation": float(first["max_activation"]),
                "min_precision": float(first["min_precision"]),
                "min_benefit_recall": float(first.get("min_benefit_recall", 0.0)),
                "min_opportunity_recall": float(first.get("min_opportunity_recall", 0.0)),
                "objective": str(first["objective"]),
                "bad_weight": float(first["bad_weight"]),
                "seeds": int(len(group)),
                "mean_ratio_mean": float(np.mean(ratio)),
                "mean_ratio_std": float(np.std(ratio, ddof=1)) if ratio.size > 1 else 0.0,
                "mean_ratio_max": float(np.max(ratio)),
                "benefit_precision_mean": float(np.mean(benp)),
                "benefit_recall_mean": float(np.mean(benr)),
                "bad_gt_5pct_mean": float(np.mean(bad5)),
                "bad_gt_5pct_max": float(np.max(bad5)),
                "activation_mean": float(np.mean(act)),
                "opportunity_ratio_mean": float(np.mean(opp_ratio)),
                "opportunity_recall_mean": float(np.mean(opp_recall)),
                "no_opportunity_ratio_mean": float(np.mean(noopp_ratio)),
                "ci_improvement_low_min": float(np.min(ci_low)) if ci_low.size else float("nan"),
                "safety_pass": bool(np.max(bad5) <= float(target_bad5)),
                "positive_ci_pass": bool(np.min(ci_low) > 0.0) if ci_low.size else False,
            }
        )
    aggregates.sort(
        key=lambda x: (
            not bool(x["safety_pass"]),
            not bool(x["positive_ci_pass"]),
            float(x["mean_ratio_mean"]),
            -float(x["opportunity_recall_mean"]),
            float(x["bad_gt_5pct_max"]),
        )
    )
    return aggregates


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    records_cache_dir = Path(args.records_cache_run_dir)
    cache_run_dir = Path(args.cache_run_dir)
    train_config = load_train_config(args.train_config)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))

    print(f"{BOLD}{CYAN}{'=' * 112}{RESET}", flush=True)
    cprint(CYAN, "RCOG safe-cap refined search")
    cprint(CYAN, f"run_dir={run_dir}")
    cprint(CYAN, f"records_cache_dir={records_cache_dir}")
    print(f"{BOLD}{CYAN}{'=' * 112}{RESET}", flush=True)

    runtime = build_runtime(args, train_config, device)
    train_pool, train_eval_args = load_cached_pool(cache_path_for(cache_run_dir, "train", int(args.train_pool_size)))
    val_pool, val_eval_args = load_cached_pool(cache_path_for(cache_run_dir, "val", int(args.eval_pool_size)))
    test_pool, test_eval_args = load_cached_pool(cache_path_for(cache_run_dir, "test", int(args.eval_pool_size)))
    agent = build_agent(args, train_config, train_eval_args, runtime)
    train_records = load_or_build_records(
        name="train",
        pool=train_pool,
        runtime=runtime,
        eval_args=train_eval_args,
        agent=agent,
        args=args,
        run_dir=records_cache_dir,
    )
    val_records = load_or_build_records(
        name="val",
        pool=val_pool,
        runtime=runtime,
        eval_args=val_eval_args,
        agent=agent,
        args=args,
        run_dir=records_cache_dir,
    )
    test_records = load_or_build_records(
        name="test",
        pool=test_pool,
        runtime=runtime,
        eval_args=test_eval_args,
        agent=agent,
        args=args,
        run_dir=records_cache_dir,
    )

    val_args = copy.copy(args)
    val_args.save_full_sweep = True
    val_args.selection_objective = "time"
    val_args.max_bad_gt_5pct = 100.0
    val_args.max_bad_gt_1pct = 100.0
    val_args.min_benefit_precision = 0.0
    val_args.min_benefit_recall = 0.0
    val_args.max_activation = 100.0

    caps = parse_float_list(args.caps)
    max_activations = parse_float_list(args.max_activations)
    min_precisions = parse_float_list(args.min_precisions)
    min_benefit_recalls = parse_float_list(args.min_benefit_recalls)
    min_opportunity_recalls = parse_float_list(args.min_opportunity_recalls)
    bad_weights = parse_float_list(args.bad_weights)
    objectives = parse_objectives(args.objectives)
    seeds = parse_seed_list(args.seeds)

    test_arrays = make_search_arrays(test_records, test_eval_args)
    raw_nonzero = test_arrays.raw_action_nonzero
    raw_actions = test_arrays.raw_action
    opp_mask = test_arrays.opportunity_label
    noopp_mask = ~test_arrays.opportunity_label
    q_margins = np.asarray([q_summary(r.q_values, r.mask, r.raw_action)["q_margin"] for r in test_records], dtype=np.float32)

    all_rows: list[dict] = []
    detail_rows: list[dict] = []
    top_cache: dict[tuple[str, int], dict] = {}
    total_models = len(seeds) * len(bad_weights)
    model_done = 0
    start_all = time.time()

    for seed in seeds:
        train_indices, train_mode = train_indices_for_seed(seed, len(train_records), args)
        for bad_weight in bad_weights:
            model_done += 1
            model_start = time.time()
            model_args = copy.copy(args)
            model_args.bad_positive_weight_mult = float(bad_weight)
            cprint(
                BLUE,
                f"[MODEL {model_done}/{total_models}] seed={seed} mode={train_mode} bad_weight={bad_weight}",
            )
            models = fit_gate_models(train_records, seed=int(seed), args=model_args, train_indices=train_indices)
            val_probs = make_probs(models, val_records)
            test_probs = make_probs(models, test_records)
            _, full_sweep = select_gate_config(val_records, val_probs, val_eval_args, val_args)
            score = test_probs["opportunity"] * test_probs["benefit"] * (1.0 - test_probs["bad"])
            evaluated = 0
            for cap in caps:
                for max_act in max_activations:
                    for min_prec in min_precisions:
                        for min_benr in min_benefit_recalls:
                            for min_oppr in min_opportunity_recalls:
                                for objective in objectives:
                                    cfg = selected_cfg_from_sweep(
                                        full_sweep,
                                        cap=float(cap),
                                        max_activation=float(max_act),
                                        min_precision=float(min_prec),
                                        min_benefit_recall=float(min_benr),
                                        min_opportunity_recall=float(min_oppr),
                                        objective=objective,
                                        min_activation=float(args.min_activation),
                                        max_bad_gt_1pct=float(args.max_bad_gt_1pct),
                                    )
                                    if cfg is None:
                                        continue
                                    gate = gate_from_cfg(
                                        probs=test_probs,
                                        score=score,
                                        q_margins=q_margins,
                                        raw_nonzero=raw_nonzero,
                                        cfg=cfg,
                                    )
                                    stats = fast_stats_for_gate(gate, test_arrays)
                                    opp_stats = chosen_ratio_stats(test_arrays, gate, opp_mask)
                                    noopp_stats = chosen_ratio_stats(test_arrays, gate, noopp_mask)
                                    name = variant_name(
                                        cap=float(cap),
                                        max_activation=float(max_act),
                                        min_precision=float(min_prec),
                                        min_benefit_recall=float(min_benr),
                                        min_opportunity_recall=float(min_oppr),
                                        objective=objective,
                                        bad_weight=float(bad_weight),
                                    )
                                    row = {
                                        "variant": name,
                                        "seed": int(seed),
                                        "train_mode": train_mode,
                                        "cap": float(cap),
                                        "max_activation": float(max_act),
                                        "min_precision": float(min_prec),
                                        "min_benefit_recall": float(min_benr),
                                        "min_opportunity_recall": float(min_oppr),
                                        "objective": objective,
                                        "bad_weight": float(bad_weight),
                                        "selected_val_bad_gt_5pct": float(cfg.get("bad_gt_5pct", float("nan"))),
                                        "selected_val_bad_gt_1pct": float(cfg.get("bad_gt_1pct", float("nan"))),
                                        "selected_val_activation": float(cfg.get("activation_rate", float("nan"))),
                                        **flatten_metrics("test", stats),
                                        "opp_ratio": float(opp_stats["ratio"]),
                                        "opp_activation": float(opp_stats["activation"]),
                                        "opp_recall": float(opp_stats["opportunity_recall"]),
                                        "noopp_ratio": float(noopp_stats["ratio"]),
                                        "noopp_activation": float(noopp_stats["activation"]),
                                    }
                                    all_rows.append(row)
                                    detail_rows.append({**row, "selected_config": cfg})
                                    top_cache[(name, int(seed))] = {
                                        "cfg": cfg,
                                        "test_probs": test_probs,
                                        "score": score,
                                        "q_margins": q_margins,
                                        "raw_nonzero": raw_nonzero,
                                    }
                                    evaluated += 1
            elapsed = time.time() - start_all
            progress = model_done / max(total_models, 1)
            eta = elapsed * (1.0 - progress) / max(progress, 1e-9)
            cprint(
                CYAN,
                f"[MODEL DONE] seed={seed} bad_weight={bad_weight} candidates={len(full_sweep)} "
                f"evaluated={evaluated} model_elapsed={format_seconds(time.time() - model_start)} "
                f"total={model_done}/{total_models} ETA={format_seconds(eta)}",
            )

    write_rows(run_dir / "refined_seed_metrics_raw.csv", all_rows)
    aggregates_pre = aggregate_rows(all_rows, seed_count=len(seeds), target_bad5=float(args.target_bad_gt_5pct))
    top_names = [str(x["variant"]) for x in aggregates_pre[: int(args.bootstrap_top_k)]]
    cprint(BLUE, f"[BOOTSTRAP] confirming top {len(top_names)} variants: {', '.join(top_names[:5])}")

    # Add bootstrap CIs only for the final shortlist.
    for row in all_rows:
        if row["variant"] not in top_names:
            continue
        cache = top_cache[(str(row["variant"]), int(row["seed"]))]
        gate = gate_from_cfg(
            probs=cache["test_probs"],
            score=cache["score"],
            q_margins=cache["q_margins"],
            raw_nonzero=cache["raw_nonzero"],
            cfg=cache["cfg"],
        )
        actions = actions_from_gate(raw_actions, gate)
        ci = bootstrap_ci(test_records, actions, seed=int(row["seed"]) + 5003, iters=int(args.bootstrap_iters))
        row.update(flatten_metrics("ci", ci))

    aggregates = aggregate_rows(all_rows, seed_count=len(seeds), target_bad5=float(args.target_bad_gt_5pct))
    write_rows(run_dir / "refined_seed_metrics.csv", all_rows)
    write_rows(run_dir / "refined_aggregate_metrics.csv", aggregates)
    with (run_dir / "selected_configs_top.json").open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "row": r,
                    "selected_config": next(
                        (d["selected_config"] for d in detail_rows if d["variant"] == r["variant"] and int(d["seed"]) == int(r["seed"])),
                        None,
                    ),
                }
                for r in all_rows
                if r["variant"] in top_names
            ],
            f,
            indent=2,
        )
    summary = {
        "run_dir": str(run_dir),
        "cache_run_dir": str(cache_run_dir),
        "records_cache_dir": str(records_cache_dir),
        "seeds": seeds,
        "train_resample_mode": str(args.train_resample_mode),
        "search_space": {
            "caps": caps,
            "max_activations": max_activations,
            "min_precisions": min_precisions,
            "min_benefit_recalls": min_benefit_recalls,
            "min_opportunity_recalls": min_opportunity_recalls,
            "bad_weights": bad_weights,
            "objectives": objectives,
        },
        "target_bad_gt_5pct": float(args.target_bad_gt_5pct),
        "baseline_safe_cap0.30_reference": {
            "mean_ratio": 0.998420,
            "benefit_precision": 63.2,
            "benefit_recall": 60.9,
            "bad_gt_5pct_mean": 0.33,
            "bad_gt_5pct_max": 0.43,
            "opportunity_ratio": 0.984218,
            "opportunity_recall": 51.9,
        },
        "aggregates_ranked": aggregates[:100],
        "recommendation": aggregates[0] if aggregates else {},
        "outputs": {
            "seed_metrics": str(run_dir / "refined_seed_metrics.csv"),
            "aggregate_metrics": str(run_dir / "refined_aggregate_metrics.csv"),
            "selected_configs_top": str(run_dir / "selected_configs_top.json"),
        },
    }
    out_path = run_dir / "rcog_safety_refined_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{BOLD}{GREEN}{'=' * 112}{RESET}", flush=True)
    cprint(GREEN, f"Saved refined summary: {out_path}")
    for row in aggregates[:15]:
        color = GREEN if row["safety_pass"] and row["positive_ci_pass"] else YELLOW
        cprint(
            color,
            f"{row['variant']:<38} ratio={row['mean_ratio_mean']:.6f} "
            f"benP={row['benefit_precision_mean']:.1f}% benR={row['benefit_recall_mean']:.1f}% "
            f"bad5={row['bad_gt_5pct_mean']:.2f}/{row['bad_gt_5pct_max']:.2f}% "
            f"oppR={row['opportunity_recall_mean']:.1f}% opp_ratio={row['opportunity_ratio_mean']:.6f}",
        )
    print(f"{BOLD}{GREEN}{'=' * 112}{RESET}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refined safety-cap search around safe_cap0.30.")
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
    p.add_argument("--caps", type=str, default="0.26,0.28,0.30,0.32,0.34,0.36")
    p.add_argument("--max-activations", type=str, default="8,9,10,11,12")
    p.add_argument("--min-precisions", type=str, default="58,60,62,64,66")
    p.add_argument("--min-benefit-recalls", type=str, default="0")
    p.add_argument("--min-opportunity-recalls", type=str, default="0")
    p.add_argument("--bad-weights", type=str, default="1.0,1.5,2.0")
    p.add_argument("--objectives", type=str, default="ratio,precision,balanced")
    p.add_argument("--bootstrap-top-k", type=int, default=20)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--target-bad-gt-5pct", type=float, default=0.5)
    p.add_argument("--rebuild-records", action="store_true")
    p.add_argument("--model-iter", type=int, default=160)
    p.add_argument("--selection-objective", type=str, default="constrained_balanced")
    p.add_argument("--threshold-search-grid", type=str, default="dense", choices=["coarse", "dense"])
    p.add_argument("--save-full-sweep", action="store_true")
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
