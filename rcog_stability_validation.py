from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np

from candidate_route_reranker_experiment import RouteChoiceCandidate
from evaluate_candidate_route_reranker import load_train_config
from gated_candidate_route_reranker_experiment import build_runtime
from no_topk_rcog_gate_experiment import (
    RCOGRecord,
    actions_for_config,
    build_agent,
    build_records,
    evaluate_actions,
    fit_hgb_classifier,
    make_probs,
    select_gate_config,
)
from real_data_uncertain_routing_experiment import configure_runtime, format_seconds, resolve_device


RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"


def cprint(color: str, msg: str) -> None:
    print(f"{color}{msg}{RESET}", flush=True)


def parse_seed_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def load_cached_pool(path: Path) -> tuple[list[RouteChoiceCandidate], argparse.Namespace]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    pool = obj["pool"]
    eval_args = obj["eval_args"]
    eval_args.feature_set = "base"
    return pool, eval_args


def cache_path_for(cache_run_dir: Path, split: str, size: int) -> Path:
    if split == "train":
        return cache_run_dir / f"pool_train_{size}" / "candidate_pool.pkl"
    return cache_run_dir / f"eval_{split}" / f"pool_{split}_{size}" / "candidate_pool.pkl"


def load_or_build_records(
    *,
    name: str,
    pool: list[RouteChoiceCandidate],
    runtime,
    eval_args: argparse.Namespace,
    agent,
    args: argparse.Namespace,
    run_dir: Path,
) -> list[RCOGRecord]:
    cache_path = run_dir / f"records_{name}.pkl"
    if cache_path.exists() and not args.rebuild_records:
        with cache_path.open("rb") as f:
            records = pickle.load(f)
        cprint(CYAN, f"[CACHE] loaded {name} records: {len(records)} from {cache_path}")
        return records
    start = time.time()
    cprint(BLUE, f"[BUILD] building {name} records from {len(pool)} candidates")
    records = build_records(
        pool,
        runtime=runtime,
        eval_args=eval_args,
        agent=agent,
        benefit_delta=float(args.benefit_delta),
        bad_delta=float(args.bad_delta),
    )
    with cache_path.open("wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    cprint(CYAN, f"[CACHE] saved {name} records: {len(records)} in {format_seconds(time.time() - start)}")
    return records


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


def fit_gate_models(
    train_records: list[RCOGRecord],
    *,
    seed: int,
    args: argparse.Namespace,
    train_indices: np.ndarray | None = None,
) -> dict:
    if train_indices is None:
        selected = train_records
    else:
        selected = [train_records[int(i)] for i in train_indices]
    x_train = np.stack([r.feature for r in selected]).astype(np.float32)
    labels = {
        "opportunity": np.asarray([r.opportunity_label for r in selected], dtype=np.int32),
        "benefit": np.asarray([r.benefit_label for r in selected], dtype=np.int32),
        "bad": np.asarray([r.bad_override_label for r in selected], dtype=np.int32),
    }
    weight_cfg = {
        "opportunity": (float(args.opportunity_positive_weight_mult), float(args.opportunity_negative_weight_mult)),
        "benefit": (float(args.benefit_positive_weight_mult), float(args.benefit_negative_weight_mult)),
        "bad": (float(args.bad_positive_weight_mult), float(args.bad_negative_weight_mult)),
    }
    models = {}
    for idx, (name, y) in enumerate(labels.items()):
        models[name] = fit_hgb_classifier(
            x_train,
            y,
            seed=int(seed) + idx * 17,
            max_iter=int(args.model_iter),
            positive_weight_mult=weight_cfg[name][0],
            negative_weight_mult=weight_cfg[name][1],
        )
    return models


def oracle_actions(records: list[RCOGRecord]) -> list[int]:
    out = []
    for rec in records:
        oracle_path = rec.cand.candidate_oracle.path
        action = 0
        for idx, route in enumerate(rec.cand.candidates):
            if route.path == oracle_path:
                action = idx
                break
        out.append(int(action))
    return out


def chosen_times(records: list[RCOGRecord], actions: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chosen = []
    baseline = []
    oracle = []
    for rec, action in zip(records, actions, strict=True):
        action = int(np.clip(action, 0, len(rec.cand.candidates) - 1))
        chosen.append(float(rec.cand.candidates[action].real_result.travel_time))
        baseline.append(float(rec.cand.dijkstra_pred.real_result.travel_time))
        oracle.append(float(rec.cand.candidate_oracle.real_result.travel_time))
    return (
        np.asarray(chosen, dtype=np.float64),
        np.asarray(baseline, dtype=np.float64),
        np.asarray(oracle, dtype=np.float64),
    )


def bootstrap_ci(
    records: list[RCOGRecord],
    actions: list[int],
    *,
    seed: int,
    iters: int,
) -> dict:
    chosen, baseline, oracle = chosen_times(records, actions)
    ratio = chosen / np.maximum(baseline, 1e-9)
    improvement = 1.0 - ratio
    oracle_gap = chosen / np.maximum(oracle, 1e-9)
    n = len(ratio)
    rng = np.random.RandomState(int(seed))
    if n == 0 or iters <= 0:
        return {}
    sample_ratio = np.empty(int(iters), dtype=np.float64)
    sample_improvement = np.empty(int(iters), dtype=np.float64)
    sample_oracle_gap = np.empty(int(iters), dtype=np.float64)
    for i in range(int(iters)):
        idx = rng.randint(0, n, size=n)
        sample_ratio[i] = float(np.mean(ratio[idx]))
        sample_improvement[i] = float(np.mean(improvement[idx]))
        sample_oracle_gap[i] = float(np.mean(oracle_gap[idx]))
    return {
        "mean_ratio": float(np.mean(ratio)),
        "mean_ratio_ci95_low": float(np.quantile(sample_ratio, 0.025)),
        "mean_ratio_ci95_high": float(np.quantile(sample_ratio, 0.975)),
        "mean_improvement_pct": float(np.mean(improvement) * 100.0),
        "mean_improvement_pct_ci95_low": float(np.quantile(sample_improvement, 0.025) * 100.0),
        "mean_improvement_pct_ci95_high": float(np.quantile(sample_improvement, 0.975) * 100.0),
        "prob_mean_ratio_below_1": float(np.mean(sample_ratio < 1.0)),
        "mean_oracle_gap": float(np.mean(oracle_gap)),
        "mean_oracle_gap_ci95_low": float(np.quantile(sample_oracle_gap, 0.025)),
        "mean_oracle_gap_ci95_high": float(np.quantile(sample_oracle_gap, 0.975)),
        "bootstrap_iters": int(iters),
    }


def subset_masks(records: list[RCOGRecord]) -> dict[str, np.ndarray]:
    n = len(records)
    hours = np.asarray([int(r.meta_row.get("hour", -1)) for r in records], dtype=np.int32)
    months = np.asarray([str(getattr(r.meta_row.get("date"), "strftime", lambda _fmt: r.meta_row.get("date"))("%Y-%m")) for r in records])
    distances = np.asarray([float(r.cand.dijkstra_pred.real_result.travel_dist) for r in records], dtype=np.float64)
    volatility = np.asarray([float(r.cand.dijkstra_pred.pred_speed_std) for r in records], dtype=np.float64)
    dispatch_count = np.asarray([float(r.cand.dispatch_count) for r in records], dtype=np.float64)
    opportunity = np.asarray([bool(r.opportunity_label) for r in records], dtype=bool)

    masks: dict[str, np.ndarray] = {
        "all": np.ones(n, dtype=bool),
        "opportunity": opportunity,
        "no_opportunity": ~opportunity,
        "peak": np.isin(hours, np.asarray([7, 8, 9, 16, 17, 18, 19])),
        "day": np.isin(hours, np.asarray([10, 11, 12, 13, 14, 15])),
        "offpeak": ~(np.isin(hours, np.asarray([7, 8, 9, 16, 17, 18, 19, 10, 11, 12, 13, 14, 15]))),
    }
    for label, values in [("distance", distances), ("volatility", volatility), ("dispatch", dispatch_count)]:
        finite = values[np.isfinite(values)]
        if finite.size:
            med = float(np.median(finite))
            masks[f"{label}_low"] = values <= med
            masks[f"{label}_high"] = values > med
    for month in sorted(set(months.tolist())):
        masks[f"month_{month}"] = months == month
    return masks


def evaluate_subset(
    records: list[RCOGRecord],
    actions: list[int],
    eval_args: argparse.Namespace,
    mask: np.ndarray,
    *,
    policy: str,
) -> dict | None:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    sub_records = [records[int(i)] for i in idx]
    sub_actions = [actions[int(i)] for i in idx]
    stats = evaluate_actions(sub_records, sub_actions, eval_args, policy=policy, out_csv=None)
    stats["n"] = int(idx.size)
    return stats


def flatten_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in metrics.items() if isinstance(v, (int, float, str))}


def print_policy_line(name: str, stats: dict) -> None:
    print(
        f"{name:<22} ratio={stats['ddqn_over_pred']:.6f} "
        f"benP={stats.get('benefit_precision', 0.0):5.1f}% benR={stats.get('benefit_recall', 0.0):5.1f}% "
        f"oppP={stats.get('opportunity_precision', 0.0):5.1f}% oppR={stats.get('opportunity_recall', 0.0):5.1f}% "
        f"bad5={stats.get('bad_gt_5pct', 0.0):4.1f}% act={stats.get('activation_rate', 0.0):5.1f}%",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_run_dir = Path(args.cache_run_dir)
    train_config = load_train_config(args.train_config)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))

    print(f"{BOLD}{CYAN}{'=' * 108}{RESET}", flush=True)
    cprint(CYAN, "RCOG stability validation: multi-seed, independent cached test, bootstrap CI, subset analysis")
    cprint(CYAN, f"run_dir={run_dir}")
    cprint(CYAN, f"cache_run_dir={cache_run_dir}")
    print(f"{BOLD}{CYAN}{'=' * 108}{RESET}", flush=True)

    runtime = build_runtime(args, train_config, device)
    train_pool, train_eval_args = load_cached_pool(cache_path_for(cache_run_dir, "train", int(args.train_pool_size)))
    val_pool, val_eval_args = load_cached_pool(cache_path_for(cache_run_dir, "val", int(args.eval_pool_size)))
    test_pool, test_eval_args = load_cached_pool(cache_path_for(cache_run_dir, "test", int(args.eval_pool_size)))
    cprint(CYAN, f"[CACHE] pools loaded: train={len(train_pool)} val={len(val_pool)} test={len(test_pool)}")

    agent = build_agent(args, train_config, train_eval_args, runtime)
    train_records = load_or_build_records(
        name="train",
        pool=train_pool,
        runtime=runtime,
        eval_args=train_eval_args,
        agent=agent,
        args=args,
        run_dir=run_dir,
    )
    val_records = load_or_build_records(
        name="val",
        pool=val_pool,
        runtime=runtime,
        eval_args=val_eval_args,
        agent=agent,
        args=args,
        run_dir=run_dir,
    )
    test_records = load_or_build_records(
        name="test",
        pool=test_pool,
        runtime=runtime,
        eval_args=test_eval_args,
        agent=agent,
        args=args,
        run_dir=run_dir,
    )

    raw_actions = [r.raw_action for r in test_records]
    dijkstra_actions = [0 for _ in test_records]
    oracle = oracle_actions(test_records)
    policy_comparison = {
        "dijkstra": evaluate_actions(test_records, dijkstra_actions, test_eval_args, policy="dijkstra", out_csv=None),
        "raw_ddqn": evaluate_actions(test_records, raw_actions, test_eval_args, policy="raw_ddqn", out_csv=None),
        "candidate_oracle": evaluate_actions(test_records, oracle, test_eval_args, policy="candidate_oracle", out_csv=None),
    }

    cprint(BLUE, "[BASELINES] independent cached test pool")
    for name, stats in policy_comparison.items():
        print_policy_line(name, stats)

    seed_rows: list[dict] = []
    subset_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    seed_reports = []
    seed_list = parse_seed_list(args.seeds)
    total_start = time.time()
    masks = subset_masks(test_records)

    for seed_idx, model_seed in enumerate(seed_list, start=1):
        seed_start = time.time()
        if args.train_resample_mode == "bootstrap":
            rng = np.random.RandomState(int(model_seed) + 991)
            train_indices = rng.randint(0, len(train_records), size=len(train_records))
            train_mode = "bootstrap"
        elif args.train_resample_mode == "subsample":
            rng = np.random.RandomState(int(model_seed) + 991)
            sample_n = max(1, int(round(len(train_records) * float(args.train_subsample_frac))))
            train_indices = rng.choice(len(train_records), size=sample_n, replace=False)
            train_mode = f"subsample_{float(args.train_subsample_frac):.2f}"
        else:
            train_indices = None
            train_mode = "full"
        cprint(BLUE, f"[SEED {seed_idx}/{len(seed_list)}] model_seed={model_seed} train_mode={train_mode}")
        models = fit_gate_models(train_records, seed=int(model_seed), args=args, train_indices=train_indices)
        val_probs = make_probs(models, val_records)
        test_probs = make_probs(models, test_records)
        best_cfg, val_sweep = select_gate_config(val_records, val_probs, val_eval_args, args)
        actions = actions_for_config(test_records, test_probs, best_cfg)
        seed_dir = run_dir / f"seed_{model_seed}_{train_mode}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        with (seed_dir / "models.pkl").open("wb") as f:
            pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
        (seed_dir / "selected_config.json").write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
        (seed_dir / "val_sweep_top.json").write_text(json.dumps(val_sweep[:100], indent=2), encoding="utf-8")
        test_stats = evaluate_actions(
            test_records,
            actions,
            test_eval_args,
            policy=f"rcog_seed_{model_seed}",
            out_csv=seed_dir / "test_rcog_episode_metrics.csv",
        )
        ci = bootstrap_ci(test_records, actions, seed=int(model_seed) + 2027, iters=int(args.bootstrap_iters))
        seed_elapsed = time.time() - seed_start
        report = {
            "seed": int(model_seed),
            "train_mode": train_mode,
            "elapsed": format_seconds(seed_elapsed),
            "selected_config": best_cfg,
            "test": test_stats,
            "bootstrap_ci": ci,
        }
        (seed_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        seed_reports.append(report)
        row = {
            "seed": int(model_seed),
            "train_mode": train_mode,
            "elapsed": format_seconds(seed_elapsed),
            **flatten_metrics("test", test_stats),
            **flatten_metrics("ci", ci),
        }
        seed_rows.append(row)
        bootstrap_rows.append({"seed": int(model_seed), **ci})
        print_policy_line(f"RCOG seed {model_seed}", test_stats)
        cprint(
            GREEN if ci.get("prob_mean_ratio_below_1", 0.0) >= 0.95 else YELLOW,
            f"[CI] mean improvement={ci.get('mean_improvement_pct', float('nan')):.3f}% "
            f"95%CI=({ci.get('mean_improvement_pct_ci95_low', float('nan')):.3f}, "
            f"{ci.get('mean_improvement_pct_ci95_high', float('nan')):.3f}) "
            f"P(mean ratio<1)={ci.get('prob_mean_ratio_below_1', float('nan')):.3f}",
        )
        for subset_name, mask in masks.items():
            stats = evaluate_subset(test_records, actions, test_eval_args, mask, policy=f"rcog_seed_{model_seed}")
            if stats is None:
                continue
            subset_rows.append(
                {
                    "seed": int(model_seed),
                    "train_mode": train_mode,
                    "subset": subset_name,
                    **flatten_metrics("test", stats),
                }
            )
        progress = seed_idx / max(len(seed_list), 1)
        elapsed = time.time() - total_start
        eta = elapsed * (1.0 - progress) / max(progress, 1e-9)
        cprint(CYAN, f"[TOTAL] {seed_idx}/{len(seed_list)} elapsed={format_seconds(elapsed)} ETA={format_seconds(eta)}")

    write_rows(run_dir / "seed_metrics.csv", seed_rows)
    write_rows(run_dir / "subset_metrics.csv", subset_rows)
    write_rows(run_dir / "bootstrap_ci.csv", bootstrap_rows)
    write_rows(
        run_dir / "policy_comparison.csv",
        [{"policy": name, **flatten_metrics("test", stats)} for name, stats in policy_comparison.items()],
    )

    ratios = np.asarray([float(r["test"]["ddqn_over_pred"]) for r in seed_reports], dtype=np.float64)
    benefit_precision = np.asarray([float(r["test"]["benefit_precision"]) for r in seed_reports], dtype=np.float64)
    benefit_recall = np.asarray([float(r["test"]["benefit_recall"]) for r in seed_reports], dtype=np.float64)
    bad5 = np.asarray([float(r["test"]["bad_gt_5pct"]) for r in seed_reports], dtype=np.float64)
    summary = {
        "run_dir": str(run_dir),
        "cache_run_dir": str(cache_run_dir),
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "train_pool_size": int(args.train_pool_size),
        "eval_pool_size": int(args.eval_pool_size),
        "seeds": seed_list,
        "train_resample_mode": str(args.train_resample_mode),
        "selection_objective": str(args.selection_objective),
        "baseline_reference": {
            "run": "runs/rcog_v2_classifier_train10000_final_eval3000",
            "benefit_precision": 61.57760814249363,
            "benefit_recall": 78.06451612903226,
            "bad_gt_5pct": 0.4,
            "mean_ratio": 0.9974396875254729,
            "note": "Original result used the earlier test candidate pool. This validation uses the cached independent full-scale test pool unless stated otherwise.",
        },
        "policy_comparison": policy_comparison,
        "seed_reports": seed_reports,
        "aggregate": {
            "mean_ratio_mean": float(np.mean(ratios)),
            "mean_ratio_std": float(np.std(ratios, ddof=1)) if ratios.size > 1 else 0.0,
            "mean_ratio_min": float(np.min(ratios)),
            "mean_ratio_max": float(np.max(ratios)),
            "benefit_precision_mean": float(np.mean(benefit_precision)),
            "benefit_precision_std": float(np.std(benefit_precision, ddof=1)) if benefit_precision.size > 1 else 0.0,
            "benefit_recall_mean": float(np.mean(benefit_recall)),
            "benefit_recall_std": float(np.std(benefit_recall, ddof=1)) if benefit_recall.size > 1 else 0.0,
            "bad_gt_5pct_mean": float(np.mean(bad5)),
            "bad_gt_5pct_max": float(np.max(bad5)),
        },
        "outputs": {
            "seed_metrics": str(run_dir / "seed_metrics.csv"),
            "subset_metrics": str(run_dir / "subset_metrics.csv"),
            "bootstrap_ci": str(run_dir / "bootstrap_ci.csv"),
            "policy_comparison": str(run_dir / "policy_comparison.csv"),
        },
        "leakage_note": "Gate inputs are decision-time features/Q-values/history. Realized speeds are used for train labels, validation threshold selection, final evaluation, and bootstrap analysis only.",
    }
    out_path = run_dir / "rcog_stability_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{BOLD}{GREEN}{'=' * 108}{RESET}", flush=True)
    cprint(GREEN, f"Saved stability summary: {out_path}")
    cprint(
        GREEN if float(np.mean(ratios)) < 1.0 else YELLOW,
        f"Aggregate RCOG mean ratio={np.mean(ratios):.6f} +/- {np.std(ratios, ddof=1) if ratios.size > 1 else 0.0:.6f}; "
        f"benP={np.mean(benefit_precision):.1f}%; benR={np.mean(benefit_recall):.1f}%; bad5 max={np.max(bad5):.2f}%",
    )
    print(f"{BOLD}{GREEN}{'=' * 108}{RESET}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the best no-topk RCOG-DDQN result for stability and significance.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--cache-run-dir", type=str, required=True)
    p.add_argument("--train-pool-size", type=int, default=10000)
    p.add_argument("--eval-pool-size", type=int, default=3000)
    p.add_argument("--seeds", type=str, default="121,122,123,124,125")
    p.add_argument("--train-resample-mode", type=str, default="full", choices=["full", "bootstrap", "subsample"])
    p.add_argument("--train-subsample-frac", type=float, default=0.8)
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--rebuild-records", action="store_true")
    p.add_argument("--model-iter", type=int, default=160)
    p.add_argument("--selection-objective", type=str, default="constrained_balanced", choices=["time", "benefit_recall", "opportunity_recall", "opportunity_precision_at_recall", "opportunity_balanced", "constrained_balanced", "precision_at_recall", "two_stage_precision_at_recall"])
    p.add_argument("--threshold-search-grid", type=str, default="dense", choices=["coarse", "dense"])
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
