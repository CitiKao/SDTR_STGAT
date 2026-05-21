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
    evaluate_actions,
    make_probs,
    select_gate_config,
)
from rcog_stability_validation import (
    bootstrap_ci,
    cache_path_for,
    cprint,
    evaluate_subset,
    fit_gate_models,
    flatten_metrics,
    load_cached_pool,
    load_or_build_records,
    parse_seed_list,
    print_policy_line,
    subset_masks,
)
from real_data_uncertain_routing_experiment import configure_runtime, format_seconds, resolve_device


RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"


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


def default_variants() -> list[dict]:
    return [
        {
            "name": "baseline_cap0.50",
            "max_bad_gt_5pct": 0.50,
            "max_bad_gt_1pct": 3.0,
            "max_activation": 12.0,
            "bad_positive_weight_mult": 1.0,
        },
        {
            "name": "safe_cap0.40",
            "max_bad_gt_5pct": 0.40,
            "max_bad_gt_1pct": 3.0,
            "max_activation": 12.0,
            "bad_positive_weight_mult": 1.0,
        },
        {
            "name": "safe_cap0.30",
            "max_bad_gt_5pct": 0.30,
            "max_bad_gt_1pct": 3.0,
            "max_activation": 12.0,
            "bad_positive_weight_mult": 1.0,
        },
        {
            "name": "safe_cap0.25",
            "max_bad_gt_5pct": 0.25,
            "max_bad_gt_1pct": 3.0,
            "max_activation": 12.0,
            "bad_positive_weight_mult": 1.0,
        },
        {
            "name": "safe_cap0.30_badw2",
            "max_bad_gt_5pct": 0.30,
            "max_bad_gt_1pct": 3.0,
            "max_activation": 12.0,
            "bad_positive_weight_mult": 2.0,
        },
        {
            "name": "safe_cap0.25_badw2",
            "max_bad_gt_5pct": 0.25,
            "max_bad_gt_1pct": 3.0,
            "max_activation": 12.0,
            "bad_positive_weight_mult": 2.0,
        },
    ]


def variant_args(base_args: argparse.Namespace, variant: dict) -> argparse.Namespace:
    args = copy.copy(base_args)
    for key, value in variant.items():
        if key != "name":
            setattr(args, key, value)
    return args


def model_key(seed: int, train_mode: str, args: argparse.Namespace) -> tuple:
    return (
        int(seed),
        str(train_mode),
        float(args.opportunity_positive_weight_mult),
        float(args.opportunity_negative_weight_mult),
        float(args.benefit_positive_weight_mult),
        float(args.benefit_negative_weight_mult),
        float(args.bad_positive_weight_mult),
        float(args.bad_negative_weight_mult),
        int(args.model_iter),
    )


def train_indices_for_seed(seed: int, n: int, args: argparse.Namespace) -> tuple[np.ndarray | None, str]:
    if args.train_resample_mode == "bootstrap":
        rng = np.random.RandomState(int(seed) + 991)
        return rng.randint(0, n, size=n), "bootstrap"
    if args.train_resample_mode == "subsample":
        rng = np.random.RandomState(int(seed) + 991)
        sample_n = max(1, int(round(n * float(args.train_subsample_frac))))
        return rng.choice(n, size=sample_n, replace=False), f"subsample_{float(args.train_subsample_frac):.2f}"
    return None, "full"


def aggregate_variant(rows: list[dict], target_bad5: float) -> dict:
    def vals(key: str) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows if row.get(key) not in ("", None)], dtype=np.float64)

    ratio = vals("test_ddqn_over_pred")
    benp = vals("test_benefit_precision")
    benr = vals("test_benefit_recall")
    bad5 = vals("test_bad_gt_5pct")
    act = vals("test_activation_rate")
    opp_ratio = vals("opp_test_ddqn_over_pred")
    opp_recall = vals("opp_test_opportunity_recall")
    no_opp_ratio = vals("noopp_test_ddqn_over_pred")
    ci_low = vals("ci_mean_improvement_pct_ci95_low")
    return {
        "variant": rows[0]["variant"],
        "seeds": len(rows),
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
        "no_opportunity_ratio_mean": float(np.mean(no_opp_ratio)),
        "min_ci_improvement_low": float(np.min(ci_low)) if ci_low.size else float("nan"),
        "safety_pass": bool(np.max(bad5) <= float(target_bad5)),
        "positive_ci_pass": bool(np.min(ci_low) > 0.0) if ci_low.size else False,
    }


def print_result_line(variant: str, seed: int, stats: dict, opp_stats: dict, ci: dict) -> None:
    print(
        f"{variant:<24} seed={seed:<4} "
        f"ratio={stats['ddqn_over_pred']:.6f} benP={stats['benefit_precision']:5.1f}% "
        f"benR={stats['benefit_recall']:5.1f}% bad5={stats['bad_gt_5pct']:4.2f}% "
        f"act={stats['activation_rate']:5.1f}% opp_ratio={opp_stats['ddqn_over_pred']:.6f} "
        f"oppR={opp_stats['opportunity_recall']:5.1f}% "
        f"CI_low={ci.get('mean_improvement_pct_ci95_low', float('nan')):.3f}%",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_run_dir = Path(args.cache_run_dir)
    records_cache_dir = Path(args.records_cache_run_dir) if args.records_cache_run_dir else run_dir
    train_config = load_train_config(args.train_config)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))

    print(f"{BOLD}{CYAN}{'=' * 112}{RESET}", flush=True)
    cprint(CYAN, "RCOG safety constraint experiment")
    cprint(CYAN, f"run_dir={run_dir}")
    cprint(CYAN, f"cache_run_dir={cache_run_dir}")
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

    baseline_actions = [0 for _ in test_records]
    raw_actions = [r.raw_action for r in test_records]
    cprint(BLUE, "[REFERENCE]")
    print_policy_line("dijkstra", evaluate_actions(test_records, baseline_actions, test_eval_args, policy="dijkstra"))
    print_policy_line("raw_ddqn", evaluate_actions(test_records, raw_actions, test_eval_args, policy="raw_ddqn"))

    variants = default_variants()
    seeds = parse_seed_list(args.seeds)
    masks = subset_masks(test_records)
    opp_mask = masks["opportunity"]
    noopp_mask = masks["no_opportunity"]
    variant_rows: list[dict] = []
    all_reports: list[dict] = []
    model_cache: dict[tuple, tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    total_jobs = len(variants) * len(seeds)
    completed = 0
    start_all = time.time()

    for seed in seeds:
        train_indices, train_mode = train_indices_for_seed(seed, len(train_records), args)
        for variant in variants:
            v_args = variant_args(args, variant)
            key = model_key(seed, train_mode, v_args)
            if key not in model_cache:
                cprint(BLUE, f"[FIT] seed={seed} mode={train_mode} bad_weight={v_args.bad_positive_weight_mult}")
                models = fit_gate_models(train_records, seed=int(seed), args=v_args, train_indices=train_indices)
                model_cache[key] = (models, make_probs(models, val_records), make_probs(models, test_records))
            models, val_probs, test_probs = model_cache[key]
            best_cfg, val_sweep = select_gate_config(val_records, val_probs, val_eval_args, v_args)
            actions = actions_for_config(test_records, test_probs, best_cfg)
            stats = evaluate_actions(test_records, actions, test_eval_args, policy=variant["name"])
            ci = bootstrap_ci(test_records, actions, seed=int(seed) + 4049, iters=int(args.bootstrap_iters))
            opp_stats = evaluate_subset(test_records, actions, test_eval_args, opp_mask, policy=variant["name"]) or {}
            noopp_stats = evaluate_subset(test_records, actions, test_eval_args, noopp_mask, policy=variant["name"]) or {}
            row = {
                "variant": variant["name"],
                "seed": int(seed),
                "train_mode": train_mode,
                "val_max_bad_gt_5pct": float(v_args.max_bad_gt_5pct),
                "bad_positive_weight_mult": float(v_args.bad_positive_weight_mult),
                "selected_val_bad_gt_5pct": float(best_cfg.get("bad_gt_5pct", float("nan"))),
                "selected_val_bad_gt_1pct": float(best_cfg.get("bad_gt_1pct", float("nan"))),
                "selected_val_activation": float(best_cfg.get("activation_rate", float("nan"))),
                **flatten_metrics("test", stats),
                **flatten_metrics("opp_test", opp_stats),
                **flatten_metrics("noopp_test", noopp_stats),
                **flatten_metrics("ci", ci),
            }
            variant_rows.append(row)
            all_reports.append(
                {
                    "variant": variant,
                    "seed": int(seed),
                    "train_mode": train_mode,
                    "selected_config": best_cfg,
                    "test": stats,
                    "opportunity_subset": opp_stats,
                    "no_opportunity_subset": noopp_stats,
                    "bootstrap_ci": ci,
                }
            )
            variant_dir = run_dir / variant["name"] / f"seed_{seed}_{train_mode}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            (variant_dir / "selected_config.json").write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
            (variant_dir / "val_sweep_top.json").write_text(json.dumps(val_sweep[:100], indent=2), encoding="utf-8")
            (variant_dir / "summary.json").write_text(json.dumps(all_reports[-1], indent=2), encoding="utf-8")
            print_result_line(variant["name"], seed, stats, opp_stats, ci)
            completed += 1
            progress = completed / max(total_jobs, 1)
            elapsed = time.time() - start_all
            eta = elapsed * (1.0 - progress) / max(progress, 1e-9)
            cprint(CYAN, f"[TOTAL] {completed}/{total_jobs} elapsed={format_seconds(elapsed)} ETA={format_seconds(eta)}")

    aggregates = [aggregate_variant([r for r in variant_rows if r["variant"] == v["name"]], args.target_bad_gt_5pct) for v in variants]
    aggregates.sort(
        key=lambda r: (
            not r["safety_pass"],
            not r["positive_ci_pass"],
            r["mean_ratio_mean"],
            -r["opportunity_recall_mean"],
            r["bad_gt_5pct_max"],
        )
    )
    write_rows(run_dir / "variant_seed_metrics.csv", variant_rows)
    write_rows(run_dir / "variant_aggregate_metrics.csv", aggregates)
    summary = {
        "run_dir": str(run_dir),
        "cache_run_dir": str(cache_run_dir),
        "records_cache_dir": str(records_cache_dir),
        "target_bad_gt_5pct": float(args.target_bad_gt_5pct),
        "seeds": seeds,
        "train_resample_mode": str(args.train_resample_mode),
        "variants": variants,
        "aggregates_ranked": aggregates,
        "reports": all_reports,
        "recommendation": aggregates[0] if aggregates else {},
        "outputs": {
            "variant_seed_metrics": str(run_dir / "variant_seed_metrics.csv"),
            "variant_aggregate_metrics": str(run_dir / "variant_aggregate_metrics.csv"),
        },
    }
    out_path = run_dir / "rcog_safety_constraint_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{BOLD}{GREEN}{'=' * 112}{RESET}", flush=True)
    cprint(GREEN, f"Saved safety summary: {out_path}")
    for row in aggregates:
        color = GREEN if row["safety_pass"] and row["positive_ci_pass"] else YELLOW
        cprint(
            color,
            f"{row['variant']:<24} ratio={row['mean_ratio_mean']:.6f} "
            f"benP={row['benefit_precision_mean']:.1f}% benR={row['benefit_recall_mean']:.1f}% "
            f"bad5_mean={row['bad_gt_5pct_mean']:.2f}% bad5_max={row['bad_gt_5pct_max']:.2f}% "
            f"opp_ratio={row['opportunity_ratio_mean']:.6f} oppR={row['opportunity_recall_mean']:.1f}% "
            f"safety={row['safety_pass']}",
        )
    print(f"{BOLD}{GREEN}{'=' * 112}{RESET}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune validation-side safety constraints for RCOG-DDQN.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--cache-run-dir", type=str, required=True)
    p.add_argument("--records-cache-run-dir", type=str, default="")
    p.add_argument("--train-pool-size", type=int, default=10000)
    p.add_argument("--eval-pool-size", type=int, default=3000)
    p.add_argument("--seeds", type=str, default="121,122,123,124,125")
    p.add_argument("--train-resample-mode", type=str, default="bootstrap", choices=["full", "bootstrap", "subsample"])
    p.add_argument("--train-subsample-frac", type=float, default=0.8)
    p.add_argument("--bootstrap-iters", type=int, default=1000)
    p.add_argument("--target-bad-gt-5pct", type=float, default=0.5)
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
    p.add_argument("--max-bad-gt-1pct", type=float, default=3.0)
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
