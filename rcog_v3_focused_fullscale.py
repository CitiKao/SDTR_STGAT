from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from evaluate_candidate_route_reranker import load_train_config, value
from gated_candidate_route_reranker_experiment import build_pool_for_split, build_runtime
from no_topk_rcog_gate_experiment import build_agent, build_records
from rcog_v3_suggestion_suite import (
    BOLD,
    CYAN,
    GREEN,
    RESET,
    Progress,
    cprint,
    make_summary_table,
    run_dynamic_bucket_variant,
    run_gate_variant,
)
from real_data_uncertain_routing_experiment import configure_runtime, resolve_device


def cached_pool_for_split(
    *,
    split: str,
    pool_size: int,
    run_dir: Path,
    args: argparse.Namespace,
    train_config: dict,
    runtime,
    seed_offset: int,
):
    split_dir = run_dir / f"pool_{split}_{pool_size}"
    cache_path = split_dir / "candidate_pool.pkl"
    meta_path = split_dir / "candidate_pool_cache_meta.json"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        pool = cached["pool"]
        eval_args = cached["eval_args"]
        cprint(CYAN, f"[CACHE] loaded {split} pool: {len(pool)}/{pool_size} from {cache_path}")
        return pool, eval_args

    pool, eval_args = build_pool_for_split(
        split=split,
        pool_size=pool_size,
        run_dir=run_dir,
        args=args,
        train_config=train_config,
        runtime=runtime,
        seed_offset=seed_offset,
    )
    split_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp")
    with tmp_path.open("wb") as f:
        pickle.dump({"pool": pool, "eval_args": eval_args}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(cache_path)
    meta_path.write_text(
        json.dumps(
            {
                "split": split,
                "pool_size": int(pool_size),
                "cached_records": int(len(pool)),
                "cache_path": str(cache_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cprint(CYAN, f"[CACHE] saved {split} pool: {len(pool)}/{pool_size} to {cache_path}")
    return pool, eval_args


def run(args: argparse.Namespace) -> None:
    train_config = load_train_config(args.train_config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))
    runtime = build_runtime(args, train_config, device)

    args.k_routes = int(value(args, train_config, "k_routes") or 6)
    args.min_unique_routes = int(value(args, train_config, "min_unique_routes") or 3)
    args.min_pred_hops = int(value(args, train_config, "min_pred_hops") or 2)
    args.min_pred_distance_km = float(value(args, train_config, "min_pred_distance_km") or 5.0)

    total_units = float(args.train_pool_size + 2 * args.eval_pool_size) / max(float(args.candidate_unit), 1.0)
    total_units += 2.0
    progress = Progress(total_units)

    print(f"{BOLD}{CYAN}{'=' * 104}{RESET}", flush=True)
    cprint(CYAN, "RCOG v3 focused full-scale: larger gate train + dynamic bucket threshold")
    cprint(CYAN, f"run_dir={run_dir} device={device} train_pool={args.train_pool_size} eval_pool={args.eval_pool_size}")
    print(f"{BOLD}{CYAN}{'=' * 104}{RESET}", flush=True)

    train_pool, train_eval_args = cached_pool_for_split(
        split="train",
        pool_size=int(args.train_pool_size),
        run_dir=run_dir,
        args=args,
        train_config=train_config,
        runtime=runtime,
        seed_offset=0,
    )
    train_eval_args.feature_set = "base"
    progress.add(float(args.train_pool_size) / max(float(args.candidate_unit), 1.0))
    print(progress.line("after train pool"), flush=True)

    val_pool, val_eval_args = cached_pool_for_split(
        split="val",
        pool_size=int(args.eval_pool_size),
        run_dir=run_dir / "eval_val",
        args=args,
        train_config=train_config,
        runtime=runtime,
        seed_offset=1000,
    )
    val_eval_args.feature_set = "base"
    progress.add(float(args.eval_pool_size) / max(float(args.candidate_unit), 1.0))
    print(progress.line("after val pool"), flush=True)

    test_pool, test_eval_args = cached_pool_for_split(
        split="test",
        pool_size=int(args.eval_pool_size),
        run_dir=run_dir / "eval_test",
        args=args,
        train_config=train_config,
        runtime=runtime,
        seed_offset=2000,
    )
    test_eval_args.feature_set = "base"
    progress.add(float(args.eval_pool_size) / max(float(args.candidate_unit), 1.0))
    print(progress.line("after test pool"), flush=True)

    cprint(CYAN, "[BUILD] loading current best DDQN and building full records")
    agent = build_agent(args, train_config, train_eval_args, runtime)
    train_records = build_records(
        train_pool,
        runtime=runtime,
        eval_args=train_eval_args,
        agent=agent,
        benefit_delta=float(args.benefit_delta),
        bad_delta=float(args.bad_delta),
    )
    val_records = build_records(
        val_pool,
        runtime=runtime,
        eval_args=val_eval_args,
        agent=agent,
        benefit_delta=float(args.benefit_delta),
        bad_delta=float(args.bad_delta),
    )
    test_records = build_records(
        test_pool,
        runtime=runtime,
        eval_args=test_eval_args,
        agent=agent,
        benefit_delta=float(args.benefit_delta),
        bad_delta=float(args.bad_delta),
    )

    reports = []
    reports.append(
        run_gate_variant(
            name=f"larger_gate_train{len(train_records)}_eval{len(test_records)}",
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            val_eval_args=val_eval_args,
            test_eval_args=test_eval_args,
            args=args,
            run_dir=run_dir,
            progress=progress,
            ensemble=1,
        )
    )
    reports.append(
        run_dynamic_bucket_variant(
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            val_eval_args=val_eval_args,
            test_eval_args=test_eval_args,
            args=args,
            run_dir=run_dir,
            progress=progress,
        )
    )

    summary = {
        "baseline_reference": {
            "run": "runs/rcog_v2_classifier_train10000_final_eval3000",
            "benefit_precision": 61.57760814249363,
            "benefit_recall": 78.06451612903226,
            "bad_gt_5pct": 0.4,
            "mean_ratio": 0.9974396875254729,
        },
        "train_pool_size": int(args.train_pool_size),
        "eval_pool_size": int(args.eval_pool_size),
        "reports": reports,
        "ranked_summary": make_summary_table(reports),
    }
    out_path = run_dir / "rcog_v3_focused_fullscale_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{BOLD}{GREEN}{'=' * 104}{RESET}", flush=True)
    cprint(GREEN, f"Saved summary: {out_path}")
    for row in summary["ranked_summary"]:
        print(
            f"{row['variant']:<40} ratio={row['mean_ratio']:.6f} "
            f"benP={row['benefit_precision']:.1f} benR={row['benefit_recall']:.1f} "
            f"bad5={row['bad_gt_5pct']:.1f} act={row['activation']:.1f}",
            flush=True,
        )
    print(progress.line("all done"), flush=True)
    print(f"{BOLD}{GREEN}{'=' * 104}{RESET}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Focused full-scale RCOG v3 run.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--train-pool-size", type=int, default=10000)
    p.add_argument("--eval-pool-size", type=int, default=3000)
    p.add_argument("--model-iter", type=int, default=160)
    p.add_argument("--selection-objective", type=str, default="constrained_balanced", choices=["time", "benefit_recall", "opportunity_recall", "opportunity_precision_at_recall", "opportunity_balanced", "constrained_balanced", "precision_at_recall", "two_stage_precision_at_recall"])
    p.add_argument("--threshold-search-grid", type=str, default="dense", choices=["coarse", "dense"])
    p.add_argument("--benefit-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--bad-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--benefit-delta", type=float, default=0.005)
    p.add_argument("--bad-delta", type=float, default=0.05)
    p.add_argument("--max-bad-gt-1pct", type=float, default=3.0)
    p.add_argument("--max-bad-gt-5pct", type=float, default=0.5)
    p.add_argument("--min-activation", type=float, default=6.0)
    p.add_argument("--max-activation", type=float, default=22.0)
    p.add_argument("--min-benefit-precision", type=float, default=55.0)
    p.add_argument("--min-benefit-recall", type=float, default=0.0)
    p.add_argument("--min-opportunity-precision", type=float, default=0.0)
    p.add_argument("--min-opportunity-recall", type=float, default=0.0)
    p.add_argument("--min-bucket-val-records", type=int, default=250)
    p.add_argument("--candidate-unit", type=float, default=200.0)
    p.add_argument("--candidate-build-batch-size", type=int, default=16)
    p.add_argument("--candidate-build-log-interval", type=int, default=500)
    p.add_argument("--candidate-pool-attempt-multiplier", type=int, default=1800)
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
