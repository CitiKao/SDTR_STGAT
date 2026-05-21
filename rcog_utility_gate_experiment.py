from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from evaluate_candidate_route_reranker import load_train_config
from gated_candidate_route_reranker_experiment import build_pool_for_split, build_runtime, eval_args_from_config
from no_topk_rcog_gate_experiment import (
    build_agent,
    build_records,
    evaluate_actions,
    fast_stats_for_gate,
    make_search_arrays,
    predict_prob,
)
from real_data_uncertain_routing_experiment import configure_runtime, format_seconds, resolve_device


def fit_regressor(x: np.ndarray, y: np.ndarray, *, seed: int, max_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=seed,
    ).fit(x, y)


def fit_classifier(x: np.ndarray, y: np.ndarray, *, seed: int, max_iter: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=seed,
    ).fit(x, y)


def model_predictions(models: dict, records) -> dict[str, np.ndarray]:
    x = np.stack([r.feature for r in records]).astype(np.float32)
    gain = np.maximum(models["gain"].predict(x), 0.0)
    loss = np.maximum(models["loss"].predict(x), 0.0)
    bad = predict_prob(models["bad"], x)
    return {"gain": gain, "loss": loss, "bad": bad}


def actions_for_utility(records, preds: dict[str, np.ndarray], cfg: dict) -> list[int]:
    utility = preds["gain"] - float(cfg["lambda_loss"]) * preds["loss"]
    active = (
        (utility >= float(cfg["utility_threshold"]))
        & (preds["bad"] <= float(cfg["bad_prob_max"]))
        & (np.asarray([r.raw_action for r in records]) != 0)
    )
    return [int(r.raw_action) if bool(flag) else 0 for r, flag in zip(records, active, strict=True)]


def select_utility_config(records, preds: dict[str, np.ndarray], eval_args: argparse.Namespace, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    arrays = make_search_arrays(records, eval_args)
    results: list[dict] = []
    bad_maxes = [float(x) for x in args.bad_prob_maxes.split(",")]
    lambda_losses = [float(x) for x in args.lambda_losses.split(",")]
    for lam in lambda_losses:
        utility = preds["gain"] - lam * preds["loss"]
        finite = utility[np.isfinite(utility)]
        thresholds = sorted(
            set(
                [0.0]
                + [float(np.quantile(finite, q)) for q in np.linspace(0.40, 0.995, int(args.threshold_quantiles))]
                + [float(x) for x in np.linspace(float(np.min(finite)), float(np.max(finite)), 25)]
            )
        )
        for threshold in thresholds:
            for bad_max in bad_maxes:
                gate = (utility >= threshold) & (preds["bad"] <= bad_max)
                stats = fast_stats_for_gate(gate, arrays)
                results.append(
                    {
                        "lambda_loss": float(lam),
                        "utility_threshold": float(threshold),
                        "bad_prob_max": float(bad_max),
                        **stats,
                    }
                )

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
        return True

    strict_items = [x for x in results if feasible(x, True)]
    pool = strict_items or [x for x in results if feasible(x, False)] or results
    objective = str(args.selection_objective)
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
    elif objective == "time":
        pool.sort(
            key=lambda x: (
                float(x.get("ddqn_over_pred", float("inf"))),
                -float(x.get("benefit_precision", 0.0)),
                -float(x.get("benefit_recall", 0.0)),
                float(x.get("bad_gt_1pct", float("inf"))),
            )
        )
    else:
        pool.sort(
            key=lambda x: (
                -float(x.get("benefit_recall", 0.0)),
                -float(x.get("benefit_precision", 0.0)),
                float(x.get("bad_gt_1pct", float("inf"))),
                float(x.get("ddqn_over_pred", float("inf"))),
            )
        )
    sweep = results if args.save_full_sweep else sorted(results, key=lambda x: float(x.get("ddqn_over_pred", float("inf"))))[:200]
    return pool[0], sweep


def build_eval_agent(args: argparse.Namespace, train_config: dict, runtime):
    train_eval_args = eval_args_from_config(
        args,
        train_config,
        run_dir=Path(args.run_dir) / "agent_config",
        split="train",
        pool_size=max(1, int(args.train_pool_size)),
    )
    train_eval_args.feature_set = "base"
    return build_agent(args, train_config, train_eval_args, runtime)


def run(args: argparse.Namespace) -> None:
    train_config = load_train_config(args.train_config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))
    runtime = build_runtime(args, train_config, device)

    print("=" * 96, flush=True)
    print("RCOG utility gate experiment", flush=True)
    print("Gate objective: predicted_gain - lambda_loss * predicted_loss, with bad-risk guard.", flush=True)
    print(f"checkpoint={args.checkpoint}", flush=True)
    print(f"run_dir={run_dir} device={device}", flush=True)
    print("=" * 96, flush=True)

    agent = build_eval_agent(args, train_config, runtime)
    if args.load_utility_models:
        model_path = Path(args.load_utility_models)
        with model_path.open("rb") as f:
            models = pickle.load(f)
        label_summary = {}
        print(f"Loaded utility models from {model_path}", flush=True)
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
        baseline = np.asarray([r.cand.dijkstra_pred.real_result.travel_time for r in train_records], dtype=np.float64)
        raw_time = np.asarray([r.cand.candidates[int(r.raw_action)].real_result.travel_time for r in train_records], dtype=np.float64)
        gain = np.maximum(baseline - raw_time, 0.0)
        loss = np.maximum(raw_time - baseline, 0.0)
        bad = np.asarray([r.bad_override_label for r in train_records], dtype=np.int32)
        label_summary = {
            "gain_mean": float(np.mean(gain)),
            "gain_pos_rate": float(np.mean(gain > 0.0)),
            "loss_mean": float(np.mean(loss)),
            "loss_pos_rate": float(np.mean(loss > 0.0)),
            "bad_rate": float(np.mean(bad)),
        }
        print("Train utility label summary: " + json.dumps(label_summary, indent=2), flush=True)
        start = time.time()
        models = {
            "gain": fit_regressor(x_train, gain, seed=int(args.seed), max_iter=int(args.model_iter)),
            "loss": fit_regressor(x_train, loss, seed=int(args.seed) + 17, max_iter=int(args.model_iter)),
            "bad": fit_classifier(x_train, bad, seed=int(args.seed) + 34, max_iter=int(args.model_iter)),
        }
        print(f"Fitted utility models in {format_seconds(time.time() - start)}", flush=True)
        with (run_dir / "utility_models.pkl").open("wb") as f:
            pickle.dump(models, f)

    reports = {
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "train_pool_size": int(args.train_pool_size),
        "eval_pool_size": int(args.eval_pool_size),
        "feature_set": "no_topk_rcog_utility",
        "label_summary": label_summary,
        "selection_objective": str(args.selection_objective),
        "leakage_note": "Utility gate features use decision-time predictions/history/Q-values only. Realized future speeds are used only for train labels and final evaluation.",
        "splits": {},
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
    val_preds = model_predictions(models, val_records)
    best_cfg, val_sweep = select_utility_config(val_records, val_preds, val_eval_args, args)
    (run_dir / "val_utility_threshold_sweep.json").write_text(json.dumps(val_sweep, indent=2), encoding="utf-8")
    (run_dir / "selected_utility_config.json").write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
    print("Selected utility config on val:", flush=True)
    print(json.dumps(best_cfg, indent=2), flush=True)

    for split, records, preds, eval_args in [("val", val_records, val_preds, val_eval_args)]:
        actions = actions_for_utility(records, preds, best_cfg)
        raw_actions = [r.raw_action for r in records]
        dij_actions = [0 for _ in records]
        split_dir = run_dir / f"eval_{split}"
        split_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "utility": evaluate_actions(records, actions, eval_args, policy="utility", out_csv=split_dir / f"{split}_utility_episode_metrics.csv"),
            "raw_ddqn": evaluate_actions(records, raw_actions, eval_args, policy="raw_ddqn", out_csv=split_dir / f"{split}_raw_ddqn_episode_metrics.csv"),
            "dijkstra": evaluate_actions(records, dij_actions, eval_args, policy="dijkstra", out_csv=split_dir / f"{split}_dijkstra_episode_metrics.csv"),
        }
        reports["splits"][split] = report
        print(f"{split} utility:", json.dumps(report["utility"], indent=2), flush=True)

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
        preds = model_predictions(models, records)
        actions = actions_for_utility(records, preds, best_cfg)
        raw_actions = [r.raw_action for r in records]
        dij_actions = [0 for _ in records]
        report = {
            "utility": evaluate_actions(records, actions, eval_args, policy="utility", out_csv=split_dir / f"{split}_utility_episode_metrics.csv"),
            "raw_ddqn": evaluate_actions(records, raw_actions, eval_args, policy="raw_ddqn", out_csv=split_dir / f"{split}_raw_ddqn_episode_metrics.csv"),
            "dijkstra": evaluate_actions(records, dij_actions, eval_args, policy="dijkstra", out_csv=split_dir / f"{split}_dijkstra_episode_metrics.csv"),
        }
        reports["splits"][split] = report
        print(f"{split} utility:", json.dumps(report["utility"], indent=2), flush=True)

    (run_dir / "utility_summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Saved summary: {run_dir / 'utility_summary.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-topk RCOG utility gate.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--train-pool-size", type=int, default=2000)
    p.add_argument("--eval-pool-size", type=int, default=1000)
    p.add_argument("--model-iter", type=int, default=160)
    p.add_argument("--load-utility-models", type=str, default="")
    p.add_argument("--selection-objective", type=str, default="constrained_balanced", choices=["time", "benefit_recall", "constrained_balanced", "precision_at_recall"])
    p.add_argument("--lambda-losses", type=str, default="1.0,1.5,2.0,3.0,5.0")
    p.add_argument("--bad-prob-maxes", type=str, default="0.10,0.15,0.20,0.25,0.30,0.40")
    p.add_argument("--threshold-quantiles", type=int, default=80)
    p.add_argument("--save-full-sweep", action="store_true")
    p.add_argument("--benefit-delta", type=float, default=0.005)
    p.add_argument("--bad-delta", type=float, default=0.05)
    p.add_argument("--max-bad-gt-1pct", type=float, default=3.0)
    p.add_argument("--max-bad-gt-5pct", type=float, default=0.5)
    p.add_argument("--min-activation", type=float, default=12.0)
    p.add_argument("--max-activation", type=float, default=20.0)
    p.add_argument("--min-benefit-precision", type=float, default=55.0)
    p.add_argument("--min-benefit-recall", type=float, default=70.0)
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
