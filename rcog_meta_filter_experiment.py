from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from evaluate_candidate_route_reranker import load_train_config
from gated_candidate_route_reranker_experiment import build_pool_for_split, build_runtime, eval_args_from_config
from no_topk_rcog_gate_experiment import (
    RCOGRecord,
    actions_for_config,
    build_agent,
    build_records,
    evaluate_actions,
    fast_stats_for_gate,
    fit_hgb_classifier,
    gate_mask_for_config,
    make_probs,
    make_search_arrays,
    q_summary,
    select_gate_config,
    simple_gate_mask,
)
from real_data_uncertain_routing_experiment import configure_runtime, format_seconds, resolve_device


class ConstantProb:
    def __init__(self, value: float):
        self.value = float(value)

    def predict_proba(self, x):
        p = np.full((len(x), 1), self.value, dtype=np.float64)
        return np.concatenate([1.0 - p, p], axis=1)


def meta_feature_matrix(records: list[RCOGRecord], probs: dict[str, np.ndarray]) -> np.ndarray:
    base_x = np.stack([r.feature for r in records]).astype(np.float32)
    p_o = probs["opportunity"].astype(np.float32)
    p_b = probs["benefit"].astype(np.float32)
    p_d = probs["bad"].astype(np.float32)
    score = p_o * p_b * (1.0 - p_d)
    q_rows = []
    for r in records:
        qs = q_summary(r.q_values, r.mask, r.raw_action)
        q_rows.append(
            [
                qs["q_margin"],
                qs["q_top2_margin"],
                qs["q_best_prob"],
                qs["q_dijkstra_prob"],
                qs["q_entropy"],
            ]
        )
    q_x = np.asarray(q_rows, dtype=np.float32)
    prob_x = np.stack(
        [
            p_o,
            p_b,
            p_d,
            score,
            p_o * p_b,
            p_b * (1.0 - p_d),
            p_b - p_d,
            p_o - p_d,
            np.maximum(p_b - 0.5, 0.0),
            np.maximum(0.5 - p_d, 0.0),
        ],
        axis=1,
    ).astype(np.float32)
    return np.nan_to_num(np.concatenate([base_x, prob_x, q_x], axis=1), nan=0.0, posinf=1e6, neginf=-1e6)


def fit_meta_classifier(x: np.ndarray, y: np.ndarray, *, seed: int, max_iter: int, pos_mult: float, neg_mult: float):
    if len(np.unique(y)) < 2:
        return ConstantProb(float(np.mean(y)) if len(y) else 0.0)
    return fit_hgb_classifier(
        x,
        y.astype(np.int32),
        seed=seed,
        max_iter=max_iter,
        positive_weight_mult=pos_mult,
        negative_weight_mult=neg_mult,
    )


def predict_prob(model, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1].astype(np.float32)


def select_meta_config(
    records: list[RCOGRecord],
    base_probs: dict[str, np.ndarray],
    meta_probs: dict[str, np.ndarray],
    base_cfg: dict,
    eval_args: argparse.Namespace,
    args: argparse.Namespace,
) -> tuple[dict, list[dict]]:
    arrays = make_search_arrays(records, eval_args)
    base_gate = gate_mask_for_config(records, base_probs, base_cfg)
    mb = meta_probs["benefit"]
    md = meta_probs["bad"]
    active_benefit = mb[base_gate]
    if active_benefit.size:
        benefit_thresholds = sorted(
            set(
                [0.0]
                + [float(np.quantile(active_benefit, q)) for q in np.linspace(0.01, 0.99, int(args.meta_threshold_quantiles))]
                + [float(x) for x in np.linspace(0.05, 0.95, 37)]
            )
        )
    else:
        benefit_thresholds = [0.0]
    bad_maxes = [float(x) for x in args.meta_bad_maxes.split(",")]
    results = []
    for bt in benefit_thresholds:
        for bm in bad_maxes:
            gate = base_gate & (mb >= bt) & (md <= bm)
            stats = fast_stats_for_gate(gate, arrays)
            results.append({"meta_benefit_threshold": float(bt), "meta_bad_max": float(bm), **stats})

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
    if str(args.selection_objective) == "recall_at_precision":
        pool.sort(
            key=lambda x: (
                -float(x.get("benefit_recall", 0.0)),
                -float(x.get("benefit_precision", 0.0)),
                float(x.get("bad_gt_1pct", float("inf"))),
                float(x.get("ddqn_over_pred", float("inf"))),
            )
        )
    else:
        pool.sort(
            key=lambda x: (
                -float(x.get("benefit_precision", 0.0)),
                -float(x.get("benefit_recall", 0.0)),
                float(x.get("bad_gt_1pct", float("inf"))),
                float(x.get("ddqn_over_pred", float("inf"))),
            )
        )
    cfg = {"base": base_cfg, "meta": pool[0]}
    return cfg, (results if args.save_full_sweep else pool[:200])


def actions_for_meta(records: list[RCOGRecord], base_probs: dict[str, np.ndarray], meta_probs: dict[str, np.ndarray], cfg: dict) -> list[int]:
    base_gate = gate_mask_for_config(records, base_probs, cfg["base"])
    meta_cfg = cfg["meta"]
    gate = (
        base_gate
        & (meta_probs["benefit"] >= float(meta_cfg["meta_benefit_threshold"]))
        & (meta_probs["bad"] <= float(meta_cfg["meta_bad_max"]))
    )
    return [int(r.raw_action) if bool(active) else 0 for r, active in zip(records, gate, strict=True)]


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
    agent = build_eval_agent(args, train_config, runtime)

    print("=" * 96, flush=True)
    print("RCOG meta-filter experiment", flush=True)
    print("Stage 1: high-recall RCOG. Stage 2: benefit/bad meta-filter inside activated region.", flush=True)
    print(f"run_dir={run_dir} device={device}", flush=True)
    print("=" * 96, flush=True)

    if not args.load_rcog_models:
        raise ValueError("--load-rcog-models is required so the meta-filter is trained on top of the selected RCOG.")
    with Path(args.load_rcog_models).open("rb") as f:
        base_models = pickle.load(f)
    print(f"Loaded base RCOG models from {args.load_rcog_models}", flush=True)

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
    train_base_probs = make_probs(base_models, train_records)

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
    val_base_probs = make_probs(base_models, val_records)

    base_select_args = argparse.Namespace(**vars(args))
    base_select_args.selection_objective = str(args.base_selection_objective)
    base_select_args.min_activation = float(args.base_min_activation)
    base_select_args.max_activation = float(args.base_max_activation)
    base_select_args.min_benefit_precision = float(args.base_min_benefit_precision)
    base_select_args.min_benefit_recall = float(args.base_min_benefit_recall)
    base_select_args.max_bad_gt_1pct = float(args.base_max_bad_gt_1pct)
    base_select_args.max_bad_gt_5pct = float(args.base_max_bad_gt_5pct)
    base_cfg, base_sweep = select_gate_config(val_records, val_base_probs, val_eval_args, base_select_args)
    print("Selected stage-1 base config on val:", flush=True)
    print(json.dumps(base_cfg, indent=2), flush=True)

    train_base_gate = gate_mask_for_config(train_records, train_base_probs, base_cfg)
    x_meta_all = meta_feature_matrix(train_records, train_base_probs)
    x_meta = x_meta_all[train_base_gate]
    y_benefit = np.asarray([r.benefit_label for r in train_records], dtype=np.int32)[train_base_gate]
    y_bad = np.asarray([r.bad_override_label for r in train_records], dtype=np.int32)[train_base_gate]
    print(
        f"Meta-filter train region: n={len(x_meta)} benefit_rate={float(np.mean(y_benefit))*100 if len(y_benefit) else 0:.2f}% "
        f"bad_rate={float(np.mean(y_bad))*100 if len(y_bad) else 0:.2f}%",
        flush=True,
    )
    start = time.time()
    meta_models = {
        "benefit": fit_meta_classifier(
            x_meta,
            y_benefit,
            seed=int(args.seed) + 101,
            max_iter=int(args.meta_model_iter),
            pos_mult=float(args.meta_benefit_positive_weight_mult),
            neg_mult=float(args.meta_benefit_negative_weight_mult),
        ),
        "bad": fit_meta_classifier(
            x_meta,
            y_bad,
            seed=int(args.seed) + 202,
            max_iter=int(args.meta_model_iter),
            pos_mult=float(args.meta_bad_positive_weight_mult),
            neg_mult=float(args.meta_bad_negative_weight_mult),
        ),
    }
    print(f"Fitted meta-filter in {format_seconds(time.time() - start)}", flush=True)
    with (run_dir / "meta_filter_models.pkl").open("wb") as f:
        pickle.dump({"base_models": base_models, "meta_models": meta_models, "base_cfg": base_cfg}, f)

    val_meta_x = meta_feature_matrix(val_records, val_base_probs)
    val_meta_probs = {
        "benefit": predict_prob(meta_models["benefit"], val_meta_x),
        "bad": predict_prob(meta_models["bad"], val_meta_x),
    }
    best_cfg, meta_sweep = select_meta_config(val_records, val_base_probs, val_meta_probs, base_cfg, val_eval_args, args)
    (run_dir / "selected_meta_filter_config.json").write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
    (run_dir / "val_meta_filter_sweep.json").write_text(json.dumps(meta_sweep, indent=2), encoding="utf-8")
    print("Selected meta-filter config on val:", flush=True)
    print(json.dumps(best_cfg, indent=2), flush=True)

    reports = {
        "checkpoint": str(args.checkpoint),
        "train_config": str(args.train_config),
        "train_pool_size": int(args.train_pool_size),
        "eval_pool_size": int(args.eval_pool_size),
        "base_rcog_models": str(args.load_rcog_models),
        "base_cfg": base_cfg,
        "selected_cfg": best_cfg,
        "splits": {},
        "leakage_note": "Meta-filter uses only decision-time RCOG features, base gate probabilities, and Q summaries. Future realized times are labels/evaluation only.",
    }

    for split, records, base_probs, eval_args in [("val", val_records, val_base_probs, val_eval_args)]:
        meta_x = meta_feature_matrix(records, base_probs)
        meta_probs = {"benefit": predict_prob(meta_models["benefit"], meta_x), "bad": predict_prob(meta_models["bad"], meta_x)}
        actions = actions_for_meta(records, base_probs, meta_probs, best_cfg)
        base_actions = actions_for_config(records, base_probs, base_cfg)
        dij_actions = [0 for _ in records]
        split_dir = run_dir / f"eval_{split}"
        split_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "meta_filter": evaluate_actions(records, actions, eval_args, policy="meta_filter", out_csv=split_dir / f"{split}_meta_filter_episode_metrics.csv"),
            "stage1_base": evaluate_actions(records, base_actions, eval_args, policy="stage1_base", out_csv=split_dir / f"{split}_stage1_base_episode_metrics.csv"),
            "dijkstra": evaluate_actions(records, dij_actions, eval_args, policy="dijkstra", out_csv=split_dir / f"{split}_dijkstra_episode_metrics.csv"),
        }
        reports["splits"][split] = report
        print(f"{split} meta_filter:", json.dumps(report["meta_filter"], indent=2), flush=True)

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
        base_probs = make_probs(base_models, records)
        meta_x = meta_feature_matrix(records, base_probs)
        meta_probs = {"benefit": predict_prob(meta_models["benefit"], meta_x), "bad": predict_prob(meta_models["bad"], meta_x)}
        actions = actions_for_meta(records, base_probs, meta_probs, best_cfg)
        base_actions = actions_for_config(records, base_probs, base_cfg)
        dij_actions = [0 for _ in records]
        report = {
            "meta_filter": evaluate_actions(records, actions, eval_args, policy="meta_filter", out_csv=split_dir / f"{split}_meta_filter_episode_metrics.csv"),
            "stage1_base": evaluate_actions(records, base_actions, eval_args, policy="stage1_base", out_csv=split_dir / f"{split}_stage1_base_episode_metrics.csv"),
            "dijkstra": evaluate_actions(records, dij_actions, eval_args, policy="dijkstra", out_csv=split_dir / f"{split}_dijkstra_episode_metrics.csv"),
        }
        reports["splits"][split] = report
        print(f"{split} meta_filter:", json.dumps(report["meta_filter"], indent=2), flush=True)

    (run_dir / "meta_filter_summary.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"Saved summary: {run_dir / 'meta_filter_summary.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RCOG second-stage meta-filter experiment.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--load-rcog-models", type=str, required=True)
    p.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--train-pool-size", type=int, default=10000)
    p.add_argument("--eval-pool-size", type=int, default=1000)
    p.add_argument("--model-iter", type=int, default=160)
    p.add_argument("--meta-model-iter", type=int, default=180)
    p.add_argument("--selection-objective", type=str, default="precision_at_recall", choices=["precision_at_recall", "recall_at_precision"])
    p.add_argument("--threshold-search-grid", type=str, default="dense", choices=["coarse", "dense"])
    p.add_argument("--save-full-sweep", action="store_true")
    p.add_argument("--meta-threshold-quantiles", type=int, default=120)
    p.add_argument("--meta-bad-maxes", type=str, default="0.05,0.10,0.15,0.20,0.30,0.40,0.50")
    p.add_argument("--meta-benefit-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--meta-benefit-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--meta-bad-positive-weight-mult", type=float, default=2.0)
    p.add_argument("--meta-bad-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--base-selection-objective", type=str, default="benefit_recall")
    p.add_argument("--base-min-activation", type=float, default=12.0)
    p.add_argument("--base-max-activation", type=float, default=22.0)
    p.add_argument("--base-min-benefit-precision", type=float, default=45.0)
    p.add_argument("--base-min-benefit-recall", type=float, default=78.0)
    p.add_argument("--base-max-bad-gt-1pct", type=float, default=5.0)
    p.add_argument("--base-max-bad-gt-5pct", type=float, default=0.8)
    p.add_argument("--benefit-delta", type=float, default=0.005)
    p.add_argument("--bad-delta", type=float, default=0.05)
    p.add_argument("--max-bad-gt-1pct", type=float, default=3.0)
    p.add_argument("--max-bad-gt-5pct", type=float, default=0.5)
    p.add_argument("--min-activation", type=float, default=8.0)
    p.add_argument("--max-activation", type=float, default=22.0)
    p.add_argument("--min-benefit-precision", type=float, default=55.0)
    p.add_argument("--min-benefit-recall", type=float, default=73.0)
    p.add_argument("--candidate-build-batch-size", type=int, default=16)
    p.add_argument("--candidate-build-log-interval", type=int, default=500)
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
