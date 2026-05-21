from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from candidate_route_reranker_experiment import (
    CandidateDoubleDQNAgent,
    RouteChoiceCandidate,
    build_state_and_mask,
    choose_reward,
    state_dim_for_args,
)
from evaluate_candidate_route_reranker import load_train_config, value
from gated_candidate_route_reranker_experiment import build_pool_for_split, build_runtime
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
from rcog_enhanced_ablation_experiment import (
    add_history_features,
    add_perturb_features,
    clone_records_with_features,
    fit_gate_models,
    record_matrix,
    route_candidate_features,
    stable_variant_seed,
)
from real_data_uncertain_routing_experiment import configure_runtime, format_seconds, resolve_device


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"


class Progress:
    def __init__(self, total_units: float) -> None:
        self.total_units = float(total_units)
        self.done_units = 0.0
        self.start = time.time()

    def add(self, units: float) -> None:
        self.done_units += float(units)

    def line(self, label: str) -> str:
        elapsed = time.time() - self.start
        progress = self.done_units / max(self.total_units, 1e-9)
        eta = elapsed * (1.0 - progress) / max(progress, 1e-9) if progress > 0 else float("nan")
        eta_txt = format_seconds(eta) if math.isfinite(eta) else "unknown"
        pct = 100.0 * progress
        return (
            f"{CYAN}[TOTAL]{RESET} {label} "
            f"{BOLD}{self.done_units:.1f}/{self.total_units:.1f}{RESET} "
            f"({pct:5.1f}%) elapsed={format_seconds(elapsed)} total_ETA={eta_txt}"
        )


def cprint(color: str, msg: str) -> None:
    print(f"{color}{msg}{RESET}", flush=True)


def value_or(train_config: dict, name: str, default):
    raw = train_config.get(name)
    return default if raw is None else raw


def make_train_agent(args: argparse.Namespace, train_config: dict, eval_args: argparse.Namespace, runtime) -> CandidateDoubleDQNAgent:
    return CandidateDoubleDQNAgent(
        num_nodes=runtime.num_nodes,
        action_dim=int(eval_args.k_routes),
        state_dim=state_dim_for_args(eval_args),
        embed_dim=int(value(args, train_config, "embed_dim") or 16),
        hidden_dim=int(value(args, train_config, "hidden_dim") or 128),
        lr=float(value(args, train_config, "lr") or 1e-3),
        gamma=float(value(args, train_config, "gamma") or 0.0),
        epsilon_start=float(args.ddqn_epsilon_start),
        epsilon_end=float(args.ddqn_epsilon_end),
        epsilon_decay=int(args.ddqn_epsilon_decay),
        buffer_capacity=int(args.ddqn_buffer_capacity),
        batch_size=int(args.ddqn_batch_size),
        target_update=int(args.ddqn_target_update),
        device=str(runtime.device),
    )


def bad_risk_reward(cand: RouteChoiceCandidate, chosen, args: argparse.Namespace, eval_args: argparse.Namespace) -> float:
    reward = choose_reward(cand, chosen, eval_args)
    baseline = max(float(cand.dijkstra_pred.real_result.travel_time), 1e-9)
    chosen_time = float(chosen.real_result.travel_time)
    is_override = chosen.path != cand.dijkstra_pred.path
    if is_override:
        rel_loss = max(chosen_time / baseline - 1.0, 0.0)
        if rel_loss > 0.0:
            reward -= float(args.ddqn_bad_loss_weight) * rel_loss * float(value_or(vars(eval_args), "reward_scale", 10.0))
        if chosen_time > baseline * (1.0 + float(args.bad_delta)):
            reward -= float(args.ddqn_bad_gt5_penalty)
        if chosen_time >= baseline * (1.0 - float(args.benefit_delta)):
            reward -= float(args.ddqn_no_gain_penalty)
    return float(reward)


def train_bad_risk_ddqn(
    *,
    train_pool: list[RouteChoiceCandidate],
    runtime,
    train_config: dict,
    eval_args: argparse.Namespace,
    args: argparse.Namespace,
    run_dir: Path,
    progress: Progress,
) -> CandidateDoubleDQNAgent:
    cprint(MAGENTA, f"[DDQN] bad-risk reward retraining start episodes={args.ddqn_episodes}")
    agent = make_train_agent(args, train_config, eval_args, runtime)
    rng = np.random.RandomState(int(args.seed) + 7777)
    zero_next_mask = np.zeros(int(eval_args.k_routes), dtype=np.float32)
    start = time.time()
    rows = []
    for ep in range(1, int(args.ddqn_episodes) + 1):
        cand = train_pool[int(rng.randint(0, len(train_pool)))]
        meta_row = runtime.time_meta.iloc[cand.target_t]
        state, mask = build_state_and_mask(cand, meta_row=meta_row, args=eval_args)
        action = agent.select_action(state, mask, greedy=False)
        action = int(np.clip(action, 0, len(cand.candidates) - 1))
        reward = bad_risk_reward(cand, cand.candidates[action], args, eval_args)
        agent.store_transition(state, action, reward, state.copy(), True, mask, zero_next_mask)
        loss = agent.learn()
        rows.append((reward, loss if loss is not None else float("nan")))
        if ep % int(args.ddqn_log_interval) == 0 or ep == int(args.ddqn_episodes):
            elapsed = time.time() - start
            prog = ep / max(int(args.ddqn_episodes), 1)
            eta = elapsed * (1.0 - prog) / max(prog, 1e-9)
            recent = rows[-int(args.ddqn_log_interval) :]
            reward_mean = float(np.mean([r for r, _ in recent])) if recent else 0.0
            finite_losses = [l for _, l in recent if math.isfinite(l)]
            loss_mean = float(np.mean(finite_losses)) if finite_losses else float("nan")
            cprint(
                MAGENTA,
                f"[DDQN] {ep}/{args.ddqn_episodes} ({prog*100:5.1f}%) "
                f"ETA={format_seconds(eta)} elapsed={format_seconds(elapsed)} "
                f"eps={agent.epsilon:.3f} reward={reward_mean:.3f} loss={loss_mean:.4f}",
            )
    agent.save(run_dir / "bad_risk_ddqn.pt")
    progress.add(float(args.ddqn_episodes) / max(float(args.ddqn_episode_unit), 1.0))
    print(progress.line("after bad-risk DDQN"), flush=True)
    return agent


def avg_probs(prob_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = prob_list[0].keys()
    return {k: np.mean([p[k] for p in prob_list], axis=0).astype(np.float32) for k in keys}


def fit_probs(
    *,
    train_records: list[RCOGRecord],
    val_records: list[RCOGRecord],
    test_records: list[RCOGRecord],
    args: argparse.Namespace,
    name: str,
    ensemble: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], object]:
    models = []
    for i in range(int(ensemble)):
        models.append(
            fit_gate_models(
                train_records,
                seed=stable_variant_seed(int(args.seed) + i * 997, name),
                max_iter=int(args.model_iter),
                benefit_negative_mult=float(args.benefit_negative_weight_mult),
                bad_positive_mult=float(args.bad_positive_weight_mult),
            )
        )
    val_probs = avg_probs([make_probs(m, val_records) for m in models])
    test_probs = avg_probs([make_probs(m, test_records) for m in models])
    return val_probs, test_probs, models


def run_gate_variant(
    *,
    name: str,
    train_records: list[RCOGRecord],
    val_records: list[RCOGRecord],
    test_records: list[RCOGRecord],
    val_eval_args: argparse.Namespace,
    test_eval_args: argparse.Namespace,
    args: argparse.Namespace,
    run_dir: Path,
    progress: Progress,
    ensemble: int = 1,
) -> dict:
    stage_start = time.time()
    cprint(BLUE, f"[VARIANT] {name} start ensemble={ensemble}")
    val_probs, test_probs, models = fit_probs(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        args=args,
        name=name,
        ensemble=ensemble,
    )
    best_cfg, sweep = select_gate_config(val_records, val_probs, val_eval_args, args)
    actions = actions_for_config(test_records, test_probs, best_cfg)
    variant_dir = run_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "selected_config.json").write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
    (variant_dir / "val_sweep_top.json").write_text(json.dumps(sweep[:50], indent=2), encoding="utf-8")
    with (variant_dir / "models.pkl").open("wb") as f:
        pickle.dump(models, f)
    report = {
        "variant": name,
        "ensemble": int(ensemble),
        "elapsed": format_seconds(time.time() - stage_start),
        "selected_config": best_cfg,
        "test": evaluate_actions(test_records, actions, test_eval_args, policy=name, out_csv=variant_dir / "test_episode_metrics.csv"),
    }
    (variant_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_metric_line(name, report["test"])
    progress.add(1.0)
    print(progress.line(f"after {name}"), flush=True)
    return report


def bucket_name(hour: int) -> str:
    if hour in {7, 8, 9, 16, 17, 18, 19}:
        return "peak"
    if hour in {10, 11, 12, 13, 14, 15}:
        return "day"
    return "offpeak"


def subset_records(records: list[RCOGRecord], indices: list[int]) -> list[RCOGRecord]:
    return [records[i] for i in indices]


def subset_probs(probs: dict[str, np.ndarray], indices: list[int]) -> dict[str, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    return {k: v[idx] for k, v in probs.items()}


def run_dynamic_bucket_variant(
    *,
    train_records: list[RCOGRecord],
    val_records: list[RCOGRecord],
    test_records: list[RCOGRecord],
    val_eval_args: argparse.Namespace,
    test_eval_args: argparse.Namespace,
    args: argparse.Namespace,
    run_dir: Path,
    progress: Progress,
) -> dict:
    name = "dynamic_bucket_threshold"
    cprint(BLUE, f"[VARIANT] {name} start")
    val_probs, test_probs, models = fit_probs(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        args=args,
        name=name,
        ensemble=1,
    )
    all_cfg, _ = select_gate_config(val_records, val_probs, val_eval_args, args)
    val_buckets: dict[str, list[int]] = {"peak": [], "day": [], "offpeak": []}
    test_buckets: dict[str, list[int]] = {"peak": [], "day": [], "offpeak": []}
    for i, rec in enumerate(val_records):
        val_buckets[bucket_name(int(rec.meta_row.get("hour", -1)))].append(i)
    for i, rec in enumerate(test_records):
        test_buckets[bucket_name(int(rec.meta_row.get("hour", -1)))].append(i)

    cfgs = {}
    actions = [0 for _ in test_records]
    for bname, vidx in val_buckets.items():
        tidx = test_buckets[bname]
        if len(vidx) < int(args.min_bucket_val_records) or not tidx:
            cfg = all_cfg
        else:
            cfg, _ = select_gate_config(
                subset_records(val_records, vidx),
                subset_probs(val_probs, vidx),
                val_eval_args,
                args,
            )
        cfgs[bname] = cfg
        local_actions = actions_for_config(
            subset_records(test_records, tidx),
            subset_probs(test_probs, tidx),
            cfg,
        )
        for pos, action in zip(tidx, local_actions, strict=True):
            actions[pos] = int(action)

    variant_dir = run_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "bucket_configs.json").write_text(json.dumps(cfgs, indent=2), encoding="utf-8")
    with (variant_dir / "models.pkl").open("wb") as f:
        pickle.dump(models, f)
    report = {
        "variant": name,
        "bucket_configs": cfgs,
        "test": evaluate_actions(test_records, actions, test_eval_args, policy=name, out_csv=variant_dir / "test_episode_metrics.csv"),
    }
    (variant_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_metric_line(name, report["test"])
    progress.add(1.0)
    print(progress.line(f"after {name}"), flush=True)
    return report


def train_route_bad_classifier(records: list[RCOGRecord], runtime, args: argparse.Namespace):
    xs = []
    ys = []
    for rec in records:
        action = int(np.clip(rec.raw_action, 0, len(rec.cand.candidates) - 1))
        xs.append(route_candidate_features(rec.cand, rec.meta_row, action, runtime))
        ys.append(int(rec.bad_override_label))
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.int32)
    return fit_hgb_classifier(
        x,
        y,
        seed=int(args.seed) + 4455,
        max_iter=int(args.model_iter),
        positive_weight_mult=float(args.safety_bad_positive_weight_mult),
    )


def route_bad_probs(model, records: list[RCOGRecord], runtime) -> np.ndarray:
    xs = []
    for rec in records:
        action = int(np.clip(rec.raw_action, 0, len(rec.cand.candidates) - 1))
        xs.append(route_candidate_features(rec.cand, rec.meta_row, action, runtime))
    x = np.stack(xs).astype(np.float32)
    return model.predict_proba(x)[:, 1].astype(np.float32)


def run_safety_filter_variant(
    *,
    train_records: list[RCOGRecord],
    val_records: list[RCOGRecord],
    test_records: list[RCOGRecord],
    val_eval_args: argparse.Namespace,
    test_eval_args: argparse.Namespace,
    runtime,
    args: argparse.Namespace,
    run_dir: Path,
    progress: Progress,
) -> dict:
    name = "ddqn_supervised_safety_filter"
    cprint(BLUE, f"[VARIANT] {name} start")
    val_probs, test_probs, models = fit_probs(
        train_records=train_records,
        val_records=val_records,
        test_records=test_records,
        args=args,
        name=name,
        ensemble=1,
    )
    cfg, _ = select_gate_config(val_records, val_probs, val_eval_args, args)
    val_actions = actions_for_config(val_records, val_probs, cfg)
    model = train_route_bad_classifier(train_records, runtime, args)
    val_risk = route_bad_probs(model, val_records, runtime)
    thresholds = sorted(set([0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50] + [float(np.quantile(val_risk, q)) for q in np.linspace(0.3, 0.95, 14)]))
    candidates = []
    for th in thresholds:
        filtered = [0 if (a != 0 and val_risk[i] > th) else int(a) for i, a in enumerate(val_actions)]
        stats = evaluate_actions(val_records, filtered, val_eval_args, policy=f"{name}_val")
        if stats["bad_gt_5pct"] <= float(args.max_bad_gt_5pct):
            candidates.append((th, stats))
    pool = candidates or [(1.0, evaluate_actions(val_records, val_actions, val_eval_args, policy=f"{name}_val"))]
    pool.sort(key=lambda x: (-float(x[1]["benefit_recall"]), -float(x[1]["benefit_precision"]), float(x[1]["bad_gt_5pct"])))
    best_th = float(pool[0][0])
    test_actions = actions_for_config(test_records, test_probs, cfg)
    test_risk = route_bad_probs(model, test_records, runtime)
    filtered_test = [0 if (a != 0 and test_risk[i] > best_th) else int(a) for i, a in enumerate(test_actions)]
    variant_dir = run_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    with (variant_dir / "route_bad_model.pkl").open("wb") as f:
        pickle.dump(model, f)
    report = {
        "variant": name,
        "selected_risk_threshold": best_th,
        "selected_gate_config": cfg,
        "test": evaluate_actions(test_records, filtered_test, test_eval_args, policy=name, out_csv=variant_dir / "test_episode_metrics.csv"),
    }
    (variant_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_metric_line(name, report["test"])
    progress.add(1.0)
    print(progress.line(f"after {name}"), flush=True)
    return report


def print_metric_line(name: str, stats: dict) -> None:
    bad_color = GREEN if float(stats["bad_gt_5pct"]) <= 0.4 else RED
    ratio_color = GREEN if float(stats["ddqn_over_pred"]) < 1.0 else YELLOW
    print(
        f"{GREEN}[RESULT]{RESET} {BOLD}{name}{RESET} "
        f"act={stats['activation_rate']:.1f}% "
        f"oppP={stats['opportunity_precision']:.1f}% oppR={stats['opportunity_recall']:.1f}% "
        f"benP={stats['benefit_precision']:.1f}% benR={stats['benefit_recall']:.1f}% "
        f"bad1={stats['bad_gt_1pct']:.1f}% {bad_color}bad5={stats['bad_gt_5pct']:.1f}%{RESET} "
        f"{ratio_color}ratio={stats['ddqn_over_pred']:.6f}{RESET}",
        flush=True,
    )


def make_summary_table(reports: list[dict]) -> list[dict]:
    rows = []
    for report in reports:
        stats = report["test"]
        rows.append(
            {
                "variant": report["variant"],
                "activation": float(stats["activation_rate"]),
                "opportunity_precision": float(stats["opportunity_precision"]),
                "opportunity_recall": float(stats["opportunity_recall"]),
                "benefit_precision": float(stats["benefit_precision"]),
                "benefit_recall": float(stats["benefit_recall"]),
                "bad_gt_1pct": float(stats["bad_gt_1pct"]),
                "bad_gt_5pct": float(stats["bad_gt_5pct"]),
                "mean_ratio": float(stats["ddqn_over_pred"]),
            }
        )
    rows.sort(key=lambda r: (r["mean_ratio"], -r["benefit_recall"], -r["benefit_precision"]))
    return rows


def run(args: argparse.Namespace) -> None:
    train_config = load_train_config(args.train_config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))
    runtime = build_runtime(args, train_config, device)
    total_variants = 7.0 + (1.0 if int(args.ddqn_episodes) > 0 else 0.0)
    total_units = float(args.train_pool_size + 2 * args.eval_pool_size) / max(float(args.candidate_unit), 1.0)
    total_units += total_variants
    if int(args.ddqn_episodes) > 0:
        total_units += float(args.ddqn_episodes) / max(float(args.ddqn_episode_unit), 1.0)
    progress = Progress(total_units)

    print(f"{BOLD}{CYAN}{'=' * 104}{RESET}", flush=True)
    cprint(CYAN, "RCOG v3 suggestion suite: DDQN bad-risk, ensemble gate, uncertainty features, dynamic thresholds, safety filter")
    cprint(CYAN, f"run_dir={run_dir} device={device} train_pool={args.train_pool_size} eval_pool={args.eval_pool_size}")
    print(f"{BOLD}{CYAN}{'=' * 104}{RESET}", flush=True)

    args.k_routes = int(value(args, train_config, "k_routes") or 6)
    args.min_unique_routes = int(value(args, train_config, "min_unique_routes") or 3)
    args.min_pred_hops = int(value(args, train_config, "min_pred_hops") or 2)
    args.min_pred_distance_km = float(value(args, train_config, "min_pred_distance_km") or 5.0)

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
    progress.add(float(args.train_pool_size) / max(float(args.candidate_unit), 1.0))
    print(progress.line("after train pool"), flush=True)

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
    progress.add(float(args.eval_pool_size) / max(float(args.candidate_unit), 1.0))
    print(progress.line("after val pool"), flush=True)

    test_pool, test_eval_args = build_pool_for_split(
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

    cprint(YELLOW, "[BUILD] loading current best DDQN and building base RCOG records")
    best_agent = build_agent(args, train_config, train_eval_args, runtime)
    base_train_records_all = build_records(
        train_pool,
        runtime=runtime,
        eval_args=train_eval_args,
        agent=best_agent,
        benefit_delta=float(args.benefit_delta),
        bad_delta=float(args.bad_delta),
    )
    small_n = min(int(args.small_train_size), len(base_train_records_all))
    base_train_records = base_train_records_all[:small_n]
    big_train_records = base_train_records_all
    val_records = build_records(
        val_pool,
        runtime=runtime,
        eval_args=val_eval_args,
        agent=best_agent,
        benefit_delta=float(args.benefit_delta),
        bad_delta=float(args.bad_delta),
    )
    test_records = build_records(
        test_pool,
        runtime=runtime,
        eval_args=test_eval_args,
        agent=best_agent,
        benefit_delta=float(args.benefit_delta),
        bad_delta=float(args.bad_delta),
    )

    reports: list[dict] = []
    reports.append(
        run_gate_variant(
            name=f"baseline_small_train{small_n}",
            train_records=base_train_records,
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
        run_gate_variant(
            name=f"larger_gate_train{len(big_train_records)}",
            train_records=big_train_records,
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
        run_gate_variant(
            name="ensemble_gate_3",
            train_records=big_train_records,
            val_records=val_records,
            test_records=test_records,
            val_eval_args=val_eval_args,
            test_eval_args=test_eval_args,
            args=args,
            run_dir=run_dir,
            progress=progress,
            ensemble=3,
        )
    )

    cprint(YELLOW, "[FEATURE] building uncertainty/perturbation features")
    split_records = {"train": big_train_records, "val": val_records, "test": test_records}
    pert = add_perturb_features(split_records, runtime, sims=int(args.perturb_sims), sigma=float(args.perturb_sigma))
    reports.append(
        run_gate_variant(
            name="uncertainty_perturb_gate",
            train_records=clone_records_with_features(big_train_records, pert["train"]),
            val_records=clone_records_with_features(val_records, pert["val"]),
            test_records=clone_records_with_features(test_records, pert["test"]),
            val_eval_args=val_eval_args,
            test_eval_args=test_eval_args,
            args=args,
            run_dir=run_dir,
            progress=progress,
            ensemble=1,
        )
    )

    cprint(YELLOW, "[FEATURE] building historical OD/time success features")
    hist = add_history_features(big_train_records, split_records, alpha=float(args.history_alpha))
    reports.append(
        run_gate_variant(
            name="history_bucket_features",
            train_records=clone_records_with_features(big_train_records, hist["train"]),
            val_records=clone_records_with_features(val_records, hist["val"]),
            test_records=clone_records_with_features(test_records, hist["test"]),
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
            train_records=big_train_records,
            val_records=val_records,
            test_records=test_records,
            val_eval_args=val_eval_args,
            test_eval_args=test_eval_args,
            args=args,
            run_dir=run_dir,
            progress=progress,
        )
    )

    reports.append(
        run_safety_filter_variant(
            train_records=big_train_records,
            val_records=val_records,
            test_records=test_records,
            val_eval_args=val_eval_args,
            test_eval_args=test_eval_args,
            runtime=runtime,
            args=args,
            run_dir=run_dir,
            progress=progress,
        )
    )

    if int(args.ddqn_episodes) > 0:
        bad_agent = train_bad_risk_ddqn(
            train_pool=train_pool,
            runtime=runtime,
            train_config=train_config,
            eval_args=train_eval_args,
            args=args,
            run_dir=run_dir,
            progress=progress,
        )
        bad_train_records = build_records(
            train_pool,
            runtime=runtime,
            eval_args=train_eval_args,
            agent=bad_agent,
            benefit_delta=float(args.benefit_delta),
            bad_delta=float(args.bad_delta),
        )
        bad_val_records = build_records(
            val_pool,
            runtime=runtime,
            eval_args=val_eval_args,
            agent=bad_agent,
            benefit_delta=float(args.benefit_delta),
            bad_delta=float(args.bad_delta),
        )
        bad_test_records = build_records(
            test_pool,
            runtime=runtime,
            eval_args=test_eval_args,
            agent=bad_agent,
            benefit_delta=float(args.benefit_delta),
            bad_delta=float(args.bad_delta),
        )
        reports.append(
            run_gate_variant(
                name="bad_reward_ddqn_rcog",
                train_records=bad_train_records,
                val_records=bad_val_records,
                test_records=bad_test_records,
                val_eval_args=val_eval_args,
                test_eval_args=test_eval_args,
                args=args,
                run_dir=run_dir,
                progress=progress,
                ensemble=1,
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
        "small_train_size": int(small_n),
        "eval_pool_size": int(args.eval_pool_size),
        "reports": reports,
        "ranked_summary": make_summary_table(reports),
    }
    out_path = run_dir / "rcog_v3_suggestion_suite_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{BOLD}{GREEN}{'=' * 104}{RESET}", flush=True)
    cprint(GREEN, f"Saved summary: {out_path}")
    for row in summary["ranked_summary"]:
        print(
            f"{WHITE}{row['variant']:<32}{RESET} ratio={row['mean_ratio']:.6f} "
            f"benP={row['benefit_precision']:.1f} benR={row['benefit_recall']:.1f} "
            f"bad5={row['bad_gt_5pct']:.1f} act={row['activation']:.1f}",
            flush=True,
        )
    print(progress.line("all done"), flush=True)
    print(f"{BOLD}{GREEN}{'=' * 104}{RESET}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RCOG v3 suggestion suite with colorful progress and ETA.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--train-pool-size", type=int, default=2000)
    p.add_argument("--small-train-size", type=int, default=1000)
    p.add_argument("--eval-pool-size", type=int, default=500)
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
    p.add_argument("--perturb-sims", type=int, default=12)
    p.add_argument("--perturb-sigma", type=float, default=0.08)
    p.add_argument("--history-alpha", type=float, default=20.0)
    p.add_argument("--min-bucket-val-records", type=int, default=80)
    p.add_argument("--safety-bad-positive-weight-mult", type=float, default=5.0)
    p.add_argument("--ddqn-episodes", type=int, default=15000)
    p.add_argument("--ddqn-log-interval", type=int, default=1500)
    p.add_argument("--ddqn-episode-unit", type=float, default=2500.0)
    p.add_argument("--ddqn-epsilon-start", type=float, default=1.0)
    p.add_argument("--ddqn-epsilon-end", type=float, default=0.05)
    p.add_argument("--ddqn-epsilon-decay", type=int, default=6000)
    p.add_argument("--ddqn-buffer-capacity", type=int, default=100000)
    p.add_argument("--ddqn-batch-size", type=int, default=64)
    p.add_argument("--ddqn-target-update", type=int, default=200)
    p.add_argument("--ddqn-bad-loss-weight", type=float, default=8.0)
    p.add_argument("--ddqn-bad-gt5-penalty", type=float, default=4.0)
    p.add_argument("--ddqn-no-gain-penalty", type=float, default=0.5)
    p.add_argument("--candidate-unit", type=float, default=200.0)
    p.add_argument("--candidate-build-batch-size", type=int, default=16)
    p.add_argument("--candidate-build-log-interval", type=int, default=100)
    p.add_argument("--candidate-pool-attempt-multiplier", type=int, default=1600)
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
