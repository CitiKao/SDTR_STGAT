from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from candidate_route_reranker_experiment import RouteChoiceCandidate, build_state_and_mask
from evaluate_candidate_route_reranker import load_train_config
from gated_candidate_route_reranker_experiment import build_pool_for_split, build_runtime
from no_topk_rcog_gate_experiment import (
    RCOGRecord,
    actions_for_config,
    evaluate_actions,
    fit_hgb_classifier,
    make_probs,
    q_summary,
    rcog_feature,
    select_gate_config,
)
from rcog_enhanced_ablation_experiment import fit_gate_models, route_candidate_features, stable_variant_seed
from real_data_uncertain_routing_experiment import configure_runtime, format_seconds, resolve_device


@dataclass
class RouteScorer:
    kind: str
    ratio: HistGradientBoostingRegressor | None = None
    benefit: HistGradientBoostingClassifier | None = None
    bad: HistGradientBoostingClassifier | None = None
    gain: HistGradientBoostingRegressor | None = None
    loss: HistGradientBoostingRegressor | None = None
    benefit_weight: float = 1.0
    gain_weight: float = 2.0
    bad_weight: float = 2.5
    loss_weight: float = 1.0
    ratio_weight: float = 0.1

    def q_values(self, x: np.ndarray, *, k: int) -> np.ndarray:
        q = np.full(k, -np.inf, dtype=np.float32)
        if x.size == 0:
            return q
        n = min(int(x.shape[0]), k)
        if self.kind == "ratio":
            pred_ratio = np.asarray(self.ratio.predict(x[:n]), dtype=np.float32)
            q[:n] = -pred_ratio
            return q

        p_benefit = np.asarray(self.benefit.predict_proba(x[:n])[:, 1], dtype=np.float32)
        p_bad = np.asarray(self.bad.predict_proba(x[:n])[:, 1], dtype=np.float32)
        gain = np.maximum(np.asarray(self.gain.predict(x[:n]), dtype=np.float32), 0.0)
        loss = np.maximum(np.asarray(self.loss.predict(x[:n]), dtype=np.float32), 0.0)
        ratio = np.asarray(self.ratio.predict(x[:n]), dtype=np.float32) if self.ratio is not None else np.ones(n, dtype=np.float32)
        score = (
            self.benefit_weight * p_benefit
            + self.gain_weight * gain
            - self.bad_weight * p_bad
            - self.loss_weight * loss
            - self.ratio_weight * np.maximum(ratio - 1.0, 0.0)
        )
        q[:n] = score.astype(np.float32)
        if n > 0:
            q[0] = 0.0
        return q


def fit_regressor(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    sample_weight: np.ndarray | None = None,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=int(max_iter),
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=int(seed),
    ).fit(x, y, sample_weight=sample_weight)


def fit_classifier_weighted(
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    max_iter: int,
    sample_weight: np.ndarray,
) -> HistGradientBoostingClassifier:
    clf = HistGradientBoostingClassifier(
        max_iter=int(max_iter),
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.01,
        random_state=int(seed),
    )
    clf.fit(x, y, sample_weight=sample_weight)
    return clf


def train_route_scorer(
    pool: list[RouteChoiceCandidate],
    runtime,
    *,
    kind: str,
    seed: int,
    max_iter: int,
    benefit_delta: float,
    bad_delta: float,
    benefit_positive_weight_mult: float,
    bad_positive_weight_mult: float,
    direct_gain_weight: float,
    direct_bad_weight: float,
    direct_loss_weight: float,
    direct_ratio_weight: float,
    hard_negative_mult: float,
    route_bad_train_delta: float,
) -> RouteScorer:
    xs: list[np.ndarray] = []
    ratio_y: list[float] = []
    benefit_y: list[int] = []
    bad_y: list[int] = []
    gain_y: list[float] = []
    loss_y: list[float] = []
    is_alt: list[int] = []
    route_ratio: list[float] = []
    for cand in pool:
        meta_row = runtime.time_meta.iloc[cand.target_t]
        baseline = max(float(cand.dijkstra_pred.real_result.travel_time), 1e-9)
        for action_idx, route in enumerate(cand.candidates):
            ratio = float(route.real_result.travel_time) / baseline
            xs.append(route_candidate_features(cand, meta_row, action_idx, runtime))
            ratio_y.append(ratio)
            benefit_y.append(int(action_idx != 0 and ratio < 1.0 - float(benefit_delta)))
            bad_y.append(int(action_idx != 0 and ratio > 1.0 + float(bad_delta)))
            gain_y.append(max(1.0 - ratio, 0.0))
            loss_y.append(max(ratio - 1.0, 0.0))
            is_alt.append(int(action_idx != 0))
            route_ratio.append(ratio)

    x = np.stack(xs).astype(np.float32)
    ratio_arr = np.asarray(ratio_y, dtype=np.float32)
    is_alt_arr = np.asarray(is_alt, dtype=bool)
    route_ratio_arr = np.asarray(route_ratio, dtype=np.float32)
    hard_negative = is_alt_arr & (route_ratio_arr >= 1.0 - float(benefit_delta))
    near_or_bad = is_alt_arr & (route_ratio_arr > 1.0 + float(route_bad_train_delta))
    reg_weight = np.ones_like(ratio_arr, dtype=np.float64)
    reg_weight[hard_negative] += float(hard_negative_mult)
    reg_weight[near_or_bad] += float(hard_negative_mult)
    scorer = RouteScorer(
        kind=kind,
        ratio=fit_regressor(x, ratio_arr, seed=seed, max_iter=max_iter, sample_weight=reg_weight),
        gain_weight=float(direct_gain_weight),
        bad_weight=float(direct_bad_weight),
        loss_weight=float(direct_loss_weight),
        ratio_weight=float(direct_ratio_weight),
    )
    if kind == "ratio":
        return scorer

    benefit_arr = np.asarray(benefit_y, dtype=np.int32)
    bad_arr = np.asarray(bad_y, dtype=np.int32)
    route_risk_arr = (near_or_bad | (bad_arr == 1)).astype(np.int32)

    pos = float(np.sum(benefit_arr == 1))
    neg = float(np.sum(benefit_arr == 0))
    benefit_weight = np.ones_like(benefit_arr, dtype=np.float64)
    if pos > 0 and neg > 0:
        benefit_weight[benefit_arr == 1] = min(neg / max(pos, 1.0), 50.0) * float(benefit_positive_weight_mult)
    benefit_weight[hard_negative & (benefit_arr == 0)] *= float(hard_negative_mult)
    scorer.benefit = fit_classifier_weighted(
        x,
        benefit_arr,
        seed=seed + 11,
        max_iter=max_iter,
        sample_weight=benefit_weight,
    )

    pos = float(np.sum(route_risk_arr == 1))
    neg = float(np.sum(route_risk_arr == 0))
    bad_weight = np.ones_like(route_risk_arr, dtype=np.float64)
    if pos > 0 and neg > 0:
        bad_weight[route_risk_arr == 1] = min(neg / max(pos, 1.0), 50.0) * float(bad_positive_weight_mult)
    bad_weight[bad_arr == 1] *= float(hard_negative_mult)
    scorer.bad = fit_classifier_weighted(
        x,
        route_risk_arr,
        seed=seed + 23,
        max_iter=max_iter,
        sample_weight=bad_weight,
    )
    gain_weight = np.ones_like(ratio_arr, dtype=np.float64)
    gain_weight[benefit_arr == 1] += float(benefit_positive_weight_mult)
    loss_weight = np.ones_like(ratio_arr, dtype=np.float64)
    loss_weight[hard_negative] += float(hard_negative_mult)
    loss_weight[near_or_bad] += float(hard_negative_mult)
    loss_weight[bad_arr == 1] += float(hard_negative_mult)
    scorer.gain = fit_regressor(
        x,
        np.asarray(gain_y, dtype=np.float32),
        seed=seed + 31,
        max_iter=max_iter,
        sample_weight=gain_weight,
    )
    scorer.loss = fit_regressor(
        x,
        np.asarray(loss_y, dtype=np.float32),
        seed=seed + 43,
        max_iter=max_iter,
        sample_weight=loss_weight,
    )
    return scorer


def slice_pool(pool: list[RouteChoiceCandidate], k: int) -> list[RouteChoiceCandidate]:
    out: list[RouteChoiceCandidate] = []
    for cand in pool:
        candidates = cand.candidates[: min(int(k), len(cand.candidates))]
        if len(candidates) < 2:
            continue
        oracle = min(candidates, key=lambda r: r.real_result.travel_time)
        out.append(replace(cand, candidates=candidates, candidate_oracle=oracle))
    return out


def clone_eval_args(eval_args: argparse.Namespace, *, k: int) -> argparse.Namespace:
    out = argparse.Namespace(**vars(eval_args))
    out.k_routes = int(k)
    out.feature_set = "base"
    return out


def build_scorer_records(
    pool: list[RouteChoiceCandidate],
    runtime,
    eval_args: argparse.Namespace,
    scorer: RouteScorer,
    *,
    benefit_delta: float,
    bad_delta: float,
) -> list[RCOGRecord]:
    records: list[RCOGRecord] = []
    k = int(eval_args.k_routes)
    for cand in pool:
        meta_row = runtime.time_meta.iloc[cand.target_t]
        state, mask = build_state_and_mask(cand, meta_row=meta_row, args=eval_args)
        feats = np.stack([route_candidate_features(cand, meta_row, i, runtime) for i in range(len(cand.candidates))]).astype(np.float32)
        q_values = scorer.q_values(feats, k=k)
        q_values = np.where(mask > 0, q_values, -np.inf)
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


def run_variant(
    *,
    name: str,
    train_records: list[RCOGRecord],
    val_records: list[RCOGRecord],
    test_records: list[RCOGRecord],
    val_eval_args: argparse.Namespace,
    test_eval_args: argparse.Namespace,
    args: argparse.Namespace,
    run_dir: Path,
    gate_objective: str,
    min_opportunity_precision: float,
    min_opportunity_recall: float,
    max_activation: float,
    max_bad_gt_1pct: float | None = None,
    max_bad_gt_5pct: float | None = None,
    min_benefit_precision: float | None = None,
    benefit_negative_weight_mult: float | None = None,
    bad_positive_weight_mult: float | None = None,
) -> dict:
    print("=" * 88, flush=True)
    print(f"[VARIANT] {name} gate={gate_objective}", flush=True)
    start = time.time()
    gate_args = argparse.Namespace(**vars(args))
    gate_args.selection_objective = gate_objective
    gate_args.min_opportunity_precision = float(min_opportunity_precision)
    gate_args.min_opportunity_recall = float(min_opportunity_recall)
    gate_args.max_activation = float(max_activation)
    if max_bad_gt_1pct is not None:
        gate_args.max_bad_gt_1pct = float(max_bad_gt_1pct)
    if max_bad_gt_5pct is not None:
        gate_args.max_bad_gt_5pct = float(max_bad_gt_5pct)
    if min_benefit_precision is not None:
        gate_args.min_benefit_precision = float(min_benefit_precision)
    if benefit_negative_weight_mult is not None:
        gate_args.benefit_negative_weight_mult = float(benefit_negative_weight_mult)
    if bad_positive_weight_mult is not None:
        gate_args.bad_positive_weight_mult = float(bad_positive_weight_mult)
    models = fit_gate_models(
        train_records,
        seed=stable_variant_seed(int(args.seed), name),
        max_iter=int(args.model_iter),
        benefit_negative_mult=float(gate_args.benefit_negative_weight_mult),
        bad_positive_mult=float(gate_args.bad_positive_weight_mult),
    )
    val_probs = make_probs(models, val_records)
    best_cfg, sweep = select_gate_config(val_records, val_probs, val_eval_args, gate_args)
    variant_dir = run_dir / name
    variant_dir.mkdir(parents=True, exist_ok=True)
    (variant_dir / "selected_config.json").write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")
    if bool(args.save_sweeps):
        (variant_dir / "val_sweep.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
    print("Selected config:", json.dumps(best_cfg, indent=2), flush=True)
    val_actions = actions_for_config(val_records, val_probs, best_cfg)
    test_probs = make_probs(models, test_records)
    test_actions = actions_for_config(test_records, test_probs, best_cfg)
    report = {
        "variant": name,
        "gate_objective": gate_objective,
        "elapsed": format_seconds(time.time() - start),
        "raw_action_val": evaluate_actions(val_records, [r.raw_action for r in val_records], val_eval_args, policy=f"{name}_raw"),
        "raw_action_test": evaluate_actions(test_records, [r.raw_action for r in test_records], test_eval_args, policy=f"{name}_raw"),
        "val": evaluate_actions(val_records, val_actions, val_eval_args, policy=name, out_csv=variant_dir / "val_episode_metrics.csv"),
        "test": evaluate_actions(test_records, test_actions, test_eval_args, policy=name, out_csv=variant_dir / "test_episode_metrics.csv"),
    }
    (variant_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (variant_dir / "gate_models.pkl").open("wb") as f:
        pickle.dump(models, f)
    print(f"{name} test:", json.dumps(report["test"], indent=2), flush=True)
    return report


def print_total_eta(label: str, start_time: float, completed_units: float, total_units: float) -> None:
    elapsed = time.time() - start_time
    progress = completed_units / max(total_units, 1e-9)
    eta = elapsed * (1.0 - progress) / max(progress, 1e-9) if progress > 0 else float("nan")
    eta_text = format_seconds(eta) if math.isfinite(eta) else "unknown"
    print(
        f"[TOTAL] {label} progress={completed_units:.1f}/{total_units:.1f} "
        f"elapsed={format_seconds(elapsed)} total_ETA={eta_text}",
        flush=True,
    )


def experiment_matrix(args: argparse.Namespace) -> list[dict]:
    base_k = int(args.base_k)
    max_k = int(args.max_k)
    rows = [
        {
            "name": f"ratio_k{base_k}_balanced",
            "kind": "ratio",
            "profile": "ratio",
            "k": base_k,
            "gate_objective": "constrained_balanced",
            "min_opportunity_precision": 0.0,
            "min_opportunity_recall": 0.0,
            "max_activation": float(args.max_activation),
        },
        {
            "name": f"ratio_k{max_k}_balanced",
            "kind": "ratio",
            "profile": "ratio",
            "k": max_k,
            "gate_objective": "constrained_balanced",
            "min_opportunity_precision": 0.0,
            "min_opportunity_recall": 0.0,
            "max_activation": float(args.max_activation),
        },
        {
            "name": f"direct_k{max_k}_balanced",
            "kind": "direct",
            "profile": "direct_default",
            "k": max_k,
            "gate_objective": "constrained_balanced",
            "min_opportunity_precision": 0.0,
            "min_opportunity_recall": 0.0,
            "max_activation": float(args.max_activation),
        },
        {
            "name": f"direct_k{max_k}_recall_gate",
            "kind": "direct",
            "profile": "direct_default",
            "k": max_k,
            "gate_objective": "opportunity_balanced",
            "min_opportunity_precision": float(args.recall_gate_min_opportunity_precision),
            "min_opportunity_recall": float(args.recall_gate_min_opportunity_recall),
            "max_activation": float(args.recall_gate_max_activation),
        },
    ]
    if bool(args.include_ratio_recall_gate):
        rows.append(
            {
                "name": f"ratio_k{max_k}_recall_gate",
                "kind": "ratio",
                "profile": "ratio",
                "k": max_k,
                "gate_objective": "opportunity_balanced",
                "min_opportunity_precision": float(args.recall_gate_min_opportunity_precision),
                "min_opportunity_recall": float(args.recall_gate_min_opportunity_recall),
                "max_activation": float(args.recall_gate_max_activation),
            }
        )
    if bool(args.include_hard_negative_variants):
        rows.extend(
            [
                {
                    "name": f"direct_k{max_k}_badpenalty",
                    "kind": "direct",
                    "profile": "badpenalty",
                    "k": max_k,
                    "gate_objective": "constrained_balanced",
                    "min_opportunity_precision": 0.0,
                    "min_opportunity_recall": 0.0,
                    "max_activation": min(float(args.max_activation), 20.0),
                    "max_bad_gt_1pct": 2.5,
                    "max_bad_gt_5pct": 0.3,
                    "min_benefit_precision": 55.0,
                    "bad_positive_weight_mult": 3.0,
                    "benefit_negative_weight_mult": 1.5,
                    "scorer": {
                        "direct_bad_weight": 5.5,
                        "direct_loss_weight": 2.5,
                        "direct_ratio_weight": 0.5,
                        "hard_negative_mult": 4.0,
                        "route_bad_train_delta": 0.01,
                        "direct_bad_positive_weight_mult": 4.0,
                        "direct_benefit_positive_weight_mult": 1.0,
                    },
                },
                {
                    "name": f"direct_k{max_k}_hardneg",
                    "kind": "direct",
                    "profile": "hardneg",
                    "k": max_k,
                    "gate_objective": "constrained_balanced",
                    "min_opportunity_precision": 0.0,
                    "min_opportunity_recall": 0.0,
                    "max_activation": min(float(args.max_activation), 18.0),
                    "max_bad_gt_1pct": 2.0,
                    "max_bad_gt_5pct": 0.25,
                    "min_benefit_precision": 55.0,
                    "bad_positive_weight_mult": 5.0,
                    "benefit_negative_weight_mult": 2.0,
                    "scorer": {
                        "direct_bad_weight": 8.0,
                        "direct_loss_weight": 4.0,
                        "direct_ratio_weight": 0.8,
                        "hard_negative_mult": 7.0,
                        "route_bad_train_delta": 0.0,
                        "direct_bad_positive_weight_mult": 6.0,
                        "direct_benefit_positive_weight_mult": 1.0,
                    },
                },
                {
                    "name": f"direct_k{max_k}_hardneg_recall_gate",
                    "kind": "direct",
                    "profile": "hardneg",
                    "k": max_k,
                    "gate_objective": "opportunity_balanced",
                    "min_opportunity_precision": max(float(args.recall_gate_min_opportunity_precision), 55.0),
                    "min_opportunity_recall": min(float(args.recall_gate_min_opportunity_recall), 50.0),
                    "max_activation": min(float(args.recall_gate_max_activation), 22.0),
                    "max_bad_gt_1pct": 2.5,
                    "max_bad_gt_5pct": 0.3,
                    "min_benefit_precision": 55.0,
                    "bad_positive_weight_mult": 5.0,
                    "benefit_negative_weight_mult": 2.0,
                    "scorer": {
                        "direct_bad_weight": 8.0,
                        "direct_loss_weight": 4.0,
                        "direct_ratio_weight": 0.8,
                        "hard_negative_mult": 7.0,
                        "route_bad_train_delta": 0.0,
                        "direct_bad_positive_weight_mult": 6.0,
                        "direct_benefit_positive_weight_mult": 1.0,
                    },
                },
            ]
        )
    return rows


def run(args: argparse.Namespace) -> None:
    train_config = load_train_config(args.train_config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))
    runtime = build_runtime(args, train_config, device)

    total_start = time.time()
    max_k = int(args.max_k)
    args.k_routes = max_k
    if args.min_unique_routes is not None:
        args.min_unique_routes = min(int(args.min_unique_routes), max_k)
    total_candidate_units = float(args.train_pool_size + 2 * args.eval_pool_size)
    print("=" * 96, flush=True)
    print("Supervised reranker v2 experiment", flush=True)
    print(f"run_dir={run_dir} device={device} base_k={args.base_k} max_k={max_k}", flush=True)
    print(
        "Tests: candidate K quality, direct override target, and recall-aware gate selection.",
        flush=True,
    )
    print("=" * 96, flush=True)

    train_pool, train_eval_args = build_pool_for_split(
        split="train",
        pool_size=int(args.train_pool_size),
        run_dir=run_dir,
        args=args,
        train_config=train_config,
        runtime=runtime,
        seed_offset=0,
    )
    print_total_eta("after train pool", total_start, float(args.train_pool_size), total_candidate_units)
    val_pool, val_eval_args = build_pool_for_split(
        split="val",
        pool_size=int(args.eval_pool_size),
        run_dir=run_dir / "eval_val",
        args=args,
        train_config=train_config,
        runtime=runtime,
        seed_offset=1000,
    )
    print_total_eta("after val pool", total_start, float(args.train_pool_size + args.eval_pool_size), total_candidate_units)
    test_pool, test_eval_args = build_pool_for_split(
        split="test",
        pool_size=int(args.eval_pool_size),
        run_dir=run_dir / "eval_test",
        args=args,
        train_config=train_config,
        runtime=runtime,
        seed_offset=2000,
    )
    print_total_eta("after test pool", total_start, total_candidate_units, total_candidate_units)

    pools_by_k: dict[int, tuple[list[RouteChoiceCandidate], list[RouteChoiceCandidate], list[RouteChoiceCandidate]]] = {}
    eval_args_by_k: dict[int, tuple[argparse.Namespace, argparse.Namespace, argparse.Namespace]] = {}
    for k in sorted({int(args.base_k), int(args.max_k)}):
        pools_by_k[k] = (slice_pool(train_pool, k), slice_pool(val_pool, k), slice_pool(test_pool, k))
        eval_args_by_k[k] = (
            clone_eval_args(train_eval_args, k=k),
            clone_eval_args(val_eval_args, k=k),
            clone_eval_args(test_eval_args, k=k),
        )
        print(
            f"[POOL] k={k} train/val/test={len(pools_by_k[k][0])}/{len(pools_by_k[k][1])}/{len(pools_by_k[k][2])}",
            flush=True,
        )

    scorers: dict[tuple[str, int], RouteScorer] = {}
    records_cache: dict[tuple[str, int], tuple[list[RCOGRecord], list[RCOGRecord], list[RCOGRecord]]] = {}
    reports: list[dict] = []
    matrix = experiment_matrix(args)
    for idx, item in enumerate(matrix, start=1):
        kind = str(item["kind"])
        k = int(item["k"])
        profile = str(item.get("profile", kind))
        scorer_cfg = dict(item.get("scorer", {}))
        key = (kind, profile, k)
        print(
            f"[STAGE] {idx}/{len(matrix)} preparing {item['name']} "
            f"elapsed_total={format_seconds(time.time() - total_start)}",
            flush=True,
        )
        if key not in scorers:
            train_pool_k, _, _ = pools_by_k[k]
            scorers[key] = train_route_scorer(
                train_pool_k,
                runtime,
                kind=kind,
                seed=stable_variant_seed(int(args.seed), f"{profile}_{k}"),
                max_iter=int(args.supervised_model_iter),
                benefit_delta=float(args.benefit_delta),
                bad_delta=float(args.bad_delta),
                benefit_positive_weight_mult=float(scorer_cfg.get("direct_benefit_positive_weight_mult", args.direct_benefit_positive_weight_mult)),
                bad_positive_weight_mult=float(scorer_cfg.get("direct_bad_positive_weight_mult", args.direct_bad_positive_weight_mult)),
                direct_gain_weight=float(scorer_cfg.get("direct_gain_weight", args.direct_gain_weight)),
                direct_bad_weight=float(scorer_cfg.get("direct_bad_weight", args.direct_bad_weight)),
                direct_loss_weight=float(scorer_cfg.get("direct_loss_weight", args.direct_loss_weight)),
                direct_ratio_weight=float(scorer_cfg.get("direct_ratio_weight", args.direct_ratio_weight)),
                hard_negative_mult=float(scorer_cfg.get("hard_negative_mult", args.hard_negative_mult)),
                route_bad_train_delta=float(scorer_cfg.get("route_bad_train_delta", args.route_bad_train_delta)),
            )
            with (run_dir / f"{profile}_k{k}_route_scorer.pkl").open("wb") as f:
                pickle.dump(scorers[key], f)
        if key not in records_cache:
            tr_pool, va_pool, te_pool = pools_by_k[k]
            tr_args, va_args, te_args = eval_args_by_k[k]
            records_cache[key] = (
                build_scorer_records(tr_pool, runtime, tr_args, scorers[key], benefit_delta=float(args.benefit_delta), bad_delta=float(args.bad_delta)),
                build_scorer_records(va_pool, runtime, va_args, scorers[key], benefit_delta=float(args.benefit_delta), bad_delta=float(args.bad_delta)),
                build_scorer_records(te_pool, runtime, te_args, scorers[key], benefit_delta=float(args.benefit_delta), bad_delta=float(args.bad_delta)),
            )
        tr, va, te = records_cache[key]
        _, va_args, te_args = eval_args_by_k[k]
        reports.append(
            run_variant(
                name=str(item["name"]),
                train_records=tr,
                val_records=va,
                test_records=te,
                val_eval_args=va_args,
                test_eval_args=te_args,
                args=args,
                run_dir=run_dir,
                gate_objective=str(item["gate_objective"]),
                min_opportunity_precision=float(item["min_opportunity_precision"]),
                min_opportunity_recall=float(item["min_opportunity_recall"]),
                max_activation=float(item["max_activation"]),
                max_bad_gt_1pct=item.get("max_bad_gt_1pct"),
                max_bad_gt_5pct=item.get("max_bad_gt_5pct"),
                min_benefit_precision=item.get("min_benefit_precision"),
                benefit_negative_weight_mult=item.get("benefit_negative_weight_mult"),
                bad_positive_weight_mult=item.get("bad_positive_weight_mult"),
            )
        )
        print(
            f"[TOTAL] variant {idx}/{len(matrix)} done elapsed={format_seconds(time.time() - total_start)}",
            flush=True,
        )

    summary = {
        "baseline_reference": {
            "run": "runs/rcog_v2_classifier_train10000_final_eval3000",
            "benefit_precision": 61.57760814249363,
            "benefit_recall": 78.06451612903226,
            "bad_gt_5pct": 0.4,
            "ddqn_over_pred": 0.9974396875254729,
        },
        "train_pool_size": int(args.train_pool_size),
        "eval_pool_size": int(args.eval_pool_size),
        "base_k": int(args.base_k),
        "max_k": int(args.max_k),
        "reports": reports,
    }
    (run_dir / "supervised_reranker_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {run_dir / 'supervised_reranker_v2_summary.json'}", flush=True)
    print(f"[TOTAL] all done elapsed={format_seconds(time.time() - total_start)} total_ETA=00:00", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised reranker v2 ablations for recall/precision.")
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--base-k", type=int, default=6)
    p.add_argument("--max-k", type=int, default=12)
    p.add_argument("--train-pool-size", type=int, default=5000)
    p.add_argument("--eval-pool-size", type=int, default=1000)
    p.add_argument("--model-iter", type=int, default=160)
    p.add_argument("--supervised-model-iter", type=int, default=220)
    p.add_argument("--threshold-search-grid", type=str, default="dense", choices=["coarse", "dense"])
    p.add_argument("--save-sweeps", action="store_true")
    p.add_argument("--benefit-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--bad-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--direct-benefit-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--direct-bad-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--direct-gain-weight", type=float, default=2.0)
    p.add_argument("--direct-bad-weight", type=float, default=2.5)
    p.add_argument("--direct-loss-weight", type=float, default=1.0)
    p.add_argument("--direct-ratio-weight", type=float, default=0.1)
    p.add_argument("--hard-negative-mult", type=float, default=0.0)
    p.add_argument("--route-bad-train-delta", type=float, default=0.05)
    p.add_argument("--benefit-delta", type=float, default=0.005)
    p.add_argument("--bad-delta", type=float, default=0.05)
    p.add_argument("--max-bad-gt-1pct", type=float, default=3.0)
    p.add_argument("--max-bad-gt-5pct", type=float, default=0.5)
    p.add_argument("--min-activation", type=float, default=6.0)
    p.add_argument("--max-activation", type=float, default=24.0)
    p.add_argument("--min-benefit-precision", type=float, default=55.0)
    p.add_argument("--min-benefit-recall", type=float, default=0.0)
    p.add_argument("--min-opportunity-precision", type=float, default=0.0)
    p.add_argument("--min-opportunity-recall", type=float, default=0.0)
    p.add_argument("--recall-gate-min-opportunity-precision", type=float, default=55.0)
    p.add_argument("--recall-gate-min-opportunity-recall", type=float, default=55.0)
    p.add_argument("--recall-gate-max-activation", type=float, default=30.0)
    p.add_argument("--include-ratio-recall-gate", action="store_true")
    p.add_argument("--include-hard-negative-variants", action="store_true")
    p.add_argument("--candidate-build-batch-size", type=int, default=16)
    p.add_argument("--candidate-build-log-interval", type=int, default=500)
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
