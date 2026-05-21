from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from candidate_route_reranker_experiment import (
    RouteChoiceCandidate,
    build_state_and_mask,
)
from evaluate_candidate_route_reranker import load_train_config
from gated_candidate_route_reranker_experiment import build_pool_for_split, build_runtime, eval_args_from_config, masked_q_values
from no_topk_rcog_gate_experiment import (
    RCOGRecord,
    actions_for_config,
    build_agent,
    build_records,
    evaluate_actions,
    fit_hgb_classifier,
    make_probs,
    q_summary,
    rcog_feature,
    select_gate_config,
)
from real_data_uncertain_routing_experiment import configure_runtime, format_seconds, resolve_device


def clone_records_with_features(records: list[RCOGRecord], features: np.ndarray) -> list[RCOGRecord]:
    return [replace(rec, feature=features[idx].astype(np.float32)) for idx, rec in enumerate(records)]


def record_matrix(records: list[RCOGRecord]) -> np.ndarray:
    return np.stack([r.feature for r in records]).astype(np.float32)


def fit_regressor(x: np.ndarray, y: np.ndarray, *, seed: int, max_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=int(max_iter),
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=int(seed),
    ).fit(x, y)


def regret_targets(records: list[RCOGRecord]) -> dict[str, np.ndarray]:
    baseline = np.asarray([r.cand.dijkstra_pred.real_result.travel_time for r in records], dtype=np.float64)
    raw = np.asarray([r.cand.candidates[int(r.raw_action)].real_result.travel_time for r in records], dtype=np.float64)
    rel_regret = (raw - baseline) / np.maximum(baseline, 1e-9)
    return {
        "rel_regret": rel_regret.astype(np.float32),
        "gain_mag": np.maximum(-rel_regret, 0.0).astype(np.float32),
        "loss_mag": np.maximum(rel_regret, 0.0).astype(np.float32),
        "large_gain": (rel_regret < -0.01).astype(np.float32),
        "large_loss": (rel_regret > 0.01).astype(np.float32),
    }


def add_regret_features(train_records: list[RCOGRecord], split_records: dict[str, list[RCOGRecord]], *, seed: int, max_iter: int) -> dict[str, np.ndarray]:
    x_train = record_matrix(train_records)
    targets = regret_targets(train_records)
    models = {
        name: fit_regressor(x_train, y, seed=seed + idx * 17, max_iter=max_iter)
        for idx, (name, y) in enumerate(targets.items())
    }
    out: dict[str, np.ndarray] = {}
    for split, records in split_records.items():
        x = record_matrix(records)
        preds = []
        for name in ["rel_regret", "gain_mag", "loss_mag", "large_gain", "large_loss"]:
            p = models[name].predict(x).astype(np.float32)
            preds.append(p)
        p_regret, p_gain, p_loss, p_large_gain, p_large_loss = preds
        extra = np.stack(
            [
                p_regret,
                p_gain,
                p_loss,
                p_gain - 2.0 * p_loss,
                p_large_gain,
                p_large_loss,
                p_gain / np.maximum(p_loss + 1e-4, 1e-4),
                (p_regret < 0.0).astype(np.float32),
            ],
            axis=1,
        )
        out[split] = np.concatenate([x, np.nan_to_num(extra, nan=0.0, posinf=1e6, neginf=-1e6)], axis=1).astype(np.float32)
    return out


class HistoryStats:
    def __init__(self, records: list[RCOGRecord], *, alpha: float = 20.0):
        self.alpha = float(alpha)
        benefit = np.asarray([r.benefit_label for r in records], dtype=np.float64)
        bad = np.asarray([r.bad_override_label for r in records], dtype=np.float64)
        self.global_benefit = float(np.mean(benefit)) if benefit.size else 0.0
        self.global_bad = float(np.mean(bad)) if bad.size else 0.0
        self.tables: dict[str, dict[tuple, list[float]]] = {name: {} for name in ["od", "oh", "dh", "hour", "origin", "dest"]}
        for rec in records:
            for name, key in self.keys(rec).items():
                row = self.tables[name].setdefault(key, [0.0, 0.0, 0.0])
                row[0] += 1.0
                row[1] += float(rec.benefit_label)
                row[2] += float(rec.bad_override_label)

    @staticmethod
    def keys(rec: RCOGRecord) -> dict[str, tuple]:
        hour = int(rec.meta_row.get("hour", -1))
        return {
            "od": (int(rec.cand.origin), int(rec.cand.dest)),
            "oh": (int(rec.cand.origin), hour),
            "dh": (int(rec.cand.dest), hour),
            "hour": (hour,),
            "origin": (int(rec.cand.origin),),
            "dest": (int(rec.cand.dest),),
        }

    def lookup(self, table: str, key: tuple) -> tuple[float, float, float]:
        n, b, d = self.tables[table].get(key, [0.0, 0.0, 0.0])
        benefit_rate = (b + self.alpha * self.global_benefit) / max(n + self.alpha, 1e-9)
        bad_rate = (d + self.alpha * self.global_bad) / max(n + self.alpha, 1e-9)
        return float(benefit_rate), float(bad_rate), float(math.log1p(n))

    def features(self, records: list[RCOGRecord]) -> np.ndarray:
        rows = []
        for rec in records:
            vals = []
            for name, key in self.keys(rec).items():
                vals.extend(self.lookup(name, key))
            rows.append(vals)
        return np.asarray(rows, dtype=np.float32)


def add_history_features(train_records: list[RCOGRecord], split_records: dict[str, list[RCOGRecord]], *, alpha: float) -> dict[str, np.ndarray]:
    hist = HistoryStats(train_records, alpha=alpha)
    out = {}
    for split, records in split_records.items():
        out[split] = np.concatenate([record_matrix(records), hist.features(records)], axis=1).astype(np.float32)
    return out


def route_time_with_edge_speeds(route, speed_map: dict[int, float], pred_profile: np.ndarray, edge_lengths: np.ndarray) -> float:
    total = 0.0
    for eid in route.edge_ids:
        speed = speed_map.get(int(eid), float(pred_profile[int(eid), 0]))
        total += float(edge_lengths[int(eid)]) / max(float(speed), 1e-5)
    return total


def perturb_features_for_record(rec: RCOGRecord, edge_speeds: np.ndarray, edge_lengths: np.ndarray, *, sims: int, sigma: float) -> np.ndarray:
    raw_action = int(np.clip(rec.raw_action, 0, len(rec.cand.candidates) - 1))
    if raw_action == 0:
        return np.zeros(10, dtype=np.float32)
    d0 = rec.cand.dijkstra_pred
    raw = rec.cand.candidates[raw_action]
    union_edges = sorted(edge_set(d0) | edge_set(raw))
    if not union_edges:
        return np.zeros(10, dtype=np.float32)
    seed = (int(rec.cand.window_idx) * 73856093 + int(rec.cand.origin) * 19349663 + int(rec.cand.dest) * 83492791) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    hist_t = max(0, int(rec.cand.target_t) - 1)
    base_speeds = {eid: float(max(rec.cand.pred_profile[eid, 0], 1e-5)) for eid in union_edges}
    hist_speeds = np.asarray(edge_speeds[max(0, hist_t - 5) : hist_t + 1, union_edges], dtype=np.float64)
    hist_cv = np.std(hist_speeds, axis=0) / np.maximum(np.abs(np.mean(hist_speeds, axis=0)), 1e-5)
    hist_cv = np.clip(hist_cv, 0.0, 1.0)
    rel_regrets = []
    for _ in range(int(sims)):
        noise = rng.normal(0.0, float(sigma), size=len(union_edges)) * (1.0 + hist_cv)
        mult = np.clip(1.0 + noise, 0.55, 1.60)
        speed_map = {eid: base_speeds[eid] * float(mult[idx]) for idx, eid in enumerate(union_edges)}
        d_time = route_time_with_edge_speeds(d0, speed_map, rec.cand.pred_profile, edge_lengths)
        r_time = route_time_with_edge_speeds(raw, speed_map, rec.cand.pred_profile, edge_lengths)
        rel_regrets.append((r_time - d_time) / max(d_time, 1e-9))
    arr = np.asarray(rel_regrets, dtype=np.float64)
    return np.asarray(
        [
            float(np.mean(arr < 0.0)),
            float(np.mean(arr < -0.005)),
            float(np.mean(arr > 0.005)),
            float(np.mean(arr > 0.01)),
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.quantile(arr, 0.10)),
            float(np.quantile(arr, 0.90)),
            float(np.min(arr)),
            float(np.max(arr)),
        ],
        dtype=np.float32,
    )


def add_perturb_features(split_records: dict[str, list[RCOGRecord]], runtime, *, sims: int, sigma: float) -> dict[str, np.ndarray]:
    out = {}
    for split, records in split_records.items():
        start = time.time()
        extras = np.stack(
            [perturb_features_for_record(r, runtime.edge_speeds, runtime.edge_lengths, sims=sims, sigma=sigma) for r in records]
        ).astype(np.float32)
        print(f"Built perturb features for {split}: n={len(records)} elapsed={format_seconds(time.time() - start)}", flush=True)
        out[split] = np.concatenate([record_matrix(records), extras], axis=1).astype(np.float32)
    return out


def edge_set(route) -> set[int]:
    return set(int(x) for x in route.edge_ids)


def route_candidate_features(cand: RouteChoiceCandidate, meta_row, action_idx: int, runtime) -> np.ndarray:
    route = cand.candidates[int(action_idx)]
    d0 = cand.dijkstra_pred
    d0_edges = edge_set(d0)
    r_edges = edge_set(route)
    inter = len(d0_edges & r_edges)
    union = max(len(d0_edges | r_edges), 1)
    hour = float(meta_row.get("hour", 0.0))
    hour_angle = 2.0 * math.pi * hour / 24.0
    t0 = max(float(d0.pred_result.travel_time), 1e-9)
    return np.asarray(
        [
            float(action_idx),
            float(action_idx != 0),
            float(cand.origin),
            float(cand.dest),
            math.sin(hour_angle),
            math.cos(hour_angle),
            math.log1p(max(float(cand.dispatch_count), 0.0)),
            float(route.pred_result.travel_time / t0),
            float(route.pred_result.travel_dist / max(float(d0.pred_result.travel_dist), 1e-9)),
            float(route.pred_result.steps / max(float(d0.pred_result.steps), 1.0)),
            float(route.min_pred_speed) / 130.0,
            float(route.mean_pred_speed) / 130.0,
            float(route.pred_time_std),
            float(route.pred_speed_std) / 130.0,
            float(route.low_speed_frac),
            float(route.pred_time_trend_ratio),
            float(inter / union),
            float(1.0 - inter / union),
            float(len(r_edges)),
        ],
        dtype=np.float32,
    )


def train_supervised_route_regressor(pool, runtime, eval_args, *, seed: int, max_iter: int):
    xs = []
    ys = []
    for cand in pool:
        meta_row = runtime.time_meta.iloc[cand.target_t]
        baseline = max(float(cand.dijkstra_pred.real_result.travel_time), 1e-9)
        for action_idx, route in enumerate(cand.candidates):
            xs.append(route_candidate_features(cand, meta_row, action_idx, runtime))
            ys.append(float(route.real_result.travel_time) / baseline)
    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    return fit_regressor(x, y, seed=seed, max_iter=max_iter)


def build_supervised_records(pool, runtime, eval_args, scorer, *, benefit_delta: float, bad_delta: float) -> list[RCOGRecord]:
    records = []
    for cand in pool:
        meta_row = runtime.time_meta.iloc[cand.target_t]
        state, mask = build_state_and_mask(cand, meta_row=meta_row, args=eval_args)
        feats = np.stack([route_candidate_features(cand, meta_row, i, runtime) for i in range(len(cand.candidates))]).astype(np.float32)
        pred_ratio = scorer.predict(feats).astype(np.float32)
        q_values = -pred_ratio
        q_values = np.where(mask > 0, q_values, -np.inf)
        raw_action = int(np.argmax(q_values))
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


def fit_gate_models(train_records: list[RCOGRecord], *, seed: int, max_iter: int, benefit_negative_mult: float, bad_positive_mult: float) -> dict:
    x_train = record_matrix(train_records)
    labels = {
        "opportunity": np.asarray([r.opportunity_label for r in train_records], dtype=np.int32),
        "benefit": np.asarray([r.benefit_label for r in train_records], dtype=np.int32),
        "bad": np.asarray([r.bad_override_label for r in train_records], dtype=np.int32),
    }
    print(
        "Gate label rates: " + ", ".join(f"{k}={float(v.mean())*100:.2f}%" for k, v in labels.items()),
        flush=True,
    )
    return {
        "opportunity": fit_hgb_classifier(x_train, labels["opportunity"], seed=seed, max_iter=max_iter),
        "benefit": fit_hgb_classifier(
            x_train,
            labels["benefit"],
            seed=seed + 17,
            max_iter=max_iter,
            negative_weight_mult=float(benefit_negative_mult),
        ),
        "bad": fit_hgb_classifier(
            x_train,
            labels["bad"],
            seed=seed + 34,
            max_iter=max_iter,
            positive_weight_mult=float(bad_positive_mult),
        ),
    }


def run_variant(name: str, train_records: list[RCOGRecord], val_records: list[RCOGRecord], test_records: list[RCOGRecord], val_eval_args, test_eval_args, args, run_dir: Path) -> dict:
    print("=" * 80, flush=True)
    print(f"Variant: {name}", flush=True)
    start = time.time()
    models = fit_gate_models(
        train_records,
        seed=stable_variant_seed(int(args.seed), name),
        max_iter=int(args.model_iter),
        benefit_negative_mult=float(args.benefit_negative_weight_mult),
        bad_positive_mult=float(args.bad_positive_weight_mult),
    )
    val_probs = make_probs(models, val_records)
    best_cfg, sweep = select_gate_config(val_records, val_probs, val_eval_args, args)
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
        "elapsed": format_seconds(time.time() - start),
        "val": evaluate_actions(val_records, val_actions, val_eval_args, policy=name, out_csv=variant_dir / "val_episode_metrics.csv"),
        "test": evaluate_actions(test_records, test_actions, test_eval_args, policy=name, out_csv=variant_dir / "test_episode_metrics.csv"),
    }
    (variant_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{name} test:", json.dumps(report["test"], indent=2), flush=True)
    return report


def stable_variant_seed(base_seed: int, name: str) -> int:
    offset = sum((idx + 1) * ord(ch) for idx, ch in enumerate(name)) % 10000
    return int(base_seed) + offset


def print_total_eta(label: str, start_time: float, completed_units: float, total_units: float) -> None:
    elapsed = time.time() - start_time
    progress = completed_units / max(total_units, 1e-9)
    eta = elapsed * (1.0 - progress) / max(progress, 1e-9) if progress > 0 else float("nan")
    print(
        f"[TOTAL] {label} progress={completed_units:.1f}/{total_units:.1f} "
        f"elapsed={format_seconds(elapsed)} total_ETA={format_seconds(eta) if math.isfinite(eta) else 'unknown'}",
        flush=True,
    )


def run(args: argparse.Namespace) -> None:
    train_config = load_train_config(args.train_config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))
    runtime = build_runtime(args, train_config, device)
    print("=" * 96, flush=True)
    print("Enhanced RCOG ablation experiment", flush=True)
    print(f"run_dir={run_dir} device={device}", flush=True)
    print("=" * 96, flush=True)
    total_start = time.time()
    total_candidate_units = float(args.train_pool_size + 2 * args.eval_pool_size)
    rough_total_sec = total_candidate_units * float(args.rough_seconds_per_candidate)
    variant_names = ", ".join(args.variants)
    print(
        f"[TOTAL] variants={variant_names} rough_total_ETA={format_seconds(rough_total_sec)} "
        f"(candidate-pool dominated estimate; stage logs update as work completes)",
        flush=True,
    )

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
    train_eval_args.feature_set = "base"
    agent = build_agent(args, train_config, train_eval_args, runtime)
    train_records = build_records(
        train_pool,
        runtime=runtime,
        eval_args=train_eval_args,
        agent=agent,
        benefit_delta=float(args.benefit_delta),
        bad_delta=float(args.bad_delta),
    )

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
    val_eval_args.feature_set = "base"
    val_records = build_records(
        val_pool,
        runtime=runtime,
        eval_args=val_eval_args,
        agent=agent,
        benefit_delta=float(args.benefit_delta),
        bad_delta=float(args.bad_delta),
    )

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
    test_eval_args.feature_set = "base"
    test_records = build_records(
        test_pool,
        runtime=runtime,
        eval_args=test_eval_args,
        agent=agent,
        benefit_delta=float(args.benefit_delta),
        bad_delta=float(args.bad_delta),
    )

    split_records = {"train": train_records, "val": val_records, "test": test_records}
    variants: dict[str, tuple[list[RCOGRecord], list[RCOGRecord], list[RCOGRecord]]] = {
        "base_reproduced": (train_records, val_records, test_records)
    }

    if "regret" in args.variants or "all3" in args.variants:
        stage_start = time.time()
        print("[STAGE] 1/4 regret magnitude features start", flush=True)
        feats = add_regret_features(train_records, split_records, seed=int(args.seed) + 3100, max_iter=int(args.regret_model_iter))
        variants["regret_magnitude"] = (
            clone_records_with_features(train_records, feats["train"]),
            clone_records_with_features(val_records, feats["val"]),
            clone_records_with_features(test_records, feats["test"]),
        )
        print(f"[STAGE] regret magnitude features done elapsed={format_seconds(time.time() - stage_start)}", flush=True)

    if "history" in args.variants or "all3" in args.variants:
        stage_start = time.time()
        print("[STAGE] 2/4 historical OD/time success-rate features start", flush=True)
        feats = add_history_features(train_records, split_records, alpha=float(args.history_alpha))
        variants["history_success"] = (
            clone_records_with_features(train_records, feats["train"]),
            clone_records_with_features(val_records, feats["val"]),
            clone_records_with_features(test_records, feats["test"]),
        )
        print(f"[STAGE] history success-rate features done elapsed={format_seconds(time.time() - stage_start)}", flush=True)

    if "perturb" in args.variants or "all3" in args.variants:
        stage_start = time.time()
        print("[STAGE] 3/4 local perturbation sensitivity features start", flush=True)
        feats = add_perturb_features(split_records, runtime, sims=int(args.perturb_sims), sigma=float(args.perturb_sigma))
        variants["perturb_sensitivity"] = (
            clone_records_with_features(train_records, feats["train"]),
            clone_records_with_features(val_records, feats["val"]),
            clone_records_with_features(test_records, feats["test"]),
        )
        print(f"[STAGE] perturbation sensitivity features done elapsed={format_seconds(time.time() - stage_start)}", flush=True)

    if "all3" in args.variants:
        x_train = record_matrix(train_records)
        x_val = record_matrix(val_records)
        x_test = record_matrix(test_records)
        reg = add_regret_features(train_records, split_records, seed=int(args.seed) + 4100, max_iter=int(args.regret_model_iter))
        hist = add_history_features(train_records, split_records, alpha=float(args.history_alpha))
        pert = add_perturb_features(split_records, runtime, sims=int(args.perturb_sims), sigma=float(args.perturb_sigma))
        variants["regret_history_perturb"] = (
            clone_records_with_features(train_records, np.concatenate([x_train, reg["train"][:, x_train.shape[1]:], hist["train"][:, x_train.shape[1]:], pert["train"][:, x_train.shape[1]:]], axis=1)),
            clone_records_with_features(val_records, np.concatenate([x_val, reg["val"][:, x_val.shape[1]:], hist["val"][:, x_val.shape[1]:], pert["val"][:, x_val.shape[1]:]], axis=1)),
            clone_records_with_features(test_records, np.concatenate([x_test, reg["test"][:, x_test.shape[1]:], hist["test"][:, x_test.shape[1]:], pert["test"][:, x_test.shape[1]:]], axis=1)),
        )

    if "supervised" in args.variants:
        stage_start = time.time()
        print("[STAGE] 4/4 supervised route regressor start", flush=True)
        scorer = train_supervised_route_regressor(train_pool, runtime, train_eval_args, seed=int(args.seed) + 5100, max_iter=int(args.supervised_model_iter))
        sup_train = build_supervised_records(train_pool, runtime, train_eval_args, scorer, benefit_delta=float(args.benefit_delta), bad_delta=float(args.bad_delta))
        sup_val = build_supervised_records(val_pool, runtime, val_eval_args, scorer, benefit_delta=float(args.benefit_delta), bad_delta=float(args.bad_delta))
        sup_test = build_supervised_records(test_pool, runtime, test_eval_args, scorer, benefit_delta=float(args.benefit_delta), bad_delta=float(args.bad_delta))
        variants["supervised_reranker"] = (sup_train, sup_val, sup_test)
        with (run_dir / "supervised_route_regressor.pkl").open("wb") as f:
            pickle.dump(scorer, f)
        print(f"[STAGE] supervised route regressor done elapsed={format_seconds(time.time() - stage_start)}", flush=True)

    reports = []
    selected_variants = [v for v in args.variants if v != "all3"]
    run_names = ["base_reproduced"]
    if "regret" in selected_variants:
        run_names.append("regret_magnitude")
    if "history" in selected_variants:
        run_names.append("history_success")
    if "perturb" in selected_variants:
        run_names.append("perturb_sensitivity")
    if "all3" in args.variants:
        run_names.append("regret_magnitude")
        run_names.append("history_success")
        run_names.append("perturb_sensitivity")
        run_names.append("regret_history_perturb")
    if "supervised" in selected_variants:
        run_names.append("supervised_reranker")
    seen = set()
    variant_total = len([n for n in run_names if n in variants])
    variant_done = 0
    for name in run_names:
        if name in seen or name not in variants:
            continue
        seen.add(name)
        variant_done += 1
        print(
            f"[VARIANT] {variant_done}/{variant_total} {name} training/eval start "
            f"elapsed_total={format_seconds(time.time() - total_start)}",
            flush=True,
        )
        tr, va, te = variants[name]
        reports.append(run_variant(name, tr, va, te, val_eval_args, test_eval_args, args, run_dir))
        print(
            f"[VARIANT] {variant_done}/{variant_total} {name} done "
            f"elapsed_total={format_seconds(time.time() - total_start)}",
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
        "reports": reports,
    }
    (run_dir / "enhanced_ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {run_dir / 'enhanced_ablation_summary.json'}", flush=True)
    print(f"[TOTAL] all done elapsed={format_seconds(time.time() - total_start)} total_ETA=00:00", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enhanced RCOG ablations: regret, history, perturbation, supervised reranker.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--variants", nargs="+", default=["regret"], choices=["regret", "history", "perturb", "supervised", "all3"])
    p.add_argument("--train-pool-size", type=int, default=10000)
    p.add_argument("--eval-pool-size", type=int, default=1000)
    p.add_argument("--model-iter", type=int, default=160)
    p.add_argument("--regret-model-iter", type=int, default=160)
    p.add_argument("--supervised-model-iter", type=int, default=220)
    p.add_argument("--selection-objective", type=str, default="constrained_balanced", choices=["time", "benefit_recall", "opportunity_recall", "opportunity_precision_at_recall", "opportunity_balanced", "constrained_balanced", "precision_at_recall", "two_stage_precision_at_recall"])
    p.add_argument("--threshold-search-grid", type=str, default="dense", choices=["coarse", "dense"])
    p.add_argument("--save-sweeps", action="store_true")
    p.add_argument("--benefit-negative-weight-mult", type=float, default=1.0)
    p.add_argument("--bad-positive-weight-mult", type=float, default=1.0)
    p.add_argument("--history-alpha", type=float, default=20.0)
    p.add_argument("--perturb-sims", type=int, default=16)
    p.add_argument("--perturb-sigma", type=float, default=0.08)
    p.add_argument("--rough-seconds-per-candidate", type=float, default=0.50)
    p.add_argument("--benefit-delta", type=float, default=0.005)
    p.add_argument("--bad-delta", type=float, default=0.05)
    p.add_argument("--max-bad-gt-1pct", type=float, default=3.0)
    p.add_argument("--max-bad-gt-5pct", type=float, default=0.5)
    p.add_argument("--min-activation", type=float, default=8.0)
    p.add_argument("--max-activation", type=float, default=22.0)
    p.add_argument("--min-benefit-precision", type=float, default=55.0)
    p.add_argument("--min-benefit-recall", type=float, default=70.0)
    p.add_argument("--min-opportunity-precision", type=float, default=0.0)
    p.add_argument("--min-opportunity-recall", type=float, default=0.0)
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
