from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from candidate_route_reranker_experiment import (
    CandidateQNetwork,
    build_state_and_mask,
    choose_reward,
    row_from_choice,
    state_dim_for_args,
    summarize,
    write_csv,
)
from evaluate_candidate_route_reranker import load_train_config
from gated_candidate_route_reranker_experiment import (
    build_pool_for_split,
    build_runtime,
    eval_args_from_config,
)
from real_data_uncertain_routing_experiment import configure_runtime, format_seconds, resolve_device


def oracle_action(cand) -> int:
    for idx, route in enumerate(cand.candidates):
        if route.path == cand.candidate_oracle.path:
            return int(idx)
    return 0


def tensorize_pool(pool, time_meta, eval_args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    for cand in pool:
        state, mask = build_state_and_mask(cand, meta_row=time_meta.iloc[cand.target_t], args=eval_args)
        states.append(state)
        masks.append(mask)
        labels.append(oracle_action(cand))
    return (
        np.stack(states).astype(np.float32),
        np.stack(masks).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def evaluate_model(model, pool, time_meta, eval_args, device, run_dir: Path, split: str) -> dict:
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for idx, cand in enumerate(pool, start=1):
            meta_row = time_meta.iloc[cand.target_t]
            state, mask = build_state_and_mask(cand, meta_row=meta_row, args=eval_args)
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q = model(s).squeeze(0).detach().cpu().numpy()
            q = np.where(mask > 0, q, -np.inf)
            action = int(np.argmax(q))
            action = int(np.clip(action, 0, len(cand.candidates) - 1))
            reward = choose_reward(cand, cand.candidates[action], eval_args)
            rows.append(
                row_from_choice(
                    episode=idx,
                    cand=cand,
                    action=action,
                    reward=reward,
                    loss=None,
                    epsilon=0.0,
                    meta_row=meta_row,
                )
            )
    stats = summarize(rows)
    write_csv(run_dir / f"{split}_supervised_episode_metrics.csv", rows)
    return stats


def run(args: argparse.Namespace) -> None:
    train_config = load_train_config(args.train_config)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    configure_runtime(device, int(args.seed))
    runtime = build_runtime(args, train_config, device)

    print("=" * 96, flush=True)
    print("Supervised candidate-route scorer experiment", flush=True)
    print("Train labels use train-split realized candidate oracle; eval decisions use predicted features only.", flush=True)
    print(f"run_dir={run_dir} device={device}", flush=True)
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
    state_dim = state_dim_for_args(train_eval_args)
    model = CandidateQNetwork(
        num_nodes=runtime.num_nodes,
        action_dim=int(train_eval_args.k_routes),
        state_dim=state_dim,
        embed_dim=int(args.embed_dim or train_config.get("embed_dim", 16)),
        hidden_dim=int(args.hidden_dim or train_config.get("hidden_dim", 128)),
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.CrossEntropyLoss()

    states, masks, labels = tensorize_pool(train_pool, runtime.time_meta, train_eval_args)
    x = torch.tensor(states, dtype=torch.float32, device=device)
    m = torch.tensor(masks, dtype=torch.float32, device=device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    rng = np.random.RandomState(int(args.seed) + 171)
    start = time.time()
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = rng.permutation(len(train_pool))
        losses: list[float] = []
        correct = 0
        total = 0
        for begin in range(0, len(order), int(args.batch_size)):
            idx = order[begin : begin + int(args.batch_size)]
            logits = model(x[idx])
            logits = logits.masked_fill(m[idx] <= 0, -1e9)
            loss = loss_fn(logits, y[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            losses.append(float(loss.item()))
            pred = logits.argmax(dim=1)
            correct += int((pred == y[idx]).sum().item())
            total += int(len(idx))
        avg_loss = float(np.mean(losses)) if losses else float("nan")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % int(args.log_interval) == 0 or epoch == int(args.epochs):
            elapsed = time.time() - start
            progress = epoch / max(int(args.epochs), 1)
            eta = elapsed * (1.0 - progress) / max(progress, 1e-9)
            print(
                f"epoch={epoch}/{args.epochs} loss={avg_loss:.5f} "
                f"train_acc={correct / max(total, 1) * 100:.2f}% "
                f"elapsed={format_seconds(elapsed)} ETA={format_seconds(eta)}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": int(state_dim),
            "action_dim": int(train_eval_args.k_routes),
            "config": vars(args),
        },
        run_dir / "supervised_route_scorer.pt",
    )

    reports = {
        "train_config": str(args.train_config),
        "train_pool_size": int(args.train_pool_size),
        "best_train_loss": float(best_loss),
        "splits": {},
        "leakage_note": "Realized future speeds are used only for train labels and final evaluation metrics.",
    }
    train_stats = evaluate_model(model, train_pool, runtime.time_meta, train_eval_args, device, run_dir, "train")
    reports["splits"]["train"] = train_stats
    print("train supervised eval:", json.dumps(train_stats, indent=2), flush=True)

    for split in args.eval_splits:
        split_dir = run_dir / f"eval_{split}"
        split_dir.mkdir(parents=True, exist_ok=True)
        pool, eval_args = build_pool_for_split(
            split=split,
            pool_size=int(args.eval_pool_size),
            run_dir=split_dir,
            args=args,
            train_config=train_config,
            runtime=runtime,
            seed_offset=1000 + 1000 * args.eval_splits.index(split),
        )
        stats = evaluate_model(model, pool, runtime.time_meta, eval_args, device, split_dir, split)
        reports["splits"][split] = stats
        print(f"{split} supervised eval:", json.dumps(stats, indent=2), flush=True)

    (run_dir / "supervised_route_scorer_summary.json").write_text(
        json.dumps(reports, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {run_dir / 'supervised_route_scorer_summary.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Supervised candidate route scorer using historical oracle labels.")
    p.add_argument("--train-config", type=str, required=True)
    p.add_argument("--run-dir", type=str, required=True)
    p.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--train-pool-size", type=int, default=3000)
    p.add_argument("--eval-pool-size", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
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
    p.add_argument("--feature-set", type=str, default=None, choices=[None, "base", "uncertainty", "pattern_topk"])
    p.add_argument("--reward-mode", type=str, default=None, choices=[None, "shaped", "direct_regret", "time_only"])
    p.add_argument("--pattern-topk", type=int, default=None)
    p.add_argument("--pattern-attention-temperature", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
