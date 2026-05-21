from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from data_loader import load_nyc_graph_for_rl
from ddqn_agent import DoubleDQNAgent
from graph_env import CityGraph, RoutingEnv, create_graph_from_data, infer_max_neighbors


@dataclass
class RouteResult:
    reached: bool
    travel_time: float
    travel_dist: float
    reward: float
    steps: int
    path: list[int]
    last_loss: float | None = None


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 CUDA，但目前環境沒有可用 GPU。")
    return device


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def configure_runtime(device: torch.device, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")


def build_uncertain_real_speeds(
    pred_speed_profile: np.ndarray,
    *,
    seed: int,
    noise_low: float,
    noise_high: float,
    incident_prob: float,
    incident_low: float,
    incident_high: float,
    min_speed: float,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    pred = np.maximum(np.asarray(pred_speed_profile, dtype=np.float32), min_speed)
    real = pred * rng.uniform(noise_low, noise_high, size=pred.shape).astype(np.float32)
    if incident_prob > 0:
        incident_mask = rng.random_sample(pred.shape) < float(incident_prob)
        if incident_mask.any():
            real[incident_mask] *= rng.uniform(
                incident_low,
                incident_high,
                size=int(incident_mask.sum()),
            ).astype(np.float32)
    return np.maximum(real.astype(np.float32), min_speed)


def sample_od(num_nodes: int, rng: np.random.RandomState) -> tuple[int, int]:
    while True:
        origin = int(rng.randint(0, num_nodes))
        dest = int(rng.randint(0, num_nodes))
        if origin != dest:
            return origin, dest


def edge_lookup(edge_index: np.ndarray) -> dict[tuple[int, int], int]:
    return {
        (int(src), int(dst)): int(i)
        for i, (src, dst) in enumerate(np.asarray(edge_index, dtype=np.int32))
    }


def simulate_path_under_real_speeds(
    path: list[int],
    *,
    edge_id: dict[tuple[int, int], int],
    edge_lengths: np.ndarray,
    real_speed_profile: np.ndarray,
    start_slot: int,
    time_slot_duration_hours: float,
    num_time_slots: int,
    alpha: float,
    beta: float,
    rho: float,
) -> RouteResult:
    if len(path) < 2:
        return RouteResult(False, float("inf"), float("inf"), -20.0, 0, path)

    total_time = 0.0
    total_dist = 0.0
    reward = 0.0
    reached = True
    for step, (src, dst) in enumerate(zip(path[:-1], path[1:]), start=1):
        idx = edge_id.get((int(src), int(dst)))
        if idx is None:
            return RouteResult(False, float("inf"), float("inf"), -20.0, step - 1, path[:step])
        slot = int(
            np.clip(
                int(start_slot + total_time / max(time_slot_duration_hours, 1e-6)),
                0,
                num_time_slots - 1,
            )
        )
        length = float(edge_lengths[idx])
        speed = float(max(real_speed_profile[idx, slot], 1e-5))
        travel_time = length / speed
        total_time += travel_time
        total_dist += length
        reward += -alpha * travel_time - beta * length

    reward += rho
    return RouteResult(reached, total_time, total_dist, reward, len(path) - 1, list(path))


def run_ddqn_episode(
    env: RoutingEnv,
    agent: DoubleDQNAgent,
    origin: int,
    dest: int,
    time_slot: int,
    *,
    train: bool,
) -> RouteResult:
    state, mask, _ = env.reset(origin, dest, time_slot)
    done = False
    total_reward = 0.0
    path = [origin]
    final_info: dict | None = None
    last_loss: float | None = None

    while not done:
        action = agent.select_action(state, mask, greedy=not train)
        result = env.step(action)
        next_state = result.state
        next_mask = result.action_mask
        if train:
            agent.store_transition(
                state,
                action,
                result.reward,
                next_state,
                result.done,
                mask,
                next_mask,
            )
            maybe_loss = agent.learn()
            if maybe_loss is not None:
                last_loss = float(maybe_loss)
        total_reward += result.reward
        state = next_state
        mask = next_mask
        done = result.done
        final_info = result.info
        path.append(env.current_node)

    assert final_info is not None
    return RouteResult(
        bool(final_info["reached_goal"]),
        float(final_info["total_travel_time"]),
        float(final_info["total_distance"]),
        float(total_reward),
        int(final_info["steps"]),
        path,
        last_loss,
    )


def dijkstra_result(
    graph: CityGraph,
    origin: int,
    dest: int,
    *,
    use_real_speed: bool,
    edge_id: dict[tuple[int, int], int],
    edge_lengths: np.ndarray,
    real_speed_profile: np.ndarray,
    start_slot: int,
    time_slot_duration_hours: float,
    num_time_slots: int,
    alpha: float,
    beta: float,
    rho: float,
) -> RouteResult:
    path, _, _ = graph.dijkstra(origin, dest, use_real_speed=use_real_speed)
    if not path:
        return RouteResult(False, float("inf"), float("inf"), -20.0, 0, [])
    return simulate_path_under_real_speeds(
        path,
        edge_id=edge_id,
        edge_lengths=edge_lengths,
        real_speed_profile=real_speed_profile,
        start_slot=start_slot,
        time_slot_duration_hours=time_slot_duration_hours,
        num_time_slots=num_time_slots,
        alpha=alpha,
        beta=beta,
        rho=rho,
    )


def mean_finite(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def summarize_window(rows: list[dict]) -> dict:
    if not rows:
        return {}
    ddqn_success = np.asarray([r["ddqn_reached"] for r in rows], dtype=np.float32)
    fallback_used = np.asarray([r["fallback_used"] for r in rows], dtype=np.float32)
    pred_times = [float(r["dijkstra_pred_time"]) for r in rows]
    real_times = [float(r["dijkstra_real_time"]) for r in rows]
    ddqn_times = [float(r["ddqn_time"]) for r in rows if r["ddqn_reached"]]
    hybrid_times = [float(r["hybrid_time"]) for r in rows if r["hybrid_reached"]]
    rewards = [float(r["ddqn_reward"]) for r in rows]
    losses = [float(r["loss"]) for r in rows if r["loss"] is not None and math.isfinite(float(r["loss"]))]
    pred_mean = mean_finite(pred_times)
    real_mean = mean_finite(real_times)
    ddqn_mean = mean_finite(ddqn_times)
    hybrid_mean = mean_finite(hybrid_times)
    return {
        "ddqn_sr": float(ddqn_success.mean() * 100.0),
        "fallback_rate": float(fallback_used.mean() * 100.0),
        "avg_reward": float(np.mean(rewards)),
        "avg_loss": float(np.mean(losses)) if losses else float("nan"),
        "ddqn_time": ddqn_mean,
        "hybrid_time": hybrid_mean,
        "dijkstra_pred_time": pred_mean,
        "dijkstra_real_time": real_mean,
        "ddqn_over_pred": float(ddqn_mean / pred_mean) if np.isfinite(ddqn_mean) and pred_mean > 0 else float("nan"),
        "hybrid_over_pred": float(hybrid_mean / pred_mean) if np.isfinite(hybrid_mean) and pred_mean > 0 else float("nan"),
        "pred_over_real": float(pred_mean / real_mean) if np.isfinite(pred_mean) and real_mean > 0 else float("nan"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    configure_runtime(device, args.seed)
    rng = np.random.RandomState(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    gdata = load_nyc_graph_for_rl(
        args.data_dir,
        edge_length_source=args.edge_length_source,
        speed_seed=args.seed,
        routing_locationid_max=args.routing_locationid_max,
        routing_graph_mode=args.routing_graph_mode,
        superzone_dir=args.superzone_dir or None,
    )
    edge_index = gdata["edge_index"]
    num_nodes = int(gdata["adj"].shape[0])
    max_nb = infer_max_neighbors(edge_index, num_nodes, minimum=args.max_neighbors)
    pred_speed_profile = np.asarray(gdata["pred_speed_profile"], dtype=np.float32)
    if args.max_speed_slots > 0:
        pred_speed_profile = pred_speed_profile[:, : args.max_speed_slots]
    num_time_slots = int(pred_speed_profile.shape[1])
    if num_time_slots <= 0:
        raise ValueError("No speed slots available for dynamic routing experiment.")

    real_speed_profile = build_uncertain_real_speeds(
        pred_speed_profile,
        seed=args.uncertainty_seed,
        noise_low=args.noise_low,
        noise_high=args.noise_high,
        incident_prob=args.incident_prob,
        incident_low=args.incident_low,
        incident_high=args.incident_high,
        min_speed=args.min_speed,
    )
    time_slot_duration_hours = float(gdata["time_slot_minutes"]) / 60.0

    graph = create_graph_from_data(
        num_nodes,
        edge_index,
        gdata["edge_lengths"],
        pred_speed_profile[:, 0],
        real_speed_profile[:, 0],
        max_neighbors=max_nb,
    )
    env = RoutingEnv(
        graph,
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
        rho=args.rho,
        max_steps=args.max_steps,
        num_time_slots=num_time_slots,
        time_slot_duration_hours=time_slot_duration_hours,
        use_real_speed=True,
        dynamic_edge_index=edge_index,
        dynamic_pred_speeds=pred_speed_profile,
        dynamic_real_speeds=real_speed_profile,
    )
    agent = DoubleDQNAgent(
        num_nodes=graph.num_nodes,
        max_neighbors=graph.max_neighbors,
        state_dim=env.state_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update=args.target_update,
        device=str(device),
    )

    edge_id = edge_lookup(edge_index)
    rows: list[dict] = []
    summary_rows: list[dict] = []
    started = time.time()

    config = vars(args).copy()
    config.update(
        {
            "num_nodes": num_nodes,
            "num_edges": int(edge_index.shape[0]),
            "max_neighbors": int(graph.max_neighbors),
            "state_dim": int(env.state_dim),
            "speed_slots_used": num_time_slots,
            "routing_graph_mode": gdata.get("routing_graph_mode", "legacy"),
            "superzone_dir": gdata.get("superzone_dir", ""),
            "device_resolved": str(device),
        }
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 88, flush=True)
    print("Dynamic uncertain routing DDQN experiment", flush=True)
    print(
        f"nodes={num_nodes} edges={edge_index.shape[0]} max_neighbors={graph.max_neighbors} "
        f"state_dim={env.state_dim} slots={num_time_slots} device={device}",
        flush=True,
    )
    print(
        "uncertainty="
        f"uniform[{args.noise_low:.2f},{args.noise_high:.2f}] "
        f"incident_prob={args.incident_prob:.3f} incident_factor="
        f"[{args.incident_low:.2f},{args.incident_high:.2f}]",
        flush=True,
    )
    print(f"run_dir={run_dir}", flush=True)
    print("=" * 88, flush=True)

    for ep in range(1, args.episodes + 1):
        origin, dest = sample_od(num_nodes, rng)
        time_slot = int(rng.randint(0, num_time_slots))
        env.reset(origin, dest, time_slot)
        pred_result = dijkstra_result(
            graph,
            origin,
            dest,
            use_real_speed=False,
            edge_id=edge_id,
            edge_lengths=gdata["edge_lengths"],
            real_speed_profile=real_speed_profile,
            start_slot=time_slot,
            time_slot_duration_hours=time_slot_duration_hours,
            num_time_slots=num_time_slots,
            alpha=args.alpha,
            beta=args.beta,
            rho=args.rho,
        )
        env.reset(origin, dest, time_slot)
        real_result = dijkstra_result(
            graph,
            origin,
            dest,
            use_real_speed=True,
            edge_id=edge_id,
            edge_lengths=gdata["edge_lengths"],
            real_speed_profile=real_speed_profile,
            start_slot=time_slot,
            time_slot_duration_hours=time_slot_duration_hours,
            num_time_slots=num_time_slots,
            alpha=args.alpha,
            beta=args.beta,
            rho=args.rho,
        )
        ddqn_result = run_ddqn_episode(
            env,
            agent,
            origin,
            dest,
            time_slot,
            train=True,
        )

        fallback_used = (
            (not ddqn_result.reached)
            or (
                pred_result.reached
                and ddqn_result.reached
                and ddqn_result.travel_time > args.fallback_ratio * pred_result.travel_time
            )
        )
        hybrid = pred_result if fallback_used else ddqn_result
        row = {
            "episode": ep,
            "origin": origin,
            "dest": dest,
            "time_slot": time_slot,
            "ddqn_reached": int(ddqn_result.reached),
            "ddqn_time": ddqn_result.travel_time,
            "ddqn_dist": ddqn_result.travel_dist,
            "ddqn_reward": ddqn_result.reward,
            "ddqn_steps": ddqn_result.steps,
            "dijkstra_pred_reached": int(pred_result.reached),
            "dijkstra_pred_time": pred_result.travel_time,
            "dijkstra_pred_dist": pred_result.travel_dist,
            "dijkstra_real_reached": int(real_result.reached),
            "dijkstra_real_time": real_result.travel_time,
            "dijkstra_real_dist": real_result.travel_dist,
            "fallback_used": int(fallback_used),
            "hybrid_reached": int(hybrid.reached),
            "hybrid_time": hybrid.travel_time,
            "hybrid_dist": hybrid.travel_dist,
            "epsilon": agent.epsilon,
            "loss": ddqn_result.last_loss,
        }
        rows.append(row)

        if ep % args.log_interval == 0 or ep == args.episodes:
            window = rows[-args.log_interval :]
            stats = summarize_window(window)
            elapsed = time.time() - started
            progress = ep / max(args.episodes, 1)
            eta = elapsed * (1.0 - progress) / max(progress, 1e-9)
            summary = {
                "episode": ep,
                "progress": progress,
                "elapsed_sec": elapsed,
                "eta_sec": eta,
                "epsilon": agent.epsilon,
                **stats,
            }
            summary_rows.append(summary)
            print(
                f"[{ep:>5d}/{args.episodes:<5d} {progress*100:>6.2f}%] "
                f"ETA={format_seconds(eta)} elapsed={format_seconds(elapsed)} "
                f"eps={agent.epsilon:.3f} "
                f"SR={stats['ddqn_sr']:>5.1f}% "
                f"DDQN={stats['ddqn_time']:.4f}h "
                f"DijPred={stats['dijkstra_pred_time']:.4f}h "
                f"DijReal={stats['dijkstra_real_time']:.4f}h "
                f"DDQN/Pred={stats['ddqn_over_pred']:.3f} "
                f"Hybrid/Pred={stats['hybrid_over_pred']:.3f} "
                f"fallback={stats['fallback_rate']:.1f}% "
                f"R={stats['avg_reward']:.2f} loss={stats['avg_loss']:.4f}",
                flush=True,
            )
            write_csv(run_dir / "episode_metrics.csv", rows)
            write_csv(run_dir / "window_summary.csv", summary_rows)
            (run_dir / "summary.json").write_text(
                json.dumps(summary_rows[-1], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            agent.save(run_dir / "ddqn_latest.pt")

    agent.save(run_dir / "ddqn_final.pt")
    final_stats = summarize_window(rows)
    final_payload = {
        "config": config,
        "final_all_episode_metrics": final_stats,
        "last_window": summary_rows[-1] if summary_rows else {},
        "outputs": {
            "checkpoint": str(run_dir / "ddqn_final.pt"),
            "episode_metrics": str(run_dir / "episode_metrics.csv"),
            "window_summary": str(run_dir / "window_summary.csv"),
        },
    }
    (run_dir / "final_report.json").write_text(
        json.dumps(final_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("=" * 88, flush=True)
    print("Training finished", flush=True)
    print(json.dumps(final_payload["final_all_episode_metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"Saved: {run_dir / 'ddqn_final.pt'}", flush=True)
    print("=" * 88, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDQN dynamic uncertain routing experiment")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--edge-length-source", type=str, default="osrm", choices=["osrm", "centroid"])
    p.add_argument("--routing-locationid-max", type=int, default=63)
    p.add_argument("--routing-graph-mode", type=str, default="superzone", choices=["legacy", "superzone"])
    p.add_argument("--superzone-dir", type=str, default="")
    p.add_argument("--run-dir", type=str, default="runs/dynamic_uncertain_routing")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--max-speed-slots", type=int, default=288)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-neighbors", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--uncertainty-seed", type=int, default=123)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--log-interval", type=int, default=25)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=2500)
    p.add_argument("--buffer-capacity", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--target-update", type=int, default=100)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--rho", type=float, default=50.0)
    p.add_argument("--noise-low", type=float, default=0.75)
    p.add_argument("--noise-high", type=float, default=1.25)
    p.add_argument("--incident-prob", type=float, default=0.03)
    p.add_argument("--incident-low", type=float, default=0.15)
    p.add_argument("--incident-high", type=float, default=0.55)
    p.add_argument("--min-speed", type=float, default=1.0)
    p.add_argument("--fallback-ratio", type=float, default=1.2)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
