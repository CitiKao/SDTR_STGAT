"""
train_ddqn.py — Double DQN 訓練腳本

訓練流程：
  1. 建立 / 載入城市圖
  2. 隨機取樣 (start, destination) 組合
  3. 在 RoutingEnv 中跑 episode
  4. 收集 transition 並定期呼叫 agent.learn()
  5. 記錄 average episode reward / success rate / travel time / distance

使用方式（需 data/ 含 adjacency 與 edge_speeds，與 STGAT 路網一致）：
    python train_ddqn.py
    python train_ddqn.py --device auto
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from data_loader import load_nyc_graph_for_rl

from ddqn_agent import DoubleDQNAgent
from graph_env import CityGraph, RoutingEnv, create_graph_from_data, infer_max_neighbors


# ────────────────────────────────────────────────────────────────
#  OD pair 生成
# ────────────────────────────────────────────────────────────────

def sample_od_pairs(
    num_nodes: int, n: int, rng: np.random.RandomState
) -> List[tuple[int, int]]:
    """隨機產生 n 組 (origin, destination)，保證 origin ≠ destination"""
    pairs = []
    while len(pairs) < n:
        o = rng.randint(0, num_nodes)
        d = rng.randint(0, num_nodes)
        if o != d:
            pairs.append((o, d))
    return pairs


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 CUDA，但目前環境沒有可用 GPU。")
    return device


def configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# ────────────────────────────────────────────────────────────────
#  訓練
# ────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    rng = np.random.RandomState(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ── 建圖（與 STGAT 相同紐約路網；max_neighbors 至少覆蓋最大出度）──
    gdata = load_nyc_graph_for_rl(
        args.data_dir,
        edge_length_source=args.edge_length_source,
        speed_seed=args.seed,
        routing_locationid_max=args.routing_locationid_max,
        routing_graph_mode=args.routing_graph_mode,
        superzone_dir=args.superzone_dir or None,
    )
    adj = gdata["adj"]
    edge_index = gdata["edge_index"]
    n = adj.shape[0]
    max_nb = infer_max_neighbors(edge_index, n, minimum=args.max_neighbors)
    pred_speed_profile = gdata["pred_speed_profile"]
    real_speed_profile = gdata["real_speed_profile"]
    num_time_slots = int(gdata["num_time_slots"])
    time_slot_duration_hours = float(gdata["time_slot_minutes"]) / 60.0

    graph: CityGraph = create_graph_from_data(
        n,
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
        use_real_speed=False,
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

    # ── 預先產生 OD pair（也可每 episode 隨機）──
    od_pairs = sample_od_pairs(graph.num_nodes, args.num_episodes, rng)

    # ── 日誌 ──
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    # ── 滑動窗口指標 ──
    window = args.log_interval
    ep_rewards: List[float] = []
    ep_successes: List[int] = []
    ep_times: List[float] = []
    ep_dists: List[float] = []

    print(f"開始訓練 | Episodes={args.num_episodes} | Device={device}")
    print(f"圖規模: {graph.num_nodes} 節點 | max_neighbors={graph.max_neighbors}")
    print(
        f"Dynamic speed slots: {num_time_slots} | "
        f"slot_minutes={gdata['time_slot_minutes']}"
    )
    print(f"Routing graph mode: {gdata.get('routing_graph_mode', 'legacy')}")
    if gdata.get("routing_graph_mode") == "superzone":
        print(f"Superzone artifacts: {gdata.get('superzone_dir')}")
    elif args.routing_locationid_max > 0:
        print(f"RL zone scope: LocationID <= {args.routing_locationid_max}")
    else:
        print("RL zone scope: full graph")
    print("-" * 70)

    t0 = time.time()

    for ep in range(1, args.num_episodes + 1):
        start, dest = od_pairs[ep - 1]
        time_slot = rng.randint(0, num_time_slots)

        state, action_mask, _ = env.reset(start, dest, time_slot)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, action_mask)
            result = env.step(action)
            next_state = result.state
            next_mask = result.action_mask

            agent.store_transition(
                state, action, result.reward,
                next_state, result.done, action_mask, next_mask,
            )

            agent.learn()

            state = next_state
            action_mask = next_mask
            episode_reward += result.reward
            done = result.done
            info = result.info

        ep_rewards.append(episode_reward)
        ep_successes.append(int(info["reached_goal"]))
        ep_times.append(info["total_travel_time"])
        ep_dists.append(info["total_distance"])

        # ── 定期輸出 ──
        if ep % window == 0:
            avg_r = np.mean(ep_rewards[-window:])
            sr = np.mean(ep_successes[-window:]) * 100
            avg_t = np.mean(ep_times[-window:])
            avg_d = np.mean(ep_dists[-window:])
            elapsed = time.time() - t0

            record = {
                "episode": ep,
                "avg_reward": round(float(avg_r), 3),
                "success_rate": round(float(sr), 2),
                "avg_travel_time": round(float(avg_t), 4),
                "avg_travel_dist": round(float(avg_d), 4),
                "epsilon": round(agent.epsilon, 4),
                "elapsed_sec": round(elapsed, 1),
            }
            history.append(record)

            print(
                f"[Ep {ep:>6d}]  AvgR={avg_r:>8.2f}  "
                f"SR={sr:>5.1f}%  AvgTime={avg_t:>7.3f}  "
                f"AvgDist={avg_d:>7.3f}  ε={agent.epsilon:.4f}  "
                f"({elapsed:.0f}s)"
            )

        # ── 定期存檔 ──
        if ep % args.save_interval == 0:
            ckpt_path = log_dir / f"ddqn_ep{ep}.pt"
            agent.save(ckpt_path)

    # ── 訓練結束 ──
    final_path = log_dir / "ddqn_final.pt"
    agent.save(final_path)
    print(f"\n訓練完成，模型已儲存至 {final_path}")

    log_path = log_dir / "train_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"訓練日誌已儲存至 {log_path}")


# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Double DQN for routing")

    # 環境（圖來自 data/，與 train_predictor 一致）
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument(
        "--edge-length-source",
        type=str,
        default="osrm",
        choices=["osrm", "centroid"],
    )
    p.add_argument(
        "--max-neighbors",
        type=int,
        default=6,
        help="至少為圖的最大出度（會自動與實際出度取 max）",
    )
    p.add_argument(
        "--routing-locationid-max",
        type=int,
        default=63,
        help="Use only zones with LocationID <= this value for RL; 0 uses the full graph",
    )
    p.add_argument(
        "--routing-graph-mode",
        type=str,
        default="superzone",
        choices=["legacy", "superzone"],
        help="Use the legacy induced subgraph or the K=64 superzone RL action graph.",
    )
    p.add_argument(
        "--superzone-dir",
        type=str,
        default="",
        help="Directory produced by build_superzone_graph.py; defaults to data/superzones_k64.",
    )
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--rho", type=float, default=50.0)

    # Agent
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=int, default=5000)
    p.add_argument("--buffer-capacity", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--target-update", type=int, default=200)

    # 訓練
    p.add_argument("--num-episodes", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--save-interval", type=int, default=500)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
