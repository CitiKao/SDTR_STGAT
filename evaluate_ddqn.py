"""
evaluate_ddqn.py — 評估腳本

功能：
  1. 載入訓練好的 Double DQN 模型
  2. 在測試 OD pair 上以 greedy 策略跑路徑
  3. 與 Dijkstra 最短路徑做對比
  4. 輸出 success rate / avg travel time / avg distance / 與最短路之比

使用方式（與 train_ddqn 相同 --data-dir / --edge-length-source；--speed-seed 須與訓練 --seed 一致）：
    python evaluate_ddqn.py --model runs/ddqn_final.pt
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import torch

from data_loader import load_nyc_graph_for_rl

from ddqn_agent import DoubleDQNAgent
from graph_env import CityGraph, RoutingEnv, create_graph_from_data, infer_max_neighbors


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 CUDA，但目前環境沒有可用 GPU。")
    return str(device)


# ────────────────────────────────────────────────────────────────
#  評估邏輯
# ────────────────────────────────────────────────────────────────

def run_episode(
    env: RoutingEnv,
    agent: DoubleDQNAgent,
    start: int,
    dest: int,
    time_slot: int = 0,
) -> dict:
    """用 greedy 策略跑一個 episode，回傳 info dict"""
    state, mask, _ = env.reset(start, dest, time_slot)
    done = False
    total_reward = 0.0
    path = [start]

    while not done:
        action = agent.select_action(state, mask, greedy=True)
        result = env.step(action)
        state = result.state
        mask = result.action_mask
        total_reward += result.reward
        done = result.done
        path.append(env.current_node)

    return {
        "start": start,
        "dest": dest,
        "reached": result.info["reached_goal"],
        "reward": total_reward,
        "travel_time": result.info["total_travel_time"],
        "travel_dist": result.info["total_distance"],
        "steps": result.info["steps"],
        "path": path,
    }


def evaluate(args: argparse.Namespace) -> None:
    rng = np.random.RandomState(args.seed)
    device = resolve_device(args.device)

    # ── 建圖（需與 train_ddqn 相同 data/ 與邊長來源）──
    gdata = load_nyc_graph_for_rl(
        args.data_dir,
        edge_length_source=args.edge_length_source,
        speed_seed=args.speed_seed,
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
        use_real_speed=args.use_real_speed,
        dynamic_edge_index=edge_index,
        dynamic_pred_speeds=pred_speed_profile,
        dynamic_real_speeds=real_speed_profile,
    )

    # ── 載入模型 ──
    agent = DoubleDQNAgent(
        num_nodes=graph.num_nodes,
        max_neighbors=graph.max_neighbors,
        state_dim=env.state_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        device=device,
    )
    agent.load(args.model)
    print(f"模型已載入: {args.model}\n")
    print(
        f"Dynamic speed slots: {num_time_slots} | "
        f"slot_minutes={gdata['time_slot_minutes']}"
    )

    # ── 產生測試 OD pair ──
    pairs: List[Tuple[int, int]] = []
    while len(pairs) < args.num_tests:
        o = rng.randint(0, graph.num_nodes)
        d = rng.randint(0, graph.num_nodes)
        if o != d:
            pairs.append((o, d))

    # ── 跑評估 ──
    results: List[dict] = []
    dijk_times: List[float] = []
    dijk_dists: List[float] = []

    for o, d in pairs:
        ts = rng.randint(0, num_time_slots)
        env.reset(o, d, ts)
        _, dt, dd = graph.dijkstra(o, d, use_real_speed=args.use_real_speed)
        res = run_episode(env, agent, o, d, ts)
        results.append(res)

        dijk_times.append(dt)
        dijk_dists.append(dd)

    # ── 統計 ──
    successes = [r for r in results if r["reached"]]
    sr = len(successes) / len(results) * 100

    avg_reward = np.mean([r["reward"] for r in results])
    avg_time = np.mean([r["travel_time"] for r in successes]) if successes else float("nan")
    avg_dist = np.mean([r["travel_dist"] for r in successes]) if successes else float("nan")
    avg_steps = np.mean([r["steps"] for r in successes]) if successes else float("nan")

    dijk_avg_time = np.mean(
        [t for t, r in zip(dijk_times, results) if r["reached"] and t < float("inf")]
    ) if successes else float("nan")
    dijk_avg_dist = np.mean(
        [d for d, r in zip(dijk_dists, results) if r["reached"] and d < float("inf")]
    ) if successes else float("nan")

    print("=" * 60)
    print("              Double DQN 路徑規劃評估結果")
    print("=" * 60)
    print(f"  測試 OD 數量        : {len(results)}")
    print(f"  成功率 (SR)         : {sr:.1f}%")
    print(f"  平均 Episode Reward : {avg_reward:.3f}")
    print("-" * 60)
    print("                       DDQN         Dijkstra")
    print(f"  平均行駛時間        : {avg_time:>10.4f}   {dijk_avg_time:>10.4f}")
    print(f"  平均行駛距離        : {avg_dist:>10.4f}   {dijk_avg_dist:>10.4f}")
    print(f"  平均步數            : {avg_steps:>10.2f}       —")

    if successes and not np.isnan(dijk_avg_time) and dijk_avg_time > 0:
        ratio_t = avg_time / dijk_avg_time
        ratio_d = avg_dist / dijk_avg_dist if dijk_avg_dist > 0 else float("nan")
        print("-" * 60)
        print(f"  DDQN/Dijkstra 時間比 : {ratio_t:.3f}")
        print(f"  DDQN/Dijkstra 距離比 : {ratio_d:.3f}")
    print("=" * 60)

    # ── 展示幾條路徑範例 ──
    print("\n路徑範例（前 5 條成功路徑）：")
    shown = 0
    for i, r in enumerate(results):
        if r["reached"] and shown < 5:
            dpath, _, _ = graph.dijkstra(r["start"], r["dest"])
            print(f"  OD({r['start']}→{r['dest']}):")
            print(f"    DDQN     : {r['path']}  time={r['travel_time']:.4f}  dist={r['travel_dist']:.4f}")
            print(f"    Dijkstra : {dpath}")
            shown += 1


# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained Double DQN")

    p.add_argument("--model", type=str, default="runs/ddqn_final.pt")
    p.add_argument("--num-tests", type=int, default=200)
    p.add_argument("--seed", type=int, default=123, help="測試 OD 抽樣隨機種子")
    p.add_argument(
        "--speed-seed",
        type=int,
        default=42,
        help="須與 train_ddqn 的 --seed 相同（邊上微擾動真實速）",
    )

    # 環境（需與 train_ddqn 一致）
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument(
        "--edge-length-source",
        type=str,
        default="osrm",
        choices=["osrm", "centroid"],
    )
    p.add_argument("--max-neighbors", type=int, default=6)
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
    p.add_argument("--use-real-speed", action="store_true",
                   help="使用 real_speed 模擬真實行駛")

    # Agent 結構（需與訓練時一致）
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu")

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
