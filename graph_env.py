"""
graph_env.py — 城市道路有向圖與路徑規劃 RL 環境

將城市道路建模為有向圖，每條邊有長度與預測/真實速度。
RL 環境在給定起點與終點後，讓 agent 逐步選擇下一個節點走出一條路徑。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ────────────────────────────────────────────────────────────────
#  Edge & Graph
# ────────────────────────────────────────────────────────────────

@dataclass
class Edge:
    """有向邊：從 src → dst"""
    dst: int
    length: float
    predicted_speed: float
    real_speed: float


class CityGraph:
    """
    城市道路有向圖。

    Parameters
    ----------
    num_nodes : 節點總數
    max_neighbors : 任一節點最多保留幾條出邊（用於固定動作空間）
    """

    def __init__(self, num_nodes: int, max_neighbors: int = 6) -> None:
        self.num_nodes: int = num_nodes
        self.max_neighbors: int = max_neighbors
        self.adj: Dict[int, List[Edge]] = {i: [] for i in range(num_nodes)}

    # ── 建圖 ──────────────────────────────────────────────────

    def add_edge(
        self,
        src: int,
        dst: int,
        length: float,
        predicted_speed: float,
        real_speed: float,
    ) -> None:
        self.adj[src].append(Edge(dst, length, predicted_speed, real_speed))
        if len(self.adj[src]) > self.max_neighbors:
            self.max_neighbors = len(self.adj[src])

    # ── 查詢 ──────────────────────────────────────────────────

    def neighbors(self, node: int) -> List[Edge]:
        return self.adj[node]

    def padded_neighbor_info(
        self, node: int
    ) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        回傳 padding 至 max_neighbors 的鄰居資訊。

        Returns
        -------
        neighbor_ids : 實際鄰居節點 ID（長度 ≤ max_neighbors）
        lengths      : shape (max_neighbors,)
        pred_speeds  : shape (max_neighbors,)
        real_speeds  : shape (max_neighbors,)
        action_mask  : shape (max_neighbors,)  1=合法 0=非法
        """
        edges = self.adj[node][: self.max_neighbors]
        M = self.max_neighbors

        neighbor_ids: List[int] = []
        lengths = np.zeros(M, dtype=np.float32)
        pred_speeds = np.zeros(M, dtype=np.float32)
        real_speeds = np.zeros(M, dtype=np.float32)
        action_mask = np.zeros(M, dtype=np.float32)

        for i, e in enumerate(edges):
            neighbor_ids.append(e.dst)
            lengths[i] = e.length
            pred_speeds[i] = e.predicted_speed
            real_speeds[i] = e.real_speed
            action_mask[i] = 1.0

        return neighbor_ids, lengths, pred_speeds, real_speeds, action_mask

    # ── 動態速度更新 ─────────────────────────────────────────

    def update_predicted_speeds(
        self,
        edge_index: np.ndarray,
        new_speeds: np.ndarray,
    ) -> None:
        """
        以 STGAT 預測結果批量更新邊的 predicted_speed。

        Parameters
        ----------
        edge_index : (|E|, 2)  — 有向邊 (src, dst)
        new_speeds : (|E|,)    — 新的預測速度
        """
        speed_map: Dict[Tuple[int, int], float] = {}
        for idx in range(edge_index.shape[0]):
            src, dst = int(edge_index[idx, 0]), int(edge_index[idx, 1])
            speed_map[(src, dst)] = float(new_speeds[idx])

        for src in range(self.num_nodes):
            for edge in self.adj[src]:
                key = (src, edge.dst)
                if key in speed_map:
                    edge.predicted_speed = speed_map[key]

    def update_real_speeds(
        self,
        edge_index: np.ndarray,
        new_speeds: np.ndarray,
    ) -> None:
        """
        批量更新邊的 real_speed。

        Parameters
        ----------
        edge_index : (|E|, 2)  — 有向邊 (src, dst)
        new_speeds : (|E|,)    — 新的真實速度
        """
        speed_map: Dict[Tuple[int, int], float] = {}
        for idx in range(edge_index.shape[0]):
            src, dst = int(edge_index[idx, 0]), int(edge_index[idx, 1])
            speed_map[(src, dst)] = float(new_speeds[idx])

        for src in range(self.num_nodes):
            for edge in self.adj[src]:
                key = (src, edge.dst)
                if key in speed_map:
                    edge.real_speed = speed_map[key]

    # ── Dijkstra（用於 baseline 對比）──────────────────────────

    def dijkstra(
        self, src: int, dst: int, use_real_speed: bool = False
    ) -> Tuple[List[int], float, float]:
        """
        回傳 (path, total_travel_time, total_distance)。
        若不可達回傳 ([], inf, inf)。
        """
        dist = {i: float("inf") for i in range(self.num_nodes)}
        prev: Dict[int, Optional[int]] = {i: None for i in range(self.num_nodes)}
        dist[src] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, src)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == dst:
                break
            for e in self.adj[u]:
                speed = e.real_speed if use_real_speed else e.predicted_speed
                tt = e.length / max(speed, 1e-5)
                nd = d + tt
                if nd < dist[e.dst]:
                    dist[e.dst] = nd
                    prev[e.dst] = u
                    heapq.heappush(pq, (nd, e.dst))

        if dist[dst] == float("inf"):
            return [], float("inf"), float("inf")

        path: List[int] = []
        cur: Optional[int] = dst
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()

        total_time = 0.0
        total_dist = 0.0
        for i in range(len(path) - 1):
            for e in self.adj[path[i]]:
                if e.dst == path[i + 1]:
                    speed = e.real_speed if use_real_speed else e.predicted_speed
                    total_time += e.length / max(speed, 1e-5)
                    total_dist += e.length
                    break

        return path, total_time, total_dist


# ────────────────────────────────────────────────────────────────
#  RL Environment
# ────────────────────────────────────────────────────────────────

# 速度、長度歸一化上界（邊長 km，速度 km/h，與 build_speed_features / OSRM 一致）
_SPEED_NORM = 130.0
_LENGTH_NORM = 5.0


@dataclass
class StepResult:
    state: np.ndarray
    action_mask: np.ndarray
    reward: float
    done: bool
    info: dict


class RoutingEnv:
    """
    路徑規劃 RL 環境。

    State 向量（float32, dim = 2 + 2*M + 1）：
        [current_node, destination,
         pred_speed_0/SPEED_NORM, …, pred_speed_{M-1}/SPEED_NORM,
         length_0/LENGTH_NORM, …, length_{M-1}/LENGTH_NORM,
         time_slot / num_time_slots]

    Action：0 ~ max_neighbors-1，對應第 i 個鄰居。

    Reward:
        r = -α·travel_time − β·edge_length − δ·pingpong_penalty + ρ·goal_reward
    """

    def __init__(
        self,
        graph: CityGraph,
        *,
        alpha: float = 1.0,
        beta: float = 0.1,
        delta: float = 5.0,
        rho: float = 50.0,
        max_steps: int = 50,
        num_time_slots: int = 24,
        time_slot_duration_hours: float = 1.0,
        use_real_speed: bool = False,
        dynamic_edge_index: Optional[np.ndarray] = None,
        dynamic_pred_speeds: Optional[np.ndarray] = None,
        dynamic_real_speeds: Optional[np.ndarray] = None,
    ) -> None:
        self.graph = graph
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.rho = rho
        self.max_steps = max_steps
        self.num_time_slots = num_time_slots
        self.time_slot_duration_hours = max(float(time_slot_duration_hours), 1e-6)
        self.use_real_speed = use_real_speed
        self.dynamic_edge_index = dynamic_edge_index
        self.dynamic_pred_speeds = dynamic_pred_speeds
        self.dynamic_real_speeds = dynamic_real_speeds

        M = graph.max_neighbors
        self.state_dim: int = 2 + 2 * M + 1
        self.action_dim: int = M

        # 以下由 reset() 初始化
        self.current_node: int = -1
        self.destination: int = -1
        self.visited: Set[int] = set()
        self.path_history: List[int] = []
        self.pingpong_streak: int = 0
        self.step_count: int = 0
        self.start_time_slot: int = 0
        self.current_time_slot: int = 0
        self.total_travel_time: float = 0.0
        self.total_distance: float = 0.0

    # ── 狀態構造 ──────────────────────────────────────────────

    def _build_state(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        nids, lens, pspeeds, rspeeds, mask = self.graph.padded_neighbor_info(
            self.current_node
        )
        state = np.concatenate(
            [
                np.array(
                    [self.current_node, self.destination], dtype=np.float32
                ),
                pspeeds / _SPEED_NORM,
                lens / _LENGTH_NORM,
                np.array(
                    [self.current_time_slot / self.num_time_slots],
                    dtype=np.float32,
                ),
            ]
        )
        return state, mask, nids

    # ── reset / step ──────────────────────────────────────────

    def reset(
        self, start: int, destination: int, time_slot: int = 0
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """回傳 (state, action_mask, neighbor_ids)"""
        self.current_node = start
        self.destination = destination
        self.visited = {start}
        self.path_history = [start]
        self.pingpong_streak = 0
        self.step_count = 0
        self.start_time_slot = int(np.clip(time_slot, 0, self.num_time_slots - 1))
        self.current_time_slot = self.start_time_slot
        self.total_travel_time = 0.0
        self.total_distance = 0.0
        self._apply_dynamic_speed_profiles()
        return self._build_state()

    def _apply_dynamic_speed_profiles(self) -> None:
        """Refresh edge predicted/real speeds from the current time-slot schedule."""
        if self.dynamic_edge_index is None:
            return

        if self.dynamic_pred_speeds is not None:
            if self.dynamic_pred_speeds.ndim != 2 or self.dynamic_pred_speeds.shape[1] == 0:
                return
            slot_idx = int(
                np.clip(self.current_time_slot, 0, self.dynamic_pred_speeds.shape[1] - 1)
            )
            self.graph.update_predicted_speeds(
                self.dynamic_edge_index,
                self.dynamic_pred_speeds[:, slot_idx],
            )

        if self.dynamic_real_speeds is not None:
            if self.dynamic_real_speeds.ndim != 2 or self.dynamic_real_speeds.shape[1] == 0:
                return
            slot_idx = int(
                np.clip(self.current_time_slot, 0, self.dynamic_real_speeds.shape[1] - 1)
            )
            self.graph.update_real_speeds(
                self.dynamic_edge_index,
                self.dynamic_real_speeds[:, slot_idx],
            )

    def _pingpong_penalty(self, next_node: int) -> float:
        """
        Penalize repeated back-and-forth oscillation such as ABAB, ABABA, ...

        A single rollback ABA is allowed. The penalty starts when the agent keeps
        extending the same two-node oscillation.
        """
        continues_pingpong = (
            len(self.path_history) >= 3
            and self.path_history[-3] == self.path_history[-1]
            and self.path_history[-2] == next_node
        )
        if continues_pingpong:
            self.pingpong_streak += 1
        else:
            self.pingpong_streak = 0
        return float(self.pingpong_streak)

    def step(self, action: int) -> StepResult:
        nids, lens, pspeeds, rspeeds, mask = self.graph.padded_neighbor_info(
            self.current_node
        )

        # 非法動作：直接結束並重罰
        if action < 0 or action >= len(nids) or mask[action] == 0.0:
            state, new_mask, _ = self._build_state()
            return StepResult(state, new_mask, -20.0, True, self._info(False))

        next_node = nids[action]
        edge_len = lens[action]
        speed = rspeeds[action] if self.use_real_speed else pspeeds[action]
        travel_time = edge_len / max(speed, 1e-5)

        pingpong_penalty = self._pingpong_penalty(next_node)
        reached = next_node == self.destination
        goal_reward = 1.0 if reached else 0.0

        reward = float(
            -self.alpha * travel_time
            - self.beta * edge_len
            - self.delta * pingpong_penalty
            + self.rho * goal_reward
        )

        # 更新環境
        self.current_node = next_node
        self.visited.add(next_node)
        self.path_history.append(next_node)
        self.total_travel_time += travel_time
        self.total_distance += edge_len
        self.step_count += 1
        elapsed_slots = int(self.total_travel_time / self.time_slot_duration_hours)
        self.current_time_slot = min(
            self.start_time_slot + elapsed_slots,
            self.num_time_slots - 1,
        )
        self._apply_dynamic_speed_profiles()

        done = reached or self.step_count >= self.max_steps
        # 若步數用完仍未到達，額外懲罰
        if not reached and self.step_count >= self.max_steps:
            reward -= 10.0

        state, new_mask, _ = self._build_state()
        return StepResult(state, new_mask, reward, done, self._info(reached))

    def _info(self, reached: bool) -> dict:
        return {
            "reached_goal": reached,
            "total_travel_time": self.total_travel_time,
            "total_distance": self.total_distance,
            "steps": self.step_count,
            "visited": self.visited.copy(),
            "path_history": self.path_history.copy(),
            "pingpong_streak": self.pingpong_streak,
        }


# ────────────────────────────────────────────────────────────────
#  Sample Graph 工廠
# ────────────────────────────────────────────────────────────────

def infer_max_neighbors(
    edge_index: np.ndarray,
    num_nodes: int,
    minimum: int = 0,
) -> int:
    """
    Infer the required fixed action-space size from graph out-degree.

    `minimum` is kept as a lower bound so callers can preserve checkpoint
    compatibility when they intentionally trained with a larger padded action
    space than the graph strictly requires.
    """
    if edge_index.size == 0 or num_nodes <= 0:
        return int(max(minimum, 0))

    src = edge_index[:, 0].astype(np.int64)
    max_out = int(np.bincount(src, minlength=num_nodes).max())
    return int(max(minimum, max_out))


def create_sample_graph(
    grid_size: int = 5,
    max_neighbors: int = 6,
    seed: int = 42,
) -> CityGraph:
    """
    建立 grid_size × grid_size 的網格城市圖（雙向邊），
    邊的長度與速度隨機生成，供快速測試使用。
    """
    rng = np.random.RandomState(seed)
    num_nodes = grid_size * grid_size
    graph = CityGraph(num_nodes, max_neighbors)

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    for r in range(grid_size):
        for c in range(grid_size):
            node = r * grid_size + c
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    neighbor = nr * grid_size + nc
                    length = rng.uniform(0.5, 3.0)
                    pred_speed = rng.uniform(20.0, 60.0)
                    real_speed = pred_speed * rng.uniform(0.7, 1.3)
                    graph.add_edge(node, neighbor, length, pred_speed, real_speed)

    return graph


def create_graph_from_data(
    num_nodes: int,
    edge_index: np.ndarray,
    edge_lengths: np.ndarray,
    pred_speeds: np.ndarray,
    real_speeds: Optional[np.ndarray] = None,
    max_neighbors: int = 6,
) -> CityGraph:
    """
    從預測模型的資料格式建構 CityGraph，用於 pipeline 串接。

    Parameters
    ----------
    edge_index   : (|E|, 2)
    edge_lengths : (|E|,)
    pred_speeds  : (|E|,) — STGAT 預測速度
    real_speeds  : (|E|,) — 真實速度（若無則用 pred_speeds）
    """
    if real_speeds is None:
        real_speeds = pred_speeds
    graph = CityGraph(
        num_nodes,
        infer_max_neighbors(edge_index, num_nodes, minimum=max_neighbors),
    )
    for idx in range(edge_index.shape[0]):
        src, dst = int(edge_index[idx, 0]), int(edge_index[idx, 1])
        graph.add_edge(
            src, dst,
            float(edge_lengths[idx]),
            float(pred_speeds[idx]),
            float(real_speeds[idx]),
        )
    return graph
