"""
dispatch.py — 計程車派遣模組

基於預測的乘客需求 D̂ 與空車供給 Ĉ，計算供需缺口 Δ = D − C，
並以貪婪策略將盈餘區域的空車派遣至缺口區域。

產出：dispatch_matrix M ∈ Z^{N×N}
      M[i,j] = 從區域 i 派遣到區域 j 的車輛數
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def compute_travel_time_matrix(
    adj: np.ndarray,
    lengths: np.ndarray,
    speeds: np.ndarray,
) -> np.ndarray:
    """
    計算任意兩區域間的直接行駛時間。

    Parameters
    ----------
    adj     : (N, N) 鄰接矩陣 (0/1)
    lengths : (N, N) 路段長度（無邊處為 0）
    speeds  : (N, N) 預測速度（無邊處為 0）

    Returns
    -------
    tt : (N, N) 行駛時間，無直接連線的設為 inf
    """
    N = adj.shape[0]
    tt = np.full((N, N), np.inf)
    mask = (adj > 0) & (speeds > 1e-5)
    tt[mask] = lengths[mask] / speeds[mask]
    np.fill_diagonal(tt, 0.0)
    return tt


def greedy_dispatch(
    demand: np.ndarray,
    supply: np.ndarray,
    travel_time: np.ndarray,
    *,
    skip_unreachable: bool = True,
) -> np.ndarray:
    """
    貪婪派遣：依缺口大小排序，將最近盈餘區域的空車派過去。

    Parameters
    ----------
    demand      : (N,) 預測需求
    supply      : (N,) 預測空車數
    travel_time : (N, N) 行駛時間矩陣

    Returns
    -------
    M : (N, N) int — 派遣矩陣
    """
    N = len(demand)
    gap = demand - supply  # >0 缺車, <0 盈餘

    deficit: List[Tuple[int, float]] = []
    surplus: List[List[float]] = [0.0] * N  # mutable surplus per region
    surplus_vals = np.zeros(N)

    for i in range(N):
        if gap[i] > 0:
            deficit.append((i, gap[i]))
        elif gap[i] < 0:
            surplus_vals[i] = -gap[i]

    # 按缺口從大到小處理
    deficit.sort(key=lambda x: -x[1])

    M = np.zeros((N, N), dtype=np.int32)

    for j, needed in deficit:
        if needed <= 0:
            continue
        remaining = needed
        # 按行駛時間排序，找最近的盈餘區域
        sorted_sources = np.argsort(travel_time[:, j])
        for i in sorted_sources:
            if skip_unreachable and not np.isfinite(travel_time[i, j]):
                continue
            if surplus_vals[i] <= 0:
                continue
            dispatch_count = min(int(remaining), int(surplus_vals[i]))
            if dispatch_count <= 0:
                continue
            M[i, j] = dispatch_count
            surplus_vals[i] -= dispatch_count
            remaining -= dispatch_count
            if remaining <= 0:
                break

    return M


def dispatch_reachability_report(
    M: np.ndarray,
    travel_time: np.ndarray,
) -> dict:
    """Summarize whether assigned dispatch OD pairs have finite travel cost."""
    pairs = build_dispatch_od_pairs(M)
    total_pairs = len(pairs)
    total_vehicles = int(sum(count for _, _, count in pairs))
    reachable_pairs = 0
    reachable_vehicles = 0
    unreachable: list[tuple[int, int, int]] = []
    for origin, dest, count in pairs:
        if np.isfinite(travel_time[origin, dest]):
            reachable_pairs += 1
            reachable_vehicles += int(count)
        else:
            unreachable.append((int(origin), int(dest), int(count)))
    return {
        "pairs": int(total_pairs),
        "vehicles": int(total_vehicles),
        "reachable_pairs": int(reachable_pairs),
        "reachable_vehicles": int(reachable_vehicles),
        "reachable_pair_rate": float(reachable_pairs / total_pairs) if total_pairs else 1.0,
        "reachable_vehicle_rate": float(reachable_vehicles / total_vehicles) if total_vehicles else 1.0,
        "unreachable_pairs": unreachable,
    }


def build_dispatch_od_pairs(
    M: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """
    將派遣矩陣轉換為 (origin, destination, count) 列表，
    供後續路徑規劃使用。
    """
    pairs = []
    N = M.shape[0]
    for i in range(N):
        for j in range(N):
            if M[i, j] > 0:
                pairs.append((i, j, int(M[i, j])))
    return pairs
