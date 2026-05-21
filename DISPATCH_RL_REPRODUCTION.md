# STDR 派遣模塊與強化學習模塊復現文檔

生成日期：2026-05-16  
原始工作目錄：`D:\STDR`

本文檔整理目前本機的兩個核心模塊：派遣模塊與強化學習 / Double DQN 路由模塊。內容包含架構說明、資料依賴、環境建立、訓練與評估命令、復現注意事項，以及完整程式碼附錄。

## 1. 目前模塊關係

整體流程由 `pipeline.py` 串接：

```text
STGAT predictor
  -> demand_pred / supply_pred / speed_pred
  -> greedy dispatch
  -> dispatch OD pairs
  -> Double DQN routing
  -> routes
```

其中：

- 派遣決策核心：`dispatch.py`
- 完整管線串接：`pipeline.py`
- DDQN 訓練入口：`train_ddqn.py`
- DDQN 評估入口：`evaluate_ddqn.py`
- RL 環境：`graph_env.py`
- DDQN agent：`ddqn_agent.py`
- Q-network：`q_network.py`
- Replay buffer：`replay_buffer.py`
- NYC / superzone 資料載入：`data_loader.py`
- Superzone 圖與派遣矩陣建立：`superzone_graph.py`、`build_superzone_graph.py`

## 2. 派遣模塊

派遣模塊的主要任務是根據需求與供給缺口，決定從哪些區域調車到哪些區域。

核心函數：

```text
compute_travel_time_matrix(adj, lengths, speeds)
greedy_dispatch(demand, supply, travel_time, skip_unreachable=False)
dispatch_reachability_report(M, travel_time)
build_dispatch_od_pairs(M)
```

派遣矩陣定義：

```text
M[i, j] = 從區域 i 派遣到區域 j 的車輛數
```

缺口定義：

```text
gap_i = demand_i - supply_i
gap_i > 0 代表區域 i 缺車
gap_i < 0 代表區域 i 有餘車
```

目前派遣策略是貪婪法：

1. 計算每個區域的 `gap = demand - supply`。
2. 將缺車區按照缺口大小由大到小排序。
3. 對每個缺車區，根據 travel time 從近到遠尋找有餘車的區域。
4. 每次派遣數量為 `min(剩餘缺口, 可用餘車)`。
5. 若 `skip_unreachable=True`，不可達的 OD pair 會被跳過。
6. 輸出派遣矩陣與 OD 清單。

目前 `pipeline.py` 預設使用 `superzone` 模式。這代表 263 個 NYC taxi zones 會先被聚合成 64 個 superzones，再做派遣。派遣時使用第一個預測 horizon：

```text
demand_pred[:, 0]
supply_pred[:, 0]
```

因此目前派遣是單步即時貪婪派遣，不是多時段全局最佳化。

## 3. 強化學習 / Double DQN 路由模塊

強化學習模塊負責對派遣產生的 OD pair 規劃路徑。每個 episode 給定一組 `(start, destination)`，agent 每一步選擇目前節點的一條出邊，直到到達目的地或超過最大步數。

目前預設路由圖是 `superzone` action graph：

```text
節點數：64
有向邊數：640
每個節點出邊數：10
```

State 格式：

```text
[
  current_node,
  destination,
  pred_speed_0 / SPEED_NORM,
  ...,
  pred_speed_{M-1} / SPEED_NORM,
  length_0 / LENGTH_NORM,
  ...,
  length_{M-1} / LENGTH_NORM,
  current_time_slot / num_time_slots
]
```

在目前 superzone 圖中，`M = 10`，所以 state dimension 是：

```text
2 + 2 * 10 + 1 = 23
```

Action 定義：

```text
action = 0 ... M-1
```

每個 action 對應目前節點的第 `action` 個鄰居。不存在的鄰居會由 action mask 標記為非法；DDQN 選動作時會把非法 action 的 Q value 設成 `-inf`。

Reward 設計：

```text
r =
  - alpha * travel_time
  - beta  * edge_length
  - delta * pingpong_penalty
  + rho   * goal_reward
```

預設參數：

```text
alpha = 1.0
beta = 0.1
delta = 5.0
rho = 50.0
max_steps = 50
```

特殊情況：

- 非法 action：立即結束 episode，reward = `-20`。
- 成功到達目的地：獲得 `rho * goal_reward`，預設為 `+50`。
- 超過 `max_steps` 還沒到達：額外扣 `-10`。
- ping-pong penalty 用來懲罰 ABAB 類型的反覆來回震盪。

DDQN 更新邏輯：

1. `online_net` 計算目前狀態的 `Q(s, a)`。
2. 下一狀態由 `online_net` 選出 best action。
3. 下一狀態的 Q value 由 `target_net` 評估。
4. target 為：

```text
target = reward + gamma * Q_target(next_state, best_action) * (1 - done)
```

5. loss 使用 `SmoothL1Loss`。
6. 做 gradient clipping，`max_norm = 10.0`。
7. 每 `target_update` 次 learn 同步 target network。

## 4. 需要同步到另一台電腦的資料

若要完全復現目前本機狀態，至少需要同步以下資料：

```text
data/adjacency_matrix.npy
data/edge_index.npy
data/edge_lengths.npy
data/edge_lengths_osrm.npy
data/edge_durations_osrm.npy
data/edge_speeds.npy
data/edge_speeds_avg.npy
data/node_demand.npy
data/node_supply.npy
data/time_meta.csv
data/zone_info.csv
data/superzones_k64/
```

目前 `data/superzones_k64/` 是已建立好的 superzone 派遣與 RL 圖資料。若另一台電腦直接同步這個資料夾，最容易復現目前行為；若重新建立，OSRM 查詢結果、網路狀況或 fallback 設定可能造成差異。

目前 `superzone_meta.json` 的關鍵設定：

```text
num_base_zones = 263
num_superzones = 64
osrm_topk = 8
connector_count = 2
cost_source = osrm_table
cost_matrix_origin = cached_osrm_table
no_stay_action = true
rl_speed_profile_source = dynamic_stgat_edge_aggregation
```

目前主要資料形狀：

```text
region_membership.npy: (263,)
region_node_demand.npy: (35040, 64)
region_node_supply.npy: (35040, 64)
dispatch_distance_km.npy: (64, 64)
dispatch_duration_hours.npy: (64, 64)
dispatch_reachable.npy: (64, 64)
rl_edge_index.npy: (640, 2)
rl_edge_lengths.npy: (640,)
rl_edge_durations_hours.npy: (640,)
rl_edge_speeds_kmh.npy: (640,)
rl_edge_speed_mapping_offsets.npy: (641,)
rl_edge_speed_mapping_indices.npy: (17757,)
```

## 5. 建立環境

建議使用 conda：

```powershell
cd D:\STDR
conda env create -f environment.yml
conda activate STDR
python -c "import torch, numpy, pandas; print(torch.__version__)"
```

也可以使用 venv：

```powershell
cd D:\STDR
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

主要依賴：

```text
python = 3.10
numpy >= 1.24
pandas >= 2.0
torch >= 2.0
geopandas >= 0.14
pyogrio >= 0.8
pyproj >= 3.6
shapely >= 2.0
matplotlib >= 3.8
pyarrow >= 14.0
```

## 6. 驗證資料是否可讀

在另一台電腦同步資料後，可先執行：

```powershell
python - <<'PY'
import numpy as np
from pathlib import Path

root = Path("data")
for name in [
    "adjacency_matrix.npy",
    "edge_index.npy",
    "edge_speeds_avg.npy",
    "node_demand.npy",
    "node_supply.npy",
]:
    arr = np.load(root / name, mmap_mode="r")
    print(name, arr.shape, arr.dtype)

sroot = root / "superzones_k64"
for name in [
    "region_membership.npy",
    "dispatch_duration_hours.npy",
    "rl_edge_index.npy",
    "rl_edge_speeds_kmh.npy",
]:
    arr = np.load(sroot / name, mmap_mode="r")
    print("superzone", name, arr.shape, arr.dtype)
PY
```

## 7. 重建 superzone artifacts

若沒有直接同步 `data/superzones_k64/`，可以重建：

```powershell
python build_superzone_graph.py --data-dir data --output-dir data/superzones_k64
python validate_superzone_graph.py --data-dir data --superzone-dir data/superzones_k64
```

注意：重建時需要 OSRM table 查詢。若另一台電腦無法穩定連線，建議直接同步本機已生成的 `data/superzones_k64/`。

## 8. 訓練 Double DQN

```powershell
python train_ddqn.py ^
  --data-dir data ^
  --routing-graph-mode superzone ^
  --superzone-dir data/superzones_k64 ^
  --device auto ^
  --log-dir runs/ddqn_superzone
```

訓練完成後應產生：

```text
runs/ddqn_superzone/ddqn_final.pt
runs/ddqn_superzone/train_log.json
```

目前本機沒有找到 `ddqn_final.pt` 或 `ddqn_best.pt`，所以如果另一台電腦要執行 DDQN route，需要先重新訓練，或從其他備份取得 DDQN 權重。

## 9. 評估 Double DQN

```powershell
python evaluate_ddqn.py ^
  --model runs/ddqn_superzone/ddqn_final.pt ^
  --data-dir data ^
  --routing-graph-mode superzone ^
  --superzone-dir data/superzones_k64 ^
  --num-tests 200 ^
  --device cpu
```

評估會輸出：

```text
success rate
average episode reward
average DDQN travel time
average DDQN travel distance
DDQN / Dijkstra time ratio
DDQN / Dijkstra distance ratio
sample paths
```

## 10. 執行完整 STDR pipeline

完整 pipeline 需要兩個 checkpoint：

```text
STGAT predictor checkpoint
DDQN router checkpoint
```

範例：

```powershell
python pipeline.py ^
  --data-dir data ^
  --predictor-ckpt runs/stgat_best.pt ^
  --router-ckpt runs/ddqn_superzone/ddqn_final.pt ^
  --routing-graph-mode superzone ^
  --superzone-dir data/superzones_k64 ^
  --device cpu
```

若只要復現派遣與 DDQN 路由模塊，可以先不用完整 pipeline，只跑 `train_ddqn.py` 和 `evaluate_ddqn.py`。

## 11. 目前本機狀態與限制

- 派遣模塊程式碼存在。
- 強化學習 / DDQN 路由模塊程式碼存在。
- `data/superzones_k64/` 已存在，最近修改時間是 2026-05-13 22:57:16。
- 目前沒有找到 DDQN 訓練後權重 `ddqn_final.pt` / `ddqn_best.pt`。
- DDQN 訓練中的 `real_speed_profile` 目前是由 `pred_speed_profile * uniform(0.7, 1.3)` 產生的擾動版本，不是真實觀測速度。
- 派遣策略目前是單步貪婪派遣，不是多時段全局最佳化。
- 完整 pipeline 需要 STGAT predictor 權重；單獨訓練 / 評估 DDQN 則不需要 predictor checkpoint。

## 12. 完整程式碼附錄

以下附錄直接嵌入目前本機程式碼。若原始 `.py` 檔遺失，可以從這些 code block 還原；實際搬移時仍建議直接複製原始檔。

## A. Code File Manifest

| File | Bytes | Last modified | SHA256 prefix |
| --- | ---: | --- | --- |
| `dispatch.py` | 4289 | 2026-05-13 17:56:12 | `8610c0ac1cece5ab` |
| `pipeline.py` | 30476 | 2026-05-13 22:39:54 | `83edae4890ab6c1c` |
| `train_ddqn.py` | 11686 | 2026-05-13 22:23:42 | `061ca840bb1500d8` |
| `evaluate_ddqn.py` | 9693 | 2026-05-13 22:23:45 | `a277b5dfb8d16e48` |
| `ddqn_agent.py` | 9316 | 2026-04-09 12:57:25 | `577e6ab20c43ff01` |
| `graph_env.py` | 18762 | 2026-05-13 22:52:28 | `4f2c3718288a192b` |
| `q_network.py` | 2074 | 2026-04-09 12:57:25 | `ddaab9e1a1eb6c73` |
| `replay_buffer.py` | 3101 | 2026-04-09 12:57:25 | `6163c5fe3b42920a` |
| `data_loader.py` | 27680 | 2026-05-13 22:25:32 | `4313305c38d2fe5c` |
| `superzone_graph.py` | 44202 | 2026-05-13 22:56:12 | `df1630424f884197` |
| `build_superzone_graph.py` | 2185 | 2026-05-13 22:53:50 | `bf4bbce1a252cc4b` |
| `validate_superzone_graph.py` | 36250 | 2026-05-13 22:56:24 | `ea6e3faa0345aa30` |
| `stgat_model.py` | 21902 | 2026-04-09 12:57:25 | `7384161cd7e58dca` |
| `predictor_normalization.py` | 6619 | 2026-04-08 15:35:08 | `a9f83ce76400f4cb` |
| `requirements.txt` | 188 | 2026-04-09 12:57:25 | `d5ff3366632408ae` |
| `environment.yml` | 170 | 2026-03-27 16:49:17 | `0dca5fd8378f36e8` |

## B. Required Data Manifest

| File | Bytes | Last modified | SHA256 prefix |
| --- | ---: | --- | --- |
| `data/adjacency_matrix.npy` | 276804 | 2026-03-25 16:23:01 | `db8a4b9f83b936b4` |
| `data/edge_index.npy` | 10624 | 2026-03-28 06:15:54 | `1246d50bfbad78c7` |
| `data/edge_lengths.npy` | 276804 | 2026-03-25 18:04:08 | `1fdbf9029c245c08` |
| `data/edge_lengths_osrm.npy` | 276804 | 2026-03-25 18:35:49 | `ab56afd5ea23f716` |
| `data/edge_durations_osrm.npy` | 276804 | 2026-03-25 18:35:49 | `f4c75d7f2e44ae4a` |
| `data/edge_speeds.npy` | 183890048 | 2026-03-28 06:15:55 | `8101df6eb579c47d` |
| `data/edge_speeds_avg.npy` | 503936 | 2026-03-28 06:15:55 | `3b1df70b00f682f6` |
| `data/node_demand.npy` | 36862208 | 2026-03-28 06:15:55 | `dd6ca33cd72076c9` |
| `data/node_supply.npy` | 36862208 | 2026-03-28 06:15:55 | `2c8acfcf41881dd3` |
| `data/time_meta.csv` | 1263408 | 2026-03-28 06:15:55 | `a68f65df5b5bbd38` |
| `data/zone_info.csv` | 7940 | 2026-03-25 16:23:01 | `2caf005add2e1351` |

## C. Superzone Artifact Manifest

| File | Bytes | Last modified | SHA256 prefix |
| --- | ---: | --- | --- |
| `data/superzones_k64/superzone_meta.json` | 557 | 2026-05-13 22:57:16 | `711fb9b664432835` |
| `data/superzones_k64/region_membership.npy` | 1180 | 2026-05-13 22:57:16 | `19797b41994c292b` |
| `data/superzones_k64/region_membership.csv` | 1981 | 2026-05-13 22:57:16 | `68bf968534c37a4a` |
| `data/superzones_k64/region_info.csv` | 13570 | 2026-05-13 22:57:16 | `b62b15052ad8e235` |
| `data/superzones_k64/region_node_demand.npy` | 8970368 | 2026-05-13 22:57:16 | `0a50307bb974fbc6` |
| `data/superzones_k64/region_node_supply.npy` | 8970368 | 2026-05-13 22:57:16 | `588f3c91d49a77c1` |
| `data/superzones_k64/dispatch_distance_km.npy` | 16512 | 2026-05-13 22:57:16 | `418f30d0dd8c7d68` |
| `data/superzones_k64/dispatch_duration_hours.npy` | 16512 | 2026-05-13 22:57:16 | `71b6d38e52712fc3` |
| `data/superzones_k64/dispatch_reachable.npy` | 4224 | 2026-05-13 22:57:16 | `59424a138729536c` |
| `data/superzones_k64/rl_action_info.csv` | 55142 | 2026-05-13 22:57:16 | `b65569a6a11c0fe6` |
| `data/superzones_k64/rl_edge_index.npy` | 5248 | 2026-05-13 22:57:16 | `6230dadd76563d5b` |
| `data/superzones_k64/rl_edge_lengths.npy` | 2688 | 2026-05-13 22:57:16 | `f21ffed216768a98` |
| `data/superzones_k64/rl_edge_durations_hours.npy` | 2688 | 2026-05-13 22:57:16 | `9b60c7aeaa436743` |
| `data/superzones_k64/rl_edge_speeds_kmh.npy` | 2688 | 2026-05-13 22:57:16 | `9181b7f34895a271` |
| `data/superzones_k64/rl_edge_speed_mapping_offsets.npy` | 2692 | 2026-05-13 22:57:16 | `16953353b82049b9` |
| `data/superzones_k64/rl_edge_speed_mapping_indices.npy` | 71156 | 2026-05-13 22:57:16 | `3f1c831f6495fbf1` |

## D. Source Code

### dispatch.py

```python
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
    skip_unreachable: bool = False,
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
```

### pipeline.py

```python
"""
pipeline.py — STDR 完整管線

端到端流程：
  1. STGAT 預測 → 未來需求 D̂、空車 Ĉ、速度 V̂
  2. Dispatch   → 依供需缺口生成派遣矩陣 M
  3. Double DQN → 對每組 (origin, destination) 規劃路徑

使用方式：
    python pipeline.py
    python pipeline.py --predictor-ckpt runs/stgat_best.pt --router-ckpt runs/ddqn_final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from data_loader import (
    build_induced_subgraph,
    load_nyc_real_graph_features,
    load_zone_metadata,
    select_zone_indices_by_locationid_max,
)
from ddqn_agent import DoubleDQNAgent
from dispatch import (
    build_dispatch_od_pairs,
    compute_travel_time_matrix,
    dispatch_reachability_report,
    greedy_dispatch,
)
from graph_env import RoutingEnv, create_graph_from_data, infer_max_neighbors
from predictor_normalization import (
    denormalize_count_values,
    denormalize_speed_values,
    load_normalization_stats,
    normalize_node_features,
    normalize_speed_features,
)
from stgat_model import STGATPredictor
from superzone_graph import aggregate_speed_profile_to_rl_edges, load_superzone_artifacts


def _load_predictor_checkpoint(
    predictor: STGATPredictor,
    path: str | Path,
    *,
    device: str,
    num_nodes: int,
    num_edges: int,
) -> None:
    """Load a predictor checkpoint with a graph-specific compatibility error."""
    ckpt = torch.load(path, map_location=device, weights_only=True)

    mismatches: List[str] = []
    ckpt_num_nodes = None
    ckpt_num_edges = None

    emb_src = ckpt.get("emb_src")
    if emb_src is not None and emb_src.ndim >= 1:
        ckpt_num_nodes = int(emb_src.shape[0])
    elif "emb_dst" in ckpt and ckpt["emb_dst"].ndim >= 1:
        ckpt_num_nodes = int(ckpt["emb_dst"].shape[0])

    edge_index_ckpt = ckpt.get("edge_index")
    if edge_index_ckpt is not None and edge_index_ckpt.ndim >= 1:
        ckpt_num_edges = int(edge_index_ckpt.shape[0])

    if ckpt_num_nodes is not None and ckpt_num_nodes != num_nodes:
        mismatches.append(f"num_nodes checkpoint={ckpt_num_nodes}, current={num_nodes}")
    if ckpt_num_edges is not None and ckpt_num_edges != num_edges:
        mismatches.append(f"num_edges checkpoint={ckpt_num_edges}, current={num_edges}")

    try:
        predictor.load_state_dict(ckpt)
    except RuntimeError as exc:
        mismatch_text = "; ".join(mismatches) if mismatches else str(exc).splitlines()[0]
        raise ValueError(
            "Predictor checkpoint is incompatible with the current graph setup: "
            f"{mismatch_text}. Use a checkpoint trained on the same 263-zone graph."
        ) from exc


# ════════════════════════════════════════════════════════════════
#  Pipeline
# ════════════════════════════════════════════════════════════════

class STDRPipeline:
    """
    STDR 完整管線：預測 → 派遣 → 路徑規劃

    Parameters
    ----------
    predictor    : 已載入權重的 STGATPredictor
    router       : 已載入權重的 DoubleDQNAgent
    edge_index   : (|E|, 2)
    edge_lengths : (|E|,)
    adj          : (N, N)
    max_neighbors: RL 環境的最大鄰居數
    device       : torch device
    """

    def __init__(
        self,
        predictor: STGATPredictor,
        router: DoubleDQNAgent,
        edge_index: np.ndarray,
        edge_lengths: np.ndarray,
        adj: np.ndarray,
        *,
        normalization_stats: dict | None = None,
        routing_full_node_indices: np.ndarray | None = None,
        routing_graph_mode: str = "superzone",
        superzone_artifacts: dict | None = None,
        superzone_data_dir: str | Path = "data",
        superzone_dir: str | Path | None = None,
        max_neighbors: int = 6,
        time_slot_duration_hours: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.predictor = predictor
        self.router = router
        self.edge_index = edge_index
        self.edge_lengths = edge_lengths
        self.adj = adj
        self.num_nodes = adj.shape[0]
        self.max_neighbors = max_neighbors
        self.time_slot_duration_hours = max(float(time_slot_duration_hours), 1e-6)
        self.device = torch.device(device)
        self.normalization_stats = normalization_stats
        self.routing_graph_mode = routing_graph_mode
        self.superzone_artifacts = superzone_artifacts
        self.last_dispatch_reachability_report: dict | None = None
        if routing_graph_mode == "superzone":
            if superzone_artifacts is None:
                superzone_artifacts = load_superzone_artifacts(superzone_data_dir, superzone_dir)
            self.superzone_artifacts = superzone_artifacts
            self.region_membership = superzone_artifacts["membership"].astype(np.int32)
            self.routing_adj = np.zeros(
                (
                    superzone_artifacts["region_demand"].shape[1],
                    superzone_artifacts["region_demand"].shape[1],
                ),
                dtype=np.float32,
            )
            self.routing_edge_index = superzone_artifacts["rl_edge_index"].astype(np.int32)
            if self.routing_edge_index.size > 0:
                self.routing_adj[self.routing_edge_index[:, 0], self.routing_edge_index[:, 1]] = 1.0
            self.routing_edge_lengths = superzone_artifacts["rl_edge_lengths"].astype(np.float32)
            self.routing_full_node_indices = np.arange(self.routing_adj.shape[0], dtype=np.int32)
            self.routing_full_to_sub = self.routing_full_node_indices.copy()
            self.routing_full_edge_indices = np.arange(self.routing_edge_index.shape[0], dtype=np.int32)
            self.routing_num_nodes = self.routing_adj.shape[0]
            self.max_neighbors = infer_max_neighbors(
                self.routing_edge_index,
                self.routing_num_nodes,
                minimum=max_neighbors,
            )
            self.dispatch_reachable = superzone_artifacts["dispatch_reachable"].astype(bool)
            self.dispatch_travel_time = np.where(
                self.dispatch_reachable,
                superzone_artifacts["dispatch_duration_hours"].astype(np.float32),
                np.inf,
            )
            np.fill_diagonal(self.dispatch_travel_time, 0.0)
            return
        if routing_graph_mode != "legacy":
            raise ValueError(
                f"Unsupported routing_graph_mode={routing_graph_mode!r}; expected 'legacy' or 'superzone'."
            )
        if routing_full_node_indices is None:
            routing_full_node_indices = np.arange(self.num_nodes, dtype=np.int32)
        self.routing_full_node_indices = np.asarray(
            routing_full_node_indices, dtype=np.int32
        )
        routing_subgraph = build_induced_subgraph(
            self.num_nodes,
            self.edge_index,
            self.edge_lengths,
            self.routing_full_node_indices,
        )
        self.routing_adj = routing_subgraph["adj"]
        self.routing_edge_index = routing_subgraph["edge_index"]
        self.routing_edge_lengths = routing_subgraph["edge_lengths"]
        self.routing_full_to_sub = routing_subgraph["full_to_sub"]
        self.routing_full_edge_indices = routing_subgraph["full_edge_indices"]
        self.routing_num_nodes = self.routing_full_node_indices.shape[0]
        self.max_neighbors = infer_max_neighbors(
            self.routing_edge_index,
            self.routing_num_nodes,
            minimum=max_neighbors,
        )

    # ── 1. 預測 ──────────────────────────────────────────────

    def predict(
        self, node_seq: np.ndarray, speed_seq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ----------
        node_seq  : (N, h, C)
        speed_seq : (|E|, h)

        Returns
        -------
        demand_pred : (N, p)
        supply_pred : (N, p)
        speed_pred  : (|E|, p)
        """
        node_seq = normalize_node_features(node_seq, self.normalization_stats)
        speed_seq = normalize_speed_features(
            speed_seq,
            self.normalization_stats,
            edge_axis=0,
        )
        self.predictor.eval()
        with torch.no_grad():
            ns = torch.from_numpy(node_seq).float().unsqueeze(0).to(self.device)
            ss = torch.from_numpy(speed_seq).float().unsqueeze(0).to(self.device)
            d, c, v = self.predictor(ns, ss)
        d_pred = d.squeeze(0).cpu().numpy()
        c_pred = c.squeeze(0).cpu().numpy()
        v_pred = v.squeeze(0).cpu().numpy()
        return (
            denormalize_count_values(d_pred, self.normalization_stats, task="demand"),
            denormalize_count_values(c_pred, self.normalization_stats, task="supply"),
            denormalize_speed_values(v_pred, self.normalization_stats, edge_axis=0),
        )

    # ── 2. 派遣 ──────────────────────────────────────────────

    def dispatch(
        self,
        demand_pred: np.ndarray,
        supply_pred: np.ndarray,
        speed_pred: np.ndarray,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """
        Parameters
        ----------
        demand_pred : (N, p)
        supply_pred : (N, p)
        speed_pred  : (|E|, p)

        Returns
        -------
        M      : legacy mode returns a zone-level (N, N) matrix; superzone
                 mode returns a region-level (K, K) matrix.
        pairs  : list of (origin, destination, count) in the same level as M
        """
        if self.routing_graph_mode == "superzone":
            demand = np.bincount(
                self.region_membership,
                weights=np.maximum(demand_pred[:, 0], 0.0),
                minlength=self.routing_num_nodes,
            ).astype(np.float32)
            supply = np.bincount(
                self.region_membership,
                weights=np.maximum(supply_pred[:, 0], 0.0),
                minlength=self.routing_num_nodes,
            ).astype(np.float32)
            routing_M = greedy_dispatch(
                demand,
                supply,
                self.dispatch_travel_time,
                skip_unreachable=True,
            )
            self.last_dispatch_reachability_report = dispatch_reachability_report(
                routing_M,
                self.dispatch_travel_time,
            )
            return routing_M, build_dispatch_od_pairs(routing_M)

        # Dispatch is performed on the RL subgraph so it matches the routing scope.
        full_speeds = np.maximum(speed_pred[:, 0], 1.0)
        node_idx = self.routing_full_node_indices
        demand = np.maximum(demand_pred[node_idx, 0], 0)
        supply = np.maximum(supply_pred[node_idx, 0], 0)
        speeds = full_speeds[self.routing_full_edge_indices]

        N = self.routing_num_nodes
        speed_matrix = np.zeros((N, N), dtype=np.float32)
        length_matrix = np.zeros((N, N), dtype=np.float32)
        for idx in range(self.routing_edge_index.shape[0]):
            i, j = int(self.routing_edge_index[idx, 0]), int(self.routing_edge_index[idx, 1])
            speed_matrix[i, j] = speeds[idx]
            length_matrix[i, j] = self.routing_edge_lengths[idx]

        tt = compute_travel_time_matrix(self.routing_adj, length_matrix, speed_matrix)
        routing_M = greedy_dispatch(demand, supply, tt)
        routing_pairs = build_dispatch_od_pairs(routing_M)

        M = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int32)
        pairs: List[Tuple[int, int, int]] = []
        for origin_sub, dest_sub, count in routing_pairs:
            origin = int(self.routing_full_node_indices[origin_sub])
            dest = int(self.routing_full_node_indices[dest_sub])
            M[origin, dest] = count
            pairs.append((origin, dest, count))
        return M, pairs

    # ── 3. 路徑規劃 ──────────────────────────────────────────

    def route(
        self,
        pairs: List[Tuple[int, int, int]],
        speed_pred: np.ndarray,
        real_speeds: np.ndarray | None = None,
    ) -> List[Dict]:
        """
        對每組 (origin, dest) 用 Double DQN 規劃路徑。

        Parameters
        ----------
        pairs       : [(origin, dest, count), ...]
        speed_pred  : (|E|, p) — 依預測時間槽動態更新的速度序列
        real_speeds : (|E|,) — 模擬實際行駛用（若 None 用 pred）

        Returns
        -------
        list of route dicts
        """
        if self.routing_graph_mode == "superzone":
            base_speeds = np.maximum(
                self.superzone_artifacts["rl_edge_speeds_kmh"].astype(np.float32),
                1.0,
            )
            pred_speed_series = aggregate_speed_profile_to_rl_edges(
                speed_pred,
                self.superzone_artifacts["rl_edge_speed_mapping_offsets"],
                self.superzone_artifacts["rl_edge_speed_mapping_indices"],
                base_speeds,
            )
            pred_sp = pred_speed_series[:, 0]
            if real_speeds is not None:
                real_profile = np.asarray(real_speeds, dtype=np.float32)
                if real_profile.ndim == 1:
                    real_profile = real_profile[:, None]
                real_speed_series = aggregate_speed_profile_to_rl_edges(
                    real_profile,
                    self.superzone_artifacts["rl_edge_speed_mapping_offsets"],
                    self.superzone_artifacts["rl_edge_speed_mapping_indices"],
                    base_speeds,
                )
                real_sp = real_speed_series[:, 0]
            else:
                real_speed_series = None
                real_sp = pred_sp
        else:
            pred_speed_series_full = np.maximum(speed_pred, 1.0)
            pred_speed_series = pred_speed_series_full[self.routing_full_edge_indices]
            pred_sp = pred_speed_series[:, 0]
            if real_speeds is not None:
                real_profile = np.asarray(real_speeds, dtype=np.float32)
                if real_profile.ndim == 1:
                    real_speed_series = real_profile[self.routing_full_edge_indices, None]
                else:
                    real_speed_series = real_profile[self.routing_full_edge_indices]
                real_sp = real_speed_series[:, 0]
            else:
                real_speed_series = None
                real_sp = pred_sp

        graph = create_graph_from_data(
            self.routing_num_nodes,
            self.routing_edge_index,
            self.routing_edge_lengths,
            pred_sp,
            real_sp,
            max_neighbors=self.max_neighbors,
        )
        env = RoutingEnv(
            graph,
            max_steps=50,
            num_time_slots=pred_speed_series.shape[1],
            time_slot_duration_hours=self.time_slot_duration_hours,
            use_real_speed=(real_speeds is not None),
            dynamic_edge_index=self.routing_edge_index,
            dynamic_pred_speeds=pred_speed_series,
            dynamic_real_speeds=real_speed_series,
        )

        routes = []
        for origin, dest, count in pairs:
            if self.routing_graph_mode == "superzone":
                origin_sub = int(origin)
                dest_sub = int(dest)
            else:
                origin_sub = int(self.routing_full_to_sub[origin])
                dest_sub = int(self.routing_full_to_sub[dest])
            if origin_sub < 0 or dest_sub < 0:
                routes.append({
                    "origin": origin,
                    "destination": dest,
                    "count": count,
                    "path": [origin],
                    "reached": False,
                    "travel_time": float("nan"),
                    "travel_dist": float("nan"),
                    "reward": 0.0,
                    "skipped": True,
                    "skip_reason": "outside_routing_subgraph",
                })
                continue

            state, mask, _ = env.reset(origin_sub, dest_sub)
            path = [origin]
            done = False
            total_reward = 0.0

            while not done:
                action = self.router.select_action(state, mask, greedy=True)
                result = env.step(action)
                state = result.state
                mask = result.action_mask
                total_reward += result.reward
                done = result.done
                if self.routing_graph_mode == "superzone":
                    path.append(int(env.current_node))
                else:
                    path.append(int(self.routing_full_node_indices[env.current_node]))

            routes.append({
                "origin": origin,
                "destination": dest,
                "count": count,
                "path": path,
                "reached": result.info["reached_goal"],
                "travel_time": result.info["total_travel_time"],
                "travel_dist": result.info["total_distance"],
                "reward": total_reward,
                "skipped": False,
            })

        return routes

    # ── 端到端 ────────────────────────────────────────────────

    def run(
        self,
        node_seq: np.ndarray,
        speed_seq: np.ndarray,
        real_speeds: np.ndarray | None = None,
    ) -> Dict:
        """
        端到端執行：預測 → 派遣 → 路由

        Returns
        -------
        dict with keys: predictions, dispatch_matrix, od_pairs, routes
        """
        d_pred, c_pred, v_pred = self.predict(node_seq, speed_seq)
        M, pairs = self.dispatch(d_pred, c_pred, v_pred)
        routes = self.route(pairs, v_pred, real_speeds)

        return {
            "demand_pred": d_pred,
            "supply_pred": c_pred,
            "speed_pred": v_pred,
            "dispatch_matrix": M,
            "dispatch_matrix_level": "superzone" if self.routing_graph_mode == "superzone" else "zone",
            "dispatch_region_membership": self.region_membership if self.routing_graph_mode == "superzone" else None,
            "dispatch_reachability_report": self.last_dispatch_reachability_report,
            "od_pairs": pairs,
            "routes": routes,
        }


# ════════════════════════════════════════════════════════════════
#  Demo / CLI
# ════════════════════════════════════════════════════════════════

def demo(args: argparse.Namespace) -> None:
    device = args.device
    meta_path = Path(args.log_dir) / "stgat_meta.json"

    # ── 載入元資訊 ──
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        print("找不到 stgat_meta.json，使用預設模型超參數；圖與時序仍從 data/ 載入")
        meta = {
            "hidden_dim": 32, "num_heads": 4,
            "num_st_blocks": 2, "num_gtcn_layers": 2, "kernel_size": 3,
            "pred_horizon": 3, "hist_len": 12,
            "node_feat_dim": 2, "use_time_features": False,
        }
    normalization_stats = load_normalization_stats(meta.get("normalization"))

    # ── 資料 ──
    print("使用紐約真實路網與 data/ 時序特徵 ...")
    data = load_nyc_real_graph_features(
        args.data_dir,
        max_time_steps=args.max_time_steps or 0,
        edge_length_source=args.edge_length_source,
        add_time_features=bool(meta.get("use_time_features", False)),
    )
    adj = data["adj"]
    edge_index = data["edge_index"]
    edge_lengths = data["edge_lengths"]
    node_feat = data["node_features"]
    edge_speeds = data["edge_speeds"]
    N = adj.shape[0]
    nE = edge_index.shape[0]
    meta_num_nodes = meta.get("num_nodes")
    meta_num_edges = meta.get("num_edges")
    if meta_num_nodes not in (None, N) or meta_num_edges not in (None, nE):
        print(
            "警告: stgat_meta.json 與目前 data/ 圖大小不一致，"
            f"meta=({meta_num_nodes} nodes, {meta_num_edges} edges), "
            f"current=({N} nodes, {nE} edges)"
        )
    superzone_artifacts = None
    if args.routing_graph_mode == "superzone":
        superzone_artifacts = load_superzone_artifacts(args.data_dir, args.superzone_dir or None)
        routing_node_indices = np.arange(
            superzone_artifacts["region_demand"].shape[1], dtype=np.int32
        )
        routing_subgraph = {
            "adj": np.zeros(
                (
                    superzone_artifacts["region_demand"].shape[1],
                    superzone_artifacts["region_demand"].shape[1],
                ),
                dtype=np.float32,
            ),
            "edge_index": superzone_artifacts["rl_edge_index"],
            "edge_lengths": superzone_artifacts["rl_edge_lengths"],
        }
        if routing_subgraph["edge_index"].size > 0:
            routing_subgraph["adj"][
                routing_subgraph["edge_index"][:, 0],
                routing_subgraph["edge_index"][:, 1],
            ] = 1.0
    elif args.routing_locationid_max > 0:
        zone_info = load_zone_metadata(args.data_dir)
        routing_node_indices = select_zone_indices_by_locationid_max(
            zone_info, args.routing_locationid_max
        )
        routing_subgraph = build_induced_subgraph(
            N, edge_index, edge_lengths, routing_node_indices
        )
    else:
        routing_node_indices = np.arange(N, dtype=np.int32)
        routing_subgraph = build_induced_subgraph(
            N, edge_index, edge_lengths, routing_node_indices
        )
    routing_N = routing_subgraph["adj"].shape[0]
    h = meta["hist_len"]
    if node_feat.shape[0] <= h:
        raise ValueError(
            f"時間步 {node_feat.shape[0]} 不足以構成歷史窗口 h={h}，請縮短 hist_len 或增加資料"
        )

    # ── 載入 predictor ──
    predictor = STGATPredictor(
        num_nodes=N,
        edge_index=torch.from_numpy(edge_index),
        edge_lengths=torch.from_numpy(edge_lengths),
        adj_matrix=torch.from_numpy(adj),
        hidden_dim=meta["hidden_dim"],
        num_heads=meta["num_heads"],
        num_st_blocks=meta["num_st_blocks"],
        num_gtcn_layers=meta["num_gtcn_layers"],
        kernel_size=meta["kernel_size"],
        pred_horizon=meta["pred_horizon"],
        node_feat_dim=meta.get("node_feat_dim", int(node_feat.shape[-1])),
    ).to(device)

    pred_ckpt = Path(args.predictor_ckpt)
    if pred_ckpt.exists():
        _load_predictor_checkpoint(
            predictor,
            pred_ckpt,
            device=device,
            num_nodes=N,
            num_edges=nE,
        )
        print(f"Predictor 載入: {pred_ckpt}")
    else:
        print(f"警告: {pred_ckpt} 不存在，使用隨機權重")

    # ── 載入 router ──
    max_nb = infer_max_neighbors(
        routing_subgraph["edge_index"], routing_N, minimum=args.max_neighbors
    )
    env_state_dim = 2 + 2 * max_nb + 1
    router = DoubleDQNAgent(
        num_nodes=routing_N,
        max_neighbors=max_nb,
        state_dim=env_state_dim,
        device=device,
    )
    print(f"Resolved router max_neighbors={max_nb}")
    if args.routing_graph_mode == "superzone":
        print(
            f"Routing graph: K={routing_N} superzones, top-k action graph from {superzone_artifacts['root']}"
        )
    elif args.routing_locationid_max > 0:
        print(
            f"Routing subgraph: {routing_N} zones with LocationID <= {args.routing_locationid_max}"
        )
    else:
        print(f"Routing subgraph: full graph ({routing_N} zones)")
    router_ckpt = Path(args.router_ckpt)
    if router_ckpt.exists():
        router.load(router_ckpt)
        print(f"Router 載入: {router_ckpt}")
    else:
        print(f"警告: {router_ckpt} 不存在，使用隨機 router")

    # ── 建構 pipeline ──
    pipeline = STDRPipeline(
        predictor=predictor,
        router=router,
        edge_index=edge_index,
        edge_lengths=edge_lengths,
        adj=adj,
        normalization_stats=normalization_stats,
        routing_full_node_indices=routing_node_indices,
        routing_graph_mode=args.routing_graph_mode,
        superzone_artifacts=superzone_artifacts,
        superzone_data_dir=args.data_dir,
        superzone_dir=args.superzone_dir or None,
        max_neighbors=max_nb,
        time_slot_duration_hours=float(args.time_slot_minutes) / 60.0,
        device=device,
    )

    # ── 取一段歷史資料做 demo ──
    # 需 t_start + h <= T，故 t_start <= T - h
    t_start = min(100, max(0, node_feat.shape[0] - h))
    node_seq = node_feat[t_start: t_start + h].transpose(1, 0, 2)  # (N, h, 2)
    speed_seq = edge_speeds[t_start: t_start + h].T                 # (|E|, h)

    print("\n" + "=" * 60)
    print("           STDR Pipeline — 端到端 Demo")
    print("=" * 60)

    # ── Step 1: 預測 ──
    d_pred, c_pred, v_pred = pipeline.predict(node_seq, speed_seq)

    print(f"\n【Step 1 — STGAT 預測結果】(下一步)")
    print(f"  需求範圍: {d_pred[:, 0].min():.2f} ~ {d_pred[:, 0].max():.2f}")
    print(f"  空車範圍: {c_pred[:, 0].min():.2f} ~ {c_pred[:, 0].max():.2f}")
    print(f"  速度範圍: {v_pred[:, 0].min():.2f} ~ {v_pred[:, 0].max():.2f} km/h")
    gap = d_pred[:, 0] - c_pred[:, 0]
    deficit_zones = np.where(gap > 0.5)[0]
    surplus_zones = np.where(gap < -0.5)[0]
    print(f"  缺車區域 (Δ>0.5): {deficit_zones.tolist()}")
    print(f"  盈餘區域 (Δ<-0.5): {surplus_zones.tolist()}")

    # ── Step 2: 派遣 ──
    M, pairs = pipeline.dispatch(d_pred, c_pred, v_pred)

    print(f"\n【Step 2 — 派遣決策】")
    total_dispatched = sum(c for _, _, c in pairs)
    print(f"  派遣 OD 組數: {len(pairs)}, 總派遣車輛: {total_dispatched}")
    print(f"  派遣與路由共用子圖規模: {pipeline.routing_num_nodes} 區")
    if pipeline.last_dispatch_reachability_report is not None:
        report = pipeline.last_dispatch_reachability_report
        print(
            "  Dispatch OD finite-cost rate: "
            f"{report['reachable_pair_rate'] * 100:.2f}% pairs, "
            f"{report['reachable_vehicle_rate'] * 100:.2f}% vehicles"
        )
    for o, d, c in pairs[:10]:
        print(f"    Zone {o} -> Zone {d} : {c} 輛")

    # 若派遣為空，補充隨機 OD pair 做路由展示
    if not pairs:
        print("  (供需平衡，無需派遣；自動生成示範 OD pair)")
        rng = np.random.RandomState(99)
        demo_pairs = []
        while len(demo_pairs) < 5:
            o = int(rng.choice(routing_node_indices))
            d = int(rng.choice(routing_node_indices))
            if o != d:
                demo_pairs.append((o, d, 1))
        pairs = demo_pairs

    # ── Step 3: 路徑規劃 ──
    routes = pipeline.route(pairs, v_pred)

    print(f"\n【Step 3 — Double DQN 路徑規劃】")
    skipped = sum(1 for r in routes if r.get("skipped"))
    routed = [r for r in routes if not r.get("skipped")]
    success = sum(1 for r in routed if r["reached"])
    print(f"  可實際規劃的 OD 組數: {len(routed)}")
    print(f"  因不在 RL 子圖而跳過: {skipped}")
    print(f"  成功到達: {success}/{len(routed)}")
    if routes:
        reached = [r for r in routed if r["reached"]]
        if reached:
            avg_time = np.mean([r["travel_time"] for r in reached])
            avg_dist = np.mean([r["travel_dist"] for r in reached])
            print(f"  平均行駛時間: {avg_time:.4f}")
            print(f"  平均行駛距離: {avg_dist:.4f}")

    print(f"\n路徑範例：")
    for r in routes[:8]:
        if r.get("skipped"):
            tag = "SKIP"
        else:
            tag = "OK" if r["reached"] else "FAIL"
        print(
            f"  [{tag}] Zone {r['origin']}->{r['destination']} "
            f"path={r['path']}  time={r['travel_time']:.4f}  dist={r['travel_dist']:.4f}"
        )
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STDR Pipeline Demo")
    p.add_argument("--predictor-ckpt", type=str, default="runs/stgat_best.pt")
    p.add_argument("--router-ckpt", type=str, default="runs/ddqn_final.pt")
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--time-slot-minutes", type=int, default=15)
    p.add_argument("--max-time-steps", type=int, default=0, help="截斷時間步，0=不截斷")
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
        help="Fixed action-space lower bound; runtime uses max(this, graph max out-degree)",
    )
    p.add_argument(
        "--routing-locationid-max",
        type=int,
        default=63,
        help="Use only zones with LocationID <= this value for RL routing; 0 uses the full graph",
    )
    p.add_argument(
        "--routing-graph-mode",
        type=str,
        default="superzone",
        choices=["legacy", "superzone"],
        help="Use the legacy induced subgraph or the K=64 superzone dispatch/RL graph.",
    )
    p.add_argument(
        "--superzone-dir",
        type=str,
        default="",
        help="Directory produced by build_superzone_graph.py; defaults to data/superzones_k64.",
    )
    return p.parse_args()


if __name__ == "__main__":
    demo(parse_args())
```

### train_ddqn.py

```python
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
    device = resolve_device(args.device)
    configure_cuda_runtime(device)

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
```

### evaluate_ddqn.py

```python
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

from data_loader import load_nyc_graph_for_rl

from ddqn_agent import DoubleDQNAgent
from graph_env import CityGraph, RoutingEnv, create_graph_from_data, infer_max_neighbors


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
        device=args.device,
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
```

### ddqn_agent.py

```python
"""
ddqn_agent.py — Double DQN Agent

核心演算法：
  y = r + γ · Q_target(s', argmax_a Q_online(s', a))

包含：
  - epsilon-greedy 動作選擇（含 action masking）
  - replay buffer 取樣訓練
  - target network 定期同步
  - 模型存取
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from q_network import QNetwork
from replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    """
    Parameters
    ----------
    num_nodes       : 圖中節點總數
    max_neighbors   : 最大鄰居數 = 動作空間大小
    state_dim       : 狀態向量維度
    embed_dim       : 節點 Embedding 維度
    hidden_dim      : MLP 隱藏層寬度
    lr              : 學習率
    gamma           : 折扣因子
    epsilon_start   : 初始探索率
    epsilon_end     : 最低探索率
    epsilon_decay   : 線性衰減步數
    buffer_capacity : replay buffer 容量
    batch_size      : 訓練批次大小
    target_update   : 每隔幾次 learn() 同步 target network
    device          : 'cpu' 或 'cuda'
    """

    def __init__(
        self,
        num_nodes: int,
        max_neighbors: int,
        state_dim: int,
        *,
        embed_dim: int = 32,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 5000,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update: int = 200,
        device: str = "cpu",
    ) -> None:
        self.num_nodes = num_nodes
        self.max_neighbors = max_neighbors
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device(device)

        # ε-greedy 參數
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # 網路
        self.online_net = QNetwork(
            num_nodes, max_neighbors, embed_dim, hidden_dim
        ).to(self.device)
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

        self.learn_step_count: int = 0

    # ── 動作選擇 ──────────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        *,
        greedy: bool = False,
    ) -> int:
        """
        ε-greedy action selection with action masking.
        greedy=True 時使用純貪婪策略（評估用）。
        """
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            return 0  # fallback，環境會判定非法

        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            s = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q = self.online_net(s).squeeze(0)
            # 非法動作的 Q-value 設為 -inf
            mask_t = torch.tensor(
                action_mask, dtype=torch.float32, device=self.device
            )
            q[mask_t == 0] = -float("inf")
            return int(q.argmax().item())

    # ── 訓練 ──────────────────────────────────────────────────

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        self.buffer.push(
            state, action, reward, next_state, done, action_mask, next_action_mask
        )

    def learn(self) -> Optional[float]:
        """
        從 replay buffer 取樣並執行一步 Double DQN 更新。
        回傳 loss 值；若 buffer 不夠大則回傳 None。
        """
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size, self.device)

        # ── online Q(s, a) ──
        q_online = self.online_net(batch.states)                    # (B, M)
        q_sa = q_online.gather(1, batch.actions)                    # (B, 1)

        # ── Double DQN target ──
        with torch.no_grad():
            # 用 online net 選動作
            q_next_online = self.online_net(batch.next_states)      # (B, M)
            # 非法動作 mask 為 -inf
            q_next_online[batch.next_action_masks == 0] = -float("inf")
            best_actions = q_next_online.argmax(dim=1, keepdim=True)  # (B, 1)

            # 用 target net 估計 Q 值
            q_next_target = self.target_net(batch.next_states)      # (B, M)
            q_next_val = q_next_target.gather(1, best_actions)      # (B, 1)

            target = batch.rewards + self.gamma * q_next_val * (1.0 - batch.dones)

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止爆炸
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ε 衰減（線性）
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end)
            * self.learn_step_count
            / self.epsilon_decay,
        )

        self.learn_step_count += 1

        # 定期同步 target network
        if self.learn_step_count % self.target_update == 0:
            self.sync_target()

        return loss.item()

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── 模型持久化 ────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "learn_step_count": self.learn_step_count,
                "num_nodes": self.num_nodes,
                "max_neighbors": self.max_neighbors,
                "state_dim": self.state_dim,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self._validate_checkpoint_compatibility(ckpt)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = ckpt["epsilon"]
        self.learn_step_count = ckpt["learn_step_count"]

    def _validate_checkpoint_compatibility(self, ckpt: dict) -> None:
        expected_num_nodes = int(
            ckpt.get(
                "num_nodes",
                ckpt["online_state_dict"]["embed.weight"].shape[0],
            )
        )
        expected_max_neighbors = int(
            ckpt.get(
                "max_neighbors",
                ckpt["online_state_dict"]["net.4.bias"].shape[0],
            )
        )
        expected_state_dim = int(
            ckpt.get(
                "state_dim",
                2 * expected_max_neighbors + 3,
            )
        )

        mismatches: list[str] = []
        if expected_num_nodes != self.num_nodes:
            mismatches.append(
                f"num_nodes checkpoint={expected_num_nodes}, current={self.num_nodes}"
            )
        if expected_max_neighbors != self.max_neighbors:
            mismatches.append(
                "max_neighbors "
                f"checkpoint={expected_max_neighbors}, current={self.max_neighbors}"
            )
        if expected_state_dim != self.state_dim:
            mismatches.append(
                f"state_dim checkpoint={expected_state_dim}, current={self.state_dim}"
            )

        if mismatches:
            mismatch_text = "; ".join(mismatches)
            raise ValueError(
                "Checkpoint is incompatible with the current DoubleDQNAgent "
                f"configuration: {mismatch_text}. "
                "Use a checkpoint trained on the same graph/action-space setup."
            )
```

### graph_env.py

```python
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
        if action >= len(nids) or mask[action] == 0.0:
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
```

### q_network.py

```python
"""
q_network.py — Q 網路（MLP + 節點 Embedding）

網路結構：
  1. 將 current_node 與 destination 各自通過 Embedding 層取得向量
  2. 與連續特徵（鄰居預測速度、鄰居邊長、時間槽）拼接
  3. 經過多層 MLP
  4. 輸出 max_neighbors 維 Q-values
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Parameters
    ----------
    num_nodes      : 圖中節點總數（Embedding 詞表大小）
    max_neighbors  : 最大鄰居數 = 動作空間大小
    embed_dim      : 節點 Embedding 維度
    hidden_dim     : 隱藏層寬度
    """

    def __init__(
        self,
        num_nodes: int,
        max_neighbors: int,
        embed_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.max_neighbors = max_neighbors
        self.embed = nn.Embedding(num_nodes, embed_dim)

        # 連續特徵維度：pred_speeds(M) + lengths(M) + time_slot(1)
        cont_dim = 2 * max_neighbors + 1
        input_dim = 2 * embed_dim + cont_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_neighbors),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : (B, state_dim)  其中 state[:, 0] = current_node,
                state[:, 1] = destination, state[:, 2:] = 連續特徵

        Returns
        -------
        q_values : (B, max_neighbors)
        """
        current_node = state[:, 0].long()
        destination = state[:, 1].long()
        cont_features = state[:, 2:]

        curr_emb = self.embed(current_node)   # (B, embed_dim)
        dest_emb = self.embed(destination)     # (B, embed_dim)

        x = torch.cat([curr_emb, dest_emb, cont_features], dim=1)
        return self.net(x)
```

### replay_buffer.py

```python
"""
replay_buffer.py — 經驗回放緩衝區

儲存 (state, action, reward, next_state, done, action_mask, next_action_mask)
並支援均勻隨機取樣供 Double DQN 訓練使用。
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, NamedTuple

import numpy as np
import torch


class Transition(NamedTuple):
    state: np.ndarray            # (state_dim,)
    action: int
    reward: float
    next_state: np.ndarray       # (state_dim,)
    done: bool
    action_mask: np.ndarray      # (max_neighbors,)
    next_action_mask: np.ndarray # (max_neighbors,)


class BatchTensors(NamedTuple):
    """從 replay buffer 取樣後轉為 GPU-ready tensors"""
    states: torch.Tensor            # (B, state_dim)
    actions: torch.Tensor           # (B, 1)  long
    rewards: torch.Tensor           # (B, 1)
    next_states: torch.Tensor       # (B, state_dim)
    dones: torch.Tensor             # (B, 1)
    action_masks: torch.Tensor      # (B, max_neighbors)
    next_action_masks: torch.Tensor # (B, max_neighbors)


class ReplayBuffer:
    """固定容量的經驗回放緩衝區，使用 deque 自動淘汰最舊資料。"""

    def __init__(self, capacity: int = 50_000) -> None:
        self.buffer: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
        next_action_mask: np.ndarray,
    ) -> None:
        self.buffer.append(
            Transition(state, action, reward, next_state, done, action_mask, next_action_mask)
        )

    def sample(self, batch_size: int, device: torch.device) -> BatchTensors:
        batch: List[Transition] = random.sample(self.buffer, batch_size)

        states = np.stack([t.state for t in batch])
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones = np.array([t.done for t in batch], dtype=np.float32)
        action_masks = np.stack([t.action_mask for t in batch])
        next_action_masks = np.stack([t.next_action_mask for t in batch])

        return BatchTensors(
            states=torch.tensor(states, dtype=torch.float32, device=device),
            actions=torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1),
            rewards=torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            next_states=torch.tensor(next_states, dtype=torch.float32, device=device),
            dones=torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
            action_masks=torch.tensor(action_masks, dtype=torch.float32, device=device),
            next_action_masks=torch.tensor(next_action_masks, dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.buffer)
```

### data_loader.py

```python
"""
data_loader.py — 資料載入與 PyTorch Dataset

提供：
  - load_nyc_real_graph_features() — 263 Zone 真實鄰接 + OSRM 邊長 + build_speed_features 產物
  - load_nyc_graph_for_rl()        — 僅路網與邊速統計，供 Double DQN 訓練
  - load_nyc_taxi_data()           — 舊版 CSV 流程（較少使用）

Dataset 以滑動窗口方式產生 (input, target) 組，供 STGAT 訓練使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from superzone_graph import aggregate_speed_profile_to_rl_edges, load_superzone_artifacts


def edge_index_from_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    與 build_speed_features.build_edge_speeds 相同順序：列優先掃描 adj[i,j]>0。
    """
    adj = np.asarray(adj)
    edges: list[tuple[int, int]] = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                edges.append((i, j))
    return np.array(edges, dtype=np.int32)


def _load_nyc_adj_edge_lengths(
    root: Path,
    edge_length_source: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """載入 adjacency_matrix、edge_index、依邊展開的邊長 (km)。"""
    adj_path = root / "adjacency_matrix.npy"
    if not adj_path.exists():
        raise FileNotFoundError(f"找不到 {adj_path}，請先執行 build_adjacency.py")

    adj = np.load(adj_path).astype(np.float32)
    n = adj.shape[0]

    ei_path = root / "edge_index.npy"
    if ei_path.exists():
        edge_index = np.load(ei_path).astype(np.int32)
    else:
        edge_index = edge_index_from_adjacency(adj)

    if edge_length_source == "osrm":
        osrm_path = root / "edge_lengths_osrm.npy"
        cen_path = root / "edge_lengths.npy"
        if osrm_path.exists():
            len_mat = np.load(osrm_path).astype(np.float32)
        elif cen_path.exists():
            len_mat = np.load(cen_path).astype(np.float32)
        else:
            raise FileNotFoundError(
                f"需要 {osrm_path} 或 {cen_path}，請執行 update_edge_lengths_osrm.py 或 build_adjacency.py"
            )
    else:
        cen_path = root / "edge_lengths.npy"
        if not cen_path.exists():
            raise FileNotFoundError(f"找不到 {cen_path}")
        len_mat = np.load(cen_path).astype(np.float32)

    n_e = edge_index.shape[0]
    edge_lengths = np.zeros(n_e, dtype=np.float32)
    for e in range(n_e):
        i, j = int(edge_index[e, 0]), int(edge_index[e, 1])
        if i < n and j < n:
            edge_lengths[e] = float(len_mat[i, j])

    return adj, edge_index, edge_lengths


def _as_edge_slot_matrix(
    arr: np.ndarray,
    n_e: int,
    *,
    name: str,
) -> np.ndarray:
    """
    Normalize a speed array to shape (num_edges, num_time_slots).
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.shape[0] != n_e:
            raise ValueError(f"{name} 長度 {arr.shape[0]} 與邊數 {n_e} 不一致")
        return arr.reshape(n_e, 1).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} 應為一維或二維，得到 {arr.shape}")
    if arr.shape[0] == n_e:
        return arr.astype(np.float32)
    if arr.shape[1] == n_e:
        return arr.T.astype(np.float32)
    raise ValueError(f"無法對齊邊數 n_e={n_e} 與 {name} 形狀 {arr.shape}")


def _resolve_default_shapefile() -> Path:
    for path in Path("NYCtaxizone").glob("*.shp"):
        return path
    raise FileNotFoundError("Could not find an NYC Taxi Zone shapefile under NYCtaxizone/")


def _build_temporal_features(
    root: Path,
    num_time_steps: int,
) -> Tuple[np.ndarray, list[str]]:
    """
    Build cyclical month / weekday / slot features from time_meta.csv.

    Returns
    -------
    temporal_features : (T, 6)
        [month_sin, month_cos, weekday_sin, weekday_cos, slot_sin, slot_cos]
    feature_names : list[str]
    """
    time_meta_path = root / "time_meta.csv"
    if not time_meta_path.exists():
        raise FileNotFoundError(f"找不到 {time_meta_path}，請先執行 build_speed_features.py")

    time_meta = pd.read_csv(time_meta_path)
    if len(time_meta) < num_time_steps:
        raise ValueError(
            f"time_meta 行數 {len(time_meta)} 少於時間步 {num_time_steps}"
        )
    time_meta = time_meta.iloc[:num_time_steps].copy()
    if "date" not in time_meta.columns or "slot" not in time_meta.columns:
        raise ValueError("time_meta.csv 至少需要 date 與 slot 欄位")

    dates = pd.to_datetime(time_meta["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("time_meta.csv 的 date 欄位存在無法解析的值")

    slot = pd.to_numeric(time_meta["slot"], errors="coerce")
    if slot.isna().any():
        raise ValueError("time_meta.csv 的 slot 欄位存在無法解析的值")

    if "day_of_week" in time_meta.columns:
        weekday = pd.to_numeric(time_meta["day_of_week"], errors="coerce")
    else:
        weekday = dates.dt.dayofweek.astype(np.float32)
    if weekday.isna().any():
        raise ValueError("time_meta.csv 的 day_of_week 欄位存在無法解析的值")

    month = dates.dt.month.astype(np.float32)
    weekday = weekday.astype(np.float32)
    slot = slot.astype(np.float32)
    slots_per_day = float(slot.max()) + 1.0
    if slots_per_day <= 0:
        raise ValueError("time_meta.csv 的 slot 欄位無法推導有效的每日時間槽數")

    month_angle = 2.0 * np.pi * (month - 1.0) / 12.0
    weekday_angle = 2.0 * np.pi * weekday / 7.0
    slot_angle = 2.0 * np.pi * slot / slots_per_day

    features = np.stack(
        [
            np.sin(month_angle),
            np.cos(month_angle),
            np.sin(weekday_angle),
            np.cos(weekday_angle),
            np.sin(slot_angle),
            np.cos(slot_angle),
        ],
        axis=1,
    ).astype(np.float32)
    feature_names = [
        "month_sin",
        "month_cos",
        "weekday_sin",
        "weekday_cos",
        "slot_sin",
        "slot_cos",
    ]
    return features, feature_names


def load_zone_metadata(
    data_dir: str | Path = "data",
    *,
    shapefile: str | Path | None = None,
) -> "pd.DataFrame":
    """
    Load zone metadata aligned with the graph node order.

    `zone_info.csv` generated by older runs may not contain `locationid`, so this
    helper backfills it from the shapefile while preserving row order.
    """
    import pandas as pd

    root = Path(data_dir)
    zone_info_path = root / "zone_info.csv"
    if not zone_info_path.exists():
        raise FileNotFoundError(f"Could not find {zone_info_path}")

    zone_info = pd.read_csv(zone_info_path)
    zone_info.columns = [str(col).strip().lower() for col in zone_info.columns]

    if "index" not in zone_info.columns:
        zone_info.insert(0, "index", np.arange(len(zone_info), dtype=np.int32))

    if "locationid" in zone_info.columns:
        zone_info["locationid"] = zone_info["locationid"].astype("Int64")
        return zone_info

    shp_path = Path(shapefile) if shapefile is not None else _resolve_default_shapefile()
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    gdf.columns = [str(col).strip().lower() for col in gdf.columns]
    if "locationid" not in gdf.columns:
        raise ValueError(f"Shapefile {shp_path} does not contain a locationid column")
    if len(gdf) != len(zone_info):
        raise ValueError(
            f"zone_info rows ({len(zone_info)}) do not match shapefile rows ({len(gdf)})"
        )

    zone_info = zone_info.copy()
    zone_info["locationid"] = gdf["locationid"].astype("Int64").values
    if "zone_name" not in zone_info.columns and "zone" in gdf.columns:
        zone_info["zone_name"] = gdf["zone"].values
    if "borough" not in zone_info.columns and "borough" in gdf.columns:
        zone_info["borough"] = gdf["borough"].values
    return zone_info


def select_zone_indices_by_locationid_max(
    zone_info: "pd.DataFrame",
    locationid_max: int,
) -> np.ndarray:
    """Select node indices whose Taxi Zone LocationID is <= `locationid_max`."""
    if locationid_max <= 0:
        return zone_info["index"].to_numpy(dtype=np.int32)

    mask = zone_info["locationid"].astype("Int64") <= int(locationid_max)
    selected = zone_info.loc[mask, "index"].to_numpy(dtype=np.int32)
    if selected.size == 0:
        raise ValueError(
            f"No zones matched LocationID <= {locationid_max}; check zone metadata."
        )
    return selected


def build_induced_subgraph(
    num_nodes: int,
    edge_index: np.ndarray,
    edge_lengths: np.ndarray,
    selected_full_nodes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Build an induced directed subgraph from a full graph.

    Returns the subgraph edge list, edge lengths, adjacency, and node/edge mappings.
    """
    selected = np.asarray(selected_full_nodes, dtype=np.int32).reshape(-1)
    if selected.size == 0:
        raise ValueError("selected_full_nodes must not be empty")

    full_to_sub = np.full(num_nodes, -1, dtype=np.int32)
    full_to_sub[selected] = np.arange(selected.size, dtype=np.int32)

    src = edge_index[:, 0].astype(np.int32)
    dst = edge_index[:, 1].astype(np.int32)
    edge_mask = (full_to_sub[src] >= 0) & (full_to_sub[dst] >= 0)
    selected_edge_idx = np.flatnonzero(edge_mask).astype(np.int32)

    sub_edge_index = np.stack(
        [full_to_sub[src[edge_mask]], full_to_sub[dst[edge_mask]]],
        axis=1,
    ).astype(np.int32)
    sub_edge_lengths = edge_lengths[edge_mask].astype(np.float32)

    sub_num_nodes = int(selected.size)
    sub_adj = np.zeros((sub_num_nodes, sub_num_nodes), dtype=np.float32)
    if sub_edge_index.size > 0:
        sub_adj[sub_edge_index[:, 0], sub_edge_index[:, 1]] = 1.0

    return {
        "full_node_indices": selected,
        "full_to_sub": full_to_sub,
        "edge_mask": edge_mask,
        "full_edge_indices": selected_edge_idx,
        "adj": sub_adj,
        "edge_index": sub_edge_index,
        "edge_lengths": sub_edge_lengths,
    }


def load_nyc_real_graph_features(
    data_dir: str | Path = "data",
    *,
    max_time_steps: int = 0,
    edge_length_source: str = "osrm",
    add_time_features: bool = False,
) -> Dict[str, np.ndarray]:
    """
    載入紐約 263 Taxi Zone 真實路網與 `build_speed_features.py` 產生的時序特徵。

    必要檔案：
      - adjacency_matrix.npy
      - node_demand.npy, node_supply.npy  — (T, N)
      - edge_speeds.npy                   — (T, |E|)，與 edge_index 邊序一致
      - edge_index.npy（強烈建議；若缺則由 adj 重建）

    邊長（km，與速度 km/h 一致）：
      - edge_length_source=\"osrm\"  → edge_lengths_osrm.npy（優先），否則 edge_lengths.npy
      - edge_length_source=\"centroid\" → edge_lengths.npy

    Parameters
    ----------
    max_time_steps
        >0 時只取前 T 個時間步（除錯或減輕顯存）；0 表示使用全部。
    """
    root = Path(data_dir)
    adj, edge_index, edge_lengths = _load_nyc_adj_edge_lengths(root, edge_length_source)
    n = adj.shape[0]
    n_e = edge_index.shape[0]

    dem_path = root / "node_demand.npy"
    sup_path = root / "node_supply.npy"
    spd_path = root / "edge_speeds.npy"
    for p in (dem_path, sup_path, spd_path):
        if not p.exists():
            raise FileNotFoundError(f"找不到 {p}，請先執行 build_speed_features.py")

    demand = np.load(dem_path).astype(np.float32)
    supply = np.load(sup_path).astype(np.float32)
    edge_speeds = np.load(spd_path).astype(np.float32)

    if demand.shape != supply.shape:
        raise ValueError(f"node_demand {demand.shape} 與 node_supply {supply.shape} 不一致")

    if edge_speeds.ndim != 2:
        raise ValueError(f"edge_speeds 應為二維，得到 {edge_speeds.shape}")

    # 標準為 (T, |E|)；若存成 (|E|, T) 則轉置
    if edge_speeds.shape[1] == n_e:
        pass
    elif edge_speeds.shape[0] == n_e:
        edge_speeds = edge_speeds.T
    else:
        raise ValueError(
            f"無法對齊邊數 n_e={n_e} 與 edge_speeds 形狀 {edge_speeds.shape}"
        )

    t_feat = demand.shape[0]
    if edge_speeds.shape[0] != t_feat:
        raise ValueError(
            f"時間步不一致: demand T={t_feat}, edge_speeds {edge_speeds.shape} "
            f"（預期 (T, |E|) 且 T={t_feat}）"
        )
    if edge_speeds.shape[1] != n_e:
        raise ValueError(
            f"邊數不一致: edge_index 有 {n_e} 條邊, edge_speeds 第二維為 {edge_speeds.shape[1]}"
        )

    if demand.shape[1] != n or supply.shape[1] != n:
        raise ValueError(f"節點數應為 {n}，得到 demand {demand.shape}")

    if max_time_steps > 0:
        t_use = min(t_feat, max_time_steps)
        demand = demand[:t_use]
        supply = supply[:t_use]
        edge_speeds = edge_speeds[:t_use]
    else:
        t_use = t_feat
    base_node_features = np.stack([demand, supply], axis=-1)
    time_feature_names: list[str] = []
    if add_time_features:
        temporal_features, time_feature_names = _build_temporal_features(root, t_use)
        temporal_node_features = np.broadcast_to(
            temporal_features[:, None, :],
            (t_use, n, temporal_features.shape[1]),
        )
        node_features = np.concatenate(
            [base_node_features, temporal_node_features],
            axis=-1,
        ).astype(np.float32)
    else:
        node_features = base_node_features.astype(np.float32)

    return {
        "adj": adj,
        "edge_index": edge_index,
        "edge_lengths": edge_lengths,
        "node_features": node_features,
        "edge_speeds": edge_speeds,
        "time_feature_names": time_feature_names,
        "source": "nyc_real",
    }


def load_nyc_graph_for_rl(
    data_dir: str | Path = "data",
    *,
    edge_length_source: str = "osrm",
    speed_seed: int = 42,
    routing_locationid_max: int = 63,
    routing_graph_mode: str = "superzone",
    superzone_dir: str | Path | None = None,
    shapefile: str | Path | None = None,
) -> Dict[str, np.ndarray]:
    """
    載入 Double DQN 用路網：不需 node_demand / node_supply，但需邊速檔。

    優先使用 ``edge_speeds_avg.npy`` 作為跨日平均時槽速度序列；若無則退回
    ``edge_speeds.npy``。回傳每邊的動態預測速度序列，以及由其微擾動得到的
    動態「真實」速度序列，供 RL 訓練 / 評估使用。
    """
    root = Path(data_dir)
    adj, edge_index, edge_lengths = _load_nyc_adj_edge_lengths(root, edge_length_source)
    n_e = edge_index.shape[0]

    spd_path = root / "edge_speeds.npy"
    avg_path = root / "edge_speeds_avg.npy"

    profile_source = ""
    if avg_path.exists():
        pred_speed_profile = _as_edge_slot_matrix(
            np.load(avg_path),
            n_e,
            name="edge_speeds_avg",
        )
        profile_source = "edge_speeds_avg"
    elif spd_path.exists():
        pred_speed_profile = _as_edge_slot_matrix(
            np.load(spd_path, mmap_mode="r"),
            n_e,
            name="edge_speeds",
        )
        profile_source = "edge_speeds"
    else:
        raise FileNotFoundError(
            f"需要 {spd_path} 或 {avg_path}（請先執行 build_speed_features.py）"
        )

    pred_speed_profile = np.maximum(pred_speed_profile.astype(np.float32), 1.0)
    avg_speeds = pred_speed_profile.mean(axis=1).astype(np.float32)

    if avg_speeds.shape[0] != n_e:
        raise ValueError(
            f"邊速長度 {avg_speeds.shape[0]} 與 edge_index 邊數 {n_e} 不一致"
        )

    rng = np.random.RandomState(speed_seed)
    real_speed_profile = (
        pred_speed_profile * rng.uniform(0.7, 1.3, size=pred_speed_profile.shape)
    ).astype(np.float32)
    real_speeds = real_speed_profile.mean(axis=1).astype(np.float32)
    num_time_slots = int(pred_speed_profile.shape[1])
    if profile_source == "edge_speeds_avg" and num_time_slots > 0 and 1440 % num_time_slots == 0:
        time_slot_minutes = int(1440 // num_time_slots)
    else:
        time_slot_minutes = 15

    if routing_graph_mode == "superzone":
        artifacts = load_superzone_artifacts(data_dir, superzone_dir)
        edge_index = artifacts["rl_edge_index"].astype(np.int32)
        edge_lengths = artifacts["rl_edge_lengths"].astype(np.float32)
        base_speeds = np.maximum(artifacts["rl_edge_speeds_kmh"].astype(np.float32), 1.0)
        pred_speed_profile = aggregate_speed_profile_to_rl_edges(
            pred_speed_profile,
            artifacts["rl_edge_speed_mapping_offsets"],
            artifacts["rl_edge_speed_mapping_indices"],
            base_speeds,
        )
        real_speed_profile = aggregate_speed_profile_to_rl_edges(
            real_speed_profile,
            artifacts["rl_edge_speed_mapping_offsets"],
            artifacts["rl_edge_speed_mapping_indices"],
            base_speeds,
        )
        k_regions = int(artifacts["meta"].get("num_superzones", artifacts["region_demand"].shape[1]))
        adj = np.zeros((k_regions, k_regions), dtype=np.float32)
        if edge_index.size > 0:
            adj[edge_index[:, 0], edge_index[:, 1]] = 1.0
        full_nodes = np.arange(k_regions, dtype=np.int32)
        region_info = artifacts["region_info"].copy()
        region_info["index"] = region_info["region_id"].astype(np.int32)
        region_info["locationid"] = region_info["region_id"].astype(np.int32) + 1
        region_info["zone_name"] = region_info["member_zone_names"].astype(str)
        region_info["sub_index"] = region_info["region_id"].astype(np.int32)
        return {
            "adj": adj,
            "edge_index": edge_index,
            "edge_lengths": edge_lengths,
            "avg_speeds": base_speeds.astype(np.float32),
            "real_speeds": real_speed_profile.mean(axis=1).astype(np.float32),
            "pred_speed_profile": pred_speed_profile,
            "real_speed_profile": real_speed_profile,
            "num_time_slots": num_time_slots,
            "time_slot_minutes": int(time_slot_minutes),
            "full_node_indices": full_nodes,
            "full_to_sub": full_nodes,
            "full_edge_indices": np.arange(edge_index.shape[0], dtype=np.int32),
            "zone_info": region_info,
            "routing_locationid_max": int(routing_locationid_max),
            "routing_graph_mode": "superzone",
            "superzone_dir": str(artifacts["root"]),
            "region_membership": artifacts["membership"],
            "region_node_demand": artifacts["region_demand"],
            "region_node_supply": artifacts["region_supply"],
            "dispatch_duration_hours": artifacts["dispatch_duration_hours"],
            "dispatch_distance_km": artifacts["dispatch_distance_km"],
            "dispatch_reachable": artifacts["dispatch_reachable"],
            "superzone_meta": artifacts["meta"],
        }
    if routing_graph_mode != "legacy":
        raise ValueError(
            f"Unsupported routing_graph_mode={routing_graph_mode!r}; expected 'legacy' or 'superzone'."
        )

    if routing_locationid_max > 0:
        zone_info = load_zone_metadata(data_dir, shapefile=shapefile)
        selected_nodes = select_zone_indices_by_locationid_max(zone_info, routing_locationid_max)
    else:
        import pandas as pd

        selected_nodes = np.arange(adj.shape[0], dtype=np.int32)
        zone_info = pd.DataFrame(
            {
                "index": selected_nodes,
                "locationid": selected_nodes + 1,
                "sub_index": selected_nodes,
            }
        )
    subgraph = build_induced_subgraph(adj.shape[0], edge_index, edge_lengths, selected_nodes)
    sub_edge_idx = subgraph["full_edge_indices"]

    selected_zone_info = zone_info.loc[
        zone_info["index"].isin(selected_nodes)
    ].copy()
    selected_zone_info["sub_index"] = selected_zone_info["index"].map(
        {int(full_idx): i for i, full_idx in enumerate(selected_nodes.tolist())}
    )
    selected_zone_info = selected_zone_info.sort_values("sub_index").reset_index(drop=True)

    return {
        "adj": subgraph["adj"],
        "edge_index": subgraph["edge_index"],
        "edge_lengths": subgraph["edge_lengths"],
        "avg_speeds": avg_speeds[sub_edge_idx].astype(np.float32),
        "real_speeds": real_speeds[sub_edge_idx].astype(np.float32),
        "pred_speed_profile": pred_speed_profile[sub_edge_idx].astype(np.float32),
        "real_speed_profile": real_speed_profile[sub_edge_idx].astype(np.float32),
        "num_time_slots": num_time_slots,
        "time_slot_minutes": int(time_slot_minutes),
        "full_node_indices": subgraph["full_node_indices"],
        "full_to_sub": subgraph["full_to_sub"],
        "full_edge_indices": subgraph["full_edge_indices"],
        "zone_info": selected_zone_info,
        "routing_locationid_max": int(routing_locationid_max),
    }


# ════════════════════════════════════════════════════════════════
#  NYC Taxi 資料載入（可選）
# ════════════════════════════════════════════════════════════════

def load_nyc_taxi_data(
    trip_csv: str,
    num_zones: int = 63,
    time_slot_minutes: int = 15,
    adj_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    從 NYC TLC CSV 載入資料並聚合為時空特徵。

    需要的 CSV 欄位：
      tpep_pickup_datetime, tpep_dropoff_datetime,
      PULocationID, DOLocationID, trip_distance

    Parameters
    ----------
    trip_csv         : CSV 檔路徑
    num_zones        : 使用的區域數（取 LocationID < num_zones）
    time_slot_minutes: 時間槽長度
    adj_path         : 鄰接矩陣 .npy 路徑（若無則自動從行程推導）
    """
    import pandas as pd

    df = pd.read_csv(trip_csv, parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    df = df[(df["PULocationID"] < num_zones) & (df["DOLocationID"] < num_zones)]
    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    df = df[(df["duration_min"] > 1) & (df["duration_min"] < 180)]

    # 時間槽
    df["slot"] = (
        (df["tpep_pickup_datetime"] - df["tpep_pickup_datetime"].min())
        .dt.total_seconds()
        / (time_slot_minutes * 60)
    ).astype(int)
    T = df["slot"].max() + 1
    N = num_zones

    # 需求：每區域每時間槽的上車數
    demand = np.zeros((T, N), dtype=np.float32)
    grp = df.groupby(["slot", "PULocationID"]).size()
    for (t, z), cnt in grp.items():
        if t < T and z < N:
            demand[t, z] = cnt

    # 空車估計：下車數 − 上車數（累積 + 初始值）
    dropoff_cnt = np.zeros((T, N), dtype=np.float32)
    grp2 = df.groupby(["slot", "DOLocationID"]).size()
    for (t, z), cnt in grp2.items():
        if t < T and z < N:
            dropoff_cnt[t, z] = cnt

    supply = np.zeros((T, N), dtype=np.float32)
    supply[0] = 5.0  # 初始每區 5 台空車
    for t in range(1, T):
        supply[t] = np.maximum(supply[t - 1] + dropoff_cnt[t - 1] - demand[t - 1], 0)

    node_features = np.stack([demand, supply], axis=-1)

    # 鄰接矩陣
    if adj_path and Path(adj_path).exists():
        adj = np.load(adj_path).astype(np.float32)
    else:
        adj = np.zeros((N, N), dtype=np.float32)
        pairs = df.groupby(["PULocationID", "DOLocationID"]).size()
        for (i, j), cnt in pairs.items():
            if i < N and j < N and cnt > 5:
                adj[i, j] = 1.0
        np.fill_diagonal(adj, 0)

    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=1)
    nE = edge_index.shape[0]

    # 邊速度與長度
    edge_lengths = np.ones(nE, dtype=np.float32) * 2.0  # 預設 2km
    edge_speeds = np.full((T, nE), 30.0, dtype=np.float32)  # 預設 30km/h

    for idx, (i, j) in enumerate(edge_index):
        sub = df[(df["PULocationID"] == i) & (df["DOLocationID"] == j)]
        if len(sub) > 0:
            mean_dist = sub["trip_distance"].mean() * 1.609  # mile → km
            edge_lengths[idx] = max(mean_dist, 0.1)
        for t in range(T):
            tsub = sub[sub["slot"] == t]
            if len(tsub) >= 2:
                avg_speed = (tsub["trip_distance"].mean() * 1.609) / (
                    tsub["duration_min"].mean() / 60.0 + 1e-5
                )
                edge_speeds[t, idx] = np.clip(avg_speed, 5.0, 130.0)

    return {
        "adj": adj,
        "edge_index": edge_index,
        "edge_lengths": edge_lengths,
        "node_features": node_features,
        "edge_speeds": edge_speeds,
    }


# ════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ════════════════════════════════════════════════════════════════

class SpatioTemporalDataset(Dataset):
    """
    滑動窗口 Dataset。

    每筆樣本：
      input : (node_seq (N, h, C),  speed_seq (|E|, h))
      target: (demand (N, p),  supply (N, p),  speed (|E|, p))
    """

    def __init__(
        self,
        node_features: np.ndarray,    # (T, N, C)
        edge_speeds: np.ndarray,      # (T, |E|)
        hist_len: int = 12,
        pred_horizon: int = 3,
    ) -> None:
        super().__init__()
        self.node_feat = node_features.astype(np.float32)
        self.edge_speed = edge_speeds.astype(np.float32)
        self.h = hist_len
        self.p = pred_horizon
        self.total = node_features.shape[0] - hist_len - pred_horizon + 1

    def __len__(self) -> int:
        return max(self.total, 0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = idx + self.h  # 窗口結束位置

        # input
        node_seq = self.node_feat[idx: t]                # (h, N, C)
        speed_seq = self.edge_speed[idx: t]              # (h, |E|)

        # target
        demand_target = self.node_feat[t: t + self.p, :, 0].T   # (N, p)
        supply_target = self.node_feat[t: t + self.p, :, 1].T   # (N, p)
        speed_target = self.edge_speed[t: t + self.p].T          # (|E|, p)

        return {
            "node_seq": torch.from_numpy(node_seq.transpose(1, 0, 2)),   # (N, h, C)
            "speed_seq": torch.from_numpy(speed_seq.T),                   # (|E|, h)
            "demand_target": torch.from_numpy(demand_target),
            "supply_target": torch.from_numpy(supply_target),
            "speed_target": torch.from_numpy(speed_target),
        }
```

### superzone_graph.py

```python
from __future__ import annotations

import json
import math
import time
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


AIRPORT_NAMES = ("Newark Airport", "JFK Airport", "LaGuardia Airport")


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _resolve_shapefile() -> Path:
    for path in Path("NYCtaxizone").glob("*.shp"):
        return path
    raise FileNotFoundError("Could not find an NYC Taxi Zone shapefile under NYCtaxizone/")


def load_zone_table(data_dir: str | Path = "data", shapefile: str | Path | None = None) -> pd.DataFrame:
    """Load zone metadata and backfill LocationID/name/borough from the shapefile."""
    data_root = _as_path(data_dir)
    zone_info = pd.read_csv(data_root / "zone_info.csv")
    zone_info.columns = [str(c).strip().lower() for c in zone_info.columns]
    if "index" not in zone_info.columns:
        zone_info.insert(0, "index", np.arange(len(zone_info), dtype=np.int32))

    shp_path = _as_path(shapefile) if shapefile is not None else _resolve_shapefile()
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("geopandas is required to build superzone metadata") from exc

    gdf = gpd.read_file(shp_path)
    gdf.columns = [str(c).strip().lower() for c in gdf.columns]
    if len(gdf) != len(zone_info):
        raise ValueError(
            f"zone_info rows ({len(zone_info)}) do not match shapefile rows ({len(gdf)})"
        )
    if "locationid" not in zone_info.columns:
        zone_info["locationid"] = gdf["locationid"].astype(int).values
    if "zone_name" not in zone_info.columns and "zone" in gdf.columns:
        zone_info["zone_name"] = gdf["zone"].values
    if "borough" not in zone_info.columns and "borough" in gdf.columns:
        zone_info["borough"] = gdf["borough"].values

    centroids = gdf.to_crs(epsg=2263).geometry.centroid.to_crs(epsg=4326)
    zone_info["lon"] = [float(p.x) for p in centroids]
    zone_info["lat"] = [float(p.y) for p in centroids]
    zone_info["index"] = zone_info["index"].astype(int)
    zone_info["locationid"] = zone_info["locationid"].astype(int)
    return zone_info


def undirected_neighbors(adj: np.ndarray) -> list[set[int]]:
    n = int(adj.shape[0])
    neighbors = [set(np.flatnonzero(adj[i] > 0).astype(int).tolist()) for i in range(n)]
    rows, cols = np.where(adj > 0)
    for i, j in zip(rows.astype(int), cols.astype(int)):
        neighbors[j].add(i)
    return neighbors


def _component_nodes(nodes: Iterable[int], neighbors: list[set[int]]) -> list[list[int]]:
    node_set = set(int(n) for n in nodes)
    seen: set[int] = set()
    comps: list[list[int]] = []
    for start in sorted(node_set):
        if start in seen:
            continue
        queue: deque[int] = deque([start])
        seen.add(start)
        comp: list[int] = []
        while queue:
            node = queue.popleft()
            comp.append(node)
            for nb in neighbors[node]:
                if nb in node_set and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        comps.append(sorted(comp))
    comps.sort(key=lambda x: (-len(x), x[0] if x else -1))
    return comps


def _allocate_counts(component_weights: list[float], component_sizes: list[int], count: int) -> list[int]:
    if count < len(component_weights):
        raise ValueError("region count is smaller than the number of non-empty components")
    total = float(sum(component_weights))
    total_size = float(max(sum(component_sizes), 1))
    max_component_region_size = max(1, int(math.ceil((total_size / max(count, 1)) * 2.0)))
    min_alloc = [
        max(1, min(size, int(math.ceil(size / max_component_region_size))))
        for size in component_sizes
    ]
    if total <= 0:
        raw = [count * (size / total_size) for size in component_sizes]
    else:
        raw = [
            count * (0.2 * (w / total) + 0.8 * (size / total_size))
            for w, size in zip(component_weights, component_sizes)
        ]
    if sum(min_alloc) > count:
        min_alloc = [1 for _ in component_sizes]
    alloc = [
        max(min_count, min(size, int(math.floor(v))))
        for v, size, min_count in zip(raw, component_sizes, min_alloc)
    ]
    while sum(alloc) < count:
        candidates = [
            (raw[i] - math.floor(raw[i]), component_weights[i], i)
            for i in range(len(alloc))
            if alloc[i] < component_sizes[i]
        ]
        if not candidates:
            break
        _, _, idx = max(candidates)
        alloc[idx] += 1
    while sum(alloc) > count:
        candidates = [
            (alloc[i] - raw[i], component_weights[i], i)
            for i in range(len(alloc))
            if alloc[i] > min_alloc[i]
        ]
        if not candidates:
            break
        _, _, idx = max(candidates)
        alloc[idx] -= 1
    return alloc


def _graph_distances(nodes: list[int], neighbors: list[set[int]], start: int) -> dict[int, int]:
    node_set = set(nodes)
    dist = {start: 0}
    queue: deque[int] = deque([start])
    while queue:
        node = queue.popleft()
        for nb in neighbors[node]:
            if nb in node_set and nb not in dist:
                dist[nb] = dist[node] + 1
                queue.append(nb)
    return dist


def _choose_seeds(nodes: list[int], count: int, weights: np.ndarray, neighbors: list[set[int]]) -> list[int]:
    if count >= len(nodes):
        return sorted(nodes)
    seeds = [max(nodes, key=lambda n: (weights[n], -n))]
    distances = [_graph_distances(nodes, neighbors, seeds[0])]
    while len(seeds) < count:
        best_node = None
        best_score = -1.0
        for node in nodes:
            if node in seeds:
                continue
            min_dist = min(d.get(node, 10_000) for d in distances)
            score = (min_dist + 1.0) + 0.05 * math.log1p(max(float(weights[node]), 1.0))
            if score > best_score:
                best_score = score
                best_node = node
        assert best_node is not None
        seeds.append(best_node)
        distances.append(_graph_distances(nodes, neighbors, best_node))
    return seeds


def _local_osrm_compact_cost(
    node: int,
    cluster: list[int],
    osrm_distances: np.ndarray | None,
    fallback_cost: float,
) -> float:
    if osrm_distances is None:
        return 0.0
    vals: list[float] = []
    for member in cluster:
        for src, dst in ((node, member), (member, node)):
            value = float(osrm_distances[src, dst])
            if math.isfinite(value) and value > 0:
                vals.append(value)
    if not vals:
        return fallback_cost
    return float(np.median(vals))


def _grow_clusters(
    nodes: list[int],
    count: int,
    weights: np.ndarray,
    neighbors: list[set[int]],
    osrm_distances: np.ndarray | None = None,
) -> list[list[int]]:
    if count >= len(nodes):
        return [[n] for n in sorted(nodes)]

    seeds = _choose_seeds(nodes, count, weights, neighbors)
    clusters = [[seed] for seed in seeds]
    cluster_sets = [set(c) for c in clusters]
    cluster_weights = [float(weights[seed]) for seed in seeds]
    assigned = set(seeds)
    unassigned = set(nodes) - assigned
    target = float(sum(weights[n] for n in nodes)) / float(count)
    target_size = float(len(nodes)) / float(count)
    size_cap = max(1, int(math.ceil(target_size * 2.0)))
    if osrm_distances is not None:
        valid = osrm_distances[np.isfinite(osrm_distances) & (osrm_distances > 0)]
        osrm_scale = float(np.median(valid)) if valid.size else 1.0
    else:
        osrm_scale = 1.0

    while unassigned:
        progressed = False
        order = sorted(range(count), key=lambda i: (len(clusters[i]), cluster_weights[i], i))
        for cluster_idx in order:
            if len(clusters[cluster_idx]) >= size_cap and any(len(c) < size_cap for c in clusters):
                continue
            candidates: set[int] = set()
            for node in cluster_sets[cluster_idx]:
                candidates.update(nb for nb in neighbors[node] if nb in unassigned)
            if not candidates:
                continue
            choice = min(
                candidates,
                key=lambda n: (
                    abs((len(clusters[cluster_idx]) + 1) - target_size) / max(target_size, 1.0),
                    abs((cluster_weights[cluster_idx] + float(weights[n])) - target) / max(target, 1.0),
                    _local_osrm_compact_cost(
                        n,
                        clusters[cluster_idx],
                        osrm_distances,
                        fallback_cost=osrm_scale * 10.0,
                    ) / max(osrm_scale, 1e-6),
                    -float(weights[n]),
                    n,
                ),
            )
            clusters[cluster_idx].append(choice)
            cluster_sets[cluster_idx].add(choice)
            cluster_weights[cluster_idx] += float(weights[choice])
            assigned.add(choice)
            unassigned.remove(choice)
            progressed = True
            if not unassigned:
                break
        if not progressed:
            # If the size cap blocks every adjacent expansion, relax it for
            # one step but still require adjacency so contiguity is preserved.
            relaxed_choice: tuple[int, int] | None = None
            relaxed_score: tuple[float, float, float, float, int] | None = None
            for cluster_idx in range(count):
                candidates: set[int] = set()
                for node in cluster_sets[cluster_idx]:
                    candidates.update(nb for nb in neighbors[node] if nb in unassigned)
                for candidate in candidates:
                    score = (
                        abs((len(clusters[cluster_idx]) + 1) - target_size) / max(target_size, 1.0),
                        abs((cluster_weights[cluster_idx] + float(weights[candidate])) - target) / max(target, 1.0),
                        _local_osrm_compact_cost(
                            candidate,
                            clusters[cluster_idx],
                            osrm_distances,
                            fallback_cost=osrm_scale * 10.0,
                        ) / max(osrm_scale, 1e-6),
                        -float(weights[candidate]),
                        candidate,
                    )
                    if relaxed_score is None or score < relaxed_score:
                        relaxed_score = score
                        relaxed_choice = (cluster_idx, candidate)
            if relaxed_choice is None:
                raise RuntimeError("Could not grow contiguous superzone clusters")
            best_cluster, node = relaxed_choice
            clusters[best_cluster].append(node)
            cluster_sets[best_cluster].add(node)
            cluster_weights[best_cluster] += float(weights[node])
            unassigned.remove(node)

    return [sorted(c) for c in clusters]


def build_superzone_membership(
    data_dir: str | Path = "data",
    *,
    k: int = 64,
    shapefile: str | Path | None = None,
    demand_supply_weight: float = 0.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create a deterministic contiguous K-superzone mapping from the 263-zone graph."""
    data_root = _as_path(data_dir)
    adj = np.load(data_root / "adjacency_matrix.npy").astype(np.float32)
    demand = np.load(data_root / "node_demand.npy", mmap_mode="r")
    supply = np.load(data_root / "node_supply.npy", mmap_mode="r")
    zone_info = load_zone_table(data_root, shapefile=shapefile)
    compact_path = data_root / "edge_lengths_osrm.npy"
    osrm_compact_distances = None
    if compact_path.exists():
        osrm_compact_distances = np.load(compact_path).astype(np.float32)
        osrm_compact_distances[osrm_compact_distances <= 0] = np.inf

    weights = np.asarray(demand.sum(axis=0), dtype=np.float64)
    if demand_supply_weight > 0:
        weights = weights + demand_supply_weight * np.asarray(supply.sum(axis=0), dtype=np.float64)
    weights = np.maximum(weights, 1.0)

    protected = []
    for name in AIRPORT_NAMES:
        matches = zone_info.index[zone_info["zone_name"].astype(str).str.lower() == name.lower()].tolist()
        if matches:
            protected.append(int(zone_info.loc[matches[0], "index"]))
    protected = sorted(set(protected))
    if k <= len(protected):
        raise ValueError("k must be larger than the number of protected airport zones")

    n = adj.shape[0]
    membership = np.full(n, -1, dtype=np.int32)
    region_rows: list[dict] = []

    next_region = 0
    for node in protected:
        membership[node] = next_region
        row = zone_info.loc[zone_info["index"] == node].iloc[0]
        region_rows.append(
            {
                "region_id": next_region,
                "member_indices": str(node),
                "member_locationids": str(int(row["locationid"])),
                "member_zone_names": str(row["zone_name"]),
                "boroughs": str(row["borough"]),
                "num_zones": 1,
                "demand_weight": float(weights[node]),
                "is_reserved": 1,
            }
        )
        next_region += 1

    remaining_nodes = [i for i in range(n) if i not in protected]
    neighbors = undirected_neighbors(adj)
    components = _component_nodes(remaining_nodes, neighbors)
    comp_weights = [float(sum(weights[n] for n in comp)) for comp in components]
    comp_sizes = [len(comp) for comp in components]
    counts = _allocate_counts(comp_weights, comp_sizes, k - len(protected))

    for comp, count in zip(components, counts):
        clusters = _grow_clusters(
            comp,
            count,
            weights,
            neighbors,
            osrm_distances=osrm_compact_distances,
        )
        for cluster in clusters:
            region_id = next_region
            for node in cluster:
                membership[node] = region_id
            rows = zone_info[zone_info["index"].isin(cluster)].sort_values("index")
            region_rows.append(
                {
                    "region_id": region_id,
                    "member_indices": " ".join(str(int(x)) for x in rows["index"].tolist()),
                    "member_locationids": " ".join(str(int(x)) for x in rows["locationid"].tolist()),
                    "member_zone_names": " | ".join(str(x) for x in rows["zone_name"].tolist()),
                    "boroughs": " ".join(sorted(set(str(x) for x in rows["borough"].tolist()))),
                    "num_zones": int(len(cluster)),
                    "demand_weight": float(sum(weights[n] for n in cluster)),
                    "is_reserved": 0,
                }
            )
            next_region += 1

    if next_region != k:
        raise RuntimeError(f"Expected {k} superzones, produced {next_region}")
    if (membership < 0).any():
        missing = np.flatnonzero(membership < 0).tolist()
        raise RuntimeError(f"Unassigned zones: {missing}")

    region_info = pd.DataFrame(region_rows).sort_values("region_id").reset_index(drop=True)
    return membership, region_info


def aggregate_nodes(values: np.ndarray, membership: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values)
    out = np.zeros((values.shape[0], k), dtype=np.float32)
    for region_id in range(k):
        members = np.flatnonzero(membership == region_id)
        if members.size:
            out[:, region_id] = values[:, members].sum(axis=1)
    return out


def _region_representatives(zone_info: pd.DataFrame, membership: np.ndarray, k: int, weights: np.ndarray) -> pd.DataFrame:
    rows = []
    for region_id in range(k):
        members = np.flatnonzero(membership == region_id)
        region_weights = weights[members].astype(np.float64)
        region_weights = region_weights / max(float(region_weights.sum()), 1.0)
        sub = zone_info[zone_info["index"].isin(members)].sort_values("index")
        lon = float(np.sum(sub["lon"].to_numpy(dtype=np.float64) * region_weights))
        lat = float(np.sum(sub["lat"].to_numpy(dtype=np.float64) * region_weights))
        medoid = int(members[int(np.argmax(weights[members]))])
        rows.append(
            {
                "region_id": region_id,
                "representative_lon": lon,
                "representative_lat": lat,
                "representative_zone_index": medoid,
            }
        )
    return pd.DataFrame(rows)


def query_osrm_table(
    lons: list[float],
    lats: list[float],
    *,
    url_base: str = "http://router.project-osrm.org",
    timeout: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in zip(lons, lats))
    url = f"{url_base.rstrip('/')}/table/v1/driving/{coords}?annotations=distance,duration"
    req = urllib.request.Request(url, headers={"User-Agent": "STDR-superzone-builder/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("code") != "Ok":
        raise RuntimeError(f"OSRM table failed: {payload.get('code')} {payload.get('message')}")
    distances_raw = payload.get("distances")
    durations_raw = payload.get("durations")
    distances = np.asarray(
        [[np.inf if x is None else float(x) / 1000.0 for x in row] for row in distances_raw],
        dtype=np.float32,
    )
    durations = np.asarray(
        [[np.inf if x is None else float(x) / 3600.0 for x in row] for row in durations_raw],
        dtype=np.float32,
    )
    reachable = np.isfinite(distances) & np.isfinite(durations)
    return distances, durations, reachable


def _fallback_region_costs(
    base_distances: np.ndarray,
    membership: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.full((k, k), np.inf, dtype=np.float32)
    for i in range(k):
        src = np.flatnonzero(membership == i)
        for j in range(k):
            if i == j:
                distances[i, j] = 0.0
                continue
            dst = np.flatnonzero(membership == j)
            block = base_distances[np.ix_(src, dst)]
            valid = block[np.isfinite(block) & (block > 0)]
            if valid.size:
                distances[i, j] = float(np.median(valid))
    durations = distances / 30.0
    reachable = np.isfinite(distances)
    return distances, durations, reachable


def build_rl_edges(
    distances: np.ndarray,
    durations: np.ndarray,
    reachable: np.ndarray,
    demand_matrix: np.ndarray,
    *,
    topk: int = 8,
    connector_count: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    k = distances.shape[0]
    edges: list[tuple[int, int]] = []
    lengths: list[float] = []
    durations_out: list[float] = []
    action_rows: list[dict] = []
    edge_set: set[tuple[int, int]] = set()

    def add_edge(src: int, dst: int, kind: str, rank: int) -> bool:
        if src == dst or not bool(reachable[src, dst]) or (src, dst) in edge_set:
            return False
        edge_set.add((src, dst))
        edges.append((src, dst))
        lengths.append(float(distances[src, dst]))
        durations_out.append(float(durations[src, dst]))
        action_rows.append(
            {
                "src": src,
                "dst": dst,
                "action_type": kind,
                "action_rank": rank,
                "distance_km": float(distances[src, dst]),
                "duration_hours": float(durations[src, dst]),
            }
        )
        return True

    for src in range(k):
        candidates = [
            dst for dst in range(k)
            if dst != src and bool(reachable[src, dst]) and np.isfinite(durations[src, dst])
        ]
        candidates.sort(key=lambda dst: (float(distances[src, dst]), float(durations[src, dst]), dst))
        for rank, dst in enumerate(candidates[:topk], start=1):
            add_edge(src, dst, "osrm_topk", rank)

        demand_candidates = [
            dst for dst in range(k)
            if dst != src and bool(reachable[src, dst]) and float(demand_matrix[src, dst]) > 0
        ]
        demand_candidates.sort(key=lambda dst: (-float(demand_matrix[src, dst]), float(durations[src, dst]), dst))
        added_connectors = 0
        for dst in demand_candidates:
            if add_edge(src, dst, "high_demand_connector", added_connectors + 1):
                added_connectors += 1
                if added_connectors >= connector_count:
                    break

    def current_sccs() -> list[list[int]]:
        adj = [[] for _ in range(k)]
        rev = [[] for _ in range(k)]
        for src, dst in edge_set:
            adj[src].append(dst)
            rev[dst].append(src)

        seen: set[int] = set()
        order: list[int] = []

        def dfs1(node: int) -> None:
            seen.add(node)
            for nb in adj[node]:
                if nb not in seen:
                    dfs1(nb)
            order.append(node)

        def dfs2(node: int, comp: list[int]) -> None:
            seen.add(node)
            comp.append(node)
            for nb in rev[node]:
                if nb not in seen:
                    dfs2(nb, comp)

        for node in range(k):
            if node not in seen:
                dfs1(node)
        seen.clear()
        comps: list[list[int]] = []
        for node in reversed(order):
            if node not in seen:
                comp: list[int] = []
                dfs2(node, comp)
                comps.append(comp)
        return comps

    def best_cross_component_edge(src_comp: list[int], dst_comp: list[int]) -> tuple[int, int] | None:
        demand_priority = np.asarray(demand_matrix.sum(axis=0) + demand_matrix.sum(axis=1), dtype=np.float64)
        best: tuple[float, float, int, int, int, int] | None = None
        for src in src_comp:
            for dst in dst_comp:
                if src == dst or not bool(reachable[src, dst]) or not np.isfinite(durations[src, dst]):
                    continue
                score = (
                    float(durations[src, dst]),
                    -float(demand_priority[dst]),
                    int(src),
                    int(dst),
                    int(src),
                    int(dst),
                )
                if best is None or score < best:
                    best = score
        if best is None:
            return None
        return best[4], best[5]

    comps = current_sccs()
    if len(comps) > 1:
        demand_priority = np.asarray(demand_matrix.sum(axis=0) + demand_matrix.sum(axis=1), dtype=np.float64)
        comps.sort(key=lambda comp: (-float(demand_priority[comp].sum()), min(comp)))
        for rank, src_comp in enumerate(comps, start=1):
            dst_comp = comps[rank % len(comps)]
            edge = best_cross_component_edge(src_comp, dst_comp)
            if edge is not None:
                add_edge(edge[0], edge[1], "scc_high_demand_connector", rank)

    if not edges:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            pd.DataFrame(action_rows),
        )
    return (
        np.asarray(edges, dtype=np.int32),
        np.asarray(lengths, dtype=np.float32),
        np.asarray(durations_out, dtype=np.float32),
        pd.DataFrame(action_rows),
    )


def aggregate_od_by_membership(od_matrix: np.ndarray, membership: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((k, k), dtype=np.float64)
    rows, cols = np.nonzero(od_matrix)
    for src, dst in zip(rows.astype(int), cols.astype(int)):
        out[int(membership[src]), int(membership[dst])] += float(od_matrix[src, dst])
    return out.astype(np.float32)


def build_rl_edge_speed_mapping(
    base_edge_index: np.ndarray,
    membership: np.ndarray,
    rl_edge_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Map each superzone RL edge to original STGAT speed edges."""
    base_edge_index = np.asarray(base_edge_index, dtype=np.int32)
    membership = np.asarray(membership, dtype=np.int32)
    rl_edge_index = np.asarray(rl_edge_index, dtype=np.int32)
    src_regions = membership[base_edge_index[:, 0]]
    dst_regions = membership[base_edge_index[:, 1]]
    all_indices: list[int] = []
    offsets = [0]
    rows: list[dict] = []
    all_base = np.arange(base_edge_index.shape[0], dtype=np.int32)

    for rl_idx, (src_region, dst_region) in enumerate(rl_edge_index.astype(int)):
        direct = all_base[(src_regions == src_region) & (dst_regions == dst_region)]
        mapping_type = "direct_cross_superzone"
        selected = direct
        if selected.size == 0:
            selected = all_base[(src_regions == src_region) | (dst_regions == dst_region)]
            mapping_type = "incident_superzone"
        if selected.size == 0:
            selected = all_base[src_regions == src_region]
            mapping_type = "source_outgoing"
        if selected.size == 0:
            selected = all_base[dst_regions == dst_region]
            mapping_type = "destination_incoming"
        if selected.size == 0:
            selected = all_base
            mapping_type = "global_fallback"

        selected = np.asarray(selected, dtype=np.int32)
        all_indices.extend(int(x) for x in selected.tolist())
        offsets.append(len(all_indices))
        rows.append(
            {
                "rl_edge_id": int(rl_idx),
                "src": int(src_region),
                "dst": int(dst_region),
                "speed_mapping_type": mapping_type,
                "speed_mapping_edge_count": int(selected.size),
            }
        )

    return (
        np.asarray(offsets, dtype=np.int32),
        np.asarray(all_indices, dtype=np.int32),
        pd.DataFrame(rows),
    )


def aggregate_speed_profile_to_rl_edges(
    base_speed_profile: np.ndarray,
    offsets: np.ndarray,
    indices: np.ndarray,
    fallback_speeds: np.ndarray,
) -> np.ndarray:
    """Aggregate original-edge dynamic speed profiles onto superzone RL edges."""
    profile = np.asarray(base_speed_profile, dtype=np.float32)
    if profile.ndim == 1:
        profile = profile[:, None]
    offsets = np.asarray(offsets, dtype=np.int32)
    indices = np.asarray(indices, dtype=np.int32)
    fallback = np.asarray(fallback_speeds, dtype=np.float32)
    if offsets.shape[0] != fallback.shape[0] + 1:
        raise ValueError("speed mapping offsets length must be E_rl + 1")
    out = np.zeros((fallback.shape[0], profile.shape[1]), dtype=np.float32)
    for edge_id in range(fallback.shape[0]):
        start = int(offsets[edge_id])
        end = int(offsets[edge_id + 1])
        selected = indices[start:end]
        if selected.size:
            out[edge_id] = profile[selected].mean(axis=0)
        else:
            out[edge_id] = fallback[edge_id]
    return np.maximum(out, 1.0).astype(np.float32)


def region_compactness_table(base_distances: np.ndarray | None, membership: np.ndarray, k: int) -> pd.DataFrame:
    rows: list[dict] = []
    for region_id in range(k):
        members = np.flatnonzero(membership == region_id)
        if members.size <= 1 or base_distances is None:
            rows.append(
                {
                    "region_id": region_id,
                    "intra_osrm_edge_count": 0,
                    "intra_osrm_median_km": 0.0,
                    "intra_osrm_mean_km": 0.0,
                }
            )
            continue
        block = base_distances[np.ix_(members, members)]
        mask = np.isfinite(block) & (block > 0)
        vals = block[mask]
        rows.append(
            {
                "region_id": region_id,
                "intra_osrm_edge_count": int(vals.size),
                "intra_osrm_median_km": float(np.median(vals)) if vals.size else float("nan"),
                "intra_osrm_mean_km": float(np.mean(vals)) if vals.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_superzone_artifacts(
    data_dir: str | Path = "data",
    *,
    output_dir: str | Path | None = None,
    k: int = 64,
    topk: int = 8,
    connector_count: int = 2,
    shapefile: str | Path | None = None,
    use_osrm: bool = True,
    osrm_url: str = "http://router.project-osrm.org",
    reuse_existing_costs: bool = True,
    allow_fallback_costs: bool = False,
) -> dict:
    data_root = _as_path(data_dir)
    out = _as_path(output_dir) if output_dir is not None else data_root / f"superzones_k{k}"
    out.mkdir(parents=True, exist_ok=True)

    membership, region_info = build_superzone_membership(data_root, k=k, shapefile=shapefile)
    zone_info = load_zone_table(data_root, shapefile=shapefile)
    demand = np.load(data_root / "node_demand.npy", mmap_mode="r")
    supply = np.load(data_root / "node_supply.npy", mmap_mode="r")
    weights = np.asarray(demand.sum(axis=0), dtype=np.float64)
    reps = _region_representatives(zone_info, membership, k, np.maximum(weights, 1.0))
    region_info = region_info.merge(reps, on="region_id", how="left")
    base_osrm_distances = None
    base_osrm_path = data_root / "edge_lengths_osrm.npy"
    if base_osrm_path.exists():
        base_osrm_distances = np.load(base_osrm_path).astype(np.float32)
        base_osrm_distances[base_osrm_distances <= 0] = np.inf
    region_info = region_info.merge(
        region_compactness_table(base_osrm_distances, membership, k),
        on="region_id",
        how="left",
    )

    region_demand = aggregate_nodes(demand, membership, k)
    region_supply = aggregate_nodes(supply, membership, k)
    np.save(out / "region_membership.npy", membership)
    pd.DataFrame(
        {
            "base_zone_index": np.arange(membership.shape[0], dtype=np.int32),
            "superzone_id": membership.astype(np.int32),
        }
    ).to_csv(out / "region_membership.csv", index=False, encoding="utf-8-sig")
    np.save(out / "region_node_demand.npy", region_demand)
    np.save(out / "region_node_supply.npy", region_supply)
    region_info.to_csv(out / "region_info.csv", index=False, encoding="utf-8-sig")

    distances = durations = reachable = None
    cost_source = "osrm_table"
    cost_matrix_origin = "unresolved"
    existing_cost_paths = (
        out / "dispatch_distance_km.npy",
        out / "dispatch_duration_hours.npy",
        out / "dispatch_reachable.npy",
    )
    cached_meta_path = out / "superzone_meta.json"
    cached_meta = json.loads(cached_meta_path.read_text(encoding="utf-8")) if cached_meta_path.exists() else {}
    if reuse_existing_costs and all(path.exists() for path in existing_cost_paths):
        cached_distances = np.load(existing_cost_paths[0]).astype(np.float32)
        cached_durations = np.load(existing_cost_paths[1]).astype(np.float32)
        cached_reachable = np.load(existing_cost_paths[2]).astype(bool)
        offdiag = ~np.eye(k, dtype=bool)
        cache_is_valid_osrm = (
            cached_meta.get("cost_source") == "osrm_table"
            and cached_distances.shape == (k, k)
            and cached_durations.shape == (k, k)
            and cached_reachable.shape == (k, k)
            and float(np.isfinite(cached_durations)[offdiag].mean()) >= 0.999
            and float(cached_reachable[offdiag].mean()) >= 0.999
        )
        if cache_is_valid_osrm:
            distances, durations, reachable = cached_distances, cached_durations, cached_reachable
            cost_source = "osrm_table"
            cost_matrix_origin = "cached_osrm_table"
    if use_osrm:
        if distances is not None and durations is not None and reachable is not None:
            pass
        else:
            try:
                distances, durations, reachable = query_osrm_table(
                    region_info["representative_lon"].astype(float).tolist(),
                    region_info["representative_lat"].astype(float).tolist(),
                    url_base=osrm_url,
                )
                cost_matrix_origin = "queried_osrm_table"
            except Exception as exc:
                raise RuntimeError(
                    "OSRM table query failed. Re-run with --no-osrm only for a local fallback smoke test."
                ) from exc
    if distances is None or durations is None or reachable is None:
        if not allow_fallback_costs:
            raise RuntimeError(
                "No valid cached OSRM dispatch matrices are available. "
                "Run without --no-osrm to query OSRM, or pass --allow-fallback-costs "
                "only for a local fallback smoke test."
            )
        if base_osrm_distances is None:
            raise FileNotFoundError(f"Could not find {base_osrm_path} for fallback region costs")
        distances, durations, reachable = _fallback_region_costs(base_osrm_distances, membership, k)
        cost_source = "fallback_existing_edge_lengths"
        cost_matrix_origin = "fallback_existing_edge_lengths"

    np.fill_diagonal(distances, 0.0)
    np.fill_diagonal(durations, 0.0)
    np.fill_diagonal(reachable, True)
    np.save(out / "dispatch_distance_km.npy", distances.astype(np.float32))
    np.save(out / "dispatch_duration_hours.npy", durations.astype(np.float32))
    np.save(out / "dispatch_reachable.npy", reachable.astype(np.bool_))

    # A lightweight demand connector proxy from aggregate annual node movements.
    # If OD tensors are unavailable, this still creates useful local OSRM top-k
    # actions while keeping connector_count harmless.
    od_proxy = np.outer(region_demand.sum(axis=0), region_demand.sum(axis=0)).astype(np.float32)
    np.fill_diagonal(od_proxy, 0.0)
    rl_edge_index, rl_lengths, rl_durations, action_info = build_rl_edges(
        distances,
        durations,
        reachable,
        od_proxy,
        topk=topk,
        connector_count=connector_count,
    )
    safe_durations = np.maximum(rl_durations, 1e-5)
    rl_speeds = np.divide(rl_lengths, safe_durations, out=np.zeros_like(rl_lengths), where=safe_durations > 0)
    base_edge_index = np.load(data_root / "edge_index.npy").astype(np.int32)
    mapping_offsets, mapping_indices, mapping_info = build_rl_edge_speed_mapping(
        base_edge_index,
        membership,
        rl_edge_index,
    )
    action_info = action_info.merge(mapping_info, on=["src", "dst"], how="left")
    np.save(out / "rl_edge_index.npy", rl_edge_index)
    np.save(out / "rl_edge_lengths.npy", rl_lengths)
    np.save(out / "rl_edge_durations_hours.npy", rl_durations)
    np.save(out / "rl_edge_speeds_kmh.npy", rl_speeds.astype(np.float32))
    np.save(out / "rl_edge_speed_mapping_offsets.npy", mapping_offsets)
    np.save(out / "rl_edge_speed_mapping_indices.npy", mapping_indices)
    action_info.to_csv(out / "rl_action_info.csv", index=False, encoding="utf-8-sig")

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_base_zones": int(membership.shape[0]),
        "num_superzones": int(k),
        "osrm_topk": int(topk),
        "connector_count": int(connector_count),
        "airport_names": list(AIRPORT_NAMES),
        "cost_source": cost_source,
        "cost_matrix_origin": cost_matrix_origin,
        "no_stay_action": True,
        "rl_speed_profile_source": "dynamic_stgat_edge_aggregation",
        "rl_speed_mapping_fallback": "incident_superzone_then_source_destination_then_global",
        "connector_demand_source": "aggregate_demand_outer_product",
    }
    (out / "superzone_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_superzone_artifacts(data_dir: str | Path = "data", superzone_dir: str | Path | None = None) -> dict:
    data_root = _as_path(data_dir)
    root = _as_path(superzone_dir) if superzone_dir is not None else data_root / "superzones_k64"
    if not root.exists():
        raise FileNotFoundError(
            f"Could not find {root}. Run build_superzone_graph.py before using superzone mode."
    )
    meta_path = root / "superzone_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    artifacts = {
        "root": root,
        "meta": meta,
        "membership": np.load(root / "region_membership.npy").astype(np.int32),
        "region_info": pd.read_csv(root / "region_info.csv"),
        "region_demand": np.load(root / "region_node_demand.npy").astype(np.float32),
        "region_supply": np.load(root / "region_node_supply.npy").astype(np.float32),
        "dispatch_distance_km": np.load(root / "dispatch_distance_km.npy").astype(np.float32),
        "dispatch_duration_hours": np.load(root / "dispatch_duration_hours.npy").astype(np.float32),
        "dispatch_reachable": np.load(root / "dispatch_reachable.npy").astype(bool),
        "rl_edge_index": np.load(root / "rl_edge_index.npy").astype(np.int32),
        "rl_edge_lengths": np.load(root / "rl_edge_lengths.npy").astype(np.float32),
        "rl_edge_durations_hours": np.load(root / "rl_edge_durations_hours.npy").astype(np.float32),
        "rl_edge_speeds_kmh": np.load(root / "rl_edge_speeds_kmh.npy").astype(np.float32),
        "rl_edge_speed_mapping_offsets": np.load(root / "rl_edge_speed_mapping_offsets.npy").astype(np.int32),
        "rl_edge_speed_mapping_indices": np.load(root / "rl_edge_speed_mapping_indices.npy").astype(np.int32),
        "action_info": pd.read_csv(root / "rl_action_info.csv") if (root / "rl_action_info.csv").exists() else pd.DataFrame(),
    }
    k = int(meta.get("num_superzones", artifacts["region_demand"].shape[1]))
    errors: list[str] = []
    if artifacts["membership"].ndim != 1:
        errors.append("region_membership.npy must be one-dimensional")
    if artifacts["membership"].shape[0] != int(meta.get("num_base_zones", artifacts["membership"].shape[0])):
        errors.append("membership length does not match num_base_zones metadata")
    if artifacts["region_demand"].shape != artifacts["region_supply"].shape:
        errors.append("region demand/supply shapes differ")
    if artifacts["region_demand"].ndim != 2 or artifacts["region_demand"].shape[1] != k:
        errors.append(f"region demand must have K={k} columns")
    for name in ("dispatch_distance_km", "dispatch_duration_hours", "dispatch_reachable"):
        if artifacts[name].shape != (k, k):
            errors.append(f"{name} must have shape ({k}, {k})")
    if artifacts["rl_edge_index"].ndim != 2 or artifacts["rl_edge_index"].shape[1] != 2:
        errors.append("rl_edge_index.npy must have shape (E, 2)")
    edge_count = artifacts["rl_edge_index"].shape[0]
    for name in ("rl_edge_lengths", "rl_edge_durations_hours", "rl_edge_speeds_kmh"):
        if artifacts[name].shape[0] != edge_count:
            errors.append(f"{name} length must match rl_edge_index rows")
    if artifacts["rl_edge_speed_mapping_offsets"].shape[0] != edge_count + 1:
        errors.append("rl_edge_speed_mapping_offsets length must be E + 1")
    if artifacts["rl_edge_speed_mapping_offsets"].size:
        if int(artifacts["rl_edge_speed_mapping_offsets"][0]) != 0:
            errors.append("rl_edge_speed_mapping_offsets must start at 0")
        if int(artifacts["rl_edge_speed_mapping_offsets"][-1]) != artifacts["rl_edge_speed_mapping_indices"].shape[0]:
            errors.append("rl_edge_speed_mapping_offsets end must match mapping index length")
        if np.any(np.diff(artifacts["rl_edge_speed_mapping_offsets"]) <= 0):
            errors.append("every RL edge must have at least one mapped STGAT speed edge")
    if edge_count:
        if artifacts["rl_edge_index"].min() < 0 or artifacts["rl_edge_index"].max() >= k:
            errors.append("rl_edge_index contains node ids outside [0, K)")
        if np.any(artifacts["rl_edge_index"][:, 0] == artifacts["rl_edge_index"][:, 1]):
            errors.append("rl_edge_index contains self-loop edges")
        for name in ("rl_edge_lengths", "rl_edge_durations_hours", "rl_edge_speeds_kmh"):
            values = artifacts[name]
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                errors.append(f"{name} must be finite and positive")
    finite_dispatch = np.isfinite(artifacts["dispatch_duration_hours"])
    if not np.array_equal(artifacts["dispatch_reachable"], finite_dispatch):
        errors.append("dispatch_reachable must match finite dispatch_duration_hours entries")
    if not bool(meta.get("no_stay_action", False)):
        errors.append("superzone_meta.json must declare no_stay_action=true")
    if errors:
        raise ValueError(f"Invalid superzone artifacts under {root}: " + "; ".join(errors))
    return artifacts


def reachability_metrics(edge_index: np.ndarray, num_nodes: int, demand_weights: np.ndarray | None = None) -> dict:
    adj = [[] for _ in range(num_nodes)]
    rev = [[] for _ in range(num_nodes)]
    for src, dst in edge_index.astype(int):
        adj[src].append(dst)
        rev[dst].append(src)

    reach_matrix = np.zeros((num_nodes, num_nodes), dtype=bool)
    reachable_counts = np.zeros(num_nodes, dtype=np.int32)
    for start in range(num_nodes):
        seen = {start}
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        for node in seen:
            if node != start:
                reach_matrix[start, node] = True
        reachable_counts[start] = int(reach_matrix[start].sum())

    if demand_weights is None:
        demand_weights = np.ones(num_nodes, dtype=np.float64)
    demand_weights = np.maximum(np.asarray(demand_weights, dtype=np.float64), 0.0)
    denom = max(float(demand_weights.sum()), 1.0)
    pair_weights = np.outer(demand_weights, demand_weights)
    np.fill_diagonal(pair_weights, 0.0)
    pair_weight_total = float(pair_weights.sum())
    weighted_reach = (
        float(pair_weights[reach_matrix].sum() / pair_weight_total)
        if pair_weight_total > 0
        else float(reach_matrix.sum() / max(num_nodes * (num_nodes - 1), 1))
    )

    # Kosaraju SCC.
    seen: set[int] = set()
    order: list[int] = []

    def dfs1(node: int) -> None:
        seen.add(node)
        for nb in adj[node]:
            if nb not in seen:
                dfs1(nb)
        order.append(node)

    def dfs2(node: int, comp: list[int]) -> None:
        seen.add(node)
        comp.append(node)
        for nb in rev[node]:
            if nb not in seen:
                dfs2(nb, comp)

    for node in range(num_nodes):
        if node not in seen:
            dfs1(node)
    seen.clear()
    comps: list[list[int]] = []
    for node in reversed(order):
        if node not in seen:
            comp: list[int] = []
            dfs2(node, comp)
            comps.append(comp)
    largest = max(comps, key=len) if comps else []
    largest_demand = float(demand_weights[largest].sum()) if largest else 0.0
    out_deg = Counter(int(src) for src in edge_index[:, 0]) if edge_index.size else Counter()
    in_deg = Counter(int(dst) for dst in edge_index[:, 1]) if edge_index.size else Counter()
    return {
        "num_nodes": int(num_nodes),
        "num_edges": int(edge_index.shape[0]),
        "raw_reachability": float(reach_matrix.sum() / max(num_nodes * (num_nodes - 1), 1)),
        "demand_weighted_reachability": weighted_reach,
        "mean_reachable_nodes": float(reachable_counts.mean()),
        "min_reachable_nodes": int(reachable_counts.min()) if num_nodes else 0,
        "num_scc": int(len(comps)),
        "largest_scc_nodes": int(len(largest)),
        "largest_scc_node_pct": float(len(largest) / max(num_nodes, 1)),
        "largest_scc_demand_pct": float(largest_demand / denom),
        "zero_out_degree_nodes": int(sum(1 for node in range(num_nodes) if out_deg[node] == 0)),
        "zero_in_degree_nodes": int(sum(1 for node in range(num_nodes) if in_deg[node] == 0)),
        "self_loop_edges": int(sum(1 for src, dst in edge_index.astype(int) if src == dst)),
    }
```

### build_superzone_graph.py

```python
from __future__ import annotations

import argparse
import json

from superzone_graph import build_superzone_artifacts, load_superzone_artifacts, reachability_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build K=64 superzone dispatch and RL graphs")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--osrm-topk", type=int, default=8)
    p.add_argument("--connector-count", type=int, default=2)
    p.add_argument("--shapefile", type=str, default="")
    p.add_argument("--osrm-url", type=str, default="http://router.project-osrm.org")
    p.add_argument(
        "--refresh-osrm",
        action="store_true",
        help="Ignore cached dispatch matrices and query OSRM Table again.",
    )
    p.add_argument(
        "--no-osrm",
        action="store_true",
        help="Use a cached OSRM Table matrix instead of querying OSRM.",
    )
    p.add_argument(
        "--allow-fallback-costs",
        action="store_true",
        help="Allow sparse edge-length fallback costs for local smoke tests only.",
    )
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    meta = build_superzone_artifacts(
        args.data_dir,
        output_dir=args.output_dir or None,
        k=args.k,
        topk=args.osrm_topk,
        connector_count=args.connector_count,
        shapefile=args.shapefile or None,
        use_osrm=not args.no_osrm,
        osrm_url=args.osrm_url,
        reuse_existing_costs=not args.refresh_osrm,
        allow_fallback_costs=args.allow_fallback_costs,
    )
    artifacts = load_superzone_artifacts(args.data_dir, args.output_dir or None)
    demand_weights = artifacts["region_demand"].sum(axis=0)
    metrics = reachability_metrics(
        artifacts["rl_edge_index"],
        int(meta["num_superzones"]),
        demand_weights,
    )
    print("Built superzone artifacts")
    print(json.dumps(meta, indent=2))
    print("RL action graph reachability")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(parse_args())
```

### validate_superzone_graph.py

```python
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from dispatch import build_dispatch_od_pairs, greedy_dispatch
from superzone_graph import load_superzone_artifacts, reachability_metrics
from graph_env import RoutingEnv, create_graph_from_data, infer_max_neighbors


def reachable_matrix(edge_index: np.ndarray, k: int) -> np.ndarray:
    adj = [[] for _ in range(k)]
    for src, dst in edge_index.astype(int):
        adj[src].append(dst)
    out = np.zeros((k, k), dtype=bool)
    for start in range(k):
        seen = {start}
        queue = [start]
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        for node in seen:
            if node != start:
                out[start, node] = True
    return out


def validate_dispatch_reachability(
    demand: np.ndarray,
    supply: np.ndarray,
    travel_time: np.ndarray,
    rl_reachable: np.ndarray,
) -> tuple[np.ndarray, dict]:
    matrix = greedy_dispatch(demand, supply, travel_time, skip_unreachable=True)
    pairs = build_dispatch_od_pairs(matrix)
    total_pairs = len(pairs)
    total_vehicles = int(sum(count for _, _, count in pairs))
    reachable_pairs = 0
    reachable_vehicles = 0
    rl_path_pairs = 0
    rl_path_vehicles = 0
    for origin, dest, count in pairs:
        if np.isfinite(travel_time[origin, dest]):
            reachable_pairs += 1
            reachable_vehicles += int(count)
        if origin == dest or bool(rl_reachable[origin, dest]):
            rl_path_pairs += 1
            rl_path_vehicles += int(count)
    return matrix, {
        "dispatch_pairs": int(total_pairs),
        "dispatch_vehicles": int(total_vehicles),
        "reachable_dispatch_pairs": int(reachable_pairs),
        "reachable_dispatch_vehicles": int(reachable_vehicles),
        "reachable_pair_rate": float(reachable_pairs / total_pairs) if total_pairs else 1.0,
        "reachable_vehicle_rate": float(reachable_vehicles / total_vehicles) if total_vehicles else 1.0,
        "rl_path_reachable_pairs": int(rl_path_pairs),
        "rl_path_reachable_vehicles": int(rl_path_vehicles),
        "rl_path_reachable_pair_rate": float(rl_path_pairs / total_pairs) if total_pairs else 1.0,
        "rl_path_reachable_vehicle_rate": float(rl_path_vehicles / total_vehicles) if total_vehicles else 1.0,
    }


def _unique_clipped_slots(candidates: list[int], max_slot: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw in candidates:
        slot = int(np.clip(int(raw), 0, max_slot))
        if slot not in seen:
            seen.add(slot)
            out.append(slot)
    return out


def parse_slot_list(raw: str, max_slot: int) -> list[int]:
    if not raw.strip():
        return []
    slots: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        slots.append(int(text))
    return _unique_clipped_slots(slots, max_slot)


def select_dispatch_slots(
    demand: np.ndarray,
    supply: np.ndarray,
    primary_slot: int,
    raw_slots: str,
) -> list[int]:
    max_slot = demand.shape[0] - 1
    explicit = parse_slot_list(raw_slots, max_slot)
    if explicit:
        return explicit

    imbalance = np.abs(demand - supply).sum(axis=1)
    positive = np.flatnonzero(imbalance > 0)
    candidates = [primary_slot]
    if positive.size:
        quantile_positions = np.linspace(0, positive.size - 1, num=min(5, positive.size), dtype=int)
        candidates.extend(int(positive[pos]) for pos in quantile_positions)
        top_count = min(4, positive.size)
        candidates.extend(int(x) for x in np.argsort(imbalance)[-top_count:][::-1])
    return _unique_clipped_slots(candidates, max_slot)


def validate_dispatch_slots(
    demand: np.ndarray,
    supply: np.ndarray,
    travel_time: np.ndarray,
    rl_reachable: np.ndarray,
    slots: list[int],
) -> dict:
    per_slot: list[dict] = []
    for slot in slots:
        matrix, metrics = validate_dispatch_reachability(
            demand[slot],
            supply[slot],
            travel_time,
            rl_reachable,
        )
        row = dict(metrics)
        row["slot"] = int(slot)
        row["dispatch_self_loop_pairs"] = int(np.count_nonzero(np.diag(matrix)))
        row["dispatch_self_loop_vehicles"] = int(np.trace(matrix))
        per_slot.append(row)

    def min_metric(name: str, default: float = 1.0) -> float:
        return float(min((float(row[name]) for row in per_slot), default=default))

    def max_metric(name: str, default: int = 0) -> int:
        return int(max((int(row[name]) for row in per_slot), default=default))

    return {
        "slots": [int(slot) for slot in slots],
        "num_slots": int(len(slots)),
        "total_dispatch_pairs": int(sum(int(row["dispatch_pairs"]) for row in per_slot)),
        "total_dispatch_vehicles": int(sum(int(row["dispatch_vehicles"]) for row in per_slot)),
        "min_dispatch_pairs": int(min((int(row["dispatch_pairs"]) for row in per_slot), default=0)),
        "min_dispatch_vehicles": int(min((int(row["dispatch_vehicles"]) for row in per_slot), default=0)),
        "min_reachable_pair_rate": min_metric("reachable_pair_rate"),
        "min_reachable_vehicle_rate": min_metric("reachable_vehicle_rate"),
        "min_rl_path_reachable_pair_rate": min_metric("rl_path_reachable_pair_rate"),
        "min_rl_path_reachable_vehicle_rate": min_metric("rl_path_reachable_vehicle_rate"),
        "max_dispatch_self_loop_pairs": max_metric("dispatch_self_loop_pairs"),
        "max_dispatch_self_loop_vehicles": max_metric("dispatch_self_loop_vehicles"),
        "per_slot": per_slot,
    }


def shape_action_smoke(artifacts: dict, k: int) -> dict:
    edge_index = artifacts["rl_edge_index"]
    max_neighbors = infer_max_neighbors(edge_index, k)
    speed = np.maximum(artifacts["rl_edge_speeds_kmh"], 1.0)
    graph = create_graph_from_data(
        k,
        edge_index,
        artifacts["rl_edge_lengths"],
        speed,
        speed,
        max_neighbors=max_neighbors,
    )
    env = RoutingEnv(
        graph,
        max_steps=5,
        num_time_slots=1,
        dynamic_edge_index=edge_index,
        dynamic_pred_speeds=speed[:, None],
    )
    out_deg = np.bincount(edge_index[:, 0], minlength=k) if edge_index.size else np.zeros(k, dtype=int)
    sources_tested = 0
    sources_failed: list[int] = []
    last_state = np.zeros(2 + 2 * max_neighbors + 1, dtype=np.float32)
    last_mask = np.zeros(max_neighbors, dtype=np.float32)
    for source in range(k):
        outgoing = np.flatnonzero(edge_index[:, 0] == source)
        if outgoing.size == 0:
            continue
        goal = int(edge_index[outgoing[0], 1])
        state, mask, _ = env.reset(source, goal)
        last_state = state
        last_mask = mask
        valid_actions = np.flatnonzero(mask > 0)
        ok = (
            state.shape[0] == 2 + 2 * max_neighbors + 1
            and mask.shape[0] == max_neighbors
            and int(mask.sum()) == int(out_deg[source])
            and valid_actions.size > 0
        )
        if ok:
            try:
                env.step(int(valid_actions[0]))
            except Exception:
                ok = False
        if not ok:
            sources_failed.append(source)
        sources_tested += 1
    return {
        "region_demand_shape": list(artifacts["region_demand"].shape),
        "region_supply_shape": list(artifacts["region_supply"].shape),
        "dispatch_duration_shape": list(artifacts["dispatch_duration_hours"].shape),
        "dispatch_reachable_shape": list(artifacts["dispatch_reachable"].shape),
        "rl_edge_index_shape": list(edge_index.shape),
        "rl_edge_lengths_len": int(artifacts["rl_edge_lengths"].shape[0]),
        "rl_edge_speeds_len": int(artifacts["rl_edge_speeds_kmh"].shape[0]),
        "max_out_degree": int(out_deg.max()) if out_deg.size else 0,
        "inferred_max_neighbors": int(max_neighbors),
        "env_state_dim": int(last_state.shape[0]),
        "reset_mask_len": int(last_mask.shape[0]),
        "sources_tested": int(sources_tested),
        "sources_failed": [int(x) for x in sources_failed],
        "one_step_ok": bool(sources_tested > 0 and not sources_failed),
        "ok": bool(
            artifacts["region_demand"].shape == artifacts["region_supply"].shape
            and artifacts["region_demand"].shape[1] == k
            and artifacts["dispatch_duration_hours"].shape == (k, k)
            and artifacts["dispatch_reachable"].shape == (k, k)
            and edge_index.ndim == 2
            and edge_index.shape[1] == 2
            and artifacts["rl_edge_lengths"].shape[0] == edge_index.shape[0]
            and artifacts["rl_edge_speeds_kmh"].shape[0] == edge_index.shape[0]
            and last_state.shape[0] == 2 + 2 * max_neighbors + 1
            and last_mask.shape[0] == max_neighbors
            and sources_tested == int(np.count_nonzero(out_deg))
            and not sources_failed
        ),
    }


def contiguity_metrics(data_dir: str, membership: np.ndarray, k: int) -> dict:
    adj = np.load(f"{data_dir}/adjacency_matrix.npy").astype(bool)
    adj = adj | adj.T
    violations: list[int] = []
    for region_id in range(k):
        nodes = np.flatnonzero(membership == region_id)
        if nodes.size <= 1:
            continue
        node_set = set(int(x) for x in nodes)
        seen = {int(nodes[0])}
        queue = [int(nodes[0])]
        while queue:
            node = queue.pop(0)
            for nb in np.flatnonzero(adj[node]):
                nb = int(nb)
                if nb in node_set and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        if len(seen) != len(node_set):
            violations.append(region_id)
    return {
        "all_regions_contiguous": bool(not violations),
        "contiguity_violations": [int(x) for x in violations],
    }


def graph_policy_metrics(artifacts: dict, k: int) -> dict:
    edge_index = artifacts["rl_edge_index"]
    self_loop_edges = int(np.sum(edge_index[:, 0] == edge_index[:, 1])) if edge_index.size else 0
    out_degree = np.bincount(edge_index[:, 0], minlength=k) if edge_index.size else np.zeros(k, dtype=int)
    action_info = artifacts.get("action_info")
    action_info_edge_match = False
    if action_info is not None and not action_info.empty and "action_type" in action_info.columns:
        if {"src", "dst"}.issubset(action_info.columns) and len(action_info) == edge_index.shape[0]:
            csv_edges = action_info[["src", "dst"]].to_numpy(dtype=np.int32)
            action_info_edge_match = bool(np.array_equal(csv_edges, edge_index))
        action_type = action_info["action_type"].astype(str)
        topk_counts = (
            action_info[action_type == "osrm_topk"]
            .groupby("src")
            .size()
            .to_dict()
        )
        high_demand_counts = (
            action_info[action_type == "high_demand_connector"]
            .groupby("src")
            .size()
            .to_dict()
        )
        scc_counts = (
            action_info[action_type == "scc_high_demand_connector"]
            .groupby("src")
            .size()
            .to_dict()
        )
        connector_counts = (
            action_info[action_type.str.contains("connector")]
            .groupby("src")
            .size()
            .to_dict()
        )
    else:
        topk_counts = {}
        high_demand_counts = {}
        scc_counts = {}
        connector_counts = {}
    topk_values = list(topk_counts.values())
    high_demand_values = list(high_demand_counts.values())
    scc_values = list(scc_counts.values())
    return {
        "no_stay_action": bool(artifacts["meta"].get("no_stay_action", False)),
        "rl_self_loop_edges": self_loop_edges,
        "min_topk_edges_per_source": int(min(topk_values, default=0)),
        "max_topk_edges_per_source": int(max(topk_counts.values(), default=0)),
        "sources_with_topk_edges": int(len(topk_counts)),
        "topk_base_edges": int(sum(topk_values)),
        "min_high_demand_connectors_per_source": int(min(high_demand_values, default=0)),
        "max_high_demand_connectors_per_source": int(max(high_demand_values, default=0)),
        "sources_with_high_demand_connectors": int(len(high_demand_counts)),
        "high_demand_connector_edges": int(sum(high_demand_values)),
        "max_scc_connectors_per_source": int(max(scc_values, default=0)),
        "scc_connector_edges": int(sum(scc_values)),
        "connector_edges": int(sum(connector_counts.values())) if connector_counts else 0,
        "actual_action_edges": int(edge_index.shape[0]),
        "min_out_degree": int(out_degree.min()) if out_degree.size else 0,
        "max_out_degree": int(out_degree.max()) if out_degree.size else 0,
        "action_info_edge_match": action_info_edge_match,
    }


def topk_nearest_metrics(artifacts: dict, k: int) -> dict:
    action_info = artifacts.get("action_info")
    topk = int(artifacts["meta"].get("osrm_topk", 8))
    distances = artifacts["dispatch_distance_km"]
    durations = artifacts["dispatch_duration_hours"]
    reachable = artifacts["dispatch_reachable"]
    failures: list[dict] = []
    checked_edges = 0
    expected_edges = 0

    required_cols = {"src", "dst", "action_type", "action_rank"}
    if action_info is None or action_info.empty or not required_cols.issubset(action_info.columns):
        return {
            "ok": False,
            "checked_topk_edges": 0,
            "expected_topk_edges": int(k * topk),
            "failure_count": 1,
            "failures": [{"reason": "rl_action_info.csv is missing required top-k columns"}],
        }

    topk_rows = action_info[action_info["action_type"].astype(str) == "osrm_topk"].copy()
    for src in range(k):
        candidates = [
            dst for dst in range(k)
            if (
                dst != src
                and bool(reachable[src, dst])
                and np.isfinite(distances[src, dst])
                and np.isfinite(durations[src, dst])
            )
        ]
        candidates.sort(key=lambda dst: (float(distances[src, dst]), float(durations[src, dst]), int(dst)))
        expected = [int(dst) for dst in candidates[:topk]]
        expected_edges += len(expected)
        rows = topk_rows[topk_rows["src"].astype(int) == src].sort_values(["action_rank", "dst"])
        actual = [int(dst) for dst in rows["dst"].tolist()]
        ranks = [int(rank) for rank in rows["action_rank"].tolist()]
        checked_edges += len(actual)
        expected_ranks = list(range(1, min(topk, len(expected)) + 1))
        if len(expected) != topk or actual != expected or ranks != expected_ranks:
            failures.append(
                {
                    "src": int(src),
                    "expected_dsts": expected,
                    "actual_dsts": actual,
                    "expected_ranks": expected_ranks,
                    "actual_ranks": ranks,
                }
            )

    return {
        "ok": bool(not failures and checked_edges == k * topk and expected_edges == k * topk),
        "checked_topk_edges": int(checked_edges),
        "expected_topk_edges": int(k * topk),
        "failure_count": int(len(failures)),
        "failures": failures[:10],
    }


def action_info_numeric_metrics(artifacts: dict) -> dict:
    action_info = artifacts.get("action_info")
    required_cols = {"src", "dst", "distance_km", "duration_hours"}
    if action_info is None or action_info.empty or not required_cols.issubset(action_info.columns):
        return {
            "ok": False,
            "distance_match": False,
            "duration_match": False,
            "failure_reason": "rl_action_info.csv is missing distance/duration columns",
        }

    src = action_info["src"].to_numpy(dtype=np.int64)
    dst = action_info["dst"].to_numpy(dtype=np.int64)
    csv_distances = action_info["distance_km"].to_numpy(dtype=np.float64)
    csv_durations = action_info["duration_hours"].to_numpy(dtype=np.float64)
    matrix_distances = artifacts["dispatch_distance_km"][src, dst].astype(np.float64)
    matrix_durations = artifacts["dispatch_duration_hours"][src, dst].astype(np.float64)
    distance_match = bool(np.allclose(csv_distances, matrix_distances, rtol=1e-5, atol=1e-5))
    duration_match = bool(np.allclose(csv_durations, matrix_durations, rtol=1e-5, atol=1e-7))
    return {
        "ok": bool(distance_match and duration_match),
        "distance_match": distance_match,
        "duration_match": duration_match,
        "max_distance_abs_error": float(np.max(np.abs(csv_distances - matrix_distances))) if csv_distances.size else 0.0,
        "max_duration_abs_error": float(np.max(np.abs(csv_durations - matrix_durations))) if csv_durations.size else 0.0,
    }


def speed_mapping_semantics_metrics(artifacts: dict, base_edge_index: np.ndarray) -> dict:
    membership = artifacts["membership"].astype(np.int32)
    rl_edge_index = artifacts["rl_edge_index"].astype(np.int32)
    offsets = artifacts["rl_edge_speed_mapping_offsets"]
    indices = artifacts["rl_edge_speed_mapping_indices"]
    action_info = artifacts.get("action_info")
    src_regions = membership[base_edge_index[:, 0]]
    dst_regions = membership[base_edge_index[:, 1]]
    all_base = np.arange(base_edge_index.shape[0], dtype=np.int32)
    failures: list[dict] = []

    for rl_idx, (src_region, dst_region) in enumerate(rl_edge_index.astype(int)):
        selected = all_base[(src_regions == src_region) & (dst_regions == dst_region)]
        mapping_type = "direct_cross_superzone"
        if selected.size == 0:
            selected = all_base[(src_regions == src_region) | (dst_regions == dst_region)]
            mapping_type = "incident_superzone"
        if selected.size == 0:
            selected = all_base[src_regions == src_region]
            mapping_type = "source_outgoing"
        if selected.size == 0:
            selected = all_base[dst_regions == dst_region]
            mapping_type = "destination_incoming"
        if selected.size == 0:
            selected = all_base
            mapping_type = "global_fallback"

        actual = indices[offsets[rl_idx]: offsets[rl_idx + 1]]
        row_type = None
        if action_info is not None and not action_info.empty and "speed_mapping_type" in action_info.columns:
            row_type = str(action_info.iloc[rl_idx]["speed_mapping_type"])
        if not np.array_equal(actual.astype(np.int32), selected.astype(np.int32)) or row_type != mapping_type:
            failures.append(
                {
                    "rl_edge_id": int(rl_idx),
                    "src": int(src_region),
                    "dst": int(dst_region),
                    "expected_type": mapping_type,
                    "actual_type": row_type,
                    "expected_count": int(selected.size),
                    "actual_count": int(actual.size),
                }
            )

    return {
        "ok": bool(not failures),
        "checked_rl_edges": int(rl_edge_index.shape[0]),
        "failure_count": int(len(failures)),
        "failures": failures[:10],
    }


def speed_mapping_metrics(artifacts: dict) -> dict:
    offsets = artifacts["rl_edge_speed_mapping_offsets"]
    counts = np.diff(offsets)
    action_info = artifacts.get("action_info")
    mapping_counts = {}
    mapping_metadata_complete = False
    mapping_edge_count_matches_offsets = False
    if action_info is not None and not action_info.empty and "speed_mapping_type" in action_info.columns:
        mapping_counts = {
            str(key): int(value)
            for key, value in action_info["speed_mapping_type"].value_counts().to_dict().items()
        }
    if (
        action_info is not None
        and not action_info.empty
        and {"speed_mapping_type", "speed_mapping_edge_count"}.issubset(action_info.columns)
        and len(action_info) == counts.shape[0]
    ):
        type_complete = bool(action_info["speed_mapping_type"].notna().all())
        edge_counts = action_info["speed_mapping_edge_count"].to_numpy(dtype=float)
        count_complete = bool(np.isfinite(edge_counts).all() and np.all(edge_counts > 0))
        mapping_metadata_complete = bool(type_complete and count_complete)
        mapping_edge_count_matches_offsets = bool(
            count_complete and np.array_equal(edge_counts.astype(np.int64), counts.astype(np.int64))
        )
    return {
        "source": str(artifacts["meta"].get("rl_speed_profile_source")),
        "fallback_policy": str(artifacts["meta"].get("rl_speed_mapping_fallback")),
        "mapped_rl_edges": int(counts.shape[0]),
        "min_base_edges_per_rl_edge": int(counts.min()) if counts.size else 0,
        "mean_base_edges_per_rl_edge": float(counts.mean()) if counts.size else 0.0,
        "max_base_edges_per_rl_edge": int(counts.max()) if counts.size else 0,
        "mapping_type_counts": mapping_counts,
        "mapping_metadata_complete": mapping_metadata_complete,
        "mapping_edge_count_matches_offsets": mapping_edge_count_matches_offsets,
        "global_fallback_edges": int(mapping_counts.get("global_fallback", 0)),
    }


def partition_metrics(artifacts: dict) -> dict:
    info = artifacts["region_info"]
    demand = info["demand_weight"].to_numpy(dtype=float)
    non_reserved = info[info.get("is_reserved", 0) == 0]
    non_demand = non_reserved["demand_weight"].to_numpy(dtype=float)
    compact_col = "intra_osrm_median_km"
    compact_vals = (
        non_reserved[compact_col].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if compact_col in non_reserved.columns
        else np.array([], dtype=float)
    )
    return {
        "k": int(artifacts["meta"].get("num_superzones", artifacts["region_demand"].shape[1])),
        "num_base_zones": int(artifacts["membership"].shape[0]),
        "reserved_regions": int(info.get("is_reserved", 0).sum()) if "is_reserved" in info.columns else 0,
        "max_region_size": int(info["num_zones"].max()) if "num_zones" in info.columns else None,
        "mean_region_size": float(info["num_zones"].mean()) if "num_zones" in info.columns else None,
        "demand_cv_all": float(demand.std() / max(demand.mean(), 1.0)),
        "demand_cv_non_reserved": float(non_demand.std() / max(non_demand.mean(), 1.0)) if non_demand.size else 0.0,
        "intra_osrm_median_km_mean": float(np.mean(compact_vals)) if compact_vals.size else None,
        "intra_osrm_median_km_p95": float(np.percentile(compact_vals, 95)) if compact_vals.size else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate superzone dispatch and RL graph artifacts")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--superzone-dir", type=str, default="")
    p.add_argument("--slot", type=int, default=100)
    p.add_argument(
        "--slots",
        type=str,
        default="",
        help=(
            "Comma-separated dispatch smoke-test slots. "
            "Defaults to the primary slot plus representative and peak-imbalance slots."
        ),
    )
    p.add_argument("--no-strict", action="store_true", help="Print metrics but do not fail on validation gates.")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    artifacts = load_superzone_artifacts(args.data_dir, args.superzone_dir or None)
    demand = artifacts["region_demand"]
    supply = artifacts["region_supply"]
    slot = int(np.clip(args.slot, 0, demand.shape[0] - 1))
    dispatch_slots = select_dispatch_slots(demand, supply, slot, args.slots)
    demand_weights = demand.sum(axis=0)
    k = int(artifacts["meta"].get("num_superzones", demand.shape[1]))
    base_edge_index = np.load(f"{args.data_dir}/edge_index.npy").astype(np.int32)
    rl_reachable = reachable_matrix(artifacts["rl_edge_index"], k)
    rl_metrics = reachability_metrics(
        artifacts["rl_edge_index"],
        k,
        demand_weights,
    )
    dispatch_matrix, dispatch_metrics = validate_dispatch_reachability(
        demand[slot],
        supply[slot],
        artifacts["dispatch_duration_hours"],
        rl_reachable,
    )
    dispatch_slot_metrics = validate_dispatch_slots(
        demand,
        supply,
        artifacts["dispatch_duration_hours"],
        rl_reachable,
        dispatch_slots,
    )
    payload = {
        "slot": slot,
        "slots": dispatch_slots,
        "artifact_meta": {
            "cost_source": artifacts["meta"].get("cost_source"),
            "cost_matrix_origin": artifacts["meta"].get("cost_matrix_origin"),
            "osrm_topk": artifacts["meta"].get("osrm_topk"),
            "connector_count": artifacts["meta"].get("connector_count"),
            "rl_speed_profile_source": artifacts["meta"].get("rl_speed_profile_source"),
            "rl_speed_mapping_fallback": artifacts["meta"].get("rl_speed_mapping_fallback"),
            "connector_demand_source": artifacts["meta"].get("connector_demand_source"),
            "created_at": artifacts["meta"].get("created_at"),
        },
        "partition": partition_metrics(artifacts),
        "contiguity": contiguity_metrics(args.data_dir, artifacts["membership"], k),
        "graph_policy": graph_policy_metrics(artifacts, k),
        "topk_nearest": topk_nearest_metrics(artifacts, k),
        "action_info_numeric": action_info_numeric_metrics(artifacts),
        "speed_mapping": speed_mapping_metrics(artifacts),
        "speed_mapping_semantics": speed_mapping_semantics_metrics(artifacts, base_edge_index),
        "shape_action_smoke": shape_action_smoke(artifacts, k),
        "rl_action_graph": rl_metrics,
        "dispatch": dispatch_metrics,
        "dispatch_slots": dispatch_slot_metrics,
        "no_self_loop": {
            "rl_self_loop_edges": int(np.sum(artifacts["rl_edge_index"][:, 0] == artifacts["rl_edge_index"][:, 1])),
            "dispatch_self_loop_pairs": int(np.count_nonzero(np.diag(dispatch_matrix))),
            "dispatch_self_loop_vehicles": int(np.trace(dispatch_matrix)),
            "dispatch_cost_diagonal_zero": bool(np.allclose(np.diag(artifacts["dispatch_duration_hours"]), 0.0)),
        },
        "dispatch_finite_pair_rate": float(np.isfinite(artifacts["dispatch_duration_hours"]).mean()),
        "dispatch_reachable_pair_rate": float(artifacts["dispatch_reachable"].mean()),
    }
    offdiag = ~np.eye(k, dtype=bool)
    payload["dispatch_finite_offdiag_pair_rate"] = float(
        np.isfinite(artifacts["dispatch_duration_hours"])[offdiag].mean()
    )
    payload["dispatch_reachable_offdiag_pair_rate"] = float(
        artifacts["dispatch_reachable"][offdiag].mean()
    )
    base_edge_count = int(base_edge_index.shape[0])
    mapping_indices = artifacts["rl_edge_speed_mapping_indices"]
    mapping_offsets = artifacts["rl_edge_speed_mapping_offsets"]
    payload["artifact_numeric"] = {
        "rl_edge_lengths_positive_finite": bool(
            np.all(np.isfinite(artifacts["rl_edge_lengths"])) and np.all(artifacts["rl_edge_lengths"] > 0)
        ),
        "rl_edge_durations_positive_finite": bool(
            np.all(np.isfinite(artifacts["rl_edge_durations_hours"])) and np.all(artifacts["rl_edge_durations_hours"] > 0)
        ),
        "rl_edge_speeds_positive_finite": bool(
            np.all(np.isfinite(artifacts["rl_edge_speeds_kmh"])) and np.all(artifacts["rl_edge_speeds_kmh"] > 0)
        ),
        "dispatch_reachable_matches_finite_duration": bool(
            np.array_equal(artifacts["dispatch_reachable"], np.isfinite(artifacts["dispatch_duration_hours"]))
        ),
        "dispatch_duration_positive_offdiag": bool(
            np.all(artifacts["dispatch_duration_hours"][offdiag] > 0)
        ),
        "dispatch_distance_positive_offdiag": bool(
            np.all(artifacts["dispatch_distance_km"][offdiag] > 0)
        ),
        "speed_mapping_offsets_match_indices": bool(
            mapping_offsets.size == artifacts["rl_edge_index"].shape[0] + 1
            and mapping_offsets[0] == 0
            and np.all(np.diff(mapping_offsets) > 0)
            and mapping_offsets[-1] == mapping_indices.shape[0]
        ),
        "speed_mapping_indices_in_base_range": bool(
            mapping_indices.size > 0
            and np.all(mapping_indices >= 0)
            and np.all(mapping_indices < base_edge_count)
        ),
    }
    print(json.dumps(payload, indent=2))
    failures = []
    if payload["partition"]["k"] != 64:
        failures.append("K is not 64")
    if artifacts["meta"].get("cost_source") != "osrm_table":
        failures.append("dispatch cost_source is not osrm_table")
    if payload["graph_policy"]["rl_self_loop_edges"] != 0:
        failures.append("RL graph contains self-loops")
    if not payload["graph_policy"]["no_stay_action"]:
        failures.append("no_stay_action metadata is not true")
    if not payload["graph_policy"]["action_info_edge_match"]:
        failures.append("rl_action_info.csv does not align with rl_edge_index.npy")
    if payload["speed_mapping"]["source"] != "dynamic_stgat_edge_aggregation":
        failures.append("RL speed profile source is not dynamic STGAT edge aggregation")
    if payload["speed_mapping"]["min_base_edges_per_rl_edge"] <= 0:
        failures.append("one or more RL edges have no mapped STGAT speed edges")
    if payload["speed_mapping"]["mapped_rl_edges"] != payload["graph_policy"]["actual_action_edges"]:
        failures.append("speed mapping edge count does not match RL action edge count")
    if not payload["speed_mapping"]["mapping_metadata_complete"]:
        failures.append("speed mapping metadata is incomplete in rl_action_info.csv")
    if not payload["speed_mapping"]["mapping_edge_count_matches_offsets"]:
        failures.append("speed mapping edge counts do not match mapping offsets")
    if payload["speed_mapping"]["global_fallback_edges"] > 0:
        failures.append("speed mapping used global fallback edges")
    if not payload["speed_mapping_semantics"]["ok"]:
        failures.append("speed mapping indices/types do not match membership-derived mapping semantics")
    if payload["graph_policy"]["max_topk_edges_per_source"] > int(artifacts["meta"].get("osrm_topk", 8)):
        failures.append("osrm_topk edges exceed configured top-k")
    if payload["graph_policy"]["min_topk_edges_per_source"] != int(artifacts["meta"].get("osrm_topk", 8)):
        failures.append("not every source has exactly the configured top-k OSRM neighbor edges")
    if payload["graph_policy"]["sources_with_topk_edges"] != payload["partition"]["k"]:
        failures.append("not every source has top-k OSRM neighbor edges")
    if payload["graph_policy"]["topk_base_edges"] != payload["partition"]["k"] * int(artifacts["meta"].get("osrm_topk", 8)):
        failures.append("total top-k base edge count does not equal K * osrm_topk")
    if not payload["topk_nearest"]["ok"]:
        failures.append("osrm_topk edges are not exactly the OSRM-nearest rank-1..rank-k destinations")
    if not payload["action_info_numeric"]["ok"]:
        failures.append("rl_action_info.csv distance/duration values do not match dispatch matrices")
    connector_count = int(artifacts["meta"].get("connector_count", 0))
    if payload["graph_policy"]["min_high_demand_connectors_per_source"] != connector_count:
        failures.append("not every source has exactly the configured high-demand connector count")
    if payload["graph_policy"]["max_high_demand_connectors_per_source"] != connector_count:
        failures.append("high-demand connector count exceeds configured connector_count")
    if payload["graph_policy"]["sources_with_high_demand_connectors"] != payload["partition"]["k"]:
        failures.append("not every source has high-demand connector edges")
    if payload["graph_policy"]["high_demand_connector_edges"] != payload["partition"]["k"] * connector_count:
        failures.append("total high-demand connector count does not equal K * connector_count")
    if payload["graph_policy"]["actual_action_edges"] != (
        payload["graph_policy"]["topk_base_edges"] + payload["graph_policy"]["connector_edges"]
    ):
        failures.append("RL action edge count does not equal top-k plus connector edges")
    max_allowed_out_degree = (
        int(artifacts["meta"].get("osrm_topk", 8))
        + connector_count
        + payload["graph_policy"]["max_scc_connectors_per_source"]
    )
    if payload["graph_policy"]["max_out_degree"] > max_allowed_out_degree:
        failures.append("RL max out-degree exceeds configured top-k plus connector budget")
    if not payload["shape_action_smoke"]["ok"]:
        failures.append("shape/action smoke failed")
    if payload["partition"]["reserved_regions"] != 3:
        failures.append("reserved airport region count is not 3")
    if not payload["contiguity"]["all_regions_contiguous"]:
        failures.append("one or more superzones are not contiguous")
    if payload["partition"]["demand_cv_non_reserved"] > 2.5:
        failures.append("non-reserved demand CV exceeds 2.5")
    if payload["partition"]["intra_osrm_median_km_p95"] is None or payload["partition"]["intra_osrm_median_km_p95"] > 10.0:
        failures.append("OSRM compactness p95 is missing or above 10 km")
    if payload["rl_action_graph"]["num_scc"] != 1:
        failures.append("RL graph is not strongly connected")
    if payload["rl_action_graph"]["demand_weighted_reachability"] < 0.999:
        failures.append("demand-weighted reachability below 0.999")
    if payload["dispatch"]["reachable_pair_rate"] < 0.999:
        failures.append("dispatch OD finite-cost reachability below 0.999")
    if payload["dispatch"]["dispatch_pairs"] <= 0 or payload["dispatch"]["dispatch_vehicles"] <= 0:
        failures.append("dispatch smoke produced no OD pairs or vehicles")
    if payload["dispatch"]["rl_path_reachable_pair_rate"] < 0.999:
        failures.append("dispatch OD RL path reachability below 0.999")
    if payload["dispatch_slots"]["num_slots"] < 2:
        failures.append("dispatch validation must cover at least two slots")
    if payload["dispatch_slots"]["min_dispatch_pairs"] <= 0 or payload["dispatch_slots"]["min_dispatch_vehicles"] <= 0:
        failures.append("one or more dispatch validation slots produced no OD pairs or vehicles")
    if payload["dispatch_slots"]["min_reachable_pair_rate"] < 0.999:
        failures.append("multi-slot dispatch finite-cost reachability below 0.999")
    if payload["dispatch_slots"]["min_rl_path_reachable_pair_rate"] < 0.999:
        failures.append("multi-slot dispatch OD RL path reachability below 0.999")
    if payload["dispatch_slots"]["max_dispatch_self_loop_pairs"] != 0:
        failures.append("multi-slot dispatch smoke contains self-loop pairs")
    if payload["dispatch_slots"]["max_dispatch_self_loop_vehicles"] != 0:
        failures.append("multi-slot dispatch smoke contains self-loop vehicles")
    if payload["dispatch_finite_pair_rate"] < 0.999:
        failures.append("dense dispatch matrix is not fully finite")
    if payload["dispatch_finite_offdiag_pair_rate"] < 0.999:
        failures.append("dense dispatch off-diagonal matrix is not fully finite")
    if payload["dispatch_reachable_offdiag_pair_rate"] < 0.999:
        failures.append("dense dispatch off-diagonal reachability is below 0.999")
    if payload["no_self_loop"]["dispatch_self_loop_pairs"] != 0:
        failures.append("dispatch smoke contains self-loop pairs")
    if payload["no_self_loop"]["dispatch_self_loop_vehicles"] != 0:
        failures.append("dispatch smoke contains self-loop vehicles")
    if not payload["no_self_loop"]["dispatch_cost_diagonal_zero"]:
        failures.append("dispatch cost diagonal is not zero")
    if not all(payload["artifact_numeric"].values()):
        failures.append("artifact numeric quality check failed")
    if failures and not args.no_strict:
        print(json.dumps({"validation_failures": failures}, indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main(parse_args())
```

### stgat_model.py

```python
"""
stgat_model.py — Spatio-Temporal Graph Attention Network 預測模型

架構（對應論文 Section 4.2）：
  ┌─────────────────────────────────────────────────┐
  │  Node Stream (需求 / 空車)                       │
  │    Path-Fixed   : [GTCN → GAT(A_fixed, edge)]×L │
  │    Path-Adaptive: [GTCN → GAT(A_adp)]×L         │
  │    → GatedFusion → demand_head / supply_head     │
  ├─────────────────────────────────────────────────┤
  │  Edge Stream (道路速度)                          │
  │    [GTCN → GAT(line_graph)]×L                    │
  │    → speed_head                                  │
  └─────────────────────────────────────────────────┘

輸出：predicted demand, vacant taxis, road speeds for next p steps
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════
#  工具函式
# ════════════════════════════════════════════════════════════════

def build_line_graph_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """
    從邊列表建構線圖 (line graph) 邊列表。
    兩條有向邊 e_i=(u,v) 與 e_j=(v,w) 相鄰，表示線圖中的 i <- j。
    """
    dst = edge_index[:, 1]  # (|E|,)
    src = edge_index[:, 0]  # (|E|,)
    recv, send = torch.nonzero(dst.unsqueeze(1) == src.unsqueeze(0), as_tuple=True)
    return torch.stack([recv, send], dim=0).long()


# ════════════════════════════════════════════════════════════════
#  GTCN — Gated Temporal Convolutional Network
# ════════════════════════════════════════════════════════════════

class GTCNLayer(nn.Module):
    """
    單層門控時間卷積（Eq. 1-2）。
    output = gate ⊗ filter + (1 - gate) ⊗ residual
    其中 gate = σ(conv(x))
    """

    def __init__(
        self, in_c: int, out_c: int, kernel_size: int = 3, dilation: int = 1
    ) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.gate_conv = nn.Conv1d(in_c, out_c, kernel_size, dilation=dilation)
        self.filter_conv = nn.Conv1d(in_c, out_c, kernel_size, dilation=dilation)
        self.skip_conv = nn.Conv1d(in_c, out_c, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, C_in) → (B, N, T, C_out)"""
        B, N, T, C = x.shape
        x = x.reshape(B * N, T, C).permute(0, 2, 1)       # (B*N, C, T)
        x = F.pad(x, (self.pad, 0))                         # causal left-pad
        gate = torch.sigmoid(self.gate_conv(x))
        out = gate * self.filter_conv(x) + (1 - gate) * self.skip_conv(x)
        return out.permute(0, 2, 1).reshape(B, N, T, -1)


class GTCN(nn.Module):
    """多層 GTCN，含殘差連接與 LayerNorm。"""

    def __init__(
        self,
        in_c: int,
        hid_c: int,
        out_c: int,
        num_layers: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            ic = in_c if i == 0 else hid_c
            oc = hid_c if i < num_layers - 1 else out_c
            self.layers.append(GTCNLayer(ic, oc, kernel_size, dilation=2 ** i))
            self.norms.append(nn.LayerNorm(oc))
        self.skip = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, T, C_in) → (B, N, T, C_out)"""
        res = self.skip(x)
        for layer, norm in zip(self.layers, self.norms):
            x = F.relu(norm(layer(x)))
        return x + res


# ════════════════════════════════════════════════════════════════
#  GAT — Graph Attention Layer
# ════════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    """
    通用圖注意力層，支援可選的邊特徵（Eq. 3-7 / 8-12）。

    若 edge_in > 0 → 區域級 GAT（含邊特徵 len, speed）
    若 edge_in == 0 → 邊級 GAT（不含額外邊特徵）
    """

    def __init__(
        self,
        in_c: int,
        d_out: int,
        num_heads: int = 4,
        edge_in: int = 0,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_out = d_out
        self.concat = concat
        self.has_edge = edge_in > 0
        total = d_out * num_heads

        self.W_q = nn.Linear(in_c, total, bias=False)
        self.W_k = nn.Linear(in_c, total, bias=False)
        self.W_v = nn.Linear(in_c, total, bias=False)

        self.a_q = nn.Parameter(torch.empty(1, 1, num_heads, d_out))
        self.a_k = nn.Parameter(torch.empty(1, 1, num_heads, d_out))
        nn.init.xavier_normal_(self.a_q)
        nn.init.xavier_normal_(self.a_k)

        if self.has_edge:
            self.W_e = nn.Linear(edge_in, total, bias=False)
            self.a_e = nn.Parameter(torch.empty(1, 1, 1, num_heads, d_out))
            nn.init.xavier_normal_(self.a_e)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h         : (B, N, C_in)
        adj       : (N, N) binary mask
        edge_feat : (B, N, N, edge_in) or None
        return    : (B, N, H*d_out) if concat else (B, N, d_out)
        """
        B, N, _ = h.shape
        H, D = self.num_heads, self.d_out

        q = self.W_q(h).view(B, N, H, D)
        k = self.W_k(h).view(B, N, H, D)
        v = self.W_v(h).view(B, N, H, D)

        # additive attention: a_q·q_i + a_k·k_j (+ a_e·e_{i,j})
        s_q = (q * self.a_q).sum(-1)  # (B, N, H)
        s_k = (k * self.a_k).sum(-1)  # (B, N, H)
        scores = s_q.unsqueeze(2) + s_k.unsqueeze(1)  # (B, N, N, H)

        if self.has_edge and edge_feat is not None:
            e = self.W_e(edge_feat).view(B, N, N, H, D)
            s_e = (e * self.a_e).sum(-1)  # (B, N, N, H)
            scores = scores + s_e

        scores = F.leaky_relu(scores, 0.2)

        # mask non-neighbors
        mask = adj.unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        scores = scores.masked_fill(mask == 0, -1e9)

        alpha = F.softmax(scores, dim=2)  # (B, N, N, H)
        alpha = torch.nan_to_num(alpha, nan=0.0)

        # aggregate: s'_i = Σ_j α_{i,j} · v_j
        out = torch.einsum("bnjh,bnjhd->bnhd", alpha, v.unsqueeze(1).expand(-1, N, -1, -1, -1))

        if self.concat:
            out = out.reshape(B, N, H * D)
        else:
            out = out.mean(dim=2)

        return F.elu(out)


class SparseGATLayer(nn.Module):
    """
    Sparse GAT on an explicit edge list.

    Instead of materializing a dense NxN attention matrix and masking most
    entries away, this layer only computes attention on real graph neighbors.
    """

    def __init__(
        self,
        in_c: int,
        d_out: int,
        num_heads: int = 4,
        edge_in: int = 0,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_out = d_out
        self.concat = concat
        self.has_edge = edge_in > 0
        total = d_out * num_heads

        self.W_q = nn.Linear(in_c, total, bias=False)
        self.W_k = nn.Linear(in_c, total, bias=False)
        self.W_v = nn.Linear(in_c, total, bias=False)

        self.a_q = nn.Parameter(torch.empty(1, 1, num_heads, d_out))
        self.a_k = nn.Parameter(torch.empty(1, 1, num_heads, d_out))
        nn.init.xavier_normal_(self.a_q)
        nn.init.xavier_normal_(self.a_k)

        if self.has_edge:
            self.W_e = nn.Linear(edge_in, total, bias=False)
            self.a_e = nn.Parameter(torch.empty(1, 1, num_heads, d_out))
            nn.init.xavier_normal_(self.a_e)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h         : (B, N, C_in)
        edge_index: (2, L) with [recv, send]
        edge_feat : (B, L, edge_in) or None
        return    : (B, N, H*d_out) if concat else (B, N, d_out)
        """
        B, N, _ = h.shape
        H, D = self.num_heads, self.d_out
        recv = edge_index[0]
        send = edge_index[1]
        num_edges = recv.shape[0]

        q = self.W_q(h).view(B, N, H, D)
        k = self.W_k(h).view(B, N, H, D)
        v = self.W_v(h).view(B, N, H, D)

        q_recv = q[:, recv]  # (B, L, H, D)
        k_send = k[:, send]  # (B, L, H, D)
        scores = (q_recv * self.a_q).sum(-1) + (k_send * self.a_k).sum(-1)
        if self.has_edge and edge_feat is not None:
            e = self.W_e(edge_feat).view(B, num_edges, H, D)
            scores = scores + (e * self.a_e).sum(-1)
        scores = F.leaky_relu(scores, 0.2).float()

        recv_index = recv.view(1, num_edges, 1).expand(B, num_edges, H)
        max_scores = torch.full(
            (B, N, H),
            -torch.inf,
            device=h.device,
            dtype=scores.dtype,
        )
        max_scores.scatter_reduce_(1, recv_index, scores, reduce="amax", include_self=True)

        attn_logits = scores - max_scores.gather(1, recv_index)
        attn_exp = torch.exp(attn_logits)
        denom = torch.zeros((B, N, H), device=h.device, dtype=attn_exp.dtype)
        denom.scatter_add_(1, recv_index, attn_exp)
        alpha = attn_exp / denom.gather(1, recv_index).clamp_min(1e-12)
        alpha = torch.nan_to_num(alpha, nan=0.0).to(v.dtype)

        messages = alpha.unsqueeze(-1) * v[:, send]
        out = torch.zeros((B, N, H, D), device=h.device, dtype=messages.dtype)
        out.scatter_add_(
            1,
            recv.view(1, num_edges, 1, 1).expand(B, num_edges, H, D),
            messages,
        )

        if self.concat:
            out = out.reshape(B, N, H * D)
        else:
            out = out.mean(dim=2)

        return F.elu(out)


# ════════════════════════════════════════════════════════════════
#  Gated Fusion（Eq. 13-14）
# ════════════════════════════════════════════════════════════════

class GatedFusion(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(dim, dim)
        self.W2 = nn.Linear(dim, dim)
        self.W3 = nn.Linear(dim, dim)

    def forward(self, h_fixed: torch.Tensor, h_adp: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.W1(h_fixed + h_adp))
        return torch.tanh(gate * self.W2(h_fixed) + (1 - gate) * self.W3(h_adp))


# ════════════════════════════════════════════════════════════════
#  STGATPredictor — 完整預測模型
# ════════════════════════════════════════════════════════════════

class STGATPredictor(nn.Module):
    """
    Parameters
    ----------
    num_nodes       : N — 區域數
    edge_index      : (|E|, 2) long — 有向邊列表
    edge_lengths    : (|E|,) float — 靜態路長
    adj_matrix      : (N, N) float — 物理鄰接 (0/1)
    hidden_dim      : 隱藏層維度（也是 GTCN 輸出與 GAT 的 heads × d_out）
    num_heads       : 注意力頭數
    num_st_blocks   : ST-Block 堆疊數
    pred_horizon    : 預測未來 p 步
    hist_len        : 歷史窗口長度 h（僅用於說明，不影響模型定義）
    adaptive_emb    : 自適應鄰接 embedding 維度
    """

    def __init__(
        self,
        num_nodes: int,
        edge_index: torch.Tensor,
        edge_lengths: torch.Tensor,
        adj_matrix: torch.Tensor,
        *,
        hidden_dim: int = 32,
        num_heads: int = 4,
        num_st_blocks: int = 2,
        num_gtcn_layers: int = 2,
        kernel_size: int = 3,
        pred_horizon: int = 3,
        adaptive_emb: int = 10,
        node_feat_dim: int = 2,
        edge_feat_dim: int = 2,
    ) -> None:
        super().__init__()
        N = num_nodes
        d_per_head = hidden_dim // num_heads
        assert hidden_dim == d_per_head * num_heads, "hidden_dim must be divisible by num_heads"

        # ── 圖結構（不需要梯度） ──
        self.register_buffer("edge_index", edge_index.long())
        self.register_buffer("edge_lengths", edge_lengths.float())
        adj_with_self = (adj_matrix + torch.eye(N)).clamp(max=1.0)
        self.register_buffer("adj_fixed", adj_with_self)
        self.register_buffer("adj_full", torch.ones(N, N))
        fixed_recv, fixed_send = torch.nonzero(adj_with_self > 0, as_tuple=True)
        self.register_buffer("fixed_edge_index", torch.stack([fixed_recv, fixed_send], dim=0).long())
        self.register_buffer("line_edge_index", build_line_graph_edge_index(edge_index))
        edge_lookup = torch.full((N, N), -1, dtype=torch.long)
        edge_lookup[edge_index[:, 0].long(), edge_index[:, 1].long()] = torch.arange(
            edge_index.shape[0],
            dtype=torch.long,
        )
        fixed_edge_ids = edge_lookup[fixed_recv, fixed_send]
        fixed_edge_lengths = torch.zeros(fixed_edge_ids.shape[0], dtype=torch.float32)
        fixed_has_road = fixed_edge_ids >= 0
        fixed_edge_lengths[fixed_has_road] = edge_lengths.float()[fixed_edge_ids[fixed_has_road]]
        self.register_buffer("fixed_edge_ids", fixed_edge_ids)
        self.register_buffer("fixed_edge_lengths", fixed_edge_lengths)
        self.register_buffer("fixed_edge_has_road", fixed_has_road)

        # ── 自適應鄰接 ──
        self.emb_src = nn.Parameter(torch.randn(N, adaptive_emb) * 0.1)
        self.emb_dst = nn.Parameter(torch.randn(N, adaptive_emb) * 0.1)

        # ── 輸入投影 ──
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_proj = nn.Linear(1, hidden_dim)

        # ── Node path — fixed topology ──
        self.n_gtcn_fix = nn.ModuleList()
        self.n_gat_fix = nn.ModuleList()
        for _ in range(num_st_blocks):
            self.n_gtcn_fix.append(
                GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size)
            )
            self.n_gat_fix.append(
                SparseGATLayer(hidden_dim, d_per_head, num_heads, edge_in=edge_feat_dim, concat=True)
            )

        # ── Node path — adaptive topology ──
        self.n_gtcn_adp = nn.ModuleList()
        self.n_gat_adp = nn.ModuleList()
        for _ in range(num_st_blocks):
            self.n_gtcn_adp.append(
                GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size)
            )
            self.n_gat_adp.append(
                GATLayer(hidden_dim, d_per_head, num_heads, edge_in=0, concat=True)
            )

        # ── Fusion ──
        self.fusion = GatedFusion(hidden_dim)

        # ── Edge path ──
        self.e_gtcn = nn.ModuleList()
        self.e_gat = nn.ModuleList()
        for _ in range(num_st_blocks):
            self.e_gtcn.append(
                GTCN(hidden_dim, hidden_dim, hidden_dim, num_gtcn_layers, kernel_size)
            )
            self.e_gat.append(
                SparseGATLayer(hidden_dim, d_per_head, num_heads, concat=True)
            )

        # ── 輸出頭 ──
        self.demand_head = nn.Linear(hidden_dim, pred_horizon)
        self.supply_head = nn.Linear(hidden_dim, pred_horizon)
        self.speed_head = nn.Linear(hidden_dim, pred_horizon)

    # ── helpers ────────────────────────────────────────────────

    def _adaptive_adj(self) -> torch.Tensor:
        return F.softmax(F.relu(self.emb_src @ self.emb_dst.T), dim=1)

    def _fixed_edge_feat(self, speed: torch.Tensor) -> torch.Tensor:
        """
        speed : (BT, |E|)  — 每條邊在某時間步的速度
        return: (BT, |A_fixed|, 2) — [road_length, speed]
        """
        BT = speed.shape[0]
        num_fixed_edges = self.fixed_edge_ids.shape[0]
        feat = torch.zeros(BT, num_fixed_edges, 2, device=speed.device, dtype=speed.dtype)
        feat[:, :, 0] = self.fixed_edge_lengths.to(speed.dtype).unsqueeze(0)
        feat[:, self.fixed_edge_has_road, 1] = speed[:, self.fixed_edge_ids[self.fixed_edge_has_road]]
        return feat

    def _run_fixed_node_path(
        self,
        node_h: torch.Tensor,
        speed_seq: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed-topology node path with sparse edge-aware attention."""
        B, N, T, _ = node_h.shape
        x = node_h
        sp_flat = speed_seq.permute(0, 2, 1).reshape(B * T, -1)
        edge_feat = self._fixed_edge_feat(sp_flat)
        for gtcn, gat in zip(self.n_gtcn_fix, self.n_gat_fix):
            x = gtcn(x)                                              # (B, N, T, C)
            x_flat = x.permute(0, 2, 1, 3).reshape(B * T, N, -1)    # (BT, N, C)
            x_flat = gat(x_flat, self.fixed_edge_index, edge_feat)   # (BT, N, C)
            x = x_flat.reshape(B, T, N, -1).permute(0, 2, 1, 3)     # (B, N, T, C)
        return x[:, :, -1, :]                                        # (B, N, C)

    def _run_adaptive_node_path(
        self,
        node_h: torch.Tensor,
    ) -> torch.Tensor:
        """Adaptive node path that keeps the original dense all-pairs logic."""
        B, N, T, _ = node_h.shape
        x = node_h
        for gtcn, gat in zip(self.n_gtcn_adp, self.n_gat_adp):
            x = gtcn(x)                                              # (B, N, T, C)
            x_flat = x.permute(0, 2, 1, 3).reshape(B * T, N, -1)    # (BT, N, C)
            x_flat = gat(x_flat, self.adj_full)                      # (BT, N, C)
            x = x_flat.reshape(B, T, N, -1).permute(0, 2, 1, 3)     # (B, N, T, C)
        return x[:, :, -1, :]                                        # (B, N, C)

    # ── forward ───────────────────────────────────────────────

    def forward(
        self, node_seq: torch.Tensor, speed_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        node_seq  : (B, N, T, 2) — [demand, vacant] 歷史序列
        speed_seq : (B, |E|, T)  — 歷史邊速度

        Returns
        -------
        demand_pred : (B, N, p)
        supply_pred : (B, N, p)
        speed_pred  : (B, |E|, p)
        """
        B, N, T, _ = node_seq.shape
        nE = speed_seq.shape[1]

        node_h = self.node_proj(node_seq)                            # (B, N, T, C)
        edge_h = self.edge_proj(speed_seq.unsqueeze(-1))             # (B, |E|, T, C)

        # ── Node: fixed path ──
        h_fix = self._run_fixed_node_path(node_h, speed_seq)

        # ── Node: adaptive path ──
        h_adp = self._run_adaptive_node_path(node_h)

        # ── Fusion ──
        h_node = self.fusion(h_fix, h_adp)                          # (B, N, C)

        # ── Edge path ──
        x_e = edge_h
        for gtcn, gat in zip(self.e_gtcn, self.e_gat):
            x_e = gtcn(x_e)                                         # (B, |E|, T, C)
            x_flat = x_e.permute(0, 2, 1, 3).reshape(B * T, nE, -1)
            x_flat = gat(x_flat, self.line_edge_index)
            x_e = x_flat.reshape(B, T, nE, -1).permute(0, 2, 1, 3)
        h_edge = x_e[:, :, -1, :]                                   # (B, |E|, C)

        # ── 輸出 ──
        demand_pred = self.demand_head(h_node)                       # (B, N, p)
        supply_pred = self.supply_head(h_node)                       # (B, N, p)
        speed_pred = self.speed_head(h_edge)                         # (B, |E|, p)

        return demand_pred, supply_pred, speed_pred
```

### predictor_normalization.py

```python
from __future__ import annotations

from typing import Any

import numpy as np

NORMALIZATION_EPS = 1e-6


def _ensure_nonnegative(name: str, values: np.ndarray) -> None:
    min_value = float(np.min(values))
    if min_value < 0:
        raise ValueError(
            f"{name} contains negative values (min={min_value:.6f}), "
            "but log1p normalization expects nonnegative inputs."
        )


def _safe_std(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.where(arr < NORMALIZATION_EPS, 1.0, arr).astype(np.float32)


def build_normalization_stats(
    node_features: np.ndarray,
    edge_speeds: np.ndarray,
    train_time_mask: np.ndarray,
) -> dict[str, Any]:
    if node_features.ndim != 3 or node_features.shape[-1] < 2:
        raise ValueError(
            "node_features must have shape (T, N, C) with demand/supply in the first two channels."
        )
    if edge_speeds.ndim != 2:
        raise ValueError("edge_speeds must have shape (T, |E|).")

    train_time_mask = np.asarray(train_time_mask, dtype=bool)
    if train_time_mask.ndim != 1 or train_time_mask.shape[0] != node_features.shape[0]:
        raise ValueError("train_time_mask must be a 1D boolean array aligned with the time axis.")
    if not np.any(train_time_mask):
        raise ValueError("train_time_mask does not select any training time steps.")

    demand = np.asarray(node_features[..., 0], dtype=np.float32)
    supply = np.asarray(node_features[..., 1], dtype=np.float32)
    speed = np.asarray(edge_speeds, dtype=np.float32)

    _ensure_nonnegative("demand", demand)
    _ensure_nonnegative("supply", supply)

    demand_log = np.log1p(demand).astype(np.float32)
    supply_log = np.log1p(supply).astype(np.float32)

    demand_train = demand_log[train_time_mask]
    supply_train = supply_log[train_time_mask]
    speed_train = speed[train_time_mask]

    return {
        "demand": {
            "transform": "log1p_zscore",
            "mean": float(demand_train.mean()),
            "std": float(max(float(demand_train.std()), NORMALIZATION_EPS)),
        },
        "supply": {
            "transform": "log1p_zscore",
            "mean": float(supply_train.mean()),
            "std": float(max(float(supply_train.std()), NORMALIZATION_EPS)),
        },
        "speed": {
            "transform": "per_edge_zscore",
            "mean": speed_train.mean(axis=0).astype(np.float32),
            "std": _safe_std(speed_train.std(axis=0)),
        },
    }


def normalize_node_features(
    node_features: np.ndarray,
    normalization_stats: dict[str, Any] | None,
) -> np.ndarray:
    arr = np.asarray(node_features, dtype=np.float32).copy()
    if normalization_stats is None:
        return arr
    if arr.shape[-1] < 2:
        raise ValueError("node_features must include demand and supply channels in the last dimension.")

    demand_stats = normalization_stats["demand"]
    supply_stats = normalization_stats["supply"]
    arr[..., 0] = (
        np.log1p(arr[..., 0]) - float(demand_stats["mean"])
    ) / float(demand_stats["std"])
    arr[..., 1] = (
        np.log1p(arr[..., 1]) - float(supply_stats["mean"])
    ) / float(supply_stats["std"])
    return arr.astype(np.float32)


def normalize_speed_features(
    speed_values: np.ndarray,
    normalization_stats: dict[str, Any] | None,
    *,
    edge_axis: int,
) -> np.ndarray:
    arr = np.asarray(speed_values, dtype=np.float32).copy()
    if normalization_stats is None:
        return arr

    speed_stats = normalization_stats["speed"]
    mean = np.asarray(speed_stats["mean"], dtype=np.float32)
    std = np.asarray(speed_stats["std"], dtype=np.float32)
    axis = edge_axis % arr.ndim
    shape = [1] * arr.ndim
    shape[axis] = mean.shape[0]
    return ((arr - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32)


def denormalize_count_values(
    normalized_values: np.ndarray,
    normalization_stats: dict[str, Any] | None,
    *,
    task: str,
) -> np.ndarray:
    arr = np.asarray(normalized_values, dtype=np.float32)
    if normalization_stats is None:
        return arr

    task_stats = normalization_stats[task]
    return np.expm1(arr * float(task_stats["std"]) + float(task_stats["mean"])).astype(np.float32)


def denormalize_speed_values(
    normalized_values: np.ndarray,
    normalization_stats: dict[str, Any] | None,
    *,
    edge_axis: int,
) -> np.ndarray:
    arr = np.asarray(normalized_values, dtype=np.float32)
    if normalization_stats is None:
        return arr

    speed_stats = normalization_stats["speed"]
    mean = np.asarray(speed_stats["mean"], dtype=np.float32)
    std = np.asarray(speed_stats["std"], dtype=np.float32)
    axis = edge_axis % arr.ndim
    shape = [1] * arr.ndim
    shape[axis] = mean.shape[0]
    return (arr * std.reshape(shape) + mean.reshape(shape)).astype(np.float32)


def serialize_normalization_stats(
    normalization_stats: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if normalization_stats is None:
        return None
    return {
        "demand": {
            "transform": normalization_stats["demand"]["transform"],
            "mean": float(normalization_stats["demand"]["mean"]),
            "std": float(normalization_stats["demand"]["std"]),
        },
        "supply": {
            "transform": normalization_stats["supply"]["transform"],
            "mean": float(normalization_stats["supply"]["mean"]),
            "std": float(normalization_stats["supply"]["std"]),
        },
        "speed": {
            "transform": normalization_stats["speed"]["transform"],
            "mean": np.asarray(normalization_stats["speed"]["mean"], dtype=np.float32).tolist(),
            "std": np.asarray(normalization_stats["speed"]["std"], dtype=np.float32).tolist(),
        },
    }


def load_normalization_stats(
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "demand": {
            "transform": str(payload["demand"]["transform"]),
            "mean": float(payload["demand"]["mean"]),
            "std": float(payload["demand"]["std"]),
        },
        "supply": {
            "transform": str(payload["supply"]["transform"]),
            "mean": float(payload["supply"]["mean"]),
            "std": float(payload["supply"]["std"]),
        },
        "speed": {
            "transform": str(payload["speed"]["transform"]),
            "mean": np.asarray(payload["speed"]["mean"], dtype=np.float32),
            "std": _safe_std(np.asarray(payload["speed"]["std"], dtype=np.float32)),
        },
    }
```

### requirements.txt

```text
# Core numerical / ML
numpy>=1.24
pandas>=2.0
torch>=2.0

# Geospatial processing
geopandas>=0.14
pyogrio>=0.8
pyproj>=3.6
shapely>=2.0

# Data IO / plotting
matplotlib>=3.8
pyarrow>=14.0
```

### environment.yml

```yaml
name: STDR
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip>=24
  - setuptools>=68
  - wheel
  - pip:
      - -r requirements.txt
```

