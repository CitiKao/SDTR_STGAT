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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 CUDA，但目前環境沒有可用 GPU。")
    return device


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
    device = resolve_device(args.device)
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
        if args.allow_random_predictor:
            print(f"警告: {pred_ckpt} 不存在，使用隨機權重")
        else:
            raise FileNotFoundError(
                f"找不到 predictor checkpoint: {pred_ckpt}. "
                "若只是 smoke demo，可加 --allow-random-predictor。"
            )

    # ── 載入 router ──
    max_nb = infer_max_neighbors(
        routing_subgraph["edge_index"], routing_N, minimum=args.max_neighbors
    )
    env_state_dim = 2 + 2 * max_nb + 1
    router = DoubleDQNAgent(
        num_nodes=routing_N,
        max_neighbors=max_nb,
        state_dim=env_state_dim,
        embed_dim=args.router_embed_dim,
        hidden_dim=args.router_hidden_dim,
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
        if args.allow_random_router:
            print(f"警告: {router_ckpt} 不存在，使用隨機 router")
        else:
            raise FileNotFoundError(
                f"找不到 router checkpoint: {router_ckpt}. "
                "若只是 smoke demo，可加 --allow-random-router。"
            )

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
    p.add_argument(
        "--allow-random-predictor",
        action="store_true",
        help="Allow demo to run with randomly initialized predictor if checkpoint is missing.",
    )
    p.add_argument(
        "--allow-random-router",
        action="store_true",
        help="Allow demo to run with randomly initialized router if checkpoint is missing.",
    )
    p.add_argument("--router-embed-dim", type=int, default=32)
    p.add_argument("--router-hidden-dim", type=int, default=128)
    return p.parse_args()


if __name__ == "__main__":
    demo(parse_args())
