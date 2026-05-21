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
