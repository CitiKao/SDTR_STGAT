from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

try:
    import osmnx as ox
except ModuleNotFoundError:  # pragma: no cover - exercised through runtime guard
    ox = None

from external_speed_benchmarks.sensor_dataset_utils import (
    _build_pseudo_edge_topology_summary,
    _cluster_points_by_radius,
    haversine_distance_matrix_km,
    normalize_sensor_ids,
)


DEFAULT_OSM_DRIVE_PADDING_KM = 5.0


def require_osmnx() -> None:
    if ox is None:
        raise ModuleNotFoundError(
            "osmnx is required for external-map road-graph preparation. "
            "Install it in the active environment before using map-backed graph builders."
        )


def _stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _compute_bbox(coords: np.ndarray, *, padding_km: float) -> tuple[float, float, float, float]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2) as [latitude, longitude].")
    lat_min = float(coords[:, 0].min())
    lat_max = float(coords[:, 0].max())
    lon_min = float(coords[:, 1].min())
    lon_max = float(coords[:, 1].max())
    mean_lat = float(coords[:, 0].mean())
    lat_pad = float(padding_km) / 111.32
    lon_pad = float(padding_km) / (111.32 * max(np.cos(np.radians(mean_lat)), 1e-6))
    left = lon_min - lon_pad
    bottom = lat_min - lat_pad
    right = lon_max + lon_pad
    top = lat_max + lat_pad
    return left, bottom, right, top


def _edge_geometry(
    graph: nx.MultiDiGraph,
    u: int,
    v: int,
    key: int,
    data: dict[str, Any] | None = None,
) -> LineString:
    edge_data = data if data is not None else graph.get_edge_data(u, v, key)
    if edge_data is None:
        raise KeyError((u, v, key))
    geometry = edge_data.get("geometry")
    if geometry is not None:
        return geometry
    return LineString(
        [
            (float(graph.nodes[u]["x"]), float(graph.nodes[u]["y"])),
            (float(graph.nodes[v]["x"]), float(graph.nodes[v]["y"])),
        ]
    )


def load_or_build_osm_drive_snapshot(
    sensor_frame: pd.DataFrame,
    *,
    dataset_label: str,
    cache_root: str | Path,
    padding_km: float = DEFAULT_OSM_DRIVE_PADDING_KM,
    simplify: bool = True,
    retain_all: bool = True,
) -> dict[str, Any]:
    require_osmnx()
    cache_root = Path(cache_root)
    sensor_frame = sensor_frame.loc[:, ["sensor_id", "latitude", "longitude"]].copy()
    sensor_frame["sensor_id"] = normalize_sensor_ids(sensor_frame["sensor_id"])
    sensor_payload = [
        {
            "sensor_id": str(row.sensor_id),
            "latitude": round(float(row.latitude), 7),
            "longitude": round(float(row.longitude), 7),
        }
        for row in sensor_frame.itertuples(index=False)
    ]
    bbox = _compute_bbox(
        sensor_frame[["latitude", "longitude"]].to_numpy(dtype=np.float64),
        padding_km=float(padding_km),
    )
    cache_key = _stable_json_hash(
        {
            "dataset_label": str(dataset_label),
            "provider": "osm_drive",
            "padding_km": round(float(padding_km), 6),
            "simplify": bool(simplify),
            "retain_all": bool(retain_all),
            "sensor_payload": sensor_payload,
        }
    )[:16]
    snapshot_dir = cache_root / str(dataset_label) / f"osm_drive_{cache_key}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    graph_path = snapshot_dir / "drive_projected.graphml"
    manifest_path = snapshot_dir / "snapshot_manifest.json"

    if graph_path.exists():
        graph = ox.io.load_graphml(graph_path)
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        return {
            "graph": graph,
            "snapshot_dir": snapshot_dir,
            "graph_path": graph_path,
            "manifest_path": manifest_path,
            "manifest": manifest,
            "cache_key": cache_key,
        }

    old_use_cache = getattr(ox.settings, "use_cache", True)
    ox.settings.use_cache = True
    try:
        graph_unprojected = ox.graph.graph_from_bbox(
            bbox,
            network_type="drive",
            simplify=bool(simplify),
            retain_all=bool(retain_all),
            truncate_by_edge=True,
        )
    finally:
        ox.settings.use_cache = old_use_cache
    if graph_unprojected.number_of_nodes() == 0 or graph_unprojected.number_of_edges() == 0:
        raise ValueError(f"No drivable OSM graph could be downloaded for {dataset_label}.")
    graph = ox.project_graph(graph_unprojected)
    ox.io.save_graphml(graph, graph_path)
    manifest = {
        "dataset_label": str(dataset_label),
        "provider": "osm_drive",
        "cache_key": cache_key,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "graph_path": str(graph_path),
        "bbox_left_bottom_right_top": [float(value) for value in bbox],
        "padding_km": float(padding_km),
        "simplify": bool(simplify),
        "retain_all": bool(retain_all),
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "crs": str(graph.graph.get("crs", "")),
        "sensor_count": int(sensor_frame.shape[0]),
        "sensor_payload_sha256": _stable_json_hash({"sensor_payload": sensor_payload}),
        "osmnx_version": getattr(ox, "__version__", "unknown"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "graph": graph,
        "snapshot_dir": snapshot_dir,
        "graph_path": graph_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "cache_key": cache_key,
    }


def _graph_transformers(graph: nx.MultiDiGraph) -> tuple[Transformer, Transformer]:
    graph_crs = CRS.from_user_input(graph.graph.get("crs"))
    to_graph = Transformer.from_crs("EPSG:4326", graph_crs, always_xy=True)
    to_geo = Transformer.from_crs(graph_crs, "EPSG:4326", always_xy=True)
    return to_graph, to_geo


def _build_match_payload(
    graph: nx.MultiDiGraph,
    *,
    point_id: Any,
    latitude: float,
    longitude: float,
    projected_x: float,
    projected_y: float,
    edge_id: tuple[int, int, int],
    snap_distance_m: float,
    to_geo: Transformer,
) -> dict[str, Any]:
    u, v, key = edge_id
    point = Point(float(projected_x), float(projected_y))
    data = graph.get_edge_data(u, v, key)
    line = _edge_geometry(graph, u, v, key, data)
    line_length_m = float(max(line.length, 1e-6))
    pos_m = float(np.clip(line.project(point), 0.0, line_length_m))
    snapped_point = line.interpolate(pos_m)
    snapped_lon, snapped_lat = to_geo.transform(float(snapped_point.x), float(snapped_point.y))
    payload = {
        "point_id": point_id,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "projected_x_m": float(projected_x),
        "projected_y_m": float(projected_y),
        "primary_u": int(u),
        "primary_v": int(v),
        "primary_key": int(key),
        "primary_edge_length_m": line_length_m,
        "primary_pos_m_from_u": pos_m,
        "primary_remaining_m_to_v": float(max(line_length_m - pos_m, 0.0)),
        "primary_snap_distance_m": float(snap_distance_m),
        "snapped_latitude": float(snapped_lat),
        "snapped_longitude": float(snapped_lon),
    }
    reverse_payload = {
        "reverse_available": False,
        "reverse_u": -1,
        "reverse_v": -1,
        "reverse_key": -1,
        "reverse_edge_length_m": 0.0,
        "reverse_pos_m_from_u": 0.0,
        "reverse_remaining_m_to_v": 0.0,
    }
    if graph.has_edge(v, u):
        best_key = None
        best_data = None
        best_distance = float("inf")
        for candidate_key, candidate_data in graph.get_edge_data(v, u).items():
            candidate_line = _edge_geometry(graph, v, u, candidate_key, candidate_data)
            candidate_distance = float(candidate_line.distance(point))
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_key = int(candidate_key)
                best_data = candidate_data
        if best_key is not None and best_data is not None:
            reverse_line = _edge_geometry(graph, v, u, best_key, best_data)
            reverse_length_m = float(max(reverse_line.length, 1e-6))
            reverse_pos_m = float(np.clip(reverse_line.project(point), 0.0, reverse_length_m))
            reverse_payload = {
                "reverse_available": True,
                "reverse_u": int(v),
                "reverse_v": int(u),
                "reverse_key": int(best_key),
                "reverse_edge_length_m": reverse_length_m,
                "reverse_pos_m_from_u": reverse_pos_m,
                "reverse_remaining_m_to_v": float(max(reverse_length_m - reverse_pos_m, 0.0)),
            }
    payload.update(reverse_payload)
    return payload


def _normalize_anchor_manifest_types(anchor_manifest: pd.DataFrame) -> pd.DataFrame:
    anchor_manifest = anchor_manifest.copy()
    int_columns = [
        "primary_u",
        "primary_v",
        "primary_key",
        "reverse_u",
        "reverse_v",
        "reverse_key",
    ]
    float_columns = [
        "latitude",
        "longitude",
        "projected_x_m",
        "projected_y_m",
        "primary_edge_length_m",
        "primary_pos_m_from_u",
        "primary_remaining_m_to_v",
        "primary_snap_distance_m",
        "snapped_latitude",
        "snapped_longitude",
        "reverse_edge_length_m",
        "reverse_pos_m_from_u",
        "reverse_remaining_m_to_v",
    ]
    for column in int_columns:
        if column in anchor_manifest:
            anchor_manifest[column] = pd.to_numeric(anchor_manifest[column], errors="coerce").fillna(-1).astype(np.int64)
    for column in float_columns:
        if column in anchor_manifest:
            anchor_manifest[column] = pd.to_numeric(anchor_manifest[column], errors="coerce").astype(np.float64)
    if "reverse_available" in anchor_manifest:
        anchor_manifest["reverse_available"] = anchor_manifest["reverse_available"].map(
            lambda value: str(value).strip().lower() in {"1", "true", "t", "yes"}
        )
    return anchor_manifest


def snap_points_to_drive_graph(
    graph: nx.MultiDiGraph,
    point_frame: pd.DataFrame,
) -> pd.DataFrame:
    require_osmnx()
    point_frame = point_frame.loc[:, ["point_id", "latitude", "longitude"]].copy()
    to_graph, to_geo = _graph_transformers(graph)
    longitudes = point_frame["longitude"].to_numpy(dtype=np.float64)
    latitudes = point_frame["latitude"].to_numpy(dtype=np.float64)
    projected_x, projected_y = to_graph.transform(longitudes, latitudes)
    matched_edges, snap_distances = ox.distance.nearest_edges(
        graph,
        X=projected_x,
        Y=projected_y,
        return_dist=True,
    )
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(point_frame.itertuples(index=False)):
        edge_id = matched_edges[idx]
        records.append(
            _build_match_payload(
                graph,
                point_id=row.point_id,
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                projected_x=float(projected_x[idx]),
                projected_y=float(projected_y[idx]),
                edge_id=(int(edge_id[0]), int(edge_id[1]), int(edge_id[2])),
                snap_distance_m=float(snap_distances[idx]),
                to_geo=to_geo,
            )
        )
    return _normalize_anchor_manifest_types(pd.DataFrame.from_records(records))


def _anchor_hash(anchor_frame: pd.DataFrame) -> str:
    payload = [
        {
            "point_id": str(row.point_id),
            "latitude": round(float(row.latitude), 7),
            "longitude": round(float(row.longitude), 7),
        }
        for row in anchor_frame.itertuples(index=False)
    ]
    return _stable_json_hash({"anchors": payload})


def load_or_build_point_anchor_artifacts(
    graph: nx.MultiDiGraph,
    *,
    point_frame: pd.DataFrame,
    cache_dir: str | Path,
    cache_prefix: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    anchor_payload = point_frame.loc[:, ["point_id", "latitude", "longitude"]].copy()
    anchor_hash = _anchor_hash(anchor_payload)
    manifest_path = cache_dir / f"{cache_prefix}_anchor_manifest.csv"
    route_matrix_path = cache_dir / f"{cache_prefix}_route_matrix_m.npy"
    meta_path = cache_dir / f"{cache_prefix}_anchor_meta.json"
    if manifest_path.exists() and route_matrix_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(meta.get("anchor_hash", "")) == anchor_hash:
            return _normalize_anchor_manifest_types(pd.read_csv(manifest_path)), np.load(route_matrix_path).astype(np.float32)

    manifest = snap_points_to_drive_graph(graph, anchor_payload)
    route_matrix_m = compute_anchor_route_distance_matrix_m(graph, manifest)
    manifest.to_csv(manifest_path, index=False)
    np.save(route_matrix_path, route_matrix_m.astype(np.float32))
    meta_path.write_text(
        json.dumps(
            {
                "cache_prefix": str(cache_prefix),
                "anchor_hash": anchor_hash,
                "point_count": int(anchor_payload.shape[0]),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest, route_matrix_m.astype(np.float32)


def _build_node_distance_matrix(
    graph: nx.MultiDiGraph,
    *,
    endpoint_nodes: Sequence[int],
) -> tuple[list[int], np.ndarray]:
    ordered_nodes = sorted({int(node) for node in endpoint_nodes})
    node_to_idx = {node: idx for idx, node in enumerate(ordered_nodes)}
    matrix = np.full((len(ordered_nodes), len(ordered_nodes)), np.inf, dtype=np.float32)
    np.fill_diagonal(matrix, 0.0)
    for source in ordered_nodes:
        lengths = nx.single_source_dijkstra_path_length(graph, source, weight="length")
        source_idx = node_to_idx[source]
        for target, value in lengths.items():
            target_idx = node_to_idx.get(int(target))
            if target_idx is not None:
                matrix[source_idx, target_idx] = float(value)
    return ordered_nodes, matrix


def compute_anchor_route_distance_matrix_m(
    graph: nx.MultiDiGraph,
    anchor_manifest: pd.DataFrame,
) -> np.ndarray:
    anchor_manifest = _normalize_anchor_manifest_types(anchor_manifest).reset_index(drop=True)
    num_points = int(anchor_manifest.shape[0])
    depart_options: list[list[tuple[int, float]]] = []
    arrival_options: list[list[tuple[int, float]]] = []
    oriented_edges: list[list[tuple[tuple[int, int, int], float]]] = []
    endpoint_nodes: list[int] = []

    for row in anchor_manifest.itertuples(index=False):
        depart = [
            (int(row.primary_v), float(row.primary_remaining_m_to_v)),
        ]
        arrive = [
            (int(row.primary_u), float(row.primary_pos_m_from_u)),
        ]
        edges_for_row = [
            ((int(row.primary_u), int(row.primary_v), int(row.primary_key)), float(row.primary_pos_m_from_u)),
        ]
        endpoint_nodes.extend([int(row.primary_u), int(row.primary_v)])
        if bool(row.reverse_available):
            depart.append((int(row.reverse_v), float(row.reverse_remaining_m_to_v)))
            arrive.append((int(row.reverse_u), float(row.reverse_pos_m_from_u)))
            edges_for_row.append(
                (
                    (int(row.reverse_u), int(row.reverse_v), int(row.reverse_key)),
                    float(row.reverse_pos_m_from_u),
                )
            )
            endpoint_nodes.extend([int(row.reverse_u), int(row.reverse_v)])
        depart_options.append(depart)
        arrival_options.append(arrive)
        oriented_edges.append(edges_for_row)

    ordered_nodes, node_distance_matrix = _build_node_distance_matrix(graph, endpoint_nodes=endpoint_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(ordered_nodes)}

    route_matrix = np.full((num_points, num_points), np.inf, dtype=np.float32)
    np.fill_diagonal(route_matrix, 0.0)
    for src_idx in range(num_points):
        for dst_idx in range(num_points):
            if src_idx == dst_idx:
                continue
            best = float("inf")
            for depart_node, depart_cost in depart_options[src_idx]:
                depart_node_idx = node_to_idx[depart_node]
                for arrive_node, arrive_cost in arrival_options[dst_idx]:
                    arrive_node_idx = node_to_idx[arrive_node]
                    base = float(node_distance_matrix[depart_node_idx, arrive_node_idx])
                    if np.isfinite(base):
                        best = min(best, depart_cost + base + arrive_cost)
            for src_edge, src_pos in oriented_edges[src_idx]:
                for dst_edge, dst_pos in oriented_edges[dst_idx]:
                    if src_edge == dst_edge and src_pos <= dst_pos:
                        best = min(best, float(dst_pos - src_pos))
            route_matrix[src_idx, dst_idx] = float(best)
    return route_matrix.astype(np.float32)


def build_sensor_node_map_distance_graph(
    sensor_frame: pd.DataFrame,
    *,
    topology_adjacency_weights: np.ndarray,
    snapshot_context: dict[str, Any],
    unreachable_policy: str = "raise",
) -> dict[str, Any]:
    graph = snapshot_context["graph"]
    snapshot_dir = Path(snapshot_context["snapshot_dir"])
    sensor_points = sensor_frame.loc[:, ["sensor_id", "latitude", "longitude"]].copy()
    sensor_points = sensor_points.rename(columns={"sensor_id": "point_id"})
    anchor_manifest, route_matrix_m = load_or_build_point_anchor_artifacts(
        graph,
        point_frame=sensor_points,
        cache_dir=snapshot_dir,
        cache_prefix="sensor",
    )
    topology_adjacency_weights = np.asarray(topology_adjacency_weights, dtype=np.float32)
    if topology_adjacency_weights.ndim != 2 or topology_adjacency_weights.shape[0] != topology_adjacency_weights.shape[1]:
        raise ValueError("topology_adjacency_weights must have shape (N, N).")
    if topology_adjacency_weights.shape[0] != route_matrix_m.shape[0]:
        raise ValueError("Topology size must match the number of snapped sensors.")
    adjacency = (topology_adjacency_weights > 0).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)
    adjacency_weights = topology_adjacency_weights.astype(np.float32).copy()
    np.fill_diagonal(adjacency_weights, 0.0)
    distance_matrix_km = (route_matrix_m / 1000.0).astype(np.float32)

    recv, send = np.where(adjacency > 0)
    edge_lengths = distance_matrix_km[recv, send].astype(np.float32)
    unreachable_mask = ~np.isfinite(edge_lengths)
    unreachable_edge_count = int(unreachable_mask.sum())
    fallback_used = 0
    if unreachable_edge_count > 0:
        if str(unreachable_policy).strip().lower() == "haversine_fallback":
            coords = sensor_frame[["latitude", "longitude"]].to_numpy(dtype=np.float64)
            haversine_km = haversine_distance_matrix_km(coords)
            edge_lengths[unreachable_mask] = haversine_km[recv[unreachable_mask], send[unreachable_mask]]
            for edge_idx in np.flatnonzero(unreachable_mask):
                distance_matrix_km[recv[edge_idx], send[edge_idx]] = edge_lengths[edge_idx]
            fallback_used = unreachable_edge_count
        else:
            sample_pairs = [
                (str(sensor_frame.iloc[int(recv[idx])]["sensor_id"]), str(sensor_frame.iloc[int(send[idx])]["sensor_id"]))
                for idx in np.flatnonzero(unreachable_mask)[:5]
            ]
            raise ValueError(
                "Map-routed sensor-node graph contains unreachable retained edges. "
                f"count={unreachable_edge_count} sample_pairs={sample_pairs}"
            )

    return {
        "adjacency": adjacency,
        "adjacency_weights": adjacency_weights,
        "edge_index": np.stack([recv, send], axis=1).astype(np.int64),
        "edge_lengths_km": edge_lengths.astype(np.float32),
        "distance_matrix_km": distance_matrix_km.astype(np.float32),
        "distance_semantics": {
            "edge_lengths_km": "external_map_directed_route_distance_km_between_snapped_sensor_points",
            "distance_matrix_km": "external_map_directed_route_distance_km_between_snapped_sensor_points",
            "adjacency_weights": "topology_source_weighted_adjacency_preserved",
        },
        "sensor_anchor_manifest": anchor_manifest,
        "map_snapshot_manifest": snapshot_context.get("manifest", {}),
        "routing_policy": "directed_shortest_path_from_snapped_sensor_points",
        "unreachable_policy": str(unreachable_policy),
        "unreachable_edge_count": int(unreachable_edge_count),
        "haversine_fallback_edge_count": int(fallback_used),
    }


def _projected_points_to_latlon(graph: nx.MultiDiGraph, xy_points: np.ndarray) -> np.ndarray:
    _, to_geo = _graph_transformers(graph)
    longitudes, latitudes = to_geo.transform(
        xy_points[:, 0].astype(np.float64),
        xy_points[:, 1].astype(np.float64),
    )
    return np.stack([latitudes, longitudes], axis=1).astype(np.float32)


def _build_point_payload_from_edge_position(
    graph: nx.MultiDiGraph,
    *,
    point_id: Any,
    edge_id: tuple[int, int, int],
    pos_m_from_u: float,
) -> dict[str, Any]:
    _, to_geo = _graph_transformers(graph)
    u, v, key = int(edge_id[0]), int(edge_id[1]), int(edge_id[2])
    line = _edge_geometry(graph, u, v, key)
    line_length_m = float(max(line.length, 1e-6))
    pos_m = float(np.clip(float(pos_m_from_u), 0.0, line_length_m))
    point = line.interpolate(pos_m)
    longitude, latitude = to_geo.transform(float(point.x), float(point.y))
    payload = _build_match_payload(
        graph,
        point_id=point_id,
        latitude=float(latitude),
        longitude=float(longitude),
        projected_x=float(point.x),
        projected_y=float(point.y),
        edge_id=(u, v, key),
        snap_distance_m=0.0,
        to_geo=to_geo,
    )
    payload["anchor_kind"] = "carrier_point"
    payload["graph_node_id"] = -1
    payload["carrier_u"] = int(u)
    payload["carrier_v"] = int(v)
    payload["carrier_key"] = int(key)
    payload["carrier_pos_m_from_u"] = float(pos_m)
    return payload


def _build_graph_node_anchor_payload(
    graph: nx.MultiDiGraph,
    *,
    point_id: Any,
    graph_node_id: int,
) -> dict[str, Any]:
    _, to_geo = _graph_transformers(graph)
    node_id = int(graph_node_id)
    projected_x = float(graph.nodes[node_id]["x"])
    projected_y = float(graph.nodes[node_id]["y"])
    longitude, latitude = to_geo.transform(projected_x, projected_y)
    return {
        "point_id": point_id,
        "latitude": float(latitude),
        "longitude": float(longitude),
        "projected_x_m": projected_x,
        "projected_y_m": projected_y,
        "primary_u": node_id,
        "primary_v": node_id,
        "primary_key": -1,
        "primary_edge_length_m": 0.0,
        "primary_pos_m_from_u": 0.0,
        "primary_remaining_m_to_v": 0.0,
        "primary_snap_distance_m": 0.0,
        "snapped_latitude": float(latitude),
        "snapped_longitude": float(longitude),
        "reverse_available": False,
        "reverse_u": -1,
        "reverse_v": -1,
        "reverse_key": -1,
        "reverse_edge_length_m": 0.0,
        "reverse_pos_m_from_u": 0.0,
        "reverse_remaining_m_to_v": 0.0,
        "anchor_kind": "graph_node",
        "graph_node_id": node_id,
        "carrier_u": -1,
        "carrier_v": -1,
        "carrier_key": -1,
        "carrier_pos_m_from_u": 0.0,
    }


def _compute_custom_anchor_route_distance_matrix_m(
    graph: nx.MultiDiGraph,
    anchor_manifest: pd.DataFrame,
) -> np.ndarray:
    anchor_manifest = anchor_manifest.copy().reset_index(drop=True)
    if anchor_manifest.empty:
        return np.zeros((0, 0), dtype=np.float32)

    num_points = int(anchor_manifest.shape[0])
    depart_options: list[list[tuple[int, float]]] = []
    arrival_options: list[list[tuple[int, float]]] = []
    oriented_edges: list[list[tuple[tuple[int, int, int], float]]] = []
    endpoint_nodes: list[int] = []

    for row in anchor_manifest.itertuples(index=False):
        anchor_kind = str(getattr(row, "anchor_kind", "carrier_point")).strip().lower()
        if anchor_kind == "graph_node":
            graph_node_id = int(getattr(row, "graph_node_id"))
            depart = [(graph_node_id, 0.0)]
            arrive = [(graph_node_id, 0.0)]
            edges_for_row: list[tuple[tuple[int, int, int], float]] = []
            endpoint_nodes.append(graph_node_id)
        else:
            depart = [
                (int(row.primary_v), float(row.primary_remaining_m_to_v)),
            ]
            arrive = [
                (int(row.primary_u), float(row.primary_pos_m_from_u)),
            ]
            edges_for_row = [
                ((int(row.primary_u), int(row.primary_v), int(row.primary_key)), float(row.primary_pos_m_from_u)),
            ]
            endpoint_nodes.extend([int(row.primary_u), int(row.primary_v)])
            if bool(row.reverse_available):
                depart.append((int(row.reverse_v), float(row.reverse_remaining_m_to_v)))
                arrive.append((int(row.reverse_u), float(row.reverse_pos_m_from_u)))
                edges_for_row.append(
                    (
                        (int(row.reverse_u), int(row.reverse_v), int(row.reverse_key)),
                        float(row.reverse_pos_m_from_u),
                    )
                )
                endpoint_nodes.extend([int(row.reverse_u), int(row.reverse_v)])
        depart_options.append(depart)
        arrival_options.append(arrive)
        oriented_edges.append(edges_for_row)

    ordered_nodes, node_distance_matrix = _build_node_distance_matrix(graph, endpoint_nodes=endpoint_nodes)
    node_to_idx = {node: idx for idx, node in enumerate(ordered_nodes)}
    route_matrix = np.full((num_points, num_points), np.inf, dtype=np.float32)
    np.fill_diagonal(route_matrix, 0.0)
    for src_idx in range(num_points):
        for dst_idx in range(num_points):
            if src_idx == dst_idx:
                continue
            best = float("inf")
            for depart_node, depart_cost in depart_options[src_idx]:
                depart_node_idx = node_to_idx[depart_node]
                for arrive_node, arrive_cost in arrival_options[dst_idx]:
                    arrive_node_idx = node_to_idx[arrive_node]
                    base = float(node_distance_matrix[depart_node_idx, arrive_node_idx])
                    if np.isfinite(base):
                        best = min(best, depart_cost + base + arrive_cost)
            for src_edge, src_pos in oriented_edges[src_idx]:
                for dst_edge, dst_pos in oriented_edges[dst_idx]:
                    if src_edge == dst_edge and src_pos <= dst_pos:
                        best = min(best, float(dst_pos - src_pos))
            route_matrix[src_idx, dst_idx] = float(best)
    return route_matrix.astype(np.float32)


def _stable_node_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    node_kind = str(record["anchor_kind"]).strip().lower()
    if node_kind == "graph_node":
        return (0, int(record["graph_node_id"]))
    return (
        1,
        int(record["carrier_u"]),
        int(record["carrier_v"]),
        int(record["carrier_key"]),
        round(float(record["carrier_pos_m_from_u"]), 6),
    )


def _edge_highway_rank(highway_value: Any) -> int:
    if isinstance(highway_value, (list, tuple)):
        candidate = str(highway_value[0]).strip().lower() if highway_value else ""
    else:
        candidate = str(highway_value).strip().lower()
    ranking = {
        "motorway": 0,
        "motorway_link": 1,
        "trunk": 2,
        "trunk_link": 3,
        "primary": 4,
        "primary_link": 5,
        "secondary": 6,
        "secondary_link": 7,
        "tertiary": 8,
        "tertiary_link": 9,
        "residential": 10,
        "service": 11,
    }
    return int(ranking.get(candidate, 50))


def _build_drive_edge_candidate_index(graph: nx.MultiDiGraph) -> dict[str, Any]:
    edge_ids: list[tuple[int, int, int]] = []
    edge_geometries: list[LineString] = []
    edge_highway_ranks: list[int] = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        if int(u) == int(v):
            continue
        edge_ids.append((int(u), int(v), int(key)))
        edge_geometries.append(_edge_geometry(graph, int(u), int(v), int(key), data))
        edge_highway_ranks.append(_edge_highway_rank(data.get("highway")))
    return {
        "edge_ids": edge_ids,
        "edge_geometries": edge_geometries,
        "edge_highway_ranks": np.asarray(edge_highway_ranks, dtype=np.int32),
        "tree": STRtree(edge_geometries),
    }


def _build_multi_candidate_anchor_manifest(
    graph: nx.MultiDiGraph,
    point_frame: pd.DataFrame,
    *,
    max_candidates: int,
    max_snap_distance_m: float,
    search_margin_m: float,
) -> pd.DataFrame:
    require_osmnx()
    candidate_index = _build_drive_edge_candidate_index(graph)
    point_frame = point_frame.loc[:, ["sensor_idx", "sensor_id", "latitude", "longitude"]].copy()
    to_graph, to_geo = _graph_transformers(graph)
    longitudes = point_frame["longitude"].to_numpy(dtype=np.float64)
    latitudes = point_frame["latitude"].to_numpy(dtype=np.float64)
    projected_x, projected_y = to_graph.transform(longitudes, latitudes)
    search_radius_m = float(max_snap_distance_m) + float(search_margin_m)
    rows: list[dict[str, Any]] = []

    for row_idx, row in enumerate(point_frame.itertuples(index=False)):
        point = Point(float(projected_x[row_idx]), float(projected_y[row_idx]))
        candidate_edge_ids = candidate_index["tree"].query(point.buffer(search_radius_m))
        candidate_edge_ids = np.asarray(candidate_edge_ids, dtype=np.int64)
        if candidate_edge_ids.size == 0:
            nearest_idx = int(candidate_index["tree"].nearest(point))
            candidate_edge_ids = np.asarray([nearest_idx], dtype=np.int64)
        candidate_edge_ids = np.unique(candidate_edge_ids)

        candidate_records: list[dict[str, Any]] = []
        best_snap_distance_m = float("inf")
        for edge_idx in candidate_edge_ids.tolist():
            geometry = candidate_index["edge_geometries"][int(edge_idx)]
            snap_distance_m = float(geometry.distance(point))
            best_snap_distance_m = min(best_snap_distance_m, snap_distance_m)
            edge_id = candidate_index["edge_ids"][int(edge_idx)]
            payload = _build_match_payload(
                graph,
                point_id=row.sensor_id,
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                projected_x=float(projected_x[row_idx]),
                projected_y=float(projected_y[row_idx]),
                edge_id=edge_id,
                snap_distance_m=snap_distance_m,
                to_geo=to_geo,
            )
            payload.update(
                {
                    "sensor_idx": int(row.sensor_idx),
                    "sensor_id": str(row.sensor_id),
                    "candidate_edge_list_idx": int(edge_idx),
                    "candidate_highway_rank": int(candidate_index["edge_highway_ranks"][int(edge_idx)]),
                }
            )
            candidate_records.append(payload)

        keep_threshold_m = max(float(max_snap_distance_m), float(best_snap_distance_m) + float(search_margin_m))
        kept_candidates = [
            record
            for record in candidate_records
            if float(record["primary_snap_distance_m"]) <= keep_threshold_m + 1e-9
        ]
        if not kept_candidates:
            kept_candidates = sorted(
                candidate_records,
                key=lambda record: (
                    float(record["primary_snap_distance_m"]),
                    int(record["candidate_highway_rank"]),
                    int(record["primary_u"]),
                    int(record["primary_v"]),
                    int(record["primary_key"]),
                ),
            )[:1]

        kept_candidates = sorted(
            kept_candidates,
            key=lambda record: (
                float(record["primary_snap_distance_m"]),
                int(record["candidate_highway_rank"]),
                int(record["primary_u"]),
                int(record["primary_v"]),
                int(record["primary_key"]),
            ),
        )[: max(int(max_candidates), 1)]

        for candidate_rank, candidate_record in enumerate(kept_candidates):
            candidate_record["candidate_rank_generation"] = int(candidate_rank)
            candidate_record["candidate_count_for_sensor"] = int(len(kept_candidates))
            candidate_record["candidate_search_radius_m"] = float(search_radius_m)
            candidate_record["candidate_keep_threshold_m"] = float(keep_threshold_m)
            candidate_record["carrier_edge_id"] = (
                f"{int(candidate_record['primary_u'])}->{int(candidate_record['primary_v'])}"
                f"#{int(candidate_record['primary_key'])}"
            )
            rows.append(candidate_record)

    manifest = _normalize_anchor_manifest_types(pd.DataFrame.from_records(rows))
    if "sensor_idx" in manifest:
        manifest["sensor_idx"] = pd.to_numeric(manifest["sensor_idx"], errors="raise").astype(np.int32)
    if "candidate_edge_list_idx" in manifest:
        manifest["candidate_edge_list_idx"] = pd.to_numeric(
            manifest["candidate_edge_list_idx"],
            errors="raise",
        ).astype(np.int32)
    if "candidate_highway_rank" in manifest:
        manifest["candidate_highway_rank"] = pd.to_numeric(
            manifest["candidate_highway_rank"],
            errors="raise",
        ).astype(np.int32)
    if "candidate_rank_generation" in manifest:
        manifest["candidate_rank_generation"] = pd.to_numeric(
            manifest["candidate_rank_generation"],
            errors="raise",
        ).astype(np.int32)
    return manifest.sort_values(
        by=["sensor_idx", "candidate_rank_generation", "primary_snap_distance_m", "primary_u", "primary_v", "primary_key"],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_sensor_neighbor_lists(
    topology_adjacency_weights: np.ndarray | None,
    *,
    top_k: int,
) -> tuple[dict[int, list[tuple[int, float]]], dict[int, list[tuple[int, float]]]]:
    if topology_adjacency_weights is None:
        return {}, {}
    weights = np.asarray(topology_adjacency_weights, dtype=np.float32)
    if weights.ndim != 2 or weights.shape[0] != weights.shape[1]:
        raise ValueError("topology_adjacency_weights must have shape (N, N) when provided.")
    num_sensors = int(weights.shape[0])
    outgoing: dict[int, list[tuple[int, float]]] = {}
    incoming: dict[int, list[tuple[int, float]]] = {}
    for sensor_idx in range(num_sensors):
        outgoing_candidates = np.flatnonzero(weights[sensor_idx] > 0)
        outgoing_candidates = outgoing_candidates[outgoing_candidates != sensor_idx]
        outgoing_sorted = sorted(
            (
                (int(neighbor_idx), float(weights[sensor_idx, neighbor_idx]))
                for neighbor_idx in outgoing_candidates.tolist()
            ),
            key=lambda item: (-item[1], item[0]),
        )[: max(int(top_k), 0)]
        outgoing[sensor_idx] = outgoing_sorted

        incoming_candidates = np.flatnonzero(weights[:, sensor_idx] > 0)
        incoming_candidates = incoming_candidates[incoming_candidates != sensor_idx]
        incoming_sorted = sorted(
            (
                (int(neighbor_idx), float(weights[neighbor_idx, sensor_idx]))
                for neighbor_idx in incoming_candidates.tolist()
            ),
            key=lambda item: (-item[1], item[0]),
        )[: max(int(top_k), 0)]
        incoming[sensor_idx] = incoming_sorted
    return outgoing, incoming


def _route_distance_between_candidate_rows_m(
    graph: nx.MultiDiGraph,
    src_row: pd.Series | dict[str, Any],
    dst_row: pd.Series | dict[str, Any],
    *,
    node_pair_cache: dict[tuple[int, int], float],
) -> float:
    src_u = int(src_row["primary_u"])
    src_v = int(src_row["primary_v"])
    src_key = int(src_row["primary_key"])
    src_pos = float(src_row["primary_pos_m_from_u"])
    src_remaining = float(src_row["primary_remaining_m_to_v"])
    dst_u = int(dst_row["primary_u"])
    dst_v = int(dst_row["primary_v"])
    dst_key = int(dst_row["primary_key"])
    dst_pos = float(dst_row["primary_pos_m_from_u"])
    best = float("inf")
    if src_u == dst_u and src_v == dst_v and src_key == dst_key and src_pos <= dst_pos:
        best = min(best, float(dst_pos - src_pos))
    node_pair = (src_v, dst_u)
    if node_pair not in node_pair_cache:
        try:
            node_pair_cache[node_pair] = float(
                nx.shortest_path_length(graph, source=src_v, target=dst_u, weight="length")
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            node_pair_cache[node_pair] = float("inf")
    base_path = float(node_pair_cache[node_pair])
    if np.isfinite(base_path):
        best = min(best, float(src_remaining + base_path + dst_pos))
    return float(best)


def _score_candidate_assignment(
    sensor_idx: int,
    candidate_row: pd.Series,
    *,
    current_assignments: dict[int, pd.Series],
    outgoing_neighbors: dict[int, list[tuple[int, float]]],
    incoming_neighbors: dict[int, list[tuple[int, float]]],
    graph: nx.MultiDiGraph,
    node_pair_cache: dict[tuple[int, int], float],
    route_cap_m: float,
    dedupe_position_eps_m: float,
) -> dict[str, Any]:
    score_outgoing = 0.0
    score_incoming = 0.0
    outgoing_weight_sum = 0.0
    incoming_weight_sum = 0.0
    unreachable_neighbors = 0
    used_neighbors = 0
    collision_count = 0

    for other_sensor_idx, other_assignment in current_assignments.items():
        if int(other_sensor_idx) == int(sensor_idx):
            continue
        if (
            int(other_assignment["primary_u"]) == int(candidate_row["primary_u"])
            and int(other_assignment["primary_v"]) == int(candidate_row["primary_v"])
            and int(other_assignment["primary_key"]) == int(candidate_row["primary_key"])
            and abs(
                float(other_assignment["primary_pos_m_from_u"]) - float(candidate_row["primary_pos_m_from_u"])
            )
            <= float(dedupe_position_eps_m)
        ):
            collision_count += 1

    for neighbor_idx, weight in outgoing_neighbors.get(sensor_idx, []):
        assigned_neighbor = current_assignments.get(int(neighbor_idx))
        if assigned_neighbor is None:
            continue
        route_distance_m = _route_distance_between_candidate_rows_m(
            graph,
            candidate_row,
            assigned_neighbor,
            node_pair_cache=node_pair_cache,
        )
        used_neighbors += 1
        if not np.isfinite(route_distance_m):
            unreachable_neighbors += 1
            route_distance_m = float(route_cap_m)
        score_outgoing += float(weight) * float(min(route_distance_m, route_cap_m))
        outgoing_weight_sum += float(weight)

    for neighbor_idx, weight in incoming_neighbors.get(sensor_idx, []):
        assigned_neighbor = current_assignments.get(int(neighbor_idx))
        if assigned_neighbor is None:
            continue
        route_distance_m = _route_distance_between_candidate_rows_m(
            graph,
            assigned_neighbor,
            candidate_row,
            node_pair_cache=node_pair_cache,
        )
        used_neighbors += 1
        if not np.isfinite(route_distance_m):
            unreachable_neighbors += 1
            route_distance_m = float(route_cap_m)
        score_incoming += float(weight) * float(min(route_distance_m, route_cap_m))
        incoming_weight_sum += float(weight)

    weighted_route_score_m = 0.0
    total_route_weight = outgoing_weight_sum + incoming_weight_sum
    if total_route_weight > 0:
        weighted_route_score_m = float((score_outgoing + score_incoming) / total_route_weight)
    snap_score_m = float(candidate_row["primary_snap_distance_m"])
    midpoint_penalty_m = float(
        abs(float(candidate_row["primary_pos_m_from_u"]) - (float(candidate_row["primary_edge_length_m"]) / 2.0))
    )
    score_total = float(weighted_route_score_m + 0.50 * snap_score_m + 0.10 * midpoint_penalty_m)
    return {
        "score_total": score_total,
        "score_route_m": float(weighted_route_score_m),
        "score_snap_m": snap_score_m,
        "score_midpoint_penalty_m": midpoint_penalty_m,
        "score_collision_count": int(collision_count),
        "score_unreachable_neighbor_count": int(unreachable_neighbors),
        "score_used_neighbor_count": int(used_neighbors),
        "score_tie_tuple": (
            int(collision_count),
            int(unreachable_neighbors),
            round(float(score_total), 6),
            round(float(snap_score_m), 6),
            round(float(midpoint_penalty_m), 6),
            int(candidate_row["candidate_highway_rank"]),
            str(candidate_row["carrier_edge_id"]),
            int(candidate_row["candidate_rank_generation"]),
        ),
    }


def _resolve_multi_candidate_carrier_assignments(
    candidate_manifest: pd.DataFrame,
    *,
    graph: nx.MultiDiGraph,
    topology_adjacency_weights: np.ndarray | None,
    neighbor_top_k: int,
    assignment_passes: int,
    route_cap_m: float,
    dedupe_position_eps_m: float,
) -> dict[str, Any]:
    candidate_manifest = candidate_manifest.copy().reset_index(drop=True)
    if candidate_manifest.empty:
        raise ValueError("candidate_manifest must contain at least one candidate per sensor.")
    if "sensor_idx" not in candidate_manifest:
        raise ValueError("candidate_manifest must contain sensor_idx.")

    grouped_candidates = {
        int(sensor_idx): group.sort_values(
            by=["candidate_rank_generation", "primary_snap_distance_m", "primary_u", "primary_v", "primary_key"],
            kind="mergesort",
        ).reset_index(drop=True)
        for sensor_idx, group in candidate_manifest.groupby("sensor_idx", sort=True)
    }
    sensor_order = sorted(grouped_candidates)
    outgoing_neighbors, incoming_neighbors = _build_sensor_neighbor_lists(
        topology_adjacency_weights,
        top_k=int(neighbor_top_k),
    )

    current_candidate_index = {sensor_idx: 0 for sensor_idx in sensor_order}
    current_assignments = {
        sensor_idx: grouped_candidates[sensor_idx].iloc[0].copy()
        for sensor_idx in sensor_order
    }
    final_scored_by_sensor: dict[int, list[dict[str, Any]]] = {}
    node_pair_cache: dict[tuple[int, int], float] = {}
    converged_pass = int(max(int(assignment_passes), 1))

    for pass_idx in range(max(int(assignment_passes), 1)):
        changed = 0
        scored_by_sensor_pass: dict[int, list[dict[str, Any]]] = {}
        for sensor_idx in sensor_order:
            candidate_group = grouped_candidates[sensor_idx]
            scored_candidates: list[dict[str, Any]] = []
            for candidate_local_idx in range(candidate_group.shape[0]):
                candidate_row = candidate_group.iloc[candidate_local_idx]
                score_payload = _score_candidate_assignment(
                    sensor_idx,
                    candidate_row,
                    current_assignments=current_assignments,
                    outgoing_neighbors=outgoing_neighbors,
                    incoming_neighbors=incoming_neighbors,
                    graph=graph,
                    node_pair_cache=node_pair_cache,
                    route_cap_m=float(route_cap_m),
                    dedupe_position_eps_m=float(dedupe_position_eps_m),
                )
                candidate_record = candidate_row.to_dict()
                candidate_record.update(
                    {
                        "candidate_local_idx": int(candidate_local_idx),
                        "assignment_pass": int(pass_idx + 1),
                        **score_payload,
                    }
                )
                scored_candidates.append(candidate_record)
            scored_candidates.sort(key=lambda record: record["score_tie_tuple"])
            best_record = scored_candidates[0]
            best_idx = int(best_record["candidate_local_idx"])
            if best_idx != current_candidate_index[sensor_idx]:
                changed += 1
            current_candidate_index[sensor_idx] = best_idx
            current_assignments[sensor_idx] = candidate_group.iloc[best_idx].copy()
            scored_by_sensor_pass[sensor_idx] = scored_candidates
        final_scored_by_sensor = scored_by_sensor_pass
        converged_pass = int(pass_idx + 1)
        if changed == 0:
            break

    scored_rows = [
        record
        for sensor_idx in sensor_order
        for record in final_scored_by_sensor.get(sensor_idx, [])
    ]
    scored_manifest = pd.DataFrame.from_records(scored_rows)
    if scored_manifest.empty:
        raise ValueError("Failed to score candidate assignments.")

    assignment_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for sensor_idx in sensor_order:
        scored_candidates = sorted(
            final_scored_by_sensor[sensor_idx],
            key=lambda record: record["score_tie_tuple"],
        )
        winner = scored_candidates[0]
        runner_up = scored_candidates[1] if len(scored_candidates) > 1 else None
        assignment_row = {
            key: value
            for key, value in winner.items()
            if key not in {"score_tie_tuple", "candidate_local_idx"}
        }
        assignment_row["selected_candidate_rank"] = int(winner["candidate_rank_generation"])
        assignment_rows.append(assignment_row)
        decision_rows.append(
            {
                "sensor_idx": int(sensor_idx),
                "sensor_id": str(winner["sensor_id"]),
                "winner_carrier_edge_id": str(winner["carrier_edge_id"]),
                "winner_candidate_rank": int(winner["candidate_rank_generation"]),
                "winner_score_total": float(winner["score_total"]),
                "winner_collision_count": int(winner["score_collision_count"]),
                "winner_unreachable_neighbor_count": int(winner["score_unreachable_neighbor_count"]),
                "winner_snap_distance_m": float(winner["score_snap_m"]),
                "winner_midpoint_penalty_m": float(winner["score_midpoint_penalty_m"]),
                "winner_route_score_m": float(winner["score_route_m"]),
                "runnerup_carrier_edge_id": (str(runner_up["carrier_edge_id"]) if runner_up is not None else ""),
                "runnerup_candidate_rank": (int(runner_up["candidate_rank_generation"]) if runner_up is not None else -1),
                "runnerup_score_total": (float(runner_up["score_total"]) if runner_up is not None else np.nan),
                "winner_margin": (
                    float(runner_up["score_total"] - winner["score_total"])
                    if runner_up is not None
                    else np.nan
                ),
                "decision_pass": int(converged_pass),
            }
        )

    scored_manifest["selected_candidate_rank"] = scored_manifest["sensor_idx"].map(
        lambda sensor_idx: int(current_candidate_index[int(sensor_idx)])
    ).astype(np.int32)
    scored_manifest["is_selected"] = (
        scored_manifest["candidate_local_idx"].astype(np.int32)
        == scored_manifest["selected_candidate_rank"].astype(np.int32)
    )
    assignment_manifest = _normalize_anchor_manifest_types(pd.DataFrame.from_records(assignment_rows))
    decision_manifest = pd.DataFrame.from_records(decision_rows)
    return {
        "assignment_manifest": assignment_manifest.sort_values(by=["sensor_idx"], kind="mergesort").reset_index(drop=True),
        "candidate_raw_manifest": candidate_manifest.sort_values(
            by=["sensor_idx", "candidate_rank_generation"],
            kind="mergesort",
        ).reset_index(drop=True),
        "candidate_scored_manifest": scored_manifest.sort_values(
            by=["sensor_idx", "candidate_rank_generation"],
            kind="mergesort",
        ).reset_index(drop=True),
        "assignment_decision_manifest": decision_manifest.sort_values(
            by=["sensor_idx"],
            kind="mergesort",
        ).reset_index(drop=True),
        "assignment_passes_used": int(converged_pass),
    }


def build_map_carrier_split_pseudo_edge_graph(
    sensor_frame: pd.DataFrame,
    *,
    sensor_ids: Sequence[object],
    snapshot_context: dict[str, Any],
    topology_adjacency_weights: np.ndarray | None = None,
    min_segment_length_m: float = 5.0,
    dedupe_position_eps_m: float = 1.0,
    unreachable_policy: str = "raise",
    max_candidate_edges: int = 6,
    max_snap_distance_m: float = 80.0,
    candidate_search_margin_m: float = 15.0,
    topology_neighbor_top_k: int = 4,
    assignment_passes: int = 4,
    assignment_route_cap_m: float = 5000.0,
) -> dict[str, Any]:
    graph = snapshot_context["graph"]
    sensor_frame = sensor_frame.copy().reset_index(drop=True)
    sensor_frame["sensor_id"] = normalize_sensor_ids(sensor_ids)
    sensor_points = sensor_frame.loc[:, ["sensor_id", "latitude", "longitude"]].copy()
    sensor_points.insert(0, "sensor_idx", np.arange(sensor_frame.shape[0], dtype=np.int32))
    candidate_manifest = _build_multi_candidate_anchor_manifest(
        graph,
        sensor_points,
        max_candidates=int(max_candidate_edges),
        max_snap_distance_m=float(max_snap_distance_m),
        search_margin_m=float(candidate_search_margin_m),
    )
    assignment_payload = _resolve_multi_candidate_carrier_assignments(
        candidate_manifest,
        graph=graph,
        topology_adjacency_weights=topology_adjacency_weights,
        neighbor_top_k=int(topology_neighbor_top_k),
        assignment_passes=int(assignment_passes),
        route_cap_m=float(assignment_route_cap_m),
        dedupe_position_eps_m=float(dedupe_position_eps_m),
    )
    carrier_assignment_manifest = assignment_payload["assignment_manifest"].copy().reset_index(drop=True)
    sensor_anchor_manifest = carrier_assignment_manifest.copy().reset_index(drop=True)

    node_records_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    pseudo_edge_rows: list[dict[str, Any]] = []
    carrier_group_rows: list[dict[str, Any]] = []

    grouped = carrier_assignment_manifest.groupby(["primary_u", "primary_v", "primary_key"], sort=True)
    for (carrier_u, carrier_v, carrier_key), carrier_rows in grouped:
        group_frame = carrier_rows.sort_values(
            by=["primary_pos_m_from_u", "sensor_id", "sensor_idx"],
            kind="mergesort",
        ).reset_index(drop=True)
        carrier_length_m = float(group_frame.loc[0, "primary_edge_length_m"])
        sensor_positions = group_frame["primary_pos_m_from_u"].to_numpy(dtype=np.float64)
        if sensor_positions.size > 1:
            position_gaps = np.diff(sensor_positions)
            if np.any(position_gaps <= float(dedupe_position_eps_m)):
                bad_idx = int(np.flatnonzero(position_gaps <= float(dedupe_position_eps_m))[0])
                sensor_pair = (
                    str(group_frame.loc[bad_idx, "sensor_id"]),
                    str(group_frame.loc[bad_idx + 1, "sensor_id"]),
                )
                raise ValueError(
                    "Carrier/split-edge build found two sensors too close on the same directed carrier edge: "
                    f"carrier={int(carrier_u)}->{int(carrier_v)}#{int(carrier_key)} sensors={sensor_pair} "
                    f"gap_m={float(position_gaps[bad_idx]):.3f}"
                )

        boundaries = np.zeros(sensor_positions.size + 1, dtype=np.float64)
        boundaries[0] = 0.0
        boundaries[-1] = carrier_length_m
        if sensor_positions.size > 1:
            boundaries[1:-1] = (sensor_positions[:-1] + sensor_positions[1:]) / 2.0
        segment_lengths_m = np.diff(boundaries)
        if np.any(segment_lengths_m <= float(min_segment_length_m)):
            bad_idx = int(np.flatnonzero(segment_lengths_m <= float(min_segment_length_m))[0])
            raise ValueError(
                "Carrier/split-edge build produced a degenerate subsegment: "
                f"carrier={int(carrier_u)}->{int(carrier_v)}#{int(carrier_key)} "
                f"segment_ordinal={bad_idx} length_m={float(segment_lengths_m[bad_idx]):.3f}"
            )

        boundary_keys: list[tuple[Any, ...]] = []
        for boundary_idx, boundary_pos_m in enumerate(boundaries):
            near_u = float(boundary_pos_m) <= float(dedupe_position_eps_m)
            near_v = float(carrier_length_m - boundary_pos_m) <= float(dedupe_position_eps_m)
            if near_u:
                node_key = ("graph_node", int(carrier_u))
                if node_key not in node_records_by_key:
                    node_records_by_key[node_key] = _build_graph_node_anchor_payload(
                        graph,
                        point_id=f"graph_node::{int(carrier_u)}",
                        graph_node_id=int(carrier_u),
                    )
            elif near_v:
                node_key = ("graph_node", int(carrier_v))
                if node_key not in node_records_by_key:
                    node_records_by_key[node_key] = _build_graph_node_anchor_payload(
                        graph,
                        point_id=f"graph_node::{int(carrier_v)}",
                        graph_node_id=int(carrier_v),
                    )
            else:
                rounded_pos_m = round(float(boundary_pos_m), 6)
                node_key = ("carrier_split", int(carrier_u), int(carrier_v), int(carrier_key), rounded_pos_m)
                if node_key not in node_records_by_key:
                    node_records_by_key[node_key] = _build_point_payload_from_edge_position(
                        graph,
                        point_id=(
                            f"carrier_split::{int(carrier_u)}::{int(carrier_v)}::{int(carrier_key)}"
                            f"::{rounded_pos_m:.6f}"
                        ),
                        edge_id=(int(carrier_u), int(carrier_v), int(carrier_key)),
                        pos_m_from_u=rounded_pos_m,
                    )
            boundary_keys.append(node_key)
            carrier_group_rows.append(
                {
                    "carrier_u": int(carrier_u),
                    "carrier_v": int(carrier_v),
                    "carrier_key": int(carrier_key),
                    "boundary_ordinal": int(boundary_idx),
                    "boundary_pos_m_from_u": float(boundary_pos_m),
                    "boundary_node_key": str(node_key),
                }
            )

        for local_idx, row in enumerate(group_frame.itertuples(index=False)):
            tail_key = boundary_keys[local_idx]
            head_key = boundary_keys[local_idx + 1]
            if tail_key == head_key:
                raise ValueError(
                    "Carrier/split-edge build produced a self-loop segment, which should be impossible "
                    f"after split validation: carrier={int(carrier_u)}->{int(carrier_v)}#{int(carrier_key)} "
                    f"sensor_id={row.sensor_id}"
                )
            segment_start_m = float(boundaries[local_idx])
            segment_end_m = float(boundaries[local_idx + 1])
            segment_mid_m = float((segment_start_m + segment_end_m) / 2.0)
            pseudo_edge_rows.append(
                {
                    "pseudo_edge_id": int(len(pseudo_edge_rows)),
                    "sensor_idx": int(row.sensor_idx),
                    "sensor_id": str(row.sensor_id),
                    "carrier_u": int(carrier_u),
                    "carrier_v": int(carrier_v),
                    "carrier_key": int(carrier_key),
                    "carrier_edge_id": f"{int(carrier_u)}->{int(carrier_v)}#{int(carrier_key)}",
                    "carrier_edge_length_m": float(carrier_length_m),
                    "carrier_group_size": int(group_frame.shape[0]),
                    "carrier_segment_ordinal": int(local_idx),
                    "carrier_sensor_pos_m_from_u": float(row.primary_pos_m_from_u),
                    "segment_start_m_from_u": segment_start_m,
                    "segment_end_m_from_u": segment_end_m,
                    "segment_mid_m_from_u": segment_mid_m,
                    "segment_length_m": float(segment_lengths_m[local_idx]),
                    "sensor_offset_from_segment_mid_m": float(row.primary_pos_m_from_u) - segment_mid_m,
                    "_tail_node_key": tail_key,
                    "_head_node_key": head_key,
                    "tail_node_key": str(tail_key),
                    "head_node_key": str(head_key),
                    "sensor_latitude": float(row.latitude),
                    "sensor_longitude": float(row.longitude),
                    "snapped_latitude": float(row.snapped_latitude),
                    "snapped_longitude": float(row.snapped_longitude),
                    "snap_distance_m": float(row.primary_snap_distance_m),
                }
            )

    ordered_node_items = sorted(node_records_by_key.items(), key=lambda item: _stable_node_sort_key(item[1]))
    node_key_to_idx = {node_key: idx for idx, (node_key, _) in enumerate(ordered_node_items)}
    pseudo_node_anchor_rows: list[dict[str, Any]] = []
    for node_idx, (node_key, node_record) in enumerate(ordered_node_items):
        record = dict(node_record)
        record["pseudo_node_id"] = int(node_idx)
        record["stable_node_key"] = str(node_key)
        pseudo_node_anchor_rows.append(record)
    pseudo_node_anchor_manifest = _normalize_anchor_manifest_types(pd.DataFrame.from_records(pseudo_node_anchor_rows))
    if "anchor_kind" in pseudo_node_anchor_manifest:
        pseudo_node_anchor_manifest["anchor_kind"] = pseudo_node_anchor_manifest["anchor_kind"].astype(str)
    if "graph_node_id" in pseudo_node_anchor_manifest:
        pseudo_node_anchor_manifest["graph_node_id"] = pd.to_numeric(
            pseudo_node_anchor_manifest["graph_node_id"],
            errors="coerce",
        ).fillna(-1).astype(np.int64)
    if "carrier_pos_m_from_u" in pseudo_node_anchor_manifest:
        pseudo_node_anchor_manifest["carrier_pos_m_from_u"] = pd.to_numeric(
            pseudo_node_anchor_manifest["carrier_pos_m_from_u"],
            errors="coerce",
        ).astype(np.float64)

    tail_counts = np.zeros(len(ordered_node_items), dtype=np.int32)
    head_counts = np.zeros(len(ordered_node_items), dtype=np.int32)
    for row in pseudo_edge_rows:
        row["tail_node"] = int(node_key_to_idx[row["_tail_node_key"]])
        row["head_node"] = int(node_key_to_idx[row["_head_node_key"]])
        tail_counts[row["tail_node"]] += 1
        head_counts[row["head_node"]] += 1
        del row["_tail_node_key"]
        del row["_head_node_key"]

    pseudo_edge_manifest = pd.DataFrame.from_records(pseudo_edge_rows)
    edge_index = pseudo_edge_manifest.loc[:, ["tail_node", "head_node"]].to_numpy(dtype=np.int64)
    edge_lengths_km = (pseudo_edge_manifest["segment_length_m"].to_numpy(dtype=np.float32) / 1000.0).astype(np.float32)

    adjacency = np.zeros((len(ordered_node_items), len(ordered_node_items)), dtype=np.float32)
    adjacency[edge_index[:, 0], edge_index[:, 1]] = 1.0
    adjacency_weights = adjacency.astype(np.float32).copy()

    route_matrix_m = _compute_custom_anchor_route_distance_matrix_m(graph, pseudo_node_anchor_manifest)
    route_matrix_km = (route_matrix_m / 1000.0).astype(np.float32)
    unreachable_pair_mask = ~np.isfinite(route_matrix_km)
    np.fill_diagonal(unreachable_pair_mask, False)
    unreachable_pair_count = int(unreachable_pair_mask.sum())
    fallback_used = 0
    if unreachable_pair_count > 0 and str(unreachable_policy).strip().lower() == "haversine_fallback":
        fallback_distance_km = haversine_distance_matrix_km(
            pseudo_node_anchor_manifest.loc[:, ["latitude", "longitude"]].to_numpy(dtype=np.float64)
        )
        route_matrix_km[unreachable_pair_mask] = fallback_distance_km[unreachable_pair_mask]
        fallback_used = unreachable_pair_count

    pseudo_node_metadata = pd.DataFrame(
        {
            "pseudo_node_id": np.arange(len(ordered_node_items), dtype=np.int32),
            "stable_node_key": pseudo_node_anchor_manifest["stable_node_key"].astype(str),
            "anchor_kind": pseudo_node_anchor_manifest["anchor_kind"].astype(str),
            "graph_node_id": pseudo_node_anchor_manifest["graph_node_id"].to_numpy(dtype=np.int64),
            "latitude": pseudo_node_anchor_manifest["latitude"].to_numpy(dtype=np.float32),
            "longitude": pseudo_node_anchor_manifest["longitude"].to_numpy(dtype=np.float32),
            "projected_x_m": pseudo_node_anchor_manifest["projected_x_m"].to_numpy(dtype=np.float32),
            "projected_y_m": pseudo_node_anchor_manifest["projected_y_m"].to_numpy(dtype=np.float32),
            "carrier_u": pseudo_node_anchor_manifest["carrier_u"].to_numpy(dtype=np.int64),
            "carrier_v": pseudo_node_anchor_manifest["carrier_v"].to_numpy(dtype=np.int64),
            "carrier_key": pseudo_node_anchor_manifest["carrier_key"].to_numpy(dtype=np.int64),
            "carrier_pos_m_from_u": pseudo_node_anchor_manifest["carrier_pos_m_from_u"].to_numpy(dtype=np.float32),
            "endpoint_count": (tail_counts + head_counts).astype(np.int32),
            "tail_endpoint_count": tail_counts.astype(np.int32),
            "head_endpoint_count": head_counts.astype(np.int32),
            "snap_distance_m": pseudo_node_anchor_manifest["primary_snap_distance_m"].to_numpy(dtype=np.float32),
        }
    )

    return {
        "adjacency": adjacency.astype(np.float32),
        "adjacency_weights": adjacency_weights.astype(np.float32),
        "edge_index": edge_index.astype(np.int64),
        "edge_lengths_km": edge_lengths_km.astype(np.float32),
        "distance_matrix_km": route_matrix_km.astype(np.float32),
        "distance_semantics": {
            "edge_lengths_km": "road_split_subsegment_length_km",
            "distance_matrix_km": "pseudo_node_road_shortest_path_km",
            "speed_values": "one_sensor_per_directed_carrier_split_subsegment",
            "adjacency_weights": "binary_split_graph_connectivity",
        },
        "pseudo_node_metadata": pseudo_node_metadata,
        "pseudo_edge_manifest": pseudo_edge_manifest,
        "pseudo_node_anchor_manifest": pseudo_node_anchor_manifest,
        "sensor_anchor_manifest": sensor_anchor_manifest,
        "carrier_assignment_manifest": carrier_assignment_manifest,
        "carrier_candidate_raw_manifest": assignment_payload["candidate_raw_manifest"],
        "carrier_candidate_scored_manifest": assignment_payload["candidate_scored_manifest"],
        "carrier_assignment_decision_manifest": assignment_payload["assignment_decision_manifest"],
        "split_node_manifest": pseudo_node_metadata.copy(),
        "carrier_boundary_manifest": pd.DataFrame.from_records(carrier_group_rows),
        "self_loop_fixes": np.int32(0),
        "topology_summary": _build_pseudo_edge_topology_summary(
            edge_index,
            num_nodes=len(ordered_node_items),
        ),
        "map_snapshot_manifest": snapshot_context.get("manifest", {}),
        "routing_policy": "directed_shortest_path_between_carrier_split_nodes",
        "unreachable_policy": str(unreachable_policy),
        "unreachable_edge_count": int(0),
        "unreachable_pair_count": int(unreachable_pair_count),
        "haversine_fallback_edge_count": int(fallback_used),
        "builder_algorithm": "carrier_split_v2_multicandidate",
        "min_segment_length_m": float(min_segment_length_m),
        "dedupe_position_eps_m": float(dedupe_position_eps_m),
        "max_candidate_edges": int(max_candidate_edges),
        "max_snap_distance_m": float(max_snap_distance_m),
        "candidate_search_margin_m": float(candidate_search_margin_m),
        "topology_neighbor_top_k": int(topology_neighbor_top_k),
        "assignment_passes_used": int(assignment_payload["assignment_passes_used"]),
        "assignment_route_cap_m": float(assignment_route_cap_m),
    }


def build_map_centered_pseudo_edge_graph(
    sensor_frame: pd.DataFrame,
    *,
    sensor_ids: Sequence[object],
    snapshot_context: dict[str, Any],
    cluster_radius_m: float | None = None,
    cluster_radius_scale: float = 1.25,
    half_length_scale: float = 0.45,
    min_half_length_m: float = 50.0,
    max_half_length_m: float = 500.0,
    unreachable_policy: str = "raise",
) -> dict[str, Any]:
    graph = snapshot_context["graph"]
    snapshot_dir = Path(snapshot_context["snapshot_dir"])
    sensor_frame = sensor_frame.copy().reset_index(drop=True)
    sensor_frame["sensor_id"] = normalize_sensor_ids(sensor_ids)
    sensor_points = sensor_frame.loc[:, ["sensor_id", "latitude", "longitude"]].rename(columns={"sensor_id": "point_id"})
    sensor_anchor_manifest, sensor_route_matrix_m = load_or_build_point_anchor_artifacts(
        graph,
        point_frame=sensor_points,
        cache_dir=snapshot_dir,
        cache_prefix="sensor",
    )
    sensor_route_matrix_m = sensor_route_matrix_m.astype(np.float32)
    route_positive = sensor_route_matrix_m.copy()
    route_positive[~np.isfinite(route_positive)] = np.inf
    np.fill_diagonal(route_positive, np.inf)
    nearest_route_m = route_positive.min(axis=1)
    finite_nearest = nearest_route_m[np.isfinite(nearest_route_m) & (nearest_route_m > 0)]
    fallback_nearest_m = (
        float(np.median(finite_nearest))
        if finite_nearest.size
        else float(max(float(min_half_length_m) * 2.0, 100.0))
    )
    nearest_route_m = np.where(np.isfinite(nearest_route_m), nearest_route_m, fallback_nearest_m)

    tail_points_xy = np.zeros((sensor_anchor_manifest.shape[0], 2), dtype=np.float32)
    head_points_xy = np.zeros((sensor_anchor_manifest.shape[0], 2), dtype=np.float32)
    actual_half_lengths_m = np.zeros(sensor_anchor_manifest.shape[0], dtype=np.float32)
    target_half_lengths_m = np.zeros(sensor_anchor_manifest.shape[0], dtype=np.float32)
    clipped_half_length_count = 0

    for idx, row in enumerate(sensor_anchor_manifest.itertuples(index=False)):
        line = _edge_geometry(graph, int(row.primary_u), int(row.primary_v), int(row.primary_key))
        target_half = float(np.clip(float(nearest_route_m[idx]) * float(half_length_scale), float(min_half_length_m), float(max_half_length_m)))
        available_half = float(max(min(float(row.primary_pos_m_from_u), float(row.primary_remaining_m_to_v)), 1.0))
        actual_half = float(min(target_half, available_half))
        if actual_half < target_half:
            clipped_half_length_count += 1
        target_half_lengths_m[idx] = target_half
        actual_half_lengths_m[idx] = actual_half
        tail_point = line.interpolate(float(row.primary_pos_m_from_u) - actual_half)
        head_point = line.interpolate(float(row.primary_pos_m_from_u) + actual_half)
        tail_points_xy[idx] = [float(tail_point.x), float(tail_point.y)]
        head_points_xy[idx] = [float(head_point.x), float(head_point.y)]

    if cluster_radius_m is None:
        cluster_radius_value = float(np.median(actual_half_lengths_m) * float(cluster_radius_scale))
    else:
        cluster_radius_value = float(cluster_radius_m)
    cluster_radius_value = float(np.clip(cluster_radius_value, 30.0, 250.0))

    endpoint_xy = np.concatenate([tail_points_xy, head_points_xy], axis=0)
    endpoint_labels = _cluster_points_by_radius(endpoint_xy / 1000.0, radius_km=cluster_radius_value / 1000.0)
    num_sensors = int(sensor_anchor_manifest.shape[0])
    tail_labels = endpoint_labels[:num_sensors].astype(np.int32).copy()
    head_labels = endpoint_labels[num_sensors:].astype(np.int32).copy()
    next_label = int(endpoint_labels.max()) + 1 if endpoint_labels.size else 0
    self_loop_fixes = 0
    for sensor_idx in range(num_sensors):
        if int(tail_labels[sensor_idx]) == int(head_labels[sensor_idx]):
            head_labels[sensor_idx] = next_label
            next_label += 1
            self_loop_fixes += 1

    raw_node_points: dict[int, list[np.ndarray]] = {}
    endpoint_roles: dict[int, dict[str, int]] = {}
    for sensor_idx in range(num_sensors):
        tail_label = int(tail_labels[sensor_idx])
        head_label = int(head_labels[sensor_idx])
        raw_node_points.setdefault(tail_label, []).append(tail_points_xy[sensor_idx])
        raw_node_points.setdefault(head_label, []).append(head_points_xy[sensor_idx])
        endpoint_roles.setdefault(tail_label, {"tail_count": 0, "head_count": 0})
        endpoint_roles.setdefault(head_label, {"tail_count": 0, "head_count": 0})
        endpoint_roles[tail_label]["tail_count"] += 1
        endpoint_roles[head_label]["head_count"] += 1

    ordered_labels = sorted(raw_node_points)
    label_remap = {raw_label: idx for idx, raw_label in enumerate(ordered_labels)}
    pseudo_node_xy = np.zeros((len(ordered_labels), 2), dtype=np.float32)
    pseudo_node_counts = np.zeros(len(ordered_labels), dtype=np.int32)
    pseudo_node_tail_counts = np.zeros(len(ordered_labels), dtype=np.int32)
    pseudo_node_head_counts = np.zeros(len(ordered_labels), dtype=np.int32)
    for raw_label, node_idx in label_remap.items():
        points = np.stack(raw_node_points[raw_label], axis=0).astype(np.float32)
        pseudo_node_xy[node_idx] = points.mean(axis=0)
        pseudo_node_counts[node_idx] = int(points.shape[0])
        pseudo_node_tail_counts[node_idx] = int(endpoint_roles[raw_label]["tail_count"])
        pseudo_node_head_counts[node_idx] = int(endpoint_roles[raw_label]["head_count"])

    edge_index = np.stack(
        [
            np.array([label_remap[int(label)] for label in tail_labels], dtype=np.int64),
            np.array([label_remap[int(label)] for label in head_labels], dtype=np.int64),
        ],
        axis=1,
    )
    pseudo_node_coords = _projected_points_to_latlon(graph, pseudo_node_xy)
    pseudo_node_points = pd.DataFrame(
        {
            "point_id": np.arange(pseudo_node_xy.shape[0], dtype=np.int32),
            "latitude": pseudo_node_coords[:, 0].astype(np.float32),
            "longitude": pseudo_node_coords[:, 1].astype(np.float32),
        }
    )
    pseudo_node_anchor_manifest = snap_points_to_drive_graph(graph, pseudo_node_points)
    pseudo_node_route_matrix_m = compute_anchor_route_distance_matrix_m(graph, pseudo_node_anchor_manifest)
    pseudo_node_route_matrix_km = (pseudo_node_route_matrix_m / 1000.0).astype(np.float32)
    edge_lengths_km = pseudo_node_route_matrix_km[edge_index[:, 0], edge_index[:, 1]].astype(np.float32)

    unreachable_edge_mask = ~np.isfinite(edge_lengths_km)
    unreachable_edge_count = int(unreachable_edge_mask.sum())
    fallback_used = 0
    if unreachable_edge_count > 0:
        if str(unreachable_policy).strip().lower() == "haversine_fallback":
            fallback_distance = haversine_distance_matrix_km(pseudo_node_coords)
            edge_lengths_km[unreachable_edge_mask] = fallback_distance[
                edge_index[unreachable_edge_mask, 0],
                edge_index[unreachable_edge_mask, 1],
            ]
            for edge_idx in np.flatnonzero(unreachable_edge_mask):
                pseudo_node_route_matrix_km[edge_index[edge_idx, 0], edge_index[edge_idx, 1]] = edge_lengths_km[edge_idx]
            fallback_used = unreachable_edge_count
        else:
            sample_sensors = sensor_frame.loc[np.flatnonzero(unreachable_edge_mask)[:5], "sensor_id"].tolist()
            raise ValueError(
                "Map-centered pseudo-edge graph contains unreachable routed pseudo-edges. "
                f"count={unreachable_edge_count} sample_sensors={sample_sensors}"
            )

    adjacency = np.zeros((pseudo_node_xy.shape[0], pseudo_node_xy.shape[0]), dtype=np.float32)
    adjacency[edge_index[:, 0], edge_index[:, 1]] = 1.0
    adjacency_weights = np.zeros_like(adjacency, dtype=np.float32)
    finite_edge_lengths_m = edge_lengths_km[np.isfinite(edge_lengths_km) & (edge_lengths_km > 0)].astype(np.float64) * 1000.0
    if finite_edge_lengths_m.size >= 2 and float(finite_edge_lengths_m.std()) > 0:
        scale = float(finite_edge_lengths_m.std())
        weights = np.exp(-np.square((edge_lengths_km.astype(np.float64) * 1000.0) / scale)).astype(np.float32)
    else:
        weights = np.ones(edge_lengths_km.shape[0], dtype=np.float32)
    adjacency_weights[edge_index[:, 0], edge_index[:, 1]] = weights

    tail_coords = _projected_points_to_latlon(graph, tail_points_xy)
    head_coords = _projected_points_to_latlon(graph, head_points_xy)
    pseudo_edge_manifest = pd.DataFrame(
        {
            "pseudo_edge_id": np.arange(num_sensors, dtype=np.int32),
            "sensor_idx": np.arange(num_sensors, dtype=np.int32),
            "sensor_id": normalize_sensor_ids(sensor_frame["sensor_id"]),
            "tail_node": edge_index[:, 0].astype(np.int32),
            "head_node": edge_index[:, 1].astype(np.int32),
            "sensor_latitude": sensor_frame["latitude"].to_numpy(dtype=np.float32),
            "sensor_longitude": sensor_frame["longitude"].to_numpy(dtype=np.float32),
            "snapped_latitude": sensor_anchor_manifest["snapped_latitude"].to_numpy(dtype=np.float32),
            "snapped_longitude": sensor_anchor_manifest["snapped_longitude"].to_numpy(dtype=np.float32),
            "carrier_u": sensor_anchor_manifest["primary_u"].to_numpy(dtype=np.int32),
            "carrier_v": sensor_anchor_manifest["primary_v"].to_numpy(dtype=np.int32),
            "carrier_key": sensor_anchor_manifest["primary_key"].to_numpy(dtype=np.int32),
            "carrier_edge_length_m": sensor_anchor_manifest["primary_edge_length_m"].to_numpy(dtype=np.float32),
            "carrier_pos_m_from_u": sensor_anchor_manifest["primary_pos_m_from_u"].to_numpy(dtype=np.float32),
            "carrier_remaining_m_to_v": sensor_anchor_manifest["primary_remaining_m_to_v"].to_numpy(dtype=np.float32),
            "snap_distance_m": sensor_anchor_manifest["primary_snap_distance_m"].to_numpy(dtype=np.float32),
            "nearest_sensor_route_m": nearest_route_m.astype(np.float32),
            "target_half_length_m": target_half_lengths_m.astype(np.float32),
            "actual_half_length_m": actual_half_lengths_m.astype(np.float32),
            "tail_latitude": tail_coords[:, 0].astype(np.float32),
            "tail_longitude": tail_coords[:, 1].astype(np.float32),
            "head_latitude": head_coords[:, 0].astype(np.float32),
            "head_longitude": head_coords[:, 1].astype(np.float32),
        }
    )
    pseudo_node_metadata = pd.DataFrame(
        {
            "pseudo_node_id": np.arange(pseudo_node_xy.shape[0], dtype=np.int32),
            "latitude": pseudo_node_coords[:, 0].astype(np.float32),
            "longitude": pseudo_node_coords[:, 1].astype(np.float32),
            "endpoint_count": pseudo_node_counts.astype(np.int32),
            "tail_endpoint_count": pseudo_node_tail_counts.astype(np.int32),
            "head_endpoint_count": pseudo_node_head_counts.astype(np.int32),
            "snap_distance_m": pseudo_node_anchor_manifest["primary_snap_distance_m"].to_numpy(dtype=np.float32),
        }
    )

    return {
        "adjacency": adjacency.astype(np.float32),
        "adjacency_weights": adjacency_weights.astype(np.float32),
        "edge_index": edge_index.astype(np.int64),
        "edge_lengths_km": edge_lengths_km.astype(np.float32),
        "distance_matrix_km": pseudo_node_route_matrix_km.astype(np.float32),
        "distance_semantics": {
            "edge_lengths_km": "external_map_directed_route_distance_km_between_centered_pseudo_nodes",
            "distance_matrix_km": "external_map_directed_route_distance_km_between_centered_pseudo_nodes",
            "speed_values": "one_sensor_per_directed_map_centered_pseudo_edge",
            "adjacency_weights": "exp(-(map_centered_pseudo_edge_length/std)^2)",
        },
        "pseudo_node_metadata": pseudo_node_metadata,
        "pseudo_edge_manifest": pseudo_edge_manifest,
        "pseudo_node_anchor_manifest": pseudo_node_anchor_manifest,
        "sensor_anchor_manifest": sensor_anchor_manifest,
        "cluster_radius_m": np.float32(cluster_radius_value),
        "probe_half_length_m": actual_half_lengths_m.astype(np.float32),
        "self_loop_fixes": np.int32(self_loop_fixes),
        "topology_summary": _build_pseudo_edge_topology_summary(
            edge_index,
            num_nodes=pseudo_node_xy.shape[0],
        ),
        "map_snapshot_manifest": snapshot_context.get("manifest", {}),
        "routing_policy": "directed_shortest_path_between_centered_pseudo_nodes",
        "unreachable_policy": str(unreachable_policy),
        "unreachable_edge_count": int(unreachable_edge_count),
        "haversine_fallback_edge_count": int(fallback_used),
        "clipped_half_length_count": int(clipped_half_length_count),
    }
