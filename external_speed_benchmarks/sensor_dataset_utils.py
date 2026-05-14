from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


OFFICIAL_DCRNN_FOLDER_URL = "https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX"
REPRESENTATION_DOMAINS = ("sensor_node", "pseudo_edge")

DATASET_SPECS = {
    "metr-la": {
        "display_name": "METR-LA",
        "traffic_filename": "metr-la.h5",
        "traffic_gdrive_id": "1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC",
        "location_filename": "graph_sensor_locations.csv",
        "location_url": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv",
        "official_sensor_ids_filename": "graph_sensor_ids.txt",
        "official_sensor_ids_url": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_ids.txt",
        "official_distance_filename": "distances_la_2012.csv",
        "official_distance_url": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_la_2012.csv",
        "official_graph_mode": "distance_csv",
        "official_normalized_k": 0.1,
        "requested_start": "2012-03-01 00:00:00",
        "requested_end": "2012-06-30 23:55:00",
    },
    "pems-bay": {
        "display_name": "PEMS-BAY",
        "traffic_filename": "pems-bay.h5",
        "traffic_gdrive_id": "1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq",
        "location_filename": "graph_sensor_locations_bay.csv",
        "location_url": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations_bay.csv",
        "official_adj_filename": "adj_mx_bay.pkl",
        "official_adj_url": "https://zenodo.org/api/records/4264005/files/adj_mx_bay.pkl/content",
        "official_adj_doi": "10.5281/zenodo.4264005",
        "official_graph_mode": "adj_pkl",
        "requested_start": "2017-01-01 00:00:00",
        "requested_end": "2017-06-30 23:55:00",
    },
}

TIME_FEATURE_NAMES = [
    "month_sin",
    "month_cos",
    "weekday_sin",
    "weekday_cos",
    "slot_sin",
    "slot_cos",
]
OUTLIER_CLEANING_MODES = ("none", "train_quantile_clip")


def normalize_sensor_ids(values: Iterable[object]) -> list[str]:
    return [str(value).strip() for value in values]


def normalize_representation_domain(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized not in REPRESENTATION_DOMAINS:
        supported = ", ".join(REPRESENTATION_DOMAINS)
        raise ValueError(f"Unsupported representation_domain: {value!r}. Expected one of {{{supported}}}.")
    return normalized


def build_processed_dataset_dir_name(display_name: str, representation_domain: str) -> str:
    normalized = normalize_representation_domain(representation_domain)
    if normalized == "sensor_node":
        return str(display_name)
    return f"{display_name}_{normalized}"


def validate_outlier_cleaning_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in OUTLIER_CLEANING_MODES:
        supported = ", ".join(OUTLIER_CLEANING_MODES)
        raise ValueError(f"Unsupported outlier cleaning mode: {mode!r}. Expected one of {{{supported}}}.")
    return normalized


def read_sensor_locations(
    csv_path: str | Path,
    *,
    dataset_name: str,
    expected_sensor_ids: Sequence[object],
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    if dataset_name == "metr-la":
        frame = pd.read_csv(csv_path)
        frame = frame.rename(
            columns={
                "sensor_id": "sensor_id",
                "latitude": "latitude",
                "longitude": "longitude",
            }
        )
    elif dataset_name == "pems-bay":
        frame = pd.read_csv(
            csv_path,
            header=None,
            names=["sensor_id", "latitude", "longitude"],
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    frame["sensor_id"] = normalize_sensor_ids(frame["sensor_id"])
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    if frame[["latitude", "longitude"]].isna().any().any():
        raise ValueError(f"{csv_path} contains invalid coordinates.")

    wanted = normalize_sensor_ids(expected_sensor_ids)
    frame = frame.drop_duplicates(subset=["sensor_id"]).set_index("sensor_id")
    missing = [sensor_id for sensor_id in wanted if sensor_id not in frame.index]
    if missing:
        raise ValueError(
            f"{csv_path} is missing {len(missing)} sensor locations, for example {missing[:5]}"
        )

    aligned = frame.loc[wanted].reset_index()
    aligned.insert(0, "sensor_idx", np.arange(len(aligned), dtype=np.int32))
    return aligned


def haversine_distance_matrix_km(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2) as [latitude, longitude].")

    lat = np.radians(coords[:, 0])[:, None]
    lon = np.radians(coords[:, 1])[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return (6371.0088 * 2.0 * np.arcsin(np.sqrt(a))).astype(np.float32)


def project_coords_to_local_xy_km(coords: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2) as [latitude, longitude].")
    lat0 = float(coords[:, 0].mean())
    lon0 = float(coords[:, 1].mean())
    lat_rad = np.radians(coords[:, 0])
    lon_rad = np.radians(coords[:, 1])
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    x = 6371.0088 * np.cos(lat0_rad) * (lon_rad - lon0_rad)
    y = 6371.0088 * (lat_rad - lat0_rad)
    xy = np.stack([x, y], axis=1).astype(np.float32)
    return xy, {"lat0": lat0, "lon0": lon0}


def local_xy_km_to_coords(xy: np.ndarray, *, reference: dict[str, float]) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2).")
    lat0 = float(reference["lat0"])
    lon0 = float(reference["lon0"])
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)
    lat = np.degrees((xy[:, 1] / 6371.0088) + lat0_rad)
    lon = np.degrees((xy[:, 0] / (6371.0088 * np.cos(lat0_rad))) + lon0_rad)
    return np.stack([lat, lon], axis=1).astype(np.float32)


def _cluster_points_by_radius(points_xy: np.ndarray, *, radius_km: float) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2).")
    if radius_km <= 0:
        raise ValueError("radius_km must be positive.")
    num_points = points_xy.shape[0]
    parent = np.arange(num_points, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b

    deltas = points_xy[:, None, :] - points_xy[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            if distances[i, j] <= radius_km:
                union(i, j)

    labels = np.zeros(num_points, dtype=np.int32)
    remap: dict[int, int] = {}
    next_label = 0
    for idx in range(num_points):
        root = find(idx)
        if root not in remap:
            remap[root] = next_label
            next_label += 1
        labels[idx] = remap[root]
    return labels


def _compute_sensor_direction_vectors(
    xy_coords: np.ndarray,
    *,
    adjacency_weights: np.ndarray,
    fallback_neighbor_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    adjacency_weights = np.asarray(adjacency_weights, dtype=np.float32)
    adjacency = adjacency_weights > 0
    num_sensors = xy_coords.shape[0]
    distance_matrix = np.linalg.norm(
        xy_coords[:, None, :] - xy_coords[None, :, :],
        axis=-1,
    ).astype(np.float32)
    np.fill_diagonal(distance_matrix, np.inf)
    direction_vectors = np.zeros((num_sensors, 2), dtype=np.float32)
    direction_strength = np.zeros(num_sensors, dtype=np.float32)
    local_scale = np.zeros(num_sensors, dtype=np.float32)

    effective_k = max(1, min(int(fallback_neighbor_k), max(num_sensors - 1, 1)))
    finite_distance_values = distance_matrix[np.isfinite(distance_matrix)]
    global_scale = float(np.median(finite_distance_values)) if finite_distance_values.size else 0.1
    if global_scale <= 0:
        global_scale = 0.1

    for sensor_idx in range(num_sensors):
        sorted_distances = np.sort(distance_matrix[sensor_idx][np.isfinite(distance_matrix[sensor_idx])])
        if sorted_distances.size:
            local_scale[sensor_idx] = float(np.median(sorted_distances[:effective_k]))
        else:
            local_scale[sensor_idx] = global_scale

        direction = np.zeros(2, dtype=np.float64)
        outgoing = np.flatnonzero(adjacency[sensor_idx])
        incoming = np.flatnonzero(adjacency[:, sensor_idx])
        if outgoing.size:
            outgoing_weights = adjacency_weights[sensor_idx, outgoing].astype(np.float64)
            if not np.any(outgoing_weights > 0):
                outgoing_weights = np.ones(outgoing.size, dtype=np.float64)
            direction += np.average(
                xy_coords[outgoing] - xy_coords[sensor_idx],
                axis=0,
                weights=outgoing_weights,
            )
        if incoming.size:
            incoming_weights = adjacency_weights[incoming, sensor_idx].astype(np.float64)
            if not np.any(incoming_weights > 0):
                incoming_weights = np.ones(incoming.size, dtype=np.float64)
            direction += np.average(
                xy_coords[sensor_idx] - xy_coords[incoming],
                axis=0,
                weights=incoming_weights,
            )
        if np.linalg.norm(direction) < 1e-6:
            neighbors = np.unique(np.concatenate([outgoing, incoming]))
            if neighbors.size:
                neighbor_offsets = xy_coords[neighbors] - xy_coords[sensor_idx]
                direction = neighbor_offsets.mean(axis=0)
        if np.linalg.norm(direction) < 1e-6:
            nearest = np.argsort(distance_matrix[sensor_idx])[:1]
            if nearest.size and np.isfinite(distance_matrix[sensor_idx, nearest[0]]):
                direction = xy_coords[nearest[0]] - xy_coords[sensor_idx]
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float64)

        strength = float(np.linalg.norm(direction))
        direction_strength[sensor_idx] = strength
        direction_vectors[sensor_idx] = (direction / max(strength, 1e-8)).astype(np.float32)

    return direction_vectors, direction_strength, local_scale


def _build_pseudo_edge_topology_summary(
    edge_index: np.ndarray,
    *,
    num_nodes: int,
) -> dict[str, int]:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    num_edges = int(edge_index.shape[0])
    predecessors = np.zeros(num_edges, dtype=np.int32)
    successors = np.zeros(num_edges, dtype=np.int32)
    line_graph_component_count = 0
    line_graph_largest_component_edges = 0
    line_graph_largest_component_ratio = 0.0
    if num_edges > 0:
        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        line_graph_links = dst[:, None] == src[None, :]
        predecessors = line_graph_links.sum(axis=1, dtype=np.int32)
        successors = line_graph_links.sum(axis=0, dtype=np.int32)
        weak_line_graph = np.logical_or(line_graph_links, line_graph_links.T)
        visited = np.zeros(num_edges, dtype=bool)
        for edge_id in range(num_edges):
            if visited[edge_id]:
                continue
            stack = [edge_id]
            visited[edge_id] = True
            component_size = 0
            while stack:
                current = stack.pop()
                component_size += 1
                neighbors = np.flatnonzero(weak_line_graph[current] & ~visited)
                if neighbors.size:
                    visited[neighbors] = True
                    stack.extend(int(neighbor) for neighbor in neighbors)
            line_graph_component_count += 1
            if component_size > line_graph_largest_component_edges:
                line_graph_largest_component_edges = component_size
        line_graph_largest_component_ratio = float(line_graph_largest_component_edges / max(num_edges, 1))
    unique_structural_edges = int(np.unique(edge_index, axis=0).shape[0]) if num_edges > 0 else 0
    return {
        "num_nodes": int(num_nodes),
        "num_edges": num_edges,
        "num_unique_structural_edges": unique_structural_edges,
        "duplicate_structural_edges": int(num_edges - unique_structural_edges),
        "source_only_edges": int(np.sum(predecessors == 0)),
        "sink_only_edges": int(np.sum(successors == 0)),
        "isolated_edges": int(np.sum((predecessors == 0) & (successors == 0))),
        "line_graph_links": int(predecessors.sum()),
        "line_graph_weak_components": int(line_graph_component_count),
        "line_graph_largest_component_edges": int(line_graph_largest_component_edges),
        "line_graph_largest_component_ratio": float(line_graph_largest_component_ratio),
    }


def build_pseudo_edge_graph(
    coords: np.ndarray,
    *,
    sensor_ids: Sequence[object],
    adjacency_weights: np.ndarray,
    cluster_radius_km: float | None = None,
    cluster_radius_scale: float = 0.75,
    probe_half_length_scale: float = 0.25,
    min_half_length_km: float = 0.05,
    max_half_length_km: float = 0.5,
    fallback_neighbor_k: int = 3,
) -> dict[str, object]:
    coords = np.asarray(coords, dtype=np.float64)
    adjacency_weights = np.asarray(adjacency_weights, dtype=np.float32)
    if adjacency_weights.ndim != 2 or adjacency_weights.shape[0] != adjacency_weights.shape[1]:
        raise ValueError("adjacency_weights must have shape (N, N).")
    if coords.shape[0] != adjacency_weights.shape[0]:
        raise ValueError("coords and adjacency_weights must have the same number of sensors.")
    num_sensors = int(coords.shape[0])
    if num_sensors <= 1:
        raise ValueError("At least two sensors are required for pseudo-edge construction.")

    sensor_ids = normalize_sensor_ids(sensor_ids)
    if len(sensor_ids) != num_sensors:
        raise ValueError("sensor_ids length must match coords.shape[0].")

    adjacency = (adjacency_weights > 0).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)
    xy_coords, reference = project_coords_to_local_xy_km(coords)
    direction_vectors, direction_strength, local_scale = _compute_sensor_direction_vectors(
        xy_coords,
        adjacency_weights=adjacency_weights,
        fallback_neighbor_k=fallback_neighbor_k,
    )

    half_lengths = np.clip(
        local_scale * float(probe_half_length_scale),
        float(min_half_length_km),
        float(max_half_length_km),
    ).astype(np.float32)
    if cluster_radius_km is None:
        cluster_radius_value = float(np.median(half_lengths) * float(cluster_radius_scale))
    else:
        cluster_radius_value = float(cluster_radius_km)
    cluster_radius_value = float(np.clip(cluster_radius_value, 0.03, 0.25))

    tail_xy = xy_coords - direction_vectors * half_lengths[:, None]
    head_xy = xy_coords + direction_vectors * half_lengths[:, None]
    endpoint_xy = np.concatenate([tail_xy, head_xy], axis=0)
    endpoint_labels = _cluster_points_by_radius(endpoint_xy, radius_km=cluster_radius_value)

    tail_labels = endpoint_labels[:num_sensors].astype(np.int32).copy()
    head_labels = endpoint_labels[num_sensors:].astype(np.int32).copy()
    next_label = int(endpoint_labels.max()) + 1 if endpoint_labels.size else 0
    self_loop_fixes = 0
    for sensor_idx in range(num_sensors):
        if int(tail_labels[sensor_idx]) == int(head_labels[sensor_idx]):
            head_labels[sensor_idx] = next_label
            next_label += 1
            self_loop_fixes += 1

    raw_node_coords: dict[int, list[np.ndarray]] = {}
    endpoint_roles: dict[int, dict[str, int]] = {}
    for sensor_idx in range(num_sensors):
        tail_label = int(tail_labels[sensor_idx])
        head_label = int(head_labels[sensor_idx])
        raw_node_coords.setdefault(tail_label, []).append(tail_xy[sensor_idx])
        raw_node_coords.setdefault(head_label, []).append(head_xy[sensor_idx])
        endpoint_roles.setdefault(tail_label, {"tail_count": 0, "head_count": 0})
        endpoint_roles.setdefault(head_label, {"tail_count": 0, "head_count": 0})
        endpoint_roles[tail_label]["tail_count"] += 1
        endpoint_roles[head_label]["head_count"] += 1

    ordered_raw_labels = sorted(raw_node_coords)
    label_remap = {raw_label: idx for idx, raw_label in enumerate(ordered_raw_labels)}
    pseudo_node_xy = np.zeros((len(ordered_raw_labels), 2), dtype=np.float32)
    pseudo_node_counts = np.zeros(len(ordered_raw_labels), dtype=np.int32)
    pseudo_node_tail_counts = np.zeros(len(ordered_raw_labels), dtype=np.int32)
    pseudo_node_head_counts = np.zeros(len(ordered_raw_labels), dtype=np.int32)
    for raw_label, node_idx in label_remap.items():
        points = np.stack(raw_node_coords[raw_label], axis=0).astype(np.float32)
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
    pseudo_node_coords = local_xy_km_to_coords(pseudo_node_xy, reference=reference)
    pseudo_node_distance_matrix = haversine_distance_matrix_km(pseudo_node_coords)
    edge_lengths = pseudo_node_distance_matrix[edge_index[:, 0], edge_index[:, 1]].astype(np.float32)

    adjacency_matrix = np.zeros((pseudo_node_xy.shape[0], pseudo_node_xy.shape[0]), dtype=np.float32)
    adjacency_matrix[edge_index[:, 0], edge_index[:, 1]] = 1.0
    adjacency_weights_out = adjacency_matrix.copy()

    pseudo_edge_manifest = pd.DataFrame(
        {
            "pseudo_edge_id": np.arange(num_sensors, dtype=np.int32),
            "sensor_idx": np.arange(num_sensors, dtype=np.int32),
            "sensor_id": sensor_ids,
            "tail_node": edge_index[:, 0].astype(np.int32),
            "head_node": edge_index[:, 1].astype(np.int32),
            "sensor_latitude": coords[:, 0].astype(np.float32),
            "sensor_longitude": coords[:, 1].astype(np.float32),
            "direction_x_km": direction_vectors[:, 0].astype(np.float32),
            "direction_y_km": direction_vectors[:, 1].astype(np.float32),
            "direction_strength": direction_strength.astype(np.float32),
            "half_length_km": half_lengths.astype(np.float32),
            "probe_length_km": (2.0 * half_lengths).astype(np.float32),
            "tail_latitude": local_xy_km_to_coords(tail_xy, reference=reference)[:, 0].astype(np.float32),
            "tail_longitude": local_xy_km_to_coords(tail_xy, reference=reference)[:, 1].astype(np.float32),
            "head_latitude": local_xy_km_to_coords(head_xy, reference=reference)[:, 0].astype(np.float32),
            "head_longitude": local_xy_km_to_coords(head_xy, reference=reference)[:, 1].astype(np.float32),
            "base_out_degree": adjacency.sum(axis=1).astype(np.int32),
            "base_in_degree": adjacency.sum(axis=0).astype(np.int32),
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
        }
    )

    return {
        "adjacency": adjacency_matrix,
        "adjacency_weights": adjacency_weights_out,
        "edge_index": edge_index.astype(np.int64),
        "edge_lengths_km": edge_lengths.astype(np.float32),
        "distance_matrix_km": pseudo_node_distance_matrix.astype(np.float32),
        "pseudo_node_metadata": pseudo_node_metadata,
        "pseudo_edge_manifest": pseudo_edge_manifest,
        "cluster_radius_km": np.float32(cluster_radius_value),
        "probe_half_length_km": half_lengths.astype(np.float32),
        "probe_reference": reference,
        "self_loop_fixes": np.int32(self_loop_fixes),
        "topology_summary": _build_pseudo_edge_topology_summary(
            edge_index,
            num_nodes=pseudo_node_xy.shape[0],
        ),
    }


def build_directed_knn_graph(
    coords: np.ndarray,
    *,
    k: int,
    symmetrize: bool = True,
) -> dict[str, np.ndarray]:
    if k < 1:
        raise ValueError("k must be >= 1.")
    coords = np.asarray(coords, dtype=np.float64)
    num_nodes = coords.shape[0]
    if num_nodes <= 1:
        raise ValueError("At least two sensor coordinates are required.")

    effective_k = min(int(k), num_nodes - 1)
    distances = haversine_distance_matrix_km(coords)
    nearest = np.argsort(distances, axis=1)[:, 1 : effective_k + 1]

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for recv in range(num_nodes):
        adj[recv, nearest[recv]] = 1.0
    if symmetrize:
        adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0.0)

    recv, send = np.where(adj > 0)
    edge_index = np.stack([recv, send], axis=1).astype(np.int64)
    edge_lengths = distances[recv, send].astype(np.float32)

    return {
        "adjacency": adj,
        "adjacency_weights": adj.copy(),
        "edge_index": edge_index,
        "edge_lengths_km": edge_lengths,
        "distance_matrix_km": distances.astype(np.float32),
        "effective_k": np.int32(effective_k),
    }


def build_graph_from_weighted_adjacency(
    adjacency_weights: np.ndarray,
    *,
    coords: np.ndarray,
    edge_length_strategy: str = "haversine",
) -> dict[str, np.ndarray]:
    adjacency_weights = np.asarray(adjacency_weights, dtype=np.float32)
    if adjacency_weights.ndim != 2 or adjacency_weights.shape[0] != adjacency_weights.shape[1]:
        raise ValueError("adjacency_weights must have shape (N, N).")
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] != adjacency_weights.shape[0]:
        raise ValueError("coords and adjacency_weights must have the same number of nodes.")
    strategy = str(edge_length_strategy).strip().lower()
    if strategy not in {"haversine", "none"}:
        raise ValueError("edge_length_strategy must be either 'haversine' or 'none'.")

    adjacency = (adjacency_weights > 0).astype(np.float32)
    np.fill_diagonal(adjacency, 0.0)
    adjacency_weights = adjacency_weights.astype(np.float32).copy()
    np.fill_diagonal(adjacency_weights, 0.0)

    recv, send = np.where(adjacency > 0)
    edge_index = np.stack([recv, send], axis=1).astype(np.int64)
    if strategy == "haversine":
        distance_matrix_km = haversine_distance_matrix_km(coords)
        edge_lengths = distance_matrix_km[recv, send].astype(np.float32)
        distance_semantics = {
            "edge_lengths_km": "sensor_geodesic_haversine_km",
            "distance_matrix_km": "sensor_geodesic_haversine_km",
            "adjacency_weights": "official_weighted_adjacency",
        }
    else:
        distance_matrix_km = np.zeros_like(adjacency_weights, dtype=np.float32)
        edge_lengths = np.zeros(edge_index.shape[0], dtype=np.float32)
        distance_semantics = {
            "edge_lengths_km": "disabled_zero_no_custom_length_feature",
            "distance_matrix_km": "disabled_zero_no_custom_distance_matrix",
            "adjacency_weights": "official_weighted_adjacency",
        }
    return {
        "adjacency": adjacency,
        "adjacency_weights": adjacency_weights,
        "edge_index": edge_index,
        "edge_lengths_km": edge_lengths,
        "distance_matrix_km": distance_matrix_km.astype(np.float32),
        "distance_semantics": distance_semantics,
    }


def build_official_distance_graph(
    distance_frame: pd.DataFrame,
    *,
    sensor_ids: Sequence[object],
    coords: np.ndarray,
    normalized_k: float = 0.1,
) -> dict[str, np.ndarray]:
    if normalized_k <= 0:
        raise ValueError("normalized_k must be positive.")
    distance_frame = distance_frame.rename(columns={"cost": "distance"})
    required_columns = {"from", "to", "distance"}
    if not required_columns.issubset(distance_frame.columns):
        raise ValueError(
            "distance_frame must contain columns ['from', 'to', 'distance'] or ['from', 'to', 'cost']."
        )

    wanted = normalize_sensor_ids(sensor_ids)
    sensor_id_to_ind = {sensor_id: idx for idx, sensor_id in enumerate(wanted)}
    num_nodes = len(wanted)
    dist_mx = np.full((num_nodes, num_nodes), np.inf, dtype=np.float32)
    for row in distance_frame[["from", "to", "distance"]].itertuples(index=False):
        src = str(row[0]).strip()
        dst = str(row[1]).strip()
        if src not in sensor_id_to_ind or dst not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[src], sensor_id_to_ind[dst]] = float(row[2]) / 1000.0

    finite = dist_mx[np.isfinite(dist_mx)]
    if finite.size == 0:
        raise ValueError("No finite official distances matched the requested sensor ids.")
    std = float(finite.std())
    if std <= 0:
        raise ValueError("Official distance standard deviation must be positive.")

    adjacency_weights = np.exp(-np.square(dist_mx / std)).astype(np.float32)
    adjacency_weights[adjacency_weights < normalized_k] = 0.0
    np.fill_diagonal(adjacency_weights, 0.0)
    graph = build_graph_from_weighted_adjacency(adjacency_weights, coords=coords)
    if graph["edge_index"].shape[0] > 0:
        official_edge_lengths = dist_mx[
            graph["edge_index"][:, 0],
            graph["edge_index"][:, 1],
        ].astype(np.float32)
        if not np.all(np.isfinite(official_edge_lengths)):
            raise ValueError("Official graph contains non-finite road distances for retained edges.")
        graph["edge_lengths_km"] = official_edge_lengths
    graph["distance_matrix_km"] = dist_mx.astype(np.float32)
    graph["distance_semantics"] = {
        "edge_lengths_km": "official_road_distance_km",
        "distance_matrix_km": "official_road_distance_km",
        "adjacency_weights": "exp(-(road_distance/std)^2) thresholded",
    }
    graph["normalized_k"] = np.float32(normalized_k)
    return graph


def load_official_adjacency_pickle(
    pickle_path: str | Path,
) -> tuple[list[str], dict[str, int], np.ndarray]:
    pickle_path = Path(pickle_path)
    if not pickle_path.exists():
        raise FileNotFoundError(pickle_path)
    with pickle_path.open("rb") as handle:
        sensor_ids, sensor_id_to_ind, adjacency = pickle.load(handle, encoding="latin1")
    return normalize_sensor_ids(sensor_ids), dict(sensor_id_to_ind), np.asarray(adjacency, dtype=np.float32)


def compute_train_quantile_clip_bounds(
    speed_values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    train_time_mask: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, np.ndarray]:
    values = np.asarray(speed_values, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    train_time_mask = np.asarray(train_time_mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError("speed_values must have shape (T, N).")
    if mask.shape != values.shape:
        raise ValueError("valid_mask must have the same shape as speed_values.")
    if train_time_mask.ndim != 1 or train_time_mask.shape[0] != values.shape[0]:
        raise ValueError("train_time_mask must be a 1D boolean array aligned with the time axis.")
    if not np.any(train_time_mask):
        raise ValueError("train_time_mask does not select any rows.")
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("lower_quantile and upper_quantile must satisfy 0 < lower < upper < 1.")

    lower = np.full(values.shape[1], np.nan, dtype=np.float32)
    upper = np.full(values.shape[1], np.nan, dtype=np.float32)
    valid_counts = np.zeros(values.shape[1], dtype=np.int32)
    train_values = values[train_time_mask]
    train_valid = mask[train_time_mask]
    for sensor_idx in range(values.shape[1]):
        sensor_train = train_values[:, sensor_idx]
        sensor_valid = train_valid[:, sensor_idx]
        observed = sensor_train[sensor_valid]
        valid_counts[sensor_idx] = int(observed.size)
        if observed.size == 0:
            continue
        lower[sensor_idx] = float(np.quantile(observed, lower_quantile))
        upper[sensor_idx] = float(np.quantile(observed, upper_quantile))
        upper[sensor_idx] = max(upper[sensor_idx], lower[sensor_idx])
    return {
        "lower": lower.astype(np.float32),
        "upper": upper.astype(np.float32),
        "valid_counts": valid_counts,
    }


def apply_valid_speed_clip(
    speed_values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> dict[str, object]:
    values = np.asarray(speed_values, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=bool)
    lower = np.asarray(lower_bounds, dtype=np.float32)
    upper = np.asarray(upper_bounds, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("speed_values must have shape (T, N).")
    if mask.shape != values.shape:
        raise ValueError("valid_mask must have the same shape as speed_values.")
    if lower.shape != (values.shape[1],) or upper.shape != (values.shape[1],):
        raise ValueError("lower_bounds and upper_bounds must each have shape (N,).")

    cleaned = values.copy()
    clipped_mask = np.zeros_like(mask, dtype=bool)
    for sensor_idx in range(values.shape[1]):
        sensor_valid = mask[:, sensor_idx]
        if not np.any(sensor_valid):
            continue
        if not np.isfinite(lower[sensor_idx]) or not np.isfinite(upper[sensor_idx]):
            continue
        sensor_values = cleaned[:, sensor_idx]
        sensor_original = sensor_values.copy()
        sensor_values[sensor_valid] = np.clip(
            sensor_values[sensor_valid],
            lower[sensor_idx],
            upper[sensor_idx],
        )
        cleaned[:, sensor_idx] = sensor_values
        clipped_mask[:, sensor_idx] = sensor_valid & (~np.isclose(sensor_values, sensor_original))

    clipped = int(clipped_mask.sum())
    valid_total = int(mask.sum())
    return {
        "cleaned_speed_values": cleaned.astype(np.float32),
        "outlier_mask": clipped_mask.astype(bool),
        "summary": {
            "mode": "train_quantile_clip",
            "replacement": "clip",
            "num_flagged": clipped,
            "flagged_ratio_all": float(clipped / max(clipped_mask.size, 1)),
            "flagged_ratio_valid": float(clipped / max(valid_total, 1)),
        },
    }


def build_time_meta(timestamps: Sequence[object]) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    if index.empty:
        raise ValueError("timestamps must not be empty.")
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        slot_minutes = 5
    else:
        slot_minutes = int(round(diffs.dt.total_seconds().mode().iloc[0] / 60.0))
    if slot_minutes <= 0:
        raise ValueError(f"Invalid slot_minutes inferred from timestamps: {slot_minutes}")

    minute_of_day = index.hour * 60 + index.minute
    slots_per_day = max(1, int(round(1440 / slot_minutes)))
    slot = (minute_of_day // slot_minutes).astype(np.int32)
    slot = np.clip(slot, 0, slots_per_day - 1)

    return pd.DataFrame(
        {
            "time_idx": np.arange(len(index), dtype=np.int32),
            "timestamp": index,
            "date": index.normalize().strftime("%Y-%m-%d"),
            "slot": slot,
            "day_of_week": index.dayofweek.astype(np.int32),
            "day_name": index.day_name(),
            "hour": index.hour.astype(np.int32),
            "minute": index.minute.astype(np.int32),
        }
    )


def build_cyclical_time_features(time_meta: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    dates = pd.to_datetime(time_meta["date"], errors="raise")
    month = dates.dt.month.to_numpy(dtype=np.float32)
    weekday = pd.to_numeric(time_meta["day_of_week"], errors="raise").to_numpy(dtype=np.float32)
    slot = pd.to_numeric(time_meta["slot"], errors="raise").to_numpy(dtype=np.float32)
    slots_per_day = float(slot.max()) + 1.0
    if slots_per_day <= 0:
        raise ValueError("Invalid slots_per_day derived from time_meta.")

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
        axis=-1,
    ).astype(np.float32)
    return features, list(TIME_FEATURE_NAMES)


def filter_dataframe_by_date_range(
    frame: pd.DataFrame,
    *,
    start: str | None,
    end: str | None,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
    filtered = frame.copy()
    if start is not None:
        filtered = filtered.loc[filtered.index >= pd.Timestamp(start)]
    if end is not None:
        filtered = filtered.loc[filtered.index <= pd.Timestamp(end)]
    actual_start = filtered.index.min() if not filtered.empty else None
    actual_end = filtered.index.max() if not filtered.empty else None
    return filtered, actual_start, actual_end


def save_json(payload: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
