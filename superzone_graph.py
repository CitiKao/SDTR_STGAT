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
