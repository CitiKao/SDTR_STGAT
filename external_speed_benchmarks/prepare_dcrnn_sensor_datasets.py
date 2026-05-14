from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_speed_benchmarks.sensor_dataset_utils import (
    DATASET_SPECS,
    OFFICIAL_DCRNN_FOLDER_URL,
    build_processed_dataset_dir_name,
    build_graph_from_weighted_adjacency,
    build_official_distance_graph,
    build_pseudo_edge_graph,
    build_directed_knn_graph,
    build_time_meta,
    filter_dataframe_by_date_range,
    load_official_adjacency_pickle,
    normalize_sensor_ids,
    normalize_representation_domain,
    read_sensor_locations,
    save_json,
)
from external_speed_benchmarks.map_graph_utils import (
    DEFAULT_OSM_DRIVE_PADDING_KM,
    build_map_carrier_split_pseudo_edge_graph,
    build_map_centered_pseudo_edge_graph,
    build_sensor_node_map_distance_graph,
    load_or_build_osm_drive_snapshot,
)


def ensure_official_assets(
    dataset_name: str,
    *,
    raw_root: Path,
) -> dict[str, Path]:
    spec = DATASET_SPECS[dataset_name]
    raw_root.mkdir(parents=True, exist_ok=True)
    traffic_path = raw_root / spec["traffic_filename"]
    location_path = raw_root / spec["location_filename"]
    asset_paths: dict[str, Path] = {
        "traffic": traffic_path,
        "locations": location_path,
    }

    if not traffic_path.exists():
        import gdown

        print(f"Downloading {spec['display_name']} traffic file from the official DCRNN folder...")
        gdown.download(
            id=spec["traffic_gdrive_id"],
            output=str(traffic_path),
            quiet=False,
        )
    if not location_path.exists():
        print(f"Downloading {spec['display_name']} sensor locations from GitHub...")
        urlretrieve(spec["location_url"], str(location_path))

    if dataset_name == "metr-la":
        sensor_ids_path = raw_root / spec["official_sensor_ids_filename"]
        distance_path = raw_root / spec["official_distance_filename"]
        if not sensor_ids_path.exists():
            print("Downloading METR-LA official sensor id list from GitHub...")
            urlretrieve(spec["official_sensor_ids_url"], str(sensor_ids_path))
        if not distance_path.exists():
            print("Downloading METR-LA official road-distance CSV from GitHub...")
            urlretrieve(spec["official_distance_url"], str(distance_path))
        asset_paths["official_sensor_ids"] = sensor_ids_path
        asset_paths["official_distances"] = distance_path
    elif dataset_name == "pems-bay":
        adj_path = raw_root / spec["official_adj_filename"]
        if not adj_path.exists():
            print("Downloading PEMS-BAY official adjacency pickle from Zenodo...")
            urlretrieve(spec["official_adj_url"], str(adj_path))
        asset_paths["official_adjacency"] = adj_path

    return asset_paths


def build_sensor_map(sensor_frame: pd.DataFrame, *, dataset_label: str, output_path: Path) -> None:
    try:
        import folium
    except ModuleNotFoundError:
        output_path.write_text(
            (
                "<html><body><p>folium is not installed in the current environment, "
                f"so the interactive sensor map for {dataset_label} was skipped.</p></body></html>"
            ),
            encoding="utf-8",
        )
        return

    center = [float(sensor_frame["latitude"].mean()), float(sensor_frame["longitude"].mean())]
    sensor_map = folium.Map(location=center, zoom_start=10, tiles="CartoDB positron")
    for row in sensor_frame.itertuples(index=False):
        folium.CircleMarker(
            location=[float(row.latitude), float(row.longitude)],
            radius=3,
            color="#d1495b",
            fill=True,
            fill_opacity=0.8,
            popup=f"{dataset_label} | sensor_id={row.sensor_id} | idx={row.sensor_idx}",
        ).add_to(sensor_map)
    sensor_map.save(str(output_path))


def resolve_prepared_dataset_dir_name(
    display_name: str,
    *,
    representation_domain: str,
    sensor_node_distance_source: str,
    pseudo_edge_construction: str,
) -> str:
    base_name = build_processed_dataset_dir_name(display_name, representation_domain)
    if representation_domain == "sensor_node" and sensor_node_distance_source == "haversine":
        return f"{display_name}_sensor_node_haversine"
    if representation_domain == "sensor_node" and sensor_node_distance_source == "map_drive":
        return f"{display_name}_sensor_node_map_route"
    if representation_domain == "pseudo_edge" and pseudo_edge_construction == "map_centered":
        return f"{display_name}_pseudo_edge_map_centered"
    if representation_domain == "pseudo_edge" and pseudo_edge_construction == "map_carrier_split":
        return f"{display_name}_pseudo_edge_map_carrier_split"
    return base_name


def prepare_dataset(
    dataset_name: str,
    *,
    raw_root: Path,
    processed_root: Path,
    graph_mode: str,
    graph_k: int,
    symmetrize: bool,
    official_normalized_k: float,
    representation_domain: str,
    pseudo_edge_cluster_radius_km: float | None,
    pseudo_edge_cluster_radius_scale: float,
    pseudo_edge_half_length_scale: float,
    pseudo_edge_min_half_length_km: float,
    pseudo_edge_max_half_length_km: float,
    pseudo_edge_fallback_neighbor_k: int,
    sensor_node_distance_source: str,
    pseudo_edge_construction: str,
    map_cache_root: Path,
    map_graph_padding_km: float,
    map_unreachable_policy: str,
) -> Path:
    spec = DATASET_SPECS[dataset_name]
    representation_domain = normalize_representation_domain(representation_domain)
    if graph_mode not in {"official", "haversine_knn"}:
        raise ValueError("graph_mode must be either 'official' or 'haversine_knn'.")
    sensor_node_distance_source = str(sensor_node_distance_source).strip().lower()
    if sensor_node_distance_source not in {"prepared", "haversine", "map_drive"}:
        raise ValueError(
            "sensor_node_distance_source must be one of: 'prepared', 'haversine', 'map_drive'."
        )
    pseudo_edge_construction = str(pseudo_edge_construction).strip().lower()
    if pseudo_edge_construction not in {"heuristic", "map_centered", "map_carrier_split"}:
        raise ValueError(
            "pseudo_edge_construction must be one of: 'heuristic', 'map_centered', 'map_carrier_split'."
        )
    map_unreachable_policy = str(map_unreachable_policy).strip().lower()
    if map_unreachable_policy not in {"raise", "haversine_fallback"}:
        raise ValueError("map_unreachable_policy must be either 'raise' or 'haversine_fallback'.")
    assets = ensure_official_assets(dataset_name, raw_root=raw_root)
    traffic_path = assets["traffic"]
    location_path = assets["locations"]
    dataset_dir = processed_root / resolve_prepared_dataset_dir_name(
        spec["display_name"],
        representation_domain=representation_domain,
        sensor_node_distance_source=sensor_node_distance_source,
        pseudo_edge_construction=pseudo_edge_construction,
    )
    dataset_dir.mkdir(parents=True, exist_ok=True)

    traffic_df = pd.read_hdf(traffic_path).sort_index()
    filtered_df, actual_start, actual_end = filter_dataframe_by_date_range(
        traffic_df,
        start=spec["requested_start"],
        end=spec["requested_end"],
    )
    if filtered_df.empty:
        raise ValueError(f"{spec['display_name']} has no samples in the requested date range.")

    sensor_frame = read_sensor_locations(
        location_path,
        dataset_name=dataset_name,
        expected_sensor_ids=filtered_df.columns,
    )
    coords = sensor_frame[["latitude", "longitude"]].to_numpy(dtype=np.float64)
    selected_sensor_ids = normalize_sensor_ids(filtered_df.columns)
    if graph_mode == "official":
        if dataset_name == "metr-la":
            official_sensor_ids = normalize_sensor_ids(
                assets["official_sensor_ids"].read_text(encoding="utf-8").strip().split(",")
            )
            missing_ids = sorted(set(selected_sensor_ids) - set(official_sensor_ids))
            if missing_ids:
                raise ValueError(
                    f"METR-LA official sensor id list is missing {len(missing_ids)} selected ids, "
                    f"for example {missing_ids[:5]}"
                )
            distance_frame = pd.read_csv(
                assets["official_distances"],
                dtype={"from": "str", "to": "str"},
            )
            graph = build_official_distance_graph(
                distance_frame,
                sensor_ids=selected_sensor_ids,
                coords=coords,
                normalized_k=official_normalized_k,
            )
            graph_source = {
                "mode": "official_distance_csv",
                "sensor_ids_file": str(assets["official_sensor_ids"]),
                "distance_file": str(assets["official_distances"]),
                "normalized_k": float(official_normalized_k),
                "reference": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/scripts/gen_adj_mx.py",
            }
        else:
            official_sensor_ids, _, adjacency_weights = load_official_adjacency_pickle(
                assets["official_adjacency"]
            )
            if set(official_sensor_ids) != set(selected_sensor_ids):
                missing_ids = sorted(set(selected_sensor_ids) - set(official_sensor_ids))
                extra_ids = sorted(set(official_sensor_ids) - set(selected_sensor_ids))
                raise ValueError(
                    "PEMS-BAY official adjacency sensor ids do not match the traffic HDF5 columns: "
                    f"missing={missing_ids[:5]} extra={extra_ids[:5]}"
                )
            order = [official_sensor_ids.index(sensor_id) for sensor_id in selected_sensor_ids]
            graph = build_graph_from_weighted_adjacency(
                adjacency_weights[np.ix_(order, order)],
                coords=coords,
                edge_length_strategy="none",
            )
            graph_source = {
                "mode": "official_adj_pkl",
                "adjacency_file": str(assets["official_adjacency"]),
                "doi": spec["official_adj_doi"],
            }
    else:
        graph = build_directed_knn_graph(
            coords,
            k=graph_k,
            symmetrize=symmetrize,
        )
        graph_source = {
            "mode": "haversine_knn",
            "k": int(graph["effective_k"]),
            "symmetrize": bool(symmetrize),
        }
    if "distance_semantics" in graph:
        graph_source["distance_semantics"] = graph["distance_semantics"]
    sensor_graph = graph
    sensor_graph_source = graph_source

    map_snapshot_manifest_payload = None
    pseudo_edge_summary_payload = None
    if representation_domain == "sensor_node" and sensor_node_distance_source == "haversine":
        graph = build_graph_from_weighted_adjacency(
            sensor_graph["adjacency_weights"],
            coords=coords,
            edge_length_strategy="haversine",
        )
        graph_source = {
            "mode": "haversine_distance_sensor_node",
            "topology_source": sensor_graph_source,
            "distance_semantics": graph["distance_semantics"],
        }
    elif representation_domain == "sensor_node" and sensor_node_distance_source == "map_drive":
        snapshot_context = load_or_build_osm_drive_snapshot(
            sensor_frame,
            dataset_label=spec["display_name"],
            cache_root=map_cache_root,
            padding_km=float(map_graph_padding_km),
        )
        map_snapshot_manifest_payload = snapshot_context.get("manifest", {})
        graph = build_sensor_node_map_distance_graph(
            sensor_frame,
            topology_adjacency_weights=sensor_graph["adjacency_weights"],
            snapshot_context=snapshot_context,
            unreachable_policy=map_unreachable_policy,
        )
        graph_source = {
            "mode": "map_route_sensor_node",
            "topology_source": sensor_graph_source,
            "map_provider": "osm_drive",
            "graph_snapshot_manifest_file": str(snapshot_context["manifest_path"]),
            "routing_policy": str(graph["routing_policy"]),
            "unreachable_policy": str(graph["unreachable_policy"]),
            "unreachable_edge_count": int(graph["unreachable_edge_count"]),
            "haversine_fallback_edge_count": int(graph["haversine_fallback_edge_count"]),
            "distance_semantics": graph["distance_semantics"],
        }
    elif representation_domain == "pseudo_edge":
        if pseudo_edge_construction == "map_centered":
            snapshot_context = load_or_build_osm_drive_snapshot(
                sensor_frame,
                dataset_label=spec["display_name"],
                cache_root=map_cache_root,
                padding_km=float(map_graph_padding_km),
            )
            map_snapshot_manifest_payload = snapshot_context.get("manifest", {})
            graph = build_map_centered_pseudo_edge_graph(
                sensor_frame,
                sensor_ids=selected_sensor_ids,
                snapshot_context=snapshot_context,
                cluster_radius_m=(
                    None if pseudo_edge_cluster_radius_km is None else float(pseudo_edge_cluster_radius_km) * 1000.0
                ),
                cluster_radius_scale=float(pseudo_edge_cluster_radius_scale),
                half_length_scale=float(pseudo_edge_half_length_scale),
                min_half_length_m=float(pseudo_edge_min_half_length_km) * 1000.0,
                max_half_length_m=float(pseudo_edge_max_half_length_km) * 1000.0,
                unreachable_policy=map_unreachable_policy,
            )
            graph_source = {
                "mode": "map_centered_pseudo_edge",
                "sensor_base_graph": sensor_graph_source,
                "map_provider": "osm_drive",
                "graph_snapshot_manifest_file": str(snapshot_context["manifest_path"]),
                "cluster_radius_km": float(graph["cluster_radius_m"]) / 1000.0,
                "cluster_radius_scale": float(pseudo_edge_cluster_radius_scale),
                "probe_half_length_scale": float(pseudo_edge_half_length_scale),
                "probe_min_half_length_km": float(pseudo_edge_min_half_length_km),
                "probe_max_half_length_km": float(pseudo_edge_max_half_length_km),
                "routing_policy": str(graph["routing_policy"]),
                "unreachable_policy": str(graph["unreachable_policy"]),
                "unreachable_edge_count": int(graph["unreachable_edge_count"]),
                "haversine_fallback_edge_count": int(graph["haversine_fallback_edge_count"]),
                "clipped_half_length_count": int(graph["clipped_half_length_count"]),
                "distance_semantics": graph["distance_semantics"],
                "self_loop_fixes": int(graph["self_loop_fixes"]),
            }
        elif pseudo_edge_construction == "map_carrier_split":
            snapshot_context = load_or_build_osm_drive_snapshot(
                sensor_frame,
                dataset_label=spec["display_name"],
                cache_root=map_cache_root,
                padding_km=float(map_graph_padding_km),
            )
            map_snapshot_manifest_payload = snapshot_context.get("manifest", {})
            graph = build_map_carrier_split_pseudo_edge_graph(
                sensor_frame,
                sensor_ids=selected_sensor_ids,
                snapshot_context=snapshot_context,
                topology_adjacency_weights=sensor_graph["adjacency_weights"],
                unreachable_policy=map_unreachable_policy,
            )
            graph_source = {
                "mode": "map_carrier_split_pseudo_edge",
                "sensor_base_graph": sensor_graph_source,
                "map_provider": "osm_drive",
                "graph_snapshot_manifest_file": str(snapshot_context["manifest_path"]),
                "routing_policy": str(graph["routing_policy"]),
                "unreachable_policy": str(graph["unreachable_policy"]),
                "unreachable_edge_count": int(graph["unreachable_edge_count"]),
                "unreachable_pair_count": int(graph.get("unreachable_pair_count", 0)),
                "haversine_fallback_edge_count": int(graph["haversine_fallback_edge_count"]),
                "builder_algorithm": str(graph["builder_algorithm"]),
                "min_segment_length_m": float(graph["min_segment_length_m"]),
                "dedupe_position_eps_m": float(graph["dedupe_position_eps_m"]),
                "max_candidate_edges": int(graph["max_candidate_edges"]),
                "max_snap_distance_m": float(graph["max_snap_distance_m"]),
                "candidate_search_margin_m": float(graph["candidate_search_margin_m"]),
                "topology_neighbor_top_k": int(graph["topology_neighbor_top_k"]),
                "assignment_passes_used": int(graph["assignment_passes_used"]),
                "assignment_route_cap_m": float(graph["assignment_route_cap_m"]),
                "distance_semantics": graph["distance_semantics"],
                "self_loop_fixes": int(graph["self_loop_fixes"]),
            }
        else:
            graph = build_pseudo_edge_graph(
                coords,
                sensor_ids=selected_sensor_ids,
                adjacency_weights=sensor_graph["adjacency_weights"],
                cluster_radius_km=pseudo_edge_cluster_radius_km,
                cluster_radius_scale=pseudo_edge_cluster_radius_scale,
                probe_half_length_scale=pseudo_edge_half_length_scale,
                min_half_length_km=pseudo_edge_min_half_length_km,
                max_half_length_km=pseudo_edge_max_half_length_km,
                fallback_neighbor_k=pseudo_edge_fallback_neighbor_k,
            )
            graph_source = {
                "mode": "pseudo_edge_from_sensor_graph",
                "sensor_base_graph": sensor_graph_source,
                "cluster_radius_km": float(graph["cluster_radius_km"]),
                "cluster_radius_scale": float(pseudo_edge_cluster_radius_scale),
                "probe_half_length_scale": float(pseudo_edge_half_length_scale),
                "probe_min_half_length_km": float(pseudo_edge_min_half_length_km),
                "probe_max_half_length_km": float(pseudo_edge_max_half_length_km),
                "fallback_neighbor_k": int(pseudo_edge_fallback_neighbor_k),
                "distance_semantics": {
                    "edge_lengths_km": "pseudo_node_centroid_haversine_km",
                    "distance_matrix_km": "pseudo_node_centroid_haversine_km",
                    "speed_values": "one_sensor_per_directed_pseudo_edge",
                },
                "self_loop_fixes": int(graph["self_loop_fixes"]),
            }
    time_meta = build_time_meta(filtered_df.index)
    speed_values = filtered_df.to_numpy(dtype=np.float32)
    speed_valid_mask = np.isfinite(speed_values) & (speed_values > 0.0)

    np.save(dataset_dir / "speed_values.npy", speed_values)
    np.save(dataset_dir / "speed_valid_mask.npy", speed_valid_mask.astype(np.bool_))
    np.save(dataset_dir / "adjacency_matrix.npy", graph["adjacency"])
    np.save(dataset_dir / "adjacency_weights.npy", graph["adjacency_weights"])
    np.save(dataset_dir / "edge_index.npy", graph["edge_index"])
    np.save(dataset_dir / "edge_lengths_km.npy", graph["edge_lengths_km"])
    np.save(dataset_dir / "distance_matrix_km.npy", graph["distance_matrix_km"])
    time_meta.to_csv(dataset_dir / "time_meta.csv", index=False)
    sensor_frame.to_csv(dataset_dir / "sensor_metadata.csv", index=False)
    if map_snapshot_manifest_payload is not None:
        save_json(map_snapshot_manifest_payload, dataset_dir / "map_snapshot_manifest.json")
    if "sensor_anchor_manifest" in graph:
        graph["sensor_anchor_manifest"].to_csv(dataset_dir / "map_sensor_anchor_manifest.csv", index=False)
    if "pseudo_node_anchor_manifest" in graph:
        graph["pseudo_node_anchor_manifest"].to_csv(dataset_dir / "map_pseudo_node_anchor_manifest.csv", index=False)
    if "carrier_assignment_manifest" in graph:
        graph["carrier_assignment_manifest"].to_csv(dataset_dir / "carrier_assignment_manifest.csv", index=False)
    if "split_node_manifest" in graph:
        graph["split_node_manifest"].to_csv(dataset_dir / "split_node_manifest.csv", index=False)
    if "carrier_boundary_manifest" in graph:
        graph["carrier_boundary_manifest"].to_csv(dataset_dir / "carrier_boundary_manifest.csv", index=False)
    if "carrier_candidate_raw_manifest" in graph:
        graph["carrier_candidate_raw_manifest"].to_csv(dataset_dir / "carrier_candidate_raw_manifest.csv", index=False)
    if "carrier_candidate_scored_manifest" in graph:
        graph["carrier_candidate_scored_manifest"].to_csv(dataset_dir / "carrier_candidate_scored_manifest.csv", index=False)
    if "carrier_assignment_decision_manifest" in graph:
        graph["carrier_assignment_decision_manifest"].to_csv(dataset_dir / "carrier_assignment_decision_manifest.csv", index=False)
    if representation_domain == "pseudo_edge":
        num_pseudo_edges = int(graph["edge_index"].shape[0])
        self_loop_fix_ratio = float(graph["self_loop_fixes"] / max(num_pseudo_edges, 1))
        isolated_edge_ratio = float(
            graph["topology_summary"]["isolated_edges"] / max(num_pseudo_edges, 1)
        )
        cluster_radius_km_value = (
            float(graph["cluster_radius_m"]) / 1000.0
            if pseudo_edge_construction == "map_centered"
            else float(graph["cluster_radius_km"])
            if pseudo_edge_construction == "heuristic"
            else None
        )
        pseudo_edge_summary_payload = {
            "representation_domain": representation_domain,
            "num_sensors": int(filtered_df.shape[1]),
            "num_pseudo_nodes": int(graph["adjacency"].shape[0]),
            "num_pseudo_edges": num_pseudo_edges,
            "cluster_radius_km": cluster_radius_km_value,
            "self_loop_fixes": int(graph["self_loop_fixes"]),
            "self_loop_fix_ratio": self_loop_fix_ratio,
            "topology_summary": graph["topology_summary"],
            "isolated_edge_ratio": isolated_edge_ratio,
            "construction": {
                "sensor_base_graph_mode": sensor_graph_source["mode"],
                "cluster_radius_scale": float(pseudo_edge_cluster_radius_scale),
                "probe_half_length_scale": float(pseudo_edge_half_length_scale),
                "probe_min_half_length_km": float(pseudo_edge_min_half_length_km),
                "probe_max_half_length_km": float(pseudo_edge_max_half_length_km),
                **(
                    {"fallback_neighbor_k": int(pseudo_edge_fallback_neighbor_k)}
                    if pseudo_edge_construction not in {"map_centered", "map_carrier_split"}
                    else {
                        "map_provider": "osm_drive",
                        "routing_policy": str(graph["routing_policy"]),
                        "unreachable_policy": str(graph["unreachable_policy"]),
                        **(
                            {"clipped_half_length_count": int(graph["clipped_half_length_count"])}
                            if pseudo_edge_construction == "map_centered"
                            else {
                                "builder_algorithm": str(graph["builder_algorithm"]),
                                "min_segment_length_m": float(graph["min_segment_length_m"]),
                                "dedupe_position_eps_m": float(graph["dedupe_position_eps_m"]),
                                "unreachable_pair_count": int(graph.get("unreachable_pair_count", 0)),
                                "max_candidate_edges": int(graph["max_candidate_edges"]),
                                "max_snap_distance_m": float(graph["max_snap_distance_m"]),
                                "candidate_search_margin_m": float(graph["candidate_search_margin_m"]),
                                "topology_neighbor_top_k": int(graph["topology_neighbor_top_k"]),
                                "assignment_passes_used": int(graph["assignment_passes_used"]),
                                "assignment_route_cap_m": float(graph["assignment_route_cap_m"]),
                            }
                        ),
                    }
                ),
            },
            "health_warnings": [
                *(
                    [f"self_loop_fix_ratio={self_loop_fix_ratio:.3f} exceeds 0.000"]
                    if pseudo_edge_construction == "map_carrier_split" and self_loop_fix_ratio > 0.0
                    else [f"self_loop_fix_ratio={self_loop_fix_ratio:.3f} exceeds 0.100"]
                    if pseudo_edge_construction != "map_carrier_split" and self_loop_fix_ratio > 0.10
                    else []
                ),
                *(
                    [f"isolated_edge_ratio={isolated_edge_ratio:.3f} exceeds 0.050"]
                    if pseudo_edge_construction == "map_carrier_split" and isolated_edge_ratio > 0.05
                    else [f"isolated_edge_ratio={isolated_edge_ratio:.3f} exceeds 0.150"]
                    if pseudo_edge_construction != "map_carrier_split" and isolated_edge_ratio > 0.15
                    else []
                ),
                *(
                    [
                        "line_graph_largest_component_ratio="
                        f"{float(graph['topology_summary']['line_graph_largest_component_ratio']):.3f} is below 0.800"
                    ]
                    if pseudo_edge_construction == "map_carrier_split"
                    and float(graph["topology_summary"]["line_graph_largest_component_ratio"]) < 0.80
                    else [
                        "line_graph_largest_component_ratio="
                        f"{float(graph['topology_summary']['line_graph_largest_component_ratio']):.3f} is below 0.250"
                    ]
                    if pseudo_edge_construction != "map_carrier_split"
                    and float(graph["topology_summary"]["line_graph_largest_component_ratio"]) < 0.25
                    else []
                ),
            ],
        }
        graph["pseudo_edge_manifest"].to_csv(dataset_dir / "pseudo_edge_manifest.csv", index=False)
        graph["pseudo_node_metadata"].to_csv(dataset_dir / "pseudo_node_metadata.csv", index=False)
        save_json(pseudo_edge_summary_payload, dataset_dir / "pseudo_edge_summary.json")
    build_sensor_map(
        sensor_frame,
        dataset_label=spec["display_name"],
        output_path=dataset_dir / "sensor_map.html",
    )

    summary = {
        "dataset_name": spec["display_name"],
        "official_dcrnn_folder": OFFICIAL_DCRNN_FOLDER_URL,
        "traffic_file": str(traffic_path),
        "location_file": str(location_path),
        "requested_start": spec["requested_start"],
        "requested_end": spec["requested_end"],
        "actual_start": actual_start.isoformat() if actual_start is not None else None,
        "actual_end": actual_end.isoformat() if actual_end is not None else None,
        "representation_domain": representation_domain,
        "num_time_steps": int(filtered_df.shape[0]),
        "num_sensors": int(filtered_df.shape[1]),
        "num_graph_nodes": int(graph["adjacency"].shape[0]),
        "num_graph_edges": int(graph["edge_index"].shape[0]),
        "graph_source": graph_source,
        "graph_k": int(graph["effective_k"]) if "effective_k" in graph else None,
        "graph_symmetrized": bool(symmetrize) if graph_mode == "haversine_knn" else None,
        "speed_missing_ratio": float(1.0 - speed_valid_mask.mean()),
        "slot_minutes": int(
            pd.to_numeric(time_meta["slot"], errors="raise").max() + 1
        )
        if len(time_meta) == 1
        else int(
            round(
                (
                    pd.to_datetime(time_meta["timestamp"]).iloc[1]
                    - pd.to_datetime(time_meta["timestamp"]).iloc[0]
                ).total_seconds()
                / 60.0
            )
        ),
        "notes": (
            "METR-LA official DCRNN h5 currently ends on 2012-06-27 23:55, "
            "so requested dates beyond that are clipped automatically. "
            "Speed zeros are stored with an explicit validity mask so masked benchmark metrics can exclude missing readings."
            if dataset_name == "metr-la"
            else "PEMS-BAY uses the full official raw range through 2017-06-30 when available. "
            "Official adjacency is loaded from the public adj_mx_bay.pkl release when graph_mode=official, "
            "and the official sensor-node path does not attach any coordinate-derived edge-length feature."
        ),
    }
    if map_snapshot_manifest_payload is not None:
        summary["map_snapshot_manifest_file"] = str(dataset_dir / "map_snapshot_manifest.json")
    if "sensor_anchor_manifest" in graph:
        summary["map_sensor_anchor_manifest_file"] = str(dataset_dir / "map_sensor_anchor_manifest.csv")
    if "pseudo_node_anchor_manifest" in graph:
        summary["map_pseudo_node_anchor_manifest_file"] = str(dataset_dir / "map_pseudo_node_anchor_manifest.csv")
    if "carrier_assignment_manifest" in graph:
        summary["carrier_assignment_manifest_file"] = str(dataset_dir / "carrier_assignment_manifest.csv")
    if "split_node_manifest" in graph:
        summary["split_node_manifest_file"] = str(dataset_dir / "split_node_manifest.csv")
    if "carrier_boundary_manifest" in graph:
        summary["carrier_boundary_manifest_file"] = str(dataset_dir / "carrier_boundary_manifest.csv")
    if "carrier_candidate_raw_manifest" in graph:
        summary["carrier_candidate_raw_manifest_file"] = str(dataset_dir / "carrier_candidate_raw_manifest.csv")
    if "carrier_candidate_scored_manifest" in graph:
        summary["carrier_candidate_scored_manifest_file"] = str(dataset_dir / "carrier_candidate_scored_manifest.csv")
    if "carrier_assignment_decision_manifest" in graph:
        summary["carrier_assignment_decision_manifest_file"] = str(dataset_dir / "carrier_assignment_decision_manifest.csv")
    if representation_domain == "pseudo_edge":
        summary["num_pseudo_nodes"] = int(graph["adjacency"].shape[0])
        summary["num_pseudo_edges"] = int(graph["edge_index"].shape[0])
        summary["pseudo_edge_summary_file"] = str(dataset_dir / "pseudo_edge_summary.json")
        summary["pseudo_edge_construction"] = pseudo_edge_summary_payload["construction"]
        if pseudo_edge_construction == "map_centered":
            summary["representation_variant_id"] = (
                "pseudo_edge__map_centered"
                f"__base-{sensor_graph_source['mode']}"
                f"__radius-{float(graph['cluster_radius_m']) / 1000.0:.3f}km"
                f"__halfscale-{float(pseudo_edge_half_length_scale):.3f}"
                f"__clusterscale-{float(pseudo_edge_cluster_radius_scale):.3f}"
            )
            summary["notes"] = (
                summary["notes"]
                + " Experimental map-centered pseudo-edge representation: each sensor is snapped to an external drive graph, centered on one synthetic directed road edge, and trained as one supervised pseudo-edge; results are not directly comparable to official sensor-node benchmarks."
            )
        elif pseudo_edge_construction == "map_carrier_split":
            summary["representation_variant_id"] = (
                f"pseudo_edge__map_carrier_split__base-{sensor_graph_source['mode']}"
                f"__builder-{graph['builder_algorithm']}"
                f"__minseg-{float(graph['min_segment_length_m']):.1f}m"
                f"__dedupe-{float(graph['dedupe_position_eps_m']):.1f}m"
            )
            summary["notes"] = (
                summary["notes"]
                + " Experimental map carrier/split-edge representation: each sensor is assigned to one directed road carrier edge on an external map and supervises one real split subsegment; fixed pseudo-edge topology is induced by shared split/junction endpoints and is not directly comparable to official sensor-node benchmarks."
            )
        else:
            summary["representation_variant_id"] = (
                f"pseudo_edge__base-{sensor_graph_source['mode']}"
                f"__radius-{float(graph['cluster_radius_km']):.3f}km"
                f"__halfscale-{float(pseudo_edge_half_length_scale):.3f}"
                f"__clusterscale-{float(pseudo_edge_cluster_radius_scale):.3f}"
                f"__fallbackk-{int(pseudo_edge_fallback_neighbor_k)}"
            )
            summary["notes"] = (
                summary["notes"]
                + " Experimental pseudo-edge representation: each sensor is reinterpreted as one directed pseudo-edge built from the sensor graph; results are not directly comparable to official sensor-node benchmarks."
            )
    else:
        if sensor_node_distance_source == "map_drive":
            summary["representation_variant_id"] = (
                f"sensor_node__map_route__base-{sensor_graph_source['mode']}"
            )
            summary["notes"] = (
                summary["notes"]
                + " Experimental sensor-node map-route variant: topology and weights follow the prepared base graph, but edge lengths come from external-map directed drivable routes between snapped sensor points."
            )
        elif sensor_node_distance_source == "haversine":
            summary["representation_variant_id"] = (
                f"sensor_node__haversine__base-{sensor_graph_source['mode']}"
            )
            summary["notes"] = (
                summary["notes"]
                + " Experimental sensor-node haversine variant: topology and weights follow the prepared base graph, but edge lengths use straight-line geodesic distances between sensor coordinates."
            )
        else:
            summary["representation_variant_id"] = "sensor_node"
    save_json(summary, dataset_dir / "dataset_summary.json")
    print(
        f"[{spec['display_name']}] prepared {filtered_df.shape[0]} time steps, "
        f"{filtered_df.shape[1]} sensors, {graph['edge_index'].shape[0]} graph edges "
        f"({representation_domain}) -> {dataset_dir}"
    )
    return dataset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare METR-LA / PEMS-BAY for local STGAT sensor-speed training.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="metr-la,pems-bay",
        help="Comma-separated dataset keys from {metr-la,pems-bay}.",
    )
    parser.add_argument(
        "--raw-root",
        type=str,
        default="data/external_datasets/raw",
        help="Where to store the downloaded official DCRNN assets.",
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default="data/external_datasets/processed",
        help="Where to write the processed arrays, map, and metadata.",
    )
    parser.add_argument(
        "--graph-mode",
        type=str,
        default="official",
        choices=["official", "haversine_knn"],
        help="Graph source used for adjacency construction.",
    )
    parser.add_argument(
        "--graph-k",
        type=int,
        default=8,
        help="Number of nearest geographic neighbors kept per sensor before optional symmetrization.",
    )
    parser.add_argument(
        "--official-normalized-k",
        type=float,
        default=0.1,
        help="Threshold used by the official DCRNN Gaussian-kernel graph construction for METR-LA.",
    )
    parser.add_argument(
        "--directed-only",
        action="store_true",
        help="Do not symmetrize the kNN graph.",
    )
    parser.add_argument(
        "--representation-domain",
        type=str,
        default="sensor_node",
        choices=["sensor_node", "pseudo_edge"],
        help="Keep the official sensor-node representation or reinterpret each sensor as one directed pseudo-edge.",
    )
    parser.add_argument(
        "--pseudo-edge-cluster-radius-km",
        type=float,
        default=0.0,
        help="Optional fixed endpoint-clustering radius for pseudo-edge mode; <= 0 enables automatic sizing.",
    )
    parser.add_argument(
        "--pseudo-edge-cluster-radius-scale",
        type=float,
        default=1.25,
        help="Automatic cluster-radius scale applied to the median pseudo-edge half length.",
    )
    parser.add_argument(
        "--pseudo-edge-half-length-scale",
        type=float,
        default=0.45,
        help="Scale applied to each sensor's local neighbor distance when sizing pseudo-edge half lengths.",
    )
    parser.add_argument(
        "--pseudo-edge-min-half-length-km",
        type=float,
        default=0.05,
        help="Lower bound for pseudo-edge half lengths in km.",
    )
    parser.add_argument(
        "--pseudo-edge-max-half-length-km",
        type=float,
        default=0.5,
        help="Upper bound for pseudo-edge half lengths in km.",
    )
    parser.add_argument(
        "--pseudo-edge-fallback-neighbor-k",
        type=int,
        default=3,
        help="How many nearest sensors to use when a sensor graph direction vector is weak or missing.",
    )
    parser.add_argument(
        "--sensor-node-distance-source",
        type=str,
        default="prepared",
        choices=["prepared", "haversine", "map_drive"],
        help="Keep the prepared edge-length semantics, replace them with straight-line haversine lengths, or replace them with external-map drivable route lengths for sensor-node mode.",
    )
    parser.add_argument(
        "--pseudo-edge-construction",
        type=str,
        default="heuristic",
        choices=["heuristic", "map_centered", "map_carrier_split"],
        help="Use the legacy heuristic pseudo-edge builder, a map-backed centered pseudo-edge builder, or a map-backed carrier/split-edge builder.",
    )
    parser.add_argument(
        "--map-cache-root",
        type=str,
        default="data/external_datasets/map_cache",
        help="Where to cache downloaded OSM drive-graph snapshots and snapped anchor artifacts.",
    )
    parser.add_argument(
        "--map-graph-padding-km",
        type=float,
        default=DEFAULT_OSM_DRIVE_PADDING_KM,
        help="Extra padding around the sensor bounding box when downloading the external drive graph.",
    )
    parser.add_argument(
        "--map-unreachable-policy",
        type=str,
        default="raise",
        choices=["raise", "haversine_fallback"],
        help="How to handle retained graph edges whose external-map route distance is unreachable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    dataset_names = [chunk.strip().lower() for chunk in args.datasets.split(",") if chunk.strip()]
    for dataset_name in dataset_names:
        if dataset_name not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        prepare_dataset(
            dataset_name,
            raw_root=raw_root,
            processed_root=processed_root,
            graph_mode=args.graph_mode,
            graph_k=args.graph_k,
            symmetrize=not args.directed_only,
            official_normalized_k=args.official_normalized_k,
            representation_domain=args.representation_domain,
            pseudo_edge_cluster_radius_km=(
                None if float(args.pseudo_edge_cluster_radius_km) <= 0 else float(args.pseudo_edge_cluster_radius_km)
            ),
            pseudo_edge_cluster_radius_scale=float(args.pseudo_edge_cluster_radius_scale),
            pseudo_edge_half_length_scale=float(args.pseudo_edge_half_length_scale),
            pseudo_edge_min_half_length_km=float(args.pseudo_edge_min_half_length_km),
            pseudo_edge_max_half_length_km=float(args.pseudo_edge_max_half_length_km),
            pseudo_edge_fallback_neighbor_k=int(args.pseudo_edge_fallback_neighbor_k),
            sensor_node_distance_source=str(args.sensor_node_distance_source),
            pseudo_edge_construction=str(args.pseudo_edge_construction),
            map_cache_root=Path(args.map_cache_root),
            map_graph_padding_km=float(args.map_graph_padding_km),
            map_unreachable_policy=str(args.map_unreachable_policy),
        )


if __name__ == "__main__":
    main()
