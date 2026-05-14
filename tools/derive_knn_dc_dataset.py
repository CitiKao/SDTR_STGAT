#!/usr/bin/env python
"""Derive a KNN-graph variant from an existing grid-based DC dataset.

This avoids re-reading raw GPS archives when only the spatial graph K changes.
Node D/C targets and time metadata are copied; the graph is rebuilt from
``zone_info.csv`` centers. Edge speeds are reprojected by first estimating
per-node speeds from the source edge-speed graph, then averaging the new edge
endpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path

import numpy as np

from build_shenzhen_dc_dataset import (
    atomic_save_npy,
    atomic_write_text,
    build_knn_graph,
    export_benchmark,
    validate_outputs,
)


def load_centers(zone_info_path: Path) -> np.ndarray:
    rows: list[tuple[int, float, float]] = []
    with zone_info_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append((int(row["index"]), float(row["lon_center"]), float(row["lat_center"])))
    rows.sort(key=lambda item: item[0])
    if not rows:
        raise ValueError(f"No zones found in {zone_info_path}")
    expected = list(range(len(rows)))
    actual = [idx for idx, _, _ in rows]
    if actual != expected:
        raise ValueError(f"zone_info indices must be contiguous 0..N-1, got first indices {actual[:10]}")
    return np.asarray([[lon, lat] for _, lon, lat in rows], dtype=np.float64)


def estimate_node_speeds(source_edge_index: np.ndarray, source_edge_speeds: np.ndarray, num_nodes: int) -> np.ndarray:
    """Least-squares estimate of node speeds from edge endpoint averages."""
    num_edges = int(source_edge_index.shape[0])
    design = np.zeros((num_edges, num_nodes), dtype=np.float32)
    design[np.arange(num_edges), source_edge_index[:, 0].astype(np.int64)] = 1.0
    design[np.arange(num_edges), source_edge_index[:, 1].astype(np.int64)] += 1.0
    print(f"[derive-knn] solving node speeds from source graph E={num_edges} N={num_nodes}", flush=True)
    pseudo_inverse = np.linalg.pinv(design, rcond=1e-4).astype(np.float32)
    node_speeds = (2.0 * source_edge_speeds.astype(np.float32)) @ pseudo_inverse.T
    return np.clip(node_speeds, 1.0, 160.0).astype(np.float32)


def build_edge_speeds(node_speeds: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    src = edge_index[:, 0].astype(np.int64)
    dst = edge_index[:, 1].astype(np.int64)
    return np.clip(0.5 * (node_speeds[:, src] + node_speeds[:, dst]), 1.0, 160.0).astype(np.float32)


def copy_static_files(source_dir: Path, output_dir: Path) -> None:
    for name in (
        "node_demand.npy",
        "node_supply.npy",
        "dropoff_events.npy",
        "targets_dc.npy",
        "time_features.npy",
        "observed_time_mask.npy",
        "time_meta.csv",
        "zone_info.csv",
    ):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive a KNN graph variant for a DC dataset.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-output-dir", type=Path, required=True)
    parser.add_argument("--knn", type=int, required=True)
    parser.add_argument("--hist-len", type=int, default=14)
    parser.add_argument("--pred-horizon", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.knn <= 0:
        raise SystemExit("--knn must be positive")
    source_dir = args.source_dir
    output_dir = args.output_dir
    benchmark_dir = args.benchmark_output_dir
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    centers = load_centers(source_dir / "zone_info.csv")
    adj, edge_index, edge_lengths = build_knn_graph(centers, args.knn)
    source_edge_index = np.load(source_dir / "edge_index.npy").astype(np.int32)
    source_edge_speeds = np.load(source_dir / "edge_speeds.npy").astype(np.float32)
    node_speeds = estimate_node_speeds(source_edge_index, source_edge_speeds, centers.shape[0])
    edge_speeds = build_edge_speeds(node_speeds, edge_index)

    copy_static_files(source_dir, output_dir)
    atomic_save_npy(output_dir / "adjacency_matrix.npy", adj)
    atomic_save_npy(output_dir / "edge_index.npy", edge_index)
    atomic_save_npy(output_dir / "edge_lengths.npy", edge_lengths)
    atomic_save_npy(output_dir / "edge_lengths_osrm.npy", edge_lengths)
    atomic_save_npy(output_dir / "edge_lengths_km.npy", edge_lengths)
    atomic_save_npy(output_dir / "edge_speeds.npy", edge_speeds)

    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8")) if source_manifest_path.exists() else {}
    validation = validate_outputs(output_dir)
    manifest = dict(source_manifest)
    manifest.update(
        {
            "schema_version": f"{source_manifest.get('schema_version', 'dc_dataset')}_knn_variant_v1",
            "dataset_name": f"{source_manifest.get('dataset_name', 'dc_dataset')}_knn{args.knn}",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "derived_from": {
                "source_dir": str(source_dir),
                "source_dataset_name": source_manifest.get("dataset_name"),
                "operation": "rebuild directed spatial KNN graph and reproject edge speeds from source edge speeds",
            },
            "graph": {
                "mode": "directed_spatial_knn",
                "knn": int(args.knn),
                "nodes": int(centers.shape[0]),
                "edges": int(edge_index.shape[0]),
                "edge_length_unit": "km",
                "edge_speed_projection": "least_squares_node_speed_from_source_edge_averages",
            },
            "arrays": {
                **dict(source_manifest.get("arrays", {})),
                "adjacency_matrix": "adjacency_matrix.npy",
                "edge_index": "edge_index.npy",
                "edge_lengths": "edge_lengths.npy",
                "edge_lengths_osrm": "edge_lengths_osrm.npy",
                "edge_lengths_km": "edge_lengths_km.npy",
                "edge_speeds": "edge_speeds.npy",
            },
            "shapes": {
                **dict(source_manifest.get("shapes", {})),
                "adjacency_matrix": list(adj.shape),
                "edge_index": list(edge_index.shape),
                "edge_speeds": list(edge_speeds.shape),
            },
            "validation": validation,
        }
    )
    atomic_write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        f"[derive-knn] output={output_dir} T={validation['T']} N={validation['N']} E={validation['E']} "
        f"D={validation['demand_sum']:.0f} C={validation['supply_sum']:.0f}",
        flush=True,
    )
    export_benchmark(output_dir, benchmark_dir, args.hist_len, args.pred_horizon)
    print(f"[derive-knn] benchmark={benchmark_dir}", flush=True)


if __name__ == "__main__":
    main()
