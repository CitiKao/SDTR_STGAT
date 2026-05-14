import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from shapely.geometry import LineString


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_speed_benchmarks.map_graph_utils import (  # noqa: E402
    build_map_carrier_split_pseudo_edge_graph,
    build_map_centered_pseudo_edge_graph,
    build_sensor_node_map_distance_graph,
)


def _make_linear_drive_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = CRS.from_epsg(3857)
    graph.add_node(0, x=0.0, y=0.0)
    graph.add_node(1, x=100.0, y=0.0)
    graph.add_node(2, x=200.0, y=0.0)
    edge_specs = [
        (0, 1, 0, [(0.0, 0.0), (100.0, 0.0)]),
        (1, 0, 0, [(100.0, 0.0), (0.0, 0.0)]),
        (1, 2, 0, [(100.0, 0.0), (200.0, 0.0)]),
        (2, 1, 0, [(200.0, 0.0), (100.0, 0.0)]),
    ]
    for u, v, key, coords in edge_specs:
        geometry = LineString(coords)
        graph.add_edge(
            u,
            v,
            key=key,
            geometry=geometry,
            length=float(geometry.length),
        )
    return graph


def _sensor_frame_from_projected_points(xs_m: list[float]) -> pd.DataFrame:
    to_geo = Transformer.from_crs(CRS.from_epsg(3857), "EPSG:4326", always_xy=True)
    longitudes, latitudes = to_geo.transform(xs_m, [0.0] * len(xs_m))
    return pd.DataFrame(
        {
            "sensor_id": [f"s{idx}" for idx in range(len(xs_m))],
            "latitude": np.asarray(latitudes, dtype=np.float32),
            "longitude": np.asarray(longitudes, dtype=np.float32),
        }
    )


class MapGraphUtilsTest(unittest.TestCase):
    def test_build_sensor_node_map_distance_graph_uses_routed_lengths(self) -> None:
        graph = _make_linear_drive_graph()
        sensor_frame = _sensor_frame_from_projected_points([25.0, 125.0])
        topology_weights = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_context = {
                "graph": graph,
                "snapshot_dir": Path(tmpdir),
                "manifest": {"provider": "unit_test_graph"},
            }
            graph_payload = build_sensor_node_map_distance_graph(
                sensor_frame,
                topology_adjacency_weights=topology_weights,
                snapshot_context=snapshot_context,
                unreachable_policy="raise",
            )

        self.assertEqual(tuple(graph_payload["edge_index"].shape), (2, 2))
        self.assertTrue(np.allclose(graph_payload["edge_lengths_km"], np.array([0.1, 0.1], dtype=np.float32)))
        self.assertEqual(len(graph_payload["sensor_anchor_manifest"]), 2)
        self.assertEqual(
            graph_payload["distance_semantics"]["edge_lengths_km"],
            "external_map_directed_route_distance_km_between_snapped_sensor_points",
        )

    def test_build_map_centered_pseudo_edge_graph_returns_finite_map_lengths(self) -> None:
        graph = _make_linear_drive_graph()
        sensor_frame = _sensor_frame_from_projected_points([25.0, 125.0, 175.0])
        sensor_ids = sensor_frame["sensor_id"].tolist()
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_context = {
                "graph": graph,
                "snapshot_dir": Path(tmpdir),
                "manifest": {"provider": "unit_test_graph"},
            }
            graph_payload = build_map_centered_pseudo_edge_graph(
                sensor_frame,
                sensor_ids=sensor_ids,
                snapshot_context=snapshot_context,
                cluster_radius_m=20.0,
                cluster_radius_scale=1.25,
                half_length_scale=0.4,
                min_half_length_m=5.0,
                max_half_length_m=20.0,
                unreachable_policy="raise",
            )

        self.assertEqual(graph_payload["edge_index"].shape[0], 3)
        self.assertEqual(len(graph_payload["pseudo_edge_manifest"]), 3)
        self.assertTrue(np.isfinite(graph_payload["edge_lengths_km"]).all())
        self.assertTrue((graph_payload["edge_lengths_km"] > 0).all())
        self.assertEqual(
            graph_payload["distance_semantics"]["speed_values"],
            "one_sensor_per_directed_map_centered_pseudo_edge",
        )

    def test_build_map_carrier_split_pseudo_edge_graph_splits_shared_carrier(self) -> None:
        graph = _make_linear_drive_graph()
        sensor_frame = _sensor_frame_from_projected_points([25.0, 75.0, 125.0])
        sensor_ids = sensor_frame["sensor_id"].tolist()
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_context = {
                "graph": graph,
                "snapshot_dir": Path(tmpdir),
                "manifest": {"provider": "unit_test_graph"},
            }
            graph_payload = build_map_carrier_split_pseudo_edge_graph(
                sensor_frame,
                sensor_ids=sensor_ids,
                snapshot_context=snapshot_context,
                min_segment_length_m=5.0,
                dedupe_position_eps_m=1.0,
                unreachable_policy="raise",
            )

        self.assertEqual(graph_payload["edge_index"].shape[0], 3)
        self.assertEqual(len(graph_payload["pseudo_edge_manifest"]), 3)
        self.assertEqual(int(graph_payload["self_loop_fixes"]), 0)
        self.assertEqual(int(graph_payload["topology_summary"]["duplicate_structural_edges"]), 0)
        self.assertTrue(np.isfinite(graph_payload["edge_lengths_km"]).all())
        self.assertTrue((graph_payload["edge_lengths_km"] > 0).all())
        self.assertEqual(
            graph_payload["distance_semantics"]["edge_lengths_km"],
            "road_split_subsegment_length_km",
        )
        self.assertEqual(
            graph_payload["distance_semantics"]["adjacency_weights"],
            "binary_split_graph_connectivity",
        )
        self.assertIn("carrier_assignment_manifest", graph_payload)
        self.assertIn("split_node_manifest", graph_payload)


if __name__ == "__main__":
    unittest.main()
