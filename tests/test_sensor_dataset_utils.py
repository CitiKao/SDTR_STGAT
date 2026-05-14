import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_speed_benchmarks.sensor_dataset_utils import (  # noqa: E402
    OUTLIER_CLEANING_MODES,
    apply_valid_speed_clip,
    build_cyclical_time_features,
    build_directed_knn_graph,
    build_graph_from_weighted_adjacency,
    build_official_distance_graph,
    build_processed_dataset_dir_name,
    build_pseudo_edge_graph,
    build_time_meta,
    compute_train_quantile_clip_bounds,
    normalize_representation_domain,
    validate_outlier_cleaning_mode,
)


class SensorDatasetUtilsTest(unittest.TestCase):
    def test_build_time_meta_and_cyclical_features(self) -> None:
        timestamps = [
            "2020-01-01 00:00:00",
            "2020-01-01 00:05:00",
            "2020-01-01 00:10:00",
        ]
        time_meta = build_time_meta(timestamps)
        self.assertEqual(time_meta["slot"].tolist(), [0, 1, 2])
        features, names = build_cyclical_time_features(time_meta)
        self.assertEqual(features.shape, (3, 6))
        self.assertEqual(names, ["month_sin", "month_cos", "weekday_sin", "weekday_cos", "slot_sin", "slot_cos"])

    def test_build_directed_knn_graph_shapes(self) -> None:
        coords = np.array(
            [
                [34.0, -118.0],
                [34.1, -118.0],
                [34.2, -118.0],
            ],
            dtype=np.float64,
        )
        graph = build_directed_knn_graph(coords, k=1, symmetrize=True)
        self.assertEqual(graph["adjacency"].shape, (3, 3))
        self.assertEqual(graph["edge_index"].shape[1], 2)
        self.assertEqual(graph["edge_lengths_km"].shape[0], graph["edge_index"].shape[0])
        self.assertTrue(np.all(np.diag(graph["adjacency"]) == 0.0))

    def test_representation_domain_helpers(self) -> None:
        self.assertEqual(normalize_representation_domain("sensor_node"), "sensor_node")
        self.assertEqual(normalize_representation_domain("pseudo_edge"), "pseudo_edge")
        self.assertEqual(build_processed_dataset_dir_name("METR-LA", "sensor_node"), "METR-LA")
        self.assertEqual(build_processed_dataset_dir_name("METR-LA", "pseudo_edge"), "METR-LA_pseudo_edge")
        with self.assertRaises(ValueError):
            normalize_representation_domain("bad_domain")

    def test_build_official_distance_graph_matches_thresholded_edges(self) -> None:
        coords = np.array(
            [
                [34.0, -118.0],
                [34.1, -118.0],
                [34.2, -118.0],
            ],
            dtype=np.float64,
        )
        distance_frame = pd.DataFrame(
            {
                "from": ["A", "A", "B", "B", "C", "C"],
                "to": ["B", "C", "A", "C", "A", "B"],
                "distance": [1000.0, 10000.0, 1000.0, 1200.0, 10000.0, 1200.0],
            }
        )
        graph = build_official_distance_graph(
            distance_frame=distance_frame,
            sensor_ids=["A", "B", "C"],
            coords=coords,
        )
        self.assertEqual(graph["adjacency"].shape, (3, 3))
        self.assertEqual(graph["edge_index"].shape[1], 2)
        self.assertTrue(np.all(np.diag(graph["adjacency"]) == 0.0))
        self.assertGreaterEqual(graph["edge_index"].shape[0], 4)
        expected_edge_lengths = graph["distance_matrix_km"][
            graph["edge_index"][:, 0],
            graph["edge_index"][:, 1],
        ]
        self.assertTrue(np.allclose(graph["edge_lengths_km"], expected_edge_lengths))
        self.assertEqual(graph["distance_semantics"]["edge_lengths_km"], "official_road_distance_km")

    def test_build_graph_from_weighted_adjacency_can_disable_custom_edge_lengths(self) -> None:
        coords = np.array(
            [
                [37.0, -122.0],
                [37.1, -122.1],
                [37.2, -122.2],
            ],
            dtype=np.float64,
        )
        adjacency_weights = np.array(
            [
                [0.0, 0.8, 0.0],
                [0.3, 0.0, 0.5],
                [0.0, 0.6, 0.0],
            ],
            dtype=np.float32,
        )
        graph = build_graph_from_weighted_adjacency(
            adjacency_weights,
            coords=coords,
            edge_length_strategy="none",
        )
        self.assertEqual(graph["edge_index"].shape[1], 2)
        self.assertTrue(np.allclose(graph["edge_lengths_km"], 0.0))
        self.assertTrue(np.allclose(graph["distance_matrix_km"], 0.0))
        self.assertEqual(
            graph["distance_semantics"]["edge_lengths_km"],
            "disabled_zero_no_custom_length_feature",
        )

    def test_build_pseudo_edge_graph_builds_one_edge_per_sensor(self) -> None:
        coords = np.array(
            [
                [34.0000, -118.0000],
                [34.0010, -118.0000],
                [34.0020, -118.0000],
            ],
            dtype=np.float64,
        )
        adjacency_weights = np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        graph = build_pseudo_edge_graph(
            coords,
            sensor_ids=["A", "B", "C"],
            adjacency_weights=adjacency_weights,
            cluster_radius_km=0.05,
            probe_half_length_scale=0.2,
            min_half_length_km=0.05,
            max_half_length_km=0.2,
            fallback_neighbor_k=2,
        )
        self.assertEqual(graph["edge_index"].shape, (3, 2))
        self.assertEqual(graph["edge_lengths_km"].shape[0], 3)
        self.assertEqual(len(graph["pseudo_edge_manifest"]), 3)
        self.assertGreaterEqual(graph["adjacency"].shape[0], 2)
        self.assertTrue(np.all(graph["edge_index"][:, 0] != graph["edge_index"][:, 1]))
        self.assertEqual(
            graph["pseudo_edge_manifest"]["sensor_id"].tolist(),
            ["A", "B", "C"],
        )
        self.assertIn("num_unique_structural_edges", graph["topology_summary"])
        self.assertIn("line_graph_weak_components", graph["topology_summary"])
        self.assertIn("line_graph_largest_component_ratio", graph["topology_summary"])
        self.assertGreaterEqual(graph["topology_summary"]["num_unique_structural_edges"], 1)

    def test_validate_outlier_cleaning_mode(self) -> None:
        self.assertEqual(validate_outlier_cleaning_mode("none"), "none")
        self.assertEqual(validate_outlier_cleaning_mode("train_quantile_clip"), "train_quantile_clip")
        self.assertIn("train_quantile_clip", OUTLIER_CLEANING_MODES)
        with self.assertRaises(ValueError):
            validate_outlier_cleaning_mode("bad_mode")

    def test_train_quantile_clip_only_changes_valid_extreme_points(self) -> None:
        speed_values = np.array(
            [
                [10.0],
                [11.0],
                [12.0],
                [300.0],
                [13.0],
            ],
            dtype=np.float32,
        )
        valid_mask = np.array([[True], [True], [True], [True], [True]], dtype=bool)
        train_time_mask = np.array([True, True, True, False, False], dtype=bool)
        bounds = compute_train_quantile_clip_bounds(
            speed_values,
            valid_mask,
            train_time_mask=train_time_mask,
            lower_quantile=0.0 + 1e-6,
            upper_quantile=0.999999,
        )
        clipped = apply_valid_speed_clip(
            speed_values,
            valid_mask,
            lower_bounds=bounds["lower"],
            upper_bounds=bounds["upper"],
        )
        self.assertAlmostEqual(float(clipped["cleaned_speed_values"][3, 0]), float(bounds["upper"][0]), places=5)
        self.assertTrue(bool(clipped["outlier_mask"][3, 0]))
        self.assertGreaterEqual(int(clipped["outlier_mask"].sum()), 1)
        self.assertAlmostEqual(float(clipped["cleaned_speed_values"][0, 0]), 10.0, places=5)

    def test_train_quantile_clip_ignores_invalid_zeros(self) -> None:
        speed_values = np.array(
            [
                [10.0],
                [0.0],
                [12.0],
                [200.0],
                [13.0],
            ],
            dtype=np.float32,
        )
        valid_mask = np.array([[True], [False], [True], [True], [True]], dtype=bool)
        train_time_mask = np.array([True, True, True, False, False], dtype=bool)
        bounds = compute_train_quantile_clip_bounds(
            speed_values,
            valid_mask,
            train_time_mask=train_time_mask,
            lower_quantile=0.01,
            upper_quantile=0.99,
        )
        clipped = apply_valid_speed_clip(
            speed_values,
            valid_mask,
            lower_bounds=bounds["lower"],
            upper_bounds=bounds["upper"],
        )
        self.assertFalse(bool(clipped["outlier_mask"][1, 0]))
        self.assertTrue(bool(clipped["outlier_mask"][3, 0]))
        self.assertGreaterEqual(int(clipped["summary"]["num_flagged"]), 1)

    def test_train_quantile_clip_skips_sensors_without_train_observations(self) -> None:
        speed_values = np.array(
            [
                [0.0],
                [0.0],
                [100.0],
                [110.0],
            ],
            dtype=np.float32,
        )
        valid_mask = np.array([[False], [False], [True], [True]], dtype=bool)
        train_time_mask = np.array([True, True, False, False], dtype=bool)
        bounds = compute_train_quantile_clip_bounds(
            speed_values,
            valid_mask,
            train_time_mask=train_time_mask,
            lower_quantile=0.01,
            upper_quantile=0.99,
        )
        clipped = apply_valid_speed_clip(
            speed_values,
            valid_mask,
            lower_bounds=bounds["lower"],
            upper_bounds=bounds["upper"],
        )
        self.assertTrue(np.isnan(bounds["lower"][0]))
        self.assertTrue(np.isnan(bounds["upper"][0]))
        self.assertAlmostEqual(float(clipped["cleaned_speed_values"][2, 0]), 100.0, places=5)
        self.assertAlmostEqual(float(clipped["cleaned_speed_values"][3, 0]), 110.0, places=5)
        self.assertFalse(bool(clipped["outlier_mask"].any()))


if __name__ == "__main__":
    unittest.main()
