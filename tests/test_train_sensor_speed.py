import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_speed_benchmarks.train_sensor_speed import (  # noqa: E402
    align_split_boundary_index,
    attach_speed_report_metrics_to_record,
    build_project_monthly_split_summary,
    build_training_control_summary,
    build_time_contained_split_indices,
    build_resume_safe_run_config,
    collect_speed_edge_trainable_parameters,
    compute_processed_dataset_fingerprint,
    is_early_stopping_enabled,
    load_prepared_sensor_dataset,
    load_checkpoint,
    resolve_outlier_cleaning,
    resolve_fixed_edge_length_feature_mode,
    resolve_split_boundaries,
    save_checkpoint,
    should_trigger_early_stop,
)
from data_loader import SpatioTemporalDataset  # noqa: E402
from stgat_model import STGATPredictor  # noqa: E402


class TrainSensorSpeedTest(unittest.TestCase):
    def test_early_stopping_is_disabled_when_patience_is_non_positive(self) -> None:
        self.assertFalse(is_early_stopping_enabled(0))
        self.assertFalse(is_early_stopping_enabled(-1))
        self.assertTrue(is_early_stopping_enabled(1))

    def test_build_training_control_summary_marks_full_budget_runs(self) -> None:
        summary = build_training_control_summary(
            configured_epochs=180,
            history=[
                {"epoch": 1, "val_raw_speed_rmse": 1.0},
                {"epoch": 180, "val_raw_speed_rmse": 0.9},
            ],
            early_stop_patience=0,
            min_epochs=10,
            training_end_reason="epoch_budget_exhausted",
        )

        self.assertEqual(summary["completed_epochs"], 180)
        self.assertEqual(summary["configured_epochs"], 180)
        self.assertTrue(summary["completed_full_budget"])
        self.assertEqual(summary["training_end_reason"], "epoch_budget_exhausted")
        self.assertFalse(summary["early_stopping"]["enabled"])
        self.assertFalse(summary["early_stopping"]["triggered"])

    def test_build_training_control_summary_marks_early_stopped_runs(self) -> None:
        summary = build_training_control_summary(
            configured_epochs=180,
            history=[
                {"epoch": 1, "val_raw_speed_rmse": 1.0},
                {"epoch": 27, "val_raw_speed_rmse": 1.1},
            ],
            early_stop_patience=8,
            min_epochs=10,
            training_end_reason="early_stop_patience",
        )

        self.assertEqual(summary["completed_epochs"], 27)
        self.assertEqual(summary["configured_epochs"], 180)
        self.assertFalse(summary["completed_full_budget"])
        self.assertEqual(summary["training_end_reason"], "early_stop_patience")
        self.assertTrue(summary["early_stopping"]["enabled"])
        self.assertTrue(summary["early_stopping"]["triggered"])

    def test_should_trigger_early_stop_respects_all_conditions(self) -> None:
        self.assertFalse(
            should_trigger_early_stop(
                should_validate=False,
                epoch=20,
                early_stop_patience=8,
                min_epochs=10,
                validations_without_improvement=8,
            )
        )
        self.assertFalse(
            should_trigger_early_stop(
                should_validate=True,
                epoch=9,
                early_stop_patience=8,
                min_epochs=10,
                validations_without_improvement=8,
            )
        )
        self.assertFalse(
            should_trigger_early_stop(
                should_validate=True,
                epoch=20,
                early_stop_patience=0,
                min_epochs=10,
                validations_without_improvement=20,
            )
        )
        self.assertTrue(
            should_trigger_early_stop(
                should_validate=True,
                epoch=20,
                early_stop_patience=8,
                min_epochs=10,
                validations_without_improvement=8,
            )
        )

    def test_align_split_boundary_index_prefers_day_boundary(self) -> None:
        timestamps = np.array(
            [
                "2024-01-01T23:55:00",
                "2024-01-02T00:00:00",
                "2024-01-02T00:05:00",
                "2024-01-03T00:00:00",
            ],
            dtype="datetime64[ns]",
        )
        import pandas as pd

        index, summary = align_split_boundary_index(
            pd.Series(pd.to_datetime(timestamps)),
            raw_index=0,
            mode="day",
            lower_bound=0,
            upper_bound=3,
        )
        self.assertEqual(index, 1)
        self.assertEqual(summary["effective_index"], 1)
        self.assertTrue(summary["matched_boundary"])

    def test_resolve_split_boundaries_supports_week_alignment(self) -> None:
        import pandas as pd

        timestamps = pd.date_range("2024-01-01", periods=14 * 24 * 12, freq="5min")
        time_meta = pd.DataFrame({"timestamp": timestamps})
        train_end, val_end, summary = resolve_split_boundaries(
            time_meta,
            alignment="week",
        )

        self.assertGreater(val_end, train_end)
        self.assertEqual(summary["alignment"], "week")
        self.assertIn("calendar", summary)
        self.assertIn("train", summary["calendar"])

    def test_build_time_contained_split_indices_accepts_explicit_boundaries(self) -> None:
        splits = build_time_contained_split_indices(
            100,
            hist_len=12,
            pred_horizon=12,
            train_end=70,
            val_end=80,
        )
        self.assertTrue(all(idx + 24 <= 70 for idx in splits["train"]))
        self.assertTrue(all(idx >= 70 and idx + 24 <= 80 for idx in splits["val"]))
        self.assertTrue(all(idx >= 80 for idx in splits["test"]))

    def test_attach_speed_report_metrics_to_record_serializes_each_horizon(self) -> None:
        record = {"epoch": 5}
        attach_speed_report_metrics_to_record(
            record,
            {
                "report": {
                    "15min": {"rmse": 1.234567},
                    "30min": {"rmse": 2.345678},
                    "60min": {"rmse": 3.456789},
                }
            },
        )

        self.assertEqual(record["val_raw_speed_rmse_15min"], 1.23457)
        self.assertEqual(record["val_raw_speed_rmse_30min"], 2.34568)
        self.assertEqual(record["val_raw_speed_rmse_60min"], 3.45679)

    def test_build_project_monthly_split_summary_records_rule_and_targets(self) -> None:
        import pandas as pd

        time_meta = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=31 * 24 * 12, freq="5min"),
            }
        )
        time_meta["date"] = time_meta["timestamp"].dt.normalize()
        split_indices = {
            "train": [0],
            "val": [20 * 24 * 12],
            "test": [24 * 24 * 12],
        }
        summary = build_project_monthly_split_summary(
            time_meta,
            split_indices,
            hist_len=12,
            pred_horizon=12,
        )

        self.assertEqual(summary["mode"], "project_monthly")
        self.assertEqual(summary["rule"]["train"], "days 1-20")
        self.assertIn("first_target_timestamp", summary["calendar"]["train"])

    def test_processed_dataset_fingerprint_ignores_path_fields(self) -> None:
        speed_values = np.array([[1.0, 2.0]], dtype=np.float32)
        speed_valid_mask = np.array([[True, True]], dtype=bool)
        dataset_summary_a = {
            "dataset_name": "METR-LA",
            "traffic_file": "D:\\a\\metr-la.h5",
            "graph_source": {
                "mode": "official_distance_csv",
                "distance_file": "D:\\a\\distances_la_2012.csv",
            },
            "actual_start": "2012-03-01T00:00:00",
        }
        dataset_summary_b = {
            "dataset_name": "METR-LA",
            "traffic_file": "E:\\different\\metr-la.h5",
            "graph_source": {
                "mode": "official_distance_csv",
                "distance_file": "E:\\different\\distances_la_2012.csv",
            },
            "actual_start": "2012-03-01T00:00:00",
        }

        fingerprint_a = compute_processed_dataset_fingerprint(
            speed_values=speed_values,
            speed_valid_mask=speed_valid_mask,
            dataset_summary=dataset_summary_a,
        )
        fingerprint_b = compute_processed_dataset_fingerprint(
            speed_values=speed_values,
            speed_valid_mask=speed_valid_mask,
            dataset_summary=dataset_summary_b,
        )

        self.assertEqual(fingerprint_a, fingerprint_b)

    def test_resolve_outlier_cleaning_none_is_noop(self) -> None:
        speed_values = np.array(
            [
                [10.0, 20.0],
                [11.0, 21.0],
                [12.0, 22.0],
            ],
            dtype=np.float32,
        )
        speed_valid_mask = np.array(
            [
                [True, True],
                [True, True],
                [True, False],
            ],
            dtype=bool,
        )
        train_time_mask = np.array([True, True, False], dtype=bool)

        cleaned, outlier_mask, summary = resolve_outlier_cleaning(
            speed_values=speed_values,
            speed_valid_mask=speed_valid_mask,
            train_time_mask=train_time_mask,
            mode="none",
            lower_quantile=0.01,
            upper_quantile=0.99,
        )

        self.assertTrue(np.array_equal(cleaned, speed_values))
        self.assertFalse(bool(outlier_mask.any()))
        self.assertFalse(summary["enabled"])
        self.assertEqual(summary["method"], "none")
        self.assertEqual(summary["cleaned_points"], 0)

    def test_load_prepared_sensor_dataset_supports_pseudo_edge_representation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            np.save(root / "speed_values.npy", np.ones((4, 3), dtype=np.float32))
            np.save(root / "speed_valid_mask.npy", np.ones((4, 3), dtype=np.bool_))
            np.save(root / "adjacency_matrix.npy", np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32))
            np.save(root / "adjacency_weights.npy", np.array([[0.0, 0.8], [0.0, 0.0]], dtype=np.float32))
            np.save(root / "edge_index.npy", np.array([[0, 1], [0, 1], [1, 0]], dtype=np.int64))
            np.save(root / "edge_lengths_km.npy", np.array([0.1, 0.2, 0.3], dtype=np.float32))
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=4, freq="5min"),
                    "date": ["2024-01-01"] * 4,
                    "slot": [0, 1, 2, 3],
                    "day_of_week": [0, 0, 0, 0],
                }
            ).to_csv(root / "time_meta.csv", index=False)
            (root / "dataset_summary.json").write_text(
                json.dumps({"representation_domain": "pseudo_edge"}),
                encoding="utf-8",
            )

            dataset, time_feature_names, _ = load_prepared_sensor_dataset(root, disable_time_features=False)
            self.assertEqual(dataset["representation_domain"], "pseudo_edge")
            self.assertEqual(dataset["node_features"].shape, (4, 2, 8))
            self.assertEqual(dataset["edge_speeds"].shape, (4, 3))
            self.assertTrue(
                np.allclose(
                    dataset["adjacency_weights"],
                    np.array([[0.0, 0.8], [0.0, 0.0]], dtype=np.float32),
                )
            )
            self.assertEqual(len(time_feature_names), 6)

    def test_resolve_fixed_edge_length_feature_mode_disables_official_adj_sensor_node(self) -> None:
        enabled, mode = resolve_fixed_edge_length_feature_mode(
            {
                "graph_source": {
                    "mode": "official_adj_pkl",
                }
            },
            representation_domain="sensor_node",
        )
        self.assertFalse(enabled)
        self.assertEqual(mode, "disabled_for_official_adj_pkl_sensor_node")

    def test_resolve_fixed_edge_length_feature_mode_keeps_pseudo_edge_enabled(self) -> None:
        enabled, mode = resolve_fixed_edge_length_feature_mode(
            {
                "graph_source": {
                    "mode": "official_adj_pkl",
                    "distance_semantics": {
                        "edge_lengths_km": "disabled_zero_no_custom_length_feature",
                    },
                }
            },
            representation_domain="pseudo_edge",
        )
        self.assertTrue(enabled)
        self.assertEqual(mode, "enabled_pseudo_edge")

    def test_spatiotemporal_dataset_emits_history_mask_and_imputed_speed_seq(self) -> None:
        node_features = np.zeros((4, 2, 2), dtype=np.float32)
        edge_speeds = np.array(
            [
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 3.0],
                [4.0, 5.0],
            ],
            dtype=np.float32,
        )
        speed_valid_mask = np.array(
            [
                [False, True],
                [True, False],
                [False, True],
                [True, True],
            ],
            dtype=bool,
        )

        dataset = SpatioTemporalDataset(
            node_features,
            edge_speeds,
            edge_speed_valid_mask=speed_valid_mask,
            hist_len=3,
            pred_horizon=1,
        )
        sample = dataset[0]

        expected_speed_seq = np.array(
            [
                [0.0, 2.0, 2.0],
                [1.0, 1.0, 3.0],
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(sample["speed_seq"].numpy(), expected_speed_seq))
        self.assertTrue(np.array_equal(sample["speed_history_mask"].numpy(), speed_valid_mask[:3].T))

    def test_spatiotemporal_dataset_can_disable_history_imputation_but_keep_target_mask(self) -> None:
        node_features = np.zeros((4, 2, 2), dtype=np.float32)
        edge_speeds = np.array(
            [
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 3.0],
                [4.0, 5.0],
            ],
            dtype=np.float32,
        )
        speed_valid_mask = np.array(
            [
                [False, True],
                [True, False],
                [False, True],
                [True, True],
            ],
            dtype=bool,
        )

        dataset = SpatioTemporalDataset(
            node_features,
            edge_speeds,
            edge_speed_valid_mask=speed_valid_mask,
            edge_speed_history_valid_mask=None,
            history_imputation_enabled=False,
            hist_len=3,
            pred_horizon=1,
        )
        sample = dataset[0]

        self.assertTrue(np.array_equal(sample["speed_seq"].numpy(), edge_speeds[:3].T))
        self.assertNotIn("speed_history_mask", sample)
        self.assertTrue(
            np.array_equal(
                sample["speed_target_mask"].numpy(),
                speed_valid_mask[3:4].T,
            )
        )

    def test_collect_speed_edge_trainable_parameters_returns_edge_modules(self) -> None:
        edge_index = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 0],
            ],
            dtype=torch.long,
        )
        edge_lengths = torch.ones(edge_index.shape[0], dtype=torch.float32)
        adjacency = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        model = STGATPredictor(
            num_nodes=3,
            edge_index=edge_index,
            edge_lengths=edge_lengths,
            adj_matrix=adjacency,
            hidden_dim=8,
            num_heads=2,
            num_st_blocks=1,
            num_gtcn_layers=1,
            pred_horizon=3,
            adaptive_topk=1,
            speed_use_adaptive=True,
            speed_adaptive_topk=1,
            v_domain="edge",
        )
        params = collect_speed_edge_trainable_parameters(model)
        self.assertGreater(len(params), 0)

    def test_build_resume_safe_run_config_ignores_epoch_budget(self) -> None:
        config = {
            "dataset_name": "METR-LA",
            "epochs": 20,
            "batch_size": 32,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
        }
        comparable = build_resume_safe_run_config(config)
        self.assertEqual(
            comparable,
            {
                "dataset_name": "METR-LA",
                "batch_size": 32,
                "hidden_dim": 32,
                "num_heads": 4,
                "num_st_blocks": 2,
            },
        )

    def test_build_resume_safe_run_config_keeps_seed_but_not_stop_threshold(self) -> None:
        config = {
            "dataset_name": "METR-LA",
            "seed": 17,
            "batch_size": 32,
            "stop_when_best_val_rmse_below": 5.65,
            "epochs": 40,
        }
        comparable = build_resume_safe_run_config(config)
        self.assertEqual(
            comparable,
            {
                "dataset_name": "METR-LA",
                "seed": 17,
                "batch_size": 32,
            },
        )

    def test_load_checkpoint_allows_epoch_budget_extension(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "stgat_latest.pt"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
            run_config = {
                "dataset_name": "METR-LA",
                "dataset_dir": "prepared/METR-LA",
                "processed_dataset_fingerprint": "abc",
                "epochs": 20,
                "batch_size": 32,
                "lr": 1e-3,
            }
            save_checkpoint(
                ckpt_path,
                epoch=20,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=[{"epoch": 20, "elapsed": 12.5, "lr": 1e-3}],
                best_val_rmse=1.23,
                best_val_loss=0.45,
                best_epoch=18,
                validations_without_improvement=2,
                elapsed_offset=12.5,
                run_config=run_config,
            )

            loaded = load_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=torch.device("cpu"),
                expected_run_config={**run_config, "epochs": 50},
            )

            self.assertEqual(loaded[0], 20)
            self.assertEqual(loaded[1][-1]["epoch"], 20)
            self.assertAlmostEqual(loaded[6], 12.5)

    def test_load_checkpoint_rejects_invariant_config_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "stgat_latest.pt"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
            run_config = {
                "dataset_name": "METR-LA",
                "dataset_dir": "prepared/METR-LA",
                "processed_dataset_fingerprint": "abc",
                "epochs": 20,
                "batch_size": 32,
                "lr": 1e-3,
            }
            save_checkpoint(
                ckpt_path,
                epoch=20,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=[],
                best_val_rmse=1.23,
                best_val_loss=0.45,
                best_epoch=18,
                validations_without_improvement=2,
                elapsed_offset=12.5,
                run_config=run_config,
            )

            with self.assertRaises(RuntimeError):
                load_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=torch.device("cpu"),
                    expected_run_config={**run_config, "dataset_name": "PEMS-BAY"},
                )

    def test_load_checkpoint_rejects_history_tail_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "stgat_latest.pt"
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
            run_config = {
                "dataset_name": "METR-LA",
                "dataset_dir": "prepared/METR-LA",
                "processed_dataset_fingerprint": "abc",
                "epochs": 20,
                "batch_size": 32,
                "lr": 1e-3,
            }
            save_checkpoint(
                ckpt_path,
                epoch=20,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=[{"epoch": 19, "elapsed": 11.0}],
                best_val_rmse=1.23,
                best_val_loss=0.45,
                best_epoch=18,
                validations_without_improvement=2,
                elapsed_offset=12.5,
                run_config=run_config,
            )

            with self.assertRaises(RuntimeError):
                load_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=torch.device("cpu"),
                    expected_run_config=run_config,
                )


if __name__ == "__main__":
    unittest.main()
