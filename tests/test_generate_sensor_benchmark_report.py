import json
import shutil
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external_speed_benchmarks.generate_sensor_benchmark_report import build_dataset_section  # noqa: E402


class GenerateSensorBenchmarkReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_tmp_root = ROOT / ".tmp_test_generate_sensor_benchmark_report"
        self.workspace_tmp_root.mkdir(parents=True, exist_ok=True)

    def test_build_dataset_section_includes_preprocessing_and_outlier_fields(self) -> None:
        root = self.workspace_tmp_root / "case_cleaned"
        if root.exists():
            shutil.rmtree(root)
        try:
            root.mkdir(parents=True, exist_ok=True)
            dataset_dir = root / "processed" / "METR-LA"
            run_dir = root / "run"
            dataset_dir.mkdir(parents=True)
            run_dir.mkdir(parents=True)

            (dataset_dir / "dataset_summary.json").write_text(
                json.dumps(
                    {
                        "actual_start": "2012-03-01T00:00:00",
                        "actual_end": "2012-06-27T23:55:00",
                        "notes": "Example note",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "stgat_meta.json").write_text(
                json.dumps(
                    {
                        "dataset_name": "METR-LA",
                        "dataset_dir": str(dataset_dir),
                        "preprocessing_variant_id": "train_quantile_clip_q0p01_0p99_masked_zero_excluded",
                        "num_nodes": 207,
                        "num_graph_edges": 1515,
                        "split_strategy": "per_month_day_1_20_train_21_24_val_25_plus_test_full_window_containment",
                        "split_description": "Monthly project split",
                        "split_summary": {
                            "mode": "project_monthly",
                            "rule": {
                                "train": "days 1-20",
                                "val": "days 21-24",
                                "test": "days 25+",
                            },
                            "calendar": {
                                "train": {"first_target_timestamp": "2012-03-01T01:00:00", "last_target_timestamp": "2012-06-20T23:55:00"},
                                "val": {"first_target_timestamp": "2012-03-21T01:00:00", "last_target_timestamp": "2012-06-24T23:55:00"},
                                "test": {"first_target_timestamp": "2012-03-25T01:00:00", "last_target_timestamp": "2012-06-27T23:55:00"},
                            },
                        },
                        "split_counts": {"train": 1, "val": 1, "test": 1},
                        "graph_topology": {
                            "adaptive_enabled": True,
                            "fixed_graph_source": {"mode": "official_distance_csv"},
                        },
                        "optimizer": {"name": "Adam", "lr": 0.001, "weight_decay": 1e-05},
                        "scheduler": {
                            "name": "ReduceLROnPlateau",
                            "monitor": "val_loss",
                            "patience": 8,
                            "cooldown": 2,
                            "min_lr": 1e-05,
                        },
                        "selected_checkpoint": {
                            "best_epoch": 3,
                            "best_val_raw_speed_rmse": 5.1234,
                        },
                        "training_control": {
                            "configured_epochs": 180,
                            "completed_epochs": 180,
                            "completed_validations": 180,
                            "completed_full_budget": True,
                            "training_end_reason": "epoch_budget_exhausted",
                            "early_stopping": {
                                "enabled": False,
                                "patience": 0,
                                "min_epochs": 10,
                                "triggered": False,
                            },
                        },
                        "outlier_cleaning": {
                            "enabled": True,
                            "method": "train_quantile_clip",
                            "fit_scope": "train_history_only",
                            "apply_scope": "all_valid_points",
                            "replace_strategy": "clip",
                            "cleaned_points": 12,
                            "cleaned_ratio": 0.015,
                        },
                        "benchmark_comparability": {
                            "is_official_like": False,
                            "deviations": ["train-only quantile clipping applied before normalization"],
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "predictor_test_metrics.json").write_text(
                json.dumps(
                    {
                        "speed_metric_protocol": "masked_zero_excluded",
                        "raw_metrics_report": {
                            "speed": {
                                "15min": {"rmse": 1.0, "mae": 0.5, "mape": 3.0, "mse": 1.0},
                                "30min": {"rmse": 2.0, "mae": 1.0, "mape": 4.0, "mse": 4.0},
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            lines = build_dataset_section(run_dir)
            joined = "\n".join(lines)
            self.assertIn("## METR-LA [sensor_node | project_monthly | cleaned:train_quantile_clip]", joined)
            self.assertIn("Completed epochs: `180/180`", joined)
            self.assertIn("Stop reason: `epoch_budget_exhausted`", joined)
            self.assertIn("Early stopping: `disabled`", joined)
            self.assertIn("Split description: `Monthly project split`", joined)
            self.assertIn("Split rule: `train=days 1-20 | val=days 21-24 | test=days 25+`", joined)
            self.assertIn("Train first/last target: `2012-03-01T01:00:00` to `2012-06-20T23:55:00`", joined)
            self.assertIn("Optimizer: `Adam | lr=0.001 | weight_decay=1e-05`", joined)
            self.assertIn("Scheduler: `ReduceLROnPlateau | monitor=val_loss | patience=8 | cooldown=2 | min_lr=1e-05`", joined)
            self.assertIn("Preprocessing variant", joined)
            self.assertIn("Outlier cleaning: `enabled | method=train_quantile_clip", joined)
            self.assertIn("Cleaned points: `12 (1.50% of valid points)`", joined)
            self.assertIn("Benchmark comparability: `custom / not directly official-like`", joined)
            self.assertIn("Benchmark deviations: `train-only quantile clipping applied before normalization", joined)
            self.assertIn("Dataset summary source: `dataset_dir:dataset_summary.json`", joined)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_build_dataset_section_marks_raw_runs_and_lists_deviations(self) -> None:
        root = self.workspace_tmp_root / "case_raw"
        if root.exists():
            shutil.rmtree(root)
        try:
            root.mkdir(parents=True, exist_ok=True)
            dataset_dir = root / "processed" / "PEMS-BAY"
            run_dir = root / "run"
            dataset_dir.mkdir(parents=True)
            run_dir.mkdir(parents=True)

            (dataset_dir / "dataset_summary.json").write_text(
                json.dumps(
                    {
                        "actual_start": "2017-01-01T00:00:00",
                        "actual_end": "2017-05-31T23:55:00",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "stgat_meta.json").write_text(
                json.dumps(
                    {
                        "dataset_name": "PEMS-BAY",
                        "dataset_dir": str(dataset_dir),
                        "preprocessing_variant_id": "raw_unmasked_all_values",
                        "num_nodes": 325,
                        "num_graph_edges": 2369,
                        "split_strategy": "per_month_day_1_20_train_21_24_val_25_plus_test_full_window_containment",
                        "split_description": "Monthly project split",
                        "split_summary": {
                            "mode": "project_monthly",
                            "rule": {
                                "train": "days 1-20",
                                "val": "days 21-24",
                                "test": "days 25+",
                            },
                            "calendar": {
                                "train": {"first_target_timestamp": "2017-01-01T01:00:00", "last_target_timestamp": "2017-05-20T23:55:00"},
                                "val": {"first_target_timestamp": "2017-01-21T01:00:00", "last_target_timestamp": "2017-05-24T23:55:00"},
                                "test": {"first_target_timestamp": "2017-01-25T01:00:00", "last_target_timestamp": "2017-05-31T23:55:00"},
                            },
                        },
                        "split_counts": {"train": 1, "val": 1, "test": 1},
                        "graph_topology": {
                            "adaptive_enabled": False,
                            "fixed_graph_source": {"mode": "official_adj_pickle"},
                        },
                        "optimizer": {"name": "Adam", "lr": 0.001, "weight_decay": 1e-05},
                        "scheduler": {
                            "name": "ReduceLROnPlateau",
                            "monitor": "val_rmse",
                            "patience": 10,
                            "cooldown": 0,
                            "min_lr": 1e-06,
                        },
                        "selected_checkpoint": {
                            "best_epoch": 5,
                            "best_val_raw_speed_rmse": 3.1234,
                        },
                        "training_control": {
                            "configured_epochs": 180,
                            "completed_epochs": 27,
                            "completed_validations": 27,
                            "completed_full_budget": False,
                            "training_end_reason": "early_stop_patience",
                            "early_stopping": {
                                "enabled": True,
                                "patience": 8,
                                "min_epochs": 10,
                                "triggered": True,
                            },
                        },
                        "outlier_cleaning": {
                            "enabled": False,
                            "method": "none",
                            "cleaned_points": 0,
                            "cleaned_ratio": 0.0,
                        },
                        "benchmark_comparability": {
                            "is_official_like": False,
                            "deviations": ["speed metrics include masked-zero disabling and are not official-like"],
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "predictor_test_metrics.json").write_text(
                json.dumps(
                    {
                        "speed_metric_protocol": "unmasked_all_values",
                        "raw_metrics_report": {
                            "speed": {
                                "15min": {"rmse": 1.2, "mae": 0.6, "mape": 2.5, "mse": 1.44},
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            joined = "\n".join(build_dataset_section(run_dir))
            self.assertIn("## PEMS-BAY [sensor_node | project_monthly | raw]", joined)
            self.assertIn("Completed epochs: `27/180`", joined)
            self.assertIn("Stop reason: `early_stop_patience`", joined)
            self.assertIn("Early stopping: `enabled | patience=8 | min_epochs=10 | triggered=yes`", joined)
            self.assertIn("Split rule: `train=days 1-20 | val=days 21-24 | test=days 25+`", joined)
            self.assertIn("Outlier cleaning: `disabled`", joined)
            self.assertIn("Benchmark deviations: `speed metrics include masked-zero disabling and are not official-like`", joined)
        finally:
            if root.exists():
                shutil.rmtree(root)

    def test_build_dataset_section_prefers_run_local_pseudo_edge_snapshot(self) -> None:
        root = self.workspace_tmp_root / "case_pseudo_edge"
        if root.exists():
            shutil.rmtree(root)
        try:
            root.mkdir(parents=True, exist_ok=True)
            dataset_dir = root / "processed" / "METR-LA_pseudo_edge"
            run_dir = root / "run"
            dataset_dir.mkdir(parents=True)
            run_dir.mkdir(parents=True)

            (dataset_dir / "dataset_summary.json").write_text(
                json.dumps(
                    {
                        "actual_start": "2012-03-01T00:00:00",
                        "actual_end": "2012-06-27T23:55:00",
                        "representation_domain": "pseudo_edge",
                        "representation_variant_id": "dataset_dir_version",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "prepared_dataset_summary.json").write_text(
                json.dumps(
                    {
                        "actual_start": "2012-03-01T00:00:00",
                        "actual_end": "2012-06-27T23:55:00",
                        "representation_domain": "pseudo_edge",
                        "representation_variant_id": "run_snapshot_version",
                        "notes": "Pseudo-edge run snapshot",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "prepared_pseudo_edge_summary.json").write_text(
                json.dumps(
                    {
                        "self_loop_fixes": 12,
                        "self_loop_fix_ratio": 0.12,
                        "isolated_edge_ratio": 0.18,
                        "cluster_radius_km": 0.15,
                        "construction": {
                            "sensor_base_graph_mode": "official_distance_csv",
                            "cluster_radius_scale": 0.75,
                            "probe_half_length_scale": 0.25,
                            "fallback_neighbor_k": 3,
                        },
                        "topology_summary": {
                            "isolated_edges": 8,
                            "num_unique_structural_edges": 18,
                            "line_graph_weak_components": 5,
                            "line_graph_largest_component_ratio": 0.4,
                        },
                        "health_warnings": ["isolated_edge_ratio=0.180 exceeds 0.150"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "stgat_meta.json").write_text(
                json.dumps(
                    {
                        "dataset_name": "METR-LA",
                        "dataset_dir": str(dataset_dir),
                        "preprocessing_variant_id": "train_quantile_clip_q0p01_0p99_masked_zero_excluded",
                        "representation_domain": "pseudo_edge",
                        "representation_variant_id": "run_snapshot_version",
                        "num_nodes": 213,
                        "num_sensors": 207,
                        "num_graph_nodes": 213,
                        "num_graph_edges": 207,
                        "num_speed_items": 207,
                        "split_strategy": "per_month_day_1_20_train_21_24_val_25_plus_test_full_window_containment",
                        "split_description": "Monthly project split",
                        "split_summary": {
                            "mode": "project_monthly",
                            "rule": {
                                "train": "days 1-20",
                                "val": "days 21-24",
                                "test": "days 25+",
                            },
                            "calendar": {
                                "train": {"first_target_timestamp": "2012-03-01T01:00:00", "last_target_timestamp": "2012-06-20T23:55:00"},
                                "val": {"first_target_timestamp": "2012-03-21T01:00:00", "last_target_timestamp": "2012-06-24T23:55:00"},
                                "test": {"first_target_timestamp": "2012-03-25T01:00:00", "last_target_timestamp": "2012-06-27T23:55:00"},
                            },
                        },
                        "split_counts": {"train": 1, "val": 1, "test": 1},
                        "graph_topology": {
                            "v_domain": "edge",
                            "adaptive_enabled": True,
                            "fixed_graph_source": {"mode": "pseudo_edge_from_sensor_graph"},
                        },
                        "optimizer": {"name": "Adam", "lr": 0.001, "weight_decay": 1e-05},
                        "scheduler": {
                            "name": "ReduceLROnPlateau",
                            "monitor": "val_loss",
                            "patience": 8,
                            "cooldown": 2,
                            "min_lr": 1e-05,
                        },
                        "selected_checkpoint": {
                            "best_epoch": 7,
                            "best_val_raw_speed_rmse": 4.5678,
                        },
                        "training_control": {
                            "configured_epochs": 20,
                            "completed_epochs": 20,
                            "completed_validations": 20,
                            "completed_full_budget": True,
                            "training_end_reason": "epoch_budget_exhausted",
                            "early_stopping": {
                                "enabled": False,
                                "patience": 0,
                                "min_epochs": 10,
                                "triggered": False,
                            },
                        },
                        "outlier_cleaning": {
                            "enabled": True,
                            "method": "train_quantile_clip",
                            "fit_scope": "train_history_only",
                            "apply_scope": "all_valid_points",
                            "replace_strategy": "clip",
                            "cleaned_points": 6,
                            "cleaned_ratio": 0.01,
                        },
                        "benchmark_comparability": {
                            "is_official_like": False,
                            "deviations": ["experimental pseudo-edge representation used instead of the official sensor-node benchmark target"],
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "predictor_test_metrics.json").write_text(
                json.dumps(
                    {
                        "speed_metric_protocol": "masked_zero_excluded",
                        "v_domain": "edge",
                        "raw_metrics_report": {
                            "speed": {
                                "15min": {"rmse": 1.0, "mae": 0.5, "mape": 3.0, "mse": 1.0},
                            }
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            joined = "\n".join(build_dataset_section(run_dir))
            self.assertIn("## METR-LA [pseudo_edge | project_monthly | cleaned:train_quantile_clip]", joined)
            self.assertIn("Representation variant: `run_snapshot_version`", joined)
            self.assertIn("V domain: `edge`", joined)
            self.assertIn("Dataset summary source: `run_snapshot:prepared_dataset_summary.json`", joined)
            self.assertIn("Pseudo-edge topology:", joined)
            self.assertIn("Pseudo-edge construction:", joined)
            self.assertIn("Pseudo-edge warnings: `isolated_edge_ratio=0.180 exceeds 0.150`", joined)
            self.assertIn("Dataset note: `Pseudo-edge run snapshot`", joined)
        finally:
            if root.exists():
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
