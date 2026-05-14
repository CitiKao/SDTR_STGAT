from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dc_benchmark.baselines import PAPER_METHODS, run_historical_average
from dc_benchmark.dataset import export_dc_benchmark, load_dc_benchmark
from dc_benchmark.metrics import evaluate_dc_predictions


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HORIZONS = ["15min", "30min", "60min"]


def _assert_report_horizon_metrics(testcase: unittest.TestCase, metrics: dict) -> None:
    testcase.assertEqual(list(metrics["horizon_reports"]), EXPECTED_HORIZONS)
    testcase.assertEqual(len(metrics["horizon_table"]), 9)
    for task in ("demand", "supply", "gap"):
        report = metrics[task]["report"]
        testcase.assertEqual(list(report), EXPECTED_HORIZONS)
        for horizon in EXPECTED_HORIZONS:
            for metric_name in ("mae", "rmse", "mape"):
                testcase.assertIn(metric_name, report[horizon])
                testcase.assertTrue(math.isfinite(report[horizon][metric_name]))


def _write_fixture(
    root: Path,
    *,
    time_steps: int = 64,
    num_nodes: int = 4,
    observed_time_mask: np.ndarray | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    base = pd.Timestamp("2024-01-01 00:00")
    timestamps = [base + pd.Timedelta(minutes=15 * idx) for idx in range(time_steps)]
    demand = np.arange(time_steps * num_nodes, dtype=np.float32).reshape(time_steps, num_nodes) % 23
    demand = demand + 1.0
    supply = demand * 2.0 + 5.0
    adjacency = np.eye(num_nodes, dtype=np.float32)
    for node in range(num_nodes - 1):
        adjacency[node, node + 1] = 1.0
        adjacency[node + 1, node] = 1.0
    edge_index = np.argwhere(adjacency > 0).astype(np.int32)
    time_meta = pd.DataFrame(
        {
            "date": [ts.date().isoformat() for ts in timestamps],
            "hour": [ts.hour for ts in timestamps],
            "minute": [ts.minute for ts in timestamps],
            "slot": [ts.hour * 4 + ts.minute // 15 for ts in timestamps],
            "day_of_week": [ts.dayofweek for ts in timestamps],
        }
    )

    np.save(root / "node_demand.npy", demand)
    np.save(root / "node_supply.npy", supply)
    np.save(root / "adjacency_matrix.npy", adjacency)
    np.save(root / "edge_index.npy", edge_index)
    if observed_time_mask is not None:
        np.save(root / "observed_time_mask.npy", observed_time_mask.astype(bool))
    time_meta.to_csv(root / "time_meta.csv", index=False)


class DCBenchmarkTest(unittest.TestCase):
    def test_all_paper_methods_have_script_entrypoints(self) -> None:
        scripts_text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "dc_benchmark" / "paper_runs").glob("*.py"))
        for method_id in PAPER_METHODS:
            with self.subTest(method_id=method_id):
                self.assertIn(f'"{method_id}"', scripts_text)

    def test_export_load_and_ha_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source"
            output = tmp_path / "benchmark"
            _write_fixture(source)

            export_dc_benchmark(
                source_data_dir=source,
                output_dir=output,
                hist_len=2,
                pred_horizon=4,
                report_horizons_minutes="15,30,60",
                split_policy="benchmark_contiguous",
                force=True,
            )

            self.assertTrue((output / "manifest.json").exists())
            self.assertTrue((output / "splits.json").exists())
            self.assertTrue((output / "targets_dc.npy").exists())
            self.assertTrue((output / "node_demand.npy").exists())
            self.assertTrue((output / "node_supply.npy").exists())
            self.assertTrue((output / "observed_time_mask.npy").exists())

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], "dc_benchmark_v1")
            self.assertEqual(manifest["hist_len"], 2)
            self.assertEqual(manifest["pred_horizon"], 4)
            self.assertEqual(manifest["target_channels"], ["demand", "supply"])
            self.assertEqual(manifest["shapes"]["targets_dc"], [64, 4, 2])
            self.assertEqual(manifest["report_horizons"]["resolved_minutes"], [15, 30, 60])
            self.assertEqual(manifest["report_horizons"]["resolved_steps"], [1, 2, 4])
            self.assertGreater(splits["sample_counts"]["train"], 0)
            self.assertGreater(splits["sample_counts"]["val"], 0)
            self.assertGreater(splits["sample_counts"]["test"], 0)

            window_len = 6
            for idx in splits["indices"]["train"]:
                self.assertLessEqual(idx + window_len, 45)
            for idx in splits["indices"]["val"]:
                self.assertGreaterEqual(idx, 45)
                self.assertLessEqual(idx + window_len, 51)
            for idx in splits["indices"]["test"]:
                self.assertGreaterEqual(idx, 51)
                self.assertLessEqual(idx + window_len, 64)

            benchmark = load_dc_benchmark(output)
            self.assertEqual(tuple(benchmark["targets"].shape), (64, 4, 2))
            self.assertEqual(tuple(benchmark["observed_time_mask"].shape), (64,))
            result = run_historical_average(benchmark, batch_size=8)
            metrics = result["metrics"]
            self.assertTrue(math.isfinite(metrics["demand"]["rmse"]))
            self.assertTrue(math.isfinite(metrics["supply"]["rmse"]))
            _assert_report_horizon_metrics(self, metrics)
            self.assertAlmostEqual(
                metrics["raw_dc"],
                metrics["demand"]["rmse"] + metrics["supply"]["rmse"],
                places=7,
            )

    def test_dc_metrics_exact_values(self) -> None:
        demand_target = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        demand_pred = np.array([12.0, 18.0, 33.0], dtype=np.float32)
        supply_target = np.array([3.0, 5.0, 10.0], dtype=np.float32)
        supply_pred = np.array([2.0, 8.0, 9.0], dtype=np.float32)
        target = np.stack([demand_target, supply_target], axis=-1).reshape(1, 3, 1, 2)
        pred = np.stack([demand_pred, supply_pred], axis=-1).reshape(1, 3, 1, 2)

        metrics = evaluate_dc_predictions(
            pred,
            target,
            pred_horizon=1,
            report_horizons={"slot_minutes": 15, "resolved_minutes": [15], "resolved_steps": [1]},
        )

        self.assertAlmostEqual(metrics["demand"]["mae"], 7.0 / 3.0, places=7)
        self.assertAlmostEqual(metrics["demand"]["rmse"], math.sqrt(17.0 / 3.0), places=7)
        self.assertAlmostEqual(metrics["demand"]["mape"], 13.333333333333334, places=7)
        self.assertAlmostEqual(metrics["supply"]["mae"], 5.0 / 3.0, places=7)
        self.assertAlmostEqual(metrics["supply"]["rmse"], math.sqrt(11.0 / 3.0), places=7)
        self.assertAlmostEqual(metrics["supply"]["mape"], 34.44444444444444, places=7)
        self.assertAlmostEqual(metrics["gap"]["mae"], 4.0, places=7)
        self.assertAlmostEqual(metrics["gap"]["rmse"], math.sqrt(50.0 / 3.0), places=7)
        self.assertAlmostEqual(metrics["gap"]["mape"], 32.06349206349206, places=7)
        self.assertAlmostEqual(
            metrics["raw_dc"],
            math.sqrt(17.0 / 3.0) + math.sqrt(11.0 / 3.0),
            places=7,
        )

    def test_dc_metrics_horizon_table_includes_15_30_60_minutes(self) -> None:
        target = np.ones((1, 2, 4, 2), dtype=np.float32)
        pred = target.copy()
        pred[:, :, 0, 0] += 1.0
        pred[:, :, 1, 1] += 2.0
        pred[:, :, 3, :] += 3.0

        metrics = evaluate_dc_predictions(
            pred,
            target,
            pred_horizon=4,
            report_horizons={"slot_minutes": 15, "resolved_minutes": [15, 30, 60], "resolved_steps": [1, 2, 4]},
        )

        _assert_report_horizon_metrics(self, metrics)
        self.assertAlmostEqual(metrics["horizon_reports"]["15min"]["demand"]["rmse"], 1.0, places=7)
        self.assertAlmostEqual(metrics["horizon_reports"]["30min"]["supply"]["rmse"], 2.0, places=7)
        self.assertAlmostEqual(metrics["horizon_reports"]["60min"]["raw_dc"], 6.0, places=7)

    def test_paper_named_entrypoint_help_and_ha_smoke(self) -> None:
        script = ROOT / "dc_benchmark" / "paper_runs" / "HA.py"
        help_result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("paper-named DC benchmark", help_result.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source"
            dataset = tmp_path / "benchmark"
            runs = tmp_path / "runs"
            _write_fixture(source)
            run_result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--auto-export",
                    "--source-data-dir",
                    str(source),
                    "--dataset-dir",
                    str(dataset),
                    "--output-dir",
                    str(runs),
                    "--hist-len",
                    "2",
                    "--pred-horizon",
                    "4",
                    "--report-horizons-minutes",
                    "15,30,60",
                    "--split-policy",
                    "benchmark_contiguous",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("Historical Average", run_result.stdout)
            for horizon in EXPECTED_HORIZONS:
                self.assertIn(horizon, run_result.stdout)
            payload = json.loads((runs / "HA_dc_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["method_id"], "ha")
            self.assertTrue(payload["not_official_paper_result"])
            self.assertIn("raw_dc", payload["metrics"])
            _assert_report_horizon_metrics(self, payload["metrics"])

            csv_path = runs / "HA_horizon_metrics.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 9)
            self.assertEqual([row["horizon"] for row in rows[:3]], ["15min", "15min", "15min"])
            self.assertEqual(rows[0]["result_json"], "HA_dc_metrics.json")

            combined_csv = runs / "combined.csv"
            combined_md = runs / "combined.md"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dc_benchmark.summarize_paper_results",
                    "--input-dir",
                    str(runs),
                    "--output-csv",
                    str(combined_csv),
                    "--output-md",
                    str(combined_md),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            with combined_csv.open(encoding="utf-8", newline="") as handle:
                combined_rows = list(csv.DictReader(handle))
            self.assertEqual(len(combined_rows), 9)
            self.assertIn("30min", {row["horizon"] for row in combined_rows})
            self.assertIn("step 2 ahead", combined_md.read_text(encoding="utf-8"))

    def test_export_filters_windows_touching_unobserved_time_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source"
            output = tmp_path / "benchmark"
            observed = np.ones(64, dtype=bool)
            observed[8:10] = False
            _write_fixture(source, observed_time_mask=observed)

            export_dc_benchmark(
                source_data_dir=source,
                output_dir=output,
                hist_len=2,
                pred_horizon=4,
                report_horizons_minutes="15,30,60",
                split_policy="benchmark_contiguous",
                force=True,
            )

            splits = json.loads((output / "splits.json").read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            exported_mask = np.load(output / "observed_time_mask.npy")
            self.assertFalse(bool(exported_mask[8]))
            self.assertGreater(sum(splits["excluded_by_observed_time_mask"].values()), 0)
            self.assertEqual(manifest["observed_time"]["observed_slots"], 62)
            for indices in splits["indices"].values():
                for idx in indices:
                    self.assertTrue(observed[idx: idx + 6].all())


if __name__ == "__main__":
    unittest.main()
