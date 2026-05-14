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
import torch

from dc_benchmark.dataset import export_dc_benchmark, load_dc_benchmark
from dc_benchmark.neural import (
    DCWindowTorchDataset,
    DeepMultiScaleConvLSTMPaperReimplementation,
    MLRNNPaperReimplementation,
    MTMFGCNPaperReimplementation,
    build_paper_reimplementation_model,
    run_neural_paper_baseline,
)
from dc_benchmark.official_sources import require_official_method


ROOT = Path(__file__).resolve().parents[1]
TARGET_METHODS = ("mlrnn_taxi_demand", "deep_multiconvlstm", "mt_mf_gcn")
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


def _write_fixture(root: Path, *, time_steps: int = 64, num_nodes: int = 4) -> None:
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
    time_meta.to_csv(root / "time_meta.csv", index=False)


class PaperReimplementationTest(unittest.TestCase):
    def _make_benchmark(self, tmp_path: Path) -> Path:
        source = tmp_path / "source"
        dataset = tmp_path / "benchmark"
        _write_fixture(source)
        export_dc_benchmark(
            source_data_dir=source,
            output_dir=dataset,
            hist_len=4,
            pred_horizon=4,
            report_horizons_minutes="15,30,60",
            split_policy="benchmark_contiguous",
            force=True,
        )
        return dataset

    def test_paper_models_return_dc_prediction_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = load_dc_benchmark(self._make_benchmark(Path(tmp)))
            dataset = DCWindowTorchDataset(benchmark, "train", limit=2)
            history = torch.stack([dataset[0]["history"], dataset[1]["history"]], dim=0)
            target = torch.stack([dataset[0]["target"], dataset[1]["target"]], dim=0)

            for method_id in TARGET_METHODS:
                with self.subTest(method_id=method_id):
                    model = build_paper_reimplementation_model(
                        method_id=method_id,
                        benchmark=benchmark,
                        hidden_dim=4,
                    )
                    pred = model(history)
                    self.assertEqual(tuple(pred.shape), (2, 4, 4, 2))
                    self.assertTrue(torch.isfinite(pred).all())

                    if method_id == "mlrnn_taxi_demand":
                        self.assertIsInstance(model, MLRNNPaperReimplementation)
                        self.assertGreaterEqual(model.num_clusters, 2)
                        loss = model.training_loss(history, target)
                        self.assertEqual(tuple(model.node_weights.shape), (1, 4, 1, 1))
                        self.assertTrue(torch.isfinite(loss))
                    elif method_id == "deep_multiconvlstm":
                        self.assertIsInstance(model, DeepMultiScaleConvLSTMPaperReimplementation)
                        self.assertEqual((model.grid_height, model.grid_width), (2, 2))
                    elif method_id == "mt_mf_gcn":
                        self.assertIsInstance(model, MTMFGCNPaperReimplementation)
                        self.assertEqual(int(model.graphs.shape[0]), 4)

    def test_target_methods_are_paper_based_and_not_official(self) -> None:
        for method_id in TARGET_METHODS:
            with self.subTest(method_id=method_id):
                with self.assertRaises(ValueError):
                    require_official_method(method_id)

        with tempfile.TemporaryDirectory() as tmp:
            benchmark = load_dc_benchmark(self._make_benchmark(Path(tmp)))
            for method_id in TARGET_METHODS:
                with self.subTest(run=method_id):
                    result = run_neural_paper_baseline(
                        benchmark,
                        method_id=method_id,
                        epochs=1,
                        batch_size=2,
                        hidden_dim=4,
                        device="cpu",
                        max_train_samples=4,
                        max_eval_samples=2,
                    )
                    self.assertEqual(result["claim_type"], "paper_based_reimplementation")
                    self.assertFalse(result["official_source_verified"])
                    self.assertTrue(result["not_official_paper_result"])
                    self.assertTrue(result["run_config"]["paper_based_reimplementation"])
                    self.assertTrue(math.isfinite(result["metrics"]["raw_dc"]))
                    _assert_report_horizon_metrics(self, result["metrics"])
                    self.assertAlmostEqual(
                        result["metrics"]["raw_dc"],
                        result["metrics"]["demand"]["rmse"] + result["metrics"]["supply"]["rmse"],
                        places=7,
                    )

    def test_target_paper_entrypoints_help_and_write_metric_payloads(self) -> None:
        scripts = {
            "mlrnn_taxi_demand": ROOT / "dc_benchmark" / "paper_runs" / "MLRNN.py",
            "deep_multiconvlstm": ROOT / "dc_benchmark" / "paper_runs" / "Deep_MultiConvLSTM.py",
            "mt_mf_gcn": ROOT / "dc_benchmark" / "paper_runs" / "MT_MF_GCN.py",
        }
        for script in scripts.values():
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
            for method_id, script in scripts.items():
                result_name = f"{method_id}.json"
                subprocess.run(
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
                        "4",
                        "--pred-horizon",
                        "4",
                        "--report-horizons-minutes",
                        "15,30,60",
                        "--split-policy",
                        "benchmark_contiguous",
                        "--epochs",
                        "1",
                        "--batch-size",
                        "2",
                        "--hidden-dim",
                        "4",
                        "--max-train-samples",
                        "4",
                        "--max-eval-samples",
                        "2",
                        "--device",
                        "cpu",
                        "--result-name",
                        result_name,
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=True,
                )
                payload = json.loads((runs / result_name).read_text(encoding="utf-8"))
                self.assertEqual(payload["method_id"], method_id)
                self.assertEqual(payload["claim_type"], "paper_based_reimplementation")
                self.assertTrue(payload["not_official_paper_result"])
                self.assertIn("split_hash", payload["benchmark_manifest"])
                _assert_report_horizon_metrics(self, payload["metrics"])

                csv_path = runs / f"{Path(result_name).stem}_horizon_metrics.csv"
                self.assertTrue(csv_path.exists())
                with csv_path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 9)
                self.assertEqual({row["horizon"] for row in rows}, set(EXPECTED_HORIZONS))
                self.assertEqual({row["target"] for row in rows}, {"demand", "supply", "gap"})
                self.assertTrue(all(row["result_json"] == result_name for row in rows))


if __name__ == "__main__":
    unittest.main()
