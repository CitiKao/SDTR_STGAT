from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dc_benchmark.dataset import export_dc_benchmark
from dc_benchmark.official_export import export_official_dataset
from dc_benchmark.official_sources import require_official_method


ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(root: Path, *, time_steps: int = 64, num_nodes: int = 4) -> None:
    root.mkdir(parents=True, exist_ok=True)
    base = pd.Timestamp("2024-01-01 00:00")
    timestamps = [base + pd.Timedelta(minutes=15 * idx) for idx in range(time_steps)]
    demand = np.arange(time_steps * num_nodes, dtype=np.float32).reshape(time_steps, num_nodes) % 23
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


class OfficialDCBenchmarkTest(unittest.TestCase):
    def _make_benchmark(self, tmp_path: Path) -> Path:
        source = tmp_path / "source"
        dataset = tmp_path / "benchmark"
        _write_fixture(source)
        export_dc_benchmark(
            source_data_dir=source,
            output_dir=dataset,
            hist_len=2,
            pred_horizon=2,
            report_horizons_minutes="15,30",
            split_policy="benchmark_contiguous",
            force=True,
        )
        return dataset

    def test_dcrnn_official_export_has_demand_and_supply_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = self._make_benchmark(tmp_path)
            output = export_official_dataset(
                method_id="dcrnn",
                dataset_dir=dataset,
                output_dir=tmp_path / "official",
                max_samples_per_split=3,
                project_root=ROOT,
            )

            manifest = json.loads((output / "official_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["method_id"], "dcrnn")
            self.assertEqual(manifest["official_source"]["repo_url"], "https://github.com/liyaguang/DCRNN")
            self.assertEqual(manifest["export"]["num_nodes"], 4)
            self.assertEqual(manifest["export"]["pred_horizon"], 2)
            self.assertIn("data_loader", manifest["allowed_modifications"])
            self.assertIn("output_head", manifest["allowed_modifications"])

            with np.load(output / "demand" / "train.npz") as train:
                self.assertEqual(tuple(train["x"].shape), (3, 2, 4, 2))
                self.assertEqual(tuple(train["y"].shape), (3, 2, 4, 2))
            self.assertTrue((output / "adj_mx.pkl").exists())
            self.assertTrue((output / "demand" / "dcrnn_config.yaml").exists())
            self.assertTrue(all("FROM_MANIFEST" not in command for command in manifest["command_templates"]))

    def test_graph_wavenet_command_template_uses_manifest_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = self._make_benchmark(tmp_path)
            output = export_official_dataset(
                method_id="graph_wavenet",
                dataset_dir=dataset,
                output_dir=tmp_path / "official",
                max_samples_per_split=2,
                project_root=ROOT,
            )

            manifest = json.loads((output / "official_manifest.json").read_text(encoding="utf-8"))
            commands = "\n".join(manifest["command_templates"])
            self.assertIn("--seq_length 2", commands)
            self.assertIn("--num_nodes 4", commands)
            self.assertNotIn("FROM_MANIFEST", commands)
            with np.load(output / "supply" / "train.npz") as train:
                self.assertEqual(tuple(train["x"].shape), (2, 2, 4, 2))
                self.assertEqual(tuple(train["y"].shape), (2, 2, 4, 2))

    def test_stgcn_official_export_writes_sequence_adapter_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = self._make_benchmark(tmp_path)
            output = export_official_dataset(
                method_id="stgcn",
                dataset_dir=dataset,
                output_dir=tmp_path / "official",
                max_samples_per_split=2,
                project_root=ROOT,
            )

            manifest = json.loads((output / "official_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["method_id"], "stgcn")
            self.assertEqual(manifest["official_source"]["repo_url"], "https://github.com/VeritasYin/STGCN_IJCAI-18")
            arr = np.load(output / "demand" / "train_sequence.npy")
            self.assertEqual(tuple(arr.shape), (2, 4, 4, 1))
            self.assertTrue((output / "STDR_W.csv").exists())
            self.assertTrue((output / "supply" / "stgcn_stats.json").exists())
            commands = "\n".join(manifest["command_templates"])
            self.assertIn("official_adapters", commands)
            self.assertIn("--n-route 4", commands)
            self.assertIn("--n-his 2", commands)
            self.assertIn("--n-pred 2", commands)

    def test_unverified_methods_are_rejected_as_official(self) -> None:
        with self.assertRaisesRegex(ValueError, "No verified public official MLRNN"):
            require_official_method("mlrnn_taxi_demand")

    def test_paper_named_official_entrypoint_help(self) -> None:
        script = ROOT / "dc_benchmark" / "official_runs" / "DCRNN.py"
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("verified official-code", result.stdout)
        self.assertIn("--print-commands", result.stdout)


if __name__ == "__main__":
    unittest.main()
