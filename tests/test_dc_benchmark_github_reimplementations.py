from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dc_benchmark.dataset import export_dc_benchmark
from dc_benchmark.github_grid_export import export_github_grid_dataset
from dc_benchmark.github_reimplementation_sources import get_github_reimplementation_info
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


class GitHubReimplementationSourceTest(unittest.TestCase):
    def _make_benchmark(self, tmp_path: Path) -> Path:
        source = tmp_path / "source"
        dataset = tmp_path / "benchmark"
        _write_fixture(source)
        export_dc_benchmark(
            source_data_dir=source,
            output_dir=dataset,
            hist_len=4,
            pred_horizon=2,
            report_horizons_minutes="15,30",
            split_policy="benchmark_contiguous",
            force=True,
        )
        return dataset

    def test_st_resnet_source_is_registered_but_not_official(self) -> None:
        info = get_github_reimplementation_info("st_resnet", project_root=ROOT)
        self.assertEqual(info["repo_url"], "https://github.com/topazape/ST-ResNet")
        self.assertEqual(info["license"], "Unlicense")
        self.assertEqual(info["claim_type"], "github_reimplementation_with_dataset_adapter")
        self.assertTrue(info["not_official_paper_result"])
        self.assertTrue(info["available_locally"])
        with self.assertRaisesRegex(ValueError, "unavailable|currently"):
            require_official_method("st_resnet")

    def test_convlstm_source_is_registered_but_not_official(self) -> None:
        info = get_github_reimplementation_info("convlstm", project_root=ROOT)
        self.assertEqual(info["repo_url"], "https://github.com/ndrplz/ConvLSTM_pytorch")
        self.assertEqual(info["license"], "MIT")
        self.assertEqual(info["claim_type"], "github_reimplementation_with_dataset_adapter")
        self.assertTrue(info["not_official_paper_result"])
        self.assertTrue(info["available_locally"])
        with self.assertRaisesRegex(ValueError, "no verified public official"):
            require_official_method("convlstm")

    def test_github_convlstm_imports_and_runs_forward(self) -> None:
        source = ROOT / "dc_benchmark" / "external_github_reimplementations" / "ConvLSTM_pytorch" / "convlstm.py"
        spec = importlib.util.spec_from_file_location("github_convlstm", source)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = module.ConvLSTM(
            input_dim=2,
            hidden_dim=[4],
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
            bias=True,
            return_all_layers=False,
        )
        layer_outputs, states = model(torch.randn(2, 3, 2, 5, 6))
        self.assertEqual(tuple(layer_outputs[0].shape), (2, 3, 4, 5, 6))
        self.assertEqual(tuple(states[0][0].shape), (2, 4, 5, 6))

    def test_github_st_resnet_imports_and_runs_forward(self) -> None:
        upstream = ROOT / "dc_benchmark" / "external_github_reimplementations" / "ST-ResNet"
        sys.path.insert(0, str(upstream))
        try:
            from stresnet.models import STResNet

            model = STResNet(
                len_closeness=3,
                len_period=1,
                len_trend=1,
                external_dim=None,
                nb_flow=2,
                map_height=5,
                map_width=5,
                nb_residual_unit=1,
            )
            out = model(
                torch.randn(2, 6, 5, 5),
                torch.randn(2, 2, 5, 5),
                torch.randn(2, 2, 5, 5),
                None,
            )
            self.assertEqual(tuple(out.shape), (2, 2, 5, 5))
        finally:
            if str(upstream) in sys.path:
                sys.path.remove(str(upstream))

    def test_convlstm_grid_export_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = self._make_benchmark(tmp_path)
            output = export_github_grid_dataset(
                method_id="convlstm",
                dataset_dir=dataset,
                output_dir=tmp_path / "github_grid",
                max_samples_per_split=2,
                project_root=ROOT,
            )
            manifest = json.loads((output / "github_grid_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["claim_type"], "github_reimplementation_with_dataset_adapter")
            self.assertTrue(manifest["not_official_paper_result"])
            self.assertEqual(manifest["grid_layout"]["height"], 2)
            self.assertEqual(manifest["grid_layout"]["width"], 2)
            with np.load(output / "train.npz") as train:
                self.assertEqual(tuple(train["x"].shape), (2, 4, 2, 2, 2))
                self.assertEqual(tuple(train["y"].shape), (2, 2, 2, 2, 2))

    def test_st_resnet_grid_export_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dataset = self._make_benchmark(tmp_path)
            output = export_github_grid_dataset(
                method_id="st_resnet",
                dataset_dir=dataset,
                output_dir=tmp_path / "github_grid",
                max_samples_per_split=2,
                project_root=ROOT,
            )
            manifest = json.loads((output / "github_grid_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source"]["repo_url"], "https://github.com/topazape/ST-ResNet")
            self.assertEqual(manifest["st_resnet_temporal_slices"]["len_closeness"], 3)
            with np.load(output / "train.npz") as train:
                self.assertEqual(tuple(train["xc"].shape), (2, 6, 2, 2))
                self.assertEqual(tuple(train["xp"].shape), (2, 2, 2, 2))
                self.assertEqual(tuple(train["xt"].shape), (2, 2, 2, 2))
                self.assertEqual(tuple(train["y"].shape), (2, 2, 2, 2, 2))
                self.assertEqual(tuple(train["y_first"].shape), (2, 2, 2, 2))


if __name__ == "__main__":
    unittest.main()
