import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_predictor import evaluate_loader_raw_metrics, resolve_report_horizons


class SingleBatchDataset(Dataset):
    def __init__(self) -> None:
        self.node_seq = torch.tensor(
            [[[1.0, 2.0, 0.1], [3.0, 4.0, 0.2]]],
            dtype=torch.float32,
        )  # (N=1, h=2, C=3)
        self.speed_seq = torch.tensor([[5.0, 6.0]], dtype=torch.float32)  # (E=1, h=2)
        self.speed_target = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float32)
        self.zero_target = torch.zeros((1, 4), dtype=torch.float32)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "node_seq": self.node_seq.clone(),
            "speed_seq": self.speed_seq.clone(),
            "demand_target": self.zero_target.clone(),
            "supply_target": self.zero_target.clone(),
            "speed_target": self.speed_target.clone(),
        }


class FakeVModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "pred_speed",
            torch.tensor([[[11.0, 18.0, 33.0, 38.0]]], dtype=torch.float32),
        )

    def forward_v(
        self,
        speed_seq: torch.Tensor,
        temporal_context: torch.Tensor | None,
    ) -> torch.Tensor:
        if temporal_context is None:
            raise AssertionError("Expected temporal context for the V-only path test.")
        batch_size = speed_seq.shape[0]
        return self.pred_speed.expand(batch_size, -1, -1)


class VReportMetricsTest(unittest.TestCase):
    def test_v_report_metrics_preserve_aggregate_and_report_steps(self) -> None:
        loader = DataLoader(SingleBatchDataset(), batch_size=1, shuffle=False)
        model = FakeVModel()
        report_horizons = resolve_report_horizons(
            time_slot_minutes=15,
            pred_horizon=4,
            requested_minutes=[15, 30, 60],
            strict=True,
        )

        metrics = evaluate_loader_raw_metrics(
            model,
            loader,
            train_task="v",
            device=torch.device("cpu"),
            non_blocking=False,
            amp_enabled=False,
            amp_dtype=None,
            normalization_stats={
                "speed": {
                    "transform": "zscore",
                    "mean": np.zeros((1,), dtype=np.float32),
                    "std": np.ones((1,), dtype=np.float32),
                }
            },
            time_slot_minutes=15,
            report_horizons=report_horizons,
        )

        speed_metrics = metrics["speed"]
        self.assertAlmostEqual(speed_metrics["mae"], 2.0, places=7)
        self.assertAlmostEqual(speed_metrics["rmse"], math.sqrt(4.5), places=7)

        self.assertAlmostEqual(speed_metrics["per_step"]["step_1"]["mae"], 1.0, places=7)
        self.assertAlmostEqual(speed_metrics["per_step"]["step_2"]["mae"], 2.0, places=7)
        self.assertAlmostEqual(speed_metrics["per_step"]["step_4"]["mae"], 2.0, places=7)
        self.assertEqual(speed_metrics["per_step"]["step_4"]["minutes"], 60.0)

        self.assertEqual(metrics["report_horizons"]["resolved_steps"], [1, 2, 4])
        self.assertAlmostEqual(speed_metrics["report"]["15min"]["rmse"], 1.0, places=7)
        self.assertAlmostEqual(speed_metrics["report"]["30min"]["rmse"], 2.0, places=7)
        self.assertAlmostEqual(speed_metrics["report"]["60min"]["rmse"], 2.0, places=7)

    def test_resolve_report_horizons_fails_when_step_is_unavailable(self) -> None:
        with self.assertRaises(ValueError):
            resolve_report_horizons(
                time_slot_minutes=15,
                pred_horizon=3,
                requested_minutes=[15, 30, 60],
                strict=True,
            )


if __name__ == "__main__":
    unittest.main()
