import math
import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_predictor import evaluate_loader_raw_metrics, resolve_report_horizons


class SingleBatchDcDataset(Dataset):
    def __init__(self) -> None:
        self.node_seq = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]]],
            dtype=torch.float32,
        )  # (N=1, h=2, C=2)
        self.speed_seq = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
        self.demand_target = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float32)
        self.supply_target = torch.tensor([[4.0, 8.0, 12.0, 16.0]], dtype=torch.float32)
        self.speed_target = torch.zeros((1, 4), dtype=torch.float32)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "node_seq": self.node_seq.clone(),
            "speed_seq": self.speed_seq.clone(),
            "demand_target": self.demand_target.clone(),
            "supply_target": self.supply_target.clone(),
            "speed_target": self.speed_target.clone(),
        }


class FakeDcModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "demand_pred",
            torch.tensor([[[11.0, 18.0, 33.0, 38.0]]], dtype=torch.float32),
        )
        self.register_buffer(
            "supply_pred",
            torch.tensor([[[5.0, 10.0, 9.0, 18.0]]], dtype=torch.float32),
        )
        self.register_buffer(
            "speed_pred",
            torch.zeros((1, 1, 4), dtype=torch.float32),
        )

    def forward(
        self,
        node_seq: torch.Tensor,
        speed_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = node_seq.shape[0]
        return (
            self.demand_pred.expand(batch_size, -1, -1),
            self.supply_pred.expand(batch_size, -1, -1),
            self.speed_pred.expand(batch_size, -1, -1),
        )


class DcReportMetricsTest(unittest.TestCase):
    def test_dc_report_metrics_preserve_aggregate_and_report_steps(self) -> None:
        loader = DataLoader(SingleBatchDcDataset(), batch_size=1, shuffle=False)
        model = FakeDcModel()
        report_horizons = resolve_report_horizons(
            time_slot_minutes=15,
            pred_horizon=4,
            requested_minutes=[15, 30, 60],
            strict=True,
        )

        metrics = evaluate_loader_raw_metrics(
            model,
            loader,
            train_task="dc",
            device=torch.device("cpu"),
            non_blocking=False,
            amp_enabled=False,
            amp_dtype=None,
            normalization_stats=None,
            time_slot_minutes=15,
            report_horizons=report_horizons,
        )

        demand_metrics = metrics["demand"]
        self.assertAlmostEqual(demand_metrics["mae"], 2.0, places=7)
        self.assertAlmostEqual(demand_metrics["rmse"], math.sqrt(4.5), places=7)
        self.assertAlmostEqual(demand_metrics["per_step"]["step_1"]["mae"], 1.0, places=7)
        self.assertAlmostEqual(demand_metrics["per_step"]["step_2"]["mae"], 2.0, places=7)
        self.assertAlmostEqual(demand_metrics["report"]["60min"]["rmse"], 2.0, places=7)

        supply_metrics = metrics["supply"]
        self.assertAlmostEqual(supply_metrics["mae"], 2.0, places=7)
        self.assertAlmostEqual(supply_metrics["per_step"]["step_3"]["mae"], 3.0, places=7)
        self.assertAlmostEqual(supply_metrics["report"]["30min"]["rmse"], 2.0, places=7)
        self.assertEqual(metrics["report_horizons"]["resolved_steps"], [1, 2, 4])

    def test_resolve_report_horizons_fails_when_60min_is_unavailable(self) -> None:
        with self.assertRaises(ValueError):
            resolve_report_horizons(
                time_slot_minutes=15,
                pred_horizon=3,
                requested_minutes=[15, 30, 60],
                strict=True,
            )


if __name__ == "__main__":
    unittest.main()
