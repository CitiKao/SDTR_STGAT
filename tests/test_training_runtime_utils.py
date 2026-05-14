import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_runtime_utils import (  # noqa: E402
    build_optimizer,
    masked_regression_loss,
    maybe_apply_linear_warmup,
)


class TrainingRuntimeUtilsTest(unittest.TestCase):
    def test_build_optimizer_supports_adamw(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        optimizer = build_optimizer(
            [param],
            optimizer_name="adamw",
            lr=1e-3,
            weight_decay=1e-2,
        )
        self.assertIsInstance(optimizer, torch.optim.AdamW)

    def test_masked_regression_loss_supports_multiple_loss_types(self) -> None:
        prediction = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        target = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        mse_loss = masked_regression_loss(
            prediction,
            target,
            loss_name="mse",
            target_mask=mask,
        )
        rmse_loss = masked_regression_loss(
            prediction,
            target,
            loss_name="rmse",
            target_mask=mask,
        )
        huber_loss = masked_regression_loss(
            prediction,
            target,
            loss_name="huber",
            target_mask=mask,
            huber_delta=1.0,
        )
        charbonnier_loss = masked_regression_loss(
            prediction,
            target,
            loss_name="charbonnier",
            target_mask=mask,
            charbonnier_eps=1e-3,
        )

        self.assertAlmostEqual(float(mse_loss), 4.0, places=6)
        self.assertAlmostEqual(float(rmse_loss), 2.0, places=6)
        self.assertAlmostEqual(float(huber_loss), 1.5, places=6)
        self.assertGreater(float(charbonnier_loss), 1.99)
        self.assertLess(float(charbonnier_loss), 2.01)

    def test_maybe_apply_linear_warmup_updates_lr(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        optimizer = build_optimizer(
            [param],
            optimizer_name="adam",
            lr=1e-3,
            weight_decay=0.0,
        )
        warmup_lr = maybe_apply_linear_warmup(
            optimizer,
            base_lr=1e-3,
            global_step=2,
            warmup_steps=4,
        )

        self.assertAlmostEqual(warmup_lr, 5e-4, places=9)
        self.assertAlmostEqual(float(optimizer.param_groups[0]["lr"]), 5e-4, places=9)


if __name__ == "__main__":
    unittest.main()
