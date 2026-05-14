import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predictor_normalization import (  # noqa: E402
    build_normalization_stats,
    impute_missing_speed_history,
    normalize_speed_features,
)


class PredictorNormalizationTest(unittest.TestCase):
    def test_speed_stats_ignore_masked_zeros_and_fill_invalid_with_zero_after_normalization(self) -> None:
        node_features = np.zeros((4, 2, 2), dtype=np.float32)
        edge_speeds = np.array(
            [
                [10.0, 0.0],
                [12.0, 20.0],
                [14.0, 22.0],
                [16.0, 24.0],
            ],
            dtype=np.float32,
        )
        speed_valid_mask = np.array(
            [
                [True, False],
                [True, True],
                [True, True],
                [True, True],
            ],
            dtype=bool,
        )
        train_time_mask = np.array([True, True, True, False], dtype=bool)

        stats = build_normalization_stats(
            node_features,
            edge_speeds,
            train_time_mask,
            speed_valid_mask=speed_valid_mask,
        )

        self.assertTrue(np.allclose(stats["speed"]["mean"], np.array([12.0, 21.0], dtype=np.float32)))

        normalized = normalize_speed_features(
            edge_speeds,
            stats,
            edge_axis=1,
            speed_valid_mask=speed_valid_mask,
        )
        self.assertAlmostEqual(float(normalized[0, 1]), 0.0, places=7)
        self.assertAlmostEqual(float(normalized[1, 0]), 0.0, places=7)

    def test_impute_missing_speed_history_forward_fills_with_zero_fallback(self) -> None:
        speed_values = np.array(
            [
                [0.0, 1.0],
                [2.0, 0.0],
                [0.0, 3.0],
            ],
            dtype=np.float32,
        )
        speed_valid_mask = np.array(
            [
                [False, True],
                [True, False],
                [False, True],
            ],
            dtype=bool,
        )

        imputed = impute_missing_speed_history(
            speed_values,
            speed_valid_mask=speed_valid_mask,
            time_axis=0,
            fill_value=0.0,
        )

        expected = np.array(
            [
                [0.0, 1.0],
                [2.0, 1.0],
                [2.0, 3.0],
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(imputed, expected))


if __name__ == "__main__":
    unittest.main()
