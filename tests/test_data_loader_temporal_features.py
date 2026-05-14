import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import (  # noqa: E402
    NYC_TIME_FEATURE_MODE_CHOICES,
    build_temporal_features_from_time_meta,
    resolve_nyc_time_feature_mode,
)


class DataLoaderTemporalFeaturesTest(unittest.TestCase):
    def setUp(self) -> None:
        # All rows share the same month / weekday / slot but differ in month position.
        self.time_meta = pd.DataFrame(
            {
                "date": [
                    "2020-01-06",
                    "2020-01-13",
                    "2020-01-20",
                    "2020-01-27",
                ],
                "day_of_week": [0, 0, 0, 0],
                "slot": [0, 0, 0, 0],
            }
        )
        self.days_in_month = np.array([31.0, 31.0, 31.0, 31.0], dtype=np.float32)

    def test_baseline_mode_preserves_original_six_features(self) -> None:
        features, names = build_temporal_features_from_time_meta(
            self.time_meta,
            time_feature_mode="baseline",
        )
        self.assertEqual(features.shape, (4, 6))
        self.assertEqual(
            names,
            [
                "month_sin",
                "month_cos",
                "weekday_sin",
                "weekday_cos",
                "slot_sin",
                "slot_cos",
            ],
        )
        self.assertTrue(np.allclose(features[0], features[1]))
        self.assertTrue(np.allclose(features[1], features[2]))

    def test_day_of_month_mode_adds_month_position_signal(self) -> None:
        features, names = build_temporal_features_from_time_meta(
            self.time_meta,
            time_feature_mode="day_of_month",
        )
        self.assertEqual(features.shape, (4, 8))
        self.assertEqual(names[-2:], ["day_of_month_sin", "day_of_month_cos"])
        self.assertFalse(np.allclose(features[0], features[1]))
        self.assertFalse(np.allclose(features[1], features[2]))

    def test_week_of_month_mode_adds_week_position_signal(self) -> None:
        features, names = build_temporal_features_from_time_meta(
            self.time_meta,
            time_feature_mode="week_of_month",
        )
        self.assertEqual(features.shape, (4, 8))
        self.assertEqual(names[-2:], ["week_of_month_sin", "week_of_month_cos"])
        expected_week = np.floor((np.array([6.0, 13.0, 20.0, 27.0], dtype=np.float32) - 1.0) / 7.0)
        expected_weeks_in_month = np.floor((self.days_in_month - 1.0) / 7.0) + 1.0
        expected_angle = 2.0 * np.pi * expected_week / expected_weeks_in_month
        self.assertTrue(np.allclose(features[:, 6], np.sin(expected_angle)))
        self.assertTrue(np.allclose(features[:, 7], np.cos(expected_angle)))

    def test_combined_mode_adds_both_month_position_signals(self) -> None:
        features, names = build_temporal_features_from_time_meta(
            self.time_meta,
            time_feature_mode="day_of_month_and_week_of_month",
        )
        self.assertEqual(features.shape, (4, 10))
        self.assertEqual(
            names[-4:],
            [
                "day_of_month_sin",
                "day_of_month_cos",
                "week_of_month_sin",
                "week_of_month_cos",
            ],
        )
        baseline_features, _ = build_temporal_features_from_time_meta(
            self.time_meta,
            time_feature_mode="baseline",
        )
        dom_features, _ = build_temporal_features_from_time_meta(
            self.time_meta,
            time_feature_mode="day_of_month",
        )
        wom_features, _ = build_temporal_features_from_time_meta(
            self.time_meta,
            time_feature_mode="week_of_month",
        )
        self.assertTrue(np.allclose(features[:, :6], baseline_features))
        self.assertTrue(np.allclose(features[:, 6:8], dom_features[:, 6:8]))
        self.assertTrue(np.allclose(features[:, 8:10], wom_features[:, 6:8]))

    def test_week_of_month_mode_handles_month_length_and_final_partial_week(self) -> None:
        time_meta = pd.DataFrame(
            {
                "date": [
                    "2021-02-22",
                    "2021-02-28",
                    "2020-01-31",
                ],
                "day_of_week": [0, 6, 4],
                "slot": [0, 0, 0],
            }
        )
        features, names = build_temporal_features_from_time_meta(
            time_meta,
            time_feature_mode="week_of_month",
        )
        self.assertEqual(names[-2:], ["week_of_month_sin", "week_of_month_cos"])

        day_of_month = np.array([22.0, 28.0, 31.0], dtype=np.float32)
        days_in_month = np.array([28.0, 28.0, 31.0], dtype=np.float32)
        expected_week = np.floor((day_of_month - 1.0) / 7.0)
        expected_weeks_in_month = np.floor((days_in_month - 1.0) / 7.0) + 1.0
        expected_angle = 2.0 * np.pi * expected_week / expected_weeks_in_month

        self.assertTrue(np.allclose(features[:, 6], np.sin(expected_angle)))
        self.assertTrue(np.allclose(features[:, 7], np.cos(expected_angle)))

    def test_invalid_time_feature_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_nyc_time_feature_mode("not_a_mode")
        self.assertIn("baseline", NYC_TIME_FEATURE_MODE_CHOICES)


if __name__ == "__main__":
    unittest.main()
