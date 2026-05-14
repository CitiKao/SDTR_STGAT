import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_predictor import (  # noqa: E402
    build_time_contained_split_indices,
    build_time_meta_timestamps,
    filter_split_indices_by_time_mask,
    resolve_split_boundaries,
)


class TrainPredictorSplitPolicyTest(unittest.TestCase):
    def test_build_time_contained_split_indices_accepts_explicit_boundaries(self) -> None:
        splits = build_time_contained_split_indices(
            100,
            hist_len=12,
            pred_horizon=12,
            train_end=70,
            val_end=80,
        )
        self.assertTrue(all(idx + 24 <= 70 for idx in splits["train"]))
        self.assertTrue(all(idx >= 70 and idx + 24 <= 80 for idx in splits["val"]))
        self.assertTrue(all(idx >= 80 for idx in splits["test"]))

    def test_resolve_split_boundaries_none_alignment_uses_ratio_boundaries(self) -> None:
        time_meta = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=100, freq="15min").normalize(),
                "hour": pd.date_range("2024-01-01", periods=100, freq="15min").hour,
                "minute": pd.date_range("2024-01-01", periods=100, freq="15min").minute,
            }
        )
        train_end, val_end, summary = resolve_split_boundaries(time_meta, alignment="none")

        self.assertEqual(train_end, 70)
        self.assertEqual(val_end, 80)
        self.assertEqual(summary["mode"], "contiguous_ratio_70_10_20")
        self.assertEqual(summary["alignment"], "none")

    def test_build_time_meta_timestamps_combines_date_hour_and_minute(self) -> None:
        time_meta = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "hour": [0, 1],
                "minute": [15, 30],
            }
        )
        timestamps = build_time_meta_timestamps(time_meta)
        self.assertEqual(str(timestamps.iloc[0]), "2024-01-01 00:15:00")
        self.assertEqual(str(timestamps.iloc[1]), "2024-01-01 01:30:00")

    def test_filter_split_indices_drops_windows_touching_unobserved_slots(self) -> None:
        splits = {"train": [0, 1, 2, 3, 4], "val": [5], "test": [6]}
        observed = pd.Series([True, True, True, False, True, True, True, True, True, True]).to_numpy()

        filtered = filter_split_indices_by_time_mask(
            splits,
            observed,
            hist_len=2,
            pred_horizon=2,
        )

        self.assertEqual(filtered["train"], [4])
        self.assertEqual(filtered["val"], [5])
        self.assertEqual(filtered["test"], [6])


if __name__ == "__main__":
    unittest.main()
