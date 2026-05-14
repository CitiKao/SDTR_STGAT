from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dc_benchmark.dataset import DEFAULT_DATASET_DIR, export_dc_benchmark
else:
    from .dataset import DEFAULT_DATASET_DIR, export_dc_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the unified STDR D/C benchmark dataset.")
    parser.add_argument("--source-data-dir", default="data")
    parser.add_argument("--output-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--hist-len", type=int, default=14)
    parser.add_argument("--pred-horizon", type=int, default=4)
    parser.add_argument("--report-horizons-minutes", default="15,30,60")
    parser.add_argument("--split-policy", default="project_monthly", choices=["project_monthly", "benchmark_contiguous"])
    parser.add_argument("--split-alignment", default="none", choices=["none", "day", "week", "month"])
    parser.add_argument("--time-feature-mode", default="baseline")
    parser.add_argument("--max-time-steps", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = export_dc_benchmark(
        source_data_dir=args.source_data_dir,
        output_dir=args.output_dir,
        hist_len=args.hist_len,
        pred_horizon=args.pred_horizon,
        report_horizons_minutes=args.report_horizons_minutes,
        split_policy=args.split_policy,
        split_alignment=args.split_alignment,
        time_feature_mode=args.time_feature_mode,
        max_time_steps=args.max_time_steps,
        force=args.force,
    )
    print(f"DC benchmark exported to {output}")


if __name__ == "__main__":
    main()
