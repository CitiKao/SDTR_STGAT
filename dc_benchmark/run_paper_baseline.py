from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dc_benchmark.baselines import PAPER_METHODS, run_autoregressive, run_historical_average
    from dc_benchmark.dataset import DEFAULT_DATASET_DIR, export_dc_benchmark, load_dc_benchmark
    from dc_benchmark.metrics import build_horizon_metric_table
    from dc_benchmark.neural import run_neural_paper_baseline
else:
    from .baselines import PAPER_METHODS, run_autoregressive, run_historical_average
    from .dataset import DEFAULT_DATASET_DIR, export_dc_benchmark, load_dc_benchmark
    from .metrics import build_horizon_metric_table
    from .neural import run_neural_paper_baseline


NEURAL_METHODS = {
    "lstm",
    "convlstm",
    "st_resnet",
    "stgcn",
    "dcrnn",
    "graph_wavenet",
    "mlrnn_taxi_demand",
    "deep_multiconvlstm",
    "mt_mf_gcn",
}


def _write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_horizon_csv(path: Path, result: dict, *, result_json_name: str) -> None:
    rows = result["metrics"].get("horizon_table") or build_horizon_metric_table(result["metrics"])
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_name",
        "method_id",
        "target",
        "horizon",
        "minutes",
        "step",
        "mae",
        "rmse",
        "mape",
        "count",
        "raw_dc_horizon",
        "result_json",
        "split_hash",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "paper_name": result["paper_name"],
                    "method_id": result["method_id"],
                    "target": row["target"],
                    "horizon": row["horizon"],
                    "minutes": row["minutes"],
                    "step": row["step"],
                    "mae": row["mae"],
                    "rmse": row["rmse"],
                    "mape": row["mape"],
                    "count": row["count"],
                    "raw_dc_horizon": row["raw_dc_horizon"],
                    "result_json": result_json_name,
                    "split_hash": result["benchmark_manifest"]["split_hash"],
                }
            )


def _print_horizon_report(result: dict) -> None:
    reports = result["metrics"].get("horizon_reports", {})
    for horizon, report in reports.items():
        demand = report.get("demand", {})
        supply = report.get("supply", {})
        gap = report.get("gap", {})
        print(
            f"{result['paper_name']} | {horizon} "
            f"D_MAE={demand.get('mae', float('nan')):.6f} "
            f"D_RMSE={demand.get('rmse', float('nan')):.6f} "
            f"D_MAPE={demand.get('mape', float('nan')):.6f}% "
            f"C_MAE={supply.get('mae', float('nan')):.6f} "
            f"C_RMSE={supply.get('rmse', float('nan')):.6f} "
            f"C_MAPE={supply.get('mape', float('nan')):.6f}% "
            f"gap_MAE={gap.get('mae', float('nan')):.6f} "
            f"gap_RMSE={gap.get('rmse', float('nan')):.6f} "
            f"gap_MAPE={gap.get('mape', float('nan')):.6f}%"
        )


def run_method(args: argparse.Namespace) -> dict:
    dataset_dir = Path(args.dataset_dir)
    if args.auto_export and not (dataset_dir / "manifest.json").exists():
        export_dc_benchmark(
            source_data_dir=args.source_data_dir,
            output_dir=dataset_dir,
            hist_len=args.hist_len,
            pred_horizon=args.pred_horizon,
            report_horizons_minutes=args.report_horizons_minutes,
            split_policy=args.split_policy,
            split_alignment=args.split_alignment,
            time_feature_mode=args.time_feature_mode,
            max_time_steps=args.max_time_steps,
        )
    benchmark = load_dc_benchmark(dataset_dir)

    if args.method == "ha":
        result = run_historical_average(benchmark, batch_size=args.batch_size)
    elif args.method in {"arima", "xgboost"}:
        result = run_autoregressive(
            benchmark,
            method_id=args.method,
            batch_size=args.batch_size,
            max_train_rows=args.max_train_rows,
        )
    elif args.method in NEURAL_METHODS:
        result = run_neural_paper_baseline(
            benchmark,
            method_id=args.method,
            epochs=args.epochs,
            early_stop_patience=args.early_stop_patience,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            device=args.device,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
        )
    else:
        raise ValueError(f"Unsupported method {args.method!r}.")

    result["created_at"] = datetime.now().isoformat()
    result["benchmark_manifest"] = {
        "dataset_name": benchmark["manifest"]["dataset_name"],
        "schema_version": benchmark["manifest"]["schema_version"],
        "split_hash": benchmark["manifest"]["split_hash"],
        "split_counts": benchmark["manifest"]["split_counts"],
        "hist_len": benchmark["manifest"]["hist_len"],
        "pred_horizon": benchmark["manifest"]["pred_horizon"],
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a paper-named DC benchmark baseline.")
    parser.add_argument("--method", default="", choices=sorted(PAPER_METHODS))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--source-data-dir", default="data")
    parser.add_argument("--output-dir", default="runs/dc_benchmark")
    parser.add_argument("--auto-export", action="store_true")
    parser.add_argument("--hist-len", type=int, default=14)
    parser.add_argument("--pred-horizon", type=int, default=4)
    parser.add_argument("--report-horizons-minutes", default="15,30,60")
    parser.add_argument("--split-policy", default="project_monthly", choices=["project_monthly", "benchmark_contiguous"])
    parser.add_argument("--split-alignment", default="none", choices=["none", "day", "week", "month"])
    parser.add_argument("--time-feature-mode", default="baseline")
    parser.add_argument("--max-time-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--result-name", default="")
    parser.add_argument("--no-horizon-csv", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.method:
        parser.error("--method is required when using run_paper_baseline.py directly.")
    result = run_method(args)
    name = args.result_name or f"{args.method}_dc_metrics.json"
    output_path = Path(args.output_dir) / name
    _write_result(output_path, result)
    csv_path = output_path.with_name(output_path.stem.replace("_dc_metrics", "") + "_horizon_metrics.csv")
    if not args.no_horizon_csv:
        _write_horizon_csv(csv_path, result, result_json_name=output_path.name)
    metrics = result["metrics"]
    print(
        f"{result['paper_name']} | raw_dc={metrics['raw_dc']:.6f} "
        f"D_RMSE={metrics['demand']['rmse']:.6f} C_RMSE={metrics['supply']['rmse']:.6f}"
    )
    _print_horizon_report(result)
    print(f"wrote {output_path}")
    if not args.no_horizon_csv:
        print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
