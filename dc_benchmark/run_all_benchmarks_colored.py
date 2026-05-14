from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dc_benchmark.baselines import PAPER_METHODS, run_autoregressive, run_historical_average
from dc_benchmark.dataset import export_dc_benchmark, load_dc_benchmark
from dc_benchmark.neural import run_neural_paper_baseline
from dc_benchmark.run_paper_baseline import NEURAL_METHODS, _write_horizon_csv, _write_result
from dc_benchmark.summarize_paper_results import collect_rows, write_csv, write_markdown


ALL_METHODS = [
    "ha",
    "arima",
    "xgboost",
    "lstm",
    "convlstm",
    "st_resnet",
    "stgcn",
    "dcrnn",
    "graph_wavenet",
    "mlrnn_taxi_demand",
    "deep_multiconvlstm",
    "mt_mf_gcn",
]

RESULT_NAMES = {
    "ha": "HA_dc_metrics.json",
    "arima": "ARIMA_dc_metrics.json",
    "xgboost": "XGBoost_dc_metrics.json",
    "lstm": "LSTM_dc_metrics.json",
    "convlstm": "ConvLSTM_dc_metrics.json",
    "st_resnet": "ST_ResNet_dc_metrics.json",
    "stgcn": "STGCN_dc_metrics.json",
    "dcrnn": "DCRNN_dc_metrics.json",
    "graph_wavenet": "Graph_WaveNet_dc_metrics.json",
    "mlrnn_taxi_demand": "MLRNN_dc_metrics.json",
    "deep_multiconvlstm": "Deep_MultiConvLSTM_dc_metrics.json",
    "mt_mf_gcn": "MT_MF_GCN_dc_metrics.json",
}

RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
ORANGE = "\033[38;5;208m"
CYAN = "\033[96m"


def color(text: str, code: str, *, enabled: bool = True) -> str:
    return f"{code}{text}{RESET}" if enabled else text


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_horizon_rmse(metrics_or_reports: dict[str, Any] | None) -> str:
    if not metrics_or_reports:
        return ""
    reports = metrics_or_reports.get("horizon_reports") if "horizon_reports" in metrics_or_reports else metrics_or_reports
    if not isinstance(reports, dict):
        return ""
    parts: list[str] = []
    for label in ("15min", "30min", "60min"):
        row = reports.get(label)
        if not isinstance(row, dict):
            continue
        demand = row.get("demand", {})
        supply = row.get("supply", {})
        if not isinstance(demand, dict) or not isinstance(supply, dict):
            continue
        if "rmse" not in demand or "rmse" not in supply:
            continue
        raw_dc = float(row.get("raw_dc", float(demand["rmse"]) + float(supply["rmse"])))
        parts.append(
            f"{label}:D={float(demand['rmse']):.3f}/C={float(supply['rmse']):.3f}/DC={raw_dc:.3f}"
        )
    return " ".join(parts)


def parse_methods(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(ALL_METHODS)
    methods = [part.strip() for part in value.split(",") if part.strip()]
    unknown = [method for method in methods if method not in PAPER_METHODS]
    if unknown:
        raise SystemExit(f"Unknown benchmark methods: {', '.join(unknown)}")
    return methods


def manifest_matches(path: Path, args: argparse.Namespace) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "observed_time" not in manifest:
        return False
    report = manifest.get("report_horizons", {})
    requested = [int(x) for x in args.report_horizons_minutes.split(",") if x.strip()]
    return (
        int(manifest.get("hist_len", -1)) == int(args.hist_len)
        and int(manifest.get("pred_horizon", -1)) == int(args.pred_horizon)
        and str(manifest.get("time_feature_mode", "")) == str(args.time_feature_mode)
        and [int(x) for x in report.get("requested_minutes", [])] == requested
    )


def ensure_dataset(args: argparse.Namespace, *, colors: bool) -> Path:
    dataset_dir = Path(args.dataset_dir)
    if args.force_dataset or not manifest_matches(dataset_dir, args):
        print(
            color(
                "Export benchmark dataset "
                f"source={args.source_data_dir} output={dataset_dir} "
                f"hist_len={args.hist_len} pred_horizon={args.pred_horizon} "
                f"horizons={args.report_horizons_minutes}",
                RED,
                enabled=colors,
            ),
            flush=True,
        )
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
            force=True,
        )
    return dataset_dir


def add_manifest(result: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
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


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ordinal",
        "total",
        "method_id",
        "paper_name",
        "status",
        "raw_dc",
        "demand_rmse",
        "supply_rmse",
        "gap_rmse",
        "completed_epochs",
        "best_epoch",
        "selection_score",
        "result_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_final_markdown(path: Path, rows: list[dict[str, Any]], *, dataset_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# DC Benchmark Formal Training Summary",
        "",
        f"- Dataset: `{dataset_dir}`",
        f"- Epochs: {args.epochs}",
        f"- Early stop patience: {args.early_stop_patience}",
        f"- Hist/pred: {args.hist_len}/{args.pred_horizon}",
        "",
        "| Rank | Method | Paper | raw DC | D RMSE | C RMSE | Gap RMSE | Completed | Best Epoch |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ok_rows = [row for row in rows if row["status"] == "ok"]
    for rank, row in enumerate(sorted(ok_rows, key=lambda item: float(item["raw_dc"])), start=1):
        lines.append(
            "| {rank} | `{method_id}` | {paper_name} | {raw_dc:.6f} | {demand_rmse:.6f} | "
            "{supply_rmse:.6f} | {gap_rmse:.6f} | {completed_epochs} | {best_epoch} |".format(
                rank=rank,
                **row,
            )
        )
    failed = [row for row in rows if row["status"] != "ok"]
    if failed:
        lines.extend(["", "## Failed", ""])
        for row in failed:
            lines.append(f"- `{row['method_id']}`: {row['status']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_method(
    *,
    method_id: str,
    ordinal: int,
    total: int,
    benchmark: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    colors: bool,
    completed_method_seconds: list[float],
) -> dict[str, Any]:
    paper_name = PAPER_METHODS[method_id]["paper_name"]
    prefix = color(f"[{ordinal}/{total}]", YELLOW, enabled=colors)
    method_start = time.time()
    print(color("=" * 96, CYAN, enabled=colors), flush=True)
    print(f"{prefix} {paper_name} ({method_id})", flush=True)
    print(
        color(
            "PARAM "
            f"method={method_id} epochs={args.epochs} early_stop={args.early_stop_patience} "
            f"batch={args.batch_size} hidden={args.hidden_dim} lr={args.lr} device={args.device}",
            RED,
            enabled=colors,
        ),
        flush=True,
    )

    if method_id == "ha":
        print(f"{prefix} no-epoch baseline running...", flush=True)
        result = run_historical_average(benchmark, batch_size=args.batch_size)
    elif method_id in {"arima", "xgboost"}:
        print(f"{prefix} no-epoch autoregressive baseline running...", flush=True)
        result = run_autoregressive(
            benchmark,
            method_id=method_id,
            batch_size=args.batch_size,
            max_train_rows=args.max_train_rows,
        )
    elif method_id in NEURAL_METHODS:

        def progress(payload: dict[str, Any]) -> None:
            elapsed = time.time() - method_start
            epoch = int(payload["epoch"])
            epochs = int(payload["epochs"])
            epoch_sec = elapsed / max(epoch, 1)
            eta_method = epoch_sec * max(epochs - epoch, 0)
            estimated_full_method = epoch_sec * epochs
            completed_avg = (
                sum(completed_method_seconds) / len(completed_method_seconds)
                if completed_method_seconds
                else estimated_full_method
            )
            eta_total = eta_method + max(total - ordinal, 0) * completed_avg
            improved = bool(payload["improved"])
            best_text = f"best_val={payload['best_val_raw_dc']:.6f}"
            best_epoch = f"best_epoch={int(payload['best_epoch'])}"
            line = (
                f"{prefix} {paper_name} "
                f"Ep {epoch:03d}/{epochs:03d} "
                f"train_loss={float(payload['train_loss']):.6f} "
                f"val_raw_dc={float(payload['val_raw_dc']):.6f} "
                f"{color(best_text, GREEN, enabled=colors)} "
                f"{color(best_epoch, ORANGE, enabled=colors)} "
                f"wait={int(payload['epochs_since_improvement'])}/{int(payload['patience'])} "
                f"epoch={epoch_sec:.1f}s ETA={format_duration(eta_method)} "
                f"TOTAL_ETA={format_duration(eta_total)}"
            )
            horizon_text = format_horizon_rmse(payload.get("horizon_reports") or payload.get("val_metrics"))
            if horizon_text:
                line += " " + color(f"H[{horizon_text}]", CYAN, enabled=colors)
            if improved:
                line += " " + color("NEW BEST", GREEN, enabled=colors)
            print(line, flush=True)

        result = run_neural_paper_baseline(
            benchmark,
            method_id=method_id,
            epochs=args.epochs,
            early_stop_patience=args.early_stop_patience,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            device=args.device,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            checkpoint_path=(
                Path(args.checkpoint_dir) / f"{method_id}_best.pt"
                if args.checkpoint_dir
                else output_dir / "checkpoints" / f"{method_id}_best.pt"
            ),
            progress_callback=progress,
        )
    else:
        raise ValueError(f"Unsupported method: {method_id}")

    add_manifest(result, benchmark)
    metrics = result["metrics"]
    run_config = result.get("run_config", {})
    print(
        f"{prefix} "
        + color(
            f"DONE raw_dc={metrics['raw_dc']:.6f} "
            f"D_RMSE={metrics['demand']['rmse']:.6f} "
            f"C_RMSE={metrics['supply']['rmse']:.6f} "
            f"Gap_RMSE={metrics['gap']['rmse']:.6f}",
            GREEN,
            enabled=colors,
        )
        + " "
        + color(
            f"best_epoch={run_config.get('best_epoch', '')}",
            ORANGE,
            enabled=colors,
        ),
        flush=True,
    )
    horizon_text = format_horizon_rmse(metrics)
    if horizon_text:
        print(f"{prefix} " + color(f"HORIZON_RMSE {horizon_text}", CYAN, enabled=colors), flush=True)
    print(f"{prefix} elapsed={format_duration(time.time() - method_start)}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all DC benchmark baselines with colored live progress.")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--source-data-dir", default="data")
    parser.add_argument("--dataset-dir", default="data/dc_benchmark_h16")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--hist-len", type=int, default=16)
    parser.add_argument("--pred-horizon", type=int, default=4)
    parser.add_argument("--report-horizons-minutes", default="15,30,60")
    parser.add_argument("--split-policy", default="project_monthly", choices=["project_monthly", "benchmark_contiguous"])
    parser.add_argument("--split-alignment", default="none", choices=["none", "day", "week", "month"])
    parser.add_argument("--time-feature-mode", default="baseline")
    parser.add_argument("--max-time-steps", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--ordinal-offset", type=int, default=0)
    parser.add_argument("--total-override", type=int, default=0)
    parser.add_argument("--no-color", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    colors = not args.no_color
    methods = parse_methods(args.methods)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"runs/dc_benchmark_formal_h16_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = ensure_dataset(args, colors=colors)
    benchmark = load_dc_benchmark(dataset_dir)
    manifest = benchmark["manifest"]
    print(
        color(
            f"DATASET {dataset_dir} hist_len={manifest['hist_len']} pred_horizon={manifest['pred_horizon']} "
            f"split={manifest['split_counts']} split_hash={manifest['split_hash']}",
            CYAN,
            enabled=colors,
        ),
        flush=True,
    )
    print(color(f"OUTPUT {output_dir}", CYAN, enabled=colors), flush=True)

    summary_rows: list[dict[str, Any]] = []
    total = int(args.total_override) if int(args.total_override) > 0 else len(methods)
    completed_method_seconds: list[float] = []
    for local_ordinal, method_id in enumerate(methods, start=1):
        ordinal = int(args.ordinal_offset) + local_ordinal
        result_name = RESULT_NAMES[method_id]
        method_wall_start = time.time()
        try:
            result = run_method(
                method_id=method_id,
                ordinal=ordinal,
                total=total,
                benchmark=benchmark,
                args=args,
                output_dir=output_dir,
                colors=colors,
                completed_method_seconds=completed_method_seconds,
            )
            result_path = output_dir / result_name
            _write_result(result_path, result)
            _write_horizon_csv(
                result_path.with_name(result_path.stem.replace("_dc_metrics", "") + "_horizon_metrics.csv"),
                result,
                result_json_name=result_path.name,
            )
            metrics = result["metrics"]
            run_config = result.get("run_config", {})
            row = {
                "ordinal": ordinal,
                "total": total,
                "method_id": method_id,
                "paper_name": result["paper_name"],
                "status": "ok",
                "raw_dc": float(metrics["raw_dc"]),
                "demand_rmse": float(metrics["demand"]["rmse"]),
                "supply_rmse": float(metrics["supply"]["rmse"]),
                "gap_rmse": float(metrics["gap"]["rmse"]),
                "completed_epochs": int(run_config.get("completed_epochs") or 0),
                "best_epoch": int(run_config.get("best_epoch") or 0),
                "selection_score": (
                    float(run_config["selection_score"])
                    if run_config.get("selection_score") is not None
                    else ""
                ),
                "result_json": result_path.name,
            }
        except Exception as exc:
            print(color(f"[{ordinal}/{total}] FAILED {method_id}: {exc}", RED, enabled=colors), flush=True)
            row = {
                "ordinal": ordinal,
                "total": total,
                "method_id": method_id,
                "paper_name": PAPER_METHODS[method_id]["paper_name"],
                "status": f"failed: {exc}",
                "raw_dc": float("inf"),
                "demand_rmse": float("inf"),
                "supply_rmse": float("inf"),
                "gap_rmse": float("inf"),
                "completed_epochs": 0,
                "best_epoch": 0,
                "selection_score": "",
                "result_json": RESULT_NAMES[method_id],
            }
        completed_method_seconds.append(time.time() - method_wall_start)
        summary_rows.append(row)
        write_summary(output_dir / "summary.tsv", summary_rows)
        write_final_markdown(output_dir / "final_summary.md", summary_rows, dataset_dir=dataset_dir, args=args)

    rows = collect_rows(output_dir)
    write_csv(output_dir / "dc_benchmark_horizon_metrics.csv", rows)
    write_markdown(output_dir / "dc_benchmark_horizon_metrics.md", rows)
    print(color("=" * 96, CYAN, enabled=colors), flush=True)
    print(color(f"ALL DONE output={output_dir}", GREEN, enabled=colors), flush=True)


if __name__ == "__main__":
    main()
