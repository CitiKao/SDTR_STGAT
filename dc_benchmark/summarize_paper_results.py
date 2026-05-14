from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .metrics import build_horizon_metric_table


DEFAULT_REQUIRED_HORIZONS = ("15min", "30min", "60min")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_files(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def rows_from_result(path: Path) -> list[dict[str, Any]]:
    result = _read_json(path)
    metrics = result["metrics"]
    table = metrics.get("horizon_table") or build_horizon_metric_table(metrics)
    manifest = result.get("benchmark_manifest", {})
    rows: list[dict[str, Any]] = []
    for row in table:
        rows.append(
            {
                "paper_name": result.get("paper_name", ""),
                "method_id": result.get("method_id", ""),
                "implementation_scope": result.get("implementation_scope", ""),
                "claim_type": result.get("claim_type", ""),
                "official_source_verified": result.get("official_source_verified", False),
                "target": row["target"],
                "horizon": row["horizon"],
                "minutes": int(row["minutes"]),
                "step": int(row["step"]),
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "mape": float(row["mape"]),
                "count": float(row["count"]),
                "raw_dc_horizon": row.get("raw_dc_horizon"),
                "raw_dc_overall": metrics.get("raw_dc"),
                "result_json": path.name,
                "split_hash": manifest.get("split_hash", ""),
                "hist_len": manifest.get("hist_len", ""),
                "pred_horizon": manifest.get("pred_horizon", ""),
                "created_at": result.get("created_at", ""),
            }
        )
    return rows


def collect_rows(input_dir: Path, *, pattern: str = "*_dc_metrics.json") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _result_files(input_dir, pattern):
        rows.extend(rows_from_result(path))
    return rows


def validate_required_horizons(
    rows: list[dict[str, Any]],
    *,
    required_horizons: tuple[str, ...] = DEFAULT_REQUIRED_HORIZONS,
) -> list[str]:
    missing: list[str] = []
    by_method: dict[str, set[str]] = {}
    for row in rows:
        by_method.setdefault(str(row["method_id"]), set()).add(str(row["horizon"]))
    for method_id, horizons in sorted(by_method.items()):
        for horizon in required_horizons:
            if horizon not in horizons:
                missing.append(f"{method_id}:{horizon}")
    return missing


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "paper_name",
        "method_id",
        "implementation_scope",
        "claim_type",
        "official_source_verified",
        "target",
        "horizon",
        "minutes",
        "step",
        "mae",
        "rmse",
        "mape",
        "count",
        "raw_dc_horizon",
        "raw_dc_overall",
        "result_json",
        "split_hash",
        "hist_len",
        "pred_horizon",
        "created_at",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DC Benchmark Horizon Metrics",
        "",
        "All rows are raw-scale prediction-ahead metrics. For a 15-minute slot dataset, `30min` means step 2 ahead, not a 30-minute aggregate bin.",
        "",
        "| Paper | Method | Target | Horizon | Step | MAE | RMSE | MAPE (%) | raw_dc@horizon | Result |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        raw_dc = row["raw_dc_horizon"]
        raw_dc_text = "" if raw_dc is None else f"{float(raw_dc):.6f}"
        lines.append(
            "| {paper_name} | `{method_id}` | {target} | {horizon} | {step} | {mae:.6f} | {rmse:.6f} | {mape:.6f} | {raw_dc} | `{result_json}` |".format(
                paper_name=row["paper_name"],
                method_id=row["method_id"],
                target=row["target"],
                horizon=row["horizon"],
                step=int(row["step"]),
                mae=float(row["mae"]),
                rmse=float(row["rmse"]),
                mape=float(row["mape"]),
                raw_dc=raw_dc_text,
                result_json=row["result_json"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize DC paper benchmark horizon metrics.")
    parser.add_argument("--input-dir", default="runs/dc_benchmark")
    parser.add_argument("--pattern", default="*_dc_metrics.json")
    parser.add_argument("--output-csv", default="runs/dc_benchmark/dc_benchmark_horizon_metrics.csv")
    parser.add_argument("--output-md", default="runs/dc_benchmark/dc_benchmark_horizon_metrics.md")
    parser.add_argument("--required-horizons", default="15min,30min,60min")
    parser.add_argument("--allow-missing-horizons", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = collect_rows(Path(args.input_dir), pattern=args.pattern)
    if not rows:
        raise SystemExit(f"No result files matched {args.pattern!r} in {args.input_dir!r}.")
    required = tuple(part.strip() for part in args.required_horizons.split(",") if part.strip())
    missing = validate_required_horizons(rows, required_horizons=required)
    if missing and not args.allow_missing_horizons:
        raise SystemExit("Missing required horizon rows: " + ", ".join(missing))
    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
