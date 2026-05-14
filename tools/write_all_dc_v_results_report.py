from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "runs" / "final_dc_v_export_20260514_134425"
REPORT_PATH = EXPORT_DIR / "ALL_DC_V_RESULTS.md"
MANIFEST_PATH = EXPORT_DIR / "checkpoint_manifest.json"

DATASET_ORDER = ["Shanghai", "NYC", "Shenzhen", "Chengdu"]
HORIZONS = ["15min", "30min", "60min"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def as_float(value: Any) -> float | None:
    if value in (None, "", "n/a"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 6) -> str:
    number = as_float(value)
    if number is None or not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def fmt_int(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    return str(int(number))


def md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(item) for item in row) + " |")
    return lines


@dataclass
class DcRun:
    dataset: str
    method_id: str
    method_name: str
    family: str
    metrics_path: Path
    source_label: str
    completed_epochs: str = "n/a"
    best_epoch: str = "n/a"


def dc_json_dir_for_nyc(row: dict[str, str]) -> Path:
    source_run = row.get("source_run", "")
    if source_run.startswith("top2"):
        return ROOT / "runs" / "dc_benchmark_top2_h16_ep150_noearlystop_20260506_230958"
    return ROOT / "runs" / "dc_benchmark_formal_h16_20260506_184748"


def load_dc_baselines() -> list[DcRun]:
    runs: list[DcRun] = []

    standard_specs = [
        (
            "Chengdu",
            ROOT / "runs" / "chengdu_dc_full_20260513_093044" / "benchmarks",
            "chengdu_full100",
        ),
        (
            "Shenzhen",
            ROOT / "runs" / "shenzhen_dc_full_20260508_002254" / "benchmarks",
            "shenzhen_full100",
        ),
    ]
    for dataset, base_dir, source_label in standard_specs:
        for row in read_tsv(base_dir / "summary.tsv"):
            if row.get("status") != "ok":
                continue
            runs.append(
                DcRun(
                    dataset=dataset,
                    method_id=row["method_id"],
                    method_name=row["paper_name"],
                    family="benchmark",
                    metrics_path=base_dir / row["result_json"],
                    source_label=source_label,
                    completed_epochs=fmt_int(row.get("completed_epochs")),
                    best_epoch=fmt_int(row.get("best_epoch")),
                )
            )

    nyc_summary = ROOT / "runs" / "dc_benchmark_current_best" / "summary.tsv"
    for row in read_tsv(nyc_summary):
        if row.get("status") != "ok":
            continue
        json_dir = dc_json_dir_for_nyc(row)
        runs.append(
            DcRun(
                dataset="NYC",
                method_id=row["method_id"],
                method_name=row["paper_name"],
                family="benchmark",
                metrics_path=json_dir / row["result_json"],
                source_label=row.get("source_run", "nyc_current_best"),
                completed_epochs=fmt_int(row.get("completed_epochs")),
                best_epoch=fmt_int(row.get("best_epoch")),
            )
        )

    shanghai_candidates: dict[str, tuple[float, DcRun]] = {}
    shanghai_specs = [
        (
            ROOT / "runs" / "shanghai_dc_full_20260514_055501" / "benchmarks",
            "shanghai_full100",
        ),
        (
            ROOT / "runs" / "shanghai_benchmarks_selected_ep150_20260514_073131",
            "shanghai_selected150",
        ),
    ]
    for base_dir, source_label in shanghai_specs:
        for row in read_tsv(base_dir / "summary.tsv"):
            if row.get("status") != "ok":
                continue
            raw_dc = as_float(row.get("raw_dc"))
            if raw_dc is None:
                continue
            candidate = DcRun(
                dataset="Shanghai",
                method_id=row["method_id"],
                method_name=row["paper_name"],
                family="benchmark",
                metrics_path=base_dir / row["result_json"],
                source_label=source_label,
                completed_epochs=fmt_int(row.get("completed_epochs")),
                best_epoch=fmt_int(row.get("best_epoch")),
            )
            current = shanghai_candidates.get(row["method_id"])
            if current is None or raw_dc < current[0]:
                shanghai_candidates[row["method_id"]] = (raw_dc, candidate)
    runs.extend(candidate for _, candidate in shanghai_candidates.values())
    return runs


def load_stgat_dc_runs() -> list[DcRun]:
    specs = [
        ("NYC", "dc_nyc"),
        ("Shenzhen", "dc_shenzhen"),
        ("Chengdu", "dc_chengdu"),
        ("Shanghai", "dc_shanghai"),
    ]
    return [
        DcRun(
            dataset=dataset,
            method_id="stgat",
            method_name="STGAT (ours)",
            family="ours",
            metrics_path=EXPORT_DIR / "checkpoints" / export_id / "predictor_test_metrics.json",
            source_label=export_id,
        )
        for dataset, export_id in specs
    ]


def dc_metric_block(run: DcRun) -> dict[str, Any]:
    payload = load_json(run.metrics_path)
    if "metrics" in payload:
        metrics = payload["metrics"]
        targets = {
            "demand": metrics.get("demand", {}),
            "supply": metrics.get("supply", {}),
            "gap": metrics.get("gap", {}),
        }
        report_key = "report"
    else:
        targets = {
            "demand": payload.get("raw_metrics", {}).get("demand", {}),
            "supply": payload.get("raw_metrics", {}).get("supply", {}),
            "gap": {},
        }
        per_horizon = payload.get("raw_metrics_report", {})
        targets["demand"]["report"] = per_horizon.get("demand", {})
        targets["supply"]["report"] = per_horizon.get("supply", {})
        targets["gap"]["report"] = {}
        report_key = "report"
    return {"targets": targets, "report_key": report_key}


def avg(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def dc_overall_rows(dc_runs: list[DcRun]) -> list[list[str]]:
    collected: list[dict[str, Any]] = []
    for run in dc_runs:
        block = dc_metric_block(run)
        d = block["targets"]["demand"]
        c = block["targets"]["supply"]
        g = block["targets"]["gap"]
        d_rmse = as_float(d.get("rmse"))
        c_rmse = as_float(c.get("rmse"))
        d_mse = as_float(d.get("mse"))
        c_mse = as_float(c.get("mse"))
        dc_rmse = None if d_rmse is None or c_rmse is None else d_rmse + c_rmse
        dc_mse = None if d_mse is None or c_mse is None else d_mse + c_mse
        collected.append(
            {
                "dataset": run.dataset,
                "method": run.method_name,
                "family": run.family,
                "d_mse": d_mse,
                "d_rmse": d_rmse,
                "d_mape": as_float(d.get("mape")),
                "c_mse": c_mse,
                "c_rmse": c_rmse,
                "c_mape": as_float(c.get("mape")),
                "gap_mse": as_float(g.get("mse")),
                "gap_rmse": as_float(g.get("rmse")),
                "gap_mape": as_float(g.get("mape")),
                "dc_mse": dc_mse,
                "dc_rmse": dc_rmse,
                "dc_mape": avg([as_float(d.get("mape")), as_float(c.get("mape"))]),
                "epochs": run.completed_epochs,
                "best_epoch": run.best_epoch,
                "source": run.source_label,
            }
        )

    rows: list[list[str]] = []
    for dataset in DATASET_ORDER:
        dataset_rows = [r for r in collected if r["dataset"] == dataset]
        dataset_rows.sort(key=lambda r: (float("inf") if r["dc_rmse"] is None else r["dc_rmse"], r["method"]))
        for rank, row in enumerate(dataset_rows, 1):
            rows.append(
                [
                    dataset,
                    str(rank),
                    row["method"],
                    row["family"],
                    fmt(row["d_mse"]),
                    fmt(row["d_rmse"]),
                    fmt(row["d_mape"]),
                    fmt(row["c_mse"]),
                    fmt(row["c_rmse"]),
                    fmt(row["c_mape"]),
                    fmt(row["gap_mse"]),
                    fmt(row["gap_rmse"]),
                    fmt(row["gap_mape"]),
                    fmt(row["dc_mse"]),
                    fmt(row["dc_rmse"]),
                    fmt(row["dc_mape"]),
                    row["epochs"],
                    row["best_epoch"],
                    row["source"],
                ]
            )
    return rows


def dc_horizon_rows(dc_runs: list[DcRun]) -> list[list[str]]:
    rows: list[dict[str, Any]] = []
    for run in dc_runs:
        block = dc_metric_block(run)
        targets = block["targets"]
        for horizon in HORIZONS:
            d = targets["demand"].get("report", {}).get(horizon, {})
            c = targets["supply"].get("report", {}).get(horizon, {})
            g = targets["gap"].get("report", {}).get(horizon, {})
            d_rmse = as_float(d.get("rmse"))
            c_rmse = as_float(c.get("rmse"))
            d_mse = as_float(d.get("mse"))
            c_mse = as_float(c.get("mse"))
            rows.append(
                {
                    "dataset": run.dataset,
                    "method": run.method_name,
                    "family": run.family,
                    "horizon": horizon,
                    "d_mse": d_mse,
                    "d_rmse": d_rmse,
                    "d_mape": as_float(d.get("mape")),
                    "c_mse": c_mse,
                    "c_rmse": c_rmse,
                    "c_mape": as_float(c.get("mape")),
                    "gap_mse": as_float(g.get("mse")),
                    "gap_rmse": as_float(g.get("rmse")),
                    "gap_mape": as_float(g.get("mape")),
                    "dc_mse": None if d_mse is None or c_mse is None else d_mse + c_mse,
                    "dc_rmse": None if d_rmse is None or c_rmse is None else d_rmse + c_rmse,
                }
            )
    rows.sort(
        key=lambda r: (
            DATASET_ORDER.index(r["dataset"]),
            float("inf") if r["dc_rmse"] is None else r["dc_rmse"],
            HORIZONS.index(r["horizon"]),
            r["method"],
        )
    )
    return [
        [
            row["dataset"],
            row["method"],
            row["family"],
            row["horizon"],
            fmt(row["d_mse"]),
            fmt(row["d_rmse"]),
            fmt(row["d_mape"]),
            fmt(row["c_mse"]),
            fmt(row["c_rmse"]),
            fmt(row["c_mape"]),
            fmt(row["gap_mse"]),
            fmt(row["gap_rmse"]),
            fmt(row["gap_mape"]),
            fmt(row["dc_mse"]),
            fmt(row["dc_rmse"]),
        ]
        for row in rows
    ]


def load_v_final_rows() -> tuple[list[list[str]], list[list[str]]]:
    specs = [
        ("NYC V", "v_nyc"),
        ("PEMS-BAY", "v_pems_bay"),
        ("METR-LA", "v_metr_la"),
    ]
    overall = []
    horizon_rows = []
    for dataset, export_id in specs:
        path = EXPORT_DIR / "checkpoints" / export_id / "predictor_test_metrics.json"
        payload = load_json(path)
        speed = payload["raw_metrics"]["speed"]
        overall.append(
            {
                "dataset": dataset,
                "mse": as_float(speed.get("mse")),
                "rmse": as_float(speed.get("rmse")),
                "mae": as_float(speed.get("mae")),
                "mape": as_float(speed.get("mape")),
                "source": export_id,
            }
        )
        report = payload.get("raw_metrics_report", {}).get("speed", {})
        for horizon in HORIZONS:
            item = report.get(horizon, {})
            horizon_rows.append(
                [
                    dataset,
                    horizon,
                    fmt(item.get("mse")),
                    fmt(item.get("rmse")),
                    fmt(item.get("mae")),
                    fmt(item.get("mape")),
                    export_id,
                ]
            )
    overall.sort(key=lambda row: float("inf") if row["rmse"] is None else row["rmse"])
    overall_rows = [
        [
            str(rank),
            row["dataset"],
            fmt(row["mse"]),
            fmt(row["rmse"]),
            fmt(row["mae"]),
            fmt(row["mape"]),
            row["source"],
        ]
        for rank, row in enumerate(overall, 1)
    ]
    return overall_rows, horizon_rows


def parse_sensor_report(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")
    generated_match = re.search(r"Generated at `([^`]+)`", text)
    generated = generated_match.group(1) if generated_match else path.stem.replace("sensor_benchmark_report_", "")
    sections = re.split(r"(?m)^## ", text)
    rows: list[list[str]] = []
    for section in sections[1:]:
        lines = section.splitlines()
        if not lines:
            continue
        heading = lines[0].strip()
        if "[" in heading and heading.endswith("]"):
            dataset, config = heading[:-1].split("[", 1)
            dataset = dataset.strip()
            config = config.strip()
        else:
            dataset = heading.strip()
            config = "default"
        run_dir = "n/a"
        best_epoch = "n/a"
        for line in lines:
            if line.startswith("- Run dir:"):
                match = re.search(r"`([^`]+)`", line)
                if match:
                    run_dir = match.group(1)
            elif line.startswith("- Best epoch:"):
                match = re.search(r"`([^`]+)`", line)
                if match:
                    best_epoch = match.group(1)
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("|") or stripped.startswith("| ---") or "Horizon" in stripped:
                continue
            parts = [part.strip() for part in stripped.strip("|").split("|")]
            if len(parts) < 4:
                continue
            horizon = parts[0]
            rmse = parts[1]
            mae = parts[2]
            if len(parts) >= 5:
                mape = parts[3].replace("%", "")
                mse = parts[4]
            else:
                mape = "n/a"
                mse = parts[3]
            rows.append([generated, dataset, config, horizon, mse, rmse, mae, mape, best_epoch, run_dir])
    return rows


def v_historical_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for report in sorted((ROOT / "external_speed_benchmarks" / "results").glob("sensor_benchmark_report_*.md")):
        rows.extend(parse_sensor_report(report))
    return rows


def checkpoint_rows() -> tuple[list[list[str]], list[list[str]]]:
    manifest = load_json(MANIFEST_PATH)
    checkpoint_table = []
    source_table = []
    for exp in manifest["experiments"]:
        checkpoint_table.append(
            [
                exp["id"],
                exp["task"],
                exp["dataset"],
                exp["exported_dir"],
                ", ".join(item["kind"] for item in exp.get("copied_checkpoints", [])),
            ]
        )
        source_table.append(
            [
                exp["id"],
                exp["label"],
                exp["metrics_path"],
                exp["run_dir"],
                exp.get("note", ""),
            ]
        )
    return checkpoint_table, source_table


def write_report() -> None:
    dc_runs = load_dc_baselines() + load_stgat_dc_runs()
    v_overall, v_horizons = load_v_final_rows()
    checkpoint_table, source_table = checkpoint_rows()

    lines: list[str] = []
    lines.extend(
        [
            "# STGAT Final DC and V Experiment Results",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "This export contains the selected final/best STGAT results, all available DC benchmark method results, selected V final results, historical V benchmark report snapshots, and exported best checkpoints.",
            "",
            "Metric convention: DC_RMSE = D_RMSE + C_RMSE. DC_MSE_SUM = D_MSE + C_MSE. DC_MAPE_AVG is the mean of D_MAPE and C_MAPE when both are available. For STGAT DC rows, MAPE was not stored in the original final JSONs, so those cells are marked n/a.",
            "",
            "## DC Overall Leaderboards",
            "",
        ]
    )
    lines.extend(
        md_table(
            [
                "Dataset",
                "Rank",
                "Method",
                "Family",
                "D_MSE",
                "D_RMSE",
                "D_MAPE",
                "C_MSE",
                "C_RMSE",
                "C_MAPE",
                "Gap_MSE",
                "Gap_RMSE",
                "Gap_MAPE",
                "DC_MSE_SUM",
                "DC_RMSE",
                "DC_MAPE_AVG",
                "Epochs",
                "Best Epoch",
                "Source",
            ],
            dc_overall_rows(dc_runs),
        )
    )
    lines.extend(["", "## DC Horizon Details", ""])
    lines.extend(
        md_table(
            [
                "Dataset",
                "Method",
                "Family",
                "Horizon",
                "D_MSE",
                "D_RMSE",
                "D_MAPE",
                "C_MSE",
                "C_RMSE",
                "C_MAPE",
                "Gap_MSE",
                "Gap_RMSE",
                "Gap_MAPE",
                "DC_MSE_SUM",
                "DC_RMSE",
            ],
            dc_horizon_rows(dc_runs),
        )
    )
    lines.extend(["", "## V Final Overall Results", ""])
    lines.extend(
        md_table(
            ["Rank", "Dataset", "V_MSE", "V_RMSE", "V_MAE", "V_MAPE", "Selected Run"],
            v_overall,
        )
    )
    lines.extend(["", "## V Final Horizon Results", ""])
    lines.extend(
        md_table(
            ["Dataset", "Horizon", "V_MSE", "V_RMSE", "V_MAE", "V_MAPE", "Selected Run"],
            v_horizons,
        )
    )
    historical = v_historical_rows()
    if historical:
        lines.extend(["", "## V Historical Benchmark Report Snapshots", ""])
        lines.extend(
            md_table(
                [
                    "Generated",
                    "Dataset",
                    "Config",
                    "Horizon",
                    "V_MSE",
                    "V_RMSE",
                    "V_MAE",
                    "V_MAPE",
                    "Best Epoch",
                    "Run Dir",
                ],
                historical,
            )
        )
    lines.extend(["", "## Exported Checkpoints", ""])
    lines.extend(
        md_table(
            ["ID", "Task", "Dataset", "Export Folder", "Copied Best Checkpoints"],
            checkpoint_table,
        )
    )
    lines.extend(["", "## Source Runs", ""])
    lines.extend(md_table(["ID", "Label", "Metrics JSON", "Original Run Dir", "Note"], source_table))
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_report()
