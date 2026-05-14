from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

RESET = "\033[0m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
ORANGE = "\033[38;5;208m"
CYAN = "\033[96m"


def color(text: str, code: str, *, enabled: bool = True) -> str:
    return f"{code}{text}{RESET}" if enabled else text


def duration(seconds: float) -> str:
    total = int(round(max(float(seconds), 0.0)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def group_prefix(ordinal: int, total: int, *, colors: bool) -> str:
    return color(f"[第 {ordinal}/{total} 组]", YELLOW, enabled=colors)


def env() -> dict[str, str]:
    values = os.environ.copy()
    values["PYTHONUNBUFFERED"] = "1"
    values["PYTHONUTF8"] = "1"
    values["PYTHONIOENCODING"] = "utf-8"
    return values


def default_configs() -> list[dict[str, Any]]:
    return [
        {
            "name": "control_h14p4_lr0010_bs8_h32_topk16_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 1e-3,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0007_bs8_h32_topk16_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 7e-4,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0010_bs8_h32_topk16_c15",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 1e-3,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.5,
        },
        {
            "name": "lr0008_bs8_h32_topk16_c2",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 2.0,
        },
        {
            "name": "lr0008_bs8_h64_topk16_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h64_topk16_c15",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.5,
        },
        {
            "name": "lr0005_bs8_h64_b3_topk16_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 5e-4,
            "batch_size": 8,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_st_blocks": 3,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h64_topk32_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 32,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0010_bs16_h32_topk16_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 1e-3,
            "batch_size": 16,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "h12p4_lr0010_bs8_h32_topk16_c1",
            "hist_len": 12,
            "pred_horizon": 4,
            "lr": 1e-3,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "h16p4_lr0010_bs8_h32_topk16_c1",
            "hist_len": 16,
            "pred_horizon": 4,
            "lr": 1e-3,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "h20p4_lr0010_bs8_h32_topk16_c1",
            "hist_len": 20,
            "pred_horizon": 4,
            "lr": 1e-3,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0005_bs8_h32_topk16_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 5e-4,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0015_bs8_h32_topk16_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 1.5e-3,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h32_topk8_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 8,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h32_topk12_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 12,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h32_topk20_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 20,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h32_topk24_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 24,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h32_topk32_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 32,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h48_topk16_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 48,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h48_topk16_c15",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 48,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.5,
        },
        {
            "name": "lr0008_bs8_h48_topk24_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 48,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 24,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h64_topk24_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 24,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0005_bs8_h64_b3_topk24_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 5e-4,
            "batch_size": 8,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_st_blocks": 3,
            "adaptive_topk": 24,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0010_bs16_h64_topk16_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 1e-3,
            "batch_size": 16,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h64_heads8_topk16_c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 64,
            "num_heads": 8,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.0,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h32_topk16_d15c1",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 1.5,
            "lambda2": 1.0,
        },
        {
            "name": "lr0008_bs8_h32_topk16_d08c12",
            "hist_len": 14,
            "pred_horizon": 4,
            "lr": 8e-4,
            "batch_size": 8,
            "hidden_dim": 32,
            "num_heads": 4,
            "num_st_blocks": 2,
            "adaptive_topk": 16,
            "lambda1": 0.8,
            "lambda2": 1.2,
        },
    ]


def validate_dataset(data_dir: Path, hist_len: int, pred_horizon: int, colors: bool) -> None:
    required = [
        "node_demand.npy",
        "node_supply.npy",
        "targets_dc.npy",
        "edge_speeds.npy",
        "adjacency_matrix.npy",
        "edge_index.npy",
        "time_meta.csv",
        "observed_time_mask.npy",
        "manifest.json",
    ]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {data_dir}: {missing}")
    mask = np.load(data_dir / "observed_time_mask.npy").astype(bool)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    print(
        color(
            f"DATA {data_dir} observed={int(mask.sum())}/{len(mask)} "
            f"unobserved={int((~mask).sum())} hist/pred={hist_len}/{pred_horizon} "
            f"missing={manifest.get('temporal_partition', {}).get('missing_archive_dates_between_start_end', [])}",
            CYAN,
            enabled=colors,
        ),
        flush=True,
    )


def config_hist_len(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    return int(cfg.get("hist_len", args.hist_len))


def config_pred_horizon(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    return int(cfg.get("pred_horizon", args.pred_horizon))


def read_stgat_result(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "predictor_test_metrics.json"
    log_path = run_dir / "predictor_log.json"
    if not metrics_path.exists() or not log_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = json.loads(log_path.read_text(encoding="utf-8"))
    raw = metrics.get("raw_metrics", {})
    demand = raw.get("demand", {})
    supply = raw.get("supply", {})
    test_raw_dc = raw.get("raw_dc")
    if test_raw_dc is None and "rmse" in demand and "rmse" in supply:
        test_raw_dc = float(demand["rmse"]) + float(supply["rmse"])
    best_epoch = 0
    best_val = float("inf")
    for record in history:
        value = record.get("val_raw_dc")
        if value is not None and float(value) < best_val:
            best_epoch = int(record["epoch"])
            best_val = float(value)
    if best_epoch == 0 and metrics.get("selected_checkpoint_metric") is not None:
        best_val = float(metrics["selected_checkpoint_metric"])
    return {
        "selection_score": float(best_val),
        "best_epoch": int(best_epoch),
        "test_raw_dc": float(test_raw_dc),
        "demand_rmse": float(demand.get("rmse", float("nan"))),
        "supply_rmse": float(supply.get("rmse", float("nan"))),
        "demand_mae": float(demand.get("mae", float("nan"))),
        "supply_mae": float(supply.get("mae", float("nan"))),
        "completed_epochs": int(len(history)),
    }


def baseline_result(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if (path / "stgat").exists():
        return read_stgat_result(path / "stgat")
    return read_stgat_result(path)


def build_command(
    *,
    data_dir: Path,
    run_dir: Path,
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    hist_len = config_hist_len(cfg, args)
    pred_horizon = config_pred_horizon(cfg, args)
    command = [
        sys.executable,
        "-u",
        "train_predictor.py",
        "--data-dir",
        str(data_dir),
        "--log-dir",
        str(run_dir),
        "--hist-len",
        str(hist_len),
        "--pred-horizon",
        str(pred_horizon),
        "--report-horizons-minutes",
        args.report_horizons_minutes,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(cfg["batch_size"]),
        "--lr",
        str(cfg["lr"]),
        "--hidden-dim",
        str(cfg["hidden_dim"]),
        "--num-heads",
        str(cfg["num_heads"]),
        "--num-st-blocks",
        str(cfg["num_st_blocks"]),
        "--adaptive-topk",
        str(cfg["adaptive_topk"]),
        "--lambda1",
        str(cfg["lambda1"]),
        "--lambda2",
        str(cfg["lambda2"]),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--num-workers",
        str(args.num_workers),
        "--train-task",
        "dc",
        "--monitor-task",
        "raw_dc",
        "--val-interval",
        "1",
        "--log-interval",
        "1",
    ]
    if int(cfg.get("num_gtcn_layers", args.num_gtcn_layers)) != 2:
        command += ["--num-gtcn-layers", str(cfg.get("num_gtcn_layers", args.num_gtcn_layers))]
    if int(cfg.get("kernel_size", args.kernel_size)) != 3:
        command += ["--kernel-size", str(cfg.get("kernel_size", args.kernel_size))]
    return command


def append_status(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows_sorted = sorted(rows, key=lambda row: float(row.get("selection_score", float("inf"))))
    for rank, row in enumerate(rows_sorted, start=1):
        row["rank_by_val"] = rank
    fields = [
        "rank_by_val",
        "ordinal",
        "name",
        "status",
        "selection_score",
        "test_raw_dc",
        "demand_rmse",
        "supply_rmse",
        "demand_mae",
        "supply_mae",
        "completed_epochs",
        "best_epoch",
        "lr",
        "batch_size",
        "hidden_dim",
        "num_heads",
        "num_st_blocks",
        "adaptive_topk",
        "hist_len",
        "pred_horizon",
        "lambda1",
        "lambda2",
        "run_dir",
    ]
    with (output_dir / "sweep_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_sorted)
    lines = [
        "# Shenzhen STGAT Tuning Summary",
        "",
        "| Rank(val) | Config | Val rawDC | Test rawDC | D RMSE | C RMSE | Epochs | Best Ep | Params |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows_sorted:
        if row.get("status") != "ok":
            continue
        params = (
            f"lr={row['lr']}, bs={row['batch_size']}, h={row['hidden_dim']}, "
            f"hist/pred={row.get('hist_len')}/{row.get('pred_horizon')}, "
            f"blocks={row['num_st_blocks']}, topk={row['adaptive_topk']}, "
            f"lambda=({row['lambda1']},{row['lambda2']})"
        )
        lines.append(
            f"| {row['rank_by_val']} | `{row['name']}` | {float(row['selection_score']):.5f} | "
            f"{float(row['test_raw_dc']):.5f} | {float(row['demand_rmse']):.5f} | "
            f"{float(row['supply_rmse']):.5f} | {row['completed_epochs']} | {row['best_epoch']} | {params} |"
        )
    failed = [row for row in rows_sorted if row.get("status") != "ok"]
    if failed:
        lines += ["", "## Failed", ""]
        for row in failed:
            lines.append(f"- `{row['name']}`: {row.get('status')}")
    (output_dir / "sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_config(
    *,
    cfg: dict[str, Any],
    ordinal: int,
    total: int,
    output_dir: Path,
    data_dir: Path,
    args: argparse.Namespace,
    colors: bool,
    global_best: dict[str, Any],
    completed_seconds: list[float],
    rows: list[dict[str, Any]],
) -> None:
    name = str(cfg["name"])
    run_dir = output_dir / f"{ordinal:03d}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = group_prefix(ordinal, total, colors=colors)
    result = read_stgat_result(run_dir)
    if result and args.resume:
        row = {**cfg, **result, "ordinal": ordinal, "status": "ok", "run_dir": str(run_dir)}
        rows.append(row)
        print(
            f"{prefix} SKIP existing {name} "
            + color(f"val={result['selection_score']:.5f}", GREEN, enabled=colors)
            + f" test={result['test_raw_dc']:.5f}",
            flush=True,
        )
        return

    command = build_command(data_dir=data_dir, run_dir=run_dir, cfg=cfg, args=args)
    command_log = run_dir / "command.txt"
    command_log.write_text(command_text(command) + "\n", encoding="utf-8")
    stdout_log = run_dir / "stdout.log"
    hist_len = config_hist_len(cfg, args)
    pred_horizon = config_pred_horizon(cfg, args)
    print(color("=" * 96, CYAN, enabled=colors), flush=True)
    print(f"{prefix} STGAT tuning {name}", flush=True)
    print(
        color(
            "PARAM "
            f"lr={cfg['lr']} bs={cfg['batch_size']} hidden={cfg['hidden_dim']} "
            f"hist/pred={hist_len}/{pred_horizon} "
            f"heads={cfg['num_heads']} blocks={cfg['num_st_blocks']} topk={cfg['adaptive_topk']} "
            f"lambda=({cfg['lambda1']},{cfg['lambda2']}) epochs={args.epochs}",
            RED,
            enabled=colors,
        ),
        flush=True,
    )

    start = time.time()
    best_val = float("inf")
    best_epoch = 0
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env(),
    )
    assert process.stdout is not None
    epoch_pattern = re.compile(r"\[Ep\s*(\d+)\]")
    val_pattern = re.compile(r"ValRawDC=([0-9]+(?:\.[0-9]+)?)")
    dc_pattern = re.compile(r"\(D:([0-9]+(?:\.[0-9]+)?)\s+C:([0-9]+(?:\.[0-9]+)?)\)")

    with stdout_log.open("w", encoding="utf-8") as log_handle:
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            text = line.rstrip()
            epoch_match = epoch_pattern.search(text)
            if not epoch_match:
                if text:
                    print(f"{prefix} {text}", flush=True)
                continue
            epoch = int(epoch_match.group(1))
            elapsed = time.time() - start
            epoch_sec = elapsed / max(epoch, 1)
            eta_method = epoch_sec * max(args.epochs - epoch, 0)
            average_completed = (
                sum(completed_seconds) / len(completed_seconds)
                if completed_seconds
                else epoch_sec * args.epochs
            )
            eta_total = eta_method + max(total - ordinal, 0) * average_completed
            val_match = val_pattern.search(text)
            improved = False
            val_text = "val_raw_dc=waiting"
            if val_match:
                val = float(val_match.group(1))
                improved = val < best_val
                if improved:
                    best_val = val
                    best_epoch = epoch
                val_text = f"val_raw_dc={val:.5f}"
                dc_match = dc_pattern.search(text)
                if dc_match:
                    val_text += f" D={float(dc_match.group(1)):.4f} C={float(dc_match.group(2)):.4f}"
            current_global = float(global_best.get("selection_score", float("inf")))
            global_text = (
                f"global_best_val={current_global:.5f} ({global_best.get('name', 'none')})"
                if current_global < float("inf")
                else "global_best_val=none"
            )
            output = (
                f"{prefix} Ep {epoch:03d}/{args.epochs:03d} {val_text} "
                f"{color(f'best_val={best_val:.5f}', GREEN, enabled=colors)} "
                f"{color(f'best_epoch={best_epoch}', ORANGE, enabled=colors)} "
                f"epoch={epoch_sec:.1f}s ETA={duration(eta_method)} TOTAL_ETA={duration(eta_total)} "
                f"{color(global_text, GREEN, enabled=colors)}"
            )
            if improved:
                output += " " + color("NEW RUN BEST", GREEN, enabled=colors)
            print(output, flush=True)
            append_status(
                output_dir / "sweep_status.jsonl",
                {
                    "event": "epoch",
                    "name": name,
                    "ordinal": ordinal,
                    "total": total,
                    "group_label": f"{ordinal}/{total}",
                    "hist_len": hist_len,
                    "pred_horizon": pred_horizon,
                    "epoch": epoch,
                    "epochs": args.epochs,
                    "val_raw_dc": float(val_match.group(1)) if val_match else None,
                    "best_val_raw_dc": best_val if best_val < float("inf") else None,
                    "best_epoch": best_epoch,
                    "elapsed_sec": elapsed,
                    "eta_method_sec": eta_method,
                    "eta_total_sec": eta_total,
                },
            )
    return_code = process.wait()
    elapsed = time.time() - start
    completed_seconds.append(elapsed)
    if return_code != 0:
        row = {**cfg, "ordinal": ordinal, "status": f"failed:{return_code}", "run_dir": str(run_dir)}
        rows.append(row)
        write_summary(output_dir, rows)
        print(color(f"{prefix} FAILED {name} exit={return_code}", RED, enabled=colors), flush=True)
        return

    result = read_stgat_result(run_dir)
    if result is None:
        row = {**cfg, "ordinal": ordinal, "status": "failed:no_metrics", "run_dir": str(run_dir)}
        rows.append(row)
        write_summary(output_dir, rows)
        print(color(f"{prefix} FAILED {name}: no metrics", RED, enabled=colors), flush=True)
        return
    row = {**cfg, **result, "ordinal": ordinal, "status": "ok", "run_dir": str(run_dir)}
    rows.append(row)
    if float(result["selection_score"]) < float(global_best.get("selection_score", float("inf"))):
        global_best.clear()
        global_best.update({"name": name, **result})
        global_message = color("NEW GLOBAL BEST BY VAL", GREEN, enabled=colors)
    else:
        global_message = ""
    print(
        f"{prefix} "
        + color(
            f"DONE {name} val={result['selection_score']:.5f} test_raw={result['test_raw_dc']:.5f} "
            f"D_RMSE={result['demand_rmse']:.5f} C_RMSE={result['supply_rmse']:.5f}",
            GREEN,
            enabled=colors,
        )
        + " "
        + color(f"best_epoch={result['best_epoch']}", ORANGE, enabled=colors)
        + (" " + global_message if global_message else ""),
        flush=True,
    )
    print(f"{prefix} elapsed={duration(elapsed)}", flush=True)
    write_summary(output_dir, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colored Shenzhen STGAT hyperparameter tuning.")
    parser.add_argument("--data-dir", default="data/shenzhen_dc")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--baseline-dir", default="runs/shenzhen_dc_full_20260508_002254")
    parser.add_argument("--hist-len", type=int, default=14)
    parser.add_argument("--pred-horizon", type=int, default=4)
    parser.add_argument("--report-horizons-minutes", default="15,30,60")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--num-gtcn-layers", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    colors = not args.no_color
    data_dir = Path(args.data_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"runs/shenzhen_stgat_tuning_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_dataset(data_dir, args.hist_len, args.pred_horizon, colors)

    configs = default_configs()
    if args.max_configs and args.max_configs > 0:
        configs = configs[: args.max_configs]
    baseline = baseline_result(Path(args.baseline_dir))
    global_best: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    if baseline is not None:
        global_best.update({"name": "baseline_previous_stgat", **baseline})
        rows.append(
            {
                "ordinal": 0,
                "name": "baseline_previous_stgat",
                "status": "ok",
                "lr": 1e-3,
                "batch_size": 8,
                "hidden_dim": 32,
                "num_heads": 4,
                "num_st_blocks": 2,
                "adaptive_topk": 16,
                "hist_len": 14,
                "pred_horizon": 4,
                "lambda1": 1.0,
                "lambda2": 1.0,
                "run_dir": str(Path(args.baseline_dir) / "stgat"),
                **baseline,
            }
        )
    manifest = {
        "created_at": datetime.now().isoformat(),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "baseline_dir": args.baseline_dir,
        "selection_policy": "Choose configurations by validation raw_dc; test raw_dc is report-only.",
        "hist_len": args.hist_len,
        "pred_horizon": args.pred_horizon,
        "report_horizons_minutes": args.report_horizons_minutes,
        "epochs": args.epochs,
        "configs": configs,
    }
    (output_dir / "sweep_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(output_dir, rows)

    print(color("=" * 96, CYAN, enabled=colors), flush=True)
    print(color(f"Shenzhen STGAT tuning output={output_dir}", CYAN, enabled=colors), flush=True)
    if baseline is not None:
        print(
            color(
                f"BASELINE val={baseline['selection_score']:.5f} test_raw={baseline['test_raw_dc']:.5f} "
                f"D={baseline['demand_rmse']:.5f} C={baseline['supply_rmse']:.5f} best_epoch={baseline['best_epoch']}",
                GREEN,
                enabled=colors,
            ),
            flush=True,
        )
    print(color(f"Selection: validation raw_dc; configs={len(configs)} epochs={args.epochs}", RED, enabled=colors), flush=True)
    if args.dry_run:
        print(color("DRY RUN", ORANGE, enabled=colors), flush=True)
        for idx, cfg in enumerate(configs, start=1):
            cmd = build_command(data_dir=data_dir, run_dir=output_dir / f"{idx:03d}_{cfg['name']}", cfg=cfg, args=args)
            print(f"{group_prefix(idx, len(configs), colors=colors)} {cfg['name']} :: {command_text(cmd)}", flush=True)
        return

    completed_seconds: list[float] = []
    for ordinal, cfg in enumerate(configs, start=1):
        run_config(
            cfg=cfg,
            ordinal=ordinal,
            total=len(configs),
            output_dir=output_dir,
            data_dir=data_dir,
            args=args,
            colors=colors,
            global_best=global_best,
            completed_seconds=completed_seconds,
            rows=rows,
        )
    write_summary(output_dir, rows)
    print(color("=" * 96, CYAN, enabled=colors), flush=True)
    print(
        color(
            f"ALL DONE best_by_val={global_best.get('name')} "
            f"val={float(global_best.get('selection_score', float('nan'))):.5f} "
            f"test_raw={float(global_best.get('test_raw_dc', float('nan'))):.5f} "
            f"summary={output_dir / 'sweep_summary.md'}",
            GREEN,
            enabled=colors,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
