from __future__ import annotations

import argparse
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dc_benchmark.run_all_benchmarks_colored import ALL_METHODS, parse_methods  # noqa: E402


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


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def validate_dataset(source_data_dir: Path, benchmark_dir: Path, hist_len: int, pred_horizon: int, colors: bool) -> None:
    print(color("数据检查", CYAN, enabled=colors), flush=True)
    mask_path = source_data_dir / "observed_time_mask.npy"
    manifest_path = source_data_dir / "manifest.json"
    if not mask_path.exists():
        raise FileNotFoundError(f"找不到观测遮罩: {mask_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 manifest: {manifest_path}")

    mask = np.load(mask_path).astype(bool)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing = manifest.get("temporal_partition", {}).get("missing_archive_dates_between_start_end", [])
    print(
        color(
            f"深圳源数据 {source_data_dir} | observed={int(mask.sum())}/{len(mask)} "
            f"unobserved={int((~mask).sum())} missing_dates={missing}",
            GREEN if int((~mask).sum()) > 0 else CYAN,
            enabled=colors,
        ),
        flush=True,
    )

    splits_path = benchmark_dir / "splits.json"
    if not splits_path.exists():
        print(color(f"benchmark 尚未导出，将由后续脚本生成: {benchmark_dir}", ORANGE, enabled=colors), flush=True)
        return
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    if int(splits.get("hist_len", -1)) != int(hist_len) or int(splits.get("pred_horizon", -1)) != int(pred_horizon):
        print(
            color(
                f"benchmark hist/pred 与本次参数不同，将由后续脚本重导出: "
                f"old={splits.get('hist_len')}/{splits.get('pred_horizon')} new={hist_len}/{pred_horizon}",
                ORANGE,
                enabled=colors,
            ),
            flush=True,
        )
        return

    window_len = int(hist_len) + int(pred_horizon)
    for split_name, indices in splits.get("indices", {}).items():
        for idx in indices:
            start = int(idx)
            if not mask[start:start + window_len].all():
                raise ValueError(f"{benchmark_dir} 的 {split_name} window={idx} 触碰未观测 slot")
    print(
        color(
            f"benchmark 窗口检查通过 | counts={splits.get('sample_counts')} "
            f"excluded_by_mask={splits.get('excluded_by_observed_time_mask')}",
            GREEN,
            enabled=colors,
        ),
        flush=True,
    )


def stream_stgat(
    command: list[str],
    *,
    log_dir: Path,
    epochs: int,
    total_methods: int,
    colors: bool,
) -> None:
    prefix = color(f"[1/{total_methods}]", YELLOW, enabled=colors)
    print(color("=" * 96, CYAN, enabled=colors), flush=True)
    print(f"{prefix} STGAT Predictor (my_method)", flush=True)
    print(color("PARAM " + command_text(command), RED, enabled=colors), flush=True)

    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=build_env(),
    )
    assert process.stdout is not None

    epoch_pattern = re.compile(r"\[Ep\s*(\d+)\]")
    raw_dc_pattern = re.compile(r"ValRawDC=([0-9]+(?:\.[0-9]+)?)")
    d_pattern = re.compile(r"\(D:([0-9]+(?:\.[0-9]+)?)\s+C:([0-9]+(?:\.[0-9]+)?)\)")
    best_raw = float("inf")
    best_epoch = 0
    start = time.time()

    for line in process.stdout:
        text = line.rstrip()
        epoch_match = epoch_pattern.search(text)
        if not epoch_match:
            if text:
                print(f"{prefix} {text}", flush=True)
            continue

        epoch = int(epoch_match.group(1))
        elapsed = time.time() - start
        epoch_sec = elapsed / max(epoch, 1)
        eta = epoch_sec * max(int(epochs) - epoch, 0)
        raw_match = raw_dc_pattern.search(text)
        improved = False
        val_text = ""
        if raw_match:
            val_raw = float(raw_match.group(1))
            improved = val_raw < best_raw
            if improved:
                best_raw = val_raw
                best_epoch = epoch
            val_text = f"val_raw_dc={val_raw:.6f}"
            dc_match = d_pattern.search(text)
            if dc_match:
                val_text += f" D_RMSE={float(dc_match.group(1)):.6f} C_RMSE={float(dc_match.group(2)):.6f}"
        else:
            val_text = "val_raw_dc=waiting"

        best_text = (
            f"best_raw_dc={best_raw:.6f}" if best_epoch else "best_raw_dc=waiting"
        )
        output = (
            f"{prefix} STGAT Ep {epoch:03d}/{int(epochs):03d} "
            f"{val_text} "
            f"{color(best_text, GREEN, enabled=colors)} "
            f"{color(f'best_epoch={best_epoch}', ORANGE, enabled=colors)} "
            f"epoch={epoch_sec:.1f}s ETA={format_duration(eta)}"
        )
        if improved:
            output += " " + color("NEW BEST", GREEN, enabled=colors)
        print(output, flush=True)

    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(f"STGAT 训练失败，退出码 {return_code}")

    metrics_path = log_dir / "predictor_test_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        raw = metrics.get("raw_metrics", {})
        demand = raw.get("demand", {})
        supply = raw.get("supply", {})
        raw_dc = raw.get("raw_dc")
        if raw_dc is None and "rmse" in demand and "rmse" in supply:
            raw_dc = float(demand["rmse"]) + float(supply["rmse"])
        print(
            f"{prefix} "
            + color(
                f"DONE raw_dc={float(raw_dc):.6f} "
                f"D_RMSE={float(demand.get('rmse', float('nan'))):.6f} "
                f"C_RMSE={float(supply.get('rmse', float('nan'))):.6f}",
                GREEN,
                enabled=colors,
            )
            + " "
            + color(f"best_epoch={best_epoch}", ORANGE, enabled=colors),
            flush=True,
        )
    print(f"{prefix} elapsed={format_duration(time.time() - start)}", flush=True)


def stream_plain_command(command: list[str], *, colors: bool) -> None:
    print(color("=" * 96, CYAN, enabled=colors), flush=True)
    print(color("启动 benchmark: " + command_text(command), RED, enabled=colors), flush=True)
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=build_env(),
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise SystemExit(f"benchmark 训练失败，退出码 {return_code}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Shenzhen DC STGAT + 12 benchmark methods with colored progress.")
    parser.add_argument("--source-data-dir", default="data/shenzhen_dc")
    parser.add_argument("--benchmark-dir", default="data/shenzhen_dc_benchmark")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--hist-len", type=int, default=14)
    parser.add_argument("--pred-horizon", type=int, default=4)
    parser.add_argument("--report-horizons-minutes", default="15,30,60")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--stgat-batch-size", type=int, default=8)
    parser.add_argument("--benchmark-batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--force-dataset", action="store_true")
    parser.add_argument("--skip-stgat", action="store_true")
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    colors = not args.no_color
    methods = parse_methods(args.methods)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"runs/shenzhen_dc_full_{stamp}")
    stgat_log_dir = output_dir / "stgat"
    benchmark_output_dir = output_dir / "benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)

    source_data_dir = Path(args.source_data_dir)
    benchmark_dir = Path(args.benchmark_dir)
    validate_dataset(source_data_dir, benchmark_dir, args.hist_len, args.pred_horizon, colors)

    total_methods = (0 if args.skip_stgat else 1) + (0 if args.skip_benchmarks else len(methods))
    if total_methods <= 0:
        raise SystemExit("没有要运行的方法。")

    stgat_command = [
        sys.executable,
        "-u",
        "train_predictor.py",
        "--data-dir",
        str(source_data_dir),
        "--log-dir",
        str(stgat_log_dir),
        "--hist-len",
        str(args.hist_len),
        "--pred-horizon",
        str(args.pred_horizon),
        "--report-horizons-minutes",
        args.report_horizons_minutes,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.stgat_batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--lr",
        str(args.lr),
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

    benchmark_command = [
        sys.executable,
        "-u",
        "dc_benchmark/run_all_benchmarks_colored.py",
        "--methods",
        args.methods,
        "--source-data-dir",
        str(source_data_dir),
        "--dataset-dir",
        str(benchmark_dir),
        "--output-dir",
        str(benchmark_output_dir),
        "--hist-len",
        str(args.hist_len),
        "--pred-horizon",
        str(args.pred_horizon),
        "--report-horizons-minutes",
        args.report_horizons_minutes,
        "--split-policy",
        "project_monthly",
        "--time-feature-mode",
        "baseline",
        "--epochs",
        str(args.epochs),
        "--early-stop-patience",
        str(args.early_stop_patience),
        "--batch-size",
        str(args.benchmark_batch_size),
        "--max-train-rows",
        str(args.max_train_rows),
        "--hidden-dim",
        str(args.hidden_dim),
        "--lr",
        str(args.lr),
        "--device",
        args.device,
        "--max-train-samples",
        str(args.max_train_samples),
        "--max-eval-samples",
        str(args.max_eval_samples),
        "--ordinal-offset",
        "0" if args.skip_stgat else "1",
        "--total-override",
        str(total_methods),
    ]
    if args.force_dataset:
        benchmark_command.append("--force-dataset")
    if not colors:
        benchmark_command.append("--no-color")

    run_manifest = {
        "created_at": datetime.now().isoformat(),
        "source_data_dir": str(source_data_dir),
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(output_dir),
        "methods": ["stgat"] * (0 if args.skip_stgat else 1) + ([] if args.skip_benchmarks else methods),
        "epochs": int(args.epochs),
        "early_stop_patience": int(args.early_stop_patience),
        "hist_len": int(args.hist_len),
        "pred_horizon": int(args.pred_horizon),
        "report_horizons_minutes": args.report_horizons_minutes,
        "stgat_command": stgat_command,
        "benchmark_command": benchmark_command,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(color("=" * 96, CYAN, enabled=colors), flush=True)
    print(color(f"深圳 DC 自动实验 | 总方法数={total_methods} | 输出={output_dir}", CYAN, enabled=colors), flush=True)
    print(color(f"STGAT epochs={args.epochs}; benchmark epochs={args.epochs}, early_stop={args.early_stop_patience}", RED, enabled=colors), flush=True)

    if args.dry_run:
        print(color("DRY RUN，不启动训练。", ORANGE, enabled=colors), flush=True)
        if not args.skip_stgat:
            print(command_text(stgat_command), flush=True)
        if not args.skip_benchmarks:
            print(command_text(benchmark_command), flush=True)
        return

    if not args.skip_stgat:
        stream_stgat(stgat_command, log_dir=stgat_log_dir, epochs=args.epochs, total_methods=total_methods, colors=colors)
    if not args.skip_benchmarks:
        stream_plain_command(benchmark_command, colors=colors)

    print(color("=" * 96, CYAN, enabled=colors), flush=True)
    print(color(f"全部实验完成 | 输出目录={output_dir}", GREEN, enabled=colors), flush=True)


if __name__ == "__main__":
    main()
