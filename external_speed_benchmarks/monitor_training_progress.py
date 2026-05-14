from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch


def is_transient_checkpoint_read_error(exc: BaseException) -> bool:
    text = str(exc)
    return (
        "PytorchStreamReader" in text
        or "file read failed" in text
        or "failed finding central directory" in text
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor external speed training runs by polling stgat_latest.pt history."
    )
    parser.add_argument(
        "--run-dirs",
        required=True,
        help="Comma-separated run directories to watch in order.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print the current status once and exit.",
    )
    return parser.parse_args()


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_float(value: object) -> str:
    if value is None:
        return "nan"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(number):
        return "nan"
    return f"{number:.4f}"


def format_lr(value: object) -> str:
    if value is None:
        return "nan"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(number):
        return "nan"
    if number == 0.0:
        return "0"
    if abs(number) < 1e-4:
        return f"{number:.3e}"
    return f"{number:.6f}"


def format_elapsed(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(seconds):
        return "n/a"
    return f"{seconds:.1f}s"


def load_checkpoint(path: Path) -> dict | None:
    if not path.exists():
        return None
    last_exc: BaseException | None = None
    for attempt in range(5):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                return torch.load(path, map_location="cpu")
        except RuntimeError as exc:
            last_exc = exc
            if not is_transient_checkpoint_read_error(exc) or attempt == 4:
                raise
            time.sleep(1.0)
    if last_exc is not None:
        raise last_exc
    return None


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def print_epoch_line(run_name: str, record: dict, best_epoch: object, best_val_rmse: object) -> None:
    epoch = record.get("epoch", "?")
    lr = record.get("lr")
    eta = record.get("eta") or "n/a"
    horizon_parts = []
    for label in ("15min", "30min", "60min"):
        key = f"val_raw_speed_rmse_{label}"
        if key in record:
            horizon_parts.append(f"{label}={format_float(record.get(key))}")
    horizon_text = f" | {' '.join(horizon_parts)}" if horizon_parts else ""
    print(
        f"[{now_text()}] {run_name} | epoch {epoch} | "
        f"val_rmse={format_float(record.get('val_raw_speed_rmse'))} | "
        f"best={format_float(best_val_rmse)} @ {best_epoch if best_epoch is not None else '?'} | "
        f"lr={format_lr(lr)} | elapsed={format_elapsed(record.get('elapsed'))}{horizon_text} | eta={eta}",
        flush=True,
    )


def print_completion_summary(run_dir: Path, metrics: dict) -> None:
    horizons = metrics.get("test_horizons") or {}
    best_metric = metrics.get("selected_checkpoint_metric")
    summary = (
        f"[{now_text()}] {run_dir.name} | completed | "
        f"best_val_rmse={format_float(best_metric)} | "
        f"test_rmse_15={format_float((horizons.get('15min') or {}).get('rmse'))} | "
        f"test_rmse_30={format_float((horizons.get('30min') or {}).get('rmse'))} | "
        f"test_rmse_60={format_float((horizons.get('60min') or {}).get('rmse'))}"
    )
    print(summary, flush=True)


def watch_run(run_dir: Path, interval: float, once: bool) -> None:
    printed_epochs: set[int] = set()
    completion_printed = False
    waiting_printed = False

    while True:
        latest_path = run_dir / "stgat_latest.pt"
        metrics_path = run_dir / "predictor_test_metrics.json"

        if not run_dir.exists():
            if not waiting_printed:
                print(f"[{now_text()}] waiting for {run_dir}", flush=True)
                waiting_printed = True
            if once:
                return
            time.sleep(interval)
            continue

        ckpt = load_checkpoint(latest_path)
        if ckpt is None:
            if not waiting_printed:
                print(f"[{now_text()}] {run_dir.name} exists but checkpoint not ready yet", flush=True)
                waiting_printed = True
            if once:
                return
            time.sleep(interval)
            continue

        waiting_printed = False
        history = ckpt.get("history") or []
        best_epoch = ckpt.get("best_epoch")
        best_val_rmse = ckpt.get("best_val_rmse")

        for record in history:
            epoch = record.get("epoch")
            if isinstance(epoch, int) and epoch not in printed_epochs:
                print_epoch_line(run_dir.name, record, best_epoch, best_val_rmse)
                printed_epochs.add(epoch)

        metrics = load_metrics(metrics_path)
        if metrics is not None and not completion_printed:
            print_completion_summary(run_dir, metrics)
            completion_printed = True
            return

        if once:
            return

        time.sleep(interval)


def main() -> None:
    args = parse_args()
    run_dirs = [Path(item.strip()) for item in args.run_dirs.split(",") if item.strip()]
    for run_dir in run_dirs:
        watch_run(run_dir, interval=args.interval, once=args.once)


if __name__ == "__main__":
    main()
