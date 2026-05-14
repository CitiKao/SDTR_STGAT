from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    r"D:\Citi\STDR\STDR_STGAT\runs\external_speed\mse_vs_rmse_aligned_20260425_132915"
)

RUNS = [
    {
        "label": "ALL_MSE",
        "path": "all_mse_topk20_seed19_ep40",
        "pid": 35764,
    },
    {
        "label": "ALL_RMSE",
        "path": "all_rmse_topk20_seed19_ep40",
        "pid": 26276,
    },
]

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def load_torch_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import torch

        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as exc:
        return {"_load_error": str(exc)}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_pid_alive(pid: int | None) -> bool | None:
    if not pid:
        return None
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return f'"{pid}"' in result.stdout or f",{pid}," in result.stdout
    except Exception:
        return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def fmt(value: Any, ndigits: int = 4) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{ndigits}f}"


def fmt_epoch(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "n/a"


def summarize_run(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = root / spec["path"]
    latest_path = run_dir / "stgat_latest.pt"
    best_path = run_dir / "stgat_best.pt"
    meta = read_json(run_dir / "stgat_meta.json")
    state = load_torch_checkpoint(latest_path)

    summary: dict[str, Any] = {
        "label": spec["label"],
        "pid": spec.get("pid"),
        "pid_alive": is_pid_alive(spec.get("pid")),
        "run_dir": run_dir,
        "latest_path": latest_path,
        "best_path": best_path,
        "latest_mtime": None,
        "load_error": None,
        "epoch": None,
        "target_epoch": None,
        "best_epoch": None,
        "best_val_rmse": None,
        "best_val_loss": None,
        "last_train": None,
        "last_val_loss": None,
        "last_val_rmse": None,
        "last_h15": None,
        "last_h30": None,
        "last_h60": None,
        "eta": None,
        "config": {},
    }

    if latest_path.exists():
        summary["latest_mtime"] = datetime.fromtimestamp(latest_path.stat().st_mtime)

    if state is None:
        return summary
    if "_load_error" in state:
        summary["load_error"] = state["_load_error"]
        return summary

    run_config = state.get("run_config") or meta.get("run_config") or meta.get("config") or {}
    history = state.get("history") or []
    last = history[-1] if history else {}
    horizon = last.get("horizon_metrics") or last.get("val_horizon_metrics") or {}

    summary["epoch"] = state.get("epoch")
    summary["target_epoch"] = run_config.get("epochs") or meta.get("target_epoch")
    summary["best_epoch"] = state.get("best_epoch") or meta.get("best_epoch")
    summary["best_val_rmse"] = state.get("best_val_rmse")
    summary["best_val_loss"] = state.get("best_val_loss")
    summary["last_train"] = last.get("train_loss") or last.get("train_v_loss") or last.get("train_v")
    summary["last_val_loss"] = last.get("val_loss") or last.get("val_v_loss") or last.get("val_v")
    summary["last_val_rmse"] = (
        last.get("val_rmse")
        or last.get("val_raw_speed_rmse")
        or last.get("val_raw_rmse")
        or last.get("raw_rmse")
    )
    summary["last_h15"] = (
        horizon.get("15")
        or horizon.get(15)
        or last.get("h15_rmse")
        or last.get("val_raw_speed_rmse_15min")
    )
    summary["last_h30"] = (
        horizon.get("30")
        or horizon.get(30)
        or last.get("h30_rmse")
        or last.get("val_raw_speed_rmse_30min")
    )
    summary["last_h60"] = (
        horizon.get("60")
        or horizon.get(60)
        or last.get("h60_rmse")
        or last.get("val_raw_speed_rmse_60min")
    )
    summary["eta"] = last.get("eta") or last.get("eta_local")
    summary["config"] = {
        "dataset": run_config.get("dataset_name") or run_config.get("dataset"),
        "v_loss": run_config.get("v_loss"),
        "scheduler": run_config.get("scheduler_monitor"),
        "best_ckpt": run_config.get("best_checkpoint_monitor"),
        "seed": run_config.get("seed"),
        "topk": run_config.get("adaptive_topk") or run_config.get("speed_adaptive_topk"),
        "lr": run_config.get("lr"),
        "batch": run_config.get("batch_size"),
        "hist": run_config.get("hist_len"),
        "split": run_config.get("split_policy"),
        "graph": run_config.get("graph_source"),
    }
    return summary


def print_dashboard(rows: list[dict[str, Any]], root: Path) -> None:
    valid = [row for row in rows if as_float(row.get("best_val_rmse")) is not None]
    best = min(valid, key=lambda row: float(row["best_val_rmse"])) if valid else None
    best_label = best["label"] if best else None

    clear_screen()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{BOLD}PEMS-BAY MSE vs RMSE live watcher{RESET} | refresh={now}")
    print(f"root: {root}")
    print()

    if best:
        print(
            f"{GREEN}{BOLD}CURRENT GLOBAL BEST by Best Val RMSE: "
            f"{best['label']} | BestValRMSE={fmt(best['best_val_rmse'])} "
            f"@ epoch {fmt_epoch(best['best_epoch'])}{RESET}"
        )
    else:
        print(f"{YELLOW}CURRENT GLOBAL BEST: waiting for readable checkpoints{RESET}")
    print()

    header = (
        "RUN        PID      LIVE  EPOCH       LAST_VAL_RMSE  BEST_VAL_RMSE  "
        "BEST_EP  15/30/60_RMSE                     CONFIG"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        color = GREEN if row["label"] == best_label else RESET
        live = "yes" if row.get("pid_alive") else "no"
        epoch = f"{fmt_epoch(row.get('epoch'))}/{fmt_epoch(row.get('target_epoch'))}"
        horizons = f"{fmt(row.get('last_h15'), 3)} / {fmt(row.get('last_h30'), 3)} / {fmt(row.get('last_h60'), 3)}"
        cfg = row.get("config") or {}
        cfg_text = (
            f"loss={cfg.get('v_loss')} sched={cfg.get('scheduler')} "
            f"best={cfg.get('best_ckpt')} topk={cfg.get('topk')} "
            f"seed={cfg.get('seed')} lr={cfg.get('lr')}"
        )
        line = (
            f"{row['label']:<10} {str(row.get('pid')):<8} {live:<5} {epoch:<11} "
            f"{fmt(row.get('last_val_rmse')):<14} {fmt(row.get('best_val_rmse')):<14} "
            f"{fmt_epoch(row.get('best_epoch')):<8} {horizons:<32} {cfg_text}"
        )
        print(f"{color}{line}{RESET}")
        if row.get("load_error"):
            print(f"{RED}  checkpoint read warning: {row['load_error']}{RESET}")

    print()
    for row in rows:
        latest_mtime = row.get("latest_mtime")
        stamp = latest_mtime.strftime("%Y-%m-%d %H:%M:%S") if latest_mtime else "n/a"
        eta = row.get("eta") or "n/a"
        print(f"{row['label']}: checkpoint_updated={stamp} | eta={eta}")
    print()
    print("Tip: the green row is the current lowest Best Val RMSE among these two active runs.")
    print("Press Ctrl+C to stop this watcher only; training processes keep running.")
    try:
        sys.stdout.flush()
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    args = parser.parse_args()

    while True:
        rows = [summarize_run(args.root, spec) for spec in RUNS]
        print_dashboard(rows, args.root)
        if args.once:
            return 0
        time.sleep(max(2.0, args.interval))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nWatcher stopped.")
        raise SystemExit(0)
