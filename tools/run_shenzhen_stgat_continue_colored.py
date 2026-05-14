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


def env() -> dict[str, str]:
    values = os.environ.copy()
    values["PYTHONUNBUFFERED"] = "1"
    values["PYTHONUTF8"] = "1"
    values["PYTHONIOENCODING"] = "utf-8"
    return values


def command_text(command: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def group_prefix(index: int, total: int, *, colors: bool) -> str:
    return color(f"[\u7b2c {index}/{total} \u7ec4]", YELLOW, enabled=colors)


def parse_ordinals(text: str) -> list[str]:
    ordinals = [part.strip().lstrip("0") or "0" for part in text.split(",") if part.strip()]
    if not ordinals:
        raise ValueError("--ordinals cannot be empty")
    return ordinals


def load_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {str(int(float(row["ordinal"]))): row for row in rows}


def checkpoint_path(source_run_dir: Path, checkpoint: str) -> Path:
    if checkpoint == "final":
        return source_run_dir / "stgat_final.pt"
    if checkpoint == "best":
        return source_run_dir / "stgat_best.pt"
    if checkpoint == "best_raw_dc":
        return source_run_dir / "stgat_best_raw_dc.pt"
    candidate = Path(checkpoint)
    return candidate if candidate.is_absolute() else source_run_dir / checkpoint


def read_elapsed_from_source(source_run_dir: Path) -> float | None:
    log_path = source_run_dir / "predictor_log.json"
    if not log_path.exists():
        return None
    history = json.loads(log_path.read_text(encoding="utf-8"))
    if not history:
        return None
    return float(history[-1].get("elapsed", 0.0))


def read_result(run_dir: Path) -> dict[str, Any] | None:
    metrics_path = run_dir / "predictor_test_metrics.json"
    log_path = run_dir / "predictor_log.json"
    if not metrics_path.exists() or not log_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = json.loads(log_path.read_text(encoding="utf-8"))
    raw = metrics.get("raw_metrics", {})
    demand = raw.get("demand")
    supply = raw.get("supply")
    if not isinstance(demand, dict) or not isinstance(supply, dict):
        return None
    selection_score = float(metrics.get("selected_checkpoint_metric"))
    best_epoch = 0
    best_distance = float("inf")
    for record in history:
        val = record.get("val_raw_dc")
        if val is None:
            continue
        distance = abs(float(val) - selection_score)
        if distance < best_distance:
            best_distance = distance
            best_epoch = int(record.get("epoch", 0))
    return {
        "selection_score": selection_score,
        "test_raw_dc": float(demand["rmse"]) + float(supply["rmse"]),
        "demand_rmse": float(demand["rmse"]),
        "supply_rmse": float(supply["rmse"]),
        "demand_mae": float(demand["mae"]),
        "supply_mae": float(supply["mae"]),
        "completed_epochs": int(len(history)),
        "best_epoch": int(best_epoch),
        "end_epoch": int(max((record.get("epoch", 0) for record in history), default=0)),
    }


def build_command(
    *,
    row: dict[str, str],
    data_dir: Path,
    run_dir: Path,
    init_checkpoint: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "train_predictor.py",
        "--data-dir",
        str(data_dir),
        "--log-dir",
        str(run_dir),
        "--hist-len",
        row["hist_len"],
        "--pred-horizon",
        row["pred_horizon"],
        "--report-horizons-minutes",
        args.report_horizons_minutes,
        "--epochs",
        str(args.epochs),
        "--epoch-offset",
        str(args.epoch_offset),
        "--init-checkpoint",
        str(init_checkpoint),
        "--batch-size",
        row["batch_size"],
        "--lr",
        row["lr"],
        "--hidden-dim",
        row["hidden_dim"],
        "--num-heads",
        row["num_heads"],
        "--num-st-blocks",
        row["num_st_blocks"],
        "--adaptive-topk",
        row["adaptive_topk"],
        "--lambda1",
        row["lambda1"],
        "--lambda2",
        row["lambda2"],
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
    return command


def append_status(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows_by_val = sorted(rows, key=lambda row: float(row.get("selection_score", float("inf"))))
    rows_by_test = sorted(rows, key=lambda row: float(row.get("test_raw_dc", float("inf"))))
    for rank, row in enumerate(rows_by_val, start=1):
        row["rank_by_val"] = rank
    test_ranks = {id(row): rank for rank, row in enumerate(rows_by_test, start=1)}
    for row in rows:
        row["rank_by_test"] = test_ranks[id(row)]

    fields = [
        "rank_by_val",
        "rank_by_test",
        "source_ordinal",
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
        "end_epoch",
        "source_val_raw_dc",
        "source_test_raw_dc",
        "source_best_epoch",
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
        "source_run_dir",
        "run_dir",
    ]
    with (output_dir / "continue_summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_by_val)

    lines = [
        "# Shenzhen STGAT Continuation Summary",
        "",
        "| Rank(val) | Rank(test) | Source | Config | Val rawDC | Test rawDC | D RMSE | C RMSE | Best Ep | Source Test | Params |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows_by_val:
        if row.get("status") != "ok":
            continue
        params = (
            f"lr={row['lr']}, bs={row['batch_size']}, h={row['hidden_dim']}, "
            f"heads={row['num_heads']}, blocks={row['num_st_blocks']}, "
            f"topk={row['adaptive_topk']}, lambda=({row['lambda1']},{row['lambda2']})"
        )
        lines.append(
            f"| {row['rank_by_val']} | {row['rank_by_test']} | {row['source_ordinal']} | `{row['name']}` | "
            f"{float(row['selection_score']):.5f} | {float(row['test_raw_dc']):.5f} | "
            f"{float(row['demand_rmse']):.5f} | {float(row['supply_rmse']):.5f} | "
            f"{row['best_epoch']} | {float(row['source_test_raw_dc']):.5f} | {params} |"
        )
    (output_dir / "continue_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(
    *,
    row: dict[str, str],
    index: int,
    total: int,
    output_dir: Path,
    data_dir: Path,
    args: argparse.Namespace,
    colors: bool,
    completed_seconds: list[float],
    result_rows: list[dict[str, Any]],
    global_best: dict[str, Any],
) -> None:
    source_ordinal = str(int(float(row["ordinal"])))
    name = row["name"]
    prefix = group_prefix(index, total, colors=colors)
    source_run_dir = Path(row["run_dir"])
    init_checkpoint = checkpoint_path(source_run_dir, args.source_checkpoint)
    run_dir = output_dir / f"{int(source_ordinal):03d}_{name}_continue_{args.epochs}_from_{args.source_checkpoint}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = read_result(run_dir)
    if result and args.resume:
        out = {
            **row,
            **result,
            "source_ordinal": source_ordinal,
            "source_val_raw_dc": row["selection_score"],
            "source_test_raw_dc": row["test_raw_dc"],
            "source_best_epoch": row["best_epoch"],
            "source_run_dir": str(source_run_dir),
            "run_dir": str(run_dir),
            "status": "ok",
        }
        result_rows.append(out)
        print(
            f"{prefix} SKIP existing {name} "
            + color(f"val={result['selection_score']:.5f}", GREEN, enabled=colors)
            + f" test={result['test_raw_dc']:.5f}",
            flush=True,
        )
        return

    if not init_checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint for group {source_ordinal}: {init_checkpoint}")

    command = build_command(
        row=row,
        data_dir=data_dir,
        run_dir=run_dir,
        init_checkpoint=init_checkpoint,
        args=args,
    )
    (run_dir / "command.txt").write_text(command_text(command) + "\n", encoding="utf-8")

    print(color("=" * 96, CYAN, enabled=colors), flush=True)
    print(f"{prefix} continue source={source_ordinal} {name}", flush=True)
    print(
        color(
            "PARAM "
            f"lr={row['lr']} bs={row['batch_size']} hidden={row['hidden_dim']} heads={row['num_heads']} "
            f"blocks={row['num_st_blocks']} topk={row['adaptive_topk']} "
            f"lambda=({row['lambda1']},{row['lambda2']}) "
            f"epochs={args.epoch_offset + 1}-{args.epoch_offset + args.epochs} "
            f"init={init_checkpoint}",
            RED,
            enabled=colors,
        ),
        flush=True,
    )
    print(
        f"{prefix} source val={float(row['selection_score']):.5f} "
        f"test={float(row['test_raw_dc']):.5f} best_epoch={row['best_epoch']}",
        flush=True,
    )

    start = time.time()
    best_val = float("inf")
    best_epoch = 0
    epoch_pattern = re.compile(r"\[Ep\s*(\d+)\]")
    val_pattern = re.compile(r"ValRawDC=([0-9]+(?:\.[0-9]+)?)")
    dc_pattern = re.compile(r"\(D:([0-9]+(?:\.[0-9]+)?)\s+C:([0-9]+(?:\.[0-9]+)?)\)")
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
    with (run_dir / "stdout.log").open("w", encoding="utf-8") as log_handle:
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            text = line.rstrip()
            epoch_match = epoch_pattern.search(text)
            if not epoch_match:
                if text:
                    print(f"{prefix} {text}", flush=True)
                continue
            display_epoch = int(epoch_match.group(1))
            local_epoch = max(display_epoch - args.epoch_offset, 1)
            elapsed = time.time() - start
            epoch_sec = elapsed / max(local_epoch, 1)
            eta_method = epoch_sec * max(args.epochs - local_epoch, 0)
            average_completed = (
                sum(completed_seconds) / len(completed_seconds)
                if completed_seconds
                else epoch_sec * args.epochs
            )
            eta_total = eta_method + max(total - index, 0) * average_completed
            val_match = val_pattern.search(text)
            improved = False
            val_text = "val_raw_dc=waiting"
            if val_match:
                val = float(val_match.group(1))
                improved = val < best_val
                if improved:
                    best_val = val
                    best_epoch = display_epoch
                val_text = f"val_raw_dc={val:.5f}"
                dc_match = dc_pattern.search(text)
                if dc_match:
                    val_text += f" D={float(dc_match.group(1)):.4f} C={float(dc_match.group(2)):.4f}"
            global_text = (
                f"global_best_val={float(global_best['selection_score']):.5f} ({global_best['name']})"
                if global_best
                else "global_best_val=none"
            )
            output = (
                f"{prefix} Ep {display_epoch:03d}/{args.epoch_offset + args.epochs:03d} {val_text} "
                f"{color(f'best_val={best_val:.5f}', GREEN, enabled=colors)} "
                f"{color(f'best_epoch={best_epoch}', ORANGE, enabled=colors)} "
                f"epoch={epoch_sec:.1f}s ETA={duration(eta_method)} TOTAL_ETA={duration(eta_total)} "
                f"{color(global_text, GREEN, enabled=colors)}"
            )
            if improved:
                output += " " + color("NEW RUN BEST", GREEN, enabled=colors)
            print(output, flush=True)
            append_status(
                output_dir / "continue_status.jsonl",
                {
                    "event": "epoch",
                    "source_ordinal": source_ordinal,
                    "name": name,
                    "ordinal": index,
                    "total": total,
                    "epoch": display_epoch,
                    "local_epoch": local_epoch,
                    "end_epoch": args.epoch_offset + args.epochs,
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
        out = {
            **row,
            "source_ordinal": source_ordinal,
            "source_val_raw_dc": row["selection_score"],
            "source_test_raw_dc": row["test_raw_dc"],
            "source_best_epoch": row["best_epoch"],
            "source_run_dir": str(source_run_dir),
            "run_dir": str(run_dir),
            "status": f"failed:{return_code}",
        }
        result_rows.append(out)
        write_summary(output_dir, result_rows)
        print(color(f"{prefix} FAILED {name} exit={return_code}", RED, enabled=colors), flush=True)
        return

    result = read_result(run_dir)
    if result is None:
        out = {
            **row,
            "source_ordinal": source_ordinal,
            "source_val_raw_dc": row["selection_score"],
            "source_test_raw_dc": row["test_raw_dc"],
            "source_best_epoch": row["best_epoch"],
            "source_run_dir": str(source_run_dir),
            "run_dir": str(run_dir),
            "status": "failed:no_metrics",
        }
        result_rows.append(out)
        write_summary(output_dir, result_rows)
        print(color(f"{prefix} FAILED {name}: no metrics", RED, enabled=colors), flush=True)
        return

    out = {
        **row,
        **result,
        "source_ordinal": source_ordinal,
        "source_val_raw_dc": row["selection_score"],
        "source_test_raw_dc": row["test_raw_dc"],
        "source_best_epoch": row["best_epoch"],
        "source_run_dir": str(source_run_dir),
        "run_dir": str(run_dir),
        "status": "ok",
    }
    result_rows.append(out)
    if float(result["selection_score"]) < float(global_best.get("selection_score", float("inf"))):
        global_best.clear()
        global_best.update({"name": name, **result})
        global_message = color("NEW GLOBAL BEST BY VAL", GREEN, enabled=colors)
    else:
        global_message = ""
    test_delta = float(result["test_raw_dc"]) - float(row["test_raw_dc"])
    print(
        f"{prefix} "
        + color(
            f"DONE {name} val={result['selection_score']:.5f} test_raw={result['test_raw_dc']:.5f} "
            f"D_RMSE={result['demand_rmse']:.5f} C_RMSE={result['supply_rmse']:.5f} "
            f"test_delta={test_delta:+.5f}",
            GREEN if test_delta < 0 else ORANGE,
            enabled=colors,
        )
        + " "
        + color(f"best_epoch={result['best_epoch']}", ORANGE, enabled=colors)
        + (" " + global_message if global_message else ""),
        flush=True,
    )
    print(f"{prefix} elapsed={duration(elapsed)}", flush=True)
    write_summary(output_dir, result_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue selected Shenzhen STGAT tuning runs with colored progress.")
    parser.add_argument("--data-dir", default="data/shenzhen_dc")
    parser.add_argument("--summary", default="runs/shenzhen_stgat_tuning_20260508_023118/sweep_summary.tsv")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--ordinals", default="6,24,26")
    parser.add_argument("--source-checkpoint", default="final", help="final, best, best_raw_dc, or a checkpoint filename.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--epoch-offset", type=int, default=100)
    parser.add_argument("--report-horizons-minutes", default="15,30,60")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    colors = not args.no_color
    summary_path = Path(args.summary)
    data_dir = Path(args.data_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or f"runs/shenzhen_stgat_continue_{stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(summary_path)
    selected_ordinals = parse_ordinals(args.ordinals)
    selected_rows: list[dict[str, str]] = []
    for ordinal in selected_ordinals:
        if ordinal not in summary:
            raise KeyError(f"ordinal {ordinal} not found in {summary_path}")
        selected_rows.append(summary[ordinal])

    estimates = [
        read_elapsed_from_source(Path(row["run_dir"]))
        for row in selected_rows
    ]
    known_estimates = [value for value in estimates if value is not None and value > 0]
    estimated_total = sum(known_estimates)
    if len(known_estimates) != len(selected_rows) and known_estimates:
        estimated_total += (len(selected_rows) - len(known_estimates)) * (
            sum(known_estimates) / len(known_estimates)
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "summary": str(summary_path),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "ordinals": selected_ordinals,
        "source_checkpoint": args.source_checkpoint,
        "epochs": args.epochs,
        "epoch_offset": args.epoch_offset,
        "estimated_total_sec_from_source_logs": estimated_total,
        "selected": selected_rows,
    }
    (output_dir / "continue_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(color("=" * 96, CYAN, enabled=colors), flush=True)
    print(color(f"Shenzhen STGAT continuation output={output_dir}", CYAN, enabled=colors), flush=True)
    print(
        color(
            f"Continue ordinals={','.join(selected_ordinals)} checkpoint={args.source_checkpoint} "
            f"epochs={args.epoch_offset + 1}-{args.epoch_offset + args.epochs}",
            RED,
            enabled=colors,
        ),
        flush=True,
    )
    if estimated_total > 0:
        print(color(f"Estimated total time from previous logs: {duration(estimated_total)}", ORANGE, enabled=colors), flush=True)

    if args.dry_run:
        print(color("DRY RUN", ORANGE, enabled=colors), flush=True)
        for index, row in enumerate(selected_rows, start=1):
            source_run_dir = Path(row["run_dir"])
            init_checkpoint = checkpoint_path(source_run_dir, args.source_checkpoint)
            run_dir = output_dir / f"{int(float(row['ordinal'])):03d}_{row['name']}_continue_{args.epochs}_from_{args.source_checkpoint}"
            command = build_command(
                row=row,
                data_dir=data_dir,
                run_dir=run_dir,
                init_checkpoint=init_checkpoint,
                args=args,
            )
            print(f"{group_prefix(index, len(selected_rows), colors=colors)} {row['name']} :: {command_text(command)}")
        return

    result_rows: list[dict[str, Any]] = []
    completed_seconds: list[float] = []
    global_best: dict[str, Any] = {}
    for index, row in enumerate(selected_rows, start=1):
        run_one(
            row=row,
            index=index,
            total=len(selected_rows),
            output_dir=output_dir,
            data_dir=data_dir,
            args=args,
            colors=colors,
            completed_seconds=completed_seconds,
            result_rows=result_rows,
            global_best=global_best,
        )
    write_summary(output_dir, result_rows)
    print(color("=" * 96, CYAN, enabled=colors), flush=True)
    print(color(f"Continuation finished. Summary: {output_dir / 'continue_summary.md'}", GREEN, enabled=colors), flush=True)


if __name__ == "__main__":
    main()
