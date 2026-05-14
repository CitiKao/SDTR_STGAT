from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local RTX 4090 batch sizes.")
    parser.add_argument("--train-task", default="dc", choices=["dc", "v", "joint"])
    parser.add_argument("--monitor-task", default="dc", choices=["dc", "v", "raw_dc"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-sizes", default="32,64,128,256,384,512")
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--base-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16", choices=["auto", "bf16", "fp32"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--num-st-blocks", type=int, default=2)
    parser.add_argument("--adaptive-topk", type=int, default=16)
    parser.add_argument("--max-time-steps", type=int, default=0)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--log-root", default="runs/local_4090_dc_batchsize")
    parser.add_argument("--summary-dir", default="local_4090/results")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    return parser.parse_args()


def parse_batch_sizes(raw_value: str) -> list[int]:
    batch_sizes: list[int] = []
    for chunk in raw_value.split(","):
        value = int(chunk.strip())
        if value <= 0:
            raise ValueError("batch sizes must be positive integers")
        batch_sizes.append(value)
    return batch_sizes


def query_total_gpu_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return None

    first_line = output.strip().splitlines()[0].strip()
    if not first_line:
        return None
    try:
        value = int(first_line)
    except ValueError:
        return None
    if value < 0 or value > 131072:
        return None
    return value


class GpuMemoryMonitor:
    def __init__(self, poll_seconds: float) -> None:
        self.poll_seconds = poll_seconds
        self.stop_event = threading.Event()
        self.samples: list[int] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            value = query_total_gpu_memory_mib()
            if value is not None:
                self.samples.append(value)
            self.stop_event.wait(self.poll_seconds)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    @property
    def peak_total_mib(self) -> int | None:
        return max(self.samples) if self.samples else None


def tail_text(text: str, max_lines: int = 25) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def print_safe(text: str) -> None:
    encoding = sys.stdout.encoding or "utf-8"
    safe_text = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe_text, end="")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_console_metrics(console_text: str) -> dict:
    summary: dict[str, float | dict[str, float] | None] = {
        "test_dc": None,
        "test_demand": None,
        "test_supply": None,
        "test_demand_rmse": None,
        "test_supply_rmse": None,
    }

    test_line = re.search(r"Test=DC=([0-9.]+)\s+\(D=([0-9.]+)\s+C=([0-9.]+)\)", console_text)
    if test_line:
        summary["test_dc"] = float(test_line.group(1))
        summary["test_demand"] = float(test_line.group(2))
        summary["test_supply"] = float(test_line.group(3))

    rmse_line = re.search(r"Test RMSE=D:([0-9.]+)\s+C:([0-9.]+)", console_text)
    if rmse_line:
        summary["test_demand_rmse"] = float(rmse_line.group(1))
        summary["test_supply_rmse"] = float(rmse_line.group(2))

    return summary


def choose_recommended(results: list[dict]) -> dict | None:
    successful = [
        result
        for result in results
        if result["status"] in {"ok", "ok_with_export_error"}
    ]
    if not successful:
        return None
    return min(
        successful,
        key=lambda item: (
            item.get("seconds_per_epoch", float("inf")),
            -item.get("batch_size", 0),
        ),
    )


def build_markdown(summary: dict) -> str:
    lines = [
        "# Local RTX 4090 DC Batch Size Benchmark",
        "",
        f"- Timestamp: `{summary['timestamp']}`",
        f"- Train task: `{summary['train_task']}`",
        f"- Monitor task: `{summary['monitor_task']}`",
        f"- Epochs per run: `{summary['epochs']}`",
        f"- Device: `{summary['device']}`",
        f"- Precision: `{summary['precision']}`",
        "",
        "## Recommendation",
        "",
    ]

    recommendation = summary.get("recommended")
    if recommendation is None:
        lines.append("No successful runs were recorded.")
    else:
        lines.extend(
            [
                f"- Recommended batch size: `{recommendation['batch_size']}`",
                f"- Avg seconds / epoch: `{recommendation['seconds_per_epoch']:.2f}`",
                f"- Approx train samples / second: `{recommendation['train_samples_per_second']:.2f}`",
                f"- Approx peak GPU memory increase: `{recommendation['peak_gpu_delta_mib']}` MiB",
            ]
        )

    lines.extend(["", "## Results", "", "| Batch | Status | LR | Sec/Epoch | Train samples/s | Peak GPU delta MiB | Last val metric |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for result in summary["results"]:
        metric = result.get("last_val_metric")
        metric_text = f"{metric:.5f}" if isinstance(metric, (int, float)) else "-"
        seconds_text = (
            f"{result['seconds_per_epoch']:.2f}"
            if isinstance(result.get("seconds_per_epoch"), (int, float))
            else "-"
        )
        throughput_text = (
            f"{result['train_samples_per_second']:.2f}"
            if isinstance(result.get("train_samples_per_second"), (int, float))
            else "-"
        )
        delta_text = result["peak_gpu_delta_mib"] if result.get("peak_gpu_delta_mib") is not None else "-"
        lines.append(
            f"| {result['batch_size']} | {result['status']} | {result['lr']:.6f} | {seconds_text} | {throughput_text} | {delta_text} | {metric_text} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    log_root = (repo_root / args.log_root).resolve()
    summary_dir = (repo_root / args.summary_dir).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: list[dict] = []

    for batch_size in batch_sizes:
        run_name = f"{args.train_task}_bs{batch_size}_ep{args.epochs}_{timestamp}"
        run_dir = log_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        lr = args.base_lr * math.sqrt(batch_size / args.base_batch_size)

        cmd = [
            sys.executable,
            "train_predictor.py",
            "--data-dir",
            args.data_dir,
            "--log-dir",
            str(run_dir),
            "--device",
            args.device,
            "--precision",
            args.precision,
            "--batch-size",
            str(batch_size),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(lr),
            "--num-workers",
            str(args.num_workers),
            "--val-interval",
            str(args.val_interval),
            "--num-st-blocks",
            str(args.num_st_blocks),
            "--adaptive-topk",
            str(args.adaptive_topk),
            "--train-task",
            args.train_task,
            "--monitor-task",
            args.monitor_task,
            "--log-interval",
            "1",
        ]
        if args.max_time_steps > 0:
            cmd.extend(["--max-time-steps", str(args.max_time_steps)])

        console_log_path = run_dir / "console.log"
        baseline_gpu_mib = query_total_gpu_memory_mib()
        monitor = GpuMemoryMonitor(args.poll_seconds)
        start_time = time.perf_counter()
        console_buffer: list[str] = []

        print(f"\n=== Running batch_size={batch_size} lr={lr:.6f} ===", flush=True)
        process = subprocess.Popen(
            cmd,
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        monitor.start()
        try:
            with open(console_log_path, "w", encoding="utf-8") as console_handle:
                assert process.stdout is not None
                for line in process.stdout:
                    console_buffer.append(line)
                    print_safe(line)
                    console_handle.write(line)
            return_code = process.wait()
        finally:
            monitor.stop()
        wall_seconds = time.perf_counter() - start_time

        peak_total_mib = monitor.peak_total_mib
        peak_delta_mib = None
        if baseline_gpu_mib is not None and peak_total_mib is not None:
            peak_delta_mib = max(peak_total_mib - baseline_gpu_mib, 0)

        result: dict = {
            "batch_size": batch_size,
            "lr": lr,
            "run_dir": str(run_dir),
            "console_log": str(console_log_path),
            "wall_seconds": wall_seconds,
            "peak_gpu_total_mib": peak_total_mib,
            "peak_gpu_delta_mib": peak_delta_mib,
        }

        history_path = run_dir / "predictor_log.json"
        meta_path = run_dir / "stgat_meta.json"
        metrics_path = run_dir / "predictor_test_metrics.json"
        console_text = "".join(console_buffer)
        console_metrics = extract_console_metrics(console_text)
        if not history_path.exists() or not meta_path.exists():
            result["status"] = "failed"
            result["return_code"] = return_code
            result["error_tail"] = tail_text(console_text)
            results.append(result)
            continue

        history = load_json(history_path)
        meta = load_json(meta_path)
        metrics = load_json(metrics_path) if metrics_path.exists() else {}
        last_record = history[-1] if history else {}
        total_elapsed = float(last_record.get("elapsed", wall_seconds))
        seconds_per_epoch = total_elapsed / max(args.epochs, 1)
        train_count = int(meta.get("split_counts", {}).get("train", 0))
        train_samples_per_second = (
            (train_count * args.epochs) / total_elapsed if total_elapsed > 0 and train_count > 0 else None
        )

        if args.monitor_task == "raw_dc":
            last_val_metric = last_record.get("val_raw_dc")
        elif args.monitor_task == "dc":
            last_val_metric = last_record.get("val_dc")
        elif args.monitor_task == "v":
            last_val_metric = last_record.get("val_v")
        else:
            last_val_metric = None

        result.update(
            {
                "status": "ok" if return_code == 0 else "ok_with_export_error",
                "return_code": return_code,
                "total_elapsed": total_elapsed,
                "seconds_per_epoch": seconds_per_epoch,
                "train_samples": train_count,
                "train_samples_per_second": train_samples_per_second,
                "last_val_metric": last_val_metric,
                "selected_checkpoint_metric": metrics.get("selected_checkpoint_metric"),
                "test_raw_metrics": metrics.get("raw_metrics"),
                "console_metrics": console_metrics,
            }
        )
        if return_code != 0:
            result["error_tail"] = tail_text(console_text)
        results.append(result)

    recommended = choose_recommended(results)
    summary = {
        "timestamp": timestamp,
        "train_task": args.train_task,
        "monitor_task": args.monitor_task,
        "epochs": args.epochs,
        "device": args.device,
        "precision": args.precision,
        "batch_sizes": batch_sizes,
        "results": results,
        "recommended": recommended,
    }

    summary_json_path = summary_dir / f"batchsize_benchmark_{timestamp}.json"
    summary_md_path = summary_dir / f"batchsize_benchmark_{timestamp}.md"
    latest_json_path = summary_dir / "latest_batchsize_benchmark.json"
    latest_md_path = summary_dir / "latest_batchsize_benchmark.md"

    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(summary_md_path, "w", encoding="utf-8") as handle:
        handle.write(build_markdown(summary))
    with open(latest_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(latest_md_path, "w", encoding="utf-8") as handle:
        handle.write(build_markdown(summary))

    print(f"\nSaved summary to {summary_json_path}")
    print(f"Saved summary to {summary_md_path}")
    if recommended is not None:
        print(
            "Recommended batch size:",
            recommended["batch_size"],
            f"(seconds/epoch={recommended['seconds_per_epoch']:.2f},",
            f"peak_gpu_delta_mib={recommended['peak_gpu_delta_mib']})",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
