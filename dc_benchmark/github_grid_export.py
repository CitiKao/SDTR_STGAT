from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .dataset import DEFAULT_DATASET_DIR, load_dc_benchmark
    from .github_reimplementation_sources import get_github_reimplementation_info
except ImportError:  # pragma: no cover - direct script execution
    from dc_benchmark.dataset import DEFAULT_DATASET_DIR, load_dc_benchmark
    from dc_benchmark.github_reimplementation_sources import get_github_reimplementation_info


GITHUB_GRID_SCHEMA = "dc_github_grid_adapter_v1"
GRID_METHODS = {"convlstm", "st_resnet"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def square_grid_shape(num_nodes: int) -> tuple[int, int]:
    side = int(math.ceil(math.sqrt(num_nodes)))
    return side, side


def node_grid_layout(num_nodes: int) -> dict[str, Any]:
    height, width = square_grid_shape(num_nodes)
    node_grid_index = np.asarray([[idx // width, idx % width] for idx in range(num_nodes)], dtype=np.int32)
    mask = np.zeros((height, width), dtype=bool)
    for row, col in node_grid_index:
        mask[int(row), int(col)] = True
    return {
        "height": height,
        "width": width,
        "node_grid_index": node_grid_index,
        "mask": mask,
    }


def sequence_to_grid(sequence: np.ndarray, *, layout: dict[str, Any]) -> np.ndarray:
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError(f"Expected sequence shape (steps, nodes, 2), got {values.shape}")
    grid = np.zeros((values.shape[0], 2, int(layout["height"]), int(layout["width"])), dtype=np.float32)
    node_grid_index = np.asarray(layout["node_grid_index"], dtype=np.int32)
    if values.shape[1] != node_grid_index.shape[0]:
        raise ValueError(f"Node count mismatch: values={values.shape[1]}, layout={node_grid_index.shape[0]}")
    for node_idx, (row, col) in enumerate(node_grid_index):
        grid[:, :, int(row), int(col)] = values[:, node_idx, :]
    return grid


def _window_indices(benchmark: dict[str, Any], split: str, max_samples: int) -> list[int]:
    indices = list(benchmark["splits"]["indices"][split])
    if max_samples > 0:
        indices = indices[: int(max_samples)]
    return [int(idx) for idx in indices]


def _export_convlstm_grid(
    benchmark: dict[str, Any],
    *,
    method_dir: Path,
    max_samples_per_split: int,
) -> dict[str, Any]:
    manifest = benchmark["manifest"]
    hist_len = int(manifest["hist_len"])
    pred_horizon = int(manifest["pred_horizon"])
    layout = node_grid_layout(int(manifest["shapes"]["targets_dc"][1]))
    targets = np.asarray(benchmark["targets"], dtype=np.float32)

    split_shapes: dict[str, dict[str, list[int]]] = {}
    for split in ("train", "val", "test"):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for idx in _window_indices(benchmark, split, max_samples_per_split):
            target_start = idx + hist_len
            xs.append(sequence_to_grid(targets[idx:target_start], layout=layout))
            ys.append(sequence_to_grid(targets[target_start:target_start + pred_horizon], layout=layout))
        x = np.stack(xs, axis=0).astype(np.float32)
        y = np.stack(ys, axis=0).astype(np.float32)
        np.savez_compressed(method_dir / f"{split}.npz", x=x, y=y)
        split_shapes[split] = {"x": list(x.shape), "y": list(y.shape)}
    return split_shapes


def _pad_time_channels(grid_sequence: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0:
        raise ValueError("steps must be positive")
    channels = grid_sequence[-steps:] if grid_sequence.shape[0] >= steps else grid_sequence
    if channels.shape[0] < steps:
        pad = np.repeat(channels[:1], steps - channels.shape[0], axis=0)
        channels = np.concatenate([pad, channels], axis=0)
    return channels.reshape(steps * channels.shape[1], channels.shape[2], channels.shape[3])


def _export_st_resnet_grid(
    benchmark: dict[str, Any],
    *,
    method_dir: Path,
    max_samples_per_split: int,
    len_closeness: int,
    len_period: int,
    len_trend: int,
) -> dict[str, Any]:
    manifest = benchmark["manifest"]
    hist_len = int(manifest["hist_len"])
    pred_horizon = int(manifest["pred_horizon"])
    layout = node_grid_layout(int(manifest["shapes"]["targets_dc"][1]))
    targets = np.asarray(benchmark["targets"], dtype=np.float32)

    split_shapes: dict[str, dict[str, list[int]]] = {}
    for split in ("train", "val", "test"):
        xc_rows: list[np.ndarray] = []
        xp_rows: list[np.ndarray] = []
        xt_rows: list[np.ndarray] = []
        y_rows: list[np.ndarray] = []
        for idx in _window_indices(benchmark, split, max_samples_per_split):
            target_start = idx + hist_len
            history_grid = sequence_to_grid(targets[idx:target_start], layout=layout)
            xc_rows.append(_pad_time_channels(history_grid, len_closeness))
            xp_rows.append(_pad_time_channels(history_grid[:-len_closeness] if hist_len > len_closeness else history_grid, len_period))
            prefix = history_grid[: max(1, hist_len - len_closeness - len_period)]
            xt_rows.append(_pad_time_channels(prefix, len_trend))
            y_rows.append(sequence_to_grid(targets[target_start:target_start + pred_horizon], layout=layout))
        xc = np.stack(xc_rows, axis=0).astype(np.float32)
        xp = np.stack(xp_rows, axis=0).astype(np.float32)
        xt = np.stack(xt_rows, axis=0).astype(np.float32)
        y = np.stack(y_rows, axis=0).astype(np.float32)
        np.savez_compressed(method_dir / f"{split}.npz", xc=xc, xp=xp, xt=xt, y=y, y_first=y[:, 0])
        split_shapes[split] = {
            "xc": list(xc.shape),
            "xp": list(xp.shape),
            "xt": list(xt.shape),
            "y": list(y.shape),
            "y_first": list(y[:, 0].shape),
        }
    return split_shapes


def export_github_grid_dataset(
    *,
    method_id: str,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_dir: str | Path = Path("data") / "dc_benchmark" / "github_grid",
    max_samples_per_split: int = 0,
    project_root: str | Path = ".",
    len_closeness: int = 3,
    len_period: int = 1,
    len_trend: int = 1,
) -> Path:
    if method_id not in GRID_METHODS:
        raise ValueError(f"Unsupported GitHub grid method: {method_id}")
    benchmark = load_dc_benchmark(dataset_dir)
    output = Path(output_dir)
    method_dir = output / method_id
    method_dir.mkdir(parents=True, exist_ok=True)

    manifest = benchmark["manifest"]
    num_nodes = int(manifest["shapes"]["targets_dc"][1])
    layout = node_grid_layout(num_nodes)
    np.save(method_dir / "node_grid_index.npy", layout["node_grid_index"])
    np.save(method_dir / "node_grid_mask.npy", layout["mask"])

    if method_id == "convlstm":
        split_shapes = _export_convlstm_grid(
            benchmark,
            method_dir=method_dir,
            max_samples_per_split=max_samples_per_split,
        )
        model_inputs = {"x": "(samples, hist_len, 2, grid_h, grid_w)", "y": "(samples, pred_horizon, 2, grid_h, grid_w)"}
    else:
        split_shapes = _export_st_resnet_grid(
            benchmark,
            method_dir=method_dir,
            max_samples_per_split=max_samples_per_split,
            len_closeness=len_closeness,
            len_period=len_period,
            len_trend=len_trend,
        )
        model_inputs = {
            "xc": "(samples, len_closeness * 2, grid_h, grid_w)",
            "xp": "(samples, len_period * 2, grid_h, grid_w)",
            "xt": "(samples, len_trend * 2, grid_h, grid_w)",
            "y_first": "(samples, 2, grid_h, grid_w)",
        }

    source_info = get_github_reimplementation_info(method_id, project_root=project_root)
    grid_manifest = {
        "schema_version": GITHUB_GRID_SCHEMA,
        "method_id": method_id,
        "claim_type": source_info["claim_type"],
        "not_official_paper_result": True,
        "allowed_modifications": ["data_loader", "output_head"],
        "source": source_info,
        "benchmark_manifest": {
            "dataset_name": manifest["dataset_name"],
            "schema_version": manifest["schema_version"],
            "split_hash": manifest["split_hash"],
            "hist_len": manifest["hist_len"],
            "pred_horizon": manifest["pred_horizon"],
            "split_counts": manifest["split_counts"],
        },
        "grid_layout": {
            "num_nodes": num_nodes,
            "height": int(layout["height"]),
            "width": int(layout["width"]),
            "valid_grid_cells": int(layout["mask"].sum()),
            "empty_grid_cells": int(layout["mask"].size - layout["mask"].sum()),
            "node_grid_index_file": "node_grid_index.npy",
            "node_grid_mask_file": "node_grid_mask.npy",
            "layout_note": "Nodes are placed by stable node index order into a square pseudo-grid; empty cells are masked.",
        },
        "model_inputs": model_inputs,
        "split_shapes": split_shapes,
    }
    if method_id == "st_resnet":
        grid_manifest["st_resnet_temporal_slices"] = {
            "len_closeness": int(len_closeness),
            "len_period": int(len_period),
            "len_trend": int(len_trend),
            "note": "Period/trend tensors are derived from earlier available history slices, not original TaxiBJ day/week files.",
        }
    _write_json(method_dir / "github_grid_manifest.json", grid_manifest)
    return method_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export pseudo-grid data for GitHub reimplementation baselines.")
    parser.add_argument("--method", required=True, choices=sorted(GRID_METHODS))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(Path("data") / "dc_benchmark" / "github_grid"))
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--len-closeness", type=int, default=3)
    parser.add_argument("--len-period", type=int, default=1)
    parser.add_argument("--len-trend", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = export_github_grid_dataset(
        method_id=args.method,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_samples_per_split=args.max_samples_per_split,
        project_root=args.project_root,
        len_closeness=args.len_closeness,
        len_period=args.len_period,
        len_trend=args.len_trend,
    )
    print(f"github reimplementation grid dataset exported to {output}")


if __name__ == "__main__":
    main()
