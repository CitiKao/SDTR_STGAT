from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .dataset import DEFAULT_DATASET_DIR, load_dc_benchmark
    from .official_sources import get_official_source_info, require_official_method
except ImportError:  # pragma: no cover - direct script execution
    from dc_benchmark.dataset import DEFAULT_DATASET_DIR, load_dc_benchmark
    from dc_benchmark.official_sources import get_official_source_info, require_official_method


OFFICIAL_DATASET_SCHEMA = "dc_official_adapter_v1"
TARGET_CHANNELS = {"demand": 0, "supply": 1}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _time_in_day(benchmark: dict[str, Any]) -> np.ndarray:
    meta = benchmark["time_meta"]
    if {"hour", "minute"}.issubset(meta.columns):
        return ((meta["hour"].to_numpy(dtype=np.float32) * 60.0 + meta["minute"].to_numpy(dtype=np.float32)) / 1440.0)
    if "slot" in meta.columns:
        slot = meta["slot"].to_numpy(dtype=np.float32)
        return slot / max(float(np.nanmax(slot) + 1.0), 1.0)
    return np.linspace(0.0, 1.0, num=len(meta), endpoint=False, dtype=np.float32)


def _target_windows_for_channel(
    benchmark: dict[str, Any],
    split: str,
    channel: str,
    *,
    max_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    manifest = benchmark["manifest"]
    hist_len = int(manifest["hist_len"])
    pred_horizon = int(manifest["pred_horizon"])
    channel_idx = TARGET_CHANNELS[channel]
    indices = list(benchmark["splits"]["indices"][split])
    if max_samples > 0:
        indices = indices[: int(max_samples)]
    targets = np.asarray(benchmark["targets"], dtype=np.float32)
    tod = _time_in_day(benchmark).astype(np.float32)

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for idx in indices:
        idx = int(idx)
        target_start = idx + hist_len
        hist_slice = slice(idx, target_start)
        pred_slice = slice(target_start, target_start + pred_horizon)

        hist_value = targets[hist_slice, :, channel_idx]
        pred_value = targets[pred_slice, :, channel_idx]
        hist_tod = np.repeat(tod[hist_slice, None], hist_value.shape[1], axis=1)
        pred_tod = np.repeat(tod[pred_slice, None], pred_value.shape[1], axis=1)

        x_rows.append(np.stack([hist_value, hist_tod], axis=-1))
        y_rows.append(np.stack([pred_value, pred_tod], axis=-1))
    return np.stack(x_rows, axis=0).astype(np.float32), np.stack(y_rows, axis=0).astype(np.float32)


def _write_adj_pickle(path: Path, adjacency: np.ndarray) -> None:
    num_nodes = int(adjacency.shape[0])
    sensor_ids = [str(i) for i in range(num_nodes)]
    sensor_id_to_ind = {sensor_id: idx for idx, sensor_id in enumerate(sensor_ids)}
    with path.open("wb") as fh:
        pickle.dump((sensor_ids, sensor_id_to_ind, np.asarray(adjacency, dtype=np.float32)), fh, protocol=pickle.HIGHEST_PROTOCOL)


def _write_dcrnn_config(path: Path, dataset_dir: Path, adj_path: Path, benchmark: dict[str, Any], channel: str) -> None:
    manifest = benchmark["manifest"]
    num_nodes = int(manifest["shapes"]["targets_dc"][1])
    hist_len = int(manifest["hist_len"])
    pred_horizon = int(manifest["pred_horizon"])
    text = f"""---
base_dir: runs/dc_benchmark_official/dcrnn/{channel}
log_level: INFO
data:
  batch_size: 64
  dataset_dir: {dataset_dir.as_posix()}
  test_batch_size: 64
  val_batch_size: 64
  graph_pkl_filename: {adj_path.as_posix()}
model:
  cl_decay_steps: 2000
  filter_type: dual_random_walk
  horizon: {pred_horizon}
  input_dim: 2
  l1_decay: 0
  max_diffusion_step: 2
  num_nodes: {num_nodes}
  num_rnn_layers: 2
  output_dim: 1
  rnn_units: 64
  seq_len: {hist_len}
  use_curriculum_learning: true
train:
  base_lr: 0.01
  dropout: 0
  epoch: 0
  epochs: 100
  epsilon: 1.0e-3
  global_step: 0
  lr_decay_ratio: 0.1
  max_grad_norm: 5
  max_to_keep: 100
  min_learning_rate: 2.0e-06
  optimizer: adam
  patience: 15
  steps: [20, 30, 40, 50]
  test_every_n_epochs: 10
"""
    path.write_text(text, encoding="utf-8")


def _export_dcrnn_like(
    benchmark: dict[str, Any],
    *,
    method_id: str,
    output_dir: Path,
    max_samples_per_split: int = 0,
) -> dict[str, Any]:
    method_dir = output_dir / method_id
    method_dir.mkdir(parents=True, exist_ok=True)
    adj_path = method_dir / "adj_mx.pkl"
    _write_adj_pickle(adj_path, np.asarray(benchmark["adjacency"], dtype=np.float32))
    x_offsets = np.arange(-(int(benchmark["manifest"]["hist_len"]) - 1), 1, dtype=np.int32).reshape(-1, 1)
    y_offsets = np.arange(1, int(benchmark["manifest"]["pred_horizon"]) + 1, dtype=np.int32).reshape(-1, 1)

    channel_payloads: dict[str, Any] = {}
    num_nodes = int(benchmark["manifest"]["shapes"]["targets_dc"][1])
    hist_len = int(benchmark["manifest"]["hist_len"])
    pred_horizon = int(benchmark["manifest"]["pred_horizon"])
    for channel in TARGET_CHANNELS:
        channel_dir = method_dir / channel
        channel_dir.mkdir(parents=True, exist_ok=True)
        split_shapes: dict[str, dict[str, list[int]]] = {}
        for split in ("train", "val", "test"):
            x, y = _target_windows_for_channel(
                benchmark,
                split,
                channel,
                max_samples=max_samples_per_split,
            )
            np.savez_compressed(channel_dir / f"{split}.npz", x=x, y=y, x_offsets=x_offsets, y_offsets=y_offsets)
            split_shapes[split] = {"x": list(x.shape), "y": list(y.shape)}
        if method_id == "dcrnn":
            _write_dcrnn_config(channel_dir / "dcrnn_config.yaml", channel_dir, adj_path, benchmark, channel)
        channel_payloads[channel] = {
            "dataset_dir": str(channel_dir),
            "split_shapes": split_shapes,
            "target_channel": channel,
            "target_channel_index": TARGET_CHANNELS[channel],
            "input_dim": 2,
            "output_dim": 1,
        }
    return {
        "method_dataset_dir": str(method_dir),
        "adjacency_pickle": str(adj_path),
        "num_nodes": num_nodes,
        "hist_len": hist_len,
        "pred_horizon": pred_horizon,
        "data_format": "DCRNN-style npz: x/y=(samples, horizon, nodes, 2), channel 0 target, channel 1 time_in_day",
        "channels": channel_payloads,
    }


def _export_stgcn(
    benchmark: dict[str, Any],
    *,
    output_dir: Path,
    max_samples_per_split: int = 0,
) -> dict[str, Any]:
    method_dir = output_dir / "stgcn"
    method_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(method_dir / "STDR_W.csv", np.asarray(benchmark["adjacency"], dtype=np.float32), delimiter=",")

    manifest = benchmark["manifest"]
    hist_len = int(manifest["hist_len"])
    pred_horizon = int(manifest["pred_horizon"])
    num_nodes = int(manifest["shapes"]["targets_dc"][1])
    targets = np.asarray(benchmark["targets"], dtype=np.float32)
    channel_payloads: dict[str, Any] = {}
    for channel, channel_idx in TARGET_CHANNELS.items():
        channel_dir = method_dir / channel
        channel_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(channel_dir / "STDR_V.csv", targets[:, :, channel_idx], delimiter=",")
        train_values: list[np.ndarray] = []
        split_shapes: dict[str, list[int]] = {}
        for split in ("train", "val", "test"):
            indices = list(benchmark["splits"]["indices"][split])
            if max_samples_per_split > 0:
                indices = indices[: int(max_samples_per_split)]
            sequences = []
            for idx in indices:
                idx = int(idx)
                seq = targets[idx : idx + hist_len + pred_horizon, :, channel_idx]
                sequences.append(seq[:, :, None])
            arr = np.stack(sequences, axis=0).astype(np.float32)
            if split == "train":
                train_values.append(arr)
            np.save(channel_dir / f"{split}_sequence.npy", arr)
            split_shapes[split] = list(arr.shape)
        train_concat = np.concatenate(train_values, axis=0)
        stats = {"mean": float(train_concat.mean()), "std": float(max(train_concat.std(), 1e-6))}
        _write_json(channel_dir / "stgcn_stats.json", stats)
        channel_payloads[channel] = {
            "dataset_dir": str(channel_dir),
            "split_shapes": split_shapes,
            "target_channel": channel,
            "target_channel_index": channel_idx,
            "sequence_shape": "(samples, hist_len + pred_horizon, nodes, 1)",
            "stats": stats,
        }
    return {
        "method_dataset_dir": str(method_dir),
        "weighted_adjacency_csv": str(method_dir / "STDR_W.csv"),
        "num_nodes": num_nodes,
        "hist_len": hist_len,
        "pred_horizon": pred_horizon,
        "data_format": "STGCN adapter arrays: sequence=(samples, hist_len + pred_horizon, nodes, 1)",
        "channels": channel_payloads,
    }


def official_command_templates(method_id: str, export_payload: dict[str, Any], *, project_root: str | Path = ".") -> list[str]:
    project_root = Path(project_root)
    if method_id == "dcrnn":
        upstream = project_root / "external_official_code" / "DCRNN" / "dcrnn_train.py"
        return [
            f"python {upstream} --config_filename {Path(ch['dataset_dir']) / 'dcrnn_config.yaml'}"
            for ch in export_payload["channels"].values()
        ]
    if method_id == "graph_wavenet":
        upstream = project_root / "external_official_code" / "Graph-WaveNet" / "train.py"
        adj = export_payload["adjacency_pickle"]
        pred_horizon = int(export_payload["pred_horizon"])
        num_nodes = int(export_payload["num_nodes"])
        return [
            "python {script} --data {data} --adjdata {adj} --adjtype doubletransition "
            "--gcn_bool --addaptadj --randomadj --seq_length {horizon} --in_dim 2 --num_nodes {nodes} "
            "--epochs 100 "
            "--device cuda:0 --save runs/dc_benchmark_official/graph_wavenet/{channel}".format(
                script=upstream,
                data=ch["dataset_dir"],
                adj=adj,
                horizon=pred_horizon,
                nodes=num_nodes,
                channel=channel,
            )
            for channel, ch in export_payload["channels"].items()
        ]
    if method_id == "stgcn":
        adapter = project_root / "dc_benchmark" / "official_adapters" / "stgcn_train.py"
        graph = export_payload["weighted_adjacency_csv"]
        n_route = int(export_payload["num_nodes"])
        n_his = int(export_payload["hist_len"])
        n_pred = int(export_payload["pred_horizon"])
        return [
            "python {script} --dataset-dir {data} --graph-csv {graph} --n-route {n_route} "
            "--n-his {n_his} --n-pred {n_pred} --batch-size 50 --epoch 100 --inf-mode merge".format(
                script=adapter,
                data=ch["dataset_dir"],
                graph=graph,
                n_route=n_route,
                n_his=n_his,
                n_pred=n_pred,
            )
            for ch in export_payload["channels"].values()
        ]
    return []


def export_official_dataset(
    *,
    method_id: str,
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    output_dir: str | Path = Path("data") / "dc_benchmark_official",
    max_samples_per_split: int = 0,
    project_root: str | Path = ".",
) -> Path:
    require_official_method(method_id)
    benchmark = load_dc_benchmark(dataset_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if method_id in {"dcrnn", "graph_wavenet"}:
        payload = _export_dcrnn_like(
            benchmark,
            method_id=method_id,
            output_dir=output,
            max_samples_per_split=max_samples_per_split,
        )
    elif method_id == "stgcn":
        payload = _export_stgcn(benchmark, output_dir=output, max_samples_per_split=max_samples_per_split)
    else:
        raise ValueError(f"Unsupported official method: {method_id}")

    source_info = get_official_source_info(method_id, project_root=project_root)
    manifest = {
        "schema_version": OFFICIAL_DATASET_SCHEMA,
        "method_id": method_id,
        "claim_type": source_info["claim_type"],
        "adapter_only": True,
        "allowed_modifications": ["data_loader", "output_head"],
        "output_head_change": "none; demand and supply are exported as separate official univariate target runs",
        "official_source": source_info,
        "benchmark_manifest": {
            "dataset_name": benchmark["manifest"]["dataset_name"],
            "schema_version": benchmark["manifest"]["schema_version"],
            "split_hash": benchmark["manifest"]["split_hash"],
            "hist_len": benchmark["manifest"]["hist_len"],
            "pred_horizon": benchmark["manifest"]["pred_horizon"],
            "split_counts": benchmark["manifest"]["split_counts"],
        },
        "export": payload,
        "command_templates": official_command_templates(method_id, payload, project_root=project_root),
    }
    _write_json(output / method_id / "official_manifest.json", manifest)
    return output / method_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export official-code-compatible DC benchmark data.")
    parser.add_argument("--method", required=True, choices=["dcrnn", "graph_wavenet", "stgcn"])
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--output-dir", default=str(Path("data") / "dc_benchmark_official"))
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = export_official_dataset(
        method_id=args.method,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        max_samples_per_split=args.max_samples_per_split,
    )
    print(f"official {args.method} dataset exported to {output}")


if __name__ == "__main__":
    main()
