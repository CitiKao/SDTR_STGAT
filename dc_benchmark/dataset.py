from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data_loader import build_temporal_features_from_time_meta
from train_predictor import (
    CALENDAR_SPLIT_STRATEGY,
    build_monthly_split_indices,
    build_time_contained_split_indices,
    build_window_time_mask,
    filter_split_indices_by_time_mask,
    infer_time_slot_minutes,
    load_observed_time_mask,
    parse_report_horizons_minutes,
    resolve_report_horizons,
    resolve_split_boundaries,
    validate_split_alignment,
)


SCHEMA_VERSION = "dc_benchmark_v1"
DEFAULT_DATASET_DIR = Path("data") / "dc_benchmark"


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_time_meta(data_dir: Path, limit: int | None = None) -> pd.DataFrame:
    path = data_dir / "time_meta.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing time metadata: {path}")
    time_meta = pd.read_csv(path)
    if limit is not None:
        time_meta = time_meta.iloc[:limit].copy()
    if "date" not in time_meta.columns:
        raise ValueError("time_meta.csv must contain a date column.")
    time_meta["date"] = pd.to_datetime(time_meta["date"], errors="raise")
    if "slot" not in time_meta.columns:
        time_meta["slot"] = np.arange(len(time_meta), dtype=np.int32)
    return time_meta


def _build_splits(
    time_meta: pd.DataFrame,
    *,
    hist_len: int,
    pred_horizon: int,
    split_policy: str,
    split_alignment: str,
) -> tuple[dict[str, list[int]], dict[str, Any] | None, str]:
    if split_policy == "project_monthly":
        return (
            build_monthly_split_indices(time_meta, hist_len, pred_horizon),
            None,
            CALENDAR_SPLIT_STRATEGY,
        )
    if split_policy != "benchmark_contiguous":
        raise ValueError(f"Unsupported split_policy: {split_policy}")

    alignment = validate_split_alignment(split_alignment)
    train_end, val_end, boundary_summary = resolve_split_boundaries(
        time_meta,
        alignment=alignment,
    )
    return (
        build_time_contained_split_indices(
            len(time_meta),
            hist_len=hist_len,
            pred_horizon=pred_horizon,
            train_end=train_end,
            val_end=val_end,
        ),
        boundary_summary,
        "contiguous_ratio_70_10_20_time_split_full_window_containment",
    )


def _target_stats(targets: np.ndarray, time_mask: np.ndarray) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for task_idx, task_name in enumerate(("demand", "supply")):
        values = targets[time_mask, :, task_idx].astype(np.float64).reshape(-1)
        log_values = np.log1p(values)
        stats[task_name] = {
            "raw_mean": float(values.mean()),
            "raw_std": float(values.std()),
            "raw_min": float(values.min()),
            "raw_max": float(values.max()),
            "log1p_mean": float(log_values.mean()),
            "log1p_std": float(max(log_values.std(), 1e-6)),
        }
    return stats


def _split_hash(splits: dict[str, list[int]]) -> str:
    payload = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def export_dc_benchmark(
    *,
    source_data_dir: str | Path = "data",
    output_dir: str | Path = DEFAULT_DATASET_DIR,
    hist_len: int = 14,
    pred_horizon: int = 4,
    report_horizons_minutes: str = "15,30,60",
    split_policy: str = "project_monthly",
    split_alignment: str = "none",
    time_feature_mode: str = "baseline",
    max_time_steps: int = 0,
    force: bool = False,
) -> Path:
    source = Path(source_data_dir)
    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not force:
        return output

    demand_path = source / "node_demand.npy"
    supply_path = source / "node_supply.npy"
    adjacency_path = source / "adjacency_matrix.npy"
    edge_index_path = source / "edge_index.npy"
    for path in (demand_path, supply_path, adjacency_path, edge_index_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing required source file: {path}")

    demand = np.load(demand_path).astype(np.float32)
    supply = np.load(supply_path).astype(np.float32)
    if demand.shape != supply.shape:
        raise ValueError(f"node_demand shape {demand.shape} != node_supply shape {supply.shape}")

    if max_time_steps > 0:
        limit = min(int(max_time_steps), demand.shape[0])
        demand = demand[:limit]
        supply = supply[:limit]
    else:
        limit = demand.shape[0]

    time_meta = _load_time_meta(source, limit)
    if len(time_meta) != demand.shape[0]:
        raise ValueError(
            f"time_meta rows ({len(time_meta)}) must match target time steps ({demand.shape[0]})."
        )
    source_manifest_path = source / "manifest.json"
    source_manifest: dict[str, Any] = {}
    if source_manifest_path.exists():
        source_manifest = _read_json(source_manifest_path)
    source_dataset_name = source_manifest.get("dataset_name")
    if source_dataset_name:
        dataset_name = (
            source_dataset_name
            if str(source_dataset_name).endswith("_benchmark")
            else f"{source_dataset_name}_benchmark"
        )
    else:
        dataset_name = "stdr_nyc_dc_benchmark"

    targets = np.stack([demand, supply], axis=-1).astype(np.float32)
    temporal_features, temporal_feature_names = build_temporal_features_from_time_meta(
        time_meta,
        time_feature_mode=time_feature_mode,
    )
    time_slot_minutes = infer_time_slot_minutes(time_meta)
    report_horizons = resolve_report_horizons(
        time_slot_minutes=time_slot_minutes,
        pred_horizon=pred_horizon,
        requested_minutes=parse_report_horizons_minutes(report_horizons_minutes),
        strict=True,
    )
    splits, split_boundary_summary, split_strategy = _build_splits(
        time_meta,
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        split_policy=split_policy,
        split_alignment=split_alignment,
    )
    unfiltered_sample_counts = {name: len(indices) for name, indices in splits.items()}
    observed_time_mask = load_observed_time_mask(source, len(time_meta))
    if observed_time_mask is None:
        observed_time_mask = np.ones(len(time_meta), dtype=bool)
    else:
        observed_time_mask = observed_time_mask.astype(bool)
        splits = filter_split_indices_by_time_mask(
            splits,
            observed_time_mask,
            hist_len,
            pred_horizon,
        )
    filtered_sample_counts = {name: len(indices) for name, indices in splits.items()}
    excluded_sample_counts = {
        name: unfiltered_sample_counts[name] - filtered_sample_counts[name]
        for name in unfiltered_sample_counts
    }
    train_time_mask = build_window_time_mask(
        len(time_meta),
        splits["train"],
        hist_len,
        pred_horizon,
    )
    if not np.any(train_time_mask):
        raise ValueError("Train split produced an empty time mask.")

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "targets_dc.npy", targets)
    np.save(output / "node_demand.npy", demand)
    np.save(output / "node_supply.npy", supply)
    np.save(output / "time_features.npy", temporal_features.astype(np.float32))
    np.save(output / "observed_time_mask.npy", observed_time_mask.astype(bool))
    np.save(output / "adjacency_matrix.npy", np.load(adjacency_path).astype(np.float32))
    np.save(output / "edge_index.npy", np.load(edge_index_path).astype(np.int32))
    time_meta.to_csv(output / "time_meta.csv", index=False)

    split_payload = {
        "schema_version": SCHEMA_VERSION,
        "hist_len": int(hist_len),
        "pred_horizon": int(pred_horizon),
        "split_policy": split_policy,
        "split_alignment": split_alignment if split_policy == "benchmark_contiguous" else "n/a",
        "split_strategy": split_strategy,
        "split_boundary_summary": split_boundary_summary,
        "sample_counts": filtered_sample_counts,
        "unfiltered_sample_counts": unfiltered_sample_counts,
        "excluded_by_observed_time_mask": excluded_sample_counts,
        "observed_time_mask": "observed_time_mask.npy",
        "indices": splits,
    }
    _write_json(output / "splits.json", split_payload)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "implementation_scope": "same-data DC benchmark for paper-inspired baselines",
        "source_data_dir": str(source),
        "source_dataset_name": source_dataset_name,
        "source_city": source_manifest.get("source_city"),
        "created_from_files": {
            "node_demand.npy": _sha256_path(demand_path),
            "node_supply.npy": _sha256_path(supply_path),
            "adjacency_matrix.npy": _sha256_path(adjacency_path),
            "edge_index.npy": _sha256_path(edge_index_path),
        },
        "array_files": {
            "targets_dc": "targets_dc.npy",
            "node_demand": "node_demand.npy",
            "node_supply": "node_supply.npy",
            "time_features": "time_features.npy",
            "observed_time_mask": "observed_time_mask.npy",
            "adjacency_matrix": "adjacency_matrix.npy",
            "edge_index": "edge_index.npy",
            "time_meta": "time_meta.csv",
            "splits": "splits.json",
        },
        "shapes": {
            "targets_dc": list(targets.shape),
            "time_features": list(temporal_features.shape),
        },
        "target_channels": ["demand", "supply"],
        "hist_len": int(hist_len),
        "pred_horizon": int(pred_horizon),
        "report_horizons": report_horizons,
        "split_hash": _split_hash(splits),
        "split_counts": split_payload["sample_counts"],
        "unfiltered_split_counts": unfiltered_sample_counts,
        "excluded_by_observed_time_mask": excluded_sample_counts,
        "observed_time": {
            "mask_file": "observed_time_mask.npy",
            "observed_slots": int(observed_time_mask.sum()),
            "total_slots": int(len(observed_time_mask)),
            "policy": (
                "Samples are kept only when the full history+target window touches observed slots. "
                "Missing raw dates may remain in time_meta for calendar alignment but are never "
                "treated as real zero-demand labels."
            ),
        },
        "target_stats_train_time_mask": _target_stats(targets, train_time_mask),
        "time_feature_mode": time_feature_mode,
        "time_feature_names": temporal_feature_names,
        "notes": [
            "All baselines must train only on train split windows.",
            "Validation is for model selection; test metrics are final report metrics.",
            "Paper-named runners are paper-inspired reimplementations unless explicitly replaced by official code.",
        ],
    }
    _write_json(manifest_path, manifest)
    return output


def load_dc_benchmark(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    *,
    mmap_mode: str | None = None,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest = _read_json(root / "manifest.json")
    splits = _read_json(root / "splits.json")
    arrays = manifest["array_files"]
    return {
        "root": root,
        "manifest": manifest,
        "splits": splits,
        "targets": np.load(root / arrays["targets_dc"], mmap_mode=mmap_mode),
        "time_features": np.load(root / arrays["time_features"], mmap_mode=mmap_mode),
        "observed_time_mask": np.load(root / arrays["observed_time_mask"], mmap_mode=mmap_mode)
        if "observed_time_mask" in arrays
        else np.ones(manifest["shapes"]["targets_dc"][0], dtype=bool),
        "adjacency": np.load(root / arrays["adjacency_matrix"], mmap_mode=mmap_mode),
        "edge_index": np.load(root / arrays["edge_index"], mmap_mode=mmap_mode),
        "time_meta": pd.read_csv(root / arrays["time_meta"]),
    }


def iter_dc_windows(
    targets: np.ndarray,
    indices: list[int],
    *,
    hist_len: int,
    pred_horizon: int,
    batch_size: int = 32,
):
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        history = []
        target = []
        target_times = []
        for idx in batch_indices:
            t = int(idx) + hist_len
            history.append(np.asarray(targets[idx:t], dtype=np.float32).transpose(1, 0, 2))
            target.append(np.asarray(targets[t:t + pred_horizon], dtype=np.float32).transpose(1, 0, 2))
            target_times.append(np.arange(t, t + pred_horizon, dtype=np.int32))
        yield {
            "indices": np.asarray(batch_indices, dtype=np.int32),
            "history": np.stack(history, axis=0),
            "target": np.stack(target, axis=0),
            "target_times": np.stack(target_times, axis=0),
        }
