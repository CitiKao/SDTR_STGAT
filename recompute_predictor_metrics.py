from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data_loader import SpatioTemporalDataset, load_nyc_real_graph_features
from predictor_normalization import (
    build_normalization_stats,
    load_normalization_stats,
    normalize_node_features,
    normalize_speed_features,
    serialize_normalization_stats,
)
from stgat_model import STGATPredictor
from train_predictor import (
    CALENDAR_SPLIT_DESCRIPTION,
    CALENDAR_SPLIT_STRATEGY,
    assign_calendar_split,
    build_window_time_mask,
    build_monthly_split_indices,
    configure_cuda_runtime,
    evaluate_loader,
    evaluate_loader_raw_metrics,
    load_time_meta_for_training,
    resolve_device,
    resolve_num_workers,
    resolve_precision,
)


COMPILED_PREFIX = "_orig_mod."


def _strip_compiled_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    stripped: dict[str, Any] = {}
    for key, value in state_dict.items():
        if key.startswith(COMPILED_PREFIX):
            stripped[key[len(COMPILED_PREFIX):]] = value
        else:
            stripped[key] = value
    return stripped


def infer_model_config(state_dict: dict[str, Any]) -> dict[str, Any]:
    sd = _strip_compiled_prefix(state_dict)

    try:
        node_proj_weight = sd["node_proj.weight"]
        demand_head_weight = sd["demand_head.weight"]
        gat_q = sd["n_gat_fix.0.a_q"]
        gate_conv_weight = sd["n_gtcn_fix.0.layers.0.gate_conv.weight"]
    except KeyError as exc:
        raise KeyError(
            "Checkpoint does not look like an STGAT predictor state_dict. "
            f"Missing key: {exc}"
        ) from exc

    block_ids = sorted(
        {
            int(parts[1])
            for key in sd
            for parts in [key.split(".")]
            if len(parts) > 3 and parts[0] == "n_gtcn_fix" and parts[2] == "layers"
        }
    )
    layer_ids = sorted(
        {
            int(parts[3])
            for key in sd
            for parts in [key.split(".")]
            if len(parts) > 4 and parts[0] == "n_gtcn_fix" and parts[2] == "layers"
        }
    )
    if not block_ids or not layer_ids:
        raise ValueError("Unable to infer STGAT block configuration from checkpoint.")

    node_feat_dim = int(node_proj_weight.shape[1])
    return {
        "hidden_dim": int(node_proj_weight.shape[0]),
        "node_feat_dim": node_feat_dim,
        "pred_horizon": int(demand_head_weight.shape[0]),
        "num_heads": int(gat_q.shape[2]),
        "num_st_blocks": int(max(block_ids) + 1),
        "num_gtcn_layers": int(max(layer_ids) + 1),
        "kernel_size": int(gate_conv_weight.shape[-1]),
        "use_time_features": bool(node_feat_dim > 2),
    }


def load_saved_run_metadata(ckpt_path: Path) -> dict[str, Any] | None:
    meta_path = ckpt_path.parent / "stgat_meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else None


def infer_checkpoint_task(ckpt_path: Path, saved_meta: dict[str, Any] | None) -> str | None:
    stem = ckpt_path.stem.lower()
    if stem.endswith("_dc"):
        return "dc"
    if stem.endswith("_v"):
        return "v"
    checkpoint_selection = saved_meta.get("checkpoint_selection", {}) if saved_meta else {}
    default_monitor = checkpoint_selection.get("default_monitor")
    return str(default_monitor) if default_monitor else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recompute predictor metrics and metadata from an existing STGAT checkpoint."
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to stgat_best.pt")
    p.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="Directory to write regenerated JSON files; defaults to checkpoint parent",
    )
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--max-time-steps", type=int, default=0)
    p.add_argument(
        "--edge-length-source",
        type=str,
        default="osrm",
        choices=["osrm", "centroid"],
    )
    p.add_argument("--hist-len", type=int, default=12)
    p.add_argument(
        "--adaptive-topk",
        type=int,
        default=16,
        help="Fallback adaptive top-k when checkpoint metadata is unavailable; 0 keeps the dense graph.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp32"],
    )
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--lambda1", type=float, default=1.0)
    p.add_argument("--lambda2", type=float, default=1.0)
    p.add_argument("--lambda3", type=float, default=1.0)
    p.add_argument(
        "--train-task",
        type=str,
        default="joint",
        choices=["joint", "dc"],
        help="Training mode used by the checkpoint; recorded in regenerated metadata.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    log_dir = Path(args.log_dir) if args.log_dir else ckpt_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)
    saved_meta = load_saved_run_metadata(ckpt_path)

    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    precision = resolve_precision(device, args.precision)
    amp_enabled = device.type == "cuda" and precision == "bf16"
    amp_dtype = torch.bfloat16 if amp_enabled else None
    num_workers = resolve_num_workers(args.num_workers, device)
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    raw_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_state, dict):
        raise TypeError("Checkpoint must contain a state_dict-like mapping.")
    state_dict = _strip_compiled_prefix(raw_state)
    config = infer_model_config(state_dict)
    hist_len = int(saved_meta.get("hist_len", args.hist_len)) if saved_meta else int(args.hist_len)
    adaptive_topk = int(saved_meta.get("adaptive_topk", args.adaptive_topk)) if saved_meta else int(args.adaptive_topk)
    train_task = str(saved_meta.get("train_task", args.train_task)) if saved_meta else str(args.train_task)
    selected_checkpoint_task = infer_checkpoint_task(ckpt_path, saved_meta)
    saved_normalization = (
        load_normalization_stats(saved_meta.get("normalization"))
        if saved_meta
        else None
    )

    print(f"Loading checkpoint: {ckpt_path}")
    print(
        "Inferred config | "
        f"hidden_dim={config['hidden_dim']} "
        f"heads={config['num_heads']} "
        f"st_blocks={config['num_st_blocks']} "
        f"gtcn_layers={config['num_gtcn_layers']} "
        f"kernel={config['kernel_size']} "
        f"pred_horizon={config['pred_horizon']} "
        f"hist_len={hist_len} "
        f"adaptive_topk={adaptive_topk} "
        f"use_time_features={config['use_time_features']}"
    )

    data = load_nyc_real_graph_features(
        args.data_dir,
        max_time_steps=args.max_time_steps,
        edge_length_source=args.edge_length_source,
        add_time_features=config["use_time_features"],
    )

    adj = data["adj"]
    edge_index = data["edge_index"]
    edge_lengths = data["edge_lengths"]
    node_feat = data["node_features"]
    edge_speeds = data["edge_speeds"]
    time_feature_names = data.get("time_feature_names", [])

    if int(node_feat.shape[-1]) != int(config["node_feat_dim"]):
        raise ValueError(
            "Checkpoint expects node_feat_dim="
            f"{config['node_feat_dim']}, but loaded data produced {node_feat.shape[-1]}. "
            "Check whether the original run used time features."
        )

    t_steps = int(node_feat.shape[0])
    time_meta = load_time_meta_for_training(args.data_dir, t_steps)
    split_indices = build_monthly_split_indices(
        time_meta,
        hist_len,
        config["pred_horizon"],
    )
    train_time_mask = build_window_time_mask(
        t_steps,
        split_indices["train"],
        hist_len,
        config["pred_horizon"],
    )
    normalization_stats = saved_normalization or build_normalization_stats(
        node_feat,
        edge_speeds,
        train_time_mask,
    )
    node_feat = normalize_node_features(node_feat, normalization_stats)
    edge_speeds = normalize_speed_features(edge_speeds, normalization_stats, edge_axis=1)

    full_ds = SpatioTemporalDataset(
        node_feat,
        edge_speeds,
        hist_len=hist_len,
        pred_horizon=config["pred_horizon"],
    )
    val_ds = Subset(full_ds, split_indices["val"])
    test_ds = Subset(full_ds, split_indices["test"])

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    model = STGATPredictor(
        num_nodes=adj.shape[0],
        edge_index=torch.from_numpy(edge_index),
        edge_lengths=torch.from_numpy(edge_lengths),
        adj_matrix=torch.from_numpy(adj),
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_st_blocks=config["num_st_blocks"],
        num_gtcn_layers=config["num_gtcn_layers"],
        kernel_size=config["kernel_size"],
        pred_horizon=config["pred_horizon"],
        node_feat_dim=config["node_feat_dim"],
        adaptive_topk=adaptive_topk,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    mse = torch.nn.MSELoss()
    val_losses = evaluate_loader(
        model,
        val_loader,
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        mse=mse,
        lam1=args.lambda1,
        lam2=args.lambda2,
        lam3=args.lambda3,
    )
    test_losses = evaluate_loader(
        model,
        test_loader,
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        mse=mse,
        lam1=args.lambda1,
        lam2=args.lambda2,
        lam3=args.lambda3,
    )
    test_raw_metrics = evaluate_loader_raw_metrics(
        model,
        test_loader,
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        normalization_stats=normalization_stats,
    )

    meta = {
        "num_nodes": int(adj.shape[0]),
        "num_edges": int(edge_index.shape[0]),
        "edge_index": edge_index.tolist(),
        "edge_lengths": edge_lengths.tolist(),
        "adj": adj.tolist(),
        "hidden_dim": int(config["hidden_dim"]),
        "num_heads": int(config["num_heads"]),
        "num_st_blocks": int(config["num_st_blocks"]),
        "num_gtcn_layers": int(config["num_gtcn_layers"]),
        "kernel_size": int(config["kernel_size"]),
        "adaptive_topk": adaptive_topk,
        "pred_horizon": int(config["pred_horizon"]),
        "hist_len": hist_len,
        "node_feat_dim": int(config["node_feat_dim"]),
        "use_time_features": bool(time_feature_names),
        "time_feature_names": time_feature_names,
        "loss_space": "normalized",
        "loss_tasks": {
            "dc": {"formula": f"{args.lambda1} * demand + {args.lambda2} * supply"},
            "v": {"formula": f"{args.lambda3} * speed"},
        },
        "train_task": train_task,
        "optimized_tasks": ["dc"] if train_task == "dc" else ["dc", "v"],
        "normalization": serialize_normalization_stats(normalization_stats),
        "split_strategy": CALENDAR_SPLIT_STRATEGY,
        "split_description": CALENDAR_SPLIT_DESCRIPTION,
        "normalization_time_steps": int(train_time_mask.sum()),
        "split_counts": {
            "train": len(split_indices["train"]),
            "val": len(split_indices["val"]),
            "test": len(split_indices["test"]),
        },
        "data_source": "nyc_real",
        "data_dir": args.data_dir,
        "checkpoint": str(ckpt_path),
        "recomputed": True,
    }

    with open(log_dir / "stgat_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(log_dir / "predictor_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss_space": "normalized",
                "normalized_loss": test_losses,
                "raw_metrics": test_raw_metrics,
                "val_normalized_loss": val_losses,
                "recomputed_from_checkpoint": str(ckpt_path),
                "selected_checkpoint": str(ckpt_path),
                "selected_checkpoint_task": selected_checkpoint_task,
                "train_task": train_task,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Recomputed validation loss: {val_losses}")
    print(f"Recomputed test loss: {test_losses}")
    print(f"Recomputed raw test metrics: {test_raw_metrics}")
    print(f"Saved {log_dir / 'stgat_meta.json'}")
    print(f"Saved {log_dir / 'predictor_test_metrics.json'}")
    print("Note: predictor_log.json cannot be reconstructed from a checkpoint alone.")


if __name__ == "__main__":
    main()
