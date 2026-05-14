from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "external_official_code" / "STGCN_IJCAI-18"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_official_dataset(dataset_dir: Path):
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from data_loader.data_utils import Dataset

    stats = _load_json(dataset_dir / "stgcn_stats.json")
    data = {}
    for split in ("train", "val", "test"):
        raw = np.load(dataset_dir / f"{split}_sequence.npy").astype(np.float32)
        data[split] = (raw - float(stats["mean"])) / float(stats["std"])
    return Dataset(data, {"mean": float(stats["mean"]), "std": float(stats["std"])})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run official STGCN with the DC benchmark loader adapter.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--graph-csv", required=True)
    parser.add_argument("--n-route", type=int, required=True)
    parser.add_argument("--n-his", type=int, required=True)
    parser.add_argument("--n-pred", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--save", type=int, default=10)
    parser.add_argument("--ks", type=int, default=3)
    parser.add_argument("--kt", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--opt", type=str, default="RMSProp")
    parser.add_argument("--inf-mode", type=str, default="merge")
    parser.add_argument("--cuda-visible-devices", default="0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    graph_csv = Path(args.graph_csv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))

    import tensorflow as tf
    from models.tester import model_test
    from models.trainer import model_train
    from utils.math_graph import cheb_poly_approx, scaled_laplacian, weight_matrix

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.Session(config=config)

    w = weight_matrix(str(graph_csv))
    lk = cheb_poly_approx(scaled_laplacian(w), args.ks, args.n_route)
    tf.add_to_collection(name="graph_kernel", value=tf.cast(tf.constant(lk), tf.float32))

    dataset = _load_official_dataset(dataset_dir)
    blocks = [[1, 32, 64], [64, 32, 128]]
    model_train(dataset, blocks, args)
    model_test(dataset, dataset.get_len("test"), args.n_his, args.n_pred, args.inf_mode)


if __name__ == "__main__":
    main()
