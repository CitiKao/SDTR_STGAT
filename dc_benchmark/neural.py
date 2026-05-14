from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .baselines import _report_for_method
from .metrics import DCMetricAccumulator


GRAPH_METHODS = {"stgcn", "dcrnn", "graph_wavenet", "mt_mf_gcn"}
MULTISCALE_METHODS = {"convlstm", "deep_multiconvlstm", "graph_wavenet"}
PAPER_REIMPLEMENTATION_METHODS = {"mlrnn_taxi_demand", "deep_multiconvlstm", "mt_mf_gcn"}


PAPER_REIMPLEMENTATION_NOTES: dict[str, str] = {
    "mlrnn_taxi_demand": (
        "Paper-based MLRNN reimplementation: demand-correlation zone clustering, "
        "cluster-level recurrent predictors, and a global recurrent predictor are averaged for D/C outputs. "
        "Training uses a benchmark-adapted mixed weighted MSE over cluster, global, and consistency terms."
    ),
    "deep_multiconvlstm": (
        "Paper-based Deep Multi-Scale ConvLSTM reimplementation: node D/C sequences are mapped to a masked "
        "pseudo-grid, processed at multiple spatial scales, and gathered back to nodes."
    ),
    "mt_mf_gcn": (
        "Paper-based MT-MF-GCN reimplementation: adjacency and demand/supply semantic graphs feed a mixture "
        "GCN encoder with matrix-factorized task decoders. OD demand is unavailable in this benchmark, so D/C "
        "are treated as the two co-predicted zone-level tasks."
    ),
}


def _normalization_arrays(manifest: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    stats = manifest["target_stats_train_time_mask"]
    mean = np.array(
        [stats["demand"]["log1p_mean"], stats["supply"]["log1p_mean"]],
        dtype=np.float32,
    )
    std = np.array(
        [stats["demand"]["log1p_std"], stats["supply"]["log1p_std"]],
        dtype=np.float32,
    )
    return mean, std


def _normalize_counts(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((np.log1p(values) - mean.reshape(1, 1, 1, 2)) / std.reshape(1, 1, 1, 2)).astype(np.float32)


def _denormalize_counts(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.maximum(
        np.expm1(values * std.reshape(1, 1, 1, 2) + mean.reshape(1, 1, 1, 2)),
        0.0,
    ).astype(np.float32)


def _target_time_mask(benchmark: dict[str, Any], split: str = "train") -> np.ndarray:
    manifest = benchmark["manifest"]
    hist_len = int(manifest["hist_len"])
    pred_horizon = int(manifest["pred_horizon"])
    mask = np.zeros(int(manifest["shapes"]["targets_dc"][0]), dtype=bool)
    for idx in benchmark["splits"]["indices"][split]:
        start = int(idx) + hist_len
        mask[start:start + pred_horizon] = True
    return mask


def _safe_row_normalize(matrix: np.ndarray, *, add_self: bool = False) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float32)
    if add_self:
        arr = arr + np.eye(arr.shape[0], dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    row_sum = np.maximum(arr.sum(axis=1, keepdims=True), 1e-6)
    return (arr / row_sum).astype(np.float32)


def _correlation_graph(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).transpose(1, 0, 2).reshape(values.shape[1], -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(flat, axis=1, keepdims=True)
    flat = flat / np.maximum(denom, 1e-6)
    corr = np.maximum(flat @ flat.T, 0.0)
    np.fill_diagonal(corr, 1.0)
    return _safe_row_normalize(corr)


def _build_correlation_clusters(benchmark: dict[str, Any]) -> list[np.ndarray]:
    targets = np.asarray(benchmark["targets"][_target_time_mask(benchmark)], dtype=np.float32)
    num_nodes = int(benchmark["manifest"]["shapes"]["targets_dc"][1])
    num_clusters = min(num_nodes, 10, max(2, int(round(np.sqrt(num_nodes)))))
    corr = _correlation_graph(targets)
    try:
        eigvals, eigvecs = np.linalg.eigh(corr)
        order = np.argsort(eigvecs[:, int(np.argmax(eigvals))], kind="mergesort")
    except np.linalg.LinAlgError:
        order = np.arange(num_nodes, dtype=np.int32)
    clusters = [part.astype(np.int64) for part in np.array_split(order, num_clusters) if len(part) > 0]
    return clusters


def _build_node_weights(benchmark: dict[str, Any]) -> np.ndarray:
    train_values = np.asarray(benchmark["targets"][_target_time_mask(benchmark)], dtype=np.float32)
    weights = train_values.sum(axis=(0, 2))
    if float(weights.sum()) <= 0.0:
        return np.ones(train_values.shape[1], dtype=np.float32)
    weights = weights / np.maximum(weights.mean(), 1e-6)
    return weights.astype(np.float32)


def _build_mtmf_graphs(benchmark: dict[str, Any]) -> np.ndarray:
    train_values = np.asarray(benchmark["targets"][_target_time_mask(benchmark)], dtype=np.float32)
    adjacency = _safe_row_normalize(np.asarray(benchmark["adjacency"], dtype=np.float32), add_self=True)
    demand_graph = _correlation_graph(train_values[:, :, 0:1])
    supply_graph = _correlation_graph(train_values[:, :, 1:2])
    identity = np.eye(adjacency.shape[0], dtype=np.float32)
    return np.stack([identity, adjacency, demand_graph, supply_graph], axis=0).astype(np.float32)


def _square_grid_shape(num_nodes: int) -> tuple[int, int]:
    side = int(np.ceil(np.sqrt(num_nodes)))
    return side, side


class DCWindowTorchDataset(Dataset):
    def __init__(self, benchmark: dict[str, Any], split: str, *, limit: int = 0) -> None:
        self.targets = benchmark["targets"]
        self.manifest = benchmark["manifest"]
        indices = list(benchmark["splits"]["indices"][split])
        self.indices = indices[:limit] if limit and limit > 0 else indices
        self.hist_len = int(self.manifest["hist_len"])
        self.pred_horizon = int(self.manifest["pred_horizon"])
        self.mean, self.std = _normalization_arrays(self.manifest)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = int(self.indices[item])
        t = idx + self.hist_len
        history = np.asarray(self.targets[idx:t], dtype=np.float32).transpose(1, 0, 2)[None, ...]
        target = np.asarray(self.targets[t:t + self.pred_horizon], dtype=np.float32).transpose(1, 0, 2)[None, ...]
        history_norm = _normalize_counts(history, self.mean, self.std)[0]
        target_norm = _normalize_counts(target, self.mean, self.std)[0]
        return {
            "history": torch.from_numpy(history_norm),
            "target": torch.from_numpy(target_norm),
            "target_raw": torch.from_numpy(target[0]),
        }


class CompactPaperInspiredDCModel(nn.Module):
    def __init__(
        self,
        *,
        method_id: str,
        num_nodes: int,
        hist_len: int,
        pred_horizon: int,
        adjacency: np.ndarray | None,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.method_id = method_id
        self.hist_len = hist_len
        self.pred_horizon = pred_horizon
        self.use_graph = method_id in GRAPH_METHODS
        self.use_multiscale = method_id in MULTISCALE_METHODS
        self.use_node_embedding = method_id in {"mlrnn_taxi_demand", "mt_mf_gcn", "graph_wavenet"}
        input_dim = 2 * (2 if self.use_graph else 1)
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.short_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.long_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        if self.use_node_embedding:
            self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        else:
            self.node_embedding = None
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_horizon * 2),
        )
        if adjacency is not None:
            adj = np.asarray(adjacency, dtype=np.float32)
            adj = adj + np.eye(adj.shape[0], dtype=np.float32)
            row_sum = np.maximum(adj.sum(axis=1, keepdims=True), 1.0)
            adj = adj / row_sum
            self.register_buffer("adjacency", torch.from_numpy(adj))
        else:
            self.register_buffer("adjacency", torch.empty(0))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        # history: (B, N, H, 2)
        x = history
        if self.use_graph and self.adjacency.numel() > 0:
            graph_x = torch.einsum("ij,bjhc->bihc", self.adjacency, x)
            x = torch.cat([x, graph_x], dim=-1)
        bsz, nodes, hist, channels = x.shape
        flat = x.reshape(bsz * nodes, hist, channels)
        _, hidden = self.rnn(flat)
        encoded = hidden[-1]
        if self.use_multiscale:
            conv_in = flat.transpose(1, 2)
            encoded = encoded + 0.5 * (
                self.short_conv(conv_in).mean(dim=-1) + self.long_conv(conv_in).mean(dim=-1)
            )
        encoded = encoded.reshape(bsz, nodes, -1)
        if self.node_embedding is not None:
            node_ids = torch.arange(nodes, device=history.device)
            encoded = encoded + self.node_embedding(node_ids).unsqueeze(0)
        out = self.head(encoded).reshape(bsz, nodes, self.pred_horizon, 2)
        return out


class MLRNNPaperReimplementation(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        hist_len: int,
        pred_horizon: int,
        clusters: list[np.ndarray],
        node_weights: np.ndarray,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.hist_len = hist_len
        self.pred_horizon = pred_horizon
        self.register_buffer("node_weights", torch.as_tensor(node_weights, dtype=torch.float32).view(1, num_nodes, 1, 1))
        self.global_rnn = nn.LSTM(num_nodes * 2, hidden_dim, batch_first=True)
        self.global_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_nodes * pred_horizon * 2),
        )
        self.cluster_rnns = nn.ModuleList()
        self.cluster_heads = nn.ModuleList()
        for cluster_idx, cluster in enumerate(clusters):
            self.register_buffer(f"cluster_{cluster_idx}", torch.as_tensor(cluster, dtype=torch.long))
            input_dim = int(len(cluster)) * 2
            self.cluster_rnns.append(nn.LSTM(input_dim, hidden_dim, batch_first=True))
            self.cluster_heads.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, int(len(cluster)) * pred_horizon * 2),
                )
            )
        self.num_clusters = len(clusters)

    def _cluster_indices(self, idx: int) -> torch.Tensor:
        return getattr(self, f"cluster_{idx}")

    def predict_components(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, nodes, hist, channels = history.shape
        global_in = history.permute(0, 2, 1, 3).reshape(bsz, hist, nodes * channels)
        _, (global_hidden, _) = self.global_rnn(global_in)
        global_pred = self.global_head(global_hidden[-1]).reshape(bsz, nodes, self.pred_horizon, 2)

        cluster_pred = history.new_zeros(bsz, nodes, self.pred_horizon, 2)
        for idx, (rnn, head) in enumerate(zip(self.cluster_rnns, self.cluster_heads)):
            cluster_nodes = self._cluster_indices(idx)
            cluster_history = history.index_select(1, cluster_nodes)
            c_nodes = int(cluster_nodes.numel())
            cluster_in = cluster_history.permute(0, 2, 1, 3).reshape(bsz, hist, c_nodes * channels)
            _, (cluster_hidden, _) = rnn(cluster_in)
            pred = head(cluster_hidden[-1]).reshape(bsz, c_nodes, self.pred_horizon, 2)
            cluster_pred[:, cluster_nodes, :, :] = pred

        final_pred = 0.5 * (cluster_pred + global_pred)
        return final_pred, cluster_pred, global_pred

    def _weighted_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target).pow(2) * self.node_weights.to(device=pred.device, dtype=pred.dtype)).mean()

    def training_loss(self, history: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _final_pred, cluster_pred, global_pred = self.predict_components(history)
        return (
            self._weighted_mse(cluster_pred, target)
            + self._weighted_mse(global_pred, target)
            + self._weighted_mse(cluster_pred, global_pred)
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        final_pred, _cluster_pred, _global_pred = self.predict_components(history)
        return final_pred


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def initial_state(self, x: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (x.shape[0], self.hidden_dim, height, width)
        return x.new_zeros(shape), x.new_zeros(shape)


class DeepMultiScaleConvLSTMPaperReimplementation(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        hist_len: int,
        pred_horizon: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.hist_len = hist_len
        self.pred_horizon = pred_horizon
        self.grid_height, self.grid_width = _square_grid_shape(num_nodes)
        flat_index = torch.arange(num_nodes, dtype=torch.long)
        self.register_buffer("flat_index", flat_index)
        self.scales = (1, 2, 4)
        self.cells = nn.ModuleList([ConvLSTMCell(2, hidden_dim) for _ in self.scales])
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * len(self.scales), hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
        )

    def _history_to_grid(self, history: torch.Tensor) -> torch.Tensor:
        bsz, nodes, hist, channels = history.shape
        grid = history.new_zeros(bsz, hist, channels, self.grid_height * self.grid_width)
        values = history.permute(0, 2, 3, 1)
        grid[:, :, :, self.flat_index] = values
        return grid.reshape(bsz, hist, channels, self.grid_height, self.grid_width)

    def _grid_to_nodes(self, grid_pred: torch.Tensor) -> torch.Tensor:
        bsz = grid_pred.shape[0]
        flat = grid_pred.reshape(bsz, self.pred_horizon, 2, self.grid_height * self.grid_width)
        node_values = flat[:, :, :, self.flat_index]
        return node_values.permute(0, 3, 1, 2).contiguous()

    def _run_cell(self, sequence: torch.Tensor, cell: ConvLSTMCell) -> torch.Tensor:
        bsz, seq_len, channels, height, width = sequence.shape
        h, c = cell.initial_state(sequence[:, 0], height, width)
        for step in range(seq_len):
            h, c = cell(sequence[:, step], (h, c))
        return h

    def _predict_next_grid(self, grid: torch.Tensor) -> torch.Tensor:
        encoded_scales = []
        for scale, cell in zip(self.scales, self.cells):
            scaled = grid
            if scale > 1:
                bsz, seq_len, channels, height, width = scaled.shape
                pooled = nn.functional.avg_pool2d(
                    scaled.reshape(bsz * seq_len, channels, height, width),
                    kernel_size=scale,
                    stride=scale,
                    ceil_mode=True,
                )
                scaled = pooled.reshape(bsz, seq_len, channels, pooled.shape[-2], pooled.shape[-1])
            h = self._run_cell(scaled, cell)
            if h.shape[-2:] != (self.grid_height, self.grid_width):
                h = nn.functional.interpolate(h, size=(self.grid_height, self.grid_width), mode="bilinear", align_corners=False)
            encoded_scales.append(h)
        return self.fusion(torch.cat(encoded_scales, dim=1))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        grid = self._history_to_grid(history)
        preds = []
        rolling = grid
        for _step in range(self.pred_horizon):
            next_grid = self._predict_next_grid(rolling)
            preds.append(next_grid)
            rolling = torch.cat([rolling[:, 1:], next_grid.unsqueeze(1)], dim=1)
        grid_pred = torch.stack(preds, dim=1)
        return self._grid_to_nodes(grid_pred)


class MTMFGCNPaperReimplementation(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        hist_len: int,
        pred_horizon: int,
        graphs: np.ndarray,
        hidden_dim: int,
        factor_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.hist_len = hist_len
        self.pred_horizon = pred_horizon
        rank = int(factor_rank or max(4, hidden_dim // 2))
        self.temporal_encoder = nn.GRU(2, hidden_dim, batch_first=True)
        self.register_buffer("graphs", torch.from_numpy(np.asarray(graphs, dtype=np.float32)))
        self.graph_linears = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(int(graphs.shape[0]))])
        self.graph_logits = nn.Parameter(torch.zeros(int(graphs.shape[0])))
        self.node_factor = nn.Embedding(num_nodes, rank)
        self.task_factor = nn.Parameter(torch.randn(2, rank))
        self.factor_proj = nn.Linear(hidden_dim, rank)
        self.horizon_proj = nn.Linear(rank, pred_horizon)
        self.residual_head = nn.Linear(hidden_dim, pred_horizon * 2)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        bsz, nodes, hist, channels = history.shape
        flat = history.reshape(bsz * nodes, hist, channels)
        _, hidden = self.temporal_encoder(flat)
        encoded = hidden[-1].reshape(bsz, nodes, -1)

        graph_outputs = []
        for graph, linear in zip(self.graphs, self.graph_linears):
            message = torch.einsum("ij,bjh->bih", graph, encoded)
            graph_outputs.append(torch.relu(linear(message)))
        weights = torch.softmax(self.graph_logits, dim=0).view(1, 1, -1, 1)
        encoded = (torch.stack(graph_outputs, dim=2) * weights).sum(dim=2)

        node_ids = torch.arange(nodes, device=history.device)
        node_latent = self.factor_proj(encoded) * self.node_factor(node_ids).unsqueeze(0)
        task_values = []
        for task_idx in range(2):
            task_latent = node_latent * self.task_factor[task_idx].view(1, 1, -1)
            task_values.append(self.horizon_proj(task_latent).unsqueeze(-1))
        factorized = torch.cat(task_values, dim=-1)
        residual = self.residual_head(encoded).reshape(bsz, nodes, self.pred_horizon, 2)
        return factorized + 0.1 * residual


def build_paper_reimplementation_model(
    *,
    method_id: str,
    benchmark: dict[str, Any],
    hidden_dim: int,
) -> nn.Module:
    manifest = benchmark["manifest"]
    num_nodes = int(manifest["shapes"]["targets_dc"][1])
    hist_len = int(manifest["hist_len"])
    pred_horizon = int(manifest["pred_horizon"])
    if method_id == "mlrnn_taxi_demand":
        return MLRNNPaperReimplementation(
            num_nodes=num_nodes,
            hist_len=hist_len,
            pred_horizon=pred_horizon,
            clusters=_build_correlation_clusters(benchmark),
            node_weights=_build_node_weights(benchmark),
            hidden_dim=hidden_dim,
        )
    if method_id == "deep_multiconvlstm":
        return DeepMultiScaleConvLSTMPaperReimplementation(
            num_nodes=num_nodes,
            hist_len=hist_len,
            pred_horizon=pred_horizon,
            hidden_dim=hidden_dim,
        )
    if method_id == "mt_mf_gcn":
        return MTMFGCNPaperReimplementation(
            num_nodes=num_nodes,
            hist_len=hist_len,
            pred_horizon=pred_horizon,
            graphs=_build_mtmf_graphs(benchmark),
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"Unsupported paper reimplementation method: {method_id}")


def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _restore_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({name: value.to(device) for name, value in state_dict.items()})


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    manifest: dict[str, Any],
    torch_device: torch.device,
) -> dict[str, Any]:
    mean, std = _normalization_arrays(manifest)
    accumulator = DCMetricAccumulator(
        pred_horizon=int(manifest["pred_horizon"]),
        report_horizons=manifest["report_horizons"],
        target_stats=manifest.get("target_stats_train_time_mask"),
    )
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            pred_norm = model(batch["history"].to(torch_device)).cpu().numpy()
            pred = _denormalize_counts(pred_norm, mean, std)
            accumulator.update(pred, batch["target_raw"].numpy())
    if was_training:
        model.train()
    return accumulator.finalize()


def run_neural_paper_baseline(
    benchmark: dict[str, Any],
    *,
    method_id: str,
    epochs: int = 100,
    early_stop_patience: int = 15,
    batch_size: int = 8,
    hidden_dim: int = 32,
    lr: float = 1e-3,
    device: str = "auto",
    max_train_samples: int = 0,
    max_eval_samples: int = 0,
    checkpoint_path: str | Path | None = None,
    init_checkpoint: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    manifest = benchmark["manifest"]
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    train_ds = DCWindowTorchDataset(benchmark, "train", limit=max_train_samples)
    val_ds = DCWindowTorchDataset(benchmark, "val", limit=max_eval_samples)
    test_ds = DCWindowTorchDataset(benchmark, "test", limit=max_eval_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    if method_id in PAPER_REIMPLEMENTATION_METHODS:
        model = build_paper_reimplementation_model(
            method_id=method_id,
            benchmark=benchmark,
            hidden_dim=hidden_dim,
        ).to(torch_device)
        model_note = PAPER_REIMPLEMENTATION_NOTES[method_id]
    else:
        model = CompactPaperInspiredDCModel(
            method_id=method_id,
            num_nodes=int(manifest["shapes"]["targets_dc"][1]),
            hist_len=int(manifest["hist_len"]),
            pred_horizon=int(manifest["pred_horizon"]),
            adjacency=np.asarray(benchmark["adjacency"]) if method_id in GRAPH_METHODS else None,
            hidden_dim=hidden_dim,
        ).to(torch_device)
        model_note = "Compact paper-inspired adapter for same-data DC benchmarking."
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        state = torch.load(init_path, map_location=torch_device, weights_only=True)
        model.load_state_dict(state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    configured_epochs = int(epochs)
    patience = max(int(early_stop_patience), 0)
    best_state = _clone_state_dict_to_cpu(model)
    best_val_metrics: dict[str, Any] | None = None
    best_val_score = float("inf")
    best_epoch = 0
    wait = 0
    completed_epochs = 0
    stopped_early = False
    training_history: list[dict[str, Any]] = []

    for epoch in range(1, configured_epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            history = batch["history"].to(torch_device)
            target = batch["target"].to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            if hasattr(model, "training_loss"):
                loss = model.training_loss(history, target)
            else:
                loss = loss_fn(model(history), target)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        completed_epochs = epoch
        val_metrics = _evaluate_model(model, val_loader, manifest=manifest, torch_device=torch_device)
        val_score = float(val_metrics["raw_dc"])
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        improved = val_score < best_val_score
        if improved:
            best_val_score = val_score
            best_val_metrics = val_metrics
            best_epoch = epoch
            best_state = _clone_state_dict_to_cpu(model)
            if checkpoint_path:
                ckpt_path = Path(checkpoint_path)
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, ckpt_path)
            wait = 0
        else:
            wait += 1

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_raw_dc": val_score,
                "best_val_raw_dc": best_val_score,
                "best_epoch": best_epoch,
                "epochs_since_improvement": wait,
            }
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "method_id": method_id,
                    "epoch": epoch,
                    "epochs": configured_epochs,
                    "train_loss": train_loss,
                    "val_raw_dc": val_score,
                    "val_metrics": val_metrics,
                    "horizon_reports": val_metrics.get("horizon_reports", {}),
                    "best_val_raw_dc": best_val_score,
                    "best_epoch": best_epoch,
                    "epochs_since_improvement": wait,
                    "patience": patience,
                    "improved": improved,
                }
            )
        if patience > 0 and wait >= patience:
            stopped_early = True
            break

    if best_epoch > 0:
        _restore_state_dict(model, best_state, torch_device)
    test_metrics = _evaluate_model(model, test_loader, manifest=manifest, torch_device=torch_device)

    result = _report_for_method(
        method_id,
        test_metrics,
        {
            "epochs": configured_epochs,
            "completed_epochs": completed_epochs,
            "best_epoch": best_epoch,
            "selection_metric": "val_raw_dc",
            "selection_score": best_val_score if best_epoch > 0 else None,
            "batch_size": int(batch_size),
            "hidden_dim": int(hidden_dim),
            "lr": float(lr),
            "device": str(torch_device),
            "max_train_samples": int(max_train_samples),
            "max_eval_samples": int(max_eval_samples),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
            "init_checkpoint": str(init_checkpoint) if init_checkpoint else "",
            "early_stopping": {
                "enabled": patience > 0,
                "patience": patience,
                "monitor": "val_raw_dc",
                "mode": "min",
                "stopped_early": stopped_early,
            },
            "note": model_note,
            "paper_based_reimplementation": bool(method_id in PAPER_REIMPLEMENTATION_METHODS),
            "dc_target_mapping": (
                "All models output benchmark channels (demand, supply). For original OD-demand papers, "
                "OD matrices are unavailable, so supply/capacity is treated only as a DC co-target proxy."
            ),
        },
    )
    result["validation_metrics"] = best_val_metrics
    result["training_history"] = training_history
    return result
