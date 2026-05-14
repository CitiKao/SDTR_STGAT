from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


OFFICIAL_SOURCE_TABLE: dict[str, dict[str, Any]] = {
    "dcrnn": {
        "paper_name": "DCRNN",
        "repo_url": "https://github.com/liyaguang/DCRNN",
        "local_path": Path("external_official_code") / "DCRNN",
        "license": "MIT",
        "framework": "tensorflow1",
        "claim_type": "official_code_with_dataset_adapter",
        "notes": "Official TensorFlow implementation; DC benchmark uses one official run per target channel.",
    },
    "graph_wavenet": {
        "paper_name": "Graph WaveNet",
        "repo_url": "https://github.com/nnzhan/Graph-WaveNet",
        "local_path": Path("external_official_code") / "Graph-WaveNet",
        "license": "MIT",
        "framework": "pytorch",
        "claim_type": "official_code_with_dataset_adapter",
        "notes": "Original PyTorch implementation; DC benchmark uses one official run per target channel.",
    },
    "stgcn": {
        "paper_name": "STGCN",
        "repo_url": "https://github.com/VeritasYin/STGCN_IJCAI-18",
        "local_path": Path("external_official_code") / "STGCN_IJCAI-18",
        "license": "BSD-2-Clause",
        "framework": "tensorflow1",
        "claim_type": "official_code_with_dataset_adapter",
        "notes": "Author TensorFlow implementation; official model path is used through an external dataset adapter.",
    },
}


UNSUPPORTED_OFFICIAL_METHODS: dict[str, str] = {
    "ha": "Historical Average is a traditional baseline, not a paper-specific official codebase.",
    "arima": "ARIMA is a traditional statistical baseline; use statsmodels as a library baseline, not official paper code.",
    "xgboost": "XGBoost is a general ML library baseline, not a traffic-paper official codebase.",
    "lstm": "LSTM is a general neural layer/baseline unless a specific paper/codebase is selected.",
    "convlstm": "The ConvLSTM paper is identifiable, but no verified public official codebase was found.",
    "st_resnet": "The Microsoft Research code link points to the old DeepST repository, which is currently unavailable.",
    "mlrnn_taxi_demand": "No verified public official MLRNN Taxi Demand codebase was found.",
    "deep_multiconvlstm": "No verified public official Deep MultiConvLSTM codebase was found.",
    "mt_mf_gcn": "The MT-MF-GCN paper is identifiable, but no verified public official codebase was found.",
}


def _git_commit(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def get_official_source_info(method_id: str, *, project_root: str | Path = ".") -> dict[str, Any]:
    if method_id not in OFFICIAL_SOURCE_TABLE:
        reason = UNSUPPORTED_OFFICIAL_METHODS.get(method_id, f"Unsupported official method: {method_id}")
        raise ValueError(reason)
    project_root = Path(project_root)
    info = dict(OFFICIAL_SOURCE_TABLE[method_id])
    local_path = project_root / Path(info["local_path"])
    info["local_path"] = str(local_path)
    info["commit"] = _git_commit(local_path)
    info["available_locally"] = local_path.exists()
    return info


def require_official_method(method_id: str) -> None:
    if method_id in OFFICIAL_SOURCE_TABLE:
        return
    reason = UNSUPPORTED_OFFICIAL_METHODS.get(method_id, f"Unsupported official method: {method_id}")
    raise ValueError(reason)
