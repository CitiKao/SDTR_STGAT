from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


GITHUB_REIMPLEMENTATION_TABLE: dict[str, dict[str, Any]] = {
    "st_resnet": {
        "paper_name": "ST-ResNet",
        "repo_url": "https://github.com/topazape/ST-ResNet",
        "local_path": Path("dc_benchmark") / "external_github_reimplementations" / "ST-ResNet",
        "license": "Unlicense",
        "framework": "pytorch",
        "claim_type": "github_reimplementation_with_dataset_adapter",
        "entrypoint": "stresnet.models.STResNet",
        "notes": (
            "PyTorch reimplementation. Use as an adapted GitHub implementation, not as official paper code."
        ),
    },
    "convlstm": {
        "paper_name": "ConvLSTM",
        "repo_url": "https://github.com/ndrplz/ConvLSTM_pytorch",
        "local_path": Path("dc_benchmark") / "external_github_reimplementations" / "ConvLSTM_pytorch",
        "license": "MIT",
        "framework": "pytorch",
        "claim_type": "github_reimplementation_with_dataset_adapter",
        "entrypoint": "convlstm.ConvLSTM",
        "notes": (
            "General PyTorch ConvLSTM implementation. Use as a local baseline adapter, not as taxi-demand official code."
        ),
    },
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


def get_github_reimplementation_info(method_id: str, *, project_root: str | Path = ".") -> dict[str, Any]:
    if method_id not in GITHUB_REIMPLEMENTATION_TABLE:
        raise ValueError(f"Unsupported GitHub reimplementation method: {method_id}")
    project_root = Path(project_root)
    info = dict(GITHUB_REIMPLEMENTATION_TABLE[method_id])
    local_path = project_root / Path(info["local_path"])
    info["local_path"] = str(local_path)
    info["commit"] = _git_commit(local_path)
    info["available_locally"] = local_path.exists()
    info["not_official_paper_result"] = True
    return info
