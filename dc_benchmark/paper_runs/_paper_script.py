from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dc_benchmark.run_paper_baseline import main


def main_for_method(method_id: str, result_name: str) -> None:
    if "--method" not in sys.argv:
        sys.argv[1:1] = ["--method", method_id]
    if "--result-name" not in sys.argv:
        sys.argv.extend(["--result-name", result_name])
    main()
