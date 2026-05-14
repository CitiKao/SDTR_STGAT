from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""} and str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .dataset import DEFAULT_DATASET_DIR
    from .official_export import export_official_dataset
    from .official_sources import OFFICIAL_SOURCE_TABLE, get_official_source_info, require_official_method
except ImportError:  # pragma: no cover - direct script execution
    from dc_benchmark.dataset import DEFAULT_DATASET_DIR
    from dc_benchmark.official_export import export_official_dataset
    from dc_benchmark.official_sources import OFFICIAL_SOURCE_TABLE, get_official_source_info, require_official_method


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a verified official-code DC benchmark run.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--official-data-dir", default=str(Path("data") / "dc_benchmark_official"))
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--print-commands", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_official_method(args.method)
    source = get_official_source_info(args.method, project_root=args.project_root)
    output = export_official_dataset(
        method_id=args.method,
        dataset_dir=args.dataset_dir,
        output_dir=args.official_data_dir,
        max_samples_per_split=args.max_samples_per_split,
        project_root=args.project_root,
    )
    manifest_path = output / "official_manifest.json"
    print(f"{source['paper_name']} official source: {source['repo_url']}")
    print(f"commit: {source.get('commit') or 'unknown'}")
    print(f"adapter manifest: {manifest_path}")
    if args.print_commands:
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for command in manifest.get("command_templates", []):
            print(command)


if __name__ == "__main__":
    main()
