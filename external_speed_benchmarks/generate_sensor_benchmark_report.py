from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_or_dataset_summary(run_dir: Path, meta: dict) -> tuple[dict, str]:
    run_snapshot_path = run_dir / "prepared_dataset_summary.json"
    if run_snapshot_path.exists():
        return load_json(run_snapshot_path), f"run_snapshot:{run_snapshot_path.name}"
    dataset_summary_path = Path(meta["dataset_dir"]) / "dataset_summary.json"
    if dataset_summary_path.exists():
        return load_json(dataset_summary_path), f"dataset_dir:{dataset_summary_path.name}"
    return {}, "missing"


def load_pseudo_edge_summary(run_dir: Path, dataset_summary: dict) -> dict:
    run_snapshot_path = run_dir / "prepared_pseudo_edge_summary.json"
    if run_snapshot_path.exists():
        return load_json(run_snapshot_path)
    pseudo_edge_summary_file = str(dataset_summary.get("pseudo_edge_summary_file", "")).strip()
    if not pseudo_edge_summary_file:
        return {}
    dataset_summary_path = Path(pseudo_edge_summary_file)
    if dataset_summary_path.exists() and dataset_summary_path.is_file():
        return load_json(dataset_summary_path)
    return {}


def build_dataset_section(run_dir: Path) -> list[str]:
    meta = load_json(run_dir / "stgat_meta.json")
    metrics = load_json(run_dir / "predictor_test_metrics.json")
    dataset_summary, dataset_summary_source = load_run_or_dataset_summary(run_dir, meta)
    dataset_name = meta["dataset_name"]
    report = metrics["raw_metrics_report"]["speed"]
    best = meta["selected_checkpoint"]
    training_control = meta.get("training_control", {})
    early_stopping = training_control.get("early_stopping", {})
    graph_source = meta.get("graph_topology", {}).get("fixed_graph_source", {})
    preprocessing_variant_id = meta.get("preprocessing_variant_id", "unknown")
    outlier_cleaning = meta.get("outlier_cleaning", {"enabled": False, "method": "unknown"})
    benchmark_comparability = meta.get("benchmark_comparability", {})
    deviations = benchmark_comparability.get("deviations", [])
    split_summary = meta.get("split_summary", {})
    split_calendar = split_summary.get("calendar", {})
    scheduler = meta.get("scheduler", {})
    optimizer = meta.get("optimizer", {})
    split_rule = split_summary.get("rule", {})
    metric_protocol = metrics.get(
        "speed_metric_protocol",
        meta.get("speed_metric_protocol", "unmasked_all_values"),
    )
    representation_domain = meta.get(
        "representation_domain",
        dataset_summary.get("representation_domain", "sensor_node"),
    )
    representation_variant_id = meta.get(
        "representation_variant_id",
        dataset_summary.get("representation_variant_id", representation_domain),
    )
    split_mode = split_summary.get("mode", meta.get("split_strategy", "unknown"))
    pseudo_edge_summary = load_pseudo_edge_summary(run_dir, dataset_summary)
    title_tags = [
        representation_domain,
        split_mode,
        (
            f"cleaned:{outlier_cleaning.get('method', 'unknown')}"
            if outlier_cleaning.get("enabled")
            else "raw"
        ),
    ]

    lines = [
        f"## {dataset_name} [{' | '.join(title_tags)}]",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Completed epochs: `{training_control.get('completed_epochs', 'unknown')}/{training_control.get('configured_epochs', 'unknown')}`",
        f"- Stop reason: `{training_control.get('training_end_reason', 'unknown')}`",
        (
            f"- Early stopping: `enabled | patience={early_stopping.get('patience', 'unknown')} | "
            f"min_epochs={early_stopping.get('min_epochs', 'unknown')} | "
            f"triggered={'yes' if early_stopping.get('triggered') else 'no'}`"
            if early_stopping.get("enabled")
            else "- Early stopping: `disabled`"
        ),
        f"- Best epoch: `{best['best_epoch']}`",
        f"- Best val RMSE: `{best['best_val_raw_speed_rmse']:.4f}`",
        f"- Preprocessing variant: `{preprocessing_variant_id}`",
        f"- Dataset fingerprint: `{meta.get('processed_dataset_fingerprint', 'unknown')}`",
        f"- Representation: `{representation_domain}`",
        f"- Representation variant: `{representation_variant_id}`",
        f"- V domain: `{meta.get('graph_topology', {}).get('v_domain', metrics.get('v_domain', 'unknown'))}`",
        f"- Sensors: `{meta.get('num_sensors', meta['num_nodes'])}`",
        f"- Graph nodes: `{meta.get('num_graph_nodes', meta['num_nodes'])}`",
        f"- Graph edges: `{meta['num_graph_edges']}`",
        f"- Speed items: `{meta.get('num_speed_items', meta['num_nodes'])}`",
        f"- Dataset summary source: `{dataset_summary_source}`",
        f"- Split: `{meta['split_strategy']}` with train/val/test = `{meta['split_counts']['train']}/{meta['split_counts']['val']}/{meta['split_counts']['test']}`",
        f"- Split description: `{meta.get('split_description', 'unknown')}`",
        f"- Adaptive graph: `{'on' if meta['graph_topology']['adaptive_enabled'] else 'off'}`",
        f"- Fixed graph source: `{graph_source.get('mode', 'unknown')}`",
        f"- Metric protocol: `{metric_protocol}`",
        f"- Optimizer: `{optimizer.get('name', 'unknown')} | lr={optimizer.get('lr', 'unknown')} | weight_decay={optimizer.get('weight_decay', 'unknown')}`",
        f"- Scheduler: `{scheduler.get('name', 'unknown')} | monitor={scheduler.get('monitor', 'unknown')} | patience={scheduler.get('patience', 'unknown')} | cooldown={scheduler.get('cooldown', 'unknown')} | min_lr={scheduler.get('min_lr', 'unknown')}`",
        (
            f"- Outlier cleaning: `enabled | method={outlier_cleaning.get('method', 'unknown')} | "
            f"fit={outlier_cleaning.get('fit_scope', 'unknown')} | "
            f"apply={outlier_cleaning.get('apply_scope', 'unknown')} | "
            f"replace={outlier_cleaning.get('replace_strategy', 'unknown')}`"
            if outlier_cleaning.get("enabled")
            else "- Outlier cleaning: `disabled`"
        ),
        f"- Cleaned points: `{outlier_cleaning.get('cleaned_points', 0)} ({100.0 * float(outlier_cleaning.get('cleaned_ratio', 0.0)):.2f}% of valid points)`",
        f"- Benchmark comparability: `{'official-like preprocessing' if benchmark_comparability.get('is_official_like') else 'custom / not directly official-like'}`",
        f"- Benchmark deviations: `{'; '.join(deviations) if deviations else 'none recorded'}`",
        f"- Actual date range: `{dataset_summary.get('actual_start', 'unknown')}` to `{dataset_summary.get('actual_end', 'unknown')}`",
        "",
        "| Horizon | RMSE | MAE | MAPE | MSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    if split_summary.get("mode") == "project_monthly":
        lines.insert(13, f"- Split rule: `train={split_rule.get('train', 'unknown')} | val={split_rule.get('val', 'unknown')} | test={split_rule.get('test', 'unknown')}`")
        lines.insert(14, f"- Train first/last target: `{split_calendar.get('train', {}).get('first_target_timestamp', 'unknown')}` to `{split_calendar.get('train', {}).get('last_target_timestamp', 'unknown')}`")
        lines.insert(15, f"- Val first/last target: `{split_calendar.get('val', {}).get('first_target_timestamp', 'unknown')}` to `{split_calendar.get('val', {}).get('last_target_timestamp', 'unknown')}`")
        lines.insert(16, f"- Test first/last target: `{split_calendar.get('test', {}).get('first_target_timestamp', 'unknown')}` to `{split_calendar.get('test', {}).get('last_target_timestamp', 'unknown')}`")
    else:
        lines.insert(13, f"- Split alignment: `{split_summary.get('alignment', 'unknown')}`")
        lines.insert(14, f"- Train span: `{split_calendar.get('train', {}).get('start', 'unknown')}` to `{split_calendar.get('train', {}).get('end', 'unknown')}`")
        lines.insert(15, f"- Val span: `{split_calendar.get('val', {}).get('start', 'unknown')}` to `{split_calendar.get('val', {}).get('end', 'unknown')}`")
        lines.insert(16, f"- Test span: `{split_calendar.get('test', {}).get('start', 'unknown')}` to `{split_calendar.get('test', {}).get('end', 'unknown')}`")
    if representation_domain == "pseudo_edge" and pseudo_edge_summary:
        topology_summary = pseudo_edge_summary.get("topology_summary", {})
        construction = pseudo_edge_summary.get("construction", {})
        lines.insert(
            -3,
            f"- Pseudo-edge topology: `self_loop_fixes={pseudo_edge_summary.get('self_loop_fixes', 'unknown')} "
            f"({100.0 * float(pseudo_edge_summary.get('self_loop_fix_ratio', 0.0)):.2f}%) | "
            f"isolated_edges={topology_summary.get('isolated_edges', 'unknown')} "
            f"({100.0 * float(pseudo_edge_summary.get('isolated_edge_ratio', 0.0)):.2f}%) | "
            f"unique_structural_edges={topology_summary.get('num_unique_structural_edges', 'unknown')} | "
            f"weak_components={topology_summary.get('line_graph_weak_components', 'unknown')} | "
            f"largest_component={100.0 * float(topology_summary.get('line_graph_largest_component_ratio', 0.0)):.2f}%`",
        )
        lines.insert(
            -3,
            f"- Pseudo-edge construction: `base={construction.get('sensor_base_graph_mode', 'unknown')} | "
            f"cluster_radius_km={pseudo_edge_summary.get('cluster_radius_km', 'unknown')} | "
            f"cluster_radius_scale={construction.get('cluster_radius_scale', 'unknown')} | "
            f"half_length_scale={construction.get('probe_half_length_scale', 'unknown')} | "
            f"fallback_neighbor_k={construction.get('fallback_neighbor_k', 'unknown')}`",
        )
        if pseudo_edge_summary.get("health_warnings"):
            lines.insert(
                -3,
                f"- Pseudo-edge warnings: `{'; '.join(pseudo_edge_summary['health_warnings'])}`",
            )
    if dataset_summary.get("notes"):
        lines.insert(-3, f"- Dataset note: `{dataset_summary['notes']}`")
    horizon_labels = sorted(
        report.keys(),
        key=lambda label: int(str(label).replace("min", "")),
    )
    for horizon in horizon_labels:
        values = report[horizon]
        lines.append(
            f"| {horizon} | {values['rmse']:.4f} | {values['mae']:.4f} | {values.get('mape', 0.0):.2f}% | {values['mse']:.4f} |"
        )
    lines.append("")
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a combined markdown report for the METR-LA / PEMS-BAY sensor benchmarks.")
    parser.add_argument("--run-dirs", type=str, required=True, help="Comma-separated run directories.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="external_speed_benchmarks/results",
        help="Where to write the markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(chunk.strip()) for chunk in args.run_dirs.split(",") if chunk.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"sensor_benchmark_report_{timestamp}.md"

    lines = [
        "# Sensor Benchmark Report",
        "",
        f"Generated at `{datetime.now().isoformat(timespec='seconds')}`.",
        "",
    ]
    for run_dir in run_dirs:
        lines.extend(build_dataset_section(run_dir))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
