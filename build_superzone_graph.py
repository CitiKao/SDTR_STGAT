from __future__ import annotations

import argparse
import json

from superzone_graph import build_superzone_artifacts, load_superzone_artifacts, reachability_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build K=64 superzone dispatch and RL graphs")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--output-dir", type=str, default="")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--osrm-topk", type=int, default=8)
    p.add_argument("--connector-count", type=int, default=2)
    p.add_argument("--shapefile", type=str, default="")
    p.add_argument("--osrm-url", type=str, default="http://router.project-osrm.org")
    p.add_argument(
        "--refresh-osrm",
        action="store_true",
        help="Ignore cached dispatch matrices and query OSRM Table again.",
    )
    p.add_argument(
        "--no-osrm",
        action="store_true",
        help="Use a cached OSRM Table matrix instead of querying OSRM.",
    )
    p.add_argument(
        "--allow-fallback-costs",
        action="store_true",
        help="Allow sparse edge-length fallback costs for local smoke tests only.",
    )
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    meta = build_superzone_artifacts(
        args.data_dir,
        output_dir=args.output_dir or None,
        k=args.k,
        topk=args.osrm_topk,
        connector_count=args.connector_count,
        shapefile=args.shapefile or None,
        use_osrm=not args.no_osrm,
        osrm_url=args.osrm_url,
        reuse_existing_costs=not args.refresh_osrm,
        allow_fallback_costs=args.allow_fallback_costs,
    )
    artifacts = load_superzone_artifacts(args.data_dir, args.output_dir or None)
    demand_weights = artifacts["region_demand"].sum(axis=0)
    metrics = reachability_metrics(
        artifacts["rl_edge_index"],
        int(meta["num_superzones"]),
        demand_weights,
    )
    print("Built superzone artifacts")
    print(json.dumps(meta, indent=2))
    print("RL action graph reachability")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main(parse_args())
