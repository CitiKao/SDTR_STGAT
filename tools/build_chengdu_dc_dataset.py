#!/usr/bin/env python
"""Build a Chengdu taxi D/C dataset compatible with the STDR loaders.

The Chengdu raw archives are daily ``*_train.zip`` files with rows shaped as:

    taxi_id, latitude, longitude, occupied, timestamp

Demand is inferred from occupied-state transitions ``0 -> 1``. Supply is the
number of unique empty taxis per zone and slot, counted once from each taxi's
last observed empty position in that slot. Since the raw Chengdu rows do not
contain speed, edge speeds are estimated from consecutive GPS observations for
the same taxi after temporal sorting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import BinaryIO

import numpy as np

from build_shenzhen_dc_dataset import (
    Color,
    atomic_save_npy,
    atomic_write_text,
    build_grid_metadata,
    build_knn_graph,
    build_observed_time_mask,
    build_time_features,
    build_time_meta,
    export_benchmark,
    fill_edge_speeds,
    fmt_duration,
    haversine_km,
    missing_dates_between,
    parse_float,
    pct,
    validate_outputs,
    write_time_meta,
    write_zone_info,
    zone_from_lonlat,
)


DATE_RE = re.compile(r"(20\d{6})")
DEFAULT_BBOX = (103.85, 104.22, 30.52, 30.80)  # lon_min, lon_max, lat_min, lat_max


def c(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color}{text}{Color.RESET}"


def parse_archive_date(path: Path) -> date | None:
    match = DATE_RE.search(path.name)
    if match is None:
        return None
    raw = match.group(1)
    return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))


def parse_bbox(raw: str) -> tuple[float, float, float, float]:
    if raw.lower() == "default":
        return DEFAULT_BBOX
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--bbox must be default or lon_min,lon_max,lat_min,lat_max")
    lon_min, lon_max, lat_min, lat_max = parts
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError("Invalid bbox: min values must be smaller than max values")
    return lon_min, lon_max, lat_min, lat_max


def parse_cd_time(
    raw: bytes,
    date_to_offset: dict[str, int],
    slots_per_day: int,
    slot_minutes: int,
) -> tuple[int, int, int, str] | None:
    value = raw.strip().decode("ascii", errors="ignore")
    # Expected examples: 2014/8/3 21:18:46, 2014/08/03 06:01:22
    try:
        date_text, clock_text = value.split(" ", 1)
        year_s, month_s, day_s = date_text.split("/")
        hour_s, minute_s, second_s = clock_text.split(":")
        year = int(year_s)
        month = int(month_s)
        day = int(day_s)
        hour = int(hour_s)
        minute = int(minute_s)
        second = int(float(second_s))
    except Exception:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    iso_date = f"{year:04d}-{month:02d}-{day:02d}"
    day_offset = date_to_offset.get(iso_date)
    if day_offset is None:
        return None
    slot = hour * (60 // slot_minutes) + minute // slot_minutes
    if slot < 0 or slot >= slots_per_day:
        return None
    time_idx = day_offset * slots_per_day + slot
    absolute_minute = day_offset * 1440 + hour * 60 + minute
    absolute_second = day_offset * 86400 + hour * 3600 + minute * 60 + second
    return time_idx, absolute_minute, absolute_second, iso_date


@dataclass
class TaxiState:
    prev_status: int | None = None
    prev_zone: int | None = None
    prev_time_idx: int | None = None
    prev_abs_minute: int | None = None
    prev_abs_second: int | None = None
    prev_lon: float | None = None
    prev_lat: float | None = None
    empty_slot: int | None = None
    empty_zone: int | None = None


@dataclass
class ConvertStats:
    archives: int = 0
    compressed_bytes: int = 0
    records: int = 0
    parsed_rows: int = 0
    bad_columns: int = 0
    bad_time: int = 0
    bad_status: int = 0
    valid_coord: int = 0
    bbox_hits: int = 0
    speed_rows: int = 0
    time_backwards: int = 0
    transition_gap_skipped: int = 0
    speed_gap_skipped: int = 0
    demand_events: int = 0
    dropoff_events: int = 0
    supply_observations: int = 0
    unique_taxis: set[bytes] | None = None

    def __post_init__(self) -> None:
        if self.unique_taxis is None:
            self.unique_taxis = set()


def flush_empty_supply(state: TaxiState, supply: np.ndarray, stats: ConvertStats) -> None:
    if state.empty_slot is None or state.empty_zone is None:
        return
    if 0 <= state.empty_slot < supply.shape[0] and 0 <= state.empty_zone < supply.shape[1]:
        supply[state.empty_slot, state.empty_zone] += 1.0
        stats.supply_observations += 1
    state.empty_slot = None
    state.empty_zone = None


def apply_observation(
    taxi_id: bytes,
    observation: tuple[int, int, int, int, int, float, float],
    *,
    states: dict[bytes, TaxiState],
    demand: np.ndarray,
    supply: np.ndarray,
    dropoff: np.ndarray,
    speed_sum: np.ndarray,
    speed_count: np.ndarray,
    stats: ConvertStats,
    max_transition_gap_minutes: int,
    max_speed_gap_seconds: int,
) -> None:
    time_idx, absolute_minute, absolute_second, status, zone, lon, lat = observation
    state = states.get(taxi_id)
    if state is None:
        state = TaxiState()
        states[taxi_id] = state

    if state.prev_abs_second is not None and absolute_second < state.prev_abs_second:
        stats.time_backwards += 1
        return

    if (
        state.prev_abs_second is not None
        and state.prev_lon is not None
        and state.prev_lat is not None
        and absolute_second > state.prev_abs_second
    ):
        gap_seconds = absolute_second - state.prev_abs_second
        if gap_seconds <= max_speed_gap_seconds:
            dist_km = haversine_km(state.prev_lon, state.prev_lat, lon, lat)
            speed_kmh = dist_km / (gap_seconds / 3600.0)
            if math.isfinite(speed_kmh) and 0.0 <= speed_kmh <= 160.0:
                speed_sum[time_idx, zone] += float(speed_kmh)
                speed_count[time_idx, zone] += 1
                stats.speed_rows += 1
        else:
            stats.speed_gap_skipped += 1

    if status == 0:
        if state.empty_slot is None:
            state.empty_slot = time_idx
            state.empty_zone = zone
        elif time_idx == state.empty_slot:
            state.empty_zone = zone
        elif time_idx > state.empty_slot:
            flush_empty_supply(state, supply, stats)
            state.empty_slot = time_idx
            state.empty_zone = zone

    if state.prev_status is not None and state.prev_abs_minute is not None:
        gap = absolute_minute - state.prev_abs_minute
        if gap <= max_transition_gap_minutes:
            if state.prev_status == 0 and status == 1:
                demand[time_idx, zone] += 1.0
                stats.demand_events += 1
                flush_empty_supply(state, supply, stats)
            elif state.prev_status == 1 and status == 0:
                dropoff[time_idx, zone] += 1.0
                stats.dropoff_events += 1
        else:
            stats.transition_gap_skipped += 1

    state.prev_status = status
    state.prev_zone = zone
    state.prev_time_idx = time_idx
    state.prev_abs_minute = absolute_minute
    state.prev_abs_second = absolute_second
    state.prev_lon = lon
    state.prev_lat = lat


def process_taxi_group(
    taxi_id: bytes,
    observations: list[tuple[int, int, int, int, int, float, float]],
    *,
    states: dict[bytes, TaxiState],
    demand: np.ndarray,
    supply: np.ndarray,
    dropoff: np.ndarray,
    speed_sum: np.ndarray,
    speed_count: np.ndarray,
    stats: ConvertStats,
    max_transition_gap_minutes: int,
    max_speed_gap_seconds: int,
) -> None:
    observations.sort(key=lambda item: item[2])
    for observation in observations:
        apply_observation(
            taxi_id,
            observation,
            states=states,
            demand=demand,
            supply=supply,
            dropoff=dropoff,
            speed_sum=speed_sum,
            speed_count=speed_count,
            stats=stats,
            max_transition_gap_minutes=max_transition_gap_minutes,
            max_speed_gap_seconds=max_speed_gap_seconds,
        )


def progress_line(
    *,
    archive_index: int,
    archive_count: int,
    archive_name: str,
    archive_rows: int,
    stats: ConvertStats,
    start_time: float,
    color: bool,
) -> str:
    elapsed = time.time() - start_time
    label = c("[Chengdu DC]", Color.CYAN, color)
    shard = c(f"{archive_index}/{archive_count}", Color.YELLOW, color)
    return (
        f"{label} archive={shard} file={archive_name} rows={archive_rows:,} "
        f"total={stats.records:,} taxis={len(stats.unique_taxis or []):,} "
        f"D={stats.demand_events:,} Dropoff={stats.dropoff_events:,} C={stats.supply_observations:,} "
        f"bbox={pct(stats.bbox_hits, max(stats.valid_coord, 1)):.3f}% "
        f"bad_status={pct(stats.bad_status, max(stats.records, 1)):.3f}% "
        f"speed_rows={stats.speed_rows:,} elapsed={fmt_duration(elapsed)}"
    )


def process_stream(
    stream: BinaryIO,
    *,
    archive_name: str,
    archive_index: int,
    archive_count: int,
    start_time: float,
    stats: ConvertStats,
    states: dict[bytes, TaxiState],
    demand: np.ndarray,
    supply: np.ndarray,
    dropoff: np.ndarray,
    speed_sum: np.ndarray,
    speed_count: np.ndarray,
    bbox: tuple[float, float, float, float],
    rows: int,
    cols: int,
    date_to_offset: dict[str, int],
    slots_per_day: int,
    slot_minutes: int,
    max_transition_gap_minutes: int,
    max_speed_gap_seconds: int,
    progress_rows: int,
    bucket_count: int,
    bucket_root: Path,
    color: bool,
) -> int:
    archive_rows = 0
    next_progress = progress_rows
    lon_min, lon_max, lat_min, lat_max = bbox
    if bucket_root.exists():
        shutil.rmtree(bucket_root)
    bucket_root.mkdir(parents=True, exist_ok=True)
    bucket_paths = [bucket_root / f"bucket_{idx:04d}.tsv" for idx in range(bucket_count)]
    bucket_files = [path.open("wb", buffering=1024 * 1024) for path in bucket_paths]
    sequence = 0

    try:
        for line in stream:
            if not line.strip():
                continue
            parts = line.rstrip(b"\r\n").split(b",")
            if len(parts) < 5:
                stats.bad_columns += 1
                continue
            stats.records += 1
            archive_rows += 1
            sequence += 1

            taxi_id = parts[0].strip()
            if not taxi_id:
                stats.bad_columns += 1
                continue
            stats.unique_taxis.add(taxi_id)

            lat = parse_float(parts[1])
            lon = parse_float(parts[2])
            if lon is None or lat is None or not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                continue
            stats.valid_coord += 1

            status_raw = parts[3].strip()
            if status_raw == b"0":
                status = 0
            elif status_raw == b"1":
                status = 1
            else:
                stats.bad_status += 1
                continue

            parsed_time = parse_cd_time(parts[4], date_to_offset, slots_per_day, slot_minutes)
            if parsed_time is None:
                stats.bad_time += 1
                continue
            time_idx, absolute_minute, absolute_second, _date_text = parsed_time
            stats.parsed_rows += 1

            zone = zone_from_lonlat(
                lon,
                lat,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
                rows=rows,
                cols=cols,
            )
            if zone is None:
                continue
            stats.bbox_hits += 1

            bucket_idx = zlib.crc32(taxi_id) % bucket_count
            bucket_files[bucket_idx].write(
                b"%b\t%d\t%d\t%d\t%d\t%d\t%.8f\t%.8f\t%d\n"
                % (taxi_id, absolute_minute, absolute_second, time_idx, status, zone, lon, lat, sequence)
            )

            if progress_rows > 0 and archive_rows >= next_progress:
                print(
                    progress_line(
                        archive_index=archive_index,
                        archive_count=archive_count,
                        archive_name=archive_name,
                        archive_rows=archive_rows,
                        stats=stats,
                        start_time=start_time,
                        color=color,
                    )
                    + " phase=stream",
                    flush=True,
                )
                next_progress += progress_rows
    finally:
        for fh in bucket_files:
            fh.close()

    bucket_progress_step = max(bucket_count // 8, 1)
    for bucket_idx, bucket_path in enumerate(bucket_paths, start=1):
        records: list[tuple[bytes, int, int, int, int, int, float, float, int]] = []
        if bucket_path.exists() and bucket_path.stat().st_size > 0:
            with bucket_path.open("rb") as fh:
                for raw in fh:
                    parts = raw.rstrip(b"\n").split(b"\t")
                    if len(parts) != 9:
                        continue
                    records.append(
                        (
                            parts[0],
                            int(parts[1]),
                            int(parts[2]),
                            int(parts[3]),
                            int(parts[4]),
                            int(parts[5]),
                            float(parts[6]),
                            float(parts[7]),
                            int(parts[8]),
                        )
                    )
        records.sort(key=lambda item: (item[0], item[2], item[8]))
        current_taxi: bytes | None = None
        current_observations: list[tuple[int, int, int, int, int, float, float]] = []
        for taxi_id, absolute_minute, absolute_second, time_idx, status, zone, lon, lat, _sequence in records:
            if current_taxi is None:
                current_taxi = taxi_id
            elif taxi_id != current_taxi:
                process_taxi_group(
                    current_taxi,
                    current_observations,
                    states=states,
                    demand=demand,
                    supply=supply,
                    dropoff=dropoff,
                    speed_sum=speed_sum,
                    speed_count=speed_count,
                    stats=stats,
                    max_transition_gap_minutes=max_transition_gap_minutes,
                    max_speed_gap_seconds=max_speed_gap_seconds,
                )
                current_taxi = taxi_id
                current_observations = []
            current_observations.append((time_idx, absolute_minute, absolute_second, status, zone, lon, lat))
        if current_taxi is not None and current_observations:
            process_taxi_group(
                current_taxi,
                current_observations,
                states=states,
                demand=demand,
                supply=supply,
                dropoff=dropoff,
                speed_sum=speed_sum,
                speed_count=speed_count,
                stats=stats,
                max_transition_gap_minutes=max_transition_gap_minutes,
                max_speed_gap_seconds=max_speed_gap_seconds,
            )
        try:
            bucket_path.unlink()
        except FileNotFoundError:
            pass
        if bucket_idx % bucket_progress_step == 0 or bucket_idx == bucket_count:
            print(
                progress_line(
                    archive_index=archive_index,
                    archive_count=archive_count,
                    archive_name=archive_name,
                    archive_rows=archive_rows,
                    stats=stats,
                    start_time=start_time,
                    color=color,
                )
                + f" phase=bucket bucket={bucket_idx}/{bucket_count}",
                flush=True,
            )
    shutil.rmtree(bucket_root, ignore_errors=True)
    return archive_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chengdu taxi D/C dataset for STDR.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/external_datasets/raw/dc_candidate_downloads/cd_taxi_data"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/chengdu_dc"))
    parser.add_argument("--benchmark-output-dir", type=Path, default=Path("data/chengdu_dc_benchmark"))
    parser.add_argument("--slot-minutes", type=int, default=15)
    parser.add_argument("--grid-rows", type=int, default=16)
    parser.add_argument("--grid-cols", type=int, default=17)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--bbox", default="default", help="default or lon_min,lon_max,lat_min,lat_max")
    parser.add_argument("--max-transition-gap-minutes", type=int, default=180)
    parser.add_argument("--max-speed-gap-seconds", type=int, default=300)
    parser.add_argument("--progress-rows", type=int, default=5_000_000)
    parser.add_argument("--bucket-count", type=int, default=128)
    parser.add_argument("--max-archives", type=int, default=0)
    parser.add_argument("--hist-len", type=int, default=14)
    parser.add_argument("--pred-horizon", type=int, default=4)
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if 1440 % args.slot_minutes != 0:
        raise SystemExit("--slot-minutes must divide 1440")
    if args.grid_rows <= 0 or args.grid_cols <= 0:
        raise SystemExit("grid rows/cols must be positive")
    if args.knn <= 0:
        raise SystemExit("--knn must be positive")
    if args.bucket_count <= 0:
        raise SystemExit("--bucket-count must be positive")

    color = not args.no_color
    bbox = parse_bbox(args.bbox)
    archives = sorted(args.input_dir.glob("*_train.zip"))
    if args.max_archives > 0:
        archives = archives[: args.max_archives]
    if not archives:
        raise SystemExit(f"No *_train.zip archives found in {args.input_dir}")

    archive_dates_raw = [parse_archive_date(path) for path in archives]
    if any(d is None for d in archive_dates_raw):
        raise SystemExit("Every archive name must contain a YYYYMMDD date")
    archive_dates = [d for d in archive_dates_raw if d is not None]
    missing_archive_dates = missing_dates_between(archive_dates)
    start_date = min(archive_dates)
    end_date = max(archive_dates)

    slots_per_day = 1440 // args.slot_minutes
    time_meta, date_to_offset = build_time_meta(start_date, end_date, args.slot_minutes)
    t_steps = len(time_meta)
    n_nodes = args.grid_rows * args.grid_cols
    stats = ConvertStats(
        archives=len(archives),
        compressed_bytes=sum(path.stat().st_size for path in archives),
    )

    print(
        c("[Chengdu DC]", Color.CYAN, color)
        + f" input={args.input_dir} output={args.output_dir} "
        + c(f"nodes={n_nodes} ({args.grid_rows}x{args.grid_cols})", Color.YELLOW, color)
        + f" dates={start_date}..{end_date} T={t_steps} slot={args.slot_minutes}min "
        + f"archives={len(archives)} compressed={stats.compressed_bytes / 1024**3:.2f}GB",
        flush=True,
    )
    if missing_archive_dates:
        print(
            c("[Chengdu DC]", Color.YELLOW, color)
            + f" missing archive dates between start/end: {missing_archive_dates}; "
            + "time_meta keeps continuous calendar slots and marks them unobserved.",
            flush=True,
        )

    demand = np.zeros((t_steps, n_nodes), dtype=np.float32)
    supply = np.zeros((t_steps, n_nodes), dtype=np.float32)
    dropoff = np.zeros((t_steps, n_nodes), dtype=np.float32)
    speed_sum = np.zeros((t_steps, n_nodes), dtype=np.float64)
    speed_count = np.zeros((t_steps, n_nodes), dtype=np.int32)
    states: dict[bytes, TaxiState] = {}
    bucket_base = args.output_dir / "_tmp_order_buckets"

    start_time = time.time()
    archive_summaries: list[dict[str, object]] = []
    for archive_index, archive in enumerate(archives, start=1):
        shard_start = time.time()
        print(
            c("[Chengdu DC]", Color.CYAN, color)
            + f" start {c(f'{archive_index}/{len(archives)}', Color.YELLOW, color)} file={archive.name}",
            flush=True,
        )
        rows_before = stats.records
        with zipfile.ZipFile(archive) as zf:
            members = [info for info in zf.infolist() if not info.is_dir()]
            if len(members) != 1:
                raise RuntimeError(f"{archive} expected one data file, found {len(members)}")
            with zf.open(members[0]) as stream:
                archive_rows = process_stream(
                    stream,
                    archive_name=archive.name,
                    archive_index=archive_index,
                    archive_count=len(archives),
                    start_time=start_time,
                    stats=stats,
                    states=states,
                    demand=demand,
                    supply=supply,
                    dropoff=dropoff,
                    speed_sum=speed_sum,
                    speed_count=speed_count,
                    bbox=bbox,
                    rows=args.grid_rows,
                    cols=args.grid_cols,
                    date_to_offset=date_to_offset,
                    slots_per_day=slots_per_day,
                    slot_minutes=args.slot_minutes,
                    max_transition_gap_minutes=args.max_transition_gap_minutes,
                    max_speed_gap_seconds=args.max_speed_gap_seconds,
                    progress_rows=args.progress_rows,
                    bucket_count=args.bucket_count,
                    bucket_root=bucket_base / archive.stem,
                    color=color,
                )
        elapsed = time.time() - shard_start
        archive_summaries.append(
            {
                "archive": archive.name,
                "source": "zip",
                "rows": int(stats.records - rows_before),
                "elapsed_sec": round(elapsed, 3),
            }
        )
        print(
            progress_line(
                archive_index=archive_index,
                archive_count=len(archives),
                archive_name=archive.name,
                archive_rows=archive_rows,
                stats=stats,
                start_time=start_time,
                color=color,
            )
            + f" done_in={fmt_duration(elapsed)}",
            flush=True,
        )

    shutil.rmtree(bucket_base, ignore_errors=True)

    print(c("[Chengdu DC]", Color.CYAN, color) + " flushing open empty-taxi supply states", flush=True)
    for state in states.values():
        flush_empty_supply(state, supply, stats)

    print(c("[Chengdu DC]", Color.CYAN, color) + " building KNN graph and output arrays", flush=True)
    zone_rows, centers = build_grid_metadata(rows=args.grid_rows, cols=args.grid_cols, bbox=bbox)
    for row in zone_rows:
        row["zone_name"] = str(row["zone_name"]).replace("SZ_GRID", "CD_GRID")
        row["borough"] = "Chengdu"
    adj, edge_index, edge_lengths = build_knn_graph(centers, args.knn)
    edge_speeds = fill_edge_speeds(speed_sum, speed_count, edge_index)
    targets_dc = np.stack([demand, supply], axis=-1).astype(np.float32)
    time_features = build_time_features(time_meta, slots_per_day)
    observed_time_mask = build_observed_time_mask(time_meta, missing_archive_dates)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_save_npy(output_dir / "node_demand.npy", demand)
    atomic_save_npy(output_dir / "node_supply.npy", supply)
    atomic_save_npy(output_dir / "node_dropoff.npy", dropoff)
    atomic_save_npy(output_dir / "targets_dc.npy", targets_dc)
    atomic_save_npy(output_dir / "time_features.npy", time_features)
    atomic_save_npy(output_dir / "observed_time_mask.npy", observed_time_mask)
    atomic_save_npy(output_dir / "adjacency_matrix.npy", adj)
    atomic_save_npy(output_dir / "edge_index.npy", edge_index)
    atomic_save_npy(output_dir / "edge_lengths.npy", edge_lengths)
    atomic_save_npy(output_dir / "edge_lengths_osrm.npy", edge_lengths)
    atomic_save_npy(output_dir / "edge_speeds.npy", edge_speeds)
    write_time_meta(output_dir / "time_meta.csv", time_meta)
    write_zone_info(output_dir / "zone_info.csv", zone_rows)

    validation = validate_outputs(output_dir)
    manifest = {
        "schema_version": "chengdu_dc_grid_v1",
        "dataset_name": "stdr_chengdu_dc",
        "source_city": "Chengdu",
        "source_input_dir": str(args.input_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "spatial_partition": {
            "mode": "fixed_lonlat_grid",
            "rows": int(args.grid_rows),
            "cols": int(args.grid_cols),
            "nodes": int(n_nodes),
            "bbox_lon_min_lon_max_lat_min_lat_max": list(bbox),
            "rationale": (
                "Chengdu uses a 16x17 grid (272 nodes), close to NYC 263 zones and "
                "Shenzhen 276 nodes while matching the more compact central Chengdu "
                "taxi coverage better than Shenzhen's elongated 12x23 grid."
            ),
        },
        "temporal_partition": {
            "slot_minutes": int(args.slot_minutes),
            "slots_per_day": int(slots_per_day),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "time_steps": int(t_steps),
            "archive_dates_present": [d.isoformat() for d in sorted(archive_dates)],
            "missing_archive_dates_between_start_end": missing_archive_dates,
            "observed_time_mask_file": "observed_time_mask.npy",
            "observed_slots": int(observed_time_mask.sum()),
            "unobserved_slots": int((~observed_time_mask).sum()),
            "missing_date_policy": (
                "continuous time_meta is retained for calendar alignment; slots without raw archives "
                "are placeholders marked observed_time_mask=False, and downstream training/benchmark "
                "exports exclude any history+target window touching unobserved slots"
            ),
        },
        "target_semantics": {
            "node_demand": "pickup events inferred from occupied transition 0->1",
            "node_supply": "unique empty taxis per zone/time slot, counted at each taxi's final empty observation in the slot",
            "node_dropoff": "dropoff events inferred from occupied transition 1->0, saved for audit only",
            "edge_speeds": "GPS speeds estimated from consecutive same-taxi positions, aggregated by zone/time, then averaged across each KNN edge endpoints",
        },
        "graph": {
            "mode": "directed_spatial_knn",
            "knn": int(args.knn),
            "nodes": int(adj.shape[0]),
            "edges": int(edge_index.shape[0]),
            "edge_length_unit": "km",
        },
        "raw_stats": {
            "archives": int(stats.archives),
            "compressed_gb": round(stats.compressed_bytes / 1024**3, 6),
            "records": int(stats.records),
            "parsed_rows": int(stats.parsed_rows),
            "bad_columns": int(stats.bad_columns),
            "bad_time": int(stats.bad_time),
            "bad_status": int(stats.bad_status),
            "valid_coord": int(stats.valid_coord),
            "bbox_hits": int(stats.bbox_hits),
            "bbox_hit_pct_of_valid_coord": round(pct(stats.bbox_hits, max(stats.valid_coord, 1)), 6),
            "speed_rows": int(stats.speed_rows),
            "speed_gap_skipped": int(stats.speed_gap_skipped),
            "unique_taxis": int(len(stats.unique_taxis or [])),
            "time_backwards": int(stats.time_backwards),
            "transition_gap_skipped": int(stats.transition_gap_skipped),
            "demand_events": int(stats.demand_events),
            "dropoff_events": int(stats.dropoff_events),
            "supply_observations": int(stats.supply_observations),
        },
        "shapes": {
            "node_demand": list(demand.shape),
            "node_supply": list(supply.shape),
            "node_dropoff": list(dropoff.shape),
            "targets_dc": list(targets_dc.shape),
            "edge_speeds": list(edge_speeds.shape),
            "adjacency_matrix": list(adj.shape),
            "edge_index": list(edge_index.shape),
            "time_features": list(time_features.shape),
            "observed_time_mask": list(observed_time_mask.shape),
        },
        "archive_summaries": archive_summaries,
        "validation": validation,
        "created_from_files": {archive.name: str(archive.stat().st_size) for archive in archives},
    }
    atomic_write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    if not args.no_benchmark:
        print(c("[Chengdu DC]", Color.CYAN, color) + f" exporting benchmark to {args.benchmark_output_dir}", flush=True)
        export_benchmark(output_dir, args.benchmark_output_dir, args.hist_len, args.pred_horizon)

    print(
        c("[Chengdu DC]", Color.GREEN, color)
        + f" complete output={output_dir} validation={validation}",
        flush=True,
    )


if __name__ == "__main__":
    main()
