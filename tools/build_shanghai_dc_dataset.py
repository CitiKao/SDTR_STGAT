#!/usr/bin/env python
"""Build a Shanghai taxi D/C dataset compatible with the STDR loaders.

Raw rows in ``sh_taxi_data/<day>/part-00000.gz`` are shaped as:

    taxi_id, reserved, empty_flag, op_status, alarm, brake,
    upload_time, gps_time, longitude, latitude, speed, direction, satellite_count

The local schema identifies ``empty_flag`` as 1 for empty and 0 for occupied.
Demand is inferred from transitions empty -> occupied. Supply is the number of
unique empty taxis per zone and slot, counted once from each taxi's final empty
position in that slot.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import shutil
import sys
import time
import urllib.request
import zlib
from collections import OrderedDict
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path
from typing import BinaryIO

import numpy as np
from shapely import intersects_xy
from shapely.geometry import Point, shape
from shapely.prepared import prep

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


DEFAULT_BBOX = (121.16, 121.82, 30.91, 31.46)  # lon_min, lon_max, lat_min, lat_max
DEFAULT_START = date(2015, 4, 1)
DEFAULT_END = date(2015, 4, 30)
DEFAULT_DISTRICT_GEOJSON_URL = "https://geo.datav.aliyun.com/areas_v3/bound/310000_full.json"
DEFAULT_DISTRICT_GEOJSON = Path(
    "data/external_datasets/raw/dc_candidate_downloads/shanghai_boundaries/"
    "shanghai_datav_310000_districts.geojson"
)


def c(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color}{text}{Color.RESET}"


def parse_iso_date(raw: str) -> date:
    year, month, day = [int(part) for part in raw.split("-")]
    return date(year, month, day)


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


def bbox_with_margin(bounds: tuple[float, float, float, float], margin_ratio: float = 0.002) -> tuple[float, float, float, float]:
    lon_min, lat_min, lon_max, lat_max = bounds
    lon_margin = max((lon_max - lon_min) * margin_ratio, 1e-6)
    lat_margin = max((lat_max - lat_min) * margin_ratio, 1e-6)
    return lon_min - lon_margin, lon_max + lon_margin, lat_min - lat_margin, lat_max + lat_margin


def ensure_district_geojson(path: Path, color: bool) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(
        c("[Shanghai DC]", Color.CYAN, color)
        + f" downloading district polygons from {DEFAULT_DISTRICT_GEOJSON_URL}",
        flush=True,
    )
    with urllib.request.urlopen(DEFAULT_DISTRICT_GEOJSON_URL, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))
    return path


@dataclass
class PolygonPartition:
    zone_rows: list[dict[str, object]]
    centers: np.ndarray
    geometries: list[object]
    lookup: np.ndarray
    bbox: tuple[float, float, float, float]
    resolution: int
    exact_preps: list[object]
    raster_hits: int = 0
    exact_hits: int = 0
    misses: int = 0

    def zone_from_lonlat(self, lon: float, lat: float) -> int | None:
        lon_min, lon_max, lat_min, lat_max = self.bbox
        if lon < lon_min or lon > lon_max or lat < lat_min or lat > lat_max:
            self.misses += 1
            return None
        col = int((lon - lon_min) / (lon_max - lon_min) * self.resolution)
        row = int((lat - lat_min) / (lat_max - lat_min) * self.resolution)
        col = min(max(col, 0), self.resolution - 1)
        row = min(max(row, 0), self.resolution - 1)
        zone = int(self.lookup[row, col])
        if zone >= 0:
            self.raster_hits += 1
            return zone

        point = Point(lon, lat)
        for idx, prepared in enumerate(self.exact_preps):
            if prepared.covers(point):
                self.exact_hits += 1
                return idx
        self.misses += 1
        return None

    def stats(self) -> dict[str, int]:
        return {
            "raster_hits": int(self.raster_hits),
            "exact_hits": int(self.exact_hits),
            "misses": int(self.misses),
        }


def load_polygon_partition(
    geojson_path: Path,
    *,
    lookup_resolution: int,
    color: bool,
) -> PolygonPartition:
    if lookup_resolution <= 0:
        raise ValueError("--polygon-lookup-resolution must be positive")
    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    features = payload.get("features") or []
    if not features:
        raise ValueError(f"No features found in {geojson_path}")

    zone_rows: list[dict[str, object]] = []
    geometries: list[object] = []
    centers = np.zeros((len(features), 2), dtype=np.float64)
    for idx, feature in enumerate(features):
        props = feature.get("properties") or {}
        geom = shape(feature["geometry"]).buffer(0)
        if geom.is_empty:
            raise ValueError(f"Empty geometry for feature #{idx} in {geojson_path}")
        geometries.append(geom)
        centroid = geom.representative_point()
        minx, miny, maxx, maxy = geom.bounds
        centers[idx] = [float(centroid.x), float(centroid.y)]
        zone_rows.append(
            {
                "index": idx,
                "zone_id": str(props.get("adcode", idx)),
                "zone_name": str(props.get("name", f"SH_POLYGON_{idx:03d}")),
                "borough": "Shanghai",
                "partition_level": str(props.get("level", "district")),
                "lon_min": float(minx),
                "lon_max": float(maxx),
                "lat_min": float(miny),
                "lat_max": float(maxy),
                "lon_center": float(centroid.x),
                "lat_center": float(centroid.y),
            }
        )

    union_bounds = bbox_with_margin(
        (
            min(float(geom.bounds[0]) for geom in geometries),
            min(float(geom.bounds[1]) for geom in geometries),
            max(float(geom.bounds[2]) for geom in geometries),
            max(float(geom.bounds[3]) for geom in geometries),
        )
    )
    lon_min, lon_max, lat_min, lat_max = union_bounds
    xs = lon_min + (np.arange(lookup_resolution, dtype=np.float64) + 0.5) * (
        lon_max - lon_min
    ) / lookup_resolution
    ys = lat_min + (np.arange(lookup_resolution, dtype=np.float64) + 0.5) * (
        lat_max - lat_min
    ) / lookup_resolution
    dtype = np.int16 if len(geometries) < np.iinfo(np.int16).max else np.int32
    lookup = np.full((lookup_resolution, lookup_resolution), -1, dtype=dtype)

    print(
        c("[Shanghai DC]", Color.CYAN, color)
        + f" rasterizing polygon lookup zones={len(geometries)} resolution={lookup_resolution}x{lookup_resolution}",
        flush=True,
    )
    chunk_rows = 64
    for row_start in range(0, lookup_resolution, chunk_rows):
        row_end = min(row_start + chunk_rows, lookup_resolution)
        xx, yy = np.meshgrid(xs, ys[row_start:row_end])
        chunk = lookup[row_start:row_end]
        unassigned = chunk < 0
        for zone_idx, geom in enumerate(geometries):
            mask = intersects_xy(geom, xx, yy)
            chunk[unassigned & mask] = zone_idx
            unassigned = chunk < 0
            if not unassigned.any():
                break
        if row_start == 0 or row_end == lookup_resolution or row_end % max(lookup_resolution // 8, 1) == 0:
            filled = int((lookup >= 0).sum())
            print(
                c("[Shanghai DC]", Color.CYAN, color)
                + f" polygon lookup rows={row_end}/{lookup_resolution} filled={filled:,}",
                flush=True,
            )

    return PolygonPartition(
        zone_rows=zone_rows,
        centers=centers,
        geometries=geometries,
        lookup=lookup,
        bbox=union_bounds,
        resolution=lookup_resolution,
        exact_preps=[prep(geom) for geom in geometries],
    )


def build_polygon_adjacency(
    geometries: list[object],
    centers: np.ndarray,
    *,
    touch_tolerance_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    n = len(geometries)
    lengths = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        lon_i, lat_i = centers[i]
        for j in range(n):
            if i == j:
                continue
            lon_j, lat_j = centers[j]
            lengths[i, j] = haversine_km(float(lon_i), float(lat_i), float(lon_j), float(lat_j))

    undirected_edges: list[tuple[int, int]] = []
    for i in range(n):
        boundary_i = geometries[i].boundary
        for j in range(i + 1, n):
            boundary_intersection = boundary_i.intersection(geometries[j].boundary)
            shared_boundary = (not boundary_intersection.is_empty) and (
                float(boundary_intersection.length) > touch_tolerance_deg
            )
            if shared_boundary or geometries[i].touches(geometries[j]):
                undirected_edges.append((i, j))

    edges: list[tuple[int, int]] = []
    for i, j in undirected_edges:
        edges.append((i, j))
        edges.append((j, i))
    edge_index = np.array(edges, dtype=np.int32)
    adj = np.zeros((n, n), dtype=np.float32)
    if edge_index.size:
        adj[edge_index[:, 0], edge_index[:, 1]] = 1.0
    degree = adj.sum(axis=1).astype(np.int32)
    graph_stats = {
        "mode": "polygon_boundary_adjacency",
        "undirected_edges": int(len(undirected_edges)),
        "directed_edges": int(edge_index.shape[0]),
        "isolated_nodes": [int(i) for i, value in enumerate(degree.tolist()) if int(value) == 0],
        "degree_min": int(degree.min()) if n else 0,
        "degree_max": int(degree.max()) if n else 0,
    }
    return adj, edge_index, lengths.astype(np.float32), graph_stats


def parse_sh_time(
    raw: bytes,
    date_to_offset: dict[str, int],
    slots_per_day: int,
    slot_minutes: int,
) -> tuple[int, int, int, str] | None:
    value = raw.strip().decode("ascii", errors="ignore")
    try:
        date_text, clock_text = value.split(" ", 1)
        year_s, month_s, day_s = date_text.split("-")
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


def parse_speed(raw: bytes, max_speed_kmh: float) -> float | None:
    speed = parse_float(raw)
    if speed is None or not math.isfinite(speed):
        return None
    if speed < 0.0 or speed > max_speed_kmh:
        return None
    return float(speed)


@dataclass
class TaxiState:
    prev_status: int | None = None
    prev_time_idx: int | None = None
    prev_abs_minute: int | None = None
    prev_abs_second: int | None = None
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
    date_mismatch: int = 0
    bad_empty_flag: int = 0
    valid_coord: int = 0
    bbox_hits: int = 0
    speed_rows: int = 0
    bad_speed: int = 0
    time_backwards: int = 0
    transition_gap_skipped: int = 0
    demand_events: int = 0
    dropoff_events: int = 0
    supply_observations: int = 0
    unique_taxis: set[bytes] | None = None

    def __post_init__(self) -> None:
        if self.unique_taxis is None:
            self.unique_taxis = set()

    def to_jsonable(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "unique_taxis":
                payload["unique_taxis"] = len(value or [])
            else:
                payload[field.name] = value
        return payload


class LruBucketWriter:
    def __init__(self, bucket_paths: list[Path], max_open: int) -> None:
        self.bucket_paths = bucket_paths
        self.max_open = max(1, int(max_open))
        self.handles: OrderedDict[int, BinaryIO] = OrderedDict()

    def write(self, bucket_idx: int, payload: bytes) -> None:
        handle = self.handles.get(bucket_idx)
        if handle is None:
            handle = self.bucket_paths[bucket_idx].open("ab", buffering=1024 * 1024)
            self.handles[bucket_idx] = handle
            if len(self.handles) > self.max_open:
                _, old = self.handles.popitem(last=False)
                old.close()
        else:
            self.handles.move_to_end(bucket_idx)
        handle.write(payload)

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


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
    observation: tuple[int, int, int, int, int, float | None],
    *,
    states: dict[bytes, TaxiState],
    demand: np.ndarray,
    supply: np.ndarray,
    dropoff: np.ndarray,
    speed_sum: np.ndarray,
    speed_count: np.ndarray,
    stats: ConvertStats,
    max_transition_gap_minutes: int,
) -> None:
    time_idx, absolute_minute, absolute_second, status, zone, speed = observation
    if speed is not None:
        speed_sum[time_idx, zone] += float(speed)
        speed_count[time_idx, zone] += 1
        stats.speed_rows += 1

    state = states.get(taxi_id)
    if state is None:
        state = TaxiState()
        states[taxi_id] = state

    if state.prev_abs_second is not None and absolute_second < state.prev_abs_second:
        stats.time_backwards += 1
        return

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
    state.prev_time_idx = time_idx
    state.prev_abs_minute = absolute_minute
    state.prev_abs_second = absolute_second


def process_taxi_group(
    taxi_id: bytes,
    observations: list[tuple[int, int, int, int, int, float | None]],
    *,
    states: dict[bytes, TaxiState],
    demand: np.ndarray,
    supply: np.ndarray,
    dropoff: np.ndarray,
    speed_sum: np.ndarray,
    speed_count: np.ndarray,
    stats: ConvertStats,
    max_transition_gap_minutes: int,
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
        )


def progress_line(
    *,
    archive_index: int,
    archive_count: int,
    archive_name: str,
    archive_rows: int,
    stats: ConvertStats,
    start_time: float,
    phase: str,
    color: bool,
) -> str:
    elapsed = time.time() - start_time
    label = c("[Shanghai DC]", Color.CYAN, color)
    shard = c(f"{archive_index}/{archive_count}", Color.YELLOW, color)
    bbox_pct = pct(stats.bbox_hits, max(stats.valid_coord, 1))
    speed_pct = pct(stats.speed_rows, max(stats.parsed_rows, 1))
    return (
        f"{label} phase={phase} file={shard}:{archive_name} rows={archive_rows:,} "
        f"total={stats.records:,} taxis={len(stats.unique_taxis or []):,} "
        f"D={stats.demand_events:,} Dropoff={stats.dropoff_events:,} C={stats.supply_observations:,} "
        f"bbox={bbox_pct:.3f}% speed={speed_pct:.3f}% "
        f"bad_time={stats.bad_time:,} date_mismatch={stats.date_mismatch:,} "
        f"elapsed={fmt_duration(elapsed)}"
    )


def process_stream(
    stream: BinaryIO,
    *,
    archive_name: str,
    expected_date: str,
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
    polygon_partition: PolygonPartition | None,
    date_to_offset: dict[str, int],
    slots_per_day: int,
    slot_minutes: int,
    strict_folder_date: bool,
    max_transition_gap_minutes: int,
    max_speed_kmh: float,
    progress_rows: int,
    max_rows_per_day: int,
    bucket_count: int,
    max_open_buckets: int,
    bucket_root: Path,
    color: bool,
) -> int:
    archive_rows = 0
    next_progress = progress_rows
    lon_min, lon_max, lat_min, lat_max = bbox
    if bucket_root.exists():
        shutil.rmtree(bucket_root)
    bucket_root.mkdir(parents=True, exist_ok=True)
    bucket_paths = [bucket_root / f"bucket_{idx:05d}.tsv" for idx in range(bucket_count)]
    writer = LruBucketWriter(bucket_paths, max_open=max_open_buckets)
    sequence = 0

    try:
        for line in stream:
            if not line.strip():
                continue
            parts = line.rstrip(b"\r\n").split(b",")
            if len(parts) < 13:
                stats.bad_columns += 1
                continue
            stats.records += 1
            archive_rows += 1
            sequence += 1
            if max_rows_per_day > 0 and archive_rows > max_rows_per_day:
                break

            taxi_id = parts[0].strip()
            if not taxi_id:
                stats.bad_columns += 1
                continue
            stats.unique_taxis.add(taxi_id)

            empty_raw = parts[2].strip()
            if empty_raw == b"1":
                status = 0  # empty / available
            elif empty_raw == b"0":
                status = 1  # occupied
            else:
                stats.bad_empty_flag += 1
                continue

            parsed_time = parse_sh_time(parts[7], date_to_offset, slots_per_day, slot_minutes)
            if parsed_time is None:
                stats.bad_time += 1
                continue
            time_idx, absolute_minute, absolute_second, date_text = parsed_time
            if strict_folder_date and date_text != expected_date:
                stats.date_mismatch += 1
                continue

            lon = parse_float(parts[8])
            lat = parse_float(parts[9])
            if lon is None or lat is None or not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                continue
            stats.valid_coord += 1

            if polygon_partition is None:
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
            else:
                zone = polygon_partition.zone_from_lonlat(lon, lat)
            if zone is None:
                continue
            stats.bbox_hits += 1

            speed = parse_speed(parts[10], max_speed_kmh)
            if speed is None:
                stats.bad_speed += 1
                speed_valid = 0
                speed_value = 0.0
            else:
                speed_valid = 1
                speed_value = speed
            stats.parsed_rows += 1

            bucket_idx = zlib.crc32(taxi_id) % bucket_count
            writer.write(
                bucket_idx,
                b"%b\t%d\t%d\t%d\t%d\t%d\t%d\t%.4f\t%d\n"
                % (
                    taxi_id,
                    absolute_minute,
                    absolute_second,
                    time_idx,
                    status,
                    zone,
                    speed_valid,
                    speed_value,
                    sequence,
                ),
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
                        phase="stream",
                        color=color,
                    ),
                    flush=True,
                )
                next_progress += progress_rows
    finally:
        writer.close()

    bucket_progress_step = max(bucket_count // 20, 1)
    for bucket_idx, bucket_path in enumerate(bucket_paths, start=1):
        records: list[tuple[bytes, int, int, int, int, int, float | None, int]] = []
        if bucket_path.exists() and bucket_path.stat().st_size > 0:
            with bucket_path.open("rb") as fh:
                for raw in fh:
                    parts = raw.rstrip(b"\n").split(b"\t")
                    if len(parts) != 9:
                        continue
                    speed = float(parts[7]) if int(parts[6]) else None
                    records.append(
                        (
                            parts[0],
                            int(parts[1]),
                            int(parts[2]),
                            int(parts[3]),
                            int(parts[4]),
                            int(parts[5]),
                            speed,
                            int(parts[8]),
                        )
                    )
        records.sort(key=lambda item: (item[0], item[2], item[7]))
        current_taxi: bytes | None = None
        current_observations: list[tuple[int, int, int, int, int, float | None]] = []
        for taxi_id, absolute_minute, absolute_second, time_idx, status, zone, speed, _sequence in records:
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
                )
                current_taxi = taxi_id
                current_observations = []
            current_observations.append((time_idx, absolute_minute, absolute_second, status, zone, speed))
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
                    phase=f"bucket {bucket_idx}/{bucket_count}",
                    color=color,
                ),
                flush=True,
            )
    if not any(bucket_root.iterdir()):
        bucket_root.rmdir()
    return archive_rows


def discover_archives(input_dir: Path, start: date, end: date) -> list[tuple[date, Path]]:
    archives: list[tuple[date, Path]] = []
    current = start
    while current <= end:
        day_number = (current - start).days + 1
        path = input_dir / str(day_number) / "part-00000.gz"
        if path.exists():
            archives.append((current, path))
        current = date.fromordinal(current.toordinal() + 1)
    return archives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Shanghai taxi D/C dataset for STDR.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/external_datasets/raw/dc_candidate_downloads/sh_taxi_data"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/shanghai_dc"))
    parser.add_argument("--benchmark-output-dir", type=Path, default=Path("data/shanghai_dc_benchmark"))
    parser.add_argument("--start-date", default=DEFAULT_START.isoformat())
    parser.add_argument("--end-date", default=DEFAULT_END.isoformat())
    parser.add_argument("--slot-minutes", type=int, default=15)
    parser.add_argument(
        "--partition-mode",
        choices=("grid_knn", "district_polygon"),
        default="grid_knn",
        help="grid_knn keeps the old fixed-grid KNN graph; district_polygon uses real Shanghai district polygons and boundary adjacency.",
    )
    parser.add_argument("--grid-rows", type=int, default=20)
    parser.add_argument("--grid-cols", type=int, default=20)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--bbox", default="default", help="default or lon_min,lon_max,lat_min,lat_max")
    parser.add_argument("--boundary-geojson", type=Path, default=DEFAULT_DISTRICT_GEOJSON)
    parser.add_argument("--polygon-lookup-resolution", type=int, default=1600)
    parser.add_argument("--polygon-touch-tolerance-deg", type=float, default=1e-7)
    parser.add_argument("--max-transition-gap-minutes", type=int, default=180)
    parser.add_argument("--max-speed-kmh", type=float, default=160.0)
    parser.add_argument("--progress-rows", type=int, default=1_000_000)
    parser.add_argument("--bucket-count", type=int, default=1024)
    parser.add_argument("--max-open-buckets", type=int, default=1024)
    parser.add_argument("--tmp-dir", type=Path, default=None)
    parser.add_argument("--hist-len", type=int, default=14)
    parser.add_argument("--pred-horizon", type=int, default=4)
    parser.add_argument("--max-rows-per-day", type=int, default=0, help="Smoke-test limit; 0 means full files.")
    parser.add_argument("--allow-cross-date", action="store_true", help="Keep rows whose GPS date differs from folder date.")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    color = not args.no_color
    start_time = time.time()
    input_dir = args.input_dir
    output_dir = args.output_dir
    benchmark_dir = args.benchmark_output_dir
    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    if args.slot_minutes <= 0 or 60 % args.slot_minutes != 0:
        raise SystemExit("--slot-minutes must divide 60")
    if args.partition_mode == "grid_knn" and (args.grid_rows <= 0 or args.grid_cols <= 0):
        raise SystemExit("--grid-rows/--grid-cols must be positive")
    if args.bucket_count <= 0:
        raise SystemExit("--bucket-count must be positive")

    polygon_partition: PolygonPartition | None = None
    if args.partition_mode == "district_polygon":
        boundary_path = ensure_district_geojson(args.boundary_geojson, color)
        polygon_partition = load_polygon_partition(
            boundary_path,
            lookup_resolution=args.polygon_lookup_resolution,
            color=color,
        )
        bbox = polygon_partition.bbox if args.bbox.lower() == "default" else parse_bbox(args.bbox)
        n_nodes = len(polygon_partition.zone_rows)
    else:
        bbox = parse_bbox(args.bbox)
        n_nodes = args.grid_rows * args.grid_cols

    archives = discover_archives(input_dir, start_date, end_date)
    if not archives:
        raise FileNotFoundError(f"No Shanghai archives found under {input_dir}")
    archive_dates = [day for day, _ in archives]
    missing_archive_dates = missing_dates_between(archive_dates)
    time_meta, date_to_offset = build_time_meta(start_date, end_date, args.slot_minutes)
    t_steps = len(time_meta)
    slots_per_day = 24 * (60 // args.slot_minutes)

    print("=" * 96, flush=True)
    print(
        c("[Shanghai DC]", Color.CYAN, color)
        + f" input={input_dir} archives={len(archives)} dates={start_date}..{end_date} "
        + c(f"nodes={n_nodes} partition={args.partition_mode}", Color.YELLOW, color),
        flush=True,
    )
    print(
        c("[Shanghai DC]", Color.CYAN, color)
        + f" bbox={bbox} slot={args.slot_minutes}min T={t_steps} "
        + "empty_flag mapping: 1=empty/supply, 0=occupied; demand is 1->0 transition",
        flush=True,
    )
    if args.max_rows_per_day > 0:
        print(
            c("[Shanghai DC smoke]", Color.YELLOW, color)
            + f" max_rows_per_day={args.max_rows_per_day:,}; output is for validation only",
            flush=True,
        )
    print("=" * 96, flush=True)

    demand = np.zeros((t_steps, n_nodes), dtype=np.float32)
    supply = np.zeros((t_steps, n_nodes), dtype=np.float32)
    dropoff = np.zeros((t_steps, n_nodes), dtype=np.float32)
    speed_sum = np.zeros((t_steps, n_nodes), dtype=np.float64)
    speed_count = np.zeros((t_steps, n_nodes), dtype=np.float64)
    states: dict[bytes, TaxiState] = {}
    stats = ConvertStats()
    stats.archives = len(archives)
    stats.compressed_bytes = sum(path.stat().st_size for _, path in archives)

    tmp_root = args.tmp_dir or (output_dir / "_tmp_shanghai_buckets")
    try:
        for archive_index, (archive_date, path) in enumerate(archives, start=1):
            expected_date = archive_date.isoformat()
            bucket_root = tmp_root / f"day_{archive_index:02d}_{expected_date}"
            archive_start = time.time()
            print(
                c("[Shanghai DC]", Color.CYAN, color)
                + f" start file {archive_index}/{len(archives)} {path} "
                + f"size={path.stat().st_size / 1e9:.2f}GB expected_date={expected_date}",
                flush=True,
            )
            with gzip.open(path, "rb") as stream:
                archive_rows = process_stream(
                    stream,
                    archive_name=path.name,
                    expected_date=expected_date,
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
                    polygon_partition=polygon_partition,
                    date_to_offset=date_to_offset,
                    slots_per_day=slots_per_day,
                    slot_minutes=args.slot_minutes,
                    strict_folder_date=not args.allow_cross_date,
                    max_transition_gap_minutes=args.max_transition_gap_minutes,
                    max_speed_kmh=args.max_speed_kmh,
                    progress_rows=args.progress_rows,
                    max_rows_per_day=args.max_rows_per_day,
                    bucket_count=args.bucket_count,
                    max_open_buckets=args.max_open_buckets,
                    bucket_root=bucket_root,
                    color=color,
                )
            print(
                c("[Shanghai DC]", Color.GREEN, color)
                + f" finished file {archive_index}/{len(archives)} rows={archive_rows:,} "
                + f"file_elapsed={fmt_duration(time.time() - archive_start)} "
                + f"total_elapsed={fmt_duration(time.time() - start_time)}",
                flush=True,
            )
    finally:
        if not args.keep_temp:
            shutil.rmtree(tmp_root, ignore_errors=True)

    print(c("[Shanghai DC]", Color.CYAN, color) + " flushing open taxi supply states", flush=True)
    for state in states.values():
        flush_empty_supply(state, supply, stats)

    print(c("[Shanghai DC]", Color.CYAN, color) + " building graph and edge_speeds", flush=True)
    if polygon_partition is None:
        zone_rows, centers = build_grid_metadata(rows=args.grid_rows, cols=args.grid_cols, bbox=bbox)
        for row in zone_rows:
            row["zone_name"] = f"SH_GRID_R{int(row['grid_row']):02d}_C{int(row['grid_col']):02d}"
            row["borough"] = "Shanghai"
        adj, edge_index, lengths = build_knn_graph(centers, args.knn)
        graph_stats = {
            "mode": "fixed_grid_knn",
            "knn": int(args.knn),
            "directed_edges": int(edge_index.shape[0]),
        }
    else:
        zone_rows = polygon_partition.zone_rows
        centers = polygon_partition.centers
        adj, edge_index, lengths, graph_stats = build_polygon_adjacency(
            polygon_partition.geometries,
            centers,
            touch_tolerance_deg=float(args.polygon_touch_tolerance_deg),
        )
        print(
            c("[Shanghai DC]", Color.CYAN, color)
            + f" polygon graph E={edge_index.shape[0]} isolated={graph_stats['isolated_nodes']}",
            flush=True,
        )
    edge_speeds = fill_edge_speeds(speed_sum, speed_count, edge_index)
    targets_dc = np.stack([demand, supply], axis=-1).astype(np.float32)
    time_features = build_time_features(time_meta, slots_per_day)
    observed_time_mask = build_observed_time_mask(time_meta, missing_archive_dates)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(c("[Shanghai DC]", Color.CYAN, color) + f" writing arrays to {output_dir}", flush=True)
    atomic_save_npy(output_dir / "node_demand.npy", demand)
    atomic_save_npy(output_dir / "node_supply.npy", supply)
    atomic_save_npy(output_dir / "dropoff_events.npy", dropoff)
    atomic_save_npy(output_dir / "targets_dc.npy", targets_dc)
    atomic_save_npy(output_dir / "time_features.npy", time_features)
    atomic_save_npy(output_dir / "observed_time_mask.npy", observed_time_mask)
    atomic_save_npy(output_dir / "adjacency_matrix.npy", adj)
    atomic_save_npy(output_dir / "edge_index.npy", edge_index)
    atomic_save_npy(output_dir / "edge_lengths.npy", lengths)
    atomic_save_npy(output_dir / "edge_lengths_osrm.npy", lengths)
    atomic_save_npy(output_dir / "edge_lengths_km.npy", lengths)
    atomic_save_npy(output_dir / "edge_speeds.npy", edge_speeds)
    write_time_meta(output_dir / "time_meta.csv", time_meta)
    write_zone_info(output_dir / "zone_info.csv", zone_rows)

    validation = validate_outputs(output_dir)
    if polygon_partition is None:
        spatial_partition = {
            "mode": "fixed_lonlat_grid",
            "rows": int(args.grid_rows),
            "cols": int(args.grid_cols),
            "nodes": int(n_nodes),
            "bbox_lon_min_lon_max_lat_min_lat_max": list(bbox),
            "rationale": (
                "Shanghai April 2015 sample has dense near-square core coverage; "
                "20x20 gives 400 nodes with roughly 3km cells and high occupied-cell coverage."
            ),
        }
        schema_version = "shanghai_dc_grid_v1"
    else:
        spatial_partition = {
            "mode": "real_polygon_districts",
            "nodes": int(n_nodes),
            "boundary_geojson": str(args.boundary_geojson),
            "boundary_source": DEFAULT_DISTRICT_GEOJSON_URL,
            "bbox_lon_min_lon_max_lat_min_lat_max": list(bbox),
            "lookup_resolution": int(args.polygon_lookup_resolution),
            "coordinate_note": (
                "DataV/GeoAtlas boundaries are AMap-style public administrative polygons; "
                "raw taxi coordinates are assigned directly after empirical inside-rate checks."
            ),
            "lookup_stats": polygon_partition.stats(),
        }
        schema_version = "shanghai_dc_district_polygon_v1"
    args_jsonable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    manifest = {
        "schema_version": schema_version,
        "dataset_name": "stdr_shanghai_dc",
        "city": "Shanghai",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": {
            "input_dir": str(input_dir),
            "archives": [str(path) for _, path in archives],
            "compressed_bytes": int(stats.compressed_bytes),
            "date_range": [start_date.isoformat(), end_date.isoformat()],
            "strict_folder_date": not args.allow_cross_date,
        },
        "spatial_partition": spatial_partition,
        "graph": graph_stats,
        "time": {
            "slot_minutes": int(args.slot_minutes),
            "time_steps": int(t_steps),
            "observed_time_mask_file": "observed_time_mask.npy",
            "observed_slots": int(observed_time_mask.sum()),
            "unobserved_slots": int((~observed_time_mask).sum()),
            "missing_archive_dates": missing_archive_dates,
        },
        "cleaning": {
            "coordinate_filter": "valid lon/lat plus selected bbox",
            "time_filter": "GPS time must parse and fall inside configured date range",
            "date_filter": "GPS date must match folder date unless --allow-cross-date is set",
            "status_mapping": "empty_flag 1 -> empty/supply, empty_flag 0 -> occupied",
            "demand_rule": "empty->occupied transition within max_transition_gap_minutes",
            "supply_rule": "one final empty observation per taxi per slot",
            "speed_filter": f"0 <= speed <= {args.max_speed_kmh} km/h",
        },
        "arrays": {
            "node_demand": "node_demand.npy",
            "node_supply": "node_supply.npy",
            "dropoff_events": "dropoff_events.npy",
            "targets_dc": "targets_dc.npy",
            "time_features": "time_features.npy",
            "observed_time_mask": "observed_time_mask.npy",
            "adjacency_matrix": "adjacency_matrix.npy",
            "edge_index": "edge_index.npy",
            "edge_lengths": "edge_lengths.npy",
            "edge_lengths_osrm": "edge_lengths_osrm.npy",
            "edge_lengths_km": "edge_lengths_km.npy",
            "edge_speeds": "edge_speeds.npy",
            "time_meta": "time_meta.csv",
            "zone_info": "zone_info.csv",
        },
        "shapes": {
            "node_demand": list(demand.shape),
            "node_supply": list(supply.shape),
            "dropoff_events": list(dropoff.shape),
            "targets_dc": list(targets_dc.shape),
            "time_features": list(time_features.shape),
            "edge_speeds": list(edge_speeds.shape),
            "adjacency_matrix": list(adj.shape),
            "edge_index": list(edge_index.shape),
            "edge_lengths": list(lengths.shape),
            "observed_time_mask": list(observed_time_mask.shape),
        },
        "stats": stats.to_jsonable(),
        "validation": validation,
        "args": args_jsonable
        | {"input_dir": str(input_dir), "output_dir": str(output_dir), "benchmark_output_dir": str(benchmark_dir)},
    }
    atomic_write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    if not args.no_benchmark:
        print(c("[Shanghai DC]", Color.CYAN, color) + f" exporting benchmark to {benchmark_dir}", flush=True)
        export_benchmark(output_dir, benchmark_dir, args.hist_len, args.pred_horizon)

    elapsed_total = time.time() - start_time
    print("=" * 96, flush=True)
    print(
        c("[Shanghai DC done]", Color.GREEN, color)
        + f" elapsed={fmt_duration(elapsed_total)} output={output_dir} "
        + f"T={validation['T']} N={validation['N']} E={validation['E']} "
        + f"D_sum={validation['demand_sum']:.0f} C_sum={validation['supply_sum']:.0f} "
        + f"taxis={len(stats.unique_taxis or []):,}",
        flush=True,
    )
    if not args.no_benchmark:
        print(c("[benchmark]", Color.GREEN, color) + f" output={benchmark_dir}", flush=True)
    print("=" * 96, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[Shanghai DC] interrupted", flush=True)
        raise
