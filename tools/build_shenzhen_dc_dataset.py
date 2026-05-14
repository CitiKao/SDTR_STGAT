#!/usr/bin/env python
"""Build a Shenzhen taxi D/C dataset compatible with the STDR loaders.

The converter streams the raw Shenzhen ``TRK*.tar.bz2`` archives without
extracting them to disk.  It produces the same core files as the existing NYC
training directory:

  - node_demand.npy
  - node_supply.npy
  - edge_speeds.npy
  - adjacency_matrix.npy
  - edge_index.npy
  - edge_lengths.npy / edge_lengths_osrm.npy
  - time_meta.csv

For Shenzhen, demand is inferred from occupied-state transitions ``0 -> 1``.
Supply is the number of unique empty taxis per zone and slot, counted once from
the taxi's last observed empty position in the slot.  Dropoffs ``1 -> 0`` are
also saved as ``node_dropoff.npy`` for auditability, but are not used as the
primary C/supply target.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import BinaryIO

import numpy as np


DATE_RE = re.compile(r"(20\d{6})")
DEFAULT_BBOX = (113.75, 114.65, 22.43, 22.86)  # lon_min, lon_max, lat_min, lat_max
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class Color:
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    ORANGE = "\033[38;5;208m"
    CYAN = "\033[36m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def c(text: str, color: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color}{text}{Color.RESET}"


def pct(num: int | float, den: int | float) -> float:
    if den == 0:
        return 0.0
    return 100.0 * float(num) / float(den)


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_datetime_from_now(seconds_from_now: float) -> str:
    return datetime.now().astimezone() + timedelta(seconds=max(0.0, seconds_from_now))


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def parse_archive_date(path: Path) -> date | None:
    match = DATE_RE.search(path.name)
    if match is None:
        return None
    raw = match.group(1)
    return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8]))


def parse_name_list(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def normalize_archive_key(raw: str) -> str:
    key = raw.strip()
    match = re.search(r"(\d{4})-?(\d{2})-?(\d{2})", key)
    if match is None:
        raise ValueError(f"Cannot parse archive date from {raw!r}")
    return f"TRK{match.group(1)}{match.group(2)}{match.group(3)}.tar.bz2"


def parse_recovered_line_files(raw_items: list[str]) -> dict[str, Path]:
    recovered: dict[str, Path] = {}
    for raw in raw_items:
        if "=" not in raw:
            raise ValueError(
                "--recovered-line-file must be DATE=PATH, "
                "e.g. 20090924=runs/.../TRK20090924.recovered_lines.tsv"
            )
        key, value = raw.split("=", 1)
        archive_name = normalize_archive_key(key)
        path = Path(value.strip())
        if not path.exists():
            raise ValueError(f"Recovered line file not found for {archive_name}: {path}")
        recovered[archive_name] = path
    return recovered


def missing_dates_between(dates: list[date]) -> list[str]:
    if not dates:
        return []
    present = set(dates)
    current = min(dates)
    end = max(dates)
    missing: list[str] = []
    while current <= end:
        if current not in present:
            missing.append(current.isoformat())
        current += timedelta(days=1)
    return missing


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


def parse_float(raw: bytes) -> float | None:
    try:
        return float(raw.strip())
    except Exception:
        return None


def two_digits(raw: bytes, start: int) -> int | None:
    if start + 1 >= len(raw):
        return None
    a, b = raw[start], raw[start + 1]
    if not (48 <= a <= 57 and 48 <= b <= 57):
        return None
    return (a - 48) * 10 + (b - 48)


def parse_time(
    raw: bytes,
    date_to_offset: dict[str, int],
    slots_per_day: int,
    slot_minutes: int,
) -> tuple[int, int, str] | None:
    value = raw.strip()
    # Expected: YYYY-MM-DD HH:MM:SS
    if len(value) < 16:
        return None
    try:
        date_text = value[:10].decode("ascii")
    except UnicodeDecodeError:
        return None
    day_offset = date_to_offset.get(date_text)
    if day_offset is None:
        return None
    hour = two_digits(value, 11)
    minute = two_digits(value, 14)
    if hour is None or minute is None or hour > 23 or minute > 59:
        return None
    slot = hour * (60 // slot_minutes) + minute // slot_minutes
    if slot < 0 or slot >= slots_per_day:
        return None
    time_idx = day_offset * slots_per_day + slot
    absolute_minute = day_offset * 1440 + hour * 60 + minute
    return time_idx, absolute_minute, date_text


def zone_from_lonlat(
    lon: float,
    lat: float,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    rows: int,
    cols: int,
) -> int | None:
    if lon < lon_min or lon > lon_max or lat < lat_min or lat > lat_max:
        return None
    col = int((lon - lon_min) / (lon_max - lon_min) * cols)
    row = int((lat - lat_min) / (lat_max - lat_min) * rows)
    col = min(max(col, 0), cols - 1)
    row = min(max(row, 0), rows - 1)
    return row * cols + col


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    r = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


@dataclass
class TaxiState:
    prev_status: int | None = None
    prev_zone: int | None = None
    prev_time_idx: int | None = None
    prev_abs_minute: int | None = None
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
    demand_events: int = 0
    dropoff_events: int = 0
    supply_observations: int = 0
    unique_taxis: set[bytes] | None = None

    def __post_init__(self) -> None:
        if self.unique_taxis is None:
            self.unique_taxis = set()


def build_time_meta(start: date, end: date, slot_minutes: int) -> tuple[list[dict[str, object]], dict[str, int]]:
    slots_per_day = 1440 // slot_minutes
    rows: list[dict[str, object]] = []
    date_to_offset: dict[str, int] = {}
    current = start
    offset = 0
    time_idx = 0
    while current <= end:
        date_text = current.isoformat()
        date_to_offset[date_text] = offset
        weekday = current.weekday()
        for slot in range(slots_per_day):
            minutes = slot * slot_minutes
            rows.append(
                {
                    "time_idx": time_idx,
                    "date": date_text,
                    "slot": slot,
                    "day_of_week": weekday,
                    "day_name": DAY_NAMES[weekday],
                    "hour": minutes // 60,
                    "minute": minutes % 60,
                }
            )
            time_idx += 1
        current += timedelta(days=1)
        offset += 1
    return rows, date_to_offset


def build_time_features(time_meta: list[dict[str, object]], slots_per_day: int) -> np.ndarray:
    features = np.zeros((len(time_meta), 6), dtype=np.float32)
    for i, row in enumerate(time_meta):
        month = int(str(row["date"])[5:7])
        weekday = float(row["day_of_week"])
        slot = float(row["slot"])
        month_angle = 2.0 * math.pi * (month - 1.0) / 12.0
        weekday_angle = 2.0 * math.pi * weekday / 7.0
        slot_angle = 2.0 * math.pi * slot / float(slots_per_day)
        features[i] = [
            math.sin(month_angle),
            math.cos(month_angle),
            math.sin(weekday_angle),
            math.cos(weekday_angle),
            math.sin(slot_angle),
            math.cos(slot_angle),
        ]
    return features


def build_observed_time_mask(time_meta: list[dict[str, object]], missing_dates: list[str]) -> np.ndarray:
    missing = set(missing_dates)
    return np.array([str(row["date"]) not in missing for row in time_meta], dtype=bool)


def build_grid_metadata(
    *,
    rows: int,
    cols: int,
    bbox: tuple[float, float, float, float],
) -> tuple[list[dict[str, object]], np.ndarray]:
    lon_min, lon_max, lat_min, lat_max = bbox
    zone_rows: list[dict[str, object]] = []
    centers = np.zeros((rows * cols, 2), dtype=np.float64)
    for row in range(rows):
        z_lat_min = lat_min + (lat_max - lat_min) * row / rows
        z_lat_max = lat_min + (lat_max - lat_min) * (row + 1) / rows
        for col in range(cols):
            idx = row * cols + col
            z_lon_min = lon_min + (lon_max - lon_min) * col / cols
            z_lon_max = lon_min + (lon_max - lon_min) * (col + 1) / cols
            lon_center = (z_lon_min + z_lon_max) / 2.0
            lat_center = (z_lat_min + z_lat_max) / 2.0
            centers[idx] = [lon_center, lat_center]
            zone_rows.append(
                {
                    "index": idx,
                    "zone_name": f"SZ_GRID_R{row:02d}_C{col:02d}",
                    "borough": "Shenzhen",
                    "grid_row": row,
                    "grid_col": col,
                    "lon_min": z_lon_min,
                    "lon_max": z_lon_max,
                    "lat_min": z_lat_min,
                    "lat_max": z_lat_max,
                    "lon_center": lon_center,
                    "lat_center": lat_center,
                }
            )
    return zone_rows, centers


def build_knn_graph(centers: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = centers.shape[0]
    lengths = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        lon_i, lat_i = centers[i]
        for j in range(n):
            if i == j:
                continue
            lon_j, lat_j = centers[j]
            lengths[i, j] = haversine_km(float(lon_i), float(lat_i), float(lon_j), float(lat_j))

    edges: list[tuple[int, int]] = []
    for i in range(n):
        order = np.argsort(lengths[i])
        neighbors = [int(j) for j in order if int(j) != i][:k]
        edges.extend((i, j) for j in neighbors)

    edge_index = np.array(edges, dtype=np.int32)
    adj = np.zeros((n, n), dtype=np.float32)
    if edge_index.size:
        adj[edge_index[:, 0], edge_index[:, 1]] = 1.0
    return adj, edge_index, lengths.astype(np.float32)


def write_time_meta(path: Path, time_meta: list[dict[str, object]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["time_idx", "date", "slot", "day_of_week", "day_name", "hour", "minute"],
        )
        writer.writeheader()
        writer.writerows(time_meta)
    os.replace(tmp, path)


def write_zone_info(path: Path, zone_rows: list[dict[str, object]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(zone_rows[0].keys()))
        writer.writeheader()
        writer.writerows(zone_rows)
    os.replace(tmp, path)


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
    observation: tuple[int, int, int, int, float | None],
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
    time_idx, absolute_minute, status, zone, speed = observation
    if speed is not None:
        speed_sum[time_idx, zone] += float(speed)
        speed_count[time_idx, zone] += 1
        stats.speed_rows += 1

    state = states.get(taxi_id)
    if state is None:
        state = TaxiState()
        states[taxi_id] = state

    if state.prev_abs_minute is not None and absolute_minute < state.prev_abs_minute:
        # Do not let an older observation corrupt the forward state machine.
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
    state.prev_zone = zone
    state.prev_time_idx = time_idx
    state.prev_abs_minute = absolute_minute


def process_taxi_group(
    taxi_id: bytes,
    observations: list[tuple[int, int, int, int, float | None]],
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
    observations.sort(key=lambda item: item[1])
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
    completed_compressed: int,
    current_archive_size: int,
    color: bool,
) -> str:
    elapsed = time.time() - start_time
    completed_for_eta = completed_compressed
    if stats.records > 0 and archive_rows > 0:
        # The first archive has no completed-byte baseline yet.  A small
        # Shenzhen-specific prior keeps ETA useful from the first progress line;
        # after one archive, actual rows/byte takes over.
        rows_per_byte = (
            stats.records / max(completed_compressed, 1)
            if completed_compressed > 0
            else 0.145
        )
        estimated_archive_rows = max(current_archive_size * rows_per_byte, 1.0)
        completed_for_eta += int(current_archive_size * min(0.99, archive_rows / estimated_archive_rows))
    elif archive_index > 1:
        completed_for_eta += int(current_archive_size * 0.25)
    progress = min(max(completed_for_eta / max(stats.compressed_bytes, 1), 0.0), 0.999)
    eta_sec = elapsed * (1.0 - progress) / progress if progress > 0 else 0.0
    eta_at = fmt_datetime_from_now(eta_sec).strftime("%H:%M:%S")
    label = c("[深圳DC转换]", Color.CYAN, color)
    shard = c(f"{archive_index}/{archive_count}", Color.YELLOW, color)
    return (
        f"{label} 分片={shard} 文件={archive_name} "
        f"本文件记录={archive_rows:,} 累计记录={stats.records:,} "
        f"车辆={len(stats.unique_taxis or []):,} "
        f"D={stats.demand_events:,} Dropoff={stats.dropoff_events:,} C={stats.supply_observations:,} "
        f"坐标命中={pct(stats.bbox_hits, max(stats.valid_coord, 1)):.3f}% "
        f"状态坏值={pct(stats.bad_status, max(stats.records, 1)):.3f}% "
        f"进度≈{progress*100:.2f}% 已用={fmt_duration(elapsed)} "
        f"预计剩余={fmt_duration(eta_sec)} 预计完成={eta_at}"
    )


def process_stream(
    stream: BinaryIO,
    *,
    archive_name: str,
    archive_index: int,
    archive_count: int,
    archive_size: int,
    completed_compressed: int,
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
            if len(parts) < 7:
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

            parsed_time = parse_time(parts[1], date_to_offset, slots_per_day, slot_minutes)
            if parsed_time is None:
                stats.bad_time += 1
                continue
            time_idx, absolute_minute, _date_text = parsed_time

            status_raw = parts[6].strip()
            if status_raw == b"0":
                status = 0
            elif status_raw == b"1":
                status = 1
            else:
                stats.bad_status += 1
                continue

            lon = parse_float(parts[2])
            lat = parse_float(parts[3])
            if lon is None or lat is None or not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
                continue
            stats.valid_coord += 1
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

            speed = parse_float(parts[4])
            speed_text = b""
            if speed is not None and 0.0 <= speed <= 160.0:
                speed_text = f"{speed:.3f}".encode("ascii")
            bucket_idx = zlib.crc32(taxi_id) % bucket_count
            bucket_files[bucket_idx].write(
                b"%b\t%d\t%d\t%d\t%d\t%b\t%d\n"
                % (taxi_id, absolute_minute, time_idx, status, zone, speed_text, sequence)
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
                        completed_compressed=completed_compressed,
                        current_archive_size=archive_size,
                        color=color,
                    )
                    + " 阶段=解析入桶",
                    flush=True,
                )
                next_progress += progress_rows
    finally:
        for fh in bucket_files:
            fh.close()

    bucket_progress_step = max(bucket_count // 8, 1)
    for bucket_idx, bucket_path in enumerate(bucket_paths, start=1):
        records: list[tuple[bytes, int, int, int, int, int, float | None]] = []
        if bucket_path.exists() and bucket_path.stat().st_size > 0:
            with bucket_path.open("rb") as fh:
                for raw in fh:
                    parts = raw.rstrip(b"\n").split(b"\t")
                    if len(parts) != 7:
                        continue
                    speed_value = None if not parts[5] else float(parts[5])
                    records.append(
                        (
                            parts[0],
                            int(parts[1]),
                            int(parts[6]),
                            int(parts[2]),
                            int(parts[3]),
                            int(parts[4]),
                            speed_value,
                        )
                    )
        records.sort(key=lambda item: (item[0], item[1], item[2]))
        current_taxi: bytes | None = None
        current_observations: list[tuple[int, int, int, int, float | None]] = []
        for taxi_id, absolute_minute, _sequence, time_idx, status, zone, speed_value in records:
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
            current_observations.append((time_idx, absolute_minute, status, zone, speed_value))
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
                    completed_compressed=completed_compressed,
                    current_archive_size=archive_size,
                    color=color,
                )
                + f" 阶段=排序重放 bucket={bucket_idx}/{bucket_count}",
                flush=True,
            )
    shutil.rmtree(bucket_root, ignore_errors=True)
    return archive_rows


def fill_edge_speeds(
    speed_sum: np.ndarray,
    speed_count: np.ndarray,
    edge_index: np.ndarray,
) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        node_speed = speed_sum / np.maximum(speed_count, 1)
    node_speed = node_speed.astype(np.float32)

    valid = speed_count > 0
    global_mean = float(speed_sum.sum() / max(speed_count.sum(), 1))
    if not math.isfinite(global_mean) or global_mean <= 0:
        global_mean = 30.0

    time_sum = speed_sum.sum(axis=1)
    time_count = speed_count.sum(axis=1)
    time_mean = (time_sum / np.maximum(time_count, 1)).astype(np.float32)
    time_mean[time_count == 0] = global_mean
    node_speed[~valid] = np.broadcast_to(time_mean[:, None], node_speed.shape)[~valid]

    if edge_index.size == 0:
        return np.zeros((node_speed.shape[0], 0), dtype=np.float32)
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    edge_speeds = 0.5 * (node_speed[:, src] + node_speed[:, dst])
    return np.clip(edge_speeds, 1.0, 160.0).astype(np.float32)


def export_benchmark(output_dir: Path, benchmark_dir: Path, hist_len: int, pred_horizon: int) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dc_benchmark.dataset import export_dc_benchmark

    export_dc_benchmark(
        source_data_dir=output_dir,
        output_dir=benchmark_dir,
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        report_horizons_minutes="15,30,60",
        split_policy="project_monthly",
        time_feature_mode="baseline",
        force=True,
    )


def validate_outputs(output_dir: Path) -> dict[str, object]:
    demand = np.load(output_dir / "node_demand.npy", mmap_mode="r")
    supply = np.load(output_dir / "node_supply.npy", mmap_mode="r")
    edge_speeds = np.load(output_dir / "edge_speeds.npy", mmap_mode="r")
    adj = np.load(output_dir / "adjacency_matrix.npy", mmap_mode="r")
    edge_index = np.load(output_dir / "edge_index.npy", mmap_mode="r")
    if demand.shape != supply.shape:
        raise ValueError(f"demand/supply shape mismatch: {demand.shape} vs {supply.shape}")
    if adj.shape[0] != adj.shape[1] or adj.shape[0] != demand.shape[1]:
        raise ValueError(f"adj shape {adj.shape} does not match node count {demand.shape[1]}")
    if edge_index.ndim != 2 or edge_index.shape[1] != 2:
        raise ValueError(f"edge_index must have shape (E, 2), got {edge_index.shape}")
    if edge_speeds.shape != (demand.shape[0], edge_index.shape[0]):
        raise ValueError(
            f"edge_speeds shape {edge_speeds.shape} expected {(demand.shape[0], edge_index.shape[0])}"
        )
    if not np.isfinite(demand).all() or not np.isfinite(supply).all() or not np.isfinite(edge_speeds).all():
        raise ValueError("Found non-finite values in output arrays")
    if float(demand.min()) < 0 or float(supply.min()) < 0:
        raise ValueError("Demand/supply must be non-negative")
    rows = edge_index[:, 0]
    cols = edge_index[:, 1]
    if rows.min() < 0 or cols.min() < 0 or rows.max() >= adj.shape[0] or cols.max() >= adj.shape[0]:
        raise ValueError("edge_index contains node ids outside adjacency bounds")
    if not np.all(adj[rows, cols] > 0):
        raise ValueError("edge_index contains edges not present in adjacency_matrix")
    undirected = (np.asarray(adj) > 0) | (np.asarray(adj).T > 0)
    n_nodes = int(undirected.shape[0])
    seen = np.zeros(n_nodes, dtype=bool)
    component_sizes: list[int] = []
    for start in range(n_nodes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            neighbors = np.flatnonzero(undirected[node])
            for nb in neighbors:
                nb_int = int(nb)
                if not seen[nb_int]:
                    seen[nb_int] = True
                    stack.append(nb_int)
        component_sizes.append(size)
    largest_component = max(component_sizes) if component_sizes else 0
    degree = undirected.sum(axis=1)
    return {
        "T": int(demand.shape[0]),
        "N": int(demand.shape[1]),
        "E": int(edge_index.shape[0]),
        "demand_sum": float(demand.sum(dtype=np.float64)),
        "supply_sum": float(supply.sum(dtype=np.float64)),
        "edge_speed_mean": float(edge_speeds.mean()),
        "graph_largest_weak_component_nodes": int(largest_component),
        "graph_largest_weak_component_ratio": float(largest_component / max(n_nodes, 1)),
        "graph_isolated_nodes": int(np.sum(degree == 0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Shenzhen taxi D/C dataset for STDR.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/external_datasets/raw/dc_candidate_downloads/sz_taxi_data"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/shenzhen_dc"))
    parser.add_argument("--benchmark-output-dir", type=Path, default=Path("data/shenzhen_dc_benchmark"))
    parser.add_argument("--slot-minutes", type=int, default=15)
    parser.add_argument("--grid-rows", type=int, default=12)
    parser.add_argument("--grid-cols", type=int, default=23)
    parser.add_argument("--knn", type=int, default=5)
    parser.add_argument("--bbox", default="default", help="default or lon_min,lon_max,lat_min,lat_max")
    parser.add_argument("--max-transition-gap-minutes", type=int, default=180)
    parser.add_argument("--progress-rows", type=int, default=5_000_000)
    parser.add_argument("--bucket-count", type=int, default=128)
    parser.add_argument("--max-archives", type=int, default=0)
    parser.add_argument(
        "--exclude-archives",
        default="",
        help="Comma-separated archive file names to exclude, e.g. TRK20090924.tar.bz2",
    )
    parser.add_argument(
        "--recovered-line-file",
        action="append",
        default=[],
        help=(
            "Use a recovered raw line file instead of tar extraction for a date. "
            "Format: DATE=PATH, e.g. 20090924=runs/.../TRK20090924.recovered_lines.tsv. "
            "Can be repeated."
        ),
    )
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
    archives = sorted(args.input_dir.glob("TRK*.tar.bz2"))
    try:
        recovered_line_files = parse_recovered_line_files(args.recovered_line_file)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    excluded_archives = set(parse_name_list(args.exclude_archives))
    if excluded_archives:
        known_names = {path.name for path in archives}
        unknown = sorted(excluded_archives - known_names)
        if unknown:
            raise SystemExit(f"--exclude-archives contains unknown archive names: {unknown}")
        archives = [path for path in archives if path.name not in excluded_archives]
    if args.max_archives > 0:
        archives = archives[: args.max_archives]
    if not archives:
        raise SystemExit(f"No TRK*.tar.bz2 archives found in {args.input_dir}")
    known_names = {path.name for path in archives}
    unknown_recovered = sorted(set(recovered_line_files) - known_names)
    if unknown_recovered:
        raise SystemExit(f"--recovered-line-file contains unknown archive dates: {unknown_recovered}")

    archive_dates_raw = [parse_archive_date(path) for path in archives]
    if any(d is None for d in archive_dates_raw):
        raise SystemExit("Every archive name must contain a YYYYMMDD date")
    archive_dates = [d for d in archive_dates_raw if d is not None]
    missing_archive_dates = missing_dates_between(archive_dates)
    start_date = min(archive_dates)
    end_date = max(archive_dates)
    assert start_date is not None and end_date is not None

    slots_per_day = 1440 // args.slot_minutes
    time_meta, date_to_offset = build_time_meta(start_date, end_date, args.slot_minutes)
    t_steps = len(time_meta)
    n_nodes = args.grid_rows * args.grid_cols
    stats = ConvertStats(
        archives=len(archives),
        compressed_bytes=sum(path.stat().st_size for path in archives),
    )

    print(
        c("[深圳DC转换]", Color.CYAN, color)
        + f" 输入={args.input_dir} 输出={args.output_dir} "
        + c(f"节点={n_nodes} ({args.grid_rows}x{args.grid_cols})", Color.YELLOW, color)
        + f" 时间={start_date}..{end_date} T={t_steps} slot={args.slot_minutes}min "
        + f"压缩包={len(archives)} 压缩大小={stats.compressed_bytes / 1024**3:.2f}GB",
        flush=True,
    )
    print(
        c("[分区依据]", Color.ORANGE, color)
        + " 深圳论文常见 1km grid / 102 grids(Futian)、198 regions(Futian)、20x20 grid；"
        + "本次为对齐 NYC 263 zones，采用全深圳近方形 12x23=276 grid。",
        flush=True,
    )
    if missing_archive_dates:
        print(
            c("[时间轴提醒]", Color.YELLOW, color)
            + f" 原始压缩包缺少日期={missing_archive_dates}；输出 time_meta 保持连续时间轴，缺日 slot 标为未观测，训练/benchmark 会剔除相关 window。",
            flush=True,
        )
    if excluded_archives:
        print(
            c("[数据质量提醒]", Color.RED, color)
            + f" 已排除压缩包={sorted(excluded_archives)}；对应日期会在连续 time_meta 中保留为未观测 slot。",
            flush=True,
        )
    if recovered_line_files:
        recovered_display = {name: str(path) for name, path in sorted(recovered_line_files.items())}
        print(
            c("[恢复数据]", Color.ORANGE, color)
            + f" 以下日期使用 bzip2recover 后的有效行替代损坏 tar 解包：{recovered_display}",
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
    completed_compressed = 0
    archive_summaries: list[dict[str, object]] = []
    for archive_index, archive in enumerate(archives, start=1):
        shard_start = time.time()
        print(
            c("[深圳DC转换]", Color.CYAN, color)
            + f" 开始分片={c(f'{archive_index}/{len(archives)}', Color.YELLOW, color)} 文件={archive.name}",
            flush=True,
        )
        rows_before = stats.records
        recovered_path = recovered_line_files.get(archive.name)
        if recovered_path is not None:
            print(
                c("[恢复数据]", Color.ORANGE, color)
                + f" {archive.name} 使用已恢复有效行文件={recovered_path}",
                flush=True,
            )
            with recovered_path.open("rb") as stream:
                archive_rows = process_stream(
                    stream,
                    archive_name=archive.name,
                    archive_index=archive_index,
                    archive_count=len(archives),
                    archive_size=archive.stat().st_size,
                    completed_compressed=completed_compressed,
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
                    progress_rows=args.progress_rows,
                    bucket_count=args.bucket_count,
                    bucket_root=bucket_base / archive.stem,
                    color=color,
                )
        else:
            process = subprocess.Popen(
                ["tar", "-xOf", str(archive)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert process.stdout is not None
            archive_rows = process_stream(
                process.stdout,
                archive_name=archive.name,
                archive_index=archive_index,
                archive_count=len(archives),
                archive_size=archive.stat().st_size,
                completed_compressed=completed_compressed,
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
                progress_rows=args.progress_rows,
                bucket_count=args.bucket_count,
                bucket_root=bucket_base / archive.stem,
                color=color,
            )
            _, stderr = process.communicate()
            if process.returncode != 0:
                message = stderr.decode("utf-8", errors="replace")[:1000]
                raise RuntimeError(f"tar failed for {archive}: {message}")
        completed_compressed += archive.stat().st_size
        elapsed = time.time() - shard_start
        archive_summaries.append(
            {
                "archive": archive.name,
                "source": "recovered_lines" if recovered_path is not None else "tar",
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
                completed_compressed=completed_compressed,
                current_archive_size=archive.stat().st_size,
                color=color,
            ),
            flush=True,
        )
        print(
            c("[深圳DC转换]", Color.GREEN, color)
            + f" 完成 {archive.name} 用时={fmt_duration(elapsed)} "
            + f"累计D={stats.demand_events:,} Dropoff={stats.dropoff_events:,} C={stats.supply_observations:,}",
            flush=True,
        )

    shutil.rmtree(bucket_base, ignore_errors=True)

    print(c("[深圳DC转换]", Color.CYAN, color) + " 正在刷新每辆车最后一个空车时间槽的 C/supply 计数", flush=True)
    for state in states.values():
        flush_empty_supply(state, supply, stats)

    print(c("[深圳DC转换]", Color.CYAN, color) + " 正在建立 KNN 图结构与 edge_speeds", flush=True)
    zone_rows, centers = build_grid_metadata(rows=args.grid_rows, cols=args.grid_cols, bbox=bbox)
    adj, edge_index, edge_lengths = build_knn_graph(centers, args.knn)
    edge_speeds = fill_edge_speeds(speed_sum, speed_count, edge_index)
    targets_dc = np.stack([demand, supply], axis=-1).astype(np.float32)
    time_features = build_time_features(time_meta, slots_per_day)
    observed_time_mask = build_observed_time_mask(time_meta, missing_archive_dates)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(c("[深圳DC转换]", Color.CYAN, color) + f" 正在写入 {output_dir}", flush=True)
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
        "schema_version": "shenzhen_dc_grid_v1",
        "dataset_name": "stdr_shenzhen_dc",
        "source_city": "Shenzhen",
        "source_input_dir": str(args.input_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "spatial_partition": {
            "mode": "fixed_lonlat_grid",
            "rows": int(args.grid_rows),
            "cols": int(args.grid_cols),
            "nodes": int(n_nodes),
            "bbox_lon_min_lon_max_lat_min_lat_max": list(bbox),
            "rationale": (
                "NYC current DC data uses 263 taxi zones. Shenzhen literature often uses "
                "1km grids/102 Futian grids, 198 Futian regions, or 20x20 grids. "
                "This full-city 12x23 grid gives 276 nodes, close to 263 while keeping "
                "cells roughly square under the selected Shenzhen bounding box."
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
            "excluded_archives": sorted(excluded_archives),
            "recovered_line_files": {
                name: str(path) for name, path in sorted(recovered_line_files.items())
            },
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
            "edge_speeds": "GPS point speeds aggregated by zone/time, then averaged across each KNN edge endpoints",
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
        "created_from_files": {archive.name: sha256_path(archive) for archive in archives},
    }
    atomic_write_text(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    if not args.no_benchmark:
        print(c("[深圳DC转换]", Color.CYAN, color) + f" 正在导出 benchmark 目录 {args.benchmark_output_dir}", flush=True)
        export_benchmark(output_dir, args.benchmark_output_dir, args.hist_len, args.pred_horizon)

    elapsed_total = time.time() - start_time
    print("=" * 96, flush=True)
    print(
        c("[深圳DC转换完成]", Color.GREEN, color)
        + f" 用时={fmt_duration(elapsed_total)} 输出={output_dir} "
        + f"T={validation['T']} N={validation['N']} E={validation['E']} "
        + f"D总量={validation['demand_sum']:.0f} C总量={validation['supply_sum']:.0f} "
        + f"车辆={len(stats.unique_taxis or []):,}",
        flush=True,
    )
    if not args.no_benchmark:
        print(c("[benchmark]", Color.GREEN, color) + f" 已写入 {args.benchmark_output_dir}", flush=True)
    print("=" * 96, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[深圳DC转换] 已由用户中断", flush=True)
        raise
