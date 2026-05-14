#!/usr/bin/env python
"""Stream raw taxi archives and report basic integrity statistics.

The script intentionally avoids extracting archives to disk. It prints progress
while scanning so long archive reads are visible in a terminal.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO


DATE_RE = re.compile(r"(20\d{6})")
CITY_BOUNDS = {
    "sz": (113.5, 115.0, 22.0, 23.2),
    "cd": (102.8, 105.2, 29.8, 31.6),
}
CITY_LABELS = {
    "sz": "深圳",
    "cd": "成都",
}


@dataclass
class RunningStats:
    city: str
    archives: int = 0
    inner_files: int = 0
    compressed_bytes: int = 0
    records: int = 0
    bad_columns: int = 0
    empty_lines: int = 0
    valid_coord: int = 0
    city_coord: int = 0
    zero_coord: int = 0
    valid_status: int = 0
    time_nonempty: int = 0
    lon_min: float = math.inf
    lon_max: float = -math.inf
    lat_min: float = math.inf
    lat_max: float = -math.inf
    time_min: str | None = None
    time_max: str | None = None
    ids: set[bytes] = field(default_factory=set)
    status_counts: Counter[str] = field(default_factory=Counter)
    date_rows: Counter[str] = field(default_factory=Counter)
    archive_rows: list[dict[str, object]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def update_time(self, raw: bytes) -> None:
        value = raw.strip().decode("ascii", errors="ignore")
        if not value:
            return
        self.time_nonempty += 1
        if self.time_min is None or value < self.time_min:
            self.time_min = value
        if self.time_max is None or value > self.time_max:
            self.time_max = value

    def update_coord(self, lon: float | None, lat: float | None) -> None:
        if lon == 0 and lat == 0:
            self.zero_coord += 1
        if lon is None or lat is None or not (-180 <= lon <= 180 and -90 <= lat <= 90):
            return
        self.valid_coord += 1
        self.lon_min = min(self.lon_min, lon)
        self.lon_max = max(self.lon_max, lon)
        self.lat_min = min(self.lat_min, lat)
        self.lat_max = max(self.lat_max, lat)
        lon_lo, lon_hi, lat_lo, lat_hi = CITY_BOUNDS[self.city]
        if lon_lo <= lon <= lon_hi and lat_lo <= lat <= lat_hi:
            self.city_coord += 1

    def update_status(self, raw: bytes) -> None:
        value = raw.strip()
        text = value.decode("ascii", errors="ignore")
        self.status_counts[text] += 1
        if value in (b"0", b"1"):
            self.valid_status += 1

    def to_jsonable(self) -> dict[str, object]:
        return {
            "city": self.city,
            "archives": self.archives,
            "inner_files": self.inner_files,
            "compressed_gb": round(self.compressed_bytes / 1024**3, 4),
            "records": self.records,
            "unique_taxi_ids": len(self.ids),
            "bad_column_rows": self.bad_columns,
            "empty_lines": self.empty_lines,
            "valid_coord_rows": self.valid_coord,
            "valid_coord_pct": pct(self.valid_coord, self.records),
            "city_bbox_coord_rows": self.city_coord,
            "city_bbox_coord_pct": pct(self.city_coord, self.records),
            "zero_coord_rows": self.zero_coord,
            "status_valid_rows": self.valid_status,
            "status_valid_pct": pct(self.valid_status, self.records),
            "status_counts": dict(self.status_counts),
            "time_nonempty_rows": self.time_nonempty,
            "time_nonempty_pct": pct(self.time_nonempty, self.records),
            "time_min": self.time_min,
            "time_max": self.time_max,
            "lon_range": None if self.valid_coord == 0 else [self.lon_min, self.lon_max],
            "lat_range": None if self.valid_coord == 0 else [self.lat_min, self.lat_max],
            "date_rows": dict(sorted(self.date_rows.items())),
            "missing_dates_between_archives": missing_dates_from_keys(self.date_rows.keys()),
            "archive_rows": self.archive_rows,
            "errors": self.errors,
        }


def pct(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(100.0 * num / den, 6)


def parse_float(raw: bytes) -> float | None:
    try:
        return float(raw.strip())
    except Exception:
        return None


def date_from_name(name: str) -> str | None:
    match = DATE_RE.search(name)
    if not match:
        return None
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"


def missing_dates_from_keys(keys: object) -> list[str]:
    dates: list[datetime] = []
    for key in keys:
        try:
            dates.append(datetime.strptime(str(key), "%Y-%m-%d"))
        except ValueError:
            pass
    if not dates:
        return []
    dates = sorted(set(dates))
    expected = set()
    current = dates[0]
    while current <= dates[-1]:
        expected.add(current)
        current += timedelta(days=1)
    return [d.strftime("%Y-%m-%d") for d in sorted(expected.difference(dates))]


def progress_line(stats: RunningStats, archive_name: str, file_rows: int, start_time: float) -> str:
    elapsed = time.time() - start_time
    label = CITY_LABELS.get(stats.city, stats.city.upper())
    return (
        f"[{label}] 当前文件={archive_name} "
        f"本文件记录={file_rows:,} 累计记录={stats.records:,} "
        f"车辆数={len(stats.ids):,} "
        f"坐标有效={pct(stats.valid_coord, stats.records)}% "
        f"城市范围坐标={pct(stats.city_coord, stats.records)}% "
        f"载客状态有效={pct(stats.valid_status, stats.records)}% "
        f"已用={elapsed/60:.1f}分钟"
    )


def parse_sz_line(line: bytes, stats: RunningStats) -> None:
    if not line.strip():
        stats.empty_lines += 1
        return
    parts = line.rstrip(b"\r\n").split(b",")
    if len(parts) < 7:
        stats.bad_columns += 1
        return
    stats.records += 1
    taxi_id = parts[0].strip()
    if taxi_id:
        stats.ids.add(taxi_id)
    stats.update_time(parts[1])
    stats.update_coord(parse_float(parts[2]), parse_float(parts[3]))
    stats.update_status(parts[6])


def parse_cd_line(line: bytes, stats: RunningStats) -> None:
    if not line.strip():
        stats.empty_lines += 1
        return
    parts = line.rstrip(b"\r\n").split(b",")
    if len(parts) < 5:
        stats.bad_columns += 1
        return
    stats.records += 1
    taxi_id = parts[0].strip()
    if taxi_id:
        stats.ids.add(taxi_id)
    stats.update_coord(parse_float(parts[2]), parse_float(parts[1]))
    stats.update_status(parts[3])
    stats.update_time(parts[4])


def scan_stream(
    stream: BinaryIO,
    stats: RunningStats,
    archive_name: str,
    parse_line,
    *,
    progress_rows: int,
    start_time: float,
) -> int:
    file_rows_before = stats.records
    next_progress = progress_rows
    for line in stream:
        parse_line(line, stats)
        file_rows = stats.records - file_rows_before
        if progress_rows > 0 and file_rows >= next_progress:
            print(progress_line(stats, archive_name, file_rows, start_time), flush=True)
            next_progress += progress_rows
    return stats.records - file_rows_before


def scan_sz(root: Path, *, progress_rows: int) -> tuple[RunningStats, float]:
    data_dir = root / "sz_taxi_data"
    archives = sorted(data_dir.glob("TRK*.tar.bz2"))
    stats = RunningStats(city="sz")
    stats.archives = len(archives)
    stats.compressed_bytes = sum(p.stat().st_size for p in archives)
    start = time.time()
    print(
        f"[深圳] 压缩包数={len(archives)} 压缩大小={stats.compressed_bytes / 1024**3:.2f}GB",
        flush=True,
    )
    for archive in archives:
        archive_start = time.time()
        rows_before = stats.records
        print(f"[深圳] 开始扫描 {archive.name}", flush=True)
        process = subprocess.Popen(
            ["tar", "-xOf", str(archive)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None
        rows = scan_stream(
            process.stdout,
            stats,
            archive.name,
            parse_sz_line,
            progress_rows=progress_rows,
            start_time=start,
        )
        _, stderr = process.communicate()
        if process.returncode != 0:
            stats.errors.append(
                f"{archive.name}: tar exited {process.returncode}: {stderr.decode(errors='replace')[:500]}"
            )
        stats.inner_files += 1
        date_key = date_from_name(archive.name)
        if date_key:
            stats.date_rows[date_key] += rows
        stats.archive_rows.append(
            {
                "archive": archive.name,
                "date": date_key,
                "rows": rows,
                "elapsed_sec": round(time.time() - archive_start, 3),
            }
        )
        print(progress_line(stats, archive.name, stats.records - rows_before, start), flush=True)
        print(f"[深圳] 完成 {archive.name} 用时={time.time() - archive_start:.1f}秒", flush=True)
    return stats, time.time() - start


def scan_cd(root: Path, *, progress_rows: int) -> tuple[RunningStats, float]:
    data_dir = root / "cd_taxi_data"
    archives = sorted(data_dir.glob("*_train.zip"))
    stats = RunningStats(city="cd")
    stats.archives = len(archives)
    stats.compressed_bytes = sum(p.stat().st_size for p in archives)
    start = time.time()
    print(
        f"[成都] 压缩包数={len(archives)} 压缩大小={stats.compressed_bytes / 1024**3:.2f}GB",
        flush=True,
    )
    for archive in archives:
        archive_start = time.time()
        rows_before = stats.records
        print(f"[成都] 开始扫描 {archive.name}", flush=True)
        try:
            with zipfile.ZipFile(archive) as zf:
                inner_files = [info for info in zf.infolist() if not info.is_dir()]
                stats.inner_files += len(inner_files)
                for info in inner_files:
                    with zf.open(info) as stream:
                        scan_stream(
                            stream,
                            stats,
                            archive.name,
                            parse_cd_line,
                            progress_rows=progress_rows,
                            start_time=start,
                        )
        except Exception as exc:
            stats.errors.append(f"{archive.name}: {type(exc).__name__}: {exc}")
        rows = stats.records - rows_before
        date_key = date_from_name(archive.name)
        if date_key:
            stats.date_rows[date_key] += rows
        stats.archive_rows.append(
            {
                "archive": archive.name,
                "date": date_key,
                "rows": rows,
                "elapsed_sec": round(time.time() - archive_start, 3),
            }
        )
        print(progress_line(stats, archive.name, rows, start), flush=True)
        print(f"[成都] 完成 {archive.name} 用时={time.time() - archive_start:.1f}秒", flush=True)
    return stats, time.time() - start


def print_summary(stats: RunningStats, elapsed: float) -> None:
    summary = stats.to_jsonable()
    label = CITY_LABELS.get(stats.city, stats.city.upper())
    print("=" * 96, flush=True)
    print(
        f"[{label}] 汇总 用时={elapsed/60:.2f}分钟 "
        f"记录数={summary['records']:,} 车辆数={summary['unique_taxi_ids']:,} "
        f"坐标有效={summary['valid_coord_pct']}% 城市范围坐标={summary['city_bbox_coord_pct']}% "
        f"载客状态有效={summary['status_valid_pct']}%",
        flush=True,
    )
    print(
        f"[{label}] 时间范围={summary['time_min']} -> {summary['time_max']} "
        f"经度范围={summary['lon_range']} 纬度范围={summary['lat_range']} "
        f"缺失日期={summary['missing_dates_between_archives']}",
        flush=True,
    )
    print(f"[{label}] 载客状态计数={summary['status_counts']}", flush=True)
    print("=" * 96, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan SZ/CD raw taxi archive integrity.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/external_datasets/raw/dc_candidate_downloads"),
    )
    parser.add_argument("--cities", default="sz,cd", help="Comma-separated subset: sz,cd")
    parser.add_argument("--progress-rows", type=int, default=5_000_000)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    cities = [c.strip().lower() for c in args.cities.split(",") if c.strip()]
    outputs: dict[str, object] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "cities": cities,
        "results": {},
    }
    overall_start = time.time()
    for city in cities:
        if city == "sz":
            stats, elapsed = scan_sz(root, progress_rows=args.progress_rows)
        elif city == "cd":
            stats, elapsed = scan_cd(root, progress_rows=args.progress_rows)
        else:
            raise SystemExit(f"Unsupported city: {city}")
        print_summary(stats, elapsed)
        outputs["results"][city] = {
            "elapsed_sec": round(elapsed, 3),
            **stats.to_jsonable(),
        }
    outputs["finished_at"] = datetime.now().isoformat(timespec="seconds")
    outputs["elapsed_sec"] = round(time.time() - overall_start, 3)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[报告] 已写入 {args.output_json}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[扫描] 已由用户中断", flush=True)
        raise
