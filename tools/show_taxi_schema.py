#!/usr/bin/env python
"""Show raw taxi dataset field meanings using docs plus real sample rows."""

from __future__ import annotations

import gzip
import json
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path("data/external_datasets/raw/dc_candidate_downloads")


@dataclass(frozen=True)
class FieldSpec:
    index: int
    name: str
    meaning: str
    encoding: str = ""


SCHEMAS: dict[str, list[FieldSpec]] = {
    "sz": [
        FieldSpec(1, "taxi_id", "计程车/出租车 ID"),
        FieldSpec(2, "time", "GPS 记录时间"),
        FieldSpec(3, "longitude", "经度"),
        FieldSpec(4, "latitude", "纬度"),
        FieldSpec(5, "speed", "速度"),
        FieldSpec(6, "direction", "方向/航向角"),
        FieldSpec(7, "occupied", "载客状态", "1=载客，0=空车"),
        FieldSpec(8, "unknown_reserved", "保留/未说明字段，样本常见为 0"),
    ],
    "sh": [
        FieldSpec(1, "taxi_id", "计程车/出租车 ID"),
        FieldSpec(2, "plate_color_or_company", "说明文件未完全可辨识，样本中常见为 0"),
        FieldSpec(3, "empty", "空车状态", "0=载客，1=空车"),
        FieldSpec(4, "operation_status", "运营状态"),
        FieldSpec(5, "alarm_or_logical_mv", "说明文件记作“逻辑 mv/相关状态”，样本中常见为 0"),
        FieldSpec(6, "brake", "刹车状态，样本中常见为 0"),
        FieldSpec(7, "upload_time", "数据上传/接收时间"),
        FieldSpec(8, "gps_time", "GPS 测定时间"),
        FieldSpec(9, "longitude", "经度"),
        FieldSpec(10, "latitude", "纬度"),
        FieldSpec(11, "speed", "速度"),
        FieldSpec(12, "direction", "方向/航向角"),
        FieldSpec(13, "satellite_count", "卫星数量"),
    ],
    "cd": [
        FieldSpec(1, "taxi_id", "计程车/出租车 ID"),
        FieldSpec(2, "latitude", "纬度"),
        FieldSpec(3, "longitude", "经度"),
        FieldSpec(4, "occupied", "载客状态", "1=载客，0=无客/空车"),
        FieldSpec(5, "time", "时间点"),
    ],
}


def decode_line(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").strip()


def sample_sz() -> list[str]:
    archive = ROOT / "sz_taxi_data" / "TRK20090901.tar.bz2"
    process = subprocess.Popen(
        ["tar", "-xOf", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    rows = [decode_line(process.stdout.readline()) for _ in range(3)]
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)
    return rows


def sample_sh() -> list[str]:
    path = ROOT / "sh_taxi_data" / "1" / "part-00000.gz"
    with gzip.open(path, "rb") as stream:
        return [decode_line(stream.readline()) for _ in range(3)]


def sample_cd() -> list[str]:
    archive = ROOT / "cd_taxi_data" / "20140803_train.zip"
    rows: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        names = [info for info in zf.infolist() if not info.is_dir()]
        with zf.open(names[0]) as stream:
            for _ in range(3):
                rows.append(decode_line(stream.readline()))
    return rows


def split_sample(row: str) -> list[str]:
    return [part.strip() for part in row.split(",")]


def print_schema(city: str, label: str, samples: list[str]) -> dict[str, object]:
    specs = SCHEMAS[city]
    first_parts = split_sample(samples[0]) if samples else []
    print("=" * 96, flush=True)
    print(f"[{label}] 栏位意义检测", flush=True)
    print(f"[{label}] 样本栏位数={len(first_parts)} 预期栏位数={len(specs)}", flush=True)
    print(f"[{label}] 样本原始行：", flush=True)
    for row in samples:
        print(f"  {row}", flush=True)
    print(f"[{label}] 栏位对照：", flush=True)
    rows: list[dict[str, object]] = []
    for spec in specs:
        sample_value = first_parts[spec.index - 1] if spec.index - 1 < len(first_parts) else "<缺失>"
        encoding = f" | 编码：{spec.encoding}" if spec.encoding else ""
        print(
            f"  第{spec.index:02d}栏 | {spec.name} | {spec.meaning}{encoding} | 样本值={sample_value}",
            flush=True,
        )
        rows.append(
            {
                "index": spec.index,
                "name": spec.name,
                "meaning": spec.meaning,
                "encoding": spec.encoding,
                "sample_value": sample_value,
            }
        )
    extra_count = max(0, len(first_parts) - len(specs))
    if extra_count:
        print(f"[{label}] 注意：样本比预期多 {extra_count} 栏，需要后续人工确认。", flush=True)
    elif len(first_parts) < len(specs):
        print(f"[{label}] 注意：样本比预期少 {len(specs) - len(first_parts)} 栏。", flush=True)
    else:
        print(f"[{label}] 栏位数与预期一致。", flush=True)
    return {
        "city": city,
        "label": label,
        "sample_field_count": len(first_parts),
        "expected_field_count": len(specs),
        "samples": samples,
        "fields": rows,
    }


def main() -> None:
    print("开始检测三地计程车资料集栏位意义", flush=True)
    results = [
        print_schema("sz", "深圳 SZ", sample_sz()),
        print_schema("sh", "上海 SH", sample_sh()),
        print_schema("cd", "成都 CD", sample_cd()),
    ]
    output = Path("runs/raw_data_integrity/taxi_schema_detected.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 96, flush=True)
    print(f"栏位意义 JSON 已写入：{output}", flush=True)


if __name__ == "__main__":
    main()
