"""评测数据集文件和稳定子集的输出。"""

import json
import random
from collections import defaultdict
from pathlib import Path


def save_dataset(records: list[dict], output_path: Path) -> None:
    """将记录保存为 JSONL。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    distribution: dict[str, int] = defaultdict(int)
    for record in records:
        distribution[record["industry"]] += 1
    print(f"\n[generator] 数据集已保存：{output_path}  共 {len(records)} 条")
    print("  行业分布：")
    for industry, count in sorted(distribution.items(), key=lambda item: -item[1]):
        print(f"    {industry:12s} {count:>4} 条")


def balanced_subset(records: list[dict], size: int, *, seed: int) -> list[dict]:
    """按可回答性、行业、类别和推理范围轮转抽取稳定子集。"""
    if size <= 0 or size >= len(records):
        return list(records)

    rng = random.Random(seed)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (
            bool(record.get("answerable", True)),
            str(record.get("industry") or "未知"),
            str(record.get("category") or "unknown"),
            str(record.get("reasoning_scope") or "unknown"),
        )
        buckets[key].append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    keys = list(buckets)
    rng.shuffle(keys)
    selected: list[dict] = []
    while keys and len(selected) < size:
        remaining: list[tuple] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < size:
                selected.append(bucket.pop())
            if bucket:
                remaining.append(key)
        keys = remaining
    return sorted(selected, key=lambda record: record["case_id"])


def save_dataset_bundle(
    records: list[dict],
    output_path: Path,
    *,
    smoke_size: int,
    regression_size: int,
    seed: int,
    overwrite: bool,
) -> dict[str, Path]:
    """导出完整候选集、Smoke、Regression 和清单文件。"""
    suffix = output_path.suffix or ".jsonl"
    stem = output_path.stem
    paths = {
        "full": output_path,
        "smoke": output_path.with_name(f"{stem}.smoke{suffix}"),
        "regression": output_path.with_name(f"{stem}.regression{suffix}"),
        "manifest": output_path.with_name(f"{stem}.manifest.json"),
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"输出已存在：{joined}；确认后使用 --overwrite 覆盖")

    smoke = balanced_subset(records, smoke_size, seed=seed + 1)
    regression = balanced_subset(records, regression_size, seed=seed + 2)
    save_dataset(records, paths["full"])
    save_dataset(smoke, paths["smoke"])
    save_dataset(regression, paths["regression"])

    manifest = {
        "schema_version": 1,
        "dataset_schema_version": 3,
        "seed": seed,
        "counts": {
            "full": len(records),
            "smoke": len(smoke),
            "regression": len(regression),
            "answerable": sum(record.get("answerable") is not False for record in records),
            "no_answer": sum(record.get("answerable") is False for record in records),
            "multi_chunk": sum(
                record.get("reasoning_scope") == "multi_chunk" for record in records
            ),
            "pending_review": sum(
                record.get("review_status") == "pending" for record in records
            ),
        },
        "files": {
            key: str(path) for key, path in paths.items() if key != "manifest"
        },
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    with paths["manifest"].open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    print(f"[generator] 数据清单：{paths['manifest']}")
    return paths
