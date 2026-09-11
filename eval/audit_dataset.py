"""审计 RAG 评测数据集标签完整性，不访问模型或知识库。"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval.retrieval_metrics import gold_chunk_ids, gold_source_pages


def audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    issues: Counter[str] = Counter()
    case_ids: Counter[str] = Counter()
    questions: Counter[str] = Counter()
    chunk_labelled = 0
    source_fallback = 0
    no_answer = 0

    for record in records:
        question = str(record.get("question") or "").strip()
        reference = str(
            record.get("ground_truth") or record.get("reference_answer") or ""
        ).strip()
        case_id = str(record.get("case_id") or "").strip()
        answerable = record.get("answerable") is not False

        if not question:
            issues["missing_question"] += 1
        else:
            questions[question] += 1
        if answerable and not reference:
            issues["missing_reference_answer"] += 1
        if not case_id:
            issues["missing_case_id"] += 1
        else:
            case_ids[case_id] += 1

        chunks = gold_chunk_ids(record)
        sources = gold_source_pages(record)
        if not answerable:
            no_answer += 1
        elif chunks:
            chunk_labelled += 1
        elif sources:
            source_fallback += 1
            issues["source_page_fallback_only"] += 1
        else:
            issues["missing_retrieval_labels"] += 1

    issues["duplicate_case_id"] = sum(count - 1 for count in case_ids.values() if count > 1)
    issues["duplicate_question"] = sum(count - 1 for count in questions.values() if count > 1)
    issues = Counter({key: value for key, value in issues.items() if value})
    return {
        "total_records": len(records),
        "chunk_id_labelled": chunk_labelled,
        "source_page_fallback": source_fallback,
        "no_answer_records": no_answer,
        "issues": dict(sorted(issues.items())),
    }


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    invalid_json = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            if isinstance(value, dict):
                records.append(value)
            else:
                invalid_json += 1
    return records, invalid_json


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 RAG JSONL 评测数据集")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).resolve().parent / "dataset" / "eval_dataset.jsonl"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="存在无 chunk_id、缺失字段或重复项时返回非零退出码",
    )
    args = parser.parse_args()

    records, invalid_json = load_jsonl(Path(args.dataset))
    report = audit_records(records)
    if invalid_json:
        report["issues"]["invalid_json"] = invalid_json
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.strict and report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["audit_records", "load_jsonl"]
