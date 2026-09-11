from __future__ import annotations

import math
import sqlite3

import pytest

from eval.audit_dataset import audit_records
from eval import enrich_dataset
from eval.enrich_dataset import enrich_records, load_chunks
from eval.run_eval import write_csv_detail, write_json_summary
from eval.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_retrieval,
)
from react_agent.rag.contracts import StoredChunk


def test_chunk_id_metrics_use_ranked_gold_labels() -> None:
    record = {
        "question": "测试问题",
        "ground_truth": "测试答案",
        "gold_chunk_ids": ["gold-1", "gold-2"],
        "gold_sources": [{"source_file": "reports/a.pdf", "pages": [3]}],
    }
    retrieved = [
        {"chunk_id": "gold-1", "source_file": "reports/a.pdf", "source_page": 3},
        {"chunk_id": "noise", "source_file": "reports/b.pdf", "source_page": 1},
        {"chunk_id": "gold-2", "source_file": "reports/a.pdf", "source_page": 4},
    ]

    result = evaluate_retrieval(record, retrieved, top_k=3)

    assert result["retrieval_label_mode"] == "chunk_id"
    assert result["hit_at_k"] == 1.0
    assert result["precision_at_k"] == pytest.approx(2 / 3)
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] == 1.0
    expected_dcg = 1 + 1 / math.log2(4)
    ideal_dcg = 1 + 1 / math.log2(3)
    assert result["ndcg_at_k"] == pytest.approx(expected_dcg / ideal_dcg)
    assert result["source_page_recall"] == 1.0


def test_legacy_dataset_falls_back_to_source_and_page() -> None:
    record = {
        "question": "旧数据问题",
        "ground_truth": "旧数据答案",
        "source_file": "行业/report.pdf",
        "page": 8,
    }
    retrieved = [
        {
            "chunk_id": "chunk-8",
            "source_file": "D:/data/行业/report.pdf",
            "source_page": 8,
        }
    ]

    result = evaluate_retrieval(record, retrieved, top_k=3)

    assert result["retrieval_label_mode"] == "source_page_fallback"
    assert result["hit_at_k"] == 1.0
    assert result["precision_at_k"] == pytest.approx(1 / 3)
    assert result["recall_at_k"] == 1.0


def test_no_answer_and_missing_labels_are_reported_separately() -> None:
    no_answer = evaluate_retrieval(
        {"question": "库中不存在的问题", "answerable": False},
        [],
        top_k=3,
    )
    missing = evaluate_retrieval(
        {"question": "缺标签", "ground_truth": "答案"},
        [],
        top_k=3,
    )

    summary = aggregate_retrieval_metrics([no_answer, missing])

    assert no_answer["no_answer_accuracy"] == 1.0
    assert summary["no_answer_count"] == 1
    assert summary["retrieval_missing_label_count"] == 1
    assert summary["retrieval_evaluable_count"] == 0


def test_dataset_audit_requires_stable_ids_and_chunk_labels() -> None:
    report = audit_records(
        [
            {
                "question": "有来源页但没有 chunk id",
                "ground_truth": "答案",
                "source_file": "report.pdf",
                "page": 1,
            },
            {
                "case_id": "case-2",
                "question": "无答案问题",
                "answerable": False,
            },
        ]
    )

    assert report["total_records"] == 2
    assert report["source_page_fallback"] == 1
    assert report["no_answer_records"] == 1
    assert report["issues"]["missing_case_id"] == 1
    assert report["issues"]["source_page_fallback_only"] == 1


def test_legacy_dataset_can_be_enriched_from_exact_chunk_preview() -> None:
    record = {
        "question": "营业收入是多少？",
        "ground_truth": "100亿元",
        "source_file": "行业/report.pdf",
        "page": 3,
        "chunk_text": "营业收入为100亿元。",
    }
    chunks = [
        StoredChunk(
            chunk_id="chunk-3",
            content="营业收入为100亿元。后续分析内容。",
            metadata={
                "source_file": "D:/data/行业/report.pdf",
                "page": 3,
                "chunk_id": "chunk-3",
            },
        )
    ]

    enriched, stats = enrich_records([record], chunks)

    assert stats["matched"] == 1
    assert enriched[0]["gold_chunk_ids"] == ["chunk-3"]
    assert enriched[0]["case_id"].startswith("rag_")
    assert enriched[0]["gold_sources"][0]["pages"] == [3]


def test_sqlite_only_loader_does_not_require_hnsw(tmp_path) -> None:
    database = tmp_path / "chroma.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE collections (id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE segments (id TEXT PRIMARY KEY, collection TEXT);
        CREATE TABLE embeddings (
            id INTEGER PRIMARY KEY,
            segment_id TEXT,
            embedding_id TEXT
        );
        CREATE TABLE embedding_metadata (
            id INTEGER,
            key TEXT,
            string_value TEXT,
            int_value INTEGER,
            float_value REAL,
            bool_value INTEGER
        );
        INSERT INTO collections VALUES ('collection-1', 'langchain');
        INSERT INTO segments VALUES ('segment-1', 'collection-1');
        INSERT INTO embeddings VALUES (1, 'segment-1', 'chunk-1');
        INSERT INTO embedding_metadata VALUES
            (1, 'chroma:document', '正文内容', NULL, NULL, NULL),
            (1, 'source_file', 'report.pdf', NULL, NULL, NULL),
            (1, 'page', NULL, 4, NULL, NULL),
            (1, 'has_image', NULL, NULL, NULL, 1);
        """
    )
    connection.commit()
    connection.close()

    chunks = load_chunks(tmp_path, sqlite_only=True)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk-1"
    assert chunks[0].content == "正文内容"
    assert chunks[0].metadata.source_file == "report.pdf"
    assert chunks[0].metadata.source_page == 4
    assert chunks[0].metadata.extra["has_image"] is True


def test_chunk_loader_prefers_configured_knowledge_service(
    monkeypatch,
    tmp_path,
) -> None:
    expected = [StoredChunk(chunk_id="remote-1", content="远程正文", metadata={})]
    monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "http://knowledge.test")
    monkeypatch.setattr(enrich_dataset, "_load_chunks_from_api", lambda: expected)

    assert load_chunks(tmp_path) == expected


def test_eval_reports_support_external_output_directory(tmp_path) -> None:
    summary = write_json_summary({"global": {}}, "test", tmp_path)
    detail = write_csv_detail([], "test", tmp_path)

    assert summary == tmp_path / "eval_summary_test.json"
    assert detail == tmp_path / "eval_detail_test.csv"
    assert summary.exists()
    assert detail.exists()
