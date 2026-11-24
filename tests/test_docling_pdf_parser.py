from __future__ import annotations

from knowledge.ingestion.contracts import ParseRequest
from knowledge.ingestion.infrastructure.parsing.docling_pdf import (
    DoclingServiceAdapter,
)


def _table_chunk(
    *,
    index: int,
    text: str,
    table_ref: str = "#/tables/0",
    page: int = 1,
    tokens: int = 20,
) -> dict[str, object]:
    return {
        "chunk_index": index,
        "text": text,
        "page_numbers": [page],
        "doc_items": [table_ref],
        "num_tokens": tokens,
    }


def test_docling_request_configures_hybrid_chunker() -> None:
    adapter = DoclingServiceAdapter(
        base_url="http://docling.test",
        chunking_max_tokens=320,
        chunking_tokenizer="test-tokenizer",
        chunking_merge_peers=False,
    )

    assert adapter._chunking_form_fields() == {
        "chunking_max_tokens": "320",
        "chunking_tokenizer": "test-tokenizer",
        "chunking_merge_peers": "false",
        "chunking_use_markdown_tables": "true",
        "chunking_include_raw_text": "true",
    }


def test_docling_recombines_fragments_from_the_same_table() -> None:
    request = ParseRequest(file_path="report.pdf")
    payload = {
        "chunks": [
            _table_chunk(
                index=0,
                text="| 指标 | 金额 |\n| --- | --- |\n| 收入 | 100 |",
                page=1,
                tokens=30,
            ),
            _table_chunk(
                index=1,
                text="| 指标 | 金额 |\n| --- | --- |\n| 利润 | 20 |",
                table_ref="/tables/0",
                page=2,
                tokens=25,
            ),
        ]
    }

    result = DoclingServiceAdapter._normalize_result(request, payload)

    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.doc_type == "table"
    assert chunk.source_pages == (1, 2)
    assert "| 收入 | 100 |" in chunk.content
    assert "| 利润 | 20 |" in chunk.content
    assert chunk.content.count("| 指标 | 金额 |") == 2
    assert chunk.metadata["num_tokens"] == 55
    assert chunk.metadata["docling_table_fragment_count"] == 2
    assert chunk.metadata["docling_table_preserved"] is True


def test_docling_keeps_different_tables_and_text_separate() -> None:
    request = ParseRequest(file_path="report.pdf")
    payload = {
        "chunks": [
            _table_chunk(index=0, text="| A |\n| --- |", table_ref="#/tables/0"),
            {
                "chunk_index": 1,
                "text": "普通正文",
                "page_numbers": [1],
                "doc_items": ["#/texts/0"],
                "num_tokens": 4,
            },
            _table_chunk(index=2, text="| B |\n| --- |", table_ref="#/tables/1"),
        ]
    }

    result = DoclingServiceAdapter._normalize_result(request, payload)

    assert [chunk.doc_type for chunk in result.chunks] == ["table", "text", "table"]


def test_docling_can_leave_table_fragments_split() -> None:
    request = ParseRequest(file_path="report.pdf")
    payload = {
        "chunks": [
            _table_chunk(index=0, text="| A |\n| --- |"),
            _table_chunk(index=1, text="| B |\n| --- |"),
        ]
    }

    result = DoclingServiceAdapter._normalize_result(
        request,
        payload,
        preserve_tables=False,
    )

    assert len(result.chunks) == 2


def test_docling_rejects_chunk_limits_outside_supported_range() -> None:
    for value in (63, 8_193):
        try:
            DoclingServiceAdapter(
                base_url="http://docling.test",
                chunking_max_tokens=value,
            )
        except ValueError as exc:
            assert "64 到 8192" in str(exc)
        else:
            raise AssertionError("无效的 chunking_max_tokens 应被拒绝")
