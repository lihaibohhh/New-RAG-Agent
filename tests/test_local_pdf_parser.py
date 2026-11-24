from __future__ import annotations

from typing import Any

from knowledge.ingestion.contracts import ParseRequest
from knowledge.ingestion.infrastructure.parsing import local_pdf
from knowledge.ingestion.infrastructure.parsing.local_pdf import (
    LocalPdfParserAdapter,
)


def test_local_pdf_splitter_uses_embedding_tokenizer(monkeypatch) -> None:
    tokenizer = object()
    captured: dict[str, Any] = {}
    expected_splitter = object()

    monkeypatch.setattr(local_pdf, "_load_chunk_tokenizer", lambda _name: tokenizer)

    def fake_factory(received_tokenizer: object, **kwargs: Any) -> object:
        captured["tokenizer"] = received_tokenizer
        captured.update(kwargs)
        return expected_splitter

    monkeypatch.setattr(
        local_pdf.RecursiveCharacterTextSplitter,
        "from_huggingface_tokenizer",
        staticmethod(fake_factory),
    )

    splitter, loaded_tokenizer = local_pdf._create_text_splitter(
        tokenizer_name="BAAI/bge-small-zh-v1.5",
        chunk_size=448,
        chunk_overlap=48,
    )

    assert splitter is expected_splitter
    assert loaded_tokenizer is tokenizer
    assert captured["tokenizer"] is tokenizer
    assert captured["chunk_size"] == 448
    assert captured["chunk_overlap"] == 48
    assert captured["separators"][:2] == ["\n\n", "\n"]


def test_local_pdf_adapter_forwards_shared_pdf_chunking_policy(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_loader(file_path: str, **kwargs: Any) -> list[object]:
        captured["file_path"] = file_path
        captured.update(kwargs)
        return []

    monkeypatch.setattr(local_pdf, "load_pdf_with_layout", fake_loader)
    adapter = LocalPdfParserAdapter(
        chunking_max_tokens=448,
        chunking_overlap_tokens=48,
        chunking_tokenizer="BAAI/bge-small-zh-v1.5",
    )

    result = adapter._parse_sync(ParseRequest(file_path="report.pdf"))

    assert result.parser_name == "local"
    assert captured == {
        "file_path": "report.pdf",
        "source_root": None,
        "chunk_size": 448,
        "chunk_overlap": 48,
        "tokenizer_name": "BAAI/bge-small-zh-v1.5",
    }


def test_local_pdf_adapter_rejects_invalid_token_budget() -> None:
    for kwargs in (
        {"chunking_max_tokens": 63},
        {"chunking_max_tokens": 448, "chunking_overlap_tokens": 448},
        {"chunking_tokenizer": " "},
    ):
        try:
            LocalPdfParserAdapter(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("无效的本地 PDF Token 分块配置应被拒绝")


def test_local_pdf_splitter_rejects_budget_at_model_hard_limit(monkeypatch) -> None:
    class Tokenizer:
        model_max_length = 512

    monkeypatch.setattr(local_pdf, "_load_chunk_tokenizer", lambda _name: Tokenizer())

    try:
        local_pdf._create_text_splitter(
            tokenizer_name="BAAI/bge-small-zh-v1.5",
            chunk_size=512,
            chunk_overlap=48,
        )
    except ValueError as exc:
        assert "预留空间" in str(exc)
    else:
        raise AssertionError("本地 PDF 分块不得顶满 Embedding 模型硬上限")
