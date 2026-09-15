from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from eval.dataset_generator import (
    EvidenceQuote,
    EvidenceUnit,
    QAPair,
    build_evidence_units,
    save_dataset_bundle,
    stratified_sample,
    validate_qa_pair,
)


def _chunk(
    chunk_id: str,
    *,
    source: str,
    page: int,
    industry: str,
    doc_type: str = "text",
    parser: str = "docling",
    text: str = "中芯国际2025年收入为100亿元，毛利率达到20%。" * 4,
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": chunk_id,
            "source_file": source,
            "page": page,
            "industry": industry,
            "doc_type": doc_type,
            "parser": parser,
        },
    )


def test_sampling_balances_industry_type_parser_and_document() -> None:
    chunks = [
        *[
            _chunk(f"a-{index}", source="a.pdf", page=index, industry="半导体")
            for index in range(1, 8)
        ],
        _chunk(
            "a-table",
            source="b.pdf",
            page=2,
            industry="半导体",
            doc_type="table",
        ),
        _chunk(
            "b-local",
            source="c.pdf",
            page=3,
            industry="电力",
            parser="local",
        ),
    ]

    sampled = stratified_sample(chunks, max_chunks=3, seed=7)

    assert {item.metadata["industry"] for item in sampled} == {"半导体", "电力"}
    assert {item.metadata["doc_type"] for item in sampled} == {"text", "table"}
    assert {item.metadata["parser"] for item in sampled} == {"docling", "local"}
    assert len({item.metadata["source_file"] for item in sampled}) == 3


def test_multi_chunk_units_only_pair_adjacent_chunks_from_same_document() -> None:
    chunks = [
        _chunk("a-1", source="a.pdf", page=1, industry="半导体"),
        _chunk("a-2", source="a.pdf", page=2, industry="半导体"),
        _chunk("b-8", source="b.pdf", page=8, industry="电力"),
    ]

    units = build_evidence_units(chunks, multi_chunk_ratio=0.5, seed=1)
    multi = [unit for unit in units if unit.scope == "multi_chunk"]

    assert len(multi) == 1
    assert [chunk.metadata["chunk_id"] for chunk in multi[0].chunks] == ["a-1", "a-2"]


def test_evidence_validation_checks_quotes_numbers_and_multi_chunk_coverage() -> None:
    unit = EvidenceUnit(
        chunks=(
            _chunk(
                "a-1",
                source="a.pdf",
                page=1,
                industry="半导体",
                text="中芯国际2025年收入为100亿元。",
            ),
            _chunk(
                "a-2",
                source="a.pdf",
                page=2,
                industry="半导体",
                text="中芯国际同期毛利率达到20%。",
            ),
        ),
        scope="multi_chunk",
    )
    valid_pair = QAPair(
        question="中芯国际2025年的收入和同期毛利率分别是多少？",
        ground_truth="收入为100亿元，毛利率达到20%。",
        category="comparison",
        evidence=[
            EvidenceQuote(chunk_index=1, quote="2025年收入为100亿元"),
            EvidenceQuote(chunk_index=2, quote="同期毛利率达到20%"),
        ],
    )
    invalid_pair = valid_pair.model_copy(
        update={"ground_truth": "收入为200亿元，毛利率达到20%。"}
    )

    assert validate_qa_pair(valid_pair, unit) == (True, ())
    valid, reasons = validate_qa_pair(invalid_pair, unit)
    assert valid is False
    assert "unsupported_answer_numbers:200" in reasons


def test_balanced_bundle_exports_stable_smoke_and_regression_sets(tmp_path) -> None:
    records = [
        {
            "case_id": f"case-{index:02d}",
            "question": f"问题 {index}",
            "industry": "半导体" if index % 2 else "电力",
            "category": "numeric_lookup" if index % 3 else "comparison",
            "reasoning_scope": "multi_chunk" if index % 4 == 0 else "single_chunk",
            "answerable": index != 9,
            "review_status": "pending",
        }
        for index in range(10)
    ]
    output = tmp_path / "candidate.jsonl"

    paths = save_dataset_bundle(
        records,
        output,
        smoke_size=3,
        regression_size=6,
        seed=42,
        overwrite=False,
    )

    assert len(paths["full"].read_text(encoding="utf-8").splitlines()) == 10
    assert len(paths["smoke"].read_text(encoding="utf-8").splitlines()) == 3
    assert len(paths["regression"].read_text(encoding="utf-8").splitlines()) == 6
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["counts"]["no_answer"] == 1
    assert manifest["counts"]["pending_review"] == 10

    with pytest.raises(FileExistsError):
        save_dataset_bundle(
            records,
            output,
            smoke_size=3,
            regression_size=6,
            seed=42,
            overwrite=False,
        )
