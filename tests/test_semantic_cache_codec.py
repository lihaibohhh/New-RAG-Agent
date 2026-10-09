from __future__ import annotations

import numpy as np
import pytest

from knowledge.contracts import KnowledgeDocument
from knowledge.rag.infrastructure.cache.codec import (
    deserialize_documents,
    deserialize_vector,
    serialize_documents,
    serialize_vector,
)


def test_document_codec_preserves_traceability_metadata() -> None:
    documents = [
        KnowledgeDocument(
            content="证据正文",
            metadata={
                "source_file": "report.pdf",
                "page": np.int64(8),
                "chunk_id": "chunk-8",
            },
            document_id="document-8",
        )
    ]

    decoded = deserialize_documents(serialize_documents(documents))

    assert decoded[0].content == "证据正文"
    assert decoded[0].metadata.source_file == "report.pdf"
    assert decoded[0].metadata.source_page == 8
    assert decoded[0].metadata.chunk_id == "chunk-8"
    assert decoded[0].document_id == "document-8"


def test_vector_codec_uses_float32_and_validates_dimension() -> None:
    encoded = serialize_vector(np.array([0.25, 0.5], dtype=np.float64))

    decoded = deserialize_vector(encoded, expected_size=2)

    assert decoded.dtype == np.dtype("<f4")
    assert decoded.tolist() == [0.25, 0.5]
    with pytest.raises(ValueError, match="维度不匹配"):
        deserialize_vector(encoded, expected_size=3)


def test_document_codec_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="不支持的语义缓存格式"):
        deserialize_documents('{"schema_version":2,"documents":[]}')
