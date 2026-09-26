"""Unified knowledge namespace; the root exports protocol-neutral contracts."""
from knowledge.contracts import (
    ChunkMetadata,
    EvaluationCandidate,
    EvaluationRetrievalResult,
    IngestionReport,
    KnowledgeDocument,
    KnowledgeValidationError,
    RagHealthStatus,
    RetrievedChunk,
    RetrievalResult,
    SourceReference,
    StoredChunk,
    WarmupStatus,
)

__all__ = [
    "ChunkMetadata",
    "EvaluationCandidate",
    "EvaluationRetrievalResult",
    "IngestionReport",
    "KnowledgeDocument",
    "KnowledgeValidationError",
    "RagHealthStatus",
    "RetrievedChunk",
    "RetrievalResult",
    "SourceReference",
    "StoredChunk",
    "WarmupStatus",
]
