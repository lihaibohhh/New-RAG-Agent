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
from knowledge.services import IngestionService, RagService
from knowledge.settings import KnowledgeSettings

__all__ = [
    "ChunkMetadata",
    "EvaluationCandidate",
    "EvaluationRetrievalResult",
    "IngestionReport",
    "IngestionService",
    "KnowledgeDocument",
    "KnowledgeValidationError",
    "RagHealthStatus",
    "RagService",
    "RetrievedChunk",
    "RetrievalResult",
    "SourceReference",
    "StoredChunk",
    "WarmupStatus",
    "KnowledgeSettings",
]
