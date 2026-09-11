"""RAG 模块对外稳定的数据契约。"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

OcrPolicy = Literal["auto", "force", "disabled"]
ParserHint = Literal["auto", "local", "docling"]


class RagValidationError(ValueError):
    """RAG 请求参数不符合公共契约。"""


_CORE_METADATA_KEYS = frozenset(
    {
        "source",
        "source_path",
        "source_file",
        "page",
        "source_page",
        "source_pages",
        "chunk_id",
        "doc_type",
        "industry",
        "score",
        "retrieval_score",
    }
)


def _optional_text(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _page_tuple(value: Any) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = (value,)
    pages: list[int] = []
    for item in values:
        page = _optional_int(item)
        if page is not None and page not in pages:
            pages.append(page)
    return tuple(pages)


@dataclass(frozen=True)
class SourceReference:
    """跨解析、存储和查询阶段传递的统一来源引用。"""

    source_file: str = ""
    source_path: str | None = None
    source_page: int | None = None
    source_pages: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_file", str(self.source_file or ""))
        object.__setattr__(self, "source_path", _optional_text(self.source_path))
        object.__setattr__(self, "source_page", _optional_int(self.source_page))
        object.__setattr__(self, "source_pages", _page_tuple(self.source_pages))

    def display_file(self, *, legacy_chunk_id: str | None = None) -> str:
        """返回稳定的展示来源；仅在读取旧索引时兼容历史 chunk_id。"""
        if self.source_file:
            return self.source_file.replace("\\", "/")
        if legacy_chunk_id:
            relative_name = legacy_chunk_id.split("::", 1)[0].strip()
            if relative_name:
                return relative_name.replace("\\", "/")
        if self.source_path:
            try:
                return Path(self.source_path).name.replace("\\", "/")
            except Exception:
                return self.source_path.replace("\\", "/")
        return ""


@dataclass(frozen=True)
class ChunkMetadata(Mapping[str, Any]):
    """类型化核心元数据，同时兼容现有 dict/Chroma 序列化格式。"""

    source: SourceReference = field(default_factory=SourceReference)
    chunk_id: str = ""
    doc_type: str | None = None
    industry: str | None = None
    retrieval_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = self.source
        if not isinstance(source, SourceReference):
            raise TypeError("source 必须是 SourceReference")
        object.__setattr__(self, "chunk_id", str(self.chunk_id or ""))
        object.__setattr__(self, "doc_type", _optional_text(self.doc_type))
        object.__setattr__(self, "industry", _optional_text(self.industry))
        object.__setattr__(
            self,
            "retrieval_score",
            _optional_float(self.retrieval_score),
        )
        object.__setattr__(
            self,
            "extra",
            {
                str(key): value
                for key, value in dict(self.extra or {}).items()
                if str(key) not in _CORE_METADATA_KEYS
            },
        )

    @classmethod
    def from_mapping(
        cls,
        metadata: "ChunkMetadata | Mapping[str, Any] | None",
    ) -> "ChunkMetadata":
        if isinstance(metadata, cls):
            return metadata
        raw = dict(metadata or {})
        source_page = raw.get("source_page", raw.get("page"))
        retrieval_score = raw.get("retrieval_score", raw.get("score"))
        return cls(
            source=SourceReference(
                source_file=str(raw.get("source_file") or ""),
                source_path=_optional_text(
                    raw.get("source_path", raw.get("source"))
                ),
                source_page=_optional_int(source_page),
                source_pages=_page_tuple(raw.get("source_pages")),
            ),
            chunk_id=str(raw.get("chunk_id") or ""),
            doc_type=_optional_text(raw.get("doc_type")),
            industry=_optional_text(raw.get("industry")),
            retrieval_score=_optional_float(retrieval_score),
            extra={
                str(key): value
                for key, value in raw.items()
                if str(key) not in _CORE_METADATA_KEYS
            },
        )

    @property
    def source_file(self) -> str:
        return self.source.source_file

    @property
    def source_path(self) -> str | None:
        return self.source.source_path

    @property
    def source_page(self) -> int | None:
        return self.source.source_page

    @property
    def source_pages(self) -> tuple[int, ...]:
        return self.source.source_pages

    def resolved_source_file(self) -> str:
        return self.source.display_file(legacy_chunk_id=self.chunk_id)

    def to_dict(self) -> dict[str, Any]:
        """转换为兼容现有 Chroma、LangChain 和 Redis 的平铺结构。"""
        payload = dict(self.extra)
        if self.source_path is not None:
            payload["source"] = self.source_path
        if self.source_file:
            payload["source_file"] = self.source_file
        if self.source_page is not None:
            payload["page"] = self.source_page
        if self.source_pages:
            payload["source_pages"] = ",".join(
                str(page) for page in self.source_pages
            )
        if self.chunk_id:
            payload["chunk_id"] = self.chunk_id
        if self.doc_type is not None:
            payload["doc_type"] = self.doc_type
        if self.industry is not None:
            payload["industry"] = self.industry
        if self.retrieval_score is not None:
            payload["score"] = self.retrieval_score
        return payload

    def to_storage_dict(self) -> dict[str, bool | int | float | str]:
        """生成 Chroma 可接受的标量 metadata。"""
        return {
            key: value
            for key, value in self.to_dict().items()
            if isinstance(value, (bool, int, float, str))
        }

    def matches(self, filters: Mapping[str, Any] | None) -> bool:
        return not filters or all(
            self.get(key) == value for key, value in filters.items()
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True, init=False)
class RagDocument:
    """RAG 内部跨端口传递的文档，隔离 LangChain Document。"""

    content: str
    metadata: ChunkMetadata
    document_id: str | None = None

    def __init__(
        self,
        content: str,
        metadata: ChunkMetadata | Mapping[str, Any] | None = None,
        document_id: str | None = None,
    ) -> None:
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "metadata", ChunkMetadata.from_mapping(metadata))
        object.__setattr__(
            self,
            "document_id",
            _optional_text(document_id),
        )


@dataclass(frozen=True)
class ParseRequest:
    """一次文档解析请求，不暴露具体解析器实现。"""

    file_path: str
    source_root: str | None = None
    parser_hint: ParserHint = "auto"
    ocr_policy: OcrPolicy = "auto"

    def normalized(self) -> "ParseRequest":
        file_path = (self.file_path or "").strip()
        if not file_path:
            raise RagValidationError("待解析文件路径不能为空")
        source_root = (self.source_root or "").strip() or None
        if self.parser_hint not in {"auto", "local", "docling"}:
            raise RagValidationError("parser_hint 必须是 auto、local 或 docling")
        if self.ocr_policy not in {"auto", "force", "disabled"}:
            raise RagValidationError("ocr_policy 必须是 auto、force 或 disabled")
        return ParseRequest(
            file_path=file_path,
            source_root=source_root,
            parser_hint=self.parser_hint,
            ocr_policy=self.ocr_policy,
        )


@dataclass(frozen=True)
class ParsedChunk:
    """解析阶段的统一片段，独立于 Docling 和 LangChain。"""

    content: str
    source_file: str
    source_page: int | None
    source_pages: tuple[int, ...]
    chunk_id: str
    doc_type: str = "text"
    headings: tuple[str, ...] = ()
    parser_name: str = "unknown"
    parser_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParseResult:
    """文档解析服务的统一结果。"""

    source_file: str
    parser_name: str
    chunks: tuple[ParsedChunk, ...] = ()
    processing_time: float = 0.0
    warnings: tuple[str, ...] = ()
    ocr_policy: OcrPolicy | None = None
    route_reasons: tuple[str, ...] = ()

    @property
    def has_content(self) -> bool:
        return bool(self.chunks)


@dataclass(frozen=True)
class RetrievalRequest:
    """一次知识库检索请求。"""

    query: str
    top_k: int = 3
    filters: dict[str, Any] | None = None

    def normalized(self) -> "RetrievalRequest":
        query = (self.query or "").strip()
        if not query:
            raise RagValidationError("检索词不能为空")
        try:
            top_k = int(self.top_k)
        except (TypeError, ValueError) as exc:
            raise RagValidationError("top_k 必须是整数") from exc
        return RetrievalRequest(
            query=query,
            top_k=max(1, min(top_k, 10)),
            filters=dict(self.filters or {}) or None,
        )


@dataclass(frozen=True)
class RetrievedChunk:
    """与 LangChain、Chroma 解耦的检索结果片段。"""

    content: str
    source_file: str
    source_page: int | None
    chunk_id: str
    score: float | None = None
    doc_type: str | None = None
    industry: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为协议层可直接序列化的平铺结构。"""
        item: dict[str, Any] = {
            "content": self.content,
            "source": self.source_file,
            "page": self.source_page,
        }
        for key in ("chunk_id", "score", "doc_type", "industry"):
            value = getattr(self, key)
            if value not in (None, ""):
                item[key] = value
        return item


@dataclass(frozen=True, init=False)
class StoredChunk:
    """从知识库管理面读取的原始片段，不依赖 Chroma 返回结构。"""

    chunk_id: str
    content: str
    metadata: ChunkMetadata

    def __init__(
        self,
        chunk_id: str,
        content: str,
        metadata: ChunkMetadata | Mapping[str, Any] | None = None,
    ) -> None:
        object.__setattr__(self, "chunk_id", str(chunk_id or ""))
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "metadata", ChunkMetadata.from_mapping(metadata))


@dataclass(frozen=True)
class RetrievalResult:
    """RAG 查询服务的统一返回值。"""

    query: str
    chunks: tuple[RetrievedChunk, ...] = ()
    stage: str = "unknown"
    cache_hit: bool = False
    candidates_count: int = 0
    reranked_count: int = 0
    top_score: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def has_relevant_content(self) -> bool:
        return bool(self.chunks)

    def results(self) -> list[dict[str, Any]]:
        return [chunk.to_dict() for chunk in self.chunks]


@dataclass(frozen=True)
class IngestionReport:
    """建库服务的结构化执行结果。"""

    data_dir: str
    status: str
    files_discovered: int = 0
    files_selected: int = 0
    files_processed: int = 0
    files_skipped: int = 0
    chunks_written: int = 0
    cache_invalidated: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WarmupStatus:
    """RAG 重资源预热结果。"""

    ready: bool
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RagHealthStatus:
    """RAG 运行时的轻量健康快照。"""

    ready: bool
    state: str
    details: dict[str, Any] = field(default_factory=dict)
