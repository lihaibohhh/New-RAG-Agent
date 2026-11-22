"""中文 BM25 召回、Redis 缓存与索引生命周期。"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import pickle
import re
import threading
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from knowledge.contracts import KnowledgeDocument
from knowledge.rag.contracts import Bm25RepositoryPort, ChunkCorpusPort


logger = logging.getLogger(__name__)

BM25_TOKENIZER_VERSION = "jieba-search-v1"

# 搜索分词仍会产生短词；词典用于确保金融指标、公司和技术名称
# 同时作为完整 token 进入 BM25。列表变化会改变缓存指纹。
_DOMAIN_TERMS = (
    "归母净利润",
    "扣非净利润",
    "营业收入",
    "净资产收益率",
    "毛利率",
    "净利率",
    "资本开支",
    "经营性现金流",
    "现金流量",
    "市盈率",
    "市净率",
    "同比增长",
    "环比增长",
    "同比",
    "环比",
    "基点",
    "点火价差",
    "楼面价",
    "公积金",
    "晶圆代工",
    "先进封装",
    "先进制程",
    "存储芯片",
    "光刻机",
    "覆铜板",
    "外延片",
    "台积电",
    "英伟达",
    "SK海力士",
    "中芯国际",
    "真武810E",
    "HBM4",
    "NVLink",
    "WMCM",
    "Vera Rubin",
)

_VALID_TOKEN = re.compile(
    r"^(?:[\u3400-\u4dbf\u4e00-\u9fff]+|"
    r"[a-z0-9]+(?:[._+/%\-][a-z0-9]+)*)$",
    re.IGNORECASE,
)
_PROTECTED_ASCII = re.compile(
    r"(?<![a-z0-9])(?:"
    r"[a-z0-9]+(?:[._+/%\-][a-z0-9]+)+|"
    r"[a-z]+\d+[a-z0-9]*|"
    r"\d+(?:\.\d+)?[a-z]+[a-z0-9]*"
    r")(?![a-z0-9])",
    re.IGNORECASE,
)
_tokenizer: Any | None = None
_tokenizer_lock = threading.Lock()

_INDEX_KEY = "rag:bm25_index"
_COUNT_KEY = "rag:bm25_doc_count"
_HMAC_KEY = "rag:bm25_hmac"
_VERSION_KEY = "rag:bm25_index_version"


def normalize_bm25_text(text: str) -> str:
    """统一全半角和 ASCII 大小写，不改动中文内容。"""
    return unicodedata.normalize("NFKC", str(text or "")).casefold()


def tokenize_bm25(text: str) -> list[str]:
    """使用 Jieba 搜索模式，并额外保留英文数字型号。"""
    normalized = normalize_bm25_text(text)
    if not normalized:
        return []

    tokens = [
        token
        for raw_token in _get_tokenizer().lcut_for_search(normalized, HMM=True)
        if (token := raw_token.strip()) and _VALID_TOKEN.fullmatch(token)
    ]
    # Jieba 可能把 2026H1、3.6TB/s 等再切分；额外保留完整表达。
    for protected in _PROTECTED_ASCII.findall(normalized):
        if protected not in tokens:
            tokens.append(protected)
    return tokens


def bm25_tokenizer_fingerprint() -> str:
    """生成 Redis BM25 缓存兼容性指纹。"""
    try:
        jieba_version = version("jieba")
    except PackageNotFoundError:
        jieba_version = "missing"
    normalized_terms = (normalize_bm25_text(term) for term in _DOMAIN_TERMS)
    payload = "\n".join(
        (BM25_TOKENIZER_VERSION, jieba_version, *normalized_terms)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _get_tokenizer() -> Any:
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    with _tokenizer_lock:
        if _tokenizer is None:
            import jieba

            jieba.setLogLevel(logging.WARNING)
            tokenizer = jieba.Tokenizer()
            tokenizer.initialize()
            for term in _DOMAIN_TERMS:
                tokenizer.add_word(
                    normalize_bm25_text(term),
                    freq=1_000_000,
                )
            _tokenizer = tokenizer
    return _tokenizer


class Bm25CandidateRetriever:
    """在 LangChain BM25 与内部 KnowledgeDocument 之间转换。"""

    def __init__(self, backend: Any, *, top_k: int) -> None:
        self.backend = backend
        self.backend.k = top_k

    @classmethod
    def build(
        cls,
        documents: list[KnowledgeDocument],
        *,
        top_k: int,
    ) -> "Bm25CandidateRetriever":
        from langchain_community.retrievers import BM25Retriever
        from langchain_core.documents import Document

        backend = BM25Retriever.from_documents(
            [
                Document(
                    page_content=document.content,
                    metadata=document.metadata.to_dict(),
                    id=document.document_id,
                )
                for document in documents
            ],
            preprocess_func=tokenize_bm25,
        )
        return cls(backend, top_k=top_k)

    def retrieve(self, query: str) -> list[KnowledgeDocument]:
        processed_query = self.backend.preprocess_func(query)
        scores = list(self.backend.vectorizer.get_scores(processed_query))
        if not scores or max(scores) <= 0:
            return []
        ranked_indices = sorted(
            range(len(scores)),
            key=scores.__getitem__,
            reverse=True,
        )
        documents = [
            self.backend.docs[index]
            for index in ranked_indices[: self.backend.k]
            if scores[index] > 0
        ]
        return [
            KnowledgeDocument(
                content=str(document.page_content or ""),
                metadata=dict(document.metadata or {}),
                document_id=(
                    str(document.id)
                    if getattr(document, "id", None) not in (None, "")
                    else None
                ),
            )
            for document in documents
        ]


class RedisBm25Repository:
    """使用 HMAC 校验后在 Redis 中缓存第三方 BM25 后端对象。"""

    def __init__(
        self,
        *,
        hmac_secret: str,
        ttl: int,
        redis_provider: Callable[[], Any],
    ) -> None:
        self._secret = hmac_secret.encode("utf-8") if hmac_secret else b""
        self._ttl = ttl
        self._redis_provider = redis_provider

    def _sign(self, data: bytes) -> str:
        return hmac.new(self._secret, data, hashlib.sha256).hexdigest()

    def load(self) -> Any | None:
        if not self._secret:
            logger.info("[RAG] BM25_HMAC_SECRET 未配置，跳过 Redis 索引缓存")
            return None
        try:
            redis = self._redis_provider()
            data = redis.get(_INDEX_KEY)
            if not data:
                return None
            raw_version = redis.get(_VERSION_KEY)
            version_value = (
                raw_version.decode("utf-8")
                if isinstance(raw_version, bytes)
                else str(raw_version or "")
            )
            expected_version = bm25_tokenizer_fingerprint()
            if version_value != expected_version:
                logger.info(
                    "[RAG] BM25 Redis 缓存分词版本不兼容，将重建 "
                    "cached=%s expected=%s",
                    version_value or "legacy",
                    expected_version,
                )
                return None
            raw_signature = redis.get(_HMAC_KEY)
            if not raw_signature:
                logger.warning("[RAG] BM25 Redis 缓存缺少签名，拒绝加载")
                return None
            signature = (
                raw_signature.decode("utf-8")
                if isinstance(raw_signature, bytes)
                else str(raw_signature)
            )
            if not hmac.compare_digest(self._sign(data), signature):
                logger.warning("[RAG] BM25 Redis 缓存签名验证失败，拒绝加载")
                return None
            backend = pickle.loads(data)
            raw_count = redis.get(_COUNT_KEY)
            count = int(raw_count.decode("utf-8")) if raw_count else 0
            logger.info("[RAG] BM25 索引从 Redis 加载 documents=%s", count)
            return backend
        except Exception as exc:
            logger.warning("[RAG] BM25 Redis 读取失败，将重建索引: %s", exc)
            return None

    def save(self, backend: Any, document_count: int) -> None:
        if not self._secret:
            return
        try:
            data = pickle.dumps(backend)
            pipeline = self._redis_provider().pipeline()
            pipeline.set(_INDEX_KEY, data, ex=self._ttl)
            pipeline.set(_COUNT_KEY, str(document_count), ex=self._ttl)
            pipeline.set(_HMAC_KEY, self._sign(data), ex=self._ttl)
            pipeline.set(
                _VERSION_KEY,
                bm25_tokenizer_fingerprint(),
                ex=self._ttl,
            )
            pipeline.execute()
            logger.info(
                "[RAG] BM25 索引写入 Redis documents=%s ttl=%s",
                document_count,
                self._ttl,
            )
        except Exception as exc:
            logger.warning("[RAG] BM25 Redis 写入失败，仅保留内存索引: %s", exc)

    def clear(self) -> None:
        try:
            self._redis_provider().delete(
                _INDEX_KEY,
                _COUNT_KEY,
                _HMAC_KEY,
                _VERSION_KEY,
            )
        except Exception as exc:
            raise RuntimeError("BM25 Redis 缓存清理失败") from exc


class LazyBm25CandidateRetrieverAdapter:
    """懒加载 BM25，并协调 Redis 缓存、重建和失效。"""

    def __init__(
        self,
        *,
        repository: Bm25RepositoryPort,
        corpus_reader: ChunkCorpusPort,
        top_k: int = 10,
    ) -> None:
        self._repository = repository
        self._corpus = corpus_reader
        self._top_k = max(1, int(top_k))
        self._retriever: Bm25CandidateRetriever | None = None
        self._lock = threading.Lock()
        self._generation = 0
        self._save_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="bm25-redis-save",
        )

    def prepare(self) -> bool:
        """确保 BM25 可用；空语料返回 False 而不是伪造空索引。"""
        return self._get_retriever() is not None

    def retrieve(self, query: str) -> list[KnowledgeDocument]:
        retriever = self._get_retriever()
        return retriever.retrieve(query) if retriever is not None else []

    def invalidate(self) -> None:
        with self._lock:
            self._generation += 1
            self._repository.clear()
            self._retriever = None
        logger.info("[RAG] BM25 Redis 与内存索引已失效")

    async def close(self) -> None:
        await asyncio.to_thread(self._save_executor.shutdown, wait=True)

    def _get_retriever(self) -> Bm25CandidateRetriever | None:
        if self._retriever is not None:
            return self._retriever
        with self._lock:
            if self._retriever is not None:
                return self._retriever

            backend = self._repository.load()
            if backend is not None:
                self._retriever = Bm25CandidateRetriever(
                    backend,
                    top_k=self._top_k,
                )
                return self._retriever

            documents = self._corpus.list_documents()
            if not documents:
                logger.info("[RAG] 知识库为空，本次不创建 BM25 索引")
                return None

            self._retriever = Bm25CandidateRetriever.build(
                documents,
                top_k=self._top_k,
            )
            generation = self._generation
            self._save_executor.submit(
                self._save_if_current,
                self._retriever.backend,
                len(documents),
                generation,
            )
            logger.info("[RAG] BM25 索引重建完成 documents=%s", len(documents))
            return self._retriever

    def _save_if_current(
        self,
        backend: object,
        document_count: int,
        generation: int,
    ) -> None:
        with self._lock:
            if generation != self._generation:
                logger.info("[RAG] 跳过已失效代次的 BM25 Redis 写入")
                return
            self._repository.save(backend, document_count)


__all__ = [
    "BM25_TOKENIZER_VERSION",
    "Bm25CandidateRetriever",
    "LazyBm25CandidateRetrieverAdapter",
    "RedisBm25Repository",
    "bm25_tokenizer_fingerprint",
    "normalize_bm25_text",
    "tokenize_bm25",
]
