"""RAG BM25 索引和查询共用的中文检索分词器。"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import unicodedata
from importlib.metadata import PackageNotFoundError, version
from typing import Any


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


__all__ = [
    "BM25_TOKENIZER_VERSION",
    "bm25_tokenizer_fingerprint",
    "normalize_bm25_text",
    "tokenize_bm25",
]
