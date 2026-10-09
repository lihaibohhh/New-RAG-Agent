"""语义缓存文档与向量的纯编码函数。"""
from __future__ import annotations

import json
import math
from typing import Any

import numpy as np

from knowledge.contracts import KnowledgeDocument


def json_safe(value: Any) -> Any:
    """把元数据收敛为 JSON 基本类型，不执行任意对象反序列化。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def serialize_documents(documents: list[KnowledgeDocument]) -> bytes:
    """将内部 RAG 文档编码为受限 JSON 格式。"""
    items: list[dict[str, Any]] = []
    for document in documents:
        item: dict[str, Any] = {
            "page_content": str(document.content or ""),
            "metadata": json_safe(document.metadata.to_dict()),
        }
        if document.document_id not in (None, ""):
            item["id"] = str(document.document_id)
        items.append(item)

    return json.dumps(
        {"schema_version": 1, "documents": items},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def deserialize_documents(data: bytes | str) -> list[KnowledgeDocument]:
    """读取受限 JSON，并重建内部 RAG 文档。"""
    raw = data.decode("utf-8") if isinstance(data, bytes) else data
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("不支持的语义缓存格式")

    items = payload.get("documents")
    if not isinstance(items, list):
        raise ValueError("语义缓存 documents 字段无效")

    documents: list[KnowledgeDocument] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("语义缓存文档条目无效")
        page_content = item.get("page_content")
        metadata = item.get("metadata", {})
        if not isinstance(page_content, str) or not isinstance(metadata, dict):
            raise ValueError("语义缓存文档字段无效")
        document_id = item.get("id")
        documents.append(
            KnowledgeDocument(
                content=page_content,
                metadata=metadata,
                document_id=(
                    str(document_id) if document_id not in (None, "") else None
                ),
            )
        )
    return documents


def serialize_vector(vector: np.ndarray) -> bytes:
    """以固定的小端 float32 格式保存向量。"""
    return np.asarray(vector, dtype="<f4").reshape(-1).tobytes(order="C")


def deserialize_vector(data: bytes, *, expected_size: int) -> np.ndarray:
    """从原始 float32 字节恢复向量，并校验维度。"""
    if len(data) % np.dtype("<f4").itemsize != 0:
        raise ValueError("缓存向量字节长度无效")
    vector = np.frombuffer(data, dtype="<f4")
    if vector.size != expected_size:
        raise ValueError(
            f"缓存向量维度不匹配: expected={expected_size}, actual={vector.size}"
        )
    return vector


__all__ = [
    "deserialize_documents",
    "deserialize_vector",
    "serialize_documents",
    "serialize_vector",
]
