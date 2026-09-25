"""RAG 本地模型的计算设备选择。"""

from __future__ import annotations

import logging

from knowledge.runtime.config import KnowledgeRuntimeConfig, RagProfile


logger = logging.getLogger(__name__)
_VALID_DEVICES = {"auto", "cpu", "cuda"}


def resolve_rag_device(requested: str = "auto") -> str:
    """解析 RAG_DEVICE，并在 CUDA 不可用时安全回退到 CPU。"""
    configured = (requested or "auto").strip().lower()
    if configured not in _VALID_DEVICES:
        raise ValueError(
            "RAG_DEVICE 仅支持 auto、cpu 或 cuda，"
            f"当前值为 {configured!r}"
        )

    import torch

    cuda_available = torch.cuda.is_available()
    if configured == "cpu":
        return "cpu"
    if configured == "cuda" and not cuda_available:
        logger.warning("RAG_DEVICE=cuda，但当前运行环境无法使用 CUDA；已回退到 CPU")
        return "cpu"
    if configured == "cuda" or cuda_available:
        return "cuda"
    return "cpu"


def get_rag_runtime_profile(config: KnowledgeRuntimeConfig) -> RagProfile:
    """解析设备并返回该设备对应的最终 RAG 运行档位。"""
    device = resolve_rag_device(config.requested_device)
    return config.rag.profile(device)


__all__ = ["get_rag_runtime_profile", "resolve_rag_device"]
