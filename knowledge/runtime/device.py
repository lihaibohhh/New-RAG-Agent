"""Knowledge 本地模型的计算设备选择。"""

from __future__ import annotations

import logging
from typing import Literal


logger = logging.getLogger(__name__)
_VALID_DEVICES = {"auto", "cpu", "cuda"}


def resolve_runtime_device(requested: str = "auto") -> Literal["cpu", "cuda"]:
    """解析共享模型设备，并在 CUDA 不可用时安全回退到 CPU。"""
    configured = (requested or "auto").strip().lower()
    if configured not in _VALID_DEVICES:
        raise ValueError(
            "RAG_DEVICE 仅支持 auto、cpu 或 cuda，" f"当前值为 {configured!r}"
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


__all__ = ["resolve_runtime_device"]
