"""由 Knowledge Runtime 持有的惰性 Embedding 资源提供器。"""
from __future__ import annotations

import logging
import threading
from typing import Any


logger = logging.getLogger(__name__)


class EmbeddingProviderAdapter:
    """惰性创建并保留一份供建库与检索复用的 Embedding 模型。"""

    def __init__(self, *, model_name: str, device: str) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any | None = None
        self._lock = threading.Lock()

    def get(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                from langchain_huggingface import HuggingFaceEmbeddings

                self._model = HuggingFaceEmbeddings(
                    model_name=self._model_name,
                    model_kwargs={"device": self._device},
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info(
                    "[KNOWLEDGE] Embedding 模型初始化完成 model=%s device=%s",
                    self._model_name,
                    self._device,
                )
        return self._model

    async def close(self) -> None:
        """Release the runtime's strong reference to the model."""
        with self._lock:
            self._model = None


__all__ = ["EmbeddingProviderAdapter"]
