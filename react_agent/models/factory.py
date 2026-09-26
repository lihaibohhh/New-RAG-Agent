"""根据模型引用和应用配置创建 LangChain ChatModel。"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from react_agent.configuration.settings import settings


@dataclass(frozen=True)
class ChatModelSettings:
    """从统一配置读取的模型推理参数。"""

    temperature: float = field(default_factory=lambda: settings.llm.llm_temperature)
    max_tokens: int = field(default_factory=lambda: settings.llm.llm_max_tokens)
    timeout: int = field(default_factory=lambda: settings.llm.llm_timeout)
    retries: int = field(default_factory=lambda: settings.llm.llm_retries)


def _parse_model_ref(model_ref: str) -> tuple[str, str]:
    model_ref = (model_ref or "").strip()
    if not model_ref or "/" not in model_ref:
        raise ValueError(
            "模型引用必须是 'provider/model-name' 格式，"
            "例如 'openai/gpt-4.1-mini'"
        )
    provider, model_name = (part.strip() for part in model_ref.split("/", 1))
    provider = provider.lower()
    if not provider or not model_name:
        raise ValueError("模型引用中的 provider 和 model-name 均不能为空")
    return provider, model_name


def _get_secret(attribute: str, environment_key: str, hint: str) -> str:
    value = str(getattr(settings.secrets, attribute, "") or "").strip()
    if not value:
        raise EnvironmentError(
            f"缺少密钥 {environment_key}。{hint}\n"
            f"请在 .env 或 Conda 环境中设置 {environment_key}"
        )
    return value


def _provider_name(model_ref: str) -> str:
    """Return a normalized provider name without requiring a full model ref."""
    value = (model_ref or "").strip()
    if "/" not in value:
        return value.lower()
    return value.split("/", 1)[0].strip().lower()


def output_token_limit_kwargs(
    model_ref: str,
    max_tokens: int,
) -> dict[str, object]:
    """Map the output limit to the parameter supported by each provider.

    ``ChatOpenAI`` aliases ``max_tokens`` to OpenAI's
    ``max_completion_tokens``. DeepSeek's compatible endpoint intentionally
    does not use that field, so its ``max_tokens`` parameter must be passed
    through ``extra_body``.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens 必须大于 0")
    provider = _provider_name(model_ref)
    if provider in {"deepseek", "ds"}:
        return {"extra_body": {"max_tokens": max_tokens}}
    if provider == "openai":
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


def bind_output_token_limit(
    model: BaseChatModel,
    *,
    model_ref: str,
    max_tokens: int,
) -> BaseChatModel:
    """Return a runnable with a provider-correct per-call output limit."""
    bind = getattr(model, "bind", None)
    if not callable(bind):
        return model
    return bind(**output_token_limit_kwargs(model_ref, max_tokens))


def _build_openai(model_name: str, config: ChatModelSettings) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError("未安装依赖：pip install langchain-openai") from exc
    return ChatOpenAI(
        model=model_name,
        temperature=config.temperature,
        max_completion_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.retries,
    )


def _build_anthropic(model_name: str, config: ChatModelSettings) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError("未安装依赖：pip install langchain-anthropic") from exc
    return ChatAnthropic(
        model=model_name,
        api_key=settings.secrets.ANTHROPIC_API_KEY,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.retries,
    )


def _build_local(model_name: str, config: ChatModelSettings) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError("未安装依赖：pip install langchain-openai") from exc
    return ChatOpenAI(
        model=model_name,
        base_url=settings.secrets.LOCAL_OPENAI_BASE_URL or "http://127.0.0.1:8000/v1",
        api_key=settings.secrets.LOCAL_OPENAI_API_KEY or "local-key",
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.retries,
    )


def _build_deepseek(model_name: str, config: ChatModelSettings) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError("未安装依赖：pip install langchain-openai") from exc
    return ChatOpenAI(
        model=model_name,
        base_url=_get_secret(
            "DEEPSEEK_BASE_URL",
            "DEEPSEEK_BASE_URL",
            "例如：https://api.deepseek.com",
        ),
        api_key=_get_secret("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY", "例如：sk-..."),
        temperature=config.temperature,
        extra_body={"max_tokens": config.max_tokens},
        timeout=config.timeout,
        max_retries=config.retries,
    )


@lru_cache(maxsize=32)
def load_chat_model(model_ref: str) -> BaseChatModel:
    """创建并缓存模型 Provider 适配器。"""
    provider, model_name = _parse_model_ref(model_ref)
    config = ChatModelSettings()
    if provider == "openai":
        return _build_openai(model_name, config)
    if provider == "anthropic":
        return _build_anthropic(model_name, config)
    if provider in {"local", "qwen-local", "openai-compatible"}:
        return _build_local(model_name, config)
    if provider in {"deepseek", "ds"}:
        return _build_deepseek(model_name, config)
    raise ValueError(
        f"不支持的 provider：{provider}。支持：openai / anthropic / local / deepseek"
    )


__all__ = [
    "ChatModelSettings",
    "bind_output_token_limit",
    "load_chat_model",
    "output_token_limit_kwargs",
]
