"""Agent Tool 的超时与指数退避执行策略。"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
import random
import time
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from react_agent.tooling.results import tool_error


ToolCallable = TypeVar("ToolCallable", bound=Callable[..., Any])


def with_retry(
    *,
    tool_name: str,
    max_retries: int = 3,
    timeout: float = 10.0,
    base_delay: float = 0.5,
    max_delay: float = 6.0,
    retry_on: Iterable[type[BaseException]] = (
        TimeoutError,
        ConnectionError,
        OSError,
    ),
    retry_timeouts: bool = True,
) -> Callable[[ToolCallable], ToolCallable]:
    """为同步或异步工具增加超时、指数退避和统一错误结果。"""
    retryable_exceptions = tuple(retry_on)

    def decorator(function: ToolCallable) -> ToolCallable:
        is_async = inspect.iscoroutinefunction(function)

        async def call_async(*args: Any, **kwargs: Any) -> str:
            last_exception: BaseException | None = None
            query = str(kwargs.get("query") or (args[0] if args else "")).strip()
            attempt = 0
            for attempt in range(max_retries + 1):
                started_at = time.perf_counter()
                try:
                    if is_async:
                        result = await asyncio.wait_for(
                            function(*args, **kwargs),
                            timeout=timeout,
                        )
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(function, *args, **kwargs),
                            timeout=timeout,
                        )
                    if not isinstance(result, dict):
                        return json.dumps(
                            tool_error(
                                tool_name=tool_name,
                                query=query,
                                code="BAD_TOOL_RETURN",
                                message=(
                                    "工具返回类型不是 dict，而是 "
                                    f"{type(result).__name__}"
                                ),
                                meta={"attempt": attempt, "timeout": timeout},
                            ),
                            ensure_ascii=False,
                        )
                    meta = result.setdefault("meta", {})
                    if not isinstance(meta, dict):
                        meta = result["meta"] = {}
                    meta.update(
                        attempt=attempt,
                        timeout=timeout,
                        elapsed=round(time.perf_counter() - started_at, 3),
                    )
                    return json.dumps(result, ensure_ascii=False)
                except Exception as exc:
                    last_exception = exc
                    is_timeout = isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                    if (is_timeout and not retry_timeouts) or (
                        retryable_exceptions
                        and not isinstance(exc, retryable_exceptions)
                    ):
                        break
                if attempt < max_retries:
                    delay = min(max_delay, base_delay * (2**attempt))
                    await asyncio.sleep(delay * (0.8 + 0.4 * random.random()))

            is_timeout = isinstance(
                last_exception,
                (TimeoutError, asyncio.TimeoutError),
            )
            if is_timeout:
                code = "TOOL_TIMEOUT"
                message = f"工具调用超过 {timeout:g} 秒，已终止等待"
            elif attempt:
                code = "TOOL_FAILED"
                message = (
                    f"工具调用失败（已重试 {attempt} 次）："
                    f"{type(last_exception).__name__}: {last_exception}"
                )
            else:
                code = "TOOL_FAILED"
                message = (
                    f"工具调用失败：{type(last_exception).__name__}: "
                    f"{last_exception}"
                )
            return json.dumps(
                tool_error(
                    tool_name=tool_name,
                    query=query,
                    code=code,
                    message=message,
                    meta={
                        "retries": attempt,
                        "timeout": timeout,
                        "retryable": not is_timeout,
                    },
                ),
                ensure_ascii=False,
            )

        if is_async:

            @functools.wraps(function)
            async def async_wrapper(*args: Any, **kwargs: Any) -> str:
                return await call_async(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(function)
        def sync_wrapper(*args: Any, **kwargs: Any) -> str:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(asyncio.run, call_async(*args, **kwargs)).result()
            return asyncio.run(call_async(*args, **kwargs))

        return sync_wrapper  # type: ignore[return-value]

    return decorator


__all__ = ["with_retry"]
