"""AgentContext 的环境变量装载适配器。"""

from __future__ import annotations

import logging
import os
from types import UnionType
from dataclasses import fields
from typing import Annotated, Any, Union, get_args, get_origin, get_type_hints


logger = logging.getLogger(__name__)


def _to_bool(value: str) -> bool:
    return (value or "").strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }


def _unwrap_annotated(field_type: Any) -> Any:
    if get_origin(field_type) is Annotated:
        return get_args(field_type)[0]
    return field_type


def _coerce(value: str, field_type: Any) -> Any:
    """将环境变量字符串转换成 dataclass 字段声明的类型。"""
    field_type = _unwrap_annotated(field_type)
    origin = get_origin(field_type)

    if origin is None:
        if field_type is bool:
            return _to_bool(value)
        if field_type is int:
            return int(value)
        if field_type is float:
            return float(value)
        return value

    if origin is list:
        inner = get_args(field_type)[0] if get_args(field_type) else str
        parts = [part.strip() for part in value.split(",") if part.strip()]
        return [_coerce(part, inner) for part in parts]

    if origin in {dict, tuple}:
        return value

    if origin in {Union, UnionType}:
        for candidate in get_args(field_type):
            if candidate is type(None):
                continue
            try:
                return _coerce(value, candidate)
            except (TypeError, ValueError):
                continue

    return value


def apply_environment_overrides(target: Any) -> None:
    """以大写字段名为 key，将环境变量覆盖到仍使用默认值的字段。"""
    type_hints = get_type_hints(target.__class__, include_extras=True)
    for data_field in fields(target):
        if not data_field.init:
            continue

        current = getattr(target, data_field.name)
        if current != data_field.default:
            continue

        env_key = data_field.name.upper()
        raw = os.environ.get(env_key)
        if raw is None or not raw.strip():
            continue

        field_type = type_hints.get(data_field.name, data_field.type)
        try:
            setattr(target, data_field.name, _coerce(raw, field_type))
        except (TypeError, ValueError):
            logger.warning(
                "[AgentContext] env %s=%r cannot be converted to %s; ignored",
                env_key,
                raw,
                field_type,
            )


__all__ = ["apply_environment_overrides"]
