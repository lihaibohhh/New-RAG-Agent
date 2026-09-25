"""从包资源加载并缓存内置 Skill。"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib.resources import files
from typing import Any, Mapping

import yaml

from react_agent.skills.contracts import SkillDefinition


_BUILTIN_DIRECTORY_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


def parse_skill_markdown(content: str) -> SkillDefinition:
    """解析带 YAML frontmatter 的 SKILL.md。"""
    normalized = content.lstrip("\ufeff")
    lines = normalized.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md 必须以 YAML frontmatter 开头")

    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise ValueError("SKILL.md 缺少 frontmatter 结束标记") from exc

    metadata: Any = yaml.safe_load("\n".join(lines[1:closing_index])) or {}
    if not isinstance(metadata, Mapping):
        raise ValueError("SKILL.md frontmatter 必须是键值映射")

    return SkillDefinition(
        name=str(metadata.get("name") or "").strip(),
        version=str(metadata.get("version") or "").strip(),
        description=str(metadata.get("description") or "").strip(),
        instructions="\n".join(lines[closing_index + 1 :]).strip(),
    )


@lru_cache(maxsize=None)
def load_builtin_skill(directory_name: str) -> SkillDefinition:
    """读取一个受信任的包内 Skill；结果在进程内缓存。"""
    if not _BUILTIN_DIRECTORY_PATTERN.fullmatch(directory_name):
        raise ValueError("内置 Skill 目录名必须使用小写 snake_case")
    resource = files("react_agent.skills").joinpath(
        "builtin", directory_name, "SKILL.md"
    )
    return parse_skill_markdown(resource.read_text(encoding="utf-8"))


__all__ = ["load_builtin_skill", "parse_skill_markdown"]
