"""内置 Skill 的注册、选择与解析。"""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType
from typing import Iterable, Mapping

from react_agent.skills.contracts import SkillDefinition, SkillSelection
from react_agent.skills.loader import load_builtin_skill
from react_agent.skills.selector import select_builtin_skill_name


class SkillRegistry:
    """进程内只读 Skill 注册表。"""

    def __init__(self, skills: Iterable[SkillDefinition]) -> None:
        indexed: dict[str, SkillDefinition] = {}
        for skill in skills:
            if skill.name in indexed:
                raise ValueError(f"Skill 名称重复：{skill.name}")
            indexed[skill.name] = skill
        self._skills: Mapping[str, SkillDefinition] = MappingProxyType(indexed)

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def select(self, text: str) -> SkillSelection | None:
        matched = select_builtin_skill_name(text)
        if matched is None:
            return None
        name, reason = matched
        skill = self.get(name)
        if skill is None:
            return None
        return SkillSelection(name=skill.name, version=skill.version, reason=reason)

    def resolve(self, selection: SkillSelection) -> SkillDefinition | None:
        """仅解析同名同版本 Skill，避免恢复旧状态时静默换工作流。"""
        skill = self.get(selection.name)
        if skill is None or skill.version != selection.version:
            return None
        return skill


@lru_cache(maxsize=1)
def build_builtin_skill_registry() -> SkillRegistry:
    """加载当前版本随应用发布的全部内置 Skill。"""
    return SkillRegistry((load_builtin_skill("industry_market_research"),))


__all__ = ["SkillRegistry", "build_builtin_skill_registry"]
