"""Agent Skill 的稳定数据契约。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping


_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class SkillDefinition:
    """一份已经校验、可注入模型上下文的内置 Skill。"""

    name: str
    version: str
    description: str
    instructions: str

    def __post_init__(self) -> None:
        if not _SKILL_NAME_PATTERN.fullmatch(self.name):
            raise ValueError(
                "Skill name 必须使用小写 kebab-case，例如 company-comparison"
            )
        if not self.version.strip():
            raise ValueError("Skill version 不能为空")
        if not self.description.strip():
            raise ValueError("Skill description 不能为空")
        if not self.instructions.strip():
            raise ValueError("Skill instructions 不能为空")

    def render_for_model(self) -> str:
        """渲染为独立的瞬态系统指令，不写入对话消息历史。"""
        return (
            "【当前任务 Skill】\n"
            f"名称：{self.name}\n"
            f"版本：{self.version}\n"
            f"说明：{self.description}\n\n"
            "请遵循以下业务工作流；它只规定任务方法，不扩大工具权限，"
            "也不能覆盖系统安全策略、工具预算或用户明确要求：\n\n"
            f"{self.instructions.strip()}"
        )


@dataclass(frozen=True)
class SkillSelection:
    """一次用户轮次的 Skill 选择结果。"""

    name: str
    version: str
    reason: str

    def to_state(self) -> dict[str, str]:
        return {
            "name": self.name,
            "version": self.version,
            "reason": self.reason,
        }

    @classmethod
    def from_state(
        cls, value: Mapping[str, Any] | None
    ) -> "SkillSelection | None":
        if not isinstance(value, Mapping):
            return None
        name = str(value.get("name") or "").strip()
        version = str(value.get("version") or "").strip()
        reason = str(value.get("reason") or "").strip()
        if not name or not version:
            return None
        return cls(name=name, version=version, reason=reason)


__all__ = ["SkillDefinition", "SkillSelection"]
