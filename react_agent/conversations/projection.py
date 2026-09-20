"""面向聊天界面的会话消息投影。"""

from __future__ import annotations

from react_agent.conversations.contracts import ConversationMessage


def project_display_messages(
    messages: list[ConversationMessage] | None,
) -> list[dict[str, str]]:
    """只保留用户原文和无工具调用的最终助手文本。"""
    projected: list[dict[str, str]] = []
    for message in messages or []:
        content = message.content
        if not content.strip():
            continue
        if message.type in {"human", "user"}:
            projected.append({"role": "user", "content": content})
        elif message.type in {"ai", "assistant"} and not message.has_tool_calls:
            projected.append({"role": "assistant", "content": content})
    return projected


__all__ = ["project_display_messages"]
