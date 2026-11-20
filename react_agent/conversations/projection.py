"""面向聊天界面的会话消息投影。"""

from __future__ import annotations

from react_agent.conversations.contracts import ConversationMessage


def project_display_conversation_messages(
    messages: list[ConversationMessage] | None,
) -> list[ConversationMessage]:
    """Select displayable messages while preserving stable message identifiers."""
    projected: list[ConversationMessage] = []
    for message in messages or []:
        content = message.content.strip()
        if not content:
            continue
        if message.type in {"human", "user"}:
            projected.append(
                ConversationMessage(type="human", content=content, id=message.id)
            )
        elif message.type in {"ai", "assistant"} and not message.has_tool_calls:
            projected.append(
                ConversationMessage(type="ai", content=content, id=message.id)
            )
    return projected


def project_display_messages(
    messages: list[ConversationMessage] | None,
) -> list[dict[str, str]]:
    """只保留用户原文和无工具调用的最终助手文本。"""
    return [
        {
            "role": "user" if message.type == "human" else "assistant",
            "content": message.content,
        }
        for message in project_display_conversation_messages(messages)
    ]


__all__ = [
    "project_display_conversation_messages",
    "project_display_messages",
]
