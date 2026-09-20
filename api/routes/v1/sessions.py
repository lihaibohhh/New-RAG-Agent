"""
api/routes/v1/sessions.py — 会话 CRUD（Phase 3）

IDOR 防护：
  path 里的 session_id 在服务端套 bucket_key 命名空间，构造
  thread_id = user:{bucket_key}:{session_id}。
  不同 API Key 的 bucket_key 不同，查别人的 session_id 得到的是另一个
  thread_id，checkpointer 找不到 → 统一返回 404（不泄露存在性）。

端点：
  GET  /api/v1/sessions/{session_id}/history   分页消息历史
  DELETE /api/v1/sessions/{session_id}         清除会话（204）
  GET  /api/v1/sessions                        枚举（501，checkpointer 不支持）
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import JSONResponse

from api.dependencies import get_conversation_service
from api.errors import AppError
from api.models import MessageItem, SessionHistoryResponse
from api.security import get_rate_limit_key, require_api_key
from react_agent.conversations import ConversationDeleteStatus, ConversationService

_logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["sessions v1"])


# ── GET /sessions/{session_id}/history ───────────────────────────────────────

@router.get(
    "/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="获取会话消息历史（分页）",
)
async def get_session_history(
    session_id: str,
    request: Request,
    page: int = Query(1, ge=1, description="页码（从 1 开始）"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数（1–100）"),
    conversations: ConversationService = Depends(get_conversation_service),
    api_key: str = Depends(require_api_key),
) -> SessionHistoryResponse:
    bucket_key = get_rate_limit_key(api_key, request)
    thread_id = f"user:{bucket_key}:{session_id}"

    messages = await conversations.get_history(thread_id)
    if messages is None:
        raise AppError(status=404, title="Not Found", detail=f"会话 {session_id} 不存在。")

    total = len(messages)
    offset = (page - 1) * page_size
    page_msgs = messages[offset: offset + page_size]

    return SessionHistoryResponse(
        session_id=session_id,
        messages=[MessageItem(type=m.type, content=m.content, id=m.id) for m in page_msgs],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(offset + page_size) < total,
    )


# ── DELETE /sessions/{session_id} ────────────────────────────────────────────

@router.delete(
    "/{session_id}",
    status_code=204,
    summary="删除会话（清除所有 checkpoint）",
    responses={
        204: {"description": "删除成功"},
        404: {"description": "会话不存在或不属于当前调用方"},
        501: {"description": "当前 checkpointer 后端不支持删除"},
    },
)
async def delete_session(
    session_id: str,
    request: Request,
    conversations: ConversationService = Depends(get_conversation_service),
    api_key: str = Depends(require_api_key),
) -> Response:
    bucket_key = get_rate_limit_key(api_key, request)
    thread_id = f"user:{bucket_key}:{session_id}"

    result = await conversations.delete(thread_id)

    if result is ConversationDeleteStatus.NOT_FOUND:
        # thread 不存在（含 IDOR 场景：别人的 session_id 在本 bucket_key 下找不到）
        raise AppError(status=404, title="Not Found", detail=f"会话 {session_id} 不存在。")

    if result is ConversationDeleteStatus.UNSUPPORTED:
        # 后端不支持删除
        raise AppError(
            status=501,
            title="Not Implemented",
            detail="当前 checkpointer 后端不支持删除会话，请联系管理员。",
        )

    _logger.info("session_deleted | session=%s bucket_key_prefix=%s", session_id, bucket_key[:8])
    return Response(status_code=204)


# ── GET /sessions（枚举）────────────────────────────────────────────────────

@router.get(
    "",
    status_code=501,
    summary="列举会话（不支持）",
    description="LangGraph checkpointer 标准 API 不支持跨 thread 枚举，固定返回 501。",
    include_in_schema=True,
)
async def list_sessions(
    _api_key: str = Depends(require_api_key),
) -> JSONResponse:
    raise AppError(
        status=501,
        title="Not Implemented",
        detail="会话枚举功能暂不支持：checkpointer 后端无法按 API Key 维度列举所有 thread。",
    )
