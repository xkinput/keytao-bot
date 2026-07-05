"""
Web API plugin — exposes HTTP endpoints for the Live2D chat frontend.

Routes:
  POST /api/chat          — send a message, get AI reply
  POST /api/keytao/batches/review — run LLM-backed KeyTao batch review
  DELETE /api/chat/history — clear session history

Auth: Bearer token via WEB_API_KEY env var (skip check if not set).
"""
import os
from typing import Any, Dict, Optional

from fastapi import Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nonebot import get_driver
from nonebot.log import logger

from .openai_chat import get_ai_response_core, conversation_state_store, MAX_HISTORY_MESSAGES
from ..utils.keytao_batch_review import review_keytao_batch_with_llm
from ..utils.history_store import get_history_store

driver = get_driver()
config = driver.config
WEB_API_KEY: str = getattr(config, "web_api_key", None) or os.getenv("WEB_API_KEY", "")
WEB_CORS_ORIGINS: list[str] = (
    getattr(config, "web_cors_origins", None)
    or ["http://localhost:3000", "http://localhost:3001"]
)


class ChatRequest(BaseModel):
    message: str
    session_id: str  # UUID stored in browser localStorage
    user_id: Optional[str] = None  # keytao-next user ID, injected server-side after JWT verification


class HistoryClearRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None


class KeyTaoBatchReviewRequest(BaseModel):
    batch: Dict[str, Any]
    local_review: Optional[Dict[str, Any]] = None
    focus_pr_id: Optional[int] = None


def _check_auth(authorization: Optional[str]) -> None:
    if WEB_API_KEY and authorization != f"Bearer {WEB_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# Middleware and routes must be registered at import time, before the app starts.
try:
    from nonebot import get_app
    _app = get_app()

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=WEB_CORS_ORIGINS,
        allow_origin_regex=r"https?://.*",
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @_app.post("/api/chat")
    async def chat(
        request: ChatRequest,
        authorization: Optional[str] = Header(None),
    ) -> dict:
        _check_auth(authorization)

        # Logged-in users identified by keytao user ID; anonymous by session UUID
        if request.user_id:
            platform = "web"
            user_key = request.user_id
        else:
            platform = "web-anon"
            user_key = request.session_id

        store = get_history_store()
        history = store.get_history(platform, user_key, limit=MAX_HISTORY_MESSAGES)

        reply = await get_ai_response_core(
            message=request.message,
            platform=platform,
            user_id=user_key,
            history=history,
        )

        if reply:
            store.add_conversation_round(platform, user_key, request.message, reply)

        return {"reply": reply or "抱歉，AI 暂时无法响应，请稍后再试"}

    @_app.post("/api/keytao/batches/review")
    async def keytao_batch_review(
        request: KeyTaoBatchReviewRequest,
        authorization: Optional[str] = Header(None),
    ) -> dict:
        _check_auth(authorization)
        result = await review_keytao_batch_with_llm(
            batch=request.batch,
            local_review=request.local_review,
            focus_pr_id=request.focus_pr_id,
        )
        if not result.get("success"):
            raise HTTPException(status_code=502, detail=result.get("message") or "喵喵复审失败")
        return result

    @_app.delete("/api/chat/history")
    async def clear_history(
        request: HistoryClearRequest,
        authorization: Optional[str] = Header(None),
    ) -> dict:
        _check_auth(authorization)

        platform = "web" if request.user_id else "web-anon"
        user_key = request.user_id if request.user_id else request.session_id

        store = get_history_store()
        deleted = store.clear_history(platform, user_key)
        conversation_state_store.delete((platform, user_key))
        logger.info(f"web_api: cleared {deleted} messages for {platform}/{user_key[:8]}…")
        return {"success": True, "deleted": deleted}

    logger.info(
        f"web_api: routes registered  POST /api/chat  POST /api/keytao/batches/review  DELETE /api/chat/history  "
        f"(auth={'enabled' if WEB_API_KEY else 'disabled'})"
    )

except Exception as e:
    logger.error(f"web_api: failed to register routes: {e}")
