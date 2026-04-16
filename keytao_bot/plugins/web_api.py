"""
Web API plugin — exposes HTTP endpoints for the Live2D chat frontend.

Routes:
  POST /api/chat          — send a message, get AI reply
  DELETE /api/chat/history — clear session history

Auth: Bearer token via WEB_API_KEY env var (skip check if not set).
"""
import os
from typing import Optional

from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nonebot import get_driver
from nonebot.log import logger

from .openai_chat import get_ai_response_core, conversation_states, MAX_HISTORY_MESSAGES
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


class HistoryClearRequest(BaseModel):
    session_id: str


def _check_auth(authorization: Optional[str]) -> None:
    if WEB_API_KEY and authorization != f"Bearer {WEB_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@driver.on_startup
async def _setup_web_api() -> None:
    try:
        from nonebot import get_app
        app = get_app()
    except Exception as e:
        logger.error(f"web_api: failed to get FastAPI app: {e}")
        return

    app.add_middleware(
        CORSMiddleware,
        allow_origins=WEB_CORS_ORIGINS,
        allow_origin_regex=r"https?://.*",  # allow all origins in dev
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.post("/api/chat")
    async def chat(
        request: ChatRequest,
        authorization: Optional[str] = None,
    ) -> dict:
        from fastapi import Request as FastAPIRequest
        _check_auth(authorization)

        store = get_history_store()
        history = store.get_history("web", request.session_id, limit=MAX_HISTORY_MESSAGES)

        reply = await get_ai_response_core(
            message=request.message,
            platform="web",
            user_id=request.session_id,
            history=history,
        )

        if reply:
            store.add_conversation_round("web", request.session_id, request.message, reply)

        return {"reply": reply or "抱歉，AI 暂时无法响应，请稍后再试"}

    @app.delete("/api/chat/history")
    async def clear_history(
        request: HistoryClearRequest,
        authorization: Optional[str] = None,
    ) -> dict:
        _check_auth(authorization)

        store = get_history_store()
        deleted = store.clear_history("web", request.session_id)
        conversation_states.pop(("web", request.session_id), None)
        logger.info(f"web_api: cleared {deleted} messages for session {request.session_id[:8]}…")
        return {"success": True, "deleted": deleted}

    logger.info(
        f"web_api: routes registered  POST /api/chat  DELETE /api/chat/history  "
        f"(auth={'enabled' if WEB_API_KEY else 'disabled'})"
    )
