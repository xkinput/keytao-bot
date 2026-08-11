"""
Web API plugin — exposes HTTP endpoints for the Live2D chat frontend.

Routes:
  POST /api/chat          — send a message, get AI reply
  POST /api/keytao/batches/review — run LLM-backed KeyTao batch review
  POST /api/keytao/pronunciation — infer a meaning-backed pronunciation
  DELETE /api/chat/history — clear session history

Auth: Bearer token via WEB_API_KEY, plus a signed identity for logged-in users.
"""
import os
from typing import Any, Dict, Optional

from fastapi import Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from nonebot import get_driver
from nonebot.log import logger

from .openai_chat import (
    MAX_HISTORY_MESSAGES,
    _clear_conversation_state,
    conversation_message_locks,
    current_history_generation,
    current_memory_context,
    current_memory_generation,
    draft_actor_message_locks,
    get_ai_response_core,
    handle_pending_message_core,
    history_store,
    memory_store,
    remember_conversation,
    schedule_memory_compaction,
)
from ..harness.conversation import ConversationAddress
from ..utils.keytao_batch_review import review_keytao_batch_with_llm
from ..utils.keytao_review import (
    SEMANTIC_PRONUNCIATION_GATE,
    infer_semantic_pronunciation,
)
from ..utils.memory_store import ChatMemoryContext
from ..utils.observability import (
    begin_turn_metrics,
    emit_turn_metrics,
    end_turn_metrics,
    mark_turn_outcome,
    record_history_messages,
    turn_metrics_emitted,
)
from ..utils.web_identity import (
    WebIdentityConfigError,
    WebIdentityVerificationError,
    verify_web_user_identity,
)
from ..utils.web_request_limits import RequestBodyLimitMiddleware

driver = get_driver()
config = driver.config
WEB_API_KEY: str = getattr(config, "web_api_key", None) or os.getenv("WEB_API_KEY", "")
WEB_IDENTITY_KEY: str = (
    getattr(config, "bot_identity_secret", None)
    or os.getenv("BOT_IDENTITY_SECRET", "")
)
WEB_CORS_ORIGINS: list[str] = (
    getattr(config, "web_cors_origins", None)
    or ["http://localhost:3000", "http://localhost:3001"]
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(min_length=1, max_length=128)
    user_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class HistoryClearRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1, max_length=128)
    user_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class KeyTaoBatchReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch: Dict[str, Any]
    local_review: Optional[Dict[str, Any]] = None
    focus_pr_id: Optional[int] = None


class KeyTaoPronunciationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word: str = Field(
        min_length=1,
        max_length=12,
        pattern=r"^[\u3400-\u9fff]+$",
    )


def _check_auth(authorization: Optional[str]) -> None:
    if not WEB_API_KEY:
        raise HTTPException(status_code=503, detail="Web API authentication is not configured")
    if authorization != f"Bearer {WEB_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")


def _check_web_user_identity(
    user_id: Optional[str],
    *,
    header_user_id: Optional[str],
    timestamp: Optional[str],
    nonce: Optional[str],
    signature: Optional[str],
    method: str,
    path: str,
    raw_body: bytes,
) -> Optional[str]:
    try:
        return verify_web_user_identity(
            WEB_IDENTITY_KEY,
            body_user_id=user_id,
            header_user_id=header_user_id,
            timestamp=timestamp,
            nonce=nonce,
            signature=signature,
            method=method,
            path=path,
            raw_body=raw_body,
        )
    except WebIdentityConfigError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except WebIdentityVerificationError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error


# Middleware and routes must be registered at import time, before the app starts.
try:
    from nonebot import get_app
    _app = get_app()

    _app.add_middleware(
        CORSMiddleware,
        allow_origins=WEB_CORS_ORIGINS,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Web-User-Id",
            "X-Web-User-Ts",
            "X-Web-User-Nonce",
            "X-Web-User-Sig",
        ],
    )
    _app.add_middleware(RequestBodyLimitMiddleware)

    @_app.post("/api/chat")
    async def chat(
        request: ChatRequest,
        http_request: Request,
        authorization: Optional[str] = Header(None),
        x_web_user_id: Optional[str] = Header(None),
        x_web_user_ts: Optional[str] = Header(None),
        x_web_user_nonce: Optional[str] = Header(None),
        x_web_user_sig: Optional[str] = Header(None),
    ) -> dict:
        _check_auth(authorization)
        raw_body = await http_request.body()
        verified_user_id = _check_web_user_identity(
            request.user_id,
            header_user_id=x_web_user_id,
            timestamp=x_web_user_ts,
            nonce=x_web_user_nonce,
            signature=x_web_user_sig,
            method="POST",
            path="/api/chat",
            raw_body=raw_body,
        )

        # Logged-in users identified by keytao user ID; anonymous by session UUID
        if verified_user_id:
            platform = "web"
            user_key = verified_user_id
        else:
            platform = "web-anon"
            user_key = request.session_id

        conv_key = ConversationAddress.private(platform, user_key)
        memory_context = ChatMemoryContext(
            platform=platform,
            user_id=user_key,
            space_type="private",
            space_id=user_key,
            speaker_name=user_key,
        )
        metrics_token = begin_turn_metrics(platform, "private")
        try:
            async with (
                conversation_message_locks.lock(conv_key),
                draft_actor_message_locks.lock(conv_key),
            ):
                history_token = current_history_generation.set(
                    history_store.capture_generation(conv_key)
                )
                memory_token = current_memory_generation.set(
                    memory_store.capture_generation(memory_context)
                )
                memory_context_token = current_memory_context.set(memory_context)
                try:
                    history = history_store.get_history(
                        conv_key,
                        limit=MAX_HISTORY_MESSAGES,
                    )
                    record_history_messages(len(history))
                    reply = await handle_pending_message_core(
                        request.message,
                        platform,
                        user_key,
                        conv_key,
                        history=history,
                        owner_label=user_key,
                    )
                    if reply is None:
                        reply = await get_ai_response_core(
                            message=request.message,
                            platform=platform,
                            user_id=user_key,
                            history=history,
                            memory_context=memory_context,
                        )
                    if reply:
                        remember_conversation(
                            conv_key,
                            memory_context,
                            request.message,
                            reply,
                        )
                        schedule_memory_compaction(memory_context)
                finally:
                    current_memory_context.reset(memory_context_token)
                    current_history_generation.reset(history_token)
                    current_memory_generation.reset(memory_token)
            emit_turn_metrics(logger)
            return {"reply": reply or "抱歉，AI 暂时无法响应，请稍后再试"}
        except BaseException:
            if not turn_metrics_emitted():
                mark_turn_outcome("error")
                emit_turn_metrics(logger)
            raise
        finally:
            end_turn_metrics(metrics_token)

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

    @_app.post("/api/keytao/pronunciation")
    async def keytao_pronunciation(
        request: KeyTaoPronunciationRequest,
        authorization: Optional[str] = Header(None),
        x_keytao_requester: Optional[str] = Header(None),
    ) -> dict:
        _check_auth(authorization)
        requester = (x_keytao_requester or "anonymous").strip()
        if not requester or len(requester) > 128 or not all(
            char.isalnum() or char in "-_.:" for char in requester
        ):
            raise HTTPException(status_code=400, detail="Invalid requester identity")

        decision = SEMANTIC_PRONUNCIATION_GATE.try_acquire(requester)
        if not decision.allowed:
            raise HTTPException(
                status_code=429,
                detail="Semantic pronunciation capacity exceeded",
                headers={"Retry-After": str(decision.retry_after_seconds)},
            )
        try:
            return await infer_semantic_pronunciation(request.word)
        finally:
            SEMANTIC_PRONUNCIATION_GATE.release()

    @_app.delete("/api/chat/history")
    async def clear_history(
        request: HistoryClearRequest,
        http_request: Request,
        authorization: Optional[str] = Header(None),
        x_web_user_id: Optional[str] = Header(None),
        x_web_user_ts: Optional[str] = Header(None),
        x_web_user_nonce: Optional[str] = Header(None),
        x_web_user_sig: Optional[str] = Header(None),
    ) -> dict:
        _check_auth(authorization)
        raw_body = await http_request.body()
        verified_user_id = _check_web_user_identity(
            request.user_id,
            header_user_id=x_web_user_id,
            timestamp=x_web_user_ts,
            nonce=x_web_user_nonce,
            signature=x_web_user_sig,
            method="DELETE",
            path="/api/chat/history",
            raw_body=raw_body,
        )

        platform = "web" if verified_user_id else "web-anon"
        user_key = verified_user_id if verified_user_id else request.session_id

        conv_key = ConversationAddress.private(platform, user_key)
        memory_context = ChatMemoryContext(
            platform=platform,
            user_id=user_key,
            space_type="private",
            space_id=user_key,
            speaker_name=user_key,
        )
        async with (
            conversation_message_locks.lock(conv_key),
            draft_actor_message_locks.lock(conv_key),
        ):
            deleted = history_store.count_history_rows(conv_key)
            await _clear_conversation_state(conv_key, memory_context)
        logger.info(f"web_api: cleared {deleted} messages for {platform}/{user_key[:8]}…")
        return {"success": True, "deleted": deleted}

    logger.info(
        f"web_api: routes registered  POST /api/chat  POST /api/keytao/batches/review  "
        f"POST /api/keytao/pronunciation  DELETE /api/chat/history  "
        f"(auth={'enabled' if WEB_API_KEY else 'disabled'})"
    )

except Exception as e:
    logger.error(f"web_api: failed to register routes: {e}")
