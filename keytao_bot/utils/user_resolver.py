"""User resolver utility for bot.

Finds keytao-next users by platform ID.

Configuration is read at CALL time through :mod:`keytao_bot.utils.http_client`
(lower-case driver attribute + ``os.getenv`` fallback). Reading it at import
time is wrong twice over: NoneBot lower-cases every config key it loads, and the
driver may not be configured yet when this module is first imported.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, Optional, Tuple

from nonebot.log import logger

from . import http_client


_BINDING_NOTICE_FACT: ContextVar[Optional[Tuple[str, str, Optional[bool]]]] = ContextVar(
    "binding_notice_fact", default=None,
)


def reset_binding_notice_fact() -> None:
    """Discard display-only account facts at the inbound turn boundary."""
    _BINDING_NOTICE_FACT.set(None)


def resolved_binding_for_notice(platform: str, platform_id: str) -> Optional[bool]:
    """Reuse only this task's latest result for the same actor's display notice."""
    fact = _BINDING_NOTICE_FACT.get()
    return fact[2] if fact is not None and fact[:2] == (platform, platform_id) else None


# Startup diagnostics. These call the shared helpers so the logged values are
# the same ones the request path will use.
logger.info(f"[user_resolver] KEYTAO_API_BASE: {http_client.get_keytao_url()}")
logger.info(f"[user_resolver] BOT_API_TOKEN loaded: {bool(http_client.get_bot_token())}")


async def _find_user_payload(
    platform: str,
    platform_id: str,
) -> Optional[Dict[str, Any]]:
    if not http_client.get_bot_token():
        logger.error("BOT_API_TOKEN not configured")
        return None

    try:
        data = await http_client.keytao_json(
            "POST",
            "/api/bot/user/find",
            json_body={
                "platform": platform,
                "platformId": platform_id,
            },
            timeout=10.0,
            allow_status=(404,),
            # Read-only lookup despite the POST verb: safe to replay.
            idempotent=True,
        )
    except Exception as error:
        logger.error(f"Find user error: {error}")
        return None
    return data


async def find_user_by_platform(platform: str, platform_id: str) -> Optional[Dict[str, Any]]:
    """Return user info, or ``None`` when unbound or lookup fails."""
    data = await _find_user_payload(platform, platform_id)
    if data is None:
        return None

    if data.get("found"):
        return data.get("user")
    return None


async def resolve_actor_binding(platform: str, platform_id: str) -> Optional[bool]:
    """Resolve account binding without treating lookup failure as unbound."""
    def remember(value: Optional[bool]) -> Optional[bool]:
        _BINDING_NOTICE_FACT.set((platform, platform_id, value))
        return value

    remember(None)
    if platform == "web-anon":
        return remember(False)
    if platform not in {"qq", "telegram"} or not platform_id:
        return remember(None)
    data = await _find_user_payload(platform, platform_id)
    if data is None:
        return remember(None)

    if data.get("found") is True and isinstance(data.get("user"), dict):
        return remember(True)
    if data.get("found") is False:
        return remember(False)
    return remember(None)


def get_not_bound_message() -> str:
    """Get not bound prompt message"""
    return (
        "未找到已绑定的键道账号。\n"
        "1. 登录：https://keytao.vercel.app\n"
        "2. 打开【我的资料】：https://keytao.vercel.app/profile\n"
        "3. 在【机器人账号绑定】生成并复制绑定码\n"
        "4. 使用 /bind 命令和页面生成的实际绑定码完成绑定"
    )
