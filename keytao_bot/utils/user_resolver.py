"""User resolver utility for bot.

Finds keytao-next users by platform ID.

Configuration is read at CALL time through :mod:`keytao_bot.utils.http_client`
(lower-case driver attribute + ``os.getenv`` fallback). Reading it at import
time is wrong twice over: NoneBot lower-cases every config key it loads, and the
driver may not be configured yet when this module is first imported.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from nonebot.log import logger

from . import http_client


# Startup diagnostics. These call the shared helpers so the logged values are
# the same ones the request path will use.
logger.info(f"[user_resolver] KEYTAO_API_BASE: {http_client.get_keytao_url()}")
logger.info(f"[user_resolver] BOT_API_TOKEN loaded: {bool(http_client.get_bot_token())}")


async def find_user_by_platform(platform: str, platform_id: str) -> Optional[Dict[str, Any]]:
    """Find user by platform ID.

    Args:
        platform: 'qq' or 'telegram'
        platform_id: Platform user ID

    Returns:
        User info dict, or ``None`` when the user is unknown or the lookup failed.
    """
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
            # Read-only lookup despite the POST verb: safe to replay.
            idempotent=True,
        )
    except Exception as error:
        logger.error(f"Find user error: {error}")
        return None

    if data.get("found"):
        return data.get("user")
    return None


def get_not_bound_message() -> str:
    """Get not bound prompt message"""
    return (
        "❌ 未找到你在键道平台的账号 qwq\n\n"
        "使用机器人创建词条需要先绑定键道平台账号：\n"
        "1. 访问 https://keytao.vercel.app 注册或登录账号\n"
        "2. 登录后进入【我的资料】页面\n"
        "3. 在【机器人账号绑定】部分生成绑定码\n"
        "4. 在这里发送：/bind [绑定码]\n\n"
        "绑定后就可以使用啦～ owo"
    )
