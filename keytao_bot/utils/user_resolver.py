"""
User resolver utility for bot
Finds users by platform ID
"""
import httpx
from nonebot import get_driver
from nonebot.log import logger
from typing import Optional, Dict, Any

# Get configuration from NoneBot
driver = get_driver()
config = driver.config
KEYTAO_API_BASE = getattr(config, "KEYTAO_API_BASE", "https://keytao.vercel.app")
BOT_API_TOKEN = getattr(config, "BOT_API_TOKEN", None)

# Debug log
logger.info(f"[user_resolver] KEYTAO_API_BASE: {KEYTAO_API_BASE}")
logger.info(f"[user_resolver] BOT_API_TOKEN loaded: {bool(BOT_API_TOKEN)}")


async def find_user_by_platform(platform: str, platform_id: str) -> Optional[Dict[str, Any]]:
    """
    Find user by platform ID
    
    Args:
        platform: 'qq' or 'telegram'
        platform_id: Platform user ID
        
    Returns:
        User info dict or None
    """
    if not BOT_API_TOKEN:
        logger.error("BOT_API_TOKEN not configured")
        return None
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{KEYTAO_API_BASE}/api/bot/user/find",
                headers={
                    "X-Bot-Token": BOT_API_TOKEN,
                    "Content-Type": "application/json"
                },
                json={
                    "platform": platform,
                    "platformId": platform_id
                },
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("found"):
                    return data.get("user")
            
            return None

    except Exception as e:
        logger.error(f"Find user error: {e}")
        return None


def get_not_bound_message() -> str:
    """Get not bound prompt message"""
    return (
        "❌ 未找到你在键道平台的账号 qwq\n\n"
        "使用机器人创建词条需要先绑定账号：\n"
        "1. 访问 keytao.vercel.app 注册账号\n"
        "2. 登录后进入【我的资料】页面\n"
        "3. 在【机器人账号绑定】部分生成绑定码\n"
        "4. 在这里发送：/bind [绑定码]\n\n"
        "绑定后就可以使用啦～ owo"
    )
