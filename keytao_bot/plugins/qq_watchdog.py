"""
QQ bot disconnect watchdog
Sends a Telegram notification when the QQ (OneBot) bot goes offline unexpectedly.
Restarts (where the bot reconnects within 90s) are silently ignored.
"""
import asyncio
import httpx
from datetime import datetime

from nonebot import get_driver
from nonebot.adapters.onebot.v11 import Bot as QQBot
from nonebot.log import logger

driver = get_driver()
config = driver.config

TELEGRAM_TOKEN = None
NOTIFY_CHAT_ID = getattr(config, "notify_tg_chat_id", None)
RECONNECT_GRACE = 90  # seconds to wait before treating disconnect as real offline

_tg_bots = getattr(config, "telegram_bots", None)
if _tg_bots:
    try:
        import json
        bots = json.loads(_tg_bots) if isinstance(_tg_bots, str) else _tg_bots
        if bots:
            TELEGRAM_TOKEN = bots[0].get("token")
    except Exception:
        pass

# pending tasks: bot_id -> asyncio.Task
_pending: dict[str, asyncio.Task] = {}


async def _send_tg(text: str):
    if not TELEGRAM_TOKEN or not NOTIFY_CHAT_ID:
        logger.warning("[watchdog] TG token or chat_id not configured, cannot send notification")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json={"chat_id": NOTIFY_CHAT_ID, "text": text})
            if resp.status_code != 200:
                logger.warning(f"[watchdog] TG send failed: {resp.text}")
    except Exception as e:
        logger.error(f"[watchdog] TG send error: {e}")


async def _delayed_notify(bot_id: str, ts: str):
    await asyncio.sleep(RECONNECT_GRACE)
    # still offline after grace period — real disconnect
    msg = f"⚠️ QQ bot 掉线了！\n账号：{bot_id}\n时间：{ts}\n\nNapCat 可能需要重新登录。"
    logger.warning(f"[watchdog] QQ bot {bot_id} still offline after {RECONNECT_GRACE}s, notifying")
    await _send_tg(msg)
    _pending.pop(bot_id, None)


@driver.on_bot_disconnect
async def on_qq_disconnect(bot: QQBot):
    bot_id = bot.self_id
    ts = datetime.now().strftime("%H:%M:%S")
    logger.warning(f"[watchdog] QQ bot {bot_id} disconnected, waiting {RECONNECT_GRACE}s before notify")
    task = asyncio.create_task(_delayed_notify(bot_id, ts))
    _pending[bot_id] = task


@driver.on_bot_connect
async def on_qq_connect(bot: QQBot):
    bot_id = bot.self_id
    task = _pending.pop(bot_id, None)
    if task:
        task.cancel()
        logger.info(f"[watchdog] QQ bot {bot_id} reconnected within grace period, notification cancelled")
