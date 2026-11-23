"""
QQ bot disconnect watchdog
Sends a Telegram notification when the QQ (OneBot) bot goes offline unexpectedly.
Normal restarts that reconnect within 90s are silently ignored — unless the
QQ account fails login verification after reconnect (e.g. kicked offline).

Also runs a periodic heartbeat check to catch cases where NapCat keeps the
WebSocket alive but the QQ account is actually kicked/invalid.
"""
import asyncio
import os
import httpx
from datetime import datetime

from nonebot import get_driver, get_bots
from nonebot.adapters.onebot.v11 import Bot as QQBot
from nonebot.log import logger

driver = get_driver()
config = driver.config

TELEGRAM_TOKEN = None
NOTIFY_CHAT_ID = getattr(config, "notify_tg_chat_id", None)
RECONNECT_GRACE = 90   # seconds to wait before treating no-reconnect as real offline
LOGIN_VERIFY_DELAY = 5  # seconds after connect before checking status
HEARTBEAT_INTERVAL = 60  # seconds between status checks and failed alert retries
STATUS_TIMEOUT = 10

_tg_bots = getattr(config, "telegram_bots", None)
if _tg_bots:
    try:
        import json
        bots = json.loads(_tg_bots) if isinstance(_tg_bots, str) else _tg_bots
        if bots:
            TELEGRAM_TOKEN = bots[0].get("token")
    except (ValueError, TypeError, AttributeError, IndexError, KeyError) as error:
        logger.warning(f"[watchdog] cannot read telegram_bots config, TG alerts disabled: {error}")

# pending tasks: bot_id -> asyncio.Task
_pending: dict[str, asyncio.Task] = {}
# Delivery is tracked per alert type until the account is confirmed online.
_reported: dict[str, set[str]] = {}
_checking: set[str] = set()
# strong references to fire-and-forget tasks, so the event loop cannot collect
# them mid-flight (asyncio only keeps weak references to running tasks)
_background_tasks: set[asyncio.Task] = set()


def _spawn_task(coro) -> asyncio.Task:
    """Create a background task and keep a strong reference until it finishes."""
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _send_tg(text: str) -> bool:
    if not TELEGRAM_TOKEN or not NOTIFY_CHAT_ID:
        logger.warning("[watchdog] TG token or chat_id not configured, cannot send notification")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json={"chat_id": NOTIFY_CHAT_ID, "text": text})
            if resp.status_code == 200 and resp.json().get("ok") is True:
                return True
            logger.warning(f"[watchdog] TG send failed: HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"[watchdog] TG send error: {type(e).__name__}")
    return False


async def _notify_once(bot_id: str, kind: str, title: str, detail: str, ts: str) -> bool:
    reported = _reported.setdefault(bot_id, set())
    if kind in reported:
        return True
    if not await _send_tg(f"⚠️ {title}\n账号：{bot_id}\n时间：{ts}\n\n{detail}"):
        return False
    reported.add(kind)
    logger.info(f"[watchdog] TG alert accepted for QQ bot {bot_id}: {kind}")
    return True


async def _delayed_notify(bot_id: str, ts: str):
    try:
        await asyncio.sleep(RECONNECT_GRACE)
        logger.warning(f"[watchdog] QQ bot {bot_id} still disconnected after {RECONNECT_GRACE}s, notifying")
        while not await _notify_once(
            bot_id, "transport_disconnected", "QQ Bot 与 NapCat 连接中断",
            f"连接超过 {RECONNECT_GRACE} 秒未恢复，暂时无法确认 QQ 账号在线状态。", ts,
        ):
            await asyncio.sleep(HEARTBEAT_INTERVAL)
    finally:
        if _pending.get(bot_id) is asyncio.current_task():
            _pending.pop(bot_id, None)


async def _verify_login_after_reconnect(bot: QQBot, bot_id: str):
    await asyncio.sleep(LOGIN_VERIFY_DELAY)
    await _check_qq_status(bot, bot_id)


async def _check_qq_status(bot: QQBot, bot_id: str):
    if bot_id in _checking or bot_id in _pending or get_bots().get(bot_id) is not bot:
        return
    _checking.add(bot_id)
    try:
        try:
            status = await asyncio.wait_for(bot.get_status(), timeout=STATUS_TIMEOUT)
            online = status.get("online") if isinstance(status, dict) else None
            if not isinstance(online, bool):
                raise ValueError("invalid online status")
        except Exception as error:
            logger.warning(f"[watchdog] QQ status check failed for {bot_id}: {type(error).__name__}")
            online = None

        if bot_id in _pending or get_bots().get(bot_id) is not bot:
            return
        if online is True:
            if _reported.pop(bot_id, None):
                logger.info(f"[watchdog] QQ bot {bot_id} recovered (heartbeat)")
            return

        ts = datetime.now().strftime("%H:%M:%S")
        if online is False:
            await _notify_once(
                bot_id, "account_offline", "QQ 账号已掉线",
                "NapCat 已确认账号离线，请在 NapCat WebUI 重新登录。", ts,
            )
        else:
            await _notify_once(
                bot_id, "status_unknown", "QQ 在线状态暂不可确认",
                "NapCat 未返回可用的在线状态，请检查账号和连接。", ts,
            )
    finally:
        _checking.discard(bot_id)


async def _heartbeat_loop():
    """
    Periodically check all connected QQ bots to detect cases where NapCat
    keeps the WebSocket alive but the QQ account is actually kicked/invalid.
    """
    while True:
        await asyncio.sleep(HEARTBEAT_INTERVAL)
        bots = get_bots()
        for bot_id, bot in list(bots.items()):
            if isinstance(bot, QQBot):
                await _check_qq_status(bot, bot_id)


@driver.on_startup
async def start_heartbeat():
    _spawn_task(_heartbeat_loop())
    logger.info(f"[watchdog] heartbeat started, interval={HEARTBEAT_INTERVAL}s")
    account = os.getenv("ACCOUNT", "").strip()
    if account.isascii() and account.isdecimal() and not isinstance(get_bots().get(account), QQBot):
        _pending[account] = _spawn_task(_delayed_notify(account, datetime.now().strftime("%H:%M:%S")))


@driver.on_shutdown
async def stop_heartbeat():
    tasks = tuple(_background_tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    _pending.clear()


@driver.on_bot_disconnect
async def on_qq_disconnect(bot: QQBot):
    if not isinstance(bot, QQBot):
        return
    bot_id = bot.self_id
    ts = datetime.now().strftime("%H:%M:%S")
    logger.warning(f"[watchdog] QQ bot {bot_id} disconnected, waiting {RECONNECT_GRACE}s before notify")
    # A previous grace-period task must be cancelled before it is replaced,
    # otherwise it keeps running and can fire a stale alert.
    previous = _pending.pop(bot_id, None)
    if previous and not previous.done():
        previous.cancel()
        logger.debug(f"[watchdog] cancelled stale pending notify task for {bot_id}")
    _pending[bot_id] = _spawn_task(_delayed_notify(bot_id, ts))


@driver.on_bot_connect
async def on_qq_connect(bot: QQBot):
    if not isinstance(bot, QQBot):
        return
    bot_id = bot.self_id
    task = _pending.pop(bot_id, None)
    if task:
        task.cancel()
        logger.info(f"[watchdog] QQ bot {bot_id} reconnected within grace period, verifying login...")
    _spawn_task(_verify_login_after_reconnect(bot, bot_id))
