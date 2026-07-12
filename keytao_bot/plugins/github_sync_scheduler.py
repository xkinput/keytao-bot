"""
Scheduled KeyTao GitHub dictionary sync checks.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Iterable
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from nonebot import get_bots, get_driver
from nonebot.adapters.onebot.v11 import Bot as QQBot
from nonebot.log import logger

from keytao_bot.utils.github_sync_notification import build_github_sync_notification

driver = get_driver()
config = driver.config

DEFAULT_SYNC_THRESHOLD = 10
DEFAULT_SYNC_HOUR = 10
DEFAULT_SYNC_MINUTE = 0
DEFAULT_SYNC_TIMEZONE = "Asia/Shanghai"
DEFAULT_SYNC_WEEKDAYS = {2, 6}  # Wednesday, Sunday

_scheduler_task: asyncio.Task | None = None
_run_lock = asyncio.Lock()


def _config_value(name: str, default: Any = None) -> Any:
    lower_name = name.lower()
    upper_name = name.upper()
    if hasattr(config, lower_name):
        return getattr(config, lower_name)
    if hasattr(config, upper_name):
        return getattr(config, upper_name)
    return default


def _parse_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _parse_int(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    if minimum is not None and parsed < minimum:
        return default
    if maximum is not None and parsed > maximum:
        return default
    return parsed


def _parse_group_ids(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        items: Iterable[Any] = value
    else:
        items = str(value).replace("，", ",").split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def _get_schedule_timezone() -> ZoneInfo:
    timezone_name = str(_config_value("keytao_sync_timezone", DEFAULT_SYNC_TIMEZONE)).strip() or DEFAULT_SYNC_TIMEZONE
    try:
        return ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        logger.warning(
            f"[github_sync_scheduler] invalid KEYTAO_SYNC_TIMEZONE={timezone_name}, "
            f"fall back to {DEFAULT_SYNC_TIMEZONE}"
        )
        return ZoneInfo(DEFAULT_SYNC_TIMEZONE)


def _seconds_until_next_run(now: datetime, hour: int, minute: int) -> float:
    target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    for offset in range(8):
        candidate = target_time + timedelta(days=offset)
        if candidate.weekday() in DEFAULT_SYNC_WEEKDAYS and candidate > now:
            return max((candidate - now).total_seconds(), 0)
    return 24 * 60 * 60


def _get_keytao_api_base() -> str:
    return str(_config_value("keytao_api_base", "https://keytao.vercel.app")).rstrip("/")


def _get_bot_token() -> str | None:
    token = _config_value("bot_api_token")
    return str(token).strip() if token else None


async def _call_auto_sync_endpoint(threshold: int) -> dict[str, Any]:
    bot_token = _get_bot_token()
    if not bot_token:
        raise RuntimeError("BOT_API_TOKEN is not configured")

    url = f"{_get_keytao_api_base()}/api/bot/sync-to-github/auto"
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            url,
            headers={
                "X-Bot-Token": bot_token,
                "Content-Type": "application/json",
            },
            json={"threshold": threshold},
        )

    try:
        data = response.json()
    except ValueError:
        data = {"message": response.text}

    if response.status_code >= 400:
        data.setdefault("success", False)
        data.setdefault("triggered", False)
        data["httpStatus"] = response.status_code
        data["message"] = data.get("message") or data.get("error") or response.text

    return data


async def _send_group_notification(text: str) -> None:
    group_ids = _parse_group_ids(_config_value("keytao_sync_notify_group_ids"))
    if not group_ids:
        logger.warning("[github_sync_scheduler] KEYTAO_SYNC_NOTIFY_GROUP_IDS is empty, skip group notification")
        return

    qq_bots = [bot for bot in get_bots().values() if isinstance(bot, QQBot)]
    if not qq_bots:
        logger.warning("[github_sync_scheduler] no QQ bot connected, cannot send group notification")
        return

    bot = qq_bots[0]
    for group_id in group_ids:
        try:
            await bot.send_group_msg(group_id=int(group_id), message=text)
            logger.info(f"[github_sync_scheduler] sent sync notification to QQ group {group_id}")
        except Exception as exc:
            logger.error(f"[github_sync_scheduler] failed to notify QQ group {group_id}: {exc}")


def _build_notification(data: dict[str, Any]) -> str:
    return build_github_sync_notification(data)


def _build_failure_notification(data: dict[str, Any]) -> str:
    pending_count = data.get("pendingSyncBatches")
    released_count = data.get("releasedFailedBatches")
    message = data.get("message") or data.get("error") or "未知错误"
    http_status = data.get("httpStatus")
    lines = [
        "本喵尝试进行 GitHub 词库自动同步，但这次失败了。",
        f"原因：{message}",
    ]
    if http_status:
        lines.append(f"接口状态：{http_status}")
    if pending_count is not None:
        lines.append(f"当前待同步批次：{pending_count} 个。")
    if released_count:
        lines.append(f"已释放上次失败卡住的批次：{released_count} 个。")
    lines.append("请管理员检查同步任务记录。")
    return "\n".join(lines)


async def run_github_sync_check_once() -> dict[str, Any] | None:
    async with _run_lock:
        threshold = _parse_int(
            _config_value("keytao_sync_threshold"),
            DEFAULT_SYNC_THRESHOLD,
            minimum=1,
            maximum=1000,
        )
        logger.info(f"[github_sync_scheduler] checking GitHub sync, threshold={threshold}")
        data = await _call_auto_sync_endpoint(threshold)

        if data.get("triggered") and data.get("prUrl"):
            await _send_group_notification(_build_notification(data))
        elif data.get("success") is False or data.get("httpStatus"):
            logger.error(
                "[github_sync_scheduler] sync check failed: "
                f"status={data.get('httpStatus')}, message={data.get('message')}"
            )
            await _send_group_notification(_build_failure_notification(data))
        else:
            logger.info(
                "[github_sync_scheduler] sync not triggered: "
                f"pending={data.get('pendingSyncBatches')}, reason={data.get('skippedReason')}, message={data.get('message')}"
            )

        return data


async def _scheduler_loop() -> None:
    hour = _parse_int(_config_value("keytao_sync_check_hour"), DEFAULT_SYNC_HOUR, minimum=0, maximum=23)
    minute = _parse_int(_config_value("keytao_sync_check_minute"), DEFAULT_SYNC_MINUTE, minimum=0, maximum=59)
    timezone = _get_schedule_timezone()
    logger.info(
        "[github_sync_scheduler] scheduler started, runs on "
        f"Wednesday/Sunday {hour:02d}:{minute:02d} {timezone.key}"
    )

    while True:
        delay = _seconds_until_next_run(datetime.now(timezone), hour, minute)
        logger.info(f"[github_sync_scheduler] next check in {int(delay)} seconds")
        await asyncio.sleep(delay)
        try:
            await run_github_sync_check_once()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(f"[github_sync_scheduler] scheduled sync check failed: {exc}")
        await asyncio.sleep(60)


@driver.on_startup
async def start_github_sync_scheduler() -> None:
    global _scheduler_task
    enabled = _parse_bool(_config_value("keytao_sync_schedule_enabled"), default=True)
    if not enabled:
        logger.info("[github_sync_scheduler] scheduler disabled")
        return
    if _scheduler_task and not _scheduler_task.done():
        return
    _scheduler_task = asyncio.create_task(_scheduler_loop())


@driver.on_shutdown
async def stop_github_sync_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task:
        _scheduler_task.cancel()
        _scheduler_task = None
