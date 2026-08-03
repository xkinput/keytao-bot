"""Daily vocabulary discovery.

Wires :mod:`keytao_bot.utils.word_discovery` onto the generic daily scheduler and
exposes a superuser-only manual trigger:

    word-discovery run   -> real round (writes the dictionary, reports to groups)
    word-discovery dry   -> classify only, nothing written, result sent back

This is a pure-Python path. It is deliberately NOT registered as a harness skill:
the pipeline writes to the dictionary on its own schedule and must not become a
tool the chat LLM can decide to call.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional, Set

from nonebot import get_driver, on_command
from nonebot.adapters import Bot, Event, Message
from nonebot.log import logger
from nonebot.matcher import Matcher
from nonebot.params import CommandArg
from nonebot.permission import SUPERUSER
from nonebot.rule import Rule, to_me

from keytao_bot.utils import group_notify, word_discovery
from keytao_bot.utils.daily_scheduler import (
    DEFAULT_TIMEZONE,
    DailyScheduler,
    clamp_hour,
    clamp_minute,
    resolve_timezone,
)

driver = get_driver()

TASK_NAME = "word_discovery"
LOG_PREFIX = "[word_discovery]"

# Hard ceiling for a whole round. Outbound sources, LLM extraction and per-word
# review each have their own budgets; this is the backstop that guarantees the
# scheduler loop can never be wedged by one bad day.
PIPELINE_TIMEOUT_SECONDS = 900.0

DEFAULT_HOUR = 9
DEFAULT_MINUTE = 0

# Give the QQ adapter time to connect before replaying queued digests; sending
# at on_startup would just fail again and burn a retry.
STARTUP_REPLAY_DELAY_SECONDS = 60.0

MODE_RUN = "run"
MODE_DRY = "dry"

_scheduler: Optional[DailyScheduler] = None
_background_tasks: Set[asyncio.Task] = set()


def _spawn_task(coroutine: Any) -> asyncio.Task:
    """Fire-and-forget with a strong reference (asyncio only keeps weak ones)."""
    task = asyncio.create_task(coroutine)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _schedule_hour() -> int:
    return clamp_hour(
        word_discovery._config_value("word_discovery_hour", "WORD_DISCOVERY_HOUR", None),
        DEFAULT_HOUR,
    )


def _schedule_minute() -> int:
    return clamp_minute(
        word_discovery._config_value("word_discovery_minute", "WORD_DISCOVERY_MINUTE", None),
        DEFAULT_MINUTE,
    )


def _schedule_timezone() -> Any:
    return resolve_timezone(
        word_discovery._config_value("word_discovery_timezone", "WORD_DISCOVERY_TIMEZONE", None)
        or DEFAULT_TIMEZONE
    )


async def run_discovery_round(*, dry_run: bool = False, notify: Optional[bool] = None) -> Any:
    """Run one round under the hard wall-clock cap."""
    try:
        return await asyncio.wait_for(
            word_discovery.run_word_discovery(dry_run=dry_run, notify=notify),
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"{LOG_PREFIX} pipeline exceeded {int(PIPELINE_TIMEOUT_SECONDS)}s hard limit, aborted"
        )
        raise


# ---------------------------------------------------------------------------
# Scheduler wiring
# ---------------------------------------------------------------------------


def _get_scheduler() -> DailyScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = DailyScheduler(
            TASK_NAME,
            lambda: run_discovery_round(dry_run=False),
            hour=_schedule_hour(),
            minute=_schedule_minute(),
            timezone=_schedule_timezone(),
            log_prefix=LOG_PREFIX,
        )
    return _scheduler


async def _replay_undelivered_digests() -> None:
    """Push out digests that never reached a group before the process died."""
    try:
        delivered = await word_discovery.flush_pending_notifications()
        if delivered:
            logger.info(f"{LOG_PREFIX} replayed {delivered} undelivered digest(s) on startup")
    except Exception as error:
        logger.error(f"{LOG_PREFIX} replaying undelivered digests failed: {error}")


@driver.on_startup
async def start_word_discovery_scheduler() -> None:
    if not word_discovery.discovery_enabled():
        logger.info(f"{LOG_PREFIX} disabled (WORD_DISCOVERY_ENABLED is not true)")
        return
    scheduler = _get_scheduler()
    scheduler.start()
    # Deferred: no adapter is connected yet at on_startup, so the first real
    # delivery attempt has to happen once the bot is actually online.
    _spawn_task(_replay_after_delay())


async def _replay_after_delay() -> None:
    await asyncio.sleep(STARTUP_REPLAY_DELAY_SECONDS)
    await _replay_undelivered_digests()


@driver.on_shutdown
async def stop_word_discovery_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        await _scheduler.stop()
        _scheduler = None


# ---------------------------------------------------------------------------
# Manual trigger (superusers only)
# ---------------------------------------------------------------------------


async def should_handle(bot: Bot, event: Event) -> bool:
    """Platform routing: private chats always, group chats only when addressed."""
    try:
        from nonebot.adapters.onebot.v11 import Bot as QQBot
        from nonebot.adapters.onebot.v11.event import (
            GroupMessageEvent as QQGroupMessageEvent,
            PrivateMessageEvent as QQPrivateMessageEvent,
        )
        from nonebot.adapters.telegram import Bot as TelegramBot
        from nonebot.adapters.telegram.event import (
            GroupMessageEvent as TelegramGroupMessageEvent,
            PrivateMessageEvent as TelegramPrivateMessageEvent,
        )

        if isinstance(bot, QQBot):
            if isinstance(event, QQPrivateMessageEvent):
                return True
            if isinstance(event, QQGroupMessageEvent):
                return await to_me()(bot, event, {})
            return await to_me()(bot, event, {})

        if isinstance(bot, TelegramBot):
            if isinstance(event, TelegramPrivateMessageEvent):
                return True
            if isinstance(event, TelegramGroupMessageEvent):
                return await to_me()(bot, event, {})
            return False

        return await to_me()(bot, event, {})
    except Exception as error:
        logger.error(f"{LOG_PREFIX} should_handle rule error: {error}")
        return False


def parse_command_mode(raw_arg: str) -> Optional[str]:
    """Map the command argument onto a run mode, or ``None`` when unrecognised."""
    text = str(raw_arg or "").strip().lower()
    if text in {"run", "go", "now"}:
        return MODE_RUN
    if text in {"dry", "dry-run", "dryrun", "preview"}:
        return MODE_DRY
    return None


USAGE_TEXT = (
    "用法：\n"
    "word-discovery run  立即跑一轮真实的每日词汇发现（会写库并发群报）\n"
    "word-discovery dry  只跑到分类为止，不写库，结果回给你"
)


word_discovery_cmd = on_command(
    "word-discovery",
    aliases={"词汇发现"},
    rule=Rule(should_handle),
    permission=SUPERUSER,
    priority=5,
    block=True,
)


@word_discovery_cmd.handle()
async def handle_word_discovery(bot: Bot, event: Event, matcher: Matcher, arg: Message = CommandArg()):
    mode = parse_command_mode(arg.extract_plain_text())
    if mode is None:
        await matcher.finish(USAGE_TEXT)

    # Advisory only - the real guarantee is the lock inside the pipeline, which
    # also stops the scheduled run from overlapping a manual one.
    if word_discovery.pipeline_busy():
        await matcher.finish("已经有一轮词汇发现在跑啦，等这轮跑完再来叭。")

    if mode == MODE_RUN:
        await matcher.send("好嘞，这就跑一轮每日词汇发现，完成后发到群里。")
        _spawn_task(_run_and_report(bot, event, dry_run=False))
        return

    await matcher.send("好嘞，试跑一轮（不写库），跑完把结果发给你。")
    _spawn_task(_run_and_report(bot, event, dry_run=True))


async def _reply(bot: Bot, event: Event, text: str) -> None:
    for chunk in group_notify.split_message(text):
        try:
            await bot.send(event, chunk)
        except Exception as error:
            logger.error(f"{LOG_PREFIX} failed to reply to trigger: {error}")
            return


async def _run_and_report(bot: Bot, event: Event, *, dry_run: bool) -> None:
    try:
        # A dry run never broadcasts: its whole point is a private preview.
        result = await run_discovery_round(dry_run=dry_run, notify=not dry_run)
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        await _reply(bot, event, f"这轮词汇发现超过 {int(PIPELINE_TIMEOUT_SECONDS)} 秒被中止了，请看日志。")
        return
    except Exception as error:
        logger.exception(f"{LOG_PREFIX} manual run failed")
        await _reply(bot, event, f"这轮词汇发现失败了：{error}")
        return

    if dry_run:
        await _reply(bot, event, result.report)
    else:
        await _reply(
            bot,
            event,
            f"跑完啦，自动入库 {len(result.auto_items)} 个、待推荐 {len(result.manual_items)} 个，详情已发群。",
        )
