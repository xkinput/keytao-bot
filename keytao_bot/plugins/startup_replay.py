"""Replay recent unanswered OneBot group messages after reconnect."""

from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from nonebot import get_driver
from nonebot.adapters import Bot
from nonebot.log import logger

from ..harness.state import SQLiteConversationStateStore


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


STARTUP_REPLAY_MAX_AGE_MINUTES = _bounded_env_int(
    "KEYTAO_STARTUP_REPLAY_MAX_AGE_MINUTES",
    10,
    1,
    60,
)
STARTUP_REPLAY_HISTORY_COUNT = _bounded_env_int(
    "KEYTAO_STARTUP_REPLAY_HISTORY_COUNT",
    50,
    1,
    200,
)


@dataclass
class StartupReplayCounts:
    fetched: int = 0
    replayed: int = 0
    skipped_answered: int = 0
    skipped_not_addressed: int = 0
    skipped_before_marker: int = 0
    skipped_too_old: int = 0
    skipped_invalid: int = 0
    failed: int = 0

    def add(self, other: "StartupReplayCounts") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


def _message_identity(row: object) -> Optional[tuple[float, int]]:
    if not isinstance(row, dict):
        return None
    try:
        timestamp = float(row.get("time"))
        message_id = int(str(row.get("message_id")))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp) or timestamp <= 0:
        return None
    return timestamp, message_id


def _row_user_id(row: object) -> str:
    if not isinstance(row, dict):
        return ""
    sender = row.get("sender")
    sender_id = sender.get("user_id") if isinstance(sender, dict) else ""
    return str(row.get("user_id") or sender_id or "").strip()


def _history_group_event(bot: object, group_id: str, row: dict[str, Any]) -> object:
    from nonebot.adapters.onebot.v11.event import GroupMessageEvent

    payload = dict(row)
    payload.update({
        "time": int(float(row["time"])),
        "self_id": int(str(getattr(bot, "self_id"))),
        "post_type": "message",
        "message_type": "group",
        "sub_type": str(row.get("sub_type") or "normal"),
        "message_id": int(str(row["message_id"])),
        "user_id": int(_row_user_id(row)),
        "group_id": int(group_id),
        "font": int(row.get("font") or 0),
        "sender": row.get("sender") or {"user_id": int(_row_user_id(row))},
    })
    payload.setdefault("message", row.get("raw_message") or "")
    payload.setdefault("raw_message", str(row.get("raw_message") or ""))
    return GroupMessageEvent.model_validate(payload)


async def replay_group_history(
    bot: object,
    group_id: str,
    *,
    state_store: Optional[SQLiteConversationStateStore] = None,
    history_count: int = STARTUP_REPLAY_HISTORY_COUNT,
    max_age_seconds: float = STARTUP_REPLAY_MAX_AGE_MINUTES * 60,
    now: Optional[float] = None,
    event_factory: Callable[[object, str, dict[str, Any]], object] = (
        _history_group_event
    ),
    is_addressed: Optional[Callable[[object, object], Awaitable[bool]]] = None,
    process_event: Optional[Callable[[object, object], Awaitable[None]]] = None,
) -> StartupReplayCounts:
    """Fetch one bounded history page and replay its eligible messages in order."""
    if state_store is None or is_addressed is None or process_event is None:
        from .openai_chat import (
            conversation_state_store,
            handle_ai_chat,
            should_handle,
        )

        state_store = state_store or conversation_state_store
        is_addressed = is_addressed or should_handle
        process_event = process_event or handle_ai_chat

    counts = StartupReplayCounts()
    try:
        response = await bot.get_group_msg_history(
            group_id=int(group_id),
            count=int(history_count),
        )
    except Exception as error:
        logger.warning(
            f"[startup_replay] history_failed group={group_id} "
            f"error={type(error).__name__}"
        )
        counts.failed = 1
        return counts
    rows = response.get("messages") if isinstance(response, dict) else None
    if not isinstance(rows, list):
        counts.failed = 1
        return counts
    counts.fetched = len(rows)
    ordered_rows = sorted(
        rows,
        key=lambda row: _message_identity(row) or (float("inf"), 0),
    )
    bot_id = str(getattr(bot, "self_id", "") or "").strip()
    last_bot_index = max(
        (
            index
            for index, row in enumerate(ordered_rows)
            if bot_id and _row_user_id(row) == bot_id
        ),
        default=-1,
    )
    marker = state_store.last_processed_group_message(group_id)
    marker_order = (
        (float(marker[1]), int(marker[0])) if marker is not None else None
    )
    current_time = float(time.time() if now is None else now)
    oldest_allowed = current_time - max(1.0, float(max_age_seconds))

    for index, row in enumerate(ordered_rows):
        identity = _message_identity(row)
        if identity is None or not isinstance(row, dict):
            counts.skipped_invalid += 1
            continue
        row_user_id = _row_user_id(row)
        if re.fullmatch(r"[1-9][0-9]{0,19}", row_user_id) is None:
            counts.skipped_invalid += 1
            continue
        if bot_id and row_user_id == bot_id:
            continue
        if identity[0] < oldest_allowed:
            counts.skipped_too_old += 1
            continue
        if marker_order is not None and identity <= marker_order:
            counts.skipped_before_marker += 1
            continue
        if last_bot_index > index:
            counts.skipped_answered += 1
            continue
        try:
            event = event_factory(bot, group_id, row)
            if not await is_addressed(bot, event):
                counts.skipped_not_addressed += 1
                continue
            await process_event(bot, event)
        except Exception as error:
            logger.warning(
                f"[startup_replay] message_failed group={group_id} "
                f"message_id={row.get('message_id')} "
                f"error={type(error).__name__}"
            )
            counts.failed += 1
            break
        counts.replayed += 1
    return counts


def _configured_group_ids() -> tuple[str, ...]:
    values = " ".join((
        os.getenv("KEYTAO_STARTUP_REPLAY_GROUP_IDS", ""),
        os.getenv("KEYTAO_SYNC_NOTIFY_GROUP_IDS", ""),
    ))
    return tuple(dict.fromkeys(
        token
        for token in re.split(r"[\s,;]+", values.strip())
        if re.fullmatch(r"[1-9][0-9]{4,19}", token)
    ))


async def run_startup_replay(bot: object) -> StartupReplayCounts:
    """Replay configured and previously served groups sequentially."""
    from .openai_chat import conversation_state_store

    group_ids = tuple(dict.fromkeys((
        *_configured_group_ids(),
        *conversation_state_store.served_group_ids(),
    )))
    total = StartupReplayCounts()
    for group_id in group_ids:
        total.add(await replay_group_history(bot, group_id))
    logger.info(
        "[startup_replay] complete "
        f"groups={len(group_ids)} history_calls={len(group_ids)} "
        f"fetched={total.fetched} replayed={total.replayed} "
        f"skipped_answered={total.skipped_answered} "
        f"skipped_not_addressed={total.skipped_not_addressed} "
        f"skipped_before_marker={total.skipped_before_marker} "
        f"skipped_too_old={total.skipped_too_old} "
        f"skipped_invalid={total.skipped_invalid} failed={total.failed}"
    )
    return total


async def _on_bot_connect(bot: Bot) -> None:
    try:
        from nonebot.adapters.onebot.v11 import Bot as OneBotV11Bot
    except ImportError:
        return
    if isinstance(bot, OneBotV11Bot):
        await run_startup_replay(bot)


driver = get_driver()
if hasattr(driver, "on_bot_connect"):
    driver.on_bot_connect(_on_bot_connect)
