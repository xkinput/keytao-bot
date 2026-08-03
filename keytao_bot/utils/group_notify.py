"""Reusable QQ group broadcast helper.

Any plugin that needs to push a message into one or more QQ groups should go
through :func:`send_group_notification` instead of talking to the adapter
directly. It handles bot discovery, message chunking, and a single retry.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Iterable, List, Optional, Sequence

from nonebot.log import logger

# QQ rejects very long single messages; keep a comfortable margin.
MAX_MESSAGE_CHARS = 1500
RETRY_DELAY_SECONDS = 2.0


def parse_group_ids(value: Any) -> List[str]:
    """Normalise a group-id config value into a list of trimmed strings."""
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        items: Iterable[Any] = value
    else:
        items = str(value).replace("，", ",").split(",")
    result: List[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> List[str]:
    """Split a long message on line boundaries, hard-splitting oversized lines."""
    if limit <= 0:
        return [text]
    if len(text) <= limit:
        return [text]

    chunks: List[str] = []
    current = ""
    for line in text.split("\n"):
        while len(line) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > limit:
            if current:
                chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def _get_qq_bots() -> List[Any]:
    try:
        from nonebot import get_bots
        from nonebot.adapters.onebot.v11 import Bot as QQBot
    except Exception as error:  # pragma: no cover - adapter missing
        logger.error(f"[group_notify] OneBot adapter unavailable: {error}")
        return []
    try:
        return [bot for bot in get_bots().values() if isinstance(bot, QQBot)]
    except Exception as error:
        logger.error(f"[group_notify] failed to enumerate bots: {error}")
        return []


async def _send_chunk(bot: Any, group_id: str, chunk: str, log_prefix: str) -> bool:
    for attempt in range(2):
        try:
            await bot.send_group_msg(group_id=int(group_id), message=chunk)
            return True
        except Exception as error:
            if attempt == 0:
                logger.warning(f"{log_prefix} group {group_id} send failed, retrying: {error}")
                await asyncio.sleep(RETRY_DELAY_SECONDS)
            else:
                logger.error(f"{log_prefix} group {group_id} send failed after retry: {error}")
    return False


async def send_group_notification(
    text: str,
    group_ids: Sequence[str],
    *,
    log_prefix: str = "[group_notify]",
    bot: Optional[Any] = None,
) -> Dict[str, Any]:
    """Broadcast ``text`` to every group id, chunking and retrying once.

    Returns ``{"sent": [...], "failed": [...], "chunks": n}``.
    """
    normalized_ids = parse_group_ids(group_ids)
    if not normalized_ids:
        logger.warning(f"{log_prefix} no target group configured, skip notification")
        return {"sent": [], "failed": [], "chunks": 0}

    if bot is None:
        qq_bots = _get_qq_bots()
        if not qq_bots:
            logger.warning(f"{log_prefix} no QQ bot connected, cannot send group notification")
            return {"sent": [], "failed": list(normalized_ids), "chunks": 0}
        bot = qq_bots[0]

    chunks = split_message(text)
    sent: List[str] = []
    failed: List[str] = []
    for group_id in normalized_ids:
        delivered = True
        for chunk in chunks:
            if not await _send_chunk(bot, group_id, chunk, log_prefix):
                delivered = False
                break
        if delivered:
            sent.append(group_id)
            logger.info(f"{log_prefix} sent notification to QQ group {group_id} ({len(chunks)} chunk(s))")
        else:
            failed.append(group_id)

    return {"sent": sent, "failed": failed, "chunks": len(chunks)}
