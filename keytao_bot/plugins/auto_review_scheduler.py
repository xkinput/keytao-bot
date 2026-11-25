"""Poll KeyTao Next for draft items that still need an automatic review."""

from asyncio import CancelledError, Task, create_task, sleep
from contextlib import closing, suppress
from datetime import datetime
from pathlib import Path
import sqlite3
from urllib.parse import quote
from zoneinfo import ZoneInfo

from nonebot import get_driver
from nonebot.log import logger

from keytao_bot.utils import http_client
from keytao_bot.utils.daily_scheduler import seconds_until_next_run


DB_PATH = Path(__file__).resolve().parents[2] / "data" / "auto_review_scheduler.db"
TIMEZONE = ZoneInfo("Asia/Shanghai")
QUEUE_LIMIT = 12
MAX_ATTEMPTS = 2
RATE_LIMIT_BACKOFF = 600
ERROR_BACKOFF = 120
LOG = "[auto_review_scheduler]"

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_chunks (
    day TEXT PRIMARY KEY,
    chunks INTEGER NOT NULL,
    cap_logged INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS item_attempts (
    pr_id INTEGER NOT NULL,
    digest TEXT NOT NULL,
    attempts INTEGER NOT NULL,
    PRIMARY KEY (pr_id, digest)
);
"""

driver = get_driver()
_scheduler_task: Task | None = None
_disabled_logged = False


def _int_config(name: str, default: int) -> int:
    try:
        return int(str(http_client.config_value(name.lower(), name, default)))
    except (TypeError, ValueError):
        return default


def _now() -> datetime:
    return datetime.now(TIMEZONE)


def _daily_count(db: sqlite3.Connection, day: str) -> int:
    row = db.execute("SELECT chunks FROM daily_chunks WHERE day = ?", (day,)).fetchone()
    return row[0] if row else 0


def _daily_pause(db: sqlite3.Connection, day: str, count: int) -> float:
    with db:
        marked = db.execute(
            "UPDATE daily_chunks SET cap_logged = 1 WHERE day = ? AND cap_logged = 0", (day,)
        ).rowcount
    if marked:
        logger.info(f"{LOG} daily cap reached day={day} daily_count={count}")
    now = _now()
    return seconds_until_next_run(now, 0, 0) if now.date().isoformat() == day else 0


def _reserve_chunk(
    db: sqlite3.Connection, day: str, daily_cap: int, items: dict[int, str]
) -> tuple[int, list[int]]:
    # Commit both limits before POST because a timeout or crash can hide a successful write.
    with db:
        db.execute("BEGIN IMMEDIATE")
        count = _daily_count(db, day)
        if count >= daily_cap:
            return count, []

        eligible = []
        for pr_id, digest in items.items():
            row = db.execute(
                "SELECT attempts FROM item_attempts WHERE pr_id = ? AND digest = ?", (pr_id, digest)
            ).fetchone()
            if row is None or row[0] < MAX_ATTEMPTS:
                eligible.append((pr_id, digest))
        if not eligible:
            return count, []

        db.executemany(
            "INSERT INTO item_attempts VALUES (?, ?, 1) "
            "ON CONFLICT(pr_id, digest) DO UPDATE SET attempts = attempts + 1",
            eligible,
        )
        db.execute(
            "INSERT INTO daily_chunks (day, chunks) VALUES (?, 1) "
            "ON CONFLICT(day) DO UPDATE SET chunks = chunks + 1",
            (day,),
        )
    return count + 1, [pr_id for pr_id, _ in eligible]


async def _post_chunk(batch_id: str, pr_ids: list[int], count: int) -> int:
    reviewed = skipped = None
    status = None
    try:
        response = await http_client.keytao_request(
            "POST", f"/api/bot/batches/{quote(batch_id, safe='')}/auto-review",
            json_body={"prIds": pr_ids}, timeout=180.0, retries=1,
        )
        status = response.status_code
        if status == 429:
            return RATE_LIMIT_BACKOFF
        if not 200 <= status < 300:
            raise ValueError("unsuccessful review HTTP status")
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("invalid review payload")
        reviewed_ids, skipped_ids = data.get("reviewed"), data.get("skipped")
        reviewed = len(reviewed_ids) if isinstance(reviewed_ids, list) else None
        skipped = len(skipped_ids) if isinstance(skipped_ids, list) else None
        if data.get("rateLimited") is True:
            return RATE_LIMIT_BACKOFF
        if data.get("success") is not True or reviewed is None or skipped is None:
            raise ValueError("invalid review result")
        return 0
    except Exception as error:
        logger.warning(f"{LOG} review failed status={status} error={type(error).__name__}")
        return ERROR_BACKOFF
    finally:
        logger.info(
            f"{LOG} batchId={batch_id!r} sent={len(pr_ids)} reviewed={reviewed} "
            f"skipped={skipped} daily_count={count}"
        )


async def _poll_once(db: sqlite3.Connection, daily_cap: int, interval: int) -> float:
    day = _now().date().isoformat()
    count = _daily_count(db, day)
    if count >= daily_cap:
        return _daily_pause(db, day, count)

    response = await http_client.keytao_request(
        "GET", "/api/bot/auto-review/queue", params={"limit": QUEUE_LIMIT}, retries=1,
    )
    if response.status_code == 429:
        return RATE_LIMIT_BACKOFF
    if not 200 <= response.status_code < 300:
        logger.warning(f"{LOG} queue failed status={response.status_code}")
        return ERROR_BACKOFF
    data = response.json()
    if not isinstance(data, dict) or data.get("success") is not True or "batchId" not in data:
        raise ValueError("invalid queue response")
    batch_id = data["batchId"]
    if batch_id is None:
        return interval
    if not isinstance(batch_id, str) or batch_id in {"", ".", ".."}:
        raise ValueError("invalid queue batch ID")

    raw_items = data.get("items")
    if not isinstance(raw_items, list) or len(raw_items) > QUEUE_LIMIT:
        raise ValueError("invalid queue items")
    items = {}
    for item in raw_items:
        pr_id, digest = item["prId"], item["digest"]
        if type(pr_id) is not int or not isinstance(digest, str) or pr_id in items:
            raise ValueError("invalid queue item")
        items[pr_id] = digest
    if not items:
        return interval

    day = _now().date().isoformat()
    count, pr_ids = _reserve_chunk(db, day, daily_cap, items)
    if not pr_ids:
        if count >= daily_cap:
            return _daily_pause(db, day, count)
        logger.debug(f"{LOG} queue exhausted batchId={batch_id!r}")
        return RATE_LIMIT_BACKOFF

    backoff = await _post_chunk(batch_id, pr_ids, count)
    if count >= daily_cap:
        return max(backoff, _daily_pause(db, day, count))
    return backoff or interval


async def _scheduler_loop(daily_cap: int, interval: int) -> None:
    logger.info(f"{LOG} scheduler started daily_cap={daily_cap} interval_seconds={interval}")
    while True:
        try:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(DB_PATH, timeout=0.25)) as db:
                db.executescript(SCHEMA)
                while True:
                    await sleep(await _poll_once(db, daily_cap, interval))
        except Exception as error:
            logger.warning(f"{LOG} poll failed error={type(error).__name__}")
            await sleep(ERROR_BACKOFF)


@driver.on_startup
async def start_auto_review_scheduler() -> None:
    global _scheduler_task, _disabled_logged
    if _scheduler_task and not _scheduler_task.done():
        return
    daily_cap = _int_config("KEYTAO_AUTO_REVIEW_DAILY_CHUNKS", 0)
    if daily_cap <= 0:
        if not _disabled_logged:
            logger.info(f"{LOG} scheduler disabled")
            _disabled_logged = True
        return
    interval = max(20, _int_config("KEYTAO_AUTO_REVIEW_INTERVAL_SECONDS", 60))
    _scheduler_task = create_task(_scheduler_loop(daily_cap, interval))


@driver.on_shutdown
async def stop_auto_review_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task:
        _scheduler_task.cancel()
        with suppress(CancelledError):
            await _scheduler_task
        _scheduler_task = None
