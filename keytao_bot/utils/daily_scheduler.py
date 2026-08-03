"""Generic "run this once a day at HH:MM" scheduler.

The bot has no APScheduler dependency (and must not grow one), so recurring work
is driven by a plain ``asyncio`` loop that sleeps until the next local wall-clock
occurrence of the configured time.

Two properties matter for correctness and are the reason this module exists
instead of an inline ``while True`` in every plugin:

* **Restart safety.** The date of the last completed attempt is persisted in
  SQLite. When the process starts and today's slot has already passed without a
  recorded run, the task is executed immediately ("catch-up") instead of being
  silently lost until tomorrow.
* **Loop durability.** A callback that raises is logged with a full traceback and
  the loop keeps running; only :class:`asyncio.CancelledError` stops it.

The date is recorded for every *completed* attempt, successful or not. A callback
that fails deterministically would otherwise be re-run by catch-up on every
process restart, which for expensive pipelines (LLM calls, outbound scraping)
turns a crash loop into a spend loop. Retrying a failed day is the callback's
job. A run cut short by cancellation (shutdown) is the one case that is *not*
recorded: it never reached a conclusion, so catch-up must still pick it up.
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from zoneinfo import ZoneInfo

from nonebot.log import logger

DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_DB_FILENAME = "daily_scheduler.db"

# Breathing room after a run so a task that finishes in microseconds cannot spin
# through the loop body repeatedly within the same second.
POST_RUN_GUARD_SECONDS = 60.0

DailyCallback = Callable[[], Awaitable[Any]]


# ---------------------------------------------------------------------------
# Pure helpers (unit tested without a running loop)
# ---------------------------------------------------------------------------


def resolve_timezone(name: Any, default: str = DEFAULT_TIMEZONE) -> ZoneInfo:
    """Return a :class:`ZoneInfo`, falling back to ``default`` on a bad name."""
    text = str(name or "").strip() or default
    try:
        return ZoneInfo(text)
    except Exception:
        logger.warning(f"[daily_scheduler] unknown timezone {text!r}, fall back to {default}")
        return ZoneInfo(default)


def clamp_hour(value: Any, default: int = 9) -> int:
    try:
        hour = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return hour if 0 <= hour <= 23 else default


def clamp_minute(value: Any, default: int = 0) -> int:
    try:
        minute = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return minute if 0 <= minute <= 59 else default


def target_datetime(now: datetime, hour: int, minute: int) -> datetime:
    """Today's occurrence of ``HH:MM`` in ``now``'s timezone."""
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def seconds_until_next_run(now: datetime, hour: int, minute: int) -> float:
    """Seconds until the next strictly-future occurrence of ``HH:MM``."""
    target = target_datetime(now, hour, minute)
    if target <= now:
        target = target + timedelta(days=1)
    return max((target - now).total_seconds(), 0.0)


def date_key(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d")


def should_catch_up(
    now: datetime,
    hour: int,
    minute: int,
    last_run_date: Optional[str],
) -> bool:
    """Whether today's slot was missed and must be executed right now.

    True only when today's ``HH:MM`` has already passed *and* no attempt is
    recorded for today (or any later date, which can only come from a clock that
    moved backwards and must not trigger a re-run).
    """
    if now < target_datetime(now, hour, minute):
        return False
    recorded = str(last_run_date or "").strip()
    if not recorded:
        return True
    return recorded < date_key(now)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _default_db_path() -> str:
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / DEFAULT_DB_FILENAME)


class DailyRunStore:
    """SQLite record of the last date each named daily task was attempted."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_task_runs (
                    task_name TEXT PRIMARY KEY,
                    last_run_date TEXT NOT NULL,
                    last_run_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_last_run_date(self, task_name: str) -> Optional[str]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT last_run_date FROM daily_task_runs WHERE task_name = ?",
                    (task_name,),
                ).fetchone()
        except sqlite3.Error as error:
            logger.warning(f"[daily_scheduler] cannot read last run date for {task_name}: {error}")
            return None
        return str(row[0]) if row and row[0] else None

    def set_last_run_date(self, task_name: str, run_date: str, run_at: Optional[str] = None) -> None:
        stamp = run_at or datetime.now().isoformat()
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO daily_task_runs (task_name, last_run_date, last_run_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(task_name) DO UPDATE SET
                        last_run_date = excluded.last_run_date,
                        last_run_at = excluded.last_run_at
                    """,
                    (task_name, run_date, stamp),
                )
                conn.commit()
        except sqlite3.Error as error:
            logger.warning(f"[daily_scheduler] cannot persist last run date for {task_name}: {error}")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class DailyScheduler:
    """Runs ``callback`` once per day at ``hour``:``minute`` in ``timezone``."""

    def __init__(
        self,
        name: str,
        callback: DailyCallback,
        *,
        hour: int = 9,
        minute: int = 0,
        timezone: Any = DEFAULT_TIMEZONE,
        store: Optional[DailyRunStore] = None,
        catch_up: bool = True,
        log_prefix: Optional[str] = None,
    ):
        self.name = name
        self.callback = callback
        self.hour = clamp_hour(hour)
        self.minute = clamp_minute(minute)
        self.timezone = timezone if isinstance(timezone, ZoneInfo) else resolve_timezone(timezone)
        self.catch_up = catch_up
        self.log_prefix = log_prefix or f"[daily_scheduler:{name}]"
        self._store = store
        self._task: Optional[asyncio.Task] = None
        self._run_lock = asyncio.Lock()

    # -- store is created lazily so constructing a scheduler never touches disk
    @property
    def store(self) -> DailyRunStore:
        if self._store is None:
            self._store = DailyRunStore()
        return self._store

    def now(self) -> datetime:
        return datetime.now(self.timezone)

    @property
    def is_running(self) -> bool:
        return bool(self._task and not self._task.done())

    async def run_now(self, reason: str = "manual") -> Any:
        """Execute the callback once and record the attempt date.

        Serialised against the scheduled run so a manual trigger can never
        overlap the timed one. Exceptions propagate to the caller; the loop is
        what swallows them.

        Cancellation is deliberately *not* recorded. A shutdown mid-run means
        the work never happened, so recording it would make catch-up skip the
        day entirely after restart. Only a run that reached its own conclusion -
        success or a genuine failure - counts as today's attempt.
        """
        async with self._run_lock:
            started = self.now()
            logger.info(f"{self.log_prefix} run start ({reason}) at {started.isoformat()}")
            try:
                result = await self.callback()
            except asyncio.CancelledError:
                logger.info(
                    f"{self.log_prefix} run cancelled ({reason}); "
                    "leaving today unrecorded so it can be caught up after restart"
                )
                raise
            except Exception:
                # Recorded even on failure: see the module docstring.
                self.store.set_last_run_date(self.name, date_key(started), started.isoformat())
                raise
            self.store.set_last_run_date(self.name, date_key(started), started.isoformat())
            return result

    async def _guarded_run(self, reason: str) -> None:
        try:
            await self.run_now(reason)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"{self.log_prefix} run failed ({reason})")

    async def _loop(self) -> None:
        logger.info(
            f"{self.log_prefix} scheduler started, daily at "
            f"{self.hour:02d}:{self.minute:02d} {self.timezone.key}"
        )
        if self.catch_up:
            try:
                last_run_date = self.store.get_last_run_date(self.name)
                if should_catch_up(self.now(), self.hour, self.minute, last_run_date):
                    logger.info(
                        f"{self.log_prefix} today's slot was missed "
                        f"(last run date={last_run_date or 'never'}), catching up now"
                    )
                    await self._guarded_run("catch-up")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(f"{self.log_prefix} catch-up check failed")

        while True:
            try:
                delay = seconds_until_next_run(self.now(), self.hour, self.minute)
                logger.info(f"{self.log_prefix} next run in {int(delay)} seconds")
                await asyncio.sleep(delay)
                await self._guarded_run("scheduled")
                await asyncio.sleep(POST_RUN_GUARD_SECONDS)
            except asyncio.CancelledError:
                logger.info(f"{self.log_prefix} scheduler cancelled")
                raise
            except Exception:
                # Never let a scheduling-side error kill the loop.
                logger.exception(f"{self.log_prefix} scheduler loop error")
                await asyncio.sleep(POST_RUN_GUARD_SECONDS)

    def start(self) -> Optional[asyncio.Task]:
        if self.is_running:
            return self._task
        self._task = asyncio.create_task(self._loop())
        return self._task

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if not task:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as error:  # pragma: no cover - shutdown best effort
            logger.debug(f"{self.log_prefix} scheduler task ended with: {error}")

    def register(self, driver: Any) -> None:
        """Wire ``start``/``stop`` into the NoneBot driver lifecycle."""

        @driver.on_startup
        async def _start_daily_task() -> None:  # pragma: no cover - driver callback
            self.start()

        @driver.on_shutdown
        async def _stop_daily_task() -> None:  # pragma: no cover - driver callback
            await self.stop()
