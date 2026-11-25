"""Durable reminders for idle KeyTao drafts."""

import json
import math
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, NamedTuple, Optional
from zoneinfo import ZoneInfo


DAY_SECONDS = 24 * 60 * 60
SHANGHAI = ZoneInfo("Asia/Shanghai")
CONFIRMATION_SECONDS = 2 * 60 * 60
READ_RETRY_SECONDS = 60 * 60
RECEIPT_COLUMNS = "platform,actor,batch_id,group_id,content_version,status,result_at"


class DraftReminderCursor(NamedTuple):
    modified_at: str
    batch_id: str

    def timestamp(self) -> float:
        return datetime.fromisoformat(self.modified_at).timestamp()


class DraftReminderPage(NamedTuple):
    candidates: list[Dict]
    page_end: Optional[DraftReminderCursor] = None
    resume_cursor: Optional[DraftReminderCursor] = None


def reminder_idle_minutes() -> int:
    try:
        return max(5, int(os.getenv("KEYTAO_DRAFT_REMINDER_IDLE_MINUTES", "20")))
    except ValueError:
        return 20


def _age_cutoff(now: float, setting: str, default_days: int) -> float:
    try:
        days = max(0, int(os.getenv(setting, str(default_days))))
    except ValueError:
        days = default_days
    return now - days * DAY_SECONDS


def reminder_group_cutoff(now: float) -> float:
    return _age_cutoff(now, "KEYTAO_DRAFT_REMINDER_GROUP_MAX_AGE_DAYS", 30)


def reminder_confirmation_is_current(receipt: Dict, timestamp: object) -> bool:
    delivered_at = receipt.get("result_at")
    if type(delivered_at) not in (int, float) or type(timestamp) not in (int, float):
        return False
    expires_at = delivered_at + CONFIRMATION_SECONDS
    return int(delivered_at) <= timestamp < expires_at and time.time() < expires_at


def _day_start(now: float) -> float:
    return datetime.fromtimestamp(now, SHANGHAI).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def draft_snapshot_is_current(draft: Dict, batch_id: str) -> bool:
    items = draft.get("items")
    return (
        draft.get("success") is True and draft.get("batchId") == batch_id
        and draft.get("status", draft.get("batchStatus")) == "Draft"
        and type(draft.get("contentVersion")) is int and draft["contentVersion"] >= 0
        and isinstance(items, list) and bool(items) and all(isinstance(row, dict) for row in items)
        and draft.get("count", len(items)) == len(items) and not draft.get("itemsTruncationNotice")
    )


def initialize_reminder_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS draft_reminder_batch ("
        "platform TEXT NOT NULL, actor TEXT NOT NULL, batch_id TEXT NOT NULL, "
        "scope TEXT NOT NULL, group_id TEXT NOT NULL, content_version INTEGER, "
        "last_write REAL NOT NULL, batch_url TEXT NOT NULL, active INTEGER NOT NULL, "
        "last_checked REAL NOT NULL DEFAULT 0, "
        "PRIMARY KEY (batch_id))"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS draft_reminder_attempt ("
        "platform TEXT NOT NULL, actor TEXT NOT NULL, batch_id TEXT NOT NULL, "
        "content_version INTEGER NOT NULL, group_id TEXT NOT NULL, "
        "intended_at REAL NOT NULL, status TEXT NOT NULL, message_id TEXT NOT NULL DEFAULT '', "
        "result_at REAL, PRIMARY KEY (batch_id, content_version))"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS draft_reminder_due "
        "ON draft_reminder_batch(active, last_checked, last_write)"
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS draft_reminder_batch_identity "
        "ON draft_reminder_batch(batch_id)"
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS draft_reminder_version_identity "
        "ON draft_reminder_attempt(batch_id, content_version)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS draft_reminder_receipt "
        "ON draft_reminder_attempt(platform, group_id, message_id)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS draft_reminder_actor_due "
        "ON draft_reminder_attempt(platform, actor, intended_at)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS draft_reminder_rollout ("
        "id INTEGER PRIMARY KEY CHECK (id=1), started_at REAL NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS draft_reminder_read_retry ("
        "batch_id TEXT PRIMARY KEY, retry_after REAL NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS draft_reminder_account_day ("
        "account_id TEXT NOT NULL, day_start REAL NOT NULL, PRIMARY KEY (day_start, account_id))"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS draft_reminder_web_cursor ("
        "id INTEGER PRIMARY KEY CHECK (id=1), modified_at TEXT NOT NULL, batch_id TEXT NOT NULL)"
    )


def record_draft_write(
    connection: sqlite3.Connection, scope: str, platform: str, actor: str, payload: Dict,
) -> None:
    """Record the write in the journal's existing transaction, including closures."""
    try:
        address = json.loads(scope)
        written_at = float(payload["completedAt"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return
    if (
        not isinstance(address, list) or len(address) != 4
        or address[0] != platform or address[3] != actor
        or not math.isfinite(written_at) or written_at <= 0
    ):
        return
    batch_id = str(payload.get("batchId") or "")
    if not batch_id:
        return
    after = payload.get("after")
    after = after if isinstance(after, dict) else {}
    version = after.get("contentVersion")
    version = version if type(version) is int and version >= 0 else None
    group_id = str(address[2]) if address[1] == "group" else ""
    items = after.get("items")
    active = bool(
        group_id and version is not None and after.get("batchId") == batch_id
        and isinstance(items, list) and items and not payload.get("submitted")
    )
    connection.execute(
        "INSERT INTO draft_reminder_batch "
        "(platform,actor,batch_id,scope,group_id,content_version,last_write,batch_url,active) "
        "VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(batch_id) DO UPDATE SET "
        "platform=excluded.platform, actor=excluded.actor, scope=excluded.scope, "
        "group_id=excluded.group_id, content_version=excluded.content_version, "
        "last_write=excluded.last_write, batch_url=excluded.batch_url, active=excluded.active, "
        "last_checked=0 WHERE excluded.last_write >= draft_reminder_batch.last_write",
        (platform, actor, batch_id, scope, group_id, version, written_at,
         str(payload.get("batchUrl") or after.get("batchUrl") or ""), int(active)),
    )


class DraftReminderStore:
    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = str(db_path or Path(__file__).resolve().parents[2] / "data" / "draft_mutation_claims.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connect() as connection:
            initialize_reminder_schema(connection)
        try:
            os.chmod(self.db_path, 0o600)
        except OSError:
            pass

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_write(self, scope: str, platform: str, actor: str, payload: Dict) -> None:
        with self._connect() as connection:
            record_draft_write(connection, scope, platform, actor, payload)

    def _rollout_time(self, now: float) -> float:
        with self._connect() as connection:
            row = connection.execute("SELECT started_at FROM draft_reminder_rollout WHERE id=1").fetchone()
            if row is None:
                connection.execute("INSERT OR IGNORE INTO draft_reminder_rollout VALUES (1,?)", (now,))
                row = connection.execute("SELECT started_at FROM draft_reminder_rollout WHERE id=1").fetchone()
            return row["started_at"]

    def _prepare_scan_history(self, now: float) -> list[str]:
        with self._connect() as connection:
            connection.execute("DELETE FROM draft_reminder_read_retry WHERE retry_after<=?", (now,))
            connection.execute("DELETE FROM draft_reminder_account_day WHERE day_start<?", (_day_start(now),))
            return [row["batch_id"] for row in connection.execute(
                "SELECT batch_id FROM draft_reminder_attempt GROUP BY batch_id "
                "ORDER BY MAX(intended_at) DESC, batch_id DESC LIMIT 200"
            )]

    def _web_cursor(self, modified_after: float) -> Optional[DraftReminderCursor]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT modified_at, batch_id FROM draft_reminder_web_cursor WHERE id=1"
            ).fetchone()
        cursor = DraftReminderCursor(*row) if row is not None else None
        if cursor is not None and cursor.timestamp() < modified_after:
            self._save_web_cursor(cursor, None)
            return None
        return cursor

    def _save_web_cursor(
        self, previous: Optional[DraftReminderCursor], cursor: Optional[DraftReminderCursor],
    ) -> None:
        with self._connect() as connection:
            if cursor is not None:
                connection.execute(
                    "INSERT INTO draft_reminder_web_cursor VALUES (1,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET modified_at=excluded.modified_at, batch_id=excluded.batch_id "
                    "WHERE modified_at=? AND batch_id=?",
                    (*cursor, *(previous or ("", ""))),
                )
            elif previous is not None:
                connection.execute(
                    "DELETE FROM draft_reminder_web_cursor WHERE id=1 AND modified_at=? AND batch_id=?", previous,
                )

    def _bootstrap_existing_writes(self, modified_after: float) -> None:
        """Import at most twenty pre-feature journal records per scan."""
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='completed_draft_operation'"
            ).fetchone() is None:
                return
            rows = connection.execute(
                "SELECT c.scope,c.platform,c.actor,c.payload_json FROM completed_draft_operation c "
                "JOIN draft_operation_actor_write w ON w.platform=c.platform AND w.actor=c.actor "
                "AND w.operation_id=c.operation_id "
                "WHERE json_valid(c.payload_json) AND json_extract(c.payload_json,'$.completedAt')>=? "
                "AND NOT EXISTS ("
                "SELECT 1 FROM draft_reminder_batch b WHERE b.batch_id=json_extract(c.payload_json,'$.batchId') "
                "AND b.last_write>=json_extract(c.payload_json,'$.completedAt')) LIMIT 20",
                (modified_after,),
            ).fetchall()
            for row in rows:
                record_draft_write(connection, row["scope"], row["platform"], row["actor"], json.loads(row["payload_json"]))

    def _candidates(self, now: float, idle_minutes: int, modified_after: float, web_candidates: list[Dict]) -> list[Dict]:
        idle_before = now - idle_minutes * 60
        group_after = reminder_group_cutoff(now)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT b.* FROM draft_reminder_batch b WHERE b.active=1 AND b.last_write BETWEEN ? AND ? "
                "AND b.last_checked<=? "
                "AND NOT EXISTS (SELECT 1 FROM draft_reminder_attempt a WHERE a.batch_id=b.batch_id "
                "OR (a.platform=b.platform AND a.actor=b.actor AND a.intended_at>=?))",
                (max(modified_after, group_after), idle_before, now - READ_RETRY_SECONDS, _day_start(now)),
            ).fetchall()
            candidates = {row["batch_id"]: {**row, "bot_write": dict(row)} for row in rows}
            for web in web_candidates:
                batch_id = web["batch_id"]
                local = connection.execute(
                    "SELECT * FROM draft_reminder_batch WHERE batch_id=?", (batch_id,),
                ).fetchone()
                if local is not None:
                    if not local["active"] or local["last_write"] >= web["last_write"]:
                        continue
                    if local["last_write"] >= group_after:
                        web = {**web, **{key: local[key] for key in ("platform", "actor", "group_id")}}
                if modified_after <= web["last_write"] <= idle_before:
                    candidates[batch_id] = {**web, "bot_write": dict(local) if local else None}
        return sorted(candidates.values(), key=lambda row: (row["last_write"], row["batch_id"]))

    @staticmethod
    def _attempted(connection: sqlite3.Connection, candidate: Dict, now: float) -> bool:
        return connection.execute(
            "SELECT 1 FROM draft_reminder_attempt WHERE batch_id=? "
            "OR (platform=? AND actor=? AND intended_at>=?) LIMIT 1",
            (candidate["batch_id"], candidate["platform"], candidate["actor"], _day_start(now)),
        ).fetchone() is not None

    def _claim(self, candidate: Dict, now: float, account_id: str) -> bool:
        key = (candidate["platform"], candidate["actor"], candidate["batch_id"])
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT * FROM draft_reminder_batch WHERE batch_id=?", (candidate["batch_id"],),
            ).fetchone()
            if (dict(current) if current else None) != candidate["bot_write"] or self._attempted(connection, candidate, now):
                return False
            if account_id and not connection.execute(
                "INSERT OR IGNORE INTO draft_reminder_account_day VALUES (?,?)",
                (account_id, _day_start(now)),
            ).rowcount:
                return False
            connection.execute(
                "INSERT INTO draft_reminder_attempt "
                "(platform,actor,batch_id,content_version,group_id,intended_at,status) "
                "VALUES (?,?,?,?,?,?,'intent')",
                (*key, candidate["content_version"], candidate["group_id"], now),
            )
            return True

    def resolve_receipt(self, platform: str, group_id: str, message_id: str) -> Optional[Dict]:
        if not message_id:
            return None
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {RECEIPT_COLUMNS} FROM draft_reminder_attempt WHERE platform=? AND group_id=? "
                "AND message_id=? AND status IN ('sent','pending','reply_only','confirmed')",
                (platform, group_id, str(message_id)),
            ).fetchone()
        return dict(row) if row else None

    def pending_plain_reply(
        self, platform: str, group_id: str, actor: str, *, now: Optional[float] = None,
    ) -> Optional[Dict]:
        now = time.time() if now is None else now
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {RECEIPT_COLUMNS} FROM draft_reminder_attempt "
                "WHERE platform=? AND group_id=? AND actor=? AND status='pending' "
                "AND result_at>? AND result_at<=? ORDER BY result_at DESC LIMIT 1",
                (platform, group_id, actor, now - CONFIRMATION_SECONDS, now),
            ).fetchone()
        return dict(row) if row else None

    def update_confirmation(self, receipt: Dict, *, quoted: bool = False, confirm: bool) -> bool:
        status = "confirmed" if confirm else "reply_only"
        with self._connect() as connection:
            updated = connection.execute(
                "UPDATE draft_reminder_attempt SET status=? "
                "WHERE platform=? AND group_id=? AND actor=? AND batch_id=? AND content_version=? "
                "AND (status='pending' OR (? AND status='reply_only')) AND result_at>?",
                (status, receipt["platform"], receipt["group_id"], receipt["actor"], receipt["batch_id"],
                 receipt["content_version"], quoted, time.time() - CONFIRMATION_SECONDS),
            )
            return updated.rowcount == 1

    async def scan(
        self, read_draft: Callable, send_reminder: Callable, *,
        list_web_candidates: Optional[Callable] = None, find_user: Optional[Callable] = None,
        now: Optional[float] = None,
    ) -> int:
        if os.environ.get("KEYTAO_DRAFT_REMINDER") != "1":
            return 0
        fixed_now = now
        now = time.time() if fixed_now is None else fixed_now
        rollout = self._rollout_time(now)
        if not 8 <= datetime.fromtimestamp(now, SHANGHAI).hour < 23:
            return 0
        modified_after = max(rollout, _age_cutoff(now, "KEYTAO_DRAFT_REMINDER_MAX_AGE_DAYS", 7))
        self._bootstrap_existing_writes(modified_after)
        idle_minutes = reminder_idle_minutes()
        exclude_ids = self._prepare_scan_history(now)
        saved_cursor = self._web_cursor(modified_after) if list_web_candidates else None
        pages = (
            list_web_candidates(idle_minutes, modified_after, exclude_ids, now, resume_cursor=saved_cursor)
            if list_web_candidates else None
        )
        page = await anext(pages, None) if pages is not None else None
        accounts = {}
        sent = 0
        checked_batches = set()
        check_limit = 20
        while True:
            web_asked = False
            web_candidates = page.candidates if page is not None else []
            page_end = (
                (page.page_end.timestamp(), page.page_end.batch_id)
                if page is not None and page.page_end is not None else None
            )
            check_limit += len(web_candidates)
            for candidate in self._candidates(now, idle_minutes, modified_after, web_candidates):
                if len(checked_batches) >= check_limit:
                    break
                # Leave newer local drafts for the next merge to keep oldest-first ordering.
                if page_end is not None and (candidate["last_write"], candidate["batch_id"]) > page_end:
                    break
                batch_id = candidate["batch_id"]
                if batch_id in checked_batches:
                    continue
                with self._connect() as connection:
                    if self._attempted(connection, candidate, now):
                        continue
                    if connection.execute(
                        "SELECT 1 FROM draft_reminder_read_retry WHERE batch_id=? AND retry_after>?", (batch_id, now),
                    ).fetchone():
                        continue
                checked_batches.add(batch_id)
                try:
                    draft = await read_draft(candidate)
                except Exception:
                    draft = {}
                url = str(draft.get("batchUrl") or candidate["batch_url"]) if isinstance(draft, dict) else ""
                if (
                    not url or not draft_snapshot_is_current(draft, batch_id)
                    or draft["contentVersion"] != candidate["content_version"]
                    or len(draft["items"]) != candidate.get("item_count", len(draft["items"]))
                ):
                    checked_at = time.time() if fixed_now is None else fixed_now
                    with self._connect() as connection:
                        connection.execute(
                            "INSERT INTO draft_reminder_read_retry VALUES (?,?) ON CONFLICT(batch_id) DO UPDATE SET "
                            "retry_after=MAX(retry_after, excluded.retry_after)", (batch_id, checked_at + READ_RETRY_SECONDS),
                        )
                    continue
                key = (candidate["platform"], candidate["actor"], batch_id)
                identity = ("qq", candidate["creator_qq_id"]) if candidate.get("creator_qq_id") else key[:2]
                if find_user is not None and identity not in accounts:
                    user = await find_user(*identity)
                    account_id = user.get("id") if isinstance(user, dict) else None
                    accounts[identity] = str(account_id).strip() if type(account_id) in (int, str) else ""
                intended_at = time.time() if fixed_now is None else fixed_now
                if (
                    not 8 <= datetime.fromtimestamp(intended_at, SHANGHAI).hour < 23
                    or not self._claim(candidate, intended_at, accounts.get(identity, ""))
                ):
                    continue
                web_asked = web_asked or "creator_qq_id" in candidate
                text = f"你有草稿未提交（{len(draft['items'])} 条），是否帮你提交？提交请回复「确认」\n{url}"
                message_id, status = "", "error"
                try:
                    message_id = str(await send_reminder(candidate, text) or "")
                    status = "pending" if message_id else "unknown"
                    sent += int(status == "pending")
                except Exception:
                    pass
                with self._connect() as connection:
                    connection.execute(
                        "UPDATE draft_reminder_attempt SET status=?,message_id=?,result_at=? "
                        "WHERE platform=? AND actor=? AND batch_id=? AND content_version=? AND status='intent'",
                        (status, message_id, time.time() if fixed_now is None else fixed_now,
                         *key, candidate["content_version"]),
                    )
            if page_end is None:
                if (
                    page is not None and page.resume_cursor != saved_cursor
                    and (not web_asked or page.resume_cursor is None)
                ):
                    self._save_web_cursor(saved_cursor, page.resume_cursor)
                break
            page = await anext(pages, None) if not web_asked else None
        return sent
