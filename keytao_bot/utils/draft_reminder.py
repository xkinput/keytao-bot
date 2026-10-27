"""Durable, bounded reminders backed by the bot's completed draft writes."""

import json
import math
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, Optional
from zoneinfo import ZoneInfo


DAY_SECONDS = 24 * 60 * 60
SHANGHAI = ZoneInfo("Asia/Shanghai")


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

    def _bootstrap_existing_writes(self) -> None:
        """Import at most twenty pre-feature journal records per hourly scan."""
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='completed_draft_operation'"
            ).fetchone() is None:
                return
            rows = connection.execute(
                "SELECT c.scope,c.platform,c.actor,c.payload_json FROM completed_draft_operation c "
                "JOIN draft_operation_actor_write w ON w.platform=c.platform AND w.actor=c.actor "
                "AND w.operation_id=c.operation_id "
                "WHERE json_valid(c.payload_json) AND NOT EXISTS ("
                "SELECT 1 FROM draft_reminder_batch b WHERE b.batch_id=json_extract(c.payload_json,'$.batchId') "
                "AND b.last_write>=json_extract(c.payload_json,'$.completedAt')) LIMIT 20"
            ).fetchall()
            for row in rows:
                record_draft_write(connection, row["scope"], row["platform"], row["actor"], json.loads(row["payload_json"]))

    def _candidates(self, now: float) -> list:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT b.* FROM draft_reminder_batch b WHERE b.active=1 AND b.last_write<=? "
                "AND NOT EXISTS (SELECT 1 FROM draft_reminder_attempt a WHERE a.batch_id=b.batch_id "
                "AND a.content_version=b.content_version) "
                "AND (SELECT COUNT(*) FROM draft_reminder_attempt a WHERE a.batch_id=b.batch_id)<3 "
                "AND NOT EXISTS (SELECT 1 FROM draft_reminder_attempt a WHERE a.batch_id=b.batch_id "
                "AND a.intended_at>?) "
                "ORDER BY b.last_checked,b.last_write,b.batch_id LIMIT 20",
                (now - DAY_SECONDS, now - DAY_SECONDS),
            ).fetchall()
        return [dict(row) for row in rows]

    def _claim(self, candidate: Dict, now: float) -> bool:
        key = (candidate["platform"], candidate["actor"], candidate["batch_id"])
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT platform,actor,content_version,last_write,group_id,active FROM draft_reminder_batch "
                "WHERE batch_id=?", (candidate["batch_id"],),
            ).fetchone()
            if (
                current is None or not current["active"]
                or current["platform"] != candidate["platform"] or current["actor"] != candidate["actor"]
                or current["content_version"] != candidate["content_version"]
                or current["last_write"] != candidate["last_write"]
                or current["group_id"] != candidate["group_id"]
            ):
                return False
            attempts = connection.execute(
                "SELECT content_version,intended_at FROM draft_reminder_attempt "
                "WHERE batch_id=?", (candidate["batch_id"],),
            ).fetchall()
            if (
                len(attempts) >= 3
                or any(row["content_version"] == candidate["content_version"] for row in attempts)
                or any(now - row["intended_at"] < DAY_SECONDS for row in attempts)
            ):
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
                "SELECT a.platform,a.actor,a.batch_id,a.group_id,a.content_version "
                "FROM draft_reminder_attempt a WHERE a.platform=? AND a.group_id=? "
                "AND a.message_id=? AND a.status='sent'",
                (platform, group_id, str(message_id)),
            ).fetchone()
        return dict(row) if row else None

    async def scan(
        self, read_draft: Callable, send_reminder: Callable, *, now: Optional[float] = None,
    ) -> int:
        if os.environ.get("KEYTAO_DRAFT_REMINDER") != "1":
            return 0
        fixed_now = now
        now = time.time() if fixed_now is None else fixed_now
        if not 8 <= datetime.fromtimestamp(now, SHANGHAI).hour < 23:
            return 0
        self._bootstrap_existing_writes()
        sent = 0
        for candidate in self._candidates(now):
            key = (candidate["platform"], candidate["actor"], candidate["batch_id"])
            with self._connect() as connection:
                connection.execute(
                    "UPDATE draft_reminder_batch SET last_checked=? WHERE batch_id=?",
                    (now, candidate["batch_id"]),
                )
            try:
                draft = await read_draft(candidate)
            except Exception:
                continue
            if not isinstance(draft, dict):
                continue
            items = draft.get("items")
            if (
                draft.get("success") is not True or draft.get("batchId") != candidate["batch_id"]
                or draft.get("status", draft.get("batchStatus")) != "Draft"
                or type(draft.get("contentVersion")) is not int
                or draft["contentVersion"] != candidate["content_version"]
                or not isinstance(items, list) or not items or any(not isinstance(row, dict) for row in items)
                or draft.get("count", len(items)) != len(items) or draft.get("itemsTruncationNotice")
            ):
                continue
            url = str(draft.get("batchUrl") or candidate["batch_url"])
            intended_at = time.time() if fixed_now is None else fixed_now
            if (
                not 8 <= datetime.fromtimestamp(intended_at, SHANGHAI).hour < 23
                or not url or not self._claim(candidate, intended_at)
            ):
                continue
            modified = datetime.fromtimestamp(candidate["last_write"], SHANGHAI).strftime("%Y-%m-%d %H:%M")
            text = (
                f"你的草稿还没有提交（{len(items)} 条，最后修改于 {modified}）。\n"
                f"草稿地址：{url}\n回复「提交」提交，或「清空草稿」放弃。"
            )
            message_id, status = "", "error"
            try:
                message_id = str(await send_reminder(candidate, text) or "")
                status = "sent" if message_id else "unknown"
                sent += int(status == "sent")
            except Exception:
                pass
            with self._connect() as connection:
                connection.execute(
                    "UPDATE draft_reminder_attempt SET status=?,message_id=?,result_at=? "
                    "WHERE platform=? AND actor=? AND batch_id=? AND content_version=? AND status='intent'",
                    (status, message_id, time.time() if fixed_now is None else fixed_now,
                     *key, candidate["content_version"]),
                )
        return sent
