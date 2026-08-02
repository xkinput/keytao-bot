"""Persistent fencing for draft mutations whose network result is uncertain."""

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional


_DEFAULT_STORE: Optional["DraftMutationClaimStore"] = None
_DEFAULT_STORE_LOCK = threading.Lock()


class DraftMutationClaimStore:
    """Keep one durable in-flight destructive operation per actor."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            db_path = str(data_dir / "draft_mutation_claims.db")
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        self._secure_storage()
        self._init_db()
        self._secure_storage()

    def _secure_storage(self) -> None:
        parent = Path(self.db_path).parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(parent, 0o700)
        except OSError:
            pass
        for candidate in (self.db_path, f"{self.db_path}-wal", f"{self.db_path}-shm"):
            if not os.path.exists(candidate):
                continue
            try:
                os.chmod(candidate, 0o600)
            except OSError:
                pass

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA secure_delete = ON")
            yield connection
            if connection.in_transaction:
                connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS draft_mutation_fence (
                    platform TEXT NOT NULL,
                    platform_id TEXT NOT NULL,
                    operation_kind TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'inflight',
                    result_json TEXT,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (platform, platform_id)
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(draft_mutation_fence)"
                ).fetchall()
            }
            if "status" not in columns:
                connection.execute(
                    "ALTER TABLE draft_mutation_fence "
                    "ADD COLUMN status TEXT NOT NULL DEFAULT 'inflight'"
                )
            if "result_json" not in columns:
                connection.execute(
                    "ALTER TABLE draft_mutation_fence ADD COLUMN result_json TEXT"
                )

    @staticmethod
    def _serialize(payload: Dict) -> tuple[str, str]:
        payload_json = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(payload_json.encode("utf-8")) > 1024 * 1024:
            raise ValueError("mutation claim payload is too large")
        fingerprint = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        return payload_json, fingerprint

    @staticmethod
    def _serialize_result(result: Dict) -> str:
        result_json = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(result_json.encode("utf-8")) > 1024 * 1024:
            raise ValueError("mutation result payload is too large")
        return result_json

    def begin(
        self,
        platform: str,
        platform_id: str,
        operation_kind: str,
        payload: Dict,
    ) -> Optional[str]:
        """Atomically acquire a claim and return its fingerprint."""
        payload_json, fingerprint = self._serialize(payload)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO draft_mutation_fence (
                    platform, platform_id, operation_kind,
                    fingerprint, payload_json, status, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, 'inflight', NULL, ?)
                """,
                (
                    str(platform),
                    str(platform_id),
                    str(operation_kind),
                    fingerprint,
                    payload_json,
                    time.time(),
                ),
            )
            return fingerprint if cursor.rowcount == 1 else None

    def get(
        self,
        platform: str,
        platform_id: str,
    ) -> Optional[Dict]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT fingerprint, payload_json, created_at,
                       operation_kind, status, result_json
                FROM draft_mutation_fence
                WHERE platform = ? AND platform_id = ?
                """,
                (str(platform), str(platform_id)),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row[1]))
        if not isinstance(payload, dict):
            raise ValueError("mutation claim payload must be an object")
        result = None
        if row[5] is not None:
            result = json.loads(str(row[5]))
            if not isinstance(result, dict):
                raise ValueError("mutation claim result must be an object")
        status = str(row[4] or "inflight")
        if status not in {"inflight", "resolved"}:
            raise ValueError("mutation claim status is invalid")
        return {
            "fingerprint": str(row[0]),
            "payload": payload,
            "createdAt": float(row[2]),
            "operationKind": str(row[3]),
            "status": status,
            "result": result,
        }

    def resolve(
        self,
        platform: str,
        platform_id: str,
        operation_kind: str,
        fingerprint: str,
        result: Dict,
    ) -> bool:
        """Persist a final response before it is delivered to the actor."""
        result_json = self._serialize_result(result)
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT status, result_json
                FROM draft_mutation_fence
                WHERE platform = ? AND platform_id = ?
                  AND operation_kind = ? AND fingerprint = ?
                """,
                (
                    str(platform),
                    str(platform_id),
                    str(operation_kind),
                    str(fingerprint),
                ),
            ).fetchone()
            if row is None:
                return False
            if str(row[0]) == "resolved":
                return str(row[1] or "") == result_json
            cursor = connection.execute(
                """
                UPDATE draft_mutation_fence
                SET status = 'resolved', result_json = ?
                WHERE platform = ? AND platform_id = ?
                  AND operation_kind = ? AND fingerprint = ?
                  AND status = 'inflight'
                """,
                (
                    result_json,
                    str(platform),
                    str(platform_id),
                    str(operation_kind),
                    str(fingerprint),
                ),
            )
            return cursor.rowcount == 1

    def acknowledge(
        self,
        platform: str,
        platform_id: str,
        operation_kind: str,
        fingerprint: str,
    ) -> bool:
        """Remove one resolved receipt only after its reply was sent."""
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM draft_mutation_fence
                WHERE platform = ? AND platform_id = ?
                  AND operation_kind = ? AND fingerprint = ?
                  AND status = 'resolved'
                """,
                (
                    str(platform),
                    str(platform_id),
                    str(operation_kind),
                    str(fingerprint),
                ),
            )
            return cursor.rowcount == 1

    def transition_resolved(
        self,
        platform: str,
        platform_id: str,
        prior_operation_kind: str,
        prior_fingerprint: str,
        operation_kind: str,
        payload: Dict,
    ) -> Optional[str]:
        """Atomically continue a delivered-as-one multi-step operation."""
        payload_json, fingerprint = self._serialize(payload)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE draft_mutation_fence
                SET operation_kind = ?, fingerprint = ?, payload_json = ?,
                    status = 'inflight', result_json = NULL, created_at = ?
                WHERE platform = ? AND platform_id = ?
                  AND operation_kind = ? AND fingerprint = ?
                  AND status = 'resolved'
                """,
                (
                    str(operation_kind),
                    fingerprint,
                    payload_json,
                    time.time(),
                    str(platform),
                    str(platform_id),
                    str(prior_operation_kind),
                    str(prior_fingerprint),
                ),
            )
            return fingerprint if cursor.rowcount == 1 else None

    def consume(
        self,
        platform: str,
        platform_id: str,
        operation_kind: str,
        fingerprint: str,
    ) -> bool:
        """Release only the exact claim resolved by a read or response."""
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM draft_mutation_fence
                WHERE platform = ? AND platform_id = ?
                  AND operation_kind = ? AND fingerprint = ?
                """,
                (
                    str(platform),
                    str(platform_id),
                    str(operation_kind),
                    str(fingerprint),
                ),
            )
            return cursor.rowcount == 1

    def discard_actor(self, platform: str, platform_id: str) -> Optional[Dict]:
        """Explicitly abandon one unresolved/resolved actor fence."""
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT operation_kind, fingerprint, payload_json, status
                FROM draft_mutation_fence
                WHERE platform = ? AND platform_id = ?
                """,
                (str(platform), str(platform_id)),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                DELETE FROM draft_mutation_fence
                WHERE platform = ? AND platform_id = ?
                  AND operation_kind = ? AND fingerprint = ?
                """,
                (
                    str(platform),
                    str(platform_id),
                    str(row[0]),
                    str(row[1]),
                ),
            )
        payload = json.loads(str(row[2]))
        return {
            "operationKind": str(row[0]),
            "fingerprint": str(row[1]),
            "payload": payload if isinstance(payload, dict) else {},
            "status": str(row[3] or "inflight"),
        }


def get_default_draft_mutation_claim_store() -> DraftMutationClaimStore:
    """Create the production store lazily so imports remain read-only."""
    global _DEFAULT_STORE
    if _DEFAULT_STORE is not None:
        return _DEFAULT_STORE
    with _DEFAULT_STORE_LOCK:
        if _DEFAULT_STORE is None:
            _DEFAULT_STORE = DraftMutationClaimStore()
    return _DEFAULT_STORE
