"""
Conversation History Store with SQLite
对话历史SQLite持久化存储
"""
import sqlite3
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union
from pathlib import Path
from datetime import datetime, timezone
from nonebot.log import logger

from keytao_bot.harness.conversation import ConversationAddress


DEFAULT_HISTORY_RETENTION_DAYS = 30
DEFAULT_MAX_MESSAGES_PER_CONVERSATION = 200
DEFAULT_MAX_MESSAGES_PER_SPACE = 2_000
DEFAULT_MAX_TOTAL_HISTORY_MESSAGES = 100_000
DEFAULT_HISTORY_TOMBSTONE_RETENTION_DAYS = 180
DEFAULT_MAX_HISTORY_GENERATION_TOMBSTONES = 100_000


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class HistoryGenerationToken:
    """Persistent fence captured before a potentially slow conversation turn."""

    platform: str
    space_type: str
    space_id: str
    actor_id: str
    generation: int
    scope_generation: int


class HistoryStore:
    """SQLite-based conversation history storage"""
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        max_messages_per_conversation: int = DEFAULT_MAX_MESSAGES_PER_CONVERSATION,
        max_messages_per_space: int = DEFAULT_MAX_MESSAGES_PER_SPACE,
        retention_days: int = DEFAULT_HISTORY_RETENTION_DAYS,
        max_total_messages: int = DEFAULT_MAX_TOTAL_HISTORY_MESSAGES,
        generation_tombstone_days: int = DEFAULT_HISTORY_TOMBSTONE_RETENTION_DAYS,
        max_generation_tombstones: int = DEFAULT_MAX_HISTORY_GENERATION_TOMBSTONES,
    ):
        """
        Initialize history store
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            # Default: keytao-bot/data/conversation_history.db
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True, mode=0o700)
            db_path = str(data_dir / "conversation_history.db")
        
        self.db_path = db_path
        self.max_messages_per_conversation = max(2, int(max_messages_per_conversation))
        self.max_messages_per_space = max(
            self.max_messages_per_conversation,
            int(max_messages_per_space),
        )
        self.retention_days = max(1, int(retention_days))
        self.max_total_messages = max(
            self.max_messages_per_conversation,
            int(max_total_messages),
        )
        self.generation_tombstone_days = max(
            self.retention_days,
            int(generation_tombstone_days),
        )
        self.max_generation_tombstones = max(1, int(max_generation_tombstones))
        self._secure_storage()
        self._init_db()
        self._secure_storage()
        logger.info(f"Initialized history store at: {self.db_path}")

    def _secure_storage(self) -> None:
        parent = Path(self.db_path).parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(parent, 0o700)
        except OSError as error:
            logger.warning(f"Failed to secure history directory {parent}: {error}")
        for candidate in (self.db_path, f"{self.db_path}-wal", f"{self.db_path}-shm"):
            if not os.path.exists(candidate):
                continue
            try:
                os.chmod(candidate, 0o600)
            except OSError as error:
                logger.warning(f"Failed to secure history database file {candidate}: {error}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA secure_delete = ON")
            yield conn
            if conn.in_transaction:
                conn.commit()
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

    def _truncate_wal(self, conn: sqlite3.Connection) -> None:
        """Best-effort removal of deleted plaintext from the SQLite WAL."""
        try:
            result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if result and int(result[0]) != 0:
                logger.warning(f"History WAL truncate remained busy for {self.db_path}")
        except sqlite3.OperationalError as error:
            logger.warning(f"History WAL truncate failed for {self.db_path}: {error}")
    
    def _init_db(self):
        """Initialize database schema"""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Older schema used UNIQUE(platform, user_id, timestamp), which
            # drops one side of a user/assistant round when both inserts land
            # in the same second. Migrate it away in place.
            cursor.execute("""
                SELECT sql FROM sqlite_master
                WHERE type = 'table' AND name = 'conversations'
            """)
            row = cursor.fetchone()
            schema_sql = row[0] if row and row[0] else ""
            if "UNIQUE(platform, user_id, timestamp)" in schema_sql:
                try:
                    cursor.execute("""
                        CREATE TABLE conversations_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            platform TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            role TEXT NOT NULL,
                            content TEXT NOT NULL,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            space_type TEXT NOT NULL DEFAULT 'private',
                            space_id TEXT NOT NULL DEFAULT '',
                            actor_name TEXT NOT NULL DEFAULT ''
                            ,round_id TEXT NOT NULL DEFAULT ''
                        )
                    """)
                    cursor.execute("""
                        INSERT INTO conversations_new (id, platform, user_id, role, content, timestamp)
                        SELECT id, platform, user_id, role, content, timestamp
                        FROM conversations
                        ORDER BY id
                    """)
                    cursor.execute("DROP TABLE conversations")
                    cursor.execute("ALTER TABLE conversations_new RENAME TO conversations")
                except sqlite3.OperationalError as error:
                    logger.warning(f"Skip history schema migration for readonly DB {self.db_path}: {error}")

            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    space_type TEXT NOT NULL DEFAULT 'private',
                    space_id TEXT NOT NULL DEFAULT '',
                    actor_name TEXT NOT NULL DEFAULT ''
                    ,round_id TEXT NOT NULL DEFAULT ''
                )
            """)

            cursor.execute("PRAGMA table_info(conversations)")
            columns = {row[1] for row in cursor.fetchall()}
            for name, ddl in (
                ("space_type", "TEXT NOT NULL DEFAULT ''"),
                ("space_id", "TEXT NOT NULL DEFAULT ''"),
                ("actor_name", "TEXT NOT NULL DEFAULT ''"),
                ("round_id", "TEXT NOT NULL DEFAULT ''"),
            ):
                if name not in columns:
                    cursor.execute(f"ALTER TABLE conversations ADD COLUMN {name} {ddl}")

            # Old actor-only rows mixed private and group turns under one key,
            # so no safe destination or future per-space clear can be inferred.
            # They are never read by the isolated schema; secure-delete them
            # here instead of retaining inaccessible conversation plaintext.
            cursor.execute("""
                DELETE FROM conversations
                WHERE space_type = '' OR space_id = ''
                   OR space_type IN ('legacy_actor', 'legacy_group')
            """)
            deleted_legacy_rows = max(0, cursor.rowcount)

            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_address
                ON conversations(platform, space_type, space_id, user_id, id DESC)
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history_conversation_state (
                    platform TEXT NOT NULL,
                    space_type TEXT NOT NULL,
                    space_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (platform, space_type, space_id, user_id)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS history_scope_state (
                    platform TEXT NOT NULL,
                    space_type TEXT NOT NULL,
                    space_id TEXT NOT NULL,
                    generation INTEGER NOT NULL DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (platform, space_type, space_id)
                )
            """)
            self._cleanup_retention(cursor)
            self._enforce_global_limit(cursor)
            
            conn.commit()
            if deleted_legacy_rows:
                self._truncate_wal(conn)
    
    @staticmethod
    def _address(
        address_or_platform: Union[ConversationAddress, str],
        user_id: Optional[str] = None,
    ) -> ConversationAddress:
        if isinstance(address_or_platform, ConversationAddress):
            return address_or_platform
        if user_id is None:
            raise TypeError("user_id is required for the legacy history interface")
        return ConversationAddress.private(address_or_platform, user_id)

    def get_history(
        self,
        address_or_platform: Union[ConversationAddress, str],
        user_id: Optional[str] = None,
        limit: int = 30,
    ) -> List[Dict]:
        """
        Get conversation history for a user
        
        Args:
            platform: Platform type (telegram, qq, etc.)
            user_id: User's platform ID
            limit: Maximum number of messages to return
        
        Returns:
            List of message dicts with {role, content, timestamp}
        """
        address = self._address(address_or_platform, user_id)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content, timestamp
                FROM conversations
                WHERE platform = ? AND user_id = ?
                  AND space_type = ? AND space_id = ?
                  AND strftime('%s', timestamp) IS NOT NULL
                  AND CAST(strftime('%s', timestamp) AS INTEGER)
                      >= CAST(strftime('%s', 'now') AS INTEGER) - (? * 86400)
                ORDER BY id DESC
                LIMIT ?
            """, (
                address.platform,
                address.actor_id,
                address.space_type,
                address.space_id,
                self.retention_days,
                max(0, int(limit)),
            ))
            
            # Reverse to get chronological order
            rows = cursor.fetchall()
            messages = [
                {
                    "role": row[0], 
                    "content": row[1],
                    "timestamp": row[2]
                } 
                for row in reversed(rows)
            ]
            
            logger.debug(f"Retrieved {len(messages)} history messages for {address}")
            return messages

    def get_space_history(
        self,
        address: ConversationAddress,
        limit: int = 30,
    ) -> List[Dict]:
        """Return recent turns in the current shared space with stable actors."""
        if address.space_type != "group":
            return []
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT role, content, timestamp, user_id, actor_name
                FROM conversations
                WHERE platform = ? AND space_type = ? AND space_id = ?
                  AND strftime('%s', timestamp) IS NOT NULL
                  AND CAST(strftime('%s', timestamp) AS INTEGER)
                      >= CAST(strftime('%s', 'now') AS INTEGER) - (? * 86400)
                ORDER BY id DESC
                LIMIT ?
            """, (
                address.platform,
                address.space_type,
                address.space_id,
                self.retention_days,
                max(0, int(limit)),
            )).fetchall()
        return [
            {
                "role": row[0],
                "content": row[1],
                "timestamp": row[2],
                "actor_id": row[3],
                "actor_name": row[4],
            }
            for row in reversed(rows)
        ]
    
    def add_message(self, platform: str, user_id: str, role: str, content: str):
        """
        Add a single message to history
        
        Args:
            platform: Platform type
            user_id: User's platform ID
            role: Message role (user, assistant, system)
            content: Message content
        """
        address = ConversationAddress.private(platform, user_id)
        with self._connect() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO conversations (
                        platform, user_id, role, content, timestamp,
                        space_type, space_id, actor_name, round_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    platform,
                    user_id,
                    role,
                    content,
                    _utc_now_iso(),
                    address.space_type,
                    address.space_id,
                    "",
                    uuid.uuid4().hex,
                ))
                self._enforce_limits(cursor, address)
                conn.commit()
                logger.debug(f"Added {role} message for {platform}:{user_id}")
            except sqlite3.IntegrityError as error:
                logger.warning(
                    f"Failed to add {role} message for {platform}:{user_id}: {error}"
                )
    
    def add_conversation_round(
        self,
        address_or_platform: Union[ConversationAddress, str],
        *args: str,
        speaker_name: str = "",
        generation_token: Optional[HistoryGenerationToken] = None,
    ) -> bool:
        """
        Add a complete conversation round (user + assistant)
        
        Args:
            platform: Platform type
            user_id: User's platform ID
            user_message: User's message
            assistant_message: Assistant's response
        """
        if isinstance(address_or_platform, ConversationAddress):
            if len(args) != 2:
                raise TypeError("address form requires user_message and assistant_message")
            address = address_or_platform
            user_message, assistant_message = args
        else:
            if len(args) != 3:
                raise TypeError("legacy form requires user_id, user_message and assistant_message")
            user_id, user_message, assistant_message = args
            address = ConversationAddress.private(address_or_platform, user_id)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                if (
                    generation_token is not None
                    and not self._generation_state_exists(cursor, address)
                ):
                    conn.rollback()
                    return False
                self._ensure_conversation_state(cursor, address)
                self._ensure_scope_state(cursor, address)
                generation = self._conversation_generation(cursor, address)
                scope_generation = self._scope_generation(cursor, address)
                if generation_token is not None and not self._token_matches(
                    address,
                    generation,
                    scope_generation,
                    generation_token,
                ):
                    conn.rollback()
                    return False
                cursor.execute("""
                    UPDATE history_conversation_state
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE platform = ? AND space_type = ?
                      AND space_id = ? AND user_id = ?
                """, (
                    address.platform,
                    address.space_type,
                    address.space_id,
                    address.actor_id,
                ))
                cursor.execute("""
                    UPDATE history_scope_state
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE platform = ? AND space_type = ? AND space_id = ?
                """, (
                    address.platform,
                    address.space_type,
                    address.space_id,
                ))
                timestamp = _utc_now_iso()
                round_id = uuid.uuid4().hex
                cursor.executemany("""
                    INSERT INTO conversations (
                        platform, user_id, role, content, timestamp,
                        space_type, space_id, actor_name, round_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        address.platform, address.actor_id, "user", user_message,
                        timestamp, address.space_type, address.space_id, speaker_name,
                        round_id,
                    ),
                    (
                        address.platform, address.actor_id, "assistant", assistant_message,
                        timestamp, address.space_type, address.space_id, speaker_name,
                        round_id,
                    ),
                ])
                self._enforce_limits(cursor, address)
            except Exception:
                conn.rollback()
                raise
            conn.commit()
        self._secure_storage()
        return True

    def capture_generation(self, address: ConversationAddress) -> HistoryGenerationToken:
        """Capture the current durable generation for one conversation address."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            self._ensure_conversation_state(cursor, address)
            self._ensure_scope_state(cursor, address)
            generation = self._conversation_generation(cursor, address)
            scope_generation = self._scope_generation(cursor, address)
            cursor.execute("""
                UPDATE history_conversation_state
                SET updated_at = CURRENT_TIMESTAMP
                WHERE platform = ? AND space_type = ?
                  AND space_id = ? AND user_id = ?
            """, (
                address.platform,
                address.space_type,
                address.space_id,
                address.actor_id,
            ))
            cursor.execute("""
                UPDATE history_scope_state
                SET updated_at = CURRENT_TIMESTAMP
                WHERE platform = ? AND space_type = ? AND space_id = ?
            """, (
                address.platform,
                address.space_type,
                address.space_id,
            ))
            self._enforce_generation_state_limit(cursor)
            conn.commit()
        return HistoryGenerationToken(
            platform=address.platform,
            space_type=address.space_type,
            space_id=address.space_id,
            actor_id=address.actor_id,
            generation=generation,
            scope_generation=scope_generation,
        )

    def is_generation_current(
        self,
        address: ConversationAddress,
        token: Optional[HistoryGenerationToken],
    ) -> bool:
        """Return whether a captured token still authorizes a history write."""
        if token is None:
            return True
        with self._connect() as conn:
            cursor = conn.cursor()
            if not self._generation_state_exists(cursor, address):
                return False
            generation = self._conversation_generation(cursor, address)
            scope_generation = self._scope_generation(cursor, address)
        return self._token_matches(address, generation, scope_generation, token)

    @staticmethod
    def _generation_state_exists(
        cursor: sqlite3.Cursor,
        address: ConversationAddress,
    ) -> bool:
        row = cursor.execute("""
            SELECT
                EXISTS (
                    SELECT 1 FROM history_conversation_state
                    WHERE platform = ? AND space_type = ?
                      AND space_id = ? AND user_id = ?
                ),
                EXISTS (
                    SELECT 1 FROM history_scope_state
                    WHERE platform = ? AND space_type = ? AND space_id = ?
                )
        """, (
            address.platform,
            address.space_type,
            address.space_id,
            address.actor_id,
            address.platform,
            address.space_type,
            address.space_id,
        )).fetchone()
        return bool(row and row[0] and row[1])

    @staticmethod
    def _ensure_conversation_state(
        cursor: sqlite3.Cursor,
        address: ConversationAddress,
    ) -> None:
        cursor.execute("""
            INSERT INTO history_conversation_state(
                platform, space_type, space_id, user_id
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(platform, space_type, space_id, user_id) DO NOTHING
        """, (
            address.platform,
            address.space_type,
            address.space_id,
            address.actor_id,
        ))

    @staticmethod
    def _conversation_generation(
        cursor: sqlite3.Cursor,
        address: ConversationAddress,
    ) -> int:
        row = cursor.execute("""
            SELECT generation
            FROM history_conversation_state
            WHERE platform = ? AND space_type = ? AND space_id = ? AND user_id = ?
        """, (
            address.platform,
            address.space_type,
            address.space_id,
            address.actor_id,
        )).fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _ensure_scope_state(
        cursor: sqlite3.Cursor,
        address: ConversationAddress,
    ) -> None:
        cursor.execute("""
            INSERT INTO history_scope_state(platform, space_type, space_id)
            VALUES (?, ?, ?)
            ON CONFLICT(platform, space_type, space_id) DO NOTHING
        """, (
            address.platform,
            address.space_type,
            address.space_id,
        ))

    @staticmethod
    def _scope_generation(
        cursor: sqlite3.Cursor,
        address: ConversationAddress,
    ) -> int:
        row = cursor.execute("""
            SELECT generation
            FROM history_scope_state
            WHERE platform = ? AND space_type = ? AND space_id = ?
        """, (
            address.platform,
            address.space_type,
            address.space_id,
        )).fetchone()
        return int(row[0]) if row else 0

    @staticmethod
    def _token_matches(
        address: ConversationAddress,
        generation: int,
        scope_generation: int,
        token: HistoryGenerationToken,
    ) -> bool:
        return (
            token.platform == address.platform
            and token.space_type == address.space_type
            and token.space_id == address.space_id
            and token.actor_id == address.actor_id
            and token.generation == generation
            and token.scope_generation == scope_generation
        )

    def _enforce_limits(
        self,
        cursor: sqlite3.Cursor,
        address: ConversationAddress,
    ) -> None:
        self._cleanup_retention(cursor)
        cursor.execute("""
            DELETE FROM conversations
            WHERE round_id IN (
                SELECT DISTINCT round_id FROM (
                    SELECT round_id FROM conversations
                    WHERE platform = ? AND user_id = ?
                      AND space_type = ? AND space_id = ?
                      AND round_id != ''
                    ORDER BY id DESC
                    LIMIT -1 OFFSET ?
                )
            )
        """, (
            address.platform,
            address.actor_id,
            address.space_type,
            address.space_id,
            self.max_messages_per_conversation,
        ))
        cursor.execute("""
            DELETE FROM conversations
            WHERE platform = ? AND space_type = ? AND space_id = ?
              AND round_id IN (
                  SELECT round_id FROM conversations
                  WHERE platform = ? AND space_type = ? AND space_id = ?
                    AND round_id != ''
                  GROUP BY round_id
                  ORDER BY MAX(id) DESC
                  LIMIT -1 OFFSET ?
              )
        """, (
            address.platform,
            address.space_type,
            address.space_id,
            address.platform,
            address.space_type,
            address.space_id,
            self.max_messages_per_space,
        ))
        self._enforce_global_limit(cursor)

    def _cleanup_retention(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("""
            DELETE FROM conversations
            WHERE strftime('%s', timestamp) IS NULL
               OR CAST(strftime('%s', timestamp) AS INTEGER)
                    < CAST(strftime('%s', 'now') AS INTEGER) - (? * 86400)
        """, (self.retention_days,))
        cursor.execute("""
            DELETE FROM history_conversation_state
            WHERE strftime('%s', updated_at) IS NOT NULL
              AND CAST(strftime('%s', updated_at) AS INTEGER)
                  < CAST(strftime('%s', 'now') AS INTEGER) - (? * 86400)
              AND NOT EXISTS (
                  SELECT 1 FROM conversations
                  WHERE conversations.platform = history_conversation_state.platform
                    AND conversations.space_type = history_conversation_state.space_type
                    AND conversations.space_id = history_conversation_state.space_id
                    AND conversations.user_id = history_conversation_state.user_id
              )
        """, (self.generation_tombstone_days,))
        cursor.execute("""
            DELETE FROM history_scope_state
            WHERE strftime('%s', updated_at) IS NOT NULL
              AND CAST(strftime('%s', updated_at) AS INTEGER)
                  < CAST(strftime('%s', 'now') AS INTEGER) - (? * 86400)
              AND NOT EXISTS (
                  SELECT 1 FROM history_conversation_state
                  WHERE history_conversation_state.platform = history_scope_state.platform
                    AND history_conversation_state.space_type = history_scope_state.space_type
                    AND history_conversation_state.space_id = history_scope_state.space_id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM conversations
                  WHERE conversations.platform = history_scope_state.platform
                    AND conversations.space_type = history_scope_state.space_type
                    AND conversations.space_id = history_scope_state.space_id
              )
        """, (self.generation_tombstone_days,))

    def cleanup_retention(self) -> int:
        """Delete expired rows even when no new messages arrive."""
        with self._connect() as conn:
            before = conn.total_changes
            cursor = conn.cursor()
            self._cleanup_retention(cursor)
            self._enforce_generation_state_limit(cursor)
            conn.commit()
            deleted = conn.total_changes - before
            self._truncate_wal(conn)
            return deleted

    def count_history_rows(self, address: ConversationAddress) -> int:
        """Count every physical row for one exact conversation address."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) FROM conversations
                WHERE platform = ? AND space_type = ? AND space_id = ? AND user_id = ?
                """,
                (
                    address.platform,
                    address.space_type,
                    address.space_id,
                    address.actor_id,
                ),
            ).fetchone()
        return int(row[0] if row else 0)

    def _enforce_global_limit(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("""
            WITH rounds AS (
                SELECT
                    platform,
                    space_type,
                    space_id,
                    round_id,
                    MIN(id) AS oldest_id
                FROM conversations
                WHERE round_id != '' AND space_type IN ('private', 'group')
                GROUP BY platform, space_type, space_id, round_id
            ), ranked AS (
                SELECT
                    platform,
                    space_type,
                    space_id,
                    round_id,
                    oldest_id,
                    COUNT(*) OVER (
                        PARTITION BY platform, space_type, space_id
                    ) AS scope_count,
                    ROW_NUMBER() OVER (
                        PARTITION BY platform, space_type, space_id
                        ORDER BY oldest_id ASC
                    ) AS eviction_rank
                FROM rounds
            ), victims AS (
                SELECT platform, space_type, space_id, round_id
                FROM ranked
                ORDER BY
                    (scope_count - eviction_rank) DESC,
                    oldest_id ASC
                LIMIT MAX(0, (SELECT COUNT(*) FROM rounds) - ?)
            )
            DELETE FROM conversations
            WHERE EXISTS (
                SELECT 1 FROM victims
                WHERE victims.platform = conversations.platform
                  AND victims.space_type = conversations.space_type
                  AND victims.space_id = conversations.space_id
                  AND victims.round_id = conversations.round_id
            )
        """, (self.max_total_messages,))
        self._enforce_generation_state_limit(cursor)

    def _enforce_generation_state_limit(self, cursor: sqlite3.Cursor) -> None:
        """Bound empty generation tombstones without deleting live subjects."""
        cursor.execute("""
            DELETE FROM history_conversation_state
            WHERE rowid IN (
                SELECT state.rowid
                FROM history_conversation_state AS state
                WHERE NOT EXISTS (
                    SELECT 1 FROM conversations
                    WHERE conversations.platform = state.platform
                      AND conversations.space_type = state.space_type
                      AND conversations.space_id = state.space_id
                      AND conversations.user_id = state.user_id
                )
                ORDER BY
                    CASE
                        WHEN strftime('%s', state.updated_at) IS NULL THEN 0
                        ELSE 1
                    END DESC,
                    CAST(strftime('%s', state.updated_at) AS INTEGER) DESC,
                    state.rowid DESC
                LIMIT -1 OFFSET ?
            )
        """, (self.max_generation_tombstones,))
        # Once an actor tombstone is gone, a token is fail-closed even if the
        # shared scope row is also removed. Keep only scopes still anchored by
        # content or a surviving actor state.
        cursor.execute("""
            DELETE FROM history_scope_state
            WHERE NOT EXISTS (
                SELECT 1 FROM conversations
                WHERE conversations.platform = history_scope_state.platform
                  AND conversations.space_type = history_scope_state.space_type
                  AND conversations.space_id = history_scope_state.space_id
            )
              AND NOT EXISTS (
                SELECT 1 FROM history_conversation_state
                WHERE history_conversation_state.platform = history_scope_state.platform
                  AND history_conversation_state.space_type = history_scope_state.space_type
                  AND history_conversation_state.space_id = history_scope_state.space_id
            )
        """)
    
    def clear_history(
        self,
        address_or_platform: Union[ConversationAddress, str],
        user_id: Optional[str] = None,
    ) -> int:
        """
        Clear conversation history for a user
        
        Args:
            platform: Platform type
            user_id: User's platform ID
        
        Returns:
            Number of messages deleted
        """
        address = self._address(address_or_platform, user_id)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                self._ensure_conversation_state(cursor, address)
                self._ensure_scope_state(cursor, address)
                cursor.execute("""
                    UPDATE history_conversation_state
                    SET generation = generation + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE platform = ? AND space_type = ?
                      AND space_id = ? AND user_id = ?
                """, (
                    address.platform,
                    address.space_type,
                    address.space_id,
                    address.actor_id,
                ))
                cursor.execute("""
                    UPDATE history_scope_state
                    SET generation = generation + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE platform = ? AND space_type = ? AND space_id = ?
                """, (
                    address.platform,
                    address.space_type,
                    address.space_id,
                ))
                cursor.execute("""
                    DELETE FROM conversations
                    WHERE platform = ? AND user_id = ?
                      AND space_type = ? AND space_id = ?
                """, (
                    address.platform,
                    address.actor_id,
                    address.space_type,
                    address.space_id,
                ))
                deleted = cursor.rowcount
                self._enforce_generation_state_limit(cursor)
            except Exception:
                conn.rollback()
                raise
            conn.commit()
            self._truncate_wal(conn)
            logger.info(f"Cleared {deleted} messages for {address}")
            return deleted
    
    def cleanup_old_messages(self, days: int = 30):
        """
        Clean up messages older than specified days
        
        Args:
            days: Keep messages from last N days
        
        Returns:
            Number of messages deleted
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM conversations 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            deleted = cursor.rowcount
            self._enforce_generation_state_limit(cursor)
            conn.commit()
            logger.info(f"Cleaned up {deleted} old messages (older than {days} days)")
            return deleted
    
    def get_stats(self) -> Dict:
        """Get statistics about stored conversations"""
        with self._connect() as conn:
            cursor = conn.cursor()
            
            # Total messages
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_messages = cursor.fetchone()[0]
            
            # Unique users
            cursor.execute("""
                SELECT COUNT(DISTINCT platform || ':' || space_type || ':' || space_id || ':' || user_id)
                FROM conversations
            """)
            unique_users = cursor.fetchone()[0]
            
            # Database size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                "total_messages": total_messages,
                "unique_users": unique_users,
                "db_size_bytes": db_size,
                "db_size_mb": round(db_size / 1024 / 1024, 2)
            }


# Global instance (lazy initialization)
_history_store: Optional[HistoryStore] = None


def get_history_store() -> HistoryStore:
    """Get or create global history store instance"""
    global _history_store
    if _history_store is None:
        _history_store = HistoryStore()
    return _history_store
