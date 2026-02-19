"""
Conversation History Store with SQLite
对话历史SQLite持久化存储
"""
import json
import sqlite3
import os
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime
from nonebot.log import logger


class HistoryStore:
    """SQLite-based conversation history storage"""
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize history store
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            # Default: keytao-bot/data/conversation_history.db
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "conversation_history.db")
        
        self.db_path = db_path
        self._init_db()
        logger.info(f"Initialized history store at: {self.db_path}")
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    platform TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(platform, user_id, timestamp)
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_platform_user 
                ON conversations(platform, user_id, timestamp DESC)
            """)
            
            conn.commit()
    
    def get_history(self, platform: str, user_id: str, limit: int = 30) -> List[Dict]:
        """
        Get conversation history for a user
        
        Args:
            platform: Platform type (telegram, qq, etc.)
            user_id: User's platform ID
            limit: Maximum number of messages to return
        
        Returns:
            List of message dicts with {role, content, timestamp}
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content, timestamp
                FROM conversations 
                WHERE platform = ? AND user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (platform, user_id, limit))
            
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
            
            logger.debug(f"Retrieved {len(messages)} history messages for {platform}:{user_id}")
            return messages
    
    def add_message(self, platform: str, user_id: str, role: str, content: str):
        """
        Add a single message to history
        
        Args:
            platform: Platform type
            user_id: User's platform ID
            role: Message role (user, assistant, system)
            content: Message content
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO conversations (platform, user_id, role, content)
                    VALUES (?, ?, ?, ?)
                """, (platform, user_id, role, content))
                conn.commit()
                logger.debug(f"Added {role} message for {platform}:{user_id}")
            except sqlite3.IntegrityError:
                # Duplicate message (same timestamp), ignore
                logger.warning(f"Duplicate message detected for {platform}:{user_id}")
    
    def add_conversation_round(self, platform: str, user_id: str, user_message: str, assistant_message: str):
        """
        Add a complete conversation round (user + assistant)
        
        Args:
            platform: Platform type
            user_id: User's platform ID
            user_message: User's message
            assistant_message: Assistant's response
        """
        self.add_message(platform, user_id, "user", user_message)
        self.add_message(platform, user_id, "assistant", assistant_message)
    
    def clear_history(self, platform: str, user_id: str) -> int:
        """
        Clear conversation history for a user
        
        Args:
            platform: Platform type
            user_id: User's platform ID
        
        Returns:
            Number of messages deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM conversations 
                WHERE platform = ? AND user_id = ?
            """, (platform, user_id))
            deleted = cursor.rowcount
            conn.commit()
            logger.info(f"Cleared {deleted} messages for {platform}:{user_id}")
            return deleted
    
    def cleanup_old_messages(self, days: int = 30):
        """
        Clean up messages older than specified days
        
        Args:
            days: Keep messages from last N days
        
        Returns:
            Number of messages deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM conversations 
                WHERE timestamp < datetime('now', '-' || ? || ' days')
            """, (days,))
            deleted = cursor.rowcount
            conn.commit()
            logger.info(f"Cleaned up {deleted} old messages (older than {days} days)")
            return deleted
    
    def get_stats(self) -> Dict:
        """Get statistics about stored conversations"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total messages
            cursor.execute("SELECT COUNT(*) FROM conversations")
            total_messages = cursor.fetchone()[0]
            
            # Unique users
            cursor.execute("SELECT COUNT(DISTINCT platform || ':' || user_id) FROM conversations")
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
