"""Persist community votes against exact submitted snapshots and sent receipts."""

import asyncio
import json
import os
import re
import sqlite3
import unicodedata
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from weakref import WeakValueDictionary

from nonebot.log import logger

from .user_resolver import find_user_by_platform


_pending_receipts: ContextVar[list | None] = ContextVar("community_vote_receipts", default=None)
_locks = WeakValueDictionary()
_default_store = None
_hook_installed = False


def enabled() -> bool:
    return os.getenv("KEYTAO_COMMUNITY_VOTE") == "1"


def _eligible_snapshot(data: dict) -> bool:
    items = data.get("snapshotItems")
    if items is None and isinstance(data.get("autoReview"), dict):
        items = data["autoReview"].get("snapshotItems")
    if not isinstance(items, list) or not items:
        return False
    if any(not isinstance(item, dict) or item.get("action") not in {"Create", "Change", "Delete"}
           or not item.get("word") or not item.get("code") for item in items):
        return False
    # A move must retain the same word and dictionary; an unpaired delete stays manual.
    creates = [item for item in items if item["action"] == "Create"]
    for item in items:
        if item["action"] != "Delete":
            continue
        match = next((created for created in creates
                      if created["word"] == item["word"]
                      and created.get("type", "Phrase") == item.get("type", "Phrase")
                      and created["code"] != item["code"]), None)
        if match is None:
            return False
        creates.remove(match)
    return True


def submitted_vote_receipt(data: dict) -> str | None:
    if (not enabled() or not isinstance(data, dict) or data.get("success") is not True
            or data.get("autoApproved") or data.get("requiresConfirmation")
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(data.get("batchId") or ""))
            or not isinstance(data.get("batchUrl"), str) or not data["batchUrl"]
            or type(data.get("contentVersion")) is not int or data["contentVersion"] < 0
            or not _eligible_snapshot(data)):
        return None
    install_delivery_hook()
    frozen = json.loads(json.dumps({key: data[key] for key in (
        "success", "batchId", "batchUrl", "contentVersion", "snapshotItems", "autoReview",
    ) if key in data}, ensure_ascii=False))
    receipts = _pending_receipts.get()
    if receipts is None:
        receipts = []
        _pending_receipts.set(receipts)
    # Background actions run in child tasks; update their turn's shared collector in place.
    receipts[:] = [item for item in receipts if item["batchId"] != data["batchId"]]
    receipts.append(frozen)
    return (f"批次已加入提交，审核中。\n草稿地址：{data['batchUrl']}\n\n"
            "当前投票：0/3\n可以引用该条消息，回复同意来投票通过")


def submitted_vote_receipts(receipts) -> str | None:
    if len({receipt.get("batchId") for receipt in receipts
            if receipt.get("tool") == "keytao_submit_batch"}) > 1:
        return None
    replies = [submitted_vote_receipt(receipt) for receipt in receipts
               if receipt.get("tool") == "keytao_submit_batch"]
    return "\n\n".join(dict.fromkeys(filter(None, replies))) or None


def prepare_delivery(text: str, memory_context) -> str:
    if not enabled() or memory_context is None:
        return text
    from ..plugins.chat_render import render_platform_public_links
    replies = []
    for data in tuple(_pending_receipts.get() or ()):
        url = render_platform_public_links(data["batchUrl"], memory_context.platform)
        if url in text or data["batchUrl"] in text:
            replies.append(submitted_vote_receipt(data))
    if len(replies) > 1:
        # A native quote identifies one message, so it cannot select between batches.
        _pending_receipts.get().clear()
        return "\n\n".join(reply.split("\n\n", 1)[0].replace(
            "批次已加入提交，审核中。", "✅ 批次已提交审核。") for reply in replies)
    return "\n\n".join(dict.fromkeys(filter(None, replies))) or text


class CommunityVoteStore:
    def __init__(self, db_path=None):
        self.db_path = str(Path(db_path or Path(__file__).resolve().parents[2]
                               / "data" / "community_votes.db").resolve())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            columns = conn.execute("PRAGMA table_info(community_votes)").fetchall()
            legacy = any(row["name"] == "content_version" and not row["pk"] for row in columns)
            if legacy:
                conn.execute("ALTER TABLE community_votes RENAME TO community_votes_legacy")
            conn.execute("""CREATE TABLE IF NOT EXISTS community_votes (
                batch_id TEXT NOT NULL, creator_platform TEXT NOT NULL,
                creator_id TEXT NOT NULL, creator_user_id TEXT,
                content_version INTEGER NOT NULL, batch_url TEXT NOT NULL,
                voters TEXT NOT NULL DEFAULT '[]', status TEXT NOT NULL DEFAULT 'open',
                attempts INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (batch_id, content_version))""")
            if legacy:
                conn.execute("INSERT INTO community_votes SELECT * FROM community_votes_legacy")
                conn.execute("DROP TABLE community_votes_legacy")
            conn.execute("""CREATE TABLE IF NOT EXISTS community_vote_receipts (
                platform TEXT NOT NULL, group_id TEXT NOT NULL, message_id TEXT NOT NULL,
                batch_id TEXT NOT NULL, content_version INTEGER NOT NULL,
                PRIMARY KEY (platform, group_id, message_id))""")
            if not any(row["name"] == "content_version" for row in conn.execute(
                    "PRAGMA table_info(community_vote_receipts)")):
                conn.execute("ALTER TABLE community_vote_receipts ADD COLUMN content_version INTEGER")
                conn.execute("UPDATE community_vote_receipts SET content_version=(SELECT content_version "
                             "FROM community_votes WHERE community_votes.batch_id=community_vote_receipts.batch_id)")
            conn.execute("""CREATE TABLE IF NOT EXISTS community_vote_notices (
                batch_id TEXT NOT NULL, content_version INTEGER NOT NULL,
                platform TEXT NOT NULL, user_id TEXT NOT NULL,
                PRIMARY KEY (batch_id, content_version, platform, user_id))""")
        os.chmod(self.db_path, 0o600)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA secure_delete=ON")
            with conn:
                yield conn
        finally:
            conn.close()

    def bind_receipt(self, data, platform, creator_id, group_id, message_id):
        if not submitted_vote_receipt(data) or not all((platform, creator_id, group_id, message_id)):
            return False
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM community_votes WHERE batch_id=? ORDER BY content_version DESC LIMIT 1",
                               (data["batchId"],)).fetchone()
            if row and (row["creator_platform"], row["creator_id"]) != (platform, str(creator_id)):
                return False
            status = "ended" if row and row["content_version"] > data["contentVersion"] else "open"
            conn.execute("UPDATE community_votes SET status='ended' WHERE batch_id=? AND content_version<?",
                         (data["batchId"], data["contentVersion"]))
            conn.execute("INSERT OR IGNORE INTO community_votes "
                         "(batch_id,creator_platform,creator_id,content_version,batch_url,status) VALUES (?,?,?,?,?,?)",
                         (data["batchId"], platform, str(creator_id), data["contentVersion"], data["batchUrl"], status))
            conn.execute("INSERT OR IGNORE INTO community_vote_receipts VALUES (?,?,?,?,?)",
                         (platform, str(group_id), str(message_id), data["batchId"], data["contentVersion"]))
        return True

    def resolve(self, platform, group_id, message_id):
        round_key = self.resolve_round(platform, group_id, message_id)
        return round_key[0] if round_key else None

    def resolve_round(self, platform, group_id, message_id):
        with self._connect() as conn:
            row = conn.execute("SELECT batch_id,content_version FROM community_vote_receipts "
                               "WHERE platform=? AND group_id=? AND message_id=?",
                               (platform, str(group_id), str(message_id))).fetchone()
        return tuple(row) if row else None

    def get(self, batch_id, content_version=None):
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM community_votes WHERE batch_id=? "
                               "AND (? IS NULL OR content_version=?) ORDER BY content_version DESC LIMIT 1",
                               (batch_id, content_version, content_version)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["voters"] = json.loads(data["voters"])
        return data

    def save(self, data):
        with self._connect() as conn:
            result = conn.execute("UPDATE community_votes SET creator_user_id=?,voters=?,status=?,attempts=? "
                                  "WHERE batch_id=? AND content_version=? AND status='open'",
                                  (data["creator_user_id"], json.dumps(data["voters"], ensure_ascii=False),
                                   data["status"], data["attempts"], data["batch_id"], data["content_version"]))
        return result.rowcount == 1

    def notify_once(self, data, platform, user_id, message):
        with self._connect() as conn:
            result = conn.execute("INSERT OR IGNORE INTO community_vote_notices VALUES (?,?,?,?)",
                                  (data["batch_id"], data["content_version"], platform, str(user_id)))
        return message if result.rowcount else ""


def get_store():
    global _default_store
    if _default_store is None:
        _default_store = CommunityVoteStore()
    return _default_store


def _account_id(user):
    value = user.get("id") if isinstance(user, dict) else None
    return str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value) else None


def _escape_name(name):
    return str(name).replace("&", "&amp;").replace("[", "&#91;").replace("]", "&#93;")


async def record_vote(store, platform, group_id, message_id, user_id, name, *, lookup, approve):
    if not enabled():
        return None
    round_key = store.resolve_round(platform, group_id, message_id)
    if round_key is None:
        return None
    batch_id, content_version = round_key
    arrival_attempt = store.get(*round_key)["attempts"]
    key = (store.db_path, batch_id)
    lock = _locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[key] = lock
    was_waiting = lock.locked()
    async with lock:
        data = store.get(batch_id, content_version)
        if data["status"] != "open":
            return "投票已结束。"
        if (platform, str(user_id)) == (data["creator_platform"], data["creator_id"]):
            return store.notify_once(data, platform, user_id, "提交人不能为自己投票")
        account_id = _account_id(await lookup(platform, str(user_id)))
        if store.get(*round_key)["status"] != "open":
            return "投票已结束。"
        if account_id is None:
            return store.notify_once(data, platform, user_id, "请先绑定键道账号再投票")
        creator_id = data["creator_user_id"] or _account_id(
            await lookup(data["creator_platform"], data["creator_id"]))
        if store.get(*round_key)["status"] != "open":
            return "投票已结束。"
        if creator_id is None:
            return "提交人账号暂时无法核验，请稍后重试"
        data["creator_user_id"] = creator_id
        if account_id == creator_id:
            return store.notify_once(data, platform, user_id, "提交人不能为自己投票")
        count = len(data["voters"])
        if count < 3:
            if any(voter["userId"] == account_id for voter in data["voters"]):
                return store.notify_once(data, platform, user_id, f"你已经投过了（{count}/3）")
            data["voters"].append({"platform": platform, "platformId": str(user_id),
                                   "userId": account_id, "name": name or str(user_id)})
            if not store.save(data):
                return "投票已结束。"
        names = "、".join(_escape_name(voter["name"]) for voter in data["voters"])
        if len(data["voters"]) < 3:
            return (f"@{_escape_name(name or user_id)} 已记录同意。\n草稿地址：{data['batch_url']}\n"
                    f"当前投票：{len(data['voters'])}/3\n同意：{names}")
        pending = ("投票已达 3/3，批准待确认，请等待管理员处理。" if data["attempts"] >= 2 else
                   "投票已达 3/3，批准待确认，请再次回复同意重试。")
        pending += f"\n草稿地址：{data['batch_url']}\n同意：{names}"
        if data["attempts"] >= 2:
            return store.notify_once(data, platform, user_id, pending)
        # A concurrent arrival cannot retry an attempt that was still in progress when it arrived.
        if data["attempts"] != arrival_attempt or (was_waiting and data["attempts"] > 0):
            return pending
        data["attempts"] += 1
        if not store.save(data):
            return "投票已结束。"
        if data["attempts"] == 2:
            pending = pending.replace("请再次回复同意重试", "请等待管理员处理")
        try:
            result = await approve(data["creator_platform"], data["creator_id"], batch_id,
                                   [{"platform": voter["platform"], "platformId": voter["platformId"]}
                                    for voter in data["voters"]], data["content_version"])
        except Exception:
            result = {"retryable": True}
        if store.get(*round_key)["status"] != "open":
            return "投票已结束。"
        if result.get("success") is True:
            data["status"] = "approved"
            if not store.save(data):
                return "投票已结束。"
            server_names = result.get("voters")
            if isinstance(server_names, list) and len(server_names) == 3 and all(isinstance(value, str) and value for value in server_names):
                names = "、".join(_escape_name(value) for value in server_names)
            return f"投票通过 3/3，批次已加入词库，投票结束。\n草稿地址：{data['batch_url']}\n同意：{names}"
        if result.get("retryable"):
            return store.notify_once(data, platform, user_id, pending) if data["attempts"] >= 2 else pending
        data["status"] = "ended"
        store.save(data)
        return "投票已结束：" + str(result.get("message") or "服务器拒绝批准")


def install_delivery_hook():
    global _hook_installed
    if _hook_installed:
        return
    from nonebot.adapters import Bot
    register = getattr(Bot, "on_called_api", None)
    if callable(register):
        register(_record_sent_receipts)
        _hook_installed = True


async def _record_sent_receipts(bot, exception, api, data, result):
    if not enabled() or exception is not None or api not in {"send_msg", "send_group_msg", "send_message"}:
        return
    from ..plugins.chat_commands import current_memory_context
    memory = current_memory_context.get()
    if memory is None or memory.space_type != "group":
        return
    if api == "send_message":
        platform = "telegram"
        group_id = str(data.get("chat_id") or "")
        if data.get("message_thread_id"):
            group_id += f":thread:{data['message_thread_id']}"
        text = re.sub(r"\\([\\_%*\[\]()~`>#+\-=|{}.!])", r"\1", str(data.get("text") or ""))
    else:
        platform = "qq"
        group_id = str(data.get("group_id") or "")
        message = data.get("message", "")
        text = message.extract_plain_text() if hasattr(message, "extract_plain_text") else str(message)
    if (platform, group_id) != (memory.platform, memory.space_id):
        return
    message_id = result.get("message_id") if isinstance(result, dict) else getattr(result, "message_id", None)
    if message_id is None:
        return
    try:
        for receipt in tuple(_pending_receipts.get() or ()):
            expected = submitted_vote_receipt(receipt)
            if expected and expected in text:
                get_store().bind_receipt(receipt, platform, memory.user_id, group_id, str(message_id))
    except (OSError, sqlite3.Error):
        logger.warning("Failed to persist a sent community vote receipt")


def _is_assent(text):
    text = re.sub(r"\[CQ:at,[^\]]+\]|@[\w-]+", "", text).strip()
    while text and unicodedata.category(text[-1]).startswith("P"):
        text = text[:-1].rstrip()
    return re.sub(r"\s+", "", text) == "同意"


async def _stage_community_vote(ctx):
    _pending_receipts.set([])
    if not enabled():
        return False
    install_delivery_hook()
    memory = ctx.memory_context
    if (memory is None or memory.space_type != "group" or not ctx.reply_reference.is_reply
            or not _is_assent(ctx.message_text)):
        return False
    from ..plugins import openai_chat as chat
    if ctx.platform == "qq":
        message_id = chat.extract_onebot_reply_id(ctx.event)
    elif ctx.platform == "telegram":
        message_id = getattr(getattr(ctx.event, "reply_to_message", None), "message_id", None)
    else:
        return False
    if message_id is None or get_store().resolve(ctx.platform, memory.space_id, message_id) is None:
        return False
    approve = chat.skills_manager.get_tool_function("_community_approve_submitted_batch")
    if approve is None:
        ctx.response = "投票批准暂时不可用，请稍后重试"
    else:
        ctx.response = await record_vote(get_store(), ctx.platform, memory.space_id, message_id,
                                         ctx.user_id, ctx.owner_label,
                                         lookup=find_user_by_platform, approve=approve)
    if ctx.response:
        await chat._finish_ai_chat_response(ctx.bot, ctx.event, ctx.user_id, memory, ctx.response, ctx.QQMessageSegment)
    return True
