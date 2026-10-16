"""Bounded, local word detection; offers use the normal add-word capability."""
import asyncio
import os
import re
import sqlite3
import time
from collections import OrderedDict
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from nonebot.log import logger

from .conversation import ConversationAddress
from ..utils.pinyin_reference import reference_db_path
from ..utils.pending_confirmation import PENDING_CONFIRM_ADVERTISED_FORMS


# Frequency >= 50 only reached basic words KeyTao already has; neologisms (内卷, 发薪日) sit at 3.
# Listing in >= 2 reference dictionaries keeps out made-up strings.
DISCOVERY_MIN_COMMONNESS = 3
DISCOVERY_MIN_PRESENCE = 2
MAX_MESSAGE_CHARS = 256
MAX_CANDIDATES = 12
PRESENT_CACHE_MAX_WORDS = 5000
PRESENT_CACHE_TTL = 24 * 60 * 60
_COMMON_POS = frozenset({"n", "v", "a", "d", "i", "l", "vn", "an", "t", "b", "z"})
_CHINESE = re.compile(r"[\u4e00-\u9fff]+")
_COMMAND = re.compile(
    r"^(?:[/!！.#$]|(?:(?:请|麻烦|帮我|帮忙|给我|替我)\s*)*(?:喵喵|加词|添加|加入|删除|删词|换码|改码|查词|查询|"
    r"查一下|查下|编码|词频|帮助|提交|撤回|清空|查看草稿|确认|取消|选编号|选\s*\d))"
)
_OPT_OUT = frozenset({"别问我", "不要再问我", "别再问我", "以后别问我", "以后不要再问我"})
_DAY_ZONE = ZoneInfo("Asia/Shanghai")


def extract_candidates(text, db_path=None):
    """Greedy longest matching, with indexed probes and no corpus-sized cache."""
    if not text or len(text) > MAX_MESSAGE_CHARS:
        return []
    path = reference_db_path(db_path).resolve()
    candidates = []
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True, timeout=0.1)) as conn:
        conn.execute("PRAGMA cache_size=-512")
        for run in _CHINESE.findall(text):
            index = 0
            while index < len(run):
                probes = [run[index:index + size] for size in range(2, min(16, len(run) - index) + 1)]
                row = conn.execute(
                    "SELECT word, corpus_frequency, part_of_speech, dictionary_presence_count "
                    "FROM word_commonness WHERE word IN (" + ",".join("?" for _ in probes) + ") "
                    "ORDER BY length(word) DESC LIMIT 1", probes,
                ).fetchone() if probes else None
                if row is None:
                    index += 1
                    continue
                word, frequency, pos, presence = row
                index += len(word)
                if (2 <= len(word) <= 4 and (frequency or 0) >= DISCOVERY_MIN_COMMONNESS
                        and (presence or 0) >= DISCOVERY_MIN_PRESENCE and pos in _COMMON_POS and word not in candidates):
                    candidates.append(word)
                    if len(candidates) == MAX_CANDIDATES:
                        return candidates
    return candidates


class DiscoveryStore:
    """Permanent group-word dedupe and actor limits, independent of ticket TTL."""

    def __init__(self, path, clock=time.time):
        self.path = Path(path)
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._connect() as conn:
            # Readers must not make a successfully delivered offer fail to commit.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS discovery_offers (
                    platform TEXT NOT NULL, group_id TEXT NOT NULL, word TEXT NOT NULL,
                    actor_id TEXT NOT NULL, day TEXT NOT NULL, nonce TEXT NOT NULL DEFAULT '',
                    message_id TEXT NOT NULL DEFAULT '',
                    PRIMARY KEY (platform, group_id, word)
                ) WITHOUT ROWID;
                CREATE INDEX IF NOT EXISTS discovery_actor_day
                    ON discovery_offers(platform, actor_id, day);
                CREATE TABLE IF NOT EXISTS discovery_opt_out (
                    platform TEXT NOT NULL, actor_id TEXT NOT NULL,
                    PRIMARY KEY (platform, actor_id)
                ) WITHOUT ROWID;
            """)
        self.path.chmod(0o600)

    @contextmanager
    def _connect(self, timeout=0.1):
        conn = sqlite3.connect(self.path, timeout=timeout)
        try:
            conn.execute("PRAGMA cache_size=-256")
            with conn:
                yield conn
        finally:
            conn.close()

    def _eligible(self, conn, address, word):
        if conn.execute("SELECT 1 FROM discovery_opt_out WHERE platform=? AND actor_id=?",
                        address.actor_key).fetchone():
            return False
        day = datetime.fromtimestamp(self.clock(), _DAY_ZONE).date().isoformat()
        count = conn.execute(
            "SELECT COUNT(*) FROM discovery_offers WHERE platform=? AND actor_id=? AND day=?",
            (*address.actor_key, day),
        ).fetchone()[0]
        return count < 3 and (not word or not conn.execute(
            "SELECT 1 FROM discovery_offers WHERE platform=? AND group_id=? AND word=?",
            (address.platform, address.space_id, word),
        ).fetchone())

    def eligible(self, address, word=""):
        with self._connect() as conn:
            return self._eligible(conn, address, word)

    def claim(self, address, word):
        with self.sending_offer(address, word) as conn:
            if conn is None:
                return False
            self.record_offer(conn, address, word)
            return True

    @contextmanager
    def sending_offer(self, address, word, *, timeout=0.1):
        """Serialize eligibility through delivery; failed sends record nothing."""
        with self._connect(timeout=timeout) as conn:
            conn.execute("BEGIN IMMEDIATE")
            yield conn if self._eligible(conn, address, word) else None

    def record_offer(self, conn, address, word, nonce="", message_id=""):
        day = datetime.fromtimestamp(self.clock(), _DAY_ZONE).date().isoformat()
        conn.execute("INSERT INTO discovery_offers(platform,group_id,word,actor_id,day,nonce,message_id) "
                     "VALUES(?,?,?,?,?,?,?)",
                     (address.platform, address.space_id, word, address.actor_id, day, nonce, str(message_id)))

    def bind_offer(self, address, word, nonce, message_id=""):
        with self._connect() as conn:
            conn.execute("UPDATE discovery_offers SET nonce=?,message_id=? "
                         "WHERE platform=? AND group_id=? AND word=? AND actor_id=?",
                         (nonce, str(message_id), address.platform, address.space_id, word, address.actor_id))

    def is_offer(self, address, nonce="", message_id=""):
        if not nonce and not message_id:
            return False
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM discovery_offers WHERE platform=? AND group_id=? AND actor_id=? "
                "AND (?='' OR nonce=?) AND (?='' OR message_id=?) LIMIT 1",
                (address.platform, address.space_id, address.actor_id, nonce, nonce,
                 str(message_id), str(message_id)),
            ).fetchone() is not None

    def is_offer_message(self, address, message_id):
        """Recognize discovery quotes before applying the actor-owned gate."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT 1 FROM discovery_offers WHERE platform=? AND group_id=? AND message_id=? LIMIT 1",
                (address.platform, address.space_id, str(message_id)),
            ).fetchone() is not None

    def opt_out(self, address):
        # This runs in a worker thread and may wait for an in-flight delivery.
        with self._connect(timeout=60) as conn:
            conn.execute("INSERT OR IGNORE INTO discovery_opt_out VALUES(?,?)", address.actor_key)

    def opted_out(self, address):
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM discovery_opt_out WHERE platform=? AND actor_id=?",
                                address.actor_key).fetchone() is not None


def _foreground_busy(chat, address):
    return any(store.is_locked(key) for store, key in (
        (chat.conversation_space_message_locks, chat._conversation_scope_barrier_key(address)),
        (chat.conversation_message_locks, address),
        (chat.draft_actor_message_locks, ConversationAddress.private(*address.actor_key)),
    ))


async def _prepare_offer(chat, word, address):
    """Use exactly the normal add flow's reviewed prompt, parser and snapshot."""
    import json

    commands = chat._chat_commands
    review = json.loads(await chat.call_tool_function(
        "keytao_prepare_reviewed_add", {"word": word}, *address.actor_key,
    ))
    if (not isinstance(review, dict) or review.get("success") is not True
            or review.get("word") != word or review.get("pronunciationUnresolved")
            or review.get("existing") or review.get("reviewDisposition") == "BLOCK"
            or str(review.get("type") or "Phrase") != "Phrase"):
        return None
    prompt = commands._format_reviewed_add_prompt(review)
    inventory = commands.select_candidate_inventory(review)
    pending = commands._parse_pending_add_word(prompt) if prompt else None
    if (pending is None or pending.word != word or inventory is None
            or not pending.recommended_code or any(not inventory.readings.get(code)
                                                  for code, _ in inventory.candidates)):
        return None
    pending.phrase_type = str(review.get("type") or "Phrase")
    commands._attach_server_candidate_snapshot(
        pending, [dict(status) for status in inventory.statuses], review.get("candidateOrderingAssessments"),
    )
    pending.pronunciation_codes = dict(inventory.readings)
    pending.pronunciation_recommended_codes = [inventory.group_recommended_code or pending.recommended_code]
    return pending


class WordDiscovery:
    def __init__(self, store=None, reference_path=None):
        self.store = store
        self.reference_path = reference_path
        self.task = None
        self._store_lock = asyncio.Lock()
        self._preference_generation = 0
        self._present_words = OrderedDict()
        self._sending_nonces = set()

    def _known_present(self, word):
        expires = self._present_words.get(word)
        if expires is None:
            return False
        if expires <= time.monotonic():
            del self._present_words[word]
            return False
        self._present_words.move_to_end(word)
        return True

    async def _ensure_store(self, chat):
        async with self._store_lock:
            if self.store is None:
                path = Path(chat.conversation_state_store.db_path).with_name("word_discovery.db")
                self.store = await asyncio.to_thread(DiscoveryStore, path)

    def schedule(self, bot, event, chat, *, addressed=False):
        if os.getenv("KEYTAO_WORD_DISCOVERY") != "1":
            return False
        # Preference replies are handled by the normal message rule, even when busy.
        if event.get_plaintext().strip().rstrip("。！! ") in _OPT_OUT:
            return False
        if self.task is not None and not self.task.done():
            return False
        self.task = asyncio.create_task(self._run(bot, event, chat, addressed))
        return True

    async def _handle_opt_out(self, bot, event, chat):
        text = event.get_plaintext().strip().rstrip("。！! ")
        if text not in _OPT_OUT:
            return False
        user_id = str(event.get_user_id())
        group_id = str(getattr(event, "group_id", "") or "")
        if not group_id or user_id == str(getattr(bot, "self_id", "")):
            return True
        address = ConversationAddress.group("qq", group_id, user_id)
        record = chat.conversation_state_store.get_record(address)
        reply_id = str(getattr(getattr(event, "reply", None), "message_id", "") or "")
        if not reply_id and not (getattr(event, "to_me", False)
                                 or chat.GROUP_TRIGGER_KEYWORD_ANY in text
                                 or text.startswith(chat.GROUP_TRIGGER_KEYWORD_START)):
            return False
        await self._ensure_store(chat)
        if await asyncio.to_thread(self.store.is_offer, address, record.nonce if record and not reply_id else "", reply_id):
            self._preference_generation += 1
            await asyncio.to_thread(self.store.opt_out, address)
            if (record is not None and chat.conversation_state_store.get_record(address) is record
                    and await asyncio.to_thread(self.store.is_offer, address, record.nonce)):
                # Recheck after the last await; never delete a replacement ticket.
                if chat.conversation_state_store.get_record(address) is not record:
                    return True
                chat.conversation_state_store.delete(address)
            return True
        return False

    async def _run(self, bot, event, chat, addressed):
        try:
            async with asyncio.timeout(60):
                await self.discover(bot, event, chat, addressed=addressed)
        except Exception as error:
            logger.warning("[word_discovery] skipped: %s", type(error).__name__)

    async def discover(self, bot, event, chat, *, addressed=False):
        import json

        if os.getenv("KEYTAO_WORD_DISCOVERY") != "1":
            return
        user_id = event.get_user_id()
        group_id = str(getattr(event, "group_id", "") or "")
        if not group_id or str(user_id) in {str(getattr(bot, "self_id", "")), str(getattr(event, "self_id", ""))}:
            return
        if getattr(getattr(event, "sender", None), "is_bot", False):
            return
        address = ConversationAddress.group("qq", group_id, str(user_id))
        text = event.get_plaintext().strip()
        if await self._handle_opt_out(bot, event, chat):
            return
        reply = getattr(event, "reply", None)
        if (addressed or getattr(event, "to_me", False) or reply is not None
                or not text or len(text) > MAX_MESSAGE_CHARS or _COMMAND.search(text)
                or any(segment.type != "text" for segment in event.get_message())
                or any(mark in text for mark in ("转发", "[CQ:", "\"", "'", "`", "“", "”", "‘", "’", "「", "」", ">"))
                or _foreground_busy(chat, address)
                or chat.conversation_state_store.has_pending_for_actor(address.actor_key)
                or chat.draft_operation_coordinator.find_for_actor(address.actor_key) is not None):
            return
        await self._ensure_store(chat)
        generation = self._preference_generation

        def find_candidates():
            if not self.store.eligible(address):
                return []
            return [word for word in extract_candidates(text, self.reference_path)
                    if self.store.eligible(address, word)]

        candidates = [word for word in await asyncio.to_thread(find_candidates)
                      if not self._known_present(word)]
        if not candidates or os.getenv("KEYTAO_WORD_DISCOVERY") != "1":
            return
        lookup = json.loads(await chat.call_tool_function(
            "keytao_lookup_by_words_batch", {"words": candidates}, *address.actor_key,
        ))
        if not isinstance(lookup, dict) or lookup.get("success") is not True:
            return
        rows = lookup.get("results")
        if not isinstance(rows, list):
            return
        # Missing/malformed rows are unknown, never evidence of dictionary absence.
        word = ""
        for candidate in candidates:
            matches = [row for row in rows if isinstance(row, dict) and row.get("word") == candidate]
            if len(matches) != 1 or not isinstance(phrases := matches[0].get("phrases"), list):
                continue
            if phrases and matches[0].get("success") is True:
                self._present_words[candidate] = time.monotonic() + PRESENT_CACHE_TTL
                self._present_words.move_to_end(candidate)
                if len(self._present_words) > PRESENT_CACHE_MAX_WORDS:
                    self._present_words.popitem(last=False)
            elif phrases == [] and not word:
                word = candidate
        logger.info(f"[word_discovery] candidates={len(candidates)} missing={word or '-'}")
        if not word:
            return
        if (os.getenv("KEYTAO_WORD_DISCOVERY") != "1" or _foreground_busy(chat, address)
                or chat.conversation_state_store.has_pending_for_actor(address.actor_key)):
            return
        pending = await _prepare_offer(chat, word, address)
        if pending is None or await asyncio.to_thread(self.store.opted_out, address):
            return
        if (os.getenv("KEYTAO_WORD_DISCOVERY") != "1"
                or generation != self._preference_generation or _foreground_busy(chat, address)
                or chat.conversation_state_store.has_pending_for_actor(address.actor_key)
                or chat.draft_operation_coordinator.find_for_actor(address.actor_key) is not None):
            return
        async with (
            chat.conversation_space_message_locks.lock(chat._conversation_scope_barrier_key(address)),
            chat.conversation_message_locks.lock(address),
            chat.draft_actor_message_locks.lock(ConversationAddress.private(*address.actor_key)),
        ):
            if chat.conversation_state_store.has_pending_for_actor(address.actor_key):
                return
            if not chat.conversation_state_store.set(address, pending, space_key=address.space_key):
                return
            record = chat.conversation_state_store.get_record(address)
            rendered = chat._render_live_single_candidate_record(record)
            if not rendered:
                chat.conversation_state_store.delete(address)
                return
            response = f"「{word}」词库里还没有，要加入吗？\n{rendered}\n回复：确认 / 选编号 / 取消；以后不想收到询问可回复“别问我”。"
            if not chat.conversation_state_store.bind_origin_prompt_digest(
                address, chat._prompt_capability_digest(response),
            ):
                chat.conversation_state_store.delete(address)
                return
        from nonebot.adapters.onebot.v11 import MessageSegment

        self._sending_nonces.add(record.nonce)
        try:
            if (chat.conversation_state_store.get_record(address) is not record
                    or generation != self._preference_generation or _foreground_busy(chat, address)
                    or os.getenv("KEYTAO_WORD_DISCOVERY") != "1"):
                if chat.conversation_state_store.get_record(address) is record:
                    chat.conversation_state_store.delete(address)
                return
            with self.store.sending_offer(address, word, timeout=0) as conn:
                if conn is None:
                    if chat.conversation_state_store.get_record(address) is record:
                        chat.conversation_state_store.delete(address)
                    return
                result = await bot.send(event=event, message=MessageSegment.at(user_id) + " " + response)
                message_id = result.get("message_id", "") if isinstance(result, dict) else ""
                self.store.record_offer(conn, address, word, record.nonce, message_id)
        except BaseException as error:
            if chat.conversation_state_store.get_record(address) is record:
                chat.conversation_state_store.delete(address)
            if (isinstance(error, sqlite3.OperationalError)
                    and getattr(error, "sqlite_errorcode", None) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}):
                return
            raise
        finally:
            self._sending_nonces.discard(record.nonce)
        if chat.conversation_state_store.get_record(address) is record:
            chat.add_to_history(address, text, response)


_discovery = WordDiscovery()


def schedule_discovery(bot, event, chat, *, addressed=False):
    if os.getenv("KEYTAO_WORD_DISCOVERY") != "1":
        return False
    try:
        return _discovery.schedule(bot, event, chat, addressed=addressed)
    except Exception as error:
        logger.warning("[word_discovery] not scheduled: %s", type(error).__name__)
        return False


async def discovery_reply_route(bot, event, chat):
    """Admit bound discovery controls while preserving vote-receipt priority."""
    if os.getenv("KEYTAO_WORD_DISCOVERY") != "1":
        return None
    try:
        from ..utils import community_vote

        vote_reply_id = chat.extract_onebot_reply_id(event)
        if (community_vote.enabled() and vote_reply_id is not None
                and community_vote._is_assent(event.get_plaintext())
                and community_vote.get_store().resolve("qq", str(event.group_id), vote_reply_id) is not None):
            return True
        if await _discovery._handle_opt_out(bot, event, chat):
            return False
        address = ConversationAddress.group("qq", str(event.group_id), str(event.get_user_id()))
        record = chat.conversation_state_store.get_record(address)
        from .state import PendingAddWord

        reply = getattr(event, "reply", None)
        reply_id = str(getattr(reply, "message_id", "") or "")
        if record is not None and record.nonce in _discovery._sending_nonces:
            return False
        if record is None and not reply_id:
            return None
        await _discovery._ensure_store(chat)
        quoted_offer = reply_id and await asyncio.to_thread(
            _discovery.store.is_offer_message, address, reply_id,
        )
        if (record is None or not isinstance(record.state, PendingAddWord)
                or not await asyncio.to_thread(_discovery.store.is_offer, address, record.nonce)):
            return False if quoted_offer else None
        message_text = event.get_plaintext().strip()
        text = message_text
        for prefix in ("@" + chat.GROUP_TRIGGER_KEYWORD_ANY, "@" + chat.GROUP_TRIGGER_KEYWORD_START,
                       chat.GROUP_TRIGGER_KEYWORD_ANY, chat.GROUP_TRIGGER_KEYWORD_START):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip(" \t:：，,")
                break
        affirmative = chat._chat_routing.normalize_affirmative_reply(text)
        if affirmative:
            text = affirmative
        answers = {*chat._chat_routing.AFFIRMATIVE_REPLY_TEXTS, *PENDING_CONFIRM_ADVERTISED_FORMS, "取消"}
        candidates = record.state.server_candidates
        answers.update(str(index) for index in range(1, len(candidates) + 1))
        answers.update(code for code, _ in candidates)
        addressed = (reply is None or text not in answers) and (
            await chat.to_me()(bot, event, {})
            or chat.GROUP_TRIGGER_KEYWORD_ANY in message_text
            or message_text.startswith(chat.GROUP_TRIGGER_KEYWORD_START)
        )
        if text not in answers:
            if not addressed:
                return False
            # Retire this exact offer before normal routing can infer assent.
            if (not chat.conversation_state_store.complete_execution(record)
                    and chat.conversation_state_store.get_record(address) is record):
                return False
            return None
        if reply is not None:
            if not reply_id or not await asyncio.to_thread(
                _discovery.store.is_offer, address, record.nonce, reply_id,
            ):
                return False
        elif not addressed:
            return False
        if affirmative and reply is None:
            latest = chat._get_latest_assistant_message(chat.get_history(address))
            if not record.origin_prompt_digest or chat._prompt_capability_digest(latest) != record.origin_prompt_digest:
                return False
        if record.execution_id:
            return False
        intent = (chat._chat_routing._closed_candidate_selection(text)
                  or chat._chat_routing._pending_tool_assent_intent(record.state, "确认" if affirmative else text))
        return intent is not None and chat.conversation_state_store.get_record(address) is record
    except Exception as error:
        logger.warning("[word_discovery] reply skipped: %s", type(error).__name__)
        return False
