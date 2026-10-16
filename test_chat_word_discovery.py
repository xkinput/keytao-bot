"""Offline coverage for bounded group-chat word discovery and normal tickets."""

import asyncio
import copy
import json
import os
import sqlite3
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import test_state_machine as harness

from keytao_bot.harness.conversation import ConversationAddress
from keytao_bot.harness.state import (
    DraftOperationCoordinator,
    MemoryConversationStateStore,
    PendingAddWord,
)
from keytao_bot.harness import word_discovery as discovery


def make_reference(path):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE word_commonness (word TEXT PRIMARY KEY, "
                     "corpus_frequency INTEGER, part_of_speech TEXT, "
                     "dictionary_presence_count INTEGER)")
        conn.execute("CREATE TABLE bcc_frequency (dataset TEXT, token TEXT, raw_count INTEGER, "
                     "PRIMARY KEY(dataset, token)) WITHOUT ROWID")
        conn.executemany("INSERT INTO word_commonness VALUES (?, ?, ?, ?)", [
            (word, discovery.DISCOVERY_MIN_COMMONNESS + 10, "n", 2)
            for word in ("午时", "阳光", "清风", "笑声")
        ] + [
            ("冷词", discovery.DISCOVERY_MIN_COMMONNESS - 1, "n", 2),
            ("张三", 100000, "nr", 2),
            ("台北", 100000, "ns", 2),
            ("怪词", 100000, "n", 0),
            ("好", 100000, "a", 2),
            ("AI", 100000, "n", 2),
            ("春夏秋冬天", 100000, "n", 2),
        ])


def add_bcc(path, word="燃面", counts=(60, 40)):
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executemany("INSERT INTO bcc_frequency VALUES (?, ?, ?)", [
            (dataset, word, count)
            for dataset, count in zip(discovery.BCC_DATASETS, counts)
        ])


def learning_messages(word="晕碳"):
    # Unique background characters make the five target occurrences cohesive.
    for index, actor in enumerate(("100", "101", "102", "100", "101")):
        background = "".join(chr(0x5200 + index * 40 + offset) for offset in range(36))
        yield background[:18] + word + background[18:], actor


def reviewed(word):
    return {
        "success": True, "word": word, "type": "Phrase",
        "recommendedCode": "wj", "needsManualReview": False,
        "preSubmitAudit": {
            "success": True, "verdict": "pass", "autoApprove": True,
            "summary": "Offline reviewed fixture", "issues": [],
        },
        "pronunciations": [{
            "pinyin": "wu shi", "recommendedCode": "wj",
            "sources": [{"source": "offline"}],
            "candidateStatuses": [
                {"code": "wj", "occupied": False, "label": "空位", "words": []},
                {"code": "wja", "occupied": False, "label": "空位", "words": []},
            ],
        }],
    }


class FakeMessage(list):
    def __init__(self, text, kind="text"):
        super().__init__([types.SimpleNamespace(type=kind, data={"text": text})])


class FakeEvent:
    def __init__(self, text="午时", actor="100", group="200", kind="text"):
        self.text = text
        self.user_id = actor
        self.group_id = group
        self.message_id = "300"
        self.self_id = "999"
        self.to_me = False
        self.reply = None
        self.sender = types.SimpleNamespace(is_bot=False, nickname="Offline sender", card="")
        self.message = FakeMessage(text, kind)

    def get_user_id(self):
        return self.user_id

    def get_plaintext(self):
        return self.text

    def get_message(self):
        return self.message


@contextmanager
def qq_runtime(chat, bot):
    qq = types.ModuleType("nonebot.adapters.onebot.v11")
    qq.Bot = type("QQBot", (types.SimpleNamespace,), {})
    qq.MessageSegment = types.SimpleNamespace(at=lambda actor: f"[CQ:at,qq={actor}]")
    qq_events = types.ModuleType("nonebot.adapters.onebot.v11.event")
    qq_events.GroupMessageEvent = FakeEvent
    qq_events.PrivateMessageEvent = type("QQPrivateMessageEvent", (), {})
    telegram = types.ModuleType("nonebot.adapters.telegram")
    telegram.Bot = type("TelegramBot", (), {})
    telegram_events = types.ModuleType("nonebot.adapters.telegram.event")
    telegram_events.PrivateMessageEvent = type("TelegramPrivateMessageEvent", (), {})
    telegram_events.GroupMessageEvent = type("TelegramGroupMessageEvent", (), {})
    modules = {module.__name__: module for module in (qq, qq_events, telegram, telegram_events)}
    with (
        patch.dict("sys.modules", modules),
        patch.object(chat, "to_me", return_value=AsyncMock(side_effect=lambda bot, event, state: event.to_me)),
    ):
        yield qq.Bot(self_id=bot.self_id, send=bot.send)


class DiscoveryStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "discovery.db"
        self.now = [datetime(2026, 10, 15, 15, 59, tzinfo=timezone.utc).timestamp()]
        self.store = discovery.DiscoveryStore(self.path, clock=lambda: self.now[0])
        self.actor = ConversationAddress.group("qq", "group-a", "owner")

    def test_daily_limit_is_global_across_groups_and_resets_at_shanghai_midnight(self):
        for number, word in enumerate(("午时", "阳光", "清风")):
            address = ConversationAddress.group("qq", f"group-{number}", "owner")
            self.assertTrue(self.store.claim(address, word))
        self.assertFalse(self.store.claim(self.actor, "笑声"))
        self.now[0] += 60
        self.assertTrue(self.store.claim(self.actor, "笑声"))

    def test_group_word_dedupe_is_permanent_and_shared_between_actors(self):
        self.assertTrue(self.store.claim(self.actor, "午时"))
        other = ConversationAddress.group("qq", "group-a", "other")
        self.assertFalse(self.store.claim(other, "午时"))
        self.now[0] += 2 * 86400
        restarted = discovery.DiscoveryStore(self.path, clock=lambda: self.now[0])
        self.assertFalse(restarted.claim(other, "午时"))
        self.assertTrue(restarted.claim(ConversationAddress.group("qq", "group-b", "other"), "午时"))

    def test_opt_out_persists_for_actor_across_groups(self):
        self.store.opt_out(self.actor)
        restarted = discovery.DiscoveryStore(self.path, clock=lambda: self.now[0])
        other_group = ConversationAddress.group("qq", "elsewhere", "owner")
        self.assertFalse(restarted.eligible(other_group))
        self.assertFalse(restarted.claim(other_group, "午时"))
        self.assertTrue(restarted.eligible(ConversationAddress.group("qq", "group-a", "other")))

    def test_offer_binding_does_not_grant_other_sender_or_group_opt_out(self):
        self.assertTrue(self.store.claim(self.actor, "午时"))
        self.store.bind_offer(self.actor, "午时", "nonce", "offer-message")
        self.assertTrue(self.store.is_offer(self.actor, nonce="nonce"))
        self.assertTrue(self.store.is_offer(self.actor, message_id="offer-message"))
        self.assertFalse(self.store.is_offer(self.actor, nonce="wrong"))
        self.assertFalse(self.store.is_offer(self.actor))
        for address in (
            ConversationAddress.group("qq", "group-a", "other"),
            ConversationAddress.group("qq", "group-b", "owner"),
        ):
            self.assertFalse(self.store.is_offer(address, nonce="nonce", message_id="offer-message"))

    def test_atomic_claim_cannot_exceed_daily_limit(self):
        self.assertTrue(self.store.claim(self.actor, "午时"))
        self.assertTrue(self.store.claim(self.actor, "阳光"))
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda word: self.store.claim(self.actor, word), ("清风", "笑声")))
        self.assertEqual(sum(results), 1)
        self.assertFalse(self.store.eligible(self.actor))

    def test_successful_delivery_commits_while_another_connection_holds_a_read_snapshot(self):
        with closing(sqlite3.connect(self.path)) as reader:
            reader.execute("BEGIN")
            reader.execute("SELECT COUNT(*) FROM discovery_offers").fetchone()
            with self.store.sending_offer(self.actor, "午时", timeout=0) as conn:
                self.assertIsNotNone(conn)
                self.store.record_offer(conn, self.actor, "午时", "sent-nonce", "sent-message")
            self.assertTrue(self.store.is_offer(self.actor, "sent-nonce", "sent-message"))
            reader.rollback()

    def test_model_reservation_is_atomic_and_failed_words_retry_only_next_day(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: self.store.begin_verification("燃面"), range(2)))
        self.assertEqual(sorted(status for _, status in results), ["call", "failed"])
        self.assertIsNone(self.store.get_verdict("燃面"))
        self.now[0] += 60
        self.assertEqual(self.store.begin_verification("燃面"), (None, "call"))
        self.store.cache_verdict("燃面", False)
        self.now[0] += 30 * 86400
        restarted = discovery.DiscoveryStore(self.path, clock=lambda: self.now[0])
        self.assertEqual(restarted.begin_verification("燃面"), (False, "cached"))

    def test_learning_requires_distinct_senders_and_ignores_repetition_and_known_words(self):
        for text, _ in learning_messages():
            self.store.observe(text, self.actor, set())
        self.assertNotIn("晕碳", self.store.learned_words({"晕碳"}))
        self.store.observe("哈哈哈哈", self.actor, set())
        self.store.observe("午时燃面", self.actor, {"午时", "燃面"})
        with closing(sqlite3.connect(self.path)) as conn:
            words = {row[0] for row in conn.execute("SELECT word FROM discovery_ngrams")}
        self.assertTrue({"哈哈", "哈哈哈", "哈哈哈哈", "午时", "燃面"}.isdisjoint(words))

    def test_learning_promotes_once_with_varied_contexts_and_expires(self):
        with patch.object(discovery.logger, "info") as info:
            for text, actor in learning_messages():
                self.store.observe(text, ConversationAddress.group("qq", "g", actor), set())
            self.assertIn("晕碳", self.store.learned_words({"晕碳"}))
            self.store.observe("又晕碳啦", self.actor, set())
        promotions = [call for call in info.call_args_list if "learned=" in str(call)]
        self.assertEqual(sum("晕碳" in str(call) for call in promotions), 1)
        self.now[0] += 14 * 86400 + 1
        self.assertNotIn("晕碳", self.store.learned_words({"晕碳"}))

    def test_expired_observations_cannot_be_refreshed_into_promotion(self):
        samples = list(learning_messages())
        for text, actor in samples[:4]:
            self.store.observe(text, ConversationAddress.group("qq", "g", actor), set())
        self.now[0] += 14 * 86400 + 1
        text, actor = samples[-1]
        self.store.observe(text, ConversationAddress.group("qq", "g", actor), set())
        self.assertNotIn("晕碳", self.store.learned_words({"晕碳"}))

    def test_learning_rejects_fixed_neighbours_despite_count_and_sender_thresholds(self):
        for text, actor in learning_messages():
            text = text.replace("晕碳", "甲晕碳乙")
            self.store.observe(text, ConversationAddress.group("qq", "g", actor), set())
        self.assertNotIn("晕碳", self.store.learned_words({"晕碳"}))

    def test_learning_rejects_low_cohesion_despite_varied_neighbours(self):
        for text, actor in learning_messages():
            self.store.observe(text + "，晕，碳" * 20,
                               ConversationAddress.group("qq", "g", actor), set())
        self.assertNotIn("晕碳", self.store.learned_words({"晕碳"}))

    def test_learning_missing_split_statistics_fail_closed(self):
        for text, actor in learning_messages():
            self.store.observe(text, ConversationAddress.group("qq", "g", actor), set())
        self.assertIn("晕碳", self.store.learned_words({"晕碳"}))
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("DELETE FROM discovery_counts WHERE token='晕'")
        self.assertNotIn("晕碳", self.store.learned_words({"晕碳"}))

    def test_learning_sqlite_write_failure_is_logged_and_ignored(self):
        with (
            patch.object(self.store, "_connect", side_effect=sqlite3.OperationalError("Offline lock")),
            patch.object(discovery.logger, "warning") as warning,
        ):
            self.store.observe("甲晕碳乙", self.actor, set())
        warning.assert_called_once()
        self.assertIn("learning_failed=OperationalError", str(warning.call_args))
        with closing(sqlite3.connect(self.path)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM discovery_ngrams").fetchone()[0], 0)

    def test_learning_prunes_stale_rows_and_hard_caps_substrings(self):
        self.store.observe("甲乙丙丁", self.actor, set())
        self.now[0] += 14 * 86400 + 1
        with patch.object(discovery, "LEARNED_MAX_ROWS", 8):
            self.store.observe("戊己庚辛壬癸子丑寅卯辰巳", self.actor, set())
        with closing(sqlite3.connect(self.path)) as conn:
            words = {row[0] for row in conn.execute("SELECT word FROM discovery_ngrams")}
        self.assertLessEqual(len(words), 8)
        self.assertNotIn("甲乙", words)

    def test_learning_buckets_follow_shanghai_days_instead_of_hours(self):
        self.now[0] -= 2 * 3600
        self.store.observe("甲晕碳乙", self.actor, set())
        self.now[0] += 3600
        self.store.observe("丙晕碳丁", self.actor, set())
        self.now[0] += 3660
        self.store.observe("戊晕碳己", self.actor, set())
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute("SELECT first_seen,total_count FROM discovery_ngrams "
                                "WHERE word='晕碳' ORDER BY first_seen").fetchall()
        self.assertEqual([count for _, count in rows], [2, 1])
        self.assertEqual(rows[1][0] - rows[0][0], 86400)
        self.assertTrue(all(datetime.fromtimestamp(bucket, discovery._DAY_ZONE).hour == 0
                            for bucket, _ in rows))

    def test_steady_load_preserves_multi_sender_evidence_across_ten_days(self):
        start = self.now[0]
        samples = list(learning_messages())
        cap = 4000
        observed = 0
        evicted = False
        with patch.object(discovery, "LEARNED_MAX_ROWS", cap):
            for day in range(10):
                self.now[0] = start + day * 86400
                if day in (0, 1, 2, 4, 9):
                    index = (0, 1, 2, 4, 9).index(day)
                    text, actor = samples[index]
                    self.store.observe(text, ConversationAddress.group("qq", "g", actor), set())
                for hour in range(12):
                    self.now[0] = start + day * 86400 + hour * 3600
                    offset = (day * 12 + hour) * 40
                    text = "".join(chr(0x6000 + offset + index) for index in range(36))
                    self.store.observe(text, self.actor, set())
                    observed += 1
                with closing(sqlite3.connect(self.path)) as conn:
                    rows = conn.execute("SELECT COUNT(*) FROM discovery_ngrams").fetchone()[0]
                    evicted |= rows < observed * 102
                    total = rows + conn.execute("SELECT COUNT(*) FROM discovery_counts").fetchone()[0]
                    self.assertLessEqual(total, cap)
                if day < 9:
                    self.assertNotIn("晕碳", self.store.learned_words({"晕碳"}))
        self.assertTrue(evicted)
        self.assertIn("晕碳", self.store.learned_words({"晕碳"}))
        with closing(sqlite3.connect(self.path)) as conn:
            count, first, last = conn.execute("SELECT SUM(total_count),MIN(first_seen),MAX(first_seen) "
                                              "FROM discovery_ngrams WHERE word='晕碳'").fetchone()
        self.assertEqual(count, 5)
        self.assertEqual(last - first, 9 * 86400)

    def test_counts_overflow_preserves_strong_words_and_whole_token_statistics(self):
        for text, actor in learning_messages():
            self.store.observe(text, ConversationAddress.group("qq", "g", actor), set())
        self.assertIn("晕碳", self.store.learned_words({"晕碳"}))
        with closing(sqlite3.connect(self.path)) as conn:
            old_counts = [conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                          for table in ("discovery_ngrams", "discovery_counts")]
        cap = 2 * (max(old_counts) + 10)
        self.now[0] += 86400
        text = "".join(chr(0x7200 + index) for index in range(100))
        known_words = {text[index:index + size] for index in range(len(text))
                       for size in range(2, min(4, len(text) - index) + 1)}
        with patch.object(discovery, "LEARNED_MAX_ROWS", cap):
            self.store.observe(text, self.actor, known_words)
        with closing(sqlite3.connect(self.path)) as conn:
            totals = []
            for table in ("discovery_ngrams", "discovery_counts"):
                totals.append(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            self.assertEqual(conn.execute("SELECT SUM(total_count) FROM discovery_ngrams WHERE word='晕碳'")
                             .fetchone()[0], 5)
            for token in ("晕", "碳"):
                self.assertEqual(conn.execute("SELECT SUM(total_count) FROM discovery_counts WHERE token=?",
                                              (token,)).fetchone()[0], 5)
        self.assertGreater(totals[1], 0)
        self.assertLessEqual(sum(totals), cap)
        self.assertIn("晕碳", self.store.learned_words({"晕碳"}))

    def test_learning_stores_only_bounded_statistics_and_salted_sender_hashes(self):
        actor_text = "private-sender-identity"
        for index in range(30):
            actor = ConversationAddress.group("qq", "g", actor_text + str(index))
            self.store.observe(chr(0x6200 + index) + "晕碳" + chr(0x6300 + index), actor, set())
        with closing(sqlite3.connect(self.path)) as conn:
            rows = conn.execute("SELECT senders,left_neighbours,right_neighbours "
                                "FROM discovery_ngrams WHERE word='晕碳'").fetchall()
            columns = {row[1] for row in conn.execute("PRAGMA table_info(discovery_ngrams)")}
        self.assertTrue(rows)
        self.assertNotIn("message", columns)
        self.assertNotIn("text", columns)
        for senders, left, right in rows:
            self.assertNotIn(actor_text, senders)
            self.assertLessEqual(len(json.loads(senders)), 10)
            self.assertLessEqual(len(json.loads(left)), 20)
            self.assertLessEqual(len(json.loads(right)), 20)


class CandidateExtractionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "reference.db"
        make_reference(self.path)

    def test_only_common_two_to_four_character_words_with_suitable_pos(self):
        candidates = discovery.extract_candidates("午时，阳光！冷词 张三 台北 怪词 好 AI 123 春夏秋冬天", self.path)
        self.assertEqual(candidates, [discovery.DiscoveryCandidate(word, "commonness")
                                      for word in ("午时", "阳光")])

    def test_repeated_candidates_and_long_messages_are_bounded(self):
        self.assertEqual(discovery.extract_candidates("午时，午时", self.path),
                         [discovery.DiscoveryCandidate("午时", "commonness")])
        self.assertEqual(discovery.extract_candidates("午时" * 10000, self.path), [])

    def test_bcc_sums_only_modern_word_datasets_and_prefers_longest_match(self):
        add_bcc(self.path)
        add_bcc(self.path, "燃面馆", (100,))
        add_bcc(self.path, "冷门", (99,))
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.execute("INSERT INTO bcc_frequency VALUES (?, ?, ?)",
                         ("ancient_word_freq.txt", "冷门", 10000))
        self.assertEqual(discovery.extract_candidates("燃面，冷门，燃面馆", self.path), [
            discovery.DiscoveryCandidate("燃面", "bcc"),
            discovery.DiscoveryCandidate("燃面馆", "bcc"),
        ])

    def test_commonness_wins_source_tie_and_bcc_wins_longer_match(self):
        add_bcc(self.path, "午时", (100,))
        add_bcc(self.path, "午时光", (100,))
        self.assertEqual(discovery.extract_candidates("午时，午时光", self.path), [
            discovery.DiscoveryCandidate("午时", "commonness"),
            discovery.DiscoveryCandidate("午时光", "bcc"),
        ])

    def test_bcc_cannot_override_excluded_commonness_pos(self):
        for word in ("张三", "台北", "冷词", "怪词"):
            add_bcc(self.path, word, (100,))
        self.assertEqual(discovery.extract_candidates("张三，台北，冷词，怪词", self.path), [
            discovery.DiscoveryCandidate("冷词", "bcc"),
            discovery.DiscoveryCandidate("怪词", "bcc"),
        ])


class VerificationTests(unittest.IsolatedAsyncioTestCase):
    def make_chat(self, content):
        response = types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content),
        )])
        create = AsyncMock(return_value=response)
        client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create),
        ))
        context = AsyncMock()
        context.__aenter__.return_value = client
        factory = Mock(return_value=context)
        chat = types.SimpleNamespace(AsyncOpenAI=factory, OPENAI_API_KEY="offline-key",
                                     OPENAI_BASE_URL="https://offline.invalid/v1",
                                     WORD_QUERY_INTENT_MODEL="offline-model")
        return chat, factory, create

    async def test_client_policy_json_context_and_usage_contract(self):
        chat, factory, create = self.make_chat('{"word_ok":true}')
        with (
            patch.object(discovery, "with_deepseek_chat_policy", side_effect=lambda body, **_: body) as policy,
            patch.object(discovery, "log_chat_usage") as usage,
        ):
            self.assertIs(await discovery._verify_word(chat, "燃面", "甲" * 256 + "私密尾部"), True)
        factory.assert_called_once_with(api_key="offline-key", base_url="https://offline.invalid/v1",
                                        timeout=8.0, max_retries=1)
        body = create.await_args.kwargs
        self.assertEqual(body["model"], "offline-model")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertLessEqual(body["max_tokens"], 128)
        self.assertGreater(body["max_tokens"], 0)
        prompt = " ".join(message["content"] for message in body["messages"])
        self.assertIn("燃面", prompt)
        self.assertNotIn("私密尾部", prompt)
        policy.assert_called_once()
        self.assertEqual(policy.call_args.kwargs, {"thinking": False, "json_output": True})
        usage.assert_called_once()
        self.assertEqual(usage.call_args.kwargs["operation"], "word_discovery_verify")
        self.assertEqual(usage.call_args.kwargs["model"], "offline-model")

    async def test_malformed_verdicts_and_transport_failures_close_without_accepting(self):
        for content in ('{"word_ok":"true"}', '{"word_ok":1}', '{}', '[]', 'broken'):
            with self.subTest(content=content):
                chat, _, _ = self.make_chat(content)
                self.assertIsNone(await discovery._verify_word(chat, "燃面", "燃面"))
        for error in (asyncio.TimeoutError(), RuntimeError("Offline model failure")):
            with self.subTest(error=type(error).__name__):
                chat, _, create = self.make_chat('{"word_ok":true}')
                create.side_effect = error
                self.assertIsNone(await discovery._verify_word(chat, "燃面", "燃面"))
        chat, _, _ = self.make_chat('{"word_ok":false}')
        self.assertIs(await discovery._verify_word(chat, "周佛海", "周佛海"), False)


class WordDiscoveryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.reference = self.root / "reference.db"
        make_reference(self.reference)
        self.store = discovery.DiscoveryStore(self.root / "discovery.db")
        self.discovery = discovery.WordDiscovery(self.store, reference_path=self.reference)
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.pending = MemoryConversationStateStore()
        self.event = FakeEvent()
        self.actor = ConversationAddress.group("qq", "200", "100")
        self.bot = types.SimpleNamespace(self_id="999", send=AsyncMock(return_value={"message_id": "400"}))
        self.tools = AsyncMock(side_effect=self.tool)
        self.history = []
        onebot = types.ModuleType("nonebot.adapters.onebot.v11")
        onebot.MessageSegment = types.SimpleNamespace(at=lambda actor: f"[CQ:at,qq={actor}]")
        contexts = (
            patch.dict(os.environ, {"KEYTAO_WORD_DISCOVERY": "1"}),
            patch.dict("sys.modules", {"nonebot.adapters.onebot.v11": onebot}),
            patch.object(self.chat, "conversation_state_store", self.pending),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "call_tool_function", self.tools),
            patch.object(self.chat, "add_to_history", side_effect=lambda key, message, response: self.history.extend([
                {"role": "user", "content": message}, {"role": "assistant", "content": response},
            ])),
            patch.object(self.chat, "get_history", side_effect=lambda *args: self.history),
            patch.object(self.chat, "OPENAI_API_KEY", ""),
            patch.object(self.chat, "AsyncOpenAI", None),
            patch.object(self.commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)),
        )
        for context in contexts:
            context.start()
            self.addCleanup(context.stop)

    async def asyncTearDown(self):
        task = self.discovery.task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def tool(self, name, arguments, platform=None, user_id=None, **kwargs):
        self.assertEqual((platform, str(user_id)), ("qq", "100"))
        if name == "keytao_lookup_by_words_batch":
            return json.dumps({"success": True, "results": [
                {"success": True, "word": word, "phrases": []} for word in arguments["words"]
            ]})
        if name == "keytao_lookup_by_word":
            return json.dumps({"success": True, "word": arguments["word"], "phrases": []})
        if name == "keytao_pending_items_by_words":
            return json.dumps({"success": True, "complete": True, "items": []})
        if name == "keytao_prepare_reviewed_add":
            return json.dumps(reviewed(arguments["word"]), ensure_ascii=False)
        raise AssertionError(f"Unexpected offline tool: {name}")

    async def offer(self, event=None):
        await self.discovery.discover(self.bot, event or self.event, self.chat)
        return self.pending.get_record(self.actor)

    async def test_bcc_missing_word_is_verified_before_offer_and_acceptance_is_cached(self):
        add_bcc(self.reference)
        with patch.object(discovery, "_verify_word", AsyncMock(return_value=True)) as verify:
            record = await self.offer(FakeEvent("今天吃燃面"))
            self.assertIsNotNone(record)
            self.assertEqual(record.state.word, "燃面")
            verify.assert_awaited_once_with(self.chat, "燃面", "今天吃燃面")
            self.assertIs(self.store.get_verdict("燃面"), True)
            self.pending.delete(self.actor)
            self.discovery = discovery.WordDiscovery(discovery.DiscoveryStore(self.store.path),
                                                     reference_path=self.reference)
            await self.discovery.discover(self.bot, FakeEvent("燃面", group="other-group"), self.chat)
            self.assertEqual(verify.await_count, 1)
            other = ConversationAddress.group("qq", "other-group", "100")
            self.assertEqual(self.pending.get_record(other).state.word, "燃面")

    async def test_bcc_model_rejects_are_permanent_and_person_names_are_not_offered(self):
        for word in ("燃面", "周佛海"):
            add_bcc(self.reference, word)
            with self.subTest(word=word), patch.object(
                discovery, "_verify_word", AsyncMock(return_value=False),
            ) as verify:
                for _ in range(2):
                    self.assertIsNone(await self.offer(FakeEvent(word)))
                verify.assert_awaited_once()
                self.assertIs(self.store.get_verdict(word), False)
        self.bot.send.assert_not_awaited()
        self.assertTrue(all(call.args[0] == "keytao_lookup_by_words_batch"
                            for call in self.tools.await_args_list))

    async def test_commonness_and_existing_bcc_words_do_not_call_verifier(self):
        with patch.object(discovery, "_verify_word", AsyncMock()) as verify:
            self.assertIsNotNone(await self.offer())
            self.pending.delete(self.actor)
            add_bcc(self.reference)
            self.tools.side_effect = None
            self.tools.return_value = json.dumps({"success": True, "results": [{
                "success": True, "word": "燃面", "phrases": [{"word": "燃面", "code": "rm"}],
            }]})
            self.assertIsNone(await self.offer(FakeEvent("燃面")))
        verify.assert_not_awaited()

    async def test_unavailable_bcc_candidates_do_not_hide_later_commonness_word(self):
        for word in ("发截图", "燃面"):
            add_bcc(self.reference, word)
        with closing(sqlite3.connect(self.reference)) as conn, conn:
            conn.execute("INSERT INTO word_commonness VALUES ('内卷',3,'n',2)")
        for status in ("reject", "cached-reject", "capped", "failed", "cached-failed"):
            with self.subTest(status=status):
                self.store = discovery.DiscoveryStore(self.root / f"fallback-{status}.db")
                self.discovery = discovery.WordDiscovery(self.store, reference_path=self.reference)
                if status.startswith("cached"):
                    self.store.begin_verification("发截图")
                    if status == "cached-reject":
                        self.store.cache_verdict("发截图", False)
                elif status == "capped":
                    for index in range(discovery.VERIFY_DAILY_CAP):
                        self.store.begin_verification(f"offline-{index}")
                self.bot.send.reset_mock()
                verdict = None if status == "failed" else False
                with patch.object(discovery, "_verify_word", AsyncMock(return_value=verdict)) as verify:
                    record = await self.offer(FakeEvent("我发截图给燃面内卷的人"))
                self.assertIsNotNone(record)
                self.assertEqual(record.state.word, "内卷")
                self.assertEqual(verify.await_count, 1 if status in {"reject", "cached-reject", "failed",
                                                                      "cached-failed"} else 0)
                self.bot.send.assert_awaited_once()
                with closing(sqlite3.connect(self.store.path)) as conn:
                    calls = conn.execute("SELECT SUM(calls) FROM discovery_model_days").fetchone()[0]
                self.assertEqual(calls, discovery.VERIFY_DAILY_CAP if status == "capped"
                                 else 2 if status.startswith("cached") else 1)
                self.pending.delete(self.actor)

    async def test_cached_acceptance_is_available_after_an_earlier_model_reject(self):
        for word in ("发截图", "燃面"):
            add_bcc(self.reference, word)
        self.store.begin_verification("燃面")
        self.store.cache_verdict("燃面", True)
        with patch.object(discovery, "_verify_word", AsyncMock(return_value=False)) as verify:
            record = await self.offer(FakeEvent("发截图，燃面"))
        self.assertIsNotNone(record)
        self.assertEqual(record.state.word, "燃面")
        verify.assert_awaited_once_with(self.chat, "发截图", "发截图，燃面")
        self.bot.send.assert_awaited_once()

    async def test_unsuccessful_bcc_lookup_rows_never_verify_or_reserve_budget(self):
        add_bcc(self.reference)
        self.tools.side_effect = None
        with patch.object(discovery, "_verify_word", AsyncMock(return_value=True)) as verify:
            for row in ({"word": "燃面", "success": False, "phrases": []},
                        {"word": "燃面", "phrases": []}):
                with self.subTest(row=row):
                    self.tools.return_value = json.dumps({"success": True, "results": [row]})
                    self.assertIsNone(await self.offer(FakeEvent("燃面")))
            verify.assert_not_awaited()
        with closing(sqlite3.connect(self.store.path)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM discovery_model_days").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM discovery_verdicts").fetchone()[0], 0)
        self.bot.send.assert_not_awaited()
        self.assertEqual(self.tools.await_count, 2)

    async def test_global_model_daily_cap_prevents_the_thirty_first_call(self):
        words = [chr(0x6800 + number) + "面" for number in range(31)]
        for word in words:
            add_bcc(self.reference, word)
        with patch.object(discovery, "_verify_word", AsyncMock(return_value=False)) as verify:
            for index, word in enumerate(words):
                await self.offer(FakeEvent(word, group=f"group-{index}"))
        self.assertEqual(verify.await_count, 30)
        self.assertIsNone(self.store.get_verdict(words[-1]))
        self.bot.send.assert_not_awaited()

    async def test_failed_verification_is_not_cached_or_retried_until_next_day(self):
        add_bcc(self.reference)
        now = [datetime(2026, 10, 15, 15, 59, tzinfo=timezone.utc).timestamp()]
        self.store = discovery.DiscoveryStore(self.root / "failure.db", clock=lambda: now[0])
        self.discovery = discovery.WordDiscovery(self.store, reference_path=self.reference)
        with patch.object(discovery, "_verify_word", AsyncMock(return_value=None)) as verify:
            self.assertIsNone(await self.offer(FakeEvent("燃面")))
            self.assertIsNone(await self.offer(FakeEvent("燃面")))
            self.assertEqual(verify.await_count, 1)
            self.assertIsNone(self.store.get_verdict("燃面"))
            now[0] += 60
            self.assertIsNone(await self.offer(FakeEvent("燃面")))
            self.assertEqual(verify.await_count, 2)
        self.bot.send.assert_not_awaited()

    async def test_learned_word_is_available_next_message_and_requires_verification(self):
        with patch.object(discovery, "_verify_word", AsyncMock(return_value=True)) as verify:
            for text, actor in learning_messages():
                self.assertIsNone(await self.offer(FakeEvent(text, actor=actor)))
            verify.assert_not_awaited()
            self.tools.assert_not_awaited()
            self.assertEqual(discovery.extract_candidates("晕碳", self.reference, self.store),
                             [discovery.DiscoveryCandidate("晕碳", "learned")])
            record = await self.offer(FakeEvent("晕碳"))
            self.assertIsNotNone(record)
            self.assertEqual(record.state.word, "晕碳")
            verify.assert_awaited_once_with(self.chat, "晕碳", "晕碳")
        self.bot.send.assert_awaited_once()

    async def test_learning_uses_existing_message_filters_and_survives_offer_ineligibility(self):
        observe = Mock(wraps=self.store.observe)
        with patch.object(self.store, "observe", observe):
            addressed = FakeEvent("甲晕碳乙")
            addressed.to_me = True
            for event in (addressed, FakeEvent("甲晕碳乙", kind="forward"), FakeEvent("加词 晕碳")):
                await self.offer(event)
            observe.assert_not_called()
            self.store.opt_out(self.actor)
            await self.offer(FakeEvent("甲晕碳乙"))
        observe.assert_called_once()
        self.tools.assert_not_awaited()

    async def test_missing_common_word_creates_normal_sender_owned_ticket(self):
        record = await self.offer()
        self.assertIsNotNone(record)
        self.assertIsInstance(record.state, PendingAddWord)
        self.assertEqual(record.owner_key, self.actor)
        self.assertEqual(record.state.server_candidates, [("wj", False), ("wja", False)])
        self.assertTrue(record.origin_prompt_digest)
        self.assertEqual([call.args[0] for call in self.tools.await_args_list], [
            "keytao_lookup_by_words_batch", "keytao_prepare_reviewed_add",
        ])
        self.bot.send.assert_awaited_once()
        message = self.bot.send.await_args.kwargs["message"]
        for text in ("[CQ:at,qq=100]", "「午时」词库里还没有，要加入吗？", "1", "wj", "确认", "取消"):
            self.assertIn(text, str(message))
        offered = copy.deepcopy(record.state)
        self.pending.delete(self.actor)
        await self.chat._try_handle_simple_single_word_query(
            "加词 午时", "qq", "100", self.actor, self.actor.space_key, "Offline sender",
        )
        self.assertEqual(offered, self.pending.get(self.actor))

    async def test_confirmation_and_number_selection_use_normal_resolver(self):
        for message, expected_code in (("确认", "wj"), ("2", "wja")):
            with self.subTest(message=message):
                self.store = discovery.DiscoveryStore(self.root / f"{expected_code}.db")
                self.discovery = discovery.WordDiscovery(self.store, reference_path=self.reference)
                self.pending.delete(self.actor)
                await self.offer()
                execute = AsyncMock(return_value="offline addition")
                with patch.object(self.chat, "_execute_add_to_draft", execute):
                    result = await self.chat.handle_pending_message_core(
                        message, "qq", "100", self.actor, history=[],
                        space_key=self.actor.space_key, owner_label="Offline sender",
                        allow_intent_model=False,
                    )
                self.assertTrue(result.startswith("offline addition"), result)
                execute.assert_awaited_once()
                self.assertEqual(execute.await_args.args[0:2], ("午时", expected_code))
                self.assertIsNone(self.pending.get_record(self.actor))

    async def test_present_word_never_calls_review_or_creates_ticket(self):
        self.tools.side_effect = None
        self.tools.return_value = json.dumps({"success": True, "results": [{
            "success": True, "word": "午时", "phrases": [{"word": "午时", "code": "wj"}],
        }]})
        self.assertIsNone(await self.offer())
        self.assertEqual(self.tools.await_count, 1)
        self.bot.send.assert_not_awaited()

    async def test_ineligible_text_does_not_call_dictionary(self):
        for text in ("冷词", "张三", "台北", "怪词", "好", "hello 123"):
            with self.subTest(text=text):
                self.assertIsNone(await self.offer(FakeEvent(text)))
        self.tools.assert_not_awaited()
        self.bot.send.assert_not_awaited()

    async def test_only_one_offer_and_one_batch_lookup_per_message(self):
        await self.offer(FakeEvent("午时，阳光，清风"))
        self.assertEqual(self.tools.await_count, 2)
        self.assertEqual(self.tools.await_args_list[0].args[1], {"words": ["午时", "阳光", "清风"]})
        self.bot.send.assert_awaited_once()
        self.assertEqual(self.pending.live_entry_count(), 1)

    async def test_missing_or_malformed_batch_rows_fail_closed(self):
        payloads = (
            {"success": False, "results": []},
            {"success": True},
            {"success": True, "results": []},
            {"success": True, "results": [{"word": "午时"}]},
            {"success": True, "results": [{"word": "午时", "phrases": None}]},
            {"success": True, "results": [{"word": "午时", "phrases": []}] * 2},
        )
        self.tools.side_effect = None
        for payload in payloads:
            with self.subTest(payload=payload):
                self.tools.return_value = json.dumps(payload)
                self.assertIsNone(await self.offer())
        self.assertEqual(self.tools.await_count, len(payloads))
        self.bot.send.assert_not_awaited()

    async def test_opted_out_user_and_live_ticket_in_another_group_are_skipped(self):
        other_group = ConversationAddress.group("qq", "elsewhere", "100")
        self.pending.set(other_group, PendingAddWord("阳光", "yg", [("yg", False)]))
        self.assertIsNone(await self.offer())
        self.pending.delete(other_group)
        self.store.opt_out(self.actor)
        self.assertIsNone(await self.offer())
        self.tools.assert_not_awaited()

    async def test_offer_reply_opt_out_is_permanent_and_releases_only_own_ticket(self):
        await self.offer()
        reply = FakeEvent("别问我")
        reply.reply = types.SimpleNamespace(message_id="400")
        await self.offer(reply)
        self.assertFalse(self.store.eligible(ConversationAddress.group("qq", "elsewhere", "100")))
        self.assertIsNone(self.pending.get_record(self.actor))
        self.assertEqual(self.tools.await_count, 2)

    async def test_ordinary_opt_out_phrase_without_offer_does_not_persist_preference(self):
        await self.offer(FakeEvent("别问我"))
        self.assertTrue(self.store.eligible(self.actor))
        self.tools.assert_not_awaited()

    async def test_commands_quotes_forwards_bot_and_addressed_messages_are_skipped(self):
        cases = [FakeEvent(text) for text in ("/午时", "加词 午时", "请帮我加词 午时", "“午时”", "'午时'", "> 午时", "转发 午时")]
        cases.extend([FakeEvent("午时", actor="999"), FakeEvent("午时", kind="forward")])
        addressed = FakeEvent()
        addressed.to_me = True
        cases.append(addressed)
        quoted = FakeEvent()
        quoted.reply = types.SimpleNamespace(message_id="unrelated")
        cases.append(quoted)
        bot_event = FakeEvent()
        bot_event.sender.is_bot = True
        cases.append(bot_event)
        for event in cases:
            with self.subTest(text=event.text, kind=event.message[0].type, actor=event.user_id):
                self.assertIsNone(await self.offer(event))
        self.tools.assert_not_awaited()

    async def test_flag_is_exactly_one_and_off_never_schedules(self):
        for value in ("", "0", "true", "TRUE", "yes", " 1"):
            with self.subTest(value=value), patch.dict(os.environ, {"KEYTAO_WORD_DISCOVERY": value}):
                self.assertFalse(self.discovery.schedule(self.bot, self.event, self.chat))
                self.assertFalse(discovery.schedule_discovery(self.bot, self.event, self.chat))
                await self.offer()
        self.assertIsNone(self.discovery.task)
        self.tools.assert_not_awaited()

    async def test_schedule_runs_in_background_and_skips_while_busy(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked(*args, **kwargs):
            entered.set()
            await release.wait()

        with patch.object(self.discovery, "discover", AsyncMock(side_effect=blocked)) as run:
            self.assertTrue(self.discovery.schedule(self.bot, self.event, self.chat))
            self.assertFalse(entered.is_set())
            await asyncio.wait_for(entered.wait(), timeout=1)
            first = self.discovery.task
            self.assertFalse(self.discovery.schedule(self.bot, self.event, self.chat))
            self.assertIs(self.discovery.task, first)
            run.assert_awaited_once()
            release.set()
            await asyncio.wait_for(first, timeout=1)
            self.assertTrue(self.discovery.schedule(self.bot, self.event, self.chat))
            await asyncio.wait_for(self.discovery.task, timeout=1)
            self.assertEqual(run.await_count, 2)

    async def test_directed_closed_answers_enter_normal_qq_route_and_resolver(self):
        with qq_runtime(self.chat, self.bot) as bot:
            for message, expected_code, directed in (
                ("确认", "wj", "quote"), ("2", "wja", "quote"),
                ("1", "wj", "mention"), ("喵喵 1", "wj", "name"),
                ("@喵喵 1", "wj", "name"),
                ("wj", "wj", "quote"),
            ):
                with self.subTest(message=message):
                    self.pending.delete(self.actor)
                    self.store = discovery.DiscoveryStore(self.root / f"route-{directed}-{message}.db")
                    self.discovery = discovery.WordDiscovery(self.store, reference_path=self.reference)
                    await self.offer()
                    record = self.pending.get_record(self.actor)
                    self.assertIsNotNone(record)
                    self.assertTrue(self.store.is_offer(self.actor, record.nonce))
                    # Reopen the store to prove routing uses its durable offer marker.
                    self.discovery = discovery.WordDiscovery(discovery.DiscoveryStore(self.store.path))
                    other = ConversationAddress.group("qq", "200", "101")
                    normal = PendingAddWord("阳光", "yg", [("yg", False)], server_candidates=[("yg", False)])
                    self.pending.set(other, normal)
                    execute = AsyncMock(return_value="offline routed addition")
                    with (
                        patch.object(discovery, "_discovery", self.discovery),
                        patch.object(self.chat, "_execute_add_to_draft", execute),
                    ):
                        plain = self.chat._strip_command_message_prefixes(message)
                        self.assertFalse(await self.chat.should_handle(bot, FakeEvent(plain, actor="102")))
                        self.assertFalse(await self.chat.should_handle(bot, FakeEvent(plain, actor="101")))
                        reply = FakeEvent(message)
                        if directed == "quote":
                            reply.reply = types.SimpleNamespace(message_id="400")
                        elif directed == "mention":
                            reply.to_me = True
                        self.assertTrue(await self.chat.should_handle(bot, reply))
                        execute.assert_not_awaited()
                        result = await self.chat.handle_pending_message_core(
                            message, "qq", "100", self.actor, history=[],
                            space_key=self.actor.space_key, owner_label="Offline sender",
                            allow_intent_model=False,
                        )
                    self.assertTrue(result.startswith("offline routed addition"), result)
                    execute.assert_awaited_once()
                    self.assertEqual(execute.await_args.args[:2], ("午时", expected_code))
                    self.assertIsNone(self.pending.get_record(self.actor))
                    with patch.object(self.chat, "_execute_add_to_draft", execute):
                        await self.chat.handle_pending_message_core(
                            message, "qq", "100", self.actor, history=[],
                            space_key=self.actor.space_key, owner_label="Offline sender",
                            allow_intent_model=False,
                        )
                    execute.assert_awaited_once()
                    self.assertIs(self.pending.get(other), normal)
                    self.pending.delete(other)

    async def test_addressed_non_closed_answers_release_offer_before_normal_handling(self):
        cases = (
            ("@喵喵 查一下 你好", None),
            ("@喵喵 查一下 你好", "different-message"),
            ("查一下 你好", None),
            ("查一下 你好", "different-message"),
            *((text, None) for text in ("好的呀哈哈", "加吧", "3", "unknown", "1并提交")),
        )
        for index, (text, reply_id) in enumerate(cases):
            with self.subTest(text=text, reply_id=reply_id):
                self.store = discovery.DiscoveryStore(self.root / f"addressed-{index}.db")
                self.discovery = discovery.WordDiscovery(self.store, reference_path=self.reference)
                await self.offer()
                event = FakeEvent(text)
                event.to_me = True
                if reply_id:
                    event.reply = types.SimpleNamespace(message_id=reply_id)
                with (
                    qq_runtime(self.chat, self.bot) as bot,
                    patch.object(discovery, "_discovery", self.discovery),
                ):
                    self.assertTrue(await self.chat.should_handle(bot, event))
                    self.assertIsNone(self.pending.get_record(self.actor))
                self.assertEqual(event.text, text)
                self.assertEqual(event.reply.message_id if event.reply else None, reply_id)

    async def test_addressed_non_closed_route_is_neutral_only_after_offer_removal(self):
        record = await self.offer()
        event = FakeEvent("@喵喵 查一下 你好")
        event.to_me = True
        with (
            qq_runtime(self.chat, self.bot) as bot,
            patch.object(discovery, "_discovery", self.discovery),
        ):
            with patch.object(self.pending, "_delete_persisted", return_value=False):
                self.assertFalse(await discovery.discovery_reply_route(bot, event, self.chat))
                self.assertIs(self.pending.get_record(self.actor), record)
            self.assertIsNone(await discovery.discovery_reply_route(bot, event, self.chat))
            self.assertIsNone(self.pending.get_record(self.actor))

    async def test_addressed_non_closed_reply_never_deletes_a_different_ticket(self):
        for timing in ("before-routing", "during-address-check"):
            with self.subTest(timing=timing):
                self.store = discovery.DiscoveryStore(self.root / f"replacement-{timing}.db")
                self.discovery = discovery.WordDiscovery(self.store, reference_path=self.reference)
                offer = await self.offer()
                replacement = None

                def replace_ticket():
                    nonlocal replacement
                    if replacement is None:
                        self.pending.set(self.actor, PendingAddWord(
                            "阳光", "yg", [("yg", False)], server_candidates=[("yg", False)],
                        ))
                        replacement = self.pending.get_record(self.actor)

                async def addressed(bot, event, state):
                    replace_ticket()
                    return True

                if timing == "before-routing":
                    replace_ticket()
                event = FakeEvent("@喵喵 查一下 你好")
                event.to_me = True
                event.reply = types.SimpleNamespace(message_id="different-message")
                with (
                    qq_runtime(self.chat, self.bot) as bot,
                    patch.object(self.chat, "to_me", return_value=addressed),
                    patch.object(discovery, "_discovery", self.discovery),
                ):
                    self.assertIsNone(await discovery.discovery_reply_route(bot, event, self.chat))
                    self.assertTrue(await self.chat.should_handle(bot, event))
                self.assertIsNotNone(replacement)
                self.assertNotEqual(replacement.nonce, offer.nonce)
                self.assertIs(self.pending.get_record(self.actor), replacement)
                self.pending.delete(self.actor)

    async def test_unaddressed_messages_and_wrong_quote_closed_answers_never_touch_offer(self):
        await self.offer()
        record = self.pending.get_record(self.actor)
        original = copy.deepcopy(record)
        messages = [FakeEvent(text) for text in ("好", "可以", "行", "对", "加吧", "1", "2", "确认")]
        unaddressed = FakeEvent("好")
        unaddressed.reply = types.SimpleNamespace(message_id="different-message")
        messages.append(unaddressed)
        for text in ("确认", "1", "取消"):
            wrong_quote = FakeEvent(text)
            wrong_quote.reply = types.SimpleNamespace(message_id="different-message")
            wrong_quote.to_me = True
            messages.append(wrong_quote)
        unrelated_mention = FakeEvent("@other 确认")
        unrelated_mention.reply = types.SimpleNamespace(message_id="400")
        messages.append(unrelated_mention)
        with (
            qq_runtime(self.chat, self.bot) as bot,
            patch.object(discovery, "_discovery", self.discovery),
            patch.object(self.chat, "handle_pending_message_core", AsyncMock()) as resolver,
        ):
            for event in messages:
                with self.subTest(text=event.text, reply=event.reply, to_me=event.to_me):
                    resolver.reset_mock()
                    if await self.chat.should_handle(bot, event):
                        await self.chat.handle_pending_message_core(event.text, "qq", "100", self.actor)
                    resolver.assert_not_awaited()
                    self.assertIs(self.pending.get_record(self.actor), record)
                    self.assertEqual(record, original)

    async def test_retired_offer_quotes_and_store_errors_fail_closed(self):
        await self.offer()
        reply = FakeEvent("好")
        reply.to_me = True
        reply.reply = types.SimpleNamespace(message_id="400")
        with qq_runtime(self.chat, self.bot) as bot, patch.object(discovery, "_discovery", self.discovery):
            with patch.object(self.store, "is_offer", side_effect=sqlite3.OperationalError("Offline read failure")):
                self.assertFalse(await self.chat.should_handle(bot, reply))
            self.pending.delete(self.actor)
            self.assertFalse(await self.chat.should_handle(bot, reply))
            reply.user_id = "101"
            self.assertFalse(await self.chat.should_handle(bot, reply))

    async def test_failed_preparation_render_and_send_leave_quota_and_word_available(self):
        for failure in ("prepare", "render", "send"):
            with self.subTest(failure=failure):
                self.store = discovery.DiscoveryStore(self.root / f"failed-{failure}.db")
                self.discovery = discovery.WordDiscovery(self.store, reference_path=self.reference)
                if failure == "prepare":
                    with patch.object(discovery, "_prepare_offer", AsyncMock(return_value=None)):
                        await self.offer()
                elif failure == "render":
                    with patch.object(self.chat, "_render_live_single_candidate_record", return_value=""):
                        await self.offer()
                else:
                    self.bot.send.side_effect = RuntimeError("Offline send failure")
                    with self.assertRaises(RuntimeError):
                        await self.offer()
                    self.bot.send.side_effect = None
                self.assertIsNone(self.pending.get_record(self.actor))
                self.assertTrue(self.store.eligible(self.actor, "午时"))
                self.assertIsNotNone(await self.offer())
                self.assertFalse(self.store.eligible(self.actor, "午时"))
                self.pending.delete(self.actor)

    async def test_concurrent_delivery_cannot_exceed_remaining_daily_slot(self):
        self.assertTrue(self.store.claim(self.actor, "清风"))
        self.assertTrue(self.store.claim(self.actor, "笑声"))
        entered = asyncio.Event()
        resume = asyncio.Event()

        async def paused_send(*args, **kwargs):
            entered.set()
            await resume.wait()
            return {"message_id": "400"}

        self.bot.send.side_effect = paused_send
        other_store = discovery.DiscoveryStore(self.store.path)
        other_discovery = discovery.WordDiscovery(other_store, reference_path=self.reference)
        other_address = ConversationAddress.group("qq", "other-group", "100")
        other_pending = MemoryConversationStateStore()
        other_chat = types.SimpleNamespace(**vars(self.chat))
        other_chat.conversation_state_store = other_pending
        other_chat.conversation_space_message_locks = type(self.chat.conversation_space_message_locks)()
        other_chat.conversation_message_locks = type(self.chat.conversation_message_locks)()
        other_chat.draft_actor_message_locks = type(self.chat.draft_actor_message_locks)()
        task = asyncio.create_task(self.discovery.discover(self.bot, self.event, self.chat))
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            await other_discovery.discover(self.bot, FakeEvent("阳光", group="other-group"), other_chat)
            self.bot.send.assert_awaited_once()
            self.assertIsNone(other_pending.get_record(other_address))
        finally:
            resume.set()
            await asyncio.wait_for(task, timeout=1)
        self.assertFalse(self.store.eligible(self.actor))
        self.assertFalse(other_store.claim(other_address, "阳光"))

    async def test_opt_out_during_delivery_waits_and_preserves_preference(self):
        old_address = ConversationAddress.group("qq", "old-group", "100")
        self.assertTrue(self.store.claim(old_address, "阳光"))
        self.store.bind_offer(old_address, "阳光", "old-nonce", "old-message")
        entered = asyncio.Event()
        resume = asyncio.Event()

        async def paused_send(*args, **kwargs):
            entered.set()
            await resume.wait()
            return {"message_id": "400"}

        self.bot.send.side_effect = paused_send
        task = asyncio.create_task(self.discovery.discover(self.bot, self.event, self.chat))
        opt_out = FakeEvent("别问我", group="old-group")
        opt_out.reply = types.SimpleNamespace(message_id="old-message")
        preference_task = None
        try:
            await asyncio.wait_for(entered.wait(), timeout=1)
            with patch.object(self.store, "opt_out", wraps=self.store.opt_out) as persist:
                preference_task = asyncio.create_task(self.discovery._handle_opt_out(self.bot, opt_out, self.chat))
                for _ in range(100):
                    if persist.called:
                        break
                    await asyncio.sleep(0.001)
                self.assertTrue(persist.called)
                self.assertFalse(preference_task.done())
        finally:
            resume.set()
            await asyncio.wait_for(task, timeout=1)
            if preference_task is not None:
                self.assertTrue(await asyncio.wait_for(preference_task, timeout=1))
        self.assertTrue(self.store.opted_out(self.actor))

    async def test_dictionary_presence_cache_expires_and_is_bounded_lru(self):
        self.tools.side_effect = None
        now = [0.0]

        async def present(name, arguments, *args):
            self.assertEqual(name, "keytao_lookup_by_words_batch")
            return json.dumps({"success": True, "results": [
                {"success": True, "word": word, "phrases": [{"word": word, "code": "wj"}]}
                for word in arguments["words"]
            ]})

        self.tools.side_effect = present
        with (
            patch.object(discovery, "time", types.SimpleNamespace(monotonic=lambda: now[0])),
            patch.object(discovery, "PRESENT_CACHE_MAX_WORDS", 2, create=True),
        ):
            for text in ("午时", "阳光", "午时", "清风", "午时"):
                await self.offer(FakeEvent(text))
            self.assertEqual(self.tools.await_count, 3)
            await self.offer(FakeEvent("阳光"))
            self.assertEqual(self.tools.await_count, 4)
            now[0] += 86400
            await self.offer(FakeEvent("阳光"))
            self.assertEqual(self.tools.await_count, 5)
        self.bot.send.assert_not_awaited()

    async def test_opt_out_from_old_offer_in_another_group_cancels_suspended_preparation(self):
        old_address = ConversationAddress.group("qq", "old-group", "100")
        self.assertTrue(self.store.claim(old_address, "阳光"))
        self.store.bind_offer(old_address, "阳光", "old-nonce", "old-offer-message")
        entered = asyncio.Event()
        resume = asyncio.Event()
        prepare = discovery._prepare_offer

        async def paused(*args):
            entered.set()
            await resume.wait()
            return await prepare(*args)

        with (
            patch.object(discovery, "_prepare_offer", paused),
            patch.object(discovery, "_discovery", self.discovery),
        ):
            task = asyncio.create_task(self.discovery.discover(self.bot, self.event, self.chat))
            try:
                await asyncio.wait_for(entered.wait(), timeout=1)
                opt_out = FakeEvent("别问我", group="old-group")
                opt_out.reply = types.SimpleNamespace(message_id="old-offer-message")
                self.assertFalse(await discovery.discovery_reply_route(self.bot, opt_out, self.chat))
                self.assertTrue(discovery.DiscoveryStore(self.store.path).opted_out(self.actor))
            finally:
                resume.set()
                await asyncio.wait_for(task, timeout=1)
        self.assertIsNone(self.pending.get_record(self.actor))
        self.assertEqual(self.pending.live_entry_count(), 0)
        self.bot.send.assert_not_awaited()

    async def test_foreground_ticket_installed_during_preparation_remains_unchanged(self):
        for group in ("200", "other-group"):
            with self.subTest(group=group):
                self.store = discovery.DiscoveryStore(self.root / f"foreground-{group}.db")
                self.discovery = discovery.WordDiscovery(self.store, reference_path=self.reference)
                address = ConversationAddress.group("qq", group, "100")
                entered = asyncio.Event()
                resume = asyncio.Event()
                prepare = discovery._prepare_offer

                async def paused(*args):
                    entered.set()
                    await resume.wait()
                    return await prepare(*args)

                with patch.object(discovery, "_prepare_offer", paused):
                    task = asyncio.create_task(self.discovery.discover(self.bot, self.event, self.chat))
                    try:
                        await asyncio.wait_for(entered.wait(), timeout=1)
                        state = PendingAddWord("阳光", "yg", [("yg", False)], server_candidates=[("yg", False)])
                        self.assertTrue(self.pending.set(address, state))
                        record = self.pending.get_record(address)
                        original = copy.deepcopy(record)
                    finally:
                        resume.set()
                        await asyncio.wait_for(task, timeout=1)
                self.assertIs(self.pending.get_record(address), record)
                self.assertEqual(record, original)
                self.assertEqual(self.pending.live_entry_count(), 1)
                self.bot.send.assert_not_awaited()
                self.pending.delete(address)

    async def test_flag_off_qq_rule_and_trace_do_not_call_discovery_hooks(self):
        reply_hook = AsyncMock(side_effect=AssertionError("Disabled discovery reply hook"))
        with (
            qq_runtime(self.chat, self.bot) as bot,
            patch.dict(os.environ, {"KEYTAO_WORD_DISCOVERY": "0"}),
            patch.object(discovery, "discovery_reply_route", reply_hook),
            patch.object(discovery, "schedule_discovery") as schedule,
        ):
            self.assertFalse(await self.chat.should_handle(bot, self.event))
            await self.chat.trace_sensitive_message(bot, self.event)
        reply_hook.assert_not_awaited()
        schedule.assert_not_called()
        self.tools.assert_not_awaited()

    async def test_quote_confirmation_is_admitted_and_counted_only_after_offer_delivery(self):
        entered = asyncio.Event()
        resume = asyncio.Event()

        async def paused_send(*args, **kwargs):
            entered.set()
            await resume.wait()
            return {"message_id": "delivered-offer"}

        self.bot.send.side_effect = paused_send
        with patch.object(discovery, "_discovery", self.discovery):
            task = asyncio.create_task(self.discovery.discover(self.bot, self.event, self.chat))
            try:
                await asyncio.wait_for(entered.wait(), timeout=1)
                record = self.pending.get_record(self.actor)
                self.assertIsNotNone(record)
                self.assertFalse(self.store.is_offer(self.actor, record.nonce))
                self.assertTrue(self.store.eligible(self.actor, "午时"))
                self.assertFalse(await discovery.discovery_reply_route(self.bot, FakeEvent("确认"), self.chat))
                with qq_runtime(self.chat, self.bot) as bot:
                    addressed = FakeEvent("好")
                    addressed.to_me = True
                    self.assertFalse(await self.chat.should_handle(bot, addressed))
                self.assertIs(self.pending.get_record(self.actor), record)
            finally:
                resume.set()
                await asyncio.wait_for(task, timeout=1)
            self.assertTrue(self.store.is_offer(self.actor, record.nonce, "delivered-offer"))
            reply = FakeEvent("确认")
            reply.reply = types.SimpleNamespace(message_id="delivered-offer")
            self.assertTrue(await discovery.discovery_reply_route(self.bot, reply, self.chat))
        self.bot.send.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
