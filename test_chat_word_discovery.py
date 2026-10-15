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
from unittest.mock import AsyncMock, patch

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


class CandidateExtractionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "reference.db"
        make_reference(self.path)

    def test_only_common_two_to_four_character_words_with_suitable_pos(self):
        candidates = discovery.extract_candidates("午时，阳光！冷词 张三 台北 怪词 好 AI 123 春夏秋冬天", self.path)
        self.assertEqual(candidates, ["午时", "阳光"])

    def test_repeated_candidates_and_long_messages_are_bounded(self):
        self.assertEqual(discovery.extract_candidates("午时，午时", self.path), ["午时"])
        self.assertEqual(discovery.extract_candidates("午时" * 10000, self.path), [])


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
