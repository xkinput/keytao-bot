"""Offline B9 regressions for lookup and model-call budgets."""

import json
import socket
import sys
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness.state import MemoryConversationStateStore, PendingAddWord
from keytao_bot.plugins import chat_routing as routing
from keytao_bot.utils.observability import begin_turn_metrics, end_turn_metrics
from test_semantic_parser import FakeClient, raw_operation
from keytao_bot.utils import keytao_review as review


class B9RoutingSpeedTests(unittest.IsolatedAsyncioTestCase):
    async def test_unbound_affirmative_controls_keep_the_ticket_without_models(self):
        chat = harness.openai_chat_module
        for message, digest in ((message, digest) for message in ("嗯嗯", "好滴", "是的", "没问题") for digest in ("", "a" * 64)):
            with self.subTest(message=message, digest=digest):
                store = MemoryConversationStateStore()
                author = harness.ConversationAddress.group("qq", "b9-speed", "owner")
                self.assertTrue(store.set(author, PendingAddWord(
                    "云翻", "yjfj", [("yjfj", False)], server_candidates=[("yjfj", False)],
                )))
                store.bind_origin_prompt_digest(author, digest)
                record = store.get_record(author)
                client = FakeClient()
                ctx = chat.TurnContext(
                    bot=None, event=None, platform="qq", user_id=author.actor_id,
                    normalized_message_text=message, conv_key=author, space_key=author.space_key,
                )
                with (
                    patch.object(chat, "conversation_state_store", store),
                    patch.object(chat, "get_history", return_value=[]),
                    patch.object(chat, "AsyncOpenAI", return_value=client),
                ):
                    self.assertFalse(await chat._stage_understand_operation_early(ctx))
                self.assertEqual(ctx.scoped_pending_response, "这条回复未对应当前待确认操作；请引用当前候选消息后再回复。")
                self.assertIs(store.get_record(author), record)
                client.create.assert_not_awaited()

    async def test_resolved_pending_control_uses_at_most_one_model_call(self):
        chat = harness.openai_chat_module
        store = MemoryConversationStateStore()
        author = harness.ConversationAddress.group("qq", "b9-speed", "owner")
        state = PendingAddWord("云翻", "yjfj", [("yjfj", False)])
        self.assertTrue(store.set(author, state))
        client = FakeClient(json.dumps(raw_operation(
            op="confirm", operands=["云翻"], codes=[], refers_to_pending=True,
        )))
        ctx = chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=author.actor_id,
            message_text="加入", normalized_message_text="加入", conv_key=author,
            space_key=author.space_key, current_pending_record=store.get_record(author),
        )
        ctx.command_intent_cache, ctx.command_intent_for = routing.command_intent_memoizer("加入", "加入")
        with (
            patch.dict("os.environ", {"KEYTAO_SEMANTIC_ROUTE_OPS": "pending_answer,pending_release"}),
            patch.object(chat, "AsyncOpenAI", return_value=client),
            patch.object(routing, "AsyncOpenAI", return_value=client),
            patch.object(routing, "OPENAI_API_KEY", "offline"),
            patch.object(chat, "conversation_state_store", store),
            patch.object(chat, "remember_conversation"),
            patch.object(chat, "_finish_ai_chat_matcher", AsyncMock()) as finish,
            patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")),
        ):
            self.assertFalse(await chat._stage_understand_operation(ctx))
            self.assertTrue(ctx.operation_request.refers_to_pending)
            self.assertTrue(await chat._stage_apply_scoped_pending_intent(ctx))
            await ctx.command_intent_for(state)
        self.assertLessEqual(client.create.await_count, 1)
        self.assertEqual(ctx.response, "当前没有可验证的可执行操作，本次未写入。")
        finish.assert_awaited_once_with(ctx.response)
        self.assertIs(store.get_record(author), ctx.current_pending_record)

    async def test_word_query_classification_is_reused_only_within_the_turn(self):
        classifier = AsyncMock(return_value=routing.SimpleWordQueryIntent(False))
        with patch.object(routing, "_classify_simple_word_query_intent", classifier):
            for _turn in range(2):
                token = begin_turn_metrics("qq", "group")
                try:
                    for _stage in range(2):
                        self.assertEqual(await routing._get_simple_word_query_words("洛阳纸贵 喜上眉梢"), ())
                finally:
                    end_turn_metrics(token)
            self.assertEqual(classifier.await_count, 2)


class B9LookupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.commands = harness.chat_commands_module
        self.store = MemoryConversationStateStore()
        self.calls = []
        self.entries = {"喜上眉梢": "xemev"}
        self.pending = [{
            "source": source, "batchId": batch, "batchUrl": f"https://offline.invalid/batch/{batch}",
            "batchStatus": status, "itemStatus": "Pending", "action": "Create",
            "word": word, "code": code, "type": "Phrase",
        } for source, batch, status, word, code in (
            ("draft", "draft", "Draft", "发薪", "fsxb"),
            ("submitted", "submitted", "Submitted", "云翻", "yjfj"),
        )]
        self.lookup_failure = False
        self.pending_failure = False
        self.patches = (
            patch.object(self.commands, "conversation_state_store", self.store),
            patch.object(self.commands, "call_tool_function", AsyncMock(side_effect=self.tool)),
            patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")),
        )
        for context in self.patches:
            context.start()
            self.addCleanup(context.stop)

    def row(self, word):
        code = self.entries.get(word)
        return {"success": not self.lookup_failure, "word": word,
                "phrases": [{"word": word, "code": code, "type": "Phrase"}] if code else []}

    async def tool(self, name, args, *actor, **kwargs):
        self.calls.append((name, args))
        if name == "keytao_lookup_by_word":
            result = self.row(args["word"])
        elif name == "keytao_lookup_by_words_batch":
            result = {"success": not self.lookup_failure,
                      "results": [self.row(word) for word in args["words"]]}
        elif name == "keytao_pending_items_by_words":
            result = {"success": not self.pending_failure, "complete": not self.pending_failure,
                      "message": "Offline pending unavailable",
                      "items": [row for row in self.pending if row["word"] in args["words"]]}
        elif name == "keytao_prepare_reviewed_add":
            result = {"success": False, "message": f"Offline review unavailable: {args['word']}"}
        else:
            raise AssertionError(name)
        return json.dumps(result, ensure_ascii=False)

    async def prepare(self, words):
        return await self.commands._prepare_multi_word_query(
            words, "qq", "b9-owner", None, None, "Offline owner",
        )

    async def test_multi_word_prefetch_preserves_existing_draft_and_submitted_output(self):
        words = ("喜上眉梢", "发薪", "云翻", "未收录")
        response = await self.prepare(words)
        self.assertEqual(response, "\n\n".join((
            "「发薪」已在当前草稿中（→ fsxb）：https://offline.invalid/batch/draft",
            "「云翻」已在待审核批次中（→ yjfj，审核中）：https://offline.invalid/batch/submitted",
            "Offline review unavailable: 未收录",
            "已在词库：喜上眉梢 xemev",
        )))
        self.assertEqual(self.calls, [
            ("keytao_lookup_by_words_batch", {"words": list(words)}),
            ("keytao_pending_items_by_words", {"words": list(words)}),
            ("keytao_prepare_reviewed_add", {"word": "未收录"}),
        ])

    async def test_pending_failure_keeps_each_word_fail_closed(self):
        self.pending_failure = True
        response = await self.prepare(("喜上眉梢", "未收录"))
        self.assertEqual(response, "Offline pending unavailable\n\nOffline pending unavailable")
        self.assertEqual([name for name, _args in self.calls], [
            "keytao_lookup_by_words_batch", "keytao_pending_items_by_words",
        ])
        self.assertIsNone(self.store.get_record(("qq", "b9-owner")))

    async def test_lookup_failure_keeps_the_existing_review_fallback(self):
        self.lookup_failure = True
        words = ("喜上眉梢", "未收录")
        response = await self.prepare(words)
        self.assertEqual(response, "\n\n".join(f"Offline review unavailable: {word}" for word in words))
        self.assertEqual([args["word"] for name, args in self.calls
                          if name == "keytao_prepare_reviewed_add"], list(words))
        self.assertEqual(sum(name == "keytao_lookup_by_words_batch" for name, _args in self.calls), 1)

    async def test_prefetch_chunks_at_pending_lookup_limit(self):
        words = tuple(f"离线词{index}" for index in range(51))
        self.entries.update({word: "abcd" for word in words})
        with patch.object(self.commands, "_MAX_BARE_MULTI_WORD_QUERY_ITEMS", 100):
            response = await self.prepare(words)
        self.assertEqual(response, "已在词库：" + "、".join(f"{word} abcd" for word in words))
        self.assertEqual(self.calls, [
            ("keytao_lookup_by_words_batch", {"words": list(words[:50])}),
            ("keytao_pending_items_by_words", {"words": list(words[:50])}),
            ("keytao_lookup_by_words_batch", {"words": list(words[50:])}),
            ("keytao_pending_items_by_words", {"words": list(words[50:])}),
        ])

class B9PronunciationEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        review._clear_review_caches()
        if hasattr(review, "_evidence_host_failures"):
            review._evidence_host_failures.clear()

    async def test_agreeing_local_datasets_skip_web_for_nonpolyphonic_word(self):
        local = {
            "entries": [{
                "sourceId": dataset, "dataset": dataset, "source": dataset,
                "url": "", "normalized": ["ce", "shi"],
                "pinyin": "ce shi", "category": "dictionary", "trust": 4,
            } for dataset in ("cedict", "large_pinyin")],
            "outcomes": [{"sourceId": dataset, "status": "completed", "lookupResult": "found"}
                         for dataset in ("cedict", "large_pinyin")],
        }
        with (
            patch.object(review, "_collect_local_pronunciation_evidence", return_value=local),
            patch.object(review, "detect_polyphonic_characters", return_value={}),
            patch.object(review, "query_reference_readings", return_value=[object()]),
            patch.object(review, "_fetch_text", AsyncMock(return_value="")) as fetch,
            patch.object(review, "_search_web", AsyncMock(return_value=[])) as search,
            patch.object(review, "_request_bot_evidence_proxy", AsyncMock(
                return_value=review._BotEvidenceProxyResult("unavailable"))) as proxy,
        ):
            result = await review.collect_pronunciation_evidence("测试")
        self.assertEqual(fetch.await_count + search.await_count + proxy.await_count, 0)
        self.assertTrue(result["lookupComplete"])
        self.assertEqual(result["groups"][0]["normalized"], ["ce", "shi"])

    async def test_local_disagreement_or_unknown_polyphones_still_consult_web(self):
        entry = {"sourceId": "cedict", "dataset": "cedict", "source": "cedict",
                 "url": "", "normalized": ["ce", "shi"], "pinyin": "ce shi",
                 "category": "dictionary", "trust": 4}
        cases = [
            ([entry], {}, None, [object()]),
            ([entry, {**entry, "sourceId": "large_pinyin", "dataset": "large_pinyin",
                      "normalized": ["ce", "si"]}], {}, None, [object()]),
            ([entry, {**entry, "sourceId": "large_pinyin", "dataset": "large_pinyin"}], {0: ("ce", "ze")}, None, [object()]),
            ([entry, {**entry, "sourceId": "large_pinyin", "dataset": "large_pinyin"}], {},
             review.PinyinReferenceUnavailable("missing data"), [object()]),
            ([entry, {**entry, "sourceId": "large_pinyin", "dataset": "large_pinyin"}], {}, None, []),
        ]
        for entries, polyphones, error, character_rows in cases:
            with self.subTest(entries=len(entries), polyphones=polyphones, error=error):
                review._clear_review_caches()
                with (
                    patch.object(review, "AUTHORITATIVE_SOURCES", [{
                        "id": "fixture", "label": "fixture", "domain": "fixture.invalid",
                        "category": "dictionary", "trust": 4, "query": "{word}",
                        "direct_urls": ["https://fixture.invalid/{word}"],
                    }]),
                    patch.object(review, "_collect_local_pronunciation_evidence", return_value={
                        "entries": entries, "outcomes": []}),
                    patch.object(review, "detect_polyphonic_characters", return_value=polyphones,
                                 side_effect=error),
                    patch.object(review, "query_reference_readings", return_value=character_rows),
                    patch.object(review, "_fetch_text", AsyncMock(return_value="")) as fetch,
                    patch.object(review, "_search_web", AsyncMock(return_value=[])),
                ):
                    await review.collect_pronunciation_evidence("测试")
                self.assertGreater(fetch.await_count, 0)

    def timeout_types(self):
        class TransportError(Exception):
            pass

        class TimeoutException(TransportError):
            pass

        class ConnectTimeout(TimeoutException):
            pass

        class ConnectError(TransportError):
            pass

        return TransportError, TimeoutException, ConnectTimeout, ConnectError

    async def test_timeouts_are_not_retried_including_httpx_connect_timeout(self):
        classes = self.timeout_types()
        with ExitStack() as stack:
            for name, cls in zip(("TransportError", "TimeoutException", "ConnectTimeout", "ConnectError"), classes):
                stack.enter_context(patch.object(sys.modules["httpx"], name, cls, create=True))
            for error in (TimeoutError("timed out"), classes[1]("read timeout"), classes[2]("connect timeout")):
                with self.subTest(error=type(error).__name__):
                    with patch.object(review.http_client, "guarded_fetch", AsyncMock(side_effect=error)) as fetch:
                        with self.assertRaises(type(error)):
                            await review._fetch_review_url("https://timeout.invalid/page", max_attempts=3)
                    self.assertEqual(fetch.await_count, 1)

    async def test_pronunciation_search_registry_uses_the_same_timeout_policy(self):
        classes = self.timeout_types()
        with ExitStack() as stack:
            for name, cls in zip(("TransportError", "TimeoutException", "ConnectTimeout", "ConnectError"), classes):
                stack.enter_context(patch.object(sys.modules["httpx"], name, cls, create=True))
            stack.enter_context(patch.object(review, "_registered_web_search_function", None))
            stack.enter_context(patch.object(review.http_client.asyncio, "sleep", AsyncMock()))
            fetch = stack.enter_context(patch.object(review.http_client, "guarded_fetch", AsyncMock(
                side_effect=classes[1]("read timeout"))))
            await review._registered_web_search("测试 拼音", fetch_top_n=0)
        urls = [call.args[0] for call in fetch.await_args_list]
        self.assertTrue(urls)
        self.assertEqual(len(urls), len(set(urls)))

    async def test_breaker_is_host_scoped_expires_and_success_resets_streak(self):
        classes = self.timeout_types()
        response = SimpleNamespace(status_code=200, is_success=True, text="")
        now = [1000.0]
        with ExitStack() as stack:
            for name, cls in zip(("TransportError", "TimeoutException", "ConnectTimeout", "ConnectError"), classes):
                stack.enter_context(patch.object(sys.modules["httpx"], name, cls, create=True))
            stack.enter_context(patch.object(review.time, "monotonic", side_effect=lambda: now[0]))
            fetch = stack.enter_context(patch.object(review.http_client, "guarded_fetch", AsyncMock(
                side_effect=[TimeoutError(), TimeoutError(), response, TimeoutError(), TimeoutError(),
                             TimeoutError(), response, response])))
            for _ in range(2):
                with self.assertRaises(TimeoutError):
                    await review._fetch_review_url("https://one.invalid/a", max_attempts=1)
            await review._fetch_review_url("https://one.invalid/b", max_attempts=1)
            for _ in range(3):
                with self.assertRaises(TimeoutError):
                    await review._fetch_review_url("https://one.invalid/c", max_attempts=1)
            self.assertEqual(fetch.await_count, 6)
            with self.assertRaises(TimeoutError):
                await review._fetch_review_url("https://one.invalid/d", max_attempts=1)
            unavailable = await review._fetch_text("https://one.invalid/d")
            self.assertEqual(unavailable.lookup_status, "timed_out")
            self.assertFalse(unavailable)
            self.assertEqual(fetch.await_count, 6)
            await review._fetch_review_url("https://two.invalid/a", max_attempts=1)
            now[0] += 600
            await review._fetch_review_url("https://one.invalid/e", max_attempts=1)
            self.assertEqual(fetch.await_count, 8)


if __name__ == "__main__":
    unittest.main()
