"""Offline B9 regressions for control routing and compact candidate replies."""

import socket
import unittest
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
import test_b8_cancel as prior
import test_semantic_query_route as query
from eval.routing.replay_prod_cases import Replay
from keytao_bot.harness.state import (
    PendingAddWord, PendingToolConfirm, create_pending_trusted_word_record,
    server_warning_pending_state,
)
from keytao_bot.plugins import chat_routing as routing
from test_semantic_parser import FakeClient, raw_operation


class B9WordQueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_control_text_never_calls_word_classifier(self):
        classifier = AsyncMock(return_value=routing.SimpleWordQueryIntent(True, ("取消",)))
        with patch.object(routing, "_classify_simple_word_query_intent", classifier):
            for message in ("取消", "取消。", "确认", "提交", "加入", "加", "删", "查", "算了"):
                with self.subTest(message=message):
                    self.assertEqual(await routing._get_simple_word_query_words(message), ())
            classifier.assert_not_awaited()

    async def test_classifier_words_pass_the_same_structural_exclusions(self):
        classifier = AsyncMock(return_value=routing.SimpleWordQueryIntent(
            True, ("洛阳纸贵", "取消", "确认", "提交", "加入", "加", "删", "查"),
        ))
        with patch.object(routing, "_classify_simple_word_query_intent", classifier):
            self.assertEqual(
                await routing._get_simple_word_query_words("洛阳纸贵 喜上眉梢"),
                ("洛阳纸贵",),
            )
            classifier.assert_awaited_once()


class B9CancelTests(unittest.IsolatedAsyncioTestCase):
    async def test_change_number_cannot_borrow_named_selection_binding(self):
        fixture = prior.B8CancelTests()
        replay = fixture.replay()
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            record = await fixture.offer(replay)
            replay.history[:] = [{"role": "assistant", "content": "Offline unrelated reply."}]
            replay.calls.clear()
            execute = stack.enter_context(patch.object(replay.commands, "_execute_confirmed_tool", AsyncMock(return_value="Unexpected write")))
            await replay.turn({"message": "换 2", "to_me": True}, raw_operation(
                op="select_candidate", operands=[], codes=[], frame="direct", refers_to_pending=True,
            ))
            self.assertEqual(replay.last_response, "这条回复未对应当前待确认操作；请引用当前候选消息后再回复。")
            self.assertIs(replay.store.get_record(replay.author), record)
            self.assertEqual(replay.calls, [])
            self.assertEqual(replay.writes, [])
            execute.assert_not_awaited()
            replay.client.create.assert_not_awaited()

    async def test_semantic_query_keeps_compact_present_facts_through_delivery(self):
        replay = Replay({
            "sender": "b9-owner",
            "dictionary": [
                {"word": "喜上眉梢", "code": "xemev", "type": "Phrase"},
                {"word": "欢天喜地", "code": "htxda", "type": "Phrase"},
            ],
            "candidate_facts": {"发薪": {"recommended": "fsxb", "codes": ["fsxb", "fsxba"], "pinyin": "fa xin"}},
        }, harness, FakeClient)
        words = ("喜上眉梢", "欢天喜地", "发薪")
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
            await replay.turn({"message": " ".join(words), "to_me": True}, raw_operation(
                op="query", operands=list(words), codes=[], frame="none",
            ))
            self.assertTrue(replay.last_response.startswith("已在词库：喜上眉梢 xemev、欢天喜地 htxda\n"), replay.last_response)
            self.assertIn("「换 2」", replay.last_response)
            self.assertTrue(replay.chat._advertised_reply_matches_live_record(replay.last_response, replay.store.get_record(replay.author)))
            self.assertEqual(replay.writes, [])

    async def test_cancel_without_actor_ticket_finishes_before_models_or_lookup(self):
        for message in ("取消", "取消。", "算了", "不加了", "取消本次添加"):
            with self.subTest(message=message):
                replay = Replay({"sender": "b9-owner"}, harness, FakeClient)
                other = harness.ConversationAddress.group("qq", "prod-replay", "other-owner")
                self.assertTrue(replay.store.set(other, PendingAddWord("云翻", "yjfj", [("yjfj", False)])))
                original = replay.store.get_record(other)
                with ExitStack() as stack:
                    for context in replay.patches():
                        stack.enter_context(context)
                    stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
                    main = stack.enter_context(patch.object(replay.chat, "get_ai_response_core", AsyncMock()))
                    await replay.turn({"message": message, "to_me": True}, raw_operation(
                        op="query", operands=["取消"], codes=[], frame="none",
                    ))
                    self.assertEqual(replay.last_response, "当前没有待取消的操作。")
                    self.assertEqual(replay.trace, ["_stage_understand_operation_early"])
                    self.assertEqual(replay.calls, [])
                    self.assertEqual(replay.writes, [])
                    replay.client.create.assert_not_awaited()
                    main.assert_not_awaited()
                    self.assertIs(replay.store.get_record(other), original)


class B9CandidateTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = query.SemanticQueryRouteTests.asyncSetUp
    asyncTearDown = query.SemanticQueryRouteTests.asyncTearDown
    lookup = query.SemanticQueryRouteTests.lookup

    async def test_lookup_context_is_not_an_operation_to_cancel(self):
        self.assertTrue(self.store.set(self.author, create_pending_trusted_word_record("积极", "jjuu", "Phrase")))
        ctx = query.SemanticQueryRouteTests.context(self, "取消")
        ctx.event = object()
        with patch.object(self.chat, "to_me", return_value=AsyncMock(return_value=True)):
            self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
        self.assertEqual(ctx.response, "当前没有待取消的操作。")
        self.tool.assert_not_awaited()
        self.main.assert_not_awaited()
        self.client.create.assert_not_awaited()

    async def offer(self, missing=(), *, explicit=True):
        self.entries.update({"喜上眉梢": "xemev", "欢天喜地": "htxda"})
        words = ("喜上眉梢", "欢天喜地", *missing)
        return await self.commands._try_handle_simple_single_word_query(
            ("加词：" if explicit else "") + "，".join(words),
            "qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner",
        )

    async def test_multi_word_add_and_query_collapse_all_present_words(self):
        for explicit in (True, False):
            with self.subTest(explicit=explicit):
                self.assertEqual(await self.offer(explicit=explicit), "已在词库：喜上眉梢 xemev、欢天喜地 htxda")
                self.assertIsNone(self.store.get_record(self.author))
                self.main.assert_not_awaited()
                self.client.create.assert_not_awaited()

    async def test_one_actionable_word_uses_single_layout_with_original_ticket(self):
        response = await self.offer(("发薪",))
        self.assertTrue(response.startswith("已在词库：喜上眉梢 xemev、欢天喜地 htxda\n"))
        self.assertEqual(response.count("发薪"), 1, response)
        self.assertNotIn("1 个词的候选", response)
        self.assertNotIn("可再追加一个编码", response)
        for control in ("加入", "加入并提交", "换 2", "取消"):
            self.assertIn(f"「{control}」", response)
        record = self.store.get_record(self.author)
        self.assertEqual(record.state.function_name, "keytao_batch_add_to_draft")
        self.assertEqual(record.state.args["_query_words"], ["喜上眉梢", "欢天喜地", "发薪"])
        self.assertTrue(self.chat._advertised_reply_matches_live_record(response, record))
        self.assertEqual(self.chat._render_live_batch_record(record), response)
        self.assertEqual(self.chat._append_pending_ticket_challenge(response, self.author), response)

    async def test_change_number_selects_only_live_word_and_keeps_binding(self):
        await self.offer(("发薪",))
        record = self.store.get_record(self.author)
        for message in ("换 2", "发薪 2"):
            with self.subTest(message=message):
                derived, intent, error = routing._resolve_multi_word_pending_candidate_selection(record.state, message)
                self.assertIsNone(error)
                self.assertIsNotNone(derived)
                self.assertEqual(intent.intent, "pending_confirm")
                self.assertEqual([(item["word"], item["code"]) for item in derived.args["items"]], [("发薪", "fsxba")])
                self.assertEqual(derived.args["_query_words"], record.state.args["_query_words"])
                self.assertTrue(routing.message_authorizes_live_pending_mutation(message, record.state))

    async def test_change_number_is_ambiguous_for_multiple_live_words(self):
        response = await self.offer(("发薪", "云翻"))
        self.assertIn("取消", response)
        self.assertNotIn("换 2", response)
        self.assertTrue(self.chat.advertised_reply_contract(response).candidate_selection)
        record = self.store.get_record(self.author)
        self.assertTrue(self.chat._advertised_reply_matches_live_record(response, record))
        derived, _intent, _error = routing._resolve_multi_word_pending_candidate_selection(record.state, "换 2")
        self.assertIsNone(derived)
        self.assertFalse(routing.message_authorizes_live_pending_mutation("换 2", record.state))

    async def test_change_number_on_single_word_ticket_executes_selected_code(self):
        state = PendingAddWord("午时", "wj", [("wj", False), ("wja", False)],
                               server_candidates=[("wj", False), ("wja", False)])
        self.assertTrue(self.store.set(self.author, state))
        with patch.object(self.commands, "_execute_add_to_draft", AsyncMock(return_value="Offline receipt")) as execute:
            response = await self.commands.handle_pending_message_core(
                "换 2", "qq", self.author.actor_id, self.author, allow_intent_model=False,
            )
        self.assertIn("Offline receipt", response)
        self.assertEqual(execute.await_args.args[:2], ("午时", "wja"))
        self.client.create.assert_not_awaited()

    async def test_single_actionable_risk_ticket_keeps_original_inventory_for_change_number(self):
        await self.offer(("发薪",))
        origin = self.commands._candidate_origin_for_ticket(self.store.get(self.author))
        state = server_warning_pending_state(PendingToolConfirm("keytao_create_phrase", {
            "word": "发薪", "code": "fsxb", "_candidate_origin": origin,
        }), {"batchId": "offline-batch", "contentVersion": 1, "warningDigest": "a" * 64, "warnings": []})
        self.assertTrue(self.store.set(self.author, state))
        with patch.object(self.commands, "_execute_confirmed_tool", AsyncMock(return_value="Offline preview")) as execute:
            await self.commands.try_reselect_live_ticket("换 2", "qq", self.author.actor_id, self.author)
        execute.assert_awaited_once()
        selected = execute.await_args.args[0]
        self.assertEqual((selected.args["word"], selected.args["code"]), ("发薪", "fsxba"))
        self.assertEqual(selected.args["_candidate_origin"]["queryWords"], ["喜上眉梢", "欢天喜地", "发薪"])
        self.assertTrue(execute.await_args.kwargs["require_new_confirmation"])
        self.client.create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
