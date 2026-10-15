"""Offline safety checks for semantic swap routing and the legacy swap gate."""

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import test_state_machine as harness
from test_semantic_parser import FakeClient, raw_operation

from keytao_bot.harness.operation_request import OperationRequest, understand_message
from keytao_bot.harness.state import (
    DraftOperationCoordinator,
    MemoryConversationStateStore,
    PendingAddWord,
    PendingToolConfirm,
)


class SemanticSwapRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.chat._semantic_shadow_tasks.clear()
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id="semantic-swap-owner", space_type="group",
            space_id="semantic-swap-group", speaker_name="Offline owner",
        )
        self.ctx = self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.memory.user_id,
            normalized_message_text="帮我把午时跟务实的码对调",
            conv_key=self.memory.conversation_address,
            space_key=("qq", self.memory.space_scope_id),
            memory_context=self.memory, owner_label="Offline owner",
            response="unchanged response", scoped_pending_response="old ticket response",
        )
        self.client = FakeClient(json.dumps(self.swap_raw(), ensure_ascii=False))
        self.swap = AsyncMock(return_value="sealed swap preview")
        self.finish = AsyncMock()
        self.remember = MagicMock()
        self.tool = AsyncMock(side_effect=AssertionError("Unexpected offline tool call"))
        self.info = MagicMock()
        patches = (
            patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": "swap", "KEYTAO_SEMANTIC_SHADOW": "0"}),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.commands, "try_handle_model_entry_swap", self.swap),
            patch.object(self.chat, "_finish_ai_chat_response", self.finish),
            patch.object(self.chat, "remember_conversation", self.remember),
            patch.object(self.chat, "call_tool_function", self.tool),
            patch.object(self.chat.logger, "info", self.info),
            patch.object(self.chat.logger, "warning"),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    async def asyncTearDown(self):
        tasks = tuple(self.chat._semantic_shadow_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.chat._semantic_shadow_tasks.clear()
        self.tool.assert_not_awaited()

    @staticmethod
    def swap_raw(**changes):
        return raw_operation(op="swap", operands=["午时", "务实"], codes=[], **changes)

    def set_model(self, **changes):
        self.client.create.return_value.choices[0].message.content = json.dumps(
            {**self.swap_raw(), **changes}, ensure_ascii=False,
        )

    def reviewed_batch(self):
        pairs = (("午时", "wj"), ("务实", "wja"))
        scopes = []
        for word, code in pairs:
            reviewed = PendingAddWord(
                word, code, [(code, False)], server_candidates=[(code, False)],
            )
            scopes.append({
                "word": word, "candidates": [(code, False)],
                "occupiedWords": {}, "orderingAssessments": [],
                "reviewedState": self.commands._pending_add_word_payload(reviewed),
            })
        return PendingToolConfirm("keytao_batch_add_to_draft", {
            "items": [{"action": "Create", "word": word, "code": code}
                      for word, code in pairs],
            "_candidate_scopes": scopes, "_reviewed_multi_word": True,
            "_query_words": [word for word, _code in pairs],
        })

    def assert_swap(self):
        self.swap.assert_awaited_once_with(
            "午时", "务实", "qq", self.ctx.user_id,
            self.ctx.space_key, self.ctx.owner_label,
        )
        self.assertEqual(self.ctx.response, "sealed swap preview")
        self.assertIsNone(self.ctx.scoped_pending_response)
        self.finish.assert_awaited_once()
        self.remember.assert_called_once_with(
            self.ctx.conv_key, self.ctx.memory_context,
            self.ctx.normalized_message_text, "sealed swap preview",
        )

    async def test_three_spec_phrasings_reach_shared_helper_once(self):
        for message in (
            "调整调换 午时 和 务实 词条",
            "午时和务实互换一下位置",
            "帮我把午时跟务实的码对调",
        ):
            with self.subTest(message=message):
                self.ctx.normalized_message_text = message
                self.swap.reset_mock()
                self.finish.reset_mock()
                self.remember.reset_mock()
                self.client.create.reset_mock()
                fast = understand_message(message)
                self.assertEqual(fast.op, "swap")
                if message == "帮我把午时跟务实的码对调":
                    self.assertEqual(fast.operands, ("午时", "务实的码"))
                self.assertTrue(await self.chat._stage_understand_operation(self.ctx))
                self.assert_swap()
                self.client.create.assert_awaited_once()
                self.assertEqual(self.ctx.operation_request.source, "model:offline-model")

    async def test_route_flag_requires_exact_comma_separated_swap_token(self):
        for flag in (None, "", "1", "swpa", "swaps", "SWAP", "swap;", "swap，", "swa p", "other"):
            with self.subTest(flag=flag):
                if flag is None:
                    os.environ.pop("KEYTAO_SEMANTIC_ROUTE_OPS", None)
                else:
                    os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = flag
                self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
                self.assertEqual(self.ctx.response, "unchanged response")
                self.assertEqual(self.ctx.scoped_pending_response, "old ticket response")
        self.swap.assert_not_awaited()
        self.client.create.assert_not_awaited()
        self.finish.assert_not_awaited()

    async def test_comma_empty_tokens_and_other_ops_do_not_disable_exact_swap(self):
        # The design defines comma tokens, so a trailing comma still contains swap.
        for flag in ("swap,", "swap,other", "other, swap", ",swap,", " swap "):
            with self.subTest(flag=flag):
                os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = flag
                self.swap.reset_mock()
                self.assertTrue(await self.chat._stage_understand_operation(self.ctx))
                self.swap.assert_awaited_once()

    async def test_direct_question_and_question_backstop_can_preview(self):
        cases = (
            ("帮我把午时跟务实的码对调", "direct", "direct"),
            ("帮我把午时跟务实的码对调", "question", "question"),
            ("帮我把午时跟务实的码对调吗？", "direct", "question"),
        )
        for message, frame, expected_frame in cases:
            with self.subTest(message=message, frame=frame):
                self.ctx.normalized_message_text = message
                self.set_model(frame=frame)
                self.swap.reset_mock()
                self.assertTrue(await self.chat._stage_understand_operation(self.ctx))
                self.swap.assert_awaited_once()
                self.assertEqual(self.ctx.operation_request.frame, expected_frame)

    async def test_unsafe_frames_and_frame_backstops_fall_through(self):
        cases = (
            ("别把午时和务实对调", "direct", "negated"),
            ("他说把午时和务实对调", "direct", "reported"),
            ("「把午时和务实对调」\n这是收到的可疑文本。", "direct", "reported"),
            *(("帮我把午时跟务实的码对调", frame, frame)
              for frame in ("negated", "reported", "none", "unknown")),
        )
        for message, frame, expected_frame in cases:
            with self.subTest(message=message, frame=frame):
                self.ctx.normalized_message_text = message
                self.set_model(frame=frame)
                with patch("keytao_bot.harness.operation_request.understand_message", return_value=OperationRequest()):
                    self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
                self.assertEqual(self.ctx.operation_request.frame, expected_frame)
                self.assertEqual(self.ctx.response, "unchanged response")
                self.assertEqual(self.ctx.scoped_pending_response, "old ticket response")
        self.swap.assert_not_awaited()
        self.finish.assert_not_awaited()

    async def test_invalid_operands_arity_codes_and_other_violations_fall_through(self):
        self.ctx.normalized_message_text = "帮我把午时跟务实对调，另有云帆"
        for changes in (
            {"operands": ["午时", "不存在"]},
            {"operands": ["午时"]},
            {"operands": ["午时", "务实", "云帆"]},
            {"operands": ["午时", "午时"]},
            {"operands": ["午时", "务实", None]},
            {"operands": "午时务实"},
            {"operands": ["午时", "务实", "不存在"]},
            {"codes": ["notmentioned"]},
            {"codes": ["INVALID"]},
            {"codes": None},
            {"op": "move"},
        ):
            with self.subTest(changes=changes):
                self.set_model(**changes)
                with patch("keytao_bot.harness.operation_request.understand_message", return_value=OperationRequest()):
                    self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
                self.assertEqual(self.ctx.response, "unchanged response")
                self.assertEqual(self.ctx.scoped_pending_response, "old ticket response")
        self.swap.assert_not_awaited()

    async def test_refers_to_pending_true_never_routes_even_with_author_ticket(self):
        pending = PendingAddWord("午时", "wj", [("wj", False)])
        self.ctx.current_pending_record = MagicMock(state=pending)
        self.set_model(refers_to_pending=True)
        self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
        self.assertTrue(self.ctx.operation_request.refers_to_pending)
        self.swap.assert_not_awaited()
        self.assertEqual(self.ctx.scoped_pending_response, "old ticket response")

    async def test_author_live_add_tickets_do_not_claim_complete_model_swaps(self):
        states = (
            PendingAddWord("午时", "wj", [("wj", False)], server_candidates=[("wj", False)]),
            self.reviewed_batch(),
        )
        for state in states:
            for message in (
                "调整调换 午时 和 务实 词条",
                "午时和务实互换一下位置",
                "帮我把午时跟务实的码对调",
            ):
                with self.subTest(ticket=type(state).__name__, message=message):
                    store = MemoryConversationStateStore()
                    self.assertTrue(store.set(self.ctx.conv_key, state))
                    self.ctx.normalized_message_text = message
                    self.swap.reset_mock()
                    self.finish.reset_mock()
                    self.remember.reset_mock()
                    self.client.create.reset_mock()
                    with (
                        patch.object(self.chat, "conversation_state_store", store),
                        patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
                        patch.object(self.commands, "try_verify_pending_fly_proposal", AsyncMock(return_value=None)),
                        patch.object(self.commands, "try_reselect_live_ticket", AsyncMock(return_value=None)),
                        patch.object(self.chat, "_try_handle_explicit_pending_replacement", AsyncMock(return_value=None)),
                    ):
                        self.assertFalse(await self.chat._stage_resolve_current_pending_scope(self.ctx))
                        self.assertIs(self.ctx.current_pending_record.state, state)
                        self.assertTrue(await self.chat._stage_understand_operation(self.ctx))
                    self.assert_swap()
                    self.client.create.assert_awaited_once()
                    model_message = self.client.create.await_args.kwargs["messages"][1]["content"]
                    self.assertIn("pending_context: {", model_message)
                    self.assertIn("午时", model_message.split("pending_context:", 1)[1])

    async def test_model_swap_replaces_actual_reviewed_batch_rejection(self):
        state = self.reviewed_batch()
        store = MemoryConversationStateStore()
        self.assertTrue(store.set(self.ctx.conv_key, state))
        self.ctx.normalized_message_text = "午时 先保留文字，稍后和务实调换编码"
        with (
            patch.object(self.chat, "conversation_state_store", store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.commands, "try_verify_pending_fly_proposal", AsyncMock(return_value=None)),
            patch.object(self.commands, "try_reselect_live_ticket", AsyncMock(return_value=None)),
        ):
            self.assertFalse(await self.chat._stage_resolve_current_pending_scope(self.ctx))
            self.assertIn("看到了你的选词请求", self.ctx.scoped_pending_response)
            self.assertTrue(await self.chat._stage_understand_operation(self.ctx))
            self.assertFalse(await self.chat._stage_finish_scoped_pending_response(self.ctx))
        self.assert_swap()

    async def test_invalid_model_result_preserves_scoped_response_for_existing_finisher(self):
        self.set_model(frame="none")
        with patch.object(self.chat, "_finish_ai_chat_matcher", new_callable=AsyncMock) as finish_ticket:
            self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
            self.assertTrue(await self.chat._stage_finish_scoped_pending_response(self.ctx))
        finish_ticket.assert_awaited_once_with("old ticket response")
        self.swap.assert_not_awaited()

    async def test_model_failures_and_non_json_preserve_existing_path(self):
        for error, content in (
            (RuntimeError("offline failure"), "{}"),
            (asyncio.TimeoutError(), "{}"),
            (None, "not JSON"),
            (None, "[]"),
            (None, '{"op":"swap","operands":["午时","务实"]}'),
        ):
            with self.subTest(error=type(error).__name__, content=content):
                self.client.create.side_effect = error
                self.client.create.return_value.choices[0].message.content = content
                self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
                self.assertEqual(self.ctx.response, "unchanged response")
                self.assertEqual(self.ctx.scoped_pending_response, "old ticket response")
        self.swap.assert_not_awaited()
        self.finish.assert_not_awaited()

    async def test_parser_boundary_error_and_timeout_preserve_existing_path(self):
        for error in (RuntimeError("offline parser"), asyncio.TimeoutError()):
            with self.subTest(error=type(error).__name__):
                with patch("keytao_bot.harness.semantic_parser.parse_operation", side_effect=error):
                    self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
                self.assertEqual(self.ctx.response, "unchanged response")
                self.assertEqual(self.ctx.scoped_pending_response, "old ticket response")
        self.swap.assert_not_awaited()

    async def test_cancelled_error_propagates_without_preview(self):
        self.client.create.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.chat._stage_understand_operation(self.ctx)
        self.swap.assert_not_awaited()
        self.finish.assert_not_awaited()
        self.assertEqual(self.ctx.response, "unchanged response")

    async def test_routing_and_shadow_reuse_one_model_call_and_one_log(self):
        os.environ["KEYTAO_SEMANTIC_SHADOW"] = "1"
        self.assertTrue(await self.chat._stage_understand_operation(self.ctx))
        self.client.create.assert_awaited_once()
        self.assertEqual(self.chat._semantic_shadow_tasks, set())
        shadow_logs = [call.args[0] for call in self.info.call_args_list
                       if call.args[0].startswith("[semantic_shadow]")]
        self.assertEqual(len(shadow_logs), 1)
        self.assertIn("model_op=swap", shadow_logs[0])
        self.assert_swap()

    async def test_stage_awaits_parser_before_returning_or_finishing(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked_parse(*args, **kwargs):
            started.set()
            await release.wait()
            return self.swap_raw()

        with patch("keytao_bot.harness.semantic_parser.parse_operation", side_effect=blocked_parse):
            stage = asyncio.create_task(self.chat._stage_understand_operation(self.ctx))
            try:
                await asyncio.wait_for(started.wait(), 0.5)
                self.assertFalse(stage.done())
                self.swap.assert_not_awaited()
                self.finish.assert_not_awaited()
                self.assertEqual(self.ctx.response, "unchanged response")
                release.set()
                self.assertTrue(await asyncio.wait_for(stage, 0.5))
            finally:
                if not stage.done():
                    stage.cancel()
                    await asyncio.gather(stage, return_exceptions=True)
        self.assert_swap()

    async def test_legacy_regex_gate_still_uses_shared_helper(self):
        message = "调整调换 午时 和 务实 词条"
        self.ctx.normalized_message_text = message
        self.assertEqual(understand_message(message).op, "swap")
        intent = self.chat.MessageCommandIntent(intent="entry_swap", keep_words=("午时", "务实"))
        response = await self.commands._try_handle_entry_swap_command(
            message, intent, "qq", self.ctx.user_id, self.ctx.space_key, self.ctx.owner_label,
        )
        self.assertEqual(response, "sealed swap preview")
        self.swap.assert_awaited_once_with(
            "午时", "务实", "qq", self.ctx.user_id, self.ctx.space_key, self.ctx.owner_label,
        )
        self.finish.assert_not_awaited()

    async def test_legacy_regex_gate_still_requires_matching_intent_and_words(self):
        for message, intent in (
            (self.ctx.normalized_message_text, self.chat.MessageCommandIntent(intent="entry_swap", keep_words=("午时", "务实"))),
            ("调整调换 午时 和 务实 词条", self.chat.MessageCommandIntent(intent="none")),
            ("调整调换 午时 和 务实 词条", self.chat.MessageCommandIntent(intent="entry_swap", keep_words=("务实", "午时"))),
        ):
            with self.subTest(message=message, intent=intent):
                self.assertIsNone(await self.commands._try_handle_entry_swap_command(
                    message, intent, "qq", self.ctx.user_id, self.ctx.space_key, self.ctx.owner_label,
                ))
        self.swap.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
