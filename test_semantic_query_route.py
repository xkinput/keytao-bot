"""Offline model-first query, conversation and pending-ticket isolation checks."""

import asyncio
import copy
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import test_state_machine as harness
from test_semantic_parser import FakeClient, raw_operation

from keytao_bot.harness.conversation import ConversationAddress
from keytao_bot.harness.state import (
    DraftOperationCoordinator,
    MemoryConversationStateStore,
    PendingAddWord,
)


class SemanticQueryRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.store = MemoryConversationStateStore()
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id="semantic-query-owner", space_type="group",
            space_id="semantic-query-group", speaker_name="Offline owner",
        )
        self.author = self.memory.conversation_address
        self.other = ConversationAddress.group("qq", "semantic-query-group", "other-user")
        self.client = FakeClient()
        self.finish = AsyncMock()
        self.real_main = self.chat.get_ai_response_core
        self.main = AsyncMock(return_value="这是离线解释。")
        self.remember = MagicMock()
        self.tool = AsyncMock(side_effect=self.lookup)
        self.entries = {"济济": "jjes", "积极": "jjuu", "午时": "wj"}
        self.history = []
        self.patches = (
            patch.dict(os.environ, {
                "KEYTAO_SEMANTIC_ROUTE_OPS": "query,chat,help",
                "KEYTAO_SEMANTIC_SHADOW": "0",
            }),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "call_tool_function", self.tool),
            patch.object(self.chat, "get_ai_response_core", self.main),
            patch.object(self.chat, "build_reply_context", AsyncMock(return_value="Offline quote context")),
            patch.object(self.chat, "get_history", side_effect=lambda *args: self.history),
            patch.object(self.chat, "remember_conversation", self.remember),
            patch.object(self.chat, "schedule_memory_compaction"),
            patch.object(self.chat, "_finish_ai_chat_response", self.finish),
            patch.object(self.chat.logger, "info"),
            patch.object(self.chat.logger, "warning"),
        )
        for context in self.patches:
            context.start()
            self.addCleanup(context.stop)
        self.turn_token = self.chat._current_conversational_turn.set(None)
        self.addCleanup(self.chat._current_conversational_turn.reset, self.turn_token)

    async def asyncTearDown(self):
        tasks = tuple(self.chat._semantic_shadow_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.chat._semantic_shadow_tasks.clear()

    async def lookup(self, name, args, platform, user_id, **kwargs):
        self.assertEqual(name, "keytao_lookup_by_word", "A semantic query must never generate candidates")
        self.assertEqual((platform, user_id), ("qq", self.author.actor_id))
        self.assertEqual(set(args), {"word"})
        word = args["word"]
        code = self.entries.get(word)
        return json.dumps({
            "success": True,
            "phrases": [{"word": word, "code": code, "type": "Phrase"}] if code else [],
        }, ensure_ascii=False)

    def context(self, message):
        return self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.author.actor_id,
            message_text=message, normalized_message_text=message,
            conv_key=self.author, space_key=self.author.space_key,
            memory_context=self.memory, owner_label="Offline owner",
            command_intent_for=AsyncMock(return_value=self.chat.MessageCommandIntent()),
        )

    def set_model(self, op="query", operands=("济济",), **changes):
        self.client.create.side_effect = None
        raw = raw_operation(op=op, operands=list(operands), codes=[], frame="none", **changes)
        self.client.create.return_value.choices[0].message.content = json.dumps(raw, ensure_ascii=False)
        return raw

    def ticket(self, owner):
        pending = PendingAddWord(
            "午时", "wj", [("wj", False), ("wja", False)],
            server_candidates=[("wj", False), ("wja", False)],
        )
        self.assertTrue(self.store.set(owner, pending, space_key=owner.space_key, owner_label="Offline owner"))
        record = self.store.get_record(owner)
        prompt = self.chat._render_live_single_candidate_record(record)
        self.assertTrue(self.store.bind_origin_prompt_digest(owner, self.chat._prompt_capability_digest(prompt)))
        self.history = [{"role": "assistant", "content": prompt}]
        return record, copy.deepcopy(record), prompt

    def assert_preserved(self, owner, record, original):
        self.assertIs(self.store.get_record(owner), record)
        self.assertEqual(record, original)

    def assert_main(self, message, *lookup_fragments):
        self.main.assert_awaited_once()
        call = self.main.await_args
        self.assertEqual(call.args[0], message)
        context = call.args[4] if len(call.args) > 4 else call.kwargs["reply_context"]
        for fragment in lookup_fragments:
            self.assertIn(fragment, context)
        self.assertTrue(call.kwargs["released_pending_turn"])
        self.assertTrue(call.kwargs["preserve_pending"])
        self.assertTrue(call.kwargs["conversation_only"])

    async def assert_finished(self, ctx):
        self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
        self.finish.assert_awaited_once()
        self.assertEqual(self.finish.await_args.args[4], ctx.response)
        self.remember.assert_called_once_with(self.author, self.memory, ctx.normalized_message_text, ctx.response)

    async def test_quoted_other_owner_meaning_query_looks_up_only_operand_and_calls_main(self):
        record, original, prompt = self.ticket(self.other)
        ctx = self.context("济济啥意思")
        ctx.reply_reference = harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, mentioned_user_ids=(self.other.actor_id,), text=prompt,
        )
        self.set_model()
        await self.assert_finished(ctx)
        self.tool.assert_awaited_once_with("keytao_lookup_by_word", {"word": "济济"}, "qq", self.author.actor_id)
        self.assert_main("济济啥意思", "济济", "jjes")
        self.assertIsNone(self.store.get_record(self.author))
        self.assert_preserved(self.other, record, original)

    async def test_code_question_uses_operand_lookup_and_passes_codes_to_main(self):
        ctx = self.context("积极怎么打")
        self.main.return_value = "积极的编码是 jjuu。"
        self.set_model(operands=("积极",))
        await self.assert_finished(ctx)
        self.tool.assert_awaited_once_with("keytao_lookup_by_word", {"word": "积极"}, "qq", self.author.actor_id)
        self.assert_main("积极怎么打", "积极", "jjuu")
        self.assertIn("jjuu", ctx.response)
        self.assertIsNone(self.store.get_record(self.author))

    async def test_exact_word_returns_existing_code_without_main_or_ticket(self):
        ctx = self.context("积极")
        self.set_model(operands=("积极",))
        await self.assert_finished(ctx)
        self.assertIn("积极", ctx.response)
        self.assertIn("jjuu", ctx.response)
        self.main.assert_not_awaited()
        self.tool.assert_awaited_once()
        self.assertIsNone(self.store.get_record(self.author))

    async def test_missing_word_does_not_prepare_candidates_or_create_ticket(self):
        ctx = self.context("云翻")
        self.set_model(operands=("云翻",))
        await self.assert_finished(ctx)
        self.assertIn("云翻", ctx.response)
        self.assertTrue(any(marker in ctx.response for marker in ("未收录", "没有", "未找到", "未在词库", "不存在")), ctx.response)
        self.tool.assert_awaited_once_with("keytao_lookup_by_word", {"word": "云翻"}, "qq", self.author.actor_id)
        self.main.assert_not_awaited()
        self.assertIsNone(self.store.get_record(self.author))

    async def test_multiword_query_uses_only_each_literal_operand_and_preserves_ticket(self):
        record, original, _prompt = self.ticket(self.author)
        ctx = self.context("济济 积极")
        self.set_model(operands=("济济", "积极"))
        await self.assert_finished(ctx)
        self.assertEqual([call.args[1] for call in self.tool.await_args_list], [{"word": "济济"}, {"word": "积极"}])
        self.assertIn("jjes", ctx.response)
        self.assertIn("jjuu", ctx.response)
        self.main.assert_not_awaited()
        self.assert_preserved(self.author, record, original)

    async def test_chat_with_own_live_ticket_and_release_enabled_calls_only_main(self):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "chat,pending_release"
        record, original, _prompt = self.ticket(self.author)
        ctx = self.context("哈哈好的")
        self.set_model("chat", ())
        await self.assert_finished(ctx)
        self.assert_main("哈哈好的")
        self.tool.assert_not_awaited()
        self.assert_preserved(self.author, record, original)

    async def test_help_quoting_other_ticket_reaches_main_without_claiming_it(self):
        record, original, prompt = self.ticket(self.other)
        ctx = self.context("应该怎么问你")
        ctx.reply_reference = harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, mentioned_user_ids=(self.other.actor_id,), text=prompt,
        )
        self.set_model("help", ())
        await self.assert_finished(ctx)
        self.assert_main("应该怎么问你")
        self.tool.assert_not_awaited()
        self.assertIsNone(self.store.get_record(self.author))
        self.assert_preserved(self.other, record, original)

    async def test_real_main_core_disables_tools_and_structural_fallback_in_isolated_state(self):
        record, original, _prompt = self.ticket(self.author)
        ctx = self.context("清空草稿")
        self.set_model("help", ())
        seen = []
        forbidden = AsyncMock(side_effect=AssertionError("A semantic conversation must never execute tools"))
        self.tool.side_effect = forbidden.side_effect

        async def isolated_main(orchestrator, *, message, context, **kwargs):
            seen.append(context)
            self.assertEqual(message, "清空草稿")
            self.assertTrue(context.conversation_only)
            self.assertFalse(context.mutations_allowed)
            self.assertIsNot(orchestrator._state_store, self.store)
            self.assertIsNone(orchestrator._state_store.get_record(self.author))
            sandbox = PendingAddWord("云帆", "yjfj", [("yjfj", False)])
            self.assertTrue(orchestrator._state_store.set(self.author, sandbox))
            orchestrator._state_store.delete(self.author)
            self.assert_preserved(self.author, record, original)
            self.assertEqual(self.chat._structural_draft_management_intent(message).intent, "draft_clear")
            self.assertIsNone(await orchestrator._deterministic_fallback_handler(message, context))
            return "请发送完整的词条请求。"

        with (
            patch.object(self.chat, "get_ai_response_core", self.real_main),
            patch.object(self.chat, "OPENAI_API_KEY", "offline-key"),
            patch.object(self.chat.AgentOrchestrator, "run", isolated_main),
            patch.object(self.chat, "handle_pending_message_core", forbidden),
            patch.object(self.chat, "_try_handle_draft_management_command", forbidden),
            patch.object(self.chat, "_try_handle_encoding_rule_question", forbidden),
            patch.object(self.chat.tool_executor, "call", forbidden),
            patch.object(self.chat.tool_executor, "_invoke_effective_tool", forbidden),
            patch.object(self.chat.tool_executor, "replay_shift_plan", forbidden),
            patch.object(self.chat.tool_executor, "replay_routed_warning", forbidden),
            patch.object(self.chat.memory_store, "get_context_block", return_value=""),
            patch.object(self.chat, "get_group_history_context", return_value=""),
        ):
            await self.assert_finished(ctx)
        self.assertEqual(len(seen), 1)
        self.assertEqual(ctx.response, "请发送完整的词条请求。")
        forbidden.assert_not_awaited()
        self.tool.assert_not_awaited()
        self.assert_preserved(self.author, record, original)

    async def assert_legacy_path(self, ctx):
        self.assertFalse(await self.chat._stage_understand_operation_early(ctx))
        self.assertFalse(getattr(ctx, "semantic_nonwrite_route", ""))
        self.main.assert_not_awaited()
        self.tool.assert_not_awaited()
        self.finish.assert_not_awaited()
        legacy = AsyncMock(return_value="Offline legacy reply")
        with (
            patch.object(self.chat, "_set_conversational_turn"),
            patch.object(self.chat, "_try_handle_explicit_reading_disambiguation", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_handle_compound_shift_modified_add_command", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_handle_shift_modified_add_command", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_handle_complete_add_command", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_recover_reviewed_add_from_history", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_handle_simple_single_word_query", legacy),
        ):
            self.assertFalse(await self.chat._stage_handle_simple_word_query(ctx))
        legacy.assert_awaited_once_with(
            ctx.normalized_message_text, "qq", self.author.actor_id,
            self.author, self.author.space_key, "Offline owner", actionable_lookup=False,
        )
        self.assertEqual(ctx.response, "Offline legacy reply")

    async def test_operand_absent_from_message_keeps_legacy_input_and_kwargs(self):
        self.set_model(operands=("济济", "不存在"))
        await self.assert_legacy_path(self.context("济济啥意思"))

    async def test_absent_or_nonexact_tokens_keep_legacy_path(self):
        for token in (None, "", "queries,chats,helps", "QUERY,CHAT,HELP", "query;chat;help"):
            with self.subTest(token=token):
                if token is None:
                    os.environ.pop("KEYTAO_SEMANTIC_ROUTE_OPS", None)
                else:
                    os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = token
                self.client.create.reset_mock()
                self.set_model()
                await self.assert_legacy_path(self.context("济济啥意思"))
                self.client.create.assert_not_awaited()

    async def test_each_exact_token_enables_only_its_own_route(self):
        for token, op, operands in (("chat", "query", ("济济",)), ("query", "chat", ()), ("help", "chat", ())):
            with self.subTest(token=token, op=op):
                os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = token
                self.set_model(op, operands)
                await self.assert_legacy_path(self.context("济济啥意思"))

    async def test_lookup_failure_stays_read_only_without_candidate_fallback(self):
        self.tool.side_effect = None
        self.tool.return_value = json.dumps({"success": False, "error": "Offline unavailable"})
        ctx = self.context("济济")
        self.set_model()
        await self.assert_finished(ctx)
        self.tool.assert_awaited_once_with("keytao_lookup_by_word", {"word": "济济"}, "qq", self.author.actor_id)
        self.main.assert_not_awaited()
        self.assertIsNone(self.store.get_record(self.author))

    async def test_model_failures_keep_legacy_path(self):
        for scenario in ("timeout", "invalid_json", "invalid_op"):
            with self.subTest(scenario=scenario):
                self.set_model()
                if scenario == "timeout":
                    self.client.create.side_effect = asyncio.TimeoutError()
                elif scenario == "invalid_json":
                    self.client.create.return_value.choices[0].message.content = "not JSON"
                else:
                    self.client.create.return_value.choices[0].message.content = json.dumps({"op": "QUERY"})
                await self.assert_legacy_path(self.context("济济啥意思"))

    async def test_query_requires_at_least_one_operand(self):
        self.set_model(operands=())
        await self.assert_legacy_path(self.context("济济啥意思"))

    async def test_raw_boolean_and_frame_validation_fail_closed(self):
        for field, value in (
            ("refers_to_pending", None), ("refers_to_pending", "false"),
            ("refers_to_pending", 0), ("submit_after", None),
            ("submit_after", "false"), ("frame", "invalid"),
        ):
            with self.subTest(field=field, value=value):
                raw = self.set_model()
                if value is None:
                    raw.pop(field)
                else:
                    raw[field] = value
                self.client.create.return_value.choices[0].message.content = json.dumps(raw, ensure_ascii=False)
                await self.assert_legacy_path(self.context("济济啥意思"))

    async def test_pending_reference_cannot_route_as_an_unrelated_query(self):
        self.set_model(refers_to_pending=True)
        await self.assert_legacy_path(self.context("济济啥意思"))

    async def test_frame_backstop_question_still_routes_query(self):
        raw = self.set_model()
        raw["frame"] = "direct"
        self.client.create.return_value.choices[0].message.content = json.dumps(raw, ensure_ascii=False)
        ctx = self.context("济济怎么打？")
        await self.assert_finished(ctx)
        self.assertIn("frame_backstop:question", ctx.operation_violations)
        self.assert_main("济济怎么打？", "济济", "jjes")


if __name__ == "__main__":
    unittest.main()
