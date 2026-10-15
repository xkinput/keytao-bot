"""Offline author-ticket answer routing and release safety checks."""

import asyncio
import copy
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from test_semantic_parser import FakeClient, raw_operation

from keytao_bot.harness.conversation import ConversationAddress
from keytao_bot.harness.state import (
    DraftOperationCoordinator,
    MemoryConversationStateStore,
    PendingAddWord,
    PendingToolConfirm,
)


class SemanticPendingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.store = MemoryConversationStateStore()
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id="semantic-pending-owner", space_type="group",
            space_id="semantic-pending-group", speaker_name="Offline owner",
        )
        self.author = self.memory.conversation_address
        self.other = ConversationAddress.group("qq", "semantic-pending-group", "other-user")
        self.client = FakeClient()
        self.finish = AsyncMock()
        self.add = AsyncMock(return_value="offline addition")
        self.submit = AsyncMock(return_value="offline addition and submission")
        self.tool = AsyncMock(side_effect=AssertionError("Unexpected offline tool call"))
        self.history = []
        patches = (
            patch.dict(os.environ, {
                "KEYTAO_SEMANTIC_ROUTE_OPS": "pending_release,pending_answer",
                "KEYTAO_SEMANTIC_SHADOW": "0",
            }),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "_execute_add_to_draft", self.add),
            patch.object(self.chat, "_execute_add_to_draft_and_submit", self.submit),
            patch.object(self.chat, "call_tool_function", self.tool),
            patch.object(self.chat, "_finish_ai_chat_response", self.finish),
            patch.object(self.chat, "remember_conversation"),
            patch.object(self.chat, "get_history", side_effect=lambda *args: self.history),
            patch.object(self.chat.logger, "info"),
            patch.object(self.chat.logger, "warning"),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)
        self.ctx = self.context("就这样加吧")
        self.set_model("confirm", refers_to_pending=True)

    async def asyncTearDown(self):
        tasks = tuple(self.chat._semantic_shadow_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.chat._semantic_shadow_tasks.clear()
        self.tool.assert_not_awaited()

    def context(self, message, *, actor=None, conv_key=None):
        actor = actor or self.author
        return self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=actor.actor_id,
            normalized_message_text=message, message_text=message,
            conv_key=conv_key or actor, space_key=actor.space_key,
            memory_context=self.memory, owner_label="Offline owner",
            command_intent_for=AsyncMock(return_value=self.chat.MessageCommandIntent()),
        )

    def set_model(self, op, operands=(), codes=(), **changes):
        self.client.create.side_effect = None
        self.client.create.return_value.choices[0].message.content = json.dumps(
            raw_operation(op=op, operands=list(operands), codes=list(codes), **changes),
            ensure_ascii=False,
        )

    def ticket(self):
        state = PendingAddWord(
            "午时", "wj", [("wj", False), ("wja", False)],
            server_candidates=[("wj", False), ("wja", False)],
        )
        self.assertTrue(self.store.set(
            self.author, state, space_key=self.author.space_key, owner_label="Offline owner",
        ))
        record = self.store.get_record(self.author)
        prompt = self.chat._render_live_single_candidate_record(record)
        self.assertTrue(prompt)
        self.assertTrue(self.store.bind_origin_prompt_digest(
            self.author, self.chat._prompt_capability_digest(prompt),
        ))
        self.history = [{"role": "assistant", "content": prompt}]
        return record, prompt

    def assert_untouched(self, record, original):
        self.assertIs(self.store.get_record(self.author), record)
        self.assertEqual(record, original)
        self.add.assert_not_awaited()
        self.submit.assert_not_awaited()

    async def test_stale_candidate_dispatches_complete_model_swap(self):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] += ",swap"
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        self.ctx = self.context("午时和务实互换一下位置")
        self.set_model("swap", ("午时", "务实"), refers_to_pending=False)
        swap = AsyncMock(return_value="offline sealed swap plan")
        with patch.object(self.commands, "try_handle_model_entry_swap", swap):
            self.assertTrue(await self.chat._stage_understand_operation_early(self.ctx))
        swap.assert_awaited_once_with(
            "午时", "务实", "qq", self.author.actor_id,
            self.author.space_key, "Offline owner",
        )
        self.assertFalse(self.ctx.released_pending_turn)
        self.assertEqual(self.ctx.response, "offline sealed swap plan")
        self.assert_untouched(record, original)

    async def test_real_swap_replaces_ticket_for_both_chain_shapes(self):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] += ",swap"
        for other_code in ("wj", "wja"):
            with self.subTest(other_code=other_code):
                record, _prompt = self.ticket()
                self.ctx = self.context("午时和务实互换一下位置")
                self.set_model("swap", ("午时", "务实"), refers_to_pending=False)
                rows = {
                    "午时": {"word": "午时", "code": "wj", "type": "Phrase", "weight": 100},
                    "务实": {"word": "务实", "code": other_code, "type": "Phrase", "weight": 101},
                }
                async def seal(*args, **kwargs):
                    self.assertTrue(self.store.set(self.author, PendingToolConfirm(
                        "keytao_shift_phrase_code", {"word": "午时", "target_code": "wj"},
                    )))
                    return "offline sealed swap plan"

                execute = AsyncMock(side_effect=seal)
                with (
                    patch.object(self.commands, "_load_merged_word_entries", AsyncMock(return_value=rows)),
                    patch.object(self.commands, "_execute_shift_to_code", execute),
                ):
                    self.assertTrue(await self.chat._stage_understand_operation_early(self.ctx))
                self.assertEqual(self.ctx.response, "offline sealed swap plan")
                execute.assert_awaited_once()
                self.assertEqual(execute.await_args.kwargs["expected_codes"], [other_code, "wj"])
                self.assertIsNot(self.store.get_record(self.author), record)
                self.assertEqual(self.store.get(self.author).function_name, "keytao_shift_phrase_code")

    async def test_quoted_complete_swap_supersedes_wrong_model_answer(self):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] += ",swap"
        self.ticket()
        self.ctx = self.context("午时和务实互换一下位置")
        self.ctx.reply_reference = harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, text="另一条不相关消息",
        )
        self.set_model("confirm", refers_to_pending=True)
        swap = AsyncMock(return_value="offline sealed swap plan")
        with patch.object(self.commands, "try_handle_model_entry_swap", swap):
            self.assertTrue(await self.chat._stage_understand_operation_early(self.ctx))
        swap.assert_awaited_once_with(
            "午时", "务实", "qq", self.author.actor_id, self.author.space_key, "Offline owner",
        )
        self.assertFalse(self.ctx.semantic_pending_released)

    async def test_query_release_bypasses_ticket_claims_and_reaches_query_path(self):
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        self.ctx = self.context("这个词怎么打")
        self.set_model("query", frame="none", refers_to_pending=False)
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertTrue(self.ctx.released_pending_turn)
        query = AsyncMock(return_value="offline query reply")
        claims = AsyncMock(side_effect=AssertionError("Released turn must not claim a ticket"))
        with (
            patch.object(self.commands, "try_verify_pending_fly_proposal", claims),
            patch.object(self.commands, "try_reselect_live_ticket", claims),
            patch.object(self.chat, "handle_pending_message_core", claims),
            patch.object(self.chat, "_try_handle_explicit_reading_disambiguation", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_handle_compound_shift_modified_add_command", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_handle_shift_modified_add_command", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_handle_complete_add_command", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_recover_reviewed_add_from_history", AsyncMock(return_value=None)),
            patch.object(self.chat, "_try_handle_simple_single_word_query", query),
        ):
            for stage in (
                self.chat._stage_claim_offered_answer,
                self.chat._stage_resolve_current_pending_scope,
                self.chat._stage_apply_scoped_pending_intent,
                self.chat._stage_execute_pending_state,
            ):
                self.assertFalse(await stage(self.ctx), stage.__name__)
            self.assertFalse(await self.chat._stage_handle_simple_word_query(self.ctx))
        claims.assert_not_awaited()
        query.assert_awaited_once()
        self.assertEqual(self.ctx.response, "offline query reply")
        self.assert_untouched(record, original)

    async def test_release_skips_explicit_entry_shortcuts_that_can_claim_ticket(self):
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        self.ctx = self.context("这个词怎么打")
        self.set_model("query", frame="none", refers_to_pending=False)
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        shortcut = AsyncMock(side_effect=AssertionError("Released query must bypass mutation shortcuts"))
        with (
            patch("keytao_bot.utils.same_code_reorder.try_handle_same_code_reorder", shortcut),
            patch.object(self.commands, "try_handle_move_to_code_command", shortcut),
            patch.object(self.commands, "try_handle_explicit_entry_code_command", shortcut),
        ):
            self.assertFalse(await self.chat._stage_handle_explicit_entry_operation(self.ctx))
        shortcut.assert_not_awaited()
        self.assert_untouched(record, original)

    async def test_release_skips_later_mutation_and_reset_entrypoints(self):
        cases = (
            (self.chat._stage_handle_draft_management, "draft_clear"),
            (self.chat._stage_handle_replace_character, "replace_char"),
            (self.chat._stage_handle_explicit_draft_conversation_reset, "draft_clear"),
            (self.chat._stage_handle_clear_history, "clear_history"),
            (self.chat._stage_submit_current_draft, "draft_submit"),
            (self.chat._stage_handle_completed_draft_undo, "cancel"),
        )
        for stage, intent in cases:
            with self.subTest(stage=stage.__name__):
                record, _prompt = self.ticket()
                original = copy.deepcopy(record)
                self.ctx = self.context("这个词怎么打")
                self.ctx.generic_command_intent = self.chat.MessageCommandIntent(intent=intent, confidence=1.0)
                self.set_model("query", frame="none", refers_to_pending=False)
                self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                write = AsyncMock(side_effect=AssertionError("Released turn must not enter a later mutation"))
                with (
                    patch.object(self.chat, "_try_handle_draft_management_command", write),
                    patch.object(self.chat, "_try_handle_replace_char", write),
                    patch.object(self.chat, "_try_handle_draft_clear_command", write),
                    patch.object(self.chat, "_clear_conversation_state", write),
                    patch.object(self.chat, "_perform_submit_current_draft", write),
                    patch.object(self.chat, "handle_completed_undo", write),
                    patch.object(self.chat, "_schedule_background_draft_operation", side_effect=AssertionError("No released-turn background write")) as schedule,
                    patch.object(self.chat._chat_routing, "_is_explicit_draft_conversation_reset", return_value=True),
                    patch.object(self.chat, "_message_authorizes_clear_history", return_value=True),
                    patch.object(self.store, "get_record", side_effect=AssertionError("Released stage must return before claiming state")),
                ):
                    self.assertFalse(await stage(self.ctx))
                write.assert_not_awaited()
                schedule.assert_not_called()
                self.assertIsNone(self.chat.draft_operation_coordinator.get(self.author))
                self.assert_untouched(record, original)

    async def test_release_challenge_stage_keeps_original_prompt_digest(self):
        record, _prompt = self.ticket()
        self.assertIsNotNone(self.store.arm_reconfirmation(self.author))
        original = copy.deepcopy(record)
        self.ctx = self.context("这个词怎么打")
        self.set_model("query", frame="none", refers_to_pending=False)
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.ctx.response = "新查询结果。回复「加入」或「取消」。"
        response = self.ctx.response
        with patch.object(self.chat, "_append_pending_ticket_challenge", side_effect=AssertionError("Released reply must not rebind old prompt")) as challenge:
            self.assertFalse(await self.chat._stage_append_ticket_challenge(self.ctx))
        challenge.assert_not_called()
        self.assertEqual(self.ctx.response, response)
        self.assert_untouched(record, original)

    async def test_release_main_fallback_has_no_live_ticket_or_mutation_capability(self):
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        self.ctx = self.context("那就这样吧哈哈")
        self.set_model("query", frame="none", refers_to_pending=False)
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertTrue(self.ctx.released_pending_turn)
        seen = []

        async def isolated_fallback(orchestrator, *, message, context, **kwargs):
            seen.append((message, context))
            self.assertIsNot(orchestrator._state_store, self.store)
            self.assertIsNone(orchestrator._state_store.get_record(self.author))
            self.assertFalse(context.mutations_allowed)
            sandbox = PendingAddWord("云帆", "yjfj", [("yjfj", False)])
            self.assertTrue(orchestrator._state_store.set(self.author, sandbox))
            orchestrator._state_store.delete(self.author)
            self.assertIsNone(orchestrator._state_store.get_record(self.author))
            self.assert_untouched(record, original)
            return await orchestrator._deterministic_fallback_handler(message, context) or "offline isolated fallback"

        claims = AsyncMock(side_effect=AssertionError("Released fallback must not execute old ticket"))
        token = self.chat._current_conversational_turn.set(None)
        try:
            with (
                patch.object(self.chat, "OPENAI_API_KEY", "offline-key"),
                patch.object(self.chat.AgentOrchestrator, "run", isolated_fallback),
                patch.object(self.chat, "handle_pending_message_core", claims),
                patch.object(self.chat, "_try_handle_encoding_rule_question", AsyncMock(return_value=None)),
                patch.object(self.chat.memory_store, "get_context_block", return_value=""),
                patch.object(self.chat, "get_group_history_context", return_value=""),
                patch.object(self.chat, "build_reply_context", AsyncMock(return_value="")),
            ):
                self.assertFalse(await self.chat._stage_generate_ai_response(self.ctx))
        finally:
            self.chat._current_conversational_turn.reset(token)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], "那就这样吧哈哈")
        self.assertFalse(seen[0][1].mutations_allowed)
        self.assertEqual(self.ctx.response, "offline isolated fallback")
        claims.assert_not_awaited()
        self.assert_untouched(record, original)

    async def test_release_fallback_cannot_bypass_sandbox_with_structural_draft_clear(self):
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        self.ctx = self.context("清空草稿")
        from keytao_bot.harness.operation_request import OperationRequest
        self.set_model("draft_clear", refers_to_pending=False)
        with patch("keytao_bot.harness.operation_request.understand_message", return_value=OperationRequest()):
            self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertTrue(self.ctx.released_pending_turn)
        self.assertEqual(self.chat._structural_draft_management_intent("清空草稿").intent, "draft_clear")
        seen = []

        async def fallback_only(orchestrator, *, message, context, **kwargs):
            seen.append(context)
            self.assertIsNot(orchestrator._state_store, self.store)
            self.assertIsNone(orchestrator._state_store.get_record(self.author))
            self.assertFalse(context.mutations_allowed)
            return await orchestrator._deterministic_fallback_handler(message, context) or "offline release clear fallback"

        draft = AsyncMock(side_effect=AssertionError("Released fallback must not call real draft mutation helper"))
        token = self.chat._current_conversational_turn.set(None)
        try:
            with (
                patch.object(self.chat, "OPENAI_API_KEY", "offline-key"),
                patch.object(self.chat.AgentOrchestrator, "run", fallback_only),
                patch.object(self.chat, "_try_handle_draft_management_command", draft),
                patch.object(self.chat, "_try_handle_encoding_rule_question", AsyncMock(return_value=None)),
                patch.object(self.chat.memory_store, "get_context_block", return_value=""),
                patch.object(self.chat, "get_group_history_context", return_value=""),
                patch.object(self.chat, "build_reply_context", AsyncMock(return_value="")),
            ):
                self.assertFalse(await self.chat._stage_generate_ai_response(self.ctx))
        finally:
            self.chat._current_conversational_turn.reset(token)
        self.assertEqual(len(seen), 1)
        self.assertEqual(self.ctx.response, "offline release clear fallback")
        draft.assert_not_awaited()
        self.assert_untouched(record, original)

    async def test_real_query_new_existing_and_multiword_results_preserve_ticket(self):
        code_by_word = {"奈飞": "nhfw", "云帆": "yjfj"}
        for message, existing in (("奈飞", False), ("奈飞", True), ("奈飞 云帆", False)):
            with self.subTest(message=message, existing=existing):
                record, _prompt = self.ticket()
                original = copy.deepcopy(record)
                self.ctx = self.context(message)
                words = tuple(message.split())
                self.set_model("query", words, frame="none", refers_to_pending=False)
                self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))

                async def query_tool(name, args, platform, user_id, **kwargs):
                    self.assertEqual((platform, user_id), ("qq", self.author.actor_id))
                    if name == "keytao_pending_items_by_words":
                        return json.dumps({"success": True, "complete": True, "items": []})
                    word = args["word"]
                    code = code_by_word[word]
                    if name == "keytao_lookup_by_word":
                        return json.dumps({"success": True, "phrases": [
                            {"word": word, "code": code, "type": "Phrase"},
                        ] if existing else []})
                    self.assertEqual(name, "keytao_prepare_reviewed_add")
                    return json.dumps({
                        "success": True, "word": word, "recommendedCode": code,
                        "needsManualReview": False,
                        "pronunciations": [{
                            "pinyin": "offline reading", "recommendedCode": code,
                            "candidateStatuses": [{"code": code, "occupied": False}],
                        }],
                    })

                dispatch = AsyncMock(side_effect=query_tool)
                with (
                    patch.object(self.commands, "call_tool_function", dispatch),
                    patch.object(self.commands, "_get_simple_word_query_words", AsyncMock(side_effect=lambda value: (value,))),
                ):
                    self.assertFalse(await self.chat._stage_handle_simple_word_query(self.ctx))
                for word in words:
                    self.assertIn(word, self.ctx.response)
                    self.assertIn(code_by_word[word], self.ctx.response)
                tokens = (
                    (self.chat._current_turn_message, self.chat._current_turn_message.set(message)),
                    (self.chat._current_conversational_turn, self.chat._current_conversational_turn.set(None)),
                    (self.chat.current_draft_delivery_claims, self.chat.current_draft_delivery_claims.set([])),
                )
                try:
                    with patch.object(self.chat._user_resolver, "resolved_binding_for_notice", return_value=True):
                        deliveries = (
                            self.chat._enforce_advertised_reply_contract(self.ctx.response, self.author),
                            self.chat._prepare_user_facing_reply(self.ctx.response, self.memory),
                        )
                    for delivered in deliveries:
                        for word in words:
                            self.assertIn(word, delivered)
                            self.assertIn(code_by_word[word], delivered)
                        self.assertNotIn("午时", delivered)
                        self.assertNotIn("wja", delivered)
                finally:
                    for variable, token in reversed(tokens):
                        variable.reset(token)
                self.assert_untouched(record, original)
                self.assertTrue(self.ctx.released_pending_turn)
                self.assertTrue(dispatch.await_count)

    async def test_model_selection_uses_ticket_code_once(self):
        self.ticket()
        self.ctx = self.context("我就要那个 wja")
        self.set_model("select_candidate", ("午时",), ("wja",), refers_to_pending=True)
        self.assertTrue(await self.chat._stage_understand_operation_early(self.ctx))
        self.add.assert_awaited_once()
        self.assertEqual(self.add.await_args.args[:2], ("午时", "wja"))
        self.assertIsNone(self.store.get_record(self.author))
        self.ctx = self.context("我就要那个 wja")
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.add.assert_awaited_once()

    async def test_nonliteral_nondefault_selection_falls_back_without_writing(self):
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        self.ctx = self.context("就这样加入并提交吧")
        self.set_model("select_candidate", ("午时",), ("wja",),
                       refers_to_pending=True, submit_after=True)
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertFalse(self.ctx.released_pending_turn)
        self.assert_untouched(record, original)

    async def test_all_model_answers_require_latest_bot_prompt_digest(self):
        for op, message, codes in (
            ("select_candidate", "我选 wja", ("wja",)),
            ("cancel", "这次算了吧", ()),
            ("pending_recode", "给我换一个编码吧", ()),
            ("confirm", "就这样吧", ()),
        ):
            for scenario in ("missing_history", "missing_digest", "older_prompt", "matching_old_quote"):
                with self.subTest(op=op, scenario=scenario):
                    record, prompt = self.ticket()
                    if scenario == "missing_history":
                        self.history = []
                    elif scenario == "missing_digest":
                        self.assertTrue(self.store.bind_origin_prompt_digest(self.author, ""))
                    else:
                        self.history.append({"role": "assistant", "content": "另一个更新的回复"})
                    original = copy.deepcopy(record)
                    self.ctx = self.context(message)
                    if scenario == "matching_old_quote":
                        self.ctx.reply_reference = harness.ReplyReferenceInfo(
                            is_reply=True, is_to_bot=True, text=prompt,
                        )
                    self.set_model(op, codes=codes, refers_to_pending=True)
                    execute = AsyncMock(return_value="unexpected stale ticket execution")
                    with patch.object(self.chat, "handle_pending_message_core", execute):
                        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                    execute.assert_not_awaited()
                    self.assert_untouched(record, original)

    async def test_advertised_closed_answers_survive_model_pending_release(self):
        for message, code in (
            ("加入", "wj"), ("加入并提交", "wj"), ("确认", "wj"),
            ("1", "wj"), ("2", "wja"), ("二", "wja"),
            ("wj", "wj"), ("WJA", "wja"), ("取消", None),
        ):
            with self.subTest(message=message):
                self.add.reset_mock()
                self.submit.reset_mock()
                record, prompt = self.ticket()
                if message == "确认":
                    prompt += "\n回复「确认」执行，或「取消」。"
                    self.assertTrue(self.store.bind_origin_prompt_digest(
                        self.author, self.chat._prompt_capability_digest(prompt),
                    ))
                    self.history = [{"role": "assistant", "content": prompt}]
                self.ctx = self.context(message)
                self.set_model("chat", frame="none", refers_to_pending=False)
                self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                self.assertFalse(self.ctx.semantic_pending_released)
                self.assertFalse(self.ctx.released_pending_turn)
                response = await self.commands.handle_pending_message_core(
                    message, "qq", self.author.actor_id, self.author,
                    history=self.history, space_key=self.author.space_key,
                    owner_label="Offline owner", allow_intent_model=False,
                )
                self.assertIsNotNone(response)
                if message == "加入并提交":
                    self.submit.assert_awaited_once()
                    self.assertEqual(self.submit.await_args.args[:2], ("午时", code))
                    self.add.assert_not_awaited()
                elif code is not None:
                    self.add.assert_awaited_once()
                    self.assertEqual(self.add.await_args.args[:2], ("午时", code))
                    self.submit.assert_not_awaited()
                else:
                    self.add.assert_not_awaited()
                    self.submit.assert_not_awaited()
                self.assertIsNone(self.store.get_record(self.author))

    async def test_closed_assent_uses_live_ticket_advertisement_after_unrelated_reply(self):
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        self.history.append({"role": "assistant", "content": "另一个更新的回复"})
        self.ctx = self.context("加入")
        self.set_model("chat", frame="none", refers_to_pending=False)
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertFalse(self.ctx.semantic_pending_released)
        self.assertFalse(self.ctx.released_pending_turn)
        self.assert_untouched(record, original)

    async def test_nonliteral_default_selection_uses_confirm_canonical(self):
        from keytao_bot.harness.operation_request import OperationRequest

        record, prompt = self.ticket()
        for submit_after, message, expected in (
            (False, "就这样吧", "加入"), (True, "就这样吧", "加入并提交"),
            (False, "WJA", "加入"),
        ):
            with self.subTest(submit_after=submit_after, message=message):
                request = OperationRequest(
                    op="select_candidate", operands=("午时",), codes=("wj",),
                    frame="direct", refers_to_pending=True, submit_after=submit_after,
                )
                self.assertEqual(self.commands.render_understood_pending_answer(
                    request, record.state, prompt, message,
                ), expected)

    async def test_literal_nondefault_code_and_indices_keep_selection(self):
        for message in ("我选 WJA", "2", "二"):
            with self.subTest(message=message):
                self.add.reset_mock()
                self.ticket()
                self.ctx = self.context(message)
                self.set_model("select_candidate", ("午时",), ("wja",), refers_to_pending=True)
                self.assertTrue(await self.chat._stage_understand_operation_early(self.ctx))
                self.add.assert_awaited_once()
                self.assertEqual(self.add.await_args.args[:2], ("午时", "wja"))

    async def test_multiword_nonliteral_selection_requires_complete_default_confirmation(self):
        from keytao_bot.harness.operation_request import OperationRequest

        scopes = []
        for word, code in (("午时", "wj"), ("务实", "we")):
            candidates = [(code, False), (code + "a", False)]
            state = PendingAddWord(word, code, candidates, server_candidates=candidates)
            scopes.append({
                "word": word, "candidates": candidates, "occupiedWords": {},
                "orderingAssessments": [], "reviewedState": self.commands._pending_add_word_payload(state),
            })
        state = PendingToolConfirm("keytao_batch_add_to_draft", {
            "items": [{"action": "Create", "word": word, "code": code}
                      for word, code in (("午时", "wj"), ("务实", "we"))],
            "_candidate_scopes": scopes, "_reviewed_multi_word": True,
            "_query_words": ["午时", "务实"],
        })
        for words, codes, message, expected in (
            (("午时", "务实"), ("wja", "we"), "就这样加吧", ""),
            (("午时",), ("wj",), "就这样加吧", ""),
            (("午时", "务实"), ("wj", "we"), "就这样加吧", "加入"),
            (("午时", "务实"), ("wja", "wea"), "午时 WJA，务实 WEA", "午时 wja，务实 wea，加入"),
            (("午时", "务实"), ("wja", "wea"), "午时 2，务实 2", "午时 wja，务实 wea，加入"),
            (("午时", "务实"), (), "午时 2，务实 2", "午时 wja，务实 wea，加入"),
        ):
            with self.subTest(words=words, codes=codes, message=message):
                request = OperationRequest(op="select_candidate", operands=words, codes=codes,
                                           frame="direct", refers_to_pending=True)
                self.assertEqual(self.commands.render_understood_pending_answer(
                    request, state, "回复「加入」写入草稿，或「加入并提交」。", message,
                ), expected)

    async def test_model_selection_rejects_unticketed_codes_and_operands(self):
        for message, operands, codes in (
            ("我选 zzz", ("午时",), ("zzz",)),
            ("务实选择 wj", ("务实",), ("wj",)),
            ("我选 wja wj", ("午时", "务实"), ("wja", "wj")),
        ):
            with self.subTest(message=message):
                record, _prompt = self.ticket()
                original = copy.deepcopy(record)
                self.ctx = self.context(message)
                self.set_model("select_candidate", operands, codes, refers_to_pending=True)
                self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                self.assertFalse(self.ctx.released_pending_turn)
                self.assertIsNone(self.ctx.response)
                self.assert_untouched(record, original)

    async def test_literal_index_without_model_code_roundtrips_to_advertised_choice(self):
        self.ticket()
        self.ctx = self.context("2")
        self.set_model("select_candidate", refers_to_pending=True)
        self.assertTrue(await self.chat._stage_understand_operation_early(self.ctx))
        self.add.assert_awaited_once()
        self.assertEqual(self.add.await_args.args[:2], ("午时", "wja"))
        self.assertIsNone(self.store.get_record(self.author))

    async def test_multiword_selection_keeps_each_code_bound_to_its_word(self):
        scopes = []
        for word, code in (("午时", "wj"), ("务实", "we")):
            state = PendingAddWord(word, code, [(code, False)], server_candidates=[(code, False)])
            scopes.append({
                "word": word, "candidates": list(state.candidates),
                "occupiedWords": {}, "orderingAssessments": [],
                "reviewedState": self.commands._pending_add_word_payload(state),
            })
        state = PendingToolConfirm("keytao_batch_add_to_draft", {
            "items": [{"action": "Create", "word": word, "code": code}
                      for word, code in (("午时", "wj"), ("务实", "we"))],
            "_candidate_scopes": scopes, "_reviewed_multi_word": True,
            "_query_words": ["午时", "务实"],
        })
        for codes, should_execute in ((("we", "wj"), False), (("wj", "we"), True)):
            with self.subTest(codes=codes):
                self.assertTrue(self.store.set(self.author, copy.deepcopy(state)))
                record = self.store.get_record(self.author)
                prompt = "午时 wj，务实 we。回复「加入」写入草稿，或「加入并提交」。"
                self.assertTrue(self.store.bind_origin_prompt_digest(
                    self.author, self.chat._prompt_capability_digest(prompt),
                ))
                self.history = [{"role": "assistant", "content": prompt}]
                original = copy.deepcopy(record)
                self.ctx = self.context("午时和务实就按这个组合吧")
                self.set_model("select_candidate", ("午时", "务实"), codes, refers_to_pending=True)
                execute = AsyncMock(return_value="offline multiword addition")
                with patch.object(self.commands, "_execute_confirmed_tool", execute):
                    self.assertEqual(await self.chat._stage_understand_operation_early(self.ctx), should_execute)
                if should_execute:
                    execute.assert_awaited_once()
                    self.assertEqual(
                        [(item["word"], item["code"]) for item in execute.await_args.args[0].args["items"]],
                        [("午时", "wj"), ("务实", "we")],
                    )
                    self.assertIsNone(self.store.get_record(self.author))
                else:
                    execute.assert_not_awaited()
                    self.assert_untouched(record, original)

    async def test_confirm_matches_typing_advertised_assent(self):
        self.ticket()
        literal = await self.commands.handle_pending_message_core(
            "加入", "qq", self.author.actor_id, self.author,
            history=self.history, space_key=self.author.space_key,
            owner_label="Offline owner", allow_intent_model=False,
        )
        expected = self.add.await_args
        self.add.reset_mock()
        self.ticket()
        self.assertTrue(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertEqual(self.ctx.response, literal)
        self.assertEqual(self.add.await_args, expected)
        self.add.assert_awaited_once()
        self.submit.assert_not_awaited()
        self.assertIsNone(self.store.get_record(self.author))

    async def test_submit_after_only_uses_add_and_submit_executor(self):
        self.ticket()
        self.ctx = self.context("就这样加入并提交吧")
        self.set_model("confirm", refers_to_pending=True, submit_after=True)
        self.assertTrue(await self.chat._stage_understand_operation_early(self.ctx))
        self.submit.assert_awaited_once()
        self.assertEqual(self.submit.await_args.args[:2], ("午时", "wj"))
        self.add.assert_not_awaited()
        self.assertIsNone(self.store.get_record(self.author))

    async def test_other_actor_and_forged_conversation_key_cannot_answer(self):
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        for key in (self.other, self.author):
            with self.subTest(key=key):
                self.ctx = self.context("就这样加吧", actor=self.other, conv_key=key)
                self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                self.assert_untouched(record, original)

    async def test_active_author_draft_operation_prevents_semantic_ticket_execution(self):
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        active = self.chat.draft_operation_coordinator.begin(self.author, "add")
        self.assertIsNotNone(active)
        execute = AsyncMock(side_effect=AssertionError("Active draft operation must retain arbitration"))
        with patch.object(self.chat, "handle_pending_message_core", execute):
            self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        execute.assert_not_awaited()
        self.assertIs(self.chat.draft_operation_coordinator.get(self.author), active)
        self.assertIsNone(self.ctx.response)
        self.assert_untouched(record, original)

    async def test_unrelated_quote_cannot_answer_author_ticket(self):
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        self.ctx.reply_reference = harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, text="另一条旧候选提示",
        )
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertTrue(self.ctx.released_pending_turn)
        self.assert_untouched(record, original)
        self.assertFalse(await self.chat._stage_execute_pending_state(self.ctx))
        self.assert_untouched(record, original)

    async def test_matching_quote_can_answer_current_ticket(self):
        self.ticket()
        self.ctx.reply_reference = harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, text="@Offline-owner " + self.history[0]["content"],
        )
        self.assertTrue(await self.chat._stage_understand_operation_early(self.ctx))
        self.add.assert_awaited_once()
        self.assertIsNone(self.store.get_record(self.author))

    async def test_confirm_requires_current_latest_bot_prompt_digest(self):
        for scenario in ("missing_history", "missing_digest", "older_prompt"):
            with self.subTest(scenario=scenario):
                record, _prompt = self.ticket()
                if scenario == "missing_history":
                    self.history = []
                elif scenario == "missing_digest":
                    self.assertTrue(self.store.bind_origin_prompt_digest(self.author, ""))
                else:
                    self.history.append({"role": "assistant", "content": "另一个更新的回复"})
                original = copy.deepcopy(record)
                self.ctx = self.context("就这样加吧")
                self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                self.assert_untouched(record, original)

    async def test_matching_old_quote_cannot_override_latest_prompt_safety_gate(self):
        record, prompt = self.ticket()
        original = copy.deepcopy(record)
        self.history.append({"role": "assistant", "content": "另一个更新的回复"})
        self.ctx.reply_reference = harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, text=prompt,
        )
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertIsNone(self.ctx.response)
        self.assert_untouched(record, original)

    async def test_cancel_consumes_without_write_or_replay(self):
        self.ticket()
        self.ctx = self.context("这轮先作罢吧")
        self.set_model("cancel", refers_to_pending=True)
        self.assertTrue(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertIn("取消", self.ctx.response)
        self.assertIsNone(self.store.get_record(self.author))
        self.ctx = self.context("这轮先作罢吧")
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.add.assert_not_awaited()
        self.submit.assert_not_awaited()

    async def test_roundtrip_mismatch_preserves_ticket_and_falls_through(self):
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        self.ctx = self.context("我就要那个 wja")
        self.set_model("select_candidate", ("午时",), ("wja",), refers_to_pending=True)
        wrong_choice = self.chat.MessageCommandIntent(
            intent="pending_choice", confidence=1.0, choice_index=1,
        )
        with patch.object(self.commands, "_structural_pending_add_word_intent", return_value=wrong_choice):
            self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertIsNone(self.ctx.response)
        self.assertFalse(self.ctx.released_pending_turn)
        self.assert_untouched(record, original)

    async def test_pending_recode_without_closed_roundtrip_uses_legacy_path(self):
        record, _prompt = self.ticket()
        original = copy.deepcopy(record)
        self.ctx = self.context("再给这个词换一个码吧")
        self.set_model("pending_recode", refers_to_pending=True)
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertIsNone(self.ctx.response)
        self.assertFalse(self.ctx.released_pending_turn)
        self.assert_untouched(record, original)

    async def test_answer_requires_direct_frame_and_no_nonframe_violations(self):
        for message, changes in (
            ("就这样加吧", {"frame": "question"}),
            ("就这样加吧", {"frame": "reported"}),
            ("就这样加吧", {"frame": "negated"}),
            ("就这样加吧", {"frame": "none"}),
            ("就这样加吧", {"frame": "invalid"}),
            ("不要加入", {}),
            ("确认吗？", {}),
            ("他说确认", {}),
            ("就这样加吧", {"operands": ["不存在"]}),
            ("就这样加吧", {"codes": ["INVALID"]}),
        ):
            with self.subTest(message=message, changes=changes):
                record, _prompt = self.ticket()
                original = copy.deepcopy(record)
                self.ctx = self.context(message)
                self.set_model("confirm", refers_to_pending=True, **changes)
                self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                self.assert_untouched(record, original)

    async def test_release_accepts_frame_backstop_but_rejects_other_violations(self):
        for message, operands, expected in (
            ("午时怎么打？", ("午时",), True),
            ("午时怎么打？", ("不存在",), False),
        ):
            with self.subTest(operands=operands):
                record, _prompt = self.ticket()
                original = copy.deepcopy(record)
                self.ctx = self.context(message)
                self.set_model("query", operands, refers_to_pending=False)
                self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                self.assertEqual(self.ctx.released_pending_turn, expected)
                self.assert_untouched(record, original)

    async def test_exact_answer_and_release_tokens_fail_closed(self):
        for flag in (None, "", "pending_answers,pending_releases", "PENDING_ANSWER,PENDING_RELEASE", "pending_answer;pending_release"):
            with self.subTest(flag=flag):
                if flag is None:
                    os.environ.pop("KEYTAO_SEMANTIC_ROUTE_OPS", None)
                else:
                    os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = flag
                record, _prompt = self.ticket()
                original = copy.deepcopy(record)
                self.ctx = self.context("就这样加吧")
                self.client.create.reset_mock()
                self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                self.client.create.assert_not_awaited()
                self.assertFalse(self.ctx.released_pending_turn)
                self.assert_untouched(record, original)

    async def test_each_exact_token_enables_only_its_own_behavior(self):
        for flag, op, refers in (("pending_release", "confirm", True), ("pending_answer", "query", False)):
            with self.subTest(flag=flag):
                os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = flag
                record, _prompt = self.ticket()
                original = copy.deepcopy(record)
                self.ctx = self.context("就这样加吧" if refers else "这个词怎么打")
                self.set_model(op, refers_to_pending=refers, frame="direct" if refers else "none")
                self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                self.assertFalse(self.ctx.released_pending_turn)
                self.assert_untouched(record, original)

    async def test_missing_or_nonboolean_model_decision_fields_cannot_release_or_answer(self):
        for op, refers in (("query", False), ("confirm", True)):
            for field, value in (("refers_to_pending", None), ("refers_to_pending", "false"),
                                 ("refers_to_pending", 0), ("submit_after", None), ("submit_after", "false")):
                with self.subTest(op=op, field=field, value=value):
                    record, _prompt = self.ticket()
                    original = copy.deepcopy(record)
                    self.ctx = self.context("就这样加吧" if refers else "这个词怎么打")
                    raw = raw_operation(op=op, operands=[], codes=[], refers_to_pending=refers)
                    if value is None:
                        raw.pop(field)
                    else:
                        raw[field] = value
                    self.client.create.return_value.choices[0].message.content = json.dumps(raw)
                    self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                    self.assertFalse(self.ctx.released_pending_turn)
                    self.assert_untouched(record, original)

    async def test_missing_tokens_or_failed_model_preserve_literal_resolver_path(self):
        for scenario in ("tokens_absent", "timeout", "invalid_json"):
            with self.subTest(scenario=scenario):
                self.add.reset_mock()
                self.ticket()
                self.ctx = self.context("2")
                self.set_model("select_candidate", codes=("wja",), refers_to_pending=True)
                os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "pending_answer,pending_release"
                if scenario == "tokens_absent":
                    os.environ.pop("KEYTAO_SEMANTIC_ROUTE_OPS", None)
                elif scenario == "timeout":
                    self.client.create.side_effect = asyncio.TimeoutError()
                else:
                    self.client.create.return_value.choices[0].message.content = "not JSON"
                self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                self.assertFalse(self.ctx.released_pending_turn)
                self.add.assert_not_awaited()
                response = await self.commands.handle_pending_message_core(
                    "2", "qq", self.author.actor_id, self.author,
                    history=self.history, space_key=self.author.space_key,
                    owner_label="Offline owner", allow_intent_model=False,
                )
                self.assertIn("offline addition", response)
                self.add.assert_awaited_once()
                self.assertEqual(self.add.await_args.args[:2], ("午时", "wja"))
                self.assertIsNone(self.store.get_record(self.author))


if __name__ == "__main__":
    unittest.main()
