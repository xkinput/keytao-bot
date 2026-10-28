"""Offline model-first routing and author-owned understood-command checks."""

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import test_state_machine as harness
from test_semantic_parser import FakeClient, raw_operation

from keytao_bot.harness.conversation import ConversationAddress
from keytao_bot.harness.state import MemoryConversationStateStore, PendingToolConfirm


class SemanticModelFirstTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.store = MemoryConversationStateStore()
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id="model-first-owner", space_type="group",
            space_id="model-first-group", speaker_name="Offline owner",
        )
        self.author = self.memory.conversation_address
        self.other = ConversationAddress.group("qq", "model-first-group", "other-user")
        self.client = FakeClient()
        self.finish = AsyncMock()
        self.tool = AsyncMock(side_effect=AssertionError("Unexpected offline tool call"))
        self.info = MagicMock()
        patches = (
            patch.dict(os.environ, {
                "KEYTAO_SEMANTIC_ROUTE_OPS": "draft_submit",
                "KEYTAO_SEMANTIC_SHADOW": "0",
            }),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "call_tool_function", self.tool),
            patch.object(self.chat, "_finish_ai_chat_response", self.finish),
            patch.object(self.chat, "remember_conversation"),
            patch.object(self.chat.logger, "info", self.info),
            patch.object(self.chat.logger, "warning"),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)
        self.ctx = self.context("直接提交现在草稿，不修改")
        self.set_model("draft_submit")

    async def asyncTearDown(self):
        tasks = tuple(self.chat._semantic_shadow_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.chat._semantic_shadow_tasks.clear()

    def context(self, message):
        return self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.author.actor_id,
            normalized_message_text=message, conv_key=self.author,
            space_key=self.author.space_key, memory_context=self.memory,
            owner_label="Offline owner", response="unchanged response",
            scoped_pending_response="old ticket response",
        )

    def set_model(self, op, operands=(), codes=(), **changes):
        self.client.create.return_value.choices[0].message.content = json.dumps(
            raw_operation(op=op, operands=list(operands), codes=list(codes), **changes),
            ensure_ascii=False,
        )

    @staticmethod
    def command_families():
        return (
            ("delete", "把午时 wj 删掉", ("午时",), ("wj",)),
            ("recode", "午时的 wj 改为 wja", ("午时",), ("wj", "wja")),
            ("move", "将午时调整到 wj", ("午时",), ("wj",)),
            ("reorder", "午时需要优先于务实", ("午时", "务实"), ()),
            ("batch_replace", "把午替换成晚，源条目午时 wj", ("午", "晚", "午时"), ("wj",)),
            ("draft_submit", "直接提交现在草稿，不修改", (), ()),
            ("draft_recall", "把现在草稿撤回", (), ()),
            ("draft_clear", "清掉当前草稿", (), ()),
            ("draft_keep_only", "当前草稿只留下午时与务实", ("午时", "务实"), ()),
        )

    def delivery_results(self, response, memory=None):
        memory = memory or self.memory
        tokens = (
            (self.chat._current_turn_message, self.chat._current_turn_message.set(self.ctx.normalized_message_text)),
            (self.chat._current_conversational_turn, self.chat._current_conversational_turn.set(None)),
            (self.chat.current_draft_delivery_claims, self.chat.current_draft_delivery_claims.set([])),
        )
        try:
            with (
                patch.object(self.chat._user_resolver, "resolved_binding_for_notice", return_value=True),
                patch.object(self.chat, "get_history", return_value=[]),
            ):
                return (
                    self.chat._enforce_advertised_reply_contract(response, memory.conversation_address),
                    self.chat._prepare_user_facing_reply(response, memory),
                )
        finally:
            for variable, token in reversed(tokens):
                variable.reset(token)

    async def answer(self, message, *, actor=None, conv_key=None):
        actor = actor or self.author
        return await self.commands.handle_pending_message_core(
            message, "qq", actor.actor_id, conv_key or actor,
            space_key=actor.space_key, allow_intent_model=False,
        )

    async def stage_submit(self):
        self.assertTrue(await self.chat._stage_understand_operation_early(self.ctx))
        record = self.store.get_record(self.author)
        self.assertIsNotNone(record, self.ctx.response)
        self.assertEqual(record.owner_key, self.author)
        self.assertEqual(record.owner_label, "Offline owner")
        self.assertIsInstance(record.state, PendingToolConfirm)
        self.assertEqual(record.state.function_name, "keytao_understood_command")
        self.assertEqual(record.state.args["op"], "draft_submit")
        self.assertEqual(record.state.args["command_text"], "提交草稿")
        self.assertIn("我理解为", self.ctx.response)
        self.assertIn("提交草稿", self.ctx.response)
        self.assertIn("确认", self.ctx.response)
        self.assertIn("取消", self.ctx.response)
        self.assertIsNone(self.ctx.scoped_pending_response)
        self.tool.assert_not_awaited()
        return record

    async def stage_command_family(self, op):
        if op == "delete":
            lookup = AsyncMock(return_value=json.dumps({"success": True, "phrases": [
                {"word": "午时", "code": "wj", "type": "Phrase"},
            ]}))
            with patch.object(self.commands, "call_tool_function", lookup):
                handled = await self.chat._stage_understand_operation(self.ctx)
            lookup.assert_awaited_once_with(
                "keytao_lookup_by_word", {"word": "午时"}, "qq", self.author.actor_id,
            )
            return handled
        if op != "move":
            return await self.chat._stage_understand_operation(self.ctx)

        async def lookup(name, args, *_args, **_kwargs):
            if name == "keytao_lookup_by_words_batch":
                return json.dumps({"success": True, "results": [{"word": "午时", "phrases": [
                    {"word": "午时", "code": "wja", "type": "Phrase"},
                ]}]})
            self.assertEqual((name, args), ("keytao_lookup_by_code", {"code": "wj"}))
            return json.dumps({"success": True, "phrases": [
                {"word": "务实", "code": "wj", "type": "Phrase"},
            ]})

        with patch.object(self.commands, "call_tool_function", side_effect=lookup) as reads:
            handled = await self.chat._stage_understand_operation(self.ctx)
        self.assertEqual([call.args[0] for call in reads.await_args_list], [
            "keytao_lookup_by_words_batch", "keytao_lookup_by_code",
        ])
        return handled

    def test_understanding_precedes_every_command_and_pending_stage(self):
        stages = [stage.__name__ for stage in self.chat.STAGES]
        initialize = stages.index("_stage_initialize_conversation")
        understand = stages.index("_stage_understand_operation_early")
        control = stages.index("_stage_handle_understood_command_control")
        self.assertEqual(stages[initialize + 1], "_stage_handle_discovery_reply")
        self.assertEqual(understand, initialize + 2)
        self.assertEqual(control, understand + 1)
        self.assertEqual(stages.count("_stage_understand_operation_early"), 1)
        self.assertEqual(stages.count("_stage_understand_operation"), 1)
        self.assertEqual(
            stages.index("_stage_understand_operation"),
            stages.index("_stage_resolve_current_pending_scope") + 1,
        )
        for name in (
            "_stage_handle_commonness_query", "_stage_handle_explicit_entry_operation",
            "_stage_claim_offered_answer", "_stage_resolve_current_pending_scope",
            "_stage_submit_current_draft", "_stage_handle_draft_management",
        ):
            self.assertLess(understand, stages.index(name), name)

    async def test_model_submit_author_confirm_dispatches_once_after_consumption(self):
        await self.stage_submit()

        async def submit(name, args, platform, user_id, **kwargs):
            self.assertEqual((name, platform, user_id), (
                "keytao_submit_batch", "qq", self.author.actor_id,
            ))
            self.assertIsNone(self.store.get_record(self.author))
            return json.dumps({"success": True, "batchId": "offline-batch"})

        self.tool.side_effect = submit
        response = await self.answer("确认")
        self.assertIn("提交", response)
        self.tool.assert_awaited_once()
        self.assertIsNone(await self.answer("确认"))
        self.tool.assert_awaited_once()
        self.client.create.assert_awaited_once()

    async def test_other_actor_unrelated_and_nonexact_confirm_do_not_execute(self):
        record = await self.stage_submit()
        for message, actor, key in (
            ("确认", self.other, self.other),
            ("确认", self.other, self.author),
            ("今天天气不错", self.author, self.author),
            ("好的确认吧", self.author, self.author),
        ):
            with self.subTest(message=message, actor=actor.actor_id, key=key):
                await self.answer(message, actor=actor, conv_key=key)
                self.assertIs(self.store.get_record(self.author), record)
                self.tool.assert_not_awaited()

    async def test_cancel_consumes_without_submit_or_replay(self):
        await self.stage_submit()
        self.assertIn("取消", await self.answer("取消"))
        self.assertIsNone(self.store.get_record(self.author))
        self.assertIsNone(await self.answer("确认"))
        self.tool.assert_not_awaited()

    async def test_new_command_replaces_author_previous_live_ticket(self):
        choice = self.commands._build_pending_choice_offer((
            ("A", "清理冲突后继续", None), ("B", "保留现有草稿", None),
        ))
        self.assertIsNotNone(choice)
        self.assertTrue(self.store.set(self.author, choice))
        previous = self.store.get_record(self.author)
        await self.stage_submit()
        self.assertIsNot(self.store.get_record(self.author), previous)
        self.tool.assert_not_awaited()

    async def test_all_non_swap_families_only_seal_before_confirmation(self):
        for op, message, operands, codes in self.command_families():
            with self.subTest(op=op):
                self.tool.reset_mock()
                os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = op
                self.store.delete(self.author)
                self.ctx = self.context(message)
                self.set_model(op, operands, codes)
                self.assertTrue(await self.stage_command_family(op))
                record = self.store.get_record(self.author)
                self.assertIsNotNone(record, self.ctx.response)
                self.assertEqual(record.state.function_name, "keytao_understood_command")
                self.assertEqual(record.state.args["op"], op)
                self.assertIn("确认", self.ctx.response)
                self.tool.assert_not_awaited()

    async def test_all_understood_families_survive_real_delivery_boundaries(self):
        for op, message, operands, codes in self.command_families():
            with self.subTest(op=op):
                self.tool.reset_mock()
                os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = op
                self.store.delete(self.author)
                self.ctx = self.context(message)
                self.set_model(op, operands, codes)
                self.assertTrue(await self.stage_command_family(op))
                record = self.store.get_record(self.author)
                self.assertTrue(self.chat._advertised_reply_matches_live_record(self.ctx.response, record))
                self.assertEqual(self.chat._render_live_batch_record(record), self.ctx.response)
                for delivered in self.delivery_results(self.ctx.response):
                    self.assertEqual(delivered, self.ctx.response)
                    self.assertIn("回复「确认」执行，或「取消」。", delivered)
                forged = "我理解为：「完全不同的操作」\n回复「确认」执行，或「取消」。"
                self.assertFalse(self.chat._advertised_reply_matches_live_record(forged, record))
                for delivered in self.delivery_results(forged):
                    self.assertNotEqual(delivered, forged)
                    self.assertNotIn("完全不同的操作", delivered)
                self.tool.assert_not_awaited()

    async def test_unbacked_understood_confirmation_fails_real_delivery_guards(self):
        other_memory = harness.ChatMemoryContext(
            platform="qq", user_id=self.other.actor_id, space_type="group",
            space_id="model-first-group", speaker_name="Other actor",
        )
        for op, message, operands, codes in self.command_families():
            with self.subTest(op=op):
                self.tool.reset_mock()
                os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = op
                self.store.delete(self.author)
                self.ctx = self.context(message)
                self.set_model(op, operands, codes)
                self.assertTrue(await self.stage_command_family(op))
                record = self.store.get_record(self.author)
                original = self.ctx.response

                def assert_blocked(results):
                    for delivered in results:
                        self.assertNotEqual(delivered, original)
                        self.assertNotIn("回复「确认」执行，或「取消」。", delivered)

                with self.subTest(state="other_actor"):
                    assert_blocked(self.delivery_results(original, other_memory))
                self.assertIs(self.store.get_record(self.author), record)
                self.assertTrue(self.store.begin_execution(record))
                with self.subTest(state="executing"):
                    assert_blocked(self.delivery_results(original))
                self.assertTrue(self.store.abort_execution(record))
                record.state.args["digest"] = "tampered"
                self.assertFalse(self.chat._advertised_reply_matches_live_record(original, record))
                self.assertEqual(self.chat._render_live_batch_record(record), "")
                with self.subTest(state="tampered"):
                    assert_blocked(self.delivery_results(original))
                self.assertTrue(self.store.complete_execution(record))
                with self.subTest(state="consumed"):
                    assert_blocked(self.delivery_results(original))
                self.tool.assert_not_awaited()

    async def test_delete_and_recode_confirm_use_real_typed_dispatch(self):
        for op, codes, expected_items in (
            ("delete", ("wj",), [
                {"action": "Delete", "word": "午时", "code": "wj", "type": "Phrase"},
            ]),
            ("recode", ("wj", "wja"), [
                {"action": "Delete", "word": "午时", "code": "wj", "type": "Phrase"},
                {"action": "Create", "word": "午时", "code": "wja", "type": "Phrase"},
            ]),
        ):
            with self.subTest(op=op):
                os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = op
                self.ctx = self.context("调整午时的 wj wja")
                self.set_model(op, ("午时",), codes)
                self.tool.reset_mock()
                self.assertTrue(await self.stage_command_family(op))
                self.tool.assert_not_awaited()

                async def call(name, args, platform, user_id, **kwargs):
                    self.assertIsNone(self.store.get_record(self.author))
                    if name == "keytao_lookup_by_word":
                        self.assertEqual(args, {"word": "午时"})
                        return json.dumps({"success": True, "phrases": [
                            {"word": "午时", "code": "wj", "type": "Phrase"},
                        ]})
                    self.assertEqual(name, "keytao_batch_add_to_draft")
                    self.assertEqual(args["items"], expected_items)
                    self.assertTrue(args["preview_only"])
                    return json.dumps({
                        "success": True, "successCount": len(expected_items),
                        "failedCount": 0, "batchId": "offline-batch",
                    })

                self.tool.side_effect = call
                with patch.object(self.commands, "_format_draft_response", AsyncMock(return_value="offline receipt")):
                    response = await self.answer("确认")
                self.assertIn("已加入草稿", response)
                self.assertEqual([call.args[0] for call in self.tool.await_args_list], [
                    "keytao_lookup_by_word", "keytao_batch_add_to_draft",
                ])
                self.assertIsNone(self.store.get_record(self.author))

    async def test_batch_replace_preserves_normal_handler_confirmation(self):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "batch_replace"
        self.ctx = self.context("午替换成晚，午时 wj")
        self.set_model("batch_replace", ("午", "晚", "午时"), ("wj",))
        self.assertTrue(await self.chat._stage_understand_operation(self.ctx))
        understood = self.store.get_record(self.author)
        self.tool.assert_not_awaited()
        response = await self.answer("确认")
        normal = self.store.get_record(self.author)
        self.assertIsNot(normal, understood)
        self.assertEqual(normal.state.function_name, "keytao_batch_add_to_draft")
        self.assertEqual(normal.state.args["items"], [
            {"action": "Change", "old_word": "午时", "word": "晚时", "code": "wj"},
        ])
        self.assertIn("确认", response)
        self.tool.assert_not_awaited()
        await self.answer("确认", actor=self.other)
        self.assertIs(self.store.get_record(self.author), normal)
        self.tool.assert_not_awaited()

    async def test_early_control_stage_executes_author_confirm_before_pending_claims(self):
        await self.stage_submit()
        self.ctx = self.context("确认")
        self.set_model("confirm", refers_to_pending=True)
        self.client.create.reset_mock()
        self.finish.reset_mock()
        self.tool.side_effect = None
        self.tool.return_value = json.dumps({"success": True, "batchId": "offline-batch"})
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertTrue(await self.chat._stage_handle_understood_command_control(self.ctx))
        self.tool.assert_awaited_once()
        self.assertEqual(self.tool.await_args.args[0], "keytao_submit_batch")
        self.assertIn("提交", self.ctx.response)
        self.assertIsNone(self.store.get_record(self.author))
        self.client.create.assert_awaited_once()
        self.finish.assert_awaited_once()

    async def test_quoted_unrelated_control_preserves_understood_ticket(self):
        references = (
            harness.ReplyReferenceInfo(
                is_reply=True, is_to_bot=True, mentioned_user_ids=(self.other.actor_id,),
                text="是否添加「午时」？回复「确认」或「取消」。",
            ),
            harness.ReplyReferenceInfo(
                is_reply=True, is_to_bot=True, mentioned_user_ids=(self.author.actor_id,),
                text="我理解为：「清空草稿」\n回复「确认」执行，或「取消」。",
            ),
            harness.ReplyReferenceInfo(
                is_reply=True, is_to_bot=False,
                text="我理解为：「提交草稿」\n回复「确认」执行，或「取消」。",
            ),
            harness.ReplyReferenceInfo(is_reply=True, is_to_bot=True),
        )
        for stage in (
            self.chat._stage_handle_understood_command_control,
            self.chat._stage_execute_pending_state,
        ):
            for control in ("确认", "取消"):
                for reference in references:
                    with self.subTest(stage=stage.__name__, control=control, reference=reference):
                        self.tool.reset_mock()
                        self.tool.side_effect = None
                        self.tool.return_value = json.dumps({"success": True, "batchId": "offline-batch"})
                        self.ctx = self.context("直接提交现在草稿，不修改")
                        self.set_model("draft_submit")
                        record = await self.stage_submit()
                        self.finish.reset_mock()
                        self.ctx = self.context(control)
                        self.ctx.response = None
                        self.ctx.reply_reference = reference
                        self.assertFalse(await stage(self.ctx))
                        self.assertIs(self.store.get_record(self.author), record)
                        self.assertFalse(record.execution_id)
                        self.assertIsNone(self.ctx.response)
                        self.tool.assert_not_awaited()
                        self.finish.assert_not_awaited()
                        self.assertTrue(self.ctx.released_pending_turn)

    async def test_quoted_unrelated_control_cannot_consume_via_fallback(self):
        async def fallback_only(orchestrator, *, message, context, **kwargs):
            return await orchestrator._deterministic_fallback_handler(message, context) or "offline fallback"

        for control in ("确认", "取消"):
            with self.subTest(control=control):
                self.tool.reset_mock()
                self.tool.side_effect = None
                self.tool.return_value = json.dumps({"success": True, "batchId": "offline-batch"})
                self.ctx = self.context("直接提交现在草稿，不修改")
                self.set_model("draft_submit")
                record = await self.stage_submit()
                self.ctx = self.context(control)
                self.ctx.response = None
                self.ctx.reply_reference = harness.ReplyReferenceInfo(
                    is_reply=True, is_to_bot=True, text="旧提示",
                )
                self.assertFalse(await self.chat._stage_handle_understood_command_control(self.ctx))
                self.assertFalse(await self.chat._stage_execute_pending_state(self.ctx))
                with (
                    patch.object(self.chat, "OPENAI_API_KEY", "offline-key"),
                    patch.object(self.chat.AgentOrchestrator, "run", fallback_only),
                    patch.object(self.chat, "get_history", return_value=[]),
                    patch.object(self.chat.memory_store, "get_context_block", return_value=""),
                    patch.object(self.chat, "get_group_history_context", return_value=""),
                    patch.object(self.chat, "build_reply_context", AsyncMock(return_value="旧提示")),
                ):
                    self.assertFalse(await self.chat._stage_generate_ai_response(self.ctx))
                self.assertEqual(self.ctx.response, "offline fallback")
                self.assertIs(self.store.get_record(self.author), record)
                self.tool.assert_not_awaited()

    async def test_quoted_noncontrol_does_not_release_understood_ticket(self):
        record = await self.stage_submit()
        for stage in (
            self.chat._stage_handle_understood_command_control,
            self.chat._stage_execute_pending_state,
        ):
            with self.subTest(stage=stage.__name__):
                self.ctx = self.context("查看草稿")
                self.ctx.response = None
                self.ctx.reply_reference = harness.ReplyReferenceInfo(
                    is_reply=True, is_to_bot=True, text="旧提示",
                )
                self.assertFalse(await stage(self.ctx))
                self.assertFalse(self.ctx.released_pending_turn)
                self.assertIs(self.store.get_record(self.author), record)
                self.tool.assert_not_awaited()

    async def test_quoted_own_understood_reply_executes_exactly_once(self):
        await self.stage_submit()
        prompt = self.ctx.response
        self.tool.side_effect = None
        self.tool.return_value = json.dumps({"success": True, "batchId": "offline-batch"})
        self.finish.reset_mock()
        self.ctx = self.context("确认")
        self.ctx.reply_reference = harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, text="@Offline-owner " + prompt,
        )
        self.assertTrue(await self.chat._stage_handle_understood_command_control(self.ctx))
        self.tool.assert_awaited_once()
        self.assertEqual(self.tool.await_args.args[0], "keytao_submit_batch")
        self.assertIsNone(self.store.get_record(self.author))
        self.assertFalse(await self.chat._stage_handle_understood_command_control(self.ctx))
        self.tool.assert_awaited_once()
        self.finish.assert_awaited_once()

    async def test_early_fallback_never_reparses_in_legacy_slot_with_shadow(self):
        for shadow in ("0", "1"):
            for scenario in ("timeout", "invalid_json", "nonactionable"):
                with self.subTest(shadow=shadow, scenario=scenario):
                    os.environ["KEYTAO_SEMANTIC_SHADOW"] = shadow
                    self.ctx = self.context("直接提交现在草稿，不修改")
                    self.client.create.reset_mock()
                    self.client.create.side_effect = asyncio.TimeoutError() if scenario == "timeout" else None
                    self.set_model("draft_submit", frame="negated")
                    if scenario == "invalid_json":
                        self.client.create.return_value.choices[0].message.content = "not JSON"
                    self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
                    self.assertTrue(self.ctx.operation_understood_early)
                    self.client.create.assert_awaited_once()
                    self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
                    self.client.create.assert_awaited_once()
                    self.assertEqual(self.chat._semantic_shadow_tasks, set())
                    self.assertIsNone(self.store.get_record(self.author))
                    self.assertEqual(self.ctx.response, "unchanged response")
                    self.finish.assert_not_awaited()
                    self.tool.assert_not_awaited()

    async def test_shadow_only_keeps_background_parse_in_legacy_slot(self):
        os.environ.pop("KEYTAO_SEMANTIC_ROUTE_OPS", None)
        os.environ["KEYTAO_SEMANTIC_SHADOW"] = "1"
        self.assertFalse(await self.chat._stage_understand_operation_early(self.ctx))
        self.assertFalse(self.ctx.operation_understood_early)
        self.client.create.assert_not_awaited()
        self.assertEqual(self.chat._semantic_shadow_tasks, set())
        self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
        tasks = tuple(self.chat._semantic_shadow_tasks)
        self.assertEqual(len(tasks), 1)
        await asyncio.gather(*tasks)
        self.client.create.assert_awaited_once()
        self.assertIsNone(self.store.get_record(self.author))
        self.finish.assert_not_awaited()
        self.tool.assert_not_awaited()

    async def test_flag_absent_or_without_exact_op_falls_through(self):
        for flag in (None, "", "swap", "draft_submits", "DRAFT_SUBMIT", "draft_submit;"):
            with self.subTest(flag=flag):
                if flag is None:
                    os.environ.pop("KEYTAO_SEMANTIC_ROUTE_OPS", None)
                else:
                    os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = flag
                self.ctx = self.context("直接提交现在草稿，不修改")
                self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
                self.assertIsNone(self.store.get_record(self.author))
                self.assertEqual(self.ctx.response, "unchanged response")
                self.tool.assert_not_awaited()

    async def test_negated_reported_or_unmentioned_operands_fall_through(self):
        cases = (
            ("draft_submit", "提交现在草稿", (), (), {"frame": "negated"}),
            ("draft_submit", "提交现在草稿", (), (), {"frame": "reported"}),
            ("draft_submit", "不要提交现在草稿", (), (), {}),
            ("draft_submit", "他说提交现在草稿", (), (), {}),
            ("delete", "删除午时 wj", ("不存在",), ("wj",), {}),
        )
        for op, message, operands, codes, changes in cases:
            with self.subTest(message=message, changes=changes):
                os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = op
                self.ctx = self.context(message)
                self.set_model(op, operands, codes, **changes)
                self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
                self.assertIsNone(self.store.get_record(self.author))
                self.assertEqual(self.ctx.response, "unchanged response")
                self.assertEqual(self.ctx.scoped_pending_response, "old ticket response")
                self.tool.assert_not_awaited()

    async def test_roundtrip_failure_does_not_seal_or_dispatch(self):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "delete"
        self.ctx = self.context("删除 午时\n清空草稿 wj")
        self.set_model("delete", ("午时\n清空草稿",), ("wj",))
        self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
        self.assertIsNone(self.store.get_record(self.author))
        self.tool.assert_not_awaited()

    async def test_invalid_json_and_model_timeout_keep_today_path(self):
        for error, content in (
            (asyncio.TimeoutError(), "{}"), (None, "not JSON"), (None, "[]"),
        ):
            with self.subTest(error=type(error).__name__, content=content):
                self.ctx = self.context("直接提交现在草稿，不修改")
                self.client.create.side_effect = error
                self.client.create.return_value.choices[0].message.content = content
                self.assertFalse(await self.chat._stage_understand_operation(self.ctx))
                self.assertIsNone(self.store.get_record(self.author))
                self.assertEqual(self.ctx.response, "unchanged response")
                self.tool.assert_not_awaited()

    async def test_cancelled_error_propagates_without_ticket(self):
        self.client.create.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.chat._stage_understand_operation(self.ctx)
        self.assertIsNone(self.store.get_record(self.author))
        self.tool.assert_not_awaited()

    async def test_cancelled_execution_stays_consumed_and_cannot_replay(self):
        await self.stage_submit()
        self.tool.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.answer("确认")
        self.assertIsNone(self.store.get_record(self.author))
        self.assertIsNone(await self.answer("确认"))
        self.tool.assert_awaited_once()

    async def test_awaited_understanding_and_shadow_share_one_parse(self):
        os.environ["KEYTAO_SEMANTIC_SHADOW"] = "1"
        await self.stage_submit()
        self.client.create.assert_awaited_once()
        self.assertEqual(self.chat._semantic_shadow_tasks, set())
        lines = [call.args[0] for call in self.info.call_args_list
                 if call.args and str(call.args[0]).startswith("[semantic_shadow]")]
        self.assertEqual(len(lines), 1)
        self.assertEqual(self.ctx.operation_request.op, "draft_submit")

    async def test_tampered_command_ticket_never_executes(self):
        record = await self.stage_submit()
        record.state.args["command_text"] = "清空草稿"
        await self.answer("确认")
        self.tool.assert_not_awaited()

    async def test_concurrent_author_confirmation_dispatches_only_once(self):
        await self.stage_submit()
        started, release = asyncio.Event(), asyncio.Event()

        async def submit(*args, **kwargs):
            started.set()
            await release.wait()
            return json.dumps({"success": True, "batchId": "offline-batch"})

        self.tool.side_effect = submit
        first = asyncio.create_task(self.answer("确认"))
        try:
            await asyncio.wait_for(started.wait(), 0.5)
            self.assertIsNone(await self.answer("确认"))
            self.tool.assert_awaited_once()
            release.set()
            await asyncio.wait_for(first, 0.5)
        finally:
            if not first.done():
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
