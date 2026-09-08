"""S58 proposal-only executor and actor-owned confirmation regressions."""
import copy
import json
import unittest
from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness.tools import ToolContext, ToolExecutor
from keytao_bot.harness.state import (
    MemoryConversationStateStore, PendingToolConfirm,
    server_warning_pending_state, server_warning_ticket_is_complete,
)


def preview():
    return {
        "success": False, "requiresConfirmation": True,
        "confirmationKind": "shiftPlan", "batchId": "", "contentVersion": 0,
        "planDigest": "a" * 64, "warningDigest": "b" * 64,
        "shiftPlan": {
            "word": "奈飞", "targetCode": "nhfw",
            "items": [
                {"action": "Delete", "word": "奈飞", "code": "nhfwv", "type": "Phrase"},
                {"action": "Create", "word": "奈飞", "code": "nhfw", "type": "Phrase"},
                {"action": "Delete", "word": "浓厚氛围", "code": "nhfw", "type": "Phrase"},
                {"action": "Create", "word": "浓厚氛围", "code": "nhfwa", "type": "Phrase"},
            ],
            "shifted": [{"word": "浓厚氛围", "fromCode": "nhfw", "toCode": "nhfwa"}],
        },
    }


class BridgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.calls = []
        self.proposal = preview()
        self.entries = [{"word": "奈飞", "code": "nhfwv", "type": "Phrase", "weight": 100}]
        self.context = ToolContext(platform="qq", user_id="s58-bridge", current_message="把 奈飞 迁到 nhfw", writes_allowed=False)

        async def dispatch(name, **kwargs):
            self.calls.append((name, kwargs))
            if name == "keytao_lookup_by_words_batch":
                return {"success": True, "results": [{"word": "奈飞", "phrases": copy.deepcopy(self.entries)}]}
            self.assertEqual(name, "keytao_shift_phrase_code")
            self.assertNotIn("confirmed_plan_digest", kwargs)
            self.assertNotIn("expected_warning_digest", kwargs)
            return copy.deepcopy(self.proposal)

        self.executor = ToolExecutor(
            lambda name: lambda **kwargs: dispatch(name, **kwargs),
            frozenset({"keytao_lookup_by_words_batch", "keytao_shift_phrase_code"}),
        )

    async def call(self, args=None, context=None):
        return json.loads(await self.executor.call(
            "keytao_shift_phrase_code", args or {"word": "奈飞", "target_code": "nhfw"}, context or self.context,
        ))

    async def test_real_assent_stage_preserves_stale_evidence_without_inventing_expiry(self):
        chat = harness.openai_chat_module
        from keytao_bot.utils.offered_options import render_missing_option_ticket

        key = harness.ConversationAddress.private("qq", "s58-expired")
        now = [100.0]
        store = MemoryConversationStateStore(pending_ttl_seconds=1, clock=lambda: now[0])
        state = server_warning_pending_state(PendingToolConfirm(
            "keytao_shift_phrase_code", {"word": "奈飞", "target_code": "nhfw"},
        ), preview())
        store.set(key, state)
        self.assertEqual(store.get_record(key).expires_at, 101.0)
        proposal = chat._chat_render._format_server_bound_confirmation_prompt(state)
        now[0] = 102.0
        self.assertIsNone(store.get_record(key))
        cases = (
            ([{"role": "assistant", "content": proposal}], True, "奈飞"),
            ([{"role": "assistant", "content": "此前操作曾等待确认，现在是否继续？如需执行请回复「确认」。"}], True, "上一条提议"),
            ([], False, "当前没有待执行的操作"),
            ([{"role": "assistant", "content": render_missing_option_ticket([])}], False, "当前没有待执行的操作"),
        )
        for history, stale, expected in cases:
            with self.subTest(stale=stale, history=history):
                ctx = chat.TurnContext(None, None, "qq", "s58-expired")
                ctx.conv_key = key
                ctx.normalized_message_text = "确认"
                ctx.memory_context = harness.ChatMemoryContext(
                    platform="qq", user_id="s58-expired", space_type="private", space_id="s58-expired",
                )
                with patch.object(chat, "conversation_state_store", store), patch.object(
                    chat, "draft_operation_coordinator", harness.DraftOperationCoordinator(),
                ), patch.object(chat, "get_history", return_value=history), patch.object(
                    chat, "remember_conversation",
                ), patch.object(chat, "_finish_ai_chat_response", AsyncMock()) as finish, patch.object(
                    chat, "get_ai_response_core", AsyncMock(side_effect=AssertionError("assent used model")),
                ) as model, patch.object(chat, "call_tool_function", AsyncMock(side_effect=AssertionError("assent wrote"))) as tool:
                    self.assertTrue(await chat._stage_claim_offered_answer(ctx))
                    self.assertEqual(chat._prepare_user_facing_reply(ctx.response, ctx.memory_context), ctx.response)
                    self.assertEqual(finish.await_count, 1)
                    model.assert_not_awaited()
                    tool.assert_not_awaited()
                self.assertEqual(len(ctx.response.splitlines()), 1)
                self.assertIn(expected, ctx.response)
                self.assertEqual("已过期或记录已不存在" in ctx.response, stale)
                self.assertIn("未写入", ctx.response)
                self.assertIsNone(store.get_record(key))
                self.assertEqual(render_missing_option_ticket([{"role": "assistant", "content": ctx.response}]), ctx.response)

    async def test_early_assent_stage_defers_live_referenced_and_scoped_context(self):
        chat = harness.openai_chat_module
        key = harness.ConversationAddress.group("qq", "s58-scope", "owner")
        state = server_warning_pending_state(PendingToolConfirm(
            "keytao_shift_phrase_code", {"word": "奈飞", "target_code": "nhfw"},
        ), preview())
        for kind in ("live", "reply", "active", "other_owner"):
            with self.subTest(kind=kind):
                store = MemoryConversationStateStore()
                coordinator = harness.DraftOperationCoordinator()
                ctx = chat.TurnContext(None, None, "qq", "owner")
                ctx.conv_key, ctx.space_key = key, key.space_key
                ctx.memory_context = SimpleNamespace(space_type="group")
                ctx.normalized_message_text = "确认"
                if kind == "live":
                    store.set(key, state, space_key=key.space_key)
                elif kind == "reply":
                    ctx.reply_reference = chat.ReplyReferenceInfo(is_reply=True, is_to_bot=True, text="quoted proposal")
                elif kind == "active":
                    self.assertIsNotNone(coordinator.begin(key, "add"))
                else:
                    foreign = harness.ConversationAddress.group("qq", "s58-scope", "other")
                    store.set(foreign, state, space_key=key.space_key)
                with patch.object(chat, "conversation_state_store", store), patch.object(
                    chat, "draft_operation_coordinator", coordinator,
                ), patch.object(chat, "get_history", return_value=[]), patch.object(
                    chat, "_finish_ai_chat_response", AsyncMock(),
                ) as finish:
                    self.assertFalse(await chat._stage_claim_offered_answer(ctx))
                    finish.assert_not_awaited()

    async def test_unknown_verb_builds_only_a_complete_server_proposal(self):
        result = await self.call()
        self.assertTrue(result.get("grammar_gap_bridged"), result)
        self.assertEqual([name for name, _ in self.calls], ["keytao_lookup_by_words_batch", "keytao_shift_phrase_code"])
        state = server_warning_pending_state(PendingToolConfirm(
            function_name="keytao_shift_phrase_code", args=result["grammarGapArguments"],
        ), result)
        self.assertTrue(server_warning_ticket_is_complete(state))
        self.assertEqual(state.args["target_type"], "Phrase")
        self.assertFalse(result["success"])

    async def test_unverifiable_arguments_never_reach_the_planner(self):
        for args in (
            {"word": "外部词", "target_code": "nhfw"},
            {"word": "奈飞", "target_code": "nhfwa"},
            {"word": "奈飞", "target_code": "nhfw", "confirmed_plan_digest": "a" * 64},
            {"word": "奈飞", "target_code": "nhfw", "additional_items": []},
            {"word": "奈飞", "target_code": "NHFW"},
        ):
            with self.subTest(args=args):
                self.calls.clear()
                self.assertFalse((await self.call(args)).get("grammar_gap_bridged"))
                self.assertEqual(self.calls, [])
        self.entries = []
        result = await self.call()
        self.assertFalse(result.get("grammar_gap_bridged"))
        self.assertEqual([name for name, _ in self.calls], ["keytao_lookup_by_words_batch"])

    async def test_questions_negation_reported_text_and_attachments_cannot_bridge(self):
        for message in ("把 奈飞 迁到 nhfw？", "不要把 奈飞 迁到 nhfw", "他说把 奈飞 迁到 nhfw", "例句：把 奈飞 迁到 nhfw", "把 奈飞 迁到 nhfw，但保持浓厚氛围"):
            with self.subTest(message=message):
                self.assertFalse((await self.call(context=replace(self.context, current_message=message))).get("grammar_gap_bridged"))
                self.assertEqual(self.calls, [])
        self.assertFalse((await self.call(context=replace(self.context, attachment_context=True))).get("grammar_gap_bridged"))
        self.assertEqual(self.calls, [])

    async def test_missing_seal_and_failed_plan_do_not_bridge(self):
        for key in ("planDigest", "warningDigest", "shiftPlan"):
            with self.subTest(key=key):
                self.proposal = preview()
                self.proposal.pop(key)
                self.assertFalse((await self.call()).get("grammar_gap_bridged"))
        self.proposal = {"success": False, "message": "invalid code"}
        self.assertFalse((await self.call()).get("grammar_gap_bridged"))

    async def test_real_orchestrator_saves_before_render_and_stops_the_model_loop(self):
        from keytao_bot.harness.orchestrator import AgentOrchestrator, AgentRequestContext, AgentRuntimeConfig

        store = MemoryConversationStateStore()
        client = harness._FakeClient([harness._FakeAIResponse(
            "tool_calls", content="浓厚氛围会去错误的编码 nhfwz",
            tool_calls=[SimpleNamespace(id="s58-move", type="function", function=SimpleNamespace(
                name="keytao_shift_phrase_code", arguments=json.dumps({"word": "奈飞", "target_code": "nhfw"}),
            ))],
        )])
        class ShiftSkills(harness._FakeToolSkillsManager):
            def get_tools(self):
                return [{"type": "function", "function": {
                    "name": "keytao_shift_phrase_code", "description": "Preview a shift",
                    "parameters": {"type": "object", "properties": {
                        "word": {"type": "string"}, "target_code": {"type": "string"},
                    }, "required": ["word", "target_code"]},
                }}]
        orchestrator = AgentOrchestrator(
            client_factory=lambda: client,
            runtime=AgentRuntimeConfig("fake-model", 1000, 0.0, 10.0),
            skills_manager=ShiftSkills(), tool_executor=self.executor,
            state_store=store, bind_help_text="bind help", system_prompt_core="system",
        )
        context = AgentRequestContext(platform="qq", user_id="s58-bridge", mutations_allowed=False)
        reply = await orchestrator.run(self.context.current_message, context)
        record = store.get_record(context.conversation_address)
        self.assertIsNotNone(record, reply)
        self.assertTrue(server_warning_ticket_is_complete(record.state))
        self.assertFalse(record.execution_id)
        self.assertEqual(len(client.completions.calls), 1)
        self.assertNotIn("nhfwz", reply)
        for value in ("奈飞", "浓厚氛围", "nhfwv", "nhfw", "nhfwa", "确认"):
            self.assertIn(value, reply)
        self.assertEqual(len(self.calls), 2)
        foreign = replace(context, user_id="s58-foreign")
        self.assertIsNone(store.get_record(foreign.conversation_address))

    def test_lost_ticket_copy_names_displayed_plan_without_resurrecting_it(self):
        from keytao_bot.utils.offered_options import render_missing_option_ticket
        reply = render_missing_option_ticket([{
            "role": "assistant", "content": "🔁 调整计划：\n• 奈飞：nhfwv→nhfw\n• 浓厚氛围：nhfw→nhfwa\n回复「确认」执行。",
        }])
        self.assertEqual(len(reply.splitlines()), 1)
        self.assertIn("奈飞", reply)
        self.assertIn("nhfw", reply)
        self.assertIn("没有可执行", reply)

    def test_recordless_assents_do_not_turn_previous_refusals_into_proposals(self):
        from keytao_bot.utils.offered_options import render_missing_option_ticket
        reply = render_missing_option_ticket([])
        self.assertNotIn("上一条提议", reply)
        for _ in range(5):
            repeated = render_missing_option_ticket([{"role": "assistant", "content": reply}])
            self.assertEqual(repeated, reply)
        self.assertEqual(render_missing_option_ticket([
            {"role": "assistant", "content": "你好。"},
        ]), reply)
        prior = render_missing_option_ticket([{
            "role": "assistant", "content": "🔁 调整计划：\n• 奈飞：nhfwv→nhfw\n回复「确认」执行。",
        }])
        self.assertEqual(render_missing_option_ticket([
            {"role": "assistant", "content": prior},
        ]), prior)


if __name__ == "__main__":
    unittest.main()
