"""Offline regressions for the B5 production incidents."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
import test_community_vote as vote_fixtures
from keytao_bot.harness.state import (
    DraftOperationCoordinator, MemoryConversationStateStore, PendingToolConfirm,
    pending_duplicate_confirmation_state,
)
from keytao_bot.plugins import chat_routing
from test_semantic_b5_delete import B5DeleteTests


class B5ReplySafetyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.chat = harness.openai_chat_module
        self.store = MemoryConversationStateStore()
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id="abang", space_type="group", space_id="b5",
            speaker_name="Offline owner",
        )
        self.tool = AsyncMock(return_value=json.dumps({"success": True, "items": []}))
        self.finish = AsyncMock()
        for context in (
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "call_tool_function", self.tool),
            patch.object(self.chat, "_finish_ai_chat_response", self.finish),
            patch.object(self.chat, "remember_conversation"),
            patch.object(self.chat, "get_history", return_value=[]),
            patch.object(self.chat, "to_me", return_value=AsyncMock(
                side_effect=lambda bot, event, state: event.to_me,
            )),
        ):
            context.start()
            self.addCleanup(context.stop)
        self.token = self.chat.current_draft_delivery_claims.set([])
        self.addCleanup(self.chat.current_draft_delivery_claims.reset, self.token)

    def context(self, message="确认", *, quoted=False, addressed=True):
        ctx = self.chat.TurnContext(
            bot=SimpleNamespace(), event=SimpleNamespace(to_me=addressed),
            platform="qq", user_id="abang", message_text=message,
            normalized_message_text=message, conv_key=self.memory.conversation_address,
            memory_context=self.memory, space_key=self.memory.conversation_address.space_key,
        )
        if quoted:
            ctx.reply_reference = harness.ReplyReferenceInfo(
                is_reply=True, is_to_bot=True, text="是否继续提交？回复「确认」。",
            )
        return ctx

    async def test_prod_confirm_after_unrelated_turns_and_expiry_never_calls_model(self):
        for quoted in (None, "bot", "human"):
            with self.subTest(quoted=quoted):
                self.tool.return_value = json.dumps({
                    "success": True, "batchId": "1aa01eb8", "status": "Draft",
                    "items": [{"word": "屡有", "code": "llydav", "action": "Create"}],
                })
                ctx = self.context(quoted=quoted == "bot", addressed=quoted != "bot")
                if quoted == "human":
                    ctx.reply_reference = harness.ReplyReferenceInfo(
                        is_reply=True, is_to_bot=False, text="是否继续提交？回复「确认」。",
                    )
                if quoted == "bot":
                    self.chat.get_history.return_value = [{
                        "role": "assistant", "content": ctx.reply_reference.text,
                    }]
                else:
                    self.chat.get_history.return_value = []
                with patch.object(self.chat, "AsyncOpenAI") as model:
                    self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
                model.assert_not_called()
                self.assertIn("没有可执行的确认记录" if quoted == "bot" else "当前没有待执行的操作", ctx.response)
                self.assertIn("引用原提议", ctx.response)
                self.assertEqual(len(ctx.response.splitlines()), 1)
                self.assertNotIn("已提交", ctx.response)
                self.tool.assert_not_awaited()

    async def test_foreign_live_confirmation_quote_never_echoes_or_consumes_ticket(self):
        prompt = "拟提交「私有提议」→ syty。\n是否继续提交？回复「确认」。"
        for scenario in ("other_owner", "other_space", "foreign_mention", "unbound", "other_space_active"):
            with self.subTest(scenario=scenario):
                other_space = scenario in {"other_space", "other_space_active"}
                source = harness.ConversationAddress.group(
                    "qq", "another-group" if other_space else "b5",
                    "abang" if other_space else "another-owner",
                )
                self.assertTrue(self.store.set(
                    source, PendingToolConfirm("keytao_submit_batch", {"batch_id": "other-batch"}),
                    space_key=source.space_key,
                ))
                ticket = self.store.get_record(source)
                original = copy.deepcopy(ticket)
                active = None
                if scenario == "other_space_active":
                    active = self.chat.draft_operation_coordinator.begin(source, "submit")
                    self.assertIsNotNone(active)
                    self.assertTrue(self.chat.draft_operation_coordinator.mark_awaiting_confirmation(
                        source, active.operation_id, ticket.state, prompt,
                    ))
                    original_active = copy.deepcopy(active)
                ctx = self.context("好", addressed=False)
                ctx.reply_reference = harness.ReplyReferenceInfo(
                    is_reply=True, is_to_bot=True, text=prompt,
                    mentioned_user_ids=() if scenario == "unbound" else (source.actor_id,),
                )
                histories = {source: [{"role": "assistant", "content": prompt}]}
                if scenario == "foreign_mention":
                    histories[ctx.conv_key] = histories[source]
                with (
                    patch.object(self.chat, "get_history", side_effect=lambda key: histories.get(key, [])),
                    patch.object(self.chat, "AsyncOpenAI") as model,
                    patch.object(self.chat, "_stage_understand_operation", AsyncMock()) as semantic,
                    patch.object(self.chat, "get_ai_response_core", AsyncMock()) as response_model,
                ):
                    self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
                self.assertEqual(ctx.response, "这条提议不是发给你的，本次未执行。")
                self.assertEqual(self.chat._prepare_user_facing_reply(ctx.response, self.memory), ctx.response)
                self.assertEqual(len(ctx.response.splitlines()), 1)
                self.assertIs(self.store.get_record(source), ticket)
                self.assertEqual(ticket, original)
                if active is not None:
                    self.assertIs(self.chat.draft_operation_coordinator.get(source), active)
                    self.assertEqual(active, original_active)
                self.assertIsNone(self.store.get_record(ctx.conv_key))
                self.tool.assert_not_awaited()
                model.assert_not_called()
                semantic.assert_not_awaited()
                response_model.assert_not_awaited()

    async def test_all_closed_affirmatives_without_draft_use_fixed_reply(self):
        other = harness.ConversationAddress.group("qq", "b5", "another-owner")
        self.assertTrue(self.store.set(
            other, PendingToolConfirm("keytao_submit_batch", {"batch_id": "other-batch"}),
            space_key=other.space_key,
        ))
        other_ticket = self.store.get_record(other)
        for message in chat_routing.AFFIRMATIVE_REPLY_TEXTS:
            with self.subTest(message=message):
                ctx = self.context(message)
                with patch.object(self.chat, "AsyncOpenAI") as model:
                    self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
                model.assert_not_called()
                self.assertEqual(ctx.response, "当前没有待执行的操作；本次未写入。需要引用原提议确认后重新核对。")
                self.tool.assert_not_awaited()
                self.assertIs(self.store.get_record(other), other_ticket)

    async def test_unaddressed_affirmative_and_question_are_not_stale_controls(self):
        for message, addressed in (("确认", False), ("确认？", True)):
            self.assertFalse(await self.chat._try_stale_affirmative_reply(
                self.context(message, addressed=addressed),
            ))
        self.tool.assert_not_awaited()

    def test_general_fake_success_claims_are_replaced_without_receipt(self):
        for claim in ("已提交", "已成功提交", "提交成功", "审核中", "已加入词库", "已写入", "已删除", "审核通过", "已批准", "批准成功"):
            with self.subTest(claim=claim):
                self.assertEqual(
                    self.chat._prepare_user_facing_reply("✅ 批次" + claim + "。", self.memory),
                    "本次没有可核验的对应操作成功回执，无法确认操作已完成；请查看草稿核对。",
                )

    def test_real_submit_receipt_keeps_claim_and_other_actor_receipt_does_not(self):
        receipt = {
            "platform": "qq", "platformId": "abang", "operationKind": "draft_submit",
            "batchId": "offline-batch", "toolName": "keytao_submit_batch",
        }
        self.chat.current_draft_delivery_claims.set([receipt])
        reply = self.chat._prepare_user_facing_reply("✅ 批次已提交审核。", self.memory)
        self.assertIn("✅ 批次已提交审核。", reply)
        self.assertNotIn("无法确认", reply)
        self.chat.current_draft_delivery_claims.set([{**receipt, "platformId": "someone-else"}])
        self.assertIn("没有可核验", self.chat._prepare_user_facing_reply("已提交", self.memory))

    def test_submit_receipt_does_not_authorize_dictionary_or_delete_claims(self):
        receipt = {
            "platform": "qq", "platformId": "abang", "operationKind": "draft_submit",
            "batchId": "offline-batch", "toolName": "keytao_submit_batch",
        }
        self.chat.current_draft_delivery_claims.set([receipt])
        for claim in ("已加入词库", "已删除"):
            self.assertIn("没有可核验", self.chat._prepare_user_facing_reply(claim, self.memory))
        self.chat.current_draft_delivery_claims.set([{**receipt, "autoApproved": True}])
        self.assertIn("已加入词库", self.chat._prepare_user_facing_reply("已加入词库", self.memory))

    def test_negative_submit_status_is_not_a_success_claim(self):
        self.assertEqual(
            self.chat._prepare_user_facing_reply("本轮只写入草稿，尚未提交审核。", self.memory),
            "本轮只写入草稿，尚未提交审核。",
        )

    def test_status_lookup_and_preference_replies_are_not_current_action_claims(self):
        for reply in (
            "你最近的批次已提交，目前审核中，请耐心等待。",
            "「知更鸟」已加入词库，编码 zgn。",
            "好的，已更新你的称呼偏好。",
            "审核中。",
        ):
            with self.subTest(reply=reply):
                self.assertEqual(self.chat._prepare_user_facing_reply(reply, self.memory), reply)

    def test_status_reply_does_not_authorize_an_appended_current_action_claim(self):
        for claim in ("批次已提交。", "本次已删除草稿条目。", "我已为你重新提交。", "好的，已提交。"):
            with self.subTest(claim=claim):
                reply = "你最近的批次已提交，目前审核中，请耐心等待。\n" + claim
                self.assertIn("没有可核验", self.chat._prepare_user_facing_reply(reply, self.memory))

    def test_successful_submit_without_batch_id_survives_delivery(self):
        self.chat._chat_commands._capture_successful_draft_write_delivery(
            "keytao_submit_batch", {"success": True}, "qq", "abang",
        )
        reply = "✅ 批次已提交审核。"
        self.assertEqual(self.chat._prepare_user_facing_reply(reply, self.memory), reply)

    async def test_general_model_fake_submit_is_guarded_at_delivery(self):
        ctx = self.context("刚才那批处理完了吗？")
        with (
            patch.object(self.chat, "build_reply_context", AsyncMock(return_value="")),
            patch.object(self.chat, "get_ai_response_core", AsyncMock(return_value="✅ 批次已提交，审核中。")) as model,
        ):
            self.assertFalse(await self.chat._stage_generate_ai_response(ctx))
        model.assert_awaited_once()
        self.assertIn("没有可核验", self.chat._prepare_user_facing_reply(ctx.response, self.memory))

    def test_verified_historical_pending_reminder_survives_without_current_submit(self):
        item = {
            "source": "submitted", "batchId": "offline", "batchUrl": "https://keytao.test/batch/offline",
            "batchStatus": "Submitted", "itemStatus": "Pending", "action": "Create",
            "word": "开团", "code": "khtt", "type": "Phrase",
        }
        self.chat._chat_commands._capture_successful_draft_write_delivery(
            "keytao_pending_items_by_words", {"success": True, "complete": True, "items": [item]},
            "qq", "abang",
        )
        prompt = self.chat._chat_commands.prepend_pending_word_reminders("可以查看草稿核对。", [item])
        prepared = self.chat._prepare_user_facing_reply(prompt, self.memory)
        self.assertIn("审核中", prepared)
        self.assertEqual(self.chat._prepare_user_facing_reply(prepared, self.memory), prepared)
        self.assertNotIn("没有可核验", self.chat._prepare_user_facing_reply(prompt, self.memory))
        self.assertIn("没有可核验", self.chat._prepare_user_facing_reply(prompt + "\n批次已提交。", self.memory))
        self.chat.current_draft_delivery_claims.set([])
        self.assertEqual(self.chat._prepare_user_facing_reply(prompt, self.memory), prepared)

    def test_backed_duplicate_prompt_describes_existing_review_without_claiming_new_submit(self):
        state = pending_duplicate_confirmation_state(
            PendingToolConfirm("keytao_create_phrase", {"word": "开团", "code": "khtt", "preview_only": True}),
            {"pendingDuplicateConfirmation": True, "requiresConfirmation": True, "pendingItems": [{
                "source": "submitted", "batchId": "offline", "batchUrl": "https://offline.invalid/batch/offline",
                "batchStatus": "Submitted", "itemStatus": "Pending", "action": "Create",
                "word": "开团", "code": "khtt", "type": "Phrase",
            }]},
        )
        self.assertTrue(self.store.set(self.memory.conversation_address, state))
        prompt = self.chat._chat_render._format_pending_duplicate_confirmation(state)
        self.assertIn("该词已在审核中", prompt)
        expected = self.chat.render_platform_public_links(self.chat.strip_warning_count_copy(prompt), "qq")
        self.assertEqual(self.chat._prepare_user_facing_reply(prompt, self.memory), expected)
        self.assertEqual(self.chat._prepare_user_facing_reply(expected, self.memory), expected)
        self.assertIn("确认再提交", self.chat._prepare_user_facing_reply("审词：读音 kai tuan\n" + prompt, self.memory))
        self.assertIn("没有可核验", self.chat._append_unreported_submit_status(
            self.chat.strip_warning_count_copy(prompt) + "\n批次已提交。", [], self.memory.conversation_address,
        ))
        self.assertNotIn("批次已提交", self.chat._prepare_user_facing_reply(prompt + "\n批次已提交。", self.memory))

    async def test_fixed_stale_reply_survives_actual_delivery_preparation(self):
        ctx = self.context()
        self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
        token = self.chat._current_turn_message.set("确认")
        try:
            self.assertEqual(self.chat._prepare_user_facing_reply(ctx.response, self.memory), ctx.response)
        finally:
            self.chat._current_turn_message.reset(token)

    async def test_real_community_approval_keeps_dictionary_success_claim(self):
        fixture = vote_fixtures.VoteTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.addCleanup(fixture.tmp.cleanup)
        await fixture.cast("one", "甲")
        await fixture.cast("two", "乙")
        memory = harness.ChatMemoryContext(
            platform="qq", user_id="three", space_type="group", space_id="group", speaker_name="丙",
        )
        ctx = self.chat.TurnContext(
            bot=SimpleNamespace(), event=SimpleNamespace(), platform="qq", user_id="three",
            message_text="同意", normalized_message_text="同意", memory_context=memory, owner_label="丙",
            reply_reference=harness.ReplyReferenceInfo(is_reply=True, is_to_bot=True, text=vote_fixtures.RECEIPT),
        )
        with (
            patch.object(self.chat._community_vote, "get_store", return_value=fixture.store),
            patch.object(self.chat._community_vote, "find_user_by_platform", fixture.lookup),
            patch.object(self.chat.skills_manager, "get_tool_function", return_value=fixture.approve),
            patch.object(self.chat, "extract_onebot_reply_id", return_value="receipt"),
        ):
            self.assertTrue(await self.chat._stage_community_vote(ctx))
        fixture.approve.assert_awaited_once()
        self.assertEqual(fixture.store.get(vote_fixtures.BATCH_ID)["status"], "approved")
        delivered = self.chat._prepare_user_facing_reply(ctx.response, memory)
        self.assertIn("批次已加入词库", delivered)
        self.assertNotIn("没有可核验", delivered)
