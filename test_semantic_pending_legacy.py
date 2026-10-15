"""Offline replay of legacy receipt tickets through the ordered turn stages."""

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.plugins import chat_routing
from test_semantic_parser import FakeClient, raw_operation

from keytao_bot.harness.state import (
    DraftOperationCoordinator, MemoryConversationStateStore, PendingToolConfirm,
)
from keytao_bot.harness.conversation import ConversationAddress


ROUTE_OPS = "swap,delete,recode,move,reorder,batch_replace,draft_submit,draft_recall,draft_clear,draft_keep_only,query,chat,help"
PENDING_OPS = ",pending_release,pending_answer"
RECEIPT = "已添加「发薪日」→ fxra。回复「提交」送审。"


class LegacyPendingReplayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.deliver = self.chat._finish_ai_chat_response
        self.remember_real = self.chat.remember_conversation
        self.store = MemoryConversationStateStore()
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id="legacy-owner", space_type="group",
            space_id="legacy-group", speaker_name="ねんん",
        )
        self.author = self.memory.conversation_address
        self.client = FakeClient()
        self.history = [{"role": "assistant", "content": RECEIPT}]
        self.submissions = []
        self.tool_calls = []
        self.trace = []
        self.finish = AsyncMock()
        self.main = AsyncMock(return_value="[offline main-chat fallback]")
        patches = (
            patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": ROUTE_OPS + PENDING_OPS, "KEYTAO_SEMANTIC_SHADOW": "0"}),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "call_tool_function", AsyncMock(side_effect=self.tool)),
            patch.object(self.chat, "get_ai_response_core", self.main),
            patch.object(self.chat, "get_history", side_effect=lambda *args: self.history),
            patch.object(chat_routing, "_classify_simple_word_query_intent", AsyncMock(
                side_effect=lambda text, words: chat_routing.SimpleWordQueryIntent(True, ("提交",), "word_lookup", 0.9)
                if text.strip() == "提交" else chat_routing.SimpleWordQueryIntent(False),
            )),
            patch.object(self.chat, "remember_conversation", side_effect=self.remember),
            patch.object(self.chat, "schedule_memory_compaction"),
            patch.object(self.chat, "build_reply_context", AsyncMock(return_value="")),
            patch.object(self.chat, "_finish_ai_chat_response", self.finish),
            patch.object(self.chat.logger, "info"),
            patch.object(self.chat.logger, "warning"),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)
        self.seed_ticket()

    async def asyncTearDown(self):
        tasks = tuple(self.chat._semantic_shadow_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.chat._semantic_shadow_tasks.clear()

    def seed_ticket(self):
        state = PendingToolConfirm("keytao_submit_batch", {
            "_recent_batch_ids": ["offline-batch"], "_recent_content_version": 1,
            "_recent_own_write": True,
            "_recent_written_items": [{"action": "Create", "word": "发薪日", "code": "fxra", "type": "Phrase"}],
            "batch_id": "offline-batch",
        })
        self.assertTrue(self.store.set(self.author, state, space_key=self.author.space_key, owner_label="ねんん"))
        record = self.store.get_record(self.author)
        self.assertEqual((record.origin_prompt_digest, record.origin_message_id), ("", ""))
        self.assertEqual((record.requires_reconfirmation, record.confirmation_armed, record.reconfirmation_message), (True, True, "确认"))
        return record

    def remember(self, key, memory, message, response):
        self.history.extend([{"role": "user", "content": message}, {"role": "assistant", "content": response}])

    async def tool(self, name, args, platform, user_id, **kwargs):
        self.tool_calls.append((name, args.copy()))
        self.assertEqual((platform, user_id), ("qq", self.author.actor_id))
        if name == "keytao_submit_batch":
            self.assertFalse(self.submissions, "Duplicate offline batch submission")
            self.submissions.append(args.copy())
            return json.dumps({"success": True, "batchId": "offline-batch", "status": "pending"})
        if name == "keytao_lookup_by_word":
            word = args["word"]
            return json.dumps({"success": True, "phrases": [{"word": word, "code": "tkjc", "type": "Phrase"}] if word == "提交" else []})
        if name == "keytao_get_draft_batches":
            return json.dumps({"success": True, "batches": []})
        if name == "keytao_pending_items_by_words":
            return json.dumps({"success": True, "complete": True, "items": []})
        if name == "keytao_lookup_by_words_batch":
            return json.dumps({"success": True, "results": [
                {"word": word, "phrases": [{"word": word, "code": "tkjc", "type": "Phrase"}] if word == "提交" else []}
                for word in args["words"]
            ]})
        raise AssertionError("Unexpected offline tool: " + name)

    async def turn(self, message, *, op="confirm", refers=True, operands=(), reply_reference=None):
        self.client.create.return_value.choices[0].message.content = json.dumps(
            raw_operation(op=op, operands=list(operands), codes=[], refers_to_pending=refers),
        )
        ctx = self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.author.actor_id,
            normalized_message_text=message, message_text=message,
            conv_key=self.author, space_key=self.author.space_key,
            memory_context=self.memory, owner_label="ねんん",
        )
        if reply_reference is not None:
            ctx.reply_reference = reply_reference
        ctx.command_intent_cache, ctx.command_intent_for = self.chat.command_intent_memoizer(message, message)
        self.chat._current_turn_message.set(message)
        self.chat._current_conversational_turn.set(None)
        stages = self.chat.STAGES
        start = next(i for i, stage in enumerate(stages) if stage.__name__ == "_stage_understand_operation_early")
        for stage in stages[start:]:
            handled = await stage(ctx)
            self.trace.append((message, stage.__name__, bool(getattr(ctx, "semantic_pending_released", False)), ctx.response))
            if handled:
                break
        self.assertTrue(handled, self.trace)
        return ctx

    async def replay(self, tokens_on=True):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = ROUTE_OPS + (PENDING_OPS if tokens_on else "")
        replies = []
        for message, op, refers in (("提交", "confirm", True), ("确认", "confirm", True), ("提交草稿", "draft_submit", False)):
            ctx = await self.turn(message, op=op, refers=refers)
            replies.append(ctx.response)
        return replies

    async def test_legacy_three_turn_replay_submits_once(self):
        replies = await self.replay()
        self.assertEqual(len(self.submissions), 1, replies)
        self.assertIn("已提交审核", replies[0])
        self.assertNotIn("请先处理原候选", "\n".join(replies))
        self.assertFalse(any(name == "keytao_lookup_by_word" and args.get("word") == "提交" for name, args in self.tool_calls))

    async def test_tokens_off_three_turn_replay_submits_once(self):
        replies = await self.replay(tokens_on=False)
        self.assertEqual(len(self.submissions), 1, replies)
        self.assertIn("已提交审核", replies[0])
        self.assertNotIn("请先处理原候选", "\n".join(replies))

    def quote(self, text=RECEIPT, *, is_to_bot=True):
        return harness.ReplyReferenceInfo(is_reply=True, is_to_bot=is_to_bot, text=text)

    async def test_quoted_receipt_submit_then_confirm_submits_once(self):
        replies = []
        counts = []
        for message in ("提交", "确认"):
            ctx = await self.turn(message, reply_reference=self.quote())
            replies.append(ctx.response)
            counts.append(len(self.submissions))
            if message == "提交":
                self.main.assert_not_awaited()
        self.assertEqual(counts, [1, 1], replies)
        self.assertIn("已提交审核", replies[0])
        self.assertFalse(any(name == "keytao_lookup_by_word" for name, _args in self.tool_calls))

    async def test_quoted_receipt_confirm_executes_once(self):
        ctx = await self.turn("确认", reply_reference=self.quote())
        self.assertEqual(len(self.submissions), 1, ctx.response)
        self.assertIn("已提交审核", ctx.response)
        await self.turn("确认", reply_reference=self.quote())
        self.assertEqual(len(self.submissions), 1)

    async def test_digest_receipt_quote_submits_or_confirms_once(self):
        for message in ("提交", "确认"):
            with self.subTest(message=message):
                self.seed_ticket()
                self.submissions.clear()
                self.history[:] = [{"role": "assistant", "content": RECEIPT}]
                self.store.bind_origin_prompt_digest(self.author, chat_routing._prompt_capability_digest(RECEIPT))
                ctx = await self.turn(message, reply_reference=self.quote())
                self.assertEqual(len(self.submissions), 1, ctx.response)
                await self.turn("确认", reply_reference=self.quote())
                self.assertEqual(len(self.submissions), 1)

    async def test_legacy_quote_requires_latest_own_bot_reply(self):
        for history, quote in (
            ([{"role": "assistant", "content": RECEIPT}, {"role": "assistant", "content": "更新的机器人回复"}], self.quote()),
            ([], self.quote()),
            ([{"role": "user", "content": RECEIPT}], self.quote()),
            ([{"role": "assistant", "content": RECEIPT}], self.quote(is_to_bot=False)),
            ([{"role": "assistant", "content": RECEIPT}], harness.ReplyReferenceInfo(
                is_reply=True, is_to_bot=True, text=RECEIPT, mentioned_user_ids=("other-user",),
            )),
        ):
            with self.subTest(history=history, quote=quote):
                self.seed_ticket()
                self.history[:] = history
                ctx = await self.turn("确认", reply_reference=quote)
                self.assertEqual(self.submissions, [], ctx.response)

    async def test_other_author_cannot_execute_receipt(self):
        owner = self.author
        record = self.store.get_record(owner)
        self.author = ConversationAddress.group("qq", owner.space_id, "other-user")
        try:
            ctx = await self.turn("确认", reply_reference=self.quote())
            self.assertEqual(self.submissions, [], ctx.response)
            self.assertIs(self.store.get_record(owner), record)
        finally:
            self.author = owner

    async def test_other_recipient_quote_cannot_execute_digest_receipt(self):
        self.store.bind_origin_prompt_digest(self.author, chat_routing._prompt_capability_digest(RECEIPT))
        ctx = await self.turn("确认", reply_reference=harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, text=RECEIPT, mentioned_user_ids=("other-user",),
        ))
        self.assertEqual(self.submissions, [], ctx.response)

    async def test_history_keeps_final_receipt_text(self):
        self.store.delete(self.author)
        deliveries_token = self.commands.current_draft_delivery_claims.set([])
        memory_token = self.commands.current_memory_context.set(self.memory)
        bot = type("Bot", (), {"send": AsyncMock()})()
        response = "已添加「发薪日」→ fxra。"
        try:
            self.commands._capture_successful_draft_write_delivery(
                "keytao_create_phrase", {"success": True, "batchId": "offline-batch"},
                "qq", self.author.actor_id,
            )
            with (
                patch.object(self.chat.memory_store, "is_generation_current", return_value=True),
                patch.object(self.chat.history_store, "is_generation_current", return_value=True),
                patch.object(self.chat, "add_to_history", return_value=True) as remember,
                patch.object(self.chat.memory_store, "add_conversation_round", return_value=True),
            ):
                self.assertTrue(self.remember_real(self.author, self.memory, "加词 发薪日", response))
                await self.deliver(bot, object(), self.author.actor_id, self.memory, response)
            sent = bot.send.await_args.kwargs["message"]
            self.assertEqual(remember.call_args.args[2], sent)
        finally:
            self.commands.current_memory_context.reset(memory_token)
            self.commands.current_draft_delivery_claims.reset(deliveries_token)

    async def test_delivered_receipt_ticket_binds_actual_reply(self):
        self.store.delete(self.author)
        deliveries_token = self.commands.current_draft_delivery_claims.set([])
        memory_token = self.commands.current_memory_context.set(self.memory)
        bot = type("Bot", (), {"send": AsyncMock()})()
        try:
            self.commands._capture_successful_draft_write_delivery(
                "keytao_create_phrase", {
                    "success": True, "batchId": "offline-batch", "contentVersion": 1,
                    "writtenItems": [{"action": "Create", "word": "发薪日", "code": "fxra", "type": "Phrase"}],
                }, "qq", self.author.actor_id,
            )
            await self.deliver(bot, object(), self.author.actor_id, self.memory, RECEIPT)
        finally:
            self.commands.current_memory_context.reset(memory_token)
            self.commands.current_draft_delivery_claims.reset(deliveries_token)
        sent = bot.send.await_args.kwargs["message"]
        record = self.store.get_record(self.author)
        self.assertEqual(record.origin_prompt_digest, chat_routing._prompt_capability_digest(sent))
        self.history[:] = [{"role": "assistant", "content": sent}]
        ctx = await self.turn("确认", reply_reference=self.quote(sent))
        self.assertEqual(len(self.submissions), 1, ctx.response)

    async def test_final_reply_binds_ticket_without_challenge_stage(self):
        self.store.delete(self.author)
        from keytao_bot.harness.operation_request import OperationRequest
        response = await self.commands.try_handle_model_command(
            OperationRequest(op="draft_submit", frame="direct", refers_to_pending=False),
            "qq", self.author.actor_id, self.author, self.author.space_key, "ねんん",
        )
        record = self.store.get_record(self.author)
        record.origin_prompt_digest = ""
        prepared = self.chat._prepare_user_facing_reply(response, self.memory)
        self.assertEqual(record.origin_prompt_digest, chat_routing._prompt_capability_digest(prepared))

    async def test_progress_preserves_deliveries_until_final_reply(self):
        for platform, module, segment in (
            ("qq", "offline", None),
            ("qq", "nonebot.adapters.onebot.v11", object()),
            ("telegram", "nonebot.adapters.telegram", None),
        ):
            with self.subTest(transport=module):
                memory = harness.ChatMemoryContext(
                    platform=platform, user_id=self.author.actor_id,
                    space_type="group", space_id=self.author.space_id,
                )
                address = memory.conversation_address
                self.store.delete(address)
                deliveries = []
                delivery_token = self.commands.current_draft_delivery_claims.set(deliveries)
                memory_token = self.commands.current_memory_context.set(memory)
                bot = type("Bot", (), {"__module__": module, "send": AsyncMock()})()
                event = type("Event", (), {"message_id": 123})()
                try:
                    with (
                        patch.object(self.commands, "conversation_state_store", self.store),
                        patch.object(self.chat, "_build_qq_reply_message", side_effect=lambda *args: args[3]),
                        patch.object(self.chat, "_acknowledge_delivered_draft_mutations",
                                     wraps=self.chat._acknowledge_delivered_draft_mutations) as acknowledge,
                    ):
                        for batch_id, word in (("first-batch", "甲"), ("second-batch", "乙")):
                            self.commands._capture_successful_draft_write_delivery(
                                "keytao_create_phrase", {
                                    "success": True, "batchId": batch_id, "contentVersion": 1,
                                    "writtenItems": [{"action": "Create", "word": word, "code": "aa", "type": "Phrase"}],
                                }, platform, memory.user_id,
                            )
                            if batch_id == "first-batch":
                                self.assertTrue(await self.chat._send_event_response(
                                    bot, event, memory.user_id, memory,
                                    "正在处理「甲、乙、丙…」，已完成 8/12，预计还剩 1 轮",
                                    segment, emit_metrics=False,
                                ))
                                acknowledge.assert_not_called()
                                self.assertIsNone(self.store.get_record(address))
                                self.assertEqual([item["batchId"] for item in deliveries], ["first-batch"])
                        bot.send.side_effect = RuntimeError("Offline final send failure")
                        self.assertFalse(await self.chat._send_event_response(
                            bot, event, memory.user_id, memory, "处理完成。", segment,
                        ))
                        acknowledge.assert_not_called()
                        self.assertIsNone(self.store.get_record(address))
                        self.assertEqual([item["batchId"] for item in deliveries], ["first-batch", "second-batch"])
                        bot.send.side_effect = None
                        self.assertTrue(await self.chat._send_event_response(
                            bot, event, memory.user_id, memory, "处理完成。", segment,
                        ))
                        sent = bot.send.await_args.kwargs["message"]
                        acknowledge.assert_called_once_with(sent)
                        record = self.store.get_record(address)
                        self.assertEqual(record.state.args["_recent_batch_ids"], ["first-batch", "second-batch"])
                        self.assertEqual(record.origin_prompt_digest, chat_routing._prompt_capability_digest(sent))
                        self.assertEqual(deliveries, [])
                finally:
                    self.commands.current_memory_context.reset(memory_token)
                    self.commands.current_draft_delivery_claims.reset(delivery_token)

    async def test_unrelated_reply_does_not_rebind_legacy_ticket(self):
        record = self.store.get_record(self.author)
        self.chat._prepare_user_facing_reply("普通回复", self.memory)
        self.assertEqual(record.origin_prompt_digest, "")

    async def test_complete_submit_alone_supersedes_legacy_ticket(self):
        ctx = await self.turn("提交草稿", op="draft_submit", refers=False)
        record = self.store.get_record(self.author)
        self.assertNotIn("请先处理原候选", ctx.response)
        self.assertTrue(len(self.submissions) == 1 or record is not None and record.state.function_name == "keytao_understood_command", ctx.response)

    async def test_legacy_closed_command_ignores_model_nonwrite_override(self):
        ctx = await self.turn("提交", op="query", refers=False, operands=("提交",))
        self.assertEqual(len(self.submissions), 1, ctx.response)
        self.assertFalse(any(name == "keytao_lookup_by_word" for name, _args in self.tool_calls))

    async def test_legacy_answer_does_not_enter_semantic_canonicalizer(self):
        render = self.commands.render_understood_pending_answer
        with patch.object(self.commands, "render_understood_pending_answer", wraps=render) as canonical:
            await self.turn("确认")
        canonical.assert_not_called()

    async def test_unrelated_quote_never_executes_legacy_receipt(self):
        record = self.store.get_record(self.author)
        ctx = await self.turn("提交", reply_reference=harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, text="另一条不相关消息",
        ))
        self.assertEqual(self.submissions, [], ctx.response)
        self.assertIsNot(self.store.get_record(self.author), record)
        self.assertEqual(self.store.get(self.author).function_name, "keytao_understood_command")
        self.assertIn("我理解为", ctx.response)

    async def test_unrelated_quoted_assent_preserves_legacy_receipt(self):
        record = self.store.get_record(self.author)
        ctx = await self.turn("确认", reply_reference=harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True, text="另一条不相关消息",
        ))
        self.assertEqual(self.submissions, [], ctx.response)
        self.assertIs(self.store.get_record(self.author), record)

    async def test_closed_command_with_wrong_model_answer_uses_legacy_resolver(self):
        ctx = await self.turn("提交草稿", op="confirm", refers=True)
        self.assertEqual(len(self.submissions), 1, ctx.response)


if __name__ == "__main__":
    unittest.main()
