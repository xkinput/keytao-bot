"""Offline B9 regressions for pending-ticket QQ reply message IDs."""

import socket
import tempfile
import unittest
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness.state import SQLiteConversationStateStore


chat = harness.openai_chat_module
routing = harness.chat_routing_module


class MessageIdBindingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = harness.MemoryConversationStateStore()
        self.owner = harness.ConversationAddress.group("qq", "b9-group", "b9-owner")
        self.prompt = "Original candidate prompt"
        self.assertTrue(self.store.set(
            self.owner, harness.PendingAddWord(
                "发薪", "fsxb", [("fsxb", False)], server_candidates=[("fsxb", False)],
            ),
            origin_message_id="501",
        ))
        self.record = self.store.get_record(self.owner)
        self.assertTrue(self.store.bind_origin_prompt_digest(
            self.owner, chat._prompt_capability_digest(self.prompt),
        ))
        self.patcher = patch.object(chat, "conversation_state_store", self.store)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only"))
        network.start()
        self.addCleanup(network.stop)

    def context(self, message_id, text):
        reply = SimpleNamespace(
            is_reply=True, is_to_bot=True, message_id=message_id, text=text,
            mentioned_user_ids=(), sender_id="b9-bot", images=(),
        )
        return chat.TurnContext(
            bot=object(), event=object(), platform="qq", user_id=self.owner.actor_id,
            conv_key=self.owner, space_key=self.owner.space_key,
            reply_reference=reply,
        )

    def test_current_message_id_binds_rewritten_prompt(self):
        ctx = self.context("501", "Delivered prompt after contract rewrite")
        self.assertTrue(chat._verified_pending_reply(ctx, self.record))
        self.assertTrue(routing._verified_bot_reply_matches_record(ctx.reply_reference, self.record))

    def test_known_other_record_id_rejects_identical_text(self):
        other = harness.ConversationAddress.group("qq", "b9-group", "other-owner")
        self.assertTrue(self.store.set(other, self.record.state, origin_message_id="502"))
        for origin in ("501", ""):
            with self.subTest(origin=origin):
                self.record.origin_message_id = origin
                ctx = self.context("502", self.prompt)
                self.assertFalse(chat._verified_pending_reply(ctx, self.record))
                self.assertFalse(routing._verified_bot_reply_matches_record(ctx.reply_reference, self.record))

    def test_unknown_id_and_legacy_record_keep_digest_fallback(self):
        for origin, message_id in (("501", "999"), ("", "501"), ("", "")):
            with self.subTest(origin=origin, message_id=message_id):
                self.record.origin_message_id = origin
                self.assertTrue(chat._verified_pending_reply(self.context(message_id, self.prompt), self.record))
                self.assertFalse(chat._verified_pending_reply(self.context(message_id, "Unrelated prompt"), self.record))
        self.record.origin_prompt_digest = ""
        with patch.object(chat, "get_history", return_value=[{"role": "assistant", "content": self.prompt}]):
            self.assertTrue(chat._verified_pending_reply(self.context("999", self.prompt), self.record))

    def test_invalid_ids_cannot_bypass_digest_or_be_persisted(self):
        for message_id in (None, "", 0, "0", True, 501.0, "501x", [], {}):
            with self.subTest(message_id=message_id):
                self.assertFalse(chat._verified_pending_reply(self.context(message_id, self.prompt), self.record))
                self.assertFalse(self.store.bind_origin_message_ids(self.record, [message_id]))
                self.assertEqual(self.record.origin_message_id, "501")
        self.record.origin_message_id = '["501", "invalid"]'
        self.assertFalse(chat._verified_pending_reply(self.context("501", self.prompt), self.record))

    def test_message_ids_do_not_bypass_owner_bot_or_execution_checks(self):
        for change in (
            {"user_id": "other-owner"}, {"platform": "telegram"},
            {"conv_key": harness.ConversationAddress.group("qq", "other-group", self.owner.actor_id)},
        ):
            with self.subTest(change=change):
                self.assertFalse(chat._verified_pending_reply(replace(self.context("501", ""), **change), self.record))
        for change in ({"is_to_bot": False}, {"is_reply": False}, {"mentioned_user_ids": ("other-owner",)}):
            ctx = self.context("501", "")
            ctx.reply_reference = SimpleNamespace(**{**vars(ctx.reply_reference), **change})
            self.assertFalse(chat._verified_pending_reply(ctx, self.record))
        self.record.execution_id = "executing"
        self.assertFalse(chat._verified_pending_reply(self.context("501", ""), self.record))

    def test_all_segment_ids_survive_restart_and_persistence_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/state.db"
            store = SQLiteConversationStateStore(db_path=path)
            self.assertTrue(store.set(self.owner, self.record.state, origin_message_id="501"))
            record = store.get_record(self.owner)
            with patch.object(chat, "conversation_state_store", store):
                chat._bind_pending_reply_message_ids(record, [{"message_id": 502}, {"message_id": -503}])
            loaded = SQLiteConversationStateStore(db_path=path)
            record = loaded.get_record(self.owner)
            self.assertEqual(record.origin_message_ids, ("501", "502", "-503"))
            with patch.object(chat, "conversation_state_store", loaded):
                for message_id in record.origin_message_ids:
                    self.assertTrue(chat._verified_pending_reply(self.context(message_id, ""), record))
            before = record.origin_message_id
            with patch.object(loaded, "_persist_record", return_value=False):
                self.assertFalse(loaded.bind_origin_message_ids(record, [504]))
            self.assertEqual(record.origin_message_id, before)

    def delivery(self):
        memory = harness.ChatMemoryContext(
            platform="qq", user_id=self.owner.actor_id, space_type="group", space_id=self.owner.space_id,
        )
        prompt = chat._render_live_single_candidate_record(self.record)
        self.assertTrue(prompt)
        bot = type("Bot", (), {"__module__": "nonebot.adapters.onebot.v11"})()
        bot.send = AsyncMock(return_value={"message_id": 601})
        event = SimpleNamespace(message_id=123)
        segment = SimpleNamespace(reply=lambda value: "[reply]", at=lambda value: "[at]")
        return memory, prompt, bot, event, segment

    async def test_foreground_and_background_sends_bind_only_ticket_prompts(self):
        memory, prompt, bot, event, segment = self.delivery()
        for send in (chat._finish_ai_chat_response, chat._send_event_response):
            for adapter in (segment, None):
                with self.subTest(send=send.__name__, adapter=adapter):
                    self.record.origin_message_id = ""
                    await send(bot, event, self.owner.actor_id, memory, prompt, adapter)
                    self.assertEqual(self.record.origin_message_ids, ("601",))
                    self.assertTrue(chat._verified_pending_reply(self.context("601", "Rewritten prompt"), self.record))
                    bot.send.return_value = {"message_id": 602}
                    await send(bot, event, self.owner.actor_id, memory, "Unrelated progress", adapter)
                    self.assertEqual(self.record.origin_message_ids, ("601",))
                    bot.send.return_value = {"message_id": 601}

    async def test_matcher_send_binds_before_finish_raises(self):
        memory, prompt, _bot, _event, _segment = self.delivery()
        token = chat.current_memory_context.set(memory)

        async def finish():
            self.assertIn("603", self.record.origin_message_ids)
            raise harness.FinishedException()

        try:
            with patch.object(chat.ai_chat, "send", AsyncMock(return_value={"message_id": 603}), create=True), patch.object(
                chat.ai_chat, "finish", side_effect=finish,
            ):
                with self.assertRaises(harness.FinishedException):
                    await chat._finish_ai_chat_matcher(prompt)
        finally:
            chat.current_memory_context.reset(token)

    async def test_failed_or_invalid_send_and_replaced_ticket_never_gain_ids(self):
        memory, prompt, bot, event, _segment = self.delivery()
        self.record.origin_message_id = ""
        bot.send.side_effect = RuntimeError("Offline send failed")
        self.assertFalse(await chat._send_event_response(bot, event, self.owner.actor_id, memory, prompt))
        self.assertEqual(self.record.origin_message_ids, ())
        bot.send.side_effect = None
        bot.send.return_value = {}
        await chat._finish_ai_chat_response(bot, event, self.owner.actor_id, memory, prompt)
        self.assertEqual(self.record.origin_message_ids, ())

        async def replace_ticket(**kwargs):
            self.assertTrue(self.store.set(self.owner, self.record.state))
            return {"message_id": 604}

        bot.send.side_effect = replace_ticket
        await chat._finish_ai_chat_response(bot, event, self.owner.actor_id, memory, prompt)
        self.assertEqual(self.store.get_record(self.owner).origin_message_ids, ())
        self.assertEqual(self.record.origin_message_ids, ())

    async def test_qq_adapter_preserves_reply_id_through_text_rewrite(self):
        memory, _prompt, bot, _event, _segment = self.delivery()
        module = ModuleType("nonebot.adapters.onebot.v11")
        module.Bot = type(bot)
        bot.self_id = "b9-bot"
        bot.get_msg = AsyncMock(return_value={"sender": {"user_id": "b9-bot"}, "message": "Rewritten prompt"})
        event = SimpleNamespace(message=[SimpleNamespace(type="reply", data={"id": "501"})])
        with patch.dict("sys.modules", {"nonebot.adapters.onebot.v11": module}):
            reply = await chat.extract_reply_reference_info(bot, event)
        self.assertEqual(reply.message_id, "501")
        ctx = self.context("501", "")
        ctx.reply_reference = reply
        self.assertTrue(chat._verified_pending_reply(ctx, self.record))

    def test_native_reply_id_is_available_when_reference_metadata_omits_it(self):
        ctx = self.context("", "Rewritten native reply")
        ctx.reply_reference = harness.ReplyReferenceInfo(is_reply=True, is_to_bot=True, text=ctx.reply_reference.text)
        ctx.event = SimpleNamespace(reply=SimpleNamespace(message_id=501))
        self.assertTrue(chat._verified_pending_reply(ctx, self.record))
        self.assertEqual(ctx.reply_reference.message_id, "501")

    async def test_malformed_quote_never_becomes_an_unquoted_assent(self):
        memory, prompt, bot, _event, _segment = self.delivery()
        module = ModuleType("nonebot.adapters.onebot.v11")
        module.Bot = type(bot)
        for data in ({"id": 0}, {"id": ""}, {}):
            with self.subTest(data=data):
                event = SimpleNamespace(message=[SimpleNamespace(type="reply", data=data)])
                with patch.dict("sys.modules", {"nonebot.adapters.onebot.v11": module}):
                    reply = await chat.extract_reply_reference_info(bot, event)
                self.assertTrue(reply.is_reply)
                self.assertFalse(reply.is_to_bot)
                ctx = self.context("", "")
                ctx.reply_reference = reply
                ctx.event = event
                ctx.memory_context = memory
                ctx.normalized_message_text = ctx.message_text = "喵喵 加入"
                with patch.object(chat, "get_history", return_value=[{"role": "assistant", "content": prompt}]), patch.object(
                    chat, "handle_pending_message_core", AsyncMock(),
                ) as execute:
                    self.assertFalse(await chat._try_bound_pending_reply(ctx))
                execute.assert_not_awaited()
                self.assertIs(self.store.get_record(self.owner), self.record)

    async def test_rewritten_quote_cancel_consumes_only_current_ticket_without_model(self):
        memory, prompt, bot, event, segment = self.delivery()
        await chat._finish_ai_chat_response(bot, event, self.owner.actor_id, memory, prompt, segment)
        ctx = self.context("601", "Rewritten delivered prompt")
        ctx.memory_context = memory
        ctx.normalized_message_text = ctx.message_text = "取消"
        with patch.object(chat, "get_history", return_value=[{"role": "assistant", "content": "Outdated prompt"}]), patch.object(
            chat, "remember_conversation",
        ), patch.object(chat, "_finish_ai_chat_response", AsyncMock()), patch.object(
            chat, "AsyncOpenAI", side_effect=AssertionError("No model calls"),
        ):
            self.assertTrue(await chat._stage_understand_operation_early(ctx))
        self.assertEqual(ctx.response, "已取消。")
        self.assertIsNone(self.store.get_record(self.owner))

    async def test_assent_to_rewritten_or_body_only_segment_uses_live_candidates(self):
        for quote in ("Rewritten prompt without controls", ""):
            with self.subTest(quote=quote):
                self.assertTrue(self.store.set(self.owner, self.record.state))
                self.record = self.store.get_record(self.owner)
                memory, prompt, bot, event, segment = self.delivery()
                await chat._finish_ai_chat_response(bot, event, self.owner.actor_id, memory, prompt, segment)
                ctx = self.context("601", quote)
                ctx.memory_context = memory
                ctx.normalized_message_text = ctx.message_text = "好"
                with patch.object(chat, "get_history", return_value=[]), patch.object(
                    chat, "remember_conversation",
                ), patch.object(chat, "_finish_ai_chat_response", AsyncMock()), patch.object(
                    chat, "_execute_add_to_draft", AsyncMock(return_value="Offline addition"),
                ) as execute, patch.object(chat, "AsyncOpenAI", side_effect=AssertionError("No model calls")):
                    self.assertTrue(await chat._stage_understand_operation_early(ctx))
                execute.assert_awaited_once()
                self.assertEqual(execute.await_args.args[:2], ("发薪", "fsxb"))
                self.assertIsNone(self.store.get_record(self.owner))

    async def test_id_bound_tool_confirmation_uses_live_state_without_quote_footer(self):
        shift = harness.PendingToolConfirm("keytao_shift_phrase_code", {
            "word": "午时", "target_code": "wj", "expected_content_version": 0,
            "confirmed_plan_digest": "a" * 64,
            "_pending_display": {"shiftPlan": {
                "scope": "same_code",
                "currentState": [{"word": "午时", "code": "wj", "type": "Phrase", "weight": 101}],
                "proposedState": [{"word": "午时", "code": "wj", "type": "Phrase", "weight": 100}],
            }},
        }, confirmation_source="server_warning")
        for state in (shift, harness.PendingToolConfirm("keytao_remove_draft_item", {"item_id": "offline-item"})):
            with self.subTest(function=state.function_name):
                self.assertTrue(self.store.set(self.owner, state, origin_message_id="501"))
                ctx = self.context("501", "Only the body segment")
                ctx.normalized_message_text = ctx.message_text = "好"
                with patch.object(chat, "get_history", return_value=[]), patch.object(chat, "remember_conversation"), patch.object(
                    chat, "_finish_ai_chat_response", AsyncMock(),
                ), patch.object(chat, "handle_pending_message_core", AsyncMock(return_value="Offline confirmation")) as execute:
                    self.assertTrue(await chat._try_bound_pending_reply(ctx))
                execute.assert_awaited_once()
                self.assertEqual(execute.await_args.args[0], "确认")

    async def test_id_bound_assent_to_choice_offer_still_requires_a_named_option(self):
        state = harness.chat_commands_module._build_pending_choice_offer((
            ("A", "Keep current entry", None), ("B", "Leave unchanged", None),
        ))
        self.assertTrue(self.store.set(self.owner, state, origin_message_id="501"))
        ctx = self.context("501", "Only the body segment")
        ctx.normalized_message_text = ctx.message_text = "好"
        with patch.object(chat, "get_history", return_value=[]), patch.object(chat, "remember_conversation"), patch.object(
            chat, "_finish_ai_chat_response", AsyncMock(),
        ), patch.object(chat, "handle_pending_message_core", AsyncMock()) as execute:
            self.assertTrue(await chat._try_bound_pending_reply(ctx))
        execute.assert_not_awaited()
        self.assertEqual(ctx.response, harness.chat_commands_module.render_pending_choice_offer(state))
        self.assertIs(self.store.get(self.owner), state)


class DiscoveryMessageIdTests(unittest.IsolatedAsyncioTestCase):
    async def test_semantic_selection_skips_latest_digest_only_for_known_origin_id(self):
        from test_semantic_pending import SemanticPendingTests

        fixture = SemanticPendingTests()
        fixture.addCleanup = self.addCleanup
        try:
            await fixture.asyncSetUp()
            record, prompt = fixture.ticket()
            self.assertTrue(fixture.store.bind_origin_message_ids(record, [701, 702]))
            fixture.history[:] = [{"role": "assistant", "content": "Unrelated latest reply"}]
            fixture.ctx = fixture.context("我选 wja")
            fixture.ctx.reply_reference = harness.ReplyReferenceInfo(
                is_reply=True, is_to_bot=True, text="Rewritten prompt", message_id="701",
            )
            fixture.set_model("select_candidate", codes=("wja",), refers_to_pending=True)
            self.assertTrue(await fixture.chat._stage_understand_operation_early(fixture.ctx))
            fixture.add.assert_awaited_once()
            self.assertEqual(fixture.add.await_args.args[:2], ("午时", "wja"))
            self.assertIsNone(fixture.store.get_record(fixture.author))
        finally:
            await fixture.asyncTearDown()

    async def test_discovery_prompt_persists_its_send_id_on_pending_record(self):
        from test_chat_word_discovery import WordDiscoveryTests

        fixture = WordDiscoveryTests()
        fixture.addCleanup = self.addCleanup
        try:
            await fixture.asyncSetUp()
            await fixture.discovery.discover(fixture.bot, fixture.event, fixture.chat)
            record = fixture.pending.get_record(fixture.actor)
            self.assertIsNotNone(record)
            self.assertEqual(record.origin_message_ids, ("400",))
        finally:
            await fixture.asyncTearDown()


if __name__ == "__main__":
    unittest.main()
