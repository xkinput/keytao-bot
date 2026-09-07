"""Ensure offered answers cannot be claimed as new dictionary words."""

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness

chat = harness.openai_chat_module


class OfferedAnswerClaimingTests(unittest.TestCase):
    def test_contextual_extra_code_survives_scope_and_skips_intent_model(self):
        async def run():
            for context_only in (False, True):
                for message, retained in (("加入编码yeoiau", True), ("加入编码12", True), ("春暖花开", False)):
                    key = harness.ConversationAddress.private("qq", "s57-context-scope")
                    state = harness.chat_commands_module.create_pending_trusted_word_record(
                        "嘢", "yeoiav", "Single", context_only=context_only,
                    )
                    chat.conversation_state_store.set(key, state)
                    ctx = chat.TurnContext(
                        bot=object(), event=object(), platform="qq", user_id=key.actor_id,
                        normalized_message_text=message, conv_key=key, space_key=key.space_key,
                        command_intent_for=AsyncMock(side_effect=AssertionError("generic intent called")),
                    )
                    await chat._stage_resolve_current_pending_scope(ctx)
                    self.assertEqual(ctx.current_pending_record is not None, retained, message)
                    if retained:
                        with patch.object(chat, "AsyncOpenAI", side_effect=AssertionError("intent model called")):
                            intent = await chat._classify_message_command_intent(message, state)
                            await chat._stage_apply_scoped_pending_intent(ctx)
                        self.assertEqual(intent.intent, "none")
                        self.assertEqual(ctx.generic_command_intent.intent, "none")
                        execute = AsyncMock(return_value="bound contextual request")
                        with patch.object(chat, "_resolve_pending_trusted_word_action", execute):
                            await chat._stage_execute_pending_state(ctx)
                        self.assertEqual(execute.await_count, 1)
                        self.assertEqual(execute.call_args.args[1:5], (message, "qq", key.actor_id, key))
                    chat.conversation_state_store.delete(key)
        asyncio.run(run())

    def test_force_phrases_are_closed_assents(self):
        from keytao_bot.utils.offered_options import FORCE_ASSENT_TEXTS

        state = harness.PendingToolConfirm("keytao_shift_phrase_code", {})
        for text in FORCE_ASSENT_TEXTS:
            with self.subTest(text=text):
                self.assertTrue(chat._chat_routing._pending_assent_phrase_for_state(state, text).matched)
        for text in ('不要强制', '强制吗？', '他说硬来', '"硬加"', '强制添加其他词'):
            with self.subTest(text=text):
                self.assertFalse(chat._chat_routing._pending_assent_phrase_for_state(state, text).matched)

    def test_recordless_option_is_claimed_but_genuine_word_is_not(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-claim")
            history = [{"role": "assistant", "content": "请问是要强制调序，还是保留现顺序？"}]
            for message, previous, claimed in (
                ("保留现顺序", history, True),
                ("强制调序", history, True),
                ("硬来", [], True),
                ("春暖花开", [], False),
                ("保留现顺序", [{"role": "assistant", "content": "操作已完成。"}], False),
            ):
                ctx = SimpleNamespace(
                    conv_key=key, normalized_message_text=message, memory_context=None,
                    bot=None, event=None, user_id="s57-claim", QQMessageSegment=None,
                )
                with patch.object(chat, "get_history", return_value=previous), patch.object(
                    chat, "remember_conversation",
                ), patch.object(chat, "_finish_ai_chat_response", new=AsyncMock()) as finish:
                    self.assertEqual(await chat._stage_claim_offered_answer(ctx), claimed, message)
                    self.assertEqual(finish.await_count, int(claimed), message)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
