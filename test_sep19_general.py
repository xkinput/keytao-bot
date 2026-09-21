"""Offline regressions for chat after routing and unrelated live candidates."""

import unittest
from unittest.mock import patch

import test_state_machine as harness
import test_s63_general_reply as general
import test_s54_multiword as multiword
from types import SimpleNamespace
from keytao_bot.utils.observability import begin_turn_metrics, current_turn_metrics, end_turn_metrics


chat = harness.openai_chat_module


class GeneralRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_routing_calls_still_allow_the_actual_chat_answer(self):
        for message in (
            "问下喵喵",
            "喵喵，这样加词，一年能加多少词进词库",
            "喵喵现在不会回答无关紧要的问题了",
        ):
            with self.subTest(message=message):
                token = begin_turn_metrics("qq", "group")
                try:
                    current_turn_metrics().model_calls = 2
                    response = await general.GeneralReplyTests().reply(message, "Fixture conversational answer.")
                    self.assertEqual(response, "Fixture conversational answer.")
                    self.assertEqual(current_turn_metrics().model_calls, 3)
                finally:
                    end_turn_metrics(token)


class UnrelatedCandidateDeliveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_current_answer_is_not_replaced_with_an_unrelated_live_candidate(self):
        _, store, key = await multiword.MultiwordRouteTests().discover()
        state = store.get(key)
        answer = "偿债：语料频次 37；从众：语料频次 31。两者接近。"
        ctx = SimpleNamespace(response=answer, conv_key=key,
                              generic_command_intent=harness.MessageCommandIntent(),
                              simple_word_query_words=())
        digest = store.get_record(key).origin_prompt_digest
        token = chat._current_turn_message.set("“偿债”和“从众”这两个词的词频如何")
        try:
            with patch.object(chat, "conversation_state_store", store):
                await chat._stage_append_ticket_challenge(ctx)
                await chat._stage_enforce_advertised_reply_contract(ctx)
                delivered = chat._enforce_advertised_reply_contract(ctx.response, key)
        finally:
            chat._current_turn_message.reset(token)
        self.assertEqual(delivered, answer)
        self.assertIs(store.get(key), state)
        self.assertEqual(store.get_record(key).origin_prompt_digest, digest)

    async def test_current_candidate_offer_still_binds_its_confirmation(self):
        answer, store, key = await multiword.MultiwordRouteTests().discover()
        with patch.object(chat, "conversation_state_store", store):
            self.assertTrue(answer)
            rendered = chat._append_pending_ticket_challenge(answer, key)
            self.assertTrue(store.get_record(key).origin_prompt_digest)
            self.assertIn("加入", rendered)


if __name__ == "__main__":
    unittest.main()
