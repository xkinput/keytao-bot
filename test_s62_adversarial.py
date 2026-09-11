"""Independent S62 authorization and delivery checks under the offline guard."""

import json
import unittest
from unittest.mock import patch

import test_s62_incident as incident
from test_s54_selection import fixtures
from keytao_bot.plugins import chat_commands as commands, chat_routing as routing
from keytao_bot.utils.pending_confirmation import advertised_command_suggestions

incident_record = incident.incident_record


class S62AdversarialTests(unittest.IsolatedAsyncioTestCase):
    async def test_reported_and_negative_envelopes_never_write(self):
        for prefix in ("我说", "阿邦说", "原话是", "转述", "先不要", "暂时不", "并非", "尚未", "绝不能"):
            message = f"{prefix}鸡白汤 jbtaua, 敲不死 qbso，加入并提交"
            with self.subTest(message=message):
                self.assertFalse(routing.message_authorizes_live_pending_mutation(message, incident_record()))

    async def test_reported_speech_cannot_reach_fake_mutation(self):
        writes, submitted, _, calls, _ = await incident.S62IncidentTests().replay([
            "原话是鸡白汤 jbtaua, 敲不死 qbso，加入并提交",
        ])
        self.assertEqual(writes, [])
        self.assertFalse(submitted)
        self.assertEqual(calls, [])

    def test_exact_live_lexical_targets_keep_the_selection_contract(self):
        for word in ("不明觉厉", "记忆力", "说唱"):
            record = incident_record()
            record.args = json.loads(json.dumps(record.args, ensure_ascii=False).replace("敲不死", word))
            with self.subTest(word=word):
                self.assertTrue(routing.message_authorizes_live_pending_mutation(f"{word} 2，加入", record))

    async def test_bare_selection_delivery_keeps_write_receipt_and_skipped_reason(self):
        writes, submitted, replies, _, state = await incident.S62IncidentTests().replay(
            ["鸡白汤 jbtaua, 敲不死 3"], delivery=True,
        )
        self.assertEqual([(r["word"], r["code"]) for r in writes], [("敲不死", "qbsoa")])
        self.assertFalse(submitted)
        self.assertIsNone(state)
        delivered = replies[0]
        self.assertIn("敲不死 → qbsoa", delivered)
        self.assertIn("https://keytao.rea.ink/batch/s62-fixture", delivered)
        self.assertIn("鸡白汤", delivered)
        self.assertIn("未加入", delivered)

    def test_invalid_selection_delivery_keeps_explanation_and_retry(self):
        state = incident_record()
        _, _, reply = routing._resolve_multi_word_pending_candidate_selection(state, "敲不死 99，加入")
        store = fixtures.MemoryConversationStateStore()
        key = fixtures.ConversationAddress.group("qq", "865189947", "s62-offline")
        store.set(key, state)
        with patch.object(fixtures.openai_chat_module, "conversation_state_store", store):
            delivered = fixtures.openai_chat_module._enforce_advertised_reply_contract(reply, key)
        self.assertIn("99", delivered)
        self.assertIn("超出", delivered)
        retry = advertised_command_suggestions(delivered)
        self.assertEqual(len(retry), 1)
        self.assertTrue(routing.message_authorizes_live_pending_mutation(retry[0], state))

    def test_protected_recommended_code_keeps_its_reason_at_final_delivery(self):
        state = incident_record()
        state.args["items"][0]["code"] = "qbs"
        state.args["_candidate_scopes"][0]["reviewedState"]["recommendedCode"] = "qbs"
        _, _, reply = routing._resolve_multi_word_pending_candidate_selection(state, "敲不死 1")
        self.assertIn("常用度", reply)
        store = fixtures.MemoryConversationStateStore()
        key = fixtures.ConversationAddress.group("qq", "865189947", "s62-offline")
        store.set(key, state)
        with patch.object(fixtures.openai_chat_module, "conversation_state_store", store):
            delivered = fixtures.openai_chat_module._enforce_advertised_reply_contract(reply, key)
        self.assertEqual(str(delivered), str(reply))
        self.assertIn("敲破死", delivered)
        self.assertIn("未加入", delivered)


if __name__ == "__main__":
    unittest.main()
