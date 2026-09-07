"""Replay every guarded Single advertisement through persisted live binding."""

import asyncio
import json
import tempfile
import unittest
import unicodedata
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from test_s55_single_pipeline import single_review
from keytao_bot.harness.authorization_grammar import parse_eviction_modified_add
from keytao_bot.harness.state import SQLiteConversationStateStore
from keytao_bot.utils.pending_confirmation import advertised_reply_contract


class S56AdvertisedClosureTests(unittest.TestCase):
    def test_guarded_single_commands_and_complete_bullets_bind_after_reload(self):
        async def run(db_path):
            chat = harness.openai_chat_module
            commands = harness.chat_commands_module
            store = SQLiteConversationStateStore(db_path=db_path)
            key = harness.ConversationAddress.private("qq", "s56-closure")
            reviewed = single_review()
            reviewed["recommendedCode"] = ""
            reviewed["pronunciations"][0]["recommendedCode"] = ""
            reviewed["pronunciations"][0]["candidateStatuses"] = [{
                "code": "qx", "occupied": True, "words": ["强"],
                "phrases": [{"word": "强", "code": "qx", "type": "Single", "weight": 10}],
            }]
            reviewed["candidateOrderingAssessments"] = [{
                "newWord": "鎗", "occupantWord": "强", "occupantCode": "qx",
                "freeCode": "", "newCode": "", "recommendedCode": "",
                "verdict": "behind_more_common", "decisionReason": "frequency_ratio",
                "summary": "fixture evidence: 10 vs 17342",
            }]

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(reviewed)
                raise AssertionError(name)

            with patch.object(chat, "conversation_state_store", store), patch.object(
                chat, "call_tool_function", side_effect=tool,
            ), patch.object(
                chat, "_classify_simple_word_query_intent",
                AsyncMock(return_value=harness.SimpleWordQueryIntent(True, ("鎗",), "word_lookup", 1.0)),
            ), patch.object(commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)):
                reply = await commands._try_handle_simple_single_word_query("鎗", "qq", "s56-closure", key)
                record = SQLiteConversationStateStore(db_path=db_path).get_record(key)
                self.assertIsNotNone(record, reply)
                self.assertEqual(record.state.phrase_type, "Single")
                self.assertEqual(record.state.server_candidates, [("qx", True)])
                contract = advertised_reply_contract(reply)
                self.assertEqual(contract.batch_assent_forms, ())
                self.assertEqual(contract.command_suggestions, ("添加 鎗 qx,挤掉 强", "添加 鎗 qx,顶替 强"))
                self.assertTrue(chat._advertised_reply_matches_live_record(reply, record), reply)
                for command in contract.command_suggestions:
                    parsed = parse_eviction_modified_add(command)
                    self.assertIsNotNone(parsed, command)
                    self.assertEqual((parsed.word, parsed.code, parsed.named_occupant), ("鎗", "qx", "强"))
                    self.assertTrue(chat._chat_routing.message_authorizes_live_pending_mutation(command, record.state), command)
                    envelope = next(line for line in reply.splitlines() if line.lstrip().startswith("- ") and command in unicodedata.normalize("NFKC", line))
                    self.assertTrue(chat._chat_routing.message_authorizes_live_pending_mutation(envelope, record.state), envelope)
                self.assertNotIn("推荐编码：qx", reply)
                self.assertIn("加入编码 qx+形码", reply)

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory + "/state.db"))


if __name__ == "__main__":
    unittest.main()
