"""S56 occupied-only discovery retains durable, parser-bound explicit actions."""

import asyncio
import json
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness.state import SQLiteConversationStateStore
from keytao_bot.utils.pending_confirmation import advertised_command_suggestions
from test_s55_single_pipeline import single_review

chat = harness.openai_chat_module
commands = harness.chat_commands_module


class CommonnessBindingTests(unittest.TestCase):
    def test_occupied_only_discovery_survives_reload_and_whole_reply_binding(self):
        async def run(path):
            store = SQLiteConversationStateStore(db_path=path)
            key = harness.ConversationAddress.private("qq", "s56-binding")
            review = single_review()
            review["recommendedCode"] = ""
            review["pronunciations"][0]["recommendedCode"] = ""
            review["pronunciations"][0]["candidateStatuses"] = [{
                "code": "qx", "occupied": True, "words": ["强"], "label": "已有「强」",
                "phrases": [{"code": "qx", "word": "强", "type": "Single", "weight": 10}],
            }]
            review["candidateOrderingAssessments"] = [{
                "verdict": "behind_more_common", "newWord": "鎗", "occupantWord": "强",
                "occupantCode": "qx", "freeCode": "", "newCode": "",
            }]

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(review)
                raise AssertionError(name)

            with patch.object(chat, "conversation_state_store", store), patch.object(
                chat, "call_tool_function", side_effect=tool,
            ), patch.object(chat, "_classify_simple_word_query_intent", AsyncMock(
                return_value=harness.SimpleWordQueryIntent(True, ("鎗",), "word_lookup", 1.0),
            )), patch.object(commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)):
                reply = await commands._try_handle_simple_single_word_query("鎗", "qq", "s56-binding", key)
                self.assertIn("推荐编码：暂无", reply)
                self.assertIn("审词：读音 qiāng", reply)
                reloaded = SQLiteConversationStateStore(db_path=path)
                record = reloaded.get_record(key)
                self.assertIsNotNone(record)
                self.assertEqual("", record.state.recommended_code)
                self.assertEqual("Single", record.state.phrase_type)
                self.assertEqual(review["candidateOrderingAssessments"], record.state.server_ordering_assessments)
                self.assertTrue(chat._advertised_reply_matches_live_record(reply, record), reply)
                self.assertEqual(reply, chat._enforce_advertised_reply_contract(reply, key))
                for command in advertised_command_suggestions(reply):
                    self.assertTrue(chat._chat_routing.message_authorizes_live_pending_mutation(command, record.state), command)
                for line in reply.splitlines():
                    if line.startswith("- 「"):
                        self.assertTrue(chat._chat_routing.message_authorizes_live_pending_mutation(line, record.state), line)
                self.assertFalse(chat._live_candidate_affordances_are_complete(
                    reply + "\n回复「加入」写入草稿，或回复「加入并提交」写入并提交。", record,
                ))
                self.assertTrue(chat._chat_routing.message_authorizes_live_pending_mutation("加入编码qxioio", record.state))
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory + "/state.db"))


if __name__ == "__main__":
    unittest.main()
