"""Single-type record, selector and mutation capability closure."""

import asyncio
import json
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness.state import SQLiteConversationStateStore

chat = harness.openai_chat_module
commands = harness.chat_commands_module


def single_review():
    return {
        "success": True, "word": "鎗", "type": "Single", "encodingType": "单字",
        "recommendedCode": "qx", "needsManualReview": True,
        "variantNote": "「鎗」是「枪」（繁体「槍」）的异体字",
        "manualReviewReason": "异体字需要管理员审核",
        "preSubmitAudit": {"summary": "异体字需要管理员审核"},
        "pronunciations": [{
            "pinyin": "qiāng", "normalized": ["qiang"], "codes": ["qx"],
            "recommendedCode": "qx", "requiresManualReview": True,
            "sources": [{"source": "单字资料"}],
            "candidateStatuses": [{"code": "qx", "occupied": False, "words": []}],
        }],
    }


class SinglePipelineTests(unittest.TestCase):
    def test_one_candidate_keeps_review_record_and_bound_actions_after_reload(self):
        async def run(db_path):
            store = SQLiteConversationStateStore(db_path=db_path)
            key = harness.ConversationAddress.private("qq", "s55-single")

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(single_review())
                raise AssertionError(name)

            with patch.object(chat, "conversation_state_store", store), patch.object(
                chat, "call_tool_function", side_effect=tool,
            ), patch.object(
                chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(True, ("鎗",), "word_lookup", 1.0)),
            ), patch.object(commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)):
                reply = await commands._try_handle_simple_single_word_query("鎗", "qq", "s55-single", key)
                for expected in ("单字", "审词：读音 qiāng", "异体字", "1. qx", "加入", "加入并提交"):
                    self.assertIn(expected, reply)
                reloaded = SQLiteConversationStateStore(db_path=db_path)
                record = reloaded.get_record(key)
                self.assertEqual(record.state.phrase_type, "Single")
                self.assertEqual(record.state.server_candidates, [("qx", False)])
                self.assertTrue(chat._advertised_reply_matches_live_record(reply, record))
                for action in ("1", "qx"):
                    self.assertTrue(chat._chat_routing.message_authorizes_live_pending_mutation(action, record.state), action)
                for action, intent_name in (("加入", "pending_confirm"), ("加入并提交", "pending_add_and_submit")):
                    assent = chat._chat_routing._pending_assent_phrase_for_state(record.state, action)
                    self.assertTrue(assent.matched and assent.add_requested, action)
                    intent = await chat._classify_message_command_intent(action, record.state)
                    self.assertEqual(intent.intent, intent_name)
                args = commands._create_phrase_args(record.state, "qx")
                self.assertEqual(args["type"], "Single")
                capability = commands._reviewed_create_capability("鎗", "qx", "qiāng", ("qx",))
                self.assertEqual(capability[("鎗", "qx")]["type"], "Single")
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory + "/state.db"))


if __name__ == "__main__":
    unittest.main()
