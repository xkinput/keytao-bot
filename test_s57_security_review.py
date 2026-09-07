"""Independent additional-code authorization and ambiguous-reading controls."""

import asyncio
import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from test_s57_extra_code import existing_single_review

commands = harness.chat_commands_module
chat = harness.openai_chat_module


class ExtraCodeSecurityReviewTests(unittest.TestCase):
    def test_other_actor_cannot_consume_existing_word_context(self):
        async def run():
            owner = "s57-security-owner"
            key = harness.ConversationAddress.private("qq", owner)
            state = commands.create_pending_trusted_word_record(
                "嘢", "yeoiav", "Single", context_only=True,
            )
            chat.conversation_state_store.set(key, state)
            before = chat.conversation_state_store.get_record(key)
            tool = AsyncMock(side_effect=AssertionError("foreign actor dispatched a tool"))
            with patch.object(commands, "call_tool_function", tool):
                reply = await commands.handle_pending_message_core(
                    "加入编码yeoia", "qq", "s57-security-other", key,
                    allow_intent_model=False,
                )
            self.assertIsNone(reply)
            self.assertEqual(tool.await_count, 0)
            self.assertEqual(chat.conversation_state_store.get_record(key), before)
        asyncio.run(run())

    def test_ambiguous_reviewed_prefix_cannot_authorize_an_unknown_suffix(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-security-reading")
            chat.conversation_state_store.delete(key)
            review = copy.deepcopy(existing_single_review())
            review["pronunciations"].append({
                "pinyin": "yè", "normalized": ["ye"], "codes": ["yeo"],
                "recommendedCode": "", "requiresManualReview": True,
                "candidateStatuses": [{"code": "yeo", "occupied": False, "words": []}],
            })
            tool = AsyncMock(return_value=json.dumps(review))
            execute = AsyncMock(side_effect=AssertionError("ambiguous reading reached write"))
            with patch.object(commands, "call_tool_function", tool), patch.object(
                commands, "_execute_add_to_draft", execute,
            ):
                reply = await commands.try_handle_explicit_entry_code_command(
                    '添加单字"嘢"，编码为"yeoia"', "qq", key.actor_id, key,
                )
            self.assertIn("多个待定读音", reply)
            self.assertEqual(tool.await_count, 1)
            self.assertEqual(execute.await_count, 0)
            self.assertIsNone(chat.conversation_state_store.get_record(key))
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
