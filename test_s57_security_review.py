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

    def test_explicit_shared_prefix_uses_one_reading_and_keeps_manual_seal(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-security-reading")
            chat.conversation_state_store.delete(key)
            review = copy.deepcopy(existing_single_review())
            review["pronunciations"].append({
                "pinyin": "yè", "normalized": ["ye"], "codes": ["yeo"],
                "recommendedCode": "", "requiresManualReview": True,
                "candidateStatuses": [{"code": "yeo", "occupied": False, "words": []}],
            })
            tool = AsyncMock(side_effect=[json.dumps(review), json.dumps({"success": True, "phrases": []})])
            execute = AsyncMock(return_value="fixture written")
            with patch.object(commands, "call_tool_function", tool), patch.object(
                commands, "_execute_add_to_draft", execute,
            ):
                reply = await commands.try_handle_explicit_entry_code_command(
                    '添加单字"嘢"，编码为"yeoia"', "qq", key.actor_id, key,
                )
            self.assertIn("管理员复核", reply)
            self.assertEqual(tool.await_count, 2)
            self.assertEqual(execute.await_count, 1)
            self.assertIs(execute.call_args.args[7], True)
            self.assertEqual(execute.call_args.kwargs["reviewed_pinyin"], "yě")
            self.assertIsNone(chat.conversation_state_store.get_record(key))
            with patch.object(commands, "call_tool_function", AsyncMock(return_value=json.dumps(review))), patch.object(
                commands, "_execute_add_to_draft", AsyncMock(side_effect=AssertionError("wrong prefix reached write")),
            ):
                rejected = await commands.try_handle_explicit_entry_code_command(
                    '添加单字"嘢"，编码为"qxoia"', "qq", key.actor_id, key,
                )
            self.assertIn("音码前缀不符", rejected)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
