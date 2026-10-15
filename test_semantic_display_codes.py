"""Offline regression for real draft codes in blocked-shift choices."""

import json
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness


class SemanticDisplayCodeTests(unittest.IsolatedAsyncioTestCase):
    async def test_draft_replacement_choices_show_existing_codes(self):
        chat = harness.openai_chat_module
        commands = harness.chat_commands_module
        tools = harness._draft_tools
        owner = harness.ConversationAddress.private("qq", "semantic-display-owner")
        store = harness.MemoryConversationStateStore()
        rows = [
            {"id": 81, "action": "Create", "word": "午时", "code": "wjekuo",
             "type": "Phrase", "weight": 100},
            {"id": 82, "action": "Create", "word": "务实", "code": "wjeku",
             "type": "Phrase", "weight": 100},
        ]
        snapshot = {"success": True, "batchId": "display-draft", "contentVersion": 3,
                    "items": rows}
        shift_results = []

        async def words(requested):
            return {"success": True, "results": [
                {"word": word, "phrases": []} for word in requested
            ]}

        async def codes(requested):
            return {"success": True, "results": [
                {"code": code, "phrases": []} for code in requested
            ]}

        async def call(name, args, platform, user_id, **kwargs):
            self.assertEqual((platform, user_id), ("qq", owner.actor_id))
            if name == "keytao_shift_phrase_code":
                result = await tools.keytao_shift_phrase_code(platform, user_id, **args)
                shift_results.append(result)
            elif name == "keytao_batch_remove_draft_items":
                self.assertEqual(args, {"ids": [81, 82], "batch_id": "display-draft"})
                result = {
                    "success": False, "requiresConfirmation": True,
                    "confirmationKind": "deleteTargets", "batchId": "display-draft",
                    "contentVersion": 3, "targets": rows, "targetDigest": "a" * 64,
                }
            else:
                raise AssertionError(f"Unexpected offline tool call: {name}")
            return json.dumps(result, ensure_ascii=False)

        source = harness.PendingToolConfirm("keytao_shift_phrase_code", {
            "word": "午时", "target_code": "wjeku",
            "ordered_words": ["午时", "务实"],
            "expected_codes": ["wjeku", "wjekuo"],
        })
        with (
            patch.object(chat, "conversation_state_store", store),
            patch.object(commands, "call_tool_function", side_effect=call) as dispatch,
            patch.object(tools, "keytao_list_draft_items", AsyncMock(return_value=snapshot)),
            patch.object(tools, "_lookup_words_raw", side_effect=words),
            patch.object(tools, "_lookup_codes_raw", side_effect=codes),
            patch.object(tools, "_fetch_encode_candidates", AsyncMock(return_value={
                "success": True, "candidateCodes": ["wjeku", "wjekuo"],
            })),
            patch.object(tools, "_keytao_strict_batch_add_to_draft", AsyncMock(
                side_effect=AssertionError("Blocked draft replacement wrote"),
            )) as write,
        ):
            reply = await commands._execute_confirmed_tool(source, "qq", owner.actor_id, owner)
            self.assertEqual(len(shift_results), 1)
            replacements = shift_results[0]["relatedDraftItems"]
            self.assertEqual([item["fromCode"] for item in replacements], ["wjekuo", "wjeku"])
            self.assertTrue(all("code" not in item for item in replacements))
            self.assertIn("Create 「午时」@wjekuo", reply)
            self.assertIn("Create 「务实」@wjeku", reply)
            self.assertNotIn("未知编码", reply)
            record = store.get_record(owner)
            self.assertIsNotNone(commands._pending_choice_options(record.state))
            self.assertEqual(reply, commands.render_pending_choice_offer(record.state))
            self.assertEqual([call.args[0] for call in dispatch.await_args_list], [
                "keytao_shift_phrase_code", "keytao_batch_remove_draft_items",
            ])
            write.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
