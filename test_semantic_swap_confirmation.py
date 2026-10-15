"""Offline model-swap checks through the real planner and confirmation store."""

import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from test_semantic_parser import FakeClient, raw_operation
from keytao_bot.harness.conversation import ConversationAddress
from keytao_bot.harness.state import MemoryConversationStateStore, PendingToolConfirm


class SemanticSwapConfirmationTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_swap_only_previews_and_seals_an_author_owned_plan(self):
        chat = harness.openai_chat_module
        commands = harness.chat_commands_module
        tools = harness._draft_tools
        author = ConversationAddress.group("qq", "semantic-swap-group", "swap-author")
        other = ConversationAddress.group("qq", "semantic-swap-group", "other-user")

        for second_code in ("wj", "wja"):
            with self.subTest(second_code=second_code):
                store = MemoryConversationStateStore()
                rows = [
                    {"word": "午时", "code": "wj", "type": "Phrase", "weight": 100},
                    {"word": "务实", "code": second_code, "type": "Phrase", "weight": 101},
                ]
                snapshot = {"success": True, "batchId": "", "contentVersion": 0, "items": []}
                write = AsyncMock(side_effect=AssertionError("swap wrote before author confirmation"))
                preview = AsyncMock(return_value={
                    "success": False, "requiresConfirmation": True,
                    "warningDigest": "d" * 64, "warnings": [],
                })
                shift_calls = []

                async def words(requested):
                    return {"success": True, "results": [
                        {"word": word, "phrases": [row for row in rows if row["word"] == word]}
                        for word in requested
                    ]}

                async def codes(requested):
                    return {"success": True, "results": [
                        {"code": code, "phrases": [row for row in rows if row["code"] == code]}
                        for code in requested
                    ]}

                async def strict(*args, **kwargs):
                    if kwargs.get("confirmed"):
                        return await write(*args, **kwargs)
                    return await preview(*args, **kwargs)

                async def call(name, args, platform, user_id, **kwargs):
                    self.assertEqual((platform, user_id), ("qq", author.actor_id))
                    if name == "keytao_lookup_by_words_batch":
                        result = await words(args["words"])
                    elif name == "keytao_lookup_by_code":
                        result = {"success": True, "phrases": [
                            row for row in rows if row["code"] == args["code"]
                        ]}
                    elif name == "keytao_list_draft_items":
                        result = snapshot
                    elif name == "keytao_shift_phrase_code":
                        shift_calls.append(dict(args))
                        result = await tools.keytao_shift_phrase_code(platform, user_id, **args)
                    else:
                        raise AssertionError(f"unexpected tool: {name}")
                    return json.dumps(result, ensure_ascii=False)

                ctx = chat.TurnContext(
                    bot=None, event=None, platform="qq", user_id=author.actor_id,
                    normalized_message_text="帮我把午时跟务实的码对调",
                    conv_key=author, space_key=author.space_key, owner_label="Author",
                )
                client = FakeClient(json.dumps(raw_operation(
                    op="swap", operands=["午时", "务实"], codes=[],
                )))
                with (
                    patch.dict(os.environ, {
                        "KEYTAO_SEMANTIC_ROUTE_OPS": "swap", "KEYTAO_SEMANTIC_SHADOW": "0",
                    }),
                    patch.object(chat, "AsyncOpenAI", return_value=client),
                    patch.object(chat, "conversation_state_store", store),
                    patch.object(chat, "remember_conversation"),
                    patch.object(chat, "_finish_ai_chat_response", new_callable=AsyncMock),
                    patch.object(commands, "call_tool_function", side_effect=call) as dispatch,
                    patch.object(tools, "_lookup_words_raw", side_effect=words),
                    patch.object(tools, "_lookup_codes_raw", side_effect=codes),
                    patch.object(tools, "keytao_list_draft_items", AsyncMock(return_value=snapshot)),
                    patch.object(tools, "_fetch_encode_candidates", AsyncMock(return_value={
                        "success": True, "candidateCodes": ["wj", "wja"],
                    })),
                    patch.object(tools, "_keytao_strict_batch_add_to_draft", side_effect=strict),
                ):
                    self.assertTrue(await chat._stage_understand_operation(ctx))
                    client.create.assert_awaited_once()
                    self.assertEqual(len(shift_calls), 1)
                    self.assertNotIn("confirmed_plan_digest", shift_calls[0])
                    self.assertNotIn("confirmed", shift_calls[0])
                    preview.assert_awaited_once()
                    write.assert_not_awaited()

                    record = store.get_record(author)
                    self.assertIsNotNone(record, ctx.response)
                    self.assertEqual(record.owner_key, author)
                    self.assertEqual(record.owner_label, "Author")
                    self.assertIsInstance(record.state, PendingToolConfirm)
                    self.assertEqual(record.state.function_name, "keytao_shift_phrase_code")
                    self.assertEqual(record.state.confirmation_source, "server_warning")
                    self.assertTrue(commands.server_warning_ticket_is_complete(record.state))
                    self.assertEqual(len(record.state.args["confirmed_plan_digest"]), 64)
                    self.assertIn("确认", ctx.response)
                    self.assertIn("午时", ctx.response)
                    self.assertIn("务实", ctx.response)

                    dispatch_count = dispatch.await_count
                    self.assertIsNone(store.get_record(other))
                    self.assertIsNone(await commands.handle_pending_message_core(
                        "确认", "qq", other.actor_id, other,
                        space_key=other.space_key, allow_intent_model=False,
                    ))
                    self.assertIsNone(await commands.handle_pending_message_core(
                        "今天天气不错", "qq", author.actor_id, author,
                        space_key=author.space_key, allow_intent_model=False,
                    ))
                    self.assertEqual(dispatch.await_count, dispatch_count)
                    self.assertIs(store.get_record(author), record)
                    write.assert_not_awaited()

                    confirmed, response = await commands._resolve_pending_ticket_control(
                        record, "确认", commands.MessageCommandIntent(), "qq", author.actor_id,
                    )
                    self.assertEqual(confirmed.intent, "pending_confirm")
                    self.assertIsNone(response)
                    unrelated, _ = await commands._resolve_pending_ticket_control(
                        record, "今天天气不错",
                        commands.MessageCommandIntent(intent="pending_confirm", confidence=1.0),
                        "qq", author.actor_id,
                    )
                    self.assertNotEqual(unrelated.intent, "pending_confirm")
                    write.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
