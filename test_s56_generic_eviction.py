"""Generic shift tails cannot replace a protected occupant without naming it."""

import asyncio
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness.authorization_grammar import parse_eviction_modified_add


chat = harness.openai_chat_module
commands = harness.chat_commands_module


class GenericEvictionTests(unittest.TestCase):
    async def run_command(self, message, *, preexisting=False, verdict="behind_more_common"):
        key = harness.ConversationAddress.private("qq", "s56-generic")
        store = harness.MemoryConversationStateStore()

        def reviewed_state():
            return harness.PendingAddWord(
                word="耙耙柑", recommended_code="ppgv",
                candidates=[("ppg", True), ("ppgv", False)],
                server_candidates=[("ppg", True), ("ppgv", False)],
                occupied_words={"ppg": ["琵琶骨"]},
                server_occupied_words={"ppg": ["琵琶骨"]},
                server_ordering_assessments=[{
                    "newWord": "耙耙柑", "occupantWord": "琵琶骨", "occupantCode": "ppg",
                    "freeCode": "ppgv", "newCode": "ppg" if verdict == "front_more_common" else "ppgv",
                    "verdict": verdict,
                }] if verdict else [],
                pronunciation_codes={"ppg": "pa pa gan", "ppgv": "pa pa gan"},
                needs_manual_review=True,
            )

        if preexisting:
            store.set(key, reviewed_state())

        async def review(*_args, **_kwargs):
            store.set(key, reviewed_state())
            return "reviewed fixture"

        async def lookup(name, *_args, **_kwargs):
            self.assertEqual("keytao_lookup_by_code", name)
            return json.dumps({"success": True, "phrases": [
                {"word": "琵琶骨", "code": "ppg", "type": "Phrase", "weight": 100},
            ]})

        ctx = SimpleNamespace(
            response=None, history=[], normalized_message_text=message,
            platform="qq", user_id="s56-generic", conv_key=key, space_key=None, owner_label="",
            current_pending_record=store.get_record(key),
            eviction_modified_add=parse_eviction_modified_add(message), compound_eviction_add_plan=None,
        )
        shift = AsyncMock(return_value="planned fixture")
        add = AsyncMock(side_effect=AssertionError("protected eviction cannot become a plain add"))
        with (
            patch.object(commands, "conversation_state_store", store),
            patch.object(commands, "_try_handle_simple_single_word_query", side_effect=review),
            patch.object(commands, "call_tool_function", side_effect=lookup),
            patch.object(commands, "_execute_shift_to_code", shift),
            patch.object(commands, "_execute_add_to_draft", add),
            patch.object(chat, "_try_handle_explicit_reading_disambiguation", AsyncMock(return_value=None)),
            patch.object(chat, "_try_handle_compound_shift_modified_add_command", AsyncMock(return_value=None)),
        ):
            await chat._stage_handle_simple_word_query(ctx)
        self.assertIsNotNone(store.get_record(key))
        self.assertIsNotNone(ctx.response)
        return ctx.response, shift

    def test_generic_tail_without_record_cannot_create_protected_shift_ticket(self):
        async def run():
            reply, shift = await self.run_command("加词 耙耙柑 ppg，顺延其他词条")
            self.assertEqual(0, shift.await_count, reply)
            self.assertIn("琵琶骨", reply)
            self.assertIn("本次未写入", reply)
        asyncio.run(run())

    def test_generic_tail_with_live_record_cannot_create_protected_shift_ticket(self):
        async def run():
            reply, shift = await self.run_command("加词 耙耙柑 ppg，顺延其他词条", preexisting=True)
            self.assertEqual(0, shift.await_count, reply)
            self.assertIn("琵琶骨", reply)
            self.assertIn("本次未写入", reply)
        asyncio.run(run())

    def test_lookup_inferred_occupant_is_not_user_named_authorization(self):
        async def run():
            reply, shift = await self.run_command("加词 耙耙柑 ppg 重新编码", verdict="")
            self.assertEqual(0, shift.await_count, reply)
            self.assertIn("琵琶骨", reply)
        asyncio.run(run())

    def test_user_named_occupant_keeps_the_complete_shift_path(self):
        async def run():
            reply, shift = await self.run_command("加词 耙耙柑 ppg，顺延琵琶骨", preexisting=True)
            self.assertEqual("planned fixture", reply)
            self.assertEqual(1, shift.await_count)
            self.assertTrue(shift.call_args.kwargs["auto_confirm_shift_plan"])
            self.assertEqual(("琵琶骨",), shift.call_args.kwargs["auto_confirm_expected_shifted_words"])
        asyncio.run(run())

    def test_synthetic_weaker_assessment_preserves_generic_shift(self):
        async def run():
            # This bound assessment is synthetic; it does not assert corpus frequencies.
            reply, shift = await self.run_command(
                "加词 耙耙柑 ppg，顺延其他词条", verdict="front_more_common",
            )
            self.assertEqual("planned fixture", reply)
            self.assertEqual(1, shift.await_count)
            self.assertFalse(shift.call_args.kwargs["auto_confirm_shift_plan"])
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
