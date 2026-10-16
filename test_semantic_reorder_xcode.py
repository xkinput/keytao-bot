"""Offline cross-code reorder checks through model routing and real swap handlers."""

import asyncio
import copy
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from test_semantic_parser import FakeClient, raw_operation

from keytao_bot.harness.conversation import ConversationAddress
from keytao_bot.harness.operation_request import OperationRequest
from keytao_bot.harness.state import (
    DraftOperationCoordinator,
    MemoryConversationStateStore,
    PendingToolConfirm,
)


class SemanticReorderXcodeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.author = ConversationAddress.group("qq", "reorder-xcode-group", "reorder-owner")
        self.other = ConversationAddress.group("qq", "reorder-xcode-group", "other-owner")
        self.store = MemoryConversationStateStore()
        self.client = FakeClient()
        self.rows = [self.row("前摇", "qmyzo", 101), self.row("前要", "qmyz", 100)]
        self.draft_items = []
        self.calls = []
        self.writes = []
        self.warnings = []
        self.extra_shifted = []
        patches = (
            patch.dict(os.environ, {
                "KEYTAO_SEMANTIC_ROUTE_OPS": "reorder,direct_exec",
                "KEYTAO_SEMANTIC_SHADOW": "0",
            }),
            patch.object(self.chat, "AsyncOpenAI", return_value=self.client),
            patch.object(self.chat, "WORD_QUERY_INTENT_MODEL", "offline-model"),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "call_tool_function", side_effect=self.call),
            patch.object(self.chat, "remember_conversation"),
            patch.object(self.chat, "_finish_ai_chat_response", new_callable=AsyncMock),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    async def asyncTearDown(self):
        tasks = tuple(self.chat._semantic_shadow_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.chat._semantic_shadow_tasks.clear()

    @staticmethod
    def row(word, code, weight=100):
        return {"word": word, "code": code, "type": "Phrase", "weight": weight}

    async def call(self, name, args, platform, user_id, **kwargs):
        self.assertEqual((platform, user_id), ("qq", self.author.actor_id))
        self.calls.append((name, copy.deepcopy(args)))
        if name == "keytao_lookup_by_words_batch":
            data = {"success": True, "results": [
                {"word": word, "phrases": [row for row in self.rows if row["word"] == word]}
                for word in args["words"]
            ]}
        elif name == "keytao_lookup_by_code":
            data = {"success": True, "phrases": [
                row for row in self.rows if row["code"] == args["code"]
            ]}
        elif name == "keytao_list_draft_items":
            data = {
                "success": True, "batchId": "offline-batch" if self.draft_items else "",
                "contentVersion": 1 if self.draft_items else 0,
                "items": self.draft_items,
            }
        elif name == "keytao_shift_phrase_code":
            data = {
                "success": True, "batchId": "offline-batch", "contentVersion": 2,
                "batchUrl": "https://offline.invalid/draft/offline-batch",
                "shiftPlan": {
                    "shifted": [
                        {"word": word, "fromCode": old, "toCode": new}
                        for word, old, new in (
                            ("前摇", "qmyzo", "qmyz"), ("前要", "qmyz", "qmyzo"),
                        )
                    ] + self.extra_shifted,
                    "items": [],
                },
            }
            if not args.get("confirmed_plan_digest"):
                data.update(
                    success=False, requiresConfirmation=True, confirmationKind="shiftPlan",
                    planDigest="b" * 64, warningDigest="c" * 64, warnings=self.warnings,
                )
            else:
                self.writes.append((name, copy.deepcopy(args)))
        else:
            raise AssertionError(f"Unexpected offline tool: {name} {args!r}")
        return json.dumps(data, ensure_ascii=False)

    async def stage(self, message="喵喵前摇比前要优先级要高，调在前面", operands=("前摇", "前要")):
        self.client.create.return_value.choices[0].message.content = json.dumps(
            raw_operation(op="reorder", operands=list(operands), codes=[]), ensure_ascii=False,
        )
        ctx = self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=self.author.actor_id,
            normalized_message_text=message, conv_key=self.author,
            space_key=self.author.space_key, owner_label="Offline owner",
        )
        self.assertTrue(await self.chat._stage_understand_operation(ctx), ctx.response)
        self.client.create.assert_awaited_once()
        self.assertEqual(ctx.operation_request.op, "reorder")
        self.assertEqual(ctx.operation_request.source, "model:offline-model")
        return ctx.response

    async def answer(self, message, actor=None):
        actor = actor or self.author
        return await self.commands.handle_pending_message_core(
            message, "qq", actor.actor_id, actor,
            space_key=actor.space_key, allow_intent_model=False,
        )

    def assert_swap_write(self, response):
        self.assertEqual(len(self.writes), 1, (self.calls, response))
        name, args = self.writes[0]
        self.assertEqual(name, "keytao_shift_phrase_code")
        self.assertEqual(args["confirmed_plan_digest"], "b" * 64)
        self.assertEqual(args["word"], "前摇")
        self.assertEqual(args["target_code"], "qmyz")
        self.assertEqual(args["ordered_words"], ["前摇", "前要"])
        self.assertEqual(args["expected_codes"], ["qmyz", "qmyzo"])
        self.assertEqual(args["expected_weights"], [100, 101])
        for value in ("前摇", "前要", "qmyz", "qmyzo", "https://offline.invalid/draft/offline-batch"):
            self.assertIn(value, response)
        self.assertNotIn("回复「确认」", response)
        self.assertIsNone(self.store.get_record(self.author), response)

    async def test_production_direct_reorder_swaps_exact_codes_once_with_receipt_and_no_ticket(self):
        self.assert_swap_write(await self.stage())

    async def test_reorder_plan_keeps_the_validated_direction_without_a_second_lookup(self):
        lookup_count = 0

        async def changing_call(name, args, platform, user_id, **kwargs):
            nonlocal lookup_count
            if name == "keytao_lookup_by_words_batch":
                lookup_count += 1
                if lookup_count > 1:
                    self.rows = [self.row("前摇", "qmyz", 100), self.row("前要", "qmyzo", 101)]
            return await self.call(name, args, platform, user_id, **kwargs)

        with patch.object(self.chat, "call_tool_function", side_effect=changing_call):
            self.assert_swap_write(await self.stage())
        self.assertEqual(lookup_count, 1)

    async def test_without_direct_exec_preserves_model_then_swap_confirmation_gates(self):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "reorder"
        response = await self.stage()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.writes, [])
        record = self.store.get_record(self.author)
        self.assertIsInstance(record.state, PendingToolConfirm)
        self.assertEqual(record.state.function_name, "keytao_understood_command")
        self.assertIn("回复「确认」执行", response)
        self.assertIsNone(await self.answer("确认", self.other))
        self.assertEqual(self.calls, [])
        response = await self.answer("确认")
        self.assert_swap_ticket(response)
        self.assertIsNone(await self.answer("确认", self.other))
        self.assertEqual(self.writes, [])
        self.assert_swap_write(await self.answer("确认"))
        self.assertIsNone(await self.answer("确认"))
        self.assertEqual(len(self.writes), 1)

    def assert_swap_ticket(self, response):
        self.assertEqual(self.writes, [], response)
        record = self.store.get_record(self.author)
        self.assertIsNotNone(record, response)
        self.assertEqual(record.owner_key, self.author)
        self.assertIsInstance(record.state, PendingToolConfirm)
        self.assertEqual(record.state.function_name, "keytao_shift_phrase_code")
        self.assertEqual(record.state.confirmation_source, "server_warning")
        self.assertTrue(self.commands.server_warning_ticket_is_complete(record.state))
        self.assertEqual(record.state.args["ordered_words"], ["前摇", "前要"])
        self.assertEqual(record.state.args["expected_codes"], ["qmyz", "qmyzo"])
        self.assertIn("确认", response)

    async def test_dispatcher_without_direct_context_seals_native_swap_ticket(self):
        request = OperationRequest(op="reorder", operands=("前摇", "前要"))
        response = await self.commands.execute_understood_command(
            request, self.commands.render_understood_command(request),
            "qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner",
        )
        self.assert_swap_ticket(response)
        self.assert_swap_write(await self.answer("确认"))

    async def test_already_ahead_returns_exact_no_change_reply_without_a_write(self):
        response = await self.stage("前要优先于前摇", ("前要", "前摇"))
        self.assertEqual(
            response,
            "「前要」已经排在「前摇」前面（前要 qmyz / 前摇 qmyzo），本次未修改。",
        )
        self.assertEqual(self.writes, [])
        self.assertIsNone(self.store.get_record(self.author))

    async def test_unrelated_codes_are_named_and_explained_without_a_write(self):
        self.rows = [self.row("前摇", "abcd", 101), self.row("前要", "xyz", 100)]
        response = await self.stage()
        for value in ("前摇", "前要", "abcd", "xyz"):
            self.assertIn(value, response)
        self.assertRegex(response, r"不.*同.*链")
        self.assertNotIn("规范指令无法执行", response)
        self.assertEqual(self.writes, [])
        self.assertIsNone(self.store.get_record(self.author))

    async def test_ambiguous_word_names_all_codes_without_a_write(self):
        self.rows.append(self.row("前摇", "qmyzoo", 102))
        response = await self.stage()
        for value in ("前摇", "前要"):
            self.assertIn(value, response)
        for code in ("qmyzo", "qmyzoo", "qmyz"):
            self.assertRegex(response, rf"(?<![a-z]){code}(?![a-z])")
        self.assertRegex(response, r"不唯一|多个|多条|歧义")
        self.assertNotIn("规范指令无法执行", response)
        self.assertEqual(self.writes, [])
        self.assertIsNone(self.store.get_record(self.author))

    async def test_resolution_applies_current_draft_before_deciding_the_chain(self):
        self.rows[0] = self.row("前摇", "xyz", 101)
        self.draft_items = [
            {"id": 1, "action": "Delete", **self.row("前摇", "xyz", 101)},
            {"id": 2, "action": "Create", **self.row("前摇", "qmyzo", 101)},
        ]
        self.assert_swap_write(await self.stage())

    async def test_direct_reorder_preserves_server_warning_ownership_gate(self):
        self.warnings = [{"warningType": "duplicate_code", "word": "额外词条", "code": "qmyz"}]
        self.assert_swap_ticket(await self.stage())

    async def test_direct_reorder_cannot_auto_confirm_unnamed_shifted_entries(self):
        self.extra_shifted = [{"word": "额外词条", "fromCode": "qmyz", "toCode": "qmyza"}]
        self.assert_swap_ticket(await self.stage())


if __name__ == "__main__":
    unittest.main()
