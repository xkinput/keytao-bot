"""Offline regressions for B5 submit-ticket preservation and draft deletes."""

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import test_state_machine as harness

from keytao_bot.harness.conversation import ConversationAddress
from keytao_bot.harness.operation_request import OperationRequest
from keytao_bot.harness.state import DraftOperationCoordinator, MemoryConversationStateStore, PendingAddWord, PendingToolConfirm


class B5DeleteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.commands = harness.chat_commands_module
        self.chat = harness.openai_chat_module
        self.store = MemoryConversationStateStore()
        self.author = ConversationAddress.group("qq", "b5-delete-group", "b5-delete-owner")
        self.calls = []
        self.items = []
        self.lookup_rows = []
        self.version = 1
        self.fail_remove = False
        self.delivery_token = self.commands.current_draft_delivery_claims.set([])
        self.addCleanup(self.commands.current_draft_delivery_claims.reset, self.delivery_token)
        self.tool = AsyncMock(side_effect=self.call)
        for context in (
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.commands, "call_tool_function", self.tool),
            patch.object(self.commands, "_format_draft_response", AsyncMock(return_value="草稿地址：https://offline.invalid/draft/b5-delete")),
        ):
            context.start()
            self.addCleanup(context.stop)

    def ticket(self):
        self.assertTrue(self.store.set(
            self.author,
            PendingToolConfirm("keytao_submit_batch", {"batch_id": "b5-delete"}),
            space_key=self.author.space_key,
        ))
        return self.store.get_record(self.author)

    def result(self, **changes):
        return {"success": True, "batchId": "b5-delete", "contentVersion": self.version, **changes}

    async def call(self, name, args, platform, user_id, **kwargs):
        self.assertEqual((platform, user_id), ("qq", self.author.actor_id))
        self.calls.append((name, copy.deepcopy(args)))
        if name == "keytao_lookup_by_word":
            result = {"success": True, "phrases": copy.deepcopy(self.lookup_rows)}
        elif name == "keytao_list_draft_items":
            result = self.result(items=copy.deepcopy(self.items))
        elif name == "keytao_batch_remove_draft_items":
            targets = [
                {key: item[key] for key in ("id", "word", "code", "action", "type")}
                for item in self.items if item["id"] in args["ids"]
            ]
            if self.fail_remove:
                result = {"success": False, "message": "offline removal failed"}
            elif not args.get("expected_target_digest"):
                result = self.result(
                    success=False, requiresConfirmation=True, targets=targets,
                    targetDigest="a" * 64, warningDigest="a" * 64, warnings=[],
                )
            else:
                self.items = [item for item in self.items if item["id"] not in args["ids"]]
                self.version += 1
                result = self.result(successCount=len(args["ids"]), failedCount=0)
        else:
            raise AssertionError(f"Unexpected offline tool: {name}")
        return json.dumps(result, ensure_ascii=False)

    async def delete(self, *, model=False):
        if model:
            return await self.commands.execute_model_command_direct(
                OperationRequest(op="delete", operands=("屡有",), codes=("llydav",)),
                ("屡有",), "qq", self.author.actor_id, self.author,
                self.author.space_key, "Offline owner",
            )
        return await self.commands._try_handle_draft_management_command(
            "删除 屡有 @ llydav", "qq", self.author.actor_id,
            self.author.space_key, "Offline owner",
            self.commands.MessageCommandIntent(
                intent="dictionary_delete", confidence=1.0,
                keep_words=("屡有",), requested_code="llydav",
            ),
        )

    async def test_failed_model_delete_preserves_pending_submit(self):
        record = self.ticket()
        response = await self.delete(model=True)
        self.assertIn("没有可删除", response)
        self.assertIs(self.store.get_record(self.author), record)
        self.assertFalse(record.execution_id)

    async def test_prod_delete_sequence_keeps_submit_confirm_executable(self):
        record = self.ticket()
        response = await self.commands.handle_pending_message_core(
            "删除 屡有 @ llydav", "qq", self.author.actor_id, self.author,
            space_key=self.author.space_key, allow_intent_model=False,
        )
        self.assertIsNone(response)
        self.assertIs(self.store.get_record(self.author), record)
        self.assertIn("没有可删除", await self.delete(model=True))
        execute = AsyncMock(return_value="已提交审核")
        with patch.object(self.commands, "_execute_confirmed_tool", execute):
            response = await self.commands.handle_pending_message_core(
                "确认", "qq", self.author.actor_id, self.author,
                space_key=self.author.space_key, allow_intent_model=False,
            )
        self.assertEqual(response, "已提交审核")
        self.assertEqual(execute.await_args.args[0].function_name, "keytao_submit_batch")
        self.assertIsNone(self.store.get_record(self.author))

    async def test_failed_lookup_does_not_read_draft_or_drop_submit(self):
        record = self.ticket()
        self.tool.side_effect = [
            json.dumps({"success": False, "message": "offline lookup failed"}),
        ]
        execute = AsyncMock()
        with patch.object(self.commands, "_execute_confirmed_tool", execute):
            response = await self.delete(model=True)
        self.assertIn("无法取得", response)
        self.tool.assert_awaited_once_with("keytao_lookup_by_word", {"word": "屡有"}, "qq", self.author.actor_id)
        execute.assert_not_awaited()
        self.assertIs(self.store.get_record(self.author), record)

    async def test_unavailable_draft_read_keeps_exact_live_delete_path(self):
        for draft_result in (
            {"success": False, "message": "offline network error"},
            {"success": True},
            {"success": True, "items": None},
            TimeoutError("offline timeout"),
            OSError("offline network error"),
        ):
            with self.subTest(draft_result=draft_result):
                record = self.ticket()
                self.lookup_rows = [{"word": "屡有", "code": "llydav", "type": "Phrase"}]
                draft_read = AsyncMock(
                    side_effect=draft_result if isinstance(draft_result, Exception) else None,
                    return_value=draft_result,
                )
                execute = AsyncMock(return_value="locked delete confirmation")
                with (
                    patch.object(self.commands, "_fetch_current_draft_items", draft_read),
                    patch.object(self.commands, "_execute_confirmed_tool", execute),
                ):
                    response = await self.delete(model=True)
                self.assertEqual(response, "locked delete confirmation")
                self.assertEqual(execute.await_args.args[0].args["items"], [{
                    "action": "Delete", "word": "屡有", "code": "llydav", "type": "Phrase",
                }])
                self.assertEqual(execute.await_args.args[0].confirmation_source, "local_preview")
                draft_read.assert_not_awaited()
                self.assertIs(self.store.get_record(self.author), record)

    async def test_unbound_draft_read_keeps_bind_help_and_pending_submit(self):
        record = self.ticket()
        with patch.object(self.commands, "_fetch_current_draft_items", AsyncMock(return_value={
            "success": False, "not_bound": True,
        })):
            response = await self.delete(model=True)
        self.assertEqual(response, self.commands._BIND_HELP_TEXT)
        self.tool.assert_awaited_once_with("keytao_lookup_by_word", {"word": "屡有"}, "qq", self.author.actor_id)
        self.assertIs(self.store.get_record(self.author), record)

    async def test_unrelated_invalid_command_preserves_pending_submit(self):
        record = self.ticket()
        response = await self.commands.execute_model_command_direct(
            OperationRequest(op="swap", operands=("屡有",)), ("屡有",),
            "qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner",
        )
        self.assertIn("未写入", response)
        self.assertIs(self.store.get_record(self.author), record)
        self.tool.assert_not_awaited()

    async def test_failed_replacement_review_preserves_exact_submit_ticket(self):
        record = self.ticket()
        original = copy.deepcopy(record)

        async def review(word, platform, user_id, conv_key, space_key, owner_label, **kwargs):
            self.store.set(conv_key, PendingAddWord(
                word, "n", [("n", False)], server_candidates=[("n", False)],
                pronunciation_codes={"n": "new"},
            ), space_key=space_key)
            return "offline reviewed candidate"

        self.tool.side_effect = None
        self.tool.return_value = json.dumps({"success": True, "phrases": [
            {"word": "屡有", "code": "llydav", "type": "Phrase"},
        ]})
        with patch.object(self.commands, "_try_handle_simple_single_word_query", side_effect=review):
            response = await self.commands.execute_model_command_direct(
                OperationRequest(op="move", operands=("屡有", "新词"), codes=("llydav",)),
                ("屡有", "新词"), "qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner",
            )
        self.assertIn("本次未生成替换计划", response)
        self.assertIs(self.store.get_record(self.author), record)
        self.assertEqual(record, original)

    async def test_failed_explicit_add_preserves_exact_submit_ticket(self):
        record = self.ticket()
        original = copy.deepcopy(record)
        derived = PendingAddWord(
            "新词", "xk", [("xk", False)], server_candidates=[("xk", False)],
            pronunciation_codes={"xk": "xin ci"},
        )
        self.tool.side_effect = None
        self.tool.return_value = json.dumps({"success": True, "word": "新词", "type": "Phrase"})
        with (
            patch.object(self.commands, "select_candidate_inventory", return_value=SimpleNamespace(
                candidates=[("xk", False)], readings={"xk": "xin ci"},
                statuses=[{"code": "xk", "occupied": False, "words": []}],
            )),
            patch.object(self.commands, "_bind_explicit_pending_code", AsyncMock(return_value=(derived, ""))),
            patch.object(self.commands, "_execute_add_to_draft", AsyncMock(return_value="添加失败：offline rejected")),
        ):
            response = await self.commands.execute_model_command_direct(
                OperationRequest(op="add", operands=("新词",), codes=("xk",)),
                ("新词",), "qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner",
            )
        self.assertIn("添加失败", response)
        self.assertIs(self.store.get_record(self.author), record)
        self.assertEqual(record, original)

    async def test_exact_draft_create_delete_uses_draft_removal(self):
        for model in (False, True):
            with self.subTest(model=model):
                self.items = [
                    {"id": 1, "word": "屡有", "code": "llydav", "action": "Create", "type": "Phrase"},
                    {"id": 2, "word": "屡有", "code": "llyd", "action": "Create", "type": "Phrase"},
                ]
                record = self.ticket()
                self.calls.clear()
                response = await self.delete(model=model)
                self.assertIn("草稿", response)
                self.assertEqual([name for name, _args in self.calls[:2]], ["keytao_lookup_by_word", "keytao_list_draft_items"])
                self.assertEqual([item["id"] for item in self.items], [2])
                removals = [args for name, args in self.calls if name == "keytao_batch_remove_draft_items"]
                self.assertEqual(len(removals), 2)
                self.assertTrue(all(args["ids"] == [1] for args in removals))
                self.assertEqual(removals[-1]["batch_id"], "b5-delete")
                self.assertEqual(removals[-1]["expected_content_version"], self.version - 1)
                self.assertTrue(removals[-1]["expected_target_digest"])
                self.assertNotIn("keytao_batch_add_to_draft", [name for name, _args in self.calls])
                self.assertIs(self.store.get_record(self.author), record)

    async def test_live_delete_wins_when_same_pair_also_exists_in_draft(self):
        self.items = [{"id": 1, "word": "屡有", "code": "llydav", "action": "Create", "type": "Phrase"}]
        self.lookup_rows = [{"word": "屡有", "code": "llydav", "type": "Phrase"}]
        record = self.ticket()
        execute = AsyncMock(return_value="dictionary delete staged")
        with patch.object(self.commands, "_execute_confirmed_tool", execute):
            response = await self.delete(model=True)
        self.assertEqual(response, "dictionary delete staged")
        self.assertEqual([item["id"] for item in self.items], [1])
        execute.assert_awaited_once()
        self.tool.assert_awaited_once_with("keytao_lookup_by_word", {"word": "屡有"}, "qq", self.author.actor_id)
        self.assertIs(self.store.get_record(self.author), record)

    async def test_failed_draft_remove_preserves_pending_submit(self):
        self.items = [{"id": 1, "word": "屡有", "code": "llydav", "action": "Create", "type": "Phrase"}]
        self.fail_remove = True
        record = self.ticket()
        response = await self.delete(model=True)
        self.assertIn("offline removal failed", response)
        self.assertEqual(len(self.items), 1)
        self.assertIs(self.store.get_record(self.author), record)

    async def test_replacement_ticket_announces_submit_cancellation(self):
        record = self.ticket()
        response = await self.commands.try_handle_model_command(
            OperationRequest(op="delete", operands=("屡有",), codes=("llydav",)),
            "qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner",
        )
        self.assertIn("取消", response)
        self.assertIn("提交", response)
        self.assertIsNot(self.store.get_record(self.author), record)
        self.assertEqual(self.store.get(self.author).function_name, "keytao_understood_command")
        current = self.store.get_record(self.author)
        self.assertIn({
            "platform": "qq", "platformId": self.author.actor_id,
            "operationKind": "ticket_replacement", "nonce": current.nonce, "previousNonce": record.nonce,
        }, self.commands.current_draft_delivery_claims.get())
        memory = harness.ChatMemoryContext(
            platform="qq", user_id=self.author.actor_id, space_type="group", space_id=self.author.space_id,
            speaker_name="Offline owner",
        )
        delivered = self.chat._prepare_user_facing_reply(response, memory)
        self.assertIn("原待确认提交已取消", delivered)

    def test_delivery_acknowledges_only_recall_and_delete_claims(self):
        receipts = self.commands.current_draft_delivery_claims.get()
        receipts.extend({
            "platform": "qq", "platformId": self.author.actor_id, "operationKind": kind, "fingerprint": kind,
        } for kind in ("pending_read", "dictionary_approval", "ticket_replacement", "draft_write", "draft_submit", "recall", "delete"))
        claims = SimpleNamespace(acknowledge=Mock(return_value=True))
        memory_token = self.commands.current_memory_context.set(None)
        try:
            with patch.object(self.commands, "get_default_draft_mutation_claim_store", return_value=claims):
                self.commands._acknowledge_delivered_draft_mutations()
        finally:
            self.commands.current_memory_context.reset(memory_token)
        self.assertEqual([call.args[2] for call in claims.acknowledge.call_args_list], ["recall", "delete"])

    def test_resolved_mutation_capture_preserves_independent_fact_receipts(self):
        facts = [{
            "platform": "qq", "platformId": self.author.actor_id, "operationKind": kind,
        } for kind in ("pending_read", "dictionary_approval", "ticket_replacement")]
        receipts = self.commands.current_draft_delivery_claims.get()
        for status in ("unresolved", "resolved"):
            with self.subTest(status=status):
                receipts[:] = facts
                claims = SimpleNamespace(get=Mock(return_value={
                    "operationKind": "delete", "status": status, "fingerprint": "delete-receipt",
                }))
                with patch.object(self.commands, "get_default_draft_mutation_claim_store", return_value=claims):
                    self.commands._capture_resolved_mutation_delivery(
                        "keytao_remove_draft_item", "qq", self.author.actor_id, {"success": True},
                    )
                self.assertTrue(all(fact in receipts for fact in facts))

    def test_pending_query_receipt_keeps_only_complete_trusted_facts(self):
        item = {
            "source": "submitted", "batchId": "historical", "batchUrl": "https://offline.invalid/batch/historical",
            "batchStatus": "Submitted", "itemStatus": "Pending", "action": "Create",
            "word": "开团", "code": "khtt", "type": "Phrase",
        }
        receipts = []
        token = self.commands.current_draft_delivery_claims.set(receipts)
        try:
            for incomplete in ({"success": False, "complete": True}, {"success": True, "complete": False}):
                self.commands._capture_successful_draft_write_delivery(
                    "keytao_pending_items_by_words", {**incomplete, "items": [item]}, "qq", self.author.actor_id,
                )
            self.assertEqual(receipts, [])
            self.commands._capture_successful_draft_write_delivery(
                "keytao_pending_items_by_words", {"success": True, "complete": True, "items": [
                    item, {**item, "batchUrl": "https://offline.invalid/forged"},
                ]}, "qq", self.author.actor_id,
            )
        finally:
            self.commands.current_draft_delivery_claims.reset(token)
        self.assertEqual(receipts, [{
            "platform": "qq", "platformId": self.author.actor_id, "operationKind": "pending_read",
            "toolName": "keytao_pending_items_by_words", "pendingItems": [item],
        }])

    def test_mutation_preview_captures_pending_facts_without_successful_write(self):
        item = {
            "source": "draft", "batchId": "historical", "batchUrl": "https://offline.invalid/batch/historical",
            "batchStatus": "Draft", "itemStatus": "Pending", "action": "Create",
            "word": "开团", "code": "khtt", "type": "Phrase",
        }
        receipts = []
        token = self.commands.current_draft_delivery_claims.set(receipts)
        try:
            self.commands._capture_successful_draft_write_delivery(
                "keytao_create_phrase", {"success": False, "requiresConfirmation": True, "pendingItems": [item]},
                "qq", self.author.actor_id,
            )
        finally:
            self.commands.current_draft_delivery_claims.reset(token)
        self.assertEqual(receipts, [{
            "platform": "qq", "platformId": self.author.actor_id, "operationKind": "pending_read",
            "toolName": "keytao_create_phrase", "pendingItems": [item],
        }])

    def test_submit_receipt_without_batch_id_still_requires_success(self):
        receipts = self.commands.current_draft_delivery_claims.get()
        for result in (
            {"success": False},
            {"success": False, "partialWrite": True},
            {"success": True, "requiresConfirmation": True},
        ):
            with self.subTest(result=result):
                self.commands._capture_successful_draft_write_delivery(
                    "keytao_submit_batch", result, "qq", self.author.actor_id,
                )
                self.assertEqual(receipts, [])
        self.commands._capture_successful_draft_write_delivery(
            "keytao_submit_batch", {"success": True}, "qq", self.author.actor_id,
        )
        self.assertEqual(receipts, [{
            "platform": "qq", "platformId": self.author.actor_id, "operationKind": "draft_submit",
            "batchId": "", "toolName": "keytao_submit_batch", "autoApproved": False,
        }])


if __name__ == "__main__":
    unittest.main()
