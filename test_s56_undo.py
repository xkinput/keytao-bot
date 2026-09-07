"""Completed draft writes are reverted as one actor-owned operation."""

import asyncio
import copy
import hashlib
import json
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.utils import completed_draft_undo as undo
from keytao_bot.utils.draft_mutation_store import DraftMutationClaimStore

chat = harness.openai_chat_module


def row(item_id, word, code, action="Create"):
    return {"id": item_id, "word": word, "code": code, "type": "Single", "action": action,
            "oldWord": None, "weight": 10, "remark": None, "needsManualReview": True}


CHAIN = [row(2, "鎗", "qx"), row(3, "强", "qx", "Delete"), row(4, "强", "qxa"),
         row(5, "戕", "qxa", "Delete"), row(6, "戕", "qxai")]


class DraftServer:
    def __init__(self):
        self.items = [row(1, "旧", "jq")]
        self.version = 1
        self.batch_id = "s56-undo-batch"
        self.url = "https://example.invalid/batch/" + self.batch_id
        self.submitted = False
        self.calls = []
        self.fail_delete = False
        self.remove_old_on_shift = False
        self.modify_old_on_shift = False
        self.restore_failure = ""
        self.restore_calls = []

    def get(self, name):
        async def call(**args):
            self.calls.append((name, args))
            result = {"success": True, "batchId": self.batch_id, "batchUrl": self.url, "contentVersion": self.version}
            if name == "keytao_list_draft_items":
                if self.submitted:
                    raise AssertionError("post-submit draft read")
                return {**result, "items": copy.deepcopy(self.items), "count": len(self.items)}
            if name == "keytao_shift_phrase_code":
                if self.remove_old_on_shift:
                    self.items = []
                elif self.modify_old_on_shift:
                    self.items[0]["weight"] = 12
                self.items.extend(copy.deepcopy(CHAIN))
                self.version += 1
                return {**result, "contentVersion": self.version, "successCount": len(CHAIN)}
            if name == "keytao_submit_batch":
                self.submitted = True
                self.version += 1
                return {**result, "contentVersion": self.version}
            if name == "keytao_recall_batch":
                if "expected_content_version" not in args:
                    return {**result, "success": False, "requiresConfirmation": True}
                if not self.submitted or args["expected_content_version"] != self.version:
                    return {"success": False}
                self.submitted = False
                self.version += 1
                return {**result, "contentVersion": self.version}
            if name == "keytao_batch_remove_draft_items":
                targets = [{field: item[field] for field in ("id", "word", "code", "action", "type")}
                           for item in self.items if item["id"] in args["ids"]]
                digest = hashlib.sha256(json.dumps(targets).encode()).hexdigest()
                if not args.get("expected_target_digest"):
                    return {**result, "success": False, "requiresConfirmation": True, "targets": targets, "targetDigest": digest}
                if self.fail_delete:
                    return {**result, "success": False, "message": "fixture timeout"}
                if args["expected_content_version"] != self.version or args["expected_targets"] != targets:
                    return {"success": False}
                self.items = [item for item in self.items if item["id"] not in args["ids"]]
                self.version += 1
                return {**result, "contentVersion": self.version, "successCount": len(targets)}
            raise AssertionError(name)
        return call


class CompletedUndoTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = DraftMutationClaimStore(self.directory.name + "/journal.db")
        self.store_patch = patch.object(undo, "get_default_draft_mutation_claim_store", return_value=self.store)
        self.store_factory = self.store_patch.start()
        self.capture_token = undo.current_operation_capture.set(None)
        self.key = harness.ConversationAddress.group("qq", "s56-group", "s56-undo")
        self.server = DraftServer()
        async def raw_item(item_id):
            return next(({**copy.deepcopy(item), "batchId": self.server.batch_id} for item in self.server.items if item["id"] == item_id), {})
        self.raw_patch = patch.object(undo, "_fetch_raw_item", side_effect=raw_item)
        self.raw_patch.start()
        async def restore_request(identity, batch_id, items, **ticket):
            self.server.restore_calls.append((items, ticket))
            result = {"success": True, "batchId": batch_id, "contentVersion": self.server.version}
            if ticket.get("confirmed") is not True:
                return {**result, "success": False, "requiresConfirmation": True, "warningDigest": "a" * 64}
            if self.server.restore_failure == "rejected":
                return {**result, "success": False, "message": "fixture rejected"}
            self.assertEqual(ticket["expectedContentVersion"], self.server.version)
            for index, item in enumerate(items):
                self.server.items.append({**{field: None for field in undo._FIELDS}, **item, "id": 100 + index})
            self.server.version += 1
            if self.server.restore_failure == "lost_reply":
                raise TimeoutError("fixture lost reply")
            return {**result, "contentVersion": self.server.version, "pullRequestCount": len(items)}
        self.restore_patch = patch.object(undo, "_restore_request", side_effect=restore_request)
        self.restore_patch.start()

    def tearDown(self):
        undo.current_operation_capture.reset(self.capture_token)
        self.store_patch.stop()
        self.raw_patch.stop()
        self.restore_patch.stop()
        self.directory.cleanup()

    async def write(self, submitted=False):
        undo.start_operation_turn(self.key)
        name = "keytao_shift_phrase_code"
        await undo.invoke_with_operation_journal(name, {"confirmed_plan_digest": "d" * 64}, self.server.get(name), self.server.get)
        if submitted:
            name = "keytao_submit_batch"
            await undo.invoke_with_operation_journal(name, {"confirmed": True, "batch_id": self.server.batch_id}, self.server.get(name), self.server.get)

    async def cancel(self, message="取消", reply=None, key=None):
        key = key or self.key
        undo.start_operation_turn(key)
        return await undo.handle_completed_undo(key, message, reply or SimpleNamespace(), self.server.get)

    def test_cancel_without_pending_never_falls_through_to_model(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s56-undo")
            ctx = chat.TurnContext(SimpleNamespace(), SimpleNamespace(), "qq", "s56-undo")
            ctx.conv_key = key
            ctx.normalized_message_text = "取消"
            ctx.generic_command_intent = await chat._classify_message_command_intent("取消", None)
            with patch.object(chat, "_try_handle_draft_management_command", AsyncMock(return_value=None)):
                for stage in chat.STAGES:
                    if stage.__name__ in {"_stage_handle_completed_draft_undo", "_stage_handle_draft_management"}:
                        await stage(ctx)
            self.assertIsNotNone(ctx.response, "completed-write cancellation reached model fallback")
        asyncio.run(run())

    def test_all_four_forms_restore_whole_chain_and_preserve_old_draft(self):
        async def run():
            for message in ("取消", "撤销", "撤回", "回滚"):
                self.server = DraftServer()
                await self.write()
                receipt = await self.cancel(message)
                self.assertEqual(self.server.items, [row(1, "旧", "jq")])
                for expected in ("撤销整笔", "鎗@qx", "强@qxa", "戕@qxai", self.server.url):
                    self.assertIn(expected, receipt)
                writes = [args for name, args in self.server.calls if name == "keytao_batch_remove_draft_items" and args.get("expected_target_digest")]
                self.assertEqual(len(writes), 1)
                self.assertEqual(writes[0]["ids"], [2, 3, 4, 5, 6])
        asyncio.run(run())

    def test_journal_survives_restart_and_repeated_cancel_is_idempotent(self):
        async def run():
            await self.write()
            self.store = DraftMutationClaimStore(self.directory.name + "/journal.db")
            self.store_factory.return_value = self.store
            receipt = await self.cancel()
            calls = len(self.server.calls)
            self.assertEqual(await self.cancel(), receipt)
            self.assertEqual(len(self.server.calls), calls)
        asyncio.run(run())

    def test_expired_operation_has_one_line_native_quote_confirmation(self):
        async def run():
            await self.write()
            scope = undo.conversation_scope(self.key)
            record = self.store.completed_operation(scope)
            record["completedAt"] = time.time() - 601
            self.store.update_completed_operation(scope, record["operationId"], "ready", "ready", record)
            prompt = await self.cancel()
            self.assertEqual(len(prompt.splitlines()), 1)
            self.assertIn("鎗@qx", prompt)
            self.assertIn(self.server.url, prompt)
            self.assertIn("是」或「否", prompt)
            for command in ("是", "否"):
                self.assertIsNone(await self.cancel(command, SimpleNamespace(is_to_bot=False, text=prompt)))
            receipt = await self.cancel("是", SimpleNamespace(is_to_bot=True, text=prompt))
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(self.server.items, [row(1, "旧", "jq")])
        asyncio.run(run())

    def test_intervening_write_prompts_then_removes_only_original_rows(self):
        async def run():
            await self.write()
            extra = row(7, "新", "xb")
            self.server.items.append(extra)
            self.server.version += 1
            prompt = await self.cancel()
            self.assertIn("草稿又有变化", prompt)
            self.assertEqual(len(self.server.items), 7)
            receipt = await self.cancel("确认", SimpleNamespace(is_to_bot=True, text=prompt))
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(self.server.items, [row(1, "旧", "jq"), extra])
        asyncio.run(run())

    def test_quote_no_and_untrusted_yes_never_write(self):
        async def run():
            await self.write()
            undo.start_operation_turn(self.key)
            prompt = await self.cancel()
            self.assertIsNone(await self.cancel("是", SimpleNamespace(is_to_bot=False, text=prompt)))
            self.assertIn("已保留", await self.cancel("否", SimpleNamespace(is_to_bot=True, text=prompt)))
            self.assertEqual(len(self.server.items), 6)
        asyncio.run(run())

    def test_actor_and_conversation_isolation(self):
        async def run():
            await self.write()
            for key in (harness.ConversationAddress.group("qq", "s56-group", "other"),
                        harness.ConversationAddress.group("qq", "other-group", "s56-undo")):
                self.assertIn("没有可核验", await self.cancel(key=key))
            self.assertEqual(len(self.server.items), 6)
        asyncio.run(run())

    def test_submitted_operation_recalls_before_exact_group_delete(self):
        async def run():
            await self.write(submitted=True)
            receipt = await self.cancel()
            self.assertIn("先撤回提审", receipt)
            self.assertFalse(self.server.submitted)
            self.assertEqual(self.server.items, [row(1, "旧", "jq")])
            names = [name for name, args in self.server.calls]
            self.assertLess(names.index("keytao_recall_batch"), names.index("keytao_batch_remove_draft_items"))
        asyncio.run(run())

    def test_failed_delete_has_truthful_receipt_and_durable_write_fence(self):
        async def run():
            await self.write()
            self.server.fail_delete = True
            receipt = await self.cancel()
            self.assertIn("尚未确认全部完成", receipt)
            self.assertNotIn("✅", receipt)
            self.assertTrue(self.store.actor_has_running_undo("qq", "s56-undo"))
            name = "keytao_shift_phrase_code"
            result = await undo.invoke_with_operation_journal(name, {"confirmed_plan_digest": "a" * 64}, self.server.get(name), self.server.get)
            self.assertFalse(result["success"])
            self.server.fail_delete = False
            self.assertIn("撤销整笔", await self.cancel())
            self.assertFalse(self.store.actor_has_running_undo("qq", "s56-undo"))
        asyncio.run(run())

    def test_changed_original_row_restores_original_weight(self):
        async def run():
            self.server.modify_old_on_shift = True
            await self.write()
            receipt = await self.cancel()
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(undo._semantic_rows(self.server.items), undo._semantic_rows([row(1, "旧", "jq")]))
            self.assertEqual(len(self.server.restore_calls), 2)
        asyncio.run(run())

    def test_real_stage_handles_success_before_classifier_and_model(self):
        async def run():
            await self.write()
            undo.start_operation_turn(self.key)
            ctx = chat.TurnContext(SimpleNamespace(), SimpleNamespace(), "qq", "s56-undo")
            ctx.conv_key = self.key
            ctx.normalized_message_text = "取消"
            with patch.object(chat.conversation_state_store, "get_record", return_value=None), patch.object(
                chat.skills_manager, "get_tool_function", side_effect=self.server.get,
            ), patch.object(chat, "_classify_message_command_intent", AsyncMock(side_effect=AssertionError("model classifier"))):
                await chat._stage_handle_completed_draft_undo(ctx)
            self.assertIn("撤销整笔", ctx.response)
            stages = [stage.__name__ for stage in chat.STAGES]
            self.assertLess(stages.index("_stage_handle_completed_draft_undo"), stages.index("_stage_resolve_current_pending_scope"))
        asyncio.run(run())

    def test_submit_only_cancel_reverts_status_and_keeps_rows(self):
        async def run():
            undo.start_operation_turn(self.key)
            name = "keytao_submit_batch"
            await undo.invoke_with_operation_journal(name, {"confirmed": True, "batch_id": self.server.batch_id}, self.server.get(name), self.server.get)
            receipt = await self.cancel()
            self.assertIn("已撤销本次提审", receipt)
            self.assertEqual(self.server.items, [row(1, "旧", "jq")])
            self.assertFalse(any(name == "keytao_batch_remove_draft_items" for name, _ in self.server.calls))
        asyncio.run(run())

    def test_missing_rows_without_actual_receipt_cannot_claim_success(self):
        async def run():
            await self.write()
            self.server.fail_delete = True
            await self.cancel()
            self.server.items = [row(1, "旧", "jq")]
            receipt = await self.cancel()
            self.assertIn("没有本次撤销的成功回执", receipt)
            self.assertNotIn("✅", receipt)
        asyncio.run(run())

    def test_version_changes_after_quote_requires_fresh_confirmation(self):
        async def run():
            await self.write()
            self.server.version += 1
            prompt = await self.cancel()
            self.server.version += 1
            receipt = await self.cancel("是", SimpleNamespace(is_to_bot=True, text=prompt))
            self.assertIn("草稿又有变化", receipt)
            self.assertNotIn("✅", receipt)
            self.assertEqual(len(self.server.items), 6)
        asyncio.run(run())

    def test_questions_negation_and_reported_text_do_not_undo(self):
        async def run():
            await self.write()
            for message in ("取消？", "撤销?", "不要取消", "别撤销", "他说取消", "「回滚」", "`撤销`", "如果取消", "是否撤回"):
                receipt = await self.cancel(message)
                self.assertNotIn("✅", receipt or "")
                self.assertEqual(len(self.server.items), 6, message)
            self.assertFalse(any(name == "keytao_batch_remove_draft_items" for name, _ in self.server.calls))
        asyncio.run(run())

    def test_removed_draft_restores_nullable_weight_seal_and_remark(self):
        async def run():
            original = {**row(1, "旧", "jq"), "weight": None, "remark": "Original review annotation"}
            self.server.items = [copy.deepcopy(original)]
            self.server.remove_old_on_shift = True
            await self.write()
            receipt = await self.cancel()
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(undo._semantic_rows(self.server.items), undo._semantic_rows([original]))
            self.assertNotEqual(self.server.items[0]["id"], original["id"])
            self.assertNotIn("weight", self.server.restore_calls[-1][0][0])
        asyncio.run(run())

    def test_restore_failure_preserves_partial_state_and_retries_only_definite_rejection(self):
        async def run():
            self.server.remove_old_on_shift = True
            await self.write()
            self.server.restore_failure = "rejected"
            receipt = await self.cancel()
            self.assertIn("原草稿恢复尚未全部确认", receipt)
            self.assertNotIn("✅", receipt)
            self.assertTrue(self.store.actor_has_running_undo("qq", "s56-undo"))
            self.server.restore_failure = ""
            receipt = await self.cancel()
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(len(self.server.items), 1)
        asyncio.run(run())

    def test_restore_lost_reply_is_reconciled_without_duplicate_write(self):
        async def run():
            self.server.remove_old_on_shift = True
            await self.write()
            self.server.restore_failure = "lost_reply"
            receipt = await self.cancel()
            self.assertIn("结果不确定", receipt)
            self.assertNotIn("✅", receipt)
            writes = len(self.server.restore_calls)
            receipt = await self.cancel()
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(len(self.server.restore_calls), writes)
            self.assertEqual(len(self.server.items), 1)
        asyncio.run(run())

    def test_absence_cas_accepts_only_empty_version_zero_before_first_write(self):
        async def run():
            identity = {"platform": "qq", "platform_id": "s56-undo"}
            async def absent(**_args):
                return {"success": True, "batchId": None, "contentVersion": 0, "items": []}
            self.assertIsNone(await undo._read_snapshot(absent, identity, "preview-only-uuid"))
            snapshot = await undo._read_snapshot(absent, identity, "preview-only-uuid", allow_absence=True)
            self.assertEqual(snapshot["batchId"], "")
            for version in (True, 1):
                async def invalid(**_args):
                    return {"success": True, "batchId": None, "contentVersion": version, "items": []}
                self.assertIsNone(await undo._read_snapshot(invalid, identity, "preview-only-uuid", allow_absence=True))
            undo.start_operation_turn(self.key)
            self.server.items = []
            self.server.version = 0
            async def first_write(**_args):
                self.server.items = copy.deepcopy(CHAIN)
                self.server.version = 1
                return {"success": True, "batchId": self.server.batch_id, "batchUrl": self.server.url}
            def get_tool(name):
                if name == "keytao_list_draft_items":
                    async def listed(**args):
                        if self.server.version == 0:
                            return await absent(**args)
                        return await self.server.get(name)(**args)
                    return listed
                return self.server.get(name)
            result = await undo.invoke_with_operation_journal(
                "keytao_create_phrase", {"confirmed": True, "batch_id": "preview-only-uuid", "expected_content_version": 0}, first_write, get_tool,
            )
            self.assertTrue(result["success"])
            self.assertIn("撤销整笔", await self.cancel())
            self.assertEqual(self.server.items, [])
        asyncio.run(run())

    def test_pure_draft_removal_undo_restores_removed_rows(self):
        async def run():
            undo.start_operation_turn(self.key)
            name = "keytao_batch_remove_draft_items"
            preview = await self.server.get(name)(ids=[1], batch_id=self.server.batch_id)
            result = await undo.invoke_with_operation_journal(name, {
                "ids": [1], "batch_id": self.server.batch_id,
                "expected_target_digest": preview["targetDigest"],
                "expected_content_version": self.server.version, "expected_targets": preview["targets"],
            }, self.server.get(name), self.server.get)
            self.assertTrue(result["success"])
            self.assertEqual(self.server.items, [])
            receipt = await self.cancel()
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(undo._semantic_rows(self.server.items), undo._semantic_rows([row(1, "旧", "jq")]))
        asyncio.run(run())

    def test_raw_metadata_mismatch_blocks_write_without_guessing_original_state(self):
        async def run():
            undo.start_operation_turn(self.key)
            with patch.object(undo, "_fetch_raw_item", AsyncMock(return_value={**row(1, "旧", "jq"), "batchId": "other-batch"})):
                name = "keytao_shift_phrase_code"
                result = await undo.invoke_with_operation_journal(name, {"confirmed_plan_digest": "a" * 64}, self.server.get(name), self.server.get)
            self.assertFalse(result["success"])
            self.assertEqual(self.server.items, [row(1, "旧", "jq")])
            self.assertFalse(any(name == "keytao_shift_phrase_code" for name, _ in self.server.calls))
        asyncio.run(run())

    def test_submitted_intervening_version_uses_one_line_quote_confirmation(self):
        async def run():
            await self.write(submitted=True)
            self.server.version += 1
            prompt = await self.cancel()
            self.assertIn("提审后又有变化", prompt)
            self.assertEqual(len(prompt.splitlines()), 1)
            self.assertTrue(self.server.submitted)
            receipt = await self.cancel("是", SimpleNamespace(is_to_bot=True, text=prompt))
            self.assertIn("先撤回提审", receipt)
            self.assertEqual(self.server.items, [row(1, "旧", "jq")])
        asyncio.run(run())

    def test_restore_change_delete_keeps_the_original_live_target_binding(self):
        async def run():
            target = {"id": 123, "word": "强", "code": "qx", "type": "Single", "weight": 10,
                      "remark": None, "status": "Finish", "userId": 3}
            fingerprint = hashlib.sha256(json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            original = {**row(1, "强", "qx", "Delete"), "_targetPhraseId": 123, "_targetFingerprint": fingerprint}
            phrase = {**target, "user": {"id": 3}}
            with patch.object(undo.http_client, "keytao_json", AsyncMock(return_value={"phrases": [phrase], "pagination": {"totalPages": 1}})):
                self.assertTrue(await undo._restore_targets_unchanged([original]))
                changed = {**original, "_targetFingerprint": "0" * 64}
                self.assertFalse(await undo._restore_targets_unchanged([changed]))
                self.assertFalse(await undo._restore_targets_unchanged([{**original, "_targetPhraseId": None}]))
        asyncio.run(run())

    def test_recent_write_receipt_capability_is_consumed_by_undo_only(self):
        async def run():
            await self.write()
            undo.start_operation_turn(self.key)
            ctx = chat.TurnContext(SimpleNamespace(), SimpleNamespace(), "qq", "s56-undo")
            ctx.conv_key, ctx.normalized_message_text = self.key, "取消"
            state = harness.PendingToolConfirm("keytao_submit_batch", {"_recent_own_write": True, "batch_id": self.server.batch_id, "_recent_batch_ids": [self.server.batch_id]})
            with patch.object(chat.conversation_state_store, "get_record", return_value=SimpleNamespace(state=state)), patch.object(
                chat.conversation_state_store, "delete",
            ) as delete, patch.object(chat.skills_manager, "get_tool_function", side_effect=self.server.get):
                await chat._stage_handle_completed_draft_undo(ctx)
                self.assertIn("撤销整笔", ctx.response)
                delete.assert_called_once_with(self.key)
                delete.reset_mock()
                ctx.normalized_message_text, ctx.response = "提交", None
                await chat._stage_handle_completed_draft_undo(ctx)
                self.assertIsNone(ctx.response)
                delete.assert_not_called()
        asyncio.run(run())

    def test_inflight_recall_uses_raw_batch_status_when_list_omits_status(self):
        async def run():
            await self.write(submitted=True)
            actual = self.server.get
            draft_tools = harness._draft_tools
            async def lost(**args):
                if "expected_content_version" not in args:
                    return await actual("keytao_recall_batch")(**args)
                self.store.begin("qq", "s56-undo", "recall", {"batchId": self.server.batch_id, "contentVersion": args["expected_content_version"]})
                await actual("keytao_recall_batch")(**args)
                return {"success": False, "uncertain": True, "batchId": self.server.batch_id}
            undo.start_operation_turn(self.key)
            first = await undo.handle_completed_undo(self.key, "取消", SimpleNamespace(), lambda name: lost if name == "keytao_recall_batch" else actual(name))
            self.assertNotIn("✅", first)
            async def raw(item_id):
                item = next(item for item in self.server.items if item["id"] == item_id)
                return {**item, "batchId": self.server.batch_id, "batch": {"id": self.server.batch_id, "status": "Draft"}}
            with patch.object(undo, "_fetch_raw_item", side_effect=raw), patch.object(
                draft_tools, "_draft_mutation_claims", return_value=self.store,
            ), patch.object(draft_tools, "get_bot_token", return_value="fixture-token"), patch.object(
                draft_tools, "keytao_list_draft_items", side_effect=actual("keytao_list_draft_items"),
            ), patch.object(draft_tools.httpx, "AsyncClient", side_effect=AssertionError("unexpected network"), create=True):
                undo.start_operation_turn(self.key)
                receipt = await undo.handle_completed_undo(self.key, "取消", SimpleNamespace(), lambda name: draft_tools.keytao_recall_batch if name == "keytao_recall_batch" else actual(name))
            self.assertIn("撤销整笔", receipt)
            self.assertEqual(self.server.items, [row(1, "旧", "jq")])
            self.assertEqual(self.store.get("qq", "s56-undo")["status"], "resolved")
        asyncio.run(run())

    def test_server_review_marker_refuses_undo_before_recall_or_delete(self):
        async def run():
            for marker in ("--- miao-review:start ---", "--- miao-review:end ---"):
                for submitted in (False, True):
                    with self.subTest(marker=marker, submitted=submitted):
                        self.server = DraftServer()
                        self.server.items[0]["remark"] = "Persisted review " + marker
                        self.server.modify_old_on_shift = True
                        await self.write(submitted=submitted)
                        original = copy.deepcopy(self.server.items)
                        calls_before = len(self.server.calls)
                        receipt = await self.cancel()
                        self.assertNotIn("✅", receipt)
                        self.assertIn("审核记录", receipt)
                        self.assertEqual(self.server.items, original)
                        self.assertEqual(self.server.submitted, submitted)
                        self.assertEqual(len(self.server.calls), calls_before)
                        self.assertEqual(self.server.restore_calls, [])
                        self.assertFalse(self.store.actor_has_running_undo("qq", "s56-undo"))
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
