"""Offline regressions for the September 10 draft receipt and continuation flow."""

import copy
import json
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness.state import _pending_add_word_payload, server_warning_ticket_is_complete
from keytao_bot.plugins import chat_routing as routing


chat = harness.openai_chat_module
commands = harness.chat_commands_module
BATCH = "s62-draft"
ITEM = {"id": 3397, "action": "Create", "word": "加亮", "code": "jslxa", "type": "Phrase"}
OLD_ITEM = {"id": 3396, "action": "Create", "word": "旧词", "code": "jc", "type": "Phrase"}
QUERY_WORDS = ["加量", "加梁", "加亮", "加辆", "价量", "脊索裂畸形", "排列以上词的使用频率"]


class DraftFlowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.key = harness.ConversationAddress.group("qq", "s62-room", "s62-actor")
        self.store = harness.MemoryConversationStateStore()
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id=self.key.actor_id, space_type="group", space_id="s62-room",
        )
        self.calls = []
        self.items = [copy.deepcopy(ITEM)]
        self.version = 3
        self.submitted = False

    async def tool(self, name, args, platform, user_id, **kwargs):
        self.assertEqual((platform, user_id), ("qq", self.key.actor_id))
        self.calls.append((name, copy.deepcopy(args)))
        base = {"success": True, "batchId": BATCH, "contentVersion": self.version,
                "batchUrl": f"https://keytao.rea.ink/batch/{BATCH}"}
        if name == "keytao_submit_batch":
            self.assertEqual(args.get("batch_id"), BATCH)
            if args.get("confirmed"):
                self.assertNotIn("preview_only", args)
                self.assertEqual(args["expected_content_version"], self.version)
                self.assertEqual(args["expected_server_snapshot_digest"], "a" * 64)
                self.assertEqual(args["expected_warning_digest"], "b" * 64)
                self.assertEqual(args["expected_audit_digest"], "c" * 64)
                self.assertFalse(self.submitted, "Submit replayed twice")
                self.submitted = True
                return json.dumps({**base, "status": "Submitted"})
            return json.dumps({**base, "success": False, "requiresConfirmation": True,
                               "snapshotDigest": "a" * 64, "warningDigest": "b" * 64,
                               "auditDigest": "c" * 64, "snapshotItems": self.items, "warnings": []})
        if name == "keytao_batch_add_to_draft":
            self.assertEqual([(i["word"], i["code"]) for i in args["items"]], [("加亮", "jslxa")])
            return json.dumps({**base, "writtenItems": [ITEM], "successCount": 1})
        if name == "keytao_list_draft_items":
            return json.dumps({**base, "items": self.items, "count": len(self.items)})
        if name == "keytao_get_batch_preview":
            return json.dumps({**base, "summary": {"added": len(self.items)}})
        raise AssertionError("Unexpected external tool: " + name)

    @contextmanager
    def runtime(self):
        with ExitStack() as stack:
            for patcher in (
                patch.object(chat, "conversation_state_store", self.store),
                patch.object(chat, "draft_operation_coordinator", harness.DraftOperationCoordinator()),
                patch.object(chat, "call_tool_function", side_effect=self.tool),
                patch.object(chat, "remember_conversation"),
                patch.object(chat, "schedule_memory_compaction"),
                patch.object(chat, "_finish_ai_chat_matcher", AsyncMock()),
                patch.object(chat, "_classify_message_command_intent", AsyncMock(side_effect=AssertionError("No model"))),
                patch.object(chat, "_classify_simple_word_query_intent", AsyncMock(side_effect=AssertionError("No model"))),
            ):
                stack.enter_context(patcher)
            token = commands.current_draft_delivery_claims.set([])
            memory_token = commands.current_memory_context.set(self.memory)
            try:
                yield
            finally:
                commands.current_memory_context.reset(memory_token)
                commands.current_draft_delivery_claims.reset(token)

    def acknowledge_write(self, *, with_delta=True):
        result = {"success": True, "batchId": BATCH, "contentVersion": self.version}
        if with_delta:
            result["writtenItems"] = [copy.deepcopy(ITEM)]
        commands._capture_successful_draft_write_delivery(
            "keytao_batch_add_to_draft", result, "qq", self.key.actor_id,
        )
        commands._acknowledge_delivered_draft_mutations()
        self.assertTrue(self.store.get(self.key).args["_recent_own_write"])

    def context(self, message, *, key=None):
        key = key or self.key
        quote = f"已变更：加亮 → jslxa\n草稿地址：https://keytao.rea.ink/batch/{BATCH}"
        return chat.TurnContext(
            bot=object(), event=object(), platform="qq", user_id=key.actor_id,
            normalized_message_text=message, conv_key=key, space_key=key.space_key,
            memory_context=self.memory, history=[{"role": "assistant", "content": quote}],
            reply_reference=harness.ReplyReferenceInfo(is_reply=True, is_to_bot=True, text=quote),
            command_intent_for=AsyncMock(side_effect=AssertionError("No model")),
        )

    async def continue_after_write(self, message="加入并提交"):
        ctx = self.context(message)
        await chat._stage_resolve_current_pending_scope(ctx)
        self.assertIsNotNone(self.store.get(self.key), "Recent write discarded before submission")
        self.assertFalse(await chat._stage_apply_scoped_pending_intent(ctx))
        await chat._stage_execute_pending_state(ctx)
        return ctx

    async def test_acknowledged_write_then_add_submit_previews_and_submits_once(self):
        with self.runtime():
            self.acknowledge_write()
            ctx = await self.continue_after_write()
            self.assertIn("已提交审核", ctx.response)
        self.assertTrue(self.submitted)
        self.assertEqual([name for name, _ in self.calls], ["keytao_submit_batch"] * 2)

    async def test_add_submit_with_old_items_requires_a_real_confirmation(self):
        self.items.append(copy.deepcopy(OLD_ITEM))
        with self.runtime():
            self.acknowledge_write()
            ctx = await self.continue_after_write()
            self.assertFalse(self.submitted)
            self.assertEqual(len(self.calls), 1)
            ticket = self.store.get(self.key)
            self.assertTrue(server_warning_ticket_is_complete(ticket))
            reply = chat._enforce_advertised_reply_contract(ctx.response, self.key)
            self.assertIn("旧词", reply)
            self.assertIn("确认", reply)
            confirmed = await commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key, allow_intent_model=False,
            )
            self.assertIn("已提交审核", confirmed)
        self.assertTrue(self.submitted)
        self.assertEqual(len(self.calls), 2)

    async def test_missing_delta_cannot_auto_submit_but_plain_submit_keeps_its_scope(self):
        with self.runtime():
            self.acknowledge_write(with_delta=False)
            await self.continue_after_write()
            self.assertFalse(self.submitted)
            self.assertTrue(server_warning_ticket_is_complete(self.store.get(self.key)))
        self.setUp()
        self.items.append(copy.deepcopy(OLD_ITEM))
        with self.runtime():
            self.acknowledge_write(with_delta=False)
            await self.continue_after_write("提交")
        self.assertTrue(self.submitted)

    async def test_old_quote_and_other_actor_never_restore_write_authority(self):
        with self.runtime():
            self.acknowledge_write()
            original = self.store.get_record(self.key)
            other = harness.ConversationAddress.group("qq", "s62-room", "s62-other")
            ctx = self.context("加入并提交", key=other)
            await chat._stage_resolve_current_pending_scope(ctx)
            self.assertTrue(await chat._stage_apply_scoped_pending_intent(ctx))
            self.assertIs(self.store.get_record(self.key), original)
            self.store.delete(self.key)
            ctx = self.context("加入并提交")
            await chat._stage_resolve_current_pending_scope(ctx)
            self.assertTrue(await chat._stage_apply_scoped_pending_intent(ctx))
        self.assertEqual(self.calls, [])

    async def test_incomplete_receipt_cannot_authorize_automatic_submission(self):
        with self.runtime():
            commands._capture_successful_draft_write_delivery(
                "keytao_batch_add_to_draft",
                {"success": True, "batchId": BATCH, "contentVersion": self.version,
                 "writtenItems": [ITEM], "receiptItemsUnavailable": True},
                "qq", self.key.actor_id,
            )
            commands._acknowledge_delivered_draft_mutations()
            await self.continue_after_write()
            self.assertTrue(server_warning_ticket_is_complete(self.store.get(self.key)))
        self.assertFalse(self.submitted)
        self.assertEqual(len(self.calls), 1)

    async def test_missing_or_invalid_write_version_requires_confirmation(self):
        for version in (None, True, "3", -1):
            with self.subTest(version=version):
                self.setUp()
                receipt = {"success": True, "batchId": BATCH, "writtenItems": [ITEM]}
                if version is not None:
                    receipt["contentVersion"] = version
                with self.runtime():
                    commands._capture_successful_draft_write_delivery(
                        "keytao_batch_add_to_draft", receipt, "qq", self.key.actor_id,
                    )
                    commands._acknowledge_delivered_draft_mutations()
                    await self.continue_after_write()
                    self.assertFalse(self.submitted)
                    self.assertTrue(server_warning_ticket_is_complete(self.store.get(self.key)))
                self.assertEqual(len(self.calls), 1)

    async def test_recent_write_with_multiple_batches_does_not_choose_one(self):
        with self.runtime():
            for batch in (BATCH, "s62-other-batch"):
                commands._capture_successful_draft_write_delivery(
                    "keytao_batch_add_to_draft", {"success": True, "batchId": batch, "writtenItems": [ITEM]},
                    "qq", self.key.actor_id,
                )
            commands._acknowledge_delivered_draft_mutations()
            ctx = self.context("加入并提交")
            await chat._stage_resolve_current_pending_scope(ctx)
            self.assertIn("多个草稿批次", ctx.scoped_pending_response)
            self.assertIsNone(self.store.get(self.key))
        self.assertEqual(self.calls, [])

    async def test_acknowledgement_cannot_use_another_actors_receipt(self):
        with self.runtime():
            commands._capture_successful_draft_write_delivery(
                "keytao_batch_add_to_draft", {"success": True, "batchId": BATCH, "writtenItems": [OLD_ITEM]},
                "qq", "s62-other",
            )
            commands._acknowledge_delivered_draft_mutations()
            self.assertIsNone(self.store.get(self.key))
        self.assertEqual(self.calls, [])

    async def test_edit_after_write_requires_confirmation_even_with_the_same_word_and_code(self):
        with self.runtime():
            self.acknowledge_write()
            self.items[0]["weight"] = 200
            self.version += 1
            await self.continue_after_write()
            self.assertFalse(self.submitted, "A later edit must not inherit the old write authorization")
            self.assertTrue(server_warning_ticket_is_complete(self.store.get(self.key)))
        self.assertEqual(len(self.calls), 1)

    async def test_multiple_write_receipts_cannot_hide_an_interleaved_edit(self):
        with self.runtime():
            for version, item in ((self.version, ITEM), (self.version + 2, OLD_ITEM)):
                commands._capture_successful_draft_write_delivery(
                    "keytao_batch_add_to_draft",
                    {"success": True, "batchId": BATCH, "contentVersion": version,
                     "writtenItems": [item]},
                    "qq", self.key.actor_id,
                )
            commands._acknowledge_delivered_draft_mutations()
            self.items = [dict(ITEM, weight=200), copy.deepcopy(OLD_ITEM)]
            self.version += 2
            await self.continue_after_write()
            self.assertFalse(self.submitted, "Multiple receipts cannot authorize an intervening edit")
            self.assertTrue(server_warning_ticket_is_complete(self.store.get(self.key)))
        self.assertEqual(len(self.calls), 1)

    async def test_selected_receipt_excludes_original_query_scope(self):
        items, scopes = [], []
        for word in ("加梁", "加亮"):
            reviewed = harness.PendingAddWord(
                word=word, recommended_code="jslxa", candidates=[("jslxa", False)],
                server_candidates=[("jslxa", False)], needs_manual_review=True,
                manual_review_reason="Fixture review",
            )
            items.append({"action": "Create", "word": word, "code": "jslxa", "type": "Phrase"})
            scopes.append({"word": word, "candidates": [["jslxa", False]], "occupiedWords": {},
                           "orderingAssessments": [], "reviewedState": _pending_add_word_payload(reviewed)})
        pending = harness.PendingToolConfirm(function_name="keytao_batch_add_to_draft", args={
            "items": items, "_candidate_scopes": scopes, "_reviewed_multi_word": True,
            "_query_words": QUERY_WORDS,
        })
        selected, _, error = routing._resolve_multi_word_pending_candidate_selection(pending, "加亮 jslxa")
        self.assertIsNone(error)
        with self.runtime():
            response = await commands._execute_confirmed_tool(selected, "qq", self.key.actor_id, self.key)
            delivered = chat._prepare_user_facing_reply(response, self.memory)
        self.assertIn("已变更：加亮 → jslxa", delivered)
        self.assertNotIn("未新增变更", delivered)
        self.assertNotIn("排列以上词的使用频率", delivered)

    async def test_successful_draft_view_survives_delivery_without_a_ticket(self):
        with self.runtime():
            response = await commands._try_handle_draft_view_command(
                harness.MessageCommandIntent(intent="draft_view", confidence=1.0), "qq", self.key.actor_id,
            )
            token = chat._current_turn_message.set("查看草稿")
            try:
                delivered = chat._prepare_user_facing_reply(response, self.memory)
            finally:
                chat._current_turn_message.reset(token)
        self.assertIn("加亮", delivered)
        self.assertIn("jslxa", delivered)
        self.assertIn(f"/batch/{BATCH}", delivered)
        self.assertIsNone(self.store.get(self.key))
        self.assertEqual([name for name, _ in self.calls], ["keytao_list_draft_items", "keytao_get_batch_preview"])

    async def test_targetless_add_finishes_after_recovery_without_word_or_main_model(self):
        ctx = self.context("加入草稿")
        delivered = []
        async def finish(_bot, _event, _actor, memory, response, _segment):
            delivered.append(chat._prepare_user_facing_reply(response, memory))
        with self.runtime(), patch.object(chat, "_finish_ai_chat_response", side_effect=finish):
            handled = await chat._stage_handle_simple_word_query(ctx)
        self.assertTrue(handled, "Targetless add must finish before augmentation can classify it again")
        self.assertIn("查看草稿", delivered[0])
        self.assertIn("未写入", delivered[0])
        self.assertNotIn("服务没有完成", delivered[0])
        self.assertIsNone(self.store.get(self.key))
        self.assertEqual(self.calls, [])

    async def test_history_recovery_precedes_targetless_add_refusal(self):
        with self.runtime(), patch.object(chat, "_try_recover_reviewed_add_from_history", AsyncMock(return_value="Fixture recovered candidate")):
            ctx = self.context("加入草稿")
            self.assertFalse(await chat._stage_handle_simple_word_query(ctx))
            self.assertEqual(ctx.response, "Fixture recovered candidate")


if __name__ == "__main__":
    unittest.main()
