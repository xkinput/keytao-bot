"""Offline regressions for bare 撤销/取消 routing, QQ undo quote confirmation, and recall failure recovery."""

from contextlib import ExitStack
import copy
import json
import socket
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from keytao_bot.utils.url_format import render_platform_public_links
from tests.helpers import (
    DraftMutationClaimStore,
    DraftServer,
    FakeClient,
    PendingToolConfirm,
    Replay,
    completed_draft_undo,
    harness,
    raw_operation,
    s56_undo_chat,
    s56_undo_row,
)


class RecallAndUndoRegressionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = DraftMutationClaimStore(self.directory.name + "/journal.db")
        self.server = DraftServer()
        self.server.url = "https://keytao.vercel.app/batch/d8722241-28b9-49b8-b315-d4d956e1b61a"
        self.key = harness.ConversationAddress.group("qq", "prod-replay", "evo-user")
        self.store_patch = patch.object(
            completed_draft_undo,
            "get_default_draft_mutation_claim_store",
            return_value=self.store,
        )
        self.chat_store_patch = patch.object(
            s56_undo_chat,
            "get_default_draft_mutation_claim_store",
            return_value=self.store,
        )
        self.store_patch.start()
        self.chat_store_patch.start()
        self.capture_token = completed_draft_undo.current_operation_capture.set(None)

        async def fetch_raw(item_id):
            row = next(item for item in self.server.items if item["id"] == item_id)
            return {
                **row,
                "batchId": self.server.batch_id,
                "batch": {
                    "id": self.server.batch_id,
                    "status": "Submitted" if self.server.submitted else "Draft",
                },
            }

        self.raw_patch = patch.object(completed_draft_undo, "_fetch_raw_item", side_effect=fetch_raw)
        self.raw_patch.start()

    def tearDown(self):
        completed_draft_undo.current_operation_capture.reset(self.capture_token)
        self.raw_patch.stop()
        self.chat_store_patch.stop()
        self.store_patch.stop()
        self.directory.cleanup()

    async def _record_completed_write(self, *, submitted: bool = True, age_seconds: float = 5.0):
        completed_draft_undo.start_operation_turn(self.key)
        await completed_draft_undo.invoke_with_operation_journal(
            "keytao_shift_phrase_code",
            {"confirmed_plan_digest": "d" * 64},
            self.server.get("keytao_shift_phrase_code"),
            self.server.get,
        )
        if submitted:
            await completed_draft_undo.invoke_with_operation_journal(
                "keytao_submit_batch",
                {"confirmed": True, "batch_id": self.server.batch_id},
                self.server.get("keytao_submit_batch"),
                self.server.get,
            )
        if age_seconds > 0:
            scope = completed_draft_undo.conversation_scope(self.key)
            record = self.store.completed_operation(scope)
            record["completedAt"] = time.time() - age_seconds
            self.store.update_completed_operation(scope, record["operationId"], "ready", "ready", record)

    async def test_bare_chexiao_after_recent_submit_reaches_completed_undo_not_stale_cancel(self):
        await self._record_completed_write(submitted=True, age_seconds=8.0)
        replay = Replay({"sender": "evo-user"}, harness, FakeClient)
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
            stack.enter_context(patch.object(replay.chat.skills_manager, "get_tool_function", side_effect=self.server.get))
            main = stack.enter_context(patch.object(replay.chat, "get_ai_response_core", AsyncMock()))
            completed_draft_undo.start_operation_turn(self.key)
            await replay.turn(
                {"message": "撤销", "to_me": True},
                raw_operation(op="draft_recall", operands=[], codes=[], frame="direct"),
            )
            self.assertNotEqual(replay.last_response, "当前没有待取消的操作。")
            self.assertIn("已先撤回提审", replay.last_response)
            self.assertFalse(self.server.submitted)
            self.assertEqual(self.server.items, [s56_undo_row(1, "旧", "jq")])
            replay.client.create.assert_not_awaited()
            main.assert_not_awaited()

    async def test_expired_undo_confirmation_and_qq_rendered_quote_reply_succeed_through_full_stages(self):
        await self._record_completed_write(submitted=True, age_seconds=4400.0)
        replay = Replay({"sender": "evo-user"}, harness, FakeClient)
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
            stack.enter_context(patch.object(replay.chat.skills_manager, "get_tool_function", side_effect=self.server.get))
            main = stack.enter_context(patch.object(replay.chat, "get_ai_response_core", AsyncMock()))

            completed_draft_undo.start_operation_turn(self.key)
            await replay.turn(
                {"message": "撤销", "to_me": True},
                raw_operation(op="draft_recall", operands=[], codes=[], frame="direct"),
            )
            raw_prompt = replay.last_response
            self.assertIn("是否整笔撤销", raw_prompt)
            self.assertEqual(len(raw_prompt.splitlines()), 1)

            # Simulate QQ platform delivery: vercel.app -> keytao.rea.ink + URL boundary spacing.
            qq_delivered_prompt = render_platform_public_links(raw_prompt, "qq")
            self.assertIn("https://keytao.rea.ink/batch/d8722241-28b9-49b8-b315-d4d956e1b61a", qq_delivered_prompt)

            completed_draft_undo.start_operation_turn(self.key)
            await replay.turn(
                {"message": "是", "to_me": True, "quote_target": qq_delivered_prompt},
                raw_operation(op="confirm", operands=[], codes=[], frame="direct"),
            )
            self.assertIn("已先撤回提审", replay.last_response)
            self.assertFalse(self.server.submitted)
            self.assertEqual(self.server.items, [s56_undo_row(1, "旧", "jq")])
            replay.client.create.assert_not_awaited()
            main.assert_not_awaited()

    async def test_immediate_quxiao_with_recent_own_write_ticket_actually_undoes_write(self):
        await self._record_completed_write(submitted=False, age_seconds=5.0)
        replay = Replay({"sender": "evo-user"}, harness, FakeClient)
        recent_ticket = PendingToolConfirm(
            "keytao_submit_batch",
            {
                "_recent_own_write": True,
                "batch_id": self.server.batch_id,
                "_recent_batch_ids": [self.server.batch_id],
            },
        )
        self.assertTrue(replay.store.set(self.key, recent_ticket))
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
            stack.enter_context(patch.object(replay.chat.skills_manager, "get_tool_function", side_effect=self.server.get))
            completed_draft_undo.start_operation_turn(self.key)
            await replay.turn(
                {"message": "取消", "to_me": True},
                raw_operation(op="cancel", operands=[], codes=[], frame="direct"),
            )
            self.assertIn("撤销整笔", replay.last_response)
            self.assertEqual(self.server.items, [s56_undo_row(1, "旧", "jq")])
            self.assertIsNone(replay.store.get_record(self.key))

    async def test_stale_quxiao_still_reports_no_pending_cancel_when_old_write_expired(self):
        await self._record_completed_write(submitted=True, age_seconds=4400.0)
        replay = Replay({"sender": "evo-user"}, harness, FakeClient)
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
            completed_draft_undo.start_operation_turn(self.key)
            await replay.turn(
                {"message": "取消", "to_me": True},
                raw_operation(op="cancel", operands=[], codes=[], frame="direct"),
            )
            self.assertEqual(replay.last_response, "当前没有待取消的操作。")

    async def test_bare_chexiao_without_completed_write_falls_through_to_draft_recall(self):
        replay = Replay({"sender": "evo-user"}, harness, FakeClient)
        recall_preview = {
            "success": False,
            "requiresConfirmation": True,
            "confirmationKind": "recallBatch",
            "batchId": self.server.batch_id,
            "contentVersion": 2,
            "status": "Submitted",
            "pullRequestCount": 3,
            "batchUrl": self.server.url,
            "message": "即将撤回这个已提交批次并恢复为草稿",
        }
        original_tool = replay.tool

        async def patched_tool(name, args, platform, user_id, **kwargs):
            if name == "keytao_recall_batch":
                replay.calls.append((name, copy.deepcopy(args)))
                return json.dumps(recall_preview, ensure_ascii=False)
            return await original_tool(name, args, platform, user_id, **kwargs)

        replay.tool = patched_tool
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
            completed_draft_undo.start_operation_turn(self.key)
            await replay.turn(
                {"message": "撤销", "to_me": True},
                raw_operation(op="draft_recall", operands=[], codes=[], frame="direct"),
            )
            self.assertNotEqual(replay.last_response, "当前没有待取消的操作。")
            self.assertIn("撤回已提交批次", replay.last_response)
            self.assertTrue(any(name == "keytao_recall_batch" for name, _ in replay.calls))

    async def test_definite_recall_rejection_does_not_permanently_lock_actor_undo_state(self):
        await self._record_completed_write(submitted=True, age_seconds=5.0)
        actual = self.server.get

        def get_tool(name):
            if name != "keytao_recall_batch":
                return actual(name)

            async def reject_recall(**args):
                if "expected_content_version" not in args:
                    return await actual(name)(**args)
                return {
                    "success": False,
                    "staleConfirmation": True,
                    "batchId": self.server.batch_id,
                    "message": "批次已开始逐词审核，不能整批撤销",
                }

            return reject_recall

        completed_draft_undo.start_operation_turn(self.key)
        reply = await completed_draft_undo.handle_completed_undo(
            self.key, "撤销", SimpleNamespace(), get_tool,
        )
        self.assertIn("批次已开始逐词审核", reply)
        self.assertFalse(self.store.actor_has_running_undo("qq", "evo-user"))
        scope = completed_draft_undo.conversation_scope(self.key)
        record = self.store.completed_operation(scope)
        self.assertEqual(record["status"], "ready")
        self.assertNotIn("recallStarted", record)

    def test_canonical_draft_management_matches_ganggang_submitted_batch(self):
        from keytao_bot.plugins.chat_routing import _canonical_draft_management_command

        intent = _canonical_draft_management_command("撤销刚刚提交的批次")
        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent, "draft_recall")


if __name__ == "__main__":
    unittest.main()
