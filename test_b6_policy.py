"""Offline acceptance checks for the B6 draft policies."""

import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

from keytao_bot.utils.draft_reminder import DAY_SECONDS, SHANGHAI, DraftReminderStore


class B6ReminderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.db_path = str(Path(self.directory.name) / "reminders.db")
        self.store = DraftReminderStore(self.db_path)
        self.now = datetime(2026, 10, 27, 10, tzinfo=SHANGHAI).timestamp()
        self.flag = patch.dict(os.environ, {"KEYTAO_DRAFT_REMINDER": "1"})
        self.flag.start()
        self.addCleanup(self.flag.stop)
        self.read = AsyncMock(side_effect=self.snapshot)
        self.send = AsyncMock(return_value="reminder-message")

    def write(self, *, batch="b6-draft", version=1, written=None, actor="owner", group="group", submitted=False, items=None):
        self.store.record_write(
            json.dumps(["qq", "group", group, actor]), "qq", actor,
            {
                "operationId": f"{batch}-{version}",
                "completedAt": self.now - DAY_SECONDS if written is None else written,
                "batchId": batch, "batchUrl": f"https://offline.invalid/draft/{batch}",
                "submitted": submitted,
                "after": {"batchId": batch, "contentVersion": version,
                          "items": [{"id": 1, "word": "旧词"}] if items is None else items},
            },
        )

    @staticmethod
    async def snapshot(candidate):
        return {
            "success": True, "batchId": candidate["batch_id"], "status": "Draft",
            "contentVersion": candidate["content_version"], "items": [{"id": 1}],
            "batchUrl": candidate["batch_url"],
        }

    async def scan(self, now=None, store=None):
        return await (store or self.store).scan(self.read, self.send, now=self.now if now is None else now)

    async def test_24_hour_boundary_and_exact_receipt(self):
        self.write(written=self.now - DAY_SECONDS + 1)
        self.assertEqual(await self.scan(), 0)
        self.read.assert_not_awaited()
        self.assertEqual(await self.scan(self.now + 1), 1)
        candidate, text = self.send.await_args.args
        self.assertEqual((candidate["actor"], candidate["group_id"]), ("owner", "group"))
        self.assertEqual(
            text,
            "你的草稿还没有提交（1 条，最后修改于 2026-10-26 10:00）。\n"
            "草稿地址：https://offline.invalid/draft/b6-draft\n回复「提交」提交，或「清空草稿」放弃。",
        )
        receipt = self.store.resolve_receipt("qq", "group", "reminder-message")
        self.assertEqual((receipt["actor"], receipt["batch_id"]), ("owner", "b6-draft"))
        self.assertIsNone(self.store.resolve_receipt("qq", "other-group", "reminder-message"))

    async def test_quiet_hours_defer_until_08(self):
        base = datetime(2026, 10, 27, tzinfo=SHANGHAI)
        self.write(written=(base - timedelta(days=2)).timestamp())
        for hour, minute in ((23, 0), (0, 0), (7, 59)):
            self.assertEqual(await self.scan(base.replace(hour=hour, minute=minute).timestamp()), 0)
        self.read.assert_not_awaited()
        self.assertEqual(await self.scan(base.replace(hour=8).timestamp()), 1)

    async def test_one_reminder_per_version_even_days_later(self):
        self.write()
        self.assertEqual(await self.scan(), 1)
        self.assertEqual(await self.scan(self.now + 2 * DAY_SECONDS), 0)
        self.assertEqual(self.send.await_count, 1)

    async def test_batch_cap_is_three_distinct_versions(self):
        for version in range(1, 5):
            current = self.now + (version - 1) * 2 * DAY_SECONDS
            self.write(version=version, written=current - DAY_SECONDS)
            self.assertEqual(await self.scan(current), int(version <= 3))
        self.assertEqual(self.send.await_count, 3)

    async def test_batch_caps_are_shared_across_linked_platform_accounts(self):
        for version in range(1, 5):
            current = self.now + (version - 1) * 2 * DAY_SECONDS
            platform = "qq" if version % 2 else "telegram"
            actor = "qq-owner" if platform == "qq" else "telegram-owner"
            self.store.record_write(
                json.dumps([platform, "group", "latest-group", actor]), platform, actor,
                {"operationId": f"write-{version}", "completedAt": current - DAY_SECONDS,
                 "batchId": "shared-batch", "batchUrl": "https://offline.invalid/shared-batch",
                 "after": {"batchId": "shared-batch", "contentVersion": version, "items": [{"id": 1}]}},
            )
            self.assertEqual(await self.scan(current), int(version <= 3))
        self.assertEqual(self.send.await_count, 3)

    async def test_same_version_and_latest_group_are_global_per_batch(self):
        self.write(written=self.now - 2 * DAY_SECONDS)
        self.store.record_write(
            '["telegram","group","latest-group","telegram-owner"]', "telegram", "telegram-owner",
            {"completedAt": self.now - DAY_SECONDS, "batchId": "b6-draft",
             "batchUrl": "https://offline.invalid/b6-draft",
             "after": {"batchId": "b6-draft", "contentVersion": 1, "items": [{"id": 1}]}},
        )
        self.assertEqual(await self.scan(), 1)
        candidate = self.send.await_args.args[0]
        self.assertEqual((candidate["platform"], candidate["group_id"]), ("telegram", "latest-group"))
        self.assertEqual(await self.scan(self.now + 2 * DAY_SECONDS), 0)
        self.assertEqual(self.send.await_count, 1)

    async def test_repeated_reminder_needs_full_24_hours(self):
        self.write(written=self.now - 2 * DAY_SECONDS)
        self.assertEqual(await self.scan(), 1)
        self.write(version=2, written=self.now - DAY_SECONDS)
        self.assertEqual(await self.scan(self.now + DAY_SECONDS - 1), 0)
        self.assertEqual(await self.scan(self.now + DAY_SECONDS), 1)

    async def test_batch_submitted_empty_missing_or_version_changed_meantime_skips(self):
        self.write()
        for overrides in (
            {"status": "Pending"}, {"items": []}, {"success": False},
            {"batchId": "another"}, {"contentVersion": 2}, {"count": 2},
            {"status": None}, {"itemsTruncationNotice": "truncated"},
        ):
            with self.subTest(overrides=overrides):
                draft = await self.snapshot({
                    "batch_id": "b6-draft", "content_version": 1,
                    "batch_url": "https://offline.invalid/draft/b6-draft",
                })
                self.read.side_effect = None
                self.read.return_value = {**draft, **overrides}
                self.assertEqual(await self.scan(), 0)
        self.send.assert_not_awaited()

    async def test_restart_does_not_duplicate_success_or_uncertain_delivery(self):
        self.write()
        self.assertEqual(await self.scan(), 1)
        self.assertEqual(await self.scan(self.now + DAY_SECONDS, DraftReminderStore(self.db_path)), 0)
        self.write(batch="uncertain")
        self.send.side_effect = TimeoutError("uncertain send")
        self.assertEqual(await self.scan(), 0)
        self.assertEqual(await self.scan(self.now + DAY_SECONDS, DraftReminderStore(self.db_path)), 0)
        self.assertEqual(self.send.await_count, 2)

    async def test_send_intent_is_committed_before_delivery_and_crash_is_not_retried(self):
        self.write()

        async def crash(candidate, text):
            with closing(sqlite3.connect(self.db_path)) as connection:
                self.assertEqual(connection.execute("SELECT status FROM draft_reminder_attempt").fetchone(), ("intent",))
            raise KeyboardInterrupt("process stopped after send started")

        self.send.side_effect = crash
        with self.assertRaises(KeyboardInterrupt):
            await self.scan()
        self.assertEqual(await self.scan(self.now + DAY_SECONDS, DraftReminderStore(self.db_path)), 0)
        self.assertEqual(self.send.await_count, 1)

    async def test_flag_unset_or_other_values_do_nothing(self):
        self.write()
        for value in (None, "0", "true", " 1", "1 "):
            with self.subTest(value=value), patch.dict(os.environ):
                if value is None:
                    os.environ.pop("KEYTAO_DRAFT_REMINDER", None)
                else:
                    os.environ["KEYTAO_DRAFT_REMINDER"] = value
                self.assertEqual(await self.scan(), 0)
        self.read.assert_not_awaited()
        self.send.assert_not_awaited()

    async def test_twenty_check_bound_is_fair_for_unavailable_batches(self):
        for number in range(25):
            self.write(batch=f"batch-{number:02}")
        checked = []

        async def unavailable(candidate):
            checked.append(candidate["batch_id"])
            return {"success": False}

        self.read.side_effect = unavailable
        self.assertEqual(await self.scan(), 0)
        self.assertEqual(len(checked), 20)
        self.assertEqual(await self.scan(self.now + 3600), 0)
        self.assertEqual(len(checked), 40)
        self.assertEqual(len(set(checked)), 25)

    async def test_latest_write_controls_group_and_submission_closes_candidate(self):
        self.write(group="first", written=self.now - 3 * DAY_SECONDS)
        self.write(group="latest", version=2, written=self.now - DAY_SECONDS)
        self.assertEqual(await self.scan(), 1)
        self.assertEqual(self.send.await_args.args[0]["group_id"], "latest")
        self.write(version=3, submitted=True, written=self.now + 1)
        self.assertEqual(await self.scan(self.now + 2 * DAY_SECONDS), 0)
        self.assertEqual(self.send.await_count, 1)

    async def test_concurrent_write_during_preview_cannot_send_stale_reminder(self):
        self.write()

        async def changed(candidate):
            self.write(version=2, written=self.now)
            return await self.snapshot(candidate)

        self.read.side_effect = changed
        self.assertEqual(await self.scan(), 0)
        self.send.assert_not_awaited()


class B6SubmitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import test_state_machine as harness
        from keytao_bot.utils import http_client
        self.chat = harness.openai_chat_module
        self.http = http_client
        self.new = {"id": 2, "action": "Create", "word": "屡有", "code": "llydav", "type": "Phrase"}
        self.old = {"id": 1, "action": "Create", "word": "驴友", "code": "llyd", "type": "Phrase",
                    "needsManualReview": True}
        self.preview = {
            "success": False, "requiresConfirmation": True, "batchId": "b6-draft",
            "contentVersion": 8, "snapshotDigest": "a" * 64,
            "warningDigest": "b" * 64, "auditDigest": "c" * 64,
            "snapshotItems": [self.old, self.new], "warnings": [],
            "batchUrl": "https://offline.invalid/batch/b6-draft",
            "autoReview": {"issues": ["需要管理员审核"], "structuralPreReviewLocked": True},
        }
        self.batch = {
            "id": "b6-draft", "contentVersion": 8, "status": "Draft", "creatorId": 17,
            "pullRequests": [{**item, "userId": 17} for item in self.preview["snapshotItems"]],
        }
        self.tool = AsyncMock(side_effect=self.invoke)
        self.details = AsyncMock(return_value={"batch": self.batch})
        for context in (
            patch.object(self.chat, "call_tool_function", self.tool),
            patch.object(self.http, "keytao_json", self.details),
            patch.dict(os.environ, {"KEYTAO_COMMUNITY_VOTE": "0"}),
        ):
            context.start()
            self.addCleanup(context.stop)

    async def invoke(self, name, arguments, platform, user_id):
        self.assertEqual((name, platform, user_id), ("keytao_submit_batch", "qq", "abang"))
        if arguments.get("confirmed"):
            self.assertEqual(arguments["batch_id"], "b6-draft")
            self.assertEqual(arguments["expected_content_version"], 8)
            self.assertEqual(arguments["expected_server_snapshot_digest"], "a" * 64)
            self.assertEqual(arguments["expected_warning_digest"], "b" * 64)
            self.assertEqual(arguments["expected_audit_digest"], "c" * 64)
            self.assertNotIn("preview_only", arguments)
            return json.dumps({"success": True, "batchId": "b6-draft", "contentVersion": 8,
                               "batchUrl": self.preview["batchUrl"], "snapshotItems": self.preview["snapshotItems"]})
        return json.dumps(self.preview)

    async def submit(self, **kwargs):
        return await self.chat._perform_submit_current_draft(
            "qq", "abang", auto_confirm=True, authorized_items=[self.new], **kwargs,
        )

    async def test_all_own_leftovers_submit_once_and_receipt_names_every_item(self):
        result = await self.submit()
        self.assertTrue(result.success)
        self.assertIsNone(result.pending_state)
        self.assertEqual(sum(call.args[1].get("confirmed") is True for call in self.tool.await_args_list), 1)
        self.assertIn("驴友 → llyd（本轮之前）", result.text)
        self.assertIn("屡有 → llydav", result.text)
        self.assertNotIn("屡有 → llydav（本轮之前）", result.text)
        self.assertNotIn("本轮之外的内容", result.text)
        self.details.assert_awaited_once_with("GET", "/api/batches/b6-draft", timeout=10.0)

    async def test_foreign_unknown_or_unbound_rows_keep_scope_confirmation(self):
        import copy
        original = copy.deepcopy(self.batch)
        for scenario in ("foreign", "unknown", "version", "row_mismatch", "duplicate", "missing", "unavailable"):
            with self.subTest(scenario=scenario):
                batch = copy.deepcopy(original)
                if scenario == "foreign":
                    batch["pullRequests"][0]["userId"] = 99
                elif scenario == "unknown":
                    batch["pullRequests"][0].pop("userId")
                elif scenario == "version":
                    batch["contentVersion"] = 9
                elif scenario == "row_mismatch":
                    batch["pullRequests"][0]["word"] = "别人写的"
                elif scenario == "duplicate":
                    batch["pullRequests"][1] = batch["pullRequests"][0]
                elif scenario == "missing":
                    batch["pullRequests"].pop()
                self.details.return_value = {"batch": batch} if scenario != "unavailable" else {}
                self.tool.reset_mock()
                result = await self.submit()
                self.assertFalse(result.success)
                self.assertIsNotNone(result.pending_state)
                self.assertIn("本轮之外的内容", result.text)
                self.assertIn("确认", result.text)
                self.assertEqual(self.tool.await_count, 1)

    async def test_single_round_auto_confirm_does_not_need_authorship_lookup(self):
        self.preview["snapshotItems"] = [self.new]
        result = await self.submit()
        self.assertTrue(result.success)
        self.assertEqual(self.tool.await_count, 2)
        self.details.assert_not_awaited()

    async def test_computed_display_weight_does_not_change_authorship(self):
        self.old["weight"] = 0
        self.batch["pullRequests"][0]["weight"] = 100
        self.assertTrue((await self.submit()).success)

    async def test_stale_authorized_version_or_missing_ticket_does_not_expand_scope(self):
        result = await self.submit(authorized_content_version=7)
        self.assertIsNotNone(result.pending_state)
        self.details.assert_not_awaited()
        self.preview.pop("auditDigest")
        result = await self.submit()
        self.assertFalse(result.success)
        self.details.assert_not_awaited()

    async def test_community_receipt_preserves_all_items_through_delivery(self):
        from types import SimpleNamespace
        from keytao_bot.utils import community_vote
        token = community_vote._pending_receipts.set([])
        self.addCleanup(community_vote._pending_receipts.reset, token)
        with patch.dict(os.environ, {"KEYTAO_COMMUNITY_VOTE": "1"}):
            result = await self.submit()
            delivered = community_vote.prepare_delivery(result.text, SimpleNamespace(platform="qq"))
        self.assertIn("驴友 → llyd（本轮之前）", delivered)
        self.assertIn("屡有 → llydav", delivered)
        self.assertIn("当前投票：0/3", delivered)


class B6ReminderIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        import test_state_machine as harness
        self.harness = harness
        self.chat = harness.openai_chat_module

    async def test_hourly_hook_does_nothing_with_flag_unset(self):
        with patch.dict(os.environ), patch.object(self.chat, "_draft_reminder_store") as store:
            os.environ.pop("KEYTAO_DRAFT_REMINDER", None)
            self.assertEqual(await self.chat._scan_draft_reminders_once(), 0)
            store.assert_not_called()

    async def test_latest_list_is_actor_scoped_without_rejected_batch_selector(self):
        listed = AsyncMock(return_value={"success": True, "batchId": "b6-draft", "items": [{"id": 1}]})
        with patch.object(self.chat.skills_manager, "get_tool_function", return_value=listed):
            data = await self.chat._read_reminder_draft({"platform": "qq", "actor": "owner", "batch_id": "b6-draft"})
        listed.assert_awaited_once_with(platform="qq", platform_id="owner")
        self.assertEqual(data["status"], "Draft")

    async def test_journal_transaction_records_actual_last_write_and_group(self):
        from keytao_bot.utils.draft_mutation_store import DraftMutationClaimStore
        with tempfile.TemporaryDirectory() as directory:
            journal = DraftMutationClaimStore(str(Path(directory) / "journal.db"))
            now = datetime(2026, 10, 27, 10, tzinfo=SHANGHAI).timestamp()
            journal.save_completed_operation('["qq","group","42","owner"]', "qq", "owner", {
                "operationId": "actual-write", "completedAt": now - DAY_SECONDS,
                "batchId": "b6-draft", "batchUrl": "https://offline.invalid/b6-draft",
                "after": {"batchId": "b6-draft", "contentVersion": 1, "items": [{"id": 1}]},
            })
            store = DraftReminderStore(journal.db_path)
            send = AsyncMock(return_value="receipt")
            with patch.dict(os.environ, {"KEYTAO_DRAFT_REMINDER": "1"}):
                self.assertEqual(await store.scan(B6ReminderTests.snapshot, send, now=now), 1)
            self.assertEqual(send.await_args.args[0]["group_id"], "42")

    async def test_quoted_reminder_commands_use_normal_path_for_author_only(self):
        from types import SimpleNamespace
        memory = self.harness.ChatMemoryContext(
            platform="qq", user_id="owner", space_type="group", space_id="42",
        )
        receipt_store = SimpleNamespace(resolve_receipt=lambda *args: {"actor": "owner", "batch_id": "b6-draft"})
        for actor in ("owner", "other"):
            for command in ("提交", "清空草稿"):
                with self.subTest(actor=actor, command=command):
                    ctx = self.chat.TurnContext(
                        bot=SimpleNamespace(), event=SimpleNamespace(), platform="qq", user_id=actor,
                        message_text=command, normalized_message_text=command,
                        memory_context=memory, conv_key=memory.conversation_address,
                        space_key=memory.conversation_address.space_key,
                    )
                    ctx.reply_reference = self.harness.ReplyReferenceInfo(is_reply=True)
                    with (
                        patch.dict(os.environ, {"KEYTAO_DRAFT_REMINDER": "1"}),
                        patch.object(self.chat, "_draft_reminder_store", return_value=receipt_store),
                        patch.object(self.chat, "extract_onebot_reply_id", return_value="receipt"),
                        patch.object(self.chat, "_try_handle_draft_management_command", AsyncMock(return_value="normal reply")) as normal,
                        patch.object(self.chat, "_finish_ai_chat_response", AsyncMock()),
                        patch.object(self.chat, "AsyncOpenAI") as model,
                    ):
                        self.assertTrue(await self.chat._stage_draft_reminder_reply(ctx))
                    model.assert_not_called()
                    if actor == "owner":
                        normal.assert_awaited_once()
                        self.assertEqual(normal.await_args.kwargs["command_intent"].intent,
                                         "draft_submit" if command == "提交" else "draft_clear")
                    else:
                        normal.assert_not_awaited()
                        self.assertEqual(ctx.response, "这条提议不是发给你的，本次未执行。")

    async def test_reminder_submit_cannot_bypass_running_draft_operation(self):
        from types import SimpleNamespace
        memory = self.harness.ChatMemoryContext(
            platform="qq", user_id="owner", space_type="group", space_id="42",
        )
        coordinator = self.harness.DraftOperationCoordinator()
        active = coordinator.begin(memory.conversation_address, "add_and_submit")
        ctx = self.chat.TurnContext(
            bot=SimpleNamespace(), event=SimpleNamespace(), platform="qq", user_id="owner",
            message_text="提交", normalized_message_text="提交", memory_context=memory,
            conv_key=memory.conversation_address, space_key=memory.conversation_address.space_key,
        )
        ctx.reply_reference = self.harness.ReplyReferenceInfo(is_reply=True)
        receipt_store = SimpleNamespace(resolve_receipt=lambda *args: {"actor": "owner"})
        with (
            patch.dict(os.environ, {"KEYTAO_DRAFT_REMINDER": "1"}),
            patch.object(self.chat, "draft_operation_coordinator", coordinator),
            patch.object(self.chat, "_draft_reminder_store", return_value=receipt_store),
            patch.object(self.chat, "extract_onebot_reply_id", return_value="receipt"),
            patch.object(self.chat, "_try_handle_draft_management_command", AsyncMock()) as normal,
            patch.object(self.chat, "_finish_ai_chat_response", AsyncMock()),
        ):
            self.assertTrue(await self.chat._stage_draft_reminder_reply(ctx))
        normal.assert_not_awaited()
        self.assertIs(coordinator.get(memory.conversation_address), active)


if __name__ == "__main__":
    unittest.main()
