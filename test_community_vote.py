"""Offline checks for community approval decisions and their HTTP contract."""
import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness

from keytao_bot.utils import community_vote as vote


BATCH_ID = "bc32871a-1111-2222-3333-444444444444"
BATCH_URL = f"https://keytao.rea.ink/batch/{BATCH_ID}"
RECEIPT = (
    "批次已加入提交，审核中。\n"
    f"草稿地址：{BATCH_URL}\n\n"
    "当前投票：0/3\n"
    "可以引用该条消息，回复同意来投票通过"
)


def submitted_data():
    return {
        "success": True,
        "autoApproved": False,
        "batchId": BATCH_ID,
        "batchUrl": BATCH_URL,
        "contentVersion": 7,
        "snapshotItems": [{
            "id": 1, "action": "Create", "word": "开团", "code": "khtt", "type": "Phrase",
        }],
    }


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.flag = patch.dict(os.environ, {"KEYTAO_COMMUNITY_VOTE": "1"})
        self.flag.start()
        self.addCleanup(self.flag.stop)

    def test_exact_receipt_and_exact_flag(self):
        self.assertEqual(vote.submitted_vote_receipt(submitted_data()), RECEIPT)
        for value in ("", "0", "true", "yes", " 1", "1 "):
            with self.subTest(value=value), patch.dict(os.environ, {"KEYTAO_COMMUNITY_VOTE": value}):
                self.assertFalse(vote.enabled())
                self.assertIsNone(vote.submitted_vote_receipt(submitted_data()))
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(vote.enabled())

    def test_bare_delete_is_closed_but_matching_move_is_open(self):
        data = submitted_data()
        deleted = {"id": 2, "action": "Delete", "word": "开团", "code": "khttv", "type": "Phrase"}
        data["snapshotItems"] = [deleted]
        self.assertIsNone(vote.submitted_vote_receipt(data))
        data["snapshotItems"].append(submitted_data()["snapshotItems"][0])
        self.assertEqual(vote.submitted_vote_receipt(data), RECEIPT)
        data["snapshotItems"][1]["type"] = "English"
        self.assertIsNone(vote.submitted_vote_receipt(data))

    def test_incomplete_or_non_review_result_never_advertises_voting(self):
        for field in ("batchId", "batchUrl", "contentVersion", "snapshotItems"):
            with self.subTest(missing=field):
                data = submitted_data()
                data.pop(field)
                self.assertIsNone(vote.submitted_vote_receipt(data))
        for field, value in (("success", False), ("autoApproved", True), ("contentVersion", True)):
            with self.subTest(field=field, value=value):
                data = submitted_data()
                data[field] = value
                self.assertIsNone(vote.submitted_vote_receipt(data))

    def test_final_delivery_preserves_exact_receipt_and_original_batch_url(self):
        pending_token = vote._pending_receipts.set([])
        try:
            self.assertEqual(vote.submitted_vote_receipt(submitted_data()), RECEIPT)
            memory = SimpleNamespace(platform="telegram", conversation_address=None)
            self.assertEqual(harness.openai_chat_module._prepare_user_facing_reply(RECEIPT, memory), RECEIPT)
        finally:
            vote._pending_receipts.reset(pending_token)


class VoteTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.flag = patch.dict(os.environ, {"KEYTAO_COMMUNITY_VOTE": "1"})
        self.flag.start()
        self.addCleanup(self.flag.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "community-votes.db"
        self.store = vote.CommunityVoteStore(self.db_path)
        self.assertTrue(self.store.bind_receipt(submitted_data(), "qq", "creator", "group", "receipt"))
        users = {
            "creator": {"id": 100}, "creator-alt": {"id": 100},
            "one": {"id": 1}, "one-alt": {"id": 1},
            "two": {"id": 2}, "three": {"id": 3}, "four": {"id": 4},
        }
        self.lookup = AsyncMock(side_effect=lambda _platform, user_id: users.get(user_id))
        self.approve = AsyncMock(return_value={"success": True, "voters": ["甲", "乙", "丙"]})

    async def cast(self, user_id, name=None, **kwargs):
        return await vote.record_vote(
            self.store, "qq", kwargs.pop("group_id", "group"),
            kwargs.pop("message_id", "receipt"), user_id, name or user_id,
            lookup=self.lookup, approve=self.approve, **kwargs,
        )

    async def test_three_votes_approve_once_with_creator_and_version(self):
        self.assertEqual(await self.cast("one", "甲"), (
            f"@甲 已记录同意。\n草稿地址：{BATCH_URL}\n当前投票：1/3\n同意：甲"
        ))
        self.assertIn("当前投票：2/3", await self.cast("two", "乙"))
        self.assertEqual(await self.cast("three", "丙"), (
            f"投票通过 3/3，批次已加入词库，投票结束。\n草稿地址：{BATCH_URL}\n同意：甲、乙、丙"
        ))
        self.approve.assert_awaited_once_with("qq", "creator", BATCH_ID, [
            {"platform": "qq", "platformId": "one"},
            {"platform": "qq", "platformId": "two"},
            {"platform": "qq", "platformId": "three"},
        ], 7)
        self.assertEqual(self.store.get(BATCH_ID)["status"], "approved")
        self.assertEqual(await self.cast("four"), "投票已结束。")
        self.assertEqual(self.approve.await_count, 1)

    async def test_deduplicates_bound_account_and_rejects_creator_or_unbound(self):
        self.assertEqual(await self.cast("unknown"), "请先绑定键道账号再投票")
        for creator in ("creator", "creator-alt"):
            with self.subTest(creator=creator):
                self.assertEqual(await self.cast(creator), "提交人不能为自己投票")
        await self.cast("one")
        self.assertEqual(await self.cast("one"), "你已经投过了（1/3）")
        self.assertEqual(await self.cast("one-alt"), "你已经投过了（1/3）")
        self.assertEqual(len(self.store.get(BATCH_ID)["voters"]), 1)
        self.approve.assert_not_awaited()

    async def test_concurrent_third_and_fourth_only_send_one_approval(self):
        await self.cast("one")
        await self.cast("two")
        entered = asyncio.Event()
        release = asyncio.Event()

        async def slow_approve(*_args):
            entered.set()
            await release.wait()
            return {"success": True}

        self.approve.side_effect = slow_approve
        third = asyncio.create_task(self.cast("three"))
        await asyncio.wait_for(entered.wait(), 2)
        fourth = asyncio.create_task(self.cast("four"))
        await asyncio.sleep(0)
        release.set()
        replies = await asyncio.wait_for(asyncio.gather(third, fourth), 2)
        self.assertEqual(self.approve.await_count, 1)
        self.assertEqual(len(self.store.get(BATCH_ID)["voters"]), 3)
        self.assertIn("投票已结束。", replies)

    async def test_server_refusal_ends_vote_and_never_retries(self):
        self.approve.return_value = {
            "success": False, "retryable": False, "statusCode": 409, "message": "批次内容已变更",
        }
        await self.cast("one")
        await self.cast("two")
        self.assertEqual(await self.cast("three"), "投票已结束：批次内容已变更")
        self.assertEqual(self.store.get(BATCH_ID)["status"], "ended")
        self.assertEqual(await self.cast("four"), "投票已结束。")
        self.approve.assert_awaited_once()

    async def test_transport_failure_keeps_votes_until_next_explicit_vote(self):
        self.approve.side_effect = [TimeoutError("uncertain write"), {"success": True}]
        await self.cast("one")
        await self.cast("two")
        pending = await self.cast("three")
        self.assertIn("待", pending)
        self.assertEqual(self.approve.await_count, 1)
        self.assertEqual(len(self.store.get(BATCH_ID)["voters"]), 3)
        self.assertEqual(self.store.get(BATCH_ID)["status"], "open")
        await asyncio.sleep(0)
        self.assertEqual(self.approve.await_count, 1)
        self.assertIn("投票通过 3/3", await self.cast("three"))
        self.assertEqual(self.approve.await_count, 2)
        self.assertEqual(len(self.store.get(BATCH_ID)["voters"]), 3)
        self.assertEqual(self.approve.await_args_list[0], self.approve.await_args_list[1])

    async def test_arrival_during_uncertain_write_cannot_retry_that_write(self):
        await self.cast("one")
        await self.cast("two")
        entered = asyncio.Event()
        release = asyncio.Event()

        async def uncertain_approve(*_args):
            if not release.is_set():
                entered.set()
                await release.wait()
                raise TimeoutError("uncertain write")
            return {"success": True}

        self.approve.side_effect = uncertain_approve
        third = asyncio.create_task(self.cast("three"))
        await asyncio.wait_for(entered.wait(), 2)
        fourth = asyncio.create_task(self.cast("four"))
        await asyncio.sleep(0)
        self.assertFalse(fourth.done())
        release.set()
        replies = await asyncio.wait_for(asyncio.gather(third, fourth), 2)
        self.assertEqual(self.approve.await_count, 1)
        self.assertTrue(all("待确认" in reply for reply in replies))
        self.assertEqual(len(self.store.get(BATCH_ID)["voters"]), 3)
        self.assertIn("投票通过 3/3", await self.cast("four"))
        self.assertEqual(self.approve.await_count, 2)
        self.assertEqual(self.approve.await_args_list[0], self.approve.await_args_list[1])

    async def test_restart_retains_votes_receipt_mapping_and_status(self):
        await self.cast("one")
        self.store = vote.CommunityVoteStore(self.db_path)
        self.assertEqual(self.store.resolve("qq", "group", "receipt"), BATCH_ID)
        self.assertEqual(len(self.store.get(BATCH_ID)["voters"]), 1)
        self.assertIn("当前投票：2/3", await self.cast("two"))
        await self.cast("three")
        self.store = vote.CommunityVoteStore(self.db_path)
        self.assertEqual(await self.cast("four"), "投票已结束。")

    async def test_unrelated_reference_and_flag_off_fall_through(self):
        self.assertIsNone(await self.cast("one", group_id="other-group"))
        self.assertIsNone(await self.cast("one", message_id="other-receipt"))
        self.assertIsNone(await self.cast("one", message_id=""))
        with patch.dict(os.environ, {"KEYTAO_COMMUNITY_VOTE": "0"}):
            self.assertIsNone(await self.cast("one"))
            self.assertFalse(self.store.bind_receipt(submitted_data(), "qq", "creator", "group", "new"))
        self.lookup.assert_not_awaited()
        self.approve.assert_not_awaited()

    async def test_duplicate_receipt_does_not_reset_existing_votes(self):
        await self.cast("one")
        self.assertTrue(self.store.bind_receipt(submitted_data(), "qq", "creator", "group", "receipt-two"))
        self.assertEqual(len(self.store.get(BATCH_ID)["voters"]), 1)
        self.assertEqual(self.store.resolve("qq", "group", "receipt-two"), BATCH_ID)

    async def test_resubmit_starts_fresh_round_and_old_receipts_stay_ended_after_restart(self):
        for status in ("open", "ended", "approved"):
            with self.subTest(previous_status=status):
                self.store = vote.CommunityVoteStore(Path(self.tmp.name) / f"{status}.db")
                self.store.bind_receipt(submitted_data(), "qq", "creator", "group", "receipt")
                await self.cast("one")
                old = self.store.get(BATCH_ID)
                old["status"] = status
                self.store.save(old)
                updated = {**submitted_data(), "contentVersion": 9}
                self.assertTrue(self.store.bind_receipt(updated, "qq", "creator", "group", "resubmitted"))
                self.store = vote.CommunityVoteStore(self.store.db_path)
                self.assertEqual(await self.cast("two"), "投票已结束。")
                self.assertEqual(self.store.get(BATCH_ID)["voters"], [])
                self.assertIn("当前投票：1/3", await self.cast("one", message_id="resubmitted"))
                await self.cast("two", message_id="resubmitted")
                self.assertIn("投票通过 3/3", await self.cast("three", message_id="resubmitted"))
                self.assertEqual(self.approve.await_args.args[-1], 9)
                self.assertEqual(await self.cast("four"), "投票已结束。")

    async def test_legacy_database_keeps_receipts_on_old_round_after_upgrade(self):
        legacy_path = Path(self.tmp.name) / "legacy.db"
        with closing(sqlite3.connect(legacy_path)) as conn, conn:
            conn.execute("""CREATE TABLE community_votes (
                batch_id TEXT PRIMARY KEY, creator_platform TEXT NOT NULL,
                creator_id TEXT NOT NULL, creator_user_id TEXT,
                content_version INTEGER NOT NULL, batch_url TEXT NOT NULL,
                voters TEXT NOT NULL DEFAULT '[]', status TEXT NOT NULL DEFAULT 'open',
                attempts INTEGER NOT NULL DEFAULT 0)""")
            conn.execute("""CREATE TABLE community_vote_receipts (
                platform TEXT NOT NULL, group_id TEXT NOT NULL, message_id TEXT NOT NULL,
                batch_id TEXT NOT NULL, PRIMARY KEY (platform, group_id, message_id))""")
            conn.execute("INSERT INTO community_votes (batch_id,creator_platform,creator_id,content_version,batch_url) "
                         "VALUES (?,?,?,?,?)", (BATCH_ID, "qq", "creator", 7, BATCH_URL))
            conn.execute("INSERT INTO community_vote_receipts VALUES (?,?,?,?)",
                         ("qq", "group", "receipt", BATCH_ID))
        self.store = vote.CommunityVoteStore(legacy_path)
        self.assertIn("当前投票：1/3", await self.cast("one"))
        updated = {**submitted_data(), "contentVersion": 9}
        self.assertTrue(self.store.bind_receipt(updated, "qq", "creator", "group", "resubmitted"))
        self.assertEqual(await self.cast("two"), "投票已结束。")
        self.assertIn("当前投票：1/3", await self.cast("one", message_id="resubmitted"))

    async def test_transport_failure_retries_only_once_even_after_restart(self):
        for result in (TimeoutError("uncertain write"), {"success": False, "retryable": True}):
            with self.subTest(result=result):
                self.store = vote.CommunityVoteStore(Path(self.tmp.name) / f"retry-{type(result).__name__}.db")
                self.store.bind_receipt(submitted_data(), "qq", "creator", "group", "receipt")
                self.approve = AsyncMock(side_effect=result) if isinstance(result, Exception) else AsyncMock(return_value=result)
                await self.cast("one")
                await self.cast("two")
                self.assertIn("重试", await self.cast("three"))
                pending = await self.cast("three")
                self.assertIn("管理员", pending)
                self.assertNotIn("重试", pending)
                self.store = vote.CommunityVoteStore(self.store.db_path)
                self.assertIn("管理员", await self.cast("four"))
                self.assertFalse(await self.cast("three"))
                self.assertFalse(await self.cast("four"))
                self.assertEqual(self.approve.await_count, 2)
                self.assertEqual(len(self.store.get(BATCH_ID)["voters"]), 3)

    async def test_resubmit_during_lookup_or_approval_returns_ended_for_old_round(self):
        for phase in ("voter", "creator", "approval-error", "approval-retryable"):
            with self.subTest(phase=phase):
                self.store = vote.CommunityVoteStore(Path(self.tmp.name) / f"race-{phase}.db")
                self.store.bind_receipt(submitted_data(), "qq", "creator", "group", "receipt")
                self.lookup = AsyncMock(side_effect=lambda _platform, user_id: {"id": 100 if user_id == "creator" else user_id})
                if phase.startswith("approval"):
                    await self.cast("one")
                    await self.cast("two")
                entered, release = asyncio.Event(), asyncio.Event()

                async def delayed_lookup(_platform, user_id):
                    if phase == "voter" or user_id == "creator":
                        entered.set()
                        await release.wait()
                    return None if phase == "voter" else {"id": user_id}

                async def delayed_approve(*_args):
                    entered.set()
                    await release.wait()
                    if phase == "approval-error":
                        raise TimeoutError("uncertain write")
                    return {"success": False, "retryable": True}

                if phase.startswith("approval"):
                    self.approve = AsyncMock(side_effect=delayed_approve)
                else:
                    self.lookup = AsyncMock(side_effect=delayed_lookup)
                pending = asyncio.create_task(self.cast("three"))
                await asyncio.wait_for(entered.wait(), 2)
                try:
                    self.assertTrue(self.store.bind_receipt(
                        {**submitted_data(), "contentVersion": 9}, "qq", "creator", "group", "new"))
                finally:
                    release.set()
                self.assertEqual(await asyncio.wait_for(pending, 2), "投票已结束。")
                self.assertEqual(self.store.get(BATCH_ID)["voters"], [])

    async def test_voter_names_are_cq_escaped_in_vote_pending_and_success_replies(self):
        name = "A&[CQ:at,qq=all]"
        safe = "A&amp;&#91;CQ:at,qq=all&#93;"
        first = await self.cast("one", name)
        self.assertIn(f"@{safe} 已记录同意", first)
        self.assertIn(f"同意：{safe}", first)
        await self.cast("two", name)
        self.approve.side_effect = [TimeoutError("uncertain write"), {"success": True, "voters": [name] * 3}]
        self.assertIn(f"同意：{safe}、{safe}、{safe}", await self.cast("three", name))
        success = await self.cast("four", name)
        self.assertIn(f"同意：{safe}、{safe}、{safe}", success)
        self.assertNotIn("[CQ:", success)

    async def test_ineligible_reply_is_once_per_user_round_and_survives_restart(self):
        await self.cast("one")
        for user_id, expected in (("unknown", "请先绑定"), ("creator", "提交人"),
                                  ("creator-alt", "提交人"), ("one", "你已经"), ("one-alt", "你已经")):
            with self.subTest(user_id=user_id):
                self.assertIn(expected, await self.cast(user_id))
                self.store = vote.CommunityVoteStore(self.db_path)
                self.assertFalse(await self.cast(user_id))
        updated = {**submitted_data(), "contentVersion": 9}
        self.assertTrue(self.store.bind_receipt(updated, "qq", "creator", "group", "new"))
        self.assertIn("请先绑定", await self.cast("unknown", message_id="new"))

    def context(self, *, is_reply=True, message_text="同意"):
        return SimpleNamespace(
            memory_context=SimpleNamespace(platform="qq", space_type="group", space_id="group", user_id="one"),
            reply_reference=SimpleNamespace(is_reply=is_reply),
            platform="qq", event=object(), message_text=message_text,
            user_id="one", owner_label="甲", bot=object(), QQMessageSegment=None,
            response="",
        )

    async def test_stage_nonquote_and_unmapped_quote_do_not_intercept(self):
        chat = harness.openai_chat_module
        finish = AsyncMock()
        with (
            patch.object(vote, "get_store", return_value=self.store),
            patch.object(chat, "extract_onebot_reply_id", return_value="unrelated"),
            patch.object(chat, "_finish_ai_chat_response", finish),
        ):
            self.assertFalse(await vote._stage_community_vote(self.context(is_reply=False)))
            self.assertFalse(await vote._stage_community_vote(self.context()))
        finish.assert_not_awaited()
        self.assertEqual(len(self.store.get(BATCH_ID)["voters"]), 0)

    async def test_stage_closed_assent_command_consumes_mapped_quote(self):
        chat = harness.openai_chat_module
        finish = AsyncMock()
        ctx = self.context(message_text=" [CQ:at,qq=123] @bot 同 意 ！。 ")
        with (
            patch.object(vote, "get_store", return_value=self.store),
            patch.object(vote, "find_user_by_platform", self.lookup),
            patch.object(chat, "extract_onebot_reply_id", return_value="receipt"),
            patch.object(chat, "_finish_ai_chat_response", finish),
            patch.object(chat.skills_manager, "get_tool_function", return_value=self.approve),
        ):
            self.assertTrue(await vote._stage_community_vote(ctx))
        self.assertIn("当前投票：1/3", ctx.response)
        finish.assert_awaited_once()
        self.assertEqual(len(self.store.get(BATCH_ID)["voters"]), 1)
        self.approve.assert_not_awaited()

    async def test_stage_other_text_on_open_or_ended_receipt_passes_through_untouched(self):
        chat = harness.openai_chat_module
        finish = AsyncMock()
        with (
            patch.object(vote, "get_store", return_value=self.store),
            patch.object(chat, "extract_onebot_reply_id", return_value="receipt"),
            patch.object(chat, "_finish_ai_chat_response", finish),
        ):
            for status in ("open", "ended", "approved"):
                data = self.store.get(BATCH_ID)
                data["status"] = status
                self.store.save(data)
                for text in ("撤回提交", "这个词编码对吗", "同意撤回", "", "同意吗？"):
                    with self.subTest(status=status, text=text):
                        ctx = self.context(message_text=text)
                        ctx.response = "untouched"
                        self.assertFalse(await vote._stage_community_vote(ctx))
                        self.assertEqual(ctx.response, "untouched")
        finish.assert_not_awaited()
        self.lookup.assert_not_awaited()

    async def test_silent_ineligible_assent_still_consumes_quote_before_ticket_routing(self):
        chat = harness.openai_chat_module
        await self.cast("one")
        await self.cast("one")
        finish = AsyncMock()
        with (
            patch.object(vote, "get_store", return_value=self.store),
            patch.object(vote, "find_user_by_platform", self.lookup),
            patch.object(chat, "extract_onebot_reply_id", return_value="receipt"),
            patch.object(chat, "_finish_ai_chat_response", finish),
            patch.object(chat.skills_manager, "get_tool_function", return_value=self.approve),
        ):
            self.assertTrue(await vote._stage_community_vote(self.context()))
        finish.assert_not_awaited()
        self.approve.assert_not_awaited()

    async def test_sent_receipt_binds_actual_message_id_in_its_group_only(self):
        memory = SimpleNamespace(platform="qq", space_type="group", space_id="group", user_id="creator")
        memory_token = harness.chat_commands_module.current_memory_context.set(memory)
        pending_token = vote._pending_receipts.set([submitted_data()])
        try:
            with patch.object(vote, "_default_store", self.store):
                await vote._record_sent_receipts(None, None, "send_group_msg", {
                    "group_id": "group", "message": RECEIPT,
                }, {"message_id": 42})
                await vote._record_sent_receipts(None, None, "send_group_msg", {
                    "group_id": "other-group", "message": RECEIPT,
                }, {"message_id": 43})
                await vote._record_sent_receipts(None, RuntimeError("send failed"), "send_group_msg", {
                    "group_id": "group", "message": RECEIPT,
                }, {"message_id": 44})
            self.assertEqual(self.store.resolve("qq", "group", "42"), BATCH_ID)
            self.assertIsNone(self.store.resolve("qq", "other-group", "43"))
            self.assertIsNone(self.store.resolve("qq", "group", "44"))
        finally:
            harness.chat_commands_module.current_memory_context.reset(memory_token)
            vote._pending_receipts.reset(pending_token)

    async def test_child_submission_metadata_reaches_parent_send_hook(self):
        memory = SimpleNamespace(platform="qq", space_type="group", space_id="group", user_id="creator")
        memory_token = harness.chat_commands_module.current_memory_context.set(memory)
        pending_token = vote._pending_receipts.set(None)
        try:
            self.assertFalse(await vote._stage_community_vote(self.context(is_reply=False)))

            async def submit_in_child():
                return vote.submitted_vote_receipt(submitted_data())

            receipt = await asyncio.wait_for(submit_in_child(), 2)
            with patch.object(vote, "_default_store", self.store):
                await vote._record_sent_receipts(None, None, "send_group_msg", {
                    "group_id": "group", "message": receipt,
                }, {"message_id": 45})
            self.assertEqual(self.store.resolve("qq", "group", "45"), BATCH_ID)
        finally:
            harness.chat_commands_module.current_memory_context.reset(memory_token)
            vote._pending_receipts.reset(pending_token)

    async def test_combined_batch_receipts_never_bind_an_ambiguous_quote(self):
        second = submitted_data()
        second["batchId"] = "bc32871a-1111-2222-3333-555555555555"
        second["batchUrl"] = f"https://keytao.rea.ink/batch/{second['batchId']}"
        memory = SimpleNamespace(platform="qq", space_type="group", space_id="group", user_id="creator")
        memory_token = harness.chat_commands_module.current_memory_context.set(memory)
        pending_token = vote._pending_receipts.set([])
        try:
            self.assertIsNone(vote.submitted_vote_receipts([
                {**submitted_data(), "tool": "keytao_submit_batch"},
                {**second, "tool": "keytao_submit_batch"},
            ]))
            combined = "\n\n".join((
                vote.submitted_vote_receipt(submitted_data()), vote.submitted_vote_receipt(second),
            ))
            prepared = vote.prepare_delivery(combined, memory)
            self.assertNotIn("当前投票", prepared)
            self.assertNotIn("回复同意", prepared)
            self.assertEqual(prepared.count("✅ 批次已提交审核。"), 2)
            self.assertIn(BATCH_URL, prepared)
            self.assertIn(second["batchUrl"], prepared)
            with patch.object(vote, "_default_store", self.store):
                await vote._record_sent_receipts(None, None, "send_group_msg", {
                    "group_id": "group", "message": prepared,
                }, {"message_id": 46})
            self.assertIsNone(self.store.resolve("qq", "group", "46"))
            self.assertIsNone(self.store.get(second["batchId"]))
        finally:
            harness.chat_commands_module.current_memory_context.reset(memory_token)
            vote._pending_receipts.reset(pending_token)

    async def test_submit_command_renders_canonical_receipt_or_legacy_for_bare_delete(self):
        command = harness.chat_commands_module._perform_submit_current_draft
        for bare_delete in (False, True):
            with self.subTest(bare_delete=bare_delete):
                data = submitted_data()
                if bare_delete:
                    data["snapshotItems"][0]["action"] = "Delete"
                with patch.object(harness.chat_commands_module, "call_tool_function", AsyncMock(
                    return_value=json.dumps(data, ensure_ascii=False),
                )):
                    result = await command(
                        "qq", "creator", confirmed=True, batch_id=BATCH_ID, expected_content_version=7,
                    )
                self.assertTrue(result.success)
                if bare_delete:
                    self.assertTrue(result.text.startswith("✅ 批次已提交审核。\n"))
                    self.assertIn(BATCH_URL, result.text)
                    self.assertNotIn("当前投票", result.text)
                    self.assertNotIn("回复同意", result.text)
                else:
                    self.assertEqual(result.text, RECEIPT)

    async def test_telegram_sent_receipt_is_scoped_to_topic(self):
        self.store = vote.CommunityVoteStore(Path(self.tmp.name) / "telegram-votes.db")
        memory = SimpleNamespace(platform="telegram", space_type="group", space_id="-100:thread:9", user_id="creator")
        memory_token = harness.chat_commands_module.current_memory_context.set(memory)
        pending_token = vote._pending_receipts.set([submitted_data()])
        try:
            with patch.object(vote, "_default_store", self.store):
                await vote._record_sent_receipts(None, None, "send_message", {
                    "chat_id": -100, "message_thread_id": 9, "text": RECEIPT,
                }, SimpleNamespace(message_id=42))
                await vote._record_sent_receipts(None, None, "send_message", {
                    "chat_id": -100, "message_thread_id": 10, "text": RECEIPT,
                }, SimpleNamespace(message_id=43))
            self.assertEqual(self.store.resolve("telegram", "-100:thread:9", "42"), BATCH_ID)
            self.assertIsNone(self.store.resolve("telegram", "-100:thread:10", "43"))
        finally:
            harness.chat_commands_module.current_memory_context.reset(memory_token)
            vote._pending_receipts.reset(pending_token)


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def call_endpoint(self, status=200, payload=None, error=None):
        payload = {"success": True, "voters": ["甲", "乙", "丙"]} if payload is None else payload

        class FakeResponse:
            status_code = status
            is_success = 200 <= status < 300

            def json(self):
                return payload

        post = AsyncMock(side_effect=error, return_value=FakeResponse())

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def post(self, *args, **kwargs):
                return await post(*args, **kwargs)

        with (
            patch.object(harness._draft_tools.httpx, "AsyncClient", return_value=FakeClient(), create=True),
            patch.object(harness._draft_tools, "get_keytao_url", return_value="https://keytao.test"),
            patch.object(harness._draft_tools, "get_bot_headers", return_value={"X-Test": "signed"}) as headers,
        ):
            result = await harness._draft_tools._community_approve_submitted_batch(
                "qq", "creator", BATCH_ID,
                [{"platform": "qq", "platformId": user_id} for user_id in ("one", "two", "three")], 7,
            )
        return result, post, headers

    async def test_signed_request_contains_frozen_endpoint_contract(self):
        result, post, headers = await self.call_endpoint()
        self.assertTrue(result["success"])
        post.assert_awaited_once()
        args, kwargs = post.await_args
        endpoint = f"/api/bot/batches/{BATCH_ID}/community-approve"
        self.assertEqual(args, (f"https://keytao.test{endpoint}",))
        self.assertEqual(kwargs["headers"], {"X-Test": "signed"})
        self.assertEqual(json.loads(kwargs["content"]), {
            "platform": "qq", "platformId": "creator", "expectedContentVersion": 7,
            "voters": [{"platform": "qq", "platformId": user_id} for user_id in ("one", "two", "three")],
        })
        self.assertEqual(headers.call_args.kwargs["path"], endpoint)
        self.assertEqual(headers.call_args.kwargs["method"], "POST")
        self.assertEqual(headers.call_args.kwargs["raw_body"], kwargs["content"])

    async def test_refusal_is_terminal_and_transport_is_retryable_without_automatic_retry(self):
        refused, post, _headers = await self.call_endpoint(409, {"success": False, "message": "内容已变更"})
        self.assertEqual(refused["message"], "内容已变更")
        self.assertFalse(refused["retryable"])
        post.assert_awaited_once()
        pending, post, _headers = await self.call_endpoint(error=TimeoutError("uncertain write"))
        self.assertFalse(pending["success"])
        self.assertTrue(pending["retryable"])
        post.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
