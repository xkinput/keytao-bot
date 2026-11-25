"""Offline coverage for draft reminders and exact submit confirmation."""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from nonebot.adapters.onebot.v11 import MessageSegment

from tests import test_state_machine as harness
from keytao_bot.utils import draft_reminder, http_client
from keytao_bot.utils import history_store as history_module
from keytao_bot.utils.history_store import HistoryStore


chat = harness.openai_chat_module
BATCH = "reminded-batch"
ACTOR = "12345"
GROUP = "42"
LINK = "https://fake.invalid/batch/reminded-batch"
NOW = datetime(2026, 11, 25, 10, tzinfo=draft_reminder.SHANGHAI)
ITEMS = [{"id": 1, "action": "Create", "word": "词条", "code": "ctab", "type": "Phrase"}]


class DraftReminderFixture(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.store = draft_reminder.DraftReminderStore(str(self.directory / "reminders.db"))
        self.clock = NOW.timestamp()
        self.enterContext(patch.dict(os.environ, {
            "KEYTAO_DRAFT_REMINDER": "1", "KEYTAO_DRAFT_REMINDER_IDLE_MINUTES": "",
            "KEYTAO_DRAFT_REMINDER_MAX_AGE_DAYS": "7", "KEYTAO_DRAFT_REMINDER_GROUP_MAX_AGE_DAYS": "30",
            "KEYTAO_COMMUNITY_VOTE": "0",
        }))
        self.enterContext(patch.object(draft_reminder, "time", SimpleNamespace(time=lambda: self.clock)))
        self.enterContext(patch.object(chat, "_draft_reminder_store", return_value=self.store))
        self.read = AsyncMock(side_effect=self.snapshot)
        self.send = AsyncMock(return_value="reminder")

    async def asyncSetUp(self):
        self.rollout = (NOW - timedelta(days=2)).replace(hour=0)
        await self.store.scan(self.read, self.send, now=self.rollout.timestamp())

    @staticmethod
    def snapshot(candidate):
        return {
            "success": True, "batchId": candidate["batch_id"], "status": "Draft",
            "contentVersion": candidate["content_version"], "items": ITEMS, "batchUrl": LINK,
        }

    def write(self, minutes=20, *, batch=BATCH, actor=ACTOR, group=GROUP, version=7, platform="qq"):
        self.store.record_write(json.dumps([platform, "group", group, actor]), platform, actor, {
            "completedAt": self.clock - minutes * 60, "batchId": batch, "batchUrl": LINK,
            "after": {"batchId": batch, "contentVersion": version, "items": ITEMS},
        })

    async def scan(self):
        return await self.store.scan(self.read, self.send)


class DraftReminderTests(DraftReminderFixture):
    def web_row(self, **overrides):
        return {
            "batchId": BATCH, "creatorQqId": ACTOR, "itemCount": 1,
            "lastModifiedAt": (NOW - timedelta(minutes=35)).isoformat(),
            "contentVersion": 7, "batchUrl": LINK, **overrides,
        }

    def group_history(self, *addresses, when=NOW):
        history = HistoryStore(str(self.directory / "history.db"), retention_days=36500)
        with patch.object(history_module, "_utc_now_iso", return_value=when.isoformat()):
            for address in addresses:
                history.add_conversation_round(address, "查询", "结果")
        return history

    async def scan_with_web(self, reply, history):
        with (
            patch.object(chat, "history_store", history),
            patch.object(http_client, "keytao_json", reply),
            patch.object(chat, "_read_reminder_draft", self.read),
            patch.object(chat, "_send_draft_reminder", self.send),
            patch.object(chat._user_resolver, "find_user_by_platform", AsyncMock()),
        ):
            return await chat._scan_draft_reminders_once()

    async def test_twenty_minutes_after_last_write_and_five_minute_floor(self):
        with patch.dict(os.environ):
            os.environ.pop("KEYTAO_DRAFT_REMINDER_IDLE_MINUTES")
            self.write(60, version=6)
            self.write(version=7)
            self.clock -= 1
            self.assertEqual(0, await self.scan())
            self.read.assert_not_awaited()
            self.clock += 1
            self.assertEqual(1, await self.scan())
        with patch.dict(os.environ, {"KEYTAO_DRAFT_REMINDER_IDLE_MINUTES": "2"}):
            self.clock += draft_reminder.DAY_SECONDS
            self.write(5, batch="floor")
            self.clock -= 1
            self.assertEqual(0, await self.scan())
            self.clock += 1
            self.assertEqual(1, await self.scan())

    async def test_oldest_per_actor_per_shanghai_day_and_once_per_batch(self):
        self.clock = NOW.replace(hour=22, minute=55).timestamp()
        self.write(20, batch="newer", group="45")
        self.write(35)
        self.write(40, batch="other-actor", actor="67890")
        self.assertEqual(2, await self.scan())
        self.assertEqual(["other-actor", BATCH], [
            call.args[0]["batch_id"] for call in self.send.await_args_list
        ])
        self.assertEqual(0, await self.scan())
        self.clock = (NOW + timedelta(days=1)).replace(hour=8).timestamp()
        self.write(100, version=8)
        self.store = draft_reminder.DraftReminderStore(self.store.db_path)
        self.assertEqual(1, await self.scan())
        self.assertEqual("newer", self.send.await_args.args[0]["batch_id"])
        self.clock += draft_reminder.DAY_SECONDS
        self.assertEqual(0, await self.scan())

    async def test_window_and_feature_switch_gate_every_read(self):
        self.write(240)
        web = AsyncMock(return_value=[])
        for day, hour, minute in ((0, 7, 59), (0, 23, 0), (1, 0, 0)):
            self.clock = (NOW + timedelta(days=day)).replace(hour=hour, minute=minute).timestamp()
            self.assertEqual(0, await self.store.scan(self.read, self.send, list_web_candidates=web))
        self.clock = NOW.replace(hour=8).timestamp()
        with patch.dict(os.environ, {"KEYTAO_DRAFT_REMINDER": "0"}):
            self.assertEqual(0, await self.store.scan(self.read, self.send, list_web_candidates=web))
        self.read.assert_not_awaited()
        web.assert_not_awaited()
        self.assertEqual(1, await self.scan())

    async def test_qq_send_mentions_actor_then_sends_copy_and_link(self):
        bot = type("FakeBot", (), {"__module__": "nonebot.adapters.onebot.v11"})()
        bot.send_group_msg = AsyncMock(return_value={"message_id": 123})
        self.write()
        with patch.object(sys.modules["nonebot"], "get_bots", return_value={"bot": bot}, create=True):
            self.assertEqual(1, await self.store.scan(self.read, chat._send_draft_reminder))
        kwargs = bot.send_group_msg.await_args.kwargs
        self.assertEqual(int(GROUP), kwargs["group_id"])
        message = kwargs["message"]
        self.assertEqual(MessageSegment.at(ACTOR), message[0])
        self.assertEqual(
            " 你有草稿未提交（1 条），是否帮你提交？提交请回复「确认」\n" + LINK,
            "".join(segment.data["text"] for segment in message[1:]),
        )

    async def test_web_drafts_use_recent_group_and_recheck_exact_batch(self):
        history = self.group_history(
            chat.ConversationAddress.group("qq", "41", ACTOR),
            chat.ConversationAddress.group("qq", GROUP, ACTOR),
            chat.ConversationAddress.group("telegram", "99", ACTOR),
            chat.ConversationAddress.private("qq", ACTOR),
        )
        self.group_history(chat.ConversationAddress.group("qq", "99", ACTOR), when=NOW - timedelta(days=31))
        stale_address = chat.ConversationAddress.group("qq", "98", "10001")
        self.group_history(stale_address, when=NOW - timedelta(days=31))
        self.assertEqual(2, history.count_history_rows(stale_address))
        row = self.web_row()
        web = AsyncMock(return_value={"success": True, "drafts": [
            row, {**row, "batchId": "no-group", "creatorQqId": "10001"},
        ]})
        self.write(20, batch="newer-bot-draft")
        exact_read = AsyncMock(side_effect=lambda **kwargs: self.snapshot({
            "batch_id": kwargs["batch_id"], "content_version": 7,
        }))
        with (
            patch.object(chat, "history_store", history),
            patch.object(http_client, "keytao_json", web),
            patch.object(chat.skills_manager, "get_tool_function", return_value=exact_read),
            patch.object(chat, "_send_draft_reminder", self.send),
            patch.object(chat._user_resolver, "find_user_by_platform", AsyncMock()),
        ):
            self.assertEqual(1, await chat._scan_draft_reminders_once())
        web.assert_awaited_once_with("GET", "/api/bot/drafts/idle",
                                    params={"idleMinutes": 20, "limit": 100,
                                            "modifiedAfter": self.rollout.astimezone(timezone.utc).isoformat()},
                                    timeout=5.0, retries=1)
        exact_read.assert_awaited_once_with(platform="qq", platform_id=ACTOR, batch_id=BATCH)
        self.assertEqual(GROUP, self.send.await_args.args[0]["group_id"])

    async def test_web_paginates_past_unaskable_drafts_in_one_scan(self):
        history = self.group_history(chat.ConversationAddress.group("qq", GROUP, ACTOR))
        rows = [self.web_row(
            batchId=f"old-{i}", creatorQqId="67890" if i >= 3 else ACTOR,
            lastModifiedAt=(NOW - timedelta(hours=1, seconds=120 - i)).isoformat(),
        ) for i in range(120)]
        rows[2]["batchUrl"] = ""
        rows.append(self.web_row())
        self.write(25, batch="newer-local")
        self.store.record_write(json.dumps(["qq", "private", ACTOR, ACTOR]), "qq", ACTOR, {
            "completedAt": self.clock - 40 * 60, "batchId": "old-0",
            "after": {"batchId": "old-0", "contentVersion": 7, "items": ITEMS},
        })
        with self.store._connect() as conn:
            conn.execute("INSERT INTO draft_reminder_read_retry VALUES (?,?)", ("old-1", self.clock + 60))
            conn.execute(
                "INSERT INTO draft_reminder_attempt "
                "(platform,actor,batch_id,content_version,group_id,intended_at,status) "
                "VALUES ('qq','67890','already-asked',7,'42',?,'pending')", (self.clock,),
            )
        self.read.side_effect = lambda candidate: {
            **self.snapshot(candidate), "batchUrl": candidate["batch_url"],
        }

        async def list_page(*args, params, **kwargs):
            after = datetime.fromisoformat(params["modifiedAfter"])
            matching = [row for row in rows if datetime.fromisoformat(row["lastModifiedAt"]) > after]
            return {"success": True, "drafts": matching[:params["limit"]]}

        web = AsyncMock(side_effect=list_page)
        self.assertEqual(1, await self.scan_with_web(web, history))
        self.assertEqual(BATCH, self.send.await_args.args[0]["batch_id"])
        self.assertEqual(["old-2", BATCH], [call.args[0]["batch_id"] for call in self.read.await_args_list])
        first, second = [call.kwargs["params"] for call in web.await_args_list]
        self.assertEqual((100, "already-asked"), (first["limit"], first["excludeIds"]))
        self.assertEqual({**first, "modifiedAfter": rows[99]["lastModifiedAt"]}, second)

    async def test_web_stops_after_five_full_pages_or_an_askable_page(self):
        history = self.group_history(chat.ConversationAddress.group("qq", GROUP, ACTOR))
        rows = [self.web_row(
            batchId=f"old-{i}", creatorQqId="67890",
            lastModifiedAt=(NOW - timedelta(seconds=3600 - i)).isoformat(),
        ) for i in range(600)]
        pages = [{"success": True, "drafts": rows[i:i + 100]} for i in range(0, 600, 100)]
        web = AsyncMock(side_effect=pages)
        self.assertEqual(0, await self.scan_with_web(web, history))
        self.assertEqual(5, web.await_count)
        self.read.assert_not_awaited()

        pages[0]["drafts"][0]["creatorQqId"] = ACTOR
        web = AsyncMock(side_effect=pages)
        self.assertEqual(1, await self.scan_with_web(web, history))
        web.assert_awaited_once()

    async def test_missing_url_cools_down_reads_for_one_hour(self):
        history = self.group_history(chat.ConversationAddress.group("qq", GROUP, ACTOR))
        web = AsyncMock(return_value={"success": True, "drafts": [self.web_row(batchUrl="")]})
        self.read.side_effect = lambda candidate: {**self.snapshot(candidate), "batchUrl": ""}
        self.assertEqual(0, await self.scan_with_web(web, history))
        self.clock += 3599
        self.assertEqual(0, await self.scan_with_web(web, history))
        self.read.assert_awaited_once()
        self.send.assert_not_awaited()
        self.clock += 1
        self.read.side_effect = self.snapshot
        self.assertEqual(1, await self.scan_with_web(web, history))

    async def test_saved_rollout_and_max_age_filter_web_and_bot_writes(self):
        self.write(3 * 1440, batch="pre-rollout-bot")
        self.write(60, batch="post-rollout-bot", actor="67890")
        history = self.group_history(chat.ConversationAddress.group("qq", GROUP, ACTOR))
        rows = [self.web_row(lastModifiedAt=(self.rollout - timedelta(seconds=1)).isoformat())]
        web = AsyncMock(return_value={"success": True, "drafts": rows})
        self.store = draft_reminder.DraftReminderStore(self.store.db_path)
        self.enterContext(patch.object(chat, "_draft_reminder_store", return_value=self.store))
        self.assertEqual(1, await self.scan_with_web(web, history))
        self.assertEqual("post-rollout-bot", self.send.await_args.args[0]["batch_id"])

        self.clock += 8 * draft_reminder.DAY_SECONDS
        self.write(8 * 1440, batch="too-old-bot")
        rows[0] = self.web_row()
        self.assertEqual(0, await self.scan_with_web(web, history))
        self.assertEqual(1, self.read.await_count)
        params = web.await_args.kwargs["params"]
        self.assertEqual((NOW + timedelta(days=1)).astimezone(timezone.utc).isoformat(), params["modifiedAfter"])
        self.assertEqual("post-rollout-bot", params["excludeIds"])

    async def test_web_excludes_only_the_most_recent_200_attempted_batches(self):
        with self.store._connect() as conn:
            conn.executemany(
                "INSERT INTO draft_reminder_attempt "
                "(platform,actor,batch_id,content_version,group_id,intended_at,status) "
                "VALUES ('qq','actor',?,7,'group',?,'pending')",
                [(f"asked-{i}", self.clock - 300 + i) for i in range(201)],
            )
        web = AsyncMock(return_value={"success": True, "drafts": []})
        self.assertEqual(0, await self.scan_with_web(web, None))
        self.assertEqual([f"asked-{i}" for i in range(200, 0, -1)],
                         web.await_args.kwargs["params"]["excludeIds"].split(","))

    async def test_group_age_is_configurable_and_applies_to_local_routes(self):
        history = self.group_history(chat.ConversationAddress.group("qq", GROUP, ACTOR), when=NOW - timedelta(days=2))
        self.write(2 * 1440, batch="old-local-group", group="41")
        web = AsyncMock(return_value={"success": True, "drafts": [self.web_row()]})
        with patch.dict(os.environ, {"KEYTAO_DRAFT_REMINDER_GROUP_MAX_AGE_DAYS": "1"}):
            self.assertEqual(0, await self.scan_with_web(web, history))
        self.read.assert_not_awaited()
        self.assertEqual(1, await self.scan_with_web(web, history))

    async def test_malformed_web_list_logs_once_and_bot_drafts_continue(self):
        for i, reply in enumerate(([], {"success": True, "drafts": {}},
                                   {"success": True, "drafts": [None, {}]})):
            with self.subTest(reply=reply), patch.object(chat.logger, "warning") as warning:
                self.write(batch=f"local-{i}", actor=str(i))
                self.assertEqual(1, await self.scan_with_web(AsyncMock(return_value=reply), None))
                warning.assert_called_once()

    async def test_changed_or_submitted_snapshot_cannot_send(self):
        self.write()
        for change in ({"contentVersion": 8}, {"status": "Submitted"}, {"items": []}):
            with self.subTest(change=change):
                self.read.side_effect = lambda candidate: {**self.snapshot(candidate), **change}
                self.assertEqual(0, await self.scan())
                self.clock += 3600
        self.assertEqual(3, self.read.await_count)
        self.send.assert_not_awaited()

    async def test_failed_read_cools_down_bot_and_web_batches_across_restart(self):
        self.write(batch="bot", actor="67890")
        history = self.group_history(chat.ConversationAddress.group("qq", GROUP, ACTOR))
        web = AsyncMock(return_value={"success": True, "drafts": [self.web_row()]})
        self.read.side_effect = [TimeoutError(), {"success": False}, self.snapshot({"batch_id": BATCH,
                                 "content_version": 7}), self.snapshot({"batch_id": "bot", "content_version": 7})]
        self.assertEqual(0, await self.scan_with_web(web, history))
        self.assertEqual(2, self.read.await_count)
        self.store = draft_reminder.DraftReminderStore(self.store.db_path)
        self.enterContext(patch.object(chat, "_draft_reminder_store", return_value=self.store))
        self.clock += 3599
        self.assertEqual(0, await self.scan_with_web(web, history))
        self.assertEqual(2, self.read.await_count)
        self.clock += 1
        self.assertEqual(2, await self.scan_with_web(web, history))

    async def test_daily_cap_uses_one_account_across_qq_and_telegram(self):
        self.write(40, batch="telegram", actor="67890", platform="telegram")
        self.write(30, batch="unresolved", actor="22222", platform="telegram")
        history = self.group_history(chat.ConversationAddress.group("qq", GROUP, ACTOR))
        web = AsyncMock(return_value={"success": True, "drafts": [self.web_row()]})

        async def lookup(platform, actor):
            return None if actor == "22222" else {"id": 42}

        with (
            patch.object(chat, "history_store", history),
            patch.object(http_client, "keytao_json", web),
            patch.object(chat, "_read_reminder_draft", self.read),
            patch.object(chat, "_send_draft_reminder", self.send),
            patch.object(chat._user_resolver, "find_user_by_platform", AsyncMock(side_effect=lookup)),
        ):
            self.assertEqual(2, await chat._scan_draft_reminders_once())
            self.assertEqual(["telegram", "unresolved"], [call.args[0]["batch_id"] for call in self.send.await_args_list])
            self.store = draft_reminder.DraftReminderStore(self.store.db_path)
            with patch.object(chat, "_draft_reminder_store", return_value=self.store):
                self.assertEqual(0, await chat._scan_draft_reminders_once())
                self.clock += draft_reminder.DAY_SECONDS
                self.assertEqual(1, await chat._scan_draft_reminders_once())
        self.assertEqual(BATCH, self.send.await_args.args[0]["batch_id"])


class DraftReminderConfirmationTests(DraftReminderFixture):
    def setUp(self):
        super().setUp()
        self.state = chat.MemoryConversationStateStore(clock=lambda: self.clock)
        self.coordinator = chat.DraftOperationCoordinator()
        self.finish = AsyncMock()
        self.tool = AsyncMock(side_effect=self.submit_tool)
        self.current_version = 7
        self.preview_version = 7
        self.enterContext(patch.object(chat, "conversation_state_store", self.state))
        self.enterContext(patch.object(chat, "draft_operation_coordinator", self.coordinator))
        self.enterContext(patch.object(chat, "_finish_ai_chat_response", self.finish))
        self.enterContext(patch.object(chat, "remember_conversation", return_value=True))
        self.enterContext(patch.object(chat, "call_tool_function", self.tool))
        self.enterContext(patch.object(chat._chat_commands, "_lookup_submit_others_submitted",
                                       AsyncMock(return_value=([], []))))

    def submit_tool(self, name, arguments, platform, actor):
        self.assertEqual(("keytao_submit_batch", "qq", ACTOR), (name, platform, actor))
        self.assertEqual(BATCH, arguments["batch_id"])
        data = {"batchId": BATCH, "contentVersion": self.preview_version,
                "snapshotItems": ITEMS, "batchUrl": LINK}
        if arguments.get("confirmed"):
            self.assertEqual(data["contentVersion"], arguments["expected_content_version"])
            self.assertNotIn("preview_only", arguments)
            return json.dumps({**data, "success": True})
        self.assertTrue(arguments["preview_only"])
        return json.dumps({
            **data, "success": False, "requiresConfirmation": True, "warnings": [],
            "snapshotDigest": "a" * 64, "warningDigest": "b" * 64, "auditDigest": "c" * 64,
        })

    async def remind(self):
        self.write()
        self.assertEqual(1, await self.scan())
        self.enterContext(patch.object(chat, "_read_reminder_draft", AsyncMock(
            side_effect=lambda receipt: {**self.snapshot(receipt), "contentVersion": self.current_version},
        )))

    def context(self, *, actor=ACTOR, group=GROUP, quoted=False, text="确认"):
        memory = chat.ChatMemoryContext(platform="qq", user_id=actor, space_type="group", space_id=group)
        event = SimpleNamespace(
            get_plaintext=lambda: text, message=[SimpleNamespace(type="text")],
            reply=SimpleNamespace(message_id="reminder") if quoted else None,
            time=int(self.clock),
        )
        return chat.TurnContext(
            bot=SimpleNamespace(), event=event, platform="qq", user_id=actor,
            message_text=text, normalized_message_text=text, memory_context=memory,
            conv_key=memory.conversation_address, space_key=memory.conversation_address.space_key,
            reply_reference=chat.ReplyReferenceInfo(is_reply=quoted),
        )

    async def test_owner_bare_confirmation_routes_then_previews_and_submits(self):
        await self.remind()
        ctx = self.context()
        with patch.object(chat, "get_conversation_key", return_value=ctx.conv_key):
            self.assertTrue(chat._route_draft_reminder(ctx.bot, ctx.event))
        self.assertTrue(await chat._stage_draft_reminder_reply(ctx))
        self.assertEqual(2, self.tool.await_count)
        self.assertEqual(7, self.tool.await_args.args[1]["expected_content_version"])
        self.assertIn("已提交审核", ctx.response)
        self.assertTrue(await chat._stage_draft_reminder_reply(self.context(quoted=True)))
        self.assertEqual(2, self.tool.await_count)

    async def test_other_actor_or_group_cannot_confirm_and_quote_binds_owner(self):
        await self.remind()
        self.assertFalse(await chat._stage_draft_reminder_reply(self.context(actor="67890")))
        other = self.context(actor="67890", quoted=True)
        with patch.object(chat, "get_conversation_key", return_value=other.conv_key):
            self.assertTrue(chat._route_draft_reminder(other.bot, other.event))
        self.assertTrue(await chat._stage_draft_reminder_reply(other))
        self.assertEqual("这条提醒只能由草稿作者确认", other.response)
        self.assertFalse(await chat._stage_draft_reminder_reply(self.context(group="41", quoted=True)))
        self.tool.assert_not_awaited()
        self.finish.assert_awaited_once()
        self.assertTrue(await chat._stage_draft_reminder_reply(self.context(quoted=True)))
        self.assertEqual(2, self.tool.await_count)

    async def test_changed_version_reasks_with_an_exact_preview_ticket(self):
        await self.remind()
        ctx = self.context()
        self.current_version = self.preview_version = 8
        self.assertTrue(await chat._stage_draft_reminder_reply(ctx))
        self.tool.assert_awaited_once()
        self.assertIn("草稿内容已变化，请重新确认", ctx.response)
        pending = self.state.get(ctx.conv_key)
        self.assertTrue(chat._state.server_warning_ticket_is_complete(pending))
        self.assertEqual(8, pending.args["expected_content_version"])

    async def test_version_change_during_preview_reasks_without_submitting(self):
        await self.remind()
        ctx = self.context(quoted=True)
        self.preview_version = 8
        self.assertTrue(await chat._stage_draft_reminder_reply(ctx))
        self.tool.assert_awaited_once()
        self.assertIn("草稿内容已变化，请重新确认", ctx.response)

    async def test_newer_pending_confirmation_wins_bare_but_quote_targets_reminder(self):
        await self.remind()
        self.clock += 1
        ctx = self.context()
        self.assertTrue(self.state.set(ctx.conv_key, chat.PendingToolConfirm("keytao_create_phrase", {})))
        self.assertFalse(await chat._stage_draft_reminder_reply(ctx))
        self.tool.assert_not_awaited()
        self.assertTrue(await chat._stage_draft_reminder_reply(self.context(quoted=True)))
        self.assertEqual(2, self.tool.await_count)

    async def test_next_plain_message_spends_bare_confirmation_but_quote_still_works(self):
        await self.remind()
        ctx = self.context(text="先等等")
        with patch.object(chat, "get_conversation_key", return_value=ctx.conv_key):
            self.assertFalse(chat._route_draft_reminder(ctx.bot, ctx.event))
        self.assertFalse(await chat._stage_draft_reminder_reply(self.context()))
        self.tool.assert_not_awaited()
        self.assertTrue(await chat._stage_draft_reminder_reply(self.context(quoted=True)))
        self.assertEqual(2, self.tool.await_count)

    async def test_two_hours_expires_bare_and_quoted_confirmations(self):
        await self.remind()
        self.clock += 2 * 60 * 60
        self.assertFalse(await chat._stage_draft_reminder_reply(self.context()))
        ctx = self.context(quoted=True)
        self.assertTrue(await chat._stage_draft_reminder_reply(ctx))
        self.assertEqual("提醒已过期，请回复「提交」", ctx.response)
        self.tool.assert_not_awaited()

    async def test_missing_timestamp_is_expired_and_does_not_consume_confirmation(self):
        await self.remind()
        bare, quote = self.context(), self.context(quoted=True)
        del bare.event.time, quote.event.time
        with patch.object(chat, "get_conversation_key", return_value=bare.conv_key):
            self.assertFalse(chat._route_draft_reminder(bare.bot, bare.event))
            self.assertTrue(chat._route_draft_reminder(quote.bot, quote.event))
        self.assertFalse(await chat._stage_draft_reminder_reply(bare))
        self.assertTrue(await chat._stage_draft_reminder_reply(quote))
        self.assertEqual("提醒已过期，请回复「提交」", quote.response)
        self.tool.assert_not_awaited()
        self.assertTrue(await chat._stage_draft_reminder_reply(self.context(quoted=True)))
        self.assertEqual(2, self.tool.await_count)

    async def test_messages_before_delivery_cannot_spend_or_confirm_the_reminder(self):
        await self.remind()
        previous = self.context(text="之前的消息")
        previous.event.time = int(self.clock) - 1
        with patch.object(chat, "get_conversation_key", return_value=previous.conv_key):
            self.assertFalse(chat._route_draft_reminder(previous.bot, previous.event))
        old_confirm = self.context(quoted=True)
        old_confirm.event.time = previous.event.time
        self.assertTrue(await chat._stage_draft_reminder_reply(old_confirm))
        self.tool.assert_not_awaited()
        self.assertTrue(await chat._stage_draft_reminder_reply(self.context()))
        self.assertEqual(2, self.tool.await_count)

    async def test_quoted_legacy_commands_keep_the_exact_target_after_expiry(self):
        await self.remind()
        self.clock += 2 * 60 * 60
        clear = self.context(quoted=True, text="清空草稿")
        with patch.object(chat._chat_commands, "_perform_clear_current_draft", AsyncMock(
            return_value=chat.DraftActionResult("cleared"),
        )) as run:
            self.assertTrue(await chat._stage_draft_reminder_reply(clear))
        run.assert_awaited_once_with("qq", ACTOR, batch_id=BATCH, expected_content_version=7)
        submit = self.context(quoted=True, text="提交")
        self.assertTrue(await chat._stage_draft_reminder_reply(submit))
        self.assertIn("已提交审核", submit.response)
        self.assertEqual(2, self.tool.await_count)


class RecentWriteSubmitTests(unittest.IsolatedAsyncioTestCase):
    async def test_submit_after_own_write_uses_the_fresh_server_preview(self):
        from tests.helpers import (B18RealToolTests_seed, B18RealToolTests_setUp,
                                   B18RealToolTests_turn, TOOLS, b19_partial_reply_binding_DraftHTTP)

        B18RealToolTests_setUp(self)
        finish = self.replay.chat._finish_ai_chat_response.side_effect

        def acknowledge(*args):
            token = self.replay.commands.current_memory_context.set(self.replay.memory)
            try:
                self.replay.commands._acknowledge_delivered_draft_mutations(str(args[4]))
                finish(*args)
            finally:
                self.replay.commands.current_memory_context.reset(token)

        self.replay.chat._finish_ai_chat_response.side_effect = acknowledge
        self.http = b19_partial_reply_binding_DraftHTTP()
        self.enterContext(patch.object(TOOLS.http_client, "_keytao_client", self.http))
        B18RealToolTests_seed(self, ("招笑",), reorder=False)
        await B18RealToolTests_turn(self, "招笑 2，加入")
        receipt = self.replay.store.get(self.replay.author)
        self.assertEqual(self.http.version, receipt.args["_recent_content_version"])
        self.http.version += 1
        fresh_version = self.http.version
        await B18RealToolTests_turn(self, "提交")
        self.assertEqual(1, self.http.submissions, self.replay.last_response)
        submitted, = [body for path, body in self.http.writes if path.endswith("/submit")]
        self.assertEqual(fresh_version, submitted["expectedContentVersion"])
        self.assertEqual([], self.http.errors)
