"""S61 replay: 「加入并提交」 must submit, or ask with a ticket that survives delivery."""

import contextlib
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness.state import server_warning_ticket_is_complete


chat = harness.openai_chat_module
commands = harness.chat_commands_module
BATCH_ID = "s61-3fc424e6"
WORD = "户晨风"
CODE = "hjfo"
OTHER = {"action": "Create", "word": "旧词", "code": "qtxx", "type": "Phrase", "weight": 100}


class FakeKeytaoServer:
    """One draft batch with content-version CAS, previews and an all-or-nothing submit."""

    def __init__(self, existing_items=()):
        self.items = [dict(item) for item in existing_items]
        self.version = 3 if self.items else 0
        self.created = []
        self.submitted = False
        self.calls = []

    async def __call__(self, name, args, platform, user_id, **_kwargs):
        self.calls.append((name, copy.deepcopy(args)))
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            raise AssertionError(f"Unexpected external tool: {name}")
        return json.dumps(handler(args), ensure_ascii=False)

    def _keytao_create_phrase(self, args):
        if not args.get("confirmed"):
            return {
                "success": False, "requiresConfirmation": True, "batchId": BATCH_ID,
                "contentVersion": self.version, "warningDigest": "a" * 64,
                "warnings": [], "warnedCount": 0,
            }
        assert args["batch_id"] == BATCH_ID, args
        assert args["expected_content_version"] == self.version, args
        assert args["expected_warning_digest"] == "a" * 64, args
        assert not self.created, "a confirmed create was replayed twice"
        self.version += 1
        row = {
            "id": 7001, "action": "Create", "word": args["word"],
            "code": args["code"], "type": "Phrase", "weight": 100,
        }
        self.created.append(row)
        self.items.append({key: value for key, value in row.items() if key != "id"})
        return {
            "success": True, "batchId": BATCH_ID, "contentVersion": self.version,
            "pullRequestCount": len(self.created),
            "writtenItems": copy.deepcopy(self.created),
        }

    def _keytao_submit_batch(self, args):
        if not args.get("confirmed"):
            return {
                "success": False, "requiresConfirmation": True, "batchId": BATCH_ID,
                "contentVersion": self.version, "snapshotDigest": "c" * 64,
                "warningDigest": "d" * 64, "auditDigest": "e" * 64,
                "snapshotItems": copy.deepcopy(self.items), "warnings": [],
            }
        assert args["batch_id"] == BATCH_ID, args
        assert args["expected_content_version"] == self.version, args
        assert args["expected_server_snapshot_digest"] == "c" * 64, args
        assert args["expected_warning_digest"] == "d" * 64, args
        assert args["expected_audit_digest"] == "e" * 64, args
        assert not self.submitted, "the batch was submitted twice"
        self.submitted = True
        self.version += 1
        return {
            "success": True, "status": "Submitted", "batchId": BATCH_ID,
            "contentVersion": self.version,
        }

    def tool_names(self):
        return [name for name, _args in self.calls]


class FakeShiftServer(FakeKeytaoServer):
    """The same batch reached through a confirmed ranked-shift plan."""

    PLAN_ITEMS = [{"action": "Create", "word": WORD, "code": CODE, "type": "Phrase"}]

    def _keytao_shift_phrase_code(self, args):
        assert args["confirmed_plan_digest"] == "b" * 64, args
        assert not self.created, "a confirmed shift was replayed twice"
        self.version += 1
        row = {
            "id": 7101, "action": "Create", "word": WORD,
            "code": CODE, "type": "Phrase", "weight": 100,
        }
        self.created.append(row)
        self.items.append({key: value for key, value in row.items() if key != "id"})
        return {
            "success": True, "batchId": BATCH_ID, "contentVersion": self.version,
            "pullRequestCount": len(self.created),
            "writtenItems": copy.deepcopy(self.created),
            "shiftPlan": {
                "word": WORD, "targetCode": CODE, "shifted": [],
                "items": copy.deepcopy(self.PLAN_ITEMS),
            },
        }


class FakeBot:
    def __init__(self):
        self.messages = []

    async def send(self, **kwargs):
        self.messages.append(kwargs.get("message"))


class S61SubmitReplayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.actor = "s61-fixture-actor"
        self.platform = "qq"
        self.key = harness.ConversationAddress.group("qq", "865189947", self.actor)
        self.store = harness.MemoryConversationStateStore()
        self.coordinator = harness.DraftOperationCoordinator()
        self.memory_context = harness.ChatMemoryContext(
            platform="qq", user_id=self.actor, space_type="group",
            space_id="865189947", speaker_name="Rea",
        )

    @contextlib.contextmanager
    def patches(self, server):
        async def tool(name, args, platform, user_id, **kwargs):
            # Mirror the real dispatcher: every tool response is also the
            # trusted receipt the delivery boundary rebuilds its answer from.
            result_json = await server(name, args, platform, user_id, **kwargs)
            commands._capture_successful_draft_write_delivery(
                name, json.loads(result_json), platform, user_id,
            )
            return result_json

        patchers = (
            patch.object(chat, "conversation_state_store", self.store),
            patch.object(chat, "draft_operation_coordinator", self.coordinator),
            patch.object(chat, "call_tool_function", side_effect=tool),
            patch.object(chat, "remember_conversation", return_value=True),
            patch.object(chat, "schedule_memory_compaction"),
            patch.object(chat.memory_store, "capture_generation", return_value=object()),
            patch.object(chat.memory_store, "is_generation_current", return_value=True),
            patch.object(chat.memory_store, "record_tool_receipt"),
            patch.object(chat.history_store, "capture_generation", return_value=object()),
            patch.object(chat.history_store, "is_generation_current", return_value=True),
            patch.object(chat, "_classify_message_command_intent",
                         AsyncMock(side_effect=AssertionError("S61 cannot invoke a model"))),
            patch.object(chat, "_classify_simple_word_query_intent",
                         AsyncMock(side_effect=AssertionError("S61 cannot invoke a model"))),
        )
        with contextlib.ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            yield

    async def run_add_and_submit(self, server, platform="qq"):
        """Replay the production 「加入并提交」 background operation end to end."""
        self.platform = platform
        self.key = harness.ConversationAddress.group(platform, "865189947", self.actor)
        self.memory_context = harness.ChatMemoryContext(
            platform=platform, user_id=self.actor, space_type="group",
            space_id="865189947", speaker_name="Rea",
        )
        bot = FakeBot()
        token = chat._current_turn_message.set("加入并提交")
        try:
            with self.patches(server):
                operation = self.coordinator.begin(
                    self.key, "add_and_submit", word=WORD, code=CODE,
                )
                await chat._run_background_draft_operation(
                    operation,
                    lambda: commands._perform_add_to_draft_and_submit(
                        WORD, CODE, platform, self.actor,
                        needs_manual_review=True, auto_confirm=True,
                    ),
                    bot, SimpleNamespace(message_id=None), self.actor,
                    self.memory_context, "加入并提交", object(), object(),
                )
        finally:
            chat._current_turn_message.reset(token)
        self.assertEqual(len(bot.messages), 1, bot.messages)
        print(json.dumps({
            "scenario": "S61", "mode": "fixture/fake-tools", "platform": platform,
            "instruction": "加入并提交", "batchHeldOtherItems": len(server.items) > 1,
            "submitted": server.submitted, "toolCalls": server.tool_names(),
            "paidModelCalls": 0, "realProviderCalls": 0, "receipt": bot.messages[0],
        }, ensure_ascii=False), flush=True)
        return bot.messages[0]

    async def test_a_create_path_replays_submit_when_batch_holds_only_this_turn(self):
        server = FakeKeytaoServer()
        delivered = await self.run_add_and_submit(server)
        self.assertTrue(server.submitted, delivered)
        self.assertEqual(server.tool_names(), [
            "keytao_create_phrase", "keytao_create_phrase",
            "keytao_submit_batch", "keytao_submit_batch",
        ])
        self.assertIn(f"{WORD} → {CODE}", delivered)
        self.assertIn("已提交审核", delivered)
        self.assertEqual(delivered.count(f"https://keytao.rea.ink/batch/{BATCH_ID}"), 1, delivered)
        self.assertIsNone(self.coordinator.get(self.key))

    async def test_a_create_path_replay_maps_the_telegram_link(self):
        server = FakeKeytaoServer()
        delivered = await self.run_add_and_submit(server, platform="telegram")
        self.assertTrue(server.submitted, delivered)
        self.assertIn("已提交审核", delivered)
        self.assertEqual(
            delivered.count(f"https://keytao.vercel.app/batch/{BATCH_ID}"), 1, delivered,
        )

    async def run_shift_and_submit(self, server):
        """Replay the shift-then-submit path that shares the same submit stage."""
        state = harness.PendingToolConfirm(
            function_name="keytao_shift_phrase_code",
            args={
                "word": WORD, "target_code": CODE, "batch_id": "",
                "expected_content_version": 0, "confirmed_plan_digest": "b" * 64,
                "_submit_after": True,
            },
            confirmation_source="server_warning",
        )
        token = chat._current_turn_message.set("加入并提交")
        try:
            with self.patches(server):
                return await commands._execute_confirmed_tool(
                    state, self.platform, self.actor, self.key,
                    self.key.space_key, "Rea",
                )
        finally:
            chat._current_turn_message.reset(token)

    async def test_a_shift_path_replays_submit_when_batch_holds_only_this_turn(self):
        server = FakeShiftServer()
        reply = await self.run_shift_and_submit(server)
        self.assertTrue(server.submitted, reply)
        self.assertEqual(server.tool_names(), [
            "keytao_shift_phrase_code", "keytao_submit_batch", "keytao_submit_batch",
        ])
        self.assertIn(f"{WORD} → {CODE}", reply)
        self.assertIn("已提交审核", reply)
        self.assertIsNone(self.store.get_record(self.key))

    async def test_b_shift_path_extra_items_ask_keeps_a_conversation_ticket(self):
        server = FakeShiftServer([OTHER])
        reply = await self.run_shift_and_submit(server)
        self.assertFalse(server.submitted, reply)
        self.assertIn("旧词 → qtxx", reply)
        record = self.store.get_record(self.key)
        self.assertIsNotNone(record)
        self.assertEqual(record.state.function_name, "keytao_submit_batch")
        with self.patches(server):
            self.assertEqual(chat._enforce_advertised_reply_contract(reply, self.key), reply)
        # 「只提交 X」 is answered from the live ticket, without a model call.
        ctx = chat.TurnContext(
            bot=object(), event=object(), platform=self.platform, user_id=self.actor,
            normalized_message_text=f"只提交 {WORD}", conv_key=self.key,
            space_key=self.key.space_key,
            command_intent_for=AsyncMock(
                side_effect=AssertionError("a subset submit must not need the model"),
            ),
        )
        with self.patches(server):
            await chat._stage_execute_pending_state(ctx)
        self.assertIn("整批", ctx.response)
        self.assertFalse(server.submitted)
        self.assertIs(self.store.get_record(self.key), record)

    async def test_b_extra_items_ask_survives_delivery_with_a_live_ticket(self):
        server = FakeKeytaoServer([OTHER])
        delivered = await self.run_add_and_submit(server)
        self.assertFalse(server.submitted, delivered)
        self.assertEqual(server.tool_names(), [
            "keytao_create_phrase", "keytao_create_phrase", "keytao_submit_batch",
        ])
        # C: never a write-only receipt that stays silent about the submit.
        self.assertIn(f"{WORD} → {CODE}", delivered)
        self.assertIn("旧词 → qtxx", delivered)
        self.assertIn("确认", delivered)
        self.assertNotIn("当前没有可验证的可执行操作", delivered)
        # B: the ticket is real, owned by this actor, and the gate keeps the ask.
        operation = self.coordinator.get(self.key)
        self.assertIsNotNone(operation)
        self.assertEqual(operation.status, "awaiting_confirmation")
        self.assertEqual(operation.pending_state.function_name, "keytao_submit_batch")
        self.assertTrue(
            server_warning_ticket_is_complete(operation.pending_state)
        )
        with self.patches(server):
            self.assertEqual(chat._enforce_advertised_reply_contract(delivered, self.key), delivered)

    async def test_b_full_batch_assent_still_reaches_the_live_ticket(self):
        """A plain submit command is assent to the whole batch, not a subset ask."""
        for message in ("就提交吧", "确认"):
            with self.subTest(message=message):
                self.setUp()
                server = FakeKeytaoServer([OTHER])
                await self.run_add_and_submit(server)
                operation = self.coordinator.get(self.key)
                self.assertEqual(
                    commands.subset_submit_refusal(message, operation.pending_state), "",
                )
                ctx = chat.TurnContext(
                    bot=object(), event=object(), platform=self.platform,
                    user_id=self.actor, normalized_message_text=message,
                    conv_key=self.key, space_key=self.key.space_key,
                    command_intent_for=AsyncMock(
                        side_effect=AssertionError("assent must not need the model"),
                    ),
                )

                scheduled = []
                with (
                    self.patches(server),
                    patch.object(
                        chat, "_schedule_background_draft_operation",
                        side_effect=lambda _op, factory, *a, **k: (
                            scheduled.append(factory) or True
                        ),
                    ),
                ):
                    handled = await chat._stage_arbitrate_active_operation(ctx)
                    self.assertEqual(len(scheduled), 1, ctx.response)
                    await scheduled[0]()
                self.assertTrue(handled)
                self.assertTrue(server.submitted, message)

    async def test_c_recent_own_write_pointer_does_not_disarm_the_invariant(self):
        """A local 'recent own write' pointer is not an asked submit question."""
        server = FakeKeytaoServer([OTHER])
        self.store.set(self.key, harness.PendingToolConfirm(
            function_name="keytao_submit_batch",
            args={
                "batch_id": BATCH_ID, "_recent_own_write": True,
                "_recent_batch_ids": [BATCH_ID],
            },
            confirmation_source="local_preview",
        ))
        token = chat._current_turn_message.set("加入并提交")
        try:
            with self.patches(server):
                delivery_token = commands.current_draft_delivery_claims.set([])
                try:
                    await commands._perform_add_to_draft_and_submit(
                        WORD, CODE, self.platform, self.actor,
                        needs_manual_review=True, auto_confirm=True,
                    )
                    prepared = chat._prepare_user_facing_reply(
                        "✅ 操作已完成", self.memory_context,
                    )
                finally:
                    commands.current_draft_delivery_claims.reset(delivery_token)
        finally:
            chat._current_turn_message.reset(token)
        self.assertFalse(server.submitted)
        self.assertIn("尚未提交", prepared)

    async def test_c_a_refusal_to_submit_gets_no_submit_disclaimer(self):
        server = FakeKeytaoServer()
        token = chat._current_turn_message.set("先别提交")
        try:
            with self.patches(server):
                delivery_token = commands.current_draft_delivery_claims.set([])
                try:
                    await commands._perform_add_to_draft_and_submit(
                        WORD, CODE, self.platform, self.actor,
                        needs_manual_review=True, auto_confirm=False,
                    )
                    prepared = chat._prepare_user_facing_reply(
                        "✅ 操作已完成", self.memory_context,
                    )
                finally:
                    commands.current_draft_delivery_claims.reset(delivery_token)
        finally:
            chat._current_turn_message.reset(token)
        self.assertNotIn("尚未提交", prepared)

    async def test_b_scope_notice_never_costs_a_confirmable_batch_its_ticket(self):
        """A prompt just under the cap stays confirmable once the notice is added."""
        server = FakeKeytaoServer([OTHER])
        base = "x" * (commands.MAX_REPLACE_CONFIRMATION_CHARS - 200)
        with (
            self.patches(server),
            patch.object(commands, "_format_server_warning_confirmation",
                         return_value=base),
        ):
            result = await commands._perform_submit_current_draft(
                self.platform, self.actor, batch_id=BATCH_ID, auto_confirm=True,
                authorized_items=[{"action": "Create", "word": WORD, "code": CODE}],
            )
        self.assertIsNotNone(result.pending_state)
        self.assertIn("旧词 → qtxx", result.text)
        self.assertGreater(len(result.text), len(base))
        self.assertNotIn("提交内容过长", result.text)

    async def test_b_confirm_after_the_ask_submits_the_whole_batch(self):
        server = FakeKeytaoServer([OTHER])
        await self.run_add_and_submit(server)
        operation = self.coordinator.get(self.key)
        bot = FakeBot()
        with self.patches(server):
            result = await commands._perform_active_operation_confirmation(
                operation, self.platform, self.actor,
            )
        self.assertTrue(server.submitted, result.text)
        self.assertTrue(result.success)
        self.assertIn("已提交审核", result.text)
        self.assertEqual(bot.messages, [])

    async def test_b_subset_submit_request_is_refused_truthfully(self):
        server = FakeKeytaoServer([OTHER])
        await self.run_add_and_submit(server)
        operation = self.coordinator.get(self.key)
        ctx = chat.TurnContext(
            bot=object(), event=object(), platform=self.platform, user_id=self.actor,
            normalized_message_text=f"只提交 {WORD}", conv_key=self.key,
            space_key=self.key.space_key,
            command_intent_for=AsyncMock(
                side_effect=AssertionError("a subset submit must not need the model"),
            ),
        )
        with self.patches(server), patch.object(chat, "_finish_ai_chat_matcher", AsyncMock()):
            handled = await chat._stage_arbitrate_active_operation(ctx)
        self.assertTrue(handled)
        self.assertFalse(server.submitted)
        self.assertIn("整批", ctx.response)
        self.assertIn(WORD, ctx.response)
        self.assertIn("查看草稿", ctx.response)
        # The ticket stays usable after the refusal.
        self.assertIs(self.coordinator.get(self.key), operation)
        with self.patches(server):
            self.assertEqual(chat._enforce_advertised_reply_contract(ctx.response, self.key), ctx.response)

    async def test_c_write_only_receipt_never_stays_silent_about_the_submit(self):
        """A summary that loses the submit still reports the true submit status."""
        server = FakeKeytaoServer([OTHER])
        bot = FakeBot()
        token = chat._current_turn_message.set("加入并提交")
        try:
            with self.patches(server):
                operation = self.coordinator.begin(
                    self.key, "add_and_submit", word=WORD, code=CODE,
                )
                delivery_token = commands.current_draft_delivery_claims.set([])
                try:
                    await commands._perform_add_to_draft_and_submit(
                        WORD, CODE, self.platform, self.actor,
                        needs_manual_review=True, auto_confirm=True,
                    )
                    self.coordinator.finish(self.key, operation.operation_id)
                    prepared = chat._prepare_user_facing_reply(
                        "✅ 操作已完成", self.memory_context,
                    )
                finally:
                    commands.current_draft_delivery_claims.reset(delivery_token)
        finally:
            chat._current_turn_message.reset(token)
        self.assertIn(f"{WORD} → {CODE}", prepared)
        self.assertIn("尚未提交", prepared)
        self.assertNotIn("已提交审核", prepared)
        self.assertEqual(bot.messages, [])


if __name__ == "__main__":
    unittest.main()
