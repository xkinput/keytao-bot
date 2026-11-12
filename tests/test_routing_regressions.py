"""Topic regressions: routing regressions."""

import unittest as _stage_t_unittest
from tests.helpers import (
    AgentRequestContext,
    ArtifactRecorder,
    AsyncMock,
    B5ReplySafetyTests_context,
    B5ReplySafetyTests_setUp,
    B6ReminderIntegrationTests_setUp,
    B6ReminderTests_scan,
    B6ReminderTests_setUp,
    B6ReminderTests_snapshot,
    B6ReminderTests_write,
    B6SubmitTests_invoke,
    B6SubmitTests_setUp,
    B6SubmitTests_submit,
    B7BlockedToolLoopTests_orchestrator,
    B7BlockedToolLoopTests_policy_message,
    B7BlockedToolLoopTests_run_loop,
    B7PhrasingRouteTests_answer,
    B7PhrasingRouteTests_assert_recodes,
    B7PhrasingRouteTests_asyncSetUp,
    B7PhrasingRouteTests_asyncTearDown,
    B7PhrasingRouteTests_call,
    B7PhrasingRouteTests_result,
    B7PhrasingRouteTests_review_word,
    B7PhrasingRouteTests_row,
    B7PhrasingRouteTests_stage,
    B7QueryRouteTests_assert_finished,
    B7QueryRouteTests_assert_preserved,
    B7QueryRouteTests_asyncSetUp,
    B7QueryRouteTests_asyncTearDown,
    B7QueryRouteTests_context,
    B7QueryRouteTests_lookup,
    B7QueryRouteTests_set_model,
    B7QueryRouteTests_ticket,
    B8CancelTests_assert_unbound_reply,
    B8CancelTests_offer,
    B8CancelTests_replay,
    B9CandidateTests_asyncSetUp,
    B9CandidateTests_asyncTearDown,
    B9CandidateTests_lookup,
    B9CandidateTests_offer,
    B9LookupTests_asyncSetUp,
    B9LookupTests_prepare,
    B9LookupTests_row,
    B9LookupTests_tool,
    B9ParallelTests_prepare,
    B9ParallelTests_setUp,
    B9ParallelTests_tool,
    B9PronunciationEvidenceTests_setUp,
    B9PronunciationEvidenceTests_timeout_types,
    DAY_SECONDS,
    DraftMutationClaimStore,
    DraftReminderStore,
    ExitStack,
    FakeClient,
    FreshSelectionTests_runtime,
    GeneralReplyTests_reply,
    MODEL_REPLY,
    MemoryConversationStateStore,
    NARRATION,
    NO_PLAN,
    Path,
    PendingAddWord,
    PendingToolConfirm,
    QUOTE,
    QueryPrecedenceTests_delivered,
    QueryPrecedenceTests_tool,
    REAL_CORE,
    Replay,
    SHANGHAI,
    ScenarioAssertionError,
    SimpleNamespace,
    SmallTalkTests_assert_natural,
    SmallTalkTests_delivered,
    SmallTalkTests_setUp,
    SmallTalkTests_tool,
    StaleTicketReleaseTests_assert_fresh_request,
    StaleTicketReleaseTests_replay,
    ToolContext,
    UNBOUND_BINDING_PRECHECK_NOTICE,
    _assert_s56_advertised_reply_closure,
    advertised_command_suggestions,
    ast,
    asyncio,
    b17_rest_recommended_routing,
    begin_turn_metrics,
    closing,
    copy,
    create_pending_trusted_word_record,
    current_turn_id,
    datetime,
    end_turn_metrics,
    enforce_conversational_reply,
    harness,
    json,
    logging,
    os,
    patch,
    pending,
    pending_duplicate_confirmation_state,
    policy_agent,
    policy_call,
    query,
    raw_operation,
    review,
    s55_failure_boundary_chat,
    s70_smalltalk_chat,
    server_warning_pending_state,
    set_turn_flow,
    snapshot,
    socket,
    sqlite3,
    sys,
    tempfile,
    ticket,
    timedelta,
    turn_failure_reply,
    unittest,
    urlsplit,
    vote_fixtures,
)
from tests.helpers import B5DeleteTests


# Source: test_b6_policy.py
class B6ReminderTests(unittest.IsolatedAsyncioTestCase):
    setUp = B6ReminderTests_setUp

    write = B6ReminderTests_write

    snapshot = B6ReminderTests_snapshot

    scan = B6ReminderTests_scan

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

# Source: test_b6_policy.py
class B6SubmitTests(unittest.IsolatedAsyncioTestCase):
    setUp = B6SubmitTests_setUp

    invoke = B6SubmitTests_invoke

    submit = B6SubmitTests_submit

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
        self.assertIn("之前：驴友 llyd", delivered)
        self.assertIn("本轮：屡有 llydav", delivered)
        self.assertIn("✅ 已提交审核，投票 0/3（引用本条回复「同意」）", delivered)

# Source: test_b6_policy.py
class B6ReminderIntegrationTests(unittest.IsolatedAsyncioTestCase):
    setUp = B6ReminderIntegrationTests_setUp

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
                self.assertEqual(await store.scan(B6ReminderTests_snapshot, send, now=now), 1)
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

# Source: test_b6_policy.py
class B7BlockedToolLoopTests(unittest.IsolatedAsyncioTestCase):
    policy_message = B7BlockedToolLoopTests_policy_message

    orchestrator = B7BlockedToolLoopTests_orchestrator

    run_loop = B7BlockedToolLoopTests_run_loop

    async def test_second_policy_block_stops_including_cached_duplicates(self):
        for second_word in ("纯音乐", "我送给"):
            with self.subTest(second_word=second_word):
                calls = [("keytao_shift_phrase_code", {"word": word, "target_code": "wsgu"})
                         for word in ("我送给", second_word, "皇太极")]
                agent, create, execute = self.orchestrator(calls)
                reply, termination = await self.run_loop(agent)
                self.assertEqual(reply, self.policy_message)
                self.assertEqual(termination["reason"], "policy_tool_runaway")
                self.assertEqual(create.await_count, 2)
                self.assertEqual(execute.await_count, 1 if second_word == "我送给" else 2)

    async def test_existing_intent_mapping_groups_different_tools(self):
        agent, create, execute = self.orchestrator([
            ("keytao_create_phrase", {"word": "我送给", "code": "wsgu"}),
            ("keytao_batch_add_to_draft", {"items": [{"word": "纯音乐", "code": "jyya"}]}),
        ])
        reply, termination = await self.run_loop(agent)
        self.assertEqual(reply, self.policy_message)
        self.assertEqual(termination["reason"], "policy_tool_runaway")
        self.assertEqual((create.await_count, execute.await_count), (2, 2))

    async def test_policy_block_counts_reset_for_each_turn(self):
        agent, create, execute = self.orchestrator([
            ("keytao_shift_phrase_code", {"word": "我送给", "target_code": "wsgu"}),
            ("keytao_shift_phrase_code", {"word": "纯音乐", "target_code": "jyya"}),
        ] * 2)
        for _ in range(2):
            reply, termination = await self.run_loop(agent)
            self.assertEqual(reply, self.policy_message)
            self.assertEqual(termination["reason"], "policy_tool_runaway")
        self.assertEqual((create.await_count, execute.await_count), (4, 4))

    async def test_one_block_in_each_family_does_not_stop_the_turn(self):
        agent, create, execute = self.orchestrator([
            ("keytao_shift_phrase_code", {"word": "我送给", "target_code": "wsgu"}),
            ("keytao_remove_draft_item", {"pr_id": 1}),
        ])
        reply, termination = await self.run_loop(agent)
        self.assertEqual(reply, "核验已结束。")
        self.assertNotIn("reason", termination)
        self.assertEqual((create.await_count, execute.await_count), (3, 2))

    async def test_locally_rejected_confirmation_counts_as_a_policy_block(self):
        incomplete = {"success": False, "requiresConfirmation": True, "message": "Offline incomplete plan"}
        agent, create, execute = self.orchestrator([
            ("keytao_shift_phrase_code", {"word": "我送给", "target_code": "wsgu"}),
            ("keytao_shift_phrase_code", {"word": "纯音乐", "target_code": "jyya"}),
        ], [incomplete, incomplete])
        with patch.object(agent, "_save_pending_tool_confirm", return_value=False):
            reply, termination = await self.run_loop(agent)
        self.assertIn("待确认内容不完整，本次未执行", reply)
        self.assertEqual(termination["reason"], "policy_tool_runaway")
        self.assertEqual((create.await_count, execute.await_count), (2, 2))

    async def test_stopping_keeps_earlier_successful_write_receipt(self):
        from keytao_bot.harness.orchestrator import AgentRequestContext
        blocked = {"success": False, "policyBlocked": True, "blockReason": "binding_incomplete",
                   "message": self.policy_message}
        agent, create, execute = self.orchestrator([
            ("keytao_create_phrase", {"word": "栗子", "code": "lizi"}),
            ("keytao_shift_phrase_code", {"word": "我送给", "target_code": "wsgu"}),
            ("keytao_shift_phrase_code", {"word": "纯音乐", "target_code": "jyya"}),
        ], [{"success": True, "batchId": "b7-batch", "batchUrl": "https://offline.invalid/b7-batch"},
            blocked, blocked])
        reply = await agent.run(
            "修改编码：我送给为wsgu，纯音乐为jyya",
            AgentRequestContext(platform="qq", user_id="b7-loop", mutations_allowed=True),
        )
        self.assertIn("✅ 已加入草稿：栗子 lizi", reply)
        self.assertIn("当前编码与已核验的词条不一致", reply)
        self.assertIn("https://offline.invalid/b7-batch", reply)
        self.assertEqual((create.await_count, execute.await_count), (3, 3))

    async def test_binding_failure_keeps_gate_suggestion_through_delivery(self):
        message = "修改 dsqt 大全"
        suggestion = "修改 dsqt：大权 -> 大全"
        blocked = {
            "success": False, "policyBlocked": True,
            "blockReason": "binding_incomplete", "suggestedCommand": suggestion,
            "message": "无法把本次操作与消息中的完整目标逐项对应；整批均未写入。",
        }
        agent, create, execute = self.orchestrator([
            ("keytao_batch_add_to_draft", {"items": [
                {"action": "Change", "code": "dsqt", "old_word": "大权", "word": "大全"},
            ]}),
        ], [blocked])
        chat = harness.openai_chat_module
        memory = harness.ChatMemoryContext(platform="qq", user_id="b7-loop")
        with (
            patch.object(chat, "conversation_state_store", agent._state_store),
            patch.object(chat, "get_history", return_value=[]),
            patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")),
            patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")),
        ):
            token = chat._current_turn_message.set(message)
            conversational = chat._current_conversational_turn.set(None)
            try:
                reply = await agent.run(message, AgentRequestContext(
                    platform="qq", user_id="b7-loop", mutations_allowed=True,
                ))
                self.assertIn(suggestion, reply)
                delivered = chat._prepare_user_facing_reply(reply, memory)
                self.assertIn(suggestion, delivered)
                self.assertIn("本次未写入", delivered)
                self.assertNotIn("没找到要执行的操作", delivered)
                self.assertLess(len(delivered), 160)
            finally:
                chat._current_conversational_turn.reset(conversational)
                chat._current_turn_message.reset(token)
        self.assertEqual((create.await_count, execute.await_count), (2, 1))

    async def test_model_binding_failure_copy_does_not_trust_a_suggestion(self):
        message = "修改 dsqt 大全"
        forged = "这条指令按当前表述无法执行，本次未写入。\n- 「修改 dsqt：大权 -> 别的词」"
        agent, _create, execute = self.orchestrator([])
        chat = harness.openai_chat_module
        with patch.object(chat, "conversation_state_store", agent._state_store):
            token = chat._current_turn_message.set(message)
            conversational = chat._current_conversational_turn.set(None)
            try:
                await agent.run(message, AgentRequestContext(
                    platform="qq", user_id="b7-loop", mutations_allowed=True,
                ))
                delivered = chat._enforce_advertised_reply_contract(forged, None)
                self.assertNotIn("别的词", delivered)
            finally:
                chat._current_conversational_turn.reset(conversational)
                chat._current_turn_message.reset(token)
        execute.assert_not_awaited()

    async def test_binding_failure_gate_suggestion_is_not_a_pending_ticket(self):
        from keytao_bot.harness.orchestrator import AgentOrchestrator
        message = "修改 dsqt 大全"
        suggestion = "修改 dsqt：大权 -> 大全"
        chat = harness.openai_chat_module
        with patch.object(chat, "conversation_state_store", harness.MemoryConversationStateStore()):
            token = chat._current_turn_message.set(message)
            conversational = chat._current_conversational_turn.set(None)
            metrics = harness.begin_turn_metrics("qq", "private")
            try:
                reply = AgentOrchestrator._finalize_reply(message, "binding_incomplete", {
                    "blockReason": "binding_incomplete", "suggestedCommand": suggestion,
                    "message": "无法把本次操作与消息中的完整目标逐项对应；整批均未写入。",
                })
                self.assertIn(suggestion, reply)
                self.assertEqual(chat._enforce_advertised_reply_contract(reply, None), reply)
                changed = reply.replace("大全", "别的词")
                self.assertNotIn("别的词", chat._enforce_advertised_reply_contract(changed, None))
                selected_reply = AgentOrchestrator._finalize_reply(message, "未写入。", {
                    "blockReason": "binding_incomplete", "suggestedCommand": suggestion,
                    "liveCandidateSelected": True, "message": "当前选择未通过核验，本次未写入。",
                })
                self.assertIn(suggestion, selected_reply)
                self.assertEqual(chat._enforce_advertised_reply_contract(selected_reply, None), selected_reply)
                next_turn = harness.begin_turn_metrics("qq", "private")
                try:
                    self.assertNotIn(suggestion, chat._enforce_advertised_reply_contract(reply, None))
                finally:
                    harness.end_turn_metrics(next_turn)
            finally:
                harness.end_turn_metrics(metrics)
                chat._current_conversational_turn.reset(conversational)
                chat._current_turn_message.reset(token)

    async def test_duplicate_copy_does_not_expose_internal_fragments(self):
        from keytao_bot.harness.tools import ToolContext, project_tool_result_for_model
        from keytao_bot.plugins.chat_render import _reply_has_internal_fragment
        for tool_name, succeeded in (("keytao_create_phrase", True), ("keytao_create_phrase", False),
                                     ("keytao_lookup_by_word", True)):
            with self.subTest(tool_name=tool_name, succeeded=succeeded):
                agent, _, execute = self.orchestrator([])
                args = {"word": "我送给", "code": "wsg"}
                seen = {(tool_name, json.dumps(args, sort_keys=True, ensure_ascii=False)): (1, succeeded)}
                result = await agent._call_tool_once(tool_name, args, ToolContext(), seen)
                message = json.loads(project_tool_result_for_model(tool_name, result, args))["message"]
                self.assertFalse(_reply_has_internal_fragment(message), message)
                execute.assert_not_awaited()

# Source: test_b5_prod_defects.py
class B5ReplySafetyTests(unittest.IsolatedAsyncioTestCase):
    setUp = B5ReplySafetyTests_setUp

    context = B5ReplySafetyTests_context

    async def test_prod_confirm_after_unrelated_turns_and_expiry_never_calls_model(self):
        for quoted in (None, "bot", "human"):
            with self.subTest(quoted=quoted):
                self.tool.return_value = json.dumps({
                    "success": True, "batchId": "1aa01eb8", "status": "Draft",
                    "items": [{"word": "屡有", "code": "llydav", "action": "Create"}],
                })
                ctx = self.context(quoted=quoted == "bot", addressed=quoted != "bot")
                if quoted == "human":
                    ctx.reply_reference = harness.ReplyReferenceInfo(
                        is_reply=True, is_to_bot=False, text="是否继续提交？回复「确认」。",
                    )
                if quoted == "bot":
                    self.chat.get_history.return_value = [{
                        "role": "assistant", "content": ctx.reply_reference.text,
                    }]
                else:
                    self.chat.get_history.return_value = []
                with patch.object(self.chat, "AsyncOpenAI") as model:
                    self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
                model.assert_not_called()
                self.assertIn("没有可执行的确认记录" if quoted == "bot" else "当前没有待执行的操作", ctx.response)
                self.assertIn("引用原提议", ctx.response)
                self.assertEqual(len(ctx.response.splitlines()), 1)
                self.assertNotIn("已提交", ctx.response)
                self.tool.assert_not_awaited()

    async def test_foreign_live_confirmation_quote_never_echoes_or_consumes_ticket(self):
        prompt = "拟提交「私有提议」→ syty。\n是否继续提交？回复「确认」。"
        for scenario in ("other_owner", "other_space", "foreign_mention", "unbound", "other_space_active"):
            with self.subTest(scenario=scenario):
                other_space = scenario in {"other_space", "other_space_active"}
                source = harness.ConversationAddress.group(
                    "qq", "another-group" if other_space else "b5",
                    "abang" if other_space else "another-owner",
                )
                self.assertTrue(self.store.set(
                    source, PendingToolConfirm("keytao_submit_batch", {"batch_id": "other-batch"}),
                    space_key=source.space_key,
                ))
                ticket = self.store.get_record(source)
                original = copy.deepcopy(ticket)
                active = None
                if scenario == "other_space_active":
                    active = self.chat.draft_operation_coordinator.begin(source, "submit")
                    self.assertIsNotNone(active)
                    self.assertTrue(self.chat.draft_operation_coordinator.mark_awaiting_confirmation(
                        source, active.operation_id, ticket.state, prompt,
                    ))
                    original_active = copy.deepcopy(active)
                ctx = self.context("好", addressed=False)
                ctx.reply_reference = harness.ReplyReferenceInfo(
                    is_reply=True, is_to_bot=True, text=prompt,
                    mentioned_user_ids=() if scenario == "unbound" else (source.actor_id,),
                )
                histories = {source: [{"role": "assistant", "content": prompt}]}
                if scenario == "foreign_mention":
                    histories[ctx.conv_key] = histories[source]
                with (
                    patch.object(self.chat, "get_history", side_effect=lambda key: histories.get(key, [])),
                    patch.object(self.chat, "AsyncOpenAI") as model,
                    patch.object(self.chat, "_stage_understand_operation", AsyncMock()) as semantic,
                    patch.object(self.chat, "get_ai_response_core", AsyncMock()) as response_model,
                ):
                    self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
                self.assertEqual(ctx.response, "这条提议不是发给你的，本次未执行。")
                self.assertEqual(self.chat._prepare_user_facing_reply(ctx.response, self.memory), ctx.response)
                self.assertEqual(len(ctx.response.splitlines()), 1)
                self.assertIs(self.store.get_record(source), ticket)
                self.assertEqual(ticket, original)
                if active is not None:
                    self.assertIs(self.chat.draft_operation_coordinator.get(source), active)
                    self.assertEqual(active, original_active)
                self.assertIsNone(self.store.get_record(ctx.conv_key))
                self.tool.assert_not_awaited()
                model.assert_not_called()
                semantic.assert_not_awaited()
                response_model.assert_not_awaited()

    async def test_all_closed_affirmatives_without_draft_use_fixed_reply(self):
        other = harness.ConversationAddress.group("qq", "b5", "another-owner")
        self.assertTrue(self.store.set(
            other, PendingToolConfirm("keytao_submit_batch", {"batch_id": "other-batch"}),
            space_key=other.space_key,
        ))
        other_ticket = self.store.get_record(other)
        for message in b17_rest_recommended_routing.AFFIRMATIVE_REPLY_TEXTS:
            with self.subTest(message=message):
                ctx = self.context(message)
                with patch.object(self.chat, "AsyncOpenAI") as model:
                    self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
                model.assert_not_called()
                self.assertEqual(ctx.response, "当前没有待执行的操作；本次未写入。需要引用原提议确认后重新核对。")
                self.tool.assert_not_awaited()
                self.assertIs(self.store.get_record(other), other_ticket)

    async def test_unaddressed_affirmative_and_question_are_not_stale_controls(self):
        for message, addressed in (("确认", False), ("确认？", True)):
            self.assertFalse(await self.chat._try_stale_affirmative_reply(
                self.context(message, addressed=addressed),
            ))
        self.tool.assert_not_awaited()

    def test_general_fake_success_claims_are_replaced_without_receipt(self):
        for claim in ("已提交", "已成功提交", "提交成功", "审核中", "已加入词库", "已写入", "已删除", "审核通过", "已批准", "批准成功"):
            with self.subTest(claim=claim):
                self.assertEqual(
                    self.chat._prepare_user_facing_reply("✅ 批次" + claim + "。", self.memory),
                    "本次没有可核验的对应操作成功回执，无法确认操作已完成；请查看草稿核对。",
                )

    def test_real_submit_receipt_keeps_claim_and_other_actor_receipt_does_not(self):
        receipt = {
            "platform": "qq", "platformId": "abang", "operationKind": "draft_submit",
            "batchId": "offline-batch", "toolName": "keytao_submit_batch",
        }
        self.chat.current_draft_delivery_claims.set([receipt])
        reply = self.chat._prepare_user_facing_reply("✅ 批次已提交审核。", self.memory)
        self.assertIn("✅ 批次已提交审核。", reply)
        self.assertNotIn("无法确认", reply)
        self.chat.current_draft_delivery_claims.set([{**receipt, "platformId": "someone-else"}])
        self.assertIn("没有可核验", self.chat._prepare_user_facing_reply("已提交", self.memory))

    def test_submit_receipt_does_not_authorize_dictionary_or_delete_claims(self):
        receipt = {
            "platform": "qq", "platformId": "abang", "operationKind": "draft_submit",
            "batchId": "offline-batch", "toolName": "keytao_submit_batch",
        }
        self.chat.current_draft_delivery_claims.set([receipt])
        for claim in ("已加入词库", "已删除"):
            self.assertIn("没有可核验", self.chat._prepare_user_facing_reply(claim, self.memory))
        self.chat.current_draft_delivery_claims.set([{**receipt, "autoApproved": True}])
        self.assertIn("已加入词库", self.chat._prepare_user_facing_reply("已加入词库", self.memory))

    def test_negative_submit_status_is_not_a_success_claim(self):
        self.assertEqual(
            self.chat._prepare_user_facing_reply("本轮只写入草稿，尚未提交审核。", self.memory),
            "本轮只写入草稿，尚未提交审核。",
        )

    def test_status_lookup_and_preference_replies_are_not_current_action_claims(self):
        for reply in (
            "你最近的批次已提交，目前审核中，请耐心等待。",
            "「知更鸟」已加入词库，编码 zgn。",
            "好的，已更新你的称呼偏好。",
            "审核中。",
        ):
            with self.subTest(reply=reply):
                self.assertEqual(self.chat._prepare_user_facing_reply(reply, self.memory), reply)

    def test_status_reply_does_not_authorize_an_appended_current_action_claim(self):
        for claim in ("批次已提交。", "本次已删除草稿条目。", "我已为你重新提交。", "好的，已提交。"):
            with self.subTest(claim=claim):
                reply = "你最近的批次已提交，目前审核中，请耐心等待。\n" + claim
                self.assertIn("没有可核验", self.chat._prepare_user_facing_reply(reply, self.memory))

    def test_successful_submit_without_batch_id_survives_delivery(self):
        self.chat._chat_commands._capture_successful_draft_write_delivery(
            "keytao_submit_batch", {"success": True}, "qq", "abang",
        )
        reply = "✅ 批次已提交审核。"
        self.assertEqual(self.chat._prepare_user_facing_reply(reply, self.memory), reply)

    async def test_general_model_fake_submit_is_guarded_at_delivery(self):
        ctx = self.context("刚才那批处理完了吗？")
        with (
            patch.object(self.chat, "build_reply_context", AsyncMock(return_value="")),
            patch.object(self.chat, "get_ai_response_core", AsyncMock(return_value="✅ 批次已提交，审核中。")) as model,
        ):
            self.assertFalse(await self.chat._stage_generate_ai_response(ctx))
        model.assert_awaited_once()
        self.assertIn("没有可核验", self.chat._prepare_user_facing_reply(ctx.response, self.memory))

    def test_verified_historical_pending_reminder_survives_without_current_submit(self):
        item = {
            "source": "submitted", "batchId": "offline", "batchUrl": "https://keytao.test/batch/offline",
            "batchStatus": "Submitted", "itemStatus": "Pending", "action": "Create",
            "word": "开团", "code": "khtt", "type": "Phrase",
        }
        self.chat._chat_commands._capture_successful_draft_write_delivery(
            "keytao_pending_items_by_words", {"success": True, "complete": True, "items": [item]},
            "qq", "abang",
        )
        prompt = self.chat._chat_commands.prepend_pending_word_reminders("可以查看草稿核对。", [item])
        prepared = self.chat._prepare_user_facing_reply(prompt, self.memory)
        self.assertIn("审核中", prepared)
        self.assertEqual(self.chat._prepare_user_facing_reply(prepared, self.memory), prepared)
        self.assertNotIn("没有可核验", self.chat._prepare_user_facing_reply(prompt, self.memory))
        self.assertIn("没有可核验", self.chat._prepare_user_facing_reply(prompt + "\n批次已提交。", self.memory))
        self.chat.current_draft_delivery_claims.set([])
        self.assertEqual(self.chat._prepare_user_facing_reply(prompt, self.memory), prepared)

    def test_backed_duplicate_prompt_describes_existing_review_without_claiming_new_submit(self):
        state = pending_duplicate_confirmation_state(
            PendingToolConfirm("keytao_create_phrase", {"word": "开团", "code": "khtt", "preview_only": True}),
            {"pendingDuplicateConfirmation": True, "requiresConfirmation": True, "pendingItems": [{
                "source": "submitted", "batchId": "offline", "batchUrl": "https://offline.invalid/batch/offline",
                "batchStatus": "Submitted", "itemStatus": "Pending", "action": "Create",
                "word": "开团", "code": "khtt", "type": "Phrase",
            }]},
        )
        self.assertTrue(self.store.set(self.memory.conversation_address, state))
        prompt = self.chat._chat_render._format_pending_duplicate_confirmation(state)
        self.assertIn("该词已在审核中", prompt)
        expected = self.chat.render_platform_public_links(self.chat.strip_warning_count_copy(prompt), "qq")
        self.assertEqual(self.chat._prepare_user_facing_reply(prompt, self.memory), expected)
        self.assertEqual(self.chat._prepare_user_facing_reply(expected, self.memory), expected)
        self.assertIn("确认再提交", self.chat._prepare_user_facing_reply("审词：读音 kai tuan\n" + prompt, self.memory))
        self.assertIn("没有可核验", self.chat._append_unreported_submit_status(
            self.chat.strip_warning_count_copy(prompt) + "\n批次已提交。", [], self.memory.conversation_address,
        ))
        self.assertNotIn("批次已提交", self.chat._prepare_user_facing_reply(prompt + "\n批次已提交。", self.memory))

    async def test_fixed_stale_reply_survives_actual_delivery_preparation(self):
        ctx = self.context()
        self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
        token = self.chat._current_turn_message.set("确认")
        try:
            self.assertEqual(self.chat._prepare_user_facing_reply(ctx.response, self.memory), ctx.response)
        finally:
            self.chat._current_turn_message.reset(token)

    async def test_real_community_approval_keeps_dictionary_success_claim(self):
        fixture = vote_fixtures.VoteTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.addCleanup(fixture.tmp.cleanup)
        await fixture.cast("one", "甲")
        await fixture.cast("two", "乙")
        memory = harness.ChatMemoryContext(
            platform="qq", user_id="three", space_type="group", space_id="group", speaker_name="丙",
        )
        ctx = self.chat.TurnContext(
            bot=SimpleNamespace(), event=SimpleNamespace(), platform="qq", user_id="three",
            message_text="同意", normalized_message_text="同意", memory_context=memory, owner_label="丙",
            reply_reference=harness.ReplyReferenceInfo(is_reply=True, is_to_bot=True, text=vote_fixtures.RECEIPT),
        )
        with (
            patch.object(self.chat._community_vote, "get_store", return_value=fixture.store),
            patch.object(self.chat._community_vote, "find_user_by_platform", fixture.lookup),
            patch.object(self.chat.skills_manager, "get_tool_function", return_value=fixture.approve),
            patch.object(self.chat, "extract_onebot_reply_id", return_value="receipt"),
        ):
            self.assertTrue(await self.chat._stage_community_vote(ctx))
        fixture.approve.assert_awaited_once()
        self.assertEqual(fixture.store.get(vote_fixtures.BATCH_ID)["status"], "approved")
        delivered = self.chat._prepare_user_facing_reply(ctx.response, memory)
        self.assertIn("批次已加入词库", delivered)
        self.assertNotIn("没有可核验", delivered)

# Source: test_b7_phrasing.py
class B7PhrasingRouteTests(unittest.IsolatedAsyncioTestCase):
    asyncTearDown = B7PhrasingRouteTests_asyncTearDown
    row = B7PhrasingRouteTests_row
    result = B7PhrasingRouteTests_result
    call = B7PhrasingRouteTests_call
    stage = B7PhrasingRouteTests_stage
    review_word = B7PhrasingRouteTests_review_word
    answer = B7PhrasingRouteTests_answer

    asyncSetUp = B7PhrasingRouteTests_asyncSetUp

    assert_recodes = B7PhrasingRouteTests_assert_recodes

    async def test_inline_change_header_serialized_turn_writes_exact_swap_with_real_tools(self):
        from tests.helpers import TOOLS
        from tests.test_displace_shift_and_undo import set_up_change_header_swap_http

        for message, pairs in (
            ("喵喵 修改 dsqt 大全 dsqtv 大权", (("dsqt", "大全"), ("dsqtv", "大权"))),
            ("喵喵，修改：dsqt 大全，dsqtv 大权", (("dsqt", "大全"), ("dsqtv", "大权"))),
            ("@owner，改：大全 dsqt、\n大权 dsqtv", (("dsqt", "大全"), ("dsqtv", "大权"))),
            ("修改：大全 dsqt；大泉 dsqtv\n大权 dsqtvv",
             (("dsqt", "大全"), ("dsqtv", "大泉"), ("dsqtvv", "大权"))),
        ):
            with self.subTest(message=message):
                fixture = unittest.TestCase()
                try:
                    set_up_change_header_swap_http(fixture)
                    if len(pairs) == 3:
                        fixture.http.entries.append({"word": "大泉", "code": "dsqtvv", "type": "Phrase", "weight": 100})
                    old_words = {row["code"]: row["word"] for row in fixture.http.entries}
                    functions = {**harness._lookup_tools.TOOL_FUNCTIONS, **harness._review_tools.TOOL_FUNCTIONS,
                                 **TOOLS.TOOL_FUNCTIONS}
                    with patch.object(fixture.replay.chat.tool_executor, "_get_tool_function", functions.get):
                        await fixture.replay.turn({"message": message, "to_me": True}, raw_operation(
                            op="recode", operands=[word for _code, word in pairs], codes=[code for code, _word in pairs],
                        ))
                    reply = fixture.replay.last_response
                    self.assertEqual([(row["action"], row.get("oldWord"), row["word"], row["code"])
                                      for row in fixture.http.rows],
                                     [("Change", old_words[code], word, code) for code, word in pairs],
                                     (reply, fixture.http.requests))
                    self.assertEqual(len(fixture.http.writes), 1, reply)
                    self.assertIn("已加入草稿", reply)
                    for value in ("大权", "大全", "dsqt", "dsqtv", "offline-batch"):
                        self.assertIn(value, reply)
                    self.assertEqual(fixture.http.errors, [])
                finally:
                    fixture.doCleanups()

    async def test_inline_change_header_confirmation_round_trips_exact_literal_pairs(self):
        from keytao_bot.harness.operation_request import OperationRequest
        from tests.helpers import TOOLS
        from tests.test_displace_shift_and_undo import set_up_change_header_swap_http

        set_up_change_header_swap_http(self)
        message = "喵喵 修改 大全 dsqt；大权 dsqtv"
        commands = self.replay.commands
        resolved, options, error = await commands.resolve_model_replace(
            OperationRequest(op="recode", operands=("大权", "大全"), codes=("dsqtv", "dsqt")),
            "qq", self.replay.author.actor_id, message=message,
        )
        self.assertEqual((options, error), ((), ""))
        self.assertEqual((resolved.operands, resolved.codes), (("大全", "大权"), ("dsqt", "dsqtv")))
        self.assertEqual(commands.render_understood_command(resolved), "修改 dsqt 大全 dsqtv 大权")
        functions = {**harness._lookup_tools.TOOL_FUNCTIONS, **harness._review_tools.TOOL_FUNCTIONS,
                     **TOOLS.TOOL_FUNCTIONS}
        with patch.object(self.replay.chat.tool_executor, "_get_tool_function", functions.get), patch.dict(
            os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": "recode"},
        ):
            await self.replay.turn({"message": message, "to_me": True}, raw_operation(
                op="recode", operands=["大权", "大全"], codes=["dsqtv", "dsqt"],
            ))
            state = self.replay.store.get(self.replay.author)
            self.assertIsNotNone(state, self.replay.last_response)
            sealed = commands._understood_command_request(state)
            self.assertIsNotNone(sealed, self.replay.last_response)
            self.assertEqual((sealed.op, sealed.operands, sealed.codes),
                             (resolved.op, resolved.operands, resolved.codes))
            self.assertIn("回复「确认」执行", self.replay.last_response)
            self.assertEqual(self.http.writes, [])
            self.assertEqual(self.http.rows, [])
            await self.replay.turn({"message": "确认", "to_me": True}, raw_operation(
                op="confirm", operands=[], codes=[], refers_to_pending=True,
            ))
            self.assertEqual(self.http.writes, [])
            self.assertIn("回复「确认」执行", self.replay.last_response)
            await self.replay.turn({"message": "确认", "to_me": True}, raw_operation(
                op="confirm", operands=[], codes=[], refers_to_pending=True,
            ))
        reply = self.replay.last_response
        self.assertEqual([(row["action"], row.get("oldWord"), row["word"], row["code"])
                          for row in self.http.rows],
                         [("Change", "大权", "大全", "dsqt"), ("Change", "大全", "大权", "dsqtv")],
                         (reply, self.http.requests))
        self.assertEqual(len(self.http.writes), 1, reply)
        self.assertIn("已加入草稿", reply)
        for value in ("大权", "大全", "dsqt", "dsqtv", "offline-batch"):
            self.assertIn(value, reply)
        self.assertEqual(self.http.errors, [])

    async def test_inline_change_header_invalid_code_lookup_never_writes(self):
        from tests.helpers import OfflineResponse, TOOLS
        from tests.test_displace_shift_and_undo import set_up_change_header_swap_http

        for failure in ("missing", "duplicate", "ambiguous"):
            with self.subTest(failure=failure):
                fixture = unittest.TestCase()
                try:
                    set_up_change_header_swap_http(fixture)
                    before = copy.deepcopy(fixture.http.entries)
                    results = [{"code": row["code"], "phrases": [row]} for row in before]
                    if failure == "missing":
                        results.pop()
                    elif failure == "duplicate":
                        results[1] = copy.deepcopy(results[0])
                    else:
                        results[0]["phrases"].append({**before[0], "word": "大券", "weight": 101})
                    original = fixture.http._response

                    def response(method, url, kwargs):
                        if urlsplit(url).path == "/api/bot/phrases/by-code/batch":
                            fixture.http.requests.append((method, urlsplit(url).path, copy.deepcopy(kwargs)))
                            return OfflineResponse({"success": True, "results": results})
                        return original(method, url, kwargs)

                    functions = {**harness._lookup_tools.TOOL_FUNCTIONS, **harness._review_tools.TOOL_FUNCTIONS,
                                 **TOOLS.TOOL_FUNCTIONS}
                    with patch.object(fixture.http, "_response", side_effect=response), patch.object(
                        fixture.replay.chat.tool_executor, "_get_tool_function", functions.get,
                    ):
                        await fixture.replay.turn({"message": "修改 dsqt 大全 dsqtv 大权", "to_me": True}, raw_operation(
                            op="recode", operands=["大全", "大权"], codes=["dsqt", "dsqtv"],
                        ))
                    self.assertIn("无法唯一确定", fixture.replay.last_response)
                    self.assertIn("本次未写入", fixture.replay.last_response)
                    self.assertEqual(fixture.http.writes, [])
                    self.assertEqual(fixture.http.rows, [])
                    self.assertEqual(fixture.http.entries, before)
                    self.assertEqual(fixture.http.errors, [])
                finally:
                    fixture.doCleanups()

    async def test_multiple_glued_recodes_write_one_exact_batch(self):
        handled, ctx = await self.stage(
            "修改编码: 我送给为wsgu， 纯音乐为jyya", "recode",
            ("我送给为", "纯音乐为"), ("wsgu", "jyya"),
        )
        self.assertTrue(handled)
        self.assert_recodes(ctx.response, [
            ("Delete", "我送给", "wsg"), ("Create", "我送给", "wsgu"),
            ("Delete", "纯音乐", "jyy"), ("Create", "纯音乐", "jyya"),
        ])

    async def test_each_connector_keeps_same_word_recode_route(self):
        for connector in ("为", "是", "改为", "改成"):
            with self.subTest(connector=connector):
                self.writes.clear()
                handled, ctx = await self.stage(
                    f"修改编码: 我送给{connector}wsgu", "recode",
                    (f"我送给{connector}",), ("wsgu",),
                )
                self.assertTrue(handled)
                self.assert_recodes(ctx.response, [
                    ("Delete", "我送给", "wsg"), ("Create", "我送给", "wsgu"),
                ])

    async def test_multiple_recodes_survive_sealed_confirmation(self):
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "recode"
        handled, ctx = await self.stage(
            "修改编码: 我送给为wsgu，纯音乐为jyya", "recode",
            ("我送给", "纯音乐"), ("wsgu", "jyya"),
        )
        self.assertTrue(handled)
        self.assertIn("回复「确认」执行", ctx.response)
        self.assertEqual(self.writes, [])
        self.assert_recodes(await self.answer("确认"), [
            ("Delete", "我送给", "wsg"), ("Create", "我送给", "wsgu"),
            ("Delete", "纯音乐", "jyy"), ("Create", "纯音乐", "jyya"),
        ])

    async def test_model_code_order_cannot_swap_literal_targets(self):
        for message, operands, codes in (
            ("修改编码: 我送给为wsgu，纯音乐为jyya", ("我送给", "纯音乐"), ("jyya", "wsgu")),
            ("修改编码: 我送给为wsgu，纯音乐为jyya", ("纯音乐", "我送给"), ("jyya", "wsgu")),
            ("修改编码: 我送给为wsgu，纯音乐为jyya", ("我送给", "纯音乐"), ("wsg", "jyya", "jyy", "wsgu")),
            ("我送给改为wsgu，纯音乐改为jyya", ("我送给", "纯音乐"), ("wsg", "jyya", "jyy", "wsgu")),
            ("把我送给改为wsgu，纯音乐改为jyya", ("我送给", "纯音乐"), ("wsg", "jyya", "jyy", "wsgu")),
        ):
            with self.subTest(operands=operands, codes=codes):
                self.writes.clear()
                handled, ctx = await self.stage(message, "recode", operands, codes)
                self.assertTrue(handled)
                self.assert_recodes(ctx.response, [
                    ("Delete", "我送给", "wsg"), ("Create", "我送给", "wsgu"),
                    ("Delete", "纯音乐", "jyy"), ("Create", "纯音乐", "jyya"),
                ])

    async def test_real_word_ending_in_connector_is_not_truncated(self):
        self.rows.append(self.row("作为", "zw"))
        handled, ctx = await self.stage("修改编码: 作为wsgu", "recode", ("作为",), ("wsgu",))
        self.assertTrue(handled)
        self.assert_recodes(ctx.response, [
            ("Delete", "作为", "zw"), ("Create", "作为", "wsgu"),
        ])

    async def test_ambiguous_connector_never_changes_either_word(self):
        self.rows.extend((self.row("作为", "zw"), self.row("作", "z")))
        handled, ctx = await self.stage("修改编码: 作为wsgu", "recode", ("作为",), ("wsgu",))
        self.assertTrue(handled)
        self.assertIn("唯一确定", ctx.response)
        self.assertEqual(self.writes, [])

    async def test_connector_with_space_before_code_resolves_literal_word(self):
        handled, ctx = await self.stage(
            "修改编码: 我送给为 wsgu", "recode", ("我送给为",), ("wsgu",),
        )
        self.assertTrue(handled)
        self.assert_recodes(ctx.response, [
            ("Delete", "我送给", "wsg"), ("Create", "我送给", "wsgu"),
        ])

    async def test_ambiguous_second_source_prevents_all_recodes(self):
        self.rows.append(self.row("纯音乐", "jyya"))
        handled, ctx = await self.stage(
            "修改编码: 我送给为wsgu，纯音乐为jyyb", "recode",
            ("我送给", "纯音乐"), ("wsgu", "jyyb"),
        )
        self.assertTrue(handled)
        self.assertIn("指定旧码", ctx.response)
        self.assertEqual(self.writes, [])
        self.assertIsNone(self.store.get_record(self.author))

    async def test_same_word_delete_add_overrides_wrong_model_op(self):
        for message, op, operands, codes in (
            ('删除"纯音乐"编码，添加"纯音乐"为jyya', "delete", ("纯音乐",), ("jyya",)),
            ('删除"纯音乐"的编码jyy，添加"纯音乐"的编码为jyya', "recode", ("纯音乐", "纯音乐"), ("jyya",)),
        ):
            with self.subTest(message=message):
                self.writes.clear()
                handled, ctx = await self.stage(message, op, operands, codes)
                self.assertTrue(handled)
                self.assert_recodes(ctx.response, [
                    ("Delete", "纯音乐", "jyy"), ("Create", "纯音乐", "jyya"),
                ])

    async def test_wrong_explicit_old_code_never_writes(self):
        handled, ctx = await self.stage(
            '删除"纯音乐"的编码jyyb，添加"纯音乐"的编码为jyya',
            "recode", ("纯音乐", "纯音乐"), ("jyya",),
        )
        self.assertTrue(handled)
        self.assertIn("未写入", ctx.response)
        self.assertEqual(self.writes, [])

    async def test_delete_wrong_code_cannot_seal_or_replace_ticket(self):
        for direct_exec in (True, False):
            with self.subTest(direct_exec=direct_exec):
                os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "delete" + (",direct_exec" if direct_exec else "")
                pending = self.commands.PendingToolConfirm("keytao_submit_batch", {"batch_id": "offline-batch"})
                self.assertTrue(self.store.set(self.author, pending))
                record = self.store.get_record(self.author)
                handled, ctx = await self.stage("删除纯音乐 jyya", "delete", ("纯音乐",), ("jyya",))
                self.assertTrue(handled)
                self.assertIn("jyy", ctx.response)
                self.assertIn("请确认", ctx.response)
                self.assertNotIn("删词 纯音乐 jyya", ctx.response)
                self.assertIs(self.store.get_record(self.author), record)
                self.assertEqual(self.writes, [])

    async def test_delete_guard_accepts_exact_draft_create(self):
        self.rows = [row for row in self.rows if row["word"] != "纯音乐"]
        self.draft_items = [{
            "id": 7, "word": "纯音乐", "code": "jyya", "type": "Phrase",
            "action": "Create", "weight": 100,
        }]
        os.environ["KEYTAO_SEMANTIC_ROUTE_OPS"] = "delete"
        handled, ctx = await self.stage("删除纯音乐 jyya", "delete", ("纯音乐",), ("jyya",))
        self.assertTrue(handled)
        self.assertIn("回复「确认」执行", ctx.response)
        self.assertEqual(self.writes, [])
        self.assertEqual(self.store.get(self.author).args["codes"], ["jyya"])

# Source: test_b7_phrasing.py
class B7QueryRouteTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = B7QueryRouteTests_asyncSetUp
    asyncTearDown = B7QueryRouteTests_asyncTearDown
    lookup = B7QueryRouteTests_lookup
    context = B7QueryRouteTests_context
    set_model = B7QueryRouteTests_set_model
    ticket = B7QueryRouteTests_ticket
    assert_finished = B7QueryRouteTests_assert_finished
    assert_preserved = B7QueryRouteTests_assert_preserved

    async def test_present_words_code_query_never_calls_general_or_changes_ticket(self):
        self.entries.update({"我送给": "wsg", "纯音乐": "jyy"})
        record, original, _prompt = self.ticket(self.author)
        ctx = self.context("查询编码：我送给，纯音乐")
        self.set_model(operands=("我送给", "纯音乐"))
        await self.assert_finished(ctx)
        self.assertIn("wsg", ctx.response)
        self.assertIn("jyy", ctx.response)
        self.assertEqual([call.args[1] for call in self.tool.await_args_list],
                         [{"word": "我送给"}, {"word": "纯音乐"}])
        self.main.assert_not_awaited()
        self.assert_preserved(self.author, record, original)

    async def test_comma_query_without_ticket_uses_dictionary_reply(self):
        self.entries.update({"我送给": "wsg", "纯音乐": "jyy"})
        ctx = self.context("我送给，纯音乐")
        self.set_model(operands=("我送给", "纯音乐"))
        await self.assert_finished(ctx)
        self.assertIn("wsg", ctx.response)
        self.assertIn("jyy", ctx.response)
        self.main.assert_not_awaited()
        self.assertIsNone(self.store.get_record(self.author))

# Source: test_b8_cancel.py
class B8CancelTests(unittest.IsolatedAsyncioTestCase):
    replay = B8CancelTests_replay

    offer = B8CancelTests_offer

    async def test_cancel_generated_incident_ticket_before_query_routing(self):
        replay = self.replay()
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
            await self.offer(replay)
            replay.calls.clear()
            replay.trace.clear()
            outcome = await replay.turn({"message": "取消", "to_me": True}, raw_operation(
                op="cancel", operands=["哀声叹气"], codes=[], frame="direct", refers_to_pending=True,
            ))
            self.assertEqual(replay.last_response, "已取消。")
            self.assertIsNone(replay.store.get_record(replay.author))
            self.assertEqual(replay.calls, [])
            self.assertEqual(replay.writes, [])
            self.assertEqual(outcome["op"], "cancel")
            self.assertEqual(replay.trace, ["_stage_understand_operation_early"])
            replay.client.create.assert_not_awaited()

    async def test_cancel_with_stale_quote_stays_out_of_word_query_and_preserves_binding(self):
        await self.assert_unbound_reply("取消", model_op="cancel")

    async def test_other_recognized_closed_answers_cannot_be_reinterpreted_as_queries(self):
        for message in ("取消。", "算了", "不加了", "取消本次添加", "都加入", "加入并提交", "哀声叹气 2"):
            with self.subTest(message=message):
                await self.assert_unbound_reply(message)

    async def test_digestless_legacy_ticket_keeps_its_binding_without_query_fallback(self):
        for message in ("取消", "算了"):
            with self.subTest(message=message):
                await self.assert_unbound_reply(message, digestless=True)

    assert_unbound_reply = B8CancelTests_assert_unbound_reply

# Source: test_b9_parallel.py
class B9ParallelTests(unittest.IsolatedAsyncioTestCase):
    setUp = B9ParallelTests_setUp

    tool = B9ParallelTests_tool

    prepare = B9ParallelTests_prepare

    async def test_f3_m2_3_ordering_evidence_and_sibling_recommendation_agree(self):
        from keytao_bot.harness.state import _pending_add_word_payload
        from keytao_bot.utils.commonness_copy import candidate_commonness_summary_copy
        from keytao_bot.utils.pending_confirmation import (
            front_insert_recommendation_copy, render_server_backed_single_word_candidates,
            render_server_backed_single_word_lookup,
        )

        self.words = ("肉肠", "肉偿")
        scopes = {}
        for word, tail, recommended in (("肉肠", "rdjpiu", "rdjpi"), ("肉偿", "rdjpii", "rdjp")):
            state = PendingAddWord(
                word, recommended, [("rdjp", True), ("rdjpi", False), (tail, False)],
                occupied_words={"rdjp": ["柔肠"]},
                pronunciation_codes={code: "rou chang" for code in ("rdjp", "rdjpi", tail)},
                needs_manual_review=True, manual_review_reason="需要管理员审核",
            )
            assessment = {
                "verdict": "front_more_common" if word == "肉偿" else "close",
                "newWord": word, "occupantWord": "柔肠", "occupantCode": "rdjp",
                "freeCode": "rdjpi", "newCode": recommended,
                "summary": "语料频次 20 vs 10，词典收录 2 vs 2",
            }
            state.server_candidates = state.candidates
            state.server_occupied_words = state.occupied_words
            state.server_ordering_assessments = self.replay.commands._server_ordering_snapshot(
                state, state.candidates, state.occupied_words, [assessment],
            )
            scopes[word] = {
                "word": word, "candidates": state.candidates,
                "occupiedWords": state.occupied_words,
                "orderingAssessments": state.server_ordering_assessments,
                "reviewedState": _pending_add_word_payload(state),
                "reviewedPrompt": render_server_backed_single_word_candidates(
                    word, recommended, state.candidates, state.occupied_words,
                    state.server_ordering_assessments,
                ),
            }
        with self.subTest(case="snapshot"):
            self.assertEqual(scopes["肉偿"]["orderingAssessments"][0].get("summary"), assessment["summary"])
        with self.subTest(case="missing evidence"):
            legacy = {**assessment, "summary": ""}
            self.assertNotIn("依据：", front_insert_recommendation_copy(legacy))
            self.assertNotIn("不弱于", candidate_commonness_summary_copy({**legacy, "verdict": "behind_more_common"}))
        with self.subTest(case="reserved free slot"):
            second = render_server_backed_single_word_lookup(
                "肉偿", "rdjpii", scopes["肉偿"]["candidates"], {"rdjp": ["柔肠"]},
                sibling_recommendations={"rdjpi": "肉肠"},
                ordering_assessments=[{**assessment, "verdict": "close", "newCode": "rdjpi"}],
                actionable_controls=True,
            )
            self.assertIn("3. rdjpii — ✅推荐", second)
            self.assertNotIn("推荐编码：rdjpii", second)

        async def prepare_word(word, *_args, **kwargs):
            kwargs["_prepared_scopes"].append(scopes[word])
            return ""

        with patch.object(self.replay.commands, "_try_handle_simple_single_word_query", side_effect=prepare_word):
            response = await self.prepare()
        with self.subTest(case="sibling"):
            state = self.replay.store.get(self.replay.author)
            self.assertIsNotNone(state, response)
            self.assertEqual([item["code"] for item in state.args["items"]], ["rdjpi", "rdjpii"], response)
            second = response.split("2. 「肉偿」", 1)[1]
            self.assertIn("3. rdjpii — ✅推荐", second)
            self.assertNotIn("推荐编码：rdjpii", second)
            self.assertNotIn("推荐空位 rdjpi", second)
            self.assertIsNone(self.replay.commands._pending_batch_front_insert_plan(state))

    async def test_reviews_overlap_at_most_two_and_keep_input_order(self):
        response = await self.prepare()
        self.assertEqual(self.peak, 2)
        self.assertNotEqual(self.completed[0], self.words[0])
        state = self.replay.store.get(self.replay.author)
        self.assertEqual([item["word"] for item in state.args["items"]], list(self.words))
        self.assertEqual([scope["word"] for scope in state.args["_candidate_scopes"]], list(self.words))
        positions = [response.index(f"「{word}」") for word in self.words]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(self.replay.writes, [])

    async def test_failed_reviews_keep_their_messages_and_other_candidates(self):
        self.failures = {
            "云翻": {"success": False, "message": "Offline review unavailable: 云翻"},
            "没头没尾": ValueError("Offline review crashed"),
        }
        response = await self.prepare()
        self.assertIn("Offline review unavailable: 云翻", response)
        self.assertIn("「没头没尾」暂时没有可核验的候选。", response)
        self.assertLess(response.index("云翻"), response.index("没头没尾"))
        state = self.replay.store.get(self.replay.author)
        self.assertEqual([item["word"] for item in state.args["items"]], ["发薪", "国潮"])
        self.assertEqual([scope["word"] for scope in state.args["_candidate_scopes"]], ["发薪", "国潮"])
        self.assertEqual(self.replay.writes, [])

    async def test_existing_and_pending_words_keep_their_own_results(self):
        self.replay.entries = [{"word": "云翻", "code": "yjfj", "type": "Phrase"}]
        self.replay.draft = [{
            "source": "draft", "batchId": "offline-batch", "batchStatus": "Draft",
            "batchUrl": "https://offline.invalid/batch/offline-batch", "itemStatus": "Pending",
            "action": "Create", "word": "没头没尾", "code": "mtmw", "type": "Phrase",
        }]
        response = await self.prepare()
        self.assertIn("已在词库：云翻 yjfj", response)
        self.assertIn("「没头没尾」已在当前草稿中", response)
        self.assertNotIn("暂时没有可核验的候选", response)
        state = self.replay.store.get(self.replay.author)
        self.assertEqual([item["word"] for item in state.args["items"]], ["发薪", "国潮"])
        self.assertEqual(self.completed, ["国潮", "发薪"])

    async def test_review_evidence_reaches_the_parent_delivery_guard_in_input_order(self):
        review = self.replay.commands.keytao_review
        review.record_commonness_evidence({"summary": "Offline existing evidence"})
        self.evidence = {
            "发薪": "Offline evidence for 发薪",
            "云翻": "Offline evidence for 云翻",
            "没头没尾": "Offline evidence for 没头没尾",
            "国潮": "Offline evidence for 国潮",
        }
        await self.prepare()
        self.assertEqual(review.current_commonness_evidence(), (
            "Offline existing evidence", "Offline evidence for 发薪", "Offline evidence for 云翻",
            "Offline evidence for 没头没尾", "Offline evidence for 国潮",
        ))

    async def test_unresolved_word_keeps_the_unbound_notice_in_the_parent(self):
        self.unresolved = {"云翻"}
        resolver = self.replay.commands.user_resolver
        with patch.object(resolver, "_find_user_payload", AsyncMock(return_value={"found": False})):
            response = await self.prepare()
        self.assertIs(resolver.resolved_binding_for_notice("qq", self.replay.author.actor_id), False)
        self.assertEqual(response.count(UNBOUND_BINDING_PRECHECK_NOTICE), 1)

# Source: test_b9_small.py
class B9WordQueryTests(unittest.IsolatedAsyncioTestCase):
    async def test_control_text_never_calls_word_classifier(self):
        classifier = AsyncMock(return_value=b17_rest_recommended_routing.SimpleWordQueryIntent(True, ("取消",)))
        with patch.object(b17_rest_recommended_routing, "_classify_simple_word_query_intent", classifier):
            for message in ("取消", "取消。", "确认", "提交", "加入", "加", "删", "查", "算了"):
                with self.subTest(message=message):
                    self.assertEqual(await b17_rest_recommended_routing._get_simple_word_query_words(message), ())
            classifier.assert_not_awaited()

    async def test_classifier_words_pass_the_same_structural_exclusions(self):
        classifier = AsyncMock(return_value=b17_rest_recommended_routing.SimpleWordQueryIntent(
            True, ("洛阳纸贵", "取消", "确认", "提交", "加入", "加", "删", "查"),
        ))
        with patch.object(b17_rest_recommended_routing, "_classify_simple_word_query_intent", classifier):
            self.assertEqual(
                await b17_rest_recommended_routing._get_simple_word_query_words("洛阳纸贵 喜上眉梢"),
                ("洛阳纸贵",),
            )
            classifier.assert_awaited_once()

# Source: test_b9_small.py
class B9CancelTests(unittest.IsolatedAsyncioTestCase):
    async def test_change_number_cannot_borrow_named_selection_binding(self):
        fixture = B8CancelTests()
        replay = fixture.replay()
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            record = await fixture.offer(replay)
            replay.history[:] = [{"role": "assistant", "content": "Offline unrelated reply."}]
            replay.calls.clear()
            execute = stack.enter_context(patch.object(replay.commands, "_execute_confirmed_tool", AsyncMock(return_value="Unexpected write")))
            await replay.turn({"message": "换 2", "to_me": True}, raw_operation(
                op="select_candidate", operands=[], codes=[], frame="direct", refers_to_pending=True,
            ))
            self.assertEqual(replay.last_response, "这条回复未对应当前待确认操作；请引用当前候选消息后再回复。")
            self.assertIs(replay.store.get_record(replay.author), record)
            self.assertEqual(replay.calls, [])
            self.assertEqual(replay.writes, [])
            execute.assert_not_awaited()
            replay.client.create.assert_not_awaited()

    async def test_semantic_query_keeps_compact_present_facts_through_delivery(self):
        replay = Replay({
            "sender": "b9-owner",
            "dictionary": [
                {"word": "喜上眉梢", "code": "xemev", "type": "Phrase"},
                {"word": "欢天喜地", "code": "htxda", "type": "Phrase"},
            ],
            "candidate_facts": {"发薪": {"recommended": "fsxb", "codes": ["fsxb", "fsxba"], "pinyin": "fa xin"}},
        }, harness, FakeClient)
        words = ("喜上眉梢", "欢天喜地", "发薪")
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
            await replay.turn({"message": " ".join(words), "to_me": True}, raw_operation(
                op="query", operands=list(words), codes=[], frame="none",
            ))
            self.assertTrue(replay.last_response.startswith("已在词库：喜上眉梢 xemev、欢天喜地 htxda\n"), replay.last_response)
            self.assertIn("「换 2」", replay.last_response)
            self.assertTrue(replay.chat._advertised_reply_matches_live_record(replay.last_response, replay.store.get_record(replay.author)))
            self.assertEqual(replay.writes, [])

    async def test_cancel_without_actor_ticket_finishes_before_models_or_lookup(self):
        for message in ("取消", "取消。", "算了", "不加了", "取消本次添加"):
            with self.subTest(message=message):
                replay = Replay({"sender": "b9-owner"}, harness, FakeClient)
                other = harness.ConversationAddress.group("qq", "prod-replay", "other-owner")
                self.assertTrue(replay.store.set(other, PendingAddWord("云翻", "yjfj", [("yjfj", False)])))
                original = replay.store.get_record(other)
                with ExitStack() as stack:
                    for context in replay.patches():
                        stack.enter_context(context)
                    stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
                    main = stack.enter_context(patch.object(replay.chat, "get_ai_response_core", AsyncMock()))
                    await replay.turn({"message": message, "to_me": True}, raw_operation(
                        op="query", operands=["取消"], codes=[], frame="none",
                    ))
                    self.assertEqual(replay.last_response, "当前没有待取消的操作。")
                    self.assertEqual(replay.trace, ["_stage_understand_operation_early"])
                    self.assertEqual(replay.calls, [])
                    self.assertEqual(replay.writes, [])
                    replay.client.create.assert_not_awaited()
                    main.assert_not_awaited()
                    self.assertIs(replay.store.get_record(other), original)

# Source: test_b9_small.py
class B9CandidateTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = B9CandidateTests_asyncSetUp
    asyncTearDown = B9CandidateTests_asyncTearDown
    lookup = B9CandidateTests_lookup

    async def test_lookup_context_is_not_an_operation_to_cancel(self):
        self.assertTrue(self.store.set(self.author, create_pending_trusted_word_record("积极", "jjuu", "Phrase")))
        ctx = query.SemanticQueryRouteTests.context(self, "取消")
        ctx.event = object()
        with patch.object(self.chat, "to_me", return_value=AsyncMock(return_value=True)):
            self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
        self.assertEqual(ctx.response, "当前没有待取消的操作。")
        self.tool.assert_not_awaited()
        self.main.assert_not_awaited()
        self.client.create.assert_not_awaited()

    offer = B9CandidateTests_offer

    async def test_multi_word_add_and_query_collapse_all_present_words(self):
        for explicit in (True, False):
            with self.subTest(explicit=explicit):
                self.assertEqual(await self.offer(explicit=explicit), "已在词库：喜上眉梢 xemev、欢天喜地 htxda")
                self.assertIsNone(self.store.get_record(self.author))
                self.main.assert_not_awaited()
                self.client.create.assert_not_awaited()

    async def test_one_actionable_word_uses_single_layout_with_original_ticket(self):
        response = await self.offer(("发薪",))
        self.assertTrue(response.startswith("已在词库：喜上眉梢 xemev、欢天喜地 htxda\n"))
        self.assertEqual(response.count("发薪"), 1, response)
        self.assertNotIn("1 个词的候选", response)
        self.assertNotIn("可再追加一个编码", response)
        for control in ("加入", "加入并提交", "换 2", "取消"):
            self.assertIn(f"「{control}」", response)
        record = self.store.get_record(self.author)
        self.assertEqual(record.state.function_name, "keytao_batch_add_to_draft")
        self.assertEqual(record.state.args["_query_words"], ["喜上眉梢", "欢天喜地", "发薪"])
        self.assertTrue(self.chat._advertised_reply_matches_live_record(response, record))
        self.assertEqual(self.chat._render_live_batch_record(record), response)
        self.assertEqual(self.chat._append_pending_ticket_challenge(response, self.author), response)

    async def test_change_number_selects_only_live_word_and_keeps_binding(self):
        await self.offer(("发薪",))
        record = self.store.get_record(self.author)
        for message in ("换 2", "发薪 2"):
            with self.subTest(message=message):
                derived, intent, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(record.state, message)
                self.assertIsNone(error)
                self.assertIsNotNone(derived)
                self.assertEqual(intent.intent, "pending_confirm")
                self.assertEqual([(item["word"], item["code"]) for item in derived.args["items"]], [("发薪", "fsxba")])
                self.assertEqual(derived.args["_query_words"], record.state.args["_query_words"])
                self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(message, record.state))

    async def test_change_number_is_ambiguous_for_multiple_live_words(self):
        response = await self.offer(("发薪", "云翻"))
        self.assertIn("取消", response)
        self.assertNotIn("换 2", response)
        self.assertTrue(self.chat.advertised_reply_contract(response).candidate_selection)
        record = self.store.get_record(self.author)
        self.assertTrue(self.chat._advertised_reply_matches_live_record(response, record))
        derived, _intent, _error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(record.state, "换 2")
        self.assertIsNone(derived)
        self.assertFalse(b17_rest_recommended_routing.message_authorizes_live_pending_mutation("换 2", record.state))

    async def test_change_number_on_single_word_ticket_executes_selected_code(self):
        state = PendingAddWord("午时", "wj", [("wj", False), ("wja", False)],
                               server_candidates=[("wj", False), ("wja", False)])
        self.assertTrue(self.store.set(self.author, state))
        with patch.object(self.commands, "_execute_add_to_draft", AsyncMock(return_value="Offline receipt")) as execute:
            response = await self.commands.handle_pending_message_core(
                "换 2", "qq", self.author.actor_id, self.author, allow_intent_model=False,
            )
        self.assertIn("Offline receipt", response)
        self.assertEqual(execute.await_args.args[:2], ("午时", "wja"))
        self.client.create.assert_not_awaited()

    async def test_single_actionable_risk_ticket_keeps_original_inventory_for_change_number(self):
        await self.offer(("发薪",))
        origin = self.commands._candidate_origin_for_ticket(self.store.get(self.author))
        state = server_warning_pending_state(PendingToolConfirm("keytao_create_phrase", {
            "word": "发薪", "code": "fsxb", "_candidate_origin": origin,
        }), {"batchId": "offline-batch", "contentVersion": 1, "warningDigest": "a" * 64, "warnings": []})
        self.assertTrue(self.store.set(self.author, state))
        with patch.object(self.commands, "_execute_confirmed_tool", AsyncMock(return_value="Offline preview")) as execute:
            await self.commands.try_reselect_live_ticket("换 2", "qq", self.author.actor_id, self.author)
        execute.assert_awaited_once()
        selected = execute.await_args.args[0]
        self.assertEqual((selected.args["word"], selected.args["code"]), ("发薪", "fsxba"))
        self.assertEqual(selected.args["_candidate_origin"]["queryWords"], ["喜上眉梢", "欢天喜地", "发薪"])
        self.assertTrue(execute.await_args.kwargs["require_new_confirmation"])
        self.client.create.assert_not_awaited()

# Source: test_b9_speed.py
class B9RoutingSpeedTests(unittest.IsolatedAsyncioTestCase):
    async def test_unbound_affirmative_controls_keep_the_ticket_without_models(self):
        chat = harness.openai_chat_module
        for message, digest in ((message, digest) for message in ("嗯嗯", "好滴", "是的", "没问题") for digest in ("", "a" * 64)):
            with self.subTest(message=message, digest=digest):
                store = MemoryConversationStateStore()
                author = harness.ConversationAddress.group("qq", "b9-speed", "owner")
                self.assertTrue(store.set(author, PendingAddWord(
                    "云翻", "yjfj", [("yjfj", False)], server_candidates=[("yjfj", False)],
                )))
                store.bind_origin_prompt_digest(author, digest)
                record = store.get_record(author)
                client = FakeClient()
                ctx = chat.TurnContext(
                    bot=None, event=None, platform="qq", user_id=author.actor_id,
                    normalized_message_text=message, conv_key=author, space_key=author.space_key,
                )
                with (
                    patch.object(chat, "conversation_state_store", store),
                    patch.object(chat, "get_history", return_value=[]),
                    patch.object(chat, "AsyncOpenAI", return_value=client),
                ):
                    self.assertFalse(await chat._stage_understand_operation_early(ctx))
                self.assertEqual(ctx.scoped_pending_response, "这条回复未对应当前待确认操作；请引用当前候选消息后再回复。")
                self.assertIs(store.get_record(author), record)
                client.create.assert_not_awaited()

    async def test_resolved_pending_control_uses_at_most_one_model_call(self):
        chat = harness.openai_chat_module
        store = MemoryConversationStateStore()
        author = harness.ConversationAddress.group("qq", "b9-speed", "owner")
        state = PendingAddWord("云翻", "yjfj", [("yjfj", False)])
        self.assertTrue(store.set(author, state))
        client = FakeClient(json.dumps(raw_operation(
            op="confirm", operands=["云翻"], codes=[], refers_to_pending=True,
        )))
        ctx = chat.TurnContext(
            bot=None, event=None, platform="qq", user_id=author.actor_id,
            message_text="加入", normalized_message_text="加入", conv_key=author,
            space_key=author.space_key, current_pending_record=store.get_record(author),
        )
        ctx.command_intent_cache, ctx.command_intent_for = b17_rest_recommended_routing.command_intent_memoizer("加入", "加入")
        with (
            patch.dict("os.environ", {"KEYTAO_SEMANTIC_ROUTE_OPS": "pending_answer,pending_release"}),
            patch.object(chat, "AsyncOpenAI", return_value=client),
            patch.object(b17_rest_recommended_routing, "AsyncOpenAI", return_value=client),
            patch.object(b17_rest_recommended_routing, "OPENAI_API_KEY", "offline"),
            patch.object(chat, "conversation_state_store", store),
            patch.object(chat, "remember_conversation"),
            patch.object(chat, "_finish_ai_chat_matcher", AsyncMock()) as finish,
            patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")),
        ):
            self.assertFalse(await chat._stage_understand_operation(ctx))
            self.assertTrue(ctx.operation_request.refers_to_pending)
            self.assertTrue(await chat._stage_apply_scoped_pending_intent(ctx))
            await ctx.command_intent_for(state)
        self.assertLessEqual(client.create.await_count, 1)
        self.assertEqual(ctx.response, "没找到要执行的操作，本次未写入。")
        finish.assert_awaited_once_with(ctx.response)
        self.assertIs(store.get_record(author), ctx.current_pending_record)

    async def test_word_query_classification_is_reused_only_within_the_turn(self):
        classifier = AsyncMock(return_value=b17_rest_recommended_routing.SimpleWordQueryIntent(False))
        with patch.object(b17_rest_recommended_routing, "_classify_simple_word_query_intent", classifier):
            for _turn in range(2):
                token = begin_turn_metrics("qq", "group")
                try:
                    for _stage in range(2):
                        self.assertEqual(await b17_rest_recommended_routing._get_simple_word_query_words("洛阳纸贵 喜上眉梢"), ())
                finally:
                    end_turn_metrics(token)
            self.assertEqual(classifier.await_count, 2)

# Source: test_b9_speed.py
class B9LookupTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = B9LookupTests_asyncSetUp

    row = B9LookupTests_row

    tool = B9LookupTests_tool

    prepare = B9LookupTests_prepare

    async def test_multi_word_prefetch_preserves_existing_draft_and_submitted_output(self):
        words = ("喜上眉梢", "发薪", "云翻", "未收录")
        response = await self.prepare(words)
        self.assertEqual(response, "\n\n".join((
            "「发薪」已在当前草稿中（→ fsxb）：https://offline.invalid/batch/draft",
            "「云翻」已在待审核批次中（→ yjfj，审核中）：https://offline.invalid/batch/submitted",
            "Offline review unavailable: 未收录",
            "已在词库：喜上眉梢 xemev",
        )))
        self.assertEqual(self.calls, [
            ("keytao_lookup_by_words_batch", {"words": list(words)}),
            ("keytao_pending_items_by_words", {"words": list(words)}),
            ("keytao_prepare_reviewed_add", {"word": "未收录"}),
        ])

    async def test_pending_failure_keeps_each_word_fail_closed(self):
        self.pending_failure = True
        response = await self.prepare(("喜上眉梢", "未收录"))
        self.assertEqual(response, "「喜上眉梢」暂时没有可核验的候选。\n\n「未收录」暂时没有可核验的候选。")
        self.assertEqual([name for name, _args in self.calls], [
            "keytao_lookup_by_words_batch", "keytao_pending_items_by_words",
        ])
        self.assertIsNone(self.store.get_record(("qq", "b9-owner")))

    async def test_lookup_failure_keeps_the_existing_review_fallback(self):
        self.lookup_failure = True
        words = ("喜上眉梢", "未收录")
        response = await self.prepare(words)
        self.assertEqual(response, "\n\n".join(f"Offline review unavailable: {word}" for word in words))
        self.assertEqual([args["word"] for name, args in self.calls
                          if name == "keytao_prepare_reviewed_add"], list(words))
        self.assertEqual(sum(name == "keytao_lookup_by_words_batch" for name, _args in self.calls), 1)

    async def test_prefetch_chunks_at_pending_lookup_limit(self):
        words = tuple(f"离线词{index}" for index in range(51))
        self.entries.update({word: "abcd" for word in words})
        with patch.object(self.commands, "_MAX_BARE_MULTI_WORD_QUERY_ITEMS", 100):
            response = await self.prepare(words)
        self.assertEqual(response, "已在词库：" + "、".join(f"{word} abcd" for word in words))
        self.assertEqual(self.calls, [
            ("keytao_lookup_by_words_batch", {"words": list(words[:50])}),
            ("keytao_pending_items_by_words", {"words": list(words[:50])}),
            ("keytao_lookup_by_words_batch", {"words": list(words[50:])}),
            ("keytao_pending_items_by_words", {"words": list(words[50:])}),
        ])

# Source: test_b9_speed.py
class B9PronunciationEvidenceTests(unittest.IsolatedAsyncioTestCase):
    setUp = B9PronunciationEvidenceTests_setUp

    async def test_agreeing_local_datasets_skip_web_for_nonpolyphonic_word(self):
        local = {
            "entries": [{
                "sourceId": dataset, "dataset": dataset, "source": dataset,
                "url": "", "normalized": ["ce", "shi"],
                "pinyin": "ce shi", "category": "dictionary", "trust": 4,
            } for dataset in ("cedict", "large_pinyin")],
            "outcomes": [{"sourceId": dataset, "status": "completed", "lookupResult": "found"}
                         for dataset in ("cedict", "large_pinyin")],
        }
        with (
            patch.object(review, "_collect_local_pronunciation_evidence", return_value=local),
            patch.object(review, "detect_polyphonic_characters", return_value={}),
            patch.object(review, "query_reference_readings", return_value=[object()]),
            patch.object(review, "_fetch_text", AsyncMock(return_value="")) as fetch,
            patch.object(review, "_search_web", AsyncMock(return_value=[])) as search,
            patch.object(review, "_request_bot_evidence_proxy", AsyncMock(
                return_value=review._BotEvidenceProxyResult("unavailable"))) as proxy,
        ):
            result = await review.collect_pronunciation_evidence("测试")
        self.assertEqual(fetch.await_count + search.await_count + proxy.await_count, 0)
        self.assertTrue(result["lookupComplete"])
        self.assertEqual(result["groups"][0]["normalized"], ["ce", "shi"])

    async def test_local_disagreement_or_unknown_polyphones_still_consult_web(self):
        entry = {"sourceId": "cedict", "dataset": "cedict", "source": "cedict",
                 "url": "", "normalized": ["ce", "shi"], "pinyin": "ce shi",
                 "category": "dictionary", "trust": 4}
        cases = [
            ([entry], {}, None, [object()]),
            ([entry, {**entry, "sourceId": "large_pinyin", "dataset": "large_pinyin",
                      "normalized": ["ce", "si"]}], {}, None, [object()]),
            ([entry, {**entry, "sourceId": "large_pinyin", "dataset": "large_pinyin"}], {0: ("ce", "ze")}, None, [object()]),
            ([entry, {**entry, "sourceId": "large_pinyin", "dataset": "large_pinyin"}], {},
             review.PinyinReferenceUnavailable("missing data"), [object()]),
            ([entry, {**entry, "sourceId": "large_pinyin", "dataset": "large_pinyin"}], {}, None, []),
        ]
        for entries, polyphones, error, character_rows in cases:
            with self.subTest(entries=len(entries), polyphones=polyphones, error=error):
                review._clear_review_caches()
                with (
                    patch.object(review, "AUTHORITATIVE_SOURCES", [{
                        "id": "fixture", "label": "fixture", "domain": "fixture.invalid",
                        "category": "dictionary", "trust": 4, "query": "{word}",
                        "direct_urls": ["https://fixture.invalid/{word}"],
                    }]),
                    patch.object(review, "_collect_local_pronunciation_evidence", return_value={
                        "entries": entries, "outcomes": []}),
                    patch.object(review, "detect_polyphonic_characters", return_value=polyphones,
                                 side_effect=error),
                    patch.object(review, "query_reference_readings", return_value=character_rows),
                    patch.object(review, "_fetch_text", AsyncMock(return_value="")) as fetch,
                    patch.object(review, "_search_web", AsyncMock(return_value=[])),
                ):
                    await review.collect_pronunciation_evidence("测试")
                self.assertGreater(fetch.await_count, 0)

    timeout_types = B9PronunciationEvidenceTests_timeout_types

    async def test_timeouts_are_not_retried_including_httpx_connect_timeout(self):
        classes = self.timeout_types()
        with ExitStack() as stack:
            for name, cls in zip(("TransportError", "TimeoutException", "ConnectTimeout", "ConnectError"), classes):
                stack.enter_context(patch.object(sys.modules["httpx"], name, cls, create=True))
            for error in (TimeoutError("timed out"), classes[1]("read timeout"), classes[2]("connect timeout")):
                with self.subTest(error=type(error).__name__):
                    with patch.object(review.http_client, "guarded_fetch", AsyncMock(side_effect=error)) as fetch:
                        with self.assertRaises(type(error)):
                            await review._fetch_review_url("https://timeout.invalid/page", max_attempts=3)
                    self.assertEqual(fetch.await_count, 1)

    async def test_pronunciation_search_registry_uses_the_same_timeout_policy(self):
        classes = self.timeout_types()
        with ExitStack() as stack:
            for name, cls in zip(("TransportError", "TimeoutException", "ConnectTimeout", "ConnectError"), classes):
                stack.enter_context(patch.object(sys.modules["httpx"], name, cls, create=True))
            stack.enter_context(patch.object(review, "_registered_web_search_function", None))
            stack.enter_context(patch.object(review.http_client.asyncio, "sleep", AsyncMock()))
            fetch = stack.enter_context(patch.object(review.http_client, "guarded_fetch", AsyncMock(
                side_effect=classes[1]("read timeout"))))
            await review._registered_web_search("测试 拼音", fetch_top_n=0)
        urls = [call.args[0] for call in fetch.await_args_list]
        self.assertTrue(urls)
        self.assertEqual(len(urls), len(set(urls)))

    async def test_breaker_is_host_scoped_expires_and_success_resets_streak(self):
        classes = self.timeout_types()
        response = SimpleNamespace(status_code=200, is_success=True, text="")
        now = [1000.0]
        with ExitStack() as stack:
            for name, cls in zip(("TransportError", "TimeoutException", "ConnectTimeout", "ConnectError"), classes):
                stack.enter_context(patch.object(sys.modules["httpx"], name, cls, create=True))
            stack.enter_context(patch.object(review.time, "monotonic", side_effect=lambda: now[0]))
            fetch = stack.enter_context(patch.object(review.http_client, "guarded_fetch", AsyncMock(
                side_effect=[TimeoutError(), TimeoutError(), response, TimeoutError(), TimeoutError(),
                             TimeoutError(), response, response])))
            for _ in range(2):
                with self.assertRaises(TimeoutError):
                    await review._fetch_review_url("https://one.invalid/a", max_attempts=1)
            await review._fetch_review_url("https://one.invalid/b", max_attempts=1)
            for _ in range(3):
                with self.assertRaises(TimeoutError):
                    await review._fetch_review_url("https://one.invalid/c", max_attempts=1)
            self.assertEqual(fetch.await_count, 6)
            with self.assertRaises(TimeoutError):
                await review._fetch_review_url("https://one.invalid/d", max_attempts=1)
            unavailable = await review._fetch_text("https://one.invalid/d")
            self.assertEqual(unavailable.lookup_status, "timed_out")
            self.assertFalse(unavailable)
            self.assertEqual(fetch.await_count, 6)
            await review._fetch_review_url("https://two.invalid/a", max_attempts=1)
            now[0] += 600
            await review._fetch_review_url("https://one.invalid/e", max_attempts=1)
            self.assertEqual(fetch.await_count, 8)

# Source: test_s55_failure_boundary.py
class FailureBoundaryTests(unittest.TestCase):
    def test_empty_response_logs_traceback_without_retry(self):
        async def run():
            logger = logging.getLogger("s55.empty")
            reply = AsyncMock()
            with patch.object(s55_failure_boundary_chat, "logger", logger), patch.object(s55_failure_boundary_chat, "_finish_ai_chat_matcher", reply), self.assertLogs(logger, level="ERROR") as logs:
                await s55_failure_boundary_chat._stage_reject_empty_response(SimpleNamespace(response=""))
            self.assertIn("Traceback (most recent call last)", "\n".join(logs.output))
            self.assertIn("response contract returned empty content", "\n".join(logs.output))
            self.assertNotIn("重试", reply.await_args.args[0])
        asyncio.run(run())

    def test_only_transient_failure_invites_retry(self):
        logger = logging.getLogger("s55.classification")
        for error, transient in ((TimeoutError("timeout"), True), (ValueError("invalid code"), False)):
            with self.assertLogs(logger, level="ERROR"):
                reply = turn_failure_reply(logger, error, stage="fixture", step="核验编码")
            self.assertEqual("重试" in reply, transient)
        for status, transient in ((503, True), (400, False)):
            error = RuntimeError("upstream response")
            error.status_code = status
            with self.assertLogs(logger, level="ERROR"):
                reply = turn_failure_reply(logger, error, stage="fixture", step="核验编码")
            self.assertEqual("重试" in reply, transient)

    def test_discovery_exception_logs_traceback_and_names_failed_step(self):
        async def run():
            token = begin_turn_metrics("qq", "group")
            turn_id = current_turn_id()
            set_turn_flow("word-discovery")
            logger = logging.getLogger("s55.failure")
            reply = AsyncMock()
            try:
                with patch.object(s55_failure_boundary_chat, "STAGES", (s55_failure_boundary_chat._stage_handle_simple_word_query,)), patch.object(
                    s55_failure_boundary_chat, "get_history", return_value=[],
                ), patch.object(
                    s55_failure_boundary_chat, "_try_handle_explicit_reading_disambiguation", AsyncMock(return_value=None),
                ), patch.object(
                    s55_failure_boundary_chat, "_try_handle_compound_shift_modified_add_command", AsyncMock(return_value=None),
                ), patch.object(
                    s55_failure_boundary_chat, "_try_handle_shift_modified_add_command", AsyncMock(return_value=None),
                ), patch.object(
                    s55_failure_boundary_chat, "_try_handle_complete_add_command", AsyncMock(return_value=None),
                ), patch.object(
                    s55_failure_boundary_chat, "_try_recover_reviewed_add_from_history", AsyncMock(return_value=None),
                ), patch.object(
                    s55_failure_boundary_chat, "_try_handle_simple_single_word_query", AsyncMock(side_effect=ValueError("invalid single candidate qx")),
                ), patch.object(s55_failure_boundary_chat, "logger", logger), patch.object(
                    s55_failure_boundary_chat, "_finish_ai_chat_matcher", reply,
                ), self.assertLogs(logger, level="ERROR") as logs:
                    await s55_failure_boundary_chat._handle_ai_chat_serialized(SimpleNamespace(), SimpleNamespace(), "qq", "s55")
                text = "\n".join(logs.output)
                for expected in ("Traceback (most recent call last)", "ValueError", turn_id, "flow=word-discovery", "stage=_stage_handle_simple_word_query"):
                    self.assertIn(expected, text)
                self.assertEqual(reply.await_count, 1)
                user_line = reply.await_args.args[0]
                self.assertIn("审词", user_line)
                self.assertNotIn("请重试", user_line)
                self.assertNotIn("处理请求失败", user_line)
            finally:
                end_turn_metrics(token)
        asyncio.run(run())

# Source: test_s57_e2e_closure_review.py
class S57RigEvidenceReviewTests(unittest.TestCase):
    def test_model_filter_detects_actual_recorded_provider_transport_error(self):
        tree = ast.parse(Path("e2e/s57.py").read_text())
        scenario = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "scenario_s57")
        predicate = next(node for node in scenario.body if isinstance(node, ast.FunctionDef) and node.name == "model_calls")
        with tempfile.TemporaryDirectory() as directory:
            recorder = ArtifactRecorder(Path(directory) / "artifacts")
            with recorder.scope("S57", 1):
                recorder.record_http_error(
                    request=SimpleNamespace(
                        method="POST", url="https://api.deepseek.com/chat/completions",
                        content=b'{"model":"deepseek-v4-flash","messages":[]}',
                    ),
                    error=TimeoutError("fixture provider timeout"), elapsed_seconds=1.0,
                )
            namespace = {
                "ctx": SimpleNamespace(attempt_events=lambda: recorder.events_for("S57", 1)),
                "chat": SimpleNamespace(OPENAI_BASE_URL="https://api.deepseek.com"),
                "urlsplit": urlsplit,
                "Any": object,
            }
            exec(compile(ast.Module(body=[predicate], type_ignores=[]), "e2e/s57.py", "exec"), namespace)
            self.assertTrue(
                namespace["model_calls"](0),
                "A recorded provider request that timed out must disprove zero model calls",
            )

    def test_closure_rejects_new_force_advertisements_without_a_live_record(self):
        async def run():
            chat = harness.openai_chat_module
            address = harness.ConversationAddress.private("qq", "s57-rig-closure")
            for reply in ("回复「强制调序」即可执行。", "回复「照做」即可执行。"):
                with self.subTest(reply=reply), self.assertRaises(ScenarioAssertionError):
                    await _assert_s56_advertised_reply_closure(
                        reply, chat=chat, record=None, address=address,
                        read_draft=AsyncMock(side_effect=AssertionError("unexpected draft read")),
                    )
        asyncio.run(run())

    def test_closure_rejects_structural_options_without_a_live_record(self):
        async def run():
            with self.assertRaises(ScenarioAssertionError):
                await _assert_s56_advertised_reply_closure(
                    "请问是要强制调序，还是保留现顺序？",
                    chat=harness.openai_chat_module, record=None,
                    address=harness.ConversationAddress.private("qq", "s57-rig-options"),
                    read_draft=AsyncMock(side_effect=AssertionError("unexpected draft read")),
                )
        asyncio.run(run())

# Source: test_s63_general_reply.py
class GeneralReplyTests(unittest.IsolatedAsyncioTestCase):
    reply = GeneralReplyTests_reply

    async def test_chitchat_does_not_claim_an_adjustment(self):
        response = await self.reply("ennnn", "可以回复「嗯嗯」跟我聊天。")
        self.assertNotIn("调整", response)
        self.assertNotIn("未写入", response)
        self.assertFalse(advertised_command_suggestions(response))

    async def test_code_question_does_not_claim_an_adjustment(self):
        response = await self.reply("ennnn 是什么编码", "想查这个编码的话，可以回复「查询 ennnn」。")
        self.assertNotIn("调整", response)
        self.assertNotIn("未写入", response)
        self.assertFalse(advertised_command_suggestions(response))

    async def test_explicit_mutation_still_requires_a_real_plan(self):
        response = await self.reply("把 奈飞 调到 nhfw", "请回复「确认」。")
        self.assertEqual(response, NO_PLAN)

    async def test_unrecognized_mutation_verb_still_requires_a_real_plan(self):
        response = await self.reply("把 奈飞 迁到 nhfw", "请回复「确认」。")
        self.assertEqual(response, NO_PLAN)

    async def test_surrounding_read_only_text_is_preserved(self):
        response = await self.reply("ennnn", "你好。可以回复「嗯嗯」跟我聊天。")
        self.assertEqual(response, "你好。")

    async def test_real_confirmation_ticket_is_preserved(self):
        state = harness.PendingToolConfirm(
            "keytao_shift_phrase_code",
            {"word": "奈飞", "target_code": "nhfw", "batch_id": "",
             "expected_content_version": 0, "confirmed_plan_digest": "a" * 64},
            confirmation_source="server_warning",
        )
        response = await self.reply("把 奈飞 调到 nhfw", "请回复「确认」。", state)
        self.assertEqual(response, "请回复「确认」。")

    async def test_encode_then_rejected_create_does_not_invent_a_service_failure(self):
        def call(name, arguments):
            return SimpleNamespace(id="s63-" + name, type="function", function=SimpleNamespace(
                name=name, arguments=json.dumps(arguments),
            ))

        client = harness._FakeClient([
            harness._FakeAIResponse("tool_calls", "", [call("keytao_encode", {"word": "加亮", "requested_code": "jslxa"})]),
            harness._FakeAIResponse("tool_calls", "", [call("keytao_create_phrase", {"word": "加亮", "code": "jslxa"})]),
            harness._FakeAIResponse("stop", "尝试加入草稿时服务未完成处理，本次没有写入。"),
        ])
        dispatches = []

        async def encode(**_kwargs):
            dispatches.append("keytao_encode")
            return {"success": True, "word": "加亮", "input": "加亮", "recommendedCode": "jslxa",
                    "candidateCodes": ["jslxa"], "codes": ["jslxa"],
                    "candidateStatuses": [{"code": "jslxa", "occupied": False, "words": []}]}

        async def forbidden_create(**_kwargs):
            self.fail("The rejected create must not reach its sink")

        schemas = [{"type": "function", "function": {
            "name": name, "description": "Fixture operation", "parameters": {
                "type": "object", "properties": {key: {"type": "string"} for key in keys},
                "required": ["word"] if name == "keytao_encode" else ["word", "code"],
            },
        }} for name, keys in (("keytao_encode", ("word", "requested_code")),
                             ("keytao_create_phrase", ("word", "code")))]
        manager = SimpleNamespace(get_skill_instructions=lambda: "", has_tools=lambda: True, get_tools=lambda: schemas)
        store = harness.MemoryConversationStateStore()
        context = harness.AgentRequestContext(platform="qq", user_id="s63-policy", mutations_allowed=False)
        orchestrator = harness.AgentOrchestrator(
            client_factory=lambda: client,
            runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10.0),
            skills_manager=manager,
            tool_executor=harness.ToolExecutor(
                lambda name: encode if name == "keytao_encode" else forbidden_create, frozenset(),
            ),
            state_store=store, bind_help_text="fixture", system_prompt_core="fixture",
        )
        response = await orchestrator.run("加亮 jslxa", context)
        self.assertEqual(dispatches, ["keytao_encode"])
        self.assertIsNone(store.get_record(context.conversation_address))
        self.assertEqual(len(client.completions.calls), 2, "A local refusal must not request a fabricated explanation")
        self.assertIn("看到了操作请求", response)
        self.assertIn("还没有核对出可执行的词条和编码", response)
        self.assertIn("查看草稿", response)
        self.assertNotIn("没有明确要求", response)
        self.assertNotIn("服务未完成", response)

    async def test_exclusion_binding_refusal_keeps_its_distinct_reason(self):
        args = {"items": [
            {"action": "Create", "word": "显眼包", "code": "xybo"},
            {"action": "Create", "word": "嘴替", "code": "zbtk"},
        ]}
        agent, _client = policy_agent(
            [policy_call("keytao_batch_add_to_draft", args, 1)],
            "「都加」附带的排除条件无法确认，本次未写入。",
        )
        response = await agent.run("都加 跳过绝绝子", harness.AgentRequestContext(
            platform="qq", user_id="s63-exclusion", mutations_allowed=False,
        ))
        self.assertIn("排除条件", response)
        self.assertNotIn("没有明确要求", response)

    async def test_prior_successful_write_survives_an_early_submit_refusal(self):
        agent, client = policy_agent([
            policy_call("keytao_create_phrase", {"word": "炒冷饭", "code": "wlfoo"}, 1),
            policy_call("keytao_submit_batch", {}, 2),
        ])
        success = {
            "success": True, "batchId": "s63-partial", "batchUrl": "https://example.invalid/batch/s63-partial",
            "writtenItems": [{"id": "s63-row", "action": "Create", "word": "炒冷饭", "code": "wlfoo", "type": "Phrase"}],
        }
        refusal = {"success": False, "policyBlocked": True, "blockReason": "verb_not_matched",
                   "missing": ["executionVerb"], "message": "当前消息没有明确要求提交该批次；本次未写入。"}
        with patch.object(agent, "_call_tool_once", AsyncMock(side_effect=[
            json.dumps(success), json.dumps(refusal),
        ])) as execute:
            response = await agent.run("添加 炒冷饭 wlfoo", harness.AgentRequestContext(
                platform="qq", user_id="s63-partial", mutations_allowed=True,
            ))
        self.assertEqual(execute.await_count, 2)
        self.assertEqual(len(client.completions.calls), 1)
        for text in ("✅ 已加入草稿：", "炒冷饭", "wlfoo", "提交未完成", success["batchUrl"]):
            self.assertIn(text, response)
        self.assertNotIn("本次未写入", response)

# Source: test_s70_smalltalk.py
class SmallTalkTests(unittest.IsolatedAsyncioTestCase):
    setUp = SmallTalkTests_setUp

    runtime = FreshSelectionTests_runtime
    tool = SmallTalkTests_tool

    delivered = SmallTalkTests_delivered

    assert_natural = SmallTalkTests_assert_natural

    async def test_f4_m3_07_private_help_is_fixed_without_model_calls(self):
        expected = "我是喵喵，可以帮你查词、加词、改码、删词、提交键道词库草稿；绑定请在「我的资料」生成绑定码，使用 /bind 绑定码 完成绑定。"
        state = harness.PendingAddWord(
            word="加亮", recommended_code="jslxa", candidates=[("jslxa", False)],
            server_candidates=[("jslxa", False)],
        )
        self.store.set(self.key, state)
        with self.runtime(), patch.object(
            s70_smalltalk_chat, "extract_platform_info",
        ) as platform_info, patch.object(
            s70_smalltalk_chat, "extract_memory_context", AsyncMock(return_value=self.memory),
        ), patch.object(
            s70_smalltalk_chat, "extract_reply_reference_info", AsyncMock(return_value=harness.ReplyReferenceInfo()),
        ), patch.object(
            s70_smalltalk_chat, "AsyncOpenAI", side_effect=AssertionError("Model client reached"),
        ) as model_client, patch.object(
            s70_smalltalk_chat._user_resolver, "resolve_actor_binding", AsyncMock(return_value=False),
        ) as binding, patch.object(
            s70_smalltalk_chat, "_describe_images_for_deepseek_in_slot",
            AsyncMock(side_effect=AssertionError("Vision model reached")),
        ) as vision:
            for platform in ("qq", "telegram"):
                platform_info.return_value = (platform, self.key.actor_id)
                for message in ("/start", "/help", "/start"):
                    with self.subTest(platform=platform, message=message):
                        event = SimpleNamespace(
                            message=[SimpleNamespace(type="image", data={"url": "https://example.invalid/help.png"})],
                            get_plaintext=lambda: message,
                        )
                        await s70_smalltalk_chat._handle_ai_chat_serialized(
                            object(), event, platform, self.key.actor_id,
                        )
                        s70_smalltalk_chat._finish_ai_chat_matcher.assert_awaited_once_with(expected)
                        self.assertEqual(s70_smalltalk_chat._prepare_user_facing_reply(expected, self.memory), expected)
                        self.assertIs(self.store.get(self.key), state)
                        s70_smalltalk_chat._finish_ai_chat_matcher.reset_mock()
                        s70_smalltalk_chat._finish_ai_chat_response.assert_not_awaited()
            for platform in ("qq", "telegram", "web", "web-anon"):
                for message in ("/start", "/help"):
                    with self.subTest(core_platform=platform, message=message):
                        self.assertEqual(await REAL_CORE(message, platform, self.key.actor_id), expected)
            s70_smalltalk_chat.get_ai_response_core.assert_not_awaited()
            s70_smalltalk_chat._classify_message_command_intent.assert_not_awaited()
            s70_smalltalk_chat._classify_simple_word_query_intent.assert_not_awaited()
            model_client.assert_not_called()
            binding.assert_not_awaited()
            vision.assert_not_awaited()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.writes, [])

    def test_f4_m3_08_built_prompt_requires_docs_for_scheme_claims_and_conflicts(self):
        prompt = s70_smalltalk_chat.representative_system_prompt()
        for rule in (
            "用户对方案规则（码规则、顶功、笔画归码等）的陈述或提问都不是闲聊，必须先调用 keytao_fetch_docs 核实再认同，不得编造规则。",
            "用户纠正自己的前述说法或前后说法冲突时，不得两边都附和；涉及方案规则先查 keytao_fetch_docs，再指出哪种说法正确，无法核实时明确说不确定。",
        ):
            with self.subTest(rule=rule):
                self.assertTrue(rule in prompt, "The built prompt must require docs for scheme claims and conflicts")

    async def test_incident_fake_model_reply_is_redrawn_at_delivery(self):
        reply = await self.delivered("喵喵 早", NARRATION)
        self.assert_natural(reply)
        self.assertIn("早", reply)
        self.assertEqual(self.calls, [])

    async def test_all_greetings_and_unrelated_chat_use_the_same_no_intent_path(self):
        for message in ("早", "早上好", "晚安", "在吗", "谢谢", "辛苦了", "哈哈", "今天的云像棉花糖"):
            with self.subTest(message=message):
                self.assert_natural(await self.delivered(message, NARRATION))
        self.assertEqual(self.calls, [])

    async def test_unlisted_chitchat_keeps_its_relevant_natural_answer(self):
        reply = await self.delivered("今天的云像棉花糖", "软乎乎的，看得本喵都想咬一口～")
        self.assertEqual(reply, "软乎乎的，看得本喵都想咬一口～")

    async def test_synthetic_narration_is_rejected_at_the_final_boundary(self):
        for raw in (
            NARRATION, "用户想打个招呼。", "用户向我问好，回复问候即可。", "没有找到可执行命令。", "这条消息是问候。",
            "我决定不调用工具。", "不需要补充信息，等有具体需求再说即可。",
            "早呀！\n可发送「查看草稿」。", "早呀！\n1. 查询\n2. 加词",
        ):
            with self.subTest(raw=raw):
                self.assert_natural(await self.delivered("早", raw, inject_boundary=True))

    async def test_repeated_greetings_change_wording_without_another_model_call(self):
        for raw in (NARRATION, "早呀，喵～"):
            first = await self.delivered("早", raw)
            second = await self.delivered("早", raw, history=[
                {"role": "user", "content": "早"}, {"role": "assistant", "content": first},
            ])
            self.assert_natural(second)
            self.assertNotEqual(first, second)

    async def test_real_operation_and_read_only_query_are_unaffected(self):
        reply = await self.delivered("查看草稿", "unused")
        self.assertIn("草稿", reply)
        self.assertIn("keytao_get_batch_preview", [name for name, _ in self.calls])
        answer = "「你好」的编码是 nkhzia。\n这是查询结果。"
        self.assertEqual(await self.delivered("你好的编码是什么", answer, has_operation_intent=True), answer)

    async def test_incomplete_operation_still_requests_its_target(self):
        answer = "请告诉我要添加的词条。"
        self.assertEqual(await self.delivered("帮我加个词", answer, has_operation_intent=True), answer)

    async def test_function_questions_with_no_shortcut_are_not_claimed_as_chitchat(self):
        for message, answer in (
            ("怎么绑定账号？", "请使用 /bind 绑定账号。"),
            ("nhfw 下有哪些词？", "候选：奈飞。\n这是 fixture 查询结果。"),
        ):
            with self.subTest(message=message):
                self.assertEqual(await self.delivered(message, answer, has_operation_intent=True), answer)

    def test_even_incorrect_conversation_metadata_cannot_claim_explicit_operation(self):
        token = s70_smalltalk_chat._current_conversational_turn.set(None)
        try:
            s70_smalltalk_chat._set_conversational_turn("添加 加亮 jslxa", [], harness.MessageCommandIntent(has_operation_intent=False))
            self.assertIsNone(s70_smalltalk_chat._current_conversational_turn.get())
        finally:
            s70_smalltalk_chat._current_conversational_turn.reset(token)

    async def test_compound_goodnight_does_not_become_a_morning_greeting(self):
        reply = await self.delivered("晚安，明早见", NARRATION)
        self.assertIn("晚安", reply)
        self.assertNotIn("早呀", reply)

    async def test_web_core_and_delivery_share_the_guard_without_chat_stages(self):
        for platform in ("web", "web-anon"):
            client = harness._FakeClient([harness._FakeAIResponse("stop", NARRATION)])
            token = s70_smalltalk_chat._current_turn_message.set("")
            conversation_token = s70_smalltalk_chat._current_conversational_turn.set(None)
            try:
                with self.runtime(), patch.object(s70_smalltalk_chat, "OPENAI_API_KEY", "s70-fake-key"), patch.object(
                    s70_smalltalk_chat, "AsyncOpenAI", return_value=client,
                ), patch.object(s70_smalltalk_chat, "skills_manager", harness._FakeToolSkillsManager()), patch.object(
                    s70_smalltalk_chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent(
                        has_operation_intent=False,
                    )),
                ) as classify:
                    # The runtime fixture normally forbids core calls; use the saved real function.
                    reply = await REAL_CORE("早", platform, "s70-web", history=[])
                    self.assert_natural(s70_smalltalk_chat._prepare_user_facing_reply(reply, None))
                    self.assert_natural(s70_smalltalk_chat._prepare_user_facing_reply(NARRATION, None))
                    classify.assert_awaited_once_with("早")
                    self.assertEqual(len(client.completions.calls), 1)
                    self.assertFalse(client.completions.calls[0].get("tools"))
            finally:
                s70_smalltalk_chat._current_conversational_turn.reset(conversation_token)
                s70_smalltalk_chat._current_turn_message.reset(token)

    async def test_semantic_router_carries_absent_intent_without_a_greeting_table(self):
        routing = s70_smalltalk_chat._chat_routing
        for message, has_operation in (("早", False), ("今天的云像棉花糖", False), ("怎么绑定账号？", True)):
            client = harness._FakeClient([harness._FakeAIResponse("stop", json.dumps({
                "intent": "none", "confidence": 0.99, "has_operation_intent": has_operation,
            }))])
            with patch.object(routing, "OPENAI_API_KEY", "s70-fake-key"), patch.object(
                routing, "AsyncOpenAI", return_value=client,
            ):
                intent = await routing._classify_message_command_intent(message)
            self.assertIs(intent.has_operation_intent, has_operation)
            self.assertEqual(intent.intent, "none")
            self.assertEqual(len(client.completions.calls), 1)

    def test_operation_metadata_is_independent_and_unknown_is_not_false(self):
        parse = s70_smalltalk_chat._chat_routing._parse_message_command_intent_payload
        for value in (True, False):
            self.assertIs(parse({"intent": "none", "has_operation_intent": value}).has_operation_intent, value)
        for payload in ({}, {"has_operation_intent": "false"}, {"has_operation_intent": 0}):
            self.assertIsNone(parse(payload).has_operation_intent)
        token = s70_smalltalk_chat._current_conversational_turn.set(None)
        try:
            for intent, other in (
                (harness.MessageCommandIntent(), ()),
                (harness.MessageCommandIntent(has_operation_intent=False),
                 (harness.MessageCommandIntent(has_operation_intent=True),)),
            ):
                s70_smalltalk_chat._set_conversational_turn("fixture", [], intent, other_intents=other)
                self.assertIsNone(s70_smalltalk_chat._current_conversational_turn.get())
        finally:
            s70_smalltalk_chat._current_conversational_turn.reset(token)

    async def test_live_unrelated_candidate_does_not_hijack_a_greeting(self):
        state = harness.PendingAddWord(
            word="加亮", recommended_code="jslxa", candidates=[("jslxa", False)],
            server_candidates=[("jslxa", False)],
        )
        self.store.set(self.key, state)
        reply = await self.delivered("谢谢", NARRATION)
        self.assert_natural(reply)
        self.assertIs(self.store.get(self.key), state)
        self.assertEqual(self.calls, [])

    def test_natural_copy_does_not_confuse_product_users_with_narration(self):
        self.assertEqual(enforce_conversational_reply("新界面怎么样", "用户体验挺顺手的喵。"), "用户体验挺顺手的喵。")

# Source: test_s70b_query_precedence.py
class QueryPrecedenceTests(unittest.IsolatedAsyncioTestCase):
    setUp = SmallTalkTests_setUp
    runtime = FreshSelectionTests_runtime
    assert_natural = SmallTalkTests_assert_natural

    tool = QueryPrecedenceTests_tool

    delivered = QueryPrecedenceTests_delivered

    async def test_interrogative_characters_inside_words_preserve_lexical_routes(self):
        from keytao_bot.utils.explicit_code import parse_explicit_entry_code_request

        grammar = s70_smalltalk_chat._authorization_grammar
        for message in ("鹿死谁手 lsss 加入", "加入 舍我其谁 wwqs", "加入 怎么着 zmzv"):
            with self.subTest(message=message):
                self.assertIsNotNone(parse_explicit_entry_code_request(message))
        self.assertEqual(
            b17_rest_recommended_routing._extract_explicit_reviewed_add_word("加词 鹿死谁手"),
            "鹿死谁手",
        )
        for word in ("不怎么样", "哪里哪里", "何时", "谁料", "非谁莫属"):
            with self.subTest(word=word):
                self.assertEqual(await b17_rest_recommended_routing._get_simple_word_query_words(word), (word,))
        self.assertTrue(grammar.is_interrogative_message("什么时候在词库的"))

    async def test_f2_m1_01_bare_words_override_semantic_chat(self):
        from tests.test_add_and_select import B18RealToolTests

        fixture = B18RealToolTests()
        fixture.addCleanup = self.addCleanup
        fixture.setUp()
        facts = {"哈基米": ("hjm", "hā jī mǐ"), "魔了": ("mole", "mó le"),
                 "南北绿豆": ("nbld", "nán běi lǜ dòu")}
        fixture.install_f2_review_http(facts, set())
        fixture.http.entries[:] = [{"word": "哈基米", "code": "hjm", "type": "Phrase", "weight": 100}]
        try:
            for word, meta in (
                ("哈基米", "目前没有对应的操作请求。"), ("魔了", "这轮是普通问答。"),
                ("哈基米", "“哈基米”是常见网络用语，目前没有对应的操作请求。"),
                ("南北绿豆", "当前没有可验证的可执行操作。"),
                ("南北绿豆", "没找到要执行的操作，本次未写入。"),
                ("魔了", "这轮是普通问答，没有触发写入或安全拦截。"),
            ):
                with self.subTest(word=word):
                    fixture.replay.store.delete(fixture.replay.author)
                    fixture.replay.history.clear()

                    async def model(text, *_args, **_kwargs):
                        return await GeneralReplyTests_reply(self, text, meta)

                    with patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": "query,chat"}), patch.object(
                        fixture.replay.chat, "get_ai_response_core", side_effect=model,
                    ):
                        await fixture.turn(word, raw_operation(op="chat", operands=[], codes=[], frame="none"))
                    reply = fixture.replay.last_response
                    self.assertIn(word, reply)
                    self.assertNotIn(meta, reply)
                    if word == "哈基米":
                        self.assertIn("已在词库", reply)
                        self.assertIn("hjm", reply)
                        self.assertIsNone(fixture.replay.store.get(fixture.replay.author))
                    else:
                        self.assertIn("— ✅推荐", reply)
                        self.assertIsNotNone(fixture.replay.store.get(fixture.replay.author))
                    self.assertEqual(fixture.replay.chat._finish_ai_chat_response.await_args.args[4], reply)
            self.assertEqual(fixture.http.writes, [])
        finally:
            fixture.tearDown()

    async def test_semantic_chat_keeps_short_smalltalk_without_add_tickets(self):
        from tests.test_add_and_select import B18RealToolTests

        fixture = B18RealToolTests()
        fixture.addCleanup = self.addCleanup
        fixture.setUp()
        fixture.install_f2_review_http({}, set())
        try:
            for message in ("你好可爱", "今天好热", "这个不错", "刚才那个词", "好的谢谢", "原来如此"):
                for answer in ("喵，听着呢。", "目前没有对应的操作请求。", "这轮是普通问答，没有触发写入或安全拦截。"):
                    async def model(text, *_args, **_kwargs):
                        return await GeneralReplyTests_reply(self, text, answer)

                    with self.subTest(message=message, answer=answer), patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": "query,chat"}), patch.object(
                        fixture.replay.chat, "get_ai_response_core", side_effect=model,
                    ):
                        fixture.replay.history.clear()
                        fixture.http.requests.clear()
                        await fixture.turn(message, raw_operation(op="chat", operands=[], codes=[], frame="none"))
                        if answer == "喵，听着呢。":
                            self.assertEqual(fixture.replay.last_response, answer)
                        for marker in ("普通问答", "对应的操作请求", "可验证的可执行操作", "安全拦截"):
                            self.assertNotIn(marker, fixture.replay.last_response)
                        self.assertNotIn("推荐编码", fixture.replay.last_response)
                        self.assertIsNone(fixture.replay.store.get(fixture.replay.author))
                        self.assertEqual(fixture.http.requests, [])
                        self.assertEqual(fixture.http.writes, [])
        finally:
            fixture.tearDown()

    async def test_f2_m1_03_followup_question_is_answered_once_without_review(self):
        message = "什么时候在词库的"
        answer = "暂时查不到「冲锋衣」的收录时间。"
        history = []

        def remember(_key, _memory, text, response):
            history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": response}])

        tokens = [(variable, variable.set(value)) for variable, value in (
            (s70_smalltalk_chat._current_turn_message, ""),
            (s70_smalltalk_chat._current_conversational_turn, None),
            (s70_smalltalk_chat.current_operation_capture, None),
            (s70_smalltalk_chat.current_draft_delivery_claims, []),
        )]
        try:
            with self.runtime(), patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": ""}), patch.object(
                s70_smalltalk_chat, "extract_memory_context", AsyncMock(return_value=self.memory),
            ), patch.object(s70_smalltalk_chat, "get_history", side_effect=lambda *_args: history), patch.object(
                s70_smalltalk_chat, "remember_conversation", side_effect=remember,
            ), patch.object(s70_smalltalk_chat, "build_reply_context", AsyncMock(return_value="")), patch.object(
                s70_smalltalk_chat, "get_ai_response_core", AsyncMock(return_value=answer),
            ) as model, patch.object(
                s70_smalltalk_chat, "_classify_message_command_intent",
                AsyncMock(return_value=harness.MessageCommandIntent(has_operation_intent=False)),
            ), patch.object(
                s70_smalltalk_chat, "_classify_simple_word_query_intent",
                AsyncMock(return_value=harness.SimpleWordQueryIntent(False)),
            ):
                for text in ("冲锋衣", message):
                    previous_calls = list(self.calls)
                    event = SimpleNamespace(message=[], get_plaintext=lambda: text)
                    await s70_smalltalk_chat._handle_ai_chat_serialized(object(), event, "qq", self.key.actor_id)
                    if text == message:
                        self.assertEqual(self.calls, previous_calls)
                delivered = s70_smalltalk_chat._finish_ai_chat_response.await_args_list
                self.assertEqual(len(delivered), 2)
                self.assertIn("「冲锋衣」已在词库", str(delivered[0].args[4]))
                self.assertEqual(delivered[1].args[4], answer)
                model.assert_awaited_once()
                self.assertEqual(model.await_args.args[0], message)
                self.assertIn("冲锋衣", str(model.await_args.args[3]))
                s70_smalltalk_chat._finish_ai_chat_matcher.assert_not_awaited()
            self.assertIsNone(self.store.get(self.key))
            self.assertIn(("keytao_lookup_by_word", {"word": "冲锋衣"}), self.calls)
            self.assertEqual(await b17_rest_recommended_routing._get_simple_word_query_words(message), ())
        finally:
            for variable, token in reversed(tokens):
                variable.reset(token)

    async def test_f2_m2_6_long_sentence_add_gets_short_rejection(self):
        message = "加词：“床前看月光，疑是地上霜，举头望山月，低头思故乡”"
        client = FakeClient(json.dumps(raw_operation(op="chat", operands=[], codes=[], frame="none")))
        ctx = s70_smalltalk_chat.TurnContext(
            bot=object(), event=object(), platform="qq", user_id=self.key.actor_id,
            message_text=message, normalized_message_text=message, conv_key=self.key,
            space_key=self.key.space_key, memory_context=self.memory, history=[],
        )
        with self.runtime(), patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": "query,chat"}), patch.object(
            s70_smalltalk_chat, "AsyncOpenAI", return_value=client,
        ), patch.object(s70_smalltalk_chat, "get_ai_response_core", AsyncMock(return_value="这次服务没有完成处理，本次未写入。")):
            self.assertTrue(await s70_smalltalk_chat._stage_understand_operation_early(ctx))
        self.assertEqual(ctx.response, "这是句子，请添加其中的单个词语。")
        self.assertEqual(self.calls, [])
        self.assertFalse(client.create.await_count)

        self.assertEqual(b17_rest_recommended_routing.explicit_add_sentence_rejection(
            "加词：显眼包、嘴替、松弛感、电子榨菜、情绪价值、班味、泼天富贵、精神状态、职场搭子",
        ), "")

    async def test_bare_words_with_move_verb_prefixes_query_without_classifier(self):
        for word in ("移民", "调查", "挪威", "调整", "移动", "调值"):
            with self.subTest(word=word), patch.object(
                b17_rest_recommended_routing, "_classify_simple_word_query_intent",
                AsyncMock(side_effect=AssertionError("Unexpected intent classifier")),
            ) as classifier:
                self.assertTrue(b17_rest_recommended_routing._is_bare_word_query_target(word))
                self.assertEqual(await b17_rest_recommended_routing._get_simple_word_query_words(word), (word,))
                classifier.assert_not_awaited()

    async def test_query_matrix_wins_over_no_intent_classifier(self):
        for message, words in (
            ("练枪 得吃", ("练枪", "得吃")),
            ("@喵喵 练枪 得吃", ("练枪", "得吃")),
            ("蛋粉 单份", ("蛋粉", "单份")),
            ("敲不死", ("敲不死",)),
            ("共和国", ("共和国",)),
            ("和平 共处", ("和平", "共处")),
            ("一力降十会（yi2 li4 xiang2 shi2 hui4）", ("一力降十会",)),
        ):
            with self.subTest(message=message):
                self.calls.clear()
                reply, model_calls = await self.delivered(message)
                self.assertEqual([args["word"] for name, args in self.calls
                                  if name == "keytao_lookup_by_word"], list(words))
                self.assertEqual(model_calls, [])
                for word in words:
                    self.assertIn(word, reply)

    async def test_chat_matrix_has_natural_reply_and_no_exposed_tools(self):
        for message in ("早", "晚安", "谢谢", "在吗", "辛苦了", "晚安，明早见", "今天的云像棉花糖", "哈哈哈"):
            with self.subTest(message=message):
                self.calls.clear()
                reply, model_calls = await self.delivered(message)
                self.assert_natural(reply)
                self.assertEqual(self.calls, [])
                self.assertEqual(len(model_calls), 1)
                self.assertFalse(model_calls[0].get("tools"))

    async def test_separators_and_mixed_greetings_still_query_every_word(self):
        for separator in (" ", "\n", "\t", "，", "、", "；", ";", "和", " 和 "):
            for words in (("练枪", "得吃"), ("谢谢", "练枪")):
                message = separator.join(words)
                with self.subTest(message=message):
                    self.calls.clear()
                    reply, model_calls = await self.delivered(message)
                    self.assertEqual([args["word"] for name, args in self.calls
                                      if name == "keytao_lookup_by_word"], list(words))
                    self.assertEqual(model_calls, [])
                    for word in words:
                        self.assertIn(word, reply)

    async def test_explicit_add_and_explicit_greeting_lookup_are_not_chat(self):
        for message, word in (("加词 加亮", "加亮"), ("查词 谢谢", "谢谢")):
            with self.subTest(message=message):
                self.calls.clear()
                self.missing_word = message.startswith("加词")
                self.pending_items = []
                reply, model_calls = await self.delivered(message)
                self.assertEqual([args["word"] for name, args in self.calls
                                  if name == "keytao_lookup_by_word"], [word])
                self.assertEqual(model_calls, [])
                self.assertIn(word, reply)
                if self.missing_word:
                    self.assertEqual([args["word"] for name, args in self.calls
                                      if name == "keytao_prepare_reviewed_add"], [word])
                    self.assertIsNotNone(self.store.get(self.key))

    async def test_queries_prepare_missing_words_with_no_intent_classifier(self):
        self.missing_word = True
        self.pending_items = []

        original_tool = self.tool
        # Keep the fake transport separate from the production review/record path.
        async def wrapped_tool(name, args, platform, user_id, **kwargs):
            if name == "keytao_prepare_reviewed_add":
                self.calls.append((name, args))
                return json.dumps(dict(self.review, word=args["word"]))
            return await original_tool(name, args, platform, user_id, **kwargs)
        self.tool = wrapped_tool
        reply, model_calls = await self.delivered("练枪 得吃")
        self.assertEqual([args["word"] for name, args in self.calls
                          if name == "keytao_prepare_reviewed_add"], ["练枪", "得吃"])
        self.assertEqual(model_calls, [])
        self.assertIn("练枪", reply)
        self.assertIn("得吃", reply)
        self.assertEqual([row["word"] for row in self.store.get(self.key).args["_candidate_scopes"]],
                         ["练枪", "得吃"])

    async def test_prose_and_comparison_clauses_do_not_become_bare_word_queries(self):
        for message in ("他说加入", "这个和电机哪个常用", "帮我加个词"):
            with self.subTest(message=message), patch.object(
                s70_smalltalk_chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(False)),
            ), patch.object(s70_smalltalk_chat, "call_tool_function", AsyncMock(side_effect=AssertionError("Unexpected lookup"))):
                self.assertIsNone(await harness.chat_commands_module._try_handle_simple_single_word_query(
                    message, "qq", self.key.actor_id,
                ))

    async def test_web_core_cannot_redraw_a_query_or_hide_its_tools(self):
        for platform in ("web", "web-anon"):
            for message in ("练枪 得吃", "敲不死", "一力降十会（yi2 li4 xiang2 shi2 hui4）", "谢谢 练枪"):
                with self.subTest(platform=platform, message=message):
                    client = harness._FakeClient([harness._FakeAIResponse("stop", "Fixture query response")])
                    token = s70_smalltalk_chat._current_turn_message.set(message)
                    conversation_token = s70_smalltalk_chat._current_conversational_turn.set(None)
                    try:
                        with self.runtime(), patch.object(s70_smalltalk_chat, "OPENAI_API_KEY", "s70b-fixture"), patch.object(
                            s70_smalltalk_chat, "AsyncOpenAI", return_value=client,
                        ), patch.object(s70_smalltalk_chat, "skills_manager", harness._FakeToolSkillsManager()), patch.object(
                            s70_smalltalk_chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent(
                                has_operation_intent=False,
                            )),
                        ):
                            await REAL_CORE(message, platform, "s70b-web", history=[])
                            self.assertIsNone(s70_smalltalk_chat._current_conversational_turn.get())
                            self.assertTrue(client.completions.calls[0].get("tools"))
                            self.assertEqual(s70_smalltalk_chat._prepare_user_facing_reply("Fixture query response", None),
                                             "Fixture query response")
                    finally:
                        s70_smalltalk_chat._current_conversational_turn.reset(conversation_token)
                        s70_smalltalk_chat._current_turn_message.reset(token)

    def test_whole_message_discriminator_never_authorizes_writes(self):
        for message in ("早", "谢谢", "晚安，明早见", "哈哈哈", "今天的云像棉花糖"):
            with self.subTest(message=message):
                self.assertTrue(s70_smalltalk_chat._chat_routing.is_conversational_text(message))
                self.assertFalse(s70_smalltalk_chat.message_authorizes_mutation(message))
        for message in ("谢谢 练枪", "早，查词练枪", "我觉得需要查词", "我感觉编码有问题",
                        "今天的云像棉花糖 练枪", "一力降十会（yi2 li4 xiang2 shi2 hui4）"):
            with self.subTest(message=message):
                self.assertFalse(s70_smalltalk_chat._chat_routing.is_conversational_text(message))

# Source: test_s71_stale_ticket_release.py
class StaleTicketReleaseTests(unittest.IsolatedAsyncioTestCase):
    replay = StaleTicketReleaseTests_replay

    assert_fresh_request = StaleTicketReleaseTests_assert_fresh_request

    async def test_incident_unrelated_bot_quote(self):
        await self.assert_fresh_request("把这个加进词库", QUOTE)

    async def test_deictic_ticket_commands_keep_head_behavior(self):
        # Recorded on HEAD 573a37e: all ten turns reject and deliver the live render.
        for message in (
            "加进草稿", "把这两个加进词库", "加入这两个词", "加上吧", "都加进词库吧",
        ):
            for quote in ("", "ticket"):
                with self.subTest(message=message, quote=quote):
                    result = await self.replay(message, quote)
                    self.assertEqual(
                        (result["flow"], result["stages"][-1], result["delivered"]),
                        ("pending-confirmation", "_stage_finish_scoped_pending_response",
                         [result["old_render"]]),
                    )
                    self.assertEqual(result["record"], result["before"])
                    result["execute"].assert_not_awaited()
                    result["submit"].assert_not_awaited()
                    result["model"].assert_not_awaited()
                    self.assertEqual(result["calls"], [])

    async def test_deictic_add_commands_reach_model_without_ticket(self):
        for message in ("把这个加进词库", "把这句话加进词库", "加上这个"):
            for quote in ("", QUOTE):
                with self.subTest(message=message, quote=quote):
                    result = await self.replay(message, quote, live=False)
                    result["model"].assert_awaited_once()
                    self.assertIn(quote, result["model"].await_args.args[4])
                    self.assertEqual(result["delivered"], [MODEL_REPLY])
                    self.assertFalse(any(name == "keytao_prepare_reviewed_add"
                                         for name, _args in result["calls"]))

    async def test_quoted_ticket_words_and_candidate_codes_keep_head_behavior(self):
        for quote in ("own-command", "你都没有上手", "候选 NDMEV", "候选ｙｄｂｊｖ"):
            with self.subTest(quote=quote):
                result = await self.replay("加上吧", quote)
                self.assertEqual(
                    (result["flow"], result["stages"][-1], result["delivered"]),
                    ("pending-confirmation", "_stage_finish_scoped_pending_response",
                     [result["old_render"]]),
                )
                self.assertFalse(result["ctx"].verified_current_pending_reply)
                self.assertEqual(result["record"], result["before"])
                result["execute"].assert_not_awaited()
                result["submit"].assert_not_awaited()
                result["model"].assert_not_awaited()
                self.assertEqual(result["calls"], [])

    async def test_other_quoted_deictic_add_commands_are_released(self):
        for message in (
            "加进草稿", "把这两个加进词库", "加入这两个词", "加上吧", "都加进词库吧",
            "把这句话加进词库", "加上这个",
        ):
            with self.subTest(message=message):
                await self.assert_fresh_request(message, QUOTE)

    async def test_released_model_failure_does_not_reenter_ticket(self):
        message = "加欲买桂花同载酒，终不似少年游！这句话"
        result = await self.replay(message, "ticket", model_failure=True)
        baseline = await self.replay(message, "ticket", live=False, model_failure=True)
        self.assertTrue(result["ctx"].released_pending_turn)
        result["model"].assert_awaited_once()
        self.assertEqual(result["completion"].await_count, 2)
        result["fallback_pending"].assert_not_awaited()
        baseline["fallback_pending"].assert_awaited_once()
        self.assertEqual(result["delivered"], baseline["delivered"])
        self.assertEqual(result["delivered"], ["这次服务没有完成处理，本次未写入。"])
        self.assertNotIn(result["old_render"], result["delivered"])
        self.assertEqual(result["record"], result["before"])
        result["execute"].assert_not_awaited()
        result["submit"].assert_not_awaited()
        result["schedule"].assert_not_called()
        self.assertTrue(all(name == "keytao_lookup_by_words_batch"
                            for name, _args in result["calls"]))

    async def test_matched_assent_to_other_quote_does_not_bind_ticket(self):
        result = await self.replay("把这个加入草稿", QUOTE)
        self.assertTrue(result["ctx"].released_pending_turn)
        self.assertNotEqual(result["flow"], "pending-confirmation")
        self.assertEqual(result["delivered"], [MODEL_REPLY])
        result["model"].assert_awaited_once()
        self.assertIn(QUOTE, result["model"].await_args.args[4])
        self.assertEqual(result["record"], result["before"])
        result["execute"].assert_not_awaited()
        result["submit"].assert_not_awaited()
        result["schedule"].assert_not_called()
        self.assertFalse(any(name == "keytao_prepare_reviewed_add"
                             for name, _args in result["calls"]))

    async def test_incident_new_poem_quoting_old_ticket(self):
        await self.assert_fresh_request("加欲买桂花同载酒，终不似少年游！这句话", "ticket")

    async def test_incident_explicit_new_word(self):
        await self.assert_fresh_request("加词 测试词", reviewed_word="测试词")

    async def test_ticket_assent_and_selections_remain_bound(self):
        # Selection receipts retain skipped words without echoing the chosen words.
        for message, expected, receipt in (
            ("加入", [("你都没有上手", "ndme"), ("有点不切实际", "ydbjvi")], ""),
            ("都加", [("你都没有上手", "ndme"), ("有点不切实际", "ydbjvi")], ""),
            ("你都没有上手 1，有点不切实际 3", [("你都没有上手", "ndme"), ("有点不切实际", "ydbjvi")],
             ""),
            ("有点不切实际 3", [("有点不切实际", "ydbjvi")],
             "\n未选择：「你都没有上手」；这些词本次未添加。"),
        ):
            for quote in ("", "ticket", QUOTE, "own-command"):
                with self.subTest(message=message, quote=quote):
                    result = await self.replay(message, quote)
                    self.assertEqual(result["flow"], "pending-confirmation")
                    self.assertFalse(result["ctx"].generic_intent_is_fresh_command)
                    self.assertFalse(getattr(result["ctx"], "released_pending_turn", False))
                    self.assertFalse(result["ctx"].verified_current_pending_reply)
                    result["execute"].assert_awaited_once()
                    items = result["execute"].await_args.args[0].args["items"]
                    self.assertEqual([(item["word"], item["code"]) for item in items], expected)
                    self.assertEqual(result["delivered"], ["Fixture ticket executed." + receipt])
                    result["model"].assert_not_awaited()
                    result["submit"].assert_not_awaited()
                    self.assertEqual(result["calls"], [])

    async def test_ticket_add_and_submit_remains_bound(self):
        for quote in ("", "ticket", QUOTE, "own-command"):
            with self.subTest(quote=quote):
                result = await self.replay("加入并提交", quote)
                self.assertEqual(result["flow"], "pending-confirmation")
                self.assertFalse(getattr(result["ctx"], "released_pending_turn", False))
                result["submit"].assert_awaited_once()
                self.assertEqual(result["submit"].await_args.args[0], result["before"].state.args["items"])
                self.assertEqual(result["delivered"], ["Fixture ticket submitted."])
                result["model"].assert_not_awaited()
                result["execute"].assert_not_awaited()
                self.assertEqual(result["calls"], [])

    async def test_ticket_cancel_remains_bound(self):
        result = await self.replay("取消")
        self.assertEqual(result["delivered"], ["已取消。"])
        self.assertIsNone(result["record"])
        result["execute"].assert_not_awaited()
        result["model"].assert_not_awaited()

    async def test_malformed_ticket_reference_is_not_released(self):
        for message in (
            "加入 你都没有上手 abc", "加入你都没有上手额外内容",
            "加入编码ndme额外内容", "加入编码YDBJV额外内容", "加入编码ｙｄｂｊｖ额外内容",
            "加入吗？", "他说加入", "不要加入其他词", "加入并删除其他词",
        ):
            with self.subTest(message=message):
                result = await self.replay(message)
                self.assertEqual(result["flow"], "pending-confirmation")
                self.assertEqual(result["stages"][-1], "_stage_finish_scoped_pending_response")
                self.assertEqual(result["record"], result["before"])
                result["execute"].assert_not_awaited()
                result["model"].assert_not_awaited()


class F2DraftSubmitTests(unittest.IsolatedAsyncioTestCase):
    async def test_f2_m3_05_publish_after_real_draft_receipt_submits_own_draft(self):
        from tests.helpers import (
            B18RealToolTests_setUp, B18RealToolTests_seed, B18RealToolTests_turn,
            TOOLS, b19_partial_reply_binding_DraftHTTP,
        )

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
        for command in ("发布", "提交"):
            with self.subTest(command=command):
                self.replay.store.delete(self.replay.author)
                self.http = b19_partial_reply_binding_DraftHTTP()
                self.enterContext(patch.object(TOOLS.http_client, "_keytao_client", self.http))
                B18RealToolTests_seed(self, ("招笑",), reorder=False)
                await B18RealToolTests_turn(self, "招笑 2，加入")
                self.assertTrue(self.http.writes, self.replay.last_response)
                self.assertEqual(self.http.submissions, 0)
                receipt = self.replay.store.get(self.replay.author)
                self.assertIsInstance(receipt, PendingToolConfirm)
                self.assertTrue(receipt.args.get("_recent_own_write"), receipt)
                await B18RealToolTests_turn(self, command, raw_operation(op="query", operands=["发布"], codes=[]))
                self.assertEqual(self.http.submissions, 1, self.replay.last_response)
                self.assertNotIn("「发布」已在词库", self.replay.last_response)
                self.assertEqual(self.http.errors, [])


def load_tests(loader, standard_tests, pattern):
    return _stage_t_unittest.TestSuite(
        loader.loadTestsFromTestCase(case) for case in (
            F2DraftSubmitTests,
            B6ReminderTests,
            B6SubmitTests,
            B6ReminderIntegrationTests,
            B7BlockedToolLoopTests,
            B5ReplySafetyTests,
            B7PhrasingRouteTests,
            B7QueryRouteTests,
            B8CancelTests,
            B9ParallelTests,
            B9WordQueryTests,
            B9CancelTests,
            B9CandidateTests,
            B9RoutingSpeedTests,
            B9LookupTests,
            B9PronunciationEvidenceTests,
            FailureBoundaryTests,
            S57RigEvidenceReviewTests,
            GeneralReplyTests,
            SmallTalkTests,
            QueryPrecedenceTests,
            StaleTicketReleaseTests,
            B5DeleteTests,
        )
    )


if __name__ == "__main__":
    _stage_t_unittest.main()
