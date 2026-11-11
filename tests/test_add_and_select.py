"""Topic regressions: add and select."""

import unittest as _stage_t_unittest
from tests.helpers import (
    ANSWER,
    AddKeepsSubmitTests_answer,
    AddKeepsSubmitTests_offer,
    AddKeepsSubmitTests_preview,
    AddKeepsSubmitTests_setUp,
    AddKeepsSubmitTests_tool,
    AsyncMock,
    B12SelectionTests_assert_offline_route,
    B12SelectionTests_candidates,
    B12SelectionTests_created,
    B12SelectionTests_setUp,
    B12SelectionTests_turn,
    B14MultiSelectionTests_assert_offline_selection,
    B14MultiSelectionTests_candidates,
    B14MultiSelectionTests_created,
    B14MultiSelectionTests_select,
    B14MultiSelectionTests_setUp,
    B15ChatTests_candidates,
    B15ChatTests_created,
    B15ChatTests_reset_ticket,
    B15ChatTests_setUp,
    B15ChatTests_submissions,
    B15ChatTests_turn,
    B15PendingLookupTests_batch,
    B15PendingLookupTests_lookup,
    B15PendingLookupTests_setUp,
    B17RestRouteTests_assert_offline,
    B17RestRouteTests_candidates,
    B17RestRouteTests_check_failed_then_corrected,
    B17RestRouteTests_created,
    B17RestRouteTests_select,
    B17RestRouteTests_setUp,
    B17RestRouteTests_submissions,
    B18RealToolTests_created,
    B18RealToolTests_seed,
    B18RealToolTests_setUp,
    B18RealToolTests_tearDown,
    B18RealToolTests_turn,
    B18ShiftValidationTests_setUp,
    B7SelectionTests_resolve,
    B8PresentWordsTests_add,
    B8PresentWordsTests_assert_candidates,
    B8PresentWordsTests_assert_present_copy,
    B8PresentWordsTests_asyncSetUp,
    B8PresentWordsTests_asyncTearDown,
    B8PresentWordsTests_lookup,
    B8PresentWordsTests_present,
    BATCH_URL,
    DraftOperationCoordinator,
    ExitStack,
    ExplicitSubmitTests_assert_submitted_or_actionable_submit_ticket,
    ExplicitSubmitTests_executor_runtime,
    ExplicitSubmitTests_runtime,
    ExplicitSubmitTests_seed_existing_draft,
    ExplicitSubmitTests_setUp,
    ExplicitSubmitTests_tool,
    ExplicitSubmitTests_turn,
    FORCE_ASSENT_TEXTS,
    FreshSelectionTests_runtime,
    FreshSelectionTests_setUp,
    FreshSelectionTests_tool,
    FreshSelectionTests_turn,
    GenericProposalBridgeTests_assert_ticket,
    GenericProposalBridgeTests_call,
    GenericProposalBridgeTests_setUp,
    HotfixMultiwordTests__check_standalone_shift_real_server_plan,
    HotfixMultiwordTests_assert_candidates,
    HotfixMultiwordTests_asyncSetUp,
    HotfixMultiwordTests_lookup,
    HotfixMultiwordTests_query,
    HotfixMultiwordTests_unresolved_message,
    HotfixMultiwordTests_unresolved_review,
    INSTRUCTION,
    MultiwordRouteTests_discover,
    OTHER_BATCH,
    OWN_BATCH,
    OperationRequest,
    OrderedDict,
    OfflineResponse,
    Path,
    PendingAddWord,
    PendingToolConfirm,
    S54SelectionTests_reordered_record,
    S62IncidentTests_replay,
    S62SelectionTests_execute_selection,
    SQLiteConversationStateStore,
    SelectionTests_prepare,
    SelectionTests_setUp,
    SelectionTests_tool,
    ShiftReplay,
    SimpleNamespace,
    T3,
    TOOLS,
    ToolContext,
    ToolExecutor,
    TranscriptTests_delivered_turn,
    TranscriptTests_executor_runtime,
    advertised_command_suggestions,
    advertised_reply_contract,
    already_existing_word_copy,
    asyncio,
    b11_shift_WORD,
    b14_multi_select_FACTS,
    b14_multi_select_SELECTION_ERROR,
    b14_multi_select_T2,
    b15_sibling_and_others_WORDS,
    b17_rest_recommended_FACTS,
    b17_rest_recommended_SELECTION_ERROR,
    b17_rest_recommended_TEN_WORDS,
    b17_rest_recommended_commands,
    b17_rest_recommended_routing,
    b17_rest_recommended_shift_ticket,
    b18_rest_real_tools_FACTS,
    b18_rest_real_tools_TEN_WORDS,
    b19_partial_reply_binding_DraftHTTP,
    call_tool_function,
    copy,
    grammar,
    harness,
    incident_ticket,
    is_force_assent,
    json,
    matches_previous_bot_option,
    negative_parser_case,
    notice,
    offered_option_intent,
    option_questions_bind_live_state,
    os,
    parse_eviction_modified_add,
    parse_explicit_entry_code_request,
    parse_pending_candidate_selection,
    patch,
    pending,
    pending_duplicate_confirmation_is_complete,
    pending_duplicate_confirmation_state,
    pending_item,
    positive_parser_case,
    preview,
    previous_bot_options,
    query,
    raw_operation,
    re,
    render_missing_option_ticket,
    render_remediation_reply,
    render_server_backed_batch_candidates,
    replace,
    reviewed,
    reviewed_record,
    reviewed_ticket,
    s54_multiword_chat,
    s54_multiword_commands,
    s54_multiword_review_payload,
    s55_single_pipeline_chat,
    s55_single_pipeline_commands,
    s57_claiming_chat,
    s57_options_chat,
    s57_options_routing,
    s62_incident_incident_record,
    s62_selection_incident_record,
    s63_fresh_selection_chat,
    s68_selection_FIXTURE,
    server_warning_pending_state,
    server_warning_ticket_is_complete,
    shift_plan_warnings,
    single_review,
    snapshot,
    structural_option_questions,
    tempfile,
    ticket,
    trusted_pending_word_items,
    understand_message,
    unicodedata,
    unittest,
    validate_operation,
)


# Source: test_b12_select.py
class B12SelectionTests(unittest.IsolatedAsyncioTestCase):
    setUp = B12SelectionTests_setUp

    turn = B12SelectionTests_turn

    candidates = B12SelectionTests_candidates

    assert_offline_route = B12SelectionTests_assert_offline_route

    created = B12SelectionTests_created

    def test_r1_digits_are_never_explicit_entry_codes(self):
        for message in ("削刀 3", "削刀 3，加入", "削刀 3，加入并提交", "加入 削刀 3", "给削刀加入编码3"):
            with self.subTest(message=message):
                self.assertIsNone(parse_explicit_entry_code_request(message))

    async def test_r1_single_word_number_selects_empty_candidate(self):
        prompt = await self.candidates("削刀")
        await self.turn("削刀 3，加入并提交", prompt)
        self.assertEqual(self.created(), [("削刀", "xhdzia")], self.replay.last_response)
        self.assertEqual(sum(name == "keytao_submit_batch" for name, _ in self.replay.writes), 1)
        self.assert_offline_route()

    async def test_r1_bare_word_number_selects_without_submitting(self):
        prompt = await self.candidates("削刀")
        await self.turn("削刀 3", prompt)
        self.assertEqual(self.created(), [("削刀", "xhdzia")], self.replay.last_response)
        self.assertFalse(any(name == "keytao_submit_batch" for name, _ in self.replay.writes))
        self.assert_offline_route()

    async def test_r1_multi_word_number_selects_named_candidate(self):
        prompt = await self.candidates("正手刀", "削刀")
        await self.turn("削刀 3，加入", prompt)
        self.assertEqual(self.created(), [("削刀", "xhdzia")], self.replay.last_response)
        self.assert_offline_route()

    async def test_r2_partial_selection_keeps_original_message_binding(self):
        prompt = await self.candidates("正手刀", "反手刀", "点刀", "削刀")
        self.replay.store.bind_origin_message_ids(self.replay.store.get_record(self.replay.author), [701])
        await self.turn("正手刀 3，反手刀 3，加入并提交", "Rewritten original prompt", "701")
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertEqual([scope["word"] for scope in record.state.args["_candidate_scopes"]], ["点刀", "削刀"])
        self.assertIn("701", record.origin_message_ids)
        await self.turn("削刀 3，加入并提交", "Rewritten original prompt", "701")
        self.assertEqual(self.created(), [("正手刀", "qsdzia"), ("反手刀", "fsdzia"), ("削刀", "xhdzia")], self.replay.last_response)
        self.assert_offline_route()

    async def test_r2_submit_risk_confirmation_keeps_original_remainder_binding(self):
        await self.candidates("正手刀", "反手刀", "削刀")
        self.replay.store.bind_origin_message_ids(self.replay.store.get_record(self.replay.author), [701])
        self.replay.submit_extra_items = [{"action": "Create", "word": "旧草稿", "code": "jczg", "type": "Phrase"}]
        with patch.object(self.replay.commands, "_submit_preview_is_authors_own", AsyncMock(return_value=False)):
            await self.turn("正手刀 3，反手刀 3，加入并提交", "Original prompt rewritten", "701")
            record = self.replay.store.get_record(self.replay.author)
            self.assertIsNotNone(record, self.replay.last_response)
            self.assertEqual(record.state.function_name, "keytao_submit_batch")
            self.assertEqual(self.created(), [("正手刀", "qsdzia"), ("反手刀", "fsdzia")])
            self.assertFalse(any(name == "keytao_submit_batch" for name, _ in self.replay.writes))
            await self.turn("确认", self.replay.last_response)
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertEqual([scope["word"] for scope in record.state.args["_candidate_scopes"]], ["削刀"])
        self.assertIn("701", record.origin_message_ids)
        await self.turn("削刀 3，加入", "Original prompt rewritten", "701")
        self.assertEqual(self.created(), [("正手刀", "qsdzia"), ("反手刀", "fsdzia"), ("削刀", "xhdzia")])
        self.assert_offline_route()

    async def test_r2_failed_remainder_persistence_keeps_written_ticket_locked(self):
        prompt = await self.candidates("正手刀", "反手刀", "削刀")
        record = self.replay.store.get_record(self.replay.author)
        with patch.object(self.replay.store, "_persist_record", return_value=False):
            await self.turn("正手刀 3，反手刀 3，加入", prompt)
        self.assertEqual(self.created(), [("正手刀", "qsdzia"), ("反手刀", "fsdzia")])
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertTrue(record.execution_id)
        writes = copy.deepcopy(self.replay.writes)
        await self.turn("正手刀 3，反手刀 3，加入", prompt)
        self.assertEqual(self.replay.writes, writes)
        self.assertTrue(record.execution_id)

    async def test_r2_failed_submit_remainder_persistence_locks_replacement_ticket(self):
        await self.candidates("正手刀", "反手刀", "削刀")
        original = self.replay.store.get_record(self.replay.author)
        self.replay.store.bind_origin_message_ids(original, [701])
        self.replay.submit_extra_items = [{"action": "Create", "word": "旧草稿", "code": "jczg", "type": "Phrase"}]
        with patch.object(self.replay.commands, "_submit_preview_is_authors_own", AsyncMock(return_value=False)), patch.object(
            self.replay.store, "put_back", return_value=False,
        ):
            await self.turn("正手刀 3，反手刀 3，加入并提交", "Original prompt rewritten", "701")
        current = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(current, self.replay.last_response)
        self.assertIsNot(current, original)
        self.assertEqual(current.state.function_name, "keytao_submit_batch")
        self.assertTrue(current.execution_id)
        writes = copy.deepcopy(self.replay.writes)
        await self.turn("确认", self.replay.last_response)
        self.assertEqual(self.replay.writes, writes)
        self.assertIs(self.replay.store.get_record(self.replay.author), current)
        self.assertTrue(current.execution_id)

    async def test_r2_cancel_submit_risk_cancels_remainder(self):
        await self.candidates("正手刀", "反手刀", "削刀")
        self.replay.store.bind_origin_message_ids(self.replay.store.get_record(self.replay.author), [701])
        self.replay.submit_extra_items = [{"action": "Create", "word": "旧草稿", "code": "jczg", "type": "Phrase"}]
        with patch.object(self.replay.commands, "_submit_preview_is_authors_own", AsyncMock(return_value=False)):
            await self.turn("正手刀 3，反手刀 3，加入并提交", "Original prompt rewritten", "701")
        self.assertEqual(self.replay.store.get_record(self.replay.author).state.function_name, "keytao_submit_batch")
        writes = copy.deepcopy(self.replay.writes)
        await self.turn("取消", self.replay.last_response)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        await self.turn("削刀 3", "Original prompt rewritten", "701")
        self.assertEqual(self.replay.writes, writes)
        self.assertIsInstance(self.replay.store.get_record(self.replay.author).state, harness.PendingAddWord)
        self.assert_offline_route()

    async def test_r2_cancel_cancels_remainder(self):
        prompt = await self.candidates("正手刀", "反手刀", "削刀")
        await self.turn("正手刀 3，反手刀 3，加入", prompt)
        self.assertIsNotNone(self.replay.store.get_record(self.replay.author))
        count = len(self.replay.writes)
        await self.turn("取消", prompt)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.assertEqual(len(self.replay.writes), count)
        self.assert_offline_route()

    async def test_r2_number_without_ticket_refreshes_word_candidates(self):
        await self.turn("削刀 3，加入并提交")
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertIsInstance(record.state, harness.PendingAddWord)
        self.assertEqual(record.state.word, "削刀")
        self.assertIn("xhdzia", self.replay.last_response)
        self.assertEqual(self.replay.writes, [])
        self.assert_offline_route()

    async def test_r2_stale_quote_number_refreshes_without_writing(self):
        await self.turn("削刀 3", "Old candidate prompt listing 削刀 xhdz、xhdzi、xhdzia", "701")
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertIsInstance(record.state, harness.PendingAddWord)
        self.assertEqual(record.state.word, "削刀")
        self.assertEqual(record.state.server_candidates, [("xhdz", True), ("xhdzi", True), ("xhdzia", False)])
        self.assertTrue(any(name == "keytao_prepare_reviewed_add" and args["word"] == "削刀" for name, args in self.replay.calls))
        self.assertEqual(self.replay.writes, [])
        self.assert_offline_route()

    async def test_r3_explicit_code_grammar_uses_normal_add_path(self):
        for message in ("“削刀”加入编码xhdzia", "削刀 加入编码 xhdzia", "给削刀加入编码xhdzia"):
            with self.subTest(message=message):
                self.replay.writes.clear()
                self.replay.draft.clear()
                self.replay.history.clear()
                self.replay.store.delete(self.replay.author)
                await self.turn(message)
                self.assertEqual(self.created(), [("削刀", "xhdzia")], self.replay.last_response)
                self.assert_offline_route()

    async def test_r3_explicit_code_occupied_uses_existing_ticket(self):
        await self.turn("“荡刀”加入编码dpdzia")
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertIsInstance(record.state, harness.PendingAddWord)
        self.assertEqual(record.state.word, "荡刀")
        self.assertEqual(self.replay.writes, [])
        self.assertIn("挡刀", self.replay.last_response)
        self.assert_offline_route()

    async def test_r3_operand_binding_consumes_complete_grammar(self):
        for message in ("他说“削刀”加入编码xhdzia", "不要给削刀加入编码xhdzia", "给削刀加入编码xhdzia好吗", "给削刀加入编码xhdzia然后删除挡刀"):
            with self.subTest(message=message):
                self.assertIsNone(parse_explicit_entry_code_request(message))

    def test_r3_explicit_literals_are_bound_by_grammar_not_word_denylists(self):
        for word in ("怎么", "我送给", "要领", "可能性", "哪吒", "哪里"):
            for message in (f"“{word}”加入编码xhdzia", f"{word} 加入编码 xhdzia", f"给{word}加入编码xhdzia"):
                with self.subTest(message=message):
                    request = parse_explicit_entry_code_request(message)
                    self.assertIsNotNone(request)
                    self.assertEqual((request.word, request.code), (word, "xhdzia"))

    async def test_r4_occupied_ticket_controls_keep_concise_instruction(self):
        for answer in ("加入", "加入并提交", "加入编码dpdzia"):
            with self.subTest(answer=answer):
                self.replay.store.delete(self.replay.author)
                await self.turn("加词：荡刀 dpdzia")
                record = self.replay.store.get_record(self.replay.author)
                self.assertIsInstance(record.state, harness.PendingAddWord)
                prompt = self.replay.last_response
                self.replay.client.create.reset_mock()
                await self.turn(answer, prompt)
                self.assertEqual(self.replay.last_response, "dpdzia 已有「挡刀」；回复「加入，顶替 挡刀」顶替，或换一个编码。")
                self.assertEqual(self.replay.store.get_record(self.replay.author).state, record.state)
                self.assertEqual(self.replay.writes, [])
                self.assert_offline_route()

    async def test_r5_receipt_names_selected_occupied_candidate(self):
        prompt = await self.candidates("打泡", "荡刀")
        await self.turn("荡刀 3，打泡 2，加入并提交", prompt)
        self.assertEqual(self.created(), [("打泡", "dspzi")], self.replay.last_response)
        self.assertIn("荡刀 3（dpdzia）已有「挡刀」，未加入。", self.replay.last_response.splitlines())
        self.assert_offline_route()

    async def test_r6_carried_occupied_word_renders_once_inside_count(self):
        await self.turn("加词：荡刀 dpdzia")
        await self.candidates("正手刀", "反手刀", "点刀", "削刀")
        response = self.replay.last_response
        self.assertEqual(response.count("「荡刀」候选编码："), 1, response)
        items = re.findall(r"(?m)^\d+[.、．)]\s*「", response)
        count = re.search(r"(\d+) 个词的候选", response)
        self.assertIsNotNone(count, response)
        self.assertEqual(int(count.group(1)), len(items), response)
        for word in ("点刀", "荡刀"):
            self.assertRegex(response, rf"(?m)^\d+[.、．)]\s*「{word}」")
        self.assert_offline_route()

    async def test_r6_empty_recommendation_keeps_occupied_words_and_remainder(self):
        for word in ("荡刀", "点刀"):
            self.replay.candidate_facts[word]["recommended"] = ""
        await self.turn("加词：荡刀 dpdzia")
        await self.candidates("正手刀", "反手刀", "点刀", "削刀")
        response = self.replay.last_response
        items = re.findall(r"(?m)^\d+[.、．)]\s*「([^」]+)」", response)
        self.assertEqual(set(items), {"荡刀", "正手刀", "反手刀", "点刀", "削刀"}, response)
        self.assertIn(f"{len(items)} 个词的候选", response)
        for word in ("荡刀", "点刀"):
            self.assertEqual(response.count(f"「{word}」候选编码："), 1, response)
        record = self.replay.store.get_record(self.replay.author)
        self.replay.store.bind_origin_message_ids(record, [702])
        await self.turn("正手刀 3，反手刀 3，加入并提交", "Original candidate prompt rewritten", "702")
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertEqual({scope["word"] for scope in record.state.args["_candidate_scopes"]}, {"荡刀", "点刀", "削刀"})
        self.assertIn("702", record.origin_message_ids)
        self.assertEqual(self.created(), [("正手刀", "qsdzia"), ("反手刀", "fsdzia")])
        self.assert_offline_route()

    async def test_r6_no_recommendation_never_authorizes_bare_batch_add(self):
        self.replay.candidate_facts["点刀"]["recommended"] = ""
        for action in ("加入", "加入并提交"):
            with self.subTest(action=action):
                self.replay.store.delete(self.replay.author)
                self.replay.writes.clear()
                self.replay.draft.clear()
                self.replay.history.clear()
                prompt = await self.candidates("正手刀", "点刀")
                self.assertIn("推荐编码：暂无（现有候选均需点名顶替）", prompt)
                self.assertNotIn("单独发", prompt)
                state = self.replay.store.get_record(self.replay.author).state
                self.assertEqual(state.args["_candidate_scopes"][1]["reviewedState"]["recommendedCode"], "")
                self.assertEqual([item["word"] for item in state.args["items"]], ["正手刀"])
                await self.turn(action, prompt)
                self.assertEqual(self.created(), [("正手刀", "qsdzi")], self.replay.last_response)
                writes = copy.deepcopy(self.replay.writes)
                await self.turn(action, prompt)
                self.assertEqual(self.replay.writes, writes)
                self.assert_offline_route()

    async def test_r6_no_recommendation_stays_non_actionable_in_remainder(self):
        self.replay.candidate_facts["点刀"]["recommended"] = ""
        for action in ("加入", "加入并提交"):
            with self.subTest(action=action):
                self.replay.store.delete(self.replay.author)
                self.replay.writes.clear()
                self.replay.draft.clear()
                self.replay.history.clear()
                prompt = await self.candidates("正手刀", "点刀", "削刀")
                await self.turn("正手刀 3，加入", prompt)
                self.assertEqual(self.created(), [("正手刀", "qsdzia")], self.replay.last_response)
                await self.turn(action, prompt)
                self.assertEqual(self.created(), [("正手刀", "qsdzia"), ("削刀", "xhdzia")], self.replay.last_response)
                self.assert_offline_route()

    async def test_r6_only_unbound_occupied_candidates_cannot_write(self):
        for word in ("点刀", "荡刀"):
            self.replay.candidate_facts[word]["recommended"] = ""
        prompt = await self.candidates("点刀", "荡刀")
        for action in ("加入", "加入并提交", "点刀 1，加入"):
            with self.subTest(action=action):
                await self.turn(action, prompt)
                self.assertEqual(self.replay.writes, [], self.replay.last_response)
                self.assertIsNotNone(self.replay.store.get_record(self.replay.author))
                self.general.assert_not_awaited()
        await self.turn("取消", prompt)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.assertEqual(self.replay.writes, [])

    async def test_r6_merging_keeps_unbound_occupied_candidates_without_write_items(self):
        self.replay.candidate_facts["点刀"]["recommended"] = ""
        await self.candidates("正手刀", "点刀")
        prompt = await self.candidates("削刀")
        self.assertRegex(prompt, r"(?m)^\d+[.、．)]\s*「点刀」")
        state = self.replay.store.get_record(self.replay.author).state
        self.assertEqual({scope["word"] for scope in state.args["_candidate_scopes"]}, {"正手刀", "点刀", "削刀"})
        self.assertEqual([item["word"] for item in state.args["items"]], ["正手刀", "削刀"])
        self.assertEqual(self.replay.writes, [])
        self.assert_offline_route()

    async def test_r7_direct_shift_ignores_only_own_source_slot_warning(self):
        replay = ShiftReplay()
        warning = {"warningType": "skipped_candidate_slot", "item": {"word": b11_shift_WORD, "code": "xlblu"},
                   "skippedCodes": ["xlbl"], "message": '编码链中更短候选 "xlbl" 仍是空位'}

        async def tool(name, args, *actor, **kwargs):
            if name != "keytao_shift_phrase_code":
                return await ShiftReplay.tool(replay, name, args, *actor, **kwargs)
            replay.calls.append((name, copy.deepcopy(args)))
            if args.get("confirmed_plan_digest"):
                replay.writes.append((name, copy.deepcopy(args)))
                return json.dumps(replay.result(shiftPlan={"shifted": [], "items": []}))
            return json.dumps(replay.result(
                success=False, requiresConfirmation=True, confirmationKind="shiftPlan",
                planDigest="b" * 64, warningDigest="c" * 64, warnings=[warning],
                shiftPlan={"word": b11_shift_WORD, "sourceCode": "xlbl", "targetCode": "xlblu",
                           "sourcePhrase": {"word": b11_shift_WORD, "code": "xlbl"}, "shifted": [],
                           "items": [{"action": "Delete", "word": b11_shift_WORD, "code": "xlbl"},
                                     {"action": "Create", "word": b11_shift_WORD, "code": "xlblu"}]},
            ))

        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(replay.chat, "call_tool_function", AsyncMock(side_effect=tool)))
            await replay.turn({"message": f"顺延“{b11_shift_WORD}”", "to_me": True}, raw_operation())
        self.assertEqual(len(replay.writes), 1, replay.last_response)
        self.assertEqual(replay.writes[0][1]["target_code"], "xlblu")
        self.assertNotIn('"xlbl" 仍是空位', replay.last_response)
        replay.client.create.assert_not_awaited()

    def test_r7_shift_ticket_suppresses_source_but_keeps_real_skipped_slot(self):
        source = {"warningType": "skipped_candidate_slot", "item": {"word": b11_shift_WORD, "code": "xlbluv"},
                  "skippedCodes": ["xlbl"], "message": '编码链中更短候选 "xlbl" 仍是空位'}
        genuine = {**source, "skippedCodes": ["xlblu"], "message": '编码链中更短候选 "xlblu" 仍是空位'}
        data = self.replay.result(
            success=False, requiresConfirmation=True, confirmationKind="shiftPlan",
            planDigest="b" * 64, warningDigest="c" * 64, warnings=[source, genuine],
            shiftPlan={"word": b11_shift_WORD, "sourceCode": "xlbl", "targetCode": "xlbluv",
                       "sourcePhrase": {"word": b11_shift_WORD, "code": "xlbl"}, "shifted": [],
                       "items": [{"action": "Delete", "word": b11_shift_WORD, "code": "xlbl"},
                                 {"action": "Create", "word": b11_shift_WORD, "code": "xlbluv"}]},
        )
        ticket = self.replay.commands._pending_state_from_server_warning(harness.PendingToolConfirm(
            "keytao_shift_phrase_code", {"word": b11_shift_WORD, "target_code": "xlbluv", "_source_code": "xlbl"},
        ), data)
        original = copy.deepcopy(ticket)
        response = self.replay.commands.render_pending_shift_plan(ticket)
        self.assertTrue(response)
        self.assertNotIn('"xlbl" 仍是空位', response)
        self.assertIn("⚠️ 更短的 xlblu 还空着，跳过会留空位。", response)
        self.assertEqual(ticket, original)

    def test_r7_mixed_source_and_empty_slots_preserve_sealed_data(self):
        warning = {"warningType": "skipped_candidate_slot", "item": {"word": b11_shift_WORD, "code": "xlbluv"},
                   "skippedCode": "xlbl", "skippedCodes": ["xlbl", "xlblu"],
                   "message": '编码链中更短候选 "xlbl"、"xlblu" 仍是空位'}
        data = self.replay.result(
            success=False, requiresConfirmation=True, confirmationKind="shiftPlan",
            planDigest="b" * 64, warningDigest="c" * 64, warnings=[warning],
            shiftPlan={"word": b11_shift_WORD, "sourceCode": "xlbl", "targetCode": "xlbluv",
                       "sourcePhrase": {"word": b11_shift_WORD, "code": "xlbl"}, "shifted": [],
                       "items": [{"action": "Delete", "word": b11_shift_WORD, "code": "xlbl"},
                                 {"action": "Create", "word": b11_shift_WORD, "code": "xlbluv"}]},
        )
        original = copy.deepcopy(data)
        warnings = shift_plan_warnings(data)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]["skippedCodes"], ["xlblu"])
        self.assertEqual(warnings[0]["skippedCode"], "xlblu")
        ticket = self.replay.commands._pending_state_from_server_warning(harness.PendingToolConfirm(
            "keytao_shift_phrase_code", {"word": b11_shift_WORD, "target_code": "xlbluv", "_source_code": "xlbl"},
        ), data)
        response = self.replay.commands.render_pending_shift_plan(ticket)
        self.assertIn("⚠️ 更短的 xlblu 还空着，跳过会留空位。", response)
        self.assertNotIn('"xlbl"', response)
        self.assertEqual(ticket.args["confirmed_plan_digest"], data["planDigest"])
        self.assertEqual(ticket.args["expected_warning_digest"], data["warningDigest"])
        self.assertEqual(data, original)

# Source: test_b14_multi_select.py
class B14MultiSelectionTests(unittest.IsolatedAsyncioTestCase):
    setUp = B14MultiSelectionTests_setUp

    candidates = B14MultiSelectionTests_candidates

    select = B14MultiSelectionTests_select

    assert_offline_selection = B14MultiSelectionTests_assert_offline_selection

    created = B14MultiSelectionTests_created

    async def test_r1_exact_t3_selects_seven_bound_creates(self):
        prompt = await self.candidates(*b14_multi_select_FACTS)
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsInstance(record.state, harness.PendingToolConfirm)
        self.assertEqual(len(record.state.args["_candidate_scopes"]), 6)
        self.assertIn("飞键", prompt)
        await self.select(T3, prompt)
        self.assertEqual(self.created(), [
            ("许褚", "xljj"), ("胡桃木", "htm"), ("灵能", "lgnr"),
            ("债市", "qheki"), ("债市", "fhek"), ("琴酒", "qbjq"), ("数仓", "ejcp"),
        ], self.replay.last_response)
        self.assertFalse(any(name == "keytao_submit_batch" for name, _ in self.replay.writes))
        self.assert_offline_selection()

    async def test_r1_single_and_multiword_multiselection_forms(self):
        for words in (("债市",), tuple(b14_multi_select_FACTS)):
            for message in ("债市2和4", "债市 2、4", "债市 2,4", "债市 2 4", "“债市qheki、fhek”", "「债市2和4」"):
                with self.subTest(words=words, message=message):
                    prompt = await self.candidates(*words)
                    await self.select(message, prompt)
                    self.assertEqual(self.created(), [("债市", "qheki"), ("债市", "fhek")], self.replay.last_response)
                    self.assertFalse(any(name == "keytao_submit_batch" for name, _ in self.replay.writes))
                    self.assert_offline_selection()

    async def test_r1_single_no_space_number_and_code(self):
        for words in (("债市",), tuple(b14_multi_select_FACTS)):
            for message in ("债市2", "债市qheki", "“债市2”"):
                with self.subTest(words=words, message=message):
                    prompt = await self.candidates(*words)
                    await self.select(message, prompt)
                    self.assertEqual(self.created(), [("债市", "qheki")], self.replay.last_response)
                    self.assert_offline_selection()

    async def test_r1_multiselection_keeps_add_and_submit_verbs(self):
        for suffix, submissions in (("，加入", 0), ("，加入并提交", 1)):
            with self.subTest(suffix=suffix):
                prompt = await self.candidates(*b14_multi_select_FACTS)
                await self.select("债市2、4" + suffix, prompt)
                self.assertEqual(self.created(), [("债市", "qheki"), ("债市", "fhek")], self.replay.last_response)
                self.assertEqual(sum(name == "keytao_submit_batch" for name, _ in self.replay.writes), submissions)
                self.assert_offline_selection()

    async def test_r1_unbound_tokens_reject_the_whole_message(self):
        for message in (
            "债市 2，未知词 4", "债市2和7", "债市2和0", "债市2和2",
            "债市qheki，许褚qheki", "债市 2 garbage", "债市2，胡桃木1，孤词",
            "债市2 然后", "债市2，未知词qheki", "债市2qheki",
            "不要债市2和4", "他说债市2和4", "债市2，债市2",
        ):
            with self.subTest(message=message):
                prompt = await self.candidates(*b14_multi_select_FACTS)
                record = self.replay.store.get_record(self.replay.author)
                before = copy.deepcopy(record)
                await self.select(message, prompt)
                self.assertEqual(self.replay.writes, [], self.replay.last_response)
                self.assertTrue(
                    self.replay.last_response == b14_multi_select_SELECTION_ERROR
                    or "看到了你的选词请求" in self.replay.last_response,
                    self.replay.last_response,
                )
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
                self.assertEqual(record, before)
                self.general.assert_not_awaited()

    async def test_r1_single_word_unbound_tokens_reject_the_whole_message(self):
        for message in (
            "债市2和7", "债市2和0", "债市2和2", "债市 2 garbage", "债市2 然后",
            "债市2，未知词qheki", "债市2qheki", "不要债市2和4", "他说债市2和4", "债市2，债市2",
        ):
            with self.subTest(message=message):
                prompt = await self.candidates("债市")
                record = self.replay.store.get_record(self.replay.author)
                before = copy.deepcopy(record)
                await self.select(message, prompt)
                self.assertEqual(self.replay.writes, [], self.replay.last_response)
                self.assertEqual(self.replay.last_response, b14_multi_select_SELECTION_ERROR)
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
                self.assertEqual(record, before)
                self.general.assert_not_awaited()

    async def test_r2_exact_t2_answer_keeps_live_ticket_and_bindings(self):
        prompt = await self.candidates(*b14_multi_select_FACTS)
        record = self.replay.store.get_record(self.replay.author)
        before = copy.deepcopy(record)
        await self.replay.turn({"message": b14_multi_select_T2}, raw_operation(
            op="query", frame="question", operands=["债市"], codes=["qheki", "fhek"], refers_to_pending=True,
        ))
        self.assertEqual(self.replay.last_response, ANSWER)
        self.assertNotEqual(self.replay.last_response, prompt)
        self.general.assert_awaited_once()
        self.assertEqual(self.replay.writes, [])
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertEqual(record, before)
        self.assertIn("1401", record.origin_message_ids)
        self.assertTrue(record.origin_prompt_digest)

    async def test_unpunctuated_code_queries_keep_live_ticket_and_reach_general_agent(self):
        for message, operands, codes in (
            ("查一下fhek", [], ["fhek"]),
            ("看看qheki", [], ["qheki"]),
            ("确实qhek", ["确实"], ["qhek"]),
            ("查一下fhek 的意思", [], ["fhek"]),
            ("看看 qheki", [], ["qheki"]),
            ("查一下 fhek 的意思", [], ["fhek"]),
            ("查一下债市qheki", ["债市"], ["qheki"]),
            ("看看债市qheki", ["债市"], ["qheki"]),
        ):
            with self.subTest(message=message):
                await self.candidates(*b14_multi_select_FACTS)
                record = self.replay.store.get_record(self.replay.author)
                before = copy.deepcopy(record)
                await self.replay.turn({"message": message}, raw_operation(
                    op="query", frame="none", operands=operands, codes=codes,
                ))
                self.assertEqual(self.replay.last_response, ANSWER)
                self.general.assert_awaited_once()
                self.assertEqual(self.replay.writes, [])
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
                self.assertEqual(record, before)

    async def test_single_and_multiword_unpunctuated_word_code_queries_reach_general_agent(self):
        for words in (("债市",), tuple(b14_multi_select_FACTS)):
            for message in ("查一下债市fhek", "我觉得债市fhek不错", "债市fhek是什么意思"):
                with self.subTest(words=words, message=message):
                    await self.candidates(*words)
                    record = self.replay.store.get_record(self.replay.author)
                    self.assertIsInstance(
                        record.state, harness.PendingAddWord if len(words) == 1 else harness.PendingToolConfirm,
                    )
                    await self.replay.turn({"message": message}, raw_operation(
                        op="query", frame="none", operands=["债市"], codes=["fhek"],
                    ))
                    self.general.assert_awaited_once()
                    self.assertNotEqual(self.replay.last_response, b14_multi_select_SELECTION_ERROR)
                    self.assertEqual(self.replay.writes, [])

    async def test_r2_action_advertisement_uses_short_bound_selection_line(self):
        for message, codes, selection, selected_codes in (
            (b14_multi_select_T2, ["qheki", "fhek"], "2、4", "qheki 和 fhek"),
            ("为什么债市推荐qheki？", ["qheki"], "2、3", "qheki 和 qhekio"),
            ("债市这种飞键要怎么加啊？", [], "2、3", "qheki 和 qhekio"),
        ):
            with self.subTest(message=message):
                prompt = await self.candidates(*b14_multi_select_FACTS)
                record = self.replay.store.get_record(self.replay.author)
                before = copy.deepcopy(record)
                self.general.return_value = "回复「加入」。"
                await self.replay.turn({"message": message}, raw_operation(
                    op="query", frame="question", operands=["债市"], codes=codes, refers_to_pending=True,
                ))
                self.assertEqual(self.replay.last_response, f"「债市」可多选：回复「债市 {selection}」同时加入 {selected_codes}。")
                self.assertNotEqual(self.replay.last_response, prompt)
                self.general.assert_awaited_once()
                self.assertEqual(self.replay.writes, [])
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
                self.assertEqual(record, before)

# Source: test_b15_sibling_and_others.py
class B15ChatTests(unittest.IsolatedAsyncioTestCase):
    setUp = B15ChatTests_setUp

    turn = B15ChatTests_turn

    candidates = B15ChatTests_candidates

    created = B15ChatTests_created

    submissions = B15ChatTests_submissions

    reset_ticket = B15ChatTests_reset_ticket

    async def test_r1_sibling_recommendation_footer_binding_and_bare_add(self):
        for answer, submits in (("加入", 0), ("加入并提交", 1)):
            with self.subTest(answer=answer):
                self.reset_ticket()
                prompt = await self.candidates()
                self.assertIn("1. qzbx — 本次推荐给「球坐标系」", prompt)
                self.assertIn("2. qzbxv — ✅推荐", prompt)
                self.assertNotIn("推荐编码：qzbxv", prompt)
                self.assertIn("或改选如「球坐标系 3」", prompt)
                record = self.replay.store.get_record(self.replay.author)
                self.assertEqual([(item["word"], item["code"]) for item in record.state.args["items"]],
                                 [(b15_sibling_and_others_WORDS[0], "qzbx"), (b15_sibling_and_others_WORDS[1], "qzbxv")])
                await self.turn(answer, prompt)
                self.assertEqual(self.created(), [(b15_sibling_and_others_WORDS[0], "qzbx"), (b15_sibling_and_others_WORDS[1], "qzbxv")], self.replay.last_response)
                self.assertEqual(self.submissions(), submits, self.replay.last_response)
                self.general.assert_not_awaited()
                self.replay.client.create.assert_not_awaited()

    async def test_r1_no_empty_sibling_chain_has_no_recommendation(self):
        self.replay.candidate_facts[b15_sibling_and_others_WORDS[1]]["codes"] = ["qzbx"]
        prompt = await self.candidates()
        record = self.replay.store.get_record(self.replay.author)
        self.assertEqual([(item["word"], item["code"]) for item in record.state.args["items"]], [(b15_sibling_and_others_WORDS[0], "qzbx")])
        self.assertNotIn("「柱坐标系」→ qzbx", prompt)
        await self.turn("加入", prompt)
        self.assertEqual(self.created(), [(b15_sibling_and_others_WORDS[0], "qzbx")], self.replay.last_response)

    async def test_r1_same_code_selection_rejects_all_and_keeps_ticket(self):
        for message in ("球坐标系 1，柱坐标系 1", "球坐标系 qzbx，柱坐标系 qzbx"):
            with self.subTest(message=message):
                self.reset_ticket()
                prompt = await self.candidates()
                record = self.replay.store.get_record(self.replay.author)
                before = copy.deepcopy(record)
                await self.turn(message, prompt)
                self.assertEqual(self.replay.writes, [], self.replay.last_response)
                self.assertEqual(self.replay.last_response, "球坐标系、柱坐标系 选了同一个编码 qzbx，请改选；本次未写入。")
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
                self.assertEqual(record, before)
                self.general.assert_not_awaited()

    async def test_r1_selection_tokens_bind_to_ticket_words_and_fail_closed(self):
        for message in (
            "球坐标系 1，未知词 2", "球坐标系 qzbx，未知词 qzbxv",
            "球坐标系 1，柱坐标系 2，garbage", "球坐标系 1，柱坐标系 2和2",
            "球坐标系 qzbxqzbxv", "球坐标系 0，柱坐标系 2",
            "球坐标系 1，球坐标系 1", "球坐标系 1，柱坐标系 9",
        ):
            with self.subTest(message=message):
                self.reset_ticket()
                prompt = await self.candidates()
                record = self.replay.store.get_record(self.replay.author)
                before = copy.deepcopy(record)
                await self.turn(message, prompt)
                self.assertEqual(self.replay.writes, [], self.replay.last_response)
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
                self.assertEqual(record, before)
                self.general.assert_not_awaited()

    async def test_r1_valid_named_selections_write_only_bound_codes(self):
        for message, second_code in (("球坐标系 1，柱坐标系 2", "qzbxv"),
                                     ("球坐标系 qzbx，柱坐标系 qzbxvi", "qzbxvi")):
            with self.subTest(message=message):
                self.reset_ticket()
                prompt = await self.candidates()
                await self.turn(message, prompt)
                self.assertEqual(self.created(), [(b15_sibling_and_others_WORDS[0], "qzbx"), (b15_sibling_and_others_WORDS[1], second_code)], self.replay.last_response)
                self.assertEqual(self.submissions(), 0)
                self.general.assert_not_awaited()
                self.replay.client.create.assert_not_awaited()

    async def test_r2_other_author_notice_keeps_single_and_multiword_candidates(self):
        item = pending_item()
        self.replay.pending_items = [item]
        for words in ((b15_sibling_and_others_WORDS[0],), b15_sibling_and_others_WORDS):
            for prefix in ("", "加词："):
                with self.subTest(words=words, prefix=prefix):
                    self.reset_ticket()
                    await self.turn(prefix + "，".join(words))
                    prompt = self.replay.last_response
                    self.assertTrue(prompt.startswith(notice(item)), prompt)
                    self.assertIn("1. qzbx — ✅推荐", prompt)
                    self.assertNotIn("若需撤回", prompt)
                    record = self.replay.store.get_record(self.replay.author)
                    self.assertIsNotNone(record)
                    if len(words) == 1:
                        self.assertEqual(record.state.recommended_code, "qzbx")
                    self.assertEqual(self.replay.writes, [])

    async def test_r2_same_pair_requires_duplicate_confirmation(self):
        self.replay.pending_items = [pending_item(code="qzbx")]
        for words in ((b15_sibling_and_others_WORDS[0],), b15_sibling_and_others_WORDS):
            for answer, submits in (("加入", 0), ("加入并提交", 1)):
                with self.subTest(words=words, answer=answer):
                    self.reset_ticket()
                    prompt = await self.candidates(words)
                    self.assertIn(notice(self.replay.pending_items[0]), prompt)
                    await self.turn(answer, prompt)
                    self.assertEqual(self.created(), [])
                    self.assertIn("该词已在审核中，确认再提交一条相同词条吗？", self.replay.last_response)
                    state = self.replay.store.get_record(self.replay.author).state
                    self.assertTrue(pending_duplicate_confirmation_is_complete(state))
                    await self.turn("确认", self.replay.last_response)
                    expected = [(b15_sibling_and_others_WORDS[0], "qzbx")]
                    if len(words) == 2:
                        expected.append((b15_sibling_and_others_WORDS[1], "qzbxv"))
                    self.assertEqual(self.created(), expected, self.replay.last_response)
                    self.assertEqual(self.submissions(), submits, self.replay.last_response)
                    self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r2_unseen_overlap_submit_prompts_then_confirm_submits(self):
        self.replay.draft = [{"action": "Create", "word": b15_sibling_and_others_WORDS[0], "code": "qzbx", "type": "Phrase"}]
        self.replay.pending_items = [pending_item()]
        await self.turn("提交")
        self.assertEqual(self.submissions(), 0, self.replay.last_response)
        self.assertIn(notice(self.replay.pending_items[0]), self.replay.last_response)
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record)
        self.assertEqual(record.state.function_name, "keytao_submit_batch")
        prompt = self.replay.last_response
        await self.turn("确认", prompt)
        self.assertEqual(self.submissions(), 1, self.replay.last_response)

    async def test_r2_unseen_overlap_cancel_never_submits(self):
        self.replay.draft = [{"action": "Create", "word": b15_sibling_and_others_WORDS[0], "code": "qzbx", "type": "Phrase"}]
        self.replay.pending_items = [pending_item()]
        await self.turn("提交")
        self.assertEqual(self.submissions(), 0, self.replay.last_response)
        await self.turn("取消", self.replay.last_response)
        self.assertEqual(self.submissions(), 0)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r2_shown_overlap_add_and_submit_has_no_second_prompt(self):
        self.replay.pending_items = [pending_item()]
        for words in ((b15_sibling_and_others_WORDS[0],), b15_sibling_and_others_WORDS):
            with self.subTest(words=words):
                self.reset_ticket()
                prompt = await self.candidates(words)
                self.assertIn(notice(self.replay.pending_items[0]), prompt)
                await self.turn("加入并提交", prompt)
                self.assertEqual(self.submissions(), 1, self.replay.last_response)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r2_new_overlap_after_candidate_ticket_requires_submit_confirmation(self):
        prompt = await self.candidates()
        self.replay.pending_items = [pending_item()]
        await self.turn("加入并提交", prompt)
        self.assertEqual(self.created(), [(b15_sibling_and_others_WORDS[0], "qzbx"), (b15_sibling_and_others_WORDS[1], "qzbxv")], self.replay.last_response)
        self.assertEqual(self.submissions(), 0, self.replay.last_response)
        self.assertIn(notice(self.replay.pending_items[0]), self.replay.last_response)
        self.assertEqual(self.replay.store.get_record(self.replay.author).state.function_name, "keytao_submit_batch")
        await self.turn("确认", self.replay.last_response)
        self.assertEqual(self.submissions(), 1, self.replay.last_response)

    async def test_r2_submit_lookup_failure_does_not_block_submission(self):
        self.replay.pending_complete = False
        self.replay.draft = [{"action": "Create", "word": b15_sibling_and_others_WORDS[0], "code": "qzbx", "type": "Phrase"}]
        await self.turn("提交")
        self.assertEqual(self.submissions(), 1, self.replay.last_response)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

# Source: test_b15_sibling_and_others.py
class B15PendingLookupTests(unittest.IsolatedAsyncioTestCase):
    setUp = B15PendingLookupTests_setUp

    batch = B15PendingLookupTests_batch

    lookup = B15PendingLookupTests_lookup

    async def test_r2_lookup_discovers_others_all_actions_exact_word_and_old_word(self):
        self.batch(OWN_BATCH, 42, [pending_item(source="submitted", code="qzbx")])
        self.batch(OTHER_BATCH, 7, [
            pending_item(code="qzbx"), pending_item(word="新词", oldWord=b15_sibling_and_others_WORDS[0], action="Change"),
            pending_item(word=b15_sibling_and_others_WORDS[1], action="Delete", code="qzbxv"),
            pending_item(word="长球坐标系", code="qzbxvi"), pending_item(itemStatus="Approved", status="Approved"),
        ])
        for item in self.details[OWN_BATCH]["pullRequests"] + self.details[OTHER_BATCH]["pullRequests"]:
            item.setdefault("status", "Pending")
        result = await self.lookup()
        self.assertTrue(result["complete"], result)
        own = [item for item in result["items"] if item["source"] == "submitted"]
        others = [item for item in result["items"] if item["source"] == "others_submitted"]
        self.assertEqual([(item["word"], item["code"]) for item in own], [(b15_sibling_and_others_WORDS[0], "qzbx")])
        self.assertEqual({item["action"] for item in others}, {"Create", "Change", "Delete"})
        self.assertEqual(len(others), 3, result)
        self.assertEqual(trusted_pending_word_items(others), others)
        result = await self.lookup(("坐标系",))
        self.assertEqual(result["items"], [], result)

    async def test_r2_other_detail_failure_is_logged_advisory_and_complete(self):
        self.batch(OTHER_BATCH, 7, [])
        self.details[OTHER_BATCH] = self.tools.httpx.TransportError("offline detail unavailable")
        with patch.object(self.tools.logger, "warning") as warning:
            result = await self.lookup()
        self.assertEqual(self.detail_calls, [OTHER_BATCH])
        self.assertTrue(result["success"], result)
        self.assertTrue(result["complete"], result)
        self.assertEqual(result["items"], [])
        warning.assert_called()

    async def test_r2_other_detail_budget_is_ten_and_sequential_own_is_preserved(self):
        for number in range(12):
            self.batch(f"b1500000-0000-4000-8000-{number:012d}", 7, [])
        self.batch(OWN_BATCH, 42, [])
        result = await self.lookup()
        self.assertTrue(result["complete"], result)
        self.assertEqual(len([batch for batch in self.detail_calls if batch != OWN_BATCH]), 10)
        self.assertEqual(self.detail_calls.count(OWN_BATCH), 1)
        self.assertEqual(self.maximum_active_details, 1)

    async def test_r2_own_detail_failure_keeps_incomplete_behavior(self):
        self.batch(OWN_BATCH, 42, [])
        self.details[OWN_BATCH] = self.tools.httpx.TransportError("offline own detail unavailable")
        result = await self.lookup()
        self.assertFalse(result["success"], result)
        self.assertFalse(result["complete"], result)

    async def test_r2_submit_tool_preserves_http_sequence_and_single_use_audit_ticket(self):
        snapshot = [{"action": "Create", "word": b15_sibling_and_others_WORDS[0], "code": "qzbx", "type": "Phrase"}]
        audit = {
            "success": True, "verdict": "manual_review", "autoApprove": False,
            "summary": "Offline audit", "issues": [], "batchId": OWN_BATCH,
            "contentVersion": 7, "snapshotDigest": "c" * 64, "snapshotItems": snapshot,
        }
        self.submit_preview = {
            "success": False, "requiresConfirmation": True, "batchId": OWN_BATCH,
            "contentVersion": 7, "snapshotDigest": "a" * 64,
            "warningDigest": "b" * 64, "warnings": [], "snapshotItems": snapshot,
        }
        submit_url = f"https://offline.invalid/api/bot/batches/{OWN_BATCH}/submit"
        with (
            patch.object(self.tools, "_audit_current_draft_for_auto_approval", AsyncMock(return_value=audit)) as audit_call,
            patch.object(self.tools, "_SUBMIT_AUDIT_TICKETS", self.tools._SubmitAuditTicketStore()),
        ):
            preview = await self.tools.keytao_submit_batch("qq", "b15-owner", batch_id=OWN_BATCH, preview_only=True)
            self.assertEqual(self.http_posts, [submit_url])
            self.assertNotIn("pendingItems", preview)
            self.submit_status = 200
            self.submit_preview = {"success": True, "contentVersion": 8}
            arguments = {
                "confirmed": True, "batch_id": OWN_BATCH,
                "expected_content_version": 7,
                "expected_server_snapshot_digest": preview["snapshotDigest"],
                "expected_warning_digest": preview["warningDigest"],
                "expected_audit_digest": preview["auditDigest"],
            }
            confirmed = await self.tools.keytao_submit_batch("qq", "b15-owner", **arguments)
            replayed = await self.tools.keytao_submit_batch("qq", "b15-owner", **arguments)
        self.assertTrue(confirmed["success"], confirmed)
        self.assertTrue(replayed["staleConfirmation"], replayed)
        self.assertEqual(self.http_posts, [submit_url, submit_url])
        audit_call.assert_awaited_once()

    async def test_r2_submit_preview_caller_enriches_actual_lookup_and_keeps_failures_advisory(self):
        snapshot = [{"action": "Create", "word": b15_sibling_and_others_WORDS[0], "code": "qzbx", "type": "Phrase"}]
        audit = {
            "success": True, "verdict": "manual_review", "autoApprove": False,
            "summary": "Offline audit", "issues": [], "batchId": OWN_BATCH,
            "contentVersion": 7, "snapshotDigest": "c" * 64, "snapshotItems": snapshot,
        }
        self.submit_preview = {
            "success": False, "requiresConfirmation": True, "batchId": OWN_BATCH,
            "contentVersion": 7, "snapshotDigest": "a" * 64,
            "warningDigest": "b" * 64, "warnings": [], "snapshotItems": snapshot,
        }
        for status in (200, 400):
            for failure in ("none", "other", "own", "lookup", "malformed"):
                with self.subTest(status=status, failure=failure):
                    self.batches.clear()
                    self.details.clear()
                    self.detail_calls.clear()
                    self.submit_posts.clear()
                    self.submit_status = status
                    self.batch(OTHER_BATCH, 7, [{**pending_item(), "status": "Pending"}])
                    if failure == "other":
                        self.details[OTHER_BATCH] = self.tools.httpx.TransportError("offline other detail unavailable")
                    elif failure == "own":
                        self.batch(OWN_BATCH, 42, [])
                        self.details[OWN_BATCH] = self.tools.httpx.TransportError("offline own detail unavailable")

                    async def actual_tool(name, args, platform, user_id):
                        if name == "keytao_pending_items_by_words":
                            if failure == "lookup":
                                raise self.tools.httpx.TransportError("offline lookup unavailable")
                            if failure == "malformed":
                                return "invalid json"
                        return json.dumps(await getattr(self.tools, name)(platform, user_id, **args))

                    async def actual_read(name, args, context):
                        return await actual_tool(name, args, context.platform, context.user_id)

                    with (
                        patch.object(self.tools, "_audit_current_draft_for_auto_approval", AsyncMock(return_value=audit)),
                        patch.object(self.tools, "_SUBMIT_AUDIT_TICKETS", self.tools._SubmitAuditTicketStore()),
                        patch.object(harness.chat_commands_module, "call_tool_function", AsyncMock(side_effect=actual_tool)) as call,
                        patch.object(harness.chat_commands_module.tool_executor, "call", AsyncMock(side_effect=actual_read)) as read,
                    ):
                        result = await harness.chat_commands_module._perform_submit_current_draft(
                            "qq", "b15-owner", batch_id=OWN_BATCH,
                        )
                    self.assertEqual([entry.args[0] for entry in call.await_args_list],
                                     ["keytao_submit_batch"])
                    self.assertEqual([entry.args[0] for entry in read.await_args_list],
                                     ["keytao_pending_items_by_words"])
                    self.assertEqual(read.await_args.args[1], {"words": [b15_sibling_and_others_WORDS[0]]})
                    self.assertEqual((read.await_args.args[2].platform, read.await_args.args[2].user_id),
                                     ("qq", "b15-owner"))
                    self.assertIsNotNone(result.pending_state, result.text)
                    preview = result.data
                    self.assertTrue(preview["requiresConfirmation"], preview)
                    self.assertNotIn("uncertain", preview)
                    self.assertEqual(preview["snapshotDigest"], "a" * 64)
                    self.assertEqual(preview["warningDigest"], "b" * 64)
                    self.assertEqual(preview["snapshotItems"], snapshot)
                    self.assertEqual(len(self.submit_posts), 1)
                    self.assertTrue(self.submit_posts[0][1]["previewOnly"])
                    self.assertFalse(self.submit_posts[0][1]["confirmed"])
                    if failure == "none":
                        self.assertIn(notice(preview["pendingItems"][0]), result.text)
                        self.assertEqual(self.detail_calls, [OTHER_BATCH])
                        self.assertEqual([(item["source"], item["word"]) for item in preview["pendingItems"]],
                                         [("others_submitted", b15_sibling_and_others_WORDS[0])])
                    else:
                        self.assertEqual(preview.get("pendingItems", []), [])

    async def test_r2_duplicate_guard_and_ticket_accept_other_author_create_only(self):
        item = pending_item(code="qzbx")
        for code, action, duplicate in (("qzbx", "Create", True), ("qzbxv", "Create", False), ("qzbx", "Change", False)):
            with self.subTest(code=code, action=action):
                with patch.object(self.tools, "keytao_pending_items_by_words", AsyncMock(return_value={
                    "success": True, "complete": True, "items": [{**item, "action": action}],
                })):
                    guard, _ = await self.tools._pending_create_guard("qq", "b15-owner", [{
                        "action": "Create", "word": b15_sibling_and_others_WORDS[0], "code": code, "type": "Phrase",
                    }], duplicate_confirmed=False)
                if duplicate:
                    self.assertIsNotNone(guard)
                    self.assertTrue(guard["pendingDuplicateConfirmation"])
                    state = pending_duplicate_confirmation_state(PendingToolConfirm("keytao_create_phrase", {
                        "word": b15_sibling_and_others_WORDS[0], "code": code, "type": "Phrase",
                    }), guard)
                    self.assertTrue(pending_duplicate_confirmation_is_complete(state))
                else:
                    self.assertIsNone(guard)

# Source: test_b17_rest_recommended.py
class B17ParserTests(unittest.TestCase):
    def test_plain_selection_defaults_rest_to_false(self):
        parsed = grammar.parse_reviewed_selection_command("名创 1 4，加入并提交")
        self.assertEqual(parsed.pairs, (("名创", "1"), ("名创", "4")))
        self.assertIs(getattr(parsed, "rest_recommended", None), False)

    def test_compatibility_projection_never_drops_rest_metadata(self):
        self.assertIsNone(grammar.parse_reviewed_multi_word_selection("名创 1 4，其他用推荐的 加入并提交"))
        self.assertEqual(grammar.parse_reviewed_multi_word_selection("名创 1 4"), (("名创", "1"), ("名创", "4")))

    def test_fresh_selection_cannot_drop_rest_without_ticket(self):
        for message in ("名创 1，其他用推荐的", "名创 mgwx，其他用推荐的", "名创 mgwx，其他用推荐的 加入并提交"):
            with self.subTest(message=message):
                self.assertIsNone(b17_rest_recommended_commands.fresh_entry_code_selection(message))

    def test_operation_projection_fails_closed_for_multiword_rest(self):
        ticket = b17_rest_recommended_shift_ticket()
        request = understand_message("名创 1，其他用推荐的 加入并提交", ticket)
        self.assertEqual(request.op, "none")

    def test_semantic_model_cannot_authorize_named_subset_of_rest(self):
        for op in ("add", "select_candidate"):
            with self.subTest(op=op):
                request, _ = validate_operation(raw_operation(
                    op=op, operands=["名创"], codes=["mgwx"], refers_to_pending=True,
                ), "名创 mgwx，其他用推荐的 加入并提交", b17_rest_recommended_shift_ticket())
                self.assertTrue(request.op == "none" or request.frame != "direct")

    def test_complete_rest_selection_is_not_mutation_grammar_gap(self):
        self.assertFalse(grammar.looks_like_mutation_grammar_gap("名创 1，其他用推荐的 加入并提交"))

    def test_semantic_pending_renderer_never_drops_rest(self):
        from keytao_bot.harness.operation_request import OperationRequest
        request = OperationRequest(op="select_candidate", operands=("名创",), codes=("mgwx",),
                                   frame="direct", refers_to_pending=True, submit_after=True)
        message = "名创 mgwx，其他用推荐的 加入并提交"
        rendered = b17_rest_recommended_commands.render_understood_pending_answer(request, b17_rest_recommended_shift_ticket(), "", message)
        self.assertTrue(not rendered or rendered == message, rendered)

for case_name, message, action in (
    ("required_other", "名创 1 4，其他用推荐的 加入并提交", "加入并提交"),
    ("required_remainder", "名创 1 4，其余用推荐 加入", "加入"),
    ("required_default", "名创 1 4 其他默认", ""),
    ("required_remaining", "名创 1 4，剩下的都用推荐的，加入并提交", "加入并提交"),
    ("variant_other", "名创 1 4，其它的词全按默认编码", ""),
    ("variant_rest", "名创 1 4，剩下走推荐的编码，加入草稿", "加入草稿"),
    ("variant_words", "名创 1 4，其余词都用默认的", ""),
    ("variant_action", "名创 1 4，其他推荐 加到草稿并提交", "加到草稿并提交"),
):
    setattr(B17ParserTests, "test_r1_positive_" + case_name, positive_parser_case(message, action))

for case_name, message in (
    ("negated", "名创 1 4，其他不用推荐的"),
    ("question", "名创 1 4，其他用推荐的吗"),
    ("question_mark", "名创 1 4，其他用推荐的？"),
    ("prefix", "其他用推荐的，名创 1 4"),
    ("middle", "名创 1，其他用推荐的，去骨 2"),
    ("missing_named", "其他用推荐的 加入并提交"),
    ("shift_only", "「占位词」顺延，其他用推荐的"),
    ("missing_separator", "名创 1 4其他默认"),
    ("extra_clause", "名创 1 4，其他用推荐的 然后删除全部"),
    ("repeated_rest", "名创 1 4，其他用推荐的，其余默认"),
    ("unknown_adverb", "名创 1 4，其他大概用推荐的"),
):
    setattr(B17ParserTests, "test_r1_negative_" + case_name, negative_parser_case(message))

# Source: test_b17_rest_recommended.py
class B17RestRouteTests(unittest.IsolatedAsyncioTestCase):
    setUp = B17RestRouteTests_setUp

    candidates = B17RestRouteTests_candidates

    select = B17RestRouteTests_select

    created = B17RestRouteTests_created

    submissions = B17RestRouteTests_submissions

    assert_offline = B17RestRouteTests_assert_offline

    async def test_r2_named_first_then_rest_in_ticket_order_and_submit(self):
        prompt = await self.candidates(*b17_rest_recommended_TEN_WORDS)
        await self.select("名创 1 4，其他用推荐的 加入并提交", prompt)
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm")] + [
            (word, b17_rest_recommended_FACTS[word]["recommended"]) for word in b17_rest_recommended_TEN_WORDS if word != "名创"
        ], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.assert_offline()

    async def test_r2_without_action_writes_draft_only(self):
        prompt = await self.candidates("脱骨", "名创", "去骨")
        await self.select("名创 1 4 其他默认", prompt)
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm"), ("脱骨", "togju"), ("去骨", "qjgju")], self.replay.last_response)
        self.assertEqual(self.submissions(), 0)
        self.assert_offline()

    async def test_r2_rest_matches_bare_add_sealed_items(self):
        prompt = await self.candidates("脱骨", "名创", "去骨")
        baseline = copy.deepcopy(self.replay.store.get_record(self.replay.author).state.args["items"])
        await self.select("加入", prompt)
        self.assertEqual(self.replay.draft, baseline)
        prompt = await self.candidates("脱骨", "名创", "去骨")
        sealed = copy.deepcopy(self.replay.store.get_record(self.replay.author).state.args["items"])
        await self.select("名创 1，其他用推荐的", prompt)
        self.assertEqual(self.replay.draft, [sealed[1], sealed[0], sealed[2]], self.replay.last_response)
        self.assert_offline()

    async def test_r2_give_way_preserves_sealed_manual_review_and_remark(self):
        reason = "「柱坐标系」@qzbx 证据不足"
        self.replay.review_overrides["柱坐标系"] = {
            "needsManualReview": True, "manualReviewReason": reason, "reviewDisposition": "SEAL",
            "preSubmitAudit": {"success": True, "autoApprove": False, "verdict": "manual", "summary": reason, "issues": [reason]},
        }
        async def tool(name, args, context):
            return await self.replay.tool(name, args, context.platform, context.user_id)

        with patch.object(self.replay.chat, "call_tool_function", call_tool_function), patch.object(
            self.replay.commands.tool_executor, "call", AsyncMock(side_effect=tool),
        ), patch.object(self.replay.commands, "_reviewed_add_verdicts", OrderedDict()):
            prompt = await self.candidates("球坐标系", "柱坐标系")
            sealed = copy.deepcopy(self.replay.store.get_record(self.replay.author).state.args["items"])
            self.assertEqual(sealed[1]["code"], "qzbxv")
            self.assertTrue(sealed[1]["needsManualReview"])
            self.assertIn("qzbxv", sealed[1]["manualReviewReason"])
            await self.select("球坐标系 1，其他用推荐的", prompt)
        self.assertEqual(self.replay.draft, sealed, self.replay.last_response)
        self.assert_offline()

    async def test_r2_no_recommendation_stays_on_original_ticket(self):
        self.replay.candidate_facts["去骨"]["recommended"] = ""
        self.replay.entries.extend({"word": "占位" + str(index), "code": code, "type": "Phrase"} for index, code in enumerate(b17_rest_recommended_FACTS["去骨"]["codes"]))
        prompt = await self.candidates("名创", "去骨", "脱骨")
        initial = self.replay.store.get_record(self.replay.author)
        self.assertEqual([item["word"] for item in initial.state.args["items"]], ["名创", "脱骨"])
        await self.select("名创 1 4，其他用推荐的", prompt)
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm"), ("脱骨", "togju")], self.replay.last_response)
        remaining = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(remaining)
        self.assertEqual(remaining.state.args["_query_words"], ["去骨"])
        self.assertIn("去骨", self.replay.last_response)
        self.assertIn("未选择：「去骨」；这些词本次未添加。", self.replay.last_response)
        self.assertEqual(remaining.origin_message_ids, initial.origin_message_ids)
        self.assert_offline()

    async def test_r2_duplicate_check_includes_rest_and_keeps_ticket(self):
        prompt = await self.candidates("球坐标系", "柱坐标系")
        record = self.replay.store.get_record(self.replay.author)
        before = copy.deepcopy(record)
        await self.select("球坐标系 qzbxv，其他用推荐的", prompt)
        self.assertEqual(self.replay.writes, [])
        self.assertEqual(self.replay.last_response, "球坐标系、柱坐标系 选了同一个编码 qzbxv，请改选；本次未写入。")
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertEqual(record, before)
        self.assert_offline()

    async def test_r2_invalid_named_selector_keeps_whole_ticket_and_no_write(self):
        prompt = await self.candidates("名创", "脱骨")
        record = self.replay.store.get_record(self.replay.author)
        before = copy.deepcopy(record)
        await self.select("名创 9，其他用推荐的 加入并提交", prompt)
        self.assertEqual(self.replay.writes, [])
        self.assertEqual(self.replay.last_response, b17_rest_recommended_SELECTION_ERROR)
        self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertEqual(record, before)
        self.assert_offline()

    async def test_r3_single_ticket_rest_adds_nothing_and_preserves_multiselect(self):
        prompt = await self.candidates("名创")
        await self.select("名创 1 4，其他用推荐的 加入并提交", prompt)
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm")], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        self.assert_offline()

    async def test_r3_single_ticket_structural_normalize_and_canonicalize(self):
        await self.candidates("名创")
        state = self.replay.store.get_record(self.replay.author).state
        message = "名创 1，其他用推荐的 加入并提交"
        intent = b17_rest_recommended_routing._structural_pending_add_word_intent(message, state)
        self.assertIsNotNone(intent)
        self.assertEqual((intent.intent, intent.choice_index), ("pending_add_and_submit", 1))
        self.assertEqual(b17_rest_recommended_routing._normalize_pending_candidate_selection_text(message, state.word), "1，加入并提交")
        canonical, error = await b17_rest_recommended_commands._canonicalize_pending_ticket_intent(state, message, intent, "qq", self.replay.author.actor_id)
        self.assertIsNone(error)
        self.assertEqual(canonical.choice_index, 1)
        self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(message, state))
        for reading, accepted in (("ming chuang", True), ("ming chong", False)):
            for action in ("", "，加入并提交"):
                with self.subTest(reading=reading, action=action):
                    prompt = await self.candidates("名创")
                    record = self.replay.store.get_record(self.replay.author)
                    before = copy.deepcopy(record)
                    await self.select(f"名创 {reading}: mgwx，其他用推荐的" + action, prompt)
                    self.assertEqual(self.created(), [("名创", "mgwx")] if accepted else [], self.replay.last_response)
                    self.assertEqual(self.submissions(), int(accepted and bool(action)))
                    if not accepted:
                        self.assertIs(self.replay.store.get_record(self.replay.author), record)
                        self.assertEqual(record, before)
                    self.assert_offline()

    async def test_r3_no_ticket_never_writes(self):
        await self.select("名创 1 4，其他用推荐的 加入并提交")
        self.assertEqual(self.replay.writes, [])
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r3_rest_cannot_authorize_generic_named_only_tool_without_ticket(self):
        from keytao_bot.harness.tools import ToolContext, ToolExecutor

        for message in (
            "名创 mgwx，其他用推荐的，加入并提交",
            "名创 mgwx，其他用推荐的，加入",
            "名创 mgwx，其他用推荐的",
            "名创 1，其他用推荐的，加入并提交",
            "名创 1 4，其他用推荐的 加入并提交",
        ):
            with self.subTest(message=message):
                allowed = grammar.message_authorizes_mutation(message)
                create = AsyncMock(return_value=json.dumps({"success": True}))
                executor = ToolExecutor(
                    lambda name: create if name == "keytao_create_phrase" else None,
                    frozenset(),
                )
                result = json.loads(await executor.call(
                    "keytao_create_phrase", {"word": "名创", "code": "mgwx"},
                    ToolContext(platform="qq", user_id="b17-owner", current_message=message,
                                writes_allowed=allowed),
                ))
                self.assertFalse(allowed)
                self.assertTrue(result.get("policyBlocked"), result)
                create.assert_not_awaited()

    async def test_r3_risk_confirmation_rejects_rest_reselection(self):
        await self.candidates("名创", "脱骨")
        state = copy.deepcopy(self.replay.store.get_record(self.replay.author).state)
        state.confirmation_source = "server_warning"
        derived, intent, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(state, "名创 1，其他用推荐的")
        self.assertIsNone(derived)
        self.assertIsNone(intent)
        self.assertIn("当前已进入风险确认", error or "")

    async def test_r3_unquoted_openai_route_honors_rest(self):
        await self.candidates("脱骨", "名创", "去骨")
        await self.select("名创 1 4，其他用推荐的 加入并提交")
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm"), ("脱骨", "togju"), ("去骨", "qjgju")], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        self.assert_offline()

    async def test_r4_four_words_two_saint_codes_submit(self):
        prompt = await self.candidates("圣者", "屁垫", "饰冠", "冠饰")
        await self.select("圣者 2 5，屁垫 2，饰冠 2，冠饰 2，加入并提交", prompt)
        self.assertEqual(self.created(), [("圣者", "egeri"), ("圣者", "egeruv"), ("屁垫", "pbdmi"), ("饰冠", "ekgri"), ("冠饰", "greki")], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        self.assert_offline()

    async def test_r4_ten_words_two_name_codes_keep_remainder(self):
        prompt = await self.candidates(*b17_rest_recommended_TEN_WORDS)
        self.assertIn("飞键", prompt)
        await self.select("名创 1 4，加入并提交", prompt)
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm")], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        remaining = self.replay.store.get_record(self.replay.author)
        self.assertEqual(remaining.state.args["_query_words"], [word for word in b17_rest_recommended_TEN_WORDS if word != "名创"])
        self.assert_offline()

    async def test_r4_reply_and_at_prefix_complete_route(self):
        prompt = await self.candidates(*b17_rest_recommended_TEN_WORDS)
        await self.select("[reply:id=1701][at:qq=3785773770] 名创 1 4，加入并提交", prompt)
        self.assertEqual(self.created(), [("名创", "mgwx"), ("名创", "mgwm")], self.replay.last_response)
        self.assertEqual(self.submissions(), 1)
        self.assertEqual(len(self.replay.store.get_record(self.replay.author).state.args["_query_words"]), 9)
        self.assert_offline()

    check_failed_then_corrected = B17RestRouteTests_check_failed_then_corrected

    async def test_r5_invalid_index_keeps_ticket_corrected_without_query(self):
        await self.check_failed_then_corrected("名创 9，加入并提交")

    async def test_r5_duplicate_selector_keeps_ticket_corrected_without_query(self):
        await self.check_failed_then_corrected("名创 1 1")

# Source: test_b17_rest_recommended.py
class B17ShiftPlanTests(unittest.IsolatedAsyncioTestCase):
    async def test_r2_rest_reorder_dispatch_matches_bare_add_plan(self):
        ticket = b17_rest_recommended_shift_ticket()
        baseline = b17_rest_recommended_commands._pending_batch_front_insert_plan(ticket)
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline["expectedOccupantsByCode"], {"eqtd": ["占位词"]})
        derived, intent, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(ticket, "名创 1，其他用推荐的")
        self.assertIsNone(error, error)
        self.assertIsNotNone(derived)
        self.assertEqual(intent.intent, "pending_confirm")
        self.assertEqual(b17_rest_recommended_commands._pending_batch_front_insert_plan(derived), baseline)
        calls = []
        key = harness.ConversationAddress.private("qq", "b17-shift")
        for message in ("加入", "名创 1，其他用推荐的"):
            store = harness.MemoryConversationStateStore()
            store.set(key, copy.deepcopy(ticket))
            execute = AsyncMock(return_value="Offline reorder receipt")
            with patch.object(harness.openai_chat_module, "conversation_state_store", store), patch.object(b17_rest_recommended_commands, "_execute_confirmed_tool", execute):
                response = await b17_rest_recommended_commands.handle_pending_message_core(message, "qq", "b17-shift", key, allow_intent_model=False)
            self.assertIn("Offline reorder receipt", response)
            execute.assert_awaited_once()
            sent = execute.await_args.args[0]
            self.assertEqual(sent.function_name, "keytao_shift_phrase_code")
            self.assertEqual((sent.args["word"], sent.args["target_code"]), ("刷头", "eqtd"))
            self.assertEqual(sent.args["_auto_confirm_expected_shifted_words"], ["占位词"])
            self.assertTrue(execute.await_args.kwargs["auto_confirm_shift_plan"])
            calls.append((sent.args, execute.await_args.kwargs))
        self.assertEqual(calls[0], calls[1])

class SelectionHTTP(b19_partial_reply_binding_DraftHTTP):
    def __init__(self):
        super().__init__()
        self.entries = []
        self.facts = {
            "苏洵": (["sjxw", "sjxwi", "sjxwia"], "su xun"),
            "素循": (["sjxw", "sjxwv", "sjxwva"], "su xun"),
            "苏辙": (["sjqe", "sjqei", "sjqeiv", "sjfe", "sjfei", "sjfeiv"], "su zhe"),
            "包伙": (["bzhl", "bzhlu", "bzhluo"], "bao huo"),
            "爆火": (["bzhl", "bzhlo", "bzhloo"], "bao huo"),
        }

    def _response(self, method, url, kwargs):
        from tests.helpers import urlsplit

        if urlsplit(url).path == "/api/bot/evidence/fetch":
            self.requests.append((method, "/api/bot/evidence/fetch", copy.deepcopy(kwargs.get("json") or {})))
            return OfflineResponse({}, 404)
        if urlsplit(url).path == "/api/phrases/encode":
            word = (kwargs.get("params") or {})["word"]
            self.requests.append((method, "/api/phrases/encode", {"word": word}))
            codes, pinyin = self.facts[word]
            return OfflineResponse({
                "input": word, "type": "二字词", "codes": codes, "altCodes": [],
                "flyKeyVariants": [], "phrasePinyins": pinyin.split(),
                "pronunciationSource": "zdic-phrase", "standardPronunciationStatus": "found",
                "chars": [{"char": char, "pinyin": syllable, "pinyins": [syllable],
                           "pronunciationLookupStatus": "found", "phoneticCode": codes[0][index * 2:index * 2 + 2],
                           "shapeCode": codes[1][4:]}
                          for index, (char, syllable) in enumerate(zip(word, pinyin.split()))],
            })
        return super()._response(method, url, kwargs)


class SelectionCommandTests(unittest.IsolatedAsyncioTestCase):
    tearDown = B18RealToolTests_tearDown
    turn = B18RealToolTests_turn
    created = B18RealToolTests_created

    def setUp(self):
        from tests.helpers import REAL_REVIEW_READS, review

        B18RealToolTests_setUp(self)
        self.http = SelectionHTTP()
        self.enterContext(patch.object(TOOLS.http_client, "_keytao_client", self.http))
        self.enterContext(patch.multiple(review, **REAL_REVIEW_READS))
        functions = {**harness._lookup_tools.TOOL_FUNCTIONS,
                     **harness._review_tools.TOOL_FUNCTIONS, **TOOLS.TOOL_FUNCTIONS}
        self.enterContext(patch.object(self.replay.chat.tool_executor, "_get_tool_function", functions.get))
        self.enterContext(patch.object(review.http_client, "guarded_fetch", AsyncMock(return_value=OfflineResponse({}, 404))))
        self.enterContext(patch.object(review, "_review_cache", OrderedDict()))
        self.enterContext(patch.object(review, "_evidence_host_failures", {}))

    async def test_f1_m3_03_named_selection_with_prefix_displacement(self):
        self.http.entries = [{"word": "素循", "code": "sjxw", "type": "Phrase", "weight": 100}]
        codes = [("sjxw", True), ("sjxwi", False), ("sjxwia", False)]
        state = PendingAddWord(
            "苏洵", "sjxw", codes, server_candidates=codes,
            server_occupied_words={"sjxw": ["素循"]},
            server_ordering_assessments=[{"verdict": "front_more_common", "newWord": "苏洵",
                                         "newCode": "sjxw", "occupantWord": "素循",
                                         "occupantCode": "sjxw", "freeCode": "sjxwi"}],
            pronunciation_codes={code: "su xun" for code, _ in codes}, needs_manual_review=True,
        )
        self.replay.store.set(self.replay.author, state)
        record = self.replay.store.get_record(self.replay.author)
        self.replay.history[:] = [{"role": "assistant", "content": self.replay.chat._render_live_single_candidate_record(record)}]
        await self.turn("苏洵 1，顺延“素循”")
        self.assertCountEqual(self.created(), [("苏洵", "sjxw"), ("素循", "sjxwv")], self.replay.last_response)
        self.assertEqual([(row["word"], row["code"]) for row in self.http.rows if row["action"] == "Delete"], [("素循", "sjxw")])
        self.assertEqual(len(self.http.writes), 1)

    async def test_f1_m3_12_add_word_with_possessive_explicit_code(self):
        await self.turn("加词：苏辙的编码为sjfei", raw_operation(op="other", operands=[], codes=[], frame="none"))
        self.assertEqual(self.created(), [("苏辙", "sjfei")], self.replay.last_response)
        self.assertEqual(len(self.http.writes), 1)
        self.assertTrue(any(path == "/api/phrases/encode" and body["word"] == "苏辙" for _method, path, body in self.http.requests))

    async def test_single_add_with_code_change_description(self):
        self.http.facts["洋节"] = (["ypjd", "ypjdv", "ypjdvo"], "yang jie")
        await self.turn("添加洋节，编码改为ypjd", raw_operation(op="add", operands=["洋节"], codes=["ypjd"]))
        self.assertEqual(self.created(), [("洋节", "ypjd")], self.replay.last_response)
        self.assertEqual(len(self.http.writes), 1)

    async def test_f1_m1_05_unbound_shift_looks_up_exact_word(self):
        self.http.entries = [{"word": word, "code": "bzhl", "type": "Phrase", "weight": 100 + index}
                             for index, word in enumerate(("包伙", "爆火"))]
        await self.turn("顺延包伙这个词", raw_operation(op="other", operands=[], codes=[], frame="none"))
        self.assertIn("包伙", self.replay.last_response)
        self.assertIn("bzhl", self.replay.last_response)
        self.assertNotIn("顺延包伙", self.replay.last_response)
        self.assertEqual(self.http.writes, [])
        self.assertTrue(any(path.endswith("/by-word/batch") and body["words"] == ["包伙"] for _method, path, body in self.http.requests))


# Source: test_b18_rest_real_tools.py
class B18RealToolTests(unittest.IsolatedAsyncioTestCase):
    setUp = B18RealToolTests_setUp

    tearDown = B18RealToolTests_tearDown

    seed = B18RealToolTests_seed

    turn = B18RealToolTests_turn

    created = B18RealToolTests_created

    def install_f2_review_http(self, facts, unconfirmed_words):
        from tests.helpers import REAL_REVIEW_READS, review, urlsplit

        class QueryHTTP(b19_partial_reply_binding_DraftHTTP):
            def _response(self, method, url, kwargs):
                path = urlsplit(url).path
                if path.endswith("/evidence/fetch"):
                    return OfflineResponse({}, 404)
                if path == "/api/phrases/encode":
                    word = kwargs["params"]["word"]
                    code, reading = facts[word]
                    self.requests.append((method, path, {"word": word}))
                    unconfirmed = word in unconfirmed_words
                    return OfflineResponse({
                        "input": word, "type": "多字词", "codes": [code, code + "o"],
                        "altCodes": [], "flyKeyVariants": [], "phrasePinyins": reading.split(),
                        "pronunciationSource": "zdic-unavailable" if unconfirmed else "zdic-phrase",
                        "standardPronunciationStatus": "unavailable" if unconfirmed else "found",
                        "semanticPronunciationNeeded": unconfirmed,
                        "chars": [{"char": char, "pinyin": syllable, "pinyins": [syllable],
                                   "pronunciationLookupStatus": "found"}
                                  for char, syllable in zip(word, reading.split())],
                        **self.encode_overrides.get(word, {}),
                    })
                return super()._response(method, url, kwargs)

        self.http = QueryHTTP()
        self.http.encode_overrides = {}
        self.http.entries.clear()
        self.enterContext(patch.object(TOOLS.http_client, "_keytao_client", self.http))
        self.enterContext(patch.multiple(review, **REAL_REVIEW_READS))
        self.enterContext(patch.object(review.http_client, "guarded_fetch", AsyncMock(return_value=OfflineResponse({}, 404))))
        self.enterContext(patch.object(review, "AsyncOpenAI", None))
        self.enterContext(patch.object(review, "_review_cache", OrderedDict()))
        self.enterContext(patch.object(review, "_semantic_review_cache", OrderedDict()))
        functions = {**harness._lookup_tools.TOOL_FUNCTIONS,
                     **harness._review_tools.TOOL_FUNCTIONS, **TOOLS.TOOL_FUNCTIONS}
        self.enterContext(patch.object(self.replay.chat.tool_executor, "_get_tool_function", functions.get))
        return review

    async def test_f3_m2_08_polyphonic_card_has_one_global_recommendation(self):
        self.install_f2_review_http({"降了": ("jxle", "jiàng le")}, {"降了"})
        self.http.encode_overrides["降了"] = {
            "codes": ["jxle", "jxlea", "jxleaa"],
            "pronunciationSource": "pinyin-pro-context", "standardPronunciationStatus": "absent",
            "alternatePhrasePronunciationCodes": [
                {"pinyins": ["xiáng", "le"], "codes": ["xxle", "xxlea", "xxleaa"]},
                {"pinyins": ["jiàng", "liǎo"], "codes": ["jxlc", "jxlca", "jxlcaa"]},
            ],
            "chars": [
                {"char": "降", "pinyin": "jiàng", "pinyins": ["jiàng", "xiáng"], "pronunciationLookupStatus": "found"},
                {"char": "了", "pinyin": "le", "pinyins": ["le", "liǎo"], "pronunciationLookupStatus": "found"},
            ],
        }
        self.http.entries = [{"word": word, "code": code, "type": "Phrase", "weight": 100}
                             for word, code in (("讲了", "jxle"), ("想了", "xxle"), ("浆料", "jxlc"))]
        result = await harness._review_tools.keytao_prepare_reviewed_add("降了")
        from keytao_bot.plugins.chat_render import _format_reviewed_add_prompt
        prompt = _format_reviewed_add_prompt(result)
        self.assertEqual(len(result["pronunciations"]), 1, result)
        self.assertEqual(result["pronunciations"][0]["pinyin"], "jiàng le")
        self.assertEqual(result["pronunciations"][0]["codes"], ["jxle", "jxlea", "jxleaa"])
        self.assertEqual(result["pronunciationResolution"]["status"], "encode_default")
        self.assertTrue(result["requiresManualPronunciationReview"])
        self.assertEqual(prompt.count("✅推荐"), 1, prompt)
        marker_line = next(line for line in prompt.splitlines() if "✅推荐" in line)
        self.assertRegex(marker_line, rf"^\d+\. {re.escape(result['recommendedCode'])} — [^\n]*✅推荐")
        self.assertNotIn(f"• 「降了」→ {result['recommendedCode']}（推荐）", prompt)
        self.assertEqual(self.http.writes, [])
        await self.turn("加词「降了」", raw_operation(op="add", operands=["降了"], codes=[]))
        self.assertEqual(self.replay.last_response.count("✅推荐"), 1, self.replay.last_response)
        candidate_prompt = self.replay.last_response
        self.assertTrue(self.replay.store.bind_origin_prompt_digest(
            self.replay.author, self.replay.chat._prompt_capability_digest(candidate_prompt),
        ))
        self.replay.history[:] = [{"role": "assistant", "content": candidate_prompt}]
        await self.replay.turn({"message": "降了 8，加入", "to_me": True, "quote_target": candidate_prompt},
                               raw_operation(op="select_candidate", refers_to_pending=True))
        self.assertEqual(self.http.writes, [], self.replay.last_response)
        self.assertIn("候选编号无效", self.replay.last_response)
        self.assertNotIn("jxlca", self.replay.last_response)
        await self.replay.turn({"message": "降了 2，加入", "to_me": True, "quote_target": candidate_prompt},
                               raw_operation(op="select_candidate", refers_to_pending=True))
        self.assertEqual(self.created(), [("降了", "jxlea")], self.replay.last_response)
        self.assertEqual(len(self.http.writes), 1)

    async def test_f3_m3_10_explicit_reading_survives_pre_submit_audit(self):
        self.install_f2_review_http({"什饼": ("ekbg", "shí bǐng")}, {"什饼"})
        self.http.encode_overrides["什饼"] = {"codes": ["ekbg", "ekbgi", "ekbgiu"]}
        self.http.entries = [{"word": "士兵", "code": "ekbg", "type": "Phrase", "weight": 100}]
        result = await harness._review_tools.keytao_prepare_reviewed_add("什饼", requested_reading="shi2 bing3")
        self.assertEqual(result["recommendedCode"], "ekbgi", result)
        self.assertIn("ekbgi", result["pronunciations"][0]["codes"])
        audited = result["preSubmitAudit"]["reviewedWords"]["什饼@Phrase"]
        self.assertTrue(audited["pronunciations"], audited)
        self.assertIn("ekbgi", audited["pronunciations"][0]["codes"], audited)
        self.assertEqual(audited["requestedReading"], "shi2 bing3")
        from keytao_bot.plugins.chat_render import _format_reviewed_add_prompt
        prompt = _format_reviewed_add_prompt(result)
        self.assertNotIn("不在读音候选链", prompt)
        self.assertNotIn("可选：无", prompt)
        self.assertIn("管理员审核", prompt)
        self.assertEqual(self.http.writes, [])

    async def test_f2_m1_08_unconfirmed_whole_word_uses_default_candidate_in_ten_word_list(self):
        facts = {
            "术曲": ("ejql", "shù qǔ"), "普信": ("pjxb", "pǔ xìn"),
            "关系户": ("gxho", "guān xì hù"), "萝卜岗": ("lbgi", "luó bo gǎng"),
            "代拍": ("dhph", "dài pāi"), "代抢": ("dhqx", "dài qiǎng"),
            "轻断食": ("qdev", "qīng duàn shí"), "乙游": ("ykyd", "yǐ yóu"),
            "卡砖": ("ksft", "kǎ zhuān"), "打本": ("dsbn", "dǎ běn"),
        }
        review = self.install_f2_review_http(facts, {"萝卜岗"})
        await self.turn(" ".join(facts), raw_operation(
            op="query", operands=list(facts), codes=[], refers_to_pending=False,
        ))
        reply = self.replay.last_response
        state = self.replay.store.get(self.replay.author)
        self.assertEqual([scope["word"] for scope in state.args["_candidate_scopes"]], list(facts), reply)
        self.assertIn("luó bo gǎng", reply)
        self.assertNotIn("请提供读音", reply)
        scope = next(scope for scope in state.args["_candidate_scopes"] if scope["word"] == "萝卜岗")
        self.assertTrue(scope["reviewedState"]["needsManualReview"])
        self.assertEqual(scope["reviewedState"]["pronunciationCodes"]["lbgi"], "luó bo gǎng")
        self.assertEqual(review.current_review_words.get(), ())
        self.assertEqual(self.http.writes, [])
        await self.turn("萝卜岗 1 加入")
        self.assertEqual(self.created(), [("萝卜岗", "lbgi")], self.replay.last_response)
        self.assertTrue(self.http.rows[0]["needsManualReview"])

    async def test_f2_unconfirmed_single_polyphone_with_model_down_stays_unresolved(self):
        review = self.install_f2_review_http({"枞壑": ("cyhe", "cóng hè")}, {"枞壑"})
        payment_error = RuntimeError("HTTP 402")
        payment_error.status_code = 402
        for error in (None, asyncio.TimeoutError(), payment_error):
            with self.subTest(error=type(error).__name__):
                review._semantic_review_cache.clear()
                create = AsyncMock(side_effect=error)
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                factory = None if error is None else lambda **kwargs: client
                with patch.object(review, "AsyncOpenAI", factory), patch.object(
                    review, "_review_llm_config", return_value={
                        "api_key": "offline", "base_url": "https://offline.invalid",
                        "model": "offline", "timeout": 1,
                    },
                ):
                    await self.turn("枞壑", raw_operation(
                        op="query", operands=["枞壑"], codes=[], refers_to_pending=False,
                    ))
                reply = self.replay.last_response
                self.assertIn("暂不推荐编码", reply)
                self.assertNotIn("cyhe", reply)
                self.assertIsNone(self.replay.store.get(self.replay.author), reply)
                self.assertEqual(review.current_review_words.get(), ())
                self.assertEqual(self.http.writes, [])
                if error is not None:
                    create.assert_awaited()

    async def test_f2_multiword_default_reading_refuses_context_and_unmatched_semantic_readings(self):
        facts = {"枞壑": ("cyhe", "cóng hè"), "普信": ("pjxb", "pǔ xìn")}
        review = self.install_f2_review_http(facts, {"枞壑"})
        for semantic in (False, True):
            with self.subTest(semantic=semantic):
                self.replay.store.delete(self.replay.author)
                review._semantic_review_cache.clear()
                self.http.encode_overrides = {} if semantic else {
                    "枞壑": {"contextPhrasePinyins": ["zōng", "hè"]},
                }
                response = harness._FakeAIResponse("stop", json.dumps({
                    "accepted": True, "confidence": 0.99,
                    "characters": [{"char": "枞", "pinyin": "zōng"}, {"char": "壑", "pinyin": "hè"}],
                    "meaning": "生有常绿乔木的山谷", "usageType": "place",
                }))
                create = AsyncMock(return_value=response)
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                with patch.object(review, "AsyncOpenAI", (lambda **kwargs: client) if semantic else None), patch.object(
                    review, "_review_llm_config", return_value={
                        "api_key": "offline", "base_url": "https://offline.invalid",
                        "model": "offline", "timeout": 1,
                    },
                ):
                    await self.turn(" ".join(facts), raw_operation(
                        op="query", operands=list(facts), codes=[], refers_to_pending=False,
                    ))
                state = self.replay.store.get(self.replay.author)
                self.assertEqual([scope["word"] for scope in state.args["_candidate_scopes"]], ["普信"])
                self.assertIn("「枞壑」", self.replay.last_response)
                self.assertNotIn("cyhe", self.replay.last_response)
                self.assertEqual(review.current_review_words.get(), ())
                self.assertEqual(self.http.writes, [])
                if semantic:
                    create.assert_awaited()

    async def test_f1_m3_01_multiselect_with_bare_submit_suffix(self):
        self.http = b19_partial_reply_binding_DraftHTTP()
        self.enterContext(patch.object(TOOLS.http_client, "_keytao_client", self.http))
        self.enterContext(patch.dict(b18_rest_real_tools_FACTS, {
            "债市": ("qheki", ["qhek", "qheki", "qhekiv", "fhek", "fheki", "fhekiv"], "zhai shi"),
        }))
        state = reviewed_ticket(("债市",), False).args["_candidate_scopes"][0]["reviewedState"]
        self.replay.store.set(self.replay.author, b17_rest_recommended_routing._pending_add_word_from_payload(json.loads(json.dumps(state))))
        record = self.replay.store.get_record(self.replay.author)
        self.replay.history[:] = [{"role": "assistant", "content": self.replay.chat._render_live_single_candidate_record(record)}]
        await self.turn("添加2、4，并提交")
        self.assertEqual(self.created(), [("债市", "qheki"), ("债市", "fhek")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_f1_m3_04_repeated_word_merges_distinct_candidate_selections(self):
        facts = {
            "纳兰性德": ("nlxda", ["nlxd", "nlxda", "nlxdao"], "na lan xing de"),
            "王献之": ("wxfvu", ["wxf", "wxfv", "wxfvu"], "wang xian zhi"),
            "苏辙": ("sjqei", ["sjqe", "sjqei", "sjqeiv", "sjfe", "sjfei", "sjfeiv"], "su zhe"),
            "晏几道": ("yjdo", ["yjd", "yjdo", "yjdou"], "yan ji dao"),
            "李煜": ("lkylv", ["lkyl", "lkylv", "lkylvo"], "li yu"),
            "王徽之": ("whfv", ["whf", "whfv", "whfvu"], "wang hui zhi"),
        }
        self.enterContext(patch.dict(b18_rest_real_tools_FACTS, facts))
        self.seed(tuple(facts), reorder=False)
        await self.turn("纳兰性德 2，王献之 3，苏辙 2，苏辙 5，晏几道 2，李煜 2，王徽之 2，加入并提交")
        self.assertCountEqual(self.created(), [(word, fact[0]) for word, fact in facts.items()] + [("苏辙", "sjfei")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_f2_multicode_named_plus_reorder_rest_submits_once(self):
        self.seed()
        await self.turn("招笑 1 4，其他用推荐的 加入并提交")
        expected = [("招笑", "qzxc"), ("招笑", "fzxc")] + [
            (word, b18_rest_real_tools_FACTS[word][0]) for word in b18_rest_real_tools_TEN_WORDS if word != "招笑"
        ] + [("义举", "ykjloa")]
        self.assertCountEqual(self.created(), expected, self.replay.last_response)
        self.assertIn(("义举", "ykjlo"), [(row["word"], row["code"]) for row in self.http.rows if row["action"] == "Delete"])
        self.assertEqual(sum(path.endswith("/pull-requests/batch") for path, _body in self.http.writes), 1)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r2_batch_multicode_named_plus_plain_rest(self):
        self.seed(("招笑", "小美", "小帅"), reorder=False)
        await self.turn("招笑 1 4，其他用推荐的 加入并提交")
        self.assertCountEqual(self.created(), [("招笑", "qzxc"), ("招笑", "fzxc"),
                                             ("小美", "xmww"), ("小帅", "xkeu")], self.replay.last_response)
        self.assertTrue(any(path.endswith("batch-draft") for path, _body in self.http.writes))
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_r2_named_reorder_one_code_with_plain_rest(self):
        self.seed(("宜居", "小美", "小帅"))
        await self.turn("宜居 2，其他用推荐的 加入并提交")
        self.assertCountEqual(self.created(), [("宜居", "ykjlo"), ("义举", "ykjloa"),
                                             ("小美", "xmww"), ("小帅", "xkeu")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_r2_named_reorder_two_codes_with_plain_rest(self):
        self.seed(("宜居", "小美", "小帅"))
        await self.turn("宜居 1 2，其他用推荐的 加入并提交")
        self.assertCountEqual(self.created(), [("宜居", "ykjl"), ("宜居", "ykjlo"), ("义举", "ykjloa"),
                                             ("小美", "xmww"), ("小帅", "xkeu")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_r2_named_reorder_two_codes_reversed_with_plain_rest(self):
        self.seed(("宜居", "小美", "小帅"))
        await self.turn("宜居 2 1，其他用推荐的 加入并提交")
        self.assertCountEqual(self.created(), [("宜居", "ykjl"), ("宜居", "ykjlo"), ("义举", "ykjloa"),
                                             ("小美", "xmww"), ("小帅", "xkeu")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_r2_named_reorder_word_selects_empty_code_with_plain_rest(self):
        self.seed(("宜居", "小美", "小帅"))
        await self.turn("宜居 1，其他用推荐的 加入并提交")
        self.assertCountEqual(self.created(), [("宜居", "ykjl"), ("小美", "xmww"), ("小帅", "xkeu")],
                              self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)

    async def test_r2_named_reorder_two_codes_cannot_take_occupants_only_destination(self):
        self.seed(("宜居", "小美", "小帅"))
        await self.turn("宜居 2 3，其他用推荐的 加入并提交")
        self.assertEqual(self.created(), [], self.replay.last_response)
        self.assertEqual(self.http.writes, [])

    async def test_r2_shift_multicode_named_without_submit(self):
        self.seed(("招笑", "宜居", "小美"))
        await self.turn("招笑 1 4，其他用推荐的 加入")
        self.assertCountEqual(self.created(), [("招笑", "qzxc"), ("招笑", "fzxc"), ("宜居", "ykjlo"),
                                             ("义举", "ykjloa"), ("小美", "xmww")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 0)

    async def test_r2_batch_without_submit(self):
        self.seed(("招笑", "小美"), reorder=False)
        await self.turn("招笑 1 4，其他用推荐的 加入")
        self.assertCountEqual(self.created(), [("招笑", "qzxc"), ("招笑", "fzxc"), ("小美", "xmww")],
                              self.replay.last_response)
        self.assertEqual(self.http.submissions, 0)

    async def test_r2_single_word_rest_uses_real_create_without_submit(self):
        ticket = reviewed_ticket(("小美",), False)
        payload = ticket.args["_candidate_scopes"][0]["reviewedState"]
        state = harness.PendingAddWord(
            "小美", "xmww", [("xmww", False), ("xmwwi", False)],
            server_candidates=[("xmww", False), ("xmwwi", False)],
            pronunciation_codes=payload["pronunciationCodes"], needs_manual_review=True,
        )
        self.assertTrue(self.replay.store.set(self.replay.author, state))
        await self.turn("小美 1，其他用推荐的 加入")
        self.assertEqual(self.created(), [("小美", "xmww")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 0)
        requests = [body for method, path, body in self.http.requests if path.endswith("/pull-requests/batch")]
        self.assertTrue(requests)
        self.assertTrue(all("previewOnly" in body for body in requests), requests)

    async def test_r2_single_word_rest_uses_real_create_and_submit(self):
        state = harness.PendingAddWord(
            "小美", "xmww", [("xmww", False), ("xmwwi", False)],
            server_candidates=[("xmww", False), ("xmwwi", False)],
            pronunciation_codes={"xmww": "xiao mei", "xmwwi": "xiao mei"}, needs_manual_review=True,
        )
        self.assertTrue(self.replay.store.set(self.replay.author, state))
        record = self.replay.store.get_record(self.replay.author)
        self.replay.store.bind_origin_message_ids(record, [1802])
        prompt = self.replay.chat._render_live_single_candidate_record(record)
        original = harness.ReplyReferenceInfo

        def reference(**kwargs):
            return original(**{**kwargs, "message_id": "1802"})

        with patch.object(harness, "ReplyReferenceInfo", side_effect=reference):
            await self.replay.turn({"message": "小美 1，其他用推荐的 加入并提交", "to_me": True,
                                    "quote_target": prompt}, raw_operation())
        self.assertEqual(self.created(), [("小美", "xmww")], self.replay.last_response)
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_r3_partial_selection_keeps_live_remainder_and_reply_is_truthful(self):
        rest = tuple(word for word in b18_rest_real_tools_TEN_WORDS if word != "宜居")
        record = self.seed(("马哲", *rest), reorder=False)
        self.replay.store.bind_origin_message_ids(record, [1801])
        await self.turn("马哲 2 5")
        self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea")], self.replay.last_response)
        remaining = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(remaining, self.replay.last_response)
        self.assertEqual(remaining.state.args["_query_words"], list(rest))
        self.assertIn("1801", remaining.origin_message_ids)
        self.assertEqual(remaining.created_at, record.created_at)
        self.assertEqual(remaining.expires_at, record.expires_at)
        self.assertEqual(remaining.expires_at - remaining.created_at, 14400)
        self.assertNotIn("候选仍有效，请逐词处理", self.replay.last_response)
        self.assertIn("未选择：" + "、".join("「" + word + "」" for word in rest), self.replay.last_response)
        await self.turn("其它的用默认编码")
        self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea")] + [
            (word, b18_rest_real_tools_FACTS[word][0]) for word in rest
        ], self.replay.last_response)
        self.assertEqual(self.http.submissions, 0)

    async def test_r4_rest_only_plain_words_adds_all_recommendations(self):
        for message in ("其它的用默认编码", "其他用推荐的", "其余用推荐 加入并提交"):
            with self.subTest(message=message):
                self.http.rows.clear()
                self.http.writes.clear()
                self.http.version = self.http.submissions = 0
                self.replay.store.delete(self.replay.author)
                self.seed(("小美", "小帅"), reorder=False)
                await self.turn(message)
                self.assertCountEqual(self.created(), [("小美", "xmww"), ("小帅", "xkeu")], self.replay.last_response)
                self.assertEqual(self.http.submissions, int("提交" in message), self.replay.last_response)

    async def test_r5_rest_only_without_ticket_never_looks_up_sentence(self):
        for message in ("其它的用默认编码", "其他用推荐的", "其余用推荐 加入并提交"):
            with self.subTest(message=message):
                await self.turn(message, raw_operation())
                self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
                self.assertEqual(self.http.requests, [])

    async def test_r5_semantic_confirm_without_ticket_never_looks_up_sentence(self):
        for frame in ("direct", "unknown", "question", "none"):
            with self.subTest(frame=frame):
                await self.turn("继续刚才那一批", raw_operation(op="confirm", operands=[], codes=[],
                                                          refers_to_pending=True, frame=frame))
                self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
                self.assertEqual(self.http.requests, [])

# Source: test_b18_rest_real_tools.py
class B18ShiftValidationTests(unittest.IsolatedAsyncioTestCase):
    setUp = B18ShiftValidationTests_setUp

    async def test_r1_invalid_companions_rejected_before_http(self):
        base = {"action": "Create", "word": "招笑", "code": "qzxc", "type": "Phrase", "needsManualReview": True}
        invalid = {
            "duplicate_pair": [base, dict(base)],
            "target_code": [{**base, "code": "ykjlo"}],
            "companion_code_collision": [base, {**base, "word": "小美"}],
            "non_create": [{**base, "action": "Delete"}],
            "too_many": [{**base, "word": "词" + str(index)} for index in range(21)],
            "same_word_outside_seal": [{**base, "word": "宜居", "code": "abcdef"}],
        }
        for case, items in invalid.items():
            with self.subTest(case=case), patch.object(TOOLS.httpx, "AsyncClient", side_effect=AssertionError("No HTTP"), create=True) as http:
                result = await TOOLS.keytao_shift_phrase_code(
                    "qq", "b18-owner", "宜居", "ykjlo", additional_items=items,
                    _reviewed_pinyin="yi ju", _reviewed_candidate_codes=["ykjl", "ykjlo", "ykjloa"],
                )
                self.assertFalse(result.get("success"), result)
                http.assert_not_called()

# Source: test_b7_selection.py
class B7SelectionTests(unittest.IsolatedAsyncioTestCase):
    resolve = B7SelectionTests_resolve

    def test_number_and_code_keep_advertised_reorder(self):
        for message in ("吴三桂 1，陈圆圆 1", "吴三桂 wsg，陈圆圆 jyy"):
            with self.subTest(message=message):
                derived = self.resolve(message)
                plan = b17_rest_recommended_commands._pending_batch_front_insert_plan(derived)
                self.assertIsNotNone(plan)
                self.assertFalse(plan.get("invalid"))
                self.assertEqual(plan["expectedOccupantsByCode"], {"wsg": ["我送给"], "jyy": ["纯音乐"]})
                self.assertFalse(any(item.get("manualReviewReason") == "重码添加需管理员审核" for item in derived.args["items"]))

    def test_alternate_empty_code_stays_plain_add(self):
        derived = self.resolve("吴三桂 2，陈圆圆 jyya")
        self.assertIsNone(b17_rest_recommended_commands._pending_batch_front_insert_plan(derived))
        self.assertEqual([(item["word"], item["code"]) for item in derived.args["items"]], [("吴三桂", "wsgo"), ("陈圆圆", "jyya")])

    async def test_unadvertised_occupied_code_stays_duplicate_add(self):
        ticket = incident_ticket(("皇太极",))
        scope = ticket.args["_candidate_scopes"][0]
        scope["orderingAssessments"].append({
            **scope["orderingAssessments"][0],
            "occupantWord": "火腿肠", "occupantCode": "htj", "newCode": "htj",
        })
        scope["reviewedState"]["serverOrderingAssessments"] = copy.deepcopy(scope["orderingAssessments"])
        for message in ("皇太极 1", "皇太极 htj"):
            with self.subTest(message=message):
                store = harness.MemoryConversationStateStore()
                key = harness.ConversationAddress.private("qq", "b7-selection")
                store.set(key, copy.deepcopy(ticket))
                execute = AsyncMock(return_value="Offline duplicate add receipt")
                with patch.object(harness.openai_chat_module, "conversation_state_store", store), patch.object(b17_rest_recommended_commands, "_execute_confirmed_tool", execute):
                    response = await b17_rest_recommended_commands.handle_pending_message_core(message, "qq", "b7-selection", key, allow_intent_model=False)
                self.assertIn("Offline duplicate add receipt", response)
                execute.assert_awaited_once()
                sent = execute.await_args.args[0]
                self.assertEqual(sent.function_name, "keytao_batch_add_to_draft")
                self.assertFalse(execute.await_args.kwargs.get("auto_confirm_shift_plan"))
                derived = self.resolve(message, ticket)
                item = derived.args["items"][0]
                self.assertEqual(item["code"], "htj")
                self.assertTrue(item["needsManualReview"])
                self.assertEqual(item["manualReviewReason"], "重码添加需管理员审核")
                derived_scope = derived.args["_candidate_scopes"][0]
                self.assertEqual(derived_scope["orderingAssessments"], [])
                self.assertEqual(derived_scope["reviewedState"]["serverOrderingAssessments"], [])
                self.assertIsNone(b17_rest_recommended_commands._pending_batch_front_insert_plan(derived))

    def test_exact_t7_keeps_only_selected_reorders(self):
        derived = self.resolve("一夫一妻 1， 一夫多妻 1， 吴三桂 1 ， 陈圆圆 1 ， 多尔衮 3， 皇太极 3， 福临 2")
        plan = b17_rest_recommended_commands._pending_batch_front_insert_plan(derived)
        self.assertFalse(plan.get("invalid"))
        self.assertEqual(plan["expectedOccupantsByCode"], {"wsg": ["我送给"], "jyy": ["纯音乐"]})
        self.assertEqual({(item["word"], item["code"]) for item in plan["additionalItems"]}, {
            ("一夫一妻", "yfyq"), ("一夫多妻", "yfdq"), ("多尔衮", "degov"), ("皇太极", "htjuv"), ("福临", "fjlbo"),
        })

    def test_t3_and_whitespace_mixed_shift_accept_same_advertised_pairs(self):
        for message in (
            "一夫一妻 1， 一夫多妻 1， 吴三桂 1， 我送给 顺延， 陈圆圆 1， 纯音乐顺延， 多尔衮 3",
            "吴三桂 1 我送给 顺延，陈圆圆 jyy，「纯音乐」顺延",
            "吴三桂 1 我送给顺延 陈圆圆 1 纯音乐顺延",
            '"我送给"顺延 吴三桂 wsg 陈圆圆 1 「纯音乐」顺延',
            '吴三桂 1，陈圆圆 1，"我送给" 和 「纯音乐」顺延',
            "「我送给」和「纯音乐」顺延",
            "我送给和纯音乐顺延",
            "吴三桂 1 陈圆圆 1 我送给和纯音乐顺延",
        ):
            with self.subTest(message=message):
                request = understand_message(message, incident_ticket())
                self.assertEqual(request.op, "select_candidate")
                self.assertTrue(request.refers_to_pending)
                derived = self.resolve(message)
                self.assertEqual(b17_rest_recommended_commands._pending_batch_front_insert_plan(derived)["expectedOccupantsByCode"], {"wsg": ["我送给"], "jyy": ["纯音乐"]})

    def test_shift_conjunction_preserves_quoted_and_bound_literal_words(self):
        self.assertEqual(grammar.parse_reviewed_shift_words("我送给和纯音乐顺延"), ("我送给", "纯音乐"))
        self.assertEqual(grammar.parse_reviewed_shift_words("「共和国」和纯音乐顺延"), ("共和国", "纯音乐"))
        self.assertEqual(grammar.parse_reviewed_shift_words("和平顺延"), ("和平",))
        self.assertEqual(grammar.parse_reviewed_shift_words("共和国顺延", known_words=("共和国",)), ("共和国",))
        self.assertEqual(grammar.parse_reviewed_shift_words("共和国和我送给顺延", known_words=("共和国", "我送给")), ("共和国", "我送给"))
        self.assertEqual(grammar.parse_reviewed_shift_words("「不忘初心」顺延"), ("不忘初心",))
        for message in ("他说我送给顺延", "不要我送给顺延", "我送给和她说纯音乐顺延"):
            self.assertIsNone(grammar.parse_reviewed_shift_words(message))
        ticket = incident_ticket(("吴三桂",))
        scope = ticket.args["_candidate_scopes"][0]
        scope["occupiedWords"]["wsg"] = ["共和国"]
        scope["orderingAssessments"][0]["occupantWord"] = "共和国"
        scope["reviewedState"]["serverOccupiedWords"] = scope["occupiedWords"]
        scope["reviewedState"]["serverOrderingAssessments"] = scope["orderingAssessments"]
        derived = self.resolve("吴三桂 1 共和国顺延", ticket)
        self.assertEqual(b17_rest_recommended_commands._pending_batch_front_insert_plan(derived)["expectedOccupantsByCode"], {"wsg": ["共和国"]})

    def test_unknown_occupant_contradictions_and_injections_fail_closed(self):
        for message in (
            "吴三桂 2，我送给顺延", "吴三桂 1，陌生人顺延", "吴三桂 1，我送给和陌生人顺延",
            "吴三桂 1，我送给顺延，然后删除纯音乐", "他说我送给顺延", "不要我送给顺延",
            "「我送给」顺延？", '"我送给顺延"', "吴三桂 1，吴三桂 1",
            "吴三桂 1我送给顺延", "吴三桂 1 我送给顺延删除纯音乐",
            "吴三桂 1 我送给顺延 删除纯音乐 1", '吴三桂 1，"我送给」顺延',
            '吴三桂 1，我送给顺延"', "转述吴三桂 1 我送给顺延",
            "不要吴三桂 1 我送给顺延", "吴三桂 1 我送给顺延？",
        ):
            with self.subTest(message=message):
                derived, intent, _ = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(incident_ticket(), message)
                self.assertIsNone(derived)
                self.assertIsNone(intent)
                self.assertFalse(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(message, incident_ticket()))

    async def test_selection_dispatches_existing_sealed_shift_plan(self):
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.private("qq", "b7-selection")
        store.set(key, incident_ticket())
        execute = AsyncMock(return_value="Offline reorder receipt")
        with patch.object(harness.openai_chat_module, "conversation_state_store", store), patch.object(b17_rest_recommended_commands, "_execute_confirmed_tool", execute):
            response = await b17_rest_recommended_commands.handle_pending_message_core("吴三桂 1，陈圆圆 1", "qq", "b7-selection", key, allow_intent_model=False)
        self.assertIn("Offline reorder receipt", response)
        execute.assert_awaited_once()
        sent = execute.await_args.args[0]
        self.assertEqual(sent.function_name, "keytao_shift_phrase_code")
        self.assertEqual(sent.args["_expected_occupants_by_code"], {"wsg": ["我送给"], "jyy": ["纯音乐"]})
        self.assertTrue(execute.await_args.kwargs["auto_confirm_shift_plan"])

# Source: test_b8_present_words.py
class B8PresentWordsTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = B8PresentWordsTests_asyncSetUp
    asyncTearDown = B8PresentWordsTests_asyncTearDown
    lookup = B8PresentWordsTests_lookup

    present = B8PresentWordsTests_present

    add = B8PresentWordsTests_add

    assert_present_copy = B8PresentWordsTests_assert_present_copy

    assert_candidates = B8PresentWordsTests_assert_candidates

    async def test_seven_present_and_one_candidate_keep_exact_controls_and_bindings(self):
        await self.assert_candidates(("哀声叹气",))

    async def test_seven_present_and_two_candidates_keep_exact_controls_and_bindings(self):
        await self.assert_candidates(("哀声叹气", "发薪"))

    async def test_all_present_share_one_fact_line_without_hint(self):
        response = await self.add(tuple(self.present))
        self.assert_present_copy(response)
        self.assertEqual(len(response.splitlines()), 1)
        self.assertIsNone(self.store.get_record(self.author))
        self.assertFalse(advertised_reply_contract(response).requires_live_state)

    async def test_single_present_word_keeps_original_copy(self):
        response = await self.add(("喜上眉梢",))
        self.assertEqual(response, already_existing_word_copy(
            "喜上眉梢", ("xsms",), can_choose_other_code=True,
        ))

# Source: test_b9_add_keeps_submit.py
class AddKeepsSubmitTests(unittest.IsolatedAsyncioTestCase):
    setUp = AddKeepsSubmitTests_setUp

    preview = AddKeepsSubmitTests_preview

    tool = AddKeepsSubmitTests_tool

    offer = AddKeepsSubmitTests_offer

    answer = AddKeepsSubmitTests_answer

    async def test_single_and_multi_candidates_keep_submit_through_delivery(self):
        for words in (("屡有",), ("屡有", "云翻")):
            with self.subTest(words=words):
                record = await self.offer(words)
                self.assertTrue(isinstance(record.state, PendingAddWord) or
                                record.state.function_name == "keytao_batch_add_to_draft")
                self.assertNotIn("原待确认提交已取消", self.replay.last_response)
                delivered = self.replay.chat._prepare_user_facing_reply(self.replay.last_response, self.replay.memory)
                self.assertNotIn("原待确认提交已取消", delivered)
                self.assertTrue(self.replay.chat._advertised_reply_matches_live_record(delivered, record))
                self.assertEqual(self.replay.chat.current_draft_delivery_claims.get(), [])

    async def test_add_and_submit_includes_own_old_items_once(self):
        for words in (("屡有",), ("屡有", "云翻")):
            with self.subTest(words=words):
                self.replay.draft = copy.deepcopy(self.old_draft)
                self.replay.writes.clear()
                await self.offer(words)
                response = await self.answer("加入并提交")
                self.assertIn("已提交审核", response)
                self.assertEqual([item["word"] for item in self.submitted[-1]], ["驴友", *words])
                self.assertEqual(sum(name == "keytao_submit_batch" for name, _ in self.replay.writes), 1)
                self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_foreign_old_items_still_need_scope_confirmation(self):
        self.foreign = True
        await self.offer()
        response = await self.answer("加入并提交")
        self.assertIn("本轮之外的内容", response)
        self.assertIn("确认", response)
        self.assertEqual(self.submitted, [])
        state = self.replay.store.get(self.replay.author)
        self.assertEqual(state.function_name, "keytao_submit_batch")
        self.assertIn("已提交审核", await self.answer("确认"))
        self.assertEqual([item["word"] for item in self.submitted[0]], ["驴友", "屡有", "云翻"])

    async def test_cancel_new_candidates_leaves_old_draft_untouched(self):
        await self.offer()
        self.replay.calls.clear()
        self.assertEqual(await self.answer("取消"), "已取消。")
        self.assertEqual(self.replay.draft, self.old_draft)
        self.assertEqual(self.replay.calls, [])
        self.assertEqual(self.replay.writes, [])
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

# Source: test_hotfix_multiword.py
class HotfixMultiwordTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = HotfixMultiwordTests_asyncSetUp

    unresolved_message = HotfixMultiwordTests_unresolved_message

    unresolved_review = HotfixMultiwordTests_unresolved_review

    lookup = HotfixMultiwordTests_lookup

    query = HotfixMultiwordTests_query

    assert_candidates = HotfixMultiwordTests_assert_candidates

    async def test_f2_m3_06_mixed_candidates_name_failed_word_without_generic_failure(self):
        self.codes.update({"坐坐": ("zlzl", "zlzli"), "切歌": ("qdge", "qdgev")})
        original = self.lookup
        for failure in (
            "后续处理未能生成安全回复；本次未执行新的写入。", "这次服务没有完成处理，本次未写入。",
            "「那很」后续处理未能生成安全回复；本次未执行新的写入。",
            "「那很」这次服务没有完成处理，本次未写入。",
        ):
            with self.subTest(failure=failure):
                async def lookup(name, args, *actor, **kwargs):
                    if name == "keytao_prepare_reviewed_add" and args["word"] == "那很":
                        return json.dumps({"success": False, "message": failure})
                    return await original(name, args, *actor, **kwargs)

                with patch.object(self.commands, "call_tool_function", side_effect=lookup):
                    reply = await self.commands._prepare_multi_word_query(
                        ("那很", "坐坐", "切歌"), "qq", self.author.actor_id, self.author,
                        self.author.space_key, "Offline owner",
                    )
                self.assertIn("「那很」", reply)
                self.assertIn("「坐坐」", reply)
                self.assertIn("「切歌」", reply)
                self.assertNotIn(failure, reply)
                self.assertEqual([row["word"] for row in self.store.get(self.author).args["items"]], ["坐坐", "切歌"])

    async def test_four_resolved_and_one_unresolved(self):
        await self.assert_candidates(("没头没尾", "国潮", "泥炭", "棉麻", "车损"))

    async def test_one_resolved_and_one_unresolved(self):
        await self.assert_candidates(("国潮", "车损"))

    async def test_unresolved_before_resolved(self):
        await self.assert_candidates(("车损", "国潮"))

    async def test_missing_candidate_reading_remediation(self):
        self.failure_mode = "missing_reading"
        await self.assert_candidates(("国潮", "车损"))

    async def test_two_failed_words_have_one_short_line_each(self):
        self.unresolved = {"车损", "棉麻"}
        await self.assert_candidates(("车损", "国潮", "棉麻"))

    async def test_all_resolved_unchanged(self):
        self.unresolved.clear()
        delivered = await self.assert_candidates(("没头没尾", "国潮", "泥炭", "棉麻", "车损"))
        state = self.store.get(self.author)
        self.assertEqual(delivered, self.commands.render_server_backed_batch_candidates(
            state.args["items"], state.args["_candidate_scopes"],
        ))

    async def test_all_unresolved_keeps_per_word_explanations(self):
        self.unresolved = {"国潮", "车损"}
        self.failure_mode = "pronunciation_unresolved"
        self.unresolved_message = lambda word: (
            f"「{word}」的权威整词或逐字读音服务暂不可用，当前读音无法完成交叉验证，暂不推荐编码"
        )
        raw, delivered = await self.query(("国潮", "车损"))
        self.assertEqual(delivered.splitlines(), [line for line in raw.splitlines() if line])
        for word in self.unresolved:
            self.assertIn(self.unresolved_message(word), delivered)
        self.assertNotIn("候选编码", delivered)
        self.assertIsNone(self.store.get_record(self.author))

    async def test_all_unresolved_preserves_existing_plain_explanations(self):
        self.unresolved = {"国潮", "车损"}
        self.failure_mode = "pronunciation_unresolved"
        self.unresolved_message = lambda word: f"「{word}」未取得能确认整词读音的证据，暂不推荐编码。"
        raw, delivered = await self.query(("国潮", "车损"))
        self.assertEqual(delivered.splitlines(), [line for line in raw.splitlines() if line])
        for word in self.unresolved:
            self.assertIn(self.unresolved_message(word), delivered)
        self.assertIsNone(self.store.get_record(self.author))

    async def test_no_resolved_scope_preserves_original_remediation_blocks(self):
        self.unresolved = {"国潮", "车损"}
        response = await self.commands._prepare_multi_word_query(
            ("国潮", "车损"), "qq", self.author.actor_id, self.author,
            self.author.space_key, "Offline owner",
        )
        self.assertEqual(response, "\n\n".join(
            render_remediation_reply(
                "编码服务未能返回有效候选编码", command=f"加词 {word}", words=(word,),
            ) for word in ("国潮", "车损")
        ))
        self.assertIsNone(self.store.get_record(self.author))

    async def test_mixed_query_preserves_plain_unresolved_explanation(self):
        self.failure_mode = "pronunciation_unresolved"
        raw, delivered = await self.query(("国潮", "车损"))
        self.assertIn(self.unresolved_message("车损"), delivered)
        self.assertEqual(delivered.splitlines(), [line for line in raw.splitlines() if line])
        record = self.store.get_record(self.author)
        self.assertEqual([row["word"] for row in record.state.args["items"]], ["国潮"])
        self.assertTrue(self.chat._advertised_reply_matches_live_record(delivered, record))

    async def test_real_multi_sense_message_with_reading_and_code_lines(self):
        self.failure_mode = "pronunciation_unresolved"
        self.unresolved = {"出圈"}
        self.unresolved_pronunciations = [
            {"pinyin": "chu juan", "codes": ["jjjt", "jjjto"]},
            {"pinyin": "chū quān", "codes": ["jjqt", "jjqto"]},
        ]
        self.codes["出圈"] = ("jjjt", "jjqt")
        self.unresolved_message = lambda word: (
            f"「{word}」存在含义不同的多个读音，当前含义与常用度证据"
            "未能唯一支持其中一个；本次不推荐编码，也不会创建待确认加词操作。\n"
            "- chu juan：jjjt、jjjto\n- chū quān：jjqt、jjqto\n"
            "请明确要采用的读音或具体含义。"
        )
        raw, delivered = await self.query(("国潮", "出圈"))
        self.assertEqual(delivered.splitlines(), [line for line in raw.splitlines() if line])
        self.assertIn(self.unresolved_message("出圈"), delivered)
        record = self.store.get_record(self.author)
        self.assertEqual([row["word"] for row in record.state.args["items"]], ["国潮"])
        self.assertTrue(self.chat._advertised_reply_matches_live_record(delivered, record))

    async def test_real_requested_reading_mismatch_message(self):
        self.failure_mode = "pronunciation_unresolved"
        self.unresolved = {"出圈"}
        self.codes["出圈"] = ("jjjt", "jjqt")
        self.unresolved_message = lambda word: (
            f"「{word}」的指定读音 chu qiong 与编码服务返回的"
            "候选读音都不匹配。可用读音：chū juàn、chū quān。\n"
            "- chū juàn：jjjt、jjjto\n- chū quān：jjqt、jjqto\n"
            "请明确要采用的读音或具体含义；本次未写入。"
        )
        raw, delivered = await self.query(("出圈", "国潮"))
        self.assertEqual(delivered.splitlines(), [line for line in raw.splitlines() if line])
        self.assertIn(self.unresolved_message("出圈"), delivered)
        record = self.store.get_record(self.author)
        self.assertEqual([row["word"] for row in record.state.args["items"]], ["国潮"])
        self.assertTrue(self.chat._advertised_reply_matches_live_record(delivered, record))

    async def test_explicit_add_merges_live_single_word_candidates(self):
        await self.commands._try_handle_simple_single_word_query(
            "加词 国潮", "qq", self.author.actor_id, self.author,
            self.author.space_key, "Offline owner",
        )
        reply = await self.commands._try_handle_simple_single_word_query(
            "加词 泥炭", "qq", self.author.actor_id, self.author,
            self.author.space_key, "Offline owner",
        )
        state = self.store.get(self.author)
        self.assertIsInstance(state, PendingToolConfirm)
        self.assertEqual([row["word"] for row in state.args["items"]], ["国潮", "泥炭"])
        self.assertIn("国潮", reply)
        self.assertIn("泥炭", reply)

    async def test_standalone_shift_reuses_shared_code_plan(self):
        keeper = {"word": "吴三桂", "code": "wsg", "type": "Phrase", "weight": 100}
        occupant = {"word": "我送给", "code": "wsg", "type": "Phrase", "weight": 100}

        async def lookup(name, args, *unused, **kwargs):
            if name == "keytao_lookup_by_words_batch":
                return json.dumps({"success": True, "results": [{"word": "我送给", "phrases": [occupant]}]})
            if name == "keytao_list_draft_items":
                return json.dumps({"success": True, "batchId": "offline", "contentVersion": 1,
                                   "items": [{**keeper, "action": "Create", "id": 1}]})
            if name == "keytao_lookup_by_code":
                self.assertEqual(args, {"code": "wsg"})
                return json.dumps({"success": True, "phrases": [occupant]})
            if name == "keytao_shift_phrase_code":
                result = {
                    "success": True, "batchId": "offline", "contentVersion": 2,
                    "shiftPlan": {"word": "吴三桂", "targetCode": "wsg", "items": [],
                                  "shifted": [{"word": "我送给", "fromCode": "wsg", "toCode": "wsgu"}]},
                }
                if not args.get("confirmed_plan_digest"):
                    result.update(success=False, requiresConfirmation=True, confirmationKind="shiftPlan",
                                  planDigest="a" * 64, warningDigest="b" * 64, warnings=[])
                return json.dumps(result)
            raise AssertionError(name)

        execute = AsyncMock(wraps=self.commands._execute_shift_to_code)
        with patch.object(self.commands, "call_tool_function", side_effect=lookup), patch.object(
            self.commands, "_execute_shift_to_code", execute,
        ), patch.object(
            self.commands, "_format_draft_response", AsyncMock(return_value="草稿地址：https://offline.invalid/draft"),
        ), patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": "direct_exec"}):
            reply = await self.commands.try_handle_advertised_shift(
                '"我送给"顺延', "qq", self.author.actor_id, self.author,
                self.author.space_key, "Offline owner",
            )
        self.assertIn("我送给", reply)
        self.assertIn("wsgu", reply)
        self.assertEqual(execute.await_args.args[:2], ("吴三桂", "wsg"))
        self.assertTrue(execute.await_args.kwargs["auto_confirm_shift_plan"])
        self.assertEqual(execute.await_args.kwargs["auto_confirm_expected_shifted_words"], ("我送给",))

    async def test_explicit_multi_add_merges_one_inventory_and_skips_readonly_query(self):
        prepare = self.commands._prepare_multi_word_query
        args = ("qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner")
        await prepare(("国潮", "泥炭"), *args, explicit_add=True)
        reply = await prepare(("棉麻",), *args, explicit_add=True)
        state = self.store.get(self.author)
        self.assertEqual(state.args["_query_words"], ["国潮", "泥炭", "棉麻"])
        self.assertEqual([row["word"] for row in state.args["items"]], ["国潮", "泥炭", "棉麻"])
        self.assertTrue(self.chat._advertised_reply_matches_live_record(reply, self.store.get_record(self.author)))
        selected, intent, error = self.chat._resolve_multi_word_pending_candidate_selection(state, "国潮 1，泥炭 2，棉麻 1")
        self.assertIsNone(error)
        self.assertIsNotNone(intent)
        self.assertEqual([row["code"] for row in selected.args["items"]], ["gljz", "nktfa", "mmms"])
        await prepare(("国潮", "泥炭"), *args)
        await prepare(("棉麻",), *args, explicit_add=True)
        self.assertEqual(self.store.get(self.author).args["_query_words"], ["棉麻"])

    async def test_incident_six_then_four_explicit_add_keeps_eight_actionable_words(self):
        earlier = ("一夫一妻", "一夫多妻", "吴三桂", "陈圆圆", "李自成", "多尔衮")
        added = ("皇太极", "顺治", "福临", "努尔哈赤")
        self.codes.update({
            word: (code, code + "o") for word, code in zip(
                (*earlier, "皇太极", "福临"), ("yfyq", "yfdq", "wsg", "jyy", "lzj", "drg", "htj", "fjl"),
            )
        })
        original_lookup = self.lookup

        async def lookup(name, args, *unused, **kwargs):
            if name == "keytao_lookup_by_word" and args["word"] in {"顺治", "努尔哈赤"}:
                return json.dumps({"success": True, "phrases": [{"word": args["word"], "code": "abcd", "type": "Phrase"}]})
            if name == "keytao_lookup_by_words_batch":
                return json.dumps({"success": True, "results": [
                    {"success": True, "word": word, "phrases": [
                        {"word": word, "code": "abcd", "type": "Phrase"},
                    ] if word in {"顺治", "努尔哈赤"} else []}
                    for word in args["words"]
                ]})
            return await original_lookup(name, args, *unused, **kwargs)

        args = ("qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner")
        with patch.object(self.commands, "call_tool_function", side_effect=lookup):
            await self.commands._try_handle_simple_single_word_query("加词：" + "，".join(earlier), *args)
            reply = await self.commands.try_merge_add_candidates("加词：" + "，".join(added), *args)
        expected = [*earlier, "皇太极", "福临"]
        self.assertEqual([row["word"] for row in self.store.get(self.author).args["items"]], expected)
        self.assertEqual(self.store.get_record(self.author).owner_key, self.author)
        for word in expected:
            self.assertIn(word, reply)
        selected, intent, error = self.chat._resolve_multi_word_pending_candidate_selection(
            self.store.get(self.author), "，".join(f"{word} 1" for word in expected),
        )
        self.assertIsNone(error)
        self.assertIsNotNone(intent)
        self.assertEqual([row["word"] for row in selected.args["items"]], expected)

    async def test_merge_excludes_warning_submit_and_preserving_query(self):
        args = ("qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner")
        for function_name, source in (("keytao_batch_add_to_draft", "server_warning"), ("keytao_submit_batch", "local_preview")):
            with self.subTest(function_name=function_name):
                await self.commands._prepare_multi_word_query(("国潮", "泥炭"), *args, explicit_add=True)
                state = self.store.get(self.author)
                state.function_name, state.confirmation_source = function_name, source
                self.assertEqual(self.commands._mergeable_add_candidate_scopes(*args[:4]), [])
                record = self.store.get_record(self.author)
                await self.commands._prepare_multi_word_query(("棉麻",), *args, preserve_pending=True, explicit_add=True)
                self.assertIs(self.store.get_record(self.author), record)

    async def test_add_extension_preserves_active_operation_and_ticket(self):
        args = ("qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner")
        await self.commands._prepare_multi_word_query(("国潮", "泥炭"), *args, explicit_add=True)
        record = self.store.get_record(self.author)
        coordinator = DraftOperationCoordinator()
        self.assertIsNotNone(coordinator.begin(self.author, "add"))
        with patch.object(self.commands, "draft_operation_coordinator", coordinator):
            self.assertIsNone(await self.commands.try_merge_add_candidates("加词 棉麻", *args))
        self.assertIs(self.store.get_record(self.author), record)

    async def test_standalone_shift_single_advertised_pair_uses_real_preview_and_receipt(self):
        state = PendingAddWord(
            "吴三桂", "wsg", [("wsg", True), ("wsgo", False)],
            server_candidates=[("wsg", True), ("wsgo", False)],
            server_occupied_words={"wsg": ["我送给"]},
            server_ordering_assessments=[{
                "verdict": "front_more_common", "newWord": "吴三桂", "occupantWord": "我送给",
                "occupantCode": "wsg", "freeCode": "wsgo", "newCode": "wsg",
            }],
            pronunciation_codes={"wsg": "wu san gui", "wsgo": "wu san gui"},
            needs_manual_review=False,
        )
        self.store.set(self.author, state)
        calls = []

        async def shift(name, args, *unused, **kwargs):
            self.assertEqual(name, "keytao_shift_phrase_code")
            calls.append(args)
            data = {
                "success": True, "batchId": "offline", "contentVersion": 1,
                "shiftPlan": {"word": "吴三桂", "targetCode": "wsg", "items": [],
                              "shifted": [{"word": "我送给", "fromCode": "wsg", "toCode": "wsgu"}]},
            }
            if not args.get("confirmed_plan_digest"):
                data.update(success=False, requiresConfirmation=True, confirmationKind="shiftPlan",
                            planDigest="a" * 64, warningDigest="b" * 64, warnings=[])
            return json.dumps(data)

        with patch.object(self.commands, "call_tool_function", side_effect=shift), patch.object(
            self.commands, "_format_draft_response", AsyncMock(return_value="草稿地址：https://offline.invalid/draft"),
        ):
            reply = await self.commands.try_handle_advertised_shift(
                "「我送给」顺延", "qq", self.author.actor_id, self.author,
                self.author.space_key, "Offline owner",
            )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["target_code"], "wsg")
        self.assertNotIn("confirmed_plan_digest", calls[0])
        self.assertEqual(calls[1]["confirmed_plan_digest"], "a" * 64)
        self.assertIn("我送给", reply)
        self.assertIn("wsgu", reply)

    async def test_standalone_shift_rejects_unrelated_or_foreign_words(self):
        with patch.object(self.commands, "_execute_shift_to_code", AsyncMock()) as execute:
            self.assertIsNone(await self.commands.try_handle_advertised_shift(
                "他说我送给顺延", "qq", self.author.actor_id, self.author, self.author.space_key, "Offline owner",
            ))
            reply = await self.commands.try_handle_advertised_shift(
                "我送给顺延", "qq", "different-owner", self.author, self.author.space_key, "Offline owner",
            )
            self.assertIn("本次未写入", reply)
            execute.assert_not_awaited()

    async def test_standalone_shift_real_server_plan_preserves_existing_draft_review(self):
        await self._check_standalone_shift_real_server_plan(keeper_in_draft=True)

    async def test_standalone_shift_real_server_plan_keeps_live_keeper_unchanged(self):
        await self._check_standalone_shift_real_server_plan(keeper_in_draft=False)

    _check_standalone_shift_real_server_plan = HotfixMultiwordTests__check_standalone_shift_real_server_plan

# Source: test_s54_multiword.py
class MultiwordRouteTests(unittest.TestCase):
    discover = MultiwordRouteTests_discover

    def test_bare_two_words_persists_one_reviewed_record(self):
        async def run():
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s54-offline")
            calls = []

            async def tool(name, args, *_args, **_kwargs):
                calls.append((name, args))
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(s54_multiword_review_payload(args["word"]))
                raise AssertionError(name)

            with patch.object(s54_multiword_chat, "conversation_state_store", store), patch.object(
                s54_multiword_chat, "call_tool_function", side_effect=tool,
            ), patch.object(
                s54_multiword_chat, "_classify_simple_word_query_intent",
                AsyncMock(return_value=harness.SimpleWordQueryIntent(
                    True, ("大端", "小端"), "word_lookup", 1.0,
                )),
            ), patch.object(store, "set", wraps=store.set) as persist:
                response = await s54_multiword_commands._try_handle_simple_single_word_query(
                    "大端 小端", "qq", "s54-offline", key,
                )
                self.assertIsNotNone(response, "bare multiword query fell through")
                self.assertEqual(persist.call_count, 1)
                state = store.get(key)
                self.assertIsInstance(state, harness.PendingToolConfirm)
                self.assertTrue(state.args["_reviewed_multi_word"])
                self.assertEqual([s["word"] for s in state.args["_candidate_scopes"]], ["大端", "小端"])
                self.assertEqual(response.count("审词："), 2)
                self.assertIn("dà duān", response)
                self.assertIn("常用度证据不足", response)
                self.assertTrue(s54_multiword_chat._advertised_reply_matches_live_record(response, store.get_record(key)))
                self.assertEqual(s54_multiword_chat._enforce_advertised_reply_contract(response, key), response)
            self.assertEqual([args["word"] for name, args in calls if name == "keytao_prepare_reviewed_add"], ["大端", "小端"])
        asyncio.run(run())

    def test_numbered_candidates_without_record_are_refused(self):
        key = harness.ConversationAddress.private("qq", "s54-boundary")
        store = harness.MemoryConversationStateStore()
        for code in ("dsdt", "`dsdt`", "**dsdt**"):
            response = f"候选：\n1. {code} — 已有「打断」"
            with patch.object(s54_multiword_chat, "conversation_state_store", store):
                delivered = s54_multiword_chat._enforce_advertised_reply_contract(response, key)
            self.assertNotIn("dsdt", delivered)
            self.assertIn("未", delivered)

    def test_separators_three_words_and_existing_word_share_pipeline(self):
        async def run():
            for text in ("大端 小端 肌群", "大端、小端、肌群", "大端，小端，肌群", "大端\n小端\t肌群"):
                response, store, key = await self.discover(text)
                self.assertEqual(response.count("审词："), 3)
                self.assertEqual(len(store.get(key).args["items"]), 3)
            response, store, key = await self.discover(existing=("小端",))
            self.assertIn("已在词库", response)
            self.assertNotIn("换码", response)
            self.assertEqual([i["word"] for i in store.get(key).args["items"]], ["大端"])
            self.assertTrue(s54_multiword_chat._advertised_reply_matches_live_record(response, store.get_record(key)))
            compact_reply = "\n".join(line for line in response.splitlines() if line.strip())
            self.assertTrue(s54_multiword_chat._advertised_reply_matches_live_record(compact_reply, store.get_record(key)))
            with patch.object(s54_multiword_chat, "conversation_state_store", store):
                self.assertEqual(s54_multiword_chat._enforce_advertised_reply_contract(compact_reply, key), compact_reply)
        asyncio.run(run())

    def test_cap_stops_before_any_tool_and_verbs_are_not_lexical_items(self):
        async def run():
            words = ["词" + char for char in "甲乙丙丁戊己庚辛壬癸子"]
            with patch.object(s54_multiword_chat, "call_tool_function", side_effect=AssertionError("over cap")):
                result = await s54_multiword_commands._try_handle_simple_single_word_query(" ".join(words), "qq", "s54")
            self.assertIn("最多查询 10 个词", result)
            self.assertEqual(s54_multiword_commands._bare_multi_word_query_words("加 小端"), ())
            self.assertEqual(s54_multiword_commands._bare_multi_word_query_words("删除 大端"), ())
        asyncio.run(run())

    def test_sqlite_reload_partial_selection_and_actor_isolation(self):
        async def run():
            with tempfile.TemporaryDirectory() as directory:
                path = directory + "/pending.db"
                original = SQLiteConversationStateStore(path)
                response, _store, key = await self.discover(store=original)
                reloaded = SQLiteConversationStateStore(path)
                record = reloaded.get_record(key)
                self.assertEqual(record.nonce, original.get_record(key).nonce)
                self.assertTrue(s54_multiword_chat._advertised_reply_matches_live_record(response, record))
                execute = AsyncMock(return_value="selected draft saved")
                with patch.object(s54_multiword_chat, "conversation_state_store", reloaded), patch.object(
                    s54_multiword_commands, "_execute_confirmed_tool", execute,
                ):
                    other = harness.ConversationAddress.private("qq", "other-s54")
                    self.assertIsNone(await s54_multiword_commands.handle_pending_message_core("小端 2", "qq", "other-s54", other, allow_intent_model=False))
                    receipt = await s54_multiword_commands.handle_pending_message_core("小端 2", "qq", "s54-offline", key, allow_intent_model=False)
                    execute.assert_not_awaited()
                    self.assertIn("常用度", receipt)
                    self.assertIn("未加入", receipt)
                    receipt = await s54_multiword_commands.handle_pending_message_core("小端 3", "qq", "s54-offline", key, allow_intent_model=False)
                    execute.assert_awaited_once()
                selected = execute.await_args.args[0]
                self.assertEqual([(item["word"], item["code"]) for item in selected.args["items"]], [("小端", "xcdtvo")])
                self.assertIn("未选择：「大端」", receipt)
                self.assertIsNone(reloaded.get_record(key))
        asyncio.run(run())

    def test_word_set_commands_are_not_bare_lexical_queries(self):
        for message in (
            "天选打工人先不要，其他可以加，沙县小吃也不要",
            "火星词先不要，其他都加",
            "大端先不加，小端也不要，其他加入",
            "把大端加入草稿，小端也加入草稿",
            "大端这条删除，小端保留",
            "大端先不要，小端也不要",
            "批量查询 大端 小端",
        ):
            with self.subTest(message=message):
                self.assertEqual(s54_multiword_commands._bare_multi_word_query_words(message), ())
        self.assertEqual(s54_multiword_commands._bare_multi_word_query_words("加班 加工 加法"), ("加班", "加工", "加法"))

    def test_main_chat_stage_uses_same_record_executor_and_review_seals(self):
        async def run():
            _response, store, key = await self.discover()
            result = s54_multiword_commands.DraftActionResult("batch submitted", success=True)
            execute = AsyncMock(return_value=result)
            context = SimpleNamespace(
                response=None, generic_intent_is_fresh_command=False, conv_key=key,
                normalized_message_text="加入并提交", platform="qq", user_id="s54-offline",
                history=[], space_key=key.space_key, owner_label="S54",
            )
            with patch.object(s54_multiword_chat, "conversation_state_store", store), patch.object(
                s54_multiword_chat, "_perform_batch_add_to_draft_and_submit", execute,
            ):
                await s54_multiword_chat._stage_execute_pending_state(context)
            self.assertEqual(context.response, "batch submitted")
            self.assertEqual(execute.await_count, 1)
            self.assertEqual(set(execute.await_args.kwargs["reviewed_capabilities"]), {("大端", "dsdtvo"), ("小端", "xcdtvo")})
            self.assertIsNone(store.get_record(key))
        asyncio.run(run())

    def test_completed_batch_receipt_does_not_regenerate_add_suggestions(self):
        async def run():
            receipt = (
                "✅ 批次已提交审核。\n批次地址：http://localhost:3100/batch/s54\n\n"
                "- 「大端」→ dsdtvo\n- 「小端」→ xcdti"
            )
            key = harness.ConversationAddress.private("qq", "s54-receipt")
            ctx = SimpleNamespace(response=receipt, conv_key=key)
            with patch.object(s54_multiword_chat, "conversation_state_store", harness.MemoryConversationStateStore()):
                await s54_multiword_chat._stage_normalize_response(ctx)
                self.assertEqual(ctx.response, receipt)
                self.assertEqual(s54_multiword_chat._enforce_advertised_reply_contract(ctx.response, key), receipt)
        asyncio.run(run())

    def test_main_stage_warning_replay_keeps_saved_reading_seals(self):
        async def run():
            _response, store, key = await self.discover()
            original = store.get(key)
            capabilities = s54_multiword_commands._reviewed_multi_word_capabilities(original)
            warning = server_warning_pending_state(harness.PendingToolConfirm(
                function_name="keytao_batch_add_to_draft",
                args={"items": original.args["items"], "_submit_after": True,
                      "_reviewed_batch_readings": [
                          {"word": word, "pinyin": value["pinyin"], "candidate_codes": list(value["candidate_codes"])}
                          for (word, _code), value in capabilities.items()
                      ]},
            ), {"requiresConfirmation": True, "batchId": "s54-warning", "contentVersion": 1,
                "warningDigest": "a" * 64, "warnings": ["new risk"]})
            store.set(key, warning)
            execute = AsyncMock(return_value=s54_multiword_commands.DraftActionResult("submitted", success=True))
            context = SimpleNamespace(
                response=None, generic_intent_is_fresh_command=False, conv_key=key,
                normalized_message_text="加入并提交", platform="qq", user_id="s54-offline",
                history=[], space_key=key.space_key, owner_label="S54",
            )
            with patch.object(s54_multiword_chat, "conversation_state_store", store), patch.object(
                s54_multiword_chat, "_perform_batch_add_to_draft_and_submit", execute,
            ):
                await s54_multiword_chat._stage_execute_pending_state(context)
            self.assertEqual(context.response, "词：大端、小端\nsubmitted")
            self.assertEqual(execute.await_args.kwargs["reviewed_capabilities"], capabilities)
            self.assertTrue(execute.await_args.kwargs["confirmed_add"])
        asyncio.run(run())

    def test_new_server_warning_replaces_selector_and_cannot_change_selection(self):
        async def run():
            original, store, key = await self.discover()
            warning = server_warning_pending_state(store.get(key), {
                "requiresConfirmation": True, "batchId": "s54-warning",
                "contentVersion": 1, "warningDigest": "a" * 64,
                "warnings": ["a newly observed occupied slot"],
            })
            store.set(key, warning)
            prompt = s54_multiword_chat._chat_render._format_server_bound_confirmation_prompt(warning)
            with patch.object(s54_multiword_chat, "conversation_state_store", store):
                delivered = s54_multiword_chat._enforce_advertised_reply_contract(prompt, key)
                self.assertEqual(delivered, prompt)
                self.assertIn("a newly observed occupied slot", delivered)
                self.assertNotIn("大端 3", delivered)
                selected, intent, refusal = harness.chat_routing_module._resolve_multi_word_pending_candidate_selection(warning, "大端 2")
                self.assertIsNone(selected)
                self.assertIsNone(intent)
                self.assertIn("风险确认", refusal)
                self.assertFalse(s54_multiword_chat._advertised_reply_matches_live_record(original, store.get_record(key)))
        asyncio.run(run())

    def test_same_codes_with_wrong_words_redraw_from_record(self):
        async def run():
            response, store, key = await self.discover()
            forged = "\n".join(line for line in response.replace("大端", "错误词甲").replace("小端", "错误词乙").splitlines() if not line.startswith("- "))
            with patch.object(s54_multiword_chat, "conversation_state_store", store):
                delivered = s54_multiword_chat._enforce_advertised_reply_contract(forged, key)
            self.assertEqual(delivered, response)
        asyncio.run(run())

# Source: test_s54_selection.py
class S54SelectionTests(unittest.TestCase):
    reordered_record = S54SelectionTests_reordered_record

    def test_two_reorders_and_free_word_share_one_reviewed_shift_plan(self):
        plan = b17_rest_recommended_commands._pending_batch_front_insert_plan(self.reordered_record())
        self.assertFalse(plan.get("invalid"))
        self.assertEqual([(i["word"], i["code"]) for i in plan["additionalItems"]], [("肌群", "jkqtvo")])
        self.assertEqual([(i["word"], i["code"]) for i in plan["additionalShiftItems"]],
                         [("小端", "xcdt")])
        self.assertEqual(plan["additionalItems"][0]["reviewedPinyin"], "jī qún")
        self.assertEqual(plan["expectedOccupantsByCode"], {"dsdt": ["打断"], "xcdt": ["小段"]})
        self.assertEqual(plan["expectedShiftedWords"], ("打断", "小段"))

    def test_server_warning_never_rebuilds_recommendation_plan(self):
        record = self.reordered_record()
        record.confirmation_source = "server_warning"
        self.assertIsNone(b17_rest_recommended_commands._pending_batch_front_insert_plan(record))

    def test_ordinary_same_code_recommendation_is_not_silently_reordered(self):
        record = self.reordered_record()
        scope = record.args["_candidate_scopes"][2]
        code = record.args["items"][2]["code"]
        scope["candidates"] = [(candidate, occupied or candidate == code) for candidate, occupied in scope["candidates"]]
        scope["occupiedWords"][code] = ["肌肉"]
        scope["reviewedState"]["candidates"] = scope["candidates"]
        scope["reviewedState"]["serverCandidates"] = scope["candidates"]
        scope["reviewedState"]["serverOccupiedWords"] = scope["occupiedWords"]
        plan = b17_rest_recommended_commands._pending_batch_front_insert_plan(record)
        self.assertFalse(plan.get("invalid"))
        self.assertEqual(plan["additionalItems"][0]["word"], "肌群")
        self.assertNotIn(code, plan["expectedOccupantsByCode"])
        async def run():
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s54-preserve")
            store.set(key, record)
            execute = AsyncMock(return_value="mixed plan")
            with patch.object(harness.openai_chat_module, "conversation_state_store", store), patch.object(
                b17_rest_recommended_commands, "_execute_confirmed_tool", execute,
            ):
                response = await b17_rest_recommended_commands.handle_pending_message_core("加入", "qq", "s54-preserve", key, allow_intent_model=False)
            self.assertEqual(response, "mixed plan")
            self.assertEqual(execute.await_args.args[0].args["additional_items"][0]["word"], "肌群")
        asyncio.run(run())

    def test_reorder_dispatch_seals_occupants_and_limits_automatic_replay(self):
        record = self.reordered_record()
        async def run():
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s54-shift")
            store.set(key, record)
            execute = AsyncMock(return_value="planned")
            with patch.object(harness.openai_chat_module, "conversation_state_store", store), patch.object(
                b17_rest_recommended_commands, "_execute_confirmed_tool", execute,
            ):
                await b17_rest_recommended_commands.handle_pending_message_core("加入并提交", "qq", "s54-shift", key, allow_intent_model=False)
            sent = execute.await_args.args[0]
            self.assertEqual(sent.args["_expected_occupants_by_code"], {"dsdt": ["打断"], "xcdt": ["小段"]})
            self.assertEqual(sent.args["_auto_confirm_expected_shifted_words"], ["打断", "小段"])
            self.assertEqual(sent.args["additional_items"][0]["reviewedPinyin"], "jī qún")
            self.assertTrue(execute.await_args.kwargs["auto_confirm_shift_plan"])
        asyncio.run(run())

    def test_unknown_batch_warning_retains_reviewed_readings_for_exact_replay(self):
        record = self.reordered_record()
        capabilities = b17_rest_recommended_commands._reviewed_multi_word_capabilities(record)
        async def run():
            preview = AsyncMock(return_value=json.dumps({
                "success": False, "requiresConfirmation": True,
                "batchId": "s54-local-warning", "contentVersion": 7,
                "warningDigest": "e" * 64, "warnings": [{"type": "unknown", "message": "Review new risk"}],
            }))
            with patch.object(harness.openai_chat_module, "call_tool_function", preview):
                result = await b17_rest_recommended_commands._perform_batch_add_to_draft_and_submit(
                    record.args["items"], "qq", "s54-warning", auto_confirm=True,
                    reviewed_capabilities=capabilities,
                )
            pending = result.pending_state
            self.assertIsNotNone(pending)
            self.assertEqual(pending.confirmation_source, "server_warning")
            self.assertEqual(b17_rest_recommended_commands._reviewed_multi_word_capabilities(pending), capabilities)
            self.assertEqual(preview.await_count, 1)
            replay = AsyncMock(return_value=json.dumps({"success": False, "message": "Replay probe"}))
            with patch.object(harness.openai_chat_module, "call_tool_function", replay):
                await b17_rest_recommended_commands._execute_confirmed_tool(pending, "qq", "s54-warning")
            self.assertEqual(replay.await_args.kwargs["trusted_reviewed_items_by_key"], capabilities)
            self.assertNotIn("_reviewed_batch_readings", replay.await_args.args[1])
            self.assertEqual(replay.await_args.args[1]["expected_content_version"], 7)
        asyncio.run(run())

    def test_real_mixed_shift_sink_preserves_readings_and_rejects_drift(self):
        async def run(drift=False):
            draft = harness._draft_tools
            rows = {"dsdt": [{"word": "打断", "code": "dsdt", "type": "Phrase"}],
                    "xcdt": [{"word": "小段", "code": "xcdt", "type": "Phrase"}],
                    "jkqt": [{"word": "肌肉", "code": "jkqt", "type": "Phrase"}]}
            if drift:
                rows["dsdt"] = []
            async def encode(word, _code=None):
                return {"success": True, "word": word, "candidateCodes": {
                    "打断": ["dsdt", "dsdti"], "小段": ["xcdt", "xcdtv"], "肌肉": ["jkqt", "jkqto"],
                }[word]}
            async def words(targets):
                return {"success": True, "results": [{"word": word, "phrases": []} for word in targets]}
            async def codes(targets):
                return {"success": True, "results": [{"code": code, "phrases": rows.get(code, [])} for code in targets]}
            async def strict_preview(_platform, _user, items, **_kwargs):
                valid, failed = await draft._split_items_by_code_validation(items)
                self.assertEqual(failed, [])
                self.assertEqual(len(valid), len(items))
                return {"success": False, "requiresConfirmation": True, "warningDigest": "d" * 64, "warnings": []}
            strict = AsyncMock(side_effect=strict_preview)
            companion = {"word": "肌群", "code": "jkqt", "_reviewed_pinyin": "jī qún", "_reviewed_candidate_codes": ["jkqt", "jkqti"]}
            with patch.object(draft, "_fetch_encode_candidates", side_effect=encode), patch.object(
                draft, "_lookup_words_raw", side_effect=words,
            ), patch.object(draft, "_lookup_codes_raw", side_effect=codes), patch.object(
                draft, "keytao_list_draft_items", AsyncMock(return_value={"success": True, "batchId": "", "contentVersion": 0, "items": []}),
            ), patch.object(draft, "_keytao_strict_batch_add_to_draft", strict):
                result = await draft.keytao_shift_phrase_code(
                    "qq", "s54-offline", "大端", "dsdt", target_needs_manual_review=True,
                    _reviewed_pinyin="dà duān", _reviewed_candidate_codes=["dsdt", "dsdtvo"],
                    additional_shift_items=[
                        {"word": "小端", "code": "xcdt", "_reviewed_pinyin": "xiǎo duān", "_reviewed_candidate_codes": ["xcdt", "xcdti"]},
                    ], additional_items=[companion],
                    _expected_occupants_by_code={"dsdt": ["打断"], "xcdt": ["小段"]},
                )
            return result, strict.await_count
        accepted, calls = asyncio.run(asyncio.wait_for(run(), timeout=2))
        self.assertTrue(accepted.get("requiresConfirmation"), accepted)
        self.assertEqual(calls, 1)
        self.assertEqual({item["word"] for item in accepted["shiftPlan"]["shifted"]}, {"打断", "小段"})
        self.assertIn(("Create", "肌群", "jkqt"), {(i["action"], i["word"], i["code"]) for i in accepted["shiftPlan"]["items"]})
        rejected, calls = asyncio.run(asyncio.wait_for(run(drift=True), timeout=2))
        self.assertIn("占用者已变化", rejected["message"])
        self.assertEqual(calls, 0)

    def test_mixed_dispatch_binds_reviewed_readings_at_real_harness_boundary(self):
        plan = b17_rest_recommended_commands._pending_batch_front_insert_plan(self.reordered_record())
        captured = []
        executor = ToolExecutor.__new__(ToolExecutor)
        async def call_tool(name, args, _platform, _user, **kwargs):
            context = ToolContext(trusted_reviewed_items_by_key=kwargs.get("trusted_reviewed_items_by_key"))
            captured.append(executor._with_trusted_mutation_fields(name, args, context))
            return json.dumps({"success": False, "message": "Offline boundary probe"})
        async def run():
            with patch.object(harness.openai_chat_module, "call_tool_function", side_effect=call_tool):
                capability = plan["reviewedCapability"]
                await b17_rest_recommended_commands._execute_shift_to_code(
                    "大端", "dsdt", "qq", "s54-boundary",
                    additional_items=plan["additionalItems"],
                    additional_shift_items=plan["additionalShiftItems"],
                    reviewed_pinyin=capability["pinyin"],
                    reviewed_candidate_codes=tuple(capability["candidate_codes"]),
                    expected_occupants_by_code=plan["expectedOccupantsByCode"],
                )
        asyncio.run(asyncio.wait_for(run(), timeout=2))
        self.assertEqual(len(captured), 1)
        args = captured[0]
        self.assertEqual(args["_reviewed_pinyin"], "dà duān")
        self.assertEqual(args["additional_shift_items"][0]["_reviewed_pinyin"], "xiǎo duān")
        self.assertEqual(args["additional_items"][0]["_reviewed_pinyin"], "jī qún")
        for item in args["additional_shift_items"] + args["additional_items"]:
            self.assertNotIn("reviewedPinyin", item)
            self.assertNotIn("reviewedCandidateCodes", item)
            self.assertIn(item["code"], item["_reviewed_candidate_codes"])
        forged = {"word": "未知", "code": "evil", "_reviewed_pinyin": "forged",
                  "_reviewed_candidate_codes": ["evil"], "reviewedPinyin": "forged",
                  "reviewedCandidateCodes": ["evil"]}
        sanitized = executor._with_trusted_mutation_fields(
            "keytao_shift_phrase_code", {"word": "大端", "target_code": "dsdt", "additional_items": [forged]},
            ToolContext(),
        )["additional_items"]
        self.assertEqual(sanitized, [{"word": "未知", "code": "evil"}])

    def test_shift_unknown_warning_never_automatically_replays(self):
        async def run(warning):
            store = harness.MemoryConversationStateStore()
            replies = [json.dumps({
                "requiresConfirmation": True, "confirmationKind": "shiftPlan", "batchId": "s54-risk",
                "contentVersion": 1, "planDigest": "a" * 64, "warningDigest": "b" * 64,
                "warnings": [warning], "shiftPlan": {"shifted": [], "items": []},
            }), json.dumps({"success": False, "message": "replay probe"})]
            call = AsyncMock(side_effect=replies)
            with patch.object(harness.openai_chat_module, "conversation_state_store", store), patch.object(
                harness.openai_chat_module, "call_tool_function", call,
            ):
                await b17_rest_recommended_commands._execute_shift_to_code("大端", "dsdt", "qq", "s54-risk", auto_confirm_shift_plan=True)
            return call.await_count
        for warning in ({"warningType": "unknown", "word": "大端", "code": "dsdt"},
                        {"warningType": "duplicate_code", "word": "未授权", "code": "dsdt"}):
            self.assertEqual(asyncio.run(run(warning)), 1)
        self.assertEqual(asyncio.run(run({"warningType": "duplicate_code", "word": "大端", "code": "dsdt"})), 2)

    def test_real_parser_recognizes_only_complete_unquoted_pairs(self):
        self.assertEqual(grammar.parse_reviewed_multi_word_selection("大端 3，小端 xcdti"),
                         (("大端", "3"), ("小端", "xcdti")))
        self.assertIsNone(grammar.parse_reviewed_multi_word_selection("大端 3，大端 3"))
        self.assertIsNone(grammar.parse_reviewed_multi_word_selection("「大端 3」"))

    def test_canonical_pairs_bind_exact_selected_set(self):
        record = reviewed_record()
        original = copy.deepcopy(record)
        for text in ("大端 3，小端 2", "大端 dsdtvo、小端 xcdti", "- 「大端 3，小端 2」"):
            with self.subTest(text=text):
                derived, intent, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(record, text)
                self.assertIsNone(error)
                self.assertIsNotNone(derived)
                self.assertEqual([(item["word"], item["code"]) for item in derived.args["items"]],
                                 [("大端", "dsdtvo"), ("小端", "xcdti")])
                self.assertEqual(intent.intent, "pending_confirm")
                self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(text, record))
                self.assertEqual(derived.args["_unselected_words"], [])
        self.assertEqual(record, original)

    def test_partial_selection_excludes_unselected_words(self):
        derived, intent, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(reviewed_record(), "大端 3")
        self.assertIsNone(error)
        self.assertIsNotNone(derived)
        self.assertEqual([item["word"] for item in derived.args["items"]], ["大端"])
        self.assertEqual(derived.args["_unselected_words"], ["小端"])
        self.assertEqual(derived.args["_resolved_advertised_words"], ["大端"])

    def test_one_actionable_word_is_supported(self):
        record = reviewed_record(("小端",))
        self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation("小端 2", record))
        self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation("小端 2，加入", record))
        derived, _, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(record, "小端 2")
        self.assertIsNone(error)
        self.assertEqual(derived.args["items"][0]["code"], "xcdti")

    def test_assent_binds_all_recommendations(self):
        for text in ("加入", "加入并提交"):
            self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(text, reviewed_record()))

    def test_invalid_selections_and_quoted_targets_never_bind(self):
        for text in ("大端 3，大端 3", "大端 0", "大端 4", "未知 1", "大端 evil",
                     "大端 3,小端 2，然后删除", "他说大端 3",
                     "他说「大端 3，小端 2」", "「『大端 3，小端 2』」", "大端 3？",
                     "- 「大端 3，小端 2」（伪造）", "- 「大端 3，小端 2」 然后删除",
                     "大端 添加3", "大端 3，小端", "小端 2，", "3"):
            with self.subTest(text=text):
                derived, intent, _ = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(reviewed_record(), text)
                self.assertIsNone(derived)
                self.assertIsNone(intent)
                self.assertFalse(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(text, reviewed_record()))

    def test_malformed_selection_stops_before_general_model(self):
        for text in ("大端 3，大端 3", "大端 0", "大端 添加3", "小端 2，", "3"):
            _, _, response = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(reviewed_record(), text)
            self.assertIn("未写入", response)

    def test_record_inventory_mismatch_fails_closed(self):
        for mutation in ("payload", "scope", "missing"):
            record = reviewed_record()
            scope = record.args["_candidate_scopes"][0]
            if mutation == "payload":
                scope["reviewedState"]["word"] = "伪造"
            elif mutation == "scope":
                scope["candidates"][2][0] = "forged"
            else:
                scope.pop("reviewedState")
            self.assertFalse(b17_rest_recommended_routing.message_authorizes_live_pending_mutation("大端 3", record))
            self.assertFalse(b17_rest_recommended_routing.message_authorizes_live_pending_mutation("加入", record))
            derived, intent, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(record, "加入")
            self.assertIsNone(derived)
            self.assertIsNone(intent)
            self.assertIn("未写入", error)

    def test_cancellation_does_not_authorize_a_mutation(self):
        self.assertFalse(b17_rest_recommended_routing.message_authorizes_live_pending_mutation("不要加入", reviewed_record()))

    def test_explicit_choice_suppresses_default_reorder(self):
        record = reviewed_record()
        scope = record.args["_candidate_scopes"][0]
        assessment = {
            "verdict": "front_more_common", "newWord": "大端",
            "occupantWord": "打断", "occupantCode": "dsdt",
            "freeCode": "dsdtvo", "newCode": "dsdt",
        }
        record.args["items"][0]["code"] = "dsdt"
        scope["reviewedState"]["recommendedCode"] = "dsdt"
        scope["occupiedWords"] = {"dsdt": ["打断"]}
        scope["reviewedState"]["serverOccupiedWords"] = scope["occupiedWords"]
        scope["orderingAssessments"] = [assessment]
        scope["reviewedState"]["serverOrderingAssessments"] = [assessment]
        self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation("加入", record))
        self.assertIsNotNone(b17_rest_recommended_commands._pending_batch_front_insert_plan(record))
        derived, _, _ = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(record, "大端 3")
        self.assertIsNotNone(derived)
        self.assertIsNone(b17_rest_recommended_commands._pending_batch_front_insert_plan(derived))
        self.assertEqual(derived.args["_candidate_scopes"][0]["orderingAssessments"], [])
        self.assertEqual(record.args["_candidate_scopes"][0]["orderingAssessments"], [assessment])

    def test_legacy_scoped_selection_still_keeps_other_words(self):
        record = reviewed_record()
        record.args.pop("_reviewed_multi_word")
        derived, _, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(record, "大端 添加3")
        self.assertIsNone(error)
        self.assertEqual([item["word"] for item in derived.args["items"]], ["大端", "小端"])

    def test_internal_batch_preserves_only_record_bound_reviewed_readings(self):
        items = [
            {"action": "Create", "word": "大端", "code": "dsdtvo", "remark": "Keep review",
             "needsManualReview": True, "manualReviewReason": "Keep reason"},
            {"action": "Create", "word": "小端", "code": "xcdti", "needsManualReview": False},
            {"action": "Create", "word": "未知", "code": "evil", "_reviewed_pinyin": "forged",
             "_reviewed_candidate_codes": ["evil"], "reviewedPinyin": "forged",
             "reviewedCandidateCodes": ["evil"]},
        ]
        original = copy.deepcopy(items)
        context = ToolContext(trusted_reviewed_items_by_key={
            ("大端", "dsdtvo"): {"pinyin": "dà duān", "candidate_codes": ["dsdt", "dsdtv", "dsdtvo"]},
            ("小端", "xcdti"): {"pinyin": "xiǎo duān", "candidate_codes": ["xcdt", "xcdti", "xcdtio"]},
        })
        result = ToolExecutor.__new__(ToolExecutor)._with_trusted_mutation_fields(
            "keytao_batch_add_to_draft", {"items": items}, context,
        )["items"]
        self.assertEqual(result[0]["_reviewed_pinyin"], "dà duān")
        self.assertEqual(result[1]["_reviewed_pinyin"], "xiǎo duān")
        self.assertEqual(result[0]["_reviewed_candidate_codes"], ["dsdt", "dsdtv", "dsdtvo"])
        self.assertEqual(result[0]["remark"], "Keep review")
        self.assertEqual(result[0]["manualReviewReason"], "Keep reason")
        self.assertTrue(result[0]["needsManualReview"])
        self.assertFalse(result[1]["needsManualReview"])
        self.assertFalse(any(key in result[2] for key in (
            "_reviewed_pinyin", "_reviewed_candidate_codes", "reviewedPinyin", "reviewedCandidateCodes",
        )))
        self.assertEqual(items, original)

# Source: test_s55_single_pipeline.py
class SinglePipelineTests(unittest.TestCase):
    def test_one_candidate_keeps_review_record_and_bound_actions_after_reload(self):
        async def run(db_path):
            store = SQLiteConversationStateStore(db_path=db_path)
            key = harness.ConversationAddress.private("qq", "s55-single")

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(single_review())
                raise AssertionError(name)

            with patch.object(s55_single_pipeline_chat, "conversation_state_store", store), patch.object(
                s55_single_pipeline_chat, "call_tool_function", side_effect=tool,
            ), patch.object(
                s55_single_pipeline_chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(True, ("鎗",), "word_lookup", 1.0)),
            ), patch.object(s55_single_pipeline_commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)):
                reply = await s55_single_pipeline_commands._try_handle_simple_single_word_query("鎗", "qq", "s55-single", key)
                for expected in ("单字", "qiāng，需管理员审核", "1. qx", "加入", "加入并提交"):
                    self.assertIn(expected, reply)
                reloaded = SQLiteConversationStateStore(db_path=db_path)
                record = reloaded.get_record(key)
                self.assertEqual(record.state.phrase_type, "Single")
                self.assertEqual(record.state.server_candidates, [("qx", False)])
                self.assertIn("异体字", record.state.code_remarks["qx"])
                self.assertTrue(s55_single_pipeline_chat._advertised_reply_matches_live_record(reply, record))
                for action in ("1", "qx"):
                    self.assertTrue(s55_single_pipeline_chat._chat_routing.message_authorizes_live_pending_mutation(action, record.state), action)
                for action, intent_name in (("加入", "pending_confirm"), ("加入并提交", "pending_add_and_submit")):
                    assent = s55_single_pipeline_chat._chat_routing._pending_assent_phrase_for_state(record.state, action)
                    self.assertTrue(assent.matched and assent.add_requested, action)
                    intent = await s55_single_pipeline_chat._classify_message_command_intent(action, record.state)
                    self.assertEqual(intent.intent, intent_name)
                args = s55_single_pipeline_commands._create_phrase_args(record.state, "qx")
                self.assertEqual(args["type"], "Single")
                capability = s55_single_pipeline_commands._reviewed_create_capability("鎗", "qx", "qiāng", ("qx",))
                self.assertEqual(capability[("鎗", "qx")]["type"], "Single")
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory + "/state.db"))

# Source: test_s56_e2e_closure.py
class S56AdvertisedClosureTests(unittest.TestCase):
    def test_guarded_single_commands_and_complete_bullets_bind_after_reload(self):
        async def run(db_path):
            chat = harness.openai_chat_module
            commands = harness.chat_commands_module
            store = SQLiteConversationStateStore(db_path=db_path)
            key = harness.ConversationAddress.private("qq", "s56-closure")
            reviewed = single_review()
            reviewed["recommendedCode"] = ""
            reviewed["pronunciations"][0]["recommendedCode"] = ""
            reviewed["pronunciations"][0]["candidateStatuses"] = [{
                "code": "qx", "occupied": True, "words": ["强"],
                "phrases": [{"word": "强", "code": "qx", "type": "Single", "weight": 10}],
            }]
            reviewed["candidateOrderingAssessments"] = [{
                "newWord": "鎗", "occupantWord": "强", "occupantCode": "qx",
                "freeCode": "", "newCode": "", "recommendedCode": "",
                "verdict": "behind_more_common", "decisionReason": "frequency_ratio",
                "summary": "fixture evidence: 10 vs 17342",
            }]

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(reviewed)
                raise AssertionError(name)

            with patch.object(chat, "conversation_state_store", store), patch.object(
                chat, "call_tool_function", side_effect=tool,
            ), patch.object(
                chat, "_classify_simple_word_query_intent",
                AsyncMock(return_value=harness.SimpleWordQueryIntent(True, ("鎗",), "word_lookup", 1.0)),
            ), patch.object(commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)):
                reply = await commands._try_handle_simple_single_word_query("鎗", "qq", "s56-closure", key)
                record = SQLiteConversationStateStore(db_path=db_path).get_record(key)
                self.assertIsNotNone(record, reply)
                self.assertEqual(record.state.phrase_type, "Single")
                self.assertEqual(record.state.server_candidates, [("qx", True)])
                contract = advertised_reply_contract(reply)
                self.assertEqual(contract.batch_assent_forms, ())
                self.assertEqual(contract.command_suggestions, ("添加 鎗 qx,顶替 强",))
                self.assertTrue(chat._advertised_reply_matches_live_record(reply, record), reply)
                for command in contract.command_suggestions:
                    parsed = parse_eviction_modified_add(command)
                    self.assertIsNotNone(parsed, command)
                    self.assertEqual((parsed.word, parsed.code, parsed.named_occupant), ("鎗", "qx", "强"))
                    self.assertTrue(chat._chat_routing.message_authorizes_live_pending_mutation(command, record.state), command)
                    envelope = next(line for line in reply.splitlines() if line.lstrip().startswith("- ") and command in unicodedata.normalize("NFKC", line))
                    self.assertTrue(chat._chat_routing.message_authorizes_live_pending_mutation(envelope, record.state), envelope)
                self.assertNotIn("推荐编码：qx", reply)
                self.assertIn("加入编码 qx+形码", reply)

        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory + "/state.db"))

# Source: test_s57_claiming.py
class OfferedAnswerClaimingTests(unittest.TestCase):
    def test_contextual_extra_code_survives_scope_and_skips_intent_model(self):
        async def run():
            for context_only in (False, True):
                for message, retained in (("加入编码yeoiau", True), ("加入编码12", True), ("春暖花开", False)):
                    key = harness.ConversationAddress.private("qq", "s57-context-scope")
                    state = harness.chat_commands_module.create_pending_trusted_word_record(
                        "嘢", "yeoiav", "Single", context_only=context_only,
                    )
                    s57_claiming_chat.conversation_state_store.set(key, state)
                    ctx = s57_claiming_chat.TurnContext(
                        bot=object(), event=object(), platform="qq", user_id=key.actor_id,
                        normalized_message_text=message, conv_key=key, space_key=key.space_key,
                        command_intent_for=AsyncMock(side_effect=AssertionError("generic intent called")),
                    )
                    await s57_claiming_chat._stage_resolve_current_pending_scope(ctx)
                    self.assertEqual(ctx.current_pending_record is not None, retained, message)
                    if retained:
                        with patch.object(s57_claiming_chat, "AsyncOpenAI", side_effect=AssertionError("intent model called")):
                            intent = await s57_claiming_chat._classify_message_command_intent(message, state)
                            await s57_claiming_chat._stage_apply_scoped_pending_intent(ctx)
                        self.assertEqual(intent.intent, "none")
                        self.assertEqual(ctx.generic_command_intent.intent, "none")
                        execute = AsyncMock(return_value="bound contextual request")
                        with patch.object(s57_claiming_chat, "_resolve_pending_trusted_word_action", execute):
                            await s57_claiming_chat._stage_execute_pending_state(ctx)
                        self.assertEqual(execute.await_count, 1)
                        self.assertEqual(execute.call_args.args[1:5], (message, "qq", key.actor_id, key))
                    s57_claiming_chat.conversation_state_store.delete(key)
        asyncio.run(run())

    def test_force_phrases_are_closed_assents(self):
        from keytao_bot.utils.offered_options import FORCE_ASSENT_TEXTS

        state = harness.PendingToolConfirm("keytao_shift_phrase_code", {})
        for text in FORCE_ASSENT_TEXTS:
            with self.subTest(text=text):
                self.assertTrue(s57_claiming_chat._chat_routing._pending_assent_phrase_for_state(state, text).matched)
        for text in ('不要强制', '强制吗？', '他说硬来', '"硬加"', '强制添加其他词'):
            with self.subTest(text=text):
                self.assertFalse(s57_claiming_chat._chat_routing._pending_assent_phrase_for_state(state, text).matched)

    def test_recordless_option_is_claimed_but_genuine_word_is_not(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-claim")
            history = [{"role": "assistant", "content": "请问是要强制调序，还是保留现顺序？"}]
            for message, previous, claimed in (
                ("保留现顺序", history, True),
                ("强制调序", history, True),
                ("硬来", [], True),
                ("春暖花开", [], False),
                ("保留现顺序", [{"role": "assistant", "content": "操作已完成。"}], False),
            ):
                ctx = SimpleNamespace(
                    conv_key=key, normalized_message_text=message, memory_context=None,
                    bot=None, event=None, user_id="s57-claim", QQMessageSegment=None,
                )
                with patch.object(s57_claiming_chat, "get_history", return_value=previous), patch.object(
                    s57_claiming_chat, "remember_conversation",
                ), patch.object(s57_claiming_chat, "_finish_ai_chat_response", new=AsyncMock()) as finish:
                    self.assertEqual(await s57_claiming_chat._stage_claim_offered_answer(ctx), claimed, message)
                    self.assertEqual(finish.await_count, int(claimed), message)
        asyncio.run(run())

# Source: test_s57_options.py
class S57OptionBoundaryTests(unittest.TestCase):
    def test_real_orchestrator_refuses_unbacked_write_option_question(self):
        async def run():
            raw = "请问是要强制调序，还是保留现顺序？"
            client = harness._FakeClient([harness._FakeAIResponse("stop", raw)])
            orchestrator = harness.AgentOrchestrator(
                client_factory=lambda: client,
                runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10.0),
                skills_manager=harness._FakeSkillsManager(),
                tool_executor=harness.ToolExecutor(lambda _name: None, frozenset()),
                state_store=harness.MemoryConversationStateStore(),
                bind_help_text="bind help", system_prompt_core="system",
            )
            reply = await orchestrator.run(
                "把 嘢 排到 咽 前面",
                harness.AgentRequestContext(platform="qq", user_id="s57-options", mutations_allowed=True),
            )
            self.assertIn("未写入", reply)
            self.assertNotIn("还是", reply)
            self.assertEqual(len(client.completions.calls), 1)
        asyncio.run(run())

    def test_no_ticket_option_question_is_refused_at_delivery(self):
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.group("qq", "s57-options", "s57-owner")
        raw = "本次未调整。请问是要强制调序，还是保留现顺序？"
        with patch.object(s57_options_chat, "conversation_state_store", store):
            actual = s57_options_chat._enforce_advertised_reply_contract(raw, key)
        self.assertNotEqual(raw, actual)
        self.assertNotIn("还是", actual)
        self.assertIn("未写入", actual)

    def test_bound_option_question_round_trips_every_choice(self):
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.group("qq", "s57-options", "s57-owner")
        state = ticket()
        store.set(key, state)
        raw = "请问是要强制调序，还是保留现顺序？"
        for option, expected in state.args["_offered_options"].items():
            with self.subTest(option=option):
                parsed = s57_options_routing._pending_tool_assent_intent(state, option)
                self.assertEqual(f"pending_{expected}", parsed.intent)
                self.assertTrue(s57_options_routing._message_authorizes_pending_state_control(state, option, parsed))
        self.assertTrue(option_questions_bind_live_state(raw, state))
        with patch.object(s57_options_chat, "conversation_state_store", store):
            self.assertEqual(raw, s57_options_chat._enforce_advertised_reply_contract(raw, key))

    def test_lexicon_is_exact_and_cannot_bind_incomplete_or_foreign_ticket(self):
        raw = "强制调序还是保留现顺序？"
        malformed = ticket()
        malformed.args.pop("confirmed_plan_digest")
        extra_alias = ticket({"强制调序": "confirm", "保留现顺序": "cancel", "偷偷执行": "confirm"})
        wrong_alias = ticket({"强制": "confirm", "保留现顺序": "cancel"})
        local = ticket()
        local.confirmation_source = "local_preview"
        for state in (None, malformed, local, extra_alias, wrong_alias):
            with self.subTest(state=state):
                self.assertFalse(option_questions_bind_live_state(raw, state))
        store = harness.MemoryConversationStateStore()
        owner = harness.ConversationAddress.group("qq", "s57-group", "owner")
        other = harness.ConversationAddress.group("qq", "s57-group", "other")
        another_group = harness.ConversationAddress.group("qq", "s57-another", "owner")
        store.set(owner, ticket())
        with patch.object(s57_options_chat, "conversation_state_store", store):
            for address in (other, another_group):
                self.assertNotEqual(raw, s57_options_chat._enforce_advertised_reply_contract(raw, address))
            store.get_record(owner).execution_id = "already-claimed"
            self.assertNotEqual(raw, s57_options_chat._enforce_advertised_reply_contract(raw, owner))

    def test_arbitrary_trusted_choices_bind_after_sqlite_reload(self):
        raw = "现在落实，或维持原样？"
        with tempfile.TemporaryDirectory() as directory:
            path = directory + "/pending.db"
            key = harness.ConversationAddress.private("qq", "s57-durable")
            SQLiteConversationStateStore(path).set(key, ticket({"现在落实": "confirm", "维持原样": "cancel"}))
            state = SQLiteConversationStateStore(path).get_record(key).state
            self.assertTrue(option_questions_bind_live_state(raw, state))
            self.assertEqual("confirm", offered_option_intent("现在落实", state))
            self.assertEqual("cancel", offered_option_intent("维持原样", state))

    def test_read_only_question_can_offer_explanatory_topics(self):
        token = s57_options_chat._current_turn_message.set("介绍编码方案")
        try:
            raw = "你想先了解音码，还是形码？"
            self.assertEqual(raw, s57_options_chat._prepare_user_facing_reply(raw, None))
        finally:
            s57_options_chat._current_turn_message.reset(token)

    def test_unrelated_live_ticket_does_not_turn_explanation_into_write_flow(self):
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.private("qq", "s57-read-live")
        store.set(key, ticket())
        token = s57_options_chat._current_turn_message.set("介绍编码方案")
        try:
            raw = "你想先了解音码，还是形码？"
            with patch.object(s57_options_chat, "conversation_state_store", store):
                self.assertEqual(raw, s57_options_chat._enforce_advertised_reply_contract(raw, key))
        finally:
            s57_options_chat._current_turn_message.reset(token)

    def test_missing_ticket_recovery_survives_real_final_delivery(self):
        for history in ([], [{"role": "assistant", "content": "请问是要强制调序，还是保留现顺序？"}],
                        [{"role": "assistant", "content": "将把 嘢 排到 咽 之前：嘢 11→10、咽 10→11"}]):
            token = s57_options_chat._current_turn_message.set("强制调序")
            try:
                raw = render_missing_option_ticket(history)
                actual = s57_options_chat._prepare_user_facing_reply(raw, None)
                self.assertEqual(raw, actual)
                self.assertIn("引用原提议", actual)
                self.assertIn("未写入", actual)
                if history:
                    self.assertIn("上一条提议", actual)
            finally:
                s57_options_chat._current_turn_message.reset(token)

# Source: test_s57_options.py
class S57OptionStructureTests(unittest.TestCase):
    def test_structural_question_forms_without_confirmation_keywords(self):
        for text, expected in (
            ("请问是要强制调序，还是保留现顺序？", ("强制调序", "保留现顺序")),
            ("「现在落实」或「维持原样」?", ("现在落实", "维持原样")),
            ("选择 A. 现在落实或者 B. 维持原样？", ("现在落实", "维持原样")),
            ("是否进行上述调整？", ()),
        ):
            with self.subTest(text=text):
                parsed = structural_option_questions(text)
                self.assertEqual(len(parsed), 1)
                self.assertEqual(parsed[0].options, expected)
        for text in ("现在落实或维持原样。", "这是含有或字的普通说明", "怎样查看草稿？"):
            self.assertFalse(structural_option_questions(text), text)

    def test_force_words_are_closed_controls_not_questions_or_negated_text(self):
        for text in FORCE_ASSENT_TEXTS:
            self.assertTrue(is_force_assent(text), text)
            self.assertTrue(is_force_assent("  " + text + "！"), text)
        for text in ("不要强制", "强制？", "强制吗", "他说强制", "「强制」", "硬加安全吗", "强制加错字"):
            self.assertFalse(is_force_assent(text), text)

    def test_only_immediately_preceding_bot_options_claim_the_answer(self):
        history = [{"role": "assistant", "content": "强制调序还是保留现顺序？"}]
        self.assertEqual(previous_bot_options(history), ("强制调序", "保留现顺序"))
        self.assertTrue(matches_previous_bot_option("「强制 调序」", history))
        self.assertFalse(matches_previous_bot_option("强制调序？", history))
        self.assertFalse(matches_previous_bot_option("强制调序", history + [{"role": "assistant", "content": "好的。"}]))
        self.assertFalse(matches_previous_bot_option("强制调序", history + [{"role": "user", "content": "还在吗"}]))

# Source: test_s57_options.py
class S57OptionClaimTests(unittest.TestCase):
    def test_quoted_sealed_options_keep_priority_over_literal_word_lookup(self):
        async def run():
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "quoted-options")
            store.set(key, ticket({"现在落实": "confirm", "维持原样": "cancel"}))
            for option, expected in (("现在落实", "pending_confirm"), ("维持原样", "pending_cancel")):
                message = "「" + option + "」"
                ctx = s57_options_chat.TurnContext(
                    bot=object(), event=object(), platform="qq", user_id=key.actor_id,
                    conv_key=key, normalized_message_text=message,
                    literal_phrase_word=s57_options_routing.parse_literal_phrase_query(message) or "",
                    command_intent_for=AsyncMock(side_effect=AssertionError("Unexpected classifier")),
                )
                with patch.object(s57_options_chat, "conversation_state_store", store):
                    await s57_options_chat._stage_resolve_current_pending_scope(ctx)
                    await s57_options_chat._stage_apply_scoped_pending_intent(ctx)
                self.assertIsNone(ctx.scoped_pending_response)
                self.assertEqual(ctx.generic_command_intent.intent, expected)
                self.assertFalse(ctx.literal_phrase_word)
        asyncio.run(run())

    def test_live_force_and_exact_options_skip_intent_models_in_both_stages(self):
        async def run():
            state = ticket({"现在落实": "confirm", "维持原样": "cancel"})
            for message in (*FORCE_ASSENT_TEXTS, "现在落实", "维持原样"):
                with self.subTest(message=message):
                    ctx = SimpleNamespace(
                        current_pending_record=SimpleNamespace(state=state),
                        normalized_message_text=message, eviction_modified_add=None,
                        literal_phrase_word="",
                        compound_eviction_add_plan=None, scoped_pending_intent=None,
                        resolved_advertised_words=(), quoted_pending_add_control=False,
                        command_intent_for=AsyncMock(side_effect=AssertionError("generic intent model called")),
                    )
                    self.assertFalse(await s57_options_chat._stage_apply_scoped_pending_intent(ctx))
                    expected = "pending_cancel" if message == "维持原样" else "pending_confirm"
                    self.assertEqual(ctx.generic_command_intent.intent, expected)
                    ctx.command_intent_for.assert_not_awaited()
                    with patch.object(s57_options_routing, "_classify_message_command_intent", AsyncMock(side_effect=AssertionError("pending intent model called"))):
                        _cache, classify = s57_options_routing.command_intent_memoizer(message, message)
                        self.assertEqual((await classify(state)).intent, expected)
        asyncio.run(run())

    def test_no_ticket_offered_answer_is_claimed_before_word_discovery(self):
        async def run():
            key = harness.ConversationAddress.group("qq", "s57-claim", "owner")
            history = [{"role": "assistant", "content": "现在落实还是维持原样？"}]
            ctx = SimpleNamespace(
                conv_key=key, normalized_message_text="现在落实", memory_context=None,
                platform="qq", user_id="owner", bot=None, event=None, QQMessageSegment=None,
            )
            with patch.object(s57_options_chat, "conversation_state_store", harness.MemoryConversationStateStore()), patch.object(
                s57_options_chat, "get_history", return_value=history,
            ), patch.object(s57_options_chat, "remember_conversation"), patch.object(s57_options_chat, "_finish_ai_chat_response", AsyncMock()) as finish:
                self.assertTrue(await s57_options_chat._stage_claim_offered_answer(ctx))
                self.assertIn("未写入", ctx.response)
                self.assertEqual(finish.await_count, 1)
        asyncio.run(run())

    def test_genuine_four_character_query_without_options_remains_unclaimed(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-word")
            ctx = SimpleNamespace(conv_key=key, normalized_message_text="海阔天空")
            with patch.object(s57_options_chat, "get_history", return_value=[{"role": "assistant", "content": "请发要查询的词。"}]):
                self.assertFalse(await s57_options_chat._stage_claim_offered_answer(ctx))
            calls = []

            async def tool(name, args, *_args, **_kwargs):
                calls.append((name, args))
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps({
                        "success": True, "word": "海阔天空", "recommendedCode": "hktk",
                        "needsManualReview": True, "manualReviewReason": "离线核验",
                        "pronunciations": [{"pinyin": "hǎi kuò tiān kōng", "recommendedCode": "hktk",
                            "sources": [{"source": "汉典"}], "candidateStatuses": [
                                {"code": "hktk", "occupied": False, "words": [], "label": "空位"},
                                {"code": "hktkv", "occupied": False, "words": [], "label": "空位"},
                            ]}],
                    })
                raise AssertionError(name)

            store = harness.MemoryConversationStateStore()
            with patch.object(s57_options_chat, "conversation_state_store", store), patch.object(
                s57_options_chat, "call_tool_function", side_effect=tool,
            ), patch.object(s57_options_chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(
                True, ("海阔天空",), "word_lookup", 1.0,
            ))):
                reply = await harness.chat_commands_module._try_handle_simple_single_word_query(
                    "海阔天空", "qq", "s57-word", key,
                )
            self.assertIn("hǎi kuò tiān kōng，需管理员审核", reply)
            self.assertIsInstance(store.get(key), harness.PendingAddWord)
            self.assertIn(("keytao_prepare_reviewed_add", {"word": "海阔天空"}), calls)
        asyncio.run(run())

# Source: test_s62_selection.py
class S62SelectionTests(unittest.IsolatedAsyncioTestCase):
    execute_selection = S62SelectionTests_execute_selection

    async def test_spaced_add_selector_writes_only_the_selected_live_word(self):
        record = s62_selection_incident_record()
        original = copy.deepcopy(record)
        writes, calls, response, pending = await self.execute_selection(
            "加亮 添加 jslxa", record,
        )
        self.assertEqual([(item["word"], item["code"]) for item in writes],
                         [("加亮", "jslxa")])
        self.assertEqual(len(calls), 2)
        self.assertIsNone(pending)
        self.assertIn("小端", response)
        self.assertEqual(record, original)

    async def test_canonical_and_spaced_number_selectors_keep_the_same_write_scope(self):
        for message in ("加亮 jslxa", "加亮 3", "加亮 添加 3",
                        "加亮 jslxa，加入", "加亮 3，加入"):
            with self.subTest(message=message):
                writes, calls, response, pending = await self.execute_selection(message)
                self.assertEqual([(item["word"], item["code"]) for item in writes],
                                 [("加亮", "jslxa")])
                self.assertEqual(len(calls), 2)
                self.assertIsNone(pending)
                self.assertIn("加亮 → jslxa", response)
                self.assertIn("https://keytao.rea.ink/batch/s62-selection-batch", response)
                self.assertNotIn("已提交", response)

    async def test_comma_separated_choices_bind_each_words_own_inventory(self):
        for message in ("加亮 3，小端 xcdti", "加亮 jslxa、小端 2",
                        "加亮 添加 3，小端 xcdti", "加亮 jslxa、小端 添加 2"):
            with self.subTest(message=message):
                writes, _calls, _response, _pending = await self.execute_selection(message)
                self.assertEqual([(item["word"], item["code"]) for item in writes],
                                 [("加亮", "jslxa"), ("小端", "xcdti")])

    async def test_invalid_or_unbound_spaced_choices_never_write(self):
        for message in (
            "加亮 添加3", "加亮 添加jslxa", "加亮 添加 jslxa，加亮 3",
            "不要加亮 添加 jslxa", "他说加亮 添加 jslxa", "他说「加亮 添加 jslxa」",
            "「『加亮 添加 jslxa』」", "加亮 添加 jslxa？",
            "加亮 添加 jslxa，然后删除", "加亮 添加 jslxa；提交", "加亮 添加 3，",
            "未知 添加 1", "加亮 添加 evil", "加亮 添加 xcdti",
            "加亮 添加 0", "加亮 添加 4",
        ):
            with self.subTest(message=message):
                self.assertFalse(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(
                    message, s62_selection_incident_record(),
                ))
                writes, calls, _response, _pending = await self.execute_selection(message)
                self.assertEqual(writes, [])
                self.assertEqual(calls, [])

    async def test_spaced_choices_still_require_the_live_review_seal(self):
        for mutation in ("payload", "scope", "missing", "server_warning"):
            with self.subTest(mutation=mutation):
                record = s62_selection_incident_record()
                scope = record.args["_candidate_scopes"][0]
                if mutation == "payload":
                    scope["reviewedState"]["word"] = "伪造"
                elif mutation == "scope":
                    scope["candidates"][2][0] = "forged"
                elif mutation == "missing":
                    scope.pop("reviewedState")
                else:
                    record.confirmation_source = "server_warning"
                for message in ("加亮 jslxa", "加亮 添加 jslxa"):
                    writes, calls, response, _pending = await self.execute_selection(message, record)
                    self.assertEqual(writes, [])
                    self.assertEqual(calls, [])
                    self.assertIn("未写入", response)

    async def test_occupied_selection_requires_exact_commonness_evidence(self):
        for verdict in ("front_more_common", "behind_more_common", "close", "uncertain", "missing", "wrong_word", "wrong_code", "unknown_occupant"):
            with self.subTest(verdict=verdict):
                record = s62_selection_incident_record()
                scope = record.args["_candidate_scopes"][0]
                occupants = {} if verdict == "unknown_occupant" else {"jslx": ["加量"]}
                assessments = [] if verdict == "missing" else [{
                    "newWord": "别词" if verdict == "wrong_word" else "加亮",
                    "occupantWord": "加量",
                    "occupantCode": "jslxo" if verdict == "wrong_code" else "jslx",
                    "verdict": "front_more_common" if verdict.startswith("wrong_") else verdict,
                }]
                scope["occupiedWords"] = scope["reviewedState"]["serverOccupiedWords"] = occupants
                scope["orderingAssessments"] = scope["reviewedState"]["serverOrderingAssessments"] = assessments
                original = copy.deepcopy(record)
                writes, calls, response, _ = await self.execute_selection("加亮 1", record)
                if verdict == "front_more_common":
                    self.assertEqual([(row["word"], row["code"]) for row in writes], [("加亮", "jslx")])
                    self.assertTrue(writes[0]["needsManualReview"])
                    self.assertIn("管理员审核", writes[0]["manualReviewReason"])
                    self.assertEqual(len(calls), 2)
                else:
                    self.assertEqual(writes, [])
                    self.assertEqual(calls, [])
                    self.assertIn("未加入", response)
                    self.assertIn("占用" if verdict == "unknown_occupant" else "常用度", response)
                    if verdict != "unknown_occupant":
                        self.assertIn("加量", response)
                self.assertEqual(record, original)

    async def test_protected_occupied_selection_does_not_block_other_selected_words(self):
        record = s62_selection_incident_record()
        scope = record.args["_candidate_scopes"][0]
        scope["occupiedWords"] = scope["reviewedState"]["serverOccupiedWords"] = {"jslx": ["加量"]}
        for message in ("加亮 1，小端 2", "小端 xcdti，加亮 jslx"):
            with self.subTest(message=message):
                writes, calls, response, pending = await self.execute_selection(message, record)
                self.assertEqual([(row["word"], row["code"]) for row in writes], [("小端", "xcdti")])
                self.assertEqual(len(calls), 2)
                self.assertIsNone(pending)
                self.assertIn("✅ 已加入草稿：小端 xcdti", response)
                self.assertIn("加量", response)
                self.assertIn("常用度", response)
                self.assertIn("未加入", response)
                self.assertIn("https://keytao.rea.ink/batch/s62-selection-batch", response)
                self.assertNotIn("已提交", response)

    def test_parser_rejects_quoted_or_joined_add_choices(self):
        self.assertEqual(grammar.parse_reviewed_multi_word_selection("加亮 添加 JSLXA"),
                         (("加亮", "jslxa"),))
        for message in ("「加亮 添加 jslxa」", "加亮 添加3", "加亮 添加jslxa"):
            with self.subTest(message=message):
                self.assertIsNone(grammar.parse_reviewed_multi_word_selection(message))

# Source: test_s63_fresh_selection.py
class FreshSelectionTests(unittest.IsolatedAsyncioTestCase):
    setUp = FreshSelectionTests_setUp

    tool = FreshSelectionTests_tool

    runtime = FreshSelectionTests_runtime

    turn = FreshSelectionTests_turn

    async def test_fresh_pair_reads_own_draft_and_returns_its_real_link_without_model(self):
        with self.runtime():
            response = await self.turn("加亮 jslxa")
        self.assertIn("已在当前草稿", response)
        self.assertIn(BATCH_URL, response)
        self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])
        self.assertIsNone(self.store.get(self.key))

    async def test_textual_bot_mention_reaches_the_same_fresh_draft_lookup(self):
        with self.runtime():
            response = await self.turn("@喵喵 加亮 jslxa")
        self.assertIn("已在当前草稿", response)
        self.assertIn(BATCH_URL, response)
        self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])
        self.assertIsNone(self.store.get(self.key))

    async def test_real_ttl_expiry_uses_fresh_pending_facts_without_model(self):
        self.store.set(self.key, s62_selection_incident_record())
        self.now += 11
        with self.runtime():
            response = await self.turn("加亮 jslxa")
        self.assertIn(BATCH_URL, response)
        self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])
        self.assertIsNone(self.store.get(self.key))

    async def test_new_pair_prepares_exact_code_then_real_confirmation_writes_once(self):
        self.pending_items = []
        with self.runtime():
            response = await self.turn("加亮 jslxa")
            self.assertIn("尚未写入", response)
            self.assertIn("1. jslxa — ✅推荐", response)
            self.assertNotIn("jslxao", response)
            self.assertIn("jiā liàng", response)
            self.assertIn("需管理员审核", response)
            state = self.store.get(self.key)
            self.assertIsInstance(state, harness.PendingAddWord)
            self.assertEqual(state.recommended_code, "jslxa")
            self.assertEqual(state.server_candidates, [("jslxa", False)])
            self.assertTrue(state.needs_manual_review)
            self.assertEqual(self.writes, [])
            self.assertEqual([name for name, _ in self.calls], [
                "keytao_pending_items_by_words", "keytao_prepare_reviewed_add", "keytao_lookup_by_code",
            ])
            delivered = s63_fresh_selection_chat._enforce_advertised_reply_contract(response, self.key)
            self.assertIn("jslxa", delivered)
            self.assertNotIn("jslxao", delivered)
            confirmed = await b17_rest_recommended_commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key, allow_intent_model=False,
            )
            self.assertIn("已", confirmed)
            await b17_rest_recommended_commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key, allow_intent_model=False,
            )
        self.assertEqual([(item["word"], item["code"]) for item in self.writes], [("加亮", "jslxa")])
        self.assertEqual(sum(name == "keytao_create_phrase" for name, _ in self.calls), 2)

    async def test_blocked_review_never_creates_a_confirmation_record(self):
        self.pending_items = []
        self.review["reviewDisposition"] = "BLOCK"
        with self.runtime():
            response = await self.turn("加亮 jslxa")
        self.assertIn("未写入", response)
        self.assertIsNone(self.store.get(self.key))
        self.assertEqual([name for name, _ in self.calls], [
            "keytao_pending_items_by_words", "keytao_prepare_reviewed_add",
        ])

    async def test_submitted_same_pair_is_reported_without_review_or_write(self):
        self.pending_items[0].update(source="submitted", batchStatus="Submitted")
        with self.runtime():
            response = await self.turn("加亮 JSLXA")
        self.assertIn("待审核批次", response)
        self.assertIn(BATCH_URL, response)
        self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])

    async def test_unknown_pending_facts_never_become_an_empty_draft(self):
        for complete, items in ((False, []), (True, None), (True, {})):
            with self.subTest(complete=complete, items=items):
                self.setUp()
                self.pending_complete, self.pending_items = complete, items
                with self.runtime():
                    response = await self.turn("加亮 jslxa")
                self.assertIn("无法核验", response)
                self.assertIsNone(self.store.get(self.key))
                self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])

    async def test_invalid_code_or_unreviewed_reading_never_arms_a_ticket(self):
        for code, changes in (
            ("evil", {}), ("jslxaaa", {}),
            ("jslxa", {"pronunciations": []}),
            ("jslxa", {"pronunciationUnresolved": True}),
            ("jslxa", {"word": "加量"}),
            ("jslxa", {"type": "Single"}),
        ):
            with self.subTest(code=code, changes=changes):
                self.setUp()
                self.pending_items = []
                self.review.update(changes)
                with self.runtime():
                    response = await self.turn("加亮 " + code)
                self.assertIn("未写入", response)
                self.assertIsNone(self.store.get(self.key))
                self.assertEqual([name for name, _ in self.calls], [
                    "keytao_pending_items_by_words", "keytao_prepare_reviewed_add",
                ])

    async def test_dictionary_same_identity_refuses_a_duplicate_after_exact_lookup(self):
        self.pending_items = []
        self.lookup["phrases"] = [{"word": "加亮", "code": "jslxa", "type": "Phrase", "weight": 100}]
        with self.runtime():
            response = await self.turn("加亮 jslxa")
        self.assertIn("不能重复添加", response)
        self.assertIsNone(self.store.get(self.key))
        self.assertEqual(self.writes, [])

    async def test_other_code_advisory_cannot_override_the_selected_code(self):
        self.pending_items = []
        self.review["pronunciations"][0]["candidateStatuses"][0].update(occupied=True, words=["加量"])
        self.review["candidateOrderingAssessments"] = [{
            "verdict": "front_more_common", "newWord": "加亮", "occupantWord": "加量",
            "occupantCode": "jslx", "freeCode": "jslxao", "newCode": "jslx",
        }]
        with self.runtime():
            response = await self.turn("加亮 jslxa")
            response = s63_fresh_selection_chat._enforce_advertised_reply_contract(response, self.key)
        self.assertIn("1. jslxa — ✅推荐", response)
        self.assertNotIn("需重排", response)
        self.assertEqual(self.store.get(self.key).recommended_code, "jslxa")
        self.assertEqual(self.store.get(self.key).server_ordering_assessments, [])

    async def test_group_additional_code_warning_keeps_a_real_actor_ticket(self):
        self.key = harness.ConversationAddress.group("qq", "s63-room", "s63-fresh")
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id=self.key.actor_id, space_type="group", space_id="s63-room",
        )
        self.pending_items = []
        self.review["existing"] = [{"word": "加亮", "code": "jslx", "type": "Phrase"}]
        self.create_warnings = [{
            "warningType": "multiple_code", "message": "该词已有其他编码，请核对是否继续添加。",
            "item": {"action": "Create", "word": "加亮", "code": "jslxa", "type": "Phrase"},
            "existing": self.review["existing"][0],
        }]
        with self.runtime():
            await self.turn("加亮 jslxa")
            response = await b17_rest_recommended_commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key,
                space_key=self.key.space_key, allow_intent_model=False,
            )
            self.assertEqual(self.writes, [])
            self.assertIn("⚠️ 「加亮」将以 jslxa 形成多编码词条", response)
            ticket = self.store.get(self.key)
            self.assertTrue(server_warning_ticket_is_complete(ticket))
            self.assertEqual(ticket.args["code"], "jslxa")
            self.assertIsNone(self.store.get(harness.ConversationAddress.private("qq", self.key.actor_id)))
            response = s63_fresh_selection_chat._enforce_advertised_reply_contract(response, self.key)
            self.assertIn("确认", response)
            await b17_rest_recommended_commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key,
                space_key=self.key.space_key, allow_intent_model=False,
            )
        self.assertEqual([(item["word"], item["code"]) for item in self.writes], [("加亮", "jslxa")])
        self.assertEqual(len([name for name, _args in self.calls if name == "keytao_create_phrase"]), 2)

    async def test_foreign_actor_ticket_is_never_used_for_a_fresh_pair(self):
        self.key = harness.ConversationAddress.group("qq", "s63-room", "s63-fresh")
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id=self.key.actor_id, space_type="group", space_id="s63-room",
        )
        other = harness.ConversationAddress.group("qq", "s63-room", "s63-other")
        self.store.set(other, s62_selection_incident_record())
        original = self.store.get_record(other)
        with self.runtime():
            response = await self.turn("加亮 jslxa")
        self.assertIn(BATCH_URL, response)
        self.assertIs(self.store.get_record(other), original)
        self.assertIsNone(self.store.get(self.key))
        self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])

    async def test_negative_quoted_and_extra_clause_pairs_never_enter_preparation(self):
        self.assertFalse(s63_fresh_selection_chat._chat_routing.message_authorizes_mutation("加亮 jslxa"))
        for message in (
            "不要加亮 jslxa", "别加亮 jslxa", "他说加亮 jslxa",
            "据说加亮 jslxa", "「加亮 jslxa」", "加亮 jslxa？",
            "加亮 jslxa，然后删除", "加亮 jslxa；提交", "加亮 3",
            "加亮 jslxa，小端 xcdti",
        ):
            with self.subTest(message=message):
                self.setUp()
                self.assertIsNone(b17_rest_recommended_commands.fresh_entry_code_selection(message))
                with self.runtime(), patch.object(
                    s63_fresh_selection_chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent()),
                ), patch.object(
                    s63_fresh_selection_chat, "_classify_simple_word_query_intent", AsyncMock(return_value=[]),
                ), patch.object(s63_fresh_selection_chat, "get_ai_response_core", AsyncMock(return_value="只读说明")):
                    await self.turn(message)
                self.assertEqual(self.calls, [])
                self.assertIsNone(self.store.get(self.key))

    async def test_unrelated_live_ticket_is_not_replaced_by_fresh_preparation(self):
        pending = harness.PendingToolConfirm("keytao_submit_batch", {"batch_id": "s63-unrelated"})
        self.store.set(self.key, pending)
        original = self.store.get_record(self.key)
        with self.runtime(), patch.object(
            s63_fresh_selection_chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent()),
        ), patch.object(s63_fresh_selection_chat, "get_ai_response_core", AsyncMock(return_value="只读说明")):
            await self.turn("加亮 jslxa")
        self.assertIs(self.store.get_record(self.key), original)
        self.assertEqual(self.calls, [])

    async def test_quoted_old_display_does_not_create_a_new_ticket(self):
        reply = harness.ReplyReferenceInfo(
            is_reply=True, is_to_bot=True,
            text=s63_fresh_selection_chat.render_server_backed_single_word_candidates(
                "加亮", "jslxa", [("jslxa", False)], {},
            ),
        )
        with self.runtime(), patch.object(
            s63_fresh_selection_chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent()),
        ), patch.object(s63_fresh_selection_chat, "get_ai_response_core", AsyncMock(return_value="只读说明")):
            await self.turn("加亮 jslxa", reply_reference=reply)
        self.assertEqual(self.calls, [])
        self.assertIsNone(self.store.get(self.key))

    async def test_live_s62_pair_keeps_its_existing_sealed_selection_executor(self):
        self.store.set(self.key, s62_selection_incident_record())
        with self.runtime():
            response = await self.turn("加亮 jslxa")
        self.assertIn("已", response)
        self.assertEqual([(item["word"], item["code"]) for item in self.writes], [("加亮", "jslxa")])
        self.assertNotIn("keytao_pending_items_by_words", [name for name, _ in self.calls])
        self.assertEqual(sum(name == "keytao_batch_add_to_draft" for name, _ in self.calls), 2)

# Source: test_s64_explicit_submit.py
class ExplicitSubmitTests(unittest.IsolatedAsyncioTestCase):

    runtime = ExplicitSubmitTests_runtime

    setUp = ExplicitSubmitTests_setUp

    turn = ExplicitSubmitTests_turn

    tool = ExplicitSubmitTests_tool

    executor_runtime = ExplicitSubmitTests_executor_runtime

    seed_existing_draft = ExplicitSubmitTests_seed_existing_draft

    assert_submitted_or_actionable_submit_ticket = ExplicitSubmitTests_assert_submitted_or_actionable_submit_ticket

    async def test_exact_fresh_instruction_does_not_fall_through_to_intent_model(self):
        with self.runtime():
            response = await self.turn(INSTRUCTION)
            self.assert_submitted_or_actionable_submit_ticket(response)
        names = [name for name, _ in self.calls]
        self.assertIn("keytao_prepare_reviewed_add", names)
        self.assertIn("keytao_lookup_by_code", names)
        self.assertEqual(sum(name == "keytao_create_phrase" for name in names), 2)
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("加亮", "jslxa")])
        self.assertIn("keytao_submit_batch", names)

    async def test_expired_candidate_requires_fresh_review_and_preserves_submit_intent(self):
        self.store.set(self.key, s62_selection_incident_record())
        self.now += 11
        with self.runtime():
            response = await self.turn(INSTRUCTION)
            self.assert_submitted_or_actionable_submit_ticket(response)
        self.assertIn("keytao_prepare_reviewed_add", [name for name, _ in self.calls])
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("加亮", "jslxa")])

    async def test_existing_same_draft_is_submitted_without_a_duplicate_create(self):
        self.seed_existing_draft()
        with self.runtime():
            response = await self.turn(INSTRUCTION)
            self.assert_submitted_or_actionable_submit_ticket(response)
        names = [name for name, _ in self.calls]
        self.assertIn("keytao_submit_batch", names)
        self.assertNotIn("keytao_create_phrase", names)
        self.assertEqual(self.writes, [])

    async def test_existing_draft_with_other_words_requires_a_real_snapshot_confirmation(self):
        self.seed_existing_draft(other_item=True)
        with self.runtime():
            response = await self.turn(INSTRUCTION)
            self.assertFalse(self.submitted)
            self.assert_submitted_or_actionable_submit_ticket(response)
            self.assertIn("旧词", response)
            self.assertIn("qtxx", response)
            confirmed = await b17_rest_recommended_commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key,
                space_key=self.key.space_key, allow_intent_model=False,
            )
            self.assertTrue(self.submitted, confirmed)
        self.assertEqual(self.writes, [])

    async def test_group_mention_uses_the_speakers_own_empty_draft(self):
        self.key = harness.ConversationAddress.group("qq", "s64-room", "s64-rea")
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id=self.key.actor_id, space_type="group", space_id="s64-room",
        )
        other_key = harness.ConversationAddress.group("qq", "s64-room", "s64-other")
        self.store.set(other_key, s62_selection_incident_record())
        other_record = self.store.get_record(other_key)
        with self.runtime():
            response = await self.turn("@喵喵 " + INSTRUCTION)
            self.assert_submitted_or_actionable_submit_ticket(response)
        self.assertIs(self.store.get_record(other_key), other_record)
        self.assertIsNone(self.store.get(harness.ConversationAddress.private("qq", self.key.actor_id)))

    async def test_action_first_form_keeps_the_existing_reviewed_submit_path(self):
        with self.runtime():
            response = await self.turn("添加 加亮 jslxa 并提交")
            self.assert_submitted_or_actionable_submit_ticket(response)
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("加亮", "jslxa")])

    async def test_blocked_or_unresolved_review_never_arms_or_writes(self):
        for change in ({"reviewDisposition": "BLOCK"}, {"pronunciationUnresolved": True}):
            with self.subTest(change=change):
                self.setUp()
                self.review.update(change)
                with self.runtime():
                    response = await self.turn(INSTRUCTION)
                self.assertIn("未写入", response)
                self.assertEqual(self.writes, [])
                self.assertFalse(self.submitted)
                self.assertIsNone(self.store.get(self.key))
                self.assertEqual([name for name, _ in self.calls], [
                    "keytao_pending_items_by_words", "keytao_prepare_reviewed_add",
                ])

    async def test_unknown_pending_facts_cannot_be_treated_as_an_empty_draft(self):
        for complete, items in ((False, []), (True, None), (True, {})):
            with self.subTest(complete=complete, items=items):
                self.setUp()
                self.pending_complete, self.pending_items = complete, items
                with self.runtime():
                    response = await self.turn(INSTRUCTION)
                self.assertIn("无法核验", response)
                self.assertEqual(self.writes, [])
                self.assertFalse(self.submitted)
                self.assertIsNone(self.store.get(self.key))
                self.assertEqual([name for name, _ in self.calls], ["keytao_pending_items_by_words"])

    async def test_new_word_preserves_other_draft_items_and_requires_snapshot_confirmation(self):
        old_row = {"id": 6402, "action": "Create", "word": "旧词", "code": "qtxx", "type": "Phrase"}
        self.items = [copy.deepcopy(old_row)]
        self.version = 3
        with self.runtime():
            response = await self.turn(INSTRUCTION)
            self.assertFalse(self.submitted)
            self.assert_submitted_or_actionable_submit_ticket(response)
            self.assertIn("旧词", response)
            self.assertIn("qtxx", response)
            self.assertEqual(self.items[0], old_row)
            confirmed = await b17_rest_recommended_commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key,
                space_key=self.key.space_key, allow_intent_model=False,
            )
            self.assertTrue(self.submitted, confirmed)
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("加亮", "jslxa")])
        self.assertEqual(self.items[0], old_row)

    async def test_live_review_keeps_the_explicit_code_instead_of_its_recommendation(self):
        candidates = [("jslx", False), ("jslxa", False), ("jslxao", False)]
        state = harness.PendingAddWord(
            word="加亮", recommended_code="jslxao", candidates=candidates,
            server_candidates=list(candidates), phrase_type="Phrase", needs_manual_review=True,
            manual_review_reason="常用度证据不足",
            pronunciation_codes={code: "jiā liàng" for code, _ in candidates},
            code_remarks={code: "喵喵审词：读音 jiā liàng；常用度证据不足" for code, _ in candidates},
        )
        self.store.set(self.key, state)
        with self.runtime():
            response = await self.turn(INSTRUCTION)
            self.assert_submitted_or_actionable_submit_ticket(response)
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("加亮", "jslxa")])

    async def test_real_dispatcher_and_executor_create_and_submit_cas_with_reviewed_capability(self):
        with self.executor_runtime():
            response = await self.turn(INSTRUCTION)
            self.assertTrue(self.submitted, response)
            self.assertIn("已提交", response)
        self.assertEqual([name for name, _ in self.calls], [
            "keytao_pending_items_by_words", "keytao_prepare_reviewed_add", "keytao_lookup_by_code",
            "keytao_create_phrase", "keytao_create_phrase", "keytao_submit_batch", "keytao_submit_batch",
        ])
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("加亮", "jslxa")])

    async def test_real_dispatcher_keeps_existing_other_items_behind_submit_confirmation(self):
        self.seed_existing_draft(other_item=True)
        with self.executor_runtime():
            response = await self.turn(INSTRUCTION)
            self.assertFalse(self.submitted)
            self.assert_submitted_or_actionable_submit_ticket(response)
            self.assertIn("旧词", response)
            confirmed = await b17_rest_recommended_commands.handle_pending_message_core(
                "确认", "qq", self.key.actor_id, self.key,
                space_key=self.key.space_key, allow_intent_model=False,
            )
            self.assertTrue(self.submitted, confirmed)
        self.assertEqual([name for name, _ in self.calls], [
            "keytao_pending_items_by_words", "keytao_submit_batch", "keytao_submit_batch",
        ])
        self.assertEqual(self.writes, [])

    async def test_submit_ticket_save_failure_keeps_reason_and_link_without_confirmation_advertisement(self):
        from keytao_bot.utils.pending_confirmation import advertised_command_suggestions

        self.seed_existing_draft(other_item=True)
        with self.runtime(), patch.object(self.store, "set", return_value=False):
            response = await self.turn(INSTRUCTION)
            delivered = s63_fresh_selection_chat._enforce_advertised_reply_contract(response, self.key)
        self.assertFalse(self.submitted)
        self.assertEqual(self.writes, [])
        self.assertIsNone(self.store.get(self.key))
        self.assertIn("提交确认记录未能保存", delivered)
        self.assertIn(self.batch_url, delivered)
        self.assertFalse(advertised_command_suggestions(delivered))

    async def test_deferred_negative_prefixes_cannot_submit_a_matching_literal_draft(self):
        from keytao_bot.utils.explicit_code import parse_explicit_entry_code_request

        for word in ("先不加亮", "暂不加亮", "暂时不加亮", "请先不加亮"):
            with self.subTest(word=word):
                self.setUp()
                self.seed_existing_draft()
                self.items[0]["word"] = word
                self.pending_items[0]["word"] = word
                message = word + " jslxa，加入草稿并提交。"
                self.assertIsNone(parse_explicit_entry_code_request(message))
                with self.runtime():
                    response = await b17_rest_recommended_commands.try_handle_explicit_entry_code_command(
                        message, "qq", self.key.actor_id, self.key,
                        self.key.space_key, "Rea",
                    )
                self.assertIsNone(response)
                self.assertEqual(self.calls, [])
                self.assertEqual(self.writes, [])
                self.assertFalse(self.submitted)
                self.assertIsNone(self.store.get(self.key))

# Source: test_s64_explicit_submit.py
class ExplicitSubmitGrammarTests(unittest.TestCase):
    def test_literal_code_suffix_and_possessive_code_marker(self):
        for message, word, code in (
            ("添加转编码 zbm", "转编码", "zbm"),
            ("添加我的编码 abcd", "我的编码", "abcd"),
            ("添加洋节，编码改为ypjd", "洋节", "ypjd"),
            ("添加洋节，编码是ypjd", "洋节", "ypjd"),
            ("添加洋节的编码改为ypjd", "洋节", "ypjd"),
            ("加词：苏辙的编码为sjfei", "苏辙", "sjfei"),
            ("添加苏辙的编码是sjfei", "苏辙", "sjfei"),
            ("添加苏辙的编码：sjfei", "苏辙", "sjfei"),
            ("添加苏辙的编码:sjfei", "苏辙", "sjfei"),
        ):
            with self.subTest(message=message):
                request = parse_explicit_entry_code_request(message)
                self.assertIsNotNone(request)
                self.assertEqual((request.word, request.code), (word, code))

    def test_code_description_keeps_single_add_without_hiding_other_operations(self):
        from keytao_bot.harness.semantic_parser import has_mixed_entry_operations

        for message, mixed in (
            ("添加洋节，编码改为ypjd", False),
            ("添加洋节，编码是ypjd", False),
            ("添加洋节 ypjd，删除纯音乐", True),
            ("添加洋节 ypjd，把纯音乐改为jyya", True),
            ("添加洋节 ypjd，“纯音乐”编码改为jyya", True),
            ("添加洋节，编码改为ypjd，然后删除纯音乐", True),
        ):
            with self.subTest(message=message):
                self.assertEqual(has_mixed_entry_operations(message), mixed)

    def test_named_pair_and_action_forms_preserve_submit_intent(self):
        from keytao_bot.utils.explicit_code import parse_explicit_entry_code_request

        for message in (
            INSTRUCTION, "加亮 jslxa,加入草稿并提交。", "加亮 jslxa 加入草稿并提交",
            "加亮 jslxa，添加到草稿并提交。", "添加 加亮 jslxa 并提交",
        ):
            with self.subTest(message=message):
                request = parse_explicit_entry_code_request(message)
                self.assertIsNotNone(request)
                self.assertEqual((request.word, request.code, request.submit_after), ("加亮", "jslxa", True))

    def test_negative_reported_quoted_and_extra_target_forms_are_not_commands(self):
        from keytao_bot.utils.explicit_code import parse_explicit_entry_code_request

        for message in (
            "不要加亮 jslxa，加入草稿并提交。", "他说加亮 jslxa，加入草稿并提交。",
            "「加亮 jslxa，加入草稿并提交。」", "加亮 jslxa，加入草稿并提交？",
            "加亮 jslxa，小端 xcdti，加入草稿并提交。",
            "加亮 jslxa，加入草稿并提交，然后删除加量。",
            "加亮 jslxa，加入到",
        ):
            with self.subTest(message=message):
                self.assertIsNone(parse_explicit_entry_code_request(message))

# Source: test_s62_incident.py
class S62IncidentTests(unittest.IsolatedAsyncioTestCase):
    replay = S62IncidentTests_replay

    async def test_turn_two_to_seven_fixture_branches_reach_final_delivery(self):
        for selection in (
            "鸡白汤 jbtaua, 敲不死 qbso，加入草稿并提交",
            "鸡白汤 jbtaua, 敲不死 qbso; 加入草稿并提交.",
            "鸡白汤 jī bái tāng：jbtaua，敲不死 qbso; 加入并提交",
            "敲不死 2 加入并提交",
        ):
            with self.subTest(selection=selection):
                writes, submitted, replies, calls, _ = await self.replay(
                    [selection, "查看草稿"], discovery=True, delivery=True,
                )
                self.assertEqual([(row["word"], row["code"]) for row in writes], [("敲不死", "qbso")], replies)
                self.assertTrue(submitted, replies)
                self.assertIn("已提交审核", replies[1])
                if "鸡白汤" in selection:
                    self.assertIn("鸡白汤", replies[1])
                    self.assertIn("读音", replies[1])
                    self.assertIn("未加入", replies[1])
                self.assertIn("当前草稿", replies[2])
                self.assertNotIn("候选", replies[2])
                self.assertNotIn("鸡白汤", replies[2])
                self.assertEqual(sum(name == "keytao_batch_add_to_draft" for name, _ in calls), 2)
                self.assertEqual(sum(name == "keytao_submit_batch" for name, _ in calls), 2)

    async def test_real_dispatcher_and_executor_bind_the_mixed_selection(self):
        writes, submitted, replies, _, _ = await self.replay(
            ["鸡白汤 jbtaua, 敲不死 qbso，加入草稿并提交"],
            discovery=True, delivery=True, executor=True,
        )
        self.assertEqual([(row["word"], row["code"]) for row in writes], [("敲不死", "qbso")], replies)
        self.assertTrue(submitted, replies)
        self.assertIn("鸡白汤", replies[-1])
        self.assertIn("未加入", replies[-1])

    async def test_combined_action_forms_and_separators(self):
        for action in ("加入", "加入草稿", "加到草稿", "写入草稿", "加入并提交", "加入草稿并提交", "加到草稿并提交"):
            for separator in (" ", "，", ",", "、", "；", ";"):
                for message in (f"敲不死 2{separator}{action}。", f"{action}{separator}敲不死 qbso."):
                    with self.subTest(message=message):
                        parsed = grammar.parse_reviewed_selection_command(message)
                        self.assertIsNotNone(parsed)
                        self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(message, s62_incident_incident_record()))
                        writes, submitted, replies, _, _ = await self.replay([message])
                        self.assertEqual([(r["word"], r["code"]) for r in writes], [("敲不死", "qbso")], replies)
                        self.assertEqual(submitted, "提交" in action, replies)

    async def test_mixed_message_keeps_bindable_selection(self):
        for message in ("鸡白汤 jbtaua, 敲不死 qbso，加入草稿并提交",
                        "鸡白汤 jbtaua, 敲不死 qbso; 加入草稿并提交."):
            writes, submitted, replies, _, _ = await self.replay([message])
            self.assertEqual([(r["word"], r["code"]) for r in writes], [("敲不死", "qbso")], replies)
            self.assertTrue(submitted, replies)
            self.assertIn("鸡白汤", replies[0])
            self.assertIn("读音", replies[0])
            self.assertIn("未加入", replies[0])

    async def test_bare_selection_writes_once_with_receipt_and_never_submits(self):
        for message in ("敲不死 2", "鸡白汤 jbtaua, 敲不死 qbso"):
            with self.subTest(message=message):
                writes, submitted, replies, calls, state = await self.replay(
                    [message], discovery=True, delivery=True, executor=True,
                )
                self.assertEqual([(r["word"], r["code"]) for r in writes], [("敲不死", "qbso")], replies)
                self.assertFalse(submitted)
                self.assertIsNone(state)
                self.assertEqual(sum(name == "keytao_batch_add_to_draft" for name, _ in calls), 2)
                self.assertFalse(any(name == "keytao_submit_batch" for name, _ in calls))
                self.assertTrue(writes[0]["needsManualReview"])
                self.assertIn("敲不死 → qbso", replies[-1])
                self.assertIn("https://keytao.rea.ink/batch/s62-fixture", replies[-1])
                self.assertNotIn("已提交", replies[-1])
                self.assertNotIn("尚未加入", replies[-1])
                if "鸡白汤" in message:
                    self.assertIn("「鸡白汤」还没确定读音，未加入", replies[-1])

    async def test_nonrecommended_selection_writes_and_mixed_errors_do_not_poison_valid_choice(self):
        writes, submitted, replies, _, _ = await self.replay(["敲不死 3"], delivery=True)
        self.assertEqual([(r["word"], r["code"]) for r in writes], [("敲不死", "qbsoa")], replies)
        self.assertFalse(submitted)
        for invalid, reason in (("9", "超出"), ("evil", "不在当前候选")):
            state = reviewed_record()
            selected, intent, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(
                state, f"小端 {invalid}，大端 3，加入并提交",
            )
            self.assertIsNone(error)
            self.assertEqual([(row["word"], row["code"]) for row in selected.args["items"]], [("大端", "dsdtvo")])
            self.assertEqual(intent.intent, "pending_add_and_submit")
            self.assertIn(reason, selected.args["_selection_skipped"][0])

    async def test_complete_envelope_and_review_seals_still_required(self):
        for message in (
            "他说鸡白汤 jbtaua, 敲不死 qbso，加入并提交",
            "不要鸡白汤 jbtaua, 敲不死 qbso，加入并提交",
            "鸡白汤 jbtaua, 敲不死 qbso，加入并提交？",
            "敲不死 2，加入并提交，然后删除", "敲不死 2，敲不死 2，加入",
            "他说「敲不死 2，加入并提交」", "「『敲不死 2，加入并提交』」",
        ):
            with self.subTest(message=message):
                writes, _, _, calls, _ = await self.replay([message])
                self.assertEqual(writes, [])
                self.assertEqual(calls, [])
        for mutation in ("missing", "candidate", "warning"):
            record = s62_incident_incident_record()
            if mutation == "missing":
                record.args["_candidate_scopes"][0].pop("reviewedState")
            elif mutation == "candidate":
                record.args["_candidate_scopes"][0]["candidates"][1][0] = "evil"
            else:
                record.confirmation_source = "server_warning"
            self.assertFalse(b17_rest_recommended_routing.message_authorizes_live_pending_mutation("敲不死 2，加入并提交", record))

    def test_footer_and_added_commands_close_over_parser_and_binding(self):
        record = s62_incident_incident_record()
        footer = render_server_backed_batch_candidates(record.args["items"], record.args["_candidate_scopes"])
        controls = advertised_command_suggestions(footer)
        self.assertEqual(controls, ("加入", "加入并提交", "敲不死 2", "敲不死 2，加入并提交"))
        for command in controls:
            with self.subTest(command=command):
                if command == "取消":
                    self.assertEqual(b17_rest_recommended_routing._pending_tool_assent_intent(record, command).intent, "pending_cancel")
                elif grammar.parse_reviewed_multi_word_selection(command):
                    selected, intent, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(record, command)
                    self.assertIsNotNone(selected, error)
                    self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(command, record))
                else:
                    self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(command, record))

# Source: test_s62_adversarial.py
class S62AdversarialTests(unittest.IsolatedAsyncioTestCase):
    async def test_reported_and_negative_envelopes_never_write(self):
        for prefix in ("我说", "阿邦说", "原话是", "转述", "先不要", "暂时不", "并非", "尚未", "绝不能"):
            message = f"{prefix}鸡白汤 jbtaua, 敲不死 qbso，加入并提交"
            with self.subTest(message=message):
                self.assertFalse(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(message, s62_incident_incident_record()))

    async def test_reported_speech_cannot_reach_fake_mutation(self):
        writes, submitted, _, calls, _ = await S62IncidentTests().replay([
            "原话是鸡白汤 jbtaua, 敲不死 qbso，加入并提交",
        ])
        self.assertEqual(writes, [])
        self.assertFalse(submitted)
        self.assertEqual(calls, [])

    def test_exact_live_lexical_targets_keep_the_selection_contract(self):
        for word in ("不明觉厉", "记忆力", "说唱"):
            record = s62_incident_incident_record()
            record.args = json.loads(json.dumps(record.args, ensure_ascii=False).replace("敲不死", word))
            with self.subTest(word=word):
                self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(f"{word} 2，加入", record))

    async def test_bare_selection_delivery_keeps_write_receipt_and_skipped_reason(self):
        writes, submitted, replies, _, state = await S62IncidentTests().replay(
            ["鸡白汤 jbtaua, 敲不死 3"], delivery=True,
        )
        self.assertEqual([(r["word"], r["code"]) for r in writes], [("敲不死", "qbsoa")])
        self.assertFalse(submitted)
        self.assertIsNone(state)
        delivered = replies[0]
        self.assertIn("敲不死 → qbsoa", delivered)
        self.assertIn("https://keytao.rea.ink/batch/s62-fixture", delivered)
        self.assertIn("鸡白汤", delivered)
        self.assertIn("未加入", delivered)

    def test_invalid_selection_delivery_keeps_explanation_and_retry(self):
        state = s62_incident_incident_record()
        _, _, reply = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(state, "敲不死 99，加入")
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.group("qq", "865189947", "s62-offline")
        store.set(key, state)
        with patch.object(harness.openai_chat_module, "conversation_state_store", store):
            delivered = harness.openai_chat_module._enforce_advertised_reply_contract(reply, key)
        self.assertIn("99", delivered)
        self.assertIn("超出", delivered)
        retry = advertised_command_suggestions(delivered)
        self.assertEqual(len(retry), 1)
        self.assertTrue(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(retry[0], state))

    def test_protected_recommended_code_keeps_its_reason_at_final_delivery(self):
        state = s62_incident_incident_record()
        state.args["items"][0]["code"] = "qbs"
        state.args["_candidate_scopes"][0]["reviewedState"]["recommendedCode"] = "qbs"
        _, _, reply = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(state, "敲不死 1")
        self.assertIn("常用度", reply)
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.group("qq", "865189947", "s62-offline")
        store.set(key, state)
        with patch.object(harness.openai_chat_module, "conversation_state_store", store):
            delivered = harness.openai_chat_module._enforce_advertised_reply_contract(reply, key)
        self.assertEqual(str(delivered), str(reply))
        self.assertIn("敲破死", delivered)
        self.assertIn("未加入", delivered)

# Source: test_s68_selection.py
class SelectionTests(unittest.IsolatedAsyncioTestCase):
    runtime = ExplicitSubmitTests_runtime
    executor_runtime = TranscriptTests_executor_runtime
    turn = FreshSelectionTests_turn
    delivered_turn = TranscriptTests_delivered_turn

    setUp = SelectionTests_setUp

    tool = SelectionTests_tool

    prepare = SelectionTests_prepare

    async def test_t1_t2_selects_ordinal_three_without_model(self):
        with self.executor_runtime():
            t1 = await self.prepare()
            self.assertIn("yhjluu", t1)
            self.assertIn("lejluu", t1)
            self.assertIn("11.25", t1)
            self.assertIn("6.28", t1)
            self.assertIsInstance(self.store.get(self.key), PendingAddWord)
            t2 = await self.delivered_turn("只加入3")
            await b17_rest_recommended_commands.handle_pending_message_core("只加入3", "qq", self.key.actor_id, self.key, allow_intent_model=False)
        self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("乐句", "yhjluu")])
        self.assertEqual([row["modelCalls"] for row in self.trace], [0, 0])
        self.assertIn("未添加", t2)
        for code in ("yhjl", "yhjlu", "lejl", "lejlu", "lejluu"):
            self.assertIn(code, t2)
        path = Path("/tmp/keytao-s68")
        path.mkdir(exist_ok=True)
        (path / "replay.json").write_text(json.dumps({"turns": self.trace, "writes": self.writes}, ensure_ascii=False, indent=2) + "\n")

    async def test_variants_use_live_global_ordinals_and_exact_set(self):
        cases = [(message, ["yhjluu"]) for message in (
            "仅加入 3", "只添加3", "加入3", "3", "就加入3", "光加入3", "只要加入3",
            "3，只加入。", "只加入，３．", "添加3",
        )] + [(message, ["yhjluu", "lejl"]) for message in (
            "只加入3、4", "仅添加３；４。", "3,4,只加入", "只要加入 3;4.",
        )]
        for message, codes in cases:
            with self.subTest(message=message):
                self.setUp()
                with self.executor_runtime():
                    await self.prepare()
                    response = await self.delivered_turn(message)
                self.assertEqual([item["code"] for item in self.writes], codes)
                self.assertEqual(self.trace[-1]["modelCalls"], 0)
                self.assertIn("未选择：", response)
                self.assertNotIn("已提交", response)
                self.assertFalse(any(name == "keytao_submit_batch" for name, _ in self.calls))

    async def test_two_words_require_scope_and_only_add_the_named_word(self):
        for message in ("只加入 乐句 3", "仅添加 乐句 3", "乐句 3，只加入。"):
            with self.subTest(message=message):
                self.setUp()
                with self.executor_runtime():
                    await self.prepare("乐句 小端")
                    response = await self.delivered_turn(message)
                self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("乐句", "yhjluu")])
                self.assertIn("未选择：「小端」", response)
                self.assertEqual(self.trace[-1]["modelCalls"], 0)
        self.setUp()
        with self.runtime():
            await self.prepare("乐句 小端")
            for message in ("只加入3", "3"):
                self.assertFalse(b17_rest_recommended_routing.message_authorizes_live_pending_mutation(message, self.store.get(self.key)))
                response = await b17_rest_recommended_commands.handle_pending_message_core(message, "qq", self.key.actor_id, self.key, allow_intent_model=False)
                self.assertIn("未写入", response)
        self.assertEqual(self.writes, [])

    async def test_invalid_negated_quoted_stale_foreign_and_unsealed_selections_never_write(self):
        with self.runtime():
            await self.prepare()
            state = self.store.get(self.key)
            for message in ("不要只加入3", "他说只加入3", "只加入3？", "只加入3然后删除", "只加入0", "只加入7", "只加入3、3", "只加入3、", "只加入3,4，删除"):
                self.store.set(self.key, copy.deepcopy(state))
                before = len(self.writes)
                await b17_rest_recommended_commands.handle_pending_message_core(message, "qq", self.key.actor_id, self.key, allow_intent_model=False)
                self.assertEqual(len(self.writes), before, message)
            foreign = harness.ConversationAddress.group("qq", "865189947", "foreign")
            await b17_rest_recommended_commands.handle_pending_message_core("只加入3", "qq", foreign.actor_id, foreign, allow_intent_model=False)
            self.store.set(self.key, replace(state, server_candidates=[]))
            self.assertFalse(b17_rest_recommended_routing.message_authorizes_live_pending_mutation("只加入3", self.store.get(self.key)))
            self.now += 11
            await b17_rest_recommended_commands.handle_pending_message_core("只加入3", "qq", self.key.actor_id, self.key, allow_intent_model=False)
        self.assertEqual(self.writes, [])

    def test_single_word_batch_record_requires_the_complete_live_inventory(self):
        record = reviewed_record(("大端",))
        for message in ("只加入3", "仅加入 3", "只添加3", "加入3", "3"):
            selected, intent, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(record, message)
            self.assertIsNone(error)
            self.assertEqual(intent.intent, "pending_confirm")
            self.assertEqual([(row["word"], row["code"]) for row in selected.args["items"]], [("大端", "dsdtvo")])
        record.args["_query_words"] = ["大端", "小端"]
        selected, intent, error = b17_rest_recommended_routing._resolve_multi_word_pending_candidate_selection(record, "只加入3")
        self.assertIsNone(selected)
        self.assertIsNone(intent)
        self.assertIn("未写入", error)

    def test_raw_selection_parser_rejects_quotes_and_extra_commands(self):
        for message in ("「只加入3」", "「『只加入3』」", "他说只加入3", "只加入3再删除", "只加入3？", "只加入3；提交"):
            self.assertIsNone(parse_pending_candidate_selection(message), message)

    def test_subset_submit_is_not_a_candidate_selection(self):
        for message in ("只加入3", "仅加入3", "只添加3"):
            self.assertIsNone(b17_rest_recommended_commands._SUBSET_SUBMIT_RE.fullmatch(message))
        for message in ("只提交3", "仅提审3"):
            self.assertIsNotNone(b17_rest_recommended_commands._SUBSET_SUBMIT_RE.fullmatch(message))
            self.assertIsNone(parse_pending_candidate_selection(message))

    def test_restrictive_bare_ordinal_grammar(self):
        for message in ("只加入3", "仅加入 3", "只添加3", "加入3", "3", "就加入3", "光加入3", "只要加入3", "3，只加入。"):
            with self.subTest(message=message):
                parsed = parse_pending_candidate_selection(message)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.indices, (3,))

# Source: test_s68_selection.py
class ProposalTests(unittest.IsolatedAsyncioTestCase):
    setUp = GenericProposalBridgeTests_setUp
    call = GenericProposalBridgeTests_call
    assert_ticket = GenericProposalBridgeTests_assert_ticket

    async def test_trusted_structured_batch_does_not_need_prose_operands(self):
        with patch("keytao_bot.harness.tools.logger.info") as log:
            result = await self.call("keytao_batch_add_to_draft", {
                "items": [{"word": "奈飞", "code": "nhfw", "remark": "Model prose must not be trusted"}],
            }, "只安排加入第三项")
        self.assertTrue(any("grammar_gap_bridged=True" in call.args[0] for call in log.call_args_list))
        state = self.assert_ticket(result, "keytao_batch_add_to_draft")
        self.assertEqual(state.args["items"][0]["remark"], reviewed("奈飞", "nhfw")["remark"])

    async def test_unverifiable_structured_batch_has_no_ticket(self):
        result = await self.call("keytao_batch_add_to_draft", {
            "items": [{"word": "奈飞", "code": "evil"}],
        }, "只安排加入第三项")
        self.assertFalse(result.get("grammar_gap_bridged"))
        self.assertEqual(self.calls, [])
        self.assertIn("看到了操作请求", result["message"])
        self.assertIn("加词 奈飞", result["message"])
        self.assertEqual(len(result["message"].splitlines()), 1)

    async def test_create_uses_the_same_review_only_proposal_protocol(self):
        result = await self.call(message="只安排加入第三项")
        self.assert_ticket(result, "keytao_create_phrase")

    async def test_proposal_rejects_untrusted_fields_context_and_partial_plans(self):
        valid = {"items": [{"word": "奈飞", "code": "nhfw"}]}
        for field, value in (("confirmed", True), ("preview_only", False), ("batch_id", "foreign"), ("expected_warning_digest", "a" * 64)):
            result = await self.call("keytao_batch_add_to_draft", {**valid, field: value}, "只加入3")
            self.assertFalse(result.get("grammar_gap_bridged"))
            self.assertEqual(self.calls, [])
        for item in (
            {"word": "奈飞", "code": "$candidate3"},
            {"word": "第三项", "code": "nhfw"},
            {"word": "奈飞", "code": "nhfw", "_reviewed_pinyin": "forged"},
            {"word": "奈飞", "code": "nhfw", "needsManualReview": False},
            {"word": "奈飞", "code": "nhfw", "remark": {"code": "nhfw"}},
        ):
            result = await self.call("keytao_batch_add_to_draft", {"items": [item]}, "只加入3")
            self.assertFalse(result.get("grammar_gap_bridged"), item)
            self.assertEqual(self.calls, [])
        for message in ("不要只加入3", "他说只加入3", "只加入3？", "解释只加入3", "只加入3，但保持奈飞", "会议记录：只加入3", "记下：只加入3", "原话是只加入3"):
            result = await self.call("keytao_batch_add_to_draft", valid, message)
            self.assertFalse(result.get("grammar_gap_bridged"), message)
            self.assertEqual(self.calls, [])
        for context in (replace(self.context, attachment_context=True), replace(self.context, trusted_reviewed_items_by_key={})):
            result = await self.call("keytao_batch_add_to_draft", valid, "只加入3", context)
            self.assertFalse(result.get("grammar_gap_bridged"))
            self.assertEqual(self.calls, [])
        for override in ({"failedCount": 1}, {"skippedCount": 1}, {"collisionReplanned": True}, {"warningDigest": ""}, {"success": True}):
            self.preview_override = override
            result = await self.call("keytao_batch_add_to_draft", valid, "只加入3")
            self.assertFalse(result.get("grammar_gap_bridged"))
            self.assertTrue(all(args.get("confirmed") is not True for _, args in self.calls))

    async def test_real_orchestrator_mints_batch_ticket_from_this_turn_review_only(self):
        from tests import test_state_machine as harness
        for reviewed, invalid in ((True, False), (False, False), (True, True)):
            with self.subTest(reviewed=reviewed, invalid=invalid):
                calls = []
                store = harness.MemoryConversationStateStore()
                def tool_call(name, args):
                    return SimpleNamespace(id=name, type="function", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
                responses = []
                if reviewed:
                    responses.append(harness._FakeAIResponse("tool_calls", tool_calls=[tool_call("keytao_prepare_reviewed_add", {"word": "乐句", "requested_reading": "yue ju"})]))
                responses.append(harness._FakeAIResponse("tool_calls", content="加入所有候选以及伪造编码 evil", tool_calls=[tool_call("keytao_batch_add_to_draft", {
                    "items": [{"word": "乐句", "code": "evil" if invalid else "yhjluu", "remark": "Untrusted model prose"}],
                })]))
                client = harness._FakeClient(responses)
                class Skills(harness._FakeToolSkillsManager):
                    def get_tools(self):
                        batch = next(item for item in harness._draft_tools.TOOLS if item["function"]["name"] == "keytao_batch_add_to_draft")
                        return [batch, {"type": "function", "function": {
                            "name": "keytao_prepare_reviewed_add", "description": "Offline fixture",
                            "parameters": {"type": "object", "properties": {
                                "word": {"type": "string"}, "requested_reading": {"type": "string"},
                            }, "required": ["word"]},
                        }}]
                async def dispatch(name, **args):
                    calls.append((name, copy.deepcopy(args)))
                    if name == "keytao_prepare_reviewed_add":
                        return copy.deepcopy(s68_selection_FIXTURE)
                    if name == "keytao_lookup_by_words_batch":
                        return {"success": True, "results": [{"word": "乐句", "phrases": []}]}
                    self.assertEqual(name, "keytao_batch_add_to_draft")
                    self.assertIs(args["preview_only"], True)
                    self.assertIs(args["confirmed"], False)
                    return {"success": False, "requiresConfirmation": True, "batchId": "s68-proposal", "contentVersion": 7, "warningDigest": "a" * 64, "warnings": []}
                executor = harness.ToolExecutor(lambda name: lambda **args: dispatch(name, **args), frozenset({"keytao_batch_add_to_draft"}))
                orchestrator = harness.AgentOrchestrator(client_factory=lambda: client,
                    runtime=harness.AgentRuntimeConfig("fake-model", 1000, 0.0, 10.0), skills_manager=Skills(), tool_executor=executor,
                    state_store=store, bind_help_text="bind help", system_prompt_core="system")
                context = harness.AgentRequestContext(platform="qq", user_id="s68-proposal", mutations_allowed=False)
                reply = await orchestrator.run("只加入3", context)
                record = store.get_record(context.conversation_address)
                if reviewed and not invalid:
                    self.assertIsNotNone(record, reply)
                    self.assertTrue(server_warning_ticket_is_complete(record.state))
                    self.assertEqual([(item["word"], item["code"]) for item in record.state.args["items"]], [("乐句", "yhjluu")])
                    self.assertEqual(record.state.args["items"][0]["_reviewed_pinyin"], "yuè jù")
                    self.assertNotIn("Untrusted model prose", reply)
                    self.assertNotIn("evil", reply)
                    self.assertIn("确认", reply)
                    self.assertIsNone(store.get_record(replace(context, user_id="foreign").conversation_address))
                else:
                    self.assertIsNone(record, reply)
                    self.assertFalse(any(name == "keytao_batch_add_to_draft" for name, _ in calls))
                    self.assertIn("未写入", reply)
                self.assertEqual(len(client.completions.calls), 2 if reviewed else 1)


def load_tests(loader, standard_tests, pattern):
    return _stage_t_unittest.TestSuite(
        loader.loadTestsFromTestCase(case) for case in (
            B12SelectionTests,
            B14MultiSelectionTests,
            B15ChatTests,
            B15PendingLookupTests,
            B17ParserTests,
            B17RestRouteTests,
            B17ShiftPlanTests,
            SelectionCommandTests,
            B18RealToolTests,
            B18ShiftValidationTests,
            B7SelectionTests,
            B8PresentWordsTests,
            AddKeepsSubmitTests,
            HotfixMultiwordTests,
            MultiwordRouteTests,
            S54SelectionTests,
            SinglePipelineTests,
            S56AdvertisedClosureTests,
            OfferedAnswerClaimingTests,
            S57OptionBoundaryTests,
            S57OptionStructureTests,
            S57OptionClaimTests,
            S62SelectionTests,
            FreshSelectionTests,
            ExplicitSubmitTests,
            ExplicitSubmitGrammarTests,
            S62IncidentTests,
            S62AdversarialTests,
            SelectionTests,
            ProposalTests,
        )
    )


if __name__ == "__main__":
    _stage_t_unittest.main()
