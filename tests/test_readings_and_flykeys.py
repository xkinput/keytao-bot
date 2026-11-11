"""Topic regressions: readings and flykeys."""

import unittest as _stage_t_unittest
from tests.helpers import (
    ALTERNATE,
    ALTERNATE_CODES,
    AsyncMock,
    B10FlyKeyTests_add_fly_key,
    B10FlyKeyTests_assert_pending_code_allows_add,
    B10FlyKeyTests_setUp,
    B13DefaultReadingTests_setUp,
    B19PartialReplyBindingTests_created,
    B19PartialReplyBindingTests_matcher_finish,
    B19PartialReplyBindingTests_matcher_send,
    B25SessionTests_greeting,
    B25SessionTests_history_for,
    B25SessionTests_remember,
    B25SessionTests_tearDown,
    B25SessionTests_turn,
    B27ReadingTests_finish_add,
    B27ReadingTests_modifier_adds,
    B27ReadingTests_reset_session,
    B27ReadingTests_seed,
    B27ReadingTests_setUp,
    ENCODE,
    ExplicitAddHotfixTests_assert_created,
    ExplicitAddHotfixTests_asyncSetUp,
    ExplicitAddHotfixTests_call,
    ExplicitAddHotfixTests_execute,
    ExplicitAddHotfixTests_mark_resolvable_ambiguity,
    ExplicitCodeRequest,
    ExplicitEntryCodeRequest,
    ExplicitReadingReplayTests_setUp,
    ExplicitReadingReplayTests_tool,
    ExplicitSubmitTests_runtime,
    ExplicitSubmitTests_turn,
    FLY_CODES,
    FlyTranscriptTests_setUp,
    FlyTranscriptTests_tool,
    FreshSelectionTests_turn,
    LiveTicketTests_prepare_ticket,
    LiveTicketTests_setUp,
    LiveTicketTests_tool,
    MODIFIERS,
    OLD_CODES,
    Path,
    PendingAddWord,
    PendingToolConfirm,
    READING,
    READINGS,
    REPLIES,
    REREVIEW,
    ReadingRereviewPersistenceTests_assert_batch_release_dash,
    ReadingRereviewPersistenceTests_assert_incident,
    ReadingRereviewPersistenceTests_assert_new_word_dash,
    ReadingRereviewPersistenceTests_asyncSetUp,
    ReadingRereviewPersistenceTests_initial_review,
    ReadingRereviewPersistenceTests_rereview,
    ReadingRereviewPersistenceTests_tool,
    ReadingRereviewPersistenceTests_turn,
    ReadingResolutionTests_execute,
    ReadingResolutionTests_prepare_ambiguous,
    SQLiteConversationStateStore,
    SimpleNamespace,
    SingleReviewTests_audit_single,
    SingleReviewTests_prepare,
    T1,
    ToolContext,
    TranscriptTests_delivered_turn,
    TranscriptTests_executor_runtime,
    TranscriptTests_setUp,
    TranscriptTests_tool,
    additional_code_warning,
    advertised_command_suggestions,
    advertised_single_word_candidate_codes,
    ambiguous_review,
    asyncio,
    b10_flykey_CODES,
    b10_flykey_WORD,
    b17_rest_recommended_commands,
    b25_session_commands,
    b27_reading_WORD,
    chat_render,
    copy,
    create_pending_trusted_word_record,
    existing_single_review,
    harness,
    hotfix_explicit_add_WORD,
    json,
    normalize_mutation_command_source,
    os,
    parse_explicit_code_request,
    parse_parenthesised_readings,
    patch,
    pending,
    prepare_fly_review,
    raw_operation,
    replace,
    review,
    review_fixture,
    reviewed_single,
    reviewed_word,
    s54_multiword_review_payload,
    s56_explicit_code_chat,
    s56_explicit_code_commands,
    s56_explicit_code_routing,
    s56_security_chat,
    s56_security_commands,
    s56_security_routing,
    s57_extra_code_chat,
    s57_extra_code_commands,
    s57_extra_code_routing,
    s57_security_review_chat,
    s57_security_review_commands,
    s63_fresh_selection_chat,
    s66_readings_CODES,
    s66_readings_T2,
    s66_readings_WORDS,
    s69_fly_ticket_chat,
    s69_fly_ticket_commands,
    s73_reading_rereview_persistence_chat,
    select_candidate_inventory,
    single_encoding,
    single_review,
    tempfile,
    ticket,
    trusted_word_record_is_complete,
    unittest,
    validate_explicit_code,
    validate_operation,
)


# Source: test_b10_flykey.py
class B10FlyKeyTests(unittest.IsolatedAsyncioTestCase):
    setUp = B10FlyKeyTests_setUp

    add_fly_key = B10FlyKeyTests_add_fly_key

    async def test_selector_is_preserved_by_semantic_validation(self):
        request, violations = validate_operation(raw_operation(
            op="add", operands=[b10_flykey_WORD], codes=[], code_selector="fly_key",
        ), f"新增“{b10_flykey_WORD}”的飞键编码", None)
        self.assertEqual(getattr(request, "code_selector", None), "fly_key")
        self.assertEqual(violations, [])

    async def test_selector_requires_literal_fly_key_operand_and_known_value(self):
        for message, changes in (
            (f"添加{b10_flykey_WORD}", {"code_selector": "fly_key"}),
            (f"新增{b10_flykey_WORD}的飞键编码", {"code_selector": "invented"}),
            (f"新增{b10_flykey_WORD}的飞键编码", {"code_selector": ["fly_key"]}),
            (f"新增{b10_flykey_WORD}的飞键编码", {"operands": ["找麻烦"], "code_selector": "fly_key"}),
            (f"新增{b10_flykey_WORD}的飞键编码", {"codes": ["fmfva"], "code_selector": "fly_key"}),
        ):
            with self.subTest(message=message, changes=changes):
                payload = raw_operation(op="add", operands=[b10_flykey_WORD], codes=[])
                payload.update(changes)
                request, violations = validate_operation(payload, message, None)
                self.assertTrue(violations)
                self.assertFalse(request.op == "add" and getattr(request, "code_selector", "") == "fly_key")

    async def test_fly_key_add_uses_first_empty_marked_candidate(self):
        outcome = await self.add_fly_key()
        self.assertEqual(outcome["added"], [[b10_flykey_WORD, "fmfva"]], self.replay.last_response)
        self.assertIn("fmfva", self.replay.last_response)
        self.assertIn("https://", self.replay.last_response)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.general.assert_not_awaited()

    async def test_disabling_direct_exec_requires_confirmation_and_retains_fly_selector(self):
        with patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": "add"}):
            outcome = await self.add_fly_key()
            self.assertEqual(outcome["added"], [])
            self.assertEqual(self.replay.calls, [])
            record = self.replay.store.get_record(self.replay.author)
            self.assertIsInstance(record.state, PendingToolConfirm)
            self.assertEqual(record.state.function_name, "keytao_understood_command")
            self.assertEqual(record.state.args.get("code_selector"), "fly_key")
            outcome = await self.replay.turn({"message": "确认", "to_me": True}, raw_operation(
                op="confirm", operands=[], codes=[], refers_to_pending=True,
            ))
        self.assertEqual(outcome["added"], [[b10_flykey_WORD, "fmfva"]], self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_supported_fly_key_phrasings_share_direct_route(self):
        for message in (
            f"新增{b10_flykey_WORD}的飞键编码", f"加“{b10_flykey_WORD}”飞键码", f"添加「{b10_flykey_WORD}」飞键",
            f"给{b10_flykey_WORD}加飞键编码", f'给"{b10_flykey_WORD}"加飞键码', f"新增'{b10_flykey_WORD}'的飞键编码",
        ):
            with self.subTest(message=message):
                self.replay.writes.clear()
                self.replay.draft.clear()
                outcome = await self.add_fly_key(message)
                self.assertEqual(outcome["added"], [[b10_flykey_WORD, "fmfva"]], self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_candidate_marking_is_authoritative_instead_of_code_prefix(self):
        self.replay.fly_codes = ["qmfva"]
        outcome = await self.add_fly_key()
        self.assertEqual(outcome["added"], [[b10_flykey_WORD, "qmfva"]], self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_existing_dictionary_code_allows_an_additional_fly_key_code(self):
        self.replay.entries.append({"word": b10_flykey_WORD, "code": "qmfv", "type": "Phrase"})
        outcome = await self.add_fly_key()
        self.assertEqual(outcome["added"], [[b10_flykey_WORD, "fmfva"]], self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_existing_draft_code_allows_an_additional_fly_key_code(self):
        await self.assert_pending_code_allows_add("Draft", "draft")

    async def test_submitted_code_allows_an_additional_fly_key_code(self):
        await self.assert_pending_code_allows_add("Pending", "submitted")

    assert_pending_code_allows_add = B10FlyKeyTests_assert_pending_code_allows_add

    async def test_no_fly_key_candidate_explains_and_never_calls_general_agent(self):
        self.replay.fly_codes = []
        outcome = await self.add_fly_key()
        self.assertEqual(outcome["added"], [])
        self.assertIn(b10_flykey_WORD, self.replay.last_response)
        self.assertRegex(self.replay.last_response, r"(?:没有|无).*(?:飞键)")
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.general.assert_not_awaited()

    async def test_all_occupied_fly_keys_offer_only_bound_selectable_fly_candidates(self):
        self.replay.entries.append({"word": "非默认候选", "code": "fmfva", "type": "Phrase"})
        outcome = await self.add_fly_key()
        self.assertEqual(outcome["added"], [])
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record)
        self.assertIsInstance(record.state, PendingAddWord)
        self.assertEqual([code for code, _occupied in record.state.candidates], list(FLY_CODES))
        self.assertTrue(self.replay.chat._advertised_reply_matches_live_record(self.replay.last_response, record))
        self.assertTrue(record.origin_prompt_digest)
        for code in b10_flykey_CODES[:3]:
            self.assertNotRegex(self.replay.last_response, rf"\d+\. {code}\b")
        self.replay.calls.clear()
        await self.replay.turn({"message": "换 3", "to_me": True}, raw_operation(
            op="select_candidate", operands=[], codes=[], refers_to_pending=True,
        ))
        self.assertIn("非默认候选", self.replay.last_response)
        self.assertNotIn("找麻烦", self.replay.last_response)
        self.assertNotIn("殖民者", self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_confirm_without_live_ticket_finishes_before_models_and_tools(self):
        await self.replay.turn({"message": "请执行，不要复读我的话", "to_me": True}, raw_operation(
            op="confirm", operands=[], codes=[], refers_to_pending=True,
        ))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.replay.trace[-1], "_stage_understand_operation_early")
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_confirm_variants_without_ticket_skip_model_and_other_actor_ticket(self):
        other = harness.ConversationAddress.group("qq", self.replay.author.space_id, "b10-other")
        self.assertTrue(self.replay.store.set(other, PendingAddWord("云翻", "yjfj", [("yjfj", False)])))
        record = self.replay.store.get_record(other)
        for message in ("确认", "请执行", "执行", "执行吧", "好的执行"):
            with self.subTest(message=message):
                await self.replay.turn({"message": message, "to_me": True}, raw_operation(
                    op="confirm", operands=[], codes=[], refers_to_pending=True,
                ))
                self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
                self.assertEqual(self.replay.calls, [])
                self.assertIs(self.replay.store.get_record(other), record)
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_recent_own_write_summary_is_returned_without_new_model_or_tools(self):
        with patch.object(self.replay.chat.time, "time", return_value=1000):
            outcome = await self.add_fly_key()
            self.assertEqual(outcome["added"], [[b10_flykey_WORD, "fmfva"]])
        self.replay.calls.clear()
        self.replay.client.create.reset_mock()
        with patch.object(self.replay.chat.time, "time", return_value=1119):
            await self.replay.turn({"message": "请执行", "to_me": True}, raw_operation(
                op="confirm", operands=[], codes=[], refers_to_pending=True,
            ))
        self.assertIn(f"上一条已执行：{b10_flykey_WORD} → fmfva", self.replay.last_response)
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_completed_receipt_expires_after_two_minutes(self):
        with patch.object(self.replay.chat.time, "time", return_value=1000):
            await self.add_fly_key()
        self.replay.calls.clear()
        self.replay.client.create.reset_mock()
        with patch.object(self.replay.chat.time, "time", return_value=1121):
            await self.replay.turn({"message": "执行吧", "to_me": True}, raw_operation(
                op="confirm", operands=[], codes=[], refers_to_pending=True,
            ))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_quoted_plain_write_receipt_is_not_an_invitation_to_submit(self):
        with patch.object(self.replay.chat.time, "time", return_value=1000):
            outcome = await self.add_fly_key()
            self.assertEqual(outcome["added"], [[b10_flykey_WORD, "fmfva"]])
        receipt = self.replay.last_response
        self.assertFalse(self.replay.chat.advertised_reply_contract(receipt).requires_live_state)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))
        self.replay.calls.clear()
        self.replay.writes.clear()
        self.replay.client.create.reset_mock()
        with patch.object(self.replay.chat.time, "time", return_value=1001):
            await self.replay.turn({
                "message": "确认", "to_me": True, "quote_target": receipt,
            }, raw_operation(op="confirm", operands=[], codes=[], refers_to_pending=True))
        self.assertEqual(self.replay.last_response, f"当前没有待确认的操作。\n上一条已执行：{b10_flykey_WORD} → fmfva")
        self.assertEqual(self.replay.calls, [])
        self.assertEqual(self.replay.writes, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_completed_receipt_belongs_only_to_its_actor(self):
        with patch.object(self.replay.chat.time, "time", return_value=1000):
            await self.add_fly_key()
        self.replay.author = harness.ConversationAddress.group("qq", self.replay.author.space_id, "b10-other")
        self.replay.memory = replace(self.replay.memory, user_id=self.replay.author.actor_id)
        self.replay.calls.clear()
        self.replay.client.create.reset_mock()
        with patch.object(self.replay.chat.time, "time", return_value=1001):
            await self.replay.turn({"message": "确认", "to_me": True}, raw_operation(
                op="confirm", operands=[], codes=[], refers_to_pending=True,
            ))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_history_claim_alone_does_not_create_a_recent_receipt(self):
        self.replay.history[:] = [{"role": "assistant", "content": f"已变更：{b10_flykey_WORD} → fmfva"}]
        await self.replay.turn({"message": "请执行", "to_me": True}, raw_operation(
            op="confirm", operands=[], codes=[], refers_to_pending=True,
        ))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_context_records_and_recent_submit_pointer_are_not_confirmation_tickets(self):
        states = (
            create_pending_trusted_word_record(b10_flykey_WORD, "qmfv", "Phrase"),
            PendingToolConfirm("keytao_submit_batch", {
                "batch_id": "offline-batch", "_recent_own_write": True,
                "_recent_batch_ids": ["offline-batch"],
            }),
        )
        for state in states:
            with self.subTest(state=type(state).__name__):
                self.assertTrue(self.replay.store.set(self.replay.author, state))
                record = self.replay.store.get_record(self.replay.author)
                await self.replay.turn({"message": "请执行", "to_me": True}, raw_operation(
                    op="confirm", operands=[], codes=[], refers_to_pending=True,
                ))
                self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
                self.assertIs(self.replay.store.get_record(self.replay.author), record)
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

    async def test_live_candidate_confirmation_keeps_its_existing_write_route(self):
        pending = PendingAddWord(
            b10_flykey_WORD, "fmfva", [("fmfva", False)], server_candidates=[("fmfva", False)],
            pronunciation_codes={"fmfva": "zhao meng fu"}, needs_manual_review=False,
        )
        self.assertTrue(self.replay.store.set(self.replay.author, pending))
        record = self.replay.store.get_record(self.replay.author)
        prompt = self.replay.chat._render_live_single_candidate_record(record)
        self.assertTrue(self.replay.store.bind_origin_prompt_digest(
            self.replay.author, self.replay.chat._prompt_capability_digest(prompt),
        ))
        self.replay.history[:] = [{"role": "assistant", "content": prompt}]
        outcome = await self.replay.turn({"message": "确认", "to_me": True}, raw_operation(
            op="confirm", operands=[b10_flykey_WORD], codes=[], refers_to_pending=True,
        ))
        self.assertEqual(outcome["added"], [[b10_flykey_WORD, "fmfva"]], self.replay.last_response)
        self.general.assert_not_awaited()

    async def test_semantically_understood_confirmation_has_no_further_model_call(self):
        await self.replay.turn({"message": "帮我执行刚才那项操作", "to_me": True}, raw_operation(
            op="confirm", operands=[], codes=[], refers_to_pending=True,
        ))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_awaited_once()
        self.general.assert_not_awaited()

    async def test_completed_receipt_is_not_replayed_into_another_group(self):
        with patch.object(self.replay.chat.time, "time", return_value=1000):
            await self.add_fly_key()
        self.replay.author = harness.ConversationAddress.group("qq", "b10-other-group", self.replay.author.actor_id)
        self.replay.memory = replace(self.replay.memory, space_id=self.replay.author.space_id)
        self.replay.calls.clear()
        self.replay.client.create.reset_mock()
        with patch.object(self.replay.chat.time, "time", return_value=1001):
            await self.replay.turn({"message": "确认", "to_me": True}, raw_operation(
                op="confirm", operands=[], codes=[], refers_to_pending=True,
            ))
        self.assertEqual(self.replay.last_response, "当前没有待确认的操作。")
        self.assertEqual(self.replay.calls, [])
        self.replay.client.create.assert_not_awaited()
        self.general.assert_not_awaited()

# Source: test_b13_default_reading.py
class B13DefaultReadingTests(unittest.IsolatedAsyncioTestCase):
    setUp = B13DefaultReadingTests_setUp

    async def test_no_evidence_polyphone_reply_has_default_candidates_and_manual_review(self):
        result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        response = harness.chat_commands_module._format_reviewed_add_prompt(result)
        self.assertIn("dfyvo", response, response)
        self.assertIn("dà féi yú，需管理员审核", response)
        self.assertEqual(result["autoReviewReason"], "未找到权威来源，仅使用编码服务默认读音")
        self.assertEqual(result["pronunciations"][0]["normalized"], ["da", "fei", "yu"])
        self.assertIs(result["autoReviewable"], False)
        self.assertIs(result["needsManualReview"], True)
        self.assertEqual(result["reviewDisposition"], "SEAL")
        self.fetch_encode.assert_awaited_once_with(None, "大肥鱼")

    async def test_no_evidence_keeps_only_default_encoder_reading_group(self):
        self.encode["candidateCodes"].append("dfyva")
        self.encode["alternatePhrasePronunciationCodes"] = [{
            "char": "大", "charIndex": 0, "pinyin": "dài", "codes": ["dfyva"],
        }]
        result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        response = harness.chat_commands_module._format_reviewed_add_prompt(result)
        self.assertIn("dfyvo", response, response)
        self.assertNotIn("dfyva", response, response)
        self.assertIn("需管理员审核", response)
        self.assertEqual([item["normalized"] for item in result["pronunciations"]],
                         [["da", "fei", "yu"]])
        self.assertEqual(result["reviewDisposition"], "SEAL")
        self.assertIs(result["autoReviewable"], False)
        self.fetch_encode.assert_awaited_once_with(None, "大肥鱼")

    async def test_weak_evidence_agreeing_with_default_keeps_manual_candidates(self):
        self.web.update(status="weak", candidates=[{"normalized": ["da", "fei", "yu"]}])
        result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        response = harness.chat_commands_module._format_reviewed_add_prompt(result)
        self.assertIn("dfyvo", response, response)
        self.assertIn("需管理员审核", response)
        self.assertEqual(result["reviewDisposition"], "SEAL")
        self.assertIs(result["autoReviewable"], False)

    async def test_semantic_needed_flag_alone_keeps_manual_candidates(self):
        self.encode["semanticPronunciationNeeded"] = True
        result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        response = harness.chat_commands_module._format_reviewed_add_prompt(result)
        self.assertIn("dfyvo", response, response)
        self.assertEqual(result["reviewDisposition"], "SEAL")
        self.assertIs(result["autoReviewable"], False)

    async def test_weak_evidence_contradicting_default_still_blocks(self):
        self.web.update(status="weak", candidates=[{"normalized": ["dai", "fei", "yu"]}])
        result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        self.assertEqual(result["reviewDisposition"], "BLOCK")
        self.assertEqual(result["recommendedCode"], "")

    async def test_valid_semantic_conflict_still_blocks(self):
        self.encode["candidateCodes"].append("dfyva")
        self.encode["alternatePhrasePronunciationCodes"] = [{
            "char": "大", "charIndex": 0, "pinyin": "dài", "codes": ["dfyva"],
        }]
        with patch.object(review, "_infer_semantic_pronunciation_for_review", AsyncMock(return_value={
            "accepted": True, "pinyins": ["dai", "fei", "yu"],
            "confidence": 0.8, "meaning": "一条体型肥大的鱼",
        })):
            result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        self.assertEqual(result["reviewDisposition"], "BLOCK")
        self.assertEqual(result["recommendedCode"], "")

    async def test_unbound_semantic_reading_does_not_block_default(self):
        with patch.object(review, "_infer_semantic_pronunciation_for_review", AsyncMock(return_value={
            "accepted": True, "pinyins": ["hong", "fei", "yu"],
            "confidence": 0.8, "meaning": "一条体型肥大的鱼",
        })):
            result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        self.assertEqual(result["reviewDisposition"], "SEAL")
        self.assertIn("dfyvo", result["pronunciations"][0]["codes"])

    async def test_web_disagreement_still_blocks_without_a_matching_alternative(self):
        self.web.update(status="disagreement", candidates=[
            {"normalized": ["da", "fei", "yu"]},
            {"normalized": ["dai", "fei", "yu"]},
        ])
        result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        self.assertEqual(result["pronunciationResolution"]["status"], "no_resolution")
        self.assertEqual(result["reviewDisposition"], "BLOCK")
        self.assertEqual(result["recommendedCode"], "")
        self.assertEqual(result["pronunciations"], [])

    async def test_encoder_context_conflict_still_blocks(self):
        self.encode["contextPhrasePinyins"] = ["dài", "féi", "yú"]
        result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        self.assertEqual(result["reviewDisposition"], "BLOCK")
        self.assertEqual(result["recommendedCode"], "")

    async def test_invalid_own_character_reading_still_blocks(self):
        self.encode["chars"][0]["pinyins"] = ["dài"]
        result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        self.assertEqual(result["reviewDisposition"], "BLOCK")
        self.assertEqual(result["recommendedCode"], "")

    async def test_evidence_read_error_is_not_retried(self):
        import httpx

        class ReadError(httpx.TransportError):
            pass

        with (
            patch.object(review.http_client, "guarded_fetch", AsyncMock(side_effect=ReadError())) as fetch,
            patch.object(review.http_client.asyncio, "sleep", AsyncMock()),
        ):
            with self.assertRaises(ReadError):
                await review._fetch_review_url("https://www.so.com/s", max_attempts=2)
        self.assertEqual(fetch.await_count, 1)

# Source: test_b27_reading.py
class B27ReadingTests(unittest.IsolatedAsyncioTestCase):
    tearDown = B25SessionTests_tearDown
    matcher_send = B19PartialReplyBindingTests_matcher_send
    matcher_finish = B19PartialReplyBindingTests_matcher_finish
    created = B19PartialReplyBindingTests_created
    history_for = B25SessionTests_history_for
    remember = B25SessionTests_remember
    greeting = B25SessionTests_greeting
    turn = B25SessionTests_turn

    setUp = B27ReadingTests_setUp

    reset_session = B27ReadingTests_reset_session

    seed = B27ReadingTests_seed

    finish_add = B27ReadingTests_finish_add

    async def test_k1_review_keeps_non_authoritative_encoder_reading(self):
        result = json.loads(await b25_session_commands.call_tool_function(
            "keytao_prepare_reviewed_add", {"word": b27_reading_WORD, "requested_code": ALTERNATE[0]},
            "qq", self.replay.author.actor_id,
        ))
        by_reading = {tuple(row["normalized"]): row for row in result["pronunciations"]}
        self.assertIn(("zhou", "nei"), by_reading, result)
        alternate = by_reading[("zhou", "nei")]
        self.assertEqual(alternate["codes"], ALTERNATE)
        self.assertTrue(alternate["requiresManualReview"])
        self.assertFalse(alternate["sources"], alternate)

    async def test_ordinary_add_keeps_authoritative_auto_review(self):
        self.http.entries = [row for row in self.http.entries if row["word"] != b27_reading_WORD]
        review = json.loads(await b25_session_commands.call_tool_function(
            "keytao_prepare_reviewed_add", {"word": b27_reading_WORD}, "qq", self.replay.author.actor_id,
        ))
        await self.turn("喵喵 加词 " + b27_reading_WORD, await self.greeting())
        self.assertEqual(self.http.writes, [])
        await self.turn("确认", self.bot.last_id)
        self.assertEqual(self.created(), [(b27_reading_WORD, "fdnsu")], self.replay.last_response)
        self.assertFalse(self.http.rows[0]["needsManualReview"], self.http.rows)
        self.assertTrue(review["autoReviewable"], review)

    async def test_k1_quoted_additional_code_creates_manual_review_item(self):
        await self.seed(b27_reading_WORD)
        await self.finish_add("加入编码 fdnw")

    async def test_quoted_authoritative_additional_code_keeps_auto_review(self):
        await self.seed(b27_reading_WORD)
        await self.turn("加入编码 fdnsui", self.original_id)
        if not self.http.writes:
            await self.turn("确认", self.bot.last_id)
        self.assertEqual(self.created(), [(b27_reading_WORD, "fdnsui")], self.replay.last_response)
        self.assertFalse(self.http.rows[0]["needsManualReview"], self.http.rows)
        self.assertEqual(
            self.http.rows[0]["remark"],
            "喵喵审词：读音 zhou na；权威来源、编码和常用度证据一致，允许本喵自动通过",
        )

    async def test_k1_wrong_code_still_refuses(self):
        await self.seed(b27_reading_WORD)
        await self.turn("加入编码 abcd", self.original_id)
        self.assertEqual(self.http.writes, [])
        self.assertIn("编码 abcd 的音码前缀不符", self.replay.last_response)
        self.assertIn("本次未写入", self.replay.last_response)

    modifier_adds = B27ReadingTests_modifier_adds

    async def test_k2_modifiers_route_additional_code(self):
        await self.modifier_adds("加入编码 fdnw")

    async def test_k2_modifiers_route_add_code(self):
        await self.modifier_adds("加入 fdnw")

    async def test_k2_modifiers_route_named_add(self):
        await self.modifier_adds("添加 周内 fdnw")

    async def test_k2_modifiers_preserve_exact_validation_refusal(self):
        for command in ("加入编码 abcd", "加入 abcd", "添加 周内 abcd"):
            self.reset_session()
            await self.seed(b27_reading_WORD)
            await self.turn(command, self.original_id)
            expected = self.replay.last_response
            self.assertIn("编码 abcd 的音码前缀不符", expected)
            for modifier in MODIFIERS:
                with self.subTest(command=command, modifier=modifier):
                    self.reset_session()
                    await self.seed(b27_reading_WORD)
                    await self.turn(modifier + command, self.original_id)
                    self.assertEqual(self.http.writes, [])
                    self.assertEqual(self.replay.last_response, expected)

    async def test_k3_existing_word_hint_uses_second_reading_first_code(self):
        await self.seed(b27_reading_WORD)
        self.assertIn("「周内」已在词库（fdnsu）", self.replay.last_response)
        self.assertIn("可再追加一个编码，格式为：加入编码 fdnw", self.replay.last_response)
        self.assertNotIn("<code>", self.replay.last_response)

    async def test_k3_existing_word_commonness_comparison_is_kept(self):
        await self.seed(b27_reading_WORD)
        self.assertIn("常用度对比：", self.replay.last_response)
        self.assertIn("「周纳」较「周内」更常用", self.replay.last_response)

    async def test_k3_without_second_reading_keeps_placeholder(self):
        self.http.extra_readings = []
        await self.seed(b27_reading_WORD)
        self.assertIn("可再追加一个编码，格式为：加入编码 <code>", self.replay.last_response)

    async def test_k3_single_reading_with_legacy_code_keeps_placeholder(self):
        self.http.extra_readings = []
        self.http.entries[0]["code"] = "fdnsuo"
        await self.seed(b27_reading_WORD)
        self.assertIn("可再追加一个编码，格式为：加入编码 <code>", self.replay.last_response)

    async def test_k3_multiple_other_readings_keep_placeholder(self):
        self.http.extra_readings.append(("ne", ["fdnf", "fdnfu", "fdnfui"]))
        await self.seed(b27_reading_WORD)
        self.assertIn("可再追加一个编码，格式为：加入编码 <code>", self.replay.last_response)

    async def test_other_speaker_cannot_use_owner_ticket(self):
        original = copy.deepcopy(await self.seed(b27_reading_WORD))
        for modifier in ("", *MODIFIERS):
            with self.subTest(modifier=modifier):
                await self.turn(modifier + "加入编码 fdnw", self.original_id, actor="b27-other-speaker")
                self.assertEqual(self.http.writes, [])
                self.assertEqual(self.replay.store.get_record(self.replay.author), original)
                self.assertIn("不是发给你的", self.replay.last_response)

    async def test_other_speaker_cannot_recover_owner_cold_quote(self):
        await self.seed(b27_reading_WORD)
        self.replay.store.delete(self.replay.author)
        for modifier in ("", *MODIFIERS):
            with self.subTest(modifier=modifier):
                await self.turn(modifier + "加入编码 fdnw", self.original_id, actor="b27-other-speaker")
                self.assertEqual(self.http.writes, [])
                self.assertIsNone(self.replay.store.get_record(self.replay.author))

# Source: test_hotfix_explicit_add.py
class ExplicitAddHotfixTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = ExplicitAddHotfixTests_asyncSetUp

    call = ExplicitAddHotfixTests_call

    execute = ExplicitAddHotfixTests_execute

    assert_created = ExplicitAddHotfixTests_assert_created

    def test_reproduction_identifies_unbound_alternate_candidates(self):
        inventory = select_candidate_inventory(reviewed_word())
        self.assertIsNotNone(inventory)
        self.assertEqual(inventory.readings["mtmw"], READING)
        self.assertEqual([code for code, _ in inventory.candidates if not inventory.readings.get(code)], ALTERNATE_CODES)

    async def test_production_model_route_creates_once_with_reviewed_reading(self):
        self.assert_created(await self.execute())

    async def test_plain_grammar_creates_once_with_reviewed_reading(self):
        self.assert_created(await self.execute("grammar"))

    async def test_status_layouts_keep_the_same_reading_binding(self):
        for layout in ("top_level", "per_group"):
            with self.subTest(layout=layout):
                self.review = reviewed_word(layout)
                self.calls.clear()
                self.writes.clear()
                self.store.delete(self.author)
                self.assert_created(await self.execute())

    async def test_unrelated_group_without_reading_does_not_block_requested_code(self):
        self.review = reviewed_word("per_group")
        self.review["pronunciations"][1]["pinyin"] = ""
        self.assert_created(await self.execute())

    async def test_code_outside_reviewed_groups_is_rejected(self):
        # A valid phonetic prefix alone must not authorize an unreviewed suffix.
        for route in ("model", "grammar"):
            with self.subTest(route=route):
                reply = await self.execute(route, code="mtmwi")
                self.assertEqual(self.writes, [], reply)
                self.assertIn("未写入", reply)
                self.assertFalse(any(name == "keytao_lookup_by_code" for name, _ in self.calls))

    async def test_shared_code_across_different_readings_uses_first_reviewed_reading(self):
        alternate = copy.deepcopy(self.review["pronunciations"][0])
        alternate.update(pinyin="mo tou mo wei", normalized=["mo", "tou", "mo", "wei"])
        self.review["pronunciations"].append(alternate)
        for route in ("model", "grammar"):
            with self.subTest(route=route):
                self.calls.clear()
                self.writes.clear()
                self.store.delete(self.author)
                self.assert_created(await self.execute(route))

    async def test_explicit_reading_resolves_a_shared_reviewed_code(self):
        alternate = copy.deepcopy(self.review["pronunciations"][0])
        alternate.update(pinyin="mo tou mo wei", normalized=["mo", "tou", "mo", "wei"])
        self.review["pronunciations"].append(alternate)
        reply = await self.commands._execute_explicit_entry_code_request(
            ExplicitEntryCodeRequest(hotfix_explicit_add_WORD, "mtmw", pinyin=READING),
            f"添加 {hotfix_explicit_add_WORD} {READING}:mtmw", "qq", self.author.actor_id,
            self.author, self.author.space_key, "Offline owner",
        )
        self.assert_created(reply)

    async def test_requested_candidate_without_a_reading_is_rejected(self):
        self.review["pronunciations"][0]["pinyin"] = ""
        reply = await self.execute()
        self.assertEqual(self.writes, [], reply)
        self.assertIn("缺少可核验的读音", reply)
        self.assertEqual([name for name, _ in self.calls], ["keytao_prepare_reviewed_add"])

    mark_resolvable_ambiguity = ExplicitAddHotfixTests_mark_resolvable_ambiguity

    async def test_ambiguous_review_with_unique_code_requires_manual_review(self):
        for route in ("model", "grammar"):
            with self.subTest(route=route):
                self.review = reviewed_word()
                self.mark_resolvable_ambiguity()
                self.calls.clear()
                self.writes.clear()
                self.store.delete(self.author)
                self.assert_created(await self.execute(route))
                self.assertIs(self.writes[0]["needs_manual_review"], True)
                self.assertIn("管理员复核", self.writes[0]["remark"])

    async def test_ambiguous_review_with_shared_code_requires_manual_review(self):
        alternate = copy.deepcopy(self.review["pronunciations"][0])
        alternate["pinyin"] = "mo tou mo wei"
        self.review["pronunciations"].append(alternate)
        self.mark_resolvable_ambiguity()
        for route in ("model", "grammar"):
            with self.subTest(route=route):
                self.calls.clear()
                self.writes.clear()
                self.store.delete(self.author)
                self.assert_created(await self.execute(route))
                self.assertIs(self.writes[0]["needs_manual_review"], True)
                self.assertIn("管理员复核", self.writes[0]["remark"])

    async def test_ambiguous_review_with_explicit_reading_requires_manual_review(self):
        alternate = copy.deepcopy(self.review["pronunciations"][0])
        alternate["pinyin"] = "mo tou mo wei"
        self.review["pronunciations"].append(alternate)
        self.mark_resolvable_ambiguity()
        reply = await self.commands._execute_explicit_entry_code_request(
            ExplicitEntryCodeRequest(hotfix_explicit_add_WORD, "mtmw", pinyin=READING),
            f"添加 {hotfix_explicit_add_WORD} {READING}:mtmw", "qq", self.author.actor_id,
            self.author, self.author.space_key, "Offline owner",
        )
        self.assert_created(reply)
        self.assertIs(self.writes[0]["needs_manual_review"], True)
        self.assertIn("管理员复核", self.writes[0]["remark"])

    async def test_unresolved_or_blocked_review_cannot_write(self):
        for change in (
            {"pronunciationUnresolved": True}, {"reviewDisposition": "BLOCK"},
            {"pronunciationUnresolved": True, "reviewDisposition": "BLOCK",
             "reviewVerdictSite": "pronunciation_unresolved", "multiSenseChoice": {"status": "ambiguous"},
             "lookupFailed": True},
        ):
            with self.subTest(change=change):
                self.review = {**reviewed_word(), **change}
                self.calls.clear()
                reply = await self.execute()
                self.assertEqual(self.writes, [], reply)
                self.assertEqual([name for name, _ in self.calls], ["keytao_prepare_reviewed_add"])

# Source: test_s55_single_review.py
class SingleReviewTests(unittest.IsolatedAsyncioTestCase):
    prepare = SingleReviewTests_prepare

    async def test_variant_missing_shape_keeps_authoritative_single_base_and_manual_seal(self):
        result = await self.prepare()
        self.assertTrue(result["success"])
        self.assertEqual(result["type"], "Single")
        self.assertEqual(result["encodingType"], "单字")
        self.assertEqual(result["pronunciations"][0]["codes"], ["qx"])
        self.assertEqual(result["pronunciations"][0]["normalized"], ["qiang"])
        self.assertEqual(result["recommendedCode"], "qx")
        self.assertIsNone(result["chars"][0]["shapeCode"])
        self.assertIn("枪", result["variantNote"])
        self.assertIn("异体", result["variantNote"])
        self.assertTrue(result["needsManualReview"])
        self.assertEqual(result["reviewDisposition"], "SEAL")

    async def test_occupancy_and_existing_are_single_table_only(self):
        result = await self.prepare(single_encoding(shape="io"), occupants={
            "qx": [{"word": "抢", "code": "qx", "type": "Single", "weight": 10}],
            "qxi": [{"word": "抢先", "code": "qxi", "type": "Phrase", "weight": 10}],
        }, existing=[
            {"word": "鎗", "code": "qxy", "type": "Phrase"},
            {"word": "鎗", "code": "qxio", "type": "Single"},
        ])
        self.assertEqual(result["recommendedCode"], "qxi")
        statuses = result["pronunciations"][0]["candidateStatuses"]
        self.assertEqual([(s["code"], s["occupied"]) for s in statuses], [("qx", True), ("qxi", False), ("qxio", False)])
        self.assertEqual([r["type"] for r in result["existing"]], ["Single"])

    async def test_missing_encoding_returns_one_truthful_line_without_candidates(self):
        result = await self.prepare({"success": False, "message": "not encodable"})
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "编码服务无法为「鎗」生成单字编码。")
        self.assertNotIn("\n", result["message"])
        self.assertNotIn("recommendedCode", result)

    async def test_transient_encoding_failure_is_labelled_temporary(self):
        result = await self.prepare({"success": False, "upstreamTransient": True})
        self.assertFalse(result["success"])
        self.assertIn("暂时不可用", result["message"])
        self.assertIn("稍后重试", result["message"])

    async def test_phrase_type_and_foreign_character_payload_fail_closed(self):
        for field, value in (("type", "二字词"), ("chars", [{"char": "枪"}])):
            encoding = single_encoding()
            encoding[field] = value
            self.assertFalse((await self.prepare(encoding))["success"])

    async def test_single_polyphone_uses_character_scheme_without_phrase_ladder(self):
        result = await self.prepare(single_encoding(readings=["qiāng", "chēng"]))
        self.assertEqual([r["codes"] for r in result["pronunciations"]], [["qx"]])
        result = await self.prepare(single_encoding(readings=["qiāng", "chēng"]), requested_reading="cheng")
        self.assertFalse(result["success"])
        encoding = single_encoding(readings=["qiāng", "chēng"])
        encoding["alternatePronunciationCodes"] = [{"pinyin": "chēng", "codes": ["jr"]}]
        result = await self.prepare(encoding, requested_reading="cheng")
        self.assertEqual(result["recommendedCode"], "jr")
        self.assertEqual(result["pronunciations"][0]["normalized"], ["cheng"])
        self.assertNotIn("variantNote", result)
        hinted = await self.prepare(encoding, requested_reading="鎗读chēng")
        self.assertEqual(hinted["recommendedCode"], "jr")

    async def test_foreign_phrase_chain_is_not_accepted_as_single_candidates(self):
        encoding = single_encoding()
        encoding["codes"] = ["qx", "qxvo"]
        self.assertFalse((await self.prepare(encoding))["success"])

    async def test_failed_occupancy_never_becomes_a_free_candidate(self):
        with patch.object(review, "fetch_keytao_encode", AsyncMock(return_value=single_encoding())), patch.object(
            review, "lookup_words", AsyncMock(return_value={}),
        ), patch.object(review, "lookup_codes", AsyncMock(side_effect=review.KeytaoApiError("fixture unavailable"))):
            result = await review.prepare_reviewed_word(review.ReviewHttpConfig("https://fixture.invalid", "fixture"), "鎗")
        self.assertFalse(result["success"])
        self.assertTrue(result["lookupFailed"])
        self.assertNotIn("recommendedCode", result)
        self.assertEqual(result["reviewDisposition"], "BLOCK")

    audit_single = SingleReviewTests_audit_single

    async def test_single_pre_submit_audit_preserves_real_character_basis(self):
        prepared = await self.prepare()
        audit, semantic_calls = await self.audit_single(prepared)
        self.assertFalse(audit["autoApprove"])
        self.assertEqual(semantic_calls, 0)
        self.assertIn(prepared["manualReviewReason"], " ".join(audit["issues"]))
        self.assertNotIn("整词", " ".join(audit["issues"]))
        self.assertTrue(audit["structuredManualReviewIssues"])
        self.assertEqual(audit.get("semanticContextAutoPassItems", []), [])

    async def test_single_audit_still_checks_duplicate_candidate_membership_and_occupancy(self):
        prepared = await self.prepare()
        duplicate = copy.deepcopy(prepared)
        duplicate["existing"] = [{"word": "鎗", "code": "qx", "type": "Single"}]
        unavailable = {**prepared, "lookupFailed": True}
        for value, code, expected in (
            (duplicate, "qx", "重复"),
            (prepared, "qxvo", "不在读音候选链"),
            (unavailable, "qx", review.LOOKUP_FAILURE_REASON),
        ):
            with self.subTest(expected=expected):
                audit, semantic_calls = await self.audit_single(value, code=code)
                self.assertFalse(audit["autoApprove"])
                self.assertIn(expected, " ".join(audit["issues"]))
                self.assertEqual(semantic_calls, 0)

# Source: test_s56_explicit_code.py
class ExplicitCodeTests(unittest.TestCase):
    def test_valid_explicit_forms_bind_to_live_review(self):
        state = reviewed_single()
        for message in ("加入编码qxioio", "加入 qxioio", "添加 鎗 qxioio", "用 qxioio"):
            with self.subTest(message=message):
                self.assertTrue(s56_explicit_code_routing.message_authorizes_live_pending_mutation(message, state))
                self.assertIsNone(s56_explicit_code_routing._pending_assent_rejection_response(state, message))

    def test_invalid_explicit_code_names_the_failure(self):
        for code, reason in (("abioio", "音码"), ("qxioioo", "6"), ("qx12", "小写字母")):
            with self.subTest(code=code):
                response = s56_explicit_code_routing._pending_assent_rejection_response(reviewed_single(), "加入编码" + code)
                self.assertIsNotNone(response)
                self.assertIn(reason, response)

    def test_wrong_word_and_reported_commands_never_bind(self):
        for message in ("添加 枪 qxioio", "他说加入编码qxioio", "不要加入编码qxioio", "加入编码qxioio？", "加入编码qxioio 删除强"):
            self.assertFalse(s56_explicit_code_routing.message_authorizes_live_pending_mutation(message, reviewed_single()), message)

    def test_numeric_explicit_codes_are_not_ordinal_selectors(self):
        from keytao_bot.utils.explicit_code import parse_explicit_code_request
        for message in ("加入编码12", "加入 编码 12", "添加 鎗 12"):
            with self.subTest(message=message):
                reply = s56_explicit_code_routing._pending_assent_rejection_response(reviewed_single(), message)
                self.assertIn("只能包含小写字母", reply)
                self.assertFalse(s56_explicit_code_routing.message_authorizes_live_pending_mutation(message, reviewed_single()))
        for message in ("加入 2", "用 2"):
            self.assertIsNone(parse_explicit_code_request(message, "鎗"))

    def test_explicit_code_executes_sealed_without_shifting(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s56-explicit")
            s56_explicit_code_chat.conversation_state_store.set(key, reviewed_single())
            execute = AsyncMock(return_value="done")
            shift = AsyncMock(side_effect=AssertionError("unexpected shift"))
            with patch.object(s56_explicit_code_chat, "call_tool_function", AsyncMock(return_value=json.dumps({"success": True, "phrases": []}))), patch.object(
                s56_explicit_code_commands, "_execute_add_to_draft", execute,
            ), patch.object(s56_explicit_code_commands, "_execute_shift_to_code", shift):
                reply = await s56_explicit_code_commands.handle_pending_message_core("加入编码qxioio", "qq", "s56-explicit", key, allow_intent_model=False)
            self.assertIn("形码 ioio 未能核验，需管理员复核", reply)
            self.assertTrue(reply.endswith("done"))
            self.assertEqual(execute.await_count, 1)
            self.assertEqual(execute.call_args.args[0:2], ("鎗", "qxioio"))
            self.assertIn("形码 ioio 未能核验，需管理员复核", execute.call_args.args[6])
            self.assertIs(execute.call_args.args[7], True)
            self.assertEqual(execute.call_args.kwargs["reviewed_pinyin"], "qiāng")
            self.assertIn("qxioio", execute.call_args.kwargs["reviewed_candidate_codes"])
        asyncio.run(run())

    def test_alternate_phonetic_base_is_checked_against_reviewed_reading(self):
        from keytao_bot.utils.explicit_code import validate_explicit_code
        state = harness.PendingAddWord(
            word="哲思", recommended_code="fesk", candidates=[("fesk", False)],
            server_candidates=[("fesk", False)], pronunciation_codes={"fesk": "zhé sī"},
        )
        self.assertTrue(validate_explicit_code(state, "feskio").valid)
        alternate = validate_explicit_code(state, "qeskio")
        self.assertTrue(alternate.valid)
        self.assertIn("管理员复核", alternate.manual_reason)
        self.assertFalse(validate_explicit_code(state, "zeskio").valid)

    def test_bare_assent_preserves_protected_record_and_named_eviction_executes(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s56-protected")
            for text in ("加入", "加入并提交", "好"):
                state = reviewed_single()
                state.recommended_code = ""
                s56_explicit_code_chat.conversation_state_store.set(key, state)
                shift = AsyncMock(side_effect=AssertionError("unexpected shift"))
                with patch.object(s56_explicit_code_commands, "_execute_shift_to_code", shift):
                    reply = await s56_explicit_code_commands.handle_pending_message_core(text, "qq", "s56-protected", key, allow_intent_model=False)
                self.assertIn("本次未写入", reply)
                self.assertIn("强", reply)
                self.assertIn("完整编码", reply)
                record = s56_explicit_code_chat.conversation_state_store.get_record(key)
                self.assertIsNotNone(record)
                self.assertFalse(record.execution_id)
                self.assertTrue(s56_explicit_code_routing.message_authorizes_live_pending_mutation("加入，顶替 强", record.state))
                self.assertTrue(s56_explicit_code_chat._advertised_reply_matches_live_record(reply, record), reply)
                self.assertEqual(s56_explicit_code_chat._enforce_advertised_reply_contract(reply, key), reply)
            shift = AsyncMock(return_value="shifted")
            with patch.object(s56_explicit_code_commands, "_execute_shift_to_code", shift):
                reply = await s56_explicit_code_commands.handle_pending_message_core("加入，顶替 强", "qq", "s56-protected", key, allow_intent_model=False)
            self.assertEqual(reply, "shifted")
            self.assertEqual(shift.await_count, 1)
            self.assertEqual(shift.call_args.args[:2], ("鎗", "qx"))
            self.assertEqual(shift.call_args.kwargs["target_item"]["type"], "Single")
        asyncio.run(run())

# Source: test_s56_security.py
class S56SecurityTests(unittest.TestCase):
    def test_same_turn_retry_using_exact_command_is_not_advertised(self):
        message = "将草稿中「亮面」lxmmov 的权重调整为 101"
        reason = "当前草稿里还没有任何条目，找不到「亮面」（lxmmov），所以无法调整权重。"
        for retry in ("之后再用", "之后再次使用"):
            with self.subTest(retry=retry):
                raw = f"{reason}\n先往草稿里加上这个条目，{retry}「{message}」即可。"
                reply = harness.AgentOrchestrator._finalize_reply(message, raw, {})
                self.assertNotIn(message, reply)
                self.assertIn("没有", reply)
        for reply in (reason, f"收到「{message}」，但草稿为空。", "先补充条目后，再用「查看草稿」核对。", "当前草稿为空；添加条目后再用「查看草稿」核对。"):
            self.assertEqual(reply, harness.AgentOrchestrator._finalize_reply(message, reply, {}))

    def test_binding_meta_redraw_uses_existing_plain_answer(self):
        raw = (
            "会的。所有词库操作（加词、草稿、提交）都以你绑定键道账号为前提"
            "——未绑定的话工具调用会直接失败，所以我会先提示你完成绑定。"
        )
        for message, expected in (
            ("你是否会先确认对方有没有绑定账号？", "会的：写入前会校验绑定；未绑定会给出绑定引导。"),
            ("解释候选编码", "这次没有生成可发送的回复；本次未写入。"),
        ):
            with self.subTest(message=message):
                token = s56_security_chat._current_turn_message.set(message)
                try:
                    self.assertEqual(expected, s56_security_chat._prepare_user_facing_reply(raw, None))
                finally:
                    s56_security_chat._current_turn_message.reset(token)

    def test_advertised_shift_cancel_closes_only_the_ticket_without_model_or_write(self):
        async def run():
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s56-shift-cancel")
            state = harness.PendingToolConfirm(
                function_name="keytao_shift_phrase_code",
                args={"word": "发布会", "target_code": "fbh"},
                confirmation_source="local_preview",
            )
            store.set(key, state)
            with patch.object(s56_security_chat, "conversation_state_store", store), patch.object(
                s56_security_chat, "call_tool_function", AsyncMock(side_effect=AssertionError("cancel wrote")),
            ), patch.object(s56_security_commands, "_classify_message_command_intent", AsyncMock(
                side_effect=AssertionError("cancel used intent model"),
            )):
                response = await s56_security_commands.handle_pending_message_core(
                    "取消", "qq", "s56-shift-cancel", key, allow_intent_model=False,
                )
            self.assertEqual("已取消。", response)
            self.assertIsNone(store.get_record(key))
            for text in ("取消？", "他说：取消", "不要取消", "请解释取消"):
                intent = s56_security_routing._pending_tool_assent_intent(state, text)
                self.assertFalse(intent and intent.intent == "pending_cancel", text)
        asyncio.run(run())

    def test_default_front_insert_retains_manual_seal_after_auto_review(self):
        async def run():
            state = reviewed_single()
            state.word = "发布会"
            state.phrase_type = "Phrase"
            state.needs_manual_review = False
            state.candidates = [("fbh", True), ("fbhu", False)]
            state.server_candidates = list(state.candidates)
            state.recommended_code = "fbh"
            state.occupied_words = {"fbh": ["重病号"]}
            state.server_occupied_words = dict(state.occupied_words)
            state.pronunciation_codes = {code: "fa bu hui" for code, _ in state.candidates}
            state.code_remarks = {"fbh": "喵喵审词：读音 fa bu hui；来源 汉典"}
            state.server_ordering_assessments = [{
                "newWord": "发布会", "occupantWord": "重病号", "occupantCode": "fbh",
                "freeCode": "fbhu", "newCode": "fbh", "recommendedCode": "fbh",
                "verdict": "front_more_common",
            }]
            for message in ("加入", "加入并提交"):
                with self.subTest(message=message):
                    shift = AsyncMock(return_value="sealed shift")
                    with patch.object(s56_security_commands, "_execute_shift_to_code", shift):
                        reply = await s56_security_commands._handle_pending_add_word(
                            state, message, "qq", "s56-front-seal", [],
                            command_intent=harness.MessageCommandIntent(
                                intent="pending_add_and_submit" if "提交" in message else "pending_confirm",
                                confidence=1.0, submit_after="提交" in message,
                            ),
                        )
                    self.assertEqual("sealed shift", reply)
                    self.assertIs(shift.call_args.kwargs["target_item"]["needsManualReview"], True)
                    self.assertEqual("Phrase", shift.call_args.kwargs["target_item"]["type"])
                    self.assertEqual(state.code_remarks["fbh"], shift.call_args.kwargs["target_item"]["remark"])
                    self.assertEqual("fa bu hui", shift.call_args.kwargs["reviewed_pinyin"])
        asyncio.run(run())

    def test_complete_explicit_code_uses_early_live_pending_replacement_route(self):
        async def run():
            state = reviewed_single()
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s56-complete-stage")
            store.set(key, state)
            execute = AsyncMock(return_value="early pending route")
            with patch.object(s56_security_chat, "conversation_state_store", store), patch.object(s56_security_commands, "handle_pending_message_core", execute):
                response = await s56_security_commands._try_handle_explicit_pending_replacement(state, "添加 鎗 qxioio", "qq", "s56-complete-stage", key)
            self.assertEqual("early pending route", response)
            self.assertEqual(1, execute.await_count)
            self.assertFalse(execute.call_args.kwargs["allow_intent_model"])
        asyncio.run(run())

    def test_unknown_shape_seal_reaches_real_sink_validator_and_cannot_be_forged(self):
        async def run():
            with patch.object(s56_security_chat, "call_tool_function", AsyncMock(return_value=json.dumps({"success": True, "phrases": []}))):
                derived, error = await s56_security_commands._bind_explicit_pending_code(reviewed_single(), ExplicitCodeRequest("qxioio"), "qq", "s56-seal")
            self.assertFalse(error)
            args = s56_security_commands._create_phrase_args(derived, "qxioio")
            pinyin, codes = s56_security_commands._pending_reviewed_reading(derived, "qxioio")
            capability = s56_security_commands._reviewed_create_capability("鎗", "qxioio", pinyin, codes)
            trusted = s56_security_commands.tool_executor.canonicalize_arguments(
                "keytao_create_phrase", args,
                ToolContext("qq", "s56-seal", trusted_reviewed_items_by_key=capability),
            )
            item = {"action": "Create", "word": "鎗", "code": "qxioio", "type": "Single", "remark": trusted.get("remark"), "needs_manual_review": trusted.get("needs_manual_review")}
            with patch.object(harness._draft_tools, "_fetch_encode_candidates", AsyncMock(side_effect=AssertionError("known reviewed capability re-encoded"))):
                validation = await harness._draft_tools._validate_draft_item_code(item, reviewed_pinyin=trusted.get("_reviewed_pinyin", ""), reviewed_candidate_codes=trusted.get("_reviewed_candidate_codes"))
            self.assertTrue(validation["success"])
            harness._draft_tools._stamp_item_review_flag(item, validation)
            self.assertTrue(item["needsManualReview"])
            self.assertIn("形码 ioio 未能核验", item["remark"])
            forged = s56_security_commands.tool_executor.canonicalize_arguments("keytao_create_phrase", args, ToolContext("qq", "s56-foreign"))
            self.assertNotIn("_reviewed_pinyin", forged)
            self.assertNotIn("_reviewed_candidate_codes", forged)
            with patch.object(harness._draft_tools, "_fetch_encode_candidates", AsyncMock(return_value={"success": True, "candidateCodes": ["qx"]})):
                rejected = await harness._draft_tools._validate_draft_item_code(item, reviewed_pinyin=forged.get("_reviewed_pinyin", ""), reviewed_candidate_codes=forged.get("_reviewed_candidate_codes"))
            self.assertFalse(rejected["success"])
        asyncio.run(run())

    def test_invalid_or_framed_input_never_authorizes_extension(self):
        source = reviewed_single()
        for message in (
            "加入编码abioio", "加入编码qxioioo", "加入编码qx12", "加入编码QXioio",
            "加入编码qxioio 删除 强", "加入编码qxioio，添加 枪 qxi", "加入编码qxioio吗",
            "他说：加入编码qxioio", "请解释‘加入编码qxioio’", "不要加入编码qxioio",
            "加入编码qxioio？", "如果需要就加入编码qxioio", "可以加入编码qxioio吗？",
        ):
            with self.subTest(message=message):
                self.assertFalse(s56_security_routing.message_authorizes_live_pending_mutation(message, source))
        source.server_candidates = []
        self.assertFalse(s56_security_routing.message_authorizes_live_pending_mutation("加入编码qxioio", source))

    def test_same_group_other_actor_has_no_record_or_lookup(self):
        async def run():
            store = harness.MemoryConversationStateStore()
            owner = harness.ConversationAddress.group("qq", "s56-group", "s56-owner")
            other = harness.ConversationAddress.group("qq", "s56-group", "s56-other")
            store.set(owner, reviewed_single(), space_key=owner.space_key)
            with patch.object(s56_security_chat, "conversation_state_store", store), patch.object(
                s56_security_chat, "call_tool_function", AsyncMock(side_effect=AssertionError("foreign actor lookup")),
            ):
                reply = await s56_security_commands.handle_pending_message_core(
                    "加入编码qxioio", "qq", "s56-other", other,
                    space_key=other.space_key, allow_intent_model=False,
                )
            self.assertIsNone(reply)
            self.assertIsNotNone(store.get_record(owner))
        asyncio.run(run())

    def test_exact_six_code_duplicate_does_not_write(self):
        async def run():
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s56-duplicate")
            store.set(key, reviewed_single())
            with patch.object(s56_security_chat, "conversation_state_store", store), patch.object(s56_security_chat, "call_tool_function", AsyncMock(
                return_value=json.dumps({"success": True, "phrases": [{"word": "鎗", "code": "qxioio", "type": "Single", "weight": 10}]}),
            )), patch.object(s56_security_commands, "_execute_add_to_draft", AsyncMock(side_effect=AssertionError("duplicate write"))):
                reply = await s56_security_commands.handle_pending_message_core("加入编码qxioio", "qq", "s56-duplicate", key, allow_intent_model=False)
            self.assertIn("不能重复添加", reply)
            self.assertEqual([("qx", True)], store.get_record(key).state.server_candidates)
        asyncio.run(run())

    def test_empty_multi_selection_does_not_inherit_protected_recommendation(self):
        async def run():
            state = reviewed_single()
            state.candidates = [("qx", True), ("qxi", False), ("qxio", False)]
            state.server_candidates = list(state.candidates)
            state.pronunciation_codes = {code: "qiāng" for code, _ in state.candidates}
            key = harness.ConversationAddress.private("qq", "s56-multiple-empty")
            store = harness.MemoryConversationStateStore()
            store.set(key, state)
            execute = AsyncMock(return_value="empty choices written")
            with patch.object(s56_security_chat, "conversation_state_store", store), patch.object(
                s56_security_commands, "_execute_add_multiple_codes_to_draft", execute,
            ):
                reply = await s56_security_commands.handle_pending_message_core("添加2、3", "qq", "s56-multiple-empty", key, allow_intent_model=False)
            self.assertEqual("empty choices written", reply)
            self.assertEqual(["qxi", "qxio"], execute.call_args.args[1])
        asyncio.run(run())

    def test_multiword_query_keeps_blocked_item_review_and_other_safe_candidate(self):
        async def run(message="鎗 大端", all_blocked=False):
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s56-multiword")
            protected = single_review()
            protected["recommendedCode"] = ""
            pronunciation = protected["pronunciations"][0]
            pronunciation["recommendedCode"] = ""
            pronunciation["candidateStatuses"] = [{"code": "qx", "occupied": True, "words": ["强"], "label": "已有「强」"}]
            protected["candidateOrderingAssessments"] = [{"newWord": "鎗", "occupantWord": "强", "occupantCode": "qx", "freeCode": "", "newCode": "", "verdict": "behind_more_common"}]

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    if len(args["word"]) == 1:
                        payload = json.loads(json.dumps(protected))
                        payload["word"] = args["word"]
                        payload["candidateOrderingAssessments"][0]["newWord"] = args["word"]
                        return json.dumps(payload)
                    return json.dumps(s54_multiword_review_payload(args["word"]))
                raise AssertionError(name)

            with patch.object(s56_security_chat, "conversation_state_store", store), patch.object(
                s56_security_chat, "call_tool_function", side_effect=tool,
            ), patch.object(s56_security_commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)):
                reply = await s56_security_commands._try_handle_simple_single_word_query(message, "qq", "s56-multiword", key)
            self.assertIn("审词：读音 qiāng", reply)
            if all_blocked:
                self.assertIn("「槍」本次未选中", reply)
                self.assertIsNone(store.get_record(key))
                self.assertEqual((), advertised_single_word_candidate_codes(reply))
                self.assertEqual((), advertised_command_suggestions(reply))
                with patch.object(s56_security_chat, "conversation_state_store", store):
                    self.assertEqual(reply, s56_security_chat._enforce_advertised_reply_contract(reply, key))
                return
            self.assertIn("大端", reply)
            self.assertIn("dsdtvo", reply)
            self.assertNotIn("展示校验未通过", reply)
            self.assertIn("「鎗」本次未选中", reply)
            self.assertIn("qx — 已有「强」", reply)
            self.assertNotIn("1. qx", reply)
            self.assertNotIn("添加 鎗", reply)
            self.assertNotIn("加入编码 qx+形码", reply)
            record = store.get_record(key)
            self.assertEqual(["大端"], [item["word"] for item in record.state.args["items"]])
            with patch.object(s56_security_chat, "conversation_state_store", store):
                self.assertTrue(s56_security_chat._advertised_reply_matches_live_record(reply, record))
                self.assertEqual(reply, s56_security_chat._enforce_advertised_reply_contract(reply, key))
        asyncio.run(run())
        asyncio.run(run("鎗 槍", all_blocked=True))

# Source: test_s57_extra_code.py
class ExtraCodeTests(unittest.TestCase):
    def test_additional_code_warning_rejects_foreign_targets_mixed_risks_and_incomplete_cas(self):
        async def run():
            cases = (
                ("foreign-word", lambda p: p["warnings"][0]["item"].update(word="咽")),
                ("foreign-code", lambda p: p["warnings"][0]["item"].update(code="yeoiu")),
                ("foreign-type", lambda p: p["warnings"][0]["item"].update(type="Phrase")),
                ("foreign-existing-word", lambda p: p["warnings"][0]["existing"].update(word="咽")),
                ("foreign-existing-code", lambda p: p["warnings"][0]["existing"].update(code="yeoiu")),
                ("duplicate-identity", lambda p: p["warnings"][0]["existing"].update(code="yeoia")),
                ("foreign-existing-type", lambda p: p["warnings"][0]["existing"].update(type="Phrase")),
                ("changed-remark", lambda p: p["warnings"][0]["item"].update(remark="")),
                ("lost-seal", lambda p: p["warnings"][0]["item"].update(needsManualReview=False)),
                ("mixed-risk", lambda p: p["warnings"].append({**p["warnings"][0], "warningType": "duplicate_code"})),
                ("missing-digest", lambda p: p.pop("warningDigest")),
                ("boolean-version", lambda p: p.update(contentVersion=True)),
                ("stale-version", lambda p: p.update(contentVersionConflict=True)),
            )
            for label, mutate in cases:
                with self.subTest(label=label):
                    key = harness.ConversationAddress.private("qq", "s57-warning-" + label)
                    s57_extra_code_chat.conversation_state_store.delete(key)
                    warning = additional_code_warning()
                    mutate(warning)
                    creates = []

                    async def tool(name, args, *_args, **kwargs):
                        if name == "keytao_prepare_reviewed_add":
                            return json.dumps(existing_single_review())
                        if name == "keytao_lookup_by_code":
                            return json.dumps({"success": True, "phrases": []})
                        if name == "keytao_create_phrase":
                            creates.append(dict(args))
                            return json.dumps(warning)
                        raise AssertionError(name)

                    with patch.object(s57_extra_code_chat, "call_tool_function", side_effect=tool):
                        await s57_extra_code_commands.try_handle_explicit_entry_code_command(
                            '添加单字"嘢"，编码为"yeoia"', "qq", key.actor_id, key,
                        )
                    self.assertEqual(len(creates), 1)
                    self.assertIsNot(creates[0].get("confirmed"), True)
        asyncio.run(run())

    def test_additional_code_capability_is_cleared_when_confirmed_execution_raises(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-warning-error")
            s57_extra_code_chat.conversation_state_store.delete(key)

            async def tool(name, args, *_args, **kwargs):
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(existing_single_review())
                if name == "keytao_lookup_by_code":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_create_phrase":
                    if args.get("confirmed"):
                        raise RuntimeError("fixture interrupted confirmed dispatch")
                    return json.dumps(additional_code_warning())
                raise AssertionError(name)

            with patch.object(s57_extra_code_chat, "call_tool_function", side_effect=tool):
                with self.assertRaisesRegex(RuntimeError, "fixture interrupted"):
                    await s57_extra_code_commands.try_handle_explicit_entry_code_command(
                        '添加单字"嘢"，编码为"yeoia"', "qq", key.actor_id, key,
                    )
            self.assertIsNone(s57_extra_code_commands._current_explicit_additional_code.get())
            self.assertFalse(s57_extra_code_commands._create_preview_can_auto_confirm(
                additional_code_warning(), {"word": "嘢", "code": "yeoia"},
            ))
        asyncio.run(run())

    def test_real_create_executor_replays_exact_multiple_code_warning_only_for_explicit_request(self):
        async def run():
            for explicit_request, expected_writes in ((True, 1), (False, 0)):
                key = harness.ConversationAddress.private("qq", "s57-multiple-warning")
                s57_extra_code_chat.conversation_state_store.delete(key)
                creates = []

                async def tool(name, args, *_args, **kwargs):
                    if name == "keytao_prepare_reviewed_add":
                        return json.dumps(existing_single_review())
                    if name == "keytao_lookup_by_code":
                        return json.dumps({"success": True, "phrases": []})
                    if name == "keytao_create_phrase":
                        creates.append((dict(args), kwargs))
                        if args.get("confirmed") is True:
                            return json.dumps({"success": True, "batchId": "s57-existing-batch", "contentVersion": 2})
                        return json.dumps(additional_code_warning())
                    raise AssertionError(name)

                with patch.object(s57_extra_code_chat, "call_tool_function", side_effect=tool), patch.object(
                    s57_extra_code_commands, "_format_draft_response", AsyncMock(return_value="Single 嘢 yeoia"),
                ):
                    if explicit_request:
                        reply = await s57_extra_code_commands.try_handle_explicit_entry_code_command(
                            '添加单字"嘢"，编码为"yeoia"', "qq", key.actor_id, key,
                        )
                    else:
                        reply = await s57_extra_code_commands._execute_add_to_draft(
                            "嘢", "yeoia", "qq", key.actor_id,
                            reviewed_pinyin="yě", reviewed_candidate_codes=("ye", "yeoia"),
                        )
                actual_writes = [call for call in creates if call[0].get("confirmed") is True]
                self.assertEqual(len(actual_writes), expected_writes, reply)
                self.assertTrue(creates[0][0].get("preview_only"))
                if explicit_request:
                    self.assertEqual(len(creates), 2)
                    arguments, kwargs = actual_writes[0]
                    self.assertEqual((arguments["word"], arguments["code"]), ("嘢", "yeoia"))
                    self.assertEqual(arguments["batch_id"], "s57-existing-batch")
                    self.assertEqual(arguments["expected_content_version"], 1)
                    self.assertEqual(arguments["expected_warning_digest"], additional_code_warning()["warningDigest"])
                    self.assertEqual(kwargs["trusted_reviewed_items_by_key"][("嘢", "yeoia")]["type"], "Single")
                    self.assertIn("形码 oia 未能核验", arguments["remark"])
                    self.assertTrue(arguments["needs_manual_review"])
                    self.assertIn("已", reply)
                else:
                    self.assertIn("确认", reply)
        asyncio.run(run())

    def test_quoted_and_natural_forms_bind_to_the_same_word(self):
        for message in (
            '添加单字"嘢"，编码为"yeoia"', '添加 单字 “嘢” ，编码为 “yeoia”',
            '添加单字「嘢」，编码为「yeoia」', '添加单字＂嘢＂，编码为＂yeoia＂',
            '给 嘢 加一个码 yeoia', '嘢 也放到 yeoia', '加入编码yeoia',
        ):
            with self.subTest(message=message):
                request = parse_explicit_code_request(message, "嘢")
                self.assertIsNotNone(request, message)
                self.assertEqual(request.code, "yeoia")
                state = harness.PendingAddWord(
                    word="嘢", recommended_code="ye", candidates=[("ye", False)],
                    server_candidates=[("ye", False)], pronunciation_codes={"ye": "yě"},
                    phrase_type="Single",
                )
                self.assertTrue(s57_extra_code_routing.message_authorizes_live_pending_mutation(message, state), message)
                self.assertIsNone(s57_extra_code_routing._pending_assent_rejection_response(state, message))

    def test_explicit_submit_suffix_is_not_part_of_the_code(self):
        for message in ('添加单字"嘢"，编码为"yeoia"并提交', "给 嘢 加一个码 yeoia 并提交"):
            request = parse_explicit_code_request(message, "嘢")
            self.assertEqual(request.code, "yeoia")
            self.assertTrue(request.submit_after)

    def test_wrong_word_and_non_commands_never_gain_explicit_authority(self):
        for message in (
            '添加单字"咽"，编码为"yeoia"', '他说添加单字"嘢"，编码为"yeoia"',
            '不要给 嘢 加一个码 yeoia', '嘢 也放到 yeoia？',
            '给 嘢 加一个码 yeoia 删除也', '添加单字"嘢"，编码为"yeoia”，提交',
        ):
            self.assertIsNone(parse_explicit_code_request(message, "嘢"), message)

    def test_cold_existing_single_add_uses_new_code_occupancy_and_manual_seal(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-extra-cold")
            s57_extra_code_chat.conversation_state_store.delete(key)
            calls = []

            async def tool(name, args, *_args, **_kwargs):
                calls.append((name, args))
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(existing_single_review())
                if name == "keytao_lookup_by_code":
                    return json.dumps({"success": True, "phrases": []})
                raise AssertionError(name)

            execute = AsyncMock(return_value="已加入草稿。")
            with patch.object(s57_extra_code_chat, "call_tool_function", side_effect=tool), patch.object(
                s57_extra_code_commands, "_execute_add_to_draft", execute,
            ):
                reply = await s57_extra_code_commands.try_handle_explicit_entry_code_command(
                    '添加单字"嘢"，编码为"yeoia"', "qq", "s57-extra-cold", key,
                )
            self.assertIn("已加入草稿", reply)
            self.assertIn("形码 oia 未能核验，需管理员复核", reply)
            self.assertNotIn("也", reply)
            self.assertNotIn("耶", reply)
            self.assertEqual(execute.await_count, 1)
            self.assertEqual(execute.call_args.args[:2], ("嘢", "yeoia"))
            self.assertIs(execute.call_args.args[7], True)
            self.assertEqual(execute.call_args.kwargs["reviewed_pinyin"], "yě")
            self.assertIn("yeoia", execute.call_args.kwargs["reviewed_candidate_codes"])
            self.assertEqual([args["code"] for name, args in calls if name == "keytao_lookup_by_code"], ["yeoia"])
        asyncio.run(run())

    def test_review_identity_and_wrong_prefix_refuse_before_occupancy(self):
        async def run():
            for changes, code, expected in (
                ({"word": "咽"}, "yeoia", "已审读音"),
                ({"type": "Phrase"}, "yeoia", "正确词条类型"),
                ({}, "aboia", "音码前缀"),
                ({}, "yeoiaaa", "最多 6 位"),
                ({}, "yeoi1", "小写字母"),
            ):
                key = harness.ConversationAddress.private("qq", "s57-extra-invalid")
                s57_extra_code_chat.conversation_state_store.delete(key)
                tool = AsyncMock(return_value=json.dumps({**existing_single_review(), **changes}))
                execute = AsyncMock(side_effect=AssertionError("unexpected write"))
                with patch.object(s57_extra_code_chat, "call_tool_function", tool), patch.object(s57_extra_code_commands, "_execute_add_to_draft", execute):
                    reply = await s57_extra_code_commands.try_handle_explicit_entry_code_command(
                        f'添加单字"嘢"，编码为"{code}"', "qq", key.actor_id, key,
                    )
                self.assertIn(expected, reply)
                self.assertEqual(tool.await_count, 1)
                self.assertEqual(execute.await_count, 0)
        asyncio.run(run())

    def test_multi_code_existing_context_survives_reload_without_row_action_authority(self):
        async def run(db_path):
            store = SQLiteConversationStateStore(db_path=db_path)
            key = harness.ConversationAddress.private("qq", "s57-extra-multiple")

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": [
                        {"word": "嘢", "code": code, "type": "Single", "weight": 11}
                        for code in ("yeoiav", "yeoi")
                    ]})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                raise AssertionError(name)

            with patch.object(s57_extra_code_chat, "conversation_state_store", store), patch.object(
                s57_extra_code_chat, "call_tool_function", side_effect=tool,
            ), patch.object(s57_extra_code_chat, "_classify_simple_word_query_intent", AsyncMock(
                return_value=harness.SimpleWordQueryIntent(True, ("嘢",), "word_lookup", 1.0),
            )):
                reply = await s57_extra_code_commands._try_handle_simple_single_word_query("嘢", "qq", key.actor_id, key)
            reloaded = SQLiteConversationStateStore(db_path=db_path)
            record = reloaded.get_record(key)
            self.assertTrue(record.state.context_only)
            self.assertTrue(trusted_word_record_is_complete(record.state))
            self.assertFalse(trusted_word_record_is_complete(replace(record.state, context_only=False)))
            self.assertFalse(trusted_word_record_is_complete(replace(record.state, context_only=1)))
            self.assertIn("追加一个编码", reply)
            self.assertTrue(s57_extra_code_routing.message_authorizes_live_pending_mutation("加入编码yeoia", record.state))
            for action in ("删除", "换码", "确认", "硬来"):
                self.assertFalse(s57_extra_code_routing.message_authorizes_live_pending_mutation(action, record.state), action)
            self.assertTrue(s57_extra_code_chat._advertised_reply_matches_live_record(reply, record))
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(run(directory + "/state.db"))

    def test_occupied_new_code_preserves_guard_and_force_binds_only_selected_slot(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-extra-protected")
            s57_extra_code_chat.conversation_state_store.delete(key)

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(existing_single_review())
                if name == "keytao_lookup_by_code":
                    self.assertEqual(args["code"], "yeoia")
                    return json.dumps({"success": True, "phrases": [{
                        "word": "咽", "code": "yeoia", "type": "Single", "weight": 10,
                    }]})
                raise AssertionError(name)

            shift = AsyncMock(return_value="shifted")
            with patch.object(s57_extra_code_chat, "call_tool_function", side_effect=tool), patch.object(
                s57_extra_code_commands.keytao_review, "assess_candidate_chain_commonness", AsyncMock(return_value=[]),
            ), patch.object(s57_extra_code_commands, "_execute_shift_to_code", shift):
                reply = await s57_extra_code_commands.try_handle_explicit_entry_code_command(
                    "给 嘢 加一个码 yeoia", "qq", key.actor_id, key,
                )
                self.assertIn("本次未写入", reply)
                self.assertEqual(shift.await_count, 0)
                record = s57_extra_code_chat.conversation_state_store.get_record(key)
                self.assertEqual(record.state.recommended_code, "yeoia")
                self.assertTrue(s57_extra_code_routing.message_authorizes_live_pending_mutation("硬加", record.state))
                reply = await s57_extra_code_commands.handle_pending_message_core(
                    "硬加", "qq", key.actor_id, key, allow_intent_model=False,
                )
            self.assertEqual(reply, "shifted")
            self.assertEqual(shift.await_count, 1)
            self.assertEqual(shift.call_args.args[:2], ("嘢", "yeoia"))
            self.assertTrue(shift.call_args.kwargs["auto_confirm_shift_plan"])
            self.assertEqual(shift.call_args.kwargs["target_item"]["type"], "Single")
        asyncio.run(run())

    def test_contextual_existing_record_rechecks_reading_before_add(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-extra-context")
            state = s57_extra_code_commands.create_pending_trusted_word_record("嘢", "yeoiav", "Single")
            s57_extra_code_chat.conversation_state_store.set(key, state)
            execute = AsyncMock(return_value="已加入草稿。")

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_prepare_reviewed_add":
                    self.assertEqual(args, {"word": "嘢"})
                    return json.dumps(existing_single_review())
                if name == "keytao_lookup_by_code":
                    self.assertEqual(args, {"code": "yeoia"})
                    return json.dumps({"success": True, "phrases": []})
                raise AssertionError(name)

            with patch.object(s57_extra_code_chat, "call_tool_function", side_effect=tool), patch.object(
                s57_extra_code_commands, "_execute_add_to_draft", execute,
            ):
                reply = await s57_extra_code_commands.handle_pending_message_core(
                    "加入编码yeoia", "qq", "s57-extra-context", key, allow_intent_model=False,
                )
            self.assertIn("已加入草稿", reply)
            self.assertEqual(execute.await_count, 1)
        asyncio.run(run())

    def test_existing_phrase_keeps_its_table_and_reviewed_phonetic_prefix(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-extra-phrase")
            s57_extra_code_chat.conversation_state_store.delete(key)
            review = {
                "success": True, "word": "财宝", "type": "Phrase", "recommendedCode": "cdbz",
                "needsManualReview": False,
                "existing": [{"word": "财宝", "code": "cdbz", "type": "Phrase", "weight": 100}],
                "pronunciations": [{
                    "pinyin": "cái bǎo", "recommendedCode": "cdbz", "codes": ["cdbz"],
                    "candidateStatuses": [{"code": "cdbz", "occupied": True, "words": ["财宝"]}],
                }],
            }

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_prepare_reviewed_add":
                    return json.dumps(review)
                if name == "keytao_lookup_by_code":
                    self.assertEqual(args["code"], "cdbzio")
                    return json.dumps({"success": True, "phrases": []})
                raise AssertionError(name)

            async def execute(word, code, *_args, **kwargs):
                state = s57_extra_code_chat.conversation_state_store.get(key)
                self.assertEqual(state.phrase_type, "Phrase")
                self.assertEqual((word, code), ("财宝", "cdbzio"))
                self.assertEqual(kwargs["reviewed_pinyin"], "cái bǎo")
                self.assertTrue(state.needs_manual_review)
                return "已加入草稿。"

            with patch.object(s57_extra_code_chat, "call_tool_function", side_effect=tool), patch.object(
                s57_extra_code_commands, "_execute_add_to_draft", side_effect=execute,
            ):
                reply = await s57_extra_code_commands.try_handle_explicit_entry_code_command(
                    '添加词组「财宝」，编码为「cdbzio」', "qq", key.actor_id, key,
                )
            self.assertIn("形码 io 未能核验，需管理员复核", reply)
            self.assertIn("已加入草稿", reply)
        asyncio.run(run())

    def test_exact_identity_is_rejected_while_other_table_is_ignored(self):
        async def run():
            for occupant_type, writes in (("Single", 0), ("Phrase", 1)):
                key = harness.ConversationAddress.private("qq", "s57-extra-identity-" + occupant_type)
                s57_extra_code_chat.conversation_state_store.delete(key)
                execute = AsyncMock(return_value="已加入草稿。")

                async def tool(name, args, *_args, **_kwargs):
                    if name == "keytao_prepare_reviewed_add":
                        return json.dumps(existing_single_review())
                    if name == "keytao_lookup_by_code":
                        return json.dumps({"success": True, "phrases": [{
                            "word": "嘢", "code": "yeoia", "type": occupant_type, "weight": 11,
                        }]})
                    raise AssertionError(name)

                with patch.object(s57_extra_code_chat, "call_tool_function", side_effect=tool), patch.object(
                    s57_extra_code_commands, "_execute_add_to_draft", execute,
                ):
                    reply = await s57_extra_code_commands.try_handle_explicit_entry_code_command(
                        "给 嘢 加一个码 yeoia", "qq", key.actor_id, key,
                    )
                self.assertEqual(execute.await_count, writes)
                self.assertIn("不能重复添加" if not writes else "已加入草稿", reply)
        asyncio.run(run())

    def test_completed_receipt_allows_new_explicit_code_but_unfinished_ticket_does_not(self):
        async def run():
            for recent, writes in ((True, 1), (False, 0)):
                key = harness.ConversationAddress.private("qq", "s57-extra-receipt")
                s57_extra_code_chat.conversation_state_store.set(key, harness.PendingToolConfirm(
                    function_name="keytao_submit_batch",
                    args={"batch_id": "s57-batch", "_recent_own_write": recent},
                ))

                async def tool(name, args, *_args, **_kwargs):
                    if name == "keytao_prepare_reviewed_add":
                        return json.dumps(existing_single_review())
                    if name == "keytao_lookup_by_code":
                        return json.dumps({"success": True, "phrases": []})
                    raise AssertionError(name)

                execute = AsyncMock(return_value="已加入草稿。")
                with patch.object(s57_extra_code_chat, "call_tool_function", side_effect=tool), patch.object(
                    s57_extra_code_commands, "_execute_add_to_draft", execute,
                ):
                    reply = await s57_extra_code_commands.try_handle_explicit_entry_code_command(
                        '添加单字"嘢"，编码为"yeoia"', "qq", key.actor_id, key,
                    )
                self.assertIsNotNone(reply)
                self.assertEqual(execute.await_count, writes)
        asyncio.run(run())

# Source: test_s57_security_review.py
class ExtraCodeSecurityReviewTests(unittest.TestCase):
    def test_other_actor_cannot_consume_existing_word_context(self):
        async def run():
            owner = "s57-security-owner"
            key = harness.ConversationAddress.private("qq", owner)
            state = s57_security_review_commands.create_pending_trusted_word_record(
                "嘢", "yeoiav", "Single", context_only=True,
            )
            s57_security_review_chat.conversation_state_store.set(key, state)
            before = s57_security_review_chat.conversation_state_store.get_record(key)
            tool = AsyncMock(side_effect=AssertionError("foreign actor dispatched a tool"))
            with patch.object(s57_security_review_commands, "call_tool_function", tool):
                reply = await s57_security_review_commands.handle_pending_message_core(
                    "加入编码yeoia", "qq", "s57-security-other", key,
                    allow_intent_model=False,
                )
            self.assertIsNone(reply)
            self.assertEqual(tool.await_count, 0)
            self.assertEqual(s57_security_review_chat.conversation_state_store.get_record(key), before)
        asyncio.run(run())

    def test_explicit_shared_prefix_uses_one_reading_and_keeps_manual_seal(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s57-security-reading")
            s57_security_review_chat.conversation_state_store.delete(key)
            review = copy.deepcopy(existing_single_review())
            review["pronunciations"].append({
                "pinyin": "yè", "normalized": ["ye"], "codes": ["yeo"],
                "recommendedCode": "", "requiresManualReview": True,
                "candidateStatuses": [{"code": "yeo", "occupied": False, "words": []}],
            })
            tool = AsyncMock(side_effect=[json.dumps(review), json.dumps({"success": True, "phrases": []})])
            execute = AsyncMock(return_value="fixture written")
            with patch.object(s57_security_review_commands, "call_tool_function", tool), patch.object(
                s57_security_review_commands, "_execute_add_to_draft", execute,
            ):
                reply = await s57_security_review_commands.try_handle_explicit_entry_code_command(
                    '添加单字"嘢"，编码为"yeoia"', "qq", key.actor_id, key,
                )
            self.assertIn("管理员复核", reply)
            self.assertEqual(tool.await_count, 2)
            self.assertEqual(execute.await_count, 1)
            self.assertIs(execute.call_args.args[7], True)
            self.assertEqual(execute.call_args.kwargs["reviewed_pinyin"], "yě")
            self.assertIsNone(s57_security_review_chat.conversation_state_store.get_record(key))
            with patch.object(s57_security_review_commands, "call_tool_function", AsyncMock(return_value=json.dumps(review))), patch.object(
                s57_security_review_commands, "_execute_add_to_draft", AsyncMock(side_effect=AssertionError("wrong prefix reached write")),
            ):
                rejected = await s57_security_review_commands.try_handle_explicit_entry_code_command(
                    '添加单字"嘢"，编码为"qxoia"', "qq", key.actor_id, key,
                )
            self.assertIn("音码前缀不符", rejected)
        asyncio.run(run())

# Source: test_s62_reading_resolution.py
class ReadingResolutionTests(unittest.IsolatedAsyncioTestCase):
    prepare_ambiguous = ReadingResolutionTests_prepare_ambiguous

    async def test_same_code_readings_prepare_a_manual_candidate_without_asking(self):
        review = await self.prepare_ambiguous(same_code=True)
        self.assertNotEqual(review.get("reviewDisposition"), "BLOCK", review)
        self.assertIsNot(review.get("pronunciationUnresolved"), True)
        self.assertEqual(review["recommendedCode"], "jbtaua")
        self.assertIs(review["needsManualReview"], True)
        self.assertEqual(review["pronunciations"][0]["pinyin"], "jī bái tāng")
        self.assertIn("jī bái tāng", review["manualReviewReason"])
        from keytao_bot.utils.candidate_inventory import select_candidate_inventory

        self.assertEqual(select_candidate_inventory(review).readings["jbtaua"], "jī bái tāng")

    async def test_distinct_codes_still_ask_without_an_explicit_choice(self):
        review = await self.prepare_ambiguous(same_code=False)
        self.assertEqual(review["reviewDisposition"], "BLOCK")
        self.assertIs(review["pronunciationUnresolved"], True)
        self.assertEqual(review["recommendedCode"], "")

    execute = ReadingResolutionTests_execute

    async def test_explicit_shared_code_reaches_sealed_submit(self):
        response, execute, _calls = await self.execute("鸡白汤 jbtaua，加入草稿并提交")
        self.assertEqual(execute.await_count, 1, response)
        self.assertEqual(execute.call_args.args[:2], ("鸡白汤", "jbtaua"))
        self.assertIs(execute.call_args.args[7], True)
        self.assertEqual(execute.call_args.kwargs["reviewed_pinyin"], "jī bái tāng")
        self.assertIn("jī bái tāng", response)
        self.assertIn("管理员", response)

    async def test_explicit_reading_selects_its_matching_code(self):
        response, execute, _calls = await self.execute("鸡白汤 jī bái tāng：jbtaua；加入并提交.")
        self.assertEqual(execute.await_count, 1, response)
        self.assertEqual(execute.call_args.kwargs["reviewed_pinyin"], "jī bái tāng")
        self.assertIs(execute.call_args.args[7], True)

    async def test_invalid_code_or_reading_cannot_write(self):
        for operand, reason in (("zzzzzz", "音码"), ("jbtaua7", "小写字母"),
                                ("jbtauaa", "6"), ("jī zì tāng：jbtaua", "音码"),
                                ("jī hóng tāng：jbtaua", "候选读音")):
            with self.subTest(operand=operand):
                response, execute, calls = await self.execute(f"鸡白汤 {operand} 加入并提交")
                self.assertEqual(execute.await_count, 0)
                self.assertIn(reason, response)
                self.assertEqual(len(calls), 1)

    async def test_other_blocks_and_incomplete_candidates_stay_closed(self):
        for change in ({"reviewVerdictSite": "lookup_unavailable"},
                       {"lookupFailed": True}, {"pronunciations": []},
                       {"multiSenseChoice": {}}):
            review = ambiguous_review()
            review.update(change)
            response, execute, _calls = await self.execute("鸡白汤 jbtaua 加入并提交", review)
            self.assertEqual(execute.await_count, 0)
            self.assertIn("未写入", response)

# Source: test_s62_reading_resolution.py
class ExplicitReadingReplayTests(unittest.IsolatedAsyncioTestCase):
    runtime = ExplicitSubmitTests_runtime
    turn = ExplicitSubmitTests_turn

    setUp = ExplicitReadingReplayTests_setUp

    tool = ExplicitReadingReplayTests_tool

    async def test_explicit_code_and_reading_run_real_stage_create_submit_and_seal(self):
        for message in ("鸡白汤 jbtaua，加入草稿并提交", "鸡白汤 jī bái tāng：jbtaua；加入并提交."):
            with self.subTest(message=message):
                self.setUp()
                with self.runtime():
                    response = await self.turn(message)
                self.assertEqual([(row["word"], row["code"]) for row in self.writes], [("鸡白汤", "jbtaua")], response)
                self.assertTrue(self.submitted, response)
                self.assertIs(self.writes[0]["needsManualReview"], True)
                self.assertIn("jī bái tāng", self.writes[0]["remark"])
                self.assertEqual(sum(name == "keytao_create_phrase" for name, _ in self.calls), 2)
                self.assertEqual(sum(name == "keytao_submit_batch" for name, _ in self.calls), 2)
                self.assertIn("提交", response)

# Source: test_s66_readings.py
class TranscriptTests(unittest.IsolatedAsyncioTestCase):
    runtime = ExplicitSubmitTests_runtime
    turn = FreshSelectionTests_turn

    setUp = TranscriptTests_setUp

    delivered_turn = TranscriptTests_delivered_turn

    tool = TranscriptTests_tool

    executor_runtime = TranscriptTests_executor_runtime

    async def test_exact_t1_and_t2_stay_deterministic(self):
        for message in (T1, s66_readings_T2):
            with self.subTest(message=message):
                self.setUp()
                with self.executor_runtime():
                    reply = await self.delivered_turn(message)
                    self.assertEqual(reply.count("审词："), 2, reply)
                    self.assertIn("加入并提交", reply)
                    self.assertTrue(s63_fresh_selection_chat._advertised_reply_matches_live_record(reply, self.store.get_record(self.key)))
                    self.assertIn("读音按标准", reply)
                    if message == T1:
                        self.assertIn("xiang3、hui", reply)
                    self.assertIn("yi2", reply)
                    self.assertEqual(self.trace[-1]["flow"], "word-discovery")
                    self.assertEqual(self.trace[-1]["modelCalls"], 0)
                    self.assertEqual(self.writes, [])
                    ticket = self.store.get(self.key)
                    self.assertEqual(ticket.args["_candidate_scopes"][0]["requestedReading"],
                                     "yi2 li4 xiang3 shi2 hui" if message == T1 else "yi2 li4 xiang2 shi2 hui4")
                    receipt = await self.delivered_turn("加入并提交")
                    self.assertIn("已提交审核", receipt)
                    self.assertTrue(self.submitted)
                    self.assertEqual([(row["word"], row["code"]) for row in self.writes],
                                     [(word, codes[0]) for word, codes in zip(s66_readings_WORDS, s66_readings_CODES)])
                    self.assertEqual(self.trace[-1]["modelCalls"], 0)
                args = [args for name, args in self.calls if name == "keytao_prepare_reviewed_add"]
                self.assertEqual([row["word"] for row in args], list(s66_readings_WORDS))
                self.assertEqual(args[0]["requested_reading"],
                                 "yi2 li4 xiang3 shi2 hui" if message == T1 else "yi2 li4 xiang2 shi2 hui4")
                self.assertEqual(args[1]["requested_reading"], "da4 li4 chu1 qi2 ji1")
                Path("/tmp/keytao-s66").mkdir(exist_ok=True)
                Path(f"/tmp/keytao-s66/{'t1' if message == T1 else 't2'}-replay.json").write_text(
                    json.dumps({"turns": self.trace, "calls": self.calls, "writes": self.writes,
                                "submitted": self.submitted}, ensure_ascii=False, indent=2) + "\n",
                )

    def test_three_mentions_are_removed_before_parsing(self):
        self.assertEqual(normalize_mutation_command_source("@喵喵 加词 @喵喵 小端、大端 @喵喵"), "加词  小端、大端")

    async def test_three_mentions_route_cleanly(self):
        with self.runtime():
            reply = await self.delivered_turn(s66_readings_T2 + " @喵喵")
        self.assertEqual(reply.count("审词："), 2)
        self.assertEqual(self.trace[-1]["modelCalls"], 0)

    async def test_marked_and_plain_readings_and_mixed_separators_route_without_model(self):
        for message in (
            "一力降十会( yī lì xiáng shí huì )和大力出奇迹（ dà lì chū qí jì ）",
            "添加 一力降十会（yi li xiang shi hui）；大力出奇迹( da li chu qi ji )",
            "加词 一力降十会（yī lì xiáng shí huì：ylxhu）、大力出奇迹（da4 li4 chu1 qi2 ji1）",
        ):
            self.setUp()
            with self.runtime():
                reply = await self.delivered_turn(message)
            self.assertEqual(reply.count("审词："), 2, reply)
            self.assertEqual(self.trace[-1]["modelCalls"], 0)

    async def test_item_cap_stops_before_tools(self):
        message = "，".join(f"词{char}（ci2 zi4）" for char in "甲乙丙丁戊己庚辛壬癸子")
        with self.runtime():
            reply = await self.delivered_turn(message)
        self.assertIn("最多查询 10 个词", reply)
        self.assertEqual(self.calls, [])

    async def test_foreign_actor_cannot_confirm_the_batch(self):
        with self.executor_runtime():
            await self.delivered_turn(s66_readings_T2)
            other = harness.ConversationAddress.group("qq", "865189947", "other-s66")
            await b17_rest_recommended_commands.handle_pending_message_core(
                "加入并提交", "qq", other.actor_id, other, allow_intent_model=False,
            )
            self.assertEqual(self.writes, [])
            self.assertFalse(self.submitted)

    async def test_conflicting_annotation_does_not_block_the_other_word(self):
        with self.runtime():
            reply = await self.delivered_turn(T1.replace("xiang3", "jiang4"))
        ticket = self.store.get(self.key)
        self.assertIn("音节差异", reply)
        self.assertIn("yljhui", reply)
        self.assertIn("加入并提交", reply)
        self.assertEqual([row["word"] for row in ticket.args["items"]], [s66_readings_WORDS[1]])

    async def test_t1_t2_t3_in_one_conversation(self):
        with self.executor_runtime():
            await self.delivered_turn(T1)
            reply = await self.delivered_turn(s66_readings_T2)
            self.assertEqual(reply.count("审词："), 2, self.trace)
            self.assertEqual(self.writes, [])
            receipt = await self.delivered_turn("加入并提交")
            self.assertIn("已提交审核", receipt)
            self.assertTrue(self.submitted)
            self.assertIn("用户标注读音 yi2 li4 xiang2 shi2 hui4", self.writes[0]["remark"])
            self.assertEqual([row["modelCalls"] for row in self.trace], [0, 0, 0])
        Path("/tmp/keytao-s66/t1-t2-t3-replay.json").write_text(
            json.dumps({"turns": self.trace, "calls": self.calls, "writes": self.writes},
                       ensure_ascii=False, indent=2) + "\n",
        )

    async def test_syllable_conflict_shows_both_chains_then_explicit_choice_resolves(self):
        with self.runtime():
            reply = await self.delivered_turn("一力降十会（yi1 li4 jiang4 shi2 hui4）")
            self.assertIn("音节差异", reply, self.trace)
            self.assertIn(READINGS[0], reply)
            self.assertIn("yī lì jiàng shí huì", reply)
            for code in (*s66_readings_CODES[0], "yljh", "yljhu", "yljhui"):
                self.assertIn(code, reply)
            self.assertEqual(reply.count("请明确"), 1)
            self.assertIsNone(self.store.get(self.key))
            selected = await b17_rest_recommended_commands._try_handle_explicit_reading_disambiguation(
                "一力降十会 读音 yi1 li4 jiang4 shi2 hui4", [], "qq", self.key.actor_id, self.key,
            )
            self.assertIn("yljh", selected)
            self.assertNotIn("请明确", selected)
            self.assertIsNotNone(self.store.get(self.key))
        self.assertEqual(self.writes, [])

    async def test_unavailable_syllable_keeps_known_chains_without_inventing_codes(self):
        with self.runtime():
            reply = await self.delivered_turn("一力降十会（yi1 li4 hong2 shi2 hui4）")
        self.assertIn("都不匹配", reply)
        self.assertIn("ylxhui", reply)
        self.assertIn("yljhui", reply)
        self.assertIsNone(self.store.get(self.key))

    async def test_explicit_parenthesised_code_is_bound_or_refused(self):
        for code in ("ylxhu", "zzzz"):
            self.setUp()
            with self.runtime():
                reply = await self.delivered_turn(f"添加 一力降十会（ yi2 li4 xiang3 shi2 hui ：{code} ）")
            if code == "ylxhu":
                self.assertEqual(self.store.get(self.key).args["items"][0]["code"], code)
                self.assertIn("读音按标准", reply)
            else:
                self.assertIn("不在该读音", reply)
                self.assertIn("ylxhui", reply)
                self.assertIn("yljhui", reply)
                self.assertEqual(reply.count("请明确"), 1)
                self.assertIsNone(self.store.get(self.key))

# Source: test_s66_readings.py
class ReadingGrammarTests(unittest.TestCase):
    def test_mentions_keep_other_users_and_lexical_substrings(self):
        for token in ("@喵喵", "喵喵", "键道", "@键道"):
            rows = parse_parenthesised_readings(
                f"{token} 加词 {token} 一力降十会(yi li xiang shi hui) {token}",
            )
            self.assertEqual(len(rows), 1)
        self.assertEqual(normalize_mutation_command_source("添加 @别人 喵喵叫( miao miao jiao )"),
                         "添加 @别人 喵喵叫( miao miao jiao )")
    def test_separators_tones_parentheses_and_inner_spaces(self):
        for prefix in ("", "加词 ", "添加 "):
            for separator in ("，", "、", "；", ",", ";", "和", " 和 "):
                for left, right in (("（", "）"), ("(", ")")):
                    for reading in ("yi2 li4 xiang3 shi2 hui", "yī lì xiáng shí huì", "yi li xiang shi hui"):
                        message = f"{prefix}一力降十会{left} {reading} {right}{separator}大力出奇迹{left}da4 li4 chu1 qi2 ji1{right}"
                        rows = parse_parenthesised_readings(message)
                        self.assertEqual(tuple(row.word for row in rows), s66_readings_WORDS, message)
                        self.assertEqual(rows[0].reading, reading)

    def test_malformed_or_extra_commands_do_not_gain_a_ticket(self):
        for message in (
            "不要加词 一力降十会(yi li xiang shi hui)",
            "他说 加词 一力降十会(yi li xiang shi hui)",
            "加词 一力降十会(yi li xiang shi hui)，删除小端",
            "一力降十会(yi li xiang shi hui)?", "一力降十会(yi li xiang shi hui",
            "一力降十会(yi li xiang shi hui), @别人 大力出奇迹(da li chu qi ji)",
            "一力降十会(yi li xiang shi hui)，一力降十会(yi li jiang shi hui)",
        ):
            self.assertFalse(parse_parenthesised_readings(message), message)

    def test_equivalent_tone_notation_does_not_claim_a_correction(self):
        result = review._requested_reading_context("yi1 li4 xiang2 shi2 hui4", READINGS[0])
        self.assertNotIn("readingCorrection", result)
        result = review._requested_reading_context("yi li xiang shi hui", READINGS[0])
        self.assertIn("readingCorrection", result)

# Source: test_s69_fly_ticket.py
class FlyReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_both_chains_of_reading_two_are_rendered_fe_first(self):
        result = await prepare_fly_review()
        self.assertEqual(result["pronunciations"][1]["codes"], ["fees", "feesi", "feesiu", "qees", "qeesi", "qeesiu"])
        rendered = chat_render._format_reviewed_add_prompt(result)
        self.assertIn("4. fees", rendered)
        self.assertIn("7. qees", rendered)
        self.assertIn("飞键", next(line for line in rendered.splitlines() if "4. fees" in line))

    async def test_existing_entry_surfaces_missing_fly_chain(self):
        result = await prepare_fly_review(existing=True)
        rendered = chat_render._format_reviewed_add_prompt(result)
        self.assertIn("已有", rendered)
        self.assertIn("fees", rendered)
        self.assertFalse(result["pronunciations"][1]["candidateStatuses"][0]["occupied"])

    def test_requested_analysis_rechecks_second_reading(self):
        normalized = harness._normalize_encode_response("折煞", copy.deepcopy(ENCODE))
        analysis = normalized["requestedCodeAnalysis"]
        self.assertTrue(analysis["supported"], analysis)
        self.assertEqual(analysis["matchType"], "flyKey")
        self.assertIn("fees", analysis["seriesCodes"])

    def test_requested_analysis_distinguishes_wrong_from_unknown(self):
        for code, source, expected in (("zzzz", "pinyin-pro-context", "unsupported"),
                                       ("feesio", "pinyin-pro-context", "sameSeries"),
                                       ("fees", "zdic-unavailable", "unverified")):
            payload = copy.deepcopy(ENCODE)
            payload["pronunciationSource"] = source
            payload["requestedCodeAnalysis"]["code"] = code
            result = harness._normalize_encode_response("折煞", payload)["requestedCodeAnalysis"]
            self.assertEqual(result["matchType"], expected)
            if expected == "unverified":
                self.assertIn("未能核验", result["message"])
            elif expected == "unsupported":
                self.assertIn("音码", result["message"])
            else:
                self.assertTrue(result["supported"])
                self.assertTrue(result["needsManualReview"])

    def test_all_three_fly_rules_and_phrase_positions(self):
        from keytao_bot.utils.keytao_encoding import expand_fly_key_codes, scheme_phonetic_bases
        for syllables, standard, expected in (
            (["zhé", "shā"], "qees", "fees"),
            (["zhāi", "chē"], "qhje", "fhwe"),
            (["chāo", "guāng"], "jzgm", "wzgx"),
            (["zhuāng"], "fx", "fm"),
            (["zhé", "chē", "guāng"], "qjg", "fwg"),
            (["zhé", "chē", "shā", "zhé", "chē"], "qjej", "fwew"),
        ):
            with self.subTest(syllables=syllables):
                self.assertIn(expected, scheme_phonetic_bases(syllables))
                self.assertIn(expected + "i", expand_fly_key_codes(syllables, [standard, standard + "i"]))
        self.assertEqual(scheme_phonetic_bases(["zhèn", "gǎn"]), ["qngf"])

    async def test_explicit_missing_fly_is_sealed_and_wrong_code_never_writes(self):
        state = harness.PendingAddWord(
            word="折煞", recommended_code="qeesi", candidates=[("qees", True), ("qeesi", False)],
            server_candidates=[("qees", True), ("qeesi", False)],
            pronunciation_codes={"qees": "zhé shā", "qeesi": "zhé shā"},
        )
        key = harness.ConversationAddress.private("qq", "s69-fly")
        store = harness.MemoryConversationStateStore()
        store.set(key, state)
        execute = AsyncMock(return_value="fixture written")
        lookup = AsyncMock(return_value=json.dumps({"success": True, "phrases": []}))
        with patch.object(s69_fly_ticket_chat, "conversation_state_store", store), patch.object(s69_fly_ticket_chat, "call_tool_function", lookup), patch.object(
            s69_fly_ticket_commands, "_execute_add_to_draft", execute,
        ):
            refused = await s69_fly_ticket_commands.handle_pending_message_core("加入编码zzzz", "qq", key.actor_id, key, allow_intent_model=False)
            self.assertIn("音码", refused)
            execute.assert_not_called()
            accepted = await s69_fly_ticket_commands.handle_pending_message_core("加入编码feesio", "qq", key.actor_id, key, allow_intent_model=False)
            self.assertIn("形码 io 未能核验", accepted)
        self.assertEqual(execute.call_args.args[:2], ("折煞", "feesio"))
        self.assertIs(execute.call_args.args[7], True)
        self.assertEqual(execute.call_args.kwargs["reviewed_pinyin"], "zhé shā")
        self.assertIn("feesio", execute.call_args.kwargs["reviewed_candidate_codes"])

# Source: test_s69_fly_ticket.py
class FlyCodeTests(unittest.TestCase):
    def test_missing_fly_code_is_valid_for_second_reviewed_reading(self):
        state = harness.PendingAddWord(
            word="折煞", recommended_code="eeesi",
            candidates=[("eees", True), ("eeesi", False), ("qees", True), ("qeesi", False)],
            server_candidates=[("eees", True), ("eeesi", False), ("qees", True), ("qeesi", False)],
            pronunciation_codes={"eees": "shé shā", "eeesi": "shé shā",
                                 "qees": "zhé shā", "qeesi": "zhé shā"},
        )
        result = validate_explicit_code(state, "fees")
        self.assertTrue(result.valid, result.reason)
        self.assertEqual(result.pinyin, "zhé shā")

# Source: test_s69_fly_ticket.py
class FlyTranscriptTests(unittest.IsolatedAsyncioTestCase):
    runtime = ExplicitSubmitTests_runtime
    executor_runtime = TranscriptTests_executor_runtime
    turn = FreshSelectionTests_turn
    delivered_turn = TranscriptTests_delivered_turn

    setUp = FlyTranscriptTests_setUp

    tool = FlyTranscriptTests_tool

    async def test_real_delivery_keeps_both_numbered_chains_and_fly_labels(self):
        self.review = await prepare_fly_review()
        with self.executor_runtime(), patch.object(s69_fly_ticket_chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent())), patch.object(
            s69_fly_ticket_chat, "_get_simple_word_query_words", AsyncMock(return_value=("折煞",)),
        ):
            reply = await self.delivered_turn("@喵喵 折煞")
        self.assertIn("4. fees", reply)
        self.assertIn("7. qees", reply)
        self.assertIn("飞键", reply)
        self.assertTrue(s69_fly_ticket_chat._advertised_reply_matches_live_record(reply, self.store.get_record(self.key)))
        self.assertEqual(self.trace[-1]["modelCalls"], 0)

# Source: test_s69_fly_ticket.py
class LiveTicketTests(unittest.IsolatedAsyncioTestCase):
    runtime = ExplicitSubmitTests_runtime
    executor_runtime = TranscriptTests_executor_runtime
    turn = FreshSelectionTests_turn
    delivered_turn = TranscriptTests_delivered_turn

    setUp = LiveTicketTests_setUp

    tool = LiveTicketTests_tool

    prepare_ticket = LiveTicketTests_prepare_ticket

    async def test_live_ordinal_repreviews_without_reorder_then_confirms(self):
        with self.executor_runtime():
            await self.prepare_ticket()
            result = await self.delivered_turn("@喵喵 2")
            self.assertIn("qngfv", result)
            self.assertNotIn("真敢", result)
            self.assertEqual(self.writes, [])
            self.assertEqual(self.store.get(self.key).args["code"], "qngfv")
            confirmed = await s69_fly_ticket_commands.handle_pending_message_core("确认", "qq", self.key.actor_id, self.key, allow_intent_model=False)
            if not self.submitted:
                self.assertEqual(self.store.get(self.key).function_name, "keytao_submit_batch")
                confirmed = await s69_fly_ticket_commands.handle_pending_message_core("确认", "qq", self.key.actor_id, self.key, allow_intent_model=False)
        self.assertEqual([row["code"] for row in self.writes], ["qngfv"])
        self.assertTrue(self.submitted, (confirmed, self.calls))

    async def test_expired_foreign_and_quoted_selectors_do_not_repreview(self):
        with self.runtime():
            await self.prepare_ticket()
            ticket = self.store.get(self.key)
            before = len(self.calls)
            for message in ("他说2", "不要2", "2？", '"2"', "2然后删除", "只提交2"):
                response = await s69_fly_ticket_commands.try_reselect_live_ticket(message, "qq", self.key.actor_id, self.key)
                self.assertIsNone(response, message)
                self.assertIs(self.store.get(self.key), ticket)
            foreign = harness.ConversationAddress.private("qq", "s69-foreign")
            self.assertIsNone(await s69_fly_ticket_commands.try_reselect_live_ticket("2", "qq", foreign.actor_id, foreign))
            self.now += 11
            self.assertIsNone(await s69_fly_ticket_commands.try_reselect_live_ticket("2", "qq", self.key.actor_id, self.key))
            self.assertEqual(len(self.calls), before)

    async def test_failed_repreview_retains_old_ticket_without_claiming_selection(self):
        with self.runtime():
            await self.prepare_ticket()
            original = self.store.get(self.key)
            with patch.object(s69_fly_ticket_chat, "call_tool_function", AsyncMock(return_value=json.dumps({
                "success": False, "message": "Fixture preview unavailable",
            }))):
                response = await s69_fly_ticket_commands.try_reselect_live_ticket("2", "qq", self.key.actor_id, self.key)
            self.assertIs(self.store.get(self.key), original)
            self.assertIn("改选预览未完成", response)
            self.assertNotIn("已改选", response)
            self.assertFalse(self.store.get_record(self.key).execution_id)
            self.assertEqual(self.writes, [])

    async def test_correction_is_verified_without_model_or_mutation(self):
        pairs = [("eees", True), ("qees", True)]
        self.store.set(self.key, harness.PendingAddWord(
            word="折煞", recommended_code="qees", candidates=pairs, server_candidates=pairs,
            pronunciation_codes={"eees": "shé shā", "qees": "zhé shā"},
        ))
        with self.runtime():
            result = await self.delivered_turn("应该是 fees 这个飞键")
            self.assertIn("符合", result)
            self.assertIn("zhé shā", result)
            self.assertNotIn("不是", result)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.trace[-1]["modelCalls"], 0)

    async def test_cancel_and_invalid_or_multiword_ordinal_preserve_boundaries(self):
        with self.runtime():
            await self.prepare_ticket()
            ticket = copy.deepcopy(self.store.get(self.key))
            for text in ("0", "4", "2、2"):
                response = await s69_fly_ticket_commands.handle_pending_message_core(text, "qq", self.key.actor_id, self.key, allow_intent_model=False)
                self.assertIn("未写入", response)
            ticket.args["_candidate_origin"]["queryWords"] = ["震感", "小端"]
            self.store.set(self.key, ticket)
            response = await self.delivered_turn("2")
            self.assertIn("多个词", response)
            self.assertEqual(self.writes, [])
            response = await s69_fly_ticket_commands.handle_pending_message_core("取消", "qq", self.key.actor_id, self.key, allow_intent_model=False)
            self.assertIn("已取消", response)
            self.assertIsNone(self.store.get(self.key))

# Source: test_s69_fly_ticket.py
class PublishReceiptTests(unittest.TestCase):
    def test_reorder_receipt_names_the_unpublished_revision(self):
        rows = []
        for word, old, new in (("衣品", "ykpbo", "ykpb"), ("一品", "ykpb", "ykpbv")):
            for action, code in (("Delete", old), ("Create", new)):
                rows.append({"id": len(rows) + 1, "action": action, "word": word, "code": code, "type": "Phrase"})
        result = chat_render.finalize_draft_receipt("✅ 操作已完成", {
            "success": True, "batchId": "s69-fixture", "writtenItems": rows,
        })
        self.assertNotIn("发布状态：", result)

    def test_submission_receipt_does_not_claim_revision_is_still_a_draft(self):
        change = {"id": 1, "action": "Change", "word": "衣品", "code": "ykpb", "type": "Phrase"}
        for outcome, expected in (({"status": "Submitted"}, "待审核"), ({"autoApproved": True}, "已入库")):
            result = chat_render.finalize_draft_receipt("✅ 操作已完成", {
                "success": True, "batchId": "s69-fixture", **outcome,
            }, {"success": True, "batchId": "s69-fixture", "writtenItems": [change]})
            line = next(line for line in result.splitlines() if line.startswith("发布状态："))
            self.assertIn(expected, line)
            self.assertNotIn("草稿", line)

# Source: test_s73_reading_rereview_persistence.py
class ReadingRereviewPersistenceTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = ReadingRereviewPersistenceTests_asyncSetUp

    tool = ReadingRereviewPersistenceTests_tool

    turn = ReadingRereviewPersistenceTests_turn

    initial_review = ReadingRereviewPersistenceTests_initial_review

    rereview = ReadingRereviewPersistenceTests_rereview

    assert_incident = ReadingRereviewPersistenceTests_assert_incident

    async def test_incident_dash(self):
        await self.assert_incident("dash")

    async def test_incident_colon(self):
        await self.assert_incident("colon")

    async def test_incident_bold(self):
        await self.assert_incident("bold")

    async def test_incident_prose(self):
        await self.assert_incident("prose")

    async def test_bound_prose_still_renders_review(self):
        with patch.dict(REPLIES, {"prose": "• 「田调」→ tmdc（推荐）\n回复「加入」写入草稿。"}):
            await self.assert_incident("prose")

    async def test_other_word_ticket_is_untouched(self):
        quote = await self.initial_review("天条")
        before = copy.deepcopy(self.store.get_record(self.key))
        self.review = review_fixture(corrected=True)
        await self.rereview("dash", quote)
        self.assertEqual(self.before_final_reply, before)
        self.assertEqual(self.store.get(self.key).word, "田调")

    async def test_other_actor_ticket_is_untouched(self):
        quote = await self.initial_review()
        other_key = self.key
        before = copy.deepcopy(self.store.get_record(other_key))
        self.memory = harness.ChatMemoryContext(
            platform="qq", user_id="s73-other", space_type="group", space_id="s73-group",
        )
        self.key = self.memory.conversation_address
        self.review = review_fixture(corrected=True)
        await self.rereview("dash", quote)
        self.assertEqual(self.store.get_record(other_key), before)

    async def test_tool_confirmation_is_untouched(self):
        self.store.set(self.key, harness._sealed_submit_advertisement_ticket())
        before = copy.deepcopy(self.store.get_record(self.key))
        self.review = review_fixture(corrected=True)
        await self.rereview("dash", "田调，" + REREVIEW, message="把这个加进词库")
        self.assertEqual(self.before_final_reply, before)
        self.assertIsInstance(self.store.get(self.key), harness.PendingAddWord)
        self.assertEqual(self.store.get(self.key).word, "田调")

    assert_new_word_dash = ReadingRereviewPersistenceTests_assert_new_word_dash

    async def test_other_word_dash_replaces_ticket_and_adds_new_word(self):
        await self.initial_review()
        await self.assert_new_word_dash("帮我把天调加进词库，读音 tian diao")

    assert_batch_release_dash = ReadingRereviewPersistenceTests_assert_batch_release_dash

    async def test_batch_ticket_releases_join_word_dash(self):
        await self.assert_batch_release_dash("加入天调")

    async def test_batch_ticket_releases_add_word_dash(self):
        await self.assert_batch_release_dash("帮我加个词：天调")

    async def test_other_word_prose_does_not_render_old_candidates(self):
        await self.initial_review()
        before = copy.deepcopy(self.store.get_record(self.key))
        self.review = review_fixture(corrected=True, word="天调")
        text = await self.rereview(
            "prose", "", message="帮我把天调加进词库，读音 tian diao", word="天调",
        )
        self.assertEqual(self.before_final_reply, before)
        self.assertEqual(text, "当前没有可验证的可执行操作，本次未写入。")
        self.assertEqual(self.store.get(self.key), before.state)

    async def test_prose_without_fresh_review_does_not_render_old_candidates(self):
        quote = await self.initial_review()
        before = copy.deepcopy(self.store.get_record(self.key))
        self.client.completions.responses.append(harness._FakeAIResponse("stop", REPLIES["prose"]))
        text = await self.turn(REREVIEW, quote)
        self.assertEqual(self.flow, "general")
        self.assertEqual(text, "当前没有可验证的可执行操作，本次未写入。")
        self.assertEqual(self.store.get_record(self.key).nonce, before.nonce)
        self.assertEqual(self.store.get(self.key), before.state)
        self.assertEqual(self.writes, [])

    async def test_other_word_review_with_old_word_prose_requires_fresh_nonce(self):
        for reply in (
            "你说的是「天调」不是「田调」吧？「天调」已审核，回复「加入」写入草稿。",
            "「田调」已审核，回复「加入」写入草稿。",
        ):
            with self.subTest(reply=reply):
                self.store.delete(self.key)
                self.client.completions.calls.clear()
                self.client.completions.responses.clear()
                self.calls.clear()
                self.writes.clear()
                await self.initial_review()
                before = copy.deepcopy(self.store.get_record(self.key))
                self.assertEqual(
                    [code for code, _ in before.state.server_candidates], list(OLD_CODES),
                )
                self.review = review_fixture(corrected=True, word="天调")
                review_args = {"word": "天调", "requested_reading": "tian diao"}
                self.client.completions.responses.extend([
                    harness._FakeAIResponse("tool_calls", "", [SimpleNamespace(
                        id="s73-review", type="function", function=SimpleNamespace(
                            name="keytao_prepare_reviewed_add", arguments=json.dumps(review_args),
                        ),
                    )]),
                    harness._FakeAIResponse("stop", reply),
                ])
                create = self.client.completions.create

                async def complete(**kwargs):
                    if len(self.client.completions.calls) == 1:
                        self.before_final_reply = copy.deepcopy(self.store.get_record(self.key))
                    return await create(**kwargs)

                with patch.object(self.client.completions, "create", side_effect=complete):
                    text = await self.turn("帮我把天调加进词库，读音 tian diao")
                self.assertEqual(self.flow, "general")
                self.assertIn(("keytao_prepare_reviewed_add", review_args), self.calls)
                self.assertEqual(len(self.client.completions.calls), 2)
                self.assertEqual(self.before_final_reply, before)
                for stale in (*OLD_CODES, "tian tiao"):
                    self.assertNotIn(stale, text)
                self.assertEqual(self.store.get_record(self.key).nonce, before.nonce)
                self.assertEqual(self.store.get(self.key), before.state)
                self.assertEqual(self.writes, [])

    async def test_fresh_review_does_not_back_another_word_prose(self):
        quote = await self.initial_review()
        self.review = review_fixture(corrected=True)
        with patch.dict(REPLIES, {"prose": REPLIES["prose"].replace("田调", "天调")}):
            text = await self.rereview("prose", quote)
        self.assertEqual(text, "当前没有可验证的可执行操作，本次未写入。")
        self.assertEqual(self.store.get(self.key).recommended_code, "tmdc")

    async def test_unresolved_review_removes_stale_ticket(self):
        quote = await self.initial_review()
        self.review = {
            "success": True, "word": "田调", "pronunciationUnresolved": True,
            "recommendedCode": "", "message": "读音尚未完成核验",
        }
        text = await self.rereview("colon", quote)
        self.assertIsNone(self.store.get_record(self.key))
        for stale in (*OLD_CODES, "tian tiao"):
            self.assertNotIn(stale, text)

    async def test_short_inventory_removes_stale_ticket(self):
        quote = await self.initial_review()
        self.review = review_fixture(corrected=True)
        self.review["pronunciations"][0]["candidateStatuses"] = [
            {"code": "tmdc", "occupied": False, "words": []},
        ]
        text = await self.rereview("prose", quote)
        self.assertIsNone(self.store.get_record(self.key))
        self.assertNotIn("tmtc", text)

    async def test_unresolved_inventory_cannot_reestablish_ticket(self):
        quote = await self.initial_review()
        self.review = review_fixture(corrected=True)
        self.review["pronunciationUnresolved"] = True
        text = await self.rereview("dash", quote)
        self.assertIsNone(self.store.get_record(self.key))
        self.assertNotIn("tmtc", text)

    async def test_invalid_status_removes_stale_ticket(self):
        quote = await self.initial_review()
        self.review = review_fixture(corrected=True)
        self.review["pronunciations"][0]["candidateStatuses"][0]["occupied"] = "unknown"
        text = await self.rereview("colon", quote)
        self.assertIsNone(self.store.get_record(self.key))
        self.assertNotIn("tmtc", text)

    async def test_prose_without_state_still_fails_honestly(self):
        text = s73_reading_rereview_persistence_chat._enforce_advertised_reply_contract(REPLIES["prose"], self.key)
        self.assertEqual(text, "当前没有可验证的可执行操作，本次未写入。")


def load_tests(loader, standard_tests, pattern):
    return _stage_t_unittest.TestSuite(
        loader.loadTestsFromTestCase(case) for case in (
            B10FlyKeyTests,
            B13DefaultReadingTests,
            B27ReadingTests,
            ExplicitAddHotfixTests,
            SingleReviewTests,
            ExplicitCodeTests,
            S56SecurityTests,
            ExtraCodeTests,
            ExtraCodeSecurityReviewTests,
            ReadingResolutionTests,
            ExplicitReadingReplayTests,
            TranscriptTests,
            ReadingGrammarTests,
            FlyReviewTests,
            FlyCodeTests,
            FlyTranscriptTests,
            LiveTicketTests,
            PublishReceiptTests,
            ReadingRereviewPersistenceTests,
        )
    )


if __name__ == "__main__":
    _stage_t_unittest.main()
