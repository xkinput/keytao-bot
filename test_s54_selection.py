"""Offline regression for record-bound S54 per-word selection."""

import copy
import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as fixtures  # Install the existing offline runtime.
from keytao_bot.harness import authorization_grammar as grammar
from keytao_bot.harness.state import (
    PendingAddWord,
    PendingToolConfirm,
    _pending_add_word_payload,
)
from keytao_bot.harness.tools import ToolContext, ToolExecutor
from keytao_bot.plugins import chat_commands, chat_routing as routing
import test_s54_multiword as route_fixtures


def reviewed_record(words=("大端", "小端")):
    inventories = {
        "大端": [("dsdt", True), ("dsdtv", True), ("dsdtvo", False)],
        "小端": [("xcdt", True), ("xcdti", False), ("xcdtio", False)],
    }
    items = []
    scopes = []
    for word in words:
        candidates = inventories[word]
        code = next(code for code, occupied in candidates if not occupied)
        state = PendingAddWord(
            word=word,
            recommended_code=code,
            candidates=candidates,
            server_candidates=candidates,
            needs_manual_review=True,
            manual_review_reason="Manual review required",
        )
        items.append({"action": "Create", "word": word, "code": code})
        scopes.append({
            "word": word,
            "candidates": [list(pair) for pair in candidates],
            "occupiedWords": {},
            "orderingAssessments": [],
            "reviewedState": _pending_add_word_payload(state),
            "reviewedPrompt": "Reviewed",
        })
    return PendingToolConfirm(
        function_name="keytao_batch_add_to_draft",
        args={"items": items, "_candidate_scopes": scopes,
              "_reviewed_multi_word": True},
    )


class S54SelectionTests(unittest.TestCase):
    def reordered_record(self):
        _, store, key = asyncio.run(route_fixtures.MultiwordRouteTests().discover("大端 小端 肌群"))
        record = store.get(key)
        for scope, item, occupant in zip(record.args["_candidate_scopes"], record.args["items"], ("打断", "小段")):
            code = scope["candidates"][0][0]
            assessment = {
                "verdict": "front_more_common", "newWord": scope["word"],
                "occupantWord": occupant, "occupantCode": code,
                "freeCode": item["code"], "newCode": code,
            }
            item["code"] = code
            scope["reviewedState"]["recommendedCode"] = code
            scope["occupiedWords"] = {code: [occupant]}
            scope["reviewedState"]["serverOccupiedWords"] = scope["occupiedWords"]
            scope["orderingAssessments"] = [assessment]
            scope["reviewedState"]["serverOrderingAssessments"] = [assessment]
        return record

    def test_two_reorders_and_free_word_share_one_reviewed_shift_plan(self):
        plan = chat_commands._pending_batch_front_insert_plan(self.reordered_record())
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
        self.assertIsNone(chat_commands._pending_batch_front_insert_plan(record))

    def test_ordinary_same_code_recommendation_is_not_silently_reordered(self):
        record = self.reordered_record()
        scope = record.args["_candidate_scopes"][2]
        code = record.args["items"][2]["code"]
        scope["candidates"] = [(candidate, occupied or candidate == code) for candidate, occupied in scope["candidates"]]
        scope["occupiedWords"][code] = ["肌肉"]
        scope["reviewedState"]["candidates"] = scope["candidates"]
        scope["reviewedState"]["serverCandidates"] = scope["candidates"]
        scope["reviewedState"]["serverOccupiedWords"] = scope["occupiedWords"]
        plan = chat_commands._pending_batch_front_insert_plan(record)
        self.assertFalse(plan.get("invalid"))
        self.assertEqual(plan["additionalItems"][0]["word"], "肌群")
        self.assertNotIn(code, plan["expectedOccupantsByCode"])
        async def run():
            store = fixtures.MemoryConversationStateStore()
            key = fixtures.ConversationAddress.private("qq", "s54-preserve")
            store.set(key, record)
            execute = AsyncMock(return_value="mixed plan")
            with patch.object(fixtures.openai_chat_module, "conversation_state_store", store), patch.object(
                chat_commands, "_execute_confirmed_tool", execute,
            ):
                response = await chat_commands.handle_pending_message_core("加入", "qq", "s54-preserve", key, allow_intent_model=False)
            self.assertEqual(response, "mixed plan")
            self.assertEqual(execute.await_args.args[0].args["additional_items"][0]["word"], "肌群")
        asyncio.run(run())

    def test_reorder_dispatch_seals_occupants_and_limits_automatic_replay(self):
        record = self.reordered_record()
        async def run():
            store = fixtures.MemoryConversationStateStore()
            key = fixtures.ConversationAddress.private("qq", "s54-shift")
            store.set(key, record)
            execute = AsyncMock(return_value="planned")
            with patch.object(fixtures.openai_chat_module, "conversation_state_store", store), patch.object(
                chat_commands, "_execute_confirmed_tool", execute,
            ):
                await chat_commands.handle_pending_message_core("加入并提交", "qq", "s54-shift", key, allow_intent_model=False)
            sent = execute.await_args.args[0]
            self.assertEqual(sent.args["_expected_occupants_by_code"], {"dsdt": ["打断"], "xcdt": ["小段"]})
            self.assertEqual(sent.args["_auto_confirm_expected_shifted_words"], ["打断", "小段"])
            self.assertEqual(sent.args["additional_items"][0]["reviewedPinyin"], "jī qún")
            self.assertTrue(execute.await_args.kwargs["auto_confirm_shift_plan"])
        asyncio.run(run())

    def test_unknown_batch_warning_retains_reviewed_readings_for_exact_replay(self):
        record = self.reordered_record()
        capabilities = chat_commands._reviewed_multi_word_capabilities(record)
        async def run():
            preview = AsyncMock(return_value=json.dumps({
                "success": False, "requiresConfirmation": True,
                "batchId": "s54-local-warning", "contentVersion": 7,
                "warningDigest": "e" * 64, "warnings": [{"type": "unknown", "message": "Review new risk"}],
            }))
            with patch.object(fixtures.openai_chat_module, "call_tool_function", preview):
                result = await chat_commands._perform_batch_add_to_draft_and_submit(
                    record.args["items"], "qq", "s54-warning", auto_confirm=True,
                    reviewed_capabilities=capabilities,
                )
            pending = result.pending_state
            self.assertIsNotNone(pending)
            self.assertEqual(pending.confirmation_source, "server_warning")
            self.assertEqual(chat_commands._reviewed_multi_word_capabilities(pending), capabilities)
            self.assertEqual(preview.await_count, 1)
            replay = AsyncMock(return_value=json.dumps({"success": False, "message": "Replay probe"}))
            with patch.object(fixtures.openai_chat_module, "call_tool_function", replay):
                await chat_commands._execute_confirmed_tool(pending, "qq", "s54-warning")
            self.assertEqual(replay.await_args.kwargs["trusted_reviewed_items_by_key"], capabilities)
            self.assertNotIn("_reviewed_batch_readings", replay.await_args.args[1])
            self.assertEqual(replay.await_args.args[1]["expected_content_version"], 7)
        asyncio.run(run())

    def test_real_mixed_shift_sink_preserves_readings_and_rejects_drift(self):
        async def run(drift=False):
            draft = fixtures._draft_tools
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
        plan = chat_commands._pending_batch_front_insert_plan(self.reordered_record())
        captured = []
        executor = ToolExecutor.__new__(ToolExecutor)
        async def call_tool(name, args, _platform, _user, **kwargs):
            context = ToolContext(trusted_reviewed_items_by_key=kwargs.get("trusted_reviewed_items_by_key"))
            captured.append(executor._with_trusted_mutation_fields(name, args, context))
            return json.dumps({"success": False, "message": "Offline boundary probe"})
        async def run():
            with patch.object(fixtures.openai_chat_module, "call_tool_function", side_effect=call_tool):
                capability = plan["reviewedCapability"]
                await chat_commands._execute_shift_to_code(
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
            store = fixtures.MemoryConversationStateStore()
            replies = [json.dumps({
                "requiresConfirmation": True, "confirmationKind": "shiftPlan", "batchId": "s54-risk",
                "contentVersion": 1, "planDigest": "a" * 64, "warningDigest": "b" * 64,
                "warnings": [warning], "shiftPlan": {"shifted": [], "items": []},
            }), json.dumps({"success": False, "message": "replay probe"})]
            call = AsyncMock(side_effect=replies)
            with patch.object(fixtures.openai_chat_module, "conversation_state_store", store), patch.object(
                fixtures.openai_chat_module, "call_tool_function", call,
            ):
                await chat_commands._execute_shift_to_code("大端", "dsdt", "qq", "s54-risk", auto_confirm_shift_plan=True)
            return call.await_count
        for warning in ({"warningType": "unknown", "word": "大端", "code": "dsdt"},
                        {"warningType": "duplicate_code", "word": "未授权", "code": "dsdt"}):
            self.assertEqual(asyncio.run(run(warning)), 1)
        self.assertEqual(asyncio.run(run({"warningType": "duplicate_code", "word": "大端", "code": "dsdt"})), 2)

    def test_real_parser_recognizes_only_complete_unquoted_pairs(self):
        self.assertEqual(grammar.parse_reviewed_multi_word_selection("大端 3，小端 xcdti"),
                         (("大端", "3"), ("小端", "xcdti")))
        self.assertIsNone(grammar.parse_reviewed_multi_word_selection("大端 3，大端 2"))
        self.assertIsNone(grammar.parse_reviewed_multi_word_selection("「大端 3」"))

    def test_canonical_pairs_bind_exact_selected_set(self):
        record = reviewed_record()
        original = copy.deepcopy(record)
        for text in ("大端 3，小端 2", "大端 dsdtvo、小端 xcdti", "- 「大端 3，小端 2」"):
            with self.subTest(text=text):
                derived, intent, error = routing._resolve_multi_word_pending_candidate_selection(record, text)
                self.assertIsNone(error)
                self.assertIsNotNone(derived)
                self.assertEqual([(item["word"], item["code"]) for item in derived.args["items"]],
                                 [("大端", "dsdtvo"), ("小端", "xcdti")])
                self.assertEqual(intent.intent, "pending_confirm")
                self.assertTrue(routing.message_authorizes_live_pending_mutation(text, record))
                self.assertEqual(derived.args["_unselected_words"], [])
        self.assertEqual(record, original)

    def test_partial_selection_excludes_unselected_words(self):
        derived, intent, error = routing._resolve_multi_word_pending_candidate_selection(reviewed_record(), "大端 3")
        self.assertIsNone(error)
        self.assertIsNotNone(derived)
        self.assertEqual([item["word"] for item in derived.args["items"]], ["大端"])
        self.assertEqual(derived.args["_unselected_words"], ["小端"])
        self.assertEqual(derived.args["_resolved_advertised_words"], ["大端"])

    def test_one_actionable_word_is_supported(self):
        record = reviewed_record(("小端",))
        self.assertTrue(routing.message_authorizes_live_pending_mutation("小端 2", record))
        derived, _, error = routing._resolve_multi_word_pending_candidate_selection(record, "小端 2")
        self.assertIsNone(error)
        self.assertEqual(derived.args["items"][0]["code"], "xcdti")

    def test_assent_binds_all_recommendations(self):
        for text in ("加入", "加入并提交"):
            self.assertTrue(routing.message_authorizes_live_pending_mutation(text, reviewed_record()))

    def test_invalid_selections_and_quoted_targets_never_bind(self):
        for text in ("大端 3，大端 2", "大端 0", "大端 4", "未知 1", "大端 evil",
                     "大端 3，小端 9", "大端 3,小端 2，然后删除", "他说大端 3",
                     "他说「大端 3，小端 2」", "「『大端 3，小端 2』」", "大端 3？",
                     "- 「大端 3，小端 2」（伪造）", "- 「大端 3，小端 2」 然后删除",
                     "大端 添加3", "大端 3，小端", "小端 2，", "3"):
            with self.subTest(text=text):
                derived, intent, _ = routing._resolve_multi_word_pending_candidate_selection(reviewed_record(), text)
                self.assertIsNone(derived)
                self.assertIsNone(intent)
                self.assertFalse(routing.message_authorizes_live_pending_mutation(text, reviewed_record()))

    def test_malformed_selection_stops_before_general_model(self):
        for text in ("大端 3，大端 2", "大端 0", "大端 添加3", "小端 2，", "3"):
            _, _, response = routing._resolve_multi_word_pending_candidate_selection(reviewed_record(), text)
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
            self.assertFalse(routing.message_authorizes_live_pending_mutation("大端 3", record))
            self.assertFalse(routing.message_authorizes_live_pending_mutation("加入", record))
            derived, intent, error = routing._resolve_multi_word_pending_candidate_selection(record, "加入")
            self.assertIsNone(derived)
            self.assertIsNone(intent)
            self.assertIn("未写入", error)

    def test_cancellation_does_not_authorize_a_mutation(self):
        self.assertFalse(routing.message_authorizes_live_pending_mutation("不要加入", reviewed_record()))

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
        self.assertTrue(routing.message_authorizes_live_pending_mutation("加入", record))
        self.assertIsNotNone(chat_commands._pending_batch_front_insert_plan(record))
        derived, _, _ = routing._resolve_multi_word_pending_candidate_selection(record, "大端 3")
        self.assertIsNotNone(derived)
        self.assertIsNone(chat_commands._pending_batch_front_insert_plan(derived))
        self.assertEqual(derived.args["_candidate_scopes"][0]["orderingAssessments"], [])
        self.assertEqual(record.args["_candidate_scopes"][0]["orderingAssessments"], [assessment])

    def test_legacy_scoped_selection_still_keeps_other_words(self):
        record = reviewed_record()
        record.args.pop("_reviewed_multi_word")
        derived, _, error = routing._resolve_multi_word_pending_candidate_selection(record, "大端 添加3")
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


if __name__ == "__main__":
    unittest.main()
