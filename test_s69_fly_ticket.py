"""S69 offline regression fixtures for fly keys and live candidate changes."""

import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.utils.explicit_code import validate_explicit_code
from keytao_bot.utils import keytao_review as review
from keytao_bot.plugins import chat_render
from keytao_bot.harness.state import server_warning_ticket_is_complete
import test_s63_fresh_selection as fresh
import test_s66_readings as replay

commands = harness.chat_commands_module
chat = harness.openai_chat_module


ENCODE = {
    "success": True, "input": "折煞", "codes": ["eees", "eeesi", "eeesiu"],
    "altCodes": [], "flyKeyVariants": [], "phrasePinyins": ["shé", "shā"],
    "pronunciationSource": "pinyin-pro-context", "standardPronunciationStatus": "absent",
    "chars": [
        {"char": "折", "pinyin": "shé", "pinyins": ["shé", "zhé", "zhē"],
         "phoneticCode": "ee", "shapeCode": "iu", "pronunciationLookupStatus": "found"},
        {"char": "煞", "pinyin": "shā", "pinyins": ["shā", "shà"],
         "phoneticCode": "es", "shapeCode": "uu", "pronunciationLookupStatus": "found"},
    ],
    "requestedCodeAnalysis": {"code": "fees", "supported": False, "matchType": "unsupported"},
}


async def prepare_fly_review(existing=False):
    groups = [{"pinyin": reading, "normalized": plain.split(), "sources": [], "score": 0,
               "requiresManualReview": True}
              for reading, plain in (("shé shā", "she sha"), ("zhé shā", "zhe sha"))]
    with patch.object(review, "collect_pronunciation_evidence_limited", AsyncMock(return_value={
        "success": True, "groups": groups, "sources": [], "lookupComplete": True,
    })), patch.object(review, "fetch_keytao_encode", AsyncMock(return_value=copy.deepcopy(ENCODE))), patch.object(
        review, "lookup_words", AsyncMock(return_value={"折煞": [{"word": "折煞", "code": "qees", "type": "Phrase"}]} if existing else {}),
    ), patch.object(review, "lookup_codes", AsyncMock(return_value={
        "eees": [{"word": "射杀", "code": "eees", "type": "Phrase"}],
        "qees": [{"word": "折煞" if existing else "折杀", "code": "qees", "type": "Phrase"}],
    })), patch.object(review, "_resolve_multi_sense_pronunciation_choice", AsyncMock(return_value=(groups, {"status": "resolved"}))), patch.object(
        review, "AsyncOpenAI", side_effect=AssertionError("Model calls forbidden"),
    ):
        return await review.prepare_reviewed_word(None, "折煞")


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
        with patch.object(chat, "conversation_state_store", store), patch.object(chat, "call_tool_function", lookup), patch.object(
            commands, "_execute_add_to_draft", execute,
        ):
            refused = await commands.handle_pending_message_core("加入编码zzzz", "qq", key.actor_id, key, allow_intent_model=False)
            self.assertIn("音码", refused)
            execute.assert_not_called()
            accepted = await commands.handle_pending_message_core("加入编码feesio", "qq", key.actor_id, key, allow_intent_model=False)
            self.assertIn("形码 io 未能核验", accepted)
        self.assertEqual(execute.call_args.args[:2], ("折煞", "feesio"))
        self.assertIs(execute.call_args.args[7], True)
        self.assertEqual(execute.call_args.kwargs["reviewed_pinyin"], "zhé shā")
        self.assertIn("feesio", execute.call_args.kwargs["reviewed_candidate_codes"])


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


class FlyTranscriptTests(unittest.IsolatedAsyncioTestCase):
    runtime = replay.TranscriptTests.runtime
    executor_runtime = replay.TranscriptTests.executor_runtime
    turn = fresh.FreshSelectionTests.turn
    delivered_turn = replay.TranscriptTests.delivered_turn

    def setUp(self):
        fresh.FreshSelectionTests.setUp(self)
        self.trace = []

    async def tool(self, name, args, platform, user_id, **kwargs):
        self.calls.append((name, copy.deepcopy(args)))
        if name == "keytao_prepare_reviewed_add":
            return json.dumps(self.review)
        if name == "keytao_lookup_by_word":
            return json.dumps({"success": True, "phrases": []})
        if name == "keytao_pending_items_by_words":
            return json.dumps({"success": True, "complete": True, "items": []})
        raise AssertionError("Unexpected tool: " + name)

    async def test_real_delivery_keeps_both_numbered_chains_and_fly_labels(self):
        self.review = await prepare_fly_review()
        with self.executor_runtime(), patch.object(chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent())), patch.object(
            chat, "_get_simple_word_query_words", AsyncMock(return_value=("折煞",)),
        ):
            reply = await self.delivered_turn("@喵喵 折煞")
        self.assertIn("4. fees", reply)
        self.assertIn("7. qees", reply)
        self.assertIn("飞键", reply)
        self.assertTrue(chat._advertised_reply_matches_live_record(reply, self.store.get_record(self.key)))
        self.assertEqual(self.trace[-1]["modelCalls"], 0)


class LiveTicketTests(unittest.IsolatedAsyncioTestCase):
    runtime = replay.TranscriptTests.runtime
    executor_runtime = replay.TranscriptTests.executor_runtime
    turn = fresh.FreshSelectionTests.turn
    delivered_turn = replay.TranscriptTests.delivered_turn

    def setUp(self):
        fresh.FreshSelectionTests.setUp(self)
        self.trace = []
        self.submitted = False
        pairs = [("qngf", True), ("qngfv", False), ("qngfvv", False)]
        self.original = harness.PendingAddWord(
            word="震感", recommended_code="qngf", candidates=pairs,
            server_candidates=pairs, occupied_words={"qngf": ["真敢"]},
            server_occupied_words={"qngf": ["真敢"]},
            pronunciation_codes={code: "zhèn gǎn" for code, _ in pairs},
            needs_manual_review=True,
            server_ordering_assessments=[{
                "newWord": "震感", "occupantWord": "真敢", "occupantCode": "qngf",
                "newCode": "qngf", "freeCode": "qngfv", "verdict": "front_more_common",
                "summary": "Fixture commonness supports front insertion",
            }],
        )
        self.store.set(self.key, self.original)

    async def tool(self, name, args, platform, user_id, **kwargs):
        self.calls.append((name, copy.deepcopy(args)))
        result = {"batchId": "s69-fixture", "contentVersion": 0, "warningDigest": "a" * 64}
        if name == "keytao_shift_phrase_code":
            self.assertFalse(args.get("confirmed_plan_digest"), "Old reorder plan executed")
            result.update(success=False, requiresConfirmation=True, planDigest="b" * 64,
                          shiftPlan={"word": "震感", "targetCode": "qngf", "items": [
                              {"word": "震感", "code": "qngf", "type": "Phrase", "action": "Create"},
                              {"word": "真敢", "code": "qngf", "type": "Phrase", "action": "Delete"},
                              {"word": "真敢", "code": "qngfu", "type": "Phrase", "action": "Create"},
                          ], "shifted": [{"word": "真敢", "fromCode": "qngf", "toCode": "qngfu"}]})
        elif name == "keytao_create_phrase":
            self.assertEqual((args["word"], args["code"]), ("震感", "qngfv"))
            if args.get("confirmed"):
                self.assertEqual(args["expected_warning_digest"], "a" * 64)
                self.writes.append({"id": 69, "action": "Create", "word": "震感", "code": "qngfv", "type": "Phrase"})
                result.update(success=True, writtenItems=copy.deepcopy(self.writes), contentVersion=1)
            else:
                self.assertIs(args.get("preview_only"), True)
                result.update(success=False, requiresConfirmation=True, warnings=[])
        elif name in {"keytao_list_draft_items", "keytao_get_batch_preview"}:
            result.update(success=True, items=self.writes, count=len(self.writes), status="Draft")
        elif name == "keytao_submit_batch":
            if args.get("confirmed"):
                self.submitted = True
                result.update(success=True, status="Submitted")
            else:
                result.update(success=False, requiresConfirmation=True, snapshotDigest="c" * 64,
                              auditDigest="d" * 64, snapshotItems=self.writes, warnings=[])
        else:
            raise AssertionError("Unexpected tool: " + name)
        return json.dumps(result, ensure_ascii=False)

    async def prepare_ticket(self):
        result = await commands.handle_pending_message_core("加入并提交", "qq", self.key.actor_id, self.key, allow_intent_model=False)
        self.assertIn("真敢", result)
        self.assertTrue(server_warning_ticket_is_complete(self.store.get(self.key)))
        return result

    async def test_live_ordinal_repreviews_without_reorder_then_confirms(self):
        with self.executor_runtime():
            await self.prepare_ticket()
            result = await self.delivered_turn("@喵喵 2")
            self.assertIn("qngfv", result)
            self.assertNotIn("真敢", result)
            self.assertEqual(self.writes, [])
            self.assertEqual(self.store.get(self.key).args["code"], "qngfv")
            confirmed = await commands.handle_pending_message_core("确认", "qq", self.key.actor_id, self.key, allow_intent_model=False)
            if not self.submitted:
                self.assertEqual(self.store.get(self.key).function_name, "keytao_submit_batch")
                confirmed = await commands.handle_pending_message_core("确认", "qq", self.key.actor_id, self.key, allow_intent_model=False)
        self.assertEqual([row["code"] for row in self.writes], ["qngfv"])
        self.assertTrue(self.submitted, (confirmed, self.calls))

    async def test_expired_foreign_and_quoted_selectors_do_not_repreview(self):
        with self.runtime():
            await self.prepare_ticket()
            ticket = self.store.get(self.key)
            before = len(self.calls)
            for message in ("他说2", "不要2", "2？", '"2"', "2然后删除", "只提交2"):
                response = await commands.try_reselect_live_ticket(message, "qq", self.key.actor_id, self.key)
                self.assertIsNone(response, message)
                self.assertIs(self.store.get(self.key), ticket)
            foreign = harness.ConversationAddress.private("qq", "s69-foreign")
            self.assertIsNone(await commands.try_reselect_live_ticket("2", "qq", foreign.actor_id, foreign))
            self.now += 11
            self.assertIsNone(await commands.try_reselect_live_ticket("2", "qq", self.key.actor_id, self.key))
            self.assertEqual(len(self.calls), before)

    async def test_failed_repreview_retains_old_ticket_without_claiming_selection(self):
        with self.runtime():
            await self.prepare_ticket()
            original = self.store.get(self.key)
            with patch.object(chat, "call_tool_function", AsyncMock(return_value=json.dumps({
                "success": False, "message": "Fixture preview unavailable",
            }))):
                response = await commands.try_reselect_live_ticket("2", "qq", self.key.actor_id, self.key)
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
                response = await commands.handle_pending_message_core(text, "qq", self.key.actor_id, self.key, allow_intent_model=False)
                self.assertIn("未写入", response)
            ticket.args["_candidate_origin"]["queryWords"] = ["震感", "小端"]
            self.store.set(self.key, ticket)
            response = await self.delivered_turn("2")
            self.assertIn("多个词", response)
            self.assertEqual(self.writes, [])
            response = await commands.handle_pending_message_core("取消", "qq", self.key.actor_id, self.key, allow_intent_model=False)
            self.assertIn("已取消", response)
            self.assertIsNone(self.store.get(self.key))


class PublishReceiptTests(unittest.TestCase):
    def test_reorder_receipt_names_the_unpublished_revision(self):
        rows = []
        for word, old, new in (("衣品", "ykpbo", "ykpb"), ("一品", "ykpb", "ykpbv")):
            for action, code in (("Delete", old), ("Create", new)):
                rows.append({"id": len(rows) + 1, "action": action, "word": word, "code": code, "type": "Phrase"})
        result = chat_render.finalize_draft_receipt("✅ 操作已完成", {
            "success": True, "batchId": "s69-fixture", "writtenItems": rows,
        })
        line = next((line for line in result.splitlines() if line.startswith("发布状态：")), "")
        self.assertIn("衣品", line)
        self.assertIn("一品", line)
        self.assertIn("草稿", line)
        self.assertIn("未发布", line)

    def test_submission_receipt_does_not_claim_revision_is_still_a_draft(self):
        change = {"id": 1, "action": "Change", "word": "衣品", "code": "ykpb", "type": "Phrase"}
        for outcome, expected in (({"status": "Submitted"}, "待审核"), ({"autoApproved": True}, "已入库")):
            result = chat_render.finalize_draft_receipt("✅ 操作已完成", {
                "success": True, "batchId": "s69-fixture", **outcome,
            }, {"success": True, "batchId": "s69-fixture", "writtenItems": [change]})
            line = next(line for line in result.splitlines() if line.startswith("发布状态："))
            self.assertIn(expected, line)
            self.assertNotIn("草稿", line)


if __name__ == "__main__":
    unittest.main()
