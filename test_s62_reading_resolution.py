"""Offline S62 reading choices use trusted candidate codes and manual seals."""

import copy
import json
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
import test_s64_explicit_submit as explicit_submit
from keytao_bot.utils.explicit_code import parse_explicit_entry_code_request

commands = harness.chat_commands_module
chat = harness.openai_chat_module


def ambiguous_review():
    return {
        "success": True, "word": "鸡白汤", "type": "Phrase",
        "recommendedCode": "", "reviewDisposition": "BLOCK",
        "reviewVerdictSite": "pronunciation_unresolved",
        "pronunciationUnresolved": True, "lookupFailed": False,
        "multiSenseChoice": {"status": "ambiguous"},
        "pronunciations": [
            {"pinyin": pinyin, "codes": [code], "recommendedCode": code,
             "candidateStatuses": [{"code": code, "occupied": False, "words": []}]}
            for pinyin, code in (("jī bái tāng", "jbtaua"),
                                 ("jī bó tāng", "jbtaua"),
                                 ("jī zì tāng", "jztaua"))
        ],
    }


class ReadingResolutionTests(unittest.IsolatedAsyncioTestCase):
    async def prepare_ambiguous(self, *, same_code):
        from keytao_bot.utils import keytao_review as review_module

        groups = [
            {"pinyin": pinyin, "normalized": normalized, "sources": [], "score": 0,
             "requiresManualReview": True}
            for pinyin, normalized in (("jī bái tāng", ["ji", "bai", "tang"]),
                                       ("jī bó tāng", ["ji", "bo", "tang"]))
        ]
        alternate_code = "jbtaua" if same_code else "jbtauo"
        encode = {
            "success": True, "word": "鸡白汤", "codes": ["jbtaua"],
            "candidateCodes": ["jbtaua", alternate_code],
            "pronunciationSource": "pinyin-pro-context", "standardPronunciationStatus": "absent",
            "phrasePinyins": ["jī", "bái", "tāng"], "contextPhrasePinyins": ["jī", "bái", "tāng"],
            "alternatePhrasePronunciationCodes": [{
                "char": "白", "charIndex": 1, "pinyin": "bó", "codes": [alternate_code],
            }],
            "chars": [
                {"char": char, "pinyin": readings[0], "pinyins": readings,
                 "pronunciationLookupStatus": "found"}
                for char, readings in (("鸡", ["jī"]), ("白", ["bái", "bó"]), ("汤", ["tāng"]))
            ],
        }
        evidence = {"success": True, "groups": groups, "sources": [], "lookupComplete": True}
        with patch.object(review_module, "collect_pronunciation_evidence_limited", AsyncMock(return_value=evidence)), patch.object(
            review_module, "fetch_keytao_encode", AsyncMock(return_value=encode),
        ), patch.object(review_module, "lookup_words", AsyncMock(return_value={})), patch.object(
            review_module, "lookup_codes", AsyncMock(return_value={}),
        ), patch.object(review_module, "_resolve_multi_sense_pronunciation_choice", AsyncMock(return_value=(groups, {"status": "ambiguous"}))):
            return await review_module.prepare_reviewed_word(None, "鸡白汤")

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

    async def execute(self, message, review=None):
        request = parse_explicit_entry_code_request(message)
        self.assertIsNotNone(request, message)
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.private("qq", "s62-reading")
        review = copy.deepcopy(review or ambiguous_review())
        calls = []

        async def tool(name, args, platform, user_id):
            calls.append((name, args))
            if name == "keytao_prepare_reviewed_add":
                return json.dumps(review)
            if name == "keytao_lookup_by_code":
                return json.dumps({"success": True, "phrases": []})
            raise AssertionError("Unexpected external tool: " + name)

        execute = AsyncMock(return_value="fixture submitted")
        with patch.object(chat, "conversation_state_store", store), patch.object(
            chat, "call_tool_function", side_effect=tool,
        ), patch.object(commands, "_execute_add_to_draft_and_submit", execute), patch.object(
            chat, "_classify_message_command_intent",
            AsyncMock(side_effect=AssertionError("No model calls")),
        ):
            response = await commands._execute_explicit_entry_code_request(
                request, message, "qq", key.actor_id, key, None, "",
            )
        return response, execute, calls

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


class ExplicitReadingReplayTests(unittest.IsolatedAsyncioTestCase):
    runtime = explicit_submit.ExplicitSubmitTests.runtime
    turn = explicit_submit.ExplicitSubmitTests.turn

    def setUp(self):
        explicit_submit.ExplicitSubmitTests.setUp(self)
        self.review = ambiguous_review()

    async def tool(self, name, args, platform, user_id, **kwargs):
        self.assertEqual((platform, user_id), ("qq", self.key.actor_id))
        self.calls.append((name, copy.deepcopy(args)))
        if name == "keytao_pending_items_by_words":
            result = {"success": True, "complete": True, "items": []}
        elif name == "keytao_prepare_reviewed_add":
            self.assertEqual(args["word"], "鸡白汤")
            result = self.review
        elif name == "keytao_lookup_by_code":
            self.assertEqual(args["code"], "jbtaua")
            result = {"success": True, "phrases": []}
        elif name == "keytao_create_phrase":
            self.assertEqual((args["word"], args["code"]), ("鸡白汤", "jbtaua"))
            self.assertIs(args["needs_manual_review"], True)
            self.assertIn("jī bái tāng", args["remark"])
            self.assertIn("管理员复核", args["remark"])
            capability = kwargs["trusted_reviewed_items_by_key"][("鸡白汤", "jbtaua")]
            self.assertEqual(capability["pinyin"], "jī bái tāng")
            if not args.get("confirmed"):
                self.assertTrue(args["preview_only"])
                result = {"success": False, "requiresConfirmation": True,
                          "batchId": self.batch_id, "batchUrl": self.batch_url,
                          "contentVersion": self.version, "warningDigest": "a" * 64,
                          "warnings": [], "warnedCount": 0}
            else:
                self.assertEqual(args["batch_id"], self.batch_id)
                self.assertEqual(args["expected_content_version"], self.version)
                self.assertEqual(args["expected_warning_digest"], "a" * 64)
                self.assertFalse(self.writes)
                row = {"id": 6201, "action": "Create", "word": args["word"],
                       "code": args["code"], "type": "Phrase",
                       "needsManualReview": args["needs_manual_review"], "remark": args["remark"]}
                self.items.append(row)
                self.writes.append(copy.deepcopy(row))
                self.version += 1
                result = {"success": True, "batchId": self.batch_id, "batchUrl": self.batch_url,
                          "contentVersion": self.version, "writtenItems": copy.deepcopy(self.writes),
                          "pullRequestCount": 1}
        elif name in {"keytao_list_draft_items", "keytao_get_batch_preview"}:
            result = {"success": True, "batchId": self.batch_id, "batchUrl": self.batch_url,
                      "contentVersion": self.version, "status": "Draft",
                      "items": copy.deepcopy(self.items), "count": len(self.items)}
        elif name == "keytao_submit_batch":
            self.assertEqual(args["batch_id"], self.batch_id)
            if not args.get("confirmed"):
                result = {"success": False, "requiresConfirmation": True,
                          "batchId": self.batch_id, "batchUrl": self.batch_url,
                          "contentVersion": self.version, "snapshotDigest": "c" * 64,
                          "warningDigest": "d" * 64, "auditDigest": "e" * 64,
                          "snapshotItems": copy.deepcopy(self.items), "warnings": []}
            else:
                self.assertEqual(args["expected_content_version"], self.version)
                self.assertEqual(args["expected_server_snapshot_digest"], "c" * 64)
                self.assertEqual(args["expected_warning_digest"], "d" * 64)
                self.assertEqual(args["expected_audit_digest"], "e" * 64)
                self.assertFalse(self.submitted)
                self.submitted = True
                self.version += 1
                result = {"success": True, "status": "Submitted", "batchId": self.batch_id,
                          "batchUrl": self.batch_url, "contentVersion": self.version}
        else:
            raise AssertionError("Unexpected external tool: " + name)
        commands._capture_successful_draft_write_delivery(name, result, platform, user_id)
        return json.dumps(result)

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


if __name__ == "__main__":
    unittest.main()
