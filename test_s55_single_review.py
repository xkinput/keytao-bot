"""S55 Single review contracts at the real public review implementation seam."""

import copy
import unittest
from unittest.mock import AsyncMock, patch

import test_review_gate as harness

review = harness.review_module


def single_encoding(word="鎗", *, shape=None, readings=None):
    readings = readings or ["qiāng"]
    codes = ["qx" + (shape or "")[:index] for index in range(len(shape or "") + 1)]
    return {
        "success": True, "input": word, "type": "单字", "codes": codes,
        "candidateCodes": codes, "phrasePinyins": [readings[0]],
        "chars": [{
            "char": word, "pinyin": readings[0], "pinyins": readings,
            "pronunciationLookupStatus": "found", "phoneticCode": "qx",
            "shapeCode": shape, "c1": None, "c2": None,
            "fullCode": "qx" + (shape or ""),
        }],
    }


class SingleReviewTests(unittest.IsolatedAsyncioTestCase):
    async def prepare(self, encoding=None, *, occupants=None, existing=None, **kwargs):
        encoding = copy.deepcopy(encoding if encoding is not None else single_encoding())
        with patch.object(review, "fetch_keytao_encode", AsyncMock(return_value=encoding)), patch.object(
            review, "lookup_words", AsyncMock(return_value={"鎗": existing or []}),
        ), patch.object(review, "lookup_codes", AsyncMock(return_value=occupants or {})), patch.object(
            review, "collect_pronunciation_evidence_limited", AsyncMock(side_effect=AssertionError("phrase evidence lane")),
        ):
            return await review.prepare_reviewed_word(review.ReviewHttpConfig("https://fixture.invalid", "fixture"), "鎗", **kwargs)

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

    async def audit_single(self, prepared, *, code="qx"):
        with patch.object(review, "prepare_reviewed_word", AsyncMock(return_value=prepared)), patch.object(
            review, "_assess_semantic_context_auto_pass", return_value={"accepted": False},
        ) as semantic:
            audit = await review.audit_draft_items(
                review.ReviewHttpConfig("https://fixture.invalid", "fixture"),
                [{"id": -55, "action": "Create", "word": "鎗", "code": code, "type": "Single"}],
            )
        return audit, semantic.call_count

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


if __name__ == "__main__":
    unittest.main()
