"""Offline B13 regressions for default candidates and evidence read errors."""

import socket
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.utils import keytao_review as review


class B13DefaultReadingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.encode = {
            "success": True, "word": "大肥鱼", "codes": ["dfy", "dfyv", "dfyvo"],
            "candidateCodes": ["dfy", "dfyv", "dfyvo"],
            "pronunciationSource": "pinyin-pro-context", "standardPronunciationStatus": "absent",
            "phrasePinyins": ["dà", "féi", "yú"], "contextPhrasePinyins": ["dà", "féi", "yú"],
            "chars": [
                {"char": char, "pinyin": readings[0], "pinyins": readings,
                 "pronunciationLookupStatus": "found"}
                for char, readings in (("大", ["dà", "dài"]), ("肥", ["féi"]), ("鱼", ["yú"]))
            ],
        }
        self.evidence = {
            "success": True, "groups": [], "sources": [], "lookupComplete": False,
            "rejections": [{"reason": "search_anchor_not_exact_word", "sourceId": "hwxnet"}],
            "sourceOutcomes": [{"sourceId": "so.com", "status": "errored"}],
        }
        self.web = {"status": "no_evidence", "candidates": [], "searchComplete": False}
        self.fetch_encode = AsyncMock(return_value=self.encode)
        tools = harness._review_tools
        for context in (
            patch.object(review, "collect_pronunciation_evidence_limited", AsyncMock(return_value=self.evidence)),
            patch.object(review, "fetch_keytao_encode", self.fetch_encode),
            patch.object(review, "lookup_words", AsyncMock(return_value={})),
            patch.object(review, "lookup_codes", AsyncMock(return_value={})),
            patch.object(review, "detect_polyphonic_characters", return_value={0: ("da", "dai")}),
            patch.object(review, "resolve_compositional_pronunciation", return_value=None),
            patch.object(review, "_search_pronunciation_web_evidence", AsyncMock(return_value=self.web)),
            patch.object(review, "_infer_semantic_pronunciation_for_review", AsyncMock(return_value={"accepted": False})),
            patch.object(review, "_infer_entity_knowledge", AsyncMock(return_value={"recognized": False})),
            patch.object(review, "_contextual_pronunciation_group", AsyncMock(return_value=None)),
            patch.object(tools, "_review_config", return_value=None),
            patch.object(tools, "_build_pre_submit_audit", AsyncMock(return_value={"autoApprove": False})),
            patch.object(tools, "assess_candidate_chain_commonness", AsyncMock(return_value=[])),
            patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")),
            patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")),
        ):
            self.enterContext(context)

    async def test_no_evidence_polyphone_reply_has_default_candidates_and_manual_review(self):
        result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        response = harness.chat_commands_module._format_reviewed_add_prompt(result)
        self.assertIn("dfyvo", response, response)
        self.assertIn("自动审核：未找到权威来源，仅使用编码服务默认读音，需要管理员审核", response)
        self.assertEqual(result["pronunciations"][0]["normalized"], ["da", "fei", "yu"])
        self.assertIs(result["autoReviewable"], False)
        self.assertIs(result["needsManualReview"], True)
        self.assertEqual(result["reviewDisposition"], "SEAL")
        self.fetch_encode.assert_awaited_once_with(None, "大肥鱼")

    async def test_no_evidence_keeps_all_encoder_reading_groups(self):
        self.encode["candidateCodes"].append("dfyva")
        self.encode["alternatePhrasePronunciationCodes"] = [{
            "char": "大", "charIndex": 0, "pinyin": "dài", "codes": ["dfyva"],
        }]
        result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        response = harness.chat_commands_module._format_reviewed_add_prompt(result)
        self.assertIn("dfyvo", response, response)
        self.assertIn("dfyva", response, response)
        self.assertIn("需要管理员审核", response)
        self.assertEqual([item["normalized"] for item in result["pronunciations"]],
                         [["da", "fei", "yu"], ["dai", "fei", "yu"]])
        self.assertEqual(result["reviewDisposition"], "SEAL")
        self.assertIs(result["autoReviewable"], False)
        self.fetch_encode.assert_awaited_once_with(None, "大肥鱼")

    async def test_weak_evidence_agreeing_with_default_keeps_manual_candidates(self):
        self.web.update(status="weak", candidates=[{"normalized": ["da", "fei", "yu"]}])
        result = await harness._review_tools.keytao_prepare_reviewed_add("大肥鱼")
        response = harness.chat_commands_module._format_reviewed_add_prompt(result)
        self.assertIn("dfyvo", response, response)
        self.assertIn("需要管理员审核", response)
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


if __name__ == "__main__":
    unittest.main()
