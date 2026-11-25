"""Offline coverage of common-sense pronunciation and review priority."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from keytao_bot.utils import keytao_batch_review as batch_review
from keytao_bot.utils import keytao_review as review
from keytao_bot.utils.review_flags import ReviewDisposition, review_disposition_for_site


WORD = "羊吃人"
CODE = "yirk"
READINGS = ("yáng", "chī", "rén")
CONFIG = review.ReviewHttpConfig(api_base="https://fake.invalid", bot_token="fake")


class IdiomReviewPriorityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.evidence = self.enterContext(patch.object(
            review, "collect_pronunciation_evidence_limited",
            AsyncMock(return_value={"success": True, "groups": [], "lookupComplete": True}),
        ))
        self.encode = {
            "success": True,
            "word": WORD,
            "codes": [CODE],
            "pronunciationSource": "zdic-character-default",
            "standardPronunciationStatus": "absent",
            "phrasePinyins": list(READINGS),
            "contextPhrasePinyins": list(READINGS),
            "chars": [
                {"char": char, "pinyin": pinyin, "pinyins": [pinyin], "pronunciationLookupStatus": "found"}
                for char, pinyin in zip(WORD, READINGS)
            ],
        }
        self.enterContext(patch.object(review, "fetch_keytao_encode", AsyncMock(return_value=self.encode)))
        self.enterContext(patch.object(review, "lookup_words", AsyncMock(return_value={})))
        self.lookup_codes = self.enterContext(patch.object(review, "lookup_codes", AsyncMock(return_value={})))
        self.entity = {
            "recognized": True,
            "entityType": "transparent_compound",
            "confidence": 0.96,
            "description": "指圈地运动中牧羊利益挤压农民生计的现象",
            "pinyin": "yang chi ren",
            "commonTransparent": True,
        }
        self.enterContext(patch.object(review, "_infer_entity_knowledge", AsyncMock(return_value=self.entity)))
        self.enterContext(patch.object(review, "detect_polyphonic_characters", return_value=[]))
        self.enterContext(patch.object(review, "_search_web", AsyncMock(return_value=[])))
        self.enterContext(patch.object(
            review, "_query_commonness_reference",
            side_effect=lambda word: {"available": True, "corpusFrequency": 8000 if len(word) == 1 else None},
        ))
        self.enterContext(patch.object(review, "_review_code_chain_priority", AsyncMock(return_value={})))
        self.reference = self.enterContext(patch.object(batch_review, "_query_commonness_reference", return_value={}))
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=AsyncMock(side_effect=self.fake_completion),
        )))
        self.enterContext(patch.object(batch_review, "AsyncOpenAI", new=object()))
        self.enterContext(patch.object(batch_review, "get_llm_client", return_value=self.client))
        self.enterContext(patch.object(batch_review, "_llm_config", return_value={
            "api_key": "fake", "base_url": "https://fake.invalid", "model": "fake",
            "timeout": 5, "temperature": 0.2, "max_tokens": 2000, "max_tokens_cap": 4000,
        }))

    def fake_completion(self, **request):
        payload = json.loads(request["messages"][1]["content"])
        content = {"items": [{
            "prId": item["id"], "status": "pass", "title": "读音相符",
            "reasons": ["读音明确"], "suggestions": ["保留编码"], "evidence": ["已复审"],
            "sources": ["语言常识"], "priority": "low", "priorityReason": "模型自填",
        } for item in payload["pullRequests"]]}
        return SimpleNamespace(choices=[
            SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(content))),
        ])

    async def audit_and_review(self, *codes, item_fields=None):
        items = [
            {"id": index, "word": WORD, "code": code, "type": "Phrase", **(item_fields or {})}
            for index, code in enumerate(codes, 1)
        ]
        audit = await review.audit_draft_items(CONFIG, items)
        result = await batch_review.review_keytao_batch_with_llm(
            {"pullRequests": items}, precomputed_audit=audit, items_only=True,
        )
        return audit, result["aiReview"]["items"]

    async def test_credible_idiom_passes_only_on_its_candidate_chain(self):
        prepared = await review.prepare_reviewed_word(CONFIG, WORD, requested_code=CODE)
        self.assertEqual("common_sense_reading", prepared["reviewVerdictSite"])
        self.assertEqual("PASS", prepared["reviewDisposition"])
        self.assertFalse(prepared["needsManualReview"])
        self.assertTrue(prepared["commonSenseSources"][0].startswith("语言常识："))
        self.assertIs(prepared["semanticContextAutoPass"]["checks"]["polyphoneStandardReading"], True)

        audit, (matching, mismatching) = await self.audit_and_review(CODE, "zzzz")
        self.assertEqual([1], audit["approvedItemIds"])
        self.assertEqual([2], audit["sealedItemIds"])
        self.assertTrue(audit["semanticContextAutoPassItems"][0]["basisLine"].startswith("语言常识："))
        self.assertEqual(("pass", "medium", "仅凭语言常识"),
                         (matching["status"], matching["priority"], matching["priorityReason"]))
        self.assertEqual(("manual_review", "high", "编码未经候选链佐证"),
                         (mismatching["status"], mismatching["priority"], mismatching["priorityReason"]))
        self.assertEqual(prepared["commonSenseSources"], matching["reviewRecord"]["commonSenseSources"])

        audit, _ = await self.audit_and_review(CODE)
        self.assertTrue(audit["autoApprove"])
        prompt = self.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        for rule in ("熟语/常见词的可信读音可用语言常识作为来源", "标为 pass", "sources 填“语言常识”", "标为 manual_review"):
            self.assertIn(rule, prompt)
        self.assertIs(ReviewDisposition.PASS, review_disposition_for_site("common_sense_reading"))

    async def test_code_from_another_reading_does_not_clear_the_seal(self):
        prepared = await review.prepare_reviewed_word(CONFIG, WORD)
        prepared["pronunciations"].append({"pinyin": "yang qi ren", "codes": ["yqrk"]})
        with patch.object(review, "prepare_reviewed_word", AsyncMock(return_value=prepared)):
            audit, (item,) = await self.audit_and_review("yqrk")
        self.assertFalse(audit["approvedItems"])
        self.assertEqual([1], audit["sealedItemIds"])
        self.assertEqual("manual_review", item["status"])
        self.assertEqual("high", item["priority"])

    async def test_authority_and_local_attestation_reduce_priority(self):
        self.encode.update(standardPronunciationStatus="found", pronunciationSource="zdic-phrase")
        audit, (item,) = await self.audit_and_review(CODE)
        self.assertTrue(audit["autoApprove"])
        self.assertEqual(("pass", "low", "有权威读音来源"),
                         (item["status"], item["priority"], item["priorityReason"]))

        self.encode.update(standardPronunciationStatus="absent", pronunciationSource="zdic-character-default")
        self.reference.return_value = {"dictionaryPresenceCount": 1}
        _, (item,) = await self.audit_and_review(CODE)
        self.assertEqual(("pass", "low"), (item["status"], item["priority"]))
        self.reference.return_value = {"bcc": {"attested": True, "channels": [
            {"channel": "新闻", "count": 8, "perMillion": 4},
        ]}}
        _, (item,) = await self.audit_and_review(CODE)
        self.assertEqual(("pass", "low"), (item["status"], item["priority"]))

    async def test_conflict_and_pure_delete_take_priority_over_a_pass(self):
        for fields, reason in (
            ({"hasConflict": True}, "条目存在冲突"),
            ({"action": "Delete"}, "纯删除需人工确认"),
        ):
            with self.subTest(fields=fields):
                _, (item,) = await self.audit_and_review(CODE, item_fields=fields)
                self.assertEqual(("manual_review", "high", reason),
                                 (item["status"], item["priority"], item["priorityReason"]))

    async def test_other_seals_and_lookup_failure_remain_manual(self):
        audit, (item,) = await self.audit_and_review(CODE, item_fields={"needsManualReview": True})
        self.assertEqual([1], audit["sealedItemIds"])
        self.assertEqual(("manual_review", "high"), (item["status"], item["priority"]))

        self.lookup_codes.side_effect = review.KeytaoApiError("fake lookup failure")
        prepared = await review.prepare_reviewed_word(CONFIG, WORD, requested_code=CODE)
        self.assertEqual("lookup_unavailable", prepared["reviewVerdictSite"])
        audit, (item,) = await self.audit_and_review(CODE)
        self.assertEqual([1], audit["sealedItemIds"])
        self.assertEqual(("manual_review", "high", audit["structuredManualReviewIssues"][0]),
                         (item["status"], item["priority"], item["priorityReason"]))

    async def test_weak_semantic_evidence_stays_sealed(self):
        self.entity["confidence"] = 0.74
        prepared = await review.prepare_reviewed_word(CONFIG, WORD, requested_code=CODE)
        self.assertEqual("missing_authoritative_page", prepared["reviewVerdictSite"])
        self.assertTrue(prepared["needsManualReview"])
