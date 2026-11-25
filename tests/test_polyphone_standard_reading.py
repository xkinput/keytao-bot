"""Offline regression coverage for pronunciation authority and candidate seals."""

from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from keytao_bot.utils import keytao_batch_review as batch_review
from keytao_bot.utils import keytao_review as review
from keytao_bot.utils.explicit_code import select_explicit_candidate_inventory


WORD = "苕皮"
CONFIG = review.ReviewHttpConfig(api_base="https://fake.invalid", bot_token="fake")


class PolyphoneStandardReadingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        review._clear_review_caches()
        self.addCleanup(review._clear_review_caches)
        self.fixture("苕皮", "tiao pi", ["tcpk", "tcpki"], "shao pi", "ezpk")
        self.evidence = self.enterContext(patch.object(
            review, "collect_pronunciation_evidence_limited",
            AsyncMock(return_value={"success": True, "groups": [], "lookupComplete": True}),
        ))
        self.enterContext(patch.object(
            review, "fetch_keytao_encode", AsyncMock(side_effect=lambda *_: self.encode),
        ))
        self.enterContext(patch.object(review, "lookup_words", AsyncMock(return_value={})))
        self.lookup_codes = self.enterContext(patch.object(review, "lookup_codes", AsyncMock(return_value={})))
        self.enterContext(patch.object(
            review, "detect_polyphonic_characters", return_value={"苕": ["tiao", "shao"]},
        ))
        self.enterContext(patch.object(review, "resolve_compositional_pronunciation", return_value=None))
        self.web = self.enterContext(patch.object(
            review, "_search_pronunciation_web_evidence",
            AsyncMock(return_value={"status": "no_evidence", "candidates": [], "searchComplete": True}),
        ))
        self.infer = self.enterContext(patch.object(review, "_infer_semantic_pronunciation_for_review", AsyncMock()))
        self.propose("tiao")
        self.enterContext(patch.object(review, "_infer_entity_knowledge", AsyncMock(return_value={})))
        self.enterContext(patch.object(review, "_contextual_pronunciation_group", AsyncMock(return_value=None)))
        self.enterContext(patch.object(
            review, "_query_commonness_reference",
            side_effect=lambda word: {"available": True, "corpusFrequency": {"苕": 1115, "皮": 7184}.get(word)},
        ))
        self.enterContext(patch.object(review, "_review_code_chain_priority", AsyncMock(return_value={})))

    def fixture(self, word, reading, codes, alternate, alternate_code, *, authority=False):
        readings = reading.split()
        alternates = alternate.split()
        self.encode = {
            "success": True,
            "word": word,
            "codes": codes,
            "candidateCodes": [*codes, alternate_code],
            "pronunciationSource": "zdic-phrase" if authority else "zdic-character-default",
            "standardPronunciationStatus": "found" if authority else "absent",
            "phrasePinyins": readings,
            "contextPhrasePinyins": readings,
            "chars": [
                {"char": char, "pinyin": pinyin, "pinyins": list(dict.fromkeys((pinyin, other))),
                 "pronunciationLookupStatus": "found"}
                for char, pinyin, other in zip(word, readings, alternates)
            ],
            "alternatePhrasePronunciationCodes": [{"pinyins": alternates, "codes": [alternate_code]}],
        }

    def propose(self, first_syllable):
        syllables = [first_syllable, "pi"]
        self.infer.return_value = {
            "accepted": True, "confidence": 0.9, "commonTransparent": True,
            "meaning": "用红薯淀粉制成的一种片状食物", "usageType": "common_word",
            "pinyins": syllables,
            "characters": [{"char": char, "pinyin": pinyin} for char, pinyin in zip(WORD, syllables)],
        }

    async def audit(self, code, **fields):
        item = {"id": 1, "word": self.encode["word"], "code": code, "type": "Phrase", **fields}
        return item, await review.audit_draft_items(CONFIG, [item])

    def normalize_pass(self, item, audit):
        return batch_review._normalize_llm_review(
            {"items": [{"prId": item["id"], "status": "pass", "priority": "low"}]},
            [item], None, audit,
        )["items"][0]

    async def test_tiao_pi_semantic_polyphone_is_sealed(self):
        prepared = await review.prepare_reviewed_word(CONFIG, WORD, requested_code="tcpk")
        assessment = review._assess_semantic_context_auto_pass(WORD, "tcpk", prepared)
        self.assertFalse(assessment["accepted"])
        self.assertEqual(["polyphoneStandardReading"], assessment["failedChecks"])
        self.assertTrue(assessment["policy"]["polyphoneRequiresStandardAuthority"])
        self.assertEqual("entity_context_reading", prepared["reviewVerdictSite"])
        self.assertTrue(prepared["needsManualReview"])
        _, audit = await self.audit("tcpk")
        self.assertEqual([1], audit["sealedItemIds"])
        self.assertEqual([], audit["approvedItemIds"])

    async def test_shao_pi_semantic_still_sealed_but_recommended(self):
        self.propose("shao")
        prepared = await review.prepare_reviewed_word(CONFIG, WORD, requested_code="ezpk")
        self.assertTrue(prepared["recommendedCode"].startswith("ez"))
        self.assertEqual("SEAL", prepared["reviewDisposition"])
        _, audit = await self.audit("ezpk")
        self.assertEqual([1], audit["sealedItemIds"])
        self.assertEqual([], audit["approvedItemIds"])

    async def test_generic_web_and_llm_agreement_on_tiao_pi_is_sealed(self):
        self.web.return_value = review.score_web_pronunciations(WORD, [
            {"url": f"https://{domain}/tiao-pi", "title": WORD, "snippet": "苕皮 拼音：tiáo pí"}
            for domain in ("recipes.example", "food.example")
        ])
        self.assertEqual("resolved", self.web.return_value["status"])
        prepared = await review.prepare_reviewed_word(CONFIG, WORD)
        self.assertEqual("web_model_agreement", prepared["pronunciationResolution"]["status"])
        self.assertEqual("SEAL", prepared["reviewDisposition"])
        self.assertFalse(prepared["autoReviewable"])
        for fields in ({}, {"action": "Change", "oldWord": "苕皮儿"}):
            item, audit = await self.audit("tcpk", **fields)
            self.assertEqual([1], audit["sealedItemIds"])
            self.assertEqual("manual_review", self.normalize_pass(item, audit)["status"])

    async def test_web_agreement_uses_only_the_exact_dictionary_entry(self):
        for path, approved in (("blog/苕皮", False), ("hans/小苕皮", False), ("hans/苕皮", True)):
            with self.subTest(path=path):
                review._clear_review_caches()
                self.web.return_value = review.score_web_pronunciations(WORD, [{
                    "url": f"https://www.zdic.net/{path}", "title": WORD, "snippet": "苕皮 拼音：tiáo pí",
                }])
                result = await review.prepare_reviewed_word(CONFIG, WORD)
                self.assertIs(approved, result["autoReviewable"])

    async def test_prompt_states_standard_mandarin_rule(self):
        create = AsyncMock(return_value=SimpleNamespace(choices=[]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(review, "AsyncOpenAI", return_value=client), patch.object(
            review, "_review_llm_config", return_value={
                "api_key": "fake", "base_url": "https://fake.invalid", "model": "fake", "timeout": 1,
            },
        ):
            await review._infer_semantic_pronunciation_proposal(WORD)
        prompt = create.call_args.kwargs["messages"][0]["content"]
        for text in ("普通话标准读音", "不得覆盖标准读音", "sháo"):
            self.assertIn(text, prompt)
        key = review._semantic_pronunciation_cache_key(WORD, "")
        self.assertEqual(f"std-mandarin-v1\x1f{WORD}", key)
        self.assertTrue(review._semantic_pronunciation_cache_key(WORD, "hint").startswith(f"{key}\x1f"))

    async def assert_authority(self, word, reading, code, alternate, alternate_code):
        self.fixture(word, reading, [code], alternate, alternate_code, authority=True)
        with patch.object(review, "_assess_semantic_context_auto_pass") as assess:
            result = await review.prepare_reviewed_word(CONFIG, word, requested_code=code)
            _, audit = await self.audit(code)
        assess.assert_not_called()
        self.infer.assert_not_awaited()
        self.assertEqual(code, result["recommendedCode"])
        self.assertEqual([reading.split()], [group["normalized"] for group in result["pronunciations"]])
        self.assertFalse(result["pronunciations"][0]["semanticPronunciation"])
        self.assertTrue(result["autoReviewable"])
        self.assertFalse(result["needsManualReview"])
        self.assertEqual([1], audit["approvedItemIds"])
        self.assertEqual([], audit["sealedItemIds"])

    async def test_hong_shao_authority_unaffected(self):
        await self.assert_authority("红苕", "hong shao", "hyeza", "hong tiao", "hytca")

    async def test_tiao_zhou_authority_unaffected(self):
        await self.assert_authority("苕帚", "tiao zhou", "tcfn", "shao zhou", "ezfn")

    async def test_tiao_pi_word_unaffected(self):
        await self.assert_authority("调皮", "tiao pi", "tcpk", "diao pi", "dcpk")

    async def test_out_of_chain_always_sealed(self):
        for authority in (True, False):
            with self.subTest(authority=authority):
                self.fixture(WORD, "tiao pi", ["tcpk"], "shao pi", "ezpk", authority=authority)
                prepared = await review.prepare_reviewed_word(CONFIG, WORD)
                prepared["requiresManualPronunciationReview"] = False
                prepared["candidateCodes"] = ["zzzz"]
                with patch.object(review, "prepare_reviewed_word", AsyncMock(return_value=prepared)):
                    item, audit = await self.audit("zzzz")
                self.assertEqual([1], audit["sealedItemIds"])
                self.assertEqual([], audit["approvedItemIds"])
                self.assertIn(audit["sealedItemReasons"][1], audit["structuredManualReviewIssues"])
                for audit_payload in (audit, {"reviewedWords": {WORD: prepared}}):
                    result = self.normalize_pass(item, audit_payload)
                    self.assertEqual(("manual_review", "high", "编码未经候选链佐证"),
                                     (result["status"], result["priority"], result["priorityReason"]))
                    self.assertTrue(result["needsManualReview"])

    async def test_types_without_phonetic_chains_avoid_out_of_chain_seal(self):
        self.fixture(WORD, "tiao pi", ["tcpk"], "shao pi", "ezpk", authority=True)
        prepared = await review.prepare_reviewed_word(CONFIG, WORD)
        for phrase_type in ("Supplement", "Symbol", "Link"):
            for action in ("Create", "Change"):
                with self.subTest(phrase_type=phrase_type, action=action), patch.object(
                    review, "prepare_reviewed_word", AsyncMock(return_value=prepared),
                ):
                    item, audit = await self.audit(
                        "zzzz", type=phrase_type, action=action, oldWord="旧词",
                    )
                    self.assertEqual([], audit["sealedItemIds"])
                    self.assertEqual("pass", self.normalize_pass(item, audit)["status"])
                    fallback = await batch_review._fallback_audit_with_encode_inner(CONFIG, [item], "超时")
                    self.assertEqual([], fallback["sealedItemIds"])

    async def test_explicit_code_in_alternate_group_is_writeable_but_sealed(self):
        prepared = await review.prepare_reviewed_word(CONFIG, WORD, requested_code="ezpk")
        inventory, failure = select_explicit_candidate_inventory(prepared, "ezpk")
        self.assertEqual("", failure)
        self.assertEqual("shao pi", inventory.readings["ezpk"])
        self.assertEqual("SEAL", prepared["reviewDisposition"])
        self.assertEqual("requested_encode_group", prepared["reviewVerdictSite"])
        self.assertTrue(prepared["needsManualReview"])
        self.assertIn("shao pi", prepared["manualReviewReason"])
        _, audit = await self.audit("ezpk", needsManualReview=True, manualReviewReason=prepared["manualReviewReason"])
        self.assertEqual([1], audit["sealedItemIds"])
        self.assertEqual([], audit["approvedItemIds"])
        for code in ("zzzz", "tcpkq"):
            wrong = await review.prepare_reviewed_word(CONFIG, WORD, requested_code=code)
            inventory, failure = select_explicit_candidate_inventory(wrong, code)
            self.assertIsNone(inventory)
            self.assertTrue(failure)

    async def test_transient_lookup_seals_have_a_retryable_source(self):
        for failure in ("occupancy", "evidence"):
            with self.subTest(failure=failure):
                self.fixture(WORD, "tiao pi", ["tcpk"], "shao pi", "ezpk", authority=True)
                if failure == "occupancy":
                    self.lookup_codes.side_effect = review.KeytaoApiError("fixture occupancy failure")
                else:
                    self.lookup_codes.side_effect = None
                    self.encode["standardPronunciationStatus"] = "unavailable"
                item, audit = await self.audit("tcpk")
                self.assertEqual([1], audit["sealedItemIds"])
                result = self.normalize_pass(item, audit)
                self.assertEqual("manual_review", result["status"])
                self.assertEqual("bot-lookup-transient", result["reviewRecord"]["source"])
                conflicted = self.normalize_pass({**item, "hasConflict": True}, audit)
                self.assertEqual("bot-deterministic", conflicted["reviewRecord"]["source"])
                item, audit = await self.audit("tcpk", needsManualReview=True, manualReviewReason="既有封存")
                self.assertEqual("bot-deterministic", self.normalize_pass(item, audit)["reviewRecord"]["source"])
