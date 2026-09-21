"""Offline replay of truncated whole-title pronunciation inference.

The payloads describe evidence contracts, not independent linguistic claims.
No dictionary, model, provider, or production service is contacted.
"""

import copy
import json
import socket
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_review_gate as harness

review = harness.review_module

TITLE = "查拉图斯特拉如是说"
DEFAULT_PINYINS = ["chá", "lā", "tú", "sī", "tè", "lā", "rú", "shì", "shuō"]
TITLE_PINYINS = ["zhā", *DEFAULT_PINYINS[1:]]


def title_encode():
    variants = [(0, "zhā", "zlts"), (7, "tí", "cltx"),
                (8, "shuì", "cltu"), (8, "yuè", "cltv")]
    readings = {0: ["chá", "zhā"], 7: ["shì", "tí"],
                8: ["shuō", "shuì", "yuè"]}
    return {
        "success": True, "word": TITLE, "codes": ["clts"],
        "candidateCodes": ["clts", *(code for _, _, code in variants)],
        "pronunciationSource": "pinyin-pro-context",
        "standardPronunciationStatus": "absent",
        "phrasePinyins": DEFAULT_PINYINS,
        "contextPhrasePinyins": DEFAULT_PINYINS,
        "alternatePhrasePronunciationCodes": [
            {"char": TITLE[index], "charIndex": index, "pinyin": pinyin,
             "codes": [code]}
            for index, pinyin, code in variants
        ],
        "chars": [
            {"char": char, "pinyin": pinyin,
             "pinyins": readings.get(index, [pinyin]),
             "pronunciationLookupStatus": "found"}
            for index, (char, pinyin) in enumerate(zip(TITLE, DEFAULT_PINYINS))
        ],
    }


def complete_title_proposal():
    return {
        "accepted": True, "confidence": 0.98, "usageType": "book",
        "pinyins": TITLE_PINYINS,
        "characters": [{"char": char, "pinyin": pinyin}
                       for char, pinyin in zip(TITLE, TITLE_PINYINS)],
        "meaning": "该完整字符串在本测试的独立整词证据中唯一指向一部已有书名",
        "rationale": "测试提供整词语境支持的逐字读音，单字的其他读法不构成书名词义",
        "commonTransparent": False,
        "commonnessReason": "专名无需满足透明构词条件才能确定其语境读音",
    }


def provider_response(payload=None, *, truncated=False):
    content = json.dumps(payload or complete_title_proposal(), ensure_ascii=False)
    if truncated:
        content = content[:120]
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason="length" if truncated else "stop",
            message=SimpleNamespace(content=content),
        )],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=450 if truncated else 600,
                              total_tokens=451 if truncated else 601),
    )


class Sep19PronunciationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")))
        self.stack.enter_context(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Network forbidden")))
        self.stack.enter_context(patch.object(socket, "getaddrinfo", side_effect=AssertionError("DNS forbidden")))
        review._clear_review_caches()
        self.calls = []

    def install_provider(self, responses):
        responses = iter(responses)

        async def create(**kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            return next(responses)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        self.stack.enter_context(patch.object(review, "AsyncOpenAI", return_value=client))
        self.stack.enter_context(patch.object(review, "_review_llm_config", return_value={
            "api_key": "offline-fixture", "base_url": "https://offline.invalid",
            "model": "offline-fixture", "timeout": 1,
        }))

    async def prepare(self, responses):
        self.install_provider(responses)
        patches = {
            "collect_pronunciation_evidence_limited": AsyncMock(return_value={
                "success": True, "groups": [], "sources": [], "lookupComplete": True,
            }),
            "fetch_keytao_encode": AsyncMock(return_value=title_encode()),
            "lookup_words": AsyncMock(return_value={}),
            "lookup_codes": AsyncMock(return_value={}),
            "detect_polyphonic_characters": lambda _word: {"查": ["cha", "zha"]},
            "resolve_compositional_pronunciation": lambda _word: None,
            "_search_pronunciation_web_evidence": AsyncMock(return_value={
                "status": "no_evidence", "candidates": [], "searchComplete": False,
                "timedOut": True, "registryCalls": 0, "outboundAttempts": 0,
            }),
            "_infer_entity_knowledge": AsyncMock(return_value={"recognized": False}),
            "_contextual_pronunciation_group": AsyncMock(return_value=None),
        }
        for name, value in patches.items():
            self.stack.enter_context(patch.object(review, name, value))
        return await review.prepare_reviewed_word(None, TITLE)

    async def test_complete_whole_title_proposal_selects_supported_reading(self):
        result = await self.prepare([provider_response()])
        self.assertIsNot(result.get("pronunciationUnresolved"), True, result)
        self.assertEqual(result["recommendedCode"], "zlts")
        self.assertEqual(len(result["pronunciations"]), 1)
        self.assertIs(result["needsManualReview"], True)

    async def test_truncated_inference_does_not_invent_multiple_word_senses(self):
        result = await self.prepare([provider_response(truncated=True)])
        symptom = {
            "max_tokens": self.calls[0]["max_tokens"],
            "multiSenseChoice": result.get("multiSenseChoice"),
            "readings": [item["pinyin"] for item in result.get("pronunciations", [])],
            "message": result.get("message"),
        }
        self.assertNotEqual(result.get("multiSenseChoice", {}).get("status"),
                            "ambiguous", json.dumps(symptom, ensure_ascii=False))
        self.assertIs(result.get("pronunciationUnresolved"), True)
        self.assertEqual(result.get("recommendedCode"), "")
        self.assertEqual(result.get("pronunciations"), [])

    async def test_truncated_inference_is_not_cached_as_semantic_rejection(self):
        self.install_provider([provider_response(truncated=True), provider_response()])
        first = await review._infer_semantic_pronunciation_for_review(TITLE)
        second = await review._infer_semantic_pronunciation_for_review(TITLE)
        self.assertIs(first["accepted"], False)
        self.assertEqual(len(self.calls), 2, "Truncated provider output was cached as a completed rejection")
        self.assertIs(second["accepted"], True)

    async def test_completed_rejection_does_not_promote_character_variants_to_senses(self):
        result = await self.prepare([provider_response({"accepted": False})])
        self.assertIs(result.get("pronunciationUnresolved"), True)
        self.assertEqual(result.get("recommendedCode"), "")
        self.assertNotEqual(result.get("multiSenseChoice", {}).get("status"), "ambiguous")
        self.assertEqual(result.get("pronunciations"), [])

    async def test_length_finish_cannot_accept_a_parseable_prefix(self):
        response = provider_response()
        response.choices[0].finish_reason = "length"
        self.install_provider([response])
        result = await review._infer_semantic_pronunciation_for_review(TITLE)
        self.assertIs(result["accepted"], False)
        self.assertIs(result.get("inferenceIncomplete"), True)

    async def test_malformed_response_is_not_a_completed_rejection(self):
        response = provider_response(truncated=True)
        response.choices[0].finish_reason = "stop"
        self.install_provider([response, provider_response()])
        first = await review._infer_semantic_pronunciation_for_review(TITLE)
        second = await review._infer_semantic_pronunciation_for_review(TITLE)
        self.assertIs(first.get("inferenceIncomplete"), True)
        self.assertIs(second["accepted"], True)

    async def test_conflicting_whole_word_evidence_remains_unresolved(self):
        groups = [{
            "pinyin": pinyin, "normalized": normalized,
            "sources": [{"source": "Offline whole-word evidence", "trust": 5}],
            "sourceIds": ["zdic_cibs"], "score": 5, "fallback": False,
        } for pinyin, normalized in (("xíng zhǎng", ["xing", "zhang"]),
                                     ("háng zhǎng", ["hang", "zhang"]))]
        with patch.object(review, "_infer_semantic_pronunciation_for_review",
                          AsyncMock(return_value={"accepted": False})):
            retained, choice = await review._resolve_multi_sense_pronunciation_choice("行长", groups)
        self.assertEqual(choice["status"], "ambiguous")
        self.assertEqual(retained, groups)


if __name__ == "__main__":
    unittest.main()
