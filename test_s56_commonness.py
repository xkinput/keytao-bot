"""S56 protects occupied codes at the shared reviewed-candidate boundary."""

import unittest
from unittest.mock import AsyncMock, patch

import test_review_gate as harness

review = harness.review_module

from keytao_bot.utils.candidate_inventory import protected_candidate_occupants
from keytao_bot.utils.pending_confirmation import (
    advertised_command_suggestions,
    advertised_single_word_lookup_codes,
    render_server_backed_single_word_lookup,
)
from keytao_bot.harness.authorization_grammar import parse_eviction_modified_add
from keytao_bot.plugins.chat_render import _format_reviewed_add_prompt


def occupied_review(word="鎗", occupant="强", code="qx", *, phrase_type="Single", free=""):
    statuses = [{
        "code": code, "occupied": True, "words": [occupant],
        "phrases": [{"word": occupant, "code": code, "type": phrase_type, "weight": 10}],
    }]
    if free:
        statuses.append({"code": free, "occupied": False, "words": [], "phrases": []})
    return {
        "success": True, "word": word, "type": phrase_type,
        "recommendedCode": free or code,
        "pronunciations": [{"pinyin": "qiāng", "recommendedCode": free or code,
                            "candidateStatuses": statuses}],
    }


class S56CommonnessTests(unittest.IsolatedAsyncioTestCase):
    async def assess(self, subject, verdict):
        compare = AsyncMock(return_value={
            "success": True, "verdict": verdict,
            "summary": "fixture comparison evidence",
        })
        with patch.object(review, "compare_word_commonness", compare):
            assessments = await review.assess_candidate_chain_commonness(subject)
        return review.apply_candidate_ordering_recommendation(subject, assessments), compare

    async def test_single_occupied_candidate_is_compared_and_has_no_default(self):
        subject, compare = await self.assess(occupied_review(), "behind_more_common")
        compare.assert_awaited_once_with("鎗", "强")
        self.assertEqual("", subject["recommendedCode"])
        self.assertEqual("", subject["pronunciations"][0]["recommendedCode"])
        self.assertEqual("", subject["candidateOrderingAssessments"][0]["freeCode"])

    async def test_close_or_unknown_single_candidate_cannot_displace_by_default(self):
        for verdict in ("close", "not_enough_evidence"):
            with self.subTest(verdict=verdict):
                subject, _ = await self.assess(occupied_review(), verdict)
                self.assertEqual("", subject["recommendedCode"])

    async def test_weaker_word_occupant_retains_default_front_insert(self):
        subject, compare = await self.assess(
            occupied_review("单份", "蛋粉", "djff", phrase_type="Phrase", free="djffu"),
            "front_more_common",
        )
        compare.assert_awaited_once_with("单份", "蛋粉")
        self.assertEqual("djff", subject["recommendedCode"])
        self.assertEqual("djffu", subject["candidateOrderingAssessments"][0]["freeCode"])

    async def test_stronger_word_occupant_uses_free_slot(self):
        subject, _ = await self.assess(
            occupied_review("单份", "蛋粉", "djff", phrase_type="Phrase", free="djffu"),
            "behind_more_common",
        )
        self.assertEqual("djffu", subject["recommendedCode"])

    async def test_weaker_singleton_has_no_invented_empty_fallback(self):
        subject, _ = await self.assess(occupied_review(), "front_more_common")
        source = _format_reviewed_add_prompt(subject)
        self.assertEqual("qx", subject["recommendedCode"])
        self.assertIn("强」顺延", source)
        self.assertNotIn("不重排选", source)
        self.assertIn("回复「加入」", source)

    async def test_one_weaker_occupant_cannot_authorize_displacing_stronger_peer(self):
        subject = occupied_review(free="qxi")
        status = subject["pronunciations"][0]["candidateStatuses"][0]
        status["words"] = ["戕", "强"]
        status["phrases"] = [{"word": word, "code": "qx", "type": "Single"} for word in status["words"]]
        compare = AsyncMock(side_effect=[
            {"success": True, "verdict": "front_more_common"},
            {"success": True, "verdict": "behind_more_common"},
        ])
        with patch.object(review, "compare_word_commonness", compare):
            assessments = await review.assess_candidate_chain_commonness(subject)
        review.apply_candidate_ordering_recommendation(subject, assessments)
        self.assertEqual("qxi", subject["recommendedCode"])
        rendered = _format_reviewed_add_prompt(subject)
        self.assertNotIn("占 qx", rendered)
        self.assertIn("→ qxi（推荐）", rendered)

    async def test_explicit_code_occupant_uses_shared_comparator(self):
        compare = AsyncMock(return_value={"success": True, "verdict": "front_more_common"})
        with patch.object(review, "compare_word_commonness", compare):
            assessments = await review.assess_explicit_code_commonness(
                "单份", "djffuo", [{"word": "蛋粉", "type": "Phrase"}],
            )
        compare.assert_awaited_once_with("单份", "蛋粉")
        self.assertEqual((), protected_candidate_occupants(
            "单份", "djffuo", [("djffuo", True)], {"djffuo": ["蛋粉"]}, assessments,
        ))

    async def test_no_free_slot_renderer_keeps_review_and_advertises_named_eviction_only(self):
        subject, _ = await self.assess(occupied_review(), "behind_more_common")
        source = _format_reviewed_add_prompt(subject)
        self.assertIsNotNone(source)
        self.assertIn("推荐编码：暂无（现有候选均需点名顶替）", source)
        rendered = render_server_backed_single_word_lookup(
            "鎗", "", [("qx", True)], {"qx": ["强"]},
            reviewed_prompt=source, actionable_controls=True,
            ordering_assessments=subject["candidateOrderingAssessments"],
        )
        self.assertIn("读音 qiāng", rendered)
        self.assertIn("不弱于", rendered)
        self.assertIn("加入编码 qx+形码", rendered)
        self.assertNotIn("推荐编码：qx", rendered)
        self.assertNotIn("回复「加入」", rendered)
        self.assertEqual(("qx",), advertised_single_word_lookup_codes(rendered))
        commands = advertised_command_suggestions(rendered)
        self.assertEqual(("添加 鎗 qx，挤掉 强", "添加 鎗 qx，顶替 强"), commands)
        for command in commands:
            parsed = parse_eviction_modified_add(command)
            self.assertIsNotNone(parsed, command)
            self.assertEqual(("鎗", "qx", "强"), (parsed.word, parsed.code, parsed.named_occupant))

    def test_default_guard_requires_a_bound_verdict_for_every_occupant(self):
        assessments = [{"newWord": "鎗", "occupantWord": "强", "occupantCode": "qx", "verdict": "front_more_common"}]
        self.assertEqual(("戕",), protected_candidate_occupants(
            "鎗", "qx", [("qx", True)], {"qx": ["强", "戕"]}, assessments,
        ))
        self.assertEqual(("强",), protected_candidate_occupants(
            "鎗", "qxi", [("qxi", True)], {"qxi": ["强"]}, assessments,
        ))

    def test_incomplete_occupied_snapshot_never_authorizes_default_displacement(self):
        for occupants in ([], [""], None, "强"):
            with self.subTest(occupants=occupants):
                self.assertEqual(("",), protected_candidate_occupants(
                    "鎗", "qx", [("qx", True)], {"qx": occupants}, [],
                ))


if __name__ == "__main__":
    unittest.main()
