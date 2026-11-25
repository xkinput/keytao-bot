"""Freeze the B3 commonness decisions against synthetic, offline evidence."""

from contextlib import closing
import json
import math
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from keytao_bot.utils import keytao_review as review, word_discovery
from keytao_bot.utils.commonness_copy import render_commonness_comparison
from tests.helpers import CHANNELS, create_schema, seed_historical


PAIRS = {
    "modern_semantic": ("脆肚", "茂才"),
    "dictionary_margin_front": ("词典甲", "词典乙"),
    "dictionary_margin_behind": ("词典乙", "词典甲"),
    "dictionary_margin_incumbent": ("词典甲", "低频旧词"),
    "incumbent_frequency": ("无频新词", "低频旧词"),
    "historical": ("古词甲", "古词乙"),
    "news_over_spoken_and_balanced": ("新闻甲", "文学乙"),
    "literature_only": ("文学丙", "文学乙"),
    "char_vs_word": ("语", "新闻甲"),
    "unattested_both_with_historical_supplement": ("注甲", "注乙"),
    "balanced_exact_tie": ("平词甲", "平词乙"),
    "balanced_breaks_peak_tie": ("平词甲", "平词丙"),
    "primary_one_sided": ("新闻甲", "文学丙"),
    "secondary_only_pair": ("副词甲", "副词乙"),
    "secondary_only_cross_type": ("副", "副词丙"),
    "secondary_only_close": ("副词丙", "文学独词"),
}
SNAPSHOT = Path(__file__).with_name("golden") / "commonness_b3.json"


def seed_reference(path):
    modern = {
        "脆肚": {"口语": (4, 80)},
        "新闻甲": {"新闻": (80, 60), "口语": (7, 120), "多领域": (5, 150)},
        "文学乙": {"文学": (20, 110), "口语": (9, 98), "多领域": (8, 96)},
        "文学丙": {"文学": (120, 45)},
        "语": {"多领域": (2, 8)},
        "平词甲": {"多领域": (7, 90), "新闻": (50, 40)},
        "平词乙": {"多领域": (7, 95), "文学": (50, 41)},
        "平词丙": {"多领域": (2, 130), "文学": (50, 41)},
        "副词甲": {"新闻": (40, 20), "文学": (2, 200)},
        "副词乙": {"新闻": (5, 100), "文学": (20, 50)},
        "副": {"新闻": (50, 500)},
        "副词丙": {"文学": (5, 1)},
        "文学独词": {"文学": (5, 100)},
    }
    with closing(sqlite3.connect(path)) as database, database:
        create_schema(database)
        database.execute(
            "CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, "
            "part_of_speech TEXT, dictionary_presence_count INTEGER)"
        )
        database.executemany("INSERT INTO word_commonness VALUES (?, ?, NULL, ?)", [
            ("茂才", 2, 1),
            ("词典甲", None, 3),
            ("词典乙", None, 1),
            ("低频旧词", 9, 0),
            ("副词甲", 2, 0),
            ("副词乙", 1000, 0),
            ("文学独词", 999, 0),
        ])
        for kind in ("char", "word"):
            for channel in CHANNELS:
                database.execute("INSERT INTO bcc_dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    f"{kind}-{channel}", channel, kind, "fixture", "fixture", "fixture",
                    1, 1, 1000, 1000000,
                ))
        for word, channels in modern.items():
            kind = "char" if len(word) == 1 else "word"
            for channel, (count, rank) in channels.items():
                database.execute("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)", (
                    f"{kind}-{channel}", word, count, count, rank,
                ))
    seed_historical(path, counts={"古词甲": (80, 3), "古词乙": (5, 40), "注甲": (1, 90)})


def candidate_review(front, behind):
    result = {
        "success": True,
        "word": front,
        "recommendedCode": "abcdx",
        "autoReviewable": True,
        "needsManualReview": False,
        "candidateStatuses": [
            {"code": "abcd", "occupied": True, "words": [behind]},
            {"code": "abcdx", "occupied": False, "words": []},
        ],
    }
    if (front, behind) == PAIRS["modern_semantic"]:
        result["preSubmitAudit"] = {"semanticContextAutoPassItems": [{
            "word": front,
            "assessment": {
                "accepted": True,
                "confidence": 1.0,
                "meaning": "火锅菜品",
                "nonObscurity": {
                    "route": "common_characters_and_llm",
                    "characterReferences": [{"corpusFrequency": 100000}] * len(front),
                },
            },
        }]}
    return result


def comparison_fields(result):
    return {key: result.get(key) for key in ("verdict", "decisionReason", "summary")}


class CommonnessGoldenTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        path = Path(directory) / "reference.sqlite"
        seed_reference(path)
        self.enterContext(patch.object(review, "reference_db_path", return_value=path))
        self.block(review, "_estimate_word_commonness_web_fallback")
        for name in (
            "_estimate_entity_knowledge_signal", "_infer_semantic_pronunciation_proposal",
            "get_llm_client", "_search_web", "_registered_web_search",
            "_search_pronunciation_web_evidence", "_fetch_review_url", "_fetch_text",
            "_fetch_entity_direct_hits", "fetch_keytao_encode", "lookup_words",
        ):
            self.block(review, name)
        self.block(word_discovery, "_call_extraction_llm")
        for name in ("connect", "connect_ex", "sendto"):
            self.block(socket.socket, name)
        self.block(socket, "getaddrinfo")
        self.enterContext(patch.object(review, "prepare_css_reviewed_item", side_effect=self.css_review))

    def block(self, target, name):
        guard = self.enterContext(patch.object(target, name, side_effect=AssertionError(f"Offline only: {name}")))
        self.addCleanup(guard.assert_not_called)
        return guard

    async def css_review(self, config, item):
        return {"success": True, "autoReviewable": item["action"] == "Create"}

    def assert_offline(self, evidence):
        if isinstance(evidence, dict):
            self.assertFalse(evidence.get("degradation"), evidence)
            self.assertFalse(evidence.get("webFallback"), evidence)
            for value in evidence.values():
                self.assert_offline(value)
        elif isinstance(evidence, list):
            for value in evidence:
                self.assert_offline(value)

    async def audit_rows(self, front, behind, *, web=False):
        config = review.ReviewHttpConfig("offline", "")
        change = await review.audit_draft_items(config, [
            {"action": "Change", "word": front, "oldWord": behind, "code": "a", "type": "CSSSingle"},
        ])
        moves = await review.audit_draft_items(config, [
            {"action": action, "word": word, "code": code, "type": "CSSSingle"}
            for action, word, code in (
                ("Delete", front, "a"), ("Delete", behind, "b"),
                ("Create", front, "b"), ("Create", behind, "c"),
            )
        ])
        changed = change["reviewedWords"][review.reviewed_word_key(front, "CSSSingle")]["commonnessComparison"]
        [priority] = moves["commonnessComparisons"]
        moved = priority["result"]
        rows = {
            "cssChange": {**comparison_fields(changed), "status": change["verdict"]},
            "priorityMove": {
                **comparison_fields(moved),
                "status": "approved" if any(
                    item.startswith("顺序调整：") for item in moves["approvedItems"]
                ) else "without_priority_approval",
            },
        }
        if web:
            for field, result in (("cssChange", changed), ("priorityMove", moved)):
                self.assertIs(result.get("webFallback"), True)
                self.assertNotIn("decisionReason", result)
                rows[field]["webFallback"] = True
        else:
            self.assert_offline(change)
            self.assert_offline(moves)
        return rows

    async def collect_rows(self):
        snapshot = {}
        for case, (front, behind) in PAIRS.items():
            with self.subTest(case=case):
                compared = await review.compare_word_commonness(front, behind)
                self.assert_offline(compared)
                if case == "unattested_both_with_historical_supplement":
                    for side in ("front", "behind"):
                        self.assertIs(compared[side]["reference"]["attested"], False)
                subject = candidate_review(front, behind)
                assessments = await review.assess_candidate_chain_commonness(subject)
                self.assert_offline(assessments)
                [assessment] = assessments
                review.apply_candidate_ordering_recommendation(subject, assessments)
                auto, manual = word_discovery.classify_reviewed([
                    {"candidate": word_discovery.WordCandidate(front), "review": subject},
                ])
                [classified] = auto + manual
                row = {
                    "compare": {**comparison_fields(compared), "comparisonCopy": render_commonness_comparison(compared)},
                    "assessment": {
                        **comparison_fields(assessment),
                        "recommendedCode": subject["recommendedCode"],
                        "comparisonCopy": render_commonness_comparison(assessment),
                    },
                    "discovery": {
                        "status": classified.group, "recommendedCode": classified.code, "summary": classified.reason,
                    },
                }
                entries = [
                    {"word": word, "code": "abcd", "type": "Phrase", "weight": weight}
                    for weight, word in enumerate((front, behind))
                ]
                semantic_loader = AsyncMock(return_value=subject) if case == "modern_semantic" else None
                for key, tie_break in (("chain", None), ("chainListedOrder", (behind, front))):
                    ranked = await review.rank_code_chain_by_commonness(
                        entries, semantic_review_loader=semantic_loader, tie_break_words=tie_break,
                    )
                    self.assert_offline(ranked)
                    [comparison] = ranked["comparisons"]
                    row[key] = {
                        **comparison_fields(comparison), "status": ranked["status"],
                        "proposedOrder": [entry["word"] for entry in ranked.get("proposedOrder", [])],
                    }
                row.update(await self.audit_rows(front, behind))
                snapshot[case] = row

        estimate = await review.estimate_word_commonness("文学独词")
        self.assert_offline(estimate)
        self.assertAlmostEqual(estimate["signals"]["corpus"], math.log1p(5) / math.log1p(20))
        snapshot["single_literature_only"] = {"estimate": {
            "attested": estimate["bcc"]["attested"],
            "primaryPerMillion": estimate["bcc"]["perMillion"],
            "corpusSource": estimate["corpusSource"],
            "rawCorpusPerMillion": estimate["rawSignals"]["corpus"],
            "score": round(estimate["score"], 6),
        }}

        front, behind = "网页甲", "网页乙"
        web = {
            "success": True, "verdict": "front_more_common", "webFallback": True,
            "frontWord": front, "behindWord": behind, "summary": "离线 fixture 中的网页回退结果",
            "front": {"score": 0.9}, "behind": {"score": 0.1},
        }
        with patch.object(review, "compare_word_commonness", return_value=web) as compare:
            snapshot["reasonless_web_shape"] = await self.audit_rows(front, behind, web=True)
        self.assertEqual([call.args for call in compare.await_args_list], [(front, behind)] * 2)
        return snapshot

    async def test_commonness_decisions(self):
        actual = await self.collect_rows()
        expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        self.assertEqual(actual.keys(), expected.keys())
        for case, row in expected.items():
            self.assertEqual(actual[case].keys(), row.keys(), case)
            for site, outcome in row.items():
                with self.subTest(case=case, site=site):
                    self.assertEqual(actual[case][site], outcome)

    async def test_secondary_pair_uses_the_same_strict_ratio_threshold(self):
        with patch.object(review, "COMMONNESS_FREQUENCY_RATIO_THRESHOLD", 2):
            result = await review.compare_word_commonness(*PAIRS["secondary_only_pair"])
        self.assert_offline(result)
        self.assertEqual((result["verdict"], result["decisionReason"]),
                         ("close", "bcc_secondary_channels_below_threshold"))

    async def test_auto_approval_whitelist_at_both_audit_sites(self):
        allowed = (
            "bcc_frequency_ratio",
            "bcc_frequency_ratio_balanced_tiebreak",
            "bcc_relative_rank_ratio",
            "bcc_relative_rank_ratio_balanced_tiebreak",
            "bcc_primary_one_sided",
            "bcc_secondary_channels",
            "frequency_ratio",
            "corpus_attested_with_no_presence_deficit",
        )
        blocked = (
            "bcc_frequency_ratio_below_threshold",
            "bcc_frequency_ratio_balanced_tiebreak_below_threshold",
            "bcc_relative_rank_ratio_below_threshold",
            "bcc_relative_rank_ratio_balanced_tiebreak_below_threshold",
            "bcc_secondary_channels_below_threshold",
            "bcc_modern_signal_unavailable",
            "bcc_historical_classical_frequency_ratio",
            "bcc_historical_classical_relative_rank_ratio",
            "bcc_historical_early_modern_frequency_ratio",
            "bcc_historical_early_modern_relative_rank_ratio",
            "local_signal_insufficient",
            "frequency_ratio_below_threshold",
            "frequency_below_minimum",
            "corpus_and_dictionary_vs_absent",
            "dictionary_presence_margin",
            "incumbent_frequency_vs_absent",
            "incumbent_historical_frequency_vs_absent",
            "modern_frequency_insufficient",
            "modern_semantic_vs_dictionary_dominated",
            "listed_order_tiebreak",
            "future_reason",
            "", None,
        )
        cases = [(reason, "front_more_common", True) for reason in allowed]
        cases += [(reason, "front_more_common", False) for reason in blocked]
        cases += [("frequency_ratio", verdict, False)
                  for verdict in ("behind_more_common", "close", "not_enough_evidence")]
        for reason, verdict, approve in cases:
            with self.subTest(reason=reason, verdict=verdict):
                with patch.object(review, "compare_word_commonness", return_value={
                    "success": True, "verdict": verdict, "decisionReason": reason,
                }):
                    rows = await self.audit_rows("新词甲", "旧词乙")
                self.assertEqual(rows["cssChange"]["status"], "pass" if approve else "needs_admin")
                self.assertEqual(rows["priorityMove"]["status"],
                                 "approved" if approve else "without_priority_approval")

    async def test_completely_unattested_pair_requires_fallback(self):
        # Without even a historical supplement the public comparison has no offline result.
        # Replace the setup guard for this call so its own fallback guard can assert zero calls.
        with patch.object(review, "_estimate_word_commonness_web_fallback",
                          side_effect=AssertionError("Fully unattested fallback blocked")) as fallback:
            with self.assertRaisesRegex(AssertionError, "Fully unattested fallback blocked"):
                await review.compare_word_commonness("空白甲", "空白乙")
        self.assertEqual([call.args for call in fallback.await_args_list], [("空白甲",), ("空白乙",)])
