"""Offline tests for the admin item-review output contract."""

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from keytao_bot.utils import keytao_batch_review as batch_review
from keytao_bot.utils import keytao_review as review
from keytao_bot.utils.bcc_reference import CHANNELS, create_schema
from keytao_bot.utils.candidate_inventory import protected_candidate_occupants


def chain_entry(word, weight, *, code="abcd", phrase_type="Phrase"):
    return {"word": word, "code": code, "type": phrase_type, "weight": weight, "source": "current"}


def code_chain(after, *, code="abcd", phrase_type="Phrase"):
    return {"code": code, "type": phrase_type, "after": after}


class BatchReviewOutputTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.reference_path = Path(directory) / "reference.sqlite"
        with closing(sqlite3.connect(self.reference_path)) as db, db:
            create_schema(db)
            db.execute(
                "CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, "
                "part_of_speech TEXT, dictionary_presence_count INTEGER)"
            )
            db.executemany("INSERT INTO word_commonness VALUES (?, NULL, NULL, ?)", [
                ("日常词", 2), ("进阶词", 1),
            ])
            db.execute("CREATE TABLE standard_entry (word TEXT, source TEXT, level INTEGER)")
            db.executemany("INSERT INTO standard_entry VALUES (?, ?, ?)", [
                ("爱", "tgscc", 1), ("学习", "hsk3", 3), ("进阶词", "hsk3", 7),
            ])
            for index, channel in enumerate(CHANNELS, 1):
                db.execute("INSERT INTO bcc_dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    channel, channel, "word", f"2026-05-0{index}T10:00:00Z", "fixture",
                    "fixture", 1, 1, 100, 10000,
                ))
            db.executemany("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, NULL)", [
                ("口语", "日常词", 20, 12.3), ("多领域", "日常词", 12, 5.0),
                ("新闻", "日常词", 100, 99.0), ("口语", "冷词", 1, 0.4),
                ("新闻", "报刊词", 10, 8.4), ("文学", "报刊词", 7, 7.0),
            ])
        self.enterContext(patch.object(review, "reference_db_path", return_value=self.reference_path))
        self.web = self.enterContext(patch.object(
            review, "_estimate_word_commonness_web_fallback",
            side_effect=AssertionError("Web fallback is forbidden"),
        ))
        self.entity = self.enterContext(patch.object(
            review, "_estimate_entity_knowledge_signal",
            side_effect=AssertionError("Entity knowledge is forbidden"),
        ))

        self.payloads = []
        self.create_completion = AsyncMock(side_effect=self.fake_completion)
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=self.create_completion),
        ))
        self.enterContext(patch.object(batch_review, "AsyncOpenAI", new=object()))
        self.enterContext(patch.object(batch_review, "get_llm_client", return_value=client))
        self.enterContext(patch.object(batch_review, "_llm_config", return_value={
            "api_key": "fake", "base_url": "https://fake.invalid", "model": "fake",
            "timeout": 5, "temperature": 0.2, "max_tokens": 2000, "max_tokens_cap": 4000,
        }))

    def fake_completion(self, **request):
        payload = json.loads(request["messages"][1]["content"])
        self.payloads.append(payload)
        content = {"items": [{
            "prId": item["id"], "status": "pass", "title": "已复审",
            "reasons": ["读音相符"], "suggestions": ["保留编码"],
            "sources": ["汉典"], "evidence": ["已核对读音"],
            "authoritySources": ["模型自填"], "commonSenseSources": ["模型自填"],
            "reviewRecord": {"commonness": {"daily": "模型自填"}},
            "suggestedOrder": {"code": "fake"},
        } for item in payload["pullRequests"]]}
        return SimpleNamespace(choices=[
            SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(content))),
        ])

    async def review_items(self, words, *, chains=(), remark="", focus=None):
        items = [
            {"id": index, "action": "Create", "word": word,
             "code": "abcd", "type": "Phrase", "remark": remark}
            for index, word in enumerate(words, 1)
        ]
        result = await batch_review.review_keytao_batch_with_llm(
            {"pullRequests": items}, {"codeChains": list(chains)}, focus_pr_id=focus,
            precomputed_audit={}, items_only=True,
        )
        self.assertTrue(result["success"])
        self.assertEqual({"generatedAt", "auditSealed", "items"}, set(result["aiReview"]))
        return result["aiReview"]["items"]

    async def test_commonness_is_deterministic_and_stays_in_its_own_field(self):
        items = await self.review_items(["日常词", "学习", "爱", "空白词", "报刊词", "进阶词"])
        self.assertEqual([
            {"daily": "口语/多领域 每百万 12.3（BCC 2026-05-04）", "authority": "词典收录 2"},
            {"authority": "HSK 3级"},
            {"authority": "通用规范一级"},
            {},
            {"daily": "仅新闻/文学 每百万 8.4（BCC 2026-05-02）"},
            {"authority": "HSK 7-9级；词典收录 1"},
        ], [item["reviewRecord"]["commonness"] for item in items])
        for item in items:
            with self.subTest(pr_id=item["prId"]):
                self.assertEqual("pass", item["status"])
                self.assertEqual(["读音相符"], item["reasons"])
                self.assertEqual(["保留编码"], item["suggestions"])
                record = item["reviewRecord"]
                self.assertEqual(["汉典"], record["sources"])
                self.assertEqual(["已核对读音", "来源：汉典"], record["evidence"])
                self.assertNotIn("authoritySources", record)
                self.assertNotIn("commonSenseSources", record)
                self.assertNotIn("suggestedOrder", item)
        self.create_completion.assert_awaited()

    async def test_reorders_matching_chain_for_the_focused_item_only(self):
        after = [chain_entry("冷词", 1), chain_entry("日常词", 2)]
        wrong_type = code_chain([], phrase_type="CSS")
        items = await self.review_items(
            ["日常词", "冷词"], chains=[wrong_type, code_chain(after)], focus=1,
        )
        digest = hashlib.sha256("abcd|Phrase|冷词|1\nabcd|Phrase|日常词|2".encode()).hexdigest()[:16]
        self.assertEqual({
            "code": "abcd", "type": "Phrase",
            "orderedWords": ["日常词", "冷词"], "chainDigest": digest,
        }, items[0]["suggestedOrder"])
        self.assertNotIn("suggestedOrder", items[1])
        self.web.assert_not_called()
        self.entity.assert_not_called()

    async def test_digest_sorts_equal_weights_by_word_and_null_weights_last(self):
        after = [chain_entry("丙", None), chain_entry("乙", 2), chain_entry("甲", 2)]
        with patch.object(batch_review, "rank_code_chain_by_commonness", return_value={
            "status": "reorder", "proposedOrder": list(reversed(after)),
        }) as rank:
            item, = await self.review_items(["甲"], chains=[code_chain(after)])
        rank.assert_awaited_once_with(after, offline_only=True)
        digest = hashlib.sha256("abcd|Phrase|乙|2\nabcd|Phrase|甲|2\nabcd|Phrase|丙|None".encode())
        self.assertEqual(digest.hexdigest()[:16], item["suggestedOrder"]["chainDigest"])

    async def test_suggestion_keeps_an_unresolved_null_weight_last(self):
        after = [chain_entry("A", 1), chain_entry("B", 2), chain_entry("C", None)]

        def compare(front, behind, *, offline_only):
            return {
                "success": True,
                "verdict": "behind_more_common" if (front, behind) == ("A", "B") else "close",
                "behind": {"reference": {"attested": True}},
            }

        with patch.object(review, "compare_word_commonness", side_effect=compare):
            item, = await self.review_items(["B"], chains=[code_chain(after)])
        self.assertEqual(["B", "A", "C"], item["suggestedOrder"]["orderedWords"])

    async def test_omits_non_reorders_unchanged_orders_and_mismatched_membership(self):
        after = [chain_entry("日常词", 1), chain_entry("冷词", 2)]
        for status, proposed in (
            ("close", after[::-1]), ("ask", after[::-1]), ("already_ordered", after),
            ("reorder", after), ("reorder", [after[1], chain_entry("局外词", 3)]),
            ("reorder", [after[1], after[0], after[1]]),
        ):
            with self.subTest(status=status, proposed=proposed), patch.object(
                batch_review, "rank_code_chain_by_commonness",
                return_value={"status": status, "proposedOrder": proposed},
            ):
                item, = await self.review_items(["日常词"], chains=[code_chain(after)])
                self.assertNotIn("suggestedOrder", item)

    async def test_oversized_or_malformed_chains_never_call_the_ranker(self):
        oversized = [chain_entry(f"词语{i}", i) for i in range(9)]
        malformed = [chain_entry("日常词", "as text"), chain_entry("冷词", 1)]
        with patch.object(batch_review, "rank_code_chain_by_commonness") as rank:
            item, = await self.review_items(
                ["日常词"], chains=[code_chain(oversized), code_chain(malformed)],
            )
        rank.assert_not_called()
        self.assertNotIn("suggestedOrder", item)

    async def test_offline_only_returns_insufficient_evidence_without_web_or_entity(self):
        words = ["空白甲", "空白乙", "空白丙"]
        comparison = await review.compare_word_commonness(*words[:2], offline_only=True)
        self.assertEqual("not_enough_evidence", comparison["verdict"])
        with patch.object(review, "reference_db_path",
                          return_value=self.reference_path.with_name("missing.sqlite")):
            comparison = await review.compare_word_commonness(*words[:2], offline_only=True)
            self.assertEqual("not_enough_evidence", comparison["verdict"])
            self.assertFalse(comparison["webFallback"])
        after = [chain_entry(word, i) for i, word in enumerate(words)]
        ranked = await review.rank_code_chain_by_commonness(after, offline_only=True)
        self.assertEqual("ask", ranked["status"])
        item, = await self.review_items(words[:1], chains=[code_chain(after)])
        self.assertNotIn("suggestedOrder", item)
        self.web.assert_not_called()
        self.entity.assert_not_called()

    async def test_remark_strips_all_protected_lines_before_truncating(self):
        remark = ("常用：" + "x" * 1200 + "\n写给审核员\n  收录：词典收录 2\n"
                  "排序：冷词在前\n校验：旧摘要\n常用:旧频次\n  收录:旧词典\n"
                  "排序:旧排名\n校验:旧结果\n读音：rì cháng")
        await self.review_items(["日常词"], remark=remark)
        self.assertEqual("写给审核员\n读音：rì cháng", self.payloads[0]["pullRequests"][0]["remark"])

    async def test_llm_failure_preserves_deterministic_fields(self):
        after = [chain_entry("冷词", 1), chain_entry("日常词", 2)]
        with patch.object(batch_review, "_call_llm_chunked", side_effect=RuntimeError("fake outage")):
            item, = await self.review_items(["日常词"], chains=[code_chain(after)])
        self.assertEqual("attention", item["status"])
        self.assertEqual("词典收录 2", item["reviewRecord"]["commonness"]["authority"])
        self.assertEqual(["日常词", "冷词"], item["suggestedOrder"]["orderedWords"])

    async def test_enrichment_errors_preserve_normal_and_fallback_items(self):
        after = [chain_entry("冷词", 1), chain_entry("日常词", 2)]
        query_reference = batch_review._query_commonness_reference

        def query(word):
            if word == "日常词":
                raise RuntimeError("reference unavailable")
            return query_reference(word)

        for llm_error in (None, RuntimeError("fake outage")):
            with (
                self.subTest(llm_error=llm_error),
                patch.object(batch_review, "_query_commonness_reference", side_effect=query),
                patch.object(batch_review, "rank_code_chain_by_commonness",
                             side_effect=RuntimeError("rank unavailable")),
                patch.object(batch_review, "_call_llm_chunked", wraps=batch_review._call_llm_chunked,
                             side_effect=llm_error),
                patch.object(batch_review.logger, "warning") as warning,
            ):
                failed, healthy = await self.review_items(["日常词", "学习"], chains=[code_chain(after)])
                self.assertEqual("attention" if llm_error else "pass", failed["status"])
                self.assertNotIn("commonness", failed["reviewRecord"])
                self.assertNotIn("suggestedOrder", failed)
                self.assertNotIn("suggestedOrder", healthy)
                self.assertEqual({"authority": "HSK 3级"}, healthy["reviewRecord"]["commonness"])
                self.assertIn("rank unavailable", str(warning.call_args_list))
                self.assertIn("reference unavailable", str(warning.call_args_list))


class LegacyOccupantAssessmentTests(unittest.TestCase):
    def test_legacy_reasons_do_not_authorize_eviction(self):
        for reason, expected in (
            ("modern_semantic_vs_dictionary_dominated", ("冷词",)),
            ("dictionary_presence_margin", ("冷词",)),
            ("frequency_ratio", ()),
        ):
            with self.subTest(reason=reason):
                self.assertEqual(expected, protected_candidate_occupants(
                    "日常词", "abcd", [("abcd", True)], {"abcd": ["冷词"]},
                    [{"newWord": "日常词", "occupantWord": "冷词", "occupantCode": "abcd",
                      "verdict": "front_more_common", "decisionReason": reason}],
                ))
