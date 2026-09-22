"""Offline S65 evidence delivery through the candidate render paths."""

import asyncio
import ast
from contextlib import closing
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_state_machine as harness
import test_s63_bcc as fixtures
from keytao_bot.plugins import chat_render
from keytao_bot.utils import keytao_review as review
from keytao_bot.utils import commonness_copy, commonness_query, pending_confirmation as pending
from keytao_bot.utils import same_code_reorder


def candidate_fixture(comparison, code="qbs"):
    word, occupant = comparison["frontWord"], comparison["behindWord"]
    assessment = review._candidate_commonness_assessment({
        "newWord": word, "occupantWord": occupant,
        "occupantCode": code, "freeCode": code + "o",
    }, comparison)
    statuses = [
        {"code": code, "occupied": True, "words": [occupant], "label": f"已有「{occupant}」"},
        {"code": code + "o", "occupied": False, "words": [], "label": "空位"},
    ]
    recommended = assessment["recommendedCode"]
    payload = {
        "success": True, "word": word, "type": "Phrase", "recommendedCode": recommended,
        "candidateOrderingAssessments": [assessment],
        "pronunciations": [{"pinyin": "fixture", "recommendedCode": recommended,
                            "candidateStatuses": statuses}],
    }
    state = harness.AgentOrchestrator._trusted_single_pending_add(
        {word: tuple((row["code"], row["occupied"]) for row in statuses)},
        {word: tuple(statuses)}, {word: recommended}, {},
        candidate_ordering_by_word={word: [assessment]},
    )
    return assessment, payload, state


class CommonnessRenderTests(unittest.TestCase):
    def setUp(self):
        fixtures.BccCommonnessTests.setUp(self)

    def local_comparison(self, first, second):
        return review._compare_reference_commonness(
            first, second, review._query_commonness_reference(first),
            review._query_commonness_reference(second),
        )

    def seed_both(self, second_rate=0.0371):
        # Synthetic counts/denominator reproduce the supplied ppm pair exactly.
        # They are not presented as production channel/count observations.
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("UPDATE bcc_dataset SET total_count=10000000000 WHERE filename='multi_domain_total_word_freq.txt'")
            connection.executemany(
                "INSERT INTO bcc_frequency VALUES ('multi_domain_total_word_freq.txt', ?, ?, ?, ?)",
                [("蛋粉", 944, 0.0944, 1), ("单份", round(second_rate * 10000), second_rate, 2)],
            )

    def assert_delivery(self, comparison, code="qbs"):
        assessment, payload, state = candidate_fixture(comparison, code)
        self.assertEqual(assessment["summary"], comparison["summary"])
        self.assertEqual(assessment["decisionReason"], comparison["decisionReason"])
        chat = harness.openai_chat_module
        store = harness.MemoryConversationStateStore()
        key = harness.ConversationAddress.private("qq", "s65-render")
        store.set(key, state)
        scope = {"word": state.word, "candidates": state.server_candidates,
                 "occupiedWords": state.server_occupied_words,
                 "orderingAssessments": state.server_ordering_assessments}
        item = {"word": state.word, "code": assessment["recommendedCode"], "needsManualReview": True}
        # Once the assessment exists, no renderer may query again.
        with patch.object(review, "_query_commonness_reference", side_effect=AssertionError("Render must use the snapshot")):
            prompt = chat_render._format_reviewed_add_prompt(payload)
            rebuilt = pending.render_server_backed_single_word_candidates(
                state.word, assessment["recommendedCode"], state.server_candidates,
                state.server_occupied_words, state.server_ordering_assessments,
            )
            lookup = pending.render_server_backed_single_word_lookup(
                state.word, assessment["recommendedCode"], state.server_candidates,
                state.server_occupied_words, ordering_assessments=state.server_ordering_assessments,
                reviewed_prompt=prompt, actionable_controls=True,
            )
            second = {"word": "测试另词", "candidates": [("abcd", False)],
                      "occupiedWords": {}, "orderingAssessments": []}
            batch = pending.render_server_backed_batch_candidates(
                [item, {"word": "测试另词", "code": "abcd"}], [scope, second],
            )
            for name, text in (("reviewed", prompt), ("rebuilt", rebuilt), ("lookup", lookup), ("batch", batch)):
                with self.subTest(path=name, verdict=comparison["verdict"]):
                    self.assertIn(comparison["summary"], text)
            self.assertIn(comparison["summary"], review.build_review_note({
                "commonnessComparisons": [{"frontWord": state.word,
                    "behindWord": assessment["occupantWord"], "code": code, "result": comparison}],
            }))
            review.record_commonness_evidence(comparison)
            with patch.object(chat, "conversation_state_store", store):
                delivered = chat._enforce_advertised_reply_contract(prompt, key)
                self.assertIn(comparison["summary"], delivered)
                self.assertIn(comparison["summary"], chat._prepare_user_facing_reply(
                    delivered, SimpleNamespace(conversation_address=key, platform="qq")))
        self.web.assert_not_called()
        return chat_render._format_candidate_ordering_assessment(assessment, {code: 1, code + "o": 2})

    def test_one_sided_candidate_keeps_bcc_evidence(self):
        comparison = asyncio.run(review.compare_word_commonness("敲不死", "情报所"))
        self.assertEqual(comparison["verdict"], "not_enough_evidence")
        assessment = review._candidate_commonness_assessment({
            "newWord": "敲不死", "occupantWord": "情报所",
            "occupantCode": "qbs", "freeCode": "qbso",
        }, comparison)
        rendered = chat_render._format_candidate_ordering_assessment(
            assessment, {"qbs": 1, "qbso": 2},
        )
        self.assertIn("多领域 34（每百万 0.08）", rendered)
        self.assertIn("新闻 99（每百万 0.11）", rendered)
        self.assertIn("单边", rendered)
        self.assertIn("推荐空位 qbso", rendered)
        self.assertNotIn("常用度信号不足", rendered)
        self.assertNotIn("口语", rendered)
        self.assertNotIn("文学", rendered)
        self.assertNotIn("历史", rendered)
        self.assert_delivery(comparison)

    def test_both_attested_decide_with_both_figures_in_both_directions(self):
        self.seed_both()
        for first, second, verdict in (("蛋粉", "单份", "front_more_common"),
                                       ("单份", "蛋粉", "behind_more_common")):
            with self.subTest(first=first):
                comparison = asyncio.run(review.compare_word_commonness(first, second))
                self.assertEqual(comparison["verdict"], verdict)
                text = self.assert_delivery(comparison, "dffn")
                self.assertIn("每百万 0.0944", text)
                self.assertIn("每百万 0.0371", text)
                self.assertIn("2.54×", text)
                self.assertIn("「蛋粉」较「单份」更常用", text)
                self.assertNotIn("历史", text)
                self.assertNotIn("新闻", text)

    def test_both_attested_close_does_not_invent_a_winner(self):
        self.seed_both(0.08)
        comparison = asyncio.run(review.compare_word_commonness("蛋粉", "单份"))
        self.assertEqual(comparison["verdict"], "close")
        text = self.assert_delivery(comparison, "dffn")
        self.assertIn("常用度接近", text)
        self.assertNotIn("更常用", text)
        self.assertIn("推荐空位 dffno", text)

    def test_neither_attested_keeps_insufficient_and_historical_supplement(self):
        # Local comparator seam avoids exercising the unrelated web fallback.
        comparison = self.local_comparison("空例甲", "空例乙")
        self.assertEqual(comparison["verdict"], "not_enough_evidence")
        text = self.assert_delivery(comparison)
        self.assertIn("常用度信号不足", text)
        self.assertIn("四个现代频道均未收录", text)
        self.assertIn("历史补充", text)
        self.assertIn("未参与判定", text)
        self.assertNotIn("每百万", text)
        self.assertNotIn("古代汉语", text)
        self.assertNotIn("近代汉语", text)

    def test_one_sided_history_is_only_a_supplement(self):
        fixtures.seed_historical(self.db, counts={"古例甲": (12, 1)})
        comparison = self.local_comparison("古例甲", "古例乙")
        self.assertEqual(comparison["verdict"], "not_enough_evidence")
        text = self.assert_delivery(comparison)
        self.assertIn("历史补充", text)
        self.assertIn("古代汉语 12（每百万 1200）", text)
        self.assertIn("未参与判定", text)
        self.assertNotIn("近代汉语", text)

    def test_historical_supplement_is_explicit_only_when_allowed(self):
        fixtures.seed_historical(self.db, counts={"古例甲": (1234, 1), "古例乙": (12, 50)})
        comparison = asyncio.run(review.compare_word_commonness("古例甲", "古例乙"))
        self.assertEqual(comparison["decisionReason"], "bcc_historical_classical_frequency_ratio")
        text = self.assert_delivery(comparison)
        self.assertIn("现代四频道均未收录", text)
        self.assertIn("按古代汉语频次", text)
        self.assertIn("1,234（每百万 123400）", text)
        self.assertIn("12（每百万 1200）", text)
        self.assertIn("仅作历史语料末级破平", text)
        comparison = asyncio.run(review.compare_word_commonness("古例甲", "情报所"))
        text = self.assert_delivery(comparison)
        self.assertNotIn("古代汉语", text)
        self.assertNotIn("历史补充", text)

    def test_guard_and_explicit_advisory_keep_the_same_comparator_evidence(self):
        comparison = asyncio.run(review.compare_word_commonness("敲不死", "情报所"))
        assessment, _, state = candidate_fixture(comparison)
        state.server_candidates = state.server_candidates[:1]
        state.server_ordering_assessments = [{**assessment, "freeCode": ""}]
        guard = pending.candidate_commonness_guard_copy(
            state.word, state.server_candidates, state.server_occupied_words,
            state.server_ordering_assessments,
        )
        protected = harness.chat_commands_module._protected_eviction_response(state, "qbs", ("情报所",))
        advisory = same_code_reorder._advisory("敲不死", "情报所")
        for text in (guard, protected, advisory):
            self.assertIn(comparison["summary"], text)

    def test_render_sites_use_the_shared_evidence_renderer(self):
        comparison = asyncio.run(review.compare_word_commonness("敲不死", "情报所"))
        marker = "S65_SHARED_EVIDENCE"
        # Runtime guard: all candidate variants must deliver the shared result.
        for module in (commonness_copy, pending, commonness_query,
                       same_code_reorder, harness.chat_commands_module):
            original = module.render_commonness_summary
            with patch.object(module, "render_commonness_summary", return_value=marker) as render:
                if module is commonness_copy:
                    assessment, _, _ = candidate_fixture(comparison)
                    text = chat_render._format_candidate_ordering_assessment(assessment, {})
                elif module is pending:
                    text = pending.front_insert_recommendation_copy({
                        "newWord": "蛋粉", "occupantWord": "单份", "occupantCode": "dffn",
                        "freeCode": "dffno", "summary": comparison["summary"],
                    })
                elif module is same_code_reorder:
                    text = module._advisory("敲不死", "情报所")
                elif module is commonness_query:
                    from keytao_bot.utils.word_commonness import lookup_word_commonness
                    text = module.render_commonness_table(lookup_word_commonness(["敲不死", "情报所"]))
                else:
                    _, _, state = candidate_fixture(comparison)
                    text = module._protected_eviction_response(state, "qbs", ("情报所",))
                self.assertIn(marker, text, module.__name__)
                render.assert_called()
            self.assertIs(module.render_commonness_summary, original)


class CommonnessRenderArchitectureTests(unittest.TestCase):
    # Builders own verdict wording. The other exceptions are parsers/prompts or
    # audit-policy diagnostics, not pairwise user-facing renderers.
    VERDICT_OWNERS = {
        ("utils/keytao_review.py", "_reference_comparison_summary"),
        ("utils/keytao_review.py", "_web_fallback_summary"),
        ("utils/commonness_copy.py", "render_commonness_summary"),
    }
    NON_RENDERERS = {
        ("plugins/chat_routing.py", "_extract_referenced_word_targets"),
        ("utils/keytao_batch_review.py", "_call_llm"),
        ("utils/keytao_review.py", "can_llm_override_audit_issues"),
        ("skills/keytao-draft/tools.py", "_fallback_draft_audit_with_encode"),
    }
    SITES = {
        "plugins/chat_render.py": {"_format_candidate_ordering_assessment", "_format_reviewed_add_prompt"},
        "utils/pending_confirmation.py": {
            "front_insert_recommendation_copy", "candidate_commonness_guard_copy",
            "render_server_backed_single_word_candidates", "render_server_backed_single_word_lookup",
            "render_server_backed_batch_candidates", "render_server_backed_batch_lookup",
        },
        "utils/same_code_reorder.py": {"_advisory"},
        "utils/commonness_query.py": {"render_commonness_table"},
        "utils/keytao_review.py": {"_chain_recommendation_text", "build_review_note"},
        "plugins/chat_commands.py": {
            "_protected_eviction_response", "_generate_usage_comparison_note",
            "_format_code_chain_reorder_confirmation", "_reorder_evidence_lines",
            "_format_reorder_noop",
        },
    }

    def test_every_enumerated_render_site_reaches_shared_evidence(self):
        root = Path(review.__file__).parents[1]
        trees = {path: ast.parse((root / path).read_text()) for path in self.SITES}
        trees["utils/commonness_copy.py"] = ast.parse((root / "utils/commonness_copy.py").read_text())
        functions = {node.name: node for tree in trees.values() for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

        def reaches_shared(name, seen):
            if name == "render_commonness_summary":
                return True
            if name in seen or name not in functions:
                return False
            calls = [node.func.id for node in ast.walk(functions[name])
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
            return any(reaches_shared(call, seen | {name}) for call in calls)

        for path, names in self.SITES.items():
            for name in names:
                with self.subTest(path=path, name=name):
                    self.assertTrue(reaches_shared(name, set()), name)

    def test_no_consumer_hardcodes_a_commonness_verdict(self):
        root = Path(review.__file__).parents[1]
        verdict = re.compile(r"不弱于|更常用|常用度接近|的常用度(?:信号|证据)不足|现有证据未分出高低")
        # Discover new files too: a fourth/fifth renderer must not evade the
        # class guard merely because it was omitted from the site inventory.
        for path in root.rglob("*.py"):
            relative = str(path.relative_to(root))
            tree = ast.parse(path.read_text())
            for function in (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
                if (relative, function.name) in self.VERDICT_OWNERS | self.NON_RENDERERS:
                    continue
                for node in ast.walk(function):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        self.assertIsNone(verdict.search(node.value),
                                          f"{relative}:{node.lineno} hardcodes a verdict")


if __name__ == "__main__":
    unittest.main()
