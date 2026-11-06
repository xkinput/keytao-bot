"""B23 candidate ordering with real local frequency lookups and rendering."""

import json
import socket
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
import test_b18_rest_real_tools as b18
from keytao_bot.utils import keytao_review as review
from keytao_bot.utils.bcc_reference import CHANNELS, HISTORICAL_CHANNELS, create_schema
from keytao_bot.utils.word_commonness import lookup_word_commonness


class B23CommonnessTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.db = Path(directory) / "reference.db"
        with closing(sqlite3.connect(self.db)) as connection, connection:
            create_schema(connection)
            connection.execute(
                "CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, "
                "part_of_speech TEXT, dictionary_presence_count INTEGER)"
            )
            connection.execute("INSERT INTO word_commonness VALUES ('羽衣甘蓝', NULL, NULL, 2)")
            for channel in (*CHANNELS, *HISTORICAL_CHANNELS):
                connection.execute("INSERT INTO bcc_dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (channel, channel, "word", "offline", "offline", "offline", 1, 1,
                     1377730, 835646749))
            connection.execute("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)",
                ("近代汉语", "用过", 1952, 1952 * 1000000 / 835646749, 33200))
        self.enterContext(patch.object(review, "reference_db_path", return_value=self.db))
        self.web = self.enterContext(patch.object(
            review, "_estimate_word_commonness_web_fallback",
            AsyncMock(side_effect=AssertionError("B23 must stay offline")),
        ))
        self.enterContext(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
        self.enterContext(patch.object(socket.socket, "connect_ex", side_effect=AssertionError("Offline only")))

    async def candidate_review(self):
        subject = {
            "success": True, "word": "羽衣甘蓝", "recommendedCode": "yygla",
            "pronunciations": [{
                "pinyin": "yu yi gan lan", "recommendedCode": "yygla",
                "candidateStatuses": [
                    {"code": "yygl", "occupied": True, "words": ["用过"], "label": "已有「用过」"},
                    {"code": "yygla", "occupied": False, "words": []},
                    {"code": "yyglao", "occupied": False, "words": []},
                ],
            }],
        }
        assessments = await review.assess_candidate_chain_commonness(subject)
        review.apply_candidate_ordering_recommendation(subject, assessments)
        prompt = harness.openai_chat_module._format_reviewed_add_prompt(subject)
        self.assertIsNotNone(prompt)
        self.web.assert_not_awaited()
        return subject, assessments[0], prompt

    async def test_historical_incumbent_keeps_empty_slot_recommendation(self):
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "behind_more_common")
        self.assertEqual(assessment["newCode"], "yygla")
        self.assertIn("2. yygla — ✅ 推荐（空位）", prompt)
        self.assertNotIn("需重排", prompt)
        self.assertIn("1,952", assessment["summary"])

    async def test_dictionary_counts_alone_never_reorder(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("DELETE FROM bcc_frequency")
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "not_enough_evidence")
        self.assertNotIn("需重排", prompt)

    async def test_dictionary_comparison_remains_available_to_read_only_lookup(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("DELETE FROM bcc_frequency")
            connection.execute("INSERT INTO word_commonness VALUES ('用过', NULL, NULL, 0)")
        comparison = await review.compare_word_commonness("羽衣甘蓝", "用过")
        self.assertEqual(comparison["verdict"], "front_more_common")
        self.assertEqual(comparison["decisionReason"], "dictionary_presence_margin")
        result = lookup_word_commonness(["用过", "羽衣甘蓝"])
        self.assertEqual([row["word"] for row in result["words"]], ["羽衣甘蓝", "用过"])
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "not_enough_evidence")
        self.assertNotIn("需重排", prompt)

    async def test_dictionary_counts_alone_cannot_reorder_current_chain(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("DELETE FROM bcc_frequency")
            connection.execute("INSERT INTO word_commonness VALUES ('用过', NULL, NULL, 0)")
        entries = [
            {"word": word, "code": "yygl", "type": "Phrase", "weight": index}
            for index, word in enumerate(("用过", "羽衣甘蓝"))
        ]
        result = await review.rank_code_chain_by_commonness(entries)
        self.assertEqual(result["status"], "ask", result)
        self.assertEqual(result["currentOrder"], entries)
        self.web.assert_not_awaited()

    async def test_dictionary_counts_alone_cannot_approve_css_change(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.executemany("INSERT INTO word_commonness VALUES (?, NULL, NULL, ?)",
                                   [("甲", 3), ("乙", 1)])
        http = b18.DraftHTTP()
        http.entries = [{"word": "乙", "code": "a", "type": "CSSSingle", "weight": 100}]
        with patch.object(review.http_client, "_keytao_client", http):
            result = await review.audit_draft_items(review.ReviewHttpConfig("offline", ""), [
                {"action": "Change", "word": "甲", "oldWord": "乙", "code": "a", "type": "CSSSingle"},
            ])
        comparison = result["reviewedWords"]["甲@CSSSingle"]["commonnessComparison"]
        self.assertEqual(comparison["verdict"], "front_more_common")
        self.assertEqual(comparison["decisionReason"], "dictionary_presence_margin")
        self.assertFalse(result["autoApprove"], result)
        self.assertIn("缺少足够的词库审查证据", result["issues"][0])
        self.assertEqual(result["approvedItems"], [])
        self.assertEqual(http.errors, [])
        self.assertEqual(http.writes, [])
        self.web.assert_not_awaited()

    async def test_dictionary_counts_alone_cannot_approve_advisory_priority(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.executemany("INSERT INTO word_commonness VALUES (?, NULL, NULL, ?)",
                                   [("甲", 3), ("乙", 1)])
        http = b18.DraftHTTP()
        http.entries = [
            {"word": word, "code": code, "type": "CSSSingle", "weight": 100}
            for word, code in (("甲", "a"), ("乙", "b"))
        ]
        items = [
            {"action": action, "word": word, "code": code, "type": "CSSSingle"}
            for action, word, code in (
                ("Delete", "甲", "a"), ("Delete", "乙", "b"),
                ("Create", "甲", "b"), ("Create", "乙", "c"),
            )
        ]
        with patch.object(review.http_client, "_keytao_client", http):
            result = await review.audit_draft_items(review.ReviewHttpConfig("offline", ""), items)
        self.assertEqual(len(result["commonnessComparisons"]), 1)
        comparison = result["commonnessComparisons"][0]["result"]
        self.assertEqual(comparison["verdict"], "front_more_common")
        self.assertEqual(comparison["decisionReason"], "dictionary_presence_margin")
        self.assertEqual(len(result["approvedItems"]), 2, result)
        self.assertTrue(all(item.startswith("调码删除原位：") for item in result["approvedItems"]))
        self.assertFalse(result["autoApprove"], result)
        self.assertEqual(http.errors, [])
        self.assertEqual(http.writes, [])
        self.web.assert_not_awaited()

    async def test_new_corpus_frequency_cannot_displace_incumbent_without_modern_data(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("UPDATE word_commonness SET corpus_frequency=1000 WHERE word='羽衣甘蓝'")
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "not_enough_evidence")
        self.assertNotIn("需重排", prompt)

    async def test_comparable_modern_corpus_frequency_still_reorders(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("UPDATE word_commonness SET corpus_frequency=1000 WHERE word='羽衣甘蓝'")
            connection.execute("INSERT INTO word_commonness VALUES ('用过', 100, 'v', 2)")
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygl", prompt)
        self.assertEqual(assessment["verdict"], "front_more_common")
        self.assertEqual(assessment["decisionReason"], "frequency_ratio")
        self.assertIn("需重排", prompt)

    async def test_comparable_modern_bcc_frequency_still_reorders(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.executemany("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)", [
                ("多领域", "羽衣甘蓝", 1000, 10, 100),
                ("多领域", "用过", 100, 1, 1000),
            ])
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygl", prompt)
        self.assertEqual(assessment["decisionReason"], "bcc_frequency_ratio")
        self.assertIn("需重排", prompt)

    async def test_historical_frequency_alone_cannot_displace_incumbent(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("UPDATE word_commonness SET dictionary_presence_count=1")
            connection.execute("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)",
                ("近代汉语", "羽衣甘蓝", 10000, 10000 * 1000000 / 835646749, 100))
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "not_enough_evidence")
        self.assertNotIn("需重排", prompt)

    async def test_historical_incumbent_without_dictionary_rows_stays_offline(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("DELETE FROM word_commonness")
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "behind_more_common")

    async def test_incumbent_corpus_frequency_wins_without_dictionary_support(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO word_commonness VALUES ('用过', 10, 'v', 0)")
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "behind_more_common")

    async def test_historical_advantage_cannot_override_close_modern_frequencies(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("UPDATE word_commonness SET corpus_frequency=10, dictionary_presence_count=1")
            connection.execute("INSERT INTO word_commonness VALUES ('用过', 12, 'v', 1)")
            connection.execute("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)",
                ("近代汉语", "羽衣甘蓝", 10000, 10000 * 1000000 / 835646749, 100))
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "not_enough_evidence")
        self.assertNotIn("需重排", prompt)

    async def test_incumbent_bcc_frequency_is_cited_as_keep_order_evidence(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)",
                ("多领域", "用过", 1000, 10, 100))
        subject, assessment, prompt = await self.candidate_review()
        self.assertEqual(subject["recommendedCode"], "yygla", prompt)
        self.assertEqual(assessment["verdict"], "behind_more_common")
        self.assertIn("多领域 1,000", assessment["summary"])
        self.assertIn("保留现有排序", assessment["summary"])
        self.assertNotIn("单边 BCC 收录不决定高低", assessment["summary"])

    def seed_historical_chain(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO word_commonness VALUES ('甲词', NULL, NULL, 3)")
            connection.execute("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, ?)",
                ("近代汉语", "乙词", 1, 1000000 / 835646749, 100))

    async def test_historical_only_later_entry_cannot_reorder_current_chain(self):
        self.seed_historical_chain()
        entries = [
            {"word": word, "code": "jsci", "type": "Phrase", "weight": index}
            for index, word in enumerate(("甲词", "乙词"))
        ]
        result = await review.rank_code_chain_by_commonness(entries)
        self.assertEqual(result["status"], "ask", result)
        self.assertEqual(result["currentOrder"], entries)
        self.web.assert_not_awaited()

    async def test_one_sided_modern_bcc_cannot_reorder_current_chain(self):
        fixture = json.loads((Path(__file__).parent / "e2e/fixtures/bcc/s63.json").read_text())
        with closing(sqlite3.connect(self.db)) as connection, connection:
            for table in ("bcc_dataset", "bcc_frequency", "word_commonness"):
                connection.execute(f"DELETE FROM {table}")
            for table, rows in (("bcc_dataset", fixture["datasets"]),
                                ("bcc_frequency", fixture["rows"]),
                                ("word_commonness", fixture["jieba"])):
                for row in rows:
                    connection.execute(
                        f'INSERT INTO {table} ({",".join(row)}) VALUES ({",".join("?" for _ in row)})',
                        tuple(row.values()),
                    )
        entries = [
            {"word": word, "code": "qbsi", "type": "Phrase", "weight": index}
            for index, word in enumerate(("敲不死", "情报所"))
        ]
        result = await review.rank_code_chain_by_commonness(entries)
        self.assertEqual(result["status"], "ask", result)
        self.assertEqual(result["currentOrder"], entries)
        self.web.assert_not_awaited()

    async def test_batch_create_collision_keeps_listed_order_with_history_only(self):
        self.seed_historical_chain()
        b18.B18RealToolTests.setUp(self)
        self.enterContext(patch.dict(b18.FACTS, {
            word: ("jsci", ["jsci", "jscia"], "jia ci" if word == "甲词" else "yi ci")
            for word in ("甲词", "乙词")
        }))
        items = [{"word": word, "code": "jsci", "type": "Phrase", "action": "Create"}
                 for word in ("甲词", "乙词")]
        preview = await b18.TOOLS.keytao_batch_add_to_draft(
            "qq", "b23-owner", items, batch_id=self.http.batch_id,
        )
        self.assertTrue(preview.get("requiresConfirmation"), preview)
        self.assertTrue(preview.get("collisionReplanned"), preview)
        self.assertEqual(self.http.writes, [])
        result = await b18.TOOLS.keytao_batch_add_to_draft(
            "qq", "b23-owner", preview["effectiveItems"], batch_id=self.http.batch_id,
            confirmed=True, expected_content_version=0, expected_warning_digest="a" * 64,
        )
        self.assertEqual(result.get("successCount"), 2, result)
        self.assertEqual(
            [(row["word"], row["code"]) for row in self.http.rows],
            [("甲词", "jsci"), ("乙词", "jscia")],
        )
        self.assertEqual(self.http.errors, [])
        self.web.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
