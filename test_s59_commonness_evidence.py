"""S59 local evidence contracts use only temporary SQLite fixtures."""

import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, patch

from keytao_bot.utils import keytao_review as review
from keytao_bot.utils.word_commonness import (
    lookup_word_commonness,
    validate_commonness_words,
)


class CommonnessEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="s59-commonness-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "reference.db"
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "CREATE TABLE word_commonness (word TEXT PRIMARY KEY, "
                "corpus_frequency INTEGER, part_of_speech TEXT, "
                "dictionary_presence_count INTEGER)"
            )
        self.path_patch = patch.object(review, "reference_db_path", return_value=self.path)
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        self.web = AsyncMock(side_effect=AssertionError("S59 must stay offline"))
        self.web_patch = patch.object(review, "_estimate_word_commonness_web_fallback", self.web)
        self.web_patch.start()
        self.addCleanup(self.web_patch.stop)

    def insert(self, *rows):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.executemany("INSERT INTO word_commonness VALUES (?, ?, ?, ?)", rows)

    def test_incident_words_use_review_rule_and_retain_absence(self):
        self.insert(
            ("耶博", 100, "n", 2),
            ("伊莎贝拉", 1000, "nr", 3),
            ("夜泊", 400, "v", 3),
        )
        result = lookup_word_commonness(["耶博", "伊莎贝拉", "夜泊", "一身本领"])
        self.assertTrue(result["success"])
        self.assertTrue(result["referenceAvailable"])
        self.assertEqual(result["method"], "offline_reference")
        self.assertEqual(
            [row["word"] for row in result["words"]],
            ["伊莎贝拉", "夜泊", "耶博", "一身本领"],
        )
        self.assertEqual([row["rank"] for row in result["words"]], [1, 2, 3, None])
        self.assertEqual([row["verdict"] for row in result["words"]], ["ranked"] * 3 + ["unknown"])
        self.assertEqual(result["words"][-1]["dictionaryPresenceCount"], None)
        self.assertEqual(result["words"][-1]["corpusFrequency"], None)
        self.assertFalse(result["words"][-1]["known"])
        comparison = next(item for item in result["comparisons"] if
                          item["frontWord"] == "耶博" and item["behindWord"] == "夜泊")
        expected = review._compare_reference_commonness(
            "耶博", "夜泊", review._query_commonness_reference("耶博"),
            review._query_commonness_reference("夜泊"),
        )
        self.assertEqual(comparison, expected)
        self.assertIn(comparison["summary"], review.current_commonness_evidence())
        self.web.assert_not_called()

    def test_close_is_not_sorted_by_weighted_score(self):
        self.insert(("低频", 100, "n", 1), ("高频", 150, "n", 3))
        result = lookup_word_commonness(["低频", "高频"])
        self.assertEqual([row["word"] for row in result["words"]], ["低频", "高频"])
        self.assertEqual([row["verdict"] for row in result["words"]], ["close", "close"])
        self.assertEqual(result["comparisons"][0]["verdict"], "close")

    def test_equal_frequency_and_unknown_relation_remain_explicit(self):
        self.insert(("相同甲", 100, "n", 1), ("相同乙", 100, "n", 1))
        result = lookup_word_commonness(["相同甲", "相同乙"])
        self.assertEqual([row["verdict"] for row in result["words"]], ["close", "close"])
        self.insert(("词典甲", None, None, 1), ("词典乙", None, None, 2))
        result = lookup_word_commonness(["词典甲", "词典乙"])
        self.assertTrue(all(row["known"] for row in result["words"]))
        self.assertEqual([row["verdict"] for row in result["words"]], ["unknown", "unknown"])
        self.assertEqual(result["comparisons"][0]["verdict"], "not_enough_evidence")

    def test_dictionary_only_and_valid_zero_presence_keep_missing_distinct(self):
        self.insert(("语料词", 100, "n", 0), ("词典词", None, None, 3))
        result = lookup_word_commonness(["语料词", "词典词", "未知词"])
        by_word = {row["word"]: row for row in result["words"]}
        self.assertEqual(by_word["语料词"]["dictionaryPresenceCount"], 0)
        self.assertTrue(by_word["词典词"]["known"])
        self.assertIsNone(by_word["词典词"]["corpusFrequency"])
        self.assertIsNone(by_word["未知词"]["dictionaryPresenceCount"])
        self.assertEqual(result["words"][0]["word"], "词典词")

    def test_zero_frequency_is_invalid_reference_data_not_absence_coerced_to_zero(self):
        self.insert(("零频", 0, "n", 1))
        row = lookup_word_commonness(["零频"])["words"][0]
        self.assertFalse(row["known"])
        self.assertFalse(row["referenceAvailable"])
        self.assertIsNone(row["corpusFrequency"])
        self.assertIsNone(row["dictionaryPresenceCount"])

    def test_absent_and_unavailable_database_never_use_web(self):
        for path, expected_available in ((self.path, True), (self.path.with_name("missing.db"), False)):
            with self.subTest(path=path), patch.object(review, "reference_db_path", return_value=path):
                result = lookup_word_commonness(["不存在甲", "不存在乙"])
                self.assertEqual(result["referenceAvailable"], expected_available)
                self.assertTrue(all(not row["known"] for row in result["words"]))
                self.assertTrue(all(row["rank"] is None for row in result["words"]))
        self.web.assert_not_called()

    def test_bounded_input_validation_precedes_database_lookup(self):
        self.assertEqual(validate_commonness_words(["  词  "]), ["词"])
        self.assertEqual(len(validate_commonness_words([f"词{i}" for i in range(12)])), 12)
        invalid = (None, "甲乙", [], [""], [None], ["甲", "甲"], ["甲", " 甲 "],
                   ["x" * 65], ["甲\n乙"], [f"词{i}" for i in range(13)])
        with patch.object(review, "_query_commonness_reference", side_effect=AssertionError("must validate first")):
            for words in invalid:
                with self.subTest(words=words), self.assertRaises(ValueError):
                    lookup_word_commonness(words)
        with self.assertRaises(ValueError):
            validate_commonness_words(["单词"], min_words=2)

    def test_shared_stable_order_matches_review_chain_and_detects_cycles(self):
        self.insert(("甲词", 100, "n", 2), ("乙词", 150, "n", 2), ("丙词", 400, "n", 2))
        words = ["甲词", "乙词", "丙词"]
        result = lookup_word_commonness(words)
        chain = asyncio.run(review.rank_code_chain_by_commonness([
            {"word": word, "weight": index + 1} for index, word in enumerate(words)
        ]))
        self.assertEqual([row["word"] for row in result["words"]],
                         [row["word"] for row in chain["proposedOrder"]])
        self.assertIsNone(review._stable_commonness_order(words, {
            "甲词": {"乙词"}, "乙词": {"丙词"}, "丙词": {"甲词"},
        }))

    def test_conflicting_evidence_retains_all_rows_without_claiming_ranks(self):
        self.insert(("语料领先", 100, "n", 0), ("两类收录", 50, "n", 3),
                    ("词典领先", None, None, 3))
        words = ["语料领先", "两类收录", "词典领先", "未知词"]
        result = lookup_word_commonness(words)
        self.assertEqual(result["ordering"], "conflicting_evidence")
        self.assertEqual([row["word"] for row in result["words"]], words)
        self.assertTrue(all(row["rank"] is None for row in result["words"]))
        self.assertTrue(all(row["verdict"] == "unknown" for row in result["words"]))
        self.assertEqual([row["corpusFrequency"] for row in result["words"]], [100, 50, None, None])


if __name__ == "__main__":
    unittest.main()
