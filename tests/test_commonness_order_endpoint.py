"""HTTP contract tests for offline draft editor commonness ordering."""

from contextlib import closing
import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from keytao_bot.utils import keytao_batch_review, keytao_review
from keytao_bot.utils.bcc_reference import CHANNELS, create_schema


PATH = "/api/keytao/commonness-order"
AUTH = {"Authorization": "Bearer test-web-api-key"}


def entry(word, weight, **fields):
    return {"word": word, "weight": weight, "source": "current", **fields}


def chain(*entries, code="abcd", phrase_type="Phrase"):
    return {"code": code, "type": phrase_type, "entries": list(entries)}


class CommonnessOrderEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = FastAPI()
        web_api_spec = importlib.util.find_spec("keytao_bot.plugins.web_api")
        cls.web_api = importlib.util.module_from_spec(web_api_spec)
        driver = SimpleNamespace(config=SimpleNamespace(web_api_key="test-web-api-key"))
        nonebot = sys.modules["nonebot"]
        with (
            patch.object(nonebot, "get_driver", return_value=driver),
            patch.object(nonebot, "get_app", return_value=cls.app, create=True),
            patch.dict(sys.modules, {"keytao_bot.plugins.openai_chat": Mock()}),
        ):
            web_api_spec.loader.exec_module(cls.web_api)

        directory = cls.enterClassContext(tempfile.TemporaryDirectory())
        cls.reference_path = Path(directory) / "reference.sqlite"
        with closing(sqlite3.connect(cls.reference_path)) as db, db:
            create_schema(db)
            db.execute("CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, "
                       "part_of_speech TEXT, dictionary_presence_count INTEGER)")
            db.execute("INSERT INTO word_commonness VALUES ('龘', NULL, NULL, 3)")
            db.execute("CREATE TABLE standard_entry (word TEXT, source TEXT, level INTEGER)")
            db.executemany("INSERT INTO standard_entry VALUES (?, ?, ?)", [
                ("爱", "tgscc", 1), ("龘", "hsk3", 7),
            ])
            db.executemany("INSERT INTO bcc_dataset VALUES (?, ?, 'word', 'fixture', "
                           "'fixture', 'fixture', 1, 1, 1000, 1000000)",
                           [(channel, channel) for channel in CHANNELS])
            db.executemany("INSERT INTO bcc_frequency VALUES (?, ?, ?, ?, NULL)", [
                ("口语", "新闻甲", 7, 7), ("文学", "文学丙", 120, 120),
            ])

    def setUp(self):
        self.client = self.enterContext(TestClient(self.app))
        self.enterContext(patch.object(keytao_review, "reference_db_path", return_value=self.reference_path))
        for module, name in (
            (keytao_review, "_estimate_word_commonness_web_fallback"),
            (keytao_review, "_estimate_entity_knowledge_signal"),
            (keytao_review, "get_llm_client"),
            (keytao_batch_review, "get_llm_client"),
        ):
            guard = self.enterContext(patch.object(module, name, side_effect=AssertionError(name)))
            self.addCleanup(guard.assert_not_called)

    def post(self, *chains):
        return self.client.post(PATH, json={"chains": list(chains)}, headers=AUTH)

    def test_requires_existing_bearer_auth(self):
        with patch.object(self.web_api, "rank_code_chain_by_commonness") as rank:
            for headers in ({}, {"Authorization": "Bearer wrong"}):
                response = self.client.post(PATH, content="not JSON", headers=headers)
                self.assertEqual(401, response.status_code)
            rank.assert_not_called()

    def test_validation_returns_400_before_ranking(self):
        valid = chain(entry("甲", 1), entry("乙", None))
        invalid_chains = [
            [], [valid] * 21, [chain(entry("甲", 1))], [chain(*([entry("甲", 1)] * 9))],
            [chain(*valid["entries"], code="x" * 17)], [chain(*valid["entries"], code=123)],
            [chain(*valid["entries"], phrase_type="x" * 17)], [chain(*valid["entries"], phrase_type=None)],
            [chain(*(entry(f"词{number}", number) for number in range(start, start + 8)))
             for start in range(0, 128, 8)],
        ]
        for invalid_entry in (
            entry("", 1), entry(" ", 1), entry("词" * 31, 1), entry("甲\n", 1), entry("甲\u2028乙", 1),
            entry("甲", True), entry("甲", "1"), entry("甲", 1, source="other"),
            entry("甲", 1, prId="1"), entry("甲", 1, prId=None), entry("甲", 1, code="other"),
            entry("甲", 1, type="Other"), {"word": "甲", "source": "current"},
        ):
            invalid_chains.append([chain(invalid_entry, entry("乙", 2))])
        with patch.object(self.web_api, "rank_code_chain_by_commonness") as rank:
            for invalid in invalid_chains:
                response = self.post(*invalid)
                self.assertEqual(400, response.status_code, response.text)
            malformed = self.client.post(PATH, content="{", headers=AUTH)
            self.assertEqual(400, malformed.status_code)
            rank.assert_not_called()

    def test_offline_reorder_keep_ask_and_word_lines(self):
        with (
            patch.object(self.web_api, "rank_code_chain_by_commonness",
                         wraps=keytao_review.rank_code_chain_by_commonness) as rank,
            patch.object(keytao_batch_review, "item_commonness",
                         wraps=keytao_batch_review.item_commonness) as lines,
        ):
            response = self.post(
                chain(entry("文学丙", 1), entry("新闻甲", 2, source="draft", prId=7)),
                chain(entry("文学丙", 2), entry("新闻甲", 1), code="efgh"),
                chain(entry("unknown-d", None, source="draft"), entry("unknown-b", -1),
                      entry("unknown-c", None), entry("unknown-a", -1), code="ijkl"),
                chain(entry("爱", 1), entry("龘", 2), code="a", phrase_type="CSSSingle"),
            )
        self.assertEqual(200, response.status_code, response.text)
        result = response.json()
        self.assertIs(result["success"], True)
        reorder, keep, ask, authority = result["chains"]
        self.assertEqual(("abcd", "Phrase", "reorder", ["新闻甲", "文学丙"]),
                         (reorder["code"], reorder["type"], reorder["status"], reorder["orderedWords"]))
        self.assertTrue(reorder["evidence"])
        self.assertTrue(all(isinstance(line, str) for line in reorder["evidence"]))
        self.assertEqual(("keep", ["新闻甲", "文学丙"]), (keep["status"], keep["orderedWords"]))
        self.assertEqual(("ask", ["unknown-a", "unknown-b", "unknown-c", "unknown-d"]),
                         (ask["status"], ask["orderedWords"]))
        self.assertEqual("CSSSingle", authority["type"])
        self.assertEqual({
            "新闻甲": {"daily": "口语/多领域 每百万 7（BCC fixture）"},
            "文学丙": {"daily": "仅新闻/文学 每百万 120（BCC fixture）"},
            "爱": {"authority": "通用规范一级"},
            "龘": {"authority": "HSK 7-9级；词典收录 3"},
        }, result["words"])
        self.assertEqual(8, lines.call_count)
        self.assertEqual(4, rank.await_count)
        self.assertTrue(all(call.kwargs == {"offline_only": True} for call in rank.await_args_list))
        self.assertEqual({"code": "abcd", "type": "Phrase", "word": "新闻甲", "weight": 2,
                          "source": "draft", "prId": 7}, rank.await_args_list[0].args[0][1])

    def test_unsafe_proposals_keep_current_words_and_ask(self):
        original = chain(entry("甲", 2), entry("乙", 1))
        for words in (["乙", "丙"], ["乙", "甲", "乙"]):
            with patch.object(self.web_api, "rank_code_chain_by_commonness", return_value={
                "status": "reorder", "proposedOrder": [{"word": word} for word in words],
            }):
                response = self.post(original)
            self.assertEqual(200, response.status_code, response.text)
            [result] = response.json()["chains"]
            self.assertEqual(("ask", ["乙", "甲"]), (result["status"], result["orderedWords"]))

    def test_ranking_and_word_line_exceptions_are_local_to_affected_chains(self):
        real_rank = keytao_review.rank_code_chain_by_commonness
        real_lines = keytao_batch_review.item_commonness

        async def rank(entries, **kwargs):
            if entries[0]["code"] == "failed-rank":
                raise RuntimeError("ranking failed")
            return await real_rank(entries, **kwargs)

        def lines(word):
            if word == "broken":
                raise RuntimeError("line builder failed")
            return real_lines(word)

        with (
            patch.object(self.web_api, "rank_code_chain_by_commonness", side_effect=rank),
            patch.object(keytao_batch_review, "item_commonness", side_effect=lines),
            patch.object(self.web_api.logger, "warning") as warning,
        ):
            response = self.post(
                chain(entry("文学丙", None), entry("新闻甲", 1), code="failed-rank"),
                chain(entry("broken", None), entry("爱", 1)),
                chain(entry("文学丙", 1), entry("新闻甲", 2), code="healthy"),
            )
        self.assertEqual(200, response.status_code, response.text)
        result = response.json()
        self.assertEqual([
            ("ask", ["新闻甲", "文学丙"]), ("ask", ["爱", "broken"]), ("reorder", ["新闻甲", "文学丙"]),
        ], [(row["status"], row["orderedWords"]) for row in result["chains"]])
        self.assertNotIn("broken", result["words"])
        self.assertEqual({"authority": "通用规范一级"}, result["words"]["爱"])
        self.assertEqual(2, warning.call_count)
