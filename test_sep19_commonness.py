"""Offline regressions for natural comparison routing and evidence order."""

import asyncio
from contextlib import closing
import json
from pathlib import Path
import socket
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_state_machine as harness

from keytao_bot.utils import commonness_query as query
from keytao_bot.utils import keytao_review as review


MESSAGES = (
    "偿债 和 从众，谁的词频更高",
    "“偿债”和“从众”这两个词的词频如何",
)
WORDS = ("偿债", "从众")
chat = harness.openai_chat_module
commands = harness.chat_commands_module


class Sep19CommonnessTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="sep19-commonness-")
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "reference.db"
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute(
                "CREATE TABLE word_commonness (word TEXT PRIMARY KEY, "
                "corpus_frequency INTEGER, part_of_speech TEXT, "
                "dictionary_presence_count INTEGER)"
            )
            connection.executemany("INSERT INTO word_commonness VALUES (?, ?, ?, ?)", (
                ("偿债", 100, "v", 3), ("从众", 400, "v", 3),
            ))
        for patcher in (
            patch.object(review, "reference_db_path", return_value=path),
            patch.object(review, "_estimate_word_commonness_web_fallback",
                         side_effect=AssertionError("Network evidence is forbidden")),
            patch.object(socket.socket, "connect",
                         side_effect=AssertionError("Network sockets are forbidden")),
            patch.object(socket, "create_connection",
                         side_effect=AssertionError("Network sockets are forbidden")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        query.reset_commonness_delivery()
        self.addCleanup(query.reset_commonness_delivery)

    def test_frequency_question_extracts_only_two_literal_words(self):
        self.assertEqual(query.parse_commonness_query(MESSAGES[0]), WORDS)

    def test_quoted_frequency_question_extracts_only_two_literal_words(self):
        self.assertEqual(query.parse_commonness_query(MESSAGES[1]), WORDS)

    def test_pair_question_variants_keep_literal_words_and_question_marks(self):
        for message in (
            MESSAGES[0] + "？", MESSAGES[1] + "?",
            "偿债和从众，谁的词频更高？",
            "偿债、从众这两个词的词频如何",
            "「偿债」和「从众」这两个词的词频怎么样？",
            '"偿债"与"从众"这2个词的词频怎样',
        ):
            with self.subTest(message=message):
                self.assertEqual(query.parse_commonness_query(message), WORDS)
        self.assertEqual(query.parse_commonness_query(
            "“共和国”和“和平”这两个词的词频如何？",
        ), ("共和国", "和平"))
        for message in (
            "偿债 和 从众", "“偿债”和“从众”", "偿债从众", "谁的词频更高",
            "偿债这两个词的词频如何", "偿债、偿债，谁的词频更高",
            "偿债、从众、团圆这两个词的词频如何",
            "“偿债”和“从众这两个词的词频如何",
            "“偿债”和“从众」这两个词的词频如何",
            "偿债、" + "词" * 33 + "，谁的词频更高",
        ):
            with self.subTest(message=message):
                self.assertIsNone(query.parse_commonness_query(message))

    def test_frequency_questions_finish_in_read_only_stage_without_pending_write(self):
        async def run(message, with_stale_pending):
            owner = harness.ConversationAddress.private("qq", "sep19-commonness")
            store = harness.MemoryConversationStateStore()
            if with_stale_pending:
                self.assertTrue(store.set(owner, harness.PendingToolConfirm(
                    function_name="keytao_batch_add_to_draft",
                    args={"items": [{"word": "谁的词频更高", "code": "uvvd", "type": "Phrase"}]},
                )))
            previous_record = store.get_record(owner)
            calls = []
            delivered = []

            async def dispatch(name, args, *actor):
                calls.append((name, args))
                self.assertEqual(actor, ("qq", owner.actor_id))
                codes = {"偿债": "uvvd", "从众": "csva"}
                if name == "keytao_lookup_by_words_batch":
                    self.assertEqual(args, {"words": list(WORDS)})
                    return json.dumps({"success": True, "results": [
                        {"word": word, "phrases": [{
                            "word": word, "code": code, "type": "Phrase", "weight": 100,
                        }]}
                        for word, code in codes.items()
                    ]})
                self.assertEqual(name, "keytao_encode")
                self.assertEqual(set(args), {"word"})
                self.assertIn(args["word"], WORDS)
                return json.dumps({"success": True, "word": args["word"],
                                   "codes": [codes[args["word"]]]})

            async def finish(_bot, _event, _user, memory, response, _segment):
                delivered.append(chat._prepare_user_facing_reply(response, memory))

            memory = SimpleNamespace(conversation_address=owner, platform="qq")
            ctx = SimpleNamespace(
                normalized_message_text=message, platform="qq", user_id=owner.actor_id,
                conv_key=owner, memory_context=memory, bot=object(), event=object(),
                QQMessageSegment=None,
            )
            token = chat._current_turn_message.set(message)
            try:
                with (
                    patch.object(chat, "conversation_state_store", store),
                    patch.object(commands, "conversation_state_store", store),
                    patch.object(commands, "call_tool_function", side_effect=dispatch),
                    patch.object(chat, "remember_conversation"),
                    patch.object(chat, "_finish_ai_chat_response", side_effect=finish),
                    patch.object(chat, "get_ai_response_core",
                                 side_effect=AssertionError("Models are forbidden")),
                    patch.object(chat, "_classify_message_command_intent",
                                 side_effect=AssertionError("Classifiers are forbidden")),
                ):
                    self.assertTrue(await chat._stage_handle_commonness_query(ctx))
            finally:
                chat._current_turn_message.reset(token)
            self.assertEqual(len(delivered), 1)
            self.assertIn("1 | 从众 | 400 | 3", delivered[0])
            self.assertIn("2 | 偿债 | 100 | 3", delivered[0])
            self.assertEqual(calls[0], ("keytao_lookup_by_words_batch", {"words": list(WORDS)}))
            self.assertTrue(all(name in {"keytao_lookup_by_words_batch", "keytao_encode"}
                                for name, _args in calls))
            self.assertIs(store.get_record(owner), previous_record)
            self.assertNotIn("加入草稿", delivered[0])
            self.assertNotIn("谁的词频更高", delivered[0])

        for message in MESSAGES:
            for with_stale_pending in (False, True):
                with self.subTest(message=message, with_stale_pending=with_stale_pending):
                    asyncio.run(run(message, with_stale_pending))

    def test_negated_reported_and_mixed_write_comparisons_remain_unmatched(self):
        for message in (
            "不要" + MESSAGES[0],
            "他说" + MESSAGES[0],
            "「" + MESSAGES[0] + "」",
            MESSAGES[0] + "，然后删除从众",
            MESSAGES[1] + "并加入草稿",
            "偿债 和 从众 和 删除加量，谁的词频更高",
            "“偿债”和“从众”这两个词的词频如何；提交草稿",
        ):
            with self.subTest(message=message):
                self.assertIsNone(query.parse_commonness_query(message))

    def test_reference_summary_numbers_follow_named_comparison_order(self):
        for rare_presence in (3, 1):
            with self.subTest(rare_presence=rare_presence):
                rare = {"available": True, "attested": True,
                        "corpusFrequency": 12, "dictionaryPresenceCount": rare_presence}
                common = {"available": True, "attested": True,
                          "corpusFrequency": 185, "dictionaryPresenceCount": 3}
                backwards = review._compare_reference_commonness("头痛医头", "团圆", rare, common)
                forwards = review._compare_reference_commonness("团圆", "头痛医头", common, rare)
                self.assertEqual(backwards["verdict"], "behind_more_common")
                self.assertEqual(forwards["verdict"], "front_more_common")
                self.assertEqual(backwards["summary"], forwards["summary"])
                self.assertIn("「团圆」较「头痛医头」更常用", backwards["summary"])
                self.assertIn("语料频次 185 vs 12", backwards["summary"])
                self.assertIn(f"词典收录 3 vs {rare_presence}", backwards["summary"])


if __name__ == "__main__":
    unittest.main()
