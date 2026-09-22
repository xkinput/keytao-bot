"""S67 offline production-slice rendering and placement contract regressions."""

import asyncio
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_state_machine as harness
import test_s63_bcc as fixtures
from keytao_bot.harness.authorization_grammar import strip_self_mention_tokens
from keytao_bot.utils import commonness_query as query, keytao_review as review
from keytao_bot.utils.bcc_reference import CHANNELS, format_frequency
from keytao_bot.utils.word_commonness import lookup_word_commonness


MESSAGES = ("喵喵 词频排序：咽 嘢 强 鎗", "喵喵 词频排序：之乎 鎗 敲不死")
COMMAND = "把 咽 yeoiav 排到 嘢 yeoiav 前面"


class S67RankingTests(unittest.TestCase):
    def setUp(self):
        fixtures.BccCommonnessTests.setUp(self)
        fixture = json.loads((Path(__file__).parent / "e2e/fixtures/bcc/s67.json").read_text())
        with closing(sqlite3.connect(self.db)) as connection, connection:
            for table, rows in (("bcc_frequency", fixture["rows"]), ("word_commonness", fixture["jieba"])):
                for row in rows:
                    connection.execute(f'INSERT INTO {table} ({",".join(row)}) VALUES ({",".join("?" for _ in row)})', tuple(row.values()))
        query.reset_commonness_delivery()
        self.addCleanup(query.reset_commonness_delivery)
        self.owner = harness.ConversationAddress.private("qq", "s67-fixture")

    async def reply(self, message=MESSAGES[0], *, weights=(10, 11), entries=None):
        message = strip_self_mention_tokens(message)
        words = query.parse_commonness_query(message)
        rows = entries if entries is not None else [
            {"word": word, "code": "yeoiav", "type": "Single", "weight": weight}
            for word, weight in zip(("咽", "嘢"), weights)
        ]
        calls = []

        async def dispatch(name, args, *actor):
            calls.append(name)
            self.assertEqual(actor, ("qq", self.owner.actor_id))
            if name == "keytao_lookup_by_words_batch":
                self.assertEqual(args, {"words": list(words)})
                return json.dumps({"success": True, "results": [
                    {"word": word, "phrases": [row for row in rows if row["word"] == word]}
                    for word in words
                ]})
            if name == "keytao_encode":
                return json.dumps({"success": True, "word": args["word"], "codes": []})
            raise AssertionError("Unexpected external or mutation tool: " + name)

        chat = harness.openai_chat_module
        memory = SimpleNamespace(conversation_address=self.owner, platform="qq")
        ctx = SimpleNamespace(normalized_message_text=message, platform="qq", user_id=self.owner.actor_id,
                              conv_key=self.owner, memory_context=memory, bot=object(), event=object(), QQMessageSegment=None)
        delivered = []

        async def finish(_bot, _event, _user, context, response, _segment):
            delivered.append(chat._prepare_user_facing_reply(response, context))

        token = chat._current_turn_message.set(message)
        try:
            with (patch.object(harness.chat_commands_module, "call_tool_function", side_effect=dispatch),
                  patch.object(chat, "remember_conversation"),
                  patch.object(chat, "_finish_ai_chat_response", side_effect=finish),
                  patch.object(chat, "get_ai_response_core", side_effect=AssertionError("Model forbidden")),
                  patch.object(chat, "_classify_message_command_intent", side_effect=AssertionError("Model forbidden"))):
                self.assertTrue(await chat._stage_handle_commonness_query(ctx))
        finally:
            chat._current_turn_message.reset(token)
        self.assertEqual(len(delivered), 1)
        self.assertTrue(query.matches_commonness_delivery(delivered[0], self.owner, message))
        self.web.assert_not_called()
        return delivered[0]

    def cells(self, text, word):
        return next(line.split(" | ") for line in text.splitlines() if f" | {word} | " in line)

    def test_modern_absent_row_names_jieba_and_absence(self):
        text = asyncio.run(self.reply())
        frequency = self.cells(text, "鎗")[2]
        self.assertIn("现代四频道未收录", frequency)
        self.assertIn("jieba 参考 10", frequency)
        self.assertNotRegex(frequency, r"^\d+$")

    def test_current_order_consistent_has_no_advertisement(self):
        for weights in ((10, 11), (10, 10)):
            with self.subTest(weights=weights):
                text = asyncio.run(self.reply(weights=weights))
                self.assertFalse(query.advertised_command_suggestions(text))

    def test_reversed_current_order_advertises_real_change(self):
        text = asyncio.run(self.reply(weights=(11, 10)))
        self.assertEqual(query.advertised_command_suggestions(text), (COMMAND,))

    def test_unknown_relation_does_not_use_display_order_as_verdict(self):
        rows = [{"word": word, "code": "yeoiav", "type": "Single", "weight": weight}
                for word, weight in (("嘢", 11), ("鎗", 10))]
        text = asyncio.run(self.reply("词频排序：嘢 鎗", entries=rows))
        self.assertFalse(query.advertised_command_suggestions(text))

    def test_close_relation_does_not_advertise(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("INSERT INTO word_commonness VALUES ('甲词',100,'n',1)")
            connection.execute("INSERT INTO word_commonness VALUES ('乙词',110,'n',1)")
        rows = [{"word": word, "code": "abcd", "type": "Phrase", "weight": weight}
                for word, weight in (("甲词", 101), ("乙词", 100))]
        text = asyncio.run(self.reply("词频排序：甲词 乙词", entries=rows))
        self.assertFalse(query.advertised_command_suggestions(text))

    def test_row_status_is_attestation_across_pair_kinds(self):
        for words in (("咽", "嘢"), ("咽", "鎗"), ("之乎", "鎗"), ("敲不死", "空例词")):
            with self.subTest(words=words):
                text = query.render_commonness_table(lookup_word_commonness(words))
                self.assertIn("词典收录 | 数据收录 | 历史补充", text)
                for word in words:
                    status = self.cells(text, word)[4]
                    if word in ("咽", "嘢"):
                        self.assertEqual(status, "现代 BCC 收录")
                    elif word == "之乎":
                        self.assertEqual(status, "仅历史 BCC 收录")
                    elif word == "鎗":
                        self.assertEqual(status, "jieba、历史 BCC 收录")
                    else:
                        self.assertEqual(status, "无数据")

    def test_second_production_message_is_explicit_about_history(self):
        text = asyncio.run(self.reply(MESSAGES[1]))
        self.assertIn("现代四频道未收录", self.cells(text, "之乎")[2])
        self.assertEqual(self.cells(text, "之乎")[4], "仅历史 BCC 收录")
        self.assertIn("近代汉语 1,311", self.cells(text, "之乎")[5])
        self.assertFalse(query.advertised_command_suggestions(text))

    def test_verdict_uses_table_precision_two_strongest_channels_and_exact_ratio(self):
        for words in (("曰", "强"), ("强", "曰")):
            result = lookup_word_commonness(words)
            summary = result["comparisons"][0]["summary"]
            self.assertIn("所用频次比 32.09×", summary)
            self.assertLessEqual(sum(channel in summary for channel in CHANNELS), 2)
            self.assertNotIn("975.6388", summary)
            for count, rate in re.findall(r"([\d,]+)（每百万 ([\d.e+-]+)）", summary):
                self.assertIn(f"{count}（每百万 {rate}）", "\n".join(
                    self.cells(query.render_commonness_table(result), word)[2] for word in words))
            # The headline maximum of each word must remain visible, even in different channels.
            for row in result["words"]:
                strongest = max((r for r in row["bcc"]["channels"] if r["count"] is not None), key=lambda r: r["perMillion"])
                self.assertIn(format_frequency(strongest["count"], strongest["perMillion"]), summary)

    def test_unavailable_bcc_and_other_legacy_sources_never_show_bare_number(self):
        result = lookup_word_commonness(("鎗", "之乎"))
        row = next(row for row in result["words"] if row["word"] == "鎗")
        row["bcc"] = {"available": False}
        for source in ("jieba", "legacy-fixture"):
            row["corpusSource"] = source
            frequency = self.cells(query.render_commonness_table(result), "鎗")[2]
            self.assertIn(f"{source} 参考 10", frequency)
            self.assertNotIn("未收录", frequency)

    def test_shorter_existing_code_is_not_advertised_as_a_move(self):
        async def run():
            for first_code, second_code, expected in (("abcd", "abcda", ()),
                                                       ("abcda", "abcd", ("把 甲词 放在 乙词 前面",))):
                async def read(name, args):
                    if name == "keytao_lookup_by_words_batch":
                        return {"success": True, "results": [
                            {"word": word, "phrases": [{"word": word, "code": code, "type": "Phrase", "weight": 100}]}
                            for word, code in (("甲词", first_code), ("乙词", second_code))
                        ]}
                    self.assertEqual(name, "keytao_encode")
                    return {"success": True, "word": args["word"], "codes": ["abcd", "abcda"]}
                self.assertEqual(await query._placement_commands(("甲词", "乙词"), [
                    {"frontWord": "乙词", "behindWord": "甲词", "verdict": "behind_more_common"},
                ], read), expected)
        asyncio.run(run())

    def test_missing_weight_or_duplicate_slot_suppresses_advertisement(self):
        base = [{"word": word, "code": "yeoiav", "type": "Single", "weight": weight}
                for word, weight in (("咽", 11), ("嘢", 10))]
        for rows in ([{k: v for k, v in row.items() if k != "weight"} for row in base],
                     [*base, base[0]]):
            with self.subTest(rows=rows):
                text = asyncio.run(self.reply(entries=rows))
                self.assertFalse(query.advertised_command_suggestions(text))

    def test_balanced_tiebreak_and_tiny_rate_use_shared_precision(self):
        first, second = [review._query_commonness_reference(word) for word in ("咽", "嘢")]
        # Synthetic equal maxima force the existing balanced-channel tiebreak.
        for reference, balanced in ((first, 0.004123456), (second, 0.00123456)):
            bcc = reference["bcc"]
            bcc["perMillion"] = 10.0
            for row in bcc["channels"]:
                row.update(count=1, perMillion=balanced if row["channel"] == "多领域" else 10.0)
        comparison = review._compare_reference_commonness("咽", "嘢", first, second)
        self.assertIn("balanced_tiebreak", comparison["decisionReason"])
        summary = comparison["summary"]
        self.assertIn("采用多领域比较", summary)
        self.assertIn("所用频次比 3.34×", summary)
        self.assertLessEqual(sum(channel in summary for channel in CHANNELS), 2)
        for value in (0.004123456, 0.00123456):
            self.assertIn(format_frequency(1, value), summary)

    def test_reference_unavailable_is_not_reported_as_modern_absence(self):
        with patch.object(review, "reference_db_path", return_value=self.db.parent / "missing.db"):
            text = query.render_commonness_table(lookup_word_commonness(("咽", "嘢")))
        for word in ("咽", "嘢"):
            self.assertEqual(self.cells(text, word)[2], "—")
            self.assertEqual(self.cells(text, word)[4], "无数据")
        self.assertNotIn("现代四频道未收录", text)


if __name__ == "__main__":
    unittest.main()
