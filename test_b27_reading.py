"""B27 multi-reading additions through real tools and offline QQ boundaries."""

import copy
import json
import sqlite3
import unittest
from contextlib import closing
from unittest.mock import patch
from urllib.parse import urlsplit

import test_b25_session as b25


WORD = "周内"
PRIMARY = ["fdns", "fdnsu", "fdnsui"]
ALTERNATE = ["fdnw", "fdnwu", "fdnwui"]
MODIFIERS = ("强制", "强行", "硬加")


class ReadingHTTP(b25.SessionHTTP):
    def __init__(self):
        super().__init__()
        self.extra_readings = [("nei", ALTERNATE)]
        self.entries = [
            {"word": word, "code": code, "type": "Phrase", "weight": 100}
            for word, code in ((WORD, "fdnsu"), ("周纳", "fdns"))
        ]

    def _response(self, method, url, kwargs):
        if urlsplit(url).path in ("/api/phrases/encode", "/api/bot/phrases/encode"):
            word = (kwargs.get("params") or {})["word"]
            if word in (WORD, "周纳"):
                self.requests.append((method, "/api/phrases/encode", {"word": word}))
                extras = self.extra_readings if word == WORD else []
                return b25.b19.b18.OfflineResponse({
                    "input": word, "type": "二字词", "codes": PRIMARY,
                    "altCodes": [code for _reading, codes in extras for code in codes],
                    "candidateCodes": PRIMARY + [code for _reading, codes in extras for code in codes],
                    "alternatePhrasePronunciationCodes": [
                        {"pinyins": ["zhou", reading], "codes": codes}
                        for reading, codes in extras
                    ],
                    "flyKeyVariants": [], "phrasePinyins": ["zhou", "na"],
                    "pronunciationSource": "zdic-phrase", "standardPronunciationStatus": "found",
                    "chars": [
                        {"char": "周", "pinyin": "zhou", "pinyins": ["zhou"],
                         "pronunciationLookupStatus": "found", "phoneticCode": "fd", "shapeCode": "u"},
                        {"char": word[1], "pinyin": "na", "pinyins": ["na", *[r for r, _c in extras]],
                         "pronunciationLookupStatus": "found", "phoneticCode": "ns", "shapeCode": "ui"},
                    ],
                })
        return super()._response(method, url, kwargs)


class B27ReadingTests(unittest.IsolatedAsyncioTestCase):
    tearDown = b25.B25SessionTests.tearDown
    matcher_send = b25.B25SessionTests.matcher_send
    matcher_finish = b25.B25SessionTests.matcher_finish
    created = b25.B25SessionTests.created
    history_for = b25.B25SessionTests.history_for
    remember = b25.B25SessionTests.remember
    greeting = b25.B25SessionTests.greeting
    turn = b25.B25SessionTests.turn

    def setUp(self):
        b25.B25SessionTests.setUp(self)
        self.http = ReadingHTTP()
        self.enterContext(patch.object(b25.b19.b18.TOOLS.http_client, "_keytao_client", self.http))
        self.replay.client.create.side_effect = TimeoutError("B27 semantic parser timeout")
        with closing(sqlite3.connect(b25.b24.review.reference_db_path())) as connection, connection:
            connection.execute("INSERT OR REPLACE INTO word_commonness VALUES (?, 30000, NULL, 3)", ("周纳",))
            connection.execute("UPDATE word_commonness SET corpus_frequency = 1 WHERE word = ?", (WORD,))

    def reset_session(self):
        self.replay.store.delete(self.replay.author)
        self.http.rows.clear()
        self.http.writes.clear()
        self.http.version = 0
        self.http.submissions = 0
        self.histories.clear()
        b25.b24.review._review_cache.clear()

    async def seed(self, word=WORD):
        await self.turn("喵喵 " + word, await self.greeting())
        self.original_id = self.bot.last_id
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        return record

    async def finish_add(self, message):
        await self.turn(message, self.original_id)
        if not self.http.writes:
            record = self.replay.store.get_record(self.replay.author)
            self.assertIsNotNone(record, self.replay.last_response)
            self.assertIsInstance(record.state, b25.b19.harness.PendingToolConfirm, self.replay.last_response)
            await self.turn("确认", self.bot.last_id)
        self.assertEqual(self.created(), [(WORD, ALTERNATE[0])], self.replay.last_response)
        self.assertTrue(self.http.rows[0]["needsManualReview"], self.http.rows)
        self.assertIn("管理员", self.replay.last_response)
        self.assertEqual(self.http.submissions, 0)

    async def test_k1_review_keeps_non_authoritative_encoder_reading(self):
        result = json.loads(await b25.commands.call_tool_function(
            "keytao_prepare_reviewed_add", {"word": WORD, "requested_code": ALTERNATE[0]},
            "qq", self.replay.author.actor_id,
        ))
        by_reading = {tuple(row["normalized"]): row for row in result["pronunciations"]}
        self.assertIn(("zhou", "nei"), by_reading, result)
        alternate = by_reading[("zhou", "nei")]
        self.assertEqual(alternate["codes"], ALTERNATE)
        self.assertTrue(alternate["requiresManualReview"])
        self.assertFalse(alternate["sources"], alternate)

    async def test_ordinary_add_keeps_authoritative_auto_review(self):
        self.http.entries = [row for row in self.http.entries if row["word"] != WORD]
        review = json.loads(await b25.commands.call_tool_function(
            "keytao_prepare_reviewed_add", {"word": WORD}, "qq", self.replay.author.actor_id,
        ))
        await self.turn("喵喵 加词 " + WORD, await self.greeting())
        self.assertEqual(self.http.writes, [])
        await self.turn("确认", self.bot.last_id)
        self.assertEqual(self.created(), [(WORD, "fdnsu")], self.replay.last_response)
        self.assertFalse(self.http.rows[0]["needsManualReview"], self.http.rows)
        self.assertTrue(review["autoReviewable"], review)

    async def test_k1_quoted_additional_code_creates_manual_review_item(self):
        await self.seed(WORD)
        await self.finish_add("加入编码 fdnw")

    async def test_quoted_authoritative_additional_code_keeps_auto_review(self):
        await self.seed(WORD)
        await self.turn("加入编码 fdnsui", self.original_id)
        if not self.http.writes:
            await self.turn("确认", self.bot.last_id)
        self.assertEqual(self.created(), [(WORD, "fdnsui")], self.replay.last_response)
        self.assertFalse(self.http.rows[0]["needsManualReview"], self.http.rows)
        self.assertEqual(
            self.http.rows[0]["remark"],
            "喵喵审词：读音 zhou na；权威来源、编码和常用度证据一致，允许本喵自动通过",
        )

    async def test_k1_wrong_code_still_refuses(self):
        await self.seed(WORD)
        await self.turn("加入编码 abcd", self.original_id)
        self.assertEqual(self.http.writes, [])
        self.assertIn("编码 abcd 的音码前缀不符", self.replay.last_response)
        self.assertIn("本次未写入", self.replay.last_response)

    async def modifier_adds(self, command):
        for modifier in MODIFIERS:
            with self.subTest(modifier=modifier):
                self.reset_session()
                await self.seed(WORD)
                await self.finish_add(modifier + command)

    async def test_k2_modifiers_route_additional_code(self):
        await self.modifier_adds("加入编码 fdnw")

    async def test_k2_modifiers_route_add_code(self):
        await self.modifier_adds("加入 fdnw")

    async def test_k2_modifiers_route_named_add(self):
        await self.modifier_adds("添加 周内 fdnw")

    async def test_k2_modifiers_preserve_exact_validation_refusal(self):
        for command in ("加入编码 abcd", "加入 abcd", "添加 周内 abcd"):
            self.reset_session()
            await self.seed(WORD)
            await self.turn(command, self.original_id)
            expected = self.replay.last_response
            self.assertIn("编码 abcd 的音码前缀不符", expected)
            for modifier in MODIFIERS:
                with self.subTest(command=command, modifier=modifier):
                    self.reset_session()
                    await self.seed(WORD)
                    await self.turn(modifier + command, self.original_id)
                    self.assertEqual(self.http.writes, [])
                    self.assertEqual(self.replay.last_response, expected)

    async def test_k3_existing_word_hint_uses_second_reading_first_code(self):
        await self.seed(WORD)
        self.assertIn("「周内」已在词库（fdnsu）", self.replay.last_response)
        self.assertIn("可再追加一个编码，格式为：加入编码 fdnw", self.replay.last_response)
        self.assertNotIn("<code>", self.replay.last_response)

    async def test_k3_existing_word_commonness_comparison_is_kept(self):
        await self.seed(WORD)
        self.assertIn("常用度对比：", self.replay.last_response)
        self.assertIn("「周纳」较「周内」更常用", self.replay.last_response)

    async def test_k3_without_second_reading_keeps_placeholder(self):
        self.http.extra_readings = []
        await self.seed(WORD)
        self.assertIn("可再追加一个编码，格式为：加入编码 <code>", self.replay.last_response)

    async def test_k3_single_reading_with_legacy_code_keeps_placeholder(self):
        self.http.extra_readings = []
        self.http.entries[0]["code"] = "fdnsuo"
        await self.seed(WORD)
        self.assertIn("可再追加一个编码，格式为：加入编码 <code>", self.replay.last_response)

    async def test_k3_multiple_other_readings_keep_placeholder(self):
        self.http.extra_readings.append(("ne", ["fdnf", "fdnfu", "fdnfui"]))
        await self.seed(WORD)
        self.assertIn("可再追加一个编码，格式为：加入编码 <code>", self.replay.last_response)

    async def test_other_speaker_cannot_use_owner_ticket(self):
        original = copy.deepcopy(await self.seed(WORD))
        for modifier in ("", *MODIFIERS):
            with self.subTest(modifier=modifier):
                await self.turn(modifier + "加入编码 fdnw", self.original_id, actor="b27-other-speaker")
                self.assertEqual(self.http.writes, [])
                self.assertEqual(self.replay.store.get_record(self.replay.author), original)
                self.assertIn("不是发给你的", self.replay.last_response)

    async def test_other_speaker_cannot_recover_owner_cold_quote(self):
        await self.seed(WORD)
        self.replay.store.delete(self.replay.author)
        for modifier in ("", *MODIFIERS):
            with self.subTest(modifier=modifier):
                await self.turn(modifier + "加入编码 fdnw", self.original_id, actor="b27-other-speaker")
                self.assertEqual(self.http.writes, [])
                self.assertIsNone(self.replay.store.get_record(self.replay.author))


if __name__ == "__main__":
    unittest.main()
