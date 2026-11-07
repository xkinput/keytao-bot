"""B26 display and routing replays with real tools and offline QQ boundaries."""

import re
import sqlite3
import unittest
from contextlib import closing
from unittest.mock import patch

import test_b25_session as b25
from keytao_bot.utils.pending_confirmation import candidate_commonness_guard_copy


class B26DisplayTests(unittest.IsolatedAsyncioTestCase):
    setUp = b25.B25SessionTests.setUp
    tearDown = b25.B25SessionTests.tearDown
    matcher_send = b25.B25SessionTests.matcher_send
    matcher_finish = b25.B25SessionTests.matcher_finish
    created = b25.B25SessionTests.created
    history_for = b25.B25SessionTests.history_for
    remember = b25.B25SessionTests.remember
    greeting = b25.B25SessionTests.greeting
    turn = b25.B25SessionTests.turn
    seed = b25.B25SessionTests.seed
    partial_session = b25.B25SessionTests.partial_session
    finish_replacement = b25.B25SessionTests.finish_replacement

    async def lookup_replacement_product(self, message, word="替代品"):
        facts = {
            "替代品": (["tdpk", "tdpko", "tdpkoo"], "ti dai pin"),
            "查替代品": (["wtdp", "wtdpo", "wtdpoo"], "cha ti dai pin"),
            "查一下替代品": (["wyxt", "wyxto", "wyxtoo"], "cha yi xia ti dai pin"),
        }
        with patch.dict(b25.FACTS, facts):
            await self.turn(message, await self.greeting())
        record = self.replay.store.get_record(self.replay.author)
        self.assertIsNotNone(record, self.replay.last_response)
        self.assertIsInstance(record.state, b25.b19.harness.PendingAddWord, self.replay.last_response)
        self.assertEqual(record.state.word, word, self.replay.last_response)
        self.assertIn(facts[word][0][0], self.replay.last_response)
        self.assertEqual(self.http.writes, [])

    async def test_h0_glued_lookup_keeps_candidate_ticket(self):
        await self.lookup_replacement_product("喵喵 查词替代品")

    async def test_h0_short_glued_lookup_keeps_candidate_ticket(self):
        await self.lookup_replacement_product("查替代品", "查替代品")

    async def test_h0_polite_glued_lookup_keeps_candidate_ticket(self):
        await self.lookup_replacement_product("喵喵 查一下替代品", "查一下替代品")

    async def test_h0_nonoccupant_replacement_keeps_chat_route(self):
        await self.turn("收拾 替换 键", await self.greeting())
        self.assertEqual(self.replay.last_response, "这是离线解释。")
        self.assertEqual(self.http.requests, [])
        self.assertEqual(self.http.writes, [])
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

    def test_h0_command_verbs_are_not_displacement_words(self):
        for verb in ("查", "查词", "查一下", "搜", "加", "加词", "添加", "删除", "删掉",
                     "删", "删词", "删了", "移除", "修改", "改成", "顺延", "换码",
                     "查询", "搜索", "查看", "提交", "确认", "取消", "审词", "写入"):
            for operand in (verb, "「" + verb + "」"):
                with self.subTest(operand=operand):
                    self.assertIsNone(b25.commands.parse_reviewed_displacement(operand + "替代首诗"))

    async def test_h0_actual_reviewed_occupant_still_replaces(self):
        await self.seed()
        await self.finish_replacement("守时替代首诗")
        self.assertEqual(self.http.submissions, 0)

    async def test_h0_live_nonoccupant_replacement_keeps_original_chat_route(self):
        await self.seed("收拾")
        self.http.requests.clear()
        await self.turn("收拾 替换 键", self.original_id)
        self.assertEqual(self.replay.last_response, "这是离线解释。")
        self.assertEqual(self.http.requests, [])
        self.assertEqual(self.http.writes, [])
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_h0_live_actual_occupant_with_spaces_still_replaces(self):
        await self.seed()
        await self.finish_replacement("守时 替换 首诗")
        self.assertEqual(self.http.submissions, 0)

    async def test_h1_partial_receipt_has_one_actionable_replacement_line(self):
        reply_id = await self.partial_session()
        lines = self.replay.last_response.splitlines()
        occupied = "守时 3（edekoo）已有「首诗」，未加入。"
        actionable = "要顶替请回复「守时 3，顶替 首诗」"
        self.assertIn(occupied, lines)
        self.assertEqual(lines[lines.index(occupied) + 1], actionable)
        self.assertEqual(lines.count(actionable), 1)
        self.assertIn("未选择：", self.replay.last_response)
        self.assertNotIn("候选仍有效，请逐词处理", self.replay.last_response)
        record = self.replay.store.get_record(self.replay.author)
        self.assertEqual(record.state.args["_query_words"], [b25.WORD])
        self.http.writes.clear()
        await self.finish_replacement("守时 3，顶替 首诗", reply_id)

    async def test_h1_live_remainder_omits_redundant_hint(self):
        await self.partial_session()
        record = self.replay.store.get_record(self.replay.author)
        self.assertEqual(record.state.args["_query_words"], [b25.WORD])
        self.assertIn("未选择：「守时」；这些词本次未添加。", self.replay.last_response)
        self.assertNotIn("候选仍有效，请逐词处理", self.replay.last_response)

    @staticmethod
    def displacement_suggestions(text):
        return re.findall(r"添加 守时 ([a-z]+)，(?:挤掉|顶替) ([^」\n]+)", text)

    async def test_h2_no_empty_slot_prefers_insufficient_signal_occupant(self):
        with closing(sqlite3.connect(b25.b24.review.reference_db_path())) as connection, connection:
            connection.execute("UPDATE word_commonness SET corpus_frequency = NULL WHERE word = ?",
                               (b25.OCCUPANT,))
        await self.seed()
        suggestions = self.displacement_suggestions(self.replay.last_response)
        self.assertEqual(suggestions, [(b25.CODES[2], b25.OCCUPANT)], self.replay.last_response)
        self.assertIn("常用度信号不足", self.replay.last_response)
        self.assertEqual(self.http.writes, [])

    def test_h2_weaker_occupants_are_unique_and_limited_to_two(self):
        occupants = {code: [word] for code, word in zip(b25.CODES, ("收拾", "首饰", b25.OCCUPANT))}
        assessments = [
            {"newWord": b25.WORD, "occupantCode": code, "occupantWord": occupants[code][0],
             "verdict": verdict, "freeCode": ""}
            for code, verdict in zip(b25.CODES,
                                     ("not_enough_evidence", "front_more_common", "not_enough_evidence"))
        ]
        text = candidate_commonness_guard_copy(
            b25.WORD, [(code, True) for code in b25.CODES], occupants, assessments,
        )
        suggestions = self.displacement_suggestions(text)
        self.assertEqual(len(suggestions), 2, text)
        self.assertEqual(len({word for _code, word in suggestions}), 2, text)
        self.assertEqual(suggestions, [(b25.CODES[0], "收拾"), (b25.CODES[1], "首饰")], text)

    async def test_h3_replacement_receipt_labels_deleted_draft_item(self):
        await self.seed()
        await self.finish_replacement("添加 守时 edekoo，顶替 首诗")
        draft_lines = [line for line in self.replay.last_response.splitlines() if line.startswith("草稿内容：")]
        self.assertEqual(len(draft_lines), 1, self.replay.last_response)
        self.assertIn("删除 首诗 edekoo", draft_lines[0])
        self.assertIn("守时 edekoo", draft_lines[0])
        self.assertNotIn("、首诗 edekoo", draft_lines[0])
        self.assertEqual(self.http.submissions, 0)

    async def test_h3_unsubmitted_receipt_omits_draft_publication_status(self):
        await self.seed()
        await self.finish_replacement("添加 守时 edekoo，顶替 首诗")
        self.assertNotIn("发布状态：", self.replay.last_response)
        self.assertIn("草稿地址：https://keytao.rea.ink/batch/" + self.http.batch_id,
                      self.replay.last_response)
        self.assertEqual(self.http.submissions, 0)


if __name__ == "__main__":
    unittest.main()
