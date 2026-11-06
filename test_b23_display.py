"""B23 selection receipts through real draft tools and offline QQ replies."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import test_b19_partial_reply_binding as b19


class B23DisplayTests(unittest.IsolatedAsyncioTestCase):
    setUp = b19.B19PartialReplyBindingTests.setUp
    tearDown = b19.B19PartialReplyBindingTests.tearDown
    matcher_send = b19.B19PartialReplyBindingTests.matcher_send
    matcher_finish = b19.B19PartialReplyBindingTests.matcher_finish
    created = b19.B19PartialReplyBindingTests.created
    seed = b19.B19PartialReplyBindingTests.seed
    turn = b19.B19PartialReplyBindingTests.turn

    async def test_single_word_selection_omits_unselected_codes(self):
        word = "羽衣甘蓝"
        codes = ["yygl", "yygla", "yyglao"]
        with patch.dict(b19.b18.FACTS, {word: (codes[1], codes, "yu yi gan lan")}):
            candidates = [(code, code == "yygl") for code in codes]
            state = b19.harness.PendingAddWord(
                word, codes[1], candidates, server_candidates=candidates,
                server_occupied_words={"yygl": ["用过"]},
                pronunciation_codes={code: "yu yi gan lan" for code in codes},
                needs_manual_review=True,
            )
            self.assertTrue(self.replay.store.set(self.replay.author, state))
            record = self.replay.store.get_record(self.replay.author)
            prompt = b19.chat._render_live_single_candidate_record(record)
            await b19.REAL_FINISH(
                self.bot, SimpleNamespace(message_id=1), self.replay.author.actor_id,
                self.replay.memory, prompt,
            )
            await self.turn("换 2", self.bot.last_id)
        self.assertEqual(self.created(), [(word, "yygla")], self.replay.last_response)
        self.assertIn("草稿内容：羽衣甘蓝 yygla", self.replay.last_response)
        self.assertNotIn("未选择：", self.replay.last_response)
        self.assertNotIn("这些候选本次未添加", self.replay.last_response)

    async def test_multi_word_partial_selection_keeps_exact_remainder_line(self):
        await self.seed()
        await self.turn("马哲 2 5", self.original_id)
        self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea")])
        lines = [line for line in self.replay.last_response.splitlines() if line.startswith("未选择：")]
        self.assertEqual(lines, [
            "未选择：" + "、".join(f"「{word}」" for word in b19.REST)
            + "；这些词本次未添加。",
        ])
        self.assertEqual(self.replay.store.get_record(self.replay.author).state.args["_query_words"],
                         list(b19.REST))


if __name__ == "__main__":
    unittest.main()
