"""B21 display contracts through real tools and offline QQ replies."""

import copy
import re
import unittest
from unittest.mock import patch

import test_b19_partial_reply_binding as b19
from keytao_bot.harness.orchestrator import _command_suggestions_match_pending_batch
from keytao_bot.utils.pending_confirmation import advertised_command_suggestions


HINT = "同一词可选多个编码、其余词用推荐：如「{}」"
FOOTER = "可只写其中部分词；选择后即加入草稿，提交需明确回复提交指令。"


class B21DisplayTests(unittest.IsolatedAsyncioTestCase):
    tearDown = b19.B19PartialReplyBindingTests.tearDown
    matcher_send = b19.B19PartialReplyBindingTests.matcher_send
    matcher_finish = b19.B19PartialReplyBindingTests.matcher_finish
    created = b19.B19PartialReplyBindingTests.created
    seed = b19.B19PartialReplyBindingTests.seed
    turn = b19.B19PartialReplyBindingTests.turn

    def setUp(self):
        b19.B19PartialReplyBindingTests.setUp(self)
        tools = {**b19.b18.TOOLS.TOOL_FUNCTIONS, **b19.harness._review_tools.TOOL_FUNCTIONS}
        self.enterContext(patch.object(b19.b18.commands.tool_executor, "_get_tool_function", tools.get))
        self.enterContext(patch.dict(b19.b18.FACTS, {
            "名创": ("mgwm", ["mgwm", "mgwmi"], "ming chuang"),
        }))

    async def test_c1_single_create_keeps_written_word_without_change_line(self):
        await self.seed(("名创",))
        await self.turn("加词 名创mgwm", self.original_id, raw=b19.b18.raw_operation(
            op="add", operands=["名创"], codes=["mgwm"], refers_to_pending=True,
        ))
        self.assertEqual(self.created(), [("名创", "mgwm")], self.replay.last_response)
        self.assertNotIn("已变更：", self.replay.last_response)
        self.assertIn("✅ 已加入草稿", self.replay.last_response)
        self.assertIn("+1 新增", self.replay.last_response)
        self.assertIn("草稿内容：名创 mgwm", self.replay.last_response)

    async def test_c1_multiple_creates_name_only_selected_codes(self):
        await self.seed(("马哲", "小美"))
        await self.turn("马哲 2 5", self.original_id)
        self.assertCountEqual(self.created(), [("马哲", "msqea"), ("马哲", "msfea")])
        self.assertNotIn("已变更：", self.replay.last_response)
        self.assertIn("草稿内容：马哲 msqea、马哲 msfea", self.replay.last_response)
        self.assertIn("未选择：「小美」", self.replay.last_response)

    def test_c1_create_receipts_without_complete_draft_contents_keep_change_line(self):
        data = {"writtenItems": [
            {"id": 1, "word": "名创", "code": "mgwm", "action": "Create"},
            {"id": 2, "word": "马哲", "code": "msqea", "action": "Create"},
        ]}
        change_line = "已变更：名创 → mgwm；马哲 → msqea"
        for contents in (
            "",
            "\n草稿内容：名创 mgwm",
            "\n草稿内容：名创 mgwm、马哲 msqeai",
            "\n草稿内容：名创 msqea、马哲 mgwm",
            "\n草稿内容：名创 mgwm\n草稿内容：马哲 msqea",
        ):
            with self.subTest(contents=contents):
                original = "✅ 已加入草稿\n" + change_line + contents
                actual = b19.chat._chat_render.finalize_draft_receipt(original, data)
                self.assertEqual(actual, original)

    def test_c1_change_delete_and_mixed_receipts_are_byte_identical(self):
        cases = (
            ([{"id": 1, "word": "夜泊", "code": "yebo", "type": "Phrase", "action": "Change",
               "oldWord": "夜泊", "weight": 100, "previousWeight": 101}],
             "夜泊 yebo 权重 101→100", "夜泊"),
            ([{"id": 1, "word": "旧词", "code": "jc", "type": "Phrase", "action": "Delete"}],
             "删除 旧词 @ jc", "旧词"),
            ([{"id": 1, "word": "新词", "code": "xc", "type": "Phrase", "action": "Create"},
              {"id": 2, "word": "旧词", "code": "jc", "type": "Phrase", "action": "Delete"}],
             "新词 → xc；删除 旧词 @ jc", "旧词"),
        )
        for rows, summary, revised_word in cases:
            with self.subTest(summary=summary):
                data = {"batchId": "b21-batch", "writtenItems": rows}
                original = copy.deepcopy(data)
                actual = b19.chat._chat_render.finalize_draft_receipt(
                    "✅ 操作已完成\n+0 新增\n草稿内容：保留原内容", data, platform="qq",
                )
                expected = (f"✅ 操作已完成\n已变更：{summary}\n"
                            f"发布状态：「{revised_word}」的本次修改现为草稿（未发布）。\n"
                            "+0 新增\n草稿内容：保留原内容\n\n"
                            "草稿地址：https://keytao.rea.ink/batch/b21-batch")
                self.assertEqual(actual.encode(), expected.encode())
                self.assertEqual(data, original)

    async def test_c2_exact_footer_example_executes_and_submits(self):
        record = await self.seed(("小美", "马哲", "小帅"))
        prompt = self.bot.messages[self.original_id]["message"]
        command = "小美 1 2，其他用推荐的 加入并提交"
        self.assertEqual(prompt.count(HINT.format(command)), 1)
        self.assertIn(FOOTER + "\n" + HINT.format(command), prompt)
        self.assertIn(command, advertised_command_suggestions(prompt))
        self.assertTrue(_command_suggestions_match_pending_batch(prompt, record.state))
        quoted_example = re.search(r"同一词可选多个编码、其余词用推荐：如「([^」]+)」", prompt).group(1)
        await self.turn(quoted_example, self.original_id)
        self.assertCountEqual(self.created(), [("小美", "xmww"), ("小美", "xmwwi"),
                                              ("马哲", "msqe"), ("小帅", "xkeu")])
        self.assertEqual(self.http.submissions, 1, self.replay.last_response)
        self.assertIsNone(self.replay.store.get_record(self.replay.author))

    async def test_c2_first_qualifying_word_uses_real_recommended_and_next_empty_indices(self):
        self.enterContext(patch.dict(b19.b18.FACTS, {
            "小美": ("xmww", ["xmww"], "xiao mei"),
            "马哲": ("msqea", ["msqe", "msqea", "msqeai", "msfe", "msfea"], "ma zhe"),
        }))
        ticket = b19.b18.reviewed_ticket(("小美", "马哲", "小帅"), reorder=False)
        scope = ticket.args["_candidate_scopes"][1]
        scope["candidates"][2] = ("msqeai", True)
        scope["occupiedWords"] = {"msqeai": ["占位词"]}
        scope["reviewedState"]["serverCandidates"][2] = ("msqeai", True)
        scope["reviewedState"]["serverOccupiedWords"] = scope["occupiedWords"]
        scope["reviewedPrompt"] = b19.b18.render_server_backed_single_word_candidates(
            "马哲", "msqea", scope["candidates"], scope["occupiedWords"], [],
        )
        self.assertTrue(self.replay.store.set(self.replay.author, ticket))
        record = self.replay.store.get_record(self.replay.author)
        prompt = self.replay.chat._render_live_batch_record(record)
        self.assertIn(HINT.format("马哲 2 4，其他用推荐的 加入并提交"), prompt)
        self.assertEqual(prompt.count("同一词可选多个编码、其余词用推荐"), 1)
        self.assertTrue(_command_suggestions_match_pending_batch(prompt, record.state))

    async def test_c2_no_qualifying_word_omits_hint(self):
        self.enterContext(patch.dict(b19.b18.FACTS, {
            "小美": ("xmww", ["xmww"], "xiao mei"),
            "小帅": ("xkeu", ["xkeu"], "xiao shuai"),
        }))
        await self.seed(("小美", "小帅"))
        self.assertNotIn("同一词可选多个编码", self.replay.prompt)
        self.assertIn(FOOTER, self.replay.prompt)

    async def test_c2_single_word_ticket_omits_hint(self):
        await self.seed(("小美",))
        self.assertNotIn("同一词可选多个编码", self.replay.prompt)


if __name__ == "__main__":
    unittest.main()
