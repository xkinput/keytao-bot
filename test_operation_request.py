"""Offline checks for closed-grammar operation projection and its shadow stage."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from keytao_bot.harness.operation_request import understand_message


class OperationRequestTests(unittest.TestCase):
    def test_supported_operation_mappings(self):
        cases = (
            ("调换 午时 和 务实", "swap", ("午时", "务实"), ()),
            ("删词 五百万 wbw", "delete", ("五百万",), ("wbw",)),
            ("删除 这厮 fesk；添加 这厮 fesko", "recode", ("这厮",), ("fesk", "fesko")),
            ("把 嘢 排到 咽 前面", "reorder", ("嘢", "咽"), ()),
            ("把 奈飞 调到 nhfw", "move", ("奈飞",), ("nhfw",)),
            ("添加 加亮 jslxa", "add", ("加亮",), ("jslxa",)),
            ("添加 加亮 JSLXA", "add", ("加亮",), ("jslxa",)),
            ("查看草稿", "draft_view", (), ()),
            ("提交草稿", "draft_submit", (), ()),
            ("撤回上次提交", "draft_recall", (), ()),
            ("清空草稿", "draft_clear", (), ()),
        )
        for message, op, operands, codes in cases:
            with self.subTest(message=message):
                request = understand_message(message)
                self.assertEqual(request.op, op)
                self.assertEqual(request.operands, operands)
                self.assertEqual(request.codes, codes)
                self.assertEqual(request.frame, "direct")
                self.assertTrue(request.source.startswith("fast_path:"))

    def test_incident_swap_and_non_direct_frames(self):
        request = understand_message("调整调换 午时 和 务实 词条")
        self.assertEqual((request.op, request.operands), ("swap", ("午时", "务实")))
        for message in (
            "午时和务实能调换吗？",
            "不要调换午时和务实",
            "他说调换午时和务实",
        ):
            with self.subTest(message=message):
                request = understand_message(message)
                self.assertEqual((request.op, request.frame, request.source),
                                 ("none", "unknown", "none"))

    def test_complete_write_wins_over_pending_selection(self):
        pending = SimpleNamespace(word="加亮")
        request = understand_message("加亮 jslxa，加入草稿并提交。", pending)
        self.assertEqual((request.op, request.operands, request.codes),
                         ("add", ("加亮",), ("jslxa",)))
        self.assertTrue(request.submit_after)
        self.assertIsNot(request.refers_to_pending, True)
        for message, op in (
            ("调整调换 午时 和 务实 词条", "swap"),
            ("删词 五百万 wbw", "delete"),
            ("删除 这厮 fesk；添加 这厮 fesko", "recode"),
            ("把 奈飞 调到 nhfw", "move"),
        ):
            with self.subTest(message=message):
                request = understand_message(message, pending)
                self.assertEqual(request.op, op)
                self.assertFalse(request.refers_to_pending)

    def test_pending_controls_require_pending_state(self):
        pending = SimpleNamespace(word="奈飞")
        for message, op, submit_after in (
            ("3", "select_candidate", False),
            ("加入并提交", "confirm", True),
            ("取消", "cancel", False),
        ):
            with self.subTest(message=message):
                self.assertEqual(understand_message(message).op, "none")
                request = understand_message(message, pending)
                self.assertEqual(request.op, op)
                self.assertTrue(request.refers_to_pending)
                self.assertEqual(request.submit_after, submit_after)
        for message, codes in (
            ("奈飞 1", ()),
            ("奈飞 nhfw", ("nhfw",)),
        ):
            with self.subTest(message=message):
                request = understand_message(message, pending)
                self.assertEqual(request.op, "select_candidate")
                self.assertEqual(request.operands, ("奈飞",))
                self.assertEqual(request.codes, codes)
                self.assertTrue(request.refers_to_pending)

    def test_single_word_tool_ticket_supplies_omitted_operand(self):
        from keytao_bot.harness.state import PendingToolConfirm

        pending = PendingToolConfirm(
            "keytao_create_phrase", {"word": "奈飞", "code": "nhfw"},
        )
        for message, op in (("3", "select_candidate"), ("加入", "confirm"), ("取消", "cancel")):
            with self.subTest(message=message):
                request = understand_message(message, pending)
                self.assertEqual((request.op, request.operands), (op, ("奈飞",)))
                self.assertTrue(request.refers_to_pending)

    def test_missing_operand_parsers_remain_unmapped(self):
        for message in (
            "把这些编码里的粘替换成黏：nm abcd",
            "草稿只保留大盘鸡，其他删除",
        ):
            with self.subTest(message=message):
                self.assertEqual(understand_message(message).op, "none")


class OperationRequestShadowStageTests(unittest.IsolatedAsyncioTestCase):
    async def test_stage_is_non_terminal_and_survives_parser_failure(self):
        # Import after unittest discovery: this harness replaces runtime modules.
        import test_state_machine as harness

        chat = harness.openai_chat_module
        ctx = chat.TurnContext(
            bot=None,
            event=None,
            platform="qq",
            user_id="operation-request-owner",
            normalized_message_text="调整调换 午时 和 务实 词条",
            response="existing response",
        )
        with patch.object(chat.logger, "info") as info:
            self.assertFalse(await chat._stage_understand_operation(ctx))
        self.assertEqual(ctx.operation_request.op, "swap")
        self.assertEqual(ctx.response, "existing response")
        info.assert_called_once()
        self.assertIn("[operation_request]", info.call_args.args[0])

        with (
            patch("keytao_bot.harness.operation_request.understand_message",
                  side_effect=RuntimeError("parser failure")),
            patch.object(chat.logger, "warning") as warning,
        ):
            self.assertFalse(await chat._stage_understand_operation(ctx))
        self.assertIsNone(ctx.operation_request)
        self.assertEqual(ctx.response, "existing response")
        warning.assert_called_once()

        pending = SimpleNamespace(word="奈飞")
        ctx.current_pending_record = SimpleNamespace(state=pending)
        with (
            patch("keytao_bot.harness.operation_request.understand_message",
                  return_value=understand_message("3", pending)) as understand,
            patch.object(chat.logger, "info"),
        ):
            self.assertFalse(await chat._stage_understand_operation(ctx))
        understand.assert_called_once_with(ctx.normalized_message_text, pending)
        stage_index = chat.STAGES.index(chat._stage_resolve_current_pending_scope)
        self.assertIs(chat.STAGES[stage_index + 1], chat._stage_understand_operation)


if __name__ == "__main__":
    unittest.main()
