"""S55 exception reporting at the real chat stage boundary."""

import asyncio
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.utils.observability import (
    begin_turn_metrics, current_turn_id, end_turn_metrics, set_turn_flow,
    turn_failure_reply,
)

chat = harness.openai_chat_module


class FailureBoundaryTests(unittest.TestCase):
    def test_empty_response_logs_traceback_without_retry(self):
        async def run():
            logger = logging.getLogger("s55.empty")
            reply = AsyncMock()
            with patch.object(chat, "logger", logger), patch.object(chat, "_finish_ai_chat_matcher", reply), self.assertLogs(logger, level="ERROR") as logs:
                await chat._stage_reject_empty_response(SimpleNamespace(response=""))
            self.assertIn("Traceback (most recent call last)", "\n".join(logs.output))
            self.assertIn("response contract returned empty content", "\n".join(logs.output))
            self.assertNotIn("重试", reply.await_args.args[0])
        asyncio.run(run())

    def test_only_transient_failure_invites_retry(self):
        logger = logging.getLogger("s55.classification")
        for error, transient in ((TimeoutError("timeout"), True), (ValueError("invalid code"), False)):
            with self.assertLogs(logger, level="ERROR"):
                reply = turn_failure_reply(logger, error, stage="fixture", step="核验编码")
            self.assertEqual("重试" in reply, transient)
        for status, transient in ((503, True), (400, False)):
            error = RuntimeError("upstream response")
            error.status_code = status
            with self.assertLogs(logger, level="ERROR"):
                reply = turn_failure_reply(logger, error, stage="fixture", step="核验编码")
            self.assertEqual("重试" in reply, transient)

    def test_discovery_exception_logs_traceback_and_names_failed_step(self):
        async def run():
            token = begin_turn_metrics("qq", "group")
            turn_id = current_turn_id()
            set_turn_flow("word-discovery")
            logger = logging.getLogger("s55.failure")
            reply = AsyncMock()
            try:
                with patch.object(chat, "STAGES", (chat._stage_handle_simple_word_query,)), patch.object(
                    chat, "get_history", return_value=[],
                ), patch.object(
                    chat, "_try_handle_explicit_reading_disambiguation", AsyncMock(return_value=None),
                ), patch.object(
                    chat, "_try_handle_compound_shift_modified_add_command", AsyncMock(return_value=None),
                ), patch.object(
                    chat, "_try_handle_shift_modified_add_command", AsyncMock(return_value=None),
                ), patch.object(
                    chat, "_try_handle_complete_add_command", AsyncMock(return_value=None),
                ), patch.object(
                    chat, "_try_recover_reviewed_add_from_history", AsyncMock(return_value=None),
                ), patch.object(
                    chat, "_try_handle_simple_single_word_query", AsyncMock(side_effect=ValueError("invalid single candidate qx")),
                ), patch.object(chat, "logger", logger), patch.object(
                    chat, "_finish_ai_chat_matcher", reply,
                ), self.assertLogs(logger, level="ERROR") as logs:
                    await chat._handle_ai_chat_serialized(SimpleNamespace(), SimpleNamespace(), "qq", "s55")
                text = "\n".join(logs.output)
                for expected in ("Traceback (most recent call last)", "ValueError", turn_id, "flow=word-discovery", "stage=_stage_handle_simple_word_query"):
                    self.assertIn(expected, text)
                self.assertEqual(reply.await_count, 1)
                user_line = reply.await_args.args[0]
                self.assertIn("审词", user_line)
                self.assertNotIn("请重试", user_line)
                self.assertNotIn("处理请求失败", user_line)
            finally:
                end_turn_metrics(token)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
