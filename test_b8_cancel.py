"""Offline B8 regressions for closed answers after a mixed multi-word add."""

import copy
import json
import socket
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import test_state_machine as harness
from eval.routing.replay_prod_cases import Replay
from keytao_bot.harness.operation_request import understand_message
from keytao_bot.harness.state import PendingToolConfirm
from test_semantic_parser import FakeClient, raw_operation


# The combined suite imports memory-safety after the established no-runtime harness.
harness._fake_nonebot.init = lambda: None
harness._fake_log.logger.exception = harness._fake_log.logger.error


PRESENT = (
    ("喜上眉梢", "xsms"), ("欢天喜地", "htxd"), ("眉开眼笑", "mkyx"),
    ("喜气洋洋", "xqyy"), ("眉飞色舞", "mfsw"), ("愁眉苦脸", "jmkl"),
    ("唉声叹气", "astq"),
)
WORDS = tuple(word for word, _code in PRESENT) + ("哀声叹气",)


class B8CancelTests(unittest.IsolatedAsyncioTestCase):
    def replay(self):
        return Replay({
            "sender": "offline-abang",
            "dictionary": [
                {"word": word, "code": code, "type": "Phrase"}
                for word, code in PRESENT + (("取消", "qa"), ("取消", "qlxc"))
            ],
            "candidate_facts": {
                "哀声叹气": {"recommended": "astqa", "codes": ["astq", "astqa"],
                         "pinyin": "ai sheng tan qi"},
            },
        }, harness, FakeClient)

    async def offer(self, replay):
        old_ticket = PendingToolConfirm("keytao_submit_batch", {"batch_id": "offline-batch"})
        self.assertTrue(replay.store.set(replay.author, old_ticket))
        replay.history[:] = [{"role": "assistant", "content": "回复「确认」提交，或「取消」。"}]
        message = "加词：" + "，".join(WORDS)
        await replay.turn({"message": message, "to_me": True}, raw_operation(
            op="add", operands=list(WORDS), codes=[], frame="direct", refers_to_pending=False,
        ))
        self.assertIn("原待确认提交已取消，已改为以下新操作。", replay.last_response)
        message_token = replay.chat._current_turn_message.set(message)
        try:
            # Delivery can redraw the prompt after the stage has remembered it.
            replay.last_response = replay.chat._prepare_user_facing_reply(replay.last_response, replay.memory)
        finally:
            replay.chat._current_turn_message.reset(message_token)
        record = replay.store.get_record(replay.author)
        self.assertIsInstance(record.state, PendingToolConfirm)
        self.assertEqual(record.state.function_name, "keytao_batch_add_to_draft")
        self.assertEqual(record.state.args["_query_words"], list(WORDS))
        self.assertEqual([item["word"] for item in record.state.args["items"]], ["哀声叹气"])
        self.assertIn("加入", replay.last_response)
        self.assertIn("加入并提交", replay.last_response)
        self.assertIn("换 2", replay.last_response)
        self.assertTrue(replay.chat._advertised_reply_matches_live_record(replay.last_response, record))
        self.assertEqual(understand_message("取消", record.state).op, "cancel")
        return record

    async def test_cancel_generated_incident_ticket_before_query_routing(self):
        replay = self.replay()
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
            await self.offer(replay)
            replay.calls.clear()
            replay.trace.clear()
            outcome = await replay.turn({"message": "取消", "to_me": True}, raw_operation(
                op="cancel", operands=["哀声叹气"], codes=[], frame="direct", refers_to_pending=True,
            ))
            self.assertEqual(replay.last_response, "已取消。")
            self.assertIsNone(replay.store.get_record(replay.author))
            self.assertEqual(replay.calls, [])
            self.assertEqual(replay.writes, [])
            self.assertEqual(outcome["op"], "cancel")
            self.assertEqual(replay.trace, ["_stage_understand_operation_early"])
            replay.client.create.assert_not_awaited()

    async def test_cancel_with_stale_quote_stays_out_of_word_query_and_preserves_binding(self):
        await self.assert_unbound_reply("取消", model_op="cancel")

    async def test_other_recognized_closed_answers_cannot_be_reinterpreted_as_queries(self):
        for message in ("取消。", "算了", "不加了", "取消本次添加", "都加入", "加入并提交", "哀声叹气 2"):
            with self.subTest(message=message):
                await self.assert_unbound_reply(message)

    async def test_digestless_legacy_ticket_keeps_its_binding_without_query_fallback(self):
        for message in ("取消", "算了"):
            with self.subTest(message=message):
                await self.assert_unbound_reply(message, digestless=True)

    async def assert_unbound_reply(self, message, *, model_op="query", digestless=False):
        replay = self.replay()
        if not any(row["word"] == message for row in replay.entries):
            replay.entries.append({"word": message, "code": "abcd", "type": "Phrase"})
        original_tool = replay.tool

        async def tool(name, args, *positional, **kwargs):
            if name == "keytao_encode":
                replay.calls.append((name, copy.deepcopy(args)))
                return json.dumps({"success": False, "codes": []})
            return await original_tool(name, args, *positional, **kwargs)

        replay.tool = tool
        with ExitStack() as stack:
            for context in replay.patches():
                stack.enter_context(context)
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Offline only")))
            record = await self.offer(replay)
            if digestless:
                self.assertTrue(replay.store.bind_origin_prompt_digest(replay.author, ""))
            original = copy.deepcopy(record)
            raw_prompt = replay.history[-1]["content"]
            self.assertNotEqual(replay.chat._prompt_capability_digest(raw_prompt), record.origin_prompt_digest)
            replay.client.create.return_value.choices[0].message.content = json.dumps({
                "intent": "word_lookup", "words": [message], "confidence": 0.99,
            })
            stack.enter_context(patch.object(replay.chat, "OPENAI_API_KEY", "offline"))
            replay.calls.clear()
            replay.trace.clear()
            quote = "Another unrelated bot message" if digestless else raw_prompt
            await replay.turn({"message": message, "to_me": True, "quote_target": quote}, raw_operation(
                op=model_op, operands=["哀声叹气"] if model_op == "cancel" else [message],
                codes=[], frame="direct" if model_op == "cancel" else "none",
                refers_to_pending=model_op == "cancel",
            ))
            self.assertEqual(replay.last_response, "这条回复未对应当前待确认操作；请引用当前候选消息后再回复。")
            self.assertIs(replay.store.get_record(replay.author), record)
            self.assertEqual(record, original)
            self.assertEqual(replay.calls, [])
            self.assertEqual(replay.writes, [])
            replay.client.create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
