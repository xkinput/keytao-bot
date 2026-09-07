"""Explicit user codes remain bound to one reviewed reading and actor ticket."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness

chat = harness.openai_chat_module
commands = harness.chat_commands_module
routing = chat._chat_routing


def reviewed_single():
    return harness.PendingAddWord(
        word="鎗", recommended_code="qx", candidates=[("qx", True)],
        server_candidates=[("qx", True)], occupied_words={"qx": ["强"]},
        server_occupied_words={"qx": ["强"]},
        pronunciation_codes={"qx": "qiāng"}, phrase_type="Single",
        needs_manual_review=True, manual_review_reason="异体字需管理员审核",
    )


class ExplicitCodeTests(unittest.TestCase):
    def test_valid_explicit_forms_bind_to_live_review(self):
        state = reviewed_single()
        for message in ("加入编码qxioio", "加入 qxioio", "添加 鎗 qxioio", "用 qxioio"):
            with self.subTest(message=message):
                self.assertTrue(routing.message_authorizes_live_pending_mutation(message, state))
                self.assertIsNone(routing._pending_assent_rejection_response(state, message))

    def test_invalid_explicit_code_names_the_failure(self):
        for code, reason in (("abioio", "音码"), ("qxioioo", "6"), ("qx12", "小写字母")):
            with self.subTest(code=code):
                response = routing._pending_assent_rejection_response(reviewed_single(), "加入编码" + code)
                self.assertIsNotNone(response)
                self.assertIn(reason, response)

    def test_wrong_word_and_reported_commands_never_bind(self):
        for message in ("添加 枪 qxioio", "他说加入编码qxioio", "不要加入编码qxioio", "加入编码qxioio？", "加入编码qxioio 删除强"):
            self.assertFalse(routing.message_authorizes_live_pending_mutation(message, reviewed_single()), message)

    def test_numeric_explicit_codes_are_not_ordinal_selectors(self):
        from keytao_bot.utils.explicit_code import parse_explicit_code_request
        for message in ("加入编码12", "加入 编码 12", "添加 鎗 12"):
            with self.subTest(message=message):
                reply = routing._pending_assent_rejection_response(reviewed_single(), message)
                self.assertIn("只能包含小写字母", reply)
                self.assertFalse(routing.message_authorizes_live_pending_mutation(message, reviewed_single()))
        for message in ("加入 2", "用 2"):
            self.assertIsNone(parse_explicit_code_request(message, "鎗"))

    def test_explicit_code_executes_sealed_without_shifting(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s56-explicit")
            chat.conversation_state_store.set(key, reviewed_single())
            execute = AsyncMock(return_value="done")
            shift = AsyncMock(side_effect=AssertionError("unexpected shift"))
            with patch.object(chat, "call_tool_function", AsyncMock(return_value=json.dumps({"success": True, "phrases": []}))), patch.object(
                commands, "_execute_add_to_draft", execute,
            ), patch.object(commands, "_execute_shift_to_code", shift):
                reply = await commands.handle_pending_message_core("加入编码qxioio", "qq", "s56-explicit", key, allow_intent_model=False)
            self.assertIn("形码 ioio 未能核验，需管理员复核", reply)
            self.assertTrue(reply.endswith("done"))
            self.assertEqual(execute.await_count, 1)
            self.assertEqual(execute.call_args.args[0:2], ("鎗", "qxioio"))
            self.assertIn("形码 ioio 未能核验，需管理员复核", execute.call_args.args[6])
            self.assertIs(execute.call_args.args[7], True)
            self.assertEqual(execute.call_args.kwargs["reviewed_pinyin"], "qiāng")
            self.assertIn("qxioio", execute.call_args.kwargs["reviewed_candidate_codes"])
        asyncio.run(run())

    def test_alternate_phonetic_base_comes_from_reviewed_inventory(self):
        from keytao_bot.utils.explicit_code import validate_explicit_code
        state = harness.PendingAddWord(
            word="哲思", recommended_code="fesk", candidates=[("fesk", False)],
            server_candidates=[("fesk", False)], pronunciation_codes={"fesk": "zhé sī"},
        )
        self.assertTrue(validate_explicit_code(state, "feskio").valid)
        self.assertFalse(validate_explicit_code(state, "qeskio").valid)

    def test_bare_assent_preserves_protected_record_and_named_eviction_executes(self):
        async def run():
            key = harness.ConversationAddress.private("qq", "s56-protected")
            for text in ("加入", "加入并提交", "好"):
                state = reviewed_single()
                state.recommended_code = ""
                chat.conversation_state_store.set(key, state)
                shift = AsyncMock(side_effect=AssertionError("unexpected shift"))
                with patch.object(commands, "_execute_shift_to_code", shift):
                    reply = await commands.handle_pending_message_core(text, "qq", "s56-protected", key, allow_intent_model=False)
                self.assertIn("本次未写入", reply)
                self.assertIn("强", reply)
                self.assertIn("完整编码", reply)
                record = chat.conversation_state_store.get_record(key)
                self.assertIsNotNone(record)
                self.assertFalse(record.execution_id)
                self.assertTrue(routing.message_authorizes_live_pending_mutation("加入，顶替 强", record.state))
                self.assertTrue(chat._advertised_reply_matches_live_record(reply, record), reply)
                self.assertEqual(chat._enforce_advertised_reply_contract(reply, key), reply)
            shift = AsyncMock(return_value="shifted")
            with patch.object(commands, "_execute_shift_to_code", shift):
                reply = await commands.handle_pending_message_core("加入，顶替 强", "qq", "s56-protected", key, allow_intent_model=False)
            self.assertEqual(reply, "shifted")
            self.assertEqual(shift.await_count, 1)
            self.assertEqual(shift.call_args.args[:2], ("鎗", "qx"))
            self.assertEqual(shift.call_args.kwargs["target_item"]["type"], "Single")
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
