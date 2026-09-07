"""Adversarial S56 checks at live routing, typed lookup and discovery seams."""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from test_s56_explicit_code import reviewed_single
from test_s55_single_pipeline import single_review
from test_s54_multiword import review_payload
from keytao_bot.harness.tools import ToolContext
from keytao_bot.utils.explicit_code import ExplicitCodeRequest
from keytao_bot.utils.pending_confirmation import advertised_command_suggestions, advertised_single_word_candidate_codes

chat = harness.openai_chat_module
commands = harness.chat_commands_module
routing = chat._chat_routing


class S56SecurityTests(unittest.TestCase):
    def test_same_turn_retry_using_exact_command_is_not_advertised(self):
        message = "将草稿中「亮面」lxmmov 的权重调整为 101"
        reason = "当前草稿里还没有任何条目，找不到「亮面」（lxmmov），所以无法调整权重。"
        for retry in ("之后再用", "之后再次使用"):
            with self.subTest(retry=retry):
                raw = f"{reason}\n先往草稿里加上这个条目，{retry}「{message}」即可。"
                reply = harness.AgentOrchestrator._finalize_reply(message, raw, {})
                self.assertNotIn(message, reply)
                self.assertIn("没有", reply)
        for reply in (reason, f"收到「{message}」，但草稿为空。", "先补充条目后，再用「查看草稿」核对。", "当前草稿为空；添加条目后再用「查看草稿」核对。"):
            self.assertEqual(reply, harness.AgentOrchestrator._finalize_reply(message, reply, {}))

    def test_binding_meta_redraw_uses_existing_plain_answer(self):
        raw = (
            "会的。所有词库操作（加词、草稿、提交）都以你绑定键道账号为前提"
            "——未绑定的话工具调用会直接失败，所以我会先提示你完成绑定。"
        )
        for message, expected in (
            ("你是否会先确认对方有没有绑定账号？", "会的：写入前会校验绑定；未绑定会给出绑定引导。"),
            ("解释候选编码", "这次没有生成可发送的回复；本次未写入。"),
        ):
            with self.subTest(message=message):
                token = chat._current_turn_message.set(message)
                try:
                    self.assertEqual(expected, chat._prepare_user_facing_reply(raw, None))
                finally:
                    chat._current_turn_message.reset(token)

    def test_advertised_shift_cancel_closes_only_the_ticket_without_model_or_write(self):
        async def run():
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s56-shift-cancel")
            state = harness.PendingToolConfirm(
                function_name="keytao_shift_phrase_code",
                args={"word": "发布会", "target_code": "fbh"},
                confirmation_source="local_preview",
            )
            store.set(key, state)
            with patch.object(chat, "conversation_state_store", store), patch.object(
                chat, "call_tool_function", AsyncMock(side_effect=AssertionError("cancel wrote")),
            ), patch.object(commands, "_classify_message_command_intent", AsyncMock(
                side_effect=AssertionError("cancel used intent model"),
            )):
                response = await commands.handle_pending_message_core(
                    "取消", "qq", "s56-shift-cancel", key, allow_intent_model=False,
                )
            self.assertEqual("已取消。", response)
            self.assertIsNone(store.get_record(key))
            for text in ("取消？", "他说：取消", "不要取消", "请解释取消"):
                intent = routing._pending_tool_assent_intent(state, text)
                self.assertFalse(intent and intent.intent == "pending_cancel", text)
        asyncio.run(run())

    def test_default_front_insert_retains_manual_seal_after_auto_review(self):
        async def run():
            state = reviewed_single()
            state.word = "发布会"
            state.phrase_type = "Phrase"
            state.needs_manual_review = False
            state.candidates = [("fbh", True), ("fbhu", False)]
            state.server_candidates = list(state.candidates)
            state.recommended_code = "fbh"
            state.occupied_words = {"fbh": ["重病号"]}
            state.server_occupied_words = dict(state.occupied_words)
            state.pronunciation_codes = {code: "fa bu hui" for code, _ in state.candidates}
            state.code_remarks = {"fbh": "喵喵审词：读音 fa bu hui；来源 汉典"}
            state.server_ordering_assessments = [{
                "newWord": "发布会", "occupantWord": "重病号", "occupantCode": "fbh",
                "freeCode": "fbhu", "newCode": "fbh", "recommendedCode": "fbh",
                "verdict": "front_more_common",
            }]
            for message in ("加入", "加入并提交"):
                with self.subTest(message=message):
                    shift = AsyncMock(return_value="sealed shift")
                    with patch.object(commands, "_execute_shift_to_code", shift):
                        reply = await commands._handle_pending_add_word(
                            state, message, "qq", "s56-front-seal", [],
                            command_intent=harness.MessageCommandIntent(
                                intent="pending_add_and_submit" if "提交" in message else "pending_confirm",
                                confidence=1.0, submit_after="提交" in message,
                            ),
                        )
                    self.assertEqual("sealed shift", reply)
                    self.assertIs(shift.call_args.kwargs["target_item"]["needsManualReview"], True)
                    self.assertEqual("Phrase", shift.call_args.kwargs["target_item"]["type"])
                    self.assertEqual(state.code_remarks["fbh"], shift.call_args.kwargs["target_item"]["remark"])
                    self.assertEqual("fa bu hui", shift.call_args.kwargs["reviewed_pinyin"])
        asyncio.run(run())

    def test_complete_explicit_code_uses_early_live_pending_replacement_route(self):
        async def run():
            state = reviewed_single()
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s56-complete-stage")
            store.set(key, state)
            execute = AsyncMock(return_value="early pending route")
            with patch.object(chat, "conversation_state_store", store), patch.object(commands, "handle_pending_message_core", execute):
                response = await commands._try_handle_explicit_pending_replacement(state, "添加 鎗 qxioio", "qq", "s56-complete-stage", key)
            self.assertEqual("early pending route", response)
            self.assertEqual(1, execute.await_count)
            self.assertFalse(execute.call_args.kwargs["allow_intent_model"])
        asyncio.run(run())

    def test_unknown_shape_seal_reaches_real_sink_validator_and_cannot_be_forged(self):
        async def run():
            with patch.object(chat, "call_tool_function", AsyncMock(return_value=json.dumps({"success": True, "phrases": []}))):
                derived, error = await commands._bind_explicit_pending_code(reviewed_single(), ExplicitCodeRequest("qxioio"), "qq", "s56-seal")
            self.assertFalse(error)
            args = commands._create_phrase_args(derived, "qxioio")
            pinyin, codes = commands._pending_reviewed_reading(derived, "qxioio")
            capability = commands._reviewed_create_capability("鎗", "qxioio", pinyin, codes)
            trusted = commands.tool_executor.canonicalize_arguments(
                "keytao_create_phrase", args,
                ToolContext("qq", "s56-seal", trusted_reviewed_items_by_key=capability),
            )
            item = {"action": "Create", "word": "鎗", "code": "qxioio", "type": "Single", "remark": trusted.get("remark"), "needs_manual_review": trusted.get("needs_manual_review")}
            with patch.object(harness._draft_tools, "_fetch_encode_candidates", AsyncMock(side_effect=AssertionError("known reviewed capability re-encoded"))):
                validation = await harness._draft_tools._validate_draft_item_code(item, reviewed_pinyin=trusted.get("_reviewed_pinyin", ""), reviewed_candidate_codes=trusted.get("_reviewed_candidate_codes"))
            self.assertTrue(validation["success"])
            harness._draft_tools._stamp_item_review_flag(item, validation)
            self.assertTrue(item["needsManualReview"])
            self.assertIn("形码 ioio 未能核验", item["remark"])
            forged = commands.tool_executor.canonicalize_arguments("keytao_create_phrase", args, ToolContext("qq", "s56-foreign"))
            self.assertNotIn("_reviewed_pinyin", forged)
            self.assertNotIn("_reviewed_candidate_codes", forged)
            with patch.object(harness._draft_tools, "_fetch_encode_candidates", AsyncMock(return_value={"success": True, "candidateCodes": ["qx"]})):
                rejected = await harness._draft_tools._validate_draft_item_code(item, reviewed_pinyin=forged.get("_reviewed_pinyin", ""), reviewed_candidate_codes=forged.get("_reviewed_candidate_codes"))
            self.assertFalse(rejected["success"])
        asyncio.run(run())

    def test_invalid_or_framed_input_never_authorizes_extension(self):
        source = reviewed_single()
        for message in (
            "加入编码abioio", "加入编码qxioioo", "加入编码qx12", "加入编码QXioio",
            "加入编码qxioio 删除 强", "加入编码qxioio，添加 枪 qxi", "加入编码qxioio吗",
            "他说：加入编码qxioio", "请解释‘加入编码qxioio’", "不要加入编码qxioio",
            "加入编码qxioio？", "如果需要就加入编码qxioio", "可以加入编码qxioio吗？",
        ):
            with self.subTest(message=message):
                self.assertFalse(routing.message_authorizes_live_pending_mutation(message, source))
        source.server_candidates = []
        self.assertFalse(routing.message_authorizes_live_pending_mutation("加入编码qxioio", source))

    def test_same_group_other_actor_has_no_record_or_lookup(self):
        async def run():
            store = harness.MemoryConversationStateStore()
            owner = harness.ConversationAddress.group("qq", "s56-group", "s56-owner")
            other = harness.ConversationAddress.group("qq", "s56-group", "s56-other")
            store.set(owner, reviewed_single(), space_key=owner.space_key)
            with patch.object(chat, "conversation_state_store", store), patch.object(
                chat, "call_tool_function", AsyncMock(side_effect=AssertionError("foreign actor lookup")),
            ):
                reply = await commands.handle_pending_message_core(
                    "加入编码qxioio", "qq", "s56-other", other,
                    space_key=other.space_key, allow_intent_model=False,
                )
            self.assertIsNone(reply)
            self.assertIsNotNone(store.get_record(owner))
        asyncio.run(run())

    def test_exact_six_code_duplicate_does_not_write(self):
        async def run():
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s56-duplicate")
            store.set(key, reviewed_single())
            with patch.object(chat, "conversation_state_store", store), patch.object(chat, "call_tool_function", AsyncMock(
                return_value=json.dumps({"success": True, "phrases": [{"word": "鎗", "code": "qxioio", "type": "Single", "weight": 10}]}),
            )), patch.object(commands, "_execute_add_to_draft", AsyncMock(side_effect=AssertionError("duplicate write"))):
                reply = await commands.handle_pending_message_core("加入编码qxioio", "qq", "s56-duplicate", key, allow_intent_model=False)
            self.assertIn("不能重复添加", reply)
            self.assertEqual([("qx", True)], store.get_record(key).state.server_candidates)
        asyncio.run(run())

    def test_empty_multi_selection_does_not_inherit_protected_recommendation(self):
        async def run():
            state = reviewed_single()
            state.candidates = [("qx", True), ("qxi", False), ("qxio", False)]
            state.server_candidates = list(state.candidates)
            state.pronunciation_codes = {code: "qiāng" for code, _ in state.candidates}
            key = harness.ConversationAddress.private("qq", "s56-multiple-empty")
            store = harness.MemoryConversationStateStore()
            store.set(key, state)
            execute = AsyncMock(return_value="empty choices written")
            with patch.object(chat, "conversation_state_store", store), patch.object(
                commands, "_execute_add_multiple_codes_to_draft", execute,
            ):
                reply = await commands.handle_pending_message_core("添加2、3", "qq", "s56-multiple-empty", key, allow_intent_model=False)
            self.assertEqual("empty choices written", reply)
            self.assertEqual(["qxi", "qxio"], execute.call_args.args[1])
        asyncio.run(run())

    def test_multiword_query_keeps_blocked_item_review_and_other_safe_candidate(self):
        async def run(message="鎗 大端", all_blocked=False):
            store = harness.MemoryConversationStateStore()
            key = harness.ConversationAddress.private("qq", "s56-multiword")
            protected = single_review()
            protected["recommendedCode"] = ""
            pronunciation = protected["pronunciations"][0]
            pronunciation["recommendedCode"] = ""
            pronunciation["candidateStatuses"] = [{"code": "qx", "occupied": True, "words": ["强"], "label": "已有「强」"}]
            protected["candidateOrderingAssessments"] = [{"newWord": "鎗", "occupantWord": "强", "occupantCode": "qx", "freeCode": "", "newCode": "", "verdict": "behind_more_common"}]

            async def tool(name, args, *_args, **_kwargs):
                if name == "keytao_lookup_by_word":
                    return json.dumps({"success": True, "phrases": []})
                if name == "keytao_pending_items_by_words":
                    return json.dumps({"success": True, "complete": True, "items": []})
                if name == "keytao_prepare_reviewed_add":
                    if len(args["word"]) == 1:
                        payload = json.loads(json.dumps(protected))
                        payload["word"] = args["word"]
                        payload["candidateOrderingAssessments"][0]["newWord"] = args["word"]
                        return json.dumps(payload)
                    return json.dumps(review_payload(args["word"]))
                raise AssertionError(name)

            with patch.object(chat, "conversation_state_store", store), patch.object(
                chat, "call_tool_function", side_effect=tool,
            ), patch.object(commands.user_resolver, "resolve_actor_binding", AsyncMock(return_value=True)):
                reply = await commands._try_handle_simple_single_word_query(message, "qq", "s56-multiword", key)
            self.assertIn("审词：读音 qiāng", reply)
            if all_blocked:
                self.assertIn("「槍」本次未选中", reply)
                self.assertIsNone(store.get_record(key))
                self.assertEqual((), advertised_single_word_candidate_codes(reply))
                self.assertEqual((), advertised_command_suggestions(reply))
                with patch.object(chat, "conversation_state_store", store):
                    self.assertEqual(reply, chat._enforce_advertised_reply_contract(reply, key))
                return
            self.assertIn("大端", reply)
            self.assertIn("dsdtvo", reply)
            self.assertNotIn("展示校验未通过", reply)
            self.assertIn("「鎗」本次未选中", reply)
            self.assertIn("qx — 已有「强」", reply)
            self.assertNotIn("1. qx", reply)
            self.assertNotIn("添加 鎗", reply)
            self.assertNotIn("加入编码 qx+形码", reply)
            record = store.get_record(key)
            self.assertEqual(["大端"], [item["word"] for item in record.state.args["items"]])
            with patch.object(chat, "conversation_state_store", store):
                self.assertTrue(chat._advertised_reply_matches_live_record(reply, record))
                self.assertEqual(reply, chat._enforce_advertised_reply_contract(reply, key))
        asyncio.run(run())
        asyncio.run(run("鎗 槍", all_blocked=True))


if __name__ == "__main__":
    unittest.main()
