"""S70 fake-model regressions for conversational replies at delivery."""

import json
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
import test_s63_general_reply as general
import test_s63_fresh_selection as fresh
from keytao_bot.utils.conversational_reply import enforce_conversational_reply


chat = harness.openai_chat_module
REAL_CORE = chat.get_ai_response_core
NARRATION = (
    "用户发的这条消息只是「早」，是在打招呼，没有任何词库操作请求，"
    "也没有对应的可执行命令。\n不需要补充信息，等有具体需求再说即可。"
)


class SmallTalkTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        fresh.FreshSelectionTests.setUp(self)
        self.history = []

    runtime = fresh.FreshSelectionTests.runtime
    async def tool(self, name, args, platform, user_id, **kwargs):
        if name == "keytao_lookup_by_words_batch":
            self.calls.append((name, args))
            return json.dumps({"success": True, "results": []})
        return await fresh.FreshSelectionTests.tool(self, name, args, platform, user_id, **kwargs)

    async def delivered(self, message, raw, *, inject_boundary=False, history=None, has_operation_intent=False):
        async def fake_model(text, *_args, **_kwargs):
            return await general.GeneralReplyTests().reply(text, raw)

        async def fake_intent(text, *_args):
            return chat._chat_routing.parse_draft_view_command(text) or harness.MessageCommandIntent(
                has_operation_intent=has_operation_intent,
            )

        token = chat._current_turn_message.set(message)
        conversation_token = chat._current_conversational_turn.set(None)
        try:
            with self.runtime(), patch.object(chat, "get_ai_response_core", side_effect=fake_model), patch.object(
                chat, "_classify_message_command_intent", side_effect=fake_intent,
            ), patch.object(
                chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(False)),
            ), patch.object(chat, "get_history", return_value=history or []):
                cache, classifier = chat.command_intent_memoizer(message, message)
                ctx = chat.TurnContext(
                    bot=object(), event=object(), platform="qq", user_id=self.key.actor_id,
                    message_text=message, normalized_message_text=message, conv_key=self.key,
                    space_key=self.key.space_key, memory_context=self.memory, history=history or [],
                    reply_reference=harness.ReplyReferenceInfo(), command_intent_cache=cache,
                    command_intent_for=classifier,
                )
                await chat._stage_normalize_message_text(ctx)
                start = chat.STAGES.index(chat._stage_resolve_current_pending_scope)
                end = chat.STAGES.index(chat._stage_enforce_advertised_reply_contract)
                for stage in chat.STAGES[start:end + 1]:
                    if await stage(ctx):
                        break
                if inject_boundary:
                    ctx.response = raw
                response = chat._prepare_user_facing_reply(ctx.response, self.memory)
                self.assertEqual(response, chat._prepare_user_facing_reply(response, self.memory))
                return response
        finally:
            chat._current_conversational_turn.reset(conversation_token)
            chat._current_turn_message.reset(token)

    def assert_natural(self, reply):
        self.assertTrue(reply)
        self.assertNotIn("\n", reply)
        self.assertLess(len(reply), 80)
        for text in ("用户", "命令", "候选", "草稿", "证据", "可发送", "未写入", "无需", "不需要", "具体需求"):
            self.assertNotIn(text, reply)

    async def test_incident_fake_model_reply_is_redrawn_at_delivery(self):
        reply = await self.delivered("喵喵 早", NARRATION)
        self.assert_natural(reply)
        self.assertIn("早", reply)
        self.assertEqual(self.calls, [])

    async def test_all_greetings_and_unrelated_chat_use_the_same_no_intent_path(self):
        for message in ("早", "早上好", "晚安", "在吗", "谢谢", "辛苦了", "哈哈", "今天的云像棉花糖"):
            with self.subTest(message=message):
                self.assert_natural(await self.delivered(message, NARRATION))
        self.assertEqual(self.calls, [])

    async def test_unlisted_chitchat_keeps_its_relevant_natural_answer(self):
        reply = await self.delivered("今天的云像棉花糖", "软乎乎的，看得本喵都想咬一口～")
        self.assertEqual(reply, "软乎乎的，看得本喵都想咬一口～")

    async def test_synthetic_narration_is_rejected_at_the_final_boundary(self):
        for raw in (
            NARRATION, "用户想打个招呼。", "用户向我问好，回复问候即可。", "没有找到可执行命令。", "这条消息是问候。",
            "我决定不调用工具。", "不需要补充信息，等有具体需求再说即可。",
            "早呀！\n可发送「查看草稿」。", "早呀！\n1. 查询\n2. 加词",
        ):
            with self.subTest(raw=raw):
                self.assert_natural(await self.delivered("早", raw, inject_boundary=True))

    async def test_repeated_greetings_change_wording_without_another_model_call(self):
        for raw in (NARRATION, "早呀，喵～"):
            first = await self.delivered("早", raw)
            second = await self.delivered("早", raw, history=[
                {"role": "user", "content": "早"}, {"role": "assistant", "content": first},
            ])
            self.assert_natural(second)
            self.assertNotEqual(first, second)

    async def test_real_operation_and_read_only_query_are_unaffected(self):
        reply = await self.delivered("查看草稿", "unused")
        self.assertIn("草稿", reply)
        self.assertIn("keytao_get_batch_preview", [name for name, _ in self.calls])
        answer = "「你好」的编码是 nkhzia。\n这是查询结果。"
        self.assertEqual(await self.delivered("你好的编码是什么", answer, has_operation_intent=True), answer)

    async def test_incomplete_operation_still_requests_its_target(self):
        answer = "请告诉我要添加的词条。"
        self.assertEqual(await self.delivered("帮我加个词", answer, has_operation_intent=True), answer)

    async def test_function_questions_with_no_shortcut_are_not_claimed_as_chitchat(self):
        for message, answer in (
            ("怎么绑定账号？", "请使用 /bind 绑定账号。"),
            ("nhfw 下有哪些词？", "候选：奈飞。\n这是 fixture 查询结果。"),
        ):
            with self.subTest(message=message):
                self.assertEqual(await self.delivered(message, answer, has_operation_intent=True), answer)

    def test_even_incorrect_conversation_metadata_cannot_claim_explicit_operation(self):
        token = chat._current_conversational_turn.set(None)
        try:
            chat._set_conversational_turn("添加 加亮 jslxa", [], harness.MessageCommandIntent(has_operation_intent=False))
            self.assertIsNone(chat._current_conversational_turn.get())
        finally:
            chat._current_conversational_turn.reset(token)

    async def test_compound_goodnight_does_not_become_a_morning_greeting(self):
        reply = await self.delivered("晚安，明早见", NARRATION)
        self.assertIn("晚安", reply)
        self.assertNotIn("早呀", reply)

    async def test_web_core_and_delivery_share_the_guard_without_chat_stages(self):
        for platform in ("web", "web-anon"):
            client = harness._FakeClient([harness._FakeAIResponse("stop", NARRATION)])
            token = chat._current_turn_message.set("")
            conversation_token = chat._current_conversational_turn.set(None)
            try:
                with self.runtime(), patch.object(chat, "OPENAI_API_KEY", "s70-fake-key"), patch.object(
                    chat, "AsyncOpenAI", return_value=client,
                ), patch.object(chat, "skills_manager", harness._FakeToolSkillsManager()), patch.object(
                    chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent(
                        has_operation_intent=False,
                    )),
                ) as classify:
                    # The runtime fixture normally forbids core calls; use the saved real function.
                    reply = await REAL_CORE("早", platform, "s70-web", history=[])
                    self.assert_natural(chat._prepare_user_facing_reply(reply, None))
                    self.assert_natural(chat._prepare_user_facing_reply(NARRATION, None))
                    classify.assert_awaited_once_with("早")
                    self.assertEqual(len(client.completions.calls), 1)
                    self.assertFalse(client.completions.calls[0].get("tools"))
            finally:
                chat._current_conversational_turn.reset(conversation_token)
                chat._current_turn_message.reset(token)

    async def test_semantic_router_carries_absent_intent_without_a_greeting_table(self):
        routing = chat._chat_routing
        for message, has_operation in (("早", False), ("今天的云像棉花糖", False), ("怎么绑定账号？", True)):
            client = harness._FakeClient([harness._FakeAIResponse("stop", json.dumps({
                "intent": "none", "confidence": 0.99, "has_operation_intent": has_operation,
            }))])
            with patch.object(routing, "OPENAI_API_KEY", "s70-fake-key"), patch.object(
                routing, "AsyncOpenAI", return_value=client,
            ):
                intent = await routing._classify_message_command_intent(message)
            self.assertIs(intent.has_operation_intent, has_operation)
            self.assertEqual(intent.intent, "none")
            self.assertEqual(len(client.completions.calls), 1)

    def test_operation_metadata_is_independent_and_unknown_is_not_false(self):
        parse = chat._chat_routing._parse_message_command_intent_payload
        for value in (True, False):
            self.assertIs(parse({"intent": "none", "has_operation_intent": value}).has_operation_intent, value)
        for payload in ({}, {"has_operation_intent": "false"}, {"has_operation_intent": 0}):
            self.assertIsNone(parse(payload).has_operation_intent)
        token = chat._current_conversational_turn.set(None)
        try:
            for intent, other in (
                (harness.MessageCommandIntent(), ()),
                (harness.MessageCommandIntent(has_operation_intent=False),
                 (harness.MessageCommandIntent(has_operation_intent=True),)),
            ):
                chat._set_conversational_turn("fixture", [], intent, other_intents=other)
                self.assertIsNone(chat._current_conversational_turn.get())
        finally:
            chat._current_conversational_turn.reset(token)

    async def test_live_unrelated_candidate_does_not_hijack_a_greeting(self):
        state = harness.PendingAddWord(
            word="加亮", recommended_code="jslxa", candidates=[("jslxa", False)],
            server_candidates=[("jslxa", False)],
        )
        self.store.set(self.key, state)
        reply = await self.delivered("谢谢", NARRATION)
        self.assert_natural(reply)
        self.assertIs(self.store.get(self.key), state)
        self.assertEqual(self.calls, [])

    def test_natural_copy_does_not_confuse_product_users_with_narration(self):
        self.assertEqual(enforce_conversational_reply("新界面怎么样", "用户体验挺顺手的喵。"), "用户体验挺顺手的喵。")


if __name__ == "__main__":
    unittest.main()
