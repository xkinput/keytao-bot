"""S70b production routing under an always-no-intent fake classifier."""

import json
import unittest
from unittest.mock import AsyncMock, patch

import test_s70_smalltalk as smalltalk


chat = smalltalk.chat
harness = smalltalk.harness


class QueryPrecedenceTests(unittest.IsolatedAsyncioTestCase):
    setUp = smalltalk.SmallTalkTests.setUp
    runtime = smalltalk.SmallTalkTests.runtime
    assert_natural = smalltalk.SmallTalkTests.assert_natural

    async def tool(self, name, args, platform, user_id, **kwargs):
        if name == "keytao_lookup_by_word":
            self.calls.append((name, args))
            return json.dumps({"success": True, "phrases": [{
                "word": args["word"], "code": "abcd", "type": "Phrase",
            }] if not getattr(self, "missing_word", False) else []})
        return await smalltalk.SmallTalkTests.tool(self, name, args, platform, user_id, **kwargs)

    async def delivered(self, message, raw=smalltalk.NARRATION):
        client = harness._FakeClient([harness._FakeAIResponse("stop", raw)])
        token = chat._current_turn_message.set(message)
        conversation_token = chat._current_conversational_turn.set(None)
        try:
            with self.runtime(), patch.object(
                chat, "get_ai_response_core", side_effect=smalltalk.REAL_CORE,
            ), patch.object(chat, "OPENAI_API_KEY", "s70b-fixture"), patch.object(
                chat, "AsyncOpenAI", return_value=client,
            ), patch.object(chat, "skills_manager", harness._FakeToolSkillsManager()), patch.object(
                chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent(
                    has_operation_intent=False,
                )),
            ) as classifier, patch.object(
                chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(False)),
            ):
                ctx = chat.TurnContext(
                    bot=object(), event=object(), platform="qq", user_id=self.key.actor_id,
                    message_text=message, normalized_message_text=message, conv_key=self.key,
                    space_key=self.key.space_key, memory_context=self.memory, history=[],
                    reply_reference=harness.ReplyReferenceInfo(),
                )
                await chat._stage_normalize_message_text(ctx)
                ctx.command_intent_cache, ctx.command_intent_for = chat.command_intent_memoizer(
                    message, ctx.normalized_message_text,
                )
                start = chat.STAGES.index(chat._stage_resolve_current_pending_scope)
                end = chat.STAGES.index(chat._stage_enforce_advertised_reply_contract)
                for stage in chat.STAGES[start:end + 1]:
                    if await stage(ctx):
                        break
                response = chat._prepare_user_facing_reply(ctx.response, self.memory)
                self.assertGreater(classifier.await_count, 0)
                return response, client.completions.calls
        finally:
            chat._current_conversational_turn.reset(conversation_token)
            chat._current_turn_message.reset(token)

    async def test_query_matrix_wins_over_no_intent_classifier(self):
        for message, words in (
            ("练枪 得吃", ("练枪", "得吃")),
            ("@喵喵 练枪 得吃", ("练枪", "得吃")),
            ("蛋粉 单份", ("蛋粉", "单份")),
            ("敲不死", ("敲不死",)),
            ("共和国", ("共和国",)),
            ("和平 共处", ("和平", "共处")),
            ("一力降十会（yi2 li4 xiang2 shi2 hui4）", ("一力降十会",)),
        ):
            with self.subTest(message=message):
                self.calls.clear()
                reply, model_calls = await self.delivered(message)
                self.assertEqual([args["word"] for name, args in self.calls
                                  if name == "keytao_lookup_by_word"], list(words))
                self.assertEqual(model_calls, [])
                for word in words:
                    self.assertIn(word, reply)

    async def test_chat_matrix_has_natural_reply_and_no_exposed_tools(self):
        for message in ("早", "晚安", "谢谢", "在吗", "辛苦了", "晚安，明早见", "今天的云像棉花糖", "哈哈哈"):
            with self.subTest(message=message):
                self.calls.clear()
                reply, model_calls = await self.delivered(message)
                self.assert_natural(reply)
                self.assertEqual(self.calls, [])
                self.assertEqual(len(model_calls), 1)
                self.assertFalse(model_calls[0].get("tools"))

    async def test_separators_and_mixed_greetings_still_query_every_word(self):
        for separator in (" ", "\n", "\t", "，", "、", "；", ";", "和", " 和 "):
            for words in (("练枪", "得吃"), ("谢谢", "练枪")):
                message = separator.join(words)
                with self.subTest(message=message):
                    self.calls.clear()
                    reply, model_calls = await self.delivered(message)
                    self.assertEqual([args["word"] for name, args in self.calls
                                      if name == "keytao_lookup_by_word"], list(words))
                    self.assertEqual(model_calls, [])
                    for word in words:
                        self.assertIn(word, reply)

    async def test_explicit_add_and_explicit_greeting_lookup_are_not_chat(self):
        for message, word in (("加词 加亮", "加亮"), ("查词 谢谢", "谢谢")):
            with self.subTest(message=message):
                self.calls.clear()
                self.missing_word = message.startswith("加词")
                self.pending_items = []
                reply, model_calls = await self.delivered(message)
                self.assertEqual([args["word"] for name, args in self.calls
                                  if name == "keytao_lookup_by_word"], [word])
                self.assertEqual(model_calls, [])
                self.assertIn(word, reply)
                if self.missing_word:
                    self.assertEqual([args["word"] for name, args in self.calls
                                      if name == "keytao_prepare_reviewed_add"], [word])
                    self.assertIsNotNone(self.store.get(self.key))

    async def test_queries_prepare_missing_words_with_no_intent_classifier(self):
        self.missing_word = True
        self.pending_items = []

        original_tool = self.tool
        # Keep the fake transport separate from the production review/record path.
        async def wrapped_tool(name, args, platform, user_id, **kwargs):
            if name == "keytao_prepare_reviewed_add":
                self.calls.append((name, args))
                return json.dumps(dict(self.review, word=args["word"]))
            return await original_tool(name, args, platform, user_id, **kwargs)
        self.tool = wrapped_tool
        reply, model_calls = await self.delivered("练枪 得吃")
        self.assertEqual([args["word"] for name, args in self.calls
                          if name == "keytao_prepare_reviewed_add"], ["练枪", "得吃"])
        self.assertEqual(model_calls, [])
        self.assertIn("练枪", reply)
        self.assertIn("得吃", reply)
        self.assertEqual([row["word"] for row in self.store.get(self.key).args["_candidate_scopes"]],
                         ["练枪", "得吃"])

    async def test_prose_and_comparison_clauses_do_not_become_bare_word_queries(self):
        for message in ("他说加入", "这个和电机哪个常用", "帮我加个词"):
            with self.subTest(message=message), patch.object(
                chat, "_classify_simple_word_query_intent", AsyncMock(return_value=harness.SimpleWordQueryIntent(False)),
            ), patch.object(chat, "call_tool_function", AsyncMock(side_effect=AssertionError("Unexpected lookup"))):
                self.assertIsNone(await harness.chat_commands_module._try_handle_simple_single_word_query(
                    message, "qq", self.key.actor_id,
                ))

    async def test_web_core_cannot_redraw_a_query_or_hide_its_tools(self):
        for platform in ("web", "web-anon"):
            for message in ("练枪 得吃", "敲不死", "一力降十会（yi2 li4 xiang2 shi2 hui4）", "谢谢 练枪"):
                with self.subTest(platform=platform, message=message):
                    client = harness._FakeClient([harness._FakeAIResponse("stop", "Fixture query response")])
                    token = chat._current_turn_message.set(message)
                    conversation_token = chat._current_conversational_turn.set(None)
                    try:
                        with self.runtime(), patch.object(chat, "OPENAI_API_KEY", "s70b-fixture"), patch.object(
                            chat, "AsyncOpenAI", return_value=client,
                        ), patch.object(chat, "skills_manager", harness._FakeToolSkillsManager()), patch.object(
                            chat, "_classify_message_command_intent", AsyncMock(return_value=harness.MessageCommandIntent(
                                has_operation_intent=False,
                            )),
                        ):
                            await smalltalk.REAL_CORE(message, platform, "s70b-web", history=[])
                            self.assertIsNone(chat._current_conversational_turn.get())
                            self.assertTrue(client.completions.calls[0].get("tools"))
                            self.assertEqual(chat._prepare_user_facing_reply("Fixture query response", None),
                                             "Fixture query response")
                    finally:
                        chat._current_conversational_turn.reset(conversation_token)
                        chat._current_turn_message.reset(token)

    def test_whole_message_discriminator_never_authorizes_writes(self):
        for message in ("早", "谢谢", "晚安，明早见", "哈哈哈", "今天的云像棉花糖"):
            with self.subTest(message=message):
                self.assertTrue(chat._chat_routing.is_conversational_text(message))
                self.assertFalse(chat.message_authorizes_mutation(message))
        for message in ("谢谢 练枪", "早，查词练枪", "我觉得需要查词", "我感觉编码有问题",
                        "今天的云像棉花糖 练枪", "一力降十会（yi2 li4 xiang2 shi2 hui4）"):
            with self.subTest(message=message):
                self.assertFalse(chat._chat_routing.is_conversational_text(message))


if __name__ == "__main__":
    unittest.main()
