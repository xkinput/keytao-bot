"""Offline regression checks ensuring questions are answered instead of entering add-word flows."""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from tests import test_state_machine as harness

from keytao_bot.harness.authorization_grammar import is_interrogative_message
from keytao_bot.harness.state import DraftOperationCoordinator, MemoryConversationStateStore


ChatMemoryContext = harness.ChatMemoryContext


class QuestionNotWordAddTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.routing = harness.chat_routing_module
        self.store = MemoryConversationStateStore()
        self.memory = ChatMemoryContext(
            platform="qq",
            user_id="question-user",
            space_type="group",
            space_id="question-group",
            speaker_name="提问者",
        )
        self.author = self.memory.conversation_address
        patches = (
            patch.dict(os.environ, {
                "KEYTAO_SEMANTIC_ROUTE_OPS": "reorder,direct_exec,pending_release,pending_answer",
                "KEYTAO_SEMANTIC_SHADOW": "0",
            }),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.commands, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.commands, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "remember_conversation"),
            patch.object(self.chat, "schedule_memory_compaction"),
            patch.object(self.chat, "get_history", return_value=[]),
            patch.object(self.chat, "build_reply_context", AsyncMock(return_value="")),
            patch.object(self.chat, "_finish_ai_chat_response", new_callable=AsyncMock),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    async def asyncTearDown(self):
        tasks = tuple(self.chat._semantic_shadow_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.chat._semantic_shadow_tasks.clear()

    def test_leading_keytao_subject_is_not_stripped_while_address_prefixes_still_work(self):
        strip = self.routing._strip_command_message_prefixes
        # Subject sentences starting with 键道 must remain intact.
        for sentence in (
            "键道打字规则是声母加韵母加笔画是吧",
            "键道规则是什么",
            "键道怎么编码？",
            "键道双拼怎么打",
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(strip(sentence), sentence)
                self.assertFalse(self.routing._is_prefixed_fresh_word_query(sentence, strip(sentence)))

        # Real address prefixes still strip cleanly.
        self.assertEqual(strip("键道 百岁山"), "百岁山")
        self.assertEqual(strip("键道：查词 午时"), "查词 午时")
        self.assertEqual(strip("键道查词午时"), "查词午时")
        self.assertEqual(strip("喵喵 敬德"), "敬德")
        self.assertTrue(self.routing._is_prefixed_fresh_word_query("键道 百岁山", strip("键道 百岁山")))
        self.assertFalse(
            self.routing._is_prefixed_fresh_word_query(
                "@喵喵 键道打字规则是声母加韵母加笔画是吧",
                strip("@喵喵 键道打字规则是声母加韵母加笔画是吧"),
            )
        )

    async def test_unpunctuated_questions_never_become_bare_word_queries(self):
        questions = (
            "键道打字规则是声母加韵母加笔画是吧",
            "打字规则是声母加韵母加笔画是吧",
            "键道规则是什么",
            "键道怎么打字",
            "声母加韵母加笔画对吧",
            "键道是不是双拼加形码",
            "为什么词库里没有这个词",
            "双拼规则是对的吧",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(
                    is_interrogative_message(question)
                    or self.routing._looks_like_chat_sentence(question)
                )
                self.assertEqual(self.chat._bare_lexical_query_words(question), ())
                self.assertEqual(await self.routing._get_simple_word_query_words(question), ())

        # Real standalone dictionary words are still recognized normally.
        for word in ("百岁山", "敬德", "进度条", "超导体", "基督徒"):
            with self.subTest(word=word):
                self.assertFalse(is_interrogative_message(word))
                self.assertEqual(self.chat._bare_lexical_query_words(word), (word,))
                self.assertEqual(await self.routing._get_simple_word_query_words(word), (word,))

    async def test_production_incident_routes_to_ai_answer_not_word_add_even_if_semantic_parser_fails(self):
        message = "键道打字规则是声母加韵母加笔画是吧"
        ctx = self.chat.TurnContext(
            bot=None,
            event=None,
            platform="qq",
            user_id=self.author.actor_id,
            message_text=message,
            normalized_message_text=self.routing._strip_command_message_prefixes(message),
            conv_key=self.author,
            space_key=self.author.space_key,
            memory_context=self.memory,
            owner_label="提问者",
        )
        ctx.command_intent_cache, ctx.command_intent_for = self.routing.command_intent_memoizer(
            ctx.message_text, ctx.normalized_message_text,
        )
        ctx.message_is_prefixed_fresh_word_query = self.routing._is_prefixed_fresh_word_query(
            ctx.message_text, ctx.normalized_message_text,
        )
        tool_mock = AsyncMock(side_effect=AssertionError("Question must never trigger add-word tools"))
        ai_answer = "对的，键道的基本规则是双拼声母+韵母出音，再加上笔画/字根形码辅助。"

        with (
            patch.object(self.chat, "call_tool_function", tool_mock),
            patch.object(self.commands, "call_tool_function", tool_mock),
            patch.object(self.chat, "to_me", return_value=AsyncMock(return_value=False)),
            patch.object(self.chat, "get_ai_response_core", AsyncMock(return_value=ai_answer)) as ai_mock,
            patch("keytao_bot.harness.semantic_parser.parse_operation", AsyncMock(return_value=None)),
        ):
            for stage in (
                self.chat._stage_understand_operation_early,
                self.chat._stage_resolve_current_pending_scope,
                self.chat._stage_apply_scoped_pending_intent,
                self.chat._stage_arbitrate_active_operation,
                self.chat._stage_execute_pending_state,
                self.chat._stage_handle_draft_management,
                self.chat._stage_handle_simple_word_query,
                self.chat._stage_generate_ai_response,
            ):
                if await stage(ctx):
                    break

        tool_mock.assert_not_awaited()
        ai_mock.assert_awaited_once()
        self.assertEqual(ctx.response, ai_answer)
        self.assertIsNone(self.store.get_record(self.author))


if __name__ == "__main__":
    unittest.main()
