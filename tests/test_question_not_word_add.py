"""Offline regression checks ensuring questions are answered instead of entering add-word flows."""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, patch

from tests import test_state_machine as harness

from keytao_bot.harness.authorization_grammar import (
    is_interrogative_message,
    is_usage_help_question,
    looks_like_mutation_grammar_gap,
)
from keytao_bot.harness.orchestrator import HELP_TURN_GUIDANCE
from keytao_bot.harness.state import (
    DraftOperationCoordinator,
    MemoryConversationStateStore,
    PendingAddWord,
)
from keytao_bot.utils.pending_confirmation import (
    UsageHelpReply,
    is_safe_usage_help_reply,
)


ChatMemoryContext = harness.ChatMemoryContext


class _AsyncFakeClient(harness._FakeClient):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


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

    async def test_usage_help_questions_are_not_grammar_gaps_or_bare_word_queries(self):
        usage_questions = (
            "喵喵 我怎么找你加词",
            "我怎么找你加词",
            "怎么加词",
            "怎么删词",
            "怎么改码",
            "怎么重新排序",
            "怎么给词库加词",
            "怎么在群里加词",
            "怎么提交草稿",
            "怎么绑定账号",
            "喵喵 你有哪些功能",
            "你能干嘛",
        )
        for question in usage_questions:
            stripped = self.routing._strip_command_message_prefixes(question)
            with self.subTest(question=question):
                self.assertTrue(is_interrogative_message(stripped))
                self.assertTrue(is_usage_help_question(question))
                self.assertFalse(looks_like_mutation_grammar_gap(question))
                self.assertFalse(looks_like_mutation_grammar_gap(stripped))
                self.assertFalse(self.routing._is_prefixed_fresh_word_query(question, stripped))
                self.assertEqual(self.chat._bare_lexical_query_words(stripped), ())
                self.assertEqual(await self.routing._get_simple_word_query_words(stripped), ())

        # Dictionary words and real unrecognized write verbs keep their existing behavior.
        for word in ("不怎么样", "怎么着", "怎么后移"):
            with self.subTest(word=word):
                self.assertFalse(is_interrogative_message(word))
                self.assertFalse(is_usage_help_question(word))
        self.assertFalse(is_usage_help_question("怎么"))
        self.assertTrue(looks_like_mutation_grammar_gap("把 奈飞 迁到 nhfw"))
        self.assertFalse(is_usage_help_question("把 奈飞 迁到 nhfw"))
        self.assertFalse(is_usage_help_question("ennnn 是什么编码"))

    async def test_how_to_add_word_help_reply_preserves_usage_examples_end_to_end(self):
        help_answer = (
            "直接把想加的词发给我，或者发「加词 词条」就行啦～\n"
            "我会先帮你查重、审词并给出候选编码，你再按提示回复「加入」或「加入并提交」就能加入草稿；"
            "也可以直接指定编码，比如「加词 进度条 jdt」。\n"
            "想先检查再提审的话，可以发「查看草稿」，确认后再发「提交」。"
        )
        for semantic_payload, route_ops in (
            (
                {
                    "op": "help",
                    "operands": [],
                    "codes": [],
                    "code_selector": "",
                    "submit_after": False,
                    "refers_to_pending": False,
                    "frame": "none",
                    "confidence": "high",
                    "note": "",
                    "relation": "",
                },
                "swap,delete,recode,move,reorder,batch_replace,draft_submit,draft_recall,draft_clear,draft_keep_only,query,chat,help,pending_release,pending_answer,direct_exec,add",
            ),
            (None, "reorder,direct_exec,pending_release,pending_answer"),
        ):
            with self.subTest(semantic_op=None if semantic_payload is None else semantic_payload["op"]):
                self.store.delete(self.author)
                client = _AsyncFakeClient([harness._FakeAIResponse("stop", help_answer)])
                message = "喵喵 我怎么找你加词"
                ctx = self.chat.TurnContext(
                    bot=None,
                    event=None,
                    platform="telegram",
                    user_id=self.author.actor_id,
                    message_text=message,
                    normalized_message_text=self.routing._strip_command_message_prefixes(message),
                    conv_key=self.author,
                    space_key=self.author.space_key,
                    memory_context=self.memory,
                    owner_label="Rea",
                )
                ctx.command_intent_cache, ctx.command_intent_for = self.routing.command_intent_memoizer(
                    ctx.message_text, ctx.normalized_message_text,
                )
                ctx.message_is_prefixed_fresh_word_query = self.routing._is_prefixed_fresh_word_query(
                    ctx.message_text, ctx.normalized_message_text,
                )
                self.assertFalse(ctx.message_is_prefixed_fresh_word_query)
                tool_mock = AsyncMock(side_effect=AssertionError("Help question must not invoke write tools"))
                token = self.chat._current_turn_message.set(ctx.normalized_message_text)
                try:
                    with (
                        patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": route_ops, "KEYTAO_SEMANTIC_SHADOW": "0"}),
                        patch.object(self.chat, "OPENAI_API_KEY", "offline-key"),
                        patch.object(self.chat, "AsyncOpenAI", return_value=client),
                        patch.object(self.routing, "OPENAI_API_KEY", ""),
                        patch.object(self.chat, "call_tool_function", tool_mock),
                        patch.object(self.commands, "call_tool_function", tool_mock),
                        patch.object(self.chat, "to_me", return_value=AsyncMock(return_value=True)),
                        patch.object(self.chat.memory_store, "get_context_block", return_value=""),
                        patch.object(self.chat, "get_group_history_context", return_value=""),
                        patch(
                            "keytao_bot.harness.semantic_parser.parse_operation",
                            AsyncMock(return_value=semantic_payload),
                        ),
                    ):
                        for stage in (
                            self.chat._stage_understand_operation_early,
                            self.chat._stage_resolve_current_pending_scope,
                            self.chat._stage_apply_scoped_pending_intent,
                            self.chat._stage_arbitrate_active_operation,
                            self.chat._stage_execute_pending_state,
                            self.chat._stage_handle_draft_management,
                            self.chat._stage_handle_simple_word_query,
                            *self.chat.RESPONSE_STAGES,
                        ):
                            if await stage(ctx):
                                break
                    delivered = self.chat._prepare_user_facing_reply(ctx.response, self.memory)
                finally:
                    self.chat._current_turn_message.reset(token)

                tool_mock.assert_not_awaited()
                self.assertEqual(len(client.completions.calls), 1)
                system_prompts = [
                    item["content"]
                    for item in client.completions.calls[0]["messages"]
                    if item.get("role") == "system"
                ]
                self.assertIn(HELP_TURN_GUIDANCE, system_prompts)
                if semantic_payload is not None:
                    self.assertLess(len(system_prompts[0]), 2000)
                self.assertEqual(str(ctx.response), help_answer)
                self.assertIsInstance(ctx.response, UsageHelpReply)
                self.assertEqual(str(delivered), help_answer)
                self.assertNotIn("还没有可确认的计划", delivered)
                self.assertNotIn("没找到要执行的操作", delivered)

    async def test_targetless_operation_question_from_semantic_parser_routes_to_fast_help_branch(self):
        help_answer = "想删词可以直接发「删除 词条」，我会先核对词条并给你确认计划，确认后再发「提交」即可。"
        client = _AsyncFakeClient([harness._FakeAIResponse("stop", help_answer)])
        message = "喵喵 怎么把词库里的词删掉"
        ctx = self.chat.TurnContext(
            bot=None,
            event=None,
            platform="telegram",
            user_id=self.author.actor_id,
            message_text=message,
            normalized_message_text=self.routing._strip_command_message_prefixes(message),
            conv_key=self.author,
            space_key=self.author.space_key,
            memory_context=self.memory,
            owner_label="Rea",
        )
        ctx.command_intent_cache, ctx.command_intent_for = self.routing.command_intent_memoizer(
            ctx.message_text, ctx.normalized_message_text,
        )
        semantic_payload = {
            "op": "delete",
            "operands": [],
            "codes": [],
            "code_selector": "",
            "submit_after": False,
            "refers_to_pending": False,
            "frame": "question",
            "confidence": "high",
            "note": "",
            "relation": "",
        }
        token = self.chat._current_turn_message.set(ctx.normalized_message_text)
        try:
            with (
                patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": "delete,help,chat,query", "KEYTAO_SEMANTIC_SHADOW": "0"}),
                patch.object(self.chat, "OPENAI_API_KEY", "offline-key"),
                patch.object(self.chat, "AsyncOpenAI", return_value=client),
                patch.object(self.routing, "OPENAI_API_KEY", ""),
                patch.object(self.chat.memory_store, "get_context_block", return_value=""),
                patch.object(self.chat, "get_group_history_context", return_value=""),
                patch("keytao_bot.harness.semantic_parser.parse_operation", AsyncMock(return_value=semantic_payload)),
            ):
                self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
            delivered = self.chat._prepare_user_facing_reply(ctx.response, self.memory)
        finally:
            self.chat._current_turn_message.reset(token)

        self.assertEqual(ctx.semantic_nonwrite_route, "help")
        self.assertEqual(str(delivered), help_answer)
        self.assertEqual(len(client.completions.calls), 1)
        self.assertLess(len(client.completions.calls[0]["messages"][0]["content"]), 2000)

    async def test_help_turn_with_unrelated_preserved_ticket_does_not_redraw_old_ticket(self):
        existing_ticket = PendingAddWord(
            word="午时",
            recommended_code="wsu",
            candidates=[("wsu", False)],
            server_candidates=[("wsu", False)],
        )
        self.store.set(self.author, existing_ticket, owner_label="Rea")
        self.store.bind_origin_prompt_digest(
            self.author,
            self.routing._prompt_capability_digest("「午时」未收录\n1. wsu — ✅推荐\n回复「加入」或「加入并提交」。"),
        )
        help_answer = "直接发想加的词，或者发「加词 词条」就行；审词后按提示回复「加入」或「加入并提交」即可。"
        client = _AsyncFakeClient([harness._FakeAIResponse("stop", help_answer)])
        message = "喵喵 我怎么找你加词"
        ctx = self.chat.TurnContext(
            bot=None,
            event=None,
            platform="telegram",
            user_id=self.author.actor_id,
            message_text=message,
            normalized_message_text=self.routing._strip_command_message_prefixes(message),
            conv_key=self.author,
            space_key=self.author.space_key,
            memory_context=self.memory,
            owner_label="Rea",
        )
        ctx.command_intent_cache, ctx.command_intent_for = self.routing.command_intent_memoizer(
            ctx.message_text, ctx.normalized_message_text,
        )
        semantic_payload = {
            "op": "help",
            "operands": [],
            "codes": [],
            "code_selector": "",
            "submit_after": False,
            "refers_to_pending": False,
            "frame": "none",
            "confidence": "high",
            "note": "",
            "relation": "",
        }
        token = self.chat._current_turn_message.set(ctx.normalized_message_text)
        try:
            with (
                patch.dict(os.environ, {"KEYTAO_SEMANTIC_ROUTE_OPS": "help,pending_release,pending_answer", "KEYTAO_SEMANTIC_SHADOW": "0"}),
                patch.object(self.chat, "OPENAI_API_KEY", "offline-key"),
                patch.object(self.chat, "AsyncOpenAI", return_value=client),
                patch.object(self.routing, "OPENAI_API_KEY", ""),
                patch.object(self.chat.memory_store, "get_context_block", return_value=""),
                patch.object(self.chat, "get_group_history_context", return_value=""),
                patch("keytao_bot.harness.semantic_parser.parse_operation", AsyncMock(return_value=semantic_payload)),
            ):
                self.assertTrue(await self.chat._stage_understand_operation_early(ctx))
            delivered = self.chat._prepare_user_facing_reply(ctx.response, self.memory)
        finally:
            self.chat._current_turn_message.reset(token)

        self.assertEqual(str(delivered), help_answer)
        self.assertIs(self.store.get_record(self.author).state, existing_ticket)

    def test_help_guard_still_blocks_forged_candidates_bare_confirm_and_fake_writes(self):
        question = "我怎么找你加词"
        unsafe_replies = (
            "请回复「确认」。",
            "回复「加入并提交」即可执行。",
            "1. jdt — ✅推荐\n2. jdtv — 空位\n回复「加入」。",
            "- 「进度条」 → jdt\n回复「加入」或「加入并提交」。",
            "已加入草稿，可以发「提交」审核。",
            "可以把列表中的词加入草稿。",
        )
        for reply in unsafe_replies:
            with self.subTest(reply=reply):
                self.assertFalse(is_safe_usage_help_reply(question, reply, help_turn=True))
                token = self.chat._current_turn_message.set(question)
                try:
                    delivered = self.chat._enforce_advertised_reply_contract(reply, self.author)
                finally:
                    self.chat._current_turn_message.reset(token)
                if "已加入草稿" not in reply:
                    self.assertEqual(delivered, "没找到要执行的操作，本次未写入。")


if __name__ == "__main__":
    unittest.main()
