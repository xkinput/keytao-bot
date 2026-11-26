"""Offline regression tests for natural pending-card follow-ups (subset, order tweak, side Q&A)."""

import asyncio
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

from tests import test_state_machine as harness

from keytao_bot.harness.operation_request import OperationRequest
from keytao_bot.harness.semantic_parser import validate_operation
from keytao_bot.harness.state import (
    DraftOperationCoordinator,
    MemoryConversationStateStore,
    PendingAddWord,
    PendingToolConfirm,
)


ChatMemoryContext = harness.ChatMemoryContext


class _AsyncFakeClient(harness._FakeClient):
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


class PendingCardFollowupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.commands = harness.chat_commands_module
        self.routing = harness.chat_routing_module
        self.store = MemoryConversationStateStore()
        self.memory = ChatMemoryContext(
            platform="telegram",
            user_id="followup-owner",
            space_type="group",
            space_id="followup-group",
            speaker_name="Rea",
        )
        self.author = self.memory.conversation_address
        patches = (
            patch.dict(os.environ, {
                "KEYTAO_SEMANTIC_ROUTE_OPS": "reorder,pending_release,pending_answer,query,chat,help,direct_exec",
                "KEYTAO_SEMANTIC_SHADOW": "0",
            }),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.commands, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.commands, "draft_operation_coordinator", DraftOperationCoordinator()),
            patch.object(self.chat, "remember_conversation"),
            patch.object(self.chat, "schedule_memory_compaction"),
            patch.object(self.chat, "build_reply_context", AsyncMock(return_value="")),
            patch.object(self.chat, "_finish_ai_chat_response", new_callable=AsyncMock),
        )
        for ctx in patches:
            ctx.start()
            self.addCleanup(ctx.stop)

    def _multiword_state(self) -> PendingToolConfirm:
        scopes = []
        pairs = (("显眼包", "xybo"), ("嘴替", "zbtk"), ("松弛感", "swg"))
        for word, code in pairs:
            candidates = [(code, False), (code + "a", False)]
            single = PendingAddWord(word, code, candidates, server_candidates=candidates)
            scopes.append({
                "word": word,
                "candidates": candidates,
                "occupiedWords": {},
                "orderingAssessments": [],
                "reviewedState": self.commands._pending_add_word_payload(single),
            })
        return PendingToolConfirm(
            "keytao_batch_add_to_draft",
            {
                "items": [
                    {"action": "Create", "word": word, "code": code}
                    for word, code in pairs
                ],
                "_candidate_scopes": scopes,
                "_reviewed_multi_word": True,
                "_query_words": [word for word, _code in pairs],
            },
        )

    def _shift_plan_state(self) -> PendingToolConfirm:
        return PendingToolConfirm(
            "keytao_shift_phrase_code",
            {
                "word": "基督徒",
                "target_code": "jdt",
                "batch_id": "b1",
                "expected_content_version": 1,
                "confirmed_plan_digest": "a" * 64,
                "_pending_display": {
                    "shiftPlan": {
                        "word": "基督徒",
                        "targetCode": "jdt",
                        "currentState": [
                            {"word": "超导体", "code": "jdt", "weight": 100, "source": "dictionary"},
                            {"word": "基督徒", "code": "jdtv", "weight": 100, "source": "dictionary"},
                            {"word": "进度条", "code": "jdtvo", "weight": 100, "source": "dictionary"},
                        ],
                        "proposedState": [
                            {"word": "基督徒", "code": "jdt", "weight": 100, "source": "dictionary"},
                            {"word": "超导体", "code": "jdtv", "weight": 100, "source": "dictionary"},
                            {"word": "进度条", "code": "jdtvo", "weight": 100, "source": "dictionary"},
                        ],
                        "shifted": [
                            {"word": "基督徒", "fromCode": "jdtv", "toCode": "jdt"},
                            {"word": "超导体", "fromCode": "jdt", "toCode": "jdtv"},
                        ],
                    }
                },
            },
            confirmation_source="server_warning",
        )

    def test_pending_context_exposes_shift_plan_order_and_multiword_candidate_scopes(self):
        shift_payload = json.loads(
            self.routing._pending_context_for_command_intent(self._shift_plan_state())
        )
        self.assertEqual(shift_payload["proposed_order"], ["基督徒", "超导体", "进度条"])
        self.assertEqual(
            [(row["word"], row["code"]) for row in shift_payload["proposed_entries"]],
            [("基督徒", "jdt"), ("超导体", "jdtv"), ("进度条", "jdtvo")],
        )

        multi_payload = json.loads(
            self.routing._pending_context_for_command_intent(self._multiword_state())
        )
        self.assertEqual(
            [(row["index"], row["word"], row["recommended_code"]) for row in multi_payload["candidate_scopes"]],
            [(1, "显眼包", "xybo"), (2, "嘴替", "zbtk"), (3, "松弛感", "swg")],
        )

    def test_multiword_card_natural_subset_selection_and_safety_contrasts(self):
        state = self._multiword_state()
        prompt = "回复「加入」写入草稿，或「加入并提交」。"
        allowed_cases = (
            (("显眼包", "嘴替"), (), False, "只加前两个", "显眼包 xybo，嘴替 zbtk，加入"),
            (("显眼包", "嘴替"), ("xybo", "zbtk"), True, "只加前两个并提交", "显眼包 xybo，嘴替 zbtk，加入并提交"),
            (("显眼包",), (), False, "只加第一个", "显眼包 xybo，加入"),
            (("显眼包", "松弛感"), (), False, "只加第1、3个", "显眼包 xybo，松弛感 swg，加入"),
            (("显眼包", "嘴替"), (), False, "只要显眼包和嘴替", "显眼包 xybo，嘴替 zbtk，加入"),
            (("显眼包",), ("xybo",), False, "先加显眼包，嘴替先别加", "显眼包 xybo，加入"),
            (("显眼包", "嘴替"), (), False, "除了松弛感其他都加", "显眼包 xybo，嘴替 zbtk，加入"),
            (("显眼包", "松弛感"), (), False, "除了第二个其他都加", "显眼包 xybo，松弛感 swg，加入"),
        )
        for words, codes, submit_after, message, expected in allowed_cases:
            with self.subTest(message=message):
                request = OperationRequest(
                    op="select_candidate",
                    operands=words,
                    codes=codes,
                    submit_after=submit_after,
                    refers_to_pending=True,
                    frame="direct",
                )
                self.assertEqual(
                    self.commands.render_understood_pending_answer(request, state, prompt, message),
                    expected,
                )

        # Safety contrasts: ambiguous/dropped/hallucinated subsets are still rejected.
        rejected_cases = (
            (("显眼包",), ("xybo",), "就这样加吧"),
            (("嘴替",), (), "只加第一个"),
            (("显眼包",), ("xyba",), "只加显眼包"),
            (("显眼包", "嘴替"), (), "显眼包、嘴替和松弛感都加"),
        )
        for words, codes, message in rejected_cases:
            with self.subTest(rejected=message):
                request = OperationRequest(
                    op="select_candidate",
                    operands=words,
                    codes=codes,
                    refers_to_pending=True,
                    frame="direct",
                )
                self.assertEqual(
                    self.commands.render_understood_pending_answer(request, state, prompt, message),
                    "",
                )

    async def test_shift_plan_card_natural_order_tweak_rebuilds_plan_end_to_end(self):
        state = self._shift_plan_state()
        prompt = self.commands.render_pending_shift_plan(state)
        self.store.set(self.author, state, owner_label="Rea")
        self.store.bind_origin_prompt_digest(
            self.author,
            self.routing._prompt_capability_digest(prompt),
        )

        # Validate semantic parser preserves refers_to_pending for shift-plan order tweaks.
        raw_tweak = {
            "op": "reorder",
            "operands": ["进度条", "基督徒", "超导体"],
            "codes": [],
            "code_selector": "",
            "submit_after": False,
            "refers_to_pending": True,
            "frame": "direct",
            "confidence": "high",
            "note": "",
            "relation": "explicit_order",
        }
        validated, violations = validate_operation(raw_tweak, "把进度条放第一个", state)
        self.assertEqual(violations, [])
        self.assertTrue(validated.refers_to_pending)
        self.assertEqual(validated.operands, ("进度条", "基督徒", "超导体"))

        ctx = self.chat.TurnContext(
            bot=None,
            event=None,
            platform="telegram",
            user_id=self.author.actor_id,
            message_text="把进度条放第一个",
            normalized_message_text="把进度条放第一个",
            conv_key=self.author,
            space_key=self.author.space_key,
            memory_context=self.memory,
            owner_label="Rea",
            history=[{"role": "assistant", "content": prompt}],
        )
        ctx.command_intent_cache, ctx.command_intent_for = self.routing.command_intent_memoizer(
            ctx.message_text, ctx.normalized_message_text,
        )
        new_plan_reply = "🔁 调整计划：\n• 进度条：jdtvo → jdt\n回复「确认」或「确认并提交」执行，或「取消」。"
        reorder_mock = AsyncMock(return_value=new_plan_reply)
        client = _AsyncFakeClient([])
        with (
            patch.object(self.chat, "OPENAI_API_KEY", "offline-key"),
            patch.object(self.chat, "AsyncOpenAI", return_value=client),
            patch.object(self.chat, "get_history", return_value=ctx.history),
            patch.object(self.commands, "_try_handle_explicit_order_reorder_command", reorder_mock),
            patch("keytao_bot.harness.semantic_parser.parse_operation", AsyncMock(return_value=raw_tweak)),
        ):
            self.assertTrue(await self.chat._stage_understand_operation_early(ctx))

        reorder_mock.assert_awaited_once()
        cmd = reorder_mock.await_args.args[0]
        self.assertEqual(cmd.words, ("进度条", "基督徒", "超导体"))
        self.assertEqual(ctx.response, new_plan_reply)

    async def test_side_question_about_live_card_without_literal_word_preserves_ticket_and_passes_card_context(self):
        state = self._shift_plan_state()
        prompt = self.commands.render_pending_shift_plan(state)
        self.store.set(self.author, state, owner_label="Rea")
        self.store.bind_origin_prompt_digest(
            self.author,
            self.routing._prompt_capability_digest(prompt),
        )
        message = "为什么这个排前面？"
        ctx = self.chat.TurnContext(
            bot=None,
            event=None,
            platform="telegram",
            user_id=self.author.actor_id,
            message_text=message,
            normalized_message_text=message,
            conv_key=self.author,
            space_key=self.author.space_key,
            memory_context=self.memory,
            owner_label="Rea",
            history=[{"role": "assistant", "content": prompt}],
        )
        ctx.command_intent_cache, ctx.command_intent_for = self.routing.command_intent_memoizer(
            ctx.message_text, ctx.normalized_message_text,
        )
        raw_question = {
            "op": "query",
            "operands": [],
            "codes": [],
            "code_selector": "",
            "submit_after": False,
            "refers_to_pending": True,
            "frame": "none",
            "confidence": "high",
            "note": "",
            "relation": "",
        }
        client = _AsyncFakeClient([])
        ai_mock = AsyncMock(return_value="因为「基督徒」在语料中的常用度更高，所以默认排在首位。")
        with (
            patch.object(self.chat, "OPENAI_API_KEY", "offline-key"),
            patch.object(self.chat, "AsyncOpenAI", return_value=client),
            patch.object(self.chat, "get_history", return_value=ctx.history),
            patch.object(self.chat, "to_me", return_value=AsyncMock(return_value=True)),
            patch.object(self.chat, "get_ai_response_core", ai_mock),
            patch("keytao_bot.harness.semantic_parser.parse_operation", AsyncMock(return_value=raw_question)),
        ):
            self.assertFalse(await self.chat._stage_understand_operation_early(ctx))
            self.assertTrue(ctx.semantic_pending_released)
            await self.chat._stage_generate_ai_response(ctx)

        ai_mock.assert_awaited_once()
        self.assertTrue(ai_mock.await_args.kwargs["preserve_pending"])
        passed_reply_context = ai_mock.await_args.args[4]
        self.assertIn("当前待确认卡片内容", passed_reply_context)
        self.assertIn("基督徒", passed_reply_context)
        # Live pending card stays intact for the user's next confirm/cancel reply!
        self.assertIs(self.store.get_record(self.author).state, state)


if __name__ == "__main__":
    unittest.main()
