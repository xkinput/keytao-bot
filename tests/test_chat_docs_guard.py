"""Offline coverage for the one-shot scheme documentation correction."""

import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests import test_state_machine as harness


FALLBACK = "文档里没查到相关说明，请查阅 https://keytao-docs.vercel.app"


class DocsSkills:
    def get_skill_instructions(self):
        return ""

    def has_tools(self):
        return True

    def get_tools(self):
        return [{
            "type": "function",
            "function": {
                "name": "keytao_fetch_docs",
                "description": "Fetch scheme documentation",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }]


class ChatDocsGuardTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.chat = harness.openai_chat_module
        self.store = harness.MemoryConversationStateStore()
        self.tools = AsyncMock(return_value={
            "success": True, "content": "Offline documentation result", "sources": [],
        })
        patches = (
            patch.object(self.chat, "OPENAI_API_KEY", "offline-key"),
            patch.object(self.chat, "skills_manager", DocsSkills()),
            patch.object(self.chat, "conversation_state_store", self.store),
            patch.object(self.chat, "draft_operation_coordinator", harness.DraftOperationCoordinator()),
            patch.object(self.chat, "tool_executor", harness.ToolExecutor(lambda _name: self.tools, frozenset())),
            patch.object(self.chat, "_record_agent_tool_receipt", Mock()),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)
        token = harness.begin_turn_metrics("qq", "private")
        self.addCleanup(harness.end_turn_metrics, token)

    async def answer(self, message, responses, **kwargs):
        client = harness._FakeClient(responses)
        with patch.object(self.chat, "AsyncOpenAI", return_value=client):
            result = await self.chat.get_ai_response_core(
                message, "qq", "docs-owner", command_intent=harness.MessageCommandIntent(),
                **kwargs,
            )
        return result, client.completions.calls

    async def test_tool_free_concept_answer_gets_exactly_one_corrective_retry(self):
        result, calls = await self.answer("v的二分有啥用", [
            harness._FakeAIResponse("stop", "Ungrounded scheme rules"),
            harness._FakeAIResponse("stop", "Still ungrounded"),
        ], max_iterations=1)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result, FALLBACK)
        correction = calls[1]["messages"][-1]
        self.assertEqual(correction["role"], "system")
        self.assertIn("keytao_fetch_docs", correction["content"])
        self.assertIn("keytao_encode", correction["content"])
        self.assertIn("不得猜测", correction["content"])

    async def test_corrective_tool_call_proceeds_to_grounded_completion(self):
        tool_call = types.SimpleNamespace(
            id="offline-docs-call", type="function",
            function=types.SimpleNamespace(
                name="keytao_fetch_docs",
                arguments=json.dumps({"query": "v的二分有啥用"}, ensure_ascii=False),
            ),
        )
        result, calls = await self.answer("v 的二分有啥用？", [
            harness._FakeAIResponse("stop", "Ungrounded scheme rules"),
            harness._FakeAIResponse("tool_calls", tool_calls=[tool_call]),
            harness._FakeAIResponse("stop", "文档提供的二分说明。"),
        ], conversation_only=True, preserve_pending=True)
        self.assertEqual(len(calls), 3)
        self.tools.assert_awaited_once_with(query="v的二分有啥用")
        self.assertEqual(result, "文档提供的二分说明。")
        self.assertIn("keytao_fetch_docs", {tool["function"]["name"] for tool in calls[1]["tools"]})

    async def test_existing_tool_call_in_whole_turn_does_not_retry(self):
        from keytao_bot.utils.observability import current_turn_metrics

        current_turn_metrics().tool_calls = 1
        result, calls = await self.answer("v的二分有啥用", [
            harness._FakeAIResponse("stop", "Uses the earlier tool result"),
        ])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result, "Uses the earlier tool result")

    async def test_greetings_and_word_lookups_do_not_retry(self):
        for message in ("你好", "你好？", "查词 海东青？", "海东青是什么", "朱祁镇的编码是什么？"):
            with self.subTest(message=message):
                result, calls = await self.answer(message, [
                    harness._FakeAIResponse("stop", "Offline answer"),
                ])
                self.assertEqual(len(calls), 1)
                self.assertEqual(result, "Offline answer")

    async def test_keyword_only_message_does_not_retry(self):
        result, calls = await self.answer("v的二分有啥用", [
            harness._FakeAIResponse("stop", "Offline answer"),
        ], docs_guard_allowed=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result, "Offline answer")

    async def test_pending_and_active_operation_turns_do_not_retry(self):
        address = harness.ConversationAddress.private("qq", "docs-owner")
        self.store.set(address, harness.PendingAddWord(word="午时", recommended_code="wj", candidates=[("wj", False)]))
        result, calls = await self.answer("v的二分有啥用", [
            harness._FakeAIResponse("stop", "Offline answer"),
        ])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result, "Offline answer")
        self.store.delete(address)
        self.chat.draft_operation_coordinator.begin(address, "offline-operation")
        _, calls = await self.answer("v的二分有啥用", [
            harness._FakeAIResponse("stop", "Offline answer"),
        ])
        self.assertEqual(len(calls), 1)

    async def test_add_and_command_flows_do_not_retry(self):
        for message in ("加入二分规则词", "添加 编码规则", "提交草稿？"):
            with self.subTest(message=message):
                _, calls = await self.answer(message, [harness._FakeAIResponse("stop", "Offline answer")])
                self.assertEqual(len(calls), 1)

    async def test_keyword_only_stage_disables_guard(self):
        ctx = self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id="docs-owner",
            normalized_message_text="v的二分有啥用", history=[],
        )
        with (
            patch.object(self.chat, "to_me", return_value=AsyncMock(return_value=False)),
            patch.object(self.chat, "build_reply_context", AsyncMock(return_value="")),
            patch.object(self.chat, "get_ai_response_core", AsyncMock(return_value="Offline answer")) as core,
        ):
            await self.chat._stage_generate_ai_response(ctx)
        self.assertFalse(core.await_args.kwargs["docs_guard_allowed"])

    async def test_addressed_concept_stage_enables_guard_without_another_classifier(self):
        ctx = self.chat.TurnContext(
            bot=None, event=None, platform="qq", user_id="docs-owner",
            normalized_message_text="键道怎么编码？", history=[],
        )
        with (
            patch.object(self.chat, "to_me", return_value=AsyncMock(return_value=True)),
            patch.object(self.chat, "build_reply_context", AsyncMock(return_value="")),
            patch.object(self.chat, "get_ai_response_core", AsyncMock(return_value="Offline answer")) as core,
        ):
            await self.chat._stage_generate_ai_response(ctx)
        self.assertTrue(core.await_args.kwargs["docs_guard_allowed"])


if __name__ == "__main__":
    unittest.main()
