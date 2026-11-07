"""Offline GLM S49 regression tests with exact fake-client requests."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import nonebot

with patch("nonebot.config.dotenv_values", return_value={}):
    nonebot.init(openai_api_key="fixture-key", openai_model="glm-5.3")

from keytao_bot.harness.orchestrator import (
    AgentOrchestrator,
    AgentRequestContext,
    AgentRuntimeConfig,
    _is_reasoning_only_exhaustion,
    _REASONING_RETRY_SYSTEM_PROMPT,
)
from keytao_bot.harness.state import MemoryConversationStateStore
from keytao_bot.harness.tools import ToolExecutor
from keytao_bot.utils.llm_policy import chat_usage_metrics


def fake_response(usage):
    return SimpleNamespace(
        model="glm-5.3", usage=usage,
        choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(
            content="", tool_calls=[], reasoning_content=None,
        ))],
    )


USAGE_SHAPES = (
    {"completion_tokens": 1800, "completion_tokens_details": {"reasoning_tokens": 1800}},
    SimpleNamespace(completion_tokens=1800,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=1800)),
    {"completion_tokens": 1800},
    {"completion_tokens": 1800, "completion_tokens_details": {}},
    {"completion_tokens": 1800, "completion_tokens_details": None},
    {"completion_tokens": 1800, "completion_tokens_details": {"reasoning_tokens": None}},
    None,
)


class GlmUsageTests(unittest.TestCase):
    def test_usage_shapes_keep_the_guard_active_without_inventing_counts(self):
        for usage in USAGE_SHAPES:
            with self.subTest(usage=usage):
                response = fake_response(usage)
                self.assertTrue(_is_reasoning_only_exhaustion(
                    response, finish_reason="length", content="", tool_calls=[],
                ))
        self.assertNotIn("reasoning_tokens", chat_usage_metrics(fake_response(
            {"completion_tokens": 1800},
        )))

    def test_visible_output_tools_and_reported_nonreasoning_are_not_reasoning_only(self):
        for finish, content, tool_calls in (
            ("stop", "", []), ("length", "answer", []), ("length", "", [object()]),
        ):
            with self.subTest(finish=finish, content=content, tools=bool(tool_calls)):
                self.assertFalse(_is_reasoning_only_exhaustion(
                    fake_response(None), finish_reason=finish, content=content, tool_calls=tool_calls,
                ))
        self.assertFalse(_is_reasoning_only_exhaustion(
            fake_response({"completion_tokens": 1800,
                           "completion_tokens_details": {"reasoning_tokens": 0}}),
            finish_reason="length", content="", tool_calls=[],
        ))


class GlmRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_retry_body_lowers_effort_without_increasing_tokens(self):
        for usage in USAGE_SHAPES:
            with self.subTest(usage=usage):
                bodies = []

                async def create(**body):
                    bodies.append(copy.deepcopy(body))
                    return fake_response(usage)

                client = SimpleNamespace(chat=SimpleNamespace(
                    completions=SimpleNamespace(create=create),
                ))
                tools = Mock(side_effect=AssertionError("No tool dispatch is allowed"))
                orchestrator = AgentOrchestrator(
                    client_factory=lambda: client,
                    runtime=AgentRuntimeConfig(model="glm-5.3", max_tokens=1800,
                        temperature=0.2, timeout=10.0, max_tokens_cap=7200),
                    skills_manager=SimpleNamespace(get_skill_instructions=lambda: "",
                                                   has_tools=lambda: False),
                    tool_executor=ToolExecutor(tools, frozenset()),
                    state_store=MemoryConversationStateStore(),
                    bind_help_text="fixture", system_prompt_core="fixture-system " * 1000,
                )
                termination = {}
                await orchestrator._run_loop(
                    "Explain the fixture.",
                    AgentRequestContext(platform="qq", user_id="fixture", history=[
                        {"role": "user", "content": "old context"},
                        {"role": "assistant", "content": "old answer"},
                    ]),
                    termination_state=termination,
                )
                self.assertEqual(len(bodies), 2)
                self.assertEqual(bodies[0], {
                    "model": "glm-5.3", "messages": bodies[0]["messages"],
                    "max_tokens": 1800, "temperature": 0.2,
                    "extra_body": {"thinking": {"type": "enabled"}},
                    "reasoning_effort": "high",
                })
                self.assertEqual(bodies[1], {
                    "model": "glm-5.3", "messages": [
                        {"role": "system", "content": _REASONING_RETRY_SYSTEM_PROMPT},
                        {"role": "user", "content": "[当前请求] Explain the fixture."},
                    ],
                    "max_tokens": 1800, "temperature": 0.2,
                    "extra_body": {"thinking": {"type": "enabled"}},
                    "reasoning_effort": "low",
                })
                self.assertEqual(termination["reason"], "reasoning_runaway")
                tools.assert_not_called()


if __name__ == "__main__":
    unittest.main()
