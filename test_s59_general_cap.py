"""S59 offline tests for bounded tool-free completions and prompt copy."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import nonebot

with patch("nonebot.config.dotenv_values", return_value={}):
    nonebot.init(openai_api_key="fixture-key", openai_model="glm-5.3")

from keytao_bot.harness.orchestrator import (
    AgentOrchestrator,
    AgentRequestContext,
    AgentRuntimeConfig,
    READ_ONLY_TURN_GUIDANCE,
    _REASONING_RETRY_SYSTEM_PROMPT,
)
from keytao_bot.harness.state import MemoryConversationStateStore
from keytao_bot.harness.tools import ToolExecutor
from keytao_bot.utils.observability import (
    begin_turn_metrics,
    current_turn_metrics,
    end_turn_metrics,
    observe_model_call,
)


def fake_response(finish="stop", content="", *, reasoning=None, tool_calls=()):
    return SimpleNamespace(
        model="glm-5.3",
        usage={"completion_tokens": 1800,
               "completion_tokens_details": {"reasoning_tokens": 0}},
        choices=[SimpleNamespace(
            finish_reason=finish,
            message=SimpleNamespace(content=content, tool_calls=list(tool_calls),
                                    reasoning_content=reasoning),
        )],
    )


class GeneralCapTests(unittest.IsolatedAsyncioTestCase):
    async def run_fake(self, responses, *, fallback=None, schema=None):
        bodies = []
        answers = iter(responses)

        async def create(**body):
            bodies.append(copy.deepcopy(body))
            response = next(answers, fake_response(content="Unexpected third answer."))
            if isinstance(response, Exception):
                raise response
            return response

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        dispatch = AsyncMock(return_value={"success": True, "message": "Fixture result."})
        manager = SimpleNamespace(
            get_skill_instructions=lambda: "",
            has_tools=lambda: bool(schema),
            get_tools=lambda: [schema] if schema else [],
        )
        orchestrator = AgentOrchestrator(
            client_factory=lambda: client,
            runtime=AgentRuntimeConfig(model="glm-5.3", max_tokens=1800,
                temperature=0.2, timeout=10.0, max_tokens_cap=7200),
            skills_manager=manager,
            tool_executor=ToolExecutor(lambda _name: dispatch, frozenset()),
            state_store=MemoryConversationStateStore(),
            bind_help_text="fixture", system_prompt_core="fixture-system " * 1000,
            deterministic_fallback_handler=fallback,
        )
        termination = {}
        reply = await orchestrator._run_loop(
            "Explain the fixture.",
            AgentRequestContext(platform="qq", user_id="fixture", history=[
                {"role": "user", "content": "old context"},
                {"role": "assistant", "content": "old answer"},
            ]),
            termination_state=termination,
        )
        return reply, bodies, dispatch, termination

    def assert_retry_shape(self, bodies, *, read_only_tools=False):
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
                *([{"role": "system", "content": READ_ONLY_TURN_GUIDANCE}]
                  if read_only_tools else []),
            ],
            "max_tokens": 1800, "temperature": 0.2,
            "extra_body": {"thinking": {"type": "enabled"}},
            "reasoning_effort": "low",
        })

    async def test_tool_free_incomplete_responses_share_one_compacted_retry(self):
        cases = (
            (fake_response("length"), fake_response("length")),
            (fake_response("stop", "  "), fake_response("length")),
            (fake_response("length"), fake_response("stop", "  ")),
            (fake_response("length", "Partial output"), fake_response("length", "Still partial")),
            (fake_response("stop", reasoning="Fixture reasoning"), fake_response("stop")),
            (fake_response("stop"), fake_response("length", reasoning="Fixture reasoning")),
        )
        for index, responses in enumerate(cases):
            with self.subTest(case=index):
                reply, bodies, dispatch, termination = await self.run_fake(responses)
                self.assert_retry_shape(bodies)
                self.assertNotIn("Unexpected", reply)
                self.assertNotIn("Partial", reply)
                self.assertTrue(termination.get("reason"))
                dispatch.assert_not_awaited()

    async def test_second_final_answer_is_retained(self):
        reply, bodies, dispatch, _ = await self.run_fake([
            fake_response("length"), fake_response(content="Fixture answer."),
        ])
        self.assert_retry_shape(bodies)
        self.assertEqual(reply, "Fixture answer.")
        dispatch.assert_not_awaited()

    async def test_failure_uses_existing_deterministic_fallback(self):
        fallback = AsyncMock(return_value="Fixture fallback.")
        reply, bodies, dispatch, _ = await self.run_fake([
            fake_response(), fake_response(),
        ], fallback=fallback)
        self.assert_retry_shape(bodies)
        self.assertEqual(reply, "Fixture fallback.")
        fallback.assert_awaited_once()
        dispatch.assert_not_awaited()

    async def test_provider_errors_and_complete_answers_do_not_retry(self):
        for response in (RuntimeError("Fixture transport failure"),
                         fake_response(content="Fixture answer."),
                         SimpleNamespace(choices=[])):
            with self.subTest(response=response):
                _reply, bodies, dispatch, _termination = await self.run_fake([response])
                self.assertEqual(len(bodies), 1)
                dispatch.assert_not_awaited()

    async def test_real_tool_on_retry_allows_the_following_summary(self):
        schema = {"type": "function", "function": {
            "name": "fixture_lookup", "description": "Read fixture data.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }}
        call = SimpleNamespace(id="fixture-call", type="function", function=SimpleNamespace(
            name="fixture_lookup", arguments="{}",
        ))
        reply, bodies, dispatch, _ = await self.run_fake([
            fake_response(), fake_response("tool_calls", tool_calls=[call]),
            fake_response(content="Fixture result answer."),
        ], schema=schema)
        self.assertEqual(len(bodies), 3)
        self.assertEqual(reply, "Fixture result answer.")
        self.assertEqual(bodies[1]["reasoning_effort"], "low")
        self.assertEqual(bodies[1]["max_tokens"], 1800)
        self.assertEqual(bodies[1]["tools"], [schema])
        self.assertEqual(bodies[1]["tool_choice"], "auto")
        self.assertEqual(bodies[2]["messages"][-1]["role"], "tool")
        self.assertEqual(bodies[2]["messages"][-1]["tool_call_id"], "fixture-call")
        dispatch.assert_awaited_once()

    async def test_routing_calls_are_charged_to_the_same_two_call_budget(self):
        for prior_calls in (0, 1, 2):
            with self.subTest(prior_calls=prior_calls):
                token = begin_turn_metrics("qq", "group")
                try:
                    for _ in range(prior_calls):
                        classifier = AsyncMock(return_value=fake_response(content='{"intent":"none"}'))
                        await observe_model_call(classifier())
                    reply, bodies, dispatch, termination = await self.run_fake([
                        fake_response(), fake_response(),
                    ])
                    self.assertEqual(current_turn_metrics().model_calls, 2)
                    self.assertEqual(len(bodies), 2 - prior_calls)
                    self.assertEqual(current_turn_metrics().tool_calls, 0)
                    self.assertEqual(termination["reason"], "no_tool_response_exhausted")
                    self.assertTrue(reply)
                    dispatch.assert_not_awaited()
                    if prior_calls == 1:
                        self.assertEqual(bodies[0], {
                            "model": "glm-5.3", "messages": [
                                {"role": "system", "content": _REASONING_RETRY_SYSTEM_PROMPT},
                                {"role": "user", "content": "[当前请求] Explain the fixture."},
                            ],
                            "max_tokens": 1800, "temperature": 0.2,
                            "extra_body": {"thinking": {"type": "enabled"}},
                            "reasoning_effort": "low",
                        })
                finally:
                    end_turn_metrics(token)

    async def test_previous_tools_keep_the_existing_continuation_budget(self):
        token = begin_turn_metrics("qq", "group")
        try:
            current_turn_metrics().model_calls = 2
            current_turn_metrics().tool_calls = 1
            reply, bodies, dispatch, _ = await self.run_fake([
                fake_response("length"), fake_response("length"),
                fake_response(content="Fixture answer."),
            ])
            self.assertEqual(len(bodies), 3)
            self.assertEqual(reply, "Fixture answer.")
            self.assertEqual(bodies[0]["reasoning_effort"], "high")
            dispatch.assert_not_awaited()
        finally:
            end_turn_metrics(token)

    async def test_invalid_tool_retry_cannot_spend_a_third_tool_free_call(self):
        schema = {"type": "function", "function": {
            "name": "fixture_lookup", "description": "Read fixture data.",
            "parameters": {"type": "object", "properties": {"word": {"type": "string"}},
                           "required": ["word"]},
        }}
        call = SimpleNamespace(id="fixture-call", type="function", function=SimpleNamespace(
            name="fixture_lookup", arguments='{"word":',
        ))
        _reply, bodies, dispatch, termination = await self.run_fake([
            fake_response(), fake_response("tool_calls", tool_calls=[call]),
        ], schema=schema)
        self.assertEqual(len(bodies), 2)
        self.assertEqual(termination["reason"], "no_tool_response_exhausted")
        dispatch.assert_not_awaited()

    async def test_locally_blocked_tools_do_not_unlock_the_zero_tool_budget(self):
        schema = {"type": "function", "function": {
            "name": "keytao_prepare_reviewed_add", "description": "Review a word.",
            "parameters": {"type": "object", "properties": {"word": {"type": "string"}},
                           "required": ["word"]},
        }}
        call = SimpleNamespace(id="fixture-blocked", type="function", function=SimpleNamespace(
            name="keytao_prepare_reviewed_add", arguments='{"word":"???"}',
        ))
        for tracked in (False, True):
            with self.subTest(tracked=tracked):
                token = begin_turn_metrics("qq", "group") if tracked else None
                try:
                    _reply, bodies, dispatch, termination = await self.run_fake([
                        fake_response("tool_calls", tool_calls=[call]),
                        fake_response(), fake_response(content="Unexpected third answer."),
                    ], schema=schema)
                    self.assertEqual(len(bodies), 2)
                    self.assertEqual(bodies[1]["reasoning_effort"], "low")
                    self.assertEqual(bodies[1]["max_tokens"], 1800)
                    self.assertEqual(bodies[1]["messages"][0], {
                        "role": "system", "content": _REASONING_RETRY_SYSTEM_PROMPT,
                    })
                    self.assertNotIn("old context", str(bodies[1]["messages"]))
                    self.assertEqual(termination["reason"], "no_tool_response_exhausted")
                    dispatch.assert_not_awaited()
                    if tracked:
                        self.assertEqual(current_turn_metrics().model_calls, 2)
                        self.assertEqual(current_turn_metrics().tool_calls, 0)
                finally:
                    if token is not None:
                        end_turn_metrics(token)

    async def test_first_invalid_tool_arguments_use_the_same_compacted_retry(self):
        schema = {"type": "function", "function": {
            "name": "fixture_lookup", "description": "Read fixture data.",
            "parameters": {"type": "object", "properties": {"word": {"type": "string"}},
                           "required": ["word"]},
        }}
        call = SimpleNamespace(id="fixture-malformed", type="function", function=SimpleNamespace(
            name="fixture_lookup", arguments='{"word":',
        ))
        reply, bodies, dispatch, _termination = await self.run_fake([
            fake_response("tool_calls", tool_calls=[call]),
            fake_response(content="Fixture answer."),
        ], schema=schema)
        self.assert_retry_shape([
            {key: value for key, value in body.items() if key not in {"tools", "tool_choice"}}
            for body in bodies
        ], read_only_tools=True)
        self.assertEqual(bodies[1]["tools"], [schema])
        self.assertEqual(bodies[1]["tool_choice"], "auto")
        self.assertEqual(reply, "Fixture answer.")
        dispatch.assert_not_awaited()


class GuidanceTests(unittest.TestCase):
    def test_read_only_guidance_uses_plain_chinese_for_command_policy(self):
        for token in ("suggestedCommand", "blockReason", "boundTarget",
                      "policyBlocked", "requiresTextFollowUp"):
            self.assertNotIn(token, READ_ONLY_TURN_GUIDANCE)
            self.assertNotIn(token, _REASONING_RETRY_SYSTEM_PROMPT)
        self.assertIn("可执行命令", READ_ONLY_TURN_GUIDANCE)


if __name__ == "__main__":
    unittest.main()
