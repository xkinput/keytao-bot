"""Offline S63 regressions for chat replies and truthful policy refusals."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import test_state_machine as harness
from keytao_bot.harness.authorization_grammar import message_authorizes_mutation
from keytao_bot.utils.pending_confirmation import advertised_command_suggestions


NO_PLAN = "这次调整还没有可确认的计划，本次未写入。"


def policy_agent(calls, final_text="Fixture response"):
    client = harness._FakeClient([
        harness._FakeAIResponse("tool_calls", "", calls),
        harness._FakeAIResponse("stop", final_text),
    ])
    properties = {
        "word": {"type": "string"}, "code": {"type": "string"},
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "word": {"type": "string"}, "code": {"type": "string"}, "action": {"type": "string"},
        }}},
    }
    schemas = [{"type": "function", "function": {
        "name": name, "description": "Fixture", "parameters": {"type": "object", "properties": properties},
    }} for name in {item.function.name for item in calls}]
    manager = SimpleNamespace(get_skill_instructions=lambda: "", has_tools=lambda: True, get_tools=lambda: schemas)

    async def forbidden(**_kwargs):
        raise AssertionError("No real sink may execute")

    agent = harness.AgentOrchestrator(
        client_factory=lambda: client,
        runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10),
        skills_manager=manager,
        tool_executor=harness.ToolExecutor(lambda _name: forbidden, frozenset()),
        state_store=harness.MemoryConversationStateStore(), bind_help_text="fixture", system_prompt_core="fixture",
    )
    return agent, client


def policy_call(name, args, index):
    return SimpleNamespace(id=f"s63-policy-{index}", type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(args),
    ))


class GeneralReplyTests(unittest.IsolatedAsyncioTestCase):
    async def reply(self, message, raw, pending=None):
        client = harness._FakeClient([harness._FakeAIResponse("stop", raw)])
        store = harness.MemoryConversationStateStore()
        context = harness.AgentRequestContext(
            platform="qq", user_id="s63-general",
            mutations_allowed=message_authorizes_mutation(message),
        )
        if pending is not None:
            store.set(context.conversation_address, pending)

        def forbidden_tool(_name):
            raise AssertionError("A text-only reply must not dispatch tools")

        orchestrator = harness.AgentOrchestrator(
            client_factory=lambda: client,
            runtime=harness.AgentRuntimeConfig(
                model="fake-model", max_tokens=1000, temperature=0.0, timeout=10.0,
            ),
            skills_manager=harness._FakeSkillsManager(),
            tool_executor=harness.ToolExecutor(forbidden_tool, frozenset()),
            state_store=store, bind_help_text="fixture", system_prompt_core="fixture",
        )
        response = await orchestrator.run(message, context)
        self.assertEqual(len(client.completions.calls), 1)
        return response

    async def test_chitchat_does_not_claim_an_adjustment(self):
        response = await self.reply("ennnn", "可以回复「嗯嗯」跟我聊天。")
        self.assertNotIn("调整", response)
        self.assertNotIn("未写入", response)
        self.assertFalse(advertised_command_suggestions(response))

    async def test_code_question_does_not_claim_an_adjustment(self):
        response = await self.reply("ennnn 是什么编码", "想查这个编码的话，可以回复「查询 ennnn」。")
        self.assertNotIn("调整", response)
        self.assertNotIn("未写入", response)
        self.assertFalse(advertised_command_suggestions(response))

    async def test_explicit_mutation_still_requires_a_real_plan(self):
        response = await self.reply("把 奈飞 调到 nhfw", "请回复「确认」。")
        self.assertEqual(response, NO_PLAN)

    async def test_unrecognized_mutation_verb_still_requires_a_real_plan(self):
        response = await self.reply("把 奈飞 迁到 nhfw", "请回复「确认」。")
        self.assertEqual(response, NO_PLAN)

    async def test_surrounding_read_only_text_is_preserved(self):
        response = await self.reply("ennnn", "你好。可以回复「嗯嗯」跟我聊天。")
        self.assertEqual(response, "你好。")

    async def test_real_confirmation_ticket_is_preserved(self):
        state = harness.PendingToolConfirm(
            "keytao_shift_phrase_code",
            {"word": "奈飞", "target_code": "nhfw", "batch_id": "",
             "expected_content_version": 0, "confirmed_plan_digest": "a" * 64},
            confirmation_source="server_warning",
        )
        response = await self.reply("把 奈飞 调到 nhfw", "请回复「确认」。", state)
        self.assertEqual(response, "请回复「确认」。")

    async def test_encode_then_rejected_create_does_not_invent_a_service_failure(self):
        def call(name, arguments):
            return SimpleNamespace(id="s63-" + name, type="function", function=SimpleNamespace(
                name=name, arguments=json.dumps(arguments),
            ))

        client = harness._FakeClient([
            harness._FakeAIResponse("tool_calls", "", [call("keytao_encode", {"word": "加亮", "requested_code": "jslxa"})]),
            harness._FakeAIResponse("tool_calls", "", [call("keytao_create_phrase", {"word": "加亮", "code": "jslxa"})]),
            harness._FakeAIResponse("stop", "尝试加入草稿时服务未完成处理，本次没有写入。"),
        ])
        dispatches = []

        async def encode(**_kwargs):
            dispatches.append("keytao_encode")
            return {"success": True, "word": "加亮", "input": "加亮", "recommendedCode": "jslxa",
                    "candidateCodes": ["jslxa"], "codes": ["jslxa"],
                    "candidateStatuses": [{"code": "jslxa", "occupied": False, "words": []}]}

        async def forbidden_create(**_kwargs):
            self.fail("The rejected create must not reach its sink")

        schemas = [{"type": "function", "function": {
            "name": name, "description": "Fixture operation", "parameters": {
                "type": "object", "properties": {key: {"type": "string"} for key in keys},
                "required": ["word"] if name == "keytao_encode" else ["word", "code"],
            },
        }} for name, keys in (("keytao_encode", ("word", "requested_code")),
                             ("keytao_create_phrase", ("word", "code")))]
        manager = SimpleNamespace(get_skill_instructions=lambda: "", has_tools=lambda: True, get_tools=lambda: schemas)
        store = harness.MemoryConversationStateStore()
        context = harness.AgentRequestContext(platform="qq", user_id="s63-policy", mutations_allowed=False)
        orchestrator = harness.AgentOrchestrator(
            client_factory=lambda: client,
            runtime=harness.AgentRuntimeConfig(model="fake-model", max_tokens=1000, temperature=0.0, timeout=10.0),
            skills_manager=manager,
            tool_executor=harness.ToolExecutor(
                lambda name: encode if name == "keytao_encode" else forbidden_create, frozenset(),
            ),
            state_store=store, bind_help_text="fixture", system_prompt_core="fixture",
        )
        response = await orchestrator.run("加亮 jslxa", context)
        self.assertEqual(dispatches, ["keytao_encode"])
        self.assertIsNone(store.get_record(context.conversation_address))
        self.assertEqual(len(client.completions.calls), 2, "A local refusal must not request a fabricated explanation")
        self.assertIn("没有明确要求", response)
        self.assertNotIn("服务未完成", response)

    async def test_exclusion_binding_refusal_keeps_its_distinct_reason(self):
        args = {"items": [
            {"action": "Create", "word": "显眼包", "code": "xybo"},
            {"action": "Create", "word": "嘴替", "code": "zbtk"},
        ]}
        agent, _client = policy_agent(
            [policy_call("keytao_batch_add_to_draft", args, 1)],
            "「都加」附带的排除条件无法确认，本次未写入。",
        )
        response = await agent.run("都加 跳过绝绝子", harness.AgentRequestContext(
            platform="qq", user_id="s63-exclusion", mutations_allowed=False,
        ))
        self.assertIn("排除条件", response)
        self.assertNotIn("没有明确要求", response)

    async def test_prior_successful_write_survives_an_early_submit_refusal(self):
        agent, client = policy_agent([
            policy_call("keytao_create_phrase", {"word": "炒冷饭", "code": "wlfoo"}, 1),
            policy_call("keytao_submit_batch", {}, 2),
        ])
        success = {
            "success": True, "batchId": "s63-partial", "batchUrl": "https://example.invalid/batch/s63-partial",
            "writtenItems": [{"id": "s63-row", "action": "Create", "word": "炒冷饭", "code": "wlfoo", "type": "Phrase"}],
        }
        refusal = {"success": False, "policyBlocked": True, "blockReason": "verb_not_matched",
                   "missing": ["executionVerb"], "message": "当前消息没有明确要求提交该批次；本次未写入。"}
        with patch.object(agent, "_call_tool_once", AsyncMock(side_effect=[
            json.dumps(success), json.dumps(refusal),
        ])) as execute:
            response = await agent.run("添加 炒冷饭 wlfoo", harness.AgentRequestContext(
                platform="qq", user_id="s63-partial", mutations_allowed=True,
            ))
        self.assertEqual(execute.await_count, 2)
        self.assertEqual(len(client.completions.calls), 1)
        for text in ("本轮已完成的写操作", "炒冷饭", "wlfoo", "提交未完成", success["batchUrl"]):
            self.assertIn(text, response)
        self.assertNotIn("本次未写入", response)


if __name__ == "__main__":
    unittest.main()
