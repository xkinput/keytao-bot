"""S59 offline read-only schema and model-visible result regressions."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import nonebot

with patch("nonebot.config.dotenv_values", return_value={}):
    nonebot.init(openai_api_key="fixture-key", openai_model="glm-5.3")

from keytao_bot.harness.orchestrator import (
    AgentOrchestrator, AgentRequestContext, AgentRuntimeConfig,
)
from keytao_bot.harness.state import MemoryConversationStateStore
from keytao_bot.harness.tools import (
    MUTATING_TOOL_NAMES, ToolContext, ToolExecutor, project_tool_result_for_model,
)
from keytao_bot.skills import SkillsManager


FORBIDDEN = (
    "suggestedCommand", "blockReason", "boundTarget", "policyBlocked",
    "requiresTextFollowUp",
)
TOOL_NAME = "keytao_word_commonness"
ROOT = Path(__file__).parent


def lookup_skills():
    manager = SkillsManager()
    manager.load_skill(ROOT / "keytao_bot/skills/keytao-lookup")
    return manager


class CommonnessToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_is_bounded_and_rejects_before_read_only_dispatch(self):
        manager = lookup_skills()
        schema = manager.get_tool_schema(TOOL_NAME)
        self.assertIsNotNone(schema)
        self.assertNotIn(TOOL_NAME, MUTATING_TOOL_NAMES)
        parameters = schema["function"]["parameters"]
        self.assertEqual(parameters["required"], ["words"])
        self.assertFalse(parameters["additionalProperties"])
        self.assertEqual(parameters["properties"]["words"]["minItems"], 1)
        self.assertEqual(parameters["properties"]["words"]["maxItems"], 12)
        dispatch = Mock(side_effect=AssertionError("Invalid input must not dispatch"))
        executor = ToolExecutor(dispatch, frozenset(), get_tool_schema=manager.get_tool_schema)
        for arguments in ({}, {"words": []}, {"words": ["word"] * 13},
                          {"words": [""]}, {"words": [7]}, {"words": "word"}):
            with self.subTest(arguments=arguments):
                result = json.loads(await executor.call(TOOL_NAME, arguments, ToolContext()))
                self.assertTrue(result.get("invalidArguments"), result)
        dispatch.assert_not_called()

    async def test_function_rejects_invalid_input_without_schema_or_reference_reads(self):
        manager = lookup_skills()
        executor = ToolExecutor(manager.get_tool_function, frozenset())
        with patch("keytao_bot.utils.word_commonness.review._query_commonness_reference",
                   side_effect=AssertionError("Invalid input must not read evidence")) as reference:
            for arguments in ({"words": []}, {"words": ["word"] * 13},
                              {"words": [" "]}, {"words": ["word", " word "]},
                              {"words": ["word"], "confirmed": True}):
                with self.subTest(arguments=arguments):
                    result = json.loads(await executor.call(TOOL_NAME, arguments, ToolContext()))
                    self.assertIn(result.get("errorType"), {"ValueError", "TypeError"}, result)
        reference.assert_not_called()

    async def test_fake_model_requests_registered_lookup_and_receives_exact_local_result(self):
        manager = lookup_skills()
        self.assertIsNotNone(manager.get_tool_function(TOOL_NAME))
        words = ["耶博", "伊莎贝拉", "夜泊", "一身本领"]
        payload = {
            "success": True, "method": "offline_reference", "referenceAvailable": True,
            "words": [{"word": word, "known": index < 3,
                       "corpusFrequency": 30 - index if index < 3 else None,
                       "dictionaryPresenceCount": 3 if index < 3 else None,
                       "rank": index + 1 if index < 3 else None,
                       "verdict": "ranked" if index < 3 else "unknown"}
                      for index, word in enumerate(words)],
            "comparisons": [], "ordering": words,
        }
        requests = []

        async def create(**request):
            requests.append(copy.deepcopy(request))
            tool_calls = []
            if len(requests) == 1:
                tool_calls = [SimpleNamespace(id="s59-commonness", type="function",
                    function=SimpleNamespace(name=TOOL_NAME, arguments=json.dumps({"words": words})))]
            return SimpleNamespace(usage=None, model="glm-5.3", choices=[SimpleNamespace(
                finish_reason="tool_calls" if tool_calls else "stop",
                message=SimpleNamespace(content="" if tool_calls else "夜泊有语料记录。",
                                        tool_calls=tool_calls, reasoning_content=None),
            )])

        executor = ToolExecutor(manager.get_tool_function, frozenset(),
                                get_tool_schema=manager.get_tool_schema)
        orchestrator = AgentOrchestrator(
            client_factory=lambda: SimpleNamespace(chat=SimpleNamespace(
                completions=SimpleNamespace(create=create))),
            runtime=AgentRuntimeConfig(model="glm-5.3", max_tokens=1800,
                temperature=0.2, timeout=10.0, max_tokens_cap=7200),
            skills_manager=manager, tool_executor=executor,
            state_store=MemoryConversationStateStore(), bind_help_text="fixture",
            system_prompt_core="Use the local evidence tool for the named words.",
        )
        with patch("keytao_bot.utils.word_commonness.lookup_word_commonness",
                   return_value=copy.deepcopy(payload)) as lookup:
            reply = await orchestrator._run_loop(
                "这几个名字平时有人用吗：耶博、伊莎贝拉、夜泊、一身本领",
                AgentRequestContext(platform="qq", user_id="s59-commonness-tool", history=[]),
            )
        self.assertEqual(reply, "夜泊有语料记录。")
        lookup.assert_called_once_with(words)
        self.assertEqual(len(requests), 2)
        self.assertIn(manager.get_tool_schema(TOOL_NAME), requests[0]["tools"])
        tool_messages = [message for message in requests[1]["messages"] if message["role"] == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0]["name"], TOOL_NAME)
        self.assertEqual(tool_messages[0]["tool_call_id"], "s59-commonness")
        self.assertEqual(json.loads(tool_messages[0]["content"]), payload)


class ModelCopyTests(unittest.TestCase):
    def test_every_result_path_hides_internal_tokens_without_mutating_executor_result(self):
        payload = {
            "success": False, "policyBlocked": True, "blockReason": "binding_incomplete",
            "requiresTextFollowUp": True, "suggestedCommand": "把 夜泊 排到 耶博 前面",
            "missing": ["boundTarget"],
            "nested": [{"message": "缺少 boundTarget，不得虚构 suggestedCommand"}],
        }
        original = copy.deepcopy(payload)
        for tool_name in ("keytao_encode", "keytao_create_phrase", TOOL_NAME, "unregistered_fixture"):
            with self.subTest(tool_name=tool_name):
                source = json.dumps(payload, ensure_ascii=False)
                result = project_tool_result_for_model(tool_name, source)
                for token in FORBIDDEN:
                    self.assertNotIn(token, result)
                projected = json.loads(result)
                self.assertEqual(projected["可执行命令"], payload["suggestedCommand"])
                self.assertTrue(projected["本次操作已拒绝"])
                self.assertEqual(payload, original)
                self.assertEqual(json.loads(source), original)

    def test_success_nonobject_and_nonjson_copies_have_the_same_boundary(self):
        for source in (
            json.dumps({"success": True, "policyBlocked": False, "message": "requiresTextFollowUp"}),
            json.dumps([{"boundTarget": "夜泊", "blockReason": "suggestedCommand"}]),
            "suggestedCommand / blockReason / boundTarget / policyBlocked / requiresTextFollowUp",
        ):
            with self.subTest(source=source):
                projected = project_tool_result_for_model("fixture", source)
                for token in FORBIDDEN:
                    self.assertNotIn(token, projected)

    def test_runtime_skill_instructions_and_descriptions_do_not_teach_internal_tokens(self):
        manager = SkillsManager()
        manager.load_all_skills()
        rendered = manager.get_skill_instructions() + json.dumps(manager.get_tools(), ensure_ascii=False)
        for token in FORBIDDEN:
            self.assertNotIn(token, rendered)


if __name__ == "__main__":
    unittest.main()
