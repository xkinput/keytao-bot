"""Tests for DeepSeek-specific request policy and usage normalization."""

import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import OpenAI

_POLICY_PATH = Path(__file__).parent / "keytao_bot" / "utils" / "llm_policy.py"
_POLICY_SPEC = importlib.util.spec_from_file_location("llm_policy_for_test", _POLICY_PATH)
_POLICY_MODULE = importlib.util.module_from_spec(_POLICY_SPEC)
assert _POLICY_SPEC.loader is not None
_POLICY_SPEC.loader.exec_module(_POLICY_MODULE)

chat_usage_metrics = _POLICY_MODULE.chat_usage_metrics
log_chat_usage = _POLICY_MODULE.log_chat_usage
with_deepseek_chat_policy = _POLICY_MODULE.with_deepseek_chat_policy


class DeepSeekChatPolicyTests(unittest.TestCase):
    def test_enabled_thinking_sets_effort_and_removes_ignored_sampling(self):
        request = with_deepseek_chat_policy(
            {
                "model": "deepseek-v4-flash",
                "temperature": 0.7,
                "top_p": 0.9,
                "extra_body": {"trace_id": "keep-me"},
            },
            thinking=True,
            reasoning_effort="high",
        )

        self.assertEqual(request["reasoning_effort"], "high")
        self.assertEqual(request["extra_body"]["thinking"], {"type": "enabled"})
        self.assertEqual(request["extra_body"]["trace_id"], "keep-me")
        self.assertNotIn("temperature", request)
        self.assertNotIn("top_p", request)

    def test_disabled_thinking_keeps_sampling_and_enables_json_output(self):
        request = with_deepseek_chat_policy(
            {
                "model": "deepseek-v4-flash",
                "temperature": 0.0,
            },
            thinking=False,
            json_output=True,
        )

        self.assertEqual(request["temperature"], 0.0)
        self.assertEqual(request["extra_body"]["thinking"], {"type": "disabled"})
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertNotIn("reasoning_effort", request)

    def test_non_deepseek_request_is_unchanged(self):
        original = {"model": "gemini-2.0-flash", "temperature": 0.2}

        request = with_deepseek_chat_policy(
            original,
            thinking=False,
            json_output=True,
        )

        self.assertEqual(request, original)
        self.assertIsNot(request, original)

    def test_invalid_reasoning_effort_fails_fast(self):
        for effort in ("low", "medium"):
            with self.subTest(effort=effort):
                with self.assertRaisesRegex(ValueError, "Unsupported DeepSeek reasoning effort"):
                    with_deepseek_chat_policy(
                        {"model": "deepseek-v4-flash"},
                        thinking=True,
                        reasoning_effort=effort,
                    )

    def test_openai_sdk_serializes_deepseek_fields(self):
        captured = {}

        def handle_request(request):
            captured.update(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-v4-flash",
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "{}"},
                }],
            })

        request = with_deepseek_chat_policy(
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "Return JSON."}],
                "temperature": 0.7,
            },
            thinking=True,
            reasoning_effort="high",
            json_output=True,
        )
        with httpx.Client(transport=httpx.MockTransport(handle_request)) as http_client:
            with OpenAI(
                api_key="test-key",
                base_url="https://api.deepseek.com",
                http_client=http_client,
                max_retries=0,
            ) as client:
                client.chat.completions.create(**request)

        self.assertEqual(captured["thinking"], {"type": "enabled"})
        self.assertEqual(captured["reasoning_effort"], "high")
        self.assertEqual(captured["response_format"], {"type": "json_object"})
        self.assertNotIn("temperature", captured)


class UsageMetricsTests(unittest.TestCase):
    def test_normalizes_deepseek_chat_usage(self):
        response = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=40,
            total_tokens=160,
            prompt_cache_hit_tokens=80,
            prompt_cache_miss_tokens=40,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=25),
        ))

        self.assertEqual(chat_usage_metrics(response), {
            "input_tokens": 120,
            "output_tokens": 40,
            "total_tokens": 160,
            "reasoning_tokens": 25,
            "cached_tokens": 80,
            "cache_miss_tokens": 40,
        })

    def test_normalizes_responses_style_usage_and_computes_total(self):
        response = {
            "usage": {
                "input_tokens": 90,
                "output_tokens": 30,
                "input_tokens_details": {"cached_tokens": 50},
                "output_tokens_details": {"reasoning_tokens": 12},
            }
        }

        self.assertEqual(chat_usage_metrics(response), {
            "input_tokens": 90,
            "output_tokens": 30,
            "total_tokens": 120,
            "reasoning_tokens": 12,
            "cached_tokens": 50,
            "cache_miss_tokens": 40,
        })

    def test_missing_usage_returns_no_metrics(self):
        self.assertEqual(chat_usage_metrics(SimpleNamespace(usage=None)), {})

    def test_usage_logger_includes_operation_model_and_tokens(self):
        messages = []
        logger = SimpleNamespace(info=messages.append)
        response = SimpleNamespace(usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ))

        log_chat_usage(
            logger,
            response,
            operation="command_intent",
            model="deepseek-v4-flash",
        )

        self.assertEqual(len(messages), 1)
        self.assertIn("operation=command_intent", messages[0])
        self.assertIn("model=deepseek-v4-flash", messages[0])
        self.assertIn("total_tokens=15", messages[0])


if __name__ == "__main__":
    unittest.main()
