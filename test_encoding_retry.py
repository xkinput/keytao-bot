#!/usr/bin/env python3
"""Focused regression tests for resilient KeyTao encoding requests."""

import sys
import types
import unittest
from unittest.mock import AsyncMock, patch

import httpx


_fake_nonebot = types.ModuleType("nonebot")


class _FakeConfig:
    keytao_api_base = "https://fake"
    bot_api_token = "fake"


class _FakeDriver:
    config = _FakeConfig()


_fake_nonebot.get_driver = lambda: _FakeDriver()
sys.modules["nonebot"] = _fake_nonebot

_fake_log = types.ModuleType("nonebot.log")
_fake_log.logger = types.SimpleNamespace(
    debug=lambda *args, **kwargs: None,
    info=lambda *args, **kwargs: None,
    warning=lambda *args, **kwargs: None,
    error=lambda *args, **kwargs: None,
)
sys.modules["nonebot.log"] = _fake_log

from keytao_bot.utils import keytao_review
from keytao_bot.utils.keytao_review import ReviewHttpConfig


CONFIG = ReviewHttpConfig(api_base="https://fake", bot_token="fake")


class _EncodingResponse:
    status_code = 200
    is_success = True

    def json(self):
        return {
            "input": "唐扬",
            "codes": ["tpyp", "tpypo", "tpypoi"],
            "chars": [
                {
                    "char": "唐",
                    "pinyin": "táng",
                    "pinyins": ["táng"],
                    "phoneticCode": "tp",
                    "shapeCode": "ovav",
                },
                {
                    "char": "扬",
                    "pinyin": "yáng",
                    "pinyins": ["yáng"],
                    "phoneticCode": "yp",
                    "shapeCode": "iuau",
                },
            ],
        }


class _EncodingService:
    def __init__(self, *, always_timeout=False):
        self.always_timeout = always_timeout
        self.attempts = 0
        self.timeouts = []

    async def request(self, method, url, **kwargs):
        self.attempts += 1
        self.timeouts.append(float(kwargs["timeout"]))
        if self.always_timeout or self.attempts == 1 or kwargs["timeout"] < 25.0:
            raise httpx.ReadTimeout(
                "" if self.always_timeout else "simulated slow encoder",
                request=httpx.Request(method, url),
            )
        return _EncodingResponse()


class EncodingRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_reviewed_add_recovers_when_slow_encoder_succeeds_on_retry(self):
        service = _EncodingService()

        with patch.object(
            keytao_review.http_client,
            "get_keytao_client",
            AsyncMock(return_value=service),
        ), patch.object(keytao_review.http_client.asyncio, "sleep", return_value=None):
            result = await keytao_review.fetch_keytao_encode(CONFIG, "唐扬")

        self.assertEqual(result.get("codes"), ["tpyp", "tpypo", "tpypoi"])
        self.assertEqual(service.attempts, 2)
        self.assertTrue(all(timeout >= 25.0 for timeout in service.timeouts))

    async def test_reviewed_add_reports_bounded_retry_failure_clearly(self):
        service = _EncodingService(always_timeout=True)

        with patch.object(
            keytao_review.http_client,
            "get_keytao_client",
            AsyncMock(return_value=service),
        ), patch.object(keytao_review.http_client.asyncio, "sleep", return_value=None):
            result = await keytao_review.fetch_keytao_encode(CONFIG, "唐扬")

        self.assertEqual(service.attempts, 2)
        self.assertIn("重试后仍不可用", result.get("message", ""))
        self.assertIn("ReadTimeout", result.get("message", ""))


if __name__ == "__main__":
    unittest.main()
