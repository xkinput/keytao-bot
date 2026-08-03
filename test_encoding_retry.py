#!/usr/bin/env python3
"""Focused regression tests for resilient KeyTao encoding requests."""

import sys
import types
import unittest
from unittest.mock import patch

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

    def client_factory(self, *, timeout):
        service = self

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def get(self, url, **kwargs):
                service.attempts += 1
                service.timeouts.append(float(timeout))
                if service.always_timeout or service.attempts == 1 or timeout < 25.0:
                    raise httpx.ReadTimeout(
                        "" if service.always_timeout else "simulated slow encoder",
                        request=httpx.Request("GET", url),
                    )
                return _EncodingResponse()

        return _Client()


class EncodingRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_reviewed_add_recovers_when_slow_encoder_succeeds_on_retry(self):
        service = _EncodingService()

        with patch.object(
            keytao_review.httpx,
            "AsyncClient",
            side_effect=service.client_factory,
        ), patch.object(keytao_review.asyncio, "sleep", return_value=None):
            result = await keytao_review.fetch_keytao_encode(CONFIG, "唐扬")

        self.assertEqual(result.get("codes"), ["tpyp", "tpypo", "tpypoi"])
        self.assertEqual(service.attempts, 2)
        self.assertTrue(all(timeout >= 25.0 for timeout in service.timeouts))

    async def test_reviewed_add_reports_bounded_retry_failure_clearly(self):
        service = _EncodingService(always_timeout=True)

        with patch.object(
            keytao_review.httpx,
            "AsyncClient",
            side_effect=service.client_factory,
        ), patch.object(keytao_review.asyncio, "sleep", return_value=None):
            result = await keytao_review.fetch_keytao_encode(CONFIG, "唐扬")

        self.assertEqual(service.attempts, 2)
        self.assertIn("重试后仍不可用", result.get("message", ""))
        self.assertIn("ReadTimeout", result.get("message", ""))


if __name__ == "__main__":
    unittest.main()
