#!/usr/bin/env python3
"""Focused regression tests for resilient KeyTao encoding requests."""

import sys
import types
import unittest
from types import SimpleNamespace
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

from keytao_bot.utils import http_client, keytao_review
from keytao_bot.utils.keytao_review import ReviewHttpConfig
from keytao_bot.utils.observability import (
    begin_turn_metrics,
    emit_turn_metrics,
    end_turn_metrics,
)


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
        if self.always_timeout or kwargs["timeout"] < 20.0:
            raise httpx.ReadTimeout(
                "" if self.always_timeout else "simulated slow encoder",
                request=httpx.Request(method, url),
            )
        return _EncodingResponse()


class EncodingRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_retries_three_times_then_raises(self):
        attempts = 0

        async def request():
            nonlocal attempts
            attempts += 1
            raise httpx.ReadTimeout(
                "still slow",
                request=httpx.Request("GET", "https://fake/read"),
            )

        with patch.object(http_client.asyncio, "sleep", new=AsyncMock()) as sleep, \
             patch.object(http_client.logger, "info") as log_info:
            with self.assertRaises(httpx.ReadTimeout):
                await http_client.request_with_retries(
                    request,
                    method="GET",
                    url="/read",
                )

        self.assertEqual(attempts, 4)
        self.assertEqual(sleep.await_count, 3)
        self.assertEqual(log_info.call_count, 3)

    async def test_read_succeeds_on_attempt_three(self):
        attempts = 0
        expected = object()

        async def request():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise httpx.ReadTimeout(
                    "slow start",
                    request=httpx.Request("GET", "https://fake/read"),
                )
            return expected

        with patch.object(http_client.asyncio, "sleep", new=AsyncMock()) as sleep:
            result = await http_client.request_with_retries(
                request,
                method="GET",
                url="/read",
            )

        self.assertIs(result, expected)
        self.assertEqual(attempts, 3)
        self.assertEqual(sleep.await_count, 2)

    async def test_write_retries_connect_timeout_but_not_read_timeout(self):
        connect_attempts = 0
        expected = object()

        async def connect_timeout_then_success():
            nonlocal connect_attempts
            connect_attempts += 1
            if connect_attempts < 3:
                raise httpx.ConnectTimeout(
                    "connect stalled",
                    request=httpx.Request("POST", "https://fake/write"),
                )
            return expected

        with patch.object(http_client.asyncio, "sleep", new=AsyncMock()) as sleep:
            result = await http_client.request_with_retries(
                connect_timeout_then_success,
                method="POST",
                url="/write",
            )

        self.assertIs(result, expected)
        self.assertEqual(connect_attempts, 3)
        self.assertEqual(sleep.await_count, 2)

        read_attempts = 0

        async def read_timeout():
            nonlocal read_attempts
            read_attempts += 1
            raise httpx.ReadTimeout(
                "response stalled",
                request=httpx.Request("POST", "https://fake/write"),
            )

        with patch.object(http_client.asyncio, "sleep", new=AsyncMock()) as sleep:
            with self.assertRaises(httpx.ReadTimeout):
                await http_client.request_with_retries(
                    read_timeout,
                    method="POST",
                    url="/write",
                )

        self.assertEqual(read_attempts, 1)
        self.assertEqual(sleep.await_count, 0)

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
        self.assertEqual(service.timeouts, [10.0, 20.0])

    async def test_encode_retry_log_resolves_the_next_timeout_budget(self):
        service = _EncodingService()

        with patch.object(
            keytao_review.http_client,
            "get_keytao_client",
            AsyncMock(return_value=service),
        ), patch.object(
            keytao_review.http_client.asyncio,
            "sleep",
            return_value=None,
        ), patch.object(keytao_review.logger, "info") as log_info:
            result = await keytao_review.fetch_keytao_encode(CONFIG, "唐扬")

        self.assertEqual(result.get("codes"), ["tpyp", "tpypo", "tpypoi"])
        self.assertTrue(
            any(
                len(call.args) == 1
                and "retry 1/2" in str(call.args[0])
                and "next_timeout=20s" in str(call.args[0])
                for call in log_info.call_args_list
            ),
            log_info.call_args_list,
        )

    async def test_reviewed_add_reports_bounded_retry_failure_clearly(self):
        service = _EncodingService(always_timeout=True)

        def reference_readings(word):
            rows = {
                "钉": [
                    SimpleNamespace(
                        display="dīng", normalized=("ding",), dataset="cedict",
                    ),
                    SimpleNamespace(
                        display="dìng", normalized=("ding",), dataset="cedict",
                    ),
                ],
                "选": [
                    SimpleNamespace(
                        display="xuǎn", normalized=("xuan",), dataset="cedict",
                    ),
                ],
            }
            return rows.get(word, [])

        with patch.object(
            keytao_review.http_client,
            "get_keytao_client",
            AsyncMock(return_value=service),
        ), patch.object(
            keytao_review.http_client.asyncio,
            "sleep",
            return_value=None,
        ), patch.object(
            keytao_review,
            "query_reference_readings",
            side_effect=reference_readings,
        ), patch.object(
            keytao_review,
            "_query_commonness_reference",
            return_value={
                "available": True,
                "attested": True,
                "word": "钉选",
                "corpusFrequency": 12,
                "partOfSpeech": "n",
                "dictionaryPresenceCount": 0,
            },
        ):
            result = await keytao_review.fetch_keytao_encode(CONFIG, "钉选")

        self.assertEqual(service.attempts, 3)
        self.assertEqual(service.timeouts, [10.0, 20.0, 30.0])
        self.assertIn("重试后仍不可用", result.get("message", ""))
        self.assertIn("ReadTimeout", result.get("message", ""))
        self.assertTrue(result.get("upstreamTransient"))
        self.assertFalse(result.get("encodeServiceConfirmed"))
        offline = result.get("offlineReference", {})
        self.assertEqual(
            [item.get("pinyin") for item in offline.get("readings", [])],
            ["dīng xuǎn", "dìng xuǎn"],
        )
        self.assertEqual(
            [item.get("candidateCodes") for item in offline.get("readings", [])],
            [["dgxt"], ["dgxt"]],
        )
        self.assertEqual(offline.get("frequency", {}).get("corpusFrequency"), 12)

    async def test_encode_attempt_latencies_join_the_turn_metrics_line(self):
        service = _EncodingService()
        metrics_token = begin_turn_metrics("qq", "group")
        try:
            with patch.object(
                keytao_review.http_client,
                "get_keytao_client",
                AsyncMock(return_value=service),
            ), patch.object(
                keytao_review.http_client.asyncio,
                "sleep",
                return_value=None,
            ):
                result = await keytao_review.fetch_keytao_encode(CONFIG, "唐扬")
            metrics_line = emit_turn_metrics(types.SimpleNamespace(info=lambda _line: None))
        finally:
            end_turn_metrics(metrics_token)

        self.assertEqual(result.get("codes"), ["tpyp", "tpypo", "tpypoi"])
        self.assertIn("encode_calls=2", metrics_line or "")
        self.assertIn("encode_retry_calls=1", metrics_line or "")
        per_call = (metrics_line or "").split("encode_call_seconds=", 1)[1].split(" ", 1)[0]
        self.assertEqual(len(per_call.split(",")), 2)
        self.assertIn("encode_retry_flags=0,1", metrics_line or "")


if __name__ == "__main__":
    unittest.main()
