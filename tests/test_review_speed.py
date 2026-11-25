"""Offline regression tests for bounded batch review."""

import asyncio
import importlib.util
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

import httpx
import nonebot
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai import RateLimitError

from keytao_bot.utils import keytao_batch_review as batch_review
from keytao_bot.utils import keytao_review as review


CONFIG = {
    "api_key": "fake",
    "base_url": "https://example.invalid",
    "model": "deepseek-v4-flash",
    "timeout": 180.0,
    "quick_timeout": 30.0,
    "temperature": 0.2,
    "max_tokens": 2500,
    "max_tokens_cap": 12000,
}


def draft_item(item_id, **fields):
    return {"id": item_id, "word": f"词语{item_id}", "code": f"{chr(96 + item_id)}abc", **fields}


def valid_payload(items):
    return {"items": [{
        "prId": item["id"], "status": "pass", "title": "已复审",
        "reasons": ["读音相符"], "suggestions": ["保留编码"], "evidence": ["候选链相符"],
    } for item in items]}


def completion(payload, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason, message=SimpleNamespace(content=json.dumps(payload)),
    )])


def rate_limit(retry_after):
    request = httpx.Request("POST", CONFIG["base_url"])
    response = httpx.Response(429, request=request, headers={"Retry-After": str(retry_after)})
    return RateLimitError("rate limited", response=response, body=None)


class ReviewSpeedTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enterContext(patch.dict(review._llm_clients, clear=True))
        self.enterContext(patch.object(batch_review, "_llm_config", return_value=CONFIG))
        self.enterContext(patch.object(batch_review, "item_commonness", return_value={}))
        self.create = AsyncMock()
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        self.with_options = Mock(return_value=client)
        self.factory = Mock(return_value=SimpleNamespace(with_options=self.with_options))
        self.enterContext(patch.object(batch_review, "AsyncOpenAI", self.factory))
        self.enterContext(patch.object(review, "AsyncOpenAI", self.factory))

    async def call_review(self, **kwargs):
        return await batch_review._call_llm({}, [draft_item(1)], {}, None, None, **kwargs)

    async def test_timeout_is_per_operation_in_either_creation_order(self):
        self.create.return_value = completion(valid_payload([draft_item(1)]))
        with (
            patch.object(review, "_review_llm_config", return_value=CONFIG),
            patch.object(review, "_cache_get", return_value=None),
            patch.object(review, "_cache_set", side_effect=lambda word, kind, data: data),
        ):
            for quick_first in (True, False):
                with self.subTest(quick_first=quick_first):
                    review._llm_clients.clear()
                    self.factory.reset_mock()
                    self.with_options.reset_mock()
                    if quick_first:
                        await review._infer_entity_knowledge("甲乙")
                        await self.call_review()
                    else:
                        await self.call_review()
                        await review._infer_entity_knowledge("甲乙")
                    self.factory.assert_called_once()
                    expected = [call(timeout=30.0), call(timeout=180.0, max_retries=0)]
                    self.assertEqual(expected if quick_first else expected[::-1], self.with_options.call_args_list)

    async def test_rate_limit_waits_and_only_two_retries_are_free(self):
        self.create.side_effect = [rate_limit(delay) for delay in (17, 1, 120, 10, 25)]
        with patch.object(batch_review.asyncio, "sleep", new_callable=AsyncMock) as sleep:
            with self.assertRaisesRegex(RuntimeError, "rate limited"):
                await self.call_review()
        self.assertEqual([call(17.0), call(5.0), call(60.0), call(10.0)], sleep.await_args_list)
        self.assertEqual(5, self.create.await_count)
        self.assertEqual(6000, self.create.await_args_list[0].kwargs["max_tokens"])

    async def test_length_then_transport_retry_grows_budget_and_recomputes_timeout(self):
        self.create.side_effect = [
            completion({}, "length"), TimeoutError("fake timeout"), completion(valid_payload([draft_item(1)])),
        ]
        with (
            patch.object(batch_review, "_llm_config", return_value={**CONFIG, "review_min_max_tokens": 3000}),
            patch.object(batch_review, "_remaining_review_time", side_effect=[200.0, 90.0, 45.0]),
            patch.object(batch_review.asyncio, "sleep", new_callable=AsyncMock) as sleep,
        ):
            await self.call_review(deadline=1.0)
        self.assertEqual([3000, 6000, 12000], [args.kwargs["max_tokens"] for args in self.create.await_args_list])
        self.assertEqual([call(timeout=value, max_retries=0) for value in (180.0, 90.0, 45.0)],
                         self.with_options.call_args_list)
        sleep.assert_awaited_once_with(1.0)

    async def test_sealed_only_batch_never_starts_model_chunks(self):
        items = [draft_item(1, needsManualReview=True, manualReviewReason="证据待确认"), draft_item(2)]
        audit = {"sealedItemIds": [2], "sealedItemReasons": {2: "编码冲突"}}
        with (
            patch.object(batch_review, "audit_draft_items", AsyncMock(return_value=audit)) as audited,
            patch.object(batch_review, "_call_llm_chunked", new_callable=AsyncMock) as chunked,
        ):
            result = await batch_review.review_keytao_batch_with_llm({"pullRequests": items})
        self.assertEqual([1, 2], [item["id"] for item in audited.await_args.args[1]])
        chunked.assert_not_awaited()
        self.create.assert_not_awaited()
        self.assertEqual("manual_review", result["aiReview"]["verdict"])
        self.assertEqual(
            [("manual_review", "high", "证据待确认"), ("manual_review", "high", "编码冲突")],
            [(item["status"], item["priority"], item["priorityReason"]) for item in result["aiReview"]["items"]],
        )

    async def test_mixed_batch_sends_only_open_items_and_returns_every_item(self):
        items = [draft_item(1, needsManualReview=True), draft_item(2), draft_item(3), draft_item(4, word="词语3")]
        issue = "「词语3」@cabc 读音来源待确认"
        audit = {"sealedItemIds": [2], "structuredManualReviewIssues": [issue]}
        self.create.return_value = completion(valid_payload([items[3]]))
        result = await batch_review.review_keytao_batch_with_llm(
            {"pullRequests": items}, focus_pr_id=3, precomputed_audit=audit, items_only=True,
        )
        self.create.assert_awaited_once()
        payload = json.loads(self.create.await_args.kwargs["messages"][1]["content"])
        self.assertEqual([4], [item["id"] for item in payload["pullRequests"]])
        self.assertIsNone(payload["focusPrId"])
        self.assertEqual([
            (1, "manual_review"), (2, "manual_review"), (3, "manual_review"), (4, "pass"),
        ], [(item["prId"], item["status"]) for item in result["aiReview"]["items"]])
        self.assertEqual(issue, result["aiReview"]["items"][2]["priorityReason"])

    async def test_deadline_preserves_finished_chunks_and_cancels_the_rest(self):
        items = [draft_item(item_id) for item_id in range(1, 7)]
        cancelled = asyncio.Event()

        async def review_chunk(_batch, chunk, *_args, **_kwargs):
            if chunk[0]["id"] == 1:
                return valid_payload(chunk)
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with (
            patch.object(batch_review, "_review_chunk_size", return_value=2),
            patch.object(batch_review, "_review_chunk_concurrency", return_value=1),
            patch.object(batch_review, "_call_llm", side_effect=review_chunk) as llm,
            patch.object(batch_review, "_REVIEW_DEADLINE_MARGIN_SECONDS", 9.9),
        ):
            result = await batch_review.review_keytao_batch_with_llm(
                {"pullRequests": items}, precomputed_audit={}, items_only=True, deadline_ms=10_000,
            )
        self.assertTrue(cancelled.is_set())
        self.assertEqual(2, llm.await_count)
        self.assertEqual(["pass", "pass", *["attention"] * 4],
                         [item["status"] for item in result["aiReview"]["items"]])
        self.assertIn("时限", result["warning"])

    async def test_deadline_cancels_audit_without_trying_the_model_or_remote_fallback(self):
        cancelled = asyncio.Event()

        async def audit(*_args):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        with (
            patch.object(batch_review, "audit_draft_items", side_effect=audit),
            patch.object(batch_review, "_fallback_audit_with_encode", new_callable=AsyncMock) as fallback,
            patch.object(batch_review, "_REVIEW_DEADLINE_MARGIN_SECONDS", 9.9),
        ):
            result = await batch_review.review_keytao_batch_with_llm(
                {"pullRequests": [draft_item(1)]}, deadline_ms=10_000,
            )
        self.assertTrue(cancelled.is_set())
        fallback.assert_not_awaited()
        self.create.assert_not_awaited()
        self.assertEqual("attention", result["aiReview"]["items"][0]["status"])

    async def test_disconnect_stops_queued_chunks(self):
        items = [draft_item(item_id) for item_id in range(1, 7)]
        self.create.return_value = completion(valid_payload(items[:2]))
        is_disconnected = AsyncMock(side_effect=[False, True])
        with (
            patch.object(batch_review, "_review_chunk_size", return_value=2),
            patch.object(batch_review, "_review_chunk_concurrency", return_value=1),
        ):
            result = await batch_review.review_keytao_batch_with_llm(
                {"pullRequests": items}, precomputed_audit={}, is_disconnected=is_disconnected,
            )
        self.create.assert_awaited_once()
        self.assertEqual(2, is_disconnected.await_count)
        self.assertEqual(["pass", "pass", *["attention"] * 4],
                         [item["status"] for item in result["aiReview"]["items"]])
        self.assertIn("断开连接", result["warning"])


class ReviewDeadlineEndpointTests(unittest.TestCase):
    def test_deadline_validation_defaults_and_disconnect_callback(self):
        app = FastAPI()
        spec = importlib.util.find_spec("keytao_bot.plugins.web_api")
        web_api = importlib.util.module_from_spec(spec)
        driver = SimpleNamespace(config=SimpleNamespace(web_api_key="test-key"))
        with (
            patch.object(nonebot, "get_driver", return_value=driver),
            patch.object(nonebot, "get_app", return_value=app, create=True),
            patch.dict("sys.modules", {"keytao_bot.plugins.openai_chat": Mock()}),
        ):
            spec.loader.exec_module(web_api)
        path = "/api/keytao/batches/review"
        auth = {"Authorization": "Bearer test-key"}
        run_review = AsyncMock(return_value={"success": True})
        with TestClient(app) as client, patch.object(web_api, "review_keytao_batch_with_llm", run_review):
            for invalid in (9999, 290001, "15000", 15000.0, True):
                self.assertEqual(422, client.post(path, headers=auth, json={"batch": {}, "deadline_ms": invalid}).status_code)
            run_review.assert_not_awaited()
            for headers, deadline, expected in (
                (auth, None, 170_000),
                ({**auth, "X-KeyTao-Review-Output": "items"}, None, 280_000),
                (auth, 10_000, 10_000),
                (auth, 290_000, 290_000),
            ):
                body = {"batch": {}} if deadline is None else {"batch": {}, "deadline_ms": deadline}
                self.assertEqual(200, client.post(path, headers=headers, json=body).status_code)
                kwargs = run_review.await_args.kwargs
                self.assertEqual(expected, kwargs["deadline_ms"])
                self.assertIs(web_api.Request.is_disconnected, kwargs["is_disconnected"].__func__)
