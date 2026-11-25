"""Offline lifecycle and retry tests for the automatic draft review poller."""

import asyncio
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
import nonebot


class FakeClock:
    def __init__(self, timezone):
        self.now = datetime(2026, 11, 25, 23, 58, tzinfo=timezone)
        self.delays = asyncio.Queue()
        self.resume = asyncio.Semaphore(0)

    async def sleep(self, delay):
        await self.delays.put(delay)
        await self.resume.acquire()
        self.now += timedelta(seconds=delay)


class AutoReviewSchedulerTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        driver = Mock(config=SimpleNamespace())
        driver.on_startup.side_effect = driver.on_shutdown.side_effect = lambda func: func
        with patch.object(nonebot, "get_driver", return_value=driver):
            from keytao_bot.plugins import auto_review_scheduler

        cls.scheduler = auto_review_scheduler

    async def asyncSetUp(self):
        scheduler = self.scheduler
        self.clock = FakeClock(scheduler.TIMEZONE)
        self.requests = []
        self.set_queue((41, "original"))
        self.review = self.successful_review
        client = await self.enterAsyncContext(
            httpx.AsyncClient(transport=httpx.MockTransport(self.handle_request))
        )
        directory = self.enterContext(TemporaryDirectory())
        self.db_path = Path(directory) / "data" / "reviews.sqlite"
        self.enterContext(patch.object(scheduler, "DB_PATH", self.db_path))
        self.enterContext(patch.object(scheduler, "_now", lambda: self.clock.now))
        self.enterContext(patch.object(scheduler, "sleep", self.clock.sleep))
        self.enterContext(patch.object(scheduler, "_disabled_logged", False))
        self.enterContext(patch.object(scheduler.http_client, "get_keytao_client", return_value=client))
        self.logger = self.enterContext(patch.object(scheduler, "logger"))
        self.enterContext(patch.object(nonebot, "get_driver", return_value=Mock(config=SimpleNamespace())))
        self.enterContext(patch.dict(os.environ, {
            "KEYTAO_API_BASE": "https://keytao.invalid",
            "BOT_API_TOKEN": "fake-token",
        }))
        os.environ.pop("KEYTAO_AUTO_REVIEW_DAILY_CHUNKS", None)
        os.environ.pop("KEYTAO_AUTO_REVIEW_INTERVAL_SECONDS", None)
        self.addAsyncCleanup(scheduler.stop_auto_review_scheduler)

    def set_queue(self, *items):
        self.queue = {
            "success": True,
            "batchId": "draft-1" if items else None,
            "items": [{"prId": pr_id, "digest": digest} for pr_id, digest in items],
        }

    async def successful_review(self, request):
        return httpx.Response(200, json={
            "success": True, "reviewed": json.loads(request.content)["prIds"], "skipped": [],
        })

    async def handle_request(self, request):
        self.requests.append(request)
        self.assertEqual(request.headers["X-Bot-Token"], "fake-token")
        self.assertEqual(request.url.host, "keytao.invalid")
        if request.method == "GET":
            self.assertEqual(request.url.path, "/api/bot/auto-review/queue")
            self.assertEqual(dict(request.url.params), {"limit": "12"})
            return httpx.Response(200, json=self.queue)
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.url.path, "/api/bot/batches/draft-1/auto-review")
        return await self.review(request)

    def posts(self):
        return [json.loads(request.content) for request in self.requests if request.method == "POST"]

    async def start(self, cap=10):
        os.environ["KEYTAO_AUTO_REVIEW_DAILY_CHUNKS"] = str(cap)
        await self.scheduler.start_auto_review_scheduler()

    async def pause(self, expected, *, advance=False):
        if advance:
            self.clock.resume.release()
        self.assertEqual(await asyncio.wait_for(self.clock.delays.get(), 2), expected)

    async def test_disabled_settings_leave_database_and_http_untouched(self):
        for value in (None, "0", "invalid", "-1"):
            with self.subTest(value=value):
                if value is not None:
                    os.environ["KEYTAO_AUTO_REVIEW_DAILY_CHUNKS"] = value
                await self.scheduler.start_auto_review_scheduler()
                self.assertIsNone(self.scheduler._scheduler_task)
        self.assertFalse(self.db_path.exists())
        self.assertEqual(self.requests, [])
        self.logger.info.assert_called_once_with("[auto_review_scheduler] scheduler disabled")

    async def test_daily_cap_survives_restart_and_resets_at_shanghai_midnight(self):
        await self.start(cap=2)
        await self.pause(60)
        self.set_queue((42, "second"))
        await self.pause(60, advance=True)
        self.assertEqual(self.posts(), [{"prIds": [41]}, {"prIds": [42]}])
        await self.scheduler.stop_auto_review_scheduler()

        self.set_queue((43, "tomorrow"))
        await self.start(cap=2)
        await self.pause(60)
        self.assertEqual(len(self.requests), 4)
        await self.pause(60, advance=True)
        self.assertEqual(self.clock.now.hour, 0)
        self.assertEqual(self.posts()[-1], {"prIds": [43]})
        messages = [call.args[0] for call in self.logger.info.call_args_list]
        self.assertEqual(sum("daily cap reached" in message for message in messages), 1)
        self.assertEqual(
            [message.split("daily_count=")[-1] for message in messages if "batchId=" in message],
            ["1", "2", "1"],
        )

    async def test_failed_attempts_persist_and_only_a_new_digest_resets_limit(self):
        async def unavailable(_request):
            return httpx.Response(503)

        async def timed_out(_request):
            raise httpx.ConnectTimeout("fake timeout")

        self.review = unavailable
        await self.start()
        await self.pause(120)
        await self.scheduler.stop_auto_review_scheduler()

        self.review = timed_out
        await self.start()
        await self.pause(120)
        await self.pause(600, advance=True)
        self.assertEqual(self.posts(), [{"prIds": [41]}] * 2)

        self.review = self.successful_review
        self.set_queue((41, "original"), (42, "second"))
        await self.pause(60, advance=True)
        self.assertEqual(self.posts()[-1], {"prIds": [42]})
        self.set_queue((41, "changed"), (42, "second"))
        await self.pause(60, advance=True)
        self.assertEqual(self.posts()[-1], {"prIds": [41, 42]})
        self.set_queue((41, "original"), (42, "second"))
        await self.pause(600, advance=True)
        self.assertEqual(len(self.posts()), 4)

    async def test_rate_limits_back_off_for_both_response_forms(self):
        async def http_limit(_request):
            return httpx.Response(429)

        async def body_limit(_request):
            return httpx.Response(200, json={
                "success": True, "reviewed": [], "skipped": [41], "rateLimited": True,
            })

        self.review = http_limit
        await self.start()
        await self.pause(600)
        self.review = body_limit
        await self.pause(600, advance=True)
        self.assertEqual(self.posts(), [{"prIds": [41]}] * 2)

    async def test_startup_and_in_flight_post_are_serial(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_review(request):
            entered.set()
            await release.wait()
            return await self.successful_review(request)

        self.review = slow_review
        await self.start()
        await asyncio.wait_for(entered.wait(), 2)
        task = self.scheduler._scheduler_task
        await self.start()
        self.assertIs(self.scheduler._scheduler_task, task)
        self.assertTrue(self.clock.delays.empty())
        self.assertEqual(len(self.requests), 2)
        release.set()
        await self.pause(60)
        await self.pause(60, advance=True)
        self.assertEqual(len(self.posts()), 2)

    async def test_empty_queue_uses_poll_interval_without_spending_a_chunk(self):
        self.set_queue()
        os.environ["KEYTAO_AUTO_REVIEW_INTERVAL_SECONDS"] = "5"
        await self.start(cap=1)
        await self.pause(20)
        await self.pause(20, advance=True)
        self.assertEqual(self.posts(), [])
        self.set_queue((41, "original"))
        await self.pause(80, advance=True)
        self.assertEqual(self.posts(), [{"prIds": [41]}])
