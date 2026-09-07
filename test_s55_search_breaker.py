"""Offline S55 search-backend health regressions; all HTTP is fixture-only."""
import asyncio
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


_SPEC = importlib.util.spec_from_file_location(
    "s55_web_search_tools",
    Path(__file__).parent / "keytao_bot/skills/web-search/tools.py",
)
web = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(web)


class SearchBreakerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.health_patch = patch.object(web, "_SEARCH_BACKEND_HEALTH", {})
        self.status_patch = patch.object(web, "_LAST_BACKEND_STATUS", {})
        self.exa_patch = patch.object(web, "_exa_api_key", return_value=None)
        for active_patch in (self.health_patch, self.status_patch, self.exa_patch):
            active_patch.start()
            self.addCleanup(active_patch.stop)

    @staticmethod
    def result(provider):
        return [{"title": "鎗 qiāng", "url": "https://example.com/qiang", "snippet": "鎗 qiāng", "provider": provider}]

    @staticmethod
    def trip(backend="so360"):
        for _ in range(web.SEARCH_FAILURE_THRESHOLD):
            generation = web._acquire_search_backend(backend)
            web._record_search_outcome(backend, generation, success=False, elapsed=2.0)

    async def test_three_failed_turns_skip_dead_backend(self):
        calls = []

        async def search(provider, query, max_results):
            calls.append(provider)
            if provider == "so360":
                raise TimeoutError("fixture backend timeout")
            return []

        with patch.object(web, "_search_with_provider", search), patch.object(web, "_exa_api_key", return_value=None):
            for _ in range(3):
                await web.web_search("鎗 拼音", max_results=1)
            self.assertEqual(calls.count("so360"), 3)
            calls.clear()
            await web.web_search("鎗 读音", max_results=1)
        self.assertNotIn("so360", calls, "a repeatedly dead backend consumed the next pronunciation turn")

    async def test_failure_immediately_demotes_backend_before_trip(self):
        calls = []

        async def search(provider, query, max_results):
            calls.append(provider)
            if provider == "so360":
                raise TimeoutError("fixture backend timeout")
            return self.result(provider)

        with patch.object(web, "_search_with_provider", search):
            self.assertTrue((await web.web_search("鎗 拼音", max_results=1))["success"])
            calls.clear()
            result = await web.web_search("鎗 读音", max_results=1)
        self.assertEqual(calls, ["bing"])
        self.assertEqual(result["providersTried"], ["bing"])
        self.assertEqual(web._SEARCH_BACKEND_HEALTH["so360"].failures, 1)

    async def test_recent_success_and_latency_ordering(self):
        with patch.object(web, "monotonic", return_value=100.0):
            for provider, elapsed in (("so360", 1.2), ("bing", 0.18)):
                token = web._acquire_search_backend(provider)
                web._record_search_outcome(provider, token, success=True, elapsed=elapsed)
            self.assertEqual(web.search_backend_chain("鎗", exa_enabled=False)[:2], ["bing", "so360"])
        with patch.object(web, "monotonic", return_value=701.0):
            self.assertEqual(web.search_backend_chain("鎗", exa_enabled=False)[0], "so360")

    async def test_success_resets_consecutive_failure_counter(self):
        for success in (False, False, True, False, False):
            token = web._acquire_search_backend("so360")
            web._record_search_outcome("so360", token, success=success, elapsed=0.1)
        health = web._SEARCH_BACKEND_HEALTH["so360"]
        self.assertEqual((health.failures, health.state), (2, "closed"))

    async def test_cooldown_allows_exactly_one_concurrent_probe(self):
        now = [100.0]
        calls = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def search(provider, query, max_results):
            calls.append(provider)
            if provider == "so360":
                started.set()
                await release.wait()
            return self.result(provider)

        with patch.object(web, "monotonic", side_effect=lambda: now[0]), patch.object(web, "logger") as logs, patch.object(web, "_search_with_provider", search):
            self.trip()
            for _ in range(4):
                self.assertIsNone(web._acquire_search_backend("so360"))
            self.assertEqual(logs.warning.call_count, 1)
            now[0] += web.SEARCH_CIRCUIT_COOLDOWN
            first = asyncio.create_task(web.web_search("鎗 拼音", max_results=1))
            await started.wait()
            second = await web.web_search("鎗 读音", max_results=1)
            self.assertEqual(calls, ["so360", "bing"])
            self.assertEqual(web._SEARCH_BACKEND_HEALTH["so360"].state, "half_open")
            release.set()
            self.assertTrue((await first)["success"])
            self.assertTrue(second["success"])
            self.assertEqual(logs.info.call_count, 2)
        self.assertEqual(web._SEARCH_BACKEND_HEALTH["so360"].state, "closed")

    async def test_failed_probe_reopens_for_full_cooldown(self):
        now = [100.0]
        with patch.object(web, "monotonic", side_effect=lambda: now[0]):
            self.trip()
            now[0] += web.SEARCH_CIRCUIT_COOLDOWN
            token = web._acquire_search_backend("so360")
            web._record_search_outcome("so360", token, success=False, elapsed=0.1)
            health = web._SEARCH_BACKEND_HEALTH["so360"]
            self.assertEqual(health.state, "open")
            self.assertEqual(health.open_until, now[0] + web.SEARCH_CIRCUIT_COOLDOWN)
            self.assertIsNone(web._acquire_search_backend("so360"))

    async def test_stale_inflight_success_does_not_close_open_circuit(self):
        old_token = web._acquire_search_backend("so360")
        self.trip()
        web._record_search_outcome("so360", old_token, success=True, elapsed=0.1)
        self.assertEqual(web._SEARCH_BACKEND_HEALTH["so360"].state, "open")

    async def test_caller_deadline_cancellation_counts_and_releases_probe(self):
        started = asyncio.Event()

        async def search(provider, query, max_results):
            started.set()
            await asyncio.Event().wait()

        with patch.object(web, "search_backend_chain", return_value=["so360"]), patch.object(web, "_search_with_provider", search):
            for _ in range(3):
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(web.web_search("鎗 拼音"), timeout=0.01)
            self.assertEqual(web._SEARCH_BACKEND_HEALTH["so360"].state, "open")
            with patch.object(web, "monotonic", return_value=web._SEARCH_BACKEND_HEALTH["so360"].open_until):
                started.clear()
                probe = asyncio.create_task(web.web_search("鎗 读音"))
                await started.wait()
                probe.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await probe
            self.assertEqual(web._SEARCH_BACKEND_HEALTH["so360"].state, "open")

    async def test_backend_timeout_fails_over_inside_caller_budget(self):
        calls = []

        async def search(provider, query, max_results):
            calls.append(provider)
            if provider == "so360":
                await asyncio.Event().wait()
            return self.result(provider)

        with patch.object(web, "SEARCH_BACKEND_TIMEOUT", 0.01), patch.object(web, "_search_with_provider", search):
            result = await asyncio.wait_for(web.web_search("鎗 拼音", max_results=1), timeout=0.2)
        self.assertTrue(result["success"])
        self.assertEqual(calls, ["so360", "bing"])
        self.assertEqual(result["providerErrors"]["so360"], "TimeoutError")
        self.assertEqual(web._SEARCH_BACKEND_HEALTH["so360"].failures, 1)

    async def test_so360_redirect_body_never_scores_as_results(self):
        with patch.object(web, "search_backend_chain", return_value=["so360"]), patch.object(web, "_get_text", return_value=(302, "fixture anti-bot redirect")), patch.object(web, "_extract_so360", return_value=self.result("so360")) as extract:
            for _ in range(3):
                result = await web.web_search("鎗 拼音", max_results=1)
                self.assertFalse(result["success"])
                self.assertEqual(result["attempts"][0]["reason"], "HTTP 302")
            extract.assert_not_called()
        self.assertEqual(web._SEARCH_BACKEND_HEALTH["so360"].state, "open")

    async def test_so360_request_disables_redirect_following(self):
        class Response:
            status_code = 302
            text = "fixture anti-bot redirect"

        with patch.object(web, "_guarded_request", return_value=Response()) as request:
            self.assertEqual(await web._get_text(web.SO360_ENDPOINT, params={"q": "鎗"}), (302, Response.text))
        request.assert_awaited_once_with(web.SO360_ENDPOINT, params={"q": "鎗"}, max_hops=0)


if __name__ == "__main__":
    unittest.main()
