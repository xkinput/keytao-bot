"""Serve deterministic pronunciation-search snippets to offline E2E scenarios."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


WEB_PRONUNCIATION_FIXTURES_BY_SCENARIO: dict[
    str, dict[str, tuple[dict[str, str], ...]]
] = {
    "S53": {
        "薄肌 拼音": ({
            "title": "薄肌 - 读音资料甲",
            "url": "https://lexicon-a.example/thin-muscle",
            "snippet": "薄肌 拼音：bó jī，解剖学复合词。",
            "provider": "e2e-web-evidence",
        },),
        "薄肌 读音": ({
            "title": "薄肌（bó jī）术语资料",
            "url": "https://lexicon-b.example/thin-muscle",
            "snippet": "薄肌（bó jī），又称股薄肌。",
            "provider": "e2e-web-evidence",
        },),
        "薄荷味糖 拼音": ({
            "title": "薄荷味糖 - 读音资料甲",
            "url": "https://mint-a.example/pronunciation",
            "snippet": "薄荷味糖 拼音：bò hé wèi táng。",
            "provider": "e2e-web-evidence",
        },),
        "薄荷味糖 读音": ({
            "title": "薄荷味糖词语读音",
            "url": "https://mint-b.example/words/mint-candy",
            "snippet": "薄荷味糖 拼音：bò hé wèi táng。",
            "provider": "e2e-web-evidence",
        },),
        "薄肌腱 拼音": ({
            "title": "薄肌腱读音",
            "url": "https://weak-lexicon.example/thin-tendon",
            "snippet": "薄肌腱 读音：bo2 ji1 jian4。",
            "provider": "e2e-web-evidence",
        },),
        "薄肌腱 读音": ({
            "title": "薄肌腱的读音",
            "url": "https://weak-lexicon.example/pronunciation/thin-tendon",
            "snippet": "薄肌腱 拼音：bó jī jiàn。",
            "provider": "e2e-web-evidence",
        },),
        "校肌 拼音": ({
            "title": "校肌 - 百度百科",
            "url": "https://baike.baidu.com/item/%E6%A0%A1%E8%82%8C",
            "snippet": "校肌 拼音：jiào jī。",
            "provider": "e2e-web-evidence",
        },),
        "校肌 读音": ({
            "title": "校肌的读音",
            "url": "https://hanyu.baidu.com/s?wd=%E6%A0%A1%E8%82%8C",
            "snippet": "校肌 读音：xiào jī。",
            "provider": "e2e-web-evidence",
        },),
    },
}


@dataclass
class WebPronunciationEvidenceController:
    """Record registry calls and return only scenario-owned canned snippets."""

    scenario_id: str = ""
    queries: list[str] = field(default_factory=list)

    def reset(self, scenario_id: str) -> None:
        self.scenario_id = str(scenario_id or "").strip().upper()
        self.queries.clear()

    def calls_for(self, word: str) -> int:
        prefix = f"{str(word or '').strip()} "
        return sum(query.startswith(prefix) for query in self.queries)

    async def search(
        self,
        query: str,
        max_results: int = 5,
        fetch_top_n: int = 0,
        *,
        channel: str = "web",
    ) -> dict[str, Any]:
        normalized = str(query or "").strip()
        self.queries.append(normalized)
        fixture = WEB_PRONUNCIATION_FIXTURES_BY_SCENARIO.get(
            self.scenario_id,
            {},
        )
        results = [
            dict(item)
            for item in fixture.get(normalized, ())[:max(0, int(max_results))]
        ]
        return {
            "success": True,
            "query": normalized,
            "channel": channel,
            "provider": "e2e-fixture",
            "results": results,
            "fetchedPages": [] if fetch_top_n else [],
            "count": len(results),
            "attempts": [{
                "backend": "e2e-web-evidence",
                "status": "success" if results else "empty",
                "reason": "scenario-owned canned search result",
            }],
        }


async def exercise_tripped_pronunciation_backend() -> dict[str, Any]:
    """Exercise the real S53 registry/rung with isolated in-memory backend fixtures."""
    import importlib.util
    import time
    from pathlib import Path
    from unittest.mock import patch

    from keytao_bot.utils import keytao_review as review_module

    spec = importlib.util.spec_from_file_location(
        "e2e_s55_web_search_tools",
        Path(__file__).resolve().parents[1] / "keytao_bot/skills/web-search/tools.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("S55 could not load the actual search registry")
    web = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(web)
    calls: list[dict[str, str]] = []

    async def backend(provider: str, query: str, max_results: int) -> list[dict[str, str]]:
        calls.append({"backend": provider, "query": query})
        if provider == "so360":
            raise AssertionError("S55 dispatched a tripped backend")
        if provider != "bing":
            return []
        return [{
            "title": "鎗字读音",
            "url": f"https://single-{index}.example/qiang",
            "snippet": "鎗 拼音：qiāng；是枪的异体字。",
            "provider": provider,
        } for index in range(max_results)]

    with patch.object(web, "_SEARCH_BACKEND_HEALTH", {}), patch.object(
        web, "_search_with_provider", new=backend,
    ), patch.object(web, "_exa_api_key", return_value=None), patch.object(
        review_module, "_registered_web_search_function", new=web.web_search,
    ), patch.object(
        review_module.PronunciationResolutionCache, "get", return_value=None,
    ), patch.object(review_module.PronunciationResolutionCache, "set"):
        for _ in range(web.SEARCH_FAILURE_THRESHOLD):
            generation = web._acquire_search_backend("so360")
            if generation is None:
                raise AssertionError("S55 breaker tripped before its configured threshold")
            web._record_search_outcome("so360", generation, success=False, elapsed=2.0)
        health = web._SEARCH_BACKEND_HEALTH["so360"]
        if health.state != "open":
            raise AssertionError("S55 fixture did not open the real circuit breaker")
        started = time.monotonic()
        evidence = await review_module._search_pronunciation_web_evidence("鎗")
        elapsed = time.monotonic() - started
        if (
            not calls
            or any(row["backend"] == "so360" for row in calls)
            or evidence.get("status") != "resolved"
            or evidence.get("registryCalls") != 2
            or evidence.get("timedOut") is not False
            or elapsed >= 1.0
        ):
            raise AssertionError(f"S55 dead backend consumed the web rung: {calls}; {evidence}")
        return {
            "backend": "so360",
            "breakerState": health.state,
            "consecutiveFailures": health.failures,
            "deadBackendCalls": 0,
            "fixtureBackendCalls": calls,
            "elapsedSeconds": round(elapsed, 6),
            "evidence": evidence,
            "externalSearchRequests": 0,
        }


__all__ = [
    "WEB_PRONUNCIATION_FIXTURES_BY_SCENARIO",
    "WebPronunciationEvidenceController",
    "exercise_tripped_pronunciation_backend",
]
