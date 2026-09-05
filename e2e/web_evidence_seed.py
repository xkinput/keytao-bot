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


__all__ = [
    "WEB_PRONUNCIATION_FIXTURES_BY_SCENARIO",
    "WebPronunciationEvidenceController",
]
