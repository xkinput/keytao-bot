#!/usr/bin/env python3
"""Regression tests for the daily word-discovery pipeline.

Self-contained: stubs nonebot/httpx/openai the same way test_review_gate.py does,
so it runs without a NoneBot runtime and never touches the network.

    uv run python test_word_discovery.py
"""
import asyncio
import os
import re
import sys
import tempfile
import types
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- external module stubs (must come before importing keytao_bot) ----------

sys.modules["httpx"] = types.ModuleType("httpx")

_fake_nonebot = types.ModuleType("nonebot")


class _FakeMatcher:
    def handle(self):
        return lambda f: f

    async def finish(self, *a, **kw):
        pass


_fake_nonebot.on_message = lambda **kw: _FakeMatcher()
_fake_nonebot.on_command = lambda *a, **kw: _FakeMatcher()


class _FakeConfig:
    openai_api_key = "fake"
    openai_base_url = "https://fake"
    openai_model = "fake-model"
    openai_max_tokens = 1000
    openai_temperature = 0.7
    keytao_api_base = "https://fake"
    bot_api_token = "fake"


class _FakeDriver:
    config = _FakeConfig()

    def on_startup(self, func):
        return func

    def on_shutdown(self, func):
        return func


_fake_nonebot.get_driver = lambda: _FakeDriver()
_fake_nonebot.get_bots = lambda: {}
sys.modules["nonebot"] = _fake_nonebot

_fake_adapters = types.ModuleType("nonebot.adapters")
_fake_adapters.Bot = type("Bot", (), {})
_fake_adapters.Event = type("Event", (), {})
_fake_adapters.Message = type("Message", (), {})
sys.modules["nonebot.adapters"] = _fake_adapters

_fake_rule = types.ModuleType("nonebot.rule")
_fake_rule.Rule = lambda f: f
_fake_rule.to_me = lambda: lambda: None
sys.modules["nonebot.rule"] = _fake_rule

_fake_log = types.ModuleType("nonebot.log")


class _FakeLogger:
    def info(self, *a, **kw):
        pass

    def debug(self, *a, **kw):
        pass

    def warning(self, *a, **kw):
        pass

    def error(self, *a, **kw):
        pass

    def exception(self, *a, **kw):
        pass


_fake_log.logger = _FakeLogger()
sys.modules["nonebot.log"] = _fake_log

_fake_exception = types.ModuleType("nonebot.exception")


class FinishedException(Exception):
    pass


_fake_exception.FinishedException = FinishedException
sys.modules["nonebot.exception"] = _fake_exception

_fake_openai = types.ModuleType("openai")
_fake_openai.AsyncOpenAI = None
sys.modules["openai"] = _fake_openai

sys.modules["duckduckgo_search"] = types.ModuleType("duckduckgo_search")

# --- modules under test -----------------------------------------------------

from keytao_bot.utils import daily_scheduler as scheduler_module  # noqa: E402
from keytao_bot.utils import group_notify  # noqa: E402
from keytao_bot.utils import http_client  # noqa: E402
from keytao_bot.utils import keytao_review as review_module  # noqa: E402
from keytao_bot.utils import word_discovery as wd  # noqa: E402
from keytao_bot.utils.daily_scheduler import (  # noqa: E402
    DailyRunStore,
    clamp_hour,
    clamp_minute,
    date_key,
    seconds_until_next_run,
    should_catch_up,
)
from keytao_bot.utils.http_client import KeytaoApiError  # noqa: E402
from keytao_bot.utils.review_flags import MANUAL_REVIEW_FIELD  # noqa: E402
from keytao_bot.utils.word_discovery import (  # noqa: E402
    GROUP_AUTO,
    GROUP_MANUAL,
    ClassifiedCandidate,
    DiscoveryStore,
    SourceDocument,
    WordCandidate,
    build_daily_report,
    build_discovery_remark,
    build_draft_items,
    chunk_words,
    classify_reviewed,
    clean_candidates,
    filter_recent_history,
    is_recently_processed,
    parse_bilibili_hotword,
    parse_bilibili_ranking,
    parse_candidate_payload,
    parse_exa_results,
    parse_v2ex_topics,
    resolve_draft_outcome,
    sanitize_source_url,
)

passed = 0
failed = 0

_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F1E6-\U0001F1FF☀-➿⬀-⯿️]"
)


def check(name: str, result: bool):
    global passed, failed
    if result:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}")


def _clean_review(code: str = "ceek", **overrides):
    """A review result that clears every auto-ingest gate."""
    review = {
        "success": True,
        "word": "测试",
        "existing": [],
        "recommendedCode": code,
        "autoReviewable": True,
        "autoReviewReason": "至少一个权威来源给出读音",
        "lookupFailed": False,
        MANUAL_REVIEW_FIELD: False,
    }
    review.update(overrides)
    return review


def _entry(word: str, review, error: str = "", source_url: str = ""):
    return {
        "candidate": WordCandidate(word=word, reason="流行语", source_url=source_url),
        "review": review,
        "error": error,
    }


# ---------------------------------------------------------------------------


def test_candidate_cleaning_rules():
    print("\n[1] Candidate cleaning rules")

    allowed = {"https://a.example/1", "https://b.example/2"}
    raw = [
        {"word": "  绝绝子 ", "reason": "热梗", "sourceUrl": "https://a.example/1"},
        {"word": "绝绝子", "reason": "duplicate should be dropped"},
        {"word": "我", "reason": "too short"},
        {"word": "一二三四五六七八九", "reason": "too long"},
        {"word": "yyds", "reason": "no CJK at all"},
        {"word": "热 梗", "reason": "internal whitespace"},
        {"word": "", "reason": "empty"},
        {"word": None, "reason": "none"},
        "栓Q住了",
        12345,
        {"word": "破防", "source_url": "https://b.example/2"},
    ]
    cleaned = clean_candidates(raw, allowed)
    words = [item.word for item in cleaned]

    check("outer whitespace is stripped", "绝绝子" in words)
    check("duplicates collapse to the first occurrence", words.count("绝绝子") == 1)
    check("single-character candidates are rejected", "我" not in words)
    check("candidates longer than 8 chars are rejected", "一二三四五六七八九" not in words)
    check("pure-ASCII candidates are rejected", "yyds" not in words)
    check("candidates with internal whitespace are rejected", "热 梗" not in words)
    check("mixed CJK+latin candidates are accepted", "栓Q住了" in words)
    check("non-string entries are ignored", 12345 not in words)
    check("snake_case sourceUrl is accepted", any(i.word == "破防" and i.source_url.endswith("/2") for i in cleaned))
    check("sourceUrl is carried through", any(i.word == "绝绝子" and i.source_url.endswith("/1") for i in cleaned))

    overflow = clean_candidates([{"word": f"新词{index:03d}"} for index in range(120)], allowed)
    check("candidate list is capped at MAX_LLM_CANDIDATES", len(overflow) == wd.MAX_LLM_CANDIDATES)

    check("WordCandidate instances round-trip", clean_candidates([WordCandidate("摆烂")], allowed)[0].word == "摆烂")

    bidi = clean_candidates([{"word": "绝‮绝子", "reason": "带​控制符"}], allowed)
    check("bidi/zero-width characters are stripped from words", bidi and bidi[0].word == "绝绝子")
    check("bidi characters are stripped from reasons", bidi and bidi[0].reason == "带控制符")


def test_source_url_is_allowlisted():
    print("\n[1b] LLM sourceUrl must match a URL this round actually collected")

    allowed = {"https://good.example/post/1"}

    check(
        "a collected URL survives (exact match)",
        sanitize_source_url("https://good.example/post/1", allowed) == "https://good.example/post/1",
    )
    check("an uncollected URL is dropped", sanitize_source_url("https://evil.example/phish", allowed) == "")
    check(
        "a near-miss on the same host is dropped",
        sanitize_source_url("https://good.example/post/1/../evil", allowed) == "",
    )
    check("a non-http scheme is dropped", sanitize_source_url("javascript:alert(1)", allowed) == "")
    check("a data URI is dropped", sanitize_source_url("data:text/html,<script>", allowed) == "")
    check("an empty allowlist fails closed", sanitize_source_url("https://good.example/post/1", set()) == "")
    check("a missing allowlist fails closed", sanitize_source_url("https://good.example/post/1") == "")
    check(
        "bidi control characters are stripped before matching",
        sanitize_source_url("https://good.example/post/1‮", allowed) == "https://good.example/post/1",
    )

    # End to end: a prompt-injected phishing link never reaches the digest.
    documents = [
        SourceDocument(source="v2ex", title="今天的热帖", url="https://good.example/post/1"),
        SourceDocument(source="v2ex", title="没有链接的帖子"),
    ]
    urls = wd.collected_source_urls(documents)
    check("the allowlist is built from the fetched documents", urls == {"https://good.example/post/1"})

    injected = clean_candidates(
        [
            {"word": "绝绝子", "sourceUrl": "https://evil.example/steal-your-qq"},
            {"word": "破防", "sourceUrl": "https://good.example/post/1"},
        ],
        urls,
    )
    check("the injected link is discarded", injected[0].source_url == "")
    check("the candidate itself survives without its link", injected[0].word == "绝绝子")
    check("a genuine link is kept", injected[1].source_url == "https://good.example/post/1")

    remark = build_discovery_remark("2026-07-26", injected[0].source_url)
    check("a discarded link cannot reach the API remark", remark == "daily-discovery 2026-07-26")

    report = build_daily_report(
        "2026-07-26",
        [ClassifiedCandidate(injected[0], GROUP_AUTO, code="aaaa")],
        [],
        {"candidates": 1, "fresh": 1, "existing": 0, "skipped_history": 0},
    )
    check("a discarded link cannot reach the group digest", "evil.example" not in report)


def test_llm_payload_parsing():
    print("\n[2] LLM payload parsing (retry contract)")

    check(
        "fenced JSON array parses",
        parse_candidate_payload('```json\n[{"word":"摆烂"}]\n```') == [{"word": "摆烂"}],
    )
    check(
        "array embedded in prose is recovered",
        parse_candidate_payload('好的：[{"word":"emo"}] 以上') == [{"word": "emo"}],
    )
    check("object wrapper is unwrapped", parse_candidate_payload('{"words":[{"word":"上头"}]}') == [{"word": "上头"}])
    check("empty array is a valid answer, not a failure", parse_candidate_payload("[]") == [])
    check("garbage returns None so the caller retries", parse_candidate_payload("对不起我不会") is None)
    check("empty content returns None", parse_candidate_payload("   ") is None)
    check("non-dict members are dropped", parse_candidate_payload('["a", {"word":"b"}]') == [{"word": "b"}])


def test_history_dedupe_window():
    print("\n[3] 30-day local history dedupe")

    today = "2026-07-26"
    check("29 days ago is still inside the window", is_recently_processed("2026-06-27", today) is True)
    check("exactly 30 days ago has left the window", is_recently_processed("2026-06-26", today) is False)
    check("31 days ago has left the window", is_recently_processed("2026-06-25", today) is False)
    check("same day counts as recently processed", is_recently_processed(today, today) is True)
    check("unparseable stored date never blocks a word", is_recently_processed("not-a-date", today) is False)
    check("missing stored date never blocks a word", is_recently_processed(None, today) is False)
    check("future stamp (clock skew) still counts as handled", is_recently_processed("2026-08-01", today) is True)

    candidates = [WordCandidate("旧词"), WordCandidate("新词"), WordCandidate("陈词")]
    action_dates = {"旧词": "2026-07-20", "陈词": "2026-01-01"}
    fresh, skipped = filter_recent_history(candidates, action_dates, today)
    check("recently handled words are skipped", [i.word for i in skipped] == ["旧词"])
    check("stale and unseen words stay fresh", [i.word for i in fresh] == ["新词", "陈词"])

    with tempfile.TemporaryDirectory() as tmp:
        store = DiscoveryStore(db_path=os.path.join(tmp, "discovery.db"))
        store.record("绝绝子", wd.ACTION_AUTO_ADDED, "2026-07-26")
        store.record("绝绝子", wd.ACTION_RECOMMENDED, "2026-07-27")
        dates = store.get_action_dates()
        check("store upserts on the word primary key", dates == {"绝绝子": "2026-07-27"})


def test_classification_routes_every_unknown_to_humans():
    print("\n[4] Classification (needsManualReview / duplicate / no code / limit)")

    reviewed = [
        _entry("干净词", _clean_review("aaaa")),
        _entry("需审词", _clean_review("bbbb", **{MANUAL_REVIEW_FIELD: True, "manualReviewReason": "证据不足"})),
        _entry("重复词", _clean_review("cccc", existing=[{"word": "重复词", "code": "cccc"}])),
        _entry("查失败", _clean_review("dddd", lookupFailed=True)),
        _entry("无码词", _clean_review("")),
        _entry("不可自动", _clean_review("eeee", autoReviewable=False, autoReviewReason="未找到权威来源")),
        _entry("裸结果", {"success": True, "recommendedCode": "ffff", "autoReviewable": True, "existing": []}),
        _entry("失败词", None, error="连接超时"),
        _entry("未成功", {"success": False, "message": "编码服务未返回有效结果"}),
    ]

    auto_items, manual_items = classify_reviewed(reviewed, limit=10, auto_approve=True)
    auto_words = [i.word for i in auto_items]
    manual = {i.word: i for i in manual_items}

    check("only the fully cleared candidate is auto-ingested", auto_words == ["干净词"])
    check("needsManualReview goes to the recommend group", manual["需审词"].reason == "证据不足")
    check("duplicate goes to the recommend group", manual["重复词"].reason == review_module.DUPLICATE_REASON)
    check(
        "failed occupancy lookup goes to the recommend group",
        manual["查失败"].reason == review_module.LOOKUP_FAILURE_REASON,
    )
    check("missing recommended code goes to the recommend group", manual["无码词"].reason == wd.MANUAL_REASON_NO_CODE)
    check("autoReviewable=False goes to the recommend group", manual["不可自动"].reason == "未找到权威来源")
    check(
        "a review without a structured verdict is not auto-approved",
        manual["裸结果"].reason == "审核结果缺少结构化判定",
    )
    check("review exceptions go to the recommend group", "连接超时" in manual["失败词"].reason)
    check("unsuccessful review goes to the recommend group", "编码服务" in manual["未成功"].reason)
    check("recommend items keep the suggested code when known", manual["需审词"].code == "bbbb")
    check("every non-auto candidate is accounted for", len(manual_items) == len(reviewed) - 1)
    check("groups are tagged", auto_items[0].group == GROUP_AUTO and manual_items[0].group == GROUP_MANUAL)

    # Daily limit: overflow is recommended, never dropped.
    many = [_entry(f"词{index}", _clean_review(f"c{index}")) for index in range(5)]
    auto_items, manual_items = classify_reviewed(many, limit=2, auto_approve=True)
    check("daily limit caps the auto group", [i.word for i in auto_items] == ["词0", "词1"])
    check("overflow is recommended, not dropped", [i.word for i in manual_items] == ["词2", "词3", "词4"])
    check(
        "overflow carries the limit reason",
        all(i.reason == wd.MANUAL_REASON_OVER_DAILY_LIMIT for i in manual_items),
    )

    auto_items, manual_items = classify_reviewed(many, limit=10, auto_approve=False)
    check("auto-approve off recommends everything", auto_items == [] and len(manual_items) == 5)
    check(
        "auto-approve off carries its own reason",
        all(i.reason == wd.MANUAL_REASON_AUTO_APPROVE_OFF for i in manual_items),
    )

    auto_items, manual_items = classify_reviewed(many, limit=0, auto_approve=True)
    check("a zero limit recommends everything", auto_items == [] and len(manual_items) == 5)

    auto_items, manual_items = classify_reviewed(
        many, limit=10, auto_approve=False, auto_disabled_reason="词库批量查重失败，无法确认是否重复"
    )
    check(
        "the caller can say why auto-ingest is off",
        all(i.reason == "词库批量查重失败，无法确认是否重复" for i in manual_items),
    )


def test_source_failures_are_isolated():
    print("\n[5] Source isolation (one bad signal must not sink the run)")

    async def _run():
        async def boom():
            raise RuntimeError("v2ex 502")

        async def slow():
            await asyncio.sleep(1.0)
            return [SourceDocument(source="bilibili", title="never")]

        async def good():
            return [SourceDocument(source="bilibili", title="热搜词"), SourceDocument(source="bilibili", title="梗")]

        with patch.object(wd, "exa_api_key", return_value=""), \
                patch.object(wd, "collect_v2ex_hot", side_effect=boom), \
                patch.object(wd, "collect_bilibili_hot", side_effect=good):
            documents, failures = await wd.collect_sources()
        check("a failing source does not stop the others", len(documents) == 2)
        check("the failure is reported by name", any(note.startswith("v2ex") for note in failures))
        check("only the failing source is reported", len(failures) == 1)

        with patch.object(wd, "SOURCE_TIMEOUT_SECONDS", 0.05), \
                patch.object(wd, "exa_api_key", return_value=""), \
                patch.object(wd, "collect_v2ex_hot", side_effect=good), \
                patch.object(wd, "collect_bilibili_hot", side_effect=slow):
            documents, failures = await wd.collect_sources()
        check("a hung source is cut off by its own timeout", len(documents) == 2)
        check("the timeout is reported", any("超时" in note for note in failures))

        # Exa is skipped entirely when no key is configured.
        with patch.object(wd, "exa_api_key", return_value=""), \
                patch.object(wd, "collect_exa_signals", side_effect=boom), \
                patch.object(wd, "collect_v2ex_hot", side_effect=good), \
                patch.object(wd, "collect_bilibili_hot", side_effect=good):
            documents, failures = await wd.collect_sources()
        check("Exa is not attempted without EXA_API_KEY", failures == [])

        # Bilibili hotword -> ranking fallback.
        calls = []

        async def fake_get(url, **kwargs):
            calls.append(url)
            if url == wd.BILIBILI_HOTWORD_URL:
                raise RuntimeError("404")
            return {"data": {"list": [{"title": "标题一"}, {"title": "标题二"}]}}

        with patch.object(wd, "_external_get_json", side_effect=fake_get):
            documents = await wd.collect_bilibili_hot()
        check("bilibili falls back to the ranking endpoint", len(documents) == 2)
        check("the fallback actually hit the ranking URL", wd.BILIBILI_RANKING_URL in calls)

    asyncio.run(_run())

    check(
        "v2ex payload parsing keeps title and content",
        [d.title for d in parse_v2ex_topics([{"title": "标题", "content": "正文"}])] == ["标题"],
    )
    check("v2ex parser tolerates a non-list payload", parse_v2ex_topics({"error": "nope"}) == [])
    check(
        "bilibili hotword parser reads the nested list",
        [d.title for d in parse_bilibili_hotword({"list": [{"keyword": "热词"}]})] == ["热词"],
    )
    check("bilibili ranking parser tolerates junk", parse_bilibili_ranking({"data": "nope"}) == [])
    check(
        "exa parser prefers highlights",
        parse_exa_results({"results": [{"title": "T", "highlights": ["高亮句"]}]}, "q")[0].content == "高亮句",
    )


def test_outbound_fetches_are_pinned_to_trusted_hosts():
    print("\n[5b] Outbound fetches stay on hard-coded hosts (SSRF boundary)")

    async def _run():
        # Every constant this module fetches must be on the pinned list.
        for url in (wd.V2EX_HOT_URL, wd.BILIBILI_HOTWORD_URL, wd.BILIBILI_RANKING_URL, wd.EXA_SEARCH_URL):
            wd.assert_trusted_source_host(url)
        check("all four configured endpoints are pinned hosts", True)

        def rejects(url):
            try:
                wd.assert_trusted_source_host(url)
            except wd.UntrustedFetchError:
                return True
            return False

        check("an internal address is refused", rejects("http://169.254.169.254/latest/meta-data/"))
        check("localhost is refused", rejects("http://127.0.0.1:8080/admin"))
        check("an arbitrary result page is refused", rejects("https://evil.example/post/1"))
        check("a non-http scheme is refused", rejects("file:///etc/passwd"))
        check("a userinfo-prefixed lookalike is refused", rejects("https://api.exa.ai@evil.example/"))
        check("a suffix lookalike is refused", rejects("https://api.exa.ai.evil.example/search"))
        check("a subdomain lookalike is refused", rejects("https://evil.example/api.exa.ai"))

        # The guard is wired into both transports, not just documented.
        blocked = []

        async def never_called(*a, **kw):
            blocked.append(a)
            raise AssertionError("the client must not be reached for an untrusted host")

        with patch.object(http_client, "get_external_client", side_effect=never_called):
            check("the GET helper refuses before opening a client", rejects and True)
            try:
                await wd._external_get_json("https://evil.example/x")
                refused_get = False
            except wd.UntrustedFetchError:
                refused_get = True
            try:
                await wd._external_post_json("https://evil.example/x", json_body={})
                refused_post = False
            except wd.UntrustedFetchError:
                refused_post = True
        check("_external_get_json refuses an untrusted host", refused_get is True)
        check("_external_post_json refuses an untrusted host", refused_post is True)
        check("no client was ever opened for the refused fetches", blocked == [])

        # Exa: we call the fixed endpoint and consume the highlights it returns.
        # We must never crawl the result URLs it hands back.
        fetched = []

        async def record_post(url, *, json_body, headers=None):
            fetched.append(url)
            return {
                "results": [
                    {"title": "热词盘点", "url": "http://169.254.169.254/latest/meta-data/", "highlights": ["新词一"]},
                    {"title": "流行语", "url": "https://evil.example/phish", "highlights": ["新词二"]},
                ]
            }

        async def record_get(url, **kwargs):
            fetched.append(url)
            return {}

        with patch.object(wd, "exa_api_key", return_value="key"), \
                patch.object(wd, "_external_post_json", side_effect=record_post), \
                patch.object(wd, "_external_get_json", side_effect=record_get):
            documents = await wd.collect_exa_signals()

        check("Exa is queried at its fixed endpoint only", set(fetched) == {wd.EXA_SEARCH_URL})
        check("Exa result URLs are never fetched", "https://evil.example/phish" not in fetched)
        check("no metadata endpoint was touched", not any("169.254" in url for url in fetched))
        check("the highlights are still harvested", [d.content for d in documents][:1] == ["新词一"])
        check(
            "hostile result URLs stay inert data",
            {d.url for d in documents} == {
                "http://169.254.169.254/latest/meta-data/",
                "https://evil.example/phish",
            },
        )
        check(
            "every configured Exa query ran",
            len(fetched) == len(wd.EXA_QUERIES) and len(documents) == 2 * len(wd.EXA_QUERIES),
        )

    asyncio.run(_run())


def test_dictionary_lookup_is_chunked():
    print("\n[6] Dictionary lookup batching and failure handling")

    async def _run():
        words = [f"词{index:04d}" for index in range(1200)]
        check("chunking respects the 500-word ceiling", [len(c) for c in chunk_words(words)] == [500, 500, 200])

        seen_sizes = []

        async def fake_lookup(config, chunk):
            seen_sizes.append(len(chunk))
            return {}

        with patch.object(wd, "lookup_words", side_effect=fake_lookup):
            await wd.lookup_existing_words(words)
        check("lookup_existing_words issues one request per chunk", seen_sizes == [500, 500, 200])

        candidates = [WordCandidate("已有词"), WordCandidate("新词")]

        async def fake_map(_words):
            return {"已有词": [{"word": "已有词", "code": "aaaa"}]}

        with patch.object(wd, "lookup_existing_words", side_effect=fake_map):
            missing, existing, lookup_failed = await wd.split_by_dictionary(candidates)
        check("existing words are separated out", [i.word for i in existing] == ["已有词"])
        check("new words continue down the pipeline", [i.word for i in missing] == ["新词"])
        check("a clean lookup is not marked failed", lookup_failed is False)

        async def blow_up(_words):
            raise KeytaoApiError("词库词条批量查询失败")

        with patch.object(wd, "lookup_existing_words", side_effect=blow_up):
            missing, existing, lookup_failed = await wd.split_by_dictionary(candidates)
        check("a failed lookup is never read as an empty dictionary", lookup_failed is True)
        check("a failed lookup marks nothing as existing", existing == [])

    asyncio.run(_run())


def test_auto_ingest_chain_and_degradation():
    print("\n[7] Auto-ingest chain and failure degradation")

    items = [
        ClassifiedCandidate(WordCandidate("绝绝子", source_url="https://a.example/1"), GROUP_AUTO, code="aaaa"),
        ClassifiedCandidate(WordCandidate("破防", source_url=""), GROUP_AUTO, code="bbbb"),
    ]

    draft_items = build_draft_items(items, "2026-07-26")
    check("draft items are Create/Phrase", all(i["action"] == "Create" and i["type"] == "Phrase" for i in draft_items))
    check(
        "remarks are code-generated with the date prefix",
        draft_items[0]["remark"] == "daily-discovery 2026-07-26 https://a.example/1",
    )
    check("a missing source URL leaves a bare remark", draft_items[1]["remark"] == "daily-discovery 2026-07-26")
    check(
        "draft items carry the structured auto-pass verdict",
        all(i[MANUAL_REVIEW_FIELD] is False for i in draft_items),
    )
    check("remark builder trims empty URLs", build_discovery_remark("2026-07-26", "  ") == "daily-discovery 2026-07-26")

    async def _run():
        happy = [
            {"batchId": "batch-1"},
            {"successCount": 2},
            {"success": True},
            {"success": True},
        ]

        recorded = []

        def make_stub(responses):
            queue = list(responses)

            async def stub(method, path, **kwargs):
                recorded.append((method, path, kwargs))
                value = queue.pop(0)
                if isinstance(value, Exception):
                    raise value
                return value

            return stub

        with patch.object(wd, "bot_platform_id", return_value="10001"), \
                patch.object(http_client, "keytao_json", side_effect=make_stub(happy)):
            result = await wd.submit_discovered_words(items, "2026-07-26")

        paths = [path for _method, path, _kw in recorded]
        check("the full chain succeeds", result.get("success") is True and result.get("count") == 2)
        check(
            "the chain is draft -> batch-draft -> submit -> auto-approve",
            paths == [
                "/api/bot/batches/latest-draft",
                "/api/bot/pull-requests/batch-draft",
                "/api/bot/batches/batch-1/submit",
                "/api/bot/batches/batch-1/auto-approve",
            ],
        )
        check("every call is made as the bot QQ identity", all(kw.get("platform") == "qq" for _m, _p, kw in recorded))
        check("no step retries more than once", all(kw.get("retries") == 2 for _m, _p, kw in recorded))
        check("submit is confirmed", recorded[2][2]["json_body"].get("confirmed") is True)
        check(
            "the review note is code-generated",
            recorded[3][2]["json_body"]["reviewNote"].startswith("daily-discovery 2026-07-26"),
        )

        for step, responses, label in (
            (0, [KeytaoApiError("boom")], "latest-draft"),
            (1, [{"batchId": "b"}, KeytaoApiError("boom")], "batch-draft"),
            (1, [{"batchId": "b"}, {"successCount": 0, "message": "全部冲突"}], "batch-draft wrote nothing"),
            (2, [{"batchId": "b"}, {"successCount": 1}, {"success": False, "message": "no"}], "submit"),
            (3, [{"batchId": "b"}, {"successCount": 1}, {"success": True}, KeytaoApiError("no")], "auto-approve"),
        ):
            recorded.clear()
            with patch.object(wd, "bot_platform_id", return_value="10001"), \
                    patch.object(http_client, "keytao_json", side_effect=make_stub(responses)):
                result = await wd.submit_discovered_words(items, "2026-07-26")
            check(f"failure at {label} aborts the chain", result.get("success") is False)

        recorded.clear()
        with patch.object(wd, "bot_platform_id", return_value=""), \
                patch.object(http_client, "keytao_json", side_effect=make_stub(happy)):
            result = await wd.submit_discovered_words(items, "2026-07-26")
        check("an unconfigured bot QQ number blocks ingest", result.get("success") is False)
        check("nothing is sent without a bot identity", recorded == [])

        recorded.clear()
        with patch.object(wd, "bot_platform_id", return_value="10001"), \
                patch.object(http_client, "keytao_json", side_effect=make_stub([{"batchId": "../../evil"}])):
            result = await wd.submit_discovered_words(items, "2026-07-26")
        check("an unsafe batch id is refused before any write", result.get("success") is False and len(recorded) == 1)

    asyncio.run(_run())


def test_partial_ingest_is_attributed_per_word():
    print("\n[7b] Partial ingest attribution and compensation")

    items = [
        ClassifiedCandidate(WordCandidate("甲词"), GROUP_AUTO, code="aaaa"),
        ClassifiedCandidate(WordCandidate("乙词"), GROUP_AUTO, code="bbbb"),
    ]

    # -- pure attribution ---------------------------------------------------
    accepted, rejected, ids = resolve_draft_outcome(items, {"successCount": 2})
    check("a clean count accepts everything", [i.word for i in accepted] == ["甲词", "乙词"])

    accepted, rejected, ids = resolve_draft_outcome(
        items, {"successCount": 1, "failed": [{"word": "乙词", "reason": "编码冲突"}]}
    )
    check("an explicitly failed word is rejected", [i.word for i in accepted] == ["甲词"])
    check("the server's reason is kept", rejected == [(items[1], "编码冲突")])

    accepted, rejected, ids = resolve_draft_outcome(
        items, {"successCount": 1, "skipped": [{"word": "甲词", "message": "已存在"}]}
    )
    check("a skipped word is rejected too", [i.word for i in accepted] == ["乙词"])
    check("the skip reason is kept", rejected[0][1] == "已存在")

    # The exact bug: 2 requested, 1 written, no per-item detail.
    accepted, rejected, ids = resolve_draft_outcome(items, {"successCount": 1})
    check("an unattributable partial write claims nothing", accepted == [])
    check("both words are reported as unconfirmed", len(rejected) == 2)
    check("the reason names the mismatch", "无法逐条确认" in rejected[0][1])

    accepted, rejected, ids = resolve_draft_outcome(
        items,
        {
            "successCount": 2,
            "draftItems": [
                {"id": 11, "word": "甲词", "code": "aaaa"},
                {"id": 12, "word": "乙词", "code": "bbbb"},
            ],
        },
    )
    check("the draft snapshot confirms both words", [i.word for i in accepted] == ["甲词", "乙词"])
    check("draft item ids are collected for rollback", ids == [11, 12])

    accepted, rejected, ids = resolve_draft_outcome(
        items, {"successCount": 2, "draftItems": [{"id": 11, "word": "甲词", "code": "aaaa"}]}
    )
    check("a word missing from the snapshot is not claimed", [i.word for i in accepted] == ["甲词"])
    check("the missing word is explained", "未找到" in rejected[0][1])
    check("only confirmed ids are collected", ids == [11])

    # -- chain behaviour ----------------------------------------------------
    async def _run():
        recorded = []

        def make_stub(responses):
            queue = list(responses)

            async def stub(method, path, **kwargs):
                recorded.append((method, path, kwargs))
                value = queue.pop(0) if queue else {"success": True}
                if isinstance(value, Exception):
                    raise value
                return value

            return stub

        # Partial success: one word in, one word back to the humans.
        partial = [
            {"batchId": "b1"},
            {
                "successCount": 1,
                "failed": [{"word": "乙词", "reason": "编码冲突"}],
                "draftItems": [{"id": 21, "word": "甲词", "code": "aaaa"}],
            },
            {"success": True},
            {"success": True},
        ]
        recorded.clear()
        with patch.object(wd, "bot_platform_id", return_value="10001"), \
                patch.object(http_client, "keytao_json", side_effect=make_stub(partial)):
            result = await wd.submit_discovered_words(items, "2026-07-26")
        check("a partial ingest still succeeds for the good word", result.get("success") is True)
        check("only the written word is counted", result.get("count") == 1)
        check("only the written word is accepted", [i.word for i in result["accepted"]] == ["甲词"])
        check("the failed word comes back with its reason", result["rejected"] == [(items[1], "编码冲突")])
        check("the message says it was partial", "部分入库" in result.get("message", ""))

        # submit fails -> the draft rows we wrote are deleted again.
        submit_fails = [
            {"batchId": "b1"},
            {"successCount": 2, "draftItems": [
                {"id": 31, "word": "甲词", "code": "aaaa"},
                {"id": 32, "word": "乙词", "code": "bbbb"},
            ]},
            {"success": False, "message": "批次校验失败"},
            {"success": True},  # the compensating DELETE
        ]
        recorded.clear()
        with patch.object(wd, "bot_platform_id", return_value="10001"), \
                patch.object(http_client, "keytao_json", side_effect=make_stub(submit_fails)):
            result = await wd.submit_discovered_words(items, "2026-07-26")
        methods = [(m, p) for m, p, _kw in recorded]
        check("a failed submit rolls the draft rows back", ("DELETE", "/api/bot/pull-requests/batch-draft") in methods)
        check("the rollback targets exactly our rows", recorded[-1][2]["json_body"]["ids"] == [31, 32])
        check("a failed submit reports failure", result.get("success") is False)
        check("a successful rollback leaves nothing pending", result.get("pendingRecovery") == "")

        # auto-approve fails -> recall, then delete.
        approve_fails = [
            {"batchId": "b1"},
            {"successCount": 2, "draftItems": [
                {"id": 41, "word": "甲词", "code": "aaaa"},
                {"id": 42, "word": "乙词", "code": "bbbb"},
            ]},
            {"success": True},
            {"success": False, "message": "审核服务不可用"},
            {"success": True},  # recall
            {"success": True},  # delete
        ]
        recorded.clear()
        with patch.object(wd, "bot_platform_id", return_value="10001"), \
                patch.object(http_client, "keytao_json", side_effect=make_stub(approve_fails)):
            result = await wd.submit_discovered_words(items, "2026-07-26")
        paths = [p for _m, p, _kw in recorded]
        check("a failed approval recalls the submitted batch", "/api/bot/batches/recall" in paths)
        check(
            "the recalled batch's rows are then deleted",
            recorded[-1][1] == "/api/bot/pull-requests/batch-draft" and recorded[-1][0] == "DELETE",
        )
        check("a fully compensated failure leaves nothing pending", result.get("pendingRecovery") == "")

        # A write whose read timed out is no longer retried by http_client, so
        # "the draft write raised" is genuinely ambiguous. It must be settled by
        # reading the draft, not guessed.
        ambiguous = [
            {"batchId": "b1"},
            KeytaoApiError("ReadTimeout；该请求可能已生效，未自动重试"),
            {"success": True, "items": [
                {"id": 61, "word": "甲词", "code": "aaaa"},
                {"id": 62, "word": "乙词", "code": "bbbb"},
                {"id": 63, "word": "别人的词", "code": "zzzz"},
            ]},
            {"success": True},  # the compensating DELETE
        ]
        recorded.clear()
        with patch.object(wd, "bot_platform_id", return_value="10001"), \
                patch.object(http_client, "keytao_json", side_effect=make_stub(ambiguous)):
            result = await wd.submit_discovered_words(items, "2026-07-26")
        probe = [(m, p) for m, p, _kw in recorded]
        check(
            "an ambiguous draft write probes the draft instead of guessing",
            ("GET", "/api/bot/batches/latest-draft/items") in probe,
        )
        check("the probe is an idempotent GET (it may be replayed)", probe[2][0] == "GET")
        check("orphan rows found by the probe are deleted", recorded[-1][2]["json_body"]["ids"] == [61, 62])
        check("the probe never touches rows we did not write", 63 not in recorded[-1][2]["json_body"]["ids"])
        check("a settled ambiguity leaves nothing pending", result.get("pendingRecovery") == "")
        check("the round still reports failure", result.get("success") is False)

        # Same failure, but the write provably never landed: no delete at all.
        never_landed = [
            {"batchId": "b1"},
            KeytaoApiError("ConnectError"),
            {"success": True, "items": []},
        ]
        recorded.clear()
        with patch.object(wd, "bot_platform_id", return_value="10001"), \
                patch.object(http_client, "keytao_json", side_effect=make_stub(never_landed)):
            result = await wd.submit_discovered_words(items, "2026-07-26")
        check(
            "an empty draft means nothing to roll back",
            all(m != "DELETE" for m, _p, _kw in recorded),
        )
        check("a clean failure leaves nothing pending", result.get("pendingRecovery") == "")

        # The probe itself failing is the one case that must escalate.
        with tempfile.TemporaryDirectory() as tmp:
            store = DiscoveryStore(db_path=os.path.join(tmp, "probe.db"))
            unprobeable = [
                {"batchId": "b7"},
                KeytaoApiError("ReadTimeout；该请求可能已生效，未自动重试"),
                KeytaoApiError("草稿服务不可用"),
            ]
            recorded.clear()
            with patch.object(wd, "bot_platform_id", return_value="10001"), \
                    patch.object(http_client, "keytao_json", side_effect=make_stub(unprobeable)):
                result = await wd.submit_discovered_words(items, "2026-07-26", store=store)
            pending = store.list_pending_recovery()
            check("an unverifiable draft state escalates", result.get("pendingRecovery") != "")
            check("the unverifiable state is persisted", len(pending) == 1)
            check("the record explains it could not be confirmed", "无法确认草稿状态" in pending[0]["detail"])

        # Compensation itself fails -> pending_recovery is persisted, never silent.
        with tempfile.TemporaryDirectory() as tmp:
            store = DiscoveryStore(db_path=os.path.join(tmp, "recovery.db"))
            comp_fails = [
                {"batchId": "b9"},
                {"successCount": 2, "draftItems": [
                    {"id": 51, "word": "甲词", "code": "aaaa"},
                    {"id": 52, "word": "乙词", "code": "bbbb"},
                ]},
                {"success": True},
                {"success": False, "message": "审核服务不可用"},
                KeytaoApiError("recall 挂了"),
            ]
            recorded.clear()
            with patch.object(wd, "bot_platform_id", return_value="10001"), \
                    patch.object(http_client, "keytao_json", side_effect=make_stub(comp_fails)):
                result = await wd.submit_discovered_words(items, "2026-07-26", store=store)
            pending = store.list_pending_recovery()
            check("a failed rollback is escalated, not swallowed", result.get("pendingRecovery") != "")
            check("the half-written batch is recorded in sqlite", len(pending) == 1)
            check("the record names the batch", pending[0]["batchId"] == "b9")
            check("the record names the stage", pending[0]["stage"] == wd.RECOVERY_STAGE_SUBMITTED)
            check("the record names the words", pending[0]["words"] == "甲词、乙词")

    asyncio.run(_run())


def test_pipeline_reports_partial_ingest():
    print("\n[7c] The round only remembers words that were really written")

    async def _run():
        candidates = [WordCandidate("甲词"), WordCandidate("乙词")]

        async def fake_sources():
            return [SourceDocument(source="v2ex", title="热帖")], []

        async def fake_extract(_documents):
            return candidates

        async def fake_split(items):
            return list(items), [], False

        async def fake_review(items):
            return [_entry(item.word, _clean_review(f"c{index}")) for index, item in enumerate(items)]

        sent = {}

        async def fake_notify(text, run_date, store):
            sent["text"] = text
            return {"sent": ["1"], "failed": [], "chunks": 1, "configured": True}

        async def partial_submit(items, _date, **_kwargs):
            return {
                "success": True,
                "message": "部分入库",
                "count": 1,
                "accepted": [items[0]],
                "rejected": [(items[1], "编码冲突")],
                "pendingRecovery": "",
            }

        with tempfile.TemporaryDirectory() as tmp:
            store = DiscoveryStore(db_path=os.path.join(tmp, "partial.db"))
            with patch.object(wd, "collect_sources", side_effect=fake_sources), \
                    patch.object(wd, "extract_word_candidates", side_effect=fake_extract), \
                    patch.object(wd, "split_by_dictionary", side_effect=fake_split), \
                    patch.object(wd, "review_candidates", side_effect=fake_review), \
                    patch.object(wd, "submit_discovered_words", side_effect=partial_submit), \
                    patch.object(wd, "deliver_digest", side_effect=fake_notify):
                result = await wd.run_word_discovery(store=store, today="2026-07-26")

            check("only the written word is reported as ingested", [i.word for i in result.auto_items] == ["甲词"])
            check("the rejected word is recommended instead", [i.word for i in result.manual_items] == ["乙词"])
            check("the rejected word keeps the server's reason", result.manual_items[0].reason == "编码冲突")
            check("the digest says the ingest was partial", "部分入库" in result.report)

            dates = store.get_action_dates()
            check("both words are remembered", set(dates) == {"甲词", "乙词"})
            check("the failed word is not recorded as ingested", "乙词" in dates)
            check(
                "the unspent quota is refunded",
                store.remaining_daily_quota("2026-07-26", 10) == 9,
            )

    asyncio.run(_run())


def test_daily_quota_is_atomic_and_shared():
    print("\n[7d] The daily limit is a shared, atomic budget")

    with tempfile.TemporaryDirectory() as tmp:
        store = DiscoveryStore(db_path=os.path.join(tmp, "quota.db"))

        check("a first reservation is granted in full", store.consume_daily_quota("2026-07-26", 10, 4) == 4)
        check("a second run sees the budget already spent", store.consume_daily_quota("2026-07-26", 10, 8) == 6)
        check("an exhausted budget grants nothing", store.consume_daily_quota("2026-07-26", 10, 3) == 0)
        check("the remaining view agrees", store.remaining_daily_quota("2026-07-26", 10) == 0)
        check("a new day starts fresh", store.consume_daily_quota("2026-07-27", 10, 10) == 10)
        check("a zero limit grants nothing", store.consume_daily_quota("2026-07-28", 0, 5) == 0)

        store.refund_daily_quota("2026-07-26", 4)
        check("refunds return budget", store.remaining_daily_quota("2026-07-26", 10) == 4)
        store.refund_daily_quota("2026-07-26", 999)
        check("refunds never go negative", store.remaining_daily_quota("2026-07-26", 10) == 10)

    # Concurrent reservations must sum to the limit, never exceed it.
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            store = DiscoveryStore(db_path=os.path.join(tmp, "race.db"))
            grants = await asyncio.gather(
                *(asyncio.to_thread(store.consume_daily_quota, "2026-07-26", 10, 6) for _ in range(4))
            )
            check("concurrent reservations never oversell the day", sum(grants) == 10)
            check("the store agrees the day is spent", store.remaining_daily_quota("2026-07-26", 10) == 0)

        # A manual trigger cannot run while the scheduled round holds the lock.
        started = asyncio.Event()
        release = asyncio.Event()
        order = []

        async def slow_round():
            order.append("first-in")
            started.set()
            await release.wait()
            order.append("first-out")

        async def _hold():
            async with wd.pipeline_lock():
                await slow_round()

        holder = asyncio.create_task(_hold())
        await started.wait()
        check("the pipeline reports itself busy while a round runs", wd.pipeline_busy() is True)

        async def _second():
            async with wd.pipeline_lock():
                order.append("second-in")

        second = asyncio.create_task(_second())
        await asyncio.sleep(0)
        check("a concurrent round is blocked, not interleaved", order == ["first-in"])
        release.set()
        await asyncio.gather(holder, second)
        check("rounds run strictly one after another", order == ["first-in", "first-out", "second-in"])
        check("the lock is released afterwards", wd.pipeline_busy() is False)

    asyncio.run(_run())

    # Quota is enforced end to end across two consecutive rounds.
    async def _rounds():
        async def fake_sources():
            return [SourceDocument(source="v2ex", title="热帖")], []

        async def fake_split(items):
            return list(items), [], False

        async def fake_review(items):
            return [_entry(item.word, _clean_review(f"c{index}")) for index, item in enumerate(items)]

        async def ok_submit(items, _date, **_kwargs):
            return {"success": True, "count": len(items), "accepted": list(items), "rejected": []}

        async def quiet(text, run_date, store):
            return {"sent": ["1"], "failed": [], "chunks": 1, "configured": True}

        with tempfile.TemporaryDirectory() as tmp:
            store = DiscoveryStore(db_path=os.path.join(tmp, "limit.db"))

            async def run_with(words):
                async def extract(_documents):
                    return [WordCandidate(word) for word in words]

                with patch.object(wd, "collect_sources", side_effect=fake_sources), \
                        patch.object(wd, "extract_word_candidates", side_effect=extract), \
                        patch.object(wd, "split_by_dictionary", side_effect=fake_split), \
                        patch.object(wd, "review_candidates", side_effect=fake_review), \
                        patch.object(wd, "daily_limit", return_value=3), \
                        patch.object(wd, "submit_discovered_words", side_effect=ok_submit), \
                        patch.object(wd, "deliver_digest", side_effect=quiet):
                    return await wd.run_word_discovery(store=store, today="2026-07-26")

            first = await run_with(["甲", "乙", "丙", "丁"])
            check("the first round is capped at the daily limit", len(first.auto_items) == 3)
            check("the overflow is recommended", len(first.manual_items) == 1)
            check(
                "the overflow says why",
                first.manual_items[0].reason == wd.MANUAL_REASON_OVER_DAILY_LIMIT,
            )

            second = await run_with(["戊", "己"])
            check("a second trigger cannot mint a fresh allowance", second.auto_items == [])
            check("the second round recommends everything", len(second.manual_items) == 2)

    asyncio.run(_rounds())


def test_undelivered_digest_is_queued_and_replayed():
    print("\n[7e] A digest that reached nobody is queued and replayed")

    check(
        "a total send failure is detected",
        wd.notification_reached_nobody({"sent": [], "failed": ["1", "2"], "configured": True}) is True,
    )
    check(
        "a partial send is not treated as a failure",
        wd.notification_reached_nobody({"sent": ["1"], "failed": ["2"], "configured": True}) is False,
    )
    check(
        "an unconfigured broadcast is not a failure",
        wd.notification_reached_nobody({"sent": [], "failed": [], "configured": False}) is False,
    )

    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            store = DiscoveryStore(db_path=os.path.join(tmp, "outbox.db"))

            async def all_fail(_text):
                return {"sent": [], "failed": ["100", "200"], "chunks": 1, "configured": True}

            with patch.object(wd, "_notify_groups", side_effect=all_fail):
                await wd.deliver_digest("今日词汇发现", "2026-07-26", store)
            queued = store.list_pending_notifications()
            check("a lost digest is queued", len(queued) == 1)
            check("the queued body is intact", queued[0]["body"] == "今日词汇发现")

            async def all_ok(_text):
                return {"sent": ["100"], "failed": [], "chunks": 1, "configured": True}

            with patch.object(wd, "_notify_groups", side_effect=all_ok):
                delivered = await wd.flush_pending_notifications(store)
            check("the queued digest is replayed", delivered == 1)
            check("a delivered digest leaves the queue", store.list_pending_notifications() == [])

            # A successful send never queues anything.
            with patch.object(wd, "_notify_groups", side_effect=all_ok):
                await wd.deliver_digest("另一条", "2026-07-27", store)
            check("a delivered digest is not queued", store.list_pending_notifications() == [])

            # Retries are bounded so an unreachable group cannot grow the outbox.
            with patch.object(wd, "_notify_groups", side_effect=all_fail):
                await wd.deliver_digest("卡住的简报", "2026-07-28", store)
                for _ in range(wd.MAX_NOTIFY_ATTEMPTS):
                    await wd.flush_pending_notifications(store)
            check("a permanently undeliverable digest is eventually dropped", store.list_pending_notifications() == [])

            # With no groups configured the row is kept for later, not burned.
            async def unconfigured(_text):
                return {"sent": [], "failed": [], "chunks": 0, "configured": False}

            with patch.object(wd, "_notify_groups", side_effect=all_fail):
                await wd.deliver_digest("等群配置", "2026-07-29", store)
            with patch.object(wd, "_notify_groups", side_effect=unconfigured):
                delivered = await wd.flush_pending_notifications(store)
            check("an unconfigured replay delivers nothing", delivered == 0)
            check("an unconfigured replay keeps the row", len(store.list_pending_notifications()) == 1)

        # The pipeline itself queues when the broadcast reaches nobody.
        async def fake_sources():
            return [], []

        async def fake_extract(_documents):
            return []

        with tempfile.TemporaryDirectory() as tmp:
            store = DiscoveryStore(db_path=os.path.join(tmp, "pipeline_outbox.db"))
            with patch.object(wd, "collect_sources", side_effect=fake_sources), \
                    patch.object(wd, "extract_word_candidates", side_effect=fake_extract), \
                    patch.object(wd, "_notify_groups", side_effect=all_fail):
                await wd.run_word_discovery(store=store, today="2026-07-26")
            check("a round whose digest was lost queues it", len(store.list_pending_notifications()) == 1)

    asyncio.run(_run())


def test_pipeline_degrades_when_ingest_fails():
    print("\n[8] Whole-round degradation when auto-ingest fails")

    async def _run():
        candidates = [WordCandidate("绝绝子", source_url="https://a.example/1"), WordCandidate("破防")]

        async def fake_sources():
            return [SourceDocument(source="v2ex", title="热帖")], []

        async def fake_extract(_documents):
            return candidates

        async def fake_split(items):
            return list(items), [], False

        async def fake_review(items):
            return [_entry(item.word, _clean_review(f"c{index}")) for index, item in enumerate(items)]

        async def failing_submit(_items, _date, **_kwargs):
            return {"success": False, "message": "提交批次失败：HTTP 500", "count": 0}

        sent = {}

        async def fake_notify(text):
            sent["text"] = text
            return {"sent": ["1"], "failed": [], "chunks": 1}

        with tempfile.TemporaryDirectory() as tmp:
            store = DiscoveryStore(db_path=os.path.join(tmp, "discovery.db"))
            with patch.object(wd, "collect_sources", side_effect=fake_sources), \
                    patch.object(wd, "extract_word_candidates", side_effect=fake_extract), \
                    patch.object(wd, "split_by_dictionary", side_effect=fake_split), \
                    patch.object(wd, "review_candidates", side_effect=fake_review), \
                    patch.object(wd, "submit_discovered_words", side_effect=failing_submit), \
                    patch.object(wd, "_notify_groups", side_effect=fake_notify):
                result = await wd.run_word_discovery(store=store, today="2026-07-26")

            check("a failed ingest empties the auto group", result.auto_items == [])
            check("every auto candidate is demoted to the recommend group", len(result.manual_items) == 2)
            check(
                "the demotion reason is recorded",
                all(i.reason == wd.MANUAL_REASON_INGEST_FAILED for i in result.manual_items),
            )
            check("the suggested codes survive the demotion", [i.code for i in result.manual_items] == ["c0", "c1"])
            check("the failure is explained in the report", "自动入库失败" in result.report)
            check("the group still gets a digest", "text" in sent)
            check(
                "demoted words are remembered as recommended",
                store.get_action_dates() == {"绝绝子": "2026-07-26", "破防": "2026-07-26"},
            )

        # Dry run: classify only, write nothing, remember nothing.
        submitted = {"called": False}

        async def tracking_submit(_items, _date, **_kwargs):
            submitted["called"] = True
            return {"success": True, "count": 2}

        with tempfile.TemporaryDirectory() as tmp:
            store = DiscoveryStore(db_path=os.path.join(tmp, "dry.db"))
            with patch.object(wd, "collect_sources", side_effect=fake_sources), \
                    patch.object(wd, "extract_word_candidates", side_effect=fake_extract), \
                    patch.object(wd, "split_by_dictionary", side_effect=fake_split), \
                    patch.object(wd, "review_candidates", side_effect=fake_review), \
                    patch.object(wd, "submit_discovered_words", side_effect=tracking_submit), \
                    patch.object(wd, "_notify_groups", side_effect=fake_notify):
                result = await wd.run_word_discovery(store=store, today="2026-07-26", dry_run=True)

            check("a dry run never writes the dictionary", submitted["called"] is False)
            check("a dry run still classifies", len(result.auto_items) == 2)
            check("a dry run leaves no history behind", store.get_action_dates() == {})
            check("a dry run labels its report", "试运行" in result.report)

        # A failed dictionary lookup forces the whole round to recommend-only.
        submitted["called"] = False

        async def failed_split(items):
            return list(items), [], True

        with tempfile.TemporaryDirectory() as tmp:
            store = DiscoveryStore(db_path=os.path.join(tmp, "failed.db"))
            with patch.object(wd, "collect_sources", side_effect=fake_sources), \
                    patch.object(wd, "extract_word_candidates", side_effect=fake_extract), \
                    patch.object(wd, "split_by_dictionary", side_effect=failed_split), \
                    patch.object(wd, "review_candidates", side_effect=fake_review), \
                    patch.object(wd, "submit_discovered_words", side_effect=tracking_submit), \
                    patch.object(wd, "_notify_groups", side_effect=fake_notify):
                result = await wd.run_word_discovery(store=store, today="2026-07-26")
            check("a failed dictionary lookup blocks all auto-ingest", result.auto_items == [])
            check("the blocked round says why", "词库批量查重失败" in result.report)
            check("nothing was submitted during a blocked round", submitted["called"] is False)
            check(
                "each blocked word carries the lookup-failure reason",
                all(i.reason == wd.MANUAL_REASON_DICTIONARY_LOOKUP_FAILED for i in result.manual_items),
            )

    asyncio.run(_run())


def test_report_rendering_and_chunking():
    print("\n[9] Daily report rendering and chunking")

    auto_items = [
        ClassifiedCandidate(WordCandidate("绝绝子", source_url="https://a.example/1"), GROUP_AUTO, code="aaaa"),
    ]
    manual_items = [
        ClassifiedCandidate(WordCandidate("破防"), GROUP_MANUAL, code="bbbb", reason="证据不足"),
        ClassifiedCandidate(WordCandidate("上头"), GROUP_MANUAL, code="", reason=wd.MANUAL_REASON_NO_CODE),
    ]
    stats = {"candidates": 9, "fresh": 5, "existing": 2, "skipped_history": 2}

    report = build_daily_report("2026-07-26", auto_items, manual_items, stats)
    lines = report.split("\n")

    check("the title carries the date", lines[0] == "喵喵每日词汇发现 2026-07-26")
    check("the title has no emoji", not _EMOJI_RE.search(lines[0]))
    check("the body has at most one emoji", len(_EMOJI_RE.findall(report)) <= 1)
    check("auto entries show word, code and source", "· 绝绝子 — aaaa — https://a.example/1" in report)
    check("recommend entries show the suggested code", "· 破防 — 建议编码 bbbb — 证据不足" in report)
    check("recommend entries without a code say so", "· 上头 — 暂无建议编码 —" in report)
    check("group sizes are stated", "自动入库 1 个" in report and "待人工推荐 2 个" in report)
    check("a single stats line closes the report", report.count("统计：") == 1)
    check("stats numbers are consistent", "候选 9 · 新词 5 · 自动入库 1 · 待推荐 2 · 词库已有 2 · 近期已处理 2" in report)

    empty = build_daily_report("2026-07-26", [], [], {"candidates": 0, "fresh": 0, "existing": 0, "skipped_history": 0})
    check("an empty round still produces a digest", "今天没有筛出可推荐的新词。" in empty)
    check("an empty round still reports stats", "统计：" in empty)

    with_failures = build_daily_report(
        "2026-07-26", auto_items, manual_items, stats, source_failures=["v2ex: 超时"]
    )
    check("source failures are surfaced", "信源异常：v2ex: 超时" in with_failures)

    # Chunking: a long digest must survive the QQ message ceiling intact.
    long_manual = [
        ClassifiedCandidate(WordCandidate(f"新词{index:03d}"), GROUP_MANUAL, code=f"c{index:03d}", reason="证据不足")
        for index in range(200)
    ]
    long_report = build_daily_report("2026-07-26", auto_items, long_manual, stats)
    chunks = group_notify.split_message(long_report)
    check("a long digest is split into several messages", len(chunks) > 1)
    check(
        "no chunk exceeds the message ceiling",
        all(len(chunk) <= group_notify.MAX_MESSAGE_CHARS for chunk in chunks),
    )
    check("no line is lost when chunking", all(f"新词{index:03d}" in "\n".join(chunks) for index in range(200)))
    check("a short digest stays a single message", len(group_notify.split_message(report)) == 1)


def test_scheduler_catch_up_and_timing():
    print("\n[10] Daily scheduler catch-up and timing")

    now = datetime(2026, 7, 26, 10, 30)

    check("never run and the slot has passed -> catch up", should_catch_up(now, 9, 0, None) is True)
    check("never run and the slot is still ahead -> wait", should_catch_up(now, 23, 0, None) is False)
    check("yesterday's run and the slot has passed -> catch up", should_catch_up(now, 9, 0, "2026-07-25") is True)
    check("already ran today -> no catch up", should_catch_up(now, 9, 0, "2026-07-26") is False)
    check("a future stamp (clock skew) -> no catch up", should_catch_up(now, 9, 0, "2026-07-27") is False)
    check("blank stamp behaves like never run", should_catch_up(now, 9, 0, "  ") is True)
    check(
        "exactly at the slot with no prior run -> catch up",
        should_catch_up(datetime(2026, 7, 26, 9, 0), 9, 0, None) is True,
    )
    check(
        "one minute before the slot -> wait",
        should_catch_up(datetime(2026, 7, 26, 8, 59), 9, 0, None) is False,
    )

    check("next run later today", seconds_until_next_run(now, 12, 0) == 90 * 60)
    check("a passed slot rolls to tomorrow", seconds_until_next_run(now, 9, 0) == (24 - 1.5) * 3600)
    check("the current instant rolls to tomorrow", seconds_until_next_run(now, 10, 30) == 24 * 3600)
    check("date_key is ISO", date_key(now) == "2026-07-26")

    check("bad hours fall back", clamp_hour("25", 9) == 9 and clamp_hour("x", 9) == 9 and clamp_hour("0", 9) == 0)
    check("bad minutes fall back", clamp_minute("60", 0) == 0 and clamp_minute("30", 0) == 30)

    with tempfile.TemporaryDirectory() as tmp:
        store = DailyRunStore(db_path=os.path.join(tmp, "sched.db"))
        check("an unknown task has no recorded date", store.get_last_run_date("word_discovery") is None)
        store.set_last_run_date("word_discovery", "2026-07-26")
        store.set_last_run_date("word_discovery", "2026-07-27")
        check("the recorded date is upserted", store.get_last_run_date("word_discovery") == "2026-07-27")
        check("tasks are isolated by name", store.get_last_run_date("other") is None)

    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            store = DailyRunStore(db_path=os.path.join(tmp, "run.db"))
            calls = []

            async def callback():
                calls.append(1)
                return "ok"

            scheduler = scheduler_module.DailyScheduler(
                "unit-task", callback, hour=9, minute=0, store=store
            )
            result = await scheduler.run_now("test")
            check("run_now executes the callback", result == "ok" and calls == [1])
            check("run_now records today's date", store.get_last_run_date("unit-task") == date_key(scheduler.now()))

            async def boom():
                raise RuntimeError("callback exploded")

            failing = scheduler_module.DailyScheduler(
                "failing-task", boom, hour=9, minute=0, store=store
            )
            raised = False
            try:
                await failing.run_now("test")
            except RuntimeError:
                raised = True
            check("run_now propagates callback errors to its caller", raised is True)
            check(
                "a failed attempt is still recorded (no restart storm)",
                store.get_last_run_date("failing-task") is not None,
            )

            # The loop, by contrast, must swallow the error and keep going.
            await failing._guarded_run("test")
            check("the loop's guarded run never raises", True)

            # Shutdown mid-run must NOT count as today's attempt, or catch-up
            # would skip the whole day after the restart.
            cancel_started = asyncio.Event()

            async def never_finishes():
                cancel_started.set()
                await asyncio.sleep(3600)

            cancelled = scheduler_module.DailyScheduler(
                "cancelled-task", never_finishes, hour=9, minute=0, store=store
            )
            task = asyncio.create_task(cancelled.run_now("scheduled"))
            await cancel_started.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            check(
                "a cancelled run records no date",
                store.get_last_run_date("cancelled-task") is None,
            )
            check(
                "so catch-up still picks the day up after restart",
                should_catch_up(
                    datetime(2026, 7, 26, 10, 0), 9, 0, store.get_last_run_date("cancelled-task")
                ) is True,
            )

            # stop() cancels the loop; the same rule applies through that path.
            running = scheduler_module.DailyScheduler(
                "shutdown-task", never_finishes, hour=0, minute=0, store=store, catch_up=True
            )
            cancel_started.clear()
            running.start()
            await cancel_started.wait()
            await running.stop()
            check(
                "shutting the scheduler down leaves the day unrecorded",
                store.get_last_run_date("shutdown-task") is None,
            )

    asyncio.run(_run())


def main():
    test_candidate_cleaning_rules()
    test_source_url_is_allowlisted()
    test_llm_payload_parsing()
    test_history_dedupe_window()
    test_classification_routes_every_unknown_to_humans()
    test_source_failures_are_isolated()
    test_outbound_fetches_are_pinned_to_trusted_hosts()
    test_dictionary_lookup_is_chunked()
    test_auto_ingest_chain_and_degradation()
    test_partial_ingest_is_attributed_per_word()
    test_pipeline_reports_partial_ingest()
    test_daily_quota_is_atomic_and_shared()
    test_undelivered_digest_is_queued_and_replayed()
    test_pipeline_degrades_when_ingest_fails()
    test_report_rendering_and_chunking()
    test_scheduler_catch_up_and_timing()

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{passed + failed} passed" + (f", {failed} failed" if failed else ""))
    if failed:
        print("❌ SOME TESTS FAILED")
        return 1
    print("✅ ALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
