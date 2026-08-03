"""Daily Chinese-neologism discovery pipeline.

Once a day the bot samples a few public "what is trending in Chinese right now"
signals, asks the LLM to pull neologism candidates out of the raw text, and then
puts every candidate through the same safety gates a human draft goes through:

    sources -> LLM extraction -> cleaning -> local history -> dictionary lookup
            -> pronunciation review -> classification -> (auto ingest) -> report

The pipeline is deliberately built out of small pure functions with the network
confined to a handful of thin ``async`` wrappers, so the interesting logic
(cleaning rules, history windows, classification, report rendering) is unit
testable without touching a socket.

Two rules drive every judgement call in here:

* **A failed lookup is not a clean slot.** Anything the review chain cannot
  positively confirm - a failed dictionary query, a missing recommended code, a
  ``needsManualReview`` verdict, a duplicate - drops to the "recommend to a
  human" group. Nothing unknown is ever auto-approved.
* **One source failing degrades only that source.** Each signal is fetched under
  its own timeout and ``try``/``except``; the run continues on whatever came
  back, and the group report says which sources were down.
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from nonebot.log import logger

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - optional dependency guard
    AsyncOpenAI = None  # type: ignore

from . import http_client
from .http_client import KeytaoApiError
from .keytao_review import (
    DUPLICATE_REASON,
    LOOKUP_FAILURE_REASON,
    ReviewHttpConfig,
    get_llm_client,
    llm_config,
    lookup_words,
    prepare_reviewed_word,
)
from .review_flags import (
    apply_manual_review_flag,
    manual_review_reason,
    read_manual_review_flag,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V2EX_HOT_URL = "https://www.v2ex.com/api/topics/hot.json"
BILIBILI_HOTWORD_URL = "https://s.search.bilibili.com/main/hotword"
BILIBILI_RANKING_URL = "https://api.bilibili.com/x/web-interface/ranking/v2"
EXA_SEARCH_URL = "https://api.exa.ai/search"

EXA_QUERIES = (
    "最近流行的中文网络新词 热梗",
    "本周互联网流行语",
    "最新中文网络热词 出处 含义",
)
EXA_NUM_RESULTS = 8

# Every third-party signal gets its own budget; a slow source degrades itself only.
SOURCE_TIMEOUT_SECONDS = 15.0

MAX_LLM_CANDIDATES = 40
MAX_SOURCE_DOCUMENTS = 120
MAX_DOCUMENT_CHARS = 400
MAX_PROMPT_CHARS = 24000

WORD_MIN_LENGTH = 2
WORD_MAX_LENGTH = 8
HISTORY_SKIP_DAYS = 30

# keytao-next caps the by-word batch endpoint at 500 entries per request.
WORD_LOOKUP_BATCH_SIZE = 500

REVIEW_CONCURRENCY = 3
DEFAULT_DAILY_LIMIT = 10

BOT_PLATFORM = "qq"
REMARK_PREFIX = "daily-discovery"

DISCOVERY_DB_FILENAME = "word_discovery.db"

# Quota bucket name. One bucket for the whole feature: the scheduled run and
# every manual trigger draw from the same daily allowance.
QUOTA_KEY = "word_discovery_auto_ingest"

# An undelivered digest is retried on later runs, then given up on so a
# permanently unreachable group cannot grow the outbox without bound.
MAX_NOTIFY_ATTEMPTS = 5

RECOVERY_STAGE_DRAFT = "draft"
RECOVERY_STAGE_SUBMITTED = "submitted"
RECOVERY_STAGE_APPROVED = "approved"

ACTION_AUTO_ADDED = "auto_added"
ACTION_RECOMMENDED = "recommended"
ACTION_ALREADY_EXISTS = "already_exists"

GROUP_AUTO = "auto"
GROUP_MANUAL = "manual"

_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_WHITESPACE_RE = re.compile(r"\s")

# Bidi overrides / zero-width joiners let a crafted string render as something
# other than what it is. Everything in this pipeline originates from scraped
# third-party text, so it is stripped before the text reaches a group message or
# an API payload.
_BIDI_CONTROL_RE = re.compile(
    "[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\u061c\ufeff]"
)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ALLOWED_URL_SCHEMES = ("http://", "https://")

MANUAL_REASON_AUTO_APPROVE_OFF = "自动入库已关闭，仅作推荐"
MANUAL_REASON_OVER_DAILY_LIMIT = "超出今日自动入库上限，转人工推荐"
MANUAL_REASON_NO_CODE = "未获得可用的推荐编码"
MANUAL_REASON_NO_PLATFORM_ID = "未配置喵喵Bot 的 QQ 号，无法自动入库"
MANUAL_REASON_INGEST_FAILED = "自动入库链路失败，已转人工推荐"
MANUAL_REASON_DICTIONARY_LOOKUP_FAILED = "词库批量查重失败，无法确认是否重复"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _config_value(name: str, env_name: str, default: Any = None) -> Any:
    return http_client.config_value(name, env_name, default)


def _as_int(value: Any, default: int, minimum: int = 0, maximum: int = 10_000) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    if parsed < minimum or parsed > maximum:
        return default
    return parsed


def _as_bool(value: Any, default: bool) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def discovery_enabled() -> bool:
    return _as_bool(_config_value("word_discovery_enabled", "WORD_DISCOVERY_ENABLED", None), False)


def auto_approve_enabled() -> bool:
    return _as_bool(_config_value("word_discovery_auto_approve", "WORD_DISCOVERY_AUTO_APPROVE", None), True)


def daily_limit() -> int:
    return _as_int(
        _config_value("word_discovery_daily_limit", "WORD_DISCOVERY_DAILY_LIMIT", None),
        DEFAULT_DAILY_LIMIT,
        minimum=0,
        maximum=500,
    )


def bot_platform_id() -> str:
    return str(
        _config_value("word_discovery_bot_platform_id", "WORD_DISCOVERY_BOT_PLATFORM_ID", "") or ""
    ).strip()


def exa_api_key() -> str:
    return str(_config_value("exa_api_key", "EXA_API_KEY", "") or "").strip()


def notify_group_ids() -> Any:
    return _config_value("word_discovery_group_ids", "WORD_DISCOVERY_GROUP_IDS", "")


# Serialises every round: the scheduled task and the manual command must never
# run concurrently, or they would each reserve and write against a stale view of
# the day's budget. Created lazily so importing this module needs no event loop.
_pipeline_lock: Optional[asyncio.Lock] = None


def pipeline_lock() -> asyncio.Lock:
    global _pipeline_lock
    if _pipeline_lock is None:
        _pipeline_lock = asyncio.Lock()
    return _pipeline_lock


def pipeline_busy() -> bool:
    """Whether a round is currently running (advisory, for a friendly reply)."""
    return _pipeline_lock is not None and _pipeline_lock.locked()


def _review_config() -> ReviewHttpConfig:
    """Legacy handle expected by the review helpers (real values read per request)."""
    return ReviewHttpConfig(
        api_base=http_client.get_keytao_url(),
        bot_token=http_client.get_bot_token() or "",
    )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SourceDocument:
    """One raw snippet of trending text, kept with its origin for citation."""

    source: str
    title: str = ""
    content: str = ""
    url: str = ""

    def as_prompt_text(self) -> str:
        body = f"{self.title}\n{self.content}".strip()
        return body[:MAX_DOCUMENT_CHARS]


@dataclass
class WordCandidate:
    word: str
    reason: str = ""
    source_url: str = ""


@dataclass
class ClassifiedCandidate:
    """A candidate after review, routed into the auto-ingest or recommend group."""

    candidate: WordCandidate
    group: str
    code: str = ""
    reason: str = ""

    @property
    def word(self) -> str:
        return self.candidate.word

    @property
    def source_url(self) -> str:
        return self.candidate.source_url


@dataclass
class DiscoveryResult:
    run_date: str = ""
    auto_items: List[ClassifiedCandidate] = field(default_factory=list)
    manual_items: List[ClassifiedCandidate] = field(default_factory=list)
    existing_words: List[str] = field(default_factory=list)
    skipped_history: List[str] = field(default_factory=list)
    source_failures: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    report: str = ""
    ingest: Dict[str, Any] = field(default_factory=dict)
    notify: Dict[str, Any] = field(default_factory=dict)
    quota: Dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Outbound HTTP helpers (shared client + global scraping semaphore)
# ---------------------------------------------------------------------------


# Every outbound request in this module goes to one of these compile-time hosts,
# which is exactly the contract of the unguarded shared client. URLs discovered
# at runtime (hot-list links, Exa result pages) are never fetched here - they are
# only quoted into the prompt and used as the source allowlist - so no call site
# needs ``guarded_fetch``. The check below enforces that rather than trusting it:
# a future call site that passes a response-derived URL fails closed instead of
# silently bypassing the SSRF guard.
TRUSTED_SOURCE_HOSTS = frozenset({
    "www.v2ex.com",
    "s.search.bilibili.com",
    "api.bilibili.com",
    "api.exa.ai",
})


class UntrustedFetchError(RuntimeError):
    """Raised when this module is asked to fetch a host it does not pin."""


def assert_trusted_source_host(url: str) -> None:
    """Reject anything that is not one of this module's hard-coded endpoints."""
    from urllib.parse import urlparse

    parsed = urlparse(str(url or ""))
    # ``hostname`` lower-cases the host and drops any port/userinfo, so
    # "https://api.exa.ai@evil.example/" correctly resolves to evil.example.
    if parsed.scheme not in ("http", "https") or parsed.hostname not in TRUSTED_SOURCE_HOSTS:
        raise UntrustedFetchError(
            f"refusing to fetch {url!r} through the unguarded client; "
            "runtime-discovered URLs must go through http_client.guarded_fetch"
        )


async def _external_get_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    assert_trusted_source_host(url)
    client = await http_client.get_external_client()
    async with http_client.external_fetch_semaphore():
        response = await client.get(url, params=params, headers=headers)
    response.raise_for_status()
    return response.json()


async def _external_post_json(
    url: str,
    *,
    json_body: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
) -> Any:
    assert_trusted_source_host(url)
    client = await http_client.get_external_client()
    async with http_client.external_fetch_semaphore():
        response = await client.post(url, json=json_body, headers=headers)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Source parsing (pure)
# ---------------------------------------------------------------------------


def strip_unsafe_characters(value: Any) -> str:
    """Remove bidi/zero-width/control characters from third-party text."""
    text = str(value or "")
    text = _BIDI_CONTROL_RE.sub("", text)
    return _CONTROL_CHAR_RE.sub("", text)


def _clean_text(value: Any, limit: int = MAX_DOCUMENT_CHARS) -> str:
    text = re.sub(r"\s+", " ", strip_unsafe_characters(value)).strip()
    return text[:limit]


def parse_v2ex_topics(payload: Any) -> List[SourceDocument]:
    if not isinstance(payload, list):
        return []
    documents: List[SourceDocument] = []
    for topic in payload:
        if not isinstance(topic, dict):
            continue
        title = _clean_text(topic.get("title"))
        content = _clean_text(topic.get("content") or topic.get("content_rendered"))
        if not title and not content:
            continue
        documents.append(
            SourceDocument(
                source="v2ex",
                title=title,
                content=content,
                url=_clean_text(topic.get("url"), 300),
            )
        )
    return documents


def parse_bilibili_hotword(payload: Any) -> List[SourceDocument]:
    """Parse the search-hotword feed, which nests the list under a few keys."""
    if not isinstance(payload, dict):
        return []
    raw_list = payload.get("list")
    if not isinstance(raw_list, list):
        inner = payload.get("data")
        raw_list = inner.get("list") if isinstance(inner, dict) else inner
    if not isinstance(raw_list, list):
        return []
    documents: List[SourceDocument] = []
    for entry in raw_list:
        if isinstance(entry, str):
            keyword = _clean_text(entry)
        elif isinstance(entry, dict):
            keyword = _clean_text(entry.get("keyword") or entry.get("show_name") or entry.get("name"))
        else:
            continue
        if keyword:
            documents.append(SourceDocument(source="bilibili-hotword", title=keyword))
    return documents


def parse_bilibili_ranking(payload: Any) -> List[SourceDocument]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    videos = data.get("list") if isinstance(data, dict) else None
    if not isinstance(videos, list):
        return []
    documents: List[SourceDocument] = []
    for video in videos:
        if not isinstance(video, dict):
            continue
        title = _clean_text(video.get("title"))
        if not title:
            continue
        documents.append(
            SourceDocument(
                source="bilibili-ranking",
                title=title,
                content=_clean_text(video.get("desc")),
                url=_clean_text(video.get("short_link_v2") or video.get("short_link"), 300),
            )
        )
    return documents


def parse_exa_results(payload: Any, query: str = "") -> List[SourceDocument]:
    """Keep the highlight sentences: that is where fresh slang actually shows up."""
    if not isinstance(payload, dict):
        return []
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    documents: List[SourceDocument] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        highlights = result.get("highlights")
        if isinstance(highlights, list):
            body = " ".join(_clean_text(item, 200) for item in highlights if item)
        else:
            body = _clean_text(highlights)
        if not body:
            body = _clean_text(result.get("text") or result.get("summary"))
        title = _clean_text(result.get("title"))
        if not title and not body:
            continue
        documents.append(
            SourceDocument(
                source=f"exa:{query}" if query else "exa",
                title=title,
                content=body,
                url=_clean_text(result.get("url"), 300),
            )
        )
    return documents


# ---------------------------------------------------------------------------
# Source collection (network)
# ---------------------------------------------------------------------------


async def collect_v2ex_hot() -> List[SourceDocument]:
    payload = await _external_get_json(
        V2EX_HOT_URL,
        headers={"User-Agent": http_client.EXTERNAL_USER_AGENT, "Accept": "application/json"},
    )
    return parse_v2ex_topics(payload)


async def collect_bilibili_hot() -> List[SourceDocument]:
    """Search hotwords first; fall back to the video ranking titles."""
    try:
        payload = await _external_get_json(BILIBILI_HOTWORD_URL)
        documents = parse_bilibili_hotword(payload)
        if documents:
            return documents
        logger.debug("[word_discovery] bilibili hotword feed returned nothing, trying ranking")
    except Exception as error:
        logger.debug(f"[word_discovery] bilibili hotword feed failed: {error}, trying ranking")

    payload = await _external_get_json(BILIBILI_RANKING_URL, params={"rid": 0, "type": "all"})
    return parse_bilibili_ranking(payload)


async def collect_exa_signals() -> List[SourceDocument]:
    api_key = exa_api_key()
    if not api_key:
        return []
    documents: List[SourceDocument] = []
    for query in EXA_QUERIES:
        try:
            payload = await _external_post_json(
                EXA_SEARCH_URL,
                json_body={
                    "query": query,
                    "numResults": EXA_NUM_RESULTS,
                    "contents": {"highlights": True},
                },
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
            )
        except Exception as error:
            # One bad query must not sink the other Exa queries.
            logger.warning(f"[word_discovery] Exa query failed ({query}): {error}")
            continue
        documents.extend(parse_exa_results(payload, query))
    return documents


async def _run_source(name: str, coroutine: Any) -> Tuple[List[SourceDocument], str]:
    """Run one collector under its own timeout, converting failure into a note."""
    try:
        documents = await asyncio.wait_for(coroutine, timeout=SOURCE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning(f"[word_discovery] source {name} timed out after {SOURCE_TIMEOUT_SECONDS}s")
        return [], f"{name}: 超时"
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.warning(f"[word_discovery] source {name} failed: {error}")
        return [], f"{name}: {error}"
    return list(documents or []), ""


async def collect_sources() -> Tuple[List[SourceDocument], List[str]]:
    """Gather every configured signal. Returns ``(documents, failure notes)``."""
    planned: List[Tuple[str, Any]] = [
        ("v2ex", collect_v2ex_hot()),
        ("bilibili", collect_bilibili_hot()),
    ]
    if exa_api_key():
        planned.append(("exa", collect_exa_signals()))

    outcomes = await asyncio.gather(
        *(_run_source(name, coroutine) for name, coroutine in planned)
    )

    documents: List[SourceDocument] = []
    failures: List[str] = []
    for collected, failure in outcomes:
        documents.extend(collected)
        if failure:
            failures.append(failure)
    if len(documents) > MAX_SOURCE_DOCUMENTS:
        documents = documents[:MAX_SOURCE_DOCUMENTS]
    return documents, failures


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = (
    "你是中文新词与网络流行语的发现助手。"
    "给你一批来自中文互联网热榜、热搜与语义搜索的原始文本，请从中挑出真实出现的中文新词、网络流行语或热梗候选。"
    "严格要求："
    "1) 每个候选是 2 到 6 个字的中文词，不要句子、不要英文缩写、不要人名地名等普通专名；"
    "2) 只挑原文里真的出现过的词，不要凭空造词；"
    "3) reason 用一句话说明它为什么算新词或流行语；"
    "4) sourceUrl 必须来自给定素材里的链接，没有就留空字符串；"
    f"5) 最多返回 {MAX_LLM_CANDIDATES} 条。"
    '只输出一个 JSON 数组，形如 [{"word":"","reason":"","sourceUrl":""}]，不要任何解释文字。'
)


def build_extraction_prompt(documents: Sequence[SourceDocument]) -> str:
    """Render the collected documents into a bounded prompt payload."""
    lines: List[str] = []
    used = 0
    for index, document in enumerate(documents, start=1):
        text = document.as_prompt_text()
        if not text:
            continue
        line = f"[{index}] ({document.source}) {text}"
        if document.url:
            line += f" | {document.url}"
        if used + len(line) > MAX_PROMPT_CHARS:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def parse_candidate_payload(content: str) -> Optional[List[Dict[str, Any]]]:
    """Parse the model's JSON array. ``None`` means "unparseable, retry".

    An empty list is a legitimate answer ("nothing trending today") and is
    deliberately distinguished from a parse failure.
    """
    text = (content or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except Exception:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except Exception:
            return None
    if isinstance(value, dict):
        for key in ("words", "candidates", "items", "data"):
            inner = value.get(key)
            if isinstance(inner, list):
                value = inner
                break
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, dict)]


async def _call_extraction_llm(prompt: str) -> str:
    config = llm_config()
    if not config["api_key"] or AsyncOpenAI is None:
        raise RuntimeError("LLM 未配置（缺少 OPENAI_API_KEY）")
    client = get_llm_client(
        AsyncOpenAI,
        config["base_url"],
        config["api_key"],
        config["timeout"],
    )
    response = await client.chat.completions.create(
        model=config["model"],
        temperature=0.2,
        max_tokens=config["max_tokens"],
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    if not response.choices:
        return ""
    return response.choices[0].message.content or ""


async def extract_word_candidates(documents: Sequence[SourceDocument]) -> List[WordCandidate]:
    """Ask the LLM for candidates, retrying a malformed answer exactly once."""
    prompt = build_extraction_prompt(documents)
    if not prompt:
        return []
    # Only URLs we actually fetched this round may be echoed back at us.
    allowed_urls = collected_source_urls(documents)

    for attempt in range(2):
        try:
            content = await _call_extraction_llm(prompt)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning(f"[word_discovery] LLM extraction call failed (attempt {attempt + 1}): {error}")
            if attempt == 1:
                return []
            continue
        parsed = parse_candidate_payload(content)
        if parsed is not None:
            return clean_candidates(parsed, allowed_urls)
        logger.warning(f"[word_discovery] LLM extraction returned unparseable JSON (attempt {attempt + 1})")
    return []


# ---------------------------------------------------------------------------
# Cleaning (pure)
# ---------------------------------------------------------------------------


def contains_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(text or ""))


def collected_source_urls(documents: Iterable[SourceDocument]) -> set:
    """Every URL this round actually fetched, for exact-match validation."""
    urls: set = set()
    for document in documents or []:
        url = str(getattr(document, "url", "") or "").strip()
        if url:
            urls.add(url)
    return urls


def sanitize_source_url(raw: Any, allowed_urls: Optional[set] = None) -> str:
    """Return ``raw`` only if it is an http(s) URL this round actually collected.

    The LLM reads attacker-controlled text (hot-list titles, search highlights),
    so anything it echoes back as a "source" is untrusted: a prompt-injected post
    could hand us a phishing link to broadcast into a QQ group. Membership of the
    round's own URL set is the only accepted proof, and an empty/absent allowlist
    fails closed.
    """
    text = strip_unsafe_characters(raw).strip()
    if not text:
        return ""
    if not text.lower().startswith(_ALLOWED_URL_SCHEMES):
        return ""
    if not allowed_urls or text not in allowed_urls:
        return ""
    return text


def clean_candidates(
    raw_items: Iterable[Any],
    allowed_urls: Optional[set] = None,
) -> List[WordCandidate]:
    """Normalise LLM output into de-duplicated, length-checked candidates.

    Rules: strip bidi/control characters, trim, no internal whitespace, 2-8
    characters, must contain CJK, first occurrence of a word wins. ``sourceUrl``
    survives only when it exactly matches one of ``allowed_urls``; anything else
    is dropped and the candidate keeps its word but loses the link.
    """
    cleaned: List[WordCandidate] = []
    seen: set[str] = set()
    for item in raw_items or []:
        if isinstance(item, WordCandidate):
            word, reason, source_url = item.word, item.reason, item.source_url
        elif isinstance(item, dict):
            word = item.get("word")
            reason = item.get("reason")
            source_url = item.get("sourceUrl") or item.get("source_url")
        elif isinstance(item, str):
            word, reason, source_url = item, "", ""
        else:
            continue

        word = strip_unsafe_characters(word).strip()
        if not word or _WHITESPACE_RE.search(word):
            continue
        if len(word) < WORD_MIN_LENGTH or len(word) > WORD_MAX_LENGTH:
            continue
        if not contains_cjk(word):
            continue
        if word in seen:
            continue
        seen.add(word)
        cleaned.append(
            WordCandidate(
                word=word,
                reason=_clean_text(reason, 120),
                source_url=sanitize_source_url(source_url, allowed_urls),
            )
        )
        if len(cleaned) >= MAX_LLM_CANDIDATES:
            break
    return cleaned


# ---------------------------------------------------------------------------
# Local history store
# ---------------------------------------------------------------------------


def _default_discovery_db_path() -> str:
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / DISCOVERY_DB_FILENAME)


class DiscoveryStore:
    """SQLite memory of every word this pipeline has already acted on."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_discovery_db_path()
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS discovered_words (
                    word TEXT PRIMARY KEY,
                    first_seen TEXT NOT NULL,
                    last_action TEXT NOT NULL,
                    action_date TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_discovered_action_date ON discovered_words(action_date)"
            )
            # Per-day auto-ingest budget. Shared by the scheduled run and every
            # manual trigger, so "10 a day" cannot be multiplied by re-triggering.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_quota (
                    quota_key TEXT NOT NULL,
                    quota_date TEXT NOT NULL,
                    used INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (quota_key, quota_date)
                )
                """
            )
            # Half-written batches whose compensation also failed. These must be
            # visible to a human, never silently dropped.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_recovery (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    run_date TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    words TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    resolved INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # Digests that could not reach a single group. Replayed on the next
            # tick / process start so an ingest is never invisible.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notify_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    run_date TEXT NOT NULL,
                    body TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()

    # -- daily quota --------------------------------------------------------

    def consume_daily_quota(self, quota_date: str, limit: int, wanted: int) -> int:
        """Atomically reserve up to ``wanted`` units of today's budget.

        ``BEGIN IMMEDIATE`` takes the write lock before the read, so two runs
        racing on the same day cannot both observe the same ``used`` value and
        each grant themselves a full allowance.
        """
        wanted = max(0, int(wanted))
        limit = max(0, int(limit))
        if wanted <= 0 or limit <= 0:
            return 0
        try:
            conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=30.0)
        except sqlite3.Error as error:
            logger.error(f"[word_discovery] cannot open quota store: {error}")
            return 0
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT used FROM daily_quota WHERE quota_key = ? AND quota_date = ?",
                (QUOTA_KEY, quota_date),
            ).fetchone()
            used = int(row[0]) if row and row[0] is not None else 0
            granted = max(0, min(wanted, limit - used))
            if granted:
                conn.execute(
                    """
                    INSERT INTO daily_quota (quota_key, quota_date, used)
                    VALUES (?, ?, ?)
                    ON CONFLICT(quota_key, quota_date) DO UPDATE SET used = used + ?
                    """,
                    (QUOTA_KEY, quota_date, granted, granted),
                )
            conn.execute("COMMIT")
            return granted
        except sqlite3.Error as error:
            logger.error(f"[word_discovery] quota reservation failed: {error}")
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            return 0
        finally:
            conn.close()

    def refund_daily_quota(self, quota_date: str, amount: int) -> None:
        """Give back reservations that were never actually written."""
        amount = max(0, int(amount))
        if not amount:
            return
        try:
            conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=30.0)
        except sqlite3.Error as error:
            logger.warning(f"[word_discovery] cannot open quota store to refund: {error}")
            return
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE daily_quota SET used = MAX(0, used - ?)
                WHERE quota_key = ? AND quota_date = ?
                """,
                (amount, QUOTA_KEY, quota_date),
            )
            conn.execute("COMMIT")
        except sqlite3.Error as error:
            logger.warning(f"[word_discovery] quota refund failed: {error}")
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        finally:
            conn.close()

    def remaining_daily_quota(self, quota_date: str, limit: int) -> int:
        """Read-only view of what is left today (used by dry runs)."""
        limit = max(0, int(limit))
        try:
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT used FROM daily_quota WHERE quota_key = ? AND quota_date = ?",
                    (QUOTA_KEY, quota_date),
                ).fetchone()
        except sqlite3.Error as error:
            logger.warning(f"[word_discovery] cannot read quota: {error}")
            return limit
        used = int(row[0]) if row and row[0] is not None else 0
        return max(0, limit - used)

    # -- pending recovery ---------------------------------------------------

    def record_pending_recovery(
        self,
        run_date: str,
        batch_id: str,
        stage: str,
        words: Sequence[str],
        detail: str,
    ) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO pending_recovery
                        (created_at, run_date, batch_id, stage, words, detail)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().isoformat(),
                        run_date,
                        str(batch_id or ""),
                        str(stage or ""),
                        "、".join(str(word) for word in words),
                        str(detail or "")[:500],
                    ),
                )
                conn.commit()
        except sqlite3.Error as error:
            logger.error(f"[word_discovery] cannot persist pending recovery record: {error}")

    def list_pending_recovery(self) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT id, run_date, batch_id, stage, words, detail
                    FROM pending_recovery WHERE resolved = 0 ORDER BY id
                    """
                ).fetchall()
        except sqlite3.Error as error:
            logger.warning(f"[word_discovery] cannot read pending recovery records: {error}")
            return []
        return [
            {
                "id": row[0],
                "runDate": row[1],
                "batchId": row[2],
                "stage": row[3],
                "words": row[4],
                "detail": row[5],
            }
            for row in rows
        ]

    # -- notification outbox ------------------------------------------------

    def enqueue_notification(self, run_date: str, body: str) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO notify_outbox (created_at, run_date, body) VALUES (?, ?, ?)",
                    (datetime.now().isoformat(), run_date, body),
                )
                conn.commit()
        except sqlite3.Error as error:
            logger.error(f"[word_discovery] cannot enqueue undelivered digest: {error}")

    def list_pending_notifications(self, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute(
                    "SELECT id, run_date, body, attempts FROM notify_outbox ORDER BY id LIMIT ?",
                    (max(1, int(limit)),),
                ).fetchall()
        except sqlite3.Error as error:
            logger.warning(f"[word_discovery] cannot read notification outbox: {error}")
            return []
        return [{"id": row[0], "runDate": row[1], "body": row[2], "attempts": int(row[3] or 0)} for row in rows]

    def drop_notification(self, row_id: int) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM notify_outbox WHERE id = ?", (row_id,))
                conn.commit()
        except sqlite3.Error as error:
            logger.warning(f"[word_discovery] cannot drop outbox row {row_id}: {error}")

    def bump_notification_attempt(self, row_id: int) -> int:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE notify_outbox SET attempts = attempts + 1 WHERE id = ?", (row_id,)
                )
                conn.commit()
                row = conn.execute(
                    "SELECT attempts FROM notify_outbox WHERE id = ?", (row_id,)
                ).fetchone()
        except sqlite3.Error as error:
            logger.warning(f"[word_discovery] cannot bump outbox row {row_id}: {error}")
            return 0
        return int(row[0]) if row and row[0] is not None else 0

    def get_action_dates(self) -> Dict[str, str]:
        """Map every remembered word to the date it was last acted on."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("SELECT word, action_date FROM discovered_words").fetchall()
        except sqlite3.Error as error:
            logger.warning(f"[word_discovery] cannot read discovery history: {error}")
            return {}
        return {str(row[0]): str(row[1] or "") for row in rows if row and row[0]}

    def record(self, word: str, action: str, action_date: str) -> None:
        word = str(word or "").strip()
        if not word:
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO discovered_words (word, first_seen, last_action, action_date)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(word) DO UPDATE SET
                        last_action = excluded.last_action,
                        action_date = excluded.action_date
                    """,
                    (word, action_date, action, action_date),
                )
                conn.commit()
        except sqlite3.Error as error:
            logger.warning(f"[word_discovery] cannot persist discovery history for {word}: {error}")

    def record_many(self, words: Iterable[str], action: str, action_date: str) -> None:
        for word in words:
            self.record(word, action, action_date)


def _parse_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def is_recently_processed(
    action_date: Any,
    today: Any,
    days: int = HISTORY_SKIP_DAYS,
) -> bool:
    """Whether ``action_date`` falls inside the ``days``-day skip window.

    An unparseable stored date is treated as "not recent" so a corrupt row can
    never permanently block a word.
    """
    seen = _parse_date(action_date)
    reference = _parse_date(today)
    if seen is None or reference is None:
        return False
    if seen > reference:
        # Clock skew: a future stamp still means "handled", do not re-process.
        return True
    return (reference - seen) < timedelta(days=days)


def filter_recent_history(
    candidates: Sequence[WordCandidate],
    action_dates: Dict[str, str],
    today: Any,
    days: int = HISTORY_SKIP_DAYS,
) -> Tuple[List[WordCandidate], List[WordCandidate]]:
    """Split candidates into ``(fresh, skipped_because_recently_handled)``."""
    fresh: List[WordCandidate] = []
    skipped: List[WordCandidate] = []
    for candidate in candidates:
        if is_recently_processed(action_dates.get(candidate.word), today, days):
            skipped.append(candidate)
        else:
            fresh.append(candidate)
    return fresh, skipped


# ---------------------------------------------------------------------------
# Dictionary lookup
# ---------------------------------------------------------------------------


def chunk_words(words: Sequence[str], size: int = WORD_LOOKUP_BATCH_SIZE) -> List[List[str]]:
    step = max(1, size)
    return [list(words[index:index + step]) for index in range(0, len(words), step)]


async def lookup_existing_words(words: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Batch dictionary lookup, chunked to the endpoint's 500-word ceiling.

    Propagates :class:`KeytaoApiError` - callers must never read a failed lookup
    as "the dictionary does not have these words".
    """
    unique: List[str] = []
    seen: set[str] = set()
    for word in words:
        text = str(word or "").strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    if not unique:
        return {}

    config = _review_config()
    merged: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunk_words(unique):
        merged.update(await lookup_words(config, chunk))
    return merged


async def split_by_dictionary(
    candidates: Sequence[WordCandidate],
) -> Tuple[List[WordCandidate], List[WordCandidate], bool]:
    """Return ``(not_in_dictionary, already_in_dictionary, lookup_failed)``."""
    if not candidates:
        return [], [], False
    try:
        existing_map = await lookup_existing_words([item.word for item in candidates])
    except KeytaoApiError as error:
        logger.error(f"[word_discovery] dictionary batch lookup failed: {error}")
        return list(candidates), [], True
    except Exception as error:
        logger.error(f"[word_discovery] dictionary batch lookup failed: {error}")
        return list(candidates), [], True

    missing: List[WordCandidate] = []
    existing: List[WordCandidate] = []
    for candidate in candidates:
        if existing_map.get(candidate.word):
            existing.append(candidate)
        else:
            missing.append(candidate)
    return missing, existing, False


# ---------------------------------------------------------------------------
# Review
# ---------------------------------------------------------------------------


async def review_candidates(candidates: Sequence[WordCandidate]) -> List[Dict[str, Any]]:
    """Run the shared pronunciation/occupancy review over each candidate."""
    if not candidates:
        return []
    config = _review_config()
    semaphore = asyncio.Semaphore(REVIEW_CONCURRENCY)

    async def review_one(candidate: WordCandidate) -> Dict[str, Any]:
        async with semaphore:
            try:
                review = await prepare_reviewed_word(config, candidate.word)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning(f"[word_discovery] review failed for {candidate.word}: {error}")
                return {"candidate": candidate, "review": None, "error": str(error)}
        return {"candidate": candidate, "review": review, "error": ""}

    return list(await asyncio.gather(*(review_one(candidate) for candidate in candidates)))


# ---------------------------------------------------------------------------
# Classification (pure)
# ---------------------------------------------------------------------------


def _manual_block_reason(review: Optional[Dict[str, Any]], error: str) -> str:
    """Return why this candidate cannot be auto-ingested, or ``""`` when it can.

    Every unknown resolves to a reason: the only way out of this function with
    an empty string is a review that positively cleared every gate.
    """
    if error:
        return f"审核异常：{error}"
    if not isinstance(review, dict):
        return "审核未返回结果"
    if not review.get("success", False):
        return str(review.get("message") or "审核未通过")

    if review.get("lookupFailed"):
        return str(review.get("lookupFailureReason") or LOOKUP_FAILURE_REASON)
    if review.get("existing"):
        return DUPLICATE_REASON

    flag = read_manual_review_flag(review)
    if flag is True:
        return manual_review_reason(review) or str(review.get("autoReviewReason") or "需管理员审核")

    if not str(review.get("recommendedCode") or "").strip():
        return MANUAL_REASON_NO_CODE
    if not review.get("autoReviewable"):
        return str(review.get("autoReviewReason") or "未达到自动入库条件")
    # ``None`` means the review carried no structured verdict at all; without a
    # positive "no manual review needed" signal we do not auto-approve.
    if flag is None:
        return "审核结果缺少结构化判定"
    return ""


def classify_reviewed(
    reviewed: Sequence[Dict[str, Any]],
    *,
    limit: int = DEFAULT_DAILY_LIMIT,
    auto_approve: bool = True,
    auto_disabled_reason: str = MANUAL_REASON_AUTO_APPROVE_OFF,
) -> Tuple[List[ClassifiedCandidate], List[ClassifiedCandidate]]:
    """Route reviewed candidates into ``(auto_ingest, needs_human)`` groups.

    The daily limit applies to the auto group only; the overflow is recommended
    to a human rather than dropped. ``auto_disabled_reason`` is what the report
    tells the group when ``auto_approve`` is off, so the caller can say *why* it
    is off (config switch vs. a failed dictionary lookup).
    """
    auto_items: List[ClassifiedCandidate] = []
    manual_items: List[ClassifiedCandidate] = []

    for entry in reviewed:
        candidate = entry.get("candidate")
        if not isinstance(candidate, WordCandidate):
            continue
        review = entry.get("review") if isinstance(entry.get("review"), dict) else None
        error = str(entry.get("error") or "")
        code = str((review or {}).get("recommendedCode") or "").strip()

        block_reason = _manual_block_reason(review, error)
        if block_reason:
            manual_items.append(
                ClassifiedCandidate(candidate=candidate, group=GROUP_MANUAL, code=code, reason=block_reason)
            )
            continue
        if not auto_approve:
            manual_items.append(
                ClassifiedCandidate(
                    candidate=candidate,
                    group=GROUP_MANUAL,
                    code=code,
                    reason=auto_disabled_reason or MANUAL_REASON_AUTO_APPROVE_OFF,
                )
            )
            continue
        if len(auto_items) >= max(0, limit):
            manual_items.append(
                ClassifiedCandidate(
                    candidate=candidate,
                    group=GROUP_MANUAL,
                    code=code,
                    reason=MANUAL_REASON_OVER_DAILY_LIMIT,
                )
            )
            continue
        auto_items.append(
            ClassifiedCandidate(
                candidate=candidate,
                group=GROUP_AUTO,
                code=code,
                reason=str((review or {}).get("autoReviewReason") or ""),
            )
        )

    return auto_items, manual_items


def demote_to_manual(
    items: Sequence[ClassifiedCandidate],
    reason: str,
) -> List[ClassifiedCandidate]:
    """Rebuild an auto group as a recommendation group with a shared reason."""
    return [
        ClassifiedCandidate(
            candidate=item.candidate,
            group=GROUP_MANUAL,
            code=item.code,
            reason=reason,
        )
        for item in items
    ]


def demote_with_reasons(
    pairs: Sequence[Tuple[ClassifiedCandidate, str]],
) -> List[ClassifiedCandidate]:
    """Demote items that each failed for their own, server-reported reason."""
    return [
        ClassifiedCandidate(
            candidate=item.candidate,
            group=GROUP_MANUAL,
            code=item.code,
            reason=reason or MANUAL_REASON_INGEST_FAILED,
        )
        for item, reason in pairs
    ]


# ---------------------------------------------------------------------------
# Auto ingest (bot identity)
# ---------------------------------------------------------------------------

_UNSAFE_PATH_SEGMENT_RE = re.compile(r"[/\\?#%\s]")


def _safe_path_segment(value: Any) -> Optional[str]:
    """Return ``value`` usable as one URL path segment, else ``None``."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or len(text) > 128:
        return None
    if ".." in text or _UNSAFE_PATH_SEGMENT_RE.search(text):
        return None
    return text


def build_discovery_remark(run_date: str, source_url: str = "") -> str:
    """Code-generated remark. Never let the LLM author this string."""
    remark = f"{REMARK_PREFIX} {run_date}"
    url = str(source_url or "").strip()
    if url:
        remark = f"{remark} {url}"
    return remark


def build_review_note(run_date: str, count: int) -> str:
    return f"{REMARK_PREFIX} {run_date}：{count} 条每日发现词自动入库"


def build_draft_items(items: Sequence[ClassifiedCandidate], run_date: str) -> List[Dict[str, Any]]:
    """Build batch-draft payload items carrying the structured auto-pass verdict."""
    payload: List[Dict[str, Any]] = []
    for item in items:
        entry: Dict[str, Any] = {
            "action": "Create",
            "word": item.word,
            "code": item.code,
            "type": "Phrase",
            "remark": build_discovery_remark(run_date, item.source_url),
        }
        # The verdict is structured; the remark above is display only.
        apply_manual_review_flag(entry, False)
        payload.append(entry)
    return payload


def _rejection_index(response: Dict[str, Any]) -> Dict[str, str]:
    """Map word -> server-reported reason across the ``failed``/``skipped`` lists."""
    reasons: Dict[str, str] = {}
    for key in ("failed", "skipped"):
        entries = response.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if isinstance(entry, str):
                word, reason = entry.strip(), ""
            elif isinstance(entry, dict):
                word = str(entry.get("word") or "").strip()
                reason = str(
                    entry.get("reason")
                    or entry.get("message")
                    or entry.get("conflictReason")
                    or ""
                ).strip()
            else:
                continue
            if word:
                reasons[word] = _clean_text(reason, 120) or ("词库拒绝写入" if key == "failed" else "被跳过")
    return reasons


def _draft_item_index(response: Dict[str, Any]) -> Tuple[Dict[Tuple[str, str], Any], Dict[str, Any], bool]:
    """Index the returned draft snapshot by ``(word, code)`` and by word."""
    entries = response.get("draftItems")
    if not isinstance(entries, list) or not entries:
        return {}, {}, False
    by_pair: Dict[Tuple[str, str], Any] = {}
    by_word: Dict[str, Any] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        word = str(entry.get("word") or "").strip()
        if not word:
            continue
        code = str(entry.get("code") or "").strip().lower()
        item_id = entry.get("id")
        by_pair.setdefault((word, code), item_id)
        by_word.setdefault(word, item_id)
    return by_pair, by_word, True


def resolve_draft_outcome(
    requested: Sequence[ClassifiedCandidate],
    response: Dict[str, Any],
) -> Tuple[List[ClassifiedCandidate], List[Tuple[ClassifiedCandidate, str]], List[int]]:
    """Attribute a batch-draft response to individual words.

    Returns ``(accepted, [(candidate, reason)], draft_item_ids)``. A word is
    accepted only on positive evidence: it is absent from ``failed``/``skipped``
    **and** either the returned draft snapshot contains it, or the reported
    ``successCount`` exactly accounts for every remaining word. When the counts
    disagree and there is no snapshot to disambiguate, nothing is claimed - a
    partially written batch must not be reported as fully ingested.
    """
    rejected_reasons = _rejection_index(response)
    by_pair, by_word, has_snapshot = _draft_item_index(response)

    accepted: List[ClassifiedCandidate] = []
    rejected: List[Tuple[ClassifiedCandidate, str]] = []
    item_ids: List[int] = []

    for candidate in requested:
        reason = rejected_reasons.get(candidate.word)
        if reason:
            rejected.append((candidate, reason))
            continue
        if has_snapshot:
            key = (candidate.word, str(candidate.code or "").strip().lower())
            item_id = by_pair.get(key, by_word.get(candidate.word))
            if item_id is None and candidate.word not in by_word:
                rejected.append((candidate, "草稿中未找到该条目，未确认写入"))
                continue
            accepted.append(candidate)
            numeric_id = _safe_item_id(item_id)
            if numeric_id is not None:
                item_ids.append(numeric_id)
            continue
        accepted.append(candidate)

    if not has_snapshot:
        reported = response.get("successCount")
        if isinstance(reported, bool) or not isinstance(reported, int):
            reported = _as_int(reported, -1, minimum=-1, maximum=100000)
        if reported != len(accepted):
            # Partial write with no per-item evidence: refuse to guess.
            return [], [
                (candidate, f"写入结果无法逐条确认（接口报告 {reported} 条）")
                for candidate in accepted
            ] + rejected, []

    return accepted, rejected, item_ids


def _safe_item_id(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


async def _delete_draft_items(
    item_ids: Sequence[int],
    identity: Dict[str, str],
    call_kwargs: Dict[str, Any],
) -> bool:
    """Best-effort removal of draft rows this run wrote. ``True`` when clean."""
    ids = [i for i in (_safe_item_id(value) for value in item_ids) if i is not None]
    if not ids:
        return False
    try:
        data = await http_client.keytao_json(
            "DELETE",
            "/api/bot/pull-requests/batch-draft",
            json_body={**identity, "ids": ids},
            timeout=30.0,
            **call_kwargs,
        )
    except Exception as error:
        logger.error(f"[word_discovery] compensating delete failed: {error}")
        return False
    if data.get("success") is False:
        logger.error(f"[word_discovery] compensating delete rejected: {data.get('message')}")
        return False
    return True


async def _find_written_items(
    words: Sequence[str],
    identity: Dict[str, str],
    call_kwargs: Dict[str, Any],
) -> Tuple[Optional[List[int]], str]:
    """Ask the draft which of ``words`` are actually sitting in it right now.

    ``http_client`` no longer replays a write whose read timed out, because the
    server may well have applied it. That turns "the draft write raised" into a
    genuinely unknown state, and guessing either way is wrong: assume success and
    we lose words, assume failure and we leave orphan draft rows that get written
    again tomorrow. So we settle it with an idempotent GET (which *does* retry)
    and act on evidence.

    Returns ``(ids, "")`` on a successful probe - an empty list meaning nothing
    was written - or ``(None, reason)`` when the draft could not be read at all.
    """
    try:
        data = await http_client.keytao_json(
            "GET",
            "/api/bot/batches/latest-draft/items",
            params=identity,
            timeout=20.0,
            **call_kwargs,
        )
    except Exception as error:
        logger.error(f"[word_discovery] draft state probe failed: {error}")
        return None, str(error)

    entries = data.get("items")
    if not isinstance(entries, list):
        return None, "草稿条目响应格式异常"

    wanted = {str(word).strip() for word in words if str(word).strip()}
    found: List[int] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("word") or "").strip() not in wanted:
            continue
        item_id = _safe_item_id(entry.get("id"))
        if item_id is not None:
            found.append(item_id)
    return found, ""


async def _recall_batch(identity: Dict[str, str], call_kwargs: Dict[str, Any]) -> bool:
    """Revert the batch we just submitted back to draft so it can be cleaned up."""
    try:
        data = await http_client.keytao_json(
            "POST",
            "/api/bot/batches/recall",
            json_body=dict(identity),
            timeout=30.0,
            **call_kwargs,
        )
    except Exception as error:
        logger.error(f"[word_discovery] compensating recall failed: {error}")
        return False
    if data.get("success") is False:
        logger.error(f"[word_discovery] compensating recall rejected: {data.get('message')}")
        return False
    return True


async def submit_discovered_words(
    items: Sequence[ClassifiedCandidate],
    run_date: str,
    *,
    store: Optional["DiscoveryStore"] = None,
) -> Dict[str, Any]:
    """Push the auto group through draft -> submit -> auto-approve as the bot.

    Every API call retries at most once. The result reports *per word* what
    actually landed: only words the server positively confirmed are returned as
    ``accepted``; everything else comes back in ``rejected`` with a reason so the
    caller can recommend them to a human instead of burying them in history.

    When a later step fails after rows were already written, the write is undone
    (recall if needed, then delete the draft rows). If the compensation itself
    fails, a ``pending_recovery`` row is persisted and surfaced in the digest -
    the one thing this function will never do is leave a half-written batch that
    nobody hears about.
    """
    if not items:
        return {"success": True, "message": "没有需要入库的词", "count": 0, "accepted": [], "rejected": []}

    platform_id = bot_platform_id()
    if not platform_id:
        logger.error("[word_discovery] WORD_DISCOVERY_BOT_PLATFORM_ID is not configured")
        return {
            "success": False,
            "message": MANUAL_REASON_NO_PLATFORM_ID,
            "count": 0,
            "accepted": [],
            "rejected": [(item, MANUAL_REASON_NO_PLATFORM_ID) for item in items],
        }

    identity = {"platform": BOT_PLATFORM, "platformId": platform_id}
    # retries=2 caps the attempt count at "one try plus one replay". Under the
    # shared client's idempotency policy that replay only ever happens for the
    # GETs here and for a write whose connection was never established; a write
    # that timed out mid-flight is surfaced, never repeated. None of the
    # compensation below assumes a retry took place - each step is judged on the
    # state it can actually observe.
    call_kwargs = {
        "platform": BOT_PLATFORM,
        "platform_id": platform_id,
        "retries": 2,
    }

    def failure(
        message: str,
        *,
        batch_id: str = "",
        rejected: Optional[List[Tuple[ClassifiedCandidate, str]]] = None,
        pending: str = "",
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "message": message,
            "count": 0,
            "batchId": batch_id,
            "accepted": [],
            "rejected": rejected if rejected is not None else [(item, message) for item in items],
            "pendingRecovery": pending,
        }

    def leave_pending(stage: str, batch_id: str, words: Sequence[str], detail: str) -> str:
        """Persist an un-compensated half-write and return the digest note."""
        note = (
            f"批次 {batch_id or '未知'} 处于 {stage} 状态未能回滚"
            f"（{', '.join(words) or '无词条'}），需要管理员处理：{detail}"
        )
        logger.error(f"[word_discovery] PENDING RECOVERY {note}")
        if store is not None:
            store.record_pending_recovery(run_date, batch_id, stage, words, detail)
        return note

    try:
        draft = await http_client.keytao_json(
            "GET",
            "/api/bot/batches/latest-draft",
            params=identity,
            timeout=15.0,
            **call_kwargs,
        )
    except Exception as error:
        logger.error(f"[word_discovery] latest-draft failed: {error}")
        return failure(f"获取草稿批次失败：{error}")

    batch_id = str(draft.get("batchId") or "").strip()
    safe_batch_id = _safe_path_segment(batch_id)
    if not safe_batch_id:
        logger.error(f"[word_discovery] refusing unsafe or missing batch id: {batch_id!r}")
        return failure("未获得合法的草稿批次编号")

    try:
        added = await http_client.keytao_json(
            "POST",
            "/api/bot/pull-requests/batch-draft",
            json_body={
                **identity,
                "batchId": safe_batch_id,
                "confirmed": True,
                "items": build_draft_items(items, run_date),
            },
            timeout=60.0,
            **call_kwargs,
        )
    except Exception as error:
        # Not proof that nothing was written: a read timeout on a write is no
        # longer retried, so the rows may exist. Settle it by reading the draft.
        message = f"写入草稿失败：{error}"
        logger.error(f"[word_discovery] batch-draft failed: {error}")
        all_words = [item.word for item in items]
        written, probe_error = await _find_written_items(all_words, identity, call_kwargs)
        pending = ""
        if written is None:
            pending = leave_pending(
                RECOVERY_STAGE_DRAFT,
                batch_id,
                all_words,
                f"{message}；且无法确认草稿状态（{probe_error}）",
            )
        elif written and not await _delete_draft_items(written, identity, call_kwargs):
            pending = leave_pending(RECOVERY_STAGE_DRAFT, batch_id, all_words, message)
        return failure(message, batch_id=batch_id, pending=pending)

    accepted, rejected, item_ids = resolve_draft_outcome(items, added)
    reported_count = _as_int(added.get("successCount"), 0, minimum=0, maximum=100000)

    if not accepted:
        message = str(added.get("message") or "草稿写入未确认任何词条")
        pending = ""
        if reported_count > 0:
            # Rows exist but could not be attributed. Fall back to the draft
            # probe when the response carried no ids to delete.
            rollback_ids = list(item_ids)
            if not rollback_ids:
                probed, probe_error = await _find_written_items(
                    [item.word for item in items], identity, call_kwargs
                )
                if probed is None:
                    pending = leave_pending(
                        RECOVERY_STAGE_DRAFT,
                        batch_id,
                        [item.word for item in items],
                        f"草稿写入 {reported_count} 条但无法逐条确认，且无法读取草稿（{probe_error}）",
                    )
                else:
                    rollback_ids = probed
            if not pending and rollback_ids and not await _delete_draft_items(
                rollback_ids, identity, call_kwargs
            ):
                pending = leave_pending(
                    RECOVERY_STAGE_DRAFT,
                    batch_id,
                    [item.word for item in items],
                    f"草稿写入 {reported_count} 条但无法逐条确认，回滚失败",
                )
        logger.error(f"[word_discovery] batch-draft confirmed nothing: {message}")
        return failure(message, batch_id=batch_id, rejected=rejected or None, pending=pending)

    accepted_words = [item.word for item in accepted]

    try:
        submitted = await http_client.keytao_json(
            "POST",
            f"/api/bot/batches/{safe_batch_id}/submit",
            json_body={**identity, "confirmed": True},
            timeout=60.0,
            **call_kwargs,
        )
        submit_error = "" if submitted.get("success") is not False else str(
            submitted.get("message") or "提交批次被拒绝"
        )
    except Exception as error:
        submit_error = f"提交批次失败：{error}"

    if submit_error:
        logger.error(f"[word_discovery] submit failed: {submit_error}")
        pending = ""
        if not await _delete_draft_items(item_ids, identity, call_kwargs):
            pending = leave_pending(RECOVERY_STAGE_DRAFT, batch_id, accepted_words, submit_error)
        return failure(
            submit_error,
            batch_id=batch_id,
            rejected=[(item, submit_error) for item in accepted] + rejected,
            pending=pending,
        )

    review_note = build_review_note(run_date, len(accepted))
    approve_payload = apply_manual_review_flag({**identity, "reviewNote": review_note}, False)
    try:
        approved = await http_client.keytao_json(
            "POST",
            f"/api/bot/batches/{safe_batch_id}/auto-approve",
            json_body=approve_payload,
            timeout=60.0,
            **call_kwargs,
        )
        approve_error = "" if approved.get("success") is not False else str(
            approved.get("message") or "自动批准被拒绝"
        )
    except Exception as error:
        approve_error = f"自动批准失败：{error}"

    if approve_error:
        logger.error(f"[word_discovery] auto-approve failed: {approve_error}")
        # The batch is submitted and awaiting review: recall it back to draft,
        # then delete our rows. Either step failing leaves it for a human.
        pending = ""
        if not await _recall_batch(identity, call_kwargs):
            pending = leave_pending(RECOVERY_STAGE_SUBMITTED, batch_id, accepted_words, approve_error)
        elif not await _delete_draft_items(item_ids, identity, call_kwargs):
            pending = leave_pending(RECOVERY_STAGE_DRAFT, batch_id, accepted_words, approve_error)
        return failure(
            approve_error,
            batch_id=batch_id,
            rejected=[(item, approve_error) for item in accepted] + rejected,
            pending=pending,
        )

    return {
        "success": True,
        "message": "已自动入库" if not rejected else f"部分入库：{len(accepted)} 成功、{len(rejected)} 未写入",
        "count": len(accepted),
        "batchId": batch_id,
        "accepted": accepted,
        "rejected": rejected,
        "pendingRecovery": "",
    }


# ---------------------------------------------------------------------------
# Report rendering (pure)
# ---------------------------------------------------------------------------


def build_daily_report(
    run_date: str,
    auto_items: Sequence[ClassifiedCandidate],
    manual_items: Sequence[ClassifiedCandidate],
    stats: Dict[str, Any],
    *,
    source_failures: Sequence[str] = (),
    dry_run: bool = False,
) -> str:
    """Render the group digest. Always returns text, even for an empty run."""
    title = f"喵喵每日词汇发现 {run_date}"
    if dry_run:
        title += "（试运行，未写库）"
    lines: List[str] = [title, ""]

    if auto_items:
        lines.append(f"自动入库 {len(auto_items)} 个")
        for item in auto_items:
            parts = [item.word, item.code or "无编码"]
            parts.append(item.source_url or "无来源链接")
            lines.append("· " + " — ".join(parts))
        lines.append("")

    if manual_items:
        lines.append(f"待人工推荐 {len(manual_items)} 个")
        for item in manual_items:
            suggestion = f"建议编码 {item.code}" if item.code else "暂无建议编码"
            reason = item.reason or "需人工确认"
            lines.append(f"· {item.word} — {suggestion} — {reason}")
        lines.append("")

    if not auto_items and not manual_items:
        lines.append("今天没有筛出可推荐的新词。")
        lines.append("")

    lines.append(
        "统计：候选 {candidates} · 新词 {fresh} · 自动入库 {auto} · 待推荐 {manual} · "
        "词库已有 {existing} · 近期已处理 {skipped}".format(
            candidates=int(stats.get("candidates", 0)),
            fresh=int(stats.get("fresh", 0)),
            auto=len(auto_items),
            manual=len(manual_items),
            existing=int(stats.get("existing", 0)),
            skipped=int(stats.get("skipped_history", 0)),
        )
    )
    if source_failures:
        lines.append("信源异常：" + "；".join(source_failures))
    for note in report_notes(stats):
        lines.append(note)
    return "\n".join(lines).strip()


def report_notes(stats: Dict[str, Any]) -> List[str]:
    """Trailing warning lines for the digest, in a stable order."""
    notes: List[str] = []
    single = str(stats.get("note") or "").strip()
    if single:
        notes.append(single)
    for note in stats.get("notes") or []:
        text = str(note or "").strip()
        if text and text not in notes:
            notes.append(text)
    return notes


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def _notify_groups(text: str) -> Dict[str, Any]:
    """Broadcast the digest. ``configured`` tells failure apart from "no groups"."""
    from . import group_notify

    group_ids = group_notify.parse_group_ids(notify_group_ids())
    if not group_ids:
        logger.warning("[word_discovery] WORD_DISCOVERY_GROUP_IDS is empty, skip group report")
        return {"sent": [], "failed": [], "chunks": 0, "configured": False}
    result = await group_notify.send_group_notification(
        text,
        group_ids,
        log_prefix="[word_discovery]",
    )
    result["configured"] = True
    return result


def notification_reached_nobody(result: Dict[str, Any]) -> bool:
    """Whether a configured broadcast reached zero groups."""
    if not isinstance(result, dict) or not result.get("configured"):
        return False
    return not result.get("sent")


async def deliver_digest(text: str, run_date: str, store: "DiscoveryStore") -> Dict[str, Any]:
    """Send the digest, queueing it for retry when it reached nobody.

    Words may already be in the dictionary by this point, so a broadcast that
    silently fails is a real loss of information, not a cosmetic one.
    """
    result = await _notify_groups(text)
    if notification_reached_nobody(result):
        logger.error(
            f"[word_discovery] digest for {run_date} reached no group "
            f"(failed={result.get('failed')}), queued for retry"
        )
        store.enqueue_notification(run_date, text)
    return result


async def flush_pending_notifications(store: Optional["DiscoveryStore"] = None) -> int:
    """Replay queued digests. Returns how many were delivered."""
    discovery_store = store if store is not None else DiscoveryStore()
    pending = discovery_store.list_pending_notifications()
    delivered = 0
    for row in pending:
        result = await _notify_groups(row["body"])
        if not result.get("configured"):
            # No groups configured right now; keep the row for later.
            break
        if result.get("sent"):
            discovery_store.drop_notification(row["id"])
            delivered += 1
            logger.info(f"[word_discovery] replayed queued digest from {row['runDate']}")
            continue
        attempts = discovery_store.bump_notification_attempt(row["id"])
        if attempts >= MAX_NOTIFY_ATTEMPTS:
            logger.error(
                f"[word_discovery] giving up on queued digest from {row['runDate']} "
                f"after {attempts} attempts"
            )
            discovery_store.drop_notification(row["id"])
    return delivered


async def run_word_discovery(
    *,
    dry_run: bool = False,
    notify: Optional[bool] = None,
    store: Optional[DiscoveryStore] = None,
    today: Optional[str] = None,
) -> DiscoveryResult:
    """Run one full discovery round, serialised against every other round.

    ``dry_run`` stops right after classification: nothing is written to the
    dictionary, no daily quota is consumed and nothing is recorded in the local
    history, so the caller can preview exactly what a real run would do.
    """
    # The scheduled run and every manual trigger share this lock, so two rounds
    # can never interleave their quota reservations or their draft writes.
    async with pipeline_lock():
        return await _run_word_discovery_locked(
            dry_run=dry_run, notify=notify, store=store, today=today
        )


async def _run_word_discovery_locked(
    *,
    dry_run: bool = False,
    notify: Optional[bool] = None,
    store: Optional[DiscoveryStore] = None,
    today: Optional[str] = None,
) -> DiscoveryResult:
    if notify is None:
        notify = not dry_run

    run_date = today or datetime.now().strftime("%Y-%m-%d")
    result = DiscoveryResult(run_date=run_date, dry_run=dry_run)
    discovery_store = store if store is not None else DiscoveryStore()

    if not dry_run:
        try:
            await flush_pending_notifications(discovery_store)
        except Exception as error:
            logger.warning(f"[word_discovery] replaying queued digests failed: {error}")

    documents, failures = await collect_sources()
    result.source_failures = failures
    logger.info(
        f"[word_discovery] collected {len(documents)} documents "
        f"({len(failures)} source failure(s))"
    )

    candidates = await extract_word_candidates(documents)
    logger.info(f"[word_discovery] LLM produced {len(candidates)} cleaned candidates")

    action_dates = discovery_store.get_action_dates()
    fresh, skipped = filter_recent_history(candidates, action_dates, run_date)
    result.skipped_history = [item.word for item in skipped]

    missing, existing, lookup_failed = await split_by_dictionary(fresh)
    result.existing_words = [item.word for item in existing]

    reviewed = await review_candidates(missing)
    # A failed dictionary lookup cannot tell duplicates from new words, so the
    # whole round drops to recommend-only regardless of the config switch.
    auto_approve = auto_approve_enabled() and not lookup_failed
    # Classify without a cap first: the daily budget is reserved atomically from
    # SQLite below, so the cap must be applied against what the reservation
    # actually granted, not against a value read before other runs raced us.
    eligible, manual_items = classify_reviewed(
        reviewed,
        limit=len(reviewed),
        auto_approve=auto_approve,
        auto_disabled_reason=(
            MANUAL_REASON_DICTIONARY_LOOKUP_FAILED
            if lookup_failed
            else MANUAL_REASON_AUTO_APPROVE_OFF
        ),
    )

    stats: Dict[str, Any] = {
        "candidates": len(candidates),
        "fresh": len(fresh),
        "existing": len(existing),
        "skipped_history": len(skipped),
        "documents": len(documents),
        "notes": [],
    }
    if lookup_failed:
        stats["notes"].append("词库批量查重失败，本轮全部转为人工推荐。")

    limit = daily_limit()
    if dry_run:
        granted = min(len(eligible), discovery_store.remaining_daily_quota(run_date, limit))
    else:
        granted = discovery_store.consume_daily_quota(run_date, limit, len(eligible))

    auto_items = eligible[:granted]
    if len(eligible) > granted:
        manual_items = demote_to_manual(eligible[granted:], MANUAL_REASON_OVER_DAILY_LIMIT) + manual_items
    result.quota = {"limit": limit, "granted": granted, "eligible": len(eligible)}

    if not dry_run and auto_items:
        ingest = await submit_discovered_words(auto_items, run_date, store=discovery_store)
        result.ingest = ingest
        accepted = [item for item in ingest.get("accepted") or [] if isinstance(item, ClassifiedCandidate)]
        rejected = [
            pair for pair in ingest.get("rejected") or []
            if isinstance(pair, tuple) and isinstance(pair[0], ClassifiedCandidate)
        ]
        if not ingest.get("success"):
            logger.error(f"[word_discovery] auto ingest failed: {ingest.get('message')}")
            stats["notes"].append(f"自动入库失败（{ingest.get('message')}），相关词条已转人工推荐。")
        elif rejected:
            logger.warning(
                f"[word_discovery] partial ingest: {len(accepted)} written, {len(rejected)} rejected"
            )
            stats["notes"].append(f"部分入库：{len(rejected)} 个词未写入，已转人工推荐。")

        # Anything the server did not positively confirm goes back to humans,
        # and its reserved quota is returned so a failure cannot eat the budget.
        demoted = demote_with_reasons(rejected) if rejected else []
        unaccounted = [
            item for item in auto_items
            if item.word not in {i.word for i in accepted} and item.word not in {p[0].word for p in rejected}
        ]
        if unaccounted:
            demoted += demote_to_manual(unaccounted, MANUAL_REASON_INGEST_FAILED)
        if demoted:
            manual_items = demoted + manual_items
        discovery_store.refund_daily_quota(run_date, len(auto_items) - len(accepted))
        auto_items = accepted

        pending = str(ingest.get("pendingRecovery") or "").strip()
        if pending:
            stats["notes"].append(f"需要人工处理：{pending}")

    result.auto_items = auto_items
    result.manual_items = manual_items
    result.stats = stats
    result.report = build_daily_report(
        run_date,
        auto_items,
        manual_items,
        stats,
        source_failures=failures,
        dry_run=dry_run,
    )

    if not dry_run:
        discovery_store.record_many([item.word for item in existing], ACTION_ALREADY_EXISTS, run_date)
        discovery_store.record_many([item.word for item in auto_items], ACTION_AUTO_ADDED, run_date)
        discovery_store.record_many([item.word for item in manual_items], ACTION_RECOMMENDED, run_date)

    if notify:
        result.notify = await deliver_digest(result.report, run_date, discovery_store)

    return result
