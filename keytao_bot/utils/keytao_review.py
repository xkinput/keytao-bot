"""Pronunciation-backed review helpers for KeyTao draft automation."""
from __future__ import annotations

import asyncio
import html
import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx
from nonebot import get_driver
from nonebot.log import logger

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - optional dependency guard
    AsyncOpenAI = None  # type: ignore

from .keytao_encoding import (
    build_phrase_code_chain,
    normalize_contextual_phrase_encoding,
    pinyin_to_phonetic_code,
)


SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"
BING_ENDPOINT = "https://www.bing.com/search"
SO360_ENDPOINT = "https://www.so.com/s"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

REVIEW_SIGNAL_WEIGHTS = {
    "corpus": 0.45,
    "search": 0.25,
    "dictionary": 0.20,
    "encyclopedia": 0.10,
}

COMMONNESS_SIGNAL_WEIGHTS = {
    "corpus": 0.45,
    "search": 0.25,
    "dictionary": 0.20,
    "encyclopedia": 0.10,
}

COMMONNESS_SEARCH_QUERIES = [
    ('"{word}"', "search"),
    ('"{word}" 现代汉语', "corpus"),
    ('"{word}" 语料库', "corpus"),
    ('"{word}" 词典 OR 辞典', "dictionary"),
    ('"{word}" 百度百科 OR 维基百科', "encyclopedia"),
]
CODE_CHAIN_PRIORITY_WINDOW_AFTER = 2
CODE_CHAIN_PRIORITY_MAX_OCCUPANTS = 8
CODE_CHAIN_REORDER_SCORE_MARGIN = 0.20
PERSON_ALIAS_SEARCH_QUERIES = [
    '"{word}" "字"',
    '"{word}" "号"',
    '"{word}" "别名"',
    '"{word}" "又名"',
    '"{word}" "名将"',
    '"{word}" "历史人物"',
]
PERSON_ALIAS_HINTS = (
    "字",
    "号",
    "别名",
    "又名",
    "又称",
    "人称",
    "名将",
    "历史人物",
    "人物",
    "传",
    "门神",
    "隋末",
    "唐初",
)
ENTITY_TYPE_HINTS = {
    "common_word": ("词典", "现代汉语", "意思", "读音"),
    "idiom": ("成语", "典故", "出处", "读音"),
    "person": ("人物", "简介", "百度百科", "维基百科"),
    "celebrity": ("明星", "演员", "歌手", "艺人", "百度百科"),
    "historical_person": ("历史人物", "名将", "传", "百度百科"),
    "courtesy_name": ("字", "号", "别名", "历史人物", "名将"),
    "stage_name": ("艺名", "原名", "明星", "歌手", "演员"),
    "fictional_character": ("角色", "人物", "作品", "动漫", "游戏"),
    "brand": ("品牌", "官网", "公司", "百科"),
    "product": ("产品", "品牌", "官网", "百科"),
    "place": ("地名", "城市", "景点", "行政区", "百科"),
    "organization": ("机构", "公司", "组织", "官网", "百科"),
    "work": ("作品", "电影", "电视剧", "小说", "歌曲", "百科"),
    "technical_term": ("术语", "百科", "定义", "读音"),
}
ENTITY_ACCEPTED_TYPES = set(ENTITY_TYPE_HINTS)
COMMON_KNOWN_MIN_SCORE = 0.55
COMMON_KNOWN_MIN_ACTIVE_SIGNALS = 2
COMMON_KNOWN_RELAXED_MIN_SCORE = 0.35
CSS_REVIEW_TYPES = {"CSS", "CSSSingle"}
PRONUNCIATION_EVIDENCE_TIMEOUT = 4.0
ENTITY_DIRECT_FETCH_TIMEOUT = 3.0
ENTITY_PRONUNCIATION_MIN_CONFIDENCE = 0.75
CONTEXT_ENTITY_SOURCE_DOMAINS = ("baike.baidu.com", "zh.wikipedia.org")

AUTHORITATIVE_SOURCES = [
    {
        "id": "handian",
        "label": "汉典",
        "domain": "zdic.net",
        "category": "dictionary",
        "trust": 5,
        "query": 'site:zdic.net "{word}" 拼音',
        "direct_urls": [
            "https://www.zdic.net/hans/{word}",
        ],
    },
    {
        "id": "moedict",
        "label": "萌典",
        "domain": "moedict.tw",
        "category": "dictionary",
        "trust": 5,
        "query": 'site:moedict.tw "{word}" 拼音',
        "direct_urls": [
            "https://www.moedict.tw/{word}",
        ],
    },
    {
        "id": "baidu_baike",
        "label": "百度百科",
        "domain": "baike.baidu.com",
        "category": "encyclopedia",
        "trust": 4,
        "query": 'site:baike.baidu.com "{word}" 拼音',
        "direct_urls": [
            "https://baike.baidu.com/item/{word}",
        ],
    },
    {
        "id": "wikipedia",
        "label": "维基百科",
        "domain": "zh.wikipedia.org",
        "category": "encyclopedia",
        "trust": 4,
        "query": 'site:zh.wikipedia.org "{word}" 拼音 OR pinyin',
        "direct_urls": [
            "https://zh.wikipedia.org/wiki/{word}",
        ],
    },
    {
        "id": "cidian",
        "label": "汉语辞典",
        "domain": "cidian.qianp.com",
        "category": "dictionary",
        "trust": 3,
        "query": 'site:cidian.qianp.com "{word}" 拼音',
        "direct_urls": [],
    },
]

_PINYIN_CHAR_CLASS = (
    "A-Za-z"
    "üÜvV:"
    "āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜ"
    "ĀÁǍÀŌÓǑÒĒÉĚÈĪÍǏÌŪÚǓÙǕǗǙǛ"
    "ńňǹḿ"
    "012345"
)
_PINYIN_TOKEN_RE = re.compile(rf"^[{_PINYIN_CHAR_CLASS}]+$")
_PINYIN_LABEL_RE = re.compile(
    rf"(?:拼音|讀音|读音|汉语拼音|漢語拼音|pinyin)\s*[:：]?\s*"
    rf"[\[【（(]?\s*([{_PINYIN_CHAR_CLASS}\s·,，、/\\-]{{1,120}})",
    re.IGNORECASE,
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_CJK_WORD_RE = re.compile(r"^[\u3400-\u9fff]+$")


@dataclass(frozen=True)
class ReviewHttpConfig:
    api_base: str
    bot_token: str


def _config_value(name: str, env_name: str, default: Any = None) -> Any:
    try:
        value = getattr(get_driver().config, name, None)
        if value not in (None, ""):
            return value
    except Exception:
        pass
    return os.getenv(env_name, default)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_json_object_from_model_text(content: str) -> Dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def _review_llm_config() -> Dict[str, Any]:
    timeout_value = (
        _config_value("openai_timeout", "OPENAI_TIMEOUT", None)
        or _config_value("gemini_timeout", "GEMINI_TIMEOUT", None)
        or _config_value("ark_timeout", "ARK_TIMEOUT", None)
        or 20
    )
    return {
        "api_key": str(_config_value("openai_api_key", "OPENAI_API_KEY", "") or ""),
        "base_url": str(_config_value("openai_base_url", "OPENAI_BASE_URL", "https://api.deepseek.com") or ""),
        "model": str(
            _config_value("keytao_review_model", "KEYTAO_REVIEW_MODEL", "")
            or _config_value("openai_model", "OPENAI_MODEL", "deepseek-chat")
        ),
        "timeout": min(_as_float(timeout_value, 20.0), 30.0),
    }


def normalize_pinyin_syllable(value: str) -> str:
    text = (
        value.strip().lower()
        .replace("u:", "v")
        .translate(str.maketrans("üǖǘǚǜ", "vvvvv"))
    )
    text = re.sub(r"[1-5]$", "", text)
    normalized = unicodedata.normalize("NFD", text)
    return "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    ).replace("ê", "e")


def normalize_pinyin_sequence(value: str) -> Tuple[str, ...]:
    cleaned = (
        value.replace("·", " ")
        .replace("/", " ")
        .replace("\\", " ")
        .replace("-", " ")
        .replace(",", " ")
        .replace("，", " ")
        .replace("、", " ")
    )
    result: List[str] = []
    for token in cleaned.split():
        token = token.strip("[]【】()（）;；。.:：")
        if not token or not _PINYIN_TOKEN_RE.match(token):
            continue
        normalized = normalize_pinyin_syllable(token)
        if normalized:
            result.append(normalized)
    return tuple(result)


def pinyin_sequence_label(sequence: Sequence[str]) -> str:
    return " ".join(sequence)


def _strip_tags(value: str) -> str:
    text = _SCRIPT_STYLE_RE.sub(" ", value)
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_result_url(raw_url: str) -> str:
    decoded = html.unescape(raw_url)
    parsed = urlparse(decoded)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        query = parse_qs(parsed.query)
        uddg = query.get("uddg")
        if uddg:
            return unquote(uddg[0])
    return decoded


def _extract_search_results(content: str, max_results: int) -> List[Dict[str, str]]:
    anchors = list(
        re.finditer(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            content,
            re.IGNORECASE | re.DOTALL,
        )
    )
    results: List[Dict[str, str]] = []
    for index, match in enumerate(anchors[:max_results]):
        nearby_start = match.end()
        nearby_end = anchors[index + 1].start() if index + 1 < len(anchors) else min(len(content), nearby_start + 2000)
        nearby_html = content[nearby_start:nearby_end]
        snippet_match = re.search(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(.*?)</div>',
            nearby_html,
            re.IGNORECASE | re.DOTALL,
        )
        snippet_html = ""
        if snippet_match:
            snippet_html = snippet_match.group(1) or snippet_match.group(2) or ""
        title = _strip_tags(match.group(2))
        url = _normalize_result_url(match.group(1))
        snippet = _strip_tags(snippet_html)
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


def _dedupe_search_results(results: List[Dict[str, str]], max_results: int) -> List[Dict[str, str]]:
    deduped: List[Dict[str, str]] = []
    seen: set[str] = set()
    for item in results:
        url = _normalize_result_url(str(item.get("url") or "")).strip()
        title = str(item.get("title") or "").strip()
        if not url or not title:
            continue
        parsed = urlparse(url)
        key = parsed._replace(fragment="", query=parsed.query[:160]).geturl()
        if key in seen:
            continue
        seen.add(key)
        deduped.append({
            "title": title[:180],
            "url": url,
            "snippet": str(item.get("snippet") or "").strip()[:360],
            "provider": str(item.get("provider") or "").strip(),
        })
        if len(deduped) >= max_results:
            break
    return deduped


def _extract_duckduckgo_lite_results(content: str, max_results: int) -> List[Dict[str, str]]:
    matches = list(
        re.finditer(
            r"<a[^>]+class=['\"]result-link['\"][^>]+href=['\"]([^'\"]+)['\"][^>]*>(.*?)</a>",
            content,
            re.IGNORECASE | re.DOTALL,
        )
    )
    snippets = list(
        re.finditer(
            r"<td[^>]+class=['\"]result-snippet['\"][^>]*>(.*?)</td>",
            content,
            re.IGNORECASE | re.DOTALL,
        )
    )
    results: List[Dict[str, str]] = []
    for index, match in enumerate(matches[:max_results]):
        snippet = snippets[index].group(1) if index < len(snippets) else ""
        results.append({
            "title": _strip_tags(match.group(2)),
            "url": _normalize_result_url(match.group(1)),
            "snippet": _strip_tags(snippet),
            "provider": "duckduckgo-lite",
        })
    return _dedupe_search_results(results, max_results)


def _extract_bing_results(content: str, max_results: int) -> List[Dict[str, str]]:
    matches = list(re.finditer(
        r"<h2[^>]*>.*?<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>.*?</h2>",
        content,
        re.IGNORECASE | re.DOTALL,
    ))
    results: List[Dict[str, str]] = []
    for index, match in enumerate(matches[:max_results * 3]):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else min(len(content), start + 2600)
        nearby_html = content[start:end]
        snippet_match = re.search(r"<p[^>]*>(.*?)</p>", nearby_html, re.IGNORECASE | re.DOTALL)
        results.append({
            "title": _strip_tags(match.group(2)),
            "url": _normalize_result_url(match.group(1)),
            "snippet": _strip_tags(snippet_match.group(1) if snippet_match else ""),
            "provider": "bing",
        })
        if len(results) >= max_results:
            break
    return _dedupe_search_results(results, max_results)


def _extract_so360_results(content: str, max_results: int) -> List[Dict[str, str]]:
    blocks = re.findall(
        r'<li[^>]+class="res-list"[^>]*>(.*?)</li>',
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    results: List[Dict[str, str]] = []
    for block in blocks[:max_results * 2]:
        link_match = re.search(r"<h3[^>]*>.*?<a([^>]*)>(.*?)</a>.*?</h3>", block, re.IGNORECASE | re.DOTALL)
        if not link_match:
            continue
        attrs = link_match.group(1)
        href_match = re.search(r'href=["\']([^"\']+)["\']', attrs, re.IGNORECASE)
        mdurl_match = re.search(r'data-mdurl=["\']([^"\']+)["\']', attrs, re.IGNORECASE)
        snippet_match = re.search(
            r'<p[^>]+class=["\']res-desc["\'][^>]*>(.*?)</p>|<span[^>]+class=["\']res-list-summary["\'][^>]*>(.*?)</span>',
            block,
            re.IGNORECASE | re.DOTALL,
        )
        url = html.unescape(mdurl_match.group(1)) if mdurl_match else _normalize_result_url(href_match.group(1) if href_match else "")
        results.append({
            "title": _strip_tags(link_match.group(2)),
            "url": url,
            "snippet": _strip_tags((snippet_match.group(1) or snippet_match.group(2)) if snippet_match else ""),
            "provider": "so360",
        })
        if len(results) >= max_results:
            break
    return _dedupe_search_results(results, max_results)


async def _search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    query = query.strip()
    if not query:
        return []

    providers = (
        ("so360", SO360_ENDPOINT, {"q": query}, _extract_so360_results),
        ("bing", BING_ENDPOINT, {"q": query, "setlang": "zh-CN"}, _extract_bing_results),
        ("duckduckgo-html", SEARCH_ENDPOINT, {"q": query, "kl": "cn-zh"}, _extract_search_results),
        ("duckduckgo-lite", DUCKDUCKGO_LITE_ENDPOINT, {"q": query, "kl": "cn-zh"}, _extract_duckduckgo_lite_results),
    )
    merged: List[Dict[str, str]] = []
    failures: List[str] = []
    try:
        async with httpx.AsyncClient(
            timeout=12.0,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
            follow_redirects=True,
        ) as client:
            for provider, endpoint, params, extractor in providers:
                try:
                    response = await client.get(endpoint, params=params)
                    response.raise_for_status()
                    results = extractor(response.text, max_results)
                    for result in results:
                        result.setdefault("provider", provider)
                    merged = _dedupe_search_results(merged + results, max_results)
                    if len(merged) >= max_results:
                        break
                except Exception as error:
                    failures.append(f"{provider}: {error}")
                    logger.debug(f"Review search provider {provider} failed for {query}: {error}")
        if not merged and failures:
            logger.debug(f"Review search returned no results for {query}; provider failures: {'; '.join(failures)}")
        return merged
    except Exception as error:
        logger.warning(f"Review search failed for {query}: {error}")
        return []


async def _fetch_text(url: str) -> str:
    try:
        async with httpx.AsyncClient(
            timeout=12.0,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
        return _strip_tags(response.text[:150000])
    except Exception as error:
        logger.debug(f"Review page fetch failed for {url}: {error}")
        return ""


def _source_by_id(source_id: str) -> Dict[str, Any]:
    for source in AUTHORITATIVE_SOURCES:
        if source["id"] == source_id:
            return source
    return {}


def _extract_labeled_pinyin_sequences(text: str, word_length: int) -> List[Tuple[str, ...]]:
    sequences: List[Tuple[str, ...]] = []
    seen: set[Tuple[str, ...]] = set()
    for match in _PINYIN_LABEL_RE.finditer(text):
        raw = match.group(1)
        raw = re.split(r"(?:释义|解释|词语|出处|英文|繁体|注音|词性|意思|基本)", raw, maxsplit=1)[0]
        sequence = normalize_pinyin_sequence(raw)
        if not sequence:
            continue
        if word_length > 1 and len(sequence) != word_length:
            continue
        if word_length == 1 and len(sequence) != 1:
            continue
        if sequence not in seen:
            seen.add(sequence)
            sequences.append(sequence)
    return sequences


async def collect_pronunciation_evidence(word: str) -> Dict[str, Any]:
    word = word.strip()
    if not word:
        return {"success": False, "message": "词不能为空", "groups": [], "sources": []}

    source_entries: List[Dict[str, Any]] = []

    async def inspect_source(source: Dict[str, Any]) -> None:
        texts: List[Tuple[str, str, str]] = []
        for url_template in source.get("direct_urls", []):
            url = url_template.format(word=quote(word))
            text = await _fetch_text(url)
            if text:
                texts.append((source["label"], url, text[:12000]))

        results = await _search_web(source["query"].format(word=word), max_results=2)
        for result in results:
            parsed = urlparse(result.get("url", ""))
            if source["domain"] not in parsed.netloc:
                continue
            combined = f"{result.get('title', '')} {result.get('snippet', '')}"
            texts.append((source["label"], result.get("url", ""), combined))
            page_text = await _fetch_text(result.get("url", ""))
            if page_text:
                texts.append((source["label"], result.get("url", ""), page_text[:12000]))

        for label, url, text in texts:
            for sequence in _extract_labeled_pinyin_sequences(text, len(word)):
                source_entries.append({
                    "sourceId": source["id"],
                    "source": label,
                    "url": url,
                    "pinyin": pinyin_sequence_label(sequence),
                    "normalized": list(sequence),
                    "category": source["category"],
                    "trust": source["trust"],
                })

    await asyncio.gather(*(inspect_source(source) for source in AUTHORITATIVE_SOURCES))

    groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    for entry in source_entries:
        key = tuple(entry["normalized"])
        if key not in groups:
            groups[key] = {
                "pinyin": pinyin_sequence_label(key),
                "normalized": list(key),
                "sources": [],
                "sourceIds": [],
                "score": 0,
            }
        group = groups[key]
        group["sources"].append({
            "source": entry["source"],
            "url": entry["url"],
            "category": entry["category"],
            "trust": entry["trust"],
        })
        if entry["sourceId"] not in group["sourceIds"]:
            group["sourceIds"].append(entry["sourceId"])
            group["score"] += int(entry["trust"])

    sorted_groups = sorted(groups.values(), key=lambda item: (-item["score"], item["pinyin"]))
    return {
        "success": True,
        "word": word,
        "groups": sorted_groups,
        "sources": source_entries,
        "hasEvidence": bool(sorted_groups),
    }


async def collect_pronunciation_evidence_limited(word: str) -> Dict[str, Any]:
    try:
        return await asyncio.wait_for(
            collect_pronunciation_evidence(word),
            timeout=PRONUNCIATION_EVIDENCE_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.debug(f"Pronunciation evidence lookup timed out for {word}")
        return {
            "success": False,
            "word": word,
            "message": "权威读音搜索超时，已使用编码服务默认读音",
            "groups": [],
            "sources": [],
            "timeout": True,
        }


async def _call_keytao_api(config: ReviewHttpConfig, path: str, payload: Optional[Dict] = None, method: str = "POST") -> Dict:
    if not config.bot_token:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}
    url = f"{config.api_base}{path}"
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            if method == "GET":
                response = await client.get(url, params=payload, headers={"X-Bot-Token": config.bot_token})
            else:
                response = await client.post(
                    url,
                    json=payload or {},
                    headers={"X-Bot-Token": config.bot_token, "Content-Type": "application/json"},
                )
        data = response.json()
        if response.is_success:
            return data
        return {"success": False, "message": data.get("message") or data.get("error") or f"HTTP {response.status_code}"}
    except Exception as error:
        return {"success": False, "message": str(error)}


async def fetch_keytao_encode(config: ReviewHttpConfig, word: str) -> Dict:
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(f"{config.api_base}/api/phrases/encode", params={"word": word})
            if not response.is_success:
                return {"success": False, "message": f"编码服务返回错误: {response.status_code}"}
            data = response.json()
        return normalize_contextual_phrase_encoding(word, data)
    except Exception as error:
        return {"success": False, "message": f"编码服务暂时不可用: {error}"}


async def lookup_codes(config: ReviewHttpConfig, codes: Sequence[str]) -> Dict[str, List[Dict]]:
    unique_codes = []
    seen = set()
    for code in codes:
        normalized = str(code or "").strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_codes.append(normalized)
    if not unique_codes:
        return {}
    data = await _call_keytao_api(config, "/api/bot/phrases/by-code/batch", {"codes": unique_codes})
    result: Dict[str, List[Dict]] = {}
    if not data.get("success"):
        return result
    for item in data.get("results", []):
        if isinstance(item, dict):
            result[str(item.get("code") or "")] = [
                phrase for phrase in item.get("phrases", [])
                if isinstance(phrase, dict)
            ]
    return result


async def lookup_words(config: ReviewHttpConfig, words: Sequence[str]) -> Dict[str, List[Dict]]:
    unique_words = []
    seen = set()
    for word in words:
        normalized = str(word or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_words.append(normalized)
    if not unique_words:
        return {}
    data = await _call_keytao_api(config, "/api/bot/phrases/by-word/batch", {"words": unique_words})
    result: Dict[str, List[Dict]] = {}
    if not data.get("success"):
        return result
    for item in data.get("results", []):
        if isinstance(item, dict):
            result[str(item.get("word") or "")] = [
                phrase for phrase in item.get("phrases", [])
                if isinstance(phrase, dict)
            ]
    return result


def _encode_default_pinyin_sequence(encode_data: Dict) -> Tuple[str, ...]:
    chars = encode_data.get("chars")
    if not isinstance(chars, list):
        return ()
    result: List[str] = []
    for item in chars:
        if not isinstance(item, dict):
            return ()
        pinyin = str(item.get("pinyin") or "").strip()
        normalized = normalize_pinyin_syllable(pinyin)
        if not normalized:
            return ()
        result.append(normalized)
    return tuple(result)


def _entity_pronunciation_group(
    word: str,
    entity: Dict[str, Any],
    default_sequence: Sequence[str],
) -> Optional[Dict[str, Any]]:
    """Build a context-aware pronunciation only from high-confidence entity knowledge."""
    if not entity.get("recognized"):
        return None
    if float(entity.get("confidence") or 0.0) < ENTITY_PRONUNCIATION_MIN_CONFIDENCE:
        return None

    sequence = normalize_pinyin_sequence(str(entity.get("pinyin") or ""))
    if len(sequence) != len(word):
        return None
    if any(not pinyin_to_phonetic_code(syllable) for syllable in sequence):
        return None

    entity_type = str(entity.get("entityType") or "unclear")
    label = _entity_type_label(entity_type)
    normalized_default = tuple(default_sequence)
    corrected = bool(normalized_default and sequence != normalized_default)
    return {
        "pinyin": pinyin_sequence_label(sequence),
        "normalized": list(sequence),
        "sources": [],
        "sourceIds": [],
        "score": 0,
        "fallback": True,
        "semanticPronunciation": True,
        "sourceSummary": f"本喵实体语境判断（{label}，暂无权威页）",
        "contextPronunciation": {
            "entityType": entity_type,
            "label": label,
            "confidence": float(entity.get("confidence") or 0.0),
            "description": str(entity.get("description") or "").strip(),
            "correctedDefault": corrected,
            "defaultPinyin": pinyin_sequence_label(normalized_default),
        },
    }


def _context_entity_name(word: str, result: Dict[str, str]) -> str:
    parsed = urlparse(str(result.get("url") or ""))
    if not any(domain in parsed.netloc for domain in CONTEXT_ENTITY_SOURCE_DOMAINS):
        return ""

    title_head = re.split(r"[（(_|｜-]", str(result.get("title") or ""), maxsplit=1)[0]
    normalized = re.sub(r"\s+", "", title_head)
    match = re.search(r"[\u3400-\u9fff]+", normalized)
    candidate = match.group(0) if match else ""
    if not candidate.startswith(word) or len(candidate) <= len(word) or len(candidate) > len(word) + 8:
        return ""
    return candidate


async def _contextual_pronunciation_group(
    config: ReviewHttpConfig,
    word: str,
    entity: Dict[str, Any],
    default_sequence: Sequence[str],
) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[str, str]] = []
    seen: set[str] = set()

    if (
        entity.get("recognized")
        and float(entity.get("confidence") or 0.0) >= ENTITY_PRONUNCIATION_MIN_CONFIDENCE
    ):
        for name in [*(entity.get("canonicalNames") or []), *(entity.get("aliases") or [])]:
            candidate = str(name or "").strip()
            if candidate.startswith(word) and len(candidate) > len(word) and candidate not in seen:
                seen.add(candidate)
                candidates.append((candidate, "本喵实体识别"))

    search_results = await _search_web(f'"{word}" 百度百科 OR 维基百科', max_results=5)
    for result in search_results:
        candidate = _context_entity_name(word, result)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        candidates.append((candidate, str(result.get("url") or "百科搜索结果")))

    if not candidates:
        return None

    encoded = await asyncio.gather(*(
        fetch_keytao_encode(config, candidate)
        for candidate, _source in candidates[:4]
    ))
    sequence_sources: Dict[Tuple[str, ...], List[Tuple[str, str]]] = {}
    for (candidate, source), encode_data in zip(candidates[:4], encoded):
        sequence = _encode_default_pinyin_sequence(encode_data)[:len(word)]
        if len(sequence) != len(word):
            continue
        if any(not pinyin_to_phonetic_code(syllable) for syllable in sequence):
            continue
        sequence_sources.setdefault(sequence, []).append((candidate, source))

    if len(sequence_sources) != 1:
        return None

    sequence, supporting_names = next(iter(sequence_sources.items()))
    canonical_name, source = supporting_names[0]
    normalized_default = tuple(default_sequence)
    return {
        "pinyin": pinyin_sequence_label(sequence),
        "normalized": list(sequence),
        "sources": [],
        "sourceIds": [],
        "score": 0,
        "fallback": True,
        "semanticPronunciation": True,
        "sourceSummary": f"百科实体全称语境（{canonical_name}，暂无独立读音页）",
        "contextPronunciation": {
            "entityType": str(entity.get("entityType") or "unclear"),
            "label": _entity_type_label(str(entity.get("entityType") or "unclear")),
            "confidence": float(entity.get("confidence") or 0.0),
            "description": str(entity.get("description") or "").strip(),
            "correctedDefault": bool(normalized_default and sequence != normalized_default),
            "defaultPinyin": pinyin_sequence_label(normalized_default),
            "canonicalName": canonical_name,
            "source": source,
            "method": "entity_full_name_context",
        },
    }


def _codes_for_pinyin_sequence(encode_data: Dict, sequence: Sequence[str]) -> List[str]:
    chars = encode_data.get("chars")
    if not isinstance(chars, list) or len(chars) != len(sequence):
        return []

    normalized_sequence = tuple(normalize_pinyin_syllable(str(item)) for item in sequence)
    default_sequence = _encode_default_pinyin_sequence(encode_data)
    if normalized_sequence == default_sequence:
        service_codes: List[str] = []
        for code in [*(encode_data.get("codes") or []), *(encode_data.get("altCodes") or [])]:
            normalized_code = str(code or "").strip().lower()
            if normalized_code and normalized_code not in service_codes:
                service_codes.append(normalized_code)
        if service_codes:
            return service_codes

    phonetic_codes: List[str] = []
    for index, syllable in enumerate(normalized_sequence):
        char_info = chars[index] if isinstance(chars[index], dict) else {}
        service_phonetic = str(char_info.get("phoneticCode") or "").strip().lower()
        if index < len(default_sequence) and syllable == default_sequence[index] and service_phonetic:
            phonetic_codes.append(service_phonetic)
        else:
            phonetic_codes.append(pinyin_to_phonetic_code(syllable) or "")
    if any(not code for code in phonetic_codes):
        return []
    return build_phrase_code_chain(chars, phonetic_codes)


def _status_label(phrases: List[Dict]) -> str:
    if not phrases:
        return "空位"
    words = [str(item.get("word") or "") for item in phrases if item.get("word")]
    if not words:
        return "已有占用"
    label = "已有「" + "、".join(words[:3]) + "」"
    if len(words) > 3:
        label += f"等 {len(words)} 个词"
    return label


def _build_statuses_for_codes(codes: Sequence[str], code_map: Dict[str, List[Dict]]) -> List[Dict]:
    statuses = []
    for code in codes:
        phrases = code_map.get(code, [])
        statuses.append({
            "code": code,
            "occupied": bool(phrases),
            "label": _status_label(phrases),
            "phrases": phrases,
            "words": [phrase.get("word", "") for phrase in phrases if phrase.get("word")],
        })
    return statuses


async def prepare_reviewed_word(config: ReviewHttpConfig, word: str) -> Dict:
    word = word.strip()
    if not word:
        return {"success": False, "message": "词不能为空"}

    evidence, encode_data, existing_words, entity_knowledge = await asyncio.gather(
        collect_pronunciation_evidence_limited(word),
        fetch_keytao_encode(config, word),
        lookup_words(config, [word]),
        _infer_entity_knowledge(word),
    )
    if not encode_data.get("success", True) and not encode_data.get("codes"):
        return {"success": False, "message": encode_data.get("message", "编码服务未返回有效结果")}

    groups = evidence.get("groups", []) if evidence.get("success") else []
    if not groups:
        default_sequence = _encode_default_pinyin_sequence(encode_data)
        entity_group = _entity_pronunciation_group(word, entity_knowledge, default_sequence)
        if not entity_group and default_sequence:
            entity_group = await _contextual_pronunciation_group(
                config,
                word,
                entity_knowledge,
                default_sequence,
            )
        if entity_group:
            groups = [entity_group]
        elif default_sequence:
            groups = [{
                "pinyin": pinyin_sequence_label(default_sequence),
                "normalized": list(default_sequence),
                "sources": [],
                "sourceIds": [],
                "score": 0,
                "fallback": True,
            }]

    all_codes: List[str] = []
    pronunciations: List[Dict] = []
    for group in groups:
        sequence = tuple(group.get("normalized", []))
        codes = _codes_for_pinyin_sequence(encode_data, sequence)
        if not codes:
            continue
        for code in codes:
            if code not in all_codes:
                all_codes.append(code)
        pronunciations.append({
            "pinyin": group.get("pinyin", pinyin_sequence_label(sequence)),
            "normalized": list(sequence),
            "codes": codes,
            "sources": group.get("sources", []),
            "score": group.get("score", 0),
            "fallback": bool(group.get("fallback")),
            "semanticPronunciation": bool(group.get("semanticPronunciation")),
            "sourceSummary": str(group.get("sourceSummary") or "").strip(),
            "contextPronunciation": group.get("contextPronunciation"),
        })

    if not pronunciations:
        return {"success": False, "message": f"未能把「{word}」的读音映射到键道候选编码"}

    code_map = await lookup_codes(config, all_codes)
    global_recommended = ""
    for pronunciation in pronunciations:
        statuses = _build_statuses_for_codes(pronunciation["codes"], code_map)
        pronunciation["candidateStatuses"] = statuses
        recommended = next((item["code"] for item in statuses if not item["occupied"]), statuses[0]["code"] if statuses else "")
        pronunciation["recommendedCode"] = recommended
        if not global_recommended and recommended:
            global_recommended = recommended

    has_authority = any(pron.get("sources") for pron in pronunciations)
    has_semantic_pronunciation = any(pron.get("semanticPronunciation") for pron in pronunciations)
    return {
        "success": True,
        "word": word,
        "existing": existing_words.get(word, []),
        "pronunciations": pronunciations,
        "recommendedCode": global_recommended,
        "autoReviewable": has_authority,
        "autoReviewReason": (
            "至少一个权威来源给出读音"
            if has_authority else
            "本喵已按明确实体语境纠正读音，仍需结合常用词/实体信号完成预审"
            if has_semantic_pronunciation else
            "未找到权威来源，仅使用编码服务默认读音"
        ),
        "entityKnowledge": entity_knowledge if entity_knowledge.get("recognized") else None,
        "sourcePolicy": {
            "acceptedSources": [
                {key: source[key] for key in ("id", "label", "domain", "category", "trust")}
                for source in AUTHORITATIVE_SOURCES
            ],
            "reviewSignalWeights": REVIEW_SIGNAL_WEIGHTS,
        },
    }


def _candidate_codes_from_review(review: Dict, *, include_fallback: bool = False) -> set[str]:
    codes: set[str] = set()
    for pronunciation in review.get("pronunciations", []):
        if not isinstance(pronunciation, dict):
            continue
        if not include_fallback and not pronunciation.get("sources"):
            continue
        for code in pronunciation.get("codes", []):
            if isinstance(code, str):
                codes.add(code)
    return codes


def _is_common_known_word(word: str, commonness: Dict) -> bool:
    if not word or not _CJK_WORD_RE.match(word):
        return False
    if len(word) < 2 or len(word) > 8:
        return False
    if not commonness.get("success"):
        return False
    if (commonness.get("entityKnowledge") or {}).get("accepted"):
        return True
    if (commonness.get("personAlias") or {}).get("accepted"):
        return True

    signals = commonness.get("signals") or {}
    score = float(commonness.get("score") or 0.0)
    active_signals = sum(1 for value in signals.values() if float(value or 0.0) > 0.15)
    has_language_signal = (
        float(signals.get("corpus") or 0.0) > 0.15
        or float(signals.get("dictionary") or 0.0) > 0.15
        or float(signals.get("encyclopedia") or 0.0) > 0.15
    )
    has_search_signal = float(signals.get("search") or 0.0) > 0.35
    return (
        (
            score >= COMMON_KNOWN_MIN_SCORE
            and active_signals >= COMMON_KNOWN_MIN_ACTIVE_SIGNALS
            and (has_language_signal or has_search_signal)
        )
        or (
            score >= COMMON_KNOWN_RELAXED_MIN_SCORE
            and active_signals >= 1
            and has_language_signal
        )
    )


def _common_known_review_type(commonness: Dict) -> str:
    entity = commonness.get("entityKnowledge") or {}
    if entity.get("accepted"):
        return str(entity.get("entityType") or "entity_knowledge")
    if (commonness.get("personAlias") or {}).get("accepted"):
        return "courtesy_name"
    return "common_known_word"


def _common_known_review_label(commonness: Dict) -> str:
    entity = commonness.get("entityKnowledge") or {}
    if entity.get("accepted"):
        return str(entity.get("label") or _entity_type_label(str(entity.get("entityType") or "")))
    if (commonness.get("personAlias") or {}).get("accepted"):
        return "名人字号/别名"
    return "常见词/熟语"


_MANUAL_PREAUDIT_MARKERS = (
    "自动审核：该词需管理员审核",
    "自动审核:该词需管理员审核",
    "自动审核：该词需要管理员审核",
    "自动审核:该词需要管理员审核",
    "自动审核：预计需管理员审核",
    "自动审核:预计需管理员审核",
    "自动审核：预计需要管理员审核",
    "自动审核:预计需要管理员审核",
    "自动审核：需管理员审核",
    "自动审核:需管理员审核",
    "自动审核：该词暂未完成预审",
    "自动审核:该词暂未完成预审",
)


def manual_preaudit_issue_for_item(item: Dict) -> str:
    """Return a conservative batch blocker recorded during add-stage review."""
    remark = str(item.get("remark") or "").strip()
    if not remark:
        return ""

    marker = next((value for value in _MANUAL_PREAUDIT_MARKERS if value in remark), "")
    if not marker:
        return ""

    word = str(item.get("word") or "").strip() or "该词"
    tail = remark.split(marker, 1)[1]
    reason_match = re.match(r"\s*[（(]([^）)]+)[）)]", tail)
    reason = reason_match.group(1).strip() if reason_match else ""
    if reason:
        return f"「{word}」加词预审已标记为需管理员审核：{reason}"
    return f"「{word}」加词预审已标记为需管理员审核"


def can_llm_override_audit_issues(audit: Dict) -> bool:
    """Return whether unresolved audit issues are safe to send through LLM review."""
    issues = audit.get("issues") or []
    if not issues:
        return False
    allowed_fragments = (
        "没有权威读音来源",
        "常用词信号不足",
        "常用度证据不足",
        "可比较的常用度信号不足",
        "声笔笔短码",
        "声笔笔短码表",
    )
    blocked_fragments = (
        "纯删除",
        "不在读音候选链",
        "不在权威读音候选链",
        "改词",
        "歧义",
        "审词失败",
        "词或编码为空",
    )
    return all(
        any(fragment in issue for fragment in allowed_fragments)
        and not any(fragment in issue for fragment in blocked_fragments)
        for issue in issues
    )


def _is_css_review_type(phrase_type: str) -> bool:
    return str(phrase_type or "").strip() in CSS_REVIEW_TYPES


def _same_type_phrases(phrases: Sequence[Dict], phrase_type: str) -> List[Dict]:
    return [
        phrase for phrase in phrases
        if isinstance(phrase, dict) and str(phrase.get("type") or "Phrase") == phrase_type
    ]


async def prepare_css_reviewed_item(config: ReviewHttpConfig, item: Dict) -> Dict:
    """Review CSS/CSSSingle entries as curated short-code table edits, not phrase pinyin encodings."""
    word = str(item.get("word") or "").strip()
    code = str(item.get("code") or "").strip().lower()
    old_word = str(item.get("oldWord") or item.get("old_word") or "").strip()
    phrase_type = str(item.get("type") or "CSS").strip() or "CSS"
    if not word or not code:
        return {"success": False, "message": "词或编码为空"}

    lookup_words_result, lookup_codes_result = await asyncio.gather(
        lookup_words(config, [word] + ([old_word] if old_word else [])),
        lookup_codes(config, [code]),
    )
    word_existing = _same_type_phrases(lookup_words_result.get(word, []), phrase_type)
    code_existing = _same_type_phrases(lookup_codes_result.get(code, []), phrase_type)
    exact_existing = [
        phrase for phrase in word_existing
        if str(phrase.get("code") or "").lower() == code
    ]
    commonness = await estimate_word_commonness(word)

    return {
        "success": True,
        "word": word,
        "code": code,
        "type": phrase_type,
        "oldWord": old_word or None,
        "autoReviewable": bool(exact_existing) or _is_common_known_word(word, commonness),
        "autoReviewReason": (
            "同类型声笔笔词库已存在该词条"
            if exact_existing else
            "声笔笔按短码表和日常优先级审查，不能按普通词组音码判错"
        ),
        "cssShortCodeReview": {
            "accepted": True,
            "policy": (
                "CSS/CSSSingle 是键道声笔笔短码表；编码体现声笔笔码位和词频/结构优先级，"
                "不等同于普通 Phrase 的双拼+形码候选链。"
            ),
            "sameTypeExistingForWord": word_existing[:8],
            "sameTypeExistingForCode": code_existing[:8],
            "exactExisting": exact_existing[:8],
            "commonness": commonness,
        },
    }


def _bounded_log_score(value: float) -> float:
    if value <= 0:
        return 0.0
    return math.log1p(value)


def _count_word_mentions(word: str, result: Dict[str, str]) -> int:
    text = f"{result.get('title', '')} {result.get('snippet', '')}"
    if not word:
        return 0
    return text.count(word)


def _list_of_short_strings(value: Any, *, limit: int = 8, max_len: int = 60) -> List[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: List[str] = []
    seen = set()
    for item in value:
        text = str(item or "").strip()
        if not text or len(text) > max_len or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _normalize_entity_knowledge(word: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    entity_type = str(payload.get("entityType") or payload.get("type") or "unclear").strip()
    confidence = 0.0
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except Exception:
        confidence = 0.0
    recognized = bool(payload.get("recognized")) and entity_type in ENTITY_ACCEPTED_TYPES and confidence >= 0.50
    return {
        "recognized": recognized,
        "word": word,
        "entityType": entity_type if entity_type in ENTITY_ACCEPTED_TYPES else "unclear",
        "confidence": max(0.0, min(confidence, 1.0)),
        "canonicalNames": _list_of_short_strings(payload.get("canonicalNames"), limit=6),
        "aliases": _list_of_short_strings(payload.get("aliases"), limit=8),
        "description": str(payload.get("description") or "").strip()[:160],
        "pinyin": str(payload.get("pinyin") or "").strip()[:80],
        "searchQueries": _list_of_short_strings(payload.get("searchQueries"), limit=10, max_len=90),
        "reviewHint": str(payload.get("reviewHint") or "").strip()[:180],
    }


async def _infer_entity_knowledge(word: str) -> Dict[str, Any]:
    word = word.strip()
    if not word or not _CJK_WORD_RE.match(word) or len(word) > 12:
        return {"recognized": False, "word": word, "entityType": "unclear", "confidence": 0.0}

    config = _review_llm_config()
    if not config["api_key"] or AsyncOpenAI is None:
        return {"recognized": False, "word": word, "entityType": "unclear", "confidence": 0.0}

    system_prompt = (
        "你是中文词语和中文实体常识识别器。给你一个短中文词，只判断它是否可能是大众熟知或稳定存在的词/实体。"
        "可识别类型：common_word, idiom, person, celebrity, historical_person, courtesy_name, stage_name, "
        "fictional_character, brand, product, place, organization, work, technical_term, unclear。"
        "如果是明星、艺名、历史人物、人物字号/别名、角色名、品牌简称、作品名等，请给出全称/别名和适合搜索核验的中文查询。"
        "pinyin 必须按完整词语的真实语境给出逐字拼音，特别检查地名、人名、术语里的多音字；不能沿用脱离语境的逐字默认音。"
        "如果不能确定完整读音，pinyin 留空，不要猜测。"
        "不要为了通过审核而编造；陌生专名、临时网名、含义不明或你不确定时 recognized=false。"
        "只返回 JSON 对象。"
    )
    user_prompt = {
        "word": word,
        "requiredJson": {
            "recognized": True,
            "entityType": "celebrity",
            "confidence": 0.0,
            "canonicalNames": ["全称或标准名"],
            "aliases": ["别名/简称/艺名"],
            "description": "一句话说明它是什么",
            "pinyin": "可选拼音",
            "searchQueries": [f'"{word}" 百度百科', f'"{word}" 是谁'],
            "reviewHint": "为什么它可作为常识实体审查",
        },
    }

    try:
        client = AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=config["timeout"],
        )
        response = await client.chat.completions.create(
            model=config["model"],
            temperature=0.0,
            max_tokens=700,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
            ],
        )
        if not response.choices:
            return {"recognized": False, "word": word, "entityType": "unclear", "confidence": 0.0}
        content = response.choices[0].message.content or ""
        return _normalize_entity_knowledge(word, _load_json_object_from_model_text(content))
    except Exception as error:
        logger.debug(f"Entity knowledge inference failed for {word}: {error}")
        return {"recognized": False, "word": word, "entityType": "unclear", "confidence": 0.0}


def _looks_like_person_alias_result(word: str, result: Dict[str, str]) -> bool:
    text = re.sub(r"\s+", "", f"{result.get('title', '')} {result.get('snippet', '')}")
    if not word or word not in text:
        return False
    if re.search(rf"(?:字|号|别名|又名|又称|人称).{{0,10}}{re.escape(word)}", text):
        return True
    if re.search(rf"{re.escape(word)}.{{0,10}}(?:字|号|别名|又名|又称|人称)", text):
        return True
    return any(hint in text for hint in PERSON_ALIAS_HINTS)


async def _estimate_person_alias_signal(word: str) -> Dict[str, Any]:
    if not word or not _CJK_WORD_RE.match(word) or len(word) > 6:
        return {"accepted": False, "word": word, "hits": [], "score": 0.0}

    query_results = await asyncio.gather(*(
        _search_web(query.format(word=word), max_results=4)
        for query in PERSON_ALIAS_SEARCH_QUERIES
    ))
    hits: List[Dict[str, str]] = []
    seen_urls = set()
    for results in query_results:
        for result in results:
            url = str(result.get("url") or "")
            key = url or f"{result.get('title', '')}:{result.get('snippet', '')}"
            if key in seen_urls:
                continue
            if not _looks_like_person_alias_result(word, result):
                continue
            seen_urls.add(key)
            hits.append(result)

    exact_mentions = sum(_count_word_mentions(word, result) for result in hits)
    score = _bounded_log_score(len(hits) + exact_mentions * 0.5)
    accepted = len(hits) >= 2 or (
        len(hits) >= 1
        and any(
            any(str(result.get(field) or "").find(strong_hint) >= 0 for field in ("title", "snippet"))
            for result in hits
            for strong_hint in ("名将", "历史人物", "人物", "字", "号", "别名", "又名", "门神")
        )
    )
    return {
        "accepted": accepted,
        "word": word,
        "score": score,
        "hits": [
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "provider": result.get("provider", ""),
            }
            for result in hits[:5]
        ],
        "summary": (
            f"搜索结果显示「{word}」有明确历史人物字号/别名信号"
            if accepted else
            "未取得足够的历史人物字号/别名信号"
        ),
    }


def _entity_query_terms(word: str, entity: Dict[str, Any]) -> List[str]:
    terms = [word]
    terms.extend(entity.get("canonicalNames") or [])
    terms.extend(entity.get("aliases") or [])
    return _list_of_short_strings(terms, limit=8)


def _entity_search_queries(word: str, entity: Dict[str, Any]) -> List[str]:
    entity_type = str(entity.get("entityType") or "unclear")
    terms = _entity_query_terms(word, entity)
    queries: List[str] = []
    queries.extend(entity.get("searchQueries") or [])
    hints = ENTITY_TYPE_HINTS.get(entity_type, ())
    for term in terms:
        queries.append(f'"{term}"')
        queries.append(f'"{term}" 百度百科 OR 维基百科')
        for hint in hints[:4]:
            queries.append(f'"{term}" "{hint}"')
    if entity_type in {"person", "celebrity", "historical_person", "courtesy_name", "stage_name"}:
        queries.append(f'"{word}" 是谁')
    return _list_of_short_strings(queries, limit=12, max_len=100)


def _looks_like_entity_result(word: str, result: Dict[str, str], entity: Dict[str, Any]) -> bool:
    text = re.sub(r"\s+", "", f"{result.get('title', '')} {result.get('snippet', '')}")
    return _looks_like_entity_text(word, text, entity)


def _looks_like_entity_text(word: str, text: str, entity: Dict[str, Any]) -> bool:
    text = re.sub(r"\s+", "", text)
    terms = _entity_query_terms(word, entity)
    if not any(term and term in text for term in terms):
        return False
    entity_type = str(entity.get("entityType") or "unclear")
    hints = ENTITY_TYPE_HINTS.get(entity_type, ())
    if any(hint in text for hint in hints):
        return True
    canonical_names = [
        name for name in entity.get("canonicalNames") or []
        if name and name != word
    ]
    if word in text and any(name in text for name in canonical_names):
        return True
    if entity_type in {"person", "celebrity", "historical_person", "courtesy_name", "stage_name"}:
        if re.search(rf"(?:字|号|别名|又名|又称|人称).{{0,12}}{re.escape(word)}", text):
            return True
        if re.search(rf"{re.escape(word)}.{{0,12}}(?:字|号|别名|又名|又称|人称)", text):
            return True
    description = str(entity.get("description") or "")
    if description and any(token and token in text for token in re.split(r"[\s，,。；;、/]+", description)[:5]):
        return True
    return entity_type in {"common_word", "idiom", "technical_term"} and word in text


def _entity_direct_source_urls(word: str, entity: Dict[str, Any]) -> List[Tuple[str, str]]:
    urls: List[Tuple[str, str]] = []
    seen = set()
    terms = _list_of_short_strings([
        *(entity.get("canonicalNames") or []),
        *(entity.get("aliases") or []),
        word,
    ], limit=6)
    sources = list(AUTHORITATIVE_SOURCES)
    entity_type = str(entity.get("entityType") or "unclear")
    if entity_type not in {"common_word", "idiom", "technical_term"}:
        sources.sort(key=lambda source: 0 if source.get("category") == "encyclopedia" else 1)
    for term in terms:
        encoded = quote(term)
        for source in sources:
            if source.get("category") not in {"dictionary", "encyclopedia"}:
                continue
            for template in source.get("direct_urls", []):
                url = template.format(word=encoded)
                if url in seen:
                    continue
                seen.add(url)
                urls.append((str(source.get("label") or ""), url))
    return urls[:10]


async def _fetch_entity_direct_hits(word: str, entity: Dict[str, Any]) -> List[Dict[str, str]]:
    async def inspect_url(label: str, url: str) -> Optional[Dict[str, str]]:
        text = await _fetch_text(url)
        if not text:
            return None
        if not _looks_like_entity_text(word, text[:16000], entity):
            return None
        return {
            "title": label or url,
            "url": url,
            "snippet": text[:240],
            "provider": "direct-source",
        }

    hits: List[Dict[str, str]] = []
    tasks = [
        asyncio.create_task(inspect_url(label, url))
        for label, url in _entity_direct_source_urls(word, entity)
    ]
    if not tasks:
        return hits
    try:
        for task in asyncio.as_completed(tasks):
            hit = await task
            if not hit:
                continue
            hits.append(hit)
            if len(hits) >= 3:
                break
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
    return hits


def _entity_type_label(entity_type: str) -> str:
    return {
        "common_word": "常见词",
        "idiom": "成语/熟语",
        "person": "公众人物",
        "celebrity": "明星/公众人物",
        "historical_person": "历史人物",
        "courtesy_name": "名人字号/别名",
        "stage_name": "艺名/别名",
        "fictional_character": "角色名",
        "brand": "品牌",
        "product": "产品名",
        "place": "地名",
        "organization": "组织/机构名",
        "work": "作品名",
        "technical_term": "专业术语",
    }.get(entity_type, "常识实体")


async def _estimate_entity_knowledge_signal(word: str) -> Dict[str, Any]:
    entity = await _infer_entity_knowledge(word)
    if not entity.get("recognized"):
        fallback = await _estimate_person_alias_signal(word)
        if fallback.get("accepted"):
            return {
                "accepted": True,
                "word": word,
                "entityType": "courtesy_name",
                "label": "名人字号/别名",
                "confidence": 0.60,
                "description": fallback.get("summary", ""),
                "searchQueries": [query.format(word=word) for query in PERSON_ALIAS_SEARCH_QUERIES],
                "hits": fallback.get("hits", []),
                "score": fallback.get("score", 0.0),
                "summary": fallback.get("summary", ""),
                "source": "search_fallback",
            }
        return {
            "accepted": False,
            "word": word,
            "entityType": "unclear",
            "confidence": entity.get("confidence", 0.0),
            "hits": [],
            "score": 0.0,
            "summary": "LLM 未能稳定识别为常见词或常识实体",
            "source": "llm",
        }

    confidence = float(entity.get("confidence") or 0.0)
    entity_type = str(entity.get("entityType") or "unclear")
    llm_high_confidence = (
        confidence >= 0.90
        and entity_type in ENTITY_ACCEPTED_TYPES
        and bool(entity.get("description"))
        and bool(entity.get("canonicalNames") or entity.get("aliases"))
    )
    try:
        direct_hits = await asyncio.wait_for(
            _fetch_entity_direct_hits(word, entity),
            timeout=ENTITY_DIRECT_FETCH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.debug(f"Entity direct-source lookup timed out for {word}")
        direct_hits = []
    queries = _entity_search_queries(word, entity)
    query_results = []
    if not direct_hits and not llm_high_confidence:
        query_results = await asyncio.gather(*(
            _search_web(query, max_results=4)
            for query in queries
        ))
    hits: List[Dict[str, str]] = []
    seen_urls = set()
    for result in direct_hits:
        url = str(result.get("url") or "")
        key = url or f"{result.get('title', '')}:{result.get('snippet', '')}"
        if key in seen_urls:
            continue
        seen_urls.add(key)
        hits.append(result)
    for results in query_results:
        for result in results:
            url = str(result.get("url") or "")
            key = url or f"{result.get('title', '')}:{result.get('snippet', '')}"
            if key in seen_urls:
                continue
            if not _looks_like_entity_result(word, result, entity):
                continue
            seen_urls.add(key)
            hits.append(result)

    exact_mentions = sum(_count_word_mentions(word, result) for result in hits)
    score = _bounded_log_score(len(hits) + exact_mentions * 0.5)
    accepted = (
        len(hits) >= 2
        or (len(hits) >= 1 and confidence >= 0.70)
        or (bool(direct_hits) and confidence >= 0.60)
        or llm_high_confidence
    )
    label = _entity_type_label(entity_type)
    if accepted and hits:
        summary = f"本喵先识别为{label}，并取得权威页面/搜索核验信号"
        source = "llm_direct_source" if direct_hits else "llm_then_search"
    elif accepted and llm_high_confidence:
        summary = f"本喵先识别为{label}，LLM 基础常识给出明确标准名/别名和说明"
        source = "llm_high_confidence"
    else:
        summary = f"本喵先识别为{label}，但搜索核验信号不足"
        source = "llm_then_search"
    return {
        **entity,
        "accepted": accepted,
        "label": label,
        "score": score,
        "hits": [
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
                "provider": result.get("provider", ""),
            }
            for result in hits[:5]
        ],
        "searchQueries": queries,
        "summary": summary,
        "source": source,
    }


async def estimate_word_commonness(word: str) -> Dict:
    word = word.strip()
    if not word:
        return {"success": False, "word": word, "message": "词不能为空", "signals": {}, "score": 0.0}

    signal_raw = {key: 0.0 for key in COMMONNESS_SIGNAL_WEIGHTS}
    evidence: Dict[str, List[str]] = {key: [] for key in COMMONNESS_SIGNAL_WEIGHTS}

    def build_result(entity_knowledge: Dict[str, Any]) -> Dict[str, Any]:
        signals = {
            key: _bounded_log_score(value)
            for key, value in signal_raw.items()
        }
        weighted_score = sum(
            signals[key] * COMMONNESS_SIGNAL_WEIGHTS[key]
            for key in COMMONNESS_SIGNAL_WEIGHTS
        )
        return {
            "success": True,
            "word": word,
            "score": weighted_score,
            "signals": signals,
            "rawSignals": signal_raw,
            "evidence": {
                key: list(dict.fromkeys(value))[:5]
                for key, value in evidence.items()
                if value
            },
            "weights": COMMONNESS_SIGNAL_WEIGHTS,
            "entityKnowledge": entity_knowledge,
            "personAlias": entity_knowledge if entity_knowledge.get("entityType") == "courtesy_name" else {
                "accepted": False,
                "word": word,
                "hits": [],
                "score": 0.0,
            },
        }

    entity_knowledge = await _estimate_entity_knowledge_signal(word)
    if entity_knowledge.get("accepted"):
        signal_raw["encyclopedia"] += 6.0
        signal_raw["corpus"] += 3.0
        signal_raw["search"] += max(1.0, float(entity_knowledge.get("score") or 0.0))
        evidence["encyclopedia"].extend(
            hit.get("url", "")
            for hit in entity_knowledge.get("hits", [])[:3]
            if hit.get("url")
        )
        return build_result(entity_knowledge)

    evidence_data, query_results = await asyncio.gather(
        collect_pronunciation_evidence_limited(word),
        asyncio.gather(*(
            _search_web(query.format(word=word), max_results=5)
            for query, _signal in COMMONNESS_SEARCH_QUERIES
        )),
    )

    if evidence_data.get("success"):
        for group in evidence_data.get("groups", []):
            for source in group.get("sources", []):
                category = str(source.get("category") or "")
                trust = float(source.get("trust") or 0)
                label = str(source.get("source") or "").strip()
                if category == "dictionary":
                    signal_raw["dictionary"] += trust
                    if label:
                        evidence["dictionary"].append(label)
                elif category == "encyclopedia":
                    signal_raw["encyclopedia"] += trust
                    if label:
                        evidence["encyclopedia"].append(label)

    for (query, signal), results in zip(COMMONNESS_SEARCH_QUERIES, query_results):
        exact_mentions = sum(_count_word_mentions(word, result) for result in results)
        signal_raw[signal] += len(results) + exact_mentions * 0.5
        if results:
            evidence[signal].extend(
                result.get("url", "")
                for result in results[:3]
                if result.get("url")
            )

    if entity_knowledge.get("accepted"):
        signal_raw["encyclopedia"] += 6.0
        signal_raw["corpus"] += 3.0
        signal_raw["search"] += max(1.0, float(entity_knowledge.get("score") or 0.0))
        evidence["encyclopedia"].extend(
            hit.get("url", "")
            for hit in entity_knowledge.get("hits", [])[:3]
            if hit.get("url")
        )

    return build_result(entity_knowledge)


def _commonness_signal_votes(front: Dict, behind: Dict) -> Dict[str, str]:
    votes: Dict[str, str] = {}
    front_signals = front.get("signals") or {}
    behind_signals = behind.get("signals") or {}
    for signal in COMMONNESS_SIGNAL_WEIGHTS:
        left = float(front_signals.get(signal) or 0)
        right = float(behind_signals.get(signal) or 0)
        if max(left, right) <= 0:
            continue
        margin = max(0.10, max(left, right) * 0.12)
        if left > right + margin:
            votes[signal] = "front"
        elif right > left + margin:
            votes[signal] = "behind"
        else:
            votes[signal] = "tie"
    return votes


async def compare_word_commonness(front_word: str, behind_word: str) -> Dict:
    front, behind = await asyncio.gather(
        estimate_word_commonness(front_word),
        estimate_word_commonness(behind_word),
    )
    if not front.get("success") or not behind.get("success"):
        return {
            "success": False,
            "verdict": "not_enough_evidence",
            "frontWord": front_word,
            "behindWord": behind_word,
            "summary": "常用度信号获取失败",
            "front": front,
            "behind": behind,
        }

    votes = _commonness_signal_votes(front, behind)
    front_wins = [signal for signal, vote in votes.items() if vote == "front"]
    behind_wins = [signal for signal, vote in votes.items() if vote == "behind"]
    comparable_count = len(votes)
    score_delta = float(front.get("score") or 0) - float(behind.get("score") or 0)

    if comparable_count < 2:
        verdict = "not_enough_evidence"
        summary = "可比较的常用度信号不足"
    elif behind_wins:
        verdict = "behind_more_common"
        summary = f"反向信号显示「{behind_word}」更常用或不弱于「{front_word}」"
    elif score_delta < 0.15:
        verdict = "close"
        summary = f"「{front_word}」相对「{behind_word}」优势不足"
    else:
        verdict = "front_more_common"
        summary = f"常用度证据支持「{front_word}」排在「{behind_word}」前"

    return {
        "success": True,
        "verdict": verdict,
        "frontWord": front_word,
        "behindWord": behind_word,
        "summary": summary,
        "scoreDelta": score_delta,
        "votes": votes,
        "front": front,
        "behind": behind,
    }


def _active_commonness_signals(commonness: Dict) -> int:
    signals = commonness.get("signals") or {}
    return sum(1 for value in signals.values() if float(value or 0) > 0.15)


def _commonness_is_confident(commonness: Dict) -> bool:
    if not commonness.get("success"):
        return False
    if (commonness.get("entityKnowledge") or {}).get("accepted"):
        return True
    if (commonness.get("personAlias") or {}).get("accepted"):
        return True
    return _active_commonness_signals(commonness) >= 2 or float(commonness.get("score") or 0) >= 0.65


def _word_usage_summary(word: str, commonness: Dict) -> str:
    entity = commonness.get("entityKnowledge") or {}
    if entity.get("accepted"):
        label = str(entity.get("label") or _entity_type_label(str(entity.get("entityType") or "")))
        summary = str(entity.get("summary") or "").strip()
        return f"{label}；{summary}" if summary else label
    person_alias = commonness.get("personAlias") or {}
    if person_alias.get("accepted"):
        summary = str(person_alias.get("summary") or "").strip()
        return f"名人字号/别名；{summary}" if summary else "名人字号/别名"

    evidence = commonness.get("evidence") or {}
    if evidence.get("dictionary") and evidence.get("encyclopedia"):
        return "词典/百科均有信号的固定词语或实体名"
    if evidence.get("dictionary"):
        return "词典可查的常规汉语词语"
    if evidence.get("encyclopedia"):
        return "百科可查的实体、术语或专名"
    if float(commonness.get("score") or 0) >= 0.65:
        return "搜索/语料信号较多的日常表达或网络常用词"
    if len(word) >= 4:
        return "用途信号不足，暂按多字固定表达复核"
    return "用途信号不足，需结合上下文人工判断"


def _pronunciation_statuses_for_code(review: Dict, code: str) -> List[Dict]:
    for pronunciation in review.get("pronunciations", []):
        if not isinstance(pronunciation, dict):
            continue
        statuses = pronunciation.get("candidateStatuses")
        if not isinstance(statuses, list):
            continue
        if any(isinstance(status, dict) and status.get("code") == code for status in statuses):
            return [status for status in statuses if isinstance(status, dict)]
    return []


def _same_type_chain_phrases(status: Dict, phrase_type: str) -> List[Dict]:
    phrases = status.get("phrases") if isinstance(status.get("phrases"), list) else []
    return _same_type_phrases(phrases, phrase_type)


async def _review_code_chain_priority(item: Dict, review: Dict) -> Dict:
    word = str(item.get("word") or "").strip()
    code = str(item.get("code") or "").strip().lower()
    phrase_type = str(item.get("type") or "Phrase").strip() or "Phrase"
    commonness = await estimate_word_commonness(word)
    usage = _word_usage_summary(word, commonness)
    base_result = {
        "word": word,
        "code": code,
        "type": phrase_type,
        "usage": usage,
        "commonness": commonness,
        "hasRecommendation": False,
        "priorityOk": True,
        "summary": "同编码链未发现需要调整的高置信优先级问题",
        "currentOrder": [],
        "recommendedOrder": [],
        "recommendedMoves": [],
    }

    statuses = _pronunciation_statuses_for_code(review, code)
    if not statuses:
        base_result["summary"] = "未拿到可比较的候选编码链，暂不建议调序"
        return base_result

    current_index = next(
        (index for index, status in enumerate(statuses) if status.get("code") == code),
        -1,
    )
    if current_index < 0:
        base_result["summary"] = "目标编码不在候选编码链中，暂不建议调序"
        return base_result

    end_index = min(len(statuses), current_index + CODE_CHAIN_PRIORITY_WINDOW_AFTER + 1)
    entries: List[Dict[str, Any]] = []
    seen_words: set[str] = set()
    for index, status in enumerate(statuses[:end_index]):
        status_code = str(status.get("code") or "").strip().lower()
        for phrase in _same_type_chain_phrases(status, phrase_type)[:1]:
            phrase_word = str(phrase.get("word") or "").strip()
            if not phrase_word or phrase_word == word or phrase_word in seen_words:
                continue
            entries.append({
                "word": phrase_word,
                "code": status_code,
                "position": index,
                "current": False,
            })
            seen_words.add(phrase_word)
            if len(entries) >= CODE_CHAIN_PRIORITY_MAX_OCCUPANTS:
                break
        if len(entries) >= CODE_CHAIN_PRIORITY_MAX_OCCUPANTS:
            break

    entries.append({
        "word": word,
        "code": code,
        "position": current_index,
        "current": True,
    })
    entries.sort(key=lambda entry: (int(entry["position"]), 0 if entry["current"] else 1))

    words_to_score = [entry["word"] for entry in entries]
    commonness_by_word: Dict[str, Dict] = {word: commonness}
    missing_words = [entry_word for entry_word in words_to_score if entry_word not in commonness_by_word]
    if missing_words:
        estimates = await asyncio.gather(*(estimate_word_commonness(entry_word) for entry_word in missing_words))
        commonness_by_word.update(dict(zip(missing_words, estimates)))

    for entry in entries:
        entry_commonness = commonness_by_word.get(entry["word"], {})
        entry["score"] = float(entry_commonness.get("score") or 0)
        entry["usage"] = _word_usage_summary(entry["word"], entry_commonness)
        entry["confident"] = _commonness_is_confident(entry_commonness)

    current_order = [
        {
            "word": entry["word"],
            "code": entry["code"],
            "score": entry["score"],
            "usage": entry["usage"],
            "current": entry["current"],
        }
        for entry in entries
    ]
    base_result["currentOrder"] = current_order

    if len(entries) <= 1:
        base_result["summary"] = "同编码链暂无其他同类型词可比较，暂不建议调序"
        return base_result

    if not all(entry["confident"] for entry in entries):
        base_result["summary"] = "同编码链存在常用度信号不足的词，暂不自动建议调序"
        return base_result

    ordered_entries = sorted(
        entries,
        key=lambda entry: (-entry["score"], int(entry["position"]), entry["word"]),
    )
    original_words = [entry["word"] for entry in entries]
    ordered_words = [entry["word"] for entry in ordered_entries]
    top_delta = max(entry["score"] for entry in entries) - min(entry["score"] for entry in entries)
    if ordered_words == original_words or top_delta < CODE_CHAIN_REORDER_SCORE_MARGIN:
        base_result["summary"] = "同编码链常用度顺序基本合理，不建议新的排序"
        return base_result

    target_codes = [entry["code"] for entry in entries]
    recommended_order = []
    recommended_moves = []
    original_code_by_word = {entry["word"]: entry["code"] for entry in entries}
    for entry, target_code in zip(ordered_entries, target_codes):
        recommended = {
            "word": entry["word"],
            "fromCode": original_code_by_word.get(entry["word"], ""),
            "toCode": target_code,
            "score": entry["score"],
            "usage": entry["usage"],
            "current": entry["current"],
        }
        recommended_order.append(recommended)
        if recommended["fromCode"] != target_code:
            recommended_moves.append(recommended)

    if not recommended_moves:
        base_result["summary"] = "同编码链常用度顺序基本合理，不建议新的排序"
        return base_result

    base_result.update({
        "hasRecommendation": True,
        "priorityOk": False,
        "summary": "同编码链常用度显示当前排序可优化，建议按推荐顺序重排",
        "recommendedOrder": recommended_order,
        "recommendedMoves": recommended_moves,
    })
    return base_result


def _find_move_pairs(items: Sequence[Dict]) -> Dict[Tuple[str, str], Dict]:
    creates_by_word: Dict[str, List[Dict]] = {}
    for item in items:
        if item.get("action") == "Create":
            creates_by_word.setdefault(str(item.get("word") or ""), []).append(item)
    pairs: Dict[Tuple[str, str], Dict] = {}
    for item in items:
        if item.get("action") != "Delete":
            continue
        word = str(item.get("word") or "")
        old_code = str(item.get("code") or "")
        for created in creates_by_word.get(word, []):
            new_code = str(created.get("code") or "")
            if new_code and new_code != old_code:
                pairs[(word, old_code)] = created
                break
    return pairs


def _find_priority_comparisons(items: Sequence[Dict]) -> List[Dict[str, str]]:
    moves: List[Dict[str, str]] = []
    move_pairs = _find_move_pairs(items)
    for (word, old_code), created in move_pairs.items():
        new_code = str(created.get("code") or "").strip().lower()
        if new_code:
            moves.append({"word": word, "oldCode": old_code, "newCode": new_code})

    moves_by_old_code: Dict[str, List[Dict[str, str]]] = {}
    for move in moves:
        moves_by_old_code.setdefault(move["oldCode"], []).append(move)

    comparisons: List[Dict[str, str]] = []
    seen: set[Tuple[str, str, str]] = set()
    for move in moves:
        displaced_moves = moves_by_old_code.get(move["newCode"], [])
        for displaced in displaced_moves:
            if displaced["word"] == move["word"]:
                continue
            key = (move["word"], displaced["word"], move["newCode"])
            if key in seen:
                continue
            seen.add(key)
            comparisons.append({
                "frontWord": move["word"],
                "behindWord": displaced["word"],
                "code": move["newCode"],
            })
    return comparisons


def _purpose_review_from_commonness(word: str, code: str, phrase_type: str, commonness: Dict) -> Dict:
    return {
        "word": word,
        "code": code,
        "type": phrase_type,
        "usage": _word_usage_summary(word, commonness),
        "commonnessScore": float(commonness.get("score") or 0),
        "activeSignals": _active_commonness_signals(commonness),
        "confident": _commonness_is_confident(commonness),
        "commonness": commonness,
    }


def _chain_recommendation_text(priority_review: Dict) -> str:
    moves = priority_review.get("recommendedMoves") or []
    if not moves:
        return priority_review.get("summary", "建议复核同编码链顺序")
    move_text = "、".join(
        f"「{move.get('word')}」→{move.get('toCode')}"
        for move in moves[:6]
        if move.get("word") and move.get("toCode")
    )
    return f"{priority_review.get('summary', '建议重排')}：{move_text}"


async def audit_draft_items(config: ReviewHttpConfig, items: Sequence[Dict]) -> Dict:
    if not items:
        return {
            "success": True,
            "verdict": "needs_admin",
            "autoApprove": False,
            "summary": "草稿为空，不能自动审核",
            "issues": ["草稿为空"],
            "approvedItems": [],
        }

    issues: List[str] = []
    approved_items: List[str] = []
    common_known_items: List[Dict[str, Any]] = []
    word_purpose_reviews: List[Dict[str, Any]] = []
    code_chain_priority_reviews: List[Dict[str, Any]] = []
    reviewed_words: Dict[str, Dict] = {}
    move_pairs = _find_move_pairs(items)
    priority_comparisons = _find_priority_comparisons(items)

    for item in items:
        action = str(item.get("action") or "Create")
        word = str(item.get("word") or "").strip()
        code = str(item.get("code") or "").strip().lower()
        old_word = str(item.get("oldWord") or item.get("old_word") or "").strip()
        phrase_type = str(item.get("type") or "Phrase").strip() or "Phrase"

        if not word or not code:
            issues.append("存在词或编码为空的草稿条目")
            continue

        if action == "Delete":
            if (word, code) in move_pairs:
                approved_items.append(f"调码删除原位：{word}@{code}")
                continue
            issues.append(f"纯删除「{word}」@{code} 必须由管理员审核")
            continue

        preaudit_issue = manual_preaudit_issue_for_item(item)
        if preaudit_issue:
            issues.append(preaudit_issue)
            continue

        if _is_css_review_type(phrase_type):
            css_review = await prepare_css_reviewed_item(config, item)
            reviewed_words[word] = css_review
            if not css_review.get("success"):
                issues.append(f"「{word}」声笔笔审查失败：{css_review.get('message', '未知错误')}")
                continue

            css_info = css_review.get("cssShortCodeReview") or {}
            exact_existing = css_info.get("exactExisting") or []
            css_commonness = css_info.get("commonness") if isinstance(css_info.get("commonness"), dict) else {}
            if css_commonness:
                word_purpose_reviews.append(_purpose_review_from_commonness(word, code, phrase_type, css_commonness))
            if action == "Change" and old_word:
                comparison = await compare_word_commonness(word, old_word)
                css_review["commonnessComparison"] = comparison
                if exact_existing or comparison.get("verdict") == "front_more_common":
                    approved_items.append(
                        f"声笔笔改词：{old_word}→{word}@{code}，按 CSS 短码表/常用度优先级通过"
                    )
                    continue
                issues.append(
                    f"声笔笔短码替换「{old_word}→{word}」需要确认："
                    f"{comparison.get('summary', '请按 CSS 短码表、词频和结构对齐复核')}"
                )
                continue

            if css_review.get("autoReviewable"):
                approved_items.append(f"{action}：{word}@{code}，按声笔笔短码表/常见词优先级通过")
                continue
            issues.append(
                f"「{word}」@{code} 是声笔笔短码表条目，不能按普通词组音码判错；"
                "但缺少同类型词库记录或足够常用度证据，需要管理员确认优先级"
            )
            continue

        review_word = word
        if action == "Change" and old_word:
            old_review, new_review = await asyncio.gather(
                prepare_reviewed_word(config, old_word),
                prepare_reviewed_word(config, word),
            )
            reviewed_words[old_word] = old_review
            reviewed_words[word] = new_review
            if new_review.get("autoReviewable") and not old_review.get("autoReviewable"):
                approved_items.append(f"改词：{old_word}→{word}@{code}，新词有权威读音证据，旧词未找到权威证据")
                continue
            issues.append(f"改词「{old_word}→{word}」存在歧义，需要管理员判断哪个词形更正确")
            continue

        if review_word not in reviewed_words:
            reviewed_words[review_word] = await prepare_reviewed_word(config, review_word)
        review = reviewed_words[review_word]
        if not review.get("success"):
            issues.append(f"「{word}」审词失败：{review.get('message', '未知错误')}")
            continue

        candidate_codes = _candidate_codes_from_review(
            review,
            include_fallback=not bool(review.get("autoReviewable")),
        )
        if not review.get("autoReviewable"):
            if code not in candidate_codes:
                available = ", ".join(sorted(candidate_codes)[:8])
                issues.append(f"「{word}」编码 {code} 不在读音候选链中，可选：{available or '无'}")
                continue

            commonness = await estimate_word_commonness(word)
            word_purpose_reviews.append(_purpose_review_from_commonness(word, code, phrase_type, commonness))
            if _is_common_known_word(word, commonness):
                priority_review = await _review_code_chain_priority(item, review)
                code_chain_priority_reviews.append(priority_review)
                if priority_review.get("hasRecommendation"):
                    issues.append(f"「{word}」@{code} 同编码链优先级建议调整：{_chain_recommendation_text(priority_review)}")
                    continue
                common_known_label = _common_known_review_label(commonness)
                common_known_type = _common_known_review_type(commonness)
                summary = (
                    f"「{word}」未找到权威读音页，但属于{common_known_label}，"
                    f"且编码 {code} 在读音候选链中"
                )
                review["commonKnownReview"] = {
                    "accepted": True,
                    "summary": summary,
                    "type": common_known_type,
                    "commonness": commonness,
                    "policy": {
                        "minScore": COMMON_KNOWN_MIN_SCORE,
                        "minActiveSignals": COMMON_KNOWN_MIN_ACTIVE_SIGNALS,
                    },
                }
                common_known_items.append({
                    "word": word,
                    "code": code,
                    "summary": summary,
                    "type": common_known_type,
                    "commonness": commonness,
                })
                approved_items.append(f"{action}：{word}@{code}，本喵按{common_known_label}语言常识通过")
                continue

            issues.append(f"「{word}」没有权威读音来源，且常用词信号不足，需要管理员审核")
            continue
        if code not in candidate_codes:
            available = ", ".join(sorted(candidate_codes)[:8])
            issues.append(f"「{word}」编码 {code} 不在权威读音候选链中，可选：{available or '无'}")
            continue
        priority_review = await _review_code_chain_priority(item, review)
        code_chain_priority_reviews.append(priority_review)
        word_purpose_reviews.append(_purpose_review_from_commonness(
            word,
            code,
            phrase_type,
            priority_review.get("commonness") or {},
        ))
        if priority_review.get("hasRecommendation"):
            issues.append(f"「{word}」@{code} 同编码链优先级建议调整：{_chain_recommendation_text(priority_review)}")
            continue
        approved_items.append(f"{action}：{word}@{code}")

    commonness_results: List[Dict] = []
    for comparison in priority_comparisons:
        commonness = await compare_word_commonness(
            comparison["frontWord"],
            comparison["behindWord"],
        )
        commonness_results.append({**comparison, "result": commonness})
        if commonness.get("verdict") == "front_more_common":
            approved_items.append(
                f"顺序调整：{comparison['frontWord']}@{comparison['code']} 排在 {comparison['behindWord']} 前，常用度证据一致"
            )
            continue
        issues.append(
            f"顺序调整「{comparison['frontWord']}」排在「{comparison['behindWord']}」前的常用度证据不足："
            f"{commonness.get('summary', '未知原因')}"
        )

    auto_approve = not issues and bool(approved_items)
    if auto_approve and common_known_items:
        summary = "读音编码可验证，常见词/实体常识信号足够，允许本喵自动通过"
    elif auto_approve:
        summary = "权威来源、编码和常用度证据一致，允许本喵自动通过"
    else:
        summary = "存在不确定项，需要管理员审核"
    return {
        "success": True,
        "verdict": "pass" if auto_approve else "needs_admin",
        "autoApprove": auto_approve,
        "summary": summary,
        "issues": issues,
        "approvedItems": approved_items,
        "commonKnownItems": common_known_items,
        "wordPurposeReviews": word_purpose_reviews,
        "codeChainPriorityReviews": code_chain_priority_reviews,
        "reviewedWords": reviewed_words,
        "commonnessComparisons": commonness_results,
        "sourcePolicy": {
            "acceptedSources": [
                {key: source[key] for key in ("id", "label", "domain", "category", "trust")}
                for source in AUTHORITATIVE_SOURCES
            ],
            "reviewSignalWeights": REVIEW_SIGNAL_WEIGHTS,
            "commonnessSignalWeights": COMMONNESS_SIGNAL_WEIGHTS,
            "commonKnownWordPolicy": {
                "minScore": COMMON_KNOWN_MIN_SCORE,
                "relaxedMinScore": COMMON_KNOWN_RELAXED_MIN_SCORE,
                "minActiveSignals": COMMON_KNOWN_MIN_ACTIVE_SIGNALS,
                "requiresCandidateCodeMatch": True,
            },
            "cssShortCodePolicy": (
                "CSS/CSSSingle 按键道声笔笔短码表和同码链优先级审查；"
                "不得用普通 Phrase 双拼+形码规则判定 fa/fao 等码位的读音矛盾。"
            ),
        },
    }


def build_review_note(audit: Dict) -> str:
    lines = ["喵喵自动审词报告"]
    lines.append(f"结论：{audit.get('summary', '')}")
    if audit.get("approvedItems"):
        lines.append("通过项：")
        lines.extend(f"- {item}" for item in audit.get("approvedItems", [])[:20])
    if audit.get("issues"):
        lines.append("需人工项：")
        lines.extend(f"- {item}" for item in audit.get("issues", [])[:20])
    if audit.get("commonnessComparisons"):
        lines.append("常用度比较：")
        for item in audit.get("commonnessComparisons", [])[:10]:
            result = item.get("result") or {}
            lines.append(
                f"- {item.get('frontWord')} > {item.get('behindWord')} @ {item.get('code')}："
                f"{result.get('summary', '未给出结论')}"
            )
    if audit.get("wordPurposeReviews"):
        lines.append("词语用途判断：")
        for item in audit.get("wordPurposeReviews", [])[:10]:
            lines.append(
                f"- 「{item.get('word')}」@{item.get('code')}：{item.get('usage', '用途未判定')}；"
                f"常用度分 {float(item.get('commonnessScore') or 0):.2f}"
            )
    if audit.get("codeChainPriorityReviews"):
        lines.append("同编码链优先级：")
        for item in audit.get("codeChainPriorityReviews", [])[:10]:
            if item.get("hasRecommendation"):
                lines.append(f"- 「{item.get('word')}」@{item.get('code')}：{_chain_recommendation_text(item)}")
            else:
                lines.append(f"- 「{item.get('word')}」@{item.get('code')}：{item.get('summary', '不建议调序')}")
    if audit.get("commonKnownItems"):
        lines.append("常见词/熟语/名人字号语言常识通过：")
        for item in audit.get("commonKnownItems", [])[:10]:
            commonness = item.get("commonness") or {}
            lines.append(
                f"- {item.get('word')}@{item.get('code')}：{item.get('summary')}；"
                f"常用度分 {float(commonness.get('score') or 0):.2f}"
            )
    lines.append("权重：语料 0.45，搜索 0.25，词典 0.20，百科 0.10；自动通过要求读音、编码和调序常用度证据一致。")
    return "\n".join(lines)
