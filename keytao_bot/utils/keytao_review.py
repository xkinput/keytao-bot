"""Pronunciation-backed review helpers for KeyTao draft automation."""
from __future__ import annotations

import asyncio
import html
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx
from nonebot.log import logger

from .keytao_encoding import build_phrase_code_chain, pinyin_to_phonetic_code


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
COMMON_KNOWN_MIN_SCORE = 0.55
COMMON_KNOWN_MIN_ACTIVE_SIGNALS = 2
COMMON_KNOWN_RELAXED_MIN_SCORE = 0.35
CSS_REVIEW_TYPES = {"CSS", "CSSSingle"}

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


def normalize_pinyin_syllable(value: str) -> str:
    text = value.strip().lower()
    text = text.replace("u:", "v").replace("ü", "v")
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
        return data
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


def _codes_for_pinyin_sequence(encode_data: Dict, sequence: Sequence[str]) -> List[str]:
    chars = encode_data.get("chars")
    if not isinstance(chars, list) or len(chars) != len(sequence):
        return []
    phonetic_codes = [pinyin_to_phonetic_code(item) or "" for item in sequence]
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

    evidence, encode_data, existing_words = await asyncio.gather(
        collect_pronunciation_evidence(word),
        fetch_keytao_encode(config, word),
        lookup_words(config, [word]),
    )
    if not encode_data.get("success", True) and not encode_data.get("codes"):
        return {"success": False, "message": encode_data.get("message", "编码服务未返回有效结果")}

    groups = evidence.get("groups", []) if evidence.get("success") else []
    if not groups:
        default_sequence = _encode_default_pinyin_sequence(encode_data)
        if default_sequence:
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

    return {
        "success": True,
        "word": word,
        "existing": existing_words.get(word, []),
        "pronunciations": pronunciations,
        "recommendedCode": global_recommended,
        "autoReviewable": any(pron.get("sources") for pron in pronunciations),
        "autoReviewReason": "至少一个权威来源给出读音" if any(pron.get("sources") for pron in pronunciations) else "未找到权威来源，仅使用编码服务默认读音",
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


async def estimate_word_commonness(word: str) -> Dict:
    word = word.strip()
    if not word:
        return {"success": False, "word": word, "message": "词不能为空", "signals": {}, "score": 0.0}

    signal_raw = {key: 0.0 for key in COMMONNESS_SIGNAL_WEIGHTS}
    evidence: Dict[str, List[str]] = {key: [] for key in COMMONNESS_SIGNAL_WEIGHTS}

    evidence_data, query_results = await asyncio.gather(
        collect_pronunciation_evidence(word),
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
    }


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

        if _is_css_review_type(phrase_type):
            css_review = await prepare_css_reviewed_item(config, item)
            reviewed_words[word] = css_review
            if not css_review.get("success"):
                issues.append(f"「{word}」声笔笔审查失败：{css_review.get('message', '未知错误')}")
                continue

            css_info = css_review.get("cssShortCodeReview") or {}
            exact_existing = css_info.get("exactExisting") or []
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
            if _is_common_known_word(word, commonness):
                summary = (
                    f"「{word}」未找到权威读音页，但属于常见现代汉语词语/熟语，"
                    f"且编码 {code} 在读音候选链中"
                )
                review["commonKnownReview"] = {
                    "accepted": True,
                    "summary": summary,
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
                    "commonness": commonness,
                })
                approved_items.append(f"{action}：{word}@{code}，本喵按常见词/熟语语言常识通过")
                continue

            issues.append(f"「{word}」没有权威读音来源，且常用词信号不足，不能自动通过")
            continue
        if code not in candidate_codes:
            available = ", ".join(sorted(candidate_codes)[:8])
            issues.append(f"「{word}」编码 {code} 不在权威读音候选链中，可选：{available or '无'}")
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
    return {
        "success": True,
        "verdict": "pass" if auto_approve else "needs_admin",
        "autoApprove": auto_approve,
        "summary": "证据一致，允许本喵自动通过" if auto_approve else "存在不确定项，提交后等待管理员审核",
        "issues": issues,
        "approvedItems": approved_items,
        "commonKnownItems": common_known_items,
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
    if audit.get("commonKnownItems"):
        lines.append("常见词/熟语语言常识通过：")
        for item in audit.get("commonKnownItems", [])[:10]:
            commonness = item.get("commonness") or {}
            lines.append(
                f"- {item.get('word')}@{item.get('code')}：{item.get('summary')}；"
                f"常用度分 {float(commonness.get('score') or 0):.2f}"
            )
    lines.append("权重：语料 0.45，搜索 0.25，词典 0.20，百科 0.10；自动通过要求读音、编码和调序常用度证据一致。")
    return "\n".join(lines)
