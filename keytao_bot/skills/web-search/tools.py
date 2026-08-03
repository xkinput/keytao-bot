"""
Web Search Skill Tools
通用网络搜索和网页正文抓取工具实现

Security notes
--------------
``web_fetch`` takes a URL straight from the model, so it is an SSRF sink. The
hardening in this module is deliberately layered:

* only ``http`` / ``https`` schemes are accepted;
* every host is resolved and every resulting address is checked against the
  private / loopback / link-local / reserved / multicast / cloud-metadata
  blocklist (IPv6 forms that wrap an IPv4 address are unwrapped first);
* redirects are followed manually, one hop at a time, re-validating the target
  each time — automatic redirect following would let a public host bounce us
  into the private network;
* response bodies are capped at :data:`MAX_FETCH_BYTES` *before* any regex runs
  over them;
* every outbound request passes through the process-wide concurrency gate in
  :mod:`keytao_bot.utils.http_client`.
"""
from __future__ import annotations

import asyncio
import functools
import html
import ipaddress
import zlib
import re
import socket
from typing import Any, Dict, List, Optional, Tuple
import base64
from urllib.parse import parse_qs, unquote, urljoin, urlparse

import httpx
from nonebot.log import logger

from keytao_bot.utils import http_client


DUCKDUCKGO_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"
BING_ENDPOINT = "https://www.bing.com/search"
SO360_ENDPOINT = "https://www.so.com/s"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 12.0

# Hard ceiling on any single response body we are willing to buffer and regex.
MAX_FETCH_BYTES = http_client.MAX_FETCH_BYTES
# Maximum number of redirects we will follow manually (initial request excluded).
MAX_REDIRECT_HOPS = http_client.MAX_REDIRECT_HOPS
ALLOWED_SCHEMES = http_client.ALLOWED_FETCH_SCHEMES
_REDIRECT_STATUS = http_client._REDIRECT_STATUS
# Cloud instance-metadata endpoints. Both already fall inside a blocked range
# (link-local / unique-local) but are listed explicitly as a tripwire.

JINA_READER_PREFIX = "https://r.jina.ai/"

# Ask servers not to compress: the size cap is enforced on raw wire bytes, and
# an identity body makes that cap mean exactly what it says.
_IDENTITY_ENCODING = http_client._IDENTITY_ENCODING

# Accept header used when fetching an HTML page for text extraction.
_HTML_ACCEPT_HEADER = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
}


BlockedUrlError = http_client.BlockedUrlError


# ---------------------------------------------------------------------------
# SSRF guards
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SSRF guards
# ---------------------------------------------------------------------------
# The real implementation lives in keytao_bot.utils.http_client, which owns the
# single guarded egress (resolve -> validate every A/AAAA -> pin the connection
# to a validated IP with Host/SNI preserved -> re-validate every redirect hop).
# These names are kept as thin aliases so this module and its tests have one
# vocabulary, but there is deliberately no second implementation to drift.

_unwrap_ipv6 = http_client._unwrap_ipv6
_is_blocked_ip = http_client.is_blocked_ip
_assert_peer_allowed = http_client._assert_peer_allowed
_peer_addresses = http_client._peer_addresses
_read_capped_body = http_client.read_capped_body
_decode_body = http_client.decode_body
_decompress_bounded = http_client._decompress_bounded


def _validate_scheme(url: str) -> Optional[str]:
    """Return an error message when ``url`` does not use http/https."""
    try:
        http_client.validate_fetch_scheme(url)
    except BlockedUrlError as error:
        return str(error)
    return None


async def _resolve_and_validate_host(host: Optional[str], port: Optional[int] = None) -> Optional[str]:
    """Return an error message when ``host`` resolves to a forbidden address."""
    try:
        await http_client.resolve_validated_addresses(host, port)
    except BlockedUrlError as error:
        return str(error)
    return None


async def _validate_fetch_target(url: str) -> Optional[str]:
    """Full scheme + address validation for one absolute URL."""
    scheme_error = _validate_scheme(url)
    if scheme_error:
        return scheme_error
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError:
        return "URL 端口无效"
    return await _resolve_and_validate_host(parsed.hostname, port)


async def _get_text(url: str, *, params: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Search-provider fetch through the guarded, IP-pinned egress."""
    response = await http_client.guarded_fetch(url, params=params)
    return response.status_code, response.text


def _strip_tags(value: str) -> str:
    text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _normalize_result_url(raw_url: str) -> str:
    if not raw_url:
        return ""

    decoded = html.unescape(raw_url)
    if decoded.startswith("//"):
        decoded = "https:" + decoded
    parsed = urlparse(decoded)

    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        query = parse_qs(parsed.query)
        uddg = query.get("uddg")
        if uddg:
            return unquote(uddg[0])

    if parsed.netloc.endswith("bing.com") and parsed.path.startswith("/ck/"):
        query = parse_qs(parsed.query)
        encoded_target = (query.get("u") or [""])[0]
        if encoded_target.startswith("a1"):
            try:
                padded = encoded_target[2:] + "=" * (-len(encoded_target[2:]) % 4)
                target = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8", "ignore")
                if target:
                    return target
            except Exception:
                pass

    return decoded

def _dedupe_results(results: List[Dict[str, str]], max_results: int) -> List[Dict[str, str]]:
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

def _extract_duckduckgo_html(content: str, max_results: int) -> List[Dict[str, str]]:
    anchors = list(
        re.finditer(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            content,
            re.IGNORECASE | re.DOTALL,
        )
    )
    results: List[Dict[str, str]] = []
    for index, match in enumerate(anchors[:max_results]):
        search_start = match.end()
        search_end = anchors[index + 1].start() if index + 1 < len(anchors) else min(len(content), search_start + 2400)
        nearby_html = content[search_start:search_end]
        snippet_match = re.search(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(.*?)</div>',
            nearby_html,
            re.IGNORECASE | re.DOTALL,
        )
        results.append({
            "title": _strip_tags(match.group(2)),
            "url": _normalize_result_url(match.group(1)),
            "snippet": _strip_tags((snippet_match.group(1) or snippet_match.group(2)) if snippet_match else ""),
            "provider": "duckduckgo-html",
        })
    return _dedupe_results(results, max_results)

def _extract_duckduckgo_lite(content: str, max_results: int) -> List[Dict[str, str]]:
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
    return _dedupe_results(results, max_results)

def _extract_bing(content: str, max_results: int) -> List[Dict[str, str]]:
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
    return _dedupe_results(results, max_results)

def _extract_so360(content: str, max_results: int) -> List[Dict[str, str]]:
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
    return _dedupe_results(results, max_results)

def _has_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value))

def _is_probably_url(value: str) -> bool:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    return bool(parsed.netloc and "." in parsed.netloc)


async def _search_with_provider(provider: str, query: str, max_results: int) -> List[Dict[str, str]]:
    if provider == "duckduckgo-html":
        status, text = await _get_text(DUCKDUCKGO_HTML_ENDPOINT, params={"q": query, "kl": "cn-zh"})
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")
        return _extract_duckduckgo_html(text, max_results)
    if provider == "duckduckgo-lite":
        status, text = await _get_text(DUCKDUCKGO_LITE_ENDPOINT, params={"q": query, "kl": "cn-zh"})
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")
        return _extract_duckduckgo_lite(text, max_results)
    if provider == "bing":
        status, text = await _get_text(BING_ENDPOINT, params={"q": query, "setlang": "zh-CN"})
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")
        return _extract_bing(text, max_results)
    if provider == "so360":
        status, text = await _get_text(SO360_ENDPOINT, params={"q": query})
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")
        return _extract_so360(text, max_results)
    return []


def _extract_title(content: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
    return _strip_tags(match.group(1)) if match else ""


def _extract_meta_description(content: str) -> str:
    match = re.search(
        r'<meta[^>]+(?:name|property)=["\'](?:description|og:description)["\'][^>]+content=["\']([^"\']+)["\']',
        content,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        match = re.search(
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:name|property)=["\'](?:description|og:description)["\']',
            content,
            re.IGNORECASE | re.DOTALL,
        )
    return _strip_tags(match.group(1)) if match else ""


def _extract_main_text(content: str, max_chars: int) -> str:
    text = re.sub(r"<(script|style|noscript|svg|canvas|nav|footer|header|aside)[^>]*>.*?</\1>", " ", content, flags=re.IGNORECASE | re.DOTALL)
    article_match = re.search(r"<article[^>]*>(.*?)</article>", text, re.IGNORECASE | re.DOTALL)
    if article_match:
        text = article_match.group(1)
    else:
        main_match = re.search(r"<main[^>]*>(.*?)</main>", text, re.IGNORECASE | re.DOTALL)
        if main_match:
            text = main_match.group(1)
    stripped = _strip_tags(text)
    stripped = re.sub(r"(\S)\s+([，。！？；：、])", r"\1\2", stripped)
    stripped = re.sub(r"\s{2,}", " ", stripped)
    return stripped[:max_chars].strip()


def _jina_reader_url(url: str) -> str:
    """Wrap ``url`` with the Jina reader prefix exactly once.

    The previous implementation concatenated the prefix twice (and downgraded the
    target to ``http://``), producing ``https://r.jina.ai/http://r.jina.ai/...``
    which the reader cannot resolve.
    """
    target = (url or "").strip()
    lowered_prefix = JINA_READER_PREFIX.lower()
    # Strip any prefix the caller already applied (in either scheme form).
    while True:
        lowered = target.lower()
        if lowered.startswith(lowered_prefix):
            target = target[len(JINA_READER_PREFIX):]
            continue
        if lowered.startswith("http://r.jina.ai/"):
            target = target[len("http://r.jina.ai/"):]
            continue
        break
    if not target.lower().startswith(("http://", "https://")):
        target = "https://" + target
    return f"{JINA_READER_PREFIX}{target}"


def _parse_jina_reader_text(content: str, max_chars: int) -> Dict[str, str]:
    title = ""
    source_url = ""
    text = content
    title_match = re.search(r"^Title:\s*(.+)$", content, re.MULTILINE)
    source_match = re.search(r"^URL Source:\s*(.+)$", content, re.MULTILINE)
    markdown_match = re.search(r"^Markdown Content:\s*([\s\S]*)$", content, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
    if source_match:
        source_url = source_match.group(1).strip()
    if markdown_match:
        text = markdown_match.group(1).strip()
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return {
        "title": title,
        "url": source_url,
        "text": text[:max_chars].strip(),
    }


async def _web_fetch_via_jina(url: str, max_chars: int, reason: str) -> Dict[str, Any]:
    reader_url = _jina_reader_url(url)
    try:
        response = await http_client.guarded_fetch(reader_url)
        if not response.is_success:
            raise RuntimeError(f"HTTP {response.status_code}")
        parsed = _parse_jina_reader_text(response.text, max_chars)
        if not parsed.get("text"):
            return {"success": False, "url": url, "error": reason or "网页抓取失败"}
        return {
            "success": True,
            "url": parsed.get("url") or url,
            "status": response.status_code,
            "title": parsed.get("title", ""),
            "description": "",
            "contentType": response.headers.get("content-type", ""),
            "text": parsed["text"],
            "truncated": len(parsed["text"]) >= max_chars,
            "provider": "jina-reader",
        }
    except Exception as exc:
        logger.warning(f"Jina reader fetch failed for {url}: {exc}")
        return {"success": False, "url": url, "error": reason or f"网页抓取失败: {exc}"}


async def web_fetch(url: str, max_chars: int = 4000) -> Dict[str, Any]:
    """Fetch a webpage and return readable text for synthesis.

    The URL comes from the model, so it is treated as hostile input: scheme and
    resolved addresses are validated up front and again on every redirect hop,
    and the body is capped before any parsing happens.
    """
    normalized_url = (url or "").strip()
    if not normalized_url:
        return {"success": False, "url": url, "error": "URL 不能为空"}
    if "://" in normalized_url:
        scheme_error = _validate_scheme(normalized_url)
        if scheme_error:
            return {"success": False, "url": url, "error": scheme_error}
    else:
        normalized_url = "https://" + normalized_url
    if not _is_probably_url(normalized_url):
        return {"success": False, "url": url, "error": "看起来不是有效 URL"}

    target_error = await _validate_fetch_target(normalized_url)
    if target_error:
        logger.warning(f"Web fetch blocked for {normalized_url}: {target_error}")
        return {"success": False, "url": url, "error": target_error}

    max_chars = max(800, min(max_chars, 12000))
    try:
        response = await http_client.guarded_fetch(
            normalized_url, headers=_HTML_ACCEPT_HEADER,
        )
        final_url = response.url
        raw_text = response.text
        content_type = response.headers.get("content-type", "")

        title = _extract_title(raw_text)
        description = _extract_meta_description(raw_text)
        text = raw_text if "text/plain" in content_type else _extract_main_text(raw_text, max_chars)
        text = _strip_tags(text)[:max_chars].strip()
        if not text:
            return await _web_fetch_via_jina(final_url, max_chars, "页面可访问，但没有提取到正文")
        return {
            "success": True,
            "url": final_url,
            "status": response.status_code,
            "title": title,
            "description": description,
            "contentType": content_type,
            "text": text,
            "truncated": len(text) >= max_chars,
        }
    except BlockedUrlError as exc:
        logger.warning(f"Web fetch blocked for {normalized_url}: {exc}")
        return {"success": False, "url": url, "error": str(exc)}
    except httpx.TimeoutException:
        return await _web_fetch_via_jina(normalized_url, max_chars, "网页抓取超时")
    except httpx.HTTPError as exc:
        logger.warning(f"Web fetch HTTP error for {normalized_url}: {exc}")
        return await _web_fetch_via_jina(normalized_url, max_chars, f"网页抓取失败: {exc}")
    except Exception as exc:
        logger.exception(f"Web fetch failed for {normalized_url}: {exc}")
        return await _web_fetch_via_jina(normalized_url, max_chars, f"网页抓取失败: {exc}")


async def web_search(query: str, max_results: int = 5, fetch_top_n: int = 0) -> Dict[str, Any]:
    """
    Search the web and return structured result snippets.

    Args:
        query: Search query or full question
        max_results: Number of results to return, clamped to 1-10
        fetch_top_n: Optionally fetch readable text from the top N results, clamped to 0-3

    Returns:
        dict: Search result payload
    """
    normalized_query = query.strip()
    if not normalized_query:
        return {
            "success": False,
            "query": query,
            "error": "搜索词不能为空",
            "results": [],
        }

    max_results = max(1, min(max_results, 10))
    fetch_top_n = max(0, min(fetch_top_n, 3))
    providers = (
        ["so360", "bing", "duckduckgo-html", "duckduckgo-lite"]
        if _has_cjk(normalized_query)
        else ["bing", "duckduckgo-html", "duckduckgo-lite", "so360"]
    )
    provider_errors: Dict[str, str] = {}
    merged: List[Dict[str, str]] = []

    try:
        for provider in providers:
            try:
                results = await _search_with_provider(provider, normalized_query, max_results)
                merged = _dedupe_results(merged + results, max_results)
                if len(merged) >= max_results:
                    break
            except Exception as exc:
                provider_errors[provider] = str(exc)
                logger.warning(f"Web search provider {provider} failed for {normalized_query}: {exc}")

        if not merged:
            return {
                "success": False,
                "query": normalized_query,
                "provider": "multi",
                "providersTried": providers,
                "providerErrors": provider_errors,
                "error": "没有拿到可用搜索结果，可能是搜索引擎限制或网络异常",
                "results": [],
            }

        fetched_pages: List[Dict[str, Any]] = []
        if fetch_top_n:
            for item in merged[:fetch_top_n]:
                fetched = await web_fetch(item["url"], max_chars=3000)
                if fetched.get("success"):
                    fetched_pages.append({
                        "title": fetched.get("title") or item.get("title"),
                        "url": fetched.get("url") or item.get("url"),
                        "text": fetched.get("text", "")[:3000],
                    })

        return {
            "success": True,
            "query": normalized_query,
            "provider": "multi",
            "providersTried": providers,
            "providerErrors": provider_errors,
            "results": merged,
            "fetchedPages": fetched_pages,
            "count": len(merged),
        }
    except Exception as exc:
        logger.exception(f"Web search failed: {exc}")
        return {
            "success": False,
            "query": normalized_query,
            "provider": "multi",
            "error": f"搜索失败: {exc}",
            "results": [],
        }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "通用网络搜索。用于查询实时信息、站外资料、新闻、公告、GitHub/官网页面、外部教程，"
                "或当用户明确要求搜索、你不确定答案、问题需要最新资讯时调用。不要用于键道站内词条查询。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索词或完整问题，例如 'nonebot2 function calling', 'DeepSeek API 最新模型', '键道 输入法 安装 教程'"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认 5，范围 1-10",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5
                    },
                    "fetch_top_n": {
                        "type": "integer",
                        "description": "是否顺手抓取前 N 个结果正文用于整合，默认 0，范围 0-3",
                        "minimum": 0,
                        "maximum": 3,
                        "default": 0
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "抓取指定网页正文。用于用户发来 URL、搜索结果摘要不够、需要核对原文/公告/文档时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要抓取的网页 URL"
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "最多返回正文字符数，默认 4000，范围 800-12000",
                        "minimum": 800,
                        "maximum": 12000,
                        "default": 4000
                    }
                },
                "required": ["url"]
            }
        }
    }
]


TOOL_FUNCTIONS = {
    "web_search": web_search,
    "web_fetch": web_fetch,
}
