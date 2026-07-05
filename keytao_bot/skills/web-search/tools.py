"""
Web Search Skill Tools
通用网络搜索和网页正文抓取工具实现
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional, Tuple
import base64
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from nonebot.log import logger


DUCKDUCKGO_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"
BING_ENDPOINT = "https://www.bing.com/search"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
DEFAULT_TIMEOUT = 12.0


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


def _is_probably_url(value: str) -> bool:
    parsed = urlparse(value if "://" in value else f"https://{value}")
    return bool(parsed.netloc and "." in parsed.netloc)


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


async def _get_text(client: httpx.AsyncClient, url: str, *, params: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    response = await client.get(url, params=params)
    return response.status_code, response.text


async def _search_with_provider(client: httpx.AsyncClient, provider: str, query: str, max_results: int) -> List[Dict[str, str]]:
    if provider == "duckduckgo-html":
        status, text = await _get_text(client, DUCKDUCKGO_HTML_ENDPOINT, params={"q": query, "kl": "cn-zh"})
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")
        return _extract_duckduckgo_html(text, max_results)
    if provider == "duckduckgo-lite":
        status, text = await _get_text(client, DUCKDUCKGO_LITE_ENDPOINT, params={"q": query, "kl": "cn-zh"})
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")
        return _extract_duckduckgo_lite(text, max_results)
    if provider == "bing":
        status, text = await _get_text(client, BING_ENDPOINT, params={"q": query, "setlang": "zh-CN"})
        if status >= 400:
            raise RuntimeError(f"HTTP {status}")
        return _extract_bing(text, max_results)
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


async def web_fetch(url: str, max_chars: int = 4000) -> Dict[str, Any]:
    """Fetch a webpage and return readable text for synthesis."""
    normalized_url = url.strip()
    if not normalized_url:
        return {"success": False, "url": url, "error": "URL 不能为空"}
    if not normalized_url.startswith(("http://", "https://")):
        normalized_url = "https://" + normalized_url
    if not _is_probably_url(normalized_url):
        return {"success": False, "url": url, "error": "看起来不是有效 URL"}

    max_chars = max(800, min(max_chars, 12000))
    try:
        async with httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
            },
            follow_redirects=True,
        ) as client:
            response = await client.get(normalized_url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            raw_text = response.text

        title = _extract_title(raw_text)
        description = _extract_meta_description(raw_text)
        text = raw_text if "text/plain" in content_type else _extract_main_text(raw_text, max_chars)
        text = _strip_tags(text)[:max_chars].strip()
        if not text:
            return {
                "success": False,
                "url": str(response.url),
                "status": response.status_code,
                "error": "页面可访问，但没有提取到正文",
            }
        return {
            "success": True,
            "url": str(response.url),
            "status": response.status_code,
            "title": title,
            "description": description,
            "contentType": content_type,
            "text": text,
            "truncated": len(text) >= max_chars,
        }
    except httpx.TimeoutException:
        return {"success": False, "url": normalized_url, "error": "网页抓取超时"}
    except httpx.HTTPError as exc:
        logger.warning(f"Web fetch HTTP error for {normalized_url}: {exc}")
        return {"success": False, "url": normalized_url, "error": f"网页抓取失败: {exc}"}
    except Exception as exc:
        logger.exception(f"Web fetch failed for {normalized_url}: {exc}")
        return {"success": False, "url": normalized_url, "error": f"网页抓取失败: {exc}"}


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
    providers = ["duckduckgo-html", "duckduckgo-lite", "bing"]
    provider_errors: Dict[str, str] = {}
    merged: List[Dict[str, str]] = []

    try:
        async with httpx.AsyncClient(
            timeout=DEFAULT_TIMEOUT,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            follow_redirects=True,
        ) as client:
            for provider in providers:
                try:
                    results = await _search_with_provider(client, provider, normalized_query, max_results)
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
