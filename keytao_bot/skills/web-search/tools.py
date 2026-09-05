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
import base64
import functools
import html
import ipaddress
import json
import re
import socket
import xml.etree.ElementTree as ElementTree
import zlib
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urljoin, urlparse

import httpx
from nonebot.log import logger

from keytao_bot.utils import http_client


DUCKDUCKGO_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DUCKDUCKGO_LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"
BING_ENDPOINT = "https://www.bing.com/search"
SO360_ENDPOINT = "https://www.so.com/s"
EXA_SEARCH_ENDPOINT = "https://api.exa.ai/search"
GITHUB_API_PREFIX = "https://api.github.com"
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

_JSON_ACCEPT_HEADER = {
    "Accept": "application/vnd.github+json",
    "User-Agent": "keytao-bot-web-search",
    "X-GitHub-Api-Version": "2022-11-28",
}
_RSS_ACCEPT_HEADER = {
    "Accept": "application/atom+xml,application/rss+xml,application/xml,text/xml;q=0.9,*/*;q=0.5",
}
_TEXT_ACCEPT_HEADER = {
    "Accept": "text/plain,text/markdown;q=0.9,*/*;q=0.5",
}

LOGIN_REQUIRED_MESSAGE = "该渠道需要登录会话，本服务器未启用"
_LOGIN_REQUIRED_REASON = "需要复用桌面 Chrome 登录态；共享生产服务器禁止启动浏览器会话"

# Registry entries are data, not executable integrations. Keeping disabled
# agent-reach channels here makes capability reporting honest without pulling
# OpenCLI, Chrome, Playwright, MCP clients, or persistent processes into the bot.
CHANNEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "web": {
        "enabled": True,
        "fetch_backends": ("jina-reader", "direct"),
        "search_backends": (),
        "reason": "通用网页；Jina Reader 主用，hardened direct 备用",
    },
    "github": {
        "enabled": True,
        "fetch_backends": ("github-rest", "jina-reader", "direct"),
        "search_backends": ("github-search",),
        "reason": "公开仓库内容使用 GitHub REST，无需凭据",
    },
    "rss": {
        "enabled": True,
        "fetch_backends": ("rss-parser", "jina-reader", "direct"),
        "search_backends": (),
        "reason": "标准库 XML 解析，响应体先经过 hardened fetch 与字节上限",
    },
    "plain-text": {
        "enabled": True,
        "fetch_backends": ("plain-text", "jina-reader", "direct"),
        "search_backends": (),
        "reason": "纯文本和 Markdown 直接读取",
    },
    "youtube": {
        "enabled": False,
        "fetch_backends": (),
        "search_backends": (),
        "reason": "本轮跳过：字幕需新增 yt-dlp，且对中文输入法词典机器人的核心用例不足",
    },
    "bilibili": {
        "enabled": False,
        "fetch_backends": (),
        "search_backends": (),
        "reason": "本轮未接入 bili-cli、OpenCLI 或站点搜索适配器",
    },
    "twitter": {
        "enabled": False,
        "login_required": True,
        "fetch_backends": (),
        "search_backends": (),
        "reason": _LOGIN_REQUIRED_REASON,
    },
    "reddit": {
        "enabled": False,
        "login_required": True,
        "fetch_backends": (),
        "search_backends": (),
        "reason": _LOGIN_REQUIRED_REASON,
    },
    "facebook": {
        "enabled": False,
        "login_required": True,
        "fetch_backends": (),
        "search_backends": (),
        "reason": _LOGIN_REQUIRED_REASON,
    },
    "instagram": {
        "enabled": False,
        "login_required": True,
        "fetch_backends": (),
        "search_backends": (),
        "reason": _LOGIN_REQUIRED_REASON,
    },
    "tiktok": {
        "enabled": False,
        "login_required": True,
        "fetch_backends": (),
        "search_backends": (),
        "reason": _LOGIN_REQUIRED_REASON,
    },
    "xiaohongshu": {
        "enabled": False,
        "login_required": True,
        "fetch_backends": (),
        "search_backends": (),
        "reason": _LOGIN_REQUIRED_REASON,
    },
    "linkedin": {
        "enabled": False,
        "login_required": True,
        "fetch_backends": (),
        "search_backends": (),
        "reason": _LOGIN_REQUIRED_REASON,
    },
}

_CHANNEL_HOSTS = {
    "github": ("github.com", "raw.githubusercontent.com", "gist.github.com", "api.github.com"),
    "youtube": ("youtube.com", "youtu.be"),
    "bilibili": ("bilibili.com", "b23.tv"),
    "twitter": ("twitter.com", "x.com"),
    "reddit": ("reddit.com", "redd.it"),
    "facebook": ("facebook.com", "fb.com"),
    "instagram": ("instagram.com",),
    "tiktok": ("tiktok.com",),
    "xiaohongshu": ("xiaohongshu.com", "xhslink.com"),
    "linkedin": ("linkedin.com",),
}

_QUERY_INTENTS = (
    ("github", (r"\bgithub\b", r"site:\s*(?:www\.)?github\.com")),
    ("rss", (r"\brss\b", r"\batom\s+feed\b", r"filetype:\s*(?:rss|atom|xml)\b")),
    ("plain-text", (r"filetype:\s*(?:txt|md|markdown)\b",)),
    ("youtube", (r"\byoutube\b", r"油管", r"site:\s*(?:www\.)?youtube\.com")),
    ("bilibili", (r"\bbilibili\b", r"B站", r"site:\s*(?:www\.)?bilibili\.com")),
    ("twitter", (r"\btwitter\b", r"推特", r"site:\s*(?:www\.)?(?:twitter|x)\.com")),
    ("reddit", (r"\breddit\b", r"site:\s*(?:www\.)?reddit\.com")),
    ("facebook", (r"\bfacebook\b", r"site:\s*(?:www\.)?facebook\.com")),
    ("instagram", (r"\binstagram\b", r"site:\s*(?:www\.)?instagram\.com")),
    ("tiktok", (r"\btiktok\b", r"site:\s*(?:www\.)?tiktok\.com")),
    ("xiaohongshu", (r"小红书", r"site:\s*(?:www\.)?xiaohongshu\.com")),
    ("linkedin", (r"\blinkedin\b", r"领英", r"site:\s*(?:www\.)?linkedin\.com")),
)

_LAST_BACKEND_STATUS: Dict[str, Dict[str, str]] = {}


BlockedUrlError = http_client.BlockedUrlError


class EmptyBackendResult(Exception):
    """A backend completed safely but returned no usable content."""


def _host_matches(host: str, candidate: str) -> bool:
    normalized = (host or "").lower().rstrip(".")
    return normalized == candidate or normalized.endswith("." + candidate)


def detect_url_channel(url: str) -> str:
    """Classify a URL without making a network request."""
    parsed = urlparse((url or "").strip())
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower().rstrip("/")
    if (
        path.endswith((".rss", ".atom", ".xml"))
        or path.endswith(("/feed", "/rss", "/atom"))
    ):
        return "rss"
    for channel, hosts in _CHANNEL_HOSTS.items():
        if any(_host_matches(host, candidate) for candidate in hosts):
            return channel
    if path.endswith((".txt", ".text", ".md", ".markdown")):
        return "plain-text"
    return "web"


def detect_query_channel(query: str) -> str:
    """Classify explicit host/platform intent in a search query."""
    value = (query or "").strip()
    for raw_url in re.findall(r"https?://[^\s<>\]\[()]+", value, flags=re.IGNORECASE):
        channel = detect_url_channel(raw_url.rstrip(".,，。!?！？;；"))
        if channel != "web":
            return channel
    for channel, patterns in _QUERY_INTENTS:
        if any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns):
            return channel
    return "web"


def _exa_api_key() -> Optional[str]:
    value = http_client.config_value("exa_api_key", "EXA_API_KEY", None)
    return str(value).strip() if value else None


def search_backend_chain(
    query: str,
    *,
    channel: str = "web",
    exa_enabled: Optional[bool] = None,
) -> List[str]:
    """Return the ordered search chain while preserving the legacy CJK order."""
    legacy = (
        ["so360", "bing", "duckduckgo-html", "duckduckgo-lite"]
        if _has_cjk(query)
        else ["bing", "duckduckgo-html", "duckduckgo-lite", "so360"]
    )
    configured = bool(_exa_api_key()) if exa_enabled is None else bool(exa_enabled)
    prefix = list(CHANNEL_REGISTRY.get(channel, {}).get("search_backends", ()))
    if configured:
        prefix.append("exa")
    return [*prefix, *legacy]


def _record_backend_status(channel: str, backend: str, status: str, reason: str) -> None:
    _LAST_BACKEND_STATUS[f"{channel}:{backend}"] = {
        "status": status,
        "reason": str(reason)[:300],
    }


def _safe_reason(value: object) -> str:
    return (str(value).strip() or "未知原因")[:300]


def _disabled_channel_result(
    channel: str,
    *,
    url: Optional[str] = None,
    query: Optional[str] = None,
) -> Dict[str, Any]:
    spec = CHANNEL_REGISTRY[channel]
    login_required = bool(spec.get("login_required"))
    error = LOGIN_REQUIRED_MESSAGE if login_required else "该渠道本轮未启用"
    result: Dict[str, Any] = {
        "success": False,
        "channel": channel,
        "error": error,
        "reason": str(spec.get("reason") or "未配置可用后端"),
        "attempts": [],
    }
    if url is not None:
        result["url"] = url
    if query is not None:
        result.update({"query": query, "provider": "multi", "results": []})
    return result


def web_channels_doctor() -> Dict[str, Any]:
    """Return static configuration and last-known in-process status only.

    This function deliberately performs no DNS resolution and no HTTP call.
    Live probes are separated into :func:`probe_web_channels` and require the
    standalone script's explicit ``--live`` flag.
    """
    exa_enabled = bool(_exa_api_key())
    channels: List[Dict[str, Any]] = []
    for channel, spec in CHANNEL_REGISTRY.items():
        fetch_backends = list(spec.get("fetch_backends", ()))
        search_cjk = (
            search_backend_chain("中文", channel=channel, exa_enabled=exa_enabled)
            if spec.get("enabled")
            else []
        )
        search_latin = (
            search_backend_chain("english", channel=channel, exa_enabled=exa_enabled)
            if spec.get("enabled")
            else []
        )
        statuses = {
            backend: _LAST_BACKEND_STATUS.get(
                f"{channel}:{backend}",
                {"status": "unknown", "reason": "本进程尚未调用"},
            )
            for backend in dict.fromkeys([*fetch_backends, *search_cjk, *search_latin])
        }
        observed = [value.get("status") for value in statuses.values()]
        if not spec.get("enabled"):
            channel_status = "disabled"
        elif "success" in observed:
            channel_status = "available"
        elif any(value != "unknown" for value in observed):
            channel_status = "degraded"
        else:
            channel_status = "unknown"
        channels.append({
            "channel": channel,
            "enabled": bool(spec.get("enabled")),
            "loginRequired": bool(spec.get("login_required")),
            "reason": str(spec.get("reason") or ""),
            "fetchBackends": fetch_backends,
            "searchBackendsCjk": search_cjk,
            "searchBackendsLatin": search_latin,
            "lastKnownStatus": channel_status,
            "backendStatus": statuses,
        })
    return {
        "mode": "static",
        "liveProbe": False,
        "exaEnabled": exa_enabled,
        "channels": channels,
    }


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


async def _require_public_target(url: str) -> None:
    error = await _validate_fetch_target(url)
    if error:
        raise BlockedUrlError(error)


async def _guarded_request(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    method: str = "GET",
    json_body: Optional[Dict[str, Any]] = None,
    max_hops: Optional[int] = None,
) -> Any:
    """Use the single guarded egress and its three-retry policy."""
    normalized_method = method.upper()
    fetch_options: Dict[str, Any] = {
        "params": params,
        "headers": headers,
        "method": normalized_method,
        "json_body": json_body,
    }
    if max_hops is not None:
        fetch_options["max_hops"] = max_hops
    return await http_client.request_with_retries(
        lambda: http_client.guarded_fetch(url, **fetch_options),
        method=normalized_method,
        url=url,
        # Exa search is a read operation carried over POST and is safe to replay.
        idempotent=normalized_method in {"GET", "HEAD", "OPTIONS"} or url == EXA_SEARCH_ENDPOINT,
        max_attempts=4,
    )


async def _get_text(url: str, *, params: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Search-provider fetch through the guarded, IP-pinned egress."""
    response = await _guarded_request(url, params=params)
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


async def _search_exa(query: str, max_results: int) -> List[Dict[str, str]]:
    api_key = _exa_api_key()
    if not api_key:
        raise RuntimeError("Exa 未配置")
    response = await _guarded_request(
        EXA_SEARCH_ENDPOINT,
        method="POST",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json_body={
            "query": query,
            "numResults": max_results,
            "contents": {"text": {"maxCharacters": 360}},
        },
        # Never forward the API key to a redirect target.
        max_hops=0,
    )
    if not response.is_success:
        raise RuntimeError(f"HTTP {response.status_code}")
    try:
        payload = json.loads(response.text)
    except (TypeError, ValueError) as error:
        raise EmptyBackendResult(f"Exa 返回无法解析的 JSON：{error}")
    results: List[Dict[str, str]] = []
    for item in payload.get("results", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        snippet = item.get("text")
        if not snippet:
            highlights = item.get("highlights")
            snippet = " ".join(str(value) for value in highlights[:3]) if isinstance(highlights, list) else ""
        results.append({
            "title": str(item.get("title") or item.get("url") or "")[:180],
            "url": str(item.get("url") or ""),
            "snippet": str(snippet or "")[:360],
            "provider": "exa",
        })
    return _dedupe_results(results, max_results)


async def _search_github(query: str, max_results: int) -> List[Dict[str, str]]:
    cleaned = re.sub(r"site:\s*(?:www\.)?github\.com", " ", query, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bgithub\b", " ", cleaned, flags=re.IGNORECASE).strip() or query
    response = await _guarded_request(
        f"{GITHUB_API_PREFIX}/search/repositories",
        params={"q": cleaned, "per_page": str(max_results)},
        headers=_JSON_ACCEPT_HEADER,
    )
    if not response.is_success:
        raise RuntimeError(f"HTTP {response.status_code}")
    try:
        payload = json.loads(response.text)
    except (TypeError, ValueError) as error:
        raise EmptyBackendResult(f"GitHub 返回无法解析的 JSON：{error}")
    results: List[Dict[str, str]] = []
    for item in payload.get("items", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        results.append({
            "title": str(item.get("full_name") or item.get("name") or "")[:180],
            "url": str(item.get("html_url") or ""),
            "snippet": str(item.get("description") or "")[:360],
            "provider": "github-search",
        })
    return _dedupe_results(results, max_results)


async def _search_with_provider(provider: str, query: str, max_results: int) -> List[Dict[str, str]]:
    if provider == "exa":
        return await _search_exa(query, max_results)
    if provider == "github-search":
        return await _search_github(query, max_results)
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


async def _fetch_via_jina(url: str, max_chars: int) -> Dict[str, Any]:
    await _require_public_target(url)
    reader_url = _jina_reader_url(url)
    response = await _guarded_request(reader_url)
    if not response.is_success:
        raise RuntimeError(f"HTTP {response.status_code}")
    parsed = _parse_jina_reader_text(response.text, max_chars)
    if not parsed.get("text"):
        raise EmptyBackendResult("Jina Reader 没有提取到正文")
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


async def _web_fetch_via_jina(url: str, max_chars: int, reason: str) -> Dict[str, Any]:
    """Compatibility wrapper retained for existing direct callers/tests."""
    try:
        return await _fetch_via_jina(url, max_chars)
    except Exception as error:
        logger.warning(f"Jina reader fetch failed for {url}: {error}")
        return {"success": False, "url": url, "error": reason or f"网页抓取失败: {error}"}


async def _fetch_direct(url: str, max_chars: int) -> Dict[str, Any]:
    await _require_public_target(url)
    response = await _guarded_request(url, headers=_HTML_ACCEPT_HEADER)
    if not response.is_success:
        raise RuntimeError(f"HTTP {response.status_code}")
    raw_text = response.text
    content_type = response.headers.get("content-type", "")
    title = _extract_title(raw_text)
    description = _extract_meta_description(raw_text)
    extracted = raw_text if "text/plain" in content_type else _extract_main_text(raw_text, max_chars)
    text = _strip_tags(extracted)[:max_chars].strip()
    if not text:
        raise EmptyBackendResult("页面可访问，但没有提取到正文")
    return {
        "success": True,
        "url": response.url,
        "status": response.status_code,
        "title": title,
        "description": description,
        "contentType": content_type,
        "text": text,
        "truncated": len(text) >= max_chars,
        "provider": "direct",
    }


async def _fetch_plain_text(url: str, max_chars: int) -> Dict[str, Any]:
    await _require_public_target(url)
    response = await _guarded_request(url, headers=_TEXT_ACCEPT_HEADER)
    if not response.is_success:
        raise RuntimeError(f"HTTP {response.status_code}")
    text = str(response.text or "").replace("\x00", "").strip()
    if not text:
        raise EmptyBackendResult("纯文本响应为空")
    return {
        "success": True,
        "url": response.url,
        "status": response.status_code,
        "title": urlparse(response.url).path.rsplit("/", 1)[-1],
        "description": "",
        "contentType": response.headers.get("content-type", ""),
        "text": text[:max_chars],
        "truncated": len(text) > max_chars,
        "provider": "plain-text",
    }


def _xml_name(tag: object) -> str:
    return str(tag).rsplit("}", 1)[-1].lower()


def _xml_child_text(node: Any, *names: str) -> str:
    wanted = {name.lower() for name in names}
    for child in list(node):
        if _xml_name(child.tag) in wanted:
            value = "".join(child.itertext()).strip()
            if value:
                return value
    return ""


def _parse_rss_or_atom(content: str, max_chars: int) -> Dict[str, str]:
    if "<!DOCTYPE" in content.upper():
        raise EmptyBackendResult("RSS/Atom 包含不允许的 DOCTYPE")
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise EmptyBackendResult(f"不是可解析的 RSS/Atom：{error}")

    is_rss = _xml_name(root.tag) in {"rss", "rdf"}
    feed = next((child for child in list(root) if _xml_name(child.tag) == "channel"), root)
    title = _xml_child_text(feed, "title")
    description = _strip_tags(_xml_child_text(feed, "description", "subtitle"))[:360]
    entry_names = {"item"} if is_rss else {"entry"}
    entries = [node for node in feed.iter() if _xml_name(node.tag) in entry_names][:20]
    lines: List[str] = []
    for entry in entries:
        entry_title = _strip_tags(_xml_child_text(entry, "title"))
        summary = _strip_tags(_xml_child_text(entry, "description", "summary", "content", "encoded"))
        link = _xml_child_text(entry, "link")
        if not link:
            for child in list(entry):
                if _xml_name(child.tag) == "link" and child.attrib.get("href"):
                    link = str(child.attrib["href"])
                    break
        if not entry_title and not summary:
            continue
        item = entry_title or "未命名条目"
        if summary:
            item += "\n" + summary[:500]
        if link:
            item += "\n" + link
        lines.append(item)
    text = "\n\n".join(lines).strip()
    if not text:
        raise EmptyBackendResult("RSS/Atom 没有可用条目")
    return {"title": title, "description": description, "text": text[:max_chars]}


async def _fetch_rss(url: str, max_chars: int) -> Dict[str, Any]:
    await _require_public_target(url)
    response = await _guarded_request(url, headers=_RSS_ACCEPT_HEADER)
    if not response.is_success:
        raise RuntimeError(f"HTTP {response.status_code}")
    parsed = _parse_rss_or_atom(response.text, max_chars)
    return {
        "success": True,
        "url": response.url,
        "status": response.status_code,
        "title": parsed["title"],
        "description": parsed["description"],
        "contentType": response.headers.get("content-type", ""),
        "text": parsed["text"],
        "truncated": len(parsed["text"]) >= max_chars,
        "provider": "rss-parser",
    }


def _github_api_target(url: str) -> Tuple[str, str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    segments = [unquote(value) for value in parsed.path.split("/") if value]
    if host == "api.github.com":
        return url, "api"
    if host == "raw.githubusercontent.com" and len(segments) >= 4:
        owner, repo, ref = segments[:3]
        path = "/".join(segments[3:])
        endpoint = (
            f"{GITHUB_API_PREFIX}/repos/{quote(owner)}/{quote(repo)}/contents/"
            f"{quote(path, safe='/')}?ref={quote(ref, safe='')}"
        )
        return endpoint, "file"
    if host not in {"github.com", "www.github.com"} or len(segments) < 2:
        raise EmptyBackendResult("不是受支持的 GitHub 公共内容 URL")

    owner, repo = segments[0], segments[1].removesuffix(".git")
    prefix = f"{GITHUB_API_PREFIX}/repos/{quote(owner)}/{quote(repo)}"
    if len(segments) == 2:
        return prefix, "repo"
    if len(segments) >= 4 and segments[2] == "issues" and segments[3].isdigit():
        return f"{prefix}/issues/{segments[3]}", "issue"
    if len(segments) >= 5 and segments[2] in {"blob", "tree"}:
        ref = segments[3]
        path = "/".join(segments[4:])
        return f"{prefix}/contents/{quote(path, safe='/')}?ref={quote(ref, safe='')}", "file"
    raise EmptyBackendResult("GitHub URL 不是仓库、文件、README 或 issue 形态")


def _decode_github_content(payload: Dict[str, Any]) -> str:
    encoded = payload.get("content")
    if not isinstance(encoded, str) or payload.get("encoding") != "base64":
        return ""
    try:
        return base64.b64decode(encoded, validate=False).decode("utf-8", errors="ignore")
    except (ValueError, TypeError):
        return ""


def _render_github_payload(payload: Any, shape: str, max_chars: int) -> Tuple[str, str, str]:
    if isinstance(payload, list):
        rows = [
            f"{item.get('type', 'item')}: {item.get('name', '')} {item.get('html_url', '')}".strip()
            for item in payload[:100]
            if isinstance(item, dict)
        ]
        text = "\n".join(rows)
        return "GitHub directory", "", text[:max_chars]
    if not isinstance(payload, dict):
        return "", "", ""
    if shape == "repo" or payload.get("full_name"):
        title = str(payload.get("full_name") or payload.get("name") or "")
        description = str(payload.get("description") or "")
        text = "\n".join(filter(None, [
            description,
            f"Language: {payload.get('language')}" if payload.get("language") else "",
            f"Stars: {payload.get('stargazers_count')}" if payload.get("stargazers_count") is not None else "",
            f"Forks: {payload.get('forks_count')}" if payload.get("forks_count") is not None else "",
            f"Default branch: {payload.get('default_branch')}" if payload.get("default_branch") else "",
            str(payload.get("html_url") or ""),
        ]))
        return title, description[:360], text[:max_chars]
    if shape == "issue" or (payload.get("number") is not None and payload.get("title")):
        title = f"#{payload.get('number')} {payload.get('title', '')}".strip()
        labels = payload.get("labels") if isinstance(payload.get("labels"), list) else []
        label_names = [str(item.get("name")) for item in labels if isinstance(item, dict) and item.get("name")]
        text = "\n".join(filter(None, [
            f"State: {payload.get('state')}" if payload.get("state") else "",
            f"Labels: {', '.join(label_names)}" if label_names else "",
            str(payload.get("body") or ""),
            str(payload.get("html_url") or ""),
        ]))
        return title, "", text[:max_chars]
    decoded = _decode_github_content(payload)
    if decoded:
        return str(payload.get("name") or "GitHub file"), "", decoded[:max_chars]
    return str(payload.get("name") or "GitHub content"), "", json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )[:max_chars]


async def _fetch_github_rest(url: str, max_chars: int) -> Dict[str, Any]:
    await _require_public_target(url)
    endpoint, shape = _github_api_target(url)
    response = await _guarded_request(endpoint, headers=_JSON_ACCEPT_HEADER)
    if not response.is_success:
        raise RuntimeError(f"HTTP {response.status_code}")
    try:
        payload = json.loads(response.text)
    except (TypeError, ValueError) as error:
        raise EmptyBackendResult(f"GitHub 返回无法解析的 JSON：{error}")
    title, description, text = _render_github_payload(payload, shape, max_chars)

    # A repository URL also exposes its README through the documented REST
    # shape. Failure here does not discard useful repository metadata.
    if shape == "repo" and isinstance(payload, dict):
        readme_url = endpoint.rstrip("/") + "/readme"
        try:
            readme_response = await _guarded_request(readme_url, headers=_JSON_ACCEPT_HEADER)
            if readme_response.is_success:
                readme_payload = json.loads(readme_response.text)
                readme = _decode_github_content(readme_payload) if isinstance(readme_payload, dict) else ""
                if readme:
                    text = (text + "\n\nREADME\n" + readme)[:max_chars]
        except Exception as error:
            logger.debug(f"GitHub README fetch failed for {url}: {error}")
    if not text.strip():
        raise EmptyBackendResult("GitHub REST 没有返回可读内容")
    return {
        "success": True,
        "url": url,
        "status": response.status_code,
        "title": title,
        "description": description,
        "contentType": response.headers.get("content-type", ""),
        "text": text[:max_chars].strip(),
        "truncated": len(text) >= max_chars,
        "provider": "github-rest",
    }


FETCH_BACKENDS = {
    "jina-reader": _fetch_via_jina,
    "direct": _fetch_direct,
    "github-rest": _fetch_github_rest,
    "rss-parser": _fetch_rss,
    "plain-text": _fetch_plain_text,
}


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

    channel = detect_url_channel(normalized_url)
    channel_spec = CHANNEL_REGISTRY[channel]
    if not channel_spec.get("enabled"):
        return _disabled_channel_result(channel, url=normalized_url)

    target_error = await _validate_fetch_target(normalized_url)
    if target_error:
        logger.warning(f"Web fetch blocked for {normalized_url}: {target_error}")
        return {"success": False, "url": url, "channel": channel, "error": target_error, "attempts": []}

    max_chars = max(800, min(max_chars, 12000))
    attempts: List[Dict[str, str]] = []
    for backend_name in channel_spec.get("fetch_backends", ()):
        backend = FETCH_BACKENDS[backend_name]
        try:
            result = await backend(normalized_url, max_chars)
            reason = f"返回 {len(str(result.get('text') or ''))} 字符"
            attempts.append({"backend": backend_name, "status": "success", "reason": reason})
            _record_backend_status(channel, backend_name, "success", reason)
            return {**result, "channel": channel, "attempts": attempts}
        except EmptyBackendResult as error:
            reason = _safe_reason(error)
            attempts.append({"backend": backend_name, "status": "empty", "reason": reason})
            _record_backend_status(channel, backend_name, "empty", reason)
        except Exception as error:
            reason = _safe_reason(error)
            attempts.append({"backend": backend_name, "status": "error", "reason": reason})
            _record_backend_status(channel, backend_name, "error", reason)
            logger.warning(f"Web fetch backend {backend_name} failed for {normalized_url}: {error}")

    return {
        "success": False,
        "url": normalized_url,
        "channel": channel,
        "error": "所有可用抓取后端均失败或未返回正文",
        "attempts": attempts,
    }


async def web_search(
    query: str,
    max_results: int = 5,
    fetch_top_n: int = 0,
    *,
    channel: Optional[str] = None,
) -> Dict[str, Any]:
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

    channel = channel or detect_query_channel(normalized_query)
    if channel not in CHANNEL_REGISTRY:
        channel = "web"
    channel_spec = CHANNEL_REGISTRY[channel]
    if not channel_spec.get("enabled"):
        return _disabled_channel_result(channel, query=normalized_query)

    max_results = max(1, min(max_results, 10))
    fetch_top_n = max(0, min(fetch_top_n, 3))
    providers = search_backend_chain(normalized_query, channel=channel)
    provider_errors: Dict[str, str] = {}
    merged: List[Dict[str, str]] = []
    attempts: List[Dict[str, str]] = []

    try:
        for provider in providers:
            try:
                results = await _search_with_provider(provider, normalized_query, max_results)
                if not results:
                    attempts.append({"backend": provider, "status": "empty", "reason": "未返回结果"})
                    _record_backend_status(channel, provider, "empty", "未返回结果")
                    continue
                previous_count = len(merged)
                merged = _dedupe_results(merged + results, max_results)
                added = len(merged) - previous_count
                if added <= 0:
                    attempts.append({"backend": provider, "status": "empty", "reason": "结果均重复"})
                    _record_backend_status(channel, provider, "empty", "结果均重复")
                    continue
                reason = f"返回 {added} 条结果"
                attempts.append({"backend": provider, "status": "success", "reason": reason})
                _record_backend_status(channel, provider, "success", reason)
                if len(merged) >= max_results:
                    break
            except EmptyBackendResult as exc:
                reason = _safe_reason(exc)
                attempts.append({"backend": provider, "status": "empty", "reason": reason})
                _record_backend_status(channel, provider, "empty", reason)
            except Exception as exc:
                reason = _safe_reason(exc)
                provider_errors[provider] = reason
                attempts.append({"backend": provider, "status": "error", "reason": reason})
                _record_backend_status(channel, provider, "error", reason)
                logger.warning(f"Web search provider {provider} failed for {normalized_query}: {exc}")

        if not merged:
            return {
                "success": False,
                "query": normalized_query,
                "channel": channel,
                "provider": "multi",
                "providersTried": providers,
                "providerErrors": provider_errors,
                "error": "没有拿到可用搜索结果，可能是搜索引擎限制或网络异常",
                "results": [],
                "attempts": attempts,
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
            "channel": channel,
            "provider": "multi",
            "providersTried": providers,
            "providerErrors": provider_errors,
            "results": merged,
            "fetchedPages": fetched_pages,
            "count": len(merged),
            "attempts": attempts,
        }
    except Exception as exc:
        logger.exception(f"Web search failed: {exc}")
        return {
            "success": False,
            "query": normalized_query,
            "channel": channel,
            "provider": "multi",
            "error": f"搜索失败: {exc}",
            "results": [],
            "attempts": attempts,
        }


async def probe_web_channels() -> Dict[str, Any]:
    """Run explicit lightweight live probes for the standalone doctor script."""
    probes = {
        "web": await web_fetch("https://example.com/", max_chars=800),
        "github": await web_fetch(
            "https://github.com/Panniantong/agent-reach",
            max_chars=800,
        ),
        "rss": await web_fetch("https://planetpython.org/rss20.xml", max_chars=800),
        "plain-text": await web_fetch(
            "https://www.rfc-editor.org/rfc/rfc9110.txt",
            max_chars=800,
        ),
        "search": await web_search("KeyTao 输入法", max_results=1),
    }
    report = web_channels_doctor()
    report.update({"mode": "live", "liveProbe": True, "probes": probes})
    return report


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


logger.info(
    "[web_channels] doctor="
    + json.dumps(web_channels_doctor(), ensure_ascii=False, separators=(",", ":"))
)
