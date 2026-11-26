"""Platform-aware public URL formatting and trailing-punctuation isolation."""

import re
from typing import List
from urllib.parse import urlsplit, urlunsplit

from . import http_client


_DEFAULT_PUBLIC_BASE_BY_PLATFORM = {
    "qq": "https://keytao.rea.ink",
    "telegram": "https://keytao.vercel.app",
    "web": "https://keytao.vercel.app",
    "web-anon": "https://keytao.vercel.app",
}
_PUBLIC_BASE_CONFIG_BY_PLATFORM = {
    "qq": ("keytao_public_base_qq", "KEYTAO_PUBLIC_BASE_QQ"),
    "telegram": (
        "keytao_public_base_telegram",
        "KEYTAO_PUBLIC_BASE_TELEGRAM",
    ),
    "web": ("keytao_public_base_web", "KEYTAO_PUBLIC_BASE_WEB"),
    "web-anon": ("keytao_public_base_web", "KEYTAO_PUBLIC_BASE_WEB"),
}

_PLAIN_URL_RE = re.compile(
    r"https?://[^\s<>\"'`\[\]{}（）【】「」『』《》〈〉〔〕〖〗“”‘’，。；：、！？［］｛｝]+"
)
_URL_TRAILING_ASCII_PUNCT = ",.;:!?"
_BARE_BATCH_UUID_RE = re.compile(
    r"(?<![0-9A-Fa-f])"
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
    r"(?![0-9A-Fa-f])"
)


def public_base_for_platform(platform: str) -> str:
    """Return the configured public KeyTao base for one reply platform."""
    normalized = str(platform or "").strip().lower()
    default = _DEFAULT_PUBLIC_BASE_BY_PLATFORM.get(
        normalized,
        _DEFAULT_PUBLIC_BASE_BY_PLATFORM["web"],
    )
    attr_name, env_name = _PUBLIC_BASE_CONFIG_BY_PLATFORM.get(
        normalized,
        _PUBLIC_BASE_CONFIG_BY_PLATFORM["web"],
    )
    configured = str(http_client.config_value(attr_name, env_name, default) or "").strip()
    parsed = urlsplit(configured)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return default
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _trim_url_match_end(url: str) -> str:
    """Stop at the first unmatched ASCII ')' and trim trailing ASCII punctuation."""
    depth = 0
    cutoff = len(url)
    for index, ch in enumerate(url):
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth == 0:
                cutoff = index
                break
            depth -= 1
    trimmed = url[:cutoff]
    while trimmed and trimmed[-1] in _URL_TRAILING_ASCII_PUNCT:
        trimmed = trimmed[:-1]
    return trimmed


def separate_url_boundaries(text: str) -> str:
    """Ensure no URL is immediately followed by closing brackets or punctuation.

    Telegram and other clients' URL auto-linkers swallow adjacent fullwidth
    brackets/punctuation (e.g. ``https://example.com/profile）``) into the
    clickable href unless separated by whitespace.
    """
    source = str(text or "")
    if "http://" not in source and "https://" not in source:
        return source
    parts: List[str] = []
    cursor = 0
    for match in _PLAIN_URL_RE.finditer(source):
        start = match.start()
        raw_url = match.group(0)
        url = _trim_url_match_end(raw_url)
        if not url or url in {"http://", "https://"}:
            continue
        end = start + len(url)
        parts.append(source[cursor:end])
        if end < len(source) and not source[end].isspace() and source[end] != "`":
            parts.append(" ")
        cursor = end
    if cursor == 0:
        return source
    parts.append(source[cursor:])
    return "".join(parts)


def render_platform_public_links(
    text: str,
    platform: str,
    *,
    base_resolver=None,
) -> str:
    """Render KeyTao public links on the configured host for one platform."""
    resolver = base_resolver or public_base_for_platform
    public_base = urlsplit(resolver(platform))
    public_hosts = {
        "keytao.rea.ink",
        "www.keytao.rea.ink",
        "keytao.vercel.app",
        "www.keytao.vercel.app",
    }

    def replace_url(match: re.Match) -> str:
        raw_url = match.group(0)
        url = _trim_url_match_end(raw_url)
        suffix = raw_url[len(url):]
        if not url or url in {"http://", "https://"}:
            return raw_url
        source = urlsplit(url)
        if source.hostname not in public_hosts and "/batch/" not in source.path:
            return raw_url
        rewritten = urlunsplit((
            public_base.scheme,
            public_base.netloc,
            source.path,
            source.query,
            source.fragment,
        ))
        return rewritten + suffix

    rendered = _PLAIN_URL_RE.sub(replace_url, str(text or ""))
    return separate_url_boundaries(rendered)


def strip_bare_batch_ids(text: str) -> str:
    """Hide UUID-shaped batch identifiers everywhere except inside links."""
    urls: List[str] = []

    def stash_url(match: re.Match) -> str:
        raw_url = match.group(0)
        url = _trim_url_match_end(raw_url)
        suffix = raw_url[len(url):]
        urls.append(url)
        return f"\x00KEYTAO_URL_{len(urls) - 1}\x00{suffix}"

    stashed = _PLAIN_URL_RE.sub(stash_url, str(text or ""))
    lines: List[str] = []
    for line in stashed.splitlines():
        cleaned = _BARE_BATCH_UUID_RE.sub("", line)
        cleaned = re.sub(r"(?:关联批次|批次(?:ID|编号)?)[：:]\s*$", "", cleaned)
        cleaned = re.sub(r"批次\s+的", "批次的", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).rstrip()
        if cleaned.strip(" \t-•，,。；;：:"):
            lines.append(cleaned)
    rendered = "\n".join(lines)
    for index, url in enumerate(urls):
        rendered = rendered.replace(f"\x00KEYTAO_URL_{index}\x00", url)
    return rendered
