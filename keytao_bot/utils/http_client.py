"""Shared HTTP layer for keytao-bot.

Single place for:
  * lazily created, process-wide ``httpx.AsyncClient`` instances (closed on shutdown)
  * KeyTao API configuration helpers (base URL / bot token / request headers)
  * an authenticated KeyTao API wrapper with bounded exponential-backoff retries
  * a global concurrency gate for outbound scraping of third-party sites

``httpx`` is imported lazily inside functions so that importing this module never
requires the dependency to be present (test harnesses stub it out).
"""
from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import socket
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional

from nonebot.log import logger

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

DEFAULT_KEYTAO_API_BASE = "https://keytao.vercel.app"

# NoneBot lower-cases every config key it loads from .env, so attribute lookups
# must use the lower-case name. ``os.getenv`` is the fallback for contexts where
# the driver is not initialised yet (tests, standalone scripts).
_DEFAULT_TIMEOUT = 20.0
_EXTERNAL_TIMEOUT = 12.0
_EXTERNAL_FETCH_CONCURRENCY = 8

_KEYTAO_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
# Methods whose repetition is safe by definition. Everything else must opt in
# via ``idempotent=True`` before a failed attempt is replayed.
_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def config_value(attr_name: str, env_name: str, default: Any = None) -> Any:
    """Read a setting from the NoneBot driver config, falling back to env vars."""
    try:
        from nonebot import get_driver

        value = getattr(get_driver().config, attr_name, None)
        if value not in (None, ""):
            return value
    except Exception:
        pass
    value = os.getenv(env_name)
    if value not in (None, ""):
        return value
    return default


def get_keytao_url() -> str:
    """Base URL of the keytao-next deployment (never ends with a slash)."""
    raw = config_value("keytao_api_base", "KEYTAO_API_BASE", DEFAULT_KEYTAO_API_BASE)
    return str(raw or DEFAULT_KEYTAO_API_BASE).rstrip("/")


def get_bot_token() -> Optional[str]:
    """Shared bot token used to authenticate against keytao-next bot APIs."""
    token = config_value("bot_api_token", "BOT_API_TOKEN", None)
    return str(token) if token else None


def _parse_json_mapping(value: object) -> Dict[str, str]:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if v}
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        data = json.loads(value)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v}


def get_user_api_key(platform: str, platform_id: str) -> Optional[str]:
    """Return the KeyTao developer API key bound to exactly this platform account.

    There is deliberately no ``default``/global fallback: acting on behalf of a
    user with somebody else's API key would attribute their writes to the wrong
    account. When no key is bound the caller must send no ``X-API-Key`` at all.
    """
    if not platform or not platform_id:
        return None
    mapping = _parse_json_mapping(
        config_value("keytao_user_api_keys", "KEYTAO_USER_API_KEYS", None)
        or config_value("bot_user_api_keys", "BOT_USER_API_KEYS", None)
    )
    return mapping.get(f"{platform}:{platform_id}") or None


def get_bot_headers(
    platform: Optional[str] = None,
    platform_id: Optional[str] = None,
    content_type: bool = False,
) -> Dict[str, str]:
    """Build request headers for keytao-next bot endpoints."""
    headers: Dict[str, str] = {}
    token = get_bot_token()
    if token:
        headers["X-Bot-Token"] = token
    if content_type:
        headers["Content-Type"] = "application/json"

    if platform and platform_id:
        user_api_key = get_user_api_key(platform, platform_id)
        if user_api_key:
            headers["X-API-Key"] = user_api_key

    return headers


# ---------------------------------------------------------------------------
# Shared clients
# ---------------------------------------------------------------------------

EXTERNAL_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

_keytao_client: Any = None
_external_client: Any = None
_client_lock: Optional[asyncio.Lock] = None
_external_semaphore: Optional[asyncio.Semaphore] = None
_shutdown_hook_registered = False


def _get_client_lock() -> asyncio.Lock:
    global _client_lock
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    return _client_lock


def external_fetch_semaphore() -> asyncio.Semaphore:
    """Process-wide gate limiting simultaneous third-party page fetches."""
    global _external_semaphore
    if _external_semaphore is None:
        _external_semaphore = asyncio.Semaphore(_EXTERNAL_FETCH_CONCURRENCY)
    return _external_semaphore


def _register_shutdown_hook() -> None:
    global _shutdown_hook_registered
    if _shutdown_hook_registered:
        return
    _shutdown_hook_registered = True
    try:
        from nonebot import get_driver

        get_driver().on_shutdown(aclose_clients)
    except Exception:
        # Driver unavailable (tests / standalone use): clients are closed by GC.
        _shutdown_hook_registered = False


def _is_usable(client: Any) -> bool:
    return client is not None and not getattr(client, "is_closed", False)


async def get_keytao_client() -> Any:
    """Shared client for keytao-next API traffic."""
    global _keytao_client
    if _is_usable(_keytao_client):
        return _keytao_client
    async with _get_client_lock():
        if _is_usable(_keytao_client):
            return _keytao_client
        import httpx

        _keytao_client = httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )
        _register_shutdown_hook()
        return _keytao_client


def _build_validating_transport() -> Any:
    """Transport that refuses any request aimed at a forbidden address.

    The client follows redirects itself, and httpx dispatches every hop through
    the transport as its own request. Validating here therefore covers redirect
    targets as well as the original URL, which a check at the call site cannot
    do: a trusted vendor endpoint answering 302 with an internal or metadata
    address would otherwise be followed with no validation at all, which is a
    bypass around the guarded egress.
    """
    import httpx

    class _ValidatingTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self._inner = httpx.AsyncHTTPTransport()

        async def handle_async_request(self, request: Any) -> Any:
            url = str(request.url)
            validate_fetch_scheme(url)
            # Raises BlockedUrlError, which surfaces to the caller as a failed
            # request rather than a silently followed redirect.
            await resolve_validated_addresses(request.url.host, request.url.port)
            return await self._inner.handle_async_request(request)

        async def aclose(self) -> None:
            await self._inner.aclose()

    return _ValidatingTransport()


async def get_external_client() -> Any:
    """Shared client for HARD-CODED, trusted third-party endpoints only.

    Use this only when the host is a compile-time constant (a vendor API such as
    api.github.com or a fixed feed).

    Redirects ARE followed, because these vendor APIs legitimately use them, but
    every hop goes through a validating transport that rejects private,
    loopback, link-local and cloud-metadata addresses. So a trusted endpoint
    cannot bounce this client into the internal network.

    What this client does NOT do is pin the connection to the address it
    validated, so a hostile DNS server could still rebind between validation and
    connect. That is acceptable only because the hostnames here are compile-time
    constants belonging to trusted vendors.

    For any URL an attacker can influence — a search-result link, a redirect
    target, anything discovered at runtime — use :func:`guarded_fetch`, which
    additionally pins the connection to a validated IP.
    """
    global _external_client
    if _is_usable(_external_client):
        return _external_client
    async with _get_client_lock():
        if _is_usable(_external_client):
            return _external_client
        import httpx

        _external_client = httpx.AsyncClient(
            timeout=_EXTERNAL_TIMEOUT,
            follow_redirects=True,
            transport=_build_validating_transport(),
            headers={
                "User-Agent": EXTERNAL_USER_AGENT,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        )
        _register_shutdown_hook()
        return _external_client


async def aclose_clients() -> None:
    """Close every shared client. Registered as a driver shutdown hook."""
    global _keytao_client, _external_client
    for client in (_keytao_client, _external_client):
        if _is_usable(client):
            try:
                await client.aclose()
            except Exception as error:  # pragma: no cover - shutdown best effort
                logger.debug(f"[http_client] close failed: {error}")
    _keytao_client = None
    _external_client = None


# ---------------------------------------------------------------------------
# KeyTao API wrapper
# ---------------------------------------------------------------------------


class KeytaoApiError(Exception):
    """Raised when a keytao-next API call cannot be completed.

    Callers must treat this as "unknown", never as "the dictionary is empty".
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


async def request_with_retries(
    request_call: Callable[[], Awaitable[Any]],
    *,
    method: str,
    url: str,
    idempotent: Optional[bool] = None,
    max_attempts: int = 4,
    retry_statuses: Iterable[int] = (),
    retry_connect_errors: bool = False,
    retry_transport_errors: bool = False,
    retry_exceptions: Iterable[type[BaseException]] = (),
) -> Any:
    """Run one HTTP request with timeout retries that respect replay safety.

    Reads retry any ``httpx.TimeoutException``. Writes retry only
    ``httpx.ConnectTimeout``, which happens before a request reaches the
    server. Callers may opt into the existing transport/status retry behavior
    used by :func:`keytao_request`; bare-client call sites use timeout retries
    only.
    """
    import httpx

    normalized_method = method.upper()
    replay_safe = (
        idempotent if idempotent is not None else normalized_method in _IDEMPOTENT_METHODS
    )
    attempts = max(1, int(max_attempts))
    retryable_statuses = frozenset(retry_statuses)
    additional_retry_exceptions = tuple(retry_exceptions)

    async def wait_before_retry(attempt: int, reason: str) -> None:
        logger.info(
            f"[http_client] retry {attempt}/{attempts - 1} "
            f"{normalized_method} {url}: {reason}"
        )
        await asyncio.sleep(0.5 * (2 ** (attempt - 1)))

    for attempt in range(1, attempts + 1):
        try:
            response = await request_call()
        except httpx.ConnectTimeout as error:
            if attempt >= attempts:
                raise
            await wait_before_retry(attempt, f"{type(error).__name__}: {error}")
            continue
        except httpx.TimeoutException as error:
            if not replay_safe or attempt >= attempts:
                raise
            await wait_before_retry(attempt, f"{type(error).__name__}: {error}")
            continue
        except httpx.ConnectError as error:
            if not retry_connect_errors or attempt >= attempts:
                raise
            await wait_before_retry(attempt, f"{type(error).__name__}: {error}")
            continue
        except httpx.TransportError as error:
            if not (replay_safe and retry_transport_errors) or attempt >= attempts:
                raise
            await wait_before_retry(attempt, f"{type(error).__name__}: {error}")
            continue
        except additional_retry_exceptions as error:
            if not replay_safe or attempt >= attempts:
                raise
            await wait_before_retry(attempt, f"{type(error).__name__}: {error}")
            continue

        status_code = getattr(response, "status_code", None)
        if (
            replay_safe
            and status_code in retryable_statuses
            and attempt < attempts
        ):
            await wait_before_retry(attempt, f"HTTP {status_code}")
            continue
        return response

    raise RuntimeError("HTTP retry loop exited without a response")  # pragma: no cover


async def keytao_request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
    content: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    platform: Optional[str] = None,
    platform_id: Optional[str] = None,
    timeout: Optional[float] = None,
    retries: int = 4,
    require_token: bool = True,
    idempotent: Optional[bool] = None,
) -> Any:
    """Perform an authenticated keytao-next request with bounded retries.

    Retry policy is driven by idempotency, because a blind retry of a write is
    not a resilience feature — it is a double execution. Redeeming a bind code
    twice, for example, reports failure to a user whose account is in fact
    already bound.

    * Idempotent methods (GET/HEAD/OPTIONS) retry 5xx, timeouts and transport
      errors with exponential backoff.
    * Every other method retries **only** on ``ConnectTimeout``, so the server
      provably never saw the request. A read timeout, other transport error, or
      5xx is not retried: the request may well have been applied.
    * Pass ``idempotent=True`` explicitly for a write endpoint that is safe to
      repeat, to opt back into full retries.
    """
    import httpx

    token = get_bot_token()
    if require_token and not token:
        raise KeytaoApiError("喵喵配置错误：缺少 BOT_API_TOKEN")

    request_headers = get_bot_headers(
        platform, platform_id, content_type=json_body is not None or content is not None
    )
    if headers:
        request_headers.update(headers)

    url = f"{get_keytao_url()}{path if path.startswith('/') else '/' + path}"
    client = await get_keytao_client()
    normalized_method = method.upper()
    replay_safe = (
        idempotent if idempotent is not None else normalized_method in _IDEMPOTENT_METHODS
    )
    attempts = max(1, retries)

    async def send_request() -> Any:
        return await client.request(
            normalized_method,
            url,
            params=params,
            json=json_body,
            content=content,
            headers=request_headers,
            timeout=timeout if timeout is not None else _DEFAULT_TIMEOUT,
        )

    try:
        return await request_with_retries(
            send_request,
            method=normalized_method,
            url=path,
            idempotent=replay_safe,
            max_attempts=attempts,
            retry_statuses=_KEYTAO_RETRYABLE_STATUS,
            retry_connect_errors=replay_safe,
            retry_transport_errors=True,
        )
    except (httpx.ConnectError, httpx.ConnectTimeout) as error:
        last_error = f"{type(error).__name__}: {error}"
        raise KeytaoApiError(
            f"KeyTao API 请求失败（{normalized_method} {path}）：{last_error}"
        ) from error
    except (httpx.TimeoutException, httpx.TransportError) as error:
        last_error = f"{type(error).__name__}: {error}"
        suffix = "" if replay_safe else "；该请求可能已生效，未自动重试"
        if not replay_safe and (
            "/batches/" in path or "/pull-requests/" in path
        ):
            suffix += "；请先查看草稿核对状态，再决定是否重试"
        raise KeytaoApiError(
            f"KeyTao API 请求失败（{normalized_method} {path}）：{last_error}{suffix}"
        ) from error


async def keytao_json(
    method: str,
    path: str,
    *,
    allow_status: Optional[Iterable[int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """``keytao_request`` + JSON decoding.

    Raises :class:`KeytaoApiError` on transport failure, non-2xx status, or a
    body that is not valid JSON, so that callers can never mistake a failed
    lookup for an empty result.

    ``allow_status`` opts specific non-2xx codes out of that rule, for the
    endpoints whose *answer* is carried by an error status: the draft write
    replies 400 with the confirmation snapshot the caller has to echo back.
    The caller then has to judge the body itself.
    """
    permitted = frozenset(allow_status or ())
    response = await keytao_request(method, path, **kwargs)
    try:
        data = response.json()
    except Exception:
        raise KeytaoApiError(
            f"KeyTao API 返回非 JSON 响应（HTTP {response.status_code}）",
            status_code=response.status_code,
        )
    if not response.is_success and response.status_code not in permitted:
        message = ""
        if isinstance(data, dict):
            message = str(data.get("message") or data.get("error") or "")
        raise KeytaoApiError(
            message or f"HTTP {response.status_code}",
            status_code=response.status_code,
        )
    return data if isinstance(data, dict) else {"data": data}


async def external_get(url: str, **kwargs: Any) -> Any:
    """GET a third-party URL through the shared client and concurrency gate."""
    client = await get_external_client()
    async with external_fetch_semaphore():
        return await client.get(url, **kwargs)


# ===========================================================================
# Guarded outbound fetch (SSRF-safe egress)
# ===========================================================================
#
# This is the ONLY sanctioned way to fetch a URL that an attacker can influence
# (search-result links, user-supplied pages, anything discovered at runtime).
#
# Design: resolve-then-pin. For every hop we
#   1. validate the scheme,
#   2. resolve the hostname to ALL of its A/AAAA records,
#   3. reject unless EVERY answer passes the blocklist,
#   4. connect to one of the validated IPs *literally* — the request URL carries
#      the IP, the original hostname travels in the Host header and in the TLS
#      ``sni_hostname`` extension.
#
# Because the connect target is an IP we chose, there is no second DNS lookup
# and therefore no rebinding window and no blind-SSRF window: the request is
# only ever emitted at an address that already passed validation. Certificate
# verification still runs against the real hostname (``sni_hostname`` drives
# ``server_hostname``), so pinning does not weaken TLS, and CDNs are unaffected
# because we do not require a second resolution to agree with the first.

ALLOWED_FETCH_SCHEMES = frozenset({"http", "https"})
MAX_FETCH_BYTES = 2 * 1024 * 1024
MAX_REDIRECT_HOPS = 3
_REDIRECT_STATUS = frozenset({301, 302, 303, 307, 308})
_IDENTITY_ENCODING = {"Accept-Encoding": "identity"}

# Cloud metadata endpoints, which are not covered by the generic private/
# link-local checks in every representation.
_CLOUD_METADATA_ADDRESSES = frozenset({
    "169.254.169.254",
    "fd00:ec2::254",
})

# Explicit IANA special-purpose ranges. The ipaddress ``is_*`` properties are
# NOT sufficient on their own: notably 100.64.0.0/10 (CGNAT shared address
# space) reports is_private=False, and Alibaba Cloud serves instance metadata at
# 100.100.100.200 inside that range — so a blocklist built only from those
# properties hands out cloud credentials. IPv6 fec0::/10 (deprecated
# site-local) is likewise not flagged by any property.
_BLOCKED_IPV4_NETWORKS = tuple(ipaddress.ip_network(cidr) for cidr in (
    "0.0.0.0/8",            # "this host on this network"
    "10.0.0.0/8",           # private
    "100.64.0.0/10",        # CGNAT / shared address space (Alibaba metadata)
    "127.0.0.0/8",          # loopback
    "169.254.0.0/16",       # link-local (AWS/GCP/Azure metadata)
    "172.16.0.0/12",        # private
    "192.0.0.0/24",         # IETF protocol assignments
    "192.0.2.0/24",         # TEST-NET-1
    "192.31.196.0/24",      # AS112-v4
    "192.52.193.0/24",      # AMT
    "192.88.99.0/24",       # deprecated 6to4 relay anycast
    "192.168.0.0/16",       # private
    "192.175.48.0/24",      # direct delegation AS112
    "198.18.0.0/15",        # benchmarking
    "198.51.100.0/24",      # TEST-NET-2
    "203.0.113.0/24",       # TEST-NET-3
    "224.0.0.0/4",          # multicast
    "240.0.0.0/4",          # reserved for future use
    "255.255.255.255/32",   # limited broadcast
))
_BLOCKED_IPV6_NETWORKS = tuple(ipaddress.ip_network(cidr) for cidr in (
    "::/128",               # unspecified
    "::1/128",              # loopback
    "64:ff9b::/96",         # NAT64 well-known prefix (can embed private IPv4)
    "64:ff9b:1::/48",       # NAT64 local-use
    "100::/64",             # discard-only
    "2001::/32",            # Teredo
    "2001:20::/28",         # ORCHIDv2
    "2001:db8::/32",        # documentation
    "5f00::/16",            # segment routing
    "fc00::/7",             # unique local
    "fe80::/10",            # link-local
    "fec0::/10",            # deprecated site-local, still routed on some nets
    "ff00::/8",             # multicast
))


class BlockedUrlError(Exception):
    """Raised when a fetch target fails scheme or address validation."""

    def __init__(self, message: str, *, transient: bool = False):
        super().__init__(message)
        self.transient = transient


class TransientFetchError(BlockedUrlError):
    """Raised when guarded egress validation failed for a retryable reason."""

    def __init__(self, message: str):
        super().__init__(message, transient=True)


def _unwrap_ipv6(address: Any) -> list:
    """Return every IPv4 address embedded in an IPv6 wrapper form."""
    if not isinstance(address, ipaddress.IPv6Address):
        return []
    embedded: list = []
    for attribute in ("ipv4_mapped", "sixtofour"):
        try:
            value = getattr(address, attribute)
        except Exception:  # pragma: no cover - stdlib should not raise
            value = None
        if value is not None:
            embedded.append(value)
    try:
        teredo = address.teredo
    except Exception:  # pragma: no cover - defensive
        teredo = None
    if teredo:
        embedded.extend(part for part in teredo if part is not None)
    return embedded


def is_blocked_ip(address: Any) -> bool:
    """True when the address must never be contacted from the bot process."""
    if isinstance(address, str):
        try:
            address = ipaddress.ip_address(address.split("%", 1)[0])
        except ValueError:
            return True

    for candidate in [address, *_unwrap_ipv6(address)]:
        # Property checks first (cheap, catches most), then the explicit
        # registry which covers the ranges the properties miss.
        if (
            candidate.is_private
            or candidate.is_loopback
            or candidate.is_link_local
            or candidate.is_reserved
            or candidate.is_multicast
            or candidate.is_unspecified
        ):
            return True
        if str(candidate) in _CLOUD_METADATA_ADDRESSES:
            return True
        networks = (
            _BLOCKED_IPV4_NETWORKS
            if isinstance(candidate, ipaddress.IPv4Address)
            else _BLOCKED_IPV6_NETWORKS
        )
        if any(candidate in network for network in networks):
            return True
    return False


def validate_fetch_scheme(url: str) -> None:
    """Raise :class:`BlockedUrlError` unless the URL is http/https."""
    from urllib.parse import urlparse

    scheme = urlparse(url).scheme.lower()
    if scheme not in ALLOWED_FETCH_SCHEMES:
        raise BlockedUrlError(f"不支持的 URL 协议（{scheme or '缺失'}），只允许 http/https")


async def resolve_validated_addresses(host: Optional[str], port: Optional[int]) -> list:
    """Resolve ``host`` and return its addresses, or raise if any is forbidden.

    Every answer must pass: a hostname that resolves to one public and one
    private address is rejected outright rather than filtered, because we cannot
    know which one a later lookup would have produced.
    """
    hostname = (host or "").strip().strip("[]")
    if not hostname:
        raise BlockedUrlError("URL 缺少主机名")

    # A literal IP never touches DNS.
    try:
        literal = ipaddress.ip_address(hostname)
    except ValueError:
        literal = None
    if literal is not None:
        if is_blocked_ip(literal):
            raise BlockedUrlError(f"禁止访问内网/保留地址：{hostname}")
        return [literal]

    try:
        loop = asyncio.get_running_loop()
        infos = await loop.run_in_executor(
            None,
            lambda: socket.getaddrinfo(hostname, port or 0, 0, socket.SOCK_STREAM),
        )
    except Exception as error:
        raise TransientFetchError(f"域名解析失败：{hostname}（{error}）")

    if not infos:
        raise TransientFetchError(f"域名解析失败：{hostname}")

    addresses = []
    for info in infos:
        raw = str(info[4][0]).split("%", 1)[0]
        try:
            resolved = ipaddress.ip_address(raw)
        except ValueError:
            raise BlockedUrlError(f"无法识别解析结果：{hostname} → {raw}")
        if is_blocked_ip(resolved):
            raise BlockedUrlError(f"禁止访问内网/保留地址：{hostname} → {resolved}")
        if resolved not in addresses:
            addresses.append(resolved)
    return addresses


def _pin_request(url: str, address: Any) -> tuple:
    """Rewrite ``url`` to connect to ``address`` literally.

    Returns ``(pinned_url, host_header, sni_hostname)``.
    """
    from urllib.parse import urlunparse, urlparse

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").strip()
    port = parsed.port
    literal = (
        f"[{address}]" if isinstance(address, ipaddress.IPv6Address) else str(address)
    )
    netloc = f"{literal}:{port}" if port else literal
    pinned = urlunparse((
        parsed.scheme, netloc, parsed.path or "/", parsed.params, parsed.query, "",
    ))
    # urlparse strips the brackets from an IPv6 literal host, but RFC 3986
    # requires them back in the Host header or "::1:8080" parses as a bare
    # address with a bogus final group.
    host_literal = f"[{hostname}]" if _is_ipv6_literal(hostname) else hostname
    host_header = f"{host_literal}:{port}" if port else host_literal
    return pinned, host_header, hostname


def _is_ipv6_literal(hostname: str) -> bool:
    try:
        return isinstance(ipaddress.ip_address(hostname), ipaddress.IPv6Address)
    except ValueError:
        return False


def _decompress_bounded(headers: Any, body: bytes) -> bytes:
    """Decompress with a hard output cap, or return the body unchanged."""
    import zlib

    encoding = ""
    if headers is not None:
        try:
            encoding = str(headers.get("content-encoding") or "").strip().lower()
        except Exception:  # pragma: no cover - defensive
            encoding = ""
    if not encoding or encoding == "identity" or not body:
        return body
    if encoding in ("gzip", "x-gzip"):
        wbits = 16 + zlib.MAX_WBITS
    elif encoding == "deflate":
        wbits = zlib.MAX_WBITS
    else:
        # br/zstd: no bounded decoder here, so refuse rather than risk an
        # unbounded allocation.
        return b""
    try:
        return zlib.decompressobj(wbits).decompress(body, MAX_FETCH_BYTES)
    except zlib.error:
        return b""


async def read_capped_body(response: Any, max_bytes: int = MAX_FETCH_BYTES) -> bytes:
    """Buffer at most ``max_bytes`` of a streaming response.

    Reads the RAW stream: ``aiter_bytes`` yields decoded output, so a
    compression bomb could hand back a single multi-hundred-MB chunk before any
    size check ran.
    """
    chunks: list = []
    total = 0
    iterator = getattr(response, "aiter_raw", None) or response.aiter_bytes
    async for chunk in iterator():
        chunks.append(chunk)
        total += len(chunk)
        if total >= max_bytes:
            break
    return _decompress_bounded(
        getattr(response, "headers", None), b"".join(chunks)[:max_bytes]
    )


def decode_body(response: Any, body: bytes) -> str:
    encoding = getattr(response, "encoding", None) or "utf-8"
    try:
        return body.decode(encoding, errors="ignore")
    except (LookupError, TypeError, ValueError):
        return body.decode("utf-8", errors="ignore")


def _peer_addresses(response: Any) -> list:
    """Addresses this response was actually served from, if the transport says."""
    extensions = getattr(response, "extensions", None) or {}
    stream = extensions.get("network_stream") if isinstance(extensions, dict) else None
    if stream is None:
        return []
    addresses = []
    for key in ("server_addr", "peer_addr"):
        try:
            info = stream.get_extra_info(key)
        except Exception:  # pragma: no cover - transport specific
            continue
        if not info:
            continue
        raw = info[0] if isinstance(info, (tuple, list)) and info else info
        try:
            addresses.append(ipaddress.ip_address(str(raw).split("%", 1)[0]))
        except ValueError:
            continue
    return addresses


def _assert_peer_allowed(response: Any, url: str) -> None:
    """Defence in depth on top of pinning.

    Pinning already guarantees the destination, so an absent peer address is
    tolerated here (UDS / ASGI / MockTransport expose none). A peer that *is*
    reported and is forbidden still aborts before the body is read.
    """
    for address in _peer_addresses(response):
        if is_blocked_ip(address):
            raise BlockedUrlError(f"禁止访问内网/保留地址：{url} 实际连接到 {address}")


class FetchResult:
    """Outcome of a guarded fetch."""

    __slots__ = ("status_code", "url", "text", "headers", "encoding")

    def __init__(self, status_code: int, url: str, text: str, headers: Any, encoding: Any):
        self.status_code = status_code
        self.url = url
        self.text = text
        self.headers = headers
        self.encoding = encoding

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300


_guarded_client: Any = None


async def get_guarded_client() -> Any:
    """Client used exclusively for pinned, guarded fetches."""
    global _guarded_client
    if _is_usable(_guarded_client):
        return _guarded_client
    async with _get_client_lock():
        if _is_usable(_guarded_client):
            return _guarded_client
        import httpx

        _guarded_client = httpx.AsyncClient(
            timeout=_EXTERNAL_TIMEOUT,
            # Redirects are followed manually so every hop is re-validated and
            # re-pinned; trust_env is off because a proxy would silently undo
            # pinning by terminating the connection somewhere else entirely.
            follow_redirects=False,
            trust_env=False,
            headers={
                "User-Agent": EXTERNAL_USER_AGENT,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            # Keep-alive is DISABLED here on purpose. httpcore keys pooled
            # connections on scheme+host+port, and for pinned requests the host
            # is the IP — so two different hostnames resolving to the same IP
            # (shared CDN/hosting, extremely common) would reuse one TLS
            # connection. SNI is only read when the TLS connection is
            # established, so the second hostname's certificate would never be
            # validated while we still send its Host header. Paying a handshake
            # per request is the cost of keeping certificate validation honest;
            # this path is rate-limited by external_fetch_semaphore anyway.
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=0),
        )
        _register_shutdown_hook()
        return _guarded_client


def _absolutize(url: str, params: Optional[Dict[str, Any]]) -> str:
    """Fold ``params`` into the URL up front.

    Applying params per-request would either be lost on redirect or be appended
    twice when a redirect chain returns to the same URL. Folding them in once
    means every hop carries a complete, self-describing URL.
    """
    if not params:
        return url
    from urllib.parse import urlencode, urlparse, urlunparse, parse_qsl

    parsed = urlparse(url)
    merged = parse_qsl(parsed.query, keep_blank_values=True)
    merged.extend((str(k), str(v)) for k, v in params.items())
    return urlunparse((
        parsed.scheme, parsed.netloc, parsed.path, parsed.params,
        urlencode(merged), parsed.fragment,
    ))


async def guarded_fetch(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    method: str = "GET",
    json_body: Optional[Dict[str, Any]] = None,
    max_bytes: int = MAX_FETCH_BYTES,
    max_hops: int = MAX_REDIRECT_HOPS,
    timeout: Optional[float] = None,
) -> FetchResult:
    """Request a possibly-hostile URL safely.

    Validates and pins every hop, caps the body on the wire, and follows at most
    ``max_hops`` redirects. ``GET`` remains the default; ``POST`` exists only
    for read-shaped third-party search endpoints such as Exa. Raises
    :class:`BlockedUrlError` when a hop targets a forbidden address or the
    redirect budget is exhausted.
    """
    from urllib.parse import urljoin, urlparse

    normalized_method = str(method or "GET").upper()
    if normalized_method not in {"GET", "POST"}:
        raise BlockedUrlError(f"guarded egress 不支持 HTTP {normalized_method}")
    if normalized_method == "GET" and json_body is not None:
        raise BlockedUrlError("guarded GET 不接受 JSON 请求体")

    client = await get_guarded_client()
    current_url = _absolutize(url, params)

    for _ in range(max_hops + 1):
        validate_fetch_scheme(current_url)
        parsed = urlparse(current_url)
        try:
            port = parsed.port
        except ValueError:
            raise BlockedUrlError("URL 端口无效")

        addresses = await resolve_validated_addresses(parsed.hostname, port)
        pinned_url, host_header, sni_hostname = _pin_request(current_url, addresses[0])

        request_headers = dict(_IDENTITY_ENCODING)
        if headers:
            request_headers.update(headers)
        request_headers["Host"] = host_header

        async with external_fetch_semaphore():
            stream_kwargs = {
                "headers": request_headers,
                "extensions": {"sni_hostname": sni_hostname},
                "timeout": timeout if timeout is not None else _EXTERNAL_TIMEOUT,
            }
            if json_body is not None:
                stream_kwargs["json"] = json_body
            async with client.stream(
                normalized_method,
                pinned_url,
                **stream_kwargs,
            ) as response:
                _assert_peer_allowed(response, current_url)
                if response.status_code in _REDIRECT_STATUS:
                    location = response.headers.get("location")
                    if location:
                        current_url = urljoin(current_url, location)
                        continue
                body = await read_capped_body(response, max_bytes)
                return FetchResult(
                    response.status_code,
                    current_url,
                    decode_body(response, body),
                    response.headers,
                    getattr(response, "encoding", None),
                )

    raise BlockedUrlError(f"重定向次数超过上限（最多 {max_hops} 次跳转）")
