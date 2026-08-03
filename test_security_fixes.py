#!/usr/bin/env python3
"""
Focused regression tests for the security hardening work.

Covers:
  * web-search SSRF blocklist (scheme + address validation, redirect guard,
    response size cap)
  * the Jina reader double-prefix bug fix
  * harness JSON-Schema argument validation (missing required / wrong type /
    bad enum) and the batch-item cap

Run with: uv run python test_security_fixes.py
"""
import asyncio
import importlib.util
import ipaddress
import json
import os
import sys
import types
from urllib.parse import urlparse, urlunparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Stub the NoneBot runtime before importing anything from keytao_bot.
# ---------------------------------------------------------------------------
if "nonebot" not in sys.modules:
    _fake_nonebot = types.ModuleType("nonebot")

    class _FakeConfig:
        bot_api_token = "test-bot-token"
        bot_identity_secret = "test-bot-token"
        web_api_key = "test-web-api-key"
        keytao_api_base = "https://fake"

    class _FakeDriver:
        config = _FakeConfig()

    def _no_asgi_app():
        raise RuntimeError("no ASGI app in this test harness")

    _fake_nonebot.get_driver = lambda: _FakeDriver()
    _fake_nonebot.get_app = _no_asgi_app
    sys.modules["nonebot"] = _fake_nonebot

    _fake_log = types.ModuleType("nonebot.log")

    class _FakeLogger:
        def info(self, *a, **kw): pass
        def debug(self, *a, **kw): pass
        def warning(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def exception(self, *a, **kw): pass

    _fake_log.logger = _FakeLogger()
    sys.modules["nonebot.log"] = _fake_log


# ---------------------------------------------------------------------------
# Tiny check harness
# ---------------------------------------------------------------------------
passed = 0
failed = 0


def check(label: str, condition: bool) -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  [ok]   {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}")


def _load_module_from_path(name: str, relative_path: str):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


web_tools = _load_module_from_path(
    "web_search_tools_for_security_test",
    os.path.join("keytao_bot", "skills", "web-search", "tools.py"),
)

from keytao_bot.harness.tools import (  # noqa: E402
    MAX_BATCH_ITEMS,
    ToolContext,
    ToolExecutor,
    _validate_arguments,
    _validate_batch_size,
)


# ---------------------------------------------------------------------------
# Fake DNS + fake httpx client
# ---------------------------------------------------------------------------
import socket as _real_socket  # noqa: E402
from keytao_bot.utils import http_client  # noqa: E402


class _FakeSocketModule:
    """Drop-in replacement for the module-global ``socket`` in web-search tools."""

    SOCK_STREAM = _real_socket.SOCK_STREAM

    def __init__(self, mapping):
        self._mapping = mapping

    def getaddrinfo(self, host, port, family=0, type=0, proto=0, flags=0):
        if host not in self._mapping:
            raise _real_socket.gaierror(f"unknown host {host}")
        infos = []
        for address in self._mapping[host]:
            parsed = ipaddress.ip_address(address)
            if parsed.version == 4:
                infos.append((2, self.SOCK_STREAM, 6, "", (address, port or 0)))
            else:
                infos.append((30, self.SOCK_STREAM, 6, "", (address, port or 0, 0, 0)))
        return infos


class _PatchedDns:
    """Context manager swapping the module-global socket for a fake resolver."""

    def __init__(self, mapping):
        self._fake = _FakeSocketModule(mapping)
        self._original = None

    def __enter__(self):
        # Resolution lives in the shared guarded egress now.
        self._original = http_client.socket
        http_client.socket = self._fake
        return self

    def __exit__(self, *exc):
        http_client.socket = self._original
        return False


class _FakeNetworkStream:
    """Stand-in for httpx's transport stream, exposing the connected peer."""

    def __init__(self, server_addr):
        self._server_addr = server_addr

    def get_extra_info(self, key):
        if key == "server_addr" and self._server_addr:
            return (self._server_addr, 443)
        return None


class _FakeStreamResponse:
    def __init__(
        self,
        status_code=200,
        headers=None,
        body=b"",
        encoding="utf-8",
        peer_addr="93.184.216.34",
        raw_body=None,
    ):
        self.status_code = status_code
        self.headers = headers or {}
        self.encoding = encoding
        self._body = body
        # Raw (pre-decompression) bytes, as aiter_raw would yield them.
        self._raw_body = raw_body if raw_body is not None else body
        # peer_addr=None models a transport that exposes no peer at all.
        self.extensions = {"network_stream": _FakeNetworkStream(peer_addr)} if peer_addr else {}

    async def aiter_bytes(self):
        step = 65536
        for offset in range(0, len(self._body), step):
            yield self._body[offset:offset + step]

    async def aiter_raw(self):
        step = 65536
        for offset in range(0, len(self._raw_body), step):
            yield self._raw_body[offset:offset + step]

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    """Stands in for the shared guarded client.

    guarded_fetch pins connections to an IP, so the URL it hands the client has
    the IP in the netloc and the real hostname in the Host header. The responder
    is keyed on the logical URL, so reconstruct that from the Host header.
    """

    is_closed = False

    def __init__(self, responder):
        self._responder = responder
        self.requested = []
        self.pinned = []
        self.last_headers = {}
        self.last_extensions = {}

    def stream(self, method, url, params=None, headers=None, extensions=None, **kwargs):
        headers = headers or {}
        self.last_headers = headers
        self.last_extensions = extensions or {}
        self.pinned.append(url)
        logical = _unpin(url, headers.get("Host"))
        self.requested.append(logical)
        return _FakeStreamContext(self._responder(logical))


def _unpin(url, host_header):
    """Rebuild the logical URL from a pinned URL plus its Host header."""
    if not host_header:
        return url
    parsed = urlparse(url)
    return urlunparse((
        parsed.scheme, host_header, parsed.path, parsed.params, parsed.query, parsed.fragment,
    ))


class _InstalledGuardedClient:
    """Install a fake as the process-wide guarded client for one test."""

    def __init__(self, responder):
        self.client = _FakeClient(responder)

    def __enter__(self):
        self._original = http_client._guarded_client
        http_client._guarded_client = self.client
        return self.client

    def __exit__(self, *exc):
        http_client._guarded_client = self._original
        return False


# ---------------------------------------------------------------------------
# SSRF: address blocklist
# ---------------------------------------------------------------------------
def test_ssrf_ip_blocklist():
    print("\n[test] SSRF address blocklist")
    blocked = [
        "127.0.0.1",          # loopback
        "127.1.2.3",
        "10.0.0.7",           # RFC1918
        "10.255.255.255",
        "172.16.5.4",
        "192.168.1.1",
        "169.254.169.254",    # AWS/GCP metadata (link-local)
        "0.0.0.0",            # unspecified
        "224.0.0.1",          # multicast
        "240.0.0.1",          # reserved
        "::1",                # IPv6 loopback
        "fd00:ec2::254",      # AWS IPv6 metadata (unique-local)
        "fe80::1",            # IPv6 link-local
        "::ffff:127.0.0.1",   # IPv4-mapped loopback
        "::ffff:169.254.169.254",
        "2002:a00:1::",       # 6to4 wrapping 10.0.0.1
    ]
    for address in blocked:
        check(f"blocked {address}", web_tools._is_blocked_ip(ipaddress.ip_address(address)))

    allowed = ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2001:4860:4860::8888"]
    for address in allowed:
        check(f"allowed {address}", not web_tools._is_blocked_ip(ipaddress.ip_address(address)))


def test_ssrf_scheme_validation():
    print("\n[test] SSRF scheme validation")
    for url in ["file:///etc/passwd", "ftp://example.com/x", "gopher://127.0.0.1:11211/", "data:text/html,x"]:
        check(f"rejects {url}", web_tools._validate_scheme(url) is not None)
    for url in ["http://example.com/", "https://example.com/", "HTTPS://Example.com/"]:
        check(f"accepts {url}", web_tools._validate_scheme(url) is None)


def test_ssrf_host_resolution():
    print("\n[test] SSRF host resolution")

    async def _run():
        dns = {
            "public.example.com": ["93.184.216.34"],
            "evil.example.com": ["127.0.0.1"],
            "mixed.example.com": ["93.184.216.34", "10.0.0.9"],
            "meta.example.com": ["169.254.169.254"],
        }
        with _PatchedDns(dns):
            check(
                "public host resolves and is accepted",
                await web_tools._resolve_and_validate_host("public.example.com", 443) is None,
            )
            check(
                "host resolving to loopback is rejected",
                await web_tools._resolve_and_validate_host("evil.example.com", 443) is not None,
            )
            check(
                "host with one private answer is rejected (DNS rebinding)",
                await web_tools._resolve_and_validate_host("mixed.example.com", 443) is not None,
            )
            check(
                "host resolving to cloud metadata is rejected",
                await web_tools._resolve_and_validate_host("meta.example.com", 80) is not None,
            )
            check(
                "unresolvable host is rejected",
                await web_tools._resolve_and_validate_host("nx.example.com", 80) is not None,
            )
            check(
                "empty host is rejected",
                await web_tools._resolve_and_validate_host("", 80) is not None,
            )
            # Literal IPs must never reach DNS at all.
            check(
                "literal private IP is rejected without DNS",
                await web_tools._resolve_and_validate_host("10.0.0.5", 80) is not None,
            )
            check(
                "literal public IP is accepted",
                await web_tools._resolve_and_validate_host("8.8.8.8", 80) is None,
            )
            check(
                "bracketed IPv6 loopback is rejected",
                await web_tools._resolve_and_validate_host("[::1]", 80) is not None,
            )
            check(
                "full target validation accepts a public URL",
                await web_tools._validate_fetch_target("https://public.example.com/page") is None,
            )
            check(
                "full target validation rejects a private URL",
                await web_tools._validate_fetch_target("http://evil.example.com/") is not None,
            )

    asyncio.run(_run())


def test_web_fetch_rejects_dangerous_targets():
    print("\n[test] web_fetch target rejection")

    async def _run():
        with _PatchedDns({}):
            for url in [
                "file:///etc/passwd",
                "ftp://example.com/x",
                "http://127.0.0.1:8080/admin",
                "https://10.0.0.5/internal",
                "http://169.254.169.254/latest/meta-data/",
                "http://[::1]:9000/",
                "https://192.168.1.1/router",
            ]:
                result = await web_tools.web_fetch(url)
                check(f"web_fetch refuses {url}", result.get("success") is False and bool(result.get("error")))

    asyncio.run(_run())


def test_redirect_guard_and_body_cap():
    print("\n[test] redirect guard and body cap")

    async def _run():
        dns = {
            "public.example.com": ["93.184.216.34"],
            "evil.example.com": ["127.0.0.1"],
        }
        with _PatchedDns(dns):
            # 1. A redirect chain that never terminates is capped.
            def always_redirect(url):
                return _FakeStreamResponse(
                    302,
                    headers={"location": "https://public.example.com/next"},
                )

            with _InstalledGuardedClient(always_redirect) as client:
                try:
                    await http_client.guarded_fetch("https://public.example.com/start")
                    check("redirect loop is capped", False)
                except http_client.BlockedUrlError as error:
                    check("redirect loop is capped", "重定向次数超过上限" in str(error))
                check(
                    f"redirect budget is 1 + {http_client.MAX_REDIRECT_HOPS} requests",
                    len(client.requested) == http_client.MAX_REDIRECT_HOPS + 1,
                )

            # 2. A public page that redirects into the private network is blocked.
            def redirect_to_private(url):
                if url.endswith("/start"):
                    return _FakeStreamResponse(302, headers={"location": "http://127.0.0.1:8080/admin"})
                return _FakeStreamResponse(200, body=b"<html>internal</html>")

            with _InstalledGuardedClient(redirect_to_private) as client:
                try:
                    await http_client.guarded_fetch("https://public.example.com/start")
                    check("redirect into private network is blocked", False)
                except http_client.BlockedUrlError as error:
                    check("redirect into private network is blocked", "禁止访问内网" in str(error))
                check("blocked redirect never issued the second request", len(client.requested) == 1)

            # 3. A single safe hop is followed and re-validated.
            def one_safe_hop(url):
                if url.endswith("/start"):
                    return _FakeStreamResponse(301, headers={"location": "/final"})
                return _FakeStreamResponse(200, body=b"<html><title>ok</title>body</html>")

            with _InstalledGuardedClient(one_safe_hop):
                result = await http_client.guarded_fetch("https://public.example.com/start")
            check("safe redirect is followed", result.url == "https://public.example.com/final")
            check("redirected body is returned", "<title>ok</title>" in result.text)

            # 4. Bodies are truncated to MAX_FETCH_BYTES before any parsing.
            oversized = b"x" * (http_client.MAX_FETCH_BYTES + 1024 * 1024)

            def huge_body(url):
                return _FakeStreamResponse(200, body=oversized, raw_body=oversized)

            with _InstalledGuardedClient(huge_body):
                big = await http_client.guarded_fetch("https://public.example.com/big")
            check("MAX_FETCH_BYTES is 2 MB", http_client.MAX_FETCH_BYTES == 2 * 1024 * 1024)
            check("body is capped at MAX_FETCH_BYTES", len(big.text) <= http_client.MAX_FETCH_BYTES)

    asyncio.run(_run())


def test_connection_is_pinned_to_validated_ip():
    """The socket target must be an already-validated IP, not a hostname."""
    print("\n[test] connections are pinned to a validated IP")

    async def _run():
        with _PatchedDns({"public.example.com": ["93.184.216.34"]}):
            with _InstalledGuardedClient(lambda url: _FakeStreamResponse(200, body=b"ok")) as client:
                await http_client.guarded_fetch("https://public.example.com/page?a=1")

            pinned = client.pinned[0]
            check("request URL carries the validated IP", "93.184.216.34" in pinned)
            check("request URL no longer carries the hostname", "public.example.com" not in pinned)
            check("Host header preserves the real hostname",
                  client.last_headers.get("Host") == "public.example.com")
            check("SNI preserves the real hostname",
                  client.last_extensions.get("sni_hostname") == "public.example.com")
            check("path and query survive pinning", "/page?a=1" in pinned)

        # A host whose answers include one private address is refused outright,
        # so a split-horizon answer can never be partially trusted.
        with _PatchedDns({"mixed.example.com": ["93.184.216.34", "10.0.0.5"]}):
            with _InstalledGuardedClient(lambda url: _FakeStreamResponse(200)) as client:
                blocked = False
                try:
                    await http_client.guarded_fetch("https://mixed.example.com/")
                except http_client.BlockedUrlError:
                    blocked = True
            check("mixed public/private resolution is refused", blocked)
            check("no request was issued for the mixed host", not client.requested)

    asyncio.run(_run())


def test_params_survive_redirects():
    """Query parameters must not be lost or duplicated across hops."""
    print("\n[test] params survive redirects")

    async def _run():
        with _PatchedDns({"search.example.com": ["93.184.216.34"]}):
            seen = []

            def redirect_once(url):
                seen.append(url)
                if "/search" in url and "/redirected" not in url:
                    return _FakeStreamResponse(
                        302, headers={"location": "https://search.example.com/redirected?q=nonebot"},
                    )
                return _FakeStreamResponse(200, body=b"results")

            with _InstalledGuardedClient(redirect_once):
                result = await http_client.guarded_fetch(
                    "https://search.example.com/search", params={"q": "nonebot"},
                )

            check("params are folded into the first request", "q=nonebot" in seen[0])
            check("redirect target keeps its own query", "q=nonebot" in seen[1])
            check("final URL is the redirect target", result.url.endswith("/redirected?q=nonebot"))
            check("params are never appended twice", seen[1].count("q=nonebot") == 1)

            # Returning to the same URL must not accumulate duplicated params.
            hops = []

            def bounce(url):
                hops.append(url)
                if len(hops) == 1:
                    return _FakeStreamResponse(
                        302, headers={"location": "https://search.example.com/s?q=x"},
                    )
                return _FakeStreamResponse(200, body=b"ok")

            with _InstalledGuardedClient(bounce):
                await http_client.guarded_fetch(
                    "https://search.example.com/s", params={"q": "x"},
                )
            check("bouncing back to the same URL does not duplicate params",
                  all(hop.count("q=x") == 1 for hop in hops))

    asyncio.run(_run())


def test_jina_reader_prefix_applied_once():
    print("\n[test] Jina reader prefix")
    url = web_tools._jina_reader_url("https://example.com/a?b=c")
    check("prefix applied once", url.count("r.jina.ai") == 1)
    check("target preserved verbatim", url == "https://r.jina.ai/https://example.com/a?b=c")
    check("no bogus http downgrade", "http://r.jina.ai" not in url)

    already = web_tools._jina_reader_url("https://r.jina.ai/https://example.com/a")
    check("idempotent on an already-wrapped URL", already.count("r.jina.ai") == 1)

    legacy = web_tools._jina_reader_url("https://r.jina.ai/http://r.jina.ai/http://example.com/a")
    check("legacy double prefix is collapsed", legacy == "https://r.jina.ai/http://example.com/a")

    bare = web_tools._jina_reader_url("example.com/a")
    check("scheme-less target gets https", bare == "https://r.jina.ai/https://example.com/a")


# ---------------------------------------------------------------------------
# Harness JSON-Schema validation
# ---------------------------------------------------------------------------
_DEMO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "demo_tool",
        "parameters": {
            "type": "object",
            "properties": {
                "word": {"type": "string"},
                "count": {"type": "integer"},
                "action": {"type": "string", "enum": ["Create", "Change", "Delete"]},
            },
            "required": ["word"],
        },
    },
}


def test_schema_validation_rejects_bad_arguments():
    print("\n[test] harness JSON-Schema validation")

    check(
        "valid arguments pass",
        _validate_arguments("demo_tool", {"word": "你好", "count": 3, "action": "Create"}, _DEMO_SCHEMA) is None,
    )

    missing = _validate_arguments("demo_tool", {"count": 1}, _DEMO_SCHEMA)
    check("missing required argument is rejected", bool(missing) and missing["invalidArguments"] is True)
    check("missing required argument reports the tool", missing["tool"] == "demo_tool")
    check("missing required argument message is Chinese", "参数校验失败" in missing["message"])

    wrong_type = _validate_arguments("demo_tool", {"word": "你好", "count": "three"}, _DEMO_SCHEMA)
    check("wrong argument type is rejected", bool(wrong_type) and wrong_type["invalidArguments"] is True)

    bad_enum = _validate_arguments("demo_tool", {"word": "你好", "action": "Explode"}, _DEMO_SCHEMA)
    check("bad enum value is rejected", bool(bad_enum) and bad_enum["invalidArguments"] is True)

    check(
        "unknown extra properties are tolerated",
        _validate_arguments("demo_tool", {"word": "你好", "platform": "qq", "platform_id": "1"}, _DEMO_SCHEMA) is None,
    )
    check(
        "non-dict arguments are rejected",
        (_validate_arguments("demo_tool", ["not", "a", "dict"], _DEMO_SCHEMA) or {}).get("invalidArguments") is True,
    )
    check("no schema means no validation", _validate_arguments("demo_tool", {}, None) is None)
    check("empty parameters mean no validation", _validate_arguments("demo_tool", {}, {"function": {}}) is None)


def test_executor_schema_validation_blocks_dispatch():
    print("\n[test] ToolExecutor schema gate")

    async def _run():
        calls = []

        async def fake_tool(**kwargs):
            calls.append(kwargs)
            return {"success": True}

        executor = ToolExecutor(
            lambda name: fake_tool if name == "demo_tool" else None,
            frozenset(),
            get_tool_schema=lambda name: _DEMO_SCHEMA if name == "demo_tool" else None,
        )

        rejected = json.loads(await executor.call("demo_tool", {"count": 1}, ToolContext()))
        check("executor returns invalidArguments", rejected.get("invalidArguments") is True)
        check("executor did not dispatch the bad call", len(calls) == 0)

        accepted = json.loads(await executor.call("demo_tool", {"word": "你好"}, ToolContext()))
        check("executor dispatches valid arguments", accepted.get("success") is True)
        check("valid call reached the tool", len(calls) == 1)

        # Backwards compatibility: two positional args, no schema getter.
        legacy = ToolExecutor(lambda name: fake_tool if name == "demo_tool" else None, frozenset())
        legacy_result = json.loads(await legacy.call("demo_tool", {"count": 1}, ToolContext()))
        check("legacy construction skips validation", legacy_result.get("success") is True)

    asyncio.run(_run())


def test_batch_item_cap():
    print("\n[test] batch item cap")
    check("MAX_BATCH_ITEMS is 200", MAX_BATCH_ITEMS == 200)

    oversized_items = [{"word": f"词{i}", "code": "abc"} for i in range(MAX_BATCH_ITEMS + 1)]
    blocked = _validate_batch_size("keytao_batch_add_to_draft", {"items": oversized_items})
    check("201 items are blocked", bool(blocked) and blocked["policyBlocked"] is True)
    check("cap message names the limit", "上限 200 条" in blocked["message"])
    check("cap message names the actual count", "201 条" in blocked["message"])

    check(
        "200 items are allowed",
        _validate_batch_size("keytao_batch_add_to_draft", {"items": oversized_items[:MAX_BATCH_ITEMS]}) is None,
    )
    check(
        "201 draft ids are blocked",
        (_validate_batch_size("keytao_batch_remove_draft_items", {"ids": list(range(201))}) or {}).get("policyBlocked")
        is True,
    )
    check(
        "201 audit items are blocked",
        (_validate_batch_size("keytao_audit_draft_items", {"items": oversized_items}) or {}).get("policyBlocked")
        is True,
    )
    check("unrelated tools are untouched", _validate_batch_size("web_search", {"items": oversized_items}) is None)

    async def _run():
        calls = []

        async def fake_tool(**kwargs):
            calls.append(kwargs)
            return {"success": True}

        executor = ToolExecutor(
            lambda name: fake_tool,
            frozenset({"keytao_batch_add_to_draft"}),
        )
        result = json.loads(
            await executor.call(
                "keytao_batch_add_to_draft",
                {"items": oversized_items},
                ToolContext(platform="qq", user_id="123", current_message="批量加词"),
            )
        )
        check("executor blocks the oversized batch", result.get("policyBlocked") is True)
        check("oversized batch never reached the tool", len(calls) == 0)

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Web API auth surface
# ---------------------------------------------------------------------------
def _load_web_api():
    """Import web_api with its heavy siblings stubbed out.

    The stubs go into ``sys.modules`` before the import so the relative imports
    inside ``web_api`` resolve to placeholders instead of dragging in the whole
    chat plugin. Run this test last — the stubs stay in ``sys.modules``.
    """
    import importlib

    import keytao_bot.plugins  # noqa: F401
    import keytao_bot.utils  # noqa: F401

    openai_chat = types.ModuleType("keytao_bot.plugins.openai_chat")
    openai_chat.get_ai_response_core = None
    openai_chat.conversation_state_store = None
    openai_chat.MAX_HISTORY_MESSAGES = 10
    openai_chat._clear_conversation_state = lambda *args, **kwargs: None
    openai_chat.conversation_message_locks = {}
    openai_chat.current_history_generation = None
    openai_chat.current_memory_context = None
    openai_chat.current_memory_generation = None
    openai_chat.draft_actor_message_locks = {}
    openai_chat.handle_pending_message_core = None
    openai_chat.history_store = None
    openai_chat.memory_store = None
    openai_chat.remember_conversation = None
    openai_chat.schedule_memory_compaction = None
    sys.modules.setdefault("keytao_bot.plugins.openai_chat", openai_chat)

    batch_review = types.ModuleType("keytao_bot.utils.keytao_batch_review")
    batch_review.review_keytao_batch_with_llm = None
    sys.modules.setdefault("keytao_bot.utils.keytao_batch_review", batch_review)

    history_store = types.ModuleType("keytao_bot.utils.history_store")
    history_store.get_history_store = lambda: None
    sys.modules.setdefault("keytao_bot.utils.history_store", history_store)

    memory_store = types.ModuleType("keytao_bot.utils.memory_store")
    memory_store.ChatMemoryContext = type("ChatMemoryContext", (), {})
    sys.modules.setdefault("keytao_bot.utils.memory_store", memory_store)

    return importlib.import_module("keytao_bot.plugins.web_api")


def _test_legacy_web_api_auth_surface():
    print("\n[test] web API auth surface")
    import hashlib
    import hmac
    import time

    web_api = _load_web_api()
    from fastapi import HTTPException

    def status_of(fn, *args):
        try:
            fn(*args)
            return 200
        except HTTPException as error:
            return error.status_code

    check("user signature is required by default", web_api.WEB_API_REQUIRE_SIGNATURE is True)

    # --- bearer layer ---
    check("correct bearer accepted", status_of(web_api._check_auth, "Bearer test-web-api-key") == 200)
    check("wrong bearer is 401", status_of(web_api._check_auth, "Bearer wrong") == 401)
    check("missing bearer is 401", status_of(web_api._check_auth, None) == 401)
    check("non-ascii bearer is 401 not a crash", status_of(web_api._check_auth, "Bearer 密码") == 401)

    saved_key = web_api.WEB_API_KEY
    web_api.WEB_API_KEY = ""
    check("unset WEB_API_KEY disables the API with 503", status_of(web_api._check_auth, "Bearer anything") == 503)
    web_api.WEB_API_KEY = saved_key

    # --- user signature layer ---
    # Signed message is "{METHOD}.{path}.{user_id}.{timestamp}"; method and path
    # are bound in so a signature cannot be replayed onto another route.
    token = "test-bot-token"
    user_id = "user_abc123"
    timestamp = str(int(time.time()))
    CHAT = ("POST", "/api/chat")
    HISTORY = ("DELETE", "/api/chat/history")

    def sign(method, path, uid, ts, key=None):
        message = f"{method.upper()}.{path}.{uid}.{ts}"
        return hmac.new((key or token).encode(), message.encode(), hashlib.sha256).hexdigest()

    verify = web_api._verify_user_signature
    chat_sig = sign(*CHAT, user_id, timestamp)
    history_sig = sign(*HISTORY, user_id, timestamp)

    check(
        "canonical message format matches the spec",
        web_api.build_user_signature_message("post", "/api/chat", user_id, timestamp)
        == f"POST./api/chat.{user_id}.{timestamp}",
    )

    check(
        "valid POST /api/chat signature accepted",
        status_of(verify, *CHAT, user_id, user_id, timestamp, chat_sig) == 200,
    )
    check(
        "valid DELETE /api/chat/history signature accepted",
        status_of(verify, *HISTORY, user_id, user_id, timestamp, history_sig) == 200,
    )
    check(
        "anonymous request skips signature",
        status_of(verify, *CHAT, None, None, None, None) == 200,
    )
    check(
        "missing signature headers is 401",
        status_of(verify, *CHAT, user_id, None, None, None) == 401,
    )
    check(
        "tampered signature is 401",
        status_of(verify, *CHAT, user_id, user_id, timestamp, "0" * 64) == 401,
    )
    check(
        "identity mismatch is 401",
        status_of(verify, *CHAT, user_id, "someone_else", timestamp, chat_sig) == 401,
    )
    check(
        "non-numeric timestamp is 401",
        status_of(verify, *CHAT, user_id, user_id, "abc", chat_sig) == 401,
    )
    check(
        "uppercase hex signature accepted",
        status_of(verify, *CHAT, user_id, user_id, timestamp, chat_sig.upper()) == 200,
    )

    # --- cross-route / cross-method replay ---
    check(
        "chat signature cannot be replayed on /api/chat/history",
        status_of(verify, *HISTORY, user_id, user_id, timestamp, chat_sig) == 401,
    )
    check(
        "history signature cannot be replayed on /api/chat",
        status_of(verify, *CHAT, user_id, user_id, timestamp, history_sig) == 401,
    )
    check(
        "same path with a different method is rejected",
        status_of(verify, "DELETE", "/api/chat", user_id, user_id, timestamp, chat_sig) == 401,
    )
    check(
        "same method on a different path is rejected",
        status_of(verify, "POST", "/api/chat/history", user_id, user_id, timestamp, chat_sig) == 401,
    )
    check(
        "trailing-slash path variant is rejected",
        status_of(verify, "POST", "/api/chat/", user_id, user_id, timestamp, chat_sig) == 401,
    )
    check(
        "method casing does not weaken the binding",
        status_of(verify, "post", "/api/chat", user_id, user_id, timestamp, chat_sig) == 200,
    )
    check(
        "another user's signature on the same route is rejected",
        status_of(verify, *CHAT, user_id, user_id, timestamp, sign(*CHAT, "other_user", timestamp)) == 401,
    )

    stale = str(int(time.time()) - 600)
    check(
        "expired timestamp is 401",
        status_of(verify, *CHAT, user_id, user_id, stale, sign(*CHAT, user_id, stale)) == 401,
    )

    # A leaked WEB_API_KEY must not be enough to forge a user identity.
    check(
        "signature keyed by WEB_API_KEY is rejected",
        status_of(
            verify, *CHAT, user_id, user_id, timestamp,
            sign(*CHAT, user_id, timestamp, key="test-web-api-key"),
        ) == 401,
    )

    original_get_bot_token = web_api.http_client.get_bot_token
    web_api.http_client.get_bot_token = lambda: None
    check(
        "missing BOT_API_TOKEN fails closed with 503",
        status_of(verify, *CHAT, user_id, user_id, timestamp, chat_sig) == 503,
    )
    web_api.http_client.get_bot_token = original_get_bot_token

    web_api.WEB_API_REQUIRE_SIGNATURE = False
    check("flag off skips verification", status_of(verify, *CHAT, user_id, None, None, None) == 200)
    web_api.WEB_API_REQUIRE_SIGNATURE = True

    for value in ["0", "false", "FALSE", "no", "off", "", " Off "]:
        check(f"_as_bool({value!r}) is False", web_api._as_bool(value, True) is False)
    for value in ["1", "true", "yes", "on", True]:
        check(f"_as_bool({value!r}) is True", web_api._as_bool(value, False) is True)
    check("_as_bool(None) keeps the default", web_api._as_bool(None, True) is True)

    source = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "keytao_bot", "plugins", "web_api.py"),
        encoding="utf-8",
    ).read()
    check("wildcard allow_origin_regex removed", "allow_origin_regex" not in source)
    check("empty-key auth bypass removed", "if WEB_API_KEY and authorization" not in source)


def test_web_api_auth_surface():
    """Current nonce/body-bound web identity contract fails closed."""
    print("\n[test] web API auth surface")
    import time

    web_api = _load_web_api()
    from fastapi import HTTPException
    from keytao_bot.utils.web_identity import build_web_identity_signature

    def status_of(fn, *args):
        try:
            fn(*args)
            return 200
        except HTTPException as error:
            return error.status_code

    check("correct bearer accepted", status_of(web_api._check_auth, "Bearer test-web-api-key") == 200)
    check("wrong bearer is 401", status_of(web_api._check_auth, "Bearer wrong") == 401)
    check("missing bearer is 401", status_of(web_api._check_auth, None) == 401)

    saved_key = web_api.WEB_API_KEY
    web_api.WEB_API_KEY = ""
    check(
        "unset WEB_API_KEY disables the API with 503",
        status_of(web_api._check_auth, "Bearer anything") == 503,
    )
    web_api.WEB_API_KEY = saved_key

    user_id = "user_abc123"
    timestamp = str(int(time.time()))
    raw_body = b'{"message":"hello","user_id":"user_abc123"}'

    def identity_status(*, nonce, signature, method="POST", path="/api/chat"):
        try:
            web_api._check_web_user_identity(
                user_id,
                header_user_id=user_id,
                timestamp=timestamp,
                nonce=nonce,
                signature=signature,
                method=method,
                path=path,
                raw_body=raw_body,
            )
            return 200
        except HTTPException as error:
            return error.status_code

    nonce = "01" * 16
    signature = build_web_identity_signature(
        "test-bot-token",
        method="POST",
        path="/api/chat",
        user_id=user_id,
        timestamp=timestamp,
        nonce=nonce,
        raw_body=raw_body,
    )
    check(
        "valid nonce/body-bound identity accepted",
        identity_status(nonce=nonce, signature=signature) == 200,
    )
    check(
        "the same nonce cannot be replayed",
        identity_status(nonce=nonce, signature=signature) == 401,
    )
    check(
        "cross-method replay is rejected",
        identity_status(nonce="02" * 16, signature=signature, method="DELETE") == 401,
    )
    check(
        "anonymous request carries no delegated identity",
        web_api._check_web_user_identity(
            None,
            header_user_id=None,
            timestamp=None,
            nonce=None,
            signature=None,
            method="POST",
            path="/api/chat",
            raw_body=b"",
        ) is None,
    )

    source = open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "keytao_bot",
            "plugins",
            "web_api.py",
        ),
        encoding="utf-8",
    ).read()
    check("wildcard allow_origin_regex removed", "allow_origin_regex" not in source)
    check("empty-key auth bypass removed", "if WEB_API_KEY and authorization" not in source)


# ---------------------------------------------------------------------------
# Adversarial review follow-ups
# ---------------------------------------------------------------------------
def test_dns_rebinding_peer_check():
    """A host that validates public but connects private must be aborted."""
    print("\n[test] SSRF: DNS rebinding (validate public, connect private)")

    # Validation sees a public address...
    with _PatchedDns({"rebind.example": ["93.184.216.34"]}):
        error = asyncio.run(web_tools._validate_fetch_target("https://rebind.example/x"))
    check("first resolution passes validation", error is None)

    # ...but the socket actually landed on the metadata service.
    rebound = _FakeStreamResponse(peer_addr="169.254.169.254", body=b"secrets")
    blocked = False
    try:
        web_tools._assert_peer_allowed(rebound, "https://rebind.example/x")
    except web_tools.BlockedUrlError:
        blocked = True
    check("rebound connection to metadata IP is aborted", blocked)

    for addr in ("127.0.0.1", "10.1.2.3", "192.168.5.5", "::1", "::ffff:127.0.0.1"):
        response = _FakeStreamResponse(peer_addr=addr)
        was_blocked = False
        try:
            web_tools._assert_peer_allowed(response, "https://public.example/")
        except web_tools.BlockedUrlError:
            was_blocked = True
        check(f"rebound connection to {addr} is aborted", was_blocked)

    ok_response = _FakeStreamResponse(peer_addr="93.184.216.34")
    survived = True
    try:
        web_tools._assert_peer_allowed(ok_response, "https://public.example/")
    except web_tools.BlockedUrlError:
        survived = False
    check("genuine public peer is allowed", survived)

    # With IP pinning the destination is fixed before the request is emitted,
    # so an absent peer address is no longer a security gap -- it just means the
    # transport does not report one (UDS / ASGI / MockTransport).
    no_info = _FakeStreamResponse(peer_addr=None)
    tolerated = True
    try:
        web_tools._assert_peer_allowed(no_info, "https://public.example/")
    except web_tools.BlockedUrlError:
        tolerated = False
    check("missing peer info is tolerated once pinning guarantees the target", tolerated)


def test_compression_bomb_is_bounded():
    """Raw-byte cap plus bounded inflate keeps a zip bomb from exploding."""
    print("\n[test] compression bomb bounded")
    import zlib

    payload = b"A" * (64 * 1024 * 1024)  # 64 MB of decompressed output
    compressor = zlib.compressobj(9, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    bomb = compressor.compress(payload) + compressor.flush()
    check("bomb compresses to a small wire payload", len(bomb) < 128 * 1024)

    response = _FakeStreamResponse(
        headers={"content-encoding": "gzip"},
        body=payload,
        raw_body=bomb,
    )
    body = asyncio.run(web_tools._read_capped_body(response))
    check(
        "decompressed body is capped at MAX_FETCH_BYTES",
        len(body) <= web_tools.MAX_FETCH_BYTES,
    )
    check("decompressed body is far below the raw bomb output", len(body) < len(payload))

    # An unbounded-decoder encoding is refused rather than expanded.
    br = _FakeStreamResponse(headers={"content-encoding": "br"}, raw_body=b"\x00" * 1024)
    check("unknown encoding yields no body", asyncio.run(web_tools._read_capped_body(br)) == b"")

    # Identity bodies still flow through untouched.
    plain = _FakeStreamResponse(body=b"hello", raw_body=b"hello")
    check("identity body is unchanged", asyncio.run(web_tools._read_capped_body(plain)) == b"hello")

    check(
        "requests advertise Accept-Encoding: identity",
        web_tools._IDENTITY_ENCODING.get("Accept-Encoding") == "identity",
    )


def test_non_object_arguments_do_not_crash_policy():
    """A JSON array from the model must return a correctable error, not raise."""
    print("\n[test] harness: non-object arguments")
    from keytao_bot.harness.tools import ToolContext, ToolExecutor

    executor = ToolExecutor(lambda name: None, frozenset())
    context = ToolContext(platform="qq", user_id="1", current_message="删除全部草稿")

    for bad in ([], ["a", "b"], "string", 42, None):
        raised = False
        raw = None
        try:
            raw = asyncio.run(executor.call("keytao_batch_remove_draft_items", bad, context))
        except Exception:
            raised = True
        check(f"{type(bad).__name__} arguments do not raise", not raised)
        if raw is not None:
            payload = json.loads(raw)
            check(
                f"{type(bad).__name__} arguments report invalidArguments",
                payload.get("invalidArguments") is True,
            )


def test_retry_policy_is_idempotency_aware():
    """Writes must not be replayed; reads still retry."""
    print("\n[test] http_client retry idempotency")
    from keytao_bot.utils import http_client

    check(
        "GET is replay-safe by default",
        "GET" in http_client._IDEMPOTENT_METHODS,
    )
    for method in ("POST", "PUT", "PATCH", "DELETE"):
        check(
            f"{method} is not replay-safe by default",
            method not in http_client._IDEMPOTENT_METHODS,
        )

    source = open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "keytao_bot", "plugins", "account_bind.py",
        ),
        encoding="utf-8",
    ).read()
    check(
        "bind verify does not opt into retries",
        "idempotent=True" not in source,
    )


def test_fly_key_variants_follow_pronunciation_correction():
    """Corrected readings must not keep fly-key groups from the old reading."""
    print("\n[test] flyKeyVariants follow contextual correction")
    from keytao_bot.utils.keytao_encoding import _filter_fly_key_variants

    derivable = {"hyf", "hyfi"}
    stale = [{"baseCode": "sfb", "codes": ["sfb", "sfbo"]}]
    check("stale fly-key group is dropped", _filter_fly_key_variants(stale, derivable) == [])

    anchored = [{"baseCode": "hyf", "codes": ["hyf", "hyfz"]}]
    check("anchored fly-key group survives", _filter_fly_key_variants(anchored, derivable) == anchored)

    by_code = [{"baseCode": "zzz", "codes": ["hyfi"]}]
    check("group anchored via codes survives", _filter_fly_key_variants(by_code, derivable) == by_code)
    check("non-list input yields empty list", _filter_fly_key_variants(None, derivable) == [])




def test_outbound_clients_are_proxy_free_and_hop_validated():
    """There must be exactly one guarded egress, and it must not use a proxy."""
    print("\n[test] outbound egress hardening")
    root = os.path.dirname(os.path.abspath(__file__))
    web_source = open(
        os.path.join(root, "keytao_bot", "skills", "web-search", "tools.py"), encoding="utf-8",
    ).read()
    review_source = open(
        os.path.join(root, "keytao_bot", "utils", "keytao_review.py"), encoding="utf-8",
    ).read()
    client_source = open(
        os.path.join(root, "keytao_bot", "utils", "http_client.py"), encoding="utf-8",
    ).read()

    check("web-search builds no client of its own", "httpx.AsyncClient(" not in web_source)
    check("web-search routes through the guarded egress",
          "http_client.guarded_fetch(" in web_source)
    check("review search routes through the guarded egress",
          "http_client.guarded_fetch(" in review_source)
    check("review module no longer uses the unguarded external client",
          "get_external_client()" not in review_source)

    def kwarg_lines(name):
        return [
            line for line in client_source.splitlines()
            if line.strip().startswith(f"{name}=") and not line.strip().startswith("#")
        ]

    guarded_block = client_source[client_source.index("async def get_guarded_client("):]
    guarded_block = guarded_block[:guarded_block.index("def _absolutize(")]
    check("guarded client disables proxy env", "trust_env=False" in guarded_block)
    check("guarded client disables automatic redirects",
          "follow_redirects=False" in guarded_block)
    check("the vendor client is documented as trusted-hosts-only",
          "HARD-CODED, trusted third-party endpoints only" in client_source)
    check("the vendor client validates redirect hops",
          "transport=_build_validating_transport()" in client_source)
    check("its residual (no pinning) limitation is documented",
          "does NOT do is pin" in client_source)
    check("guarded egress pins the connection", "sni_hostname" in client_source)
    check("guarded egress sets an explicit Host header",
          '"Host"' in client_source or "'Host'" in client_source)


def test_search_provider_redirect_is_hop_validated():
    """A search endpoint redirecting into the private network must be blocked."""
    print("\n[test] search provider redirect is hop-validated")

    def responder(url):
        if "duckduckgo" in url:
            return _FakeStreamResponse(
                status_code=302,
                headers={"location": "http://169.254.169.254/latest/meta-data/"},
            )
        return _FakeStreamResponse(body=b"<html>ok</html>")

    with _PatchedDns({
        "html.duckduckgo.com": ["93.184.216.34"],
        "169.254.169.254": ["169.254.169.254"],
    }):
        with _InstalledGuardedClient(responder) as client:
            blocked = False
            try:
                asyncio.run(web_tools._get_text(
                    web_tools.DUCKDUCKGO_HTML_ENDPOINT, params={"q": "x"},
                ))
            except web_tools.BlockedUrlError:
                blocked = True

    check("search redirect into metadata IP is blocked", blocked)
    check("metadata URL was never requested", all(
        "169.254.169.254" not in url for url in client.requested
    ))


def test_review_page_fetch_is_guarded():
    """Search-result pages fetched during review are attacker-influencable."""
    print("\n[test] review page fetch is guarded")
    from keytao_bot.utils import keytao_review

    async def _run():
        # A result URL that resolves straight to the metadata service.
        with _PatchedDns({"evil.example.com": ["169.254.169.254"]}):
            with _InstalledGuardedClient(
                lambda url: _FakeStreamResponse(200, body=b"SECRET")
            ) as client:
                text = await keytao_review._fetch_text("http://evil.example.com/page")
        check("blocked review fetch returns no text", text == "")
        check("blocked review fetch issued no request", not client.requested)

        # A well-behaved public page still works.
        with _PatchedDns({"good.example.com": ["93.184.216.34"]}):
            with _InstalledGuardedClient(
                lambda url: _FakeStreamResponse(200, body=b"<html>hello world</html>")
            ):
                text = await keytao_review._fetch_text("http://good.example.com/page")
        check("allowed review fetch returns text", "hello world" in text)

    asyncio.run(_run())



def test_external_client_validates_every_redirect_hop():
    """The trusted-vendor client must not be bounceable into the private network.

    get_external_client follows redirects (vendor APIs use them), so validation
    cannot live at the call site: only the first hop is known there. A validating
    transport sees every hop, including redirect-generated ones.
    """
    print("\n[test] external client validates every redirect hop")

    import httpx

    transport = http_client._build_validating_transport()
    dispatched = []

    class _RecordingInner:
        async def handle_async_request(self, request):
            dispatched.append(str(request.url))
            return "RESPONSE"

        async def aclose(self):
            return None

    transport._inner = _RecordingInner()

    async def _run():
        with _PatchedDns({
            "api.vendor.example": ["93.184.216.34"],
            "evil.vendor.example": ["169.254.169.254"],
            "internal.vendor.example": ["10.0.0.5"],
        }):
            # A legitimate vendor host is dispatched.
            result = await transport.handle_async_request(
                httpx.Request("GET", "https://api.vendor.example/v1/data")
            )
            check("trusted vendor request is dispatched", result == "RESPONSE")
            check("trusted vendor request reached the inner transport", len(dispatched) == 1)

            # A redirect hop into the internal network is refused before dispatch.
            for host in ("evil.vendor.example", "internal.vendor.example"):
                blocked = False
                try:
                    await transport.handle_async_request(
                        httpx.Request("GET", f"https://{host}/next")
                    )
                except http_client.BlockedUrlError:
                    blocked = True
                check(f"redirect hop to {host} is refused", blocked)
            check("refused hops never reached the inner transport", len(dispatched) == 1)

            # A literal internal address is refused too.
            blocked = False
            try:
                await transport.handle_async_request(
                    httpx.Request("GET", "http://169.254.169.254/latest/meta-data/")
                )
            except http_client.BlockedUrlError:
                blocked = True
            check("literal metadata address is refused", blocked)
            check("still nothing extra dispatched", len(dispatched) == 1)

    asyncio.run(_run())

    source = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "keytao_bot", "utils", "http_client.py"),
        encoding="utf-8",
    ).read()
    check("external client installs the validating transport",
          "transport=_build_validating_transport()" in source)
    check("the residual rebinding risk is documented",
          "rebind" in source)




def test_special_purpose_ranges_are_blocked():
    """IANA special-purpose ranges the ipaddress is_* properties miss.

    100.64.0.0/10 (CGNAT) reports is_private=False, and Alibaba Cloud serves
    instance metadata at 100.100.100.200 inside it -- a probe confirmed
    guarded_fetch returned live metadata before this was fixed. IPv6 fec0::/10
    is likewise flagged by no property.
    """
    print("\n[test] IANA special-purpose ranges")

    must_block = [
        "100.100.100.200",        # Alibaba Cloud metadata (CGNAT)
        "100.64.0.1",             # CGNAT lower bound
        "100.127.255.254",        # CGNAT upper region
        "169.254.169.254",        # AWS/GCP/Azure metadata
        "0.1.2.3",                # "this network"
        "192.0.0.1",              # IETF protocol assignments
        "192.0.2.5",              # TEST-NET-1
        "192.88.99.1",            # deprecated 6to4 relay anycast
        "198.18.0.1",             # benchmarking
        "198.51.100.5",           # TEST-NET-2
        "203.0.113.5",            # TEST-NET-3
        "240.0.0.1",              # future use
        "255.255.255.255",        # limited broadcast
        "fec0::1",                # deprecated site-local
        "fec0::abcd",
        "fc00::1",                # unique local
        "2001:db8::1",            # documentation
        "2001::1",                # Teredo
        "100::1",                 # discard-only
        "::ffff:100.100.100.200",  # IPv4-mapped CGNAT
        "::ffff:169.254.169.254",  # IPv4-mapped metadata
        "64:ff9b::100.100.100.200",  # NAT64-embedded CGNAT
    ]
    for address in must_block:
        check(f"blocked {address}", web_tools._is_blocked_ip(ipaddress.ip_address(address)))

    must_allow = ["8.8.8.8", "1.1.1.1", "93.184.216.34", "2606:4700:4700::1111",
                  "2001:4860:4860::8888", "99.255.255.255", "101.0.0.1"]
    for address in must_allow:
        check(f"allowed {address}", not web_tools._is_blocked_ip(ipaddress.ip_address(address)))


def test_metadata_fetch_is_refused_end_to_end():
    """guarded_fetch must refuse the addresses above, not just the helper."""
    print("\n[test] metadata fetch refused end to end")

    async def _run():
        for url in (
            "http://100.100.100.200/latest/meta-data/",
            "http://100.64.0.1/",
            "http://[fec0::1]/x",
            "http://169.254.169.254/latest/meta-data/",
        ):
            refused = False
            try:
                await http_client.guarded_fetch(url)
            except http_client.BlockedUrlError:
                refused = True
            check(f"guarded_fetch refuses {url[:38]}", refused)

        # A hostname resolving into CGNAT is refused too (not just literals).
        with _PatchedDns({"meta.example.com": ["100.100.100.200"]}):
            refused = False
            try:
                await http_client.guarded_fetch("http://meta.example.com/latest/meta-data/")
            except http_client.BlockedUrlError:
                refused = True
            check("hostname resolving into CGNAT is refused", refused)

    asyncio.run(_run())


def test_pinned_client_does_not_reuse_connections_across_hosts():
    """SNI is read only at TLS setup, so pinned connections must not be pooled.

    Two hostnames sharing an IP would otherwise reuse one TLS connection and the
    second host's certificate would never be validated.
    """
    print("\n[test] pinned connections are not pooled across hosts")
    source = open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "keytao_bot", "utils", "http_client.py"),
        encoding="utf-8",
    ).read()
    guarded = source[source.index("async def get_guarded_client("):]
    guarded = guarded[:guarded.index("def _absolutize(")]
    check("guarded client disables keep-alive",
          "max_keepalive_connections=0" in guarded)
    check("the SNI reuse hazard is documented", "SNI" in guarded)


def test_ipv6_host_header_is_bracketed():
    """RFC 3986 requires brackets around an IPv6 literal in the Host header."""
    print("\n[test] IPv6 Host header bracketing")
    address_v6 = ipaddress.ip_address("2606:4700:4700::1111")

    _, host, sni = http_client._pin_request(
        "http://[2606:4700:4700::1111]:8080/x", address_v6)
    check("IPv6 host with port is bracketed", host == "[2606:4700:4700::1111]:8080")
    check("IPv6 SNI has no brackets", sni == "2606:4700:4700::1111")

    _, host_np, _ = http_client._pin_request("https://[2606:4700::1111]/y", address_v6)
    check("IPv6 host without port is bracketed", host_np == "[2606:4700::1111]")

    address_v4 = ipaddress.ip_address("93.184.216.34")
    _, host_named, _ = http_client._pin_request("https://example.com:8443/z", address_v4)
    check("named host with port is unchanged", host_named == "example.com:8443")
    _, host_plain, _ = http_client._pin_request("https://example.com/z", address_v4)
    check("named host without port is unchanged", host_plain == "example.com")



if __name__ == "__main__":
    print("=" * 60)
    print("Security hardening regression tests")
    print("=" * 60)

    test_ssrf_ip_blocklist()
    test_ssrf_scheme_validation()
    test_ssrf_host_resolution()
    test_web_fetch_rejects_dangerous_targets()
    test_redirect_guard_and_body_cap()
    test_connection_is_pinned_to_validated_ip()
    test_params_survive_redirects()
    test_jina_reader_prefix_applied_once()
    test_schema_validation_rejects_bad_arguments()
    test_executor_schema_validation_blocks_dispatch()
    test_batch_item_cap()
    test_dns_rebinding_peer_check()
    test_compression_bomb_is_bounded()
    test_outbound_clients_are_proxy_free_and_hop_validated()
    test_search_provider_redirect_is_hop_validated()
    test_review_page_fetch_is_guarded()
    test_external_client_validates_every_redirect_hop()
    test_special_purpose_ranges_are_blocked()
    test_metadata_fetch_is_refused_end_to_end()
    test_pinned_client_does_not_reuse_connections_across_hosts()
    test_ipv6_host_header_is_bracketed()
    test_non_object_arguments_do_not_crash_policy()
    test_retry_policy_is_idempotency_aware()
    test_fly_key_variants_follow_pronunciation_correction()
    # Last: it installs permanent module stubs in sys.modules.
    test_web_api_auth_surface()

    total = passed + failed
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed else 0)
