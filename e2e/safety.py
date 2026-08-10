"""Fail-closed network, database, and test-binding safety rails."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional
from urllib.parse import parse_qs, urlparse


LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
PRODUCTION_KEYTAO_HOSTS = frozenset({"keytao.vercel.app", "www.keytao.vercel.app"})
RESERVED_BINDING_PREFIX = "keytao-e2e-llm-rig-"
RESERVED_EMAIL_SUFFIX = "@example.invalid"
MIN_SYNTHETIC_QQ_DIGITS = 30
ADMIN_ROLE_VALUES = frozenset({"R:ROOT", "R:MANAGER"})


class SafetyViolation(RuntimeError):
    """Raised before a forbidden target can be contacted or mutated."""


def _normalized_host(value: Optional[str]) -> str:
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="strict")
    return str(value or "").strip().lower().rstrip(".")


def is_local_host(host: Optional[str]) -> bool:
    normalized = _normalized_host(host)
    if normalized in LOCAL_HOSTS:
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlparse(str(url or "").strip())
    host = _normalized_host(parsed.hostname)
    if not parsed.scheme or not host:
        raise SafetyViolation(f"URL has no absolute origin: {url!r}")
    if parsed.username or parsed.password:
        raise SafetyViolation("Credentials are forbidden in endpoint URLs")
    try:
        port = parsed.port
    except ValueError as error:
        raise SafetyViolation(f"Invalid endpoint port: {url!r}") from error
    if port is None:
        port = 443 if parsed.scheme.lower() == "https" else 80
    return parsed.scheme.lower(), host, port


def validate_keytao_base(url: str) -> str:
    scheme, host, _port = _origin(url)
    if scheme != "http" or not is_local_host(host):
        raise SafetyViolation(
            "KEYTAO_API_BASE must use plain HTTP on localhost; "
            f"refusing {scheme}://{host}"
        )
    parsed = urlparse(url)
    if parsed.query or parsed.fragment or (parsed.path not in {"", "/"}):
        raise SafetyViolation("KEYTAO_API_BASE must be a bare localhost origin")
    return str(url).rstrip("/")


def validate_llm_base(url: str) -> str:
    scheme, host, _port = _origin(url)
    if scheme not in {"http", "https"}:
        raise SafetyViolation("The LLM endpoint must use HTTP or HTTPS")
    if host in PRODUCTION_KEYTAO_HOSTS or host.endswith(".keytao.vercel.app"):
        raise SafetyViolation("The LLM endpoint cannot be a KeyTao production host")
    parsed = urlparse(url)
    if parsed.fragment:
        raise SafetyViolation("The LLM endpoint cannot contain a fragment")
    return str(url).rstrip("/") + "/"


def validate_next_database_url(url: str) -> dict[str, Any]:
    parsed = urlparse(str(url or "").strip())
    scheme = parsed.scheme.lower()
    if scheme not in {"postgres", "postgresql"}:
        raise SafetyViolation(
            "keytao-next DATABASE_URL must be a local PostgreSQL URL"
        )
    host = _normalized_host(parsed.hostname)
    if not is_local_host(host):
        raise SafetyViolation(
            f"keytao-next DATABASE_URL is non-local: host={host or '<missing>'}"
        )
    query_hosts = [
        item
        for item in parse_qs(parsed.query).get("host", [])
        if str(item).strip()
    ]
    if any(not is_local_host(item) for item in query_hosts):
        raise SafetyViolation("DATABASE_URL query overrides host to a non-local target")
    try:
        port = parsed.port or 5432
    except ValueError as error:
        raise SafetyViolation("DATABASE_URL contains an invalid port") from error
    return {"scheme": scheme, "host": host, "port": port}


def validate_reserved_identity(
    *,
    platform_id: str,
    expected_name: str,
    expected_email: str,
) -> None:
    binding = str(platform_id or "")
    if (
        not binding.isdecimal()
        or len(binding) < MIN_SYNTHETIC_QQ_DIGITS
        or not expected_name.startswith(RESERVED_BINDING_PREFIX)
        or not expected_email.endswith(RESERVED_EMAIL_SUFFIX)
    ):
        raise SafetyViolation("The requested QQ binding is not structurally synthetic")


def _validate_identity_metadata(
    *,
    expected_name: str,
    expected_email: str,
    user: dict[str, Any],
) -> set[str]:
    actual_name = str(user.get("name") or "")
    actual_email = str(user.get("email") or "")
    if actual_name != expected_name or actual_email != expected_email:
        raise SafetyViolation(
            "The reserved test binding maps to unexpected user metadata; "
            "refusing to treat it as an E2E account"
        )
    role_values = {
        str(role.get("value") or "")
        for role in user.get("roles", [])
        if isinstance(role, dict)
    }
    return role_values


def validate_test_binding(
    *,
    platform_id: str,
    expected_name: str,
    expected_email: str,
    user: dict[str, Any],
) -> None:
    validate_reserved_identity(
        platform_id=platform_id,
        expected_name=expected_name,
        expected_email=expected_email,
    )
    role_values = _validate_identity_metadata(
        expected_name=expected_name,
        expected_email=expected_email,
        user=user,
    )
    if not {"R:NORMAL", "R:BOT"}.issubset(role_values):
        raise SafetyViolation("The E2E account is missing its dedicated bot roles")


def validate_admin_identity(
    *,
    platform_id: str,
    expected_name: str,
    expected_email: str,
    user: dict[str, Any],
) -> None:
    """Require a reserved local rig identity with a real admin database role."""

    validate_reserved_identity(
        platform_id=platform_id,
        expected_name=expected_name,
        expected_email=expected_email,
    )
    role_values = _validate_identity_metadata(
        expected_name=expected_name,
        expected_email=expected_email,
        user=user,
    )
    if not {"R:NORMAL", "R:BOT"}.issubset(role_values):
        raise SafetyViolation("The E2E admin account is missing its dedicated bot roles")
    if not role_values.intersection(ADMIN_ROLE_VALUES):
        raise SafetyViolation("The E2E admin account has no administrator role")


@dataclass
class EncodeDelayController:
    """Inject one pre-dispatch timeout into an encode GET when explicitly armed."""

    delay_seconds: float
    attempt_timeout_seconds: float
    armed_scenario: str = ""
    injected: bool = False

    def arm(self, scenario_id: str) -> None:
        self.armed_scenario = scenario_id
        self.injected = False

    def disarm(self) -> None:
        self.armed_scenario = ""

    def should_inject(self, *, scenario_id: str, method: str, path: str) -> bool:
        if (
            not self.armed_scenario
            or self.injected
            or scenario_id != self.armed_scenario
            or method.upper() != "GET"
        ):
            return False
        if path not in {"/api/phrases/encode", "/api/bot/phrases/encode"}:
            return False
        self.injected = True
        return True


@dataclass
class PronunciationPoisonController:
    """Inject one wrong-entry search hit and page for the armed scenario."""

    armed_scenario: str = ""
    search_injected: bool = False
    page_injected: bool = False

    def arm(self, scenario_id: str) -> None:
        self.armed_scenario = scenario_id
        self.search_injected = False
        self.page_injected = False

    def disarm(self) -> None:
        self.armed_scenario = ""

    @property
    def injected(self) -> bool:
        return self.search_injected and self.page_injected

    def response_for(
        self,
        *,
        scenario_id: str,
        method: str,
        url: str,
    ) -> Optional[tuple[str, str]]:
        if (
            not self.armed_scenario
            or scenario_id != self.armed_scenario
            or method.upper() != "GET"
        ):
            return None

        parsed = urlparse(url)
        host = _normalized_host(parsed.hostname)
        if (
            host == "www.bing.com"
            and parsed.path == "/search"
            and not self.search_injected
        ):
            query = " ".join(parse_qs(parsed.query).get("q", []))
            if "site:zdic.net" in query and "亮面" in query:
                self.search_injected = True
                return (
                    "wrong-entry-search-hit",
                    '<h2><a href="https://www.zdic.net/hans/%E5%85%89%E9%9D%A2">'
                    "光面_汉典</a></h2><p>光面 拼音：guāng miàn</p>",
                )

        if (
            host in {"zdic.net", "www.zdic.net"}
            and parsed.path == "/hans/%E5%85%89%E9%9D%A2"
            and not self.page_injected
        ):
            self.page_injected = True
            return (
                "wrong-entry-page",
                "<html><head><title>光面_汉典</title></head>"
                "<body><h1>光面</h1><div>拼音：guāng miàn</div>"
                "<p>光滑的表面。</p></body></html>",
            )
        return None


class NetworkAllowlist:
    """Process-wide guard for HTTP hops and raw Python socket connections."""

    def __init__(
        self,
        *,
        llm_base_url: str,
        recorder: Any = None,
        scenario_getter: Optional[Callable[[], str]] = None,
        encode_delay: Optional[EncodeDelayController] = None,
        pronunciation_poison: Optional[PronunciationPoisonController] = None,
    ) -> None:
        self.llm_base_url = validate_llm_base(llm_base_url)
        self.llm_origin = _origin(self.llm_base_url)
        self.recorder = recorder
        self.scenario_getter = scenario_getter or (lambda: "")
        self.encode_delay = encode_delay
        self.pronunciation_poison = pronunciation_poison
        self._llm_ips = set(self._resolve_llm_ips(self.llm_origin[1], self.llm_origin[2]))
        self._patches: list[tuple[Any, str, Any]] = []

    @staticmethod
    def _resolve_llm_ips(host: str, port: int) -> frozenset[str]:
        try:
            values = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except OSError:
            # Some managed shells expose DNS only when the first real endpoint
            # request is made. Keep the hostname allowlist exact and learn the
            # returned IPs from the guarded getaddrinfo call below.
            return frozenset()
        addresses = {
            str(item[4][0]).split("%", 1)[0]
            for item in values
            if item and len(item) > 4 and item[4]
        }
        return frozenset(addresses)

    def assert_url_allowed(self, url: Any) -> None:
        scheme, host, port = _origin(str(url))
        if host in PRODUCTION_KEYTAO_HOSTS or host.endswith(".keytao.vercel.app"):
            raise SafetyViolation(f"Blocked KeyTao production URL before dispatch: {host}")
        if scheme not in {"http", "https"}:
            raise SafetyViolation(f"Blocked non-HTTP network URL: {scheme}://{host}")
        if is_local_host(host):
            return
        if (scheme, host, port) == self.llm_origin:
            return
        raise SafetyViolation(
            f"Network target is outside the E2E allowlist: {scheme}://{host}:{port}"
        )

    def assert_socket_allowed(self, host: Any, port: Any) -> None:
        normalized = _normalized_host(host).split("%", 1)[0]
        if is_local_host(normalized):
            return
        if normalized == self.llm_origin[1] and int(port) == self.llm_origin[2]:
            return
        try:
            address = str(ipaddress.ip_address(normalized))
        except ValueError:
            address = ""
        if address in self._llm_ips and int(port) == self.llm_origin[2]:
            return
        raise SafetyViolation(
            f"Raw socket target is outside the E2E allowlist: {normalized}:{port}"
        )

    def _set_patch(self, target: Any, name: str, replacement: Any) -> None:
        self._patches.append((target, name, getattr(target, name)))
        setattr(target, name, replacement)

    def install(self) -> None:
        if self._patches:
            return
        import httpx

        guard = self
        original_async_send = httpx.AsyncClient.send
        original_async_single = httpx.AsyncClient._send_single_request
        original_sync_send = httpx.Client.send
        original_sync_single = httpx.Client._send_single_request
        original_connect = socket.socket.connect
        original_connect_ex = socket.socket.connect_ex
        original_create_connection = socket.create_connection
        original_getaddrinfo = socket.getaddrinfo

        async def async_single(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            guard.assert_url_allowed(request.url)
            return await original_async_single(client, request, *args, **kwargs)

        async def async_send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            scenario_id = guard.scenario_getter()
            synthetic = (
                guard.pronunciation_poison.response_for(
                    scenario_id=scenario_id,
                    method=request.method,
                    url=str(request.url),
                )
                if guard.pronunciation_poison is not None
                else None
            )
            if synthetic is not None:
                injection_kind, body = synthetic
                if guard.recorder is not None:
                    guard.recorder.record_fault_injection(
                        scenario_id=scenario_id,
                        method=request.method,
                        url=str(request.url),
                        injection_kind=injection_kind,
                    )
                return httpx.Response(200, text=body, request=request)
            guard.assert_url_allowed(request.url)
            if (
                guard.encode_delay is not None
                and guard.encode_delay.should_inject(
                    scenario_id=scenario_id,
                    method=request.method,
                    path=request.url.path,
                )
            ):
                await asyncio.sleep(guard.encode_delay.delay_seconds)
                if guard.recorder is not None:
                    guard.recorder.record_fault_injection(
                        scenario_id=scenario_id,
                        method=request.method,
                        url=str(request.url),
                        delay_seconds=guard.encode_delay.delay_seconds,
                        attempt_timeout_seconds=guard.encode_delay.attempt_timeout_seconds,
                    )
                raise httpx.ReadTimeout(
                    "E2E artificial encode delay exceeded the per-attempt timeout",
                    request=request,
                )
            started = asyncio.get_running_loop().time()
            try:
                response = await original_async_send(client, request, *args, **kwargs)
            except BaseException as error:
                if guard.recorder is not None:
                    guard.recorder.record_http_error(
                        request=request,
                        error=error,
                        elapsed_seconds=asyncio.get_running_loop().time() - started,
                    )
                raise
            if guard.recorder is not None:
                guard.recorder.record_http_response(
                    request=request,
                    response=response,
                    elapsed_seconds=asyncio.get_running_loop().time() - started,
                    llm_origin=guard.llm_origin,
                )
            return response

        def sync_single(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            guard.assert_url_allowed(request.url)
            return original_sync_single(client, request, *args, **kwargs)

        def sync_send(client: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
            guard.assert_url_allowed(request.url)
            return original_sync_send(client, request, *args, **kwargs)

        def connect(sock: socket.socket, address: Any) -> Any:
            guard.assert_socket_allowed(address[0], address[1])
            return original_connect(sock, address)

        def connect_ex(sock: socket.socket, address: Any) -> Any:
            guard.assert_socket_allowed(address[0], address[1])
            return original_connect_ex(sock, address)

        def create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
            guard.assert_socket_allowed(address[0], address[1])
            return original_create_connection(address, *args, **kwargs)

        def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
            guard.assert_socket_allowed(host, port)
            values = original_getaddrinfo(host, port, *args, **kwargs)
            if (
                _normalized_host(host).split("%", 1)[0] == guard.llm_origin[1]
                and int(port) == guard.llm_origin[2]
            ):
                guard._llm_ips.update(
                    str(item[4][0]).split("%", 1)[0]
                    for item in values
                    if item and len(item) > 4 and item[4]
                )
            return values

        self._set_patch(httpx.AsyncClient, "_send_single_request", async_single)
        self._set_patch(httpx.AsyncClient, "send", async_send)
        self._set_patch(httpx.Client, "_send_single_request", sync_single)
        self._set_patch(httpx.Client, "send", sync_send)
        self._set_patch(socket.socket, "connect", connect)
        self._set_patch(socket.socket, "connect_ex", connect_ex)
        self._set_patch(socket, "create_connection", create_connection)
        self._set_patch(socket, "getaddrinfo", getaddrinfo)

    def restore(self) -> None:
        while self._patches:
            target, name, original = self._patches.pop()
            setattr(target, name, original)

    def __enter__(self) -> "NetworkAllowlist":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.restore()


def assert_all_urls_allowed(guard: NetworkAllowlist, urls: Iterable[str]) -> None:
    for url in urls:
        guard.assert_url_allowed(url)
