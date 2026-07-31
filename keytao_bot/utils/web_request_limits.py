"""ASGI request-body limits for the small public web API surface."""

import json
from collections.abc import Awaitable, Callable
from typing import Any


ASGIReceive = Callable[..., Awaitable[dict[str, Any]]]
ASGISend = Callable[..., Awaitable[None]]
ASGIApp = Callable[[dict[str, Any], ASGIReceive, ASGISend], Awaitable[None]]

REQUEST_BODY_LIMITS: dict[tuple[str, str], int] = {
    ("POST", "/api/chat"): 32 * 1024,
    ("DELETE", "/api/chat/history"): 4 * 1024,
    ("POST", "/api/keytao/batches/review"): 512 * 1024,
}


async def _send_error(send: ASGISend, status: int, detail: str) -> None:
    body = json.dumps(
        {"detail": detail},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _parse_content_length(scope: dict[str, Any]) -> tuple[int | None, bool]:
    values = [
        value
        for name, value in scope.get("headers", [])
        if name.lower() == b"content-length"
    ]
    if not values:
        return None, True
    if len(values) != 1:
        return None, False
    try:
        text = values[0].decode("ascii")
    except UnicodeDecodeError:
        return None, False
    if not text or not text.isdecimal():
        return None, False
    try:
        return int(text), True
    except ValueError:
        return None, False


class RequestBodyLimitMiddleware:
    """Buffer and bound selected request bodies before route parsing begins."""

    def __init__(
        self,
        app: ASGIApp,
        limits: dict[tuple[str, str], int] | None = None,
    ) -> None:
        self.app = app
        self.limits = REQUEST_BODY_LIMITS if limits is None else limits

    async def __call__(self, scope, receive, send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        key = (str(scope.get("method", "")).upper(), str(scope.get("path", "")))
        limit = self.limits.get(key)
        if limit is None:
            await self.app(scope, receive, send)
            return

        declared_length, valid_length = _parse_content_length(scope)
        if not valid_length:
            await _send_error(send, 400, "Invalid Content-Length")
            return
        if declared_length is not None and declared_length > limit:
            await _send_error(send, 413, "Request body too large")
            return

        events: list[dict[str, Any]] = []
        received_length = 0
        while True:
            event = await receive()
            event_type = event.get("type")
            if event_type == "http.disconnect":
                return
            if event_type != "http.request":
                await _send_error(send, 400, "Invalid request body")
                return
            body = event.get("body", b"")
            if not isinstance(body, bytes):
                await _send_error(send, 400, "Invalid request body")
                return
            received_length += len(body)
            if received_length > limit:
                await _send_error(send, 413, "Request body too large")
                return
            events.append(event)
            if not event.get("more_body", False):
                break

        if declared_length is not None and received_length != declared_length:
            await _send_error(send, 400, "Content-Length does not match request body")
            return

        event_index = 0

        async def replay_receive() -> dict[str, Any]:
            nonlocal event_index
            if event_index >= len(events):
                return {"type": "http.disconnect"}
            event = events[event_index]
            event_index += 1
            return event

        await self.app(scope, replay_receive, send)
