"""Artifact recording for messages, model exchanges, tools, HTTP, and logs."""

from __future__ import annotations

import contextlib
import contextvars
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse


SENSITIVE_FIELDS = frozenset(
    {
        "access_token",
        "api_key",
        "authorization",
        "password",
        "refresh_token",
        "token",
    }
)


def _redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if str(key).lower() in SENSITIVE_FIELDS else _redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _json_value(data: bytes) -> Any:
    if not data:
        return None
    text = data.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _request_body(request: Any) -> Any:
    try:
        return _redact_sensitive(_json_value(bytes(request.content)))
    except Exception:
        return {"unavailable": "request body is streaming"}


def _response_body(response: Any) -> Any:
    try:
        return _redact_sensitive(_json_value(bytes(response.content)))
    except Exception:
        return {"unavailable": "response body is streaming"}


class ArtifactRecorder:
    """Append-only in-memory recorder flushed to per-attempt JSON artifacts."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=False)
        self._scenario = contextvars.ContextVar("e2e_scenario", default="setup")
        self._attempt = contextvars.ContextVar("e2e_attempt", default=0)
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []
        self._log_sink_id: Optional[int] = None

    def current_scenario(self) -> str:
        return self._scenario.get()

    def current_attempt(self) -> int:
        return self._attempt.get()

    @contextlib.contextmanager
    def scope(self, scenario_id: str, attempt: int) -> Iterator[None]:
        scenario_token = self._scenario.set(scenario_id)
        attempt_token = self._attempt.set(attempt)
        try:
            yield
        finally:
            self._attempt.reset(attempt_token)
            self._scenario.reset(scenario_token)

    def _append(self, kind: str, **data: Any) -> None:
        item = {
            "sequence": 0,
            "timestamp": time.time(),
            "scenarioId": self.current_scenario(),
            "attempt": self.current_attempt(),
            "kind": kind,
            **data,
        }
        with self._lock:
            item["sequence"] = len(self.events) + 1
            self.events.append(item)

    def install_log_sink(self) -> None:
        if self._log_sink_id is not None:
            return
        from nonebot.log import logger

        def sink(message: Any) -> None:
            record = message.record
            self._append(
                "log",
                level=str(record["level"].name),
                logger=str(record.get("name") or ""),
                message=str(record["message"]),
            )

        self._log_sink_id = logger.add(sink, level="DEBUG", enqueue=False)

    def remove_log_sink(self) -> None:
        if self._log_sink_id is None:
            return
        from nonebot.log import logger

        logger.remove(self._log_sink_id)
        self._log_sink_id = None

    def record_message(
        self,
        *,
        direction: str,
        text: str,
        platform_id: str,
        message_id: int | None = None,
        reply_message_id: int | None = None,
    ) -> None:
        payload = {
            "direction": direction,
            "text": text,
            "platform": "qq",
            "platformId": platform_id,
        }
        if message_id is not None:
            payload["messageId"] = int(message_id)
        if reply_message_id is not None:
            payload["replyMessageId"] = int(reply_message_id)
        self._append("message", **payload)

    def record_tool(
        self,
        *,
        phase: str,
        name: str,
        arguments: dict[str, Any],
        result: Any,
        elapsed_seconds: float,
    ) -> None:
        try:
            parsed_result = json.loads(result) if isinstance(result, str) else result
        except (TypeError, ValueError):
            parsed_result = result
        self._append(
            "tool",
            phase=phase,
            name=name,
            arguments=arguments,
            result=parsed_result,
            elapsedSeconds=elapsed_seconds,
        )

    def record_fault_injection(self, **data: Any) -> None:
        self._append("faultInjection", **data)

    def record_http_error(
        self,
        *,
        request: Any,
        error: BaseException,
        elapsed_seconds: float,
    ) -> None:
        self._append(
            "http",
            method=request.method,
            url=str(request.url),
            requestBody=_request_body(request),
            status=None,
            errorType=type(error).__name__,
            error=str(error),
            elapsedSeconds=elapsed_seconds,
        )

    def record_http_response(
        self,
        *,
        request: Any,
        response: Any,
        elapsed_seconds: float,
        llm_origin: tuple[str, str, int],
    ) -> None:
        request_data = _request_body(request)
        response_data = _response_body(response)
        parsed = urlparse(str(request.url))
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        is_llm = (parsed.scheme, str(parsed.hostname or "").lower(), port) == llm_origin
        self._append(
            "http",
            method=request.method,
            url=str(request.url),
            requestBody=request_data,
            status=response.status_code,
            responseBody=response_data,
            elapsedSeconds=elapsed_seconds,
            isLlm=is_llm,
        )
        if is_llm:
            self._append(
                "modelExchange",
                url=str(request.url),
                request=request_data,
                response=response_data,
                status=response.status_code,
                elapsedSeconds=elapsed_seconds,
            )

    def events_for(self, scenario_id: str, attempt: int) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(item)
                for item in self.events
                if item["scenarioId"] == scenario_id and item["attempt"] == attempt
            ]

    def write_attempt(
        self,
        *,
        scenario_id: str,
        attempt: int,
        result: dict[str, Any],
    ) -> Path:
        payload = {
            "scenarioId": scenario_id,
            "attempt": attempt,
            "result": result,
            "events": self.events_for(scenario_id, attempt),
        }
        path = self.artifact_dir / f"{scenario_id}-attempt-{attempt}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.artifact_dir / name
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def cost_summary(self, scenario_id: str, attempt: int) -> dict[str, Any]:
        exchanges = [
            event
            for event in self.events_for(scenario_id, attempt)
            if event["kind"] == "modelExchange"
        ]
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        models: set[str] = set()
        for exchange in exchanges:
            request = exchange.get("request")
            response = exchange.get("response")
            if isinstance(request, dict) and request.get("model"):
                models.add(str(request["model"]))
            usage = response.get("usage") if isinstance(response, dict) else None
            if isinstance(usage, dict):
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                completion_tokens += int(usage.get("completion_tokens") or 0)
                total_tokens += int(usage.get("total_tokens") or 0)
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens
        return {
            "modelRequests": len(exchanges),
            "models": sorted(models),
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": total_tokens,
            "monetaryCost": None,
            "costNote": "Provider billing price is not available locally; token usage is recorded.",
        }
