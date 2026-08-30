"""Bounded, content-free turn and process-state observability."""

from __future__ import annotations

import functools
import json
import re
import sqlite3
import time
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple, TypeVar

from .llm_policy import chat_usage_metrics


TURN_FLOWS = frozenset({
    "word-discovery",
    "pending-confirmation",
    "multi-add",
    "draft-op",
    "general",
})
TURN_OUTCOMES = frozenset({
    "replied",
    "policy-blocked",
    "asked-confirmation",
    "error",
})
_FLOW_PRIORITY = {
    "general": 0,
    "word-discovery": 1,
    "draft-op": 2,
    "multi-add": 3,
    "pending-confirmation": 4,
}
_OUTCOME_PRIORITY = {
    "replied": 0,
    "asked-confirmation": 1,
    "policy-blocked": 2,
    "error": 3,
}


@dataclass
class TurnMetrics:
    turn_id: str
    platform: str
    space_kind: str
    started_at: float
    flow: str = "general"
    model_calls: int = 0
    model_seconds: float = 0.0
    tool_calls: int = 0
    tool_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    system_prompt_chars: int = 0
    history_messages: int = 0
    model_tool_result_chars: int = 0
    encode_calls: List[Tuple[float, bool]] = field(default_factory=list)
    outcome: str = "replied"
    emitted: bool = False


_CURRENT_TURN: ContextVar[Optional[TurnMetrics]] = ContextVar(
    "current_turn_metrics",
    default=None,
)


def begin_turn_metrics(
    platform: str,
    space_kind: str,
    *,
    started_at: Optional[float] = None,
) -> Token:
    """Start metrics collection at the inbound-message boundary."""
    metrics = TurnMetrics(
        turn_id=uuid.uuid4().hex[:8],
        platform=_bounded_label(platform, fallback="unknown"),
        space_kind=_bounded_label(space_kind, fallback="unknown"),
        started_at=time.monotonic() if started_at is None else float(started_at),
    )
    return _CURRENT_TURN.set(metrics)


def end_turn_metrics(token: Token) -> None:
    _CURRENT_TURN.reset(token)


def suspend_turn_metrics() -> Token:
    """Detach unrelated background maintenance from the originating turn."""
    return _CURRENT_TURN.set(None)


def current_turn_metrics() -> Optional[TurnMetrics]:
    return _CURRENT_TURN.get()


def current_turn_id() -> str:
    metrics = current_turn_metrics()
    return metrics.turn_id if metrics is not None else "-"


def turn_metrics_emitted() -> bool:
    metrics = current_turn_metrics()
    return bool(metrics is not None and metrics.emitted)


def set_turn_flow(flow: str) -> None:
    metrics = current_turn_metrics()
    if metrics is None or flow not in TURN_FLOWS:
        return
    if _FLOW_PRIORITY[flow] >= _FLOW_PRIORITY[metrics.flow]:
        metrics.flow = flow


def mark_turn_outcome(outcome: str) -> None:
    metrics = current_turn_metrics()
    if metrics is None or outcome not in TURN_OUTCOMES:
        return
    if _OUTCOME_PRIORITY[outcome] >= _OUTCOME_PRIORITY[metrics.outcome]:
        metrics.outcome = outcome


def record_system_prompt_chars(chars: int) -> None:
    metrics = current_turn_metrics()
    if metrics is None:
        return
    metrics.system_prompt_chars = max(metrics.system_prompt_chars, max(0, int(chars)))


def record_history_messages(count: int) -> None:
    metrics = current_turn_metrics()
    if metrics is None:
        return
    metrics.history_messages = max(metrics.history_messages, max(0, int(count)))


def record_model_tool_result_chars(chars: int) -> None:
    """Add the exact tool-result characters serialized into model messages."""
    metrics = current_turn_metrics()
    if metrics is None:
        return
    metrics.model_tool_result_chars += max(0, int(chars))


def record_encode_call(seconds: float, *, retry: bool) -> None:
    """Retain bounded, content-free latency for each encode HTTP attempt."""
    metrics = current_turn_metrics()
    if metrics is None or len(metrics.encode_calls) >= 32:
        return
    metrics.encode_calls.append((max(0.0, float(seconds)), bool(retry)))


T = TypeVar("T")


async def observe_model_call(
    awaitable: Awaitable[T],
    *,
    system_prompt_chars: int = 0,
) -> T:
    """Measure one model request and sum its normalized token usage."""
    metrics = current_turn_metrics()
    if metrics is None:
        return await awaitable
    record_system_prompt_chars(system_prompt_chars)
    started_at = time.monotonic()
    response: Any = None
    try:
        response = await awaitable
        return response
    finally:
        metrics.model_calls += 1
        metrics.model_seconds += time.monotonic() - started_at
        if response is not None:
            usage = chat_usage_metrics(response)
            metrics.input_tokens += usage.get("input_tokens", 0)
            metrics.output_tokens += usage.get("output_tokens", 0)
            metrics.cached_tokens += usage.get("cached_tokens", 0)
            metrics.reasoning_tokens += usage.get("reasoning_tokens", 0)


F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def observe_tool_call(function: F) -> F:
    """Decorate the central tool boundary without changing its result."""
    @functools.wraps(function)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        metrics = current_turn_metrics()
        if metrics is None:
            return await function(*args, **kwargs)
        started_at = time.monotonic()
        try:
            result = await function(*args, **kwargs)
        finally:
            metrics.tool_calls += 1
            metrics.tool_seconds += time.monotonic() - started_at
        observe_tool_result(result)
        return result

    return wrapped  # type: ignore[return-value]


def observe_tool_result(result: Any) -> None:
    """Project existing structured tool outcomes onto the turn outcome."""
    payload = result
    if isinstance(result, str):
        try:
            payload = json.loads(result)
        except (TypeError, ValueError):
            return
    if not isinstance(payload, Mapping):
        return
    if payload.get("policyBlocked") is True:
        mark_turn_outcome("policy-blocked")
    elif payload.get("requiresConfirmation") is True:
        mark_turn_outcome("asked-confirmation")


def emit_turn_metrics(logger: Any, *, ended_at: Optional[float] = None) -> Optional[str]:
    """Emit the one bounded metrics line for the current turn."""
    metrics = current_turn_metrics()
    if metrics is None or metrics.emitted:
        return None
    metrics.emitted = True
    finished_at = time.monotonic() if ended_at is None else float(ended_at)
    encode_call_seconds = (
        ",".join(f"{seconds:.3f}" for seconds, _retry in metrics.encode_calls)
        or "-"
    )
    encode_retry_flags = (
        ",".join("1" if retry else "0" for _seconds, retry in metrics.encode_calls)
        or "-"
    )
    line = (
        "[turn_metrics] "
        f"turn_id={metrics.turn_id} platform={metrics.platform} "
        f"space_kind={metrics.space_kind} flow={metrics.flow} "
        f"e2e_seconds={max(0.0, finished_at - metrics.started_at):.3f} "
        f"model_calls={metrics.model_calls} model_seconds={metrics.model_seconds:.3f} "
        f"tool_calls={metrics.tool_calls} tool_seconds={metrics.tool_seconds:.3f} "
        f"input_tokens={metrics.input_tokens} output_tokens={metrics.output_tokens} "
        f"cached_tokens={metrics.cached_tokens} reasoning_tokens={metrics.reasoning_tokens} "
        f"system_prompt_chars={metrics.system_prompt_chars} "
        f"history_messages={metrics.history_messages} "
        f"model_tool_result_chars={metrics.model_tool_result_chars} "
        f"encode_calls={len(metrics.encode_calls)} "
        f"encode_retry_calls={sum(1 for _seconds, retry in metrics.encode_calls if retry)} "
        f"encode_call_seconds={encode_call_seconds} "
        f"encode_retry_flags={encode_retry_flags} "
        f"outcome={metrics.outcome}"
    )
    logger.info(line)
    return line


def log_state_metrics(
    logger: Any,
    data_dir: Path,
    *,
    cache_entries: Mapping[str, int],
    pending_live: int,
    system_prompt_chars: int,
) -> str:
    """Read small SQLite metadata and emit one content-free state line."""
    database_bytes: Dict[str, int] = {}
    database_rows: Dict[str, int] = {}
    for db_path in sorted(data_dir.glob("*.db"), key=lambda path: path.name):
        db_name = _bounded_label(db_path.name, fallback="unknown.db")
        try:
            database_bytes[db_name] = max(0, db_path.stat().st_size)
        except OSError:
            database_bytes[db_name] = -1
            continue
        try:
            uri = f"file:{db_path.resolve()}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=1.0) as connection:
                tables = connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                    "ORDER BY name"
                ).fetchall()
                for (raw_table_name,) in tables:
                    table_name = str(raw_table_name)
                    quoted_table = table_name.replace('"', '""')
                    count = connection.execute(
                        f'SELECT COUNT(*) FROM "{quoted_table}"'
                    ).fetchone()[0]
                    row_key = f"{db_name}.{_bounded_label(table_name, fallback='unknown')}"
                    database_rows[row_key] = max(0, int(count))
        except (OSError, sqlite3.Error, TypeError, ValueError):
            database_rows[f"{db_name}.unavailable"] = -1

    byte_summary = _metric_pairs(database_bytes)
    row_summary = _metric_pairs(database_rows)
    cache_summary = _metric_pairs({
        _bounded_label(name, fallback="unknown"): max(0, int(value))
        for name, value in cache_entries.items()
    })
    line = (
        "[state_metrics] "
        f"db_bytes={byte_summary} db_rows={row_summary} "
        f"cache_entries={cache_summary} pending_live={max(0, int(pending_live))} "
        f"system_prompt_chars={max(0, int(system_prompt_chars))}"
    )
    logger.info(line)
    return line


def _metric_pairs(values: Mapping[str, int]) -> str:
    if not values:
        return "none"
    return ",".join(f"{name}:{value}" for name, value in sorted(values.items()))


def _bounded_label(value: Any, *, fallback: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())[:80]
    return normalized or fallback
