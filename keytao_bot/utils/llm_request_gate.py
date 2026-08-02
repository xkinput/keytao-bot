"""In-process cost and concurrency gate for authenticated LLM endpoints."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str = ""
    retry_after_seconds: int = 1


class RequestWindowGate:
    """Atomically bound global and requester traffic at the billed boundary."""

    def __init__(
        self,
        *,
        global_limit: int,
        requester_limit: int,
        window_seconds: float,
        max_concurrent: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min(global_limit, requester_limit, max_concurrent) <= 0:
            raise ValueError("request limits must be positive")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._global_limit = global_limit
        self._requester_limit = requester_limit
        self._window_seconds = window_seconds
        self._max_concurrent = max_concurrent
        self._clock = clock
        self._lock = threading.Lock()
        self._global_requests: Deque[float] = deque()
        self._requester_requests: Dict[str, Deque[float]] = {}
        self._active = 0

    def _prune(self, requests: Deque[float], now: float) -> None:
        cutoff = now - self._window_seconds
        while requests and requests[0] <= cutoff:
            requests.popleft()

    def _retry_after(self, requests: Deque[float], now: float) -> int:
        if not requests:
            return 1
        return max(1, math.ceil(requests[0] + self._window_seconds - now))

    def try_acquire(self, requester: str) -> GateDecision:
        requester_key = requester or "anonymous"
        with self._lock:
            now = self._clock()
            self._prune(self._global_requests, now)

            if len(self._requester_requests) > self._global_limit * 2:
                for key, requests in list(self._requester_requests.items()):
                    self._prune(requests, now)
                    if not requests:
                        del self._requester_requests[key]

            requester_requests = self._requester_requests.setdefault(
                requester_key,
                deque(),
            )
            self._prune(requester_requests, now)

            if self._active >= self._max_concurrent:
                return GateDecision(False, "concurrency", 1)
            if len(self._global_requests) >= self._global_limit:
                return GateDecision(
                    False,
                    "global-window",
                    self._retry_after(self._global_requests, now),
                )
            if len(requester_requests) >= self._requester_limit:
                return GateDecision(
                    False,
                    "requester-window",
                    self._retry_after(requester_requests, now),
                )

            self._global_requests.append(now)
            requester_requests.append(now)
            self._active += 1
            return GateDecision(True)

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)
