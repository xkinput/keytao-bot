"""Verify signed web-user identities forwarded by keytao-next."""

from __future__ import annotations

import hashlib
import hmac
import re
import threading
import time
from collections import OrderedDict
from typing import Optional


class WebIdentityConfigError(RuntimeError):
    """The shared identity-signing secret is unavailable."""


class WebIdentityVerificationError(ValueError):
    """The forwarded identity is missing, stale, or invalid."""


class WebIdentityReplayCache:
    """Bounded process-local nonce cache for single-use signed requests."""

    def __init__(self, *, max_entries: int = 10000) -> None:
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def consume(self, nonce: str, *, now: float, ttl_seconds: int) -> bool:
        with self._lock:
            cutoff = now - max(1, ttl_seconds)
            while self._entries:
                _, seen_at = next(iter(self._entries.items()))
                if seen_at >= cutoff:
                    break
                self._entries.popitem(last=False)
            if nonce in self._entries:
                return False
            self._entries[nonce] = now
            self._entries.move_to_end(nonce)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
            return True


_WEB_IDENTITY_REPLAY_CACHE = WebIdentityReplayCache()


def build_web_identity_signature(
    secret: str,
    *,
    method: str,
    path: str,
    user_id: str,
    timestamp: str,
    nonce: str,
    raw_body: bytes,
) -> str:
    """Return the shared canonical HMAC used by keytao-next and the bot."""
    body_digest = hashlib.sha256(raw_body).hexdigest()
    message = "\n".join((
        method.upper(),
        path,
        user_id,
        timestamp,
        nonce,
        body_digest,
    )).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()


def verify_web_user_identity(
    secret: str,
    *,
    body_user_id: Optional[str],
    header_user_id: Optional[str],
    timestamp: Optional[str],
    nonce: Optional[str],
    signature: Optional[str],
    method: str,
    path: str,
    raw_body: bytes,
    now: Optional[float] = None,
    max_clock_skew_seconds: int = 300,
    replay_cache: Optional[WebIdentityReplayCache] = None,
) -> Optional[str]:
    """Return the verified user ID, or ``None`` for an anonymous request."""
    supplied_headers = any((header_user_id, timestamp, nonce, signature))
    normalized_body_user_id = str(body_user_id or "").strip()

    if not normalized_body_user_id:
        if supplied_headers:
            raise WebIdentityVerificationError(
                "Signed user headers require a body user_id"
            )
        return None

    if not secret:
        raise WebIdentityConfigError(
            "BOT_IDENTITY_SECRET is not configured for web identity verification"
        )
    if not header_user_id or not timestamp or not nonce or not signature:
        raise WebIdentityVerificationError("Signed web user identity is required")

    normalized_header_user_id = str(header_user_id).strip()
    if normalized_header_user_id != normalized_body_user_id:
        raise WebIdentityVerificationError("Signed user ID does not match body user_id")
    if (
        len(timestamp) > 12
        or not timestamp.isascii()
        or not timestamp.isdecimal()
    ):
        raise WebIdentityVerificationError("Invalid web identity timestamp")

    timestamp_seconds = int(timestamp)
    current_seconds = time.time() if now is None else float(now)
    if abs(current_seconds - timestamp_seconds) > max_clock_skew_seconds:
        raise WebIdentityVerificationError("Expired web identity signature")

    normalized_nonce = str(nonce).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", normalized_nonce):
        raise WebIdentityVerificationError("Invalid web identity nonce")

    normalized_signature = str(signature).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized_signature):
        raise WebIdentityVerificationError("Invalid web identity signature")

    expected_signature = build_web_identity_signature(
        secret,
        method=method,
        path=path,
        user_id=normalized_header_user_id,
        timestamp=timestamp,
        nonce=normalized_nonce,
        raw_body=raw_body,
    )
    if not hmac.compare_digest(normalized_signature, expected_signature):
        raise WebIdentityVerificationError("Invalid web identity signature")

    nonce_cache = replay_cache or _WEB_IDENTITY_REPLAY_CACHE
    if not nonce_cache.consume(
        normalized_nonce,
        now=current_seconds,
        ttl_seconds=max_clock_skew_seconds,
    ):
        raise WebIdentityVerificationError("Replayed web identity signature")

    return normalized_header_user_id
