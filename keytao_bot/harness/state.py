"""Conversation and operation state primitives for the agent harness."""
import asyncio
import json
import math
import os
import re
import sqlite3
import time
import unicodedata
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple, Union

from nonebot.log import logger

from keytao_bot.utils.observability import mark_turn_outcome
from keytao_bot.utils.pending_confirmation import pending_confirmation_copy

from .conversation import (
    ConversationAddress,
    ConversationKey,
    SpaceKey,
    normalize_conversation_key,
)

ActorKey = Tuple[str, str]


@dataclass
class PendingAddWord:
    """User has been shown candidate codes, waiting for choice."""
    word: str
    recommended_code: str
    candidates: List[Tuple[str, bool]]
    occupied_words: Dict[str, List[str]] = field(default_factory=dict)
    # This capability is populated only from structured server candidate data.
    # Display text parsing must leave it empty so quoted or model-authored prose
    # can never mint positional create authority.
    server_candidates: List[Tuple[str, bool]] = field(default_factory=list)
    server_occupied_words: Dict[str, List[str]] = field(default_factory=dict)
    server_entries_by_code: Dict[str, List[Tuple[str, int]]] = field(
        default_factory=dict
    )
    # Advisory ordering facts are accepted only with the same structured
    # server snapshot as candidate occupancy. They do not authorize a reply.
    server_ordering_assessments: List[Dict[str, str]] = field(
        default_factory=list
    )
    code_remarks: Dict[str, str] = field(default_factory=dict)
    pronunciation_codes: Dict[str, str] = field(default_factory=dict)
    pronunciation_recommended_codes: List[str] = field(default_factory=list)
    # Structured auto-review verdict carried from keytao_prepare_reviewed_add.
    # ``code_remarks`` are parsed out of LLM prose and are display-only; this
    # boolean is the authoritative one, so the seal cannot be lost when the
    # model rewords its answer, nor forged when it happens to echo the prefix.
    needs_manual_review: Optional[bool] = None
    manual_review_reason: str = ""


@dataclass
class PendingToolConfirm:
    """A staged mutation or server warning waiting for user confirmation."""
    function_name: str
    args: Dict
    confirmation_source: str = "local_preview"


def _server_confirmation_display(data: Dict) -> Dict:
    """Keep JSON-safe server facts for display, never for write authority."""
    display: Dict[str, Any] = {}
    batch_url = str(data.get("batchUrl") or "").strip()
    if (
        data.get("batchIdProvisional") is not True
        and len(batch_url) <= 2048
        and re.fullmatch(r"https?://[^\s]+", batch_url)
    ):
        display["batchUrl"] = batch_url

    for key in ("warnings", "snapshotItems", "targets", "items"):
        value = data.get(key)
        if not isinstance(value, list):
            continue
        try:
            display[key] = json.loads(json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
            ))
        except (TypeError, ValueError):
            continue

    shift_plan = data.get("shiftPlan")
    if isinstance(shift_plan, dict):
        try:
            display["shiftPlan"] = json.loads(json.dumps(
                shift_plan,
                ensure_ascii=False,
                allow_nan=False,
            ))
        except (TypeError, ValueError):
            pass
    return display


def server_warning_pending_state(
    state: PendingToolConfirm,
    data: Dict,
) -> PendingToolConfirm:
    """Seal one pending state to the server preview that produced it."""
    args = dict(state.args)
    args.pop("confirmed", None)
    args.pop("preview_only", None)
    response_content_version = data.get("contentVersion")
    planned_content_version = args.get("expected_content_version")
    planned_absence = state.function_name == "keytao_shift_phrase_code" and (
        (
            "batch_id" in args
            and not str(args.get("batch_id") or "").strip()
            and isinstance(planned_content_version, int)
            and not isinstance(planned_content_version, bool)
            and planned_content_version == 0
        )
        or (
            "batchId" in data
            and not str(data.get("batchId") or "").strip()
            and isinstance(response_content_version, int)
            and not isinstance(response_content_version, bool)
            and response_content_version == 0
        )
    )
    batch_id = (
        ""
        if planned_absence
        else str(data.get("batchId") or args.get("batch_id") or "").strip()
    )
    bound_functions = {
        "keytao_create_phrase",
        "keytao_submit_batch",
        "keytao_batch_add_to_draft",
        "keytao_shift_phrase_code",
        "keytao_recall_batch",
        "keytao_remove_draft_item",
        "keytao_batch_remove_draft_items",
    }
    if (batch_id or planned_absence) and state.function_name in bound_functions:
        args["batch_id"] = batch_id
    content_version = 0 if planned_absence else response_content_version
    if (
        state.function_name in bound_functions
        and isinstance(content_version, int)
        and not isinstance(content_version, bool)
        and content_version >= 0
    ):
        args["expected_content_version"] = content_version
    if state.function_name == "keytao_shift_phrase_code":
        plan_digest = str(data.get("planDigest") or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", plan_digest):
            args["confirmed_plan_digest"] = plan_digest
    if state.function_name in {
        "keytao_create_phrase",
        "keytao_batch_add_to_draft",
        "keytao_shift_phrase_code",
    }:
        warning_digest = str(data.get("warningDigest") or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", warning_digest):
            args["expected_warning_digest"] = warning_digest
    if state.function_name == "keytao_submit_batch":
        digest_fields = {
            "expected_server_snapshot_digest": "snapshotDigest",
            "expected_warning_digest": "warningDigest",
            "expected_audit_digest": "auditDigest",
        }
        for argument_name, response_name in digest_fields.items():
            digest = str(data.get(response_name) or "").strip().lower()
            if re.fullmatch(r"[0-9a-f]{64}", digest):
                args[argument_name] = digest
    if state.function_name in {
        "keytao_remove_draft_item",
        "keytao_batch_remove_draft_items",
    }:
        target_digest = str(data.get("targetDigest") or "").strip().lower()
        targets = data.get("targets")
        if re.fullmatch(r"[0-9a-f]{64}", target_digest) and isinstance(targets, list):
            args["expected_target_digest"] = target_digest
            args["expected_targets"] = targets
    pending_display = _server_confirmation_display(data)
    if pending_display:
        args["_pending_display"] = pending_display
    else:
        args.pop("_pending_display", None)
    return PendingToolConfirm(
        function_name=state.function_name,
        args=args,
        confirmation_source="server_warning",
    )


def server_warning_ticket_is_complete(state: PendingToolConfirm) -> bool:
    """Return whether a server ticket has every field required for one replay."""
    if state.confirmation_source != "server_warning":
        return False
    args = state.args if isinstance(state.args, dict) else {}
    version = args.get("expected_content_version")
    valid_version = bool(
        isinstance(version, int)
        and not isinstance(version, bool)
        and version >= 0
    )
    function_name = state.function_name
    if function_name in {
        "keytao_remove_draft_item",
        "keytao_batch_remove_draft_items",
    }:
        return bool(
            args.get("batch_id")
            and valid_version
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(args.get("expected_target_digest") or ""),
            )
            and isinstance(args.get("expected_targets"), list)
            and args.get("expected_targets")
        )
    if function_name == "keytao_recall_batch":
        return bool(args.get("batch_id") and valid_version)
    if function_name == "keytao_shift_phrase_code":
        return bool(
            valid_version
            and (args.get("batch_id") or version == 0)
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(args.get("confirmed_plan_digest") or ""),
            )
        )
    if function_name in {"keytao_create_phrase", "keytao_batch_add_to_draft"}:
        return bool(
            args.get("batch_id")
            and valid_version
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(args.get("expected_warning_digest") or ""),
            )
        )
    if function_name == "keytao_submit_batch":
        return bool(
            args.get("batch_id")
            and valid_version
            and all(
                re.fullmatch(r"[0-9a-f]{64}", str(args.get(key) or ""))
                for key in (
                    "expected_server_snapshot_digest",
                    "expected_warning_digest",
                    "expected_audit_digest",
                )
            )
        )
    return False


def pending_execution_args(state: PendingToolConfirm) -> Dict:
    """Return sealed execution arguments without display-only facts."""
    args = dict(state.args)
    args.pop("_pending_display", None)
    return args


def pending_batch_display_pairs(
    state: PendingToolConfirm,
) -> Tuple[Tuple[str, str], ...]:
    """Return the normalized ordered word/code facts visible for a batch."""
    if state.function_name != "keytao_batch_add_to_draft":
        return ()
    items = state.args.get("items")
    if not isinstance(items, list) or not items:
        return ()
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            return ()
        word = unicodedata.normalize(
            "NFKC",
            str(item.get("word") or ""),
        ).strip()
        code = unicodedata.normalize(
            "NFKC",
            str(item.get("code") or ""),
        ).strip().lower()
        pair = (word, code)
        if not word or not code or pair in seen:
            return ()
        seen.add(pair)
        pairs.append(pair)
    return tuple(pairs)


@dataclass(frozen=True)
class AdvertisedWordSetSnapshot:
    """One actor-owned word universe derived from a server batch lookup."""

    token: str
    words: Tuple[str, ...]
    created_at: float
    expires_at: float


@dataclass
class PendingAdvertisedWordSets:
    """Live advertised universes awaiting one set-reference selection."""

    snapshots: List[AdvertisedWordSetSnapshot] = field(default_factory=list)


PendingState = Union[
    PendingAddWord,
    PendingAdvertisedWordSets,
    PendingToolConfirm,
    None,
]


@dataclass
class ActiveDraftOperation:
    """One serialized draft mutation that may continue in the background."""
    operation_id: str
    owner_key: ConversationAddress
    kind: str
    word: str = ""
    code: str = ""
    remark: str = ""
    status: str = "queued"
    pending_state: PendingState = None
    prompt_text: str = ""
    confirmation_code: str = ""
    trusted_links: Dict[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.owner_key = normalize_conversation_key(self.owner_key)

    @property
    def description(self) -> str:
        if self.word and self.code:
            return f"「{self.word}」→ {self.code}"
        if self.word:
            return f"「{self.word}」"
        return "当前草稿"

    @property
    def confirmation_command(self) -> str:
        return "确认" if isinstance(self.pending_state, PendingToolConfirm) else ""


class ConversationLockStore:
    """Provide one message-order lock per actor without blocking other actors."""

    def __init__(self) -> None:
        self._locks: Dict[ConversationAddress, asyncio.Lock] = {}
        self._users: Dict[ConversationAddress, int] = {}

    def get(self, key: ConversationKey) -> asyncio.Lock:
        address = normalize_conversation_key(key)
        lock = self._locks.get(address)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[address] = lock
        return lock

    @asynccontextmanager
    async def lock(self, key: ConversationKey) -> AsyncIterator[None]:
        """Serialize an actor and retire the lock after every queued user exits."""
        address = normalize_conversation_key(key)
        lock = self.get(address)
        self._users[address] = self._users.get(address, 0) + 1
        try:
            async with lock:
                yield
        finally:
            remaining = self._users.get(address, 1) - 1
            if remaining > 0:
                self._users[address] = remaining
            else:
                self._users.pop(address, None)
                if self._locks.get(address) is lock:
                    self._locks.pop(address, None)

    def __len__(self) -> int:
        return len(self._locks)


class DraftOperationCoordinator:
    """Track one draft mutation lifecycle per bound chat actor."""

    def __init__(self, confirmation_ttl_seconds: float = 7200.0) -> None:
        self._active: Dict[ConversationAddress, ActiveDraftOperation] = {}
        self._confirmation_ttl_seconds = max(1.0, confirmation_ttl_seconds)

    def get(self, key: ConversationKey) -> Optional[ActiveDraftOperation]:
        address = self._resolve_address(key)
        operation = self._active.get(address)
        if (
            operation is not None
            and operation.status == "awaiting_confirmation"
            and time.monotonic() - operation.updated_at > self._confirmation_ttl_seconds
        ):
            self._active.pop(address, None)
            return None
        return operation

    def begin(
        self,
        key: ConversationKey,
        kind: str,
        *,
        word: str = "",
        code: str = "",
        remark: str = "",
    ) -> Optional[ActiveDraftOperation]:
        address = normalize_conversation_key(key)
        if self.find_for_actor(address.actor_key) is not None:
            return None
        operation = ActiveDraftOperation(
            operation_id=uuid.uuid4().hex,
            owner_key=address,
            kind=kind,
            word=word,
            code=code,
            remark=remark,
        )
        self._active[address] = operation
        return operation

    def mark_running(self, key: ConversationKey, operation_id: str) -> bool:
        operation = self._matching(key, operation_id)
        if operation is None:
            return False
        operation.status = "running"
        operation.updated_at = time.monotonic()
        return True

    def mark_queued(self, key: ConversationKey, operation_id: str) -> bool:
        operation = self._matching(key, operation_id)
        if operation is None:
            return False
        operation.status = "queued"
        operation.updated_at = time.monotonic()
        return True

    def mark_awaiting_confirmation(
        self,
        key: ConversationKey,
        operation_id: str,
        pending_state: PendingState,
        prompt_text: str,
        *,
        rotate_code: bool = True,
    ) -> bool:
        operation = self._matching(key, operation_id)
        if operation is None:
            return False
        operation.status = "awaiting_confirmation"
        operation.pending_state = pending_state
        if rotate_code or not operation.confirmation_code:
            operation.confirmation_code = uuid.uuid4().hex[:6].upper()
            operation.prompt_text = prompt_text.rstrip()
            if pending_confirmation_copy() not in operation.prompt_text:
                operation.prompt_text += "\n\n" + pending_confirmation_copy()
        elif not operation.prompt_text:
            operation.prompt_text = prompt_text.rstrip()
        operation.updated_at = time.monotonic()
        return True

    def finish(self, key: ConversationKey, operation_id: str) -> bool:
        address = self._resolve_address(key)
        if self._matching(address, operation_id) is None:
            return False
        self._active.pop(address, None)
        return True

    def clear(self, key: ConversationKey) -> None:
        self._active.pop(self._resolve_address(key), None)

    def find_for_actor(self, actor_key: ActorKey) -> Optional[ActiveDraftOperation]:
        """Return the one active draft operation for an actor, in any space."""
        for address in list(self._active):
            operation = self.get(address)
            if operation is not None and address.actor_key == actor_key:
                return operation
        return None

    def _matching(
        self,
        key: ConversationKey,
        operation_id: str,
    ) -> Optional[ActiveDraftOperation]:
        operation = self._active.get(self._resolve_address(key))
        if operation is None or operation.operation_id != operation_id:
            return None
        return operation

    def _resolve_address(self, key: ConversationKey) -> ConversationAddress:
        if isinstance(key, ConversationAddress):
            return key
        return normalize_conversation_key(key)


@dataclass
class PendingStateRecord:
    """Pending state plus the actor/space that owns it."""
    state: PendingState
    owner_key: ConversationAddress
    space_key: Optional[SpaceKey] = None
    owner_label: str = ""
    created_at: float = 0.0
    expires_at: float = 0.0
    nonce: str = ""
    origin_message_id: str = ""
    origin_prompt_digest: str = ""
    requires_reconfirmation: bool = False
    confirmation_armed: bool = False
    reconfirmation_code: str = ""
    reconfirmation_message: str = ""
    reconfirmation_intent: Dict[str, object] = field(default_factory=dict)
    execution_id: str = ""
    execution_started_at: float = 0.0

    def __post_init__(self) -> None:
        """Normalize compatibility records into the full-address invariant."""
        self.owner_key = normalize_conversation_key(self.owner_key, self.space_key)
        self.space_key = self.owner_key.space_key

    def arm_reconfirmation(
        self,
        confirmation_message: str = "确认",
        *,
        confirmation_intent: Optional[Dict[str, object]] = None,
        rotate_code: bool = False,
    ) -> str:
        """Bind one exact challenge to a structured pending-control choice."""
        self.requires_reconfirmation = True
        self.confirmation_armed = True
        if rotate_code or not self.reconfirmation_code:
            self.reconfirmation_code = uuid.uuid4().hex[:6].upper()
        self.reconfirmation_message = str(confirmation_message or "确认")[:256]
        self.reconfirmation_intent = dict(
            confirmation_intent
            or {"intent": "pending_confirm", "confidence": 1.0}
        )
        return self.reconfirmation_code


class MemoryConversationStateStore:
    """In-memory pending tickets keyed by the full conversation address."""

    def __init__(
        self,
        states: Optional[Dict[ConversationKey, PendingState]] = None,
        *,
        pending_ttl_seconds: float = 14400.0,
        max_pending: int = 2048,
        max_pending_payload_bytes: int = 262_144,
        max_pending_items: int = 200,
        clock: Callable[[], float] = time.time,
    ):
        self._states: Dict[ConversationAddress, PendingState] = {}
        self._records: Dict[ConversationAddress, PendingStateRecord] = {}
        self._pending_ttl_seconds = max(1.0, float(pending_ttl_seconds))
        self._max_pending = max(1, int(max_pending))
        self._max_pending_payload_bytes = max(1024, int(max_pending_payload_bytes))
        self._max_pending_items = max(1, int(max_pending_items))
        self._clock = clock
        for key, state in (states or {}).items():
            if state is not None:
                self.set(key, state)

    @property
    def states(self) -> Dict[ConversationAddress, PendingState]:
        return self._states

    def live_entry_count(self) -> int:
        """Return the non-expired in-memory ticket count without mutating state."""
        now = self._clock()
        return sum(
            1
            for record in self._records.values()
            if record.expires_at > now
        )

    def get(self, key: ConversationKey) -> PendingState:
        address = self._resolve_address(key)
        self._purge_expired()
        return self._states.get(address) if address is not None else None

    def get_record(self, key: ConversationKey) -> Optional[PendingStateRecord]:
        address = self._resolve_address(key)
        self._purge_expired()
        return self._records.get(address) if address is not None else None

    def set(
        self,
        key: ConversationKey,
        state: PendingState,
        space_key: Optional[SpaceKey] = None,
        owner_label: str = "",
        origin_message_id: str = "",
    ) -> bool:
        if state is None:
            self.delete(key)
            return True
        if not self._state_within_limits(state):
            return False
        self._purge_expired()
        address = normalize_conversation_key(key, space_key)
        now = self._clock()
        previous = self._records.get(address)
        # A fresh ticket always represents a new authorization instance, even
        # when its visible payload happens to be equal to the previous one.
        # Restoring the same live ticket must use put_back(), which preserves
        # its nonce instead of creating a replacement authorization.
        is_mutating_pending = isinstance(state, (PendingAddWord, PendingToolConfirm))
        # Every mutating tool ticket gets its own exact challenge, including
        # the first ticket in an otherwise empty slot.  Otherwise a delayed
        # bare "confirm" sent for a consumed ticket could authorize whichever
        # replacement ticket happens to be current when the message arrives.
        requires_reconfirmation = is_mutating_pending or previous is not None
        advertised_created_at = (
            min(snapshot.created_at for snapshot in state.snapshots)
            if isinstance(state, PendingAdvertisedWordSets) and state.snapshots
            else now
        )
        advertised_expires_at = (
            max(snapshot.expires_at for snapshot in state.snapshots)
            if isinstance(state, PendingAdvertisedWordSets) and state.snapshots
            else now + self._pending_ttl_seconds
        )
        record = PendingStateRecord(
            state=state,
            owner_key=address,
            space_key=address.space_key,
            owner_label=owner_label,
            created_at=advertised_created_at,
            expires_at=advertised_expires_at,
            nonce=uuid.uuid4().hex,
            origin_message_id=str(origin_message_id or ""),
            requires_reconfirmation=requires_reconfirmation,
            confirmation_armed=is_mutating_pending,
            reconfirmation_code=(
                uuid.uuid4().hex[:6].upper()
                if requires_reconfirmation
                else ""
            ),
            reconfirmation_message="确认" if is_mutating_pending else "",
            reconfirmation_intent=(
                {"intent": "pending_confirm", "confidence": 1.0}
                if is_mutating_pending
                else {}
            ),
        )
        if not self._persist_record(record):
            return False
        self._states[address] = state
        self._records[address] = record
        self._evict_over_capacity()
        saved = self._records.get(address) is not None
        if saved:
            mark_turn_outcome("asked-confirmation")
        return saved

    def add_advertised_word_set(
        self,
        key: ConversationKey,
        words: Tuple[str, ...] | List[str],
        *,
        space_key: Optional[SpaceKey] = None,
        owner_label: str = "",
    ) -> str:
        """Append one bounded server-derived universe without replacing a ticket."""
        normalized = tuple(dict.fromkeys(
            str(word or "").strip()
            for word in words
            if str(word or "").strip()
        ))
        if len(normalized) < 2 or len(normalized) > self._max_pending_items:
            return ""
        if any(len(word.encode("utf-8")) > 512 for word in normalized):
            return ""
        self._purge_expired()
        address = normalize_conversation_key(key, space_key)
        current = self._records.get(address)
        if current is not None and not isinstance(
            current.state,
            PendingAdvertisedWordSets,
        ):
            return ""
        now = self._clock()
        snapshots = list(
            current.state.snapshots
            if current is not None
            and isinstance(current.state, PendingAdvertisedWordSets)
            else []
        )
        snapshots = [
            snapshot
            for snapshot in snapshots
            if snapshot.expires_at > now and snapshot.words != normalized
        ]
        token = uuid.uuid4().hex
        snapshots.append(AdvertisedWordSetSnapshot(
            token=token,
            words=normalized,
            created_at=now,
            expires_at=now + self._pending_ttl_seconds,
        ))
        if len(snapshots) > 4:
            return ""
        saved = self.set(
            address,
            PendingAdvertisedWordSets(snapshots=snapshots),
            space_key=address.space_key,
            owner_label=owner_label,
        )
        return token if saved else ""

    def advertised_word_sets(
        self,
        key: ConversationKey,
    ) -> Tuple[AdvertisedWordSetSnapshot, ...]:
        """Return only live universes owned by this exact conversation actor."""
        record = self.get_record(key)
        if record is None or not isinstance(
            record.state,
            PendingAdvertisedWordSets,
        ):
            return ()
        return tuple(record.state.snapshots)

    def replace_advertised_word_set(
        self,
        key: ConversationKey,
        token: str,
        pending: PendingToolConfirm,
        *,
        space_key: Optional[SpaceKey] = None,
        owner_label: str = "",
    ) -> bool:
        """Consume one exact advertised snapshot by replacing it with its ticket."""
        record = self.get_record(key)
        if (
            record is None
            or record.execution_id
            or not isinstance(record.state, PendingAdvertisedWordSets)
            or sum(
                1
                for snapshot in record.state.snapshots
                if snapshot.token == token
            )
            != 1
        ):
            return False
        return self.set(
            record.owner_key,
            pending,
            space_key=space_key or record.space_key,
            owner_label=owner_label or record.owner_label,
        )

    def pop(self, key: ConversationKey) -> PendingState:
        record = self.pop_record(key)
        return record.state if record else None

    def pop_record(self, key: ConversationKey) -> Optional[PendingStateRecord]:
        address = self._resolve_address(key)
        self._purge_expired()
        if address is None:
            return None
        if address in self._records or address in self._states:
            if not self._delete_persisted(address):
                return None
        record = self._records.pop(address, None)
        state = self._states.pop(address, None)
        if record is not None:
            return record
        if state is not None:
            now = self._clock()
            return PendingStateRecord(
                state=state,
                owner_key=address,
                space_key=address.space_key,
                created_at=now,
                expires_at=now,
                nonce=uuid.uuid4().hex,
            )
        return None

    def put_back(self, record: PendingStateRecord) -> bool:
        """Restore an unconsumed ticket without extending its TTL or nonce."""
        self._purge_expired()
        now = self._clock()
        if record.state is None or record.expires_at <= now:
            return False
        address = normalize_conversation_key(record.owner_key, record.space_key)
        record.owner_key = address
        record.space_key = address.space_key
        if not self._persist_record(record):
            return False
        self._states[address] = record.state
        self._records[address] = record
        self._evict_over_capacity()
        return self._records.get(address) is record

    def begin_execution(self, record: PendingStateRecord) -> bool:
        """Atomically mark an exact ticket as executing without discarding it."""
        self._purge_expired()
        address = normalize_conversation_key(record.owner_key, record.space_key)
        current = self._records.get(address)
        if current is not record or current.execution_id:
            return False
        current.execution_id = uuid.uuid4().hex
        current.execution_started_at = self._clock()
        return True

    def complete_execution(self, record: PendingStateRecord) -> bool:
        """Consume the exact ticket, preserving any replacement staged meanwhile."""
        address = normalize_conversation_key(record.owner_key, record.space_key)
        if self._records.get(address) is not record:
            return False
        if not self._delete_persisted(address):
            return False
        self._records.pop(address, None)
        self._states.pop(address, None)
        return True

    def abort_execution(self, record: PendingStateRecord) -> bool:
        """Release a claim only when no external mutation could have started."""
        address = normalize_conversation_key(record.owner_key, record.space_key)
        if self._records.get(address) is not record:
            return False
        record.execution_id = ""
        record.execution_started_at = 0.0
        return True

    def arm_reconfirmation(
        self,
        key: ConversationKey,
        confirmation_message: str = "确认",
        *,
        confirmation_intent: Optional[Dict[str, object]] = None,
        rotate_code: bool = False,
    ) -> Optional[str]:
        """Require an exact nonce before a staged mutation can be consumed."""
        record = self.get_record(key)
        if record is None:
            return None
        old_values = (
            record.requires_reconfirmation,
            record.confirmation_armed,
            record.reconfirmation_code,
            record.reconfirmation_message,
            dict(record.reconfirmation_intent),
        )
        confirmation_code = record.arm_reconfirmation(
            confirmation_message,
            confirmation_intent=confirmation_intent,
            rotate_code=rotate_code,
        )
        if not self._persist_record(record):
            (
                record.requires_reconfirmation,
                record.confirmation_armed,
                record.reconfirmation_code,
                record.reconfirmation_message,
                record.reconfirmation_intent,
            ) = old_values
            return None
        return confirmation_code

    def bind_origin_prompt_digest(self, key: ConversationKey, digest: str) -> bool:
        """Persist the exact bot-prompt digest used by native reply binding."""
        record = self.get_record(key)
        if record is None:
            return False
        previous = record.origin_prompt_digest
        record.origin_prompt_digest = str(digest or "")
        if self._persist_record(record):
            return True
        record.origin_prompt_digest = previous
        return False

    def _persist_record(self, record: PendingStateRecord) -> bool:
        """Persistence hook used by the durable production implementation."""
        return True

    def _delete_persisted(self, address: ConversationAddress) -> bool:
        """Persistence hook used by the durable production implementation."""
        return True

    @staticmethod
    def states_equivalent(left: PendingState, right: PendingState) -> bool:
        """Compare pending states by the operation they represent."""
        if left is None or right is None or type(left) is not type(right):
            return False
        if isinstance(left, PendingAddWord) and isinstance(right, PendingAddWord):
            display_fields_match = (
                left.word == right.word
                and left.recommended_code == right.recommended_code
                and left.candidates == right.candidates
                and left.occupied_words == right.occupied_words
                and left.code_remarks == right.code_remarks
                and left.pronunciation_codes == right.pronunciation_codes
                and left.pronunciation_recommended_codes
                == right.pronunciation_recommended_codes
                and left.needs_manual_review == right.needs_manual_review
                and left.manual_review_reason == right.manual_review_reason
            )
            if not display_fields_match:
                return False
            if left.server_candidates and right.server_candidates:
                return (
                    left.server_candidates == right.server_candidates
                    and left.server_occupied_words == right.server_occupied_words
                    and left.server_entries_by_code == right.server_entries_by_code
                )
            # Parsed display text may select an already-live record, but it can
            # never create the server capability that is absent from the text.
            return True
        if isinstance(left, PendingToolConfirm) and isinstance(right, PendingToolConfirm):
            if (
                left.function_name == right.function_name
                == "keytao_batch_add_to_draft"
            ):
                left_pairs = pending_batch_display_pairs(left)
                right_pairs = pending_batch_display_pairs(right)
                # Parsed batch display text may SELECT only the actor's own
                # already-live record. Execution must keep using that live
                # record's args, digests/warningDigest, CAS/version, nonce,
                # one-shot consumption, and actor binding; no quoted field may
                # reach the write payload or create/restore capability.
                return bool(left_pairs and left_pairs == right_pairs)
            return (
                left.function_name == right.function_name
                and left.args == right.args
                and left.confirmation_source == right.confirmation_source
            )
        if isinstance(left, PendingAdvertisedWordSets) and isinstance(
            right,
            PendingAdvertisedWordSets,
        ):
            # These snapshots are server-derived and have no display-text
            # parser. Keep their tokens, lifetimes, and word order exact.
            return left.snapshots == right.snapshots
        return left == right

    def delete(self, key: ConversationKey) -> None:
        address = self._resolve_address(key)
        if address is None:
            return
        if not self._delete_persisted(address):
            return
        self._states.pop(address, None)
        self._records.pop(address, None)

    def contains(self, key: ConversationKey) -> bool:
        return self.get(key) is not None

    def find_pending_for_other_owner(
        self,
        space_key: Optional[SpaceKey],
        owner_key: ConversationKey,
    ) -> Optional[PendingStateRecord]:
        """Return a pending state in the same space that belongs to another user."""
        if space_key is None:
            return None
        self._purge_expired()
        owner_address = normalize_conversation_key(owner_key, space_key)
        for record in self._records.values():
            if record.space_key == space_key and record.owner_key != owner_address:
                return record
        return None

    def find_matching_pending_for_other_owner(
        self,
        space_key: Optional[SpaceKey],
        owner_key: ConversationKey,
        state: PendingState,
    ) -> Optional[PendingStateRecord]:
        """Return another user's pending state in the same space that matches state."""
        if state is None:
            return None
        self._purge_expired()
        owner_address = normalize_conversation_key(owner_key, space_key)
        for record in self._records.values():
            if record.owner_key == owner_address:
                continue
            if record.space_key == space_key and self.states_equivalent(record.state, state):
                return record
        return None

    # Tickets for these tools describe a plan computed against the draft, so a
    # batch-level change (recall, clear) can invalidate them.  Anything else,
    # including an unconsumed code choice, survives.
    _DRAFT_SCOPED_FUNCTIONS = frozenset({
        "keytao_create_phrase",
        "keytao_batch_add_to_draft",
        "keytao_remove_draft_item",
        "keytao_batch_remove_draft_items",
        "keytao_shift_phrase_code",
        "keytao_submit_batch",
        "keytao_recall_batch",
    })

    def invalidate_actor_related(
        self,
        actor_key: ActorKey,
        *,
        batch_id: str = "",
    ) -> int:
        """Drop only the actor's tickets that this batch change invalidates.

        A recall used to wipe every pending ticket the user owned, including
        ones it had nothing to do with.  A ticket is now dropped only when it
        plans against the affected batch (or against whichever batch is current,
        when it carries no anchor of its own).
        """
        self._purge_expired()
        anchor = str(batch_id or "").strip()
        dropped = 0
        for address in list(self._states):
            if address.actor_key != actor_key:
                continue
            state = self._states.get(address)
            if not isinstance(state, PendingToolConfirm):
                continue
            if state.function_name not in self._DRAFT_SCOPED_FUNCTIONS:
                continue
            ticket_batch = str((state.args or {}).get("batch_id") or "").strip()
            if anchor and ticket_batch and ticket_batch != anchor:
                continue
            if not self._delete_persisted(address):
                continue
            self._states.pop(address, None)
            self._records.pop(address, None)
            dropped += 1
        return dropped

    def delete_actor(self, actor_key: ActorKey) -> int:
        """Delete every pending ticket owned by an actor across spaces."""
        self._purge_expired()
        addresses = [
            address for address in self._states
            if address.actor_key == actor_key
        ]
        deleted = 0
        for address in addresses:
            if not self._delete_persisted(address):
                continue
            self._states.pop(address, None)
            self._records.pop(address, None)
            deleted += 1
        return deleted

    def _resolve_address(self, key: ConversationKey) -> Optional[ConversationAddress]:
        if isinstance(key, ConversationAddress):
            return key
        return normalize_conversation_key(key)

    def _purge_expired(self) -> None:
        now = self._clock()
        for address, record in list(self._records.items()):
            if not isinstance(record.state, PendingAdvertisedWordSets):
                continue
            live = [
                snapshot
                for snapshot in record.state.snapshots
                if snapshot.expires_at > now
            ]
            if live:
                record.state.snapshots = live
                record.created_at = min(snapshot.created_at for snapshot in live)
                record.expires_at = max(snapshot.expires_at for snapshot in live)
                self._states[address] = record.state
            else:
                record.expires_at = now
        expired = [
            address for address, record in self._records.items()
            if record.expires_at <= now
        ]
        for address in expired:
            # Expiry is fail-closed in memory even if SQLite cleanup is
            # temporarily unavailable. A later boot re-applies the same TTL.
            self._delete_persisted(address)
            self._records.pop(address, None)
            self._states.pop(address, None)

    def _evict_over_capacity(self) -> None:
        overflow = len(self._records) - self._max_pending
        if overflow <= 0:
            return
        oldest = sorted(
            self._records.items(),
            key=lambda item: (item[1].created_at, item[1].nonce),
        )[:overflow]
        for address, _ in oldest:
            self._delete_persisted(address)
            self._records.pop(address, None)
            self._states.pop(address, None)

    def _state_within_limits(self, state: PendingState) -> bool:
        if isinstance(state, PendingToolConfirm):
            items = state.args.get("items")
            if isinstance(items, list) and len(items) > self._max_pending_items:
                return False
            try:
                payload = json.dumps(state.args, ensure_ascii=False, default=str).encode("utf-8")
            except (TypeError, ValueError):
                return False
            return len(payload) <= self._max_pending_payload_bytes
        if isinstance(state, PendingAddWord):
            return (
                len(state.word.encode("utf-8")) <= 512
                and len(state.candidates) <= self._max_pending_items
            )
        if isinstance(state, PendingAdvertisedWordSets):
            if not 1 <= len(state.snapshots) <= 4:
                return False
            tokens = [snapshot.token for snapshot in state.snapshots]
            return bool(
                len(set(tokens)) == len(tokens)
                and all(re.fullmatch(r"[0-9a-f]{32}", token) for token in tokens)
                and all(
                    2 <= len(snapshot.words) <= self._max_pending_items
                    and len(set(snapshot.words)) == len(snapshot.words)
                    and all(
                        word and len(word.encode("utf-8")) <= 512
                        for word in snapshot.words
                    )
                    and math.isfinite(snapshot.created_at)
                    and math.isfinite(snapshot.expires_at)
                    and snapshot.expires_at > snapshot.created_at
                    for snapshot in state.snapshots
                )
                and len(repr(state).encode("utf-8"))
                <= self._max_pending_payload_bytes
            )
        return len(repr(state).encode("utf-8")) <= self._max_pending_payload_bytes


class SQLiteConversationStateStore(MemoryConversationStateStore):
    """SQLite-backed tool-confirmation tickets with process-local claims."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        pending_ttl_seconds: float = 14400.0,
        max_pending: int = 2048,
        max_pending_payload_bytes: int = 262_144,
        max_pending_items: int = 200,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(
            pending_ttl_seconds=pending_ttl_seconds,
            max_pending=max_pending,
            max_pending_payload_bytes=max_pending_payload_bytes,
            max_pending_items=max_pending_items,
            clock=clock,
        )
        if db_path is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True, mode=0o700)
            db_path = str(data_dir / "pending_confirmations.db")
        self.db_path = str(db_path)
        self._secure_storage()
        self._init_db()
        self._load_persisted()
        self._secure_storage()
        logger.info(f"Initialized pending-confirmation store at: {self.db_path}")

    def _secure_storage(self) -> None:
        parent = Path(self.db_path).parent
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(parent, 0o700)
        except OSError as error:
            logger.warning(
                f"Failed to secure pending-confirmation directory {parent}: {error}"
            )
        for candidate in (
            self.db_path,
            f"{self.db_path}-wal",
            f"{self.db_path}-shm",
        ):
            if not os.path.exists(candidate):
                continue
            try:
                os.chmod(candidate, 0o600)
            except OSError as error:
                logger.warning(
                    f"Failed to secure pending-confirmation database {candidate}: {error}"
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        try:
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA secure_delete = ON")
            yield conn
            if conn.in_transaction:
                conn.commit()
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_confirmations (
                    platform TEXT NOT NULL,
                    space_type TEXT NOT NULL,
                    space_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    function_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    confirmation_source TEXT NOT NULL,
                    owner_label TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    nonce TEXT NOT NULL,
                    origin_message_id TEXT NOT NULL DEFAULT '',
                    origin_prompt_digest TEXT NOT NULL DEFAULT '',
                    requires_reconfirmation INTEGER NOT NULL DEFAULT 0,
                    confirmation_armed INTEGER NOT NULL DEFAULT 0,
                    reconfirmation_code TEXT NOT NULL DEFAULT '',
                    reconfirmation_message TEXT NOT NULL DEFAULT '',
                    reconfirmation_intent_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (platform, space_type, space_id, actor_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pending_confirmations_expiry
                ON pending_confirmations(expires_at)
            """)

    @staticmethod
    def _canonical_json(value: object) -> Tuple[str, object]:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return encoded, json.loads(encoded)

    def _persist_record(self, record: PendingStateRecord) -> bool:
        address = normalize_conversation_key(record.owner_key, record.space_key)
        if not isinstance(record.state, PendingToolConfirm):
            # Candidate selection and other conversational state remain
            # intentionally process-local in this round. Replacing a durable
            # tool ticket must still remove the old authorization from disk.
            return self._delete_persisted(address)
        try:
            arguments_json, canonical_arguments = self._canonical_json(
                record.state.args
            )
            intent_json, canonical_intent = self._canonical_json(
                record.reconfirmation_intent
            )
        except (TypeError, ValueError) as error:
            logger.warning(
                f"Pending confirmation contains non-canonical JSON data: {error}"
            )
            return False
        if not isinstance(canonical_arguments, dict) or not isinstance(
            canonical_intent, dict
        ):
            return False
        canonical_state = PendingToolConfirm(
            function_name=str(record.state.function_name),
            args=canonical_arguments,
            confirmation_source=str(record.state.confirmation_source),
        )
        if not self._state_within_limits(canonical_state):
            return False
        try:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO pending_confirmations (
                        platform, space_type, space_id, actor_id,
                        function_name, arguments_json, confirmation_source,
                        owner_label, created_at, expires_at, nonce,
                        origin_message_id, origin_prompt_digest,
                        requires_reconfirmation, confirmation_armed,
                        reconfirmation_code, reconfirmation_message,
                        reconfirmation_intent_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(platform, space_type, space_id, actor_id)
                    DO UPDATE SET
                        function_name = excluded.function_name,
                        arguments_json = excluded.arguments_json,
                        confirmation_source = excluded.confirmation_source,
                        owner_label = excluded.owner_label,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at,
                        nonce = excluded.nonce,
                        origin_message_id = excluded.origin_message_id,
                        origin_prompt_digest = excluded.origin_prompt_digest,
                        requires_reconfirmation = excluded.requires_reconfirmation,
                        confirmation_armed = excluded.confirmation_armed,
                        reconfirmation_code = excluded.reconfirmation_code,
                        reconfirmation_message = excluded.reconfirmation_message,
                        reconfirmation_intent_json = excluded.reconfirmation_intent_json,
                        updated_at = excluded.updated_at
                """, (
                    address.platform,
                    address.space_type,
                    address.space_id,
                    address.actor_id,
                    canonical_state.function_name,
                    arguments_json,
                    canonical_state.confirmation_source,
                    str(record.owner_label or ""),
                    float(record.created_at),
                    float(record.expires_at),
                    str(record.nonce or ""),
                    str(record.origin_message_id or ""),
                    str(record.origin_prompt_digest or ""),
                    int(bool(record.requires_reconfirmation)),
                    int(bool(record.confirmation_armed)),
                    str(record.reconfirmation_code or ""),
                    str(record.reconfirmation_message or ""),
                    intent_json,
                    float(self._clock()),
                ))
            self._secure_storage()
            return True
        except (OSError, sqlite3.Error) as error:
            logger.error(f"Failed to persist pending confirmation: {error}")
            return False

    def _delete_persisted(self, address: ConversationAddress) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("""
                    DELETE FROM pending_confirmations
                    WHERE platform = ? AND space_type = ?
                      AND space_id = ? AND actor_id = ?
                """, (
                    address.platform,
                    address.space_type,
                    address.space_id,
                    address.actor_id,
                ))
            return True
        except (OSError, sqlite3.Error) as error:
            logger.error(f"Failed to delete pending confirmation: {error}")
            return False

    def _load_persisted(self) -> None:
        now = self._clock()
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT rowid, *
                FROM pending_confirmations
                ORDER BY created_at, nonce
            """).fetchall()
            invalid_rowids: List[int] = []
            for row in rows:
                try:
                    address = ConversationAddress(
                        platform=str(row["platform"]),
                        space_type=str(row["space_type"]),
                        space_id=str(row["space_id"]),
                        actor_id=str(row["actor_id"]),
                    )
                    if (
                        not address.platform
                        or not address.actor_id
                        or address.space_type not in {"private", "group"}
                        or not address.space_id
                        or (
                            address.space_type == "private"
                            and address.space_id != address.actor_id
                        )
                    ):
                        raise ValueError("invalid conversation address")
                    arguments = json.loads(str(row["arguments_json"]))
                    reconfirmation_intent = json.loads(
                        str(row["reconfirmation_intent_json"])
                    )
                    state = PendingToolConfirm(
                        function_name=str(row["function_name"]),
                        args=arguments,
                        confirmation_source=str(row["confirmation_source"]),
                    )
                    created_at = float(row["created_at"])
                    stored_expiry = float(row["expires_at"])
                    expires_at = min(
                        stored_expiry,
                        created_at + self._pending_ttl_seconds,
                    )
                    nonce = str(row["nonce"])
                    origin_prompt_digest = str(row["origin_prompt_digest"])
                    reconfirmation_code = str(row["reconfirmation_code"])
                    if (
                        not isinstance(arguments, dict)
                        or not isinstance(reconfirmation_intent, dict)
                        or not state.function_name
                        or state.confirmation_source
                        not in {"local_preview", "server_warning"}
                        or not self._state_within_limits(state)
                        or not math.isfinite(created_at)
                        or not math.isfinite(expires_at)
                        or expires_at <= now
                        or not nonce
                        or len(nonce) > 128
                        or (
                            origin_prompt_digest
                            and not re.fullmatch(
                                r"[0-9a-f]{64}", origin_prompt_digest
                            )
                        )
                        or (
                            reconfirmation_code
                            and not re.fullmatch(
                                r"[A-F0-9]{6}", reconfirmation_code
                            )
                        )
                        or row["requires_reconfirmation"] not in (0, 1)
                        or row["confirmation_armed"] not in (0, 1)
                    ):
                        raise ValueError("invalid pending confirmation row")
                    record = PendingStateRecord(
                        state=state,
                        owner_key=address,
                        owner_label=str(row["owner_label"]),
                        created_at=created_at,
                        expires_at=expires_at,
                        nonce=nonce,
                        origin_message_id=str(row["origin_message_id"]),
                        origin_prompt_digest=origin_prompt_digest,
                        requires_reconfirmation=bool(
                            row["requires_reconfirmation"]
                        ),
                        confirmation_armed=bool(row["confirmation_armed"]),
                        reconfirmation_code=reconfirmation_code,
                        reconfirmation_message=str(
                            row["reconfirmation_message"]
                        ),
                        reconfirmation_intent=reconfirmation_intent,
                        # execution_id and execution_started_at are deliberately
                        # reconstructed as empty process-local claims.
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    invalid_rowids.append(int(row["rowid"]))
                    continue
                self._states[address] = state
                self._records[address] = record
            if invalid_rowids:
                conn.executemany(
                    "DELETE FROM pending_confirmations WHERE rowid = ?",
                    [(rowid,) for rowid in invalid_rowids],
                )
        self._evict_over_capacity()
