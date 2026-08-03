"""Conversation and operation state primitives for the agent harness."""
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Dict, List, Optional, Tuple, Union

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


PendingState = Union[PendingAddWord, PendingToolConfirm, None]


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
        target = " ".join(part for part in (self.word, self.code) if part)
        if (
            isinstance(self.pending_state, PendingToolConfirm)
            and self.pending_state.confirmation_source != "server_warning"
        ):
            if self.pending_state.function_name == "keytao_submit_batch":
                if target:
                    return f"确认提交 {target}"
            if self.pending_state.function_name in {
                "keytao_create_phrase",
                "keytao_batch_add_to_draft",
            }:
                if target:
                    return f"确认加入 {target}"
        if self.confirmation_code:
            return f"确认操作 {self.confirmation_code}"
        return ""


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
            confirmation_command = operation.confirmation_command
            operation.prompt_text = (
                prompt_text.rstrip()
                + "\n\n"
                + f"请回复「{confirmation_command}」继续。"
            )
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
        pending_ttl_seconds: float = 7200.0,
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
        self._states[address] = state
        self._records[address] = PendingStateRecord(
            state=state,
            owner_key=address,
            space_key=address.space_key,
            owner_label=owner_label,
            created_at=now,
            expires_at=now + self._pending_ttl_seconds,
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
        self._evict_over_capacity()
        return self._records.get(address) is not None

    def pop(self, key: ConversationKey) -> PendingState:
        record = self.pop_record(key)
        return record.state if record else None

    def pop_record(self, key: ConversationKey) -> Optional[PendingStateRecord]:
        address = self._resolve_address(key)
        self._purge_expired()
        if address is None:
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
        return record.arm_reconfirmation(
            confirmation_message,
            confirmation_intent=confirmation_intent,
            rotate_code=rotate_code,
        )

    @staticmethod
    def states_equivalent(left: PendingState, right: PendingState) -> bool:
        """Compare pending states by the operation they represent."""
        if left is None or right is None or type(left) is not type(right):
            return False
        if isinstance(left, PendingAddWord) and isinstance(right, PendingAddWord):
            return left == right
        if isinstance(left, PendingToolConfirm) and isinstance(right, PendingToolConfirm):
            return (
                left.function_name == right.function_name
                and left.args == right.args
                and left.confirmation_source == right.confirmation_source
            )
        return left == right

    def delete(self, key: ConversationKey) -> None:
        address = self._resolve_address(key)
        if address is None:
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

    def delete_actor(self, actor_key: ActorKey) -> int:
        """Delete every pending ticket owned by an actor across spaces."""
        self._purge_expired()
        addresses = [
            address for address in self._states
            if address.actor_key == actor_key
        ]
        for address in addresses:
            self._states.pop(address, None)
            self._records.pop(address, None)
        return len(addresses)

    def _resolve_address(self, key: ConversationKey) -> Optional[ConversationAddress]:
        if isinstance(key, ConversationAddress):
            return key
        return normalize_conversation_key(key)

    def _purge_expired(self) -> None:
        now = self._clock()
        expired = [
            address for address, record in self._records.items()
            if record.expires_at <= now
        ]
        for address in expired:
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
        return len(repr(state).encode("utf-8")) <= self._max_pending_payload_bytes
