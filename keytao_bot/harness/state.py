"""Conversation and operation state primitives for the agent harness."""
import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Optional, Tuple, Union


ConversationKey = Tuple[str, str]
SpaceKey = Tuple[str, str]


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


@dataclass
class PendingToolConfirm:
    """A tool returned requiresConfirmation, waiting for user to confirm."""
    function_name: str
    args: Dict


PendingState = Union[PendingAddWord, PendingToolConfirm, None]


@dataclass
class ActiveDraftOperation:
    """One serialized draft mutation that may continue in the background."""
    operation_id: str
    owner_key: ConversationKey
    kind: str
    word: str = ""
    code: str = ""
    remark: str = ""
    status: str = "running"
    pending_state: PendingState = None
    prompt_text: str = ""
    started_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)

    @property
    def description(self) -> str:
        if self.word and self.code:
            return f"「{self.word}」→ {self.code}"
        if self.word:
            return f"「{self.word}」"
        return "当前草稿"


class ConversationLockStore:
    """Provide one message-order lock per actor without blocking other actors."""

    def __init__(self) -> None:
        self._locks: Dict[ConversationKey, asyncio.Lock] = {}
        self._users: Dict[ConversationKey, int] = {}

    def get(self, key: ConversationKey) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    @asynccontextmanager
    async def lock(self, key: ConversationKey) -> AsyncIterator[None]:
        """Serialize an actor and retire the lock after every queued user exits."""
        lock = self.get(key)
        self._users[key] = self._users.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            remaining = self._users.get(key, 1) - 1
            if remaining > 0:
                self._users[key] = remaining
            else:
                self._users.pop(key, None)
                if self._locks.get(key) is lock:
                    self._locks.pop(key, None)

    def __len__(self) -> int:
        return len(self._locks)


class DraftOperationCoordinator:
    """Track one draft mutation lifecycle per bound chat actor."""

    def __init__(self, confirmation_ttl_seconds: float = 7200.0) -> None:
        self._active: Dict[ConversationKey, ActiveDraftOperation] = {}
        self._confirmation_ttl_seconds = max(1.0, confirmation_ttl_seconds)

    def get(self, key: ConversationKey) -> Optional[ActiveDraftOperation]:
        operation = self._active.get(key)
        if (
            operation is not None
            and operation.status == "awaiting_confirmation"
            and time.monotonic() - operation.updated_at > self._confirmation_ttl_seconds
        ):
            self._active.pop(key, None)
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
        if self.get(key) is not None:
            return None
        operation = ActiveDraftOperation(
            operation_id=uuid.uuid4().hex,
            owner_key=key,
            kind=kind,
            word=word,
            code=code,
            remark=remark,
        )
        self._active[key] = operation
        return operation

    def mark_running(self, key: ConversationKey, operation_id: str) -> bool:
        operation = self._matching(key, operation_id)
        if operation is None:
            return False
        operation.status = "running"
        operation.updated_at = time.monotonic()
        return True

    def mark_awaiting_confirmation(
        self,
        key: ConversationKey,
        operation_id: str,
        pending_state: PendingState,
        prompt_text: str,
    ) -> bool:
        operation = self._matching(key, operation_id)
        if operation is None:
            return False
        operation.status = "awaiting_confirmation"
        operation.pending_state = pending_state
        operation.prompt_text = prompt_text
        operation.updated_at = time.monotonic()
        return True

    def finish(self, key: ConversationKey, operation_id: str) -> bool:
        if self._matching(key, operation_id) is None:
            return False
        self._active.pop(key, None)
        return True

    def clear(self, key: ConversationKey) -> None:
        self._active.pop(key, None)

    def _matching(
        self,
        key: ConversationKey,
        operation_id: str,
    ) -> Optional[ActiveDraftOperation]:
        operation = self._active.get(key)
        if operation is None or operation.operation_id != operation_id:
            return None
        return operation


@dataclass
class PendingStateRecord:
    """Pending state plus the actor/space that owns it."""
    state: PendingState
    owner_key: ConversationKey
    space_key: Optional[SpaceKey] = None
    owner_label: str = ""


class MemoryConversationStateStore:
    """In-memory pending-state store with actor ownership metadata."""

    def __init__(self, states: Optional[Dict[ConversationKey, PendingState]] = None):
        self._states: Dict[ConversationKey, PendingState] = states if states is not None else {}
        self._records: Dict[ConversationKey, PendingStateRecord] = {}
        if self._states:
            for key, state in self._states.items():
                if state is not None:
                    self._records[key] = PendingStateRecord(
                        state=state,
                        owner_key=key,
                    )

    @property
    def states(self) -> Dict[ConversationKey, PendingState]:
        return self._states

    def get(self, key: ConversationKey) -> PendingState:
        return self._states.get(key)

    def get_record(self, key: ConversationKey) -> Optional[PendingStateRecord]:
        return self._records.get(key)

    def set(
        self,
        key: ConversationKey,
        state: PendingState,
        space_key: Optional[SpaceKey] = None,
        owner_label: str = "",
    ) -> None:
        if state is None:
            self.delete(key)
            return
        self._states[key] = state
        self._records[key] = PendingStateRecord(
            state=state,
            owner_key=key,
            space_key=space_key,
            owner_label=owner_label,
        )

    def pop(self, key: ConversationKey) -> PendingState:
        record = self.pop_record(key)
        return record.state if record else None

    def pop_record(self, key: ConversationKey) -> Optional[PendingStateRecord]:
        record = self._records.pop(key, None)
        state = self._states.pop(key, None)
        if record is not None:
            return record
        if state is not None:
            return PendingStateRecord(state=state, owner_key=key)
        return None

    @staticmethod
    def states_equivalent(left: PendingState, right: PendingState) -> bool:
        """Compare pending states by the operation they represent."""
        if left is None or right is None or type(left) is not type(right):
            return False
        if isinstance(left, PendingAddWord) and isinstance(right, PendingAddWord):
            return (
                left.word == right.word
                and left.recommended_code == right.recommended_code
            )
        if isinstance(left, PendingToolConfirm) and isinstance(right, PendingToolConfirm):
            return (
                left.function_name == right.function_name
                and left.args == right.args
            )
        return left == right

    def delete(self, key: ConversationKey) -> None:
        self._states.pop(key, None)
        self._records.pop(key, None)

    def contains(self, key: ConversationKey) -> bool:
        return key in self._states

    def find_pending_for_other_owner(
        self,
        space_key: Optional[SpaceKey],
        owner_key: ConversationKey,
    ) -> Optional[PendingStateRecord]:
        """Return a pending state in the same space that belongs to another user."""
        if space_key is None:
            return None
        for record in self._records.values():
            if record.space_key == space_key and record.owner_key != owner_key:
                return record
            if (
                record.space_key is None
                and record.owner_key != owner_key
                and record.owner_key[0] == space_key[0]
            ):
                return record
        for key, state in self._states.items():
            if key == owner_key or state is None or key in self._records:
                continue
            platform, user_id = key
            legacy_space_key = (platform, f"{platform}:private:{user_id}")
            if legacy_space_key == space_key:
                return PendingStateRecord(
                    state=state,
                    owner_key=key,
                    space_key=legacy_space_key,
                )
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
        for record in self._records.values():
            if record.owner_key == owner_key:
                continue
            same_space = (
                record.space_key == space_key
                or (
                    record.space_key is None
                    and space_key is not None
                    and record.owner_key[0] == space_key[0]
                )
            )
            if same_space and self.states_equivalent(record.state, state):
                return record
        for key, candidate in self._states.items():
            if key == owner_key or candidate is None or key in self._records:
                continue
            platform, user_id = key
            legacy_space_key = (platform, f"{platform}:private:{user_id}")
            if legacy_space_key == space_key and self.states_equivalent(candidate, state):
                return PendingStateRecord(
                    state=candidate,
                    owner_key=key,
                    space_key=legacy_space_key,
                )
        return None
