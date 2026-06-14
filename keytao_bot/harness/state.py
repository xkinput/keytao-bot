"""Conversation state primitives for the agent harness."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


ConversationKey = Tuple[str, str]
SpaceKey = Tuple[str, str]


@dataclass
class PendingAddWord:
    """User has been shown candidate codes, waiting for choice."""
    word: str
    recommended_code: str
    candidates: List[Tuple[str, bool]]
    occupied_words: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class PendingToolConfirm:
    """A tool returned requiresConfirmation, waiting for user to confirm."""
    function_name: str
    args: Dict


PendingState = Union[PendingAddWord, PendingToolConfirm, None]


@dataclass
class PendingStateRecord:
    """Pending state plus the actor/space that owns it."""
    state: PendingState
    owner_key: ConversationKey
    space_key: Optional[SpaceKey] = None


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
    ) -> None:
        if state is None:
            self.delete(key)
            return
        self._states[key] = state
        self._records[key] = PendingStateRecord(
            state=state,
            owner_key=key,
            space_key=space_key,
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
        return None
