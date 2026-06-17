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
