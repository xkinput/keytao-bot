"""Conversation state primitives for the agent harness."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union


ConversationKey = Tuple[str, str]


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


class MemoryConversationStateStore:
    """In-memory conversation state store with a small explicit interface."""

    def __init__(self, states: Optional[Dict[ConversationKey, PendingState]] = None):
        self._states: Dict[ConversationKey, PendingState] = states if states is not None else {}

    @property
    def states(self) -> Dict[ConversationKey, PendingState]:
        return self._states

    def get(self, key: ConversationKey) -> PendingState:
        return self._states.get(key)

    def set(self, key: ConversationKey, state: PendingState) -> None:
        if state is None:
            self.delete(key)
            return
        self._states[key] = state

    def pop(self, key: ConversationKey) -> PendingState:
        return self._states.pop(key, None)

    def delete(self, key: ConversationKey) -> None:
        self._states.pop(key, None)

    def contains(self, key: ConversationKey) -> bool:
        return key in self._states
