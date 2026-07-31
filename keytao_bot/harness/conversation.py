"""Canonical identity for one actor inside one chat space."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union


SpaceKey = Tuple[str, str]


@dataclass(frozen=True)
class ConversationAddress:
    """Identify one actor's conversation without mixing chat spaces."""

    platform: str
    space_type: str
    space_id: str
    actor_id: str

    @classmethod
    def private(cls, platform: str, actor_id: str) -> "ConversationAddress":
        return cls(platform=platform, space_type="private", space_id=actor_id, actor_id=actor_id)

    @classmethod
    def group(
        cls,
        platform: str,
        space_id: str,
        actor_id: str,
    ) -> "ConversationAddress":
        return cls(platform=platform, space_type="group", space_id=space_id, actor_id=actor_id)

    @property
    def actor_key(self) -> Tuple[str, str]:
        return (self.platform, self.actor_id)

    @property
    def space_scope_id(self) -> str:
        if self.space_type == "group":
            return f"{self.platform}:group:{self.space_id}"
        return f"{self.platform}:private:{self.actor_id}"

    @property
    def space_key(self) -> SpaceKey:
        return (self.platform, self.space_scope_id)


LegacyConversationKey = Tuple[str, str]
ConversationKey = Union[ConversationAddress, LegacyConversationKey]


def address_from_space_key(
    platform: str,
    actor_id: str,
    space_key: Optional[SpaceKey],
) -> ConversationAddress:
    """Build an address from the existing namespaced space-key format."""

    if not space_key:
        return ConversationAddress.private(platform, actor_id)
    _, scope_id = space_key
    group_prefix = f"{platform}:group:"
    if scope_id.startswith(group_prefix):
        return ConversationAddress.group(platform, scope_id[len(group_prefix):], actor_id)
    return ConversationAddress.private(platform, actor_id)


def normalize_conversation_key(
    key: ConversationKey,
    space_key: Optional[SpaceKey] = None,
) -> ConversationAddress:
    """Normalize legacy actor keys at compatibility adapters."""

    if isinstance(key, ConversationAddress):
        return key
    platform, actor_id = key
    return address_from_space_key(platform, actor_id, space_key)
