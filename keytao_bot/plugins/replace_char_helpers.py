"""Pure helpers for chunked batch character-replacement confirmations."""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from ..harness.authorization_grammar import trusted_mutation_source
from .chat_render import _format_replace_char_confirmation


_RE_WORD_CODE_LINE = re.compile(r"^(\S+)\s+([a-z]+)\s*$")
MAX_REPLACE_CHAR_ITEMS = 50
MAX_REPLACE_CONFIRMATION_CHARS = 3500

_TYPE_HINTS = [
    ("声笔笔单字", "CSSSingle"),
    ("CSSSingle", "CSSSingle"),
    ("css-single", "CSSSingle"),
    ("声笔笔", "CSS"),
    ("CSS", "CSS"),
    ("词组", "Phrase"),
    ("词语", "Phrase"),
    ("单字", "Single"),
    ("补充", "Supplement"),
    ("符号", "Symbol"),
    ("链接", "Link"),
    ("英文", "English"),
]


def _replace_char_continuation_digest(
    current_items: List[Dict],
    continuation: Dict[str, Any],
) -> str:
    payload = {
        "currentItems": current_items,
        "oldChar": continuation.get("oldChar"),
        "newChar": continuation.get("newChar"),
        "completed": continuation.get("completed"),
        "total": continuation.get("total"),
        "remainingItems": continuation.get("remainingItems"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sealed_replace_char_continuation(
    current_items: List[Dict],
    remaining_items: List[Dict],
    old_char: str,
    new_char: str,
    *,
    completed: int,
    total: int,
) -> Dict[str, Any]:
    continuation: Dict[str, Any] = {
        "oldChar": old_char,
        "newChar": new_char,
        "completed": completed,
        "total": total,
        "remainingItems": remaining_items,
    }
    continuation["digest"] = _replace_char_continuation_digest(
        current_items,
        continuation,
    )
    return continuation


def _valid_replace_char_continuation(
    current_items: Any,
    continuation: Any,
) -> bool:
    if not isinstance(current_items, list) or not isinstance(continuation, dict):
        return False
    remaining = continuation.get("remainingItems")
    completed = continuation.get("completed")
    total = continuation.get("total")
    old_char = continuation.get("oldChar")
    new_char = continuation.get("newChar")
    return bool(
        current_items
        and isinstance(remaining, list)
        and remaining
        and isinstance(completed, int)
        and not isinstance(completed, bool)
        and completed >= 0
        and isinstance(total, int)
        and not isinstance(total, bool)
        and total == completed + len(current_items) + len(remaining)
        and isinstance(old_char, str)
        and old_char
        and isinstance(new_char, str)
        and new_char
        and old_char != new_char
        and re.fullmatch(r"[0-9a-f]{64}", str(continuation.get("digest") or ""))
        and continuation["digest"]
        == _replace_char_continuation_digest(current_items, continuation)
    )


def _replace_char_chunk(
    items: List[Dict],
    old_char: str,
    new_char: str,
) -> Tuple[List[Dict], List[Dict], str]:
    """Choose the largest fully displayable first chunk, preserving order."""
    chunk_size = min(len(items), MAX_REPLACE_CHAR_ITEMS)
    while chunk_size > 0:
        chunk = items[:chunk_size]
        confirmation = _format_replace_char_confirmation(
            chunk,
            old_char,
            new_char,
        )
        if len(confirmation) <= MAX_REPLACE_CONFIRMATION_CHARS:
            return chunk, items[chunk_size:], confirmation
        chunk_size -= 1
    return [], list(items), ""


def _extract_explicit_phrase_type(message: str) -> Optional[str]:
    for hint, phrase_type in _TYPE_HINTS:
        if hint in message:
            return phrase_type
    return None


def _build_replace_char_items(
    message: str,
    old_char: str,
    new_char: str,
) -> List[Dict]:
    """Build trusted draft-change arguments from the current raw message only."""
    trusted_source = trusted_mutation_source(message)
    phrase_type = _extract_explicit_phrase_type(trusted_source)
    command_line_index = next(
        (
            index
            for index, line in enumerate(trusted_source.splitlines())
            if old_char in line
            and new_char in line
            and re.search(r"替换|改为|改成|换成", line)
        ),
        None,
    )
    if command_line_index is None:
        return []

    items: List[Dict] = []
    for line in trusted_source.splitlines()[command_line_index + 1:]:
        stripped_line = line.strip()
        if not stripped_line:
            continue
        match = _RE_WORD_CODE_LINE.match(stripped_line)
        if not match:
            if any(hint in stripped_line for hint, _phrase_type in _TYPE_HINTS):
                continue
            break
        old_word, code = match.group(1), match.group(2)
        if old_char not in old_word:
            break
        item = {
            "action": "Change",
            "old_word": old_word,
            "word": old_word.replace(old_char, new_char),
            "code": code,
        }
        if phrase_type:
            item["type"] = phrase_type
        items.append(item)
    return items
