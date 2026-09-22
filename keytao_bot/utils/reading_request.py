"""Parse closed lexical lists with parenthesised pronunciation annotations."""

from dataclasses import dataclass
import re
import unicodedata

from ..harness.authorization_grammar import (
    looks_like_lexical_review_target,
    normalize_mutation_command_source,
)


@dataclass(frozen=True)
class ReadingRequest:
    word: str
    reading: str = ""
    code: str = ""


_SYLLABLE = r"[A-Za-züÜāáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ]+[0-5]?"
_ITEM = re.compile(
    r"(?P<word>[\u3400-\u9fff]{1,20})"
    rf"(?:\s*\(\s*(?P<reading>{_SYLLABLE}(?:\s+{_SYLLABLE})*)"
    r"\s*(?::\s*(?P<code>[a-z]{1,6})\s*)?\))?",
    re.IGNORECASE,
)
_SEPARATOR = re.compile(r"(?:\s*[，,、；;和]\s*|\s+)")


def parse_parenthesised_readings(message: str) -> tuple[ReadingRequest, ...]:
    """Consume the entire message; annotations never authorize a write."""
    source = unicodedata.normalize("NFKC", normalize_mutation_command_source(message))
    source = re.sub(r"^(?:加词|添加词|新增词|添加)\s*[:：]?\s*", "", source).strip()
    rows = []
    offset = 0
    while offset < len(source):
        match = _ITEM.match(source, offset)
        if match is None:
            return ()
        word = match.group("word")
        if (not looks_like_lexical_review_target(word)
                or re.match(r"^(?:不要|取消|确认|提交|删除|加词|添加|加入|他说|引用)", word)):
            return ()
        rows.append(ReadingRequest(word, str(match.group("reading") or "").strip(),
                                   str(match.group("code") or "").lower()))
        offset = match.end()
        if offset == len(source):
            break
        separator = _SEPARATOR.match(source, offset)
        if separator is None or separator.end() == len(source):
            return ()
        offset = separator.end()
    if not any(row.reading for row in rows) or len({row.word for row in rows}) != len(rows):
        return ()
    return tuple(rows)
