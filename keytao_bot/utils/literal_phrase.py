"""Extract bounded literal phrase operands without granting mutation authority."""

import re
from typing import Optional

from ..harness.authorization_grammar import (
    looks_like_lexical_review_target,
    message_mentions_change_request,
)


_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
_PHRASE_SHAPE_RE = re.compile(r"[\u3400-\u9fff]+(?:[，,、；;][\u3400-\u9fff]+)*")
_QUOTE_PAIRS = (("“", "”"), ("「", "」"), ("『", "』"), ('"', '"'))
_DATA_FRAGMENT_RE = re.compile(
    r"^(?:他说|她说|有人说|据说|听说|报道称|消息称|转发|转述|引用|记录)"
)
_COMMAND_FRAGMENT_RE = re.compile(
    r"^(?:(?:请|麻烦|帮我|帮忙|给我|先|再|然后)\s*)*"
    r"(?:不要|不用|无需|请勿|别|禁止|不得|不能|切勿|不)?"
    r"(?:加词|添加词|新增词|删词|删除|删掉|移除|提交|确认|取消|"
    r"加入|写入|放入|查询|查词|查看|解释|比较|排序|重排|"
    r"(?:把|将).+(?:加|删|移|调|改|换|写|放))"
)


def _source(message: str) -> str:
    if not isinstance(message, str) or _CONTROL_CHAR_RE.search(message):
        return ""
    return message.strip()


def _literal_operand(operand: str) -> Optional[str]:
    for opening, closing in _QUOTE_PAIRS:
        if not (operand.startswith(opening) and operand.endswith(closing)):
            continue
        phrase = operand[len(opening):-len(closing)]
        if not 1 <= len(phrase) <= 20 or not _PHRASE_SHAPE_RE.fullmatch(phrase):
            return None
        segments = re.split(r"[，,、；;]", phrase)
        if any(
            not looks_like_lexical_review_target(segment)
            or _DATA_FRAGMENT_RE.search(segment)
            or _COMMAND_FRAGMENT_RE.search(segment)
            or message_mentions_change_request(segment)
            for segment in segments
        ):
            return None
        return phrase
    return None


def parse_literal_phrase_query(message: str) -> Optional[str]:
    """Treat one closed lexical quote as one exact read-only lookup target."""
    return _literal_operand(_source(message))


def parse_explicit_single_phrase_add(message: str) -> Optional[str]:
    """Extract a directly requested single-entry review target, preserving punctuation."""
    source = _source(message)
    match = re.fullmatch(
        r"(?:请|麻烦)?\s*(?:帮我|帮忙|给我)?\s*(?:将|把)\s*"
        r"(?P<operand>.+?)\s*(?:作为|当作)\s*"
        r"(?:单个词条|单个词|一个词条|一个词)\s*"
        r"(?:加入|添加到|加到)\s*(?:词库|草稿)[。.!！]?",
        source,
    )
    return _literal_operand(match.group("operand")) if match else None
