"""Closed option answers backed by an existing actor-owned operation ticket."""

from dataclasses import dataclass
import re
import unicodedata
from typing import Mapping, Sequence


FORCE_ASSENT_TEXTS = frozenset({
    "强制", "硬加", "硬来", "强行", "就按我说的", "强制调序", "就这样", "照做",
})
_QUESTION_RE = re.compile(r"[^。！!？?\n]+[？?]")
_OPTION_SEPARATOR_RE = re.compile(r"\s*[,，]?\s*(?:还是|或者|或)\s*")
_OPTION_LEAD_RE = re.compile(
    r"^(?:(?:请问|您|你|我们|想要|想|是要|要|是|选择|回复|请)\s*)+"
)
_OPTION_LABEL_RE = re.compile(r"^(?:[A-Z]|[1-9][0-9]?)[.．、:：)）]\s*")
_QUOTE_PAIRS = (("「", "」"), ("『", "』"), ("“", "”"), ('"', '"'), ("'", "'"))


@dataclass(frozen=True)
class OptionQuestion:
    question: str
    options: tuple[str, ...]


def normalized_option_answer(text: str) -> str:
    """Normalize spacing and presentation punctuation, preserving sentence scope."""
    source = unicodedata.normalize("NFKC", str(text or "")).strip()
    source = re.sub(r"[。.!！]+$", "", source).strip()
    for opening, closing in _QUOTE_PAIRS:
        if source.startswith(opening) and source.endswith(closing):
            source = source[len(opening):-len(closing)].strip()
            break
    return re.sub(r"\s+", "", source)


def is_force_assent(text: str) -> bool:
    """Recognize complete force controls without accepting quoted/reported text."""
    source = unicodedata.normalize("NFKC", str(text or "")).strip()
    if any(mark in source for mark in '?？"\'`“”‘’「」『』'):
        return False
    return normalized_option_answer(source) in FORCE_ASSENT_TEXTS


def _option_text(text: str, *, first: bool) -> str:
    source = str(text or "").strip(" \t，,：:；;")
    if first:
        source = _OPTION_LEAD_RE.sub("", source)
    source = _OPTION_LABEL_RE.sub("", source).strip()
    return normalized_option_answer(source)


def structural_option_questions(text: str) -> tuple[OptionQuestion, ...]:
    """Find interrogative alternatives structurally, including implicit yes/no.

    An implicit ``是否`` question has no literal choices to bind. Its empty
    options deliberately force a deterministic redraw instead of inventing
    affirmative or negative strings absent from the displayed question.
    """
    result = []
    for match in _QUESTION_RE.finditer(str(text or "")):
        question = match.group(0).strip()
        body = question[:-1].strip()
        if not _OPTION_SEPARATOR_RE.search(body):
            if "是否" in body:
                result.append(OptionQuestion(question, ()))
            continue
        parts = _OPTION_SEPARATOR_RE.split(body)
        options = tuple(
            _option_text(value, first=index == 0)
            for index, value in enumerate(parts)
        )
        if len(options) < 2 or any(not option for option in options):
            options = ()
        result.append(OptionQuestion(question, options))
    return tuple(result)


def previous_bot_options(history: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """Read only the last bot turn from already actor/conversation-scoped history."""
    if not history or history[-1].get("role") != "assistant":
        return ()
    text = str(history[-1].get("content") or "")
    return tuple(dict.fromkeys(
        option
        for question in structural_option_questions(text)
        for option in question.options
    ))


def matches_previous_bot_option(
    text: str, history: Sequence[Mapping[str, object]],
) -> bool:
    source = normalized_option_answer(text)
    return bool(source and source in previous_bot_options(history))


def offered_option_intent(text: str, state: object) -> str:
    """Resolve an exact sealed alias without deriving any write target from prose."""
    from keytao_bot.harness.state import PendingToolConfirm, server_warning_ticket_is_complete

    if not isinstance(state, PendingToolConfirm) or not server_warning_ticket_is_complete(state):
        return ""
    mapping = state.args.get("_offered_options")
    if not isinstance(mapping, dict) or not mapping:
        return ""
    source = normalized_option_answer(text)
    if not source or re.search(r"[?？]", source):
        return ""
    for option, intent in mapping.items():
        if not isinstance(option, str) or option != normalized_option_answer(option):
            return ""
        if intent not in {"confirm", "cancel"}:
            return ""
    value = mapping.get(source)
    return value if value in {"confirm", "cancel"} else ""


def option_questions_bind_live_state(text: str, state: object) -> bool:
    """Round-trip every literal choice through the live parser and its binder."""
    questions = structural_option_questions(text)
    if not questions:
        return True
    from keytao_bot.plugins.chat_routing import (
        _message_authorizes_pending_state_control,
        _pending_tool_assent_intent,
    )

    mapping = getattr(state, "args", {}).get("_offered_options")
    if not isinstance(mapping, dict) or set(mapping) != {
        option for question in questions for option in question.options
    }:
        return False

    for question in questions:
        if not question.options:
            return False
        for option in question.options:
            expected = offered_option_intent(option, state)
            if not expected:
                return False
            parsed = _pending_tool_assent_intent(state, option)
            if (
                parsed is None
                or parsed.intent != f"pending_{expected}"
                or not _message_authorizes_pending_state_control(state, option, parsed)
            ):
                return False
    return True


def render_missing_option_ticket(history: Sequence[Mapping[str, object]]) -> str:
    """Recall display context without reviving a mutation from assistant prose."""
    empty = "当前没有待执行的操作；本次未写入。需要引用原提议确认后重新核对。"
    missing = "当前没有可执行的确认记录；本次未写入。需要引用原提议确认后重新核对。"
    stale_missing = "当前没有可执行的确认记录；之前的确认已过期或记录已不存在；本次未写入。需要引用原提议确认后重新核对。"
    previous = (
        str(history[-1].get("content") or "").strip()
        if history and history[-1].get("role") == "assistant"
        else ""
    )
    if not previous or previous == empty:
        return empty
    if previous.startswith(("上一条提议是：", "上一条提议的选项是：")) and previous.endswith((missing, stale_missing)):
        return previous
    options = previous_bot_options(history)
    if options:
        proposal = "、".join(f"「{option}」" for option in options)
        return (
            f"上一条提议的选项是：{proposal}。"
            f"{missing}"
        )
    from .pending_confirmation import advertised_command_suggestions, parse_pending_assent_phrase

    suggestions = advertised_command_suggestions(previous)
    if any(parse_pending_assent_phrase(command).matched for command in suggestions):
        missing = stale_missing
    if not suggestions and not re.search(r"计划|方案|将执行|将把|是否|建议|要不要", previous):
        return empty
    display_lines = [
        line.strip() for line in previous.splitlines()
        if line.strip() and not advertised_command_suggestions(line)
    ]
    proposal = "；".join((display_lines or list(suggestions))[:4])[:240].replace("？", "。").replace("?", ".")
    return (
        f"上一条提议是：{proposal}。"
        f"{missing}"
    )
