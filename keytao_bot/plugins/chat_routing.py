"""Classify chat intent and match deterministic command structures."""

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from nonebot.log import logger

from ..harness.state import (
    ActiveDraftOperation,
    AdvertisedWordSetSnapshot,
    PendingAddWord,
    PendingAdvertisedWordSets,
    PendingState,
    PendingStateRecord,
    PendingTrustedWordRecord,
    PendingToolConfirm,
    pending_batch_display_pairs,
    trusted_word_record_is_complete,
)
from ..harness.authorization_grammar import (
    explicit_complete_add_item,
    parse_dictionary_delete_command,
    parse_entry_move_plan,
    parse_entry_swap,
    parse_eviction_modified_add,
    parse_pending_positional_add,
)
from ..harness.tools import (
    _COMMAND_PREFIX_PATTERN,
    _whole_message_unquoted_source,
    message_authorizes_mutation,
    trusted_mutation_source,
)
from ..utils.llm_policy import log_chat_usage, with_deepseek_chat_policy
from ..utils.observability import observe_model_call, set_turn_flow
from ..utils.pending_confirmation import (
    PENDING_ASSENT_TEXTS,
    PENDING_BATCH_ADD_AND_SUBMIT_ASSENT_TEXTS,
    PENDING_BATCH_ADD_ASSENT_TEXTS,
    PENDING_CONFIRM_ASSENT_TEXTS,
    parse_advertised_set_reference,
    parse_pending_assent_phrase,
    parse_pending_candidate_selection,
    pending_confirmation_copy,
    render_executable_suggestion,
    render_remediation_reply,
)
from .chat_adapters import (
    AsyncOpenAI,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_TIMEOUT,
    ReplyReferenceInfo,
    config,
)
from .chat_render import _format_pending_state_details


GROUP_TRIGGER_KEYWORD_START = "键道"


GROUP_TRIGGER_KEYWORD_ANY = "喵喵"


_ADVERTISED_COMMAND_QUOTE_SPAN_PATTERNS = (
    re.compile(r"「(?P<content>[^「」]*)」", re.DOTALL),
    re.compile(r"“(?P<content>[^“”]*)”", re.DOTALL),
    re.compile(r"『(?P<content>[^『』]*)』", re.DOTALL),
)


def _has_rejected_advertised_command_envelope(
    message_text: str,
    live_words: Tuple[str, ...],
) -> bool:
    """Recognize a live-set command quote whose outside residue is not allowed."""
    if _whole_message_unquoted_source(message_text, live_words) is not None:
        return False
    return any(
        parse_advertised_set_reference(match.group("content")).matched
        for pattern in _ADVERTISED_COMMAND_QUOTE_SPAN_PATTERNS
        for match in pattern.finditer(str(message_text or ""))
    )


@dataclass(frozen=True)
class AdvertisedWordSetSelection:
    """One closed set-subtraction resolution, or a deterministic ASK."""

    matched: bool = False
    snapshot_token: str = ""
    resolved_words: Tuple[str, ...] = ()
    exclusions: Tuple[str, ...] = ()
    ask: str = ""


def _resolve_advertised_word_set_selection(
    state: PendingState,
    message_text: str,
) -> AdvertisedWordSetSelection:
    """Resolve snapshot minus literal current-message exclusions, fail closed."""
    if not isinstance(state, PendingAdvertisedWordSets):
        return AdvertisedWordSetSelection()
    snapshots = tuple(state.snapshots)
    live_words = snapshots[0].words if len(snapshots) == 1 else ()
    source = _whole_message_unquoted_source(
        message_text,
        live_words,
    ) or message_text
    command = parse_advertised_set_reference(source)
    if not command.matched:
        return AdvertisedWordSetSelection()
    if len(snapshots) != 1:
        groups = "；".join(
            f"第{index}组：" + "、".join(snapshot.words)
            for index, snapshot in enumerate(snapshots, start=1)
        )
        suggestions = tuple(
            render_executable_suggestion(
                "加词 " + " ".join(snapshot.words),
                words=tuple(snapshot.words),
            )
            for snapshot in snapshots
        )
        return AdvertisedWordSetSelection(
            matched=True,
            ask=(
                f"当前有两组候选（{groups}），请选择一组：\n"
                + "\n".join(suggestion for suggestion in suggestions if suggestion)
            ),
        )
    snapshot: AdvertisedWordSetSnapshot = snapshots[0]
    if command.expected_count is not None and command.expected_count != len(snapshot.words):
        return AdvertisedWordSetSelection(
            matched=True,
            snapshot_token=snapshot.token,
            ask=render_remediation_reply(
                f"当前有效候选共有 {len(snapshot.words)} 个，不是指令中的 "
                f"{command.expected_count} 个；本次未写入",
                command="加词 " + " ".join(snapshot.words),
                words=tuple(snapshot.words),
            ),
        )
    exclusions = command.exclusions
    unknown = tuple(word for word in exclusions if word not in snapshot.words)
    if unknown:
        return AdvertisedWordSetSelection(
            matched=True,
            snapshot_token=snapshot.token,
            exclusions=exclusions,
            ask=render_remediation_reply(
                "以下暂不添加的词不在刚才的候选中："
                + "、".join(f"「{word}」" for word in unknown)
                + "。请只从「"
                + "、".join(snapshot.words)
                + "」中选择；本次未写入",
                command="加词 " + " ".join(snapshot.words),
                words=tuple(snapshot.words),
            ),
        )
    resolved = tuple(word for word in snapshot.words if word not in exclusions)
    if not resolved:
        return AdvertisedWordSetSelection(
            matched=True,
            snapshot_token=snapshot.token,
            exclusions=exclusions,
            ask=render_remediation_reply(
                "按这些排除项计算后没有剩余词；当前有效候选为「"
                + "、".join(snapshot.words)
                + "」；必须由你重新选择至少一个词，本次未写入",
                command="加词 " + " ".join(snapshot.words),
                words=tuple(snapshot.words),
            ),
        )
    return AdvertisedWordSetSelection(
        matched=True,
        snapshot_token=snapshot.token,
        resolved_words=resolved,
        exclusions=exclusions,
    )


_LEADING_COMMAND_PREFIX_RE = re.compile(
    r"^(?:@\S+|键道|喵喵)[\s:：，,]*",
    re.IGNORECASE,
)


_PURE_CHINESE_WORDS_RE = re.compile(r'^[\u4e00-\u9fff]+(?:[\s、，,；;]+[\u4e00-\u9fff]+)*$')


_PURE_CHINESE_TOKEN_RE = re.compile(r'^[\u4e00-\u9fff]{1,30}$')


_CODE_TOKEN_RE = re.compile(r"^[a-z]{2,12}$", re.IGNORECASE)


_REFERENCED_WORD_QUERY_HINTS = (
    "这两个词",
    "这俩词",
    "这几个词",
    "这些词",
    "上面两个词",
    "上面几个词",
    "引用里",
    "引用的",
)


_WORD_LIBRARY_QUERY_HINTS = (
    "词库",
    "收录",
    "编码",
)


_REFERENCED_WORD_PRESENCE_WHOLE_RE = re.compile(
    r"^(?:这两个词|这俩词|这几个词|这些词|上面两个词|上面几个词|"
    r"引用里(?:的)?这些?词|引用的(?:这两个|这俩|这几个|这些)?词)"
    r"(?:[，,])?(?:现在|目前)?"
    r"(?:"
    r"(?:是否|是不是)?(?:都|也)?在词库(?:里|中)?(?:都|也)?(?:有|收录(?:了)?)?|"
    r"(?:是否|是不是)?词库(?:里|中)?(?:都|也)?(?:有|收录)(?:了)?|"
    r"(?:是否|是不是)?(?:都|也)?(?:已经?)?(?:被)?(?:词库)?收录(?:了)?|"
    r"(?:是否|是不是)?(?:都|也)?有编码|"
    r"有没有(?:被词库)?收录|在不在词库"
    r")"
    r"(?:吗|么|嘛|没|没有)?[?？]?$"
)


@dataclass(frozen=True)
class CodeChainReorderCommand:
    """One direct request to reorder the server-resolved same-code chain."""

    code: str
    focus_words: Tuple[str, ...] = ()


@dataclass(frozen=True)
class WordListReorderCommand:
    """One explicit ordered list whose scope must be resolved server-side."""

    words: Tuple[str, ...]


_WORD_LIST_REORDER_WORD_PATTERN = r"[\u3400-\u9fff]{1,30}"
_WORD_LIST_REORDER_SEPARATOR_PATTERN = r"(?:\s+|[、\-‐‑‒–—]+)"
_WORD_LIST_REORDER_LIST_PATTERN = (
    rf"{_WORD_LIST_REORDER_WORD_PATTERN}"
    rf"(?:{_WORD_LIST_REORDER_SEPARATOR_PATTERN}"
    rf"{_WORD_LIST_REORDER_WORD_PATTERN}){{1,19}}"
)
_WORD_LIST_REORDER_VERB_PATTERN = (
    r"(?:重新排序(?:下|一下)?|排一下|按优先级排(?:一下)?)"
)
_WORD_LIST_REORDER_PREFIX_RE = re.compile(
    rf"^(?P<verb>{_WORD_LIST_REORDER_VERB_PATTERN})\s+"
    rf"(?P<words>{_WORD_LIST_REORDER_LIST_PATTERN})(?:吧)?$"
)
_WORD_LIST_REORDER_SUFFIX_RE = re.compile(
    rf"^(?P<words>{_WORD_LIST_REORDER_LIST_PATTERN})\s+"
    rf"(?P<verb>{_WORD_LIST_REORDER_VERB_PATTERN})(?:吧)?$"
)


def _parse_word_list_reorder_command(
    message_text: str,
) -> Optional[WordListReorderCommand]:
    """Parse a positive whole-message reorder over two or more literal words."""
    source = unicodedata.normalize(
        "NFKC",
        _strip_command_message_prefixes(message_text),
    ).strip()
    if (
        not source
        or re.search(r"[?？\"'“”‘’『』]", source)
        or re.search(r"^(?:不要|别|无需|不用|不必)", source)
        or re.search(r"^(?:他说|她说|他们说|有人说|引用|转述)[：:]?", source)
    ):
        return None
    source = re.sub(r"[。.!！~～]+$", "", source).strip()
    match = (
        _WORD_LIST_REORDER_PREFIX_RE.fullmatch(source)
        or _WORD_LIST_REORDER_SUFFIX_RE.fullmatch(source)
    )
    if match is None:
        return None
    words = tuple(
        token
        for token in re.split(
            _WORD_LIST_REORDER_SEPARATOR_PATTERN,
            str(match.group("words") or "").strip(),
        )
        if token
    )
    if (
        len(words) < 2
        or len(set(words)) != len(words)
        or any(_PURE_CHINESE_TOKEN_RE.fullmatch(word) is None for word in words)
    ):
        return None
    return WordListReorderCommand(words=words)


_CODE_CHAIN_REORDER_DIRECT_RE = re.compile(
    r"(?:重新排序|重新排|重排|排序|排)(?:下|一下)?"
    r"(?P<code>[a-z]{1,6})"
    r"(?:(?:这条)?(?:编码)?链)?"
    r"(?:(?:这几个|这些|这)词)?"
    r"(?:按(?:优先级|常用度)(?:排(?:序)?(?:一下)?|重新排序)?)?"
    r"(?:一下|吧)?",
    re.IGNORECASE,
)


_CODE_CHAIN_REORDER_BA_RE = re.compile(
    r"(?:把|将)?(?P<code>[a-z]{1,6})"
    r"(?:(?:这条)?(?:编码)?链)?"
    r"(?:(?:这几个|这些|这)词)?"
    r"(?:按(?:优先级|常用度))?"
    r"(?:重新排序|重新排|重排|排(?:序)?)(?:下|一下|吧)?",
    re.IGNORECASE,
)


_CODE_CHAIN_REORDER_CRITERION_FIRST_RE = re.compile(
    r"按(?:优先级|常用度)"
    r"(?:重新排序|重新排|重排|排(?:序)?)(?:下|一下)?"
    r"(?P<code>[a-z]{1,6})"
    r"(?:(?:这条)?(?:编码)?链)?"
    r"(?:(?:这几个|这些|这)词)?(?:一下|吧)?",
    re.IGNORECASE,
)


_CODE_CHAIN_REORDER_PAIR_RE = re.compile(
    r"调整(?P<code>[a-z]{1,6})"
    r"(?P<left>[\u3400-\u9fff]{1,30}?)(?:和|与|及|、)"
    r"(?P<right>[\u3400-\u9fff]{1,30}?)"
    r"(?:明显(?P<preferred>[\u3400-\u9fff]{1,30})更常用)?"
    r"(?:一下|吧)?",
    re.IGNORECASE,
)


def _parse_code_chain_reorder_command(
    message_text: str,
) -> Optional[CodeChainReorderCommand]:
    """Parse a closed, positive whole-message code-chain reorder command."""
    raw = unicodedata.normalize(
        "NFKC",
        _strip_command_message_prefixes(message_text),
    ).strip()
    if not raw:
        return None
    compact = re.sub(r"\s+", "", raw)
    compact = re.sub(r"[。.!！~～]+$", "", compact)
    polite = r"(?:(?:请|请你|帮我|麻烦|麻烦你|劳驾|拜托|给我))?"
    match = None
    for pattern in (
        _CODE_CHAIN_REORDER_DIRECT_RE,
        _CODE_CHAIN_REORDER_BA_RE,
        _CODE_CHAIN_REORDER_CRITERION_FIRST_RE,
    ):
        match = re.fullmatch(
            polite + r"(?:" + pattern.pattern + r")",
            compact,
            re.IGNORECASE,
        )
        if match is not None:
            break
    focus_words: Tuple[str, ...] = ()
    if match is None:
        pair_source = compact.replace("，", "").replace(",", "")
        match = re.fullmatch(
            polite + r"(?:" + _CODE_CHAIN_REORDER_PAIR_RE.pattern + r")",
            pair_source,
            re.IGNORECASE,
        )
        if match is not None:
            focus_words = (
                str(match.group("left") or "").strip(),
                str(match.group("right") or "").strip(),
            )
            preferred = str(match.group("preferred") or "").strip()
            if (
                len(set(focus_words)) != 2
                or (preferred and preferred not in focus_words)
            ):
                return None
    if match is None:
        return None
    code = str(match.group("code") or "").strip().lower()
    return (
        CodeChainReorderCommand(code=code, focus_words=focus_words)
        if _CODE_TOKEN_RE.fullmatch(code)
        else None
    )


_DRAFT_SUBMIT_COMMANDS = {
    "提交",
    "提审",
    "送审",
    "提交草稿",
    "提交批次",
    "提交审核",
    "提交当前草稿",
    "提交这个草稿",
    "发起审核",
}


_ACTION_SPECIFIC_DRAFT_SUBMIT_COMMANDS = {
    "确认提交",
    "继续提交",
}


_PENDING_CONFIRM_ASSENT_TEXTS = PENDING_CONFIRM_ASSENT_TEXTS


_PENDING_CONTROL_TEXTS = {
    *PENDING_ASSENT_TEXTS,
    "提交",
    "确认提交",
    "继续提交",
    "取消",
    "不用",
    "不要",
    "不了",
    "算了",
}


_PENDING_ADD_AND_SUBMIT_COMMANDS = PENDING_BATCH_ADD_AND_SUBMIT_ASSENT_TEXTS


WORD_QUERY_INTENT_MODEL = (
    getattr(config, "word_query_intent_model", None)
    or getattr(config, "openai_intent_model", None)
    or getattr(config, "gemini_intent_model", None)
    or OPENAI_MODEL
)


@dataclass(frozen=True)
class SimpleWordQueryIntent:
    should_handle: bool
    words: Tuple[str, ...] = ()
    intent: str = "not_word_lookup"
    confidence: float = 0.0


@dataclass(frozen=True)
class MessageCommandIntent:
    intent: str = "none"
    confidence: float = 0.0
    keep_words: Tuple[str, ...] = ()
    submit_after: bool = False
    clear_after: bool = False
    current_user_only: bool = False
    choice_index: Optional[int] = None
    choice_indices: Tuple[int, ...] = ()
    requested_code: str = ""
    requested_codes: Tuple[str, ...] = ()
    target_word: str = ""
    old_char: str = ""
    new_char: str = ""


_DRAFT_FLOW_INTENTS = frozenset({
    "draft_submit",
    "draft_view",
    "draft_recall",
    "draft_clear",
    "draft_keep_only",
    "operation_recall",
    "batch_replace_char",
    "code_chain_reorder",
    "word_list_reorder",
    "entry_swap",
    "entry_move_plan",
    "dictionary_delete",
})


def _record_flow_for_intent(command_intent: MessageCommandIntent) -> None:
    """Map the existing router result onto the stable observability buckets."""
    if command_intent.intent.startswith("pending_"):
        set_turn_flow("pending-confirmation")
    elif command_intent.intent in _DRAFT_FLOW_INTENTS:
        set_turn_flow("draft-op")


def _strip_command_message_prefixes(message_text: str) -> str:
    text = message_text.strip()
    while text:
        stripped = _LEADING_COMMAND_PREFIX_RE.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped
    return text


_EXECUTION_QUESTION_SUFFIX_RE = re.compile(
    r"(?:吗|么|好不好|行不行|可不可以|可以吗|好吗|行吗)$"
)


_EXECUTION_RESULT_SUFFIX_RE = re.compile(
    r"(?:(?:并|并且|然后|再|完成后|处理完后|操作完后)?"
    r"(?:告诉我|回复我|通知我)(?:一下)?(?:处理)?(?:结果)?"
    r"|(?:并|并且|然后|再)?(?:告诉我|回复我|通知我)(?:一下)?)$"
)


def _normalized_execution_command_text(message_text: str) -> str:
    """Normalize one positive execution request without trusting its targets."""
    if not message_authorizes_mutation(message_text):
        return ""
    compact = _compact_command_text(message_text)
    if not compact:
        return ""
    compact = _EXECUTION_QUESTION_SUFFIX_RE.sub("", compact)
    compact = _EXECUTION_RESULT_SUFFIX_RE.sub("", compact)
    return compact


def _matches_draft_submit_command(compact: str) -> bool:
    if not compact:
        return False
    prefix = _COMMAND_PREFIX_PATTERN
    target = r"(?:(?:当前|这个|我的)?(?:草稿|批次))"
    action = r"(?:提交|提审|送审)(?:审核)?"
    polite = r"(?:一下)?(?:吧|啦|了)?"
    return bool(
        re.fullmatch(rf"{prefix}(?:{action}(?:{target})?|(?:把|将)?{target}{action}|发起审核){polite}", compact)
    )


def _is_plain_draft_submit_request(message_text: str) -> bool:
    return _matches_draft_submit_command(
        _normalized_execution_command_text(message_text)
    )


def _is_explicit_draft_submit_request(message_text: str) -> bool:
    """Recognize a current-draft submit command without stealing questions."""
    return _is_plain_draft_submit_request(message_text)


def _is_pending_assent_then_submit_request(message_text: str) -> bool:
    """Recognize confirmation of one live write followed by draft submit."""
    compact = re.sub(
        r"[\s，,。.!！~～]+",
        "",
        str(message_text or ""),
    )
    return bool(re.fullmatch(
        r"(?:请|麻烦|帮我|给我|现在|立即|直接)*"
        r"确认(?:并|并且|然后|后|再)(?:提交|提审|送审)"
        r"(?:审核|草稿|批次)?(?:一下)?(?:吧|啦|了)?",
        compact,
    ))


@dataclass(frozen=True)
class KeepOnlyDraftCommand:
    keep_words: Tuple[str, ...]
    submit_after: bool


def _keep_only_command_from_intent(command_intent: MessageCommandIntent) -> Optional[KeepOnlyDraftCommand]:
    if command_intent.intent != "draft_keep_only" or not command_intent.keep_words:
        return None
    return KeepOnlyDraftCommand(
        keep_words=command_intent.keep_words,
        submit_after=command_intent.submit_after,
    )


def _is_sensitive_pending_control_intent(command_intent: MessageCommandIntent) -> bool:
    return command_intent.intent in {
        "pending_confirm",
        "pending_cancel",
        "pending_add_and_submit",
        "pending_recode",
        "pending_code_request",
        "pending_choice",
    }


def _pending_tool_assent_intent(
    state: PendingState,
    message_text: str,
) -> Optional[MessageCommandIntent]:
    """Resolve shared natural assent against one server-backed live state."""
    if isinstance(state, PendingToolConfirm):
        continuation_command = str(
            state.args.get("_continuation_command") or ""
        ).strip()
        if (
            continuation_command
            and _compact_command_text(message_text)
            == _compact_command_text(continuation_command)
            and not re.search(r"[?？]", message_text)
        ):
            return MessageCommandIntent(intent="pending_confirm", confidence=1.0)
    if (
        isinstance(state, PendingToolConfirm)
        and state.function_name == "keytao_submit_batch"
        and _is_explicit_draft_submit_request(message_text)
    ):
        return MessageCommandIntent(intent="pending_confirm", confidence=1.0)
    if (
        _is_explicit_draft_submit_request(message_text)
        and not _is_pending_assent_then_submit_request(message_text)
    ):
        # "提交" and "确认提交" target the current draft. They must not be
        # reinterpreted as assent to an unrelated add candidate merely because
        # the latter happens to be live in the same conversation.
        return None
    assent = _pending_assent_phrase_for_state(state, message_text)
    add_ticket = bool(
        isinstance(state, PendingAddWord)
        or (
            isinstance(state, PendingToolConfirm)
            and state.function_name in {
                "keytao_create_phrase",
                "keytao_batch_add_to_draft",
            }
        )
    )
    if (
        add_ticket
        and assent.rejection == "negation"
        and assent.cancel_requested
    ):
        return MessageCommandIntent(intent="pending_cancel", confidence=1.0)
    if not assent.matched:
        return None
    if isinstance(state, PendingAddWord):
        server_backed = bool(
            state.server_candidates
            and state.server_candidates == state.candidates
            and state.recommended_code
            in {code for code, _occupied in state.server_candidates}
        )
        if not server_backed:
            return None
        if assent.submit_after:
            return MessageCommandIntent(
                intent="pending_add_and_submit",
                confidence=1.0,
                submit_after=True,
            )
        return MessageCommandIntent(intent="pending_confirm", confidence=1.0)
    if not isinstance(state, PendingToolConfirm):
        return None
    add_confirmation_tool = state.function_name in {
        "keytao_create_phrase",
        "keytao_batch_add_to_draft",
    }
    if add_confirmation_tool and assent.submit_after:
        return MessageCommandIntent(
            intent="pending_add_and_submit",
            confidence=1.0,
            submit_after=True,
        )
    if add_confirmation_tool:
        return MessageCommandIntent(intent="pending_confirm", confidence=1.0)
    if not assent.add_requested:
        return MessageCommandIntent(intent="pending_confirm", confidence=1.0)
    return None


_TRUSTED_WORD_ACTION_NAMES = frozenset({"换码", "删除", "删掉", "删了", "移除"})


def _pending_trusted_word_action_matches(
    state: PendingState,
    message_text: str,
) -> bool:
    """Match a bare action name against the previous turn's trusted row."""
    normalized = _compact_command_text(message_text)
    return bool(
        isinstance(state, PendingTrustedWordRecord)
        and trusted_word_record_is_complete(state)
        and not re.search(r"[?？]", message_text)
        and normalized in _TRUSTED_WORD_ACTION_NAMES
    )


def _pending_positional_add_intent(
    state: Optional[PendingState],
    message_text: str,
) -> Optional[MessageCommandIntent]:
    """Bind an omitted add subject to one live server-backed candidate only."""
    if (
        not isinstance(state, PendingAddWord)
        or not state.server_candidates
        or state.server_candidates != state.candidates
    ):
        return None
    command = parse_pending_positional_add(message_text)
    if command is None:
        return None
    destination_codes = [
        code
        for code, words in state.server_occupied_words.items()
        if command.destination_word in words
    ]
    occupied = dict(state.server_candidates)
    if (
        len(destination_codes) != 1
        or occupied.get(destination_codes[0]) is not True
    ):
        return None
    return MessageCommandIntent(
        intent=(
            "pending_add_and_submit"
            if command.submit_after
            else "pending_confirm"
        ),
        confidence=1.0,
        submit_after=command.submit_after,
        requested_codes=(destination_codes[0],),
        target_word=command.destination_word,
    )


def _pending_assent_state_operands(state: PendingState) -> Tuple[str, ...]:
    """Allow text to repeat only one state's exact display, never select a subset."""
    if isinstance(state, PendingAddWord):
        return tuple(
            value
            for value in (state.word, state.recommended_code)
            if str(value or "").strip()
        )
    if not isinstance(state, PendingToolConfirm):
        return ()
    if state.function_name == "keytao_create_phrase":
        return tuple(
            value
            for value in (
                str(state.args.get("word") or "").strip(),
                str(state.args.get("code") or "").strip().lower(),
            )
            if value
        )
    pairs = pending_batch_display_pairs(state)
    if len(pairs) == 1:
        return pairs[0]
    return ()


def _pending_assent_phrase_for_state(
    state: PendingState,
    message_text: str,
):
    operands = _pending_assent_state_operands(state)
    live_words = tuple(
        value
        for value in operands
        if not re.fullmatch(r"[a-z]{1,12}", value, re.IGNORECASE)
    )
    unwrapped = _whole_message_unquoted_source(message_text, live_words)
    source = unwrapped if unwrapped is not None else message_text
    return parse_pending_assent_phrase(source, allowed_operands=operands)


def _pending_assent_rejection_response(
    state: PendingState,
    message_text: str,
) -> Optional[str]:
    """Explain why assent-like text did not authorize the one live state."""
    if (
        parse_eviction_modified_add(message_text) is not None
        or explicit_complete_add_item(message_text) is not None
        or _pending_positional_add_intent(state, message_text) is not None
    ):
        return None
    add_candidate = bool(
        isinstance(state, PendingAddWord)
        or (
            isinstance(state, PendingToolConfirm)
            and state.function_name in {
                "keytao_create_phrase",
                "keytao_batch_add_to_draft",
            }
        )
    )
    if not add_candidate:
        return None
    compact = _compact_command_text(message_text)
    exact_rereview = bool(
        isinstance(state, PendingAddWord)
        and compact == _compact_command_text(f"加词 {state.word}")
    )
    if (
        exact_rereview
        or parse_pending_candidate_selection(message_text) is not None
        or (
            isinstance(state, PendingAddWord)
            and _structural_pending_add_word_intent(message_text, state)
            is not None
        )
    ):
        return None
    assent = _pending_assent_phrase_for_state(state, message_text)
    if assent.cancel_requested:
        return None
    if not assent.recognized or assent.matched:
        return None
    reasons = {
        "question": "这条回复是问句，不能作为当前候选的执行确认；本次未写入",
        "negation": "这条回复含有否定，不能作为当前候选的执行确认；本次未写入",
        "framed": "这条回复是引用或转述，不是当前发送者的直接确认；本次未写入",
        "other_action": "这条回复还包含加入候选之外的其他动作；本次未写入",
        "extra_content": (
            "这条回复还包含当前候选之外的词条、编码或第二目标；"
            "系统不会用这些文字改写候选；本次未写入"
        ),
    }
    live_words = (
        (state.word,)
        if isinstance(state, PendingAddWord)
        else tuple(word for word, _code in pending_batch_display_pairs(state))
    )
    if not live_words and isinstance(state, PendingToolConfirm):
        word = str(state.args.get("word") or "").strip()
        live_words = (word,) if word else ()
    command = (
        ("加入并提交" if assent.submit_after else "加入")
        if live_words and assent.rejection != "negation"
        else ""
    )
    return render_remediation_reply(
        reasons.get(
            assent.rejection,
            "这条回复不是仅针对当前候选的同意；本次未写入",
        ),
        command=command,
        words=live_words,
    )


def _format_live_ticket_precedence_message(state: PendingState) -> str:
    details = _describe_pending_state(state)
    return f"{details}\n{pending_confirmation_copy()}"


def _compact_command_text(message_text: str) -> str:
    return re.sub(
        r"[\s，,。.!！?？~～…、;；:：\"'“”‘’（）()【】\[\]<>《》]+",
        "",
        _strip_command_message_prefixes(trusted_mutation_source(message_text)),
    )


def _is_short_add_and_submit_request(message_text: str) -> bool:
    """Recognize target-free add-and-submit controls."""
    stripped = _strip_command_message_prefixes(message_text)
    unwrapped = _whole_message_unquoted_source(stripped)
    source = unwrapped if unwrapped is not None else stripped
    if (
        re.search(r"[?？]", source)
        or not message_authorizes_mutation(source)
    ):
        return False
    assent = parse_pending_assent_phrase(source)
    return bool(
        assent.matched
        and assent.add_requested
        and assent.submit_after
    )


def _is_target_bound_add_and_submit_request(
    message_text: str,
    state: PendingAddWord,
) -> bool:
    """Require an unquoted add-and-submit command to name its exact target."""
    if not message_authorizes_mutation(_strip_command_message_prefixes(message_text)):
        return False
    compact = _compact_command_text(message_text)
    word = str(state.word or "").strip()
    code = str(state.recommended_code or "").strip().lower()
    if not word or not code:
        return False
    request_prefix = _COMMAND_PREFIX_PATTERN
    add_clause = r"(?:加词|添加|加入|新增)(?:词条)?"
    code_clause = rf"(?:(?:用|以|按)?(?:编码)?(?:为|是)?)?{re.escape(code)}"
    submit_clause = r"(?:并|并且|然后|再|同时|以及)?(?:提交|提审|送审)(?:审核|草稿|批次)?"
    polite = r"(?:一下)?(?:吧|啦|了)?"
    return bool(
        re.fullmatch(
            rf"{request_prefix}{add_clause}{re.escape(word)}{code_clause}{submit_clause}{polite}",
            compact,
            flags=re.IGNORECASE,
        )
    )


# Rendering/compatibility vocabulary only. Acceptance is owned exclusively by
# ``parse_pending_assent_phrase`` and the structural live-state parsers above.
_QUOTED_PENDING_ADD_CONFIRM_TEXTS = {
    *_PENDING_CONFIRM_ASSENT_TEXTS,
    *PENDING_BATCH_ADD_ASSENT_TEXTS,
}


def _quoted_pending_add_control_intent(
    message_text: str,
    state: PendingAddWord,
) -> Optional[MessageCommandIntent]:
    """Resolve controls carried by a verified native reply to a bot candidate."""
    assent = _pending_assent_phrase_for_state(state, message_text)
    if assent.matched:
        return MessageCommandIntent(
            intent=(
                "pending_add_and_submit"
                if assent.submit_after
                else "pending_confirm"
            ),
            confidence=1.0,
            submit_after=assent.submit_after,
        )
    structural = _structural_pending_add_word_intent(message_text, state)
    if structural is not None:
        return structural
    return None


def _message_authorizes_clear_history(
    message_text: str,
    command_intent: MessageCommandIntent,
) -> bool:
    if (
        command_intent.intent != "clear_history"
        or command_intent.confidence < 0.85
        or not message_authorizes_mutation(message_text)
    ):
        return False
    compact = _compact_command_text(message_text)
    return bool(
        re.search(r"(?:清空|清除|重置|删除).{0,8}(?:对话|聊天|历史|记忆)", compact)
        and not re.search(r"(?:什么意思|解释|如何|怎么|为什么|翻译|假设|如果)", compact)
    )


def _compact_requests_draft_clear_all(
    compact: str,
    *,
    allow_combined: bool = True,
) -> bool:
    """Recognize an entire clear-all clause, never a named draft subset."""
    request_prefix = r"(?:请|麻烦|帮我|给我|现在|立即|直接|确认|我要|我想|替我|为我)*"
    draft_target = r"(?:恢复(?:后(?:的)?|的))?(?:当前|我的)?(?:草稿|批次)"
    all_suffix = r"(?:(?:的)?(?:全部|所有)(?:条目|内容)?)?"
    clear_clause = (
        rf"(?:(?:清空|清除|清理){draft_target}{all_suffix}"
        rf"|(?:把|将)?{draft_target}{all_suffix}(?:清空|清除|清理)"
        rf"|(?:草稿|批次)(?:中的)?(?:全部|都|所有)(?:条目|内容)?(?:删除|删掉|移除)"
        rf"|(?:全部|都|所有)(?:删除|删掉|移除){draft_target}(?:条目|内容)?"
        rf"|(?:删除|删掉|移除)(?:全部|所有){draft_target}(?:条目|内容)?)"
    )
    polite = r"(?:一下)?(?:吧|啦|了)?"
    if re.fullmatch(rf"{request_prefix}{clear_clause}{polite}", compact):
        return True
    return bool(
        allow_combined
        and re.search(
            rf"(?:并|并且|然后|再|同时|以及){clear_clause}{polite}$",
            compact,
        )
    )


def _canonical_draft_management_command(
    message_text: str,
) -> Optional[MessageCommandIntent]:
    """Parse the complete positive syntax that may authorize draft mutations."""
    unwrapped = _whole_message_unquoted_source(message_text)
    if (
        unwrapped is None
        and re.search(r"[「」“”『』《》【】\[\]]", str(message_text or ""))
    ):
        return None
    raw = _strip_command_message_prefixes(
        trusted_mutation_source(
            unwrapped if unwrapped is not None else message_text
        )
    ).strip()
    if not raw:
        return None

    compact = _normalized_execution_command_text(raw)
    if not compact:
        raw_compact = _compact_command_text(raw)
        if (
            not re.search(r"[?？]", raw)
            and not _EXECUTION_QUESTION_SUFFIX_RE.search(raw_compact)
            and re.match(
                r"(?:请|麻烦|帮我|给我|现在|立即|直接|我要|我想|替我|为我)*"
                r"取消(?:(?:最近|上次|刚才)(?:一次|的)?)?(?:提审|送审)",
                raw_compact,
            )
        ):
            compact = raw_compact
        else:
            return None
    prefix = (
        r"(?:请|麻烦|帮我|给我|现在|立即|直接|确认|那就|可以|就|我要|我想|替我|为我|"
        r"能不能|可不可以|能否|可否|可以帮我|可以请你)*"
    )
    scope = r"(?:(?:最近|上次|刚才)(?:一次|的)?)?"
    recall_core = (
        rf"(?:(?:撤回|撤销|召回|回退)(?:"
        rf"{scope}(?:提交|提审|送审|审核|批次)?|"
        rf"{scope}(?:提交|提审|送审)(?:的)?批次)"
        rf"|取消{scope}(?:提审|送审)"
        rf"|(?:最近|上次|刚才)(?:那次|一次|的)?"
        rf"(?:提交|提审|送审)(?:撤回|撤销|召回|回退))"
    )
    draft_target = r"(?:恢复(?:后(?:的)?|的))?(?:当前|我的)?(?:草稿|批次)"
    all_suffix = r"(?:(?:的)?(?:全部|所有)(?:条目|内容)?)?"
    clear_clause = (
        rf"(?:(?:清空|清除|清理){draft_target}{all_suffix}"
        rf"|(?:把|将)?{draft_target}{all_suffix}(?:清空|清除|清理)"
        rf"|(?:草稿|批次)(?:中的)?(?:全部|都|所有)(?:条目|内容)?(?:删除|删掉|移除)"
        rf"|(?:全部|都|所有)(?:删除|删掉|移除){draft_target}(?:条目|内容)?"
        rf"|(?:删除|删掉|移除)(?:全部|所有){draft_target}(?:条目|内容)?)"
    )
    connector = r"(?:并|并且|然后|再|同时|以及)?"
    polite = r"(?:一下)?(?:吧|啦|了)?"

    if re.fullmatch(
        rf"{prefix}{recall_core}{polite}(?:{connector}{clear_clause}{polite})?",
        compact,
    ):
        return MessageCommandIntent(
            intent="draft_recall",
            confidence=1.0,
            clear_after=_compact_requests_draft_clear_all(compact),
        )
    if re.fullmatch(rf"{prefix}{clear_clause}{polite}", compact):
        return MessageCommandIntent(intent="draft_clear", confidence=1.0)
    return None


def _message_requests_draft_clear_all(message_text: str) -> bool:
    return _compact_requests_draft_clear_all(
        _compact_command_text(trusted_mutation_source(message_text))
    )


def _message_authorizes_draft_recall(
    message_text: str,
    command_intent: MessageCommandIntent,
) -> bool:
    """Bind a semantic recall decision to an explicit current-user command."""
    if (
        command_intent.intent != "draft_recall"
        or command_intent.confidence < 0.85
    ):
        return False
    canonical = _canonical_draft_management_command(message_text)
    return bool(
        canonical is not None
        and canonical.intent == "draft_recall"
        and (not command_intent.clear_after or canonical.clear_after)
    )


def _message_authorizes_draft_clear(
    message_text: str,
    command_intent: MessageCommandIntent,
) -> bool:
    """Bind semantic clear-all intent to the sender's current draft only."""
    if (
        command_intent.intent != "draft_clear"
        or command_intent.confidence < 0.85
    ):
        return False
    canonical = _canonical_draft_management_command(message_text)
    return bool(canonical is not None and canonical.intent == "draft_clear")


def _message_authorizes_keep_only(
    message_text: str,
    command_intent: MessageCommandIntent,
) -> bool:
    return _canonical_keep_only_command(message_text, command_intent) is not None


def _canonical_keep_only_command(
    message_text: str,
    command_intent: MessageCommandIntent,
) -> Optional[KeepOnlyDraftCommand]:
    """Bind every destructive keep target to the explicit trusted command clause."""
    if (
        command_intent.intent != "draft_keep_only"
        or command_intent.confidence < 0.85
        or not command_intent.keep_words
        or not message_authorizes_mutation(message_text)
    ):
        return None

    trusted_source = trusted_mutation_source(message_text)
    compact = _compact_command_text(trusted_source)
    delete_structure = re.search(
        r"(?:"
        r"保留.{0,30}(?:其余|其他).{0,8}(?:删|删除|删掉|去掉)|"
        r"除了.{1,30}(?:其余|其他|别的).{0,8}(?:删|删除|删掉|去掉)|"
        r"(?:撤销|删除|删掉|去掉).{0,12}除了"
        r")",
        compact,
    )
    if not delete_structure:
        return None
    if re.search(r"(?:不要|别(?!的)|无需|不用).{0,12}(?:删|去掉|保留)", compact):
        return None

    target_matches = list(re.finditer(
        r"(?:"
        r"(?:只|仅)?(?:保留|留下)\s*(?P<keep>[^\n，,。.!！?？;；:：]{1,80}?)"
        r"(?=\s*[，,]?\s*(?:其余|其他|别的|再?提交|提审|送审|$))|"
        r"除了\s*(?P<except>[^\n，,。.!！?？;；:：]{1,80}?)"
        r"(?:以外|之外)?(?=\s*[，,]?\s*(?:其余|其他|别的|都|再?提交|提审|送审|$))"
        r")",
        trusted_source,
    ))
    if len(target_matches) != 1:
        return None
    match = target_matches[0]
    target_region = str(match.group("keep") or match.group("except") or "").strip()
    if not target_region:
        return None

    indexed_words: List[Tuple[int, str]] = []
    remaining = target_region
    for word in sorted(command_intent.keep_words, key=len, reverse=True):
        if remaining.count(word) != 1:
            return None
        indexed_words.append((target_region.index(word), word))
        remaining = remaining.replace(word, "", 1)
    remaining = re.sub(r"(?:以及|还有|和|与|及|、|\s)+", "", remaining)
    if remaining:
        return None

    canonical_words = tuple(word for _index, word in sorted(indexed_words))
    submit_after = bool(re.search(r"(?:再|然后|并|并且)?(?:提交|提审|送审)", trusted_source))
    return KeepOnlyDraftCommand(
        keep_words=canonical_words,
        submit_after=submit_after,
    )


def _message_authorizes_replace_char(
    message_text: str,
    command_intent: MessageCommandIntent,
) -> bool:
    if (
        command_intent.intent != "batch_replace_char"
        or command_intent.confidence < 0.85
        or not command_intent.old_char
        or not command_intent.new_char
        or not message_authorizes_mutation(message_text)
    ):
        return False
    compact = _compact_command_text(message_text)
    return command_intent.old_char in compact and command_intent.new_char in compact


def _message_authorizes_pending_control(
    message_text: str,
    command_intent: MessageCommandIntent,
) -> bool:
    """Require a short explicit current reply before consuming a structured ticket."""
    if command_intent.confidence < 0.85:
        return False
    compact = _compact_command_text(message_text).lower()
    if not compact or len(compact) > 48:
        return False
    assent = parse_pending_assent_phrase(message_text)
    if command_intent.intent == "pending_confirm":
        return bool(assent.matched and not assent.add_requested)
    if command_intent.intent == "pending_cancel":
        assent = parse_pending_assent_phrase(message_text)
        return bool(
            assent.rejection == "negation"
            and assent.cancel_requested
        )
    if command_intent.intent == "pending_add_and_submit":
        return bool(
            assent.matched
            and assent.add_requested
            and assent.submit_after
        )
    if command_intent.intent == "pending_choice":
        parsed_choice = _parse_pending_choice_index(compact)
        return bool(
            parsed_choice is not None
            and command_intent.choice_index == parsed_choice
        )
    if command_intent.intent == "pending_code_request":
        code = command_intent.requested_code
        code_match = re.fullmatch(
            r"(?:(?:用|以|选|改成)|"
            r"(?:确认)?(?:加|添加)(?:用|以|选|改成)?|"
            r"确认(?:用|以|选))?"
            r"([a-z]{2,12})",
            compact,
        )
        return bool(
            code
            and code_match
            and code == code_match.group(1)
        )
    if command_intent.intent == "pending_recode":
        structure_matches = bool(
            re.fullmatch(
                r"(?:(?:第)?(?:\d{1,2}|[一二三四五六七八九十两]+)(?:个)?)?"
                r"(?:重新编码|挪开|顺延|改成)",
                compact,
            )
        )
        if not structure_matches:
            return False
        parsed_choice = _parse_pending_choice_index(
            re.sub(r"(?:重新编码|挪开|顺延|改成)$", "", compact)
        )
        if (
            command_intent.choice_index is not None
            and command_intent.choice_index != parsed_choice
        ):
            return False
        return not command_intent.target_word or command_intent.target_word in compact
    return False


_ORIGINAL_COMMAND_LINE_RE = re.compile(
    r"^(?:原始操作指令|原始指令|原指令)\s*[：:]\s*[「“\"]?(.+?)[」”\"]?$"
)

# Compatibility/rendering vocabulary; stale acceptance is parser-owned below.
_STALE_CONFIRMATION_ONLY_TEXTS = _PENDING_CONFIRM_ASSENT_TEXTS


def _is_unambiguous_stale_confirmation(message_text: str) -> bool:
    """Accept only a positive whole-message generic assent as stale control."""
    assent = parse_pending_assent_phrase(message_text)
    return bool(
        assent.matched
        and not assent.add_requested
        and not assent.submit_after
    )


def _recover_original_command_from_confirmation_quote(
    reply_reference: ReplyReferenceInfo,
) -> str:
    """Recover only an explicitly labeled original command from a bot quote."""
    if (
        not reply_reference.is_reply
        or not reply_reference.is_to_bot
        or not reply_reference.text
    ):
        return ""
    for line in reply_reference.text.splitlines():
        match = _ORIGINAL_COMMAND_LINE_RE.fullmatch(line.strip())
        if match is None:
            continue
        command = match.group(1).strip()
        if 0 < len(command) <= 160 and message_authorizes_mutation(command):
            return command
    return ""


def _format_stale_confirmation_response(
    message_text: str,
    reply_reference: ReplyReferenceInfo,
) -> Optional[str]:
    """Return no-write guidance for a standalone confirmation with no state."""
    if not _is_unambiguous_stale_confirmation(message_text):
        return None
    original_command = _recover_original_command_from_confirmation_quote(reply_reference)
    return render_remediation_reply(
        "之前等待确认的内容已经过期或不存在；本次不会执行",
        command=original_command,
    )


def _parse_pending_choice_index(text: str) -> Optional[int]:
    """Parse the small ordinal forms accepted by pending-choice commands."""
    candidate = str(text or "").strip()
    candidate = re.sub(r"^第", "", candidate)
    candidate = re.sub(r"个$", "", candidate)
    if not candidate:
        return None
    if candidate.isascii() and candidate.isdecimal():
        value = int(candidate)
        return value if value > 0 else None
    digit_map = {
        "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9,
    }
    if candidate in digit_map:
        return digit_map[candidate]
    if candidate == "十":
        return 10
    if candidate.startswith("十") and candidate[1:] in digit_map:
        return 10 + digit_map[candidate[1:]]
    if candidate.endswith("十") and candidate[:-1] in digit_map:
        return digit_map[candidate[:-1]] * 10
    if "十" in candidate:
        tens, ones = candidate.split("十", 1)
        if tens in digit_map and ones in digit_map:
            return digit_map[tens] * 10 + digit_map[ones]
    return None


_PENDING_NUMBERED_SELECTOR_PATTERN = (
    r"(?:第)?(?:[1-9]\d{0,2}|[一二两三四五六七八九十]{1,3})(?:个|号|项)?"
)


_PENDING_NUMBERED_ADD_REPLY_RE = re.compile(
    rf"^(?:编号)?(?P<selector>{_PENDING_NUMBERED_SELECTOR_PATTERN})"
    r"(?:确认)?"
    r"(?P<add>都加|全加|全部加|加词|添加|加入|加到(?:当前)?草稿|新增|创建|"
    r"写入|放入|收录|录入|记入)"
    r"(?:一下)?"
    r"(?P<submit>(?:(?:并|并且|然后|再|同时|以及))?"
    r"(?:提交|提审|送审)(?:审核|草稿|批次)?)?"
    r"(?:一下)?(?:吧|啦|了)?$",
)


def _structural_pending_add_word_intent(
    message_text: str,
    state: PendingAddWord,
) -> Optional[MessageCommandIntent]:
    """Parse exact selectors advertised by a live add-word prompt."""
    stripped = _strip_command_message_prefixes(
        trusted_mutation_source(message_text)
    ).strip()
    if (
        not stripped
        or re.search(r"[?？\"'“”‘’「」『』]", stripped)
    ):
        return None
    compact = _compact_command_text(stripped).lower()
    if not compact:
        return None

    multi_selection = parse_pending_candidate_selection(stripped)
    if multi_selection is not None:
        server_backed = bool(
            state.server_candidates
            and state.server_candidates == state.candidates
        )
        if not server_backed:
            return None
        resolved_codes = multi_selection.codes
        if multi_selection.indices and all(
            1 <= index <= len(state.candidates)
            for index in multi_selection.indices
        ):
            resolved_codes = tuple(
                state.candidates[index - 1][0]
                for index in multi_selection.indices
            )
        return MessageCommandIntent(
            intent=(
                "pending_add_and_submit"
                if multi_selection.submit_after
                else "pending_choice"
            ),
            confidence=1.0,
            submit_after=multi_selection.submit_after,
            choice_indices=multi_selection.indices,
            requested_codes=resolved_codes,
        )

    prefixed_number = re.fullmatch(
        r"(?:添加|加入|加词|加)(?P<selector>[1-9]\d{0,2})"
        r"(?P<submit>并提交)?",
        compact,
    )
    if prefixed_number is not None:
        choice_index = int(prefixed_number.group("selector"))
        submit_after = prefixed_number.group("submit") is not None
        return MessageCommandIntent(
            intent=(
                "pending_add_and_submit" if submit_after else "pending_choice"
            ),
            confidence=1.0,
            submit_after=submit_after,
            choice_index=choice_index,
        )

    numbered_add = _PENDING_NUMBERED_ADD_REPLY_RE.fullmatch(compact)
    if numbered_add is not None:
        choice_index = _parse_pending_choice_index(
            numbered_add.group("selector")
        )
        if choice_index is None:
            return None
        submit_after = numbered_add.group("submit") is not None
        return MessageCommandIntent(
            intent=(
                "pending_add_and_submit" if submit_after else "pending_choice"
            ),
            confidence=1.0,
            submit_after=submit_after,
            choice_index=choice_index,
        )

    choice_index = _parse_pending_choice_index(compact)
    if choice_index is not None:
        return MessageCommandIntent(
            intent="pending_choice",
            confidence=1.0,
            choice_index=choice_index,
        )

    code_matches: List[MessageCommandIntent] = []
    for candidate_code, _occupied in state.candidates:
        candidate_intent = MessageCommandIntent(
            intent="pending_code_request",
            confidence=1.0,
            requested_code=str(candidate_code or "").strip().lower(),
        )
        if (
            candidate_intent.requested_code
            and _message_authorizes_pending_control(stripped, candidate_intent)
        ):
            code_matches.append(candidate_intent)
    if len(code_matches) == 1:
        return code_matches[0]

    recode_match = re.fullmatch(
        r"(.+?)(重新编码|挪开|顺延|改成)",
        compact,
    )
    if recode_match is None:
        return None
    selector = recode_match.group(1)
    choice_index = _parse_pending_choice_index(selector)
    if choice_index is not None:
        return MessageCommandIntent(
            intent="pending_recode",
            confidence=1.0,
            choice_index=choice_index,
        )

    matching_codes = {
        code
        for code, occupied in state.candidates
        if occupied and selector in state.occupied_words.get(code, [])
    }
    if len(matching_codes) != 1:
        return None
    return MessageCommandIntent(
        intent="pending_recode",
        confidence=1.0,
        target_word=selector,
    )


def message_authorizes_live_pending_mutation(
    message_text: str,
    state: PendingAddWord,
) -> bool:
    """Treat a closed selector as write intent only for one live server record."""
    if not (
        isinstance(state, PendingAddWord)
        and state.server_candidates
        and state.server_candidates == state.candidates
    ):
        return False
    intent = _structural_pending_add_word_intent(message_text, state)
    if intent is None or intent.intent not in {
        "pending_confirm",
        "pending_add_and_submit",
        "pending_recode",
        "pending_code_request",
        "pending_choice",
    }:
        return False
    if intent.intent == "pending_choice" and intent.choice_index is not None:
        return 1 <= intent.choice_index <= len(state.server_candidates)
    return _message_authorizes_pending_control(message_text, intent)


def _closed_candidate_selection(
    text: str,
) -> Optional[Tuple[Tuple[int, ...], Tuple[str, ...], bool]]:
    """Parse the exact numbered/code selection forms advertised in discovery."""
    parsed = parse_pending_candidate_selection(text)
    if parsed is not None:
        return parsed.indices, parsed.codes, parsed.submit_after
    compact = _compact_command_text(text).lower()
    match = re.fullmatch(
        r"(?:(?:添加|加入|加词|加)?(?P<prefix_index>[1-9]\d{0,2})"
        r"(?P<prefix_submit>并提交)?|"
        r"(?P<suffix_index>[1-9]\d{0,2})(?:添加|加入|加词|加)"
        r"(?P<suffix_submit>并提交)?)",
        compact,
    )
    if match is None:
        return None
    index = match.group("prefix_index") or match.group("suffix_index")
    return (
        (int(index),),
        (),
        bool(match.group("prefix_submit") or match.group("suffix_submit")),
    )


def _multi_word_candidate_scope_rows(
    state: PendingState,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Tuple[str, bool]]]]:
    """Validate the structured candidate metadata carried by a batch ticket."""
    if (
        not isinstance(state, PendingToolConfirm)
        or state.function_name != "keytao_batch_add_to_draft"
    ):
        return [], {}
    items = state.args.get("items")
    if not isinstance(items, list):
        return [], {}
    clean_items = [item for item in items if isinstance(item, dict)]
    words = [str(item.get("word") or "").strip() for item in clean_items]
    if (
        len(clean_items) != len(items)
        or len(words) < 2
        or any(not word for word in words)
        or len(set(words)) != len(words)
    ):
        return [], {}
    if any(
        str(item.get("action") or "Create") != "Create"
        or item.get("old_word")
        or item.get("oldWord")
        for item in clean_items
    ):
        # This parser is only for choosing among advertised add candidates.
        # A batch of Change operations (for example a sealed chain reorder)
        # must reach its own exact-set validator unchanged.
        return [], {}

    raw_scopes = state.args.get("_candidate_scopes")
    if not isinstance(raw_scopes, list):
        return clean_items, {}
    scopes: Dict[str, List[Tuple[str, bool]]] = {}
    for raw_scope in raw_scopes:
        if not isinstance(raw_scope, dict):
            return clean_items, {}
        word = str(raw_scope.get("word") or "").strip()
        raw_candidates = raw_scope.get("candidates")
        if word not in words or word in scopes or not isinstance(raw_candidates, list):
            return clean_items, {}
        candidates: List[Tuple[str, bool]] = []
        seen_codes: set[str] = set()
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, (list, tuple)) or len(raw_candidate) != 2:
                return clean_items, {}
            code = str(raw_candidate[0] or "").strip().lower()
            occupied = raw_candidate[1]
            if (
                not re.fullmatch(r"[a-z]{1,12}", code)
                or code in seen_codes
                or not isinstance(occupied, bool)
            ):
                return clean_items, {}
            seen_codes.add(code)
            candidates.append((code, occupied))
        if not candidates:
            return clean_items, {}
        current_code = str(
            next(item for item in clean_items if str(item.get("word") or "").strip() == word).get("code")
            or ""
        ).strip().lower()
        if current_code not in seen_codes:
            return clean_items, {}
        scopes[word] = candidates
    if set(scopes) != set(words):
        return clean_items, {}
    return clean_items, scopes


def _resolve_multi_word_pending_candidate_selection(
    state: PendingState,
    message_text: str,
) -> Tuple[Optional[PendingToolConfirm], Optional[MessageCommandIntent], Optional[str]]:
    """Require word scope when several live candidate lists reuse numbers."""
    items, scopes = _multi_word_candidate_scope_rows(state)
    if len(items) < 2:
        return None, None, None
    words = tuple(str(item["word"]).strip() for item in items)
    unwrapped = _whole_message_unquoted_source(message_text, words)
    if unwrapped is None and _has_rejected_advertised_command_envelope(
        message_text,
        words,
    ):
        return None, None, render_remediation_reply(
            "命令引号外含有其他内容，本次未写入；当前确认仍有效",
            command=f"将这 {len(words)} 个词加入草稿",
            words=words,
        )
    set_reference = parse_advertised_set_reference(unwrapped or message_text)
    if set_reference.matched:
        if (
            set_reference.expected_count is not None
            and set_reference.expected_count != len(words)
        ):
            return None, None, render_remediation_reply(
                f"当前有效候选共有 {len(words)} 个，不是指令中的 "
                f"{set_reference.expected_count} 个；只能从「"
                + "、".join(words)
                + "」中选择；本次未写入",
                command=f"将这 {len(words)} 个词加入草稿",
                words=words,
            )
        unknown = tuple(
            word for word in set_reference.exclusions if word not in words
        )
        if unknown:
            return None, None, render_remediation_reply(
                "以下暂不添加的词不在当前确认范围中："
                + "、".join(f"「{word}」" for word in unknown)
                + "。当前有效候选为「"
                + "、".join(words)
                + "」；本次未写入",
                command=f"将这 {len(words)} 个词加入草稿",
                words=words,
            )
        resolved_words = tuple(
            word for word in words if word not in set_reference.exclusions
        )
        if not resolved_words:
            return None, None, render_remediation_reply(
                "按这些排除项计算后没有剩余词；当前有效候选为「"
                + "、".join(words)
                + "」；本次未写入",
                command=f"将这 {len(words)} 个词加入草稿",
                words=words,
            )
        resolved_set = set(resolved_words)
        derived_args = dict(state.args)
        derived_args["items"] = [
            dict(item)
            for item in items
            if str(item.get("word") or "").strip() in resolved_set
        ]
        derived_args["_resolved_advertised_words"] = list(resolved_words)
        raw_scopes = derived_args.get("_candidate_scopes")
        if isinstance(raw_scopes, list):
            derived_args["_candidate_scopes"] = [
                dict(scope)
                for scope in raw_scopes
                if isinstance(scope, dict)
                and str(scope.get("word") or "").strip() in resolved_set
            ]
        derived_state = PendingToolConfirm(
            function_name=state.function_name,
            args=derived_args,
            confirmation_source=state.confirmation_source,
        )
        intent = MessageCommandIntent(
            intent=(
                "pending_add_and_submit"
                if set_reference.submit_after
                else "pending_confirm"
            ),
            confidence=1.0,
            submit_after=set_reference.submit_after,
        )
        return derived_state, intent, None
    source = unicodedata.normalize(
        "NFKC",
        _strip_command_message_prefixes(trusted_mutation_source(message_text)),
    ).strip()
    if (
        not source
        or re.search(r"[?？\"'“”‘’「」『』]", source)
        or re.search(r"(?:不要|别|取消|删除|移除|解释|复述|他说)", source)
    ):
        return None, None, None

    target_word = ""
    selection_text = ""
    for word in sorted(words, key=len, reverse=True):
        match = re.fullmatch(rf"{re.escape(word)}[\s:：，,]*(.+)", source)
        if match is not None:
            target_word = word
            selection_text = match.group(1).strip()
            break

    selection = _closed_candidate_selection(selection_text or source)
    unscoped_recode = bool(
        not target_word
        and re.fullmatch(r"(?:第)?[1-9]\d{0,2}(?:个|号|项)?重新编码", _compact_command_text(source))
    )
    if not target_word and (selection is not None or unscoped_recode):
        named_words = "、".join(f"「{word}」" for word in words)
        example = words[-1]
        first = render_executable_suggestion(
            f"{example} 添加1",
            words=(example,),
        )
        multiple = render_executable_suggestion(
            f"{example} 添加2、4",
            words=(example,),
        )
        return None, None, (
            f"当前有多个词（{named_words}），每个候选列表都从 1 开始。"
            "可执行选择示例：\n"
            + "\n".join(item for item in (first, multiple) if item)
        )
    if not target_word or selection is None:
        return None, None, None
    if not scopes:
        return None, None, render_remediation_reply(
            "当前多词候选缺少编号，本次未写入",
            command="加词 " + " ".join(words),
            words=words,
        )

    indices, requested_codes, submit_after = selection
    candidates = scopes[target_word]
    if indices:
        if (
            len(set(indices)) != len(indices)
            or any(not 1 <= index <= len(candidates) for index in indices)
        ):
            return None, None, render_remediation_reply(
                f"「{target_word}」只接受 1-{len(candidates)} 之间的编号",
                command=f"{target_word} 添加1",
                words=(target_word,),
            )
        selected_codes = tuple(candidates[index - 1][0] for index in indices)
    else:
        candidate_codes = {code for code, _occupied in candidates}
        if (
            not requested_codes
            or len(set(requested_codes)) != len(requested_codes)
            or any(code not in candidate_codes for code in requested_codes)
        ):
            return None, None, render_remediation_reply(
                f"所选编码不全在「{target_word}」当前候选中",
                command=f"{target_word} 添加1",
                words=(target_word,),
            )
        selected_codes = requested_codes

    target_item = next(
        item
        for item in items
        if str(item.get("word") or "").strip() == target_word
    )
    occupancy = dict(candidates)
    selected_items: List[Dict[str, Any]] = []
    for code in selected_codes:
        item = dict(target_item)
        item["code"] = code
        if occupancy[code]:
            item["needsManualReview"] = True
            item["manualReviewReason"] = "重码添加需管理员审核"
        selected_items.append(item)

    derived_items: List[Dict[str, Any]] = []
    for item in items:
        if str(item.get("word") or "").strip() == target_word:
            derived_items.extend(selected_items)
        else:
            derived_items.append(dict(item))
    derived_args = dict(state.args)
    derived_args.pop("_candidate_scopes", None)
    derived_args["items"] = derived_items
    derived_state = PendingToolConfirm(
        function_name=state.function_name,
        args=derived_args,
        confirmation_source=state.confirmation_source,
    )
    intent = MessageCommandIntent(
        intent="pending_add_and_submit" if submit_after else "pending_confirm",
        confidence=1.0,
        submit_after=submit_after,
        choice_indices=indices,
        requested_codes=selected_codes,
        target_word=target_word,
    )
    return derived_state, intent, None


def _is_fresh_current_user_command_intent(
    command_intent: MessageCommandIntent,
    message_text: str = "",
) -> bool:
    if explicit_complete_add_item(message_text) is not None:
        return True
    if command_intent.intent == "draft_submit":
        return _is_explicit_draft_submit_request(message_text)
    if command_intent.intent == "clear_history":
        return _message_authorizes_clear_history(message_text, command_intent)
    if command_intent.intent == "draft_keep_only":
        return _message_authorizes_keep_only(message_text, command_intent)
    if command_intent.intent == "draft_recall":
        return _message_authorizes_draft_recall(message_text, command_intent)
    if command_intent.intent == "draft_clear":
        return _message_authorizes_draft_clear(message_text, command_intent)
    if command_intent.intent == "batch_replace_char":
        return _message_authorizes_replace_char(message_text, command_intent)
    if command_intent.intent == "code_chain_reorder":
        command = _parse_code_chain_reorder_command(message_text)
        return bool(
            command is not None
            and command.code == command_intent.requested_code
        )
    if command_intent.intent == "word_list_reorder":
        command = _parse_word_list_reorder_command(message_text)
        return bool(
            command is not None
            and command.words == command_intent.keep_words
        )
    if command_intent.intent == "entry_move_plan":
        command = parse_entry_move_plan(message_text)
        return bool(
            command is not None
            and tuple(move.word for move in command.moves)
            == command_intent.keep_words
            and tuple(move.target_code for move in command.moves)
            == command_intent.requested_codes
        )
    if command_intent.intent == "dictionary_delete":
        command = parse_dictionary_delete_command(message_text)
        return bool(
            command is not None
            and (command.word,) == command_intent.keep_words
            and command.code == command_intent.requested_code
        )
    if command_intent.intent == "entry_swap":
        command = parse_entry_swap(message_text)
        return bool(
            command is not None
            and command_intent.keep_words
            == (command.first_word, command.second_word)
        )
    return command_intent.intent in {
        "draft_view",
        "operation_recall",
    }


def _is_prefixed_fresh_word_query(message_text: str, normalized_message_text: str) -> bool:
    raw = message_text.strip()
    normalized = normalized_message_text.strip()
    if not raw or not normalized or raw == normalized:
        return False
    words = _extract_pure_chinese_words(normalized)
    if not words:
        return False
    if (
        parse_pending_assent_phrase(normalized).recognized
        or _is_explicit_draft_submit_request(normalized)
        or _canonical_draft_management_command(normalized) is not None
    ):
        return False
    return True


def _should_block_for_other_owner_pending(
    space_type: str,
    has_current_pending: bool,
    other_pending_record: Optional[PendingStateRecord],
    generic_command_intent: MessageCommandIntent,
    other_pending_command_intent: MessageCommandIntent,
    message_text: str = "",
    current_contextual_reply: bool = False,
) -> bool:
    return (
        space_type == "group"
        and other_pending_record is not None
        and not has_current_pending
        and not current_contextual_reply
        and not _is_fresh_current_user_command_intent(generic_command_intent, message_text)
        and _is_sensitive_pending_control_intent(other_pending_command_intent)
    )


def _extract_pure_chinese_words(message_text: str) -> List[str]:
    """Extract structurally simple Chinese tokens without deciding intent."""
    text = message_text.strip()
    if not text or not _PURE_CHINESE_WORDS_RE.fullmatch(text):
        return []
    return [token for token in re.split(r'[\s、，,；;]+', text) if token]


def _load_json_object_from_model_text(content: str) -> Dict:
    """Parse the first JSON object from a model response."""
    text = (content or "").strip()
    if not text:
        return {}
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
        except Exception:
            return {}
    return value if isinstance(value, dict) else {}


def _sanitize_simple_word_intent_words(words: object, fallback_words: Tuple[str, ...]) -> Tuple[str, ...]:
    """Keep only clean Chinese tokens returned by the intent classifier."""
    if not isinstance(words, list):
        words = []
    sanitized = []
    for word in words:
        token = str(word or "").strip()
        if token and _PURE_CHINESE_TOKEN_RE.fullmatch(token):
            sanitized.append(token)
    if sanitized:
        return tuple(dict.fromkeys(sanitized))
    return fallback_words


def _parse_simple_word_query_intent_payload(
    payload: Dict,
    fallback_words: Tuple[str, ...],
) -> SimpleWordQueryIntent:
    """Normalize model JSON into a simple word-query intent decision."""
    intent = str(payload.get("intent") or "").strip().lower()
    should_handle = intent == "word_lookup" or payload.get("should_handle") is True
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    words = _sanitize_simple_word_intent_words(payload.get("words"), fallback_words)
    if not should_handle:
        return SimpleWordQueryIntent(
            should_handle=False,
            words=(),
            intent=intent or "not_word_lookup",
            confidence=confidence,
        )
    return SimpleWordQueryIntent(
        should_handle=True,
        words=words,
        intent=intent or "word_lookup",
        confidence=confidence,
    )


def _sanitize_optional_code(value: object) -> str:
    code = str(value or "").strip().lower()
    return code if _CODE_TOKEN_RE.fullmatch(code) else ""


def _sanitize_optional_codes(value: object) -> Tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    codes = (
        _sanitize_optional_code(item)
        for item in value
    )
    return tuple(dict.fromkeys(code for code in codes if code))


def _sanitize_optional_positive_int(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value if value > 0 else None
    text = str(value or "").strip()
    if not text.isdigit():
        return None
    number = int(text)
    return number if number > 0 else None


def _sanitize_optional_single_char(value: object) -> str:
    text = str(value or "").strip()
    return text if len(text) == 1 and not text.isspace() else ""


def _sanitize_optional_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0", ""}:
        return False
    return False


def _sanitize_command_words(words: object) -> Tuple[str, ...]:
    if not isinstance(words, list):
        return ()
    result = []
    for word in words:
        token = str(word or "").strip()
        if token and _PURE_CHINESE_TOKEN_RE.fullmatch(token):
            result.append(token)
    return tuple(dict.fromkeys(result))


def _parse_message_command_intent_payload(payload: Dict) -> MessageCommandIntent:
    """Normalize model JSON into command-routing metadata."""
    allowed_intents = {
        "none",
        "clear_history",
        "draft_submit",
        "draft_view",
        "draft_recall",
        "draft_clear",
        "draft_keep_only",
        "operation_recall",
        "batch_replace_char",
        "code_chain_reorder",
        "word_list_reorder",
        "pending_confirm",
        "pending_cancel",
        "pending_add_and_submit",
        "pending_recode",
        "pending_code_request",
        "pending_choice",
    }
    intent = str(payload.get("intent") or "none").strip().lower()
    if intent not in allowed_intents:
        intent = "none"
    try:
        confidence = float(payload.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return MessageCommandIntent(
        intent=intent,
        confidence=confidence,
        keep_words=_sanitize_command_words(payload.get("keep_words")),
        submit_after=_sanitize_optional_bool(payload.get("submit_after")),
        clear_after=_sanitize_optional_bool(payload.get("clear_after")),
        current_user_only=_sanitize_optional_bool(payload.get("current_user_only")),
        choice_index=_sanitize_optional_positive_int(payload.get("choice_index")),
        requested_code=_sanitize_optional_code(payload.get("requested_code")),
        requested_codes=_sanitize_optional_codes(payload.get("requested_codes")),
        target_word=str(payload.get("target_word") or "").strip(),
        old_char=_sanitize_optional_single_char(payload.get("old_char")),
        new_char=_sanitize_optional_single_char(payload.get("new_char")),
    )


def _pending_context_for_command_intent(state: Optional[PendingState]) -> str:
    if isinstance(state, PendingAddWord):
        candidates = [
            {
                "index": index,
                "code": code,
                "occupied": occupied,
                "occupied_words": state.occupied_words.get(code, []),
            }
            for index, (code, occupied) in enumerate(state.candidates, start=1)
        ]
        return json.dumps(
            {
                "type": "pending_add_word",
                "word": state.word,
                "recommended_code": state.recommended_code,
                "candidates": candidates,
                "pronunciation_codes": state.pronunciation_codes,
                "pronunciation_recommended_codes": state.pronunciation_recommended_codes,
            },
            ensure_ascii=False,
        )
    if isinstance(state, PendingToolConfirm):
        return json.dumps(
            {
                "type": "pending_tool_confirm",
                "function_name": state.function_name,
                "args": {
                    key: value
                    for key, value in state.args.items()
                    if not str(key).startswith("_")
                },
            },
            ensure_ascii=False,
        )
    return "none"


def _structural_draft_management_intent(
    message_text: str,
) -> Optional[MessageCommandIntent]:
    """Keep the safest common recall/clear commands independent of the LLM."""
    word_list_reorder = _parse_word_list_reorder_command(message_text)
    if word_list_reorder is not None:
        return MessageCommandIntent(
            intent="word_list_reorder",
            confidence=1.0,
            keep_words=word_list_reorder.words,
        )
    entry_move_plan = parse_entry_move_plan(message_text)
    if entry_move_plan is not None:
        return MessageCommandIntent(
            intent="entry_move_plan",
            confidence=1.0,
            keep_words=tuple(move.word for move in entry_move_plan.moves),
            requested_codes=tuple(
                move.target_code for move in entry_move_plan.moves
            ),
        )
    dictionary_delete = parse_dictionary_delete_command(message_text)
    if dictionary_delete is not None:
        return MessageCommandIntent(
            intent="dictionary_delete",
            confidence=1.0,
            keep_words=(dictionary_delete.word,),
            requested_code=dictionary_delete.code,
        )
    swap = parse_entry_swap(message_text)
    if swap is not None:
        return MessageCommandIntent(
            intent="entry_swap",
            confidence=1.0,
            keep_words=(swap.first_word, swap.second_word),
        )
    chain_reorder = _parse_code_chain_reorder_command(message_text)
    if chain_reorder is not None:
        return MessageCommandIntent(
            intent="code_chain_reorder",
            confidence=1.0,
            requested_code=chain_reorder.code,
        )
    return _canonical_draft_management_command(message_text)


async def _classify_message_command_intent(
    message_text: str,
    pending_state: Optional[PendingState] = None,
) -> MessageCommandIntent:
    """Use the configured flash/intent model for command and pending-control semantics."""
    if not message_text.strip():
        return MessageCommandIntent()
    pending_positional_add = _pending_positional_add_intent(
        pending_state,
        message_text,
    )
    if pending_positional_add is not None:
        return pending_positional_add
    pending_tool_assent = _pending_tool_assent_intent(
        pending_state,
        message_text,
    )
    if pending_tool_assent is not None:
        return pending_tool_assent
    if (
        _is_explicit_draft_submit_request(message_text)
        and not _is_pending_assent_then_submit_request(message_text)
    ):
        if (
            isinstance(pending_state, PendingToolConfirm)
            and pending_state.function_name == "keytao_submit_batch"
        ):
            return MessageCommandIntent(intent="pending_confirm", confidence=1.0)
        return MessageCommandIntent(intent="draft_submit", confidence=1.0)
    compact_message = re.sub(
        r"[\s，,。.!！?？~～]+",
        "",
        _strip_command_message_prefixes(message_text),
    )
    if (
        isinstance(pending_state, PendingAddWord)
        and _is_target_bound_add_and_submit_request(message_text, pending_state)
    ):
        return MessageCommandIntent(intent="pending_add_and_submit", confidence=1.0)
    if isinstance(pending_state, PendingAddWord):
        # Keep intent recognition separate from authorization. A historical
        # candidate quote can carry a clear control shape, but only a current
        # server-backed state may pass the pending-state authorization gate.
        assent = _pending_assent_phrase_for_state(pending_state, message_text)
        if assent.matched:
            return MessageCommandIntent(
                intent=(
                    "pending_add_and_submit"
                    if assent.submit_after
                    else "pending_confirm"
                ),
                confidence=1.0,
                submit_after=assent.submit_after,
            )
        structural_pending_intent = _structural_pending_add_word_intent(
            message_text,
            pending_state,
        )
        if structural_pending_intent is not None:
            return structural_pending_intent
    if (
        isinstance(pending_state, PendingToolConfirm)
        and _pending_tool_confirmation_matches(pending_state, message_text)
    ):
        return MessageCommandIntent(intent="pending_confirm", confidence=1.0)
    if pending_state is not None and compact_message.startswith("确认票据"):
        # The retired code form is handled deterministically as non-executable.
        return MessageCommandIntent()
    structural_draft_intent = _structural_draft_management_intent(message_text)
    if structural_draft_intent is not None:
        return structural_draft_intent
    if not OPENAI_API_KEY or not AsyncOpenAI:
        logger.warning("Command intent model unavailable; falling through to main AI flow")
        return MessageCommandIntent()

    pending_context = _pending_context_for_command_intent(pending_state)
    system_prompt = (
        "你是键道机器人喵喵的轻量语义路由器。"
        "只判断当前消息是否应由程序快捷处理；不要执行操作，不要回答用户。\n"
        "输出必须是 JSON 对象，不要解释。\n"
        "intent 只能是：none, clear_history, draft_submit, draft_view, draft_keep_only, "
        "draft_recall, draft_clear, operation_recall, batch_replace_char, "
        "pending_confirm, pending_cancel, pending_add_and_submit, pending_recode, "
        "pending_code_request, pending_choice。\n"
        "clear_history：用户明确要求清空/重置本轮聊天历史。\n"
        "draft_submit：用户明确要求提交/提审自己的当前草稿。\n"
        "draft_view：用户要查看自己当前草稿。\n"
        "draft_recall：用户明确要求撤回/撤销自己最近一次提交或提审；"
        "如果还要求把撤回后恢复的草稿全部清空，则 clear_after=true。\n"
        "draft_clear：用户明确要求清空自己当前草稿中的全部条目。\n"
        "draft_keep_only：用户要在自己草稿里只保留指定词，keep_words 必须列出保留词；"
        "如果语义还要求随后提交，则 submit_after=true。\n"
        "operation_recall：用户询问最近通过喵喵经手的词库操作；"
        "如果只问自己，则 current_user_only=true。\n"
        "batch_replace_char：用户要求把下方词码列表里的某个字符批量替换成另一个字符；"
        "old_char/new_char 必须是单个字符。\n"
        "pending_add_and_submit：当前待确认的是加词或批量加词，且用户要求先加入再立即提交；"
        "例如“加入并提交”。这种情况绝不能归类为 draft_submit。\n"
        "pending_* 只在 pending_context 不是 none，且用户在回应该待确认操作时使用。"
        "普通提问、词义/常用度比较、泛泛讨论、如何使用功能、以及新的复杂操作都返回 none，交给主模型。"
    )
    user_prompt = (
        f"当前消息：{message_text}\n"
        f"pending_context：{pending_context}\n"
        "请只返回 JSON，字段包括：intent, confidence, keep_words, submit_after, "
        "clear_after, current_user_only, choice_index, requested_code, target_word, old_char, new_char。\n"
        "例如："
        '{"intent":"none","confidence":0.9,"keep_words":[],"submit_after":false,'
        '"clear_after":false,"current_user_only":false,"choice_index":null,"requested_code":"",'
        '"target_word":"","old_char":"","new_char":""}'
    )

    try:
        client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=min(OPENAI_TIMEOUT, 20.0),
            max_retries=1,
        )
        response = await observe_model_call(client.chat.completions.create(**with_deepseek_chat_policy(
            {
                "model": WORD_QUERY_INTENT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 260,
                "temperature": 0.0,
            },
            thinking=False,
            json_output=True,
        )), system_prompt_chars=len(system_prompt))
        log_chat_usage(
            logger,
            response,
            operation="command_intent",
            model=WORD_QUERY_INTENT_MODEL,
        )
        if not response.choices:
            return MessageCommandIntent()
        payload = _load_json_object_from_model_text(response.choices[0].message.content or "")
        return _parse_message_command_intent_payload(payload)
    except Exception as error:
        logger.warning(f"Failed to classify command intent: {error}")
        return MessageCommandIntent()


async def _classify_simple_word_query_intent(
    message_text: str,
    structural_words: Tuple[str, ...],
) -> SimpleWordQueryIntent:
    """Use the configured flash/intent model to decide whether this is a bare word lookup."""
    if not structural_words:
        return SimpleWordQueryIntent(False)
    if not OPENAI_API_KEY or not AsyncOpenAI:
        logger.warning("Word-query intent model unavailable; falling through to main AI flow")
        return SimpleWordQueryIntent(False)

    system_prompt = (
        "你是键道机器人喵喵的轻量语义路由器。"
        "只判断当前消息是否应该进入“裸词查词/编码”快捷流程。\n"
        "输出必须是 JSON 对象，不要解释。\n"
        "字段：intent 为 word_lookup 或 not_word_lookup；"
        "words 为应查询的词语数组；confidence 为 0 到 1。\n"
        "word_lookup 仅表示用户只给出一个或多个独立中文词、短语、成语或专名，"
        "希望了解词义、键道编码、词库位置或候选顺序。\n"
        "not_word_lookup 表示自然句、问答、比较、解释、闲聊、命令、草稿操作、确认操作、"
        "或任何需要由主对话模型理解后再决定工具调用的请求。"
    )
    user_prompt = (
        f"当前消息：{message_text}\n"
        f"结构切分候选：{json.dumps(list(structural_words), ensure_ascii=False)}\n"
        "请只返回 JSON，例如："
        '{"intent":"word_lookup","words":["示例词"],"confidence":0.9}'
    )

    try:
        client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=min(OPENAI_TIMEOUT, 20.0),
            max_retries=1,
        )
        response = await observe_model_call(client.chat.completions.create(**with_deepseek_chat_policy(
            {
                "model": WORD_QUERY_INTENT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 180,
                "temperature": 0.0,
            },
            thinking=False,
            json_output=True,
        )), system_prompt_chars=len(system_prompt))
        log_chat_usage(
            logger,
            response,
            operation="word_query_intent",
            model=WORD_QUERY_INTENT_MODEL,
        )
        if not response.choices:
            return SimpleWordQueryIntent(False)
        content = response.choices[0].message.content or ""
        payload = _load_json_object_from_model_text(content)
        return _parse_simple_word_query_intent_payload(payload, structural_words)
    except Exception as error:
        logger.warning(f"Failed to classify simple word-query intent: {error}")
        return SimpleWordQueryIntent(False)


async def _get_simple_word_query_words(message_text: str) -> Tuple[str, ...]:
    """Return model-approved word-query targets, or empty when the main AI should handle it."""
    structural_words = tuple(_extract_pure_chinese_words(message_text))
    if not structural_words:
        return ()
    intent = await _classify_simple_word_query_intent(message_text, structural_words)
    if not intent.should_handle:
        logger.info(
            "Simple Chinese message fell through to main AI: "
            f"intent={intent.intent} confidence={intent.confidence:.2f}"
        )
        return ()
    return intent.words


def command_intent_memoizer(
    message_text: str,
    normalized_message_text: str,
) -> Tuple[
    Dict[Tuple[str, str], MessageCommandIntent],
    Callable[[Optional[PendingState]], Awaitable[MessageCommandIntent]],
]:
    """Build the per-turn command-intent cache used by the chat pipeline."""
    command_intent_cache: Dict[Tuple[str, str], MessageCommandIntent] = {}

    async def command_intent_for(
        pending_state: Optional[PendingState] = None,
    ) -> MessageCommandIntent:
        if not message_text:
            return MessageCommandIntent()
        cache_key = (
            pending_state.__class__.__name__ if pending_state is not None else "none",
            _describe_pending_state(pending_state) if pending_state is not None else "",
        )
        structural_tool_intent = _pending_tool_assent_intent(
            pending_state,
            normalized_message_text,
        )
        if structural_tool_intent is not None:
            command_intent_cache[cache_key] = structural_tool_intent
            return structural_tool_intent
        if cache_key not in command_intent_cache:
            classified = await _classify_message_command_intent(
                normalized_message_text,
                pending_state,
            )
            if (
                pending_state is not None
                and _is_sensitive_pending_control_intent(classified)
                and not _message_authorizes_pending_state_control(
                    pending_state,
                    normalized_message_text,
                    classified,
                )
            ):
                classified = MessageCommandIntent()
            command_intent_cache[cache_key] = classified
        return command_intent_cache[cache_key]

    return command_intent_cache, command_intent_for


_EXPLICIT_REVIEWED_ADD_WORD_RE = re.compile(
    r"^(?:请|麻烦)?\s*(?:帮我|帮忙|给我)?\s*"
    r"(?:加词|添加词|新增词|加一个词|添加一个词)"
    r"\s*[:：,，]?\s*(?P<word>[\u3400-\u9fff]{1,20})$"
)


def _extract_explicit_reviewed_add_word(message_text: str) -> Optional[str]:
    """Return the target word for a structural `加词 X` request."""
    text = _strip_command_message_prefixes(message_text)
    text = re.sub(r"\s+", " ", text).strip()
    match = _EXPLICIT_REVIEWED_ADD_WORD_RE.fullmatch(text)
    if not match:
        return None
    return match.group("word").strip()


def _should_augment_simple_word_query(message_text: str, response: str) -> bool:
    """Skip query augmentation for confirmations and action-result replies."""
    text = message_text.strip()
    if not text:
        return False

    response_text = response.strip()
    action_markers = (
        "加入草稿",
        "当前草稿",
        "发送「提交」",
        "发送“提交”",
        "批次已提交审核",
        "草稿已成功提交审核",
        "已提交审核",
        "撤回成功",
        "草稿已恢复",
        "已从草稿删除",
        "删除成功",
        "diff Phrase",
        "草稿地址：",
        "批次地址：",
        "✅ 已将",
        "✅ 已写入草稿",
        "✅ 已只保留",
        "✅ 草稿里已经只保留",
        "插入编码",
        "调整到编码",
        "拟执行 ",
        "待确认",
        "安全拦截",
        "尚未写入",
    )
    return not any(marker in response_text for marker in action_markers)


def _referenced_owner_key_from_reply_reference(
    reply_reference: ReplyReferenceInfo,
    platform: str,
) -> Optional[Tuple[str, str]]:
    """Return the user explicitly mentioned by a quoted bot prompt."""
    for mentioned_user_id in reply_reference.mentioned_user_ids:
        owner_id = str(mentioned_user_id or "").strip()
        if owner_id and owner_id.lower() != "all":
            return (platform, owner_id)
    return None


def _pending_owner_label(record: PendingStateRecord) -> str:
    owner_id = str(record.owner_key.actor_id or "").strip()
    owner_label = str(record.owner_label or "").strip()
    if owner_label and owner_label != owner_id:
        return owner_label
    return "这位用户"


def _describe_pending_state(state: PendingState) -> str:
    return _format_pending_state_details(state)


_DIRECT_OWNER_PENDING_ADD_INTENTS = {
    "pending_confirm",
    "pending_add_and_submit",
    "pending_recode",
    "pending_code_request",
    "pending_choice",
}


def _pending_tool_confirmation_matches(
    _state: PendingToolConfirm,
    _message_text: str,
) -> bool:
    """Retired: target-bearing confirmation commands are no longer accepted."""
    return False


def _message_authorizes_pending_state_control(
    state: PendingState,
    message_text: str,
    command_intent: MessageCommandIntent,
) -> bool:
    """Allow a generic short control or an exact target-bound local preview."""
    if re.search(r"[?？]", message_text):
        return False
    positional_intent = _pending_positional_add_intent(state, message_text)
    if positional_intent == command_intent:
        return True
    structural_tool_intent = _pending_tool_assent_intent(state, message_text)
    if (
        structural_tool_intent is not None
        and structural_tool_intent.intent == command_intent.intent
    ):
        return True
    if (
        isinstance(state, PendingAddWord)
        and command_intent.intent == "pending_add_and_submit"
        and _is_target_bound_add_and_submit_request(message_text, state)
    ):
        return True
    if isinstance(state, PendingAddWord):
        structural_intent = _structural_pending_add_word_intent(
            message_text,
            state,
        )
        structural_recode_target = (
            _resolve_shift_target_code(state, structural_intent)
            if structural_intent is not None
            and structural_intent.intent == "pending_recode"
            else None
        )
        command_recode_target = (
            _resolve_shift_target_code(state, command_intent)
            if command_intent.intent == "pending_recode"
            else None
        )
        recode_targets_match = bool(
            structural_recode_target
            and structural_recode_target == command_recode_target
        )
        if (
            recode_targets_match
            or (
                structural_intent is not None
                and structural_intent.intent == command_intent.intent
                and structural_intent.choice_index == command_intent.choice_index
                and structural_intent.choice_indices == command_intent.choice_indices
                and structural_intent.requested_code == command_intent.requested_code
                and structural_intent.requested_codes == command_intent.requested_codes
                and structural_intent.target_word == command_intent.target_word
            )
        ):
            return True
    if _message_authorizes_pending_control(message_text, command_intent):
        return True
    return False


def _ticket_payload_from_command_intent(
    command_intent: MessageCommandIntent,
) -> Dict[str, object]:
    """Persist a validated choice without asking a model to reinterpret it later."""
    return {
        "intent": command_intent.intent,
        "confidence": 1.0,
        "submit_after": bool(command_intent.submit_after),
        "choice_index": command_intent.choice_index,
        "choice_indices": list(command_intent.choice_indices),
        "requested_code": command_intent.requested_code,
        "requested_codes": list(command_intent.requested_codes),
        "target_word": command_intent.target_word,
    }


def _describe_pending_ticket_choice(
    state: PendingState,
    command_intent: MessageCommandIntent,
) -> str:
    if isinstance(state, PendingAddWord):
        target_code = state.recommended_code
        if command_intent.requested_codes:
            target_code = "、".join(command_intent.requested_codes)
        if (
            not command_intent.requested_codes
            and
            command_intent.choice_index is not None
            and 1 <= command_intent.choice_index <= len(state.candidates)
        ):
            target_code = state.candidates[command_intent.choice_index - 1][0]
        elif command_intent.intent == "pending_code_request":
            target_code = command_intent.requested_code or target_code
        action = "加入并提交" if command_intent.intent == "pending_add_and_submit" else "加词"
        if command_intent.intent == "pending_recode":
            action = "重新编码后加词"
        return f"{action}「{state.word}」→ {target_code}"
    return _describe_pending_state(state)


def _prompt_capability_digest(text: str) -> str:
    normalized = re.sub(r"^\s*@\S+\s*", "", str(text or ""), count=1)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _verified_bot_reply_matches_record(
    reply_reference: ReplyReferenceInfo,
    record: Optional[PendingStateRecord],
) -> bool:
    return bool(
        record is not None
        and reply_reference.is_reply
        and reply_reference.is_to_bot
        and reply_reference.text
        and record.origin_prompt_digest
        and _prompt_capability_digest(reply_reference.text)
        == record.origin_prompt_digest
    )


def _active_operation_confirmation_matches(
    operation: ActiveDraftOperation,
    message_text: str,
) -> bool:
    """Accept assent only for the one current actor-owned active operation."""
    if operation.status != "awaiting_confirmation":
        return False
    if re.search(r"[?？]", message_text):
        return False
    if not isinstance(operation.pending_state, PendingToolConfirm):
        return False
    intent = _pending_tool_assent_intent(operation.pending_state, message_text)
    return bool(intent is not None and intent.intent == "pending_confirm")


def _is_referenced_word_presence_query(message_text: str) -> bool:
    """Detect deictic quoted-message questions like "这两个词词库都有吗"."""
    text = _strip_command_message_prefixes(message_text)
    text = re.sub(r"\s+", "", text)
    return _REFERENCED_WORD_PRESENCE_WHOLE_RE.fullmatch(text) is not None


def _dedupe_words(words: List[str], limit: int) -> List[str]:
    result: List[str] = []
    seen = set()
    for word in words:
        token = str(word or "").strip()
        if not token or not _PURE_CHINESE_TOKEN_RE.fullmatch(token):
            continue
        if token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= limit:
            break
    return result


def _split_reference_word_group(text: str) -> List[str]:
    parts = re.split(r"[、,，/／和与及\s]+", text)
    return [
        part.strip()
        for part in parts
        if part.strip() and _PURE_CHINESE_TOKEN_RE.fullmatch(part.strip())
    ]


def _clean_reference_heading_line(line: str) -> str:
    text = line.strip()
    text = re.sub(r"^[\s#>*\-•·\d.、:：|]+", "", text)
    text = re.sub(r"^[^\u4e00-\u9fff「」]+", "", text)
    return text.strip()


def _extract_referenced_word_targets(reference_text: str, expected_count: int = 2) -> List[str]:
    """Extract the primary compared words from a quoted bot answer."""
    limit = max(1, min(expected_count or 2, 8))
    heading_words: List[str] = []

    for raw_line in (reference_text or "").splitlines():
        line = _clean_reference_heading_line(raw_line)
        if not line or "|" in line:
            continue

        quoted_heading = re.fullmatch(r"「([\u4e00-\u9fff]{1,30})」", line)
        plain_heading = re.fullmatch(r"([\u4e00-\u9fff]{1,30})", line)
        if quoted_heading:
            heading_words.extend(_split_reference_word_group(quoted_heading.group(1)))
        elif plain_heading:
            heading_words.append(plain_heading.group(1))

    words = _dedupe_words(heading_words, limit)
    if len(words) >= limit:
        return words

    comparison_words: List[str] = []
    for match in re.finditer(r"([\u4e00-\u9fff]{1,12})\s*(?:≫|>|＞|更常用|优于|高于|大于)\s*([\u4e00-\u9fff]{1,12})", reference_text or ""):
        comparison_words.extend([match.group(1), match.group(2)])

    words = _dedupe_words(words + comparison_words, limit)
    if len(words) >= limit:
        return words

    quoted_words: List[str] = []
    for quoted in re.findall(r"「([^」]{1,40})」", reference_text or ""):
        quoted_words.extend(_split_reference_word_group(quoted))
    return _dedupe_words(words + quoted_words, limit)


def _resolve_shift_target_code(
    state: PendingAddWord,
    command_intent: MessageCommandIntent,
) -> Optional[str]:
    """Resolve which occupied candidate the user wants to shift for."""
    if command_intent.intent != "pending_recode":
        return None

    if command_intent.choice_index is not None:
        idx = command_intent.choice_index - 1
        if 0 <= idx < len(state.candidates):
            code, occupied = state.candidates[idx]
            if occupied:
                return code

    for code, occupied in state.candidates:
        if not occupied:
            continue
        for occupant_word in state.occupied_words.get(code, []):
            if occupant_word and occupant_word == command_intent.target_word:
                return code

    occupied_codes = [code for code, occupied in state.candidates if occupied]
    if len(occupied_codes) == 1:
        return occupied_codes[0]
    return None


def _is_pending_tool_confirm_message(
    state: PendingToolConfirm,
    command_intent: MessageCommandIntent,
) -> bool:
    if state.function_name == "keytao_submit_batch":
        return command_intent.intent == "pending_confirm"
    return command_intent.intent in {"pending_confirm", "pending_add_and_submit"}


def _pending_tool_state_with_trailing_submit(
    state: PendingToolConfirm,
    command_intent: MessageCommandIntent,
) -> PendingToolConfirm:
    """Carry an explicit trailing submit through the consumed write ticket."""
    if (
        command_intent.intent != "pending_add_and_submit"
        or state.function_name == "keytao_submit_batch"
    ):
        return state
    return PendingToolConfirm(
        function_name=state.function_name,
        args={**state.args, "_submit_after": True},
        confirmation_source=state.confirmation_source,
    )
