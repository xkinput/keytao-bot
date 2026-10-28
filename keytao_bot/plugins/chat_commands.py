"""Execute deterministic chat commands and manage their pending state."""

import asyncio
import hashlib
import json
import os
import re
import time
import unicodedata
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import asdict, dataclass, is_dataclass, replace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from nonebot.log import logger

from ..utils.community_vote import submitted_vote_receipt
from ..harness.orchestrator import AUTHORITATIVE_LINK_TOOLS, AgentRequestContext
from ..harness.conversation import (
    ConversationAddress,
    ConversationKey,
    normalize_conversation_key,
)
from ..harness.authorization_grammar import (
    _NEGATIVE_MODAL_RE,
    _PHRASE_TYPE_BASE_WEIGHTS,
    _POSITIONAL_REPORTED_CONTEXT_RE,
    dictionary_recode_items_match,
    explicit_same_code_requested,
    explicit_complete_add_item,
    explicit_shift_modified_add_item,
    looks_like_lexical_review_target,
    message_mentions_change_request,
    parse_compound_eviction_add_plan,
    parse_dictionary_delete_command,
    parse_dictionary_recode_command,
    parse_entry_move_plan,
    parse_entry_swap,
    parse_contextual_relative_position,
    parse_eviction_modified_add,
    parse_replace_at_code,
    parse_reviewed_multi_word_selection,
    parse_reviewed_selection_command,
    replace_at_code_items_match,
    server_warning_confirmation_binding,
    unnamed_eviction_modified_add_item,
)
from ..harness.state import (
    ActiveDraftOperation,
    ConversationLockStore,
    DraftOperationCoordinator,
    PendingAddWord,
    PendingState,
    PendingStateRecord,
    PendingTrustedWordRecord,
    PendingToolConfirm,
    SQLiteConversationStateStore,
    _pending_add_word_payload,
    _pending_add_word_from_payload,
    create_pending_trusted_word_record,
    pending_batch_display_pairs,
    pending_duplicate_confirmation_state,
    pending_execution_args as _pending_execution_args,
    server_warning_pending_state as _pending_state_from_server_warning,
    server_warning_ticket_is_complete,
    trusted_word_record_is_complete,
)
from ..harness.tools import (
    ToolContext,
    ToolExecutor,
    batch_warning_confirmation_binding,
    create_warning_confirmation_binding,
    trusted_mutation_source,
)
from ..utils.history_store import HistoryGenerationToken, get_history_store
from ..utils.draft_mutation_store import get_default_draft_mutation_claim_store
from ..utils.candidate_inventory import select_candidate_inventory, protected_candidate_occupants
from ..utils.commonness_copy import render_commonness_summary
from ..utils.explicit_code import (
    ExplicitEntryCodeRequest,
    parse_explicit_code_request,
    parse_explicit_entry_code_request,
    validate_explicit_code,
)
from ..utils.llm_policy import log_chat_usage, with_deepseek_chat_policy
from ..utils import keytao_review, review_flags, user_resolver
from ..utils.observability import observe_model_call, set_turn_flow
from ..utils.pending_confirmation import (
    ServerBackedQueryReply,
    already_existing_word_copy,
    already_existing_words_copy,
    append_unbound_binding_notice,
    advertised_batch_binding_pairs,
    advertised_single_word_lookup_codes,
    advertised_single_word_lookup_word,
    ensure_single_word_candidate_copy,
    ensure_multi_word_candidate_copy,
    parse_advertised_set_reference,
    parse_pending_assent_phrase,
    parse_pending_candidate_selection,
    pending_batch_confirmation_copy,
    pending_confirmation_copy,
    pending_word_reminder_lines,
    prepend_pending_word_reminders,
    render_executable_suggestion,
    render_server_backed_batch_candidates,
    render_remediation_reply,
    render_server_backed_single_word_lookup,
    single_word_candidate_footer,
    trusted_pending_word_items,
    validated_front_insert_recommendation,
)
from ..utils.memory_store import (
    ChatMemoryContext,
    MemoryGenerationToken,
    get_memory_store,
)
from .chat_adapters import (
    AsyncOpenAI,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_TIMEOUT,
    ReplyReferenceInfo,
    _as_float,
    _as_int,
    config,
)
from .chat_prompt import skills_manager
from .chat_render import (
    _BIND_HELP_TEXT,
    _append_batch_url_if_missing,
    _append_submit_review_lines,
    _assert_plain_user_facing_reply,
    _capture_trusted_result_links,
    _create_notice_lines,
    _dedupe_authoritative_link_lines,
    _draft_item_display_line,
    _format_active_draft_operation_message,
    _format_changed_server_confirmation_prompt,
    _format_full_add_and_submit_instruction,
    _format_operation_memory_for_reply,
    _format_pre_submit_audit_preview,
    _format_pronunciation_source,
    _format_referenced_word_presence_response,
    _format_replace_char_confirmation,
    _format_reviewed_add_prompt,
    _format_submit_conflict_failure,
    _normalize_generated_review_copy,
    _plain_warning_line,
    _plain_warning_message,
    _trusted_batch_url,
)
from .chat_routing import (
    note_rule_use,
    is_conversational_text,
    KeepOnlyDraftCommand,
    MessageCommandIntent,
    _DIRECT_OWNER_PENDING_ADD_INTENTS,
    _canonical_draft_management_command,
    _canonical_keep_only_command,
    _classify_message_command_intent,
    _closed_candidate_selection,
    _compact_command_text,
    _describe_pending_state,
    _describe_pending_ticket_choice,
    _extract_explicit_reviewed_add_word,
    _extract_referenced_word_targets,
    _format_live_ticket_precedence_message,
    _get_simple_word_query_words,
    _is_bare_word_query_target,
    _is_explicit_draft_submit_request,
    _is_pending_tool_confirm_message,
    _is_referenced_word_presence_query,
    _is_sensitive_pending_control_intent,
    _is_short_add_and_submit_request,
    _message_authorizes_draft_clear,
    _message_authorizes_draft_recall,
    _message_authorizes_pending_state_control,
    _message_authorizes_replace_char,
    _parse_code_chain_reorder_command,
    _parse_word_list_reorder_command,
    _parse_pending_choice_index,
    _pending_owner_label,
    _pending_assent_rejection_response,
    _pending_trusted_word_action_matches,
    _pending_tool_assent_intent,
    _pending_tool_confirmation_matches,
    _pending_tool_state_with_trailing_submit,
    _prompt_capability_digest,
    _record_flow_for_intent,
    _resolve_multi_word_pending_candidate_selection,
    _resolve_shift_target_code,
    _should_augment_simple_word_query,
    _strip_command_message_prefixes,
    _structural_pending_add_word_intent,
    _ticket_payload_from_command_intent,
)


KEYTAO_BACKGROUND_OPERATION_TIMEOUT: float = max(
    30.0,
    _as_float(
        getattr(config, "keytao_background_operation_timeout", None) or 420,
        420.0,
    ),
)


KEYTAO_BACKGROUND_MAX_CONCURRENCY = max(
    1,
    min(
        _as_int(getattr(config, "keytao_background_max_concurrency", None), 4),
        16,
    ),
)


MEMORY_COMPACTION_MAX_CONCURRENCY = max(
    1,
    min(
        _as_int(getattr(config, "memory_compaction_max_concurrency", None), 1),
        4,
    ),
)


_draft_operation_semaphore = asyncio.Semaphore(KEYTAO_BACKGROUND_MAX_CONCURRENCY)


_memory_compaction_semaphore = asyncio.Semaphore(MEMORY_COMPACTION_MAX_CONCURRENCY)


history_store = get_history_store()


memory_store = get_memory_store()


memory_store.compaction_lease_seconds = max(
    memory_store.compaction_lease_seconds,
    (OPENAI_TIMEOUT * 2) + 120.0,
)


MAX_HISTORY_MESSAGES = 24


conversation_state_store = SQLiteConversationStateStore(
    os.getenv("KEYTAO_PENDING_CONFIRMATIONS_DB") or None
)


conversation_states: Dict[ConversationAddress, PendingState] = conversation_state_store.states


conversation_message_locks = ConversationLockStore()


conversation_space_message_locks = ConversationLockStore()


draft_actor_message_locks = ConversationLockStore()


draft_operation_coordinator = DraftOperationCoordinator()


background_draft_tasks: set[asyncio.Task[Any]] = set()


background_draft_tasks_by_conversation: Dict[ConversationAddress, set[asyncio.Task[Any]]] = {}


memory_compaction_tasks: Dict[Tuple[str, str], asyncio.Task[Any]] = {}


current_memory_context: ContextVar[Optional[ChatMemoryContext]] = ContextVar(
    "current_memory_context",
    default=None,
)


@dataclass(frozen=True)
class _ExplicitAdditionalCodeCapability:
    word: str
    code: str
    phrase_type: str
    existing_codes: Tuple[str, ...]


_current_explicit_additional_code: ContextVar[Optional[_ExplicitAdditionalCodeCapability]] = ContextVar(
    "current_explicit_additional_code", default=None,
)


current_history_generation: ContextVar[Optional[HistoryGenerationToken]] = ContextVar(
    "current_history_generation",
    default=None,
)


current_memory_generation: ContextVar[Optional[MemoryGenerationToken]] = ContextVar(
    "current_memory_generation",
    default=None,
)


current_draft_operation_id: ContextVar[Optional[str]] = ContextVar(
    "current_draft_operation_id",
    default=None,
)


current_draft_result_links: ContextVar[Optional[Dict[str, str]]] = ContextVar(
    "current_draft_result_links",
    default=None,
)


current_draft_delivery_claims: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "current_draft_delivery_claims",
    default=None,
)


current_recall_clear_batch_id: ContextVar[Optional[str]] = ContextVar(
    "current_recall_clear_batch_id",
    default=None,
)


def _parse_pending_add_word(response: str) -> Optional[PendingAddWord]:
    """Parse AI response for the candidate code confirmation pattern.

    Looks for: 是否以编码 XXX 将「YYY」加入草稿
    and the numbered candidate list.
    """
    if "本次仅查询" in response:
        return None
    confirm_match = re.search(r'以编码\s*([a-z]+)\s*将「(.+?)」加入草稿', response)
    compact_match = re.search(
        r'(?m)^\s*[•-]\s*「(?P<word>[^」\n]+)」\s*→\s*'
        r'(?P<code>[a-z]+)\s*[（(]推荐[）)]\s*$',
        response,
    )
    reorder_match = re.search(
        r'(?m)^\s*[-•]\s*[“『]「(?P<word>[^」\n]+)」\s*占\s*'
        r'(?P<code>[a-z]+)\s*、\s*「[^」\n]+」顺延[”』]'
        r'(?:\s*[（(][^）)\n]*[）)])?\s*$',
        response,
    )
    if reorder_match is None:
        reorder_match = re.search(
            r'(?m)^\s*推荐[：:]\s*「(?P<word>[^」\n]+)」\s*占\s*'
            r'(?P<code>[a-z]+)\s*、\s*「[^」\n]+」顺延(?:[；;].*)?$',
            response,
        )
    if confirm_match is not None:
        recommended_code = confirm_match.group(1)
        word = confirm_match.group(2)
    elif reorder_match is not None:
        recommended_code = reorder_match.group("code")
        word = reorder_match.group("word")
    elif compact_match is not None:
        recommended_code = compact_match.group("code")
        word = compact_match.group("word")
    elif "推荐编码：暂无（现有候选均需点名顶替）" in response:
        blocked_word = re.search(r"(?:词库暂无收录「([^」]+)」|「([^」]+)」候选编码：)", response)
        if blocked_word is None:
            return None
        word = blocked_word.group(1) or blocked_word.group(2)
        recommended_code = ""
    else:
        return None

    candidates: List[Tuple[str, bool]] = []
    occupied_words: Dict[str, List[str]] = {}
    code_remarks: Dict[str, str] = {}
    pronunciation_codes: Dict[str, str] = {}
    pronunciation_recommended_codes: List[str] = []
    seen_codes = set()
    for m in re.finditer(r'(?m)^\s*(?:\d+\.\s*)?([a-z]+)\s*[-—–]\s*(.+?)\s*$', response):
        code = m.group(1)
        desc = m.group(2)
        if code in seen_codes:
            continue
        seen_codes.add(code)

        desc_text = desc.strip()
        is_available = desc_text.startswith("✅") or "空位" in desc_text
        candidates.append((code, not is_available))
        if "读音" in desc_text or "来源" in desc_text:
            code_remarks[code] = "喵喵审词：" + desc_text
        pinyin_match = re.search(r'读音\s*([A-Za-züÜvV:āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ\s]+)', desc_text)
        if pinyin_match:
            pronunciation_codes[code] = re.sub(r"\s+", " ", pinyin_match.group(1)).strip()
        if "该读音推荐" in desc_text or "推荐" in desc_text:
            pronunciation_recommended_codes.append(code)
        occupied_match = re.search(r'已有「(.+?)」', desc)
        if occupied_match:
            occupied_words[code] = [
                part.strip()
                for part in occupied_match.group(1).split('、')
                if part.strip()
            ]
        elif not is_available:
            cleaned_desc = re.sub(r'已有\s*', '', desc_text)
            cleaned_desc = cleaned_desc.replace("✔️", "")
            cleaned_desc = re.sub(r'[（(].*?[）)]', '', cleaned_desc)
            occupied_words[code] = [
                part.strip()
                for part in re.split(r'[、,，]\s*', cleaned_desc)
                if part.strip()
            ]

    if not candidates:
        candidates = [(recommended_code, False)]

    reading_pinyins = {
        int(match.group("index")): re.sub(
            r"\s+",
            " ",
            match.group("pinyin"),
        ).strip()
        for match in re.finditer(
            r"(?m)^\s*(?P<index>\d+)\.\s*"
            r"(?P<pinyin>[A-Za-züÜvV:āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ\s]+?)"
            r"\s*[；;]\s*来源(?:\s|$)",
            response,
        )
    }
    group_headings = list(re.finditer(
        r"(?m)^\s*候选编码[（(]读音\s*(?P<index>\d+)[）)]\s*[:：]\s*$",
        response,
    ))
    for heading_index, heading in enumerate(group_headings):
        pinyin = reading_pinyins.get(int(heading.group("index")), "")
        if not pinyin:
            continue
        section_end = (
            group_headings[heading_index + 1].start()
            if heading_index + 1 < len(group_headings)
            else len(response)
        )
        section = response[heading.end():section_end]
        for match in re.finditer(
            r"(?m)^\s*(?:\d+\.\s*)?(?P<code>[a-z]+)\s*[-—–]\s*.+$",
            section,
        ):
            code = match.group("code")
            if code in seen_codes:
                pronunciation_codes[code] = pinyin

    review_line_match = re.search(
        r'(?m)^\s*(?:喵喵)?审词：(.+?)\s*$',
        response,
    )
    if review_line_match:
        review_text = review_line_match.group(1).strip()
        if review_text:
            for code, _ in candidates:
                code_remarks.setdefault(code, "喵喵审词：" + review_text)
            pinyin_match = re.search(r'读音\s*([A-Za-züÜvV:āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ\s]+)', review_text)
            if pinyin_match:
                pinyin = re.sub(r"\s+", " ", pinyin_match.group(1)).strip()
                for code, _ in candidates:
                    pronunciation_codes.setdefault(code, pinyin)

    needs_manual_review, manual_review_reason = _take_reviewed_add_verdict(word)
    return PendingAddWord(
        word=word,
        recommended_code=recommended_code,
        candidates=candidates,
        occupied_words=occupied_words,
        code_remarks=code_remarks,
        pronunciation_codes=pronunciation_codes,
        pronunciation_recommended_codes=pronunciation_recommended_codes,
        needs_manual_review=needs_manual_review,
        manual_review_reason=manual_review_reason,
    )


def _server_candidate_snapshot(
    statuses: List[Dict],
) -> Tuple[
    List[Tuple[str, bool]],
    Dict[str, List[str]],
    Dict[str, List[Tuple[str, int]]],
]:
    """Freeze candidate occupancy from structured server fields only."""
    by_code: Dict[str, Tuple[bool, Tuple[str, ...]]] = {}
    order: List[str] = []
    for status in statuses:
        if not isinstance(status, dict):
            return [], {}, {}
        code = str(status.get("code") or "").strip().lower()
        if not re.fullmatch(r"[a-z]{1,6}", code):
            return [], {}, {}
        raw_occupied = status.get("occupied")
        raw_words = status.get("words") or []
        raw_phrases = status.get("phrases") or []
        if (
            not isinstance(raw_occupied, bool)
            or not isinstance(raw_words, list)
            or not isinstance(raw_phrases, list)
        ):
            return [], {}, {}
        occupied = raw_occupied
        words = [
            str(word or "").strip()
            for word in raw_words
            if str(word or "").strip()
        ]
        if not words:
            words = [
                str(phrase.get("word") or "").strip()
                for phrase in raw_phrases
                if isinstance(phrase, dict)
                and str(phrase.get("word") or "").strip()
            ]
        normalized = tuple(dict.fromkeys(words if occupied else []))
        value = (occupied, normalized)
        if code in by_code and by_code[code] != value:
            return [], {}, {}
        if code not in by_code:
            order.append(code)
        by_code[code] = value
    candidates = [(code, by_code[code][0]) for code in order]
    occupied_words = {
        code: list(by_code[code][1])
        for code in order
        if by_code[code][0]
    }
    entries_by_code: Dict[str, List[Tuple[str, int]]] = {}
    for status in statuses:
        code = str(status.get("code") or "").strip().lower()
        entries: List[Tuple[str, int]] = []
        for phrase in status.get("phrases") or []:
            if not isinstance(phrase, dict):
                entries = []
                break
            word = str(phrase.get("word") or "").strip()
            weight = phrase.get("weight")
            if (
                not word
                or not isinstance(weight, int)
                or isinstance(weight, bool)
                or weight < 0
            ):
                entries = []
                break
            entry = (word, weight)
            if entry not in entries:
                entries.append(entry)
        if entries:
            entries_by_code[code] = sorted(
                entries,
                key=lambda item: (item[1], item[0]),
            )
    return candidates, occupied_words, entries_by_code


def _server_ordering_snapshot(
    state: PendingAddWord,
    candidates: List[Tuple[str, bool]],
    occupied_words: Dict[str, List[str]],
    assessments: Any,
) -> List[Dict[str, str]]:
    """Validate advisory ordering facts against the structured occupancy snapshot."""
    if assessments is None:
        return []
    if not isinstance(assessments, list):
        return []
    occupied_by_code = dict(candidates)
    allowed_verdicts = {
        "front_more_common",
        "behind_more_common",
        "close",
        "not_enough_evidence",
    }
    snapshot: List[Dict[str, str]] = []
    for assessment in assessments[:2]:
        if not isinstance(assessment, dict):
            return []
        verdict = str(assessment.get("verdict") or "")
        new_word = str(assessment.get("newWord") or "").strip()
        occupant_word = str(assessment.get("occupantWord") or "").strip()
        occupant_code = str(assessment.get("occupantCode") or "").strip().lower()
        free_code = str(assessment.get("freeCode") or "").strip().lower()
        new_code = str(
            assessment.get("newCode")
            or assessment.get("recommendedCode")
            or ""
        ).strip().lower()
        expected_new_code = (
            occupant_code if verdict == "front_more_common" else free_code
        )
        if (
            verdict not in allowed_verdicts
            or new_word != state.word
            or not occupant_word
            or occupied_by_code.get(occupant_code) is not True
            or (occupied_by_code.get(free_code) is not False and not (
                not free_code and all(occupied_by_code.values())
            ))
            or occupant_word not in occupied_words.get(occupant_code, [])
            or new_code != expected_new_code
        ):
            return []
        snapshot.append({
            "verdict": verdict,
            "newWord": new_word,
            "occupantWord": occupant_word,
            "occupantCode": occupant_code,
            "freeCode": free_code,
            "newCode": new_code,
        })
    return snapshot


def _attach_server_candidate_snapshot(
    state: PendingAddWord,
    statuses: List[Dict],
    ordering_assessments: Any = None,
) -> PendingAddWord:
    candidates, occupied_words, entries_by_code = _server_candidate_snapshot(
        statuses
    )
    if candidates == state.candidates:
        state.server_candidates = candidates
        state.server_occupied_words = occupied_words
        state.server_entries_by_code = entries_by_code
        state.server_ordering_assessments = _server_ordering_snapshot(
            state,
            candidates,
            occupied_words,
            ordering_assessments,
        )
    return state


def _pending_add_ordering_summary(state: PendingAddWord, code: str) -> str:
    """Describe the default tail insertion using only server-backed entries."""
    existing_entries = state.server_entries_by_code.get(code, [])
    existing_words = [
        word
        for word, _weight in sorted(existing_entries, key=lambda item: item[1])
    ]
    if not existing_words:
        existing_words = list(state.server_occupied_words.get(code, []))
    if existing_words:
        return (
            f"{code}：{' → '.join([*existing_words, state.word])}"
            "（新词按默认权重排在后）"
        )
    return ""


def _protected_eviction_response(
    state: PendingAddWord, code: str, protected: Tuple[str, ...],
) -> str:
    """Share the same protected-slot advice across pending and complete commands."""
    if "" in protected:
        return f"编码 {code} 的占用词条未能核验，请重新查询；本次未写入。"
    occupants = protected or tuple(dict.fromkeys(
        word for words in state.server_occupied_words.values() for word in words
    ))
    base = next(iter(state.server_candidates), ("", False))[0]
    suggestions = "\n".join(f"- 「加入，顶替 {word}」" for word in occupants)
    reason = "；".join(render_commonness_summary(next((
        assessment for assessment in state.server_ordering_assessments
        if assessment.get("newWord") == state.word
        and assessment.get("occupantWord") == word
        and assessment.get("occupantCode") == code
    ), {"newWord": state.word, "occupantWord": word})) for word in occupants)
    return (
        reason + "，保留现有位置；本次未写入。\n"
        f"可指定形码后的完整编码，格式为：加入编码 {base}+形码（请替换形码部分）。"
        + ("\n若要顶替，请点名：\n" + suggestions if suggestions else "")
    )


def _pending_add_reorder_recommendation(
    state: PendingAddWord,
) -> Optional[Dict[str, str]]:
    """Resolve the live default only from its bound comparator snapshot."""
    return validated_front_insert_recommendation(
        state.word,
        state.server_candidates,
        state.server_occupied_words,
        state.server_ordering_assessments,
    )


def _pending_batch_front_insert_plan(
    state: PendingToolConfirm,
) -> Optional[Dict[str, Any]]:
    """Bind one comparator front insert to its sealed companion additions."""
    if (
        state.function_name != "keytao_batch_add_to_draft"
        or state.confirmation_source == "server_warning"
    ):
        return None
    items = state.args.get("items")
    scopes = state.args.get("_candidate_scopes")
    if not isinstance(items, list) or not isinstance(scopes, list):
        return None
    items_by_word = {
        str(item.get("word") or "").strip(): item
        for item in items
        if isinstance(item, dict) and str(item.get("word") or "").strip()
    }
    recommendations: List[Tuple[Dict[str, str], Dict[str, Any]]] = []
    for scope in scopes:
        if not isinstance(scope, dict):
            return {"invalid": True}
        word = str(scope.get("word") or "").strip()
        recommendation = validated_front_insert_recommendation(
            word,
            scope.get("candidates"),
            scope.get("occupiedWords"),
            scope.get("orderingAssessments"),
        )
        if recommendation is None:
            continue
        item = items_by_word.get(word)
        if (
            not isinstance(item, dict)
            or str(item.get("code") or "").strip().lower()
            != recommendation["occupantCode"]
        ):
            return {"invalid": True}
        recommendations.append((recommendation, item))
    if not recommendations:
        return None
    if len(recommendations) != 1 and state.args.get("_reviewed_multi_word") is not True:
        return {"unsupportedCount": len(recommendations)}
    recommendation, target_item = recommendations[0]
    if state.args.get("_reviewed_multi_word") is True:
        capabilities = _reviewed_multi_word_capabilities(state)
        scopes_by_word = {scope["word"]: scope for scope in scopes}
        reorder_words = {value[0]["newWord"] for value in recommendations}
        ordered_items = [target_item, *(item for item in items if item is not target_item)]
        expected_occupants: Dict[str, List[str]] = {}
        additional_shifts: List[Dict[str, Any]] = []
        companions: List[Dict[str, Any]] = []
        selected_codes: set[str] = set()
        for item in ordered_items:
            word, code = item["word"], item["code"]
            capability = capabilities.get((word, code))
            scope = scopes_by_word.get(word, {})
            occupancy = dict(scope.get("candidates", []))
            occupants = scope.get("occupiedWords", {}).get(code, [])
            if (
                not capability or code not in occupancy or code in selected_codes
                or not isinstance(occupants, list)
                or bool(occupants) != occupancy[code]
            ):
                return {"invalid": True}
            selected_codes.add(code)
            if word in reorder_words:
                expected_occupants[code] = list(occupants)
            if item is not target_item:
                destination = additional_shifts if word in reorder_words else companions
                destination.append({
                    **item, "reviewedPinyin": capability["pinyin"],
                    "reviewedCandidateCodes": list(capability["candidate_codes"]),
                })
        return {
            "recommendation": recommendation, "targetItem": dict(target_item),
            "additionalItems": companions, "additionalShiftItems": additional_shifts,
            "reviewedCapability": capabilities[(target_item["word"], target_item["code"])],
            "expectedOccupantsByCode": expected_occupants,
            "expectedShiftedWords": tuple(word for words in expected_occupants.values() for word in words),
        }
    return {
        "recommendation": recommendation,
        "targetItem": dict(target_item),
        "additionalItems": [
            dict(item)
            for item in items
            if isinstance(item, dict) and item is not target_item
        ],
        "additionalShiftItems": [],
        "reviewedCapability": {},
        "expectedOccupantsByCode": {},
        "expectedShiftedWords": (),
    }


def _reviewed_multi_word_capabilities(state: PendingToolConfirm) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Project selected codes from persisted per-word reviews into tool seals."""
    items = state.args.get("items")
    if (
        state.function_name == "keytao_batch_add_to_draft"
        and isinstance(items, list)
        and not isinstance(state.args.get("_reviewed_batch_readings"), list)
        and state.args.get("_reviewed_multi_word") is not True
        and any(isinstance(item, dict) and (
            "_reviewed_pinyin" in item or "_reviewed_candidate_codes" in item
        ) for item in items)
    ):
        if not server_warning_ticket_is_complete(state):
            return {}
        capabilities = {}
        for item in items:
            if not isinstance(item, dict) or item.get("action", "Create") not in {"Create", "Change", "Delete"}:
                return {}
            if item.get("action") == "Delete":
                continue
            word, code = item.get("word"), item.get("code")
            pinyin = item.get("_reviewed_pinyin")
            codes = item.get("_reviewed_candidate_codes")
            if (
                not isinstance(word, str) or not word
                or not isinstance(code, str) or not re.fullmatch(r"[a-z]{1,6}", code)
                or not isinstance(pinyin, str) or not pinyin.strip()
                or not isinstance(codes, list) or code not in codes
                or any(not isinstance(value, str) or not re.fullmatch(r"[a-z]{1,6}", value) for value in codes)
                or len(set(codes)) != len(codes)
                or item.get("type") not in _PHRASE_TYPE_BASE_WEIGHTS
                or not isinstance(item.get("needsManualReview"), bool)
            ):
                return {}
            capability = {
                "type": item["type"], "pinyin": pinyin,
                "candidate_codes": tuple(codes),
                "needs_manual_review": item["needsManualReview"],
                "remark": str(item.get("remark") or ""),
            }
            key = (word, code)
            if key in capabilities and capabilities[key] != capability:
                return {}
            capabilities[key] = capability
        return capabilities
    saved = state.args.get("_reviewed_batch_readings")
    if isinstance(saved, list):
        capabilities = {}
        for item in state.args.get("items", []):
            matches = [row for row in saved if isinstance(row, dict) and row.get("word") == item.get("word")]
            if len(matches) != 1:
                return {}
            capability = _reviewed_create_capability(
                item["word"], item["code"], str(matches[0].get("pinyin") or ""),
                tuple(matches[0].get("candidate_codes") or ()),
            )
            if capability is None:
                return {}
            capabilities.update(capability)
        return capabilities
    if state.args.get("_reviewed_multi_word") is not True:
        return {}
    scopes = {scope["word"]: scope for scope in state.args.get("_candidate_scopes", [])}
    capabilities: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in state.args.get("items", []):
        word, code = item["word"], item["code"]
        try:
            reviewed = _pending_add_word_from_payload(json.loads(json.dumps(scopes[word]["reviewedState"])))
        except (KeyError, TypeError, ValueError):
            return {}
        pinyin, codes = _pending_reviewed_reading(reviewed, code)
        capability = _reviewed_create_capability(word, code, pinyin, codes)
        if capability is None:
            return {}
        capabilities.update(capability)
    return capabilities


def _create_phrase_args(state: PendingAddWord, code: str) -> Dict:
    """Build mutation arguments without losing the structured review verdict."""
    args: Dict = {"word": state.word, "code": code}
    if state.phrase_type == "Single":
        args["type"] = "Single"
    remark = state.code_remarks.get(code)
    if remark:
        args["remark"] = remark
    if state.needs_manual_review is not None:
        args["needs_manual_review"] = bool(state.needs_manual_review)
    ordering_summary = _pending_add_ordering_summary(state, code)
    if ordering_summary:
        args["_ordering_summary"] = ordering_summary
    reviewed_pinyin, reviewed_candidate_codes = _pending_reviewed_reading(
        state,
        code,
    )
    if reviewed_pinyin and reviewed_candidate_codes:
        args["_reviewed_pinyin"] = reviewed_pinyin
        args["_reviewed_candidate_codes"] = list(reviewed_candidate_codes)
    return args


def _pending_reviewed_reading(
    state: PendingAddWord,
    code: str,
) -> Tuple[str, Tuple[str, ...]]:
    """Return the one reviewed reading group containing ``code``."""
    pinyin = str(state.pronunciation_codes.get(code) or "").strip()
    if not pinyin:
        return "", ()
    codes = tuple(
        candidate_code
        for candidate_code, _occupied in state.candidates
        if str(state.pronunciation_codes.get(candidate_code) or "").strip()
        == pinyin
    )
    return (pinyin, codes) if code in codes else ("", ())


def _reviewed_create_capability(
    word: str,
    code: str,
    pinyin: str,
    candidate_codes: Tuple[str, ...],
) -> Optional[Dict[Tuple[str, str], Dict[str, Any]]]:
    if not pinyin or code not in candidate_codes:
        return None
    return {
        (word, code): {
            "type": "Single" if len(word) == 1 and re.fullmatch(r"[\u3400-\u9fff\U00020000-\U0002fa1f]", word) else "Phrase",
            "pinyin": pinyin,
            "candidate_codes": candidate_codes,
        },
    }


def _batch_review_remark(response: str, word: str) -> str:
    """Extract the reviewed-add line belonging to one word section."""
    header = re.search(rf"(?m)^「{re.escape(word)}」[^\n]*$", response)
    if not header:
        return ""
    block_start = header.end()
    next_header = re.search(r"(?m)^「[^」]+」[^\n]*$", response[block_start:])
    block_end = block_start + next_header.start() if next_header else len(response)
    block = response[block_start:block_end]
    review_match = re.search(r"(?m)^\s*(?:喵喵)?审词：(.+?)\s*$", block)
    if not review_match:
        return ""
    review_text = _normalize_generated_review_copy(review_match.group(1).strip())
    return f"喵喵审词：{review_text}" if review_text else ""


def _parse_pending_batch_add(response: str) -> Optional[PendingToolConfirm]:
    """Parse AI response for a multi-word add confirmation prompt."""
    normalized_response = _normalize_generated_review_copy(response)
    batch_prompt_markers = (
        "一起加入草稿",
        "一起加这两个词",
        "两个词是否一起加",
        "两词是否一起加",
    )
    visible_pairs = advertised_batch_binding_pairs(normalized_response)
    if (
        len(visible_pairs) < 2
        and not any(marker in normalized_response for marker in batch_prompt_markers)
    ):
        return None

    confirm_line = next(
        (
            line.strip()
            for line in normalized_response.splitlines()
            if "一起加入草稿" in line and "将「" in line
        ),
        "",
    )

    items = []
    seen = set()
    inline_pairs = re.findall(
        r'(?:以编码\s*)?([a-z]{2,12})\s*将「(.+?)」',
        confirm_line,
        re.IGNORECASE,
    ) if confirm_line else []
    arrow_pairs = [
        (match.group(2), match.group(1))
        for match in re.finditer(
            r'(?m)^\s*[-•]\s*「([^」]+)」\s*(?:→|->)\s*([a-z]{2,12})\s*$',
            normalized_response,
            re.IGNORECASE,
        )
    ]
    inline_arrow_pairs = [
        (match.group(3), match.group(1) or match.group(2))
        for match in re.finditer(
            r'(?:「([^」]+)」|([\u4e00-\u9fff]{1,12}))\s*(?:→|->)\s*([a-z]{2,12})',
            normalized_response,
            re.IGNORECASE,
        )
    ]
    exact_line_pairs = [(code, word) for word, code in visible_pairs]
    for code, word in [
        *exact_line_pairs,
        *inline_pairs,
        *arrow_pairs,
        *inline_arrow_pairs,
    ]:
        code = code.lower()
        word = word.strip()
        key = (word, code)
        if key in seen:
            continue
        seen.add(key)
        item = {"word": word, "code": code, "action": "Create"}
        remark = _batch_review_remark(normalized_response, word)
        if remark:
            item["remark"] = remark
        verdict, verdict_reason = _take_reviewed_add_verdict(word)
        if verdict is not None:
            review_flags.apply_manual_review_flag(item, verdict, verdict_reason)
        items.append(item)

    if len(items) < 2:
        return None

    return PendingToolConfirm(
        function_name="keytao_batch_add_to_draft",
        args={"items": items},
    )


def _can_use_unrelated_group_pending(reply_reference: ReplyReferenceInfo) -> bool:
    """Never bind a quoted bot reply to an unrelated user's group pending state."""
    return not reply_reference.is_to_bot


def _get_latest_assistant_message(history: Optional[List[Dict]]) -> str:
    """Return the most recent assistant message, if any."""
    if not history:
        return ""
    for msg in reversed(history):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "") or "")
    return ""


async def _try_recover_reviewed_add_from_history(
    message_text: str,
    history: Optional[List[Dict]],
    platform: str,
    user_id: str,
    conv_key: Optional[ConversationKey] = None,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> Optional[str]:
    """Re-review a stale display, but never execute its old selector this turn."""
    assent = parse_pending_assent_phrase(message_text)
    selection = _closed_candidate_selection(message_text)
    if selection is None and not (assent.matched and assent.add_requested):
        return None
    latest_assistant = _get_latest_assistant_message(history)
    displayed = _parse_pending_add_word(latest_assistant)
    displayed_word = (
        displayed.word
        if isinstance(displayed, PendingAddWord)
        else advertised_single_word_lookup_word(latest_assistant)
    )
    if not displayed_word:
        return None
    displayed_codes = (
        tuple(code for code, _occupied in displayed.candidates)
        if isinstance(displayed, PendingAddWord)
        else advertised_single_word_lookup_codes(latest_assistant)
    )
    if not displayed_codes:
        return None
    refreshed = await _try_handle_simple_single_word_query(
        f"加词 {displayed_word}",
        platform,
        user_id,
        conv_key,
        space_key,
        owner_label,
    )
    if conv_key is None:
        return refreshed
    record = conversation_state_store.get_record(conv_key)
    state = record.state if record is not None else None
    if not isinstance(state, PendingAddWord) or not state.server_candidates:
        return (
            "刚才的候选记录已失效，暂时无法重新生成；请重新查询这个词条。"
        )

    # The current selector came from a display whose actor-owned record is gone.
    # The fresh server snapshot above may mint authority only for a later turn;
    # carrying the stale ordinal/code into it would silently retarget a write.
    return ServerBackedQueryReply(
        "刚才的候选记录已失效，已重新生成；请按下面的新列表重新选择。\n"
        + str(refreshed or "")
    )


_EXPLICIT_READING_TOKEN_PATTERN = (
    r"[A-Za-züÜvV:āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ]+[0-5]?"
    r"(?:\s+[A-Za-züÜvV:āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ]+[0-5]?){0,19}"
)


def _latest_unresolved_review_word(history: Optional[List[Dict]]) -> str:
    latest = _get_latest_assistant_message(history)
    match = re.search(
        r"「(?P<word>[^「」\r\n]{1,20})」(?:存在[^\r\n]{0,40}多个读音|"
        r"的[^\r\n]{0,20}读音[^\r\n]{0,20}(?:冲突|未确定))",
        latest,
    )
    if match is not None:
        note_rule_use("C:_latest_unresolved_review_word:re.search")
    return match.group("word").strip() if match is not None else ""


def _extract_explicit_reading_request(
    message_text: str,
    history: Optional[List[Dict]] = None,
) -> Optional[Tuple[str, str]]:
    """Bind a literal pinyin reading to the named or currently unresolved word."""
    source = unicodedata.normalize("NFKC", str(message_text or "")).strip()
    if not source or re.search(r"[?？]", source):
        return None
    character_reading = re.fullmatch(
        rf"^(?:(?:请|麻烦)?\s*(?:帮我|帮忙|给我)?\s*"
        rf"(?:加词|添加词|新增词)\s*)?"
        rf"(?P<word>[\u3400-\u9fff]{{1,20}})\s*[，,；;]?\s+"
        rf"(?P<char>[\u3400-\u9fff])(?:字)?(?:的)?(?:读音)?"
        rf"(?:是|为|读作|读成|读)\s*[:：]?\s*"
        rf"(?P<pinyin>[A-Za-züÜvV:āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ]+)"
        rf"(?:\s*[。.!！])?$",
        source,
        re.IGNORECASE,
    )
    if character_reading is not None:
        note_rule_use("C:_extract_explicit_reading_request:re.fullmatch")
        word = character_reading.group("word").strip()
        character = character_reading.group("char")
        pinyin = keytao_review.normalize_pinyin_syllable(
            character_reading.group("pinyin")
        )
        if character in word and pinyin:
            return word, f"{character}={pinyin}"
    patterns = (
        re.compile(
            rf"^(?:(?:请|麻烦)?\s*(?:帮我|帮忙|给我)?\s*"
            rf"(?:加词|添加词|新增词)\s*)?"
            rf"(?P<word>[\u3400-\u9fff]{{1,20}})\s*[，,；;]?\s*"
            rf"(?:的)?(?:读音(?:是|为)?|读作)\s*[:：]?\s*"
            rf"(?P<reading>{_EXPLICIT_READING_TOKEN_PATTERN})(?:\s*[。.!！])?$",
            re.IGNORECASE,
        ),
        re.compile(
            rf"^(?P<word>[\u3400-\u9fff]{{1,20}})\s*为\s*"
            rf"(?P<reading>{_EXPLICIT_READING_TOKEN_PATTERN})"
            rf"(?:\s*[，,；;].*)?$",
            re.IGNORECASE,
        ),
        re.compile(
            rf"^(?:(?P<word>[\u3400-\u9fff]{{1,20}})\s*)?按\s*"
            rf"(?P<reading>{_EXPLICIT_READING_TOKEN_PATTERN})\s*读"
            rf"(?:\s*[。.!！])?$",
            re.IGNORECASE,
        ),
        re.compile(
            rf"^(?P<reading>{_EXPLICIT_READING_TOKEN_PATTERN})\s*那个"
            rf"(?:\s*[。.!！])?$",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        match = pattern.fullmatch(source)
        if match is None:
            continue
        if pattern is patterns[2]:
            note_rule_use("C:_extract_explicit_reading_request:patterns[2]")
        if pattern is patterns[3]:
            note_rule_use("C:_extract_explicit_reading_request:patterns[3]")
        word = str(match.groupdict().get("word") or "").strip()
        if not word:
            word = _latest_unresolved_review_word(history)
        reading = keytao_review.normalize_pinyin_sequence(
            str(match.group("reading") or "")
        )
        if word and len(reading) == len(word) and all(reading):
            return word, " ".join(reading)
    return None


def _extract_explicit_reading_meaning_request(
    message_text: str,
    history: Optional[List[Dict]] = None,
) -> Optional[Tuple[str, str]]:
    """Bind a concrete sense description to one named or unresolved word."""
    source = unicodedata.normalize("NFKC", str(message_text or "")).strip()
    if (
        not source
        or len(source) > 180
        or re.search(r"[?？]", source)
        or not re.search(r"(?:意思|含义|指的是|语境|用法|那个)", source)
    ):
        return None
    note_rule_use("C:_extract_explicit_reading_meaning_request:re.search")
    explicit = re.fullmatch(
        r"(?:(?:请|麻烦)?\s*(?:帮我|帮忙|给我)?\s*"
        r"(?:加词|添加词|新增词)\s*)"
        r"(?P<word>[\u3400-\u9fff]{1,20})\s*[，,；;]?\s*"
        r"(?P<meaning>.+)",
        source,
    )
    if explicit is not None:
        note_rule_use("C:_extract_explicit_reading_meaning_request:re.fullmatch")
        word = explicit.group("word").strip()
        meaning = explicit.group("meaning").strip()
        if meaning:
            note_rule_use("C:_extract_explicit_reading_meaning_request")
        return (word, meaning) if meaning else None
    word = _latest_unresolved_review_word(history)
    if not word or re.search(r"(?:不要|别|取消|删除|移除|提交)", source):
        return None
    note_rule_use("C:_extract_explicit_reading_meaning_request")
    return word, source


async def _try_handle_explicit_reading_disambiguation(
    message_text: str,
    history: Optional[List[Dict]],
    platform: str,
    user_id: str,
    conv_key: Optional[ConversationKey] = None,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> Optional[str]:
    """Re-review a named reading before the free-form model can consume a turn."""
    explicit = _extract_explicit_reading_request(message_text, history)
    requested_reading = ""
    requested_meaning = ""
    if explicit is not None:
        word, requested_reading = explicit
    else:
        meaning_request = _extract_explicit_reading_meaning_request(
            message_text,
            history,
        )
        if meaning_request is None:
            return None
        word, requested_meaning = meaning_request
    return await _try_handle_simple_single_word_query(
        f"加词 {word}",
        platform,
        user_id,
        conv_key,
        space_key,
        owner_label,
        requested_reading=requested_reading,
        requested_meaning=requested_meaning,
    )


async def _try_handle_complete_add_command(
    message_text: str,
    platform: str,
    user_id: str,
    conv_key: Optional[ConversationKey] = None,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> Optional[str]:
    """Execute one whole-message word+code add through fresh server review."""
    command = explicit_complete_add_item(message_text)
    if command is None or conv_key is None:
        return None
    word = str(command.get("word") or "").strip()
    if not word:
        return None
    request = ExplicitEntryCodeRequest(word, str(command["code"]), submit_after=bool(command["submitAfter"]))
    response = await _execute_explicit_entry_code_request(
        request, f"添加 {word} {request.code}" + (" 并提交" if request.submit_after else ""),
        platform, user_id, conv_key, space_key, owner_label,
    )
    if response:
        note_rule_use("C:_try_handle_complete_add_command")
    return response


async def _try_handle_compound_shift_modified_add_command(
    message_text: str,
    platform: str,
    user_id: str,
    conv_key: Optional[ConversationKey] = None,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> Optional[str]:
    """Review every line, then start one sealed multi-eviction plan."""
    plan = parse_compound_eviction_add_plan(message_text)
    if plan is None:
        return None
    if conv_key is None:
        note_rule_use("C:_try_handle_compound_shift_modified_add_command")
        return (
            "已识别两行添加并顺延指令，但当前会话无法保存共同确认票据；"
            "本次未写入。"
        )

    def refuse(message: str) -> str:
        conversation_state_store.delete(conv_key)
        note_rule_use("C:_try_handle_compound_shift_modified_add_command")
        return message

    base_reviewed_states: Dict[str, PendingAddWord] = {}
    for item in plan.items:
        if item.word in base_reviewed_states:
            continue
        refreshed = await _try_handle_simple_single_word_query(
            f"加词 {item.word}",
            platform,
            user_id,
            conv_key,
            space_key,
            owner_label,
        )
        record = conversation_state_store.get_record(conv_key)
        state = record.state if record is not None else None
        if not isinstance(state, PendingAddWord) or not state.server_candidates:
            reason_parts = re.split(
                r"(?:请|回复|可复制)",
                str(refreshed or "服务端审词没有返回完整候选"),
                maxsplit=1,
            )
            if len(reason_parts) > 1:
                note_rule_use("C:_try_handle_compound_shift_modified_add_command:re.split")
            reason = reason_parts[0].strip().rstrip("；;。")
            return refuse(
                f"无法为「{item.word}」建立两行共同顺延计划；"
                f"原因：{reason or '服务端审词没有返回完整候选'}；本次未写入。"
            )
        base_reviewed_states[item.word] = state

    reviewed_states: Dict[Tuple[str, str], PendingAddWord] = {}
    for item in plan.items:
        state = base_reviewed_states[item.word]
        if item.code not in dict(state.server_candidates):
            encode_json = await call_tool_function(
                "keytao_encode",
                {"word": item.word, "requested_code": item.code},
                platform,
                user_id,
            )
            try:
                encoding = json.loads(encode_json)
            except Exception:
                encoding = {}
            requested_state = (
                _pending_from_requested_encode(item.word, item.code, encoding)
                if isinstance(encoding, dict) and encoding.get("success") is True
                else None
            )
            if (
                not isinstance(requested_state, PendingAddWord)
                or not requested_state.server_candidates
            ):
                return refuse(
                    f"第 {len(reviewed_states) + 1} 行的编码 {item.code} "
                    f"不在「{item.word}」当前服务端审词候选中；本次未写入。"
                )
            state = requested_state
        reviewed_states[(item.word, item.code)] = state

    resolved_plan: List[Dict[str, Any]] = []
    shift_targets: List[Dict[str, Any]] = []
    for index, item in enumerate(plan.items, start=1):
        state = reviewed_states[(item.word, item.code)]
        candidates = dict(state.server_candidates)
        if item.code not in candidates:
            return refuse(
                f"第 {index} 行的编码 {item.code} 不在「{item.word}」"
                "本轮服务端候选中；本次未写入。"
            )
        occupants = tuple(dict.fromkeys(
            str(value or "").strip()
            for value in state.server_occupied_words.get(item.code, [])
            if str(value or "").strip()
        ))
        if candidates[item.code] is not True or occupants != (
            item.named_occupant,
        ):
            actual = "、".join(f"「{value}」" for value in occupants) or "空位"
            return refuse(
                f"第 {index} 行的占用快照与指令不一致：{item.code} 当前为{actual}，"
                f"不能按「{item.named_occupant}」顺延；本次未写入。"
            )
        reviewed_pinyin, reviewed_codes = _pending_reviewed_reading(
            state,
            item.code,
        )
        if not reviewed_pinyin or item.code not in reviewed_codes:
            return refuse(
                f"第 {index} 行的「{item.word}」→ {item.code} "
                "没有保留对应的服务端读音候选；本次未写入。"
            )
        create_args = _create_phrase_args(state, item.code)
        target = {
            "action": "Create",
            "word": item.word,
            "code": item.code,
            "type": "Phrase",
            "remark": str(create_args.get("remark") or ""),
            "needsManualReview": bool(
                create_args.get("needs_manual_review", True)
            ),
            "namedOccupant": item.named_occupant,
            "reviewedPinyin": reviewed_pinyin,
            "reviewedCandidateCodes": list(reviewed_codes),
        }
        shift_targets.append(target)
        resolved_plan.append({
            "index": index,
            "word": item.word,
            "code": item.code,
            "modifier": "recode",
            "occupants": [item.named_occupant],
        })

    primary = shift_targets[0]
    response = await _execute_shift_to_code(
        str(primary["word"]),
        str(primary["code"]),
        platform,
        user_id,
        space_key,
        owner_label,
        target_item=primary,
        additional_shift_items=shift_targets[1:],
        resolved_candidate_plan=resolved_plan,
        reviewed_pinyin=str(primary["reviewedPinyin"]),
        reviewed_candidate_codes=tuple(primary["reviewedCandidateCodes"]),
    )
    if response:
        note_rule_use("C:_try_handle_compound_shift_modified_add_command")
    return response


async def _try_handle_shift_modified_add_command(
    message_text: str,
    platform: str,
    user_id: str,
    conv_key: Optional[ConversationKey] = None,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> Optional[str]:
    """Re-review a closed add-plus-shift command and start its bound shift plan."""
    parsed_command = parse_eviction_modified_add(message_text)
    if parsed_command is None and conv_key is not None:
        contextual_record = conversation_state_store.get_record(conv_key)
        contextual_state = (
            contextual_record.state if contextual_record is not None else None
        )
        if (
            contextual_record is not None
            and not contextual_record.execution_id
            and isinstance(contextual_state, PendingAddWord)
        ):
            parsed_command = parse_contextual_relative_position(
                message_text,
                contextual_state.word,
            )
    if (
        parsed_command is not None
        and not parsed_command.code
        and not explicit_same_code_requested(message_text)
        and parsed_command.modifier
        in {"前面", "之前", "前", "后面", "之后", "后"}
    ):
        if conv_key is None:
            return None
        return await _handle_relative_position_command(
            parsed_command,
            platform,
            user_id,
            conv_key,
            space_key,
            owner_label,
        )
    explicitly_named_occupant = str(
        parsed_command.named_occupant if parsed_command is not None else ""
    ).strip()
    if re.fullmatch(
        r"其他(?:相关(?:的)?)?词条",
        explicitly_named_occupant,
    ):
        note_rule_use("C:_try_handle_shift_modified_add_command:re.fullmatch:explicitly_named_occupant")
        explicitly_named_occupant = ""
    command = explicit_shift_modified_add_item(message_text)
    if command is None:
        incomplete = unnamed_eviction_modified_add_item(message_text)
        if incomplete is None:
            return None
        word = str(incomplete.get("word") or "").strip()
        code = str(incomplete.get("code") or "").strip().lower()
        try:
            occupant_data = json.loads(await call_tool_function(
                "keytao_lookup_by_code",
                {"code": code},
                platform,
                user_id,
            ))
        except Exception:
            occupant_data = {}
        raw_occupants = occupant_data.get("phrases")
        if occupant_data.get("success") is not True or not isinstance(
            raw_occupants,
            list,
        ):
            return (
                f"无法取得 {code} 的完整占位记录；"
                "本次未执行添加或顺延。"
            )
        occupants = list(dict.fromkeys(
            str(item.get("word") or "").strip()
            for item in raw_occupants
            if isinstance(item, dict)
            and str(item.get("code") or "").strip().lower() == code
            and str(item.get("word") or "").strip()
        ))
        if len(occupants) != 1 or occupants[0] == word:
            conflict = "、".join(f"「{value}」" for value in occupants) or "无现有词条"
            return (
                f"{code} 的现有占位无法唯一解析为待顺延词条：{conflict}；"
                "本次未执行添加或顺延。"
            )
        command = {
            **incomplete,
            "namedOccupant": occupants[0],
        }
    if conv_key is None:
        return None
    word = str(command.get("word") or "").strip()
    code = str(command.get("code") or "").strip().lower()
    refreshed = await _try_handle_simple_single_word_query(
        f"加词 {word}",
        platform,
        user_id,
        conv_key,
        space_key,
        owner_label,
    )
    record = conversation_state_store.get_record(conv_key)
    state = record.state if record is not None else None
    if not isinstance(state, PendingAddWord) or not state.server_candidates:
        return refreshed
    candidates = dict(state.server_candidates)
    if code not in candidates:
        return (
            f"{code} 不是「{word}」本轮服务端审词记录中的候选编码；"
            "本次未执行添加或顺延"
        )
    named_occupant = str(
        command.get("namedOccupant")
        or (
            parsed_command.named_occupant
            if parsed_command is not None
            else ""
        )
        or ""
    ).strip()
    if re.fullmatch(r"其他(?:相关(?:的)?)?词条", named_occupant):
        note_rule_use("C:_try_handle_shift_modified_add_command:re.fullmatch:named_occupant")
        named_occupant = ""
    if named_occupant:
        actual_occupants = tuple(dict.fromkeys(
            str(value or "").strip()
            for value in state.server_occupied_words.get(code, [])
            if str(value or "").strip()
        ))
        if actual_occupants != (named_occupant,):
            actual = "、".join(
                f"「{value}」" for value in actual_occupants
            ) or "空位"
            return (
                f"{code} 的当前占位为{actual}，与指令中的"
                f"「{named_occupant}」不一致；本次未执行添加或顺延"
            )
    reviewed_pinyin, reviewed_candidate_codes = _pending_reviewed_reading(
        state,
        code,
    )
    if not reviewed_pinyin or code not in reviewed_candidate_codes:
        return (
            f"「{word}」的候选编码 {code} 没有保留对应读音；"
            "本次未执行添加或顺延"
        )
    create_args = _create_phrase_args(state, code)
    if candidates[code] is not True:
        return await _execute_add_to_draft(
            word,
            code,
            platform,
            user_id,
            space_key,
            owner_label,
            remark=str(create_args.get("remark") or ""),
            needs_manual_review=create_args.get("needs_manual_review"),
            reviewed_pinyin=reviewed_pinyin,
            reviewed_candidate_codes=reviewed_candidate_codes,
        )
    protected = protected_candidate_occupants(
        state.word, code, state.server_candidates,
        state.server_occupied_words, state.server_ordering_assessments,
    )
    if protected and not explicitly_named_occupant:
        return _protected_eviction_response(state, code, protected)
    response = await _execute_shift_to_code(
        word,
        code,
        platform,
        user_id,
        space_key,
        owner_label,
        target_item={
            "type": state.phrase_type,
            "remark": str(create_args.get("remark") or ""),
            "needsManualReview": bool(
                create_args.get("needs_manual_review", True)
            ),
        },
        reviewed_pinyin=reviewed_pinyin,
        reviewed_candidate_codes=reviewed_candidate_codes,
        auto_confirm_shift_plan=bool(explicitly_named_occupant),
        auto_confirm_expected_shifted_words=(
            (explicitly_named_occupant,)
            if explicitly_named_occupant
            else ()
        ),
    )
    if response.startswith("✅ 操作已完成"):
        conversation_state_store.complete_execution(record)
    return response


def _exact_word_codes(data: Dict[str, Any], word: str) -> Tuple[str, ...]:
    """Keep only unique exact-word codes from one server lookup response."""
    phrases = data.get("phrases")
    if data.get("success") is not True or not isinstance(phrases, list):
        return ()
    return tuple(dict.fromkeys(
        str(item.get("code") or "").strip().lower()
        for item in phrases
        if isinstance(item, dict)
        and str(item.get("word") or "").strip() == word
        and re.fullmatch(
            r"[a-z]{1,6}",
            str(item.get("code") or "").strip().lower(),
        )
    ))


def _server_relative_chain(
    encoding: Dict[str, Any],
    current_code: str,
) -> Tuple[str, ...]:
    """Return the server series that contains the destination's current code."""
    for key in ("requestedCandidateCodes", "candidateCodes"):
        raw_codes = encoding.get(key)
        if not isinstance(raw_codes, list):
            continue
        codes = tuple(dict.fromkeys(
            str(value or "").strip().lower()
            for value in raw_codes
            if re.fullmatch(
                r"[a-z]{1,6}",
                str(value or "").strip().lower(),
            )
        ))
        if current_code in codes:
            return codes
    return ()


async def _resolve_relative_target_code(
    command: Any,
    platform: str,
    user_id: str,
) -> Tuple[str, str]:
    """Resolve before/after to an exact server-returned code, never a suffix."""
    word = str(command.word or "").strip()
    destination = str(command.named_occupant or "").strip()
    try:
        lookup = json.loads(await call_tool_function(
            "keytao_lookup_by_word",
            {"word": destination},
            platform,
            user_id,
        ))
    except Exception:
        lookup = {}
    destination_codes = _exact_word_codes(lookup, destination)
    if len(destination_codes) != 1:
        actual = "、".join(destination_codes) or "无现有编码"
        return "", (
            f"无法唯一确定「{destination}」的当前位置（{actual}）；"
            "本次未执行添加或顺延"
        )
    destination_code = destination_codes[0]
    if command.modifier in {"前面", "之前", "前"}:
        return destination_code, ""

    try:
        encoding = json.loads(await call_tool_function(
            "keytao_encode",
            {"word": word, "requested_code": destination_code},
            platform,
            user_id,
        ))
    except Exception:
        encoding = {}
    chain = (
        _server_relative_chain(encoding, destination_code)
        if encoding.get("success") is True
        else ()
    )
    if not chain or chain.index(destination_code) + 1 >= len(chain):
        return "", (
            f"服务端没有返回「{word}」在「{destination}」当前位置 "
            f"{destination_code} 后面的候选位；"
            "本次未执行添加或顺延"
        )
    return chain[chain.index(destination_code) + 1], ""


async def _handle_relative_position_command(
    command: Any,
    platform: str,
    user_id: str,
    conv_key: ConversationKey,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> str:
    """Bind a relative phrase placement to the existing server shift planner."""
    word = str(command.word or "").strip()
    destination = str(command.named_occupant or "").strip()
    if not word or not destination or word == destination:
        return "相对位置中的词条无法唯一确定；本次未执行添加或顺延"

    initial_record = conversation_state_store.get_record(conv_key)
    initial_state = initial_record.state if initial_record is not None else None
    had_live_subject = bool(
        initial_record is not None
        and not initial_record.execution_id
        and isinstance(initial_state, (PendingAddWord, PendingTrustedWordRecord))
        and initial_state.word == word
    )

    target_code, target_error = await _resolve_relative_target_code(
        command,
        platform,
        user_id,
    )
    if target_error:
        return target_error

    record = conversation_state_store.get_record(conv_key)
    state = record.state if record is not None else None
    if not (
        record is not None
        and not record.execution_id
        and isinstance(state, (PendingAddWord, PendingTrustedWordRecord))
        and state.word == word
    ):
        if record is not None and not record.execution_id:
            conversation_state_store.delete(conv_key)
        refreshed = await _try_handle_simple_single_word_query(
            f"加词 {word}",
            platform,
            user_id,
            conv_key,
            space_key,
            owner_label,
        )
        record = conversation_state_store.get_record(conv_key)
        state = record.state if record is not None else None
        if not isinstance(state, (PendingAddWord, PendingTrustedWordRecord)):
            return refreshed or (
                f"无法取得「{word}」的服务端词条或审词记录；"
                "本次未执行添加或顺延"
            )

    if isinstance(state, PendingAddWord):
        candidates = dict(state.server_candidates)
        if target_code not in candidates:
            return (
                f"{target_code} 不是「{word}」本轮服务端审词记录中的候选编码；"
                "本次未执行添加或顺延"
            )
        reviewed_pinyin, reviewed_candidate_codes = _pending_reviewed_reading(
            state,
            target_code,
        )
        if (
            not reviewed_pinyin
            or target_code not in reviewed_candidate_codes
        ):
            return (
                f"「{word}」的候选编码 {target_code} 没有保留对应读音；"
                "本次未执行添加或顺延"
            )
        create_args = _create_phrase_args(state, target_code)
        return await _execute_shift_to_code(
            word,
            target_code,
            platform,
            user_id,
            space_key,
            owner_label,
            target_item={
                "type": "Phrase",
                "remark": str(create_args.get("remark") or ""),
                "needsManualReview": bool(
                    create_args.get("needs_manual_review", True)
                ),
            },
            reviewed_pinyin=reviewed_pinyin,
            reviewed_candidate_codes=reviewed_candidate_codes,
            auto_confirm_shift_plan=not had_live_subject,
            auto_confirm_expected_shifted_words=(
                (destination,)
                if (
                    not had_live_subject
                    and command.modifier in {"前面", "之前", "前"}
                )
                else ()
            ),
            auto_confirm_only_if_no_shifts=bool(
                not had_live_subject
                and command.modifier in {"后面", "之后", "后"}
            ),
        )

    try:
        encoding = json.loads(await call_tool_function(
            "keytao_encode",
            {"word": word, "requested_code": target_code},
            platform,
            user_id,
        ))
    except Exception:
        encoding = {}
    if (
        encoding.get("success") is not True
        or target_code not in _server_relative_chain(encoding, target_code)
    ):
        return (
            f"{target_code} 不是「{word}」当前服务端编码链中的候选；"
            "本次未执行移动或顺延"
        )
    return await _execute_shift_to_code(
        word,
        target_code,
        platform,
        user_id,
        space_key,
        owner_label,
        auto_confirm_shift_plan=not had_live_subject,
        auto_confirm_expected_shifted_words=(
            (destination,)
            if (
                not had_live_subject
                and command.modifier in {"前面", "之前", "前"}
            )
            else ()
        ),
        auto_confirm_only_if_no_shifts=bool(
            not had_live_subject
            and command.modifier in {"后面", "之后", "后"}
        ),
    )


def _looks_like_submit_reconfirm_prompt(response: str) -> bool:
    """Detect a prior assistant message asking the user to reconfirm submission."""
    text = (response or "").strip()
    if not text or "提交" not in text or "加入草稿" in text:
        return False

    hints = (
        "是否继续提交",
        "确认提交",
        "继续提交吗",
        "继续提审",
        "确认后继续提交",
        "回复「确认」继续提交",
        "回复“确认”继续提交",
        "确认继续提交",
    )
    return any(hint in text for hint in hints)


def _parse_pending_state_from_response(response: str) -> PendingState:
    """Parse any pending operation represented by an assistant response."""
    batch_pending = _parse_pending_batch_add(response)
    if batch_pending is not None:
        return batch_pending

    pending_add = _parse_pending_add_word(response)
    if pending_add is not None:
        return pending_add

    return None


_CONTEXTUAL_SHORT_REPLIES = {
    "不用",
    "不用了",
    "不要",
    "不要了",
    "不需要",
    "不需要了",
    "先不用",
    "先不用了",
    "暂时不用",
    "暂时不用了",
    "算了",
    "不了",
    "不",
    "不加",
    "不加了",
    "不改",
    "不改了",
    "不用加",
    "不用加了",
    "不用改",
    "不用改了",
    "取消",
    "撤销",
    "要",
    "要的",
    "要加",
    "加",
    "加吧",
    "好",
    "好的",
    "好呀",
    "好啊",
    "行",
    "可以",
    "可以的",
    "可",
    "嗯",
    "嗯嗯",
    "是",
    "是的",
    "对",
    "对的",
    "确认",
    "同意",
    "就这样",
    "按这个",
    "这样",
    "这样加",
    "这么加",
    "都加",
    "选这个",
}


_CONTEXTUAL_REPLY_SUFFIXES = ("一下", "吧", "啦", "了", "哦", "喔", "呀", "呢", "哈", "嘛")

_CONTEXTUAL_NEGATIVE_REPLY_WHOLE_RE = re.compile(
    r"^(?:先|暂时)?(?:不|不用|不要|不需要|不了)(?:加|改)?(?:了)?(?:吧|啦)?$"
)
_CONTEXTUAL_POSITIVE_REPLY_WHOLE_RE = re.compile(
    r"^(?:要(?:的|加)?|加|可(?:以(?:的)?)?|嗯+|是(?:的)?|对(?:的)?|"
    r"这样(?:加)?|这么加|都加|选这个)(?:一下)?(?:吧|啦|了|哦|呀)?$"
)


_CONTEXTUAL_ASSISTANT_REPLY_HINTS = (
    "?",
    "？",
    "要这样",
    "要不要",
    "是否",
    "还是",
    "需要",
    "可以",
    "要加",
    "要改",
    "要哪个",
    "选哪个",
    "回复",
    "确认",
    "同意",
    "要我",
)


def _normalize_contextual_short_reply(message_text: str) -> str:
    """Normalize a short conversational reply without changing its meaning."""
    text = _strip_command_message_prefixes(message_text)
    text = re.sub(r"[\s，,。.!！?？~～…、;；:：\"'“”‘’（）()【】\[\]<>《》]+", "", text)
    return text.strip()


def _is_contextual_short_reply(message_text: str) -> bool:
    """Detect short replies that depend on the current user's own latest context."""
    text = _normalize_contextual_short_reply(message_text)
    if not text:
        return False
    assent = parse_pending_assent_phrase(message_text)
    if (
        assent.recognized
        or _CONTEXTUAL_NEGATIVE_REPLY_WHOLE_RE.fullmatch(text)
        or _CONTEXTUAL_POSITIVE_REPLY_WHOLE_RE.fullmatch(text)
    ):
        return True
    if re.fullmatch(r"\d{1,2}", text):
        return True
    if re.fullmatch(r"第?[一二三四五六七八九十两]+个?", text):
        return True

    return False


def _latest_assistant_message_invites_contextual_reply(history: Optional[List[Dict]]) -> bool:
    """Return true when the latest assistant turn is an open conversational prompt."""
    assistant_message = _get_latest_assistant_message(history)
    if not assistant_message:
        return False
    if _parse_pending_state_from_response(assistant_message) is not None:
        return False
    compact = re.sub(r"\s+", "", assistant_message)
    if not compact:
        return False
    return any(hint in compact for hint in _CONTEXTUAL_ASSISTANT_REPLY_HINTS)


def _is_contextual_reply_to_current_user_history(
    message_text: str,
    history: Optional[List[Dict]],
) -> bool:
    """Protect the sender's own short replies from another user's pending state."""
    return (
        _is_contextual_short_reply(message_text)
        and _latest_assistant_message_invites_contextual_reply(history)
    )


def _recover_pending_state_from_history(history: Optional[List[Dict]]) -> PendingState:
    """Never recreate an authorization ticket from assistant prose."""
    return None


def _recover_matching_pending_state_from_history(
    referenced_state: PendingState,
    history: Optional[List[Dict]],
) -> PendingState:
    """Never recreate an authorization ticket from stored assistant text."""
    return None


def _ensure_current_pending_from_referenced_owner(
    referenced_state: PendingState,
    referenced_owner_key: Optional[Tuple[str, str]],
    conv_key: ConversationKey,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> Optional[PendingStateRecord]:
    """Trust an explicit @owner on the quoted bot prompt for current-user ownership."""
    referenced_address = (
        normalize_conversation_key(referenced_owner_key, space_key)
        if referenced_owner_key is not None
        else None
    )
    current_address = normalize_conversation_key(conv_key, space_key)
    if referenced_state is None or referenced_address != current_address:
        return None

    current_record = conversation_state_store.get_record(conv_key)
    if (
        current_record is not None
        and conversation_state_store.states_equivalent(current_record.state, referenced_state)
    ):
        return current_record

    return None


def _record_from_referenced_owner(
    referenced_state: PendingState,
    referenced_owner_key: Optional[Tuple[str, str]],
    conv_key: ConversationKey,
    space_key: Optional[Tuple[str, str]],
) -> Optional[PendingStateRecord]:
    """Build an owner record from an explicit @owner on a quoted bot prompt."""
    if referenced_state is None or referenced_owner_key is None:
        return None
    referenced_address = normalize_conversation_key(referenced_owner_key, space_key)
    if referenced_address == normalize_conversation_key(conv_key, space_key):
        return None
    owner_record = conversation_state_store.get_record(referenced_address)
    if (
        owner_record is not None
        and conversation_state_store.states_equivalent(owner_record.state, referenced_state)
    ):
        return owner_record

    return None


def _ensure_current_pending_matches_reference(
    referenced_state: PendingState,
    conv_key: ConversationKey,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
    history: Optional[List[Dict]],
) -> Optional[PendingStateRecord]:
    """Restore current user's matching pending before checking other owners."""
    current_record = conversation_state_store.get_record(conv_key)
    if (
        current_record is not None
        and conversation_state_store.states_equivalent(current_record.state, referenced_state)
    ):
        return current_record

    return None


def _restore_current_pending_from_history_for_sensitive_control(
    command_intent: MessageCommandIntent,
    conv_key: ConversationKey,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
    history: Optional[List[Dict]],
) -> Optional[PendingStateRecord]:
    """Return an existing live ticket; history prose can never recreate one."""
    return conversation_state_store.get_record(conv_key)


async def _canonicalize_pending_ticket_intent(
    state: PendingState,
    message_text: str,
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
) -> Tuple[Optional[MessageCommandIntent], Optional[str]]:
    """Freeze the exact action and target that the ticket will later execute."""
    if not isinstance(state, PendingAddWord):
        return command_intent, None
    explicit = parse_explicit_code_request(
        _strip_command_message_prefixes(trusted_mutation_source(message_text)), state.word,
    )
    if explicit is not None and state.server_candidates:
        validation = validate_explicit_code(state, explicit.code, explicit.pinyin)
        if validation.valid:
            return command_intent, None
        return None, validation.reason + "；本次未写入。"

    multi_selection = parse_pending_candidate_selection(
        _strip_command_message_prefixes(trusted_mutation_source(message_text))
    )
    if multi_selection is not None:
        if multi_selection.indices:
            duplicate_indices = tuple(
                dict.fromkeys(
                    index
                    for index in multi_selection.indices
                    if multi_selection.indices.count(index) > 1
                )
            )
            if duplicate_indices:
                return None, render_remediation_reply(
                    "编号 " + "、".join(map(str, duplicate_indices))
                    + " 在同一计划中重复；每个编号只能出现一次；本次未写入",
                    command="加入",
                    words=(state.word,),
                )
            out_of_range = tuple(
                index
                for index in multi_selection.indices
                if not 1 <= index <= len(state.candidates)
            )
            if out_of_range:
                return None, render_remediation_reply(
                    "编号 " + "、".join(map(str, out_of_range))
                    + f" 超出当前候选范围 1-{len(state.candidates)}；"
                    + f"只接受 1-{len(state.candidates)}；本次未写入",
                    command="加入",
                    words=(state.word,),
                )
            requested_codes = tuple(
                state.candidates[index - 1][0]
                for index in multi_selection.indices
            )
            duplicate_codes = tuple(
                dict.fromkeys(
                    code
                    for code in requested_codes
                    if requested_codes.count(code) > 1
                )
            )
            if duplicate_codes:
                return None, render_remediation_reply(
                    "编码 " + "、".join(duplicate_codes)
                    + " 由多个所选编号重复解析；同一计划不能重复写入同一码；本次未写入",
                    command="加入",
                    words=(state.word,),
                )
            if multi_selection.recode_indices:
                if not (
                    state.server_candidates
                    and state.server_candidates == state.candidates
                ):
                    return None, render_remediation_reply(
                        "重新编码项缺少当前服务端候选记录；本次未写入",
                        command=f"加词 {state.word}",
                        words=(state.word,),
                    )
                if len(multi_selection.recode_indices) != 1:
                    return None, render_remediation_reply(
                        "一次复合选择只能包含一个「重新编码」项；本次未写入",
                        command="加入",
                        words=(state.word,),
                    )
                recode_index = multi_selection.recode_indices[0]
                recode_code, occupied = state.candidates[recode_index - 1]
                if not occupied:
                    return None, render_remediation_reply(
                        f"编号 {recode_index} 对应空位 {recode_code}，"
                        "不能使用「重新编码」；本次未写入",
                        command=f"加入 {recode_index}",
                        words=(state.word,),
                    )
        else:
            candidate_codes = {code for code, _occupied in state.candidates}
            duplicate_codes = tuple(
                dict.fromkeys(
                    code
                    for code in multi_selection.codes
                    if multi_selection.codes.count(code) > 1
                )
            )
            if duplicate_codes:
                return None, render_remediation_reply(
                    "编码 " + "、".join(duplicate_codes)
                    + " 在同一计划中重复；本次未写入",
                    command="加入",
                    words=(state.word,),
                )
            if any(code not in candidate_codes for code in multi_selection.codes):
                natural_assent = parse_pending_assent_phrase(message_text)
                if natural_assent.matched:
                    return replace(
                        command_intent,
                        intent=(
                            "pending_add_and_submit"
                            if natural_assent.submit_after
                            else "pending_confirm"
                        ),
                        submit_after=natural_assent.submit_after,
                        choice_index=None,
                        choice_indices=(),
                        recode_indices=(),
                        requested_code="",
                        requested_codes=(),
                    ), None
                return None, render_remediation_reply(
                    "所选编码不全在当前候选中",
                    command="加入",
                    words=(state.word,),
                )
            requested_codes = multi_selection.codes
        if (
            not multi_selection.submit_after
            and not multi_selection.recode_indices
            and len(multi_selection.indices) == 1
        ):
            return replace(
                command_intent,
                intent="pending_choice",
                submit_after=False,
                choice_index=multi_selection.indices[0],
                choice_indices=(),
                recode_indices=(),
                requested_code="",
                requested_codes=(),
            ), None
        if (
            not multi_selection.submit_after
            and len(multi_selection.codes) == 1
        ):
            return replace(
                command_intent,
                intent="pending_code_request",
                submit_after=False,
                choice_index=None,
                choice_indices=(),
                recode_indices=(),
                requested_code=multi_selection.codes[0],
                requested_codes=(),
            ), None
        return replace(
            command_intent,
            intent=(
                "pending_add_and_submit"
                if multi_selection.submit_after
                else "pending_choice"
            ),
            submit_after=multi_selection.submit_after,
            choice_index=None,
            choice_indices=multi_selection.indices,
            recode_indices=multi_selection.recode_indices,
            requested_code="",
            requested_codes=requested_codes,
        ), None

    requested_codes = tuple(
        _requested_codes_from_pending_message(message_text, state)
    )
    if requested_codes:
        command_intent = replace(
            command_intent,
            requested_codes=requested_codes,
        )

    if (
        command_intent.intent == "pending_add_and_submit"
        and command_intent.choice_index is not None
    ):
        if not 1 <= command_intent.choice_index <= len(state.candidates):
            return None, render_remediation_reply(
                f"只接受 1-{len(state.candidates)} 之间的编号",
                command="加入",
                words=(state.word,),
            )
        return command_intent, None

    if command_intent.intent == "pending_choice":
        closed_selection = _closed_candidate_selection(message_text)
        closed_indices = (
            closed_selection[0]
            if closed_selection is not None
            and not closed_selection[1]
            and closed_selection[2] is False
            else ()
        )
        choice_index = (
            closed_indices[0]
            if len(closed_indices) == 1
            else _parse_pending_choice_index(_compact_command_text(message_text))
        )
        if choice_index is None or not 1 <= choice_index <= len(state.candidates):
            return None, render_remediation_reply(
                f"只接受 1-{len(state.candidates)} 之间的编号",
                command="加入",
                words=(state.word,),
            )
        return replace(command_intent, choice_index=choice_index), None

    if command_intent.intent == "pending_recode":
        compact = _compact_command_text(message_text)
        ordinal_text = re.sub(r"(?:重新编码|挪开|顺延|改成)$", "", compact)
        parsed_choice = _parse_pending_choice_index(ordinal_text)
        canonical_intent = replace(
            command_intent,
            choice_index=parsed_choice or command_intent.choice_index,
        )
        target_code = _resolve_shift_target_code(state, canonical_intent)
        if target_code is None:
            return None, render_remediation_reply(
                "无法唯一确定要顺延的占用编码",
                command=f"加词 {state.word}",
                words=(state.word,),
            )
        target_index = next(
            (
                index
                for index, (candidate_code, _occupied) in enumerate(
                    state.candidates,
                    start=1,
                )
                if candidate_code == target_code
            ),
            None,
        )
        if target_index is None:
            return None, render_remediation_reply(
                "顺延目标不在当前候选中，本次未执行",
                command=f"加词 {state.word}",
                words=(state.word,),
            )
        return replace(
            canonical_intent,
            choice_index=target_index,
            target_word="",
        ), None

    if command_intent.intent == "pending_code_request":
        requested_target = await _resolve_requested_code_for_pending_add(
            state,
            command_intent.requested_code,
            platform,
            user_id,
        )
        if requested_target is None:
            return None, render_remediation_reply(
                "该编码不在当前候选中",
                command=f"加词 {state.word}",
                words=(state.word,),
            )
        target_code, _occupied = requested_target
        return replace(command_intent, requested_code=target_code), None

    return command_intent, None


async def _resolve_pending_ticket_control(
    state_record: PendingStateRecord,
    message_text: str,
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
    *,
    verified_bot_reply: bool = False,
) -> Tuple[MessageCommandIntent, Optional[str]]:
    """Resolve one actor-owned pending choice without exposing its internal nonce."""
    if not state_record.requires_reconfirmation:
        return command_intent, None
    if command_intent.intent == "pending_cancel":
        return command_intent, None

    structural_tool_intent = _pending_tool_assent_intent(
        state_record.state,
        message_text,
    )
    if (
        structural_tool_intent is not None
        and isinstance(state_record.state, PendingToolConfirm)
        and state_record.owner_key.actor_id == str(user_id)
    ):
        return structural_tool_intent, None

    compact_control = _compact_command_text(message_text)
    if compact_control.startswith("确认票据"):
        return MessageCommandIntent(), (
            "确认码已停用。\n"
            f"待确认：{_describe_pending_state(state_record.state)}。\n"
            f"{pending_confirmation_copy()}"
        )

    if (
        isinstance(state_record.state, PendingAddWord)
        and state_record.owner_key.actor_id == str(user_id)
        and command_intent.intent in _DIRECT_OWNER_PENDING_ADD_INTENTS
    ):
        canonical_intent, canonical_error = await _canonicalize_pending_ticket_intent(
            state_record.state,
            message_text,
            command_intent,
            platform,
            user_id,
        )
        if canonical_intent is None:
            return MessageCommandIntent(), canonical_error
        return canonical_intent, None

    if (
        isinstance(state_record.state, PendingToolConfirm)
        and state_record.owner_key.actor_id == str(user_id)
        and (
            (
                command_intent.intent == "pending_confirm"
                and _pending_tool_confirmation_matches(
                    state_record.state,
                    message_text,
                )
            )
            or (
                verified_bot_reply
                and command_intent.intent in {
                    "pending_confirm",
                    "pending_add_and_submit",
                }
                and _message_authorizes_pending_state_control(
                    state_record.state,
                    message_text,
                    command_intent,
                )
            )
        )
    ):
        return command_intent, None

    if _is_sensitive_pending_control_intent(command_intent):
        canonical_intent, canonical_error = await _canonicalize_pending_ticket_intent(
            state_record.state,
            message_text,
            command_intent,
            platform,
            user_id,
        )
        if canonical_intent is None:
            return MessageCommandIntent(), canonical_error
        confirmation_intent = _ticket_payload_from_command_intent(canonical_intent)
        if conversation_state_store.get_record(state_record.owner_key) is state_record:
            confirmation_code = conversation_state_store.arm_reconfirmation(
                state_record.owner_key,
                message_text,
                confirmation_intent=confirmation_intent,
                rotate_code=True,
            )
            if confirmation_code is None:
                return MessageCommandIntent(), render_remediation_reply(
                    "当前确认无法保存"
                )
        else:
            # Compatibility for isolated helpers that intentionally pass a
            # detached record; production records always take the durable path.
            confirmation_code = state_record.arm_reconfirmation(
                message_text,
                confirmation_intent=confirmation_intent,
                rotate_code=True,
            )
        description = _describe_pending_ticket_choice(
            state_record.state,
            canonical_intent,
        )
        return MessageCommandIntent(), (
            f"待确认：{description}。\n"
            f"{pending_confirmation_copy()}"
        )

    return command_intent, None


def _append_pending_ticket_challenge(
    response: str,
    conv_key: ConversationKey,
) -> str:
    """Bind the displayed prompt while keeping the internal nonce invisible."""
    record = conversation_state_store.get_record(conv_key)
    if (
        record is None
        or not isinstance(record.state, (PendingAddWord, PendingToolConfirm))
        or not record.requires_reconfirmation
    ):
        return response

    def bind_prompt(text: str) -> str:
        conversation_state_store.bind_origin_prompt_digest(
            conv_key,
            _prompt_capability_digest(text),
        )
        return text

    if isinstance(record.state, PendingAddWord) or record.state.function_name == _PENDING_CHOICE_OFFER_FUNCTION:
        return bind_prompt(response if "「取消」" in response else response.rstrip() + "\n或「取消」。")
    challenge = pending_confirmation_copy(record.state)
    response = response.replace(pending_confirmation_copy(), challenge)
    if pending_batch_confirmation_copy() in response:
        compact = response.replace(pending_confirmation_copy(), "").rstrip()
        if "「取消」" not in compact:
            compact += "或「取消」。"
        return bind_prompt(compact)
    if challenge in response:
        return bind_prompt(response)
    return bind_prompt(response.rstrip() + "\n\n" + challenge)


def _active_operation_message_for_request(
    operation: ActiveDraftOperation,
    platform: str,
    user_id: str,
) -> str:
    """Avoid exposing an operation's details outside its owning conversation."""
    memory_context = current_memory_context.get()
    request_address = (
        memory_context.conversation_address
        if memory_context is not None
        and memory_context.platform == platform
        and memory_context.user_id == user_id
        else ConversationAddress.private(platform, user_id)
    )
    if operation.owner_key != request_address:
        return render_remediation_reply(
            "你在另一个对话空间有草稿操作进行中；回到原对话属于站外切换"
        )
    return _format_active_draft_operation_message(operation)


_UNCERTAIN_TICKET_READ_RE = re.compile(
    r"^(?:查看(?:我的|当前)?草稿|草稿详情)$"
)
_UNCERTAIN_TICKET_READ_COMMANDS = frozenset({
    "查看草稿",
    "查看我的草稿",
    "查看当前草稿",
    "草稿详情",
})


def _resolve_uncertain_ticket_action(
    record: PendingStateRecord,
    message_text: str,
) -> Tuple[str, str]:
    """Allow read-only reconciliation or cancellation of one uncertain operation."""
    compact_message = re.sub(r"\s+", "", str(message_text or "")).strip()
    if _UNCERTAIN_TICKET_READ_RE.fullmatch(compact_message):
        return "read", ""

    cancellation = parse_pending_assent_phrase(compact_message)
    if (
        cancellation.rejection == "negation"
        and cancellation.cancel_requested
    ):
        conversation_state_store.complete_execution(record)
        return "discard", "已放弃上一次结果不确定的操作。"

    view_command = render_executable_suggestion("查看草稿")
    return (
        "block",
        "上一次操作正在执行或结果不确定，不会重复写入。\n"
        f"{view_command}\n或引用本消息回复「取消」。"
    )


def _format_other_owner_pending_message(
    owner_label: str,
    state: PendingState,
) -> str:
    description = _describe_pending_state(state)
    words = (
        (state.word,)
        if isinstance(state, PendingAddWord)
        else tuple(word for word, _code in pending_batch_display_pairs(state))
    )
    return render_remediation_reply(
        f"这是 {owner_label} 的待确认操作：{description}；你不能代为确认",
        command=("加词 " + " ".join(words)) if words else "",
        words=words,
    )


def _format_batch_display_mismatch(
    live_state: PendingState,
    referenced_state: PendingState,
) -> Optional[str]:
    """Name visible batch differences without treating quoted text as data."""
    if not (
        isinstance(live_state, PendingToolConfirm)
        and isinstance(referenced_state, PendingToolConfirm)
        and live_state.function_name
        == referenced_state.function_name
        == "keytao_batch_add_to_draft"
    ):
        return None
    live_pairs = pending_batch_display_pairs(live_state)
    referenced_pairs = pending_batch_display_pairs(referenced_state)
    if not live_pairs or not referenced_pairs or live_pairs == referenced_pairs:
        return None

    def render(pairs: Tuple[Tuple[str, str], ...]) -> str:
        preview = "、".join(
            f"「{word}」→ {code}"
            for word, code in pairs[:3]
        )
        if len(pairs) > 3:
            preview += f" 等 {len(pairs)} 条"
        return preview

    referenced_only = tuple(
        pair for pair in referenced_pairs if pair not in live_pairs
    )
    live_only = tuple(pair for pair in live_pairs if pair not in referenced_pairs)
    if referenced_only or live_only:
        differences = []
        if referenced_only:
            differences.append("引用显示 " + render(referenced_only))
        if live_only:
            differences.append("当前显示 " + render(live_only))
        detail = "；".join(differences)
    else:
        detail = (
            "展示顺序不同：引用为 "
            + render(referenced_pairs)
            + "；当前为 "
            + render(live_pairs)
        )
    live_words = tuple(word for word, _code in live_pairs)
    suggestion = render_executable_suggestion(
        f"将这 {len(live_words)} 个词加入草稿",
        words=live_words,
    )
    return (
        f"引用的批量加词与当前内容不一致：{detail}；本次未执行。\n"
        + suggestion
    )


def _handle_referenced_pending_from_other_user(
    referenced_state: PendingState,
    current_record: Optional[PendingStateRecord],
    other_record: Optional[PendingStateRecord],
    conv_key: ConversationKey,
    space_key: Tuple[str, str],
    owner_label: str,
    command_intent: MessageCommandIntent,
) -> Optional[str]:
    """Handle a user replying to a bot pending prompt that is not their own."""
    if referenced_state is None:
        return None
    if not _is_sensitive_pending_control_intent(command_intent):
        return None
    if current_record and conversation_state_store.states_equivalent(current_record.state, referenced_state):
        return None

    recode_requested = command_intent.intent == "pending_recode"
    if other_record is not None:
        return _format_other_owner_pending_message(
            _pending_owner_label(other_record),
            referenced_state,
        )

    if current_record is not None:
        mismatch = _format_batch_display_mismatch(
            current_record.state,
            referenced_state,
        )
        if mismatch is not None:
            return mismatch

    if recode_requested:
        return None

    referenced_words = (
        tuple(word for word, _code in pending_batch_display_pairs(referenced_state))
        if isinstance(referenced_state, PendingToolConfirm)
        else (
            (referenced_state.word,)
            if isinstance(referenced_state, PendingAddWord)
            else ()
        )
    )
    return render_remediation_reply(
        f"引用的待确认操作不可恢复：{_describe_pending_state(referenced_state)}",
        command=("加词 " + " ".join(referenced_words)) if referenced_words else "",
        words=referenced_words,
    )


async def _revalidate_referenced_add_pending(
    referenced_state: PendingAddWord,
    platform: str,
    user_id: str,
    *,
    selected_code: str = "",
    refresh_states: Optional[List[PendingAddWord]] = None,
    failure_reasons: Optional[List[str]] = None,
) -> Optional[PendingAddWord]:
    """Rebuild a quote while validating only its selected executable target."""
    def reject(reason: str) -> None:
        if failure_reasons is not None:
            failure_reasons.append(reason)
        logger.info(
            "[pending_add_revalidation] rejected word=%s code=%s reason=%s",
            referenced_state.word,
            referenced_state.recommended_code,
            reason,
        )
        return None

    review_json = await call_tool_function(
        "keytao_prepare_reviewed_add",
        {"word": referenced_state.word},
        platform,
        user_id,
    )
    try:
        review = json.loads(review_json)
    except Exception:
        return reject("当前审词服务没有返回可验证结果")
    if not isinstance(review, dict) or not review.get("success"):
        return reject("当前审词服务没有返回可验证结果")
    if review.get("pronunciationUnresolved"):
        return reject(f"「{referenced_state.word}」当前读音无法确定")
    current_word = str(review.get("word") or "").strip()
    if current_word != referenced_state.word:
        return reject(
            f"当前审词结果指向「{current_word or '未知词条'}」，"
            f"不再是「{referenced_state.word}」"
        )
    current_type = str(review.get("type") or "Phrase")
    if current_type != referenced_state.phrase_type:
        return reject("当前审词条目类型与原候选不一致")

    referenced_recommended = str(
        referenced_state.recommended_code or ""
    ).strip().lower()
    referenced_candidates = [
        (str(code).strip().lower(), bool(occupied))
        for code, occupied in referenced_state.candidates
    ]
    referenced_codes = [code for code, _occupied in referenced_candidates]
    selected_target_was_explicit = bool(str(selected_code or "").strip())
    selected_target = str(selected_code or referenced_recommended).strip().lower()
    if (
        not selected_target
        or selected_target not in referenced_codes
        or len(referenced_codes) != len(set(referenced_codes))
    ):
        return reject("引用候选缺少唯一、可验证的所选编码")
    referenced_occupancy = dict(referenced_candidates)

    inventory = select_candidate_inventory(review)
    if inventory is None:
        return reject("当前审词结果没有唯一、可验证的候选编码组")
    current_statuses = [dict(status) for status in inventory.statuses]
    current_candidates = list(inventory.candidates)
    current_pronunciation_codes = dict(inventory.readings)
    current_pronunciation_recommended_codes = list(dict.fromkeys(
        str(pronunciation.get("recommendedCode") or "").strip().lower()
        for pronunciation in review.get("pronunciations") or []
        if isinstance(pronunciation, dict)
        and str(pronunciation.get("recommendedCode") or "").strip()
    ))
    code_remarks: Dict[str, str] = {}
    current_recommended = str(
        review.get("recommendedCode") or ""
    ).strip().lower()
    current_codes = [code for code, _occupied in current_candidates]
    if not current_recommended or current_recommended not in current_codes:
        return reject("当前审词结果缺少可验证的推荐编码")
    audit_preview = _format_pre_submit_audit_preview(review, current_recommended)
    if not audit_preview:
        audit_preview = "自动审核：当前审词证据不足，需要管理员审核"
    for code, _occupied in current_candidates:
        pinyin = current_pronunciation_codes.get(code, "")
        source = "当前整词审读"
        for pronunciation in review.get("pronunciations") or []:
            if not isinstance(pronunciation, dict):
                continue
            pronunciation_codes = {
                str(status.get("code") or "").strip().lower()
                for status in pronunciation.get("candidateStatuses") or []
                if isinstance(status, dict)
            }
            if code in pronunciation_codes:
                source = _format_pronunciation_source(pronunciation)
                break
        review_parts = [
            f"读音 {pinyin}" if pinyin else "读音待确认",
            f"来源 {source}",
            audit_preview,
        ]
        code_remarks[code] = "喵喵审词：" + "；".join(review_parts)
    occupied_words: Dict[str, List[str]] = {}
    status_map = {
        str(status.get("code") or "").strip().lower(): status
        for status in current_statuses
    }
    for code, occupied in current_candidates:
        if not occupied:
            continue
        status = status_map[code]
        words = [
            str(word or "").strip()
            for word in status.get("words") or []
            if str(word or "").strip()
        ]
        if not words:
            words = [
                str(phrase.get("word") or "").strip()
                for phrase in status.get("phrases") or []
                if isinstance(phrase, dict) and str(phrase.get("word") or "").strip()
            ]
        occupied_words[code] = words
    current_ordering = _server_ordering_snapshot(
        PendingAddWord(
            word=referenced_state.word,
            recommended_code=current_recommended,
            candidates=current_candidates,
            occupied_words=occupied_words,
        ),
        current_candidates,
        occupied_words,
        review.get("candidateOrderingAssessments"),
    )
    current_reorder = validated_front_insert_recommendation(
        referenced_state.word,
        current_candidates,
        occupied_words,
        current_ordering,
    )
    effective_recommended = (
        current_reorder["occupantCode"]
        if current_reorder is not None
        else current_recommended
    )
    current_needs_manual_review = review_flags.read_manual_review_flag(review)
    if current_needs_manual_review is None:
        return reject("当前审词结果缺少可验证的审核结论")
    current_state = _attach_server_candidate_snapshot(PendingAddWord(
        word=referenced_state.word,
        recommended_code=effective_recommended,
        candidates=current_candidates,
        occupied_words=occupied_words,
        code_remarks=code_remarks,
        pronunciation_codes=current_pronunciation_codes,
        pronunciation_recommended_codes=current_pronunciation_recommended_codes,
        needs_manual_review=bool(current_needs_manual_review),
        manual_review_reason=review_flags.manual_review_reason(review),
        phrase_type=current_type,
    ), current_statuses, review.get("candidateOrderingAssessments"))

    def reject_with_refresh(reason: str) -> None:
        if refresh_states is not None:
            refresh_states.append(current_state)
        return reject(reason)

    if selected_target not in current_codes:
        return reject_with_refresh(
            f"所选编码 {selected_target} 已不在当前候选中"
        )
    current_occupancy = dict(current_candidates)
    if current_occupancy[selected_target] != referenced_occupancy[selected_target]:
        before = "已占用" if referenced_occupancy[selected_target] else "空位"
        after = "已占用" if current_occupancy[selected_target] else "空位"
        return reject_with_refresh(
            f"编码 {selected_target} 的占用状态已从{before}变为{after}"
        )
    if (
        not selected_target_was_explicit
        and effective_recommended != referenced_recommended
    ):
        return reject_with_refresh(
            f"「{referenced_state.word}」的推荐编码已从 {referenced_recommended} "
            f"变为 {effective_recommended or '不可解析'}"
        )

    pinyin = current_pronunciation_codes.get(selected_target, "")
    referenced_pinyin = str(
        referenced_state.pronunciation_codes.get(selected_target) or ""
    ).strip()
    normalized_pinyin = re.sub(r"\s+", " ", _plain_pinyin(pinyin)).strip()
    normalized_referenced_pinyin = re.sub(
        r"\s+",
        " ",
        _plain_pinyin(referenced_pinyin),
    ).strip()
    if not normalized_referenced_pinyin:
        return reject(
            f"引用候选没有保留编码 {selected_target} 对应的读音"
        )
    if not normalized_pinyin:
        return reject_with_refresh(
            f"编码 {selected_target} 的当前读音无法解析"
        )
    if normalized_pinyin != normalized_referenced_pinyin:
        return reject_with_refresh(
            f"编码 {selected_target} 的读音已从 {referenced_pinyin} 变为 {pinyin}"
        )

    if (
        referenced_state.needs_manual_review is not None
        and bool(referenced_state.needs_manual_review)
        != bool(current_needs_manual_review)
    ):
        before = (
            "需管理员审核"
            if referenced_state.needs_manual_review
            else "可自动通过"
        )
        after = "需管理员审核" if current_needs_manual_review else "可自动通过"
        return reject_with_refresh(f"审核结论已从{before}变为{after}")

    return current_state


def _ensure_pending_add_word_guidance(response: str) -> str:
    """Append deterministic guidance for occupied candidate choices."""
    if _parse_pending_batch_add(response) is not None:
        return ensure_multi_word_candidate_copy(response)

    response = re.sub(
        r"(?m)^若所选编号显示[“\"]已有…[”\"]，直接回复该编号表示添加重码；"
        r"回复[“\"]编号 重新编码[”\"]或[“\"]原词 重新编码[”\"]则挪开原词。\s*$",
        "",
        response,
    )
    response = re.sub(
        r"(?m)^若选的是已有词编码，回复[“\"]编号 重新编码[”\"]可挪开原词。\s*$",
        "",
        response,
    )
    response = re.sub(
        r"(?m)^若要挪开(?:已有词「[^」\n]+」|该已有词)，"
        r"回复[“\"][1-9]\d{0,2} 重新编码[”\"]。\s*$",
        "",
        response,
    )
    pending = _parse_pending_add_word(response)
    if pending is not None:
        response = ensure_single_word_candidate_copy(
            response,
            len(pending.candidates),
        )
    return response


def _build_existing_word_priority_note(word: str, lookup_entry: Dict, encode_data: Dict) -> Optional[str]:
    """Render bounded facts about an existing word without narrating a placement."""
    existing_codes = _existing_word_codes(lookup_entry)
    if not existing_codes:
        return None

    ordered_codes = _ordered_existing_word_codes(existing_codes, encode_data)
    primary_code = ordered_codes[0]
    facts = [f"• 「{word}」已在词库：{'、'.join(ordered_codes)}。"]

    primary_prior = _extract_prior_occupied_candidates(
        primary_code,
        encode_data,
        candidate_codes=_candidate_reading_group_codes(primary_code, encode_data),
    )
    if primary_prior:
        prior_text = "；".join(
            f"{item.get('code', '')} {item.get('label', '')}"
            for item in primary_prior[:3]
        )
        facts.append(f"• {primary_code} 前的候选码位已被占用：{prior_text}。")

    primary_phrase = next(
        (
            phrase
            for phrase in lookup_entry.get("phrases", [])
            if isinstance(phrase, dict)
            and str(phrase.get("code") or "").strip().lower() == primary_code
        ),
        None,
    )
    duplicate_info = (
        primary_phrase.get("duplicate_info")
        if isinstance(primary_phrase, dict)
        and isinstance(primary_phrase.get("duplicate_info"), dict)
        else None
    )
    if duplicate_info and len(duplicate_info.get("all_words", [])) > 1:
        position_label = duplicate_info.get("position_label") or "首位"
        duplicate_text = "、".join(
            (
                f"{item.get('word', '')}（{item.get('label')}）"
                if item.get("label")
                else str(item.get("word", ""))
            )
            for item in duplicate_info.get("all_words", [])[:5]
            if isinstance(item, dict) and item.get("word")
        )
        facts.append(
            f"• {primary_code} 这个码位里，{word} 排在{position_label}；"
            f"同码词有：{duplicate_text}。"
        )

    secondary_fact_count = sum(
        bool(_extract_prior_occupied_candidates(
            code,
            encode_data,
            candidate_codes=_candidate_reading_group_codes(code, encode_data),
        ))
        for code in ordered_codes[1:]
    )
    if secondary_fact_count:
        facts.append(
            f"• 其他已有编码还有 {secondary_fact_count} 组不同的前置占位情况，已省略明细。"
        )
    return "\n".join(dict.fromkeys(facts))


def _existing_word_codes(lookup_entry: Dict) -> Tuple[str, ...]:
    """Return each exact existing code once in lookup order."""
    return tuple(dict.fromkeys(
        str(phrase.get("code") or "").strip().lower()
        for phrase in lookup_entry.get("phrases", [])
        if isinstance(phrase, dict) and str(phrase.get("code") or "").strip()
    ))


def _ordered_existing_word_codes(
    existing_codes: Tuple[str, ...],
    encode_data: Dict,
) -> Tuple[str, ...]:
    """Lead with the recommended/current candidate when it is already present."""
    recommended = str(encode_data.get("recommendedCode") or "").strip().lower()
    status_order = tuple(
        str(item.get("code") or "").strip().lower()
        for item in encode_data.get("candidateStatuses", [])
        if isinstance(item, dict) and str(item.get("code") or "").strip()
    )
    preferred = [recommended] if recommended in existing_codes else []
    preferred.extend(code for code in status_order if code in existing_codes)
    preferred.extend(existing_codes)
    return tuple(dict.fromkeys(preferred))


def _candidate_reading_group_codes(
    current_code: str,
    encode_data: Dict,
) -> Tuple[str, ...]:
    """Find the one server-returned reading group that owns a candidate code."""
    grouped_codes: List[Tuple[str, ...]] = []
    for field in (
        "alternatePhrasePronunciationCodes",
        "alternatePronunciationCodes",
    ):
        for group in encode_data.get(field, []) or []:
            if not isinstance(group, dict):
                continue
            codes = tuple(
                str(code or "").strip().lower()
                for code in group.get("codes", []) or []
                if str(code or "").strip()
            )
            if codes:
                grouped_codes.append(codes)
    for group in encode_data.get("candidateDisplayGroups", []) or []:
        if not isinstance(group, dict):
            continue
        codes = tuple(
            str(item.get("code") or "").strip().lower()
            for item in group.get("items", []) or []
            if isinstance(item, dict) and str(item.get("code") or "").strip()
        )
        if codes:
            grouped_codes.append(codes)
    return next((codes for codes in grouped_codes if current_code in codes), ())


def _extract_prior_occupied_candidates(
    current_code: str,
    encode_data: Dict,
    *,
    candidate_codes: Tuple[str, ...] = (),
) -> List[Dict]:
    """Return occupied candidate slots before the current code."""
    candidate_statuses = encode_data.get("candidateStatuses", [])
    if not isinstance(candidate_statuses, list):
        return []
    if candidate_codes:
        allowed_codes = set(candidate_codes)
        candidate_statuses = [
            item for item in candidate_statuses
            if isinstance(item, dict)
            and str(item.get("code") or "").strip().lower() in allowed_codes
        ]
    current_index = next(
        (idx for idx, item in enumerate(candidate_statuses) if isinstance(item, dict) and item.get("code") == current_code),
        None,
    )
    if current_index is None or current_index <= 0:
        return []
    result = []
    for item in candidate_statuses[:current_index]:
        if not isinstance(item, dict) or not item.get("occupied"):
            continue
        result.append({
            "code": item.get("code", ""),
            "label": item.get("label", ""),
        })
    return result


def _extract_words_from_candidate_label(label: str) -> List[str]:
    """Extract occupied words from candidate label like 已有「甲、乙」."""
    if not label:
        return []
    match = re.search(r'已有「(.+?)」', label)
    if not match:
        return []
    return [part.strip() for part in match.group(1).split('、') if part.strip()]


async def _generate_usage_comparison_note(
    word: str,
    current_code: str,
    prior_occupied: List[Dict],
) -> Optional[str]:
    """Return only evidence-backed copy from the commonness comparator."""
    del current_code
    if not prior_occupied:
        return None

    occupied_words: List[str] = []
    for item in prior_occupied:
        occupied_words.extend(_extract_words_from_candidate_label(str(item.get("label", ""))))
    occupied_words = list(dict.fromkeys(
        occupied_word
        for occupied_word in occupied_words
        if occupied_word and occupied_word != word
    ))[:2]
    if not occupied_words:
        return None

    summaries: List[str] = []
    for occupied_word in occupied_words:
        try:
            comparison = await keytao_review.compare_word_commonness(
                word,
                occupied_word,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning(
                "Commonness comparator failed for %s/%s: %s",
                word,
                occupied_word,
                error,
            )
            continue
        if (
            not isinstance(comparison, dict)
            or not keytao_review._commonness_comparison_has_evidence(comparison)
        ):
            continue
        summary = render_commonness_summary(comparison)
        if summary:
            summaries.append(summary)
    if not summaries:
        return None
    combined = "；".join(summaries)
    keytao_review.record_commonness_evidence({"summary": combined})
    return combined


_CODE_APPENDIX_WORD_PATTERNS = (
    re.compile(
        r"^(?P<word>[\u3400-\u9fff]{1,20}?)(?:怎么|如何)(?:查)?(?:键道)?编码$"
    ),
    re.compile(
        r"^(?P<word>[\u3400-\u9fff]{1,20}?)(?:的)?(?:键道)?"
        r"(?:编码|码位|候选)(?:是什么|有哪些|情况)?$"
    ),
)


def _extract_code_appendix_query_words(message_text: str) -> Tuple[str, ...]:
    """Extract one explicit code-query target without relying on model routing."""
    text = re.sub(r"[？?！!。]+$", "", message_text.strip())
    original_text = text
    text = re.sub(r"^(?:请问|麻烦|帮我|帮忙|给我|查一下|看一下|看看)\s*", "", text)
    if text != original_text:
        note_rule_use("C:_extract_code_appendix_query_words:re.sub")
    for pattern in _CODE_APPENDIX_WORD_PATTERNS:
        match = pattern.fullmatch(text)
        if match is not None:
            if pattern is _CODE_APPENDIX_WORD_PATTERNS[0]:
                note_rule_use("C:_CODE_APPENDIX_WORD_PATTERNS[0]")
            word = match.group("word").strip()
            if word:
                return (word,)
    return ()


async def _augment_simple_word_query_response(
    message_text: str,
    response: str,
    platform: str,
    user_id: str,
    *,
    handled_as_command: bool = False,
    query_words: Optional[Tuple[str, ...]] = None,
) -> str:
    """Append deterministic code-priority notes for simple word-only queries."""
    if handled_as_command:
        return response
    if not _should_augment_simple_word_query(message_text, response):
        return response

    words = (
        await _get_simple_word_query_words(message_text)
        if query_words is None
        else tuple(query_words)
    )
    if not words:
        words = _extract_code_appendix_query_words(message_text)
    if not words:
        return response
    words = tuple(dict.fromkeys(words))

    lookup_json = await call_tool_function(
        "keytao_lookup_by_words_batch", {"words": list(words)}, platform, user_id,
    )
    try:
        lookup_data = json.loads(lookup_json)
    except Exception:
        return response
    if not lookup_data.get("success"):
        return response

    lookup_map = {
        item.get("word", ""): item
        for item in lookup_data.get("results", [])
        if isinstance(item, dict) and item.get("word")
    }

    note_blocks: List[str] = []
    seen_lines = {
        line.strip()
        for line in response.splitlines()
        if line.strip()
    }
    for word in words:
        lookup_entry = lookup_map.get(word, {})
        if not lookup_entry.get("phrases"):
            continue
        encode_json = await call_tool_function(
            "keytao_encode", {"word": word}, platform, user_id,
        )
        try:
            encode_data = json.loads(encode_json)
        except Exception:
            continue
        note = _build_existing_word_priority_note(word, lookup_entry, encode_data)
        note_lines = []
        if note:
            note_lines = [
                line for line in note.splitlines()
                if line.strip() and line.strip() not in response
            ]
        comparison_notes: List[str] = []
        existing_codes = _existing_word_codes(lookup_entry)
        ordered_codes = _ordered_existing_word_codes(existing_codes, encode_data)
        if ordered_codes:
            primary_code = ordered_codes[0]
            prior_occupied = _extract_prior_occupied_candidates(
                primary_code,
                encode_data,
                candidate_codes=_candidate_reading_group_codes(
                    primary_code,
                    encode_data,
                ),
            )
            comparison = await _generate_usage_comparison_note(
                word,
                primary_code,
                prior_occupied,
            )
            comparison_line = f"• 常用度对比：{comparison}" if comparison else ""
            if comparison_line and comparison_line not in response:
                comparison_notes.append(comparison_line)
        if note_lines or comparison_notes:
            block_parts = [f"{word} 的编码位置说明："]
            if note_lines:
                block_parts.extend(note_lines)
            if comparison_notes:
                block_parts.extend(comparison_notes)
            unique_block_parts = []
            for line in block_parts:
                formatted = line.strip()
                if not formatted or formatted in seen_lines:
                    continue
                unique_block_parts.append(formatted)
                seen_lines.add(formatted)
            if len(unique_block_parts) > 1:
                note_blocks.append("\n".join(unique_block_parts))

    if not note_blocks:
        return response
    has_supplement_header = any(
        line.strip() == "补充说明："
        for line in response.splitlines()
    )
    separator = "\n" if has_supplement_header else "\n\n补充说明：\n"
    return response.rstrip() + separator + "\n\n".join(note_blocks)


async def _try_handle_referenced_word_presence_query(
    message_text: str,
    reply_reference: ReplyReferenceInfo,
    platform: str,
    user_id: str,
) -> Optional[str]:
    """Answer word-presence questions strictly from the quoted message text."""
    if not _is_referenced_word_presence_query(message_text):
        return None
    if not reply_reference.is_reply:
        return None
    set_turn_flow("word-discovery")
    if not reply_reference.text:
        return render_remediation_reply(
            "平台未提供引用原文，无法确定查询目标"
        )

    expected_count = 2 if re.search(r"(两个|俩)", message_text) else 6
    if expected_count == 2:
        note_rule_use("C:_try_handle_referenced_word_presence_query:re.search")
    words = _extract_referenced_word_targets(reply_reference.text, expected_count=expected_count)
    if not words:
        return render_remediation_reply(
            "无法从引用消息中确定要查询的词，本次未查询"
        )

    lookup_json = await call_tool_function(
        "keytao_lookup_by_words_batch",
        {"words": words},
        platform,
        user_id,
    )
    try:
        lookup_data = json.loads(lookup_json)
    except Exception:
        lookup_data = {}

    if not lookup_data.get("success"):
        message = lookup_data.get("message") or lookup_data.get("error") or "词库查询暂时失败"
        return (
            f"查询 {'、'.join(f'「{word}」' for word in words)} 失败：{message}。"
        )

    return _format_referenced_word_presence_response(words, lookup_data)


_MAX_BARE_MULTI_WORD_QUERY_ITEMS = 15


def _bare_multi_word_query_words(message_text: str) -> Tuple[str, ...]:
    """Recognize a bounded lexical list without asking a model to route it."""
    source = _strip_command_message_prefixes(message_text).strip()
    if is_conversational_text(source):
        return ()
    if not re.fullmatch(r"[\u3400-\u9fff\s、，,；;]+", source):
        return ()
    if (
        parse_advertised_set_reference(source).matched
        or message_mentions_change_request(source)
    ):
        return ()
    tokens = re.split(r"[\s、，,；;]+", source)
    if len(tokens) == 1:
        # Preserve lexical compounds such as Republic; an unspaced conjunction
        # must have word-length operands on both sides. Ambiguity queries whole.
        parts = source.split("和")
        if len(parts) < 2 or any(len(part) < 2 for part in parts):
            return ()
        tokens = parts
    elif tokens[0] == "和" or tokens[-1] == "和":
        return ()
    words = tuple(dict.fromkeys(token for token in tokens if token != "和"))
    if len(words) < 2 or any(
        not _is_bare_word_query_target(word)
        for word in words
    ):
        return ()
    return words


def _render_pending_preserving_lookup(word, code, candidates, occupied, ordering) -> str:
    text = render_server_backed_single_word_lookup(
        word, code, candidates, occupied, ordering_assessments=ordering,
        actionable_controls=False, include_controls=False,
    )
    text = re.sub(r"(?m)^\s*\d+[.)、]\s*(?=[a-z]+\s*[—–-])", "", text)
    return text.replace("候选编码", "编码参考").replace("候选", "编码参考")


def _mergeable_add_candidate_scopes(
    platform: str, user_id: str, conv_key: Optional[ConversationKey],
    space_key: Optional[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    if draft_operation_coordinator.find_for_actor((platform, user_id)) is not None:
        return []
    address = normalize_conversation_key(conv_key or (platform, user_id), space_key)
    if address != normalize_conversation_key((platform, user_id), space_key):
        return []
    record = conversation_state_store.get_record(address)
    if record is None or record.execution_id or record.owner_key != address:
        return []
    state = record.state
    if isinstance(state, PendingAddWord) and state.server_candidates:
        return [{
            "word": state.word, "candidates": list(state.server_candidates),
            "occupiedWords": dict(state.server_occupied_words),
            "orderingAssessments": list(state.server_ordering_assessments),
            "reviewedState": _pending_add_word_payload(state),
            "reviewedPrompt": render_server_backed_single_word_lookup(
                state.word, state.recommended_code, state.server_candidates,
                state.server_occupied_words,
                ordering_assessments=state.server_ordering_assessments,
            ),
        }]
    if not (
        isinstance(state, PendingToolConfirm)
        and state.function_name == "keytao_batch_add_to_draft"
        and state.confirmation_source == "local_preview"
        and state.args.get("_reviewed_multi_word") is True
        and state.args.get("_explicit_add") is not False
        and not state.args.get("_resolved_advertised_words")
        and not state.args.get("_resolved_candidate_plan")
    ):
        return []
    unanswered = {
        item.get("word") for item in state.args.get("items", [])
        if isinstance(item, dict) and item.get("action") == "Create"
    }
    return [
        dict(scope) for scope in state.args.get("_candidate_scopes", [])
        if isinstance(scope, dict) and scope.get("word") in unanswered
    ]


def _explicit_add_candidate_words(message: str) -> Tuple[str, ...]:
    source = _strip_command_message_prefixes(message).strip()
    match = re.match(r"^(?:加词|添加)(?:\s*[:：]\s*|\s+)", source)
    if match is None:
        return ()
    words = _bare_multi_word_query_words(source[match.end():])
    single = _extract_explicit_reviewed_add_word(source) if not words else ""
    return words or ((single,) if single else ())


async def try_merge_add_candidates(
    message: str, platform: str, user_id: str, conv_key: Optional[ConversationKey],
    space_key: Optional[Tuple[str, str]], owner_label: str,
) -> Optional[str]:
    """Extend a live add inventory before semantic parsing can replace it."""
    words = _explicit_add_candidate_words(message)
    if not words or not _mergeable_add_candidate_scopes(platform, user_id, conv_key, space_key):
        return None
    return await _prepare_multi_word_query(
        words, platform, user_id, conv_key, space_key, owner_label, explicit_add=True,
    )


async def _prepare_multi_word_query(
    words: Tuple[str, ...],
    platform: str,
    user_id: str,
    conv_key: Optional[ConversationKey],
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
    *,
    reading_requests: tuple = (),
    preserve_pending: bool = False,
    query_other_blocks: Tuple[str, ...] = (),
    explicit_add: bool = False,
) -> str:
    """Collect the single-word pipeline into one durable candidate capability."""
    if len(words) > _MAX_BARE_MULTI_WORD_QUERY_ITEMS:
        return f"一次最多查询 {_MAX_BARE_MULTI_WORD_QUERY_ITEMS} 个词，请拆分后重新发送。"
    previous_scopes = (
        _mergeable_add_candidate_scopes(platform, user_id, conv_key, space_key)
        if explicit_add and not preserve_pending else []
    )
    combined_words = tuple(dict.fromkeys([*(scope["word"] for scope in previous_scopes), *words]))
    if len(combined_words) > _MAX_BARE_MULTI_WORD_QUERY_ITEMS:
        return (
            f"合并后超过 {_MAX_BARE_MULTI_WORD_QUERY_ITEMS} 个词；本次保留原候选"
            f"（{'、'.join(scope['word'] for scope in previous_scopes)}），请先选择后再加新词。"
        )
    scopes: List[Dict[str, Any]] = [scope for scope in previous_scopes if scope["word"] not in words]
    other_blocks: List[str] = list(query_other_blocks)
    other_block_words: Dict[int, str] = {}
    present_word_codes: Optional[Dict[str, Tuple[str, ...]]] = (
        {} if len(combined_words) > 1 else None
    )
    prepared_data: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
    for start in range(0, len(words), 50):
        chunk = list(words[start:start + 50])
        lookup_json = await call_tool_function(
            "keytao_lookup_by_words_batch", {"words": chunk}, platform, user_id,
        )
        try:
            lookup_data = json.loads(lookup_json)
        except Exception:
            lookup_data = {}
        lookup_rows = {
            row.get("word"): row for row in lookup_data.get("results", [])
            if isinstance(row, dict)
        } if lookup_data.get("success") else {}
        pending_json = await call_tool_function(
            "keytao_pending_items_by_words", {"words": chunk}, platform, user_id,
        )
        try:
            pending_data = json.loads(pending_json)
        except Exception:
            pending_data = {}
        for word in chunk:
            prepared_data[word] = (lookup_rows.get(word, {}), pending_data)
    semaphore = asyncio.Semaphore(2)

    async def prepare_word(word: str):
        async with semaphore:
            word_scopes: List[Dict[str, Any]] = []
            existing_words = {} if present_word_codes is not None else None
            binding_notices: List[bool] = []
            request = next((row for row in reading_requests if row.word == word), None)
            response = await _try_handle_simple_single_word_query(
                word, platform, user_id, conv_key, space_key, owner_label,
                _prepared_scopes=word_scopes,
                _prepared_existing_words=existing_words,
                _prepared_binding_notices=binding_notices,
                _prepared_data=prepared_data[word],
                requested_reading=request.reading if request else "",
                requested_code=request.code if request else "",
            )
            return word_scopes, existing_words, response, keytao_review.current_commonness_evidence(), binding_notices

    results = await asyncio.gather(*(prepare_word(word) for word in words), return_exceptions=True)
    for word, result in zip(words, results):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, BaseException):
            logger.warning(f"Per-word review failed for {word}: {result}")
            word_scopes, existing_words, response, evidence, binding_notices = [], None, None, (), []
        else:
            word_scopes, existing_words, response, evidence, binding_notices = result
        scopes.extend(word_scopes)
        if present_word_codes is not None:
            present_word_codes.update(existing_words or {})
        # Child task ContextVar writes must reach the parent delivery guard.
        keytao_review.record_commonness_evidence({"evidenceLines": list(evidence)})
        if binding_notices:
            actor_is_bound = await user_resolver.resolve_actor_binding(platform, user_id)
            response = append_unbound_binding_notice(response, actor_is_bound)
        if not word_scopes and not (present_word_codes is not None and word in present_word_codes):
            other_block_words[len(other_blocks)] = word
            other_blocks.append(str(response or f"「{word}」暂时没有可核验的候选。"))

    if present_word_codes:
        other_blocks.append(already_existing_words_copy(present_word_codes))

    actionable_scopes: List[Dict[str, Any]] = []
    if preserve_pending:
        for scope in scopes:
            other_blocks.append(_render_pending_preserving_lookup(
                scope["word"], scope["reviewedState"]["recommendedCode"],
                scope["candidates"], scope["occupiedWords"],
                scope["orderingAssessments"],
            ))
        return ServerBackedQueryReply("\n\n".join(other_blocks))
    for scope in scopes:
        if scope["reviewedState"]["recommendedCode"]:
            actionable_scopes.append(scope)
            continue
        diagnostic = render_server_backed_single_word_lookup(
            scope["word"], "", scope["candidates"], scope["occupiedWords"],
            reviewed_prompt=scope["reviewedPrompt"],
            ordering_assessments=scope["orderingAssessments"],
            actionable_controls=False, include_controls=False,
        )
        diagnostic_lines = []
        for line in diagnostic.splitlines():
            if "格式为：加入编码" in line:
                continue
            original_line = line
            line = re.sub(r"^\s*\d+[.)、]\s*(?=[a-z]+\s*[—–-])", "", line)
            if line != original_line:
                note_rule_use("C:_prepare_multi_word_query:re.sub")
            line = line.replace(f"「{scope['word']}」候选编码：", f"「{scope['word']}」占用情况：")
            diagnostic_lines.append(line)
        diagnostic_lines.append(
            f"「{scope['word']}」本次未选中；请单独查询此词，再指定完整编码或点名顶替。"
        )
        other_block_words[len(other_blocks)] = scope["word"]
        other_blocks.append("\n".join(diagnostic_lines))
    scopes = actionable_scopes

    if not scopes:
        if conv_key is not None:
            conversation_state_store.delete(conv_key)
        return "\n\n".join(other_blocks)
    from ..utils.pending_confirmation import advertised_reply_contract

    # Diagnostic instructions cannot borrow the resolved words' batch ticket.
    for index, word in other_block_words.items():
        if advertised_reply_contract(other_blocks[index]).requires_live_state:
            other_blocks[index] = f"「{word}」未能确认读音或候选编码，请提供读音或具体编码。"
    items: List[Dict[str, Any]] = []
    for scope in scopes:
        reviewed = scope["reviewedState"]
        code = reviewed["recommendedCode"]
        item = {
            "action": "Create", "word": scope["word"], "code": code,
            "type": reviewed.get("phraseType", "Phrase"), "remark": reviewed["codeRemarks"].get(code, ""),
        }
        review_flags.apply_manual_review_flag(
            item,
            reviewed["needsManualReview"] is not False,
            reviewed["manualReviewReason"],
        )
        items.append(item)
    pending = PendingToolConfirm(
        function_name="keytao_batch_add_to_draft",
        args={
            "items": items, "_candidate_scopes": scopes,
            "_reviewed_multi_word": True, "_query_words": list(combined_words),
            "_query_other_blocks": other_blocks,
            "_explicit_add": explicit_add,
        },
    )
    target_key = conv_key or (
        current_memory_context.get().conversation_address
        if current_memory_context.get() is not None
        else ConversationAddress.private(platform, user_id)
    )
    if not conversation_state_store.set(
        target_key, pending, space_key=space_key, owner_label=owner_label,
    ):
        return "这批词的候选编码暂时无法保存，请重新查询后再选择。"
    rendered = render_server_backed_batch_candidates(items, scopes)
    if not rendered:
        conversation_state_store.delete(target_key)
        return "这批词的候选展示校验未通过，本次未写入，请重新查询。"
    return ServerBackedQueryReply("\n\n".join([*other_blocks, rendered]))


async def _try_handle_simple_single_word_query(
    message_text: str,
    platform: str,
    user_id: str,
    conv_key: Optional[ConversationKey] = None,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
    *,
    requested_reading: str = "",
    requested_meaning: str = "",
    requested_code: str = "",
    actionable_lookup: bool = False,
    _prepared_scopes: Optional[List[Dict[str, Any]]] = None,
    _prepared_existing_words: Optional[Dict[str, Tuple[str, ...]]] = None,
    _prepared_binding_notices: Optional[List[bool]] = None,
    _prepared_data: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    preserve_pending: bool = False,
    query_only: bool = False,
    _missing_words: Optional[List[str]] = None,
    _literal_word: bool = False,
) -> Optional[str]:
    """Handle a single Chinese word add/query via tools before the model can invent codes."""
    from ..utils.reading_request import parse_parenthesised_readings

    explicit_words = (
        _explicit_add_candidate_words(message_text)
        if _prepared_scopes is None and not query_only and not _literal_word else ()
    )
    if len(explicit_words) > 1:
        return await _prepare_multi_word_query(
            explicit_words, platform, user_id, conv_key, space_key, owner_label,
            preserve_pending=preserve_pending, explicit_add=True,
        )
    reading_requests = (
        parse_parenthesised_readings(message_text)
        if _prepared_scopes is None and not query_only and not _literal_word else ()
    )
    if reading_requests:
        set_turn_flow("word-discovery")
        return await _prepare_multi_word_query(
            tuple(row.word for row in reading_requests), platform, user_id, conv_key, space_key, owner_label,
            reading_requests=reading_requests,
            preserve_pending=preserve_pending,
        )
    multi_words = (
        _bare_multi_word_query_words(message_text)
        if _prepared_scopes is None and not query_only and not _literal_word else ()
    )
    if multi_words:
        set_turn_flow("word-discovery")
        return await _prepare_multi_word_query(
            multi_words, platform, user_id, conv_key, space_key, owner_label,
            preserve_pending=preserve_pending,
        )
    explicit_add_word = (
        _extract_explicit_reviewed_add_word(message_text)
        if not query_only and not _literal_word else ""
    )
    if (
        explicit_add_word and _prepared_scopes is None and not preserve_pending
        and _mergeable_add_candidate_scopes(platform, user_id, conv_key, space_key)
    ):
        return await _prepare_multi_word_query(
            (explicit_add_word,), platform, user_id, conv_key, space_key, owner_label,
            explicit_add=True,
        )
    words = (
        (message_text,) if _prepared_scopes is not None or query_only or _literal_word
        else (explicit_add_word,) if explicit_add_word
        else await _get_simple_word_query_words(message_text)
    )
    if len(words) != 1:
        return None
    set_turn_flow("word-query" if query_only else "word-discovery")

    word = words[0]
    if _prepared_data is not None:
        lookup_data, pending_data = _prepared_data
    else:
        lookup_json = await call_tool_function(
            "keytao_lookup_by_word", {"word": word}, platform, user_id,
        )
        try:
            lookup_data = json.loads(lookup_json)
        except Exception:
            lookup_data = {}

        if query_only:
            pending_data = {"success": True, "complete": True, "items": []}
        else:
            pending_json = await call_tool_function(
                "keytao_pending_items_by_words",
                {"words": [word]},
                platform,
                user_id,
            )
            try:
                pending_data = json.loads(pending_json)
            except Exception:
                pending_data = {}
    if (
        pending_data.get("success") is not True
        or pending_data.get("complete") is not True
    ):
        return str(
            pending_data.get("message")
            or "暂时无法核验当前草稿和待审核批次，本次未继续"
        ).strip()
    pending_items = pending_data.get("items") or []
    reminder_lines = pending_word_reminder_lines(
        pending_items,
        words=(word,),
    )
    if reminder_lines and not explicit_add_word:
        if _prepared_scopes is not None:
            return prepend_pending_word_reminders("", pending_items, words=(word,))
        has_submitted = any(
            isinstance(item, dict) and item.get("source") == "submitted"
            for item in pending_items
        )
        action_copy = (
            "若要再加其他编码，需要提供实际编码；"
            "若需撤回，请先核对上述待审核批次。"
            if has_submitted
            else "若要再加其他编码，需要提供实际编码；"
            "提交前请先核对上述当前草稿。"
        )
        return prepend_pending_word_reminders(
            action_copy,
            pending_items,
            words=(word,),
        )

    if lookup_data.get("success") and lookup_data.get("phrases"):
        matching_rows = tuple(
            phrase
            for phrase in lookup_data.get("phrases") or []
            if isinstance(phrase, dict)
            and str(phrase.get("word") or word).strip() == word
            and str(phrase.get("code") or "").strip()
            and (len(word) != 1 or phrase.get("type") == "Single")
        )
        existing_codes = tuple(dict.fromkeys(
            str(phrase.get("code") or "").strip().lower()
            for phrase in matching_rows
        ))
        if existing_codes:
            if _prepared_existing_words is not None:
                _prepared_existing_words[word] = existing_codes
                return ""
            if conv_key is not None and _prepared_scopes is None and not preserve_pending and not query_only:
                phrase_types = tuple(dict.fromkeys(
                    str(phrase.get("type") or "").strip()
                    for phrase in matching_rows
                    if str(phrase.get("code") or "").strip().lower()
                    == existing_codes[0]
                    and str(phrase.get("type") or "").strip()
                ))
                trusted_row = (
                    create_pending_trusted_word_record(
                        word,
                        existing_codes[0],
                        phrase_types[0],
                        context_only=len(existing_codes) > 1,
                    )
                    if len(phrase_types) == 1
                    else None
                )
                if (
                    trusted_row is not None
                    and (
                        conversation_state_store.get_record(conv_key) is None
                        or isinstance(conversation_state_store.get(conv_key), PendingTrustedWordRecord)
                    )
                ):
                    conversation_state_store.set(
                        conv_key,
                        trusted_row,
                        space_key=space_key,
                        owner_label=owner_label,
                    )
            live_record = (
                conversation_state_store.get_record(conv_key)
                if conv_key is not None and _prepared_scopes is None else None
            )
            can_recode = bool(
                live_record is not None
                and not live_record.execution_id
                and isinstance(live_record.state, PendingTrustedWordRecord)
                and live_record.state.word == word
                and live_record.state.code in existing_codes
                and _pending_trusted_word_action_matches(live_record.state, "换码")
            )
            existing_reply = already_existing_word_copy(
                word,
                existing_codes,
                can_choose_other_code=_prepared_scopes is None and len(existing_codes) == 1,
                advertise_controls=can_recode,
            )
            if query_only:
                return ServerBackedQueryReply(existing_reply.splitlines()[0])
            response = ("类型：单字\n" + existing_reply) if len(word) == 1 else existing_reply
            if (
                live_record is not None
                and isinstance(live_record.state, PendingTrustedWordRecord)
                and live_record.state.word == word
                and live_record.state.code in existing_codes
                and not preserve_pending
            ):
                conversation_state_store.bind_origin_prompt_digest(
                    conv_key, _prompt_capability_digest(response),
                )
            return response
        if matching_rows or len(word) != 1:
            return "词库查询返回了不完整的已有词条记录；本次未继续。"

    if query_only:
        if lookup_data.get("success") is True and _missing_words is not None:
            _missing_words.append(word)
            return None
        return ServerBackedQueryReply(
            f"「{word}」未在词库中找到。" if lookup_data.get("success") is True
            else f"暂时无法查询「{word}」的词库记录。"
        )

    review_args = {"word": word}
    if requested_reading:
        review_args["requested_reading"] = requested_reading
    if requested_meaning:
        review_args["requested_meaning"] = requested_meaning
    review_json = await call_tool_function(
        "keytao_prepare_reviewed_add", review_args, platform, user_id,
    )
    try:
        review = json.loads(review_json)
    except Exception:
        review = {}

    # An initial annotation is context, not a second confirmation of a
    # syllable conflict. The existing explicit-reading follow-up resolves it.
    choice = review.get("multiSenseChoice") or {}
    if _prepared_scopes is not None and choice.get("differsFromAuthoritativeReading"):
        chains = choice.get("availableReadings") or []
        return (
            f"「{word}」标注的读音 {requested_reading} 与已审读音存在音节差异。\n"
            + "\n".join(f"- {row['pinyin']}：{'、'.join(row['codes'])}" for row in chains)
            + "\n请明确要采用的读音或具体含义；本次未写入。"
        )
    if requested_code and not review.get("pronunciationUnresolved"):
        inventory = select_candidate_inventory(review)
        if inventory is None or requested_code not in dict(inventory.candidates):
            chains = choice.get("availableReadings") or review.get("pronunciations") or []
            return (
                f"「{word}」标注的编码 {requested_code} 不在该读音的已审候选链中，本次未写入。\n"
                + "\n".join(
                    f"- {group.get('pinyin', '')}："
                    + "、".join(group.get("codes") or [
                        str(row.get("code") or "") for row in group.get("candidateStatuses") or []
                    ])
                    for group in chains
                )
                + "\n请明确要采用的读音或编码。"
            )
        review["recommendedCode"] = requested_code
        for group in review.get("pronunciations") or []:
            if any(row.get("code") == requested_code for row in group.get("candidateStatuses") or []):
                group["recommendedCode"] = requested_code

    reviewed_prompt = _format_reviewed_add_prompt(review)
    if reviewed_prompt:
        reviewed_prompt = prepend_pending_word_reminders(
            reviewed_prompt,
            pending_items,
            words=(word,),
        )
        pending = _parse_pending_add_word(reviewed_prompt)
        if pending is not None:
            pending.phrase_type = str(review.get("type") or "Phrase")
            if len(word) == 1 and pending.phrase_type != "Single":
                return f"「{word}」未取得可核验的单字编码，本次不展示可执行候选。"
            inventory = select_candidate_inventory(review)
            # Seal exactly the inventory chosen by the shared selector.  A
            # flattened payload can mix pronunciation groups and would make
            # the persisted ordinal map differ from the rendered list.
            server_statuses = (
                [dict(status) for status in inventory.statuses]
                if inventory is not None
                else []
            )
            _attach_server_candidate_snapshot(
                pending,
                server_statuses,
                review.get("candidateOrderingAssessments"),
            )
            if inventory is None or any(
                not str(inventory.readings.get(code) or "").strip()
                for code, _occupied in pending.server_candidates
            ):
                return render_remediation_reply(
                    "审词记录没有为每个候选保留读音，本次不展示可执行候选",
                    command=f"加词 {word}",
                    words=(word,),
                )
            pending.pronunciation_codes = dict(inventory.readings)
            if requested_reading:
                for code in inventory.readings:
                    pending.code_remarks[code] = (
                        pending.code_remarks.get(code, "")
                        + f"；用户标注读音 {requested_reading}"
                    ).lstrip("；")
            group_recommended = str(
                inventory.group_recommended_code or pending.recommended_code
            ).strip().lower()
            pending.pronunciation_recommended_codes = (
                [group_recommended] if group_recommended else []
            )
            if _prepared_scopes is not None:
                _prepared_scopes.append({
                    "word": pending.word,
                    "candidates": list(pending.server_candidates),
                    "occupiedWords": dict(pending.server_occupied_words),
                    "orderingAssessments": list(pending.server_ordering_assessments),
                    "reviewedState": _pending_add_word_payload(pending),
                    "reviewedPrompt": reviewed_prompt,
                    "requestedReading": requested_reading,
                })
                return ""
            if preserve_pending:
                return ServerBackedQueryReply(_render_pending_preserving_lookup(
                    pending.word, pending.recommended_code, pending.server_candidates,
                    pending.server_occupied_words,
                    pending.server_ordering_assessments,
                ))
            target_key = conv_key or (
                current_memory_context.get().conversation_address
                if current_memory_context.get() is not None
                else ConversationAddress.private(platform, user_id)
            )
            # Every displayed candidate selector is an executable contract.
            # Persist the exact shared inventory before rendering any form of
            # it, including a plain lexical lookup.
            if not conversation_state_store.set(
                target_key,
                pending,
                space_key=space_key,
                owner_label=owner_label,
            ):
                return (
                    f"「{word}」的候选编码暂时无法保存，请重新查询后再选择。"
                )
            if not explicit_add_word:
                read_only_candidates = render_server_backed_single_word_lookup(
                    pending.word,
                    pending.recommended_code,
                    pending.server_candidates,
                    pending.server_occupied_words,
                    ordering_assessments=pending.server_ordering_assessments,
                    reviewed_prompt=reviewed_prompt,
                    actionable_controls=True,
                )
                reviewed_prompt = read_only_candidates
                if not reviewed_prompt:
                    conversation_state_store.delete(target_key)
                    raise ValueError("persisted candidate inventory could not be rendered")
        if _prepared_binding_notices is not None:
            _prepared_binding_notices.append(True)
            return reviewed_prompt
        actor_is_bound = await user_resolver.resolve_actor_binding(platform, user_id)
        rendered_reply = append_unbound_binding_notice(reviewed_prompt, actor_is_bound)
        return (
            ServerBackedQueryReply(rendered_reply)
            if isinstance(reviewed_prompt, ServerBackedQueryReply)
            else rendered_reply
        )

    offline_reference = review.get("offlineReference")
    if (
        review.get("encodeServiceConfirmed") is False
        and isinstance(offline_reference, dict)
    ):
        readings = tuple(dict.fromkeys(
            str(item.get("pinyin") or "").strip()
            for item in offline_reference.get("readings") or []
            if isinstance(item, dict) and str(item.get("pinyin") or "").strip()
        ))
        candidate_codes = tuple(dict.fromkeys(
            str(code or "").strip().lower()
            for item in offline_reference.get("readings") or []
            if isinstance(item, dict)
            for code in item.get("candidateCodes") or []
            if re.fullmatch(r"[a-z]+", str(code or "").strip().lower())
        ))
        if candidate_codes:
            note_rule_use("C:_try_handle_simple_single_word_query:re.fullmatch")
        frequency = offline_reference.get("frequency")
        corpus_frequency = (
            frequency.get("corpusFrequency")
            if isinstance(frequency, dict)
            else None
        )
        failure_label = (
            "编码服务上游暂时不可用"
            if review.get("upstreamTransient") is True
            else "编码服务当前不可用"
        )
        lines = [f"「{word}」{failure_label}。"]
        if readings:
            lines.append(f"离线参考读音：{' / '.join(readings)}。")
        if candidate_codes:
            lines.append(
                "离线可推导候选链："
                f"{'、'.join(candidate_codes)}（仅音码基码；形码扩展无法离线确认）。"
            )
        elif readings:
            lines.append("离线参考无法可靠推导候选编码。")
        if (
            isinstance(corpus_frequency, int)
            and not isinstance(corpus_frequency, bool)
            and corpus_frequency > 0
        ):
            lines.append(f"离线词频：{corpus_frequency}（jieba）。")
        if not readings and not candidate_codes and not (
            isinstance(corpus_frequency, int) and corpus_frequency > 0
        ):
            lines.append("离线参考也未找到可用读音、词频或候选编码。")
        lines.append("以上均未经编码服务确认；本次不提供写入。")
        retry_copy = render_executable_suggestion(f"查词 {word}")
        if retry_copy:
            lines.append(retry_copy)
        return "\n".join(lines)

    review_message = str(
        review.get("message")
        or review.get("error")
        or "审词工具暂时没有返回可靠读音"
    ).strip()
    return review_message


_INJECT_PLATFORM_TOOLS = frozenset({
    'keytao_prepare_reviewed_add',
    'keytao_pending_items_by_words',
    'keytao_create_phrase', 'keytao_submit_batch',
    'keytao_list_draft_items', 'keytao_remove_draft_item',
    'keytao_update_draft_item_weight',
    'keytao_batch_add_to_draft', 'keytao_batch_remove_draft_items',
    'keytao_shift_phrase_code', 'keytao_recall_batch', 'keytao_get_batch_preview',
})


_DRAFT_MUTATION_TOOLS = frozenset({
    "keytao_create_phrase",
    "keytao_remove_draft_item",
    "keytao_update_draft_item_weight",
    "keytao_batch_add_to_draft",
    "keytao_batch_remove_draft_items",
    "keytao_shift_phrase_code",
    "keytao_recall_batch",
    "keytao_submit_batch",
})


def _guard_draft_mutation(
    context: ToolContext,
    tool_name: str,
    arguments: Dict,
) -> Optional[Dict]:
    if not context.platform or not context.user_id:
        return None
    operation = draft_operation_coordinator.find_for_actor(
        (context.platform, context.user_id)
    )
    if (
        operation is None
        or current_draft_operation_id.get() == operation.operation_id
    ):
        try:
            claim = get_default_draft_mutation_claim_store().get(
                context.platform,
                context.user_id,
            )
        except Exception as error:
            logger.error(
                "Failed to read draft mutation fence: %s: %s",
                type(error).__name__,
                error,
            )
            return {
                "success": False,
                "policyBlocked": True,
                "message": "无法核对草稿操作安全状态，本次未执行写入。",
            }
        if claim is None:
            return None
        operation_kind = str(claim.get("operationKind") or "")
        payload = claim.get("payload") if isinstance(claim, dict) else None
        claimed_batch_id = str((payload or {}).get("batchId") or "")
        if (
            operation_kind == "recall"
            and str(claim.get("status") or "") == "resolved"
            and tool_name in {
                "keytao_remove_draft_item",
                "keytao_batch_remove_draft_items",
            }
            and current_recall_clear_batch_id.get() == claimed_batch_id
            and str(arguments.get("batch_id") or "") == claimed_batch_id
        ):
            return None
        allowed_resolution_tools = {
            "recall": frozenset({"keytao_recall_batch"}),
            "delete": frozenset({
                "keytao_remove_draft_item",
                "keytao_batch_remove_draft_items",
            }),
        }
        if tool_name in allowed_resolution_tools.get(operation_kind, frozenset()):
            return None
        batch_id = claimed_batch_id
        return {
            "success": False,
            "uncertain": True,
            "operationInProgress": True,
            "policyBlocked": True,
            "batchId": batch_id,
            "message": render_remediation_reply(
                "上一次草稿写入结果仍在核验；已锁定原批次，"
                "不会执行新的草稿修改",
                command="查看草稿",
            ),
        }
    return {
        "success": False,
        "operationInProgress": True,
        "policyBlocked": True,
        "message": _active_operation_message_for_request(
            operation,
            context.platform,
            context.user_id,
        ),
    }


tool_executor = ToolExecutor(
    skills_manager.get_tool_function,
    _INJECT_PLATFORM_TOOLS,
    mutation_guard=_guard_draft_mutation,
    get_tool_schema=skills_manager.get_tool_schema,
)


_REVIEWED_ADD_VERDICT_TTL_SECONDS = 1800


_REVIEWED_ADD_VERDICT_MAX_ENTRIES = 256


_reviewed_add_verdicts: "OrderedDict[str, Tuple[float, bool, str]]" = OrderedDict()


def _record_reviewed_add_verdict(tool_name: str, arguments: Dict, result: str) -> None:
    if tool_name != "keytao_prepare_reviewed_add":
        return
    try:
        payload = json.loads(result)
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    word = str(payload.get("word") or (arguments or {}).get("word") or "").strip()
    flag = review_flags.read_manual_review_flag(payload)
    if not word or flag is None:
        return
    now = time.time()
    _reviewed_add_verdicts[word] = (
        now,
        bool(flag),
        review_flags.manual_review_reason(payload),
    )
    _reviewed_add_verdicts.move_to_end(word)
    while len(_reviewed_add_verdicts) > _REVIEWED_ADD_VERDICT_MAX_ENTRIES:
        _reviewed_add_verdicts.popitem(last=False)


def _take_reviewed_add_verdict(word: str) -> Tuple[Optional[bool], str]:
    normalized_word = str(word or "").strip()
    entry = _reviewed_add_verdicts.get(normalized_word)
    if not entry:
        return None, ""
    stored_at, flag, reason = entry
    if time.time() - stored_at > _REVIEWED_ADD_VERDICT_TTL_SECONDS:
        _reviewed_add_verdicts.pop(normalized_word, None)
        return None, ""
    return flag, reason


_DRAFT_RESOLUTION_TOOL_KINDS = {
    "keytao_recall_batch": "recall",
    "keytao_remove_draft_item": "delete",
    "keytao_batch_remove_draft_items": "delete",
}

_DRAFT_WRITE_RECEIPT_TOOLS = frozenset({
    "keytao_create_phrase",
    "keytao_batch_add_to_draft",
    "keytao_shift_phrase_code",
    "keytao_remove_draft_item",
    "keytao_batch_remove_draft_items",
    "keytao_update_draft_item_weight",
})


def _capture_successful_draft_write_delivery(
    tool_name: str,
    result: Dict[str, Any],
    platform: Optional[str],
    user_id: Optional[str],
) -> None:
    """Keep a successful operation receipt until its reply is delivered."""
    deliveries = current_draft_delivery_claims.get()
    if deliveries is None or not platform or not user_id:
        return
    pending_items = []
    if tool_name == "keytao_pending_items_by_words" and result.get("success") is True and result.get("complete") is True:
        pending_items = trusted_pending_word_items(result.get("items"))
    elif tool_name in _DRAFT_MUTATION_TOOLS and (result.get("success") is True or result.get("requiresConfirmation") is True):
        pending_items = trusted_pending_word_items(result.get("pendingItems"))
    if pending_items:
        read_receipt = {
            "platform": str(platform), "platformId": str(user_id),
            "operationKind": "pending_read", "toolName": tool_name, "pendingItems": pending_items,
        }
        if read_receipt not in deliveries:
            deliveries.append(read_receipt)
    is_write = tool_name in _DRAFT_WRITE_RECEIPT_TOOLS
    is_submit = tool_name == "keytao_submit_batch"
    if is_submit and result.get("success") is not True:
        return
    if not (is_write or is_submit) or not (result.get("success") is True or result.get("partialWrite") is True):
        return
    if result.get("requiresConfirmation") is True:
        return
    batch_id = str(result.get("batchId") or "").strip()
    receipt = {
        "platform": str(platform),
        "platformId": str(user_id),
        "operationKind": "draft_submit" if is_submit else "draft_write",
        "batchId": batch_id,
        "toolName": tool_name,
    }
    if is_submit:
        receipt["autoApproved"] = result.get("autoApproved") is True
    if "writtenItems" in result or "updatedItems" in result:
        receipt["writeReceipt"] = {
            **{key: result[key] for key in (
                "writtenItems", "updatedItems", "requestedItems", "requestedWords", "receiptItemsUnavailable",
                "failed", "skipped", "noWrite", "batchId", "batchUrl", "contentVersion", "warnings", "autoApproved",
            ) if key in result},
            "tool": tool_name,
        }
        for previous in deliveries:
            if (all(previous.get(key) == receipt[key] for key in ("platform", "platformId", "operationKind", "batchId"))
                    and previous.get("writeReceipt", {}).get("tool") == tool_name
                    and previous.get("writeReceipt", {}).get("writtenItems") == result.get("writtenItems")
                    and previous.get("writeReceipt", {}).get("updatedItems") == result.get("updatedItems")):
                previous.update(receipt)
                return
    if receipt not in deliveries:
        deliveries.append(receipt)


def _capture_resolved_mutation_delivery(
    tool_name: str,
    platform: Optional[str],
    user_id: Optional[str],
    result: Optional[Dict[str, Any]] = None,
) -> None:
    if result is not None and result.get("success") is not True:
        return
    deliveries = current_draft_delivery_claims.get()
    operation_kind = _DRAFT_RESOLUTION_TOOL_KINDS.get(tool_name)
    if deliveries is None or not operation_kind or not platform or not user_id:
        return
    try:
        claim = get_default_draft_mutation_claim_store().get(platform, user_id)
    except Exception as error:
        logger.error(
            "Failed to capture resolved draft receipt: %s: %s",
            type(error).__name__,
            error,
        )
        return
    if claim is None or str(claim.get("operationKind") or "") != operation_kind:
        return
    if str(claim.get("status") or "") != "resolved":
        deliveries[:] = [
            existing
            for existing in deliveries
            if not (
                existing.get("platform") == str(platform)
                and existing.get("platformId") == str(user_id)
                and existing.get("operationKind") not in {"pending_read", "dictionary_approval", "ticket_replacement"}
            )
        ]
        return
    receipt = {
        "platform": str(platform),
        "platformId": str(user_id),
        "operationKind": operation_kind,
        "fingerprint": str(claim.get("fingerprint") or ""),
        "toolName": tool_name,
    }
    deliveries[:] = [
        existing
        for existing in deliveries
        if not (
            existing.get("platform") == receipt["platform"]
            and existing.get("platformId") == receipt["platformId"]
            and existing.get("operationKind") not in {"pending_read", "dictionary_approval", "ticket_replacement"}
        )
    ]
    if receipt["fingerprint"]:
        deliveries.append(receipt)


def _acknowledge_delivered_draft_mutations(delivered_response: str = "") -> None:
    deliveries = current_draft_delivery_claims.get()
    if not deliveries:
        return
    mutation_claims = [
        receipt for receipt in deliveries
        if receipt.get("operationKind") in {"recall", "delete"}
    ]
    for receipt in mutation_claims:
        try:
            acknowledged = get_default_draft_mutation_claim_store().acknowledge(
                receipt["platform"],
                receipt["platformId"],
                receipt["operationKind"],
                receipt["fingerprint"],
            )
        except Exception as error:
            logger.error(
                "Failed to acknowledge delivered draft receipt: %s: %s",
                type(error).__name__,
                error,
            )
            continue
        if not acknowledged:
            logger.warning(
                "Draft receipt was sent but could not be acknowledged: %s:%s %s",
                receipt["platform"],
                receipt["platformId"],
                receipt["operationKind"],
            )
    memory_context = current_memory_context.get()
    if memory_context is not None:
        own_deliveries = [
            receipt for receipt in deliveries
            if receipt.get("platform") == memory_context.platform
            and receipt.get("platformId") == memory_context.user_id
        ]
        submitted_ids = {
            str(receipt.get("batchId") or "").strip()
            for receipt in own_deliveries
            if receipt.get("operationKind") == "draft_submit"
        }
        recent_batch_ids = tuple(dict.fromkeys(
            str(receipt.get("batchId") or "").strip()
            for receipt in own_deliveries
            if receipt.get("operationKind") == "draft_write"
            and str(receipt.get("batchId") or "").strip()
            and str(receipt.get("batchId") or "").strip() not in submitted_ids
        ))
        address = memory_context.conversation_address
        if recent_batch_ids and conversation_state_store.get_record(address) is None:
            from ..utils.draft_receipts import merge_receipt_deltas

            receipts = [
                receipt.get("writeReceipt") for receipt in own_deliveries
                if receipt.get("operationKind") == "draft_write"
                and receipt.get("batchId") in recent_batch_ids
            ]
            # Only one complete write can bind the later submit to an unchanged version.
            recent_items = []
            recent_version = None
            if len(recent_batch_ids) == 1 and len(receipts) == 1 and all(
                isinstance(receipt, dict)
                and ("writtenItems" in receipt or "updatedItems" in receipt)
                and not receipt.get("receiptItemsUnavailable")
                and type(receipt.get("contentVersion")) is int
                and receipt["contentVersion"] >= 0
                for receipt in receipts
            ):
                delta = merge_receipt_deltas(receipts)
                recent_items = [*delta["writtenItems"], *delta["updatedItems"]]
                recent_version = receipts[0]["contentVersion"]
            stored = conversation_state_store.set(
                address,
                PendingToolConfirm(
                    function_name="keytao_submit_batch",
                    args={
                        "batch_id": (
                            recent_batch_ids[0]
                            if len(recent_batch_ids) == 1
                            else ""
                        ),
                        "_recent_own_write": True,
                        "_recent_batch_ids": list(recent_batch_ids),
                        "_recent_written_items": recent_items,
                        "_recent_content_version": recent_version,
                    },
                    confirmation_source="local_preview",
                ),
                space_key=address.space_key,
                owner_label=memory_context.speaker_name,
            )
            if stored and delivered_response:
                conversation_state_store.bind_origin_prompt_digest(
                    address, _prompt_capability_digest(delivered_response),
                )
    deliveries.clear()


async def call_tool_function(
    tool_name: str,
    arguments: Dict,
    platform: Optional[str] = None,
    user_id: Optional[str] = None,
    *,
    trusted_reviewed_items_by_key: Optional[
        Dict[Tuple[str, str], Dict[str, Any]]
    ] = None,
) -> str:
    """Call a tool function and return result as JSON string."""
    if platform and user_id and tool_name in _DRAFT_MUTATION_TOOLS:
        operation = draft_operation_coordinator.find_for_actor((platform, user_id))
        if (
            operation is not None
            and current_draft_operation_id.get() != operation.operation_id
        ):
            logger.info(
                "[draft_operation] blocked out-of-band mutation "
                f"operation={operation.operation_id} owner={platform}:{user_id} tool={tool_name}"
            )
            return json.dumps({
                "success": False,
                "operationInProgress": True,
                "message": _active_operation_message_for_request(
                    operation,
                    platform,
                    user_id,
                ),
            }, ensure_ascii=False)
    result_json = await tool_executor.call(
        tool_name,
        arguments,
        ToolContext(
            platform,
            user_id,
            trusted_reviewed_items_by_key=trusted_reviewed_items_by_key,
        ),
    )
    _record_reviewed_add_verdict(tool_name, arguments, result_json)
    result_data: Optional[Dict[str, Any]] = None
    try:
        parsed_result = json.loads(result_json)
        if isinstance(parsed_result, dict):
            result_data = parsed_result
            operation_links = current_draft_result_links.get()
            if operation_links is not None and tool_name in AUTHORITATIVE_LINK_TOOLS:
                _capture_trusted_result_links(parsed_result, operation_links)
            _capture_resolved_mutation_delivery(tool_name, platform, user_id, parsed_result)
            _capture_successful_draft_write_delivery(
                tool_name,
                parsed_result,
                platform,
                user_id,
            )
    except (TypeError, ValueError):
        pass
    memory_context = current_memory_context.get()
    if memory_context is not None and result_data is not None:
        memory_store.record_tool_receipt(
            memory_context,
            tool_name,
            arguments,
            result_data,
            generation_token=current_memory_generation.get(),
        )
    return result_json


def _record_agent_tool_receipt(
    request_context: AgentRequestContext,
    tool_name: str,
    arguments: Dict,
    result: Dict,
    receipt_id: str,
) -> None:
    _capture_successful_draft_write_delivery(
        tool_name,
        result,
        request_context.platform,
        request_context.user_id,
    )
    memory_store.record_tool_receipt(
        ChatMemoryContext(
            platform=request_context.platform,
            user_id=request_context.user_id,
            space_type=request_context.space_type,
            space_id=request_context.space_id,
            speaker_name=request_context.speaker_name,
            target_user_id=request_context.target_user_id,
            target_name=request_context.target_name,
        ),
        tool_name,
        arguments,
        result,
        receipt_id=receipt_id,
        generation_token=current_memory_generation.get(),
    )


@dataclass(frozen=True)
class DraftActionResult:
    """Structured result for a draft mutation or submission."""
    text: str
    success: bool = False
    pending_state: Optional[PendingToolConfirm] = None
    data: Optional[Dict[str, Any]] = None
    invalidate_pending: bool = False


def _preserve_action_result_link(
    result: DraftActionResult,
    *fallback_sources: Dict,
    label: str = "草稿地址",
) -> DraftActionResult:
    """Carry an earlier trusted batch link through a later CAS result."""
    return replace(
        result,
        text=_append_batch_url_if_missing(
            result.text,
            result.data or {},
            *fallback_sources,
            label=label,
        ),
        data=result.data or next(
            (source for source in fallback_sources if isinstance(source, dict)),
            None,
        ),
    )


async def _execute_add_to_draft(
    word: str,
    code: str,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
    remark: str = "",
    needs_manual_review: Optional[bool] = None,
    auto_confirm: bool = True,
    reviewed_pinyin: str = "",
    reviewed_candidate_codes: Tuple[str, ...] = (),
) -> str:
    """Directly add a word to draft and return formatted response."""
    args = {"word": word, "code": code, "preview_only": True}
    if remark:
        args["remark"] = remark
    if needs_manual_review is not None:
        args["needs_manual_review"] = bool(needs_manual_review)
    reviewed_capability = _reviewed_create_capability(
        word,
        code,
        reviewed_pinyin,
        reviewed_candidate_codes,
    )
    if reviewed_capability is not None:
        result_json = await call_tool_function(
            "keytao_create_phrase",
            args,
            platform,
            user_id,
            trusted_reviewed_items_by_key=reviewed_capability,
        )
    else:
        result_json = await call_tool_function(
            "keytao_create_phrase", args, platform, user_id,
        )
    data = json.loads(result_json)

    if data.get("not_bound"):
        return _BIND_HELP_TEXT

    if data.get("requiresConfirmation"):
        conv_key = (platform, user_id)
        pending_args = dict(args)
        if reviewed_pinyin and reviewed_candidate_codes:
            pending_args["_reviewed_pinyin"] = reviewed_pinyin
            pending_args["_reviewed_candidate_codes"] = list(
                reviewed_candidate_codes
            )
        if data.get("pendingDuplicateConfirmation") is True:
            pending_state = pending_duplicate_confirmation_state(
                PendingToolConfirm("keytao_create_phrase", pending_args), data,
            )
        else:
            pending_state = _pending_state_from_server_warning(
                PendingToolConfirm(
                    function_name="keytao_create_phrase",
                    args=pending_args,
                ),
                data,
            )
        if auto_confirm and _create_preview_can_auto_confirm(data, args):
            return await _execute_confirmed_tool(
                pending_state,
                platform,
                user_id,
                conv_key,
                space_key,
                owner_label,
                carried_warnings=list(data.get("warnings") or []),
                carried_ordering_summary=_create_warning_ordering_summary(
                    data,
                    args,
                ),
            )
        conversation_state_store.set(
            conv_key,
            pending_state,
            space_key=space_key,
            owner_label=owner_label,
        )
        warnings = data.get("warnings", [])
        warn_text = (
            str(data.get("message") or "").strip()
            if data.get("pendingDuplicateConfirmation") is True
            else "\n".join(_plain_warning_line(w) for w in warnings)
            if warnings
            else data.get("message", "存在重码警告")
        )
        response = _append_batch_url_if_missing((
            f"{warn_text}\n{pending_confirmation_copy()}"
        ), data)
        return prepend_pending_word_reminders(
            response,
            data.get("pendingItems"),
            words=(word,),
        )

    if not data.get("success"):
        failure = _append_batch_url_if_missing(
            f"添加失败：{data.get('message', '未知错误')}",
            data,
        )
        return prepend_pending_word_reminders(
            failure,
            data.get("pendingItems"),
            words=(word,),
        )

    header = f"✅ 已将「{word}」以编码 {code} 加入草稿\n"
    response = header + await _format_draft_response(
        data, platform, user_id, compact=True,
    )
    return prepend_pending_word_reminders(
        response,
        data.get("pendingItems"),
        words=(word,),
    )


async def _execute_add_to_draft_and_submit(
    word: str,
    code: str,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
    remark: str = "",
    needs_manual_review: Optional[bool] = None,
    ordering_summary: str = "",
    reviewed_pinyin: str = "",
    reviewed_candidate_codes: Tuple[str, ...] = (),
) -> str:
    """Add a word to the draft, then submit the resulting batch."""
    result = await _perform_add_to_draft_and_submit(
        word,
        code,
        platform,
        user_id,
        remark=remark,
        needs_manual_review=needs_manual_review,
        auto_confirm=True,
        ordering_summary=ordering_summary,
        reviewed_pinyin=reviewed_pinyin,
        reviewed_candidate_codes=reviewed_candidate_codes,
    )
    if result.pending_state is not None:
        conversation_state_store.set(
            (platform, user_id),
            result.pending_state,
            space_key=space_key,
            owner_label=owner_label,
        )
    return result.text


def _create_preview_has_no_new_warnings(preview_data: Dict) -> bool:
    """Auto-replay only a complete create preview with no newly introduced risk."""
    content_version = preview_data.get("contentVersion")
    if (
        not str(preview_data.get("batchId") or "").strip()
        or not isinstance(content_version, int)
        or isinstance(content_version, bool)
        or content_version < 0
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(preview_data.get("warningDigest") or ""),
        )
    ):
        return False
    warnings = preview_data.get("warnings")
    if not isinstance(warnings, list) or warnings:
        return False
    warned_count = preview_data.get("warnedCount", 0)
    return (
        isinstance(warned_count, int)
        and not isinstance(warned_count, bool)
        and warned_count == 0
    )


def _create_preview_can_auto_confirm(
    preview_data: Dict,
    arguments: Dict,
) -> bool:
    """Accept either a no-warning snapshot or an exact informational warning."""
    return bool(
        _create_preview_has_no_new_warnings(preview_data)
        or create_warning_confirmation_binding(preview_data, arguments)
        or _explicit_additional_code_warning_matches(preview_data, arguments)
    )


def _explicit_additional_code_warning_matches(preview_data: Dict, arguments: Dict) -> bool:
    """A same-turn explicit additional code acknowledges only that exact warning."""
    capability = _current_explicit_additional_code.get()
    if (
        capability is None
        or (arguments.get("word"), arguments.get("code")) != (capability.word, capability.code)
        or arguments.get("action", "Create") != "Create"
        or arguments.get("type", capability.phrase_type) != capability.phrase_type
        or server_warning_confirmation_binding(preview_data) is None
        or not preview_data.get("warnings")
    ):
        return False
    for warning in preview_data["warnings"]:
        if not isinstance(warning, dict) or warning.get("warningType") != "multiple_code":
            return False
        item, existing = warning.get("item"), warning.get("existing")
        if (
            not isinstance(item, dict) or not isinstance(existing, dict)
            or (item.get("action"), item.get("word"), item.get("code"), item.get("type"))
            != ("Create", capability.word, capability.code, capability.phrase_type)
            or item.get("oldWord") not in (None, "")
            or item.get("remark", "") != arguments.get("remark", "")
            or item.get("needsManualReview") is not arguments.get("needs_manual_review")
            or existing.get("word") != capability.word
            or existing.get("code") not in capability.existing_codes
            or existing.get("code") == capability.code
            or existing.get("type", capability.phrase_type) != capability.phrase_type
        ):
            return False
    return True


def _create_warning_ordering_summary(data: Dict, arguments: Dict) -> str:
    """Describe the duplicate pair when no fuller server chain is available."""
    word = str(arguments.get("word") or "").strip()
    code = str(arguments.get("code") or "").strip().lower()
    if not word or not code:
        return ""
    existing_words: List[str] = []
    for warning in data.get("warnings") or []:
        if (
            not isinstance(warning, dict)
            or warning.get("warningType") != "duplicate_code"
        ):
            continue
        existing = warning.get("existing")
        if not isinstance(existing, dict):
            continue
        existing_code = str(existing.get("code") or "").strip().lower()
        existing_word = str(existing.get("word") or "").strip()
        if existing_code == code and existing_word and existing_word not in existing_words:
            existing_words.append(existing_word)
    if not existing_words:
        return ""
    return (
        f"{code}：{' → '.join([*existing_words, word])}"
        "（新词按默认权重排在后）"
    )


async def _perform_add_to_draft_and_submit(
    word: str,
    code: str,
    platform: str,
    user_id: str,
    *,
    remark: str = "",
    needs_manual_review: Optional[bool] = None,
    confirmed_create: bool = False,
    batch_id: str = "",
    expected_content_version: Optional[int] = None,
    expected_warning_digest: str = "",
    auto_confirm: bool = False,
    informational_warnings: Optional[List[Any]] = None,
    ordering_summary: str = "",
    reviewed_pinyin: str = "",
    reviewed_candidate_codes: Tuple[str, ...] = (),
) -> DraftActionResult:
    """Run add-and-submit without mutating conversational pending state."""
    args = {"word": word, "code": code}
    if remark:
        args["remark"] = remark
    if needs_manual_review is not None:
        args["needs_manual_review"] = bool(needs_manual_review)
    create_args = dict(args)
    if confirmed_create:
        create_args.update({
            "confirmed": True,
            "expected_content_version": expected_content_version,
            "expected_warning_digest": expected_warning_digest,
        })
    else:
        create_args["preview_only"] = True
    if batch_id:
        create_args["batch_id"] = batch_id
    reviewed_capability = _reviewed_create_capability(
        word,
        code,
        reviewed_pinyin,
        reviewed_candidate_codes,
    )
    if reviewed_capability is not None:
        create_json = await call_tool_function(
            "keytao_create_phrase",
            create_args,
            platform,
            user_id,
            trusted_reviewed_items_by_key=reviewed_capability,
        )
    else:
        create_json = await call_tool_function(
            "keytao_create_phrase", create_args, platform, user_id,
        )
    create_data = json.loads(create_json)
    pending_items = create_data.get("pendingItems") or []

    def with_pending_reminder(text: str) -> str:
        return prepend_pending_word_reminders(
            text,
            pending_items,
            words=(word,),
        )
    if create_data.get("success") is True and informational_warnings:
        create_data["warnings"] = list(informational_warnings)
        create_data["warnedCount"] = len(informational_warnings)
        create_data["autoConfirmedWarnings"] = True
    if create_data.get("success") is True and ordering_summary:
        create_data["orderingSummary"] = ordering_summary

    if create_data.get("not_bound"):
        return DraftActionResult(_BIND_HELP_TEXT)

    if create_data.get("requiresConfirmation"):
        warning_ordering_summary = (
            ordering_summary
            or _create_warning_ordering_summary(create_data, args)
        )
        pending_args = {**args, "_submit_after": True}
        if reviewed_pinyin and reviewed_candidate_codes:
            pending_args["_reviewed_pinyin"] = reviewed_pinyin
            pending_args["_reviewed_candidate_codes"] = list(
                reviewed_candidate_codes
            )
        if create_data.get("pendingDuplicateConfirmation") is True:
            pending_state = pending_duplicate_confirmation_state(
                PendingToolConfirm("keytao_create_phrase", pending_args), create_data,
            )
        else:
            pending_state = _pending_state_from_server_warning(
                PendingToolConfirm(
                    function_name="keytao_create_phrase",
                    args=pending_args,
                ),
                create_data,
            )
        if (
            auto_confirm
            and not confirmed_create
            and _create_preview_can_auto_confirm(create_data, args)
        ):
            exact_args = pending_state.args
            confirmed_result = await _perform_add_to_draft_and_submit(
                word,
                code,
                platform,
                user_id,
                remark=remark,
                needs_manual_review=needs_manual_review,
                confirmed_create=True,
                batch_id=str(exact_args.get("batch_id") or ""),
                expected_content_version=exact_args.get(
                    "expected_content_version"
                ),
                expected_warning_digest=str(
                    exact_args.get("expected_warning_digest") or ""
                ),
                auto_confirm=True,
                informational_warnings=list(create_data.get("warnings") or []),
                ordering_summary=warning_ordering_summary,
                reviewed_pinyin=reviewed_pinyin,
                reviewed_candidate_codes=reviewed_candidate_codes,
            )
            return _preserve_action_result_link(
                confirmed_result,
                create_data,
            )
        warnings = create_data.get("warnings", [])
        warn_text = (
            str(create_data.get("message") or "").strip()
            if create_data.get("pendingDuplicateConfirmation") is True
            else "\n".join(_plain_warning_line(w) for w in warnings)
            if warnings
            else create_data.get("message", "存在重码警告")
        )
        warning_response = _append_batch_url_if_missing(
            f"{warn_text}\n{pending_confirmation_copy()}",
            create_data,
        )
        warning_response = prepend_pending_word_reminders(
            warning_response,
            create_data.get("pendingItems"),
            words=(word,),
        )
        return DraftActionResult(
            warning_response,
            pending_state=pending_state,
            data=create_data,
        )

    if not create_data.get("success"):
        return DraftActionResult(
            with_pending_reminder(_append_batch_url_if_missing(
                f"添加失败：{create_data.get('message', '未知错误')}",
                create_data,
            )),
            data=create_data,
        )

    submit_batch_id = str(create_data.get("batchId") or "")
    create_notice_lines = _create_notice_lines(create_data)
    submit_result = await _perform_submit_current_draft(
        platform,
        user_id,
        batch_id=submit_batch_id,
        preview_only=True,
        auto_confirm=auto_confirm,
        authorized_items=[
            {
                "action": "Create",
                "word": word,
                "code": code,
                **(
                    {"needsManualReview": bool(needs_manual_review)}
                    if needs_manual_review is not None
                    else {}
                ),
            },
        ],
    )
    if pending_items and isinstance(submit_result.data, dict):
        submit_result.data.setdefault("pendingItems", pending_items)
    if submit_result.pending_state is not None:
        return DraftActionResult(
            with_pending_reminder(_append_batch_url_if_missing((
                f"✅ 「{word}」→ {code} 已加入草稿。\n"
                + ("\n".join(create_notice_lines) + "\n" if create_notice_lines else "")
                + submit_result.text
            ), create_data, submit_result.data or {})),
            pending_state=submit_result.pending_state,
            data=submit_result.data or create_data,
        )
    if submit_result.success:
        vote_reply = submitted_vote_receipt(submit_result.data or {})
        if vote_reply:
            return DraftActionResult(vote_reply, success=True, data=submit_result.data)
        submit_lines = [
            line for line in submit_result.text.splitlines()[1:]
            if line.strip()
        ]
        final_status = (
            f"✅ 已将「{word}」→ {code} 写入草稿，已提交审核并加入词库。"
            if isinstance(submit_result.data, dict)
            and submit_result.data.get("autoApproved") is True
            else f"✅ 已将「{word}」→ {code} 写入草稿，已提交审核。"
        )
        return DraftActionResult(
            with_pending_reminder(_append_batch_url_if_missing(
                "\n".join([
                    final_status,
                    *create_notice_lines,
                    *submit_lines,
                ]),
                submit_result.data or {},
                create_data,
                label="批次地址",
            )),
            success=True,
            data=submit_result.data or create_data,
        )
    return DraftActionResult(
        with_pending_reminder(_append_batch_url_if_missing(
            "\n".join([
                f"✅ 已将「{word}」以编码 {code} 加入草稿。",
                *create_notice_lines,
                "",
                submit_result.text,
            ]),
            create_data,
            submit_result.data or {},
        )),
        data=submit_result.data or create_data,
    )


async def _perform_batch_add_to_draft_and_submit(
    items: List[Dict],
    platform: str,
    user_id: str,
    *,
    batch_id: str = "",
    confirmed_add: bool = False,
    expected_content_version: Optional[int] = None,
    expected_warning_digest: str = "",
    auto_confirm: bool = False,
    allow_same_code: bool = False,
    reviewed_capabilities: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None,
) -> DraftActionResult:
    """Add every confirmed reviewed item, then submit exactly that batch."""
    requested_items = [dict(item) for item in items if isinstance(item, dict)]
    if not requested_items:
        return DraftActionResult(render_remediation_reply(
            "没有找到可添加的词条"
        ))

    add_arguments: Dict[str, Any] = {"items": requested_items}
    if allow_same_code:
        add_arguments["_allow_same_code"] = True
    if batch_id:
        add_arguments["batch_id"] = batch_id
    if confirmed_add:
        add_arguments.update({
            "confirmed": True,
            "expected_content_version": expected_content_version,
            "expected_warning_digest": expected_warning_digest,
        })
    else:
        add_arguments["preview_only"] = True
    add_json = await call_tool_function(
        "keytao_batch_add_to_draft",
        add_arguments,
        platform,
        user_id,
        **({"trusted_reviewed_items_by_key": reviewed_capabilities} if reviewed_capabilities else {}),
    )
    add_data = json.loads(add_json)
    pending_items = add_data.get("pendingItems") or []
    add_data["requestedWords"] = [item["word"] for item in requested_items if item.get("word")]

    def with_pending_reminder(text: str) -> str:
        return prepend_pending_word_reminders(
            text,
            pending_items,
            words=tuple(
                str(item.get("word") or "").strip()
                for item in requested_items
                if isinstance(item, dict)
            ),
        )
    if add_data.get("not_bound"):
        return DraftActionResult(_BIND_HELP_TEXT)

    effective_items = requested_items
    collision_replan_line = ""
    if add_data.get("collisionReplanned") is True:
        raw_effective = add_data.get("effectiveItems")
        if (
            not isinstance(raw_effective, list)
            or len(raw_effective) != len(requested_items)
        ):
            return DraftActionResult(render_remediation_reply(
                "同批改码结果不完整，本批未写入"
            ), data=add_data)
        validated_effective: List[Dict] = []
        changes: List[Tuple[str, str, str]] = []
        for requested, effective in zip(requested_items, raw_effective):
            if not isinstance(effective, dict):
                return DraftActionResult(render_remediation_reply(
                    "同批改码结果不完整，本批未写入"
                ), data=add_data)
            requested_action = str(requested.get("action") or "Create").strip()
            effective_action = str(effective.get("action") or "Create").strip()
            requested_word = str(requested.get("word") or "").strip()
            effective_word = str(effective.get("word") or "").strip()
            old_code = str(requested.get("code") or "").strip().lower()
            new_code = str(effective.get("code") or "").strip().lower()
            if (
                requested_action != effective_action
                or requested_word != effective_word
                or not re.fullmatch(r"[a-z]{1,6}", new_code)
            ):
                return DraftActionResult(render_remediation_reply(
                    "同批改码集合与原确认不一致，本批未写入"
                ), data=add_data)
            validated_effective.append(dict(effective))
            if old_code != new_code:
                changes.append((effective_word, old_code, new_code))
        short_create_codes = [
            str(item.get("code") or "").strip().lower()
            for item in validated_effective
            if str(item.get("action") or "Create") == "Create"
            and len(str(item.get("code") or "").strip()) < 6
        ]
        if not changes or len(short_create_codes) != len(set(short_create_codes)):
            return DraftActionResult(render_remediation_reply(
                "同批短码冲突仍未消除，本批未写入"
            ), data=add_data)
        effective_items = validated_effective
        collision_replan_line = "已调整：" + "；".join(
            f"「{word}」{old_code}→{new_code}"
            for word, old_code, new_code in changes
        ) + "（同批短码不重码）"

    if add_data.get("requiresConfirmation"):
        pending_args = {
            "items": effective_items,
            "_submit_after": True,
            **(
                {"_allow_same_code": True}
                if allow_same_code
                else {}
            ),
        }
        if reviewed_capabilities:
            pending_args["_reviewed_batch_readings"] = [
                {"word": word, "pinyin": capability["pinyin"],
                 "candidate_codes": list(capability["candidate_codes"])}
                for (word, _code), capability in reviewed_capabilities.items()
            ]
        if add_data.get("pendingDuplicateConfirmation") is True:
            pending_state = pending_duplicate_confirmation_state(
                PendingToolConfirm("keytao_batch_add_to_draft", pending_args), add_data,
            )
        else:
            pending_state = _pending_state_from_server_warning(
                PendingToolConfirm(
                    function_name="keytao_batch_add_to_draft",
                    args=pending_args,
                ),
                add_data,
            )
        if (
            auto_confirm
            and not confirmed_add
            and not collision_replan_line
            and batch_warning_confirmation_binding(
                add_data,
                {"items": requested_items},
            )
        ):
            exact_args = pending_state.args
            confirmed_result = await _perform_batch_add_to_draft_and_submit(
                requested_items,
                platform,
                user_id,
                batch_id=str(exact_args.get("batch_id") or ""),
                confirmed_add=True,
                expected_content_version=exact_args.get(
                    "expected_content_version"
                ),
                expected_warning_digest=str(
                    exact_args.get("expected_warning_digest") or ""
                ),
                auto_confirm=True,
                allow_same_code=allow_same_code,
                reviewed_capabilities=reviewed_capabilities,
            )
            return _preserve_action_result_link(
                confirmed_result,
                add_data,
            )
        warnings = add_data.get("warnings", [])
        warning_text = (
            str(add_data.get("message") or "").strip()
            if add_data.get("pendingDuplicateConfirmation") is True
            else "\n".join(_plain_warning_line(warning) for warning in warnings)
            if warnings
            else add_data.get("message", "批量添加前需要确认")
        )
        if collision_replan_line:
            warning_text = "\n".join(
                line for line in (collision_replan_line, warning_text) if line
            )
        warning_response = _append_batch_url_if_missing(
            f"{warning_text}\n{pending_confirmation_copy()}",
            add_data,
        )
        warning_response = prepend_pending_word_reminders(
            warning_response,
            add_data.get("pendingItems"),
            words=tuple(
                str(item.get("word") or "").strip()
                for item in effective_items
                if isinstance(item, dict)
            ),
        )
        return DraftActionResult(
            warning_response,
            pending_state=pending_state,
            data=add_data,
        )

    success_count = int(add_data.get("successCount") or 0)
    failed_count = int(add_data.get("failedCount") or 0)
    if success_count <= 0:
        return DraftActionResult(
            with_pending_reminder(_append_batch_url_if_missing(
                f"添加失败：{add_data.get('message', '未知错误')}",
                add_data,
            )),
            data=add_data,
        )

    if failed_count > 0 or success_count < len(effective_items):
        draft_text = await _format_draft_response(
            add_data, platform, user_id, compact=True,
        )
        return DraftActionResult(
            with_pending_reminder(
                f"仅成功加入 {success_count}/{len(effective_items)} 条，"
                f"已停止提交，避免生成不完整批次。\n\n{draft_text}"
            )
        )

    submit_result = await _perform_submit_current_draft(
        platform,
        user_id,
        batch_id=str(add_data.get("batchId") or ""),
        preview_only=True,
        auto_confirm=auto_confirm,
        authorized_items=effective_items,
    )
    item_lines = "\n".join(
        f"- 「{str(item.get('word') or '').strip()}」→ {str(item.get('code') or '').strip()}"
        for item in effective_items
        if item.get("word") and item.get("code")
    )
    if "writtenItems" in add_data:
        from ..utils.draft_receipts import receipt_change_lines
        item_lines = "\n".join(receipt_change_lines(add_data))
    text = submit_result.text
    if item_lines:
        text = f"{text}\n\n{item_lines}"
    return DraftActionResult(
        with_pending_reminder(_append_batch_url_if_missing(
            text,
            submit_result.data or {},
            add_data,
            label="批次地址" if submit_result.success else "草稿地址",
        )),
        pending_state=submit_result.pending_state,
        success=submit_result.success,
        data=submit_result.data or add_data,
    )


async def _execute_shift_to_code(
    word: str,
    target_code: str,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
    *,
    submit_after: bool = False,
    target_item: Optional[Dict[str, Any]] = None,
    additional_items: Optional[List[Dict[str, Any]]] = None,
    additional_shift_items: Optional[List[Dict[str, Any]]] = None,
    ordered_words: Optional[List[str]] = None,
    listed_words: Optional[List[str]] = None,
    expected_codes: Optional[List[str]] = None,
    expected_weights: Optional[List[int]] = None,
    evidence_lines: Optional[List[str]] = None,
    resolved_candidate_plan: Optional[List[Dict[str, Any]]] = None,
    reviewed_pinyin: str = "",
    reviewed_candidate_codes: Tuple[str, ...] = (),
    auto_confirm_shift_plan: bool = False,
    auto_confirm_expected_shifted_words: Tuple[str, ...] = (),
    auto_confirm_only_if_no_shifts: bool = False,
    expected_occupants_by_code: Optional[Dict[str, List[str]]] = None,
    receipt_requested_words: Optional[List[str]] = None,
) -> str:
    """Start a server-generated full-plan confirmation stage for a shift."""
    receipt_requested_words = receipt_requested_words or listed_words or ordered_words
    expected_occupants_by_code = expected_occupants_by_code or {
        str(item.get("code") or "").strip().lower(): [
            str(value or "").strip()
            for value in item.get("occupants") or []
            if str(value or "").strip()
        ]
        for item in resolved_candidate_plan or []
        if isinstance(item, dict)
        and str(item.get("modifier") or "").strip() == "recode"
        and str(item.get("code") or "").strip()
    }
    return await _execute_confirmed_tool(
        PendingToolConfirm(
            function_name="keytao_shift_phrase_code",
            args={
                "word": word,
                "target_code": target_code,
                **(
                    {
                        "target_type": str(target_item.get("type") or "Phrase"),
                        "target_remark": str(target_item.get("remark") or ""),
                        "target_needs_manual_review": bool(
                            target_item.get("needsManualReview", True)
                        ),
                    }
                    if isinstance(target_item, dict)
                    else {}
                ),
                **(
                    {"additional_items": [dict(item) for item in additional_items]}
                    if additional_items
                    else {}
                ),
                **(
                    {
                        "additional_shift_items": [
                            dict(item) for item in additional_shift_items
                        ]
                    }
                    if additional_shift_items
                    else {}
                ),
                **(
                    {"ordered_words": list(ordered_words)}
                    if ordered_words
                    else {}
                ),
                **(
                    {"listed_words": list(listed_words)}
                    if listed_words
                    else {}
                ),
                **(
                    {"expected_codes": list(expected_codes)}
                    if expected_codes
                    else {}
                ),
                **(
                    {"expected_weights": list(expected_weights)}
                    if expected_weights
                    else {}
                ),
                **(
                    {"evidence_lines": list(evidence_lines)}
                    if evidence_lines
                    else {}
                ),
                **(
                    {
                        "_resolved_candidate_plan": [
                            dict(item) for item in resolved_candidate_plan
                        ],
                    }
                    if resolved_candidate_plan
                    else {}
                ),
                **(
                    {
                        "_expected_occupants_by_code": (
                            expected_occupants_by_code
                        ),
                    }
                    if additional_shift_items and expected_occupants_by_code
                    else {}
                ),
                **(
                    {
                        "_reviewed_pinyin": reviewed_pinyin,
                        "_reviewed_candidate_codes": list(
                            reviewed_candidate_codes
                        ),
                    }
                    if reviewed_pinyin and reviewed_candidate_codes
                    else {}
                ),
                **({"_submit_after": True} if submit_after else {}),
                **({"_receipt_requested_words": list(receipt_requested_words)} if receipt_requested_words else {}),
                **(
                    {
                        "_auto_confirm_expected_shifted_words": list(
                            auto_confirm_expected_shifted_words
                        ),
                    }
                    if auto_confirm_expected_shifted_words
                    else {}
                ),
                **(
                    {"_auto_confirm_only_if_no_shifts": True}
                    if auto_confirm_only_if_no_shifts
                    else {}
                ),
            },
            confirmation_source="local_preview",
        ),
        platform,
        user_id,
        (platform, user_id),
        space_key,
        owner_label,
        auto_confirm_shift_plan=auto_confirm_shift_plan,
    )


def _lookup_status_occupied(encoding: Dict, code: str) -> bool:
    for status in encoding.get("candidateStatuses", []):
        if isinstance(status, dict) and status.get("code") == code:
            return bool(status.get("occupied"))
    return False


def _select_requested_code_candidate(word: str, requested_code: str, encoding: Dict) -> Optional[Tuple[str, bool]]:
    """Choose the actual candidate when the user supplied a code or phonetic prefix."""
    statuses = [
        status for status in encoding.get("candidateStatuses", [])
        if isinstance(status, dict) and isinstance(status.get("code"), str)
    ]
    status_codes = [status["code"] for status in statuses]
    candidate_codes = [
        code for code in encoding.get("candidateCodes", [])
        if isinstance(code, str)
    ]

    requested_series = [
        code for code in encoding.get("requestedCandidateCodes", [])
        if isinstance(code, str)
    ]
    if not requested_series:
        requested_series = [
            code for code in status_codes or candidate_codes
            if code.startswith(requested_code)
        ]

    if requested_series:
        # For a single character, a two-letter request is usually just the
        # phonetic route; continue along that route to the first empty slot.
        if len(word) == 1 and len(requested_code) == 2 and len(requested_series) > 1:
            for code in requested_series:
                if not _lookup_status_occupied(encoding, code):
                    return code, False
            fallback = requested_series[0]
            return fallback, _lookup_status_occupied(encoding, fallback)

        if requested_code in requested_series:
            return requested_code, _lookup_status_occupied(encoding, requested_code)

        for code in requested_series:
            if not _lookup_status_occupied(encoding, code):
                return code, False
        fallback = requested_series[0]
        return fallback, _lookup_status_occupied(encoding, fallback)

    if requested_code in status_codes or requested_code in candidate_codes:
        return requested_code, _lookup_status_occupied(encoding, requested_code)

    return None


def _requested_codes_from_pending_message(message: str, state: PendingAddWord) -> List[str]:
    candidate_codes = [code for code, _ in state.candidates]
    candidate_set = set(candidate_codes)
    requested = [
        token.lower()
        for token in re.findall(r"\b[a-z]{2,12}\b", message.lower())
        if token.lower() in candidate_set
    ]
    if requested:
        result: List[str] = []
        seen = set()
        for code in requested:
            if code not in seen:
                seen.add(code)
                result.append(code)
        return result

    normalized = message.strip()
    if (
        state.pronunciation_recommended_codes
        and any(marker in normalized for marker in ("都加", "全加", "全部", "都可以", "都要"))
    ):
        return [
            code for code in state.pronunciation_recommended_codes
            if code in candidate_set
        ]

    return []


async def _execute_add_multiple_codes_to_draft(
    state: PendingAddWord,
    codes: List[str],
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
    submit_after: bool = False,
) -> str:
    items = []
    occupancy = dict(state.candidates)
    for code in codes:
        item = {"word": state.word, "code": code, "action": "Create"}
        remark = state.code_remarks.get(code)
        if remark:
            item["remark"] = remark
        if occupancy.get(code) is True:
            item["needsManualReview"] = True
            item["manualReviewReason"] = "重码添加需管理员审核"
        elif state.needs_manual_review is not None:
            item["needsManualReview"] = bool(state.needs_manual_review)
        items.append(item)
    if not items:
        return render_remediation_reply(
            "没有找到可添加的编码",
            command=f"加词 {state.word}",
            words=(state.word,),
        )

    return await _execute_confirmed_tool(
        PendingToolConfirm(
            function_name="keytao_batch_add_to_draft",
            args={
                "items": items,
                **({"_submit_after": True} if submit_after else {}),
            },
            confirmation_source="local_preview",
        ),
        platform,
        user_id,
        (platform, user_id),
        space_key,
        owner_label,
    )


async def _resolve_requested_code_for_pending_add(
    state: PendingAddWord,
    requested_code: str,
    platform: str,
    user_id: str,
) -> Optional[Tuple[str, bool]]:
    if not requested_code:
        return None

    normalized_requested_code = requested_code.strip().lower()
    for candidate_code, occupied in state.candidates:
        if candidate_code == normalized_requested_code:
            return candidate_code, occupied

    result_json = await call_tool_function(
        "keytao_encode",
        {"word": state.word, "requested_code": normalized_requested_code},
        platform,
        user_id,
    )
    try:
        encoding = json.loads(result_json)
    except json.JSONDecodeError:
        return None

    if not encoding.get("success"):
        return None

    return _select_requested_code_candidate(
        state.word,
        normalized_requested_code,
        encoding,
    )


def _resolved_advertised_items_match(state: PendingToolConfirm) -> bool:
    """Fail closed if a snapshot-derived ticket no longer seals the exact set."""
    if "_resolved_advertised_words" not in state.args:
        return True
    expected = state.args.get("_resolved_advertised_words")
    items = state.args.get("items")
    if (
        state.function_name != "keytao_batch_add_to_draft"
        or not isinstance(expected, list)
        or not expected
        or not all(isinstance(word, str) and word.strip() for word in expected)
        or len(set(expected)) != len(expected)
        or not isinstance(items, list)
        or len(items) != len(expected)
    ):
        return False
    actual: List[str] = []
    for item in items:
        if (
            not isinstance(item, dict)
            or item.get("action") != "Create"
            or item.get("old_word")
            or item.get("oldWord")
            or not isinstance(item.get("word"), str)
            or not str(item.get("word") or "").strip()
        ):
            return False
        actual.append(str(item["word"]).strip())
    return actual == expected


def _resolved_candidate_plan_matches(state: PendingToolConfirm) -> bool:
    """Keep a compound candidate selection identical through preview and confirm."""
    if "_resolved_candidate_plan" not in state.args:
        return True
    plan = state.args.get("_resolved_candidate_plan")
    additional_items = state.args.get("additional_items")
    additional_shift_items = state.args.get("additional_shift_items")
    if (
        state.function_name != "keytao_shift_phrase_code"
        or not isinstance(plan, list)
        or len(plan) < 2
        or (
            not isinstance(additional_items, list)
            and not isinstance(additional_shift_items, list)
        )
    ):
        return False
    normalized: List[Tuple[int, str, str, str]] = []
    for raw_item in plan:
        if not isinstance(raw_item, dict):
            return False
        index = raw_item.get("index")
        word = str(raw_item.get("word") or "").strip()
        code = str(raw_item.get("code") or "").strip().lower()
        modifier = str(raw_item.get("modifier") or "").strip()
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 1
            or not word
            or not re.fullmatch(r"[a-z]{1,12}", code)
            or modifier not in {"", "recode"}
        ):
            return False
        normalized.append((index, word, code, modifier))
    if (
        len({index for index, _word, _code, _modifier in normalized})
        != len(normalized)
        or len({code for _index, _word, code, _modifier in normalized})
        != len(normalized)
    ):
        return False
    recode_items = [item for item in normalized if item[3] == "recode"]
    if not recode_items:
        return False
    _index, recode_word, recode_code, _modifier = recode_items[0]
    if (
        recode_word != str(state.args.get("word") or "").strip()
        or recode_code != str(state.args.get("target_code") or "").strip().lower()
    ):
        return False
    if isinstance(additional_shift_items, list):
        if len(recode_items) != len(normalized):
            return False
        expected_shifts = [
            (word, code)
            for _index, word, code, _modifier in normalized[1:]
        ]
        actual_shifts: List[Tuple[str, str]] = []
        for item in additional_shift_items:
            if (
                not isinstance(item, dict)
                or str(item.get("action") or "Create") != "Create"
                or item.get("old_word")
                or item.get("oldWord")
            ):
                return False
            actual_shifts.append((
                str(item.get("word") or "").strip(),
                str(item.get("code") or "").strip().lower(),
            ))
        return actual_shifts == expected_shifts

    if len(recode_items) != 1 or not isinstance(additional_items, list):
        return False
    expected_additional = [
        (word, code)
        for _index, word, code, modifier in normalized
        if modifier != "recode"
    ]
    actual_additional: List[Tuple[str, str]] = []
    for item in additional_items:
        if (
            not isinstance(item, dict)
            or str(item.get("action") or "Create") != "Create"
            or item.get("old_word")
            or item.get("oldWord")
        ):
            return False
        actual_additional.append((
            str(item.get("word") or "").strip(),
            str(item.get("code") or "").strip().lower(),
        ))
    return actual_additional == expected_additional


def _chain_reorder_plan_digest(plan: Dict[str, Any]) -> str:
    sealed = {key: value for key, value in plan.items() if key != "digest"}
    payload = json.dumps(
        sealed,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _chain_reorder_expected_items(plan: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    code = str(plan.get("code") or "").strip().lower()
    groups = plan.get("groups")
    if not re.fullmatch(r"[a-z]{2,12}", code) or not isinstance(groups, list) or not groups:
        return None
    expected: List[Dict[str, Any]] = []
    seen = set()
    for group in groups:
        if not isinstance(group, dict):
            return None
        phrase_type = str(group.get("type") or "").strip()
        base_weight = _PHRASE_TYPE_BASE_WEIGHTS.get(phrase_type)
        current = group.get("current")
        proposed = group.get("proposed")
        if (
            base_weight is None
            or not isinstance(current, list)
            or not isinstance(proposed, list)
            or not current
            or len(current) != len(proposed)
        ):
            return None
        current_by_word: Dict[str, Dict[str, Any]] = {}
        for entry in current:
            if not isinstance(entry, dict):
                return None
            word = str(entry.get("word") or "").strip()
            weight = entry.get("weight")
            identity = (phrase_type, word)
            if (
                not word
                or entry.get("code") != code
                or entry.get("type") != phrase_type
                or not isinstance(weight, int)
                or isinstance(weight, bool)
                or weight < base_weight
                or identity in seen
            ):
                return None
            seen.add(identity)
            current_by_word[word] = entry
        proposed_words = []
        for index, entry in enumerate(proposed):
            if not isinstance(entry, dict):
                return None
            word = str(entry.get("word") or "").strip()
            target_weight = entry.get("weight")
            current_entry = current_by_word.get(word)
            if (
                current_entry is None
                or entry.get("code") != code
                or entry.get("type") != phrase_type
                or target_weight != base_weight + index
            ):
                return None
            proposed_words.append(word)
            if current_entry["weight"] != target_weight:
                expected.append({
                    "action": "Change",
                    "old_word": word,
                    "word": word,
                    "code": code,
                    "type": phrase_type,
                    "weight": target_weight,
                })
        if len(set(proposed_words)) != len(current_by_word):
            return None
    return expected or None


def _resolved_chain_reorder_items_match(state: PendingToolConfirm) -> bool:
    """Fail closed if a chain-order ticket no longer carries its sealed set."""
    pending_display = state.args.get("_pending_display")
    if not isinstance(pending_display, dict) or "chainReorderPlan" not in pending_display:
        return True
    plan = pending_display.get("chainReorderPlan")
    expected_items = _chain_reorder_expected_items(plan) if isinstance(plan, dict) else None
    if (
        state.function_name != "keytao_batch_add_to_draft"
        or not isinstance(plan, dict)
        or not re.fullmatch(r"[0-9a-f]{64}", str(plan.get("digest") or ""))
        or _chain_reorder_plan_digest(plan) != plan.get("digest")
        or not isinstance(plan.get("items"), list)
        or not isinstance(state.args.get("items"), list)
        or expected_items is None
    ):
        return False
    return state.args["items"] == plan["items"] == expected_items


def _prepend_resolved_advertised_words(
    state: PendingToolConfirm,
    response: str,
) -> str:
    """Echo the exact record-derived set before its execution result."""
    expected = state.args.get("_resolved_advertised_words")
    if not isinstance(expected, list) or not expected:
        return response
    words = [str(word).strip() for word in expected if str(word).strip()]
    if len(words) != len(expected):
        return response
    unselected = state.args.get("_unselected_words")
    remainder = (
        "\n未选择：" + "、".join(f"「{word}」" for word in unselected)
        + "；这些词本次未添加。"
        if isinstance(unselected, list) and unselected else ""
    )
    skipped = state.args.get("_selection_skipped")
    skipped_line = "；".join(skipped) + "。\n" if isinstance(skipped, list) and skipped else ""
    if skipped_line:
        selected = "、".join(f"{item['word']} → {item['code']}" for item in state.args.get("items", []))
        completed = "✅ 已加入草稿" in response or "已提交审核" in response
        label = "已按你的选择加入" if completed else "本次选择"
        return f"{label}：{selected}；{skipped_line}{remainder}\n{response}"
    return f"词：{'、'.join(words)}{remainder}\n" + skipped_line + str(response or "")


def _append_submit_snapshot_lines(lines: List[str], data: Dict) -> None:
    """Append every server-bound draft item without exposing internal digests."""
    snapshot_items = (
        data.get("snapshotItems")
        if isinstance(data.get("snapshotItems"), list)
        else []
    )
    if not snapshot_items:
        return
    lines.append("提交内容：")
    for item in snapshot_items:
        if not isinstance(item, dict):
            continue
        old_word = str(item.get("oldWord") or "")
        word = str(item.get("word") or "")
        code = str(item.get("code") or "")
        action = str(item.get("action") or "")
        label = f"{old_word} → {word}" if action == "Change" and old_word else word
        lines.append(
            f"• {action} {label} @ {code}（{item.get('type') or ''}）"
        )


def _ranked_shift_warning_copy(
    warning: Any,
    proposed_state: List[Dict[str, Any]],
) -> Tuple[str, bool]:
    """Render strict item warnings against the complete sealed shift plan."""
    plain = _plain_warning_message(warning)
    if not isinstance(warning, dict) or warning.get("warningType") != "skipped_candidate_slot":
        return plain, False
    skipped_codes = [
        str(code or "").strip().lower()
        for code in warning.get("skippedCodes") or [warning.get("skippedCode")]
        if str(code or "").strip()
    ]
    proposed_by_code = {
        str(entry.get("code") or "").strip().lower(): str(entry.get("word") or "").strip()
        for entry in proposed_state
        if isinstance(entry, dict)
        and str(entry.get("code") or "").strip()
        and str(entry.get("word") or "").strip()
    }
    warning_item = warning.get("item") if isinstance(warning.get("item"), dict) else {}
    warning_word = str(warning_item.get("word") or "").strip()
    resolved = [
        (code, proposed_by_code.get(code, ""))
        for code in skipped_codes
        if proposed_by_code.get(code) and proposed_by_code.get(code) != warning_word
    ]
    if not skipped_codes or len(resolved) != len(skipped_codes):
        return plain, False
    assignments = "、".join(f"「{word}」占用 {code}" for code, word in resolved)
    return (
        f"单项预览提示「{warning_word}」跳过更短候选；"
        f"完整计划会由{assignments}，不会留下空位。",
        True,
    )


_RANKED_SHIFT_SIGNAL_RE = re.compile(
    r"^「(?P<word>[^」]+)」：(?:整词)?语料频次\s*"
    r"(?P<frequency>无|[\d,.]+)，词典收录\s*"
    r"(?P<dictionary>无|[\d,.]+)$"
)


def _compact_ranked_shift_evidence(evidence_lines: List[str]) -> str:
    """Keep one decisive visible clause while raw evidence remains in tool logs."""
    unique_lines = list(OrderedDict.fromkeys(evidence_lines))
    signals: List[Tuple[float, float, str]] = []
    for line in unique_lines:
        match = _RANKED_SHIFT_SIGNAL_RE.fullmatch(line)
        if match is None:
            continue
        frequency_text = match.group("frequency")
        dictionary_text = match.group("dictionary")
        frequency = (
            0.0
            if frequency_text == "无"
            else float(frequency_text.replace(",", ""))
        )
        dictionary = (
            0.0
            if dictionary_text == "无"
            else float(dictionary_text.replace(",", ""))
        )
        if frequency > 0 and dictionary > 0:
            summary = (
                f"{match.group('word')} 语料{frequency_text}/词典{dictionary_text}"
            )
        elif frequency > 0:
            summary = f"{match.group('word')} 语料{frequency_text}"
        elif dictionary > 0:
            summary = f"{match.group('word')} 词典{dictionary_text}"
        else:
            summary = f"{match.group('word')} 无记录"
        signals.append((frequency, dictionary, summary))
    if signals:
        signals.sort(key=lambda item: (item[0] > 0, item[0], item[1]), reverse=True)
        return "关键证据：" + "，".join(summary for _freq, _dict, summary in signals[:3])
    if not unique_lines:
        return ""
    decisive = re.split(r"[；。]", unique_lines[0], maxsplit=1)[0].strip()
    if len(decisive) > 48:
        decisive = decisive[:47].rstrip() + "…"
    return "关键证据：" + decisive


def _replace_at_code_ticket_details(
    state: PendingToolConfirm,
) -> Optional[Dict[str, Any]]:
    if state.function_name != "keytao_batch_add_to_draft":
        return None
    raw = state.args.get("_replace_at_code")
    if not isinstance(raw, dict):
        return None
    code = str(raw.get("code") or "").strip().lower()
    old_word = str(raw.get("oldWord") or "").strip()
    new_word = str(raw.get("newWord") or "").strip()
    remaining = raw.get("remainingCodes")
    if not isinstance(remaining, list):
        return None
    remaining_codes = tuple(dict.fromkeys(
        str(value or "").strip().lower()
        for value in remaining
        if str(value or "").strip()
    ))
    if (
        len(remaining_codes) != len(remaining)
        or any(
            re.fullmatch(r"[a-z]{1,12}", value) is None or value == code
            for value in remaining_codes
        )
    ):
        return None
    command = parse_replace_at_code(
        f"修改 {code}：{old_word} → {new_word}"
    )
    items = state.args.get("items")
    if command is None or not replace_at_code_items_match(command, items):
        return None
    create_item = items[1]
    return {
        "code": code,
        "oldWord": old_word,
        "newWord": new_word,
        "remainingCodes": remaining_codes,
        "needsManualReview": bool(create_item.get("needsManualReview", True)),
    }


def render_pending_replace_at_code_plan(state: PendingToolConfirm) -> str:
    """Render a code replacement only from one complete server ticket."""
    details = _replace_at_code_ticket_details(state)
    if details is None or not server_warning_ticket_is_complete(state):
        return ""
    old_word = details["oldWord"]
    new_word = details["newWord"]
    code = details["code"]
    lines = [
        "🔁 替换计划：",
        f"• 删除「{old_word}」@ {code}",
        (
            f"• 创建「{new_word}」@ {code}（需管理员审核）"
            if details["needsManualReview"]
            else f"• 创建「{new_word}」@ {code}（可自动通过）"
        ),
    ]
    remaining_codes = details["remainingCodes"]
    if remaining_codes:
        lines.append(
            f"「{old_word}」仍保留在 " + "、".join(remaining_codes) + "。"
        )
    warnings = state.args.get("_pending_display", {}).get("warnings", [])
    if isinstance(warnings, list):
        for warning in warnings:
            try:
                message = _assert_plain_user_facing_reply(_plain_warning_message(warning))
            except ValueError:
                message = "服务端风险信息无法完整展示，请查看草稿核对。"
            lines.append(f"提示：{message}")
    lines.append(pending_confirmation_copy())
    batch_url = _trusted_batch_url(state.args.get("_pending_display", {}))
    if batch_url:
        lines.append(f"草稿地址：{batch_url}")
    return _assert_plain_user_facing_reply("\n".join(lines))


def _format_server_warning_confirmation(function_name: str, data: Dict) -> str:
    if function_name in {"keytao_remove_draft_item", "keytao_batch_remove_draft_items"}:
        targets = data.get("targets") if isinstance(data.get("targets"), list) else []
        lines = [f"🗑️ 删除 {len(targets)} 个草稿项："]
        for target in targets:
            if not isinstance(target, dict):
                continue
            lines.append(
                f"• {target.get('word', '')} "
                f"@ {target.get('code', '')}（{target.get('action', '')}/{target.get('type', '')}）"
            )
        batch_url = _trusted_batch_url(data)
        if batch_url:
            lines.append(f"草稿地址：{batch_url}")
        lines.append(pending_confirmation_copy())
        return _assert_plain_user_facing_reply("\n".join(lines))

    if function_name == "keytao_batch_add_to_draft":
        pending_items = (
            data.get("effectiveItems")
            if isinstance(data.get("effectiveItems"), list)
            else data.get("pendingItems")
        )
        delete_items = [
            item
            for item in pending_items or []
            if isinstance(item, dict)
            and str(item.get("action") or "").strip() == "Delete"
        ]
        if delete_items and len(delete_items) == len(pending_items or []):
            lines = [f"🗑️ 删除 {len(delete_items)} 个词库项："]
            lines.extend(
                f"• {str(item.get('word') or '').strip()} "
                f"@ {str(item.get('code') or '').strip().lower()}"
                f"（{str(item.get('type') or '').strip()}）"
                for item in delete_items
            )
            warnings = (
                data.get("warnings")
                if isinstance(data.get("warnings"), list)
                else []
            )
            lines.extend(
                f"提示：{_plain_warning_message(warning)}"
                for warning in warnings
            )
            lines.append(pending_confirmation_copy())
            batch_url = _trusted_batch_url(data)
            if batch_url:
                lines.append(f"草稿地址：{batch_url}")
            return _assert_plain_user_facing_reply("\n".join(lines))

    if function_name == "keytao_recall_batch":
        batch_url = _trusted_batch_url(data)
        lines = ["↩️ 撤回批次："]
        if batch_url:
            lines.append(f"草稿地址：{batch_url}")
        lines.append(pending_confirmation_copy())
        return _assert_plain_user_facing_reply("\n".join(lines))

    if function_name == "keytao_shift_phrase_code":
        shift_plan = data.get("shiftPlan") if isinstance(data.get("shiftPlan"), dict) else {}
        items = shift_plan.get("items") if isinstance(shift_plan.get("items"), list) else []
        resolved_candidate_plan = (
            data.get("resolvedCandidatePlan")
            if isinstance(data.get("resolvedCandidatePlan"), list)
            else []
        )
        if resolved_candidate_plan:
            shifted = [
                item
                for item in shift_plan.get("shifted") or []
                if isinstance(item, dict)
            ]
            lines = ["🔁 调整计划："]
            for item in resolved_candidate_plan:
                if not isinstance(item, dict):
                    continue
                word = str(item.get("word") or "").strip()
                code = str(item.get("code") or "").strip().lower()
                if not word or not code:
                    continue
                line = f"• 「{word}」→ {code}"
                if item.get("modifier") == "recode":
                    change = next(
                        (
                            candidate
                            for candidate in shifted
                            if str(candidate.get("fromCode") or "").strip().lower()
                            == code
                        ),
                        None,
                    )
                    if change is not None:
                        occupant = str(change.get("word") or "").strip()
                        destination = str(change.get("toCode") or "").strip().lower()
                        if occupant and destination:
                            line += f"（「{occupant}」顺延至 {destination}）"
                        else:
                            line += "（原占用词顺延）"
                    else:
                        occupants = [
                            str(value or "").strip()
                            for value in item.get("occupants") or []
                            if str(value or "").strip()
                        ]
                        line += (
                            "（" + "、".join(f"「{value}」" for value in occupants)
                            + "顺延）"
                            if occupants
                            else "（原占用词顺延）"
                        )
                lines.append(line)
            warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
            lines.extend(
                f"提示：{_plain_warning_message(warning)}"
                for warning in warnings
            )
            lines.append(pending_confirmation_copy())
            batch_url = _trusted_batch_url(data)
            if batch_url:
                lines.append(f"草稿地址：{batch_url}")
            return _assert_plain_user_facing_reply("\n".join(lines))
        current_state = (
            shift_plan.get("currentState")
            if isinstance(shift_plan.get("currentState"), list)
            else []
        )
        proposed_state = (
            shift_plan.get("proposedState")
            if isinstance(shift_plan.get("proposedState"), list)
            else []
        )
        listed_words = [
            str(word).strip()
            for word in shift_plan.get("listedWords") or []
            if str(word).strip()
        ]
        deleted_words = {
            str(item.get("word") or "").strip()
            for item in items
            if isinstance(item, dict)
            and str(item.get("action") or "").strip() == "Delete"
        }
        created_words = {
            str(item.get("word") or "").strip()
            for item in items
            if isinstance(item, dict)
            and str(item.get("action") or "").strip() == "Create"
        }
        add_with_eviction = bool(
            deleted_words & created_words
            and created_words - deleted_words
        )
        lines = [
            "🔁 添加并顺延计划："
            if add_with_eviction
            else "🔁 调整计划："
        ]
        if add_with_eviction:
            lines.append("调整计划如下：")
        if current_state and proposed_state:
            current_by_word = {
                str(entry.get("word") or "").strip(): entry
                for entry in current_state
                if isinstance(entry, dict)
                and str(entry.get("word") or "").strip()
            }
            displayed_words: set[str] = set()
            for entry in proposed_state:
                if not isinstance(entry, dict):
                    continue
                word = str(entry.get("word") or "").strip()
                previous = current_by_word.get(word)
                if not word or not isinstance(previous, dict):
                    continue
                old_code = str(previous.get("code") or "").strip()
                new_code = str(entry.get("code") or "").strip()
                old_weight = previous.get("weight")
                new_weight = entry.get("weight")
                if old_code == new_code and old_weight == new_weight:
                    continue
                source = "草稿" if previous.get("source") == "draft" else "词库"
                lines.append(
                    f"• {word}：{old_code} / {old_weight}（{source}）"
                    f"→ {new_code} / {new_weight}"
                )
                displayed_words.add(word)
            for shifted in shift_plan.get("shifted") or []:
                if not isinstance(shifted, dict):
                    continue
                word = str(shifted.get("word") or "").strip()
                old_code = str(shifted.get("fromCode") or "").strip()
                new_code = str(shifted.get("toCode") or "").strip()
                if word and old_code and new_code and word not in displayed_words:
                    lines.append(f"• {word}：{old_code}→{new_code}（顺延）")
                    displayed_words.add(word)
            proposed_words = [
                str(entry.get("word") or "").strip()
                for entry in proposed_state
                if isinstance(entry, dict) and str(entry.get("word") or "").strip()
            ]
            evidence_summary: List[str] = []
            if listed_words:
                relation = "一致" if proposed_words == listed_words else "不一致"
                evidence_summary.append(
                    f"建议 {'→'.join(proposed_words)}（与你列的{relation}）"
                )
            evidence_lines = [
                str(line).strip()
                for line in shift_plan.get("evidenceLines") or []
                if str(line).strip()
            ]
            if any(line.startswith("将把 ") for line in evidence_lines) and any(
                line.startswith("常用度提示：") for line in evidence_lines
            ):
                evidence_summary = []
                lines.extend(evidence_lines)
            else:
                decisive_evidence = _compact_ranked_shift_evidence(evidence_lines)
                if decisive_evidence:
                    evidence_summary.append(decisive_evidence)
            if evidence_summary:
                lines.append("依据：" + "；".join(evidence_summary))
        else:
            for item in items:
                if not isinstance(item, dict):
                    continue
                raw_action = str(item.get("action") or "").strip()
                action = {
                    "Create": "添加",
                    "Delete": "删除",
                }.get(raw_action, "变更")
                word = str(item.get("word") or "")
                code = str(item.get("code") or "")
                lines.append(f"• {word}：{raw_action or 'Change'} {code}（{action}）")
        for item in items:
            if not isinstance(item, dict) or item.get("needsManualReview") is not True:
                continue
            reason = str(item.get("manualReviewReason") or "").strip()
            if reason:
                lines.append(f"提示：「{item.get('word', '')}」@ {item.get('code', '')}：{reason}")
        warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
        if warnings:
            rendered_warnings = [
                _ranked_shift_warning_copy(warning, proposed_state)
                for warning in warnings
            ]
            lines.extend(f"提示：{message}" for message, _resolved in rendered_warnings)
        lines.append(pending_confirmation_copy())
        batch_url = _trusted_batch_url(data)
        if batch_url:
            lines.append(f"草稿地址：{batch_url}")
        return _assert_plain_user_facing_reply("\n".join(lines))

    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    warning_text = "\n".join(
        _plain_warning_line(warning)
        for warning in warnings
    ) if warnings else str(data.get("message") or "服务端发现需要再次确认的风险")
    review_parts: List[str] = []
    if function_name == "keytao_submit_batch":
        _append_submit_review_lines(review_parts, data)
        _append_submit_snapshot_lines(review_parts, data)
    review_text = ("\n" + "\n".join(review_parts)) if review_parts else ""
    batch_url = _trusted_batch_url(data)
    link_text = f"\n草稿地址：{batch_url}" if batch_url else ""
    return _assert_plain_user_facing_reply(
        f"{warning_text}{review_text}\n{pending_confirmation_copy()}{link_text}"
    )


def render_pending_shift_plan(state: PendingToolConfirm) -> str:
    """Render promise and confirmation solely from one sealed shift ticket."""
    if state.function_name != "keytao_shift_phrase_code" or not server_warning_ticket_is_complete(state):
        return ""
    pending_display = state.args.get("_pending_display")
    resolved_plan = state.args.get("_resolved_candidate_plan")
    if resolved_plan is None:
        if not isinstance(pending_display, dict) or not isinstance(pending_display.get("shiftPlan"), dict):
            return ""
        if pending_display["shiftPlan"].get("scope") in {"same_code", "prefix_chain"}:
            return _format_server_warning_confirmation(state.function_name, pending_display)
        from .chat_render import _format_server_bound_confirmation_prompt

        return _format_server_bound_confirmation_prompt(state)
    if (
        not isinstance(pending_display, dict)
        or not isinstance(resolved_plan, list)
        or not _resolved_candidate_plan_matches(state)
    ):
        return ""
    return _format_server_warning_confirmation(
        state.function_name,
        {
            **pending_display,
            "resolvedCandidatePlan": [
                dict(item) for item in resolved_plan if isinstance(item, dict)
            ],
        },
    )


_RANKED_SHIFT_CONTINUATION_COMMAND = "继续调整"
_RANKED_SHIFT_SEMANTIC_ARGS = (
    "word",
    "target_code",
    "target_type",
    "target_remark",
    "target_needs_manual_review",
    "ordered_words",
    "listed_words",
    "expected_codes",
    "expected_weights",
    "evidence_lines",
)


def _ranked_shift_continuation_args(state: PendingToolConfirm) -> Dict[str, Any]:
    """Keep only user-bound semantics when recomputing a remaining delta."""
    return {
        key: state.args[key]
        for key in _RANKED_SHIFT_SEMANTIC_ARGS
        if key in state.args
    }


def _compact_ranked_shift_receipt_reason(value: Any) -> str:
    reason = re.sub(r"\s+", " ", str(value or "").strip())
    if any(marker in reason for marker in ("版本", "变化", "冲突", "待删除条目")):
        return "草稿版本冲突"
    if any(marker in reason for marker in ("目标", "条目", "找不到", "不存在")):
        return "草稿目标已变化"
    if any(marker in reason.lower() for marker in ("timeout", "network", "连接", "超时")):
        return "服务暂时不可用"
    return "草稿权重调整未完成"


def _format_ranked_shift_partial_failure(
    data: Dict[str, Any],
    *,
    continuation_staged: bool,
) -> str:
    """Render only authoritative step receipts plus one safe next action."""
    applied: List[str] = []
    remaining: List[str] = []
    receipts = data.get("receipts") if isinstance(data.get("receipts"), list) else []
    for receipt in receipts:
        if not isinstance(receipt, dict):
            continue
        step = str(receipt.get("step") or "")
        status = str(receipt.get("status") or "")
        if step == "dictionary" and status in {"applied", "alreadyApplied"}:
            changes = [
                f"{str(change.get('word') or '').strip()} "
                f"{str(change.get('fromCode') or '').strip()}→"
                f"{str(change.get('toCode') or '').strip()}"
                for change in receipt.get("changes") or []
                if isinstance(change, dict)
                and str(change.get("word") or "").strip()
                and str(change.get("fromCode") or "").strip()
                and str(change.get("toCode") or "").strip()
            ]
            applied.extend(changes)
            if not changes and int(receipt.get("itemCount") or 0) > 0:
                applied.append(f"实时词库变更 {int(receipt['itemCount'])} 项")
        elif step == "draftWeight":
            word = str(receipt.get("word") or "").strip()
            code = str(receipt.get("code") or "").strip()
            change = (
                f"{word} {code} 权重 "
                f"{receipt.get('fromWeight')}→{receipt.get('toWeight')}"
            ).strip()
            if status in {"applied", "alreadyApplied"}:
                applied.append(change)
            elif status in {"failed", "pending"}:
                if status == "failed":
                    change += "（" + _compact_ranked_shift_receipt_reason(
                        receipt.get("reason")
                    ) + "）"
                remaining.append(change)

    lines = ["⚠️ 计划部分完成"]
    if "writtenItems" in data or "updatedItems" in data:
        from ..utils.draft_receipts import receipt_change_lines
        applied = []
        lines.extend(receipt_change_lines(data))
    if applied:
        lines.append("已完成：" + "、".join(applied))
    if remaining:
        lines.append("未完成：" + "、".join(remaining))
    batch_url = _trusted_batch_url(data)
    if batch_url:
        lines.append(f"草稿地址：{batch_url}")
    if continuation_staged:
        lines.append(
            f"回复「{_RANKED_SHIFT_CONTINUATION_COMMAND}」完成剩余步骤，"
            "或「取消」放弃。"
        )
    else:
        lines.append("剩余步骤暂不可执行。")
    return _assert_plain_user_facing_reply("\n".join(lines))


def _format_ranked_shift_success(data: Dict[str, Any]) -> str:
    """Render applied shift receipts without repeating the full draft."""
    from .chat_render import finalize_draft_receipt
    if "writtenItems" in data or "updatedItems" in data:
        return _assert_plain_user_facing_reply(finalize_draft_receipt("✅ 操作已完成", data))
    changes: List[str] = []
    shift_plan = data.get("shiftPlan") if isinstance(data.get("shiftPlan"), dict) else {}
    target_word = str(shift_plan.get("word") or "").strip()
    target_code = str(shift_plan.get("targetCode") or "").strip()
    snapshot = data.get("draft_snapshot")
    if (
        data.get("success") is True
        and shift_plan.get("scope") == "same_code"
        and isinstance(snapshot, dict)
        and snapshot.get("success") is True
        and data.get("batchId")
        and snapshot.get("batchId") == data.get("batchId")
    ):
        current = {
            (entry.get("word"), entry.get("code"), entry.get("type")): entry
            for entry in shift_plan.get("currentState") or []
            if isinstance(entry, dict) and entry.get("source") == "live"
        }
        for proposed in shift_plan.get("proposedState") or []:
            if not isinstance(proposed, dict):
                continue
            identity = (proposed.get("word"), proposed.get("code"), proposed.get("type"))
            previous = current.get(identity)
            if previous is None or previous.get("weight") == proposed.get("weight"):
                continue
            matches = [
                row for row in snapshot.get("items") or []
                if isinstance(row, dict)
                and (row.get("word"), row.get("code"), row.get("type")) == identity
                and row.get("action") == "Change"
                and (row.get("oldWord") or row.get("old_word")) == identity[0]
            ]
            if (
                len(matches) == 1
                and matches[0].get("weight") == proposed.get("weight")
                and all(isinstance(weight, int) and not isinstance(weight, bool) for weight in (
                    previous.get("weight"), proposed.get("weight"),
                ))
            ):
                changes.append(
                    f"{identity[0]} {identity[1]} 权重 "
                    f"{previous['weight']}→{proposed['weight']}"
                )
    receipts = data.get("receipts") if isinstance(data.get("receipts"), list) else []
    for receipt in receipts:
        if not isinstance(receipt, dict) or receipt.get("status") not in {
            "applied", "alreadyApplied",
        }:
            continue
        if receipt.get("step") == "dictionary":
            changes.extend(
                f"{str(change.get('word') or '').strip()} "
                f"{str(change.get('fromCode') or '').strip()}→"
                f"{str(change.get('toCode') or '').strip()}"
                for change in receipt.get("changes") or []
                if isinstance(change, dict)
                and str(change.get("word") or "").strip()
                and str(change.get("fromCode") or "").strip()
                and str(change.get("toCode") or "").strip()
            )
        elif receipt.get("step") == "draftWeight":
            changes.append(
                f"{str(receipt.get('word') or '').strip()} "
                f"{str(receipt.get('code') or '').strip()} 权重 "
                f"{receipt.get('fromWeight')}→{receipt.get('toWeight')}"
            )
    if not changes:
        changes.extend(
            f"{str(change.get('word') or '').strip()} "
            f"{str(change.get('fromCode') or '').strip()}→"
            f"{str(change.get('toCode') or '').strip()}"
            for change in shift_plan.get("shifted") or []
            if isinstance(change, dict)
            and str(change.get("word") or "").strip()
            and str(change.get("fromCode") or "").strip()
            and str(change.get("toCode") or "").strip()
        )
        changes.extend(
            f"{str(update.get('word') or '').strip()} "
            f"{str(update.get('code') or '').strip()} 权重 "
            f"{update.get('fromWeight')}→{update.get('toWeight')}"
            for update in shift_plan.get("draftUpdates") or []
            if isinstance(update, dict) and str(update.get("word") or "").strip()
        )
    if target_word and target_code and shift_plan.get("scope") != "same_code":
        changes.insert(0, f"{target_word} → {target_code}")
    lines = ["✅ 操作已完成"]
    if changes:
        lines.append("已变更：" + "、".join(OrderedDict.fromkeys(changes)))
    batch_url = _trusted_batch_url(data)
    if batch_url:
        lines.append(f"草稿地址：{batch_url}")
    return _assert_plain_user_facing_reply("\n".join(lines))


async def _stage_ranked_shift_continuation(
    state: PendingToolConfirm,
    data: Dict[str, Any],
    platform: str,
    user_id: str,
    conv_key: Optional[ConversationKey],
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> bool:
    """Recompute and seal only the remaining ranked-shift delta."""
    continuation_args = _ranked_shift_continuation_args(state)
    if (
        state.function_name != "keytao_shift_phrase_code"
        or not continuation_args.get("word")
        or not continuation_args.get("target_code")
    ):
        return False
    preview_json = await call_tool_function(
        state.function_name,
        continuation_args,
        platform,
        user_id,
    )
    try:
        preview_data = json.loads(preview_json)
    except Exception:
        return False
    if preview_data.get("requiresConfirmation") is not True:
        return False
    pending = _pending_state_from_server_warning(
        PendingToolConfirm(
            function_name=state.function_name,
            args=continuation_args,
        ),
        preview_data,
    )
    pending.args["_continuation_command"] = _RANKED_SHIFT_CONTINUATION_COMMAND
    if not server_warning_ticket_is_complete(pending):
        return False
    target_key: ConversationKey = conv_key or (platform, user_id)
    return conversation_state_store.set(
        target_key,
        pending,
        space_key=space_key,
        owner_label=owner_label,
    )


def _shift_plan_warnings_are_bound(data: Dict, args: Dict) -> bool:
    """Allow informational warnings only for the exact requested additions."""
    from ..harness.authorization_grammar import _AUTO_CONFIRM_CREATE_WARNING_TYPES

    warnings = data.get("warnings")
    if not isinstance(warnings, list):
        return False
    requested = [{"word": args.get("word"), "code": args.get("target_code")}]
    for key in ("additional_items", "additional_shift_items"):
        requested.extend(item for item in args.get(key) or [] if isinstance(item, dict))
    identities = {
        ("Create", str(item.get("word") or "").strip(), str(item.get("code") or "").strip().lower())
        for item in requested
        if str(item.get("action") or "Create") == "Create"
    }
    for warning in warnings:
        if not isinstance(warning, dict) or warning.get("warningType") not in _AUTO_CONFIRM_CREATE_WARNING_TYPES:
            return False
        source = warning.get("item") if isinstance(warning.get("item"), dict) else warning
        if (
            str(source.get("action") or "Create").strip(),
            str(source.get("word") or "").strip(),
            str(source.get("code") or "").strip().lower(),
        ) not in identities:
            return False
    return True


async def _execute_confirmed_tool(
    state: PendingToolConfirm,
    platform: str,
    user_id: str,
    conv_key: Optional[ConversationKey] = None,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
    carried_warnings: Optional[List[Any]] = None,
    carried_ordering_summary: str = "",
    on_transport_failure: Optional[Callable[[], None]] = None,
    auto_confirm_shift_plan: bool = False,
    require_new_confirmation: bool = False,
) -> str:
    """Execute one staged step without bypassing unseen server warnings."""
    semantic_direct = _semantic_direct_words.get() is not None
    auto_confirm_shift_plan = auto_confirm_shift_plan or semantic_direct
    if state.confirmation_source not in {"local_preview", "server_warning"}:
        return render_remediation_reply("确认来源无效，已拒绝执行")
    if not _resolved_advertised_items_match(state):
        invalid_words = tuple(
            word for word, _code in pending_batch_display_pairs(state)
        )
        return render_remediation_reply(
            "候选集合校验失败，当前确认已失效；本次未写入",
            command=("加词 " + " ".join(invalid_words)) if invalid_words else "",
            words=invalid_words,
        )
    if not _resolved_candidate_plan_matches(state):
        return render_remediation_reply(
            "复合候选计划校验失败，当前确认已失效；本次未写入"
        )
    if not _resolved_chain_reorder_items_match(state):
        return render_remediation_reply(
            "编码链重排集合校验失败，当前确认已失效；本次未写入",
        )

    replace_at_code_details = _replace_at_code_ticket_details(state)
    args = _pending_execution_args(state)
    args.pop("preview_only", None)
    args.pop("_candidate_scopes", None)
    args.pop("_reviewed_multi_word", None)
    args.pop("_reviewed_batch_readings", None)
    args.pop("_query_words", None)
    args.pop("_query_other_blocks", None)
    args.pop("_explicit_add", None)
    receipt_requested_words = args.pop("_receipt_requested_words", None) or [
        item.get("word") for item in state.args.get("items", []) if isinstance(item, dict) and item.get("word")
    ]
    args.pop("_unselected_words", None)
    args.pop("_selection_skipped", None)
    args.pop("_resolved_advertised_words", None)
    resolved_candidate_plan = args.pop("_resolved_candidate_plan", None)
    replace_char_continuation = args.pop(
        _REPLACE_CHAR_CONTINUATION_KEY,
        None,
    )
    if (
        replace_char_continuation is not None
        and not _valid_replace_char_continuation(
            state.args.get("items"),
            replace_char_continuation,
        )
    ):
        return render_remediation_reply(
            "替换进度集合校验失败，当前确认已失效；本次未写入"
        )
    ordering_summary = str(
        args.pop("_ordering_summary", "")
        or carried_ordering_summary
        or ""
    ).strip()
    reviewed_pinyin = str(args.pop("_reviewed_pinyin", "") or "").strip()
    reviewed_candidate_codes = tuple(
        str(value or "").strip().lower()
        for value in args.pop("_reviewed_candidate_codes", [])
        if str(value or "").strip()
    )
    auto_confirm_expected_shifted_words = tuple(
        str(value or "").strip()
        for value in args.pop("_auto_confirm_expected_shifted_words", [])
        if str(value or "").strip()
    )
    auto_confirm_only_if_no_shifts = bool(
        args.pop("_auto_confirm_only_if_no_shifts", False)
    )
    additional_shift_capabilities: Dict[
        Tuple[str, str], Dict[str, Any]
    ] = {}
    for argument_name in ("additional_shift_items", "additional_items"):
        raw_additional_shift_items = args.get(argument_name)
        if not isinstance(raw_additional_shift_items, list):
            continue
        sanitized_shift_items: List[Dict[str, Any]] = []
        for raw_item in raw_additional_shift_items:
            if not isinstance(raw_item, dict):
                sanitized_shift_items.append({})
                continue
            item = dict(raw_item)
            item_pinyin = str(item.pop("reviewedPinyin", "") or "").strip()
            item_codes = tuple(
                str(value or "").strip().lower()
                for value in item.pop("reviewedCandidateCodes", [])
                if str(value or "").strip()
            )
            item_word = str(item.get("word") or "").strip()
            item_code = str(item.get("code") or "").strip().lower()
            capability = _reviewed_create_capability(
                item_word,
                item_code,
                item_pinyin,
                item_codes,
            )
            if capability is not None:
                additional_shift_capabilities.update(capability)
            sanitized_shift_items.append(item)
        args[argument_name] = sanitized_shift_items
    submit_after = bool(args.pop("_submit_after", False))
    expected_keep_words = tuple(
        str(word)
        for word in args.pop("_expected_keep_words", [])
        if str(word)
    )
    if (
        state.confirmation_source == "local_preview"
        and state.function_name in {
            "keytao_create_phrase",
            "keytao_batch_add_to_draft",
            "keytao_submit_batch",
        }
    ):
        args["preview_only"] = True
    if state.confirmation_source == "server_warning" and state.function_name in {
        "keytao_create_phrase",
        "keytao_submit_batch",
        "keytao_batch_add_to_draft",
    }:
        if state.function_name == "keytao_submit_batch" and (
            not args.get("batch_id")
            or not isinstance(args.get("expected_content_version"), int)
            or isinstance(args.get("expected_content_version"), bool)
            or args["expected_content_version"] < 0
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(args.get("expected_server_snapshot_digest") or ""),
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(args.get("expected_warning_digest") or ""),
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(args.get("expected_audit_digest") or ""),
            )
        ):
            return render_remediation_reply(
                "提交确认内容不完整，已拒绝执行",
                command="提交",
            )
        if state.function_name in {"keytao_create_phrase", "keytao_batch_add_to_draft"} and (
            not args.get("batch_id")
            or not isinstance(args.get("expected_content_version"), int)
            or isinstance(args.get("expected_content_version"), bool)
            or args["expected_content_version"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(args.get("expected_warning_digest") or ""))
        ):
            invalid_words = tuple(
                word for word, _code in pending_batch_display_pairs(state)
            )
            return render_remediation_reply(
                "添加确认内容不完整，已拒绝执行",
                command=("加词 " + " ".join(invalid_words)) if invalid_words else "",
                words=invalid_words,
            )
        args["confirmed"] = True
    if state.confirmation_source == "server_warning" and state.function_name == "keytao_shift_phrase_code":
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(args.get("confirmed_plan_digest") or ""))
            or not isinstance(args.get("expected_content_version"), int)
            or isinstance(args.get("expected_content_version"), bool)
            or args["expected_content_version"] < 0
            # "No draft existed" is a valid anchor, but only at version 0.
            or (not args.get("batch_id") and args["expected_content_version"] != 0)
        ):
            return render_remediation_reply(
                "顺延确认内容不完整，已拒绝执行",
                command="查看草稿",
            )
    if state.confirmation_source == "server_warning" and state.function_name == "keytao_recall_batch":
        if (
            not args.get("batch_id")
            or not isinstance(args.get("expected_content_version"), int)
            or isinstance(args.get("expected_content_version"), bool)
            or args["expected_content_version"] < 0
        ):
            return render_remediation_reply(
                "撤回确认内容不完整，已拒绝执行",
                command="查看草稿",
            )
    if state.confirmation_source == "server_warning" and state.function_name in {
        "keytao_remove_draft_item",
        "keytao_batch_remove_draft_items",
    }:
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(args.get("expected_target_digest") or ""))
            or not isinstance(args.get("expected_targets"), list)
            or not args.get("batch_id")
            or not isinstance(args.get("expected_content_version"), int)
            or isinstance(args.get("expected_content_version"), bool)
            or args["expected_content_version"] < 0
        ):
            return render_remediation_reply(
                "删除确认内容不完整，已拒绝执行",
                command="查看草稿",
            )
    reviewed_capability = (
        _reviewed_create_capability(
            str(args.get("word") or "").strip(),
            str(
                args.get(
                    "target_code"
                    if state.function_name == "keytao_shift_phrase_code"
                    else "code"
                )
                or ""
            ).strip().lower(),
            reviewed_pinyin,
            reviewed_candidate_codes,
        )
        if state.function_name in {
            "keytao_create_phrase",
            "keytao_shift_phrase_code",
        }
        else None
    )
    reviewed_capabilities = dict(reviewed_capability or {})
    reviewed_capabilities.update(additional_shift_capabilities)
    reviewed_capabilities.update(_reviewed_multi_word_capabilities(state))
    if (
        state.function_name == "keytao_shift_phrase_code"
        and reviewed_pinyin
        and reviewed_candidate_codes
    ):
        args["_reviewed_pinyin"] = reviewed_pinyin
        args["_reviewed_candidate_codes"] = list(reviewed_candidate_codes)
    if reviewed_capabilities:
        result_json = await call_tool_function(
            state.function_name,
            args,
            platform,
            user_id,
            trusted_reviewed_items_by_key=reviewed_capabilities,
        )
    else:
        result_json = await call_tool_function(
            state.function_name, args, platform, user_id,
        )
    data = json.loads(result_json)
    if receipt_requested_words:
        data["requestedWords"] = receipt_requested_words
        _capture_successful_draft_write_delivery(state.function_name, data, platform, user_id)

    if data.get("transportError") is True:
        if on_transport_failure is not None:
            on_transport_failure()
        return render_remediation_reply(
            "连接服务失败，结果不确定；仍可再次确认",
            command="确认",
        )

    if (
        state.function_name == "keytao_shift_phrase_code"
        and data.get("partialWrite") is True
        and isinstance(data.get("receipts"), list)
    ):
        continuation_staged = await _stage_ranked_shift_continuation(
            state,
            data,
            platform,
            user_id,
            conv_key,
            space_key,
            owner_label,
        )
        from .chat_render import finalize_draft_receipt
        return finalize_draft_receipt(
            _format_ranked_shift_partial_failure(data, continuation_staged=continuation_staged),
            data, platform=platform, requested_words=receipt_requested_words,
        )

    if data.get("success") is True and carried_warnings:
        data["warnings"] = list(carried_warnings)
        data["warnedCount"] = len(carried_warnings)
        data["autoConfirmedWarnings"] = True
    if data.get("success") is True and ordering_summary:
        data["orderingSummary"] = ordering_summary

    if data.get("not_bound"):
        return _BIND_HELP_TEXT

    if (
        state.function_name == "keytao_shift_phrase_code"
        and data.get("requiresDraftCleanup") is True
    ):
        conflict_items = data.get("conflictingDraftItems")
        if not isinstance(conflict_items, list):
            conflict_items = data.get("relatedDraftItems")
        conflict_items = [
            item for item in (conflict_items or []) if isinstance(item, dict)
        ]
        conflict_ids = list(dict.fromkeys(
            int(item["id"])
            for item in conflict_items
            if isinstance(item.get("id"), int)
            and not isinstance(item.get("id"), bool)
            and item["id"] > 0
        ))
        conflict_labels = "、".join(
            f"{str(item.get('action') or 'Create')} 「{str(item.get('word') or '未知词条')}」"
            f"@{str(item.get('code') or item.get('fromCode') or '未知编码')}"
            for item in conflict_items[:8]
        )
        reason = (
            "现有草稿行与本轮顺延计划冲突"
            + (f"：{conflict_labels}" if conflict_labels else "")
            + "；本次未修改草稿"
        )
        cleanup_plan: Optional[PendingToolConfirm] = None
        if conflict_ids:
            cleanup_args: Dict[str, Any] = {"ids": conflict_ids}
            if str(data.get("batchId") or "").strip():
                cleanup_args["batch_id"] = str(data["batchId"])
            try:
                cleanup_preview = json.loads(await call_tool_function(
                    "keytao_batch_remove_draft_items",
                    cleanup_args,
                    platform,
                    user_id,
                ))
            except Exception:
                cleanup_preview = {}
            if cleanup_preview.get("requiresConfirmation") is True:
                candidate_plan = _pending_state_from_server_warning(
                    PendingToolConfirm(
                        function_name="keytao_batch_remove_draft_items",
                        args=cleanup_args,
                    ),
                    cleanup_preview,
                )
                expected_targets = candidate_plan.args.get("expected_targets")
                expected_ids = {
                    int(target.get("id"))
                    for target in expected_targets or []
                    if isinstance(target, dict)
                    and str(target.get("id") or "").isdigit()
                }
                if (
                    server_warning_ticket_is_complete(candidate_plan)
                    and expected_ids == set(conflict_ids)
                    and len(expected_targets) == len(conflict_ids)
                ):
                    cleanup_plan = candidate_plan
        offer = _build_pending_choice_offer(
            (
                (
                    "A",
                    "删除上述冲突草稿行；删除成功后可重新执行原顺延指令",
                    cleanup_plan,
                ),
                ("B", "保留当前草稿并停止本次顺延", None),
            ),
            protected_add_pairs=((
                str(state.args.get("word") or "").strip(),
                str(state.args.get("target_code") or "").strip().lower(),
            ),),
            reason=reason + "。",
        ) if cleanup_plan is not None else None
        target_key: ConversationKey = conv_key or (platform, user_id)
        if offer is None or not conversation_state_store.set(
            target_key,
            offer,
            space_key=space_key,
            owner_label=owner_label,
        ):
            return render_remediation_reply(reason)
        return _assert_plain_user_facing_reply(render_pending_choice_offer(offer))

    if data.get("requiresConfirmation"):
        if state.confirmation_source == "local_preview" and not state.args.get("_candidate_origin"):
            context = current_memory_context.get()
            origin_key = (context.conversation_address if context is not None
                          and context.platform == platform and context.user_id == user_id
                          else conv_key or (platform, user_id))
            origin = _candidate_origin_for_ticket(conversation_state_store.get(origin_key))
            if origin:
                state = replace(state, args={**state.args, "_candidate_origin": origin})
        pending_state = _pending_state_from_server_warning(state, data)
        if state.confirmation_source == "server_warning":
            if not server_warning_ticket_is_complete(pending_state):
                return render_remediation_reply(
                    "新的确认内容不完整，原确认未执行",
                    command="查看草稿",
                )
            if _pending_execution_args(pending_state) == _pending_execution_args(state):
                return render_remediation_reply(
                    "确认内容未变化，原确认未执行",
                    command="查看草稿",
                )
        shifted_words = tuple(
            str(item.get("word") or "").strip()
            for item in (
                data.get("shiftPlan", {}).get("shifted", [])
                if isinstance(data.get("shiftPlan"), dict)
                else []
            )
            if isinstance(item, dict)
            and str(item.get("word") or "").strip()
        )
        should_auto_confirm_shift = bool(
            auto_confirm_shift_plan
            and state.function_name == "keytao_shift_phrase_code"
            and server_warning_ticket_is_complete(pending_state)
            and (
                _semantic_warnings_are_named(data, args)
                and set(shifted_words) <= set(_semantic_direct_words.get() or ())
                if semantic_direct else _shift_plan_warnings_are_bound(data, args)
            )
            and (
                (
                    auto_confirm_only_if_no_shifts
                    and not shifted_words
                )
                or (
                    not auto_confirm_only_if_no_shifts
                    and (
                        not auto_confirm_expected_shifted_words
                        or shifted_words == auto_confirm_expected_shifted_words
                    )
                )
            )
            and (
                (
                    state.confirmation_source == "local_preview"
                    and data.get("confirmationKind") == "shiftPlan"
                )
                or (
                    state.confirmation_source == "server_warning"
                    and data.get("warnings") == []
                )
            )
        )
        if should_auto_confirm_shift:
            return await _execute_confirmed_tool(
                pending_state,
                platform,
                user_id,
                conv_key,
                space_key,
                owner_label,
                carried_warnings=list(data.get("warnings") or []),
                carried_ordering_summary=ordering_summary,
                on_transport_failure=on_transport_failure,
                auto_confirm_shift_plan=True,
            )
        auto_confirm_binding = None
        if semantic_direct and state.confirmation_source == "local_preview" and state.function_name != "keytao_shift_phrase_code":
            if _semantic_warnings_are_named(data, args) and server_warning_ticket_is_complete(pending_state):
                auto_confirm_binding = _pending_execution_args(pending_state)
        elif not semantic_direct and state.confirmation_source == "local_preview" and not require_new_confirmation:
            if state.function_name == "keytao_create_phrase":
                auto_confirm_binding = create_warning_confirmation_binding(
                    data,
                    args,
                )
            elif state.function_name == "keytao_batch_add_to_draft":
                auto_confirm_binding = batch_warning_confirmation_binding(
                    data,
                    args,
                )
        if auto_confirm_binding is not None:
            warning_ordering_summary = (
                ordering_summary
                or _create_warning_ordering_summary(data, args)
            )
            return await _execute_confirmed_tool(
                pending_state,
                platform,
                user_id,
                conv_key,
                space_key,
                owner_label,
                carried_warnings=list(data.get("warnings") or []),
                carried_ordering_summary=warning_ordering_summary,
                on_transport_failure=on_transport_failure,
            )
        display_data = {
            **data,
            "submitAfter": submit_after,
            **(
                {"resolvedCandidatePlan": resolved_candidate_plan}
                if isinstance(resolved_candidate_plan, list)
                else {}
            ),
            **(
                {"pendingItems": list(args.get("items") or [])}
                if state.function_name == "keytao_batch_add_to_draft"
                else {}
            ),
        }
        warning_prompt = (
            render_pending_replace_at_code_plan(pending_state)
            or (
                _format_changed_server_confirmation_prompt(state, pending_state)
                if state.confirmation_source == "server_warning"
                else _format_server_warning_confirmation(
                    state.function_name,
                    display_data,
                )
            )
        )
        if len(warning_prompt) > MAX_REPLACE_CONFIRMATION_CHARS:
            return _append_batch_url_if_missing(
                render_remediation_reply(
                    "确认内容过长，本次未执行；请缩小范围"
                ),
                display_data,
            )
        target_key: ConversationKey = conv_key or (platform, user_id)
        saved = conversation_state_store.set(
            target_key,
            pending_state,
            space_key=space_key,
            owner_label=owner_label,
        )
        if not saved:
            return _append_batch_url_if_missing(
                render_remediation_reply(
                    "确认内容过长，本次未执行；请缩小范围"
                ),
                display_data,
            )
        return _append_batch_url_if_missing(warning_prompt, display_data)

    async def continue_with_submit_preview(
        response: str,
        *,
        authorized_items: Optional[List[Dict]] = None,
    ) -> str:
        batch_id = str(data.get("batchId") or args.get("batch_id") or "").strip()
        if not batch_id:
            return response + "\n" + render_remediation_reply(
                "操作完成，但响应缺少精确批次，已停止后续提交",
                command="查看草稿",
            )
        if authorized_items:
            submit_result = await _perform_submit_current_draft(
                platform,
                user_id,
                batch_id=batch_id,
                preview_only=True,
                auto_confirm=True,
                authorized_items=authorized_items,
            )
            if submit_result.pending_state is not None:
                target_key: ConversationKey = conv_key or (platform, user_id)
                conversation_state_store.set(
                    target_key,
                    submit_result.pending_state,
                    space_key=space_key,
                    owner_label=owner_label,
                )
            submit_response = submit_result.text
        else:
            submit_response = await _execute_confirmed_tool(
                PendingToolConfirm(
                    function_name="keytao_submit_batch",
                    args={"batch_id": batch_id},
                    confirmation_source="local_preview",
                ),
                platform,
                user_id,
                conv_key,
                space_key,
                owner_label,
            )
        from .chat_render import finalize_draft_receipt
        combined = finalize_draft_receipt(
            response + "\n\n" + submit_response,
            *((submit_result.data or {},) if authorized_items else ()), data,
            platform=platform, requested_words=receipt_requested_words,
        )
        return prepend_pending_word_reminders(
            combined,
            data.get("pendingItems"),
        )

    def stage_next_replace_char_chunk(response: str) -> str:
        if not isinstance(replace_char_continuation, dict):
            return response
        remaining = replace_char_continuation["remainingItems"]
        old_char = replace_char_continuation["oldChar"]
        new_char = replace_char_continuation["newChar"]
        completed = (
            int(replace_char_continuation["completed"])
            + len(state.args.get("items", []))
        )
        total = int(replace_char_continuation["total"])
        next_chunk, next_remaining, next_confirmation = _replace_char_chunk(
            remaining,
            old_char,
            new_char,
        )
        if not next_chunk:
            return response + "\n" + render_remediation_reply(
                f"已完成 {completed}/{total} 条；下一条无法完整展示，已停止后续写入"
            )
        next_args: Dict[str, Any] = {"items": next_chunk}
        if next_remaining:
            next_args[_REPLACE_CHAR_CONTINUATION_KEY] = (
                _sealed_replace_char_continuation(
                    next_chunk,
                    next_remaining,
                    old_char,
                    new_char,
                    completed=completed,
                    total=total,
                )
            )
        target_key: ConversationKey = conv_key or (platform, user_id)
        saved = conversation_state_store.set(
            target_key,
            PendingToolConfirm(
                function_name="keytao_batch_add_to_draft",
                args=next_args,
                confirmation_source="local_preview",
            ),
            space_key=space_key,
            owner_label=owner_label,
        )
        if not saved:
            return response + "\n" + render_remediation_reply(
                f"已完成 {completed}/{total} 条；剩余进度无法保存，已停止后续写入"
            )
        next_end = completed + len(next_chunk)
        progress = (
            f"替换进度：已完成 {completed}/{total} 条。"
            f"接下来确认第 {completed + 1}-{next_end} 条"
        )
        if next_remaining:
            progress += f"；之后还剩 {len(next_remaining)} 条"
        return response + "\n\n" + progress + "。\n" + next_confirmation

    if state.function_name == "keytao_submit_batch":
        if data.get("success"):
            batch_url = data.get("batchUrl", "")
            pr_url = data.get("prUrl", "")
            parts = ["✅ 草稿已提交审核。"]
            if data.get("autoApproved"):
                parts = ["✅ 草稿已加入词库。"]
            if batch_url:
                parts.append(f"\n草稿地址：{batch_url}")
            if pr_url:
                parts.append(f"PR：{pr_url}")
            from .chat_render import finalize_draft_receipt
            return submitted_vote_receipt(data) or finalize_draft_receipt("\n".join(parts), data, platform=platform)
        if data.get("uncertain"):
            return _append_batch_url_if_missing(
                "⚠️ " + str(
                    data.get("message")
                    or render_remediation_reply(
                        "提交结果暂时无法确定",
                        command="查看草稿",
                    )
                ),
                data,
            )
        conflict_failure = _format_submit_conflict_failure(data)
        if conflict_failure:
            return _append_batch_url_if_missing(conflict_failure, data)
        return _append_batch_url_if_missing(
            f"提交失败：{data.get('message', '未知错误')}",
            data,
        )

    if state.function_name == "keytao_batch_add_to_draft":
        if data.get("not_bound"):
            return _BIND_HELP_TEXT
        if data.get("success") or data.get("successCount", 0) > 0:
            expected_count = len(state.args.get("items", []))
            success_count = int(data.get("successCount") or 0)
            failed_count = int(data.get("failedCount") or 0)
            partial = failed_count > 0 or success_count != expected_count
            header = "⚠️ 仅部分加入草稿\n" if partial else "✅ 已加入草稿\n"
            response = header + await _format_draft_response(
                data, platform, user_id, compact=True,
            )
            if partial:
                suffix = f"\n仅成功加入 {success_count}/{expected_count} 条"
                if submit_after:
                    suffix += "，已停止后续提交"
                return prepend_pending_word_reminders(
                    response + suffix + "。",
                    data.get("pendingItems"),
                )
            if replace_at_code_details is not None:
                old_word = replace_at_code_details["oldWord"]
                new_word = replace_at_code_details["newWord"]
                code = replace_at_code_details["code"]
                receipt = (
                    "✅ 已按确认计划写入草稿："
                    f"删除「{old_word}」@ {code}；"
                    f"创建「{new_word}」@ {code}。"
                )
                remaining_codes = replace_at_code_details["remainingCodes"]
                if remaining_codes:
                    receipt += (
                        f"\n「{old_word}」仍保留在 "
                        + "、".join(remaining_codes)
                        + "。"
                    )
                response = receipt + "\n" + await _format_draft_response(
                    data,
                    platform,
                    user_id,
                    compact=True,
                )
                return prepend_pending_word_reminders(
                    response,
                    data.get("pendingItems"),
                )
            response = stage_next_replace_char_chunk(response)
            if submit_after:
                return await continue_with_submit_preview(response)
            return prepend_pending_word_reminders(
                response,
                data.get("pendingItems"),
            )
        failure = _append_batch_url_if_missing(
            f"添加失败：{data.get('message', '未知错误')}",
            data,
        )
        return prepend_pending_word_reminders(
            failure,
            data.get("pendingItems"),
        )

    if state.function_name in {
        "keytao_remove_draft_item",
        "keytao_batch_remove_draft_items",
        "keytao_shift_phrase_code",
        "keytao_recall_batch",
    }:
        if data.get("success"):
            if state.function_name == "keytao_recall_batch":
                conversation_state_store.invalidate_actor_related(
                    (platform, user_id),
                    batch_id=str(data.get("batchId") or args.get("batch_id") or ""),
                )
            if state.function_name == "keytao_batch_remove_draft_items":
                expected_count = len(state.args.get("ids", []))
                success_count = int(data.get("successCount") or 0)
                failed_count = int(data.get("failedCount") or 0)
                if failed_count > 0 or success_count != expected_count:
                    return _append_batch_url_if_missing(
                        render_remediation_reply(
                            f"批量删除只完成 {success_count}/{expected_count} 条；"
                            "已停止后续提交",
                            command="查看草稿",
                        ),
                        data,
                    )
            if state.function_name == "keytao_shift_phrase_code":
                data["requestedWords"] = receipt_requested_words or [str(state.args.get("word") or "")]
                response = _format_ranked_shift_success(data)
                if submit_after:
                    shift_plan = (
                        data.get("shiftPlan")
                        if isinstance(data.get("shiftPlan"), dict)
                        else {}
                    )
                    authorized_items = (
                        shift_plan.get("items")
                        if isinstance(shift_plan.get("items"), list)
                        and shift_plan.get("items")
                        else None
                    )
                    return await continue_with_submit_preview(
                        response,
                        authorized_items=authorized_items,
                    )
                return response
            response = "✅ 操作已完成\n" + await _format_draft_response(
                data,
                platform,
                user_id,
                compact=True,
            )
            if expected_keep_words:
                list_data = await _fetch_current_draft_items(
                    platform,
                    user_id,
                    batch_id=str(data.get("batchId") or args.get("batch_id") or ""),
                )
                current_words = [
                    _draft_item_word(item)
                    for item in list_data.get("items", [])
                    if isinstance(item, dict)
                ] if list_data.get("success") else []
                if (
                    not list_data.get("success")
                    or set(current_words) != set(expected_keep_words)
                    or any(word not in expected_keep_words for word in current_words)
                ):
                    return response + "\n" + render_remediation_reply(
                        "删除后草稿未精确匹配保留清单，已停止提交",
                        command="查看草稿",
                    )
            if submit_after:
                return await continue_with_submit_preview(response)
            return response
        uncertain_message = str(data.get("message") or "").strip()
        return _append_batch_url_if_missing(
            (
                uncertain_message
                or render_remediation_reply(
                    "操作结果不确定",
                    command="查看草稿",
                )
                if data.get("uncertain")
                else f"操作失败：{data.get('message', '未知错误')}"
            ),
            data,
        )

    if data.get("success"):
        header = "✅ 已确认添加到草稿\n"
        response = header + await _format_draft_response(
            data, platform, user_id, compact=True,
        )
        if submit_after:
            return await continue_with_submit_preview(response)
        return prepend_pending_word_reminders(
            response,
            data.get("pendingItems"),
        )
    failure = _append_batch_url_if_missing(
        f"操作失败：{data.get('message', '未知错误')}",
        data,
    )
    return prepend_pending_word_reminders(
        failure,
        data.get("pendingItems"),
    )


async def _format_draft_response(
    data: Dict,
    platform: str,
    user_id: str,
    batch_id: Optional[str] = None,
    *,
    compact: bool = False,
) -> str:
    """Format a compact mutation result or the full current-draft view."""
    # "Current draft" is an implicit pointer to the newest draft batch, and a
    # read can move it.  Whenever this turn knows which batch it just operated
    # on, the data is read from that batch, and a drifted pointer is reported
    # rather than silently rendered as an empty draft.
    #
    # The one unanchored read is the pointer probe itself, and it doubles as the
    # preview: when the pointer agrees with the anchor (the normal case) its
    # result is exactly what we wanted, so nothing is fetched twice.
    anchor = str(batch_id or data.get("batchId") or "").strip()
    preview = json.loads(await call_tool_function(
        "keytao_get_batch_preview", {}, platform, user_id
    ))
    pointer_batch_id = ""
    if anchor:
        previewed_batch_id = str(preview.get("batchId") or "")
        if previewed_batch_id and previewed_batch_id != anchor:
            pointer_batch_id = previewed_batch_id
            preview = json.loads(await call_tool_function(
                "keytao_get_batch_preview", {"batch_id": anchor}, platform, user_id
            ))

    snapshot = data.get("draft_snapshot")
    list_batch_url = ""
    list_data = {}
    if not snapshot:
        list_json = await call_tool_function(
            "keytao_list_draft_items",
            {"batch_id": anchor} if anchor else {},
            platform,
            user_id,
        )
        list_data = json.loads(list_json)
        if list_data.get("success"):
            list_batch_url = str(list_data.get("batchUrl") or "")
            snapshot = {
                "count": list_data.get("count", 0),
                "items": list_data.get("items", []),
                "summary": list_data.get("summary", {}),
            }

    parts: List[str] = []
    if pointer_batch_id:
        parts.append("⚠️ 当前草稿已切换；以下为操作批次。")

    parts.extend(_create_notice_lines(data))

    # Notes from Delete operations
    for note in data.get("notes", []):
        nw = note.get("word", "")
        nc = note.get("code", "")
        nt = note.get("type", "")
        type_label = {"Phrase": "词组", "Single": "单字"}.get(nt, nt)
        parts.append(f"📝 注意：{nw}（{nc}，{type_label}）已从词库标记删除")

    # Summary line
    summary = None
    if snapshot:
        summary = snapshot.get("summary")
    if not summary and preview.get("success"):
        summary = preview.get("summary")
    if summary:
        parts.append(
            f"+{summary.get('added', 0)} 新增  "
            f"~{summary.get('modified', 0)} 修改  "
            f"-{summary.get('deleted', 0)} 删除"
        )

    # Diff block
    diff_text = preview.get("diff_text", "") if preview.get("success") else ""
    if diff_text and not compact:
        parts.append(f"\n{diff_text}")

    # Draft items
    draft_count: Optional[int] = None
    if snapshot and compact:
        items = snapshot.get("items", [])
        if isinstance(items, list) and items:
            visible_items = [
                f"{str(item.get('word') or '').strip()} {str(item.get('code') or '').strip()}"
                for item in items[:5]
                if isinstance(item, dict)
                and str(item.get("word") or "").strip()
                and str(item.get("code") or "").strip()
            ]
            if visible_items:
                remainder = len(items) - len(visible_items)
                suffix = f" 等 {len(items)} 条" if remainder > 0 else ""
                parts.append("草稿内容：" + "、".join(visible_items) + suffix)
    elif snapshot:
        items = snapshot.get("items", [])
        count = snapshot.get("count", len(items))
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            draft_count = count
        parts.append(f"当前草稿（{count} 条）：")
        for index, item in enumerate(items, start=1):
            parts.append(_draft_item_display_line(item, index))

    # Batch URL
    batch_url = data.get("batchUrl") or preview.get("batchUrl", "") or list_batch_url
    if batch_url:
        parts.append(f"草稿地址：{batch_url}")

    if not compact and draft_count != 0:
        parts.append("提交前请核对上述草稿内容。")
    from .chat_render import finalize_draft_receipt
    return finalize_draft_receipt("\n".join(parts), data, preview, list_data, platform=platform)


async def _submit_current_draft(
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> str:
    result = await _perform_submit_current_draft(
        platform,
        user_id,
        auto_confirm=True,
        authorize_current_draft=True,
    )
    if result.pending_state is not None:
        conversation_state_store.set(
            (platform, user_id),
            result.pending_state,
            space_key=space_key,
            owner_label=owner_label,
        )
    return result.text


def _normalized_submit_items(
    items: object,
) -> Optional[List[Tuple[str, str, str]]]:
    """Return complete action/word/code rows, or nothing when one row is unusable."""
    if not isinstance(items, list) or not items:
        return None
    normalized: List[Tuple[str, str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        action = str(item.get("action") or "Create").strip()
        word = str(item.get("word") or "").strip()
        code = str(item.get("code") or "").strip().lower()
        if action not in {"Create", "Change", "Delete"} or not word or not code:
            return None
        normalized.append((action, word, code))
    return normalized


def _submit_preview_matches_authorized_items(
    submit_data: Dict,
    authorized_items: Optional[List[Dict]],
) -> bool:
    """Only auto-confirm a submit preview with the exact authorized item set."""
    if not authorized_items:
        return False
    expected = _normalized_submit_items(authorized_items)
    actual = _normalized_submit_items(submit_data.get("snapshotItems"))
    return (
        expected is not None
        and actual is not None
        and sorted(actual) == sorted(expected)
    )


async def _submit_preview_is_authors_own(submit_data: Dict) -> bool:
    """Bind row authorship to the actor-owned, versioned server submit ticket."""
    from ..utils import http_client

    batch_id = submit_data.get("batchId")
    version = submit_data.get("contentVersion")
    items = submit_data.get("snapshotItems")
    ticket = _pending_state_from_server_warning(
        PendingToolConfirm("keytao_submit_batch", {}), submit_data,
    )
    if (not server_warning_ticket_is_complete(ticket)
            or not isinstance(batch_id, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", batch_id)
            or type(version) is not int or not _normalized_submit_items(items)):
        return False
    try:
        # The submit endpoint checks creatorId against the authenticated actor;
        # its projected snapshot omits userId, so read it from the same batch.
        data = await http_client.keytao_json("GET", f"/api/batches/{batch_id}", timeout=10.0)
        batch = data.get("batch")
        if (not isinstance(batch, dict) or batch.get("id") != batch_id
                or type(batch.get("contentVersion")) is not int or batch["contentVersion"] != version
                or batch.get("status") not in {"Draft", "Rejected"}
                or type(batch.get("creatorId")) is not int or batch["creatorId"] <= 0):
            return False
        rows = batch.get("pullRequests")
        if not isinstance(rows, list) or len(rows) != len(items):
            return False
        by_id = {}
        for row in rows:
            if (not isinstance(row, dict) or type(row.get("id")) is not int
                    or row["id"] in by_id or type(row.get("userId")) is not int
                    or row["userId"] != batch["creatorId"]):
                return False
            by_id[row["id"]] = row
        seen = set()
        for item in items:
            item_id = item.get("id")
            if type(item_id) is not int or item_id in seen or item_id not in by_id:
                return False
            seen.add(item_id)
            if any(item.get(field) != by_id[item_id].get(field)
                   for field in ("action", "word", "code", "type", "oldWord", "remark")):
                return False
        return True
    except Exception:
        return False


def submit_snapshot_extra_items(
    submit_data: Dict,
    authorized_items: Optional[List[Dict]],
) -> List[Tuple[str, str]]:
    """Return the draft rows a submit would carry beyond this turn's own writes."""
    expected = _normalized_submit_items(authorized_items)
    actual = _normalized_submit_items(submit_data.get("snapshotItems"))
    if expected is None or actual is None:
        return []
    remaining = list(expected)
    extras: List[Tuple[str, str]] = []
    for entry in actual:
        if entry in remaining:
            remaining.remove(entry)
            continue
        extras.append((entry[1], entry[2]))
    return extras


_SUBMIT_SCOPE_NOTICE_LIMIT = 8
_SUBSET_SUBMIT_RE = re.compile(
    r"^(?:只|仅)\s*(?:提交|提审)\s*(?P<target>[^\n?？]{0,64})$"
)


def _submit_scope_notice(extras: List[Tuple[str, str]]) -> str:
    """Name the rows this turn did not write, so 「确认」 is never a surprise."""
    if not extras:
        return ""
    shown = extras[:_SUBMIT_SCOPE_NOTICE_LIMIT]
    lines = ["这批草稿里还有本轮之外的内容，提交会一并提交："]
    lines.extend(f"• {word} → {code}" for word, code in shown)
    if len(extras) > len(shown):
        lines.append(f"• 另有 {len(extras) - len(shown)} 条，见草稿地址")
    lines.append(
        "提交只能整批进行，不能只提交本轮的词；引用本条回复「确认」一并提交。"
    )
    return "\n".join(lines)


def subset_submit_refusal(message: str, state: object) -> str:
    """Answer 「只提交 X」 truthfully: a batch submit has no per-item scope."""
    if (
        not isinstance(state, PendingToolConfirm)
        or state.function_name != "keytao_submit_batch"
        or not server_warning_ticket_is_complete(state)
        # A plain submit command is a full-batch assent, never a subset ask.
        or _is_explicit_draft_submit_request(message)
    ):
        return ""
    match = _SUBSET_SUBMIT_RE.match(
        unicodedata.normalize("NFKC", str(message or "")).strip()
    )
    if match is None:
        return ""
    target = match.group("target").strip().strip("「」『』\"'“”‘’ ")
    named = f"只提交「{target}」" if target else "只提交其中一部分"
    return _assert_plain_user_facing_reply(
        f"提交是整批操作，无法{named}。"
        "回复「确认」提交整批，或回复「查看草稿」先整理草稿。"
    )


async def _perform_submit_current_draft(
    platform: str,
    user_id: str,
    *,
    confirmed: bool = False,
    batch_id: str = "",
    expected_content_version: Optional[int] = None,
    expected_server_snapshot_digest: str = "",
    expected_warning_digest: str = "",
    expected_audit_digest: str = "",
    preview_only: bool = True,
    auto_confirm: bool = False,
    authorized_items: Optional[List[Dict]] = None,
    authorized_content_version: Optional[int] = None,
    authorize_current_draft: bool = False,
) -> DraftActionResult:
    """Submit a draft without writing follow-up state into the conversation slot."""
    # A confirmed server ticket is the mutation stage. Keeping the public
    # default here would otherwise send confirmed=true and previewOnly=true,
    # causing the server to issue another preview ticket forever.
    preview_only = not confirmed
    arguments: Dict[str, Any] = {}
    if batch_id:
        arguments["batch_id"] = batch_id
    if preview_only:
        arguments["preview_only"] = True
    if confirmed:
        arguments["confirmed"] = True
        arguments["expected_content_version"] = expected_content_version
        arguments["expected_server_snapshot_digest"] = expected_server_snapshot_digest
        arguments["expected_warning_digest"] = expected_warning_digest
        arguments["expected_audit_digest"] = expected_audit_digest
    submit_json = await call_tool_function("keytao_submit_batch", arguments, platform, user_id)
    submit_data = json.loads(submit_json)

    if submit_data.get("not_bound"):
        return DraftActionResult(_BIND_HELP_TEXT, data=submit_data)

    if submit_data.get("requiresConfirmation"):
        pending_state = _pending_state_from_server_warning(
            PendingToolConfirm(function_name="keytao_submit_batch", args=arguments),
            submit_data,
        )
        expected = _normalized_submit_items(authorized_items)
        remaining = list(_normalized_submit_items(submit_data.get("snapshotItems")) or [])
        for item in expected or []:
            if item not in remaining:
                remaining = []
                break
            remaining.remove(item)
        own_leftovers = bool(
            auto_confirm and not confirmed and expected and remaining
            and (authorized_content_version is None
                 or submit_data.get("contentVersion") == authorized_content_version)
            and await _submit_preview_is_authors_own(submit_data)
        )
        if (
            auto_confirm
            and not confirmed
            and (
                _semantic_direct_words.get() is None
                or server_warning_ticket_is_complete(pending_state)
                and _semantic_warnings_are_named(submit_data, arguments)
            )
            and (
                (
                    _submit_preview_matches_authorized_items(submit_data, authorized_items)
                    and (
                        authorized_content_version is None
                        or (
                            type(submit_data.get("contentVersion")) is int
                            and submit_data["contentVersion"] == authorized_content_version
                        )
                    )
                )
                or (
                    authorize_current_draft
                    and isinstance(submit_data.get("snapshotItems"), list)
                    and bool(submit_data["snapshotItems"])
                )
                or own_leftovers
            )
        ):
            exact_args = pending_state.args
            confirmed_result = await _perform_submit_current_draft(
                platform,
                user_id,
                confirmed=True,
                batch_id=str(exact_args.get("batch_id") or ""),
                expected_content_version=exact_args.get(
                    "expected_content_version"
                ),
                expected_server_snapshot_digest=str(
                    exact_args.get("expected_server_snapshot_digest") or ""
                ),
                expected_warning_digest=str(
                    exact_args.get("expected_warning_digest") or ""
                ),
                expected_audit_digest=str(
                    exact_args.get("expected_audit_digest") or ""
                ),
            )
            if own_leftovers and confirmed_result.success:
                current = list(expected or [])
                submitted = []
                for item in submit_data["snapshotItems"]:
                    key = _normalized_submit_items([item])[0]
                    earlier = key not in current
                    if not earlier:
                        current.remove(key)
                    submitted.append({**item, "earlierRound": earlier})
                receipt_data = {
                    **{key: submit_data[key] for key in (
                        "batchId", "batchUrl", "contentVersion", "snapshotItems", "autoReview",
                    ) if key in submit_data},
                    **(confirmed_result.data or {}), "submittedItems": submitted,
                }
                vote_receipt = submitted_vote_receipt(receipt_data)
                if vote_receipt:
                    receipt_text = vote_receipt
                else:
                    receipt_text = confirmed_result.text + "\n提交内容：\n" + "\n".join(
                        f"• {item['action']} {item['word']} → {item['code']}"
                        + ("（本轮之前）" if item["earlierRound"] else "")
                        for item in submitted
                    )
                confirmed_result = replace(confirmed_result, text=receipt_text, data=receipt_data)
            return _preserve_action_result_link(
                confirmed_result,
                submit_data,
                label="批次地址" if confirmed_result.success else "草稿地址",
            )
        warning_prompt = _format_server_warning_confirmation(
            "keytao_submit_batch",
            submit_data,
        )
        if len(warning_prompt) > MAX_REPLACE_CONFIRMATION_CHARS:
            return DraftActionResult(
                _append_batch_url_if_missing(
                    render_remediation_reply(
                        "提交内容过长，本次未确认；请缩小范围",
                        command="查看草稿",
                    ),
                    submit_data,
                ),
                data=submit_data,
            )
        # The scope notice is a bounded prefix added after the length gate: a
        # batch that was confirmable must not become unconfirmable because we
        # explained what else it carries.
        scope_notice = _submit_scope_notice(
            submit_snapshot_extra_items(submit_data, authorized_items)
        )
        if scope_notice:
            warning_prompt = scope_notice + "\n" + warning_prompt
        return DraftActionResult(
            _append_batch_url_if_missing(warning_prompt, submit_data),
            pending_state=pending_state,
            data=submit_data,
        )

    if submit_data.get("uncertain"):
        return DraftActionResult(
            _append_batch_url_if_missing(
                "⚠️ " + str(
                    submit_data.get("message")
                    or render_remediation_reply(
                        "提交结果暂时无法确定",
                        command="查看草稿",
                    )
                ),
                submit_data,
            ),
            data=submit_data,
        )

    if not submit_data.get("success"):
        conflict_failure = _format_submit_conflict_failure(submit_data)
        if conflict_failure:
            return DraftActionResult(
                _append_batch_url_if_missing(conflict_failure, submit_data),
                data=submit_data,
            )
        return DraftActionResult(
            _append_batch_url_if_missing(
                f"提交失败：{submit_data.get('message', '未知错误')}",
                submit_data,
            ),
            data=submit_data,
        )

    batch_url = submit_data.get("batchUrl", "")
    pr_url = submit_data.get("prUrl", "")
    parts = ["✅ 批次已提交审核。"]
    if submit_data.get("autoApproved"):
        parts = ["✅ 批次已加入词库。"]
    if batch_url:
        parts.append(f"批次地址：{batch_url}")
    if pr_url:
        parts.append(f"PR：{pr_url}")
    from .chat_render import finalize_draft_receipt
    return DraftActionResult(
        submitted_vote_receipt(submit_data) or finalize_draft_receipt("\n".join(parts), submit_data, platform=platform),
        success=True, data=submit_data,
    )


async def _perform_active_operation_confirmation(
    operation: ActiveDraftOperation,
    platform: str,
    user_id: str,
) -> DraftActionResult:
    """Resume a background draft operation after its owner confirms."""
    pending_state = operation.pending_state
    if not isinstance(pending_state, PendingToolConfirm):
        return DraftActionResult(render_remediation_reply(
            "这次后台操作没有可确认的步骤",
            command="查看草稿",
        ))

    if pending_state.function_name == "keytao_submit_batch":
        args = pending_state.args
        return await _perform_submit_current_draft(
            platform,
            user_id,
            confirmed=pending_state.confirmation_source == "server_warning",
            batch_id=str(args.get("batch_id") or ""),
            expected_content_version=args.get("expected_content_version"),
            expected_server_snapshot_digest=str(
                args.get("expected_server_snapshot_digest") or ""
            ),
            expected_warning_digest=str(args.get("expected_warning_digest") or ""),
            expected_audit_digest=str(args.get("expected_audit_digest") or ""),
        )

    if pending_state.function_name == "keytao_create_phrase" and operation.kind == "add_and_submit":
        args = pending_state.args
        return await _perform_add_to_draft_and_submit(
            str(args.get("word") or operation.word),
            str(args.get("code") or operation.code),
            platform,
            user_id,
            remark=str(args.get("remark") or operation.remark),
            needs_manual_review=args.get("needs_manual_review"),
            confirmed_create=(
                pending_state.confirmation_source == "server_warning"
            ),
            batch_id=str(args.get("batch_id") or ""),
            expected_content_version=args.get("expected_content_version"),
            expected_warning_digest=str(args.get("expected_warning_digest") or ""),
            auto_confirm=True,
        )

    if (
        pending_state.function_name == "keytao_batch_add_to_draft"
        and operation.kind == "batch_add_and_submit"
    ):
        args = pending_state.args
        return await _perform_batch_add_to_draft_and_submit(
            args.get("items", []),
            platform,
            user_id,
            batch_id=str(args.get("batch_id") or ""),
            confirmed_add=pending_state.confirmation_source == "server_warning",
            expected_content_version=args.get("expected_content_version"),
            expected_warning_digest=str(args.get("expected_warning_digest") or ""),
            auto_confirm=True,
            allow_same_code=args.get("_allow_same_code") is True,
        )

    return DraftActionResult(render_remediation_reply(
        "这次后台操作无法继续",
        command="查看草稿",
    ))


def _active_operation_reply_matches(
    operation: ActiveDraftOperation,
    reply_reference: ReplyReferenceInfo,
) -> bool:
    """Return whether a quoted bot message belongs to an active operation prompt."""
    if operation.status != "awaiting_confirmation" or not reply_reference.is_to_bot:
        return False
    if operation.prompt_text:
        return bool(
            reply_reference.is_reply
            and _prompt_capability_digest(reply_reference.text)
            == _prompt_capability_digest(operation.prompt_text)
        )
    referenced_state = _parse_pending_state_from_response(reply_reference.text)
    if conversation_state_store.states_equivalent(referenced_state, operation.pending_state):
        return True
    referenced_text = reply_reference.text or ""
    return bool(
        operation.word
        and operation.word in referenced_text
        and (
            "确认添加吗" in referenced_text
            or "是否继续提交" in referenced_text
            or "确认继续提交吗" in referenced_text
        )
    )


async def _fetch_current_draft_items(
    platform: str,
    user_id: str,
    *,
    batch_id: str = "",
) -> Dict:
    arguments = {"batch_id": batch_id} if batch_id else {}
    list_json = await call_tool_function(
        "keytao_list_draft_items",
        arguments,
        platform,
        user_id,
    )
    try:
        return json.loads(list_json)
    except Exception:
        return {"success": False, "message": "草稿工具返回了无法解析的结果"}


def _draft_snapshot_from_list_data(list_data: Dict) -> Dict:
    items = list_data.get("items", [])
    return {
        "count": list_data.get("count", len(items) if isinstance(items, list) else 0),
        "items": items if isinstance(items, list) else [],
        "summary": list_data.get("summary", {}),
    }


async def _try_handle_draft_view_command(
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
) -> Optional[str]:
    if command_intent.intent != "draft_view":
        return None

    list_data = await _fetch_current_draft_items(platform, user_id)
    if list_data.get("not_bound"):
        note_rule_use("C:_try_handle_draft_view_command")
        return _BIND_HELP_TEXT
    if not list_data.get("success"):
        response = _append_batch_url_if_missing(
            f"查看草稿失败：{list_data.get('message', '未知错误')}",
            list_data,
        )
        if response:
            note_rule_use("C:_try_handle_draft_view_command")
        return response

    data = {
        "draft_snapshot": _draft_snapshot_from_list_data(list_data),
        "batchUrl": list_data.get("batchUrl", ""),
    }
    response = await _format_draft_response(data, platform, user_id)
    if response:
        note_rule_use("C:_try_handle_draft_view_command")
    return response


async def _try_handle_draft_submit_command(
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> Optional[str]:
    if command_intent.intent != "draft_submit":
        return None

    return await _submit_current_draft(platform, user_id, space_key, owner_label)


def _draft_item_word(item: Dict) -> str:
    return str(item.get("word") or item.get("text") or "").strip()


def _draft_item_id(item: Dict) -> Optional[int]:
    for key in ("id", "pr_id", "prId"):
        value = item.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


def _canonical_draft_delete_target(item: Dict) -> Optional[Dict]:
    item_id = _draft_item_id(item)
    if item_id is None:
        return None
    return {
        "id": item_id,
        "word": str(item.get("word") or ""),
        "code": str(item.get("code") or ""),
        "action": str(item.get("action") or ""),
        "type": str(item.get("type") or ""),
    }


async def _perform_exact_batch_remove(
    ids: List[int],
    platform: str,
    user_id: str,
    *,
    batch_id: str = "",
    source_content_version: int,
    source_targets: List[Dict],
) -> DraftActionResult:
    """Preview and CAS-delete an already authorized exact draft-item set."""
    unique_ids = list(dict.fromkeys(
        item for item in ids
        if isinstance(item, int) and not isinstance(item, bool) and item > 0
    ))
    if len(unique_ids) != len(ids) or not unique_ids:
        return DraftActionResult("草稿条目缺少有效 ID，未执行删除。")
    if (
        not isinstance(source_content_version, int)
        or isinstance(source_content_version, bool)
        or source_content_version < 0
        or not isinstance(source_targets, list)
        or len(source_targets) != len(unique_ids)
    ):
        return DraftActionResult("草稿内容不完整，未执行删除。")

    preview_args: Dict[str, Any] = {"ids": unique_ids}
    if batch_id:
        preview_args["batch_id"] = batch_id
    preview_json = await call_tool_function(
        "keytao_batch_remove_draft_items",
        preview_args,
        platform,
        user_id,
    )
    try:
        preview_data = json.loads(preview_json)
    except Exception:
        return DraftActionResult("删除检查返回异常，未写入草稿。")

    if preview_data.get("not_bound"):
        return DraftActionResult(_BIND_HELP_TEXT)
    if not preview_data.get("requiresConfirmation"):
        if preview_data.get("success"):
            return DraftActionResult(
                _append_batch_url_if_missing(
                    str(preview_data.get("message") or "草稿条目已删除。"),
                    preview_data,
                ),
                success=True,
                data=preview_data,
            )
        return DraftActionResult(
            _append_batch_url_if_missing(
                f"清理草稿失败：{preview_data.get('message', '删除内容不完整')}",
                preview_data,
            ),
            data=preview_data,
        )

    pending_state = _pending_state_from_server_warning(
        PendingToolConfirm(
            function_name="keytao_batch_remove_draft_items",
            args=preview_args,
        ),
        preview_data,
    )
    if _semantic_direct_words.get() is not None and not _semantic_warnings_are_named(preview_data, preview_args):
        return DraftActionResult(
            _append_batch_url_if_missing(_format_server_warning_confirmation("keytao_batch_remove_draft_items", preview_data), preview_data),
            pending_state=pending_state, data=preview_data,
        )
    exact_args = _pending_execution_args(pending_state)
    expected_batch_id = str(exact_args.get("batch_id") or "")
    expected_version = exact_args.get("expected_content_version")
    expected_digest = str(exact_args.get("expected_target_digest") or "")
    expected_targets = exact_args.get("expected_targets")
    target_ids = {
        int(target.get("id"))
        for target in expected_targets or []
        if isinstance(target, dict)
        and str(target.get("id") or "").isdigit()
    }
    if (
        not expected_batch_id
        or (batch_id and expected_batch_id != batch_id)
        or not isinstance(expected_version, int)
        or isinstance(expected_version, bool)
        or expected_version < 0
        or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
        or not isinstance(expected_targets, list)
        or target_ids != set(unique_ids)
        or len(expected_targets) != len(unique_ids)
        or expected_version != source_content_version
        or expected_targets != source_targets
    ):
        return DraftActionResult(
            _append_batch_url_if_missing(
                render_remediation_reply(
                    "草稿在清空检查期间发生变化，未执行删除",
                    command="查看草稿",
                ),
                preview_data,
            ),
            data=preview_data,
        )

    confirmed_json = await call_tool_function(
        "keytao_batch_remove_draft_items",
        exact_args,
        platform,
        user_id,
    )
    try:
        confirmed_data = json.loads(confirmed_json)
    except Exception:
        return DraftActionResult(
            _append_batch_url_if_missing(
                render_remediation_reply(
                    "删除请求结果无法解析",
                    command="查看草稿",
                ),
                preview_data,
            ),
            data=preview_data,
        )
    if confirmed_data.get("requiresConfirmation"):
        return DraftActionResult(
            _append_batch_url_if_missing(
                render_remediation_reply(
                    "删除目标在执行前发生变化，已停止",
                    command="查看草稿",
                ),
                confirmed_data,
                preview_data,
            ),
            data=confirmed_data,
        )
    confirmed_batch_id = str(confirmed_data.get("batchId") or "")
    if confirmed_data.get("success") and confirmed_batch_id != expected_batch_id:
        return DraftActionResult(
            _append_batch_url_if_missing(
                render_remediation_reply(
                    "删除结果返回了不同批次，无法确认目标状态",
                    command="查看草稿",
                ),
                confirmed_data,
                preview_data,
            ),
            data=confirmed_data,
            invalidate_pending=True,
        )
    if not confirmed_data.get("success"):
        failure_text = str(
            confirmed_data.get("message")
            or ("删除结果尚不确定" if confirmed_data.get("uncertain") else "未知错误")
        )
        return DraftActionResult(
            _append_batch_url_if_missing(
                (
                    f"清理草稿结果不确定：{failure_text}"
                    if confirmed_data.get("uncertain")
                    else f"清理草稿失败：{failure_text}"
                ),
                confirmed_data,
                preview_data,
            ),
            data=confirmed_data,
            invalidate_pending=bool(confirmed_data.get("uncertain")),
        )

    success_count = confirmed_data.get("successCount", len(unique_ids))
    if (
        not isinstance(success_count, int)
        or isinstance(success_count, bool)
        or success_count != len(unique_ids)
    ):
        return DraftActionResult(
            _append_batch_url_if_missing(
                render_remediation_reply(
                    f"草稿只删除了 {success_count}/{len(unique_ids)} 条；"
                    "已停止后续操作",
                    command="查看草稿",
                ),
                confirmed_data,
                preview_data,
            ),
            data=confirmed_data,
            invalidate_pending=(
                isinstance(success_count, int)
                and not isinstance(success_count, bool)
                and success_count > 0
            ),
        )
    return DraftActionResult(
        _append_batch_url_if_missing(
            str(confirmed_data.get("message") or "草稿条目已删除。"),
            confirmed_data,
            preview_data,
        ),
        success=True,
        data=confirmed_data,
    )


async def _perform_clear_current_draft(
    platform: str,
    user_id: str,
    *,
    batch_id: str = "",
) -> DraftActionResult:
    """Clear the sender's exact current/restored draft without a second prompt."""
    list_data = await _fetch_current_draft_items(
        platform,
        user_id,
        batch_id=batch_id,
    )
    if list_data.get("not_bound"):
        return DraftActionResult(_BIND_HELP_TEXT)
    if not list_data.get("success"):
        return DraftActionResult(
            _append_batch_url_if_missing(
                f"获取草稿失败：{list_data.get('message', '未知错误')}",
                list_data,
            ),
            data=list_data,
        )

    listed_batch_id = str(list_data.get("batchId") or "")
    if batch_id and listed_batch_id != batch_id:
        return DraftActionResult(
            _append_batch_url_if_missing(
                "草稿查询返回了不同批次，未执行清空。",
                list_data,
            ),
            data=list_data,
        )
    resolved_batch_id = listed_batch_id or str(batch_id or "")
    items = list_data.get("items")
    if not isinstance(items, list):
        return DraftActionResult(
            _append_batch_url_if_missing(
                "草稿内容不完整，未执行清空。",
                list_data,
            ),
            data=list_data,
        )
    if not items:
        parts = ["✅ 当前草稿已经是空的。"]
        batch_url = _trusted_batch_url(list_data)
        if batch_url:
            parts.append(f"草稿地址：{batch_url}")
        return DraftActionResult("\n".join(parts), success=True, data=list_data)
    if not resolved_batch_id:
        return DraftActionResult(
            _append_batch_url_if_missing(
                "草稿批次不完整，未执行清空。",
                list_data,
            ),
            data=list_data,
        )

    source_content_version = list_data.get("contentVersion")
    source_targets = [
        _canonical_draft_delete_target(item)
        for item in items
        if isinstance(item, dict)
    ]
    ids = [target.get("id") for target in source_targets if target is not None]
    if (
        not isinstance(source_content_version, int)
        or isinstance(source_content_version, bool)
        or source_content_version < 0
        or len(source_targets) != len(items)
        or any(target is None for target in source_targets)
        or len(ids) != len(items)
    ):
        return DraftActionResult(
            _append_batch_url_if_missing(
                "草稿列表缺少完整的条目 ID 或版本，未执行清空。",
                list_data,
            ),
            data=list_data,
        )
    remove_result = await _perform_exact_batch_remove(
        [int(item_id) for item_id in ids],
        platform,
        user_id,
        batch_id=resolved_batch_id,
        source_content_version=source_content_version,
        source_targets=[target for target in source_targets if target is not None],
    )
    if not remove_result.success:
        return remove_result

    verify_data = await _fetch_current_draft_items(
        platform,
        user_id,
        batch_id=resolved_batch_id,
    )
    if not verify_data.get("success"):
        batch_url = _trusted_batch_url(remove_result.data or {}, list_data)
        suffix = f"\n草稿地址：{batch_url}" if batch_url else ""
        return DraftActionResult(
            render_remediation_reply(
                f"已删除 {len(ids)} 条，但未能确认草稿最终为空",
                command="查看草稿",
            ) + suffix,
            data=remove_result.data,
            invalidate_pending=True,
        )
    verified_batch_id = str(verify_data.get("batchId") or "")
    if verified_batch_id != resolved_batch_id:
        batch_url = _trusted_batch_url(
            remove_result.data or {},
            list_data,
        )
        suffix = f"\n草稿地址：{batch_url}" if batch_url else ""
        return DraftActionResult(
            render_remediation_reply(
                "删除已经执行，但核验接口返回了不同批次",
                command="查看草稿",
            ) + suffix,
            data=verify_data,
            invalidate_pending=True,
        )
    remaining = verify_data.get("items")
    if not isinstance(remaining, list) or remaining:
        remaining_count = len(remaining) if isinstance(remaining, list) else "未知"
        batch_url = _trusted_batch_url(
            verify_data,
            remove_result.data or {},
            list_data,
        )
        suffix = f"\n草稿地址：{batch_url}" if batch_url else ""
        return DraftActionResult(
            f"已删除 {len(ids)} 条，但草稿仍有 {remaining_count} 条；"
            f"已停止，不会继续操作。{suffix}",
            data=verify_data,
            invalidate_pending=True,
        )

    parts = [f"✅ 已清空草稿，共删除 {len(ids)} 条。"]
    batch_url = _trusted_batch_url(
        verify_data,
        remove_result.data or {},
        list_data,
    )
    if batch_url:
        parts.append(f"草稿地址：{batch_url}")
    return DraftActionResult(
        "\n".join(parts),
        success=True,
        data=verify_data,
        invalidate_pending=True,
    )


def _quoted_draft_display_lines(response: str) -> List[str]:
    lines = [
        re.sub(r"\s+", " ", match.group(0)).strip()
        for match in re.finditer(
            r"(?m)^\s*•\s*\d+\.\s*(?:新增|修改|删除)\s+.+$",
            str(response or ""),
        )
    ]
    if lines:
        note_rule_use("C:_quoted_draft_display_lines:re.sub")
        note_rule_use("C:_quoted_draft_display_lines:re.finditer")
    return lines


def _quoted_draft_selection_request(
    message_text: str,
    items: List[Dict],
) -> Optional[Tuple[str, object]]:
    raw = _strip_command_message_prefixes(message_text).strip()
    if (
        not raw
        or re.search(r"[?？\"'“”‘’「」『』]", raw)
        or re.search(r"(?:不要|别|无需|不用|算了|取消)", raw)
    ):
        return None
    compact = _compact_command_text(raw)
    prefix = r"(?:请|麻烦|帮我|给我|现在|立即|直接|确认|我要|我想|替我|为我)*"
    ordinal_patterns = (
        rf"{prefix}(?:删除|删掉|移除)(?:第)?([0-9一二三四五六七八九十两]+)(?:条|个)?",
        rf"{prefix}(?:把|将)?(?:第)?([0-9一二三四五六七八九十两]+)(?:条|个)?(?:删除|删掉|移除)",
    )
    for pattern in ordinal_patterns:
        match = re.fullmatch(pattern, compact)
        if match:
            index = _parse_pending_choice_index(match.group(1))
            return ("index", index) if index is not None else None

    all_target = r"(?:上面|引用里|引用中的)?(?:这些|全部|所有)(?:草稿)?(?:条目|内容)?"
    if re.fullmatch(
        rf"{prefix}(?:(?:把|将)?{all_target}(?:都|全部)?(?:删除|删掉|移除)|(?:删除|删掉|移除){all_target})",
        compact,
    ):
        return "all", True

    for item in items:
        word = _draft_item_word(item)
        if word and re.fullmatch(
            rf"{prefix}(?:只|仅)(?:保留|留下|留){re.escape(word)}",
            compact,
        ):
            return "keep", word
    return None


async def _try_handle_quoted_draft_selection(
    message_text: str,
    reply_reference: ReplyReferenceInfo,
    platform: str,
    user_id: str,
) -> Optional[str]:
    """Apply an ordinal/all/keep selection only to an unchanged bot draft list."""
    if not (
        reply_reference.is_reply
        and reply_reference.is_to_bot
        and reply_reference.text
    ):
        return None
    quoted_lines = _quoted_draft_display_lines(reply_reference.text)
    if not quoted_lines:
        return None
    compact_message = _compact_command_text(message_text)
    if not re.search(r"删除|删掉|移除|只保留|仅保留|只留下|仅留下|只留|仅留", compact_message):
        return None

    list_data = await _fetch_current_draft_items(platform, user_id)
    if list_data.get("not_bound"):
        return _BIND_HELP_TEXT
    if not list_data.get("success"):
        return _append_batch_url_if_missing(
            f"查看草稿失败：{list_data.get('message', '未知错误')}",
            list_data,
        )
    items = list_data.get("items")
    if not isinstance(items, list):
        return render_remediation_reply(
            "当前草稿内容不完整，本次未执行",
            command="查看草稿",
        )

    selection = _quoted_draft_selection_request(message_text, items)
    if selection is None:
        return None
    expected_lines = [
        re.sub(r"\s+", " ", _draft_item_display_line(item, index)).strip()
        for index, item in enumerate(items, start=1)
        if isinstance(item, dict)
    ]
    if quoted_lines != expected_lines:
        return render_remediation_reply(
            "引用的草稿列表已变化，本次未执行",
            command="查看草稿",
        )

    active_operation = draft_operation_coordinator.find_for_actor((platform, user_id))
    if active_operation is not None:
        return _active_operation_message_for_request(active_operation, platform, user_id)

    kind, value = selection
    if kind == "all":
        result = await _perform_clear_current_draft(platform, user_id)
    else:
        selected_items: List[Dict]
        if kind == "index":
            index = int(value or 0)
            if index < 1 or index > len(items):
                return render_remediation_reply(
                    f"只接受 1-{len(items)} 之间的草稿编号",
                    command="查看草稿",
                )
            selected_items = [items[index - 1]]
        else:
            keep_matches = [item for item in items if _draft_item_word(item) == value]
            if len(keep_matches) != 1:
                return render_remediation_reply(
                    "无法唯一确定要保留的词条；没有执行删除",
                    command="查看草稿",
                )
            selected_items = [item for item in items if item is not keep_matches[0]]
            if not selected_items:
                return render_remediation_reply(
                    f"当前草稿只剩「{value}」，不需要删除"
                )

        source_targets = [
            _canonical_draft_delete_target(item)
            for item in selected_items
            if isinstance(item, dict)
        ]
        source_version = list_data.get("contentVersion")
        ids = [target.get("id") for target in source_targets if target is not None]
        if (
            not isinstance(source_version, int)
            or isinstance(source_version, bool)
            or source_version < 0
            or len(source_targets) != len(selected_items)
            or any(target is None for target in source_targets)
            or len(ids) != len(selected_items)
        ):
            return render_remediation_reply(
                "当前草稿缺少完整 ID 或版本；没有执行删除",
                command="查看草稿",
            )
        result = await _perform_exact_batch_remove(
            [int(item_id) for item_id in ids],
            platform,
            user_id,
            batch_id=str(list_data.get("batchId") or ""),
            source_content_version=source_version,
            source_targets=[target for target in source_targets if target is not None],
        )

    if result.success or result.invalidate_pending:
        conversation_state_store.delete_actor((platform, user_id))
    return result.text


async def _perform_recall_latest_batch(
    platform: str,
    user_id: str,
    *,
    clear_after: bool = False,
) -> DraftActionResult:
    """Preview and CAS-recall the sender's latest submitted batch."""
    if clear_after:
        try:
            existing_claim = get_default_draft_mutation_claim_store().get(
                platform,
                user_id,
            )
        except Exception:
            existing_claim = None
        existing_payload = (
            existing_claim.get("payload")
            if isinstance(existing_claim, dict)
            and isinstance(existing_claim.get("payload"), dict)
            else {}
        )
        continuation_batch_id = str(existing_payload.get("batchId") or "")
        if (
            isinstance(existing_claim, dict)
            and existing_claim.get("operationKind") == "delete"
            and existing_payload.get("continuation") == "recall_clear"
            and continuation_batch_id
        ):
            continuation_ids = existing_payload.get("ids")
            continuation_version = existing_payload.get("contentVersion")
            continuation_targets = existing_payload.get("targets")
            if (
                not isinstance(continuation_ids, list)
                or not continuation_ids
                or any(
                    not isinstance(item_id, int)
                    or isinstance(item_id, bool)
                    or item_id <= 0
                    for item_id in continuation_ids
                )
                or not isinstance(continuation_version, int)
                or isinstance(continuation_version, bool)
                or continuation_version < 0
                or not isinstance(continuation_targets, list)
                or len(continuation_targets) != len(continuation_ids)
            ):
                return DraftActionResult(
                _append_batch_url_if_missing(
                    render_remediation_reply(
                        "最近提审此前已经撤回，但清空安全记录不完整；"
                        "没有执行新的删除",
                        command="查看草稿",
                    ),
                        existing_payload,
                    ),
                    invalidate_pending=True,
                )
            continuation_token = current_recall_clear_batch_id.set(
                continuation_batch_id
            )
            try:
                clear_result = await _perform_exact_batch_remove(
                    continuation_ids,
                    platform,
                    user_id,
                    batch_id=continuation_batch_id,
                    source_content_version=continuation_version,
                    source_targets=continuation_targets,
                )
            finally:
                current_recall_clear_batch_id.reset(continuation_token)
            if not clear_result.success:
                return DraftActionResult(
                    _append_batch_url_if_missing(
                        "最近提审此前已经撤回，但清空仍未完成。\n"
                        f"{clear_result.text}",
                        clear_result.data or {},
                        existing_payload,
                    ),
                    data=clear_result.data,
                    invalidate_pending=True,
                )

            verify_data = await _fetch_current_draft_items(
                platform,
                user_id,
                batch_id=continuation_batch_id,
            )
            verified_items = verify_data.get("items")
            if (
                not verify_data.get("success")
                or str(verify_data.get("batchId") or "") != continuation_batch_id
                or not isinstance(verified_items, list)
                or verified_items
            ):
                remaining_count = (
                    len(verified_items)
                    if isinstance(verified_items, list)
                    else "未知"
                )
                return DraftActionResult(
                    _append_batch_url_if_missing(
                        "最近提审此前已经撤回，原删除操作也已完成，"
                        f"但原批次当前仍有 {remaining_count} 条；"
                        + render_remediation_reply(
                            "不会删除随后出现的新条目",
                            command="查看草稿",
                        ),
                        verify_data,
                        clear_result.data or {},
                        existing_payload,
                    ),
                    data=verify_data,
                    invalidate_pending=True,
                )
            return DraftActionResult(
                _append_batch_url_if_missing(
                    "✅ 已确认最近提审此前已撤回，并清空恢复后的草稿。",
                    verify_data,
                    clear_result.data or {},
                    existing_payload,
                ),
                success=True,
                data=verify_data,
                invalidate_pending=True,
            )

    preview_json = await call_tool_function(
        "keytao_recall_batch",
        {},
        platform,
        user_id,
    )
    try:
        preview_data = json.loads(preview_json)
    except Exception:
        return DraftActionResult("撤回检查返回异常，未执行撤回。")
    if preview_data.get("not_bound"):
        return DraftActionResult(_BIND_HELP_TEXT)
    already_applied = bool(
        preview_data.get("success") and preview_data.get("alreadyApplied")
    )
    if not preview_data.get("requiresConfirmation") and not already_applied:
        return DraftActionResult(
            _append_batch_url_if_missing(
                f"撤回失败：{preview_data.get('message', '没有找到可撤回的提交批次')}",
                preview_data,
            ),
            data=preview_data,
        )

    if already_applied:
        confirmed_data = preview_data
        exact_batch_id = str(confirmed_data.get("batchId") or "")
        if not exact_batch_id:
            return DraftActionResult(
                "撤回核验结果缺少原批次 ID，已停止后续操作。",
                data=confirmed_data,
                invalidate_pending=True,
            )
    else:
        pending_state = _pending_state_from_server_warning(
            PendingToolConfirm(function_name="keytao_recall_batch", args={}),
            preview_data,
        )
        if _semantic_direct_words.get() is not None and not _semantic_warnings_are_named(preview_data, {}):
            return DraftActionResult(
                _append_batch_url_if_missing(_format_server_warning_confirmation("keytao_recall_batch", preview_data), preview_data),
                pending_state=pending_state, data=preview_data,
            )
        exact_args = _pending_execution_args(pending_state)
        exact_batch_id = str(exact_args.get("batch_id") or "")
        exact_version = exact_args.get("expected_content_version")
        if (
            not exact_batch_id
            or not isinstance(exact_version, int)
            or isinstance(exact_version, bool)
            or exact_version < 0
        ):
            return DraftActionResult(
                _append_batch_url_if_missing(
                    "撤回检查缺少精确批次版本，未执行撤回。",
                    preview_data,
                ),
                data=preview_data,
            )

        confirmed_json = await call_tool_function(
            "keytao_recall_batch",
            exact_args,
            platform,
            user_id,
        )
        try:
            confirmed_data = json.loads(confirmed_json)
        except Exception:
            return DraftActionResult(
                _append_batch_url_if_missing(
                    render_remediation_reply(
                        "撤回结果无法解析；网页核对属于站外操作",
                        command="查看草稿",
                    ),
                    preview_data,
                ),
                data=preview_data,
            )
    if not confirmed_data.get("success"):
        failure_text = str(confirmed_data.get("message") or "未知错误")
        return DraftActionResult(
            _append_batch_url_if_missing(
                (
                    f"撤回结果不确定：{failure_text}"
                    if confirmed_data.get("uncertain")
                    else f"撤回失败：{failure_text}"
                ),
                confirmed_data,
                preview_data,
            ),
            data=confirmed_data,
            invalidate_pending=bool(confirmed_data.get("uncertain")),
        )
    if str(confirmed_data.get("batchId") or "") != exact_batch_id:
        return DraftActionResult(
            _append_batch_url_if_missing(
                render_remediation_reply(
                    "撤回结果返回了不同批次，无法确认状态；"
                    "打开原批次属于站外操作",
                    command="查看草稿",
                ),
                preview_data,
            ),
            data=confirmed_data,
            invalidate_pending=True,
        )

    if clear_after:
        continuation_token = current_recall_clear_batch_id.set(exact_batch_id)
        try:
            clear_result = await _perform_clear_current_draft(
                platform,
                user_id,
                batch_id=exact_batch_id,
            )
        finally:
            current_recall_clear_batch_id.reset(continuation_token)
        if clear_result.success:
            clear_lines = [
                line for line in clear_result.text.splitlines()[1:]
                if line.strip()
            ]
            return DraftActionResult(
                "\n".join([
                    "✅ 已撤回最近提审，并清空恢复后的草稿。",
                    *clear_lines,
                ]),
                success=True,
                data=clear_result.data,
                invalidate_pending=True,
            )
        return DraftActionResult(
            _append_batch_url_if_missing(
                "✅ 最近提审已经撤回并恢复为草稿，但清空没有完成。\n"
                f"{clear_result.text}",
                clear_result.data or {},
                confirmed_data,
                preview_data,
            ),
            data=confirmed_data,
            invalidate_pending=True,
        )

    formatted = await _format_draft_response(
        confirmed_data,
        platform,
        user_id,
        batch_id=exact_batch_id,
        compact=True,
    )
    return DraftActionResult(
        "✅ 已撤回最近提审，批次已恢复为草稿。\n" + formatted,
        success=True,
        data=confirmed_data,
        invalidate_pending=True,
    )


async def _try_handle_draft_recall_command(
    message_text: str,
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> Optional[str]:
    if not _message_authorizes_draft_recall(message_text, command_intent):
        return None
    canonical = _canonical_draft_management_command(message_text)
    active_operation = draft_operation_coordinator.find_for_actor(
        (platform, user_id)
    )
    if active_operation is not None:
        return _active_operation_message_for_request(
            active_operation,
            platform,
            user_id,
        )
    result = await _perform_recall_latest_batch(
        platform,
        user_id,
        clear_after=bool(canonical is not None and canonical.clear_after),
    )
    if result.pending_state is not None:
        conversation_state_store.set((platform, user_id), result.pending_state, space_key=space_key, owner_label=owner_label)
    if result.success or result.invalidate_pending:
        # Only the tickets planned against the recalled batch are void; an
        # unrelated pending choice must survive the user following our own
        # advice to recall first.
        conversation_state_store.invalidate_actor_related(
            (platform, user_id),
            batch_id=str((result.data or {}).get("batchId") or ""),
        )
    return result.text


async def _try_handle_draft_clear_command(
    message_text: str,
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> Optional[str]:
    if not _message_authorizes_draft_clear(message_text, command_intent):
        return None
    active_operation = draft_operation_coordinator.find_for_actor(
        (platform, user_id)
    )
    if active_operation is not None:
        return _active_operation_message_for_request(
            active_operation,
            platform,
            user_id,
        )
    result = await _perform_clear_current_draft(platform, user_id)
    if result.pending_state is not None:
        conversation_state_store.set((platform, user_id), result.pending_state, space_key=space_key, owner_label=owner_label)
    if result.success or result.invalidate_pending:
        conversation_state_store.delete_actor((platform, user_id))
    return result.text


async def _list_draft_items_after_optional_recall(
    command: KeepOnlyDraftCommand,
    platform: str,
    user_id: str,
) -> Tuple[Dict, Optional[str]]:
    list_data = await _fetch_current_draft_items(platform, user_id)
    if list_data.get("not_bound"):
        return list_data, None
    if not list_data.get("success"):
        return list_data, None

    return list_data, None


async def _try_handle_keep_only_draft_items_command(
    message_text: str,
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> Optional[str]:
    command = _canonical_keep_only_command(message_text, command_intent)
    if command is None:
        return None

    list_data, recall_note = await _list_draft_items_after_optional_recall(command, platform, user_id)
    if list_data.get("not_bound"):
        note_rule_use("C:_try_handle_keep_only_draft_items_command")
        return _BIND_HELP_TEXT
    if not list_data.get("success"):
        if recall_note:
            response = _append_batch_url_if_missing(recall_note, list_data)
            if response:
                note_rule_use("C:_try_handle_keep_only_draft_items_command")
            return response
        response = _append_batch_url_if_missing(
            f"获取草稿失败：{list_data.get('message', '未知错误')}",
            list_data,
        )
        if response:
            note_rule_use("C:_try_handle_keep_only_draft_items_command")
        return response

    items = list_data.get("items", [])
    if not isinstance(items, list) or not items:
        if recall_note:
            response = _append_batch_url_if_missing(recall_note, list_data)
            if response:
                note_rule_use("C:_try_handle_keep_only_draft_items_command")
            return response
        response = _append_batch_url_if_missing(
            "当前没有可处理的草稿条目。",
            list_data,
        )
        if response:
            note_rule_use("C:_try_handle_keep_only_draft_items_command")
        return response

    keep_set = set(command.keep_words)
    kept_items = [item for item in items if isinstance(item, dict) and _draft_item_word(item) in keep_set]
    if not kept_items:
        keep_label = "、".join(command.keep_words)
        response = _append_batch_url_if_missing(
            f"草稿里没找到「{keep_label}」，我不会删除其他条目。",
            list_data,
        )
        if response:
            note_rule_use("C:_try_handle_keep_only_draft_items_command")
        return response

    delete_items = [
        item for item in items
        if isinstance(item, dict) and _draft_item_word(item) not in keep_set
    ]
    delete_ids = [
        item_id for item_id in (_draft_item_id(item) for item in delete_items)
        if item_id is not None
    ]
    missing_id_count = len(delete_items) - len(delete_ids)
    if missing_id_count > 0:
        response = _append_batch_url_if_missing(
            "草稿列表里有条目缺少内部 ID，我先不批量删除，避免误删。",
            list_data,
        )
        if response:
            note_rule_use("C:_try_handle_keep_only_draft_items_command")
        return response

    keep_label = "、".join(command.keep_words)
    if delete_ids:
        response = await _execute_confirmed_tool(
            PendingToolConfirm(
                function_name="keytao_batch_remove_draft_items",
                args={
                    "ids": delete_ids,
                    "_submit_after": command.submit_after,
                    "_expected_keep_words": list(command.keep_words),
                },
                confirmation_source="local_preview",
            ),
            platform,
            user_id,
            (platform, user_id),
            space_key,
            owner_label,
        )
        if response:
            note_rule_use("C:_try_handle_keep_only_draft_items_command")
        return response
    else:
        remove_data = {
            "success": True,
            "successCount": 0,
            "draft_snapshot": _draft_snapshot_from_list_data(list_data),
            "batchUrl": list_data.get("batchUrl", ""),
        }

    deleted_count = int(remove_data.get("successCount") or len(delete_ids))
    prefix_parts = []
    if recall_note:
        prefix_parts.append(recall_note)
    if deleted_count > 0:
        prefix_parts.append(f"✅ 已只保留「{keep_label}」，从草稿删除 {deleted_count} 条。")
    else:
        prefix_parts.append(f"✅ 草稿里已经只保留「{keep_label}」。")

    if command.submit_after:
        submit_response = await _submit_current_draft(platform, user_id, space_key, owner_label)
        note_rule_use("C:_try_handle_keep_only_draft_items_command")
        return "\n".join([*prefix_parts, submit_response])

    response = "\n".join(prefix_parts) + "\n" + await _format_draft_response(
        remove_data, platform, user_id, compact=True,
    )
    if response:
        note_rule_use("C:_try_handle_keep_only_draft_items_command")
    return response


def _chain_reorder_ask(code: str, reason: str) -> str:
    return _assert_plain_user_facing_reply(
        f"编码 {code} 无法确定常用度顺序：{reason}。"
    )


def _server_codes_for_exact_word(data: Dict[str, Any], word: str) -> Tuple[str, ...]:
    """Read only exact word/code rows from one successful lookup response."""
    phrases = data.get("phrases")
    if data.get("success") is not True or not isinstance(phrases, list):
        return ()
    codes: List[str] = []
    for phrase in phrases:
        if not isinstance(phrase, dict):
            return ()
        phrase_word = str(phrase.get("word") or "").strip()
        code = str(phrase.get("code") or "").strip().lower()
        if phrase_word != word or not re.fullmatch(r"[a-z]{1,12}", code):
            continue
        if code not in codes:
            codes.append(code)
    return tuple(codes)


async def _try_handle_entry_swap_command(
    message_text: str,
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> Optional[str]:
    """Resolve both entries from the merged view and seal one exact swap."""
    command = parse_entry_swap(message_text)
    expected_words = (
        (command.first_word, command.second_word)
        if command is not None
        else ()
    )
    if (
        command is None
        or command_intent.intent != "entry_swap"
        or command_intent.keep_words != expected_words
    ):
        return None
    return await try_handle_model_entry_swap(
        command.first_word, command.second_word,
        platform, user_id, space_key, owner_label,
    )


async def try_handle_model_entry_swap(
    first_word: str,
    second_word: str,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> str:
    """Resolve a validated pair into the existing author-confirmed swap plan."""
    expected_words = (first_word, second_word)
    set_turn_flow("draft-op")
    resolved = await _load_merged_word_entries(
        expected_words,
        platform,
        user_id,
    )
    if resolved is None:
        return render_remediation_reply(
            "服务端无法为这两个词各锁定唯一的词库或草稿记录；"
            "本次未生成换位计划"
        )
    return await _execute_entry_swap_plan(
        expected_words, resolved, platform, user_id, space_key, owner_label,
    )


async def _execute_entry_swap_plan(
    expected_words: Tuple[str, str],
    resolved: Dict[str, Dict[str, Any]],
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> str:
    """Build the shared swap plan from the same snapshot used to decide intent."""
    set_turn_flow("draft-op")
    first_word, second_word = expected_words
    first = resolved[first_word]
    second = resolved[second_word]
    first_code = str(first.get("code") or "").strip().lower()
    second_code = str(second.get("code") or "").strip().lower()
    if str(first.get("type") or "") != str(second.get("type") or ""):
        return render_remediation_reply(
            "两个词的词库类型不同，不能共用一张换位计划"
        )
    if first_code == second_code:
        first_weight = first.get("weight")
        second_weight = second.get("weight")
        if (
            not isinstance(first_weight, int)
            or isinstance(first_weight, bool)
            or not isinstance(second_weight, int)
            or isinstance(second_weight, bool)
            or first_weight == second_weight
        ):
            return render_remediation_reply(
                "两个词当前权重无法形成唯一互换计划"
            )
        return await _execute_shift_to_code(
            first_word,
            first_code,
            platform,
            user_id,
            space_key,
            owner_label,
            ordered_words=list(expected_words),
            listed_words=list(expected_words),
            expected_codes=[first_code, first_code],
            expected_weights=[second_weight, first_weight],
            evidence_lines=["按本轮指令互换这两个词的现有优先级"],
        )
    first_weight = first.get("weight")
    second_weight = second.get("weight")
    target_root = min((first_code, second_code), key=lambda value: (len(value), value))
    if (
        not second_code.startswith(target_root)
        or not isinstance(first_weight, int)
        or isinstance(first_weight, bool)
        or not isinstance(second_weight, int)
        or isinstance(second_weight, bool)
    ):
        return render_remediation_reply(
            "两个词不在同一条可验证候选链，不能共用一张换位计划"
        )
    return await _execute_shift_to_code(
        first_word,
        target_root,
        platform,
        user_id,
        space_key,
        owner_label,
        ordered_words=list(expected_words),
        listed_words=list(expected_words),
        expected_codes=[second_code, first_code],
        expected_weights=[second_weight, first_weight],
        evidence_lines=["按本轮指令互换这两个词的现有编码和权重"],
    )


def _validated_dictionary_delete_entries(
    data: Dict[str, Any],
    word: str,
) -> Optional[List[Dict[str, Any]]]:
    """Return exact typed live rows from one complete word lookup."""
    phrases = data.get("phrases")
    if data.get("success") is not True or not isinstance(phrases, list):
        return None
    entries: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for phrase in phrases:
        if not isinstance(phrase, dict):
            return None
        phrase_word = str(phrase.get("word") or word).strip()
        code = str(phrase.get("code") or "").strip().lower()
        phrase_type = str(phrase.get("type") or "").strip()
        identity = (phrase_word, code, phrase_type)
        if (
            phrase_word != word
            or re.fullmatch(r"[a-z]{1,12}", code) is None
            or phrase_type not in _PHRASE_TYPE_BASE_WEIGHTS
            or identity in seen
        ):
            return None
        seen.add(identity)
        entries.append({
            "word": phrase_word,
            "code": code,
            "type": phrase_type,
        })
    return entries


async def _try_handle_dictionary_delete_command(
    message_text: str,
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> Optional[str]:
    """Stage a live dictionary Delete, or remove an exact pending Create."""
    command = parse_dictionary_delete_command(message_text)
    if (
        command is None
        or command_intent.intent != "dictionary_delete"
        or command_intent.keep_words != (command.word,)
        or command_intent.requested_code != command.code
    ):
        return None
    set_turn_flow("draft-op")
    lookup_json = await call_tool_function(
        "keytao_lookup_by_word",
        {"word": command.word},
        platform,
        user_id,
    )
    try:
        lookup_data = json.loads(lookup_json)
    except Exception:
        lookup_data = {}
    entries = _validated_dictionary_delete_entries(lookup_data, command.word)
    if entries is None:
        return render_remediation_reply(
            f"无法取得「{command.word}」的完整词库记录；本次未生成删除草稿"
        )
    matches = [
        entry
        for entry in entries
        if not command.code or entry["code"] == command.code
    ]
    if not matches:
        draft_data = {}
        if command.code:
            try:
                draft_data = await _fetch_current_draft_items(platform, user_id)
            except Exception:
                pass
        if isinstance(draft_data, dict) and draft_data.get("not_bound"):
            return _BIND_HELP_TEXT
        draft_items = (
            draft_data["items"]
            if isinstance(draft_data, dict) and draft_data.get("success") is True
            and isinstance(draft_data.get("items"), list) else []
        )
        draft_matches = [
            item for item in draft_items
            if isinstance(item, dict) and item.get("action") == "Create"
            and _draft_item_word(item) == command.word
            and item.get("code") == command.code
        ]
        if draft_matches:
            targets = [_canonical_draft_delete_target(item) for item in draft_matches]
            if len(targets) != 1 or any(target is None for target in targets):
                return "草稿中有多条匹配记录或条目 ID 不完整，请先查看草稿；本次未执行删除。"
            target = targets[0]
            result = await _perform_exact_batch_remove(
                [target["id"]], platform, user_id,
                batch_id=str(draft_data.get("batchId") or ""),
                source_content_version=draft_data.get("contentVersion"),
                source_targets=[target],
            )
            if result.pending_state is not None and not conversation_state_store.set(
                (platform, user_id), result.pending_state, space_key=space_key, owner_label=owner_label,
            ):
                return "草稿条目删除确认无法保存，本次未执行删除。"
            return f"「{command.word}」@ {target['code']} 是草稿中的待添加条目。\n" + result.text
        qualifier = f" @ {command.code}" if command.code else ""
        return _assert_plain_user_facing_reply(
            f"词库没有可删除的「{command.word}」{qualifier}。"
        )
    if len(matches) != 1:
        if _semantic_direct_words.get() is not None:
            from ..harness.operation_request import OperationRequest

            options = tuple(OperationRequest(op="delete", operands=(command.word,), codes=(entry["code"],)) for entry in matches)
            if len({option.codes for option in options}) == len(options):
                return await try_handle_model_command_options(
                    options, "", normalize_conversation_key((platform, user_id), space_key), space_key, owner_label,
                )
        choices = "、".join(
            f"{entry['code']}（{entry['type']}）" for entry in matches
        )
        suggestions = [
            render_executable_plan_suggestion(PendingToolConfirm(
                function_name="keytao_batch_add_to_draft",
                args={"items": [{
                    "action": "Delete",
                    "word": entry["word"],
                    "code": entry["code"],
                    "type": entry["type"],
                }]},
            ))
            for entry in matches
        ]
        suggestions = [value for value in suggestions if value]
        return _assert_plain_user_facing_reply("\n".join([
            f"「{command.word}」对应多条词库记录：{choices}；本次未写入。",
            *(suggestions or ["请先发送「查看草稿」核对现状。"]),
        ]))
    target = matches[0]
    return await _execute_confirmed_tool(
        PendingToolConfirm(
            function_name="keytao_batch_add_to_draft",
            args={
                "items": [{
                    "action": "Delete",
                    "word": target["word"],
                    "code": target["code"],
                    "type": target["type"],
                }],
            },
            confirmation_source="local_preview",
        ),
        platform,
        user_id,
        (platform, user_id),
        space_key,
        owner_label,
    )


async def _try_handle_replace_at_code_command(
    message_text: str,
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> Optional[str]:
    """Review and seal one exact Delete old + Create new replacement."""
    command = parse_replace_at_code(message_text)
    if (
        command is None
        or command_intent.intent != "replace_at_code"
        or command_intent.keep_words != (command.old_word, command.new_word)
        or command_intent.requested_code != command.code
    ):
        return None
    set_turn_flow("draft-op")
    conv_key = normalize_conversation_key((platform, user_id), space_key)
    lookup_json = await call_tool_function(
        "keytao_lookup_by_word",
        {"word": command.old_word},
        platform,
        user_id,
    )
    try:
        lookup_data = json.loads(lookup_json)
    except Exception:
        lookup_data = {}
    entries = _validated_dictionary_delete_entries(
        lookup_data,
        command.old_word,
    )
    if entries is None:
        return render_remediation_reply(
            f"无法取得「{command.old_word}」的完整词库记录；"
            "本次未生成替换计划"
        )
    matches = [entry for entry in entries if entry["code"] == command.code]
    if len(matches) != 1:
        return _assert_plain_user_facing_reply(
            f"词库无法唯一锁定「{command.old_word}」@{command.code}；"
            "本次未写入。"
        )
    target = matches[0]
    remaining_codes = list(dict.fromkeys(
        entry["code"]
        for entry in entries
        if entry["code"] != command.code
    ))

    previous = conversation_state_store.get_record(conv_key)
    try:
        reviewed_reply = await _try_handle_simple_single_word_query(
            command.new_word,
            platform,
            user_id,
            conv_key,
            space_key,
            owner_label,
            _literal_word=True,
        )
    finally:
        record = conversation_state_store.get_record(conv_key)
        if (
            previous is not None and isinstance(previous.state, PendingToolConfirm)
            and previous.state.function_name == "keytao_submit_batch"
            and record is not previous
            and (record is None or (
                not record.execution_id and isinstance(record.state, PendingAddWord)
                and record.state.word == command.new_word
            ))
        ):
            # Reviewing candidates is read-only; restore the exact original authorization before any write.
            if not conversation_state_store.put_back(previous) and record is not None:
                conversation_state_store.complete_execution(record)
    reviewed_state = record.state if record is not None else None
    if (
        not isinstance(reviewed_state, PendingAddWord)
        or reviewed_state.word != command.new_word
        or not reviewed_state.server_candidates
    ):
        return reviewed_reply or render_remediation_reply(
            f"无法取得「{command.new_word}」的完整审词记录；"
            "本次未生成替换计划"
        )
    candidates = dict(reviewed_state.server_candidates)
    reviewed_pinyin, _reviewed_codes = _pending_reviewed_reading(
        reviewed_state,
        command.code,
    )
    if command.code not in candidates or not reviewed_pinyin:
        if conversation_state_store.get_record(conv_key) is record:
            conversation_state_store.complete_execution(record)
        return _assert_plain_user_facing_reply(
            f"编码 {command.code} 不在「{command.new_word}」"
            "本轮服务端已核验的读音候选中；本次未生成替换计划。"
        )
    create_args = _create_phrase_args(reviewed_state, command.code)
    create_item: Dict[str, Any] = {
        "action": "Create",
        "word": command.new_word,
        "code": command.code,
        "type": target["type"],
        "needsManualReview": bool(
            create_args.get("needs_manual_review", True)
        ),
    }
    remark = str(create_args.get("remark") or "").strip()
    if remark:
        create_item["remark"] = remark
    items = [
        {
            "action": "Delete",
            "word": command.old_word,
            "code": command.code,
            "type": target["type"],
        },
        create_item,
    ]
    if not replace_at_code_items_match(
        command,
        items,
        phrase_type=target["type"],
    ):
        if conversation_state_store.get_record(conv_key) is record:
            conversation_state_store.complete_execution(record)
        return render_remediation_reply(
            "替换计划未能与本轮词库记录完整对应；本次未写入"
        )
    response = await _execute_confirmed_tool(
        PendingToolConfirm(
            function_name="keytao_batch_add_to_draft",
            args={
                "items": items,
                "_replace_at_code": {
                    "code": command.code,
                    "oldWord": command.old_word,
                    "newWord": command.new_word,
                    "remainingCodes": remaining_codes,
                },
            },
            confirmation_source="local_preview",
        ),
        platform,
        user_id,
        conv_key,
        space_key,
        owner_label,
    )
    if conversation_state_store.get_record(conv_key) is record:
        conversation_state_store.complete_execution(record)
    return response


async def _try_handle_dictionary_recode_command(
    message_text: str,
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> Optional[str]:
    """Stage an explicit same-word delete/create pair without model routing."""
    commands = [parse_dictionary_recode_command(message_text)]
    if commands[0] is None:
        commands = [parse_dictionary_recode_command(line) for line in message_text.splitlines()]
    codes = tuple(code for command in commands if command for code in (command.old_code, command.new_code))
    if (
        not commands or any(command is None for command in commands)
        or command_intent.intent != "dictionary_recode"
        or command_intent.keep_words != tuple(command.word for command in commands)
        or len(set(command_intent.keep_words)) != len(commands)
        or command_intent.requested_code != codes[0]
        or command_intent.requested_codes != codes[1:]
    ):
        return None
    set_turn_flow("draft-op")
    items = []
    for command in commands:
        lookup_json = await call_tool_function(
            "keytao_lookup_by_word", {"word": command.word}, platform, user_id,
        )
        try:
            lookup_data = json.loads(lookup_json)
        except Exception:
            lookup_data = {}
        entries = _validated_dictionary_delete_entries(lookup_data, command.word)
        if entries is None:
            return render_remediation_reply(
                f"无法取得「{command.word}」的完整词库记录；本次未生成改码草稿"
            )
        matches = [entry for entry in entries if entry["code"] == command.old_code]
        if len(matches) != 1:
            return _assert_plain_user_facing_reply(
                f"词库无法唯一锁定「{command.word}」@{command.old_code}；本次未写入。"
            )
        target = matches[0]
        items.extend([
            {"action": "Delete", "word": target["word"], "code": target["code"], "type": target["type"]},
            {"action": "Create", "word": target["word"], "code": command.new_code, "type": target["type"]},
        ])
    return await _execute_confirmed_tool(
        PendingToolConfirm(
            function_name="keytao_batch_add_to_draft",
            args={
                "items": items,
            },
            confirmation_source="local_preview",
        ),
        platform,
        user_id,
        (platform, user_id),
        space_key,
        owner_label,
    )


async def _load_merged_reorder_entries(
    words: Tuple[str, ...],
    platform: str,
    user_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """Read the validated live and draft view without discarding ambiguous rows."""
    lookup_json = await call_tool_function(
        "keytao_lookup_by_words_batch",
        {"words": list(words)},
        platform,
        user_id,
    )
    draft_json = await call_tool_function(
        "keytao_list_draft_items",
        {},
        platform,
        user_id,
    )
    try:
        lookup_data = json.loads(lookup_json)
    except Exception:
        lookup_data = {}
    try:
        draft_data = json.loads(draft_json)
    except Exception:
        draft_data = {}
    live_entries = _validated_word_lookup_entries(lookup_data, words)
    draft_snapshot = _validated_draft_reorder_snapshot(draft_data)
    if live_entries is None or draft_snapshot is None:
        return None
    return _merge_live_and_draft_reorder_entries(
        live_entries,
        draft_snapshot,
    )


async def _load_merged_word_entries(
    words: Tuple[str, ...],
    platform: str,
    user_id: str,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Resolve one unique effective row per word from live plus current draft."""
    merged = await _load_merged_reorder_entries(words, platform, user_id)
    if merged is None:
        return None
    resolved: Dict[str, Dict[str, Any]] = {}
    for word in words:
        matches = [entry for entry in merged if entry.get("word") == word]
        if len(matches) != 1:
            return None
        resolved[word] = matches[0]
    return resolved


async def try_handle_advertised_shift(
    message: str, platform: str, user_id: str, conv_key: Optional[ConversationKey],
    space_key: Optional[Tuple[str, str]], owner_label: str,
) -> Optional[str]:
    """Repair an author's same-code duplicate through the existing shift planner."""
    from ..harness.authorization_grammar import parse_reviewed_shift_words

    address = normalize_conversation_key(conv_key or (platform, user_id), space_key)
    record = conversation_state_store.get_record(address)
    known_words = tuple(
        word for occupants in record.state.server_occupied_words.values() for word in occupants
    ) if record is not None and isinstance(record.state, PendingAddWord) else ()
    words = parse_reviewed_shift_words(message, known_words=known_words)
    if not words:
        return None
    if address != normalize_conversation_key((platform, user_id), space_key):
        return "顺延请求与当前作者或对话不一致，本次未写入。"
    active = draft_operation_coordinator.find_for_actor((platform, user_id))
    if active is not None:
        return _active_operation_message_for_request(active, platform, user_id)
    if record is not None and (
        record.execution_id or isinstance(record.state, PendingToolConfirm)
        and record.state.confirmation_source == "server_warning"
    ):
        return None
    if record is not None and isinstance(record.state, PendingAddWord):
        recommendation = _pending_add_reorder_recommendation(record.state)
        if recommendation is not None and words == (recommendation["occupantWord"],):
            return await handle_pending_message_core(
                f"加入，顶替 {words[0]}", platform, user_id, address, space_key=space_key,
                owner_label=owner_label, allow_intent_model=False,
            )
    set_turn_flow("draft-op")
    try:
        lookup = json.loads(await call_tool_function(
            "keytao_lookup_by_words_batch", {"words": list(words)}, platform, user_id,
        ))
        draft = json.loads(await call_tool_function(
            "keytao_list_draft_items", {}, platform, user_id,
        ))
    except (TypeError, ValueError):
        lookup, draft = {}, {}
    if not isinstance(lookup, dict) or not isinstance(draft, dict):
        lookup, draft = {}, {}
    live = _validated_word_lookup_entries(lookup, words)
    snapshot = _validated_draft_reorder_snapshot(draft)
    merged = _merge_live_and_draft_reorder_entries(live, snapshot) if live is not None and snapshot is not None else None
    if merged is None:
        return "无法读取完整的词库和当前草稿记录，本次未执行顺延。"
    plans = []
    for word in words:
        targets = [row for row in merged if row["word"] == word]
        if len(targets) != 1:
            return f"无法唯一锁定「{word}」的现有编码，本次未执行顺延。"
        target = targets[0]
        try:
            data = json.loads(await call_tool_function(
                "keytao_lookup_by_code", {"code": target["code"]}, platform, user_id,
            ))
        except (TypeError, ValueError):
            data = {}
        if not isinstance(data, dict):
            data = {}
        phrases = data.get("phrases")
        if data.get("success") is not True or not isinstance(phrases, list):
            return f"无法核验编码 {target['code']} 的完整占用记录，本次未执行顺延。"
        code_words = tuple(dict.fromkeys(row.get("word") for row in phrases if isinstance(row, dict)))
        code_live = _validated_word_lookup_entries({
            "success": True, "results": [
                {"word": value, "phrases": [row for row in phrases if isinstance(row, dict) and row.get("word") == value]}
                for value in code_words
            ],
        }, code_words) if all(isinstance(row, dict) and row.get("code") == target["code"] for row in phrases) else None
        occupants = _merge_live_and_draft_reorder_entries(code_live, snapshot) if code_live is not None else None
        keepers = [
            row for row in occupants or []
            if row["code"] == target["code"] and row["type"] == target["type"] and row["word"] != word
        ]
        if occupants is None or len(keepers) != 1 or keepers[0]["word"] in words:
            return f"「{word}」没有可唯一确认的同码重码词条，本次未执行顺延。"
        keeper = keepers[0]
        next_code = ""
        if keeper.get("source") == "live":
            try:
                encoding = json.loads(await call_tool_function(
                    "keytao_encode", {"word": word, "requested_code": target["code"]}, platform, user_id,
                ))
            except (TypeError, ValueError):
                encoding = {}
            chain = _candidate_codes_from_encode(encoding, word) if isinstance(encoding, dict) else ()
            if target["code"] not in chain or chain.index(target["code"]) + 1 >= len(chain):
                return f"「{word}」的当前编码之后没有可核验的候选位置，本次未执行顺延。"
            next_code = chain[chain.index(target["code"]) + 1]
        plans.append((keeper, target, next_code))
    direct = "direct_exec" in {token.strip() for token in os.getenv("KEYTAO_SEMANTIC_ROUTE_OPS", "").split(",")}
    token = _semantic_direct_words.set(tuple(dict.fromkeys([*words, *(row["word"] for row, _, _ in plans)])) if direct else None)
    try:
        replies = []
        for keeper, target, next_code in plans:
            shifted_word = target["word"]
            if next_code:
                replies.append(await _execute_shift_to_code(
                    shifted_word, next_code, platform, user_id, space_key, owner_label,
                    target_item={"type": target["type"]}, auto_confirm_shift_plan=direct,
                    auto_confirm_only_if_no_shifts=True,
                    receipt_requested_words=[shifted_word, keeper["word"]],
                ))
                pending = conversation_state_store.get_record(address)
                if pending is not None and isinstance(pending.state, PendingToolConfirm) and pending.state.confirmation_source == "server_warning":
                    break
                continue
            keeper_item = {"type": keeper["type"]}
            if keeper.get("source") == "draft":
                original = next(item for item in snapshot["items"] if str(item.get("id")) == str(keeper["draftId"]))
                keeper_item.update(
                    remark=str(original.get("remark") or ""),
                    needsManualReview=original.get("needsManualReview", True),
                )
            replies.append(await _execute_shift_to_code(
                keeper["word"], keeper["code"], platform, user_id, space_key, owner_label,
                target_item=keeper_item,
                auto_confirm_shift_plan=direct,
                auto_confirm_expected_shifted_words=(shifted_word,),
                receipt_requested_words=[shifted_word, keeper["word"]],
            ))
            pending = conversation_state_store.get_record(address)
            if pending is not None and isinstance(pending.state, PendingToolConfirm) and pending.state.confirmation_source == "server_warning":
                break
        return "\n\n".join(replies)
    finally:
        _semantic_direct_words.reset(token)


async def try_handle_move_to_code_command(
    message_text: str, platform: str, user_id: str,
    space_key: Optional[Tuple[str, str]] = None, owner_label: str = "",
) -> Optional[str]:
    """Resolve a literal existing-item move before model routing and preview it."""
    from ..harness.authorization_grammar import parse_existing_entry_move

    plan = parse_entry_move_plan(message_text)
    if plan is not None:
        return await _try_handle_entry_move_plan_command(
            message_text,
            MessageCommandIntent(
                intent="entry_move_plan", confidence=1.0,
                keep_words=tuple(move.word for move in plan.moves),
                requested_codes=tuple(move.target_code for move in plan.moves),
            ), platform, user_id, space_key, owner_label,
        )
    move = parse_existing_entry_move(message_text)
    if move is None:
        return None
    set_turn_flow("draft-op")
    resolved = await _load_merged_word_entries((move.word,), platform, user_id)
    if resolved is None:
        return f"想把「{move.word}」调到 {move.target_code}，但没有查到唯一的现有词条；本次未写入。"
    if move.named_occupant:
        lookup = json.loads(await call_tool_function(
            "keytao_lookup_by_code", {"code": move.target_code}, platform, user_id,
        ))
        occupants = lookup.get("phrases") if lookup.get("success") is True else None
        if not isinstance(occupants, list) or sum(
            isinstance(row, dict) and row.get("word") == move.named_occupant
            for row in occupants
        ) != 1:
            return f"想把「{move.word}」调到 {move.target_code}，但该位置没有唯一的「{move.named_occupant}」；本次未写入。"
    return await _execute_shift_to_code(
        move.word, move.target_code, platform, user_id, space_key, owner_label,
        target_item={"type": resolved[move.word]["type"]},
    )


async def _try_handle_entry_move_plan_command(
    message_text: str,
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> Optional[str]:
    """Turn two literal target codes into one server-rechecked ring plan."""
    command = parse_entry_move_plan(message_text)
    if command is None:
        return None
    words = tuple(move.word for move in command.moves)
    target_by_word = {
        move.word: move.target_code for move in command.moves
    }
    if (
        command_intent.intent != "entry_move_plan"
        or command_intent.keep_words != words
        or command_intent.requested_codes
        != tuple(move.target_code for move in command.moves)
    ):
        return None
    set_turn_flow("draft-op")
    resolved = await _load_merged_word_entries(words, platform, user_id)
    if resolved is None:
        return render_remediation_reply(
            "无法从词库和当前草稿中锁定两条唯一记录；本次未生成调整计划"
        )
    if len({str(entry.get("type") or "") for entry in resolved.values()}) != 1:
        return render_remediation_reply(
            "两个词的词库类型不同，不能共用一张调整计划"
        )
    target_root = min(
        target_by_word.values(),
        key=lambda value: (len(value), value),
    )
    current_codes = [
        str(resolved[word].get("code") or "").strip().lower()
        for word in words
    ]
    if (
        len(set(target_by_word.values())) != len(words)
        or any(not code.startswith(target_root) for code in target_by_word.values())
        or any(not code.startswith(target_root) for code in current_codes)
    ):
        return render_remediation_reply(
            "两组目标编码不能形成唯一、可验证的同一候选链计划；本次未写入"
        )
    ordered_words = words
    expected_codes = [target_by_word[word] for word in ordered_words]
    return await _execute_shift_to_code(
        ordered_words[0],
        target_root,
        platform,
        user_id,
        space_key,
        owner_label,
        ordered_words=list(ordered_words),
        listed_words=list(words),
        expected_codes=expected_codes,
        evidence_lines=["目标编码均来自本轮指令，并由服务端候选链重新核验"],
    )


def _validated_code_chain_entries(data: Dict[str, Any], code: str) -> Optional[List[Dict[str, Any]]]:
    phrases = data.get("phrases")
    if data.get("success") is not True or not isinstance(phrases, list):
        return None
    entries: List[Dict[str, Any]] = []
    seen = set()
    for phrase in phrases:
        if not isinstance(phrase, dict):
            return None
        word = str(phrase.get("word") or "").strip()
        phrase_code = str(phrase.get("code") or "").strip().lower()
        phrase_type = str(phrase.get("type") or "").strip()
        weight = phrase.get("weight")
        identity = (phrase_type, word)
        base_weight = _PHRASE_TYPE_BASE_WEIGHTS.get(phrase_type)
        if (
            not word
            or phrase_code != code
            or base_weight is None
            or not isinstance(weight, int)
            or isinstance(weight, bool)
            or weight < base_weight
            or identity in seen
        ):
            return None
        seen.add(identity)
        entries.append({
            "word": word,
            "code": phrase_code,
            "type": phrase_type,
            "weight": weight,
        })
    entries.sort(key=lambda entry: (
        str(entry["type"]),
        int(entry["weight"]),
        str(entry["word"]),
    ))
    return entries


def _validated_draft_reorder_snapshot(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate the current draft CAS anchor and retain its full item projection."""
    items = data.get("items")
    batch_id = str(data.get("batchId") or "").strip()
    content_version = data.get("contentVersion")
    if (
        data.get("success") is not True
        or not isinstance(items, list)
        or not isinstance(content_version, int)
        or isinstance(content_version, bool)
        or content_version < 0
        or (items and not batch_id)
    ):
        return None
    return {
        "batchId": batch_id,
        "contentVersion": content_version,
        "items": items,
    }


def _draft_reorder_projection(
    draft_snapshot: Dict[str, Any],
) -> Optional[Tuple[List[Dict[str, Any]], set[Tuple[str, str, str]]]]:
    """Project draft Create/Change/Delete rows onto the live dictionary view."""
    projected: List[Dict[str, Any]] = []
    removed: set[Tuple[str, str, str]] = set()
    seen_projected: set[Tuple[str, str, str]] = set()
    for raw_item in draft_snapshot.get("items", []):
        if not isinstance(raw_item, dict):
            return None
        action = str(raw_item.get("action") or "").strip()
        word = str(raw_item.get("word") or "").strip()
        old_word = str(
            raw_item.get("oldWord") or raw_item.get("old_word") or ""
        ).strip()
        code = str(raw_item.get("code") or "").strip().lower()
        phrase_type = str(raw_item.get("type") or "").strip()
        weight = raw_item.get("weight")
        item_id = raw_item.get("id")
        base_weight = _PHRASE_TYPE_BASE_WEIGHTS.get(phrase_type)
        if (
            action not in {"Create", "Change", "Delete"}
            or not word
            or not re.fullmatch(r"[a-z]{1,12}", code)
            or base_weight is None
            or not str(item_id).isdigit()
        ):
            return None
        if action == "Change":
            if not old_word:
                return None
            removed.add((phrase_type, code, old_word))
        elif action == "Delete":
            removed.add((phrase_type, code, word))
            continue
        if (
            not isinstance(weight, int)
            or isinstance(weight, bool)
            or weight < base_weight
        ):
            return None
        identity = (phrase_type, code, word)
        if identity in seen_projected:
            return None
        seen_projected.add(identity)
        projected.append({
            "word": word,
            "code": code,
            "type": phrase_type,
            "weight": weight,
            "source": "draft",
            "draftId": int(item_id),
            "draftAction": action,
        })
    return projected, removed


def _merge_live_and_draft_reorder_entries(
    live_entries: List[Dict[str, Any]],
    draft_snapshot: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Return the effective dictionary view after applying the current draft."""
    projection = _draft_reorder_projection(draft_snapshot)
    if projection is None:
        return None
    projected, removed = projection
    merged: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()
    for raw_entry in live_entries:
        entry = dict(raw_entry)
        identity = (
            str(entry.get("type") or ""),
            str(entry.get("code") or "").lower(),
            str(entry.get("word") or ""),
        )
        if identity in removed:
            continue
        entry["source"] = "live"
        if identity in seen:
            return None
        seen.add(identity)
        merged.append(entry)
    for entry in projected:
        identity = (entry["type"], entry["code"], entry["word"])
        if identity in seen:
            return None
        seen.add(identity)
        merged.append(entry)
    merged.sort(key=lambda entry: (
        str(entry["type"]),
        str(entry["code"]),
        int(entry["weight"]),
        str(entry["word"]),
    ))
    return merged


def _validated_word_lookup_entries(
    data: Dict[str, Any],
    words: Tuple[str, ...],
) -> Optional[List[Dict[str, Any]]]:
    """Validate an exact complete batch word lookup without accepting extras."""
    results = data.get("results")
    if (
        data.get("success") is not True
        or not isinstance(results, list)
        or len(results) != len(words)
    ):
        return None
    by_word: Dict[str, List[Dict[str, Any]]] = {}
    for result in results:
        if not isinstance(result, dict):
            return None
        result_word = str(result.get("word") or "").strip()
        phrases = result.get("phrases")
        if result_word not in words or result_word in by_word or not isinstance(phrases, list):
            return None
        clean: List[Dict[str, Any]] = []
        for phrase in phrases:
            if not isinstance(phrase, dict):
                return None
            word = str(phrase.get("word") or "").strip()
            code = str(phrase.get("code") or "").strip().lower()
            phrase_type = str(phrase.get("type") or "").strip()
            weight = phrase.get("weight")
            base_weight = _PHRASE_TYPE_BASE_WEIGHTS.get(phrase_type)
            if (
                word != result_word
                or not re.fullmatch(r"[a-z]{1,12}", code)
                or base_weight is None
                or not isinstance(weight, int)
                or isinstance(weight, bool)
                or weight < base_weight
            ):
                return None
            clean.append({
                "word": word,
                "code": code,
                "type": phrase_type,
                "weight": weight,
            })
        by_word[result_word] = clean
    if set(by_word) != set(words):
        return None
    return [entry for word in words for entry in by_word[word]]


def _candidate_codes_from_encode(data: Dict[str, Any], word: str) -> Tuple[str, ...]:
    """Read the ordered candidate chain returned for one exact word."""
    returned_word = str(data.get("word") or data.get("input") or word).strip()
    if data.get("success") is not True or returned_word != word:
        return ()
    raw_codes = [
        data.get("code"),
        *(data.get("codes") or []),
        *(data.get("candidateCodes") or []),
        *(data.get("altCodes") or []),
        *(
            status.get("code")
            for status in data.get("candidateStatuses") or []
            if isinstance(status, dict)
        ),
    ]
    codes: List[str] = []
    for raw_code in raw_codes:
        code = str(raw_code or "").strip().lower()
        if re.fullmatch(r"[a-z]{1,12}", code) and code not in codes:
            codes.append(code)
    return tuple(codes)


def _shared_candidate_chain_root(
    entries: List[Dict[str, Any]],
    candidate_codes_by_word: Dict[str, Tuple[str, ...]],
) -> Optional[str]:
    """Find the shortest shared candidate prefix while validating each current code."""
    common_codes: Optional[set[str]] = None
    current_codes: List[str] = []
    for entry in entries:
        word = str(entry.get("word") or "").strip()
        current_code = str(entry.get("code") or "").strip().lower()
        candidates = candidate_codes_by_word.get(word, ())
        if not word or not current_code or current_code not in candidates:
            return None
        current_codes.append(current_code)
        candidate_set = set(candidates)
        common_codes = candidate_set if common_codes is None else common_codes & candidate_set
    roots = [
        code
        for code in common_codes or set()
        if all(current_code.startswith(code) for current_code in current_codes)
    ]
    return min(roots, key=lambda code: (len(code), code)) if roots else None


def _format_chain_order(entries: List[Dict[str, Any]]) -> str:
    return " → ".join(
        f"{entry['word']}({entry['weight']})"
        for entry in entries
    )


def _format_code_chain_reorder_confirmation(
    plan: Dict[str, Any],
    preview_data: Dict[str, Any],
) -> str:
    lines = [f"🔁 编码 {plan['code']} 调整计划："]
    groups = plan.get("groups") if isinstance(plan.get("groups"), list) else []
    for item in plan.get("moves") or []:
        if not isinstance(item, dict):
            continue
        if item.get("fromWeight") == item.get("toWeight"):
            continue
        lines.append(
            f"• 「{item.get('word') or ''}」："
            f"权重 {item.get('fromWeight')} → {item.get('toWeight')}"
        )

    evidence = [
        str(line).strip()
        for group in groups
        if isinstance(group, dict)
        for line in group.get("evidenceLines") or []
        if str(line).strip()
    ]
    summaries = [
        render_commonness_summary(comparison)
        for group in groups
        if isinstance(group, dict)
        for comparison in group.get("comparisons") or []
        if isinstance(comparison, dict) and str(comparison.get("summary") or "").strip()
    ]
    visible_evidence = list(OrderedDict.fromkeys([*evidence, *summaries]))
    evidence_parts = list(OrderedDict.fromkeys(summaries[:1]))
    compact_evidence = _compact_ranked_shift_evidence(evidence)
    if compact_evidence:
        evidence_parts.append(compact_evidence)
    elif not evidence_parts and visible_evidence:
        evidence_parts.append(_compact_ranked_shift_evidence(visible_evidence))
    if evidence_parts:
        lines.append("依据：" + "；".join(evidence_parts))

    warnings = preview_data.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(f"提示：{_plain_warning_message(warning)}" for warning in warnings)
    lines.append(pending_confirmation_copy())
    batch_url = _trusted_batch_url(preview_data)
    if batch_url:
        lines.append(f"草稿地址：{batch_url}")
    return _assert_plain_user_facing_reply("\n".join(lines))


def _reorder_evidence_lines(ranking: Dict[str, Any]) -> List[str]:
    evidence = [
        str(line).strip()
        for line in ranking.get("evidenceLines") or []
        if str(line).strip()
    ]
    summaries = [
        render_commonness_summary(comparison)
        for comparison in ranking.get("comparisons") or []
        if isinstance(comparison, dict)
        and str(comparison.get("summary") or "").strip()
    ]
    return list(OrderedDict.fromkeys([*evidence, *summaries]))


async def _load_reorder_semantic_review(
    word: str,
    platform: str,
    user_id: str,
) -> Dict[str, Any]:
    """Reuse the reviewed-add semantic evidence for every reorder shape."""
    review_json = await call_tool_function(
        "keytao_prepare_reviewed_add",
        {"word": word},
        platform,
        user_id,
    )
    try:
        review = json.loads(review_json)
    except Exception:
        return {}
    return review if isinstance(review, dict) else {}


def _ranked_words_from_complete_result(
    ranking: Dict[str, Any],
    entries: List[Dict[str, Any]],
) -> Optional[List[str]]:
    proposed = ranking.get("proposedOrder")
    if not isinstance(proposed, list) or len(proposed) != len(entries):
        return None
    current_words = [str(entry.get("word") or "").strip() for entry in entries]
    proposed_words = [
        str(entry.get("word") or "").strip()
        for entry in proposed
        if isinstance(entry, dict)
    ]
    if (
        len(proposed_words) != len(entries)
        or len(set(proposed_words)) != len(entries)
        or set(proposed_words) != set(current_words)
    ):
        return None
    return proposed_words


def _format_reorder_noop(
    *,
    code: str,
    entries: List[Dict[str, Any]],
    ranking: Dict[str, Any],
    listed_words: Tuple[str, ...] = (),
) -> str:
    """Render the unchanged order and one decisive evidence line."""
    lines = [f"编码 {code} 顺序不变：{_format_chain_order(entries)}"]
    evidence_summary: List[str] = []
    if listed_words:
        proposed_words = _ranked_words_from_complete_result(ranking, entries) or []
        relation = "一致" if tuple(proposed_words) == listed_words else "不一致"
        evidence_summary.append(
            f"建议 {'→'.join(proposed_words or listed_words)}（与你列的{relation}）"
        )
    evidence = _reorder_evidence_lines(ranking)
    compact_evidence = _compact_ranked_shift_evidence(evidence)
    if compact_evidence:
        evidence_summary.append(compact_evidence)
    if evidence_summary:
        lines.append("依据：" + "；".join(evidence_summary))
    return _assert_plain_user_facing_reply("\n".join(lines))


async def _start_generalized_reorder_plan(
    *,
    entries: List[Dict[str, Any]],
    ranking: Dict[str, Any],
    target_code: str,
    listed_words: Tuple[str, ...],
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> str:
    proposed_words = _ranked_words_from_complete_result(ranking, entries)
    if proposed_words is None:
        return _chain_reorder_ask(target_code, "排序器没有返回完整词条集合")
    if ranking.get("status") == "already_ordered":
        return _format_reorder_noop(
            code=target_code,
            entries=entries,
            ranking=ranking,
            listed_words=listed_words,
        )
    if ranking.get("status") != "reorder":
        return _chain_reorder_ask(
            target_code,
            str(ranking.get("reason") or "常用度证据不足"),
        )
    return await _execute_confirmed_tool(
        PendingToolConfirm(
            function_name="keytao_shift_phrase_code",
            args={
                "word": proposed_words[0],
                "target_code": target_code,
                "ordered_words": proposed_words,
                "listed_words": list(listed_words),
                "evidence_lines": _reorder_evidence_lines(ranking),
            },
            confirmation_source="local_preview",
        ),
        platform,
        user_id,
        (platform, user_id),
        space_key,
        owner_label,
    )


async def _try_handle_code_chain_reorder_command(
    message_text: str,
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> Optional[str]:
    command = _parse_code_chain_reorder_command(message_text)
    if (
        command is None
        or command_intent.intent != "code_chain_reorder"
        or command_intent.requested_code != command.code
    ):
        return None
    set_turn_flow("draft-op")
    code = command.code
    lookup_json = await call_tool_function(
        "keytao_lookup_by_code",
        {"code": code},
        platform,
        user_id,
    )
    try:
        lookup_data = json.loads(lookup_json)
    except Exception:
        lookup_data = {}
    live_entries = _validated_code_chain_entries(lookup_data, code)
    if live_entries is None:
        return _chain_reorder_ask(code, "服务端没有返回完整的词、类型和权重记录")

    draft_json = await call_tool_function(
        "keytao_list_draft_items",
        {},
        platform,
        user_id,
    )
    try:
        draft_data = json.loads(draft_json)
    except Exception:
        draft_data = {}
    draft_snapshot = _validated_draft_reorder_snapshot(draft_data)
    if draft_snapshot is None:
        return _chain_reorder_ask(code, "当前草稿没有返回完整的批次和内容版本")
    merged_entries = _merge_live_and_draft_reorder_entries(
        live_entries,
        draft_snapshot,
    )
    if merged_entries is None:
        return _chain_reorder_ask(code, "当前草稿里存在无法唯一投影的词条记录")
    entries = [entry for entry in merged_entries if entry.get("code") == code]
    if not entries:
        return _chain_reorder_ask(code, "词库和当前草稿的合并视图都没有词条")
    if command.focus_words:
        server_words = {str(entry.get("word") or "") for entry in entries}
        missing_words = [
            word for word in command.focus_words if word not in server_words
        ]
        if missing_words:
            return _chain_reorder_ask(
                code,
                "服务端同码链中没有「" + "、".join(missing_words) + "」",
            )
    if len(entries) > MAX_REPLACE_CHAR_ITEMS:
        return _chain_reorder_ask(
            code,
            f"完整链共有 {len(entries)} 条，超过单条确认可完整展示的 {MAX_REPLACE_CHAR_ITEMS} 条上限",
        )

    has_relevant_draft = any(entry.get("source") == "draft" for entry in entries)

    async def load_semantic_review(word: str) -> Dict[str, Any]:
        return await _load_reorder_semantic_review(word, platform, user_id)

    if has_relevant_draft:
        ranking = await keytao_review.rank_code_chain_by_commonness(
            entries,
            semantic_review_loader=load_semantic_review,
        )
        if not isinstance(ranking, dict) or ranking.get("status") == "ask":
            return _chain_reorder_ask(
                code,
                str((ranking or {}).get("reason") or "常用度证据不足"),
            )
        return await _start_generalized_reorder_plan(
            entries=entries,
            ranking=ranking,
            target_code=code,
            listed_words=command.focus_words,
            platform=platform,
            user_id=user_id,
            space_key=space_key,
            owner_label=owner_label,
        )

    groups_by_type: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for entry in entries:
        groups_by_type.setdefault(entry["type"], []).append(entry)

    groups: List[Dict[str, Any]] = []
    items: List[Dict[str, Any]] = []
    moves: List[Dict[str, Any]] = []
    any_reorder = False
    for phrase_type, current_entries in groups_by_type.items():
        ranking = await keytao_review.rank_code_chain_by_commonness(
            current_entries,
            semantic_review_loader=load_semantic_review,
        )
        if not isinstance(ranking, dict) or ranking.get("status") == "ask":
            reason = str((ranking or {}).get("reason") or "常用度证据不足")
            return _chain_reorder_ask(code, reason)
        proposed_source = ranking.get("proposedOrder")
        if not isinstance(proposed_source, list) or len(proposed_source) != len(current_entries):
            return _chain_reorder_ask(code, "排序器没有返回完整的同码集合")
        current_by_identity = {
            (entry["type"], entry["word"]): entry
            for entry in current_entries
        }
        current_position_by_identity = {
            (entry["type"], entry["word"]): index
            for index, entry in enumerate(current_entries)
        }
        proposed_entries: List[Dict[str, Any]] = []
        proposed_identities = []
        base_weight = _PHRASE_TYPE_BASE_WEIGHTS[phrase_type]
        for index, raw_entry in enumerate(proposed_source):
            if not isinstance(raw_entry, dict):
                return _chain_reorder_ask(code, "排序器返回了无效词条")
            identity = (
                str(raw_entry.get("type") or "").strip(),
                str(raw_entry.get("word") or "").strip(),
            )
            current = current_by_identity.get(identity)
            if current is None:
                return _chain_reorder_ask(code, "排序器改变了服务端词条集合")
            proposed_identities.append(identity)
            target_weight = base_weight + index
            proposed = {**current, "weight": target_weight}
            proposed_entries.append(proposed)
            if current["weight"] != target_weight:
                item = {
                    "action": "Change",
                    "old_word": current["word"],
                    "word": current["word"],
                    "code": code,
                    "type": phrase_type,
                    "weight": target_weight,
                }
                items.append(item)
            moves.append({
                "word": current["word"],
                "type": phrase_type,
                "fromWeight": current["weight"],
                "toWeight": target_weight,
                "fromPosition": current_position_by_identity[identity],
                "toPosition": index,
            })
        if len(set(proposed_identities)) != len(current_entries):
            return _chain_reorder_ask(code, "排序器改变了服务端词条集合")
        any_reorder = any_reorder or ranking.get("status") == "reorder"
        groups.append({
            "type": phrase_type,
            "current": current_entries,
            "proposed": proposed_entries,
            "comparisons": [
                {
                    key: comparison.get(key)
                    for key in (
                        "frontWord",
                        "behindWord",
                        "verdict",
                        "decisionReason",
                        "summary",
                    )
                    if comparison.get(key) is not None
                }
                for comparison in ranking.get("comparisons") or []
                if isinstance(comparison, dict)
            ],
            "evidenceLines": ranking.get("evidenceLines") or [],
        })

    if not any_reorder:
        lines = [f"编码 {code} 顺序不变。"]
        evidence: List[str] = []
        for group in groups:
            lines.append("• " + _format_chain_order(group["current"]))
            evidence.extend(
                str(item).strip()
                for item in group["evidenceLines"]
                if str(item).strip()
            )
        compact_evidence = _compact_ranked_shift_evidence(evidence)
        if compact_evidence:
            lines.append("依据：" + compact_evidence)
        return _assert_plain_user_facing_reply("\n".join(lines))
    if not items:
        return _chain_reorder_ask(code, "建议顺序没有形成可执行的权重变化")

    plan: Dict[str, Any] = {
        "code": code,
        "groups": groups,
        "moves": moves,
        "items": items,
    }
    plan["digest"] = _chain_reorder_plan_digest(plan)
    preview_json = await call_tool_function(
        "keytao_batch_add_to_draft",
        {"items": items, "preview_only": True},
        platform,
        user_id,
    )
    try:
        preview_data = json.loads(preview_json)
    except Exception:
        preview_data = {}
    if (
        preview_data.get("requiresConfirmation") is not True
        or int(preview_data.get("failedCount") or 0) != 0
        or int(preview_data.get("skippedCount") or 0) != 0
        or bool(preview_data.get("failed"))
        or bool(preview_data.get("skipped"))
        or not isinstance(preview_data.get("warnings", []), list)
    ):
        return _chain_reorder_ask(code, "修改内容不完整")
    preview_data["chainReorderPlan"] = plan
    pending = _pending_state_from_server_warning(
        PendingToolConfirm(
            function_name="keytao_batch_add_to_draft",
            args={"items": items},
        ),
        preview_data,
    )
    if not server_warning_ticket_is_complete(pending) or not _resolved_chain_reorder_items_match(pending):
        return _chain_reorder_ask(code, "确认内容不完整")

    if _semantic_direct_words.get() is not None and _semantic_warnings_are_named(preview_data, {"items": items}):
        return await _execute_confirmed_tool(pending, platform, user_id, (platform, user_id), space_key, owner_label)

    confirmation = _format_code_chain_reorder_confirmation(plan, preview_data)
    if len(confirmation) > MAX_REPLACE_CONFIRMATION_CHARS:
        return _chain_reorder_ask(code, "完整计划超出单条消息展示上限")
    saved = conversation_state_store.set(
        (platform, user_id),
        pending,
        space_key=space_key,
        owner_label=owner_label,
    )
    if not saved:
        return _chain_reorder_ask(code, "确认内容无法保存")
    return confirmation


def _format_word_resolution_rows(entries: List[Dict[str, Any]]) -> str:
    rows = []
    for entry in entries:
        source = "草稿" if entry.get("source") == "draft" else "词库"
        rows.append(
            f"{entry.get('word')}→{entry.get('code')}"
            f"（权重 {entry.get('weight')}，{source}）"
        )
    return "；".join(rows)


async def _try_handle_word_list_reorder_command(
    message_text: str,
    command_intent: MessageCommandIntent,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> Optional[str]:
    """Resolve every literal word, then choose same-code or prefix-chain scope."""
    command = _parse_word_list_reorder_command(message_text)
    if (
        command is None
        or command_intent.intent != "word_list_reorder"
        or command_intent.keep_words != command.words
    ):
        return None
    set_turn_flow("draft-op")
    words = command.words
    lookup_json = await call_tool_function(
        "keytao_lookup_by_words_batch",
        {"words": list(words)},
        platform,
        user_id,
    )
    draft_json = await call_tool_function(
        "keytao_list_draft_items",
        {},
        platform,
        user_id,
    )
    try:
        lookup_data = json.loads(lookup_json)
    except Exception:
        lookup_data = {}
    try:
        draft_data = json.loads(draft_json)
    except Exception:
        draft_data = {}
    live_entries = _validated_word_lookup_entries(lookup_data, words)
    draft_snapshot = _validated_draft_reorder_snapshot(draft_data)
    if live_entries is None or draft_snapshot is None:
        return render_remediation_reply(
            "无法取得完整词库记录或草稿版本，未生成排序计划"
        )
    merged = _merge_live_and_draft_reorder_entries(live_entries, draft_snapshot)
    if merged is None:
        return render_remediation_reply(
            "当前草稿无法确定唯一词条，未生成排序计划"
        )
    resolved: List[Dict[str, Any]] = []
    unresolved: List[str] = []
    for word in words:
        matches = [entry for entry in merged if entry.get("word") == word]
        if len(matches) != 1:
            unresolved.append(word)
        else:
            resolved.append(matches[0])
    if unresolved:
        return render_remediation_reply(
            "无法确定「" + "、".join(unresolved)
            + "」的唯一当前编码；已找到："
            + (_format_word_resolution_rows(resolved) or "无")
            + "；未生成排序计划"
        )

    phrase_types = {str(entry.get("type") or "") for entry in resolved}
    current_codes = [str(entry.get("code") or "") for entry in resolved]
    unique_codes = list(OrderedDict.fromkeys(current_codes))
    target_code = min(unique_codes, key=lambda value: (len(value), value))
    same_code_scope = len(unique_codes) == 1 and len(phrase_types) == 1
    prefix_scope = False
    if not same_code_scope:
        candidate_codes_by_word: Dict[str, Tuple[str, ...]] = {}
        for word in words:
            encode_json = await call_tool_function(
                "keytao_encode",
                {"word": word},
                platform,
                user_id,
            )
            try:
                encode_data = json.loads(encode_json)
            except Exception:
                encode_data = {}
            candidate_codes_by_word[word] = _candidate_codes_from_encode(
                encode_data,
                word,
            )
        shared_root = _shared_candidate_chain_root(
            resolved,
            candidate_codes_by_word,
        )
        prefix_scope = len(phrase_types) == 1 and shared_root is not None
        if shared_root is not None:
            target_code = shared_root
    if not same_code_scope and not prefix_scope:
        return _assert_plain_user_facing_reply(
            "这些词不在同一编码或同一候选前缀链："
            + _format_word_resolution_rows(resolved)
            + "。你想按同码权重排，还是调整各自编码？"
        )
    if len(resolved) > MAX_REPLACE_CHAR_ITEMS:
        return render_remediation_reply(
            f"完整词表共有 {len(resolved)} 条，超过单条确认可完整展示的 "
            f"{MAX_REPLACE_CHAR_ITEMS} 条上限，未生成排序计划"
        )

    ranking = await keytao_review.rank_code_chain_by_commonness(
        resolved,
        tie_break_words=words,
    )
    if not isinstance(ranking, dict) or ranking.get("status") == "ask":
        return _chain_reorder_ask(
            target_code,
            str((ranking or {}).get("reason") or "常用度证据不足"),
        )
    return await _start_generalized_reorder_plan(
        entries=resolved,
        ranking=ranking,
        target_code=target_code,
        listed_words=words,
        platform=platform,
        user_id=user_id,
        space_key=space_key,
        owner_label=owner_label,
    )


def _announce_replaced_submit(previous, conv_key, response):
    current = conversation_state_store.get_record(conv_key)
    if (
        response is not None and previous is not None
        and isinstance(previous.state, PendingToolConfirm)
        and previous.state.function_name == "keytao_submit_batch"
        and current is not None and current is not previous
        and current.owner_key == previous.owner_key
    ):
        if _candidate_origin_for_ticket(current.state):
            # Add-and-submit still previews the whole draft, including old items.
            return response
        deliveries = current_draft_delivery_claims.get()
        receipt = {
            "platform": current.owner_key.platform, "platformId": current.owner_key.actor_id,
            "operationKind": "ticket_replacement", "nonce": current.nonce, "previousNonce": previous.nonce,
        }
        if deliveries is not None and receipt not in deliveries:
            deliveries.append(receipt)
        if "原待确认提交已取消" not in response:
            return "原待确认提交已取消，已改为以下新操作。\n" + response
    return response


async def _try_handle_draft_management_command(
    message_text: str,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
    command_intent: Optional[MessageCommandIntent] = None,
) -> Optional[str]:
    conv_key = normalize_conversation_key((platform, user_id), space_key)
    previous = conversation_state_store.get_record(conv_key)
    response = await _try_handle_draft_management_command_core(
        message_text, platform, user_id, space_key, owner_label, command_intent,
    )
    return _announce_replaced_submit(previous, conv_key, response)


async def _try_handle_draft_management_command_core(
    message_text: str,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
    command_intent: Optional[MessageCommandIntent] = None,
) -> Optional[str]:
    from ..utils.same_code_reorder import try_handle_same_code_reorder
    reorder_reply = await try_handle_same_code_reorder(
        message_text, platform, user_id, space_key, owner_label,
    )
    if reorder_reply is not None:
        return reorder_reply
    compact_command = re.sub(
        r"[\s，,。.!！~～]+",
        "",
        _strip_command_message_prefixes(message_text),
    )
    if compact_command in {
        "放弃不确定操作",
        "放弃上次不确定操作",
        "放弃上一次不确定操作",
    }:
        try:
            discarded = get_default_draft_mutation_claim_store().discard_actor(
                platform,
                user_id,
            )
        except Exception as error:
            logger.error(
                "Failed to discard draft mutation fence: %s: %s",
                type(error).__name__,
                error,
            )
            return render_remediation_reply(
                "无法安全解除上一次草稿操作锁；本次没有执行任何写入",
                command="查看草稿",
            )
        if discarded is None:
            return "当前没有待核验的不确定草稿操作。"
        return render_remediation_reply(
            "✅ 已放弃上一次不确定操作的自动核验；没有执行新的草稿写入",
            command="查看草稿",
        )

    if command_intent is None:
        command_intent = await _classify_message_command_intent(message_text)

    response = await _try_handle_replace_at_code_command(
        message_text,
        command_intent,
        platform,
        user_id,
        space_key,
        owner_label,
    )
    if response is not None:
        return response

    response = await _try_handle_dictionary_recode_command(
        message_text,
        command_intent,
        platform,
        user_id,
        space_key,
        owner_label,
    )
    if response is not None:
        return response

    response = await _try_handle_dictionary_delete_command(
        message_text,
        command_intent,
        platform,
        user_id,
        space_key,
        owner_label,
    )
    if response is not None:
        return response

    response = await _try_handle_entry_move_plan_command(
        message_text,
        command_intent,
        platform,
        user_id,
        space_key,
        owner_label,
    )
    if response is not None:
        return response

    response = await _try_handle_word_list_reorder_command(
        message_text,
        command_intent,
        platform,
        user_id,
        space_key,
        owner_label,
    )
    if response is not None:
        return response

    response = await _try_handle_entry_swap_command(
        message_text,
        command_intent,
        platform,
        user_id,
        space_key,
        owner_label,
    )
    if response is not None:
        return response

    response = await _try_handle_code_chain_reorder_command(
        message_text,
        command_intent,
        platform,
        user_id,
        space_key,
        owner_label,
    )
    if response is not None:
        return response

    response = await _try_handle_draft_recall_command(
        message_text,
        command_intent,
        platform,
        user_id,
        space_key,
        owner_label,
    )
    if response is not None:
        return response

    response = await _try_handle_draft_clear_command(
        message_text,
        command_intent,
        platform,
        user_id,
        space_key,
        owner_label,
    )
    if response is not None:
        return response

    response = await _try_handle_draft_submit_command(
        command_intent if _is_explicit_draft_submit_request(message_text) else MessageCommandIntent(),
        platform,
        user_id,
        space_key,
        owner_label,
    )
    if response is not None:
        return response

    response = await _try_handle_keep_only_draft_items_command(
        message_text,
        command_intent,
        platform,
        user_id,
        space_key,
        owner_label,
    )
    if response is not None:
        return response

    return await _try_handle_draft_view_command(command_intent, platform, user_id)


def _plain_pinyin(value: str) -> str:
    source = str(value or "").lower().replace("u:", "v")
    normalized = unicodedata.normalize("NFD", source)
    result: List[str] = []
    index = 0
    while index < len(normalized):
        character = normalized[index]
        if unicodedata.combining(character):
            index += 1
            continue
        mark_index = index + 1
        marks = []
        while (
            mark_index < len(normalized)
            and unicodedata.combining(normalized[mark_index])
        ):
            marks.append(normalized[mark_index])
            mark_index += 1
        result.append(
            "v"
            if character == "u" and "\N{COMBINING DIAERESIS}" in marks
            else character
        )
        index = mark_index
    return "".join(result)


def _pending_pronunciation_correction(
    message: str,
    state: PendingAddWord,
) -> Optional[Tuple[str, str]]:
    """Extract one explicit pronunciation correction for the pending word."""
    raw = _strip_command_message_prefixes(message).strip()
    if (
        not raw
        or re.search(r"[?？\"'“”‘’「」《》【】]", raw)
        or re.search(r"(?:不要|别|不用|无需|解释|为什么|怎么|如何|假设|如果)", raw)
    ):
        return None
    matches = list(re.finditer(
        r"(?P<char>[\u3400-\u9fff])(?:字)?(?:的)?(?:读音)?"
        r"(?:应该|应当|要)?(?:读作|读成|读|是|为)\s*"
        r"(?P<pinyin>[A-Za-züÜvV:āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜńňǹḿ]{1,16})",
        raw,
    ))
    if len(matches) != 1:
        return None
    character = matches[0].group("char")
    pinyin = _plain_pinyin(matches[0].group("pinyin"))
    if character not in state.word or not re.fullmatch(r"[a-zv]{1,12}", pinyin):
        return None
    return character, pinyin


async def _try_update_pending_pronunciation(
    state: PendingAddWord,
    message: str,
    platform: str,
    user_id: str,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> Optional[str]:
    """Rebuild a pending candidate from trusted polyphone data without writing."""
    correction = _pending_pronunciation_correction(message, state)
    if correction is None:
        return None
    character, corrected_pinyin = correction
    encode_json = await call_tool_function(
        "keytao_encode",
        {"word": state.word},
        platform,
        user_id,
    )
    try:
        encoding = json.loads(encode_json)
    except Exception:
        encoding = {}
    if not encoding.get("success"):
        return render_remediation_reply(
            "读音纠正已收到，但编码服务暂时无法验证新候选；旧候选没有执行",
            command=f"加词 {state.word}",
            words=(state.word,),
        )

    variants = []
    for key in ("alternatePronunciationCodes", "alternatePhrasePronunciationCodes"):
        values = encoding.get(key)
        if isinstance(values, list):
            variants.extend(item for item in values if isinstance(item, dict))

    def variant_identity(variant: Dict) -> Tuple[object, ...]:
        normalized_pinyin = _plain_pinyin(str(variant.get("pinyin") or ""))
        normalized_codes = tuple(
            str(code)
            for code in variant.get("codes") or []
            if isinstance(code, str)
        )
        if len(state.word) == 1:
            return normalized_pinyin, normalized_codes
        raw_index = variant.get("charIndex")
        char_index = (
            raw_index
            if isinstance(raw_index, int) and not isinstance(raw_index, bool)
            else None
        )
        return (
            str(variant.get("char") or ""),
            char_index,
            normalized_pinyin,
            normalized_codes,
        )

    chars = encoding.get("chars")
    default_codes = [
        str(code)
        for code in encoding.get("codes") or []
        if isinstance(code, str)
    ]
    if isinstance(chars, list) and default_codes:
        for char_index, item in enumerate(chars):
            if not isinstance(item, dict):
                continue
            default_variant = {
                "char": str(item.get("char") or ""),
                "charIndex": char_index,
                "pinyin": str(item.get("pinyin") or ""),
                "codes": default_codes,
            }
            duplicate = any(
                variant_identity(variant) == variant_identity(default_variant)
                for variant in variants
            )
            if not duplicate:
                variants.append(default_variant)
    unique_variants: Dict[Tuple[object, ...], Dict] = {}
    for variant in variants:
        unique_variants.setdefault(variant_identity(variant), variant)
    variants = list(unique_variants.values())
    matching_variants = [
        variant
        for variant in variants
        if _plain_pinyin(str(variant.get("pinyin") or "")) == corrected_pinyin
        and (
            len(state.word) == 1
            or str(variant.get("char") or "") == character
        )
    ]
    if len(matching_variants) != 1:
        return render_remediation_reply(
            f"读音纠正已收到，但编码服务无法唯一定位「{character}」的 "
            f"{corrected_pinyin} 候选；旧候选没有执行",
            command=f"加词 {state.word}",
            words=(state.word,),
        )

    status_map = {
        str(status.get("code") or ""): status
        for status in encoding.get("candidateStatuses") or []
        if isinstance(status, dict) and status.get("code")
    }
    variant_codes = [
        str(code)
        for code in matching_variants[0].get("codes") or []
        if isinstance(code, str) and code in status_map
    ]
    if not variant_codes:
        return render_remediation_reply(
            "读音纠正已收到，但新读音的候选占用状态无法验证；旧候选没有执行",
            command=f"加词 {state.word}",
            words=(state.word,),
        )

    recommended_code = next(
        (
            code
            for code in variant_codes
            if not bool(status_map[code].get("occupied"))
        ),
        variant_codes[0],
    )
    selected_variant = matching_variants[0]
    selected_index = selected_variant.get("charIndex")
    pronunciation_parts: List[str] = []
    for key in ("contextPhrasePinyins", "phrasePinyins"):
        values = encoding.get(key)
        if isinstance(values, list) and len(values) == len(state.word):
            normalized_values = [
                _plain_pinyin(str(value or ""))
                for value in values
            ]
            if all(normalized_values):
                pronunciation_parts = normalized_values
                break
    if not pronunciation_parts and isinstance(chars, list) and len(chars) == len(state.word):
        normalized_values = [
            _plain_pinyin(str(item.get("pinyin") or ""))
            if isinstance(item, dict)
            else ""
            for item in chars
        ]
        if all(normalized_values):
            pronunciation_parts = normalized_values
    if len(state.word) == 1:
        reviewed_pinyin = corrected_pinyin
    elif (
        not isinstance(selected_index, int)
        or isinstance(selected_index, bool)
        or not 0 <= selected_index < len(state.word)
        or len(pronunciation_parts) != len(state.word)
    ):
        return render_remediation_reply(
            "读音纠正已收到，但编码服务没有返回可核验的完整整词读音；"
            "旧候选没有执行",
            command=f"加词 {state.word}",
            words=(state.word,),
        )
    else:
        pronunciation_parts[selected_index] = corrected_pinyin
        reviewed_pinyin = " ".join(pronunciation_parts)
    review_line = (
        f"读音 {reviewed_pinyin}；来源 用户当前纠正 + 编码服务多音候选；"
        "自动审核：读音纠正需人工复核，需要管理员审核"
    )
    lines = [
        f"「{state.word}」读音已纠正：",
        f"审词：{review_line}",
        "候选编码:",
    ]
    for index, code in enumerate(variant_codes, start=1):
        status = status_map[code]
        label = str(status.get("label") or "").strip()
        if not label:
            label = "已有占用" if status.get("occupied") else "空位"
        marker = " ✅ 推荐" if code == recommended_code else ""
        lines.append(f"{index}. {code} — {label}{marker}")
    lines.extend((
        f"• 「{state.word}」→ {recommended_code}（推荐）",
        single_word_candidate_footer(len(variant_codes)),
    ))
    response = "\n".join(lines)
    updated_state = _parse_pending_add_word(response)
    if updated_state is None:
        return render_remediation_reply(
            "新读音候选生成异常，旧候选没有执行",
            command=f"加词 {state.word}",
            words=(state.word,),
        )
    _attach_server_candidate_snapshot(
        updated_state,
        [status_map[code] for code in variant_codes],
    )
    stored = conversation_state_store.set(
        (platform, user_id),
        updated_state,
        space_key=space_key,
        owner_label=owner_label,
    )
    if not stored:
        return render_remediation_reply(
            "新读音候选过长，本次未执行；请缩小范围",
            command=f"加词 {state.word}",
            words=(state.word,),
        )
    return response


async def _handle_pending_add_word(
    state: PendingAddWord,
    message: str,
    platform: str,
    user_id: str,
    history: List[Dict],
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
    command_intent: Optional[MessageCommandIntent] = None,
    restore_pending: Optional[Callable[[], None]] = None,
) -> Optional[str]:
    """Keep the exact selection and its exclusions visible at both entrypoints."""
    response = await _execute_pending_add_word(
        state, message, platform, user_id, history, space_key, owner_label,
        command_intent, restore_pending,
    )
    selection = parse_pending_candidate_selection(
        _strip_command_message_prefixes(trusted_mutation_source(message))
    )
    if response is not None and selection is not None and state.server_candidates == state.candidates:
        selected = set(selection.codes) | {
            state.candidates[index - 1][0] for index in selection.indices
            if 1 <= index <= len(state.candidates)
        }
        omitted = [f"{index}. {state.word} → {code}"
                   for index, (code, _) in enumerate(state.server_candidates, 1)
                   if code not in selected]
        if selected and omitted:
            response += "\n未选择：" + "、".join(omitted) + "；这些候选本次未添加。"
    return response


async def _execute_pending_add_word(
    state: PendingAddWord,
    message: str,
    platform: str,
    user_id: str,
    history: List[Dict],
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
    command_intent: Optional[MessageCommandIntent] = None,
    restore_pending: Optional[Callable[[], None]] = None,
) -> Optional[str]:
    """Handle user response to a pending add-word prompt.

    Returns a response string if handled directly, None to fall through to AI.
    """
    msg = message.strip()
    pronunciation_response = await _try_update_pending_pronunciation(
        state,
        msg,
        platform,
        user_id,
        space_key,
        owner_label,
    )
    if pronunciation_response is not None:
        return pronunciation_response
    if command_intent is None:
        command_intent = await _classify_message_command_intent(msg, state)
    if (
        _is_sensitive_pending_control_intent(command_intent)
        and not _message_authorizes_pending_state_control(
            state,
            msg,
            command_intent,
        )
    ):
        command_intent = MessageCommandIntent()

    submit_after_add = command_intent.intent == "pending_add_and_submit"

    def reviewed_validation(code: str) -> Dict[str, Any]:
        pinyin, candidate_codes = _pending_reviewed_reading(state, code)
        return {
            "reviewed_pinyin": pinyin,
            "reviewed_candidate_codes": candidate_codes,
        }

    requested_codes = list(command_intent.requested_codes)
    if not requested_codes and _is_sensitive_pending_control_intent(command_intent):
        requested_codes = _requested_codes_from_pending_message(msg, state)
    if requested_codes:
        selected_codes = requested_codes
    elif command_intent.choice_index is not None:
        selected_codes = ([state.candidates[command_intent.choice_index - 1][0]]
                          if 1 <= command_intent.choice_index <= len(state.candidates) else [])
    else:
        target = (_resolve_shift_target_code(state, command_intent)
                  or command_intent.requested_code or state.recommended_code)
        selected_codes = [target] if target else []
    selected_code = selected_codes[0] if selected_codes else ""
    if command_intent.recode_indices:
        eviction_codes = [
            state.candidates[index - 1][0]
            for index in command_intent.recode_indices
            if 1 <= index <= len(state.candidates)
        ]
    else:
        # Multiple ordinary additions preserve existing occupants in place.
        eviction_codes = selected_codes if len(requested_codes) <= 1 else []
    protected = tuple(dict.fromkeys(
        word for code in eviction_codes for word in protected_candidate_occupants(
            state.word, code, state.server_candidates,
            state.server_occupied_words, state.server_ordering_assessments,
        )
    ))
    source_intent = _structural_pending_add_word_intent(msg, state)
    named_eviction = bool(
        source_intent is not None and source_intent.intent == "pending_recode"
        and source_intent.target_word
        and source_intent.target_word in state.server_occupied_words.get(selected_code, [])
    )
    from ..utils.offered_options import is_force_assent
    forced_selection = bool(
        is_force_assent(msg)
        and len(selected_codes) == 1
        and state.server_candidates == state.candidates
        and selected_code in dict(state.server_candidates)
        and "" not in protected
    )
    if (protected and not named_eviction and not forced_selection) or (
        not selected_code and command_intent.intent in {"pending_confirm", "pending_add_and_submit"}
    ):
        if restore_pending is not None:
            restore_pending()
        return _protected_eviction_response(state, selected_code, protected)
    if command_intent.recode_indices:
        if (
            len(command_intent.recode_indices) != 1
            or len(command_intent.choice_indices) != len(requested_codes)
            or command_intent.recode_indices[0] not in command_intent.choice_indices
        ):
            return render_remediation_reply(
                "复合候选计划不完整；本次未写入",
                command="加入",
                words=(state.word,),
            )
        recode_index = command_intent.recode_indices[0]
        recode_position = command_intent.choice_indices.index(recode_index)
        shift_code = requested_codes[recode_position]
        occupancy = dict(state.candidates)
        additional_items: List[Dict[str, Any]] = []
        resolved_plan: List[Dict[str, Any]] = []
        for index, code in zip(command_intent.choice_indices, requested_codes):
            modifier = "recode" if index == recode_index else ""
            resolved_plan.append({
                "index": index,
                "word": state.word,
                "code": code,
                "modifier": modifier,
                **(
                    {
                        "occupants": list(
                            state.server_occupied_words.get(code, [])
                        ),
                    }
                    if modifier
                    else {}
                ),
            })
            if modifier:
                continue
            item: Dict[str, Any] = {
                "word": state.word,
                "code": code,
                "action": "Create",
            }
            remark = state.code_remarks.get(code)
            if remark:
                item["remark"] = remark
            if occupancy.get(code) is True:
                item["needsManualReview"] = True
                item["manualReviewReason"] = "重码添加需管理员审核"
            elif state.needs_manual_review is not None:
                item["needsManualReview"] = bool(state.needs_manual_review)
            additional_items.append(item)

        create_args = _create_phrase_args(state, shift_code)
        reviewed_pinyin, reviewed_candidate_codes = _pending_reviewed_reading(
            state,
            shift_code,
        )
        return await _execute_shift_to_code(
            state.word,
            shift_code,
            platform,
            user_id,
            space_key,
            owner_label,
            submit_after=submit_after_add,
            target_item={
                "type": state.phrase_type,
                "remark": str(create_args.get("remark") or ""),
                "needsManualReview": bool(
                    create_args.get("needs_manual_review", True)
                ),
            },
            additional_items=additional_items,
            resolved_candidate_plan=resolved_plan,
            reviewed_pinyin=reviewed_pinyin,
            reviewed_candidate_codes=reviewed_candidate_codes,
        )
    if len(requested_codes) > 1:
        return await _execute_add_multiple_codes_to_draft(
            state,
            requested_codes,
            platform,
            user_id,
            space_key,
            owner_label,
            submit_after=submit_after_add,
        )

    shift_target_code = _resolve_shift_target_code(state, command_intent)
    if shift_target_code is not None:
        create_args = _create_phrase_args(state, shift_target_code)
        reviewed_pinyin, reviewed_candidate_codes = _pending_reviewed_reading(
            state,
            shift_target_code,
        )
        return await _execute_shift_to_code(
            state.word,
            shift_target_code,
            platform,
            user_id,
            space_key,
            owner_label,
            submit_after=submit_after_add,
            target_item={
                "type": state.phrase_type,
                "remark": str(create_args.get("remark") or ""),
                "needsManualReview": bool(
                    create_args.get("needs_manual_review", True)
                ),
            },
            reviewed_pinyin=reviewed_pinyin,
            reviewed_candidate_codes=reviewed_candidate_codes,
            auto_confirm_shift_plan=True,
        )

    reorder_recommendation = _pending_add_reorder_recommendation(state)
    if (
        reorder_recommendation is not None
        and state.recommended_code == reorder_recommendation["occupantCode"]
        and not requested_codes
        and command_intent.choice_index is None
        and command_intent.intent in {"pending_confirm", "pending_add_and_submit"}
    ):
        return await _execute_shift_to_code(
            state.word,
            reorder_recommendation["occupantCode"],
            platform,
            user_id,
            space_key,
            owner_label,
            submit_after=submit_after_add,
            target_item={
                "type": state.phrase_type,
                "remark": state.code_remarks.get(state.recommended_code, ""),
                # Keep the existing manual seal for default front insertion.
                "needsManualReview": True,
            },
            reviewed_pinyin=_pending_reviewed_reading(state, state.recommended_code)[0],
            reviewed_candidate_codes=_pending_reviewed_reading(state, state.recommended_code)[1],
        )

    if len(requested_codes) == 1:
        direct_code = requested_codes[0]
        for code, occupied in state.candidates:
            if code != direct_code:
                continue
            if not occupied:
                if submit_after_add:
                    return await _execute_add_to_draft_and_submit(
                        state.word,
                        direct_code,
                        platform,
                        user_id,
                        space_key,
                        owner_label,
                        state.code_remarks.get(direct_code, ""),
                        state.needs_manual_review,
                        **reviewed_validation(direct_code),
                    )
                return await _execute_add_to_draft(
                    state.word,
                    direct_code,
                    platform,
                    user_id,
                    space_key,
                    owner_label,
                    state.code_remarks.get(direct_code, ""),
                    state.needs_manual_review,
                    **reviewed_validation(direct_code),
                )
            if len(direct_code) < 6:
                create_args = _create_phrase_args(state, direct_code)
                reviewed_pinyin, reviewed_candidate_codes = (
                    _pending_reviewed_reading(state, direct_code)
                )
                return await _execute_shift_to_code(
                    state.word,
                    direct_code,
                    platform,
                    user_id,
                    space_key,
                    owner_label,
                    submit_after=submit_after_add,
                    target_item={
                        "type": state.phrase_type,
                        "remark": str(create_args.get("remark") or ""),
                        "needsManualReview": bool(
                            create_args.get("needs_manual_review", True)
                        ),
                    },
                    reviewed_pinyin=reviewed_pinyin,
                    reviewed_candidate_codes=reviewed_candidate_codes,
                    auto_confirm_shift_plan=True,
                )
            return await _execute_confirmed_tool(
                PendingToolConfirm(
                    function_name="keytao_create_phrase",
                    args={
                        **_create_phrase_args(state, direct_code),
                        **({"_submit_after": True} if submit_after_add else {}),
                    },
                ),
                platform,
                user_id,
                (platform, user_id),
                space_key,
                owner_label,
            )

    requested_target = await _resolve_requested_code_for_pending_add(
        state,
        command_intent.requested_code if command_intent.intent == "pending_code_request" else "",
        platform,
        user_id,
    )
    if requested_target is not None:
        target_code, is_occupied = requested_target
        if not is_occupied:
            if submit_after_add:
                return await _execute_add_to_draft_and_submit(
                    state.word,
                    target_code,
                    platform,
                    user_id,
                    space_key,
                    owner_label,
                    state.code_remarks.get(target_code, ""),
                    state.needs_manual_review,
                    **reviewed_validation(target_code),
                )
            return await _execute_add_to_draft(
                state.word,
                target_code,
                platform,
                user_id,
                space_key,
                owner_label,
                state.code_remarks.get(target_code, ""),
                state.needs_manual_review,
                **reviewed_validation(target_code),
            )
        if len(target_code) < 6:
            create_args = _create_phrase_args(state, target_code)
            reviewed_pinyin, reviewed_candidate_codes = _pending_reviewed_reading(
                state,
                target_code,
            )
            return await _execute_shift_to_code(
                state.word,
                target_code,
                platform,
                user_id,
                space_key,
                owner_label,
                submit_after=submit_after_add,
                target_item={
                    "type": state.phrase_type,
                    "remark": str(create_args.get("remark") or ""),
                    "needsManualReview": bool(
                        create_args.get("needs_manual_review", True)
                    ),
                },
                reviewed_pinyin=reviewed_pinyin,
                reviewed_candidate_codes=reviewed_candidate_codes,
                auto_confirm_shift_plan=True,
            )
        return await _execute_confirmed_tool(
            PendingToolConfirm(
                function_name="keytao_create_phrase",
                args=_create_phrase_args(state, target_code),
            ),
            platform,
            user_id,
            (platform, user_id),
            space_key,
            owner_label,
        )

    target_code: Optional[str] = None
    is_occupied = False

    if command_intent.choice_index is not None:
        idx = command_intent.choice_index - 1
        if 0 <= idx < len(state.candidates):
            target_code, is_occupied = state.candidates[idx]
        else:
            if restore_pending is not None:
                restore_pending()
            else:
                conversation_state_store.set(
                    (platform, user_id),
                    state,
                    space_key=space_key,
                    owner_label=owner_label,
                )
            return render_remediation_reply(
                f"候选编号超出范围；有效范围为 1-{len(state.candidates)}",
                command="加入",
                words=(state.word,),
            )

    elif command_intent.intent == "pending_confirm" or submit_after_add:
        target_code = state.recommended_code
        for c, occ in state.candidates:
            if c == target_code:
                is_occupied = occ
                break

    if target_code is None:
        return None  # unrecognized input, let AI handle as new request

    # Empty slot -> direct execution (no AI needed)
    if not is_occupied:
        if submit_after_add:
            return await _execute_add_to_draft_and_submit(
                state.word,
                target_code,
                platform,
                user_id,
                space_key,
                owner_label,
                state.code_remarks.get(target_code, ""),
                state.needs_manual_review,
                **reviewed_validation(target_code),
            )
        return await _execute_add_to_draft(
            state.word,
            target_code,
            platform,
            user_id,
            space_key,
            owner_label,
            state.code_remarks.get(target_code, ""),
            state.needs_manual_review,
            **reviewed_validation(target_code),
        )

    if is_occupied and len(target_code) < 6:
        create_args = _create_phrase_args(state, target_code)
        reviewed_pinyin, reviewed_candidate_codes = _pending_reviewed_reading(
            state,
            target_code,
        )
        return await _execute_shift_to_code(
            state.word,
            target_code,
            platform,
            user_id,
            space_key,
            owner_label,
            submit_after=submit_after_add,
            target_item={
                "type": state.phrase_type,
                "remark": str(create_args.get("remark") or ""),
                "needsManualReview": bool(
                    create_args.get("needs_manual_review", True)
                ),
            },
            reviewed_pinyin=reviewed_pinyin,
            reviewed_candidate_codes=reviewed_candidate_codes,
            auto_confirm_shift_plan=True,
        )

    if submit_after_add:
        return await _execute_add_to_draft_and_submit(
            state.word,
            target_code,
            platform,
            user_id,
            space_key,
            owner_label,
            state.code_remarks.get(target_code, ""),
            state.needs_manual_review,
            _pending_add_ordering_summary(state, target_code),
            **reviewed_validation(target_code),
        )

    return await _execute_confirmed_tool(
        PendingToolConfirm(
            function_name="keytao_create_phrase",
            args=_create_phrase_args(state, target_code),
        ),
        platform,
        user_id,
        (platform, user_id),
        space_key,
        owner_label,
    )


def _compact_reviewed_reading_options(review: Dict) -> str:
    """Render one bounded code-chain summary per reviewed pronunciation."""
    summaries: List[str] = []
    pronunciations = [
        value
        for value in review.get("pronunciations") or []
        if isinstance(value, dict)
    ]
    for pronunciation in pronunciations[:4]:
        pinyin = str(pronunciation.get("pinyin") or "读音待核对").strip()
        codes = list(dict.fromkeys(
            str(status.get("code") or "").strip().lower()
            for status in pronunciation.get("candidateStatuses") or []
            if isinstance(status, dict) and str(status.get("code") or "").strip()
        ))
        if not codes:
            continue
        chain = codes[0] if len(codes) == 1 else f"{codes[0]}–{codes[-1]}"
        summaries.append(f"{pinyin}：{chain}")
    if not summaries:
        return ""
    if len(pronunciations) > len(summaries):
        summaries.append(f"另 {len(pronunciations) - len(summaries)} 组")
    return "；".join(summaries)


SEMANTIC_ROUTE_OPS = frozenset({
    "add", "swap", "delete", "recode", "move", "reorder", "batch_replace",
    "draft_submit", "draft_recall", "draft_clear", "draft_keep_only",
    "pending_release", "pending_answer", "query", "chat", "help",
})


_semantic_direct_words: ContextVar[Optional[Tuple[str, ...]]] = ContextVar(
    "semantic_direct_words", default=None,
)
_semantic_direct_replace_entry: ContextVar[Optional[Tuple[str, str]]] = ContextVar(
    "semantic_direct_replace_entry", default=None,
)


def semantic_direct_exec_allowed(request, violations=()) -> bool:
    return bool(
        "direct_exec" in {token.strip() for token in os.getenv("KEYTAO_SEMANTIC_ROUTE_OPS", "").split(",")}
        and request.frame == "direct" and request.refers_to_pending is False
        and not violations
    )


_MODEL_ADD_COMMAND_PREFIXES = (
    "帮我加词", "帮我添加", "帮忙加词", "帮忙添加",
    "加个词", "加一下", "帮我加", "请添加",
    "加词", "添加", "请加", "加",
)


def resolve_model_add(request, message):
    """Bind unique literal codes by attachment, then by matching list order."""
    words, codes = request.operands, request.codes
    if (request.frame != "direct" or request.refers_to_pending is not False
        or request.relation or not words or len(set(words)) != len(words)
        or len(set(codes)) != len(codes)):
        return None

    def literal_words(source):
        source = re.sub(r"[A-Za-z]", " ", source)
        source = "".join(" " if unicodedata.category(char).startswith("P") else char
                         for char in source)
        return [(match.start(), match.end(), match.group())
                for match in re.finditer(r"\S+", source)]

    message = _strip_command_message_prefixes(message)
    word_matches = literal_words(message)
    for prefix in _MODEL_ADD_COMMAND_PREFIXES:
        if message.startswith(prefix):
            end = len(prefix)
            if prefix in {"加", "添加", "请加", "请添加"} and (
                end == len(message) or not (
                    message[end].isspace() or unicodedata.category(message[end]).startswith("P")
                )
            ):
                continue
            stripped = message[end:].lstrip()
            stripped_words = literal_words(stripped)
            if word_matches and word_matches[0][2] in words:
                if stripped_words and stripped_words[0][2] in words and stripped_words[0][2] != word_matches[0][2]:
                    return None
            else:
                if not stripped_words or len(stripped_words[0][2]) < 2:
                    return None
                message, word_matches = stripped, stripped_words
            break
    code_matches = [(match.start(), match.end(), match.group().lower())
                    for match in re.finditer(r"[A-Za-z]+", message)]
    if len(code_matches) != len(codes) or {row[2] for row in code_matches} != set(codes):
        return None
    if not codes:
        word_matches = [(match.start(), match.end(), word) for word in words
                        for match in re.finditer(re.escape(word), message)]
        # A shorter operand inside another named word is not a second occurrence.
        word_matches = [row for row in word_matches if not any(
            other[0] <= row[0] and row[1] <= other[1] and other[2] != row[2]
            for other in word_matches
        )]
    if len(word_matches) != len(words) or {row[2] for row in word_matches} != set(words):
        return None
    word_matches.sort()
    if any(start < word_end and word_start < end
           for start, end, _code in code_matches for word_start, word_end, _word in word_matches):
        return None
    code_matches.sort()
    paired = {}
    for start, _end, code in code_matches:
        attached = [word for _word_start, end, word in word_matches
                    if end <= start and re.fullmatch(r"[\s\W_]*", message[end:start])]
        if len(attached) == 1:
            if attached[0] in paired:
                return None
            paired[attached[0]] = code
    if len(paired) != len(codes):
        same_order = (tuple(row[2] for row in word_matches) == words
                      and tuple(row[2] for row in code_matches) == codes)
        separate_lists = bool(code_matches) and (
            word_matches[-1][1] <= code_matches[0][0]
            or code_matches[-1][1] <= word_matches[0][0]
        )
        ordered = dict(zip(words, codes))
        glued_conflict = any(
            end == start and ordered.get(word) != code
            for _word_start, end, word in word_matches for start, _code_end, code in code_matches
        )
        if (len(words) != len(codes) or not same_order
            or glued_conflict
            or not separate_lists and any(ordered[word] != code for word, code in paired.items())):
            return None
        paired = ordered
    ordered_words = tuple(row[2] for row in word_matches)
    return replace(request, operands=ordered_words,
                   codes=tuple(paired.get(word, "") for word in ordered_words),
                   submit_after=request.submit_after and len(words) == len(codes) == 1)


def _semantic_warnings_are_named(data: Dict, args: Dict) -> bool:
    """Unknown warning ownership requires confirmation; prose never grants it."""
    from ..harness.authorization_grammar import _AUTO_CONFIRM_CREATE_WARNING_TYPES

    named = _semantic_direct_words.get()
    warnings = data.get("warnings")
    if named is None or not isinstance(warnings, list) or server_warning_confirmation_binding(data) is None:
        return False
    if any(data.get(key) for key in ("failedCount", "skippedCount", "failed", "skipped", "partialWrite")):
        return False

    requested = [(args.get("word"), args.get("code") or args.get("target_code"))]
    for key in ("items", "additional_items", "additional_shift_items"):
        requested.extend((item.get("word"), item.get("code")) for item in args.get(key) or [] if isinstance(item, dict))
    requested.extend(zip(args.get("ordered_words") or [], args.get("expected_codes") or []))
    allowed = {
        (word, code) for word, code in requested
        if isinstance(word, str) and isinstance(code, str) and word and code
        and (word in named or (word, code) == _semantic_direct_replace_entry.get())
    }
    allowed_codes = {code for _, code in allowed}
    containers = {"item", "items", "existing", "occupant", "occupants", "displaced", "shifted",
                  "conflictWith", "phrase", "phrases", "entry", "entries"}
    metadata = {"warningType", "action", "type", "weight", "message", "remark", "needsManualReview"}

    def identities(value, inherited_codes=()):
        if isinstance(value, str):
            return {(value, code) for code in inherited_codes} if value and inherited_codes else None
        if isinstance(value, list):
            found = set()
            for item in value:
                affected = identities(item, inherited_codes)
                if not affected:
                    return None
                found.update(affected)
            return found or None
        if not isinstance(value, dict) or not value:
            return None
        codes = []
        for key, item in value.items():
            if key.lower().endswith(("code", "codes")):
                if not isinstance(item, list if key.lower().endswith("codes") else str):
                    return None
                values = item if isinstance(item, list) else [item]
                if not values or any(not isinstance(code, str) or code not in allowed_codes for code in values):
                    return None
                codes.extend(values)
        codes = tuple(codes) or inherited_codes
        found = set()
        for key, item in value.items():
            if key.lower().endswith(("word", "words")) or key in containers:
                if key.lower().endswith(("word", "words")) and not isinstance(
                    item, list if key.lower().endswith("words") else str,
                ):
                    return None
                affected = identities(item, codes)
                if not affected:
                    return None
                found.update(affected)
            elif key not in metadata and not key.lower().endswith(("code", "codes")):
                return None
            elif isinstance(item, (dict, list)) and key in metadata:
                return None
        return found

    for warning in warnings:
        if (
            not isinstance(warning, dict)
            or not isinstance(warning.get("warningType"), str)
            or warning["warningType"] not in _AUTO_CONFIRM_CREATE_WARNING_TYPES
        ):
            return False
        affected = identities(warning)
        if not affected or not affected <= allowed:
            return False
    return True


async def _resolve_target_recodes(request, message, original_operands, platform, user_id):
    """Bind targets to literal clauses and resolve connector ambiguity against live rows."""
    from collections import Counter
    from ..harness.semantic_parser import _SAME_WORD_RECODE_RE

    source = _strip_command_message_prefixes(message).strip()
    same_word = _SAME_WORD_RECODE_RE.fullmatch(source)
    candidates = list(request.operands)
    for word in request.operands:
        connector = re.fullmatch(r"(.+?)(?:改为|改成|为|是)", word)
        if connector and re.search(re.escape(word) + r"\s*[a-z]{1,12}(?![a-zA-Z])", source):
            candidates.append(connector[1])
    for operand in original_operands if isinstance(original_operands, (list, tuple)) else ():
        if not isinstance(operand, str):
            continue
        base = re.sub(r"\s*[a-zA-Z]{1,12}$", "", operand).strip()
        if any(base == word + connector for word in request.operands for connector in ("改为", "改成", "为", "是")):
            candidates.append(base)
    candidates = list(dict.fromkeys(candidates))
    clauses = []
    if same_word and same_word["old_word"].strip('「」“”"') == same_word["new_word"].strip('「」“”"'):
        clauses = [[(same_word["new_word"].strip('「」“”"'), same_word["new_code"])]]
    else:
        source, prefixed = re.subn(r"^(?:修改编码|改码)\s*[:：]?\s*", "", source, count=1)
        for clause in re.split(r"[,，;；\n]+", source.rstrip("。.!！")):
            matches = []
            for word in candidates:
                literal = re.escape(word)
                match = re.fullmatch(
                    r"\s*(?:把|将)?(?:" + literal + "|「" + literal + "」|“" + literal + '”|"' + literal + r'")'
                    + r"\s*(?:(?:的)?编码\s*)?(?P<connector>改为|改成|为|是)?\s*(?P<code>[a-z]{1,12})\s*",
                    clause,
                )
                if match and (prefixed or match["connector"]):
                    matches.append((word, match["code"]))
            clauses.append(matches)
    if (len(clauses) != len(request.operands) or any(not matches for matches in clauses)
        or Counter(matches[0][1] for matches in clauses) != Counter(request.codes)):
        return None, (), "改码目标无法与本条消息逐项对应；请逐项写明词条和目标编码，本次未写入。"
    entries = {}
    words, codes = [], []
    for matches in clauses:
        resolved = []
        for word, new_code in matches:
            if word not in entries:
                try:
                    data = json.loads(await call_tool_function(
                        "keytao_lookup_by_word", {"word": word}, platform, user_id,
                    ))
                    entries[word] = _validated_dictionary_delete_entries(data, word)
                except (ValueError, TypeError, OSError, TimeoutError):
                    entries[word] = None
            rows = entries[word]
            if rows is None:
                return None, (), f"无法取得「{word}」的完整词库记录；本次未写入。"
            resolved.extend((row, new_code) for row in rows)
        if len(resolved) != 1:
            return None, (), "词条或旧码无法唯一确定。请指定旧码；本次未写入。"
        row, new_code = resolved[0]
        if row["word"] in words or row["code"] == new_code:
            return None, (), "改码词条重复或目标编码与旧码相同；本次未写入。"
        words.append(row["word"])
        codes.extend((row["code"], new_code))
    return replace(request, operands=tuple(words), codes=tuple(codes)), (), ""


async def resolve_model_replace(request, platform, user_id, *, message="", original_operands=()):
    """Resolve model old/new fields against exact dictionary rows, never prose."""
    if request.op == "delete" and len(request.operands) == len(request.codes) == 1 and render_understood_command(request):
        word, code = request.operands[0], request.codes[0]
        try:
            data = json.loads(await call_tool_function(
                "keytao_lookup_by_word", {"word": word}, platform, user_id,
            ))
            rows = _validated_dictionary_delete_entries(data, word)
            if rows is None:
                raise ValueError("invalid word lookup")
            if any(row["code"] == code for row in rows):
                return request, (), ""
            draft = await _fetch_current_draft_items(platform, user_id)
            items = draft.get("items") if isinstance(draft, dict) and draft.get("success") is True else None
            if not isinstance(items, list):
                raise ValueError("incomplete draft lookup")
            draft_rows = [item for item in items if isinstance(item, dict)
                          and item.get("action") == "Create" and _draft_item_word(item) == word
                          and _canonical_draft_delete_target(item) is not None]
            if any(row["code"] == code for row in draft_rows):
                return request, (), ""
        except (ValueError, TypeError, OSError, TimeoutError):
            return None, (), "无法核验完整词库和草稿记录；本次未生成删除计划。"
        current_codes = "、".join(dict.fromkeys(row["code"] for row in rows + draft_rows)) or "未找到"
        return None, (), f"「{word}」当前的编码是 {current_codes}，没有 {code}。请确认要删除的编码；本次未写入。"
    if request.op == "recode" and not request.relation and request.operands:
        target_request = request
        if (len(request.codes) == 2 * len(request.operands)
            and (len(request.operands) > 1
                 or re.match(r"^(?:修改编码|改码)\s*[:：]?", _strip_command_message_prefixes(message)))):
            target_request = replace(request, codes=request.codes[1::2])
        if (len(target_request.operands) == len(target_request.codes)
            and len(set(target_request.operands)) == len(target_request.operands)):
            return await _resolve_target_recodes(target_request, message, original_operands, platform, user_id)
    words, codes = request.operands, tuple(dict.fromkeys(request.codes))
    candidate = (
        request.op in {"move", "recode"}
        or request.op == "swap" and request.relation == "replace"
    ) and (len(words) == 2 and len(codes) <= 1 or len(words) == len(codes) == 1)
    if not candidate:
        return request, (), ""
    try:
        lookup = json.loads(await call_tool_function(
            "keytao_lookup_by_words_batch", {"words": list(words)}, platform, user_id,
        ))
        results = lookup.get("results")
        if lookup.get("success") is not True or not isinstance(results, list) or len(results) != len(words):
            raise ValueError("incomplete lookup")
        entries = {}
        for result in results:
            word = result.get("word") if isinstance(result, dict) else None
            if word not in words or word in entries:
                raise ValueError("unexpected word")
            rows = _validated_dictionary_delete_entries({"success": True, "phrases": result.get("phrases")}, word)
            if rows is None:
                raise ValueError("invalid rows")
            entries[word] = rows
        occupants = None
        if codes:
            data = json.loads(await call_tool_function(
                "keytao_lookup_by_code", {"code": codes[0]}, platform, user_id,
            ))
            occupants = data.get("phrases")
            if data.get("success") is not True or not isinstance(occupants, list) or any(
                not isinstance(row, dict) or not row.get("word") or row.get("code") != codes[0]
                or row.get("type") not in _PHRASE_TYPE_BASE_WEIGHTS for row in occupants
            ):
                raise ValueError("invalid occupants")
    except (ValueError, TypeError, KeyError):
        return None, (), "无法取得完整词库记录；本次未写入。"

    old, new = words if len(words) == 2 else (None, words[0])
    if old is None and entries[new]:
        if len(entries[new]) != 1 or len(occupants) > 1:
            return None, (), f"「{new}」的来源或 {codes[0]} 的占位记录不唯一；请指定具体记录，本次未写入。"
        return replace(request, op="move", codes=codes, relation=""), (), ""
    if old is not None:
        matches = [row for row in entries[old] if not codes or row["code"] == codes[0]]
        if not matches:
            return None, (), f"词库没有找到「{old}」的对应记录；本次未写入。"
    else:
        matches = occupants
        if not matches:
            return None, (), f"编码 {codes[0]} 当前没有词条可顶替；请使用普通加词指令「加词 {new} {codes[0]}」，本次未写入。"
    if len(matches) != 1:
        options = []
        if old is not None and not codes and not entries[new]:
            for row in matches:
                option = replace(request, op="move", operands=(old, new), codes=(row["code"],), relation="replace")
                if render_understood_command(option) and option not in options:
                    options.append(option)
        return None, tuple(options) if len(options) > 1 else (), "词库记录不唯一，请指定编码和词库类型；本次未写入。"
    target = matches[0]
    code = codes[0] if codes else target["code"]
    if occupants is None:
        try:
            data = json.loads(await call_tool_function("keytao_lookup_by_code", {"code": code}, platform, user_id))
        except (ValueError, TypeError):
            data = {}
        occupants = data.get("phrases") if data.get("success") is True else None
    if not isinstance(occupants, list) or len(occupants) != 1 or not isinstance(occupants[0], dict) or (
        occupants[0].get("word"), occupants[0].get("code"), occupants[0].get("type")
    ) != (target["word"], code, target["type"]):
        return None, (), f"编码 {code} 的占位记录不唯一或不一致；本次未写入。"
    if entries[new]:
        if len(entries[new]) != 1:
            return None, (), f"「{new}」有多条现有记录，无法确定移位来源；本次未写入。"
        return replace(request, op="move", operands=(new,), codes=(code,), relation=""), (), ""
    resolved = replace(request, op="move", operands=(target["word"], new), codes=(code,), relation="replace")
    if not render_understood_command(resolved):
        return None, (), "替换方案无法通过现有指令校验；本次未写入。"
    return resolved, (), ""


def render_understood_pending_answer(request, state, prompt: str, message: str = "") -> str:
    """Round-trip a model answer through the existing ticket's closed parsers."""
    from ..harness.semantic_parser import _pending_values
    from ..utils.pending_confirmation import advertised_reply_contract
    from .chat_routing import _multi_word_candidate_scope_rows

    words, codes = _pending_values(state)
    if (
        request.frame != "direct" or request.refers_to_pending is not True
        or any(word not in words for word in request.operands)
        or any(code not in codes for code in request.codes)
    ):
        return ""
    if request.op == "select_candidate":
        selected = request.codes
        message_codes = set(re.findall(r"[a-z]+", message.lower()))
        if isinstance(state, PendingAddWord):
            if not state.server_candidates or state.server_candidates != state.candidates:
                return ""
            literal = _structural_pending_add_word_intent(message, state)
            indices = (literal.choice_indices or ((literal.choice_index,) if literal.choice_index is not None else ())) if literal else ()
            literal_codes = tuple(state.candidates[index - 1][0] for index in indices if 1 <= index <= len(state.candidates))
            if not selected:
                if literal is None or not indices or literal.recode_indices or len(literal_codes) != len(indices):
                    return ""
                selected = literal_codes
            if not selected or len(set(selected)) != len(selected) or any(code not in codes for code in selected):
                return ""
            if any(code not in message_codes and code not in literal_codes for code in selected):
                if selected != (state.recommended_code,):
                    return ""
                return render_understood_pending_answer(replace(request, op="confirm", codes=()), state, prompt, message)
            text = "、".join(selected) + ("，加入并提交" if request.submit_after else "")
            intent = _structural_pending_add_word_intent(text, state)
            if intent is None:
                return ""
            indices = intent.choice_indices or ((intent.choice_index,) if intent.choice_index is not None else ())
            actual = intent.requested_codes or ((intent.requested_code,) if intent.requested_code else ())
            if indices:
                actual = tuple(state.candidates[index - 1][0] for index in indices if 1 <= index <= len(state.candidates))
            if (
                tuple(actual) != tuple(selected) or intent.recode_indices or intent.target_word
                or intent.intent not in ({"pending_add_and_submit"} if request.submit_after else {"pending_choice", "pending_code_request"})
                or bool(intent.submit_after or intent.intent == "pending_add_and_submit") != request.submit_after
            ):
                return ""
            return text
        items, scopes = _multi_word_candidate_scope_rows(state)
        selected_words = request.operands
        if not selected_words and len(scopes) == 1:
            selected_words = tuple(scopes)
        literal, _intent, error = _resolve_multi_word_pending_candidate_selection(state, message)
        literal_codes = {item["word"]: item["code"] for item in literal.args["items"]} if literal and not error else {}
        if not selected and all(word in literal_codes for word in selected_words):
            selected = tuple(literal_codes[word] for word in selected_words)
        if not selected or len(selected_words) != len(selected) or len(set(selected_words)) != len(selected_words):
            return ""
        if any(code not in dict(scopes.get(word, ())) for word, code in zip(selected_words, selected)):
            return ""
        if any(code not in message_codes and literal_codes.get(word) != code for word, code in zip(selected_words, selected)):
            if dict(zip(selected_words, selected)) != {item["word"]: item["code"] for item in items}:
                return ""
            return render_understood_pending_answer(replace(request, op="confirm", codes=()), state, prompt, message)
        text = "，".join(f"{word} {code}" for word, code in zip(selected_words, selected))
        text += "，加入并提交" if request.submit_after else "，加入"
        derived, intent, error = _resolve_multi_word_pending_candidate_selection(state, text)
        actual = tuple((item.get("word"), item.get("code")) for item in (derived.args.get("items", []) if derived else []))
        if (
            error or intent is None or actual != tuple(zip(selected_words, selected))
            or intent.intent != ("pending_add_and_submit" if request.submit_after else "pending_confirm")
        ):
            return ""
        return text
    if request.codes or request.submit_after and request.op != "confirm":
        return ""
    if request.op == "confirm":
        contract = advertised_reply_contract(prompt)
        forms = contract.generic_assent_forms + contract.batch_assent_forms
        choices = ("加入并提交",) if request.submit_after else ("确认", "加入", "添加", "都加")
        for text in choices:
            if text not in forms:
                continue
            intent = _pending_tool_assent_intent(state, text)
            if intent is not None and intent.intent == ("pending_add_and_submit" if request.submit_after else "pending_confirm"):
                return text
        return ""
    if request.op == "cancel":
        intent = _pending_tool_assent_intent(state, "取消")
        return "取消" if intent is not None and intent.intent == "pending_cancel" else ""
    if request.op == "pending_recode":
        intent = _structural_pending_add_word_intent("换码", state) if isinstance(state, PendingAddWord) else None
        return "换码" if intent is not None and intent.intent == "pending_recode" else ""
    return ""


def render_understood_command(request) -> str:
    """Render model fields only when the existing grammar preserves them exactly."""
    from ..harness.operation_request import understand_message

    op, words, codes = request.op, request.operands, request.codes
    if not isinstance(op, str) or op not in SEMANTIC_ROUTE_OPS or op == "swap":
        return ""
    if any(not isinstance(word, str) or not word or re.search(r"[\s\x00-\x1f\x7f]", word) for word in words):
        return ""
    if any(not isinstance(code, str) or (not code and op != "add")
           or code and re.fullmatch(r"[a-z]{1,12}", code) is None for code in codes):
        return ""
    if len(set(words)) != len(words):
        return ""
    if op == "recode" and len(words) > 1 and len(codes) == 2 * len(words) and not request.submit_after and not request.relation:
        commands = [render_understood_command(replace(request, operands=(word,), codes=codes[index * 2:index * 2 + 2]))
                    for index, word in enumerate(words)]
        return "\n".join(commands) if all(commands) else ""
    command = ""
    if op == "add":
        from ..harness.authorization_grammar import _multi_add_authorization_contract, _parse_complete_add_clause

        if not words or len(words) != len(codes) or len(set(filter(None, codes))) != len(tuple(filter(None, codes))):
            return ""
        command = "；".join(f"添加 {word}" + (f" {code}" if code else "") for word, code in zip(words, codes))
        if request.submit_after:
            if len(words) != 1 or not codes[0]:
                return ""
            command += " 并提交"
        if len(words) == 1 and codes[0]:
            parsed = explicit_complete_add_item(command)
            understood = understand_message(command)
            return command if parsed is not None and (
                parsed["word"], parsed["code"], parsed["submitAfter"]
            ) == (words[0], codes[0], request.submit_after) and (
                understood.op, understood.operands, understood.codes, understood.submit_after
            ) == (op, words, codes, request.submit_after) else ""
        parsed = _multi_add_authorization_contract(command) if len(words) > 1 else None
        clauses = parsed.clauses if parsed is not None and parsed.valid else (_parse_complete_add_clause(command),)
        return command if all(clause is not None for clause in clauses) and tuple(
            (clause.word, clause.code) for clause in clauses
        ) == tuple(zip(words, codes)) else ""
    if op == "delete" and len(words) == 1 and len(codes) <= 1:
        command = f"删词 {words[0]} {' '.join(codes)}".strip()
    elif op == "recode" and len(words) == 1 and len(codes) == 2 and codes[0] != codes[1]:
        command = f"删除 {words[0]} {codes[0]}；添加 {words[0]} {codes[1]}"
    elif op == "move":
        if len(words) == len(codes) == 1:
            command = f"把{words[0]}调到{codes[0]}"
        elif len(words) == 2 and len(codes) == 1:
            command = f"修改 {codes[0]}：{words[0]} → {words[1]}"
        elif len(words) == len(codes) == 2:
            command = f"把{words[0]}调到{codes[0]}；把{words[1]}调到{codes[1]}"
    elif op == "reorder":
        if len(words) == 2 and len(codes) in {0, 2}:
            entries = [f"{word} {codes[index]}" if codes else word for index, word in enumerate(words)]
            command = f"{entries[0]} 优先于 {entries[1]}"
        elif len(words) > 2 and not codes:
            command = "重新排序 " + " ".join(words)
        elif not words and len(codes) == 1:
            command = "重排 " + codes[0]
    elif op in {"draft_submit", "draft_recall", "draft_clear"} and not words and not codes:
        command = {"draft_submit": "提交草稿", "draft_recall": "撤回提交", "draft_clear": "清空草稿"}[op]
    elif op == "draft_keep_only" and words and not codes:
        command = "只保留 " + " ".join(words) + "，其余删除"
        if request.submit_after:
            command += "，并提交"
        parsed = _canonical_keep_only_command(command, MessageCommandIntent(
            intent="draft_keep_only", confidence=1.0, keep_words=words,
        ))
        return command if parsed is not None and (parsed.keep_words, parsed.submit_after) == (words, request.submit_after) else ""
    elif op == "batch_replace" and len(words) >= 3 and len(codes) == len(words) - 2:
        old, new, *rows = words
        if len(old) != 1 or len(new) != 1 or old == new or request.submit_after:
            return ""
        command = f"把{old}替换成{new}\n" + "\n".join(f"{word} {code}" for word, code in zip(rows, codes))
        intent = MessageCommandIntent(intent="batch_replace_char", confidence=1.0, old_char=old, new_char=new)
        parsed = _build_replace_char_items(command, old, new)
        return command if (
            _message_authorizes_replace_char(command, intent)
            and [(item["old_word"], item["code"]) for item in parsed] == list(zip(rows, codes))
        ) else ""
    if not command or request.submit_after:
        return ""
    parsed = understand_message(command)
    return command if (parsed.op, parsed.operands, parsed.codes, parsed.submit_after) == (op, words, codes, request.submit_after) else ""


def _understood_command_digest(args: Dict) -> str:
    return hashlib.sha256(json.dumps(
        {key: value for key, value in args.items() if key != "digest"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


async def try_handle_model_command(request, platform, user_id, conv_key, space_key, owner_label):
    """Seal a canonical command without performing any tool call."""
    command = render_understood_command(request)
    if not command:
        return None
    record = conversation_state_store.get_record(conv_key)
    if record is not None and record.execution_id:
        return "上一项操作仍在处理中，本次未替换确认计划。"
    args = {"command_text": command, "op": request.op, "operands": list(request.operands),
            "codes": list(request.codes), "submit_after": request.submit_after}
    args["digest"] = _understood_command_digest(args)
    if not conversation_state_store.set(
        conv_key, PendingToolConfirm("keytao_understood_command", args),
        space_key=space_key, owner_label=owner_label,
    ):
        return "确认计划无法保存，本次未写入。"
    response = _append_pending_ticket_challenge(
        render_pending_understood_command(PendingToolConfirm("keytao_understood_command", args)),
        conv_key,
    )
    return _announce_replaced_submit(record, conv_key, response)


async def try_handle_model_command_options(options, explanation, conv_key, space_key, owner_label):
    record = conversation_state_store.get_record(conv_key)
    args = {"options": []}
    for request in options:
        command = render_understood_command(request)
        if not command:
            return explanation
        option = {"command_text": command, "op": request.op, "operands": list(request.operands),
                  "codes": list(request.codes), "submit_after": request.submit_after}
        option["digest"] = _understood_command_digest(option)
        args["options"].append(option)
    args["digest"] = _understood_command_digest(args)
    state = PendingToolConfirm("keytao_understood_command", args)
    if not conversation_state_store.set(conv_key, state, space_key=space_key, owner_label=owner_label):
        return "候选方案无法保存，本次未写入。"
    response = render_pending_understood_command(state)
    return _announce_replaced_submit(record, conv_key, response)


def _understood_command_options(state):
    options = state.args.get("options")
    if (
        not isinstance(options, list) or len(options) < 2
        or state.args.get("digest") != _understood_command_digest(state.args)
        or any(not isinstance(option, dict) or _understood_command_request(
            PendingToolConfirm("keytao_understood_command", option),
        ) is None for option in options)
    ):
        return ()
    return options


def _understood_command_request(state):
    from ..harness.operation_request import OperationRequest

    if state.function_name != "keytao_understood_command" or state.confirmation_source != "local_preview":
        return None
    args = state.args
    if not isinstance(args.get("operands"), list) or not isinstance(args.get("codes"), list) or not isinstance(args.get("submit_after"), bool):
        return None
    request = OperationRequest(
        op=args.get("op"), operands=tuple(args["operands"]),
        codes=tuple(args["codes"]), submit_after=args["submit_after"],
    )
    command = render_understood_command(request)
    if not command or command != args.get("command_text") or args.get("digest") != _understood_command_digest(args):
        return None
    return request


def render_pending_understood_command(state) -> str:
    """Expose controls only for the exact canonical command sealed in this ticket."""
    options = _understood_command_options(state)
    if options:
        return "\n".join([
            "词库记录不唯一，请选择具体方案；本次未写入。",
            *(f"{index}. {render_executable_suggestion(option['command_text']).removeprefix('- ')}"
              for index, option in enumerate(options, 1)),
            "回复方案编号执行，或「取消」。",
        ])
    if _understood_command_request(state) is None:
        return ""
    return f"我理解为：「{state.args['command_text']}」\n回复「确认」执行，或「取消」。"


async def handle_pending_understood_command(record, message, platform, user_id, conv_key, space_key, owner_label):
    """Consume this exact author's ticket before re-entering explicit dispatch."""
    if (
        record.owner_key != normalize_conversation_key(conv_key, space_key)
        or record.owner_key.actor_id != str(user_id)
        or record.owner_key.platform != platform
        or record.state.function_name != "keytao_understood_command"
    ):
        return None
    control = _strip_command_message_prefixes(message).strip()
    options = _understood_command_options(record.state)
    selected = options[int(control) - 1] if options and control.isascii() and control.isdigit() and 1 <= int(control) <= len(options) else None
    if control not in {"确认", "取消"} and selected is None:
        return None
    if options and control == "确认":
        return render_pending_understood_command(record.state)
    if not conversation_state_store.begin_execution(record):
        return "当前确认正在处理中，本次未重复执行。"
    if not conversation_state_store.complete_execution(record):
        return "确认计划无法安全消费，本次未写入。"
    if control == "取消":
        return "已取消。"
    state = PendingToolConfirm("keytao_understood_command", selected) if selected is not None else record.state
    request = _understood_command_request(state)
    if request is None:
        return "确认计划校验失败，本次未写入。"
    return await execute_understood_command(request, state.args["command_text"], platform, user_id, conv_key, space_key, owner_label)


async def execute_understood_command(request, command, platform, user_id, conv_key, space_key, owner_label):
    """Re-enter the existing typed handlers with one validated canonical plan."""
    active = draft_operation_coordinator.find_for_actor((platform, user_id))
    if active is not None:
        return _active_operation_message_for_request(active, platform, user_id)
    op, words, codes = request.op, request.operands, request.codes
    if op == "add":
        if render_understood_command(request) != command:
            return "规范加词指令校验失败，本次未写入。"
        pairs = [(word, code) for word, code in zip(words, codes) if code]
        missing = tuple(word for word, code in zip(words, codes) if not code)
        replies = []
        if len(pairs) == 1:
            word, code = pairs[0]
            replies.append(await _execute_explicit_entry_code_request(
                ExplicitEntryCodeRequest(word, code, submit_after=request.submit_after),
                f"添加 {word} {code}" + (" 并提交" if request.submit_after else ""),
                platform, user_id, conv_key, space_key, owner_label,
            ))
        elif pairs:
            items, readings = [], []
            for word, code in pairs:
                try:
                    review = json.loads(await call_tool_function(
                        "keytao_prepare_reviewed_add", {"word": word}, platform, user_id,
                    ))
                except (TypeError, ValueError):
                    review = {}
                phrase_type = "Single" if len(word) == 1 else "Phrase"
                if (not isinstance(review, dict) or review.get("success") is not True
                    or review.get("word") != word or review.get("type", "Phrase") != phrase_type
                    or review.get("reviewDisposition") == "BLOCK" or review.get("pronunciationUnresolved") is True):
                    return f"未能取得「{word}」的已审读音和正确词条类型，本次未写入。"
                inventory = select_candidate_inventory(review)
                if inventory is None or not all(inventory.readings.values()):
                    return f"编码 {code} 未能与「{word}」的已审读音绑定，本次未写入。"
                reviewed = PendingAddWord(
                    word, str(review.get("recommendedCode") or ""), list(inventory.candidates),
                    server_candidates=list(inventory.candidates),
                    pronunciation_codes=dict(inventory.readings), phrase_type=phrase_type,
                )
                validation = validate_explicit_code(reviewed, code)
                if not validation.valid or not validation.pinyin:
                    return (validation.reason or f"编码 {code} 缺少已审读音") + "；本次未写入。"
                pinyin = validation.pinyin
                reason = validation.manual_reason or str(review.get("manualReviewReason") or "")
                items.append({
                    "action": "Create", "word": word, "code": code, "type": phrase_type,
                    "needsManualReview": bool(validation.manual_reason) or review.get("needsManualReview") is not False,
                    "remark": f"喵喵审词：读音 {pinyin}；{reason}",
                })
                readings.append({"word": word, "pinyin": pinyin, "candidate_codes": list(dict.fromkeys([
                    value for value, reading in inventory.readings.items() if reading == pinyin
                ] + [code]))})
            replies.append(await _execute_confirmed_tool(PendingToolConfirm(
                "keytao_batch_add_to_draft", {"items": items, "_reviewed_batch_readings": readings},
            ), platform, user_id, conv_key, space_key, owner_label))
        if missing:
            preserve = bool(pairs and conversation_state_store.get_record(conv_key) is not None)
            if len(missing) == 1 and not _mergeable_add_candidate_scopes(platform, user_id, conv_key, space_key):
                replies.append(await _try_handle_simple_single_word_query(
                    missing[0], platform, user_id, conv_key, space_key, owner_label,
                    _literal_word=True, preserve_pending=preserve,
                ))
            else:
                replies.append(await _prepare_multi_word_query(
                    missing, platform, user_id, conv_key, space_key, owner_label,
                    preserve_pending=preserve,
                    explicit_add=True,
                ))
        return "\n\n".join(reply for reply in replies if reply)
    if op == "move":
        response = await try_handle_move_to_code_command(command, platform, user_id, space_key, owner_label)
        if response is not None:
            return response
    if op == "reorder" and len(words) == 2 and (not codes or len(codes) == 2 and codes[0] != codes[1]):
        merged = await _load_merged_reorder_entries(words, platform, user_id)
        if merged is None:
            return "无法读取完整的词库和草稿记录，本次未生成调序计划。"
        entries = [[entry for entry in merged if entry["word"] == word] for word in words]
        code_labels = " / ".join(
            f"「{word}」{'、'.join(dict.fromkeys(entry['code'] for entry in rows)) or '无现有编码'}"
            for word, rows in zip(words, entries)
        )
        if any(len(rows) != 1 for rows in entries):
            ambiguous = "、".join(f"「{word}」" for word, rows in zip(words, entries) if len(rows) > 1)
            reason = f"{ambiguous}有多个词条，无法唯一确定" if ambiguous else "未能锁定双方的现有词条"
            return f"{code_labels}；{reason}，本次未修改。"
        first_code, second_code = (rows[0]["code"] for rows in entries)
        if codes and codes != (first_code, second_code):
            return f"{code_labels}；现有编码与指令不一致，本次未修改。"
        if first_code != second_code:
            if not (first_code.startswith(second_code) or second_code.startswith(first_code)):
                return f"{code_labels}；两个词不在同一编码链，本次未修改。"
            if len(first_code) < len(second_code):
                return (
                    f"「{words[0]}」已经排在「{words[1]}」前面"
                    f"（{words[0]} {first_code} / {words[1]} {second_code}），本次未修改。"
                )
            return await _execute_entry_swap_plan(
                words, {word: rows[0] for word, rows in zip(words, entries)},
                platform, user_id, space_key, owner_label,
            )
    intent = MessageCommandIntent(
        intent={"delete": "dictionary_delete", "recode": "dictionary_recode",
                "move": "replace_at_code", "reorder": "word_list_reorder",
                "batch_replace": "batch_replace_char"}.get(op, op),
        confidence=1.0, keep_words=words,
        requested_code=codes[0] if codes else "",
        requested_codes=codes[1:] if op == "recode" else codes,
        submit_after=request.submit_after,
    )
    if op == "batch_replace":
        intent = replace(intent, old_char=words[0], new_char=words[1])
        previous = conversation_state_store.get_record(conv_key)
        response = await _try_handle_replace_char(command, platform, user_id, intent, conv_key, space_key, owner_label)
        if _semantic_direct_words.get() is not None:
            record = conversation_state_store.get_record(conv_key)
            expected = _build_replace_char_items(command, words[0], words[1])
            chunk, _remaining, _confirmation = _replace_char_chunk(expected, words[0], words[1])
            if (record is not None and record is not previous
                and isinstance(record.state, PendingToolConfirm)
                and record.state.function_name == "keytao_batch_add_to_draft"
                and record.state.args.get("items") == chunk):
                if not conversation_state_store.begin_execution(record) or not conversation_state_store.complete_execution(record):
                    return "替换计划无法安全消费，本次未写入。"
                response = await _execute_confirmed_tool(record.state, platform, user_id, conv_key, space_key, owner_label)
    else:
        if op == "reorder" and not words:
            intent = replace(intent, intent="code_chain_reorder")
        response = await _try_handle_draft_management_command(command, platform, user_id, space_key, owner_label, intent)
    return response or "规范指令无法执行，本次未写入。"


async def execute_model_command_direct(request, named_words, platform, user_id, conv_key, space_key, owner_label):
    active = draft_operation_coordinator.find_for_actor((platform, user_id))
    if active is not None:
        return _active_operation_message_for_request(active, platform, user_id)
    record = conversation_state_store.get_record(conv_key)
    if record is not None and record.execution_id:
        return "上一项操作仍在处理中，本次未执行。"
    if record is not None and (
        record.owner_key != normalize_conversation_key(conv_key, space_key)
        or record.owner_key.platform != platform or record.owner_key.actor_id != str(user_id)
    ):
        return "原候选无法安全释放，本次未执行。"
    if (
        record is not None
        and not (isinstance(record.state, PendingToolConfirm) and record.state.function_name == "keytao_submit_batch")
        and (not conversation_state_store.begin_execution(record) or not conversation_state_store.complete_execution(record))
    ):
        return "原候选无法安全释放，本次未执行。"
    if request.op == "add":
        # Keep ordinary add warning confirmations, including manual review.
        command = render_understood_command(request)
        if not command:
            return "还没有核对出可执行的词条和编码；本次未写入。"
        response = await execute_understood_command(request, command, platform, user_id, conv_key, space_key, owner_label)
        return _announce_replaced_submit(record, conv_key, response)
    token = _semantic_direct_words.set(tuple(named_words))
    replacement = (
        (request.operands[0], request.codes[0])
        if request.relation == "replace" and len(request.operands) == 2 and len(request.codes) == 1
        and tuple(named_words) == (request.operands[1],) else None
    )
    replacement_token = _semantic_direct_replace_entry.set(replacement)
    try:
        if request.op == "swap":
            if len(request.operands) != 2 or len(set(request.operands)) != 2 or request.codes or request.submit_after:
                return "还没有核对出唯一换位方案；本次未写入。"
            response = await try_handle_model_entry_swap(*request.operands, platform, user_id, space_key, owner_label)
        else:
            command = render_understood_command(request)
            if not command:
                return "还没有核对出可执行的词条和编码；本次未写入。"
            response = await execute_understood_command(request, command, platform, user_id, conv_key, space_key, owner_label)
        reviewed = conversation_state_store.get_record(conv_key)
        if reviewed is not record and isinstance(getattr(reviewed, "state", None), PendingAddWord):
            conversation_state_store.complete_execution(reviewed)
        return _announce_replaced_submit(record, conv_key, response)
    finally:
        _semantic_direct_replace_entry.reset(replacement_token)
        _semantic_direct_words.reset(token)


_PENDING_CHOICE_OFFER_FUNCTION = "keytao_choice_offer"
_PENDING_CHOICE_LABEL_RE = re.compile(
    r"^(?:(?:选择?|方案)\s*)?([AB])$",
    re.IGNORECASE,
)


def render_executable_plan_suggestion(state: PendingToolConfirm) -> str:
    """Render only a command that round-trips through the real plan grammar."""
    if state.function_name != "keytao_batch_add_to_draft":
        return ""
    items = state.args.get("items")
    if not isinstance(items, list):
        return ""
    if len(items) == 1 and isinstance(items[0], dict):
        item = items[0]
        command_text = (
            f"删词 {str(item.get('word') or '').strip()} "
            f"{str(item.get('code') or '').strip().lower()}"
        )
        parsed_delete = parse_dictionary_delete_command(command_text)
        if (
            parsed_delete is None
            or str(item.get("action") or "Create").strip() != "Delete"
            or parsed_delete.word != str(item.get("word") or "").strip()
            or parsed_delete.code
            != str(item.get("code") or "").strip().lower()
            or not str(item.get("type") or "").strip()
        ):
            return ""
        return render_executable_suggestion(
            command_text,
            words=(parsed_delete.word,),
        )
    if len(items) != 2:
        return ""
    delete_item = items[0] if isinstance(items[0], dict) else {}
    create_item = items[1] if isinstance(items[1], dict) else {}
    replace_command_text = (
        f"修改 {str(delete_item.get('code') or '').strip().lower()}："
        f"{str(delete_item.get('word') or '').strip()} → "
        f"{str(create_item.get('word') or '').strip()}"
    )
    parsed_replace = parse_replace_at_code(replace_command_text)
    if (
        parsed_replace is not None
        and replace_at_code_items_match(parsed_replace, items)
    ):
        return render_executable_suggestion(
            replace_command_text,
            words=(parsed_replace.old_word, parsed_replace.new_word),
        )
    command_text = (
        f"删除 {str(delete_item.get('word') or '').strip()} "
        f"{str(delete_item.get('code') or '').strip().lower()}；"
        f"添加 {str(create_item.get('word') or '').strip()} "
        f"{str(create_item.get('code') or '').strip().lower()}"
    )
    parsed = parse_dictionary_recode_command(command_text)
    if parsed is None or not dictionary_recode_items_match(parsed, items):
        return ""
    return render_executable_suggestion(command_text, words=(parsed.word,))


def _pending_choice_options(state: PendingToolConfirm) -> Optional[List[Dict[str, Any]]]:
    """Validate and return the exact plans carried by one live A/B offer."""
    if state.function_name != _PENDING_CHOICE_OFFER_FUNCTION:
        return None
    reason = state.args.get("_choice_reason", "")
    if not isinstance(reason, str) or len(reason) > 2048:
        return None
    raw_options = state.args.get("options")
    if not isinstance(raw_options, list) or len(raw_options) != 2:
        return None
    options: List[Dict[str, Any]] = []
    labels: List[str] = []
    for raw in raw_options:
        if not isinstance(raw, dict):
            return None
        label = str(raw.get("label") or "").strip().upper()
        description = str(raw.get("description") or "").strip()
        plan = raw.get("plan")
        if label not in {"A", "B"} or not description or len(description) > 240:
            return None
        if plan is not None:
            if (
                not isinstance(plan, dict)
                or not str(plan.get("function_name") or "").strip()
                or str(plan.get("function_name") or "").strip()
                == _PENDING_CHOICE_OFFER_FUNCTION
                or not isinstance(plan.get("args"), dict)
                or str(plan.get("confirmation_source") or "")
                != "server_warning"
            ):
                return None
            sealed_plan = PendingToolConfirm(
                function_name=str(plan["function_name"]),
                args=dict(plan["args"]),
                confirmation_source="server_warning",
            )
            if not server_warning_ticket_is_complete(sealed_plan):
                return None
        labels.append(label)
        options.append({
            "label": label,
            "description": description,
            "plan": dict(plan) if isinstance(plan, dict) else None,
        })
    return options if labels == ["A", "B"] else None


def _build_pending_choice_offer(
    options: Tuple[Tuple[str, str, Optional[PendingToolConfirm]], ...],
    *,
    protected_add_pairs: Tuple[Tuple[str, str], ...] = (),
    reason: str = "",
) -> Optional[PendingToolConfirm]:
    """Seal the same option plans used by both the renderer and resolver."""
    state = PendingToolConfirm(
        function_name=_PENDING_CHOICE_OFFER_FUNCTION,
        args={
            **({"_choice_reason": reason.strip()} if reason else {}),
            "options": [
                {
                    "label": str(label or "").strip().upper(),
                    "description": str(description or "").strip(),
                    "plan": (
                        {
                            "function_name": plan.function_name,
                            "args": dict(plan.args),
                            "confirmation_source": plan.confirmation_source,
                        }
                        if isinstance(plan, PendingToolConfirm)
                        else None
                    ),
                }
                for label, description, plan in options
            ],
            "protected_add_pairs": [
                [str(word or "").strip(), str(code or "").strip().lower()]
                for word, code in protected_add_pairs
                if str(word or "").strip() and str(code or "").strip()
            ],
        },
    )
    return state if _pending_choice_options(state) is not None else None


def render_pending_choice_offer(state: PendingToolConfirm) -> str:
    """Render only the A/B labels and descriptions sealed in the live record."""
    options = _pending_choice_options(state)
    if options is None:
        return ""
    return "\n".join([
        *([state.args["_choice_reason"]] if state.args.get("_choice_reason") else []),
        "请确认怎么处理，二选一：",
        *(f"{option['label']}. {option['description']}" for option in options),
        "回复 A 或 B 即可。",
    ])


def _parse_pending_choice_label(message: str) -> str:
    """Parse only the exact answer forms advertised by a live A/B offer."""
    source = re.sub(
        r"^\s*@[^\s]+\s*",
        "",
        trusted_mutation_source(message),
        count=1,
    ).strip()
    match = _PENDING_CHOICE_LABEL_RE.fullmatch(source)
    return match.group(1).upper() if match is not None else ""


async def _handle_pending_choice_offer(
    state_record: PendingStateRecord,
    message: str,
    platform: str,
    user_id: str,
    conv_key: ConversationKey,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> Optional[str]:
    state = state_record.state
    if not isinstance(state, PendingToolConfirm):
        return None
    options = _pending_choice_options(state)
    if options is None:
        return None
    label = _parse_pending_choice_label(message)
    if not label:
        add_item = explicit_complete_add_item(message)
        protected_pairs = {
            (str(value[0]).strip(), str(value[1]).strip().lower())
            for value in state.args.get("protected_add_pairs") or []
            if isinstance(value, list) and len(value) == 2
        }
        if (
            add_item is not None
            and (
                str(add_item.get("word") or "").strip(),
                str(add_item.get("code") or "").strip().lower(),
            ) in protected_pairs
        ):
            return _assert_plain_user_facing_reply(
                "当前仍有该词的腾位处理方案待选择；"
                "不会把腾位意图降级为普通重复添加。\n"
                + render_pending_choice_offer(state)
            )
        return None
    selected = next(
        (option for option in options if option["label"] == label),
        None,
    )
    if selected is None:
        return None
    if not conversation_state_store.begin_execution(state_record):
        return render_remediation_reply("当前选择正在处理中", command="查看草稿")
    raw_plan = selected.get("plan")
    if raw_plan is None:
        conversation_state_store.complete_execution(state_record)
        return _assert_plain_user_facing_reply(
            f"已选择方案 {label}：{selected['description']}。"
        )
    plan = PendingToolConfirm(
        function_name=str(raw_plan["function_name"]),
        args=dict(raw_plan["args"]),
        confirmation_source=str(
            raw_plan.get("confirmation_source") or "local_preview"
        ),
    )
    response = await _execute_confirmed_tool(
        plan,
        platform,
        user_id,
        conv_key,
        space_key,
        owner_label,
    )
    conversation_state_store.complete_execution(state_record)
    return response


def _pending_from_requested_encode(
    word: str,
    code: str,
    encoding: Dict,
) -> Optional[PendingAddWord]:
    """Build a sealed one-reading state from requested-code encode facts."""
    requested_base = code.rstrip("o") or code
    variants = [
        value
        for key in (
            "alternatePronunciationCodes",
            "alternatePhrasePronunciationCodes",
        )
        for value in encoding.get(key) or []
        if isinstance(value, dict)
    ]
    group_codes: List[str] = []
    matched_variant: Optional[Dict] = None
    for variant in variants:
        codes = list(dict.fromkeys(
            str(value or "").strip().lower()
            for value in variant.get("codes") or []
            if str(value or "").strip()
        ))
        if code in codes:
            group_codes = codes
            matched_variant = variant
            break
    if not group_codes:
        all_codes = list(dict.fromkeys(
            str(value or "").strip().lower()
            for key in (
                "requestedCandidateCodes",
                "candidateCodes",
                "codes",
                "altCodes",
            )
            for value in encoding.get(key) or []
            if str(value or "").strip()
        ))
        group_codes = [
            value
            for value in all_codes
            if (value.rstrip("o") or value) == requested_base
        ]
    if code not in group_codes:
        return None

    chars = encoding.get("chars")
    if not isinstance(chars, list) or not chars:
        return None
    pinyins = [
        str(value.get("pinyin") or "").strip()
        if isinstance(value, dict)
        else ""
        for value in chars
    ]
    if any(not value for value in pinyins):
        return None
    if matched_variant is not None:
        variant_pinyin = str(matched_variant.get("pinyin") or "").strip()
        char_index = matched_variant.get("charIndex")
        if len(pinyins) == 1 and variant_pinyin:
            pinyins[0] = variant_pinyin
        elif (
            isinstance(char_index, int)
            and not isinstance(char_index, bool)
            and 0 <= char_index < len(pinyins)
            and variant_pinyin
        ):
            pinyins[char_index] = variant_pinyin
    pinyin = " ".join(_plain_pinyin(value) for value in pinyins)
    if not pinyin.strip():
        return None

    status_map = {
        str(status.get("code") or "").strip().lower(): status
        for status in encoding.get("candidateStatuses") or []
        if isinstance(status, dict)
    }
    group_statuses = [status_map.get(candidate_code) for candidate_code in group_codes]
    if any(
        not isinstance(status, dict)
        or not isinstance(status.get("occupied"), bool)
        for status in group_statuses
    ):
        return None
    candidates = [
        (candidate_code, bool(status_map[candidate_code]["occupied"]))
        for candidate_code in group_codes
    ]
    occupied_words = {
        candidate_code: [
            str(value or "").strip()
            for value in status_map[candidate_code].get("words") or []
            if str(value or "").strip()
        ]
        for candidate_code, occupied in candidates
        if occupied
    }
    state = PendingAddWord(
        word=word,
        recommended_code=code,
        candidates=candidates,
        occupied_words=occupied_words,
        pronunciation_codes={candidate_code: pinyin for candidate_code in group_codes},
        pronunciation_recommended_codes=[code],
        needs_manual_review=True,
        manual_review_reason="用户显式选择了多音候选，需管理员审核",
    )
    return _attach_server_candidate_snapshot(
        state,
        [status for status in group_statuses if isinstance(status, dict)],
    )


async def _try_handle_explicit_pending_replacement(
    state: PendingAddWord,
    message: str,
    platform: str,
    user_id: str,
    conv_key: ConversationKey,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> Optional[str]:
    """Replace one live selection from an exact same-word word+code command."""
    command = explicit_complete_add_item(message)
    if command is None:
        return None
    word = str(command.get("word") or "").strip()
    code = str(command.get("code") or "").strip().lower()
    if word != state.word:
        return None
    explicit = parse_explicit_code_request(
        _strip_command_message_prefixes(trusted_mutation_source(message)), state.word,
    )
    if explicit is not None and state.server_candidates and state.pronunciation_codes:
        return await handle_pending_message_core(
            message, platform, user_id, conv_key, space_key=space_key,
            owner_label=owner_label, allow_intent_model=False,
        )

    review_json = await call_tool_function(
        "keytao_prepare_reviewed_add",
        {"word": word},
        platform,
        user_id,
    )
    try:
        review = json.loads(review_json)
    except Exception:
        review = {}
    if not isinstance(review, dict) or review.get("success") is not True:
        message_text = str(
            review.get("message") if isinstance(review, dict) else ""
        ).strip()
        return (
            f"无法重新核验「{word}」的编码；"
            f"{message_text or '编码服务没有返回可验证结果'}。没有执行添加"
        )
    if review.get("pronunciationUnresolved") is True:
        return str(
            review.get("message")
            or f"「{word}」当前读音无法确定；没有执行添加"
        )

    reviewed_prompt = _format_reviewed_add_prompt(review)
    replacement = (
        _parse_pending_add_word(reviewed_prompt)
        if reviewed_prompt
        else None
    )
    statuses = [
        status
        for pronunciation in review.get("pronunciations") or []
        if isinstance(pronunciation, dict)
        for status in pronunciation.get("candidateStatuses") or []
        if isinstance(status, dict)
    ]
    if replacement is not None:
        _attach_server_candidate_snapshot(
            replacement,
            statuses,
            review.get("candidateOrderingAssessments"),
        )
    candidate_codes = {
        candidate_code
        for candidate_code, _occupied in (replacement.candidates if replacement else [])
    }
    used_requested_encode = False
    if replacement is None or code not in candidate_codes:
        encode_json = await call_tool_function(
            "keytao_encode",
            {"word": word, "requested_code": code},
            platform,
            user_id,
        )
        try:
            encoding = json.loads(encode_json)
        except Exception:
            encoding = {}
        if isinstance(encoding, dict) and encoding.get("success") is True:
            requested_replacement = _pending_from_requested_encode(
                word,
                code,
                encoding,
            )
            if requested_replacement is not None:
                replacement = requested_replacement
                used_requested_encode = True
                candidate_codes = {
                    candidate_code
                    for candidate_code, _occupied in replacement.candidates
                }
    if replacement is None or code not in candidate_codes:
        options = _compact_reviewed_reading_options(review)
        suffix = f"；可选读音链：{options}" if options else ""
        return (
            f"编码 {code} 不是「{word}」当前审词结果中的有效候选编码"
            f"{suffix}；没有执行添加"
        )

    replacement.recommended_code = code
    review_flag = review_flags.read_manual_review_flag(review)
    if review_flag is None:
        audit = review.get("preSubmitAudit")
        if isinstance(audit, dict):
            review_flag = audit.get("autoApprove") is not True
    if review_flag is not None and not used_requested_encode:
        replacement.needs_manual_review = bool(review_flag)
        replacement.manual_review_reason = review_flags.manual_review_reason(
            review
        ) or str(
            (review.get("preSubmitAudit") or {}).get("summary")
            if isinstance(review.get("preSubmitAudit"), dict)
            else ""
        ).strip()
    conversation_state_store.delete(conv_key)
    pinyin, reviewed_codes = _pending_reviewed_reading(replacement, code)
    kwargs = {
        "reviewed_pinyin": pinyin,
        "reviewed_candidate_codes": reviewed_codes,
    }
    if command.get("submitAfter") is True:
        return await _execute_add_to_draft_and_submit(
            word,
            code,
            platform,
            user_id,
            space_key,
            owner_label,
            replacement.code_remarks.get(code, ""),
            replacement.needs_manual_review,
            _pending_add_ordering_summary(replacement, code),
            **kwargs,
        )
    return await _execute_add_to_draft(
        word,
        code,
        platform,
        user_id,
        space_key,
        owner_label,
        replacement.code_remarks.get(code, ""),
        replacement.needs_manual_review,
        reviewed_pinyin=pinyin,
        reviewed_candidate_codes=reviewed_codes,
    )


def _trusted_word_action_preview_args(
    state: PendingTrustedWordRecord,
) -> Optional[Dict[str, Any]]:
    """Project a trusted row into a delete preview without adding authority."""
    if not trusted_word_record_is_complete(state):
        return None
    return {
        "items": [{
            "action": "Delete",
            "word": state.word,
            "code": state.code,
            "type": state.phrase_type,
        }],
        "preview_only": True,
    }


async def _resolve_pending_trusted_word_action(
    state_record: PendingStateRecord,
    message: str,
    platform: str,
    user_id: str,
    conv_key: ConversationKey,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
) -> Optional[str]:
    """Resolve a previous-turn row into a server-sealed plan, never a write."""
    state = state_record.state
    if (
        not isinstance(state, PendingTrustedWordRecord)
        or state_record.owner_key.actor_id != str(user_id)
    ):
        return None
    explicit = parse_explicit_code_request(
        _strip_command_message_prefixes(trusted_mutation_source(message)), state.word,
    )
    if explicit is not None and trusted_word_record_is_complete(state):
        return await _execute_explicit_entry_code_request(
            ExplicitEntryCodeRequest(state.word, explicit.code, explicit.submit_after, explicit.pinyin),
            message, platform, user_id, conv_key, space_key, owner_label,
        )
    if not _pending_trusted_word_action_matches(state, message):
        conversation_state_store.delete(conv_key)
        return None
    action_name = _compact_command_text(message)
    preview_args = _trusted_word_action_preview_args(state)
    if preview_args is None:
        conversation_state_store.complete_execution(state_record)
        return render_remediation_reply(
            "上一轮词条记录不完整或已变化，本次未写入"
        )
    if not conversation_state_store.begin_execution(state_record):
        return render_remediation_reply(
            "当前操作正在处理中",
            command=action_name,
        )

    try:
        preview_json = await call_tool_function(
            "keytao_batch_add_to_draft",
            preview_args,
            platform,
            user_id,
        )
    except BaseException:
        # preview_only cannot have written; the same exact row may be retried.
        conversation_state_store.abort_execution(state_record)
        raise
    try:
        preview = json.loads(preview_json)
    except Exception:
        preview = None
    if not isinstance(preview, dict):
        conversation_state_store.complete_execution(state_record)
        return render_remediation_reply(
            "后续操作没有返回可验证的服务端计划，本次未写入"
        )
    if preview.get("transportError") is True:
        conversation_state_store.abort_execution(state_record)
        return render_remediation_reply(
            "连接服务失败；尚未生成确认计划，本次未写入",
            command=action_name,
        )
    if preview.get("requiresConfirmation") is not True:
        conversation_state_store.complete_execution(state_record)
        return render_remediation_reply(
            "后续操作没有返回完整的待确认计划，本次未写入"
        )

    pending = _pending_state_from_server_warning(
        PendingToolConfirm(
            function_name="keytao_batch_add_to_draft",
            args={"items": preview_args["items"]},
            confirmation_source="local_preview",
        ),
        preview,
    )
    if not server_warning_ticket_is_complete(pending):
        conversation_state_store.complete_execution(state_record)
        return render_remediation_reply(
            "后续操作的确认内容不完整，本次未写入"
        )
    display_data = {
        **preview,
        "pendingItems": list(preview_args["items"]),
    }
    prompt = _format_server_warning_confirmation(
        "keytao_batch_add_to_draft",
        display_data,
    )
    if len(prompt) > MAX_REPLACE_CONFIRMATION_CHARS:
        conversation_state_store.complete_execution(state_record)
        return _append_batch_url_if_missing(
            render_remediation_reply(
                "后续操作的确认内容过长，本次未写入"
            ),
            display_data,
        )
    saved = conversation_state_store.replace_claimed_trusted_word_record(
        state_record,
        pending,
        owner_label=owner_label,
    )
    if not saved:
        conversation_state_store.abort_execution(state_record)
        return render_remediation_reply(
            "后续操作已变化或失效，本次未写入"
        )
    logger.info(
        "Resolved previous-turn word record into server-warning confirmation: "
        f"owner={state_record.owner_key} action={action_name} "
        f"word={state.word} code={state.code}"
    )
    return _append_pending_ticket_challenge(
        _append_batch_url_if_missing(prompt, display_data),
        conv_key,
    )


async def _bind_explicit_pending_code(state, request, platform, user_id):
    """Seal a direct user's code only after reading and exact typed occupancy checks."""
    validation = validate_explicit_code(state, request.code, getattr(request, "pinyin", ""))
    if not validation.valid:
        return None, validation.reason + "；本次未写入。"
    raw = await call_tool_function(
        "keytao_lookup_by_code", {"code": request.code}, platform, user_id,
    )
    try:
        lookup = json.loads(raw)
    except (ValueError, TypeError):
        lookup = {}
    if lookup.get("success") is not True or not isinstance(lookup.get("phrases"), list):
        return None, f"未能核验编码 {request.code} 是否有占用，本次未写入。"
    rows = [row for row in lookup["phrases"] if isinstance(row, dict)
            and row.get("code") == request.code and row.get("type") == state.phrase_type]
    if any(row.get("word") == state.word for row in rows):
        return None, f"「{state.word}」已有编码 {request.code}，不能重复添加同一词条和编码；本次未写入。"
    if any(not isinstance(row.get("word"), str) or not row["word"] for row in rows):
        return None, f"编码 {request.code} 的占用记录不完整，本次未写入。"
    derived = _pending_add_word_from_payload(json.loads(json.dumps(_pending_add_word_payload(state))))
    code = request.code
    derived.candidates = [(value, occupied) for value, occupied in derived.candidates if value != code]
    derived.candidates.append((code, bool(rows)))
    derived.server_candidates = list(derived.candidates)
    derived.recommended_code = code
    derived.pronunciation_recommended_codes = [code]
    occupants = list(dict.fromkeys(row["word"] for row in rows))
    derived.occupied_words[code] = occupants
    derived.server_occupied_words[code] = occupants
    derived.pronunciation_codes[code] = validation.pinyin
    derived.server_entries_by_code[code] = [(row["word"], row["weight"]) for row in rows
                                          if isinstance(row.get("weight"), int)]
    if rows:
        from ..utils.keytao_review import assess_candidate_chain_commonness
        assessments = await assess_candidate_chain_commonness({
            "word": state.word, "type": state.phrase_type, "recommendedCode": code,
            "candidateStatuses": [{"code": code, "occupied": True, "words": occupants, "phrases": rows}],
        })
        derived.server_ordering_assessments = [
            assessment for assessment in derived.server_ordering_assessments
            if assessment.get("occupantCode") != code
        ] + assessments
    if validation.manual_reason:
        derived.needs_manual_review = True
        derived.manual_review_reason = validation.manual_reason
        derived.code_remarks[code] = f"喵喵审词：读音 {validation.pinyin}；{validation.manual_reason}"
    elif code not in derived.code_remarks:
        derived.code_remarks[code] = f"喵喵审词：读音 {validation.pinyin}；{derived.manual_review_reason}"
    return derived, ""


async def try_verify_pending_fly_proposal(message, platform, user_id, conv_key, space_key=None, owner_label=""):
    """Verify a code correction without treating it as an instruction to write."""
    source = _strip_command_message_prefixes(trusted_mutation_source(message)).strip()
    proposed = re.fullmatch(r"(?:应该是|应是|应该用)\s*([a-z0-9]+)\s*(?:这个|这个编码的)?飞键[。.!！]?", source)
    record = conversation_state_store.get_record(conv_key)
    if proposed is None or record is None or record.execution_id:
        return None
    origin = _candidate_origin_for_ticket(record.state)
    if len(origin.get("queryWords") or []) != 1:
        return "未能把该飞键与唯一的词和读音绑定，请明确词名；本次未写入。"
    try:
        state = _pending_add_word_from_payload(json.loads(json.dumps(origin.get("reviewedState"))))
    except (ValueError, TypeError, KeyError):
        return "当前缺少完整读音记录，未能核验该飞键；请重新查询该词。"
    code = proposed[1]
    validation = validate_explicit_code(state, code)
    if not validation.valid:
        return validation.reason + "；本次未写入。"
    # Verification grants no write or ticket replacement. A later explicit
    # add command reuses S56's occupancy check and sealed capability.
    return (f"核验结果：{code} 符合「{state.word}」读音 {validation.pinyin} 的编码规则。"
            + (f"{validation.manual_reason}。" if validation.manual_reason else "")
            + "本次仅核验，未写入。")


async def _execute_explicit_entry_code_request(
    request: ExplicitEntryCodeRequest,
    message: str,
    platform: str,
    user_id: str,
    conv_key: ConversationKey,
    space_key: Optional[Tuple[str, str]],
    owner_label: str,
    *,
    prepare_only: bool = False,
) -> str:
    """Review the named item before granting its additional-code capability."""
    raw = await call_tool_function(
        "keytao_prepare_reviewed_add", {"word": request.word}, platform, user_id,
    )
    try:
        review = json.loads(raw)
    except (TypeError, ValueError):
        review = {}
    phrase_type = "Single" if len(request.word) == 1 else "Phrase"
    resolvable_ambiguity = bool(
        isinstance(review, dict)
        and review.get("reviewVerdictSite") == "pronunciation_unresolved"
        and isinstance(review.get("multiSenseChoice"), dict)
        and review["multiSenseChoice"].get("status") == "ambiguous"
        and review.get("lookupFailed") is False
    )
    if (
        not isinstance(review, dict)
        or review.get("success") is not True
        or review.get("word") != request.word
        or str(review.get("type") or "Phrase") != phrase_type
        or (review.get("reviewDisposition") == "BLOCK" and not resolvable_ambiguity)
        or (review.get("pronunciationUnresolved") is True and not resolvable_ambiguity)
    ):
        return f"未能取得「{request.word}」的已审读音和正确词条类型，本次未写入。"
    from ..utils.explicit_code import select_explicit_candidate_inventory

    inventory, failure = select_explicit_candidate_inventory(review, request.code, request.pinyin)
    if failure:
        return f"已收到「{request.word}」的编码 {request.code}；{failure}，本次未写入。"
    if inventory is None or any(
        not str(inventory.readings.get(code) or "").strip()
        for code, _occupied in inventory.candidates
    ):
        return f"「{request.word}」缺少可核验的读音候选，本次未写入。"
    manual_reason = str(review.get("manualReviewReason") or "")
    if resolvable_ambiguity or request.pinyin:
        manual_reason = "用户指定编码或读音已匹配候选，采用所列读音，需管理员复核"
    reviewed = PendingAddWord(
        word=request.word,
        recommended_code=str(review.get("recommendedCode") or ""),
        candidates=list(inventory.candidates),
        occupied_words={status["code"]: list(status["words"]) for status in inventory.statuses},
        pronunciation_codes=dict(inventory.readings),
        phrase_type=phrase_type,
        needs_manual_review=resolvable_ambiguity or bool(request.pinyin) or review.get("needsManualReview") is not False,
        manual_review_reason=manual_reason,
        code_remarks={code: f"喵喵审词：读音 {pinyin}；{manual_reason}" for code, pinyin in inventory.readings.items()},
    )
    _attach_server_candidate_snapshot(
        reviewed, [dict(status) for status in inventory.statuses],
        review.get("candidateOrderingAssessments"),
    )
    derived, failure = await _bind_explicit_pending_code(reviewed, request, platform, user_id)
    if failure:
        return failure
    if prepare_only:
        # This turn names a code, but carries no write capability. Keep the
        # displayed choice and next-turn confirmation bound to that exact code.
        derived.candidates = [(request.code, dict(derived.server_candidates)[request.code])]
        derived.server_candidates = list(derived.candidates)
        derived.server_ordering_assessments = [
            assessment for assessment in derived.server_ordering_assessments
            if assessment.get("occupantCode") == request.code
        ]
    previous = conversation_state_store.get_record(conv_key)
    preserve_submit = bool(
        not prepare_only and previous is not None and isinstance(previous.state, PendingToolConfirm)
        and previous.state.function_name == "keytao_submit_batch"
    )
    if not preserve_submit and not conversation_state_store.set(conv_key, derived, space_key=space_key, owner_label=owner_label):
        return "指定编码的核验记录未能保存，本次未写入。"
    if prepare_only:
        from .openai_chat import _render_live_single_candidate_record

        rendered = _render_live_single_candidate_record(conversation_state_store.get_record(conv_key))
        if not rendered:
            conversation_state_store.delete(conv_key)
            if previous is not None and isinstance(previous.state, PendingToolConfirm) and previous.state.function_name == "keytao_submit_batch":
                conversation_state_store.put_back(previous)
            return "指定编码的候选展示未能通过核验，本次未写入。"
        return _announce_replaced_submit(previous, conv_key, "已重新核验指定编码，本次尚未写入。\n" + rendered)
    existing_codes = tuple(dict.fromkeys(
        row["code"] for row in review.get("existing", [])
        if isinstance(row, dict) and row.get("word") == request.word
        and row.get("type") == phrase_type
        and isinstance(row.get("code"), str)
        and re.fullmatch(r"[a-z]{1,6}", row["code"])
        and row["code"] != request.code
    )) if isinstance(review.get("existing"), list) else ()
    capability = (
        _ExplicitAdditionalCodeCapability(request.word, request.code, phrase_type, existing_codes)
        if existing_codes and dict(derived.server_candidates).get(request.code) is False
        else None
    )
    capability_token = _current_explicit_additional_code.set(capability)
    try:
        if preserve_submit:
            intent = _structural_pending_add_word_intent(message, derived)
            if intent is None:
                return "指定编码未能与已审记录绑定，本次未写入。"
            response = await _handle_pending_add_word(
                derived, message, platform, user_id, [], space_key, owner_label,
                intent, lambda: None,
            )
        else:
            response = await handle_pending_message_core(
                message, platform, user_id, conv_key,
                space_key=space_key, owner_label=owner_label, allow_intent_model=False,
            )
    finally:
        _current_explicit_additional_code.reset(capability_token)
    if response is None:
        return "指定编码未能与已审记录绑定，本次未写入。"
    note = (
        f"审词：读音 {derived.pronunciation_codes[request.code]}；{derived.manual_review_reason}\n"
        if derived.manual_review_reason else ""
    )
    return note + response


def fresh_entry_code_selection(message: str) -> Optional[ExplicitEntryCodeRequest]:
    """Recognize one closed lexical pair without granting mutation authority."""
    selection = parse_reviewed_selection_command(message)
    if selection is None or selection.action or selection.shift_words:
        return None
    pairs = parse_reviewed_multi_word_selection(message)
    if pairs is None or len(pairs) != 1:
        return None
    word, code = pairs[0]
    if (
        not code.isalpha()
        or not looks_like_lexical_review_target(word)
        or _NEGATIVE_MODAL_RE.match(word)
        or _POSITIONAL_REPORTED_CONTEXT_RE.match(word)
    ):
        return None
    return ExplicitEntryCodeRequest(word, code)


async def prepare_fresh_entry_code_selection(
    request: ExplicitEntryCodeRequest,
    message: str,
    platform: str,
    user_id: str,
    conv_key: ConversationKey,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
    *,
    prepare_only: bool = True,
) -> str:
    """Read current actor facts before preparing or executing an exact code."""
    raw = await call_tool_function(
        "keytao_pending_items_by_words", {"words": [request.word]}, platform, user_id,
    )
    try:
        pending = json.loads(raw)
    except (TypeError, ValueError):
        pending = {}
    if (
        not isinstance(pending, dict)
        or pending.get("success") is not True
        or pending.get("complete") is not True
        or not isinstance(pending.get("items"), list)
    ):
        return "暂时无法核验当前草稿和待审核批次，本次未继续。"
    phrase_type = "Single" if len(request.word) == 1 else "Phrase"
    matching = [
        item for item in trusted_pending_word_items(pending.get("items"))
        if item["word"] == request.word and item["code"] == request.code
        and item["type"] == phrase_type and item["action"] in {"Create", "Change"}
    ]
    if matching:
        drafts = [item for item in matching if item["source"] == "draft"]
        if not prepare_only and request.submit_after and len(drafts) == 1:
            result = await _perform_submit_current_draft(
                platform,
                user_id,
                batch_id=drafts[0]["batchId"],
                auto_confirm=True,
                authorized_items=[{
                    "action": "Create", "word": request.word,
                    "code": request.code, "type": phrase_type,
                }],
            )
            if result.pending_state is not None:
                saved = conversation_state_store.set(
                    conv_key, result.pending_state,
                    space_key=space_key, owner_label=owner_label,
                )
                if not saved:
                    return _append_batch_url_if_missing(
                        "提交确认记录未能保存，本次未提交；草稿内容仍保留。",
                        result.data or {}, drafts[0],
                    )
            return (
                f"已复用草稿中的「{request.word}」→ {request.code}，本次未重复添加。\n"
                + result.text
            )
        return prepend_pending_word_reminders("本次未重复添加。", matching, words=(request.word,))
    return await _execute_explicit_entry_code_request(
        request, message, platform, user_id, conv_key, space_key, owner_label,
        prepare_only=prepare_only,
    )


async def try_handle_explicit_entry_code_command(
    message: str,
    platform: str,
    user_id: str,
    conv_key: ConversationKey,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> Optional[str]:
    """Claim only a complete word-and-code command before lexical discovery."""
    source = _strip_command_message_prefixes(trusted_mutation_source(message))
    request = parse_explicit_entry_code_request(source)
    if request is None:
        return None
    record = conversation_state_store.get_record(conv_key)
    if record is not None:
        if record.execution_id:
            return "上一项操作仍在处理中，本次未追加编码。"
        if isinstance(record.state, PendingAddWord):
            if record.state.word == request.word:
                return await handle_pending_message_core(
                    message, platform, user_id, conv_key, space_key=space_key,
                    owner_label=owner_label, allow_intent_model=False,
                )
        elif isinstance(record.state, PendingToolConfirm):
            recent_write_receipt = bool(
                record.state.function_name == "keytao_submit_batch"
                and record.state.args.get("_recent_own_write") is True
            )
            if not recent_write_receipt:
                return _format_live_ticket_precedence_message(record.state)
    set_turn_flow("explicit-code")
    if request.submit_after:
        return await prepare_fresh_entry_code_selection(
            request, message, platform, user_id, conv_key, space_key, owner_label,
            prepare_only=False,
        )
    return await _execute_explicit_entry_code_request(
        request, message, platform, user_id, conv_key, space_key, owner_label,
    )


def _candidate_origin_for_ticket(state) -> Dict:
    """Retain the original numbered inventory across a server plan preview."""
    if isinstance(state, PendingAddWord) and state.server_candidates and state.server_candidates == state.candidates:
        return {"queryWords": [state.word], "reviewedState": json.loads(json.dumps(_pending_add_word_payload(state)))}
    if isinstance(state, PendingToolConfirm):
        if state.args.get("_candidate_origin"):
            return state.args["_candidate_origin"]
        if state.args.get("_reviewed_multi_word") is True:
            scopes = state.args.get("_candidate_scopes") or []
            words = state.args.get("_query_words") or [scope.get("word") for scope in scopes]
            return {"queryWords": words, "reviewedState": json.loads(json.dumps(scopes[0].get("reviewedState")))
                    if len(scopes) == 1 and scopes[0].get("word") in words else None}
    return {}


async def try_reselect_live_ticket(message, platform, user_id, conv_key, space_key=None, owner_label=""):
    """A live ordinal replaces the preview; it never confirms the old plan."""
    record = conversation_state_store.get_record(conv_key)
    if not record or record.execution_id or not isinstance(record.state, PendingToolConfirm):
        return None
    state = record.state
    if not server_warning_ticket_is_complete(state) or state.function_name not in {
        "keytao_shift_phrase_code", "keytao_create_phrase", "keytao_batch_add_to_draft",
    }:
        return None
    source = _strip_command_message_prefixes(trusted_mutation_source(message))
    selection = parse_pending_candidate_selection(source)
    if selection is None and re.fullmatch(r"[0-9\s,，、;；]+[。.!！]?", unicodedata.normalize("NFKC", source)):
        return "候选编号无效或重复，请按原候选列表选择；本次未写入。"
    if selection is None or not selection.indices:
        return None
    origin = state.args.get("_candidate_origin") or {}
    if len(origin.get("queryWords") or []) > 1 and not origin.get("reviewedState"):
        return "原候选列表包含多个词，请明确词名和编码后重新审词；本次未写入，原确认计划未执行。"
    try:
        reviewed = _pending_add_word_from_payload(origin.get("reviewedState"))
    except (ValueError, TypeError, KeyError):
        return "当前确认缺少完整的原候选记录，未能核验该编号；请重新查询，本次未写入。"
    if (not reviewed.server_candidates or reviewed.server_candidates != reviewed.candidates
            or not isinstance(origin.get("queryWords"), list)
            or reviewed.word not in origin["queryWords"]):
        return "当前确认缺少完整的原候选记录，未能核验该编号；请重新查询，本次未写入。"
    if len(selection.indices) != 1 or not 1 <= selection.indices[0] <= len(reviewed.candidates):
        return f"请从原候选 1-{len(reviewed.candidates)} 中选择一个编号；本次未写入。"
    code, occupied = reviewed.candidates[selection.indices[0] - 1]
    reading, codes = _pending_reviewed_reading(reviewed, code)
    if not reading or code not in codes:
        return "该候选缺少完整的读音核验记录，本次未写入。"
    args = (_create_phrase_args(reviewed, code) if not occupied else {
        "word": reviewed.word, "target_code": code, "target_type": reviewed.phrase_type,
        "target_remark": reviewed.code_remarks.get(code, ""), "target_needs_manual_review": True,
    })
    args.update(_reviewed_pinyin=reading, _reviewed_candidate_codes=list(codes),
                _candidate_origin=origin, _submit_after=bool(state.args.get("_submit_after") or selection.submit_after))
    if not conversation_state_store.begin_execution(record):
        return "当前确认正在处理中，本次未改选。"
    try:
        response = await _execute_confirmed_tool(PendingToolConfirm(
            function_name="keytao_shift_phrase_code" if occupied else "keytao_create_phrase", args=args,
        ), platform, user_id, conv_key, space_key, owner_label, require_new_confirmation=True)
    finally:
        # A failed read-only preview leaves the old ticket available; a saved
        # replacement has a different identity and cannot be removed here.
        conversation_state_store.abort_execution(record)
    if conversation_state_store.get_record(conv_key) is record:
        return "改选预览未完成，原确认计划仍保留。\n" + response
    return f"已改选原候选 {selection.indices[0]}：{reviewed.word} → {code}。\n" + response


async def handle_pending_message_core(
    message: str,
    platform: str,
    user_id: str,
    conv_key: ConversationKey,
    *,
    history: Optional[List[Dict]] = None,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
    allow_intent_model: bool = True,
) -> Optional[str]:
    """Consume or re-arm one pending ticket outside an adapter-specific handler."""
    if (
        parse_eviction_modified_add(message) is not None
        or parse_compound_eviction_add_plan(message) is not None
    ):
        return None
    state_record = conversation_state_store.get_record(conv_key)
    if state_record is None or state_record.state is None:
        return None
    state = state_record.state
    if isinstance(state, PendingToolConfirm) and state.function_name == "keytao_understood_command":
        return await handle_pending_understood_command(
            state_record, message, platform, user_id, conv_key, space_key, owner_label,
        )
    if isinstance(state, PendingTrustedWordRecord):
        return await _resolve_pending_trusted_word_action(
            state_record,
            message,
            platform,
            user_id,
            conv_key,
            space_key,
            owner_label,
        )
    if (
        isinstance(state, PendingToolConfirm)
        and state.function_name == _PENDING_CHOICE_OFFER_FUNCTION
    ):
        return await _handle_pending_choice_offer(
            state_record,
            message,
            platform,
            user_id,
            conv_key,
            space_key,
            owner_label,
        )
    if state_record.execution_id:
        uncertain_action, uncertain_response = _resolve_uncertain_ticket_action(
            state_record,
            message,
        )
        if uncertain_action == "read":
            return None
        return uncertain_response

    reselected = await try_reselect_live_ticket(message, platform, user_id, conv_key, space_key, owner_label)
    if reselected is not None:
        return reselected

    explicit_review_line = ""
    if isinstance(state, PendingAddWord):
        explicit = parse_explicit_code_request(
            _strip_command_message_prefixes(trusted_mutation_source(message)), state.word,
        )
        if explicit is not None and state.server_candidates and explicit.code not in dict(state.server_candidates):
            derived, failure = await _bind_explicit_pending_code(state, explicit, platform, user_id)
            if failure:
                return failure
            if not conversation_state_store.set(conv_key, derived, space_key=space_key, owner_label=owner_label):
                return "指定编码的核验记录未能保存，本次未写入。"
            state_record = conversation_state_store.get_record(conv_key)
            state = state_record.state
            if state.manual_review_reason:
                explicit_review_line = (
                    f"审词：读音 {state.pronunciation_codes[explicit.code]}；{state.manual_review_reason}\n"
                )

    scoped_state, scoped_intent, scoped_response = (
        _resolve_multi_word_pending_candidate_selection(state, message)
    )
    if scoped_response is not None:
        return scoped_response
    if scoped_state is not None:
        state = scoped_state

    assent_rejection = (
        _pending_assent_rejection_response(state, message)
        if scoped_intent is None
        else None
    )
    if assent_rejection is not None:
        return assent_rejection

    if (
        isinstance(state, PendingAddWord)
        and _is_short_add_and_submit_request(message)
        and _pending_tool_assent_intent(state, message) is None
    ):
        return _format_full_add_and_submit_instruction(state)

    preserve_after_response = False

    def preserve_pending_state() -> None:
        nonlocal preserve_after_response
        preserve_after_response = True

    structural_tool_intent = _pending_tool_assent_intent(state, message)
    structural_candidate_intent = (
        _structural_pending_add_word_intent(message, state)
        if isinstance(state, PendingAddWord)
        else None
    )
    if scoped_intent is not None:
        pending_command_intent = scoped_intent
    elif structural_candidate_intent is not None:
        pending_command_intent = structural_candidate_intent
    elif structural_tool_intent is not None:
        pending_command_intent = structural_tool_intent
    elif not allow_intent_model:
        return None
    else:
        try:
            pending_command_intent = await _classify_message_command_intent(
                message,
                state,
            )
        except BaseException:
            raise
    if (
        scoped_intent is None
        and
        _is_sensitive_pending_control_intent(pending_command_intent)
        and not _message_authorizes_pending_state_control(
            state,
            message,
            pending_command_intent,
        )
    ):
        pending_command_intent = MessageCommandIntent()
    _record_flow_for_intent(pending_command_intent)

    if (
        pending_command_intent.intent == "draft_submit"
        and _is_explicit_draft_submit_request(message)
    ):
        if isinstance(state, PendingToolConfirm):
            return _format_live_ticket_precedence_message(state)
        return await _submit_current_draft(
            platform,
            user_id,
            space_key,
            owner_label,
        )

    try:
        if scoped_intent is not None:
            ticket_response = None
        else:
            pending_command_intent, ticket_response = await _resolve_pending_ticket_control(
                state_record,
                message,
                pending_command_intent,
                platform,
                user_id,
            )
    except BaseException:
        raise
    if ticket_response is not None:
        return _append_pending_ticket_challenge(ticket_response, conv_key)

    if pending_command_intent.intent == "pending_cancel":
        conversation_state_store.complete_execution(state_record)
        return "已取消。"

    if isinstance(state, PendingAddWord):
        if _pending_pronunciation_correction(message, state) is not None:
            response = await _try_update_pending_pronunciation(
                state,
                message,
                platform,
                user_id,
                space_key,
                owner_label,
            )
            return _append_pending_ticket_challenge(response, conv_key)
        if not conversation_state_store.begin_execution(state_record):
            return render_remediation_reply(
                "当前确认正在处理中",
                command="查看草稿",
            )
        try:
            response = await _handle_pending_add_word(
                state,
                message,
                platform,
                user_id,
                history or [],
                space_key,
                owner_label,
                pending_command_intent,
                preserve_pending_state,
            )
        except BaseException:
            # Keep the ticket in an explicit uncertain state. Replaying a
            # mutation after cancellation could duplicate a completed write.
            raise
        if response is None:
            conversation_state_store.abort_execution(state_record)
            return None
        if preserve_after_response:
            conversation_state_store.abort_execution(state_record)
        else:
            conversation_state_store.complete_execution(state_record)
        return explicit_review_line + _append_pending_ticket_challenge(response, conv_key)

    if (
        isinstance(state, PendingToolConfirm)
        and _is_pending_tool_confirm_message(state, pending_command_intent)
    ):
        if not _resolved_advertised_items_match(state):
            conversation_state_store.complete_execution(state_record)
            invalid_words = tuple(
                word for word, _code in pending_batch_display_pairs(state)
            )
            return render_remediation_reply(
                "候选集合校验失败，当前确认已失效；本次未写入",
                command=("加词 " + " ".join(invalid_words)) if invalid_words else "",
                words=invalid_words,
            )
        if not conversation_state_store.begin_execution(state_record):
            return render_remediation_reply(
                "当前确认正在处理中",
                command="查看草稿",
            )
        batch_front_insert = _pending_batch_front_insert_plan(state)
        if batch_front_insert is not None and (
            batch_front_insert.get("invalid") is True
            or batch_front_insert.get("unsupportedCount")
        ):
            preserve_pending_state()
            response = render_remediation_reply(
                "这批候选包含无法唯一锁定的重排计划，本次未写入；候选仍有效，请逐词处理",
            )
        elif batch_front_insert is not None:
            recommendation = batch_front_insert["recommendation"]
            response = await _execute_shift_to_code(
                recommendation["newWord"],
                recommendation["occupantCode"],
                platform,
                user_id,
                space_key,
                owner_label,
                submit_after=(
                    pending_command_intent.intent == "pending_add_and_submit"
                ),
                target_item=batch_front_insert["targetItem"],
                additional_items=batch_front_insert["additionalItems"],
                additional_shift_items=batch_front_insert["additionalShiftItems"],
                reviewed_pinyin=batch_front_insert["reviewedCapability"].get("pinyin", ""),
                reviewed_candidate_codes=tuple(batch_front_insert["reviewedCapability"].get("candidate_codes", ())),
                auto_confirm_shift_plan=state.args.get("_reviewed_multi_word") is True,
                expected_occupants_by_code=batch_front_insert["expectedOccupantsByCode"],
                auto_confirm_expected_shifted_words=batch_front_insert["expectedShiftedWords"],
                receipt_requested_words=state.args.get("_receipt_requested_words") or [
                    item["word"] for item in state.args.get("items", []) if isinstance(item, dict) and item.get("word")
                ],
            )
        elif (
            state.function_name == "keytao_batch_add_to_draft"
            and pending_command_intent.intent == "pending_add_and_submit"
        ):
            result = await _perform_batch_add_to_draft_and_submit(
                state.args.get("items", []),
                platform,
                user_id,
                batch_id=str(state.args.get("batch_id") or ""),
                confirmed_add=(
                    state.confirmation_source == "server_warning"
                ),
                expected_content_version=state.args.get("expected_content_version"),
                expected_warning_digest=str(
                    state.args.get("expected_warning_digest") or ""
                ),
                auto_confirm=True,
                allow_same_code=(
                    state.args.get("_allow_same_code") is True
                    or explicit_same_code_requested(message)
                ),
                reviewed_capabilities=_reviewed_multi_word_capabilities(state),
            )
            if result.pending_state is not None:
                conversation_state_store.set(
                    conv_key,
                    result.pending_state,
                    space_key=space_key,
                    owner_label=owner_label,
                )
            response = result.text
        else:
            response = await _execute_confirmed_tool(
                _pending_tool_state_with_trailing_submit(
                    state,
                    pending_command_intent,
                ),
                platform,
                user_id,
                conv_key,
                space_key,
                owner_label,
                on_transport_failure=preserve_pending_state,
            )
        response = _prepend_resolved_advertised_words(state, response)
        if preserve_after_response:
            conversation_state_store.abort_execution(state_record)
        else:
            conversation_state_store.complete_execution(state_record)
        return _append_pending_ticket_challenge(response, conv_key)

    return None


_RE_WORD_CODE_LINE = re.compile(r'^(\S+)\s+([a-z]+)\s*$')


MAX_REPLACE_CHAR_ITEMS = 50


MAX_REPLACE_CONFIRMATION_CHARS = 3500


_REPLACE_CHAR_CONTINUATION_KEY = "_replace_char_continuation"


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


async def _try_handle_replace_char(
    message: str,
    platform: str,
    user_id: str,
    command_intent: MessageCommandIntent,
    conv_key: ConversationKey,
    space_key: Optional[Tuple[str, str]] = None,
    owner_label: str = "",
) -> Optional[str]:
    """Stage a current-message replacement and require explicit confirmation."""
    if not _message_authorizes_replace_char(message, command_intent):
        return None
    old_char, new_char = command_intent.old_char, command_intent.new_char
    if not old_char or not new_char or old_char == new_char:
        return None

    items = _build_replace_char_items(message, old_char, new_char)
    if not items:
        return None

    chunk, remaining, confirmation = _replace_char_chunk(
        items,
        old_char,
        new_char,
    )
    if not chunk:
        return render_remediation_reply(
            "第一条替换也无法完整展示，本次未写入；请缩小范围"
        )

    args: Dict[str, Any] = {"items": chunk}
    if remaining:
        args[_REPLACE_CHAR_CONTINUATION_KEY] = _sealed_replace_char_continuation(
            chunk,
            remaining,
            old_char,
            new_char,
            completed=0,
            total=len(items),
        )
        confirmation = (
            f"本轮共 {len(items)} 条；先处理第 1-{len(chunk)} 条，"
            f"其余 {len(remaining)} 条已按原顺序保留。\n"
            + confirmation
        )

    saved = conversation_state_store.set(
        conv_key,
        PendingToolConfirm(
            function_name="keytao_batch_add_to_draft",
            args=args,
        ),
        space_key=space_key,
        owner_label=owner_label,
    )
    if not saved:
        return render_remediation_reply(
            "替换进度无法保存，本次未写入"
        )
    logger.info(
        f"[replace_char] Staged pattern '{old_char}'→'{new_char}', "
        f"chunk={len(chunk)} remaining={len(remaining)} awaiting confirmation"
    )
    return confirmation


def _try_handle_operation_recall(
    message: str,
    memory_context: ChatMemoryContext,
    command_intent: MessageCommandIntent,
) -> Optional[str]:
    """Answer recent bot-mediated dictionary operation recall from memory."""
    text = message.strip()
    if not text or command_intent.intent != "operation_recall":
        return None

    current_user_only = command_intent.current_user_only
    operations = memory_store.get_recent_operation_candidates(
        memory_context,
        include_current_user_only=current_user_only,
        limit=8,
    )
    if not operations:
        return render_remediation_reply(
            "当前会话里没有可由工具回执验证的词库操作记录"
        )

    lines = ["最近的词库操作：" if not current_user_only else "你最近的词库操作："]
    for item in operations:
        lines.append(f"• {_format_operation_memory_for_reply(item)}")
    lines.append("仅含机器人处理记录。")
    return "\n".join(lines)
