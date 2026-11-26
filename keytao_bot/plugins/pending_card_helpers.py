"""Pure helpers for pending-card context, natural follow-up answers, and A/B offers."""

from dataclasses import replace
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import unicodedata

from ..harness.authorization_grammar import (
    _multi_add_authorization_contract,
    _parse_code_word_changes,
    _parse_complete_add_clause,
    dictionary_recode_items_match,
    explicit_complete_add_item,
    parse_dictionary_delete_command,
    parse_dictionary_recode_command,
    parse_replace_at_code,
    parse_reviewed_selection_command,
    replace_at_code_items_match,
    trusted_mutation_source,
)
from ..harness.state import (
    PendingAddWord,
    PendingState,
    PendingToolConfirm,
    server_warning_ticket_is_complete,
)
from ..utils.pending_confirmation import (
    advertised_reply_contract,
    render_executable_suggestion,
)
from .chat_routing import (
    MessageCommandIntent,
    _canonical_keep_only_command,
    _message_authorizes_replace_char,
    _multi_word_candidate_scope_rows,
    _pending_tool_assent_intent,
    _resolve_multi_word_pending_candidate_selection,
    _strip_command_message_prefixes,
    _structural_pending_add_word_intent,
)
from .replace_char_helpers import _build_replace_char_items


SEMANTIC_ROUTE_OPS = frozenset({
    "add", "swap", "delete", "recode", "move", "reorder", "batch_replace",
    "draft_submit", "draft_recall", "draft_clear", "draft_keep_only",
    "pending_release", "pending_answer", "query", "chat", "help",
})

_MODEL_ADD_COMMAND_PREFIXES = (
    "帮我加词", "帮我添加", "帮忙加词", "帮忙添加",
    "加个词", "加一下", "帮我加", "请添加",
    "加词", "添加", "请加", "加",
)

_PENDING_CHOICE_OFFER_FUNCTION = "keytao_choice_offer"
_PENDING_CHOICE_LABEL_RE = re.compile(
    r"^(?:(?:选择?|方案)\s*)?([AB])$",
    re.IGNORECASE,
)

_CHINESE_DIGITS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
_SUBSET_RESTRICTION_RE = re.compile(
    r"(?:只(?:加|要|选|留|保留)|仅(?:加|要|选|留|保留)|"
    r"先别|别加|不加|不要|不用加|去掉|跳过|排除|除了|除开|先不加|先不要)"
)
_EXCLUDED_ORDINAL_RE = re.compile(
    r"(?:除了|除开|去掉|跳过|不要|别加|不加)\s*第\s*([1-9][0-9]?|[一二两三四五六七八九十])"
    r"(?:\s*[、,，和与跟]\s*第?\s*([1-9][0-9]?|[一二两三四五六七八九十]))*\s*个?"
)
_PREFIX_COUNT_RE = re.compile(r"前\s*([1-9][0-9]?|[一二两三四五六七八九十])\s*个")
_SUFFIX_COUNT_RE = re.compile(r"后\s*([1-9][0-9]?|[一二两三四五六七八九十])\s*个")
_EXPLICIT_ORDINAL_RE = re.compile(
    r"第\s*([1-9][0-9]?|[一二两三四五六七八九十])"
    r"((?:\s*[、,，和与跟]\s*第?\s*(?:[1-9][0-9]?|[一二两三四五六七八九十]))*)\s*个?"
)


def _parse_small_card_number(token: str) -> Optional[int]:
    cleaned = str(token or "").strip()
    if cleaned.isdecimal():
        value = int(cleaned)
        return value if value >= 1 else None
    return _CHINESE_DIGITS.get(cleaned)


def pending_shift_plan_words(state: Optional[PendingState]) -> Tuple[str, ...]:
    """Return the exact ordered entry words carried by a live shift-plan card."""
    if not isinstance(state, PendingToolConfirm) or state.function_name != "keytao_shift_phrase_code":
        return ()
    pending_display = state.args.get("_pending_display")
    shift_plan = pending_display.get("shiftPlan") if isinstance(pending_display, dict) else None
    proposed_state = shift_plan.get("proposedState") if isinstance(shift_plan, dict) else None
    if isinstance(proposed_state, list):
        words = tuple(dict.fromkeys(
            str(entry.get("word") or "").strip()
            for entry in proposed_state
            if isinstance(entry, dict) and str(entry.get("word") or "").strip()
        ))
        if words:
            return words
    ordered = state.args.get("ordered_words")
    if isinstance(ordered, (list, tuple)):
        return tuple(dict.fromkeys(
            str(word or "").strip() for word in ordered if str(word or "").strip()
        ))
    return ()


def serialize_pending_context_for_semantic_parser(state: Optional[PendingState]) -> str:
    """Serialize live pending-card structure so the AI router sees exact words, codes, and order."""
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
        payload: Dict[str, Any] = {
            "type": "pending_tool_confirm",
            "function_name": state.function_name,
            "args": {
                key: value
                for key, value in state.args.items()
                if not str(key).startswith("_")
            },
        }
        proposed_words = pending_shift_plan_words(state)
        if proposed_words:
            pending_display = state.args.get("_pending_display")
            shift_plan = pending_display.get("shiftPlan") if isinstance(pending_display, dict) else None
            proposed_state = shift_plan.get("proposedState") if isinstance(shift_plan, dict) else None
            payload["proposed_order"] = list(proposed_words)
            if isinstance(proposed_state, list):
                payload["proposed_entries"] = [
                    {
                        "index": index,
                        "word": str(entry.get("word") or "").strip(),
                        "code": str(entry.get("code") or "").strip().lower(),
                    }
                    for index, entry in enumerate(proposed_state, start=1)
                    if isinstance(entry, dict) and str(entry.get("word") or "").strip()
                ]
        items, scopes = _multi_word_candidate_scope_rows(state)
        if scopes:
            recommended = {
                str(item.get("word") or "").strip(): str(item.get("code") or "").strip().lower()
                for item in items
                if isinstance(item, dict)
            }
            payload["candidate_scopes"] = [
                {
                    "index": index,
                    "word": word,
                    "recommended_code": recommended.get(word, ""),
                    "candidates": [
                        {"index": cand_idx, "code": code, "occupied": occupied}
                        for cand_idx, (code, occupied) in enumerate(candidates, start=1)
                    ],
                }
                for index, (word, candidates) in enumerate(scopes.items(), start=1)
            ]
        return json.dumps(payload, ensure_ascii=False)
    return "none"


def _message_authorizes_multiword_subset(
    message: str,
    selected_words: Tuple[str, ...],
    ordered_card_words: Tuple[str, ...],
) -> bool:
    """Confirm that a strict subset of a multi-word card was explicitly requested by the user."""
    source = unicodedata.normalize("NFKC", str(message or "")).strip()
    if (
        not source
        or not selected_words
        or len(selected_words) >= len(ordered_card_words)
        or not set(selected_words).issubset(ordered_card_words)
    ):
        return False
    selected_set = set(selected_words)
    unselected = tuple(word for word in ordered_card_words if word not in selected_set)
    has_restriction = _SUBSET_RESTRICTION_RE.search(source) is not None

    # 1. Explicit named selection (all selected words appear literally in message).
    if all(word in source for word in selected_words):
        if all(word not in source for word in unselected) or has_restriction:
            return True

    # 2. Explicit named exclusion (e.g. "除了松弛感其他都加").
    if has_restriction and unselected and all(word in source for word in unselected):
        if all(word not in source for word in selected_words):
            return True

    # 3. Explicit ordinal exclusion (e.g. "除了第二个其他都加").
    excluded_ordinal = _EXCLUDED_ORDINAL_RE.search(source)
    if excluded_ordinal is not None:
        raw_tokens = re.findall(r"[1-9][0-9]?|[一二两三四五六七八九十]", excluded_ordinal.group(0))
        excluded_indices = {
            value for token in raw_tokens
            if (value := _parse_small_card_number(token)) is not None
            and 1 <= value <= len(ordered_card_words)
        }
        if excluded_indices:
            expected = tuple(
                word for index, word in enumerate(ordered_card_words, start=1)
                if index not in excluded_indices
            )
            if expected == selected_words:
                return True

    # 4. Explicit prefix/suffix count (e.g. "只加前两个", "后两个加一下").
    prefix_match = _PREFIX_COUNT_RE.search(source)
    if prefix_match is not None:
        count = _parse_small_card_number(prefix_match.group(1))
        if count is not None and 1 <= count < len(ordered_card_words):
            if ordered_card_words[:count] == selected_words:
                return True
    suffix_match = _SUFFIX_COUNT_RE.search(source)
    if suffix_match is not None:
        count = _parse_small_card_number(suffix_match.group(1))
        if count is not None and 1 <= count < len(ordered_card_words):
            if ordered_card_words[-count:] == selected_words:
                return True

    # 5. Explicit ordinal list (e.g. "只加第一个", "先加第1、3个", "只要第一和第三个").
    positive_text = _EXCLUDED_ORDINAL_RE.sub("", source)
    ordinal_indices: List[int] = []
    for match in _EXPLICIT_ORDINAL_RE.finditer(positive_text):
        # Skip ordinals immediately negated right after them (e.g. "第二个先别加").
        tail = positive_text[match.end():match.end() + 6]
        if re.match(r"^\s*(?:先别|别加|不加|不要|跳过|去掉)", tail):
            continue
        for token in re.findall(r"[1-9][0-9]?|[一二两三四五六七八九十]", match.group(0)):
            value = _parse_small_card_number(token)
            if value is not None and 1 <= value <= len(ordered_card_words) and value not in ordinal_indices:
                ordinal_indices.append(value)
    if ordinal_indices:
        expected = tuple(ordered_card_words[index - 1] for index in ordinal_indices)
        if expected == selected_words:
            return True

    return False


def semantic_direct_exec_allowed(request, violations=()) -> bool:
    return bool(
        "direct_exec" in {token.strip() for token in os.getenv("KEYTAO_SEMANTIC_ROUTE_OPS", "").split(",")}
        and request.frame == "direct" and request.refers_to_pending is False
        and not violations
    )


def resolve_model_add(request, message):
    """Bind unique literal codes by attachment, then by matching list order."""
    words, codes = request.operands, request.codes
    if (request.frame != "direct" or request.refers_to_pending is not False
        or request.relation or not words or len(set(words)) != len(words)
        or len(set(codes)) != len(codes)):
        return None

    if request.code_selector == "fly_key":
        if len(words) != 1 or codes or request.submit_after:
            return None
        word = re.escape(words[0])
        literal = rf'''(?:{word}|「{word}」|“{word}”|"{word}"|'{word}'|‘{word}’|『{word}』)'''
        source = _strip_command_message_prefixes(message).strip()
        if re.fullmatch(
            rf"(?:(?:新增|添加|加)\s*{literal}\s*(?:的)?|给\s*{literal}\s*加)"
            r"\s*飞键(?:编码|码)?[。.!！]?", source,
        ):
            return replace(request, codes=("",))
        return None

    def literal_words(source):
        source = re.sub(r"[A-Za-z]", " ", source)
        source = "".join(" " if unicodedata.category(char).startswith("P") else char
                         for char in source)
        return [(match.start(), match.end(), match.group())
                for match in re.finditer(r"\S+", source)]

    message = _strip_command_message_prefixes(message)
    if len(words) == 1 and not codes and re.search(r"[A-Za-z]", words[0]):
        clause = _parse_complete_add_clause(message)
        if clause is not None and clause.word == words[0] and not clause.code:
            return replace(request, codes=("",), submit_after=False)
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


def render_understood_pending_answer(request, state, prompt: str, message: str = "") -> str:
    """Round-trip a model answer through the existing ticket's closed parsers."""
    from ..harness.semantic_parser import _pending_values

    selection = parse_reviewed_selection_command(_strip_command_message_prefixes(message))
    if selection is not None and selection.rest_recommended:
        return ""
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
        ordered_card_words = tuple(scopes)
        recommended_by_word = {
            item["word"]: item["code"]
            for item in items
            if isinstance(item, dict) and item.get("word") and item.get("code")
        }
        selected_words = request.operands
        if not selected_words and len(scopes) == 1:
            selected_words = tuple(scopes)
        literal, _intent, error = _resolve_multi_word_pending_candidate_selection(state, message)
        literal_codes = {item["word"]: item["code"] for item in literal.args["items"]} if literal and not error else {}
        explicit_subset = _message_authorizes_multiword_subset(
            message,
            selected_words,
            ordered_card_words,
        )
        if not selected:
            if selected_words and all(word in literal_codes for word in selected_words):
                selected = tuple(literal_codes[word] for word in selected_words)
            elif explicit_subset and all(word in recommended_by_word for word in selected_words):
                selected = tuple(recommended_by_word[word] for word in selected_words)
        if not selected or len(selected_words) != len(selected) or len(set(selected_words)) != len(selected_words):
            return ""
        if any(code not in dict(scopes.get(word, ())) for word, code in zip(selected_words, selected)):
            return ""
        if any(code not in message_codes and literal_codes.get(word) != code for word, code in zip(selected_words, selected)):
            if dict(zip(selected_words, selected)) == recommended_by_word:
                return render_understood_pending_answer(replace(request, op="confirm", codes=()), state, prompt, message)
            if not (
                explicit_subset
                and all(
                    code in message_codes
                    or literal_codes.get(word) == code
                    or recommended_by_word.get(word) == code
                    for word, code in zip(selected_words, selected)
                )
            ):
                return ""
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
        choices = ("确认并提交", "加入并提交") if request.submit_after else ("确认", "加入", "添加", "都加")
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
    if op == "recode" and len(words) == len(codes) and words and not request.submit_after and not request.relation:
        command = "修改 " + " ".join(f"{code} {word}" for word, code in zip(words, codes))
        return command if _parse_code_word_changes(command) == tuple(zip(codes, words)) else ""
    if op == "recode" and len(words) > 1 and len(codes) == 2 * len(words) and not request.submit_after and not request.relation:
        commands = [render_understood_command(replace(request, operands=(word,), codes=codes[index * 2:index * 2 + 2]))
                    for index, word in enumerate(words)]
        return "\n".join(commands) if all(commands) else ""
    command = ""
    if op == "add":
        if not words or len(words) != len(codes) or len(set(filter(None, codes))) != len(tuple(filter(None, codes))):
            return ""
        if request.code_selector:
            return f"添加 {words[0]} 的飞键编码" if (
                request.code_selector == "fly_key" and len(words) == 1
                and codes == ("",) and not request.submit_after and not request.relation
            ) else ""
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
        if request.relation == "explicit_order" and len(words) >= 2 and not codes:
            command = "重新排序为 " + " > ".join(words)
        elif len(words) == 2 and len(codes) in {0, 2}:
            entries = [f"{word} {codes[index]}" if codes else word for index, word in enumerate(words)]
            command = f"{entries[0]} 优先于 {entries[1]}"
        elif len(words) == 2 and len(codes) == 1:
            command = f"调整 {codes[0]} {words[0]} 和 {words[1]}"
        elif len(words) > 2 and not codes:
            command = "重新排序 " + " ".join(words)
        elif not words and len(codes) == 1:
            command = "重排 " + codes[0]
        elif not words and len(codes) >= 2 and len(set(codes)) == len(codes):
            command = "重新排序 " + " ".join(codes)
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
    return command if (
        (parsed.op, parsed.operands, parsed.codes, parsed.submit_after)
        == (op, words, codes, request.submit_after)
        and (op != "reorder" or parsed.relation == request.relation)
    ) else ""


def _understood_command_digest(args: Dict) -> str:
    return hashlib.sha256(json.dumps(
        {key: value for key, value in args.items() if key != "digest"},
        sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


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
