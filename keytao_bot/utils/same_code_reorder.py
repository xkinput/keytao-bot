"""Resolve explicit relative order into the existing sealed weight planner."""

from dataclasses import dataclass
import json
import re
import unicodedata

from ..harness.authorization_grammar import (
    _DATA_CONTEXT_RE,
    _NEGATIVE_MODAL_RE,
    _NEGATED_NON_POSITIONAL_MUTATION_RE,
    _PHRASE_TYPE_BASE_WEIGHTS,
    _POSITIONAL_REPORTED_CONTEXT_RE,
    _has_standalone_negation_before_mutation,
    trusted_mutation_source,
)


@dataclass(frozen=True)
class SameCodeReorder:
    first_word: str
    second_word: str
    first_code: str = ""
    second_code: str = ""
    swap: bool = False


_ENTRY = r"[\u3400-\u9fff]{1,32}(?:\s+[a-z]{1,12})?"
_OPERAND = (
    rf'(?:「\s*{_ENTRY}\s*」|『\s*{_ENTRY}\s*』|“\s*{_ENTRY}\s*”|'
    rf'"\s*{_ENTRY}\s*"|\s*{_ENTRY})'
)
_FIRST = rf"(?P<first>{_OPERAND})"
_SECOND = rf"(?P<second>{_OPERAND})"
_PATTERNS = (
    (False, re.compile(rf"(?:请\s*)?提高\s*{_FIRST}\s*的优先级\s*[,，]?\s*使之位于\s*{_SECOND}\s*之前")),
    (False, re.compile(rf"(?:请\s*)?把\s*{_FIRST}\s*(?:排到|放到)\s*{_SECOND}\s*(?:前面|之前)")),
    (False, re.compile(rf"{_FIRST}\s*优先于\s*{_SECOND}")),
    (True, re.compile(rf"{_FIRST}\s*(?:和|与)\s*{_SECOND}\s*(?:调换顺序|对换|交换顺序|互换顺序)")),
    (True, re.compile(rf"(?:请\s*)?(?:对换|调换|互换|交换)\s*{_FIRST}\s*(?:和|与)\s*{_SECOND}\s*的(?:顺序|优先级)")),
)


def parse_same_code_reorder(message: str):
    """Accept only a complete positive request with two literal operands."""
    source = unicodedata.normalize("NFKC", trusted_mutation_source(message)).strip()
    if (
        not source
        or re.search(r"[?？\n\r]", source)
        or _NEGATIVE_MODAL_RE.search(source)
        or _NEGATED_NON_POSITIONAL_MUTATION_RE.search(source)
        or _has_standalone_negation_before_mutation(source)
        or _POSITIONAL_REPORTED_CONTEXT_RE.search(source)
        or _DATA_CONTEXT_RE.search(source)
    ):
        return None
    source = source.rstrip("。.!！").strip()
    for swap, pattern in _PATTERNS:
        match = pattern.fullmatch(source)
        if match is None:
            continue
        operands = []
        for key in ("first", "second"):
            text = match.group(key).strip().strip('「」『』“”"').strip()
            parts = text.split()
            operands.append((parts[0], parts[1] if len(parts) == 2 else ""))
        if operands[0][0] == operands[1][0]:
            return None
        return SameCodeReorder(
            operands[0][0], operands[1][0], operands[0][1], operands[1][1], swap,
        )
    return None


def _advisory(first_word: str, second_word: str) -> str:
    """Use the existing local comparator; missing evidence never delays a write."""
    from . import keytao_review

    first = keytao_review._query_commonness_reference(first_word)
    second = keytao_review._query_commonness_reference(second_word)
    if not (first.get("available") and second.get("available")):
        return "常用度提示：现有证据不足，仍按你的要求执行"
    comparison = keytao_review._compare_reference_commonness(first_word, second_word, first, second)
    keytao_review.record_commonness_evidence(comparison)
    if comparison.get("verdict") == "behind_more_common":
        return f"常用度提示：{second_word} 更常用，仍按你的要求执行"
    if comparison.get("verdict") == "front_more_common":
        return f"常用度提示：{first_word} 更常用，按你的要求执行"
    return "常用度提示：现有证据未分出高低，仍按你的要求执行"


async def try_handle_same_code_reorder(
    message_text: str,
    platform: str,
    user_id: str,
    space_key=None,
    owner_label: str = "",
):
    """Read exact typed entries, preserve the chain, and stage one real ticket."""
    command = parse_same_code_reorder(message_text)
    if command is None:
        return None
    from ..plugins import chat_commands as commands

    commands.set_turn_flow("draft-op")
    words = (command.first_word, command.second_word)

    async def read(name, args):
        try:
            return json.loads(await commands.call_tool_function(name, args, platform, user_id))
        except (TypeError, ValueError):
            return {}

    lookup = await read("keytao_lookup_by_words_batch", {"words": list(words)})
    draft = await read("keytao_list_draft_items", {})
    entries = commands._validated_word_lookup_entries(lookup, words)
    snapshot = commands._validated_draft_reorder_snapshot(draft)
    if entries is None or snapshot is None:
        return "无法读取完整的词库和草稿记录，本次未生成调序计划。"
    merged = commands._merge_live_and_draft_reorder_entries(entries, snapshot)
    if merged is None:
        return "当前词库与草稿记录无法唯一合并，本次未生成调序计划。"
    slots = []
    for word, explicit_code in zip(words, (command.first_code, command.second_code)):
        slots.append({
            (entry["type"], entry["code"])
            for entry in merged
            if entry["word"] == word and (not explicit_code or entry["code"] == explicit_code)
        })
    shared = slots[0] & slots[1]
    if len(shared) != 1:
        if not shared and not command.first_code and not command.second_code:
            return None
        return "这两个词条未能锁定唯一的同码同类型记录；请在调序指令中写明双方的现有编码。"
    phrase_type, code = next(iter(shared))
    chain_lookup = await read("keytao_lookup_by_code", {"code": code})
    chain_entries = commands._validated_code_chain_entries(chain_lookup, code)
    if chain_entries is None:
        return f"编码 {code} 的完整同码链无法核验，本次未生成调序计划。"
    chain = commands._merge_live_and_draft_reorder_entries(chain_entries, snapshot)
    if chain is None:
        return f"编码 {code} 的词库与草稿记录无法唯一合并，本次未生成调序计划。"
    chain = [entry for entry in chain if entry["code"] == code and entry["type"] == phrase_type]
    chain.sort(key=lambda entry: (entry["weight"], entry["word"]))
    ordered = [entry["word"] for entry in chain]
    if len(set(ordered)) != len(ordered) or not set(words).issubset(ordered):
        return f"编码 {code} 的目标词条已变化，本次未生成调序计划。"
    first_index, second_index = (ordered.index(word) for word in words)
    if not command.swap and chain[first_index]["weight"] < chain[second_index]["weight"]:
        return f"「{words[0]}」已在「{words[1]}」之前，当前无需调整。"
    original = list(ordered)
    if command.swap:
        ordered[first_index], ordered[second_index] = ordered[second_index], ordered[first_index]
    else:
        ordered.remove(words[0])
        ordered.insert(ordered.index(words[1]), words[0])
    weights = [entry["weight"] for entry in chain]
    # Preserve existing distinct slots. Tied slots use the standing typed base
    # plus index through the shared planner, yielding a strict ascending order.
    explicit_weights = weights if len(set(weights)) == len(weights) else None
    target_weights = explicit_weights or [
        _PHRASE_TYPE_BASE_WEIGHTS[phrase_type] + index for index in range(len(ordered))
    ]
    before = {entry["word"]: entry["weight"] for entry in chain}
    changes = "、".join(
        f"{word} {before[word]}→{weight}"
        for word, weight in zip(ordered, target_weights) if before[word] != weight
    )
    front, behind = (words[1], words[0]) if command.swap and first_index < second_index else words
    evidence = [f"将把 {front} 排到 {behind} 之前：{changes}", _advisory(front, behind)]
    return await commands._execute_shift_to_code(
        ordered[0], code, platform, user_id, space_key, owner_label,
        ordered_words=ordered, listed_words=original,
        expected_codes=[code] * len(ordered), expected_weights=explicit_weights,
        evidence_lines=evidence,
    )
