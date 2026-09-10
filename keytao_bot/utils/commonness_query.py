"""Read-only commonness queries and parser-closed placement suggestions."""
import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import json
import re
import unicodedata

from .pending_confirmation import advertised_command_suggestions, render_executable_suggestion
from .same_code_reorder import parse_same_code_reorder
from ..harness.authorization_grammar import parse_eviction_modified_add


_PREFIX = re.compile(
    r"(?:请)?(?:给(?:我)?(?:这几个词|这些词|这几词)(?:的)?)?"
    r"(?:词频排序|按常用度排序|哪个更常用|常用度对比)\s*[:：]?\s*(.+)"
)
_SUFFIX = re.compile(r"(.+?)\s*(?:哪个更常用|按常用度排序|词频排序)[?？。！!]*")
_LIST_FREQUENCY_SUFFIX = re.compile(
    r"(.+?)[,，]\s*(?:请)?排列以上词的使用频率[?？。！!]*"
)
_LIST_MUTATION_VERB = r"(?:删除|删掉|移除|添加|加入|修改|写入|提交)"
_NON_LITERAL_LIST_ITEM = re.compile(
    r"(?:不要|别|不必|无需|禁止|他说|她说|引用|转述)"
    rf"|(?:并且|然后|同时|并|再){_LIST_MUTATION_VERB}"
    rf"|{_LIST_MUTATION_VERB}[\u3400-\u9fff]"
    r"|(?:把|将).+(?:删除|删掉|移除|加入|写入|排到|放到|改为|改成)"
)


def parse_commonness_query(message):
    """Accept a bounded explicit comparison as a read-only word list."""
    text = unicodedata.normalize("NFKC", str(message or "")).strip()
    match = _PREFIX.fullmatch(text) or _SUFFIX.fullmatch(text)
    list_suffix = match is None
    if list_suffix:
        match = _LIST_FREQUENCY_SUFFIX.fullmatch(text)
    if match is None:
        return None
    body = match.group(1).strip().rstrip("?？。！!").strip()
    parts = []
    for group in re.split(r"[、,，]+", body):
        tokens = group.split()
        # Explicit whitespace preserves literal words containing the conjunction.
        if len(tokens) == 1 and re.fullmatch(r".+和.+", tokens[0]):
            tokens = tokens[0].split("和")
        parts.extend(token for token in tokens if token != "和")
    if not 2 <= len(parts) <= 12 or len(set(parts)) != len(parts):
        return None
    if any(not re.fullmatch(r"[\u3400-\u9fff]{1,32}", word) for word in parts):
        return None
    # A referring suffix must not consume other clauses as literal operands.
    if list_suffix and any(_NON_LITERAL_LIST_ITEM.match(word) for word in parts):
        return None
    return tuple(parts)


def render_commonness_table(result):
    lines = ["常用度排序（本地语料与词典）", "名次 | 词 | 语料频次 | 词典收录 | 判定"]
    labels = {"ranked": "有数据", "close": "接近（并列参考）", "unknown": "无法判断"}
    for row in result["words"]:
        known = row["known"]
        rank = row["rank"] if known and row["rank"] is not None else "—"
        frequency = row["corpusFrequency"]
        presence = row["dictionaryPresenceCount"]
        verdict = labels.get(row["verdict"], "无法判断") if known else "无数据"
        lines.append(f"{rank} | {row['word']} | {frequency if frequency is not None else '—'} | {presence if presence is not None else '—'} | {verdict}")
    lines.append(result["orderingNote"] + "词频是语料内计数。")
    return "\n".join(lines)


def placement_command(first, second, *, same_code=False, code=""):
    """Round-trip literal operands through the existing S57/S50 grammar."""
    if same_code:
        command = f"把 {first} {code} 排到 {second} {code} 前面"
        parsed = parse_same_code_reorder(command)
        valid = parsed is not None and (
            parsed.first_word, parsed.second_word, parsed.first_code, parsed.second_code
        ) == (first, second, code, code)
    else:
        command = f"把 {first} 放在 {second} 前面"
        parsed = parse_eviction_modified_add(command)
        valid = parsed is not None and (parsed.word, parsed.named_occupant) == (first, second)
    return command if valid else ""


async def _placement_commands(words, ordered_words, read):
    from ..plugins import chat_commands as commands

    data = await read("keytao_lookup_by_words_batch", {"words": list(words)})
    entries = commands._validated_word_lookup_entries(data, words)
    if entries is None:
        return ()
    slots = {word: [entry for entry in entries if entry["word"] == word] for word in words}
    encoded = {}

    async def candidates(word):
        if word not in encoded:
            data = await read("keytao_encode", {"word": word})
            # Ambiguous/unconfirmed readings cannot anchor an operation affordance.
            encoded[word] = () if data.get("semanticPronunciationNeeded") or data.get("standardPronunciationStatus") == "unavailable" else commands._candidate_codes_from_encode(data, word)
        return encoded[word]

    for word in words:
        if not slots[word]:
            await candidates(word)
    advertised = []
    chains = set()
    for index, first in enumerate(ordered_words):
        for second in ordered_words[index + 1:]:
            shared = {(row["type"], row["code"]) for row in slots[first]} & {
                (row["type"], row["code"]) for row in slots[second]
            }
            if shared:
                for phrase_type, code in sorted(shared):
                    if sum(shared_code == code for _kind, shared_code in shared) != 1:
                        continue
                    if (phrase_type, code) not in chains:
                        command = placement_command(first, second, same_code=True, code=code)
                        if command:
                            advertised.append(command)
                            chains.add((phrase_type, code))
                continue
            # A relative-placement target must name one existing location.
            if len(slots[second]) != 1:
                continue
            first_codes, second_codes = await candidates(first), await candidates(second)
            if not first_codes or not second_codes:
                continue
            second_entry = slots[second][0]
            if second_entry["code"] not in first_codes:
                continue
            first_entries = slots[first] or [{
                "word": first, "code": code, "type": second_entry["type"]
            } for code in first_codes]
            for first_entry in first_entries:
                if first_entry["type"] != second_entry["type"]:
                    continue
                root = commands._shared_candidate_chain_root(
                    [first_entry, second_entry], {first: first_codes, second: second_codes},
                )
                identity = (second_entry["type"], root)
                if root and identity not in chains:
                    command = placement_command(first, second)
                    if command:
                        advertised.append(command)
                        chains.add(identity)
                    break
    return tuple(advertised)


@dataclass(frozen=True)
class CommonnessDelivery:
    owner: object
    words: tuple
    text: str
    commands: tuple


_delivery = ContextVar("commonness_delivery", default=None)


def reset_commonness_delivery():
    _delivery.set(None)


def matches_commonness_delivery(text, owner, message):
    """Trust only this turn's exact renderer output, never model-authored prose."""
    record = _delivery.get()
    return bool(record is not None and record.owner == owner
                and record.words == parse_commonness_query(message)
                and record.text == text
                and advertised_command_suggestions(text) == record.commands)


async def commonness_query_reply(message, platform, user_id, owner):
    from .word_commonness import lookup_word_commonness
    from ..plugins import chat_commands as commands

    words = parse_commonness_query(message)
    if words is None:
        return None
    result = lookup_word_commonness(list(words))
    response = render_commonness_table(result)

    async def read(name, args):
        try:
            value = json.loads(await commands.call_tool_function(name, args, platform, user_id))
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            return {}

    try:
        suggestions = () if result["ordering"] == "conflicting_evidence" else await asyncio.wait_for(
            _placement_commands(words, [row["word"] for row in result["words"]], read), timeout=5.0,
        )
    except Exception:
        suggestions = ()
    if suggestions:
        proposed = response + "\n可发送以下命令生成调整计划：\n" + "\n".join(
            render_executable_suggestion(command) for command in suggestions
        )
        if advertised_command_suggestions(proposed) == suggestions:
            response = proposed
        else:
            suggestions = ()
    _delivery.set(CommonnessDelivery(owner, words, response, suggestions))
    return response
