"""Validate user-supplied codes against a persisted reviewed pronunciation."""

from dataclasses import dataclass
import re
from types import SimpleNamespace
import unicodedata

from .candidate_inventory import select_candidate_inventory
from .keytao_encoding import build_phrase_code_chain, pinyin_to_phonetic_code


@dataclass(frozen=True)
class ExplicitCodeRequest:
    code: str
    submit_after: bool = False
    pinyin: str = ""


@dataclass(frozen=True)
class ExplicitEntryCodeRequest:
    word: str
    code: str
    submit_after: bool = False
    pinyin: str = ""


def parse_explicit_entry_code_request(message: str):
    """Bind a complete add-code command to its explicitly named item."""
    source = unicodedata.normalize("NFKC", message).strip().rstrip("。.!！").strip()
    if re.match(r"加入\s*编码", source):
        return None
    word = r'[\u3400-\u9fff\U00020000-\U0003134f]{1,32}'
    code = r'[^\s，,、；;。？?：:「」『』“”\"\']+?'

    def operand(pattern):
        return rf'(?:{pattern}|"\s*{pattern}\s*"|“\s*{pattern}\s*”|「\s*{pattern}\s*」|『\s*{pattern}\s*』)'

    word_operand = operand(word)
    code_operand = operand(code)
    action = r'(?:加入草稿|加到草稿|写入草稿|加入|添加到草稿|添加草稿|添加)(?P<submit>并提交)?'
    separator = r'(?:\s*[,，、；;]\s*|\s+)'
    reading = r'(?:(?P<pinyin>[A-Za-z\u00c0-\u024f\u1e00-\u1eff0-5]+(?:\s+[A-Za-z\u00c0-\u024f\u1e00-\u1eff0-5]+)*)\s*:\s*)?'
    pair = rf'(?P<word>{word_operand})\s+{reading}(?P<code>{code_operand})'
    combined = re.fullmatch(pair + separator + action, source)
    if combined is None:
        combined = re.fullmatch(action + separator + pair, source)
    if combined is not None:
        from ..harness.authorization_grammar import (
            _NEGATIVE_MODAL_RE,
            _POSITIONAL_REPORTED_CONTEXT_RE,
            looks_like_lexical_review_target,
        )

        selected_word = combined.group("word").strip(' "“”「」『』')
        if (
            not looks_like_lexical_review_target(selected_word)
            or re.match(r"^(?:请)?(?:先|暂时|暂)?" + _NEGATIVE_MODAL_RE.pattern, selected_word)
            or _POSITIONAL_REPORTED_CONTEXT_RE.match(selected_word)
        ):
            return None
        return ExplicitEntryCodeRequest(
            selected_word,
            combined.group("code").strip(' "“”「」『』'),
            bool(combined.group("submit")),
            str(combined.group("pinyin") or "").strip(),
        )
    patterns = (
        rf'(?:添加|加入)\s*(?P<type>单字|词组|词条)?\s*(?P<word>{word_operand})'
        rf'(?:\s*[,，]?\s*编码\s*(?:为|是)?\s*[:：]?\s*|\s+)(?P<code>{code_operand})',
        rf'给\s*(?P<word>{word_operand})\s*加(?:一个|个)?\s*(?:编码|码)\s*(?P<code>{code_operand})',
        rf'(?P<word>{word_operand})\s*也放到\s*(?P<code>{code_operand})',
    )
    for pattern in patterns:
        match = re.fullmatch(pattern + r'\s*(?P<submit>并提交)?', source)
        if match is None:
            continue
        selected_word = match.group("word").strip(' "“”「」『』')
        selected_code = match.group("code").strip(' "“”「」『』')
        if match.groupdict().get("type") == "单字" and len(selected_word) != 1:
            return None
        return ExplicitEntryCodeRequest(selected_word, selected_code, bool(match.group("submit")))
    return None


@dataclass(frozen=True)
class ExplicitCodeValidation:
    code: str
    pinyin: str = ""
    reason: str = ""
    manual_reason: str = ""
    known_candidate: bool = False

    @property
    def valid(self) -> bool:
        return bool((self.pinyin or self.known_candidate) and not self.reason)


def parse_explicit_code_request(message: str, word: str):
    """Consume one direct command; never discard an extra target or clause."""
    entry = parse_explicit_entry_code_request(message)
    if entry is not None:
        return ExplicitCodeRequest(entry.code, entry.submit_after, entry.pinyin) if entry.word == word else None
    match = re.fullmatch(
        rf"(?:加入\s*编码\s*|加入\s+|添加\s+{re.escape(word)}\s+|用\s+)"
        r"(?P<code>[^\s，,；;。？?「」“”]+?)(?:\s*(?P<submit>并提交))?",
        message.strip(),
    )
    if match is None:
        return None
    if match.group("code").isdigit() and not re.match(
        rf"(?:加入\s*编码|添加\s+{re.escape(word)}\s+)", message.strip(),
    ):
        return None
    return ExplicitCodeRequest(match.group("code"), bool(match.group("submit")))


def readings_match(left: str, right: str) -> bool:
    """Compare syllable sequences without letting tone marks change identity."""
    def normalized(value):
        value = value.lower().replace("u:", "v").translate(str.maketrans("üǖǘǚǜ", "vvvvv"))
        return tuple(re.sub(r"[1-5]$", "", "".join(
            char for char in unicodedata.normalize("NFD", syllable)
            if unicodedata.category(char) != "Mn"
        )) for syllable in value.split())
    return bool(left.strip() and right.strip() and normalized(left) == normalized(right))


def select_explicit_candidate_inventory(review, code: str, pinyin: str = ""):
    """Resolve only a declared reading ambiguity using each trusted group's codes."""
    matches = []
    failures = []
    for group in review.get("pronunciations") or []:
        if not isinstance(group, dict):
            continue
        reading = str(group.get("pinyin") or "").strip()
        if not reading or (pinyin and not readings_match(pinyin, reading)):
            continue
        inventory = select_candidate_inventory({"pronunciations": [group]})
        if inventory is None:
            continue
        state = SimpleNamespace(
            word=review["word"], phrase_type=review.get("type") or "Phrase",
            candidates=list(inventory.candidates), server_candidates=list(inventory.candidates),
            pronunciation_codes=dict(inventory.readings),
        )
        validation = validate_explicit_code(state, code, pinyin)
        if validation.valid:
            matches.append(inventory)
        else:
            failures.append(validation.reason)
    if matches:
        return matches[0], ""
    if failures:
        return None, failures[0]
    return None, (f"指定读音 {pinyin} 不在可核验的候选读音中" if pinyin else "当前缺少可核验的读音候选")


def validate_explicit_code(state, code: str, pinyin: str = "") -> ExplicitCodeValidation:
    """A shape suffix absent from the review is unknown, not an invalid code."""
    fail = lambda reason: ExplicitCodeValidation(code, reason=reason)
    if not state.server_candidates or state.server_candidates != state.candidates:
        return fail("当前没有已核验的候选记录，请先查询该词")
    if re.fullmatch(r"[a-z]+", code) is None:
        return fail(f"编码 {code} 只能包含小写字母 a-z")
    if len(code) > 6:
        return fail(f"编码 {code} 有 {len(code)} 位；单字和词组最多 6 位")
    if state.phrase_type not in {"Single", "Phrase"}:
        return fail("当前词条类型不支持按读音核验指定编码")
    inventory = dict(state.server_candidates)
    if pinyin and not any(readings_match(pinyin, reading) for reading in state.pronunciation_codes.values()):
        return fail(f"指定读音 {pinyin} 不在可核验的候选读音中")
    if code in inventory:
        # Existing inventory membership remains its own reviewed capability;
        # legacy records without a reading cannot authorize any new suffix.
        reading = state.pronunciation_codes.get(code) or ""
        if not pinyin or readings_match(pinyin, reading):
            return ExplicitCodeValidation(code, pinyin=reading, known_candidate=True)
    readings = list(dict.fromkeys(
        str(state.pronunciation_codes.get(candidate) or "").strip()
        for candidate in inventory
        if str(state.pronunciation_codes.get(candidate) or "").strip()
        and (not pinyin or readings_match(pinyin, state.pronunciation_codes[candidate]))
    ))
    matches = []
    bases = []
    for pinyin in readings:
        syllables = pinyin.split()
        if len(syllables) != len(state.word):
            continue
        phonetics = [pinyin_to_phonetic_code(syllable) for syllable in syllables]
        if not all(phonetics):
            continue
        chain = build_phrase_code_chain(
            [{"char": char, "phoneticCode": phonetic, "shapeCode": ""} for char, phonetic in zip(state.word, phonetics)],
            phonetics,
        )
        if not chain:
            continue
        base_length = len(chain[0])
        reviewed_bases = list(dict.fromkeys(
            candidate[:base_length] for candidate in inventory
            if state.pronunciation_codes.get(candidate) == pinyin and len(candidate) >= base_length
        ))
        bases.extend(reviewed_bases)
        matches.extend((pinyin, base) for base in reviewed_bases if code.startswith(base))
    if not matches:
        return fail(
            f"编码 {code} 的音码前缀不符；已审读音对应 {' / '.join(bases)}"
            if bases else "当前记录缺少可核验的读音，请先重新查询"
        )
    pinyin, base = matches[0]
    suffix = code[len(base):]
    return ExplicitCodeValidation(
        code, pinyin=pinyin,
        manual_reason=f"形码 {suffix} 未能核验，需管理员复核" if suffix else "指定音码需管理员复核",
    )
