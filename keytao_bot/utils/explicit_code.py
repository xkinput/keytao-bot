"""Validate user-supplied codes against a persisted reviewed pronunciation."""

from dataclasses import dataclass
import re
import unicodedata

from .keytao_encoding import build_phrase_code_chain, pinyin_to_phonetic_code


@dataclass(frozen=True)
class ExplicitCodeRequest:
    code: str
    submit_after: bool = False


@dataclass(frozen=True)
class ExplicitEntryCodeRequest:
    word: str
    code: str
    submit_after: bool = False


def parse_explicit_entry_code_request(message: str):
    """Bind a complete add-code command to its explicitly named item."""
    source = unicodedata.normalize("NFKC", message).strip().rstrip("。.!！").strip()
    if re.match(r"加入\s*编码", source):
        return None
    word = r'[\u3400-\u9fff\U00020000-\U0003134f]{1,32}'
    code = r'[^\s，,；;。？?「」『』“”\"\']+?'

    def operand(pattern):
        return rf'(?:{pattern}|"\s*{pattern}\s*"|“\s*{pattern}\s*”|「\s*{pattern}\s*」|『\s*{pattern}\s*』)'

    word_operand = operand(word)
    code_operand = operand(code)
    trailing_action = re.fullmatch(
        rf'(?P<word>{word_operand})\s+(?P<code>{code_operand})'
        r'(?:\s*[,，]\s*|\s+)(?:加入|添加)(?:(?:到)?草稿)?\s*(?P<submit>并提交)?',
        source,
    )
    if trailing_action is not None:
        from ..harness.authorization_grammar import (
            _NEGATIVE_MODAL_RE,
            _POSITIONAL_REPORTED_CONTEXT_RE,
            looks_like_lexical_review_target,
        )

        selected_word = trailing_action.group("word").strip(' "“”「」『』')
        if (
            not looks_like_lexical_review_target(selected_word)
            or re.match(r"^(?:请)?(?:先|暂时|暂)?" + _NEGATIVE_MODAL_RE.pattern, selected_word)
            or _POSITIONAL_REPORTED_CONTEXT_RE.match(selected_word)
        ):
            return None
        return ExplicitEntryCodeRequest(
            selected_word,
            trailing_action.group("code").strip(' "“”「」『』'),
            bool(trailing_action.group("submit")),
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
        return ExplicitCodeRequest(entry.code, entry.submit_after) if entry.word == word else None
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


def validate_explicit_code(state, code: str) -> ExplicitCodeValidation:
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
    if code in inventory:
        # Existing inventory membership remains its own reviewed capability;
        # legacy records without a reading cannot authorize any new suffix.
        return ExplicitCodeValidation(code, pinyin=state.pronunciation_codes.get(code) or "", known_candidate=True)
    readings = list(dict.fromkeys(
        str(state.pronunciation_codes.get(candidate) or "").strip()
        for candidate in inventory
        if str(state.pronunciation_codes.get(candidate) or "").strip()
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
    if len(matches) != 1:
        return fail("该编码对应多个待定读音，请先指定读音")
    pinyin, base = matches[0]
    suffix = code[len(base):]
    return ExplicitCodeValidation(
        code, pinyin=pinyin,
        manual_reason=f"形码 {suffix} 未能核验，需管理员复核" if suffix else "指定音码需管理员复核",
    )
