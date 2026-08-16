"""Shared reply forms and copy for deterministic pending confirmations."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


PENDING_CONFIRM_ADVERTISED_FORMS = ("确认", "执行")
PENDING_CONFIRM_ASSENT_TEXTS = frozenset({
    *PENDING_CONFIRM_ADVERTISED_FORMS,
    "确定",
    "好的",
    "好",
    "是",
    "对",
    "可以",
    "行",
    "同意",
    "就这样",
    "按这个",
    "执行吧",
})

PENDING_BATCH_ADD_ADVERTISED_FORMS = ("加入", "都加", "添加")
PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS = (
    "加入并提交",
    "都加并提交",
    "添加并提交",
)
PENDING_BATCH_ADD_ASSENT_TEXTS = frozenset({
    *PENDING_BATCH_ADD_ADVERTISED_FORMS,
    "加",
    "确认加入",
    "确认添加",
    "继续加入",
    "继续添加",
    "全部加",
})
PENDING_BATCH_ADD_AND_SUBMIT_ASSENT_TEXTS = frozenset({
    *PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS,
    "加并提交",
    "新增并提交",
})
PENDING_BATCH_ASSENT_TEXTS = frozenset({
    *PENDING_BATCH_ADD_ASSENT_TEXTS,
    *PENDING_BATCH_ADD_AND_SUBMIT_ASSENT_TEXTS,
})
PENDING_ASSENT_TEXTS = frozenset({
    *PENDING_CONFIRM_ASSENT_TEXTS,
    *PENDING_BATCH_ASSENT_TEXTS,
})
PENDING_BATCH_CONFIRMATION_COPY_TOKEN = "{{PENDING_BATCH_CONFIRMATION_COPY}}"


def advertised_batch_binding_pairs(text: str) -> tuple[tuple[str, str], ...]:
    """Read exact visible word/code bindings from supported batch renderings."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    positioned_pairs: list[tuple[int, str, str]] = []
    patterns = (
        re.compile(
            r"(?m)^\s*[-•]\s*「(?P<word>[^」\n]{1,128})」\s*"
            r"(?:→|->)\s*(?P<code>[a-z]{1,12})"
            r"(?:\s*[（(][^）)\n]{0,128}[）)])?\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?m)^\s*是否以编码\s+(?P<code>[a-z]{1,12})\s+"
            r"将「(?P<word>[^」\n]{1,128})」加入草稿[？?]?\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:^|[\n:：;；…]|\.{3,})\s*\d{1,3}[.)、]\s*"
            r"[「“『]?(?P<word>[\u3400-\u9fffA-Za-z0-9_-]{1,128})[」”』]?\s*"
            r"(?:→|->)\s*(?P<code>[a-z]{1,12})"
            r"(?:\s*[（(][^）)\n]{0,64}[）)])?",
            re.IGNORECASE,
        ),
    )
    positioned_pairs.extend(
        (match.start(), match.group("word"), match.group("code"))
        for pattern in patterns
        for match in pattern.finditer(normalized)
    )

    numbered_heading = re.compile(
        r"(?m)^\s*\d{1,3}[.)、]\s*"
        r"[「“『]?(?P<word>[\u3400-\u9fffA-Za-z0-9_-]{1,128})[」”』]?\s*$"
    )
    recommended_candidate = re.compile(
        r"(?:^|[：:｜|])\s*\d{1,3}[.)、]\s*"
        r"(?P<code>[a-z]{1,12})\s*(?:—|–|-)\s*"
        r"[^\n｜|]{0,64}[（(]\s*(?:✅\s*)?推荐\s*[）)]",
        re.MULTILINE | re.IGNORECASE,
    )
    headings = list(numbered_heading.finditer(normalized))
    for index, heading in enumerate(headings):
        block_end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(normalized)
        )
        block = normalized[heading.end():block_end]
        if "候选" not in block:
            continue
        candidate = recommended_candidate.search(block)
        if candidate is None:
            continue
        positioned_pairs.append(
            (heading.start(), heading.group("word"), candidate.group("code"))
        )

    for _position, raw_word, raw_code in sorted(positioned_pairs):
        pair = (
            raw_word.strip(),
            raw_code.strip().lower(),
        )
        if pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
    return tuple(pairs)


@dataclass(frozen=True)
class PendingCandidateSelection:
    """A closed multi-slot selection parsed without model interpretation."""

    indices: tuple[int, ...] = ()
    codes: tuple[str, ...] = ()
    submit_after: bool = False


@dataclass(frozen=True)
class AdvertisedSetReference:
    """Closed current-message grammar for snapshot-minus-exclusions."""

    matched: bool = False
    exclusions: tuple[str, ...] = ()
    submit_after: bool = False
    expected_count: int | None = None
    advertised_verb: str = ""


_SET_REFERENCE_REMAINDER_ACTION_RE = re.compile(
    r"(?:其他|其余|剩下的?)(?:一些|的)?(?:都|全部|也)?"
    r"(?:可以|可)?(?:加入|添加|加)(?:到|进|入)?(?:草稿)?"
)
_SET_REFERENCE_UNSAFE_FRAME_RE = re.compile(
    r"(?:他说|她说|他们说|有人说|引用|转述|复述|解释|举例|例子|"
    r"假设|如果|不要执行|别执行|不执行|不是让你|并非让你)"
)
_SET_REFERENCE_EXCLUSION_RE = re.compile(
    r"(?P<word>[\u3400-\u9fffA-Za-z0-9_-]{1,32}?)\s*"
    r"(?:也\s*)?(?:(?:先|暂时)\s*)?(?:不要|不加|别加)"
)
_SET_REFERENCE_EXCEPT_RE = re.compile(
    r"除了(?P<body>.+?)(?=(?:其他|其余|剩下的?))"
)
_SET_REFERENCE_SKIP_RE = re.compile(
    r"(?:跳过|略过)(?P<body>.+?)(?=(?:其他|其余|剩下的?))"
)
_SET_REFERENCE_WORD_RE = re.compile(r"^[\u3400-\u9fffA-Za-z0-9_-]{1,32}$")
_SET_REFERENCE_SEPARATORS_RE = re.compile(r"[\s、,，;；]+|和|与|及")
_BATCH_ADVERTISED_FORMS = tuple(sorted(
    (
        *PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS,
        *PENDING_BATCH_ADD_ADVERTISED_FORMS,
    ),
    key=lambda value: (-len(value), value),
))
_BATCH_ADVERTISED_VERB_RE = re.compile(
    r"^\s*(?P<verb>"
    + "|".join(re.escape(value) for value in _BATCH_ADVERTISED_FORMS)
    + r")(?P<tail>.*)$",
    re.DOTALL,
)
_DEICTIC_BATCH_COMMAND_RE = re.compile(
    r"^(?:将|把)这\s*(?P<count>[1-9]\d{0,2})\s*个词"
    r"(?:都|全部)?(?:加入|添加|加到|放入|写入)(?:到|进|入)?草稿"
    r"(?P<submit>(?:并|然后|再)提交)?$"
)


def _split_set_reference_words(body: str) -> tuple[str, ...]:
    tokens = tuple(
        token.strip()
        for token in _SET_REFERENCE_SEPARATORS_RE.split(str(body or ""))
        if token.strip()
    )
    if not tokens or any(_SET_REFERENCE_WORD_RE.fullmatch(token) is None for token in tokens):
        return ()
    return tuple(dict.fromkeys(tokens))


def advertised_batch_assent_verb(text: str) -> str:
    """Return the exact advertised batch verb at the start of current text."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).strip()
    match = _BATCH_ADVERTISED_VERB_RE.match(normalized)
    return match.group("verb") if match is not None else ""


_PLACEHOLDER_OPERAND_RE = re.compile(
    r"(?<![\u3400-\u9fffA-Za-z0-9_])"
    r"(?:词条|编码|x{2,}|…|\.{3,})"
    r"(?![\u3400-\u9fffA-Za-z0-9_])",
    re.IGNORECASE,
)


def advertised_command_has_placeholder(command: str) -> bool:
    """Reject operand-shaped placeholders from copyable user commands."""
    normalized = unicodedata.normalize("NFKC", str(command or ""))
    return _PLACEHOLDER_OPERAND_RE.search(normalized) is not None


def render_executable_suggestion(
    command: str,
    *,
    words: tuple[str, ...] = (),
) -> str:
    """Render one full copyable line accepted by the command-envelope parser."""
    clean_command = str(command or "").strip()
    if not clean_command or advertised_command_has_placeholder(clean_command):
        return ""
    clean_words = tuple(dict.fromkeys(
        str(word or "").strip()
        for word in words
        if str(word or "").strip()
    ))
    suffix = f"（{'、'.join(clean_words)}）" if clean_words else ""
    if "」" not in clean_command:
        wrapped = f"「{clean_command}」"
    elif "”" not in clean_command:
        wrapped = f"“{clean_command}”"
    elif "』" not in clean_command:
        wrapped = f"『{clean_command}』"
    else:
        return ""
    return f"- {wrapped}{suffix}"


def render_remediation_reply(
    reason: str,
    *,
    command: str = "",
    words: tuple[str, ...] = (),
) -> str:
    """Render a refusal with one executable command, or state that none exists."""
    clean_reason = str(reason or "").strip().rstrip("；;。")
    suggestion = render_executable_suggestion(command, words=words)
    if suggestion:
        return f"{clean_reason}。\n可执行命令：\n{suggestion}"
    return f"{clean_reason}。当前没有可安全执行的后续命令。"


def _mask_set_reference_quotes(text: str) -> str:
    source = list(unicodedata.normalize("NFKC", str(text or "")))
    normalized = "".join(source)
    for pattern in (
        re.compile(r"「[^」]*」|『[^』]*』|“[^”]*”|‘[^’]*’"),
        re.compile(r'"[^"\n]*"|\'[^\'\n]*\'|`[^`\n]*`'),
    ):
        for match in pattern.finditer(normalized):
            source[match.start():match.end()] = " " * (match.end() - match.start())
    return "".join(source)


def parse_advertised_set_reference(text: str) -> AdvertisedSetReference:
    """Parse the shared closed grammar for server-advertised word sets."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    source = _mask_set_reference_quotes(normalized)
    if (
        not source.strip()
        or source != normalized
        or "?" in source
        or "？" in source
        or _SET_REFERENCE_UNSAFE_FRAME_RE.search(source)
    ):
        return AdvertisedSetReference()

    compact = re.sub(r"\s+", "", source).strip("，,。.!！;；:：")
    deictic = _DEICTIC_BATCH_COMMAND_RE.fullmatch(compact)
    if deictic is not None:
        return AdvertisedSetReference(
            matched=True,
            submit_after=bool(deictic.group("submit")),
            expected_count=int(deictic.group("count")),
            advertised_verb=deictic.group(0),
        )

    assent = _BATCH_ADVERTISED_VERB_RE.match(source.strip())
    if assent is not None:
        verb = assent.group("verb")
        tail = assent.group("tail").strip().lstrip("，,。.!！;；:：").strip()
        if not tail:
            return AdvertisedSetReference(
                matched=True,
                submit_after=verb in PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS,
                advertised_verb=verb,
            )
        body = ""
        for pattern in (
            re.compile(r"^(?:跳过|略过)(?P<body>.+)$"),
            re.compile(r"^(?:除了|除去)(?P<body>.+)$"),
            re.compile(
                r"^(?:(?:但|但是|不过)\s*)?(?P<body>.+?)\s*"
                r"(?:也\s*)?(?:(?:先|暂时)\s*)?(?:不要|不加|别加)$"
            ),
        ):
            match = pattern.fullmatch(tail)
            if match is not None:
                body = match.group("body")
                break
        exclusions = _split_set_reference_words(body)
        if not exclusions:
            return AdvertisedSetReference()
        return AdvertisedSetReference(
            matched=True,
            exclusions=exclusions,
            submit_after=verb in PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS,
            advertised_verb=verb,
        )

    if _SET_REFERENCE_REMAINDER_ACTION_RE.search(source) is None:
        return AdvertisedSetReference()
    exclusions: list[str] = []
    for pattern in (_SET_REFERENCE_EXCEPT_RE, _SET_REFERENCE_SKIP_RE):
        for match in pattern.finditer(source):
            exclusions.extend(_split_set_reference_words(match.group("body")))
    exclusions.extend(
        match.group("word")
        for match in _SET_REFERENCE_EXCLUSION_RE.finditer(source)
    )
    if re.search(r"(?:除了|跳过|略过|不要|不加|别加)", source) and not exclusions:
        return AdvertisedSetReference()
    return AdvertisedSetReference(
        matched=True,
        exclusions=tuple(dict.fromkeys(exclusions)),
    )


def _selection_action_forms(forms: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(forms, key=lambda value: (-len(value), value)))


_SELECTION_ADD_FORMS = _selection_action_forms(PENDING_BATCH_ADD_ASSENT_TEXTS)
_SELECTION_ADD_SUBMIT_FORMS = _selection_action_forms(
    PENDING_BATCH_ADD_AND_SUBMIT_ASSENT_TEXTS
)
_SELECTION_SEPARATORS = r"(?:\s+|[、,]|和)+"
_MULTI_NUMBER_SELECTION_RE = re.compile(
    rf"[1-9]\d{{0,2}}(?:{_SELECTION_SEPARATORS}[1-9]\d{{0,2}})+"
)
_MULTI_CODE_SELECTION_RE = re.compile(
    rf"[a-z]{{2,12}}(?:{_SELECTION_SEPARATORS}[a-z]{{2,12}})+",
    re.IGNORECASE,
)


def _strip_selection_action(text: str) -> tuple[str, bool]:
    for action in _SELECTION_ADD_SUBMIT_FORMS:
        if text.startswith(action):
            return text[len(action):].strip(), True
        if text.endswith(action):
            return text[:-len(action)].strip(), True
    for action in _SELECTION_ADD_FORMS:
        if text.startswith(action):
            selection = text[len(action):].strip()
            if selection.endswith("并提交"):
                return selection[:-3].strip(), True
            return selection, False
        if text.endswith(action):
            return text[:-len(action)].strip(), False
    return text, False


def parse_pending_candidate_selection(text: str) -> PendingCandidateSelection | None:
    """Parse advertised multi-number or multi-code replies as a closed grammar."""
    source = unicodedata.normalize("NFKC", str(text or "")).strip()
    if (
        not source
        or re.search(r"[?？\"'“”‘’「」『』]", source)
        or re.search(r"(?:不要|别|取消|删除|移除|解释|复述|他说)", source)
    ):
        return None
    selector, submit_after = _strip_selection_action(source)
    selector = selector.strip()
    if _MULTI_NUMBER_SELECTION_RE.fullmatch(selector):
        values = tuple(
            int(token)
            for token in re.split(_SELECTION_SEPARATORS, selector)
        )
        return PendingCandidateSelection(
            indices=values,
            submit_after=submit_after,
        )
    if _MULTI_CODE_SELECTION_RE.fullmatch(selector):
        values = tuple(
            token.lower()
            for token in re.split(_SELECTION_SEPARATORS, selector)
        )
        return PendingCandidateSelection(
            codes=values,
            submit_after=submit_after,
        )
    return None


def _quoted_choices(forms: tuple[str, ...]) -> str:
    return "、".join(f"「{form}」" for form in forms)


def pending_confirmation_copy() -> str:
    """Render the generic forms accepted by a single actor-owned ticket."""
    return f"回复{_quoted_choices(PENDING_CONFIRM_ADVERTISED_FORMS)}继续。"


def pending_batch_confirmation_copy() -> str:
    """Render every advertised add-only and add-then-submit form."""
    return (
        f"回复{_quoted_choices(PENDING_BATCH_ADD_ADVERTISED_FORMS)}只加入草稿；"
        f"回复{_quoted_choices(PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS)}"
        "则加入后提交。"
    )


def scoped_multi_word_candidate_copy(words: tuple[str, ...]) -> str:
    """Advertise only word-scoped numbered selection for repeated lists."""
    clean_words = tuple(dict.fromkeys(
        str(word or "").strip()
        for word in words
        if str(word or "").strip()
    ))
    if len(clean_words) < 2:
        return ""
    example = clean_words[-1]
    return (
        "多个词的候选编号分别从 1 开始；选择时请带上词条，"
        f"例如「{example} 添加1」；多选请回复「{example} 添加2、4」。"
    )


@dataclass(frozen=True)
class AdvertisedReplyContract:
    """Stateful reply forms discovered from the shared advertised vocabulary."""

    generic_assent_forms: tuple[str, ...] = ()
    batch_assent_forms: tuple[str, ...] = ()
    candidate_selection: bool = False
    deictic_batch_command: bool = False

    @property
    def requires_live_state(self) -> bool:
        return bool(
            self.generic_assent_forms
            or self.batch_assent_forms
            or self.candidate_selection
            or self.deictic_batch_command
        )


_ADVERTISED_QUOTE_PAIRS = (
    ("「", "」"),
    ("“", "”"),
    ("『", "』"),
)
_DEICTIC_BATCH_ADVERTISEMENT_RE = re.compile(
    r"(?<!已)(?:将|把)这\s*[1-9]\d{0,2}\s*个词"
    r"(?:都|全部)?(?:加入|添加|加到|放入|写入)(?:到|进|入)?草稿"
    r"(?:(?:并|然后|再)提交)?"
)


def advertised_reply_contract(text: str) -> AdvertisedReplyContract:
    """Detect only stateful forms emitted by the shared reply renderers."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))

    def advertised_forms(forms: tuple[str, ...]) -> tuple[str, ...]:
        found = []
        for form in forms:
            direct_advertisement = any(
                f"回复{left}{form}{right}" in normalized
                or f"发送{left}{form}{right}" in normalized
                for left, right in _ADVERTISED_QUOTE_PAIRS
            )
            action_list_advertisement = False
            if form in PENDING_BATCH_ADD_ADVERTISED_FORMS:
                action_list_advertisement = any(
                    re.search(
                        rf"(?m)^\s*[-•]\s*[^\n]{{0,128}}"
                        rf"{re.escape(left + form + right)}[^\n]{{0,128}}"
                        r"(?:→|->)\s*只加入草稿\s*$",
                        normalized,
                    )
                    is not None
                    for left, right in _ADVERTISED_QUOTE_PAIRS
                )
            elif form in PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS:
                action_list_advertisement = any(
                    re.search(
                        rf"(?m)^\s*[-•]\s*[^\n]{{0,128}}"
                        rf"{re.escape(left + form + right)}[^\n]{{0,128}}"
                        r"(?:→|->)\s*加入后提交\s*$",
                        normalized,
                    )
                    is not None
                    for left, right in _ADVERTISED_QUOTE_PAIRS
                )
            if direct_advertisement or action_list_advertisement:
                found.append(form)
        return tuple(found)

    candidate_selection = any(
        f" 添加1{right}" in normalized
        or f" 添加2、4{right}" in normalized
        for _left, right in _ADVERTISED_QUOTE_PAIRS
    )
    return AdvertisedReplyContract(
        generic_assent_forms=advertised_forms(PENDING_CONFIRM_ADVERTISED_FORMS),
        batch_assent_forms=advertised_forms((
            *PENDING_BATCH_ADD_ADVERTISED_FORMS,
            *PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS,
        )),
        candidate_selection=candidate_selection,
        deictic_batch_command=(
            _DEICTIC_BATCH_ADVERTISEMENT_RE.search(normalized) is not None
        ),
    )


def ensure_multi_word_candidate_copy(text: str) -> str:
    """Remove ambiguous number ads and append the closed multi-word grammar."""
    pairs = advertised_batch_binding_pairs(text)
    words = tuple(word for word, _code in pairs)
    scoped_copy = scoped_multi_word_candidate_copy(words)
    if not scoped_copy:
        return str(text)

    response = str(text)
    response = response.replace("可多选，如「添加2、4」。", "")
    response = response.replace("可多选，如「添加2、4」", "")
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
    response = re.sub(r"[ \t]+(?=\n|$)", "", response)
    response = re.sub(r"\n{3,}", "\n\n", response).rstrip()
    confirmation_copy = pending_batch_confirmation_copy()
    additions = [
        copy
        for copy in (confirmation_copy, scoped_copy)
        if copy and copy not in response
    ]
    return response + (("\n\n" + "\n".join(additions)) if additions else "")


def pending_confirmation_prompt_instruction() -> str:
    """Render model guidance from the same forms used by the parser."""
    return (
        "多词候选消息末尾必须逐字使用以下确认文案：\n"
        + pending_batch_confirmation_copy()
        + "\n多词的每个候选列表都从 1 开始，禁止广告或接受不带词条的编号选择；"
        "只能提示词条作用域形式，例如「词条 添加1」或「词条 添加2、4」。"
        "单词候选才可提示裸编号及「添加2、4」。"
    )


def expand_pending_confirmation_copy(text: str) -> str:
    """Expand prompt templates without duplicating user-facing reply forms."""
    return str(text).replace(
        PENDING_BATCH_CONFIRMATION_COPY_TOKEN,
        pending_batch_confirmation_copy(),
    )
