"""Shared reply forms and copy for deterministic pending confirmations."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional
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

ADD_OPERATION_VERB_FORMS = (
    "补一个",
    "补充",
    "补上",
    "加上",
    "添上",
    "也加",
    "再加",
    "加词",
    "添加",
    "加入",
    "加到",
    "新增",
    "创建",
    "写入",
    "放入",
    "收录",
    "录入",
    "记入",
    "都加",
    "补",
)
ADD_OPERATION_VERB_PATTERN = "(?:" + "|".join(
    sorted(
        (
            r"再加(?!入|到|上|词|添)"
            if verb == "再加"
            else re.escape(verb)
            for verb in ADD_OPERATION_VERB_FORMS
        ),
        key=len,
        reverse=True,
    )
) + ")"

PENDING_BATCH_ADD_ADVERTISED_FORMS = tuple(
    verb for verb in ("加入", "都加", "添加")
    if verb in ADD_OPERATION_VERB_FORMS
)
PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS = (
    "加入并提交",
    "都加并提交",
    "添加并提交",
)
PENDING_SINGLE_ADD_ADVERTISED_FORMS = ("加入",)
PENDING_SINGLE_ADD_AND_SUBMIT_ADVERTISED_FORMS = ("加入并提交",)
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


_BIND_HELP_TEXT = (
    "你还没有绑定键道账号哦～\n\n"
    "📝 绑定步骤：\n\n"
    "1. 登录键道网站：https://keytao.vercel.app\n"
    "2. 点击右上角用户名，进入【我的资料】\n"
    "   （或直接访问：https://keytao.vercel.app/profile ）\n"
    "3. 在【机器人账号绑定】区域点击【生成绑定码】\n"
    "4. 复制绑定码\n"
    "5. 在这里发送：/bind [你的绑定码]\n\n"
    "示例：/bind AB12CD\n\n"
    "💡 群聊中需要 @我 或回复我的消息才能触发绑定"
)


REMEDIATION_NO_SAFE_FOLLOWUP_MARKER = "当前没有可安全执行的后续命令"
FAILED_WRITE_TEMPLATE_PREFIX = "这条指令按当前表述无法执行"
FAILED_WRITE_TEMPLATE_MARKER = "本次未写入"
POLICY_BLOCK_TEMPLATE_PREFIX = "安全拦截："
SYSTEM_REPLY_TEMPLATE_MARKERS = (
    FAILED_WRITE_TEMPLATE_PREFIX,
    FAILED_WRITE_TEMPLATE_MARKER,
    REMEDIATION_NO_SAFE_FOLLOWUP_MARKER,
    POLICY_BLOCK_TEMPLATE_PREFIX,
)


UNBOUND_BINDING_PRECHECK_NOTICE = (
    "提示：你还未绑定键道账号，提交前请先绑定"
    "（发送 /bind 绑定码，详见 https://keytao.vercel.app/profile）。"
)


def system_reply_template_marker(text: str) -> str:
    """Return the first registered deterministic-template marker in text."""
    rendered = str(text or "")
    return next(
        (marker for marker in SYSTEM_REPLY_TEMPLATE_MARKERS if marker in rendered),
        "",
    )


def append_unbound_binding_notice(text: str, actor_is_bound: Optional[bool]) -> str:
    """Prepend one short binding notice without blocking a useful review."""
    rendered = str(text or "").strip()
    if actor_is_bound is not False or not rendered:
        return rendered
    if UNBOUND_BINDING_PRECHECK_NOTICE in rendered:
        return rendered
    return f"{UNBOUND_BINDING_PRECHECK_NOTICE}\n{rendered}"


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


def same_unique_binding_set(
    displayed: tuple[tuple[str, str], ...],
    recorded: tuple[tuple[str, str], ...],
) -> bool:
    """Compare complete bindings as a set while rejecting duplicates/omissions."""
    return bool(
        displayed
        and len(displayed) == len(recorded)
        and len(displayed) == len(set(displayed))
        and len(recorded) == len(set(recorded))
        and set(displayed) == set(recorded)
    )


@dataclass(frozen=True)
class PendingCandidateSelection:
    """A closed multi-slot selection parsed without model interpretation."""

    indices: tuple[int, ...] = ()
    codes: tuple[str, ...] = ()
    submit_after: bool = False


@dataclass(frozen=True)
class PendingAssentPhrase:
    """A whole-message assent decision that never contributes operands."""

    recognized: bool = False
    matched: bool = False
    submit_after: bool = False
    rejection: str = ""


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


_NATURAL_ASSENT_FORMS = tuple(sorted(
    {
        *PENDING_ASSENT_TEXTS,
        "写入草稿",
        "写入",
    },
    key=lambda value: (-len(value), value),
))
_NATURAL_ASSENT_FORM_RE = re.compile(
    "|".join(re.escape(value) for value in _NATURAL_ASSENT_FORMS)
)
_NATURAL_ACTION_ASSENT_FORMS = tuple(sorted(
    {
        *PENDING_BATCH_ASSENT_TEXTS,
        "确认",
        "执行",
        "确定",
        "同意",
        "写入草稿",
        "写入",
    },
    key=lambda value: (-len(value), value),
))
_NATURAL_ACTION_ASSENT_RE = re.compile(
    "|".join(re.escape(value) for value in _NATURAL_ACTION_ASSENT_FORMS)
)
_NATURAL_SUBMIT_RE = re.compile(
    r"(?:(?:并且?|然后(?:就)?|再|接着|随后|完成后|加入后|添加后|"
    r"写入后|加好后|加完|后|完)?(?:就)?)"
    r"(?:提交|提审|送审)(?:草稿|批次|审核)?"
)
_NATURAL_QUESTION_RE = re.compile(
    r"[?？]|(?:吗|么|嘛|呢|好不好|行不行)\s*[。.!！]*$|"
    r"^\s*(?:是否|能否|能不能|可否|可不可以|要不要|怎么|如何|为什么|是不是)"
)
_NATURAL_NEGATION_RE = re.compile(
    r"(?:(?:先|暂时)\s*)?(?:不要|别|不用|无需|不必|不再|不)\s*"
    r"(?:加入|添加|加|写入|确认|执行|提交|提审|送审)|取消|停止"
)
_NATURAL_UNSAFE_FRAME_RE = re.compile(
    r"(?:他说|她说|他们说|有人说|他让|她让|他们让|叫你|让你|要求你|"
    r"引用|转述|复述|解释|举例|例子|假设|如果|原话|不是让你|并非让你)"
)
_NATURAL_OTHER_ACTION_RE = re.compile(
    r"(?:删除|删掉|移除|修改|改成|改为|换成|替换|移动|挪开|挪到|"
    r"顺延|重排|排序|调码|重新编码|放在|放到|排在|提前|靠前|靠后|撤回)"
)
_NATURAL_ASSENT_PREFIX_RE = (
    r"(?:(?:好的|好|可以|行|嗯|麻烦|劳驾|请|我确认|我想|我要|"
    r"直接|现在|马上|立刻)){0,4}"
)
_NATURAL_ASSENT_ACTOR_RE = r"(?:(?:帮我|帮忙))?"
_NATURAL_ASSENT_OBJECT_RE = (
    r"(?:当前候选|这些候选|全部候选|这个候选|这些|这个|它们|候选|"
    r"全部|所有)"
)
_NATURAL_ASSENT_TARGET_RE = (
    rf"(?:(?:把|将){_NATURAL_ASSENT_OBJECT_RE}|"
    rf"{_NATURAL_ASSENT_OBJECT_RE})?"
)
_NATURAL_ASSENT_DESTINATION_RE = (
    rf"(?:{_NATURAL_ASSENT_OBJECT_RE}|(?:到|入)?草稿)?"
)
_NATURAL_ASSENT_BRIDGE_RE = (
    r"(?:(?:确认|执行)(?=(?:加入|添加|加|都加|写入)))?"
)
_NATURAL_ASSENT_SUFFIX_RE = r"(?:(?:一下|吧|啦|了|呀|哦)){0,3}"
_NATURAL_ASSENT_WHOLE_RE = re.compile(
    rf"{_NATURAL_ASSENT_PREFIX_RE}"
    rf"{_NATURAL_ASSENT_ACTOR_RE}"
    rf"{_NATURAL_ASSENT_TARGET_RE}"
    rf"{_NATURAL_ASSENT_BRIDGE_RE}"
    rf"(?:{_NATURAL_ASSENT_FORM_RE.pattern})"
    rf"{_NATURAL_ASSENT_DESTINATION_RE}"
    rf"(?:{_NATURAL_SUBMIT_RE.pattern})?"
    rf"{_NATURAL_ASSENT_SUFFIX_RE}"
)
_NATURAL_ASSENT_ACTION_START_RE = re.compile(
    rf"^{_NATURAL_ASSENT_PREFIX_RE}"
    rf"{_NATURAL_ASSENT_ACTOR_RE}"
    rf"{_NATURAL_ASSENT_TARGET_RE}"
    rf"{_NATURAL_ASSENT_BRIDGE_RE}"
    rf"(?:{_NATURAL_ACTION_ASSENT_RE.pattern})"
)


def parse_pending_assent_phrase(
    text: str,
    *,
    allowed_operands: tuple[str, ...] = (),
) -> PendingAssentPhrase:
    """Parse natural whole-state assent while deriving zero operands from text."""
    source = unicodedata.normalize("NFKC", str(text or "")).strip()
    source = re.sub(r"^\s*@\S+\s*", "", source, count=1)
    source = re.sub(r"^\s*(?:喵喵|键道)\s*", "", source, count=1)
    submit_after = bool(
        _NATURAL_SUBMIT_RE.search(source)
        or any(
            form in source
            for form in PENDING_BATCH_ADD_AND_SUBMIT_ASSENT_TEXTS
        )
    )
    question = _NATURAL_QUESTION_RE.search(source) is not None
    negation = _NATURAL_NEGATION_RE.search(source) is not None
    framed = bool(
        re.search(r"[\"'`“”‘’「」『』]", source)
        or _NATURAL_UNSAFE_FRAME_RE.search(source)
    )

    residual = re.sub(r"[\s，,。.!！、;；:：~～…（）()【】\[\]<>《》]+", "", source)
    for operand in sorted(
        {
            unicodedata.normalize("NFKC", str(value or "")).strip()
            for value in allowed_operands
            if str(value or "").strip()
        },
        key=lambda value: (-len(value), value),
    ):
        residual = residual.replace(operand, "")
    whole_match = _NATURAL_ASSENT_WHOLE_RE.fullmatch(residual) is not None
    action_start = _NATURAL_ASSENT_ACTION_START_RE.search(residual) is not None
    action_anywhere = _NATURAL_ACTION_ASSENT_RE.search(residual) is not None
    other_action = _NATURAL_OTHER_ACTION_RE.search(residual) is not None
    recognized = bool(
        whole_match
        or action_start
        or negation
        or ((framed or other_action) and action_anywhere)
    )
    if not recognized:
        return PendingAssentPhrase(submit_after=submit_after)
    if question:
        return PendingAssentPhrase(True, False, submit_after, "question")
    if negation:
        return PendingAssentPhrase(True, False, submit_after, "negation")
    if framed:
        return PendingAssentPhrase(True, False, submit_after, "framed")
    if other_action:
        return PendingAssentPhrase(True, False, submit_after, "other_action")
    if not whole_match:
        return PendingAssentPhrase(True, False, submit_after, "extra_content")
    return PendingAssentPhrase(True, True, submit_after)


_PLACEHOLDER_OPERAND_RE = re.compile(
    r"(?<![\u3400-\u9fffA-Za-z0-9_])"
    r"(?:词条|编码|编号|原词|x{2,}|…|\.{3,})"
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
    return f"{clean_reason}。{REMEDIATION_NO_SAFE_FOLLOWUP_MARKER}。"


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
        if exclusions:
            return AdvertisedSetReference(
                matched=True,
                exclusions=exclusions,
                submit_after=verb in PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS,
                advertised_verb=verb,
            )

    natural_assent = parse_pending_assent_phrase(source)
    if natural_assent.matched:
        return AdvertisedSetReference(
            matched=True,
            submit_after=natural_assent.submit_after,
            advertised_verb=source.strip(),
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


_SERVER_BACKED_WORD_SET_HEADER = "本轮服务端查询确认未收录的词："
_SERVER_BACKED_WORD_SET_FOOTER = "可以把列表中的词加入草稿。"


def render_server_backed_word_set(words: object) -> str:
    """Render one no-code inventory solely from a trusted absent-word set."""
    if not isinstance(words, (list, tuple)):
        return ""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_word in words:
        word = str(raw_word or "").strip()
        if (
            not word
            or len(word) > 128
            or any(marker in word for marker in ("\r", "\n", "「", "」"))
            or word in seen
        ):
            return ""
        normalized.append(word)
        seen.add(word)
    if len(normalized) < 2:
        return ""
    return "\n".join((
        _SERVER_BACKED_WORD_SET_HEADER,
        *(f"- 「{word}」" for word in normalized),
        _SERVER_BACKED_WORD_SET_FOOTER,
    ))


def advertised_word_set_words(text: str) -> tuple[str, ...]:
    """Parse only the deterministic server-backed no-code inventory shape."""
    lines = str(text or "").strip().splitlines()
    if (
        len(lines) < 4
        or lines[0] != _SERVER_BACKED_WORD_SET_HEADER
        or lines[-1] != _SERVER_BACKED_WORD_SET_FOOTER
    ):
        return ()
    words: list[str] = []
    seen: set[str] = set()
    for line in lines[1:-1]:
        match = re.fullmatch(r"- 「([^」\n]{1,128})」", line)
        if match is None:
            return ()
        word = match.group(1).strip()
        if not word or word in seen:
            return ()
        words.append(word)
        seen.add(word)
    return tuple(words) if len(words) >= 2 else ()


def render_server_backed_batch_candidates(
    items: object,
    candidate_scopes: object,
) -> str:
    """Render one complete batch solely from sealed server-record fields."""
    if (
        not isinstance(items, list)
        or len(items) < 2
        or not isinstance(candidate_scopes, list)
    ):
        return ""

    scopes_by_word: dict[str, tuple[tuple[str, bool], ...]] = {}
    for raw_scope in candidate_scopes:
        if not isinstance(raw_scope, dict):
            return ""
        word = unicodedata.normalize(
            "NFKC",
            str(raw_scope.get("word") or ""),
        ).strip()
        raw_candidates = raw_scope.get("candidates")
        if (
            not word
            or word in scopes_by_word
            or not isinstance(raw_candidates, list)
            or not raw_candidates
        ):
            return ""
        candidates: list[tuple[str, bool]] = []
        seen_codes: set[str] = set()
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, (list, tuple)) or len(raw_candidate) != 2:
                return ""
            code = unicodedata.normalize(
                "NFKC",
                str(raw_candidate[0] or ""),
            ).strip().lower()
            occupied = raw_candidate[1]
            if (
                re.fullmatch(r"[a-z]{1,12}", code) is None
                or not isinstance(occupied, bool)
                or code in seen_codes
            ):
                return ""
            candidates.append((code, occupied))
            seen_codes.add(code)
        scopes_by_word[word] = tuple(candidates)

    normalized_items: list[tuple[str, str, bool]] = []
    seen_words: set[str] = set()
    for raw_item in items:
        if not isinstance(raw_item, dict):
            return ""
        word = unicodedata.normalize(
            "NFKC",
            str(raw_item.get("word") or ""),
        ).strip()
        code = unicodedata.normalize(
            "NFKC",
            str(raw_item.get("code") or ""),
        ).strip().lower()
        if (
            not word
            or len(word) > 128
            or any(marker in word for marker in ("\r", "\n", "「", "」"))
            or word in seen_words
            or re.fullmatch(r"[a-z]{1,12}", code) is None
            or word not in scopes_by_word
            or code not in {candidate for candidate, _occupied in scopes_by_word[word]}
        ):
            return ""
        normalized_items.append(
            (word, code, bool(raw_item.get("needsManualReview", True)))
        )
        seen_words.add(word)
    if seen_words != set(scopes_by_word):
        return ""

    lines = [
        f"已根据当前服务端审词记录重列以下 {len(normalized_items)} 个词的候选："
    ]
    for word_index, (word, recommended_code, needs_review) in enumerate(
        normalized_items,
        start=1,
    ):
        lines.extend(("", f"{word_index}. 「{word}」", "   候选："))
        for candidate_index, (code, occupied) in enumerate(
            scopes_by_word[word],
            start=1,
        ):
            occupancy_copy = "已占用" if occupied else "空位"
            recommended_copy = "（推荐）" if code == recommended_code else ""
            lines.append(
                f"   {candidate_index}. {code} — {occupancy_copy}{recommended_copy}"
            )
        review_copy = "需管理员审核" if needs_review else "可自动通过"
        lines.append(f"   审核结论：{review_copy}")

    lines.extend((
        "",
        "确认后才会写入草稿。",
        pending_batch_confirmation_copy(),
    ))
    return "\n".join(lines)


def pending_single_candidate_confirmation_copy() -> str:
    """Render the only whole-state assent forms for one-word candidates."""
    return (
        f"回复{_quoted_choices(PENDING_SINGLE_ADD_ADVERTISED_FORMS)}只加入草稿；"
        f"回复{_quoted_choices(PENDING_SINGLE_ADD_AND_SUBMIT_ADVERTISED_FORMS)}"
        "则加入后提交。"
    )


def single_word_candidate_footer(candidate_count: int) -> str:
    """Render truthful selection and whole-state actions for one word."""
    lines = []
    if candidate_count > 1:
        example = "2、4" if candidate_count >= 4 else "1、2"
        lines.append(
            "可回复编号或编码选择其他编码；"
            f"可多选，如「添加{example}」。"
        )
    lines.append(pending_single_candidate_confirmation_copy())
    return "\n".join(lines)


def advertised_single_word_candidate_codes(text: str) -> tuple[str, ...]:
    """Read the ordered codes in a numbered one-word CODE-CHOICE display."""
    matches = list(re.finditer(
        r"(?m)^\s*(?P<index>[1-9]\d{0,2})[.)、]\s*"
        r"(?P<code>[a-z]{1,12})\s*(?:—|–|-)\s*[^\n]+$",
        unicodedata.normalize("NFKC", str(text or "")),
        re.IGNORECASE,
    ))
    if len(matches) < 2:
        return ()
    indexes = [int(match.group("index")) for match in matches]
    codes = tuple(match.group("code").lower() for match in matches)
    if indexes != list(range(1, len(matches) + 1)) or len(set(codes)) != len(codes):
        return ()
    return codes


def render_server_backed_single_word_candidates(
    word: object,
    recommended_code: object,
    candidates: object,
    occupied_words: object,
) -> str:
    """Render one candidate interface solely from trusted same-turn records."""
    normalized_word = str(word or "").strip()
    recommended = str(recommended_code or "").strip().lower()
    if (
        not normalized_word
        or len(normalized_word) > 128
        or any(marker in normalized_word for marker in ("\r", "\n", "「", "」"))
        or not isinstance(candidates, (list, tuple))
        or not isinstance(occupied_words, dict)
    ):
        return ""
    normalized_candidates: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for raw in candidates:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return ""
        code = str(raw[0] or "").strip().lower()
        occupied = raw[1]
        if (
            re.fullmatch(r"[a-z]{1,12}", code) is None
            or not isinstance(occupied, bool)
            or code in seen
        ):
            return ""
        normalized_candidates.append((code, occupied))
        seen.add(code)
    if len(normalized_candidates) < 2 or recommended not in seen:
        return ""
    lines = [f"「{normalized_word}」的服务端候选编码："]
    for index, (code, occupied) in enumerate(normalized_candidates, start=1):
        if occupied:
            words = [
                str(value or "").strip()
                for value in occupied_words.get(code, [])
                if str(value or "").strip()
            ]
            if not words:
                return ""
            label = "已有「" + "、".join(words) + "」"
        else:
            label = "空位"
        if code == recommended:
            label += "（推荐）"
        lines.append(f"{index}. {code} — {label}")
    lines.extend((
        "",
        f"是否以编码 {recommended} 将「{normalized_word}」加入草稿？",
        single_word_candidate_footer(len(normalized_candidates)),
    ))
    return "\n".join(lines)


def ensure_single_word_candidate_copy(text: str, candidate_count: int) -> str:
    """Normalize legacy one-word footers to the shared truthful contract."""
    response = str(text or "")
    selection_copy = ""
    if candidate_count > 1:
        example = "2、4" if candidate_count >= 4 else "1、2"
        selection_copy = (
            "可回复编号或编码选择其他编码；"
            f"可多选，如「添加{example}」。"
        )
    for old in (
        "可回复编号、编码，或「都加」；可多选，如「添加2、4」。",
        "可回复编号、编码，或「都加」；",
        "可回复编号、编码，或「都加」。",
        "也可回复编号选其他编码。",
    ):
        response = response.replace(old, selection_copy)
    if candidate_count <= 1:
        response = re.sub(
            r"可多选，如[「“『]添加\d+(?:、\d+)+[」”』][。.]?",
            "",
            response,
        )
    elif "可回复编号或编码选择其他编码" not in response:
        response = response.rstrip() + "\n" + selection_copy
    confirmation_copy = pending_single_candidate_confirmation_copy()
    if confirmation_copy not in response:
        response = response.rstrip() + "\n" + confirmation_copy
    response = re.sub(r"[ \t]+(?=\n|$)", "", response)
    response = re.sub(r"\n{3,}", "\n\n", response)
    return response.strip()


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
    code_choice_advertisement: bool = False
    deictic_batch_command: bool = False
    binding_advertisement: bool = False
    word_set_advertisement: bool = False

    @property
    def requires_live_state(self) -> bool:
        return bool(
            self.generic_assent_forms
            or self.batch_assent_forms
            or self.candidate_selection
            or self.code_choice_advertisement
            or self.deictic_batch_command
            or self.binding_advertisement
            or self.word_set_advertisement
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
_WORD_SET_ADVERTISEMENT_RE = re.compile(
    r"(?:可以|可|能够|能)?(?:把|将)?"
    r"(?:列表中(?:的)?|上述|以上|这些|这批|其余|剩下(?:的)?)"
    r"(?:词|词条)?(?:都|全部)?"
    r"(?:加入|添加|加到|放入|写入)(?:到|进|入)?草稿"
)
_QUOTED_WORD_SET_COMMAND_RE = re.compile(
    r"(?:回复|发送|直接发|可直接发)[^\n]{0,32}"
    r"[「“『](?:把|将)[^→\n]{1,512}?"
    r"(?:加入|添加|加到|放入|写入)(?:到|进|入)?草稿[」”』]"
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
        re.search(
            re.escape(left) + r"添加[1-9]\d*(?:[、,，][1-9]\d*)*" + re.escape(right),
            normalized,
        )
        is not None
        or f" 添加1{right}" in normalized
        or f" 添加2、4{right}" in normalized
        for left, right in _ADVERTISED_QUOTE_PAIRS
    )
    displayed_binding_pairs = advertised_batch_binding_pairs(normalized)
    code_choice_advertisement = bool(
        advertised_single_word_candidate_codes(normalized)
        and len(displayed_binding_pairs) <= 1
    )
    generic_assent_forms = advertised_forms(PENDING_CONFIRM_ADVERTISED_FORMS)
    batch_assent_forms = advertised_forms((
            *PENDING_BATCH_ADD_ADVERTISED_FORMS,
            *PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS,
        ))
    deictic_batch_command = (
        _DEICTIC_BATCH_ADVERTISEMENT_RE.search(normalized) is not None
    )
    has_binding_pairs = bool(displayed_binding_pairs)
    word_set_advertisement = bool(
        _WORD_SET_ADVERTISEMENT_RE.search(normalized)
        or _QUOTED_WORD_SET_COMMAND_RE.search(normalized)
        or (deictic_batch_command and not has_binding_pairs)
    )
    binding_advertisement = bool(
        generic_assent_forms
        or candidate_selection
        or (
            batch_assent_forms
            and (has_binding_pairs or not word_set_advertisement)
        )
        or (deictic_batch_command and has_binding_pairs)
    )
    return AdvertisedReplyContract(
        generic_assent_forms=generic_assent_forms,
        batch_assent_forms=batch_assent_forms,
        candidate_selection=candidate_selection,
        code_choice_advertisement=code_choice_advertisement,
        deictic_batch_command=deictic_batch_command,
        binding_advertisement=binding_advertisement,
        word_set_advertisement=word_set_advertisement,
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
        "必须把当前候选消息里的真实词语逐一代入作用域命令，"
        "禁止输出“词条”“编号”“原词”等占位词。"
        "单词候选才可提示裸编号及「添加2、4」。"
    )


def expand_pending_confirmation_copy(text: str) -> str:
    """Expand prompt templates without duplicating user-facing reply forms."""
    return str(text).replace(
        PENDING_BATCH_CONFIRMATION_COPY_TOKEN,
        pending_batch_confirmation_copy(),
    )
