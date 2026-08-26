"""Shared reply forms and copy for deterministic pending confirmations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import re
from typing import Any, Iterable, Optional
import unicodedata
from urllib.parse import urlsplit


PENDING_CONFIRM_ADVERTISED_FORMS = ("确认",)
PENDING_CONFIRM_ASSENT_TEXTS = frozenset({
    *PENDING_CONFIRM_ADVERTISED_FORMS,
    "执行",
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
    "确认并提交",
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


def trusted_pending_word_items(items: object) -> list[dict[str, str]]:
    """Keep only exact display-safe draft/submitted word records."""
    if not isinstance(items, list):
        return []
    trusted: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source") or "").strip()
        batch_id = str(raw.get("batchId") or "").strip()
        batch_url = str(raw.get("batchUrl") or "").strip()
        status = str(raw.get("batchStatus") or "").strip()
        item_status = str(raw.get("itemStatus") or "").strip()
        action = str(raw.get("action") or "").strip()
        word = str(raw.get("word") or "").strip()
        code = str(raw.get("code") or "").strip().lower()
        phrase_type = str(raw.get("type") or "Phrase").strip()
        parsed_url = urlsplit(batch_url)
        safe_url = bool(
            parsed_url.scheme in {"http", "https"}
            and parsed_url.netloc
            and parsed_url.path.rstrip("/").endswith(f"/batch/{batch_id}")
            and not parsed_url.query
            and not parsed_url.fragment
        )
        valid_status = (
            source == "draft"
            and status == "Draft"
            or source == "submitted"
            and status == "Submitted"
            and item_status == "Pending"
        )
        if (
            not valid_status
            or not safe_url
            or not batch_id
            or action not in {"Create", "Change", "Delete"}
            or not word
            or len(word) > 80
            or not re.fullmatch(r"[a-z]{1,6}", code)
            or phrase_type not in {
                "Single", "Phrase", "Supplement", "Symbol",
                "Link", "CSS", "CSSSingle", "English",
            }
        ):
            continue
        key = (source, batch_id, word, code)
        if key in seen:
            continue
        seen.add(key)
        trusted.append({
            "source": source,
            "batchId": batch_id,
            "batchUrl": batch_url,
            "batchStatus": status,
            "itemStatus": item_status,
            "action": action,
            "word": word,
            "code": code,
            "type": phrase_type,
        })
    return trusted


def pending_word_reminder_lines(
    items: object,
    *,
    words: Iterable[str] = (),
) -> list[str]:
    """Render compact actor-bound reminders from trusted server records."""
    word_filter = {
        str(word or "").strip() for word in words if str(word or "").strip()
    }
    lines: list[str] = []
    for item in trusted_pending_word_items(items):
        word = item["word"]
        if word_filter and word not in word_filter:
            continue
        if item["source"] == "submitted":
            line = (
                f"「{word}」已在待审核批次中"
                f"（→ {item['code']}，审核中）：{item['batchUrl']}"
            )
        else:
            line = (
                f"「{word}」已在当前草稿中"
                f"（→ {item['code']}）：{item['batchUrl']}"
            )
        if line not in lines:
            lines.append(line)
    return lines


def prepend_pending_word_reminders(
    text: str,
    items: object,
    *,
    words: Iterable[str] = (),
) -> str:
    """Put pending facts first and avoid repeating an identical reminder."""
    reminders = pending_word_reminder_lines(items, words=words)
    if not reminders:
        return str(text or "").strip()
    body_lines = [
        line for line in str(text or "").strip().splitlines()
        if line.strip() not in reminders
    ]
    body = "\n".join(body_lines).strip()
    return "\n".join(reminders) + (("\n\n" + body) if body else "")
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
    "你还未绑定键道账号。\n"
    "1. 登录：https://keytao.vercel.app\n"
    "2. 打开【我的资料】：https://keytao.vercel.app/profile\n"
    "3. 在【机器人账号绑定】生成并复制绑定码\n"
    "4. 发送：/bind [绑定码]\n"
    "示例：/bind AB12CD\n\n"
    "群聊中请 @我 或回复我的消息。"
)


REMEDIATION_FALLBACK_GUIDANCE = (
    "可发送「查看草稿」"
)
FAILED_WRITE_TEMPLATE_PREFIX = "这条指令按当前表述无法执行"
FAILED_WRITE_TEMPLATE_MARKER = "本次未写入"
POLICY_BLOCK_TEMPLATE_PREFIX = "安全拦截："
SYSTEM_REPLY_TEMPLATE_MARKERS = (
    FAILED_WRITE_TEMPLATE_PREFIX,
    FAILED_WRITE_TEMPLATE_MARKER,
    REMEDIATION_FALLBACK_GUIDANCE,
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
            r"(?m)^\s*[-•]\s*[“『]「(?P<word>[^」\n]{1,128})」\s*"
            r"占\s*(?P<code>[a-z]{1,12})\s*、\s*"
            r"「[^」\n]{1,128}」顺延[”』]"
            r"(?:\s*[（(][^）)\n]{0,128}[）)])?\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?m)^\s*推荐[：:]\s*「(?P<word>[^」\n]{1,128})」\s*"
            r"占\s*(?P<code>[a-z]{1,12})\s*、\s*"
            r"「[^」\n]{1,128}」顺延(?:[；;][^\n]*)?\s*$",
            re.IGNORECASE,
        ),
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
    add_requested: bool = False
    cancel_requested: bool = False


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
        "都加入",
        "加进去",
        "加好",
        "加完",
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
        "都加入",
        "加进去",
        "加好",
        "加完",
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
_NATURAL_ADD_ACTION_RE = re.compile(
    "|".join(
        re.escape(value)
        for value in _NATURAL_ACTION_ASSENT_FORMS
        if value not in PENDING_CONFIRM_ASSENT_TEXTS
        and value not in {"确认", "执行", "确定", "同意"}
    )
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
    r"(?:(?:那就|就|好的|好|可以|行|嗯|麻烦|劳驾|请|我确认|我想|我要|"
    r"直接|现在|马上|立刻)){0,4}"
)
_NATURAL_ASSENT_ACTOR_RE = r"(?:(?:帮我|帮忙))?"
_NATURAL_ASSENT_OBJECT_RE = (
    r"(?:当前候选|这些候选|全部候选|这个候选|这些|这个|它们|候选|"
    r"这两个|两个|全部|所有|一起|都)"
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
_NATURAL_ASSENT_CLOSING_FILLER_RE = re.compile(
    r"(?:(?:谢谢(?:你)?|多谢|辛苦了|拜托了|麻烦了|感谢(?:你)?|"
    r"劳驾|拜托|有劳|谢啦|owo|这个词挺常用的)|[啦呀哦哈嘛])+$",
    re.IGNORECASE,
)
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
_NATURAL_ADD_CANCEL_WHOLE_RE = re.compile(
    r"(?:(?:那就|就|先)?(?:"
    r"放弃(?:(?:这|本)次)?(?:添加|加词|加入|操作)?|"
    r"取消(?:(?:这|本)次)?(?:添加|加词|加入|操作)?|"
    r"撤销(?:(?:这|本)次)?(?:添加|加词|加入|操作)|"
    r"(?:这个|这次|当前|刚才(?:这个|这次)?)(?:不要|不用)(?:了)?|"
    r"(?:不|别|不用|不要)(?:再)?(?:加|添加|加入)(?:了)?|"
    r"(?:不用|不要|不了)(?:了)?|"
    r"算了(?:不加(?:了)?)?"
    r"))"
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
    residual = _NATURAL_ASSENT_CLOSING_FILLER_RE.sub("", residual)
    whole_match = _NATURAL_ASSENT_WHOLE_RE.fullmatch(residual) is not None
    action_start = _NATURAL_ASSENT_ACTION_START_RE.search(residual) is not None
    action_anywhere = _NATURAL_ACTION_ASSENT_RE.search(residual) is not None
    add_requested = _NATURAL_ADD_ACTION_RE.search(residual) is not None
    other_action = _NATURAL_OTHER_ACTION_RE.search(residual) is not None
    cancel_requested = bool(
        not question
        and not framed
        and _NATURAL_ADD_CANCEL_WHOLE_RE.fullmatch(residual)
    )
    negation = bool(
        _NATURAL_NEGATION_RE.search(source) is not None
        or cancel_requested
    )
    recognized = bool(
        whole_match
        or action_start
        or negation
        or cancel_requested
        or ((framed or other_action) and action_anywhere)
    )
    if not recognized:
        return PendingAssentPhrase(
            submit_after=submit_after,
            add_requested=add_requested,
            cancel_requested=cancel_requested,
        )
    if question:
        return PendingAssentPhrase(
            True, False, submit_after, "question", add_requested,
            cancel_requested,
        )
    if negation:
        return PendingAssentPhrase(
            True, False, submit_after, "negation", add_requested,
            cancel_requested,
        )
    if framed:
        return PendingAssentPhrase(
            True, False, submit_after, "framed", add_requested,
            cancel_requested,
        )
    if other_action:
        return PendingAssentPhrase(
            True, False, submit_after, "other_action", add_requested,
            cancel_requested,
        )
    if not whole_match:
        return PendingAssentPhrase(
            True, False, submit_after, "extra_content", add_requested,
            cancel_requested,
        )
    return PendingAssentPhrase(
        True,
        True,
        submit_after,
        add_requested=add_requested,
        cancel_requested=cancel_requested,
    )


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
    """Render a refusal with one concrete next step or short fallback guidance."""
    clean_reason = str(reason or "").strip().rstrip("；;。")
    suggestion = render_executable_suggestion(command, words=words)
    if suggestion:
        return f"{clean_reason}。\n{suggestion}"
    if advertised_reply_contract(clean_reason).requires_live_state:
        return clean_reason + "。"
    return f"{clean_reason}。\n{REMEDIATION_FALLBACK_GUIDANCE}。"


def _humanize_warning_text(text: str) -> str:
    rendered = re.sub(r"\s+", " ", str(text or "")).strip()
    rendered = re.sub(r'词条\s*["“]([^"”]+)["”]\s*', r'「\1」', rendered)
    rendered = re.sub(r'编码\s*["“]([A-Za-z]+)["”]', r'编码 \1', rendered)
    return rendered.replace("将创建多编码词条", "这次会形成多编码词条")


def plain_warning_message(warning: Any) -> str:
    """Verbalize one structured server warning without reducing it to a count."""
    if is_dataclass(warning) and not isinstance(warning, type):
        warning = asdict(warning)
    if isinstance(warning, (list, tuple)):
        messages = [plain_warning_message(item) for item in warning]
        return "；".join(message for message in messages if message)
    if not isinstance(warning, dict):
        return _humanize_warning_text(
            str(warning or "服务端返回了一项需要确认的风险")
        )

    item = warning.get("item")
    source = item if isinstance(item, dict) else warning
    warning_type = str(warning.get("warningType") or "").strip()
    word = str(source.get("word") or warning.get("word") or "").strip()
    code = str(source.get("code") or warning.get("code") or "").strip().lower()
    weight = source.get("weight", warning.get("weight"))
    weight_copy = (
        f"（权重 {weight}）"
        if isinstance(weight, int) and not isinstance(weight, bool)
        else ""
    )

    if warning_type == "duplicate_code" and code:
        raw_existing = warning.get("existing")
        existing_rows = (
            raw_existing
            if isinstance(raw_existing, list)
            else [raw_existing]
            if isinstance(raw_existing, dict)
            else []
        )
        existing_words = tuple(dict.fromkeys(
            str(row.get("word") or "").strip()
            for row in existing_rows
            if isinstance(row, dict)
            and str(row.get("word") or "").strip()
            and str(row.get("code") or code).strip().lower() == code
        ))
        if existing_words:
            target_copy = f"「{word}」" if word else "该词"
            if word and word in existing_words:
                return (
                    f"「{word}」已在词库（{code}）；"
                    "若无意保留重复词条，无需再次写入"
                )
            return (
                f"{code} 与「{'、'.join(existing_words)}」同码，"
                f"确认后{target_copy}将作为重码写入{weight_copy}"
            )

    for field in ("message", "impact", "reason", "summary"):
        value = warning.get(field)
        if isinstance(value, str) and value.strip():
            return _humanize_warning_text(value)

    if warning_type == "code_chain_priority" and word and code:
        return f"「{word}」写入 {code} 后会改变同码顺序{weight_copy}"
    if warning_type == "skipped_candidate_slot" and word and code:
        return f"「{word}」将写入 {code}，并跳过更短的空位{weight_copy}"
    if warning_type == "multiple_code" and word and code:
        return f"「{word}」将以 {code} 形成多编码词条{weight_copy}"
    if word and code:
        return f"「{word}」写入编码 {code} 前需要确认{weight_copy}"
    if word:
        return f"「{word}」写入前需要确认{weight_copy}"
    return "服务端返回了一项需要确认的风险"


def already_existing_word_copy(
    word: str,
    codes: tuple[str, ...],
    *,
    can_choose_other_code: bool,
    can_reorder: bool = False,
) -> str:
    """Lead with an exact existing fact, then state only live options."""
    clean_codes = tuple(dict.fromkeys(
        str(code or "").strip().lower() for code in codes if str(code or "").strip()
    ))
    if not word or not clean_codes:
        return ""
    actions: list[str] = []
    if can_choose_other_code:
        actions.append("选择其他编码（回复「换码」）")
    if can_reorder:
        actions.append("调整现有排序（回复「重排」）")
    action_copy = (
        "仍可" + "或".join(actions) + "；若保留现状，无需操作。"
        if actions
        else "当前没有必须执行的变更；若保留现状，无需操作。"
    )
    return f"「{word}」已在词库（{'、'.join(clean_codes)}）。\n{action_copy}"


_WARNING_COUNT_COPY_RE = re.compile(
    r"(?:(?:存在|有|发现|共有)\s*)?"
    r"(?:\d+|[一二两三四五六七八九十]+)\s*个警告"
    r"(?:需要(?:你)?确认)?[。；;，,：:]?"
)


def strip_warning_count_copy(text: str) -> str:
    """Remove model-authored warning counts; authoritative details are appended elsewhere."""
    lines = []
    for raw_line in str(text or "").splitlines():
        cleaned = _WARNING_COUNT_COPY_RE.sub("", raw_line)
        cleaned = re.sub(r"[，,；;]\s*[。.]", "。", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


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
    return "回复「确认」执行，或「取消」。"


def pending_batch_confirmation_copy() -> str:
    """Render one canonical add-only and add-then-submit form."""
    return "回复「加入」写入草稿，或回复「加入并提交」写入并提交。"


_SERVER_BACKED_WORD_SET_HEADER = "未收录："
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
        pending_batch_confirmation_copy(),
    ))


def advertised_word_set_words(text: str) -> tuple[str, ...]:
    """Parse only the deterministic server-backed no-code inventory shape."""
    lines = str(text or "").strip().splitlines()
    if (
        len(lines) < 5
        or lines[0] != _SERVER_BACKED_WORD_SET_HEADER
        or lines[-2] != _SERVER_BACKED_WORD_SET_FOOTER
        or lines[-1] != pending_batch_confirmation_copy()
    ):
        return ()
    words: list[str] = []
    seen: set[str] = set()
    for line in lines[1:-2]:
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
    occupied_words_by_scope: dict[str, dict[str, tuple[str, ...]]] = {}
    ordering_assessments_by_scope: dict[str, list[dict[str, object]]] = {}
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
        raw_occupied_words = raw_scope.get("occupiedWords", {})
        if not isinstance(raw_occupied_words, dict):
            return ""
        occupied_words: dict[str, tuple[str, ...]] = {}
        for raw_code, raw_words in raw_occupied_words.items():
            code = unicodedata.normalize("NFKC", str(raw_code or "")).strip().lower()
            if (
                code not in seen_codes
                or not isinstance(raw_words, list)
                or not raw_words
            ):
                return ""
            normalized_words: list[str] = []
            for raw_occupied_word in raw_words:
                occupied_word = unicodedata.normalize(
                    "NFKC",
                    str(raw_occupied_word or ""),
                ).strip()
                if (
                    not occupied_word
                    or len(occupied_word) > 128
                    or any(marker in occupied_word for marker in ("\r", "\n", "「", "」"))
                    or occupied_word in normalized_words
                ):
                    return ""
                normalized_words.append(occupied_word)
            occupied_words[code] = tuple(normalized_words)
        occupied_words_by_scope[word] = occupied_words
        raw_ordering = raw_scope.get("orderingAssessments", [])
        if not isinstance(raw_ordering, list) or any(
            not isinstance(value, dict) for value in raw_ordering
        ):
            return ""
        ordering_assessments_by_scope[word] = [
            dict(value) for value in raw_ordering[:2]
        ]

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

    lines = [f"{len(normalized_items)} 个词的候选："]
    for word_index, (word, recommended_code, needs_review) in enumerate(
        normalized_items,
        start=1,
    ):
        reorder_recommendation = validated_front_insert_recommendation(
            word,
            scopes_by_word[word],
            occupied_words_by_scope[word],
            ordering_assessments_by_scope[word],
        )
        if (
            reorder_recommendation is not None
            and recommended_code != reorder_recommendation["occupantCode"]
        ):
            return ""
        existing_codes = tuple(
            code
            for code, _occupied in scopes_by_word[word]
            if word in occupied_words_by_scope[word].get(code, ())
        )
        lines.append(f"{word_index}. 「{word}」")
        existing_copy = already_existing_word_copy(
            word,
            existing_codes,
            can_choose_other_code=any(
                code not in existing_codes for code, _occupied in scopes_by_word[word]
            ),
            can_reorder=reorder_recommendation is not None,
        )
        if existing_copy:
            lines.extend(f"   {line}" for line in existing_copy.splitlines())
        lines.append("   候选：")
        for candidate_index, (code, occupied) in enumerate(
            scopes_by_word[word],
            start=1,
        ):
            occupied_words = occupied_words_by_scope[word].get(code, ())
            occupancy_copy = (
                "已有「" + "、".join(occupied_words) + "」"
                if occupied and occupied_words
                else "已占用"
                if occupied
                else "空位"
            )
            if reorder_recommendation is not None:
                if code == reorder_recommendation["occupantCode"]:
                    recommended_copy = "（推荐：需重排）"
                elif code == reorder_recommendation["freeCode"]:
                    recommended_copy = "（不调序备选）"
                else:
                    recommended_copy = ""
            else:
                recommended_copy = "（推荐）" if code == recommended_code else ""
            lines.append(
                f"   {candidate_index}. {code} — {occupancy_copy}{recommended_copy}"
            )
        if reorder_recommendation is not None:
            fallback_index = next(
                index
                for index, (code, _occupied) in enumerate(
                    scopes_by_word[word],
                    start=1,
                )
                if code == reorder_recommendation["freeCode"]
            )
            lines.extend(
                front_insert_recommendation_copy(
                    reorder_recommendation,
                    fallback_index,
                ).splitlines()
            )
        review_copy = "需管理员审核" if needs_review else "可自动通过"
        lines.append(f"   自动审核：{review_copy}")

    lines.append(pending_batch_confirmation_copy())
    scoped_copy = scoped_multi_word_candidate_copy(tuple(
        word for word, _code, _needs_review in normalized_items
    ))
    if scoped_copy:
        lines.append(scoped_copy)
    return "\n".join(lines)


def render_server_backed_batch_lookup(
    items: object,
    candidate_scopes: object,
) -> str:
    """Render a multi-word candidate snapshot without minting write controls."""
    rendered = render_server_backed_batch_candidates(items, candidate_scopes)
    words = tuple(
        str(item.get("word") or "").strip()
        for item in items or []
        if isinstance(item, dict) and str(item.get("word") or "").strip()
    )
    footer = "\n".join(filter(None, (
        pending_batch_confirmation_copy(),
        scoped_multi_word_candidate_copy(words),
    )))
    if not rendered.endswith(footer):
        return ""
    return rendered[: -len(footer)].rstrip()


def render_query_retry_reply(words: object) -> str:
    """Offer a read-only retry when a query display cannot be proven."""
    if not isinstance(words, (list, tuple)):
        return "这条候选查询无法绑定到本轮服务端编码记录；请逐词重新查询。"
    normalized: list[str] = []
    for raw_word in words:
        word = unicodedata.normalize("NFKC", str(raw_word or "")).strip()
        if (
            not word
            or len(word) > 128
            or any(marker in word for marker in ("\r", "\n", "「", "」"))
            or word in normalized
        ):
            continue
        normalized.append(word)
    if not normalized:
        return "这条候选查询无法绑定到本轮服务端编码记录；请逐词重新查询。"
    targets = "、".join(f"「{word}」" for word in normalized)
    return (
        "这条候选查询无法唯一绑定到本轮服务端编码记录。"
        f"可逐词发送 {targets} 重新查询。"
    )


def pending_single_candidate_confirmation_copy() -> str:
    """Render the only whole-state assent forms for one-word candidates."""
    return "回复「加入」写入草稿，或回复「加入并提交」写入并提交。"


def validated_front_insert_recommendation(
    word: object,
    candidates: object,
    occupied_words: object,
    assessments: object,
) -> Optional[dict[str, str]]:
    """Return the first comparator-backed front insert bound to one snapshot."""
    normalized_word = str(word or "").strip()
    if (
        not normalized_word
        or not isinstance(candidates, (list, tuple))
        or not isinstance(occupied_words, dict)
        or not isinstance(assessments, list)
    ):
        return None

    occupancy: dict[str, bool] = {}
    for raw in candidates:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return None
        code = str(raw[0] or "").strip().lower()
        occupied = raw[1]
        if (
            re.fullmatch(r"[a-z]{1,12}", code) is None
            or not isinstance(occupied, bool)
            or (code in occupancy and occupancy[code] is not occupied)
        ):
            return None
        occupancy[code] = occupied

    for assessment in assessments[:2]:
        if not isinstance(assessment, dict):
            return None
        if str(assessment.get("verdict") or "") != "front_more_common":
            continue
        new_word = str(assessment.get("newWord") or "").strip()
        occupant_word = str(assessment.get("occupantWord") or "").strip()
        occupant_code = str(assessment.get("occupantCode") or "").strip().lower()
        free_code = str(assessment.get("freeCode") or "").strip().lower()
        new_code = str(
            assessment.get("newCode")
            or assessment.get("recommendedCode")
            or ""
        ).strip().lower()
        raw_bound_occupants = occupied_words.get(occupant_code, ())
        if not isinstance(raw_bound_occupants, (list, tuple)):
            return None
        bound_occupants = tuple(
            str(value or "").strip()
            for value in raw_bound_occupants
            if str(value or "").strip()
        )
        ordered_codes = tuple(occupancy)
        if (
            new_word != normalized_word
            or not occupant_word
            or occupancy.get(occupant_code) is not True
            or occupancy.get(free_code) is not False
            or ordered_codes.index(occupant_code) >= ordered_codes.index(free_code)
            or occupant_word not in bound_occupants
            or new_code != occupant_code
        ):
            return None
        return {
            "verdict": "front_more_common",
            "newWord": new_word,
            "occupantWord": occupant_word,
            "occupantCode": occupant_code,
            "freeCode": free_code,
            "newCode": new_code,
            "summary": str(assessment.get("summary") or "").strip(),
        }
    return None


def front_insert_recommendation_copy(
    recommendation: dict[str, str],
    fallback_selector: object = None,
) -> str:
    """Render one parser-checked recommendation plus its opt-out."""
    word = str(recommendation.get("newWord") or "").strip()
    occupant = str(recommendation.get("occupantWord") or "").strip()
    occupant_code = str(recommendation.get("occupantCode") or "").strip().lower()
    free_code = str(recommendation.get("freeCode") or "").strip().lower()
    summary = str(recommendation.get("summary") or "").strip()
    command = (
        f"「{word}」占 {occupant_code}、「{occupant}」顺延"
    )
    recommendation_line = render_executable_suggestion(
        command,
        words=(word, occupant),
    )
    if not recommendation_line:
        return ""
    selector = str(fallback_selector or "").strip()
    opt_out = (
        f"不重排选 {selector}（{free_code}）。"
        if selector
        else f"不重排选 {free_code}。"
    )
    lines = ["推荐：", recommendation_line]
    if summary:
        lines.append(f"依据：{summary}")
    lines.append(opt_out)
    return "\n".join(lines)


def single_word_candidate_footer(candidate_count: int) -> str:
    """Render truthful selection and whole-state actions for one word."""
    if candidate_count > 1:
        example = "2、4" if candidate_count >= 4 else "1、2"
        return (
            "回复编号或编码选择"
            f"（可多选，如「添加{example}」）；"
            "\n"
            "回复「加入」写入草稿，或回复「加入并提交」写入并提交。"
        )
    return pending_single_candidate_confirmation_copy()


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
    ordering_assessments: object = None,
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
    reorder_recommendation = validated_front_insert_recommendation(
        normalized_word,
        normalized_candidates,
        occupied_words,
        ordering_assessments,
    )
    if reorder_recommendation is not None:
        recommended = reorder_recommendation["occupantCode"]
    existing_codes = tuple(
        code
        for code, _occupied in normalized_candidates
        if normalized_word in {
            str(value or "").strip()
            for value in occupied_words.get(code, [])
        }
    )
    lines: list[str] = []
    existing_copy = already_existing_word_copy(
        normalized_word,
        existing_codes,
        can_choose_other_code=any(
            code not in existing_codes for code, _occupied in normalized_candidates
        ),
        can_reorder=reorder_recommendation is not None,
    )
    if existing_copy:
        lines.extend(existing_copy.splitlines())
    lines.append(f"「{normalized_word}」候选编码：")
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
        if reorder_recommendation is not None:
            if code == reorder_recommendation["occupantCode"]:
                label += "（推荐：需重排）"
            elif code == reorder_recommendation["freeCode"]:
                label += "（不调序备选）"
        elif code == recommended:
            label += "（推荐）"
        lines.append(f"{index}. {code} — {label}")
    if reorder_recommendation is not None:
        fallback_index = next(
            index
            for index, (code, _occupied) in enumerate(
                normalized_candidates,
                start=1,
            )
            if code == reorder_recommendation["freeCode"]
        )
        lines.append(
            front_insert_recommendation_copy(
                reorder_recommendation,
                fallback_index,
            )
        )
    else:
        lines.append(f"• 「{normalized_word}」→ {recommended}（推荐）")
    lines.append(single_word_candidate_footer(len(normalized_candidates)))
    return "\n".join(lines)


def render_server_backed_single_word_lookup(
    word: object,
    recommended_code: object,
    candidates: object,
    occupied_words: object,
    *,
    reviewed_prompt: object = "",
) -> str:
    """Render a rich trusted read snapshot without minting write capability."""
    actionable = render_server_backed_single_word_candidates(
        word,
        recommended_code,
        candidates,
        occupied_words,
    )
    if not actionable:
        return ""
    normalized_word = str(word or "").strip()
    recommended = str(recommended_code or "").strip().lower()
    expected_codes = advertised_single_word_candidate_codes(actionable)
    source = str(reviewed_prompt or "").strip()
    body = ""
    has_reorder_copy = False
    if source and advertised_single_word_candidate_codes(source) == expected_codes:
        reorder_copy = re.search(
            rf"(?m)^\s*推荐[：:]\s*\n"
            rf"\s*[-•]\s*[“『]「{re.escape(normalized_word)}」\s*占\s*"
            rf"{re.escape(recommended)}\s*、\s*「[^」\n]+」顺延[”』]"
            rf"(?:\s*[（(][^）)\n]*[）)])?\s*\n"
            rf"(?:\s*依据[：:][^\n]*\n)?"
            r"\s*不重排选\s+\d+(?:\s*[（(][a-z]{1,12}[）)])?[。.]?\s*$",
            source,
            re.IGNORECASE,
        )
        if reorder_copy is None:
            reorder_copy = re.search(
                rf"(?m)^\s*推荐[：:]\s*「{re.escape(normalized_word)}」\s*占\s*"
                rf"{re.escape(recommended)}\s*、\s*「[^」\n]+」顺延(?:[；;][^\n]*)?\n"
                r"\s*不重排选\s+\d+(?:\s*[（(][a-z]{1,12}[）)])?[。.]?\s*$",
                source,
                re.IGNORECASE,
            )
        if reorder_copy is not None:
            body = source[:reorder_copy.end()].rstrip()
            has_reorder_copy = True
        else:
            confirmation = re.search(
                rf"(?m)^(?:如不调整现有排序，)?是否(?:仍)?以编码\s+"
                rf"{re.escape(recommended)}\s+将「{re.escape(normalized_word)}」"
                r"加入草稿[?？]\s*$",
                source,
                re.IGNORECASE,
            )
            if confirmation is None:
                confirmation = re.search(
                    rf"(?m)^\s*[•-]\s*「{re.escape(normalized_word)}」\s*→\s*"
                    rf"{re.escape(recommended)}\s*[（(]推荐[）)]\s*$",
                    source,
                    re.IGNORECASE,
                )
            if confirmation is not None:
                body = source[:confirmation.start()].rstrip()
        if body:
            body = re.sub(
                r"（回复[「“『][^」”』]+[」”』]执行）",
                "",
                body,
            )
    if not body:
        body = actionable.partition("\n\n是否以编码")[0].rstrip()
    if not body or advertised_single_word_candidate_codes(body) != expected_codes:
        return ""
    if has_reorder_copy:
        return "\n".join((
            body,
            "本次仅查询，不建立写入确认。",
            single_word_candidate_footer(len(expected_codes)),
        ))
    return "\n".join((
        body,
        f"推荐编码：{recommended}（本次仅查询）",
        single_word_candidate_footer(len(expected_codes)),
    ))


def advertised_single_word_lookup_codes(text: str) -> tuple[str, ...]:
    """Return ordered codes only from the deterministic read-only contract."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    recommended = re.search(
        r"(?m)^推荐编码:(?P<code>[a-z]{1,12})\(本次仅查询\)\s*$",
        normalized,
        re.IGNORECASE,
    )
    if recommended is None:
        recommended = re.search(
            r"(?m)^推荐:\s*\n[-•]\s*[“『]「[^「」\r\n]{1,128}」占\s*"
            r"(?P<code>[a-z]{1,12})、\s*「[^「」\r\n]{1,128}」顺延[”』]"
            r"(?:\s*\([^\r\n)]*\))?\s*\n(?:依据:[^\r\n]*\n)?"
            r"不重排选\s+[1-9]\d{0,2}"
            r"(?:\([a-z]{1,12}\))?[。.]*\s*$",
            normalized,
            re.IGNORECASE,
        )
        if recommended is None:
            recommended = re.search(
                r"(?m)^推荐:\s*「[^「」\r\n]{1,128}」占\s*"
                r"(?P<code>[a-z]{1,12})、\s*「[^「」\r\n]{1,128}」顺延"
                r"(?:;[^\r\n]*)?\n不重排选\s+[1-9]\d{0,2}"
                r"(?:\([a-z]{1,12}\))?[。.]*\s*$",
                normalized,
                re.IGNORECASE,
            )
        if recommended is None:
            return ()
        candidate_source = normalized[:recommended.start()]
    else:
        candidate_source = normalized[:recommended.start()]
    codes = advertised_single_word_candidate_codes(candidate_source)
    if not codes:
        return ()
    if recommended.group("code").lower() not in codes:
        return ()
    return codes


def advertised_single_word_lookup_word(text: str) -> str:
    """Extract only the referent from the deterministic read-only renderer."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    match = re.search(
        r"(?m)^(?:词库暂无收录)?「(?P<word>[^「」\r\n]{1,128})」"
        r"(?:候选编码)?[:：]\s*$",
        normalized,
    )
    if match is None or not advertised_single_word_lookup_codes(normalized):
        return ""
    return match.group("word").strip()


def ensure_single_word_candidate_copy(text: str, candidate_count: int) -> str:
    """Normalize legacy one-word footers to the shared truthful contract."""
    response = str(text or "")
    for old in (
        "可回复编号、编码，或「都加」；可多选，如「添加2、4」。",
        "可回复编号、编码，或「都加」；",
        "可回复编号、编码，或「都加」。",
        "也可回复编号选其他编码。",
        "可回复编号或编码选择其他编码；可多选，如「添加2、4」。",
        "可回复编号或编码选择其他编码；可多选，如「添加1、2」。",
    ):
        response = response.replace(old, "")
    if candidate_count <= 1:
        response = re.sub(
            r"可多选，如[「“『]添加\d+(?:、\d+)+[」”』][。.]?",
            "",
            response,
        )
    footer = single_word_candidate_footer(candidate_count)
    response = response.replace(footer, "")
    confirmation_copy = pending_single_candidate_confirmation_copy()
    response = response.replace(confirmation_copy, "")
    response = response.rstrip() + "\n" + footer
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
        "每个词的编号都从 1 开始；"
        f"回复「{example} 添加1」，多选回复「{example} 添加2、4」。"
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
    read_only_single_word_lookup: bool = False
    command_suggestions: tuple[str, ...] = ()

    @property
    def requires_live_state(self) -> bool:
        if self.read_only_single_word_lookup:
            return bool(
                self.generic_assent_forms
                or self.deictic_batch_command
                or self.word_set_advertisement
                or (
                    self.command_suggestions
                    and not command_suggestions_are_closed_candidate_selections(
                        self.command_suggestions
                    )
                )
            )
        return bool(
            self.generic_assent_forms
            or self.batch_assent_forms
            or self.candidate_selection
            or self.code_choice_advertisement
            or self.deictic_batch_command
            or self.binding_advertisement
            or self.word_set_advertisement
            or self.command_suggestions
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
_COMMAND_SUGGESTION_LEAD_RE = re.compile(
    r"(?:确认执行)?请发|请发送|发送下面|比如|例如"
)
_COMMAND_SUGGESTION_VERB_RE = re.compile(
    rf"(?:{ADD_OPERATION_VERB_PATTERN}|提交|删除|移除|修改|改成|改为|"
    r"改到|调整到|移到|挪到|换到|顺延|重新编码|调整权重)"
)


def advertised_command_suggestions(text: str) -> tuple[str, ...]:
    """Extract copyable model suggestions from explicit advertisement frames."""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    positioned: list[tuple[int, str]] = []
    for left, right in _ADVERTISED_QUOTE_PAIRS:
        depth = 0
        start = -1
        for index, character in enumerate(normalized):
            if character == left:
                if depth == 0:
                    start = index
                depth += 1
                continue
            if character != right or depth == 0:
                continue
            depth -= 1
            if depth != 0 or start < 0:
                continue
            command = normalized[start + 1:index].strip()
            prefix = normalized[max(0, start - 120):start]
            if (
                0 < len(command) <= 256
                and "\n" not in command
                and _COMMAND_SUGGESTION_LEAD_RE.search(prefix) is not None
                and _COMMAND_SUGGESTION_VERB_RE.search(command) is not None
                and re.search(r"[\u3400-\u9fff]", command)
            ):
                positioned.append((start, command))
    suggestions: list[str] = []
    seen: set[str] = set()
    for _position, command in sorted(positioned):
        if command not in seen:
            suggestions.append(command)
            seen.add(command)
    return tuple(suggestions)


def command_suggestions_are_closed_candidate_selections(
    suggestions: tuple[str, ...],
) -> bool:
    """Return whether every suggestion belongs to the closed selection grammar."""
    if not suggestions:
        return False
    return all(
        re.fullmatch(
            r"(?:[\u3400-\u9fff]{1,32}\s+)?"
            r"添加[1-9]\d*(?:[、,][1-9]\d*)*",
            suggestion,
        )
        is not None
        for suggestion in suggestions
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
                or f"或{left}{form}{right}" in normalized
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
    read_only_single_word_lookup = bool(
        advertised_single_word_lookup_word(normalized)
    )
    return AdvertisedReplyContract(
        generic_assent_forms=generic_assent_forms,
        batch_assent_forms=batch_assent_forms,
        candidate_selection=candidate_selection,
        code_choice_advertisement=code_choice_advertisement,
        deictic_batch_command=deictic_batch_command,
        binding_advertisement=binding_advertisement,
        word_set_advertisement=word_set_advertisement,
        read_only_single_word_lookup=read_only_single_word_lookup,
        command_suggestions=advertised_command_suggestions(normalized),
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
