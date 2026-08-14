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
    patterns = (
        re.compile(
            r"(?m)^\s*[-•]\s*「(?P<word>[^」\n]{1,128})」\s*"
            r"(?:→|->)\s*(?P<code>[a-z]{1,12})\s*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?m)^\s*是否以编码\s+(?P<code>[a-z]{1,12})\s+"
            r"将「(?P<word>[^」\n]{1,128})」加入草稿[？?]?\s*$",
            re.IGNORECASE,
        ),
    )
    matches = sorted(
        (match for pattern in patterns for match in pattern.finditer(normalized)),
        key=lambda match: match.start(),
    )
    for match in matches:
        pair = (
            match.group("word").strip(),
            match.group("code").strip().lower(),
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
