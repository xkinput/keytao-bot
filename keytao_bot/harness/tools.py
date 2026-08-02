"""Tool execution adapter for the agent harness."""
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from nonebot.log import logger


@dataclass(frozen=True)
class ToolContext:
    platform: Optional[str] = None
    user_id: Optional[str] = None
    current_message: Optional[str] = None
    writes_allowed: bool = True
    trusted_codes_by_word: Optional[Dict[str, frozenset[str]]] = None
    trusted_draft_words_by_id: Optional[Dict[str, str]] = None
    trusted_draft_items_by_id: Optional[Dict[str, Dict[str, str]]] = None
    trusted_phrase_types_by_key: Optional[Dict[Tuple[str, str], frozenset[str]]] = None
    trusted_reviewed_items_by_key: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None
    mutation_confirmed: bool = False


_DELETE_INTENT_RE = re.compile(r"删除|删掉|去掉|移除|撤销|清空|清理|全部删|都删")
_MUTATION_INTENT_RE = re.compile(
    r"添加|加入|加到|新增|创建|写入|放入|提交|删除|删掉|去掉|移除|清空|清理|"
    r"撤销|撤回|召回|修改|改成|改为|替换|顺延|挪开|重新编码|保留|批量处理"
    r"|都删|其余删|其他删"
)
_NEGATED_MUTATION_RE = re.compile(
    r"(?:不要|别(?!的)|无需|不用|禁止|不要真的).{0,12}"
    r"(?:添加|加入|加到|新增|创建|写入|放入|提交|删除|删掉|去掉|移除|清空|清理|撤销|撤回|修改|改成|改为|替换|重新编码|顺延|保留)"
)
_EXPLANATION_ONLY_RE = re.compile(
    r"(?:什么意思|什么含义|解释一下|说明一下|怎么做|如何操作|操作方法|"
    r"为什么|为何|教程|示例|假设|如果|是否支持|能否介绍|翻译|怎么理解|怎么说|"
    r"会发生什么|会怎样|会如何|有什么后果|后果是什么|有什么影响)"
)
_TEXT_TRANSFORM_RE = re.compile(r"(?:改写|润色|复述|翻译|引用|摘录|转述)")
_QUESTION_RE = re.compile(
    r"[?？]|(?:是否|能否|可否|会不会|是不是|要不要|怎么样|怎样|如何|"
    r"想知道|之后的结果|结果是什么)|"
    r"(?:吗|么|呢|好不好|行不行|可不可以|能不能|对不对|可以不|"
    r"不可以|不行)(?:[。.!！])?$"
)
_ABORT_RE = re.compile(r"(?:算了|取消|别执行|不要执行|先不要|不用了|不做了|别做了)")
_EXPLICIT_REQUEST_PREFIX_RE = re.compile(
    r"^(?:请|麻烦|帮我|给我|现在|立即|直接|确认|我要|我想|把|将|替我|为我|只|仅|除了)"
)
_META_DISCUSSION_RE = re.compile(
    r"(?:这句话|这段话|这条消息|引用|消息里|上面写|刚才说).{0,80}"
    r"(?:什么意思|什么含义|解释|说明|怎么理解)"
)
_DATA_CONTEXT_RE = re.compile(
    r"^(?:请|麻烦|帮我|给我|我要|我想)?"
    r"(?:分析|判断|翻译|解释|改写|复述|记录|排版|标注)"
    r"(?:以下|一下|这段|这句|这条|一句话|内容|消息|用户请求|是否)"
)
_QUOTED_DATA_RE = re.compile(r"「[^」]*」|“[^”]*”|‘[^’]*’|\"[^\"]*\"|'[^']*'")
_UNTRUSTED_QUOTE_PREFIX_RE = re.compile(
    r"(?:引用|复述|改写|翻译|摘录|转述|备注|示例|例句|原话|"
    r"这句|这段|消息里|写着|展示|显示|命令).{0,8}$"
)
_UNTRUSTED_DATA_TAIL_RE = re.compile(
    r"(?:以下(?:是|为)?(?:例句|示例|引用|原话|命令)|"
    r"(?:以下|后面|下方).{0,16}(?:仅记录|只记录|不要处理|无需处理|不处理|别处理)|"
    r"(?:并|同时)?(?:展示|显示|引用|摘录|复述)(?:一下|这段|命令)?|"
    r"作为备注).*$",
    re.DOTALL,
)
_INLINE_CODE_RE = re.compile(r"`[^`]*`")
_COMMAND_CLAUSE_SPLIT_RE = re.compile(r"[，,。.!！?？;；\n]+")
_COMMAND_PREFIX_RE = re.compile(
    r"^(?:请|麻烦|帮我|给我|现在|立即|直接|确认|我要|我想|替我|为我|"
    r"并|并且|同时|然后|再|还要|以及|另外|接着|顺便)*"
)
_NEGATIVE_MODAL_RE = re.compile(
    r"(?:不要|别|不|无需|不用|禁止|不得|请勿|不予|严禁|不许|不可|不能|"
    r"切勿|拒绝|莫)"
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[。！？!?;；\n]")
_ACTION_TOKENS = {
    "Create": re.compile(r"添加|加入|加到|新增|创建|写入|放入|加词"),
    "Change": re.compile(r"修改|改成|改为|替换|重新编码|顺延|挪开|移到"),
    "Delete": re.compile(r"删除|删掉|移除"),
    "Keep": re.compile(
        r"只保留|仅保留|保留|留下|别动|不要动|"
        r"不动|别删|不要删|不删|留着"
    ),
}
_WORD_LEFT_PREFIXES = (
    "添加", "加入", "加到", "新增", "创建", "写入", "放入", "加词",
    "修改", "改成", "替换", "删除", "删掉", "移除", "保留", "只保留", "仅保留",
    "把", "将", "词条", "词语",
)
_WORD_RIGHT_SUFFIXES = (
    "编码", "代码", "改成", "改为", "修改为", "替换为", "加入", "添加",
    "加到草稿", "加入草稿", "删除", "删掉", "移除", "到草稿", "放入草稿",
    "和", "与", "及", "、", "都", "并", "以", "为",
)
MUTATING_TOOL_NAMES = frozenset({
    "keytao_create_phrase",
    "keytao_remove_draft_item",
    "keytao_batch_add_to_draft",
    "keytao_batch_remove_draft_items",
    "keytao_shift_phrase_code",
    "keytao_recall_batch",
    "keytao_submit_batch",
})
_MAX_STAGED_MUTATION_PREVIEW_CHARS = 3500
_MAX_STAGED_MUTATION_LIST_ITEMS = 50
_PROTECTED_WORD_RE = (
    r"(?:别动|不要动|别改|不要改|不动|保持(?:原样)?|"
    r"保留|别碰|不要碰|别删|不要删|不删除|不修改|不顺延|不移动|"
    r"(?:不得|请勿|不予|严禁|不许|不可|不能|切勿|拒绝|莫).{0,3}"
    r"(?:添加|加入|删除|删掉|移除|修改|替换|顺延|移动))"
)
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
_PHRASE_TYPES = frozenset(value for _hint, value in _TYPE_HINTS)


def trusted_mutation_source(message: str) -> str:
    """Preserve line structure while removing quoted or marked untrusted data."""
    text = str(message or "")
    pieces: List[str] = []
    cursor = 0
    for match in _QUOTED_DATA_RE.finditer(text):
        pieces.append(text[cursor:match.start()])
        prefix = text[max(0, match.start() - 24):match.start()]
        quoted_content = match.group(0)[1:-1]
        pieces.append(
            " " * (match.end() - match.start())
            if (
                _UNTRUSTED_QUOTE_PREFIX_RE.search(prefix)
                or _MUTATION_INTENT_RE.search(quoted_content)
                or _DELETE_INTENT_RE.search(quoted_content)
            )
            else match.group(0)
        )
        cursor = match.end()
    pieces.append(text[cursor:])
    text = _INLINE_CODE_RE.sub("", "".join(pieces))
    return _UNTRUSTED_DATA_TAIL_RE.sub("", text)


def _mutation_authorization_view(message: str) -> str:
    """Return only positive command clauses plus explicit protection clauses."""
    text = trusted_mutation_source(message)

    trusted_clauses: List[str] = []
    for clause in _COMMAND_CLAUSE_SPLIT_RE.split(text):
        compact = re.sub(r"\s+", "", clause).strip()
        if not compact:
            continue
        candidate = _COMMAND_PREFIX_RE.sub("", compact, count=1)
        has_mutation = bool(_MUTATION_INTENT_RE.search(candidate))
        is_positive_command = bool(
            _MUTATION_INTENT_RE.match(candidate)
            or re.match(
                r"(?:把|将).{1,80}(?:添加|加入|加到|新增|创建|写入|放入|"
                r"提交|删除|删掉|去掉|移除|清空|撤销|撤回|修改|改成|改为|"
                r"替换|重新编码|顺延|挪开|移到|保留)",
                candidate,
            )
            or re.match(r"除了.{1,80}(?:都删|删除|删掉|去掉|移除)", candidate)
            or re.match(
                r"(?:草稿|批次)(?:中的)?(?:全部|都|所有)(?:条目)?"
                r"(?:删除|删掉|去掉|移除)",
                candidate,
            )
        )
        is_protection_clause = bool(
            (has_mutation and _NEGATIVE_MODAL_RE.search(candidate))
            or re.search(_PROTECTED_WORD_RE, candidate)
        )
        if is_positive_command or is_protection_clause:
            trusted_clauses.append(compact)
    return "；".join(trusted_clauses)


def message_authorizes_mutation(message: str) -> bool:
    """Accept write authority only from the current user's explicit raw text."""
    raw_text = re.sub(r"\s+", "", str(message or ""))
    text = re.sub(r"\s+", "", _mutation_authorization_view(message))
    authorization_text = _QUOTED_DATA_RE.sub("", text)
    if (
        not authorization_text
        or _NEGATED_MUTATION_RE.search(authorization_text)
        or _QUESTION_RE.search(raw_text)
        or _ABORT_RE.search(raw_text)
    ):
        return False
    mutation_match = _MUTATION_INTENT_RE.search(authorization_text)
    if (
        mutation_match is None
        or _META_DISCUSSION_RE.search(text)
        or _DATA_CONTEXT_RE.search(authorization_text)
        or re.search(r"(?:作为|设为).{0,8}(?:文章)?标题", authorization_text)
    ):
        return False
    if (
        _EXPLANATION_ONLY_RE.search(authorization_text)
        or _TEXT_TRANSFORM_RE.search(authorization_text)
    ):
        return False
    return mutation_match.start() == 0 or bool(
        _EXPLICIT_REQUEST_PREFIX_RE.match(authorization_text)
    ) or bool(
        re.match(
            r"(?:草稿|批次)(?:中的)?(?:全部|都|所有)(?:条目)?"
            r"(?:删除|删掉|去掉|移除)",
            authorization_text,
        )
    )


def _extract_explicit_phrase_type(message: str) -> Optional[str]:
    for hint, phrase_type in _TYPE_HINTS:
        if hint in message:
            return phrase_type
    return None


def _extract_phrase_type_for_target(message: str, target: str) -> Optional[str]:
    """Resolve an explicit type only from the target's own clause."""
    spans = _exact_target_spans(message, target)
    if not spans:
        return None
    for target_start, target_end in spans:
        clause_start = 0
        clause_end = len(message)
        for boundary in _CLAUSE_BOUNDARY_RE.finditer(message):
            if boundary.end() <= target_start:
                clause_start = boundary.end()
                continue
            if boundary.start() >= target_end:
                clause_end = boundary.start()
                break
        candidates: List[tuple[int, int, str]] = []
        for hint, phrase_type in _TYPE_HINTS:
            start = clause_start
            while True:
                index = message.find(hint, start, clause_end)
                if index < 0:
                    break
                hint_end = index + len(hint)
                if hint_end <= target_start:
                    distance = target_start - hint_end
                elif index >= target_end:
                    distance = index - target_end
                else:
                    distance = 0
                candidates.append((distance, -len(hint), phrase_type))
                start = index + len(hint)
        if candidates:
            minimum = min(
                (distance, negative_length)
                for distance, negative_length, _phrase_type in candidates
            )
            nearest = {
                phrase_type
                for distance, negative_length, phrase_type in candidates
                if (distance, negative_length) == minimum
            }
            if minimum[0] <= 24 and len(nearest) == 1:
                return next(iter(nearest))
    return None


def _is_word_protected(message: str, word: str) -> bool:
    escaped_word = re.escape(word)
    return bool(
        re.search(escaped_word + r".{0,8}" + _PROTECTED_WORD_RE, message)
        or re.search(_PROTECTED_WORD_RE + r".{0,8}" + escaped_word, message)
    )


def _action_match_is_negated(
    message: str,
    match_start: int,
    clause_start: int,
) -> bool:
    prefix = message[max(clause_start, match_start - 10):match_start]
    return bool(
        re.search(
            r"(?:不要|别|不|无需|不用|禁止|不得|请勿|不予|严禁|不许|不可|不能|切勿|拒绝|莫)"
            r"(?:再|真的)?\s*$",
            prefix,
        )
    )


def _is_han(character: str) -> bool:
    return bool(character and "\u3400" <= character <= "\u9fff")


def _exact_target_spans(message: str, target: str) -> List[tuple[int, int]]:
    """Return occurrences that represent a complete user-written target."""
    if not target:
        return []
    if re.fullmatch(r"[A-Za-z0-9_-]+", target):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_-]){re.escape(target)}(?![A-Za-z0-9_-])"
        )
        return [match.span() for match in pattern.finditer(message)]

    quoted_patterns = (
        f"「{target}」", f"“{target}”", f"‘{target}’",
        f'"{target}"', f"'{target}'",
    )
    spans: List[tuple[int, int]] = []
    for quoted in quoted_patterns:
        start = 0
        while True:
            index = message.find(quoted, start)
            if index < 0:
                break
            offset = 1
            spans.append((index + offset, index + offset + len(target)))
            start = index + len(quoted)

    start = 0
    while True:
        index = message.find(target, start)
        if index < 0:
            break
        end = index + len(target)
        before = message[:index]
        after = message[end:]
        left_char = before[-1:] or ""
        right_char = after[:1] or ""
        left_ok = (
            not left_char
            or not _is_han(left_char)
            or left_char in "和与及、把将"
            or any(before.endswith(prefix) for prefix in _WORD_LEFT_PREFIXES)
        )
        right_ok = (
            not right_char
            or not _is_han(right_char)
            or any(after.startswith(suffix) for suffix in _WORD_RIGHT_SUFFIXES)
        )
        if left_ok and right_ok:
            spans.append((index, end))
        start = index + 1
    return sorted(set(spans))


def _contains_exact_target(message: str, target: str) -> bool:
    return bool(_exact_target_spans(message, target))


def _action_is_bound_to_target(message: str, target: str, action: str) -> bool:
    """Require the nearest action in the same clause to match this target."""
    if action not in _ACTION_TOKENS:
        return False
    for target_start, target_end in _exact_target_spans(message, target):
        clause_start = 0
        clause_end = len(message)
        for boundary in _CLAUSE_BOUNDARY_RE.finditer(message):
            if boundary.end() <= target_start:
                clause_start = boundary.end()
                continue
            if boundary.start() >= target_end:
                clause_end = boundary.start()
                break
        matches: List[tuple[int, str, bool]] = []
        for label, pattern in _ACTION_TOKENS.items():
            for match in pattern.finditer(message, clause_start, clause_end):
                if match.end() <= target_start:
                    distance = target_start - match.end()
                elif match.start() >= target_end:
                    distance = match.start() - target_end
                else:
                    distance = 0
                matches.append((
                    distance,
                    label,
                    _action_match_is_negated(message, match.start(), clause_start),
                ))
        if not matches:
            continue
        minimum = min(distance for distance, _label, _negated in matches)
        nearest = [
            (label, negated)
            for distance, label, negated in matches
            if distance == minimum
        ]
        if any(negated for _label, negated in nearest):
            continue
        nearest_labels = {
            label for label, _negated in nearest
        }
        if minimum <= 24 and nearest_labels == {action}:
            return True
    return False


def _clause_bounds(message: str, start: int, end: int) -> tuple[int, int]:
    clause_start = 0
    clause_end = len(message)
    for boundary in _CLAUSE_BOUNDARY_RE.finditer(message):
        if boundary.end() <= start:
            clause_start = boundary.end()
            continue
        if boundary.start() >= end:
            clause_end = boundary.start()
            break
    return clause_start, clause_end


def _span_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] <= right[0]:
        return right[0] - left[1]
    if right[1] <= left[0]:
        return left[0] - right[1]
    return 0


_NON_CODE_ASCII_TOKENS = frozenset({
    "css", "csssingle", "single", "phrase", "supplement", "symbol",
    "link", "english", "create", "change", "delete", "confirmed",
    "true", "false",
})


def _explicit_code_spans(
    message: str,
    clause_start: int,
    clause_end: int,
    target_span: tuple[int, int],
) -> List[tuple[str, tuple[int, int]]]:
    tokens: List[tuple[str, tuple[int, int]]] = []
    for match in re.finditer(r"(?<![A-Za-z])[A-Za-z]{1,12}(?![A-Za-z])", message):
        span = match.span()
        value = match.group(0)
        if span[0] < clause_start or span[1] > clause_end:
            continue
        if _span_distance(span, target_span) == 0:
            continue
        if value.lower() in _NON_CODE_ASCII_TOKENS:
            continue
        tokens.append((value, span))
    return tokens


def _code_is_bound_to_target(
    message: str,
    target: str,
    code: str,
    trusted_codes: frozenset[str],
) -> bool:
    """Bind an explicit code by proximity, or use an unopposed read capability."""
    if not code:
        return True
    code_is_explicit = _contains_exact_target(message, code)
    for target_span in _exact_target_spans(message, target):
        clause_start, clause_end = _clause_bounds(message, *target_span)
        tokens = _explicit_code_spans(
            message,
            clause_start,
            clause_end,
            target_span,
        )
        if code_is_explicit:
            distances = [
                (_span_distance(target_span, span), value)
                for value, span in tokens
            ]
            if not distances:
                continue
            minimum = min(distance for distance, _value in distances)
            nearest = {
                value for distance, value in distances if distance == minimum
            }
            if nearest == {code}:
                return True
            continue
        if code in trusted_codes and not tokens:
            return True
    return False


def _change_transition_is_bound(message: str, old_word: str, new_word: str) -> bool:
    """Require old_word -> change verb -> new_word within one clause."""
    if not old_word or not new_word:
        return False
    for old_span in _exact_target_spans(message, old_word):
        clause_start, clause_end = _clause_bounds(message, *old_span)
        for new_span in _exact_target_spans(message, new_word):
            if new_span[0] < clause_start or new_span[1] > clause_end:
                continue
            if old_span[1] > new_span[0]:
                continue
            between = message[old_span[1]:new_span[0]]
            if (
                _ACTION_TOKENS["Change"].search(between)
                and not re.search(
                    r"(?:不要|别|不).{0,6}(?:修改|改成|改为|替换|重新编码|顺延)",
                    between,
                )
            ):
                return True
    return False


def _draft_item_reference_is_bound(
    message: str,
    item_id: str,
    words_by_id: Dict[str, str],
    items_by_id: Dict[str, Dict[str, str]],
) -> bool:
    """Bind a derived draft ID to enough current text to disambiguate it."""
    item = items_by_id.get(item_id) or {}
    word = str(item.get("word") or words_by_id.get(item_id) or "").strip()
    if (
        not word
        or not _contains_exact_target(message, word)
        or not _action_is_bound_to_target(message, word, "Delete")
        or _is_word_protected(message, word)
    ):
        return False

    same_word_ids = {
        draft_id
        for draft_id, draft_word in words_by_id.items()
        if draft_word == word
    }
    same_word_ids.update(
        draft_id
        for draft_id, draft_item in items_by_id.items()
        if str(draft_item.get("word") or "").strip() == word
    )
    if len(same_word_ids) <= 1:
        return True

    code = str(item.get("code") or "").strip()
    if not code or not _code_is_bound_to_target(message, word, code, frozenset()):
        return False
    same_code_ids = {
        draft_id
        for draft_id in same_word_ids
        if str((items_by_id.get(draft_id) or {}).get("code") or "").strip() == code
    }
    if len(same_code_ids) <= 1:
        return True

    phrase_type = str(item.get("type") or "").strip()
    explicit_type = _extract_phrase_type_for_target(message, word)
    if not phrase_type or explicit_type != phrase_type:
        return False
    same_type_ids = {
        draft_id
        for draft_id in same_code_ids
        if str((items_by_id.get(draft_id) or {}).get("type") or "").strip()
        == phrase_type
    }
    return len(same_type_ids) <= 1


def _find_code_reassignments(items: object) -> List[Dict[str, str]]:
    if not isinstance(items, list):
        return []

    deletes: Dict[str, set[str]] = {}
    creates: Dict[str, set[str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        word = item.get("word")
        code = item.get("code")
        action = item.get("action", "Create")
        if not isinstance(word, str) or not isinstance(code, str):
            continue
        if action == "Delete":
            deletes.setdefault(word, set()).add(code)
        elif action == "Create":
            creates.setdefault(word, set()).add(code)

    reassignments: List[Dict[str, str]] = []
    for word, old_codes in deletes.items():
        for new_code in creates.get(word, set()):
            for old_code in old_codes:
                if old_code != new_code:
                    reassignments.append({"word": word, "oldCode": old_code, "newCode": new_code})
    return reassignments


class ToolExecutor:
    """Executes registered skills and injects platform context when needed."""

    def __init__(
        self,
        get_tool_function: Callable[[str], Optional[Callable]],
        context_tools: frozenset[str],
        mutation_guard: Optional[
            Callable[[ToolContext, str, Dict], Optional[Dict]]
        ] = None,
    ):
        self._get_tool_function = get_tool_function
        self._context_tools = context_tools
        self._mutation_guard = mutation_guard

    def canonicalize_arguments(
        self,
        tool_name: str,
        arguments: Dict,
        context: ToolContext,
    ) -> Dict:
        """Freeze server-derived fields before execution or pending persistence."""
        return self._with_trusted_mutation_fields(tool_name, arguments, context)

    async def call(self, tool_name: str, arguments: Dict, context: ToolContext) -> str:
        call_args = self.canonicalize_arguments(tool_name, arguments, context)
        policy_error = self._validate_policy(tool_name, call_args, context)
        if policy_error:
            logger.warning(f"Tool {tool_name} blocked by policy: {policy_error}")
            return json.dumps(policy_error, ensure_ascii=False)

        tool_func = self._get_tool_function(tool_name)
        if not tool_func:
            return json.dumps({"error": f"Tool {tool_name} not found"}, ensure_ascii=False)

        try:
            if tool_name in self._context_tools:
                if not context.platform or not context.user_id:
                    return json.dumps(
                        {"error": "内部错误：无法获取用户平台信息"}, ensure_ascii=False
                    )
                call_args["platform"] = context.platform
                call_args["platform_id"] = context.user_id

            result = await tool_func(**call_args)
            result_json = json.dumps(result, ensure_ascii=False)
            logger.info(f"Tool {tool_name} result: {result_json[:300]}")
            return result_json
        except Exception as error:
            logger.error(f"Tool {tool_name} error: {type(error).__name__}: {error}")
            return json.dumps({"error": str(error)}, ensure_ascii=False)

    @staticmethod
    def _trusted_phrase_type(
        context: ToolContext,
        word: str,
        code: str,
    ) -> Optional[str]:
        values = (context.trusted_phrase_types_by_key or {}).get((word, code), frozenset())
        normalized = {value for value in values if value in _PHRASE_TYPES}
        return next(iter(normalized)) if len(normalized) == 1 else None

    def _with_trusted_mutation_fields(
        self,
        tool_name: str,
        arguments: Dict,
        context: ToolContext,
    ) -> Dict:
        call_args = dict(arguments)
        if not context.current_message:
            return call_args
        message = _mutation_authorization_view(context.current_message or "")

        def sanitize_item(item: Dict) -> Dict:
            sanitized = dict(item)
            word = str(sanitized.get("word") or "").strip()
            code = str(sanitized.get("code") or "").strip()
            action = str(sanitized.get("action") or "Create")
            old_word = str(sanitized.get("old_word") or "").strip()
            explicit_type = _extract_phrase_type_for_target(message, word)

            sanitized.pop("remark", None)
            sanitized.pop("needsManualReview", None)
            sanitized.pop("needs_manual_review", None)

            if action == "Create":
                capability = (context.trusted_reviewed_items_by_key or {}).get(
                    (word, code)
                )
                capability_type = str((capability or {}).get("type") or "").strip()
                capability_applies = bool(
                    capability
                    and capability_type in _PHRASE_TYPES
                    and (explicit_type is None or explicit_type == capability_type)
                )
                if capability_applies:
                    sanitized["type"] = capability_type
                    canonical_remark = str(capability.get("remark") or "").strip()
                    if canonical_remark:
                        sanitized["remark"] = canonical_remark
                    sanitized["needs_manual_review"] = bool(
                        capability.get("needs_manual_review", True)
                    )
                else:
                    if explicit_type:
                        sanitized["type"] = explicit_type
                    else:
                        sanitized.pop("type", None)
                    sanitized["needs_manual_review"] = True
                return sanitized

            target = old_word if action == "Change" and old_word else word
            resolved_type = (
                _extract_phrase_type_for_target(message, target)
                or self._trusted_phrase_type(context, target, code)
            )
            if resolved_type:
                sanitized["type"] = resolved_type
            else:
                sanitized.pop("type", None)
            return sanitized

        if tool_name == "keytao_create_phrase":
            return sanitize_item(call_args)

        if tool_name == "keytao_batch_add_to_draft" and isinstance(call_args.get("items"), list):
            sanitized_items = []
            for item in call_args["items"]:
                if not isinstance(item, dict):
                    sanitized_items.append(item)
                    continue
                sanitized = sanitize_item(item)
                if "needs_manual_review" in sanitized:
                    sanitized["needsManualReview"] = sanitized.pop(
                        "needs_manual_review"
                    )
                sanitized_items.append(sanitized)
            call_args["items"] = sanitized_items

        return call_args

    def _validate_policy(self, tool_name: str, arguments: Dict, context: ToolContext) -> Optional[Dict]:
        message = context.current_message or ""
        if not context.writes_allowed and tool_name in MUTATING_TOOL_NAMES:
            return {
                "success": False,
                "policyBlocked": True,
                "requiresTextFollowUp": True,
                "message": (
                    "安全拦截：历史、记忆、引用或附件内容不能授权修改草稿或提交。"
                    "请先展示拟操作内容，再让用户发送一条明确的当前文字指令。"
                ),
            }
        if tool_name in MUTATING_TOOL_NAMES and self._mutation_guard is not None:
            guard_error = self._mutation_guard(context, tool_name, arguments)
            if guard_error:
                return guard_error
        if (
            message
            and tool_name in MUTATING_TOOL_NAMES
            and not message_authorizes_mutation(message)
        ):
            return {
                "success": False,
                "policyBlocked": True,
                "requiresTextFollowUp": True,
                "message": (
                    "安全拦截：当前文字不是明确的执行指令。"
                    "问句、解释、引用、备注或已经取消的说法不能授权写操作。"
                ),
            }
        if (
            message
            and tool_name in MUTATING_TOOL_NAMES
            and "confirmed" in arguments
        ):
            return {
                "success": False,
                "policyBlocked": True,
                "requiresTextFollowUp": True,
                "message": (
                    "安全拦截：模型不能自行声明 confirmed=true。"
                    "只有服务端保存的待确认票据才能进入确认执行流程。"
                ),
            }
        if message and tool_name in MUTATING_TOOL_NAMES:
            binding_error = self._validate_current_message_binding(
                tool_name,
                arguments,
                context,
            )
            if binding_error:
                return binding_error
        if tool_name == "keytao_batch_remove_draft_items" and message:
            ids = arguments.get("ids")
            if isinstance(ids, list) and len(ids) > 3 and not _DELETE_INTENT_RE.search(message):
                return {
                    "success": False,
                    "policyBlocked": True,
                    "message": "安全拦截：当前消息不是批量删除请求，禁止一次删除多个草稿条目。请只删除本次明确需要替换的条目，或先向用户确认。",
                    "blockedIds": ids,
                }

        if tool_name != "keytao_batch_add_to_draft":
            return self._stage_agent_mutation(tool_name, arguments, context)

        reassignments = _find_code_reassignments(arguments.get("items"))
        if not reassignments or not message:
            return self._stage_agent_mutation(tool_name, arguments, context)

        blocked = [
            item for item in reassignments
            if (
                not _contains_exact_target(message, item["word"])
                or _is_word_protected(message, item["word"])
            )
        ]
        if not blocked:
            return self._stage_agent_mutation(tool_name, arguments, context)

        blocked_labels = [
            f"{item['word']} {item['oldCode']}→{item['newCode']}"
            for item in blocked
        ]
        return {
            "success": False,
            "policyBlocked": True,
            "message": "安全拦截：禁止手工迁移未点名词条。需要插入已占用编码并顺延时，必须调用 keytao_shift_phrase_code，让工具按每个被挤词自己的 encode 候选链计算。",
            "blockedReassignments": blocked_labels,
        }

    @staticmethod
    def _stage_agent_mutation(
        tool_name: str,
        arguments: Dict,
        context: ToolContext,
    ) -> Optional[Dict]:
        message = context.current_message or ""
        if (
            not message
            or tool_name not in MUTATING_TOOL_NAMES
            or context.mutation_confirmed
        ):
            return None
        preview = json.dumps(arguments, ensure_ascii=False, sort_keys=True)
        oversized_lists = {
            key: len(value)
            for key, value in arguments.items()
            if isinstance(value, list)
            and len(value) > _MAX_STAGED_MUTATION_LIST_ITEMS
        }
        if oversized_lists or len(preview) > _MAX_STAGED_MUTATION_PREVIEW_CHARS:
            return {
                "success": False,
                "policyBlocked": True,
                "message": (
                    "安全拦截：拟执行内容过大，无法在一条确认消息中完整展示。"
                    "请拆成更小批次后重新发送；本次未保存票据，也未写入。"
                ),
                "previewCharacters": len(preview),
                "oversizedLists": oversized_lists,
            }
        preview_digest = hashlib.sha256(preview.encode("utf-8")).hexdigest()[:16]
        return {
            "success": False,
            "requiresConfirmation": True,
            "localConfirmationRequired": True,
            "message": (
                f"拟执行 {tool_name}：{preview}\n"
                f"参数摘要：SHA-256 {preview_digest}\n"
                "尚未写入。请核对后按机器人给出的确认票据继续。"
            ),
        }

    @staticmethod
    def _validate_current_message_binding(
        tool_name: str,
        arguments: Dict,
        context: ToolContext,
    ) -> Optional[Dict]:
        """Bind model-generated mutation targets to entities in current raw text."""
        message = _mutation_authorization_view(context.current_message or "")
        trusted_codes = context.trusted_codes_by_word or {}
        trusted_draft_words = context.trusted_draft_words_by_id or {}
        trusted_draft_items = context.trusted_draft_items_by_id or {}
        compact_message = re.sub(r"[\s，,。.!！?？~～]+", "", message)
        if tool_name == "keytao_submit_batch":
            allowed = {
                "提交", "提审", "送审", "提交草稿", "提交批次", "提交审核",
                "提交当前草稿", "提交这个草稿", "发起审核",
            }
            candidate = compact_message
            if candidate.startswith("请"):
                candidate = candidate[1:]
            for suffix in ("一下", "吧", "啦", "了"):
                if candidate.endswith(suffix):
                    candidate = candidate[:-len(suffix)]
            if candidate not in allowed:
                return {
                    "success": False,
                    "policyBlocked": True,
                    "requiresTextFollowUp": True,
                    "message": "安全拦截：提交只能由本轮明确、独立的提交指令授权。",
                }
        if tool_name == "keytao_recall_batch":
            recall_candidate = compact_message
            if recall_candidate.startswith("请"):
                recall_candidate = recall_candidate[1:]
            if recall_candidate not in {
                "撤回", "撤回批次", "撤回最近批次", "撤回最近提交", "召回最近批次",
            }:
                return {
                    "success": False,
                    "policyBlocked": True,
                    "requiresTextFollowUp": True,
                    "message": "安全拦截：撤回只能由本轮明确、独立的撤回指令授权。",
                }
        if tool_name == "keytao_remove_draft_item":
            pr_id = str(arguments.get("pr_id") or "").strip()
            id_bound = bool(
                pr_id
                and _contains_exact_target(message, pr_id)
                and _action_is_bound_to_target(message, pr_id, "Delete")
                and not _is_word_protected(message, pr_id)
            )
            word_bound = _draft_item_reference_is_bound(
                message,
                pr_id,
                trusted_draft_words,
                trusted_draft_items,
            )
            if (
                not pr_id
                or not (id_bound or word_bound)
            ):
                return {
                    "success": False,
                    "policyBlocked": True,
                    "requiresTextFollowUp": True,
                    "message": "安全拦截：删除条目的 ID 或词条未精确出现在用户本轮原始文字中。",
                }
        if tool_name == "keytao_batch_remove_draft_items":
            ids = arguments.get("ids")
            missing_ids = [
                str(item) for item in ids or []
                if not (
                    (
                        _contains_exact_target(message, str(item))
                        and _action_is_bound_to_target(
                            message,
                            str(item),
                            "Delete",
                        )
                        and not _is_word_protected(message, str(item))
                    )
                    or (
                        _draft_item_reference_is_bound(
                            message,
                            str(item),
                            trusted_draft_words,
                            trusted_draft_items,
                        )
                    )
                )
            ] if isinstance(ids, list) else ["(missing ids)"]
            if missing_ids or not re.search(r"(?:删除|删掉|移除|只保留|仅保留)", message):
                return {
                    "success": False,
                    "policyBlocked": True,
                    "requiresTextFollowUp": True,
                    "message": "安全拦截：批量删除 ID 未全部出现在用户本轮原始文字中。",
                    "unboundIds": missing_ids[:12],
                }
        if tool_name == "keytao_create_phrase":
            word = str(arguments.get("word") or "").strip()
            code = str(arguments.get("code") or "").strip()
            action = str(arguments.get("action") or "Create")
            old_word = str(arguments.get("old_word") or "").strip()
            targets = [word]
            if action == "Change" and old_word:
                targets.append(old_word)
            missing_targets = [
                target for target in targets
                if not target or not _contains_exact_target(message, target)
            ]
            if (
                missing_targets
                or not _action_is_bound_to_target(message, word, action)
                or any(_is_word_protected(message, target) for target in targets)
                or not _code_is_bound_to_target(
                    message,
                    word,
                    code,
                    trusted_codes.get(word, frozenset()),
                )
                or (
                    action == "Change"
                    and not _change_transition_is_bound(message, old_word, word)
                )
                or (
                    action in {"Change", "Delete"}
                    and not (
                        _extract_phrase_type_for_target(
                            message,
                            old_word if action == "Change" and old_word else word,
                        )
                        or ToolExecutor._trusted_phrase_type(
                            context,
                            old_word if action == "Change" and old_word else word,
                            code,
                        )
                    )
                )
            ):
                return {
                    "success": False,
                    "policyBlocked": True,
                    "requiresTextFollowUp": True,
                    "message": (
                        f"安全拦截：{action} 操作的动作、词条或编码"
                        "未与用户本轮原始文字中的完整目标绑定。"
                    ),
                    "unboundTargets": missing_targets[:12],
                }
        elif tool_name == "keytao_batch_add_to_draft":
            items = arguments.get("items")
            if not isinstance(items, list) or not items:
                return {
                    "success": False,
                    "policyBlocked": True,
                    "message": "安全拦截：批量操作缺少可绑定的词条。",
                }
            blocked_items: List[str] = []
            for item in items:
                if not isinstance(item, dict):
                    blocked_items.append("(invalid item)")
                    continue
                word = str(item.get("word") or "").strip()
                old_word = str(item.get("old_word") or "").strip()
                code = str(item.get("code") or "").strip()
                action = str(item.get("action") or "Create")
                targets = [word]
                if action == "Change" and old_word:
                    targets.append(old_word)
                if (
                    any(not target or not _contains_exact_target(message, target) for target in targets)
                    or not _action_is_bound_to_target(message, word, action)
                    or any(_is_word_protected(message, target) for target in targets)
                    or not _code_is_bound_to_target(
                        message,
                        word,
                        code,
                        trusted_codes.get(word, frozenset()),
                    )
                    or (
                        action == "Change"
                        and not _change_transition_is_bound(message, old_word, word)
                    )
                    or (
                        action in {"Change", "Delete"}
                        and not (
                            _extract_phrase_type_for_target(
                                message,
                                old_word if action == "Change" and old_word else word,
                            )
                            or ToolExecutor._trusted_phrase_type(
                                context,
                                old_word if action == "Change" and old_word else word,
                                code,
                            )
                        )
                    )
                ):
                    blocked_items.append(f"{action}:{word or '(missing word)'}")
            if blocked_items:
                return {
                    "success": False,
                    "policyBlocked": True,
                    "requiresTextFollowUp": True,
                    "message": (
                        "安全拦截：批量操作中每一条的动作、词条和编码"
                        "都必须与用户本轮原始文字单独绑定。"
                    ),
                    "unboundItems": blocked_items[:12],
                }
        elif tool_name == "keytao_shift_phrase_code":
            word = str(arguments.get("word") or "").strip()
            target_code = str(arguments.get("target_code") or "").strip()
            if (
                not word
                or not _contains_exact_target(message, word)
                or not _action_is_bound_to_target(message, word, "Change")
                or _is_word_protected(message, word)
                or bool(re.search(_PROTECTED_WORD_RE, message))
                or not _code_is_bound_to_target(
                    message,
                    word,
                    target_code,
                    trusted_codes.get(word, frozenset()),
                )
            ):
                return {
                    "success": False,
                    "policyBlocked": True,
                    "requiresTextFollowUp": True,
                    "message": "安全拦截：顺延操作的词条或目标编码未精确绑定。",
                }
        return None
