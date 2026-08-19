"""Render deterministic chat replies and post-process platform-safe text."""

import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from nonebot.log import logger

from ..harness.state import ActiveDraftOperation, PendingAddWord, PendingState
from ..utils import review_flags
from ..utils import http_client
from ..utils.pending_confirmation import (
    _BIND_HELP_TEXT,
    _humanize_warning_text,
    pending_confirmation_copy,
    plain_warning_message,
    render_executable_suggestion,
    render_remediation_reply,
    single_word_candidate_footer,
)


def _strip_markdown(text: str) -> str:
    """Remove markdown syntax for plain-text platforms (QQ)."""
    text = re.sub(r'```[\w]*\n?(.*?)```', lambda m: m.group(1).strip(), text, flags=re.DOTALL)
    text = re.sub(r'`([^`\n]+)`', r'\1', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'__(.*?)__', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_((?!\s).*?(?<!\s))_', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


_MV2_RE = re.compile(r'([\\_%*\[\]()~`>#+\-=|{}.!])')


def _escape_mv2_segment(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2 in a plain-text segment."""
    return _MV2_RE.sub(r'\\\1', text)


def _to_markdownv2(text: str) -> str:
    """Convert common markdown to Telegram MarkdownV2."""
    result: list[str] = []
    last = 0
    for m in re.finditer(r'```[\w]*\n?.*?```|`[^`\n]+`', text, re.DOTALL):
        result.append(_escape_mv2_segment(text[last:m.start()]))
        result.append(m.group())
        last = m.end()
    result.append(_escape_mv2_segment(text[last:]))
    return ''.join(result)


def _telegram_utf16_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _split_telegram_text(text: str, limit: int = 4000) -> List[str]:
    """Split plain text without exceeding Telegram's UTF-16 message limit."""
    if not text:
        return [""]
    chunks: List[str] = []
    current: List[str] = []
    current_units = 0
    last_break_index = -1
    for character in text:
        units = _telegram_utf16_units(character)
        if current and current_units + units > limit:
            if last_break_index >= 0:
                split_at = last_break_index + 1
                chunk = "".join(current[:split_at]).rstrip()
                remainder = current[split_at:]
                chunks.append(chunk)
                current = remainder
                current_units = _telegram_utf16_units("".join(current))
            else:
                chunks.append("".join(current))
                current = []
                current_units = 0
            last_break_index = max(
                (index for index, value in enumerate(current) if value in "\n "),
                default=-1,
            )
        current.append(character)
        current_units += units
        if character in "\n ":
            last_break_index = len(current) - 1
    if current:
        chunks.append("".join(current).rstrip())
    return [chunk for chunk in chunks if chunk] or [""]


_RAW_PYTHON_REPLY_MARKERS = ("{'", "': '", "dataclass(")
_RETIRED_NO_COMMAND_COPY = "当前没有可安全执行" + "的后续命令"

_DEFAULT_PUBLIC_BASE_BY_PLATFORM = {
    "qq": "https://keytao.rea.ink",
    "telegram": "https://keytao.vercel.app",
    "web": "https://keytao.vercel.app",
    "web-anon": "https://keytao.vercel.app",
}
_PUBLIC_BASE_CONFIG_BY_PLATFORM = {
    "qq": ("keytao_public_base_qq", "KEYTAO_PUBLIC_BASE_QQ"),
    "telegram": (
        "keytao_public_base_telegram",
        "KEYTAO_PUBLIC_BASE_TELEGRAM",
    ),
    "web": ("keytao_public_base_web", "KEYTAO_PUBLIC_BASE_WEB"),
    "web-anon": ("keytao_public_base_web", "KEYTAO_PUBLIC_BASE_WEB"),
}


def public_base_for_platform(platform: str) -> str:
    """Return the configured public KeyTao base for one reply platform."""
    normalized = str(platform or "").strip().lower()
    default = _DEFAULT_PUBLIC_BASE_BY_PLATFORM.get(
        normalized,
        _DEFAULT_PUBLIC_BASE_BY_PLATFORM["web"],
    )
    attr_name, env_name = _PUBLIC_BASE_CONFIG_BY_PLATFORM.get(
        normalized,
        _PUBLIC_BASE_CONFIG_BY_PLATFORM["web"],
    )
    configured = str(http_client.config_value(attr_name, env_name, default) or "").strip()
    parsed = urlsplit(configured)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return default
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def render_platform_public_links(text: str, platform: str) -> str:
    """Render KeyTao public links on the configured host for one platform."""
    public_base = urlsplit(public_base_for_platform(platform))
    public_hosts = {
        "keytao.rea.ink",
        "www.keytao.rea.ink",
        "keytao.vercel.app",
        "www.keytao.vercel.app",
    }

    def replace_url(match: re.Match) -> str:
        source = urlsplit(match.group(0))
        if source.hostname not in public_hosts and "/batch/" not in source.path:
            return match.group(0)
        return urlunsplit((
            public_base.scheme,
            public_base.netloc,
            source.path,
            source.query,
            source.fragment,
        ))

    return re.sub(r"https?://[^\s)\]]+", replace_url, str(text or ""))


_BARE_BATCH_UUID_RE = re.compile(
    r"(?<![0-9A-Fa-f])"
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
    r"(?![0-9A-Fa-f])"
)


def strip_bare_batch_ids(text: str) -> str:
    """Hide UUID-shaped batch identifiers everywhere except inside links."""
    urls: List[str] = []

    def stash_url(match: re.Match) -> str:
        urls.append(match.group(0))
        return f"\x00KEYTAO_URL_{len(urls) - 1}\x00"

    stashed = re.sub(r"https?://[^\s)\]]+", stash_url, str(text or ""))
    lines: List[str] = []
    for line in stashed.splitlines():
        cleaned = _BARE_BATCH_UUID_RE.sub("", line)
        cleaned = re.sub(r"(?:关联批次|批次(?:ID|编号)?)[：:]\s*$", "", cleaned)
        cleaned = re.sub(r"批次\s+的", "批次的", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).rstrip()
        if cleaned.strip(" \t-•，,。；;：:"):
            lines.append(cleaned)
    rendered = "\n".join(lines)
    for index, url in enumerate(urls):
        rendered = rendered.replace(f"\x00KEYTAO_URL_{index}\x00", url)
    return rendered


_INTERNAL_REPLY_FRAGMENT_RE = re.compile(
    r"(?:\bboundTarget\b|\bblockReason\b|\bbinding_incomplete\b|"
    r"[（(]\s*缺少\s*[：:]\s*[^）)]*"
    r"(?:[a-z]+[A-Z_][A-Za-z0-9_]*|[a-z]+_[a-z0-9_]+)[^）)]*[）)])"
)


def _assert_plain_user_facing_reply(text: str) -> str:
    reply = str(text or "")
    if _RETIRED_NO_COMMAND_COPY in reply:
        logger.error("Refusing retired user-facing copy")
        raise ValueError("User-facing reply contains retired user-facing copy")
    marker = next(
        (candidate for candidate in _RAW_PYTHON_REPLY_MARKERS if candidate in reply),
        "",
    )
    if marker:
        logger.error("Refusing user-facing reply with raw Python repr marker %r", marker)
        raise ValueError("User-facing reply contains a raw Python representation")
    if _INTERNAL_REPLY_FRAGMENT_RE.search(reply):
        logger.error("Refusing user-facing reply with an internal policy identifier")
        raise ValueError("User-facing reply contains a raw Python representation")
    return reply


def _plain_warning_message(warning: Any) -> str:
    return plain_warning_message(warning)


def _plain_warning_line(warning: Any) -> str:
    return _assert_plain_user_facing_reply(f"⚠️ {_plain_warning_message(warning)}")


def _format_full_add_and_submit_instruction(
    state: Optional[PendingAddWord] = None,
    *,
    quoted: bool = False,
    referenced_words: Tuple[str, ...] = (),
) -> str:
    """Explain how to bind an add-and-submit command without a native quote."""
    if not isinstance(state, PendingAddWord):
        words = tuple(dict.fromkeys(
            str(word or "").strip()
            for word in referenced_words
            if str(word or "").strip()
        ))
        if quoted:
            return render_remediation_reply(
                "已检测到你引用了机器人消息，但这条消息对应的可执行候选状态"
                "不存在（可能已过期，或发送时未建立）；本次不会写入",
                command=("加词 " + " ".join(words)) if words else "",
                words=words,
            )
        return render_remediation_reply(
            "当前没有可执行候选状态，本次不会写入；"
            "这条消息也没有提供可绑定的具体词条"
        )

    if quoted:
        reason = (
            "引用已经选中了当前候选，但它缺少可核验的服务端候选快照；"
            "系统不能把引用文字当作词条或编码，本次未写入"
        )
    else:
        reason = (
            "当前候选缺少可核验的服务端候选快照；"
            "系统不能用这条回复补造词条或编码，本次未写入"
        )
    return render_remediation_reply(
        reason,
        command=f"加词 {state.word}",
        words=(state.word,),
    )


def _normalize_generated_review_copy(response: str) -> str:
    """Normalize model-generated review status text to the deterministic UI wording."""
    text = str(response or "")
    replacements = (
        ("自动审核：预计需管理员审核", "自动审核：该词需管理员审核"),
        ("自动审核:预计需管理员审核", "自动审核：该词需管理员审核"),
        ("自动审核：预计需要管理员审核", "自动审核：该词需管理员审核"),
        ("自动审核:预计需要管理员审核", "自动审核：该词需管理员审核"),
        ("自动审核：预计可通过", "自动审核：该词可自动通过"),
        ("自动审核:预计可通过", "自动审核：该词可自动通过"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return re.sub(
        r"[；;]\s*提交整批时会(?:重新审核|重审|复审)",
        "",
        text,
    )


def _format_active_draft_operation_message(
    operation: ActiveDraftOperation,
    pending_state: PendingState = None,
) -> str:
    """Explain why another mutation cannot start without consuming its pending state."""
    phase = "正等待你的确认" if operation.status == "awaiting_confirmation" else "正在后台处理"
    if isinstance(pending_state, PendingAddWord) and pending_state.word != operation.word:
        full_command = (
            f"添加 {pending_state.word} {pending_state.recommended_code} 并提交"
        )
        suggestion = render_executable_suggestion(
            full_command,
            words=(pending_state.word,),
        )
        return (
            f"上一批 {operation.description} {phase}，为避免两个批次写进同一份草稿，"
            f"本喵暂时不会操作「{pending_state.word}」。\n"
            f"「{pending_state.word}」的候选仍为你保留；上一批结束后请复制发送：\n"
            f"{suggestion}\n或引用候选消息回复「添加并提交」。"
        )
    message = (
        f"{operation.description} {phase}，不用重复发送。"
        "本喵完成后会直接回复最终结果。"
    )
    if operation.status == "awaiting_confirmation":
        message += f"\n{pending_confirmation_copy()}"
    return message


def _format_phrase_lookup_brief(phrase: Dict) -> str:
    code = str(phrase.get("code") or "").strip()
    type_label = str(phrase.get("type_label") or phrase.get("type") or "词条").strip()
    weight = phrase.get("weight")
    pieces = [code or "无编码", type_label]
    if weight is not None:
        pieces.append(f"权重 {weight}")

    duplicate_info = phrase.get("duplicate_info")
    if isinstance(duplicate_info, dict):
        position_label = str(duplicate_info.get("position_label") or "").strip()
        if position_label:
            pieces.append(f"同码{position_label}")
    return "（" + "，".join(pieces) + "）"


def _format_referenced_word_presence_response(words: List[str], lookup_data: Dict) -> str:
    results = {
        str(item.get("word") or "").strip(): item
        for item in lookup_data.get("results", [])
        if isinstance(item, dict) and str(item.get("word") or "").strip()
    }
    lines = [f"查的是你引用那条消息里的：{'、'.join(f'「{word}」' for word in words)}。", ""]
    all_found = True

    for word in words:
        phrases = results.get(word, {}).get("phrases", [])
        if phrases:
            briefs = "；".join(
                _format_phrase_lookup_brief(phrase)
                for phrase in phrases[:4]
                if isinstance(phrase, dict)
            )
            lines.append(f"• 「{word}」：已收录 {briefs}")
        else:
            all_found = False
            lines.append(f"• 「{word}」：当前词库未收录")

    if all_found:
        lines.append("")
        lines.append("结论：这些词当前都在词库里。")
    else:
        lines.append("")
        lines.append("结论：不是全部都在词库里，未收录的可以再让本喵按读音和编码候选走加词流程。")
    return "\n".join(lines)


def _format_encode_char_split(chars: object) -> List[str]:
    if not isinstance(chars, list):
        return []

    lines: List[str] = []
    for item in chars:
        if not isinstance(item, dict):
            continue
        char = str(item.get("char") or "").strip()
        pinyin = str(item.get("pinyin") or "").strip()
        phonetic_code = str(item.get("phoneticCode") or "").strip()
        shape_code = str(item.get("shapeCode") or "").strip()
        root_parts = [
            str(item.get(key) or "").strip()
            for key in ("c1", "c2")
            if str(item.get(key) or "").strip()
        ]

        display_char = f"{char}（{pinyin}）" if pinyin else char
        pieces = [f"• {display_char}"]
        if phonetic_code:
            pieces.append(f"音码 {phonetic_code}")
        if root_parts:
            pieces.append(f"字根 {'｜'.join(root_parts)}")
        if shape_code:
            pieces.append(f"形码 {shape_code}")
        if len(pieces) > 1:
            lines.append("　".join(pieces))

    return lines


def _candidate_statuses_from_encoding(encoding: Dict) -> List[Dict]:
    statuses = [
        status for status in encoding.get("candidateStatuses", [])
        if isinstance(status, dict) and isinstance(status.get("code"), str) and status.get("code")
    ]
    if statuses:
        return statuses

    return [
        {"code": code, "occupied": False, "label": "空位"}
        for code in encoding.get("candidateCodes", [])
        if isinstance(code, str) and code
    ]


def _format_candidate_status_line(index: int, status: Dict, recommended_code: str) -> str:
    code = str(status.get("code") or "").strip()
    occupied = bool(status.get("occupied"))
    if occupied:
        label = str(status.get("label") or "已有占用").strip()
    elif code == recommended_code:
        label = "✅ 推荐（空位）"
    else:
        label = "空位"
    return f"{index}. {code} — {label}"


def _format_tool_encoded_add_prompt(word: str, encoding: Dict) -> Optional[str]:
    statuses = _candidate_statuses_from_encoding(encoding)
    if not statuses:
        return None

    status_codes = [status.get("code", "") for status in statuses]
    recommended_code = str(encoding.get("recommendedCode") or "").strip()
    if not recommended_code or recommended_code not in status_codes:
        first_available = next(
            (str(status.get("code")) for status in statuses if not status.get("occupied")),
            "",
        )
        recommended_code = first_available or str(statuses[0].get("code") or "").strip()
    if not recommended_code:
        return None

    word_type = str(encoding.get("type") or "").strip()
    type_label = word_type or f"{len(word)}字词"
    lines = [
        f"词库暂无收录「{word}」，按工具规则计算如下：",
        "",
        f"「{word}」的键道编码（{type_label}）",
        "",
    ]

    split_lines = _format_encode_char_split(encoding.get("chars"))
    if split_lines:
        lines.extend(["逐字拆分:", *split_lines, ""])

    lines.append("候选编码:")
    lines.extend(
        _format_candidate_status_line(index, status, recommended_code)
        for index, status in enumerate(statuses, start=1)
    )
    lines.extend((
        "",
        f"是否以编码 {recommended_code} 将「{word}」加入草稿？",
        single_word_candidate_footer(len(statuses)),
    ))
    return "\n".join(lines)


def _review_source_label(source: Dict) -> str:
    label = str(source.get("source") or "").strip()
    url = str(source.get("url") or "").strip()
    if label and url:
        return f"{label} {url}"
    return label or url


def _common_known_item_for_code(review: Dict, code: str) -> Optional[Dict]:
    audit = review.get("preSubmitAudit") if isinstance(review, dict) else None
    if not isinstance(audit, dict):
        return None
    word = str(review.get("word") or "").strip()
    for item in audit.get("commonKnownItems") or []:
        if not isinstance(item, dict):
            continue
        item_code = str(item.get("code") or "").strip()
        item_word = str(item.get("word") or "").strip()
        if item_code == code and (not word or not item_word or item_word == word):
            return item
    return None


def _entity_identity_label(entity: Dict) -> str:
    names: List[str] = []
    for value in [*(entity.get("canonicalNames") or []), *(entity.get("aliases") or [])]:
        text = str(value or "").strip()
        if text and text not in names:
            names.append(text)
    return " / ".join(names[:3])


def _common_known_item_label(item: Dict) -> str:
    commonness = item.get("commonness") if isinstance(item.get("commonness"), dict) else {}
    entity = commonness.get("entityKnowledge") if isinstance(commonness.get("entityKnowledge"), dict) else {}
    label = str(entity.get("label") or "").strip()
    if label:
        return label
    item_type = str(item.get("type") or "").strip()
    return {
        "historical_person": "历史人物",
        "celebrity": "明星/公众人物",
        "courtesy_name": "名人字号/别名",
        "stage_name": "艺名/别名",
        "brand": "品牌",
        "product": "产品名",
        "fictional_character": "角色名",
        "place": "地名",
        "organization": "组织/机构名",
        "work": "作品名",
        "technical_term": "专业术语",
        "idiom": "成语/熟语",
        "common_word": "常见词",
    }.get(item_type, "常识实体")


def _clean_review_audit_reason(reason: str) -> str:
    text = str(reason or "").strip()
    replacements = [
        "提交整批时会重新审核；",
        "提交整批时会重新审核",
        "提交整批时会重审；",
        "提交整批时会重审",
        "提交整批时会复审；",
        "提交整批时会复审",
        "提交时会重新审核；",
        "提交时会重新审核",
        "提交后将等待管理员审核；",
        "提交后将等待管理员审核",
        "提交后需管理员审核；",
        "提交后需管理员审核",
        "存在不确定项，提交后等待管理员审核；",
        "存在不确定项，提交后等待管理员审核",
        "提交后等待管理员审核；",
        "提交后等待管理员审核",
        "允许本喵自动通过",
        "可由本喵自动通过",
        "允许自动通过",
        "预计可自动通过",
        "不能自动通过",
    ]
    for old in replacements:
        text = text.replace(old, "")
    text = text.strip("；。 ，,")
    return text


def _format_source_summary(sources: List[Dict]) -> str:
    labels = []
    for source in sources[:3]:
        label = _review_source_label(source)
        if label:
            labels.append(label)
    return "；".join(labels) if labels else "暂无权威页"


def _format_pronunciation_source(pronunciation: Dict) -> str:
    sources = [
        source for source in pronunciation.get("sources", [])
        if isinstance(source, dict)
    ]
    if sources:
        return _format_source_summary(sources)
    return str(pronunciation.get("sourceSummary") or "").strip() or "暂无权威页"


def _format_common_known_brief_reason(item: Optional[Dict], fallback: str) -> str:
    if not isinstance(item, dict):
        return _clean_review_audit_reason(fallback)
    commonness = item.get("commonness") if isinstance(item.get("commonness"), dict) else {}
    entity = commonness.get("entityKnowledge") if isinstance(commonness.get("entityKnowledge"), dict) else {}
    label = _common_known_item_label(item)
    identity = _entity_identity_label(entity)
    if identity:
        return f"本喵识别为{label}（{identity}），编码在候选链中"
    summary = _clean_review_audit_reason(str(item.get("summary") or "").strip())
    if summary:
        return summary
    return _clean_review_audit_reason(fallback) or f"本喵识别为{label}"


def _format_review_candidate_line(
    index: int,
    status: Dict,
    recommended_code: str,
    ordering_recommended_code: str = "",
) -> str:
    code = str(status.get("code") or "").strip()
    occupied = bool(status.get("occupied"))
    if occupied:
        label = str(status.get("label") or "已有占用").strip()
        if code == ordering_recommended_code:
            label += " ← 常用度推荐（需重排）"
    elif code == recommended_code:
        label = (
            "空位（不调序备选）"
            if ordering_recommended_code
            else "✅ 推荐（空位）"
        )
    else:
        label = "空位"
    return f"{index}. {code} — {label}"


def _format_candidate_ordering_assessment(
    assessment: Dict,
    candidate_indexes: Dict[str, int],
) -> str:
    verdict = str(assessment.get("verdict") or "")
    word = str(assessment.get("newWord") or "").strip()
    occupant = str(assessment.get("occupantWord") or "").strip()
    occupant_code = str(assessment.get("occupantCode") or "").strip().lower()
    free_code = str(assessment.get("freeCode") or "").strip().lower()
    if verdict == "front_more_common":
        selector = candidate_indexes.get(occupant_code)
        command = f"{selector} 重新编码" if selector is not None else f"{occupant_code} 重新编码"
        return (
            f"常用度评估：「{word}」较「{occupant}」更常用 → "
            f"建议「{word}」占 {occupant_code}、「{occupant}」顺延"
            f"（回复「{command}」执行）"
        )
    if verdict in {"behind_more_common", "close"}:
        return (
            f"常用度评估：「{occupant}」不弱于「{word}」，"
            f"维持现有排序，推荐空位 {free_code}"
        )
    return (
        f"常用度评估：「{word}」与「{occupant}」的常用度信号不足，"
        f"按空位 {free_code} 推荐"
    )


def _format_pre_submit_audit_preview(review: Dict, recommended_code: str) -> Optional[str]:
    audit = review.get("preSubmitAudit") if isinstance(review, dict) else None
    if not isinstance(audit, dict):
        return None

    summary = str(audit.get("summary") or "").strip()
    if audit.get("autoApprove"):
        semantic_items = [
            item
            for item in (audit.get("semanticContextAutoPassItems") or [])
            if isinstance(item, dict)
        ]
        if semantic_items and not audit.get("llmFallback"):
            basis_line = str(semantic_items[0].get("basisLine") or "").strip()
            if basis_line:
                return f"自动审核：{basis_line}"
            reason = "语境读音、具体含义和非生僻证据一致"
        elif audit.get("llmFallback"):
            reason = "语言常识、读音、编码和同码链检查一致"
        elif audit.get("commonKnownItems"):
            common_item = _common_known_item_for_code(review, recommended_code)
            reason = _format_common_known_brief_reason(
                common_item,
                summary or "常见词/实体常识信号和编码候选链一致",
            )
        else:
            reason = _clean_review_audit_reason(summary or "权威来源、编码和常用度证据一致")
        return f"自动审核：该词可自动通过（{reason}）"

    issues = [
        _plain_warning_message(issue).strip()
        for issue in (audit.get("issues") or [])
        if _plain_warning_message(issue).strip()
    ]
    reason = issues[0] if issues else summary or "证据不足"
    reason = _clean_review_audit_reason(reason)
    return f"自动审核：该词需管理员审核（{reason or '证据不足'}）"


def _format_reviewed_add_prompt(review: Dict) -> Optional[str]:
    if not review.get("success"):
        return None
    word = str(review.get("word") or "").strip()
    if word and review.get("pronunciationUnresolved"):
        message = str(review.get("message") or "").strip()
        return message or (
            f"「{word}」存在多音字语境冲突，但暂时无法确定有明确含义支撑的整词读音。"
            "本次不推荐编码，也不会创建待确认加词操作。"
        )
    recommended_code = str(review.get("recommendedCode") or "").strip()
    pronunciations = [
        item for item in review.get("pronunciations", [])
        if isinstance(item, dict) and item.get("candidateStatuses")
    ]
    if not word or not recommended_code or not pronunciations:
        return None

    ordering_assessments = [
        assessment
        for assessment in review.get("candidateOrderingAssessments") or []
        if isinstance(assessment, dict)
    ][:2]
    ordering_recommended_code = next(
        (
            str(assessment.get("occupantCode") or "").strip().lower()
            for assessment in ordering_assessments
            if assessment.get("verdict") == "front_more_common"
        ),
        "",
    )

    lines = [
        f"词库暂无收录「{word}」，先审读音和编码候选：",
        "",
    ]
    candidate_index = 1
    candidate_indexes: Dict[str, int] = {}
    pre_submit_preview = _format_pre_submit_audit_preview(review, recommended_code)
    if len(pronunciations) == 1:
        pronunciation = pronunciations[0]
        pinyin = str(pronunciation.get("pinyin") or "").strip()
        review_parts = [
            f"读音 {pinyin}" if pinyin else "读音待确认",
            f"来源 {_format_pronunciation_source(pronunciation)}",
        ]
        if pre_submit_preview:
            review_parts.append(pre_submit_preview)
        else:
            review_parts.append("自动审核：该词暂未完成预审（当前仅确认读音与候选编码）")
        lines.append("审词：" + "；".join(review_parts))
        lines.append("候选编码:")
        for status in pronunciation.get("candidateStatuses", []):
            code = str(status.get("code") or "").strip().lower()
            candidate_indexes.setdefault(code, candidate_index)
            lines.append(
                _format_review_candidate_line(
                    candidate_index,
                    status,
                    str(pronunciation.get("recommendedCode") or ""),
                    ordering_recommended_code,
                )
            )
            candidate_index += 1
        lines.append("")
    else:
        lines.append("读音与来源:")
        for index, pronunciation in enumerate(pronunciations, start=1):
            pinyin = str(pronunciation.get("pinyin") or "").strip()
            lines.append(f"{index}. {pinyin or '待确认'}；来源 {_format_pronunciation_source(pronunciation)}")
        if pre_submit_preview:
            lines.append(pre_submit_preview)
        else:
            lines.append("自动审核：该词暂未完成预审（当前仅确认读音与候选编码）")
        lines.append("")

        for index, pronunciation in enumerate(pronunciations, start=1):
            pinyin = str(pronunciation.get("pinyin") or "").strip()
            lines.append(f"候选编码（读音 {index}）:")
            for status in pronunciation.get("candidateStatuses", []):
                code = str(status.get("code") or "").strip().lower()
                candidate_indexes.setdefault(code, candidate_index)
                lines.append(
                    _format_review_candidate_line(
                        candidate_index,
                        status,
                        str(pronunciation.get("recommendedCode") or ""),
                        ordering_recommended_code,
                    )
                )
                candidate_index += 1
            lines.append("")

    while lines and not lines[-1]:
        lines.pop()
    if ordering_assessments:
        lines.append("")
        lines.extend(
            _format_candidate_ordering_assessment(assessment, candidate_indexes)
            for assessment in ordering_assessments
        )
    lines.append("")
    if ordering_recommended_code:
        lines.append(
            f"如不调整现有排序，是否仍以编码 {recommended_code} 将「{word}」加入草稿？"
        )
    else:
        lines.append(
            f"是否以编码 {recommended_code} 将「{word}」加入草稿？"
        )
    lines.append(single_word_candidate_footer(candidate_index - 1))
    occupied_choice = next(
        (
            (
                candidate_indexes.get(str(status.get("code") or "").strip().lower()),
                next(
                    (
                        str(phrase.get("word") or "").strip()
                        for phrase in status.get("phrases") or []
                        if isinstance(phrase, dict)
                        and str(phrase.get("word") or "").strip()
                    ),
                    "",
                ),
            )
            for pronunciation in pronunciations
            for status in pronunciation.get("candidateStatuses", [])
            if isinstance(status, dict) and status.get("occupied") is True
        ),
        None,
    )
    if occupied_choice and occupied_choice[0]:
        occupied_index, occupied_word = occupied_choice
        target_copy = f"已有词「{occupied_word}」" if occupied_word else "该已有词"
        lines.append(
            f"若要挪开{target_copy}，回复“{occupied_index} 重新编码”。"
        )
    return "\n".join(lines).strip()


def _dedupe_authoritative_link_lines(text: str) -> str:
    """Keep the first copy of each result URL across plain and Markdown text."""
    seen_urls: set[str] = set()
    output: List[str] = []
    for line in text.splitlines():
        def replace_url(match: re.Match) -> str:
            url = match.group(0)
            if "/batch/" in urlsplit(url).path:
                return ""
            if url in seen_urls:
                return ""
            seen_urls.add(url)
            return url

        cleaned = re.sub(r"https?://[^\s)\]]+", replace_url, line)
        cleaned = re.sub(r"\[[^\]\n]*\]\(\s*\)", "", cleaned)
        cleaned = re.sub(
            r"\s*(?:草稿地址|批次地址|草稿/批次地址|PR)[：:]?\s*$",
            "",
            cleaned,
        )
        if re.fullmatch(r"\s*[-*+]\s*", cleaned):
            cleaned = ""
        output.append(cleaned.rstrip())
    while output and not output[-1]:
        output.pop()
    return "\n".join(output)


def _canonicalize_authoritative_result_links(
    text: str,
    bundle: Dict[str, str],
    *,
    batch_label: str,
) -> str:
    """Remove stale/duplicate forms and append one canonical trusted bundle."""
    batch_url = bundle.get("batchUrl", "")
    pr_url = bundle.get("prUrl", "")
    current_urls = set(filter(None, (batch_url, pr_url)))
    stale_urls = set(filter(None, bundle.get("_staleUrls", "").splitlines()))
    urls_to_remove = sorted(current_urls | stale_urls, key=len, reverse=True)
    output: List[str] = []
    for line in text.splitlines():
        cleaned = line
        cleaned = re.sub(
            r"https?://[^\s)\]]+",
            lambda match: (
                ""
                if "/batch/" in urlsplit(match.group(0)).path
                else match.group(0)
            ),
            cleaned,
        )
        for url in urls_to_remove:
            escaped_url = re.escape(url)
            cleaned = re.sub(
                rf"\[[^\]\n]*\]\(\s*{escaped_url}\s*\)",
                "",
                cleaned,
            )
            cleaned = cleaned.replace(url, "")
        cleaned = re.sub(r"\[[^\]\n]*\]\(\s*\)", "", cleaned)
        cleaned = re.sub(
            r"\s*(?:草稿地址|批次地址|草稿/批次地址|PR|"
            r"旧\s*PR(?:地址|可见于)?|查看旧\s*PR)[：:]?\s*$",
            "",
            cleaned,
        )
        if re.fullmatch(r"\s*[-*+]\s*", cleaned):
            cleaned = ""
        output.append(cleaned.rstrip())
    while output and not output[-1]:
        output.pop()

    appended_urls: set[str] = set()
    if batch_url:
        if output:
            output.append("")
        output.append(f"{batch_label}：{batch_url}")
        appended_urls.add(batch_url)
    if pr_url and pr_url not in appended_urls:
        if not batch_url and output:
            output.append("")
        output.append(f"PR：{pr_url}")
    return "\n".join(output)


def _trusted_result_url(source: Dict, key: str) -> str:
    if not isinstance(source, dict):
        return ""
    if key == "batchUrl" and source.get("batchIdProvisional") is True:
        return ""
    value = str(source.get(key) or "").strip()
    if len(value) <= 2048 and re.fullmatch(r"https?://[^\s]+", value):
        return value
    return ""


def _capture_trusted_result_links(
    result: Dict[str, Any],
    links: Dict[str, str],
) -> None:
    """Track one internally consistent batch/PR link bundle."""
    stale_urls = set(filter(None, links.get("_staleUrls", "").splitlines()))
    stale_urls.update(filter(None, str(result.get("_staleUrls") or "").splitlines()))
    provisional_batch = result.get("batchIdProvisional") is True
    if provisional_batch:
        links["_provisionalBatch"] = "true"
    batch_url = _trusted_result_url(result, "batchUrl")
    pr_url = _trusted_result_url(result, "prUrl")
    batch_id = (
        "" if provisional_batch else str(result.get("batchId") or "").strip()
    )
    previous_batch_url = links.get("batchUrl", "")
    previous_batch_id = links.get("batchId", "")
    previous_pr_url = links.get("prUrl", "")
    has_previous_batch = bool(previous_batch_id or previous_batch_url)
    has_new_batch = bool(batch_id or batch_url)
    same_by_id = bool(batch_id and previous_batch_id and batch_id == previous_batch_id)
    same_by_url = bool(
        batch_url and previous_batch_url and batch_url == previous_batch_url
    )
    identity_conflict = bool(
        (batch_id and previous_batch_id and batch_id != previous_batch_id)
        or (batch_url and previous_batch_url and batch_url != previous_batch_url)
    )
    changed_batch = False
    if has_new_batch:
        changed_batch = bool(
            (
                has_previous_batch
                and (identity_conflict or not (same_by_id or same_by_url))
            )
            or (previous_pr_url and not has_previous_batch)
        )
    elif pr_url:
        changed_batch = bool(
            (has_previous_batch and pr_url != previous_pr_url)
            or (previous_pr_url and pr_url != previous_pr_url)
        )
    if changed_batch:
        stale_urls.update(filter(None, (previous_batch_url, previous_pr_url)))
        for key in ("batchId", "batchUrl", "prUrl"):
            links.pop(key, None)
    if batch_id:
        links["batchId"] = batch_id
        links.pop("_provisionalBatch", None)
    if batch_url:
        links["batchUrl"] = batch_url
    if pr_url:
        links["prUrl"] = pr_url
    stale_urls.discard(links.get("batchUrl", ""))
    stale_urls.discard(links.get("prUrl", ""))
    if stale_urls:
        links["_staleUrls"] = "\n".join(sorted(stale_urls))
    else:
        links.pop("_staleUrls", None)


def _create_notice_lines(data: Dict) -> List[str]:
    """Render preserved warnings and the authoritative code-chain summary."""
    lines = [
        _plain_warning_line(warning)
        for warning in data.get("warnings") or []
    ]
    ordering_summary = str(data.get("orderingSummary") or "").strip()
    if ordering_summary:
        lines.append(f"同码顺序：{ordering_summary}")
    return lines


def _draft_item_display_line(item: Dict, index: int) -> str:
    action_label = item.get("action_label") or {
        "Create": "新增", "Change": "修改", "Delete": "删除",
    }.get(item.get("action", ""), "")
    display = item.get("display_label") or f"{item.get('word', '')} → {item.get('code', '')}"
    return f"• {index}. {action_label} {display}"


def _append_submit_review_lines(parts: List[str], submit_data: object) -> None:
    if not isinstance(submit_data, dict):
        return
    auto_review = submit_data.get("autoReview")
    approve_result = submit_data.get("autoApproveResult") or {}
    if not isinstance(approve_result, dict):
        approve_result = {}
    if submit_data.get("autoApproved"):
        parts.append(_format_auto_approved_review_line(auto_review))
        return

    if isinstance(auto_review, dict):
        will_auto_approve = review_flags.audit_allows_batch_auto_approve(auto_review)
        if will_auto_approve and submit_data.get("requiresConfirmation"):
            parts.append(
                _format_auto_approved_review_line(auto_review)
                + "确认提交后将尝试自动批准入库。"
            )
            return
        if will_auto_approve and approve_result and not approve_result.get("success"):
            passed_line = _format_auto_approved_review_line(auto_review).rstrip("。")
            reason = str(approve_result.get("message") or "未知原因")
            parts.append(f"{passed_line}，但自动批准未执行：{reason}")
            return
        block_reason = _clean_review_audit_reason(
            review_flags.batch_auto_approve_block_reason(auto_review)
        )
        if block_reason and "管理员审核" not in block_reason and "管理员确认" not in block_reason:
            parts.append(f"本喵审核：该批次需管理员审核（{block_reason}）")
        else:
            parts.append("本喵审核：该批次需管理员审核。")
        issues = auto_review.get("issues") or []
        if issues:
            issue_lines = "\n".join(
                f"• {_plain_warning_message(issue).replace('不能自动通过', '需管理员审核').replace('提交后等待管理员审核', '需管理员审核')}"
                for issue in issues[:5]
            )
            parts.append("需管理员审核：\n" + issue_lines)
    if approve_result and not approve_result.get("success"):
        parts.append(f"自动批准未执行：{approve_result.get('message', '未知原因')}")


def _format_auto_approved_review_line(auto_review: Optional[Dict]) -> str:
    """Describe why an auto-approved batch passed without overstating source certainty."""
    if isinstance(auto_review, dict):
        summary = _clean_review_audit_reason(str(auto_review.get("summary") or ""))
        if (
            auto_review.get("semanticContextAutoPassItems")
            and not auto_review.get("llmFallback")
        ):
            return "本喵审核：语境读音、具体含义、非生僻证据和编码候选链检查通过。"
        if auto_review.get("llmFallback"):
            return "本喵审核：语言常识、读音、编码和同码链检查通过。"
        if auto_review.get("commonKnownItems"):
            return "本喵审核：常见词/实体常识、编码候选链和同码链检查通过。"
        if summary and summary != "证据一致":
            return f"本喵审核：{summary}。"
    return "本喵审核：权威来源、编码和常用度证据一致。"


def _trusted_batch_url(*sources: Dict) -> str:
    """Return one display-safe batch URL from trusted tool responses."""
    for source in sources:
        value = _trusted_result_url(source, "batchUrl")
        if value:
            return value
    return ""


def _trusted_pr_url(*sources: Dict) -> str:
    """Return one display-safe PR URL from trusted tool responses."""
    for source in sources:
        value = _trusted_result_url(source, "prUrl")
        if value:
            return value
    return ""


def _trusted_link_bundle(*sources: Dict) -> Dict[str, str]:
    """Merge fallback sources without crossing batch identities."""
    links: Dict[str, str] = {}
    # Earlier sources are authoritative. Processing fallbacks first lets a
    # higher-priority partial identity either verify or replace the bundle.
    for source in reversed(sources):
        if isinstance(source, dict):
            _capture_trusted_result_links(source, links)
    return links


def _append_batch_url_if_missing(
    text: str,
    *sources: Dict,
    label: str = "草稿地址",
) -> str:
    """Append trusted batch and PR links while keeping each URL once."""
    bundle = _trusted_link_bundle(*sources)
    if not bundle.get("batchUrl") and not bundle.get("prUrl"):
        output = _dedupe_authoritative_link_lines(text)
    else:
        output = _canonicalize_authoritative_result_links(
            text,
            bundle,
            batch_label=label,
        )
    if (
        bundle.get("_provisionalBatch") == "true"
        and not bundle.get("batchUrl")
        and "待确认后生成" not in output
    ):
        separator = "\n\n" if output.rstrip() else ""
        output = output.rstrip() + separator + f"{label}：待确认后生成"
    return output


_OPERATION_MEMORY_PREFIX_RE = re.compile(
    r"^词库操作：(?P<actor>.+?)(?:[（(][^)）]+[）)])?\s+(?P<rest>(?:已提交审核|已加入草稿).*)$"
)


def _format_replace_char_confirmation(
    items: List[Dict],
    old_char: str,
    new_char: str,
) -> str:
    """Describe a staged replacement without mutating the draft."""
    parts = [f"🔁 准备把 {len(items)} 条词条里的「{old_char}」替换为「{new_char}」："]
    for item in items:
        parts.append(f"• {item['old_word']} → {item['word']}（{item['code']}）")
    parts.extend((
        "",
        "确认后我才会把这批修改加入草稿。"
        f"{pending_confirmation_copy()}",
    ))
    return _assert_plain_user_facing_reply("\n".join(parts))


def _format_operation_memory_for_reply(item: Dict) -> str:
    content = str(item.get("content") or "").strip()
    speaker_name = str(item.get("speaker_name") or "").strip()
    match = _OPERATION_MEMORY_PREFIX_RE.match(content)
    if not match:
        return re.sub(r"([^\s（(]+)[（(]\d{4,}[）)]", r"\1", content)

    actor = speaker_name or match.group("actor").strip()
    rest = match.group("rest").strip()
    if not rest:
        return actor
    return f"{actor} {rest}"


def _format_clear_response(had_inflight_draft: bool) -> str:
    response = "好哒～ 对话历史已清空！我们重新开始吧 owo"
    if had_inflight_draft:
        response += (
            "\n\n注意：刚才的草稿操作已经发往服务端，清空对话不能撤销它。"
            "结果可能已经生效；请等当前操作结束后先发「查看草稿」，避免重复操作。"
        )
    return response
