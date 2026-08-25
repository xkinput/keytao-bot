"""Render deterministic chat replies and post-process platform-safe text."""

import json
import re
import unicodedata
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

from nonebot.log import logger

from ..harness.state import (
    ActiveDraftOperation,
    PendingAddWord,
    PendingAdvertisedWordSets,
    PendingState,
    PendingTrustedWordRecord,
    PendingToolConfirm,
    pending_batch_display_pairs,
)
from ..utils import review_flags
from ..utils import http_client
from ..utils.pending_confirmation import (
    _BIND_HELP_TEXT,
    _humanize_warning_text,
    already_existing_word_copy,
    pending_confirmation_copy,
    plain_warning_message,
    front_insert_recommendation_copy,
    render_executable_suggestion,
    render_remediation_reply,
    single_word_candidate_footer,
    validated_front_insert_recommendation,
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
    r"PR#\d+|"
    r"禁止(?:再次|重复)?调用|"
    r"请直接(?:根据|使用)[^\r\n。！？]{0,80}(?:回复用户|继续下一步操作)|"
    r"[（(]\s*缺少\s*[：:]\s*[^）)]*"
    r"(?:[a-z]+[A-Z_][A-Za-z0-9_]*|[a-z]+_[a-z0-9_]+)[^）)]*[）)])"
)

_INTERNAL_TOOL_IDENTIFIERS = (
    "keytao_batch_add_to_draft",
    "keytao_batch_remove_draft_items",
    "keytao_audit_draft_items",
    "keytao_compare_commonness",
    "keytao_create_phrase",
    "keytao_encode",
    "keytao_fetch_docs",
    "keytao_get_batch_preview",
    "keytao_infer_word",
    "keytao_list_draft_items",
    "keytao_lookup_by_code",
    "keytao_lookup_by_codes_batch",
    "keytao_lookup_by_word",
    "keytao_lookup_by_words_batch",
    "keytao_prepare_reviewed_add",
    "keytao_recall_batch",
    "keytao_remove_draft_item",
    "keytao_shift_phrase_code",
    "keytao_submit_batch",
    "keytao_update_draft_item_weight",
)
_NORMALIZED_INTERNAL_TOOL_IDENTIFIERS = tuple(
    re.sub(r"[\W_]+", "", identifier).lower()
    for identifier in _INTERNAL_TOOL_IDENTIFIERS
)


def _reply_has_internal_fragment(text: str) -> bool:
    """Catch policy directives and internal tool IDs despite separator changes."""
    reply = str(text or "")
    if _INTERNAL_REPLY_FRAGMENT_RE.search(reply):
        return True
    normalized = re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", reply)).lower()
    return any(
        identifier in normalized
        for identifier in _NORMALIZED_INTERNAL_TOOL_IDENTIFIERS
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
    if _reply_has_internal_fragment(reply):
        logger.error("Refusing user-facing reply with an internal policy identifier")
        raise ValueError("User-facing reply contains a raw Python representation")
    return reply


def _plain_warning_message(warning: Any) -> str:
    return plain_warning_message(warning)


def _plain_warning_line(warning: Any) -> str:
    return _assert_plain_user_facing_reply(f"⚠️ {_plain_warning_message(warning)}")


def _submit_conflict_snapshot_items(data: Dict) -> List[Dict]:
    """Return the audited draft records carried beside a submit response."""
    direct = data.get("snapshotItems")
    if isinstance(direct, list):
        return [item for item in direct if isinstance(item, dict)]
    auto_review = data.get("autoReview")
    nested = auto_review.get("snapshotItems") if isinstance(auto_review, dict) else None
    return [item for item in nested if isinstance(item, dict)] if isinstance(nested, list) else []


def _submit_conflict_draft_item(
    conflict: Dict,
    snapshot_items: List[Dict],
    used_indexes: set[int],
) -> Dict:
    """Bind one conflict to the best matching audited draft item."""
    code = str(conflict.get("code") or "").strip().lower()
    current = conflict.get("currentPhrase")
    current_word = (
        str(current.get("word") or "").strip()
        if isinstance(current, dict)
        else ""
    )
    suggestions = [
        suggestion
        for suggestion in conflict.get("suggestions", [])
        if isinstance(suggestion, dict)
    ]
    suggestion_words = tuple(dict.fromkeys(
        str(suggestion.get("word") or "").strip()
        for suggestion in suggestions
        if str(suggestion.get("word") or "").strip()
    ))
    draft_hint_words = tuple(dict.fromkeys(
        str(suggestion.get("word") or "").strip()
        for suggestion in suggestions
        if str(suggestion.get("action") or "").strip() in {"Adjust", "Cancel"}
        and str(suggestion.get("word") or "").strip()
    ))
    ranked: List[Tuple[int, int, Dict]] = []
    for index, item in enumerate(snapshot_items):
        if index in used_indexes:
            continue
        item_code = str(item.get("code") or "").strip().lower()
        if code and item_code != code:
            continue
        word = str(item.get("word") or "").strip()
        score = 1
        if word and word in draft_hint_words:
            score += 4
        elif word and word in suggestion_words:
            score += 1
        if word and word == current_word:
            score += 2
        ranked.append((score, -index, item))
    if ranked:
        _score, neg_index, item = max(ranked, key=lambda candidate: candidate[:2])
        used_indexes.add(-neg_index)
        return item

    fallback_word = next(iter(draft_hint_words), "") or next(
        iter(suggestion_words),
        "",
    ) or current_word
    return {
        "action": "",
        "word": fallback_word,
        "code": code or (
            str(current.get("code") or "").strip().lower()
            if isinstance(current, dict)
            else ""
        ),
    }


def _format_submit_conflict_failure(data: Dict) -> str:
    """Explain structured submit conflicts and render record-bound recovery."""
    conflicts = data.get("conflicts")
    if not isinstance(conflicts, list) or not conflicts:
        return ""
    snapshot_items = _submit_conflict_snapshot_items(data)
    used_indexes: set[int] = set()
    message = str(data.get("message") or "批次中存在未解决的冲突，无法提交").strip()
    lines = [f"提交失败：{message}"]
    for index, conflict in enumerate(conflicts, start=1):
        if not isinstance(conflict, dict):
            lines.append(f"{index}. 服务端返回了一条无法识别的冲突记录。")
            continue
        item = _submit_conflict_draft_item(conflict, snapshot_items, used_indexes)
        action = str(item.get("action") or "").strip()
        word = str(item.get("word") or "").strip()
        code = str(item.get("code") or conflict.get("code") or "").strip().lower()
        current = conflict.get("currentPhrase")
        current_word = (
            str(current.get("word") or "").strip()
            if isinstance(current, dict)
            else ""
        )
        current_code = (
            str(current.get("code") or "").strip().lower()
            if isinstance(current, dict)
            else ""
        )
        old_word = str(item.get("oldWord") or "").strip()

        def append_suggestion(
            command: str,
            *,
            words: Tuple[str, ...],
            guidance: str = "",
        ) -> bool:
            suggestion = render_executable_suggestion(command, words=words)
            if not suggestion:
                return False
            lines.append(suggestion)
            return True

        same_word_code_items = [
            snapshot_item
            for snapshot_item in snapshot_items
            if str(snapshot_item.get("word") or "").strip() == word
            and str(snapshot_item.get("code") or "").strip().lower() == code
        ]
        exact_create = bool(
            action == "Create"
            and word
            and code
            and current_word == word
            and current_code == code
        )
        if exact_create:
            lines.append(
                f"{index}. 草稿新增「{word}」→ {code}；"
                f"词库中已存在完全相同的「{current_word}」→ {current_code}，不能重复新增。"
            )
            append_suggestion(
                f"删除草稿里的「{word}」",
                words=(word,),
                guidance="删除这条重复草稿项。",
            )
            continue

        if (
            action == "Change"
            and word
            and code
            and current_word == word
            and current_code == code
        ):
            origin = f"把「{old_word}」改为" if old_word else "修改为"
            lines.append(
                f"{index}. 草稿{origin}「{word}」→ {code}；"
                f"词库中的修改结果「{current_word}」→ {current_code} 已经存在，"
                "这条草稿修改已无须重复提交。"
            )
            append_suggestion(
                f"删除草稿里的「{word}」",
                words=(word,),
                guidance="删除这条已经落地的草稿修改。",
            )
            continue

        if (
            action == "Create"
            and word
            and code
            and not current_word
            and len(same_word_code_items) > 1
        ):
            lines.append(
                f"{index}. 草稿新增「{word}」→ {code}；"
                f"同一批次里已有相同的新增项「{word}」→ {code}，不能重复保留。"
            )
            append_suggestion(
                f"删除草稿里的「{word}」",
                words=(word,),
                guidance="删除重复的草稿项。",
            )
            continue

        occupied_by_other = bool(
            action == "Create"
            and word
            and code
            and current_word
            and current_word != word
            and current_code == code
        )
        if occupied_by_other:
            lines.append(
                f"{index}. 草稿新增「{word}」→ {code}；"
                f"词库中的「{current_word}」→ {current_code} 现在占用这个编码位置。"
            )
            rendered_option = False
            for server_suggestion in conflict.get("suggestions", []):
                if not isinstance(server_suggestion, dict):
                    continue
                suggestion_action = str(server_suggestion.get("action") or "").strip()
                suggestion_word = str(server_suggestion.get("word") or "").strip()
                to_code = str(server_suggestion.get("toCode") or "").strip().lower()
                if not suggestion_word or not to_code or to_code == code:
                    continue
                if suggestion_action == "Move" and suggestion_word == current_word:
                    rendered_option = append_suggestion(
                        f"把「{current_word}」调整到 {to_code}",
                        words=(current_word,),
                        guidance=(
                            "顺延当前占位词，或改用服务端给出的其他编码。"
                            if not rendered_option
                            else ""
                        ),
                    ) or rendered_option
                elif suggestion_action == "Adjust" and suggestion_word == word:
                    rendered_option = append_suggestion(
                        f"删除草稿里的「{word}」，再以编码 {to_code} 添加「{word}」",
                        words=(word,),
                        guidance=(
                            "改用服务端给出的其他编码，或顺延当前占位词。"
                            if not rendered_option
                            else ""
                        ),
                    ) or rendered_option
            if not rendered_option:
                append_suggestion(
                    f"加词 {word}",
                    words=(word,),
                    guidance="重新审词并选择新的空位编码；服务端没有返回可安全照抄的调码位置。",
                )
            continue

        if action == "Change" and word and code:
            if old_word:
                current_fact = (
                    f"；该编码位置现在是「{current_word}」→ {current_code}"
                    if current_word and current_code
                    else ""
                )
                lines.append(
                    f"{index}. 草稿把「{old_word}」修改为「{word}」→ {code}；"
                    f"词库中已找不到原目标「{old_word}」→ {code}"
                    f"{current_fact}，目标已经变化。"
                )
            else:
                lines.append(
                    f"{index}. 草稿修改「{word}」→ {code}；"
                    "这条草稿没有记录原词，无法确认要修改的目标。"
                )
            append_suggestion(
                f"加词 {word}",
                words=(word,),
                guidance="按当前词库状态重新审词。",
            )
            continue

        if (
            action == "Delete"
            and word
            and code
            and (not current_word or current_word != word or current_code != code)
        ):
            current_fact = (
                f"；该编码位置现在是「{current_word}」→ {current_code}"
                if current_word and current_code
                else ""
            )
            lines.append(
                f"{index}. 草稿删除「{word}」→ {code}；"
                f"词库中已找不到要删除的「{word}」→ {code}"
                f"{current_fact}，删除目标已经变化。"
            )
            append_suggestion(
                f"加词 {word}",
                words=(word,),
                guidance="按当前词库状态重新审词。",
            )
            continue

        action_label = {
            "Create": "新增",
            "Change": "修改",
            "Delete": "删除",
        }.get(action, "操作")
        word_label = f"「{word}」" if word else "（词条未随冲突返回）"
        code_label = code or "（编码未随冲突返回）"
        lines.append(
            f"{index}. 草稿{action_label}{word_label}→ {code_label}；"
            "服务端返回了暂时无法进一步识别的冲突，未擅自推断原因。"
        )
        if word:
            append_suggestion(
                f"加词 {word}",
                words=(word,),
                guidance="只按这条草稿记录重新审词，不猜测冲突原因。",
            )
    return _assert_plain_user_facing_reply("\n".join(lines))


def _format_pending_item_line(
    item: Any,
    *,
    separator: str = "→",
    include_metadata: bool = False,
) -> str:
    """Render one word/code fact from a trusted pending-record item."""
    if not isinstance(item, dict):
        return ""
    word = str(item.get("word") or "").strip()
    code = str(item.get("code") or "").strip().lower()
    if not word or not code:
        return ""
    line = f"- 「{word}」{separator} {code}"
    if include_metadata:
        action = {
            "Create": "新增",
            "Change": "修改",
            "Delete": "删除",
        }.get(str(item.get("action") or ""), str(item.get("action") or "操作"))
        phrase_type = {
            "Single": "单字",
            "Phrase": "词组",
            "Supplement": "补充",
            "Symbol": "符号",
            "Link": "链接",
            "CSS": "声笔笔",
            "CSSSingle": "声笔笔单字",
            "English": "英文",
        }.get(str(item.get("type") or ""), str(item.get("type") or "词条"))
        line += f"（{action}/{phrase_type}）"
    return line


def _format_pending_state_details(state: PendingState) -> str:
    """Verbalize only facts persisted in one live pending record."""
    if isinstance(state, PendingAddWord):
        return f"加词「{state.word}」→ {state.recommended_code}"

    if isinstance(state, PendingAdvertisedWordSets):
        groups = [
            "、".join(f"「{word}」" for word in snapshot.words)
            for snapshot in state.snapshots
            if snapshot.words
        ]
        return "候选词：" + "；".join(groups) if groups else "候选词"

    if isinstance(state, PendingTrustedWordRecord):
        return f"上一轮词条「{state.word}」@{state.code}"

    if not isinstance(state, PendingToolConfirm):
        return "待确认"

    args = state.args if isinstance(state.args, dict) else {}
    display = (
        args.get("_pending_display")
        if isinstance(args.get("_pending_display"), dict)
        else {}
    )
    function_name = state.function_name
    lines: List[str] = []
    items: List[Any] = []
    shifted_lines: List[str] = []
    item_separator = "→"
    include_metadata = False

    if function_name == "keytao_batch_add_to_draft":
        collision_replan_line = str(
            display.get("collisionReplanLine") or ""
        ).strip()
        if collision_replan_line:
            lines.append(collision_replan_line)
        lines.append("批量加词：")
        pairs = pending_batch_display_pairs(state)
        items = [{"word": word, "code": code} for word, code in pairs]
    elif function_name == "keytao_create_phrase":
        action_label = {
            "Create": "加词",
            "Change": "修改",
            "Delete": "删除",
        }.get(str(args.get("action") or "Create"), "加词")
        word = str(args.get("word") or "").strip()
        code = str(args.get("code") or "").strip().lower()
        lines.append(
            f"{action_label}「{word}」→ {code}"
            if word and code
            else action_label
        )
    elif function_name == "keytao_submit_batch":
        lines.append("提交草稿：")
        items = (
            display.get("snapshotItems")
            if isinstance(display.get("snapshotItems"), list)
            else []
        )
        if not items:
            lines[0] = "提交草稿"
    elif function_name == "keytao_shift_phrase_code":
        lines.append("顺延调码：")
        shift_plan = (
            display.get("shiftPlan")
            if isinstance(display.get("shiftPlan"), dict)
            else {}
        )
        items = (
            shift_plan.get("items")
            if isinstance(shift_plan.get("items"), list)
            else []
        )
        shifted = (
            shift_plan.get("shifted")
            if isinstance(shift_plan.get("shifted"), list)
            else []
        )
        for shifted_item in shifted:
            if not isinstance(shifted_item, dict):
                continue
            shifted_word = str(shifted_item.get("word") or "").strip()
            from_code = str(shifted_item.get("fromCode") or "").strip().lower()
            to_code = str(shifted_item.get("toCode") or "").strip().lower()
            if shifted_word and from_code and to_code:
                shifted_lines.append(
                    f"- 「{shifted_word}」：{from_code} → {to_code}"
                )
        if not items:
            word = str(args.get("word") or "").strip()
            code = str(args.get("target_code") or "").strip().lower()
            items = [{"word": word, "code": code}] if word and code else []
    elif function_name in {
        "keytao_remove_draft_item",
        "keytao_batch_remove_draft_items",
    }:
        lines.append("删除草稿条目：")
        items = (
            args.get("expected_targets")
            if isinstance(args.get("expected_targets"), list)
            else display.get("targets")
            if isinstance(display.get("targets"), list)
            else []
        )
        item_separator = "@"
        include_metadata = True
        if not items:
            draft_id = str(args.get("draft_id") or args.get("id") or "").strip()
            lines[0] = f"删除草稿条目 {draft_id}" if draft_id else "删除草稿条目"
    elif function_name == "keytao_recall_batch":
        lines.append("撤回当前批次")
        items = (
            display.get("items")
            if isinstance(display.get("items"), list)
            else []
        )
    else:
        lines.append("待确认")

    rendered_items = [
        line
        for line in (
            _format_pending_item_line(
                item,
                separator=item_separator,
                include_metadata=include_metadata,
            )
            for item in items
        )
        if line
    ]
    lines.extend(rendered_items)
    if shifted_lines:
        lines.append("顺延变化：")
        lines.extend(shifted_lines)

    warnings = (
        display.get("warnings")
        if isinstance(display.get("warnings"), list)
        else []
    )
    if warnings:
        lines.append("风险：")
        lines.extend(
            f"- {_plain_warning_message(warning)}"
            for warning in warnings
        )

    batch_url = str(display.get("batchUrl") or "").strip()
    if batch_url:
        lines.append(f"草稿地址：{batch_url}")
    return "\n".join(lines)


def _format_server_bound_confirmation_prompt(state: PendingToolConfirm) -> str:
    """Render the first and only prompt from one sealed server ticket."""
    if state.confirmation_source != "server_warning":
        raise ValueError("Server-bound confirmation prompt requires a server ticket")
    details = _format_pending_state_details(state)
    return _assert_plain_user_facing_reply(
        f"{details}\n{pending_confirmation_copy()}"
    )


def _format_changed_server_confirmation_prompt(
    previous: PendingToolConfirm,
    current: PendingToolConfirm,
) -> str:
    """Explain a changed server ticket before asking for fresh consent."""
    if (
        previous.confirmation_source != "server_warning"
        or current.confirmation_source != "server_warning"
        or previous.function_name != current.function_name
    ):
        raise ValueError("Changed confirmation prompt requires matching server tickets")
    previous_details = re.sub(
        r"\s*\n\s*", "；", _format_pending_state_details(previous)
    )
    current_details = re.sub(
        r"\s*\n\s*", "；", _format_pending_state_details(current)
    )
    lines = [
        "确认内容已变化：",
        f"原：{previous_details}",
        f"现：{current_details}",
        pending_confirmation_copy(),
    ]
    return _assert_plain_user_facing_reply("\n".join(lines))


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
                "引用的候选已过期或不可执行，本次未写入",
                command=("加词 " + " ".join(words)) if words else "",
                words=words,
            )
        return render_remediation_reply(
            "没有可执行候选或具体词条，本次未写入"
        )

    if quoted:
        reason = "引用候选缺少可核验的候选记录，本次未写入"
    else:
        reason = "当前候选缺少可核验的候选记录，本次未写入"
    return render_remediation_reply(
        reason,
        command=f"加词 {state.word}",
        words=(state.word,),
    )


def _normalize_generated_review_copy(response: str) -> str:
    """Normalize model-generated review status text to the deterministic UI wording."""
    text = str(response or "")
    replacements = (
        ("自动审核：预计需管理员审核", "自动审核：需管理员审核"),
        ("自动审核:预计需管理员审核", "自动审核：需管理员审核"),
        ("自动审核：预计需要管理员审核", "自动审核：需管理员审核"),
        ("自动审核:预计需要管理员审核", "自动审核：需管理员审核"),
        ("自动审核：该词需管理员审核", "自动审核：需管理员审核"),
        ("自动审核：预计可通过", "自动审核：可自动通过"),
        ("自动审核:预计可通过", "自动审核：可自动通过"),
        ("自动审核：该词可自动通过", "自动审核：可自动通过"),
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
            f"上一批 {operation.description} {phase}；「{pending_state.word}」暂未处理。\n"
            f"{suggestion}\n"
            "或引用候选消息回复「添加并提交」。"
        )
    message = (
        f"{operation.description} {phase}，请勿重复发送。"
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
    lines: List[str] = []

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
            lines.append(f"• 「{word}」：未收录")
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
        f"词库暂无收录「{word}」：",
        f"键道编码（{type_label}）",
    ]

    split_lines = _format_encode_char_split(encoding.get("chars"))
    if split_lines:
        lines.extend(["逐字拆分:", *split_lines])

    lines.append("候选编码:")
    lines.extend(
        _format_candidate_status_line(index, status, recommended_code)
        for index, status in enumerate(statuses, start=1)
    )
    lines.extend((
        f"• 「{word}」→ {recommended_code}（推荐）",
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


def _compact_review_reason(reason: str) -> str:
    """Remove verdict repetition and internal review jargon from visible reasons."""
    text = _clean_review_audit_reason(reason)
    wrapped_verdict = re.fullmatch(
        r"(?:自动审核[：:]\s*)?(?:该词)?(?:预计)?"
        r"(?:可自动通过|(?:需|需要)管理员(?:审核|确认))"
        r"[（(](?P<reason>.+)[）)]",
        text,
    )
    if wrapped_verdict is not None:
        text = wrapped_verdict.group("reason").strip()
    replacements = (
        ("读音由有明确含义支撑的整词语境判定", "读音有明确含义支撑"),
        ("缺少权威整词读音来源", "缺少读音来源"),
        ("没有权威整词读音来源", "没有读音来源"),
        ("没有权威读音来源", "没有读音来源"),
        ("缺少权威读音来源", "缺少读音来源"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    text = re.sub(
        r"(?:[，,；;]\s*)?(?:该词)?(?:需要|需)管理员(?:审核|确认)[。.]?$",
        "",
        text,
    )
    text = re.sub(
        r"(?:[，,；;]\s*)?(?:该词)?可自动通过[。.]?$",
        "",
        text,
    )
    return text.strip("；;。 ，,")


def _format_source_summary(sources: List[Dict]) -> str:
    labels = []
    for source in sources[:3]:
        label = _review_source_label(source)
        if label:
            labels.append(label)
    return "；".join(labels) if labels else "暂无"


def _format_pronunciation_source(pronunciation: Dict) -> str:
    sources = [
        source for source in pronunciation.get("sources", [])
        if isinstance(source, dict)
    ]
    if sources:
        return _format_source_summary(sources)
    summary = str(pronunciation.get("sourceSummary") or "").strip()
    if not summary or "暂无权威页" in summary:
        return "暂无"
    return summary


def _format_common_known_brief_reason(item: Optional[Dict], fallback: str) -> str:
    if not isinstance(item, dict):
        return _clean_review_audit_reason(fallback)
    commonness = item.get("commonness") if isinstance(item.get("commonness"), dict) else {}
    entity = commonness.get("entityKnowledge") if isinstance(commonness.get("entityKnowledge"), dict) else {}
    label = _common_known_item_label(item)
    identity = _entity_identity_label(entity)
    if identity:
        return f"识别为{label}（{identity}），编码有效"
    summary = _clean_review_audit_reason(str(item.get("summary") or "").strip())
    if summary:
        return summary
    return _clean_review_audit_reason(fallback) or f"识别为{label}"


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
        fallback_selector = candidate_indexes.get(free_code)
        return front_insert_recommendation_copy(
            {
                **assessment,
                "newWord": word,
                "occupantWord": occupant,
                "occupantCode": occupant_code,
                "freeCode": free_code,
            },
            fallback_selector,
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
    if review_flags.read_manual_review_flag(review) is True:
        reason = (
            review_flags.manual_review_reason(review)
            or summary
            or "本词条已封印为人工复核"
        )
        reason = _compact_review_reason(reason) or "本词条已封印为人工复核"
        return f"自动审核：{reason}，需要管理员审核"
    if audit.get("autoApprove"):
        semantic_items = [
            item
            for item in (audit.get("semanticContextAutoPassItems") or [])
            if isinstance(item, dict)
        ]
        if semantic_items and not audit.get("llmFallback"):
            basis_line = str(semantic_items[0].get("basisLine") or "").strip()
            reason = basis_line or "语境读音、具体含义和非生僻证据一致"
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
        reason = _compact_review_reason(reason) or "证据一致"
        return f"自动审核：{reason}，可自动通过"

    issues = [
        _plain_warning_message(issue).strip()
        for issue in (audit.get("issues") or [])
        if _plain_warning_message(issue).strip()
    ]
    reason = issues[0] if issues else summary or "证据不足"
    reason = _compact_review_reason(reason) or "证据不足"
    return f"自动审核：{reason}，需要管理员审核"


def _format_reviewed_add_prompt(review: Dict) -> Optional[str]:
    if not review.get("success"):
        return None
    word = str(review.get("word") or "").strip()
    if word and review.get("pronunciationUnresolved"):
        message = str(review.get("message") or "").strip()
        return message or f"「{word}」读音存在冲突，暂不推荐编码。"
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
    snapshot_candidates: List[Tuple[str, bool]] = []
    snapshot_occupied_words: Dict[str, List[str]] = {}
    for pronunciation in pronunciations:
        for status in pronunciation.get("candidateStatuses", []):
            if not isinstance(status, dict):
                continue
            code = str(status.get("code") or "").strip().lower()
            occupied = status.get("occupied")
            if not code or not isinstance(occupied, bool):
                continue
            snapshot_candidates.append((code, occupied))
            words = [
                str(value or "").strip()
                for value in status.get("words") or []
                if str(value or "").strip()
            ]
            if not words:
                words = [
                    str(phrase.get("word") or "").strip()
                    for phrase in status.get("phrases") or []
                    if isinstance(phrase, dict)
                    and str(phrase.get("word") or "").strip()
                ]
            if occupied and words:
                snapshot_occupied_words.setdefault(code, []).extend(words)
    reorder_recommendation = validated_front_insert_recommendation(
        word,
        snapshot_candidates,
        snapshot_occupied_words,
        ordering_assessments,
    )
    ordering_recommended_code = (
        reorder_recommendation["occupantCode"]
        if reorder_recommendation is not None
        else ""
    )

    exact_existing_codes = tuple(dict.fromkeys(
        str(entry.get("code") or "").strip().lower()
        for entry in review.get("existing") or []
        if isinstance(entry, dict)
        and str(entry.get("word") or "").strip() == word
        and str(entry.get("code") or "").strip()
    ))
    if not exact_existing_codes:
        exact_existing_codes = tuple(dict.fromkeys(
            code
            for code, _occupied in snapshot_candidates
            if word in snapshot_occupied_words.get(code, [])
        ))
    existing_copy = already_existing_word_copy(
        word,
        exact_existing_codes,
        can_choose_other_code=any(
            code not in exact_existing_codes for code, _occupied in snapshot_candidates
        ),
        can_reorder=reorder_recommendation is not None,
    )
    lines = (
        [*existing_copy.splitlines(), f"「{word}」候选编码："]
        if existing_copy
        else [f"词库暂无收录「{word}」："]
    )
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
        lines.append("审词：" + "；".join(review_parts) + "；")
        lines.append(pre_submit_preview or "自动审核：预审未完成")
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
    else:
        lines.append("读音与来源:")
        for index, pronunciation in enumerate(pronunciations, start=1):
            pinyin = str(pronunciation.get("pinyin") or "").strip()
            lines.append(f"{index}. {pinyin or '待确认'}；来源 {_format_pronunciation_source(pronunciation)}")
        if pre_submit_preview:
            lines.append(pre_submit_preview)
        else:
            lines.append("自动审核：预审未完成")

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
    if reorder_recommendation is not None:
        lines.append(
            _format_candidate_ordering_assessment(
                reorder_recommendation,
                candidate_indexes,
            )
        )
    elif ordering_assessments:
        lines.extend(
            _format_candidate_ordering_assessment(assessment, candidate_indexes)
            for assessment in ordering_assessments
        )
    if reorder_recommendation is None:
        lines.append(f"• 「{word}」→ {recommended_code}（推荐）")
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
    if reorder_recommendation is None and occupied_choice and occupied_choice[0]:
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
            parts.append(_format_auto_approved_review_line(auto_review))
            return
        if will_auto_approve and approve_result and not approve_result.get("success"):
            passed_line = _format_auto_approved_review_line(auto_review).rstrip("。")
            reason = str(approve_result.get("message") or "未知原因")
            parts.append(f"{passed_line}，但自动批准未执行：{reason}")
            return
        block_reason = _compact_review_reason(
            review_flags.batch_auto_approve_block_reason(auto_review)
        )
        parts.append(
            f"自动审核：{block_reason}，需管理员审核"
            if block_reason
            else "自动审核：需管理员审核"
        )
        issues = auto_review.get("issues") or []
        if issues:
            issue_lines = [
                _compact_review_reason(_plain_warning_message(issue))
                for issue in issues[:5]
            ]
            parts.extend(
                f"• {line}"
                for line in issue_lines
                if line and line != block_reason
            )
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
            return "自动审核：语境读音、具体含义、非生僻证据和编码候选链一致，可自动通过"
        if auto_review.get("llmFallback"):
            return "自动审核：语言常识、读音、编码和同码链一致，可自动通过"
        if auto_review.get("commonKnownItems"):
            return "自动审核：常见词/实体常识、编码候选链和同码链一致，可自动通过"
        if summary and summary != "证据一致":
            return f"自动审核：{_compact_review_reason(summary)}，可自动通过"
    return "自动审核：权威来源、编码和常用度证据一致，可自动通过"


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
    parts.append(pending_confirmation_copy())
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
    response = "对话历史已清空。"
    if had_inflight_draft:
        response += (
            "\n草稿操作可能已生效；操作结束后请发送「查看草稿」，避免重复。"
        )
    return response
