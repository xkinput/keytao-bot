"""
Keytao Create Skill Tools
键道创建词条工具实现
"""
import asyncio
import copy
import difflib
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
import unicodedata
import httpx
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from nonebot.log import logger

from keytao_bot.utils import http_client, review_flags
from keytao_bot.utils.http_client import KeytaoApiError
from keytao_bot.utils.keytao_encoding import (
    build_alternate_pronunciation_codes,
    build_phrase_pronunciation_codes,
    normalize_contextual_phrase_encoding,
)
from keytao_bot.utils.draft_mutation_store import (
    get_default_draft_mutation_claim_store,
)
from keytao_bot.utils.keytao_review import (
    ReviewHttpConfig,
    audit_draft_items,
    build_review_note,
    can_llm_override_audit_issues,
    fetch_keytao_encode,
    manual_preaudit_issue_for_item,
    prepare_reviewed_word,
)
from keytao_bot.utils.pending_confirmation import _BIND_HELP_TEXT, render_remediation_reply


ACTION_LABELS = {
    "Create": "新增",
    "Change": "修改",
    "Delete": "删除",
}

TYPE_LABELS = {
    "Single": "单字",
    "Phrase": "词组",
    "Supplement": "补充词条",
    "Symbol": "符号",
    "Link": "链接",
    "CSS": "声笔笔",
    "CSSSingle": "声笔笔单字",
    "English": "英文",
}
VALID_PHRASE_TYPES = frozenset(TYPE_LABELS)
PHRASE_TYPE_BASE_WEIGHTS = {
    "Single": 10,
    "Phrase": 100,
    "Supplement": 100,
    "Symbol": 10,
    "Link": 10000,
    "CSS": 100,
    "CSSSingle": 10,
    "English": 100,
}


def _draft_tool_failure(
    reason: str,
    *,
    command: str = "",
    error: Optional[Exception] = None,
    log_context: str = "draft_tool",
) -> str:
    """Render stable user copy while keeping exception details in logs only."""
    if error is not None:
        logger.error(
            f"[{log_context}] {type(error).__name__}: {error}"
        )
    return render_remediation_reply(reason, command=command)


class _SubmitAuditTicketStore:
    """Bounded single-use audit snapshots for exact submit confirmations."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 7200.0,
        max_entries: int = 4096,
        max_entry_bytes: int = 1024 * 1024,
        max_total_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        self._ttl_seconds = max(1.0, float(ttl_seconds))
        self._max_entries = max(1, int(max_entries))
        self._max_entry_bytes = max(1024, int(max_entry_bytes))
        self._max_total_bytes = max(
            self._max_entry_bytes,
            int(max_total_bytes),
        )
        self._entries: OrderedDict[
            tuple[str, str, str, int, str],
            tuple[float, Dict, int, str, str],
        ] = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _key(
        platform: str,
        platform_id: str,
        batch_id: str,
        content_version: int,
        audit_digest: str,
    ) -> tuple[str, str, str, int, str]:
        return (
            str(platform),
            str(platform_id),
            str(batch_id),
            int(content_version),
            str(audit_digest).lower(),
        )

    def _purge_expired(self, now: float) -> None:
        expired_keys = [
            key
            for key, (expires_at, _review, _size_bytes, _state, _generation)
            in self._entries.items()
            if expires_at <= now
        ]
        for key in expired_keys:
            self._pop_key(key)

    def _pop_key(
        self,
        key: tuple[str, str, str, int, str],
    ) -> Optional[tuple[float, Dict, int, str, str]]:
        entry = self._entries.pop(key, None)
        if entry is not None:
            self._total_bytes -= entry[2]
        return entry

    def put(
        self,
        platform: str,
        platform_id: str,
        batch_id: str,
        content_version: int,
        audit_digest: str,
        auto_review: Dict,
    ) -> bool:
        try:
            size_bytes = len(
                json.dumps(
                    auto_review,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        except (TypeError, ValueError):
            return False
        if size_bytes > self._max_entry_bytes:
            return False
        key = self._key(
            platform,
            platform_id,
            batch_id,
            content_version,
            audit_digest,
        )
        with self._lock:
            now = time.monotonic()
            self._purge_expired(now)
            existing = self._entries.get(key)
            if existing is not None and existing[3] == "active":
                return False
            # A fresh server preview may replace a ready ticket or a request
            # whose result was explicitly marked uncertain. It must never
            # unlock a second confirmation while the first POST is in flight.
            self._pop_key(key)
            self._entries[key] = (
                now + self._ttl_seconds,
                copy.deepcopy(auto_review),
                size_bytes,
                "ready",
                secrets.token_hex(16),
            )
            self._total_bytes += size_bytes
            self._entries.move_to_end(key)
            stored = True
            while (
                len(self._entries) > self._max_entries
                or self._total_bytes > self._max_total_bytes
            ):
                evicted_key = next(
                    (
                        candidate_key
                        for candidate_key, candidate in self._entries.items()
                        if candidate[3] != "active"
                    ),
                    None,
                )
                if evicted_key is None:
                    stored = False
                    break
                self._pop_key(evicted_key)
                if evicted_key == key:
                    stored = False
                    break
        return stored

    def claim(
        self,
        platform: str,
        platform_id: str,
        batch_id: str,
        content_version: int,
        audit_digest: str,
    ) -> tuple[str, Optional[Dict], Optional[str]]:
        key = self._key(
            platform,
            platform_id,
            batch_id,
            content_version,
            audit_digest,
        )
        with self._lock:
            now = time.monotonic()
            self._purge_expired(now)
            entry = self._entries.get(key)
            if entry is None:
                return "missing", None, None
            expires_at, auto_review, size_bytes, state, generation = entry
            if state != "ready":
                return "claimed", None, None
            self._entries[key] = (
                expires_at,
                auto_review,
                size_bytes,
                "active",
                generation,
            )
            return "ok", copy.deepcopy(auto_review), generation

    def mark_uncertain(
        self,
        platform: str,
        platform_id: str,
        batch_id: str,
        content_version: int,
        audit_digest: str,
        generation: str,
    ) -> bool:
        key = self._key(
            platform,
            platform_id,
            batch_id,
            content_version,
            audit_digest,
        )
        with self._lock:
            now = time.monotonic()
            self._purge_expired(now)
            entry = self._entries.get(key)
            if entry is None or entry[3] != "active" or entry[4] != generation:
                return False
            expires_at, auto_review, size_bytes, _state, _generation = entry
            self._entries[key] = (
                expires_at,
                auto_review,
                size_bytes,
                "uncertain",
                generation,
            )
            return True

    def consume(
        self,
        platform: str,
        platform_id: str,
        batch_id: str,
        content_version: int,
        audit_digest: str,
        generation: str,
    ) -> bool:
        key = self._key(
            platform,
            platform_id,
            batch_id,
            content_version,
            audit_digest,
        )
        with self._lock:
            now = time.monotonic()
            self._purge_expired(now)
            entry = self._entries.get(key)
            if entry is None or entry[4] != generation:
                return False
            self._pop_key(key)
            return True


_SUBMIT_AUDIT_TICKETS = _SubmitAuditTicketStore()


def _draft_mutation_claims():
    return get_default_draft_mutation_claim_store()


def compute_draft_summary(items: List[Dict]) -> Dict:
    """Compute added/modified/deleted counts from a list of PR items."""
    added = sum(1 for i in items if i.get("action") == "Create")
    modified = sum(1 for i in items if i.get("action") == "Change")
    deleted = sum(1 for i in items if i.get("action") == "Delete")
    return {"added": added, "modified": modified, "deleted": deleted}


def _clean_code_list(codes: object) -> List[str]:
    if not isinstance(codes, list):
        return []
    result: List[str] = []
    seen = set()
    for code in codes:
        if not isinstance(code, str):
            continue
        normalized = code.strip().lower()
        if normalized and "?" not in normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _contains_cjk_text(word: str) -> bool:
    return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', word or ""))


def _infer_phrase_type(word: str, code: str, phrase_type: str = "Phrase") -> str:
    """Mirror keytao-next phrase type inference for bot-side guardrails."""
    if phrase_type in VALID_PHRASE_TYPES and phrase_type != "Phrase":
        return phrase_type

    is_symbol_word = bool(word) and all(
        unicodedata.category(c).startswith(('P', 'S')) for c in word if not c.isspace()
    )
    if (code and code.startswith(';')) or is_symbol_word:
        return "Symbol"
    if re.search(r'https?://|www\.', word or "", re.IGNORECASE):
        return "Link"
    if re.search(r'[a-zA-Z]', word or ""):
        return "English"
    if len(word or "") == 1 and _contains_cjk_text(word):
        return "Single"
    return phrase_type or "Phrase"


CSS_CODE_MAX_LENGTH = 8
GENERAL_CODE_MAX_LENGTH = 32
_ALPHA_CODE_RE = re.compile(r"[a-z]+")
_ALNUM_CODE_RE = re.compile(r"[a-z0-9]+")
_SYMBOL_CODE_RE = re.compile(r";?[a-z0-9]+")
CHAIN_VALIDATED_TYPES = frozenset({"Phrase", "Single"})
CODE_WRITING_ACTIONS = frozenset({"Create", "Change"})
_CODE_SHAPE_RULES: Dict[str, Tuple[re.Pattern, int, int, str]] = {
    "Phrase": (_ALPHA_CODE_RE, 1, GENERAL_CODE_MAX_LENGTH, "纯小写字母"),
    "Single": (_ALPHA_CODE_RE, 1, GENERAL_CODE_MAX_LENGTH, "纯小写字母"),
    "CSS": (_ALPHA_CODE_RE, 1, CSS_CODE_MAX_LENGTH, "纯小写字母"),
    "CSSSingle": (_ALPHA_CODE_RE, 1, CSS_CODE_MAX_LENGTH, "纯小写字母"),
    "Supplement": (_ALPHA_CODE_RE, 1, GENERAL_CODE_MAX_LENGTH, "纯小写字母"),
    "English": (_ALNUM_CODE_RE, 1, GENERAL_CODE_MAX_LENGTH, "小写字母或数字"),
    "Link": (_ALNUM_CODE_RE, 1, GENERAL_CODE_MAX_LENGTH, "小写字母或数字"),
    "Symbol": (_SYMBOL_CODE_RE, 1, GENERAL_CODE_MAX_LENGTH, "可选分号前缀加小写字母或数字"),
}
_DEFAULT_CODE_SHAPE_RULE = (
    _SYMBOL_CODE_RE,
    1,
    GENERAL_CODE_MAX_LENGTH,
    "小写字母、数字或分号前缀",
)


def _should_validate_item_code(item: Dict) -> bool:
    action = str(item.get("action") or "Create")
    if action not in CODE_WRITING_ACTIONS:
        return False
    word = str(item.get("word") or "").strip()
    old_word = str(item.get("oldWord") or item.get("old_word") or "").strip()
    # A same-word Change addresses an existing server record at this code and
    # changes only metadata such as weight. The server preview/confirmation
    # contract validates that record identity; re-running pronunciation-based
    # Create validation here would incorrectly reject legacy or custom codes.
    if action == "Change" and old_word and old_word == word:
        return False
    return bool(
        word
        and str(item.get("code") or "").strip()
    )


def _validate_code_shape(phrase_type: str, code: str) -> Optional[str]:
    pattern, min_length, max_length, charset_label = _CODE_SHAPE_RULES.get(
        phrase_type,
        _DEFAULT_CODE_SHAPE_RULE,
    )
    if not pattern.fullmatch(code):
        return f"编码 {code} 不符合 {phrase_type} 类型的字符集要求（应为{charset_label}）"
    if not min_length <= len(code) <= max_length:
        return (
            f"编码 {code} 长度 {len(code)} 超出 {phrase_type} 类型的合理范围"
            f"（{min_length}-{max_length}）"
        )
    return None


def _stamp_item_review_flag(item: Dict, validation: Optional[Dict] = None) -> Dict:
    explicit = review_flags.read_manual_review_flag(item)
    reason = review_flags.manual_review_reason(item)
    needs_manual_review = bool(explicit)
    if review_flags.remark_indicates_manual_review(item.get("remark")):
        needs_manual_review = True
        reason = reason or "加词预审备注已标记为需管理员审核"
    if isinstance(validation, dict) and validation.get("needsManualReview"):
        needs_manual_review = True
        reason = reason or str(validation.get("manualReviewReason") or "")
    if needs_manual_review:
        review_flags.apply_manual_review_flag(item, True, reason)
    elif explicit is False:
        review_flags.apply_manual_review_flag(item, False, reason)
    return item


def _normalize_draft_item_for_request(item: Dict) -> Dict:
    normalized = dict(item)
    word = normalized.get("word")
    code = normalized.get("code")
    if isinstance(word, str):
        normalized["word"] = word.strip()
    if isinstance(code, str):
        normalized["code"] = code.strip().lower()

    if (
        normalized.get("type") not in VALID_PHRASE_TYPES
        and isinstance(normalized.get("word"), str)
        and isinstance(normalized.get("code"), str)
    ):
        normalized["type"] = _infer_phrase_type(
            normalized["word"],
            normalized["code"],
            "Phrase",
        )
    return _stamp_item_review_flag(normalized)


def _build_encode_candidate_result(
    word: str,
    encode_data: Dict,
    infer_data: Optional[Dict] = None,
    requested_code: Optional[str] = None,
) -> Dict:
    infer_data = infer_data or {}
    codes = _clean_code_list(encode_data.get("codes")) or _clean_code_list(infer_data.get("codes"))
    alt_codes = _clean_code_list(encode_data.get("altCodes")) or _clean_code_list(infer_data.get("altCodes"))
    chars = encode_data.get("chars")
    alternate_pronunciation_codes = build_alternate_pronunciation_codes(chars)
    phrase_pronunciation_codes = build_phrase_pronunciation_codes(chars)
    pronunciation_variants = [*alternate_pronunciation_codes, *phrase_pronunciation_codes]
    alternate_codes = _clean_code_list(
        [
            code
            for variant in pronunciation_variants
            for code in variant.get("codes", [])
            if isinstance(variant, dict)
        ]
    )
    requested_prefix = requested_code.strip().lower() if isinstance(requested_code, str) else ""
    requested_variants = [
        variant
        for variant in pronunciation_variants
        if requested_prefix
        and isinstance(variant, dict)
        and variant.get("phoneticCode") == requested_prefix
    ]
    requested_variant_codes = _clean_code_list([
        code
        for variant in requested_variants
        for code in variant.get("codes", [])
        if isinstance(code, str)
    ])
    requested_candidate_codes = _clean_code_list(
        [
            code
            for code in [
                *requested_variant_codes,
                *alternate_codes,
            ]
            if requested_prefix and (code.startswith(requested_prefix) or code in requested_variant_codes)
        ]
    )
    candidate_codes = _clean_code_list([
        *requested_candidate_codes,
        *codes,
        *alt_codes,
        *alternate_codes,
    ])
    requested_analysis = (
        infer_data.get("requestedCodeAnalysis")
        or encode_data.get("requestedCodeAnalysis")
    )

    if not candidate_codes:
        return {"success": False, "message": f"无法计算「{word}」的候选编码"}

    result = {"success": True, "word": word, "candidateCodes": candidate_codes}
    if requested_analysis:
        result["requestedCodeAnalysis"] = requested_analysis
    if alternate_pronunciation_codes:
        result["alternatePronunciationCodes"] = alternate_pronunciation_codes
    if phrase_pronunciation_codes:
        result["alternatePhrasePronunciationCodes"] = phrase_pronunciation_codes
    if requested_candidate_codes:
        result["requestedCandidateCodes"] = requested_candidate_codes
    return result


def _select_current_phrase(word: str, phrases: List[Dict]) -> Optional[Dict]:
    matching = [phrase for phrase in phrases if phrase.get("word") == word and phrase.get("code")]
    if not matching:
        return None
    return sorted(matching, key=lambda item: (len(item.get("code", "")), item.get("code", "")))[0]


def _ordered_code_occupants(phrases: List[Dict], ignored_words: Optional[set[str]] = None) -> List[Dict]:
    ignored_words = ignored_words or set()
    candidates = [
        phrase for phrase in phrases
        if phrase.get("word") and phrase.get("word") not in ignored_words
    ]
    return sorted(candidates, key=lambda item: (item.get("weight", 0), item.get("word", "")))


def _build_code_shift_plan(
    word: str,
    target_code: str,
    target_candidate_codes: List[str],
    current_phrase: Optional[Dict],
    code_phrase_map: Dict[str, List[Dict]],
    word_candidate_code_map: Dict[str, List[str]],
    target_type: Optional[str] = None,
    target_remark: str = "",
    target_needs_manual_review: Optional[bool] = None,
) -> Dict:
    if target_code not in target_candidate_codes:
        return {
            "success": False,
            "message": f"{target_code} 不是「{word}」的有效候选编码",
        }

    current_code = current_phrase.get("code") if current_phrase else None
    current_type = current_phrase.get("type", "Phrase") if current_phrase else "Phrase"
    resolved_target_type = _infer_phrase_type(
        word,
        target_code,
        str(target_type or current_type or "Phrase"),
    )
    deletes: List[Dict] = []
    target_create: Dict = {
        "action": "Create",
        "word": word,
        "code": target_code,
        "type": resolved_target_type,
    }
    if target_remark:
        target_create["remark"] = target_remark
    if target_needs_manual_review is not None:
        review_flags.apply_manual_review_flag(
            target_create,
            target_needs_manual_review,
        )
    creates: List[Dict] = [target_create]
    shifted: List[Dict] = []
    ignored_words = {word}
    reserved_codes = {target_code}
    occupants_by_code: Dict[str, List[Dict]] = {
        code: _ordered_code_occupants(phrases, ignored_words)
        for code, phrases in code_phrase_map.items()
    }
    queue: List[Dict] = list(occupants_by_code.get(target_code, []))
    occupants_by_code[target_code] = []

    if current_code and current_code != target_code:
        deletes.append({"action": "Delete", "word": word, "code": current_code, "type": current_type or "Phrase"})

    while queue:
        occupant = queue.pop(0)
        occupant_word = occupant.get("word", "")
        probe_code = occupant.get("code", "")
        occupant_codes = word_candidate_code_map.get(occupant_word, [])
        if probe_code not in occupant_codes:
            return {
                "success": False,
                "message": f"无法顺延「{occupant_word}」：当前编码 {probe_code} 不在它自己的候选编码中",
            }

        code_index = occupant_codes.index(probe_code)
        next_code: Optional[str] = None
        for candidate_code in occupant_codes[code_index + 1:]:
            if candidate_code in reserved_codes:
                continue
            next_code = candidate_code
            break
        if not next_code:
            return {
                "success": False,
                "message": f"无法顺延「{occupant_word}」：{probe_code} 之后没有可用候选编码",
            }

        occupant_type = occupant.get("type", "Phrase") or "Phrase"
        deletes.append({"action": "Delete", "word": occupant_word, "code": probe_code, "type": occupant_type})
        creates.append({"action": "Create", "word": occupant_word, "code": next_code, "type": occupant_type})
        shifted.append({
            "word": occupant_word,
            "fromCode": probe_code,
            "toCode": next_code,
            "candidateCodes": occupant_codes,
        })
        reserved_codes.add(next_code)
        evicted = list(occupants_by_code.get(next_code, []))
        if evicted:
            queue.extend(evicted)
            occupants_by_code[next_code] = []

    return {
        "success": True,
        "items": deletes + creates,
        "shifted": shifted,
    }


def _format_preview_text(preview: Dict) -> str:
    """Convert preview API response into a unified-diff text block."""
    changes = preview.get("changes", [])
    if not changes:
        return ""

    def phrase_line(p: Dict) -> str:
        word = p.get("word", "")
        code = p.get("code", "")
        weight = p.get("weight", 0)
        return f"{word:<8} {code:<12} {weight}"

    parts: List[str] = []
    for group in changes:
        phrase_type = group.get("phraseType", "")
        codes = group.get("codes", [])
        before = [phrase_line(p) for p in group.get("before", [])]
        after = [phrase_line(p) for p in group.get("after", [])]

        unified = list(difflib.unified_diff(before, after, n=3, lineterm=""))
        if len(unified) <= 2:
            continue

        parts.append(f"diff {phrase_type}  {', '.join(codes)}")
        parts.extend(unified[2:])  # skip --- / +++ header lines
        parts.append("")

    return "\n".join(parts).strip()


def enrich_pr_item_labels(item: Dict) -> Dict:
    """Add Chinese labels and display_label for action/type fields."""
    enriched_item = dict(item)
    action = enriched_item.get("action")
    phrase_type = enriched_item.get("type")
    word = enriched_item.get("word") or ""
    old_word = enriched_item.get("oldWord")
    code = enriched_item.get("code") or ""
    weight = enriched_item.get("weight")
    conflict_reason = enriched_item.get("conflictReason")

    enriched_item["action_label"] = ACTION_LABELS.get(action, action or "未知")
    enriched_item["type_label"] = TYPE_LABELS.get(phrase_type, phrase_type or "未知")

    weight_str = f"（权重: {weight}）" if weight is not None else ""
    if action == "Change" and old_word:
        display = f"{old_word} → {word} @ {code}{weight_str}"
    elif action == "Delete":
        display = f"{word} @ {code}{weight_str}"
    else:
        display = f"{word} → {code}{weight_str}"
    enriched_item["display_label"] = display

    if conflict_reason:
        enriched_item["warning"] = conflict_reason

    return enriched_item


class UserNotFoundError(Exception):
    pass


def _not_bound_message(platform: str) -> str:
    if platform in ("web", "web-anon"):
        return render_remediation_reply(
            "当前未登录 KeyTao；登录属于站外操作"
        )
    return _BIND_HELP_TEXT


def get_keytao_url() -> str:
    """Return the normalized KeyTao API base URL."""
    return http_client.get_keytao_url()


_UNSAFE_PATH_SEGMENT_RE = re.compile(r"[/?#\\%\s]")
_MAX_PATH_SEGMENT_LENGTH = 128


def _safe_path_segment(value: object) -> Optional[str]:
    """Return ``value`` usable as one URL path segment, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text or len(text) > _MAX_PATH_SEGMENT_LENGTH:
        return None
    if ".." in text or _UNSAFE_PATH_SEGMENT_RE.search(text):
        return None
    return text


def _safe_numeric_id(value: object) -> Optional[int]:
    """Coerce a caller-supplied record id to a positive int, or ``None``."""
    if isinstance(value, bool):
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def make_batch_url(batch_id: str) -> str:
    """Build a web URL for a draft batch."""
    safe_batch_id = _safe_path_segment(batch_id)
    if not safe_batch_id:
        logger.warning(f"[make_batch_url] refusing unsafe batch id: {batch_id!r}")
        return get_keytao_url()
    return f"{get_keytao_url()}/batch/{safe_batch_id}"


def _batch_url_for_result(data: Dict) -> str:
    """Return a URL only when the result names a materialized batch."""
    if not isinstance(data, dict) or data.get("batchIdProvisional") is True:
        return ""
    batch_id = data.get("batchId")
    return make_batch_url(batch_id) if batch_id else ""


def _inject_batch_url(data: Dict) -> Dict:
    """Inject batchUrl into any response dict that contains a batchId."""
    batch_url = _batch_url_for_result(data)
    if not batch_url:
        data.pop("batchUrl", None)
        if data.get("batchIdProvisional") is True:
            data["batchUrlStatus"] = "待确认后生成"
        return data
    data["batchUrl"] = batch_url
    return data


def _mark_provisional_batch(data: Dict) -> Dict:
    """Mark an absence-CAS preview without publishing its unborn batch URL."""
    data["batchIdProvisional"] = True
    data.pop("batchUrl", None)
    data["batchUrlStatus"] = "待确认后生成"
    return data


def _inject_known_batch_url(data: Dict, batch_id: Optional[str]) -> Dict:
    """Preserve the request-bound batch on every definitive/uncertain result."""
    if batch_id:
        data.setdefault("batchId", batch_id)
    return _inject_batch_url(data)


def get_bot_token() -> Optional[str]:
    """Return the shared bot token."""
    return http_client.get_bot_token()


def get_bot_identity_secret() -> Optional[str]:
    """Get the dedicated cross-service identity HMAC secret."""
    try:
        from nonebot import get_driver
        config = get_driver().config
        return getattr(config, "bot_identity_secret", None)
    except Exception:
        return None


def _json_request_body(payload: Dict) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _parse_json_mapping(value: object) -> Dict[str, str]:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items() if v}
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        data = json.loads(value)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if v}


def get_user_api_key(platform: str, platform_id: str) -> Optional[str]:
    """Return the API key bound to exactly this platform account."""
    return http_client.get_user_api_key(platform, platform_id)


def get_bot_headers(
    platform: Optional[str] = None,
    platform_id: Optional[str] = None,
    content_type: bool = False,
    method: str = "",
    path: str = "",
    raw_body: bytes = b"",
) -> Dict[str, str]:
    token = get_bot_token()
    headers: Dict[str, str] = {}
    if token:
        headers["X-Bot-Token"] = token
    if content_type:
        headers["Content-Type"] = "application/json"

    if platform and platform_id:
        user_api_key = get_user_api_key(platform, platform_id)
        if user_api_key:
            headers["X-API-Key"] = user_api_key

    if platform == "web" and platform_id and method and path:
        identity_secret = get_bot_identity_secret()
        if not identity_secret:
            raise RuntimeError("BOT_IDENTITY_SECRET is not configured")
        timestamp = str(int(time.time()))
        nonce = secrets.token_hex(16)
        body_digest = hashlib.sha256(raw_body).hexdigest()
        canonical = "\n".join((
            method.upper(),
            path,
            platform,
            platform_id,
            timestamp,
            nonce,
            body_digest,
        )).encode("utf-8")
        headers["X-Bot-User-Ts"] = timestamp
        headers["X-Bot-User-Nonce"] = nonce
        headers["X-Bot-User-Sig"] = hmac.new(
            identity_secret.encode("utf-8"),
            canonical,
            hashlib.sha256,
        ).hexdigest()

    return headers


async def get_latest_draft_batch(platform: str, platform_id: str) -> Optional[str]:
    """
    Get or create the latest draft batch for the user
    获取或创建用户的最新草稿批次
    
    Args:
        platform: Platform type ('qq' or 'telegram')
        platform_id: User's platform ID
        
    Returns:
        str: Batch ID if successful, None if failed
    """
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()
    
    if not BOT_API_TOKEN:
        logger.error("[get_latest_draft_batch] Missing BOT_API_TOKEN")
        return None
    
    if platform == "web-anon":
        raise UserNotFoundError()

    url = f"{KEYTAO_API_BASE}/api/bot/batches/latest-draft"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.get(
                    url,
                    headers=get_bot_headers(
                        platform,
                        platform_id,
                        method="GET",
                        path="/api/bot/batches/latest-draft",
                    ),
                    params={"platform": platform, "platformId": platform_id},
                ),
                method="GET",
                url="/api/bot/batches/latest-draft",
            )

            if response.status_code == 200:
                data = response.json()
                # The endpoint is a pure read: it answers 200 with batchId=null
                # (exists=false) when the user has no draft yet, instead of
                # creating one.  ``None`` therefore means "no draft", not
                # "failed"; write paths pass it through and let the server
                # create the batch on demand.
                batch_id = data.get("batchId")
                logger.info(
                    "[get_latest_draft_batch] "
                    + (f"Got batch ID: {batch_id}" if batch_id else "No draft batch yet")
                )
                return batch_id
            elif response.status_code == 404:
                raise UserNotFoundError()
            else:
                logger.error(f"[get_latest_draft_batch] API error ({response.status_code}): {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"[get_latest_draft_batch] Error: {e}")
        return None


async def _fetch_draft_snapshot(
    platform: str,
    platform_id: str,
    batch_id: Optional[str] = None,
) -> Optional[Dict]:
    """Fetch current draft items and return as snapshot dict (best-effort, never raises).

    Callers that just wrote to a specific batch pass its id, so the snapshot
    embedded in their result cannot describe a different batch that happens to
    be the newest draft.
    """
    try:
        result = await keytao_list_draft_items(platform, platform_id, batch_id=batch_id)
        if result.get("success"):
            items = result.get("items", [])
            return {
                "count": result.get("count", 0),
                "items": items,
                "summary": compute_draft_summary(items),
            }
    except Exception as e:
        logger.warning(f"[draft_snapshot] Failed to fetch: {e}")
    return None


async def _fetch_encode_candidates(word: str, requested_code: Optional[str] = None) -> Dict:
    keytao_api_base = get_keytao_url()
    encode_url = f"{keytao_api_base}/api/phrases/encode"
    infer_url = f"{keytao_api_base}/api/phrases/infer"
    params = {"word": word}
    if requested_code:
        params["code"] = requested_code

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.get(encode_url, params=params),
                method="GET",
                url="/api/phrases/encode",
            )
            encode_data = normalize_contextual_phrase_encoding(
                word,
                response.json() if response.is_success else {},
            )
            codes = _clean_code_list(encode_data.get("codes"))
            alt_codes = _clean_code_list(encode_data.get("altCodes"))
            if not codes:
                infer_response = await http_client.request_with_retries(
                    lambda: client.get(infer_url, params=params),
                    method="GET",
                    url="/api/phrases/infer",
                )
                infer_data = normalize_contextual_phrase_encoding(
                    word,
                    infer_response.json() if infer_response.is_success else {},
                )
                return _build_encode_candidate_result(
                    word,
                    encode_data,
                    infer_data,
                    requested_code,
                )
            else:
                return _build_encode_candidate_result(
                    word,
                    {**encode_data, "codes": codes, "altCodes": alt_codes},
                    requested_code=requested_code,
                )
    except httpx.TimeoutException:
        return {
            "success": False,
            "message": _draft_tool_failure(
                f"计算「{word}」编码超时",
                command="查看草稿",
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "message": _draft_tool_failure(
                f"无法计算「{word}」的编码",
                error=e,
                log_context="shift_encode",
            ),
        }


async def _validate_draft_item_code(
    item: Dict,
    *,
    reviewed_pinyin: str = "",
    reviewed_candidate_codes: Optional[List[str]] = None,
) -> Dict:
    """Validate every code-writing item; unverifiable types require review."""
    if not _should_validate_item_code(item):
        return {"success": True, "skipped": True}

    word = str(item.get("word") or "").strip()
    code = str(item.get("code") or "").strip().lower()
    phrase_type = _infer_phrase_type(word, code, item.get("type") or "Phrase")
    if phrase_type not in VALID_PHRASE_TYPES:
        return review_flags.apply_review_disposition({
            "success": False,
            "word": word,
            "code": code,
            "reason": f"不支持的词库类型：{phrase_type or '(empty)'}",
        }, "invalid_code")
    shape_error = _validate_code_shape(phrase_type, code)
    if shape_error:
        return review_flags.apply_review_disposition({
            "success": False,
            "word": word,
            "code": code,
            "reason": shape_error,
            "candidateCodes": [],
        }, "invalid_code")
    if phrase_type not in CHAIN_VALIDATED_TYPES or not _contains_cjk_text(word):
        return review_flags.apply_review_disposition({
            "success": True,
            "word": word,
            "code": code,
            "type": phrase_type,
            "needsManualReview": True,
            "manualReviewReason": f"{phrase_type} 类型没有确定性编码校验规则，需管理员人工确认",
        }, "unvalidated_type")
    trusted_reviewed_codes = _clean_code_list(reviewed_candidate_codes)
    reviewed_syllables = [
        value
        for value in re.split(r"\s+", str(reviewed_pinyin or "").strip())
        if value
    ]
    if (
        trusted_reviewed_codes
        and code in trusted_reviewed_codes
        and len(reviewed_syllables) == len(word)
    ):
        return {
            "success": True,
            "word": word,
            "code": code,
            "candidateCodes": trusted_reviewed_codes,
            "reviewedPinyin": " ".join(reviewed_syllables),
            "validationSource": "reviewed-reading",
        }
    encoding = await _fetch_encode_candidates(word, code)
    if not encoding.get("success"):
        return review_flags.apply_review_disposition({
            "success": False,
            "word": word,
            "code": code,
            "reason": encoding.get("message", "编码校验失败"),
            "candidateCodes": encoding.get("candidateCodes", []),
        }, "code_unresolved")

    candidate_codes = encoding.get("candidateCodes", [])
    if code in candidate_codes:
        return {
            "success": True,
            "word": word,
            "code": code,
            "candidateCodes": candidate_codes,
        }

    return review_flags.apply_review_disposition({
        "success": False,
        "word": word,
        "code": code,
        "reason": f"编码 {code} 不是「{word}」的有效候选编码",
        "candidateCodes": candidate_codes,
        "requestedCodeAnalysis": encoding.get("requestedCodeAnalysis"),
    }, "invalid_code")


def _compact_candidate_chain_summary(candidate_codes: List[str]) -> str:
    """Summarize code families without dumping every expansion slot."""
    groups: Dict[str, List[str]] = {}
    for raw_code in candidate_codes:
        code = str(raw_code or "").strip().lower()
        if not code:
            continue
        base = code.rstrip("o") or code
        groups.setdefault(base, []).append(code)
    summaries = []
    for base, codes in list(groups.items())[:4]:
        ordered = sorted(set(codes), key=lambda value: (len(value), value))
        summaries.append(
            ordered[0]
            if len(ordered) == 1
            else f"{ordered[0]}–{ordered[-1]}"
        )
    if not summaries:
        return ""
    suffix = f"（共 {len(groups)} 组）" if len(groups) > len(summaries) else ""
    return "；".join(summaries) + suffix


def _format_code_validation_failure(validation: Dict, index: int = 0) -> Dict:
    candidate_codes = validation.get("candidateCodes") or []
    reason = validation.get("reason", "编码校验失败")
    if candidate_codes:
        compact = _compact_candidate_chain_summary(candidate_codes)
        if compact:
            reason += f"；可选读音链：{compact}"
    failed = {
        "index": index,
        "word": validation.get("word", ""),
        "code": validation.get("code", ""),
        "reason": reason,
        "validationError": True,
    }
    disposition = review_flags.read_review_disposition(validation)
    if disposition is not None:
        failed[review_flags.REVIEW_DISPOSITION_FIELD] = disposition.value
        failed[review_flags.REVIEW_VERDICT_SITE_FIELD] = str(
            validation.get(review_flags.REVIEW_VERDICT_SITE_FIELD) or ""
        )
    if validation.get("requestedCodeAnalysis") is not None:
        failed["requestedCodeAnalysis"] = validation.get("requestedCodeAnalysis")
    return failed


async def _split_items_by_code_validation(items: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """Return (valid_items, failed_items) after deterministic code validation."""
    if not items:
        return [], []

    semaphore = asyncio.Semaphore(8)
    normalized_items = [_normalize_draft_item_for_request(item) for item in items]

    async def validate(index: int, item: Dict) -> tuple[int, Dict, Dict]:
        async with semaphore:
            return index, item, await _validate_draft_item_code(
                item,
                reviewed_pinyin=str(item.get("_reviewed_pinyin") or ""),
                reviewed_candidate_codes=item.get("_reviewed_candidate_codes"),
            )

    checked = await asyncio.gather(
        *(validate(index, item) for index, item in enumerate(normalized_items))
    )

    valid_items: List[Dict] = []
    failed_items: List[Dict] = []
    for index, item, validation in checked:
        item.pop("_reviewed_pinyin", None)
        item.pop("_reviewed_candidate_codes", None)
        if validation.get("success"):
            valid_items.append(_stamp_item_review_flag(item, validation))
        else:
            failed_items.append(_format_code_validation_failure(validation, index))
    return valid_items, failed_items


async def _lookup_words_raw(words: List[str]) -> Dict:
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()
    if not BOT_API_TOKEN:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.post(
                    f"{KEYTAO_API_BASE}/api/bot/phrases/by-word/batch",
                    headers={"X-Bot-Token": BOT_API_TOKEN, "Content-Type": "application/json"},
                    json={"words": words},
                ),
                method="POST",
                url="/api/bot/phrases/by-word/batch",
                idempotent=True,
            )
            data = response.json()
            if not data.get("success"):
                return {"success": False, "message": data.get("message", "按词查询失败")}
            return {"success": True, "results": data.get("results", [])}
    except httpx.TimeoutException:
        return {
            "success": False,
            "message": _draft_tool_failure("按词查询超时", command="查看草稿"),
        }
    except Exception as e:
        return {
            "success": False,
            "message": _draft_tool_failure(
                "按词查询暂时不可用",
                error=e,
                log_context="lookup_words",
            ),
        }


async def _lookup_codes_raw(codes: List[str]) -> Dict:
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()
    if not BOT_API_TOKEN:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.post(
                    f"{KEYTAO_API_BASE}/api/bot/phrases/by-code/batch",
                    headers={"X-Bot-Token": BOT_API_TOKEN, "Content-Type": "application/json"},
                    json={"codes": codes},
                ),
                method="POST",
                url="/api/bot/phrases/by-code/batch",
                idempotent=True,
            )
            data = response.json()
            if not data.get("success"):
                return {"success": False, "message": data.get("message", "按编码查询失败")}
            return {"success": True, "results": data.get("results", [])}
    except httpx.TimeoutException:
        return {
            "success": False,
            "message": _draft_tool_failure("按编码查询超时", command="查看草稿"),
        }
    except Exception as e:
        return {
            "success": False,
            "message": _draft_tool_failure(
                "按编码查询暂时不可用",
                error=e,
                log_context="lookup_codes",
            ),
        }


async def keytao_create_phrase(
    platform: str,
    platform_id: str,
    word: str,
    code: str,
    action: str = "Create",
    old_word: Optional[str] = None,
    type: str = "Phrase",
    remark: Optional[str] = None,
    confirmed: bool = False,
    needs_manual_review: Optional[bool] = None,
    batch_id: Optional[str] = None,
    expected_content_version: Optional[int] = None,
    expected_warning_digest: str = "",
    preview_only: bool = False,
    weight: Optional[int] = None,
    _reviewed_pinyin: str = "",
    _reviewed_candidate_codes: Optional[List[str]] = None,
) -> Dict:
    """
    Create, modify or delete a phrase entry via bot API
    通过 bot API 创建、修改或删除词条
    
    Automatically gets or creates a draft batch for the user.
    自动获取或创建用户的草稿批次。
    
    Args:
        platform: Platform type ('qq' or 'telegram')
        platform_id: User's platform ID
        word: The word/phrase to add/modify/delete
        code: Input method code
        action: Action type ('Create', 'Change', or 'Delete'), default: 'Create'
        old_word: Old word for Change action
        type: Phrase type (default: 'Phrase')
        remark: Optional remark
        confirmed: Whether warnings are confirmed
        
    Returns:
        dict: API response with success status and details
    """
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()
    
    if not BOT_API_TOKEN:
        return {
            "success": False,
            "message": "喵喵配置错误：缺少API token"
        }
    
    # Get or create draft batch
    if not batch_id:
        try:
            batch_id = await get_latest_draft_batch(platform, platform_id)
        except UserNotFoundError:
            return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
    absence_baseline = not batch_id
    # A missing draft batch is no longer an error: the server creates one on
    # demand for the first write.  Only a confirmed ticket must still name the
    # exact batch it was issued against.
    if confirmed and not batch_id:
        return {"success": False, "message": "添加确认缺少目标批次，已停止执行"}
    if confirmed and (
        not isinstance(expected_content_version, int)
        or isinstance(expected_content_version, bool)
        or expected_content_version < 0
        or not re.fullmatch(r"[0-9a-f]{64}", expected_warning_digest)
    ):
        return {"success": False, "message": "添加确认缺少有效的服务端风险快照"}
    if (
        weight is not None
        and (
            not isinstance(weight, int)
            or isinstance(weight, bool)
            or weight < 0
        )
    ):
        return {"success": False, "message": "词条权重必须是非负整数"}

    # Auto-detect type when not explicitly specified, mirrors detectPhraseType in keytao-next
    type = _infer_phrase_type(word, code, type)
    if type not in VALID_PHRASE_TYPES:
        return {"success": False, "message": f"不支持的词库类型：{type}"}
    item: Dict = {
        "action": action,
        "word": word,
        "oldWord": old_word,
        "code": code,
        "type": type,
        "remark": remark,
    }
    if needs_manual_review is not None:
        review_flags.apply_manual_review_flag(item, bool(needs_manual_review))
    if weight is not None:
        item["weight"] = weight
    validation = await _validate_draft_item_code(
        item,
        reviewed_pinyin=_reviewed_pinyin,
        reviewed_candidate_codes=_reviewed_candidate_codes,
    )
    if not validation.get("success"):
        failed = _format_code_validation_failure(validation)
        result = {
            "success": False,
            "message": failed["reason"],
            "failed": [failed],
            "failedCount": 1,
        }
        if batch_id:
            result["batchId"] = batch_id
            result["batchUrl"] = make_batch_url(batch_id)
        return result
    _stamp_item_review_flag(item, validation)

    url = f"{KEYTAO_API_BASE}/api/bot/pull-requests/batch"
    
    request_data = {
        "platform": platform,
        "platformId": platform_id,
        "items": [item],
        "confirmed": confirmed,
        "previewOnly": preview_only,
        "batchId": batch_id,
        **(
            {
                "expectedContentVersion": expected_content_version,
                "expectedWarningDigest": expected_warning_digest,
            }
            if confirmed
            else {}
        ),
    }
    
    logger.info(f"[keytao_create_phrase] Sending request: {json.dumps(request_data, ensure_ascii=False)}")
    request_body = _json_request_body(request_data)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.post(
                    url,
                    headers=get_bot_headers(
                        platform,
                        platform_id,
                        content_type=True,
                        method="POST",
                        path="/api/bot/pull-requests/batch",
                        raw_body=request_body,
                    ),
                    content=request_body,
                ),
                method="POST",
                url="/api/bot/pull-requests/batch",
                idempotent=preview_only,
            )
            
            if response.status_code == 200:
                data = response.json()
                data.setdefault("batchId", batch_id)
                if preview_only and not data.get("requiresConfirmation"):
                    return _inject_known_batch_url({
                        "success": False,
                        "uncertain": True,
                        "message": "服务端未返回可确认的添加快照，已停止后续操作",
                        "batchId": batch_id,
                    }, batch_id)
                logger.info(f"[keytao_create_phrase] API response (200): {json.dumps(data, ensure_ascii=False)}")
                snapshot = await _fetch_draft_snapshot(
                    platform, platform_id, str(data.get("batchId") or batch_id or "") or None
                )
                if snapshot is not None:
                    data["draft_snapshot"] = snapshot
                if preview_only and data.get("requiresConfirmation") and absence_baseline:
                    _mark_provisional_batch(data)
                _inject_batch_url(data)
                return data
            elif response.status_code == 404:
                logger.warning(f"[keytao_create_phrase] API response (404): {response.text}")
                return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
            elif response.status_code == 400:
                # Conflict or warning
                data = response.json()
                data.setdefault("batchId", batch_id)
                logger.info(f"[keytao_create_phrase] API response (400): {json.dumps(data, ensure_ascii=False)}")
                # Attach draft snapshot so AI can report current state even when this item has a warning
                if data.get("requiresConfirmation"):
                    snapshot = await _fetch_draft_snapshot(
                        platform, platform_id, str(data.get("batchId") or batch_id or "") or None
                    )
                    if snapshot is not None:
                        data["draft_snapshot"] = snapshot
                    if preview_only and absence_baseline:
                        _mark_provisional_batch(data)
                _inject_known_batch_url(data, batch_id)
                return data
            else:
                logger.error(f"[keytao_create_phrase] API response ({response.status_code}): {response.text}")
                result = {
                    "success": False,
                    "message": f"创建失败: HTTP {response.status_code}"
                }
                if confirmed and response.status_code >= 500:
                    result.update({
                        "uncertain": True,
                        "message": render_remediation_reply(
                            "添加结果无法确认；请求可能已经生效",
                            command="查看草稿",
                        ),
                    })
                return _inject_known_batch_url(result, batch_id)
                
    except httpx.TimeoutException:
        if preview_only:
            return _inject_known_batch_url({
                "success": False,
                "message": _draft_tool_failure(
                    "添加预检超时",
                    command="查看草稿",
                ),
            }, batch_id)
        result = {
            "success": False,
            "uncertain": True,
            "message": render_remediation_reply(
                "添加请求超时，草稿可能已经写入",
                command="查看草稿",
            ),
        }
        return _inject_known_batch_url(result, batch_id)
    except Exception as e:
        result = {
            "success": False,
            "message": (
                render_remediation_reply(
                    "添加结果无法确认",
                    command="查看草稿",
                )
                if confirmed
                else _draft_tool_failure(
                    "添加服务暂时不可用",
                    command="查看草稿",
                    error=e,
                    log_context="create_phrase",
                )
            ),
        }
        if confirmed:
            result["uncertain"] = True
        return _inject_known_batch_url(result, batch_id)


def _review_config() -> ReviewHttpConfig:
    return ReviewHttpConfig(api_base=get_keytao_url(), bot_token=get_bot_token() or "")


def _draft_audit_timeout() -> float:
    try:
        from nonebot import get_driver
        config = get_driver().config
        value = getattr(config, "keytao_background_review_audit_timeout", None)
        if value is None:
            value = getattr(config, "keytao_batch_review_audit_timeout", None)
    except Exception:
        value = None
    try:
        return max(5.0, float(value or 90))
    except (TypeError, ValueError):
        return 90.0


async def _fallback_draft_audit_with_encode(items: List[Dict], reason: str) -> Dict:
    approved_items: List[str] = []
    issues: List[str] = []
    words = list(dict.fromkeys(
        str(item.get("word") or "").strip()
        for item in items
        if isinstance(item, dict)
        and str(item.get("action") or "Create") != "Delete"
        and str(item.get("word") or "").strip()
    ))
    review_results = await asyncio.gather(*(
        prepare_reviewed_word(_review_config(), word)
        for word in words
    ), return_exceptions=True)
    reviewed_words = dict(zip(words, review_results))

    def candidate_codes(review: Dict) -> set[str]:
        result: set[str] = set()
        for pronunciation in review.get("pronunciations", []):
            if not isinstance(pronunciation, dict):
                continue
            result.update(_clean_code_list(pronunciation.get("codes")))
            for status in pronunciation.get("candidateStatuses", []):
                if isinstance(status, dict):
                    result.update(_clean_code_list([status.get("code")]))
        return result

    for item in items:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "Create")
        word = str(item.get("word") or "")
        code = str(item.get("code") or "")
        if action == "Delete":
            issues.append(f"「{word}」是纯删除操作，需要管理员确认")
            continue
        if not word or not code:
            issues.append(f"草稿条目缺少词或编码，需要管理员确认：{item}")
            continue
        preaudit_issue = manual_preaudit_issue_for_item(item)
        if preaudit_issue:
            issues.append(preaudit_issue)
        try:
            review = reviewed_words.get(word)
            if isinstance(review, Exception) or not isinstance(review, dict) or not review.get("success"):
                encoding = await fetch_keytao_encode(_review_config(), word)
                normalized_encoding = _build_encode_candidate_result(word, encoding)
                codes = set(normalized_encoding.get("candidateCodes") or [])
                basis = "keytao_encode 默认候选链"
            else:
                codes = candidate_codes(review)
                corrected = any(
                    isinstance(pronunciation, dict)
                    and isinstance(pronunciation.get("contextPronunciation"), dict)
                    and pronunciation["contextPronunciation"].get("correctedDefault")
                    for pronunciation in review.get("pronunciations", [])
                )
                basis = "读音优先级纠正后的候选链" if corrected else "审词候选链"
            if code in codes:
                approved_items.append(f"{action}：{word}@{code}，编码在{basis}中")
            else:
                available = "、".join(sorted(codes)[:8])
                issues.append(f"「{word}」编码 {code} 不在候选链中；可选：{available or '无'}")
        except Exception as error:
            issues.append(f"「{word}」编码兜底检查失败：{error}")

    if not issues:
        issues.append(f"{reason}；可比较的常用度信号不足，需要本喵继续复审")
    return {
        "success": True,
        "verdict": "needs_admin",
        "autoApprove": False,
        "summary": f"{reason}，已并行按读音优先级重建候选链，需要本喵继续复审",
        "issues": issues,
        "approvedItems": approved_items,
        "timeout": True,
        "encodeOnly": False,
        "contextualPronunciationFallback": True,
        "reviewedWords": {
            word: review
            for word, review in reviewed_words.items()
            if isinstance(review, dict)
        },
    }


async def _audit_current_draft_for_auto_approval(
    platform: str,
    platform_id: str,
    batch_id: str,
) -> Dict:
    try:
        list_result = await keytao_list_draft_items(
            platform,
            platform_id,
            batch_id=batch_id,
        )
        if not list_result.get("success"):
            return {
                "success": False,
                "verdict": "needs_admin",
                "autoApprove": False,
                "summary": list_result.get("message", "无法读取草稿，需要管理员审核"),
                "issues": [list_result.get("message", "无法读取草稿")],
            }
        listed_batch_id = str(list_result.get("batchId") or "")
        content_version = list_result.get("contentVersion")
        if (
            listed_batch_id != batch_id
            or not isinstance(content_version, int)
            or isinstance(content_version, bool)
            or content_version < 0
        ):
            return {
                "success": False,
                "verdict": "needs_admin",
                "autoApprove": False,
                "summary": "草稿快照缺少可验证版本，已停止提交",
                "issues": ["草稿批次或内容版本不匹配"],
            }

        def bind_snapshot(review: Dict) -> Dict:
            snapshot_items = [
                {
                    key: item.get(key)
                    for key in (
                        "id", "action", "word", "oldWord", "code", "type",
                        "weight", "remark", "needsManualReview",
                    )
                }
                for item in list_result.get("items", [])
                if isinstance(item, dict)
            ]
            snapshot_digest = hashlib.sha256(json.dumps(
                {
                    "batchId": batch_id,
                    "contentVersion": content_version,
                    "items": snapshot_items,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            return {
                **review,
                "batchId": batch_id,
                "contentVersion": content_version,
                "snapshotItems": snapshot_items,
                "snapshotDigest": snapshot_digest,
            }

        items = list_result.get("items", [])
        locked_items = [
            item for item in items
            if isinstance(item, dict) and item.get("needsManualReview") is True
        ]
        if locked_items:
            labels = [
                str(item.get("word") or "未命名草稿条目")
                for item in locked_items[:10]
            ]
            return bind_snapshot({
                "success": True,
                "verdict": "needs_admin",
                "autoApprove": False,
                "summary": "加词预审已锁定整批管理员审核",
                "issues": [
                    "结构化预审锁：" + "、".join(labels)
                ],
                "approvedItems": [],
                "manualReviewLocked": True,
            })
        audit_timeout = _draft_audit_timeout()
        try:
            deterministic_audit = await asyncio.wait_for(
                audit_draft_items(_review_config(), items),
                timeout=audit_timeout,
            )
        except asyncio.TimeoutError:
            reason = f"确定性来源审查超过 {audit_timeout:.0f} 秒"
            logger.warning(f"[auto_review] deterministic audit timed out: {reason}")
            deterministic_audit = await _fallback_draft_audit_with_encode(items, reason)
        if deterministic_audit.get("autoApprove") or not can_llm_override_audit_issues(deterministic_audit):
            return bind_snapshot(deterministic_audit)
        llm_audit = await _try_llm_auto_review_for_draft(list_result, deterministic_audit)
        return bind_snapshot(llm_audit or deterministic_audit)
    except Exception as error:
        logger.warning(f"[auto_review] audit failed: {error}")
        return {
            "success": False,
            "verdict": "needs_admin",
            "autoApprove": False,
            "summary": "自动审核异常，需要管理员审核",
            "issues": [str(error)],
        }


async def _try_llm_auto_review_for_draft(list_result: Dict, deterministic_audit: Dict) -> Optional[Dict]:
    try:
        from keytao_bot.utils.keytao_batch_review import review_keytao_batch_with_llm

        items = list_result.get("items", [])
        batch = {
            "id": list_result.get("batchId") or list_result.get("batch_id") or "current-draft",
            "status": "Draft",
            "description": "键道助手草稿批次",
            "pullRequests": items,
        }
        review_result = await review_keytao_batch_with_llm(
            batch,
            precomputed_audit=deterministic_audit,
        )
        if not review_result.get("success"):
            logger.warning(f"[auto_review] LLM fallback failed: {review_result.get('message')}")
            return None

        ai_review = review_result.get("aiReview") or {}
        review_items = ai_review.get("items") if isinstance(ai_review.get("items"), list) else []
        non_pass_items = [
            item for item in review_items
            if isinstance(item, dict) and item.get("status") != "pass"
        ]
        if ai_review.get("verdict") == "pass" and not non_pass_items and review_items:
            approved_items = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                action = item.get("action") or "Create"
                word = item.get("word") or ""
                code = item.get("code") or ""
                if word and code:
                    approved_items.append(f"{action}：{word}@{code}，本喵审核通过")
            return {
                **deterministic_audit,
                "success": True,
                "verdict": "pass",
                "autoApprove": True,
                "summary": ai_review.get("headline") or "语言常识、读音和编码检查一致",
                "issues": [],
                "approvedItems": approved_items or deterministic_audit.get("approvedItems", []),
                "llmReview": ai_review,
                "llmFallback": True,
                "encodeOnly": False,
            }

        issues = []
        for item in non_pass_items[:10]:
            reasons = item.get("reasons") if isinstance(item.get("reasons"), list) else []
            title = item.get("title") or "该草稿条目需要复核"
            reason = reasons[0] if reasons else title
            issues.append(str(reason))

        return {
            **deterministic_audit,
            "summary": ai_review.get("headline") or deterministic_audit.get("summary", "存在不确定项，需要管理员审核"),
            "issues": issues or deterministic_audit.get("issues", []),
            "llmReview": ai_review,
            "llmFallback": True,
        }
    except Exception as error:
        logger.warning(f"[auto_review] LLM fallback error: {error}")
        return None


_audit_allows_batch_auto_approve = review_flags.audit_allows_batch_auto_approve


def _auto_review_confirmation_digest(auto_review: Dict) -> str:
    """Bind confirmation to the safety-relevant outcome of the bot-side audit."""
    payload = {
        "batchId": auto_review.get("batchId"),
        "contentVersion": auto_review.get("contentVersion"),
        "snapshotDigest": auto_review.get("snapshotDigest"),
        "success": auto_review.get("success"),
        "verdict": auto_review.get("verdict"),
        "autoApprove": auto_review.get("autoApprove"),
        "summary": auto_review.get("summary"),
        "issues": auto_review.get("issues") or [],
        "approvedItems": auto_review.get("approvedItems") or [],
        "manualReviewLocked": bool(auto_review.get("manualReviewLocked")),
        "llmFallback": bool(auto_review.get("llmFallback")),
        "encodeOnly": bool(auto_review.get("encodeOnly")),
        "timeout": bool(auto_review.get("timeout")),
        "evidence": auto_review.get("evidence") or [],
    }
    return hashlib.sha256(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")).hexdigest()


def _is_exact_submit_preview(
    data: object,
    batch_id: str,
    content_version: int,
) -> bool:
    """Accept only a complete, read-only server ticket for submission."""
    return bool(
        isinstance(data, dict)
        and data.get("success") is False
        and data.get("requiresConfirmation") is True
        and data.get("batchId") == batch_id
        and data.get("contentVersion") == content_version
        and not isinstance(data.get("contentVersion"), bool)
        and isinstance(data.get("warnings"), list)
        and isinstance(data.get("warningDigest"), str)
        and re.fullmatch(r"[0-9a-f]{64}", data["warningDigest"])
        and isinstance(data.get("snapshotDigest"), str)
        and re.fullmatch(r"[0-9a-f]{64}", data["snapshotDigest"])
    )


async def _auto_approve_submitted_batch(
    platform: str,
    platform_id: str,
    batch_id: str,
    auto_review: Dict,
    expected_content_version: int,
) -> Dict:
    if not _audit_allows_batch_auto_approve(auto_review):
        return {
            "success": False,
            "message": "整批审核未达到逐项通过条件，已保留给管理员审核",
        }
    KEYTAO_API_BASE = get_keytao_url()
    review_note = build_review_note(auto_review)
    safe_batch_id = _safe_path_segment(batch_id)
    if not safe_batch_id:
        logger.error(f"[auto_review] refusing unsafe batch id: {batch_id!r}")
        return {"success": False, "message": "批次编号非法，已保留给管理员审核"}
    url = f"{KEYTAO_API_BASE}/api/bot/batches/{safe_batch_id}/auto-approve"
    request_data = {
        "platform": platform,
        "platformId": platform_id,
        "reviewNote": review_note,
        "expectedContentVersion": expected_content_version,
    }
    request_body = _json_request_body(request_data)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.post(
                    url,
                    headers=get_bot_headers(
                        platform,
                        platform_id,
                        content_type=True,
                        method="POST",
                        path=f"/api/bot/batches/{safe_batch_id}/auto-approve",
                        raw_body=request_body,
                    ),
                    content=request_body,
                ),
                method="POST",
                url=f"/api/bot/batches/{safe_batch_id}/auto-approve",
            )
        try:
            data = response.json()
        except Exception:
            return {"success": False, "message": f"自动批准接口返回异常（HTTP {response.status_code}）"}
        if response.is_success:
            return data
        return {
            "success": False,
            "message": data.get("message") or data.get("error") or f"自动批准失败: HTTP {response.status_code}",
            "details": data,
        }
    except httpx.TimeoutException:
        return {
            "success": False,
            "message": render_remediation_reply(
                "批次已提交，自动批准超时且可能已经生效；"
                "转交管理员属于站外处理",
                command="查看草稿",
            ),
        }
    except Exception as error:
        logger.warning(f"[auto_review] approve failed: {error}")
        return {
            "success": False,
            "message": "批次已提交，自动批准失败，转交管理员审核",
        }


async def keytao_submit_batch(
    platform: str,
    platform_id: str,
    confirmed: bool = False,
    batch_id: Optional[str] = None,
    expected_content_version: Optional[int] = None,
    expected_server_snapshot_digest: str = "",
    expected_warning_digest: str = "",
    expected_audit_digest: str = "",
    preview_only: bool = False,
) -> Dict:
    """
    Submit current draft batch for review
    提交当前草稿批次进行审核
    
    Automatically finds and submits the user's latest draft batch.
    自动查找并提交用户的最新草稿批次。
    
    Args:
        platform: Platform type ('qq' or 'telegram')
        platform_id: User's platform ID
        confirmed: Whether the exact server preview ticket is being confirmed
        
    Returns:
        dict: API response with success status
    """
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()
    
    if not BOT_API_TOKEN:
        return {
            "success": False,
            "message": "喵喵配置错误：缺少API token"
        }
    
    if confirmed and (
        not batch_id
        or not isinstance(expected_content_version, int)
        or isinstance(expected_content_version, bool)
        or expected_content_version < 0
        or not re.fullmatch(r"[0-9a-f]{64}", expected_server_snapshot_digest)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_warning_digest)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_audit_digest)
    ):
        return _inject_known_batch_url({
            "success": False,
            "message": render_remediation_reply(
                "提交确认缺少有效的批次版本",
                command="提交",
            ),
        }, batch_id)

    # The first request may resolve the latest draft. A confirmation must use
    # the exact batch/version returned by that first server warning.
    if not batch_id:
        try:
            batch_id = await get_latest_draft_batch(platform, platform_id)
        except UserNotFoundError:
            return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
    if not batch_id:
        return {"success": False, "message": "没有找到待提交的草稿批次"}

    confirmation_claimed = False
    confirmation_generation = ""

    def consume_confirmation() -> None:
        if not confirmation_claimed or not confirmation_generation:
            return
        _SUBMIT_AUDIT_TICKETS.consume(
            platform,
            platform_id,
            batch_id,
            expected_content_version,
            expected_audit_digest,
            confirmation_generation,
        )

    def mark_confirmation_uncertain() -> None:
        if not confirmation_claimed or not confirmation_generation:
            return
        _SUBMIT_AUDIT_TICKETS.mark_uncertain(
            platform,
            platform_id,
            batch_id,
            expected_content_version,
            expected_audit_digest,
            confirmation_generation,
        )

    if confirmed:
        claim_status, auto_review, confirmation_generation = _SUBMIT_AUDIT_TICKETS.claim(
            platform,
            platform_id,
            batch_id,
            expected_content_version,
            expected_audit_digest,
        )
        if claim_status == "claimed":
            return _inject_known_batch_url({
                "success": False,
                "uncertain": True,
                "error": "submit_confirmation_already_claimed",
                "message": render_remediation_reply(
                    "这次提交正在执行或结果尚不确定",
                    command="查看草稿",
                ),
                "batchId": batch_id,
            }, batch_id)
        if claim_status != "ok" or auto_review is None:
            return _inject_known_batch_url({
                "success": False,
                "staleConfirmation": True,
                "error": "submit_confirmation_missing",
                "message": (
                    render_remediation_reply(
                        "提交检查已过期或不匹配；若这是加词后提交，"
                        "当前结果不含可绑定的词条和编码",
                        command="提交",
                    )
                ),
                "batchId": batch_id,
            }, batch_id)
        confirmation_claimed = True
    else:
        auto_review = await _audit_current_draft_for_auto_approval(
            platform,
            platform_id,
            batch_id,
        )
    audited_content_version = auto_review.get("contentVersion")
    audited_digest = _auto_review_confirmation_digest(auto_review)
    if (
        not isinstance(audited_content_version, int)
        or isinstance(audited_content_version, bool)
        or audited_content_version < 0
    ):
        consume_confirmation()
        return _inject_known_batch_url({
            "success": False,
            "error": "invalid_audit_content_version",
            "message": auto_review.get("summary") or "草稿快照缺少可验证版本，已停止提交",
            "autoReview": auto_review,
        }, batch_id)
    if confirmed and expected_content_version != audited_content_version:
        consume_confirmation()
        return _inject_known_batch_url({
            "success": False,
            "staleConfirmation": True,
            "error": "submit_content_version_changed",
            "message": render_remediation_reply(
                "草稿内容已变化，本次确认已失效",
                command="提交",
            ),
            "batchId": batch_id,
            "contentVersion": audited_content_version,
            "autoReview": auto_review,
        }, batch_id)
    if confirmed and expected_audit_digest != audited_digest:
        consume_confirmation()
        return _inject_known_batch_url({
            "success": False,
            "staleConfirmation": True,
            "error": "submit_audit_digest_changed",
            "message": render_remediation_reply(
                "提交检查与当前快照不匹配，本次确认已失效",
                command="提交",
            ),
            "batchId": batch_id,
            "contentVersion": audited_content_version,
            "auditDigest": audited_digest,
            "autoReview": auto_review,
        }, batch_id)
    submission_content_version = audited_content_version

    def prepare_submit_preview(data: object) -> Dict:
        if not _is_exact_submit_preview(
            data,
            batch_id,
            audited_content_version,
        ):
            return _inject_known_batch_url({
                "success": False,
                "uncertain": True,
                "error": "invalid_submit_preview",
                "message": render_remediation_reply(
                    "服务端未返回完整的只读提交快照",
                    command="查看草稿",
                ),
                "batchId": batch_id,
                "contentVersion": audited_content_version,
            }, batch_id)
        preview_data = dict(data)
        preview_data["autoReview"] = auto_review
        preview_data["auditDigest"] = audited_digest
        preview_data.setdefault("snapshotItems", auto_review.get("snapshotItems", []))
        preview_data["auditSnapshotDigest"] = auto_review.get("snapshotDigest", "")
        _inject_batch_url(preview_data)
        stored = _SUBMIT_AUDIT_TICKETS.put(
            platform,
            platform_id,
            batch_id,
            audited_content_version,
            audited_digest,
            auto_review,
        )
        if not stored:
            return _inject_known_batch_url({
                "success": False,
                "error": "audit_snapshot_not_stored",
                "message": render_remediation_reply(
                    "提交检查无法安全保存（结果过大或容量不足）",
                    command="查看草稿",
                ),
                "batchId": batch_id,
                "contentVersion": audited_content_version,
            }, batch_id)
        return preview_data
    
    safe_batch_id = _safe_path_segment(batch_id)
    if not safe_batch_id:
        logger.error(f"[keytao_submit_batch] refusing unsafe batch id: {batch_id!r}")
        return {"success": False, "message": "批次编号非法，无法提交"}
    url = f"{KEYTAO_API_BASE}/api/bot/batches/{safe_batch_id}/submit"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            request_data = {
                "platform": platform,
                "platformId": platform_id,
                "confirmed": confirmed,
                "previewOnly": preview_only,
            }
            request_data["expectedContentVersion"] = submission_content_version
            if confirmed:
                request_data["expectedSnapshotDigest"] = expected_server_snapshot_digest
                request_data["expectedWarningDigest"] = expected_warning_digest
            request_body = _json_request_body(request_data)
            response = await http_client.request_with_retries(
                lambda: client.post(
                    url,
                    headers=get_bot_headers(
                        platform,
                        platform_id,
                        content_type=True,
                        method="POST",
                        path=f"/api/bot/batches/{safe_batch_id}/submit",
                        raw_body=request_body,
                    ),
                    content=request_body,
                ),
                method="POST",
                url=f"/api/bot/batches/{safe_batch_id}/submit",
                idempotent=preview_only,
            )
            
            if response.status_code == 200:
                data = response.json()
                if not confirmed:
                    return prepare_submit_preview(data)
                data["batchId"] = batch_id  # inject so _inject_batch_url can build batchUrl
                submitted_batch = data.get("batch")
                reported_content_version = data.get("contentVersion")
                if (
                    not isinstance(reported_content_version, int)
                    or isinstance(reported_content_version, bool)
                    or reported_content_version < 0
                ) and isinstance(submitted_batch, dict):
                    reported_content_version = submitted_batch.get("contentVersion")
                if (
                    not isinstance(reported_content_version, int)
                    or isinstance(reported_content_version, bool)
                    or reported_content_version < 0
                ):
                    reported_content_version = submission_content_version
                data["contentVersion"] = reported_content_version
                _inject_batch_url(data)
                data["autoReview"] = auto_review
                data["auditDigest"] = audited_digest
                data.setdefault("snapshotItems", auto_review.get("snapshotItems", []))
                data["auditSnapshotDigest"] = auto_review.get("snapshotDigest", "")
                consume_confirmation()
                if _audit_allows_batch_auto_approve(auto_review):
                    approve_result = await _auto_approve_submitted_batch(
                        platform,
                        platform_id,
                        batch_id,
                        auto_review,
                        reported_content_version,
                    )
                    data["autoApproveResult"] = approve_result
                    data["autoApproved"] = bool(approve_result.get("success"))
                return data
            elif response.status_code == 404:
                consume_confirmation()
                return _inject_known_batch_url({
                    "success": False,
                    "error": "batch_not_found",
                    "message": "批次不存在或已被删除"
                }, batch_id)
            elif response.status_code == 403:
                consume_confirmation()
                return _inject_known_batch_url({
                    "success": False,
                    "error": "batch_forbidden",
                    "message": "无权限操作此批次"
                }, batch_id)
            elif response.status_code == 400:
                data = response.json()
                if not confirmed and data.get("requiresConfirmation") is True:
                    return prepare_submit_preview(data)
                consume_confirmation()
                data.setdefault("error", "submit_rejected")
                data.setdefault("batchId", batch_id)
                data.setdefault("contentVersion", submission_content_version)
                data["autoReview"] = auto_review
                data["auditDigest"] = audited_digest
                data.setdefault("snapshotItems", auto_review.get("snapshotItems", []))
                data["auditSnapshotDigest"] = auto_review.get("snapshotDigest", "")
                _inject_known_batch_url(data, batch_id)
                return data
            elif response.status_code == 409:
                data = response.json()
                consume_confirmation()
                return _inject_known_batch_url({
                    **data,
                    "success": False,
                    "error": data.get("error") or "submit_snapshot_changed",
                    "staleConfirmation": True,
                    "batchId": batch_id,
                    "message": data.get("message") or render_remediation_reply(
                        "草稿内容已变化",
                        command="提交",
                    ),
                }, batch_id)
            else:
                if confirmed:
                    mark_confirmation_uncertain()
                    return _inject_known_batch_url({
                        "success": False,
                        "uncertain": True,
                        "error": "submit_result_uncertain",
                        "message": render_remediation_reply(
                            "提交结果暂时无法确定",
                            command="查看草稿",
                        ),
                        "batchId": batch_id,
                    }, batch_id)
                return _inject_known_batch_url({
                    "success": False,
                    "error": "submit_http_error",
                    "message": f"提交失败: HTTP {response.status_code}",
                }, batch_id)
                
    except asyncio.CancelledError:
        mark_confirmation_uncertain()
        raise
    except httpx.TimeoutException:
        if preview_only:
            return _inject_known_batch_url({
                "success": False,
                "error": "submit_preview_timeout",
                "message": _draft_tool_failure(
                    "提交预检超时",
                    command="查看草稿",
                ),
            }, batch_id)
        if confirmed:
            mark_confirmation_uncertain()
            return _inject_known_batch_url({
                "success": False,
                "uncertain": True,
                "error": "submit_result_uncertain",
                "message": render_remediation_reply(
                    "提交请求超时，结果可能已经生效",
                    command="查看草稿",
                ),
                "batchId": batch_id,
            }, batch_id)
        return _inject_known_batch_url({
            "success": False,
            "uncertain": True,
            "error": "submit_timeout",
            "message": render_remediation_reply(
                "提交请求超时，草稿可能已经写入",
                command="查看草稿",
            ),
        }, batch_id)
    except Exception as e:
        if confirmed:
            mark_confirmation_uncertain()
            return _inject_known_batch_url({
                "success": False,
                "uncertain": True,
                "error": "submit_result_uncertain",
                "message": render_remediation_reply(
                    "提交结果无法确定",
                    command="查看草稿",
                ),
                "batchId": batch_id,
            }, batch_id)
        return _inject_known_batch_url({
            "success": False,
            "error": "submit_failed",
            "message": _draft_tool_failure(
                "提交服务暂时不可用",
                command="查看草稿",
                error=e,
                log_context="submit_batch",
            ),
        }, batch_id)


async def keytao_get_batch_preview(
    platform: str,
    platform_id: str,
    batch_id: Optional[str] = None,
) -> Dict:
    """
    Fetch the diff preview of the user's current draft batch.
    Returns summary stats and a formatted unified-diff text block.

    ``batch_id`` anchors the preview to a known batch, so a caller that just
    operated on one cannot be shown a different batch that happens to be the
    newest draft.
    """
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()

    if not BOT_API_TOKEN:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}

    batch_id = str(batch_id or "").strip() or None
    if not batch_id:
        try:
            batch_id = await get_latest_draft_batch(platform, platform_id)
        except UserNotFoundError:
            return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
    if not batch_id:
        return {"success": False, "message": "没有找到草稿批次"}

    safe_batch_id = _safe_path_segment(batch_id)
    if not safe_batch_id:
        logger.error(f"[keytao_get_batch_preview] refusing unsafe batch id: {batch_id!r}")
        return {"success": False, "message": "批次编号非法，无法获取预览"}
    url = f"{KEYTAO_API_BASE}/api/batches/{safe_batch_id}/preview"
    logger.info(f"[keytao_get_batch_preview] batchId={batch_id}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.get(url),
                method="GET",
                url=f"/api/batches/{safe_batch_id}/preview",
            )

        try:
            data = response.json()
        except Exception:
            return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}

        if response.status_code != 200:
            return {"success": False, "message": f"获取预览失败: HTTP {response.status_code}"}

        preview = data.get("preview", {})
        summary = preview.get("summary", {})
        diff_text = _format_preview_text(preview)

        return {
            "success": True,
            "batchId": batch_id,
            "batchUrl": make_batch_url(batch_id),
            "summary": summary,
            "diff_text": diff_text,
        }

    except httpx.TimeoutException:
        return {
            "success": False,
            "message": _draft_tool_failure(
                "草稿预览请求超时",
                command="查看草稿",
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "message": _draft_tool_failure(
                "草稿预览暂时不可用",
                command="查看草稿",
                error=e,
                log_context="get_batch_preview",
            ),
        }


async def keytao_recall_batch(
    platform: str,
    platform_id: str,
    batch_id: Optional[str] = None,
    expected_content_version: Optional[int] = None,
) -> Dict:
    """
    Recall (un-submit) the latest submitted batch, reverting it back to Draft.
    撤回最近一次提审，将批次状态恢复为草稿。
    """
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()

    if not BOT_API_TOKEN:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}

    try:
        existing_claim = _draft_mutation_claims().get(platform, platform_id)
    except Exception as error:
        logger.error(
            "Failed to read recall claim: %s: %s",
            type(error).__name__,
            error,
        )
        return _locked_mutation_result(
            str(batch_id or ""),
            "无法读取上一次草稿操作的安全状态，本次未执行撤回。",
        )

    existing_recall_payload: Optional[Dict] = None
    existing_recall_fingerprint = ""
    reuse_existing_claim = False
    if existing_claim is not None:
        operation_kind = str(existing_claim.get("operationKind") or "")
        claim_payload = existing_claim.get("payload")
        claimed_batch_id = str((claim_payload or {}).get("batchId") or "")
        if operation_kind != "recall" or not isinstance(claim_payload, dict):
            return _locked_mutation_result(
                claimed_batch_id or str(batch_id or ""),
                "上一次草稿操作结果仍不确定；已锁定原操作，本次不会撤回其他批次。",
            )
        replay = _replay_resolved_mutation_claim(existing_claim)
        if replay is not None:
            return replay
        existing_recall_payload = claim_payload
        existing_recall_fingerprint = str(existing_claim.get("fingerprint") or "")
        claimed_version = claim_payload.get("contentVersion")
        if (
            not claimed_batch_id
            or not existing_recall_fingerprint
            or not isinstance(claimed_version, int)
            or isinstance(claimed_version, bool)
        ):
            return _locked_mutation_result(
                claimed_batch_id or str(batch_id or ""),
                "上一次撤回操作的安全记录不完整，本次不会撤回其他批次。",
            )
        recalled_snapshot = await keytao_list_draft_items(
            platform,
            platform_id,
            batch_id=claimed_batch_id,
        )
        if (
            recalled_snapshot.get("success")
            and str(recalled_snapshot.get("batchId") or "") == claimed_batch_id
            and str(recalled_snapshot.get("status") or "") == "Draft"
            and isinstance(recalled_snapshot.get("items"), list)
        ):
            applied = {
                "success": True,
                "alreadyApplied": True,
                "batchId": claimed_batch_id,
                "contentVersion": recalled_snapshot.get("contentVersion"),
                "items": recalled_snapshot.get("items") or [],
                "message": "上一次撤回已通过只读草稿快照确认生效",
            }
            if recalled_snapshot.get("batchUrl"):
                applied["batchUrl"] = recalled_snapshot.get("batchUrl")
            applied = _inject_known_batch_url(applied, claimed_batch_id)
            if not _resolve_draft_mutation_claim(
                platform,
                platform_id,
                "recall",
                existing_recall_fingerprint,
                applied,
            ):
                return _locked_mutation_result(
                    claimed_batch_id,
                    "原撤回结果已核验，但最终回执无法安全保存；本次不会继续操作。",
                )
            return applied
        if batch_id:
            if (
                str(batch_id) != claimed_batch_id
                or expected_content_version != claimed_version
            ):
                return _locked_mutation_result(
                    claimed_batch_id,
                    "上一次撤回结果仍无法确认；已锁定原批次，不会撤回其他批次。",
                )
            reuse_existing_claim = True

    url = f"{KEYTAO_API_BASE}/api/bot/batches/recall"
    logger.info(f"[keytao_recall_batch] platform={platform} platformId={platform_id}")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if not batch_id:
                response = await http_client.request_with_retries(
                    lambda: client.get(
                        url,
                        headers=get_bot_headers(
                            platform,
                            platform_id,
                            method="GET",
                            path="/api/bot/batches/recall",
                        ),
                        params={"platform": platform, "platformId": platform_id},
                    ),
                    method="GET",
                    url="/api/bot/batches/recall",
                )
                try:
                    data = response.json()
                except Exception:
                    if existing_recall_payload is not None:
                        return _locked_mutation_result(
                            str(existing_recall_payload.get("batchId") or ""),
                            "撤回核验接口返回异常；已锁定原批次，不会选择新的提交批次。",
                        )
                    return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}
                if not response.is_success:
                    if existing_recall_payload is not None:
                        return _locked_mutation_result(
                            str(existing_recall_payload.get("batchId") or ""),
                            "上一次撤回结果仍无法确认；已锁定原批次，不会选择新的提交批次。",
                        )
                    return {
                        **data,
                        "success": False,
                        "message": data.get("message") or f"获取待撤回批次失败（HTTP {response.status_code}）",
                    }
                preview_batch_id = str(data.get("batchId") or "")
                preview_version = data.get("contentVersion")
                if (
                    not preview_batch_id
                    or not isinstance(preview_version, int)
                    or isinstance(preview_version, bool)
                    or preview_version < 0
                ):
                    if existing_recall_payload is not None:
                        return _locked_mutation_result(
                            str(existing_recall_payload.get("batchId") or ""),
                            "撤回核验结果不完整；已锁定原批次，不会选择新的提交批次。",
                        )
                    return {"success": False, "message": "待撤回批次缺少可验证版本"}
                if existing_recall_payload is not None:
                    if (
                        preview_batch_id
                        != str(existing_recall_payload.get("batchId") or "")
                        or preview_version
                        != existing_recall_payload.get("contentVersion")
                    ):
                        return _locked_mutation_result(
                            str(existing_recall_payload.get("batchId") or ""),
                            "上一次撤回结果仍不确定；已锁定原批次，不会撤回新的提交批次。",
                        )
                preview_data = {
                    **data,
                    "success": False,
                    "requiresConfirmation": True,
                    "confirmationKind": "recallBatch",
                    "message": "即将撤回这个已提交批次并恢复为草稿",
                }
                _inject_batch_url(preview_data)
                return preview_data
            if (
                not isinstance(expected_content_version, int)
                or isinstance(expected_content_version, bool)
                or expected_content_version < 0
            ):
                return {"success": False, "message": "撤回确认缺少有效的批次版本"}
            request_data = {
                "platform": platform,
                "platformId": platform_id,
                "batchId": batch_id,
                "expectedContentVersion": expected_content_version,
            }
            claim_payload = {
                "batchId": str(batch_id),
                "contentVersion": int(expected_content_version),
            }
            claim_fingerprint = (
                existing_recall_fingerprint
                if reuse_existing_claim
                else None
            )
            if claim_fingerprint is None:
                try:
                    claim_fingerprint = _draft_mutation_claims().begin(
                        platform,
                        platform_id,
                        "recall",
                        claim_payload,
                    )
                except Exception as error:
                    logger.error(
                        "Failed to acquire recall claim: %s: %s",
                        type(error).__name__,
                        error,
                    )
                    claim_fingerprint = None
            if not claim_fingerprint:
                return _locked_mutation_result(
                    str(batch_id),
                    "另一个草稿操作正在核验，本次未执行撤回。",
                )
            request_body = _json_request_body(request_data)
            response = await http_client.request_with_retries(
                lambda: client.post(
                    url,
                    headers=get_bot_headers(
                        platform,
                        platform_id,
                        content_type=True,
                        method="POST",
                        path="/api/bot/batches/recall",
                        raw_body=request_body,
                    ),
                    content=request_body,
                ),
                method="POST",
                url="/api/bot/batches/recall",
            )
            try:
                data = response.json()
            except Exception:
                if batch_id:
                    uncertain = {
                        "success": False,
                        "uncertain": True,
                        "batchId": batch_id,
                        "message": render_remediation_reply(
                            "撤回结果无法确认；请求可能已经生效",
                            command="查看草稿",
                        ),
                    }
                    _inject_batch_url(uncertain)
                    return uncertain
                return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}

            logger.info(f"[keytao_recall_batch] status={response.status_code} success={data.get('success')}")
            data.setdefault("batchId", batch_id)
            if response.status_code == 409:
                stale = {
                    **data,
                    "success": False,
                    "staleConfirmation": True,
                    "message": data.get("message") or "待撤回批次已变化，旧票据已作废",
                }
                _inject_batch_url(stale)
                if not _resolve_draft_mutation_claim(
                    platform,
                    platform_id,
                    "recall",
                    claim_fingerprint,
                    stale,
                ):
                    return _locked_mutation_result(
                        str(batch_id),
                        "撤回失败结果无法安全保存；已锁定原批次。",
                    )
                return stale
            if response.status_code >= 500:
                data["success"] = False
                data["uncertain"] = True
                data["message"] = render_remediation_reply(
                    "撤回结果无法确认；请求可能已经生效",
                    command="查看草稿",
                )
            _inject_batch_url(data)
            if _definitive_mutation_response(response.status_code, data):
                if not _resolve_draft_mutation_claim(
                    platform,
                    platform_id,
                    "recall",
                    claim_fingerprint,
                    data,
                ):
                    return _locked_mutation_result(
                        str(batch_id),
                        render_remediation_reply(
                            "撤回结果无法安全保存；已锁定原批次",
                            command="查看草稿",
                        ),
                    )
            elif response.status_code < 500:
                data["success"] = False
                data["uncertain"] = True
                data["message"] = render_remediation_reply(
                    "撤回接口未返回同步终态；已锁定原批次",
                    command="查看草稿",
                )
            return data

    except asyncio.CancelledError:
        if batch_id:
            logger.warning(
                "Recall cancelled with an unresolved request claim: %s",
                batch_id,
            )
        raise
    except httpx.TimeoutException:
        if batch_id:
            uncertain = {
                "success": False,
                "uncertain": True,
                "batchId": batch_id,
                "message": render_remediation_reply(
                    "撤回请求超时，结果可能已经生效",
                    command="查看草稿",
                ),
            }
            _inject_batch_url(uncertain)
            return uncertain
        if existing_recall_payload is not None:
            return _locked_mutation_result(
                str(existing_recall_payload.get("batchId") or ""),
                "撤回核验请求超时；已锁定原批次，不会选择新的提交批次。",
            )
        return {
            "success": False,
            "message": _draft_tool_failure(
                "撤回预检请求超时",
                command="查看草稿",
            ),
        }
    except Exception as e:
        if batch_id:
            uncertain = {
                "success": False,
                "uncertain": True,
                "batchId": batch_id,
                "message": render_remediation_reply(
                    "撤回结果无法确认",
                    command="查看草稿",
                ),
            }
            _inject_batch_url(uncertain)
            return uncertain
        if existing_recall_payload is not None:
            return _locked_mutation_result(
                str(existing_recall_payload.get("batchId") or ""),
                "撤回核验失败；已锁定原批次，不会选择新的提交批次。",
            )
        return {
            "success": False,
            "message": _draft_tool_failure(
                "撤回核验暂时不可用",
                command="查看草稿",
                error=e,
                log_context="recall_batch",
            ),
        }


async def keytao_list_draft_items(
    platform: str,
    platform_id: str,
    batch_id: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
) -> Dict:
    """
    List all PR items in the user's latest draft batch.

    ``offset``/``limit`` select the model-visible window in the harness; this
    function deliberately returns the full item set so deterministic item-set
    validation continues to consume full-fidelity server data.
    """
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()

    if not BOT_API_TOKEN:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}

    if platform == "web-anon":
        return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}

    url = f"{KEYTAO_API_BASE}/api/bot/batches/latest-draft/items"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.get(
                    url,
                    headers=get_bot_headers(
                        platform,
                        platform_id,
                        method="GET",
                        path="/api/bot/batches/latest-draft/items",
                    ),
                    params={
                        "platform": platform,
                        "platformId": platform_id,
                        **({"batchId": batch_id} if batch_id else {}),
                    },
                ),
                method="GET",
                url="/api/bot/batches/latest-draft/items",
            )

            try:
                data = response.json()
            except Exception:
                logger.error(f"[keytao_list_draft_items] Non-JSON response ({response.status_code}): {response.text[:200]}")
                return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}

            logger.info(f"[keytao_list_draft_items] status={response.status_code} count={data.get('count', 0)}")
            if data.get("success") and isinstance(data.get("items"), list):
                data["items"] = [enrich_pr_item_labels(item) for item in data["items"]]
                data["summary"] = compute_draft_summary(data["items"])
            _inject_batch_url(data)
            return data

    except httpx.TimeoutException:
        return {
            "success": False,
            "message": _draft_tool_failure(
                "查看草稿请求超时",
                command="查看草稿",
            ),
        }
    except httpx.TransportError as e:
        return {
            "success": False,
            "message": _draft_tool_failure(
                "查看草稿时网络暂时不可用",
                command="查看草稿",
                error=e,
                log_context="list_draft_items",
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "message": _draft_tool_failure(
                "查看草稿暂时不可用",
                command="查看草稿",
                error=e,
                log_context="list_draft_items",
            ),
        }


async def keytao_update_draft_item_weight(
    platform: str,
    platform_id: str,
    word: str,
    code: str,
    weight: int,
) -> Dict:
    """Update one exact server-known draft item through a CAS-bound route."""
    word = str(word or "").strip()
    code = str(code or "").strip().lower()
    if (
        not word
        or not code
        or not isinstance(weight, int)
        or isinstance(weight, bool)
    ):
        return {
            "success": False,
            "message": "权重调整需要完整的词条、编码和整数权重，本次未写入。",
        }

    snapshot = await keytao_list_draft_items(platform, platform_id)
    if not snapshot.get("success"):
        return snapshot
    matches = [
        item
        for item in snapshot.get("items", [])
        if isinstance(item, dict)
        and str(item.get("word") or "").strip() == word
        and str(item.get("code") or "").strip().lower() == code
    ]
    if len(matches) != 1:
        reason = "没有找到" if not matches else "找到多条"
        return {
            "success": False,
            "message": render_remediation_reply(
                f"草稿中{reason}“{word}” {code} 的唯一条目，本次未写入",
                command="查看草稿",
            ),
            "matchedCount": len(matches),
        }
    item = matches[0]
    phrase_type = str(item.get("type") or "").strip()
    base_weight = PHRASE_TYPE_BASE_WEIGHTS.get(phrase_type)
    if base_weight is None:
        return {
            "success": False,
            "message": f"草稿条目类型“{phrase_type or '未知'}”没有可验证的基础权重，本次未写入。",
        }
    if weight < base_weight:
        return {
            "success": False,
            "belowBaseWeight": True,
            "type": phrase_type,
            "baseWeight": base_weight,
            "requestedWeight": weight,
            "message": (
                f"{TYPE_LABELS[phrase_type]}的基础权重是 {base_weight}，"
                f"不能调整为 {weight}，也不会自动钳制；本次未写入。"
                "如需让它更靠前，请提高同码同类型中其他词条的权重。"
            ),
        }

    batch_id = str(snapshot.get("batchId") or "").strip()
    content_version = snapshot.get("contentVersion")
    item_id = item.get("id")
    if (
        not batch_id
        or not isinstance(content_version, int)
        or isinstance(content_version, bool)
        or content_version < 0
        or not str(item_id).isdigit()
    ):
        return {
            "success": False,
            "message": "草稿快照缺少可验证的批次、版本或条目 ID，本次未写入。",
        }
    canonical_target = {
        "id": int(item_id),
        "word": word,
        "code": code,
        "action": str(item.get("action") or ""),
        "type": phrase_type,
        "weight": item.get("weight"),
    }
    path = f"/api/bot/pull-requests/{int(item_id)}"
    request_data = {
        "platform": platform,
        "platformId": platform_id,
        "batchId": batch_id,
        "expectedContentVersion": content_version,
        "expectedTarget": canonical_target,
        "weight": weight,
    }
    request_body = _json_request_body(request_data)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.request(
                    "PATCH",
                    f"{get_keytao_url()}{path}",
                    headers=get_bot_headers(
                        platform,
                        platform_id,
                        content_type=True,
                        method="PATCH",
                        path=path,
                        raw_body=request_body,
                    ),
                    content=request_body,
                ),
                method="PATCH",
                url=path,
            )
            try:
                data = response.json()
            except Exception:
                return {
                    "success": False,
                    "uncertain": True,
                    "batchId": batch_id,
                    "message": render_remediation_reply(
                        "权重调整结果无法确认；不要直接重试",
                        command="查看草稿",
                    ),
                }
            data.setdefault("batchId", batch_id)
            if response.status_code == 409:
                data.update({
                    "success": False,
                    "staleConfirmation": True,
                    "message": data.get("message") or "草稿条目或版本已变化，本次未写入。",
                })
            _inject_batch_url(data)
            return data
    except httpx.TimeoutException:
        return {
            "success": False,
            "uncertain": True,
            "batchId": batch_id,
            "message": render_remediation_reply(
                "权重调整请求超时；不要直接重试",
                command="查看草稿",
            ),
        }
    except httpx.TransportError as error:
        return {
            "success": False,
            "message": f"权重调整网络失败：{type(error).__name__}，本次结果未确认。",
        }


def _canonical_delete_target(item: Dict) -> Dict:
    return {
        "id": int(item.get("id")),
        "word": str(item.get("word") or ""),
        "code": str(item.get("code") or ""),
        "action": str(item.get("action") or ""),
        "type": str(item.get("type") or ""),
    }


async def _prepare_delete_targets(
    platform: str,
    platform_id: str,
    ids: List[int],
    *,
    batch_id: Optional[str] = None,
) -> Dict:
    snapshot = await keytao_list_draft_items(
        platform,
        platform_id,
        batch_id=batch_id,
    )
    if not snapshot.get("success"):
        return snapshot
    current_batch_id = str(snapshot.get("batchId") or "")
    content_version = snapshot.get("contentVersion")
    if (
        not current_batch_id
        or not isinstance(content_version, int)
        or isinstance(content_version, bool)
        or content_version < 0
    ):
        return _inject_known_batch_url(
            {"success": False, "message": "草稿快照缺少可验证版本"},
            current_batch_id,
        )
    if batch_id and current_batch_id != str(batch_id):
        return _inject_known_batch_url({
            "success": False,
            "staleConfirmation": True,
            "message": "草稿查询返回了不同批次，已停止删除",
            "batchId": current_batch_id,
        }, current_batch_id)
    items_by_id = {
        int(item["id"]): item
        for item in snapshot.get("items", [])
        if isinstance(item, dict)
        and isinstance(item.get("id"), (int, str))
        and str(item.get("id")).isdigit()
    }
    requested_ids = list(dict.fromkeys(int(item_id) for item_id in ids))
    missing_ids = [item_id for item_id in requested_ids if item_id not in items_by_id]
    if missing_ids:
        return _inject_known_batch_url({
            "success": False,
            "staleConfirmation": True,
            "message": f"草稿条目已变化，找不到 ID：{missing_ids}",
        }, current_batch_id)
    targets = [_canonical_delete_target(items_by_id[item_id]) for item_id in requested_ids]
    digest_payload = {
        "batchId": current_batch_id,
        "contentVersion": content_version,
        "targets": targets,
    }
    digest = hashlib.sha256(json.dumps(
        digest_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    preview = {
        "success": True,
        "batchId": current_batch_id,
        "contentVersion": content_version,
        "targets": targets,
        "targetDigest": digest,
    }
    _inject_batch_url(preview)
    return preview


def _locked_mutation_result(batch_id: str, message: str) -> Dict:
    return _inject_known_batch_url({
        "success": False,
        "uncertain": True,
        "batchId": batch_id,
        "message": render_remediation_reply(
            message.rstrip() + "；网站状态核对属于站外操作",
            command="放弃不确定操作",
        ),
    }, batch_id)


def _resolve_draft_mutation_claim(
    platform: str,
    platform_id: str,
    operation_kind: str,
    fingerprint: str,
    result: Dict,
) -> bool:
    try:
        return _draft_mutation_claims().resolve(
            platform,
            platform_id,
            operation_kind,
            fingerprint,
            result,
        )
    except Exception as error:
        logger.error(
            "Failed to resolve draft mutation claim %s: %s: %s",
            operation_kind,
            type(error).__name__,
            error,
        )
        return False


def _replay_resolved_mutation_claim(claim: Dict) -> Optional[Dict]:
    """Return a cached final response without issuing another write."""
    if str(claim.get("status") or "") != "resolved":
        return None
    result = claim.get("result")
    if not isinstance(result, dict):
        return None
    replay = dict(result)
    replay["replayedResolvedMutation"] = True
    payload = claim.get("payload") if isinstance(claim.get("payload"), dict) else {}
    return _inject_known_batch_url(
        replay,
        str(replay.get("batchId") or payload.get("batchId") or ""),
    )


def _definitive_mutation_response(status_code: int, data: Dict) -> bool:
    """Accept only synchronous JSON terminal responses for claim resolution."""
    if 200 <= status_code < 300:
        return status_code != 202 and isinstance(data.get("success"), bool)
    return 400 <= status_code < 500


async def _resolve_existing_delete_claim(
    platform: str,
    platform_id: str,
    requested_ids: List[int],
    batch_id: Optional[str],
) -> tuple[Optional[Dict], Optional[str]]:
    """Resolve an earlier uncertain delete without selecting any new targets."""
    try:
        claim = _draft_mutation_claims().get(platform, platform_id)
    except Exception as error:
        logger.error(
            "Failed to read delete claim: %s: %s",
            type(error).__name__,
            error,
        )
        return _locked_mutation_result(
            str(batch_id or ""),
            "无法读取上一次删除操作的安全状态，本次未执行删除。",
        ), None
    if claim is None:
        return None, None

    if str(claim.get("operationKind") or "") != "delete":
        payload = claim.get("payload") if isinstance(claim, dict) else None
        result = claim.get("result") if isinstance(claim, dict) else None
        if (
            str(claim.get("operationKind") or "") == "recall"
            and str(claim.get("status") or "") == "resolved"
            and isinstance(payload, dict)
            and isinstance(result, dict)
            and str(payload.get("batchId") or "") == str(batch_id or "")
            and result.get("success") is True
            and str(result.get("batchId") or "") == str(batch_id or "")
        ):
            return None, None
        return _locked_mutation_result(
            str((payload or {}).get("batchId") or batch_id or ""),
            "上一次草稿操作结果仍不确定；已锁定原操作，本次不会删除新目标。",
        ), None

    replay = _replay_resolved_mutation_claim(claim)
    if replay is not None:
        return replay, None

    payload = claim.get("payload") if isinstance(claim, dict) else None
    fingerprint = str(claim.get("fingerprint") or "") if isinstance(claim, dict) else ""
    claimed_batch_id = str((payload or {}).get("batchId") or "")
    claimed_version = (payload or {}).get("contentVersion")
    claimed_targets = (payload or {}).get("targets")
    claimed_ids = (payload or {}).get("ids")
    if (
        not claimed_batch_id
        or not fingerprint
        or not isinstance(claimed_version, int)
        or isinstance(claimed_version, bool)
        or not isinstance(claimed_targets, list)
        or not isinstance(claimed_ids, list)
    ):
        return _locked_mutation_result(
            claimed_batch_id or str(batch_id or ""),
            "上一次删除操作的安全记录不完整，本次未执行删除。",
        ), None

    snapshot = await keytao_list_draft_items(
        platform,
        platform_id,
        batch_id=claimed_batch_id,
    )
    if (
        not snapshot.get("success")
        or str(snapshot.get("batchId") or "") != claimed_batch_id
        or str(snapshot.get("status") or "") != "Draft"
        or not isinstance(snapshot.get("items"), list)
    ):
        return _locked_mutation_result(
            claimed_batch_id,
            "上一次删除结果仍无法确认；已锁定原批次，不会改动新的草稿目标。",
        ), None

    current_targets = []
    for item in snapshot.get("items") or []:
        if not isinstance(item, dict) or not str(item.get("id") or "").isdigit():
            return _locked_mutation_result(
                claimed_batch_id,
                "草稿核验结果不完整；已锁定原批次，不会继续删除。",
            ), None
        current_targets.append(_canonical_delete_target(item))
    current_by_id = {target["id"]: target for target in current_targets}
    normalized_claimed_ids = [
        int(item_id)
        for item_id in claimed_ids
        if isinstance(item_id, int)
        and not isinstance(item_id, bool)
        and item_id > 0
    ]
    if len(normalized_claimed_ids) != len(claimed_ids):
        return _locked_mutation_result(
            claimed_batch_id,
            "上一次删除目标记录不完整，本次未执行删除。",
        ), None

    if all(item_id not in current_by_id for item_id in normalized_claimed_ids):
        result = {
            "success": True,
            "alreadyApplied": True,
            "batchId": claimed_batch_id,
            "successCount": len(normalized_claimed_ids),
            "draftItems": snapshot.get("items") or [],
            "message": "上一次删除已通过只读草稿快照确认生效",
        }
        batch_url = snapshot.get("batchUrl")
        if batch_url:
            result["batchUrl"] = batch_url
        result = _inject_known_batch_url(result, claimed_batch_id)
        if not _resolve_draft_mutation_claim(
            platform,
            platform_id,
            "delete",
            fingerprint,
            result,
        ):
            return _locked_mutation_result(
                claimed_batch_id,
                "原删除结果已核验，但最终回执无法安全保存；本次不会继续删除。",
            ), None
        return result, None

    current_claimed_targets = [
        current_by_id.get(item_id)
        for item_id in normalized_claimed_ids
    ]
    same_unchanged_snapshot = bool(
        snapshot.get("contentVersion") == claimed_version
        and current_claimed_targets == claimed_targets
    )
    same_retry_target = bool(
        str(batch_id or "") == claimed_batch_id
        and list(requested_ids) == normalized_claimed_ids
    )
    if same_unchanged_snapshot and same_retry_target:
        return None, fingerprint

    return _locked_mutation_result(
        claimed_batch_id,
        "上一次删除结果仍不确定；已锁定原批次和原目标，不会删除变化后的内容。",
    ), None


def _begin_delete_claim(
    platform: str,
    platform_id: str,
    batch_id: str,
    content_version: int,
    target_digest: str,
    targets: List[Dict],
    ids: List[int],
) -> Optional[str]:
    payload = {
        "batchId": batch_id,
        "contentVersion": content_version,
        "targetDigest": target_digest,
        "targets": targets,
        "ids": ids,
    }
    try:
        store = _draft_mutation_claims()
        fingerprint = store.begin(
            platform,
            platform_id,
            "delete",
            payload,
        )
        if fingerprint:
            return fingerprint
        existing = store.get(platform, platform_id)
        existing_payload = (
            existing.get("payload")
            if isinstance(existing, dict)
            and isinstance(existing.get("payload"), dict)
            else {}
        )
        existing_result = (
            existing.get("result")
            if isinstance(existing, dict)
            and isinstance(existing.get("result"), dict)
            else {}
        )
        if (
            isinstance(existing, dict)
            and existing.get("status") == "resolved"
            and existing.get("operationKind") == "recall"
            and existing_payload.get("batchId") == batch_id
            and existing_result.get("success") is True
            and existing_result.get("batchId") == batch_id
        ):
            payload["continuation"] = "recall_clear"
            return store.transition_resolved(
                platform,
                platform_id,
                "recall",
                str(existing.get("fingerprint") or ""),
                "delete",
                payload,
            )
        return None
    except Exception as error:
        logger.error(
            "Failed to acquire delete claim: %s: %s",
            type(error).__name__,
            error,
        )
        return None


async def keytao_remove_draft_item(
    platform: str,
    platform_id: str,
    pr_id: int,
    batch_id: Optional[str] = None,
    expected_content_version: Optional[int] = None,
    expected_target_digest: str = "",
    expected_targets: Optional[List[Dict]] = None,
) -> Dict:
    """
    Remove a specific PR item from the user's draft batch
    从用户草稿批次中删除指定的词条条目

    Args:
        platform: Platform type ('qq' or 'telegram')
        platform_id: User's platform ID
        pr_id: The numeric ID of the PR to delete (obtainable from keytao_list_draft_items)
    """
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()

    if not BOT_API_TOKEN:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}

    safe_pr_id = _safe_numeric_id(pr_id)
    if safe_pr_id is None:
        logger.error(f"[keytao_remove_draft_item] rejected invalid PR id: {pr_id!r}")
        return {"success": False, "message": f"条目 ID 非法：{pr_id}，必须是正整数"}
    pr_id = safe_pr_id

    existing_claim_result, reusable_claim_fingerprint = await _resolve_existing_delete_claim(
        platform,
        platform_id,
        [pr_id],
        batch_id,
    )
    if existing_claim_result is not None:
        return existing_claim_result

    preview = await _prepare_delete_targets(
        platform,
        platform_id,
        [pr_id],
        batch_id=batch_id,
    )
    if not preview.get("success"):
        return preview
    if not expected_target_digest:
        return {
            **preview,
            "success": False,
            "requiresConfirmation": True,
            "confirmationKind": "deleteTargets",
            "message": "删除会永久移除以下草稿条目，请核对完整目标",
        }
    if (
        batch_id != preview.get("batchId")
        or expected_content_version != preview.get("contentVersion")
        or expected_target_digest != preview.get("targetDigest")
        or expected_targets != preview.get("targets")
    ):
        return _inject_known_batch_url({
            "success": False,
            "staleConfirmation": True,
            "message": "删除目标或草稿版本已变化，旧确认已失效",
        }, str(preview.get("batchId") or batch_id or ""))

    claim_fingerprint = reusable_claim_fingerprint or _begin_delete_claim(
        platform,
        platform_id,
        str(batch_id or ""),
        int(expected_content_version),
        expected_target_digest,
        expected_targets or [],
        [pr_id],
    )
    if not claim_fingerprint:
        return _locked_mutation_result(
            str(batch_id or ""),
            "另一个草稿操作正在核验，本次未执行删除。",
        )

    url = f"{KEYTAO_API_BASE}/api/bot/pull-requests/{pr_id}"
    request_data = {
        "platform": platform,
        "platformId": platform_id,
        "batchId": batch_id,
        "expectedContentVersion": expected_content_version,
        "expectedTargets": expected_targets,
    }
    request_body = _json_request_body(request_data)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.request(
                    "DELETE",
                    url,
                    headers=get_bot_headers(
                        platform,
                        platform_id,
                        content_type=True,
                        method="DELETE",
                        path=f"/api/bot/pull-requests/{pr_id}",
                        raw_body=request_body,
                    ),
                    content=request_body,
                ),
                method="DELETE",
                url=f"/api/bot/pull-requests/{pr_id}",
            )

            try:
                data = response.json()
            except Exception:
                logger.error(f"[keytao_remove_draft_item] Non-JSON response ({response.status_code}): {response.text[:200]}")
                uncertain = {
                    "success": False,
                    "uncertain": True,
                    "batchId": batch_id,
                    "message": render_remediation_reply(
                        "删除结果无法确认；请求可能已经生效",
                        command="查看草稿",
                    ),
                }
                _inject_batch_url(uncertain)
                return uncertain

            logger.info(f"[keytao_remove_draft_item] PR#{pr_id} status={response.status_code}")
            data.setdefault("batchId", batch_id)
            if response.status_code == 409:
                stale = {
                    **data,
                    "success": False,
                    "staleConfirmation": True,
                    "message": data.get("message") or "删除目标已变化，旧票据已作废",
                }
                _inject_batch_url(stale)
                if not _resolve_draft_mutation_claim(
                    platform,
                    platform_id,
                    "delete",
                    claim_fingerprint,
                    stale,
                ):
                    return _locked_mutation_result(
                        str(batch_id or ""),
                        "删除失败结果无法安全保存；已锁定原批次。",
                    )
                return stale
            if response.status_code >= 500:
                data["success"] = False
                data["uncertain"] = True
                data["message"] = render_remediation_reply(
                    "删除结果无法确认；请求可能已经生效",
                    command="查看草稿",
                )
            if data.get("success"):
                snapshot = await _fetch_draft_snapshot(
                    platform, platform_id, str(data.get("batchId") or batch_id or "") or None
                )
                if snapshot is not None:
                    data["draft_snapshot"] = snapshot
            _inject_batch_url(data)
            if _definitive_mutation_response(response.status_code, data):
                if not _resolve_draft_mutation_claim(
                    platform,
                    platform_id,
                    "delete",
                    claim_fingerprint,
                    data,
                ):
                    return _locked_mutation_result(
                        str(batch_id or ""),
                        render_remediation_reply(
                            "删除结果无法安全保存；已锁定原批次",
                            command="查看草稿",
                        ),
                    )
            elif response.status_code < 500:
                data["success"] = False
                data["uncertain"] = True
                data["message"] = render_remediation_reply(
                    "删除接口未返回同步终态；已锁定原批次",
                    command="查看草稿",
                )
            return data

    except asyncio.CancelledError:
        logger.warning(
            "Delete draft item cancelled with an unresolved request claim: %s",
            batch_id,
        )
        raise
    except httpx.TimeoutException:
        uncertain = {
            "success": False,
            "uncertain": True,
            "batchId": batch_id,
            "message": render_remediation_reply(
                "删除请求超时，结果可能已经生效",
                command="查看草稿",
            ),
        }
        _inject_batch_url(uncertain)
        return uncertain
    except Exception as e:
        logger.error(f"Remove draft item error: {e}")
        uncertain = {
            "success": False,
            "uncertain": True,
            "batchId": batch_id,
            "message": render_remediation_reply(
                "删除结果无法确认",
                command="查看草稿",
            ),
        }
        _inject_batch_url(uncertain)
        return uncertain


# Tool definitions for OpenAI Function Calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "keytao_create_phrase",
            "description": (
                "创建、修改或删除键道词条。新词相对现有词的前后位置也调用本工具，code 传参照词所在编码；"
                "执行器会根据同码标记、服务端候选链和占用快照决定顺延、后续空位或同码权重。"
                "信息性单条重码由程序绑定服务端票据自动确认一次，其他警告保留确认流程。自动追加到草稿批次。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "要操作的词条内容（中文词组）"
                    },
                    "code": {
                        "type": "string",
                        "description": "键道输入法编码（纯字母）"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["Create", "Change", "Delete"],
                        "description": "操作类型：Create（创建）、Change（修改）、Delete（删除），默认为 Create"
                    },
                    "old_word": {
                        "type": "string",
                        "description": "【Change 操作必填，不传会被后端拒绝】修改前的原词条内容。必须先调用 keytao_lookup_by_codes_batch 查出该编码当前的词，再将查询结果填入此字段。例如：lookup 返回 fpnm 当前词为\"防粘\"，则 old_word=\"防粘\"，word=\"防黏\"。"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["Single", "Phrase", "Supplement", "Symbol", "Link", "CSS", "CSSSingle", "English"],
                        "description": "词条类型。用户明确指定类型时必须传：声笔笔=CSS，声笔笔单字=CSSSingle，词组=Phrase，单字=Single，补充=Supplement，符号=Symbol，链接=Link，英文=English。Change/Delete 若不传会默认词组，可能改错词库。"
                    },
                    "remark": {
                        "type": "string",
                        "description": "可选的备注说明"
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "仅供程序回放已保存的精确服务端确认状态；模型不得设置。默认false"
                    }
                },
                "required": ["word", "code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_submit_batch",
            "description": '提交当前草稿批次进行审核。仅当用户明确说"提交"、"提审"、"发起审核"、"submit"时才调用，不得因"确认"、"好"、"是"等模糊词而触发。',
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmed": {
                        "type": "boolean",
                        "description": "⚠️ 重要：当提交返回重码警告（requiresConfirmation=true）后，用户确认时必须设置为true。默认false"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_list_draft_items",
            "description": "查看当前草稿批次中所有待审词条。用于用户询问草稿内容、想确认已添加了哪些词条时调用。返回条目列表包含 id、词条、编码、操作类型。",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {
                        "type": "string",
                        "description": "可选：要查看的批次编号，必须是本轮工具结果里出现过的 batchId；不传表示当前草稿"
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "可选：模型可见分页起点，从 0 开始；默认 0"
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "可选：模型可见分页条数，最多 50；默认 50"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_remove_draft_item",
            "description": "从草稿批次中删除指定词条。用于用户要撤销、取消或删除某个已添加的词条时调用。需要先用 keytao_list_draft_items 获取条目 ID。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pr_id": {
                        "type": "integer",
                        "description": "要删除的条目 ID（从 keytao_list_draft_items 返回的 items[].id 获取）"
                    }
                },
                "required": ["pr_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_update_draft_item_weight",
            "description": (
                "调整草稿中一个已存在词条的权重。必须先调用 keytao_list_draft_items，"
                "并且只能使用其返回的唯一 word/code；工具会再次读取快照并以版本号和"
                "完整目标做 CAS 校验。权重低于类型基础值时拒绝，不会自动钳制。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {"type": "string", "description": "草稿中的完整词条"},
                    "code": {"type": "string", "description": "草稿中的完整编码"},
                    "weight": {"type": "integer", "description": "目标整数权重"},
                },
                "required": ["word", "code", "weight"],
            },
        },
    }
]


async def keytao_batch_add_to_draft(
    platform: str,
    platform_id: str,
    items: List[Dict],
    batch_id: Optional[str] = None,
    confirmed: bool = False,
    expected_content_version: Optional[int] = None,
    expected_warning_digest: str = "",
    preview_only: bool = False,
) -> Dict:
    """
    Batch add word entries to draft (tolerant mode).
    The first call is a complete read-only preview. Any confirmed write must
    carry the exact batch, content version, and warning digest from the server.

    Args:
        platform: Platform type ('qq' or 'telegram')
        platform_id: User's platform ID
        items: List of dicts with keys: word, code, action (optional), type (optional), remark (optional)
        batch_id: Optional existing draft batch ID
        confirmed: Whether the exact server preview ticket is being confirmed

    Returns:
        dict with successCount, failedCount, skippedCount, failed[], skipped[], draftItems[], draftTotal
    """
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()

    if not BOT_API_TOKEN:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}

    if not batch_id:
        try:
            batch_id = await get_latest_draft_batch(platform, platform_id)
        except UserNotFoundError:
            return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
    absence_baseline = not batch_id
    # A missing draft batch is no longer an error: the server creates one on
    # demand for the first write.  Only a confirmed ticket must still name the
    # exact batch it was issued against.
    if confirmed and not batch_id:
        return _inject_known_batch_url({
            "success": False,
            "message": "批量添加确认缺少目标批次，已停止执行",
        }, batch_id)
    if confirmed and (
        not isinstance(expected_content_version, int)
        or isinstance(expected_content_version, bool)
        or expected_content_version < 0
        or not re.fullmatch(r"[0-9a-f]{64}", expected_warning_digest)
    ):
        return _inject_known_batch_url({
            "success": False,
            "message": "批量添加确认缺少有效的服务端风险快照",
        }, batch_id)

    valid_items, validation_failed = await _split_items_by_code_validation(items)
    if validation_failed and not valid_items:
        result = {
            "success": False,
            "message": f"{len(validation_failed)} 条编码校验失败，未写入草稿",
            "batchId": batch_id,
            "batchUrl": make_batch_url(batch_id),
            "successCount": 0,
            "failedCount": len(validation_failed),
            "skippedCount": 0,
            "failed": validation_failed,
            "skipped": [],
            "draftItems": [],
            "draftTotal": 0,
        }
        snapshot = await _fetch_draft_snapshot(platform, platform_id, batch_id)
        if snapshot is not None:
            result["draft_snapshot"] = snapshot
            result["draftItems"] = snapshot.get("items", [])
            result["draftTotal"] = snapshot.get("count", 0)
        return result

    url = f"{KEYTAO_API_BASE}/api/bot/pull-requests/batch-draft"
    request_data = {
        "platform": platform,
        "platformId": platform_id,
        "batchId": batch_id,
        "confirmed": confirmed,
        "previewOnly": preview_only,
        **(
            {
                "expectedContentVersion": expected_content_version,
                "expectedWarningDigest": expected_warning_digest,
            }
            if confirmed
            else {}
        ),
        "items": [
            {**{k: v for k, v in item.items() if k != "old_word"},
             **({"oldWord": item["old_word"]} if "old_word" in item else {})}
            for item in valid_items
        ],
    }

    logger.info(
        f"[keytao_batch_add_to_draft] Sending {len(valid_items)} items to batch-draft "
        f"({len(validation_failed)} validation failures)"
    )
    request_body = _json_request_body(request_data)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.post(
                    url,
                    headers=get_bot_headers(
                        platform,
                        platform_id,
                        content_type=True,
                        method="POST",
                        path="/api/bot/pull-requests/batch-draft",
                        raw_body=request_body,
                    ),
                    content=request_body,
                ),
                method="POST",
                url="/api/bot/pull-requests/batch-draft",
                idempotent=preview_only,
            )
            try:
                data = response.json()
            except Exception:
                result = {
                    "success": False,
                    "message": f"API 返回异常（HTTP {response.status_code}）",
                }
                if confirmed:
                    result.update({
                        "uncertain": True,
                        "message": render_remediation_reply(
                            "批量添加结果无法确认；请求可能已经生效",
                            command="查看草稿",
                        ),
                    })
                return _inject_known_batch_url(result, batch_id)

            logger.info(
                f"[keytao_batch_add_to_draft] status={response.status_code} "
                f"success={data.get('successCount',0)} failed={data.get('failedCount',0)}"
            )
            # Enrich draft item labels
            if isinstance(data.get("draftItems"), list):
                data["draftItems"] = [enrich_pr_item_labels(item) for item in data["draftItems"]]
            data.setdefault("batchId", batch_id)
            if preview_only and not data.get("requiresConfirmation"):
                return _inject_known_batch_url({
                    "success": False,
                    "uncertain": True,
                    "message": "服务端未返回可确认的批量添加快照，已停止后续操作",
                    "batchId": batch_id,
                }, batch_id)
            if validation_failed:
                data["failed"] = [*data.get("failed", []), *validation_failed]
                data["failedCount"] = data.get("failedCount", 0) + len(validation_failed)
                data["message"] = (
                    f"{data.get('message', '已处理草稿')}；"
                    f"{len(validation_failed)} 条编码校验失败未写入"
                )
            if data.get("requiresConfirmation") and absence_baseline:
                _mark_provisional_batch(data)
            _inject_batch_url(data)
            return data

    except httpx.TimeoutException:
        if preview_only:
            return _inject_known_batch_url({
                "success": False,
                "message": _draft_tool_failure(
                    "批量添加预检超时",
                    command="查看草稿",
                ),
            }, batch_id)
        result = {
            "success": False,
            "uncertain": True,
            "message": render_remediation_reply(
                "批量添加请求超时，草稿可能已经写入；不要直接重试",
                command="查看草稿",
            ),
        }
        return _inject_known_batch_url(result, batch_id)
    except Exception as e:
        result = {
            "success": False,
            "message": (
                render_remediation_reply(
                    "批量添加结果无法确认",
                    command="查看草稿",
                )
                if confirmed
                else _draft_tool_failure(
                    "批量添加服务暂时不可用",
                    command="查看草稿",
                    error=e,
                    log_context="batch_add_to_draft",
                )
            ),
        }
        if confirmed:
            result["uncertain"] = True
        return _inject_known_batch_url(result, batch_id)


async def keytao_batch_remove_draft_items(
    platform: str,
    platform_id: str,
    ids: list[int],
    batch_id: Optional[str] = None,
    expected_content_version: Optional[int] = None,
    expected_target_digest: str = "",
    expected_targets: Optional[List[Dict]] = None,
) -> Dict:
    """Batch delete draft items by their PR IDs."""
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()

    if not BOT_API_TOKEN:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}

    safe_ids: List[int] = []
    invalid_ids: List[object] = []
    for raw_id in ids or []:
        safe_id = _safe_numeric_id(raw_id)
        if safe_id is None:
            invalid_ids.append(raw_id)
        else:
            safe_ids.append(safe_id)
    if invalid_ids:
        logger.error(
            "[keytao_batch_remove_draft_items] rejected invalid PR ids: %r",
            invalid_ids,
        )
        return {
            "success": False,
            "message": f"条目 ID 非法：{invalid_ids}，必须全部是正整数",
        }
    ids = safe_ids

    existing_claim_result, reusable_claim_fingerprint = await _resolve_existing_delete_claim(
        platform,
        platform_id,
        ids,
        batch_id,
    )
    if existing_claim_result is not None:
        return existing_claim_result

    preview = await _prepare_delete_targets(
        platform,
        platform_id,
        ids,
        batch_id=batch_id,
    )
    if not preview.get("success"):
        return preview
    if not expected_target_digest:
        return {
            **preview,
            "success": False,
            "requiresConfirmation": True,
            "confirmationKind": "deleteTargets",
            "message": "批量删除会永久移除以下草稿条目，请核对完整目标",
        }
    if (
        batch_id != preview.get("batchId")
        or expected_content_version != preview.get("contentVersion")
        or expected_target_digest != preview.get("targetDigest")
        or expected_targets != preview.get("targets")
    ):
        return _inject_known_batch_url({
            "success": False,
            "staleConfirmation": True,
            "message": "批量删除目标或草稿版本已变化，旧确认已失效",
        }, str(preview.get("batchId") or batch_id or ""))

    claim_fingerprint = reusable_claim_fingerprint or _begin_delete_claim(
        platform,
        platform_id,
        str(batch_id or ""),
        int(expected_content_version),
        expected_target_digest,
        expected_targets or [],
        list(ids),
    )
    if not claim_fingerprint:
        return _locked_mutation_result(
            str(batch_id or ""),
            "另一个草稿操作正在核验，本次未执行批量删除。",
        )

    url = f"{KEYTAO_API_BASE}/api/bot/pull-requests/batch-draft"
    payload = {
        "platform": platform,
        "platformId": platform_id,
        "ids": ids,
        "batchId": batch_id,
        "expectedContentVersion": expected_content_version,
        "expectedTargets": expected_targets,
    }
    logger.info(f"[keytao_batch_remove_draft_items] Deleting ids={ids}")
    request_body = _json_request_body(payload)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.request(
                    "DELETE",
                    url,
                    content=request_body,
                    headers=get_bot_headers(
                        platform,
                        platform_id,
                        content_type=True,
                        method="DELETE",
                        path="/api/bot/pull-requests/batch-draft",
                        raw_body=request_body,
                    ),
                ),
                method="DELETE",
                url="/api/bot/pull-requests/batch-draft",
            )
            try:
                data: Dict = response.json()
            except Exception:
                uncertain = {
                    "success": False,
                    "uncertain": True,
                    "batchId": batch_id,
                    "message": render_remediation_reply(
                        "批量删除结果无法确认；请求可能已经生效",
                        command="查看草稿",
                    ),
                }
                _inject_batch_url(uncertain)
                return uncertain
            logger.info(
                f"[keytao_batch_remove_draft_items] status={response.status_code} "
                f"success={data.get('success')} deleted={data.get('successCount')}"
            )
            data.setdefault("batchId", batch_id)
            if response.status_code == 409:
                stale = {
                    **data,
                    "success": False,
                    "staleConfirmation": True,
                    "message": data.get("message") or "批量删除目标已变化，旧票据已作废",
                }
                _inject_batch_url(stale)
                if not _resolve_draft_mutation_claim(
                    platform,
                    platform_id,
                    "delete",
                    claim_fingerprint,
                    stale,
                ):
                    return _locked_mutation_result(
                        str(batch_id or ""),
                        "批量删除失败结果无法安全保存；已锁定原批次。",
                    )
                return stale
            if response.status_code >= 500:
                data["success"] = False
                data["uncertain"] = True
                data["message"] = render_remediation_reply(
                    "批量删除结果无法确认；请求可能已经生效",
                    command="查看草稿",
                )
            if isinstance(data.get("draftItems"), list):
                data["draftItems"] = [enrich_pr_item_labels(item) for item in data["draftItems"]]
            _inject_batch_url(data)
            if _definitive_mutation_response(response.status_code, data):
                if not _resolve_draft_mutation_claim(
                    platform,
                    platform_id,
                    "delete",
                    claim_fingerprint,
                    data,
                ):
                    return _locked_mutation_result(
                        str(batch_id or ""),
                        render_remediation_reply(
                            "批量删除结果无法安全保存；已锁定原批次",
                            command="查看草稿",
                        ),
                    )
            elif response.status_code < 500:
                data["success"] = False
                data["uncertain"] = True
                data["message"] = render_remediation_reply(
                    "批量删除接口未返回同步终态；已锁定原批次",
                    command="查看草稿",
                )
            return data
    except asyncio.CancelledError:
        logger.warning(
            "Batch delete cancelled with an unresolved request claim: %s",
            batch_id,
        )
        raise
    except httpx.TimeoutException:
        uncertain = {
            "success": False,
            "uncertain": True,
            "batchId": batch_id,
            "message": render_remediation_reply(
                "批量删除请求超时，结果可能已经生效",
                command="查看草稿",
            ),
        }
        _inject_batch_url(uncertain)
        return uncertain
    except Exception as e:
        logger.error(f"[keytao_batch_remove_draft_items] Error: {e}")
        uncertain = {
            "success": False,
            "uncertain": True,
            "batchId": batch_id,
            "message": render_remediation_reply(
                "批量删除结果无法确认",
                command="查看草稿",
            ),
        }
        _inject_batch_url(uncertain)
        return uncertain


async def _keytao_strict_batch_add_to_draft(
    platform: str,
    platform_id: str,
    items: List[Dict],
    *,
    batch_id: Optional[str] = None,
    expected_content_version: Optional[int] = None,
    confirmed: bool = False,
    expected_warning_digest: str = "",
) -> Dict:
    """Write one all-or-nothing plan through the strict batch endpoint."""
    bot_api_token = get_bot_token()
    if not bot_api_token:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}

    # A caller that already carries an expected version has a baseline - which
    # may legitimately be "there was no draft" (batch_id None, version 0).
    # Re-resolving it here would silently adopt a batch that appeared after the
    # plan was built, defeating the compare-and-set the server performs.
    if not batch_id and expected_content_version is None:
        try:
            batch_id = await get_latest_draft_batch(platform, platform_id)
        except UserNotFoundError:
            return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
    absence_baseline = not batch_id and expected_content_version == 0
    valid_items, validation_failed = await _split_items_by_code_validation(items)
    if validation_failed or len(valid_items) != len(items):
        return _inject_known_batch_url({
            "success": False,
            "message": "顺延计划未通过整批编码预检，未写入任何草稿条目",
            "failed": validation_failed,
        }, batch_id)

    request_items = [
        {
            **{key: value for key, value in item.items() if key != "old_word"},
            **({"oldWord": item["old_word"]} if "old_word" in item else {}),
        }
        for item in valid_items
    ]
    url = f"{get_keytao_url()}/api/bot/pull-requests/batch"
    request_data = {
        "platform": platform,
        "platformId": platform_id,
        "items": request_items,
        "confirmed": confirmed,
        # Omitted entirely when there is no draft yet: the server materialises
        # one under its own "no draft existed" assertion.
        **({"batchId": batch_id} if batch_id else {}),
        **(
            {
                "expectedContentVersion": expected_content_version,
                **(
                    {"expectedWarningDigest": expected_warning_digest}
                    if confirmed
                    else {}
                ),
            }
            if expected_content_version is not None
            else {}
        ),
    }
    request_body = _json_request_body(request_data)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await http_client.request_with_retries(
                lambda: client.post(
                    url,
                    headers=get_bot_headers(
                        platform,
                        platform_id,
                        content_type=True,
                        method="POST",
                        path="/api/bot/pull-requests/batch",
                        raw_body=request_body,
                    ),
                    content=request_body,
                ),
                method="POST",
                url="/api/bot/pull-requests/batch",
            )
        try:
            data = response.json()
        except Exception:
            result = {
                "success": False,
                "uncertain": True,
                "message": render_remediation_reply(
                    "整批顺延结果无法确认；请求可能已经生效",
                    command="查看草稿",
                ),
            }
            return _inject_known_batch_url(result, batch_id)
        _inject_known_batch_url(data, batch_id)
        if response.status_code == 409:
            return _inject_known_batch_url({
                **data,
                "success": False,
                "staleConfirmation": True,
                "message": data.get("message") or "草稿内容已变化，顺延计划已作废",
            }, batch_id)
        if data.get("requiresConfirmation"):
            if absence_baseline:
                _mark_provisional_batch(data)
            return data
        if not response.is_success or not data.get("success"):
            result = {
                **data,
                "success": False,
                "message": data.get("message") or f"整批顺延写入失败（HTTP {response.status_code}）",
            }
            if response.status_code >= 500:
                result.update({
                    "uncertain": True,
                    "message": render_remediation_reply(
                        "整批顺延结果无法确认；请求可能已经生效",
                        command="查看草稿",
                    ),
                })
            return _inject_known_batch_url(result, batch_id)
        data["successCount"] = int(
            data.get("successCount")
            or data.get("pullRequestCount")
            or len(request_items)
        )
        snapshot = await _fetch_draft_snapshot(
            platform, platform_id, str(data.get("batchId") or batch_id or "") or None
        )
        if snapshot is not None:
            data["draft_snapshot"] = snapshot
        _inject_batch_url(data)
        return data
    except httpx.TimeoutException:
        result = {
            "success": False,
            "uncertain": True,
            "message": render_remediation_reply(
                "整批顺延请求超时；不要立即重试",
                command="查看草稿",
            ),
        }
        return _inject_known_batch_url(result, batch_id)
    except Exception as error:
        logger.error(f"[keytao_shift_phrase_code] strict batch error: {error}")
        result = {
            "success": False,
            "uncertain": True,
            "message": render_remediation_reply(
                "整批顺延结果无法确认",
                command="查看草稿",
            ),
        }
        return _inject_known_batch_url(result, batch_id)


async def keytao_shift_phrase_code(
    platform: str,
    platform_id: str,
    word: str,
    target_code: str,
    confirmed_plan_digest: str = "",
    batch_id: Optional[str] = None,
    expected_content_version: Optional[int] = None,
    expected_warning_digest: str = "",
    target_type: Optional[str] = None,
    target_remark: str = "",
    target_needs_manual_review: Optional[bool] = None,
) -> Dict:
    """Preview, bind, then atomically write a complete code-shift plan."""
    word = word.strip()
    target_code = target_code.strip().lower()
    if not word or not target_code:
        return {"success": False, "message": "必须提供词条和目标编码"}

    target_encode = await _fetch_encode_candidates(word, target_code)
    if not target_encode.get("success"):
        return target_encode
    target_candidate_codes = target_encode.get("candidateCodes", [])
    if target_code not in target_candidate_codes:
        requested_analysis = target_encode.get("requestedCodeAnalysis")
        return {
            "success": False,
            "message": f"{target_code} 不是「{word}」的有效候选编码，可选：{', '.join(target_candidate_codes)}",
            "candidateCodes": target_candidate_codes,
            "requestedCodeAnalysis": requested_analysis,
        }

    word_lookup = await _lookup_words_raw([word])
    if not word_lookup.get("success"):
        return word_lookup
    word_result = next((item for item in word_lookup.get("results", []) if item.get("word") == word), {})
    current_phrase = _select_current_phrase(word, word_result.get("phrases", []))
    if (
        target_needs_manual_review is not None
        and not isinstance(target_needs_manual_review, bool)
    ):
        return {"success": False, "message": "人工审核标记必须是布尔值"}
    if current_phrase is None and target_needs_manual_review is None:
        target_needs_manual_review = True

    ignored_words = {word}
    code_phrase_map: Dict[str, List[Dict]] = {}
    word_candidate_code_map: Dict[str, List[str]] = {word: target_candidate_codes}
    pending_occupants_by_code: Dict[str, List[Dict]] = {}

    async def ensure_code_lookup(code: str) -> Dict:
        if code in code_phrase_map:
            return {"success": True}
        code_lookup = await _lookup_codes_raw([code])
        if not code_lookup.get("success"):
            return code_lookup
        for item in code_lookup.get("results", []):
            item_code = item.get("code", "")
            occupants = _ordered_code_occupants(item.get("phrases", []), ignored_words)
            code_phrase_map[item_code] = list(occupants)
            pending_occupants_by_code[item_code] = list(occupants)
        code_phrase_map.setdefault(code, [])
        pending_occupants_by_code.setdefault(code, [])
        return {"success": True}

    lookup_result = await ensure_code_lookup(target_code)
    if not lookup_result.get("success"):
        return lookup_result

    reserved_codes = {target_code}
    queue: List[Dict] = list(pending_occupants_by_code.get(target_code, []))
    pending_occupants_by_code[target_code] = []

    while queue:
        occupant = queue.pop(0)
        occupant_word = occupant.get("word", "")
        probe_code = occupant.get("code", "")
        occupant_codes = word_candidate_code_map.get(occupant_word)
        if not occupant_codes:
            occupant_encode = await _fetch_encode_candidates(occupant_word)
            if not occupant_encode.get("success"):
                return occupant_encode
            occupant_codes = occupant_encode.get("candidateCodes", [])
            word_candidate_code_map[occupant_word] = occupant_codes
        if probe_code not in occupant_codes:
            return {
                "success": False,
                "message": f"无法顺延「{occupant_word}」：当前编码 {probe_code} 不在它自己的候选编码中",
                "word": occupant_word,
                "candidateCodes": occupant_codes,
            }
        code_index = occupant_codes.index(probe_code)

        found_next = False
        for candidate_code in occupant_codes[code_index + 1:]:
            if candidate_code in reserved_codes:
                continue
            lookup_result = await ensure_code_lookup(candidate_code)
            if not lookup_result.get("success"):
                return lookup_result
            reserved_codes.add(candidate_code)
            evicted = list(pending_occupants_by_code.get(candidate_code, []))
            if evicted:
                queue.extend(evicted)
                pending_occupants_by_code[candidate_code] = []
            found_next = True
            break

        if not found_next:
            return {
                "success": False,
                "message": f"无法顺延「{occupant_word}」：{probe_code} 之后没有可用候选编码",
                "word": occupant_word,
                "candidateCodes": occupant_codes,
            }

    plan = _build_code_shift_plan(
        word,
        target_code,
        target_candidate_codes,
        current_phrase,
        code_phrase_map,
        word_candidate_code_map,
        target_type=target_type,
        target_remark=target_remark,
        target_needs_manual_review=target_needs_manual_review,
    )
    if not plan.get("success"):
        return plan

    planned_words = {item.get("word") for item in plan.get("items", []) if item.get("word")}
    existing_draft = await keytao_list_draft_items(
        platform,
        platform_id,
        batch_id=batch_id,
    )
    if not existing_draft.get("success"):
        return {
            "success": False,
            "message": existing_draft.get("message", "无法读取当前草稿，顺延未执行"),
        }
    related_draft_items = [
        item for item in existing_draft.get("items", [])
        if isinstance(item, dict) and item.get("word") in planned_words
    ]
    if related_draft_items:
        return _inject_known_batch_url({
            "success": False,
            "policyBlocked": True,
            "requiresDraftCleanup": True,
            "message": render_remediation_reply(
                "相关词条已存在于草稿中；为避免非原子地先删后写，"
                "本次顺延未修改草稿；必须由你决定如何处理旧草稿条目",
                command="查看草稿",
            ),
            "relatedDraftItems": related_draft_items[:20],
            "shiftPlan": {
                "word": word,
                "targetCode": target_code,
                "candidateCodes": target_candidate_codes,
                "items": plan.get("items", []),
                "shifted": plan.get("shifted", []),
                "removedDraftIds": [],
            },
        }, str(existing_draft.get("batchId") or batch_id or ""))

    current_batch_id = str(existing_draft.get("batchId") or "")
    current_content_version = existing_draft.get("contentVersion")
    # "No draft at all" is a legitimate baseline, not a missing version: the
    # server states it as batchId=null + contentVersion=0 and enforces CAS on
    # that absence, so a shift right after submitting everything still works.
    # It is part of the digest, so a draft appearing in between voids the plan.
    if not current_batch_id:
        current_content_version = 0
    if (
        not isinstance(current_content_version, int)
        or isinstance(current_content_version, bool)
        or current_content_version < 0
    ):
        return _inject_known_batch_url({
            "success": False,
            "message": "当前草稿缺少可验证的内容版本，顺延未执行",
        }, current_batch_id or batch_id)

    shift_plan = {
        "word": word,
        "targetCode": target_code,
        "candidateCodes": target_candidate_codes,
        "items": plan.get("items", []),
        "shifted": plan.get("shifted", []),
        "removedDraftIds": [],
    }
    digest_payload = {
        "batchId": current_batch_id,
        "contentVersion": current_content_version,
        "shiftPlan": shift_plan,
    }
    plan_digest = hashlib.sha256(json.dumps(
        digest_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()

    if not confirmed_plan_digest:
        preview = {
            "success": False,
            "requiresConfirmation": True,
            "confirmationKind": "shiftPlan",
            "message": "顺延会移动当前编码链中的其他词条，请核对完整计划",
            "batchId": current_batch_id,
            "contentVersion": current_content_version,
            "planDigest": plan_digest,
            "shiftPlan": shift_plan,
        }
        _inject_batch_url(preview)
        return preview
    if (
        confirmed_plan_digest.strip().lower() != plan_digest
        or str(batch_id or "") != current_batch_id
        or expected_content_version != current_content_version
    ):
        return _inject_known_batch_url({
            "success": False,
            "staleConfirmation": True,
            "message": render_remediation_reply(
                "顺延计划或草稿内容已变化，旧确认已失效",
                command=f"顺延「{word}」到 {target_code}",
                words=(word,),
            ),
            "batchId": current_batch_id,
            "contentVersion": current_content_version,
        }, current_batch_id)

    write_result = await _keytao_strict_batch_add_to_draft(
        platform,
        platform_id,
        plan.get("items", []),
        # No draft yet: send no batch id and let the server materialise one
        # under the same absence baseline the plan was built on.
        batch_id=current_batch_id or None,
        expected_content_version=current_content_version,
        confirmed=bool(expected_warning_digest),
        expected_warning_digest=expected_warning_digest,
    )
    write_result["shiftPlan"] = shift_plan
    write_result["planDigest"] = plan_digest
    if plan.get("shifted"):
        shifted_text = "；".join(
            f"{item['word']} {item['fromCode']}→{item['toCode']}"
            for item in plan.get("shifted", [])
        )
        write_result["message"] = f"{write_result.get('message', '已写入草稿')}；顺延：{shifted_text}"
    return _inject_known_batch_url(write_result, current_batch_id)


TOOLS += [
    {
        "type": "function",
        "function": {
            "name": "keytao_batch_add_to_draft",
            "description": (
                "批量将词条加入草稿。适合用户一次提交大量词条时使用。"
                "首次调用只返回完整预览，不写入；所有条目和风险（包括重码、跳过更短空位）都必须由用户明确确认。"
                "确认时必须原样携带服务端返回的 batchId、contentVersion 和 warningDigest，快照变化会拒绝写入。"
                "如需把词插入已占用编码并顺延后续词，必须先使用 keytao_shift_phrase_code，不要手工计算顺延。"
                "操作完成后返回成功数、失败数及当前草稿快照。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "要添加的词条列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "word": {"type": "string", "description": "词条内容"},
                                "code": {"type": "string", "description": "键道编码（纯字母）"},
                                "action": {
                                    "type": "string",
                                    "enum": ["Create", "Change", "Delete"],
                                    "description": "操作类型，默认 Create",
                                },
                                "old_word": {
                                    "type": "string",
                                    "description": "【Change 操作必填】修改前的原词条内容，不传后端会拒绝",
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["Single", "Phrase", "Supplement", "Symbol", "Link", "CSS", "CSSSingle", "English"],
                                    "description": "词条类型。用户明确指定类型时必须传：声笔笔=CSS，声笔笔单字=CSSSingle，词组=Phrase，单字=Single，补充=Supplement，符号=Symbol，链接=Link，英文=English。Change/Delete 若不传会默认词组，可能改错词库。",
                                },
                                "remark": {
                                    "type": "string",
                                    "description": (
                                        "审词备注。若本轮调用过 keytao_prepare_reviewed_add，必须完整传入该词的读音、"
                                        "来源和自动审核结论；批次会按任一需管理员项锁定整批人工审核。"
                                    ),
                                },
                            },
                            "required": ["word", "code"],
                        },
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "当工具首次返回 requiresConfirmation=true 后，用户确认继续时必须设置为 true。默认 false",
                    },
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_shift_phrase_code",
            "description": (
                "将一个词改到指定编码，并按每个被挤走词自己的 keytao_encode 候选编码链逐个顺延。"
                "会检查目标编码是否是目标词的有效编码、每个顺延目标是否可继续挪动或为空，"
                "仅在没有相关旧草稿条目时，通过严格事务一次性写入 Delete+Create；"
                "如已有相关草稿则安全拒绝，不会先删后写。返回 shiftPlan.shifted 说明顺延了哪些词。"
                "用户要求插入到已占用编码、抢占某码位、把某词改到某个已占用编码时优先使用此工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {"type": "string", "description": "要移动/插入的目标词"},
                    "target_code": {"type": "string", "description": "目标编码，如 hyfio"},
                },
                "required": ["word", "target_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_batch_remove_draft_items",
            "description": (
                "批量从草稿中删除词条，通过条目 ID 列表指定要删除的内容。"
                "只能删除属于当前用户且处于草稿状态的条目。"
                "禁止在普通改码请求中批量删除大量草稿条目；只有用户明确要求删除/清空/撤销时才可批量删除。"
                "操作完成后返回成功数、失败信息及当前草稿快照。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "description": "要删除的草稿条目 ID 列表（整数）",
                        "items": {"type": "integer"},
                    }
                },
                "required": ["ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_recall_batch",
            "description": (
                "撤回最近一次提审，将批次从\"审核中\"状态恢复为草稿。"
                "⚠️ 仅当用户明确说\"撤回\"、\"撤销提交\"、\"取消提审\"时调用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_get_batch_preview",
            "description": (
                "获取当前草稿批次的 diff 预览。"
                "返回 summary（新增/修改/删除数量）和 diff_text（文字版 unified diff，含上下文行）。"
                "用户查看草稿时，优先调用此工具（而非 keytao_list_draft_items），以便展示完整 diff 效果。"
                "若需要条目 ID 进行删除操作，再补充调用 keytao_list_draft_items。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {
                        "type": "string",
                        "description": "可选：要查看的批次编号，必须是本轮工具结果里出现过的 batchId；不传表示当前草稿"
                    }
                },
                "required": [],
            },
        },
    },
]


# Tool registry for dynamic calling
TOOL_FUNCTIONS = {
    "keytao_create_phrase": keytao_create_phrase,
    "keytao_submit_batch": keytao_submit_batch,
    "keytao_list_draft_items": keytao_list_draft_items,
    "keytao_remove_draft_item": keytao_remove_draft_item,
    "keytao_update_draft_item_weight": keytao_update_draft_item_weight,
    "keytao_batch_add_to_draft": keytao_batch_add_to_draft,
    "keytao_batch_remove_draft_items": keytao_batch_remove_draft_items,
    "keytao_shift_phrase_code": keytao_shift_phrase_code,
    "keytao_recall_batch": keytao_recall_batch,
    "keytao_get_batch_preview": keytao_get_batch_preview,
}
