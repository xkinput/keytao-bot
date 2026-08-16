"""Tool execution adapter for the agent harness."""
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from nonebot.log import logger

from keytao_bot.utils.observability import observe_tool_call
from keytao_bot.utils.pending_confirmation import (
    advertised_batch_assent_verb,
    pending_confirmation_copy,
)

try:  # pragma: no cover - depends on the installed runtime
    import httpx as _httpx
except ImportError:  # pragma: no cover
    _httpx = None

try:  # pragma: no cover - depends on the installed runtime
    import jsonschema as _jsonschema
except ImportError:  # pragma: no cover
    _jsonschema = None


MAX_BATCH_ITEMS = 200
MODEL_TOOL_RESULT_RAW_DECISION_THRESHOLD_CHARS = 4_000
# Match the authorization layer's largest safely staged mutation list so the
# model can always inspect one complete staged set before acting on it.
MODEL_TOOL_RESULT_MAX_BATCH_ITEMS = 50
MODEL_TOOL_RESULT_MAX_REMARK_CHARS = 80
MODEL_TOOL_RESULT_MAX_DIFF_CHARS = 6_000
_BATCH_LIST_ARGUMENTS = {
    "keytao_batch_add_to_draft": "items",
    "keytao_audit_draft_items": "items",
    "keytao_batch_remove_draft_items": "ids",
}
_JSON_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}
_MAX_REPORTED_ERRORS = 5
_FALLBACK_MAX_DEPTH = 3
_missing_jsonschema_warned = False


@dataclass(frozen=True)
class ModelToolResultProjection:
    """One model-only result projection and its reviewable size budget."""

    max_chars: int
    projector: Callable[..., Dict[str, Any]]


def _present_fields(payload: Mapping[str, Any], names: Tuple[str, ...]) -> Dict[str, Any]:
    return {name: payload[name] for name in names if name in payload}


def _short_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _compact_phrase_row(value: Any, *, code: str = "") -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    row = _present_fields(value, (
        "word", "code", "type", "type_label", "weight", "label",
        "position", "position_label",
    ))
    if code and not row.get("code"):
        row["code"] = code
    return row


def _compact_phrase_rows(values: Any, *, code: str = "", limit: int = 40) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    rows: List[Dict[str, Any]] = []
    for value in values[:limit]:
        row = _compact_phrase_row(value, code=code)
        if row is not None:
            rows.append(row)
    return rows


def _compact_candidate_statuses(values: Any, *, limit: int = 40) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    statuses: List[Dict[str, Any]] = []
    for value in values[:limit]:
        if not isinstance(value, Mapping):
            continue
        status = _present_fields(value, ("code", "occupied", "label"))
        words = [
            str(word).strip()
            for word in value.get("words", [])
            if str(word).strip()
        ] if isinstance(value.get("words"), list) else []
        if not words:
            words = [
                str(phrase.get("word") or "").strip()
                for phrase in value.get("phrases", [])
                if isinstance(phrase, Mapping)
                and str(phrase.get("word") or "").strip()
            ] if isinstance(value.get("phrases"), list) else []
        if words:
            status["words"] = words[:8]
        statuses.append(status)
    return statuses


def _compact_sources(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    names: List[str] = []
    for value in values:
        name = (
            str(value.get("source") or value.get("label") or "").strip()
            if isinstance(value, Mapping)
            else str(value or "").strip()
        )
        if name and name not in names:
            names.append(name)
    return names[:6]


def _compact_review_audit(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    audit = _present_fields(value, (
        "success",
        "verdict",
        "autoApprove",
        "needsManualReview",
        "manualReviewReason",
        "reviewDisposition",
        "reviewVerdictSite",
        "auditSealed",
        "llmFallback",
        "previewOnly",
    ))
    for name, limit in (("summary", 240), ("warning", 180)):
        if name in value:
            audit[name] = _short_text(value.get(name), limit)
    for name in ("issues", "structuredManualReviewIssues"):
        if isinstance(value.get(name), list):
            audit[name] = [_short_text(item, 180) for item in value[name][:6]]
    for name in ("commonKnownItems", "semanticContextAutoPassItems"):
        if isinstance(value.get(name), list):
            audit[name] = [
                _present_fields(item, ("word", "code", "basisLine"))
                for item in value[name][:6]
                if isinstance(item, Mapping)
            ]
    return audit


def _compact_warning(value: Any) -> Any:
    if isinstance(value, str):
        return _short_text(value, 200)
    if not isinstance(value, Mapping):
        return _short_text(value, 200)
    warning = _present_fields(value, (
        "type", "code", "word", "occupantWord", "occupantCode", "verdict",
    ))
    for name in ("message", "reason", "summary", "impact", "basisLine"):
        if name in value:
            warning[name] = _short_text(value.get(name), 200)
    return warning


def _project_reviewed_add(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = _present_fields(payload, (
        "success",
        "word",
        "recommendedCode",
        "autoReviewable",
        "autoReviewReason",
        "lookupFailed",
        "lookupFailureReason",
        "pronunciationEvidenceStatus",
        "pronunciationEvidenceComplete",
        "requiresManualPronunciationReview",
        "standardPronunciationStatus",
        "pronunciationUnresolved",
        "needsManualReview",
        "manualReviewReason",
        "reviewDisposition",
        "reviewVerdictSite",
        "message",
    ))
    existing = _compact_phrase_rows(payload.get("existing"), limit=20)
    if existing:
        result["existing"] = existing

    pronunciations: List[Dict[str, Any]] = []
    raw_pronunciations = payload.get("pronunciations")
    if isinstance(raw_pronunciations, list):
        for value in raw_pronunciations[:8]:
            if not isinstance(value, Mapping):
                continue
            pronunciation = _present_fields(value, (
                "pinyin",
                "normalized",
                "codes",
                "recommendedCode",
                "fallback",
                "semanticPronunciation",
                "semanticPronunciationAccepted",
                "requiresManualReview",
                "readingEvidenceKind",
            ))
            source_names = _compact_sources(value.get("sources"))
            if source_names:
                pronunciation["sourceNames"] = source_names
            if value.get("sourceSummary"):
                pronunciation["sourceSummary"] = _short_text(
                    value.get("sourceSummary"),
                    180,
                )
            context = value.get("contextPronunciation")
            if isinstance(context, Mapping):
                pronunciation["contextPronunciation"] = {
                    **_present_fields(context, (
                        "entityType", "label", "confidence", "method", "commonTransparent",
                    )),
                    **({"description": _short_text(context.get("description"), 180)}
                       if context.get("description") else {}),
                }
            readings = value.get("characterReadings")
            if isinstance(readings, list):
                pronunciation["characterReadings"] = [
                    _present_fields(reading, ("char", "chosenPinyin", "lookupStatus"))
                    for reading in readings[:16]
                    if isinstance(reading, Mapping)
                ]
            statuses = _compact_candidate_statuses(value.get("candidateStatuses"))
            if statuses or "candidateStatuses" in value:
                pronunciation["candidateStatuses"] = statuses
            pronunciations.append(pronunciation)
    result["pronunciations"] = pronunciations

    audit = _compact_review_audit(payload.get("preSubmitAudit"))
    if audit is not None:
        result["preSubmitAudit"] = audit
    assessments = payload.get("candidateOrderingAssessments")
    if isinstance(assessments, list):
        result["candidateOrderingAssessments"] = [
            _compact_warning(value) for value in assessments[:8]
        ]
    for name in ("warning", "warnings", "auditLines", "warningLines"):
        value = payload.get(name)
        if isinstance(value, list):
            result[name] = [_compact_warning(item) for item in value[:8]]
        elif value:
            result[name] = _compact_warning(value)
    return result


def _compact_code_variants(values: Any, *, limit: int = 16) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [
        _present_fields(value, (
            "char", "charIndex", "pinyin", "pinyinLabel", "phoneticCode",
            "isDefault", "recommendedCode", "codes",
        ))
        for value in values[:limit]
        if isinstance(value, Mapping)
    ]


def _project_encode(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = _present_fields(payload, (
        "success",
        "input",
        "word",
        "type",
        "baseCode",
        "recommendedCode",
        "candidateCodes",
        "codes",
        "altCodes",
        "requestedCandidateCodes",
        "codeSource",
        "pronunciationSource",
        "standardPronunciationStatus",
        "phrasePinyins",
        "contextPhrasePinyins",
        "semanticPronunciationNeeded",
        "semanticPronunciationAccepted",
        "occupancyChecked",
        "firstAvailableCode",
        "firstRequestedAvailableCode",
        "occupancyError",
        "suggestion",
        "suggestionIndex",
        "isBaseConflict",
        "wordExists",
        "message",
    ))
    chars = payload.get("chars")
    if isinstance(chars, list):
        result["chars"] = [
            _present_fields(value, (
                "char", "pinyin", "pinyins", "phoneticCode", "c1", "c2", "shapeCode",
            ))
            for value in chars[:64]
            if isinstance(value, Mapping)
        ]
        if len(chars) > 64:
            result["charsTruncationNotice"] = (
                f"「…另有 {len(chars) - 64} 字」模型可见拆分已截断"
            )
    statuses = _compact_candidate_statuses(payload.get("candidateStatuses"))
    if statuses or "candidateStatuses" in payload:
        result["candidateStatuses"] = statuses
    for name in ("alternatePronunciationCodes", "alternatePhrasePronunciationCodes"):
        values = _compact_code_variants(payload.get(name))
        if values or name in payload:
            result[name] = values
    if isinstance(payload.get("flyKeyVariants"), list):
        result["flyKeyVariants"] = payload["flyKeyVariants"][:16]
    if isinstance(payload.get("requestedCodeAnalysis"), Mapping):
        result["requestedCodeAnalysis"] = dict(payload["requestedCodeAnalysis"])
    groups = payload.get("candidateDisplayGroups")
    if isinstance(groups, list):
        result["candidateDisplayGroups"] = [
            {
                **_present_fields(group, (
                    "pinyin", "pinyinLabel", "phoneticCode", "isDefault", "recommendedCode",
                )),
                "items": [
                    _present_fields(item, (
                        "code", "displayLabel", "state", "recommended", "occupied", "words",
                    ))
                    for item in group.get("items", [])[:16]
                    if isinstance(item, Mapping)
                ],
            }
            for group in groups[:8]
            if isinstance(group, Mapping)
        ]
    return result


def _compact_draft_item(value: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    item = _present_fields(value, (
        "id", "action", "word", "oldWord", "code", "type", "type_label",
        "display_label", "weight", "needsManualReview",
    ))
    if "remark" in value:
        item["remark"] = _short_text(value.get("remark"), MODEL_TOOL_RESULT_MAX_REMARK_CHARS)
    if value.get("warning"):
        item["warning"] = _short_text(value.get("warning"), 80)
    conflict = value.get("conflictInfo")
    if isinstance(conflict, Mapping) and conflict.get("impact"):
        item["conflictInfo"] = {
            "impact": _short_text(conflict.get("impact"), 80),
        }
    return item


def _project_draft_listing(
    payload: Mapping[str, Any],
    arguments: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    result = _present_fields(payload, (
        "success", "batchId", "batchIdProvisional", "batchUrl",
        "batchUrlStatus", "count", "summary", "message",
    ))
    values = payload.get("items")
    if isinstance(values, list):
        raw_offset = arguments.get("offset", 0) if arguments else 0
        raw_limit = (
            arguments.get("limit", MODEL_TOOL_RESULT_MAX_BATCH_ITEMS)
            if arguments else MODEL_TOOL_RESULT_MAX_BATCH_ITEMS
        )
        offset = (
            raw_offset
            if isinstance(raw_offset, int)
            and not isinstance(raw_offset, bool)
            and raw_offset >= 0
            else 0
        )
        limit = (
            raw_limit
            if isinstance(raw_limit, int)
            and not isinstance(raw_limit, bool)
            and 1 <= raw_limit <= MODEL_TOOL_RESULT_MAX_BATCH_ITEMS
            else MODEL_TOOL_RESULT_MAX_BATCH_ITEMS
        )
        window = values[offset:offset + limit]
        items = [
            item
            for value in window
            if (item := _compact_draft_item(value)) is not None
        ]
        result["items"] = items
        total = payload.get("count")
        if not isinstance(total, int) or isinstance(total, bool) or total < len(values):
            total = len(values)
        start = offset + 1 if items else 0
        end = offset + len(items) if items else 0
        result["itemsWindowNotice"] = (
            f"共 {total} 条，当前显示 {start}-{end} 条"
        )
        if offset > 0 or end < total:
            result["itemsTruncated"] = True
            pagination = "请设置 offset/limit 分页查看其他条目"
            if end < total:
                pagination = f"下一页使用 offset={end}, limit={limit}"
            result["itemsTruncationNotice"] = (
                f"共 {total} 条，当前显示 {start}-{end} 条；"
                "模型可见列表已分页或截断，不得据此判断完整条目集合；"
                f"{pagination}"
            )
    return result


def _project_batch_preview(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = _project_draft_listing(payload)
    diff_text = payload.get("diff_text")
    if isinstance(diff_text, str):
        if len(diff_text) <= MODEL_TOOL_RESULT_MAX_DIFF_CHARS:
            result["diff_text"] = diff_text
        else:
            result["diff_text"] = (
                diff_text[:MODEL_TOOL_RESULT_MAX_DIFF_CHARS].rstrip()
                + "\n…（模型可见 diff 已截断，完整预览仍由程序保留）"
            )
            result["diffTruncated"] = True
            result["diffTruncationNotice"] = (
                "模型可见 diff 不完整，不得据此判断完整条目集合"
            )
    return result


def _compact_duplicate_info(value: Any, *, code: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    info = _present_fields(value, ("position", "position_label"))
    rows = _compact_phrase_rows(value.get("all_words"), code=code, limit=40)
    if rows:
        info["all_words"] = rows
    return info or None


def _project_lookup_result(payload: Mapping[str, Any]) -> Dict[str, Any]:
    def project_one(value: Mapping[str, Any]) -> Dict[str, Any]:
        entry = _present_fields(value, ("success", "word", "code", "count"))
        phrases: List[Dict[str, Any]] = []
        raw_phrases = value.get("phrases")
        if isinstance(raw_phrases, list):
            for raw_phrase in raw_phrases[:100]:
                phrase = _compact_phrase_row(raw_phrase)
                if phrase is None:
                    continue
                duplicate = _compact_duplicate_info(
                    raw_phrase.get("duplicate_info") if isinstance(raw_phrase, Mapping) else None,
                    code=str(phrase.get("code") or entry.get("code") or ""),
                )
                if duplicate is not None:
                    phrase["duplicate_info"] = duplicate
                phrases.append(phrase)
            entry["phrases"] = phrases
        return entry

    result = project_one(payload)
    raw_results = payload.get("results")
    if isinstance(raw_results, list):
        result["results"] = [
            project_one(value)
            for value in raw_results[:100]
            if isinstance(value, Mapping)
        ]
    return result


MODEL_TOOL_RESULT_PROJECTIONS = {
    "keytao_prepare_reviewed_add": ModelToolResultProjection(3_500, _project_reviewed_add),
    "keytao_encode": ModelToolResultProjection(6_000, _project_encode),
    "keytao_list_draft_items": ModelToolResultProjection(16_000, _project_draft_listing),
    "keytao_get_batch_preview": ModelToolResultProjection(8_000, _project_batch_preview),
    "keytao_lookup_by_word": ModelToolResultProjection(8_000, _project_lookup_result),
    "keytao_lookup_by_words_batch": ModelToolResultProjection(16_000, _project_lookup_result),
    "keytao_lookup_by_code": ModelToolResultProjection(8_000, _project_lookup_result),
    "keytao_lookup_by_codes_batch": ModelToolResultProjection(16_000, _project_lookup_result),
}

# These tools intentionally remain full-fidelity for the model. Mutations carry
# server tickets/warnings, while docs and web results are explicitly deferred
# content features; a future large-result guard must treat them as reviewed.
MODEL_TOOL_RESULT_LARGE_RAW_WHITELIST = frozenset({
    "keytao_fetch_docs",
    "keytao_create_phrase",
    "keytao_submit_batch",
    "keytao_remove_draft_item",
    "keytao_update_draft_item_weight",
    "keytao_batch_add_to_draft",
    "keytao_shift_phrase_code",
    "keytao_batch_remove_draft_items",
    "keytao_recall_batch",
    "keytao_audit_draft_items",
    "web_search",
    "web_fetch",
})
MODEL_TOOL_RESULT_SMALL_RAW_WHITELIST = frozenset({
    "get_current_datetime",
})


def project_tool_result_for_model(
    tool_name: str,
    result_json: str,
    arguments: Optional[Mapping[str, Any]] = None,
) -> str:
    """Project only the copy serialized into a model ``tool`` message."""
    projection = MODEL_TOOL_RESULT_PROJECTIONS.get(tool_name)
    if projection is None:
        return result_json
    try:
        payload = json.loads(result_json)
    except (TypeError, ValueError):
        return result_json
    if not isinstance(payload, Mapping):
        return result_json
    if (
        payload.get("success") is False
        or payload.get("policyBlocked") is True
        or "error" in payload
    ):
        return result_json
    projected = (
        projection.projector(payload, arguments)
        if tool_name == "keytao_list_draft_items"
        else projection.projector(payload)
    )
    return json.dumps(projected, ensure_ascii=False, separators=(",", ":"))


def large_model_tool_result_has_policy(tool_name: str, result_json: str) -> bool:
    """Tell guard tests whether a large raw result received an explicit decision."""
    return (
        len(result_json) <= MODEL_TOOL_RESULT_RAW_DECISION_THRESHOLD_CHARS
        or tool_name in MODEL_TOOL_RESULT_PROJECTIONS
        or tool_name in MODEL_TOOL_RESULT_LARGE_RAW_WHITELIST
        or tool_name in MODEL_TOOL_RESULT_SMALL_RAW_WHITELIST
    )

_STAGED_ARGUMENT_LABELS = {
    "action": "动作",
    "batch_id": "批次",
    "code": "编码",
    "expected_content_version": "内容版本",
    "ids": "草稿条目",
    "items": "词条",
    "needs_manual_review": "管理员复核标记",
    "old_word": "原词",
    "pr_id": "草稿条目",
    "remark": "备注",
    "target_code": "目标编码",
    "type": "类型",
    "weight": "权重",
    "word": "词语",
}

_STAGED_ACTION_LABELS = {
    "Change": "修改",
    "Create": "新增",
    "Delete": "删除",
}


def _staged_argument_label(key: Any) -> str:
    key_text = str(key)
    return _STAGED_ARGUMENT_LABELS.get(key_text, f"字段 {key_text}")


def _staged_argument_value(value: Any) -> str:
    if isinstance(value, dict):
        return "；".join(
            f"{_staged_argument_label(key)}：{_staged_argument_value(item)}"
            for key, item in value.items()
        ) or "空"
    if isinstance(value, (list, tuple)):
        return "、".join(_staged_argument_value(item) for item in value) or "空"
    if isinstance(value, bool):
        return "是" if value else "否"
    if value is None:
        return "未设置"
    text = str(value)
    return _STAGED_ACTION_LABELS.get(text, text)


def _describe_staged_mutation(tool_name: str, arguments: Dict) -> str:
    lines = [f"拟执行 {tool_name}，已锁定以下内容："]
    for key, value in arguments.items():
        label = _staged_argument_label(key)
        if isinstance(value, (list, tuple)):
            lines.append(f"• {label}（{len(value)} 项）：")
            lines.extend(
                f"  {index}. {_staged_argument_value(item)}"
                for index, item in enumerate(value, start=1)
            )
        else:
            lines.append(f"• {label}：{_staged_argument_value(value)}")
    return "\n".join(lines)


def _tool_exception_payload(error: Exception) -> Dict[str, Any]:
    """Preserve whether a failed call is safe to retry as a transport attempt."""
    transport_error = isinstance(error, (TimeoutError, ConnectionError))
    if _httpx is not None:
        transport_error = transport_error or isinstance(error, _httpx.TransportError)
    return {
        "error": str(error),
        "errorType": type(error).__name__,
        "transportError": transport_error,
    }


def _warn_missing_jsonschema_once() -> None:
    global _missing_jsonschema_warned
    if _missing_jsonschema_warned:
        return
    _missing_jsonschema_warned = True
    logger.warning(
        "jsonschema is not installed; tool argument validation is using the "
        "bounded required/type/enum fallback"
    )


def _extract_parameters_schema(schema: Optional[Dict]) -> Optional[Dict]:
    if not isinstance(schema, dict):
        return None
    function_spec = schema.get("function")
    parameters = (
        function_spec.get("parameters")
        if isinstance(function_spec, dict)
        else schema.get("parameters", schema)
    )
    if not isinstance(parameters, dict) or not parameters:
        return None
    if parameters.get("additionalProperties") is False:
        parameters = {
            key: value
            for key, value in parameters.items()
            if key != "additionalProperties"
        }
    return parameters


def _matches_json_type(value: Any, json_type: str) -> bool:
    expected = _JSON_TYPE_MAP.get(json_type)
    if expected is None:
        return True
    if json_type in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _declared_types(spec: Dict) -> List[str]:
    declared = spec.get("type")
    if isinstance(declared, str):
        return [declared]
    if isinstance(declared, list):
        return [item for item in declared if isinstance(item, str)]
    return []


def _validate_object_fallback(
    value: Dict,
    parameters: Dict,
    path: str,
    depth: int,
) -> List[str]:
    errors: List[str] = []
    label = f"'{path}' 中的" if path else ""
    required = parameters.get("required")
    if isinstance(required, list):
        for key in required:
            if isinstance(key, str) and key not in value:
                errors.append(f"缺少必填参数 {label}'{key}'")

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return errors
    for key, spec in properties.items():
        if key not in value or not isinstance(spec, dict):
            continue
        child = value[key]
        child_path = f"{path}.{key}" if path else key
        types = _declared_types(spec)
        if types and not any(_matches_json_type(child, item) for item in types):
            errors.append(
                f"参数 '{child_path}' 类型应为 {'/'.join(types)}，实际为 {type(child).__name__}"
            )
            continue
        enum_values = spec.get("enum")
        if isinstance(enum_values, list) and enum_values and child not in enum_values:
            errors.append(
                f"参数 '{child_path}' 取值应为 {enum_values} 之一，实际为 {child!r}"
            )
            continue
        if depth >= _FALLBACK_MAX_DEPTH:
            continue
        if isinstance(child, dict):
            errors.extend(
                _validate_object_fallback(child, spec, child_path, depth + 1)
            )
        elif isinstance(child, list):
            item_spec = spec.get("items")
            if not isinstance(item_spec, dict):
                continue
            for index, element in enumerate(child):
                element_path = f"{child_path}[{index}]"
                item_types = _declared_types(item_spec)
                if item_types and not any(
                    _matches_json_type(element, item) for item in item_types
                ):
                    errors.append(
                        f"参数 '{element_path}' 类型应为 {'/'.join(item_types)}，"
                        f"实际为 {type(element).__name__}"
                    )
                    continue
                if isinstance(element, dict):
                    errors.extend(
                        _validate_object_fallback(
                            element, item_spec, element_path, depth + 1
                        )
                    )
                if len(errors) >= _MAX_REPORTED_ERRORS:
                    return errors
    return errors


def _validate_arguments_fallback(arguments: Dict, parameters: Dict) -> List[str]:
    return _validate_object_fallback(arguments, parameters, "", 0)


def _validate_root_type(tool_name: str, arguments: Any) -> Optional[Dict]:
    if isinstance(arguments, dict):
        return None
    return {
        "success": False,
        "invalidArguments": True,
        "message": (
            "参数校验失败：参数必须是 JSON 对象，"
            f"实际为 {type(arguments).__name__}"
        ),
        "tool": tool_name,
        "errors": ["arguments must be a JSON object"],
    }


def _validate_arguments(
    tool_name: str,
    arguments: Any,
    schema: Optional[Dict],
) -> Optional[Dict]:
    root_error = _validate_root_type(tool_name, arguments)
    if root_error:
        return root_error
    parameters = _extract_parameters_schema(schema)
    if parameters is None:
        return None
    errors: List[str] = []
    if _jsonschema is not None:
        try:
            validator_class = _jsonschema.validators.validator_for(parameters)
            validator = validator_class(parameters)
            for error in validator.iter_errors(arguments):
                location = "/".join(
                    str(part) for part in error.absolute_path
                ) or "(root)"
                errors.append(f"{location}: {error.message}")
                if len(errors) >= _MAX_REPORTED_ERRORS:
                    break
        except Exception as error:
            logger.debug(f"Schema validation skipped for {tool_name}: {error}")
            return None
    else:
        _warn_missing_jsonschema_once()
        errors = _validate_arguments_fallback(arguments, parameters)[
            :_MAX_REPORTED_ERRORS
        ]
    if not errors:
        return None
    return {
        "success": False,
        "invalidArguments": True,
        "message": f"参数校验失败：{'; '.join(errors)}",
        "tool": tool_name,
        "errors": errors,
    }


def _validate_batch_size(tool_name: str, arguments: Dict) -> Optional[Dict]:
    argument_name = _BATCH_LIST_ARGUMENTS.get(tool_name)
    if not argument_name or not isinstance(arguments, dict):
        return None
    value = arguments.get(argument_name)
    if not isinstance(value, list) or len(value) <= MAX_BATCH_ITEMS:
        return None
    return {
        "success": False,
        "message": (
            f"单次批量条目过多（{len(value)} 条），上限 {MAX_BATCH_ITEMS} 条，"
            "请分批提交。"
        ),
        "policyBlocked": True,
        "blockReason": BLOCK_REASON_BATCH_TOO_LARGE,
    }


@dataclass(frozen=True)
class PendingCandidateCapability:
    """Server-bound candidate state available to one current actor turn."""

    state_matches: bool
    word: str
    candidates: Tuple[Tuple[str, bool], ...]
    occupied_words: Tuple[Tuple[str, Tuple[str, ...]], ...]
    entries: Tuple[Tuple[str, str, int], ...] = ()


@dataclass(frozen=True)
class ToolContext:
    platform: Optional[str] = None
    user_id: Optional[str] = None
    current_message: Optional[str] = None
    writes_allowed: bool = True
    # True only when this turn carries attachment/vision derived text.  It is
    # the one case where "the source cannot authorize writes" is the honest
    # explanation; a plain text turn is never blocked for that reason.
    attachment_context: bool = False
    trusted_codes_by_word: Optional[Dict[str, frozenset[str]]] = None
    trusted_entries_by_code: Optional[
        Dict[str, Tuple[Tuple[str, int], ...]]
    ] = None
    trusted_draft_words_by_id: Optional[Dict[str, str]] = None
    trusted_draft_items_by_id: Optional[Dict[str, Dict[str, Any]]] = None
    trusted_phrase_types_by_key: Optional[Dict[Tuple[str, str], frozenset[str]]] = None
    trusted_reviewed_items_by_key: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None
    pending_candidate: Optional[PendingCandidateCapability] = None
    # Batch ids the server itself surfaced during this run.  A model may only
    # anchor a read to one of these; anything else would let injected text point
    # the bot at a stranger's batch.
    trusted_batch_ids: Optional[frozenset[str]] = None
    mutation_confirmed: bool = False
    server_warning_confirmed: bool = False
    trusted_word_lookup_codes_by_word: Optional[
        Dict[str, frozenset[str]]
    ] = None
    trusted_candidate_slots_by_word: Optional[
        Dict[str, Tuple[Tuple[str, bool], ...]]
    ] = None


from .authorization_grammar import (
    _DELETE_INTENT_RE,
    _POSITIONAL_REORDER_QUOTED_ENTRY_PATTERN,
    _POSITIONAL_REORDER_CODE_PATTERN,
    _POSITIONAL_REORDER_PLAIN_ENTRY_PATTERN,
    _POSITIONAL_REORDER_RELATION_PATTERN,
    _POSITIONAL_CREATE_FRONT_RELATIONS,
    _POSITIONAL_CREATE_BACK_RELATIONS,
    _POSITIONAL_SAME_CODE_MARKER_PATTERN,
    _POSITIONAL_SAME_CODE_MARKER_RE,
    _POSITIONAL_COMMAND_VERB_RE,
    _POSITIONAL_REORDER_ORDINAL_PATTERN,
    _POSITIONAL_REORDER_DESTINATION_PATTERN,
    _POSITIONAL_REORDER_DESTINATION_EXPRESSION_PATTERN,
    _POSITIONAL_REORDER_COMMAND_DESTINATION_EXPRESSION_PATTERN,
    _POSITIONAL_REORDER_RELATIVE_EXPRESSION_PATTERN,
    _POSITIONAL_REORDER_INTENT_PATTERN,
    _POSITIONAL_REORDER_INTENT_RE,
    _NON_POSITIONAL_MUTATION_INTENT_PATTERN,
    _NON_POSITIONAL_MUTATION_INTENT_RE,
    _MUTATION_INTENT_RE,
    _POSITIONAL_REORDER_SUBJECT_PATTERN,
    _POSITIONAL_REORDER_BARE_SUBJECT_PATTERN,
    _POSITIONAL_REORDER_COMMAND_RE,
    _POSITIONAL_REORDER_RAW_COMMAND_RE,
    _POSITIONAL_REORDER_DESTINATION_CAPTURE_RE,
    _POSITIONAL_REORDER_QUOTED_TARGET_RE,
    _POSITIONAL_REORDER_PLAIN_LOCATIVE_SUFFIX_RE,
    _POSITIONAL_REORDER_RELATIVE_FRAGMENT_RE,
    _POSITIONAL_REORDER_NARRATIVE_TAIL_RE,
    _POSITIONAL_REORDER_CHOICE_QUESTION_RE,
    _POSITIONAL_REORDER_TEMPORAL_DESTINATION_RE,
    _POSITIONAL_REORDER_TRAILING_MODIFIER_RE,
    _POSITIONAL_SUBORDINATE_CONTEXT_RE,
    _POSITIONAL_REPORTED_CONTEXT_RE,
    _POSITIONAL_BARE_DATA_CONTEXT_RE,
    _POSITIONAL_CONTEXT_NEGATION_RE,
    _NEGATED_POSITIONAL_REORDER_RE,
    _POSITIONAL_REORDER_EXPLANATION_RE,
    _POSITIONAL_REORDER_LOCATIVE_DESTINATION_RE,
    _NEGATED_MUTATION_RE,
    _NEGATED_NON_POSITIONAL_MUTATION_RE,
    _STANDALONE_NEGATION_CLAUSE_RE,
    _EXPLANATION_ONLY_RE,
    _TEXT_TRANSFORM_RE,
    _QUESTION_RE,
    _ABORT_RE,
    _EXPLICIT_REQUEST_PREFIX_RE,
    _POLITE_EXECUTION_PREFIX_RE,
    _META_DISCUSSION_RE,
    _DATA_CONTEXT_RE,
    _RECORD_FRAME_RE,
    _QUOTED_DATA_RE,
    _RECORD_FRAME_BRACKETED_DATA_RE,
    _UNTRUSTED_QUOTE_PREFIX_RE,
    _UNTRUSTED_DATA_TAIL_RE,
    _INLINE_CODE_RE,
    _COMMAND_CLAUSE_SPLIT_RE,
    _LEADING_MENTION_RE,
    _WHOLE_MESSAGE_LEADING_ADDRESS_RE,
    _WHOLE_MESSAGE_CLOSING_FILLER_RE,
    _WHOLE_MESSAGE_QUOTE_PATTERNS,
    _COMMAND_LEAD_IN_PREFIXES,
    _COMMAND_PREFIX_PATTERN,
    _COMMAND_PREFIX_RE,
    _MULTI_ADD_VERB_RE,
    _MULTI_ADD_ADDRESS_CLAUSES,
    _MAX_AUTHORIZED_MULTI_ADD_ITEMS,
    _AuthorizedAddClause,
    _MultiAddAuthorization,
    _parse_complete_add_clause,
    _multi_add_authorization_contract,
    authorized_multi_add_items,
    _multi_add_items_match_authorized_set,
    _PositionalDestination,
    _PositionalCreateBinding,
    _unquote_positional_entry,
    _parse_positional_destination,
    _positional_create_operands,
    _positional_destination_from_command,
    _raw_positional_destination_from_command,
    _has_raw_positional_relative_tail,
    _has_complete_positional_reorder_command,
    _positional_destination_is_ambiguous_non_command,
    _has_positional_choice_question,
    _ENTRY_MUTATION_FOR_FRAME_OPERAND_RE,
    _RECORD_FRAME_OPERAND_AFTER_RE,
    _RECORD_FRAME_DRAFT_DELETE_PREFIX_RE,
    _RECORD_FRAME_POSTPOSED_OPERAND_RE,
    _RECORD_FRAME_EMBEDDED_MUTATION_SUFFIX_RE,
    _RECORD_FRAME_SEPARATOR_CHARS,
    _COMPLETE_MUTATION_NOISE_RE,
    _RECORD_ANALYSIS_TRANSLATION,
    _NEGATIVE_MODAL_RE,
    _CLAUSE_BOUNDARY_RE,
    _ACTION_TOKENS,
    _WORD_LEFT_PREFIXES,
    _WORD_RIGHT_SUFFIXES,
    BLOCK_REASON_SOURCE_UNTRUSTED,
    BLOCK_REASON_VERB_NOT_MATCHED,
    BLOCK_REASON_BINDING_INCOMPLETE,
    BLOCK_REASON_TICKET_REQUIRED,
    BLOCK_REASON_BULK_DELETE_NOT_REQUESTED,
    BLOCK_REASON_MANUAL_SHIFT_FORBIDDEN,
    BLOCK_REASON_ORDERING_NOT_EXPRESSIBLE,
    BLOCK_REASON_BATCH_TOO_LARGE,
    BLOCK_REASON_UNTRUSTED_BATCH,
    SUGGESTION_MENTION_PREFIX,
    _MAX_SUGGESTED_BATCH_ITEMS,
    _MAX_SUGGESTED_DELETE_IDS,
    MUTATING_TOOL_NAMES,
    _MAX_STAGED_MUTATION_PREVIEW_CHARS,
    _MAX_STAGED_MUTATION_LIST_ITEMS,
    _PROTECTED_WORD_RE,
    _TYPE_HINTS,
    _PHRASE_TYPES,
    _PHRASE_TYPE_BASE_WEIGHTS,
    _QUOTED_COMMAND_OPERAND_RE,
    _MAX_QUOTED_ENTRY_CHARS,
    _quoted_span_is_command,
    _whole_message_quote_content,
    _whole_message_unquoted_source,
    trusted_mutation_source,
    _has_standalone_negation_before_mutation,
    _mutation_authorization_view,
    _has_mutation_instruction_shape,
    _message_authorizes_mutation_core,
    _mask_quoted_record_frames,
    _normalize_positional_same_code_markers,
    _positional_same_code_requested,
    _record_frame_is_mutation_operand,
    _has_complete_mutation_instruction,
    _record_frame_wraps_complete_mutation,
    message_authorizes_mutation,
    _POSITIONAL_CHANGE_RE,
    _CHANGE_VERB_RE,
    _WEIGHT_ADJUST_VERB_RE,
    _CREATE_VERB_RE,
    _DELETE_VERB_RE,
    _SUBMIT_VERB_RE,
    _RECALL_VERB_RE,
    _TOOL_INTENT_PATTERNS,
    _tool_intent_pattern,
    message_requests_change,
    _suggestion_operands,
    _operands_are_present,
    message_mentions_change_request,
    _type_hint_label,
    _suggested_item_command,
    _suggested_command_text,
    self_checked_suggested_command,
    policy_block,
    text_follow_up,
    _AUTO_CONFIRM_CREATE_WARNING_SITES,
    _AUTO_CONFIRM_CREATE_WARNING_TYPES,
    server_warning_confirmation_binding,
    create_warning_confirmation_binding,
    batch_warning_confirmation_binding,
    front_insert_batch_warning_confirmation_binding,
    _strip_execution_result_suffix,
    _explicit_submit_command_matches,
    _explicit_recall_command_matches,
    _extract_explicit_phrase_type,
    _extract_phrase_type_for_target,
    _is_word_protected,
    _quoted_target_spans,
    _has_protection_outside_target,
    _action_match_is_negated,
    _is_han,
    _exact_target_spans,
    _contains_exact_target,
    _action_is_bound_to_target,
    _clause_bounds,
    _span_distance,
    _NON_CODE_ASCII_TOKENS,
    _explicit_code_spans,
    _code_is_bound_to_target,
    _target_clause_has_explicit_code_token,
    _server_knows_positional_entry,
    _positional_create_order,
    _positional_message_explicitly_labels_code,
    _positional_phrase_type,
    _same_type_positional_entries,
    _pending_positional_create_binding,
    _pending_create_is_bound,
    _pending_batch_selected_items,
    _pending_batch_items_match_selection,
    _positional_destination_operand_is_exact,
    _destination_derived_positional_create_binding,
    _served_candidate_slots,
    _next_free_served_candidate,
    _positional_destination_is_bound,
    _change_transition_is_bound,
    _draft_item_reference_is_bound,
    _find_code_reassignments,
    _validate_current_message_binding,
)


@dataclass(frozen=True)
class ToolExecutionRoute:
    tool_name: str
    arguments: Dict[str, Any]
    positional_binding: Optional[_PositionalCreateBinding] = None
    response: Optional[Dict[str, Any]] = None


class ToolExecutor:
    """Executes registered skills and injects platform context when needed."""

    def __init__(
        self,
        get_tool_function: Callable[[str], Optional[Callable]],
        context_tools: frozenset[str],
        mutation_guard: Optional[
            Callable[[ToolContext, str, Dict], Optional[Dict]]
        ] = None,
        *,
        get_tool_schema: Optional[Callable[[str], Optional[Dict]]] = None,
    ):
        self._get_tool_function = get_tool_function
        self._context_tools = context_tools
        self._mutation_guard = mutation_guard
        self._get_tool_schema = get_tool_schema

    def _resolve_schema(self, tool_name: str) -> Optional[Dict]:
        if self._get_tool_schema is None:
            return None
        try:
            return self._get_tool_schema(tool_name)
        except Exception as error:
            logger.debug(f"Schema lookup failed for {tool_name}: {error}")
            return None

    def canonicalize_arguments(
        self,
        tool_name: str,
        arguments: Dict,
        context: ToolContext,
    ) -> Dict:
        """Freeze server-derived fields before execution or pending persistence."""
        return self._with_trusted_mutation_fields(tool_name, arguments, context)

    @staticmethod
    def uses_pending_positional_create(
        tool_name: str,
        arguments: Dict,
        context: ToolContext,
    ) -> bool:
        if tool_name != "keytao_create_phrase":
            return False
        binding = _pending_positional_create_binding(
            _mutation_authorization_view(context.current_message or ""),
            arguments,
            context,
        )
        return binding is not None

    def resolve_execution_route(
        self,
        tool_name: str,
        arguments: Dict,
        context: ToolContext,
    ) -> ToolExecutionRoute:
        """Resolve post-authorization positional-create execution semantics."""
        unchanged = ToolExecutionRoute(tool_name, dict(arguments))
        if tool_name != "keytao_create_phrase":
            return unchanged
        message = _mutation_authorization_view(context.current_message or "")
        binding = (
            _pending_positional_create_binding(message, arguments, context)
            or _destination_derived_positional_create_binding(
                message,
                arguments,
                context,
            )
        )
        if binding is None:
            return unchanged
        if _positional_same_code_requested(context.current_message or ""):
            if (
                binding.relation in _POSITIONAL_CREATE_FRONT_RELATIONS
                and binding.weight is not None
                and binding.bumped_entries
            ):
                create_item: Dict[str, Any] = {
                    "action": "Create",
                    "word": str(arguments.get("word") or "").strip(),
                    "code": binding.code,
                    "type": binding.phrase_type,
                    "weight": binding.weight,
                    "needsManualReview": bool(
                        arguments.get("needs_manual_review", True)
                    ),
                }
                remark = str(arguments.get("remark") or "").strip()
                if remark:
                    create_item["remark"] = remark
                items = [create_item]
                items.extend(
                    {
                        "action": "Change",
                        "old_word": entry_word,
                        "word": entry_word,
                        "code": binding.code,
                        "type": binding.phrase_type,
                        "weight": new_weight,
                    }
                    for entry_word, _old_weight, new_weight
                    in binding.bumped_entries
                )
                return ToolExecutionRoute(
                    "keytao_batch_add_to_draft",
                    {"items": items},
                    positional_binding=binding,
                )
            return ToolExecutionRoute(
                tool_name,
                dict(arguments),
                positional_binding=binding,
            )

        word = str(arguments.get("word") or "").strip()
        if binding.relation in _POSITIONAL_CREATE_FRONT_RELATIONS:
            shift_args: Dict[str, Any] = {
                "word": word,
                "target_code": binding.code,
            }
            if arguments.get("type"):
                shift_args["target_type"] = arguments["type"]
            if arguments.get("remark"):
                shift_args["target_remark"] = arguments["remark"]
            if "needs_manual_review" in arguments:
                shift_args["target_needs_manual_review"] = arguments[
                    "needs_manual_review"
                ]
            return ToolExecutionRoute(
                "keytao_shift_phrase_code",
                shift_args,
                positional_binding=binding,
            )

        if binding.relation in _POSITIONAL_CREATE_BACK_RELATIONS:
            next_code, slots = _next_free_served_candidate(
                context,
                word,
                binding.code,
            )
            if not next_code:
                candidate_codes = [code for code, _occupied in slots]
                reason = (
                    "following_candidate_unavailable"
                    if slots
                    else "candidate_chain_required"
                )
                return ToolExecutionRoute(
                    tool_name,
                    dict(arguments),
                    positional_binding=binding,
                    response=text_follow_up(
                        reason,
                        (
                            f"“{word}”在 {binding.code} 之后没有本轮已核验的空闲候选编码，"
                            "本次未写入。可以明确要求同码，或先重新准备该词的候选编码后再试。"
                        ),
                        word=word,
                        destinationWord=binding.destination_word,
                        currentCode=binding.code,
                        candidateCodes=candidate_codes,
                        nextAction={
                            "type": "prepare_candidates_or_request_same_code",
                            "sameCodeMarkers": ["同码", "同编码", "重码"],
                        },
                    ),
                )
            create_args = self._with_trusted_mutation_fields(
                "keytao_create_phrase",
                {
                    **arguments,
                    "code": next_code,
                    "action": "Create",
                },
                context,
            )
            create_args.pop("weight", None)
            return ToolExecutionRoute(
                "keytao_create_phrase",
                create_args,
                positional_binding=binding,
            )
        return unchanged

    async def _invoke_effective_tool(
        self,
        tool_name: str,
        arguments: Dict,
        context: ToolContext,
    ) -> Any:
        tool_func = self._get_tool_function(tool_name)
        if not tool_func:
            return {"error": f"Tool {tool_name} not found"}
        call_args = dict(arguments)
        if tool_name in self._context_tools:
            if not context.platform or not context.user_id:
                return {"error": "内部错误：无法获取用户平台信息"}
            call_args["platform"] = context.platform
            call_args["platform_id"] = context.user_id
        return await tool_func(**call_args)

    async def replay_shift_plan(
        self,
        authorization_tool_name: str,
        authorization_arguments: Dict,
        binding: Dict,
        context: ToolContext,
    ) -> Tuple[str, Dict]:
        """Replay one digest-bound shift after rechecking its authorization."""
        if authorization_tool_name == "keytao_shift_phrase_code":
            confirm_args = {**authorization_arguments, **binding}
            return await self.call(
                authorization_tool_name,
                confirm_args,
                context,
            ), confirm_args

        canonical = self.canonicalize_arguments(
            authorization_tool_name,
            authorization_arguments,
            context,
        )
        policy_error = self._validate_policy(
            authorization_tool_name,
            canonical,
            context,
        )
        if policy_error:
            return json.dumps(policy_error, ensure_ascii=False), canonical
        route = self.resolve_execution_route(
            authorization_tool_name,
            canonical,
            context,
        )
        if route.tool_name != "keytao_shift_phrase_code" or route.response:
            return json.dumps(
                policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    "安全拦截：顺延确认已不再对应原始位置指令。",
                    missing=["boundShiftPlan"],
                ),
                ensure_ascii=False,
            ), canonical
        confirm_args = {**route.arguments, **binding}
        try:
            result = await self._invoke_effective_tool(
                route.tool_name,
                confirm_args,
                context,
            )
            return json.dumps(result, ensure_ascii=False), confirm_args
        except Exception as error:
            logger.error(
                f"Tool {route.tool_name} replay error: "
                f"{type(error).__name__}: {error}"
            )
            return json.dumps(_tool_exception_payload(error), ensure_ascii=False), confirm_args

    async def replay_routed_warning(
        self,
        authorization_tool_name: str,
        authorization_arguments: Dict,
        expected_route: ToolExecutionRoute,
        binding: Dict,
        context: ToolContext,
    ) -> Tuple[str, Dict]:
        """Replay one routed warning ticket after rechecking its sealed route."""
        canonical = self.canonicalize_arguments(
            authorization_tool_name,
            authorization_arguments,
            context,
        )
        policy_error = self._validate_policy(
            authorization_tool_name,
            canonical,
            context,
        )
        if policy_error:
            return json.dumps(policy_error, ensure_ascii=False), canonical
        route = self.resolve_execution_route(
            authorization_tool_name,
            canonical,
            context,
        )
        if (
            route.response is not None
            or route.tool_name != expected_route.tool_name
            or route.arguments != expected_route.arguments
            or route.positional_binding != expected_route.positional_binding
        ):
            return json.dumps(
                policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    "安全拦截：确认票据已不再对应原始位置指令。",
                    missing=["boundFrontInsertPlan"],
                ),
                ensure_ascii=False,
            ), canonical
        confirm_args = {
            key: value
            for key, value in route.arguments.items()
            if key != "preview_only"
        }
        confirm_args.update(binding)
        try:
            result = await self._invoke_effective_tool(
                route.tool_name,
                confirm_args,
                context,
            )
            return json.dumps(result, ensure_ascii=False), confirm_args
        except Exception as error:
            logger.error(
                f"Tool {route.tool_name} routed warning replay error: "
                f"{type(error).__name__}: {error}"
            )
            return json.dumps(_tool_exception_payload(error), ensure_ascii=False), confirm_args

    @observe_tool_call
    async def call(self, tool_name: str, arguments: Dict, context: ToolContext) -> str:
        root_error = _validate_root_type(tool_name, arguments)
        if root_error:
            logger.warning(
                f"Tool {tool_name} rejected invalid arguments: {root_error['message']}"
            )
            return json.dumps(root_error, ensure_ascii=False)
        call_args = self.canonicalize_arguments(tool_name, arguments, context)
        policy_error = self._validate_policy(tool_name, call_args, context)
        if policy_error:
            logger.warning(f"Tool {tool_name} blocked by policy: {policy_error}")
            return json.dumps(policy_error, ensure_ascii=False)

        schema_error = _validate_arguments(
            tool_name,
            arguments,
            self._resolve_schema(tool_name),
        )
        if schema_error:
            logger.warning(
                f"Tool {tool_name} rejected invalid arguments: {schema_error['message']}"
            )
            return json.dumps(schema_error, ensure_ascii=False)

        route = self.resolve_execution_route(tool_name, call_args, context)
        if route.response is not None:
            return json.dumps(route.response, ensure_ascii=False)

        try:
            result = await self._invoke_effective_tool(
                route.tool_name,
                route.arguments,
                context,
            )
            if (
                tool_name == "keytao_create_phrase"
                and route.tool_name in {
                    "keytao_create_phrase",
                    "keytao_batch_add_to_draft",
                }
                and _positional_same_code_requested(
                    context.current_message or ""
                )
                and isinstance(result, dict)
                and (
                    result.get("success") is True
                    or result.get("requiresConfirmation") is True
                )
            ):
                message = _mutation_authorization_view(
                    context.current_message or ""
                )
                ordering = (
                    _pending_positional_create_binding(
                        message,
                        arguments,
                        context,
                    )
                    or _destination_derived_positional_create_binding(
                        message,
                        arguments,
                        context,
                    )
                )
                if ordering is not None and ordering.resulting_words:
                    side = (
                        "前"
                        if ordering.relation in _POSITIONAL_CREATE_FRONT_RELATIONS
                        else "后"
                    )
                    result.setdefault(
                        "orderingSummary",
                        f"{ordering.code}：{' → '.join(ordering.resulting_words)}"
                        f"（新词权重 {ordering.weight}，排在“{ordering.destination_word}”{side}）",
                    )
            result_json = json.dumps(result, ensure_ascii=False)
            logger.info(
                f"Tool {tool_name} executed as {route.tool_name}: "
                f"{result_json[:300]}"
            )
            return result_json
        except Exception as error:
            logger.error(
                f"Tool {tool_name} via {route.tool_name} error: "
                f"{type(error).__name__}: {error}"
            )
            return json.dumps(_tool_exception_payload(error), ensure_ascii=False)

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
                sanitized.pop("weight", None)
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
                positional_create = (
                    _pending_positional_create_binding(
                        message,
                        arguments,
                        context,
                    )
                    or _destination_derived_positional_create_binding(
                        message,
                        arguments,
                        context,
                    )
                )
                if (
                    positional_create is not None
                    and positional_create.weight is not None
                ):
                    sanitized["weight"] = positional_create.weight
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
        batch_size_error = _validate_batch_size(tool_name, arguments)
        if batch_size_error:
            return batch_size_error
        message = context.current_message or ""
        multi_add = (
            _multi_add_authorization_contract(message)
            if message and tool_name in MUTATING_TOOL_NAMES
            else None
        )
        if (
            multi_add is not None
            and multi_add.valid
            and len(multi_add.clauses) > _MAX_AUTHORIZED_MULTI_ADD_ITEMS
        ):
            refused = [
                f"{clause.word} {clause.code}".strip()
                for clause in multi_add.clauses
            ]
            logger.warning(
                "Refused multi-add above per-message limit: "
                f"count={len(refused)} items={refused}",
            )
            return text_follow_up(
                "multi_add_limit_exceeded",
                f"本次点名了 {len(refused)} 条加词，超过单次上限 "
                f"{_MAX_AUTHORIZED_MULTI_ADD_ITEMS} 条；本次未截断、未写入。"
                "请拆成更小批次后重试。",
                refusedItems=refused,
                limit=_MAX_AUTHORIZED_MULTI_ADD_ITEMS,
            )
        requested_batch = str(arguments.get("batch_id") or "").strip()
        if (
            requested_batch
            # Only model-originated calls carry the current message.  Internal
            # callers anchor to a batch they just wrote to themselves.
            and message
            and requested_batch not in (context.trusted_batch_ids or frozenset())
        ):
            # Every tool, not just the read ones: an unchecked anchor on a write
            # both aims the write at a stranger's batch and, once the id is
            # echoed back in the result, launders it into the trusted set.
            return policy_block(
                BLOCK_REASON_UNTRUSTED_BATCH,
                "安全拦截：只能操作本轮由服务端返回过的批次。"
                "请先读取当前草稿，再用它返回的批次编号。",
                missing=["trustedBatchId"],
                requestedBatchId=requested_batch[:64],
            )
        if not context.writes_allowed and tool_name in MUTATING_TOOL_NAMES:
            if context.attachment_context:
                return policy_block(
                    BLOCK_REASON_SOURCE_UNTRUSTED,
                    "安全拦截：本轮带有附件或图片，附件内容不能授权修改草稿或提交。",
                    missing=["trustedTextCommand"],
                    suggestion=self_checked_suggested_command(
                        tool_name, arguments, context
                    ),
                )
            advertised_verb = advertised_batch_assent_verb(message)
            if advertised_verb:
                return policy_block(
                    BLOCK_REASON_VERB_NOT_MATCHED,
                    f"安全拦截：识别到了已公示执行动词「{advertised_verb}」，"
                    "但附带的排除条件未通过确定性解析，或没有绑定到当前"
                    "发送者仍有效的批量确认票据；本次未写入。",
                    missing=["liveBatchTicketModifierBinding"],
                )
            return policy_block(
                BLOCK_REASON_VERB_NOT_MATCHED,
                "安全拦截：这条消息里没有识别到明确的执行指令"
                "（与历史、记忆或引用无关，只是当前这句话没被认成命令）。",
                missing=["executionVerb"],
                suggestion=self_checked_suggested_command(
                    tool_name, arguments, context
                ),
            )
        if tool_name in MUTATING_TOOL_NAMES and self._mutation_guard is not None:
            guard_error = self._mutation_guard(context, tool_name, arguments)
            if guard_error:
                return guard_error
        if (
            message
            and tool_name in MUTATING_TOOL_NAMES
            and not message_authorizes_mutation(message)
        ):
            advertised_verb = advertised_batch_assent_verb(message)
            if advertised_verb:
                return policy_block(
                    BLOCK_REASON_VERB_NOT_MATCHED,
                    f"安全拦截：识别到了已公示执行动词「{advertised_verb}」，"
                    "但附带的排除条件未通过确定性解析，或没有绑定到当前"
                    "发送者仍有效的批量确认票据；本次未写入。",
                    missing=["liveBatchTicketModifierBinding"],
                )
            return policy_block(
                BLOCK_REASON_VERB_NOT_MATCHED,
                "安全拦截：当前文字不是明确的执行指令。"
                "问句、解释、引用、备注或已经取消的说法不能授权写操作。",
                missing=["executionVerb"],
                suggestion=self_checked_suggested_command(
                    tool_name, arguments, context
                ),
            )
        if (
            message
            and tool_name in MUTATING_TOOL_NAMES
            and "confirmed" in arguments
            and not context.server_warning_confirmed
        ):
            return policy_block(
                BLOCK_REASON_TICKET_REQUIRED,
                "安全拦截：模型不能自行声明 confirmed=true。"
                "只有服务端保存的待确认票据才能进入确认执行流程。",
                missing=["serverTicket"],
            )
        if message and tool_name in MUTATING_TOOL_NAMES:
            binding_error = self._validate_current_message_binding(
                tool_name,
                arguments,
                context,
            )
            if binding_error:
                if binding_error.get("blockReason") == BLOCK_REASON_BINDING_INCOMPLETE:
                    suggestion = self_checked_suggested_command(
                        tool_name,
                        arguments,
                        context,
                    )
                    if suggestion:
                        binding_error["suggestedCommand"] = suggestion
                        binding_error["message"] = (
                            f"{binding_error['message']}"
                            "请把下面这条指令原样转述给用户，不要自创格式："
                            f"{suggestion}"
                        )
                return binding_error
        if tool_name == "keytao_batch_remove_draft_items" and message:
            ids = arguments.get("ids")
            if isinstance(ids, list) and len(ids) > 3 and not _DELETE_INTENT_RE.search(message):
                return {
                    "success": False,
                    "policyBlocked": True,
                    "blockReason": BLOCK_REASON_BULK_DELETE_NOT_REQUESTED,
                    "message": "安全拦截：当前消息不是批量删除请求，禁止一次删除多个草稿条目。请只删除本次明确需要替换的条目，或先向用户确认。",
                    "blockedIds": ids,
                }

        if tool_name != "keytao_batch_add_to_draft":
            return None

        reassignments = _find_code_reassignments(arguments.get("items"))
        if not reassignments or not message:
            return None

        blocked = [
            item for item in reassignments
            if (
                not _contains_exact_target(message, item["word"])
                or _is_word_protected(message, item["word"])
            )
        ]
        if not blocked:
            return None

        blocked_labels = [
            f"{item['word']} {item['oldCode']}→{item['newCode']}"
            for item in blocked
        ]
        return {
            "success": False,
            "policyBlocked": True,
            "blockReason": BLOCK_REASON_MANUAL_SHIFT_FORBIDDEN,
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
                f"{_describe_staged_mutation(tool_name, arguments)}\n"
                f"尚未写入。{pending_confirmation_copy()}"
                "也可使用机器人给出的确认票据。"
            ),
        }

    _validate_current_message_binding = staticmethod(
        _validate_current_message_binding
    )
