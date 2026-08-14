"""Tool execution adapter for the agent harness."""
import hashlib
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from nonebot.log import logger

from keytao_bot.utils import review_flags
from keytao_bot.utils.observability import observe_tool_call
from keytao_bot.utils.pending_confirmation import (
    parse_pending_candidate_selection,
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
MODEL_TOOL_RESULT_MAX_BATCH_ITEMS = 40
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
    projector: Callable[[Mapping[str, Any]], Dict[str, Any]]


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
    row = _present_fields(value, ("word", "code", "type", "weight"))
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


def _project_draft_listing(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = _present_fields(payload, (
        "success", "batchId", "batchUrl", "count", "summary", "message",
    ))
    values = payload.get("items")
    if isinstance(values, list):
        items = [
            item
            for value in values[:MODEL_TOOL_RESULT_MAX_BATCH_ITEMS]
            if (item := _compact_draft_item(value)) is not None
        ]
        result["items"] = items
        if len(values) > len(items):
            remaining = len(values) - len(items)
            result["itemsTruncated"] = True
            result["itemsTruncationNotice"] = (
                f"「…另有 {remaining} 条」模型可见列表已截断；"
                "不得据此判断完整条目集合"
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
    "keytao_list_draft_items": ModelToolResultProjection(12_500, _project_draft_listing),
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


def project_tool_result_for_model(tool_name: str, result_json: str) -> str:
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
    projected = projection.projector(payload)
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


_DELETE_INTENT_RE = re.compile(
    r"删除|删掉|删干净|去掉|移除|撤销|清空|清理|全部删|都删"
)
_POSITIONAL_REORDER_QUOTED_ENTRY_PATTERN = (
    r"(?:「[^」]{1,16}」|“[^”]{1,16}”|‘[^’]{1,16}’)"
)
# Keep this in sync with the Chinese Phrase validator and the encoder chain:
# Next accepts letters with a six-key ceiling; the bot normalizes generated
# phrase candidates to lowercase ASCII.  Shape is necessary but never enough
# for permission -- the binding gate also requires a server-read capability.
_POSITIONAL_REORDER_CODE_PATTERN = r"[a-z]{1,6}(?![A-Za-z])"
_POSITIONAL_REORDER_PLAIN_ENTRY_PATTERN = r"[\u3400-\u9fff]{1,8}"
_POSITIONAL_REORDER_RELATION_PATTERN = r"(?:前面|后面|之前|之后|前|后)"
_POSITIONAL_CREATE_FRONT_RELATIONS = frozenset({"前面", "之前", "前"})
_POSITIONAL_CREATE_BACK_RELATIONS = frozenset({"后面", "之后", "后"})
# Exact same-code lexicon. These forms are recognized only when they are
# unquoted and occur in the positional command clause:
#
#   同码 / 同编码 / 同代码
#   同一码 / 同一编码 / 同一代码
#   同一个码 / 同一个编码 / 同一个代码
#   相同码 / 相同编码 / 相同代码 (with optional 的)
#   码相同 / 编码相同 / 代码相同 (with optional 保持)
#   重码 / 重复码 / 重复编码 / 重复代码 (with optional 的)
_POSITIONAL_SAME_CODE_MARKER_PATTERN = (
    r"(?:"
    r"同(?:一(?:个)?|一个)?(?:码|编码|代码)|"
    r"相同(?:的)?(?:码|编码|代码)|"
    r"(?:码|编码|代码)(?:保持)?相同|"
    r"重码|重复(?:的)?(?:码|编码|代码)"
    r")"
)
_POSITIONAL_SAME_CODE_MARKER_RE = re.compile(
    _POSITIONAL_SAME_CODE_MARKER_PATTERN
)
_POSITIONAL_COMMAND_VERB_RE = re.compile(
    r"放在|放到|排在|挪到|移到|提到|提前到|往前|往后|靠前|靠后"
)
_POSITIONAL_REORDER_ORDINAL_PATTERN = (
    r"第(?:[一二三四五六七八九十百千万两]+|\d{1,3})(?:个|位)"
)
_POSITIONAL_REORDER_DESTINATION_PATTERN = (
    rf"(?:"
    rf"{_POSITIONAL_REORDER_CODE_PATTERN}|"
    rf"{_POSITIONAL_REORDER_ORDINAL_PATTERN}|"
    rf"{_POSITIONAL_REORDER_RELATION_PATTERN}|"
    rf"{_POSITIONAL_REORDER_QUOTED_ENTRY_PATTERN}"
    rf"(?:{_POSITIONAL_REORDER_RELATION_PATTERN})?|"
    rf"{_POSITIONAL_REORDER_PLAIN_ENTRY_PATTERN}"
    rf"(?:{_POSITIONAL_REORDER_RELATION_PATTERN})?"
    rf")"
)
_POSITIONAL_REORDER_DESTINATION_EXPRESSION_PATTERN = (
    rf"(?:放在|放到|排在|挪到|移到|提到|提前到)"
    rf"{_POSITIONAL_REORDER_DESTINATION_PATTERN}"
)
_POSITIONAL_REORDER_COMMAND_DESTINATION_EXPRESSION_PATTERN = (
    rf"(?:放在|放到|排在|挪到|移到|提到|提前到)\s*"
    rf"{_POSITIONAL_REORDER_DESTINATION_PATTERN}"
)
_POSITIONAL_REORDER_RELATIVE_EXPRESSION_PATTERN = (
    r"(?:往前|往后)(?:(?:挪|移)(?:一位|一下)?)?|"
    r"(?:靠前|靠后)(?:一点|一位|一些)?"
)
_POSITIONAL_REORDER_INTENT_PATTERN = (
    rf"(?:放在|放到|排在|挪到|移到|提到|提前到)"
    rf"(?={_POSITIONAL_REORDER_DESTINATION_PATTERN})|"
    r"(?:往前|往后)(?=(?:挪|移)(?:一位|一下)?)|"
    r"靠前(?=(?:一点|一位|一些))|靠后(?=(?:一点|一位|一些))"
)
_POSITIONAL_REORDER_INTENT_RE = re.compile(_POSITIONAL_REORDER_INTENT_PATTERN)
_NON_POSITIONAL_MUTATION_INTENT_PATTERN = (
    r"加词|添加|加入|加到|新增|创建|写入|放入|收录|录入|记入|提交|提审|送审|发起审核|"
    r"删除|删掉|删干净|去掉|移除|清空|清理|"
    r"撤销|撤回|召回|修改|改成|改为|替换|顺延|挪开|重新编码|保留|批量处理|"
    r"调整权重|修改权重|权重调整(?:为|到)?|权重修改(?:为|到)?|权重改(?:为|到)"
    r"|都删|其余删|其他删"
)
_NON_POSITIONAL_MUTATION_INTENT_RE = re.compile(
    _NON_POSITIONAL_MUTATION_INTENT_PATTERN
)
_MUTATION_INTENT_RE = re.compile(
    _NON_POSITIONAL_MUTATION_INTENT_PATTERN
    + r"|"
    + _POSITIONAL_REORDER_INTENT_PATTERN
)
_POSITIONAL_REORDER_SUBJECT_PATTERN = (
    r"(?:「[^」]{1,16}」|“[^”]{1,16}”|‘[^’]{1,16}’|"
    r"(?:(?![把将])[\u3400-\u9fffA-Za-z0-9_-]){1,16}?)"
)
_POSITIONAL_REORDER_BARE_SUBJECT_PATTERN = (
    r"(?:「[^」]{1,16}」|“[^”]{1,16}”|‘[^’]{1,16}’|"
    r"(?:(?![把将])[\u3400-\u9fffA-Za-z0-9_-]){1,8}?)"
)
_POSITIONAL_REORDER_COMMAND_RE = re.compile(
    rf"^(?:(?:把|将)\s*{_POSITIONAL_REORDER_SUBJECT_PATTERN}|"
    rf"{_POSITIONAL_REORDER_BARE_SUBJECT_PATTERN})\s*(?:的编码)?\s*"
    rf"(?:{_POSITIONAL_REORDER_COMMAND_DESTINATION_EXPRESSION_PATTERN}|"
    rf"{_POSITIONAL_REORDER_RELATIVE_EXPRESSION_PATTERN})"
    r"(?:一下|吧|了|好吗|行吗|可以吗)?[。.!！?？]?$"
)
_POSITIONAL_REORDER_RAW_COMMAND_RE = re.compile(
    rf"^(?:(?:把|将)\s*{_POSITIONAL_REORDER_SUBJECT_PATTERN}|"
    rf"{_POSITIONAL_REORDER_BARE_SUBJECT_PATTERN})\s*(?:的编码)?\s*"
    rf"(?:放在|放到|排在|挪到|移到|提到|提前到)\s*"
    r"(?P<destination>.+?)"
    r"(?:一下|吧|了|好吗|行吗|可以吗)?[。.!！?？]?$"
)
_POSITIONAL_REORDER_DESTINATION_CAPTURE_RE = re.compile(
    rf"(?:放在|放到|排在|挪到|移到|提到|提前到)\s*"
    rf"(?P<destination>{_POSITIONAL_REORDER_DESTINATION_PATTERN})"
    r"(?:一下|吧|了|好吗|行吗|可以吗)?[。.!！?？]?$"
)
_POSITIONAL_REORDER_QUOTED_TARGET_RE = re.compile(
    r"^(?:「[^」]{1,16}」|“[^”]{1,16}”|‘[^’]{1,16}’)$"
)
_POSITIONAL_REORDER_PLAIN_LOCATIVE_SUFFIX_RE = re.compile(
    r"(?:里|上|下|中|内|外|旁|边|侧|处|口|角|前|后|附近|周围)$"
)
_POSITIONAL_REORDER_RELATIVE_FRAGMENT_RE = re.compile(
    rf"^(?:{_POSITIONAL_REORDER_RELATIVE_EXPRESSION_PATTERN})"
    r"(?:一下|吧|了|好吗|行吗|可以吗)?[。.!！?？]?$"
)
_POSITIONAL_REORDER_NARRATIVE_TAIL_RE = re.compile(
    r"(?:挺好|很好|不错|合适|更好|比较好|不好|不妥|太差|有误|很怪|"
    r"正常|恰当|错了|错误|太怪|奇怪|较好|更差|离谱|正确|"
    r"欠妥|可行|合理|这样(?:更)?合理|看起来不错|是个好主意|比较合适|更合适|"
    r"只是陈述)[。.!！]?$"
)
_POSITIONAL_REORDER_CHOICE_QUESTION_RE = re.compile(
    r"(?:还是|或者|要么|二选一|择一|或)"
)
_POSITIONAL_REORDER_TEMPORAL_DESTINATION_RE = re.compile(
    r"^(?:今天|明天|后天|昨天|今晚|明早|稍后|以后|未来|"
    r"下周|下月|下季度|明年|周[一二三四五六日天]|"
    r"\d{1,2}(?:点|时|月|日|号))$"
)
_POSITIONAL_REORDER_TRAILING_MODIFIER_RE = re.compile(
    r"^(?:谢谢|谢谢你|多谢|辛苦了|拜托了|麻烦了|感谢|感谢你|"
    r"劳驾|拜托|有劳|谢啦|"
    r"目标编码请你自己查清楚后直接完成|不要问我)[。.!！]?$"
)
_POSITIONAL_SUBORDINATE_CONTEXT_RE = re.compile(
    r"^(?:关于)?(?:你|我|他|她)(?:刚才)?.*"
    r"(?:的内容|过的那段|的说法|的吃席|过的那批)$"
)
_POSITIONAL_REPORTED_CONTEXT_RE = re.compile(
    r"^(?:他说|她说|群里有人说|上条消息是|据说|听说|大家说|有人说|"
    r"消息里说|消息称|据悉|报道称|他称|她称|传闻|网传|我觉得|"
    r"备查|留存|存证|摘记|纪要|昨天|她|他|媒体称|外界认为|"
    r"留作备查|会议纪要|有传言称|小王表示|请存证|说|记|称|录|传|述)"
)
_POSITIONAL_BARE_DATA_CONTEXT_RE = re.compile(
    r"^(?:说|记|称|录|传|述).{0,24}"
    r"(?:放在|放到|排在|挪到|移到|提到|提前到|往前|往后|靠前|靠后)"
)
_POSITIONAL_CONTEXT_NEGATION_RE = re.compile(
    r"^(?:没|未|尚未|不应|不应该|并非|不能)(?:[，,；;：:]|.{0,8})"
    r".{0,40}(?:放在|放到|排在|挪到|移到|提到|提前到|往前|往后|靠前|靠后)"
)
_NEGATED_POSITIONAL_REORDER_RE = re.compile(
    r"(?:不要|别(?!的)|无需|不用|禁止|先不|暂时不|不必|不再|"
    r"不需要|甭|勿|不宜|无须|毋须|绝不能).{0,40}"
    r"(?:放在|放到|排在|挪到|移到|提到|提前到|往前|往后|靠前|靠后)"
)
_POSITIONAL_REORDER_EXPLANATION_RE = re.compile(
    r"^怎么.{1,40}"
    r"(?:放在|放到|排在|挪到|移到|提到|提前到|往前|往后|靠前|靠后)"
)
_POSITIONAL_REORDER_LOCATIVE_DESTINATION_RE = re.compile(
    r"(?:放在|放到|排在|挪到|移到|提到|提前到)"
    r"[\u3400-\u9fff]{1,4}[里上下中内外](?:一下|吧|了)?[。.!！?？]?$"
)
_NEGATED_MUTATION_RE = re.compile(
    rf"(?:"
    rf"(?<![把将])"
    rf"(?:不要|别(?!的)|无需|不用|禁止|不要真的|先不|暂时不|不必|不再|不需要|甭|勿)"
    rf".{{0,12}}(?:{_MUTATION_INTENT_RE.pattern})"
    rf")"
)
_NEGATED_NON_POSITIONAL_MUTATION_RE = re.compile(
    rf"(?:"
    rf"(?<![把将])"
    rf"(?:不要|别(?!的)|无需|不用|禁止|不要真的|先不|暂时不|不必|不再|不需要|甭|勿)"
    rf".{{0,12}}(?:{_NON_POSITIONAL_MUTATION_INTENT_PATTERN})"
    rf")"
)
_STANDALONE_NEGATION_CLAUSE_RE = re.compile(
    r"(?:请)?(?:不要|别|无需|不用|禁止|不要真的|先不|暂时不|不必|不再|"
    r"不需要|甭|勿|先别|暂时别)(?:了)?"
)
_EXPLANATION_ONLY_RE = re.compile(
    r"(?:什么意思|什么含义|解释一下|说明一下|怎么做|如何操作|操作方法|"
    r"为什么|为何|教程|示例|假设|如果|是否支持|能否介绍|翻译|怎么理解|怎么说|"
    r"举个例子|举例|怎么(?:把|将)|"
    r"会发生什么|会怎样|会如何|有什么后果|后果是什么|有什么影响)"
)
_TEXT_TRANSFORM_RE = re.compile(r"(?:改写|润色|复述|翻译|引用|摘录|转述)")
_QUESTION_RE = re.compile(
    r"[?？]|(?:是否|能否|可否|能不能|可不可以|会不会|是不是|要不要|怎么样|怎样|如何|"
    r"想知道|之后的结果|结果是什么)|"
    r"(?:吗|么|呢|好不好|行不行|对不对|可以不|"
    r"不可以|不行)(?:[。.!！])?$"
)
_ABORT_RE = re.compile(r"(?:算了|取消|别执行|不要执行|先不要|不用了|不做了|别做了)")
_EXPLICIT_REQUEST_PREFIX_RE = re.compile(
    r"^(?:请|麻烦|帮我|给我|现在|立即|直接|确认|我要|我想|把|将|替我|为我|"
    r"能不能|可不可以|能否|可否|可以帮我|可以请你|只|仅|除了)"
)
_POLITE_EXECUTION_PREFIX_RE = re.compile(
    r"^(?:"
    r"请(?!问)|麻烦|帮我|给我|替我|为我|"
    r"(?:能不能|可不可以|能否|可否|可以)(?:请|帮我|替我|为我)|"
    r"(?:能不能|可不可以|能否|可否).{0,80}(?:一下|现在|立即|直接|吧|好吗|行吗)"
    r")"
)
_META_DISCUSSION_RE = re.compile(
    r"(?:这句话|这段话|这条消息|引用|消息里|上面写|刚才说).{0,80}"
    r"(?:什么意思|什么含义|解释|说明|怎么理解)"
)
_DATA_CONTEXT_RE = re.compile(
    r"^(?:"
    r"(?:请|麻烦|帮我|给我|我要|我想)?"
    r"(?:分析|判断|翻译|解释|改写|复述|记录|排版|标注)"
    r"(?:以下|一下|这段|这句|这条|一句话|内容|消息|用户请求|是否)|"
    r"(?:他说|她说|群里有人说|上条消息是|据说|听说|大家说|有人说|"
    r"消息里说|消息称|据悉|报道称|他称|她称|传闻|网传|我觉得|"
    r"备查|留存|存证|摘记|纪要|昨天|她|他)[：:；;]?"
    r")"
)
_RECORD_FRAME_RE = re.compile(
    r"(?:"
    r"(?:做|作|留)\s*(?:个|份)?\s*"
    r"(?:记\s*录|記\s*錄|笔\s*记|筆\s*記|备\s*注|備\s*註|"
    r"标\s*记|標\s*記|备\s*忘|備\s*忘)|"
    r"(?:记\s*录|記\s*錄)(?:\s*下\s*(?:来|來)|\s*一\s*下)?|"
    r"(?:记|記)(?:\s*下\s*(?:来|來)?|\s*一\s*下)|"
    r"(?:备\s*注|備\s*註|标\s*记|標\s*記|登\s*记|登\s*記|"
    r"保存|记\s*载|記\s*載|备\s*忘|備\s*忘)"
    r"(?:\s*下\s*(?:来|來)|\s*一\s*下|\s*在\s*案)?|"
    r"(?:写|寫|抄|录|錄)(?:\s*下\s*(?:来|來)?|\s*一\s*下)|"
    r"(?:写|寫)\s*(?:进|進|入)\s*"
    r"(?:备\s*忘\s*录|備\s*忘\s*錄|笔\s*记|筆\s*記|记\s*录|記\s*錄)|"
    r"(?:存|归|歸|留)\s*(?:档|檔)|"
    r"(?:转\s*告|轉\s*告|转\s*达|轉\s*達|传\s*达|傳\s*達)"
    r"(?:\s*一\s*下|\s*给\s*[\u3400-\u9fff]{1,8})?"
    r")"
)
_QUOTED_DATA_RE = re.compile(r"「[^」]*」|“[^”]*”|‘[^’]*’|\"[^\"]*\"|'[^']*'")
_RECORD_FRAME_BRACKETED_DATA_RE = re.compile(
    r"『[^』]*』|《[^》]*》|【[^】]*】|\([^)]*\)|（[^）]*）|〈[^〉]*〉|〔[^〕]*〕"
)
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
# A leading platform mention is routing metadata, not part of the command.
# In this repo the plugin already strips it (openai_chat._LEADING_COMMAND_PREFIX_RE),
# so on the production path this never matches.  It exists so that the
# "@我 ..." remediation command self-checks through exactly the validators it
# will face, instead of through a stripped variant of itself.
_LEADING_MENTION_RE = re.compile(r"^\s*@[^\s@]{1,24}[\s:：]+")
_WHOLE_MESSAGE_LEADING_ADDRESS_RE = re.compile(
    r"^\s*(?:@[^\s@]{1,24}|键道|喵喵)[\s:：，,]*",
    re.IGNORECASE,
)
_WHOLE_MESSAGE_CLOSING_FILLER_RE = re.compile(
    r"\s*(?:[，,。.!！?？;；]\s*)?"
    r"(?:谢谢|谢谢你|多谢|辛苦了|拜托了|麻烦了|感谢|感谢你|"
    r"劳驾|拜托|有劳|谢啦)"
    r"[。.!！?？]*\s*$"
)
_WHOLE_MESSAGE_QUOTE_PATTERNS = (
    re.compile(r"^「(?P<content>[^」]*)」$", re.DOTALL),
    re.compile(r"^“(?P<content>[^”]*)”$", re.DOTALL),
    re.compile(r"^『(?P<content>[^』]*)』$", re.DOTALL),
)
_COMMAND_LEAD_IN_PREFIXES = (
    "请", "麻烦", "帮我", "给我", "现在", "立即", "直接", "确认", "执行",
    "我要", "我想", "替我", "为我", "能不能", "可不可以", "能否", "可否",
    "可以帮我", "可以请你", "并", "并且", "同时", "然后", "再", "还要",
    "以及", "另外", "接着", "顺便", "麻烦你", "帮忙", "劳驾", "喵喵", "你好", "在吗",
)
# Longest-first alternation keeps a shorter token such as "并" from consuming
# the start of "并且" when the prefix is stripped independently of the command.
_COMMAND_PREFIX_PATTERN = rf"(?:{'|'.join(sorted(
    (re.escape(prefix) for prefix in _COMMAND_LEAD_IN_PREFIXES),
    key=len,
    reverse=True,
))})*"
_COMMAND_PREFIX_RE = re.compile(rf"^{_COMMAND_PREFIX_PATTERN}")
_MULTI_ADD_VERB_RE = re.compile(
    r"加词|添加|加入|加到|新增|创建|写入|放入|收录|录入|记入"
)
_MULTI_ADD_ADDRESS_CLAUSES = frozenset({"喵喵"})
_MAX_AUTHORIZED_MULTI_ADD_ITEMS = 10


@dataclass(frozen=True)
class _AuthorizedAddClause:
    word: str
    code: str = ""


@dataclass(frozen=True)
class _MultiAddAuthorization:
    clauses: Tuple[_AuthorizedAddClause, ...]
    valid: bool
    refused_clauses: Tuple[str, ...] = ()


def _parse_complete_add_clause(clause: str) -> Optional[_AuthorizedAddClause]:
    """Parse one closed add command without deriving either target or code."""
    candidate = re.sub(r"\s+", " ", str(clause or "")).strip()
    candidate = _COMMAND_PREFIX_RE.sub("", candidate, count=1).strip()
    verb = _MULTI_ADD_VERB_RE.match(candidate)
    if verb is None:
        return None
    remainder = candidate[verb.end():].strip()
    remainder = re.sub(
        r"^(?:声笔笔单字|声笔笔|词组|词条|词语|单字|补充|符号|链接|英文)\s*",
        "",
        remainder,
        count=1,
    )
    remainder = remainder.lstrip("：: ")
    if not remainder:
        return None

    word = ""
    if remainder[0] in "「“‘\"'":
        closing = {"「": "」", "“": "”", "‘": "’", '"': '"', "'": "'"}[
            remainder[0]
        ]
        closing_at = remainder.find(closing, 1)
        if closing_at <= 1:
            return None
        word = remainder[1:closing_at].strip()
        remainder = remainder[closing_at + 1:].strip()
    else:
        word_match = re.match(r"[\u3400-\u9fffA-Za-z0-9_-]{1,32}", remainder)
        if word_match is None:
            return None
        word = word_match.group(0)
        remainder = remainder[word_match.end():].strip()

    if not word or _MULTI_ADD_VERB_RE.fullmatch(word):
        return None
    remainder = re.sub(
        r"^(?:(?:的)?(?:编码|代码)(?:为|是)?\s*)?[:：]?\s*",
        "",
        remainder,
        count=1,
    )
    code = ""
    if remainder:
        code_match = re.match(r"[A-Za-z]{1,12}(?![A-Za-z])", remainder)
        if code_match is not None:
            code = code_match.group(0).lower()
            remainder = remainder[code_match.end():].strip()
    remainder = re.sub(
        r"^(?:(?:加到|加入|添加|放入|写入)(?:当前)?草稿)?"
        r"(?:一下)?(?:吧|啦|了)?$",
        "",
        remainder,
    ).strip()
    if remainder:
        return None
    return _AuthorizedAddClause(word=word, code=code)


def _multi_add_authorization_contract(
    message: str,
) -> Optional[_MultiAddAuthorization]:
    """Require every clause to be an add, closed politeness, or leading address."""
    source = _LEADING_MENTION_RE.sub(
        "", trusted_mutation_source(message), count=1
    )
    raw_clauses = [
        clause.strip()
        for clause in _COMMAND_CLAUSE_SPLIT_RE.split(source)
        if clause.strip()
    ]
    add_intent_count = sum(
        1 for clause in raw_clauses if _MULTI_ADD_VERB_RE.search(clause)
    )
    if add_intent_count < 2:
        return None

    parsed: List[_AuthorizedAddClause] = []
    refused: List[str] = []
    for index, clause in enumerate(raw_clauses):
        compact = re.sub(r"\s+", "", clause)
        add_clause = _parse_complete_add_clause(clause)
        if add_clause is not None:
            parsed.append(add_clause)
            continue
        if (
            index == 0
            and compact in _MULTI_ADD_ADDRESS_CLAUSES
        ) or _POSITIONAL_REORDER_TRAILING_MODIFIER_RE.fullmatch(compact):
            continue
        refused.append(clause[:120])
    return _MultiAddAuthorization(
        clauses=tuple(parsed),
        valid=bool(parsed) and not refused and len(parsed) == add_intent_count,
        refused_clauses=tuple(refused),
    )


def authorized_multi_add_items(message: str) -> Tuple[Dict[str, str], ...]:
    """Return an exact literal multi-add item set, or an empty tuple.

    This is intentionally narrower than general mutation authorization: every
    clause must carry its own add verb, word, and literal code. The returned set
    is therefore safe to replay only after each pair has a non-BLOCK review.
    """
    authorization = _multi_add_authorization_contract(message)
    if (
        authorization is None
        or not authorization.valid
        or len(authorization.clauses) > _MAX_AUTHORIZED_MULTI_ADD_ITEMS
        or any(not clause.code for clause in authorization.clauses)
    ):
        return ()
    identities = [(clause.word, clause.code) for clause in authorization.clauses]
    if len(set(identities)) != len(identities):
        return ()
    return tuple(
        {"action": "Create", "word": word, "code": code}
        for word, code in identities
    )


def _multi_add_items_match_authorized_set(
    authorization: _MultiAddAuthorization,
    items: Any,
    trusted_codes_by_word: Dict[str, frozenset[str]],
) -> bool:
    """Prove the tool item set equals the union of authorizing add clauses."""
    if not authorization.valid or not isinstance(items, list):
        return False
    expected = list(dict.fromkeys(authorization.clauses))
    actual: List[Tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            return False
        action = str(item.get("action") or "Create").strip()
        word = str(item.get("word") or "").strip()
        code = str(item.get("code") or "").strip().lower()
        if action != "Create" or item.get("old_word") or not word or not code:
            return False
        actual.append((word, code))
    if len(actual) != len(expected) or len(set(actual)) != len(actual):
        return False

    unmatched = list(actual)
    for clause in sorted(expected, key=lambda item: not bool(item.code)):
        candidates = [
            index
            for index, (word, code) in enumerate(unmatched)
            if word == clause.word
            and (
                code == clause.code
                if clause.code
                else code in trusted_codes_by_word.get(word, frozenset())
            )
        ]
        if len(candidates) != 1:
            return False
        unmatched.pop(candidates[0])
    return not unmatched


@dataclass(frozen=True)
class _PositionalDestination:
    kind: str
    target: str = ""
    quoted: bool = False


@dataclass(frozen=True)
class _PositionalCreateBinding:
    code: str
    destination_word: str
    relation: str
    phrase_type: str = "Phrase"
    weight: Optional[int] = None
    resulting_words: Tuple[str, ...] = ()
    bumped_entries: Tuple[Tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class ToolExecutionRoute:
    tool_name: str
    arguments: Dict[str, Any]
    positional_binding: Optional[_PositionalCreateBinding] = None
    response: Optional[Dict[str, Any]] = None


def _unquote_positional_entry(value: str) -> Optional[str]:
    pairs = {"「": "」", "“": "”", "‘": "’"}
    if len(value) >= 3 and pairs.get(value[0]) == value[-1]:
        return value[1:-1]
    return None


def _parse_positional_destination(destination: str) -> Optional[_PositionalDestination]:
    value = str(destination or "").strip()
    if not value:
        return None
    if re.fullmatch(_POSITIONAL_REORDER_CODE_PATTERN, value):
        return _PositionalDestination("code", target=value)
    if re.fullmatch(_POSITIONAL_REORDER_ORDINAL_PATTERN, value):
        return _PositionalDestination("ordinal")
    if re.fullmatch(_POSITIONAL_REORDER_RELATION_PATTERN, value):
        return _PositionalDestination("relative")

    quoted = _unquote_positional_entry(value)
    if quoted is not None:
        return _PositionalDestination("entry", target=quoted, quoted=True)

    for relation in ("前面", "后面", "之前", "之后", "前", "后"):
        if not value.endswith(relation):
            continue
        raw_target = value[:-len(relation)]
        if not raw_target:
            return _PositionalDestination("relative")
        quoted_target = _unquote_positional_entry(raw_target)
        if quoted_target is not None:
            return _PositionalDestination(
                "entry",
                target=quoted_target,
                quoted=True,
            )
        if re.fullmatch(_POSITIONAL_REORDER_PLAIN_ENTRY_PATTERN, raw_target):
            return _PositionalDestination("entry", target=raw_target)
        return None

    if re.fullmatch(_POSITIONAL_REORDER_PLAIN_ENTRY_PATTERN, value):
        return _PositionalDestination("entry", target=value)
    return None


def _positional_create_operands(
    message: str,
) -> Optional[Tuple[str, str, str]]:
    """Extract subject, named destination, and explicit relative side."""
    message, _marker_found = _normalize_positional_same_code_markers(message)
    if not _has_complete_positional_reorder_command(message):
        return None
    source = _LEADING_MENTION_RE.sub(
        "", trusted_mutation_source(message), count=1
    )
    clauses = [
        clause.strip() for clause in _COMMAND_CLAUSE_SPLIT_RE.split(source)
        if clause.strip()
    ]
    while (
        len(clauses) > 1
        and _POSITIONAL_REORDER_TRAILING_MODIFIER_RE.fullmatch(clauses[-1])
    ):
        clauses.pop()
    if len(clauses) != 1:
        return None
    candidate = _COMMAND_PREFIX_RE.sub("", clauses[0], count=1).lstrip()
    subject_match = re.match(
        rf"^(?:(?:把|将)\s*(?P<prefixed>{_POSITIONAL_REORDER_SUBJECT_PATTERN})|"
        rf"(?P<bare>{_POSITIONAL_REORDER_BARE_SUBJECT_PATTERN}))\s*"
        rf"(?:的编码)?\s*(?:放在|放到|排在|挪到|移到|提到|提前到)",
        candidate,
    )
    if subject_match is None:
        return None
    raw_subject = str(
        subject_match.group("prefixed") or subject_match.group("bare") or ""
    ).strip()
    subject = _unquote_positional_entry(raw_subject) or raw_subject
    raw_destination = _raw_positional_destination_from_command(message)
    parsed_destination = _positional_destination_from_command(message)
    if (
        not subject
        or raw_destination is None
        or parsed_destination is None
        or parsed_destination.kind != "entry"
    ):
        return None
    relation = ""
    for candidate_relation in ("前面", "后面", "之前", "之后", "前", "后"):
        if not raw_destination.endswith(candidate_relation):
            continue
        base = raw_destination[:-len(candidate_relation)].strip()
        parsed_base = _parse_positional_destination(base)
        if (
            parsed_base is not None
            and parsed_base.kind == "entry"
            and parsed_base.target == parsed_destination.target
        ):
            relation = candidate_relation
            break
    return subject, parsed_destination.target, relation


def _positional_destination_from_command(message: str) -> Optional[_PositionalDestination]:
    message, _marker_found = _normalize_positional_same_code_markers(message)
    source = _LEADING_MENTION_RE.sub(
        "", trusted_mutation_source(message), count=1
    )
    clauses = [
        clause.strip() for clause in _COMMAND_CLAUSE_SPLIT_RE.split(source)
        if clause.strip()
    ]
    while (
        len(clauses) > 1
        and _POSITIONAL_REORDER_TRAILING_MODIFIER_RE.fullmatch(clauses[-1])
    ):
        clauses.pop()
    if len(clauses) != 1:
        return None
    candidate = _COMMAND_PREFIX_RE.sub("", clauses[0], count=1).lstrip()
    if not _POSITIONAL_REORDER_COMMAND_RE.fullmatch(candidate):
        return None
    match = _POSITIONAL_REORDER_DESTINATION_CAPTURE_RE.search(candidate)
    if match is None:
        return _PositionalDestination("relative")
    return _parse_positional_destination(match.group("destination"))


def _raw_positional_destination_from_command(message: str) -> Optional[str]:
    message, _marker_found = _normalize_positional_same_code_markers(message)
    source = _LEADING_MENTION_RE.sub(
        "", trusted_mutation_source(message), count=1
    )
    clauses = [
        clause.strip() for clause in _COMMAND_CLAUSE_SPLIT_RE.split(source)
        if clause.strip()
    ]
    while (
        len(clauses) > 1
        and _POSITIONAL_REORDER_TRAILING_MODIFIER_RE.fullmatch(clauses[-1])
    ):
        clauses.pop()
    if len(clauses) != 1:
        return None
    candidate = _COMMAND_PREFIX_RE.sub("", clauses[0], count=1).lstrip()
    match = _POSITIONAL_REORDER_RAW_COMMAND_RE.fullmatch(candidate)
    return match.group("destination").strip() if match else None


def _has_raw_positional_relative_tail(message: str) -> bool:
    message, _marker_found = _normalize_positional_same_code_markers(message)
    source = _LEADING_MENTION_RE.sub(
        "", trusted_mutation_source(message), count=1
    )
    clauses = [
        clause.strip() for clause in _COMMAND_CLAUSE_SPLIT_RE.split(source)
        if clause.strip()
    ]
    while (
        len(clauses) > 1
        and _POSITIONAL_REORDER_TRAILING_MODIFIER_RE.fullmatch(clauses[-1])
    ):
        clauses.pop()
    if len(clauses) != 1:
        return False
    candidate = _COMMAND_PREFIX_RE.sub("", clauses[0], count=1).lstrip()
    return bool(
        re.search(
            rf".{{1,32}}(?:{_POSITIONAL_REORDER_RELATIVE_EXPRESSION_PATTERN})"
            r"(?:一下|吧|了|好吗|行吗|可以吗)?[。.!！?？]?$",
            candidate,
        )
    )


def _has_complete_positional_reorder_command(message: str) -> bool:
    """Match positional grammar without erasing subject-boundary whitespace."""
    message, _marker_found = _normalize_positional_same_code_markers(message)
    source = _LEADING_MENTION_RE.sub(
        "", trusted_mutation_source(message), count=1
    )
    clauses = [
        clause for clause in _COMMAND_CLAUSE_SPLIT_RE.split(source)
        if clause.strip()
    ]
    while (
        len(clauses) > 1
        and _POSITIONAL_REORDER_TRAILING_MODIFIER_RE.fullmatch(
            clauses[-1].strip()
        )
    ):
        clauses.pop()
    # Only a closed trailing politeness clause may follow the instruction.
    # Arbitrary extra clauses never become positional authority.
    if len(clauses) != 1:
        return False
    for clause in clauses:
        candidate = clause.strip()
        candidate = _COMMAND_PREFIX_RE.sub("", candidate, count=1).lstrip()
        if (
            _POSITIONAL_REORDER_COMMAND_RE.fullmatch(candidate)
            and not _positional_destination_is_ambiguous_non_command(candidate)
        ):
            return True
    return False


def _positional_destination_is_ambiguous_non_command(candidate: str) -> bool:
    """Fail closed for plain destinations that also read as questions or places.

    A quoted operand, a code, or an explicit ``前面/后面`` relation has a
    structural word/code boundary.  A bare destination such as ``哪`` or
    ``冰箱旁`` does not: treating it as a dictionary word would turn questions
    and ordinary location statements into write consent.  Users can remove
    that ambiguity by quoting the word or naming the relative side.
    """
    match = _POSITIONAL_REORDER_DESTINATION_CAPTURE_RE.search(candidate)
    if match is None:
        return False
    parsed = _parse_positional_destination(match.group("destination"))
    if parsed is None:
        return True
    if parsed.kind in {"code", "ordinal", "relative"} or parsed.quoted:
        return False
    return bool(
        re.search(r"哪|谁|什么|何处|何人|不", parsed.target)
        or _POSITIONAL_REORDER_PLAIN_LOCATIVE_SUFFIX_RE.search(parsed.target)
        or _POSITIONAL_REORDER_TEMPORAL_DESTINATION_RE.fullmatch(parsed.target)
    )


def _has_positional_choice_question(text: str) -> bool:
    masked = _QUOTED_DATA_RE.sub("", text)
    return bool(_POSITIONAL_REORDER_CHOICE_QUESTION_RE.search(masked))


_ENTRY_MUTATION_FOR_FRAME_OPERAND_RE = _MUTATION_INTENT_RE
_RECORD_FRAME_OPERAND_AFTER_RE = re.compile(
    # Keep this filler vocabulary closed and bounded. It describes only the
    # entry label, its code, and the common "的编码" bridge; arbitrary prose
    # must never be skipped before looking for the mutation verb.
    r"^(?:(?:(?:这个|该)?(?:词|词条)|的(?:编码|代码)|[A-Za-z]{1,12})){0,3}"
    rf"(?:{_MUTATION_INTENT_RE.pattern})"
)
_RECORD_FRAME_DRAFT_DELETE_PREFIX_RE = re.compile(
    rf"^{_COMMAND_PREFIX_PATTERN}"
    r"(?:删除|删掉|去掉|移除)(?:草稿|批次)(?:里|里的|中|中的)$"
)
_RECORD_FRAME_POSTPOSED_OPERAND_RE = re.compile(
    r"(?:"
    r"(?:把|将)(?:这个|该)(?:词|词条)(?:加入|加到|放入|写入)草稿|"
    r"(?:添加|新增|创建)(?:词条|词语)"
    r")[：:]$"
)
_RECORD_FRAME_EMBEDDED_MUTATION_SUFFIX_RE = re.compile(
    r"^(?:(?:的(?:编码|代码)|(?:这个|该)?(?:词|词条)|[A-Za-z]{1,12})){0,3}$"
)
_RECORD_FRAME_SEPARATOR_CHARS = frozenset("、：:—，。 \t\r\n")
_COMPLETE_MUTATION_NOISE_RE = re.compile(
    r"(?:请|請|麻烦|麻煩|帮我|幫我|给我|給我|替我|为我|為我|"
    r"我要|我想|现在|現在|立即|直接|确认|確認|执行|執行|"
    r"能不能|可不可以|能否|可否|可以帮我|可以幫我|可以请你|可以請你|"
    r"把|将|將|这句|這句|这段|這段|这条|這條|一句话|一句話|"
    r"消息|内容|內容|用户请求|用戶請求|先|再|然后|然後|并且|並且|"
    r"同时|同時|一下|吧|呢|啊|哦|嘛|呀|就行|即可|可以了|"
    r"谢谢|謝謝|谢了|謝了|辛苦了)"
)
_RECORD_ANALYSIS_TRANSLATION = str.maketrans({
    "請": "请",
    "刪": "删",
    "條": "条",
    "將": "将",
    "詞": "词",
    "錄": "录",
    "寫": "写",
    "標": "标",
    "記": "记",
    "轉": "转",
    "傳": "传",
    "歸": "归",
    "檔": "档",
    "備": "备",
    "註": "注",
})
_NEGATIVE_MODAL_RE = re.compile(
    r"(?:不要|别|不|无需|不用|禁止|不得|请勿|不予|严禁|不许|不可|不能|"
    r"切勿|拒绝|莫)"
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[。！？!?;；\n]")
_ACTION_TOKENS = {
    "Create": re.compile(r"添加|加入|加到|新增|创建|写入|放入|收录|录入|记入|加词"),
    "Change": re.compile(
        r"修改|改成|改为|替换|重新编码|顺延|挪开|"
        r"调整权重|修改权重|权重调整|权重修改|权重改|"
        r"放在|放到|排在|挪到|移到|提到|提前到|往前|往后|靠前|靠后"
    ),
    "Delete": re.compile(r"删除|删掉|删干净|移除"),
    "Keep": re.compile(
        r"只保留|仅保留|保留|留下|别动|不要动|"
        r"不动|别删|不要删|不删|留着"
    ),
}
_WORD_LEFT_PREFIXES = (
    "添加", "加入", "加到", "新增", "创建", "写入", "放入", "收录", "录入", "记入", "加词",
    "修改", "改成", "替换", "删除", "删掉", "移除", "保留", "只保留", "仅保留",
    "顺延", "放在", "放到", "排在", "移到", "挪到", "提到", "提前到", "改到",
    "往前", "往后", "靠前", "靠后",
    "把", "将", "词条", "词语", "提到的",
)
_WORD_RIGHT_SUFFIXES = (
    "编码", "代码", "改成", "改为", "修改为", "替换为", "加入", "添加",
    "加到草稿", "加入草稿", "删除", "删掉", "移除", "到草稿", "放入草稿",
    "顺延", "的编码", "的代码", "到", "放在", "放到", "排在", "移到", "挪到",
    "提到", "提前到", "改到", "往前", "往后", "靠前", "靠后",
    "和", "与", "及", "、", "都", "并", "以", "为",
)
# Machine-readable block reasons.  The model is only allowed to relay the
# structured reason plus ``suggestedCommand``; it must never invent a format.
BLOCK_REASON_SOURCE_UNTRUSTED = "source_untrusted"
BLOCK_REASON_VERB_NOT_MATCHED = "verb_not_matched"
BLOCK_REASON_BINDING_INCOMPLETE = "binding_incomplete"
BLOCK_REASON_TICKET_REQUIRED = "ticket_required"
BLOCK_REASON_BULK_DELETE_NOT_REQUESTED = "bulk_delete_not_requested"
BLOCK_REASON_MANUAL_SHIFT_FORBIDDEN = "manual_shift_forbidden"
BLOCK_REASON_ORDERING_NOT_EXPRESSIBLE = "ordering_not_expressible"
BLOCK_REASON_BATCH_TOO_LARGE = "batch_too_large"
BLOCK_REASON_UNTRUSTED_BATCH = "untrusted_batch_reference"
# Group chats drop any message that does not mention the bot, so every
# remediation command has to carry the mention that makes it deliverable.
SUGGESTION_MENTION_PREFIX = "@我 "
_MAX_SUGGESTED_BATCH_ITEMS = 6
_MAX_SUGGESTED_DELETE_IDS = 12
MUTATING_TOOL_NAMES = frozenset({
    "keytao_create_phrase",
    "keytao_remove_draft_item",
    "keytao_update_draft_item_weight",
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
    r"(?:添加|加入|收录|录入|记入|删除|删掉|移除|修改|替换|顺延|移动))"
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
_PHRASE_TYPE_BASE_WEIGHTS = {
    "Single": 10,
    "Phrase": 100,
    "Supplement": 100,
    "Symbol": 10,
    "Link": 10000,
    "CSS": 100,
    "CSSSingle": 10,
    "English": 100,
}


# A quoted span is treated as an operable command (not a dictionary entry) when
# it names a verb *and* something to act on: a draft/batch/item, a quantifier,
# or a code-like token.  Real entries are short and carry none of those.
_QUOTED_COMMAND_OPERAND_RE = re.compile(r"草稿|批次|条目|全部|所有|[A-Za-z0-9]{2,}")
_MAX_QUOTED_ENTRY_CHARS = 8


def _quoted_span_is_command(content: str) -> bool:
    if not (
        _MUTATION_INTENT_RE.search(content)
        or _DELETE_INTENT_RE.search(content)
    ):
        return False
    return bool(
        len(content) > _MAX_QUOTED_ENTRY_CHARS
        or _QUOTED_COMMAND_OPERAND_RE.search(content)
    )


def _whole_message_quote_content(message: str) -> Optional[str]:
    """Return one exact whole-message quote payload after envelope removal."""
    candidate = _WHOLE_MESSAGE_LEADING_ADDRESS_RE.sub(
        "",
        str(message or ""),
        count=1,
    ).strip()
    candidate = _WHOLE_MESSAGE_CLOSING_FILLER_RE.sub("", candidate).strip()
    for pattern in _WHOLE_MESSAGE_QUOTE_PATTERNS:
        match = pattern.fullmatch(candidate)
        if match is not None:
            return match.group("content")
    return None


def _whole_message_unquoted_source(message: str) -> Optional[str]:
    """Unwrap at most one exact command quote after envelope removal."""
    content = _whole_message_quote_content(message)
    if content is None:
        return None
    # The next layer remains quoted data. Refusing the outer unwrap here keeps
    # repeated authorization passes from peeling a second quote layer.
    if _whole_message_quote_content(content) is not None:
        return None
    return content


def trusted_mutation_source(message: str) -> str:
    """Preserve line structure while removing quoted or marked untrusted data."""
    text = str(message or "")
    unquoted = _whole_message_unquoted_source(text)
    if unquoted is not None:
        text = unquoted
    pieces: List[str] = []
    cursor = 0
    for match in _QUOTED_DATA_RE.finditer(text):
        pieces.append(text[cursor:match.start()])
        prefix = text[max(0, match.start() - 24):match.start()]
        # Redacting a span merely because its content is a verb would delete the
        # very dictionary entries this bot exists to edit ("保留", "提交", "顺延").
        # A span is untrusted when its frame says so, or when it spells out a
        # whole command rather than an entry.
        quoted_content = match.group(0)[1:-1]
        pieces.append(
            " " * (match.end() - match.start())
            if (
                _UNTRUSTED_QUOTE_PREFIX_RE.search(prefix)
                or _quoted_span_is_command(quoted_content)
            )
            else match.group(0)
        )
        cursor = match.end()
    pieces.append(text[cursor:])
    text = _INLINE_CODE_RE.sub("", "".join(pieces))
    return _UNTRUSTED_DATA_TAIL_RE.sub("", text)


def _has_standalone_negation_before_mutation(message: str) -> bool:
    """Bind an unambiguous standalone negation to the immediately next clause."""
    text = _LEADING_MENTION_RE.sub("", trusted_mutation_source(message), count=1)
    # A quoted negator is lexical data, not a standalone refusal.
    text = _QUOTED_DATA_RE.sub("", text)
    clauses = [
        re.sub(r"\s+", "", clause).strip()
        for clause in _COMMAND_CLAUSE_SPLIT_RE.split(text)
    ]
    clauses = [clause for clause in clauses if clause]
    for previous, current in zip(clauses, clauses[1:]):
        if (
            _STANDALONE_NEGATION_CLAUSE_RE.fullmatch(previous)
            and _MUTATION_INTENT_RE.search(current)
        ):
            return True
    return False


def _mutation_authorization_view(message: str) -> str:
    """Return only positive command clauses plus explicit protection clauses."""
    text = _LEADING_MENTION_RE.sub("", trusted_mutation_source(message), count=1)

    trusted_clauses: List[str] = []
    for clause in _COMMAND_CLAUSE_SPLIT_RE.split(text):
        # The clause is judged without whitespace (so prefixes and verbs keep
        # matching at position 0) but stored with token boundaries intact, so
        # downstream span/distance binding can still tell "吃席 wkxk" apart
        # from "吃席wkxk" instead of guessing.
        compact = re.sub(r"\s+", "", clause).strip()
        normalized = re.sub(r"\s+", " ", clause).strip()
        if not compact:
            continue
        candidate = _COMMAND_PREFIX_RE.sub("", compact, count=1)
        has_mutation = bool(_MUTATION_INTENT_RE.search(candidate))
        mutation_at_start = _MUTATION_INTENT_RE.match(candidate)
        positional_command = bool(
            not _POSITIONAL_SUBORDINATE_CONTEXT_RE.fullmatch(candidate)
            and _has_complete_positional_reorder_command(clause)
        )
        generic_ba_command = bool(
            re.match(
                rf"(?:把|将).{{1,80}}(?:{_NON_POSITIONAL_MUTATION_INTENT_PATTERN})",
                candidate,
            )
        )
        is_positive_command = bool(
            (
                mutation_at_start
                and not _POSITIONAL_REORDER_INTENT_RE.match(candidate)
            )
            or positional_command
            or generic_ba_command
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
            trusted_clauses.append(normalized)
    return "；".join(trusted_clauses)


def _has_mutation_instruction_shape(message: str) -> bool:
    """Recognize mutation syntax without deciding whether the user consents."""
    text = re.sub(r"\s+", "", _mutation_authorization_view(message))
    authorization_text = _QUOTED_DATA_RE.sub("", text)
    stripped_command_text = _COMMAND_PREFIX_RE.sub("", text, count=1)
    positional_command = _has_complete_positional_reorder_command(message)
    mutation_match = _MUTATION_INTENT_RE.search(authorization_text)
    if not authorization_text or (mutation_match is None and not positional_command):
        return False

    # The final checks here describe instruction shape, but the input view also
    # admits explicit protection clauses via _NEGATIVE_MODAL_RE and
    # _PROTECTED_WORD_RE.  That admission is not consent: the outer core still
    # rejects negation, questions, aborts, explanations, transforms and data
    # context independently.  Moving those consent checks out of this helper is
    # precautionary separation; the measured record-frame defect was limited
    # to standalone negation, not the other consent-predicate groups.
    stripped_text = _COMMAND_PREFIX_RE.sub("", authorization_text, count=1)
    stripped_match = _MUTATION_INTENT_RE.search(stripped_text)
    return bool(
        (
            stripped_match is not None
            and stripped_match.start() == 0
            and (
                not _POSITIONAL_REORDER_INTENT_RE.match(stripped_text)
                or positional_command
            )
        )
        or positional_command
        or (
            re.match(
                rf"(?:把|将).{{1,80}}(?:{_NON_POSITIONAL_MUTATION_INTENT_PATTERN})",
                stripped_command_text,
            )
        )
        or (
            _EXPLICIT_REQUEST_PREFIX_RE.match(authorization_text)
            and (
                not _POSITIONAL_REORDER_INTENT_RE.search(authorization_text)
                or positional_command
            )
        )
        or re.match(
            r"(?:草稿|批次)(?:中的)?(?:全部|都|所有)(?:条目)?"
            r"(?:删除|删掉|去掉|移除)",
            authorization_text,
        )
    )


def _message_authorizes_mutation_core(message: str) -> bool:
    """Judge mutation consent after higher-level data framing is excluded."""
    raw_text = re.sub(r"\s+", "", str(message or ""))
    text = re.sub(r"\s+", "", _mutation_authorization_view(message))
    authorization_clauses = [
        _QUOTED_DATA_RE.sub("", clause)
        for clause in _COMMAND_CLAUSE_SPLIT_RE.split(text)
    ]
    authorization_text = "；".join(
        clause for clause in authorization_clauses if clause
    )
    has_negated_mutation_clause = any(
        _NEGATED_NON_POSITIONAL_MUTATION_RE.search(clause)
        for clause in authorization_clauses
    )
    stripped_command_text = _COMMAND_PREFIX_RE.sub("", text, count=1)
    positional_command = _has_complete_positional_reorder_command(message)
    positional_scope = bool(
        positional_command
        or _raw_positional_destination_from_command(message) is not None
        or _has_raw_positional_relative_tail(message)
    )
    direct_command_after_lead_in = bool(
        (
            _MUTATION_INTENT_RE.match(stripped_command_text)
            and not _POSITIONAL_REORDER_INTENT_RE.match(stripped_command_text)
        )
        or positional_command
        or (
            re.match(
                rf"(?:把|将).{{1,80}}(?:{_NON_POSITIONAL_MUTATION_INTENT_PATTERN})",
                stripped_command_text,
            )
        )
    )
    question_is_execution_request = bool(
        _QUESTION_RE.search(raw_text)
        and (
            _POLITE_EXECUTION_PREFIX_RE.search(raw_text)
            or (
                re.match(r"^(?:能不能|可不可以|能否|可否)", raw_text)
                and direct_command_after_lead_in
            )
        )
        and not _EXPLANATION_ONLY_RE.search(raw_text)
        and not _META_DISCUSSION_RE.search(raw_text)
        and not _DATA_CONTEXT_RE.search(raw_text)
    )
    if (
        not authorization_text
        or _has_standalone_negation_before_mutation(message)
        or has_negated_mutation_clause
        or (
            positional_scope
            and (
                _NEGATED_POSITIONAL_REORDER_RE.search(raw_text)
                or _POSITIONAL_CONTEXT_NEGATION_RE.search(raw_text)
                or _POSITIONAL_BARE_DATA_CONTEXT_RE.search(raw_text)
                or _POSITIONAL_REPORTED_CONTEXT_RE.search(raw_text)
                or _POSITIONAL_REORDER_EXPLANATION_RE.search(raw_text)
                or _POSITIONAL_REORDER_LOCATIVE_DESTINATION_RE.search(raw_text)
                or _has_positional_choice_question(raw_text)
                or _POSITIONAL_REORDER_NARRATIVE_TAIL_RE.search(raw_text)
            )
        )
        or (_QUESTION_RE.search(raw_text) and not question_is_execution_request)
        or _ABORT_RE.search(raw_text)
    ):
        return False
    if (
        _META_DISCUSSION_RE.search(text)
        or _DATA_CONTEXT_RE.search(authorization_text)
        or re.search(r"(?:作为|设为).{0,8}(?:文章)?标题", authorization_text)
    ):
        return False
    if (
        _EXPLANATION_ONLY_RE.search(authorization_text)
        or _TEXT_TRANSFORM_RE.search(authorization_text)
    ):
        return False
    return _has_mutation_instruction_shape(message)


def _mask_quoted_record_frames(message: str) -> str:
    """Keep source offsets stable while hiding quoted data and inline code."""
    masked = list(message)
    for pattern in (
        _QUOTED_DATA_RE,
        _INLINE_CODE_RE,
        _RECORD_FRAME_BRACKETED_DATA_RE,
    ):
        for match in pattern.finditer(message):
            masked[match.start():match.end()] = " " * (match.end() - match.start())
    return "".join(masked)


def _normalize_positional_same_code_markers(
    message: str,
) -> Tuple[str, bool]:
    """Remove trusted same-code markers from their positional command clause.

    The returned text keeps source offsets stable. Quoted text, inline code,
    and record-shaped data are masked before marker matching, so none of them
    can opt a positional command into duplicate behavior. A marker in a
    separate clause is also left intact and cannot modify another clause.
    """
    trusted = trusted_mutation_source(str(message or ""))
    masked = _mask_quoted_record_frames(trusted)
    normalized = list(trusted)
    marker_found = False
    clause_start = 0
    boundaries = [
        (match.start(), match.end())
        for match in _COMMAND_CLAUSE_SPLIT_RE.finditer(masked)
    ]
    for clause_end, separator_end in [*boundaries, (len(masked), len(masked))]:
        masked_clause = masked[clause_start:clause_end]
        if _POSITIONAL_COMMAND_VERB_RE.search(masked_clause):
            for marker in _POSITIONAL_SAME_CODE_MARKER_RE.finditer(masked_clause):
                start = clause_start + marker.start()
                end = clause_start + marker.end()
                normalized[start:end] = " " * (end - start)
                marker_found = True
        clause_start = separator_end
    return "".join(normalized), marker_found


def _positional_same_code_requested(message: str) -> bool:
    """Return whether a complete positional command explicitly requests a duplicate."""
    normalized, marker_found = _normalize_positional_same_code_markers(message)
    return bool(
        marker_found
        and _has_complete_positional_reorder_command(normalized)
    )


def _record_frame_is_mutation_operand(
    masked_message: str,
    frame_match: re.Match,
) -> bool:
    """Return whether the record-shaped words are the dictionary entry itself."""
    prefix = re.sub(r"\s+", "", masked_message[:frame_match.start()])
    suffix = re.sub(r"\s+", "", masked_message[frame_match.end():])
    frame_text = re.sub(r"\s+", "", frame_match.group(0))

    # A label attached to quoted data ("引用…作为备注") describes that data;
    # it does not turn an otherwise genuine mutation into reported speech.
    if re.search(r"(?:作为|当作|用作)$", prefix) and not re.match(r"[：:]", suffix):
        return True

    # Prefix verbs take the following text as their entry operand:
    # "删除记下来", "添加词条 记录一下", "顺延记下来".
    if re.fullmatch(
        rf"{_COMMAND_PREFIX_PATTERN}"
        rf"(?:{_ENTRY_MUTATION_FOR_FRAME_OPERAND_RE.pattern})"
        r"(?:词条|词语)?(?:做|作|留)?(?:个|份)?[：:]?$",
        prefix,
    ):
        return True

    # Some frame spellings begin with a mutation verb themselves ("写入笔记").
    # They are a direct entry command only when the surrounding text is exactly
    # a legal command lead-in plus optional operand decoration/code.  A
    # separator or another instruction after the frame keeps it framing data.
    if (
        _MUTATION_INTENT_RE.match(frame_text)
        and re.fullmatch(_COMMAND_PREFIX_PATTERN, prefix)
        and _RECORD_FRAME_EMBEDDED_MUTATION_SUFFIX_RE.fullmatch(suffix)
    ):
        return True

    # The regex can also begin one character inside a preceding mutation verb:
    # in "保留记录" it prefers the frame "留记录".  Recover the complete
    # mutation token before deciding whether the remainder is a direct operand.
    for mutation_match in _MUTATION_INTENT_RE.finditer(masked_message):
        if not (
            mutation_match.start() < frame_match.start() < mutation_match.end()
        ):
            continue
        lead_in = re.sub(r"\s+", "", masked_message[:mutation_match.start()])
        if (
            re.fullmatch(_COMMAND_PREFIX_PATTERN, lead_in)
            and _RECORD_FRAME_EMBEDDED_MUTATION_SUFFIX_RE.fullmatch(suffix)
        ):
            return True

    # A draft container can sit between a prefix delete verb and the entry:
    # "删除草稿里的记录". Keep this anchored to the whole prefix so reported
    # instructions such as "请把这句删除...记录下来" cannot qualify.
    if _RECORD_FRAME_DRAFT_DELETE_PREFIX_RE.fullmatch(prefix):
        return True

    # In a 把/将 construction the operand precedes the action verb:
    # "把记下来加到草稿" and "把记录这个词加入草稿".
    if (
        re.search(r"(?:把|将|將)[：:]?$", prefix)
        and _RECORD_FRAME_OPERAND_AFTER_RE.match(suffix)
    ):
        return True

    # A postposed entry after a colon is still the object referred to by
    # "这个词"/"词条", rather than a request to record the earlier command.
    return bool(_RECORD_FRAME_POSTPOSED_OPERAND_RE.search(prefix))


def _has_complete_mutation_instruction(message: str) -> bool:
    """Require a mutation verb plus material that can serve as its operand."""
    raw_view = re.sub(r"\s+", "", trusted_mutation_source(message))
    raw_stripped = _COMMAND_PREFIX_RE.sub("", raw_view, count=1)
    if (
        _POSITIONAL_REORDER_INTENT_RE.match(raw_stripped)
        or _POSITIONAL_REORDER_RELATIVE_FRAGMENT_RE.fullmatch(raw_stripped)
    ):
        # Used only after a reporting frame has been removed.  A greedy
        # "传达给某人" frame can consume the subject and leave precisely this
        # verb + destination suffix behind.
        return True
    if not _has_mutation_instruction_shape(message):
        return False

    authorization_view = re.sub(r"\s+", "", _mutation_authorization_view(message))
    stripped_view = _COMMAND_PREFIX_RE.sub("", authorization_view, count=1)
    if (
        _POSITIONAL_REORDER_COMMAND_RE.fullmatch(stripped_view)
        # A trailing record frame can greedily consume the subject and the
        # start of the verb (for example "传达给管理员把吃席放在...").  The
        # remaining verb + destination is still enough to prove that the frame
        # wrapped a complete positional instruction; this only strengthens the
        # framing refusal and grants no direct authorization class.
        or _POSITIONAL_REORDER_INTENT_RE.match(stripped_view)
        or _POSITIONAL_REORDER_RELATIVE_FRAGMENT_RE.fullmatch(stripped_view)
    ):
        return True

    authorization_text = re.sub(r"\s+", "", authorization_view)
    for mutation_match in _MUTATION_INTENT_RE.finditer(authorization_text):
        residual = (
            authorization_text[:mutation_match.start()]
            + authorization_text[mutation_match.end():]
        )
        residual = _COMPLETE_MUTATION_NOISE_RE.sub("", residual)
        if re.sub(r"[\W_]+", "", residual, flags=re.UNICODE):
            return True
    return False


def _record_frame_wraps_complete_mutation(message: str) -> bool:
    """Detect an instruction being recorded/relayed instead of executed.

    A lexical record-frame match is not enough: the matched words may be the
    dictionary entry being edited.  Only block when removing a non-operand
    frame, under either the whole-frame or mutation-overlap interpretation,
    leaves an independently complete mutation instruction.
    """
    raw_message = str(message or "")
    # A relay frame with a bounded recipient is lexically ambiguous at its
    # right edge: ``传达给管理员把吃席往前`` can be greedily consumed through
    # ``往前`` by the frame's recipient slot.  Try every exact frame boundary
    # before the generic erase-and-recheck pass.  This only strengthens the
    # record-frame refusal and never creates a positive authorization path.
    for split_at in range(1, len(raw_message)):
        frame_prefix = raw_message[:split_at].rstrip()
        command_suffix = raw_message[split_at:].lstrip(
            "、：:—，。 \t\r\n"
        )
        if not command_suffix:
            continue
        compact_suffix = re.sub(r"\s+", "", command_suffix)
        stripped_suffix = _COMMAND_PREFIX_RE.sub("", compact_suffix, count=1)
        if (
            _RECORD_FRAME_RE.fullmatch(frame_prefix)
            and _POSITIONAL_REORDER_COMMAND_RE.fullmatch(stripped_suffix)
        ):
            return True

    masked_message = _mask_quoted_record_frames(raw_message)
    frame_matches = list(_RECORD_FRAME_RE.finditer(masked_message))
    framing_matches = [
        match
        for match in frame_matches
        if not _record_frame_is_mutation_operand(masked_message, match)
    ]
    if not framing_matches:
        return False

    mutation_spans = [
        match.span()
        for match in _MUTATION_INTENT_RE.finditer(masked_message)
    ]
    for frame_match in framing_matches:
        frame_start, frame_end = frame_match.span()
        for preserve_mutation_tokens in (False, True):
            candidate = list(raw_message)

            # First test the ordinary interpretation that removes the whole
            # frame.  Then test the overlap interpretation: "保留备忘" is
            # matched as "留备忘", so erasing the whole match would also erase
            # half of the real mutation token.  "写入笔记" is another overlap,
            # with a complete mutation token at the start of the frame.
            for index in range(frame_start, frame_end):
                mutation_overlap = any(
                    start <= index < end for start, end in mutation_spans
                )
                if not preserve_mutation_tokens or not mutation_overlap:
                    candidate[index] = " "

            # Separators belong to the reporting frame, not to the command
            # being tested after that frame is removed.  Consume every
            # product-supported boundary spelling on either side, including
            # the no-separator case.
            left = frame_start - 1
            while left >= 0 and raw_message[left] in _RECORD_FRAME_SEPARATOR_CHARS:
                candidate[left] = "；"
                left -= 1
            right = frame_end
            while right < len(raw_message) and raw_message[right] in _RECORD_FRAME_SEPARATOR_CHARS:
                candidate[right] = "；"
                right += 1

            normalized_candidate = "".join(candidate).translate(_RECORD_ANALYSIS_TRANSLATION)
            if _has_complete_mutation_instruction(normalized_candidate):
                return True
    return False


def message_authorizes_mutation(message: str) -> bool:
    """Accept write authority only from the current user's explicit raw text."""
    unquoted = _whole_message_unquoted_source(message)
    if unquoted is not None:
        message = unquoted
    raw_source = re.sub(r"\s+", "", trusted_mutation_source(message))
    # The authorization view intentionally keeps a later standalone command
    # clause after harmless chatter.  Report/record frames are not harmless:
    # once they lead the raw turn, punctuation must not launder the following
    # positional text into a fresh command.
    if (
        _DATA_CONTEXT_RE.search(raw_source)
        or _POSITIONAL_BARE_DATA_CONTEXT_RE.search(raw_source)
    ):
        return False
    if _record_frame_wraps_complete_mutation(message):
        return False
    if not _message_authorizes_mutation_core(message):
        return False
    multi_add = _multi_add_authorization_contract(message)
    return multi_add is None or multi_add.valid


# Verbs that express "change where this code sits" in everyday Chinese.  Most
# remain helpfulness-only because they are too common to grant write authority
# ("放到明天再说", "调到静音模式").  The narrower positional subset above is
# promoted only when the entire dictionary reorder grammar matches.
_POSITIONAL_CHANGE_RE = re.compile(
    # Deliberately excludes bare position words ("占用|提前|前面|后面|位置"):
    # they carry no request by themselves and appear constantly in small talk.
    r"放在|放到|调到|调整到|挪到|挪开|排在|插到|插入|抢占|移到|改到|"
    r"提到|提前到|往前|往后|靠前|靠后"
)
_CHANGE_VERB_RE = re.compile(
    r"修改|改成|改为|改到|替换|重新编码|顺延|挪开|移到|"
    r"调整权重|修改权重|权重调整|权重修改|权重改|"
    + _POSITIONAL_CHANGE_RE.pattern
)
_WEIGHT_ADJUST_VERB_RE = re.compile(
    r"调整权重|修改权重|权重(?:调整|修改|改)(?:为|到)?"
)
_CREATE_VERB_RE = re.compile(r"添加|加入|加到|新增|创建|写入|放入|收录|录入|记入|加词")
_DELETE_VERB_RE = re.compile(r"删除|删掉|删干净|去掉|移除|清空|清理")
_SUBMIT_VERB_RE = re.compile(r"提交|提审|送审|发起审核")
_RECALL_VERB_RE = re.compile(r"撤回|撤销|召回|取消")
_TOOL_INTENT_PATTERNS = {
    "keytao_shift_phrase_code": _CHANGE_VERB_RE,
    "keytao_remove_draft_item": _DELETE_VERB_RE,
    "keytao_update_draft_item_weight": _WEIGHT_ADJUST_VERB_RE,
    "keytao_batch_remove_draft_items": _DELETE_VERB_RE,
    "keytao_submit_batch": _SUBMIT_VERB_RE,
    "keytao_recall_batch": _RECALL_VERB_RE,
}


def _tool_intent_pattern(tool_name: str, arguments: Dict) -> Optional[re.Pattern]:
    """Which verb the user must have used for this tool to be what they meant."""
    if tool_name in _TOOL_INTENT_PATTERNS:
        return _TOOL_INTENT_PATTERNS[tool_name]
    items = (
        [arguments]
        if tool_name == "keytao_create_phrase"
        else arguments.get("items") if isinstance(arguments, dict) else None
    )
    if not isinstance(items, list) or not items:
        return None
    actions = {
        str(item.get("action") or "Create")
        for item in items
        if isinstance(item, dict)
    }
    patterns = {
        "Create": _CREATE_VERB_RE,
        "Change": _CHANGE_VERB_RE,
        "Delete": _DELETE_VERB_RE,
    }
    if len(actions) != 1:
        return None
    return patterns.get(next(iter(actions)))


def message_requests_change(
    message: str,
    tool_name: str,
    arguments: Dict,
) -> bool:
    """Report whether this message itself asks for this kind of change.

    This is a *helpfulness* gate, never an authorization gate: it decides
    whether the bot may hand the user a ready-to-send command.  It is looser
    than ``message_authorizes_mutation`` on purpose (it accepts the positional
    verbs that must never grant authority), and stricter in one way that
    matters: a question, an explanation, a negation or an abort never counts.
    """
    if not isinstance(arguments, dict):
        return False
    unquoted = _whole_message_unquoted_source(message)
    if unquoted is not None:
        message = unquoted
    pattern = _tool_intent_pattern(tool_name, arguments)
    if pattern is None:
        return False
    source_text = trusted_mutation_source(message)
    text = re.sub(r"\s+", "", source_text)
    if not text:
        return False
    positional_command = _has_complete_positional_reorder_command(message)
    positional_scope = bool(
        positional_command
        or _raw_positional_destination_from_command(message) is not None
        or _has_raw_positional_relative_tail(message)
    )
    authorization_clauses = [
        _QUOTED_DATA_RE.sub("", re.sub(r"\s+", "", clause))
        for clause in _COMMAND_CLAUSE_SPLIT_RE.split(source_text)
    ]
    authorization_text = "；".join(
        clause for clause in authorization_clauses if clause
    )
    has_negated_mutation_clause = any(
        _NEGATED_NON_POSITIONAL_MUTATION_RE.search(clause)
        for clause in authorization_clauses
    )
    if (
        _EXPLANATION_ONLY_RE.search(text)
        or _TEXT_TRANSFORM_RE.search(text)
        or _META_DISCUSSION_RE.search(text)
        or _DATA_CONTEXT_RE.search(text)
        or _record_frame_wraps_complete_mutation(message)
        or _ABORT_RE.search(text)
        or _has_standalone_negation_before_mutation(message)
        or has_negated_mutation_clause
        or (
            positional_scope
            and (
                _NEGATED_POSITIONAL_REORDER_RE.search(text)
                or _POSITIONAL_CONTEXT_NEGATION_RE.search(text)
                or _POSITIONAL_BARE_DATA_CONTEXT_RE.search(text)
                or _POSITIONAL_REPORTED_CONTEXT_RE.search(text)
                or _POSITIONAL_REORDER_EXPLANATION_RE.search(text)
                or _POSITIONAL_REORDER_LOCATIVE_DESTINATION_RE.search(text)
                or _has_positional_choice_question(text)
                or _POSITIONAL_REORDER_NARRATIVE_TAIL_RE.search(text)
            )
        )
    ):
        return False
    if (
        tool_name == "keytao_shift_phrase_code"
        and (
            _POSITIONAL_REPORTED_CONTEXT_RE.search(text)
            or _POSITIONAL_BARE_DATA_CONTEXT_RE.search(text)
            or _POSITIONAL_CONTEXT_NEGATION_RE.search(text)
            or _POSITIONAL_REORDER_EXPLANATION_RE.search(text)
            or (
                (
                    _POSITIONAL_REORDER_NARRATIVE_TAIL_RE.search(text)
                    or _has_positional_choice_question(text)
                )
                and re.search(
                    r"放在|放到|排在|挪到|移到|提到|提前到|"
                    r"往前|往后|靠前|靠后",
                    text,
                )
            )
            or (
                _POSITIONAL_REORDER_INTENT_RE.search(text)
                and not positional_command
            )
        )
    ):
        return False
    raw_text = re.sub(r"\s+", "", str(message or ""))
    stripped_text = _COMMAND_PREFIX_RE.sub("", text, count=1)
    direct_command_after_lead_in = bool(
        (
            _MUTATION_INTENT_RE.match(stripped_text)
            and not _POSITIONAL_REORDER_INTENT_RE.match(stripped_text)
        )
        or positional_command
        or (
            re.match(
                rf"(?:把|将).{{1,80}}(?:{_NON_POSITIONAL_MUTATION_INTENT_PATTERN})",
                stripped_text,
            )
        )
    )
    question_is_execution_request = bool(
        _POLITE_EXECUTION_PREFIX_RE.search(raw_text)
        or (
            re.match(r"^(?:能不能|可不可以|能否|可否)", raw_text)
            and direct_command_after_lead_in
        )
    )
    if _QUESTION_RE.search(raw_text) and not question_is_execution_request:
        return False
    return bool(pattern.search(authorization_text))


def _suggestion_operands(tool_name: str, arguments: Dict) -> List[str]:
    """The entities a suggestion would name back at the user."""
    if not isinstance(arguments, dict):
        return []
    if tool_name in {"keytao_submit_batch", "keytao_recall_batch"}:
        return []
    if tool_name == "keytao_shift_phrase_code":
        return [
            str(arguments.get("word") or "").strip(),
            str(arguments.get("target_code") or "").strip(),
        ]
    if tool_name == "keytao_remove_draft_item":
        return [str(arguments.get("pr_id") or "").strip()]
    if tool_name == "keytao_update_draft_item_weight":
        weight = arguments.get("weight")
        return [
            str(arguments.get("word") or "").strip(),
            str(arguments.get("code") or "").strip(),
            str(weight if weight is not None else "").strip(),
        ]
    if tool_name == "keytao_batch_remove_draft_items":
        ids = arguments.get("ids")
        return [str(item).strip() for item in ids] if isinstance(ids, list) else [""]
    items = (
        [arguments]
        if tool_name == "keytao_create_phrase"
        else arguments.get("items")
    )
    if not isinstance(items, list) or not items:
        return [""]
    operands: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            return [""]
        operands.append(str(item.get("word") or "").strip())
        code = str(item.get("code") or "").strip()
        if code:
            operands.append(code)
        old_word = str(item.get("old_word") or "").strip()
        if old_word:
            operands.append(old_word)
    return operands


def _operands_are_present(message: str, tool_name: str, arguments: Dict) -> bool:
    """Require the user to have named every entity we would hand back.

    Without this, the verb alone decides: a user who says "把吃席的编码放在赤溪前面"
    would get back a ready-to-send command for whatever word and code the model
    happened to propose - which is exactly the injection path the suggestion
    mechanism must not open.
    """
    operands = _suggestion_operands(tool_name, arguments)
    return all(
        operand and _contains_exact_target(message, operand)
        for operand in operands
    )


def message_mentions_change_request(message: str) -> bool:
    """Cheap pre-check: could this message be asking for any kind of change?"""
    return any(
        message_requests_change(message, tool_name, arguments)
        for tool_name, arguments in (
            ("keytao_shift_phrase_code", {"word": "x", "target_code": "y"}),
            ("keytao_create_phrase", {"word": "x", "code": "y"}),
            ("keytao_remove_draft_item", {"pr_id": "1"}),
            (
                "keytao_update_draft_item_weight",
                {"word": "x", "code": "y", "weight": 100},
            ),
            ("keytao_submit_batch", {}),
            ("keytao_recall_batch", {}),
        )
    )


def _type_hint_label(phrase_type: str) -> str:
    """Return the Chinese hint a user can type to pin this phrase type."""
    for hint, value in _TYPE_HINTS:
        if value == phrase_type and _is_han(hint[:1]):
            return hint
    return ""


def _suggested_item_command(item: Dict) -> str:
    word = str(item.get("word") or "").strip()
    code = str(item.get("code") or "").strip()
    action = str(item.get("action") or "Create").strip() or "Create"
    old_word = str(item.get("old_word") or "").strip()
    label = _type_hint_label(str(item.get("type") or "").strip())
    if not word:
        return ""
    if action == "Create":
        return f"添加{label}「{word}」 {code}" if code else ""
    if action == "Delete":
        return f"删除{label}「{word}」 {code}".rstrip() if label else ""
    if action == "Change":
        if not old_word or not code or not label:
            return ""
        return f"把{label}「{old_word}」改成「{word}」 {code}"
    return ""


def _suggested_command_text(tool_name: str, arguments: Dict) -> str:
    """Compose the command a user could send to authorize this exact call."""
    if not isinstance(arguments, dict):
        return ""
    if tool_name == "keytao_shift_phrase_code":
        word = str(arguments.get("word") or "").strip()
        code = str(arguments.get("target_code") or "").strip()
        return f"顺延「{word}」到 {code}" if word and code else ""
    if tool_name == "keytao_create_phrase":
        return _suggested_item_command(arguments)
    if tool_name == "keytao_batch_add_to_draft":
        items = arguments.get("items")
        if (
            not isinstance(items, list)
            or not items
            or len(items) > _MAX_SUGGESTED_BATCH_ITEMS
            or not all(isinstance(item, dict) for item in items)
        ):
            return ""
        parts = [_suggested_item_command(item) for item in items]
        return "；".join(parts) if all(parts) else ""
    if tool_name == "keytao_remove_draft_item":
        pr_id = str(arguments.get("pr_id") or "").strip()
        return f"删除草稿条目 {pr_id}" if pr_id else ""
    if tool_name == "keytao_update_draft_item_weight":
        word = str(arguments.get("word") or "").strip()
        code = str(arguments.get("code") or "").strip()
        weight = arguments.get("weight")
        if (
            not word
            or not code
            or not isinstance(weight, int)
            or isinstance(weight, bool)
        ):
            return ""
        return f"将草稿中「{word}」{code} 的权重调整为 {weight}"
    if tool_name == "keytao_batch_remove_draft_items":
        ids = arguments.get("ids")
        if (
            not isinstance(ids, list)
            or not ids
            or len(ids) > _MAX_SUGGESTED_DELETE_IDS
        ):
            return ""
        joined = " ".join(str(item).strip() for item in ids)
        return f"删除草稿条目 {joined}" if joined.strip() else ""
    if tool_name == "keytao_submit_batch":
        return "提交草稿"
    if tool_name == "keytao_recall_batch":
        return "撤回提审"
    return ""


def self_checked_suggested_command(
    tool_name: str,
    arguments: Dict,
    context: "ToolContext",
) -> str:
    """Return a remediation command only if it passes the real validators.

    The bot used to invent a new "correct format" on every rejection, none of
    which the validator would have accepted.  A suggestion is now replayed
    through the same authorization and binding checks it will face when the
    user actually sends it; if it does not pass, no suggestion is offered.

    A suggestion is a ready-made authorization, so it is only ever handed to a
    user who asked for this kind of change in this very message.  Someone who
    merely asked a question - or whose turn only carries an injected proposal
    from memory, a quote or an attachment - gets the reason and nothing else.
    """
    raw_message = context.current_message or ""
    if not message_requests_change(raw_message, tool_name, arguments):
        return ""
    if not _operands_are_present(raw_message, tool_name, arguments):
        return ""
    if (
        tool_name == "keytao_shift_phrase_code"
        and _raw_positional_destination_from_command(raw_message) is not None
    ):
        word = str(arguments.get("word") or "").strip()
        target_code = str(arguments.get("target_code") or "").strip()
        trusted_codes = context.trusted_codes_by_word or {}
        if (
            target_code not in trusted_codes.get(word, frozenset())
            and not _positional_message_explicitly_labels_code(
                raw_message,
                word,
                target_code,
            )
        ):
            return ""
    candidate = _suggested_command_text(tool_name, arguments)
    if not candidate:
        return ""
    display = SUGGESTION_MENTION_PREFIX + candidate
    if not message_authorizes_mutation(display):
        return ""
    strict = ToolContext(
        platform=context.platform,
        user_id=context.user_id,
        current_message=display,
        writes_allowed=True,
    )
    if ToolExecutor._validate_current_message_binding(
        tool_name, arguments, strict
    ) is None:
        return display
    live = replace(context, current_message=display, writes_allowed=True)
    if ToolExecutor._validate_current_message_binding(
        tool_name, arguments, live
    ) is None:
        return display
    return ""


def policy_block(
    reason: str,
    message: str,
    *,
    missing: Optional[List[str]] = None,
    suggestion: str = "",
    **extra: Any,
) -> Dict:
    """Return a machine-readable rejection instead of free-form prose."""
    payload: Dict[str, Any] = {
        "success": False,
        "policyBlocked": True,
        "requiresTextFollowUp": True,
        "blockReason": reason,
        "message": message,
    }
    if missing:
        payload["missing"] = list(missing)
    if suggestion:
        payload["suggestedCommand"] = suggestion
        payload["message"] = (
            f"{message}请把下面这条指令原样转述给用户，不要自创格式：{suggestion}"
        )
    payload.update(extra)
    if reason == BLOCK_REASON_SOURCE_UNTRUSTED:
        review_flags.apply_review_disposition(
            payload,
            "injection_shaped_input",
        )
    return payload


def text_follow_up(reason: str, message: str, **extra: Any) -> Dict:
    """Return a non-policy clarification request without touching a sink."""
    payload: Dict[str, Any] = {
        "success": False,
        "requiresTextFollowUp": True,
        "reason": reason,
        "message": message,
    }
    payload.update(extra)
    return payload


_AUTO_CONFIRM_CREATE_WARNING_SITES = {
    "duplicate_code": "duplicate_formation",
    "code_chain_priority": "code_chain_priority",
}
_AUTO_CONFIRM_CREATE_WARNING_TYPES = frozenset(
    warning_type
    for warning_type, site in _AUTO_CONFIRM_CREATE_WARNING_SITES.items()
    if review_flags.review_disposition_for_site(site)
    is review_flags.ReviewDisposition.SEAL
)


def server_warning_confirmation_binding(
    preview: Dict,
) -> Optional[Dict[str, Any]]:
    """Return the CAS fields carried by one complete server warning ticket."""
    if not isinstance(preview, dict):
        return None
    content_version = preview.get("contentVersion")
    batch_id = str(preview.get("batchId") or "").strip()
    warning_digest = str(preview.get("warningDigest") or "").strip().lower()
    warnings = preview.get("warnings")
    warned_count = preview.get("warnedCount")
    conflict_markers = (
        "staleConfirmation",
        "contentVersionConflict",
        "batchStateChanged",
        "uncertain",
    )
    if (
        preview.get("success") is not False
        or preview.get("requiresConfirmation") is not True
        or not batch_id
        or not isinstance(content_version, int)
        or isinstance(content_version, bool)
        or content_version < 0
        or not re.fullmatch(r"[0-9a-f]{64}", warning_digest)
        or not isinstance(warnings, list)
        or (
            warned_count is not None
            and (
                not isinstance(warned_count, int)
                or isinstance(warned_count, bool)
                or warned_count != len(warnings)
            )
        )
        or any(preview.get(marker) for marker in conflict_markers)
        or bool(preview.get("conflicts"))
        or bool(preview.get("failed"))
        or bool(preview.get("failedCount"))
    ):
        return None
    return {
        "confirmed": True,
        "batch_id": batch_id,
        "expected_content_version": content_version,
        "expected_warning_digest": warning_digest,
    }


def create_warning_confirmation_binding(
    preview: Dict,
    arguments: Dict,
) -> Optional[Dict[str, Any]]:
    """Return an exact replay ticket for a clean or informational Create."""
    if not isinstance(preview, dict) or not isinstance(arguments, dict):
        return None
    word = str(arguments.get("word") or "").strip()
    code = str(arguments.get("code") or "").strip().lower()
    action = str(arguments.get("action") or "Create").strip()
    binding = server_warning_confirmation_binding(preview)
    warnings = preview.get("warnings")
    if (
        action != "Create"
        or not word
        or not code
        or binding is None
    ):
        return None
    for warning in warnings:
        if not isinstance(warning, dict):
            return None
        warning_type = str(warning.get("warningType") or "").strip()
        if warning_type not in _AUTO_CONFIRM_CREATE_WARNING_TYPES:
            return None
        item = warning.get("item")
        source = item if isinstance(item, dict) else warning
        warning_word = str(source.get("word") or "").strip()
        warning_code = str(source.get("code") or "").strip().lower()
        warning_action = str(source.get("action") or "Create").strip()
        if (
            warning_word != word
            or warning_code != code
            or warning_action != "Create"
        ):
            return None
    return binding


def batch_warning_confirmation_binding(
    preview: Dict,
    arguments: Dict,
) -> Optional[Dict[str, Any]]:
    """Return an exact replay ticket for a bound clean/informational batch."""
    if not isinstance(arguments, dict):
        return None
    binding = server_warning_confirmation_binding(preview)
    items = arguments.get("items")
    if binding is None or not isinstance(items, list) or not items:
        return None
    exact_items: set[Tuple[str, str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            return None
        action = str(item.get("action") or "Create").strip()
        word = str(item.get("word") or "").strip()
        code = str(item.get("code") or "").strip().lower()
        identity = (action, word, code)
        if action != "Create" or not word or not code or identity in exact_items:
            return None
        exact_items.add(identity)
    for warning in preview.get("warnings") or []:
        if not isinstance(warning, dict):
            return None
        warning_type = str(warning.get("warningType") or "").strip()
        if warning_type not in _AUTO_CONFIRM_CREATE_WARNING_TYPES:
            return None
        item = warning.get("item")
        source = item if isinstance(item, dict) else warning
        warning_identity = (
            str(source.get("action") or "Create").strip(),
            str(source.get("word") or "").strip(),
            str(source.get("code") or "").strip().lower(),
        )
        if warning_identity not in exact_items:
            return None
    return binding


def front_insert_batch_warning_confirmation_binding(
    preview: Dict,
    authorization_arguments: Dict,
    execution_route: ToolExecutionRoute,
) -> Optional[Dict[str, Any]]:
    """Bind the narrow same-code front insert already named by the user.

    Only the destination occupant may be bumped automatically. A later entry
    in the same weight chain is a real additional displacement, even when the
    +1 cascade makes it mechanically necessary, so that wider plan remains an
    explicit-confirmation operation.
    """
    if (
        not isinstance(authorization_arguments, dict)
        or execution_route.tool_name != "keytao_batch_add_to_draft"
        or execution_route.positional_binding is None
    ):
        return None
    binding = server_warning_confirmation_binding(preview)
    positional = execution_route.positional_binding
    newcomer = str(authorization_arguments.get("word") or "").strip()
    code = str(authorization_arguments.get("code") or "").strip().lower()
    action = str(authorization_arguments.get("action") or "Create").strip()
    items = execution_route.arguments.get("items")
    if (
        binding is None
        or action != "Create"
        or not newcomer
        or code != positional.code
        or positional.relation not in _POSITIONAL_CREATE_FRONT_RELATIONS
        or not isinstance(items, list)
        or len(items) != 2
        or len(positional.bumped_entries) != 1
    ):
        return None
    bumped_word, old_weight, new_weight = positional.bumped_entries[0]
    if (
        bumped_word != positional.destination_word
        or not isinstance(old_weight, int)
        or isinstance(old_weight, bool)
        or not isinstance(new_weight, int)
        or isinstance(new_weight, bool)
        or new_weight != old_weight + 1
        or positional.weight != old_weight
    ):
        return None

    create_items = [
        item for item in items
        if isinstance(item, dict)
        and str(item.get("action") or "Create").strip() == "Create"
    ]
    change_items = [
        item for item in items
        if isinstance(item, dict)
        and str(item.get("action") or "Create").strip() == "Change"
    ]
    if len(create_items) != 1 or len(change_items) != 1:
        return None
    create_item = create_items[0]
    change_item = change_items[0]
    if (
        str(create_item.get("word") or "").strip() != newcomer
        or str(create_item.get("code") or "").strip().lower() != positional.code
        or str(create_item.get("type") or "").strip() != positional.phrase_type
        or create_item.get("weight") != positional.weight
        or str(change_item.get("old_word") or "").strip()
        != positional.destination_word
        or str(change_item.get("word") or "").strip()
        != positional.destination_word
        or str(change_item.get("code") or "").strip().lower() != positional.code
        or str(change_item.get("type") or "").strip() != positional.phrase_type
        or change_item.get("weight") != new_weight
    ):
        return None

    for warning in preview.get("warnings") or []:
        if not isinstance(warning, dict):
            return None
        warning_type = str(warning.get("warningType") or "").strip()
        item = warning.get("item")
        source = item if isinstance(item, dict) else warning
        if (
            warning_type not in _AUTO_CONFIRM_CREATE_WARNING_TYPES
            or str(source.get("action") or "Create").strip() != "Create"
            or str(source.get("word") or "").strip() != newcomer
            or str(source.get("code") or "").strip().lower() != positional.code
        ):
            return None
    return binding


def _strip_execution_result_suffix(compact: str) -> str:
    compact = re.sub(r"(?:吗|么|好不好|行不行|可不可以|可以吗|好吗|行吗)$", "", compact)
    return re.sub(
        r"(?:(?:并|并且|然后|再|完成后|处理完后|操作完后)?"
        r"(?:告诉我|回复我|通知我)(?:一下)?(?:处理)?(?:结果)?"
        r"|(?:并|并且|然后|再)?(?:告诉我|回复我|通知我)(?:一下)?)$",
        "",
        compact,
    )


def _explicit_submit_command_matches(compact: str) -> bool:
    compact = _strip_execution_result_suffix(compact)
    prefix = (
        r"(?:请|麻烦|帮我|给我|现在|立即|直接|确认|我要|我想|替我|为我|"
        r"能不能|可不可以|能否|可否|可以帮我|可以请你)*"
    )
    target = r"(?:(?:当前|这个|我的)?(?:草稿|批次))"
    action = r"(?:提交|提审|送审)(?:审核)?"
    polite = r"(?:一下)?(?:吧|啦|了)?"
    return bool(
        re.fullmatch(rf"{prefix}(?:{action}(?:{target})?|(?:把|将)?{target}{action}|发起审核){polite}", compact)
    )


def _explicit_recall_command_matches(compact: str) -> bool:
    compact = _strip_execution_result_suffix(compact)
    prefix = (
        r"(?:请|麻烦|帮我|给我|现在|立即|直接|确认|我要|我想|替我|为我|"
        r"能不能|可不可以|能否|可否|可以帮我|可以请你)*"
    )
    scope = r"(?:(?:最近|上次|刚才)(?:一次|的)?)?"
    recall = (
        rf"(?:(?:撤回|撤销|召回)(?:"
        rf"{scope}(?:提交|提审|送审|审核|批次)?|"
        rf"{scope}(?:提交|提审|送审)(?:的)?批次)"
        rf"|取消{scope}(?:提审|送审))"
    )
    return bool(re.fullmatch(rf"{prefix}{recall}(?:一下)?(?:吧|啦|了)?", compact))


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


def _quoted_target_spans(message: str, target: str) -> List[tuple[int, int]]:
    """Return only the occurrences the user explicitly cited as an entry."""
    if not target:
        return []
    spans: List[tuple[int, int]] = []
    for opening, closing in (("「", "」"), ("“", "”"), ("‘", "’"), ('"', '"'), ("'", "'")):
        quoted = f"{opening}{target}{closing}"
        start = 0
        while True:
            index = message.find(quoted, start)
            if index < 0:
                break
            spans.append((index + len(opening), index + len(opening) + len(target)))
            start = index + len(quoted)
    return sorted(set(spans))


def _has_protection_outside_target(message: str, target: str) -> bool:
    """Ignore only the protection word the user cited as the entry itself.

    A quoted 「保留」 is the entry being moved; a bare "保留原编码" in the same
    message is still the user protecting something and must keep blocking.
    """
    spans = _quoted_target_spans(message, target)
    for match in re.finditer(_PROTECTED_WORD_RE, message):
        if any(
            match.start() < end and start < match.end()
            for start, end in spans
        ):
            continue
        return True
    return False


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
                    # The token is part of the target itself ("删除"/"保留" are
                    # legitimate dictionary entries).  A word cannot be its own
                    # verb, so it must not win the nearest-action vote.
                    continue
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
        # Only a token that overlaps the target is part of the target.  A code
        # written immediately next to it ("顺延：吃席 wkxk") is the whole point
        # of the instruction and must stay bindable.
        if span[0] < target_span[1] and target_span[0] < span[1]:
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


def _target_clause_has_explicit_code_token(message: str, target: str) -> bool:
    """Report whether the target clause already supplies any code candidate."""
    for target_span in _exact_target_spans(message, target):
        clause_start, clause_end = _clause_bounds(message, *target_span)
        if _explicit_code_spans(
            message,
            clause_start,
            clause_end,
            target_span,
        ):
            return True
    return False


def _server_knows_positional_entry(
    context: ToolContext,
    target: str,
) -> bool:
    if not target:
        return False
    if target in (context.trusted_draft_words_by_id or {}).values():
        return True
    if any(
        word == target
        for word, _code in (context.trusted_phrase_types_by_key or {})
    ):
        return True
    return any(
        str(item.get("word") or "").strip() == target
        for item in (context.trusted_draft_items_by_id or {}).values()
        if isinstance(item, dict)
    )


def _positional_create_order(
    entries: Tuple[Tuple[str, int], ...],
    word: str,
    destination_word: str,
    relation: str,
    base_weight: int,
) -> Tuple[
    Optional[int],
    Tuple[str, ...],
    Tuple[Tuple[str, int, int], ...],
]:
    """Plan an exact same-code ordering without ever going below type base."""
    # Ordering authority: keytao-next/lib/services/batchPriorityOrderWarnings.ts
    # lines 30-34 and 73-82 sort by ascending weight and name index + 1
    # as the entry behind the current one. Lower weights therefore rank first.
    normalized: List[Tuple[str, int]] = []
    seen_entries: set[Tuple[str, int]] = set()
    for entry_word, entry_weight in entries:
        clean_word = str(entry_word or "").strip()
        if (
            not clean_word
            or not isinstance(entry_weight, int)
            or isinstance(entry_weight, bool)
            or entry_weight < base_weight
        ):
            return None, (), ()
        entry = (clean_word, entry_weight)
        if entry not in seen_entries:
            normalized.append(entry)
            seen_entries.add(entry)
    if not normalized:
        return None, (), ()
    weights = [entry_weight for _entry_word, entry_weight in normalized]
    if len(set(weights)) != len(weights):
        return None, (), ()
    normalized.sort(key=lambda item: item[1])
    destinations = [
        index
        for index, (entry_word, _entry_weight) in enumerate(normalized)
        if entry_word == destination_word
    ]
    if len(destinations) != 1:
        return None, (), ()
    destination_index = destinations[0]
    destination_weight = normalized[destination_index][1]
    if relation in _POSITIONAL_CREATE_FRONT_RELATIONS:
        requested_weight = destination_weight
        insert_index = destination_index
        bumped_entries = tuple(
            (entry_word, entry_weight, entry_weight + 1)
            for entry_word, entry_weight in normalized
            if entry_weight >= destination_weight
        )
    elif relation in _POSITIONAL_CREATE_BACK_RELATIONS:
        next_weight = (
            normalized[destination_index + 1][1]
            if destination_index + 1 < len(normalized)
            else None
        )
        requested_weight = destination_weight + 1
        insert_index = destination_index + 1
        if next_weight is not None and requested_weight >= next_weight:
            return None, (), ()
        bumped_entries = ()
    else:
        return None, (), ()
    resulting = [entry_word for entry_word, _entry_weight in normalized]
    resulting.insert(insert_index, word)
    return requested_weight, tuple(resulting), bumped_entries


def _positional_message_explicitly_labels_code(
    message: str,
    word: str,
    target_code: str,
) -> bool:
    return bool(
        word
        and target_code
        and re.search(
            rf"{re.escape(word)}\s*的(?:编码|代码)\s*"
            rf"(?:放在|放到|排在|挪到|移到|提到|提前到)\s*"
            rf"{re.escape(target_code)}(?:\s|$|[。.!！?？])",
            message,
        )
    )


def _positional_phrase_type(
    arguments: Dict,
    context: ToolContext,
    word: str,
    code: str,
    destination_word: str,
) -> str:
    """Resolve one server-backed type for a same-code weight chain."""
    explicit = str(arguments.get("type") or "").strip()
    if explicit in _PHRASE_TYPES:
        return explicit
    reviewed = (context.trusted_reviewed_items_by_key or {}).get((word, code))
    reviewed_type = str((reviewed or {}).get("type") or "").strip()
    if reviewed_type in _PHRASE_TYPES:
        return reviewed_type
    destination_types = (
        context.trusted_phrase_types_by_key or {}
    ).get((destination_word, code), frozenset())
    normalized = {value for value in destination_types if value in _PHRASE_TYPES}
    return next(iter(normalized)) if len(normalized) == 1 else "Phrase"


def _same_type_positional_entries(
    entries: Tuple[Tuple[str, int], ...],
    code: str,
    phrase_type: str,
    context: ToolContext,
) -> Tuple[Tuple[str, int], ...]:
    """Keep only entries proven to belong to this type; retain unknown legacy data."""
    selected: List[Tuple[str, int]] = []
    for entry_word, entry_weight in entries:
        known_types = (
            context.trusted_phrase_types_by_key or {}
        ).get((entry_word, code), frozenset())
        normalized = {value for value in known_types if value in _PHRASE_TYPES}
        if not normalized or normalized == {phrase_type}:
            selected.append((entry_word, entry_weight))
    return tuple(selected)


def _pending_positional_create_binding(
    message: str,
    arguments: Dict,
    context: ToolContext,
) -> Optional[_PositionalCreateBinding]:
    """Authorize one create call from an exact live candidate conjunction."""
    capability = context.pending_candidate
    if capability is None or not capability.state_matches:
        return None

    operands = _positional_create_operands(message)
    if operands is None:
        return None
    subject, destination_word, relation = operands
    word = str(arguments.get("word") or "").strip()
    code = str(arguments.get("code") or "").strip().lower()
    action = str(arguments.get("action") or "Create").strip()
    old_word = str(arguments.get("old_word") or "").strip()

    if (
        action != "Create"
        or old_word
        or not word
        or subject != word
        or word != capability.word
    ):
        return None

    candidates = dict(capability.candidates)
    if (
        len(candidates) != len(capability.candidates)
        or code not in candidates
        or candidates[code] is not True
    ):
        return None

    occupied_words = dict(capability.occupied_words)
    destination_codes = [
        candidate_code
        for candidate_code, words in capability.occupied_words
        if destination_word in words
    ]
    if (
        len(occupied_words) != len(capability.occupied_words)
        or destination_codes != [code]
    ):
        return None

    if relation not in {"前面", "后面", "之前", "之后", "前", "后"}:
        return None

    raw_entries = tuple(
        (entry_word, entry_weight)
        for entry_code, entry_word, entry_weight in capability.entries
        if entry_code == code
    )
    phrase_type = _positional_phrase_type(
        arguments, context, word, code, destination_word
    )
    entries = _same_type_positional_entries(
        raw_entries, code, phrase_type, context
    )
    weight, resulting_words, bumped_entries = _positional_create_order(
        entries,
        word,
        destination_word,
        relation,
        _PHRASE_TYPE_BASE_WEIGHTS[phrase_type],
    )

    return _PositionalCreateBinding(
        code=code,
        destination_word=destination_word,
        relation=relation,
        phrase_type=phrase_type,
        weight=weight,
        resulting_words=resulting_words,
        bumped_entries=bumped_entries,
    )


def _pending_create_is_bound(
    message: str,
    arguments: Dict,
    context: ToolContext,
) -> bool:
    """Bind an add to a candidate preserved in a live server snapshot."""
    capability = context.pending_candidate
    if capability is None or not capability.state_matches:
        return False
    word = str(arguments.get("word") or "").strip()
    code = str(arguments.get("code") or "").strip().lower()
    action = str(arguments.get("action") or "Create").strip()
    old_word = str(arguments.get("old_word") or "").strip()
    candidates = dict(capability.candidates)
    return bool(
        action == "Create"
        and not old_word
        and word
        and word == capability.word
        and _contains_exact_target(message, word)
        and _action_is_bound_to_target(message, word, "Create")
        and not _is_word_protected(message, word)
        and len(candidates) == len(capability.candidates)
        and code in candidates
    )


def _pending_batch_selected_items(
    message: str,
    context: ToolContext,
) -> Optional[Tuple[Tuple[str, str, str], ...]]:
    """Resolve an exact multi-selection against one live server snapshot."""
    capability = context.pending_candidate
    if capability is None or not capability.state_matches:
        return None
    selection = parse_pending_candidate_selection(message)
    if selection is None:
        return None
    candidates = dict(capability.candidates)
    if len(candidates) != len(capability.candidates):
        return None
    if selection.indices:
        if (
            len(set(selection.indices)) != len(selection.indices)
            or any(
                not 1 <= index <= len(capability.candidates)
                for index in selection.indices
            )
        ):
            return None
        codes = tuple(
            capability.candidates[index - 1][0]
            for index in selection.indices
        )
    else:
        if (
            len(set(selection.codes)) != len(selection.codes)
            or any(code not in candidates for code in selection.codes)
        ):
            return None
        codes = selection.codes
    return tuple(("Create", capability.word, code) for code in codes)


def _pending_batch_items_match_selection(
    items: Any,
    selected_items: Optional[Tuple[Tuple[str, str, str], ...]],
) -> bool:
    if selected_items is None or not isinstance(items, list):
        return False
    normalized: List[Tuple[str, str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            return False
        action = str(item.get("action") or "Create").strip()
        word = str(item.get("word") or "").strip()
        code = str(item.get("code") or "").strip().lower()
        if action != "Create" or not word or not code or item.get("old_word"):
            return False
        normalized.append((action, word, code))
    return (
        len(normalized) == len(selected_items)
        and set(normalized) == set(selected_items)
    )


def _positional_destination_operand_is_exact(
    message: str,
    destination_word: str,
) -> bool:
    """Require the parsed destination to be the current command's exact Y."""
    operands = _positional_create_operands(message)
    return bool(
        operands is not None
        and destination_word
        and operands[1] == destination_word
    )


def _destination_derived_positional_create_binding(
    message: str,
    arguments: Dict,
    context: ToolContext,
) -> Optional[_PositionalCreateBinding]:
    """Bind one create to a unique destination entry read in this turn."""
    binding_message, _marker_found = _normalize_positional_same_code_markers(
        message
    )
    operands = _positional_create_operands(binding_message)
    if operands is None:
        return None
    subject, destination_word, relation = operands
    word = str(arguments.get("word") or "").strip()
    code = str(arguments.get("code") or "").strip().lower()
    action = str(arguments.get("action") or "Create").strip()
    old_word = str(arguments.get("old_word") or "").strip()
    destination_codes = (
        context.trusted_word_lookup_codes_by_word or {}
    ).get(destination_word, frozenset())
    if (
        action != "Create"
        or old_word
        or not word
        or subject != word
        or not _contains_exact_target(binding_message, word)
        or not destination_word
        or not _positional_destination_operand_is_exact(
            binding_message,
            destination_word,
        )
        or len(destination_codes) != 1
        or code not in destination_codes
        or relation not in {
            "前面", "后面", "之前", "之后", "前", "后",
        }
    ):
        return None
    phrase_type = _positional_phrase_type(
        arguments, context, word, code, destination_word
    )
    entries = _same_type_positional_entries(
        (context.trusted_entries_by_code or {}).get(code, ()),
        code,
        phrase_type,
        context,
    )
    weight, resulting_words, bumped_entries = _positional_create_order(
        entries,
        word,
        destination_word,
        relation,
        _PHRASE_TYPE_BASE_WEIGHTS[phrase_type],
    )
    return _PositionalCreateBinding(
        code=code,
        destination_word=destination_word,
        relation=relation,
        phrase_type=phrase_type,
        weight=weight,
        resulting_words=resulting_words,
        bumped_entries=bumped_entries,
    )


def _served_candidate_slots(
    context: ToolContext,
    word: str,
) -> Tuple[Tuple[str, bool], ...]:
    """Return one ordered, occupancy-checked candidate chain for ``word``."""
    capability = context.pending_candidate
    if (
        capability is not None
        and capability.state_matches
        and capability.word == word
    ):
        raw_slots = capability.candidates
    else:
        raw_slots = (context.trusted_candidate_slots_by_word or {}).get(
            word,
            (),
        )
    normalized: List[Tuple[str, bool]] = []
    seen: set[str] = set()
    for raw_code, occupied in raw_slots:
        code = str(raw_code or "").strip().lower()
        if (
            not re.fullmatch(_POSITIONAL_REORDER_CODE_PATTERN, code)
            or not isinstance(occupied, bool)
            or code in seen
        ):
            return ()
        normalized.append((code, occupied))
        seen.add(code)
    return tuple(normalized)


def _next_free_served_candidate(
    context: ToolContext,
    word: str,
    current_code: str,
) -> Tuple[str, Tuple[Tuple[str, bool], ...]]:
    slots = _served_candidate_slots(context, word)
    codes = [code for code, _occupied in slots]
    if current_code not in codes:
        return "", slots
    current_index = codes.index(current_code)
    if not slots[current_index][1]:
        return "", slots
    for code, occupied in slots[current_index + 1:]:
        if not occupied:
            return code, slots
    return "", slots


def _positional_destination_is_bound(
    message: str,
    word: str,
    target_code: str,
    context: ToolContext,
) -> bool:
    """Require provenance for every destination in positional syntax.

    Bare ASCII is not evidence that a token is a KeyTao code.  Positional
    commands therefore need a candidate code returned by a read tool for the
    moved word.  A named destination additionally needs either explicit quotes
    or a server-read entry capability.  Legacy ``顺延`` commands keep their
    existing binding contract and never enter this helper.
    """
    parsed = _positional_destination_from_command(message)
    raw_destination = _raw_positional_destination_from_command(message)
    if parsed is None and raw_destination is None:
        return True
    if parsed is None:
        return False

    trusted_codes = context.trusted_codes_by_word or {}
    word_codes = trusted_codes.get(word, frozenset())
    if not re.fullmatch(_POSITIONAL_REORDER_CODE_PATTERN, target_code):
        return False
    if parsed.kind == "code":
        return bool(
            parsed.target == target_code
            and target_code in word_codes
        )
    if target_code not in word_codes:
        return False
    if parsed.kind == "entry":
        return parsed.quoted or _server_knows_positional_entry(
            context,
            parsed.target,
        )
    return parsed.kind in {"ordinal", "relative"}


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

    @staticmethod
    def _validate_current_message_binding(
        tool_name: str,
        arguments: Dict,
        context: ToolContext,
    ) -> Optional[Dict]:
        """Bind model-generated mutation targets to entities in current raw text."""
        message = _mutation_authorization_view(context.current_message or "")
        multi_add = _multi_add_authorization_contract(
            context.current_message or ""
        )
        trusted_codes = context.trusted_codes_by_word or {}
        trusted_word_lookup_codes = (
            context.trusted_word_lookup_codes_by_word or {}
        )
        trusted_draft_words = context.trusted_draft_words_by_id or {}
        trusted_draft_items = context.trusted_draft_items_by_id or {}
        compact_message = re.sub(r"[\s，,。.!！?？~～]+", "", message)
        if tool_name == "keytao_submit_batch":
            if not _explicit_submit_command_matches(compact_message):
                return policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    "安全拦截：提交只能由本轮明确、独立的提交指令授权。",
                    missing=["submitCommand"],
                )
        if tool_name == "keytao_recall_batch":
            if not _explicit_recall_command_matches(compact_message):
                return policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    "安全拦截：撤回只能由本轮明确、独立的撤回指令授权。",
                    missing=["recallCommand"],
                )
        if tool_name == "keytao_update_draft_item_weight":
            word = str(arguments.get("word") or "").strip()
            code = str(arguments.get("code") or "").strip().lower()
            weight = arguments.get("weight")
            matching_items = [
                item
                for item in trusted_draft_items.values()
                if isinstance(item, dict)
                and str(item.get("word") or "").strip() == word
                and str(item.get("code") or "").strip().lower() == code
            ]
            value_bound = bool(
                isinstance(weight, int)
                and not isinstance(weight, bool)
                and re.search(rf"(?<!\d){weight}(?!\d)", message)
            )
            item_pair_bound = bool(
                word
                and code
                and re.search(
                    rf"[「“]?{re.escape(word)}[」”]?\s*{re.escape(code)}(?![A-Za-z])",
                    message,
                )
            )
            if not (
                word
                and code
                and len(matching_items) == 1
                and item_pair_bound
                and value_bound
                and _WEIGHT_ADJUST_VERB_RE.search(message)
                and not _is_word_protected(message, word)
            ):
                return policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    "安全拦截：权重调整必须把词条、编码、整数权重与本轮读取到的唯一草稿条目完整绑定。",
                    missing=["draftItemWord", "draftItemCode", "weight"],
                )
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
                return policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    "安全拦截：删除条目的 ID 或词条未精确出现在用户本轮原始文字中。",
                    missing=["draftItemId"],
                )
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
                return policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    "安全拦截：批量删除 ID 未全部出现在用户本轮原始文字中。",
                    missing=["draftItemId"],
                    unboundIds=missing_ids[:12],
                )
        if tool_name == "keytao_create_phrase":
            word = str(arguments.get("word") or "").strip()
            code = str(arguments.get("code") or "").strip()
            action = str(arguments.get("action") or "Create")
            old_word = str(arguments.get("old_word") or "").strip()
            binding_message, _marker_found = (
                _normalize_positional_same_code_markers(message)
            )
            pending_positional_create = _pending_positional_create_binding(
                message,
                arguments,
                context,
            )
            destination_positional_create = (
                _destination_derived_positional_create_binding(
                    message,
                    arguments,
                    context,
                )
            )
            positional_create = (
                pending_positional_create or destination_positional_create
            )
            if (
                positional_create is not None
                and positional_create.weight is None
                and _positional_same_code_requested(message)
            ):
                return text_follow_up(
                    BLOCK_REASON_ORDERING_NOT_EXPRESSIBLE,
                    "当前已有权重之间没有可用的整数位置，无法精确保持这条排序指令。"
                    "请先调整现有词条优先级，或改为只添加重码。",
                    word=word,
                    destinationWord=positional_create.destination_word,
                    relation=positional_create.relation,
                )
            targets = [word]
            if action == "Change" and old_word:
                targets.append(old_word)
            missing_targets = [
                target for target in targets
                if not target
                or not _contains_exact_target(binding_message, target)
            ]
            create_intent_is_bound = bool(
                action == "Create"
                and not old_word
                and word
                and _contains_exact_target(binding_message, word)
                and _action_is_bound_to_target(
                    binding_message,
                    word,
                    "Create",
                )
                and not _is_word_protected(binding_message, word)
            )
            explicit_create_is_bound = bool(
                create_intent_is_bound
                and code
                and _code_is_bound_to_target(
                    message,
                    word,
                    code,
                    frozenset(),
                )
            )
            pending_create_is_bound = _pending_create_is_bound(
                message,
                arguments,
                context,
            )
            create_is_bound = bool(
                explicit_create_is_bound
                or pending_create_is_bound
                or positional_create is not None
            )
            positional_operands = _positional_create_operands(binding_message)
            if (
                action == "Create"
                and not old_word
                and not create_is_bound
                and positional_operands is not None
                and positional_operands[0] == word
                and positional_operands[2] in {
                    "前面", "后面", "之前", "之后", "前", "后",
                }
                and _contains_exact_target(binding_message, word)
                and not _is_word_protected(binding_message, word)
            ):
                destination_word = positional_operands[1]
                destination_codes = trusted_word_lookup_codes.get(
                    destination_word,
                    frozenset(),
                )
                if len(destination_codes) != 1:
                    return text_follow_up(
                        (
                            "destination_code_ambiguous"
                            if len(destination_codes) > 1
                            else "code_required"
                        ),
                        (
                            f"“{destination_word}”对应多个编码："
                            f"{'、'.join(sorted(destination_codes))}。"
                            f"请问要把“{word}”加到哪个编码？"
                            if len(destination_codes) > 1
                            else (
                                f"请先调用 keytao_lookup_by_word 查询"
                                f"“{destination_word}”的全部编码，再用查询结果"
                                "重试本次 keytao_create_phrase；"
                                "此阶段不要向用户询问编码。"
                            )
                        ),
                        word=word,
                        destinationWord=destination_word,
                        candidateCodes=sorted(destination_codes),
                        nextAction=(
                            {
                                "type": "ask_user_to_choose_code",
                                "candidateCodes": sorted(destination_codes),
                            }
                            if len(destination_codes) > 1
                            else {
                                "tool": "keytao_lookup_by_word",
                                "arguments": {"word": destination_word},
                                "then": "retry_same_create",
                                "askUserForCode": False,
                            }
                        ),
                    )
            if (
                action == "Create"
                and not old_word
                and not create_is_bound
                and create_intent_is_bound
                and not _target_clause_has_explicit_code_token(message, word)
            ):
                return text_follow_up(
                    "code_required",
                    f"请问要把“{word}”添加到哪个编码？",
                    word=word,
                )
            if (
                action == "Create"
                and not create_is_bound
            ) or (action != "Create" and (
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
            )):
                return policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    f"安全拦截：{action} 操作的动作、词条或编码"
                    "未与用户本轮原始文字中的完整目标绑定。",
                    missing=["boundTarget"],
                    unboundTargets=missing_targets[:12],
                )
        elif tool_name == "keytao_batch_add_to_draft":
            items = arguments.get("items")
            if not isinstance(items, list) or not items:
                return policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    "安全拦截：批量操作缺少可绑定的词条。",
                    missing=["items"],
                )
            pending_selected_items = _pending_batch_selected_items(
                message,
                context,
            )
            pending_batch_is_bound = _pending_batch_items_match_selection(
                items,
                pending_selected_items,
            )
            if pending_selected_items is not None and not pending_batch_is_bound:
                expected_items = [
                    {"action": action, "word": word, "code": code}
                    for action, word, code in pending_selected_items
                ]
                expected_labels = "、".join(
                    f"「{word}」{code}"
                    for _action, word, code in pending_selected_items
                )
                retry = _suggested_command_text(
                    "keytao_batch_add_to_draft",
                    {"items": expected_items},
                )
                return policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    "安全拦截：本次批量写入无法与候选快照中选中的"
                    f"{expected_labels}精确对应；整批均未写入。",
                    missing=["exactAuthorizedItemSet"],
                    suggestion=(SUGGESTION_MENTION_PREFIX + retry) if retry else "",
                )
            if (
                multi_add is not None
                and not _multi_add_items_match_authorized_set(
                    multi_add,
                    items,
                    trusted_codes,
                )
            ):
                authorized_items = [
                    f"{clause.word} {clause.code}".strip()
                    for clause in dict.fromkeys(multi_add.clauses)
                ]
                return policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    "安全拦截：批量加词工具的完整条目集合必须与用户本轮"
                    "逐句授权的条目集合完全一致，不能遗漏、调换编码或夹带未点名条目。",
                    missing=["exactAuthorizedItemSet"],
                    authorizedItems=authorized_items,
                )
            blocked_items: List[str] = []
            for item in ([] if pending_batch_is_bound else items):
                if not isinstance(item, dict):
                    blocked_items.append("无效条目")
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
                    visible_word = (
                        f"「{word}」"
                        if word and _contains_exact_target(message, word)
                        else "未能对应的词条"
                    )
                    visible_code = (
                        code
                        if code and _contains_exact_target(message, code)
                        else "编码未能对应"
                    )
                    blocked_items.append(
                        f"{visible_word}（{visible_code}）"
                    )
            if blocked_items:
                return policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    "安全拦截：无法把以下条目与本轮消息逐项对应："
                    + "、".join(blocked_items[:12])
                    + "；整批均未写入。请在下一条消息中逐项写全动作、"
                    "词条和编码后重试。",
                    missing=["boundTarget"],
                    unboundItems=blocked_items[:12],
                )
        elif tool_name == "keytao_shift_phrase_code":
            word = str(arguments.get("word") or "").strip()
            target_code = str(arguments.get("target_code") or "").strip()
            if (
                not word
                or not _contains_exact_target(message, word)
                or not _action_is_bound_to_target(message, word, "Change")
                or _is_word_protected(message, word)
                # A protection word anywhere else in the message still stops the
                # shift; one that *is* the entry being moved does not.
                or _has_protection_outside_target(message, word)
                or not _positional_destination_is_bound(
                    message,
                    word,
                    target_code,
                    context,
                )
                or not _code_is_bound_to_target(
                    message,
                    word,
                    target_code,
                    trusted_codes.get(word, frozenset()),
                )
            ):
                return policy_block(
                    BLOCK_REASON_BINDING_INCOMPLETE,
                    "安全拦截：顺延操作的词条或目标编码未精确绑定。",
                    missing=["boundWord", "boundCode"],
                )
        return None
