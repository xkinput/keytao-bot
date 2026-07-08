"""LLM-backed KeyTao batch review helpers."""
from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from nonebot import get_driver
from nonebot.log import logger

try:
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - optional dependency guard
    AsyncOpenAI = None  # type: ignore

from .keytao_review import ReviewHttpConfig, audit_draft_items, fetch_keytao_encode, lookup_codes


ReviewItem = Dict[str, Any]


def _config_value(name: str, env_name: str, default: Any = None) -> Any:
    try:
        config = get_driver().config
        value = getattr(config, name, None)
        if value not in (None, ""):
            return value
    except Exception:
        pass
    return os.getenv(env_name, default)


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _review_config() -> ReviewHttpConfig:
    return ReviewHttpConfig(
        api_base=str(_config_value("keytao_api_base", "KEYTAO_API_BASE", "https://keytao.vercel.app")).rstrip("/"),
        bot_token=str(_config_value("bot_api_token", "BOT_API_TOKEN", "") or ""),
    )


def _llm_config() -> Dict[str, Any]:
    timeout_value = (
        _config_value("openai_timeout", "OPENAI_TIMEOUT", None)
        or _config_value("gemini_timeout", "GEMINI_TIMEOUT", None)
        or _config_value("ark_timeout", "ARK_TIMEOUT", None)
        or 180
    )
    temperature_value = (
        _config_value("openai_temperature", "OPENAI_TEMPERATURE", None)
        or _config_value("gemini_temperature", "GEMINI_TEMPERATURE", None)
        or _config_value("ark_temperature", "ARK_TEMPERATURE", None)
        or 0.2
    )
    return {
        "api_key": str(_config_value("openai_api_key", "OPENAI_API_KEY", "") or ""),
        "base_url": str(
            _config_value("openai_base_url", "OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
        ),
        "model": str(
            _config_value("keytao_review_model", "KEYTAO_REVIEW_MODEL", "")
            or _config_value("openai_model", "OPENAI_MODEL", "gemini-2.0-flash")
        ),
        "max_tokens": min(max(_as_int(_config_value("openai_max_tokens", "OPENAI_MAX_TOKENS", 2500), 2500), 2500), 6000),
        "timeout": _as_float(timeout_value, 180.0),
        "temperature": _as_float(temperature_value, 0.2),
    }


def _deterministic_audit_timeout() -> float:
    value = _config_value(
        "keytao_batch_review_audit_timeout",
        "KEYTAO_BATCH_REVIEW_AUDIT_TIMEOUT",
        25,
    )
    return max(5.0, _as_float(value, 25.0))


def _string(value: Any) -> str:
    return str(value or "").strip()


def _list_of_strings(value: Any, limit: int = 8) -> List[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    result: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _extract_items(batch: Dict[str, Any]) -> List[ReviewItem]:
    raw_items = batch.get("pullRequests") or batch.get("pull_requests") or []
    items: List[ReviewItem] = []
    if not isinstance(raw_items, list):
        return items
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        phrase = raw.get("phrase") if isinstance(raw.get("phrase"), dict) else {}
        try:
            pr_id = int(raw.get("id"))
        except Exception:
            continue
        items.append({
            "id": pr_id,
            "action": _string(raw.get("action") or "Create") or "Create",
            "word": _string(raw.get("word") or phrase.get("word")),
            "oldWord": _string(raw.get("oldWord") or raw.get("old_word")),
            "code": _string(raw.get("code") or phrase.get("code")).lower(),
            "type": _string(raw.get("type") or phrase.get("type") or "Phrase") or "Phrase",
            "weight": raw.get("weight"),
            "remark": _string(raw.get("remark")),
            "hasConflict": bool(raw.get("hasConflict")),
            "conflictReason": _string(raw.get("conflictReason")),
            "conflictInfo": raw.get("conflictInfo") if isinstance(raw.get("conflictInfo"), dict) else None,
        })
    return items


def _compact_json(value: Any, max_chars: int = 18000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        data = json.loads(cleaned[start:end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError("LLM did not return a JSON object")


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _move_pairs(items: Sequence[ReviewItem]) -> set[Tuple[str, str]]:
    creates_by_word: Dict[str, List[ReviewItem]] = {}
    for item in items:
        if item.get("action") == "Create":
            creates_by_word.setdefault(_string(item.get("word")), []).append(item)

    pairs: set[Tuple[str, str]] = set()
    for item in items:
        if item.get("action") != "Delete":
            continue
        word = _string(item.get("word"))
        code = _string(item.get("code")).lower()
        for created in creates_by_word.get(word, []):
            new_code = _string(created.get("code")).lower()
            if new_code and new_code != code:
                pairs.add((word, code))
                break
    return pairs


def _collect_code_strings(value: Any, result: Optional[List[str]] = None) -> List[str]:
    result = result if result is not None else []
    if isinstance(value, str):
        code = value.strip().lower()
        if code and re.fullmatch(r"[a-z]+", code) and code not in result:
            result.append(code)
        return result
    if isinstance(value, list):
        for item in value:
            _collect_code_strings(item, result)
        return result
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"code", "codes", "candidateCodes", "altCodes", "requestedCandidateCodes", "seriesCodes"}:
                _collect_code_strings(item, result)
            elif key in {"flyKeyVariants", "alternatePronunciationCodes", "alternatePhrasePronunciationCodes", "candidateStatuses", "items"}:
                _collect_code_strings(item, result)
        return result
    return result


def _encode_candidate_codes(encode_data: Dict[str, Any]) -> List[str]:
    codes: List[str] = []
    for key in ("candidateCodes", "codes", "altCodes", "requestedCandidateCodes"):
        _collect_code_strings(encode_data.get(key), codes)
    for key in ("flyKeyVariants", "alternatePronunciationCodes", "alternatePhrasePronunciationCodes", "candidateStatuses"):
        _collect_code_strings(encode_data.get(key), codes)
    return codes


def _status_label(phrases: Sequence[Dict[str, Any]]) -> str:
    words = [_string(item.get("word")) for item in phrases if isinstance(item, dict) and _string(item.get("word"))]
    if not words:
        return "空位"
    label = "已有「" + "、".join(words[:3]) + "」"
    if len(words) > 3:
        label += f"等 {len(words)} 个词"
    return label


def _pinyin_from_encode_chars(encode_data: Dict[str, Any]) -> str:
    chars = encode_data.get("chars")
    if not isinstance(chars, list):
        return ""
    pinyins = [
        _string(item.get("pinyin"))
        for item in chars
        if isinstance(item, dict) and _string(item.get("pinyin"))
    ]
    return " ".join(pinyins)


async def _fallback_audit_with_encode(config: ReviewHttpConfig, items: Sequence[ReviewItem], reason: str) -> Dict[str, Any]:
    move_pairs = _move_pairs(items)
    words: List[str] = []
    for item in items:
        action = _string(item.get("action") or "Create") or "Create"
        if action == "Delete":
            continue
        word = _string(item.get("word"))
        if word and word not in words:
            words.append(word)

    encode_results = await asyncio.gather(
        *(fetch_keytao_encode(config, word) for word in words),
        return_exceptions=True,
    )
    encode_by_word: Dict[str, Dict[str, Any]] = {}
    all_codes: List[str] = []
    for word, result in zip(words, encode_results):
        if isinstance(result, Exception):
            encode_by_word[word] = {"success": False, "message": str(result)}
            continue
        encode_by_word[word] = result
        for code in _encode_candidate_codes(result):
            if code not in all_codes:
                all_codes.append(code)

    try:
        code_map = await lookup_codes(config, all_codes)
    except Exception:
        code_map = {}

    reviewed_words: Dict[str, Dict[str, Any]] = {}
    for word, encode_data in encode_by_word.items():
        candidate_codes = _encode_candidate_codes(encode_data)
        statuses = [
            {
                "code": code,
                "occupied": bool(code_map.get(code)),
                "label": _status_label(code_map.get(code, [])),
                "phrases": code_map.get(code, []),
            }
            for code in candidate_codes
        ]
        recommended = next((item["code"] for item in statuses if not item["occupied"]), candidate_codes[0] if candidate_codes else "")
        reviewed_words[word] = {
            "success": bool(candidate_codes),
            "word": word,
            "autoReviewable": False,
            "autoReviewReason": "来源抓取超时，仅保留 keytao_encode 候选链供 LLM 复审",
            "encodeOnly": True,
            "keytaoEncode": {
                "candidateCodes": candidate_codes,
                "candidateStatuses": statuses[:12],
                "recommendedCode": recommended,
                "type": encode_data.get("type"),
                "chars": encode_data.get("chars", [])[:8] if isinstance(encode_data.get("chars"), list) else [],
            },
            "pronunciations": [
                {
                    "pinyin": _pinyin_from_encode_chars(encode_data),
                    "normalized": [],
                    "codes": candidate_codes,
                    "sources": [{"source": "keytao_encode", "url": config.api_base}],
                    "score": 0,
                    "fallback": True,
                    "candidateStatuses": statuses[:12],
                    "recommendedCode": recommended,
                }
            ] if candidate_codes else [],
        }

    issues: List[str] = []
    approved_items: List[str] = []
    for item in items:
        action = _string(item.get("action") or "Create") or "Create"
        word = _string(item.get("word"))
        code = _string(item.get("code")).lower()
        phrase_type = _string(item.get("type") or "Phrase") or "Phrase"
        if not word or not code:
            issues.append("存在词或编码为空的草稿条目")
            continue
        if action == "Delete" and (word, code) not in move_pairs:
            issues.append(f"纯删除「{word}」@{code} 必须由管理员审核")
            continue
        if phrase_type in {"CSS", "CSSSingle"}:
            approved_items.append(f"{action}：{word}@{code} 是声笔笔短码表条目，交由 LLM 按 CSS 优先级复审")
            continue
        candidate_codes = reviewed_words.get(word, {}).get("keytaoEncode", {}).get("candidateCodes") or []
        if code in candidate_codes:
            approved_items.append(f"{action}：{word}@{code}，keytao_encode 候选链包含目标编码")
        else:
            available = ", ".join(candidate_codes[:8])
            issues.append(f"「{word}」编码 {code} 不在 keytao_encode 候选链中，可选：{available or '无'}")

    return {
        "success": True,
        "verdict": "needs_admin",
        "autoApprove": False,
        "summary": "来源抓取超时，本喵已保留 keytao_encode 候选链供 LLM 复审",
        "issues": issues or [reason],
        "approvedItems": approved_items,
        "commonKnownItems": [],
        "reviewedWords": reviewed_words,
        "commonnessComparisons": [],
        "deterministicAuditTimedOut": True,
        "deterministicAuditReason": reason,
        "sourcePolicy": {
            "note": (
                "本次管理员复查未等完网页来源抓取；编码正确性必须以 keytao_encode 候选链、"
                "CSS 短码表或 KeyTao 文档为准，禁止按通用双拼盲猜。"
            ),
        },
    }


def _fallback_audit_for_llm(items: Sequence[ReviewItem], reason: str) -> Dict[str, Any]:
    move_pairs = _move_pairs(items)
    issues: List[str] = []
    approved_items: List[str] = []

    for item in items:
        action = _string(item.get("action") or "Create") or "Create"
        word = _string(item.get("word"))
        code = _string(item.get("code")).lower()
        if not word or not code:
            issues.append("存在词或编码为空的草稿条目")
            continue
        if action == "Delete" and (word, code) not in move_pairs:
            issues.append(f"纯删除「{word}」@{code} 必须由管理员审核")
            continue
        approved_items.append(f"{action}：{word}@{code} 交由 LLM 结合语言常识、编码链和本地冲突继续复审")

    return {
        "success": True,
        "verdict": "needs_admin",
        "autoApprove": False,
        "summary": "来源抓取超时，本喵已改用 LLM 继续复审",
        "issues": issues or [reason],
        "approvedItems": approved_items,
        "commonKnownItems": [],
        "reviewedWords": {},
        "commonnessComparisons": [],
        "deterministicAuditTimedOut": True,
        "deterministicAuditReason": reason,
        "sourcePolicy": {
            "note": "本次管理员复查未等完网页来源抓取，LLM 仍需按读音、编码、冲突和编码链保守判断。",
        },
    }


def _normalize_status(value: Any) -> str:
    status = _string(value).lower()
    if status in {"pass", "passed", "approve", "approved", "ok", "通过"}:
        return "pass"
    if status in {"manual_review", "manual", "reject", "danger", "人工", "需人工确认", "不通过"}:
        return "manual_review"
    return "attention"


def _severity_for_status(status: str) -> str:
    if status == "pass":
        return "success"
    if status == "manual_review":
        return "danger"
    return "warning"


def _verdict_for_items(items: Sequence[Dict[str, Any]]) -> str:
    if any(item.get("status") == "manual_review" for item in items):
        return "manual_review"
    if any(item.get("status") == "attention" for item in items):
        return "needs_attention"
    return "pass"


def _summary_from_item(item: Dict[str, Any], reasons: Sequence[str]) -> str:
    summary = _string(item.get("summary") or item.get("title"))
    if summary:
        return summary[:180]
    return (reasons[0] if reasons else "本喵已完成复审")[:180]


_GENERIC_ENCODING_GUESS_MARKERS = (
    "通用双拼",
    "常规双拼",
    "普通双拼",
    "零声母",
    "多出v",
    "多出 v",
    "声韵编码不应",
    "声母为",
    "无法判定该编码由真实读音严格推出",
    "键道具体编码方案未在",
)


def _contains_generic_encoding_guess(values: Sequence[str]) -> bool:
    text = "\n".join(values)
    return any(marker in text for marker in _GENERIC_ENCODING_GUESS_MARKERS)


def _audit_supports_item_code(audit: Optional[Dict[str, Any]], word: str, code: str) -> bool:
    if not isinstance(audit, dict) or not word or not code:
        return False
    review = (audit.get("reviewedWords") or {}).get(word)
    if not isinstance(review, dict):
        return False
    candidate_sets: List[Any] = []
    for pronunciation in review.get("pronunciations", []):
        if isinstance(pronunciation, dict):
            candidate_sets.append(pronunciation.get("codes"))
            candidate_sets.append(pronunciation.get("candidateStatuses"))
    keytao_encode = review.get("keytaoEncode") if isinstance(review.get("keytaoEncode"), dict) else {}
    candidate_sets.extend([
        review.get("candidateCodes"),
        keytao_encode.get("candidateCodes"),
        keytao_encode.get("candidateStatuses"),
    ])
    for candidate_set in candidate_sets:
        if code in _collect_code_strings(candidate_set):
            return True
    return False


def _is_css_item(pr: ReviewItem) -> bool:
    return _string(pr.get("type") or "Phrase") in {"CSS", "CSSSingle"}


def _fallback_review_from_llm_error(
    items: Sequence[ReviewItem],
    audit: Dict[str, Any],
    local_review: Optional[Dict[str, Any]],
    reason: str,
) -> Dict[str, Any]:
    move_pairs = _move_pairs(items)
    audit_summary = _string(audit.get("summary"))
    audit_issues = _list_of_strings(audit.get("issues"), limit=4)
    raw_items: List[Dict[str, Any]] = []

    for pr in items:
        word = _string(pr.get("word"))
        code = _string(pr.get("code")).lower()
        action = _string(pr.get("action") or "Create") or "Create"
        status = "attention"
        title = "本喵建议人工复核"
        reasons = [f"模型输出异常：{reason}"]
        suggestions = ["请管理员按读音、编码、冲突和编码链顺序人工确认。"]

        if audit_summary:
            reasons.append(f"确定性审查：{audit_summary}")
        if action == "Delete" and (word, code) not in move_pairs:
            status = "manual_review"
            title = "纯删除需要管理员确认"
            suggestions.insert(0, "纯删除不能自动通过，请确认该词确实应删除。")
        if pr.get("hasConflict") or (isinstance(pr.get("conflictInfo"), dict) and pr["conflictInfo"].get("hasConflict")):
            status = "manual_review"
            title = "冲突需要管理员确认"
            conflict_reason = _string(pr.get("conflictReason") or pr.get("conflictInfo", {}).get("impact"))
            if conflict_reason:
                reasons.insert(0, conflict_reason)

        raw_items.append({
            "prId": pr.get("id"),
            "status": status,
            "title": title,
            "reasons": reasons[:6],
            "suggestions": suggestions[:6],
            "sources": [],
            "evidence": [
                "本喵已调用模型复审，但模型没有返回可解析的完整 JSON。",
                *audit_issues,
            ][:8],
            "source": "bot-llm-fallback",
        })

    return _normalize_llm_review({
        "verdict": "manual_review" if any(item.get("status") == "manual_review" for item in raw_items) else "needs_attention",
        "headline": "本喵模型输出异常，已保守标记为需复核。",
        "suggestedReviewNote": (
            "本喵模型输出异常，未能稳定生成完整 JSON；"
            "已按确定性审查和本地冲突信息保守标记，请管理员人工确认。\n"
            f"异常：{reason}"
        ),
        "checklist": [
            "模型已被调用，但输出为空或 JSON 格式异常。",
            "本次结果不自动通过，仅作为人工复核提示。",
            "请重点核对读音、编码链、冲突和纯删除项。",
        ],
        "items": raw_items,
        "codeChainRecommendations": [],
    }, items, local_review, audit)


def _normalize_llm_review(
    raw: Dict[str, Any],
    items: Sequence[ReviewItem],
    local_review: Optional[Dict[str, Any]],
    audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_items = raw.get("items") if isinstance(raw.get("items"), list) else []
    raw_by_id: Dict[int, Dict[str, Any]] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        try:
            raw_by_id[int(raw_item.get("prId") or raw_item.get("id"))] = raw_item
        except Exception:
            continue

    move_pairs = _move_pairs(items)
    normalized_items: List[Dict[str, Any]] = []
    for pr in items:
        pr_id = int(pr["id"])
        raw_item = raw_by_id.get(pr_id, {})
        status = _normalize_status(raw_item.get("status"))
        reasons = _list_of_strings(raw_item.get("reasons"), limit=6)
        suggestions = _list_of_strings(raw_item.get("suggestions"), limit=6)

        word = _string(pr.get("word"))
        code = _string(pr.get("code"))
        if not reasons:
            reasons = [_string(raw_item.get("reason")) or f"本喵已复审「{word}」@{code}。"]
        if not suggestions:
            suggestions = [_string(raw_item.get("suggestion")) or "按读音、编码、冲突和编码链顺序继续复核。"]

        if pr.get("action") == "Delete" and (word, code) not in move_pairs:
            status = "manual_review"
            reasons.insert(0, "这是纯删除操作，需要管理员确认。")
            suggestions.insert(0, "确认该词确实不应存在；若是改码，请补齐新增侧。")

        if pr.get("hasConflict") or (isinstance(pr.get("conflictInfo"), dict) and pr["conflictInfo"].get("hasConflict")):
            status = "manual_review"
            conflict_reason = _string(pr.get("conflictReason") or pr.get("conflictInfo", {}).get("impact"))
            if conflict_reason:
                reasons.insert(0, conflict_reason)
            suggestions.insert(0, "先解决冲突，再决定是否批准。")

        sources = _list_of_strings(raw_item.get("sources"), limit=8)
        evidence = _list_of_strings(raw_item.get("evidence"), limit=8)
        pronunciation = _string(raw_item.get("pronunciation"))
        if pronunciation:
            evidence.insert(0, f"读音：{pronunciation}")
        if sources:
            evidence.append(f"来源：{'、'.join(sources)}")

        combined_review_text = [*reasons, *suggestions, *evidence]
        if _contains_generic_encoding_guess(combined_review_text):
            has_hard_blocker = (
                (pr.get("action") == "Delete" and (word, code) not in move_pairs)
                or bool(pr.get("hasConflict"))
                or (isinstance(pr.get("conflictInfo"), dict) and pr["conflictInfo"].get("hasConflict"))
            )
            if _audit_supports_item_code(audit, word, code):
                if not has_hard_blocker:
                    status = "pass"
                reasons = [
                    f"keytao_encode 候选链包含 {code}，编码按键道规则可推出；本喵已忽略脱离键道规则的错误推导。"
                ]
                suggestions = ["编码正确性以 keytao_encode/candidateStatuses 为准，继续核对词义、冲突和同码链顺序。"]
                evidence = [
                    f"编码依据：keytao_encode candidate chain includes {code}",
                    *[line for line in evidence if not _contains_generic_encoding_guess([line])],
                ]
            elif _is_css_item(pr):
                status = "attention" if status == "manual_review" else status
                reasons = [
                    "这是声笔笔/CSS 类型条目，不能按普通词组双拼+形码规则判定读音编码矛盾。"
                ]
                suggestions = ["请按声笔笔短码表、同码链常用度和结构对齐关系复核。"]
                evidence = [
                    "编码依据：CSS/CSSSingle 属于键道声笔笔短码表，不等同普通 Phrase 候选链。",
                    *[line for line in evidence if not _contains_generic_encoding_guess([line])],
                ]

        normalized_items.append({
            "prId": pr_id,
            "status": status,
            "severity": _severity_for_status(status),
            "title": _string(raw_item.get("title")) or ("本喵建议通过" if status == "pass" else "本喵建议复核"),
            "reasons": list(dict.fromkeys(reasons))[:6],
            "suggestions": list(dict.fromkeys(suggestions))[:6],
            "reviewRecord": {
                "reviewedBy": "Miaomiao",
                "source": _string(raw_item.get("source")) or "bot-llm",
                "summary": _summary_from_item(raw_item, reasons),
                "pronunciation": pronunciation or None,
                "sources": sources,
                "evidence": list(dict.fromkeys(evidence))[:8] or ["本喵已调用 LLM 完成复审。"],
            },
        })

    chain_recommendations = raw.get("codeChainRecommendations")
    chain_by_key: Dict[str, List[str]] = {}
    if isinstance(chain_recommendations, list):
        for chain in chain_recommendations:
            if not isinstance(chain, dict):
                continue
            code = _string(chain.get("code")).lower()
            chain_type = _string(chain.get("type") or "Phrase")
            recommendations = _list_of_strings(chain.get("recommendations"), limit=8)
            if code and recommendations:
                chain_by_key[f"{chain_type}:{code}"] = recommendations

    code_chains = []
    for chain in (local_review or {}).get("codeChains", []) if isinstance(local_review, dict) else []:
        if not isinstance(chain, dict):
            continue
        key = f"{_string(chain.get('type') or 'Phrase')}:{_string(chain.get('code')).lower()}"
        updated = dict(chain)
        if chain_by_key.get(key):
            updated["recommendations"] = chain_by_key[key]
        code_chains.append(updated)

    audit_chain_reviews = []
    audit_word_purposes = []
    if isinstance(audit, dict):
        if isinstance(audit.get("codeChainPriorityReviews"), list):
            audit_chain_reviews = [
                item for item in audit.get("codeChainPriorityReviews", [])
                if isinstance(item, dict)
            ]
        if isinstance(audit.get("wordPurposeReviews"), list):
            audit_word_purposes = [
                item for item in audit.get("wordPurposeReviews", [])
                if isinstance(item, dict)
            ]
    existing_chain_keys = {
        f"{_string(chain.get('type') or 'Phrase')}:{_string(chain.get('code')).lower()}"
        for chain in code_chains
        if isinstance(chain, dict)
    }
    for chain in audit_chain_reviews:
        key = f"{_string(chain.get('type') or 'Phrase')}:{_string(chain.get('code')).lower()}"
        if key in existing_chain_keys:
            continue
        moves = chain.get("recommendedMoves") if isinstance(chain.get("recommendedMoves"), list) else []
        recommendations = []
        if chain.get("hasRecommendation"):
            move_text = "、".join(
                f"「{move.get('word')}」→{move.get('toCode')}"
                for move in moves[:6]
                if isinstance(move, dict) and move.get("word") and move.get("toCode")
            )
            recommendations.append(
                f"{chain.get('summary', '同编码链建议重排')}"
                + (f"：{move_text}" if move_text else "")
            )
        code_chains.append({
            "code": chain.get("code"),
            "type": chain.get("type") or "Phrase",
            "currentOrder": chain.get("currentOrder") or [],
            "recommendedOrder": chain.get("recommendedOrder") or [],
            "recommendations": recommendations,
            "summary": chain.get("summary"),
        })

    pass_count = sum(1 for item in normalized_items if item["status"] == "pass")
    attention_count = sum(1 for item in normalized_items if item["status"] == "attention")
    manual_count = sum(1 for item in normalized_items if item["status"] == "manual_review")
    verdict = _verdict_for_items(normalized_items)

    headline = _string(raw.get("headline"))
    if not headline:
        if manual_count:
            headline = f"{manual_count} 项需要管理员确认，先看红色条目。"
        elif attention_count:
            headline = f"{attention_count} 项建议复核，其余条目本喵未发现硬性问题。"
        else:
            headline = f"{pass_count} 项本喵复审通过。"

    suggested_note = _string(raw.get("suggestedReviewNote"))
    if not suggested_note:
        suggested_note = headline + "\n" + "\n".join(
            f"- PR#{item['prId']} {item['title']}：{item['reasons'][0]}"
            for item in normalized_items[:12]
        )

    return {
        "reviewer": "Miaomiao",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "headline": headline,
        "suggestedReviewNote": suggested_note,
        "riskCounts": {
            "pass": pass_count,
            "attention": attention_count,
            "manualReview": manual_count,
            "botReviewed": len(normalized_items),
        },
        "checklist": _list_of_strings(raw.get("checklist"), limit=8) or [
            "本喵已调用 LLM 结合读音、编码、冲突和编码链完成复审。",
            "纯删除保持人工确认；调码按删除原位加新位处理。",
            "编码链顺序按常用度、词典与搜索证据保守判断。",
        ],
        "items": normalized_items,
        "codeChains": code_chains,
        "wordPurposeReviews": audit_word_purposes,
        "codeChainPriorityReviews": audit_chain_reviews,
    }


async def _call_llm(batch: Dict[str, Any], items: Sequence[ReviewItem], audit: Dict[str, Any], local_review: Optional[Dict[str, Any]], focus_pr_id: Optional[int]) -> Dict[str, Any]:
    config = _llm_config()
    if not config["api_key"] or AsyncOpenAI is None:
        raise RuntimeError("喵喵 LLM 未配置，无法完整复审")

    client = AsyncOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        timeout=config["timeout"],
    )
    system_prompt = (
        "你是键道输入法审词员喵喵。你必须根据给定证据做保守、专业的中文词语审核。"
        "重点检查：真实读音、编码是否由真实读音推出、同码链顺序是否合理、改词是否把正确词误改掉、"
        "纯删除是否必须人工确认、调码是否等价于删除原位并新增正确位置。"
        "每个新增/修改词都必须判断这个词的用途/语境类别（如日常词、网络词、专业术语、品牌、人物别名等），"
        "并和同编码候选链里已经占位的词比较常用度优先级。"
        "如果现有顺序合理，明确不建议调序；只有新词或链上其他词明显更常用、应该占更短码时，才给出新的重排建议。"
        "重排建议必须具体到“哪个词应到哪个编码”，不要泛泛说优化。"
        "编码正确性只能依据 deterministicAudit.reviewedWords、keytao_encode 返回的 candidateCodes/"
        "candidateStatuses/requestedCodeAnalysis、localReview 的编码链、以及 KeyTao/键道6 文档。"
        "禁止使用通用双拼、普通拼音键位、零声母猜测或你自己的声韵推导来判定键道编码；"
        "如果目标编码已经出现在 keytao_encode 候选链中，不得说“无法由读音推出”或“多出某个字母”。"
        "CSS/CSSSingle 是键道声笔笔短码表，fa/fao 等码位不等同普通 Phrase 双拼+形码，"
        "审核 CSS 时应检查短码表、同码链优先级、词频/结构对齐，不得以 zhi/fou 与 f/ao 不对应为理由判错。"
        "常见现代汉语词语、成语、熟语、大众明确知晓的固定表达，或广为人知的实体名/简称/别名，"
        "包括明星艺名、历史人物姓名/字/号/别名、角色名、品牌/产品、作品、地名、组织机构等。"
        "例如“敬德”可指尉迟敬德/尉迟恭字敬德，“杰伦”可指周杰伦。即使没有抓到该短词自己的权威读音页，"
        "只要你能明确给出读音和含义，且目标编码在 deterministicAudit 的读音候选链中，可以建议通过；"
        "此时 evidence 写“本喵语言常识：读音/含义大众通行”，sources 可以为空或写“语言常识”。"
        "陌生专名、冷僻词、网络临时造词、多音读法不稳、含义不明或编码不在候选链中，仍要标为人工确认或复核。"
        "只有读音、编码和常用度/语言常识证据一致时才建议通过；证据不足、歧义或纯删除要标为人工确认或复核。"
        "只返回 JSON，不要 Markdown。"
    )
    schema_hint = {
        "verdict": "pass | needs_attention | manual_review",
        "headline": "one short Chinese sentence",
        "suggestedReviewNote": "Chinese review note for admins",
        "checklist": ["checked item"],
        "items": [{
            "prId": 1,
            "status": "pass | attention | manual_review",
            "title": "short label",
            "reasons": ["why"],
            "suggestions": ["what admin should do"],
            "pronunciation": "optional pinyin",
            "sources": ["汉典"],
            "evidence": ["short evidence"],
        }],
        "codeChainRecommendations": [{
            "code": "abc",
            "type": "Phrase",
            "recommendations": ["priority advice"],
        }],
        "wordPurposeReviews": [{
            "word": "词",
            "usage": "用途/语境类别",
            "confidence": "high | medium | low",
        }],
    }
    user_payload = {
        "batch": {
            "id": batch.get("id"),
            "status": batch.get("status"),
            "description": batch.get("description"),
        },
        "focusPrId": focus_pr_id,
        "pullRequests": items,
        "deterministicAudit": audit,
        "localReview": local_review,
        "requiredJsonShape": schema_hint,
    }
    user_content = _compact_json(user_payload, max_chars=42000)
    last_error: Optional[Exception] = None

    for attempt in range(1, 3):
        prompt = system_prompt
        if attempt > 1:
            prompt += (
                "上一次响应为空或不是合法 JSON。现在必须重新生成一个完整 JSON 对象，"
                "不要解释，不要省略字段，不要使用 Markdown 代码块。"
            )
        response = await client.chat.completions.create(
            model=config["model"],
            temperature=min(config["temperature"], 0.2),
            max_tokens=config["max_tokens"],
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
        )
        choice = response.choices[0] if response.choices else None
        finish_reason = _string(getattr(choice, "finish_reason", "")) if choice else "no_choices"
        message = getattr(choice, "message", None) if choice else None
        content = _message_content_to_text(getattr(message, "content", "") if message else "")
        logger.info(
            "KeyTao LLM batch review response "
            f"attempt={attempt} finish_reason={finish_reason or 'unknown'} content_len={len(content)}"
        )

        if not content:
            last_error = RuntimeError(f"喵喵 LLM 没有返回审查内容（finish_reason={finish_reason or 'unknown'}）")
            continue

        try:
            return _extract_json_object(content)
        except Exception as error:
            preview = content[:600].replace("\n", "\\n")
            logger.warning(
                "KeyTao LLM batch review returned invalid JSON "
                f"attempt={attempt}: {error}; preview={preview}"
            )
            last_error = error

    raise RuntimeError(str(last_error or "喵喵 LLM 未返回可解析的审查 JSON"))


async def review_keytao_batch_with_llm(
    batch: Dict[str, Any],
    local_review: Optional[Dict[str, Any]] = None,
    focus_pr_id: Optional[int] = None,
    precomputed_audit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    items = _extract_items(batch)
    if not items:
        return {"success": False, "message": "批次没有可审查条目"}

    if precomputed_audit is not None:
        audit = precomputed_audit
    else:
        audit_timeout = _deterministic_audit_timeout()
        try:
            audit = await asyncio.wait_for(
                audit_draft_items(_review_config(), items),
                timeout=audit_timeout,
            )
        except asyncio.TimeoutError:
            reason = f"确定性来源审查超过 {audit_timeout:.0f} 秒"
            logger.warning(f"KeyTao deterministic batch audit timed out before LLM review: {reason}")
            try:
                audit = await _fallback_audit_with_encode(_review_config(), items, reason)
            except Exception as encode_error:
                logger.warning(f"KeyTao encode-only fallback audit failed: {encode_error}")
                audit = _fallback_audit_for_llm(items, reason)
        except Exception as error:
            reason = f"确定性来源审查失败：{error}"
            logger.warning(f"KeyTao deterministic batch audit failed before LLM review: {error}")
            try:
                audit = await _fallback_audit_with_encode(_review_config(), items, reason)
            except Exception as encode_error:
                logger.warning(f"KeyTao encode-only fallback audit failed: {encode_error}")
                audit = _fallback_audit_for_llm(items, reason)

    try:
        raw_review = await _call_llm(batch, items, audit, local_review, focus_pr_id)
    except Exception as error:
        logger.warning(f"KeyTao LLM batch review failed: {error}")
        ai_review = _fallback_review_from_llm_error(items, audit, local_review, str(error))
        return {
            "success": True,
            "aiReview": ai_review,
            "reviewedAt": ai_review["generatedAt"],
            "warning": str(error),
        }

    ai_review = _normalize_llm_review(raw_review, items, local_review, audit)
    return {
        "success": True,
        "aiReview": ai_review,
        "reviewedAt": ai_review["generatedAt"],
    }
