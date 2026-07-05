"""LLM-backed KeyTao batch review helpers."""
from __future__ import annotations

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

from .keytao_review import ReviewHttpConfig, audit_draft_items


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


def _normalize_llm_review(
    raw: Dict[str, Any],
    items: Sequence[ReviewItem],
    local_review: Optional[Dict[str, Any]],
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

        normalized_items.append({
            "prId": pr_id,
            "status": status,
            "severity": _severity_for_status(status),
            "title": _string(raw_item.get("title")) or ("本喵建议通过" if status == "pass" else "本喵建议复核"),
            "reasons": list(dict.fromkeys(reasons))[:6],
            "suggestions": list(dict.fromkeys(suggestions))[:6],
            "reviewRecord": {
                "reviewedBy": "Miaomiao",
                "source": "bot-llm",
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
        "只有读音、编码和常用度证据一致时才建议通过；证据不足、歧义或纯删除要标为人工确认或复核。"
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
    response = await client.chat.completions.create(
        model=config["model"],
        temperature=min(config["temperature"], 0.3),
        max_tokens=config["max_tokens"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _compact_json(user_payload, max_chars=42000)},
        ],
    )
    content = response.choices[0].message.content if response.choices else ""
    if not content:
        raise RuntimeError("喵喵 LLM 没有返回审查内容")
    return _extract_json_object(content)


async def review_keytao_batch_with_llm(
    batch: Dict[str, Any],
    local_review: Optional[Dict[str, Any]] = None,
    focus_pr_id: Optional[int] = None,
) -> Dict[str, Any]:
    items = _extract_items(batch)
    if not items:
        return {"success": False, "message": "批次没有可审查条目"}

    audit = await audit_draft_items(_review_config(), items)
    try:
        raw_review = await _call_llm(batch, items, audit, local_review, focus_pr_id)
    except Exception as error:
        logger.warning(f"KeyTao LLM batch review failed: {error}")
        return {"success": False, "message": str(error)}

    ai_review = _normalize_llm_review(raw_review, items, local_review)
    return {
        "success": True,
        "aiReview": ai_review,
        "reviewedAt": ai_review["generatedAt"],
    }
