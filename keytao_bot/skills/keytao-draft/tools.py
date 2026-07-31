"""
Keytao Create Skill Tools
键道创建词条工具实现
"""
import asyncio
import difflib
import hashlib
import hmac
import json
import re
import secrets
import time
import unicodedata
import httpx
from typing import Dict, List, Optional
from nonebot.log import logger

from keytao_bot.utils.keytao_encoding import (
    build_alternate_pronunciation_codes,
    build_phrase_pronunciation_codes,
    normalize_contextual_phrase_encoding,
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


def _should_validate_create_code(item: Dict) -> bool:
    action = item.get("action", "Create")
    if action != "Create":
        return False

    word = str(item.get("word") or "").strip()
    code = str(item.get("code") or "").strip().lower()
    if not word or not code or not re.fullmatch(r"[a-z]+", code):
        return False

    phrase_type = _infer_phrase_type(word, code, item.get("type") or "Phrase")
    return phrase_type in {"Phrase", "Single"} and _contains_cjk_text(word)


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
    if "needs_manual_review" in normalized:
        normalized["needsManualReview"] = bool(normalized.pop("needs_manual_review"))
    if "needsManualReview" not in normalized:
        normalized["needsManualReview"] = bool(
            manual_preaudit_issue_for_item(normalized)
        )
    return normalized


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
) -> Dict:
    if target_code not in target_candidate_codes:
        return {
            "success": False,
            "message": f"{target_code} 不是「{word}」的有效候选编码",
        }

    current_code = current_phrase.get("code") if current_phrase else None
    current_type = current_phrase.get("type", "Phrase") if current_phrase else "Phrase"
    deletes: List[Dict] = []
    creates: List[Dict] = [{"action": "Create", "word": word, "code": target_code, "type": current_type or "Phrase"}]
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
        return "请先登录 KeyTao 后再进行加词操作"
    return "未找到绑定账号，请先使用 /bind 命令绑定你的键道平台账号"


def get_keytao_url() -> str:
    """Get Keytao API base URL from config"""
    try:
        from nonebot import get_driver
        driver = get_driver()
        config = driver.config
        return getattr(config, "keytao_api_base", "https://keytao.vercel.app")
    except:
        return "https://keytao.vercel.app"


def make_batch_url(batch_id: str) -> str:
    """Build a web URL for a draft batch."""
    return f"{get_keytao_url()}/batch/{batch_id}"


def _inject_batch_url(data: Dict) -> Dict:
    """Inject batchUrl into any response dict that contains a batchId."""
    batch_id = data.get("batchId")
    if batch_id:
        data["batchUrl"] = make_batch_url(batch_id)
    return data


def get_bot_token() -> Optional[str]:
    """Get Bot API token from config"""
    try:
        from nonebot import get_driver
        driver = get_driver()
        config = driver.config
        return getattr(config, "bot_api_token", None)
    except:
        return None


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
    """Get a KeyTao user API key matching the bound platform account."""
    try:
        from nonebot import get_driver
        driver = get_driver()
        config = driver.config
        mapping = _parse_json_mapping(
            getattr(config, "keytao_user_api_keys", None)
            or getattr(config, "bot_user_api_keys", None)
        )
        for key in (
            f"{platform}:{platform_id}",
            platform_id,
            f"{platform}:default",
            "default",
        ):
            if mapping.get(key):
                return mapping[key]
        return (
            getattr(config, "keytao_api_key", None)
            or getattr(config, "bot_user_api_key", None)
        )
    except Exception:
        return None


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
            response = await client.get(
                url,
                headers=get_bot_headers(
                    platform,
                    platform_id,
                    method="GET",
                    path="/api/bot/batches/latest-draft",
                ),
                params={"platform": platform, "platformId": platform_id}
            )

            if response.status_code == 200:
                data = response.json()
                batch_id = data.get("batchId")
                logger.info(f"[get_latest_draft_batch] Got batch ID: {batch_id}")
                return batch_id
            elif response.status_code == 404:
                raise UserNotFoundError()
            else:
                logger.error(f"[get_latest_draft_batch] API error ({response.status_code}): {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"[get_latest_draft_batch] Error: {e}")
        return None


async def _fetch_draft_snapshot(platform: str, platform_id: str) -> Optional[Dict]:
    """Fetch current draft items and return as snapshot dict (best-effort, never raises)."""
    try:
        result = await keytao_list_draft_items(platform, platform_id)
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
            response = await client.get(encode_url, params=params)
            encode_data = normalize_contextual_phrase_encoding(
                word,
                response.json() if response.is_success else {},
            )
            codes = _clean_code_list(encode_data.get("codes"))
            alt_codes = _clean_code_list(encode_data.get("altCodes"))
            if not codes:
                infer_response = await client.get(infer_url, params=params)
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
        return {"success": False, "message": f"计算「{word}」编码超时"}
    except Exception as e:
        logger.error(f"[shift_encode] Error for {word}: {e}")
        return {"success": False, "message": f"计算「{word}」编码失败: {str(e)}"}


async def _validate_draft_item_code(item: Dict) -> Dict:
    """Ensure a Create item's code belongs to that word's encode candidate chain."""
    phrase_type = str(item.get("type") or "").strip()
    if phrase_type not in VALID_PHRASE_TYPES:
        return {
            "success": False,
            "word": str(item.get("word") or "").strip(),
            "code": str(item.get("code") or "").strip(),
            "reason": f"不支持的词库类型：{phrase_type or '(empty)'}",
        }
    if not _should_validate_create_code(item):
        return {"success": True, "skipped": True}

    word = str(item.get("word") or "").strip()
    code = str(item.get("code") or "").strip().lower()
    encoding = await _fetch_encode_candidates(word, code)
    if not encoding.get("success"):
        return {
            "success": False,
            "word": word,
            "code": code,
            "reason": encoding.get("message", "编码校验失败"),
            "candidateCodes": encoding.get("candidateCodes", []),
        }

    candidate_codes = encoding.get("candidateCodes", [])
    if code in candidate_codes:
        return {"success": True, "candidateCodes": candidate_codes}

    return {
        "success": False,
        "word": word,
        "code": code,
        "reason": f"编码 {code} 不是「{word}」的有效候选编码",
        "candidateCodes": candidate_codes,
        "requestedCodeAnalysis": encoding.get("requestedCodeAnalysis"),
    }


def _format_code_validation_failure(validation: Dict, index: int = 0) -> Dict:
    candidate_codes = validation.get("candidateCodes") or []
    reason = validation.get("reason", "编码校验失败")
    if candidate_codes:
        reason += f"；可选：{', '.join(candidate_codes[:8])}"
        if len(candidate_codes) > 8:
            reason += f" 等 {len(candidate_codes)} 个"
    failed = {
        "index": index,
        "word": validation.get("word", ""),
        "code": validation.get("code", ""),
        "reason": reason,
        "validationError": True,
    }
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
            return index, item, await _validate_draft_item_code(item)

    checked = await asyncio.gather(
        *(validate(index, item) for index, item in enumerate(normalized_items))
    )

    valid_items: List[Dict] = []
    failed_items: List[Dict] = []
    for index, item, validation in checked:
        if validation.get("success"):
            valid_items.append(item)
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
            response = await client.post(
                f"{KEYTAO_API_BASE}/api/bot/phrases/by-word/batch",
                headers={"X-Bot-Token": BOT_API_TOKEN, "Content-Type": "application/json"},
                json={"words": words},
            )
            data = response.json()
            if not data.get("success"):
                return {"success": False, "message": data.get("message", "按词查询失败")}
            return {"success": True, "results": data.get("results", [])}
    except httpx.TimeoutException:
        return {"success": False, "message": "按词查询超时"}
    except Exception as e:
        return {"success": False, "message": f"按词查询失败: {str(e)}"}


async def _lookup_codes_raw(codes: List[str]) -> Dict:
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()
    if not BOT_API_TOKEN:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{KEYTAO_API_BASE}/api/bot/phrases/by-code/batch",
                headers={"X-Bot-Token": BOT_API_TOKEN, "Content-Type": "application/json"},
                json={"codes": codes},
            )
            data = response.json()
            if not data.get("success"):
                return {"success": False, "message": data.get("message", "按编码查询失败")}
            return {"success": True, "results": data.get("results", [])}
    except httpx.TimeoutException:
        return {"success": False, "message": "按编码查询超时"}
    except Exception as e:
        return {"success": False, "message": f"按编码查询失败: {str(e)}"}


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
    if not batch_id:
        return {"success": False, "message": "无法获取草稿批次，请稍后重试"}
    if confirmed and (
        not isinstance(expected_content_version, int)
        or isinstance(expected_content_version, bool)
        or expected_content_version < 0
        or not re.fullmatch(r"[0-9a-f]{64}", expected_warning_digest)
    ):
        return {"success": False, "message": "添加确认缺少有效的服务端风险快照"}

    # Auto-detect type when not explicitly specified, mirrors detectPhraseType in keytao-next
    type = _infer_phrase_type(word, code, type)
    if type not in VALID_PHRASE_TYPES:
        return {"success": False, "message": f"不支持的词库类型：{type}"}
    if needs_manual_review is None:
        needs_manual_review = bool(manual_preaudit_issue_for_item({
            "word": word,
            "remark": remark,
        }))
    validation = await _validate_draft_item_code({
        "action": action,
        "word": word,
        "code": code,
        "type": type,
    })
    if not validation.get("success"):
        failed = _format_code_validation_failure(validation)
        return {
            "success": False,
            "message": failed["reason"],
            "failed": [failed],
            "failedCount": 1,
            "batchId": batch_id,
            "batchUrl": make_batch_url(batch_id),
        }

    url = f"{KEYTAO_API_BASE}/api/bot/pull-requests/batch"
    
    request_data = {
        "platform": platform,
        "platformId": platform_id,
        "items": [{
            "action": action,
            "word": word,
            "oldWord": old_word,
            "code": code,
            "type": type,
            "remark": remark,
            "needsManualReview": bool(needs_manual_review),
        }],
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
            response = await client.post(
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
            )
            
            if response.status_code == 200:
                data = response.json()
                data.setdefault("batchId", batch_id)
                if preview_only and not data.get("requiresConfirmation"):
                    return {
                        "success": False,
                        "uncertain": True,
                        "message": "服务端未返回可确认的添加快照，已停止后续操作",
                        "batchId": batch_id,
                    }
                logger.info(f"[keytao_create_phrase] API response (200): {json.dumps(data, ensure_ascii=False)}")
                snapshot = await _fetch_draft_snapshot(platform, platform_id)
                if snapshot is not None:
                    data["draft_snapshot"] = snapshot
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
                    snapshot = await _fetch_draft_snapshot(platform, platform_id)
                    if snapshot is not None:
                        data["draft_snapshot"] = snapshot
                    _inject_batch_url(data)
                return data
            else:
                logger.error(f"[keytao_create_phrase] API response ({response.status_code}): {response.text}")
                return {
                    "success": False,
                    "message": f"创建失败: HTTP {response.status_code}"
                }
                
    except httpx.TimeoutException:
        return {
            "success": False,
            "message": "请求超时，请稍后重试"
        }
    except Exception as e:
        logger.error(f"Create phrase error: {e}")
        return {
            "success": False,
            "message": f"创建失败: {str(e)}"
        }


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
                str(item.get("word") or f"PR#{item.get('id')}")
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
            title = item.get("title") or f"PR#{item.get('prId')} 需要复核"
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


def _audit_allows_batch_auto_approve(auto_review: Dict) -> bool:
    """Require an internally consistent all-pass result before calling approval."""
    return bool(
        isinstance(auto_review, dict)
        and auto_review.get("autoApprove") is True
        and auto_review.get("verdict") == "pass"
        and not (auto_review.get("issues") or [])
        and bool(auto_review.get("approvedItems") or [])
        and not auto_review.get("encodeOnly")
    )


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
    url = f"{KEYTAO_API_BASE}/api/bot/batches/{batch_id}/auto-approve"
    request_data = {
        "platform": platform,
        "platformId": platform_id,
        "reviewNote": review_note,
        "expectedContentVersion": expected_content_version,
    }
    request_body = _json_request_body(request_data)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers=get_bot_headers(
                    platform,
                    platform_id,
                    content_type=True,
                    method="POST",
                    path=f"/api/bot/batches/{batch_id}/auto-approve",
                    raw_body=request_body,
                ),
                content=request_body,
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
        return {"success": False, "message": "批次已提交，自动批准超时，转交管理员审核"}
    except Exception as error:
        logger.warning(f"[auto_review] approve failed: {error}")
        return {"success": False, "message": f"自动批准失败：{str(error)}"}


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
        return {
            "success": False,
            "message": "提交确认缺少有效的批次版本，请重新获取最新提交检查结果",
        }

    # The first request may resolve the latest draft. A confirmation must use
    # the exact batch/version returned by that first server warning.
    if not batch_id:
        try:
            batch_id = await get_latest_draft_batch(platform, platform_id)
        except UserNotFoundError:
            return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
    if not batch_id:
        return {"success": False, "message": "没有找到待提交的草稿批次"}

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
        return {
            "success": False,
            "message": auto_review.get("summary") or "草稿快照缺少可验证版本，已停止提交",
            "autoReview": auto_review,
        }
    if confirmed and expected_content_version != audited_content_version:
        return {
            "success": False,
            "staleConfirmation": True,
            "message": "草稿内容已变化，旧确认票据已作废，请重新提交",
            "batchId": batch_id,
            "contentVersion": audited_content_version,
            "autoReview": auto_review,
        }
    if confirmed and expected_audit_digest != audited_digest:
        return {
            "success": False,
            "staleConfirmation": True,
            "message": "本喵复审结论已变化，旧确认票据已作废，请重新提交",
            "batchId": batch_id,
            "contentVersion": audited_content_version,
            "auditDigest": audited_digest,
            "autoReview": auto_review,
        }
    submission_content_version = audited_content_version
    
    url = f"{KEYTAO_API_BASE}/api/bot/batches/{batch_id}/submit"
    
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
            response = await client.post(
                url,
                headers=get_bot_headers(
                    platform,
                    platform_id,
                    content_type=True,
                    method="POST",
                    path=f"/api/bot/batches/{batch_id}/submit",
                    raw_body=request_body,
                ),
                content=request_body,
            )
            
            if response.status_code == 200:
                data = response.json()
                data["batchId"] = batch_id  # inject so _inject_batch_url can build batchUrl
                data.setdefault("contentVersion", submission_content_version)
                _inject_batch_url(data)
                data["autoReview"] = auto_review
                data["auditDigest"] = audited_digest
                data.setdefault("snapshotItems", auto_review.get("snapshotItems", []))
                data["auditSnapshotDigest"] = auto_review.get("snapshotDigest", "")
                if preview_only:
                    data["success"] = False
                    data["requiresConfirmation"] = True
                    return data
                if _audit_allows_batch_auto_approve(auto_review):
                    approve_result = await _auto_approve_submitted_batch(
                        platform,
                        platform_id,
                        batch_id,
                        auto_review,
                        submission_content_version,
                    )
                    data["autoApproveResult"] = approve_result
                    data["autoApproved"] = bool(approve_result.get("success"))
                return data
            elif response.status_code == 404:
                return {
                    "success": False,
                    "message": "批次不存在或已被删除"
                }
            elif response.status_code == 403:
                return {
                    "success": False,
                    "message": "无权限操作此批次"
                }
            elif response.status_code == 400:
                data = response.json()
                data.setdefault("batchId", batch_id)
                data.setdefault("contentVersion", submission_content_version)
                data["autoReview"] = auto_review
                data["auditDigest"] = audited_digest
                data.setdefault("snapshotItems", auto_review.get("snapshotItems", []))
                data["auditSnapshotDigest"] = auto_review.get("snapshotDigest", "")
                return data
            elif response.status_code == 409:
                data = response.json()
                return {
                    **data,
                    "success": False,
                    "staleConfirmation": True,
                    "batchId": batch_id,
                    "message": data.get("message") or "草稿内容已变化，请重新检查后提交",
                }
            else:
                return {
                    "success": False,
                    "message": f"提交失败: HTTP {response.status_code}"
                }
                
    except httpx.TimeoutException:
        return {
            "success": False,
            "message": "请求超时，请稍后重试"
        }
    except Exception as e:
        logger.error(f"Submit batch error: {e}")
        return {
            "success": False,
            "message": f"提交失败: {str(e)}"
        }


async def keytao_get_batch_preview(
    platform: str,
    platform_id: str,
) -> Dict:
    """
    Fetch the diff preview of the user's current draft batch.
    Returns summary stats and a formatted unified-diff text block.
    """
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()

    if not BOT_API_TOKEN:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}

    try:
        batch_id = await get_latest_draft_batch(platform, platform_id)
    except UserNotFoundError:
        return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
    if not batch_id:
        return {"success": False, "message": "没有找到草稿批次"}

    url = f"{KEYTAO_API_BASE}/api/batches/{batch_id}/preview"
    logger.info(f"[keytao_get_batch_preview] batchId={batch_id}")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)

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
        return {"success": False, "message": "请求超时，请稍后重试"}
    except Exception as e:
        logger.error(f"[keytao_get_batch_preview] Error: {e}")
        return {"success": False, "message": f"获取预览失败: {str(e)}"}


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

    url = f"{KEYTAO_API_BASE}/api/bot/batches/recall"
    logger.info(f"[keytao_recall_batch] platform={platform} platformId={platform_id}")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if not batch_id:
                response = await client.get(
                    url,
                    headers=get_bot_headers(
                        platform,
                        platform_id,
                        method="GET",
                        path="/api/bot/batches/recall",
                    ),
                    params={"platform": platform, "platformId": platform_id},
                )
                try:
                    data = response.json()
                except Exception:
                    return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}
                if not response.is_success:
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
                    return {"success": False, "message": "待撤回批次缺少可验证版本"}
                return {
                    **data,
                    "success": False,
                    "requiresConfirmation": True,
                    "confirmationKind": "recallBatch",
                    "message": "即将撤回这个已提交批次并恢复为草稿",
                }
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
            request_body = _json_request_body(request_data)
            response = await client.post(
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
            )
            try:
                data = response.json()
            except Exception:
                return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}

            logger.info(f"[keytao_recall_batch] status={response.status_code} success={data.get('success')}")
            if response.status_code == 409:
                return {
                    **data,
                    "success": False,
                    "staleConfirmation": True,
                    "message": data.get("message") or "待撤回批次已变化，旧票据已作废",
                }
            _inject_batch_url(data)
            return data

    except httpx.TimeoutException:
        return {"success": False, "message": "请求超时，请稍后重试"}
    except Exception as e:
        logger.error(f"[keytao_recall_batch] Error: {e}")
        return {"success": False, "message": f"撤回失败: {str(e)}"}


async def keytao_list_draft_items(
    platform: str,
    platform_id: str,
    batch_id: Optional[str] = None,
) -> Dict:
    """
    List all PR items in the user's latest draft batch
    列出用户最新草稿批次中的所有条目
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
            response = await client.get(
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
        return {"success": False, "message": "请求超时，请稍后重试"}
    except httpx.TransportError as e:
        logger.error(f"List draft items network error: {type(e).__name__}: {e!r}")
        return {"success": False, "message": f"网络错误: {type(e).__name__}"}
    except Exception as e:
        logger.error(f"List draft items error: {type(e).__name__}: {e!r}")
        return {"success": False, "message": f"获取失败: {type(e).__name__}: {e}"}


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
        return {"success": False, "message": "草稿快照缺少可验证版本"}
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
        return {
            "success": False,
            "staleConfirmation": True,
            "message": f"草稿条目已变化，找不到 ID：{missing_ids}",
        }
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
    return {
        "success": True,
        "batchId": current_batch_id,
        "contentVersion": content_version,
        "targets": targets,
        "targetDigest": digest,
    }


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
        return {
            "success": False,
            "staleConfirmation": True,
            "message": "删除目标或草稿版本已变化，旧确认票据已作废",
        }

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
            response = await client.request(
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
            )

            try:
                data = response.json()
            except Exception:
                logger.error(f"[keytao_remove_draft_item] Non-JSON response ({response.status_code}): {response.text[:200]}")
                return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}

            logger.info(f"[keytao_remove_draft_item] PR#{pr_id} status={response.status_code}")
            if response.status_code == 409:
                return {
                    **data,
                    "success": False,
                    "staleConfirmation": True,
                    "message": data.get("message") or "删除目标已变化，旧票据已作废",
                }
            if data.get("success"):
                snapshot = await _fetch_draft_snapshot(platform, platform_id)
                if snapshot is not None:
                    data["draft_snapshot"] = snapshot
            _inject_batch_url(data)
            return data

    except httpx.TimeoutException:
        return {"success": False, "message": "请求超时，请稍后重试"}
    except Exception as e:
        logger.error(f"Remove draft item error: {e}")
        return {"success": False, "message": f"删除失败: {str(e)}"}


# Tool definitions for OpenAI Function Calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "keytao_create_phrase",
            "description": "创建、修改或删除键道词条。用于用户希望添加、修改或删除词条时。支持检测冲突和警告，如有重码警告可确认后创建。自动追加到草稿批次。",
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
                        "description": "⚠️ 重要：当工具首次返回警告（requiresConfirmation=true）后，用户确认时必须设置为true！不设置此参数会导致无限循环警告。默认false"
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
                "properties": {},
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
        if not batch_id:
            return {"success": False, "message": "无法获取草稿批次，请稍后重试"}
    if confirmed and (
        not isinstance(expected_content_version, int)
        or isinstance(expected_content_version, bool)
        or expected_content_version < 0
        or not re.fullmatch(r"[0-9a-f]{64}", expected_warning_digest)
    ):
        return {"success": False, "message": "批量添加确认缺少有效的服务端风险快照"}

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
        snapshot = await _fetch_draft_snapshot(platform, platform_id)
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
            response = await client.post(
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
            )
            try:
                data = response.json()
            except Exception:
                return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}

            logger.info(
                f"[keytao_batch_add_to_draft] status={response.status_code} "
                f"success={data.get('successCount',0)} failed={data.get('failedCount',0)}"
            )
            # Enrich draft item labels
            if isinstance(data.get("draftItems"), list):
                data["draftItems"] = [enrich_pr_item_labels(item) for item in data["draftItems"]]
            data.setdefault("batchId", batch_id)
            if preview_only and not data.get("requiresConfirmation"):
                return {
                    "success": False,
                    "uncertain": True,
                    "message": "服务端未返回可确认的批量添加快照，已停止后续操作",
                    "batchId": batch_id,
                }
            if validation_failed:
                data["failed"] = [*data.get("failed", []), *validation_failed]
                data["failedCount"] = data.get("failedCount", 0) + len(validation_failed)
                data["message"] = (
                    f"{data.get('message', '已处理草稿')}；"
                    f"{len(validation_failed)} 条编码校验失败未写入"
                )
            _inject_batch_url(data)
            return data

    except httpx.TimeoutException:
        return {"success": False, "message": "请求超时，请稍后重试"}
    except Exception as e:
        logger.error(f"[keytao_batch_add_to_draft] Error: {e}")
        return {"success": False, "message": f"批量添加失败: {str(e)}"}


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
        return {
            "success": False,
            "staleConfirmation": True,
            "message": "批量删除目标或草稿版本已变化，旧确认票据已作废",
        }

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
            response = await client.request(
                "DELETE", url,
                content=request_body,
                headers=get_bot_headers(
                    platform,
                    platform_id,
                    content_type=True,
                    method="DELETE",
                    path="/api/bot/pull-requests/batch-draft",
                    raw_body=request_body,
                ),
            )
            try:
                data: Dict = response.json()
            except Exception:
                return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}
            logger.info(
                f"[keytao_batch_remove_draft_items] status={response.status_code} "
                f"success={data.get('success')} deleted={data.get('successCount')}"
            )
            if response.status_code == 409:
                return {
                    **data,
                    "success": False,
                    "staleConfirmation": True,
                    "message": data.get("message") or "批量删除目标已变化，旧票据已作废",
                }
            if isinstance(data.get("draftItems"), list):
                data["draftItems"] = [enrich_pr_item_labels(item) for item in data["draftItems"]]
            _inject_batch_url(data)
            return data
    except httpx.TimeoutException:
        return {"success": False, "message": "请求超时，请稍后重试"}
    except Exception as e:
        logger.error(f"[keytao_batch_remove_draft_items] Error: {e}")
        return {"success": False, "message": f"批量删除失败: {str(e)}"}


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

    if not batch_id:
        try:
            batch_id = await get_latest_draft_batch(platform, platform_id)
        except UserNotFoundError:
            return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
    if not batch_id:
        return {"success": False, "message": "无法获取草稿批次，请稍后重试"}

    valid_items, validation_failed = await _split_items_by_code_validation(items)
    if validation_failed or len(valid_items) != len(items):
        return {
            "success": False,
            "message": "顺延计划未通过整批编码预检，未写入任何草稿条目",
            "failed": validation_failed,
        }

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
        "batchId": batch_id,
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
            response = await client.post(
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
            )
        try:
            data = response.json()
        except Exception:
            return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}
        if response.status_code == 409:
            return {
                **data,
                "success": False,
                "staleConfirmation": True,
                "message": data.get("message") or "草稿内容已变化，顺延计划已作废",
            }
        if data.get("requiresConfirmation"):
            return data
        if not response.is_success or not data.get("success"):
            return {
                **data,
                "success": False,
                "message": data.get("message") or f"整批顺延写入失败（HTTP {response.status_code}）",
            }
        data["successCount"] = int(
            data.get("successCount")
            or data.get("pullRequestCount")
            or len(request_items)
        )
        snapshot = await _fetch_draft_snapshot(platform, platform_id)
        if snapshot is not None:
            data["draft_snapshot"] = snapshot
        _inject_batch_url(data)
        return data
    except httpx.TimeoutException:
        return {
            "success": False,
            "uncertain": True,
            "message": "整批顺延请求超时；请先查看草稿确认状态，不要立即重试",
        }
    except Exception as error:
        logger.error(f"[keytao_shift_phrase_code] strict batch error: {error}")
        return {"success": False, "message": f"整批顺延写入失败: {str(error)}"}


async def keytao_shift_phrase_code(
    platform: str,
    platform_id: str,
    word: str,
    target_code: str,
    confirmed_plan_digest: str = "",
    batch_id: Optional[str] = None,
    expected_content_version: Optional[int] = None,
    expected_warning_digest: str = "",
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
        return {
            "success": False,
            "policyBlocked": True,
            "requiresDraftCleanup": True,
            "message": (
                "相关词条已存在于草稿中；为避免非原子地先删后写，"
                "本次顺延未修改草稿。请先明确处理这些旧草稿条目，再重新发起顺延。"
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
        }

    current_batch_id = str(existing_draft.get("batchId") or "")
    current_content_version = existing_draft.get("contentVersion")
    if (
        not current_batch_id
        or not isinstance(current_content_version, int)
        or isinstance(current_content_version, bool)
        or current_content_version < 0
    ):
        return {
            "success": False,
            "message": "当前草稿缺少可验证的内容版本，顺延未执行",
        }

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
        return {
            "success": False,
            "requiresConfirmation": True,
            "confirmationKind": "shiftPlan",
            "message": "顺延会移动当前编码链中的其他词条，请核对完整计划",
            "batchId": current_batch_id,
            "contentVersion": current_content_version,
            "planDigest": plan_digest,
            "shiftPlan": shift_plan,
        }
    if (
        confirmed_plan_digest.strip().lower() != plan_digest
        or batch_id != current_batch_id
        or expected_content_version != current_content_version
    ):
        return {
            "success": False,
            "staleConfirmation": True,
            "message": "顺延计划或草稿内容已变化，旧确认票据已作废，请重新发起",
            "batchId": current_batch_id,
            "contentVersion": current_content_version,
        }

    write_result = await _keytao_strict_batch_add_to_draft(
        platform,
        platform_id,
        plan.get("items", []),
        batch_id=current_batch_id,
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
    return write_result


TOOLS += [
    {
        "type": "function",
        "function": {
            "name": "keytao_batch_add_to_draft",
            "description": (
                "批量将词条加入草稿。适合用户一次提交大量词条时使用。"
                "首次调用只返回完整预览，不写入；所有条目和风险（包括重码、跳过更短空位）都必须由用户确认票据。"
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
                "properties": {},
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
    "keytao_batch_add_to_draft": keytao_batch_add_to_draft,
    "keytao_batch_remove_draft_items": keytao_batch_remove_draft_items,
    "keytao_shift_phrase_code": keytao_shift_phrase_code,
    "keytao_recall_batch": keytao_recall_batch,
    "keytao_get_batch_preview": keytao_get_batch_preview,
}
