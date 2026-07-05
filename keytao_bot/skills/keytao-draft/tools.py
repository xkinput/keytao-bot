"""
Keytao Create Skill Tools
键道创建词条工具实现
"""
import asyncio
import difflib
import json
import re
import unicodedata
import httpx
from typing import Dict, List, Optional
from nonebot.log import logger

from keytao_bot.utils.keytao_encoding import (
    build_alternate_pronunciation_codes,
    build_phrase_pronunciation_codes,
)
from keytao_bot.utils.keytao_review import (
    ReviewHttpConfig,
    audit_draft_items,
    build_review_note,
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
    if phrase_type and phrase_type != "Phrase":
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

    if not normalized.get("type") and isinstance(normalized.get("word"), str) and isinstance(normalized.get("code"), str):
        normalized["type"] = _infer_phrase_type(
            normalized["word"],
            normalized["code"],
            "Phrase",
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
                headers=get_bot_headers(platform, platform_id),
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
            encode_data = response.json() if response.is_success else {}
            codes = _clean_code_list(encode_data.get("codes"))
            alt_codes = _clean_code_list(encode_data.get("altCodes"))
            if not codes:
                infer_response = await client.get(infer_url, params=params)
                infer_data = infer_response.json() if infer_response.is_success else {}
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
    confirmed: bool = False
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
    try:
        batch_id = await get_latest_draft_batch(platform, platform_id)
    except UserNotFoundError:
        return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
    if not batch_id:
        return {"success": False, "message": "无法获取草稿批次，请稍后重试"}

    # Auto-detect type when not explicitly specified, mirrors detectPhraseType in keytao-next
    type = _infer_phrase_type(word, code, type)
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
            "remark": remark
        }],
        "confirmed": confirmed,
        "batchId": batch_id  # Always use the draft batch
    }
    
    logger.info(f"[keytao_create_phrase] Sending request: {json.dumps(request_data, ensure_ascii=False)}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=get_bot_headers(platform, platform_id, content_type=True),
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
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


async def _audit_current_draft_for_auto_approval(platform: str, platform_id: str) -> Dict:
    try:
        list_result = await keytao_list_draft_items(platform, platform_id)
        if not list_result.get("success"):
            return {
                "success": False,
                "verdict": "needs_admin",
                "autoApprove": False,
                "summary": list_result.get("message", "无法读取草稿，提交后等待管理员审核"),
                "issues": [list_result.get("message", "无法读取草稿")],
            }
        deterministic_audit = await audit_draft_items(_review_config(), list_result.get("items", []))
        if deterministic_audit.get("autoApprove") or not _can_llm_override_audit_issues(deterministic_audit):
            return deterministic_audit
        llm_audit = await _try_llm_auto_review_for_draft(list_result, deterministic_audit)
        return llm_audit or deterministic_audit
    except Exception as error:
        logger.warning(f"[auto_review] audit failed: {error}")
        return {
            "success": False,
            "verdict": "needs_admin",
            "autoApprove": False,
            "summary": "自动审核异常，提交后等待管理员审核",
            "issues": [str(error)],
        }


def _can_llm_override_audit_issues(audit: Dict) -> bool:
    issues = audit.get("issues") or []
    if not issues:
        return False
    allowed_fragments = (
        "没有权威读音来源",
        "常用词信号不足",
        "常用度证据不足",
        "可比较的常用度信号不足",
        "声笔笔短码",
        "声笔笔短码表",
    )
    blocked_fragments = (
        "纯删除",
        "不在读音候选链",
        "不在权威读音候选链",
        "改词",
        "歧义",
        "审词失败",
        "词或编码为空",
    )
    return all(
        any(fragment in issue for fragment in allowed_fragments)
        and not any(fragment in issue for fragment in blocked_fragments)
        for issue in issues
    )


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
        review_result = await review_keytao_batch_with_llm(batch)
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
                    approved_items.append(f"{action}：{word}@{code}，本喵 LLM 复审通过")
            return {
                **deterministic_audit,
                "success": True,
                "verdict": "pass",
                "autoApprove": True,
                "summary": ai_review.get("headline") or "本喵已结合语言常识完成复审，允许自动通过",
                "issues": [],
                "approvedItems": approved_items or deterministic_audit.get("approvedItems", []),
                "llmReview": ai_review,
                "llmFallback": True,
            }

        issues = []
        for item in non_pass_items[:10]:
            reasons = item.get("reasons") if isinstance(item.get("reasons"), list) else []
            title = item.get("title") or f"PR#{item.get('prId')} 需要复核"
            reason = reasons[0] if reasons else title
            issues.append(str(reason))

        return {
            **deterministic_audit,
            "summary": ai_review.get("headline") or deterministic_audit.get("summary", "存在不确定项，提交后等待管理员审核"),
            "issues": issues or deterministic_audit.get("issues", []),
            "llmReview": ai_review,
            "llmFallback": True,
        }
    except Exception as error:
        logger.warning(f"[auto_review] LLM fallback error: {error}")
        return None


async def _auto_approve_submitted_batch(
    platform: str,
    platform_id: str,
    batch_id: str,
    auto_review: Dict,
) -> Dict:
    KEYTAO_API_BASE = get_keytao_url()
    review_note = build_review_note(auto_review)
    url = f"{KEYTAO_API_BASE}/api/bot/batches/{batch_id}/auto-approve"
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers=get_bot_headers(platform, platform_id, content_type=True),
                json={
                    "platform": platform,
                    "platformId": platform_id,
                    "reviewNote": review_note,
                },
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
        return {"success": False, "message": "自动批准请求超时，批次已提交等待管理员审核"}
    except Exception as error:
        logger.warning(f"[auto_review] approve failed: {error}")
        return {"success": False, "message": f"自动批准失败：{str(error)}"}


async def keytao_submit_batch(
    platform: str,
    platform_id: str,
    confirmed: bool = False
) -> Dict:
    """
    Submit current draft batch for review
    提交当前草稿批次进行审核
    
    Automatically finds and submits the user's latest draft batch.
    自动查找并提交用户的最新草稿批次。
    
    Args:
        platform: Platform type ('qq' or 'telegram')
        platform_id: User's platform ID
        confirmed: Whether to confirm duplicate code / multiple code warnings
        
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
    
    # Get draft batch ID
    try:
        batch_id = await get_latest_draft_batch(platform, platform_id)
    except UserNotFoundError:
        return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
    if not batch_id:
        return {"success": False, "message": "没有找到待提交的草稿批次"}

    auto_review = await _audit_current_draft_for_auto_approval(platform, platform_id)
    
    url = f"{KEYTAO_API_BASE}/api/bot/batches/{batch_id}/submit"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=get_bot_headers(platform, platform_id, content_type=True),
                json={
                    "platform": platform,
                    "platformId": platform_id,
                    "confirmed": confirmed
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                data["batchId"] = batch_id  # inject so _inject_batch_url can build batchUrl
                _inject_batch_url(data)
                data["autoReview"] = auto_review
                if auto_review.get("autoApprove"):
                    approve_result = await _auto_approve_submitted_batch(
                        platform,
                        platform_id,
                        batch_id,
                        auto_review,
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
                data["autoReview"] = auto_review
                return data
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
            response = await client.post(
                url,
                headers=get_bot_headers(platform, platform_id, content_type=True),
                json={"platform": platform, "platformId": platform_id},
            )
            try:
                data = response.json()
            except Exception:
                return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}

            logger.info(f"[keytao_recall_batch] status={response.status_code} success={data.get('success')}")
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
                headers=get_bot_headers(platform, platform_id),
                params={"platform": platform, "platformId": platform_id}
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


async def keytao_remove_draft_item(
    platform: str,
    platform_id: str,
    pr_id: int,
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

    url = f"{KEYTAO_API_BASE}/api/bot/pull-requests/{pr_id}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.request(
                "DELETE",
                url,
                headers=get_bot_headers(platform, platform_id, content_type=True),
                content=json.dumps({"platform": platform, "platformId": platform_id})
            )

            try:
                data = response.json()
            except Exception:
                logger.error(f"[keytao_remove_draft_item] Non-JSON response ({response.status_code}): {response.text[:200]}")
                return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}

            logger.info(f"[keytao_remove_draft_item] PR#{pr_id} status={response.status_code}")
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
) -> Dict:
    """
    Batch add word entries to draft (tolerant mode).
    Hard conflicts are skipped and reported; duplicate-code warnings are auto-confirmed.

    Args:
        platform: Platform type ('qq' or 'telegram')
        platform_id: User's platform ID
        items: List of dicts with keys: word, code, action (optional), type (optional), remark (optional)
        batch_id: Optional existing draft batch ID
        confirmed: Whether to confirm warnings that must pause before writing

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

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers=get_bot_headers(platform, platform_id, content_type=True),
                json=request_data,
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
) -> Dict:
    """Batch delete draft items by their PR IDs."""
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()

    if not BOT_API_TOKEN:
        return {"success": False, "message": "喵喵配置错误：缺少API token"}

    url = f"{KEYTAO_API_BASE}/api/bot/pull-requests/batch-draft"
    payload = {"platform": platform, "platformId": platform_id, "ids": ids}
    logger.info(f"[keytao_batch_remove_draft_items] Deleting ids={ids}")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                "DELETE", url,
                json=payload,
                headers=get_bot_headers(platform, platform_id, content_type=True),
            )
            try:
                data: Dict = response.json()
            except Exception:
                return {"success": False, "message": f"API 返回异常（HTTP {response.status_code}）"}
            logger.info(
                f"[keytao_batch_remove_draft_items] status={response.status_code} "
                f"success={data.get('success')} deleted={data.get('successCount')}"
            )
            if isinstance(data.get("draftItems"), list):
                data["draftItems"] = [enrich_pr_item_labels(item) for item in data["draftItems"]]
            _inject_batch_url(data)
            return data
    except httpx.TimeoutException:
        return {"success": False, "message": "请求超时，请稍后重试"}
    except Exception as e:
        logger.error(f"[keytao_batch_remove_draft_items] Error: {e}")
        return {"success": False, "message": f"批量删除失败: {str(e)}"}


async def keytao_shift_phrase_code(
    platform: str,
    platform_id: str,
    word: str,
    target_code: str,
) -> Dict:
    """Move a word to a target code and shift occupants using each occupant's own encode chain."""
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
    existing_draft = await keytao_list_draft_items(platform, platform_id)
    removed_draft_ids: List[int] = []
    if existing_draft.get("success"):
        for item in existing_draft.get("items", []):
            if item.get("word") in planned_words and isinstance(item.get("id"), int):
                removed_draft_ids.append(item["id"])
    if removed_draft_ids:
        remove_result = await keytao_batch_remove_draft_items(platform, platform_id, removed_draft_ids)
        if not remove_result.get("success"):
            return remove_result

    write_result = await keytao_batch_add_to_draft(platform, platform_id, plan.get("items", []))
    write_result["shiftPlan"] = {
        "word": word,
        "targetCode": target_code,
        "candidateCodes": target_candidate_codes,
        "items": plan.get("items", []),
        "shifted": plan.get("shifted", []),
        "removedDraftIds": removed_draft_ids,
    }
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
                "遇到冲突的条目会跳过并在 failed 列表中说明原因，重码（同编码不同词）会自动确认写入；"
                "跳过更短空位编码会返回 requiresConfirmation，用户确认后再传 confirmed=true 写入。"
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
                                    "description": "词条类型。用户明确指定类型时必须传：声笔笔=CSS，声笔笔单字=CSSSingle，词组=Phrase，单字=Single，补充=Supplement，符号=Symbol，链接=Link，英文=English。Change/Delete 若不传会默认词组，可能改错词库。",
                                },
                                "remark": {"type": "string", "description": "备注（可选）"},
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
                "自动清理相关草稿条目后一次性写入 Delete+Create，并返回 shiftPlan.shifted 说明顺延了哪些词。"
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
