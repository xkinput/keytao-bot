"""
Keytao Create Skill Tools
键道创建词条工具实现
"""
import json
import difflib
import httpx
from typing import Dict, List, Optional
from nonebot.log import logger


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


def _select_current_phrase(word: str, phrases: List[Dict]) -> Optional[Dict]:
    matching = [phrase for phrase in phrases if phrase.get("word") == word and phrase.get("code")]
    if not matching:
        return None
    return sorted(matching, key=lambda item: (len(item.get("code", "")), item.get("code", "")))[0]


def _select_code_occupant(phrases: List[Dict], ignored_word: str) -> Optional[Dict]:
    candidates = [phrase for phrase in phrases if phrase.get("word") and phrase.get("word") != ignored_word]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item.get("weight", 0), item.get("word", "")))[0]


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
    vacated_codes = {current_code} if current_code and current_code != target_code else set()
    visited_codes = set()
    probe_code = target_code

    if current_code and current_code != target_code:
        deletes.append({"action": "Delete", "word": word, "code": current_code, "type": current_type or "Phrase"})

    while probe_code not in vacated_codes:
        if probe_code in visited_codes:
            return {"success": False, "message": f"顺延计算出现循环：{probe_code}"}
        visited_codes.add(probe_code)

        occupant = _select_code_occupant(code_phrase_map.get(probe_code, []), word)
        if not occupant:
            break

        occupant_word = occupant.get("word", "")
        occupant_codes = word_candidate_code_map.get(occupant_word, [])
        if probe_code not in occupant_codes:
            return {
                "success": False,
                "message": f"无法顺延「{occupant_word}」：当前编码 {probe_code} 不在它自己的候选编码中",
            }

        code_index = occupant_codes.index(probe_code)
        if code_index + 1 >= len(occupant_codes):
            return {
                "success": False,
                "message": f"无法顺延「{occupant_word}」：{probe_code} 之后没有可用候选编码",
            }

        next_code = occupant_codes[code_index + 1]
        occupant_type = occupant.get("type", "Phrase") or "Phrase"
        deletes.append({"action": "Delete", "word": occupant_word, "code": probe_code, "type": occupant_type})
        creates.append({"action": "Create", "word": occupant_word, "code": next_code, "type": occupant_type})
        shifted.append({
            "word": occupant_word,
            "fromCode": probe_code,
            "toCode": next_code,
            "candidateCodes": occupant_codes,
        })
        vacated_codes.add(probe_code)
        probe_code = next_code

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
                headers={"X-Bot-Token": BOT_API_TOKEN},
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
                codes = _clean_code_list(infer_data.get("codes"))
                alt_codes = _clean_code_list(infer_data.get("altCodes"))
                requested_analysis = infer_data.get("requestedCodeAnalysis")
            else:
                requested_analysis = encode_data.get("requestedCodeAnalysis")
            candidate_codes = _clean_code_list([*codes, *alt_codes])
            if not candidate_codes:
                return {"success": False, "message": f"无法计算「{word}」的候选编码"}
            result = {"success": True, "word": word, "candidateCodes": candidate_codes}
            if requested_analysis:
                result["requestedCodeAnalysis"] = requested_analysis
            return result
    except httpx.TimeoutException:
        return {"success": False, "message": f"计算「{word}」编码超时"}
    except Exception as e:
        logger.error(f"[shift_encode] Error for {word}: {e}")
        return {"success": False, "message": f"计算「{word}」编码失败: {str(e)}"}


async def _lookup_words_raw(words: List[str]) -> Dict:
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()
    if not BOT_API_TOKEN:
        return {"success": False, "message": "Bot配置错误：缺少API token"}

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
        return {"success": False, "message": "Bot配置错误：缺少API token"}

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
            "message": "Bot配置错误：缺少API token"
        }
    
    # Get or create draft batch
    try:
        batch_id = await get_latest_draft_batch(platform, platform_id)
    except UserNotFoundError:
        return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
    if not batch_id:
        return {"success": False, "message": "无法获取草稿批次，请稍后重试"}

    # Auto-detect type when not explicitly specified, mirrors detectPhraseType in keytao-next
    if type == "Phrase":
        import re, unicodedata
        is_symbol_word = word and all(
            unicodedata.category(c).startswith(('P', 'S')) for c in word if not c.isspace()
        )
        if (code and code.startswith(';')) or is_symbol_word:
            type = "Symbol"
        elif re.search(r'https?://|www\.', word, re.IGNORECASE):
            type = "Link"
        elif re.search(r'[a-zA-Z]', word):
            type = "English"
        elif len(word) == 1 and re.match(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', word):
            type = "Single"

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
                headers={
                    "X-Bot-Token": BOT_API_TOKEN,
                    "Content-Type": "application/json"
                },
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
            "message": "Bot配置错误：缺少API token"
        }
    
    # Get draft batch ID
    try:
        batch_id = await get_latest_draft_batch(platform, platform_id)
    except UserNotFoundError:
        return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
    if not batch_id:
        return {"success": False, "message": "没有找到待提交的草稿批次"}
    
    url = f"{KEYTAO_API_BASE}/api/bot/batches/{batch_id}/submit"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers={
                    "X-Bot-Token": BOT_API_TOKEN,
                    "Content-Type": "application/json"
                },
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
        return {"success": False, "message": "Bot配置错误：缺少API token"}

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
        return {"success": False, "message": "Bot配置错误：缺少API token"}

    url = f"{KEYTAO_API_BASE}/api/bot/batches/recall"
    logger.info(f"[keytao_recall_batch] platform={platform} platformId={platform_id}")

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                url,
                headers={"X-Bot-Token": BOT_API_TOKEN, "Content-Type": "application/json"},
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
        return {"success": False, "message": "Bot配置错误：缺少API token"}

    if platform == "web-anon":
        return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}

    url = f"{KEYTAO_API_BASE}/api/bot/batches/latest-draft/items"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                url,
                headers={"X-Bot-Token": BOT_API_TOKEN},
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
        return {"success": False, "message": "Bot配置错误：缺少API token"}

    url = f"{KEYTAO_API_BASE}/api/bot/pull-requests/{pr_id}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.request(
                "DELETE",
                url,
                headers={"X-Bot-Token": BOT_API_TOKEN, "Content-Type": "application/json"},
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
                        "description": "词条类型，默认为 Phrase（词组）"
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
) -> Dict:
    """
    Batch add word entries to draft (tolerant mode).
    Hard conflicts are skipped and reported; duplicate-code warnings are auto-confirmed.

    Args:
        platform: Platform type ('qq' or 'telegram')
        platform_id: User's platform ID
        items: List of dicts with keys: word, code, action (optional), type (optional), remark (optional)
        batch_id: Optional existing draft batch ID

    Returns:
        dict with successCount, failedCount, skippedCount, failed[], skipped[], draftItems[], draftTotal
    """
    KEYTAO_API_BASE = get_keytao_url()
    BOT_API_TOKEN = get_bot_token()

    if not BOT_API_TOKEN:
        return {"success": False, "message": "Bot配置错误：缺少API token"}

    if not batch_id:
        try:
            batch_id = await get_latest_draft_batch(platform, platform_id)
        except UserNotFoundError:
            return {"success": False, "not_bound": True, "message": _not_bound_message(platform)}
        if not batch_id:
            return {"success": False, "message": "无法获取草稿批次，请稍后重试"}

    url = f"{KEYTAO_API_BASE}/api/bot/pull-requests/batch-draft"
    request_data = {
        "platform": platform,
        "platformId": platform_id,
        "batchId": batch_id,
        "items": [
            {**{k: v for k, v in item.items() if k != "old_word"},
             **({"oldWord": item["old_word"]} if "old_word" in item else {})}
            for item in items
        ],
    }

    logger.info(f"[keytao_batch_add_to_draft] Sending {len(items)} items to batch-draft")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                headers={"X-Bot-Token": BOT_API_TOKEN, "Content-Type": "application/json"},
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
        return {"success": False, "message": "Bot配置错误：缺少API token"}

    url = f"{KEYTAO_API_BASE}/api/bot/pull-requests/batch-draft"
    payload = {"platform": platform, "platformId": platform_id, "ids": ids}
    logger.info(f"[keytao_batch_remove_draft_items] Deleting ids={ids}")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                "DELETE", url,
                json=payload,
                headers={"X-Bot-Token": BOT_API_TOKEN, "Content-Type": "application/json"},
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

    code_phrase_map: Dict[str, List[Dict]] = {}
    word_candidate_code_map: Dict[str, List[str]] = {word: target_candidate_codes}
    probe_code = target_code
    vacated_code = current_phrase.get("code") if current_phrase else None
    visited_codes = set()

    while probe_code != vacated_code:
        if probe_code in visited_codes:
            return {"success": False, "message": f"顺延计算出现循环：{probe_code}"}
        visited_codes.add(probe_code)

        if probe_code not in code_phrase_map:
            code_lookup = await _lookup_codes_raw([probe_code])
            if not code_lookup.get("success"):
                return code_lookup
            for item in code_lookup.get("results", []):
                code_phrase_map[item.get("code", "")] = item.get("phrases", [])

        occupant = _select_code_occupant(code_phrase_map.get(probe_code, []), word)
        if not occupant:
            break

        occupant_word = occupant.get("word", "")
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
        if code_index + 1 >= len(occupant_codes):
            return {
                "success": False,
                "message": f"无法顺延「{occupant_word}」：{probe_code} 之后没有可用候选编码",
                "word": occupant_word,
                "candidateCodes": occupant_codes,
            }
        probe_code = occupant_codes[code_index + 1]

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
                "遇到冲突的条目会跳过并在 failed 列表中说明原因，重码（同编码不同词）会自动确认写入。"
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
                                    "description": "词条类型，不传则自动推断",
                                },
                                "remark": {"type": "string", "description": "备注（可选）"},
                            },
                            "required": ["word", "code"],
                        },
                    }
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
