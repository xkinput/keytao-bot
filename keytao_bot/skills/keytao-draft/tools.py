"""
Keytao Create Skill Tools
键道创建词条工具实现
"""
import json
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


def enrich_pr_item_labels(item: Dict) -> Dict:
    """Add Chinese labels for action/type fields."""
    enriched_item = dict(item)
    action = enriched_item.get("action")
    phrase_type = enriched_item.get("type")
    enriched_item["action_label"] = ACTION_LABELS.get(action, action or "未知")
    enriched_item["type_label"] = TYPE_LABELS.get(phrase_type, phrase_type or "未知")
    return enriched_item


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
            return {
                "count": result.get("count", 0),
                "items": result.get("items", [])
            }
    except Exception as e:
        logger.warning(f"[draft_snapshot] Failed to fetch: {e}")
    return None


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
    batch_id = await get_latest_draft_batch(platform, platform_id)
    if not batch_id:
        return {
            "success": False,
            "message": "无法获取草稿批次，请稍后重试"
        }
    
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
                try:
                    data = response.json()
                    if data.get("message"):
                        return data
                except Exception:
                    pass
                return {
                    "success": False,
                    "message": "未找到绑定账号，请先使用 /bind 命令绑定你的键道平台账号到 https://keytao.vercel.app"
                }
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
    batch_id = await get_latest_draft_batch(platform, platform_id)
    if not batch_id:
        return {
            "success": False,
            "message": "没有找到待提交的草稿批次"
        }
    
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
                        "description": "旧词条内容（仅 Change 操作需要）"
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
            "description": "提交当前草稿批次进行审核。用于用户确认提交词条修改后。会自动查找并提交用户的草稿批次。",
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
        batch_id = await get_latest_draft_batch(platform, platform_id)
        if not batch_id:
            return {"success": False, "message": "无法获取草稿批次，请稍后重试"}

    url = f"{KEYTAO_API_BASE}/api/bot/pull-requests/batch-draft"
    request_data = {
        "platform": platform,
        "platformId": platform_id,
        "batchId": batch_id,
        "items": items,
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


TOOLS += [
    {
        "type": "function",
        "function": {
            "name": "keytao_batch_add_to_draft",
            "description": (
                "批量将词条加入草稿。适合用户一次提交大量词条时使用。"
                "遇到冲突的条目会跳过并在 failed 列表中说明原因，重码（同编码不同词）会自动确认写入。"
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
            "name": "keytao_batch_remove_draft_items",
            "description": (
                "批量从草稿中删除词条，通过条目 ID 列表指定要删除的内容。"
                "只能删除属于当前用户且处于草稿状态的条目。"
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
]


# Tool registry for dynamic calling
TOOL_FUNCTIONS = {
    "keytao_create_phrase": keytao_create_phrase,
    "keytao_submit_batch": keytao_submit_batch,
    "keytao_list_draft_items": keytao_list_draft_items,
    "keytao_remove_draft_item": keytao_remove_draft_item,
    "keytao_batch_add_to_draft": keytao_batch_add_to_draft,
    "keytao_batch_remove_draft_items": keytao_batch_remove_draft_items,
}
