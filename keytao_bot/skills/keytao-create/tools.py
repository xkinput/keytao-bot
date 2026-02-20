"""
Keytao Create Skill Tools
键道创建词条工具实现
"""
import json
import httpx
from typing import Dict, List, Optional
from nonebot.log import logger


def get_keytao_url() -> str:
    """Get Keytao API base URL from config"""
    try:
        from nonebot import get_driver
        driver = get_driver()
        config = driver.config
        return getattr(config, "keytao_api_base", "https://keytao.vercel.app")
    except:
        return "https://keytao.vercel.app"


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
                return data
            elif response.status_code == 404:
                logger.warning(f"[keytao_create_phrase] API response (404): {response.text}")
                return {
                    "success": False,
                    "message": "未找到绑定账号，请使用 /bind 命令绑定你的账号"
                }
            elif response.status_code == 400:
                # Conflict or warning
                data = response.json()
                logger.info(f"[keytao_create_phrase] API response (400): {json.dumps(data, ensure_ascii=False)}")
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
    platform_id: str
) -> Dict:
    """
    Submit current draft batch for review
    提交当前草稿批次进行审核
    
    Automatically finds and submits the user's latest draft batch.
    自动查找并提交用户的最新草稿批次。
    
    Args:
        platform: Platform type ('qq' or 'telegram')
        platform_id: User's platform ID
        
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
                    "platformId": platform_id
                }
            )
            
            if response.status_code == 200:
                return response.json()
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
                "properties": {},
                "required": []
            }
        }
    }
]


# Tool registry for dynamic calling
TOOL_FUNCTIONS = {
    "keytao_create_phrase": keytao_create_phrase,
    "keytao_submit_batch": keytao_submit_batch
}
