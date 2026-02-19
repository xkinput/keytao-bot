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
        "confirmed": confirmed
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


async def keytao_batch_create_phrases(
    platform: str,
    platform_id: str,
    items: List[Dict],
    confirmed: bool = False
) -> Dict:
    """
    Create multiple phrase entries in batch via bot API
    批量创建多个词条
    
    Args:
        platform: Platform type ('qq' or 'telegram')
        platform_id: User's platform ID
        items: List of items to create, each with {word, code, type?, remark?}
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
    
    url = f"{KEYTAO_API_BASE}/api/bot/pull-requests/batch"
    
    # Convert items to API format
    api_items = []
    for item in items:
        api_items.append({
            "action": "Create",
            "word": item.get("word"),
            "code": item.get("code"),
            "type": item.get("type", "Phrase"),
            "remark": item.get("remark")
        })
    
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
                    "items": api_items,
                    "confirmed": confirmed
                }
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {
                    "success": False,
                    "message": "未找到绑定账号，请使用 /bind 命令绑定你的账号"
                }
            elif response.status_code == 400:
                # Conflicts or warnings
                return response.json()
            else:
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
        logger.error(f"Batch create error: {e}")
        return {
            "success": False,
            "message": f"批量创建失败: {str(e)}"
        }


async def keytao_submit_batch(
    platform: str,
    platform_id: str,
    batch_id: str
) -> Dict:
    """
    Submit a batch for review
    提交批次进行审核
    
    Args:
        platform: Platform type ('qq' or 'telegram')
        platform_id: User's platform ID
        batch_id: Batch ID to submit
        
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
            "description": "创建、修改或删除键道词条。用于用户希望添加、修改或删除词条时。支持检测冲突和警告，如有重码警告可确认后创建。",
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
                        "description": "是否已确认重码警告（仅在首次返回警告后使用）"
                    }
                },
                "required": ["word", "code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_batch_create_phrases",
            "description": "批量创建多个键道词条。用于用户一次性添加多个词条时。",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "要创建的词条列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "word": {
                                    "type": "string",
                                    "description": "词条内容"
                                },
                                "code": {
                                    "type": "string",
                                    "description": "键道编码"
                                },
                                "type": {
                                    "type": "string",
                                    "description": "词条类型（可选）"
                                },
                                "remark": {
                                    "type": "string",
                                    "description": "备注（可选）"
                                }
                            },
                            "required": ["word", "code"]
                        }
                    },
                    "confirmed": {
                        "type": "boolean",
                        "description": "是否已确认重码警告"
                    }
                },
                "required": ["items"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_submit_batch",
            "description": "提交批次进行审核。用于用户确认提交词条修改后。",
            "parameters": {
                "type": "object",
                "properties": {
                    "batch_id": {
                        "type": "string",
                        "description": "要提交的批次ID"
                    }
                },
                "required": ["batch_id"]
            }
        }
    }
]


# Tool registry for dynamic calling
TOOL_FUNCTIONS = {
    "keytao_create_phrase": keytao_create_phrase,
    "keytao_batch_create_phrases": keytao_batch_create_phrases,
    "keytao_submit_batch": keytao_submit_batch
}
