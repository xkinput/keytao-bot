"""
Keytao Lookup Skill Tools
键道查词工具实现
"""
import httpx
from typing import Dict, List, Optional


def get_keytao_url() -> str:
    """Get Keytao Next URL from config"""
    try:
        from nonebot import get_driver
        driver = get_driver()
        config = driver.config
        return getattr(config, "keytao_next_url", "https://keytao.vercel.app")
    except:
        # Fallback to default if NoneBot not initialized
        return "https://keytao.vercel.app"


async def keytao_lookup_by_code(code: str) -> Dict:
    """
    Query phrase by code
    按编码查询词条
    
    Args:
        code: Keytao input method code (pure letters)
        
    Returns:
        dict: Query result with phrases list
    """
    KEYTAO_NEXT_URL = get_keytao_url()
    url = f"{KEYTAO_NEXT_URL}/api/phrases/by-code"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params={"code": code, "page": "1"})
            response.raise_for_status()
            data = response.json()
            phrases = data.get("phrases", [])[:5]  # Limit to 5 results
            
            return {
                "success": True,
                "code": code,
                "phrases": [
                    {
                        "word": p.get("word", ""),
                        "code": p.get("code", ""),
                        "weight": p.get("weight", 0)
                    }
                    for p in phrases
                ]
            }
    except Exception as e:
        return {
            "success": False,
            "code": code,
            "error": str(e),
            "phrases": []
        }


async def keytao_lookup_by_word(word: str) -> Dict:
    """
    Query code by word
    按词条查询编码
    
    Args:
        word: Chinese word/phrase to query
        
    Returns:
        dict: Query result with phrases list
    """
    KEYTAO_NEXT_URL = get_keytao_url()
    url = f"{KEYTAO_NEXT_URL}/api/phrases/by-word"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params={"word": word, "page": "1"})
            response.raise_for_status()
            data = response.json()
            phrases = data.get("phrases", [])[:5]  # Limit to 5 results
            
            return {
                "success": True,
                "word": word,
                "phrases": [
                    {
                        "word": p.get("word", ""),
                        "code": p.get("code", ""),
                        "weight": p.get("weight", 0)
                    }
                    for p in phrases
                ]
            }
    except Exception as e:
        return {
            "success": False,
            "word": word,
            "error": str(e),
            "phrases": []
        }


# Tool definitions for OpenAI Function Calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "keytao_lookup_by_code",
            "description": "查询键道输入法编码对应的词条。用于将字母编码（如 'abc', 'nau'）转换为中文词组",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "键道输入法编码，纯字母组合，如 'abc', 'nau'"
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_lookup_by_word",
            "description": "查询中文词条对应的键道输入法编码。用于查找如何用键道输入法打出某个词",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "要查询的中文词条，如 '你好', '世界'"
                    }
                },
                "required": ["word"]
            }
        }
    }
]


# Tool registry for dynamic calling
TOOL_FUNCTIONS = {
    "keytao_lookup_by_code": keytao_lookup_by_code,
    "keytao_lookup_by_word": keytao_lookup_by_word
}
