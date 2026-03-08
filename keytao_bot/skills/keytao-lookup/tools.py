"""
Keytao Lookup Skill Tools
键道查词工具实现
"""
import httpx
from typing import Dict, List, Optional
from nonebot.log import logger


TYPE_LABELS = {
    "Single": "单字",
    "Phrase": "词组",
    "Supplement": "补充词条",
    "Symbol": "符号",
    "Link": "链接",
    "CSS": "声笔笔",
    "CSSSingle": "声笔笔单字",
    "English": "英文"
}

POSITION_LABELS = {
    0: "",
    1: "二重",
    2: "三重",
    3: "四重",
    4: "五重",
    5: "六重"
}


def get_keytao_url() -> str:
    """Get Keytao API base URL from config"""
    try:
        from nonebot import get_driver
        driver = get_driver()
        config = driver.config
        return getattr(config, "keytao_api_base", "https://keytao.vercel.app")
    except:
        # Fallback to default if NoneBot not initialized
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


def _position_label(index: int) -> str:
    return POSITION_LABELS.get(index, f"{index + 1}重") if index > 0 else ""


def _chunk_items(items: List[str], chunk_size: int) -> List[List[str]]:
    return [items[index:index + chunk_size] for index in range(0, len(items), chunk_size)]


async def _call_bot_lookup_api(path: str, payload: Dict) -> Dict:
    keytao_api_base = get_keytao_url()
    bot_api_token = get_bot_token()

    if not bot_api_token:
        return {
            "success": False,
            "message": "Bot配置错误：缺少API token"
        }

    url = f"{keytao_api_base}{path}"

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                url,
                headers={
                    "X-Bot-Token": bot_api_token,
                    "Content-Type": "application/json"
                },
                json=payload,
            )
            data = response.json()

            if response.is_success:
                return data

            return {
                "success": False,
                "message": data.get("message") or data.get("error") or f"API error: {response.status_code}"
            }
    except Exception as error:
        return {
            "success": False,
            "message": str(error)
        }


def _format_code_lookup_result(code: str, phrases: List[Dict]) -> Dict:
    if not phrases:
        return {
            "success": True,
            "code": code,
            "phrases": []
        }

    sorted_phrases = sorted(
        phrases,
        key=lambda item: (item.get("weight", 0), item.get("word", ""))
    )

    result_phrases = []
    for index, phrase in enumerate(sorted_phrases):
        phrase_type = phrase.get("type", "")
        result_phrases.append({
            "word": phrase.get("word", ""),
            "code": phrase.get("code", code),
            "weight": phrase.get("weight", 0),
            "type": phrase_type,
            "type_label": TYPE_LABELS.get(phrase_type, phrase_type),
            "position": index,
            "position_label": _position_label(index)
        })

    logger.info(f"[keytao_lookup_by_code] code={code}, found {len(result_phrases)} phrases")
    logger.info(f"[keytao_lookup_by_code] phrases: {[(p.get('word'), p.get('weight')) for p in result_phrases]}")

    return {
        "success": True,
        "code": code,
        "phrases": result_phrases
    }


def _build_duplicate_info(target_word: str, target_phrase: Dict, phrases_for_code: List[Dict]) -> Optional[Dict]:
    if len(phrases_for_code) <= 1:
        return None

    sorted_phrases = sorted(
        phrases_for_code,
        key=lambda item: (item.get("weight", 0), item.get("word", ""))
    )

    position = -1
    for index, phrase in enumerate(sorted_phrases):
        if (
            phrase.get("word") == target_word
            and phrase.get("weight") == target_phrase.get("weight", 0)
            and phrase.get("type") == target_phrase.get("type")
        ):
            position = index
            break

    if position == -1:
        return None

    return {
        "position": position,
        "position_label": _position_label(position),
        "all_words": [
            {
                "word": phrase.get("word", ""),
                "weight": phrase.get("weight", 0),
                "position": index,
                "label": _position_label(index)
            }
            for index, phrase in enumerate(sorted_phrases)
        ]
    }


def _format_word_lookup_result(word: str, phrases: List[Dict], code_phrase_map: Dict[str, List[Dict]]) -> Dict:
    enhanced_phrases = []

    for phrase in phrases:
        phrase_type = phrase.get("type", "")
        phrase_info = {
            "word": phrase.get("word", word),
            "code": phrase.get("code", ""),
            "weight": phrase.get("weight", 0),
            "type": phrase_type,
            "type_label": TYPE_LABELS.get(phrase_type, phrase_type)
        }

        duplicate_info = _build_duplicate_info(
            word,
            phrase,
            code_phrase_map.get(phrase.get("code", ""), [])
        )
        if duplicate_info:
            phrase_info["duplicate_info"] = duplicate_info

        enhanced_phrases.append(phrase_info)

    enhanced_phrases.sort(key=lambda item: (len(item.get("code", "")), item.get("code", ""), item.get("weight", 0)))

    return {
        "success": True,
        "word": word,
        "phrases": enhanced_phrases
    }


async def keytao_lookup_by_codes_batch(codes: List[str]) -> Dict:
    """
    Query phrases by codes in batch
    批量按编码查询词条

    Args:
        codes: List of Keytao codes, max 100 items

    Returns:
        dict: Batch query result
    """
    normalized_codes = [code.strip() for code in codes if isinstance(code, str) and code.strip()]

    if not normalized_codes:
        return {
            "success": False,
            "message": "至少要提供一个编码",
            "count": 0,
            "results": []
        }

    if len(normalized_codes) > 100:
        return {
            "success": False,
            "message": "一次最多查询100个编码",
            "count": 0,
            "results": []
        }

    data = await _call_bot_lookup_api(
        "/api/bot/phrases/by-code/batch",
        {"codes": normalized_codes}
    )

    if not data.get("success"):
        return {
            "success": False,
            "message": data.get("message", "查询失败"),
            "count": 0,
            "results": []
        }

    results = [
        _format_code_lookup_result(item.get("code", ""), item.get("phrases", []))
        for item in data.get("results", [])
    ]

    return {
        "success": True,
        "count": len(results),
        "results": results
    }


async def keytao_lookup_by_words_batch(words: List[str]) -> Dict:
    """
    Query codes by words in batch
    批量按词条查询编码

    Args:
        words: List of words, max 100 items

    Returns:
        dict: Batch query result
    """
    normalized_words = [word.strip() for word in words if isinstance(word, str) and word.strip()]

    if not normalized_words:
        return {
            "success": False,
            "message": "至少要提供一个词",
            "count": 0,
            "results": []
        }

    if len(normalized_words) > 100:
        return {
            "success": False,
            "message": "一次最多查询100个词",
            "count": 0,
            "results": []
        }

    word_data = await _call_bot_lookup_api(
        "/api/bot/phrases/by-word/batch",
        {"words": normalized_words}
    )

    if not word_data.get("success"):
        return {
            "success": False,
            "message": word_data.get("message", "查询失败"),
            "count": 0,
            "results": []
        }

    code_set = []
    seen_codes = set()
    for item in word_data.get("results", []):
        for phrase in item.get("phrases", []):
            code = phrase.get("code", "")
            if code and code not in seen_codes:
                seen_codes.add(code)
                code_set.append(code)

    code_phrase_map = {}
    if code_set:
        for code_chunk in _chunk_items(code_set, 100):
            code_data = await _call_bot_lookup_api(
                "/api/bot/phrases/by-code/batch",
                {"codes": code_chunk}
            )

            if not code_data.get("success"):
                return {
                    "success": False,
                    "message": code_data.get("message", "查询失败"),
                    "count": 0,
                    "results": []
                }

            for item in code_data.get("results", []):
                code_phrase_map[item.get("code", "")] = item.get("phrases", [])

    results = [
        _format_word_lookup_result(item.get("word", ""), item.get("phrases", []), code_phrase_map)
        for item in word_data.get("results", [])
    ]

    return {
        "success": True,
        "count": len(results),
        "results": results
    }


async def keytao_lookup_by_code(code: str) -> Dict:
    """
    Query phrase by code with duplicate analysis
    按编码查询词条，并分析重码情况
    
    Args:
        code: Keytao input method code (pure letters)
        
    Returns:
        dict: Query result with phrases list and duplicate labels
    """
    result = await keytao_lookup_by_codes_batch([code])
    if not result.get("success"):
        return {
            "success": False,
            "code": code,
            "error": result.get("message", "查询失败"),
            "phrases": []
        }

    return result.get("results", [{
        "success": True,
        "code": code,
        "phrases": []
    }])[0]



async def keytao_lookup_by_word(word: str) -> Dict:
    """
    Query code by word with duplicate analysis
    按词条查询编码，并分析重码情况
    
    Args:
        word: Chinese word/phrase to query
        
    Returns:
        dict: Query result with phrases list and duplicate analysis
    """
    result = await keytao_lookup_by_words_batch([word])
    if not result.get("success"):
        return {
            "success": False,
            "word": word,
            "error": result.get("message", "查询失败"),
            "phrases": []
        }

    return result.get("results", [{
        "success": True,
        "word": word,
        "phrases": []
    }])[0]



# Tool definitions for OpenAI Function Calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "keytao_lookup_by_codes_batch",
            "description": "批量查询多个键道输入法编码对应的词条。适合用户一次给出多个编码时使用，一次最多100个编码",
            "parameters": {
                "type": "object",
                "properties": {
                    "codes": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "要查询的编码列表，一次最多100个，如 ['abc', 'nau']"
                    }
                },
                "required": ["codes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_lookup_by_words_batch",
            "description": "批量查询多个中文词条对应的键道输入法编码。适合用户一次给出多个词时使用，一次最多100个词",
            "parameters": {
                "type": "object",
                "properties": {
                    "words": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "要查询的中文词列表，一次最多100个，如 ['你好', '世界']"
                    }
                },
                "required": ["words"]
            }
        }
    },
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
    "keytao_lookup_by_codes_batch": keytao_lookup_by_codes_batch,
    "keytao_lookup_by_words_batch": keytao_lookup_by_words_batch,
    "keytao_lookup_by_code": keytao_lookup_by_code,
    "keytao_lookup_by_word": keytao_lookup_by_word
}
