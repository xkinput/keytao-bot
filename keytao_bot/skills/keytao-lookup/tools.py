"""
Keytao Lookup Skill Tools
键道查词工具实现
"""
import re

import httpx
from typing import Dict, List, Optional
from nonebot.log import logger

from keytao_bot.utils.keytao_encoding import (
    build_alternate_pronunciation_codes as _build_alternate_pronunciation_codes,
    build_phrase_pronunciation_codes as _build_phrase_pronunciation_codes,
    normalize_contextual_phrase_encoding,
)


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


def _clean_code_list(codes: object) -> List[str]:
    if not isinstance(codes, list):
        return []

    result: List[str] = []
    seen = set()
    for code in codes:
        if not isinstance(code, str):
            continue
        normalized = code.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _has_valid_codes(codes: List[str]) -> bool:
    return bool(codes) and all("?" not in code for code in codes)


def _requested_code_from_analysis(*sources: Dict) -> Optional[str]:
    for source in sources:
        analysis = source.get("requestedCodeAnalysis")
        if isinstance(analysis, dict):
            code = analysis.get("code")
            if isinstance(code, str) and code.strip():
                return code.strip().lower()
    return None


def _clean_encode_chars(chars: object) -> List[Dict]:
    if not isinstance(chars, list):
        return []

    cleaned = []
    for item in chars:
        if not isinstance(item, dict):
            continue
        cleaned.append({
            "char": item.get("char", ""),
            "pinyin": item.get("pinyin", ""),
            "pinyins": item.get("pinyins", []),
            "phoneticCode": item.get("phoneticCode", ""),
            "c1": item.get("c1"),
            "c2": item.get("c2"),
            "shapeCode": item.get("shapeCode"),
        })
    return cleaned


def _normalize_encode_response(word: str, encode_data: Dict, infer_data: Optional[Dict] = None) -> Dict:
    """Return phrase candidate codes before display-only char split data."""
    encode_data = normalize_contextual_phrase_encoding(word, encode_data)
    infer_data = normalize_contextual_phrase_encoding(word, infer_data or {})
    encode_codes = _clean_code_list(encode_data.get("codes"))
    encode_alt_codes = _clean_code_list(encode_data.get("altCodes"))
    infer_codes = _clean_code_list(infer_data.get("codes"))
    infer_alt_codes = _clean_code_list(infer_data.get("altCodes"))

    if _has_valid_codes(encode_codes):
        codes = encode_codes
        alt_codes = encode_alt_codes
        code_source = "encode"
    elif _has_valid_codes(infer_codes):
        codes = infer_codes
        alt_codes = infer_alt_codes
        code_source = "infer-fallback"
    else:
        codes = encode_codes or infer_codes
        alt_codes = encode_alt_codes or infer_alt_codes
        code_source = "invalid"

    chars = _clean_encode_chars(encode_data.get("chars"))
    alternate_pronunciation_codes = _build_alternate_pronunciation_codes(chars)
    phrase_pronunciation_codes = _build_phrase_pronunciation_codes(chars)
    pronunciation_variants = [*alternate_pronunciation_codes, *phrase_pronunciation_codes]
    alternate_codes = _clean_code_list(
        [
            code
            for variant in pronunciation_variants
            for code in variant.get("codes", [])
            if isinstance(variant, dict)
        ]
    )
    requested_code = _requested_code_from_analysis(encode_data, infer_data)
    requested_variants = [
        variant
        for variant in pronunciation_variants
        if requested_code
        and isinstance(variant, dict)
        and variant.get("phoneticCode") == requested_code
    ]
    requested_variant_codes = _clean_code_list([
        code
        for variant in requested_variants
        for code in variant.get("codes", [])
        if isinstance(code, str)
    ])
    requested_candidate_codes = _clean_code_list(
        [
            code for code in [
                *requested_variant_codes,
                *alternate_codes,
            ]
            if (requested_code and code.startswith(requested_code)) or code in requested_variant_codes
        ]
    )

    candidate_codes = _clean_code_list([
        *requested_candidate_codes,
        *codes,
        *alt_codes,
        *alternate_codes,
    ])
    base_code = codes[0] if codes else None
    suggestion = infer_data.get("suggestion")
    recommended_code = suggestion if isinstance(suggestion, str) and "?" not in suggestion else None
    if not recommended_code and codes and "?" not in codes[0]:
        recommended_code = codes[0]
    if not recommended_code and candidate_codes:
        recommended_code = candidate_codes[0]

    result = {
        "success": _has_valid_codes(codes),
        "input": encode_data.get("input") or infer_data.get("word") or word,
        "word": word,
        "type": encode_data.get("type") or infer_data.get("type", ""),
        "baseCode": base_code,
        "recommendedCode": recommended_code,
        "candidateCodes": candidate_codes,
        "codes": codes,
        "altCodes": alt_codes,
        "alternatePronunciationCodes": alternate_pronunciation_codes,
        "alternatePhrasePronunciationCodes": phrase_pronunciation_codes,
        "requestedCandidateCodes": requested_candidate_codes,
        "flyKeyVariants": encode_data.get("flyKeyVariants") or infer_data.get("flyKeyVariants") or [],
        "codeSource": code_source,
        "chars": chars,
    }

    for key in ("suggestion", "suggestionIndex", "isBaseConflict", "wordExists", "requestedCodeAnalysis"):
        if key in infer_data:
            result[key] = infer_data[key]
        elif key in encode_data:
            result[key] = encode_data[key]

    if not result["success"]:
        result["message"] = "编码服务未能返回有效候选编码，请让用户手动指定编码"
    return result


def _format_candidate_status(code: str, phrases: List[Dict]) -> Dict:
    words = [phrase.get("word", "") for phrase in phrases if phrase.get("word")]
    occupied = bool(phrases)
    if occupied:
        label = "已有「" + "、".join(words[:3]) + "」"
        if len(words) > 3:
            label += f"等 {len(words)} 个词"
    else:
        label = "空位"

    return {
        "code": code,
        "occupied": occupied,
        "label": label,
        "phrases": phrases,
    }


def _format_candidate_display_label(target_word: str, status: Dict, recommended: bool) -> Dict:
    phrases = status.get("phrases", [])
    words = [
        phrase.get("word", "")
        for phrase in phrases
        if isinstance(phrase, dict) and phrase.get("word")
    ]
    own_words = [word for word in words if word == target_word]
    other_words = [word for word in words if word != target_word]

    if not status.get("occupied"):
        display_label = "✅ （推荐）" if recommended else "✅"
        state = "available"
    elif own_words:
        display_parts = [f"已有 {target_word} ✔️"]
        display_parts.extend(other_words[:3])
        display_label = "、".join(display_parts)
        state = "occupied_self"
    else:
        display_label = "、".join(other_words[:3]) if other_words else status.get("label", "")
        state = "occupied_other"

    return {
        "code": status.get("code", ""),
        "displayLabel": display_label,
        "state": state,
        "recommended": recommended,
        "occupied": bool(status.get("occupied")),
        "words": words,
    }


def _build_candidate_display_groups(encoding: Dict, statuses: List[Dict]) -> List[Dict]:
    variants = [
        variant for variant in encoding.get("alternatePronunciationCodes", [])
        if isinstance(variant, dict)
    ]
    if len(variants) <= 1:
        return []

    status_map = {
        status.get("code", ""): status
        for status in statuses
        if isinstance(status, dict) and status.get("code")
    }
    target_word = encoding.get("word", "")
    groups: List[Dict] = []

    for variant in variants:
        codes = [
            code for code in variant.get("codes", [])
            if isinstance(code, str) and code in status_map
        ]
        if not codes:
            continue

        recommended_code = next(
            (
                code for code in codes
                if not status_map.get(code, {}).get("occupied")
            ),
            None,
        )
        pinyin = variant.get("pinyin", "")
        phonetic_code = variant.get("phoneticCode", "")
        is_default = bool(variant.get("isDefault"))
        pinyin_label = f"{pinyin}（默认音）" if is_default else pinyin

        groups.append({
            "pinyin": pinyin,
            "pinyinLabel": pinyin_label,
            "phoneticCode": phonetic_code,
            "isDefault": is_default,
            "recommendedCode": recommended_code,
            "items": [
                _format_candidate_display_label(
                    target_word,
                    status_map[code],
                    recommended=code == recommended_code,
                )
                for code in codes
            ],
        })

    return groups


def _apply_candidate_occupancy(encoding: Dict, lookup_result: Dict) -> Dict:
    candidate_codes = encoding.get("candidateCodes", [])
    if not isinstance(candidate_codes, list) or not candidate_codes:
        encoding["occupancyChecked"] = False
        encoding["candidateStatuses"] = []
        return encoding

    if not lookup_result.get("success"):
        encoding["occupancyChecked"] = False
        encoding["candidateStatuses"] = []
        encoding["occupancyError"] = lookup_result.get("message", "候选编码占用查询失败")
        return encoding

    result_map = {
        item.get("code", ""): item.get("phrases", [])
        for item in lookup_result.get("results", [])
        if isinstance(item, dict)
    }
    statuses = [
        _format_candidate_status(code, result_map.get(code, []))
        for code in candidate_codes
    ]
    requested_candidate_codes = encoding.get("requestedCandidateCodes", [])
    requested_code_set = {
        code for code in requested_candidate_codes
        if isinstance(code, str)
    }
    first_requested_available = next(
        (
            item["code"] for item in statuses
            if item["code"] in requested_code_set and not item["occupied"]
        ),
        None,
    )
    first_available = next((item["code"] for item in statuses if not item["occupied"]), None)

    encoding["occupancyChecked"] = True
    encoding["candidateStatuses"] = statuses
    encoding["firstAvailableCode"] = first_available
    if first_requested_available:
        encoding["firstRequestedAvailableCode"] = first_requested_available
        encoding["recommendedCode"] = first_requested_available
    elif first_available:
        encoding["recommendedCode"] = first_available
    display_groups = _build_candidate_display_groups(encoding, statuses)
    if display_groups:
        encoding["candidateDisplayGroups"] = display_groups
    return encoding


async def _call_bot_lookup_api(path: str, payload: Dict) -> Dict:
    keytao_api_base = get_keytao_url()
    bot_api_token = get_bot_token()

    if not bot_api_token:
        return {
            "success": False,
            "message": "喵喵配置错误：缺少API token"
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

    logger.info(f"[keytao_lookup_by_code] code={code} found={len(result_phrases)} {[(p.get('word'), p.get('weight')) for p in result_phrases]}")

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


async def keytao_encode(word: str, requested_code: Optional[str] = None) -> Dict:
    """
    Calculate keytao encoding and char split for a word (rule-based, not DB query)
    计算词条的键道编码及字根拆分（按规则计算，非数据库查询）

    Args:
        word: Chinese word or character to encode
        requested_code: Optional user-specified code to analyze against fixed fly-key rules

    Returns:
        dict: Encoding result with codes, altCodes, and per-char split data
    """
    keytao_api_base = get_keytao_url()
    encode_url = f"{keytao_api_base}/api/phrases/encode"
    infer_url = f"{keytao_api_base}/api/phrases/infer"

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            params = {"word": word}
            if requested_code:
                params["code"] = requested_code
            response = await client.get(encode_url, params=params)
            if response.is_success:
                encode_data = response.json()
                codes = _clean_code_list(encode_data.get("codes"))
                if _has_valid_codes(codes):
                    encoding = _normalize_encode_response(word, encode_data)
                    lookup_result = await keytao_lookup_by_codes_batch(encoding.get("candidateCodes", []))
                    return _apply_candidate_occupancy(encoding, lookup_result)

                infer_response = await client.get(infer_url, params=params)
                infer_data = infer_response.json() if infer_response.is_success else {}
                encoding = _normalize_encode_response(word, encode_data, infer_data)
                lookup_result = await keytao_lookup_by_codes_batch(encoding.get("candidateCodes", []))
                return _apply_candidate_occupancy(encoding, lookup_result)
            return {
                "success": False,
                "message": f"编码服务返回错误: {response.status_code}"
            }
    except Exception as error:
        logger.error(f"[keytao_encode] error: {error}")
        return {
            "success": False,
            "message": "编码服务暂时不可用，请稍后再试"
        }


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
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_encode",
            "description": "按键道规则计算词条的编码、候选码占用情况和字根拆分。返回的 candidateStatuses 是已经查过占用的候选列表，禁止回复'待查占用'；candidateCodes/codes 是唯一可用的词条候选编码，禁止根据 chars/fullCode/phoneticCode 自己拼编码。单字多音字会额外返回 alternatePronunciationCodes；词组中多音字会额外返回 alternatePhrasePronunciationCodes；这些合法候选会并入 candidateCodes。用户纠正读音或给出音码前缀时，应传 requested_code 并优先使用 requestedCandidateCodes/firstRequestedAvailableCode。与 keytao_lookup_by_word 不同，此工具是按规则实时计算（非数据库查询），会返回推荐编码、进阶选重码、飞键备用码，以及每个字的音码、字根拆分、形码。适用场景：①用户问某词的拆分是什么；②加词前自动生成编码（必须先调用此工具）；③用户问的词可能不在词库中但仍需要编码",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "要编码的中文词条或单字，如 '你好', '若'"
                    },
                    "requested_code": {
                        "type": "string",
                        "description": "可选。用户强制指定或询问的编码，如 'ffb'。提供后会返回 requestedCodeAnalysis，说明该编码是否属于标准候选或固定飞键候选，并列出支持的同系列编码"
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
    "keytao_lookup_by_word": keytao_lookup_by_word,
    "keytao_encode": keytao_encode
}
