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
    Query phrase by code with duplicate analysis
    按编码查询词条，并分析重码情况
    
    Args:
        code: Keytao input method code (pure letters)
        
    Returns:
        dict: Query result with phrases list and duplicate labels
    """
    KEYTAO_NEXT_URL = get_keytao_url()
    url = f"{KEYTAO_NEXT_URL}/api/phrases/by-code"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params={"code": code, "page": "1"})
            response.raise_for_status()
            data = response.json()
            phrases = data.get("phrases", [])
            
            if not phrases:
                return {
                    "success": True,
                    "code": code,
                    "phrases": []
                }
            
            # Filter to ensure exact code match (API might return similar codes)
            exact_matches = [p for p in phrases if p.get("code", "") == code]
            
            if not exact_matches:
                return {
                    "success": True,
                    "code": code,
                    "phrases": []
                }
            
            # Sort by weight and limit to 10 results
            sorted_phrases = sorted(exact_matches, key=lambda x: x.get("weight", 0))
            
            # Type mapping to Chinese labels
            type_labels = {
                "Single": "单字",
                "Phrase": "词组",
                "Supplement": "补充词条",
                "Symbol": "符号",
                "Link": "链接",
                "CSS": "声笔笔",
                "CSSSingle": "声笔笔单字",
                "English": "英文"
            }
            
            # Analyze duplicate positions
            weights = [p.get("weight", 0) for p in sorted_phrases]
            min_weight = min(weights)
            base_weight = (min_weight // 10) * 10
            
            # Infer base weight
            if min_weight % 10 == 1:
                base_weight = min_weight
            elif min_weight % 10 == 0:
                base_weight = min_weight
            
            # Position labels (default word has no label)
            position_labels = {
                0: "",
                1: "二重",
                2: "三重",
                3: "四重",
                4: "五重",
                5: "六重"
            }
            
            # Build result with labels
            result_phrases = []
            for p in sorted_phrases:
                p_weight = p.get("weight", 0)
                p_pos = p_weight - base_weight
                phrase_type = p.get("type", "")
                
                # Only show label if there are multiple phrases AND position > 0
                if len(sorted_phrases) > 1 and p_pos > 0:
                    label = position_labels.get(p_pos, f"{p_pos + 1}重")
                else:
                    label = ""
                
                result_phrases.append({
                    "word": p.get("word", ""),
                    "code": p.get("code", ""),
                    "weight": p_weight,
                    "type": phrase_type,
                    "type_label": type_labels.get(phrase_type, phrase_type),
                    "position": p_pos,
                    "position_label": label
                })
            
            return {
                "success": True,
                "code": code,
                "phrases": result_phrases
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
    Query code by word with duplicate analysis
    按词条查询编码，并分析重码情况
    
    Args:
        word: Chinese word/phrase to query
        
    Returns:
        dict: Query result with phrases list and duplicate analysis
    """
    KEYTAO_NEXT_URL = get_keytao_url()
    url = f"{KEYTAO_NEXT_URL}/api/phrases/by-word"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # First request to get total pages
            response = await client.get(url, params={"word": word, "page": "1"})
            response.raise_for_status()
            data = response.json()
            
            all_phrases = data.get("phrases", [])
            pagination = data.get("pagination", {})
            total_pages = pagination.get("totalPages", 1)
            
            # If there are more pages, fetch them
            if total_pages > 1:
                for page in range(2, total_pages + 1):
                    page_response = await client.get(url, params={"word": word, "page": str(page)})
                    page_response.raise_for_status()
                    page_data = page_response.json()
                    all_phrases.extend(page_data.get("phrases", []))
            
            # Type mapping to Chinese labels
            type_labels = {
                "Single": "单字",
                "Phrase": "词组",
                "Supplement": "补充词条",
                "Symbol": "符号",
                "Link": "链接",
                "CSS": "声笔笔",
                "CSSSingle": "声笔笔单字",
                "English": "英文"
            }
            
            # Enhance each phrase with duplicate analysis
            enhanced_phrases = []
            for p in all_phrases:
                phrase_code = p.get("code", "")
                phrase_weight = p.get("weight", 0)
                phrase_type = p.get("type", "")
                
                phrase_info = {
                    "word": p.get("word", ""),
                    "code": phrase_code,
                    "weight": phrase_weight,
                    "type": phrase_type,
                    "type_label": type_labels.get(phrase_type, phrase_type)  # Chinese label
                }
                
                # Check if weight suggests duplicates (last digit not 0)
                if phrase_weight % 10 != 0:
                    # Query all phrases with this code to analyze duplicates
                    dup_analysis = await analyze_duplicates(phrase_code, word, phrase_weight)
                    if dup_analysis:
                        phrase_info["duplicate_info"] = dup_analysis
                
                enhanced_phrases.append(phrase_info)
            
            # Sort by code length (shorter codes first)
            enhanced_phrases.sort(key=lambda x: len(x.get("code", "")))
            
            return {
                "success": True,
                "word": word,
                "phrases": enhanced_phrases
            }
    except Exception as e:
        return {
            "success": False,
            "word": word,
            "error": str(e),
            "phrases": []
        }


async def analyze_duplicates(code: str, target_word: str, target_weight: int) -> Optional[Dict]:
    """
    Analyze duplicate information for a code
    分析编码的重码情况
    
    Args:
        code: The code to analyze
        target_word: The word we're querying for
        target_weight: The weight of the target word
        
    Returns:
        dict: Duplicate analysis with position and all words
    """
    try:
        # Query all phrases with this code
        result = await keytao_lookup_by_code(code)
        if not result.get("success") or not result.get("phrases"):
            return None
        
        all_phrases = result["phrases"]
        if len(all_phrases) <= 1:
            return None  # No duplicates
        
        # Sort by weight
        sorted_phrases = sorted(all_phrases, key=lambda x: x.get("weight", 0))
        
        # Infer base weight (might start from 100 or 101)
        weights = [p.get("weight", 0) for p in sorted_phrases]
        min_weight = min(weights)
        base_weight = (min_weight // 10) * 10  # Get tens place
        
        # If min_weight ends with 1 (e.g., 101), base is likely min_weight
        # If min_weight ends with 0 (e.g., 100), base is min_weight
        if min_weight % 10 == 1:
            base_weight = min_weight
        elif min_weight % 10 == 0:
            base_weight = min_weight
        
        # Calculate position for target word
        position = target_weight - base_weight
        
        # Position labels
        position_labels = {
            0: "",  # Default, no label
            1: "二重",
            2: "三重",
            3: "四重",
            4: "五重",
            5: "六重"
        }
        
        # Build word list with positions
        word_list = []
        for p in sorted_phrases:
            p_weight = p.get("weight", 0)
            p_pos = p_weight - base_weight
            p_word = p.get("word", "")
            
            # Only show label for position > 0
            if p_pos > 0:
                label = position_labels.get(p_pos, f"{p_pos + 1}重")
            else:
                label = ""
            
            word_list.append({
                "word": p_word,
                "weight": p_weight,
                "position": p_pos,
                "label": label
            })
        
        return {
            "position": position,
            "position_label": position_labels.get(position, f"{position}重") if position > 0 else "",
            "base_weight": base_weight,
            "all_words": word_list
        }
        
    except Exception as e:
        return None



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
