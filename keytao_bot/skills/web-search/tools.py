"""
Web Search Skill Tools
通用网络搜索工具实现
"""
import html
import re
from typing import Dict, List
from urllib.parse import parse_qs, unquote, urlparse

import httpx
from nonebot.log import logger


SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)


def _strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_result_url(raw_url: str) -> str:
    if not raw_url:
        return ""

    decoded = html.unescape(raw_url)
    parsed = urlparse(decoded)

    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        query = parse_qs(parsed.query)
        uddg = query.get("uddg")
        if uddg:
            return unquote(uddg[0])

    return decoded


def _extract_results_from_html(content: str, max_results: int) -> List[Dict[str, str]]:
    anchors = list(
        re.finditer(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            content,
            re.IGNORECASE | re.DOTALL,
        )
    )
    if not anchors:
        return []

    results: List[Dict[str, str]] = []

    for index, match in enumerate(anchors[:max_results]):
        raw_url = match.group(1)
        title = _strip_tags(match.group(2))
        search_start = match.end()
        search_end = anchors[index + 1].start() if index + 1 < len(anchors) else min(len(content), search_start + 2000)
        nearby_html = content[search_start:search_end]

        snippet_match = re.search(
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(.*?)</div>',
            nearby_html,
            re.IGNORECASE | re.DOTALL,
        )
        snippet_raw = ""
        if snippet_match:
            snippet_raw = snippet_match.group(1) or snippet_match.group(2) or ""
        snippet = _strip_tags(snippet_raw)

        url = _normalize_result_url(raw_url)
        if not title or not url:
            continue

        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
        })

    return results


async def web_search(query: str, max_results: int = 5) -> Dict:
    """
    Search the web and return structured result snippets.

    Args:
        query: Search query or full question
        max_results: Number of results to return, clamped to 1-10

    Returns:
        dict: Search result payload
    """
    normalized_query = query.strip()
    if not normalized_query:
        return {
            "success": False,
            "query": query,
            "error": "搜索词不能为空",
            "results": [],
        }

    max_results = max(1, min(max_results, 10))

    try:
        async with httpx.AsyncClient(
            timeout=15.0,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            follow_redirects=True,
        ) as client:
            response = await client.get(
                SEARCH_ENDPOINT,
                params={"q": normalized_query, "kl": "cn-zh"},
            )
            response.raise_for_status()

        results = _extract_results_from_html(response.text, max_results)

        if not results:
            return {
                "success": False,
                "query": normalized_query,
                "provider": "duckduckgo",
                "error": "未解析到搜索结果，可能是搜索引擎页面结构变化或请求被限制",
                "results": [],
            }

        return {
            "success": True,
            "query": normalized_query,
            "provider": "duckduckgo",
            "results": results,
            "count": len(results),
        }
    except httpx.TimeoutException:
        return {
            "success": False,
            "query": normalized_query,
            "provider": "duckduckgo",
            "error": "搜索超时，请稍后重试",
            "results": [],
        }
    except httpx.HTTPError as exc:
        logger.warning(f"Web search HTTP error: {exc}")
        return {
            "success": False,
            "query": normalized_query,
            "provider": "duckduckgo",
            "error": f"搜索请求失败: {exc}",
            "results": [],
        }
    except Exception as exc:
        logger.exception(f"Web search failed: {exc}")
        return {
            "success": False,
            "query": normalized_query,
            "provider": "duckduckgo",
            "error": f"搜索失败: {exc}",
            "results": [],
        }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "通用网络搜索。用于查询实时信息、站外资料、新闻、公告、GitHub/官网页面、外部教程，或当用户明确要求‘帮我搜一下’时调用。不要用于键道站内词条查询。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索词或完整问题，例如 'nonebot2 function calling', '最近 Gemini 2.0 flash 更新', '键道 输入法 安装 教程'"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认 5，范围 1-10",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]


TOOL_FUNCTIONS = {
    "web_search": web_search,
}