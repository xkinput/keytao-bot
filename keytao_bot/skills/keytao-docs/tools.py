"""
Keytao Docs Skill Tools
键道文档查询工具实现 - 通过 GitHub Code Search API 搜索文档内容
"""
import os
import httpx
from typing import Dict, List
from nonebot.log import logger


GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_RAW_PREFIX = "https://raw.githubusercontent.com/xkinput/keytao-docs/main/"
DOCS_SITE_URL = "https://keytao-docs.vercel.app"
DOCS_REPO = "xkinput/keytao-docs"


def _gh_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def clean_markdown(content: str) -> str:
    lines = content.split('\n')
    if lines and lines[0].strip() == '---':
        try:
            end_idx = lines[1:].index('---') + 1
            lines = lines[end_idx + 1:]
        except ValueError:
            pass
    cleaned = '\n'.join(lines).strip()
    if len(cleaned) > 2000:
        cleaned = cleaned[:2000] + "\n\n..."
    return cleaned


def _path_to_url(path: str) -> str:
    return f"{DOCS_SITE_URL}/{path.replace('.md', '')}"


async def _search_docs(query: str) -> List[str]:
    """Search keytao-docs repo via GitHub Code Search, return list of file paths."""
    url = "https://api.github.com/search/code"
    params = {"q": f"{query} repo:{DOCS_REPO} extension:md", "per_page": 5}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params, headers=_gh_headers())
            resp.raise_for_status()
            data = resp.json()
            return [item["path"] for item in data.get("items", [])]
    except Exception as e:
        logger.warning(f"GitHub search failed: {e}")
        return []


async def _fetch_raw(path: str) -> str:
    url = GITHUB_RAW_PREFIX + path
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        logger.warning(f"Failed to fetch {path}: {e}")
        return ""


async def keytao_fetch_docs(query: str) -> Dict:
    """
    通过 GitHub Code Search 在键道文档仓库中搜索相关内容

    Args:
        query: 要查询的问题或关键词

    Returns:
        dict: 查询结果
    """
    try:
        paths = await _search_docs(query)

        if not paths:
            paths = ["guide/index.md"]

        paths = paths[:3]

        content_parts: List[str] = []
        sources: List[str] = []

        for path in paths:
            raw = await _fetch_raw(path)
            if not raw:
                continue
            cleaned = clean_markdown(raw)
            if not cleaned:
                continue
            title = path.split('/')[-1].replace('.md', '')
            for line in cleaned.split('\n'):
                if line.startswith('# '):
                    title = line.lstrip('#').strip()
                    break
            content_parts.append(f"【{title}】\n\n{cleaned}")
            sources.append(_path_to_url(path))

        if content_parts:
            return {
                "success": True,
                "query": query,
                "content": "\n\n---\n\n".join(content_parts),
                "sources": sources,
                "hint": f"更多详细信息请访问: {DOCS_SITE_URL}",
            }
        return {
            "success": False,
            "query": query,
            "content": "",
            "error": "未能获取文档内容",
            "hint": f"建议访问官方文档了解更多：{DOCS_SITE_URL}",
        }

    except Exception as e:
        logger.error(f"Error fetching docs: {e}")
        return {
            "success": False,
            "query": query,
            "error": str(e),
            "content": "",
            "sources": [],
            "hint": f"访问文档: {DOCS_SITE_URL}",
        }


# Tool definitions for OpenAI Function Calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "keytao_fetch_docs",
            "description": "从键道输入法官方文档获取内容。用于回答：1) 如何使用键道（零声母、顶功、简码等输入规则）; 2) 学习方法和教程; 3) 安装配置指南; 4) 输入法概念解释。注意：查询具体词条编码请用lookup工具，不要用本工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要查询的问题或关键词，如 '零声母', '顶功', 'Lua功能', '时间日期', '简码提示', '字根规则', '安装教程'等"
                    }
                },
                "required": ["query"]
            }
        }
    }
]


# Tool registry for dynamic calling
TOOL_FUNCTIONS = {
    "keytao_fetch_docs": keytao_fetch_docs
}
