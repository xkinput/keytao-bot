"""
Keytao Docs Skill Tools
键道文档查询工具实现 - 从 GitHub 获取真实文档内容
"""
import httpx
from typing import Dict, List
from nonebot.log import logger


# 文档文件映射 (关键词 -> GitHub 文档路径)
DOCS_MAPPING = {
    "规则": [
        "guide/learn-xkjd/phonetics-rules.md",  # 音码规则
        "guide/learn-xkjd/stroke-rules.md",      # 形码规则
    ],
    "学习": [
        "guide/learn-xkjd/index.md",             # 研习键道概述
        "guide/start-xkjd/index.md",             # 入门键道
        "guide/learn-xkjd/layouts.md",           # 键道图谱
    ],
    "安装": [
        "guide/get-xkjd/download-and-install.md",
        "guide/get-xkjd/index.md",
    ],
    "字根": [
        "guide/learn-xkjd/stroke-rules.md",
        "guide/learn-xkjd/layouts.md",
    ],
    "单字": [
        "guide/start-xkjd/characters.md",
    ],
    "词组": [
        "guide/start-xkjd/phrases.md",
    ],
    "顶功": [
        "guide/advance-in-xkjd/top-up.md",
    ],
    "简码": [
        "guide/advance-in-xkjd/shorthand.md",
    ],
}

# GitHub raw 文件 URL 前缀
GITHUB_RAW_PREFIX = "https://raw.githubusercontent.com/xkinput/keytao-docs/main/"


def clean_markdown(content: str) -> str:
    """清理 markdown 内容，移除 frontmatter"""
    lines = content.split('\n')
    
    # 移除 frontmatter (--- 之间的内容)
    if lines and lines[0].strip() == '---':
        try:
            end_idx = lines[1:].index('---') + 1
            lines = lines[end_idx + 1:]
        except ValueError:
            pass
    
    # 重新组合，限制长度
    cleaned = '\n'.join(lines).strip()
    
    # 限制总长度
    if len(cleaned) > 2000:
        cleaned = cleaned[:2000] + "\n\n..."
    
    return cleaned


async def fetch_doc_from_github(doc_path: str) -> str:
    """从 GitHub 获取文档内容"""
    url = GITHUB_RAW_PREFIX + doc_path
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except Exception as e:
        logger.warning(f"Failed to fetch {doc_path}: {e}")
        return ""


async def keytao_fetch_docs(query: str) -> Dict:
    """
    从键道文档 GitHub 仓库获取相关内容
    
    Args:
        query: 要查询的问题或关键词
        
    Returns:
        dict: 查询结果
    """
    try:
        query_lower = query.lower()
        
        # 匹配关键词到文档文件
        docs_to_fetch: List[str] = []
        matched_keywords = []
        
        for keyword, doc_paths in DOCS_MAPPING.items():
            if keyword in query_lower or keyword in query:
                docs_to_fetch.extend(doc_paths)
                matched_keywords.append(keyword)
        
        # 如果没有匹配到，使用默认文档
        if not docs_to_fetch:
            docs_to_fetch = ["guide/index.md"]
            matched_keywords = ["概述"]
        
        # 去重
        docs_to_fetch = list(dict.fromkeys(docs_to_fetch))[:3]  # 最多3个文档
        
        # 获取文档内容
        content_parts = []
        sources = []
        
        for doc_path in docs_to_fetch:
            doc_content = await fetch_doc_from_github(doc_path)
            if doc_content:
                cleaned = clean_markdown(doc_content)
                if cleaned:
                    # 提取文档标题（第一个 # 标题）
                    title = doc_path.split('/')[-1].replace('.md', '')
                    for line in cleaned.split('\n'):
                        if line.startswith('# '):
                            title = line.replace('# ', '').strip()
                            break
                    
                    content_parts.append(f"【{title}】\n\n{cleaned}")
                    sources.append(f"https://keytao-docs.vercel.app/{doc_path.replace('.md', '.html')}")
        
        if content_parts:
            combined = "\n\n---\n\n".join(content_parts)
            return {
                "success": True,
                "query": query,
                "content": combined,
                "sources": sources,
                "matched_keywords": matched_keywords,
                "hint": "更多详细信息请访问: https://keytao-docs.vercel.app"
            }
        else:
            return {
                "success": False,
                "query": query,
                "content": "",
                "error": "未能获取文档内容",
                "hint": f"建议访问官方文档了解更多：https://keytao-docs.vercel.app"
            }
            
    except Exception as e:
        logger.error(f"Error fetching docs: {e}")
        return {
            "success": False,
            "query": query,
            "error": str(e),
            "content": "",
            "sources": [],
            "hint": "访问文档: https://keytao-docs.vercel.app"
        }


# Tool definitions for OpenAI Function Calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "keytao_fetch_docs",
            "description": "从键道输入法官方文档网站获取相关内容。用于回答关于键道的学习方法、编码规则、教程、概念解释等问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要查询的问题或关键词，如 '学习方法', '编码规则', '字根', '安装教程'等"
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
