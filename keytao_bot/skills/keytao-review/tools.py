"""KeyTao review skill tools."""
from __future__ import annotations

from typing import Dict, List

from keytao_bot.utils.keytao_review import (
    ReviewHttpConfig,
    audit_draft_items,
    prepare_reviewed_word,
)


def get_keytao_url() -> str:
    try:
        from nonebot import get_driver
        driver = get_driver()
        config = driver.config
        return getattr(config, "keytao_api_base", "https://keytao.vercel.app")
    except Exception:
        return "https://keytao.vercel.app"


def get_bot_token() -> str:
    try:
        from nonebot import get_driver
        driver = get_driver()
        config = driver.config
        return getattr(config, "bot_api_token", "") or ""
    except Exception:
        return ""


def _review_config() -> ReviewHttpConfig:
    return ReviewHttpConfig(api_base=get_keytao_url(), bot_token=get_bot_token())


async def keytao_prepare_reviewed_add(word: str) -> Dict:
    return await prepare_reviewed_word(_review_config(), word)


async def keytao_audit_draft_items(items: List[Dict]) -> Dict:
    return await audit_draft_items(_review_config(), items)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "keytao_prepare_reviewed_add",
            "description": "加词前审词：从汉典、萌典、百度百科、维基百科、汉语辞典等来源核对真实读音，再按真实读音生成键道候选编码和当前占位。",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "要审查并准备加词的中文词语",
                    },
                },
                "required": ["word"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "keytao_audit_draft_items",
            "description": "提交前自动审核草稿条目，判断是否证据一致、编码正确、调码合理；纯删除和歧义项会返回 needs_admin。",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "description": "草稿条目列表",
                        "items": {"type": "object"},
                    },
                },
                "required": ["items"],
            },
        },
    },
]


TOOL_FUNCTIONS = {
    "keytao_prepare_reviewed_add": keytao_prepare_reviewed_add,
    "keytao_audit_draft_items": keytao_audit_draft_items,
}
