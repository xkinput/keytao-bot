"""KeyTao review skill tools."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from nonebot.log import logger

from keytao_bot.utils.keytao_review import (
    ReviewHttpConfig,
    audit_draft_items,
    can_llm_override_audit_issues,
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


def _preview_create_item(word: str, code: str) -> Dict[str, Any]:
    return {
        "action": "Create",
        "word": word,
        "code": code,
        "type": "Phrase",
    }


async def _try_llm_auto_review_for_preview(
    item: Dict[str, Any],
    deterministic_audit: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        from keytao_bot.utils.keytao_batch_review import review_keytao_batch_with_llm

        batch = {
            "id": "prepare-add-preview",
            "status": "Draft",
            "description": "键道助手加词前预审",
            "pullRequests": [item],
        }
        review_result = await review_keytao_batch_with_llm(batch)
        if not review_result.get("success"):
            return None

        ai_review = review_result.get("aiReview") or {}
        review_items = ai_review.get("items") if isinstance(ai_review.get("items"), list) else []
        non_pass_items = [
            review_item for review_item in review_items
            if isinstance(review_item, dict) and review_item.get("status") != "pass"
        ]
        if ai_review.get("verdict") == "pass" and not non_pass_items and review_items:
            word = str(item.get("word") or "")
            code = str(item.get("code") or "")
            return {
                **deterministic_audit,
                "success": True,
                "verdict": "pass",
                "autoApprove": True,
                "summary": ai_review.get("headline") or "本喵已结合语言常识完成复审，预计可自动通过",
                "issues": [],
                "approvedItems": [f"Create：{word}@{code}，本喵 LLM 复审通过"],
                "llmReview": ai_review,
                "llmFallback": True,
                "previewOnly": True,
            }

        issues: List[str] = []
        for review_item in non_pass_items[:5]:
            reasons = review_item.get("reasons") if isinstance(review_item.get("reasons"), list) else []
            title = review_item.get("title") or f"PR#{review_item.get('prId')} 需要复核"
            issues.append(str(reasons[0] if reasons else title))
        return {
            **deterministic_audit,
            "summary": ai_review.get("headline") or deterministic_audit.get("summary", "存在不确定项，提交后等待管理员审核"),
            "issues": issues or deterministic_audit.get("issues", []),
            "llmReview": ai_review,
            "llmFallback": True,
            "previewOnly": True,
        }
    except Exception as error:
        logger.warning(f"[prepare_reviewed_add] LLM preview review failed: {error}")
        return None


async def _build_pre_submit_audit(config: ReviewHttpConfig, word: str, code: str) -> Dict[str, Any]:
    item = _preview_create_item(word, code)
    deterministic_audit = await audit_draft_items(config, [item])
    deterministic_audit["previewOnly"] = True
    if deterministic_audit.get("autoApprove") or not can_llm_override_audit_issues(deterministic_audit):
        return deterministic_audit
    llm_audit = await _try_llm_auto_review_for_preview(item, deterministic_audit)
    return llm_audit or deterministic_audit


async def keytao_prepare_reviewed_add(word: str) -> Dict:
    config = _review_config()
    review = await prepare_reviewed_word(config, word)
    recommended_code = str(review.get("recommendedCode") or "").strip()
    reviewed_word = str(review.get("word") or word or "").strip()
    if review.get("success") and reviewed_word and recommended_code:
        try:
            review["preSubmitAudit"] = await _build_pre_submit_audit(
                config,
                reviewed_word,
                recommended_code,
            )
        except Exception as error:
            logger.warning(f"[prepare_reviewed_add] pre-submit audit failed: {error}")
            review["preSubmitAudit"] = {
                "success": False,
                "verdict": "needs_admin",
                "autoApprove": False,
                "summary": "加词前预审异常，提交时会重新审核",
                "issues": [str(error)],
                "previewOnly": True,
            }
    return review


async def keytao_audit_draft_items(items: List[Dict]) -> Dict:
    return await audit_draft_items(_review_config(), items)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "keytao_prepare_reviewed_add",
            "description": "加词前审词：核对真实读音，生成键道候选编码和当前占位，并用提交时同一套自动审核逻辑预判推荐编码是否可由本喵自动通过。",
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
