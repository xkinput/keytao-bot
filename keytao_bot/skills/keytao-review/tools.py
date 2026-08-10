"""KeyTao review skill tools."""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

from nonebot.log import logger

from keytao_bot.utils import http_client
from keytao_bot.utils.keytao_review import (
    ReviewHttpConfig,
    assess_candidate_chain_commonness,
    audit_draft_items,
    can_llm_override_audit_issues,
    prepare_reviewed_word,
)
from keytao_bot.utils.review_flags import (
    apply_manual_review_flag,
    apply_review_disposition,
    read_manual_review_flag,
    read_review_disposition,
)


def get_keytao_url() -> str:
    """Thin wrapper over the shared HTTP config."""
    return http_client.get_keytao_url()


def get_bot_token() -> str:
    """Thin wrapper over the shared HTTP config."""
    return http_client.get_bot_token() or ""


def _review_config() -> ReviewHttpConfig:
    return ReviewHttpConfig(api_base=get_keytao_url(), bot_token=get_bot_token())


# Preview items are never persisted, but the LLM pre-review keys items by id and
# `_extract_items` drops anything without an int id — without one the preview
# silently produced no verdict at all. Negative ids can never collide with a
# real pull-request id.
_preview_item_ids = itertools.count(-1, -1)


def _preview_create_item(word: str, code: str) -> Dict[str, Any]:
    return {
        "id": next(_preview_item_ids),
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
        review_result = await review_keytao_batch_with_llm(
            batch,
            precomputed_audit=deterministic_audit,
        )
        if not review_result.get("success"):
            return None

        ai_review = review_result.get("aiReview") or {}
        review_items = ai_review.get("items") if isinstance(ai_review.get("items"), list) else []
        non_pass_items = [
            review_item for review_item in review_items
            if isinstance(review_item, dict) and review_item.get("status") != "pass"
        ]
        # A structured verdict is terminal. If the deterministic audit already
        # sealed this item, an LLM "pass" must not write the flag back to false
        # nor set autoApprove — the preview would then advertise auto-approval
        # for something code has explicitly reserved for an admin.
        if (
            read_manual_review_flag(deterministic_audit) is True
            and deterministic_audit.get("structuredManualReviewIssues")
        ):
            logger.info(
                "[prepare_reviewed_add] structured manual-review verdict is terminal; "
                "ignoring LLM preview override"
            )
            return None

        if ai_review.get("verdict") == "pass" and not non_pass_items and review_items:
            word = str(item.get("word") or "")
            code = str(item.get("code") or "")
            return apply_manual_review_flag({
                **deterministic_audit,
                "success": True,
                "verdict": "pass",
                "autoApprove": True,
                "summary": ai_review.get("headline") or "语言常识、读音和编码检查一致",
                "issues": [],
                "approvedItems": [f"Create：{word}@{code}，本喵审核通过"],
                "llmReview": ai_review,
                "llmFallback": True,
                "previewOnly": True,
            }, False, "语言常识、读音和编码检查一致")

        issues: List[str] = []
        for review_item in non_pass_items[:5]:
            reasons = review_item.get("reasons") if isinstance(review_item.get("reasons"), list) else []
            title = review_item.get("title") or f"PR#{review_item.get('prId')} 需要复核"
            issues.append(str(reasons[0] if reasons else title))
        resolved_issues = issues or deterministic_audit.get("issues", [])
        return apply_manual_review_flag({
            **deterministic_audit,
            "summary": ai_review.get("headline") or deterministic_audit.get("summary", "存在不确定项，需要管理员审核"),
            "issues": resolved_issues,
            "llmReview": ai_review,
            "llmFallback": True,
            "previewOnly": True,
        }, True, str(resolved_issues[0]) if resolved_issues else "存在不确定项，需要管理员审核")
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


def _semantic_requester_for_actor(
    platform: Optional[str],
    platform_id: Optional[str],
) -> Optional[str]:
    normalized_platform = str(platform or "").strip()
    normalized_id = str(platform_id or "").strip()
    if not normalized_platform or not normalized_id:
        return None
    return f"bot-review:actor:{normalized_platform}:{normalized_id}"


async def keytao_prepare_reviewed_add(
    word: str,
    platform: Optional[str] = None,
    platform_id: Optional[str] = None,
) -> Dict:
    config = _review_config()
    semantic_requester = _semantic_requester_for_actor(platform, platform_id)
    if semantic_requester:
        review = await prepare_reviewed_word(
            config,
            word,
            semantic_requester=semantic_requester,
        )
    else:
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
            review["preSubmitAudit"] = apply_review_disposition(apply_manual_review_flag({
                "success": False,
                "verdict": "needs_admin",
                "autoApprove": False,
                "summary": "加词前预审异常，提交时会重新审核",
                "issues": [str(error)],
                "previewOnly": True,
            }, True, "加词前预审异常"), "pre_submit_review_unavailable")

    # Surface the pre-submit verdict on the top-level review payload so remark
    # builders can read a code-generated boolean instead of parsing Chinese text.
    pre_submit = review.get("preSubmitAudit")
    if isinstance(pre_submit, dict):
        base_disposition = read_review_disposition(review)
        flag = read_manual_review_flag(pre_submit)
        if flag is None:
            flag = not bool(pre_submit.get("autoApprove"))
        reason = str(
            (pre_submit.get("issues") or [""])[0] if flag else pre_submit.get("summary") or ""
        )
        if base_disposition is not None:
            flag = True
            reason = str(review.get("manualReviewReason") or reason)
        apply_manual_review_flag(review, bool(flag), reason)
        if flag and read_review_disposition(review) is None:
            apply_review_disposition(review, "pre_submit_judgement")
    review["candidateOrderingAssessments"] = await assess_candidate_chain_commonness(
        review
    )
    return review


async def keytao_audit_draft_items(items: List[Dict]) -> Dict:
    return await audit_draft_items(_review_config(), items)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "keytao_prepare_reviewed_add",
            "description": "加词前审词：核对真实读音，生成键道候选编码和当前占位，评估空位前占用词的常用度排序，并用提交时同一套自动审核逻辑预判推荐编码是否可由本喵自动通过。",
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
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                        },
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
