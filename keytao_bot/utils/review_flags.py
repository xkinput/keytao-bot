"""Structured auto-review verdict flags shared by the review pipeline.

The safety gates that decide whether a draft batch may be auto-approved must
read a boolean produced by review code, never a Chinese substring produced by
an LLM. This module owns:

  * the canonical remark prefixes (rendered by code from the boolean)
  * the structured field names carried on draft items and review results
  * helpers to read the verdict back out of an item

String matching against :data:`MANUAL_REVIEW_PREFIXES` remains only as a
compatibility path for items already persisted server-side before the
structured field existed; those prefixes are code-generated constants, so they
are still safe to trust in the "needs manual review" direction.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

from .pending_confirmation import render_remediation_reply

# Structured field carried on draft items / review results.
MANUAL_REVIEW_FIELD = "needsManualReview"
MANUAL_REVIEW_FIELD_SNAKE = "needs_manual_review"
MANUAL_REVIEW_REASON_FIELD = "manualReviewReason"


class ReviewDisposition(str, Enum):
    """Declared review outcomes at the write boundary."""

    PASS = "PASS"
    BLOCK = "BLOCK"
    SEAL = "SEAL"


REVIEW_DISPOSITION_FIELD = "reviewDisposition"
REVIEW_VERDICT_SITE_FIELD = "reviewVerdictSite"

# Every add/review verdict site is named here. PASS is an explicit, narrowly
# defined auto-pass condition; BLOCK means the target cannot be written safely;
# SEAL means the target is trustworthy enough to write with
# needsManualReview=True so an admin decides at approval time.
REVIEW_VERDICT_SITE_POLICIES = {
    "semantic_context_common_word": ReviewDisposition.PASS,
    "empty_word": ReviewDisposition.BLOCK,
    "pronunciation_unresolved": ReviewDisposition.BLOCK,
    "code_unresolved": ReviewDisposition.BLOCK,
    "lookup_unavailable": ReviewDisposition.BLOCK,
    "invalid_code": ReviewDisposition.BLOCK,
    "injection_shaped_input": ReviewDisposition.BLOCK,
    "missing_authoritative_page": ReviewDisposition.SEAL,
    "pronunciation_lookup_incomplete": ReviewDisposition.SEAL,
    "entity_context_reading": ReviewDisposition.SEAL,
    "unvalidated_type": ReviewDisposition.SEAL,
    "pre_submit_judgement": ReviewDisposition.SEAL,
    "pre_submit_review_unavailable": ReviewDisposition.SEAL,
    "duplicate_formation": ReviewDisposition.SEAL,
    "code_chain_priority": ReviewDisposition.SEAL,
}


def review_disposition_for_site(site: str) -> ReviewDisposition:
    """Return the declared disposition for one verdict-producing site."""
    try:
        return REVIEW_VERDICT_SITE_POLICIES[site]
    except KeyError as error:
        raise ValueError(f"Undeclared review verdict site: {site}") from error


def apply_review_disposition(
    payload: Dict[str, Any],
    site: str,
) -> Dict[str, Any]:
    """Stamp one declared PASS/BLOCK/SEAL outcome onto a payload in place."""
    disposition = review_disposition_for_site(site)
    payload[REVIEW_DISPOSITION_FIELD] = disposition.value
    payload[REVIEW_VERDICT_SITE_FIELD] = site
    return payload


def read_review_disposition(payload: Any) -> Optional[ReviewDisposition]:
    """Read a declared disposition value, rejecting unknown values."""
    if not isinstance(payload, dict):
        return None
    raw = payload.get(REVIEW_DISPOSITION_FIELD)
    if isinstance(raw, ReviewDisposition):
        return raw
    if isinstance(raw, str):
        try:
            return ReviewDisposition(raw.strip().upper())
        except ValueError:
            return None
    return None


def review_blocks_write(payload: Any) -> bool:
    """Only an explicit BLOCK disposition prevents a reviewed add write."""
    return read_review_disposition(payload) is ReviewDisposition.BLOCK

# Canonical, code-generated remark fragments. Never let an LLM author these.
AUTO_PASS_PREFIX = "自动审核：该词可自动通过"
MANUAL_REVIEW_PREFIX = "自动审核：该词需管理员审核"
PENDING_PREAUDIT_PREFIX = "自动审核：该词暂未完成预审"

# Legacy/compat markers still recognised on already-persisted remarks.
MANUAL_REVIEW_PREFIXES = (
    MANUAL_REVIEW_PREFIX,
    "自动审核:该词需管理员审核",
    "自动审核：该词需要管理员审核",
    "自动审核:该词需要管理员审核",
    "自动审核：预计需管理员审核",
    "自动审核:预计需管理员审核",
    "自动审核：预计需要管理员审核",
    "自动审核:预计需要管理员审核",
    "自动审核：需管理员审核",
    "自动审核:需管理员审核",
    PENDING_PREAUDIT_PREFIX,
    "自动审核:该词暂未完成预审",
)


def build_auto_review_remark(needs_manual_review: bool, reason: str = "") -> str:
    """Render the canonical remark fragment for a structured verdict."""
    prefix = MANUAL_REVIEW_PREFIX if needs_manual_review else AUTO_PASS_PREFIX
    reason = (reason or "").strip()
    if not reason and needs_manual_review:
        reason = "证据不足"
    return f"{prefix}（{reason}）" if reason else prefix


def read_manual_review_flag(payload: Any) -> Optional[bool]:
    """Return the structured verdict from an item/review dict, or ``None``.

    ``None`` means "the item carries no structured verdict", which callers
    should treat as unknown and fall back to their conservative default.
    """
    if not isinstance(payload, dict):
        return None
    for key in (MANUAL_REVIEW_FIELD, MANUAL_REVIEW_FIELD_SNAKE):
        value = payload.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
    return None


def manual_review_reason(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get(MANUAL_REVIEW_REASON_FIELD) or "").strip()


def apply_manual_review_flag(
    payload: Dict[str, Any],
    needs_manual_review: bool,
    reason: str = "",
) -> Dict[str, Any]:
    """Stamp the structured verdict onto a dict in place and return it."""
    payload[MANUAL_REVIEW_FIELD] = bool(needs_manual_review)
    if reason:
        payload[MANUAL_REVIEW_REASON_FIELD] = reason
    return payload


def item_requires_manual_review(item: Any) -> bool:
    """Whether a draft item is sealed for manual review.

    Structured field first; the code-generated remark prefix is only consulted
    when the item carries no structured verdict at all (rows persisted before
    the field existed). LLM prose can therefore never clear the seal, and an
    explicit ``False`` is honoured as an explicit clearance.
    """
    flag = read_manual_review_flag(item)
    if flag is not None:
        return flag
    if not isinstance(item, dict):
        return False
    return bool(remark_indicates_manual_review(item.get("remark")))


def audit_allows_batch_auto_approve(auto_review: Any) -> bool:
    """Return whether an audit is complete enough to call auto-approval."""
    return bool(
        isinstance(auto_review, dict)
        and auto_review.get("success") is True
        and auto_review.get("autoApprove") is True
        and auto_review.get("verdict") == "pass"
        and not (auto_review.get("issues") or [])
        and bool(auto_review.get("approvedItems") or [])
        and not auto_review.get("manualReviewLocked")
        and not auto_review.get("encodeOnly")
    )


def batch_auto_approve_block_reason(auto_review: Any) -> str:
    """Explain why the canonical batch predicate refuses auto-approval."""
    if not isinstance(auto_review, dict) or auto_review.get("success") is not True:
        return render_remediation_reply(
            "自动审核未完整完成；转交管理员属于站外处理"
        )
    if auto_review.get("manualReviewLocked"):
        return "批次包含已锁定的人工复核项"
    if auto_review.get("encodeOnly"):
        return "当前仅完成编码校验，尚未完成逐项自动核验"
    summary = str(auto_review.get("summary") or "").strip()
    if auto_review.get("issues") or []:
        return summary or "批次包含需管理员复核项"
    if not (auto_review.get("approvedItems") or []):
        return "纯删除项或缺少词条/编码，无法完成逐项自动核验"
    if auto_review.get("autoApprove") is not True or auto_review.get("verdict") != "pass":
        return summary or "审核结论未达到自动批准条件"
    return ""


def remark_indicates_manual_review(remark: Any) -> str:
    """Return the matched legacy marker in a remark string, or ``""``."""
    text = str(remark or "").strip()
    if not text:
        return ""
    return next((marker for marker in MANUAL_REVIEW_PREFIXES if marker in text), "")
