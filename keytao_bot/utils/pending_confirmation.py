"""Shared reply forms and copy for deterministic pending confirmations."""

from __future__ import annotations


PENDING_CONFIRM_ADVERTISED_FORMS = ("确认", "执行")
PENDING_CONFIRM_ASSENT_TEXTS = frozenset({
    *PENDING_CONFIRM_ADVERTISED_FORMS,
    "确定",
    "好的",
    "好",
    "是",
    "对",
    "可以",
    "行",
    "同意",
    "就这样",
    "按这个",
    "执行吧",
})

PENDING_BATCH_ADD_ADVERTISED_FORMS = ("加入", "都加", "添加")
PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS = (
    "加入并提交",
    "都加并提交",
    "添加并提交",
)
PENDING_BATCH_ADD_ASSENT_TEXTS = frozenset({
    *PENDING_BATCH_ADD_ADVERTISED_FORMS,
    "加",
    "确认加入",
    "确认添加",
    "继续加入",
    "继续添加",
    "全部加",
})
PENDING_BATCH_ADD_AND_SUBMIT_ASSENT_TEXTS = frozenset({
    *PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS,
    "加并提交",
    "新增并提交",
})
PENDING_BATCH_ASSENT_TEXTS = frozenset({
    *PENDING_BATCH_ADD_ASSENT_TEXTS,
    *PENDING_BATCH_ADD_AND_SUBMIT_ASSENT_TEXTS,
})
PENDING_ASSENT_TEXTS = frozenset({
    *PENDING_CONFIRM_ASSENT_TEXTS,
    *PENDING_BATCH_ASSENT_TEXTS,
})
PENDING_BATCH_CONFIRMATION_COPY_TOKEN = "{{PENDING_BATCH_CONFIRMATION_COPY}}"


def _quoted_choices(forms: tuple[str, ...]) -> str:
    return "、".join(f"「{form}」" for form in forms)


def pending_confirmation_copy() -> str:
    """Render the generic forms accepted by a single actor-owned ticket."""
    return f"回复{_quoted_choices(PENDING_CONFIRM_ADVERTISED_FORMS)}继续。"


def pending_batch_confirmation_copy() -> str:
    """Render every advertised add-only and add-then-submit form."""
    return (
        f"回复{_quoted_choices(PENDING_BATCH_ADD_ADVERTISED_FORMS)}只加入草稿；"
        f"回复{_quoted_choices(PENDING_BATCH_ADD_AND_SUBMIT_ADVERTISED_FORMS)}"
        "则加入后提交。"
    )


def pending_confirmation_prompt_instruction() -> str:
    """Render model guidance from the same forms used by the parser."""
    return (
        "多词候选消息末尾必须逐字使用以下确认文案：\n"
        + pending_batch_confirmation_copy()
    )


def expand_pending_confirmation_copy(text: str) -> str:
    """Expand prompt templates without duplicating user-facing reply forms."""
    return str(text).replace(
        PENDING_BATCH_CONFIRMATION_COPY_TOKEN,
        pending_batch_confirmation_copy(),
    )
