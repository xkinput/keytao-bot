"""Render comparator-owned summaries without querying or reinterpreting evidence."""


def render_commonness_summary(assessment: dict) -> str:
    """The comparator's summary already carries BCC/legacy/semantic evidence."""
    summary = str(assessment.get("summary") or "").strip()
    if summary:
        return summary
    word = str(assessment.get("newWord") or assessment.get("frontWord") or "").strip()
    occupant = str(assessment.get("occupantWord") or assessment.get("behindWord") or "").strip()
    verdict = assessment.get("verdict")
    # Older persisted records may contain a verdict but no evidence snapshot.
    if verdict in {"behind_more_common", "close"}:
        return f"「{occupant}」不弱于「{word}」"
    if verdict == "front_more_common":
        return "原常用度评估建议调序，未保存详细证据"
    if word and occupant:
        return f"「{word}」与「{occupant}」的常用度信号不足"
    return "常用度信号不足"


def candidate_commonness_summary_copy(assessment: dict) -> str:
    """Attach placement advice to the shared evidence, never another verdict."""
    summary = render_commonness_summary(assessment)
    free_code = str(assessment.get("freeCode") or "").strip().lower()
    if not free_code:
        return f"常用度评估：{summary}；维持现有排序。"
    if assessment.get("verdict") in {"behind_more_common", "close"}:
        return f"常用度评估：{summary}，维持现有排序，推荐空位 {free_code}"
    if not assessment.get("summary"):
        return f"常用度评估：{summary}，按空位 {free_code} 推荐"
    return f"常用度评估：{summary}；推荐空位 {free_code}"
