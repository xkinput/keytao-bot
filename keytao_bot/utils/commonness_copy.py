"""Render comparator-owned evidence without querying for another snapshot."""

import math


def render_commonness_summary(assessment: dict) -> str:
    """The comparator's summary already carries BCC/legacy/semantic evidence."""
    summary = str(assessment.get("summary") or "").strip()
    if summary:
        return summary
    word = str(assessment.get("newWord") or assessment.get("frontWord") or "").strip()
    occupant = str(assessment.get("occupantWord") or assessment.get("behindWord") or "").strip()
    # Older persisted records may contain a verdict but no evidence snapshot.
    if assessment.get("verdict") == "front_more_common":
        return ""
    if word and occupant:
        return f"「{word}」与「{occupant}」的常用度信号不足"
    return "常用度信号不足"


def _numeric_commonness_value(value: object):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value) and value >= 0:
            return value
    return None


def _comparison_frequency(commonness: dict, reason: str, web_fallback: bool):
    if web_fallback:
        return _numeric_commonness_value(commonness.get("score"))
    reference = commonness.get("reference") or {}
    if reason == "dictionary_presence_margin":
        return _numeric_commonness_value(reference.get("dictionaryPresenceCount"))
    bcc = commonness.get("bcc") or reference.get("bcc") or {}
    historical = "historical" in reason
    rows = bcc.get("historicalChannels" if historical else "channels") or []
    if historical:
        channels = ("近代汉语", "古代汉语")
        if reason.startswith("bcc_historical_"):
            channels = ("近代汉语" if "early_modern" in reason else "古代汉语",)
        for channel in channels:
            selected = [row for row in rows if row.get("channel") == channel
                        and _numeric_commonness_value(row.get("count")) is not None]
            if selected:
                break
        rows = selected
    elif reason.startswith(("frequency_", "corpus_")):
        return _numeric_commonness_value(reference.get("corpusFrequency"))
    elif "balanced_tiebreak" in reason:
        rows = [row for row in rows if row.get("channel") == "多领域"]
    signal = "rankFraction" if "relative_rank" in reason else "perMillion"
    observed = [row for row in rows
                if _numeric_commonness_value(row.get("count")) is not None
                and _numeric_commonness_value(row.get(signal)) is not None]
    if observed:
        strongest = min(observed, key=lambda row: row[signal]) if signal == "rankFraction" else max(
            observed, key=lambda row: row[signal],
        )
        if reason.startswith("bcc_"):
            value = strongest[signal]
            return (1 / value if value else None) if signal == "rankFraction" else value
        return strongest["count"]
    return _numeric_commonness_value(reference.get("corpusFrequency"))


def comparison_frequency_values(assessment: dict) -> tuple:
    """Read persisted numeric strings or the original comparator snapshot."""
    if "frontFrequency" in assessment or "behindFrequency" in assessment:
        values = []
        for key in ("frontFrequency", "behindFrequency"):
            raw = assessment.get(key, "")
            try:
                value = int(raw) if isinstance(raw, str) and raw.isdigit() else float(raw)
            except (ValueError, TypeError):
                value = None
            values.append(_numeric_commonness_value(value))
        return tuple(values)
    reason = str(assessment.get("decisionReason") or "")
    return tuple(_comparison_frequency(assessment.get(side) or {}, reason,
                                    assessment.get("webFallback") is True)
                 for side in ("front", "behind"))


def render_commonness_comparison(assessment: dict) -> str:
    """Show only snapshot values, ordered numerically rather than by verdict."""
    if assessment.get("decisionReason") == "modern_semantic_vs_dictionary_dominated":
        return ""
    words = (
        str(assessment.get("newWord") or assessment.get("frontWord") or "").strip(),
        str(assessment.get("occupantWord") or assessment.get("behindWord") or "").strip(),
    )
    values = comparison_frequency_values(assessment)
    if not all(words) or all(value is None for value in values):
        return ""
    sides = list(zip(words, values))
    sides.sort(key=lambda item: (item[1] is not None, item[1] or 0), reverse=True)
    formatted = [f"{word} {int(value) if value == int(value) else value}" if value is not None else f"{word} 无"
                 for word, value in sides]
    operator = "=" if values[0] is not None and values[0] == values[1] else ">"
    return f"{formatted[0]} {operator} {formatted[1]}"


def candidate_commonness_summary_copy(assessment: dict) -> str:
    comparison = render_commonness_comparison(assessment)
    return f"词频：{comparison}" if comparison else ""
