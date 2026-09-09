"""Bounded local commonness queries sharing the review comparator's evidence."""
from __future__ import annotations

from typing import Any, Sequence

from . import keytao_review as review


MAX_COMMONNESS_WORDS = 12
MAX_COMMONNESS_WORD_LENGTH = 64


def validate_commonness_words(words: Any, *, min_words: int = 1) -> list[str]:
    """Validate before opening the reference database; never split model text."""
    if (
        not isinstance(words, (list, tuple))
        or not min_words <= len(words) <= MAX_COMMONNESS_WORDS
    ):
        raise ValueError(f"请提供 {min_words}–{MAX_COMMONNESS_WORDS} 个词。")
    normalized: list[str] = []
    for word in words:
        if not isinstance(word, str):
            raise ValueError("每个词都需要是文字。")
        value = word.strip()
        if (
            not value
            or len(value) > MAX_COMMONNESS_WORD_LENGTH
            or any(ord(char) < 32 or ord(char) == 127 for char in value)
        ):
            raise ValueError("每个词需要为 1–64 个字符，且不能含换行或控制字符。")
        if value in normalized:
            raise ValueError("请去掉重复的词后再查询。")
        normalized.append(value)
    return normalized


def lookup_word_commonness(words: Sequence[str]) -> dict[str, Any]:
    """Return a partial ranking from local evidence, with no remote fallback.

    Rank numbers identify display positions only. A close or unknown relation
    remains explicit even when strict comparisons constrain the display order.
    Semantic review is intentionally absent: it requires evidence unavailable
    to a plain read-only word query and can make model or network requests.
    """
    normalized = validate_commonness_words(words)
    references = {
        word: review._query_commonness_reference(word) for word in normalized
    }
    known_words = [
        word for word in normalized
        if references[word].get("available") is True
        and references[word].get("attested") is True
    ]
    comparisons: list[dict[str, Any]] = []
    edges: dict[str, set[str]] = {word: set() for word in known_words}
    verdicts = {word: "ranked" for word in known_words}
    for left_index, left_word in enumerate(known_words):
        for right_word in known_words[left_index + 1:]:
            comparison = review._compare_reference_commonness(
                left_word, right_word, references[left_word], references[right_word],
            )
            comparisons.append(comparison)
            verdict = comparison["verdict"]
            if verdict in {"front_more_common", "behind_more_common"}:
                winner, loser = (
                    (left_word, right_word) if verdict == "front_more_common"
                    else (right_word, left_word)
                )
                edges[winner].add(loser)
            else:
                row_verdict = "close" if verdict == "close" else "unknown"
                for word in (left_word, right_word):
                    if verdicts[word] != "unknown":
                        verdicts[word] = row_verdict

    ordered = review._stable_commonness_order(known_words, edges)
    conflicting = ordered is None
    if conflicting:
        ordered = known_words
        verdicts = {word: "unknown" for word in known_words}
    rank_by_word = {
        word: index for index, word in enumerate(ordered, start=1)
    }
    display_words = [*ordered, *(word for word in normalized if word not in edges)]
    rows: list[dict[str, Any]] = []
    evidence_lines: list[str] = []
    for word in display_words:
        reference = references[word]
        known = word in edges
        estimate = review._reference_commonness_result(word, reference) if known else {}
        rows.append({
            "word": word,
            "corpusFrequency": reference.get("corpusFrequency") if known else None,
            "dictionaryPresenceCount": reference.get("dictionaryPresenceCount") if known else None,
            "partOfSpeech": reference.get("partOfSpeech") if known else None,
            "known": known,
            "referenceAvailable": reference.get("available") is True,
            "rank": rank_by_word.get(word) if not conflicting else None,
            "verdict": verdicts.get(word, "unknown"),
            "score": estimate.get("score"),
            "signals": estimate.get("signals", {}),
        })
        evidence_lines.append(
            review._comparison_evidence_line(word, estimate) if known
            else f"「{word}」：无数据"
        )
    result = {
        "success": True,
        "method": "offline_reference",
        "referenceAvailable": all(row["referenceAvailable"] for row in rows),
        "words": rows,
        "comparisons": comparisons,
        "evidenceLines": evidence_lines,
        "ordering": "conflicting_evidence" if conflicting else "comparison_edges",
        "orderingNote": (
            "证据方向存在冲突，保留列出顺序，暂不标排名。" if conflicting else
            "按审查比较规则展示；接近或无法判断的关系不表示严格先后，序号仅用于展示。"
        ),
    }
    review.record_commonness_evidence(result)
    return result
