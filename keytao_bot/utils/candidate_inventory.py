"""Deterministic candidate inventory selection from reviewed server records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class CandidateInventory:
    """One unambiguous ordered candidate group and its reading bindings."""

    candidates: Tuple[Tuple[str, bool], ...]
    statuses: Tuple[Dict[str, Any], ...]
    readings: Dict[str, str]
    group_recommended_code: str = ""


def select_candidate_inventory(
    payload: Mapping[str, Any],
) -> Optional[CandidateInventory]:
    """Select the same unique candidate group for render and revalidation."""
    recommended = str(payload.get("recommendedCode") or "").strip().lower()
    raw_groups: list[tuple[str, str, object, bool]] = []
    top_level = payload.get("candidateStatuses")
    if isinstance(top_level, list):
        raw_groups.append(("", "", top_level, True))
    for pronunciation in payload.get("pronunciations") or []:
        if not isinstance(pronunciation, Mapping):
            continue
        statuses = pronunciation.get("candidateStatuses")
        if isinstance(statuses, list):
            raw_groups.append((
                str(pronunciation.get("pinyin") or "").strip(),
                str(pronunciation.get("recommendedCode") or "").strip().lower(),
                statuses,
                False,
            ))

    valid_groups: list[
        tuple[
            Tuple[Tuple[str, bool], ...],
            Tuple[Dict[str, Any], ...],
            str,
            str,
            bool,
        ]
    ] = []
    readings_by_code: dict[str, set[str]] = {}
    for pinyin, group_recommended, raw_statuses, is_top_level in raw_groups:
        if not isinstance(raw_statuses, list) or not raw_statuses:
            continue
        candidates: list[tuple[str, bool]] = []
        statuses: list[Dict[str, Any]] = []
        seen: set[str] = set()
        valid = True
        for raw_status in raw_statuses:
            if not isinstance(raw_status, Mapping):
                valid = False
                break
            code = str(raw_status.get("code") or "").strip().lower()
            occupied = raw_status.get("occupied")
            if (
                re.fullmatch(r"[a-z]{1,6}", code) is None
                or code in seen
                or not isinstance(occupied, bool)
            ):
                valid = False
                break
            words = [
                str(value or "").strip()
                for value in raw_status.get("words") or []
                if str(value or "").strip()
            ]
            phrases: list[Dict[str, Any]] = []
            entries: list[tuple[str, int]] = []
            for phrase in raw_status.get("phrases") or []:
                if not isinstance(phrase, Mapping):
                    continue
                phrase_word = str(phrase.get("word") or "").strip()
                weight = phrase.get("weight")
                if phrase_word and phrase_word not in words:
                    words.append(phrase_word)
                if (
                    phrase_word
                    and isinstance(weight, int)
                    and not isinstance(weight, bool)
                    and weight >= 0
                ):
                    entries.append((phrase_word, weight))
                    phrases.append({"word": phrase_word, "weight": weight})
            candidates.append((code, occupied))
            statuses.append({
                "code": code,
                "occupied": occupied,
                "words": words,
                "phrases": phrases,
                "entries": tuple(entries),
            })
            seen.add(code)
            if pinyin:
                readings_by_code.setdefault(code, set()).add(pinyin)
        if valid:
            valid_groups.append((
                tuple(candidates),
                tuple(statuses),
                pinyin,
                group_recommended,
                is_top_level,
            ))

    top_level_groups = [group for group in valid_groups if group[4]]
    pronunciation_groups = [group for group in valid_groups if not group[4]]
    preferred = [
        group
        for group in top_level_groups
        if not recommended
        or recommended in {code for code, _occupied in group[0]}
    ]
    if not preferred and pronunciation_groups:
        combined_candidates: list[tuple[str, bool]] = []
        combined_statuses: list[Dict[str, Any]] = []
        combined_seen: set[str] = set()
        combined_valid = True
        combined_recommended = ""
        for candidates, statuses, _pinyin, group_recommended, _is_top in pronunciation_groups:
            codes = {code for code, _occupied in candidates}
            if combined_seen & codes:
                combined_valid = False
                break
            combined_candidates.extend(candidates)
            combined_statuses.extend(statuses)
            combined_seen.update(codes)
            if recommended and group_recommended == recommended:
                combined_recommended = group_recommended
        if combined_valid and (not recommended or recommended in combined_seen):
            preferred = [(
                tuple(combined_candidates),
                tuple(combined_statuses),
                "",
                combined_recommended,
                False,
            )]
    if not preferred:
        preferred = [
            group
            for group in valid_groups
            if not recommended
            or recommended in {code for code, _occupied in group[0]}
        ]
    selected_groups = [
        group
        for group in preferred
        if group[2] and group[3] == recommended
    ]
    selected_groups = selected_groups or preferred
    unique: dict[
        Tuple[Tuple[str, bool], ...],
        tuple[Tuple[Dict[str, Any], ...], str],
    ] = {}
    for candidates, statuses, _pinyin, group_recommended, _is_top in selected_groups:
        unique.setdefault(candidates, (statuses, group_recommended))
    if len(unique) != 1:
        return None
    candidates, (statuses, group_recommended) = next(iter(unique.items()))
    return CandidateInventory(
        candidates=candidates,
        statuses=statuses,
        readings={
            code: next(iter(readings))
            for code, readings in readings_by_code.items()
            if len(readings) == 1
        },
        group_recommended_code=group_recommended,
    )
