"""Bot-owned presentation preferences that refine service candidate ordering."""

from typing import Dict, Iterable, List


def _dedupe_codes(values: Iterable[object]) -> List[str]:
    return list(dict.fromkeys(
        str(value or "").strip().lower()
        for value in values
        if str(value or "").strip()
    ))


def _zhe_chain_parts(values: Iterable[object]) -> tuple[List[str], List[str], List[str]]:
    codes = _dedupe_codes(values)
    return (
        [code for code in codes if code.startswith("fe")],
        [code for code in codes if code.startswith("qe")],
        [code for code in codes if not code.startswith(("fe", "qe"))],
    )


def _reorder_statuses(payload: Dict, ordered_codes: List[str]) -> None:
    statuses = payload.get("candidateStatuses")
    if not isinstance(statuses, list):
        return
    status_by_code = {
        str(item.get("code") or "").strip().lower(): item
        for item in statuses
        if isinstance(item, dict) and str(item.get("code") or "").strip()
    }
    payload["candidateStatuses"] = [
        status_by_code[code]
        for code in ordered_codes
        if code in status_by_code
    ]


def apply_zhe_fe_chain_preference(
    word: str,
    payload: Dict,
    *,
    include_alternative_in_codes: bool = False,
) -> Dict:
    """Lead 哲(zhe) words with fe*, retaining qe* as the alternative chain."""
    if not str(word or "").strip().startswith("哲") or not isinstance(payload, dict):
        return payload
    combined = _dedupe_codes([
        *(payload.get("codes") or []),
        *(payload.get("altCodes") or []),
        *(payload.get("candidateCodes") or []),
    ])
    fe_codes, qe_codes, other_codes = _zhe_chain_parts(combined)
    if not fe_codes or not qe_codes:
        return payload

    ordered_codes = [*fe_codes, *qe_codes, *other_codes]
    payload["codes"] = (
        ordered_codes
        if include_alternative_in_codes
        else [*fe_codes, *other_codes]
    )
    payload["altCodes"] = qe_codes
    payload["candidateCodes"] = ordered_codes
    payload["baseCode"] = fe_codes[0]
    _reorder_statuses(payload, ordered_codes)
    payload["recommendedCode"] = fe_codes[0]

    for pronunciation in payload.get("pronunciations") or []:
        if not isinstance(pronunciation, dict):
            continue
        pron_codes = _dedupe_codes(pronunciation.get("codes") or [])
        pron_fe, pron_qe, pron_other = _zhe_chain_parts(pron_codes)
        if not pron_fe or not pron_qe:
            continue
        ordered_pron_codes = [*pron_fe, *pron_qe, *pron_other]
        pronunciation["codes"] = ordered_pron_codes
        _reorder_statuses(pronunciation, ordered_pron_codes)
        pronunciation["recommendedCode"] = pron_fe[0]
    pronunciations = [
        pronunciation
        for pronunciation in payload.get("pronunciations") or []
        if isinstance(pronunciation, dict)
    ]
    if pronunciations:
        payload["recommendedCode"] = next((
            str(pronunciation.get("recommendedCode") or "").strip()
            for pronunciation in pronunciations
            if str(pronunciation.get("recommendedCode") or "").strip()
        ), payload["recommendedCode"])
    return payload
