"""Validate Single encode records without deriving phrase candidates."""

from __future__ import annotations

import re
from typing import Any

from .keytao_encoding import pinyin_to_phonetic_code
from .pinyin_reference import normalize_pinyin_syllable


# Audited CC-CEDICT entry, vendored in keytao-next/data/reference-review/cedict.txt
# line 112108 on 2026-09-07. The source header attributes CC-CEDICT contributors
# and licenses the dataset under https://creativecommons.org/licenses/by-sa/4.0/.
# This factual projection retains the attribution and keeps
# the runtime independent of a sibling checkout and never infers variants.
# Source entry: 鎗 枪 [qiang1] /variant of 槍|枪[qiang1]/rifle/spear/
_CHARACTER_VARIANTS = {
    ("鎗", "qiang"): {
        "canonical": "枪",
        "traditional": "槍",
        "source": "CC-CEDICT（离线数据）",
        "sourceUrl": "https://cc-cedict.org/",
    },
}


def single_character_encoding(
    word: str,
    encoding: dict[str, Any],
    *,
    requested_reading: str = "",
    requested_meaning: str = "",
) -> dict[str, Any]:
    """Project only reading-bound codes supplied by a valid Single response."""
    failure = {
        "success": False, "word": word, "type": "Single",
        "encodeServiceConfirmed": False,
        "upstreamTransient": bool(encoding.get("upstreamTransient")),
        "message": f"编码服务无法为「{word}」生成单字编码。",
    }
    if failure["upstreamTransient"]:
        failure["message"] = f"编码服务暂时不可用，未能核验「{word}」的单字编码；可稍后重试。"
    chars = encoding.get("chars")
    if (
        encoding.get("success") is False
        or encoding.get("type") != "单字"
        or not isinstance(chars, list) or len(chars) != 1
        or not isinstance(chars[0], dict) or chars[0].get("char") != word
    ):
        return failure
    char = dict(chars[0])
    default_pinyin = str(char.get("pinyin") or "").strip()
    default_reading = normalize_pinyin_syllable(default_pinyin)
    known_readings = {
        normalize_pinyin_syllable(value)
        for value in char.get("pinyins") or []
        if isinstance(value, str) and value.strip()
    }
    if not default_reading or default_reading not in known_readings:
        return {**failure, "message": f"编码服务没有返回「{word}」可核验的单字读音，本次不推荐编码。"}
    shape = char.get("shapeCode")
    if shape is not None and (not isinstance(shape, str) or re.fullmatch(r"[a-z]*", shape) is None):
        return failure

    def validated_codes(pinyin: str, raw_codes: object) -> list[str]:
        phonetic = pinyin_to_phonetic_code(pinyin)
        if not phonetic or not isinstance(raw_codes, list) or not raw_codes:
            return []
        single_chain = [phonetic + (shape or "")[:index] for index in range(min(4, len(shape or "")) + 1)]
        if any(not isinstance(code, str) or code not in single_chain for code in raw_codes):
            return []
        codes = list(dict.fromkeys(raw_codes))
        if codes != [code for code in single_chain if code in codes]:
            return []
        return codes

    codes = validated_codes(default_pinyin, encoding.get("codes"))
    if not codes or char.get("phoneticCode") != pinyin_to_phonetic_code(default_pinyin):
        return failure
    reading = default_reading
    pinyin = default_pinyin
    requested_text = str(requested_reading or "").strip()
    character_hint = re.fullmatch(
        rf"{re.escape(word)}(?:字)?(?:的)?(?:读音)?(?:=|是|为|读作|读成|读)\s*(.+)",
        requested_text,
    )
    requested = normalize_pinyin_syllable(
        character_hint.group(1) if character_hint is not None else requested_text,
    )
    if requested and requested != default_reading:
        match = next((
            item for item in encoding.get("alternatePronunciationCodes") or []
            if isinstance(item, dict)
            and normalize_pinyin_syllable(str(item.get("pinyin") or "")) == requested
            and requested in known_readings
        ), None)
        if match is None or not (codes := validated_codes(str(match.get("pinyin") or ""), match.get("codes"))):
            return {**failure, "message": f"编码服务未返回「{word}」读音 {requested} 的单字候选编码，本次不推荐编码。"}
        pinyin, reading = str(match["pinyin"]), requested
    elif requested_meaning and not requested:
        return {**failure, "message": f"请明确「{word}」要采用的读音，才能核验对应的单字编码。"}
    result = {
        "success": True, "word": word, "type": "Single", "encodingType": "单字",
        "chars": [char], "encodeServiceConfirmed": True,
        "pronunciations": [{
            "pinyin": pinyin, "normalized": [reading], "codes": codes,
            "sources": [], "fallback": True, "requiresManualReview": True,
            "readingEvidenceKind": "own_character",
            "sourceSummary": "编码服务逐字读音",
        }],
    }
    relation = _CHARACTER_VARIANTS.get((word, reading))
    if relation:
        result["characterVariant"] = dict(relation)
        result["variantNote"] = f"「{word}」是「{relation['canonical']}」（繁体「{relation['traditional']}」）的异体字"
        result["pronunciations"][0]["sources"] = [{
            "source": relation["source"], "category": "dictionary",
        }]
    if shape is None:
        result["encodingNote"] = "编码服务未提供该字的拆分形码，当前仅返回音码候选"
    return result
