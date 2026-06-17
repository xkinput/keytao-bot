"""Shared helpers for deriving KeyTao candidate code chains."""
import re
import unicodedata
from typing import Dict, List, Optional


_FINAL_CODE_MAP = {
    "iu": "q",
    "ua": "q",
    "ei": "w",
    "un": "w",
    "vn": "w",
    "e": "e",
    "eng": "r",
    "ueng": "r",
    "ng": "r",
    "uan": "t",
    "van": "t",
    "iong": "y",
    "ong": "y",
    "ang": "p",
    "a": "s",
    "ia": "s",
    "ie": "d",
    "ou": "d",
    "an": "f",
    "ing": "g",
    "uai": "g",
    "ai": "h",
    "ue": "h",
    "ve": "h",
    "er": "j",
    "u": "j",
    "i": "k",
    "o": "l",
    "uo": "l",
    "v": "l",
    "ao": "z",
    "iang": "x",
    "uang": "x",
    "iao": "c",
    "in": "b",
    "ui": "b",
    "en": "n",
    "ian": "m",
}

_ZERO_INITIAL_CODES = {
    "a": "xs",
    "ai": "xh",
    "an": "xf",
    "ang": "xp",
    "ao": "xz",
    "e": "xe",
    "ei": "xw",
    "en": "xn",
    "eng": "xr",
    "er": "xj",
    "o": "xl",
    "ou": "xd",
}

_CH_INITIAL_BY_FINAL = {
    "ai": "j",
    "an": "j",
    "ang": "j",
    "en": "j",
    "eng": "j",
    "u": "j",
    "un": "j",
    "ao": "j",
    "e": "j",
    "a": "w",
    "i": "w",
    "ong": "w",
    "ou": "w",
    "ua": "w",
    "uai": "w",
    "uan": "w",
    "uang": "w",
    "ui": "w",
    "uo": "w",
}

_ZH_INITIAL_BY_FINAL = {
    "an": "q",
    "ang": "q",
    "ei": "q",
    "en": "q",
    "eng": "q",
    "u": "q",
    "un": "q",
    "ai": "q",
    "ao": "q",
    "e": "q",
    "a": "f",
    "i": "f",
    "ong": "f",
    "ou": "f",
    "ua": "f",
    "uai": "f",
    "uan": "f",
    "uang": "f",
    "ui": "f",
    "uo": "f",
}

_INITIALS = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w",
)


def _clean_code_list(codes: object) -> List[str]:
    if not isinstance(codes, list):
        return []

    result: List[str] = []
    seen = set()
    for code in codes:
        if not isinstance(code, str):
            continue
        normalized = code.strip().lower()
        if normalized and "?" not in normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _strip_pinyin_tone(pinyin: str) -> str:
    normalized = unicodedata.normalize("NFD", pinyin.strip().lower())
    without_marks = "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    )
    return (
        without_marks
        .replace("ü", "v")
        .replace("u:", "v")
        .replace("ê", "e")
    )


def _split_pinyin(pinyin: str) -> Optional[tuple[str, str]]:
    plain = _strip_pinyin_tone(pinyin)
    plain = re.sub(r"[^a-zv]", "", plain)
    if not plain:
        return None

    if plain in _ZERO_INITIAL_CODES:
        return "", plain

    for initial in _INITIALS:
        if plain.startswith(initial) and len(plain) > len(initial):
            return initial, plain[len(initial):]
    return "", plain


def _normalize_final(initial: str, final: str) -> str:
    if final == "iong":
        return "iong"
    if initial in {"j", "q", "x", "y"}:
        if final == "u":
            return "v"
        if final.startswith("u"):
            possible_v_final = "v" + final[1:]
            if possible_v_final in _FINAL_CODE_MAP:
                return possible_v_final
    return final


def pinyin_to_phonetic_code(pinyin: str) -> Optional[str]:
    split = _split_pinyin(pinyin)
    if not split:
        return None
    initial, raw_final = split

    if not initial:
        normalized_final = _normalize_final(initial, raw_final)
        return _ZERO_INITIAL_CODES.get(normalized_final)

    final = _normalize_final(initial, raw_final)
    final_code = _FINAL_CODE_MAP.get(final)
    if not final_code:
        return None

    if initial == "ch":
        initial_code = _CH_INITIAL_BY_FINAL.get(final)
    elif initial == "zh":
        initial_code = _ZH_INITIAL_BY_FINAL.get(final)
    elif initial == "sh":
        initial_code = "e"
    else:
        initial_code = initial

    if not initial_code:
        return None
    return f"{initial_code}{final_code}"


def build_single_char_code_chain(phonetic_code: str, shape_code: object) -> List[str]:
    if not phonetic_code or not isinstance(shape_code, str):
        return []
    return _clean_code_list(
        [
            phonetic_code + shape_code[:index]
            for index in range(0, len(shape_code) + 1)
        ]
    )


def build_alternate_pronunciation_codes(chars: object) -> List[Dict]:
    if not isinstance(chars, list) or len(chars) != 1:
        return []

    char_info = chars[0]
    if not isinstance(char_info, dict):
        return []

    pinyins = char_info.get("pinyins", [])
    if not isinstance(pinyins, list):
        return []

    default_pinyin = char_info.get("pinyin", "")
    default_phonetic = char_info.get("phoneticCode", "")
    shape_code = char_info.get("shapeCode")

    variants: List[Dict] = []
    seen_codes = set()
    for pinyin in pinyins:
        if not isinstance(pinyin, str) or not pinyin.strip():
            continue
        phonetic_code = pinyin_to_phonetic_code(pinyin)
        if not phonetic_code or phonetic_code in seen_codes:
            continue
        code_chain = build_single_char_code_chain(phonetic_code, shape_code)
        if not code_chain:
            continue
        seen_codes.add(phonetic_code)
        variants.append({
            "pinyin": pinyin,
            "phoneticCode": phonetic_code,
            "codes": code_chain,
            "isDefault": phonetic_code == default_phonetic or pinyin == default_pinyin,
        })
    return variants
