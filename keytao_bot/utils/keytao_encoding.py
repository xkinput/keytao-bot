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

_SHUAI_RATE_SUFFIXES = (
    "表率",
    "统率",
    "相率",
    "坦率",
    "直率",
    "轻率",
    "草率",
    "粗率",
    "简率",
    "真率",
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
    normalized = unicodedata.normalize(
        "NFD",
        pinyin.strip().lower()
        .replace("u:", "v")
        .translate(str.maketrans("üǖǘǚǜ", "vvvvv")),
    )
    without_marks = "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    )
    return (
        without_marks
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


def _clean_char_infos(chars: object) -> List[Dict]:
    if not isinstance(chars, list):
        return []

    cleaned: List[Dict] = []
    for item in chars:
        if not isinstance(item, dict):
            continue
        cleaned.append(item)
    return cleaned


def _shape_first_key(char_info: Dict) -> str:
    shape_code = char_info.get("shapeCode")
    return shape_code[:1] if isinstance(shape_code, str) else ""


def _build_progressive_codes(base: str, shape_steps: List[str]) -> List[str]:
    if not base:
        return []

    codes = [base]
    current = base
    for step in shape_steps:
        if not step:
            break
        current += step
        codes.append(current)
    return _clean_code_list(codes)


def _phrase_code_positions(length: int) -> List[int]:
    if length <= 0:
        return []
    if length <= 3:
        return list(range(length))
    return [0, 1, 2, length - 1]


def build_phrase_code_chain(chars: object, phonetic_codes: Optional[List[str]] = None) -> List[str]:
    """Build the standard KeyTao phrase candidate chain for encoded chars."""
    char_infos = _clean_char_infos(chars)
    if not char_infos:
        return []

    if phonetic_codes is None:
        phonetic_codes = [
            code if isinstance(code, str) else ""
            for code in (item.get("phoneticCode") for item in char_infos)
        ]
    if len(phonetic_codes) != len(char_infos) or any(not code for code in phonetic_codes):
        return []

    length = len(char_infos)
    if length == 1:
        return build_single_char_code_chain(phonetic_codes[0], char_infos[0].get("shapeCode"))

    if length == 2:
        base = "".join(phonetic_codes)
        shape_steps = [_shape_first_key(item) for item in char_infos]
    elif length == 3:
        base = "".join(code[:1] for code in phonetic_codes)
        shape_steps = [_shape_first_key(item) for item in char_infos]
    else:
        positions = _phrase_code_positions(length)
        base = "".join(phonetic_codes[index][:1] for index in positions)
        shape_steps = [_shape_first_key(char_infos[0]), _shape_first_key(char_infos[1])]

    return _build_progressive_codes(base, shape_steps)


def normalize_contextual_phrase_encoding(word: str, encode_data: Dict) -> Dict:
    """Keep productive -率 words on the contextual lǜ candidate chain."""
    if not isinstance(encode_data, dict):
        return {}

    chars = _clean_char_infos(encode_data.get("chars"))
    if (
        len(chars) <= 1
        or len(chars) != len(word)
        or chars[-1].get("char") != "率"
        or any(word.endswith(suffix) for suffix in _SHUAI_RATE_SUFFIXES)
    ):
        return encode_data

    rate_pinyins = chars[-1].get("pinyins")
    if not isinstance(rate_pinyins, list):
        return encode_data
    contextual_pinyin = next(
        (
            item for item in rate_pinyins
            if isinstance(item, str) and _strip_pinyin_tone(item) == "lv"
        ),
        None,
    )
    if not contextual_pinyin:
        return encode_data

    contextual_phonetic = pinyin_to_phonetic_code(contextual_pinyin)
    if not contextual_phonetic:
        return encode_data

    normalized_chars = [dict(item) for item in chars]
    rate_info = normalized_chars[-1]
    rate_info["pinyin"] = contextual_pinyin
    rate_info["pinyins"] = [
        contextual_pinyin,
        *[
            item for item in rate_pinyins
            if isinstance(item, str) and item != contextual_pinyin
        ],
    ]
    rate_info["phoneticCode"] = contextual_phonetic
    shape_code = rate_info.get("shapeCode")
    if isinstance(shape_code, str):
        rate_info["fullCode"] = contextual_phonetic + shape_code

    codes = build_phrase_code_chain(normalized_chars)
    if not codes:
        return encode_data

    normalized = dict(encode_data)
    normalized["chars"] = normalized_chars
    normalized["codes"] = codes
    normalized["contextPinyinCorrected"] = True
    return normalized


def build_alternate_pronunciation_codes(chars: object) -> List[Dict]:
    char_infos = _clean_char_infos(chars)
    if len(char_infos) != 1:
        return []

    char_info = char_infos[0]
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


def build_phrase_pronunciation_codes(chars: object) -> List[Dict]:
    """Build bounded pronunciation variants for polyphonic chars inside a phrase."""
    char_infos = _clean_char_infos(chars)
    if len(char_infos) <= 1:
        return []

    default_phonetic_codes = [
        code if isinstance(code, str) else ""
        for code in (item.get("phoneticCode") for item in char_infos)
    ]
    if any(not code for code in default_phonetic_codes):
        return []

    variants: List[Dict] = []
    seen_keys = set()
    for index, char_info in enumerate(char_infos):
        pinyins = char_info.get("pinyins", [])
        if not isinstance(pinyins, list) or len(pinyins) <= 1:
            continue

        default_pinyin = char_info.get("pinyin", "")
        default_phonetic = default_phonetic_codes[index]
        seen_phonetics = set()
        for pinyin in pinyins:
            if not isinstance(pinyin, str) or not pinyin.strip():
                continue
            phonetic_code = pinyin_to_phonetic_code(pinyin)
            if not phonetic_code or phonetic_code in seen_phonetics:
                continue
            seen_phonetics.add(phonetic_code)
            is_default = phonetic_code == default_phonetic or pinyin == default_pinyin
            if is_default:
                continue

            phonetic_codes = list(default_phonetic_codes)
            phonetic_codes[index] = phonetic_code
            standard_codes = build_phrase_code_chain(char_infos, phonetic_codes)
            codes = _clean_code_list(standard_codes)
            if not codes:
                continue

            key = (index, phonetic_code, tuple(codes))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            variants.append({
                "char": char_info.get("char", ""),
                "charIndex": index,
                "pinyin": pinyin,
                "phoneticCode": phonetic_code,
                "codes": codes,
                "standardCodes": standard_codes,
                "isDefault": False,
            })
    return variants
