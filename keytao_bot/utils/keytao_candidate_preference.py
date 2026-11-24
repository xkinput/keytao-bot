"""Bot-owned presentation preferences that refine service candidate ordering."""

from typing import Iterable, List

_PREFERRED_INITIALS = {"zhe": "f", "che": "j", "chu": "j"}


def order_reading_codes(pinyins: Iterable[str], codes: Iterable[str]) -> List[str]:
    """Rank legal candidates by preferred initials at the encoded positions."""
    from .keytao_encoding import _phrase_code_positions, _strip_pinyin_tone, scheme_phonetic_bases

    syllables = list(pinyins)
    ordered = _dedupe_codes(codes)
    if any(not isinstance(syllable, str) for syllable in syllables):
        return ordered
    stride = 2 if len(syllables) <= 2 else 1
    preferred = [
        (slot * stride, key)
        for slot, index in enumerate(_phrase_code_positions(len(syllables)))
        if (key := _PREFERRED_INITIALS.get(_strip_pinyin_tone(syllables[index]).rstrip("12345")))
    ]
    if not preferred:
        return ordered
    bases = scheme_phonetic_bases(syllables)
    if len(bases) < 2:
        return ordered
    base_length = len(bases[0])
    rank = {
        base: sum(base[slot] != key for slot, key in preferred)
        for base in bases
    }
    return sorted(ordered, key=lambda code: rank.get(code[:base_length], len(preferred) + 1))


def _dedupe_codes(values: Iterable[object]) -> List[str]:
    return list(dict.fromkeys(
        str(value or "").strip().lower()
        for value in values
        if str(value or "").strip()
    ))
