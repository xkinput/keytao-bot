"""Bounded local and web primitives for unknown polyphonic words."""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Optional
from urllib.parse import urlparse

from .pinyin_reference import (
    PinyinReferenceUnavailable,
    ReferenceReading,
    normalize_pinyin_syllable,
    query_overlapping_reference_readings,
    query_reference_readings,
    reference_db_path,
)
from .pinyin_reference_build import numbered_syllable_to_tone_marks


_PINYIN_CHAR_CLASS = (
    "A-Za-z"
    "üÜvV:"
    "āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜ"
    "ĀÁǍÀŌÓǑÒĒÉĚÈĪÍǏÌŪÚǓÙǕǗǙǛ"
    "ńňǹḿ"
    "012345"
)
_PINYIN_SYLLABLE = rf"[{_PINYIN_CHAR_CLASS}]+"
_PINYIN_SEPARATOR = r"[\s·,/\\-]+"
_HIGH_TRUST_DOMAIN_SUFFIXES = (
    "baike.baidu.com",
    "hanyu.baidu.com",
    "zdic.net",
    "hwxnet.com",
    "5156edu.com",
    "dict.cn",
    "med66.com",
    "yixue.com",
    "terms.naer.edu.tw",
)
_HOSTNAME_RE = re.compile(
    r"(?=.{1,253}\Z)"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)"
    r"(?:\.(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?))*\Z"
)

PRONUNCIATION_RESOLUTION_CACHE_DB_ENV = "PRONUNCIATION_RESOLUTION_CACHE_DB"
DEFAULT_PRONUNCIATION_RESOLUTION_CACHE_DB = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "pronunciation_resolution_cache.db"
)
PRONUNCIATION_CACHE_SCHEMA_VERSION = 2
POSITIVE_CACHE_TTL_SECONDS = 24 * 60 * 60
NEGATIVE_CACHE_TTL_SECONDS = 15 * 60


def _cacheable_web_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep only scored readings and validated domains in durable storage."""
    candidates: list[dict[str, Any]] = []
    raw_candidates = [
        candidate
        for candidate in payload.get("candidates") or []
        if isinstance(candidate, dict)
    ]
    selected = payload.get("selected")
    if isinstance(selected, dict) and selected not in raw_candidates:
        raw_candidates.append(selected)
    for candidate in raw_candidates:
        normalized = [
            normalize_pinyin_syllable(str(value or ""))
            for value in candidate.get("normalized") or []
        ]
        if not normalized or not all(normalized):
            continue
        domains: list[str] = []
        raw_domains = [
            *(candidate.get("domains") or []),
            *(
                evidence.get("domain")
                for evidence in candidate.get("evidence") or []
                if isinstance(evidence, dict)
            ),
        ]
        for value in raw_domains:
            domain = str(value or "").strip().lower().rstrip(".")
            if is_valid_pronunciation_domain(domain) and domain not in domains:
                domains.append(domain)
        candidates.append({"normalized": normalized, "domains": domains})
    selected_normalized = (
        [
            normalize_pinyin_syllable(str(value or ""))
            for value in selected.get("normalized") or []
        ]
        if isinstance(selected, dict)
        else []
    )
    return {
        "status": str(payload.get("status") or "no_evidence"),
        "selectedNormalized": selected_normalized,
        "candidates": candidates,
    }


def _hydrate_cached_web_result(payload: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for cached in payload.get("candidates") or []:
        if not isinstance(cached, dict):
            continue
        normalized = [
            normalize_pinyin_syllable(str(value or ""))
            for value in cached.get("normalized") or []
        ]
        domains = [
            str(value or "").strip().lower().rstrip(".")
            for value in cached.get("domains") or []
            if is_valid_pronunciation_domain(str(value or ""))
        ]
        if not normalized or not all(normalized):
            continue
        evidence = [
            {"domain": domain, "highTrust": _is_high_trust_domain(domain)}
            for domain in dict.fromkeys(domains)
        ]
        candidates.append({
            "normalized": normalized,
            "display": " ".join(normalized),
            "independentResultCount": len(evidence),
            "highTrustDomains": [
                row["domain"] for row in evidence if row["highTrust"]
            ],
            "evidence": evidence,
        })
    selected_normalized = tuple(payload.get("selectedNormalized") or [])
    selected = next((
        candidate
        for candidate in candidates
        if tuple(candidate["normalized"]) == selected_normalized
    ), None)
    return {
        "status": str(payload.get("status") or "no_evidence"),
        "selected": selected,
        "candidates": candidates,
        "resultCount": sum(
            candidate["independentResultCount"] for candidate in candidates
        ),
    }


class PronunciationResolutionCache:
    """Small durable SQLite cache beside the pronunciation reference data."""

    def __init__(self, path: Optional[Path | str] = None) -> None:
        self.path = Path(path) if path is not None else pronunciation_resolution_cache_path()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=2.0)
        connection.execute("PRAGMA busy_timeout = 2000")
        columns = connection.execute(
            "PRAGMA table_info(pronunciation_resolution_cache)"
        ).fetchall()
        primary_key = {
            str(row[1]): int(row[5])
            for row in columns
            if int(row[5]) > 0
        }
        if columns and primary_key != {"word": 1, "version": 2}:
            connection.execute("DROP TABLE pronunciation_resolution_cache")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pronunciation_resolution_cache (
                word TEXT NOT NULL,
                version INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                positive INTEGER NOT NULL CHECK (positive IN (0, 1)),
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                PRIMARY KEY (word, version)
            )
            """
        )
        return connection

    def get(self, word: str, *, now: Optional[float] = None) -> Optional[dict[str, Any]]:
        key = str(word or "").strip()
        if not key:
            return None
        current = time.time() if now is None else float(now)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT result_json, expires_at
                FROM pronunciation_resolution_cache
                WHERE word = ? AND version = ?
                """,
                (key, PRONUNCIATION_CACHE_SCHEMA_VERSION),
            ).fetchone()
            if row is None:
                return None
            if float(row[1]) <= current:
                connection.execute(
                    """
                    DELETE FROM pronunciation_resolution_cache
                    WHERE word = ? AND version = ?
                    """,
                    (key, PRONUNCIATION_CACHE_SCHEMA_VERSION),
                )
                return None
        try:
            payload = json.loads(str(row[0]))
        except (TypeError, ValueError):
            return None
        return _hydrate_cached_web_result(payload) if isinstance(payload, dict) else None

    def set(
        self,
        word: str,
        payload: dict[str, Any],
        *,
        positive: bool,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        key = str(word or "").strip()
        if not key:
            return payload
        created_at = time.time() if now is None else float(now)
        ttl = POSITIVE_CACHE_TTL_SECONDS if positive else NEGATIVE_CACHE_TTL_SECONDS
        serialized = json.dumps(
            _cacheable_web_result(payload),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO pronunciation_resolution_cache (
                    word, version, result_json, positive, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(word, version) DO UPDATE SET
                    result_json = excluded.result_json,
                    positive = excluded.positive,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (
                    key,
                    PRONUNCIATION_CACHE_SCHEMA_VERSION,
                    serialized,
                    1 if positive else 0,
                    created_at,
                    created_at + ttl,
                ),
            )
        return payload


def pronunciation_resolution_cache_path() -> Path:
    configured = os.getenv(PRONUNCIATION_RESOLUTION_CACHE_DB_ENV, "").strip()
    return Path(configured) if configured else DEFAULT_PRONUNCIATION_RESOLUTION_CACHE_DB


@lru_cache(maxsize=2048)
def _cached_character_readings(
    character: str,
    resolved_db_path: str,
    database_mtime_ns: int,
    database_size: int,
) -> tuple[ReferenceReading, ...]:
    del database_mtime_ns, database_size
    return tuple(query_reference_readings(character, db_path=resolved_db_path))


def _character_readings(
    character: str,
    db_path: Optional[Path | str],
) -> tuple[ReferenceReading, ...]:
    resolved = reference_db_path(db_path).resolve()
    try:
        stat = resolved.stat()
        identity = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        identity = (0, 0)
    return _cached_character_readings(
        character,
        str(resolved),
        identity[0],
        identity[1],
    )


def detect_polyphonic_characters(
    word: str,
    *,
    db_path: Optional[Path | str] = None,
) -> dict[int, tuple[str, ...]]:
    """Return positions whose character has multiple normalized readings."""
    result: dict[int, tuple[str, ...]] = {}
    readings_by_character: dict[str, tuple[ReferenceReading, ...]] = {}
    for index, character in enumerate(str(word or "")):
        if character not in readings_by_character:
            readings_by_character[character] = _character_readings(
                character,
                db_path,
            )
        readings = readings_by_character[character]
        options = tuple(sorted({
            reading.normalized[0]
            for reading in readings
            if len(reading.normalized) == 1 and reading.normalized[0]
        }))
        if len(options) > 1:
            result[index] = options
    return result


def _fixed_carrier_reading(
    rows: list[ReferenceReading],
) -> Optional[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
    valid_rows = [
        row for row in rows
        if len(row.normalized) == len(row.word) and all(row.normalized)
    ]
    sequences = {row.normalized for row in valid_rows}
    if len(sequences) != 1:
        return None
    sequence = next(iter(sequences))
    matching = [row for row in valid_rows if row.normalized == sequence]
    display_counts = Counter(
        tuple(row.display.split())
        for row in matching
        if len(row.display.split()) == len(row.word)
    )
    display = (
        display_counts.most_common(1)[0][0]
        if display_counts
        else sequence
    )
    datasets = tuple(dict.fromkeys(row.dataset for row in matching))
    return sequence, display, datasets


def resolve_compositional_pronunciation(
    word: str,
    *,
    db_path: Optional[Path | str] = None,
) -> Optional[dict[str, Any]]:
    """Resolve every polyphonic position from fixed overlapping local words."""
    key = str(word or "").strip()
    if len(key) < 2:
        return None
    polyphones = detect_polyphonic_characters(key, db_path=db_path)
    if not polyphones:
        return None

    rows_by_word: dict[str, list[ReferenceReading]] = defaultdict(list)
    for row in query_overlapping_reference_readings(key, db_path=db_path):
        rows_by_word[row.word].append(row)

    normalized: list[str] = [""] * len(key)
    display: list[str] = [""] * len(key)
    for index, character in enumerate(key):
        if index in polyphones:
            continue
        character_rows = _character_readings(character, db_path)
        options = {
            row.normalized[0]
            for row in character_rows
            if len(row.normalized) == 1 and row.normalized[0]
        }
        if len(options) != 1:
            return None
        option = next(iter(options))
        normalized[index] = option
        display_options = [
            row.display
            for row in character_rows
            if row.normalized == (option,) and row.display
        ]
        display[index] = Counter(display_options).most_common(1)[0][0] if display_options else option

    used_carriers: list[dict[str, Any]] = []
    for carrier_word in sorted(rows_by_word, key=lambda value: (-len(value), value)):
        fixed = _fixed_carrier_reading(rows_by_word[carrier_word])
        if fixed is None:
            continue
        carrier_sequence, carrier_display, datasets = fixed
        if key in carrier_word:
            target_start = 0
            carrier_start = carrier_word.index(key)
            shared_length = len(key)
        else:
            continue
        covered_polyphones = [
            index
            for index in polyphones
            if target_start <= index < target_start + shared_length
        ]
        if not covered_polyphones:
            continue
        proposed = {
            target_index: carrier_sequence[carrier_start + target_index - target_start]
            for target_index in covered_polyphones
        }
        if any(normalized[index] and normalized[index] != value for index, value in proposed.items()):
            return None
        for index, value in proposed.items():
            normalized[index] = value
            display[index] = carrier_display[carrier_start + index - target_start]
        used_carriers.append({
            "word": carrier_word,
            "datasets": list(datasets),
            "polyphoneIndexes": covered_polyphones,
        })

    if not used_carriers or any(not normalized[index] for index in polyphones):
        return None
    primary = used_carriers[0]
    return {
        "normalized": normalized,
        "display": " ".join(display),
        "carrierWord": primary["word"],
        "carriers": used_carriers,
        "sourceSummary": f"组合推断（{primary['word']}）",
    }


def _canonical_domain(url: str) -> str:
    parsed = urlparse(str(url or ""))
    if parsed.scheme.lower() not in {"http", "https"}:
        return ""
    domain = str(parsed.hostname or "").lower().rstrip(".")
    if not _HOSTNAME_RE.fullmatch(domain):
        return ""
    return domain[4:] if domain.startswith("www.") else domain


def is_valid_pronunciation_domain(domain: str) -> bool:
    return bool(_HOSTNAME_RE.fullmatch(str(domain or "").strip().lower().rstrip(".")))


def _is_high_trust_domain(domain: str) -> bool:
    return any(
        domain == suffix or domain.endswith("." + suffix)
        for suffix in _HIGH_TRUST_DOMAIN_SUFFIXES
    )


def _display_pinyin_tokens(tokens: list[str]) -> list[str]:
    return [numbered_syllable_to_tone_marks(token.lower()) for token in tokens]


def extract_web_pronunciations(
    word: str,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract word-bound pinyin sequences from search titles and snippets."""
    key = str(word or "").strip()
    if not key:
        return []
    sequence = _PINYIN_SEPARATOR.join(
        f"({_PINYIN_SYLLABLE})" for _ in key
    )
    patterns = (
        re.compile(
            rf"{re.escape(key)}.{{0,24}}?(?:拼音|讀音|读音|汉语拼音|漢語拼音|pinyin)"
            rf"\s*[:：是为]?\s*[\[【（(]?\s*{sequence}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"{re.escape(key)}\s*[\[【（(]\s*{sequence}\s*[\]】）)]",
            re.IGNORECASE,
        ),
    )
    extracted: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        title = str(result.get("title") or "").strip()[:180]
        snippet = str(result.get("snippet") or "").strip()[:360]
        text = " ".join(value for value in (title, snippet) if value)
        if key not in text:
            continue
        url = str(result.get("url") or "").strip()
        domain = _canonical_domain(url)
        if not domain:
            continue
        seen_sequences: set[tuple[str, ...]] = set()
        for pattern in patterns:
            for match in pattern.finditer(text):
                raw_tokens = [str(value or "").strip() for value in match.groups()]
                normalized = tuple(normalize_pinyin_syllable(token) for token in raw_tokens)
                if len(normalized) != len(key) or not all(normalized):
                    continue
                if normalized in seen_sequences:
                    continue
                seen_sequences.add(normalized)
                display_tokens = _display_pinyin_tokens(raw_tokens)
                extracted.append({
                    "normalized": list(normalized),
                    "display": " ".join(display_tokens),
                    "domain": domain,
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "provider": str(result.get("provider") or "").strip(),
                    "highTrust": _is_high_trust_domain(domain),
                })
    return extracted


def score_web_pronunciations(
    word: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score independent-domain agreement without hiding disagreement."""
    extracted = extract_web_pronunciations(word, results)
    by_sequence: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for evidence in extracted:
        by_sequence[tuple(evidence["normalized"])].append(evidence)

    candidates: list[dict[str, Any]] = []
    for normalized, evidence_rows in by_sequence.items():
        independent: dict[str, dict[str, Any]] = {}
        for row in evidence_rows:
            domain = str(row.get("domain") or "")
            identity = domain or f"provider:{row.get('provider') or row.get('url')}"
            independent.setdefault(identity, row)
        rows = list(independent.values())
        displays = [str(row.get("display") or "") for row in rows]
        display = max(
            displays,
            key=lambda value: (
                sum(ord(char) > 127 for char in value),
                -displays.index(value),
            ),
            default=" ".join(normalized),
        )
        high_trust_domains = sorted({
            str(row.get("domain") or "")
            for row in rows
            if row.get("highTrust") and row.get("domain")
        })
        candidates.append({
            "normalized": list(normalized),
            "display": display,
            "independentResultCount": len(rows),
            "highTrustDomains": high_trust_domains,
            "evidence": rows,
        })
    candidates.sort(
        key=lambda item: (
            -bool(item["highTrustDomains"]),
            -int(item["independentResultCount"]),
            tuple(item["normalized"]),
        )
    )

    high_trust_sequences = [
        item for item in candidates if item["highTrustDomains"]
    ]
    independently_agreed = [
        item for item in candidates if item["independentResultCount"] >= 2
    ]
    accepted = high_trust_sequences or independently_agreed
    if len(candidates) > 1 and (
        len(accepted) != 1
        or len(high_trust_sequences) > 1
        or not accepted
    ):
        status = "disagreement"
        selected = None
    elif len(accepted) == 1:
        status = "resolved"
        selected = accepted[0]
    elif len(candidates) > 1:
        status = "disagreement"
        selected = None
    elif candidates:
        status = "weak"
        selected = None
    else:
        status = "no_evidence"
        selected = None
    return {
        "status": status,
        "selected": selected,
        "candidates": candidates,
        "resultCount": len(extracted),
    }


__all__ = [
    "NEGATIVE_CACHE_TTL_SECONDS",
    "POSITIVE_CACHE_TTL_SECONDS",
    "PRONUNCIATION_CACHE_SCHEMA_VERSION",
    "PinyinReferenceUnavailable",
    "PRONUNCIATION_RESOLUTION_CACHE_DB_ENV",
    "PronunciationResolutionCache",
    "detect_polyphonic_characters",
    "extract_web_pronunciations",
    "is_valid_pronunciation_domain",
    "resolve_compositional_pronunciation",
    "score_web_pronunciations",
]
