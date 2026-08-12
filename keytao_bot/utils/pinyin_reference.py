"""Read-only access to the vendored pronunciation reference database."""
from __future__ import annotations

import os
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


PINYIN_REFERENCE_DB_ENV = "PINYIN_REFERENCE_DB"
DEFAULT_PINYIN_REFERENCE_DB = (
    Path(__file__).resolve().parents[2] / "data" / "pinyin_reference.db"
)

REFERENCE_DATASET_POLICIES: tuple[dict[str, object], ...] = (
    {
        "id": "zdic_cibs",
        "label": "汉典（离线数据集）",
        "domain": "local-dataset",
        "category": "dictionary",
        "trust": 5,
    },
    {
        "id": "zdic_cybs",
        "label": "汉典（离线数据集）",
        "domain": "local-dataset",
        "category": "dictionary",
        "trust": 5,
    },
    {
        "id": "large_pinyin",
        "label": "开放拼音数据（large_pinyin）",
        "domain": "local-dataset",
        "category": "dictionary",
        "trust": 4,
    },
    {
        "id": "cedict",
        "label": "开放词典数据（CC-CEDICT）",
        "domain": "local-dataset",
        "category": "dictionary",
        "trust": 4,
    },
)
REFERENCE_DATASET_POLICY_BY_ID = {
    str(policy["id"]): policy for policy in REFERENCE_DATASET_POLICIES
}

_PINYIN_CHAR_CLASS = (
    "A-Za-z"
    "üÜvV:"
    "āáǎàōóǒòēéěèīíǐìūúǔùǖǘǚǜ"
    "ĀÁǍÀŌÓǑÒĒÉĚÈĪÍǏÌŪÚǓÙǕǗǙǛ"
    "ńňǹḿ"
    "012345"
)
PINYIN_TOKEN_RE = re.compile(rf"^[{_PINYIN_CHAR_CLASS}]+$")


class PinyinReferenceUnavailable(RuntimeError):
    """The local reference database cannot be queried safely."""


@dataclass(frozen=True)
class ReferenceReading:
    word: str
    normalized: tuple[str, ...]
    display: str
    source_reading: str
    dataset: str


def normalize_pinyin_syllable(value: str) -> str:
    """Normalize one syllable exactly as the pronunciation pipeline matches it."""
    text = (
        value.strip().lower()
        .replace("u:", "v")
        .translate(str.maketrans("üǖǘǚǜ", "vvvvv"))
    )
    text = re.sub(r"[1-5]$", "", text)
    normalized = unicodedata.normalize("NFD", text)
    return "".join(
        char for char in normalized
        if unicodedata.category(char) != "Mn"
    ).replace("ê", "e")


def normalize_pinyin_tokens(tokens: Sequence[str]) -> tuple[str, ...]:
    return tuple(normalize_pinyin_syllable(token) for token in tokens)


def reference_db_path(path: Optional[Path | str] = None) -> Path:
    if path is not None:
        return Path(path)
    configured = os.getenv(PINYIN_REFERENCE_DB_ENV, "").strip()
    return Path(configured) if configured else DEFAULT_PINYIN_REFERENCE_DB


def _read_only_connection(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    if not resolved.is_file():
        raise PinyinReferenceUnavailable(
            f"Pronunciation reference database does not exist: {resolved}"
        )
    try:
        connection = sqlite3.connect(
            f"{resolved.as_uri()}?mode=ro&immutable=1",
            uri=True,
            timeout=0.1,
        )
        connection.execute("PRAGMA query_only = ON")
        return connection
    except sqlite3.Error as error:
        raise PinyinReferenceUnavailable(
            f"Pronunciation reference database cannot be opened read-only: {error}"
        ) from error


def query_reference_readings(
    word: str,
    *,
    db_path: Optional[Path | str] = None,
) -> list[ReferenceReading]:
    """Return all dataset readings for an exact simplified-word key."""
    key = str(word or "").strip()
    if not key:
        return []
    connection = _read_only_connection(reference_db_path(db_path))
    try:
        rows = connection.execute(
            """
            SELECT word, normalized, display, source_reading, dataset
            FROM readings
            WHERE word = ?
            ORDER BY normalized, dataset, display, source_reading
            """,
            (key,),
        ).fetchall()
    except sqlite3.Error as error:
        raise PinyinReferenceUnavailable(
            f"Pronunciation reference query failed: {error}"
        ) from error
    finally:
        connection.close()

    readings: list[ReferenceReading] = []
    for row_word, normalized, display, source_reading, dataset in rows:
        dataset_id = str(dataset)
        if dataset_id not in REFERENCE_DATASET_POLICY_BY_ID:
            raise PinyinReferenceUnavailable(
                f"Pronunciation reference contains unknown dataset: {dataset_id}"
            )
        readings.append(ReferenceReading(
            word=str(row_word),
            normalized=tuple(str(normalized).split()),
            display=str(display),
            source_reading=str(source_reading),
            dataset=dataset_id,
        ))
    return readings
