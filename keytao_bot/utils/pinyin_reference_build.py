"""Deterministic importer for the vendored pronunciation datasets."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, TextIO

from .pinyin_reference import (
    PINYIN_TOKEN_RE,
    REFERENCE_DATASET_POLICY_BY_ID,
    normalize_pinyin_syllable,
)


SCHEMA_VERSION = "1"
BUILDER_VERSION = "2"
MANIFEST_FILENAME = "manifest.json"
EXCLUSIONS_FILENAME = "excluded_words.txt"
_CEDICT_LINE_RE = re.compile(r"^(\S+)\s+(\S+)\s+\[([^\]]+)\]\s+/")
_TONE_MARKS = {
    "a": "āáǎà",
    "e": "ēéěè",
    "i": "īíǐì",
    "o": "ōóǒò",
    "u": "ūúǔù",
    "ü": "ǖǘǚǜ",
}
_COMBINING_TONES = {1: "\u0304", 2: "\u0301", 3: "\u030c", 4: "\u0300"}


@dataclass(frozen=True)
class ParsedReading:
    word: str
    normalized: str
    display: str
    source_reading: str


@dataclass(frozen=True)
class DatasetBuildCount:
    lines: int
    parsed: int
    imported: int
    duplicates: int
    excluded: int
    skipped: int


@dataclass(frozen=True)
class BuildResult:
    rebuilt: bool
    source_checksum: str
    build_fingerprint: str
    word_count: int
    reading_count: int
    dataset_counts: dict[str, DatasetBuildCount]

    def as_json_dict(self) -> dict[str, object]:
        return {
            "rebuilt": self.rebuilt,
            "source_checksum": self.source_checksum,
            "build_fingerprint": self.build_fingerprint,
            "word_count": self.word_count,
            "reading_count": self.reading_count,
            "dataset_counts": {
                dataset: asdict(count)
                for dataset, count in sorted(self.dataset_counts.items())
            },
        }


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _sha256_uncompressed(path: Path) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(source_dir: Path) -> dict[str, object]:
    path = source_dir / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("formatVersion") != 1:
        raise ValueError("Unsupported pronunciation dataset manifest")
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("Pronunciation dataset manifest has no datasets")
    return payload


def _load_exclusions(source_dir: Path) -> set[str]:
    path = source_dir / EXCLUSIONS_FILENAME
    if not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _verified_sources(
    source_dir: Path,
    manifest: dict[str, object],
) -> tuple[list[dict[str, object]], str]:
    datasets = manifest["datasets"]
    assert isinstance(datasets, list)
    verified: list[dict[str, object]] = []
    combined = hashlib.sha256()
    seen_ids: set[str] = set()
    for raw_entry in datasets:
        if not isinstance(raw_entry, dict):
            raise ValueError("Pronunciation dataset manifest entry is not an object")
        dataset = str(raw_entry.get("id") or "")
        filename = str(raw_entry.get("file") or "")
        expected = str(raw_entry.get("sha256") or "")
        if not dataset or dataset in seen_ids or not filename or len(expected) != 64:
            raise ValueError(f"Invalid pronunciation dataset manifest entry: {raw_entry}")
        seen_ids.add(dataset)
        source_path = source_dir / filename
        actual = _sha256_uncompressed(source_path)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {filename}: expected {expected}, got {actual}"
            )
        entry = dict(raw_entry)
        entry["path"] = source_path
        verified.append(entry)
        combined.update(dataset.encode("utf-8"))
        combined.update(b"\0")
        combined.update(actual.encode("ascii"))
        combined.update(b"\0")
    return verified, combined.hexdigest()


def numbered_syllable_to_tone_marks(value: str) -> str:
    """Convert one CC-CEDICT numbered syllable to marked pinyin."""
    syllable = value.strip().replace("u:", "ü").replace("U:", "Ü")
    syllable = syllable.replace("v", "ü").replace("V", "Ü")
    match = re.fullmatch(r"(.+?)([0-5])", syllable)
    if not match:
        return syllable
    base = match.group(1)
    tone = int(match.group(2))
    if tone in {0, 5}:
        return base

    lowered = base.lower()
    if "a" in lowered:
        mark_index = lowered.index("a")
    elif "e" in lowered:
        mark_index = lowered.index("e")
    elif "ou" in lowered:
        mark_index = lowered.index("o")
    else:
        vowel_indexes = [
            index for index, char in enumerate(lowered)
            if char in _TONE_MARKS
        ]
        mark_index = vowel_indexes[-1] if vowel_indexes else -1

    if mark_index >= 0:
        original = base[mark_index]
        marked = _TONE_MARKS[original.lower()][tone - 1]
        if original.isupper():
            marked = marked.upper()
        return base[:mark_index] + marked + base[mark_index + 1:]
    if lowered and lowered[-1] in {"m", "n"}:
        return base + _COMBINING_TONES[tone]
    return base


def numbered_sequence_to_tone_marks(value: str) -> tuple[str, ...]:
    converted: list[str] = []
    for token in value.split():
        parts = [part for part in re.split(r"['’\-]", token) if part]
        converted.extend(numbered_syllable_to_tone_marks(part) for part in parts)
    return tuple(converted)


def parse_phrase_line(line: str) -> Optional[ParsedReading]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    word, separator, raw_reading = stripped.partition(":")
    word = word.strip()
    source_reading = " ".join(raw_reading.split("#", 1)[0].split())
    if not separator or not word or not source_reading:
        return None
    tokens = tuple(source_reading.split())
    if not tokens or not all(PINYIN_TOKEN_RE.fullmatch(token) for token in tokens):
        return None
    normalized = tuple(normalize_pinyin_syllable(token) for token in tokens)
    if not all(normalized):
        return None
    return ParsedReading(
        word=word,
        normalized=" ".join(normalized),
        display=source_reading,
        source_reading=source_reading,
    )


def parse_cedict_line(line: str) -> Optional[ParsedReading]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    match = _CEDICT_LINE_RE.match(stripped)
    if not match:
        return None
    _traditional, simplified, source_reading = match.groups()
    display_tokens = numbered_sequence_to_tone_marks(source_reading)
    normalized = tuple(normalize_pinyin_syllable(token) for token in display_tokens)
    if not simplified or not display_tokens or not all(normalized):
        return None
    return ParsedReading(
        word=simplified,
        normalized=" ".join(normalized),
        display=" ".join(display_tokens),
        source_reading=source_reading,
    )


def _iter_parsed(
    source_path: Path,
    source_format: str,
) -> Iterator[tuple[bool, Optional[ParsedReading]]]:
    parser = parse_cedict_line if source_format == "cedict" else parse_phrase_line
    with _open_text(source_path) as handle:
        for line in handle:
            data_line = bool(line.strip() and not line.lstrip().startswith("#"))
            yield data_line, parser(line)


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        return dict(connection.execute("SELECT key, value FROM metadata"))
    except sqlite3.Error:
        return {}


def _existing_result(
    db_path: Path,
    source_checksum: str,
    build_fingerprint: str,
) -> Optional[BuildResult]:
    if not db_path.is_file():
        return None
    try:
        connection = sqlite3.connect(
            f"{db_path.resolve().as_uri()}?mode=ro&immutable=1",
            uri=True,
        )
        metadata = _metadata(connection)
        if (
            metadata.get("schema_version") != SCHEMA_VERSION
            or metadata.get("source_checksum") != source_checksum
            or metadata.get("build_fingerprint") != build_fingerprint
        ):
            return None
        raw_counts = json.loads(metadata.get("dataset_counts", "{}"))
        dataset_counts = {
            dataset: DatasetBuildCount(**counts)
            for dataset, counts in raw_counts.items()
        }
        return BuildResult(
            rebuilt=False,
            source_checksum=source_checksum,
            build_fingerprint=build_fingerprint,
            word_count=int(metadata["word_count"]),
            reading_count=int(metadata["reading_count"]),
            dataset_counts=dataset_counts,
        )
    except (KeyError, TypeError, ValueError, sqlite3.Error):
        return None
    finally:
        if "connection" in locals():
            connection.close()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        PRAGMA page_size = 4096;
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = MEMORY;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE readings (
            word TEXT NOT NULL,
            normalized TEXT NOT NULL,
            display TEXT NOT NULL,
            source_reading TEXT NOT NULL,
            dataset TEXT NOT NULL CHECK (
                dataset IN ('zdic_cibs', 'zdic_cybs', 'large_pinyin', 'cedict')
            ),
            PRIMARY KEY (word, dataset, normalized, display, source_reading)
        ) WITHOUT ROWID;
    """)


def _batched_insert(
    connection: sqlite3.Connection,
    rows: Iterable[tuple[str, str, str, str, str]],
    *,
    batch_size: int = 10_000,
) -> None:
    batch: list[tuple[str, str, str, str, str]] = []
    statement = """
        INSERT OR IGNORE INTO readings
            (word, normalized, display, source_reading, dataset)
        VALUES (?, ?, ?, ?, ?)
    """
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            connection.executemany(statement, batch)
            batch.clear()
    if batch:
        connection.executemany(statement, batch)


def build_reference_database(source_dir: Path | str, db_path: Path | str) -> BuildResult:
    source_root = Path(source_dir).resolve()
    destination = Path(db_path).resolve()
    manifest = _load_manifest(source_root)
    sources, source_checksum = _verified_sources(source_root, manifest)
    exclusions = _load_exclusions(source_root)
    exclusion_checksum = hashlib.sha256(
        "\n".join(sorted(exclusions)).encode("utf-8")
    ).hexdigest()
    build_fingerprint = hashlib.sha256(
        f"{source_checksum}\0{exclusion_checksum}\0{SCHEMA_VERSION}\0{BUILDER_VERSION}".encode(
            "ascii"
        )
    ).hexdigest()

    existing = _existing_result(destination, source_checksum, build_fingerprint)
    if existing is not None:
        return existing

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        temporary.unlink()

    connection = sqlite3.connect(temporary)
    dataset_counts: dict[str, DatasetBuildCount] = {}
    try:
        _create_schema(connection)
        for source in sources:
            if source.get("import") is not True:
                continue
            dataset = str(source["id"])
            if dataset not in REFERENCE_DATASET_POLICY_BY_ID:
                raise ValueError(f"Unknown imported pronunciation dataset: {dataset}")
            source_path = Path(source["path"])
            source_format = str(source.get("format") or "phrase")
            lines = parsed = excluded = skipped = 0
            rows: list[tuple[str, str, str, str, str]] = []
            for data_line, reading in _iter_parsed(source_path, source_format):
                if data_line:
                    lines += 1
                if reading is None:
                    if data_line:
                        skipped += 1
                    continue
                parsed += 1
                if reading.word in exclusions:
                    excluded += 1
                    continue
                rows.append((
                    reading.word,
                    reading.normalized,
                    reading.display,
                    reading.source_reading,
                    dataset,
                ))
                if len(rows) >= 10_000:
                    _batched_insert(connection, rows)
                    rows.clear()
            if rows:
                _batched_insert(connection, rows)
            imported = int(connection.execute(
                "SELECT COUNT(*) FROM readings WHERE dataset = ?",
                (dataset,),
            ).fetchone()[0])
            dataset_counts[dataset] = DatasetBuildCount(
                lines=lines,
                parsed=parsed,
                imported=imported,
                duplicates=max(0, parsed - excluded - imported),
                excluded=excluded,
                skipped=skipped,
            )

        reading_count = int(connection.execute(
            "SELECT COUNT(*) FROM readings"
        ).fetchone()[0])
        word_count = int(connection.execute(
            "SELECT COUNT(DISTINCT word) FROM readings"
        ).fetchone()[0])
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "source_checksum": source_checksum,
            "build_fingerprint": build_fingerprint,
            "word_count": str(word_count),
            "reading_count": str(reading_count),
            "dataset_counts": json.dumps(
                {
                    dataset: asdict(count)
                    for dataset, count in sorted(dataset_counts.items())
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        }
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            sorted(metadata.items()),
        )
        connection.commit()
        connection.execute("VACUUM")
        connection.close()
        temporary.replace(destination)
    except BaseException:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise

    return BuildResult(
        rebuilt=True,
        source_checksum=source_checksum,
        build_fingerprint=build_fingerprint,
        word_count=word_count,
        reading_count=reading_count,
        dataset_counts=dataset_counts,
    )
