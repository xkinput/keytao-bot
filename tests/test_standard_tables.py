"""Offline checks for the reference database's standard tables."""

import hashlib
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from keytao_bot.utils import keytao_review as review, pinyin_reference_build as builder
from keytao_bot.utils.bcc_reference import create_schema as create_bcc_schema


VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "pinyin_reference"
UNAVAILABLE = {"generalLevel": None, "hskLevel": None, "available": False}


class StandardTablesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.db = self.root / "reference.db"

    def write_manifest(self, datasets):
        (self.root / "manifest.json").write_text(
            json.dumps({"formatVersion": 1, "datasets": datasets}), encoding="utf-8",
        )

    def write_hsk_fixture(self, payload, *, sha256=None):
        (self.root / "hsk.tsv").write_bytes(payload)
        self.write_manifest([{
            "id": "hsk3", "file": "hsk.tsv", "format": "hsk_word", "import": True,
            "sha256": sha256 or hashlib.sha256(payload).hexdigest(),
        }])

    def seed_word_commonness(self, connection):
        connection.execute("""
            CREATE TABLE word_commonness (
                word TEXT PRIMARY KEY, corpus_frequency INTEGER, part_of_speech TEXT,
                dictionary_presence_count INTEGER
            )
        """)

    def test_imports_vendored_standards_without_commonness_rows(self):
        manifest = json.loads((VENDOR / "manifest.json").read_text(encoding="utf-8"))
        datasets = [entry for entry in manifest["datasets"] if entry["id"] in {"tgscc", "hsk3"}]
        for entry in datasets:
            (self.root / entry["file"]).symlink_to(VENDOR / entry["file"])
        self.write_manifest(datasets)

        result = builder.build_reference_database(self.root, self.db)
        self.assertEqual(result.commonness_word_count, 0)
        self.assertEqual(result.reading_count, 0)
        self.assertEqual({key: value.imported for key, value in result.dataset_counts.items()},
                         {"tgscc": 8105, "hsk3": 10969})
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(dict(connection.execute("SELECT source, COUNT(*) FROM standard_entry GROUP BY source")),
                             {"tgscc": 8105, "hsk3": 10969})
            for word, source, level in (("一", "tgscc", 1), ("乂", "tgscc", 2),
                                        ("亍", "tgscc", 3), ("爱", "hsk3", 1), ("阿拉伯语", "hsk3", 7)):
                with self.subTest(word=word, source=source):
                    self.assertEqual(connection.execute(
                        "SELECT level FROM standard_entry WHERE word = ? AND source = ?", (word, source),
                    ).fetchone(), (level,))
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            for entry in datasets:
                source = entry["id"]
                self.assertEqual(metadata[f"standard:{source}:sha256"], entry["sha256"])
                self.assertEqual(metadata[f"standard:{source}:count"], str(result.dataset_counts[source].imported))

    def test_rejects_malformed_standard_rows(self):
        for parser, line in ((builder.parse_standard_char, "双字\t1\n"),
                             (builder.parse_standard_char, "一\t4\n"),
                             (builder.parse_hsk_word, "爱\t8\n"),
                             (builder.parse_hsk_word, "爱\t1\textra\n"),
                             (builder.parse_hsk_word, "\t1\n")):
            with self.subTest(line=line), self.assertRaisesRegex(ValueError, "Malformed standard entry"):
                parser(line)
        self.write_hsk_fixture("# comment\n爱\t1\n坏词\tnot-a-level\n".encode())
        with self.assertRaisesRegex(ValueError, "Malformed standard entry"):
            builder.build_reference_database(self.root, self.db)
        self.assertFalse(self.db.exists())

    def test_rejects_checksum_mismatch(self):
        self.write_hsk_fixture("爱\t1\n".encode(), sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "Checksum mismatch for hsk.tsv"):
            builder.build_reference_database(self.root, self.db)
        self.assertFalse(self.db.exists())

    def test_returns_standards_without_attesting_unknown_or_invalid_commonness(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            self.seed_word_commonness(connection)
            connection.execute("INSERT INTO word_commonness VALUES ('一', NULL, NULL, 0)")
            connection.execute("INSERT INTO word_commonness VALUES ('亍', 0, NULL, 0)")
            connection.execute("CREATE TABLE standard_entry (word TEXT, source TEXT, level INTEGER)")
            connection.executemany("INSERT INTO standard_entry VALUES (?, ?, ?)", (
                ("爱", "tgscc", 1), ("爱", "hsk3", 1), ("一", "tgscc", 1), ("亍", "tgscc", 3),
                ("一", "hsk3", "invalid"),
                ("爱心", "tgscc", 2), ("爱心", "hsk3", 3),
            ))
        with patch.object(review, "reference_db_path", return_value=self.db):
            for word, general, hsk, available in (("爱", 1, 1, True), ("一", 1, None, True),
                                                  ("亍", 3, None, False), ("爱心", None, 3, True),
                                                  ("未收录", None, None, True)):
                with self.subTest(word=word):
                    result = review._query_commonness_reference(word)
                    self.assertEqual(result["standards"], {"generalLevel": general, "hskLevel": hsk, "available": True})
                    self.assertEqual(result["available"], available)
                    self.assertFalse(result["attested"])
                    self.assertEqual(result["dictionaryPresenceCount"], 0)
            self.assertEqual(review._query_commonness_reference("")["standards"], UNAVAILABLE)

    def test_missing_standard_table_or_source_is_unavailable(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            self.seed_word_commonness(connection)
        with patch.object(review, "reference_db_path", return_value=self.db):
            self.assertEqual(review._query_commonness_reference("爱")["standards"], UNAVAILABLE)
            with closing(sqlite3.connect(self.db)) as connection, connection:
                connection.execute("CREATE TABLE standard_entry (word TEXT, source TEXT, level INTEGER)")
                connection.execute("INSERT INTO standard_entry VALUES ('爱', 'hsk3', 1)")
            self.assertEqual(review._query_commonness_reference("爱")["standards"], UNAVAILABLE)

    def test_partial_bcc_copy_refuses_to_replace_original(self):
        self.write_hsk_fixture("爱\t1\n".encode())
        with closing(sqlite3.connect(self.db)) as connection, connection:
            create_bcc_schema(connection)
            connection.execute("INSERT INTO bcc_dataset VALUES ('x', '多领域', 'char', 'date', 'sha', 'zip', 1, 1, 1, 6)")
            connection.execute("INSERT INTO bcc_frequency VALUES ('x', '爱', 6, 1000000, 1)")
        original_db = self.db.read_bytes()
        preserve_bcc = builder.preserve_bcc
        for table in ("bcc_frequency", "bcc_dataset"):
            def copy_then_lose_row(source, connection):
                preserve_bcc(source, connection)
                connection.execute(f"DELETE FROM {table}")

            with self.subTest(table=table), patch.object(builder, "preserve_bcc", side_effect=copy_then_lose_row):
                with self.assertRaisesRegex(RuntimeError, f"BCC preservation count mismatch for {table}"):
                    builder.build_reference_database(self.root, self.db)
                self.assertEqual(self.db.read_bytes(), original_db)
                self.assertFalse(self.db.with_name(f".{self.db.name}.tmp").exists())
