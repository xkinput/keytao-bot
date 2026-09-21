"""Offline ZIP/SQLite boundary tests for the BCC importer."""
import hashlib
from contextlib import closing
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import ingest_bcc as ingest


class BccIngestTests(unittest.TestCase):
    def test_all_twelve_datasets_are_ingested_with_historical_labels(self):
        prefixes = {**ingest.CHANNEL_PREFIXES, '古代汉语': 'classical_chinese',
                    '近代汉语': 'modern_chinese'}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ingest, 'CHANNEL_PREFIXES', prefixes):
                db, cache, datasets = self.cached_corpus(Path(directory))
            result = ingest.ingest(db, cache, datasets, cached_only=True)
            self.assertEqual(len(result['datasets']), 12)
            self.assertEqual(result['rows'], 72)
            self.assertEqual({(row['channel'], row['token_type']) for row in result['datasets']
                              if row['filename'].startswith(('classical_', 'modern_'))},
                             {('古代汉语', 'char'), ('古代汉语', 'word'),
                              ('近代汉语', 'char'), ('近代汉语', 'word')})

    def cached_corpus(self, root):
        db, cache = root / 'reference.db', root / 'cache'
        cache.mkdir()
        with closing(sqlite3.connect(db)) as connection, connection:
            connection.execute('CREATE TABLE word_commonness (word TEXT PRIMARY KEY, corpus_frequency INTEGER, part_of_speech TEXT, dictionary_presence_count INTEGER)')
        datasets = []
        for channel, prefix in ingest.CHANNEL_PREFIXES.items():
            for kind in ('char', 'word'):
                name = f'{prefix}_{kind}_freq.txt'
                # Different denominators and table populations; deliberately unsorted.
                rows = [('乙', 100), ('甲', 900)] if kind == 'char' else [
                    *[(f'对照{index}', 10) for index in range(9)], ('甲词', 20)]
                payload = ('token,count\n' + ''.join(f'{token},{count}\n' for token, count in rows)).encode()
                with zipfile.ZipFile(cache / (name + '.zip'), 'w') as archive:
                    archive.writestr(name, payload)
                datasets.append(dict(filename=name, channel=channel, token_type=kind,
                    updated_at='fixture', format='txt', freq_type='freq',
                    token_column='token', count_column='count', size_bytes=len(payload)))
        return db, cache, datasets

    def test_cross_token_types_compare_relative_ranks_not_per_million(self):
        from keytao_bot.utils import keytao_review as review
        with tempfile.TemporaryDirectory() as directory:
            db, cache, datasets = self.cached_corpus(Path(directory))
            with patch.object(ingest, 'download_file', side_effect=AssertionError('Offline only')):
                result = ingest.ingest(db, cache, datasets, cached_only=True)
            self.assertEqual(result['downloaded'], [])
            with patch.object(review, 'reference_db_path', return_value=db):
                char, word = [review._query_commonness_reference(token) for token in ('甲', '甲词')]
            self.assertGreater(char['bcc']['perMillion'], word['bcc']['perMillion'] * 2)
            self.assertEqual(char['bcc']['rankFraction'], 1 / 2)
            self.assertEqual(word['bcc']['rankFraction'], 1 / 10)
            comparison = review._compare_reference_commonness('甲', '甲词', char, word)
            self.assertEqual(comparison['verdict'], 'behind_more_common')
            self.assertEqual(comparison['decisionReason'], 'bcc_relative_rank_ratio')
            inverse = review._compare_reference_commonness('甲词', '甲', word, char)
            self.assertEqual(inverse['verdict'], 'front_more_common')
            self.assertEqual(inverse['summary'], comparison['summary'])
            self.assertIn('表内排名', comparison['summary'])
            # Old optional snapshots without ranks must not compare unlike units.
            char['bcc']['rankFraction'] = None
            self.assertEqual(review._compare_reference_commonness('甲', '甲词', char, word)['verdict'], 'not_enough_evidence')

    def test_low_disk_refuses_before_download_or_staging_and_keeps_db(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            db, cache, datasets = self.cached_corpus(Path(directory))
            before = db.read_bytes()
            with (patch.object(ingest.shutil, 'disk_usage', return_value=SimpleNamespace(free=1)),
                  patch.object(ingest, 'download_file') as download,
                  patch.object(ingest, '_build_staging') as staging):
                with self.assertRaisesRegex(ingest.IngestError, 'Insufficient disk headroom'):
                    ingest.ingest(db, cache, datasets, cached_only=True)
            download.assert_not_called()
            staging.assert_not_called()
            self.assertEqual(db.read_bytes(), before)

    def test_wal_guard_preserves_database_before_any_dataset_work(self):
        with tempfile.TemporaryDirectory() as directory:
            db, cache, datasets = self.cached_corpus(Path(directory))
            before = db.read_bytes()
            Path(str(db) + '-wal').touch()
            with (patch.object(ingest, 'download_file') as download,
                  patch.object(ingest, '_build_staging') as staging):
                with self.assertRaisesRegex(ingest.IngestError, 'has a WAL'):
                    ingest.ingest(db, cache, datasets, cached_only=True)
            download.assert_not_called()
            staging.assert_not_called()
            self.assertEqual(db.read_bytes(), before)

    def test_container_command_starts_bot_after_any_ingest_exit(self):
        root = Path(__file__).resolve().parent
        command = json.loads(next(line[4:] for line in (root / 'Dockerfile').read_text().splitlines() if line.startswith('CMD ')))
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            executable = work / 'uv'
            executable.write_text('#!/bin/sh\n'
                'printf "%s\\n" "$*" >> "$BCC_TEST_LOG"\n'
                'case "$*" in\n'
                '  *scripts/ingest_bcc.py*) exit "$BCC_TEST_EXIT" ;;\n'
                'esac\n')
            executable.chmod(0o755)
            for exit_code in (0, 1, 2, 137):
                with self.subTest(exit_code=exit_code):
                    log = work / f'{exit_code}.log'
                    env = {**os.environ, 'PATH': str(work) + os.pathsep + os.environ.get('PATH', ''),
                           'BCC_TEST_LOG': str(log), 'BCC_TEST_EXIT': str(exit_code)}
                    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=5)
                    self.assertEqual(result.returncode, 0)
                    self.assertEqual(log.read_text().splitlines(), [
                        'run python scripts/build_pinyin_reference.py',
                        'run python scripts/ingest_bcc.py --timeout 15', 'run python bot.py'])
                    self.assertEqual('WARNING: BCC ingest failed' in result.stderr, exit_code != 0)

    def test_ingest_cli_reports_outage_or_corrupt_cache_without_publishing(self):
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as directory:
            db, cache, datasets = self.cached_corpus(Path(directory))
            inventory = Path(directory) / 'inventory.json'
            inventory.write_text(json.dumps({'datasets': datasets}))
            before = db.read_bytes()
            with (patch.object(sys, 'argv', ['ingest_bcc.py', '--db', str(db)]),
                  patch.object(ingest, 'download_file', side_effect=TimeoutError('fixture outage')),
                  redirect_stderr(io.StringIO()) as error):
                self.assertEqual(ingest.main(), 1)
            self.assertIn('fixture outage', error.getvalue())
            (cache / (datasets[-1]['filename'] + '.zip')).write_bytes(b'broken')
            with (patch.object(sys, 'argv', ['ingest_bcc.py', '--db', str(db), '--cache-dir', str(cache),
                    '--inventory', str(inventory), '--cached-only']),
                  patch.object(ingest, 'download_file', side_effect=AssertionError('Offline only')),
                  redirect_stderr(io.StringIO()) as error):
                self.assertEqual(ingest.main(), 1)
            self.assertIn('not a zip', error.getvalue().lower())
            self.assertEqual(db.read_bytes(), before)

    def test_corrupt_previous_database_is_rebuilt_without_bcc(self):
        from keytao_bot.utils.pinyin_reference_build import build_reference_database
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b'example 12 n\n'
            (root / 'jieba.txt').write_bytes(payload)
            (root / 'manifest.json').write_text(json.dumps({'formatVersion': 1, 'datasets': [
                {'id': 'jieba', 'file': 'jieba.txt', 'format': 'jieba', 'import': True,
                 'sha256': hashlib.sha256(payload).hexdigest()}]}))
            db = root / 'reference.db'
            db.write_bytes(b'corrupt old database')
            self.assertTrue(build_reference_database(root, db).rebuilt)
            with closing(sqlite3.connect(db)) as connection:
                self.assertEqual(connection.execute('PRAGMA quick_check').fetchone()[0], 'ok')
                self.assertEqual(connection.execute('SELECT corpus_frequency FROM word_commonness').fetchone()[0], 12)

    def test_zip_and_csv_integrity_checks_reject_bad_data(self):
        from keytao_bot.utils.bcc_reference import create_schema
        name = 'multi_domain_total_word_freq.txt'
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / 'source.zip'
            for payload, member in ((b'word,count\nword,6\n', name),
                                    (b'token,count\nword,6\nword,6\n', name),
                                    (b'token,count\nword,0\n', name),
                                    (b'token,count\nword,6\n', '../' + name)):
                with self.subTest(payload=payload, member=member):
                    with zipfile.ZipFile(archive_path, 'w') as archive:
                        archive.writestr(member, payload)
                    with closing(sqlite3.connect(':memory:')) as connection:
                        create_schema(connection)
                        with self.assertRaises((ingest.IngestError, sqlite3.IntegrityError)):
                            ingest.load_archive(connection, {'filename': name, 'size_bytes': len(payload)}, archive_path)
            payload = b'token,count\nword,6\n'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr(name, payload)
            data = archive_path.read_bytes().replace(b'word,6', b'word,7')
            archive_path.write_bytes(data)
            with closing(sqlite3.connect(':memory:')) as connection:
                create_schema(connection)
                with self.assertRaises(zipfile.BadZipFile):
                    ingest.load_archive(connection, {'filename': name, 'size_bytes': len(payload)}, archive_path)

    def test_import_is_streamed_normalized_idempotent_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / 'reference.db'
            with closing(sqlite3.connect(db)) as connection, connection:
                connection.execute('CREATE TABLE word_commonness (word TEXT PRIMARY KEY)')
            payloads = {}
            datasets = []
            for channel, prefix in ingest.CHANNEL_PREFIXES.items():
                for kind in ('word', 'char'):
                    name = f'{prefix}_{kind}_freq.txt'
                    text = 'token,count\n甲,3\n乙,1\n'.encode()
                    stream = io.BytesIO()
                    with zipfile.ZipFile(stream, 'w') as archive:
                        archive.writestr(name, text)
                    payloads[name] = stream.getvalue()
                    datasets.append(dict(filename=name, channel=channel, token_type=kind,
                        updated_at='2026-05-22', format='txt', freq_type='freq',
                        token_column='token', count_column='count', size_bytes=len(text)))

            def get(url, destination, timeout):
                name = url.split('/')[-2]
                destination.write_bytes(payloads[name])

            with patch.object(ingest, 'download_file', side_effect=get) as download:
                first = ingest.ingest(db, root / 'cache', datasets, timeout=30)
                self.assertEqual(first['rows'], 24)
                self.assertEqual(download.call_count, 12)
                with closing(sqlite3.connect(db)) as connection:
                    self.assertEqual(connection.execute(
                        "SELECT raw_count, per_million FROM bcc_frequency WHERE token='甲' LIMIT 1"
                    ).fetchone(), (3, 750000.0))
                digest = hashlib.sha256(db.read_bytes()).hexdigest()
                self.assertEqual(ingest.ingest(db, root / 'cache', datasets, timeout=30)['downloaded'], [])
                self.assertEqual(download.call_count, 12)
                self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), digest)
                # A corrupted cached ZIP must be fetched again, never trusted.
                (root / 'cache' / (datasets[0]['filename'] + '.zip')).write_bytes(b'broken')
                ingest.ingest(db, root / 'cache', datasets, timeout=30)
                self.assertEqual(download.call_count, 13)
                digest = hashlib.sha256(db.read_bytes()).hexdigest()
                # A bad later dataset cannot publish an earlier successful update.
                changed = [dict(row, updated_at='2026-06-01') for row in datasets]
                last_good_zip = payloads[datasets[-1]['filename']]
                payloads[datasets[-1]['filename']] = b'not a zip'
                with self.assertRaisesRegex(ingest.IngestError, datasets[-1]['filename']):
                    ingest.ingest(db, root / 'cache', changed, timeout=30)
                self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), digest)
                # A concurrent base-reference replacement must not be overwritten.
                payloads[datasets[-1]['filename']] = last_good_zip

                def concurrent_write(url, destination, timeout):
                    get(url, destination, timeout)
                    db.touch()

                with patch.object(ingest, 'download_file', side_effect=concurrent_write):
                    with self.assertRaisesRegex(ingest.IngestError, 'changed during ingest'):
                        ingest.ingest(db, root / 'cache', changed, timeout=30)
                self.assertEqual(hashlib.sha256(db.read_bytes()).hexdigest(), digest)

    def test_rebuilding_base_reference_preserves_bcc(self):
        from keytao_bot.utils.pinyin_reference_build import build_reference_database
        from keytao_bot.utils.bcc_reference import create_schema
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b'example 12 n\n'
            (root / 'jieba.txt').write_bytes(payload)
            (root / 'manifest.json').write_text(json.dumps({'formatVersion': 1, 'datasets': [
                {'id': 'jieba', 'file': 'jieba.txt', 'format': 'jieba', 'import': True,
                 'sha256': hashlib.sha256(payload).hexdigest()}]}))
            db = root / 'reference.db'
            build_reference_database(root, db)
            with closing(sqlite3.connect(db)) as connection, connection:
                create_schema(connection)
                connection.execute("INSERT INTO bcc_dataset VALUES ('x', '多领域', 'word', 'fixture', 'sha', 'zip', 1, 1, 1, 6)")
                connection.execute("INSERT INTO bcc_frequency (dataset, token, raw_count, per_million) VALUES ('x', '情报所', 6, 1000000)")
            (root / 'excluded_words.txt').write_text('example\n')
            self.assertTrue(build_reference_database(root, db).rebuilt)
            with closing(sqlite3.connect(db)) as connection:
                self.assertEqual(connection.execute('SELECT token, raw_count FROM bcc_frequency').fetchall(), [('情报所', 6)])
                self.assertEqual(connection.execute('SELECT content_sha256 FROM bcc_dataset').fetchone()[0], 'sha')


if __name__ == '__main__':
    unittest.main()
