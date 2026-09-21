#!/usr/bin/env python3
"""Download official static BCC ZIPs and atomically extend a local reference DB."""
from __future__ import annotations

import argparse
from contextlib import closing
import csv
import hashlib
import io
import json
from pathlib import Path
import resource
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from keytao_bot.utils.bcc_reference import create_schema

BASE_URL = 'https://bcc.blcu.edu.cn/api/datasets'
CHANNEL_PREFIXES = {'多领域': 'multi_domain_total', '新闻': 'news_total',
                    '文学': 'literature', '口语': 'dialogue',
                    '古代汉语': 'classical_chinese', '近代汉语': 'modern_chinese'}
CHUNK = 65536
MAX_ZIP_BYTES = 64 * 1024 * 1024
MAX_TEXT_BYTES = 128 * 1024 * 1024
DISK_RESERVE_BYTES = 256 * 1024 * 1024


class IngestError(RuntimeError):
    """A failed import leaves the destination database unchanged."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise IngestError('BCC dataset download redirected; refusing another endpoint')


def download_file(url, destination, timeout):
    if url != BASE_URL and url not in {
        f'{BASE_URL}/{prefix}_{kind}_freq.txt/download'
        for prefix in CHANNEL_PREFIXES.values() for kind in ('word', 'char')
    }:
        raise IngestError('Only official BCC static dataset GETs are allowed')
    start = time.monotonic()
    size = 0
    opener = urllib.request.build_opener(NoRedirect)
    with opener.open(url, timeout=timeout) as response, destination.open('wb') as output:
        while chunk := response.read(CHUNK):
            size += len(chunk)
            if time.monotonic() - start > timeout or size > MAX_ZIP_BYTES:
                raise IngestError('BCC download exceeded the per-file time/size limit')
            output.write(chunk)


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def select_datasets(inventory):
    selected = []
    for channel, prefix in CHANNEL_PREFIXES.items():
        for kind in ('word', 'char'):
            filename = f'{prefix}_{kind}_freq.txt'
            matches = [row for row in inventory if row.get('filename') == filename]
            if len(matches) != 1:
                raise IngestError(f'{filename}: missing or duplicate inventory entry')
            row = matches[0]
            if (row.get('channel'), row.get('token_type'), row.get('format'),
                row.get('freq_type'), row.get('token_column'), row.get('count_column')) != (
                    channel, kind, 'txt', 'freq', 'token', 'count'):
                raise IngestError(f'{filename}: unexpected dataset metadata')
            if not row.get('updated_at') or not 0 < row.get('size_bytes', 0) <= MAX_TEXT_BYTES:
                raise IngestError(f'{filename}: invalid update timestamp or size')
            selected.append(row)
    return selected


def load_archive(connection, dataset, archive_path):
    name = dataset['filename']
    if archive_path.stat().st_size > MAX_ZIP_BYTES:
        raise IngestError(f'{name}: cached ZIP exceeds size limit')
    with zipfile.ZipFile(archive_path) as archive:
        files = archive.infolist()
        if len(files) != 1 or files[0].filename != name or files[0].flag_bits & 1:
            raise IngestError(f'{name}: ZIP must contain exactly its unencrypted TXT member')
        member = files[0]
        if member.file_size != dataset['size_bytes'] or member.file_size > MAX_TEXT_BYTES:
            raise IngestError(f'{name}: uncompressed size differs from inventory')
        # Reading to EOF verifies CRC without extracting or holding the file in memory.
        digest = hashlib.sha256()
        with archive.open(member) as source:
            while chunk := source.read(CHUNK):
                digest.update(chunk)
        content_hash = digest.hexdigest()
        expected = dataset.get('sha256')
        if expected is not None and expected != content_hash:
            raise IngestError(f'{name}: published content hash mismatch')
        connection.execute('DELETE FROM bcc_frequency WHERE dataset=?', (name,))
        connection.execute('DELETE FROM bcc_dataset WHERE filename=?', (name,))
        total = rows = 0
        batch = []
        with archive.open(member) as raw, io.TextIOWrapper(raw, encoding='utf-8-sig', newline='') as source:
            reader = csv.reader(source, strict=True)
            if next(reader, None) != ['token', 'count']:
                raise IngestError(f'{name}: expected CSV header token,count')
            for line, row in enumerate(reader, 2):
                if len(row) != 2 or not row[0] or not row[1].isascii() or not row[1].isdigit():
                    raise IngestError(f'{name}:{line}: invalid token/count row')
                count = int(row[1])
                if count <= 0:
                    raise IngestError(f'{name}:{line}: nonpositive count')
                rows += 1
                total += count
                batch.append((name, row[0], count, 1.0))
                if len(batch) == 1000:
                    connection.executemany('INSERT INTO bcc_frequency (dataset, token, raw_count, per_million) VALUES (?, ?, ?, ?)', batch)
                    batch.clear()
            if batch:
                connection.executemany('INSERT INTO bcc_frequency (dataset, token, raw_count, per_million) VALUES (?, ?, ?, ?)', batch)
        if not rows:
            raise IngestError(f'{name}: empty dataset')
        connection.execute('UPDATE bcc_frequency SET per_million=raw_count * 1000000.0 / ? WHERE dataset=?', (total, name))
        # Competition ranks preserve ties and do not depend on upstream sort order.
        connection.execute('CREATE TEMP TABLE bcc_ranks (token TEXT PRIMARY KEY, position INTEGER) WITHOUT ROWID')
        connection.execute('INSERT INTO bcc_ranks SELECT token, RANK() OVER (ORDER BY raw_count DESC) FROM bcc_frequency WHERE dataset=?', (name,))
        connection.execute('UPDATE bcc_frequency SET frequency_rank=(SELECT position FROM bcc_ranks WHERE token=bcc_frequency.token) WHERE dataset=?', (name,))
        connection.execute('DROP TABLE bcc_ranks')
        connection.execute('INSERT INTO bcc_dataset VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (name, dataset['channel'], dataset['token_type'], dataset['updated_at'], content_hash,
             sha256(archive_path), member.file_size, archive_path.stat().st_size, rows, total))


def check_disk_headroom(db, cache, datasets):
    """Reserve staging DB/journal/sort space and maximum ZIP growth per device."""
    requests = [(db.parent, 2 * db.stat().st_size + 12 * sum(row['size_bytes'] for row in datasets) + MAX_ZIP_BYTES),
                (cache, len(datasets) * MAX_ZIP_BYTES)]
    devices = {}
    for path, required in requests:
        while not path.exists():
            path = path.parent
        device = path.stat().st_dev
        previous = devices.get(device, (path, DISK_RESERVE_BYTES))
        devices[device] = (path, previous[1] + required)
    for path, required in devices.values():
        free = shutil.disk_usage(path).free
        if free < required:
            raise IngestError(f'Insufficient disk headroom at {path}: need {required} bytes free, have {free}')


def ingest(db, cache, inventory, *, timeout=120, cached_only=False):
    db, cache = Path(db).resolve(), Path(cache).resolve()
    if not db.is_file():
        raise IngestError('Reference DB missing; run scripts/build_pinyin_reference.py first')
    datasets = select_datasets(inventory)
    before = db.stat().st_size
    original_stat = db.stat()
    if Path(str(db) + '-wal').exists():
        raise IngestError('Reference DB has a WAL; stop writers and checkpoint before ingest')
    with closing(sqlite3.connect(f'{db.as_uri()}?mode=ro', uri=True)) as source:
        source.row_factory = sqlite3.Row
        if not source.execute("SELECT 1 FROM sqlite_master WHERE name='word_commonness'").fetchone():
            raise IngestError('word_commonness missing; run scripts/build_pinyin_reference.py first')
        has_bcc = source.execute("SELECT 1 FROM sqlite_master WHERE name='bcc_dataset'").fetchone()
        existing = {row['filename']: dict(row) for row in source.execute('SELECT * FROM bcc_dataset')} if has_bcc else {}
        has_rank = 'frequency_rank' in {row[1] for row in source.execute('PRAGMA table_info(bcc_frequency)')}
        needs_rank = {row[0] for row in source.execute('SELECT DISTINCT dataset FROM bcc_frequency WHERE frequency_rank IS NULL')} if has_rank else set(existing)
    changed = []
    for dataset in datasets:
        name = dataset['filename']
        old = existing.get(name)
        cached = cache / (name + '.zip')
        if (old is None or name in needs_rank or old['updated_at'] != dataset['updated_at']
                or old['size_bytes'] != dataset['size_bytes']
                or (dataset.get('sha256') and old['content_sha256'] != dataset['sha256'])
                or (cached.exists() and sha256(cached) != old['archive_sha256'])):
            changed.append(dataset)
    peak_disk_bytes = before + sum(path.stat().st_size for path in cache.glob('*.zip'))
    downloaded = []
    if changed:
        check_disk_headroom(db, cache, changed)
        cache.mkdir(parents=True, exist_ok=True)
        # The old DB remains readable until every selected dataset is complete.
        with tempfile.TemporaryDirectory(prefix='.bcc-ingest-', dir=db.parent) as work:
            temporary = Path(work) / 'reference.db'
            stop = threading.Event()

            def measure_disk():
                nonlocal peak_disk_bytes
                while not stop.is_set():
                    paths = [db, *Path(work).rglob('*'), *cache.glob('*.zip')]
                    total = 0
                    for path in paths:
                        try:
                            if path.is_file():
                                total += path.stat().st_size
                        except FileNotFoundError:
                            pass
                    peak_disk_bytes = max(peak_disk_bytes, total)
                    stop.wait(0.02)

            monitor = threading.Thread(target=measure_disk, daemon=True)
            monitor.start()
            try:
                _build_staging(db, temporary, cache, changed, timeout, cached_only, downloaded)
                current = db.stat()
                if Path(str(db) + '-wal').exists() or (current.st_ino, current.st_size, current.st_mtime_ns) != (
                        original_stat.st_ino, original_stat.st_size, original_stat.st_mtime_ns):
                    raise IngestError('Reference DB changed during ingest; refusing to replace it')
                temporary.chmod(original_stat.st_mode & 0o777)
                # Requires a single writer: stat validation is not a compare-and-swap.
                temporary.replace(db)
            finally:
                stop.set()
                monitor.join()
    with closing(sqlite3.connect(f'{db.as_uri()}?mode=ro', uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        loaded = [dict(row) for row in connection.execute('SELECT * FROM bcc_dataset ORDER BY filename')]
    return {'downloaded': downloaded, 'updated': [row['filename'] for row in changed], 'datasets': loaded,
            'rows': sum(row['row_count'] for row in loaded),
            'beforeBytes': before, 'afterBytes': db.stat().st_size,
            'growthBytes': db.stat().st_size - before,
            'peakDiskBytesSampled': peak_disk_bytes,
            'diskSampleIntervalSeconds': 0.02,
            'cacheBytes': sum(path.stat().st_size for path in cache.glob('*.zip'))}


def _build_staging(db, temporary, cache, changed, timeout, cached_only, downloaded):
    with closing(sqlite3.connect(temporary)) as connection:
        connection.execute('PRAGMA cache_size=-8192')
        connection.execute('PRAGMA temp_store=FILE')
        with closing(sqlite3.connect(f'{db.as_uri()}?mode=ro', uri=True)) as source:
            source.backup(connection, pages=256)
        connection.execute('PRAGMA journal_mode=DELETE')
        with connection:
            create_schema(connection)
            for dataset in changed:
                name = dataset['filename']
                archive_path = temporary.parent / (name + '.zip')
                try:
                    if cached_only:
                        archive_path = cache / (name + '.zip')
                    else:
                        download_file(f'{BASE_URL}/{name}/download', archive_path, timeout)
                        downloaded.append(name)
                    load_archive(connection, dataset, archive_path)
                    if not cached_only:
                        archive_path.replace(cache / (name + '.zip'))
                except Exception as error:
                    raise IngestError(f'{name}: import failed: {error}; original DB unchanged') from error
        if connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise IngestError('BCC staging DB failed SQLite integrity check')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=ROOT / 'data/pinyin_reference.db')
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'data/bcc')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--inventory', type=Path, help='Previously downloaded official inventory JSON')
    parser.add_argument('--cached-only', action='store_true', help='Read existing ZIPs only; requires --inventory')
    args = parser.parse_args()
    if not 0 < args.timeout <= 600:
        parser.error('--timeout must be between 0 and 600 seconds')
    if args.cached_only and not args.inventory:
        parser.error('--cached-only requires --inventory')
    start = time.monotonic()
    try:
        if args.inventory:
            inventory = json.loads(args.inventory.read_text(encoding='utf-8'))['datasets']
        else:
            if shutil.disk_usage(tempfile.gettempdir()).free < MAX_ZIP_BYTES + DISK_RESERVE_BYTES:
                raise IngestError('Insufficient disk headroom for BCC inventory')
            with tempfile.TemporaryDirectory(prefix='bcc-inventory-') as directory:
                path = Path(directory) / 'inventory.json'
                download_file(BASE_URL, path, args.timeout)
                inventory = json.loads(path.read_text(encoding='utf-8'))['datasets']
        result = ingest(args.db, args.cache_dir, inventory, timeout=args.timeout, cached_only=args.cached_only)
        result['inventory'] = inventory
        result['elapsedSeconds'] = round(time.monotonic() - start, 3)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result['peakRssBytes'] = rss if sys.platform == 'darwin' else rss * 1024
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(f'BCC ingest failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
