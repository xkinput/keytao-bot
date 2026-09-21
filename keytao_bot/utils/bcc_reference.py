"""BCC schema and local-only evidence; importing this module never performs I/O."""
from contextlib import suppress
import logging
import sqlite3

CHANNELS = ("多领域", "新闻", "文学", "口语")
SCHEMA = (
    """CREATE TABLE IF NOT EXISTS bcc_dataset (
        filename TEXT PRIMARY KEY, channel TEXT NOT NULL,
        token_type TEXT NOT NULL CHECK (token_type IN ('char', 'word')),
        updated_at TEXT NOT NULL, content_sha256 TEXT NOT NULL,
        archive_sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
        archive_bytes INTEGER NOT NULL, row_count INTEGER NOT NULL,
        total_count INTEGER NOT NULL CHECK (total_count > 0),
        UNIQUE(channel, token_type)
    ) WITHOUT ROWID""",
    """CREATE TABLE IF NOT EXISTS bcc_frequency (
        dataset TEXT NOT NULL REFERENCES bcc_dataset(filename),
        token TEXT NOT NULL, raw_count INTEGER NOT NULL CHECK (raw_count > 0),
        per_million REAL NOT NULL CHECK (per_million > 0),
        frequency_rank INTEGER CHECK (frequency_rank > 0),
        PRIMARY KEY(dataset, token)
    ) WITHOUT ROWID""",
)


def create_schema(connection):
    for statement in SCHEMA:
        connection.execute(statement)
    if 'frequency_rank' not in {row[1] for row in connection.execute('PRAGMA table_info(bcc_frequency)')}:
        connection.execute('ALTER TABLE bcc_frequency ADD COLUMN frequency_rank INTEGER CHECK (frequency_rank > 0)')


def preserve_bcc(source_path, connection):
    """Carry an optional BCC snapshot through an offline base-reference rebuild."""
    try:
        if not source_path.is_file():
            return
        connection.execute('SAVEPOINT bcc_copy')
        connection.execute('ATTACH DATABASE ? AS bcc_previous',
                           (source_path.resolve().as_uri() + '?mode=ro',))
        if connection.execute(
            "SELECT 1 FROM bcc_previous.sqlite_master WHERE name='bcc_dataset'"
        ).fetchone():
            columns = ('filename, channel, token_type, updated_at, content_sha256, '
                       'archive_sha256, size_bytes, archive_bytes, row_count, total_count')
            connection.execute(f'INSERT INTO bcc_dataset ({columns}) SELECT {columns} FROM bcc_previous.bcc_dataset')
            columns = 'dataset, token, raw_count, per_million'
            if 'frequency_rank' in {row[1] for row in connection.execute('PRAGMA bcc_previous.table_info(bcc_frequency)')}:
                columns += ', frequency_rank'
            connection.execute(f'INSERT INTO bcc_frequency ({columns}) SELECT {columns} FROM bcc_previous.bcc_frequency')
        connection.execute('RELEASE bcc_copy')
    except sqlite3.Error as error:
        # Optional corpus preservation must not prevent rebuilding the base DB.
        with suppress(sqlite3.Error):
            connection.execute('ROLLBACK TO bcc_copy')
            connection.execute('RELEASE bcc_copy')
        logging.getLogger(__name__).warning('BCC preservation skipped: %s', error)
        return


def lookup_bcc(connection, word):
    kind = "char" if len(word) == 1 else "word"
    result = {"available": False, "attested": False, "tokenType": kind,
              "perMillion": None, "rankFraction": None, "channels": []}
    if not connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='bcc_dataset'"
    ).fetchone():
        return result
    has_rank = 'frequency_rank' in {row[1] for row in connection.execute('PRAGMA table_info(bcc_frequency)')}
    rows = connection.execute(f"""
        SELECT d.filename, d.channel, d.updated_at, d.content_sha256,
               d.total_count, f.raw_count, f.per_million, d.row_count,
               {'f.frequency_rank' if has_rank else 'NULL'}
        FROM bcc_dataset d LEFT JOIN bcc_frequency f
          ON f.dataset=d.filename AND f.token=?
        WHERE d.token_type=? AND d.channel IN ('多领域', '新闻', '文学', '口语')
    """, (word, kind)).fetchall()
    # Never compare a half-installed channel set against a complete one.
    if {row[1] for row in rows} != set(CHANNELS):
        return result
    channels = {row[1]: {
        "dataset": row[0], "channel": row[1], "updatedAt": row[2],
        "sha256": row[3], "totalCount": row[4], "count": row[5],
        "perMillion": row[6], "rowCount": row[7], "rank": row[8],
        "rankFraction": row[8] / row[7] if row[8] is not None else None,
    } for row in rows}
    observed = [row for row in channels.values() if row['count'] is not None]
    ranked = [row['rankFraction'] for row in observed if row['rankFraction'] is not None]
    result.update(available=True, attested=any(row[5] is not None for row in rows),
                  channels=[channels[channel] for channel in CHANNELS],
                  perMillion=max((row['perMillion'] for row in observed), default=None),
                  rankFraction=min(ranked) if observed and len(ranked) == len(observed) else None)
    return result


def comparison_signal(front, behind):
    """Return comparable positive evidence, or defer to the legacy signals."""
    if not all(item.get('available') and item.get('attested') for item in (front, behind)):
        return None
    cross_type = front['tokenType'] != behind['tokenType']
    key = 'rankFraction' if cross_type else 'perMillion'
    values = [item.get(key) for item in (front, behind)]
    if any(value is None or value <= 0 for value in values):
        return None
    # A smaller relative rank is stronger; invert it to retain the ratio rule.
    signal = [1 / value for value in values] if cross_type else values
    reason = 'bcc_relative_rank_ratio' if cross_type else 'bcc_frequency_ratio'
    if signal[0] == signal[1]:
        balanced = [next((row.get(key) for row in item['channels'] if row['channel'] == '多领域'), None)
                    for item in (front, behind)]
        if all(value is not None and value > 0 for value in balanced):
            signal = [1 / value for value in balanced] if cross_type else balanced
            if balanced[0] != balanced[1]:
                reason += '_balanced_tiebreak'
    return (*signal, reason)


def format_frequency(count, rate):
    if count is None:
        return "未收录"
    # Do not round a small but real observation down to zero.
    value = f"{rate:.2f}".rstrip("0").rstrip(".") if rate >= 0.01 else f"{rate:.2g}"
    return f"{count:,}（每百万 {value}）"


def format_bcc(bcc, other=None):
    if not bcc.get("available"):
        return ""
    other_channels = {row["channel"]: row for row in (other or {}).get("channels", [])}
    parts = []
    for row in bcc["channels"]:
        right = other_channels.get(row["channel"], {})
        if row['count'] is None and right.get('count') is None:
            continue
        value = format_frequency(row["count"], row["perMillion"])
        if other is not None:
            right = other_channels.get(row["channel"], {})
            value += " vs " + format_frequency(right.get("count"), right.get("perMillion"))
        parts.append(f"{row['channel']} {value}")
    return "BCC " + "；".join(parts) if parts else "BCC 四个现代频道均未收录（发布表最低计数为 6，未收录不等于零次）"
