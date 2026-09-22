"""BCC schema and local-only evidence; importing this module never performs I/O."""
from contextlib import suppress
import logging
import sqlite3

CHANNELS = ("多领域", "新闻", "文学", "口语")
HISTORICAL_CHANNELS = ("古代汉语", "近代汉语")
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
              "perMillion": None, "rankFraction": None, "channels": [],
              "historicalChannels": [
                  {"channel": channel, "available": False, "count": None,
                   "perMillion": None, "rankFraction": None}
                  for channel in HISTORICAL_CHANNELS]}
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
        WHERE d.token_type=? AND d.channel IN ('多领域', '新闻', '文学', '口语', '古代汉语', '近代汉语')
    """, (word, kind)).fetchall()
    channels = {row[1]: {
        "dataset": row[0], "channel": row[1], "available": True, "updatedAt": row[2],
        "sha256": row[3], "totalCount": row[4], "count": row[5],
        "perMillion": row[6], "rowCount": row[7], "rank": row[8],
        "rankFraction": row[8] / row[7] if row[8] is not None else None,
    } for row in rows}
    result['historicalChannels'] = [channels.get(row['channel'], row)
                                    for row in result['historicalChannels']]
    # Historical evidence never completes a missing modern channel set.
    if not set(CHANNELS).issubset(channels):
        return result
    observed = [channels[channel] for channel in CHANNELS if channels[channel]['count'] is not None]
    ranked = [row['rankFraction'] for row in observed if row['rankFraction'] is not None]
    result.update(available=True, attested=bool(observed),
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


def historical_attested(bcc):
    return any(row.get('count') is not None for row in bcc.get('historicalChannels', []))


def historical_comparison_signal(front, behind):
    """Use one historical channel only after confirmed absence from modern BCC."""
    if not all(item.get('available') and not item.get('attested') for item in (front, behind)):
        return None
    cross_type = front['tokenType'] != behind['tokenType']
    key = 'rankFraction' if cross_type else 'perMillion'
    # Prefer the later period when both are attested; never mix periods or units.
    for channel, label in (('近代汉语', 'early_modern'), ('古代汉语', 'classical')):
        rows = [next((row for row in item.get('historicalChannels', [])
                      if row['channel'] == channel), {}) for item in (front, behind)]
        if not all(row.get('count') is not None and row.get(key) is not None
                   and row[key] > 0 for row in rows):
            continue
        values = [1 / row[key] if cross_type else row[key] for row in rows]
        unit = 'relative_rank' if cross_type else 'frequency'
        return (*values, f'bcc_historical_{label}_{unit}_ratio')
    return None


def format_historical(bcc, *, other=None, observed_only=False):
    rows = {row['channel']: row for row in bcc.get('historicalChannels', [])}
    other_rows = {row['channel']: row for row in (other or {}).get('historicalChannels', [])}
    channels = [channel for channel in HISTORICAL_CHANNELS
                if not observed_only or rows.get(channel, {}).get('count') is not None
                or other_rows.get(channel, {}).get('count') is not None]
    return '；'.join(
        f"{channel} " + (format_frequency(rows[channel].get('count'), rows[channel].get('perMillion'))
                        if rows.get(channel, {}).get('available') else '数据未安装')
        for channel in channels) or ('历史频道均未收录'
            if all(rows.get(channel, {}).get('available') for channel in HISTORICAL_CHANNELS)
            else '暂无可用历史记录')


def format_frequency(count, rate, *, precision=2):
    if count is None:
        return "未收录"
    # Do not round a small but real observation down to zero.
    value = f"{rate:.{precision}f}".rstrip("0").rstrip(".") if rate >= 0.01 else f"{rate:.{precision}g}"
    return f"{count:,}（每百万 {value}）"


def format_bcc(bcc, other=None, *, precision=2):
    if not bcc.get("available"):
        return ""
    other_channels = {row["channel"]: row for row in (other or {}).get("channels", [])}
    parts = []
    for row in bcc["channels"]:
        right = other_channels.get(row["channel"], {})
        if row['count'] is None and right.get('count') is None:
            continue
        value = format_frequency(row["count"], row["perMillion"], precision=precision)
        if other is not None:
            right = other_channels.get(row["channel"], {})
            value += " vs " + format_frequency(right.get("count"), right.get("perMillion"), precision=precision)
        parts.append(f"{row['channel']} {value}")
    return "BCC " + "；".join(parts) if parts else "BCC 四个现代频道均未收录"
