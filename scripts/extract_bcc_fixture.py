#!/usr/bin/env python3
"""Extract the S63 exact-token slice; retain full-file totals and provenance."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

TOKENS = ('情报所', '敲不死', '环境法', '户晨风', '五角星', '沃集鲜', '的', '龘', '一一化',
          '量子薄荷鸽跃器', '星云奶酪折叠喵')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with closing(sqlite3.connect(f'{args.db.resolve().as_uri()}?mode=ro', uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        datasets = [dict(row) for row in connection.execute('SELECT * FROM bcc_dataset ORDER BY filename')]
        rows = [dict(row) for row in connection.execute(
            'SELECT * FROM bcc_frequency WHERE token IN (' + ','.join('?' for _ in TOKENS) + ') ORDER BY dataset,token', TOKENS)]
        jieba = [dict(row) for row in connection.execute(
            'SELECT * FROM word_commonness WHERE word IN (' + ','.join('?' for _ in TOKENS) + ') ORDER BY word', TOKENS)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'source': 'https://bcc.blcu.edu.cn/api/datasets',
        'tokens': TOKENS, 'datasets': datasets, 'rows': rows, 'jieba': jieba},
        ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
