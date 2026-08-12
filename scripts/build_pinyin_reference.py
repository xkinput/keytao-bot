#!/usr/bin/env python3
"""Build the local pronunciation reference database from vendored sources."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from keytao_bot.utils.pinyin_reference_build import build_reference_database  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=REPO_ROOT / "vendor" / "pinyin_reference",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=REPO_ROOT / "data" / "pinyin_reference.db",
    )
    args = parser.parse_args()
    result = build_reference_database(args.source_dir, args.db)
    print(json.dumps(result.as_json_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
