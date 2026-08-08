"""Seed the fixed S9 pronunciation cache rows in a validated local database."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from .runtime import RigInfrastructureError
from .safety import validate_next_database_url


S9_ZDIC_CACHE_ROWS: tuple[dict[str, Any], ...] = (
    {"kind": "char", "entry": "射", "status": "found", "pinyins": ["shè"]},
    {"kind": "char", "entry": "覆", "status": "found", "pinyins": ["fù"]},
    {"kind": "entry", "entry": "射覆", "status": "absent", "pinyins": []},
    {"kind": "char", "entry": "慑", "status": "found", "pinyins": ["shè"]},
    {"kind": "char", "entry": "服", "status": "found", "pinyins": ["fú"]},
    {"kind": "entry", "entry": "慑服", "status": "absent", "pinyins": []},
)

_S9_ZDIC_CACHE_SQL = """
BEGIN;

INSERT INTO "zdic_pinyin_cache" AS cache
    ("kind", "entry", "status", "pinyins", "fetchedAt")
VALUES
    ('char', '射', 'found', '["shè"]'::jsonb, CURRENT_TIMESTAMP),
    ('char', '覆', 'found', '["fù"]'::jsonb, CURRENT_TIMESTAMP),
    ('entry', '射覆', 'absent', '[]'::jsonb, CURRENT_TIMESTAMP),
    ('char', '慑', 'found', '["shè"]'::jsonb, CURRENT_TIMESTAMP),
    ('char', '服', 'found', '["fú"]'::jsonb, CURRENT_TIMESTAMP),
    ('entry', '慑服', 'absent', '[]'::jsonb, CURRENT_TIMESTAMP)
ON CONFLICT ("kind", "entry") DO UPDATE SET
    "status" = EXCLUDED."status",
    "pinyins" = EXCLUDED."pinyins",
    "fetchedAt" = EXCLUDED."fetchedAt";

COMMIT;
"""


def seed_s9_zdic_cache(database_url: str, *, next_dir: Path) -> dict[str, Any]:
    """Upsert only the fixed S9 cache rows after the rig's DB safety check."""

    database = validate_next_database_url(database_url)
    prisma = next_dir / "node_modules" / ".bin" / "prisma"
    prisma_config = next_dir / "prisma.config.ts"
    if not prisma.is_file() or not prisma_config.is_file():
        raise RigInfrastructureError(
            "keytao-next dependencies or Prisma config are unavailable for the S9 cache seed"
        )

    child_env = {
        key: value
        for key, value in os.environ.items()
        if key != "DATABASE_URL"
    }
    child_env.update(
        {
            "ALL_PROXY": "http://127.0.0.1:9",
            "CHECKPOINT_DISABLE": "1",
            "DATABASE_URL": database_url,
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "localhost,127.0.0.1,::1",
            "PRISMA_HIDE_UPDATE_MESSAGE": "1",
        }
    )
    try:
        completed = subprocess.run(
            [
                str(prisma),
                "db",
                "execute",
                "--config",
                str(prisma_config),
                "--stdin",
            ],
            input=_S9_ZDIC_CACHE_SQL,
            cwd=next_dir,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RigInfrastructureError(
            "S9 ZDIC cache seed timed out against the validated local database"
        ) from error
    if completed.returncode != 0:
        output = (completed.stdout + completed.stderr).replace(
            database_url, "[REDACTED_DATABASE_URL]"
        )
        raise RigInfrastructureError(
            "S9 ZDIC cache seed failed against the validated local database:\n"
            + output[-2000:]
        )

    return {
        "database": database,
        "table": "zdic_pinyin_cache",
        "rows": [dict(row) for row in S9_ZDIC_CACHE_ROWS],
    }
