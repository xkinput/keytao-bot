"""Seed scenario-owned pronunciation cache rows in a validated local database."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .runtime import RigInfrastructureError
from .safety import validate_next_database_url


_MULTI_ADD_ZDIC_CACHE_ROWS: tuple[dict[str, Any], ...] = (
    {"kind": "char", "entry": "王", "status": "found", "pinyins": ["wáng"]},
    {"kind": "char", "entry": "中", "status": "found", "pinyins": ["zhōng"]},
    {"kind": "char", "entry": "微", "status": "found", "pinyins": ["wēi"]},
    {"kind": "char", "entry": "服", "status": "found", "pinyins": ["fú"]},
    {"kind": "char", "entry": "务", "status": "found", "pinyins": ["wù"]},
    {"kind": "entry", "entry": "王中王", "status": "absent", "pinyins": []},
    {"kind": "entry", "entry": "微服务", "status": "absent", "pinyins": []},
)

ZDIC_FIXTURES_BY_SCENARIO: dict[str, dict[str, Any]] = {
    # 吃席 has no authoritative word page, but both characters resolve in
    # production. Seeding that exact shape is what lets the reviewed-add path
    # fall back to own-character readings instead of failing closed.
    "S2": {
        "probe_words": ("吃席",),
        "rows": (
            {"kind": "char", "entry": "吃", "status": "found", "pinyins": ["chī"]},
            {"kind": "char", "entry": "席", "status": "found", "pinyins": ["xí"]},
            {"kind": "entry", "entry": "吃席", "status": "absent", "pinyins": []},
        ),
    },
    "S9": {
        "probe_words": ("射覆",),
        "rows": (
            {"kind": "char", "entry": "射", "status": "found", "pinyins": ["shè"]},
            {"kind": "char", "entry": "覆", "status": "found", "pinyins": ["fù"]},
            {"kind": "entry", "entry": "射覆", "status": "absent", "pinyins": []},
            {"kind": "char", "entry": "慑", "status": "found", "pinyins": ["shè"]},
            {"kind": "char", "entry": "服", "status": "found", "pinyins": ["fú"]},
            {"kind": "entry", "entry": "慑服", "status": "absent", "pinyins": []},
        ),
    },
    "S10": {
        "probe_words": ("王中王", "微服务"),
        "rows": _MULTI_ADD_ZDIC_CACHE_ROWS,
    },
    "S14": {
        "probe_words": ("亮面",),
        "rows": (
            {"kind": "char", "entry": "亮", "status": "found", "pinyins": ["liàng"]},
            {"kind": "char", "entry": "面", "status": "found", "pinyins": ["miàn"]},
            {"kind": "entry", "entry": "亮面", "status": "absent", "pinyins": []},
        ),
    },
}

S9_ZDIC_CACHE_ROWS = ZDIC_FIXTURES_BY_SCENARIO["S9"]["rows"]


def _normalized_scenario_ids(scenario_ids: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item).strip().upper() for item in scenario_ids))


def _copy_row(row: dict[str, Any]) -> dict[str, Any]:
    return {**row, "pinyins": list(row["pinyins"])}


def _validated_row(raw_row: dict[str, Any], *, scenario_id: str) -> dict[str, Any]:
    row = _copy_row(raw_row)
    kind = row.get("kind")
    entry = str(row.get("entry") or "").strip()
    status = row.get("status")
    pinyins = row.get("pinyins")
    if kind not in {"char", "entry"} or not entry:
        raise RigInfrastructureError(
            f"Invalid ZDIC fixture row for {scenario_id}: {raw_row}"
        )
    if status not in {"found", "absent"} or not isinstance(pinyins, list):
        raise RigInfrastructureError(
            f"Invalid ZDIC fixture status for {scenario_id}: {raw_row}"
        )
    if any(not isinstance(item, str) or not item.strip() for item in pinyins):
        raise RigInfrastructureError(
            f"Invalid ZDIC fixture pinyins for {scenario_id}: {raw_row}"
        )
    if (status == "found") != bool(pinyins):
        raise RigInfrastructureError(
            f"Inconsistent ZDIC fixture row for {scenario_id}: {raw_row}"
        )
    return {"kind": kind, "entry": entry, "status": status, "pinyins": pinyins}


def zdic_cache_rows_for_scenarios(
    scenario_ids: Iterable[str],
) -> tuple[dict[str, Any], ...]:
    """Return validated cache rows, deduplicated by the database primary key."""

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for scenario_id in _normalized_scenario_ids(scenario_ids):
        fixture = ZDIC_FIXTURES_BY_SCENARIO.get(scenario_id)
        if fixture is None:
            continue
        for raw_row in fixture["rows"]:
            row = _validated_row(raw_row, scenario_id=scenario_id)
            key = (row["kind"], row["entry"])
            existing = rows_by_key.get(key)
            if existing is not None and existing != row:
                raise RigInfrastructureError(
                    "Conflicting ZDIC fixture declarations for "
                    f"{row['kind']}:{row['entry']}: {existing} != {row}"
                )
            rows_by_key.setdefault(key, row)
    return tuple(_copy_row(row) for row in rows_by_key.values())


def _sql_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _build_seed_sql(rows: tuple[dict[str, Any], ...]) -> str:
    values = ",\n".join(
        "    ("
        + ", ".join(
            (
                _sql_text(row["kind"]),
                _sql_text(row["entry"]),
                _sql_text(row["status"]),
                _sql_text(json.dumps(row["pinyins"], ensure_ascii=False)) + "::jsonb",
                "CURRENT_TIMESTAMP",
            )
        )
        + ")"
        for row in rows
    )
    return f"""\
BEGIN;

INSERT INTO "zdic_pinyin_cache" AS cache
    ("kind", "entry", "status", "pinyins", "fetchedAt")
VALUES
{values}
ON CONFLICT ("kind", "entry") DO UPDATE SET
    "status" = EXCLUDED."status",
    "pinyins" = EXCLUDED."pinyins",
    "fetchedAt" = EXCLUDED."fetchedAt";

COMMIT;
"""


def seed_zdic_cache(
    database_url: str,
    *,
    next_dir: Path,
    scenario_ids: Iterable[str],
) -> dict[str, Any]:
    """Upsert selected scenario rows after the rig's existing DB safety check."""

    database = validate_next_database_url(database_url)
    selected_scenarios = _normalized_scenario_ids(scenario_ids)
    rows = zdic_cache_rows_for_scenarios(selected_scenarios)
    if not rows:
        raise RigInfrastructureError(
            f"No ZDIC cache fixture is declared for scenarios {selected_scenarios}"
        )

    prisma = next_dir / "node_modules" / ".bin" / "prisma"
    prisma_config = next_dir / "prisma.config.ts"
    if not prisma.is_file() or not prisma_config.is_file():
        raise RigInfrastructureError(
            "keytao-next dependencies or Prisma config are unavailable for the ZDIC cache seed"
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
            input=_build_seed_sql(rows),
            cwd=next_dir,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RigInfrastructureError(
            "ZDIC cache seed timed out against the validated local database"
        ) from error
    if completed.returncode != 0:
        output = (completed.stdout + completed.stderr).replace(
            database_url, "[REDACTED_DATABASE_URL]"
        )
        raise RigInfrastructureError(
            "ZDIC cache seed failed against the validated local database:\n"
            + output[-2000:]
        )

    return {
        "database": database,
        "table": "zdic_pinyin_cache",
        "scenarioIds": list(selected_scenarios),
        "rows": [_copy_row(row) for row in rows],
    }


def seed_s9_zdic_cache(database_url: str, *, next_dir: Path) -> dict[str, Any]:
    """Compatibility wrapper for the original single-scenario seeder."""

    return seed_zdic_cache(database_url, next_dir=next_dir, scenario_ids=("S9",))
