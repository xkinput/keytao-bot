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
ZDIC_FIXTURES_BY_SCENARIO["S15"] = {
    "probe_words": ("射覆", "亮面"),
    "rows": (
        *ZDIC_FIXTURES_BY_SCENARIO["S9"]["rows"],
        *ZDIC_FIXTURES_BY_SCENARIO["S14"]["rows"],
    ),
}
ZDIC_FIXTURES_BY_SCENARIO["S16"] = {
    "probe_words": ("载流", "载流子", "座落在"),
    "rows": (
        {"kind": "char", "entry": "载", "status": "found", "pinyins": ["zǎi", "zài"]},
        {"kind": "char", "entry": "流", "status": "found", "pinyins": ["liú"]},
        {"kind": "char", "entry": "子", "status": "found", "pinyins": ["zǐ"]},
        {"kind": "char", "entry": "座", "status": "found", "pinyins": ["zuò"]},
        {"kind": "char", "entry": "落", "status": "found", "pinyins": ["luò"]},
        {"kind": "char", "entry": "在", "status": "found", "pinyins": ["zài"]},
        {"kind": "entry", "entry": "载流", "status": "absent", "pinyins": []},
        {"kind": "entry", "entry": "载流子", "status": "absent", "pinyins": []},
        {"kind": "entry", "entry": "座落在", "status": "absent", "pinyins": []},
    ),
}
ZDIC_FIXTURES_BY_SCENARIO["S17"] = {
    "probe_words": ("产季", "龘季"),
    "rows": (
        {"kind": "char", "entry": "产", "status": "found", "pinyins": ["chǎn"]},
        {"kind": "char", "entry": "季", "status": "found", "pinyins": ["jì"]},
        {"kind": "char", "entry": "龘", "status": "found", "pinyins": ["dá"]},
        {"kind": "entry", "entry": "产季", "status": "absent", "pinyins": []},
        {"kind": "entry", "entry": "龘季", "status": "absent", "pinyins": []},
    ),
}
ZDIC_FIXTURES_BY_SCENARIO["S18"] = {
    "probe_words": ("还车", "换车"),
    "rows": (
        {"kind": "char", "entry": "还", "status": "found", "pinyins": ["huán", "hái"]},
        {"kind": "char", "entry": "车", "status": "found", "pinyins": ["chē"]},
        {"kind": "char", "entry": "换", "status": "found", "pinyins": ["huàn"]},
        {"kind": "entry", "entry": "还车", "status": "found", "pinyins": ["huán", "chē"]},
        {"kind": "entry", "entry": "换车", "status": "absent", "pinyins": []},
    ),
}
ZDIC_FIXTURES_BY_SCENARIO["S19"] = {
    "probe_words": (
        "显眼包", "嘴替", "松弛感", "电子榨菜", "情绪价值", "班味",
        "泼天富贵", "精神状态", "职场搭子", "天选打工人", "沙县小吃",
    ),
    "rows": (
        {"kind": "char", "entry": "显", "status": "found", "pinyins": ["xiǎn"]},
        {"kind": "char", "entry": "眼", "status": "found", "pinyins": ["yǎn"]},
        {"kind": "char", "entry": "包", "status": "found", "pinyins": ["bāo"]},
        {"kind": "char", "entry": "嘴", "status": "found", "pinyins": ["zuǐ"]},
        {"kind": "char", "entry": "替", "status": "found", "pinyins": ["tì"]},
        {"kind": "char", "entry": "松", "status": "found", "pinyins": ["sōng"]},
        {"kind": "char", "entry": "弛", "status": "found", "pinyins": ["chí"]},
        {"kind": "char", "entry": "感", "status": "found", "pinyins": ["gǎn"]},
        {"kind": "char", "entry": "电", "status": "found", "pinyins": ["diàn"]},
        {"kind": "char", "entry": "子", "status": "found", "pinyins": ["zǐ"]},
        {"kind": "char", "entry": "榨", "status": "found", "pinyins": ["zhà"]},
        {"kind": "char", "entry": "菜", "status": "found", "pinyins": ["cài"]},
        {"kind": "char", "entry": "情", "status": "found", "pinyins": ["qíng"]},
        {"kind": "char", "entry": "绪", "status": "found", "pinyins": ["xù"]},
        {"kind": "char", "entry": "价", "status": "found", "pinyins": ["jià"]},
        {"kind": "char", "entry": "值", "status": "found", "pinyins": ["zhí"]},
        {"kind": "char", "entry": "班", "status": "found", "pinyins": ["bān"]},
        {"kind": "char", "entry": "味", "status": "found", "pinyins": ["wèi"]},
        {"kind": "char", "entry": "泼", "status": "found", "pinyins": ["pō"]},
        {"kind": "char", "entry": "天", "status": "found", "pinyins": ["tiān"]},
        {"kind": "char", "entry": "富", "status": "found", "pinyins": ["fù"]},
        {"kind": "char", "entry": "贵", "status": "found", "pinyins": ["guì"]},
        {"kind": "char", "entry": "精", "status": "found", "pinyins": ["jīng"]},
        {"kind": "char", "entry": "神", "status": "found", "pinyins": ["shén"]},
        {"kind": "char", "entry": "状", "status": "found", "pinyins": ["zhuàng"]},
        {"kind": "char", "entry": "态", "status": "found", "pinyins": ["tài"]},
        {"kind": "char", "entry": "职", "status": "found", "pinyins": ["zhí"]},
        {"kind": "char", "entry": "场", "status": "found", "pinyins": ["chǎng"]},
        {"kind": "char", "entry": "搭", "status": "found", "pinyins": ["dā"]},
        {"kind": "char", "entry": "选", "status": "found", "pinyins": ["xuǎn"]},
        {"kind": "char", "entry": "打", "status": "found", "pinyins": ["dǎ"]},
        {"kind": "char", "entry": "工", "status": "found", "pinyins": ["gōng"]},
        {"kind": "char", "entry": "人", "status": "found", "pinyins": ["rén"]},
        {"kind": "char", "entry": "沙", "status": "found", "pinyins": ["shā"]},
        {"kind": "char", "entry": "县", "status": "found", "pinyins": ["xiàn"]},
        {"kind": "char", "entry": "小", "status": "found", "pinyins": ["xiǎo"]},
        {"kind": "char", "entry": "吃", "status": "found", "pinyins": ["chī"]},
        {"kind": "entry", "entry": "显眼包", "status": "found", "pinyins": ["xiǎn", "yǎn", "bāo"]},
        {"kind": "entry", "entry": "嘴替", "status": "found", "pinyins": ["zuǐ", "tì"]},
        {"kind": "entry", "entry": "松弛感", "status": "found", "pinyins": ["sōng", "chí", "gǎn"]},
        {"kind": "entry", "entry": "电子榨菜", "status": "found", "pinyins": ["diàn", "zǐ", "zhà", "cài"]},
        {"kind": "entry", "entry": "情绪价值", "status": "found", "pinyins": ["qíng", "xù", "jià", "zhí"]},
        {"kind": "entry", "entry": "班味", "status": "found", "pinyins": ["bān", "wèi"]},
        {"kind": "entry", "entry": "泼天富贵", "status": "found", "pinyins": ["pō", "tiān", "fù", "guì"]},
        {"kind": "entry", "entry": "精神状态", "status": "found", "pinyins": ["jīng", "shén", "zhuàng", "tài"]},
        {"kind": "entry", "entry": "职场搭子", "status": "found", "pinyins": ["zhí", "chǎng", "dā", "zǐ"]},
        {"kind": "entry", "entry": "天选打工人", "status": "found", "pinyins": ["tiān", "xuǎn", "dǎ", "gōng", "rén"]},
        {"kind": "entry", "entry": "沙县小吃", "status": "found", "pinyins": ["shā", "xiàn", "xiǎo", "chī"]},
    ),
}
_S20_BATCH_WORDS = tuple(ZDIC_FIXTURES_BY_SCENARIO["S19"]["probe_words"][:3])
_S20_BATCH_CHARACTERS = set("".join(_S20_BATCH_WORDS))
ZDIC_FIXTURES_BY_SCENARIO["S20"] = {
    "probe_words": _S20_BATCH_WORDS,
    "rows": tuple(
        row
        for row in ZDIC_FIXTURES_BY_SCENARIO["S19"]["rows"]
        if (
            row["kind"] == "char"
            and row["entry"] in _S20_BATCH_CHARACTERS
        )
        or (
            row["kind"] == "entry"
            and row["entry"] in _S20_BATCH_WORDS
        )
    ),
}
_S21_BATCH_WORDS = (
    "显眼包", "嘴替",
)
_S21_CHARACTER_PINYINS = {
    "显": ["xiǎn"], "眼": ["yǎn"], "包": ["bāo"],
    "嘴": ["zuǐ"], "替": ["tì"],
}
_S21_ENTRY_PINYINS = {
    "显眼包": ["xiǎn", "yǎn", "bāo"],
    "嘴替": ["zuǐ", "tì"],
}
ZDIC_FIXTURES_BY_SCENARIO["S21"] = {
    "probe_words": _S21_BATCH_WORDS,
    "rows": (
        *(
            {
                "kind": "char",
                "entry": character,
                "status": "found",
                "pinyins": pinyins,
            }
            for character, pinyins in _S21_CHARACTER_PINYINS.items()
        ),
        *(
            {
                "kind": "entry",
                "entry": word,
                "status": "found",
                "pinyins": pinyins,
            }
            for word, pinyins in _S21_ENTRY_PINYINS.items()
        ),
    ),
}
def _s19_subset_fixture(word_count: int) -> dict[str, object]:
    words = tuple(ZDIC_FIXTURES_BY_SCENARIO["S19"]["probe_words"][:word_count])
    characters = set("".join(words))
    return {
        "probe_words": words,
        "rows": tuple(
            row
            for row in ZDIC_FIXTURES_BY_SCENARIO["S19"]["rows"]
            if (
                row["kind"] == "char"
                and row["entry"] in characters
            )
            or (
                row["kind"] == "entry"
                and row["entry"] in words
            )
        ),
    }


ZDIC_FIXTURES_BY_SCENARIO["S22"] = _s19_subset_fixture(2)
ZDIC_FIXTURES_BY_SCENARIO["S23"] = _s19_subset_fixture(9)
ZDIC_FIXTURES_BY_SCENARIO["S24"] = {
    "probe_words": tuple(ZDIC_FIXTURES_BY_SCENARIO["S18"]["probe_words"]),
    "rows": tuple(ZDIC_FIXTURES_BY_SCENARIO["S18"]["rows"]),
}
ZDIC_FIXTURES_BY_SCENARIO["S25"] = {
    "probe_words": ("炒冷饭",),
    "rows": (
        {"kind": "char", "entry": "炒", "status": "found", "pinyins": ["chǎo"]},
        {"kind": "char", "entry": "冷", "status": "found", "pinyins": ["lěng"]},
        {"kind": "char", "entry": "饭", "status": "found", "pinyins": ["fàn"]},
        {
            "kind": "entry",
            "entry": "炒冷饭",
            "status": "found",
            "pinyins": ["chǎo", "lěng", "fàn"],
        },
    ),
}
ZDIC_FIXTURES_BY_SCENARIO["S27"] = {
    "probe_words": ("来都来了",),
    "rows": (
        {"kind": "char", "entry": "来", "status": "found", "pinyins": ["lái"]},
        {"kind": "char", "entry": "都", "status": "found", "pinyins": ["dōu", "dū"]},
        {"kind": "char", "entry": "了", "status": "found", "pinyins": ["le", "liǎo"]},
        {
            "kind": "entry",
            "entry": "来都来了",
            "status": "found",
            "pinyins": ["lái", "dōu", "lái", "le"],
        },
    ),
}
ZDIC_FIXTURES_BY_SCENARIO["S28"] = {
    "probe_words": tuple(ZDIC_FIXTURES_BY_SCENARIO["S18"]["probe_words"]),
    "rows": tuple(ZDIC_FIXTURES_BY_SCENARIO["S18"]["rows"]),
}
ZDIC_FIXTURES_BY_SCENARIO["S29"] = {
    "probe_words": ("火锅", "电脑"),
    "rows": (
        {"kind": "char", "entry": "火", "status": "found", "pinyins": ["huǒ"]},
        {"kind": "char", "entry": "锅", "status": "found", "pinyins": ["guō"]},
        {"kind": "char", "entry": "电", "status": "found", "pinyins": ["diàn"]},
        {"kind": "char", "entry": "脑", "status": "found", "pinyins": ["nǎo"]},
        {"kind": "entry", "entry": "火锅", "status": "found", "pinyins": ["huǒ", "guō"]},
        {"kind": "entry", "entry": "电脑", "status": "found", "pinyins": ["diàn", "nǎo"]},
    ),
}
ZDIC_FIXTURES_BY_SCENARIO["S30"] = {
    "probe_words": ("吃席",),
    "rows": (
        {"kind": "char", "entry": "吃", "status": "found", "pinyins": ["chī"]},
        {"kind": "char", "entry": "席", "status": "found", "pinyins": ["xí"]},
        {"kind": "entry", "entry": "吃席", "status": "absent", "pinyins": []},
    ),
}
ZDIC_FIXTURES_BY_SCENARIO["S31"] = {
    "probe_words": ("幂等", "米等"),
    "rows": (
        {"kind": "char", "entry": "幂", "status": "found", "pinyins": ["mì"]},
        {"kind": "char", "entry": "米", "status": "found", "pinyins": ["mǐ"]},
        {"kind": "char", "entry": "等", "status": "found", "pinyins": ["děng"]},
        {"kind": "entry", "entry": "幂等", "status": "found", "pinyins": ["mì", "děng"]},
        {"kind": "entry", "entry": "米等", "status": "found", "pinyins": ["mǐ", "děng"]},
    ),
}
ZDIC_FIXTURES_BY_SCENARIO["S32"] = {
    "probe_words": ("米等", "幂等", "迷瞪"),
    "rows": (
        {"kind": "char", "entry": "米", "status": "found", "pinyins": ["mǐ"]},
        {"kind": "char", "entry": "幂", "status": "found", "pinyins": ["mì"]},
        {"kind": "char", "entry": "等", "status": "found", "pinyins": ["děng"]},
        {"kind": "char", "entry": "迷", "status": "found", "pinyins": ["mí"]},
        {"kind": "char", "entry": "瞪", "status": "found", "pinyins": ["dèng"]},
        {"kind": "entry", "entry": "米等", "status": "found", "pinyins": ["mǐ", "děng"]},
        {"kind": "entry", "entry": "幂等", "status": "found", "pinyins": ["mì", "děng"]},
        {"kind": "entry", "entry": "迷瞪", "status": "found", "pinyins": ["mí", "dèng"]},
    ),
}
ZDIC_FIXTURES_BY_SCENARIO["S33"] = {
    "probe_words": ("洒漏", "撒漏", "洒溇", "缩手", "所售", "所受"),
    "rows": (
        {"kind": "char", "entry": "洒", "status": "found", "pinyins": ["sǎ"]},
        {"kind": "char", "entry": "撒", "status": "found", "pinyins": ["sǎ", "sā"]},
        {"kind": "char", "entry": "漏", "status": "found", "pinyins": ["lòu"]},
        {"kind": "char", "entry": "溇", "status": "found", "pinyins": ["lóu"]},
        {"kind": "entry", "entry": "洒漏", "status": "found", "pinyins": ["sǎ", "lòu"]},
        {"kind": "entry", "entry": "撒漏", "status": "found", "pinyins": ["sǎ", "lòu"]},
        {"kind": "entry", "entry": "洒溇", "status": "found", "pinyins": ["sǎ", "lóu"]},
        {"kind": "char", "entry": "缩", "status": "found", "pinyins": ["suō"]},
        {"kind": "char", "entry": "手", "status": "found", "pinyins": ["shǒu"]},
        {"kind": "char", "entry": "所", "status": "found", "pinyins": ["suǒ"]},
        {"kind": "char", "entry": "售", "status": "found", "pinyins": ["shòu"]},
        {"kind": "char", "entry": "受", "status": "found", "pinyins": ["shòu"]},
        {"kind": "entry", "entry": "缩手", "status": "found", "pinyins": ["suō", "shǒu"]},
        {"kind": "entry", "entry": "所售", "status": "found", "pinyins": ["suǒ", "shòu"]},
        {"kind": "entry", "entry": "所受", "status": "found", "pinyins": ["suǒ", "shòu"]},
    ),
}

ZDIC_FIXTURES_BY_SCENARIO["S34"] = {
    "probe_words": ("开团",),
    "rows": (
        {"kind": "char", "entry": "开", "status": "found", "pinyins": ["kāi"]},
        {"kind": "char", "entry": "团", "status": "found", "pinyins": ["tuán"]},
        {"kind": "entry", "entry": "开团", "status": "found", "pinyins": ["kāi", "tuán"]},
    ),
}

ZDIC_FIXTURES_BY_SCENARIO["S35"] = {
    "probe_words": ("发布会", "重病号", "计算机", "建三江", "无事忙"),
    "rows": (
        {"kind": "char", "entry": "发", "status": "found", "pinyins": ["fā"]},
        {"kind": "char", "entry": "布", "status": "found", "pinyins": ["bù"]},
        {"kind": "char", "entry": "会", "status": "found", "pinyins": ["huì"]},
        {"kind": "char", "entry": "重", "status": "found", "pinyins": ["zhòng", "chóng"]},
        {"kind": "char", "entry": "病", "status": "found", "pinyins": ["bìng"]},
        {"kind": "char", "entry": "号", "status": "found", "pinyins": ["hào", "háo"]},
        {"kind": "char", "entry": "计", "status": "found", "pinyins": ["jì"]},
        {"kind": "char", "entry": "算", "status": "found", "pinyins": ["suàn"]},
        {"kind": "char", "entry": "机", "status": "found", "pinyins": ["jī"]},
        {"kind": "char", "entry": "建", "status": "found", "pinyins": ["jiàn"]},
        {"kind": "char", "entry": "三", "status": "found", "pinyins": ["sān"]},
        {"kind": "char", "entry": "江", "status": "found", "pinyins": ["jiāng"]},
        {"kind": "char", "entry": "无", "status": "found", "pinyins": ["wú"]},
        {"kind": "char", "entry": "事", "status": "found", "pinyins": ["shì"]},
        {"kind": "char", "entry": "忙", "status": "found", "pinyins": ["máng"]},
        {"kind": "entry", "entry": "发布会", "status": "found", "pinyins": ["fā", "bù", "huì"]},
        {"kind": "entry", "entry": "重病号", "status": "found", "pinyins": ["zhòng", "bìng", "hào"]},
        {"kind": "entry", "entry": "计算机", "status": "found", "pinyins": ["jì", "suàn", "jī"]},
        {"kind": "entry", "entry": "建三江", "status": "found", "pinyins": ["jiàn", "sān", "jiāng"]},
        {"kind": "entry", "entry": "无事忙", "status": "found", "pinyins": ["wú", "shì", "máng"]},
    ),
}

ZDIC_FIXTURES_BY_SCENARIO["S37"] = {
    "probe_words": ("耙耙柑", "琵琶骨"),
    "rows": (
        {"kind": "char", "entry": "耙", "status": "found", "pinyins": ["pá", "bà"]},
        {"kind": "char", "entry": "柑", "status": "found", "pinyins": ["gān"]},
        {"kind": "char", "entry": "琵", "status": "found", "pinyins": ["pí"]},
        {"kind": "char", "entry": "琶", "status": "found", "pinyins": ["pá"]},
        {"kind": "char", "entry": "骨", "status": "found", "pinyins": ["gǔ", "gū"]},
        {"kind": "entry", "entry": "耙耙柑", "status": "found", "pinyins": ["pá", "pá", "gān"]},
        {"kind": "entry", "entry": "琵琶骨", "status": "found", "pinyins": ["pí", "pá", "gǔ"]},
    ),
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


def dictionary_fixture_words_for_scenario(scenario_id: str) -> tuple[str, ...]:
    """Return the declared dictionary word slots for one scenario."""

    normalized_scenario_id = str(scenario_id).strip().upper()
    fixture = ZDIC_FIXTURES_BY_SCENARIO.get(normalized_scenario_id)
    if fixture is None:
        raise RigInfrastructureError(
            f"No ZDIC cache fixture is declared for scenario {normalized_scenario_id}"
        )
    words = [str(word).strip() for word in fixture["probe_words"]]
    words.extend(
        row["entry"]
        for raw_row in fixture["rows"]
        if (row := _validated_row(raw_row, scenario_id=normalized_scenario_id))[
            "kind"
        ]
        == "entry"
    )
    return tuple(dict.fromkeys(word for word in words if word))


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
