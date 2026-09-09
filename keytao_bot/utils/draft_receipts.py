"""Project verified per-invocation PR deltas into complete word receipts."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WrittenChange:
    word: str
    text: str
    created_ids: tuple
    updated_ids: tuple = ()


def receipt_changes(data: dict, requested_words=()) -> list[WrittenChange]:
    """Cover each actual PR once; a Delete/Create pair is one displayed move."""
    rows = []
    seen = set()
    for field in ("writtenItems", "updatedItems"):
        for item in data.get(field) or []:
            if not isinstance(item, dict):
                continue
            identity = item.get("id")
            if identity is None or identity in seen:
                continue
            seen.add(identity)
            rows.append((item, field == "writtenItems"))
    by_word: dict[str, list] = {}
    for item, created in rows:
        by_word.setdefault(str(item.get("word") or "（词条内容缺失）"), []).append((item, created))
    changes = {}
    for word, entries in by_word.items():
        deletes = [row for row, created in entries if created and row.get("action") == "Delete"]
        creates = [row for row, created in entries if created and row.get("action") == "Create"]
        paired = set()
        parts = []
        if len(deletes) == len(creates) == 1 and deletes[0].get("type") == creates[0].get("type"):
            old, new = deletes[0], creates[0]
            if old.get("code") != new.get("code"):
                parts.append(f"{word} {old.get('code') or '（编码缺失）'}→{new.get('code') or '（编码缺失）'}")
                paired.update((old["id"], new["id"]))
        for item, created in entries:
            if item["id"] in paired:
                continue
            code = item.get("code") or "（编码缺失）"
            action = item.get("action")
            old_word = item.get("oldWord") or item.get("old_word")
            if "previousWeight" in item and item.get("previousWeight") != item.get("weight"):
                old_weight = item["previousWeight"] if item["previousWeight"] is not None else "未指定"
                weight = item.get("weight") if item.get("weight") is not None else "未指定"
                parts.append(f"{word} {code} 权重 {old_weight}→{weight}")
            elif action == "Delete":
                parts.append(f"删除 {word} @ {code}")
            elif action == "Change":
                if old_word and old_word != word:
                    parts.append(f"{old_word}→{word} @ {code}")
                else:
                    weight = item.get("weight") if item.get("weight") is not None else "未指定"
                    parts.append(f"{word} {code} 权重 →{weight}")
            elif action == "Create" and created:
                parts.append(f"{word} → {code}")
            else:
                parts.append(f"{word} @ {code}（草稿条目已更新）")
        changes[word] = WrittenChange(
            word, "、".join(parts),
            tuple(item["id"] for item, created in entries if created),
            tuple(item["id"] for item, created in entries if not created),
        )

    # Keep each requested word next to the occupants moved out of its code.
    requested = list(dict.fromkeys(requested_words or data.get("requestedWords") or []))
    ordered = []
    def visit(word):
        if word not in changes or word in ordered:
            return
        ordered.append(word)
        codes = {item.get("code") for item, _ in by_word[word] if item.get("action") == "Create"}
        for other, entries in by_word.items():
            if other not in requested and any(item.get("action") == "Delete" and item.get("code") in codes for item, _ in entries):
                visit(other)
    for word in [*requested, *by_word]:
        visit(word)
    return [changes[word] for word in ordered]


def receipt_change_lines(data: dict, requested_words=()) -> list[str]:
    if "writtenItems" not in data and "updatedItems" not in data:
        return []
    changes = receipt_changes(data, requested_words)
    requested = list(dict.fromkeys(requested_words or data.get("requestedWords") or []))
    summary = ""
    for change in changes:
        summary += (("、" if requested and change.word not in requested else "；") if summary else "") + change.text
    lines = ["已变更：" + summary] if changes else []
    if data.get("receiptItemsUnavailable"):
        lines.append("写入结果已返回，但本轮变更明细未能完整核验；请查看原批次。")
    written_words = {change.word for change in changes}
    failures = {
        str(item.get("word") or ""): str(item.get("reason") or item.get("message") or "服务端未返回具体原因")
        for field in ("failed", "skipped") for item in data.get(field) or [] if isinstance(item, dict)
    }
    for word in requested:
        if word not in written_words:
            reason = failures.get(word) or (
                "已在草稿中，本轮未重复写入" if data.get("noWrite") else
                "未取得该词的本轮写入回执，原因尚未确认"
            )
            lines.append(f"未新增变更：{word}（{reason}）")
    return lines


def merge_receipt_deltas(receipts: list[dict[str, Any]]) -> dict:
    """Merge actual same-turn evidence without counting a PR again on submit."""
    merged = {"writtenItems": [], "updatedItems": [], "requestedWords": [], "failed": [], "skipped": []}
    positions = {}
    for receipt in receipts:
        if receipt.get("noWrite"):
            words = receipt.get("requestedWords") or [
                item.get("word") for item in receipt.get("requestedItems") or [] if isinstance(item, dict)
            ]
            for word in words:
                if word:
                    merged["skipped"].append({"word": word, "reason": "已在草稿中，本轮未重复写入"})
        for field in ("requestedWords", "failed", "skipped"):
            for value in receipt.get(field) or []:
                if value not in merged[field]:
                    merged[field].append(value)
        if not receipt.get("requestedWords"):
            for item in receipt.get("requestedItems") or []:
                word = item.get("word") if isinstance(item, dict) else None
                if word and word not in merged["requestedWords"]:
                    merged["requestedWords"].append(word)
        merged["receiptItemsUnavailable"] = bool(merged.get("receiptItemsUnavailable") or receipt.get("receiptItemsUnavailable"))
        for field in ("writtenItems", "updatedItems"):
            for row in receipt.get(field) or []:
                if not isinstance(row, dict) or row.get("id") is None:
                    continue
                identity = (receipt.get("batchId"), row["id"])
                if identity in positions:
                    previous_field, index = positions[identity]
                    previous = merged[previous_field][index]
                    merged[previous_field][index] = {**row, **({"previousWeight": previous["previousWeight"]} if "previousWeight" in previous else {})}
                else:
                    positions[identity] = (field, len(merged[field]))
                    merged[field].append(dict(row))
    return merged
