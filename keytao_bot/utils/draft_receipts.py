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
    revised_words = list(dict.fromkeys(
        str(item.get("word") or "")
        for field in ("writtenItems", "updatedItems") for item in data.get(field) or []
        if isinstance(item, dict) and item.get("id") is not None and item.get("word")
        and (item.get("action") in {"Change", "Delete"} or field == "updatedItems")
    ))
    if revised_words:
        states = {}
        for word in revised_words:
            status = next((item.get("_revisionStatus", "Draft")
                           for field in ("writtenItems", "updatedItems") for item in data.get(field) or []
                           if isinstance(item, dict) and item.get("word") == word), "Draft")
            if status != "Draft":
                states.setdefault(status, []).append(f"「{word}」")
        labels = {"Draft": "草稿（未发布）", "Submitted": "待审核（未发布）",
                  "Approved": "已入库（尚未发布）", "Published": "已发布"}
        if states:
            lines.append("发布状态：" + "；".join(
                "、".join(words) + "的本次修改现为" + labels.get(status, "状态待核验")
                for status, words in states.items()
            ) + "。")
    if data.get("receiptItemsUnavailable"):
        lines.append("写入结果已返回，但本轮变更明细未能完整核验；请查看原批次。")
    written_words = {change.word for change in changes}
    failures = {
        str(item.get("word") or ""): str(item.get("reason") or item.get("message") or "服务端未返回具体原因")
        for field in ("failed", "skipped") for item in data.get(field) or [] if isinstance(item, dict)
    }
    snapshot = data.get("draft_snapshot")
    draft_items = snapshot.get("items") if isinstance(snapshot, dict) else data.get("draftItems")
    if (not data.get("batchId") or
            isinstance(snapshot, dict) and snapshot.get("batchId") not in (None, data["batchId"])):
        draft_items = []
    for word in requested:
        if word not in written_words:
            existing = [item for item in draft_items or []
                        if isinstance(item, dict) and item.get("word") == word]
            codes = list(dict.fromkeys(
                item["code"] for item in existing
                if item.get("action") in {"Create", "Change"} and item.get("code")
            ))
            moved_codes = list(dict.fromkeys(
                item["code"] for item in existing
                if item.get("action") == "Create" and item.get("code") and any(
                    old.get("action") == "Delete" and old.get("type") == item.get("type")
                    and old.get("code") and old["code"] != item["code"] for old in existing
                )
            ))
            reason = failures.get(word) or (
                "已在本批草稿中调整为 " + "、".join(moved_codes) if moved_codes else
                "已在本批草稿中（→ " + "、".join(codes) + "）" if codes else
                "已在草稿中，本轮未重复写入" if data.get("noWrite") or word in data.get("noWriteWords", []) else
                "未取得该词的本轮写入回执，原因尚未确认"
            )
            lines.append(f"未新增变更：{word}（{reason}）")
    return lines


def merge_receipt_deltas(receipts: list[dict[str, Any]]) -> dict:
    """Merge actual same-turn evidence without counting a PR again on submit."""
    merged = {"writtenItems": [], "updatedItems": [], "requestedWords": [], "failed": [], "skipped": [],
              "noWriteWords": []}
    positions = {}
    batch_statuses = {}
    batch_ids = {receipt["batchId"] for receipt in receipts if receipt.get("batchId")}
    draft_items_by_batch = {}
    for receipt in receipts:
        snapshot = receipt.get("draft_snapshot")
        draft_items = snapshot.get("items") if isinstance(snapshot, dict) else receipt.get("draftItems")
        if (receipt.get("batchId") and isinstance(draft_items, list)
                and (not isinstance(snapshot, dict) or snapshot.get("batchId") in (None, receipt["batchId"]))):
            draft_items_by_batch[receipt["batchId"]] = draft_items
        if receipt.get("autoApproved") is True:
            batch_statuses[receipt.get("batchId")] = "Approved"
        elif receipt.get("tool") == "keytao_submit_batch" or receipt.get("operationKind") == "draft_submit":
            batch_statuses[receipt.get("batchId")] = "Approved" if receipt.get("autoApproved") else "Submitted"
        elif receipt.get("status") in {"Draft", "Submitted", "Approved", "Published"}:
            batch_statuses[receipt.get("batchId")] = receipt["status"]
    for receipt in receipts:
        words = list(receipt.get("noWriteWords") or [])
        if receipt.get("noWrite"):
            words.extend(receipt.get("requestedWords") or [
                item.get("word") for item in receipt.get("requestedItems") or [] if isinstance(item, dict)
            ])
        for word in words:
            if word and word not in merged["noWriteWords"]:
                merged["noWriteWords"].append(word)
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
                row = {**row, "_revisionStatus": batch_statuses.get(receipt.get("batchId"), row.get("_revisionStatus", "Draft"))}
                identity = (receipt.get("batchId"), row["id"])
                if identity in positions:
                    previous_field, index = positions[identity]
                    previous = merged[previous_field][index]
                    merged[previous_field][index] = {**row, **({"previousWeight": previous["previousWeight"]} if "previousWeight" in previous else {})}
                else:
                    positions[identity] = (field, len(merged[field]))
                    merged[field].append(dict(row))
    if len(batch_ids) == 1:
        merged["batchId"] = next(iter(batch_ids))
        if merged["batchId"] in draft_items_by_batch:
            merged["draftItems"] = draft_items_by_batch[merged["batchId"]]
    return merged
