"""Durable, actor-scoped undo of one completed draft operation.

The capability is a before/after server snapshot, never assistant prose.
Dictionary moves are draft Create/Delete rows and revert in one CAS deletion.
"""

import asyncio
import hashlib
import json
import re
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .draft_mutation_store import get_default_draft_mutation_claim_store
from . import http_client
from .observability import observe_tool_call

UNDO_WINDOW_SECONDS = 600
_UNDO_WORDS = {"取消", "撤销", "撤回", "回滚"}
_YES_WORDS = {"是", "是的", "确认", "确认撤销", "确定", "好", "好的"}
_NO_WORDS = {"否", "不是", "不撤销", "不撤回", "不用了"}
# The existing strict batch API rejects either delimiter anywhere in remark.
_SERVER_REVIEW_DELIMITERS = ("--- miao-review:start ---", "--- miao-review:end ---")
_MUTATION_TOOLS = {
    "keytao_create_phrase", "keytao_batch_add_to_draft", "keytao_shift_phrase_code",
    "keytao_update_draft_item_weight", "keytao_remove_draft_item",
    "keytao_batch_remove_draft_items", "keytao_recall_batch", "keytao_submit_batch",
}
_FIELDS = ("id", "word", "code", "type", "action", "oldWord", "weight", "remark", "needsManualReview")


@dataclass
class OperationCapture:
    scope: str
    platform: str
    actor: str
    operation_id: str
    previous_turn: str
    before: Optional[Dict] = None
    payload: Optional[Dict] = None


current_operation_capture: ContextVar[Optional[OperationCapture]] = ContextVar("completed_draft_operation", default=None)


def conversation_scope(key: Any) -> str:
    return json.dumps([key.platform, key.space_type, key.space_id, key.actor_id], separators=(",", ":"))


def start_operation_turn(key: Any) -> OperationCapture:
    scope = conversation_scope(key)
    turn_id = uuid.uuid4().hex
    previous = get_default_draft_mutation_claim_store().note_operation_turn(scope, turn_id)
    capture = OperationCapture(scope, key.platform, key.actor_id, turn_id, previous)
    current_operation_capture.set(capture)
    return capture


def _snapshot(data: Any, expected_batch: str = "") -> Optional[Dict]:
    if not isinstance(data, dict) or data.get("success") is not True:
        return None
    batch_id = str(data.get("batchId") or "")
    version = data.get("contentVersion")
    items = data.get("items")
    if (
        (expected_batch and batch_id != expected_batch)
        or not isinstance(version, int) or isinstance(version, bool) or version < 0
        or not isinstance(items, list) or data.get("itemsTruncationNotice")
        or data.get("count", len(items)) != len(items)
    ):
        return None
    normalized = []
    for item in items:
        if not isinstance(item, dict) or not str(item.get("id", "")).isdigit() or int(item["id"]) <= 0:
            return None
        if any(not str(item.get(field) or "") for field in ("word", "code", "type", "action")):
            return None
        normalized.append({field: int(item[field]) if field == "id" else item.get(field) for field in _FIELDS})
    if len({item["id"] for item in normalized}) != len(normalized):
        return None
    return {"batchId": batch_id, "contentVersion": version, "items": sorted(normalized, key=lambda item: item["id"]), "batchUrl": str(data.get("batchUrl") or "")}


def _will_write(name: str, args: Dict) -> bool:
    if name in {"keytao_create_phrase", "keytao_batch_add_to_draft", "keytao_submit_batch"}:
        return args.get("confirmed") is True and args.get("preview_only") is not True
    if name == "keytao_shift_phrase_code":
        return bool(args.get("confirmed_plan_digest"))
    if name in {"keytao_remove_draft_item", "keytao_batch_remove_draft_items"}:
        return bool(args.get("expected_target_digest"))
    if name == "keytao_recall_batch":
        return bool(args.get("batch_id")) and args.get("expected_content_version") is not None
    return name == "keytao_update_draft_item_weight"


async def _fetch_raw_item(item_id: int) -> Dict:
    """The public PR detail endpoint preserves nullable persisted weight."""
    data = await http_client.keytao_json("GET", f"/api/pull-requests/{item_id}", timeout=10.0)
    return data.get("pullRequest") if isinstance(data.get("pullRequest"), dict) else {}


@observe_tool_call
async def _read_snapshot(listed: Callable, identity: Dict, batch_id: str = "", *, allow_absence: bool = False) -> Optional[Dict]:
    data = await listed(**identity, **({"batch_id": batch_id} if batch_id else {}))
    absence = bool(
        allow_absence and isinstance(data, dict) and data.get("success") is True
        and data.get("batchId") in (None, "") and data.get("contentVersion") == 0
        and not isinstance(data.get("contentVersion"), bool) and data.get("items") == []
    )
    snapshot = _snapshot(data, "" if absence else batch_id)
    if snapshot is None or not snapshot["items"]:
        return snapshot
    semaphore = asyncio.Semaphore(4)
    batch_statuses = set()

    async def raw_row(row: Dict) -> Optional[Dict]:
        async with semaphore:
            raw = await _fetch_raw_item(row["id"])
        if (
            str(raw.get("batchId") or "") != snapshot["batchId"]
            or raw.get("id") != row["id"]
            or any(raw.get(field) != row.get(field) for field in ("word", "code", "type", "action", "oldWord", "remark"))
            or "weight" not in raw
        ):
            return None
        raw_batch = raw.get("batch") or {}
        if raw_batch.get("id") == snapshot["batchId"] and raw_batch.get("status"):
            batch_statuses.add(raw_batch["status"])
        # The list supplies the effective manual-review seal; raw detail
        # supplies nullable persisted weight, which must never become zero.
        return {**row, "weight": raw["weight"],
                "_targetPhraseId": raw.get("targetPhraseId"),
                "_targetFingerprint": raw.get("targetFingerprint")}

    try:
        rows = await asyncio.gather(*(raw_row(row) for row in snapshot["items"]))
    except Exception:
        return None
    if any(row is None for row in rows):
        return None
    verify = _snapshot(await listed(**identity, batch_id=snapshot["batchId"]), snapshot["batchId"])
    if verify is None or verify["contentVersion"] != snapshot["contentVersion"] or {row["id"] for row in verify["items"]} != {row["id"] for row in rows}:
        return None
    return {**snapshot, "items": rows, "batchStatus": next(iter(batch_statuses)) if len(batch_statuses) == 1 else ""}


async def invoke_with_operation_journal(name: str, args: Dict, invoke: Callable, get_tool: Callable) -> Any:
    """Wrap the single effective-tool boundary, including model ticket replays."""
    capture = current_operation_capture.get()
    if capture is None or name not in _MUTATION_TOOLS:
        return await invoke(**args)
    store = get_default_draft_mutation_claim_store()
    if store.actor_has_running_undo(capture.platform, capture.actor):
        return {"success": False, "message": "上一笔撤销仍在核验中，本次未写入；请先完成原撤销。"}
    if not _will_write(name, args):
        return await invoke(**args)
    listed = get_tool("keytao_list_draft_items")
    identity = {"platform": capture.platform, "platform_id": capture.actor}
    batch_id = str(args.get("batch_id") or "")
    # Submit/recall change status; no post-submit list or current-batch lookup.
    before = None
    if name != "keytao_recall_batch":
        before = await _read_snapshot(
            listed, identity, batch_id,
            allow_absence=(args.get("expected_content_version") == 0 and not isinstance(args.get("expected_content_version"), bool)),
        )
        if before is None:
            return {"success": False, "message": "未能核验写入前的完整草稿，本次未写入。"}
    result = await invoke(**args)
    if not isinstance(result, dict) or result.get("success") is not True or result.get("requiresConfirmation") is True:
        return result
    if result.get("replayedResolvedMutation") or result.get("alreadyApplied"):
        return result
    batch_id = str(result.get("batchId") or batch_id or "")
    if name == "keytao_submit_batch":
        after = dict(before or {})
        version = result.get("contentVersion")
        if isinstance(version, int) and not isinstance(version, bool):
            after["contentVersion"] = version
    else:
        after = await _read_snapshot(listed, identity, batch_id) if batch_id else None
    if capture.before is None:
        capture.before = before
    prior = capture.payload or {}
    payload = {
        "operationId": capture.operation_id, "completedAt": time.time(),
        "before": capture.before, "after": after,
        "batchId": batch_id, "batchUrl": str(result.get("batchUrl") or (after or {}).get("batchUrl") or prior.get("batchUrl") or ""),
        "submitted": name == "keytao_submit_batch" or bool(prior.get("submitted")),
        "tools": list(dict.fromkeys([*prior.get("tools", []), name])),
    }
    capture.payload = payload
    store.save_completed_operation(capture.scope, capture.platform, capture.actor, payload)
    return result


def _clean_command(message: str) -> str:
    return re.sub(r"[\s。！!，,]+$", "", str(message or "").strip())


def _operation_rows(record: Dict) -> Optional[list]:
    before, after = record.get("before"), record.get("after")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    old = {row["id"]: row for row in before["items"]}
    return [row for row in after["items"] if old.get(row["id"]) != row]


def _restore_rows(record: Dict) -> list:
    before, after = record.get("before") or {}, record.get("after") or {}
    current = {row["id"]: row for row in after.get("items", [])}
    return [row for row in before.get("items", []) if current.get(row["id"]) != row]


def _semantic_rows(rows: list) -> list:
    return sorted(
        (json.dumps({key: row.get(key) for key in (*_FIELDS[1:], "_targetPhraseId", "_targetFingerprint")}, ensure_ascii=False, sort_keys=True) for row in rows),
    )


async def _restore_targets_unchanged(rows: list) -> bool:
    """Never refresh an old Change/Delete capability onto a changed live target."""
    for row in rows:
        if row["action"] not in {"Change", "Delete"}:
            continue
        target_id, fingerprint = row.get("_targetPhraseId"), row.get("_targetFingerprint")
        if not isinstance(target_id, int) or not isinstance(fingerprint, str):
            return False
        word = row.get("oldWord") if row["action"] == "Change" else row["word"]
        matched = False
        for page in range(1, 33):
            data = await http_client.keytao_json("GET", "/api/phrases/by-word", params={"word": word, "page": page}, timeout=10.0)
            for phrase in data.get("phrases", []):
                if phrase.get("id") != target_id:
                    continue
                target = {key: phrase.get(key) for key in ("id", "word", "code", "type", "weight", "remark")}
                target.update(status="Finish", userId=(phrase.get("user") or {}).get("id"))
                digest = hashlib.sha256(json.dumps(target, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                matched = digest == fingerprint
            if matched or page >= (data.get("pagination") or {}).get("totalPages", 0):
                break
        if not matched:
            return False
    return True


@observe_tool_call
async def _restore_request(identity: Dict, batch_id: str, items: list, **ticket: Any) -> Dict:
    """Restore a persisted exact set through the existing strict batch CAS API."""
    return await http_client.keytao_json(
        "POST", "/api/bot/pull-requests/batch",
        platform=identity["platform"], platform_id=identity["platform_id"],
        json_body={"platform": identity["platform"], "platformId": identity["platform_id"], "batchId": batch_id, "items": items, **ticket},
        allow_status={400, 409}, timeout=30.0, retries=0,
        idempotent=ticket.get("confirmed") is not True,
    )


def operation_description(record: Dict) -> str:
    rows = _operation_rows(record)
    if rows is None:
        rows = (record.get("after") or {}).get("items", [])
    parts = []
    for row in rows:
        prefix = {"Create": "新增", "Delete": "删除", "Change": "修改"}.get(row.get("action"), "变更")
        parts.append(f"{prefix} {row['word']}@{row['code']}")
    for row in _restore_rows(record):
        parts.append(f"原草稿 {row['word']}@{row['code']}")
    return "、".join(parts) or ("提交草稿" if record.get("submitted") else "最近草稿操作")


def _confirmation(record: Dict, reason: str) -> str:
    return f"{reason}；最近操作：{operation_description(record)}；{record.get('batchUrl', '')}；是否整笔撤销？请引用本条回复「是」或「否」。"


async def handle_completed_undo(key: Any, message: str, reply_reference: Any, get_tool: Callable) -> Optional[str]:
    """Resolve a literal undo, or a native quote-confirm, before any model call."""
    command = _clean_command(message)
    scope = conversation_scope(key)
    store = get_default_draft_mutation_claim_store()
    record = store.completed_operation(scope)
    quoted = bool(
        record and record.get("confirmationPrompt")
        and getattr(reply_reference, "is_to_bot", False)
        and str(getattr(reply_reference, "text", "")).strip() == record["confirmationPrompt"]
    )
    if command not in _UNDO_WORDS and not (quoted and command in _YES_WORDS | _NO_WORDS):
        if re.fullmatch(r"(?:取消|撤销|撤回|回滚)[？?]+", command):
            return "这条消息是在询问撤销，尚未执行。"
        return None
    if record is None:
        if command == "撤回":
            return None
        return "当前对话没有可核验的已完成写入，未执行撤销。"
    link = str(record.get("batchUrl") or "")
    if quoted and command in _NO_WORDS:
        record.pop("confirmationPrompt", None)
        store.update_completed_operation(scope, record["operationId"], record["status"], record["status"], record)
        return f"已保留这笔操作。草稿地址：{link}"
    if record["status"] == "undone":
        return str(record.get("undoReceipt") or f"这笔操作此前已撤销。草稿地址：{link}")
    rows = _operation_rows(record)
    restoring = _restore_rows(record)
    submit_only = rows == [] and record.get("tools") == ["keytao_submit_batch"]
    if rows is None or (not rows and not restoring and not submit_only):
        return f"最近操作：{operation_description(record)}；其原草稿条目已被修改或移除，当前接口无法完整恢复，未执行撤销。草稿地址：{link}"
    if record["status"] == "ready" and any(
        marker in str(row.get("remark") or "")
        for row in restoring for marker in _SERVER_REVIEW_DELIMITERS
    ):
        return f"原草稿备注含服务端审核记录，当前接口不允许原样恢复；本次未撤销任何条目。草稿地址：{link}"
    if restoring and record["status"] == "ready":
        try:
            targets_unchanged = await _restore_targets_unchanged(restoring)
        except Exception:
            targets_unchanged = False
        if not targets_unchanged:
            return f"原草稿修改或删除所绑定的词库条目已变化或未能核验，未执行撤销。草稿地址：{link}"
    identity = {"platform": key.platform, "platform_id": key.actor_id}
    capture = current_operation_capture.get()
    immediate = bool(
        capture and capture.scope == scope and capture.previous_turn == record["operationId"]
        and 0 <= time.time() - record["completedAt"] <= UNDO_WINDOW_SECONDS
        and record.get("latestActorWrite") == record["operationId"]
    )
    @observe_tool_call
    async def call(name: str, **arguments: Any) -> Dict:
        result = await get_tool(name)(**identity, **arguments)
        return result if isinstance(result, dict) else {"success": False, "message": "工具返回格式不完整"}

    async def ask(reason: str, snapshot: Optional[Dict] = None) -> str:
        prompt = _confirmation(record, reason)
        record["confirmationPrompt"] = prompt
        record["confirmationVersion"] = (snapshot or record.get("after") or {}).get("contentVersion")
        store.update_completed_operation(scope, record["operationId"], record["status"], record["status"], record)
        return prompt

    if record["status"] == "ready" and not immediate and not (quoted and command in _YES_WORDS):
        return await ask("距离上次写入已有一段时间或其后有其他操作")
    batch_id = record["batchId"]
    if record.get("submitted") and not record.get("recalled"):
        if record.get("recallStarted"):
            claim = store.get(key.platform, key.actor_id) or {}
            payload = claim.get("payload") or {}
            original_version = record.get("recallVersion", record["after"]["contentVersion"])
            if (
                claim.get("operationKind") == "recall" and claim.get("status") == "inflight"
                and payload.get("batchId") == batch_id and payload.get("contentVersion") == original_version
            ):
                recovered = await _read_snapshot(get_tool("keytao_list_draft_items"), identity, batch_id)
                if (
                    recovered is not None and recovered.get("batchStatus") == "Draft"
                    and recovered["contentVersion"] == original_version + 1
                    and recovered["items"] == record["after"]["items"]
                ):
                    store.resolve(key.platform, key.actor_id, "recall", claim["fingerprint"], {
                        "success": True, "alreadyApplied": True, "batchId": batch_id,
                        "contentVersion": recovered["contentVersion"], "batchUrl": link,
                        "message": "已核验原批次撤回生效",
                    })
        preview = await call("keytao_recall_batch")
        expected_recall_version = record.get("confirmationVersion") if quoted else record["after"]["contentVersion"]
        recovered_recall = False
        if record.get("recallStarted") and preview.get("success") is True and (
            preview.get("alreadyApplied") is True or preview.get("replayedResolvedMutation") is True
        ):
            claim = store.get(key.platform, key.actor_id) or {}
            payload, result = claim.get("payload") or {}, claim.get("result") or {}
            original_version = record.get("recallVersion", record["after"]["contentVersion"])
            recovered_recall = bool(
                claim.get("operationKind") == "recall" and claim.get("status") == "resolved"
                and payload.get("batchId") == batch_id and payload.get("contentVersion") == original_version
                and result.get("success") is True and result.get("batchId") == batch_id
                and preview.get("batchId") == batch_id and preview.get("contentVersion") == original_version + 1
            )
        if (
            (preview.get("requiresConfirmation") is not True and not recovered_recall)
            or preview.get("batchId") != batch_id
        ):
            return f"原批次提审状态已变化，未撤回或撤销。草稿地址：{link}"
        if not recovered_recall and preview.get("contentVersion") != expected_recall_version:
            return await ask("原批次提审后又有变化", {"contentVersion": preview.get("contentVersion")})
        if record["status"] == "ready":
            if not store.update_completed_operation(scope, record["operationId"], "ready", "undoing", record):
                return f"这笔撤销已由另一请求接手。草稿地址：{link}"
            record["status"] = "undoing"
        record["recallStarted"] = True
        record["recallVersion"] = record.get("recallVersion", preview["contentVersion"])
        store.update_completed_operation(scope, record["operationId"], "undoing", "undoing", record)
        recalled = preview if recovered_recall else await call("keytao_recall_batch", batch_id=batch_id, expected_content_version=preview["contentVersion"])
        if recalled.get("success") is not True or recalled.get("batchId") != batch_id:
            return f"撤回提审未确认完成，尚未撤销条目：{recalled.get('message', '结果不确定')}。草稿地址：{link}"
        record["recalled"] = True
        store.update_completed_operation(scope, record["operationId"], "undoing", "undoing", record)
    if submit_only and record.get("recalled"):
        snapshot = await _read_snapshot(get_tool("keytao_list_draft_items"), identity, batch_id)
        if snapshot is None or snapshot["items"] != record["before"]["items"]:
            return f"已撤回提审，但草稿条目仍待核验。草稿地址：{link}"
        receipt = f"✅ 已撤销本次提审，原草稿条目已保留。草稿地址：{link}"
        record["undoReceipt"] = receipt
        store.update_completed_operation(scope, record["operationId"], "undoing", "undone", record)
        return receipt
    snapshot = await _read_snapshot(get_tool("keytao_list_draft_items"), identity, batch_id)
    if snapshot is None:
        return f"未能核验原批次的完整草稿，尚未撤销条目。草稿地址：{link}"
    current = {row["id"]: row for row in snapshot["items"]}
    if record["status"] == "undoing" and record.get("deleteStarted") and all(row["id"] not in current for row in rows):
        result = record.get("deleteResult") or {}
        if result.get("success") is not True:
            claim = store.get(key.platform, key.actor_id)
            payload = (claim or {}).get("payload") or {}
            if (
                (claim or {}).get("operationKind") != "delete"
                or payload.get("batchId") != batch_id
                or payload.get("ids") != [row["id"] for row in rows]
            ):
                return f"原条目当前已不存在，但没有本次撤销的成功回执，无法确认由本次撤销完成。草稿地址：{link}"
            result = await call("keytao_batch_remove_draft_items", ids=payload["ids"], batch_id=batch_id)
        if result.get("success") is not True or result.get("batchId") != batch_id or result.get("successCount") != len(rows):
            return f"原条目已不存在，但本次撤销结果仍未确认。草稿地址：{link}"
        deleted = True
    else:
        deleted = False
        if any(current.get(row["id"]) != row for row in rows):
            return f"原操作的条目已变化，未继续撤销。草稿地址：{link}"
        expected = record.get("confirmationVersion") if quoted else record["after"]["contentVersion"]
        if record["status"] == "ready" and snapshot["contentVersion"] != expected:
            return await ask("写入后草稿又有变化", snapshot)
        if record["status"] == "ready":
            if not store.update_completed_operation(scope, record["operationId"], "ready", "undoing", record):
                return f"这笔撤销已由另一请求接手。草稿地址：{link}"
            record["status"] = "undoing"
        ids = [row["id"] for row in rows]
        if not ids:
            record["deleteStarted"] = True
            record["deleteResult"] = {"success": True, "batchId": batch_id, "successCount": 0}
            record["undoRemaining"] = snapshot["items"]
            store.update_completed_operation(scope, record["operationId"], "undoing", "undoing", record)
            deleted = True
        else:
            result = await _delete_operation_rows(call, store, scope, record, snapshot, rows)
            if result is not None:
                return result
            deleted = True
    verified = await _read_snapshot(get_tool("keytao_list_draft_items"), identity, batch_id)
    if not deleted or verified is None or any(row["id"] in {item["id"] for item in verified["items"]} for row in rows):
        return f"删除请求已返回，但尚未核验整笔撤销完成。草稿地址：{link}"
    verified_by_id = {item["id"]: item for item in verified["items"]}
    if any(verified_by_id.get(item["id"]) != item for item in record.get("undoRemaining", [])):
        return f"本次操作的条目已撤销，但其余草稿又有变化，尚未确认完整恢复。草稿地址：{link}"
    if restoring:
        restore_error = await _restore_original_rows(store, scope, record, verified, restoring, identity, get_tool)
        if restore_error:
            return restore_error
    receipt = f"✅ {'已先撤回提审，并' if record.get('recalled') else '已'}撤销整笔操作：{operation_description(record)}。草稿地址：{link}"
    record["undoReceipt"] = receipt
    store.update_completed_operation(scope, record["operationId"], "undoing", "undone", record)
    return receipt


async def _delete_operation_rows(call: Callable, store: Any, scope: str, record: Dict, snapshot: Dict, rows: list) -> Optional[str]:
    batch_id, link = record["batchId"], record["batchUrl"]
    ids = [row["id"] for row in rows]
    targets = [{field: row[field] for field in ("id", "word", "code", "action", "type")} for row in rows]
    try:
        preview = await call("keytao_batch_remove_draft_items", ids=ids, batch_id=batch_id)
    except Exception:
        _release_unstarted_undo(store, scope, record)
        return f"撤销检查未完成，尚未撤销条目。草稿地址：{link}"
    if (
        preview.get("requiresConfirmation") is not True or preview.get("batchId") != batch_id
        or preview.get("contentVersion") != snapshot["contentVersion"] or preview.get("targets") != targets
        or not re.fullmatch(r"[0-9a-f]{64}", str(preview.get("targetDigest") or ""))
    ):
        _release_unstarted_undo(store, scope, record)
        return f"撤销检查的条目或版本不一致，尚未撤销条目。草稿地址：{link}"
    record["deleteStarted"] = True
    record["undoRemaining"] = [row for row in snapshot["items"] if row["id"] not in ids]
    store.update_completed_operation(scope, record["operationId"], "undoing", "undoing", record)
    result = await call(
        "keytao_batch_remove_draft_items", ids=ids, batch_id=batch_id,
        expected_content_version=preview["contentVersion"], expected_target_digest=preview["targetDigest"], expected_targets=targets,
    )
    if result.get("success") is not True or result.get("batchId") != batch_id or result.get("successCount") != len(ids):
        return f"撤销尚未确认全部完成：{result.get('message', '结果不完整')}。草稿地址：{link}"
    record["deleteResult"] = result
    store.update_completed_operation(scope, record["operationId"], "undoing", "undoing", record)
    return None


def _release_unstarted_undo(store: Any, scope: str, record: Dict) -> None:
    """A rejected read-only preview must not leave a permanent actor fence."""
    if record.get("recallStarted") or record.get("restoreStarted") or (
        record.get("deleteStarted") and _operation_rows(record)
    ):
        return
    for field in ("deleteStarted", "deleteResult", "undoRemaining"):
        record.pop(field, None)
    if store.update_completed_operation(scope, record["operationId"], "undoing", "ready", record):
        record["status"] = "ready"


async def _restore_original_rows(store: Any, scope: str, record: Dict, snapshot: Dict, restoring: list, identity: Dict, get_tool: Callable) -> Optional[str]:
    batch_id, link = record["batchId"], record["batchUrl"]
    remaining_ids = {row["id"] for row in record["undoRemaining"]}

    def is_restored(current: Dict) -> bool:
        restored = [row for row in current["items"] if row["id"] not in remaining_ids]
        remaining = [row for row in current["items"] if row["id"] in remaining_ids]
        return (
            _semantic_rows(restored) == _semantic_rows(restoring)
            and remaining == record["undoRemaining"]
            and current["contentVersion"] == record["restoreVersion"] + 1
        )

    if record.get("restoreStarted"):
        if is_restored(snapshot):
            record["restoreVerified"] = True
            store.update_completed_operation(scope, record["operationId"], "undoing", "undoing", record)
            return None
        return f"本次新条目已移除，恢复原草稿的请求仍待核验；已保留原操作记录，未重复写入。草稿地址：{link}"
    if snapshot["items"] != record["undoRemaining"]:
        return f"本次新条目已移除，但恢复前草稿又有变化，尚未恢复原条目。草稿地址：{link}"
    items = [{field: row[field] for field in _FIELDS[1:] if row.get(field) is not None} for row in restoring]
    try:
        preview = await _restore_request(identity, batch_id, items, confirmed=False, previewOnly=True)
    except Exception:
        _release_unstarted_undo(store, scope, record)
        return f"本次新条目已移除，原草稿恢复检查失败，尚未恢复原条目。草稿地址：{link}"
    if (
        preview.get("requiresConfirmation") is not True or preview.get("batchId") != batch_id
        or preview.get("contentVersion") != snapshot["contentVersion"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(preview.get("warningDigest") or ""))
    ):
        _release_unstarted_undo(store, scope, record)
        return f"本次新条目已移除，原草稿尚未恢复：{preview.get('message', '恢复检查的批次或版本不一致')}。草稿地址：{link}"
    try:
        targets_unchanged = await _restore_targets_unchanged(restoring)
    except Exception:
        targets_unchanged = False
    if not targets_unchanged:
        _release_unstarted_undo(store, scope, record)
        return f"原草稿的修改或删除目标在恢复检查期间发生变化，尚未恢复原条目。草稿地址：{link}"
    record["restoreStarted"] = True
    record["restoreVersion"] = snapshot["contentVersion"]
    record["restoreItems"] = items
    record["restoreWarningDigest"] = preview["warningDigest"]
    store.update_completed_operation(scope, record["operationId"], "undoing", "undoing", record)
    try:
        result = await _restore_request(
            identity, batch_id, items, confirmed=True,
            expectedContentVersion=record["restoreVersion"], expectedWarningDigest=record["restoreWarningDigest"],
        )
    except Exception:
        return f"本次新条目已移除，恢复原草稿的结果不确定；已锁定原操作，未重复写入。草稿地址：{link}"
    record["restoreResult"] = result
    if result.get("success") is False:
        record["restoreStarted"] = False
    store.update_completed_operation(scope, record["operationId"], "undoing", "undoing", record)
    if result.get("success") is not True or result.get("batchId") != batch_id or result.get("pullRequestCount") != len(items):
        return f"本次新条目已移除，原草稿恢复尚未全部确认：{result.get('message', '返回结果不完整')}。草稿地址：{link}"
    verified = await _read_snapshot(get_tool("keytao_list_draft_items"), identity, batch_id)
    if verified is None or not is_restored(verified):
        return f"原草稿恢复请求已返回，但编码、权重、审词标记或备注仍未全部核验。草稿地址：{link}"
    record["restoreVerified"] = True
    store.update_completed_operation(scope, record["operationId"], "undoing", "undoing", record)
    return None
