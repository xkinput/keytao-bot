"""Local stack orchestration, real QQ entry-point harness, and fixture API."""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import shutil
import subprocess
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import httpx

from .recording import ArtifactRecorder
from .safety import (
    RESERVED_BINDING_PREFIX,
    RESERVED_EMAIL_SUFFIX,
    SafetyViolation,
    validate_admin_identity,
    validate_keytao_base,
    validate_reserved_identity,
    validate_test_binding,
)


class RigInfrastructureError(RuntimeError):
    """Raised when the local stack or fixture contract is unavailable."""


LOCAL_NEXT_RETRY_BACKOFF_SECONDS = (0.5, 1.0, 2.0)
LOCAL_NEXT_CONNECT_TIMEOUT_SECONDS = 5.0
LOCAL_NEXT_REQUEST_TIMEOUT_SECONDS = 90.0


class NextServer:
    def __init__(
        self,
        *,
        next_dir: Path,
        base_url: str,
        artifact_dir: Path,
        start_timeout: float,
        child_env: dict[str, str],
    ) -> None:
        self.next_dir = next_dir
        self.base_url = validate_keytao_base(base_url)
        self.artifact_dir = artifact_dir
        self.start_timeout = start_timeout
        self.child_env = child_env
        self.process: Optional[subprocess.Popen[str]] = None
        self.log_handle: Any = None
        self.reused_existing = False

    def _prepare_runtime_dir(self) -> Path:
        """Create a writable Next workspace around the read-only source tree."""
        runtime_dir = self.artifact_dir / "keytao-next-runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        for source in self.next_dir.iterdir():
            if source.name in {
                ".DS_Store",
                ".cache",
                ".claude",
                ".direnv",
                ".next",
                ".vercel",
                "test-results",
            }:
                continue
            destination = runtime_dir / source.name
            if destination.exists() or destination.is_symlink():
                continue
            if source.name in {".git", "node_modules"} or source.name.startswith(
                ".env"
            ):
                destination.symlink_to(source, target_is_directory=source.is_dir())
            elif source.is_dir():
                shutil.copytree(source, destination, symlinks=True)
            else:
                shutil.copy2(source, destination)
        return runtime_dir

    async def _probe(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=2.0, follow_redirects=False) as client:
                response = await client.get(
                    f"{self.base_url}/api/phrases/by-word",
                    params={"word": "赤溪", "page": 1},
                )
            if response.status_code != 200:
                return False
            payload = response.json()
            return isinstance(payload.get("phrases"), list) and isinstance(
                payload.get("pagination"), dict
            )
        except (httpx.HTTPError, ValueError, TypeError):
            return False

    async def ensure_running(self) -> None:
        if await self._probe():
            self.reused_existing = True
            return
        next_binary = self.next_dir / "node_modules" / ".bin" / "next"
        if not next_binary.is_file():
            raise RigInfrastructureError(
                "keytao-next dependencies are not installed; local next binary is missing"
            )
        log_path = self.artifact_dir / "keytao-next.log"
        self.log_handle = log_path.open("w", encoding="utf-8")
        command = [
            str(next_binary),
            "dev",
            "--webpack",
            "--hostname",
            "127.0.0.1",
            "--port",
            self.base_url.rsplit(":", 1)[-1],
        ]
        runtime_dir = self._prepare_runtime_dir()
        self.process = subprocess.Popen(
            command,
            cwd=runtime_dir,
            env=self.child_env,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.start_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.log_handle.flush()
                tail = (self.artifact_dir / "keytao-next.log").read_text(
                    encoding="utf-8", errors="replace"
                )[-4000:]
                raise RigInfrastructureError(
                    f"keytao-next exited during startup ({self.process.returncode}):\n{tail}"
                )
            if await self._probe():
                return
            await asyncio.sleep(0.5)
        raise RigInfrastructureError(
            f"keytao-next did not become ready within {self.start_timeout:.0f}s"
        )

    async def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, 10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait, 5)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


class LocalNextClient:
    def __init__(self, *, base_url: str, bot_token: str) -> None:
        self.base_url = validate_keytao_base(base_url)
        self.bot_token = bot_token
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def headers(self) -> dict[str, str]:
        return {"X-Bot-Token": self.bot_token}

    def _pooled_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    LOCAL_NEXT_REQUEST_TIMEOUT_SECONDS,
                    connect=LOCAL_NEXT_CONNECT_TIMEOUT_SECONDS,
                    pool=LOCAL_NEXT_CONNECT_TIMEOUT_SECONDS,
                ),
                follow_redirects=False,
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=30.0,
                ),
            )
        return self._client

    async def _reset_client(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    async def close(self) -> None:
        """Close the shared local transport pool."""

        await self._reset_client()

    async def _request_with_retries(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        attempt_count = len(LOCAL_NEXT_RETRY_BACKOFF_SECONDS) + 1
        for attempt in range(1, attempt_count + 1):
            try:
                return await self._pooled_client().request(method, url, **kwargs)
            except httpx.TransportError as error:
                await self._reset_client()
                if attempt == attempt_count:
                    raise
                delay = LOCAL_NEXT_RETRY_BACKOFF_SECONDS[attempt - 1]
                print(
                    f"Local next transport retry {attempt}/{attempt_count} "
                    f"for {method} {url} after {type(error).__name__}: {error}; "
                    f"resetting connection pool and backing off {delay:.1f}s"
                )
                await asyncio.sleep(delay)
        raise AssertionError("local next retry loop exhausted without a result")

    async def _json(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        body: Optional[dict[str, Any]] = None,
        allowed_status: tuple[int, ...] = (200,),
        authenticated: bool = True,
        bearer_token: str = "",
    ) -> tuple[int, dict[str, Any]]:
        headers = dict(self.headers) if authenticated else {}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        response = await self._request_with_retries(
            method,
            path,
            params=params,
            json=body,
            headers=headers,
        )
        try:
            payload = response.json()
        except ValueError as error:
            raise RigInfrastructureError(
                f"Local next returned non-JSON for {method} {path}: HTTP {response.status_code}"
            ) from error
        if response.status_code not in allowed_status:
            raise RigInfrastructureError(
                f"Local next rejected {method} {path}: HTTP {response.status_code} {payload}"
            )
        if not isinstance(payload, dict):
            raise RigInfrastructureError(f"Local next returned a non-object for {method} {path}")
        return response.status_code, payload

    async def find_user(self, platform_id: str) -> Optional[dict[str, Any]]:
        status, payload = await self._json(
            "POST",
            "/api/bot/user/find",
            body={"platform": "qq", "platformId": platform_id},
            allowed_status=(200, 404),
        )
        if status == 404:
            return None
        if payload.get("found") is not True or not isinstance(payload.get("user"), dict):
            raise RigInfrastructureError("The reserved QQ binding returned an invalid user payload")
        return payload["user"]

    async def get_draft(self, platform_id: str) -> dict[str, Any]:
        _status, payload = await self._json(
            "GET",
            "/api/bot/batches/latest-draft/items",
            params={"platform": "qq", "platformId": platform_id},
        )
        if payload.get("success") is not True or not isinstance(payload.get("items"), list):
            raise RigInfrastructureError(f"Could not read E2E draft: {payload}")
        return payload

    async def clean_draft(self, platform_id: str) -> dict[str, Any]:
        snapshot = await self.get_draft(platform_id)
        items = snapshot.get("items", [])
        if not items:
            return {"success": True, "deleted": 0, "batchId": snapshot.get("batchId")}
        batch_id = snapshot.get("batchId")
        content_version = snapshot.get("contentVersion")
        if not isinstance(batch_id, str) or not isinstance(content_version, int):
            raise RigInfrastructureError("Draft cleanup lacks a concrete batch/version")
        targets = [
            {
                "id": int(item["id"]),
                "word": str(item.get("word") or ""),
                "code": str(item.get("code") or ""),
                "action": str(item.get("action") or ""),
                "type": str(item.get("type") or "Phrase"),
            }
            for item in items
        ]
        _status, result = await self._json(
            "DELETE",
            "/api/bot/pull-requests/batch-draft",
            body={
                "platform": "qq",
                "platformId": platform_id,
                "ids": [item["id"] for item in targets],
                "batchId": batch_id,
                "expectedContentVersion": content_version,
                "expectedTargets": targets,
            },
            allowed_status=(200, 409),
        )
        if result.get("success") is not True:
            raise RigInfrastructureError(f"Draft cleanup failed closed: {result}")
        final = await self.get_draft(platform_id)
        if final.get("items"):
            raise RigInfrastructureError("Draft cleanup returned success but items remain")
        return {"success": True, "deleted": len(targets), "batchId": batch_id}

    async def phrases_by_word(self, word: str) -> list[dict[str, Any]]:
        _status, payload = await self._json(
            "GET",
            "/api/phrases/by-word",
            params={"word": word, "page": 1},
            authenticated=False,
        )
        values = payload.get("phrases")
        if not isinstance(values, list):
            raise RigInfrastructureError("by-word response has no phrase list")
        return [item for item in values if isinstance(item, dict)]

    async def phrases_by_code(self, code: str) -> list[dict[str, Any]]:
        _status, payload = await self._json(
            "GET",
            "/api/phrases/by-code",
            params={"code": code, "page": 1},
            authenticated=False,
        )
        values = payload.get("phrases")
        if not isinstance(values, list):
            raise RigInfrastructureError("by-code response has no phrase list")
        return [item for item in values if isinstance(item, dict)]

    async def encode(self, word: str) -> dict[str, Any]:
        _status, payload = await self._json(
            "GET",
            "/api/phrases/encode",
            params={"word": word},
            authenticated=False,
        )
        return payload

    async def add_draft_items(
        self,
        *,
        platform_id: str,
        items: list[dict[str, Any]],
    ) -> dict[str, Any]:
        current = await self.get_draft(platform_id)
        request_body = {
            "platform": "qq",
            "platformId": platform_id,
            "batchId": current.get("batchId"),
            "confirmed": False,
            "previewOnly": True,
            "items": items,
        }
        _status, preview = await self._json(
            "POST",
            "/api/bot/pull-requests/batch-draft",
            body=request_body,
            allowed_status=(200, 400, 409),
        )
        if preview.get("requiresConfirmation") is not True:
            raise RigInfrastructureError(
                f"Draft write did not return a preview ticket: {preview}"
            )
        batch_id = preview.get("batchId")
        content_version = preview.get("contentVersion")
        warning_digest = preview.get("warningDigest")
        if (
            not isinstance(batch_id, str)
            or not isinstance(content_version, int)
            or not isinstance(warning_digest, str)
        ):
            raise RigInfrastructureError("Draft preview is missing its CAS fields")
        _status, added = await self._json(
            "POST",
            "/api/bot/pull-requests/batch-draft",
            body={
                **request_body,
                "batchId": batch_id,
                "confirmed": True,
                "previewOnly": False,
                "expectedContentVersion": content_version,
                "expectedWarningDigest": warning_digest,
            },
            allowed_status=(200, 400, 409),
        )
        if (
            added.get("success") is not True
            or added.get("successCount") != len(items)
            or added.get("failedCount") != 0
        ):
            raise RigInfrastructureError(f"Draft write failed closed: {added}")
        return added

    async def submit_batch(
        self,
        *,
        platform_id: str,
        batch_id: str,
        content_version: int,
    ) -> dict[str, Any]:
        path = f"/api/bot/batches/{quote(batch_id, safe='')}/submit"
        _status, preview = await self._json(
            "POST",
            path,
            body={
                "platform": "qq",
                "platformId": platform_id,
                "confirmed": False,
                "previewOnly": True,
                "expectedContentVersion": content_version,
            },
            allowed_status=(200, 400, 409),
        )
        required_fields = ("contentVersion", "warningDigest", "snapshotDigest")
        if preview.get("requiresConfirmation") is not True or any(
            field not in preview for field in required_fields
        ):
            raise RigInfrastructureError(
                f"Batch submit did not return a complete preview ticket: {preview}"
            )
        _status, submitted = await self._json(
            "POST",
            path,
            body={
                "platform": "qq",
                "platformId": platform_id,
                "confirmed": True,
                "previewOnly": False,
                "expectedContentVersion": preview["contentVersion"],
                "expectedWarningDigest": preview["warningDigest"],
                "expectedSnapshotDigest": preview["snapshotDigest"],
            },
            allowed_status=(200, 400, 409),
        )
        batch = submitted.get("batch")
        if (
            submitted.get("success") is not True
            or not isinstance(batch, dict)
            or batch.get("id") != batch_id
            or batch.get("status") != "Submitted"
        ):
            raise RigInfrastructureError(f"Batch submit failed closed: {submitted}")
        return {"preview": preview, "submitted": submitted}

    async def login_admin(
        self,
        *,
        identity: dict[str, str],
        password: str,
    ) -> dict[str, Any]:
        validate_reserved_identity(
            platform_id=identity["platform_id"],
            expected_name=identity["name"],
            expected_email=identity["email"],
        )
        _status, payload = await self._json(
            "POST",
            "/api/auth/login",
            body={"name": identity["name"], "password": password},
            authenticated=False,
        )
        user = payload.get("user")
        token = payload.get("token")
        if not isinstance(user, dict) or not isinstance(token, str) or not token:
            raise RigInfrastructureError("Reserved admin login returned an invalid payload")
        validate_admin_identity(
            platform_id=identity["platform_id"],
            expected_name=identity["name"],
            expected_email=identity["email"],
            user=user,
        )
        _status, current_user = await self._json(
            "GET",
            "/api/auth/me",
            authenticated=False,
            bearer_token=token,
        )
        validate_admin_identity(
            platform_id=identity["platform_id"],
            expected_name=identity["name"],
            expected_email=identity["email"],
            user=current_user,
        )
        if current_user.get("id") != user.get("id"):
            raise SafetyViolation("The admin token resolved to a different reserved user")
        return {"token": token, "user": current_user}

    async def get_admin_batch(
        self,
        *,
        batch_id: str,
        admin_token: str,
    ) -> dict[str, Any]:
        _status, payload = await self._json(
            "GET",
            f"/api/admin/batches/{quote(batch_id, safe='')}",
            authenticated=False,
            bearer_token=admin_token,
        )
        batch = payload.get("batch")
        if not isinstance(batch, dict) or not isinstance(batch.get("pullRequests"), list):
            raise RigInfrastructureError(f"Admin batch detail is invalid: {payload}")
        return batch

    async def approve_admin_batch(
        self,
        *,
        batch_id: str,
        admin_token: str,
        review_note: str,
    ) -> dict[str, Any]:
        status, payload = await self._json(
            "POST",
            f"/api/admin/batches/{quote(batch_id, safe='')}/approve",
            body={"reviewNote": review_note},
            allowed_status=(200, 400, 401, 403, 404, 409, 422, 500),
            authenticated=False,
            bearer_token=admin_token,
        )
        batch = payload.get("batch")
        if status != 200 or not isinstance(batch, dict) or batch.get("status") != "Approved":
            detail = str(payload.get("details") or payload.get("error") or payload)
            target_changed = "审批目标词条已变化或缺少实体绑定" in detail
            error_name = "BatchApprovalTargetChangedError" if target_changed else "AdminApprovalError"
            raise RigInfrastructureError(
                f"{error_name}: HTTP {status} {detail}"
            )
        return payload

    async def restore_s8_fixture(
        self,
        *,
        platform_id: str,
        admin_token: str,
        chixi_next_code: str,
    ) -> dict[str, Any]:
        """Restore S8-owned fixture codes through a compensating approved batch."""

        chixi = await self.phrases_by_word("赤溪")
        wkxk = await self.phrases_by_code("wkxk")
        shifted = await self.phrases_by_code(chixi_next_code)
        fixture_codes = {"wkxk", chixi_next_code}
        exact_wkxk = [row for row in wkxk if row.get("code") == "wkxk"]
        exact_shifted = [
            row for row in shifted if row.get("code") == chixi_next_code
        ]
        unexpected_chixi = [
            row for row in chixi if row.get("code") not in fixture_codes
        ]
        is_fixture = (
            len(chixi) == 1
            and chixi[0].get("word") == "赤溪"
            and chixi[0].get("code") == "wkxk"
            and chixi[0].get("type") == "Phrase"
            and chixi[0].get("weight") == 100
            and len(exact_wkxk) == 1
            and exact_wkxk[0].get("word") == "赤溪"
            and not exact_shifted
        )
        if is_fixture:
            return {"action": "already-restored", "verified": True}

        rows_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in [*exact_wkxk, *exact_shifted]:
            identity = (
                row.get("id"),
                row.get("word"),
                row.get("code"),
                row.get("type"),
            )
            rows_by_identity[identity] = row
        fixture_rows = list(rows_by_identity.values())
        non_rig_rows = [
            row
            for row in fixture_rows
            if not str((row.get("user") or {}).get("name") or "").startswith(
                RESERVED_BINDING_PREFIX
            )
        ]
        if unexpected_chixi or not fixture_rows or non_rig_rows:
            raise RigInfrastructureError(
                "Refusing ambiguous S8 fixture repair: "
                f"赤溪={chixi}, wkxk={wkxk}, {chixi_next_code}={shifted}, "
                f"nonRig={non_rig_rows}"
            )

        await self.clean_draft(platform_id)
        cleanup_items = [
            {
                "action": "Delete",
                "word": str(row.get("word") or ""),
                "code": str(row.get("code") or ""),
                "type": str(row.get("type") or "Phrase"),
                "needsManualReview": False,
                "remark": "E2E S8 compensating cleanup",
            }
            for row in fixture_rows
        ]
        cleanup_items.append(
            {
                "action": "Create",
                "word": "赤溪",
                "code": "wkxk",
                "type": "Phrase",
                "weight": 100,
                "needsManualReview": False,
                "remark": "E2E local fixture",
            }
        )
        added = await self.add_draft_items(platform_id=platform_id, items=cleanup_items)
        batch_id = added.get("batchId")
        content_version = added.get("contentVersion")
        if not isinstance(batch_id, str) or not isinstance(content_version, int):
            raise RigInfrastructureError("S8 cleanup draft is missing batch/version")
        submitted = await self.submit_batch(
            platform_id=platform_id,
            batch_id=batch_id,
            content_version=content_version,
        )
        approved = await self.approve_admin_batch(
            batch_id=batch_id,
            admin_token=admin_token,
            review_note="E2E S8 API-only fixture restoration",
        )
        final_chixi = await self.phrases_by_word("赤溪")
        final_wkxk = await self.phrases_by_code("wkxk")
        final_shifted = await self.phrases_by_code(chixi_next_code)
        final_exact_wkxk = [
            row for row in final_wkxk if row.get("code") == "wkxk"
        ]
        final_exact_shifted = [
            row
            for row in final_shifted
            if row.get("code") == chixi_next_code
        ]
        if not (
            len(final_chixi) == 1
            and final_chixi[0].get("code") == "wkxk"
            and final_chixi[0].get("weight") == 100
            and len(final_exact_wkxk) == 1
            and final_exact_wkxk[0].get("word") == "赤溪"
            and not final_exact_shifted
        ):
            raise RigInfrastructureError(
                "S8 cleanup approval completed without restoring the fixture: "
                f"赤溪={final_chixi}, wkxk={final_wkxk}, "
                f"{chixi_next_code}={final_shifted}"
            )
        return {
            "action": "compensating-approved-batch",
            "verified": True,
            "batchId": batch_id,
            "deletedFixtureRows": len(fixture_rows),
            "submittedStatus": submitted["submitted"]["batch"]["status"],
            "approvedStatus": approved["batch"]["status"],
        }

    async def remove_rig_owned_dictionary_words(
        self,
        *,
        platform_id: str,
        admin_token: str,
        scenario_id: str,
        fixture_words: tuple[str, ...],
    ) -> dict[str, Any]:
        """Delete declared fixture words only when every exact row is rig-owned."""

        rows_by_identity: dict[tuple[Any, ...], dict[str, Any]] = {}
        for word in fixture_words:
            exact_rows = [
                row
                for row in await self.phrases_by_word(word)
                if row.get("word") == word
            ]
            non_rig_rows = [
                row
                for row in exact_rows
                if not str((row.get("user") or {}).get("name") or "").startswith(
                    RESERVED_BINDING_PREFIX
                )
            ]
            if non_rig_rows:
                raise RigInfrastructureError(
                    f"{scenario_id} requires {word} to be absent from the local "
                    f"dictionary: {exact_rows}"
                )
            for row in exact_rows:
                identity = (
                    row.get("id"),
                    row.get("word"),
                    row.get("code"),
                    row.get("type"),
                )
                rows_by_identity[identity] = row

        fixture_rows = list(rows_by_identity.values())
        if not fixture_rows:
            return {
                "action": "already-absent",
                "verified": True,
                "fixtureWords": list(fixture_words),
                "deletedFixtureRows": 0,
            }

        await self.clean_draft(platform_id)
        items = [
            {
                "action": "Delete",
                "word": str(row.get("word") or ""),
                "code": str(row.get("code") or ""),
                "type": str(row.get("type") or "Phrase"),
                "needsManualReview": False,
                "remark": f"E2E {scenario_id} declared fixture cleanup",
            }
            for row in fixture_rows
        ]
        added = await self.add_draft_items(platform_id=platform_id, items=items)
        batch_id = added.get("batchId")
        content_version = added.get("contentVersion")
        if not isinstance(batch_id, str) or not isinstance(content_version, int):
            raise RigInfrastructureError(
                f"{scenario_id} fixture cleanup draft is missing batch/version"
            )
        submitted = await self.submit_batch(
            platform_id=platform_id,
            batch_id=batch_id,
            content_version=content_version,
        )
        approved = await self.approve_admin_batch(
            batch_id=batch_id,
            admin_token=admin_token,
            review_note=f"E2E {scenario_id} API-only declared fixture cleanup",
        )
        remaining = {
            word: [
                row
                for row in await self.phrases_by_word(word)
                if row.get("word") == word
            ]
            for word in fixture_words
        }
        remaining = {word: rows for word, rows in remaining.items() if rows}
        if remaining:
            raise RigInfrastructureError(
                f"{scenario_id} fixture cleanup approval completed but declared "
                f"words remain: {remaining}"
            )
        return {
            "action": "compensating-approved-delete",
            "verified": True,
            "batchId": batch_id,
            "fixtureWords": list(fixture_words),
            "deletedFixtureRows": len(fixture_rows),
            "submittedStatus": submitted["submitted"]["batch"]["status"],
            "approvedStatus": approved["batch"]["status"],
        }

    async def remove_s9_fixture(
        self,
        *,
        platform_id: str,
        admin_token: str,
    ) -> dict[str, Any]:
        """Remove only a rig-owned 慑服@eefj fixture through approved APIs."""
        rows = [
            row
            for row in await self.phrases_by_code("eefj")
            if row.get("code") == "eefj"
        ]
        if not rows:
            return {"action": "already-absent", "verified": True}
        invalid_rows = [
            row
            for row in rows
            if row.get("word") != "慑服"
            or row.get("type") != "Phrase"
            or not str((row.get("user") or {}).get("name") or "").startswith(
                RESERVED_BINDING_PREFIX
            )
        ]
        if invalid_rows:
            raise RigInfrastructureError(
                f"Refusing ambiguous S9 fixture cleanup: eefj={rows}"
            )

        await self.clean_draft(platform_id)
        items = [
            {
                "action": "Delete",
                "word": "慑服",
                "code": "eefj",
                "type": "Phrase",
                "needsManualReview": False,
                "remark": "E2E S9 fixture cleanup",
            }
            for _row in rows
        ]
        added = await self.add_draft_items(platform_id=platform_id, items=items)
        batch_id = added.get("batchId")
        content_version = added.get("contentVersion")
        if not isinstance(batch_id, str) or not isinstance(content_version, int):
            raise RigInfrastructureError("S9 cleanup draft is missing batch/version")
        submitted = await self.submit_batch(
            platform_id=platform_id,
            batch_id=batch_id,
            content_version=content_version,
        )
        approved = await self.approve_admin_batch(
            batch_id=batch_id,
            admin_token=admin_token,
            review_note="E2E S9 API-only fixture cleanup",
        )
        remaining = [
            row
            for row in await self.phrases_by_code("eefj")
            if row.get("code") == "eefj"
        ]
        if remaining:
            raise RigInfrastructureError(
                f"S9 cleanup approval completed but eefj remains occupied: {remaining}"
            )
        return {
            "action": "compensating-approved-delete",
            "verified": True,
            "batchId": batch_id,
            "submittedStatus": submitted["submitted"]["batch"]["status"],
            "approvedStatus": approved["batch"]["status"],
        }

    async def seed_phrase(
        self,
        *,
        platform_id: str,
        word: str,
        code: str,
        phrase_type: str = "Phrase",
    ) -> dict[str, Any]:
        current = await self.get_draft(platform_id)
        item = {
            "action": "Create",
            "word": word,
            "code": code,
            "type": phrase_type,
            "needsManualReview": False,
            "remark": "E2E local fixture",
        }
        _status, preview = await self._json(
            "POST",
            "/api/bot/pull-requests/batch-draft",
            body={
                "platform": "qq",
                "platformId": platform_id,
                "batchId": current.get("batchId"),
                "confirmed": False,
                "previewOnly": True,
                "items": [item],
            },
            allowed_status=(200, 400),
        )
        if preview.get("requiresConfirmation") is not True:
            raise RigInfrastructureError(f"Seed add did not return a preview ticket: {preview}")
        batch_id = preview.get("batchId")
        version = preview.get("contentVersion")
        warning_digest = preview.get("warningDigest")
        if not isinstance(batch_id, str) or not isinstance(version, int) or not isinstance(
            warning_digest, str
        ):
            raise RigInfrastructureError("Seed add preview is missing its CAS fields")
        _status, added = await self._json(
            "POST",
            "/api/bot/pull-requests/batch-draft",
            body={
                "platform": "qq",
                "platformId": platform_id,
                "batchId": batch_id,
                "confirmed": True,
                "previewOnly": False,
                "expectedContentVersion": version,
                "expectedWarningDigest": warning_digest,
                "items": [item],
            },
        )
        if added.get("successCount") != 1:
            raise RigInfrastructureError(f"Seed add did not write exactly one draft item: {added}")
        draft_version = added.get("contentVersion")
        if not isinstance(draft_version, int):
            draft_version = (await self.get_draft(platform_id)).get("contentVersion")
        _status, submit_preview = await self._json(
            "POST",
            f"/api/bot/batches/{quote(batch_id, safe='')}/submit",
            body={
                "platform": "qq",
                "platformId": platform_id,
                "confirmed": False,
                "previewOnly": True,
                "expectedContentVersion": draft_version,
            },
            allowed_status=(200, 400),
        )
        if submit_preview.get("requiresConfirmation") is not True:
            raise RigInfrastructureError(f"Seed submit did not return a preview ticket: {submit_preview}")
        _status, submitted = await self._json(
            "POST",
            f"/api/bot/batches/{quote(batch_id, safe='')}/submit",
            body={
                "platform": "qq",
                "platformId": platform_id,
                "confirmed": True,
                "previewOnly": False,
                "expectedContentVersion": submit_preview["contentVersion"],
                "expectedWarningDigest": submit_preview["warningDigest"],
                "expectedSnapshotDigest": submit_preview["snapshotDigest"],
            },
        )
        if submitted.get("success") is not True:
            raise RigInfrastructureError(f"Seed submit failed: {submitted}")
        submitted_version = submitted.get("batch", {}).get("contentVersion")
        _status, approved = await self._json(
            "POST",
            f"/api/bot/batches/{quote(batch_id, safe='')}/auto-approve",
            body={
                "platform": "qq",
                "platformId": platform_id,
                "reviewNote": "E2E local fixture approval",
                "expectedContentVersion": submitted_version,
            },
            allowed_status=(200, 400, 409, 422),
        )
        if approved.get("success") is not True:
            raise RigInfrastructureError(f"Seed auto-approval failed: {approved}")
        return {"batchId": batch_id, "submitted": submitted, "approved": approved}


def synthetic_qq_id(run_id: str, label: str) -> str:
    digest = hashlib.sha256(f"{run_id}:{label}".encode("utf-8")).hexdigest()
    digits = str(int(digest, 16)).zfill(78)
    return "9" + digits[:31]


def test_identity(run_id: str, label: str) -> dict[str, str]:
    normalized = label.lower().replace("_", "-")
    name = f"{RESERVED_BINDING_PREFIX}{run_id[:8]}-{normalized}"
    return {
        "platform_id": synthetic_qq_id(run_id, label),
        "name": name,
        "email": f"{name}{RESERVED_EMAIL_SUFFIX}",
    }


async def provision_test_user(
    *,
    client: LocalNextClient,
    next_dir: Path,
    next_env: dict[str, str],
    identity: dict[str, str],
    password: str = "",
) -> dict[str, Any]:
    validate_reserved_identity(
        platform_id=identity["platform_id"],
        expected_name=identity["name"],
        expected_email=identity["email"],
    )
    existing = await client.find_user(identity["platform_id"])
    if existing is not None:
        validate_test_binding(
            platform_id=identity["platform_id"],
            expected_name=identity["name"],
            expected_email=identity["email"],
            user=existing,
        )
        return existing
    tsx_binary = next_dir / "node_modules" / ".bin" / "tsx"
    if not tsx_binary.is_file():
        raise RigInfrastructureError(
            "keytao-next dependencies are not installed; local tsx binary is missing"
        )
    command = [str(tsx_binary), "scripts/initBotUser.ts"]
    child_env = dict(next_env)
    child_env.update(
        {
            "BOT_USER_NAME": identity["name"],
            "BOT_USER_EMAIL": identity["email"],
            "BOT_USER_PASSWORD": password or secrets.token_urlsafe(32),
            "BOT_QQ_ID": identity["platform_id"],
        }
    )
    completed = await asyncio.to_thread(
        subprocess.run,
        command,
        cwd=next_dir,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        raise RigInfrastructureError(
            "keytao-next initBotUser.ts failed for a reserved E2E identity:\n"
            + completed.stdout[-2000:]
            + completed.stderr[-2000:]
        )
    created = await client.find_user(identity["platform_id"])
    if created is None:
        raise RigInfrastructureError("initBotUser.ts completed but the binding is absent")
    validate_test_binding(
        platform_id=identity["platform_id"],
        expected_name=identity["name"],
        expected_email=identity["email"],
        user=created,
    )
    return created


async def provision_admin_user(
    *,
    client: LocalNextClient,
    next_dir: Path,
    next_env: dict[str, str],
    identity: dict[str, str],
    password: str,
) -> dict[str, Any]:
    """Provision a reserved bot user, then grant the real local manager role."""

    if not password:
        raise SafetyViolation("The E2E admin password must be generated in memory")
    await provision_test_user(
        client=client,
        next_dir=next_dir,
        next_env=next_env,
        identity=identity,
        password=password,
    )
    tsx_binary = next_dir / "node_modules" / ".bin" / "tsx"
    node_binary = shutil.which("node")
    promoter = Path(__file__).resolve().parent / "provision_admin.ts"
    if not node_binary or not tsx_binary.is_file() or not promoter.is_file():
        raise RigInfrastructureError("Reserved admin provisioning helpers are unavailable")
    child_env = dict(next_env)
    child_env.update(
        {
            "E2E_ADMIN_NAME": identity["name"],
            "E2E_ADMIN_EMAIL": identity["email"],
            "E2E_ADMIN_QQ_ID": identity["platform_id"],
        }
    )
    completed = await asyncio.to_thread(
        subprocess.run,
        [node_binary, "--import", "tsx", str(promoter)],
        cwd=next_dir,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        raise RigInfrastructureError(
            "Reserved local admin provisioning failed:\n"
            + completed.stdout[-2000:]
            + completed.stderr[-2000:]
        )
    user = await client.find_user(identity["platform_id"])
    if user is None:
        raise RigInfrastructureError("Reserved local admin binding disappeared after promotion")
    validate_admin_identity(
        platform_id=identity["platform_id"],
        expected_name=identity["name"],
        expected_email=identity["email"],
        user=user,
    )
    return user


class E2EBotHarness:
    """Fake only OneBot input/output while invoking the real QQ handler."""

    def __init__(
        self,
        *,
        openai_chat: Any,
        recorder: ArtifactRecorder,
        state_dir: Path,
        message_timeout: float,
    ) -> None:
        from nonebot.adapters.onebot.v11 import Bot as QQBot

        harness = self

        class BoundaryBot(QQBot):
            async def send(self, event: Any, message: Any, **_kwargs: Any) -> Any:
                extract = getattr(message, "extract_plain_text", None)
                text = str(extract() if callable(extract) else message).strip()
                platform_id = str(getattr(event, "user_id", ""))
                message_id = harness._next_bot_message_id
                harness._next_bot_message_id += 1
                harness._sent_messages[message_id] = {
                    "message_id": message_id,
                    "user_id": int(self.self_id),
                    "sender": {
                        "user_id": int(self.self_id),
                        "nickname": "喵喵",
                        "card": "",
                    },
                    "message": message,
                    "raw_message": text,
                }
                harness.last_reply_message_id = message_id
                harness.recorder.record_message(
                    direction="reply",
                    text=text,
                    platform_id=platform_id,
                    message_id=message_id,
                )
                harness.replies.append(text)
                harness.reply_event.set()
                return {"message_id": message_id}

            async def get_msg(self, *, message_id: int) -> dict[str, Any]:
                payload = harness._sent_messages.get(int(message_id))
                if payload is None:
                    raise RigInfrastructureError(
                        f"Unknown synthetic OneBot message id {message_id}"
                    )
                return payload

        self.openai_chat = openai_chat
        self.recorder = recorder
        self.message_timeout = message_timeout
        self._sent_messages: dict[int, dict[str, Any]] = {}
        self._next_bot_message_id = 1_500_000_000
        self.last_reply_message_id: int | None = None
        self.bot = BoundaryBot(adapter=None, self_id="99999999999999999999999999999999")
        self.replies: list[str] = []
        self.reply_event = asyncio.Event()
        self._current_event: ContextVar[Any] = ContextVar("e2e_current_event", default=None)
        self._original_finish = openai_chat.ai_chat.finish
        self._tool_patches: list[tuple[Any, str, Any]] = []
        self._assert_isolated_state(state_dir)
        self._install_boundary_finish()
        self._install_tool_tracing()

    def _assert_isolated_state(self, state_dir: Path) -> None:
        from keytao_bot.utils import draft_mutation_store

        expected_paths = {
            "history": (state_dir / "history.db").resolve(),
            "memory": (state_dir / "memory.db").resolve(),
            "draft mutations": (state_dir / "draft-mutation-claims.db").resolve(),
        }
        actual_paths = {
            "history": Path(self.openai_chat.history_store.db_path).resolve(),
            "memory": Path(self.openai_chat.memory_store.db_path).resolve(),
            "draft mutations": Path(draft_mutation_store._DEFAULT_STORE.db_path).resolve()
            if draft_mutation_store._DEFAULT_STORE is not None
            else None,
        }
        if actual_paths != expected_paths:
            raise SafetyViolation(
                f"Conversation state is not isolated to the run artifact: {actual_paths}"
            )

    def _install_boundary_finish(self) -> None:
        harness = self

        async def finish(message: Any = None, **_kwargs: Any) -> None:
            event = harness._current_event.get()
            if event is None:
                raise RigInfrastructureError("Matcher.finish was called outside an E2E message")
            await harness.bot.send(event=event, message=message or "")

        self.openai_chat.ai_chat.finish = finish

    def _set_tool_patch(self, target: Any, name: str, replacement: Any) -> None:
        self._tool_patches.append((target, name, getattr(target, name)))
        setattr(target, name, replacement)

    def _install_tool_tracing(self) -> None:
        from keytao_bot.harness.tools import ToolExecutor

        recorder = self.recorder
        original_call = ToolExecutor.call
        original_replay = ToolExecutor.replay_shift_plan

        async def call(
            executor: Any,
            tool_name: str,
            arguments: dict[str, Any],
            context: Any,
        ) -> str:
            started = time.monotonic()
            try:
                result = await original_call(executor, tool_name, arguments, context)
            except BaseException as error:
                recorder.record_tool(
                    phase="model-dispatch",
                    name=tool_name,
                    arguments=dict(arguments),
                    result={"raised": type(error).__name__, "message": str(error)},
                    elapsed_seconds=time.monotonic() - started,
                )
                raise
            recorder.record_tool(
                phase="model-dispatch",
                name=tool_name,
                arguments=dict(arguments),
                result=result,
                elapsed_seconds=time.monotonic() - started,
            )
            return result

        async def replay(
            executor: Any,
            authorization_tool_name: str,
            authorization_arguments: dict[str, Any],
            binding: dict[str, Any],
            context: Any,
        ) -> Any:
            started = time.monotonic()
            try:
                result = await original_replay(
                    executor,
                    authorization_tool_name,
                    authorization_arguments,
                    binding,
                    context,
                )
            except BaseException as error:
                recorder.record_tool(
                    phase="server-ticket-replay",
                    name=authorization_tool_name,
                    arguments={**authorization_arguments, **binding},
                    result={"raised": type(error).__name__, "message": str(error)},
                    elapsed_seconds=time.monotonic() - started,
                )
                raise
            recorder.record_tool(
                phase="server-ticket-replay",
                name=authorization_tool_name,
                arguments={**authorization_arguments, **binding},
                result=result[0] if isinstance(result, tuple) else result,
                elapsed_seconds=time.monotonic() - started,
            )
            return result

        self._set_tool_patch(ToolExecutor, "call", call)
        self._set_tool_patch(ToolExecutor, "replay_shift_plan", replay)

    async def reset_conversation(self, *, platform_id: str) -> None:
        from keytao_bot.harness.conversation import ConversationAddress
        from keytao_bot.utils.memory_store import ChatMemoryContext

        address = ConversationAddress.private("qq", platform_id)
        memory_context = ChatMemoryContext(
            platform="qq",
            user_id=platform_id,
            space_type="private",
            space_id=platform_id,
        )
        await self.openai_chat._clear_conversation_state(address, memory_context)
        group_id = str(self._group_id(platform_id))
        group_address = ConversationAddress.group("qq", group_id, platform_id)
        group_context = ChatMemoryContext(
            platform="qq",
            user_id=platform_id,
            space_type="group",
            space_id=group_id,
        )
        await self.openai_chat._clear_conversation_state(
            group_address,
            group_context,
        )

    @staticmethod
    def _group_id(platform_id: str) -> int:
        return int(str(platform_id)[-15:]) + 1

    async def send(self, *, platform_id: str, sender_name: str, text: str) -> str:
        from nonebot.adapters.onebot.v11 import Message
        from nonebot.adapters.onebot.v11.event import PrivateMessageEvent

        message = Message(text)
        event = PrivateMessageEvent(
            time=int(time.time()),
            self_id=int(self.bot.self_id),
            post_type="message",
            sub_type="friend",
            user_id=int(platform_id),
            message_type="private",
            message_id=int(time.time_ns() % 2_000_000_000),
            message=message,
            original_message=message,
            raw_message=text,
            font=0,
            sender={"user_id": int(platform_id), "nickname": sender_name, "card": ""},
            to_me=True,
        )
        if not await self.openai_chat.should_handle(self.bot, event):
            raise RigInfrastructureError("The real QQ rule rejected the synthetic private message")
        self.recorder.record_message(
            direction="input",
            text=text,
            platform_id=platform_id,
        )
        reply_index = len(self.replies)
        self.reply_event.clear()
        token = self._current_event.set(event)
        try:
            await asyncio.wait_for(
                self.openai_chat.handle_ai_chat(self.bot, event),
                timeout=self.message_timeout,
            )
            if len(self.replies) == reply_index:
                await asyncio.wait_for(self.reply_event.wait(), timeout=self.message_timeout)
        finally:
            self._current_event.reset(token)
        if len(self.replies) == reply_index:
            raise RigInfrastructureError("The real QQ handler completed without a reply")
        return self.replies[-1]

    async def send_group(
        self,
        *,
        platform_id: str,
        sender_name: str,
        text: str,
        to_me: bool,
        reply_message_id: int | None = None,
    ) -> str:
        from nonebot.adapters.onebot.v11 import Message, MessageSegment
        from nonebot.adapters.onebot.v11.event import GroupMessageEvent

        if (
            reply_message_id is not None
            and int(reply_message_id) not in self._sent_messages
        ):
            raise RigInfrastructureError(
                f"Cannot quote unknown synthetic bot message {reply_message_id}"
            )
        message = (
            MessageSegment.reply(int(reply_message_id)) + text
            if reply_message_id is not None
            else Message(text)
        )
        event = GroupMessageEvent(
            time=int(time.time()),
            self_id=int(self.bot.self_id),
            post_type="message",
            sub_type="normal",
            user_id=int(platform_id),
            message_type="group",
            group_id=self._group_id(platform_id),
            message_id=int(time.time_ns() % 2_000_000_000),
            message=message,
            original_message=message,
            raw_message=text,
            font=0,
            sender={"user_id": int(platform_id), "nickname": sender_name, "card": ""},
            to_me=to_me,
        )
        if not await self.openai_chat.should_handle(self.bot, event):
            raise RigInfrastructureError(
                "The real QQ rule rejected the synthetic addressed group message"
            )
        self.recorder.record_message(
            direction="input",
            text=text,
            platform_id=platform_id,
            message_id=event.message_id,
            reply_message_id=reply_message_id,
        )
        reply_index = len(self.replies)
        self.reply_event.clear()
        token = self._current_event.set(event)
        try:
            await asyncio.wait_for(
                self.openai_chat.handle_ai_chat(self.bot, event),
                timeout=self.message_timeout,
            )
            if len(self.replies) == reply_index:
                await asyncio.wait_for(
                    self.reply_event.wait(),
                    timeout=self.message_timeout,
                )
        finally:
            self._current_event.reset(token)
        if len(self.replies) == reply_index:
            raise RigInfrastructureError(
                "The real QQ group handler completed without a reply"
            )
        return self.replies[-1]

    async def send_group_reply(
        self,
        *,
        platform_id: str,
        sender_name: str,
        text: str,
        reply_message_id: int,
        to_me: bool,
    ) -> str:
        """Send a real OneBot reply segment quoting one recorded bot message."""
        return await self.send_group(
            platform_id=platform_id,
            sender_name=sender_name,
            text=text,
            to_me=to_me,
            reply_message_id=reply_message_id,
        )

    async def close(self) -> None:
        await self.openai_chat._shutdown_background_draft_tasks()
        self.openai_chat.ai_chat.finish = self._original_finish
        while self._tool_patches:
            target, name, original = self._tool_patches.pop()
            setattr(target, name, original)


def assert_runtime_configuration(openai_chat: Any, *, keytao_base: str, llm: dict[str, str]) -> None:
    from keytao_bot.utils import http_client

    actual_keytao = validate_keytao_base(http_client.get_keytao_url())
    if actual_keytao != validate_keytao_base(keytao_base):
        raise SafetyViolation(f"Runtime KEYTAO_API_BASE drifted to {actual_keytao}")
    actual_llm = str(openai_chat.OPENAI_BASE_URL).rstrip("/") + "/"
    expected_llm = str(llm["base_url"]).rstrip("/") + "/"
    if actual_llm != expected_llm:
        raise SafetyViolation("Runtime LLM base URL does not match the preflighted endpoint")
    if str(openai_chat.OPENAI_MODEL) != llm["model"]:
        raise SafetyViolation("Runtime LLM model does not match E2E configuration")
    if not openai_chat.OPENAI_API_KEY:
        raise SafetyViolation("Runtime real LLM API key is missing")
