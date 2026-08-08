"""Offline checks proving the E2E production guards fail before dispatch."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, call, patch

import httpx

from .recording import ArtifactRecorder, _redact_sensitive
from .run import S9_ZDIC_WARMUP_BACKOFF_SECONDS, ensure_s9_fixture
from .runtime import LocalNextClient, RigInfrastructureError
from .safety import (
    NetworkAllowlist,
    SafetyViolation,
    validate_admin_identity,
    validate_keytao_base,
    validate_llm_base,
    validate_next_database_url,
    validate_test_binding,
)
from .zdic_seed import seed_s9_zdic_cache


class SafetyRailTests(unittest.IsolatedAsyncioTestCase):
    def test_artifacts_redact_admin_credentials(self) -> None:
        payload = _redact_sensitive(
            {
                "name": "reserved-admin",
                "password": "secret-password",
                "nested": {"token": "signed-jwt"},
            }
        )
        self.assertEqual(payload["name"], "reserved-admin")
        self.assertEqual(payload["password"], "[REDACTED]")
        self.assertEqual(payload["nested"]["token"], "[REDACTED]")

    def test_keytao_base_requires_localhost(self) -> None:
        self.assertEqual(
            validate_keytao_base("http://localhost:3100"),
            "http://localhost:3100",
        )
        with self.assertRaises(SafetyViolation):
            validate_keytao_base("https://keytao.vercel.app")
        with self.assertRaises(SafetyViolation):
            validate_keytao_base("http://192.0.2.2:3100")

    def test_next_database_must_resolve_local(self) -> None:
        result = validate_next_database_url(
            "postgresql://dev:dev@localhost:5432/keytao"
        )
        self.assertEqual(result["host"], "localhost")
        with self.assertRaises(SafetyViolation):
            validate_next_database_url(
                "postgresql://user:pass@db.production.example:5432/keytao"
            )

    @patch("e2e.zdic_seed.subprocess.run")
    def test_zdic_seeder_refuses_non_localhost_database(self, run_mock) -> None:
        with self.assertRaises(SafetyViolation):
            seed_s9_zdic_cache(
                "postgresql://user:pass@db.production.example:5432/keytao",
                next_dir=Path("/not-inspected-before-database-validation"),
            )
        run_mock.assert_not_called()

    async def test_s9_zdic_preflight_only_asserts_seeded_reality_on_final_probe(
        self,
    ) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        unavailable = {
            "codes": ["wrong"],
            "pronunciationSource": "zdic-unavailable",
            "standardPronunciationStatus": "unavailable",
            "semanticPronunciationNeeded": True,
            "chars": [
                {"char": "射", "pronunciationLookupStatus": "unavailable"},
                {"char": "覆", "pronunciationLookupStatus": "unavailable"},
            ],
        }
        seeded = {
            "codes": ["eefj"],
            "altCodes": ["eefju", "eefjuv"],
            "pronunciationSource": "pinyin-pro-context",
            "standardPronunciationStatus": "absent",
            "semanticPronunciationNeeded": False,
            "chars": [
                {
                    "char": "射",
                    "pinyin": "shè",
                    "pinyins": ["shè"],
                    "pronunciationLookupStatus": "found",
                },
                {
                    "char": "覆",
                    "pinyin": "fù",
                    "pinyins": ["fù"],
                    "pronunciationLookupStatus": "found",
                },
            ],
        }
        occupant = {
            "word": "慑服",
            "code": "eefj",
            "type": "Phrase",
            "user": {"name": "existing-local-user"},
        }
        client.phrases_by_word = AsyncMock(return_value=[])
        client.phrases_by_code = AsyncMock(side_effect=[[occupant], [], []])
        client.encode = AsyncMock(side_effect=[unavailable, unavailable, unavailable, seeded])

        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = ArtifactRecorder(Path(temp_dir) / "artifacts")
            with (
                recorder.scope("S9", 1),
                patch("e2e.run.asyncio.sleep", new_callable=AsyncMock) as sleep_mock,
                patch("builtins.print") as print_mock,
            ):
                result = await ensure_s9_fixture(
                    client=client,
                    seed_identity={"platform_id": "9" * 32},
                    admin_token="admin",
                    recorder=recorder,
                )

            artifact = json.loads(
                (recorder.artifact_dir / "S9-zdic-warmup-attempt-1.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(result["seededCharactersFound"])
        self.assertEqual(client.encode.await_count, 4)
        self.assertEqual(
            sleep_mock.await_args_list,
            [call(delay) for delay in S9_ZDIC_WARMUP_BACKOFF_SECONDS],
        )
        self.assertEqual(print_mock.call_count, 4)
        self.assertEqual(artifact["finalAssertionAttempt"], 4)
        self.assertEqual(artifact["finalAssertionResult"], "passed")
        self.assertEqual(
            [attempt["pronunciationSource"] for attempt in artifact["attempts"]],
            ["zdic-unavailable"] * 3 + ["pinyin-pro-context"],
        )

    def test_llm_endpoint_can_never_be_keytao_production(self) -> None:
        self.assertEqual(
            validate_llm_base("https://llm.example.com/v1"),
            "https://llm.example.com/v1/",
        )
        with self.assertRaises(SafetyViolation):
            validate_llm_base("https://keytao.vercel.app/v1")

    def test_binding_requires_reserved_metadata_and_non_qq_numeric_shape(self) -> None:
        user = {
            "name": "keytao-e2e-llm-rig-run-s1",
            "email": "keytao-e2e-llm-rig-run-s1@example.invalid",
            "roles": [{"value": "R:NORMAL"}, {"value": "R:BOT"}],
        }
        validate_test_binding(
            platform_id="9" * 32,
            expected_name=user["name"],
            expected_email=user["email"],
            user=user,
        )
        with self.assertRaises(SafetyViolation):
            validate_test_binding(
                platform_id="123456789",
                expected_name="ordinary-user",
                expected_email="person@example.com",
                user={
                    "name": "ordinary-user",
                    "email": "person@example.com",
                    "roles": [{"value": "R:NORMAL"}],
                },
            )

    def test_admin_requires_reserved_binding_and_real_admin_role(self) -> None:
        user = {
            "name": "keytao-e2e-llm-rig-run-admin",
            "email": "keytao-e2e-llm-rig-run-admin@example.invalid",
            "roles": [
                {"value": "R:NORMAL"},
                {"value": "R:BOT"},
                {"value": "R:MANAGER"},
            ],
        }
        validate_admin_identity(
            platform_id="9" * 32,
            expected_name=user["name"],
            expected_email=user["email"],
            user=user,
        )
        with self.assertRaises(SafetyViolation):
            validate_admin_identity(
                platform_id="9" * 32,
                expected_name=user["name"],
                expected_email=user["email"],
                user={**user, "roles": user["roles"][:2]},
            )
        with self.assertRaises(SafetyViolation):
            validate_admin_identity(
                platform_id="123456789",
                expected_name="admin",
                expected_email="admin@example.com",
                user={
                    "name": "admin",
                    "email": "admin@example.com",
                    "roles": [{"value": "R:ROOT"}],
                },
            )

    async def test_http_guard_rejects_production_before_transport(self) -> None:
        called = False

        async def transport_handler(_request: httpx.Request) -> httpx.Response:
            nonlocal called
            called = True
            return httpx.Response(200, json={"unexpected": True})

        with patch.object(
            NetworkAllowlist,
            "_resolve_llm_ips",
            return_value=frozenset({"203.0.113.10"}),
        ):
            guard = NetworkAllowlist(llm_base_url="https://llm.example.com/v1")
        guard.install()
        try:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(transport_handler)
            ) as client:
                with self.assertRaises(SafetyViolation):
                    await client.get("https://keytao.vercel.app/api/phrases")
        finally:
            guard.restore()
        self.assertFalse(called)

    async def test_s8_repair_ignores_prefix_rows_and_removes_rig_rows(self) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        rig_user = {"name": "keytao-e2e-llm-rig-old-s8"}
        shifted = {
            "id": 11,
            "word": "赤溪",
            "code": "wkxkv",
            "type": "Phrase",
            "weight": 100,
            "user": rig_user,
        }
        occupied = {
            "id": 12,
            "word": "吃席",
            "code": "wkxk",
            "type": "Phrase",
            "weight": 100,
            "user": rig_user,
        }
        restored = {
            "id": 13,
            "word": "赤溪",
            "code": "wkxk",
            "type": "Phrase",
            "weight": 100,
            "user": rig_user,
        }
        client.phrases_by_word = AsyncMock(side_effect=[[shifted], [restored]])
        client.phrases_by_code = AsyncMock(
            side_effect=[
                [occupied, shifted],
                [shifted],
                [restored],
                [],
            ]
        )
        client.clean_draft = AsyncMock(return_value={"success": True})
        client.add_draft_items = AsyncMock(
            return_value={"batchId": "cleanup", "contentVersion": 3}
        )
        client.submit_batch = AsyncMock(
            return_value={"submitted": {"batch": {"status": "Submitted"}}}
        )
        client.approve_admin_batch = AsyncMock(
            return_value={"batch": {"status": "Approved"}}
        )

        result = await client.restore_s8_fixture(
            platform_id="9" * 32,
            admin_token="admin",
            chixi_next_code="wkxkv",
        )

        self.assertTrue(result["verified"])
        self.assertEqual(result["deletedFixtureRows"], 2)
        cleanup_items = client.add_draft_items.await_args.kwargs["items"]
        self.assertEqual(
            [(item["action"], item["word"], item["code"]) for item in cleanup_items],
            [
                ("Delete", "吃席", "wkxk"),
                ("Delete", "赤溪", "wkxkv"),
                ("Create", "赤溪", "wkxk"),
            ],
        )

    async def test_s8_repair_refuses_non_rig_rows(self) -> None:
        client = LocalNextClient(base_url="http://localhost:3100", bot_token="test")
        ordinary_user = {"name": "ordinary-user"}
        shifted = {
            "id": 21,
            "word": "赤溪",
            "code": "wkxkv",
            "type": "Phrase",
            "weight": 100,
            "user": ordinary_user,
        }
        occupied = {
            "id": 22,
            "word": "吃席",
            "code": "wkxk",
            "type": "Phrase",
            "weight": 100,
            "user": ordinary_user,
        }
        client.phrases_by_word = AsyncMock(return_value=[shifted])
        client.phrases_by_code = AsyncMock(side_effect=[[occupied, shifted], [shifted]])
        client.clean_draft = AsyncMock()
        client.add_draft_items = AsyncMock()

        with self.assertRaises(RigInfrastructureError):
            await client.restore_s8_fixture(
                platform_id="9" * 32,
                admin_token="admin",
                chixi_next_code="wkxkv",
            )

        client.clean_draft.assert_not_awaited()
        client.add_draft_items.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
