"""Offline checks proving the E2E production guards fail before dispatch."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from .safety import (
    NetworkAllowlist,
    SafetyViolation,
    validate_keytao_base,
    validate_llm_base,
    validate_next_database_url,
    validate_test_binding,
)


class SafetyRailTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()

