"""Offline tests for dedicated E2E credentials using synthetic dotenv files."""

from __future__ import annotations

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import run
from .safety import SafetyViolation


class E2EKeyGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "bot"
        self.cwd = Path(temporary.name) / "cwd"
        self.next_dir = Path(temporary.name) / "next"
        for directory in (self.root, self.cwd, self.next_dir):
            directory.mkdir()
        (self.root / ".env").write_text(
            "OPENAI_API_KEY=synthetic-bot-key\n"
            "OPENAI_BASE_URL=https://llm.example.invalid/v1\n"
            "OPENAI_MODEL=glm-5.3\n",
            encoding="utf-8",
        )
        (self.next_dir / ".env").write_text(
            "DATABASE_URL=postgresql://local:local@localhost:5432/keytao\n"
            "BOT_API_TOKEN=synthetic-local-token\n",
            encoding="utf-8",
        )
        self.args = argparse.Namespace(next_dir=self.next_dir, port=3100)
        self.enterContext(patch.object(run, "REPO_ROOT", self.root))
        self.enterContext(patch.object(Path, "cwd", return_value=self.cwd))
        self.enterContext(
            patch.dict(os.environ, {"E2E_OPENAI_API_KEY": "synthetic-e2e-key"}, clear=True)
        )

    def test_missing_key_rejects_before_reading_dotenv(self) -> None:
        os.environ.pop("E2E_OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "synthetic-bot-key"
        with patch.object(run, "dotenv_values") as read_dotenv:
            with self.assertRaisesRegex(SafetyViolation, "E2E_OPENAI_API_KEY is required"):
                run.load_configuration(self.args)
        read_dotenv.assert_not_called()

    def test_process_bot_key_is_rejected_without_echoing_key(self) -> None:
        for name in ("OPENAI_API_KEY", "openai_api_key"):
            with self.subTest(name=name), patch.dict(
                os.environ, {name: " synthetic-e2e-key "}
            ):
                with self.assertRaisesRegex(SafetyViolation, "matches.*OPENAI_API_KEY") as error:
                    run.load_configuration(self.args)
                self.assertNotIn("synthetic-e2e-key", str(error.exception))

    def test_all_visible_bot_dotenv_variants_are_rejected(self) -> None:
        for directory in (self.root, self.cwd):
            for name in (".env", ".env.prod", ".env.dev", ".env.local"):
                with self.subTest(directory=directory.name, name=name):
                    path = directory / name
                    original = path.read_text() if path.exists() else None
                    path.write_text("openai_api_key='synthetic-e2e-key'\n", encoding="utf-8")
                    try:
                        with self.assertRaisesRegex(SafetyViolation, "matches.*OPENAI_API_KEY"):
                            run.load_configuration(self.args)
                    finally:
                        if original is None:
                            path.unlink()
                        else:
                            path.write_text(original, encoding="utf-8")

    def test_interpolated_bot_key_is_rejected(self) -> None:
        (self.cwd / ".env.prod").write_text(
            "KEY_ALIAS=synthetic-e2e-key\nOPENAI_API_KEY=${KEY_ALIAS}\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(SafetyViolation, "matches.*OPENAI_API_KEY"):
            run.load_configuration(self.args)

    def test_next_dotenv_variants_are_rejected(self) -> None:
        for name in (".env", ".env.prod", ".env.dev", ".env.local"):
            with self.subTest(name=name):
                path = self.next_dir / name
                original = path.read_text() if path.exists() else None
                path.write_text("OPENAI_API_KEY=synthetic-e2e-key\n", encoding="utf-8")
                try:
                    with self.assertRaisesRegex(SafetyViolation, "matches.*OPENAI_API_KEY"):
                        run.load_configuration(self.args)
                finally:
                    if original is None:
                        path.unlink()
                    else:
                        path.write_text(original, encoding="utf-8")

    def test_cross_file_alias_uses_process_environment_not_previous_file(self) -> None:
        from nonebot.config import Config, DotEnvSettingsSource

        base = self.cwd / ".env"
        variant = self.cwd / ".env.prod"
        base.write_text("KEY_ALIAS=synthetic-e2e-key\n", encoding="utf-8")
        variant.write_text("OPENAI_API_KEY=${KEY_ALIAS}\n", encoding="utf-8")
        source = DotEnvSettingsSource(
            Config, env_file=(base, variant), env_file_encoding="utf-8"
        )
        # NoneBot parses each file separately, then merges the parsed values.
        self.assertEqual(source._read_env_files()["openai_api_key"], "")
        self.assertEqual(run.load_configuration(self.args)["llm"]["api_key"], "synthetic-e2e-key")
        os.environ["KEY_ALIAS"] = "synthetic-e2e-key"
        self.assertEqual(source._read_env_files()["openai_api_key"], "synthetic-e2e-key")
        with self.assertRaisesRegex(SafetyViolation, "matches.*OPENAI_API_KEY"):
            run.load_configuration(self.args)

    def test_unreadable_bot_dotenv_fails_closed_without_error_details(self) -> None:
        with patch.object(run, "dotenv_values", side_effect=PermissionError("secret-detail")):
            with self.assertRaisesRegex(SafetyViolation, "Cannot verify.*OPENAI_API_KEY") as error:
                run.load_configuration(self.args)
        self.assertNotIn("secret-detail", str(error.exception))

    def test_distinct_e2e_key_keeps_bot_model_configuration(self) -> None:
        os.environ["OPENAI_API_KEY"] = "synthetic-process-bot-key"
        config = run.load_configuration(self.args)
        self.assertEqual(
            config["llm"],
            {
                "api_key": "synthetic-e2e-key",
                "base_url": "https://llm.example.invalid/v1/",
                "model": "glm-5.3",
            },
        )


if __name__ == "__main__":
    unittest.main()
