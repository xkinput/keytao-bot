"""Adversarial checks for the S57 rig's evidence assertions, without network."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import AsyncMock
from urllib.parse import urlsplit

import test_state_machine as harness
from e2e.recording import ArtifactRecorder
from e2e.scenarios import ScenarioAssertionError, _assert_s56_advertised_reply_closure


class S57RigEvidenceReviewTests(unittest.TestCase):
    def test_model_filter_detects_actual_recorded_provider_transport_error(self):
        tree = ast.parse(Path("e2e/s57.py").read_text())
        scenario = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "scenario_s57")
        predicate = next(node for node in scenario.body if isinstance(node, ast.FunctionDef) and node.name == "model_calls")
        with tempfile.TemporaryDirectory() as directory:
            recorder = ArtifactRecorder(Path(directory) / "artifacts")
            with recorder.scope("S57", 1):
                recorder.record_http_error(
                    request=SimpleNamespace(
                        method="POST", url="https://api.deepseek.com/chat/completions",
                        content=b'{"model":"deepseek-v4-flash","messages":[]}',
                    ),
                    error=TimeoutError("fixture provider timeout"), elapsed_seconds=1.0,
                )
            namespace = {
                "ctx": SimpleNamespace(attempt_events=lambda: recorder.events_for("S57", 1)),
                "chat": SimpleNamespace(OPENAI_BASE_URL="https://api.deepseek.com"),
                "urlsplit": urlsplit,
                "Any": object,
            }
            exec(compile(ast.Module(body=[predicate], type_ignores=[]), "e2e/s57.py", "exec"), namespace)
            self.assertTrue(
                namespace["model_calls"](0),
                "A recorded provider request that timed out must disprove zero model calls",
            )

    def test_closure_rejects_new_force_advertisements_without_a_live_record(self):
        async def run():
            chat = harness.openai_chat_module
            address = harness.ConversationAddress.private("qq", "s57-rig-closure")
            for reply in ("回复「强制调序」即可执行。", "回复「照做」即可执行。"):
                with self.subTest(reply=reply), self.assertRaises(ScenarioAssertionError):
                    await _assert_s56_advertised_reply_closure(
                        reply, chat=chat, record=None, address=address,
                        read_draft=AsyncMock(side_effect=AssertionError("unexpected draft read")),
                    )
        asyncio.run(run())

    def test_closure_rejects_structural_options_without_a_live_record(self):
        async def run():
            with self.assertRaises(ScenarioAssertionError):
                await _assert_s56_advertised_reply_closure(
                    "请问是要强制调序，还是保留现顺序？",
                    chat=harness.openai_chat_module, record=None,
                    address=harness.ConversationAddress.private("qq", "s57-rig-options"),
                    read_draft=AsyncMock(side_effect=AssertionError("unexpected draft read")),
                )
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
