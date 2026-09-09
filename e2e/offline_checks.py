"""Run S60 checks in separate, credential-free Python processes with no network."""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path(__file__).resolve()
LOG_ROOT = Path("/tmp/keytao-s60-offline")
CHECKS = (
    ["test_state_machine.py"],
    ["test_memory_safety.py"],
    ["test_security_fixes.py"],
    ["test_review_gate.py"],
    ["test_llm_policy.py"],
    ["test_word_discovery.py"],
    ["-m", "e2e.test_safety"],
)


def install_guards() -> None:
    """Allow fake clients and new temporary fixtures, never real credentials/I/O."""
    temporary_root = Path(tempfile.gettempdir()).resolve()
    created_fixtures: set[Path] = set()

    def sensitive(path: Path) -> bool:
        name = path.name.lower()
        return (
            name == ".env" or name.startswith(".env.") or name.endswith(".env")
            or name in {".e2e_key", ".netrc", "credentials", "credentials.json",
                        "secrets.json", "secrets.yaml", "secrets.yml"}
            or name.startswith("id_rsa") or name.startswith("id_ed25519")
            or name.endswith(".key")
            or any(part in {".ssh", ".aws", ".azure", "gcloud"} for part in path.parts)
        )

    def audit(event: str, args: tuple) -> None:
        if event in {"socket.connect", "socket.bind", "socket.sendto",
                     "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"}:
            raise PermissionError("S60 offline guard: network disabled")
        if event in {"os.system", "os.posix_spawn", "os.exec", "pty.spawn"}:
            raise PermissionError("S60 offline guard: external processes disabled")
        if event == "subprocess.Popen":
            command = args[1]
            # Fixture entrypoints may only recurse through this same guarded launcher.
            if not (isinstance(command, (list, tuple)) and len(command) >= 2
                    and Path(command[0]).resolve() == Path(sys.executable).resolve()
                    and str(LAUNCHER) in command[:4]):
                raise PermissionError("S60 offline guard: unguarded child disabled")
        if event == "open" and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if not sensitive(path):
                return
            flags = args[2]
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if (writing and path.is_relative_to(temporary_root)
                    and not path.exists()):
                created_fixtures.add(path)
            if path not in created_fixtures:
                raise PermissionError("S60 offline guard: real credential file disabled")

    sys.addaudithook(audit)

    import dotenv
    import dotenv.main

    original_values = dotenv.main.dotenv_values
    original_load = dotenv.main.load_dotenv

    def fixture_values(dotenv_path=None, stream=None, **kwargs):
        if stream is not None:
            return original_values(dotenv_path=dotenv_path, stream=stream, **kwargs)
        if dotenv_path and Path(dotenv_path).resolve() in created_fixtures:
            return original_values(dotenv_path=dotenv_path, **kwargs)
        return {}

    def fixture_load(dotenv_path=None, stream=None, **kwargs):
        if stream is not None or (dotenv_path and Path(dotenv_path).resolve() in created_fixtures):
            return original_load(dotenv_path=dotenv_path, stream=stream, **kwargs)
        return False

    dotenv.dotenv_values = dotenv.main.dotenv_values = fixture_values
    dotenv.load_dotenv = dotenv.main.load_dotenv = fixture_load


def self_test() -> None:
    import socket
    import unittest
    from unittest.mock import patch

    import dotenv

    class GuardTests(unittest.TestCase):
        def test_no_inherited_credentials(self):
            self.assertFalse(any("KEY" in key or "TOKEN" in key or "SECRET" in key
                                 for key in os.environ))

        def test_network_and_real_secret_paths_are_blocked(self):
            with self.assertRaises(PermissionError):
                socket.getaddrinfo("localhost", 80)
            with self.assertRaises(PermissionError):
                (ROOT / ".e2e_key").read_text()
            self.assertEqual(dotenv.dotenv_values(ROOT / ".env"), {})

        def test_fixture_dotenv_and_mocked_values_are_allowed(self):
            with tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory) / ".env"
                fixture.write_text("FIXTURE_VALUE=fake\n")
                self.assertEqual(dotenv.dotenv_values(fixture), {"FIXTURE_VALUE": "fake"})
            with patch("dotenv.dotenv_values", return_value={"OPENAI_API_KEY": "fake"}):
                self.assertEqual(dotenv.dotenv_values(ROOT / ".env"), {"OPENAI_API_KEY": "fake"})

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GuardTests)
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)


def child(arguments: list[str]) -> None:
    install_guards()
    sys.path.insert(0, str(ROOT))
    if arguments == ["--self-test"]:
        self_test()
    if arguments[:1] == ["-m"]:
        if len(arguments) < 2 or arguments[1] not in {"unittest", "e2e.test_safety", "e2e.s60"}:
            raise SystemExit("Only unittest and the S60/safety fixture modules are allowed")
        sys.argv = arguments[1:]
        runpy.run_module(arguments[1], run_name="__main__", alter_sys=True)
    elif arguments in CHECKS:
        sys.argv = arguments
        runpy.run_path(str(ROOT / arguments[0]), run_name="__main__")
    else:
        raise SystemExit("Only prescribed offline scripts or -m fixture modules are allowed")


def main(arguments: list[str]) -> int:
    if arguments[:1] == ["--child"]:
        child(arguments[1:])
        return 0
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    checks = [arguments] if arguments else CHECKS
    # Construct a fresh environment; no inherited key, token, proxy, or provider config survives.
    child_env = {
        "PATH": f"{ROOT / '.venv/bin'}:/usr/bin:/bin:/usr/sbin:/sbin",
        "LANG": "en_US.UTF-8",
        "TZ": "UTC",
        "TMPDIR": "/tmp",
        "PYTHONUNBUFFERED": "1",
    }
    results = []
    for check in checks:
        label = "-".join(check).replace("/", "_").replace(" ", "_").strip("-")
        log = LOG_ROOT / f"{label}.log"
        command = [str(ROOT / ".venv/bin/python"), *check]
        started = time.monotonic()
        print(f"Running: {' '.join(command)}; log: {log}", flush=True)
        with log.open("w") as output:
            try:
                completed = subprocess.run(
                    [str(ROOT / ".venv/bin/python"), "-I", "-u", str(LAUNCHER), "--child", *check],
                    cwd=ROOT, env=child_env, stdout=output, stderr=subprocess.STDOUT,
                    timeout=900, check=False,
                )
                code = completed.returncode
            except subprocess.TimeoutExpired:
                code = 124
        row = {"command": command, "exit": code,
               "seconds": round(time.monotonic() - started, 3), "log": str(log)}
        results.append(row)
        print(json.dumps(row), flush=True)
    result_path = LOG_ROOT / ("results.json" if not arguments else f"{label}-results.json")
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    return 0 if all(row["exit"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
