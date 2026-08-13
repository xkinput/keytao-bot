"""Unittest entry point for the repository's function-style state-machine suite."""

from __future__ import annotations

import unittest

import test_state_machine as legacy_state_machine


def _run_legacy_state_machine_suite() -> None:
    legacy_state_machine.passed = 0
    legacy_state_machine.failed = 0
    tests = [
        value
        for name, value in vars(legacy_state_machine).items()
        if name.startswith("test_")
        and callable(value)
        and getattr(value, "__module__", None) == legacy_state_machine.__name__
    ]
    if not tests:
        raise AssertionError("No state-machine tests were discovered")
    for test in tests:
        test()
    total = legacy_state_machine.passed + legacy_state_machine.failed
    print(
        "State machine checks: "
        f"{legacy_state_machine.passed}/{total} passed, "
        f"{legacy_state_machine.failed} failed"
    )
    if legacy_state_machine.failed:
        raise AssertionError(
            f"{legacy_state_machine.failed} state-machine checks failed"
        )


def load_tests(
    _loader: unittest.TestLoader,
    _tests: unittest.TestSuite,
    _pattern: str | None,
) -> unittest.TestSuite:
    return unittest.TestSuite(
        [unittest.FunctionTestCase(_run_legacy_state_machine_suite)]
    )
