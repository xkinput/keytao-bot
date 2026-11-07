"""S60 fixture/fake-model entrypoint; never import the real-provider rig."""

import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    launcher = root / "e2e/offline_checks.py"
    modules = (
        "tests.test_receipts_and_display.S60ScenarioTests",
        "tests.test_receipts_and_display.ReceiptRegressionTests",
        "tests.test_receipts_and_display.ReceiptDeltaTests",
        "tests.test_receipts_and_display.ReceiptFinalizerTests",
    )
    results = []
    for module in modules:
        result = subprocess.run([
            sys.executable, str(launcher), "-m", "unittest", module,
        ], cwd=root, timeout=120, check=False)
        results.append({"module": module, "exit": result.returncode})
    summary = {"scenario": "S60", "mode": "fixtures/fake-model",
               "paidModelCalls": 0, "realProviderCalls": 0,
               "passed": all(row["exit"] == 0 for row in results), "checks": results}
    print(json.dumps(summary), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
