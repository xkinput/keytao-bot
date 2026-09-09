"""S59 fixture/fake-model scenario. Never import or launch the provider rig."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    launcher = root / ".tmp-work/s59-offline/run_checks.py"
    modules = (
        "test_s59_commonness_route", "test_s59_commonness_evidence",
        "test_s59_tool_prompt", "test_s59_general_cap",
    )
    results = []
    for module in modules:
        result = subprocess.run([
            sys.executable, str(launcher), "-m", "unittest", module,
        ], cwd=root, timeout=120, check=False)
        results.append({"module": module, "exit": result.returncode})
    summary = {"scenario": "S59", "mode": "fixtures/fake-model",
               "paidModelCalls": 0, "realProviderCalls": 0,
               "passed": all(row["exit"] == 0 for row in results), "checks": results}
    print(json.dumps(summary), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
