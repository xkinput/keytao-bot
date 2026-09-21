"""S63 fixture-only scenario, isolated from the paid-provider E2E runner."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

MODULES = ('test_s63_bcc', 'test_s63_bcc_ingest', 'test_s63_bcc_delivery')
ROOT = Path(__file__).resolve().parents[1]


def offline_guard(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
        raise RuntimeError('S63 fixtures prohibit network access')
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).absolute()
        if path.name == '.e2e_key' or (path.name.startswith('.env') and ROOT in path.parents):
            raise RuntimeError('S63 fixtures prohibit production credential files')


def main():
    env = {key: os.environ[key] for key in ('PATH', 'HOME', 'TMPDIR', 'LANG') if key in os.environ}
    os.environ.clear()
    os.environ.update(env)
    sys.addaudithook(offline_guard)
    if len(sys.argv) == 3 and sys.argv[1] == '--child' and sys.argv[2] in MODULES:
        suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[2])
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if len(sys.argv) != 1:
        raise SystemExit('Use python -m e2e.s63 without provider arguments')
    results = []
    for module in MODULES:
        result = subprocess.run([sys.executable, '-m', 'e2e.s63', '--child', module],
                                cwd=ROOT, env=env, timeout=120, check=False)
        results.append({'module': module, 'exit': result.returncode})
    summary = {'scenario': 'S63', 'mode': 'fixtures/fake-tools',
               'paidModelCalls': 0, 'realProviderCalls': 0,
               'passed': all(row['exit'] == 0 for row in results), 'checks': results}
    print(json.dumps(summary), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
