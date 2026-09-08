import datetime
import json
from pathlib import Path
import subprocess
import sys

checks = Path('/private/tmp/keytao-s58-checks')
checks.mkdir(exist_ok=True)
label = sys.argv[1]
commands = [
    ['.venv/bin/python', name] for name in (
        'test_state_machine.py', 'test_memory_safety.py', 'test_security_fixes.py',
        'test_review_gate.py', 'test_llm_policy.py', 'test_word_discovery.py',
    )
] + [['.venv/bin/python', '-m', 'e2e.test_safety']]
results = []
for command in commands:
    name = command[-1].replace('.py', '').replace('.', '-')
    log_path = checks / (name + '-' + label + '.log')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with log_path.open('w') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    result = {'command': command, 'log': str(log_path), 'exit': process.returncode, 'start': started, 'end': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    results.append(result)
    (checks / ('offline-' + label + '-exits.json')).write_text(json.dumps(results, indent=2))
    print(json.dumps(result), flush=True)
    print('\n'.join(log_path.read_text().splitlines()[-7:]), flush=True)
sys.exit(any(result['exit'] for result in results))
