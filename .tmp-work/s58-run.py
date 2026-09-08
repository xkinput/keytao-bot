import datetime
import json
import os
import re
import signal
from pathlib import Path
import subprocess
import sys

root = Path('/Users/rea/code/keytao-org/keytao-bot')
checks = Path('/private/tmp/keytao-s58-checks')
checks.mkdir(exist_ok=True)
label = sys.argv[1]
args = sys.argv[2:]
env = os.environ.copy()
env.update({
    'HTTPS_PROXY': 'http://127.0.0.1:7890',
    'E2E_ARTIFACT_RETENTION': '1000',
    'E2E_OPENAI_API_KEY': (root / '.e2e_key').read_text().strip(),
    'E2E_OPENAI_BASE_URL': 'https://api.deepseek.com',
    'E2E_OPENAI_MODEL': 'deepseek-v4-flash',
})
command = [str(root / '.venv/bin/python'), '-u', '-m', 'e2e.run', *args]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
with (checks / (label + '.log')).open('w') as log:
    result = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    failed_scenario = ''
    for line in result.stdout:
        log.write(line)
        log.flush()
        if not args and not failed_scenario and re.search(r'S\d+ attempt \d+ failed:', line):
            failed_scenario = line.strip()
            result.send_signal(signal.SIGINT)
    result.wait()
metadata = {'command': command, 'start': started, 'end': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'exit': result.returncode, 'firstFailure': failed_scenario}
(checks / (label + '-exit.json')).write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata))
print('\n'.join((checks / (label + '.log')).read_text().splitlines()[-12:]))
sys.exit(result.returncode)
