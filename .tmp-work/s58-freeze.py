import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path('/Users/rea/code/keytao-org/keytao-bot')
checks = Path('/private/tmp/keytao-s58-checks')
checks.mkdir(exist_ok=True)

def git(*args, directory=root):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()

paths = git('ls-files', '-co', '--exclude-standard').splitlines()
selected = sorted(set(
    path for path in paths
    if not path.startswith(('.tmp-work/', 'REPORT-', 'e2e/artifacts/', 'e2e/.runtime/'))
    and (path.endswith(('.py', '.md')) or path in ('pyproject.toml', 'uv.lock', '.env.example'))
    and (root / path).is_file()
))
data = {
    'head': git('rev-parse', 'HEAD'),
    'index': git('diff', '--cached', '--name-only'),
    'nextHead': git('rev-parse', 'HEAD', directory=root.parent/'keytao-next'),
    'nextStatus': git('status', '--short', directory=root.parent/'keytao-next'),
    'nextDirtyHash': hashlib.sha256((root.parent/'keytao-next/pnpm-workspace.yaml').read_bytes()).hexdigest(),
    'files': {path: hashlib.sha256((root/path).read_bytes()).hexdigest() for path in selected},
}
assert data['head'] == '9909dbf42a7aacdd57f69ec5811e1dee214f8cb6'
assert data['index'] == ''
assert data['nextHead'] == '1e421aae905cc201d617f347bd96b97bbbeff534'
assert data['nextStatus'] == 'M pnpm-workspace.yaml'
assert data['nextDirtyHash'] == 'c54f79df5b0c8375ec0d16da1141b2cd6ea91e039c4b4e923cc59e23e1c41287'
target = checks / ('source-freeze-' + sys.argv[2] + '.json')
if sys.argv[1] == 'save':
    target.write_text(json.dumps(data, indent=2))
else:
    previous = json.loads(target.read_text())
    drift = [path for path in set(previous['files']) | set(data['files']) if previous['files'].get(path) != data['files'].get(path)]
    assert previous == data, {'drift': drift}
print(json.dumps({'mode':sys.argv[1], 'fileCount':len(selected), 'head':data['head'], 'indexEmpty':True, 'nextUnchanged':True, 'time':datetime.datetime.now(datetime.timezone.utc).isoformat()}))
