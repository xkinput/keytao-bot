import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = []
for path in root.glob('S*-attempt-*.json'):
    if '-zdic-' in path.name:
        continue
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        continue
    result = data['result']
    rows.append({
        'scenario': data['scenarioId'], 'attempt': data['attempt'],
        'verdict': result['verdict'], 'failure': result.get('failure'),
        'seconds': round(result['durationSeconds'], 3),
    })
rows.sort(key=lambda row: (int(row['scenario'][1:]), row['attempt']))
print(json.dumps({
    'run': str(root), 'completed': len(rows),
    'passed': sum(row['verdict'] == 'PASSED' for row in rows),
    'failed': [row for row in rows if row['verdict'] != 'PASSED'],
    'latest': rows[-3:],
}, ensure_ascii=False))
