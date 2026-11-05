import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'results/operator32_comparison.json'
report = json.loads(path.read_text())
first = json.loads((ROOT / 'checks/migration32.json').read_text())
second = json.loads((ROOT / 'checks/migration8.json').read_text())
report['geometry_allocation_history'] = [
    dict(gpus=4, from_step=0, through_step=first['resume_step']),
    dict(gpus=32, from_step=first['resume_step'] + 1, through_step=second['resume_step']),
    dict(gpus=8, from_step=second['resume_step'] + 1, through_step=10626),
]
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(report, indent=2) + '\n')
temporary.replace(path)
print('Recorded the four-to-32-to-eight GPU continuation in the final report.')
