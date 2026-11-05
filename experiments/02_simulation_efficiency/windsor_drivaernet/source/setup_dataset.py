from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import sys

from util import ROOT, write

os.umask(0o077)
dataset = sys.argv[1]
out = ROOT / 'datasets' / dataset
assert not (out / 'protocol.json').exists(), 'Dataset protocol already created'
out.mkdir(parents=True, exist_ok=True)
if dataset in ['drivaernet', 'doubledelta', 'blendednet']:
    old = Path('/workspace/runs/areals_existing_data_expansion_20261025/datasets') / dataset
    manifest = json.loads((old / 'manifest.json').read_text())
    assert json.loads((old / 'ready.json').read_text())['passed']
    (out / 'data/geometry').mkdir(parents=True, exist_ok=True)
    (out / 'data/fields').mkdir(parents=True, exist_ok=True)
    for record in manifest['records']:
        for kind, original in [('geometry', Path(record['path'])), ('fields', old / 'fields' / (record['id'] + '.npz'))]:
            target = out / 'data' / kind / (record['id'] + '.npz')
            if not target.exists():
                os.link(original, target)
        record['path'] = 'geometry/' + record['id'] + '.npz'
    write(out / 'data/manifest.json', manifest)
    (out / 'evidence').mkdir(exist_ok=True)
    for name in ['ready.json', 'split_protocol.json', 'data_checks.json', 'conditions.json']:
        if (old / name).exists():
            shutil.copy2(old / name, out / 'evidence' / name)
    write(out / 'data_ready.json', dict(passed=True, source='Existing native-field preparation'))
else:
    assert json.loads((out / 'data_ready.json').read_text())['passed']
    manifest = json.loads((out / 'data/manifest.json').read_text())
for name in ['source', 'model_src', 'checkpoints']:
    if not (out / name).exists():
        (out / name).symlink_to(ROOT / name, target_is_directory=True)
spec = json.loads((ROOT / 'suite_protocol.json').read_text())
counts = dict(Counter(r['split'] for r in manifest['records']))
maximum = counts['train']
spec.update(dataset=dataset, budgets=sorted({math.ceil(maximum / denominator) for denominator in (16, 8, 4, 2, 1)}),
            split=counts, families={s: len({r['family'] for r in manifest['records'] if r['split'] == s}) for s in counts})
assert len(spec['budgets']) == 5
write(out / 'protocol.json', spec)
print(json.dumps(dict(dataset=dataset, budgets=spec['budgets'], split=counts, families=spec['families'])), flush=True)
