import json
import os
import shutil
from pathlib import Path

from study import ROOT, write

PRODUCTION = Path(os.environ['PRODUCTION_ROOT'])
PREVIOUS = Path('/workspace/runs/areals_mask_objective_control_20261025')


def main():
    for name in ['logs', 'restored_geometry', 'pretraining', 'field_models', 'bench', 'fields', 'clouds', 'cache']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PRODUCTION / 'config.json', ROOT / 'production_config.json')
    original_path = Path('/workspace/runs/objaverse_training_pool/versions/20260927T140652Z/manifest.json')
    original = json.loads(original_path.read_text())
    existing = {r['id']: r['path'] for r in json.loads((PREVIOUS / 'full_manifest.json').read_text())['records']}
    records = [{**r, 'original_path': r['path'], 'path': existing.get(r['id'], str(ROOT / 'restored_geometry' / (r['id'] + '.npz')))}
               for r in original['records']]
    assert [r['id'] for r in records] == [r['id'] for r in original['records']]
    counts = {s: sum(r['split'] == s for r in records) for s in ['train', 'val', 'test']}
    assert counts == dict(train=61822, val=1844, test=1825)
    write(ROOT / 'geometry_manifest.json', {**original, 'records': records})
    print('SETUP_COMPLETE', counts, flush=True)


if __name__ == '__main__':
    main()
