import json
import os
from pathlib import Path

os.umask(0o077)
ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/workspace/runs/gabi_doubledelta_20261023')
DATA = Path('/workspace/runs/areals_multicondition_wings_20261023')
UPSTREAM = ROOT / 'source/GABI'
SEEDS = (0, 1, 2)
VARIANTS = ('euclidean', 'surface')

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def records(split=None):
    rows = json.loads((DATA / 'manifest.json').read_text())['records']
    return [r for r in rows if split is None or r['split'] == split]

def run_dir(variant, seed):
    assert variant in VARIANTS and seed in SEEDS
    return ROOT / 'runs' / f'{variant}_seed{seed}'
