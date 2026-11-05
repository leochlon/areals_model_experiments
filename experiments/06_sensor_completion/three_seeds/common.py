import json
import os
from pathlib import Path
import sys

os.umask(0o077)
ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('/workspace/runs/areals_multicondition_wings_20261023')
GABI = Path('/workspace/runs/gabi_graph_ablation_20261028')
BASE = Path('/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/run/best_normal.pt')
BENCH = ROOT / 'bench'
sys.path[:0] = [str(ROOT / 'source'), str(ROOT / 'source/areal_src'), str(ROOT / 'source/geometry_src')]

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def model_path(seed):
    return BENCH / 'models' / f'measurements_seed{seed}'

def train_args(seed):
    return ['train-areal', '--bench', str(BENCH), '--checkpoint', str(BASE),
            '--out', str(model_path(seed)), '--epochs', '40', '--batch-size', '8',
            '--queries', '2048', '--lr', '1e-4', '--new-lr', '1e-3',
            '--weight-decay', '.05', '--visible-weight', '.25', '--normalise', 'dataset',
            '--floor', '.5', '--mix-unobserved', '.1', '--mix-sensors', '.45',
            '--workers', '0', '--select-cars', '36', '--eval-batch-size', '8',
            '--amp', 'bf16', '--seed', str(seed)]
