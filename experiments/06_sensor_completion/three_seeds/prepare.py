"""Copies the seed-0 bench, model and predictions from the seed-0 study and records the training arguments."""
import shutil
from collections import Counter

import torch

from common import ROOT, ORIGINAL, BENCH, read, write, model_path, train_args

records = read(ORIGINAL / 'manifest.json')['records']
counts = Counter(r['split'] for r in records)
assert counts == {'train': 2160, 'val': 144, 'test': 144}
(BENCH / 'observations').mkdir(parents=True, exist_ok=True)
shutil.copyfile(ORIGINAL / 'bench/config.json', BENCH / 'config.json')
zero = model_path(0)
zero.mkdir(parents=True, exist_ok=True)
for name in ['field_best.pt', 'history.json']:
    shutil.copyfile(ORIGINAL / 'bench/models/measurements' / name, zero / name)
pred = BENCH / 'predictions/measurements_seed0'
pred.mkdir(parents=True, exist_ok=True)
for split in ['val', 'test']:
    for ext in ['json', 'npz']:
        shutil.copyfile(ORIGINAL / 'bench/observations' / f'{split}.{ext}', BENCH / 'observations' / f'{split}.{ext}')
        shutil.copyfile(ORIGINAL / 'bench/predictions/measurements' / f'{split}.{ext}', pred / f'{split}.{ext}')

state = torch.load(zero / 'field_best.pt', map_location='cpu', weights_only=False)
assert state['benchmark']['seed'] == 0 and state['benchmark']['epochs'] == 40
history = read(zero / 'history.json')
assert len(history) == 40 and min(history, key=lambda r: r['val_nrmse'])['epoch'] == state['epoch']
write(ROOT / 'protocol.json', dict(seeds=[0, 1, 2], new_seeds=[1, 2],
    training_args={str(seed): train_args(seed) for seed in [1, 2]}, epochs=40, updates=10800,
    primary_setting='sensors-20', primary_metric='nrmse', secondary_metric='force_rel',
    bootstrap='physical geometries, 10000 draws, keeping all angles and seeds of a geometry together',
    seed0_metadata=state['benchmark'], counts=dict(counts)))
write(ROOT / 'runs/seed0/training_complete.json', dict(status='PASS', source='seed-0 study',
      seed=0, epochs=40, updates=10800, selected_epoch=state['epoch']))
print('prepared seeds', [0, 1, 2], flush=True)
