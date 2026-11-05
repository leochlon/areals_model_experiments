import argparse
import json

import torch

from util import ROOT, write

p = argparse.ArgumentParser()
p.add_argument('--dataset', required=True)
p.add_argument('--name', required=True)
p.add_argument('--seed', type=int, required=True)
a = p.parse_args()
d = ROOT / 'datasets' / a.dataset
out = d / 'models' / f'{a.name}_seed{a.seed}'
state = torch.load(out / 'field_best.pt', weights_only=True, map_location='cpu')
last = torch.load(out / 'field_last.pt', weights_only=True, map_location='cpu')
meta = state['benchmark']
assert meta['lr'] == meta['new_lr'] == 1e-3
assert meta['epochs'] == 40 and last['epoch'] == 40
assert meta['scratch'] == (a.name == 'scratch')
assert meta['normalise'] == 'dataset'
assert meta['seed'] == a.seed
assert all(torch.isfinite(v).all() for v in state['model'].values())
history = json.loads((out / 'history.json').read_text())
assert len(history) == 40
write(out / 'complete.json', dict(passed=True, selected_epoch=state['epoch'], final_epoch=last['epoch'],
      final_updates=history[-1]['update']))
