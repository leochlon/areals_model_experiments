import json
import sys
from pathlib import Path

import torch

from study import write


def main():
    out, base, seed = Path(sys.argv[1]), Path(sys.argv[2]).resolve(), int(sys.argv[3])
    validations = json.loads((out / 'validation_history.json').read_text())
    assert [v['update'] for v in validations] == list(range(228, 9121, 228))
    best = torch.load(out / 'field_best.pt', map_location='cpu', weights_only=True)
    last = torch.load(out / 'field_last.pt', map_location='cpu', weights_only=True)
    assert last['epoch'] == 40 and last['update'] == 9120
    assert best['update'] == min(validations, key=lambda v: v['val_nrmse'])['update']
    for state in [best, last]:
        meta = state['benchmark']
        assert Path(meta['base_checkpoint']).resolve() == base
        assert meta['seed'] == seed and meta['field_head_seed'] == 57300 + seed
        assert meta['epochs'] == 40 and meta['train_cars'] == 1824 and meta['queries'] == 2048
        assert meta['normalise'] == 'dataset'
        assert meta['mix'] == dict(unobserved=.1, sensors=.35)
        assert meta['new_lr'] == .001
        assert meta['lr'] == (0. if meta['freeze_trunk'] else .001)
    write(out / 'run_checks.json', dict(passed=True, seed=seed, updates=9120,
        epochs=40, validation_checks=40, base_checkpoint=str(base), best_update=best['update']))
    print('run checks passed', out.name, flush=True)


if __name__ == '__main__':
    main()
