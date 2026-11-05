"""Fine-tuning seed 0 of the d, nn^T-only priors (pretraining seeds 17 and 29) on WindsorML.

    python scripts/early_areals.py prepare --task CELL          # CELL 0-9: one prior and budget
    torchrun --standalone --nproc-per-node=4 scripts/early_areals.py fit --task FIT   # FIT 0-29: cell, rate

When the three rate candidates of a cell have finished, the one with the lowest validation error is scored on
the test cars (early_test_metrics.json).
"""
import argparse
import fcntl
import os

import torch

from common import ROOT, p, read, read_protocol, study, write
import pipeline


def cells():
    return read(ROOT / 'protocol/early_areals_cells.json')


def load_score(cell, winner):
    path = ROOT / cell['out'] / 'early_test_metrics.json'
    if not path.exists():
        return None
    result = read(path)
    assert result['cell'] == cell and result['winner'] == winner
    return result


def score_ready(cell):
    out = ROOT / cell['out']
    if not all((out / f'lr{i}/complete.json').exists() for i in range(3)):
        return
    with (out / '.early_score.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)  # one scorer per cell when fits finish together
        candidates = pipeline.candidate_records(cell)
        winner = min(candidates, key=lambda c: (c['validation'], c['lr']))
        protocol = read_protocol(out)
        selection = dict(cell=cell, candidates=candidates, winner=winner,
                         rule='Lowest validation error among the three rates after 8000 updates each.')
        selection_path = out / 'early_selection.json'
        if selection_path.exists():
            assert read(selection_path) == selection
        else:
            write(selection_path, selection)
        if load_score(cell, winner) is not None:
            return
        inputs = protocol['inputs']
        cars = p.Cars(inputs['windsor_index'], inputs['field_dir'], inputs['geometry_dir'], allowed_splits=('test',))
        checkpoint = ROOT / winner['checkpoint']
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['config'] == read(checkpoint.parent / 'config.json')
        device = torch.device('cuda', 0)
        p.deterministic(device)
        model = p.make_model(protocol['native_config'], 'scratch', cell['seed']).to(device)
        model.load_state_dict(state['model'], strict=True)
        metrics = p.evaluate(model, cars, cars.ids('test'), protocol['normalization'], device)
        write(out / 'early_test_metrics.json', dict(cell=cell, winner=winner, metrics=metrics))
        print('EARLY_AREALS_TEST', cell['model'], cell['budget'], metrics['mean_nrmse'], flush=True)


def prepare(task):
    pipeline.prepare(cells()[task], study()['pretraining']['steps'])


def fit(task):
    import pressure_fit
    pressure_fit.main(cells(), task)
    if int(os.environ['RANK']) == 0:
        score_ready(cells()[task // 3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'fit'])
    parser.add_argument('--task', type=int, required=True)
    args = parser.parse_args()
    prepare(args.task) if args.command == 'prepare' else fit(args.task)
