import argparse
import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'AREALS_shared_query'))
import shared_query
from shared_query import pressure as p
import torch

BENCH = Path('/workspace/runs/areals_all_simulation_efficiency_20261028/datasets/windsor')


def cell(mode, task):
    if mode == 'interim':
        if task != 0:
            raise ValueError(task)
        return 274, 0, ROOT / 'pressure/interim_step3000/n274/seed0'
    if not 0 <= task < 15:
        raise ValueError(task)
    budget, seed = (18, 35, 69, 137, 274)[task // 3], task % 3
    return budget, seed, ROOT / f'pressure/operator/n{budget}/seed{seed}'


def prepare(mode, task, manifest):
    budget, seed, out = cell(mode, task)
    checkpoint_path = ROOT / manifest[mode]['checkpoint']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['step'] != manifest[mode]['geometry_step']:
        raise ValueError('Wrong geometry training step')
    args = p.parser().parse_args(list(map(str, [
        'tune', '--checkpoint', checkpoint_path, '--windsor-index', ROOT / 'windsor/index.json',
        '--field-dir', BENCH / 'data/fields', '--geometry-dir', BENCH / 'cache/geometry',
        '--subsets', ROOT / 'windsor_protocol/subsets.json', '--budget', budget, '--seed', seed,
        '--out', out, '--device', 'cuda', '--resume'])))
    cars = p.Cars(args.windsor_index, args.field_dir, args.geometry_dir)
    protocol = p._build_protocol(args, cars, checkpoint)
    path = out / 'protocol.json'
    if path.exists() and p.read_json(path) != protocol:
        raise ValueError('Existing protocol differs')
    p.write_json(path, protocol)
    print('Prepared', mode, budget, seed, flush=True)


def read_protocol(out):
    return p.read_json(out / 'protocol.json')


def candidate(mode, task, manifest):
    _, _, out = cell(mode, task // 6)
    initialization = ('pretrained', 'scratch')[(task % 6) // 3]
    rate_index = task % 3
    protocol = read_protocol(out)
    path = out / initialization / f'lr{rate_index}'
    if (path / 'complete.json').exists():
        assert p.read_json(path / 'complete.json')['step'] == protocol['steps']
        print('Already complete', path, flush=True)
        return
    inp = protocol['inputs']
    cars = p.Cars(inp['windsor_index'], inp['field_dir'], inp['geometry_dir'])
    checkpoint_path = ROOT / manifest[mode]['checkpoint']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    began = time.time()
    with (out / f'timing-{initialization}-lr{rate_index}.jsonl').open('a', buffering=1) as timing:
        timing.write(json.dumps(dict(event='start', unix=began,
                                    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                    task=task)) + '\n')
        p.train_candidate(protocol, initialization, protocol['rates'][rate_index], path,
                          cars, checkpoint['model'], resume=True)
        timing.write(json.dumps(dict(event='complete', unix=time.time(), seconds=time.time() - began)) + '\n')


def select(mode, task):
    _, _, out = cell(mode, task)
    if (out / 'selection.json').exists():
        p.load_selection(out / 'selection.json')
        return
    protocol = read_protocol(out)
    candidates = []
    for initialization in ('pretrained', 'scratch'):
        for index, rate in enumerate(protocol['rates']):
            path = out / initialization / f'lr{index}'
            complete = p.read_json(path / 'complete.json')
            assert complete['step'] == protocol['steps']
            candidates.append(dict(initialization=initialization, lr=rate, validation=complete['validation'],
                                   checkpoint=str((path / 'best.pt').relative_to(out))))
    winners = {i: min((c for c in candidates if c['initialization'] == i), key=lambda c: (c['validation'], c['lr']))
               for i in ('pretrained', 'scratch')}
    selection = dict(version=p.VERSION, rule='minimum full-validation mean per-car area/std nRMSE; tie lower LR',
                     protocol=protocol, candidates=candidates, winners=winners)
    p.write_json(out / 'selection.json', selection)
    p.load_selection(out / 'selection.json')
    print('Selected', out, flush=True)


def score_interim():
    _, _, out = cell('interim', 0)
    if not (out / 'test_metrics.json').exists():
        p.score(p.parser().parse_args(['score', '--selection', str(out / 'selection.json'), '--device', 'cuda']))
    result = p.read_json(out / 'test_metrics.json')
    reference_path = ROOT / 'checks/windsor_previous_patch.json'
    reference = p.read_json(reference_path)
    _, selection = p.load_selection(out / 'selection.json')
    report = dict(metric='Mean per-car area-weighted pressure RMSE / reference standard deviation',
                  current_geometry_steps=3000, previous_geometry_steps=10626,
                  previous_patch=reference, current=result,
                  selected_rates={i: w['lr'] for i, w in selection['winners'].items()},
                  pretrained_relative_change_vs_previous=result['metrics']['pretrained']['mean_nrmse'] /
                      reference['pretrained_seed0_nrmse'] - 1)
    p.write_json(ROOT / 'results/interim_step3000_vs_previous_patch.json', report)
    print(json.dumps(dict(current={k: v['mean_nrmse'] for k, v in result['metrics'].items()},
                          previous=reference), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'candidate', 'select', 'score-interim'])
    parser.add_argument('--mode', choices=['interim', 'final'], default='interim')
    parser.add_argument('--task', type=int, default=0, help='cell (prepare, select: 0-14) or fit (candidate: 0-89)')
    args = parser.parse_args()
    manifest = p.read_json(ROOT / 'protocol/windsor_parallel.json')
    task = args.task
    if args.command == 'prepare':
        prepare(args.mode, task, manifest)
    elif args.command == 'candidate':
        candidate(args.mode, task, manifest)
    elif args.command == 'select':
        select(args.mode, task)
    else:
        score_interim()


if __name__ == '__main__':
    main()
