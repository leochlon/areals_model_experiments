"""The WindsorML pressure protocol of one cell (prepare) and the three rate candidates of a cell."""
import torch

from common import ROOT, p, read, read_protocol, write

BENCH = '/workspace/runs/areals_all_simulation_efficiency_20261028/datasets/windsor'


def prepare(row, steps):
    out = ROOT / row['out']; checkpoint = ROOT / row['checkpoint']
    args = p.parser().parse_args(list(map(str, [
        'tune', '--checkpoint', checkpoint, '--windsor-index', ROOT / 'windsor/index.json',
        '--field-dir', BENCH + '/data/fields', '--geometry-dir', BENCH + '/cache/geometry',
        '--subsets', ROOT / 'windsor_protocol/subsets.json', '--budget', row['budget'], '--seed', row['seed'],
        '--out', out, '--device', 'cuda'])))
    cars = p.Cars(args.windsor_index, args.field_dir, args.geometry_dir)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['step'] == steps
    protocol = p._build_protocol(args, cars, state)
    protocol['control_cell'] = row
    if (out / 'protocol.json').exists():
        assert read(out / 'protocol.json') == protocol
    write(out / 'protocol.json', protocol)


def candidate_records(cell):
    protocol = read_protocol(ROOT / cell['out'])
    candidates = []
    for index, rate in enumerate(protocol['rates']):
        out = ROOT / cell['out'] / f'lr{index}'
        complete = read(out / 'complete.json'); best = read(out / 'best_validation.json')
        assert complete['step'] == protocol['steps']
        assert best['mean_nrmse'] == complete['validation']
        candidates.append(dict(lr=rate, validation=best['mean_nrmse'], step=best['step'],
                               checkpoint=str((out / 'best.pt').relative_to(ROOT))))
    return candidates
