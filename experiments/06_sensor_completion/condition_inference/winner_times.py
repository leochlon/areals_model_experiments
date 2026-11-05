import argparse
import time
from common import *
from condition_inference import Solver, posterior, checkpoint
from latency import sensor_forward


def winning_rows():
    a = read(ROOT / 'results/matched_fp32_areals.json')['summaries']
    b = read(ROOT / 'results/condition_inference_comparison.json')['summaries']
    rows = []
    for r in a:
        candidates = [s for s in b if s['dataset'] == r['dataset'] and s['setting'] == r['setting']
                      and s['method'] in ['plugin', 'mixture']]
        for domain in sorted({s['domain'] for s in candidates}):
            methods = {s['method']: s for s in candidates if s['domain'] == domain}
            if set(methods) != {'plugin', 'mixture'}:
                continue
            assert all(s['nrmse']['seeds'] == 3 for s in [r, *methods.values()])
            if r['nrmse']['mean'] < min(s['nrmse']['mean'] for s in methods.values()):
                rows.append(dict(dataset=r['dataset'], setting=r['setting'], domain=domain,
                    areals_nrmse=r['nrmse'], inference_nrmse=methods['mixture']['nrmse'],
                    plugin_nrmse=methods['plugin']['nrmse']))
    return rows


def measure(seed):
    setup_torch()
    winners = winning_rows()
    timings = []
    checkpoints = {}
    for name in sorted({r['dataset'] for r in winners}):
        solver = Solver(name, seed)
        sensor_path = historical_checkpoint(name, 'measurements', seed)
        model, _, state = load_checkpoint(sensor_path, name, 'measurements')
        checkpoints[name] = dict(areals=str(sensor_path), inference=str(checkpoint(name, seed)))
        selection = read(ROOT / 'inference_selection' / name / f'seed{seed}.json')
        meta, masks = fb.read_observations(ROOT / 'data' / name, 'val')
        entries = {(e['car'], e['setting']): e for e in meta['entries']}
        chosen = []
        families = set()
        for record in records(name, 'val'):
            if record['family'] not in families:
                chosen.append(record)
                families.add(record['family'])
            if len(chosen) == 4:
                break
        for r in chosen:
            f, prepared_geometry = field(name, r), prepared(name, r)
            for row in [w for w in winners if w['dataset'] == name]:
                index = masks[entries[(r['id'], row['setting'])]['key']]
                sigma = selection['sigma_by_domain_and_setting'][row['domain']][row['setting']]
                for repeat in range(4):
                    measured = {}
                    order = ['areals', 'inference'] if repeat % 2 == 0 else ['inference', 'areals']
                    for method in order:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        if method == 'areals':
                            sensor_forward(model, prepared_geometry, f, index, state['benchmark'])
                        else:
                            solver.prepare(prepared_geometry, f.points)
                            _, candidates = solver.grid(row['domain'] == 'extended')
                            posterior(candidates, index, f.values[index].copy(), sigma)
                        torch.cuda.synchronize()
                        measured[method + '_seconds'] = time.perf_counter() - start
                    if repeat:
                        timings.append(dict(dataset=name, domain=row['domain'], setting=row['setting'],
                            seed=seed, case=r['id'], repeat=repeat, **measured))
        del solver, model
        torch.cuda.empty_cache()
    write(ROOT / 'results' / f'winning_latency_seed{seed}.json', dict(status='PASS', rows=timings,
          winners=winners, checkpoints=checkpoints, precision='FP32', gpu=torch.cuda.get_device_name()))


def summarise():
    parts = [read(ROOT / 'results' / f'winning_latency_seed{s}.json') for s in range(3)]
    assert all(p['status'] == 'PASS' and p['winners'] == parts[0]['winners'] for p in parts)
    results = []
    for winner in parts[0]['winners']:
        rows = [r for p in parts for r in p['rows'] if all(r[k] == winner[k] for k in ['dataset', 'domain', 'setting'])]
        assert len(rows) == 36
        times = {method: float(np.median([r[method + '_seconds'] for r in rows])) for method in ['areals', 'inference']}
        assert all(np.isfinite(t) and t > 0 for t in times.values())
        results.append(dict(winner, median_seconds=times, timing_samples=len(rows)))
    write(ROOT / 'results/winning_comparison.json', dict(status='PASS', rows=results,
        filter='only configurations where the AREALS mean nRMSE is below both plug-in and mixture',
        timing_scope='median over 3 seeds, 4 validation wings and 3 timed repeats; one warm-up excluded; '
                     'alternating method order; FP32; includes encoding, search and full-field prediction, '
                     'excludes geometry preprocessing and checkpoint loading'))
    print(json.dumps(results))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['measure', 'summarise'])
    p.add_argument('--index', type=int, default=0)
    args = p.parse_args()
    measure(args.index) if args.command == 'measure' else summarise()
