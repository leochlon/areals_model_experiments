from collections import defaultdict
import numpy as np

from common import ROOT, ORIGINAL, GABI, BENCH, read, write
import study

study.configure('measurements')
fb = study.fb
for seed in [1,2]:
    assert read(ROOT / 'runs' / f'seed{seed}/evaluation_complete.json')['status'] == 'PASS'
records = {r['id']:r for r in read(ORIGINAL / 'manifest.json')['records']}
maximum_error = {'nrmse':0., 'force_rel':0.}
checked = 0
reports = {}
for split in ['val','test']:
    meta, masks = fb.read_observations(BENCH, split)
    groups = fb.by_car(meta['entries'])
    assert len(groups) == 144 and len(meta['entries']) == 1152
    fields = {}
    for case in groups:
        fields[case] = fb.load_field(ORIGINAL / 'fields', case)
    all_scores = []
    by_method = {'areals':{}, 'gabi_surface':{}}
    for seed in [0,1,2]:
        folder = BENCH / 'predictions' / f'measurements_seed{seed}'
        info = read(folder / f'{split}.json')
        assert info['problems'] == 1152
        rows = {}
        with np.load(folder / f'{split}.npz') as predictions:
            assert set(predictions.files) == {e['key'] for e in meta['entries']}
            for case, entries in groups.items():
                field = fields[case]
                truth, areas, normals = field.values, field.weights, field.normals
                spread = np.sqrt(np.average((truth - np.average(truth, weights=areas))**2, weights=areas))
                force = np.sum((truth * areas)[:,None] * normals, axis=0)
                for entry in entries:
                    prediction = predictions[entry['key']].astype(np.float64)
                    observed = masks[entry['key']]
                    hidden = np.ones(len(truth), bool)
                    hidden[observed] = False
                    nrmse = np.sqrt(np.average((prediction[hidden]-truth[hidden])**2, weights=areas[hidden])) / spread
                    completed = np.where(hidden, prediction, truth)
                    estimate = np.sum((completed * areas)[:,None] * normals, axis=0)
                    force_rel = np.linalg.norm(estimate-force) / np.linalg.norm(force)
                    metrics = fb.problem_metrics(field, observed, prediction)
                    for metric, value in [('nrmse',nrmse), ('force_rel',force_rel)]:
                        maximum_error[metric] = max(maximum_error[metric], abs(float(value)-metrics[metric]))
                    assert all(np.isfinite(metrics[m]) for m in ['nrmse','force_rel'])
                    row = dict(case=case, physical_geometry=records[case]['family'], setting=entry['setting'],
                               seed=seed, **metrics)
                    rows[case,entry['setting']] = row
                    all_scores.append(row)
                    checked += 1
        by_method['areals'][seed] = rows
        gabi = read(GABI / 'runs' / f'surface_seed{seed}' / split / 'scores.json')
        assert gabi['status'] == 'PASS'
        by_method['gabi_surface'][seed] = {(r['case'],r['setting']):r for r in gabi['rows']}
        assert set(rows) == set(by_method['gabi_surface'][seed])
    keys = set(by_method['areals'][0])
    assert all(set(rows)==keys for method in by_method.values() for rows in method.values())
    geometries = sorted({records[case]['family'] for case in groups})
    assert len(geometries) == 16
    assert all(sum(records[c]['family']==g for c in groups)==9 for g in geometries)
    draws = np.random.default_rng(20261029).integers(0,16,(10000,16))
    settings = sorted({e['setting'] for e in meta['entries']})
    results = {}
    for setting in settings:
        results[setting] = {}
        for metric in ['nrmse','force_rel']:
            result = {}
            geometry_means = {}
            for method, runs in by_method.items():
                per_seed = {str(seed):float(np.mean([r[metric] for r in rows.values() if r['setting']==setting]))
                            for seed, rows in runs.items()}
                result[method] = dict(mean=float(np.mean(list(per_seed.values()))),
                    seed_sd=float(np.std(list(per_seed.values()),ddof=1)), per_seed=per_seed)
                geometry_means[method] = np.array([np.mean([r[metric] for rows in runs.values() for r in rows.values()
                    if r['setting']==setting and r['physical_geometry']==g]) for g in geometries])
            a,g = result['areals']['mean'],result['gabi_surface']['mean']
            delta = geometry_means['gabi_surface'] - geometry_means['areals']
            result.update(gabi_minus_areals=g-a, gabi_over_areals=g/a,
                geometry_bootstrap_difference_95ci=np.quantile(delta[draws].mean(1),[.025,.975]).tolist())
            results[setting][metric] = result
    if split == 'test':
        assert abs(results['sensors-20']['nrmse']['areals']['per_seed']['0']-.16315916671215466)<1e-10
        assert abs(results['sensors-20']['force_rel']['areals']['per_seed']['0']-.033768449488751164)<1e-10
    report = dict(status='PASS', split=split, cases=144, geometries=16, angles_per_geometry=9,
        seeds=[0,1,2], primary_setting='sensors-20', primary_metric='nrmse', results=results)
    write(ROOT / 'results' / f'{split}_scores.json', dict(rows=all_scores))
    write(ROOT / 'results' / f'{split}_comparison.json', report)
    reports[split] = report
assert checked == 6912 and max(maximum_error.values()) < 1e-10
write(ROOT / 'results/verification.json', dict(status='PASS',
    recomputed_predictions=checked, maximum_difference_from_native_metrics=maximum_error))
print('test, sensors-20:', reports['test']['results']['sensors-20'], flush=True)
