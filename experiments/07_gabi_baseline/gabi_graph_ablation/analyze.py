import argparse
import json

import numpy as np

from common import ROOT, SEEDS, VARIANTS, run_dir, write_json

def analyze(split):
    data={}
    for seed in SEEDS:
        for variant in VARIANTS:
            path=run_dir(variant,seed)
            completed=json.loads((path/'complete.json').read_text())
            assert completed['status']=='PASS' and completed['epochs']==455 and completed['updates']==10010
            assert len(json.loads((path/'history.json').read_text()))==455
            scored=json.loads((path/split/'scores.json').read_text())
            assert scored['status']=='PASS' and len(scored['rows'])==1152
            data[variant,seed]={(r['case'],r['setting']):r for r in scored['rows']}
    reference=data['euclidean',SEEDS[0]]
    keys=set(reference)
    assert all(set(rows)==keys for rows in data.values())
    settings=sorted({r['setting'] for r in reference.values()})
    geometries=sorted({r['physical_geometry'] for r in reference.values()})
    rng=np.random.default_rng(20261028)
    boot=rng.integers(0,len(geometries),(10000,len(geometries)))
    results={}
    for setting in settings:
        results[setting]={}
        for metric in ['nrmse','force_rel']:
            by_seed=[]
            for seed in SEEDS:
                means={variant:float(np.mean([r[metric] for r in data[variant,seed].values() if r['setting']==setting])) for variant in VARIANTS}
                by_seed.append(dict(seed=seed,**means,difference=means['surface']-means['euclidean']))
            differences=[]
            for geometry in geometries:
                group=[key for key,r in reference.items() if r['setting']==setting and r['physical_geometry']==geometry]
                differences.append(float(np.mean([data['surface',seed][key][metric]-data['euclidean',seed][key][metric]
                                                  for seed in SEEDS for key in group])))
            delta=np.asarray(differences)
            interval=np.quantile(delta[boot].mean(1),[.025,.975])
            e=float(np.mean([r['euclidean'] for r in by_seed]))
            s=float(np.mean([r['surface'] for r in by_seed]))
            results[setting][metric]=dict(euclidean=e,surface=s,difference=s-e,relative_change=(s-e)/e,
                paired_seed_results=by_seed,geometry_bootstrap_difference_95ci=interval.tolist(),
                geometry_differences=dict(zip(geometries,differences)))
    # Negative differences favour the surface graph; the bootstrap resamples wings, keeping their nine angles together.
    report=dict(status='PASS',split=split,geometries=len(geometries),seeds=list(SEEDS),
        primary_setting='sensors-20',primary_metric='nrmse',results=results)
    write_json(ROOT/'results'/f'{split}_comparison.json',report)
    print(json.dumps({'split':split,'primary_result':results['sensors-20']},indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['val','test'],required=True)
    analyze(parser.parse_args().split)
