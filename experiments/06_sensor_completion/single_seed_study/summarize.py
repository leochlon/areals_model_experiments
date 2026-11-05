from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/runs/areals_multicondition_wings_20261023')


def main():
    records = {r['id']: r for r in json.loads((ROOT / 'manifest.json').read_text())['records']}
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    for split in ['val', 'test']:
        path = ROOT / 'bench/scores' / f'{split}_per_problem.csv'
        rows = list(csv.DictReader(path.open()))
        families = sorted({records[r['car']]['family'] for r in rows})
        settings = list(dict.fromkeys(r['setting'] for r in rows))
        methods = ['geometry','measurements','conditioned']
        rng = np.random.default_rng(20261023)
        bootstrap = rng.integers(0, len(families), (4000, len(families)))
        result = dict(split=split, physical_geometry_groups=len(families), settings=[])
        for setting in settings:
            matrices = {}
            for metric in ['nrmse','rel_l2']:
                matrix = np.array([[
                    np.mean([float(r[metric]) for r in rows if r['setting']==setting and r['method']==method
                             and records[r['car']]['family']==family])
                    for method in methods] for family in families])
                assert np.isfinite(matrix).all()
                matrices[metric] = matrix
            square = np.array([[
                np.mean([float(r['nrmse'])**2 for r in rows if r['setting']==setting and r['method']==method
                         and records[r['car']]['family']==family])
                for method in methods] for family in families])
            mean = square.mean(0)
            denominator = mean[0] - mean[2]
            gap = float((mean[0] - mean[1]) / denominator) if denominator > 0 else None
            boot = square[bootstrap].mean(1)
            valid = boot[:,0] > boot[:,2]
            reference_gap_stable = float(np.quantile(boot[:,0]-boot[:,2], .025)) > 0
            ratios = (boot[:,0]-boot[:,1])/(boot[:,0]-boot[:,2])
            item = dict(setting=setting, nrmse={}, relative_l2={}, normalized_mse=dict(zip(methods,mean.tolist())),
                        fraction_gap_closed=gap,
                        fraction_gap_closed_95=np.quantile(ratios,[.025,.975]).tolist() if reference_gap_stable else None,
                        reference_gap_stable=reference_gap_stable,
                        bootstrap_positive_reference_gap_fraction=float(valid.mean()))
            for name, matrix in matrices.items():
                estimates = matrix[bootstrap].mean(1)
                destination = item['nrmse' if name=='nrmse' else 'relative_l2']
                for i,method in enumerate(methods):
                    destination[method] = dict(mean=float(matrix[:,i].mean()),
                                              low=float(np.quantile(estimates[:,i],.025)),
                                              high=float(np.quantile(estimates[:,i],.975)))
            result['settings'].append(item)
        (output / f'{split}_grouped.json').write_text(json.dumps(result,indent=2)+'\n')
        print('GROUPED_RESULTS',split,len(families),json.dumps(result['settings']),flush=True)
    geometry = json.loads((ROOT/'bench/predictions/geometry/test.json').read_text())
    assert geometry['method']['benchmark']['input_role']=='geometry'
    from ambiguity_bound import main as ambiguity_bound
    ambiguity_bound()
    # The scorer's own intervals treat the nine angles of a wing as independent; these group them by wing.
    print('wrote results/{val,test}_grouped.json',flush=True)


if __name__ == '__main__':
    main()
