import csv
import itertools
import json
from pathlib import Path

import numpy as np

ROOT=Path('/workspace/runs/areals_decoder_transfer_20261024')
FRESH=ROOT.parent/'areals_fresh_car_holdout_20261024'


def main():
    with (FRESH/'results/complete/per_problem.csv').open() as stream:
        rows=[r for r in csv.DictReader(stream) if r['family']=='architecture' and r['metric']=='nrmse']
    ids=sorted({r['car'] for r in rows});groups=sorted({r['group'] for r in rows})
    assert len(ids)==48 and len(groups)==7 and len(rows)==7*3*48
    lookup={(r['group'],int(r['seed']),r['car']):float(r['value']) for r in rows}
    matrices={g:np.array([[lookup[g,s,c] for c in ids] for s in range(3)]) for g in groups}
    draws=np.random.default_rng(20261024).integers(0,48,(10000,48));alpha=.05/21
    pairs=[]
    for a,b in itertools.combinations(groups,2):
        x,y=matrices[a],matrices[b];difference=(x-y).mean(0)
        low,high=np.quantile(difference[draws].mean(1),[alpha/2,1-alpha/2])
        pairs.append(dict(first=a,second=b,means=[float(x.mean()),float(y.mean())],
            mean_difference=float(difference.mean()),per_seed_differences=(x-y).mean(1).tolist(),
            bonferroni_all21_interval=[float(low),float(high)]))
    x,y=matrices['reference'],matrices['unanchored_no_rotary']
    difference=(x-y).mean(0);low,high=np.quantile(difference[draws].mean(1),[alpha/2,1-alpha/2])
    relative=1-x.mean(0)[draws].mean(1)/y.mean(0)[draws].mean(1)
    output=dict(passed=True,cars=48,groups=7,all21_pairs=pairs,
        combined_interface_comparison=dict(reference_mean=float(x.mean()),unanchored_no_rotary_mean=float(y.mean()),
            relative_mean_error_reduction=float(1-x.mean()/y.mean()),
            relative_reduction_all21_bonferroni_interval=np.quantile(relative,[alpha/2,1-alpha/2]).tolist(),
            mean_difference=float(difference.mean()),difference_all21_bonferroni_interval=[float(low),float(high)],
            per_seed_differences=(x-y).mean(1).tolist()))
    (ROOT/'results/combined_interface_contrast.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='all21_pairs'},indent=2))


if __name__=='__main__':main()
