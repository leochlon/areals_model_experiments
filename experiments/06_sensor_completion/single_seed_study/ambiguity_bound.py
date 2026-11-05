import csv
import json
from pathlib import Path

import numpy as np

import field_benchmark as fb

ROOT=Path('/workspace/runs/areals_multicondition_wings_20261023')


def main():
    config=fb.bench_config(ROOT/'bench')
    records=fb.records_of(config,'test')
    grouped={}
    for r in records: grouped.setdefault(r['grid_id'],[]).append(r)
    meta,observations=fb.read_observations(ROOT/'bench','test')
    lookup={(e['car'],e['setting']):e for e in meta['entries']}
    scored=list(csv.DictReader((ROOT/'bench/scores/test_per_problem.csv').open()))
    scores={(r['car'],r['setting'],r['method']):float(r['nrmse'])**2 for r in scored}
    priors=np.load(ROOT/'bench/predictions/geometry/test.npz')
    result=[]
    for grid,members in grouped.items():
        assert len(members)==9
        fields=[fb.load_field(config['fields'],r['id']) for r in members]
        y=np.stack([f.values for f in fields])
        weights=fields[0].weights
        assert all(np.array_equal(weights,f.weights) for f in fields)
        inverse_variance=np.array([1/fb.moments(f.values,f.weights)[1]**2 for f in fields])
        conditional_mean=(inverse_variance@y)/inverse_variance.sum()
        irreducible=((inverse_variance@(y*y))-(inverse_variance@y)**2/inverse_variance.sum())/9
        assert irreducible.min()>-1e-10
        irreducible=np.maximum(irreducible,0)
        predictions=[priors[lookup[(r['id'],'unobserved')]['key']].astype(np.float64) for r in members]
        assert all(np.array_equal(predictions[0],p) for p in predictions[1:])
        prediction=predictions[0]
        for setting in meta['settings']:
            observed=observations[lookup[(members[0]['id'],setting['name'])]['key']]
            assert all(np.array_equal(observed,observations[lookup[(r['id'],setting['name'])]['key']]) for r in members)
            hidden=np.ones(len(weights),bool);hidden[observed]=False
            w=weights[hidden]/weights[hidden].sum()
            bound=float(w@irreducible[hidden])
            geometry=float(np.mean([scores[(r['id'],setting['name'],'geometry')] for r in members]))
            excess=float(inverse_variance.sum()/9*(w@((prediction[hidden]-conditional_mean[hidden])**2)))
            assert abs(geometry-bound-excess)<1e-6,(grid,setting['name'],geometry,bound,excess)
            measurement=float(np.mean([scores[(r['id'],setting['name'],'measurements')] for r in members]))
            result.append(dict(grid_id=grid,family=members[0]['family'],setting=setting['name'],
                               exact_geometry_only_normalized_mse_lower_bound=bound,
                               geometry_model_normalized_mse=geometry,
                               measurements_normalized_mse=measurement,
                               measurement_gain_below_geometry_only_bound=bound-measurement))
    priors.close()
    # Lower bound on the normalised MSE of any one prediction that ignores which of the nine angles applies.
    fb.save_json(ROOT/'results/test_ambiguity_bound.json',dict(rows=result))


if __name__=='__main__':main()
