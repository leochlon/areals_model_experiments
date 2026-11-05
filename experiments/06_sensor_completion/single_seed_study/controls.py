import json
from pathlib import Path

import numpy as np
import torch

import field_benchmark as fb
import study

ROOT = study.ROOT


def main():
    study.configure('measurements')
    config=fb.bench_config(ROOT/'bench')
    reference=json.loads((ROOT/'conditions_reference_only.json').read_text())['cases']
    records=fb.records_of(config,'test')
    meta,observations=fb.read_observations(ROOT/'bench','test')
    entries={(e['car'],e['setting']):e for e in meta['entries']}
    grouped={}
    for r in records: grouped.setdefault(r['grid_id'],[]).append(r)
    model=fb.Areal.load('measurements',ROOT/'bench/models/measurements/field_best.pt',torch.device('cuda'),1,'bf16')
    rows=[]
    for members in grouped.values():
        members.sort(key=lambda r:reference[r['id']]['aoa'])
        recipient,donor=members[0],members[-1]
        a=fb.load_field(config['fields'],recipient['id'])
        b=fb.load_field(config['fields'],donor['id'])
        assert np.array_equal(a.points,b.points)
        for count in [5,500]:
            setting=f'sensors-{count}'
            entry=entries[(recipient['id'],setting)]
            observed=observations[entry['key']]
            assert np.array_equal(observed,observations[entries[(donor['id'],setting)]['key']])
            settings={s['name']:s for s in meta['settings']}
            own=fb.Observed('own',settings[setting],observed,a.values[observed].copy())
            other=fb.Observed('donor',settings[setting],observed,b.values[observed].copy())
            blank=fb.Observed('blank',fb.BLIND,np.empty(0,np.int64),np.empty(0))
            surface=fb.surface_of(recipient,a)
            own_prediction=model.predict_car(surface,[own])[0]
            changed=model.predict_car(surface,[other])[0]
            donor_prediction=model.predict_car(fb.surface_of(donor,b),[other])[0]
            assert np.array_equal(changed,donor_prediction), 'Prediction followed case identity rather than supplied geometry/measurements'
            blank_a=model.predict_car(surface,[blank])[0]
            blank_b=model.predict_car(fb.surface_of(donor,b),[blank])[0]
            assert np.array_equal(blank_a,blank_b), 'Hidden angle leaked into geometry-only prediction'
            rows.append(dict(family=recipient['family'],grid_id=recipient['grid_id'],sensors=count,
                             correct_measurement_error=fb.problem_metrics(a,observed,own_prediction)['nrmse'],
                             wrong_measurement_error=fb.problem_metrics(a,observed,changed)['nrmse'],
                             donor_error_with_donor_measurements=fb.problem_metrics(b,observed,changed)['nrmse'],
                             prior_error=fb.problem_metrics(a,observed,blank_a)['nrmse'],
                             swap_equivalence_max_abs=float(np.max(np.abs(changed-donor_prediction))),
                             geometry_only_condition_invariance_max_abs=float(np.max(np.abs(blank_a-blank_b)))))
        print('CONTROL',recipient['grid_id'],flush=True)
    fb.save_json(ROOT/'results/measurement_swap_controls.json',dict(passed=True,cases=rows))


if __name__=='__main__':main()
