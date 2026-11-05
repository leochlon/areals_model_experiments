import copy
import json
import sys
import time

import numpy as np
import torch

from setup_confirmation import ROOT,OLD,write
sys.path.insert(0,str(OLD/'source'))
import field_benchmark as fb
from areal_tokens.config import ExperimentConfig


def main():
    path=ROOT/'bench/config.json';raw=path.read_bytes();config=json.loads(raw)
    assert config['protocol']['patch_counts']
    improved=copy.deepcopy(config);improved['protocol']['patch_counts']=[]
    record=next(r for r in json.loads((ROOT/'car_manifest.json').read_text())['records'] if r['split']=='train')
    state=torch.load(OLD/'results/architecture/reference/seed0/best.pt',map_location='cpu',weights_only=True)
    experiment=ExperimentConfig.from_dict(state['config'])
    started=time.monotonic()
    original=fb.Completion([record],config['fields'],experiment,config['protocol'],2048,0)
    full_seconds=time.monotonic()-started
    started=time.monotonic()
    lazy=fb.Completion([record],config['fields'],experiment,improved['protocol'],2048,0)
    lazy_seconds=time.monotonic()-started
    for data in [original,lazy]:
        data.meta=dict(normalise='dataset',mean=0.,std=1.,floor=0.,mix=dict(unobserved=1.,sensors=0.))
    count=0
    for seed in range(3):
        for epoch in [1,2,3]:
            for data in [original,lazy]:data.seed=seed;data.set_epoch(epoch)
            a,b=original[0],lazy[0]
            assert a.keys()==b.keys()
            for key in a:
                if isinstance(a[key],np.ndarray):np.testing.assert_array_equal(a[key],b[key])
                else:assert a[key]==b[key]
            count+=1
    previous=ROOT/'results/bench_config_before_unused_patch_optimization.json'
    assert not previous.exists();previous.write_bytes(raw)
    write(path,improved)
    result=dict(passed=True,fixture_car=record['id'],samples_compared=count,
        all_inputs_targets_queries_masks_identical=True,full_patch_setup_seconds=full_seconds,
        unused_patch_setup_seconds=lazy_seconds,old_patch_counts=config['protocol']['patch_counts'],new_patch_counts=[])
    write(ROOT/'results/unused_patch_optimization.json',result)
    print('UNUSED_PATCH_OPTIMIZATION_VALIDATED',json.dumps(result),flush=True)


if __name__=='__main__':main()
