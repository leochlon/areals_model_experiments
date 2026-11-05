from pathlib import Path
import json
import os
from collections import Counter
import numpy as np

ROOT = Path(os.environ['TASK_ROOT'])
grf = ROOT / 'grf'
full = json.loads((grf/'manifest.json').read_text())['records']
small = json.loads((grf/'manifest_train30.json').read_text())['records']
assert Counter(r['split'] for r in full) == dict(train=1824, val=64, test=64)
assert Counter(r['split'] for r in small) == dict(train=240, val=64, test=64)
families = {}
for r in full:
    families.setdefault(r['family'], set()).add(r['split'])
assert len(families)==260 and all(len(s)==1 for s in families.values())
assert [r for r in full if r['split']=='val'] == [r for r in small if r['split']=='val']
meta = [json.loads((grf/b/'observations/val.json').read_text()) for b in ['bench','bench_train30']]
assert len(meta[0]['entries']) == len(meta[1]['entries']) == 320
assert meta[0]['entries'] == meta[1]['entries']
with np.load(grf/'bench/observations/val.npz') as a, np.load(grf/'bench_train30/observations/val.npz') as b:
    assert a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
for r in full:
    with np.load(grf/'fields'/(r['id']+'.npz')) as z:
        assert z['points'].shape==(32000,3) and z['normals'].shape==(32000,3)
        assert np.isfinite(z['values']).all()
out = dict(passed=True, training_shapes=228, low_data_shapes=30, validation_shapes=16,
           training_fields=1824, low_data_fields=240, validation_fields=64,
           validation_observation_sets=320)
(ROOT/'fields_ready.json').write_text(json.dumps(out,indent=2)+'\n')
print('FIELDS_READY',json.dumps(out),flush=True)
