import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(os.environ['TASK_ROOT'])
DD = Path(os.environ['DD_ROOT'])
records = json.loads((DD / 'manifest.json').read_text())['records']
groups = defaultdict(list)
by_inode = {}
for r in records:
    p = (DD / r['path']).resolve()
    st = p.stat()
    key = (st.st_dev, st.st_ino)
    if key not in by_inode:
        by_inode[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    groups[by_inode[key]].append(r)
assert len(records) == 2448 and len(groups) == 260
normal_checks = []
for members in groups.values():
    assert len({r['split'] for r in members}) == 1
    assert len({r['family'] for r in members}) == 1
    assert len(members) in (9, 18, 27)
    r = members[0]
    with np.load(DD / 'fields' / (r['id'] + '.npz')) as z:
        x, n, w = [np.asarray(z[k], dtype=np.float64) for k in ['points', 'normals', 'weights']]
    assert x.shape == n.shape == (32000, 3)
    assert np.isfinite(x).all() and np.isfinite(n).all() and (w > 0).all()
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, atol=1e-5)
    centre = np.average(x, axis=0, weights=w)
    flux = float((w * ((x-centre)*n).sum(1)).sum())
    assert flux > 0, (r['id'], 'normal orientation integral', flux)
    normal_checks.append(dict(id=r['id'], split=r['split'], outward_integral=flux))
mini = dict(json.loads((DD / 'manifest.json').read_text()))
mini['records'] = []
for split in ['train', 'val', 'test']:
    r = dict(next(r for r in records if r['split'] == split))
    r['path'] = str((DD / r['path']).resolve())
    mini['records'].append(r)
(ROOT / 'mini_source_manifest.json').write_text(json.dumps(mini, indent=2)+'\n')
out = dict(passed=True, records=len(records), distinct_shapes=len(groups),
           shapes_per_split=dict(Counter(m[0]['split'] for m in groups.values())),
           normal_checks=normal_checks)
(ROOT / 'data_checks.json').write_text(json.dumps(out, indent=2)+'\n')
print('data checks passed', {k:v for k,v in out.items() if k!='normal_checks'}, flush=True)
