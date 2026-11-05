import concurrent.futures
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from registry import ROOT,OLD,write

sys.path.insert(0,'/workspace/runs/field_benchmark_4gpu_20261023')
import field_benchmark as fb


def coordinates(item):
    record,path=item
    with np.load(path) as z:points=z['points'].astype(np.float32)
    order=np.lexsort(points.T[::-1])
    ordered=np.ascontiguousarray(points[order]);ordered[ordered==0]=0
    return dict(id=record['id'],split=record['split'],points=len(points),
                coordinate_sha256=hashlib.sha256(ordered.tobytes()).hexdigest())


def main():
    ready=json.loads((ROOT/'results/preparation_complete.json').read_text())
    assert ready['passed'] and ready['test_cars']==48
    records=json.loads((ROOT/'data/manifest.json').read_text())['records']
    groups={split:{r['id'] for r in records if r['split']==split} for split in ['train','val','test']}
    assert [len(groups[k]) for k in ['train','val','test']]==[420,16,48]
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    source_split=Path('/workspace/data/drivaerml_splits')
    assert set((source_split/'test_design_ids.txt').read_text().split())==groups['test']
    tasks=[(r,(ROOT/'data/fields' if r['split']=='test' else OLD/'fields')/f'{r["id"]}.npz') for r in records]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:signatures=list(executor.map(coordinates,tasks))
    pools={}
    for record in signatures:pools.setdefault(record['coordinate_sha256'],[]).append(record)
    duplicates=[rows for rows in pools.values() if any(r['split']=='test' for r in rows) and len(rows)>1]
    write(ROOT/'results/split_and_coordinate_audit.json',dict(passed=not duplicates,counts={k:len(v) for k,v in groups.items()},
          signatures=signatures,duplicate_test_coordinate_sets=duplicates))
    assert not duplicates,('test coordinate sets duplicate other records',duplicates)
    # The observation masks depend on positions and areas only, never on pressure values.
    bench=ROOT/'benches/completion'
    fb.main(['observations','--manifest',str(ROOT/'data/manifest.json'),'--fields',str(ROOT/'data/fields'),
             '--bench',str(bench),'--split','test','--unobserved'])
    meta,arrays=fb.read_observations(bench,'test')
    entries=[e for e in meta['entries'] if e['setting']=='unobserved']
    arrays={e['key']:arrays[e['key']] for e in entries}
    meta.update(entries=entries,settings=[s for s in meta['settings'] if s['name']=='unobserved'])
    folder=ROOT/'benches/geometry/observations';folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'test.npz',**arrays);write(folder/'test.json',meta)
    write(ROOT/'results/evaluation_ready.json',dict(passed=True,test_cars=48,completion_settings=26,
          primary_completion_settings=25,geometry_settings=1))
    print('evaluation ready',flush=True)


if __name__=='__main__':main()
