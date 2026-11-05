"""Prepare the frozen Windsor mesh cohort; pressure files and splits are reused."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request

import h5py
import numpy as np
from scipy.spatial import cKDTree
import torch
from prepare_corpus import ROOT, SETTINGS, canonical_mesh, remesh, patch_arrays, write
import train_mesh_geometry as patch

BENCH=Path('/workspace/runs/areals_all_simulation_efficiency_20261028/datasets/windsor')
REV='8a6ca32ae22c94f54df2186d1b0ccf9662a294c2'


def download_once(case,folder):
    number=case.rsplit('_',1)[1];name=f'windsor_{number}.stl'
    path=folder/name
    if not path.exists():
        url=f'https://huggingface.co/datasets/neashton/windsorml/resolve/{REV}/run_{number}/{name}'
        temp=path.with_suffix('.tmp')
        with urllib.request.urlopen(url,timeout=120) as response,temp.open('wb') as f:
            while data:=response.read(1024*1024):f.write(data)
        temp.replace(path)
    return path


def download(case,folder):
    for attempt in range(3):
        try:return download_once(case,folder)
        except (OSError,ValueError) as error:
            if attempt==2:raise
            print(case,'download retry',attempt+1,type(error).__name__,flush=True)
            time.sleep(2**attempt)


def one(row,out):
    path=download(row['id'],out/'raw')
    native=canonical_mesh(path)
    # Public STL: y vertical, z spanwise. Frozen native benchmark: z vertical,
    # y spanwise. Apply this fixed documented permutation before fitting.
    native.vertices=np.asarray(native.vertices)[:,[0,2,1]]
    fitted,stats=remesh(native)
    seed=int(hashlib.sha256(('windsor-mesh|'+row['id']).encode()).hexdigest()[:8],16)
    args=patch.parser().parse_args(['prepare','--meshes',str(path),'--out','unused.npz'])
    for key in ['patches','max_faces','coverage_points','target_coverage']:setattr(args,key,SETTINGS[key])
    patches,meta=patch.build_mesh_patches(path,args,np.random.default_rng(seed),fitted,native)
    if not patches:raise ValueError(f'{row["id"]}: no patches; no car may be silently dropped')
    meta.update(split=row['split'],id=row['id'],remeshing=stats,source_revision=REV)
    meta['raw_to_benchmark_axes']=[0,2,1]
    with np.load(BENCH/'data/geometry'/f'{row["id"]}.npz') as z:points=z['points'].astype(np.float64)
    # Correspondence check is geometry-only, including validation/test cars.
    import train_query_geometry as q
    closest=q.Closest(native.vertices,native.faces,'open3d')
    feet,_=closest(points[::8]);gap=np.linalg.norm(points[::8]-feet,axis=1)
    scale=float(np.linalg.norm(np.ptp(native.bounds,axis=0)))
    if np.median(gap)/scale>.005:raise ValueError(f'{row["id"]}: mesh/cache unit or alignment mismatch')
    meta['cache_mesh_gap_median_over_diagonal']=float(np.median(gap)/scale)
    meta['cache_mesh_gap_p99_over_diagonal']=float(np.quantile(gap,.99)/scale)
    return row,patch_arrays(patches,meta),meta


def pack(index):
    torch.set_num_threads(1)
    rows=json.loads((ROOT/'windsor_protocol/data_manifest.json').read_text())['records']
    out=ROOT/'windsor';(out/'raw').mkdir(exist_ok=True,parents=True)
    path=out/f'pack_{index:03d}.h5';receipt=path.with_suffix('.json')
    if receipt.exists():return
    records=[]
    # One native remesher per process; array tasks provide parallelism.
    with h5py.File(str(path)+'.tmp','w') as h5:
        for row in rows[index*8:(index+1)*8]:
            row,data,meta=one(row,out);group=h5.create_group(row['id'])
            for name,array in data.items():group.create_dataset(name,data=array,compression='lzf')
            group.attrs['metadata']=json.dumps(meta)
            records.append(dict(row,pack=str(path),group=row['id'],metadata=meta))
            print(row['id'],meta['sampled_coverage'],len(data['inputs']),flush=True)
    Path(str(path)+'.tmp').replace(path)
    write(receipt,dict(records=records))


def consolidate():
    out=ROOT/'windsor';records=[]
    for receipt in sorted(out.glob('pack_*.json')):
        records+=json.loads(receipt.read_text())['records']
    expected=json.loads((ROOT/'windsor_protocol/data_manifest.json').read_text())['records']
    assert {r['id'] for r in records}=={r['id'] for r in expected} and len(records)==344
    write(out/'index.json',dict(records=records,complete=True,counts={s:sum(r['split']==s for r in records) for s in ['train','val','test']},
        no_car_exclusions=True,source=str(BENCH),raw_revision=REV))


if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('--pack',type=int);p.add_argument('--consolidate',action='store_true');a=p.parse_args()
    if a.consolidate:consolidate()
    else:pack(a.pack)
