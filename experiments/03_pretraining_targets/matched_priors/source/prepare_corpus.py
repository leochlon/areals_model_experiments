#!/usr/bin/env python3
"""Isolated per-mesh preparation into consolidated HDF5 packs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import h5py
import numpy as np
import torch
import trimesh

import train_mesh_geometry as patch
import train_query_geometry as query

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/workspace/runs/objaverse_training_pool/versions/20260927T140652Z/manifest.json')
SETTINGS = dict(edge_length=.01, remesh_iterations=5, remesh_deviation=.001,
                minimum_coverage=.95, max_source_faces=2000000, timeout_seconds=600,
                patches=1024, max_faces=2048, coverage_points=8192, target_coverage=.98)


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n'); temp.replace(path)


def list_objects():
    """The object list protocol/objects.json: the pool manifest in id order, pointed at the restored meshes."""
    rows = json.loads(ORIGINAL.read_text())['records']
    assert Counter(r['split'] for r in rows) == dict(train=61822, val=1844, test=1825)
    assert len({r['id'] for r in rows}) == len(rows)
    families = {}
    for r in rows:
        assert families.setdefault(r['family'], r['split']) == r['split']
    rows = sorted(rows, key=lambda r: r['id'])
    restored = json.loads(Path('/workspace/runs/areals_existing_data_expansion_20261025/larger_pretraining/geometry_manifest.json').read_text())
    restored = {r['id']:r for r in restored['records']}
    for i, r in enumerate(rows):
        candidate=restored[r['id']]
        assert candidate['source_mesh']==r['source_mesh'] and candidate['family']==r['family']
        r['original_path']=r['path'];r['path']=candidate['path']
        r['index'] = i
        r['seed'] = int(hashlib.sha256(('objaverse-targets-17|' + r['id']).encode()).hexdigest()[:8], 16)
    write(ROOT / 'protocol/objects.json', rows)
    write(ROOT / 'protocol/preparation.json', dict(original_manifest=str(ORIGINAL), counts=dict(Counter(r['split'] for r in rows)),
          settings=SETTINGS, units='Remesh lengths are fractions of original bounding-box diagonal.',
          original_splits_preserved=True, sampling='area on original mesh for final coverage',
          exclusions='Per-object failures and coverage below .95 are recorded; other objects continue. No split reassignment.',
          consolidation='One HDF5 pack per 128 source objects, with independently indexed patch/query groups.',
          references=['https://pymeshlab.readthedocs.io/en/latest/filter_list.html#meshing-isotropic-explicit-remeshing',
                      'https://trimesh.org/trimesh.repair.html#trimesh.repair.fix_winding']))
    print(json.dumps(dict(objects=len(rows), packs=(len(rows)+127)//128)))


def canonical_mesh(path):
    mesh = trimesh.load(path, force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('No triangle surface')
    if not np.isfinite(mesh.vertices).all():
        raise ValueError('Nonfinite vertices')
    if len(mesh.faces) > SETTINGS['max_source_faces']:
        raise ValueError('Source face resource limit: 2000000')
    mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=12)
    mesh.update_faces(mesh.nondegenerate_faces())
    # Every orientation of each face maps to the same deterministic soup.
    faces = np.unique(np.sort(np.asarray(mesh.faces), axis=1), axis=0)
    mesh = trimesh.Trimesh(mesh.vertices.copy(), faces, process=False)
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces) or mesh.area <= 0:
        raise ValueError('Zero-area geometry')
    # Local adjacency winding is defined also for open surfaces; no outward
    # normal or inside/outside assumption is made.
    trimesh.repair.fix_winding(mesh)
    return mesh


def remesh(native):
    import pymeshlab as pm
    center = native.bounds.mean(0)
    scale = float(np.linalg.norm(np.ptp(native.bounds, axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Zero-size geometry')
    ms = pm.MeshSet()
    ms.add_mesh(pm.Mesh((native.vertices - center) / scale, native.faces.astype(np.int32)))
    ms.meshing_isotropic_explicit_remeshing(iterations=SETTINGS['remesh_iterations'], adaptive=False,
        targetlen=pm.PureValue(SETTINGS['edge_length']), featuredeg=35., checksurfdist=True,
        maxsurfdist=pm.PureValue(SETTINGS['remesh_deviation']), reprojectflag=True)
    m = ms.current_mesh()
    result = trimesh.Trimesh(m.vertex_matrix() * scale + center, m.face_matrix(), process=False)
    edges = result.edges_unique_length / scale
    return result, dict(edge_target=SETTINGS['edge_length'], edge_p10=float(np.quantile(edges,.1)),
                       edge_median=float(np.median(edges)), edge_p90=float(np.quantile(edges,.9)),
                       faces=len(result.faces))


def patch_arrays(patches, meta):
    arrays = {k: np.asarray([p[k] for p in patches]) for k in
              ['origin','frame','radius','coeff','inputs','offset','fit_rms','fit_angle','stability','fit_condition','area_mass']}
    sizes = [len(p['triangles']) for p in patches]
    arrays.update(triangles=np.concatenate([p['triangles'] for p in patches]).astype(np.float32),
                  triangle_offsets=np.concatenate(([0],np.cumsum(sizes))),
                  mesh_index=np.zeros(len(patches),np.int32),
                  split=np.full(len(patches),['train','val','test'].index(meta['split']),np.int8))
    for key in ['inputs','origin','frame','radius']:
        arrays[key] = arrays[key].astype(np.float32)
    return arrays


def cached_query_labels(row):
    with np.load(row['path'],allow_pickle=False) as z:
        saved={k:np.array(z[k],dtype=np.float64) for k in z.files}
    weights=saved['weights']/saved['weights'].sum()
    center=(saved['points']*weights[:,None]).sum(0)
    radius=float(np.sqrt((((saved['points']-center)**2).sum(1)*weights).sum()))
    mesh=query.load_mesh(row['source_mesh'])
    closest=query.Closest((mesh.vertices-center)/radius,mesh.faces,'open3d')
    x=(saved['queries']-center)/radius
    feet,_=closest(x)
    d=saved['udf'].reshape(-1)/radius;g=x-feet
    check=float(np.max(np.abs(np.linalg.norm(g,axis=1)-d)))
    if check>1e-4:raise ValueError(f'Original cached targets and raw mesh disagree: {check} R')
    normals=saved['query_normals'];normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    P=normals[:,:,None]*normals[:,None,:]
    rng=np.random.default_rng(row['seed']);scales=[.02,.05,.15];H=[];H2=[];clipped=np.zeros(len(x),bool)
    for scale in scales:
        values,flag=query.clip_eigenvalues(query.smoothed_hessian(x,scale,query.random_directions(len(x),32,rng),closest),-8.)
        H.append(values);clipped|=flag
        independent=query.smoothed_hessian(x[:2048],scale,query.random_directions(min(2048,len(x)),32,rng),closest)
        H2.append(query.clip_eigenvalues(independent,-8.)[0])
    arrays=dict(surface=((saved['points']-center)/radius).astype(np.float32),
        normals=saved['normals'].astype(np.float32),surface_weights=weights.astype(np.float32),
        x=x.astype(np.float32),d=d.astype(np.float32),g=g.astype(np.float32),
        P=query.pack(P).astype(np.float32),H=query.pack(np.stack(H,1)).astype(np.float32),
        H2=query.pack(np.stack(H2,1)).astype(np.float32),near=d<=.1)
    meta=dict(rms_radius=radius,center=center.tolist(),cached_distance_max_disagreement_R=check,
        scales=scales,clipped_fraction=float(clipped.mean()),queries=len(x),
        targets='Original AREALS d and normal-line labels at its exact 32768 queries; additional smoothed Hessians at the same queries.')
    return arrays,meta


def worker(task, output):
    torch.set_num_threads(1)
    row = json.loads(Path(task).read_text())
    result = dict(id=row['id'], split=row['split'], family=row['family'], index=row['index'],
                  source=row['source_mesh'], seed=row['seed'])
    arrays = {}
    started = time.monotonic()
    native = None
    try:
        native = canonical_mesh(row['source_mesh'])
        fitted, mesh_report = remesh(native)
        args = patch.parser().parse_args(['prepare','--meshes',row['source_mesh'],'--out',output])
        for key in ['patches','max_faces','coverage_points','target_coverage']:
            setattr(args,key,SETTINGS[key])
        patches, meta = patch.build_mesh_patches(row['source_mesh'],args,np.random.default_rng(row['seed']),
                                                mesh=fitted,reference_mesh=native)
        meta.update(split=row['split'],group=row['family'],id=row['id'],remeshing=mesh_report)
        accepted = meta['sampled_coverage'] >= SETTINGS['minimum_coverage']
        result['patch'] = dict(status='ok' if accepted else 'coverage_excluded',metadata=meta)
        if accepted:
            arrays.update({'patch__'+k:v for k,v in patch_arrays(patches,meta).items()})
    except Exception as e:
        result['patch'] = dict(status='failed',error=f'{type(e).__name__}: {e}')
    result['query'] = dict(status='not_completed')
    np.savez_compressed(output,report=np.array(json.dumps(result)),**arrays)
    # Hessian labels use the original mesh, independently of patch success.
    try:
        qdata,meta=cached_query_labels(row)
        arrays.update({'query__'+k:v for k,v in qdata.items()})
        result['query'] = dict(status='ok',metadata=meta)
    except Exception as e:
        result['query'] = dict(status='failed',error=f'{type(e).__name__}: {e}')
    result['seconds'] = time.monotonic()-started
    arrays['report'] = np.array(json.dumps(result))
    np.savez_compressed(output,**arrays)


def isolated(row, work):
    task, output = work / (row['id']+'.json'), work / (row['id']+'.npz')
    task.write_text(json.dumps(row))
    log = work / (row['id']+'.log')
    try:
        with log.open('w') as f:
            proc = subprocess.run([sys.executable,__file__,'worker','--task',str(task),'--out',str(output)],
                                  stdout=f,stderr=subprocess.STDOUT,timeout=SETTINGS['timeout_seconds'])
        if proc.returncode != 0:
            raise RuntimeError(f'worker exit {proc.returncode}: '+log.read_text()[-1200:])
        with np.load(output,allow_pickle=False) as z:
            result = json.loads(str(z['report']))
        return row,output,result
    except Exception as e:
        failure=dict(status='failed',error=f'{type(e).__name__}: {e}')
        if output.exists():
            with np.load(output,allow_pickle=False) as z:
                partial=json.loads(str(z['report']))
            partial['query']=failure
            partial['worker_failed']=True
            return row,output,partial
        return row,None,dict(id=row['id'],index=row['index'],split=row['split'],family=row['family'],
                             patch=failure,query=failure,worker_failed=True)


def pack(args):
    rows = json.loads((ROOT / 'protocol/objects.json').read_text())
    rows = rows[args.pack*128:(args.pack+1)*128]
    if args.limit:
        rows=[r for r in rows if r['split']=='train'][:args.limit]
    dest=ROOT / args.out;dest.mkdir(parents=True,exist_ok=True)
    final=dest/f'pack_{args.pack:04d}.h5'
    receipt=dest/f'pack_{args.pack:04d}.json'
    if final.exists() and receipt.exists():
        print('Already complete',final);return
    records=[]
    with tempfile.TemporaryDirectory(prefix='preparing-',dir=dest) as temp:
        work=Path(temp)
        with h5py.File(str(final)+'.tmp','w') as h5, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(isolated,row,work) for row in rows]
            for future in as_completed(futures):
                row,output,report=future.result()
                if output:
                    with np.load(output,allow_pickle=False) as z:
                        for arm in ['patch','query']:
                            if report[arm]['status']!='ok':continue
                            group=h5.create_group(arm+'/'+row['id'])
                            meta=json.dumps(report[arm]['metadata'],sort_keys=True)
                            group.attrs['metadata']=meta
                            for key in sorted(k for k in z.files if k.startswith(arm+'__')):
                                value=np.ascontiguousarray(z[key]);name=key.split('__',1)[1]
                                group.create_dataset(name,data=value,compression='lzf')
                            report[arm]['group']=arm+'/'+row['id']
                    output.unlink()
                records.append(report)
                print(json.dumps(dict(id=row['id'],done=len(records),total=len(rows),
                      patch=report['patch']['status'],query=report['query']['status'])),flush=True)
    Path(str(final)+'.tmp').replace(final)
    write(receipt,dict(pack=args.pack,path=final.name,
                      bytes=final.stat().st_size,records=sorted(records,key=lambda r:r['index'])))


def consolidate(folder='prepared'):
    root=ROOT/folder; docs=[json.loads(p.read_text()) for p in sorted(root.glob('pack_*.json'))]
    records=[]
    for d in docs:
        records.extend([dict(r,pack=str((root/d['path']).resolve())) for r in d['records']])
    assert len({r['id'] for r in records})==len(records)
    counts={}
    for arm in ['patch','query']:
        counts[arm]={s:dict(Counter(r[arm]['status'] for r in records if r['split']==s))
                     for s in ['train','val','test']}
    write(root/'index.json',dict(version='objaverse-target-hdf5-v1',objects=records,
          settings=SETTINGS,survival=counts,requested_counts=dict(train=61822,val=1844,test=1825),
          complete=len(records)==65491,pack_files=len(docs)))
    print(json.dumps(dict(objects=len(records),survival=counts),indent=2))


if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('command',choices=['objects','worker','pack','consolidate'])
    p.add_argument('--task');p.add_argument('--out',default='prepared');p.add_argument('--pack',type=int,default=0)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    if a.command=='worker':worker(a.task,a.out)
    elif a.command=='pack':pack(a)
    elif a.command=='consolidate':consolidate(a.out)
    else:list_objects()
