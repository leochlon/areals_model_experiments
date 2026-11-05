import concurrent.futures
import importlib.util
import json
import multiprocessing
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from registry import ROOT,OLD,AUDIT,write

NATIVE = Path('/workspace/home/areal_aerojepa_20260928')
CONFIG_ROOT = Path('/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/field_utonia_compare_420')
sys.path.insert(0,str(CONFIG_ROOT))
import common


def original():
    spec=importlib.util.spec_from_file_location('author_prepare_420','/workspace/home/areal_drivaerml_420_epoch14_20260927T162928Z/prepare_420.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    legacy,cfg=module.legacy()
    native_fetch=legacy.Storage.fetch
    def fetch(storage,key,destination):
        result=native_fetch(storage,key,destination)
        if str(destination).endswith('.fields.h5'):
            target=Path(destination).parent.parent/'source_fields'/Path(destination).name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():os.link(destination,target)
        return result
    legacy.Storage.fetch=fetch
    return module,cfg


def directories(data):
    for name in ['geometry','fields','data_records','raw_tmp','source_fields']:(data/name).mkdir(parents=True,exist_ok=True)


def native_arrays(case,data,raw,cfg):
    record=dict(id=case,family=case,split='test',path=str(data/'geometry'/f'{case}.npz'))
    config=common.experiment()
    p=common.field.FieldDataset._prepare(record,data/'fields'/f'{case}.npz',config.model,config.data)
    anchors=p['anchor_index'].astype(np.int64)
    assert len(np.unique(anchors))==256
    np.testing.assert_array_equal(p['points'][anchors],p['anchors'])
    order=np.concatenate([anchors,np.setdiff1d(np.arange(len(p['points'])),anchors)])
    features=np.zeros((len(order),6),np.float32)
    features[:,:3]=p['normals'][order];features[:,3]=p['point_weights'][order]
    features[:256,4]=p['anchor_weights'];features[:,5]=p['area'].reshape(-1)[0]
    author,cfg=original();legacy,_=author.legacy();storage=legacy.Storage(cfg)
    scalar=data/'raw_tmp'/f'{case}.scalars.h5'
    try:
        source=storage.fetch(cfg['source_prefix']+case+'/scalars.h5',scalar)
        with h5py.File(scalar) as f:
            conditions=np.array([float(f['upstream_velocity'][()])/40,float(f['reference_area'][()])/2.17],np.float32)
    finally:storage.close()
    with h5py.File(raw) as f:
        points=f['mesh.verts'][:].astype(np.float64)
        normals=f['mesh.verts_normals'][:].astype(np.float64)
        weights=f['mass'][:].astype(np.float64).reshape(-1)
        pressure=f['mesh.PressureCoeff'][:].astype(np.float32).reshape(-1,1)
    weights[(weights>0)&(np.linalg.norm(normals,axis=-1)<1e-8)]=0
    rng=np.random.default_rng(cfg['sampling_seed']+100000+int(case.split('_')[-1]))
    chosen=rng.choice(len(points),size=2048,replace=True,p=weights/weights.sum())
    arrays=dict(context_pos=p['points'][order],context_feat=features,pressure=p['values'][order],
                center=p['center'],scale=p['scale'],gen_params_raw=conditions,
                fixed_query_pos=((points[chosen].astype(np.float32)-p['center'])/p['scale']).astype(np.float32),
                fixed_target=pressure[chosen])
    assert all(np.isfinite(x).all() for x in arrays.values())
    return arrays,dict(scalar_source=source)


def prepare(task):
    case,destination,parity=task;data=Path(destination);directories(data)
    finished=data/'data_records'/f'{case}.complete.json'
    if finished.exists():
        return json.loads(finished.read_text())
    author,cfg=original()
    recipe_record=data/'data_records'/f'{case}.json'
    result=author.prepare_case((case,str(data)))
    raw=data/'source_fields'/f'{case}.fields.h5'
    assert raw.exists()
    if parity:
        for kind in ['geometry','fields']:
            with np.load(data/kind/f'{case}.npz') as new,np.load(OLD/kind/f'{case}.npz') as old:
                assert set(new.files)==set(old.files)
                for key in old.files:np.testing.assert_array_equal(new[key],old[key],err_msg=f'{case}/{kind}/{key}')
    arrays,metadata=native_arrays(case,data,raw,cfg)
    if parity:
        with np.load(NATIVE/'data'/f'{case}.npz') as old:
            assert set(arrays)==set(old.files)
            for key in old.files:np.testing.assert_array_equal(arrays[key],old[key],err_msg=f'{case}/native/{key}')
    output=ROOT/'frozen/data'/f'{case}.npz';output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('wb') as stream:np.savez_compressed(stream,**arrays)
    geometry=(data/'geometry'/f'{case}.npz').relative_to(ROOT)
    field=(data/'fields'/f'{case}.npz').relative_to(ROOT)
    result.update(split='parity_control' if parity else 'test',native_metadata=metadata,
           gen_params_raw=arrays['gen_params_raw'].tolist(),
           files=[str(path) for path in [geometry,field,output.relative_to(ROOT)]])
    write(finished,result)
    return result


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert len(protocol['test_ids'])==48
    control='run_203'
    result=prepare((control,str(ROOT/'controls'),True))
    write(ROOT/'results/preparation_parity.json',result)
    print('parity with the stored arrays of',control,flush=True)
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as executor:
        futures={executor.submit(prepare,(case,str(ROOT/'data'),False)):case for case in protocol['test_ids']}
        for future in concurrent.futures.as_completed(futures):
            result=future.result();results.append(result)
            write(ROOT/'results/preparation_progress.json',dict(ready=len(results),total=48,records=results))
            print('PREPARED',result['case'],len(results),'/',48,flush=True)
    old=json.loads((OLD/'manifest.json').read_text())
    old_records=[{**r,'path':str(OLD/r['path'])} for r in old['records']]
    records=old_records+[dict(id=case,family=case,split='test',path=str(ROOT/'data/geometry'/f'{case}.npz')) for case in protocol['test_ids']]
    write(ROOT/'data/manifest.json',{**old,'records':records})
    for name,old_bench in [('geometry',AUDIT/'architecture_transfer/bench'),('completion',AUDIT/'replication/bench')]:
        cfg=json.loads((old_bench/'config.json').read_text())
        cfg.update(manifest=str(ROOT/'data/manifest.json'),fields=str(ROOT/'data/fields'))
        if name=='geometry':cfg['protocol']=dict(patch_counts=[],hidden=[],sensors=[],repeats=1,unobserved=True)
        else:cfg['protocol']['unobserved']=True
        write(ROOT/'benches'/name/'config.json',cfg)
    # The frozen-surrogate models keep the training statistics and parameter normalisation of their training run.
    native=json.loads((NATIVE/'data_ready.json').read_text())
    native['records']=[dict(id=r['case'],split='test') for r in results]
    native.update(test_cars=48)
    write(ROOT/'frozen/data_ready.json',native)
    write(ROOT/'results/preparation_complete.json',dict(passed=True,test_cars=48,
          maximum_excluded_surface_area_fraction=max(r['excluded_surface_area_fraction'] for r in results),records=results))
    print('preparation complete',flush=True)


if __name__=='__main__':main()
