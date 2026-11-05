import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
P = json.loads((ROOT/'protocol.json').read_text())
BASE = Path(P['remote_baseline'])
sys.path.insert(0, str(ROOT/'code'))
from areal_tokens.config import ModelConfig
from areal_tokens.geometry import prepare_cloud, visible_mask
from areal_tokens.data import stable_seed
from areal_tokens.model import GeometryAutoencoder, surfel_readout, _anchor_spacing
from areal_tokens.symmetry import symmetry_candidates

sys.path.insert(0, str(BASE))
import sample_fields as source
sys.path.insert(0, str(BASE/'disk_comparison'))
from disk_fields import independent_readout


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def batch(arrays):
    return {k:torch.as_tensor(v, device='cuda', dtype=torch.float32)[None] for k,v in arrays.items()}


def evaluate(model, packet, queries, center, scale):
    learned, disk = [], []
    with torch.inference_mode():
        for start in range(0, len(queries), 8192):
            world = torch.as_tensor(queries[start:start+8192]*scale+center, device='cuda', dtype=torch.float32)[None]
            learned.append((model.decode(packet, world)['udf'][0]/scale).cpu().numpy())
            disk.append((surfel_readout(packet, world)[0][0]*packet.scale[0,0]/scale).cpu().numpy())
    return np.concatenate(learned), np.concatenate(disk)


def errors(identity, group, domain, truth, learned, disks):
    rows = []
    for name, values in [('full', learned), ('disk', disks)]:
        delta = values.astype(np.float64)-truth
        rows.append(dict(id=identity, group=group, domain=domain, field=name, queries=len(truth), max_absolute_error=float(np.abs(delta).max()), max_underestimate=float(np.maximum(-delta,0).max()), max_overestimate=float(np.maximum(delta,0).max())))
    return rows


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(P['checkpoint'], map_location='cpu', weights_only=True)
    config = ModelConfig(**checkpoint['config']['model'])
    model = GeometryAutoencoder(config).cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'], strict=True)
    spec = importlib.util.spec_from_file_location('areal_tokens.original_model', ROOT/'baseline/areal_tokens/model.py')
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    original = legacy.GeometryAutoencoder(config).cuda().eval().requires_grad_(False)
    original.load_state_dict(checkpoint['model'], strict=True)
    assert list(model.state_dict()) == list(original.state_dict())
    metrics, checks, cases = [], [], []
    for file in sorted((BASE/'results').glob('*/metadata.json')):
        start = time.perf_counter()
        meta = json.loads(file.read_text())
        identity = meta['id']
        print('START', identity, flush=True)
        target = ROOT/'results'/identity
        target.mkdir(exist_ok=True)
        complete_arrays = dict(np.load(file.parent/'prepared.npz'))
        fields = np.load(file.parent/'fields.npz')
        n, h = fields['true'].shape[0], float(fields['h'])
        grid = fields['low']+np.indices((n,n,n)).reshape(3,-1).T*h
        center, scale = complete_arrays['center'].astype(np.float64), float(complete_arrays['scale'][0])
        indices = np.linspace(0,len(grid)-1,256,dtype=int)
        world = torch.as_tensor(grid[indices]*scale+center, device='cuda', dtype=torch.float32)[None]
        with torch.inference_mode():
            full_packet = model.encoder(batch(complete_arrays))
            fixed = model.decode(full_packet,world)
            old = original.decode(full_packet,world)
            direct = surfel_readout(full_packet,world)[0]
            assert torch.equal(fixed['normalized_udf'], direct)
            assert torch.equal(fixed['normal_tensor'], old['normal_tensor'])
            assert torch.equal(model.decode(replace(full_packet,complete=None),world)['normalized_udf'], direct)
            old_delta = float(np.max(np.abs(old['normalized_udf'][0].cpu().numpy()-fields['predicted'].reshape(-1)[indices])))
            assert old_delta < 2e-5
        cloud, mesh, _, _, _ = source.shape_data(meta)
        visible = visible_mask(cloud, .6, 32, np.random.default_rng(stable_seed(P['seed'],identity+':mask',0)))
        arrays = prepare_cloud(cloud.subset(visible),config.latents,4096,np.random.default_rng(stable_seed(P['seed'],identity+':points',0)),2048)
        rotation, translation, support, valid, local = symmetry_candidates(arrays['points'],arrays['normals'],arrays['surfels'],arrays['surfel_normals'],config.symmetry_maps,np.random.default_rng(stable_seed(P['seed'],identity+':symmetry',0)))
        arrays.update(complete=np.array([0.],dtype=np.float32),symmetry_rotation=rotation,symmetry_translation=translation,symmetry_support=support,symmetry_valid=valid,symmetry_local=local)
        np.savez_compressed(target/'prepared.npz',**arrays)
        surfel_world = arrays['surfels'].astype(np.float64)*float(arrays['scale'][0])+arrays['center']
        leakage_distance = float(cKDTree(cloud.points[visible]).query(surfel_world)[0].max()/scale)
        assert leakage_distance < 1e-5
        with torch.inference_mode():
            packet = model.encoder(batch(arrays))
            patched = model.decode(packet,world)
            previous = original.decode(packet,world)
            assert all(torch.equal(patched[key],previous[key]) for key in patched)
            native = surfel_readout(packet,world)[0][0].cpu().numpy()
            actual_q = ((world-packet.center[:,None])/packet.scale[:,None])[0].cpu().numpy().astype(np.float64)
            spacing = float(_anchor_spacing(packet,packet.surfel_positions,packet.surfel_weights>0)[0])
        independent, _ = independent_readout(actual_q,arrays['surfels'].astype(np.float64),arrays['surfel_normals'].astype(np.float64),.5*spacing)
        formula_delta = float(np.abs(native-independent).max())
        assert formula_delta < 2e-5
        learned, disks = evaluate(model,packet,grid,center,scale)
        np.savez_compressed(target/'fields.npz',predicted=learned.reshape(n,n,n),disk=disks.reshape(n,n,n))
        metrics.extend(errors(identity,meta['group'],'grid',fields['true'].reshape(-1),learned,disks))
        probes = np.load(file.parent/'independent_queries.npz')
        p_learned, p_disk = evaluate(model,packet,probes['queries'],center,scale)
        np.savez_compressed(target/'independent_queries.npz',queries=probes['queries'],true=probes['true'],predicted=p_learned,disk=p_disk)
        metrics.extend(errors(identity,meta['group'],'independent',probes['true'],p_learned,p_disk))
        surface = (cloud.points-center)/scale
        true_surface = source.truth(meta,surface,mesh,center,scale)
        s_learned, s_disk = evaluate(model,packet,surface,center,scale)
        np.savez_compressed(target/'surface_queries.npz',queries=surface,true=true_surface,visible=visible,weights=cloud.weights/cloud.weights.sum(),predicted=s_learned,disk=s_disk)
        for domain, mask in [('visible_surface',visible),('hidden_surface',~visible)]:
            metrics.extend(errors(identity,meta['group'],domain,true_surface[mask],s_learned[mask],s_disk[mask]))
        result = dict(id=identity,group=meta['group'],actual_hidden_area_fraction=float(cloud.weights[~visible].sum()/cloud.weights.sum()),visible_points=int(visible.sum()),hidden_points=int((~visible).sum()),encoder_points=len(arrays['points']),disk_centres=len(arrays['surfels']),visible_center=arrays['center'].tolist(),visible_scale=float(arrays['scale'][0]),evaluation_center=center.tolist(),evaluation_scale=scale,valid_symmetry_maps=int(valid.sum()),disk_centre_to_visible_sample_max_error=leakage_distance,seconds=time.perf_counter()-start)
        check = dict(id=identity,complete_exactly_equals_native_disks=True,complete_normals_unchanged=True,legacy_none_equals_complete=True,masked_all_outputs_unchanged=True,old_complete_replay_max_delta=old_delta,independent_masked_disk_formula_max_delta=formula_delta)
        save(target/'metadata.json',result)
        save(target/'regression.json',check)
        cases.append(result); checks.append(check)
        save(ROOT/'results/distance_metrics.json',metrics)
        print('DONE',identity,round(result['seconds'],1),'hidden_fraction',round(result['actual_hidden_area_fraction'],4),flush=True)
    save(ROOT/'results/cases.json',cases)
    save(ROOT/'results/regression.json',checks)
    save(ROOT/'results/inference_provenance.json',dict(device=torch.cuda.get_device_name(),cases=len(cases),state_dict_keys_unchanged=True))
    print('FIELDS_COMPLETE',len(cases),flush=True)


if __name__=='__main__':
    main()
