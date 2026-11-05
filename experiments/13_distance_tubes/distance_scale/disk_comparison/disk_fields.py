import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
sys.path.insert(0, str(BASE/'vendor'))
from areal_tokens.config import ModelConfig
from areal_tokens.model import GeometryAutoencoder, surfel_readout, _anchor_spacing


def json_save(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def independent_readout(query, centres, normals, radius):
    normals = normals/np.linalg.norm(normals, axis=1, keepdims=True)
    delta = query[:,None,:]-centres[None,:,:]
    gap = np.linalg.norm(delta,axis=-1)
    height = np.sum(delta*normals[None],axis=-1)
    lateral = np.sqrt(np.maximum(np.sum(delta**2,axis=-1)-height**2,0)+1e-12)
    disk = np.sqrt(height**2+np.maximum(lateral-radius,0)**2+1e-12)
    indices = np.argpartition(gap,15,axis=1)[:,:16]
    selected = np.take_along_axis(disk,indices,axis=1).min(axis=1)
    return selected,disk.min(axis=1)


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads((BASE/'protocol.json').read_text())
    saved = torch.load(protocol['checkpoint'],map_location='cpu',weights_only=True)
    model = GeometryAutoencoder(ModelConfig(**saved['config']['model'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['model'],strict=True)
    out = ROOT/'results'
    out.mkdir(exist_ok=True)
    rows = []
    for directory in sorted((BASE/'results').glob('*/metadata.json')):
        start = time.perf_counter()
        info = json.loads(directory.read_text())
        source = directory.parent
        dest = out/info['id']
        dest.mkdir(exist_ok=True)
        arrays = np.load(source/'prepared.npz')
        fields = np.load(source/'fields.npz')
        n = fields['true'].shape[0]
        h = float(fields['h'])
        query = fields['low']+np.indices((n,n,n)).reshape(3,-1).T*h
        center = arrays['center'].astype(np.float64)
        scale = float(arrays['scale'][0])
        batch = {k:torch.as_tensor(arrays[k],device='cuda',dtype=torch.float32)[None] for k in arrays.files}
        pred = []
        with torch.inference_mode():
            packet = model.encoder(batch)
            spacing = float(_anchor_spacing(packet,packet.surfel_positions,packet.surfel_weights>0)[0])
            for s in range(0,len(query),8192):
                world = torch.as_tensor(query[s:s+8192]*scale+center,device='cuda',dtype=torch.float32)[None]
                pred.append(surfel_readout(packet,world)[0][0].cpu().numpy())
            disk = np.concatenate(pred)
            indices = np.linspace(0,len(query)-1,256,dtype=int)
            world = torch.as_tensor(query[indices]*scale+center,device='cuda',dtype=torch.float32)[None]
            full = model.decode(packet,world)['normalized_udf'][0].cpu().numpy()
            normalised_actual_query = ((world-packet.center[:,None])/packet.scale[:,None])[0].cpu().numpy().astype(np.float64)
        replay = float(np.max(np.abs(full-fields['predicted'].reshape(-1)[indices])))
        assert replay < 2e-5, (info['id'],replay)
        independent, all_disks = independent_readout(normalised_actual_query,arrays['surfels'].astype(np.float64),arrays['surfel_normals'].astype(np.float64),.5*spacing)
        reader_delta = float(np.max(np.abs(disk[indices]-independent)))
        assert reader_delta < 2e-5, (info['id'],reader_delta)
        true = fields['true'].reshape(-1)
        over_index = int(np.argmax(disk-true))
        np.savez_compressed(dest/'disk_field.npz',disk=disk.reshape(n,n,n))
        data = dict(id=info['id'],group=info['group'],disk_radius=.5*spacing,epsilon_disk_grid=float(np.max(np.abs(disk-true))),disk_max_underestimate=float(np.max(np.maximum(true-disk,0))),disk_max_overestimate=float(np.max(np.maximum(disk-true,0))),full_max_underestimate=float(np.max(np.maximum(true-fields['predicted'].reshape(-1),0))),full_max_overestimate=float(np.max(np.maximum(fields['predicted'].reshape(-1)-true,0))),full_decoder_replay_max_delta=replay,independent_disk_formula_max_delta=reader_delta,native_16_vs_all_2048_max_delta_on_256_queries=float(np.max(np.abs(independent-all_disks))),overestimate_witness=dict(index=over_index,query=query[over_index].tolist(),true=float(true[over_index]),disk=float(disk[over_index]),full=float(fields['predicted'].reshape(-1)[over_index])),seconds=time.perf_counter()-start)
        json_save(dest/'field_metadata.json',data)
        rows.append(data)
        print('DISK_DONE',info['id'],round(data['seconds'],2),data['disk_max_underestimate'],data['disk_max_overestimate'],flush=True)
    json_save(out/'fields_summary.json',rows)
    json_save(out/'inference_provenance.json',dict(device=torch.cuda.get_device_name(),cases=len(rows)))


if __name__=='__main__':
    main()
