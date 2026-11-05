import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

import author_nuisance_audit as author

ROOT = Path('/workspace/runs/areals_architecture_controls_20261024')
OUT = ROOT / 'results/nuisance_raw'
native_predict, native_view = author.predict, author.build_view
counter = 0


def view(*args, **kwargs):
    sample, rotation = native_view(*args, **kwargs)
    sample['_audit_rotation'] = np.eye(3) if rotation is None else rotation
    return sample, rotation


def predict(model, samples, device):
    global counter
    distance, normals = native_predict(model, samples, device)
    counter += 1
    np.savez_compressed(OUT / f'prediction{counter:04d}.npz', distance=distance, normals=normals,
        queries=np.stack([s['queries'] for s in samples]), truth=np.stack([s['udf'] for s in samples]),
        query_normals=np.stack([s['query_normals'] for s in samples]),
        scale=np.stack([s['scale'] for s in samples]), rotation=np.stack([s['_audit_rotation'] for s in samples]),
        normal_valid=np.stack([s['normal_valid'] for s in samples]), hidden=np.stack([s['query_hidden'] for s in samples]))
    return distance, normals


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.preferred_linalg_library('magma')
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT / 'architecture_manifest.json').read_text())
    records = sorted((r for r in source['records'] if r['split'] == 'test'),
                     key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())[:32]
    manifest = OUT / 'manifest.json'
    manifest.write_text(json.dumps({**source, 'records': records}, indent=2)+'\n')
    author.build_view, author.predict = view, predict
    sys.argv = [sys.argv[0], '--checkpoint', '/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/run/best_normal.pt',
         '--manifest', str(manifest), '--draws', '4', '--masks', '0.0,0.6', '--device', 'cuda', '--out', str(OUT / 'rows.json')]
    author.main()
