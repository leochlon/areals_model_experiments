import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree

import architecture as a
import transfer_architecture as transfer
from areal_tokens.config import ExperimentConfig
from areal_tokens.data import collate_clouds, load_cloud_file, to_device
from areal_tokens.geometry import canonical_cloud, prepare_cloud

OUT = a.ROOT / 'results/density_stress'


def sample(mesh, seed, biased):
    area = mesh.area_faces
    centers = mesh.triangles.mean(1)
    center = np.average(centers, axis=0, weights=area)
    offsets = centers - center
    axis = np.linalg.eigh((offsets * area[:, None]).T @ offsets)[1][:, -1]
    axis *= 1 if axis[np.argmax(np.abs(axis))] >= 0 else -1
    preference = np.where(offsets @ axis >= 0, 9., 1.) if biased else np.ones(len(area))
    probability = area * preference
    probability /= probability.sum()
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(area), 4096, p=probability)
    uv = rng.random((4096, 2))
    root = np.sqrt(uv[:, 0])
    bary = np.stack([1-root, root*(1-uv[:, 1]), root*uv[:, 1]], -1)
    points = (mesh.triangles[chosen] * bary[..., None]).sum(1)
    weights = area[chosen] / (4096 * probability[chosen])
    assert abs(weights.sum()/mesh.area-1) < .2
    return canonical_cloud(points, mesh.face_normals[chosen], weights)


def prepare(cloud):
    p = prepare_cloud(cloud, 256, 4096, np.random.default_rng(0))
    p.update(id='', family='', split='')
    return p


@torch.inference_mode()
def predict(model, p, queries):
    batch = to_device(collate_clouds([p]), torch.device('cuda'))
    q = torch.as_tensor(queries[None], dtype=torch.float32, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        result = model.decode(model.encoder(batch), q)
    return result['udf'][0].float().cpu().numpy(), result['normals'][0].float().cpu().numpy()


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.preferred_linalg_library('cusolver')
    OUT.mkdir(parents=True, exist_ok=True)
    records = json.loads((a.ROOT / 'architecture_manifest.json').read_text())['records']
    records = sorted((r for r in records if r['split']=='test'), key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:32]
    models = {}
    for arm in ['reference', 'ordinary_attention']:
        for seed in [0, 1, 2]:
            path = a.ROOT / 'results/architecture' / arm / f'seed{seed}/best.pt'
            state = torch.load(path, map_location='cpu', weights_only=True)
            transfer.ARM = arm
            model = transfer.base_model(ExperimentConfig.from_dict(state['config']), seed)
            model.load_state_dict(state['model'], strict=True)
            models[arm, seed] = model.cuda().eval()
    rows = []
    for number, record in enumerate(records, 1):
        _, target = load_cloud_file(record['path'])
        queries, truth = target['queries'][:1024], target['udf'][:1024]
        mesh = trimesh.load(record['source_mesh'], force='mesh', process=False, skip_materials=True)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[mesh.area_faces>0], process=False)
        baseline = prepare(sample(mesh, 101, False))
        scale = float(baseline['scale'][0])
        trials = [('uniform', 101, baseline)]
        for draw in [202, 303, 404]:
            cloud = sample(mesh, draw, True)
            p = prepare(cloud)
            trials.append(('biased_reanchor', draw, p))
            fixed = dict(p)
            fixed['points'] = ((cloud.points-baseline['center'])/baseline['scale']).astype(np.float32)
            fixed['normals'] = cloud.normals.astype(np.float32)
            fixed['point_weights'] = (cloud.weights/cloud.weights.sum()).astype(np.float32)
            for key in ['anchors','anchor_normals','anchor_weights','center','scale','area']:
                fixed[key] = baseline[key]
            trials.append(('biased_fixed_anchors_frame', draw, fixed))
        for (arm, seed), model in models.items():
            raw = dict(truth=truth, queries=queries, scale=np.asarray(scale))
            reference = None
            for mode, draw, p in trials:
                distance, normals = predict(model, p, queries)
                if reference is None:
                    reference = distance
                prefix = mode+'_'+str(draw)
                raw[prefix+'_distance'] = distance
                raw[prefix+'_normal'] = normals
                row = dict(id=record['id'], arm=arm, seed=seed, mode=mode, draw=draw,
                     normalized_mse=float(np.mean(((distance.astype(np.float64)-truth)/scale)**2)),
                     prediction_change=float(np.sqrt(np.mean(((distance.astype(np.float64)-reference)/scale)**2))))
                rows.append(row)
            np.savez_compressed(OUT / f'{record["id"]}_{arm}_seed{seed}.npz', **raw)
        a.write(OUT / 'rows.json', rows)
        print('DENSITY_STRESS', number, '/', len(records), flush=True)
    summary = {}
    for arm in ['reference', 'ordinary_attention']:
        summary[arm] = {}
        for mode in ['uniform', 'biased_fixed_anchors_frame', 'biased_reanchor']:
            selected = [r for r in rows if r['arm']==arm and r['mode']==mode]
            summary[arm][mode] = {k:float(np.mean([r[k] for r in selected])) for k in ['normalized_mse','prediction_change']}
    a.write(OUT / 'summary.json', dict(models=6, meshes=32, predictions=len(rows), settings=summary))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
