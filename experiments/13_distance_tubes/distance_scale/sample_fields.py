import importlib.metadata
import json
import sys
import time
from pathlib import Path

import igl
import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor'))
from areal_tokens.config import ModelConfig
from areal_tokens.data import stable_seed
from areal_tokens.geometry import canonical_cloud, farthest_indices, prepare_cloud
from areal_tokens.model import GeometryAutoencoder
from areal_tokens.symmetry import symmetry_candidates

P = json.loads((ROOT / 'protocol.json').read_text())
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def shape_data(record):
    rng = np.random.default_rng(stable_seed(P['seed'], record['id'] + ':surface', 0))
    count = P['surface_samples']
    mesh = None
    if record['group'] == 'heldout':
        mesh = trimesh.load(record['source_mesh'], force='mesh', process=False, skip_materials=True)
        assert isinstance(mesh, trimesh.Trimesh) and np.isfinite(mesh.vertices).all()
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[mesh.area_faces > 0], process=False)
        area = float(mesh.area)
        face = rng.choice(len(mesh.faces), count, p=mesh.area_faces/area)
        uv = rng.random((count, 2))
        root = np.sqrt(uv[:, 0])
        bary = np.stack([1-root, root*(1-uv[:, 1]), root*uv[:, 1]], axis=1)
        points = (mesh.triangles[face]*bary[:, :, None]).sum(1)
        normals = mesh.face_normals[face]
        bounds = mesh.bounds
        extra = dict(triangles=len(mesh.faces), vertices=len(mesh.vertices), watertight=bool(mesh.is_watertight))
    elif record['id'] in ('sphere', 'two_spheres'):
        radius = 1.0 if record['id'] == 'sphere' else 0.55
        normals = rng.normal(size=(count, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        points = radius*normals
        if record['id'] == 'two_spheres':
            points[:, 0] += np.where(np.arange(count) % 2, 0.65, -0.65)
            area = 8*np.pi*radius**2
            bounds = np.array([[-1.2, -.55, -.55], [1.2, .55, .55]])
        else:
            area = 4*np.pi
            bounds = np.array([[-1., -1., -1.], [1., 1., 1.]])
        extra = dict(analytic=True, radius=radius, true_feature_radius=radius if record['id']=='sphere' else 0.10)
    else:
        major, minor = 1., .35
        theta = []
        while sum(len(x) for x in theta) < count:
            t = rng.uniform(0, 2*np.pi, 2*count)
            theta.append(t[rng.random(len(t)) < (major+minor*np.cos(t))/(major+minor)])
        t = np.concatenate(theta)[:count]
        phi = rng.uniform(0, 2*np.pi, count)
        normals = np.stack([np.cos(t)*np.cos(phi), np.cos(t)*np.sin(phi), np.sin(t)], axis=1)
        points = np.stack([(major+minor*np.cos(t))*np.cos(phi), (major+minor*np.cos(t))*np.sin(phi), minor*np.sin(t)], axis=1)
        area = 4*np.pi**2*major*minor
        bounds = np.array([[-1.35, -1.35, -.35], [1.35, 1.35, .35]])
        extra = dict(analytic=True, major_radius=major, minor_radius=minor, true_feature_radius=.35)
    cloud = canonical_cloud(points.astype(np.float32), normals.astype(np.float32), np.full(count, area/count, dtype=np.float32))
    return cloud, mesh, bounds, area, extra


def truth(record, q, mesh, center, scale):
    if mesh is not None:
        v = np.ascontiguousarray((mesh.vertices-center)/scale, dtype=np.float64)
        squared, _, _ = igl.point_mesh_squared_distance(np.ascontiguousarray(q, dtype=np.float64), v, np.ascontiguousarray(mesh.faces, dtype=np.int64))
        return np.sqrt(np.maximum(squared, 0))
    world = q*scale + center
    if record['id'] == 'sphere':
        return np.abs(np.linalg.norm(world, axis=1)-1.)/scale
    if record['id'] == 'torus':
        return np.abs(np.hypot(np.linalg.norm(world[:, :2], axis=1)-1., world[:, 2])-.35)/scale
    return np.minimum(np.abs(np.linalg.norm(world-[-.65, 0, 0], axis=1)-.55), np.abs(np.linalg.norm(world-[.65, 0, 0], axis=1)-.55))/scale


def predict(model, packet, q, center, scale):
    result = []
    with torch.inference_mode():
        for start in range(0, len(q), 8192):
            query = torch.as_tensor(q[start:start+8192]*scale+center, device='cuda', dtype=torch.float32)[None]
            result.append(model.decode(packet, query)['normalized_udf'][0].cpu().numpy())
    return np.concatenate(result)


def run_case(record, model):
    start = time.perf_counter()
    dest = OUT / record['id']
    dest.mkdir(exist_ok=True)
    print('START', record['id'], flush=True)
    cloud, mesh, bounds, area, extra = shape_data(record)
    arrays = prepare_cloud(cloud, model.config.latents, P['encoder_points'], np.random.default_rng(stable_seed(P['seed'], record['id']+':points', 0)), P['surfels'])
    center = arrays['center'].astype(np.float64)
    scale = float(arrays['scale'][0])
    rotation, translation, support, valid, local = symmetry_candidates(arrays['points'], arrays['normals'], arrays['surfels'], arrays['surfel_normals'], model.config.symmetry_maps, np.random.default_rng(stable_seed(P['seed'], record['id']+':symmetry', 0)))
    arrays.update(complete=np.array([1.], dtype=np.float32), symmetry_rotation=rotation, symmetry_translation=translation, symmetry_support=support, symmetry_valid=valid, symmetry_local=local)
    np.savez_compressed(dest/'prepared.npz', **arrays)
    batch = {k: torch.as_tensor(v, device='cuda', dtype=torch.float32)[None] for k, v in arrays.items()}
    with torch.inference_mode():
        packet = model.encoder(batch)
    bounds = (bounds-center)/scale
    mid = bounds.mean(axis=0)
    width = float(np.max(bounds[1]-bounds[0])+1.2)
    n = P['grid_n']
    h = width/(n-1)
    low = mid-width/2
    axes = [low[i] + h*np.arange(n) for i in range(3)]
    q = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    true = truth(record, q, mesh, center, scale)
    pred = predict(model, packet, q, center, scale)
    assert np.isfinite(true).all() and np.isfinite(pred).all() and np.min(pred) >= 0
    error = np.abs(pred-true)
    grid_worst = int(error.argmax())
    rng = np.random.default_rng(stable_seed(P['seed'], record['id']+':independent_queries', 0))
    unif = rng.uniform(low, low+width, (P['additional_queries']['uniform'], 3))
    near_idx = rng.integers(len(cloud.points), size=P['additional_queries']['near_surface'])
    near = (cloud.points[near_idx]-center)/scale + rng.normal(0, .03, (len(near_idx), 3))
    probes = np.concatenate([unif, near])
    probe_true = truth(record, probes, mesh, center, scale)
    probe_pred = predict(model, packet, probes, center, scale)
    probe_error = np.abs(probe_pred-probe_true)
    landmark_idx = farthest_indices(cloud.points, cloud.weights, 6)
    landmarks = (cloud.points[landmark_idx]-center)/scale
    landmark_pred = predict(model, packet, landmarks, center, scale)
    np.savez_compressed(dest/'fields.npz', true=true.reshape(n, n, n), predicted=pred.reshape(n, n, n), low=low, h=np.array(h), landmarks=landmarks, landmark_predicted=landmark_pred)
    np.savez_compressed(dest/'independent_queries.npz', queries=probes, true=probe_true, predicted=probe_pred)
    if mesh is not None:
        np.savez_compressed(dest/'mesh.npz', vertices=(mesh.vertices-center)/scale, faces=mesh.faces)
    k = int(probe_error.argmax())
    info = dict(**record, **extra, center=center.tolist(), scale=scale, surface_area=area/scale**2, grid_n=n, h=h, query_count=len(q)+len(probes)+len(landmarks), epsilon_grid=float(error.max()), epsilon_probes=float(probe_error.max()), epsilon_measured=float(max(error.max(), probe_error.max())), grid_witness=dict(query=q[grid_worst].tolist(), true=float(true[grid_worst]), predicted=float(pred[grid_worst])), independent_witness=dict(query=probes[k].tolist(), true=float(probe_true[k]), predicted=float(probe_pred[k])), valid_symmetry_maps=int(valid.sum()), duration_seconds=time.perf_counter()-start)
    save_json(dest/'metadata.json', info)
    print('DONE', record['id'], 'seconds', round(info['duration_seconds'], 1), 'epsilon', info['epsilon_measured'], flush=True)


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads(Path(P['manifest']).read_text())
    cohort = json.loads((ROOT/'cohort.json').read_text())
    assert cohort == sorted([r for r in manifest['records'] if r['split']=='test'], key=lambda r:r['id'])[:8]
    train_families = {r['family'] for r in manifest['records'] if r['split']=='train'}
    assert all(r['family'] not in train_families for r in cohort)
    checkpoint = torch.load(P['checkpoint'], map_location='cpu', weights_only=True)
    model = GeometryAutoencoder(ModelConfig(**checkpoint['config']['model'])).cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'], strict=True)
    provenance = dict(checkpoint_config=checkpoint['config'], device=torch.cuda.get_device_name(), packages={k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','trimesh','libigl']})
    save_json(OUT/'provenance.json', provenance)
    records = [dict(r, group='heldout') for r in cohort] + [dict(id=k, group='control') for k in ['sphere','torus','two_spheres']]
    for record in records:
        try:
            run_case(record, model)
        except Exception as e:
            import traceback
            traceback.print_exc()
            save_json(OUT/record['id']/'failure.json', dict(type=type(e).__name__, message=str(e)))
    save_json(OUT/'fields_complete.json', dict(cases=[r['id'] for r in records], succeeded=sum((OUT/r['id']/'metadata.json').exists() for r in records), failed=sum((OUT/r['id']/'failure.json').exists() for r in records)))


if __name__ == '__main__':
    main()
