import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.nn import functional as F

from areal_tokens.data import collate_clouds, load_cloud_file, to_device
from areal_tokens.geometry import Cloud, canonical_cloud, prepare_cloud
from areal_tokens.io import load_model
from areal_tokens.model import ModelConfig, WeightedAttention

ROOT = Path('/workspace/runs/areals_architecture_controls_20261024')
BASE = Path('/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/run/best_normal.pt')
OUT = ROOT / 'results/finite_token_raw'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


@torch.inference_mode()
def attention_check():
    torch.manual_seed(12345)
    layer = WeightedAttention(ModelConfig()).cuda().eval()
    query, values = torch.randn(1, 31, 192, device='cuda'), torch.randn(1, 63, 192, device='cuda')
    qp, vp = torch.randn(1, 31, 3, device='cuda'), torch.randn(1, 63, 3, device='cuda')
    weights = torch.rand(1, 63, device='cuda') + .1
    # Nonuniform duplication changes counting measure but preserves surface measure.
    repeats = torch.ones(63, dtype=torch.long, device='cuda')
    repeats[:21] = 4
    indices = torch.repeat_interleave(torch.arange(63, device='cuda'), repeats)
    split_weights = weights[:, indices] / repeats[indices][None]
    result = {}
    for name, weighted in [('area', True), ('ordinary', False)]:
        before = layer(query, values, qp, vp, weights if weighted else torch.ones_like(weights))
        after = layer(query, values[:, indices], qp, vp[:, indices],
                      split_weights if weighted else torch.ones_like(split_weights))
        result[name] = dict(max_abs=float((before - after).abs().max()),
                            rms=float((before - after).square().mean().sqrt()))
    assert result['area']['max_abs'] < 2e-6
    assert result['ordinary']['max_abs'] > 1e-4
    return result


def area_sample(mesh, seed):
    rng = np.random.default_rng(seed)
    faces = rng.choice(len(mesh.faces), 4096, p=mesh.area_faces / mesh.area)
    uv = rng.random((4096, 2))
    root = np.sqrt(uv[:, 0])
    bary = np.stack([1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]], -1)
    points = (mesh.triangles[faces] * bary[..., None]).sum(1)
    normals = np.asarray(mesh.face_normals[faces])
    weights = np.full(4096, mesh.area / 4096)
    return canonical_cloud(points, normals, weights), mesh.triangles[faces], bary


def coupled_subdivision(triangles, bary):
    # Split each sampled face into four coplanar children and represent the exact same point there.
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
    children = np.stack([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
                         np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)], 1)
    which = np.full(len(bary), 3, int)
    which[bary[:, 0] >= .5] = 0
    which[bary[:, 1] >= .5] = 1
    which[bary[:, 2] >= .5] = 2
    child_bary = np.empty_like(bary)
    for i, (u, v, w) in enumerate(bary):
        child_bary[i] = ([2 * u - 1, 2 * v, 2 * w] if which[i] == 0 else
                         [2 * u, 2 * v - 1, 2 * w] if which[i] == 1 else
                         [2 * u, 2 * v, 2 * w - 1] if which[i] == 2 else
                         [1 - 2 * w, 1 - 2 * u, 1 - 2 * v])
    chosen = children[np.arange(len(bary)), which]
    points = (chosen * child_bary[..., None]).sum(1)
    normal = np.cross(chosen[:, 1] - chosen[:, 0], chosen[:, 2] - chosen[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return points, normal


def prepared(cloud, tokens):
    arrays = prepare_cloud(cloud, tokens, 4096, np.random.default_rng(0), tokens * 8)
    arrays.update(id='', family='', split='', complete=np.ones(1, np.float32))
    return arrays


@torch.inference_mode()
def predict(model, sample, queries):
    batch = to_device(collate_clouds([sample]), torch.device('cuda'))
    q = torch.as_tensor(queries[None], dtype=torch.float32, device='cuda')
    packet = model.encoder(batch)
    # FP32 reduces arithmetic noise in the invariance assay.
    decoded = model.decode(packet, q)
    raw = F.softplus(model.distance_head(model.query_features(packet, q)).float().squeeze(-1)) * packet.scale
    return decoded['udf'][0].cpu().numpy(), decoded['normals'][0].cpu().numpy(), raw[0].cpu().numpy()


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.preferred_linalg_library('magma')
    gate = attention_check()
    write(OUT / 'attention_control.json', gate)
    print('ATTENTION_CONTROL', json.dumps(gate), flush=True)
    records = json.loads((ROOT / 'architecture_manifest.json').read_text())['records']
    records = sorted((r for r in records if r['split'] == 'test'),
                     key=lambda r: hashlib.sha256(r['id'].encode()).hexdigest())[:32]
    model, config, state = load_model(BASE, 'cuda')
    rows = []
    for number, record in enumerate(records, 1):
        cloud, target = load_cloud_file(record['path'])
        queries = target['queries'][:1024]
        truth, true_normals = target['udf'][:1024], target['query_normals'][:1024]
        true_normals /= np.maximum(np.linalg.norm(true_normals, axis=1, keepdims=True), 1e-12)
        mesh = trimesh.load(record['source_mesh'], force='mesh', process=False, skip_materials=True)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[mesh.area_faces > 0], process=False)
        assert len(mesh.faces) and np.isfinite(mesh.vertices).all()
        first, triangles, bary = area_sample(mesh, 101)
        split_points, split_normals = coupled_subdivision(triangles, bary)
        original_points = (triangles * bary[..., None]).sum(1)
        scale = float(np.sqrt(np.mean(np.sum((original_points - original_points.mean(0)) ** 2, axis=1))))
        subdivision_error = float(np.abs(original_points - split_points).max() / scale)
        assert subdivision_error < 1e-10
        split = canonical_cloud(split_points, split_normals, np.full(4096, mesh.area / 4096))
        for tokens in [256, 1024]:
            base = prepared(first, tokens)
            reference = predict(model, base, queries)
            fixed_scale = float(base['scale'][0])
            raw_arrays = dict(queries=queries, truth=truth, query_normals=true_normals,
                              scale=np.asarray(fixed_scale), normal_band=np.asarray(config.data.normal_band))
            trials = [('reference', 101, base)]
            permuted = dict(base)
            for key in ['points', 'normals', 'point_weights']:
                permuted[key] = base[key][::-1].copy()
            trials.append(('point_order_permutation', 101, permuted))
            # The coupled subdivision uses the same surface samples, a separate control from resampling.
            trials.append(('subdivision_coupled', 101, prepared(split, tokens)))
            for seed in [202, 303]:
                alternate, _, _ = area_sample(mesh, seed)
                p = prepared(alternate, tokens)
                trials.append(('independent_resample', seed, p))
                fixed = dict(p)
                world = p['points'] * p['scale'] + p['center']
                fixed['points'] = ((world - base['center']) / base['scale']).astype(np.float32)
                for key in ['anchors', 'anchor_normals', 'anchor_weights', 'center', 'scale', 'area',
                            'surfels', 'surfel_normals', 'surfel_weights']:
                    if key in base:
                        fixed[key] = base[key]
                trials.append(('resample_fixed_anchors_frame_surfels', seed, fixed))
            for mode, seed, sample in trials:
                prediction = reference if mode == 'reference' else predict(model, sample, queries)
                d, n, raw = prediction
                prefix = mode + '_' + str(seed)
                raw_arrays[prefix + '_distance'] = d
                raw_arrays[prefix + '_normal'] = n
                raw_arrays[prefix + '_head'] = raw
                valid = truth / fixed_scale <= config.data.normal_band
                angle = np.degrees(np.arccos(np.clip(np.abs((n * true_normals).sum(-1)), 0, 1)))
                row = dict(id=record['id'], family=record['family'], tokens=tokens, mode=mode, seed=seed,
                    distance_nmse=float(np.mean(((d - truth) / fixed_scale) ** 2)),
                    normal_degrees=float(angle[valid].mean()) if valid.any() else None,
                    output_change=float(np.sqrt(np.mean(((d - reference[0]) / fixed_scale) ** 2))),
                    learned_head_change=float(np.sqrt(np.mean(((raw - reference[2]) / fixed_scale) ** 2))),
                    subdivision_coordinate_error=subdivision_error)
                rows.append(row)
            np.savez_compressed(OUT / (record['id'] + f'_{tokens}.npz'), **raw_arrays)
        write(OUT / 'rows.json', rows)
        print('FINITE_TOKEN', number, '/', len(records), record['id'], flush=True)
    summary = {}
    for tokens in [256, 1024]:
        summary[str(tokens)] = {}
        for mode in sorted({r['mode'] for r in rows}):
            chosen = [r for r in rows if r['tokens'] == tokens and r['mode'] == mode]
            summary[str(tokens)][mode] = {key: float(np.mean([r[key] for r in chosen]))
                for key in ['distance_nmse', 'output_change', 'learned_head_change']}
    write(OUT / 'summary.json', dict(objects=32, settings=summary,
         attention=gate))


if __name__ == '__main__':
    main()
