import argparse
import csv
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'dependencies'))
import gudhi
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

P = json.loads((ROOT/'protocol.json').read_text())
RADII = np.asarray(P['radii'])
PAIRS = list(itertools.combinations(range(6), 2))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def topologies(field):
    complex_ = gudhi.CubicalComplex(top_dimensional_cells=field)
    complex_.compute_persistence(homology_coeff_field=2, min_persistence=0)
    bars = [complex_.persistence_intervals_in_dimension(k) for k in range(3)]
    return [[int(np.sum((b[:, 0] <= r) & (r < b[:, 1]))) for b in bars] for r in RADII]


def endpoints(true, landmarks, low, h):
    indices = np.rint((landmarks-low)/h).astype(np.int64)
    assert (indices >= 0).all() and (indices < np.array(true.shape)).all()
    dist = np.linalg.norm(low+indices*h-landmarks, axis=1)
    assert float(dist.max()) <= np.sqrt(3)*h/2+1e-10
    assert len(np.unique(indices, axis=0)) == 6
    return indices, dist


def paths(mask, nodes, h):
    ids = np.full(mask.shape, -1, dtype=np.int32)
    ids[mask] = np.arange(int(mask.sum()), dtype=np.int32)
    attached = ids[tuple(nodes.T)]
    rows, cols, weights = [], [], []
    for offset in itertools.product([-1, 0, 1], repeat=3):
        if offset <= (0, 0, 0):
            continue
        left = tuple(slice(max(0, -d), min(s, s-d)) for s, d in zip(mask.shape, offset))
        right = tuple(slice(max(0, d), min(s, s+d)) for s, d in zip(mask.shape, offset))
        a, b = ids[left], ids[right]
        keep = (a >= 0) & (b >= 0)
        rows.append(a[keep])
        cols.append(b[keep])
        weights.append(np.full(int(keep.sum()), h*np.linalg.norm(offset), dtype=np.float64))
    distances = np.full((6, 6), np.inf)
    if mask.any() and (attached >= 0).any():
        graph = coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(int(mask.sum()), int(mask.sum()))).tocsr()
        valid = np.flatnonzero(attached >= 0)
        d = dijkstra(graph, directed=False, indices=attached[valid])
        distances[np.ix_(valid, valid)] = d[:, attached[valid]]
    return np.asarray([distances[a, b] for a, b in PAIRS]), int(np.sum(attached < 0))


def finite_list(x):
    return [float(v) if np.isfinite(v) else None for v in x]


def on_boundary(mask):
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any() or mask[:, :, 0].any() or mask[:, :, -1].any())


def measure_case(directory):
    directory = Path(directory)
    print('MEASURE_START', directory.name, flush=True)
    info = json.loads((directory/'metadata.json').read_text())
    packed = np.load(directory/'fields.npz')
    result = []
    for stride in [1, P['coarse_stride']]:
        true = packed['true'][::stride, ::stride, ::stride]
        pred = packed['predicted'][::stride, ::stride, ::stride]
        h = float(packed['h'])*stride
        betti_true, betti_pred = topologies(true), topologies(pred)
        nodes, attachments = endpoints(true, packed['landmarks'], packed['low'], h)
        epsilon = float(np.max(np.abs(pred-true)))
        for i, radius in enumerate(RADII):
            reference, learned = true <= radius, pred <= radius
            a_true = float(reference.sum()*h**3/(2*radius))
            a_pred = float(learned.sum()*h**3/(2*radius))
            ref_paths, ref_missing = paths(reference, nodes, h)
            pred_paths, pred_missing = paths(learned, nodes, h)
            connected = np.isfinite(ref_paths)
            lost = connected & ~np.isfinite(pred_paths)
            ref_node_valid = reference[tuple(nodes.T)]
            ref_pair_valid = np.array([ref_node_valid[a] and ref_node_valid[b] for a,b in PAIRS])
            invented = ref_pair_valid & ~connected & np.isfinite(pred_paths)
            both = connected & np.isfinite(pred_paths)
            rel = np.abs(pred_paths[both]-ref_paths[both])/ref_paths[both]
            # Closed-sublevel inclusions on the same grid, including roundoff.
            left_violations = int(np.sum((true <= radius-epsilon-1e-12) & ~learned))
            right_violations = int(np.sum(learned & (true > radius+epsilon+1e-12)))
            assert left_violations == right_violations == 0
            result.append(dict(id=info['id'], group=info['group'], grid_n=len(true), radius=float(radius), h=h, area_true=a_true, area_predicted=a_pred, true_surface_area=info['surface_area'], area_relative_error=abs(a_pred-a_true)/a_true if a_true else None, reference_area_bias=abs(a_true-info['surface_area'])/info['surface_area'], predicted_area_surface_error=abs(a_pred-info['surface_area'])/info['surface_area'], betti_true=betti_true[i], betti_predicted=betti_pred[i], betti_max_error=int(np.max(np.abs(np.array(betti_pred[i])-betti_true[i]))), path_reference=finite_list(ref_paths), path_predicted=finite_list(pred_paths), reference_connected_pairs=int(connected.sum()), unresolved_reference_pairs=int((~ref_pair_valid).sum()), missing_reference_endpoints=ref_missing, missing_predicted_endpoints=pred_missing, lost_pairs=int(lost.sum()), false_connections=int(invented.sum()), path_error_infinite=bool(lost.any()), path_max_finite_relative_error=float(rel.max()) if len(rel) else None, path_max_relative_error=float(rel.max()) if len(rel) and not lost.any() else None, endpoint_indices=nodes.tolist(), endpoint_attachment_distances=attachments.tolist(), max_endpoint_attachment=float(attachments.max()), exact_boundary_contact=on_boundary(reference), predicted_boundary_contact=on_boundary(learned), sandwich_violations=left_violations+right_violations, epsilon_grid=epsilon, epsilon_measured=info['epsilon_measured']))
        print('MEASURE_GRID_DONE', directory.name, len(true), flush=True)
    write_json(directory/'metrics.json', result)
    return result


def validate():
    n = 65
    axis = np.linspace(-1.6, 1.6, n)
    h = axis[1]-axis[0]
    q = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1)
    field = np.abs(np.linalg.norm(q, axis=-1)-1.)
    topology = topologies(field)
    rows = []
    for i, r in enumerate(RADII):
        observed = np.sum(field <= r)*h**3/(2*r)
        oracle = 4*np.pi*(1+r*r/3)
        rows.append(dict(radius=float(r), tube_area_relative_discretisation_error=float(abs(observed-oracle)/oracle), betti=topology[i], expected_betti=[1,0,1]))
    # The 0.14 shell is several voxels thick and has a closed-form volume and topology.
    assert topology[5] == [1,0,1]
    assert rows[5]['tube_area_relative_discretisation_error'] < .06
    shifted = field+.03
    assert np.array_equal(shifted <= .14, field <= .11)
    sphere_nodes = np.array([[52,32,32],[32,52,32],[12,32,32],[32,12,32],[32,32,52],[32,32,12]])
    d, missing = paths(field <= .14, sphere_nodes, h)
    copy, _ = paths(field.copy() <= .14, sphere_nodes, h)
    assert np.array_equal(d, copy) and np.isfinite(d).all() and missing == 0
    angles = np.asarray([np.arccos(np.dot((sphere_nodes[a]-32)/20, (sphere_nodes[b]-32)/20)) for a,b in PAIRS])
    inner = 1-.14
    oracle_paths = np.where(np.cos(angles/2) >= inner, 2*np.sin(angles/2), 2*np.sqrt(1-inner**2)+inner*(angles-2*np.arccos(inner)))
    graph_error = float(np.max(np.abs(d-oracle_paths)/oracle_paths))
    assert graph_error < .15
    # A deliberately removed corridor must not become a finite or zero error.
    cut = field <= .14
    cut[31:34] = False
    blocked, _ = paths(cut, sphere_nodes, h)
    assert (~np.isfinite(blocked)).any()
    # Independent digital topology check: 26-connected foreground and 6-connected complement.
    mask = field <= .14
    b0 = ndimage.label(mask, structure=np.ones((3,3,3)))[1]
    b2 = ndimage.label(~np.pad(mask, 1), structure=ndimage.generate_binary_structure(3,1))[1]-1
    assert b0 == topology[5][0] and b2 == topology[5][2]
    report = dict(status='PASS', sphere_grid_n=n, sphere_rows=rows, sphere_tube_path_graph_max_relative_discretisation_error=graph_error, identity_path_max_error=float(np.max(np.abs(d-copy))), shifted_distance_mask_identity=True, cut_shell_disconnections_detected=True, gudhi_b0_b2_crosscheck=True, limitations='Discrete graph distances and voxel volumes have numerical error; thin shells can have grid-induced topology.')
    write_json(ROOT/'results'/'validation.json', report)
    print('VALIDATION', json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    validate()
    if args.validate_only:
        return
    directories = sorted(p.parent for p in (ROOT/'results').glob('*/metadata.json'))
    assert len(directories) == 11, f'Need all preselected cases, found {len(directories)}'
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        groups = list(pool.map(measure_case, directories))
    rows = [r for group in groups for r in group]
    write_json(ROOT/'results'/'all_metrics.json', rows)
    columns = ['id','group','grid_n','radius','h','epsilon_measured','area_relative_error','reference_area_bias','predicted_area_surface_error','betti_true','betti_predicted','betti_max_error','path_max_relative_error','path_error_infinite','path_max_finite_relative_error','reference_connected_pairs','unresolved_reference_pairs','lost_pairs','false_connections','max_endpoint_attachment','exact_boundary_contact','predicted_boundary_contact']
    with (ROOT/'results'/'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for grid_n in [97,49]:
        for radius in RADII:
            data = [r for r in rows if r['group']=='heldout' and r['grid_n']==grid_n and r['radius']==radius]
            assert len(data) == 8
            finite_errors = [r['path_max_finite_relative_error'] for r in data if r['path_max_finite_relative_error'] is not None]
            summary.append(dict(grid_n=grid_n, radius=float(radius), max_area_relative_error=max(r['area_relative_error'] for r in data), max_reference_area_bias=max(r['reference_area_bias'] for r in data), max_predicted_area_surface_error=max(r['predicted_area_surface_error'] for r in data), max_betti_error=max(r['betti_max_error'] for r in data), topology_matching_meshes=sum(r['betti_true']==r['betti_predicted'] for r in data), path_error_infinite=any(r['path_error_infinite'] for r in data), max_finite_path_relative_error=max(finite_errors) if finite_errors else None, meshes_losing_paths=sum(r['lost_pairs']>0 for r in data), lost_pairs=sum(r['lost_pairs'] for r in data), false_connections=sum(r['false_connections'] for r in data), unresolved_reference_pairs=sum(r['unresolved_reference_pairs'] for r in data), connected_reference_pairs=sum(r['reference_connected_pairs'] for r in data), meshes_predicted_boundary_contact=sum(r['predicted_boundary_contact'] for r in data)))
    write_json(ROOT/'results'/'summary.json', summary)
    print('MEASURE_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
