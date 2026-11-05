import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path(json.loads((ROOT/'protocol.json').read_text())['remote_baseline'])
sys.path[:0] = [str(BASE/'dependencies'),str(BASE)]
import gudhi
import numpy as np
import measure as original

TOL = 1e-6
CAP = .4


def save(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def diagrams(field):
    c = gudhi.CubicalComplex(top_dimensional_cells=field)
    c.compute_persistence(homology_coeff_field=2,min_persistence=0)
    result = [c.persistence_intervals_in_dimension(k) for k in range(3)]
    assert [int(np.isinf(x[:,1]).sum()) for x in result] == [1,0,0]
    return result


def capped(diagram):
    x = diagram.copy()
    x[:,0] = np.minimum(x[:,0],CAP)
    finite = np.isfinite(x[:,1])
    x[finite,1] = np.minimum(x[finite,1],CAP)
    return x[x[:,1] > x[:,0]]


def betti(bars,radius):
    return [int(np.sum((b[:,0]<=radius)&(radius<b[:,1]))) for b in bars]


def field_metrics(identity):
    print('COMPARE_START',identity,flush=True)
    source = BASE/'results'/identity
    target = ROOT/'results'/identity
    meta = json.loads((source/'metadata.json').read_text())
    prior = json.loads((source/'metrics.json').read_text())
    prior = {(x['grid_n'],x['radius']):x for x in prior}
    data = np.load(source/'fields.npz')
    candidate = np.load(target/'fields.npz')
    disk = candidate['disk']
    scalar_rows, persistent_rows, stored = [],[],{}
    for stride in [1,2]:
        fields = {'exact':data['true'][::stride,::stride,::stride], 'full':candidate['predicted'][::stride,::stride,::stride], 'disk':disk[::stride,::stride,::stride]}
        n,h = len(fields['exact']),float(data['h'])*stride
        nodes, attachments = original.endpoints(fields['exact'],data['landmarks'],data['low'],h)
        ds = {key:diagrams(value) for key,value in fields.items()}
        for name,bars in ds.items():
            for k,b in enumerate(bars):stored[f'{name}_{n}_h{k}'] = b
        for name in ['full','disk']:
            epsilon = float(np.max(np.abs(fields[name]-fields['exact'])))
            capped_error = float(np.max(np.abs(np.minimum(fields[name],CAP)-np.minimum(fields['exact'],CAP))))
            for k in range(3):
                d = float(gudhi.bottleneck_distance(ds['exact'][k],ds[name][k],e=TOL))
                dc = float(gudhi.bottleneck_distance(capped(ds['exact'][k]),capped(ds[name][k]),e=TOL))
                assert np.isfinite(d) and np.isfinite(dc)
                assert d <= epsilon+2*TOL+1e-12
                assert dc <= capped_error+2*TOL+1e-12
                persistent_rows.append(dict(id=identity,group=meta['group'],grid_n=n,field=name,dimension=k,bottleneck=d,bottleneck_capped=dc,epsilon_grid=epsilon,epsilon_capped=capped_error,reference_bars=len(ds['exact'][k]),candidate_bars=len(ds[name][k]),reference_essential=int(np.isinf(ds['exact'][k][:,1]).sum()),candidate_essential=int(np.isinf(ds[name][k][:,1]).sum()),stability_bound_pass=True))
        for radius in original.RADII:
            reference = fields['exact'] <= radius
            area_ref = float(reference.sum()*h**3/(2*radius))
            reference_paths, reference_missing = original.paths(reference,nodes,h)
            node_valid = reference[tuple(nodes.T)]
            pair_valid = np.array([node_valid[a] and node_valid[b] for a,b in original.PAIRS])
            connected = np.isfinite(reference_paths)
            ref_betti = betti(ds['exact'],radius)
            for name in ['full','disk']:
                occupied = fields[name] <= radius
                area = float(occupied.sum()*h**3/(2*radius))
                lengths, missing = original.paths(occupied,nodes,h)
                lost = connected&~np.isfinite(lengths)
                invented = pair_valid&~connected&np.isfinite(lengths)
                both = connected&np.isfinite(lengths)
                errors = np.abs(lengths[both]-reference_paths[both])/reference_paths[both]
                b = betti(ds[name],radius)
                row = dict(id=identity,group=meta['group'],field=name,grid_n=n,radius=float(radius),h=h,area_true=area_ref,area_predicted=area,area_relative_error=abs(area-area_ref)/area_ref,reference_area_bias=abs(area_ref-meta['surface_area'])/meta['surface_area'],predicted_area_surface_error=abs(area-meta['surface_area'])/meta['surface_area'],betti_true=ref_betti,betti_predicted=b,betti_max_error=int(np.max(np.abs(np.array(b)-ref_betti))),path_reference=original.finite_list(reference_paths),path_predicted=original.finite_list(lengths),reference_connected_pairs=int(connected.sum()),unresolved_reference_pairs=int((~pair_valid).sum()),lost_pairs=int(lost.sum()),false_connections=int(invented.sum()),missing_predicted_endpoints=missing,path_error_infinite=bool(lost.any()),path_max_finite_relative_error=float(errors.max()) if len(errors) else None,path_max_relative_error=float(errors.max()) if len(errors) and not lost.any() else None,max_endpoint_attachment=float(attachments.max()),predicted_boundary_contact=original.on_boundary(occupied))
                if name=='full':
                    previous = prior[(n,float(radius))]
                    for key in ['area_true','betti_true','path_reference','unresolved_reference_pairs']:
                        assert row[key]==previous[key],(identity,n,radius,key)
                scalar_rows.append(row)
    np.savez_compressed(target/'diagrams.npz',**stored)
    save(target/'metrics.json',scalar_rows)
    save(target/'persistence.json',persistent_rows)
    print('COMPARE_DONE',identity,flush=True)
    return scalar_rows,persistent_rows


def controls():
    a = np.array([[0.,1.],[0.,np.inf]])
    b = np.array([[.125,1.125],[.125,np.inf]])
    identity_error = float(gudhi.bottleneck_distance(a,a,e=TOL))
    assert identity_error <= 2*TOL
    assert abs(gudhi.bottleneck_distance(a,b,e=TOL)-.125)<=2*TOL
    assert abs(gudhi.bottleneck_distance(np.array([[0.,.2]]),np.empty((0,2)),e=TOL)-.1)<=2*TOL
    axis = np.linspace(-1.6,1.6,25)
    q = np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
    field = np.abs(np.linalg.norm(q,axis=-1)-1)
    full,cap = diagrams(field),diagrams(np.minimum(field,CAP))
    for k in range(3):assert gudhi.bottleneck_distance(capped(full[k]),cap[k],e=TOL)<=2*TOL
    save(ROOT/'results'/'persistence_validation.json',dict(status='PASS',identity=True,identity_numerical_error=identity_error,essential_constant_shift=True,matching_to_diagonal=True,diagram_cap_matches_capped_field=True,bottleneck_additive_tolerance=TOL))


def main():
    controls()
    ids = [p.parent.name for p in sorted((BASE/'results').glob('*/metadata.json'))]
    assert len(ids)==11
    with ProcessPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(field_metrics,ids))
    metrics = [row for a,b in result for row in a]
    pers = [row for a,b in result for row in b]
    save(ROOT/'results'/'all_metrics.json',metrics)
    save(ROOT/'results'/'persistence.json',pers)
    for name,rows in [('metrics',metrics),('persistence',pers)]:
        with (ROOT/'results'/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    summary=[]
    for group in ['heldout','control']:
        for field in ['full','disk']:
            for n in [97,49]:
                for r in original.RADII:
                    a=[x for x in metrics if x['group']==group and x['field']==field and x['grid_n']==n and x['radius']==r]
                    e=[x['path_max_finite_relative_error'] for x in a if x['path_max_finite_relative_error'] is not None]
                    summary.append(dict(group=group,field=field,grid_n=n,radius=float(r),shapes=len(a),max_area_relative_error=max(x['area_relative_error'] for x in a),max_area_surface_error=max(x['predicted_area_surface_error'] for x in a),max_betti_error=max(x['betti_max_error'] for x in a),topology_matching_meshes=sum(x['betti_true']==x['betti_predicted'] for x in a),max_finite_path_relative_error=max(e) if e else None,path_error_infinite=any(x['path_error_infinite'] for x in a),lost_pairs=sum(x['lost_pairs'] for x in a),false_connections=sum(x['false_connections'] for x in a),unresolved_reference_pairs=sum(x['unresolved_reference_pairs'] for x in a),boundary_meshes=sum(x['predicted_boundary_contact'] for x in a)))
    save(ROOT/'results'/'summary.json',summary)
    save(ROOT/'results'/'analysis_provenance.json',dict(gudhi_version=gudhi.__version__,metric_rows=len(metrics),persistence_rows=len(pers),all_original_reference_metrics_reproduced=True,all_discrete_stability_bounds_passed=True))
    print('COMPARE_COMPLETE',len(metrics),len(pers),flush=True)


if __name__=='__main__':main()
