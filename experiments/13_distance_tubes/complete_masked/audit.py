import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
BASE = Path(json.loads((ROOT/'protocol.json').read_text())['remote_baseline'])


def load(path):
    return json.loads(path.read_text())


def digital_betti(mask):
    counts = [0,0,0,int(mask.sum())]
    for dimension in [0,1,2]:
        for extended in itertools.combinations(range(3),3-dimension):
            shape = tuple(n+int(k in extended) for k,n in enumerate(mask.shape))
            occupied = np.zeros(shape,dtype=bool)
            for sides in itertools.product([0,1],repeat=len(extended)):
                padding = [(0,0)]*3
                for k,side in zip(extended,sides):padding[k]=(side,1-side)
                occupied |= np.pad(mask,padding)
            counts[dimension] += int(occupied.sum())
    euler = counts[0]-counts[1]+counts[2]-counts[3]
    b0 = ndimage.label(mask,structure=np.ones((3,3,3)))[1]
    b2 = ndimage.label(~np.pad(mask,1),structure=ndimage.generate_binary_structure(3,1))[1]-1
    return [int(b0),int(b0+b2-euler),int(b2)]


def main():
    rows = load(ROOT/'results/all_metrics.json')
    distance = load(ROOT/'results/distance_metrics.json')
    cases = load(ROOT/'results/cases.json')
    assert len(rows)==396 and len(cases)==11 and len(distance)==88
    assert len({(x['id'],x['grid_n'],x['field'],x['radius']) for x in rows})==396
    checks, bands = [], []
    topology_checks = 0
    for case in cases:
        identity = case['id']
        directory = ROOT/'results'/identity
        reference = np.load(BASE/'results'/identity/'fields.npz')
        fields = np.load(directory/'fields.npz')
        for name,key in [('full','predicted'),('disk','disk')]:
            value = fields[key]
            assert np.isfinite(value).all() and (value>=0).all()
            measured = float(np.max(np.abs(value-reference['true'])))
            expected = next(x['max_absolute_error'] for x in distance if x['id']==identity and x['field']==name and x['domain']=='grid')
            assert abs(measured-expected)<1e-12
            for radius in [.05,.1,.2,.4]:
                selection = reference['true']<=radius
                bands.append(dict(id=identity,group=case['group'],field=name,true_distance_band=radius,queries=int(selection.sum()),max_absolute_error=float(np.abs(value[selection]-reference['true'][selection]).max())))
        for row in [x for x in rows if x['id']==identity]:
            stride = 1 if row['grid_n']==97 else 2
            field = fields['predicted' if row['field']=='full' else 'disk'][::stride,::stride,::stride]
            mask = field<=np.float64(row['radius'])
            area = float(mask.sum()*row['h']**3/(2*row['radius']))
            assert abs(area-row['area_predicted'])<1e-9
            p = np.array([np.inf if x is None else x for x in row['path_predicted']])
            t = np.array([np.inf if x is None else x for x in row['path_reference']])
            lost = np.isfinite(t)&~np.isfinite(p)
            assert int(lost.sum())==row['lost_pairs']
            if row['grid_n']==97 and row['radius'] in [.1,.2,.28]:
                assert digital_betti(mask)==row['betti_predicted'],(identity,row['field'],row['radius'])
                topology_checks += 1
        surface = np.load(directory/'surface_queries.npz')
        prepared = np.load(directory/'prepared.npz')
        visible = surface['visible']
        world = surface['queries']*case['evaluation_scale']+case['evaluation_center']
        weights = surface['weights'][visible]
        center = np.average(world[visible],axis=0,weights=weights)
        scale = np.sqrt(np.average(np.sum((world[visible]-center)**2,axis=1),weights=weights))
        assert np.max(np.abs(center-prepared['center']))<1e-6*case['evaluation_scale']
        assert abs(scale-float(prepared['scale'][0]))<1e-6*case['evaluation_scale']
        centres = prepared['surfels']*prepared['scale']+prepared['center']
        error = cKDTree(world[visible]).query(centres)[0].max()/case['evaluation_scale']
        assert error<1e-5
        assert prepared['complete'].item()==0
        checks.append(dict(id=identity,visible_frame_reproduced=True,disk_centres_from_visible_only=True))
    tests = ET.parse(ROOT/'results/tests.xml').getroot()
    suites = list(tests.iter('testsuite'))
    assert sum(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0)) for x in suites)==0
    regression = load(ROOT/'results/regression.json')
    assert len(regression)==11 and all(x['complete_exactly_equals_native_disks'] and x['masked_all_outputs_unchanged'] and x['complete_normals_unchanged'] for x in regression)
    summary = dict(status='PASS',cases=11,metric_rows=396,distance_rows=88,independent_topology_checks=topology_checks,area_and_lost_paths_recomputed=True,visible_frame_and_disk_provenance_checks=checks,tests_passed=sum(int(x.attrib['tests']) for x in suites),complete_and_partial_regressions_passed=True)
    (ROOT/'results/audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'results/distance_bands.json').write_text(json.dumps(bands,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='visible_frame_and_disk_provenance_checks'}),flush=True)


if __name__=='__main__':
    main()
