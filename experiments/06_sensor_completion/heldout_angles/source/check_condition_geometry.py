import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/workspace/runs/areals_architecture_controls_20261024')
STUDY = ROOT / 'unseen_conditions'
sys.path.insert(0, '/workspace/runs/areals_multicondition_wings_20261023')
import field_benchmark as fb


def digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        x = np.asarray(array)
        h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()


def main():
    records = json.loads((STUDY/'manifest.json').read_text())['records']
    references = json.loads((STUDY/'conditions_reference_only.json').read_text())['cases']
    groups, grids = defaultdict(list), defaultdict(list)
    geometry_cache = {}
    for record in records:
        path = STUDY/record['path']
        stat = path.stat()
        inode = (stat.st_dev,stat.st_ino)
        if inode not in geometry_cache:
            geometry_cache[inode] = hashlib.sha256(path.read_bytes()).hexdigest()
        field = fb.load_field(STUDY/'fields', record['id'])
        row = dict(id=record['id'],angle=references[record['id']]['aoa'],grid_id=record['grid_id'],
              split=record['split'], geometry_sha256=geometry_cache[inode],
              surface_geometry_sha256=digest(field.points,field.normals,field.weights),
              pressure_sha256=digest(field.values))
        groups[record['family']].append(row)
        grids[record['family'],record['grid_id']].append(row)
    for family, rows in groups.items():
        assert len({r['split'] for r in rows}) == 1
        assert len({r['geometry_sha256'] for r in rows}) == 1, ('mesh changes with angle', family)
    for grid, rows in grids.items():
        assert len({r['surface_geometry_sha256'] for r in rows}) == 1, ('surface grid changes with angle', grid)
        assert len({r['pressure_sha256'] for r in rows}) == len(rows), ('duplicate pressure labels',grid)
        assert len({r['angle'] for r in rows}) == len(rows)
    # Content hashes compare files and arrays: one encoder geometry per physical group, one surface grid per
    # source grid across its angles, and a different pressure field at every angle.
    result = dict(passed=True,cases=len(records),physical_geometries=len(groups),
                  unique_geometry_files=len(geometry_cache),source_surface_grids=len(grids),groups=dict(groups))
    (ROOT/'results/condition_geometry_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)


if __name__ == '__main__':
    main()
