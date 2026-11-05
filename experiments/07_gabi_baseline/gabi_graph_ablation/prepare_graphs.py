import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import heapq
import io
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from common import ROOT, DATA, records, write_json

REVISION = '9a1943ee5df9ff65cfcb7d0051fde43f42d4742c'
REPOSITORY = 'yirens/double-delta-aero'

class RangedFile(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.downloaded = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, self.size)[whence] + offset
        assert 0 <= self.position <= self.size
        return self.position

    def read(self, count=-1):
        count = min(self.size - self.position, count if count >= 0 else self.size)
        if not count:
            return b''
        assert count <= 64 << 20
        start, end = self.position, self.position + count - 1
        for attempt in range(5):
            request = urllib.request.Request(self.url + f'?range={start}-{end}&attempt={attempt}',
                headers={'Range': f'bytes={start}-{end}', 'Cache-Control': 'no-cache'})
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    assert response.status == 206, response.status
                    assert response.headers['Content-Range'] == f'bytes {start}-{end}/{self.size}'
                    payload = response.read(count + 1)
                assert len(payload) == count
                self.downloaded += count
                self.position += count
                return payload
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)

def archive_url(row):
    return f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{row["path"]}'

def inspect_archive(row):
    remote = RangedFile(archive_url(row), row['size'])
    with zipfile.ZipFile(remote) as archive:
        entries = [n for n in archive.namelist() if n.endswith('/15.0degAOA/surface_flow.vtu')]
    return dict(archive=row, entries=entries, inventory_bytes=remote.downloaded)

def inventory():
    path = ROOT / 'raw/archive_inventory.json'
    if path.exists():
        return json.loads(path.read_text())
    archives = []
    for folder in ['trainingSet', 'holdoutSet']:
        url = f'https://huggingface.co/api/datasets/{REPOSITORY}/tree/{REVISION}/{folder}?limit=1000'
        with urllib.request.urlopen(url, timeout=45) as response:
            listing = json.load(response)
        archives.extend(r for r in listing if r['path'].endswith('.zip'))
    result = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for future in as_completed([executor.submit(inspect_archive, r) for r in archives]):
            row = future.result()
            result.append(row)
            print(json.dumps({'archive': row['archive']['path'], 'reference_meshes': len(row['entries'])}), flush=True)
    write_json(path, result)
    return result

def download_mesh(task):
    grid, row, entry = task
    path = ROOT / 'raw' / (grid + '.vtu')
    receipt_path = ROOT / 'raw' / (grid + '.json')
    if path.exists() and receipt_path.exists():
        return json.loads(receipt_path.read_text())
    remote = RangedFile(archive_url(row), row['size'])
    with zipfile.ZipFile(remote) as archive:
        info = archive.getinfo(entry)
        assert info.compress_size < 64 << 20 and info.file_size < 128 << 20
        payload = archive.read(info)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(payload)
    temporary.replace(path)
    receipt = dict(grid_id=grid, repository=REPOSITORY, revision=REVISION,
                   archive=row['path'], archive_size=row['size'], entry=entry,
                   bytes=len(payload), downloaded_bytes=remote.downloaded)
    write_json(receipt_path, receipt)
    return receipt

def read_mesh(path):
    raw = path.read_bytes()
    appended = raw.index(b'<AppendedData')
    header = ET.fromstring(raw[:appended] + b'</VTKFile>')
    assert header.attrib['byte_order'] == 'LittleEndian' and header.attrib['header_type'] == 'UInt64'
    assert 'compressor' not in header.attrib
    assert b'encoding="raw"' in raw[appended:appended+50]
    start = raw.index(b'_', appended) + 1
    types = {'Float32': '<f4', 'Float64': '<f8', 'Int32': '<i4', 'Int64': '<i8', 'UInt8': 'u1'}
    def array(node):
        offset = start + int(node.attrib['offset'])
        size = int(np.frombuffer(raw, dtype='<u8', count=1, offset=offset)[0])
        dtype = np.dtype(types[node.attrib['type']])
        return np.frombuffer(raw, dtype=dtype, count=size // dtype.itemsize, offset=offset + 8)
    piece = header.find('./UnstructuredGrid/Piece')
    points = array(piece.find('./Points/DataArray')).reshape(-1, 3).copy()
    cells = {n.attrib['Name']: array(n) for n in piece.findall('./Cells/DataArray')}
    assert np.all(cells['types'] == 5) and np.all(np.diff(np.r_[0, cells['offsets']]) == 3)
    faces = cells['connectivity'].reshape(-1, 3).copy()
    assert len(points) == int(piece.attrib['NumberOfPoints']) and len(faces) == int(piece.attrib['NumberOfCells'])
    assert np.isfinite(points).all() and faces.min() >= 0 and faces.max() < len(points)
    return points, faces

def undirected(pairs):
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]

def directed(pairs):
    return np.ascontiguousarray(np.concatenate([pairs, pairs[:, ::-1]]).T, dtype=np.int64)

def describe(pairs, normals):
    n = len(normals)
    edges = directed(pairs)
    graph = coo_matrix((np.ones(edges.shape[1]), edges), shape=(n, n)).tocsr()
    components, labels = connected_components(graph, directed=False)
    degree = np.diff(graph.indptr)
    return dict(nodes=n, undirected_edges=len(pairs), connected_components=int(components),
                largest_component=int(np.bincount(labels).max()), isolated_nodes=int(np.sum(degree == 0)),
                minimum_degree=int(degree.min()), mean_degree=float(degree.mean()),
                opposite_normal_fraction=float(np.mean(np.sum(normals[pairs[:,0]] * normals[pairs[:,1]], axis=1) < -.5)))

def build_graph(record):
    started = time.monotonic()
    grid = record['grid_id']
    cache, receipt_path = ROOT / 'graphs' / (grid + '.npz'), ROOT / 'graphs' / (grid + '.json')
    if cache.exists() and receipt_path.exists():
        return json.loads(receipt_path.read_text())
    with np.load(DATA / 'fields' / (record['id'] + '.npz')) as field:
        query, normals = field['points'].copy(), field['normals'].copy().astype(np.float64)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    points, faces = read_mesh(ROOT / 'raw' / (grid + '.vtu'))
    p64 = points.astype(np.float64)
    cross = np.cross(p64[faces[:,1]] - p64[faces[:,0]], p64[faces[:,2]] - p64[faces[:,0]])
    native_normals = np.zeros_like(p64)
    for i in range(3):
        np.add.at(native_normals, faces[:,i], cross * .5)
    native_normals /= np.linalg.norm(native_normals, axis=1)[:,None]
    assert np.isfinite(native_normals).all()
    tree = cKDTree(points)
    distance, selected = tree.query(query, workers=1)
    assert np.all(distance == 0), (grid, float(distance.max()))
    # Distinct vertices can share coordinates at seams; disambiguate using supplied surface normals.
    for i in np.flatnonzero(np.max(np.abs(native_normals[selected] - normals), axis=1) > 1e-6):
        candidates = np.asarray(tree.query_ball_point(query[i], 0))
        errors = np.max(np.abs(native_normals[candidates] - normals[i]), axis=1)
        matches = candidates[errors < 1e-6]
        assert len(matches) == 1, (grid, int(i), errors.tolist())
        selected[i] = matches[0]
    normal_error = float(np.max(np.abs(native_normals[selected] - normals)))
    assert normal_error < 1e-6 and len(np.unique(selected)) == len(selected)
    assert np.array_equal(points[selected], query)
    mesh_pairs = undirected(np.concatenate([faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]]))
    lengths = np.linalg.norm(p64[mesh_pairs[:,0]] - p64[mesh_pairs[:,1]], axis=1)
    edges = directed(mesh_pairs)
    adj = coo_matrix((np.r_[lengths,lengths], edges), shape=(len(points),len(points))).tocsr()
    lookup = np.full(len(points), -1, dtype=np.int64)
    lookup[selected] = np.arange(len(selected))
    neighbors = [list(zip(adj.indices[adj.indptr[i]:adj.indptr[i+1]].tolist(),
                          adj.data[adj.indptr[i]:adj.indptr[i+1]].tolist())) for i in range(len(points))]
    surface = []
    for source_id, source in enumerate(selected.tolist()):
        queue, distance, visited, found = [(0., source)], {source: 0.}, set(), 0
        while queue and found < 8:
            dist, vertex = heapq.heappop(queue)
            if vertex in visited:
                continue
            visited.add(vertex)
            target = int(lookup[vertex])
            if target >= 0 and target != source_id:
                surface.append((source_id, target))
                found += 1
            for nxt, weight in neighbors[vertex]:
                candidate = dist + weight
                if candidate < distance.get(nxt, float('inf')):
                    distance[nxt] = candidate
                    heapq.heappush(queue, (candidate, nxt))
        assert found == 8, (grid, source_id, found)
    surface = undirected(np.asarray(surface))
    _, nearest = cKDTree(query).query(query, k=9, workers=1)
    euclidean = undirected(np.stack([np.repeat(np.arange(len(query)),8), nearest[:,1:].reshape(-1)],axis=1))
    diagnostics = {kind: describe(pairs, normals) for kind,pairs in [('euclidean',euclidean),('surface',surface)]}
    assert diagnostics['surface']['isolated_nodes'] == 0
    assert diagnostics['surface']['connected_components'] == 1, (grid, diagnostics)
    with cache.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, points=query, selected_native_vertices=selected,
                            euclidean=directed(euclidean), surface=directed(surface))
    cache.with_suffix('.tmp').replace(cache)
    receipt = dict(grid_id=grid, representative_case=record['id'], split=record['split'],
                   maximum_normal_error=normal_error,
                   raw_vertices=len(points), raw_faces=len(faces), diagnostics=diagnostics,
                   seconds=time.monotonic()-started)
    write_json(receipt_path,receipt)
    return receipt

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    representative={}
    for r in records():
        representative.setdefault(r['grid_id'],r)
    inv=inventory()
    locations={}
    for row in inv:
        for entry in row['entries']:
            grid=entry.split('/')[-3]
            assert grid not in locations, grid
            locations[grid]=(row['archive'],entry)
    assert not (set(representative)-set(locations)), sorted(set(representative)-set(locations))
    tasks=[(grid,*locations[grid]) for grid in sorted(representative)]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i,future in enumerate(as_completed([executor.submit(download_mesh,t) for t in tasks]),1):
            receipt=future.result()
            print(json.dumps({'downloaded':i,'total':len(tasks),'grid':receipt['grid_id']}),flush=True)
    built=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(build_graph,r) for r in representative.values()]):
            receipt=future.result()
            built.append(receipt)
            print(json.dumps({'built':len(built),'total':len(tasks),'grid':receipt['grid_id'],
                              'diagnostics':receipt['diagnostics']}),flush=True)
    write_json(ROOT/'graphs/complete.json',dict(status='PASS',geometries=len(built),
               definition='Eight nearest retained vertices by weighted shortest paths on original mesh edges, symmetrised; geometry only.',
               source_revision=REVISION,rows=sorted(built,key=lambda r:r['grid_id'])))

if __name__=='__main__':
    main()
