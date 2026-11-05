import json,time
from pathlib import Path
import warp as wp
from gpu_prepare_worker import ROOT,TriangleQueries,preparation,stable_seed,inspect_record,write_json
wp.config.kernel_cache_dir=str(ROOT/'cache/qualification')
wp.init()
queries=TriangleQueries(wp.get_device('cuda:0'),-1)
preparation.closest_faces=queries
plan=json.loads((ROOT/'tasks_00.json').read_text())
record=next(r for r in plan['records'] if not r['cached_original'])
started=time.monotonic()
details=preparation.prepare_mesh(record['source_mesh'],record['path'],points=16384,queries=32768,seed=stable_seed(17,record['id'],0),backend='igl')
audit=inspect_record(record)
assert 'error' not in audit,audit
assert not queries.last['cpu_full_fallback'],queries.last
result=dict(passed=True,record=record,audit=audit,gpu_queries=queries.last,seconds=time.monotonic()-started)
write_json(ROOT/'qualification.json',result)
print(json.dumps(result),flush=True)
