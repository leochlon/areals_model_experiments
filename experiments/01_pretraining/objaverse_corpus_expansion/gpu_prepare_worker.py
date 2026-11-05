"""Native Objaverse sampling, with CUDA triangle queries and guarded CPU repair."""

import argparse
import json
import time
from pathlib import Path

import igl
import numpy as np
import warp as wp

import geom_latents.prepare as preparation
from geom_latents.data import stable_seed
from areal_tokens.validation import inspect_record

ROOT = Path(__file__).resolve().parent


@wp.kernel
def closest_kernel(mesh: wp.uint64, queries: wp.array(dtype=wp.vec3),
                   distances: wp.array(dtype=float), faces: wp.array(dtype=int)):
    i = wp.tid()
    result = wp.mesh_query_point_no_sign(mesh, queries[i], 1000000.0)
    if result.result:
        point = wp.mesh_eval_position(mesh, result.face, result.u, result.v)
        distances[i] = wp.length(queries[i] - point)
        faces[i] = result.face
    else:
        distances[i] = 1000000.0
        faces[i] = -1


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class TriangleQueries:
    def __init__(self, device, worker):
        self.device = device
        self.worker = worker
        self.calls = 0
        self.last = {}

    def __call__(self, mesh, query, *, backend="igl", query_chunk=2048):
        vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        triangles = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        query = np.ascontiguousarray(query, dtype=np.float64)
        lower, upper = vertices.min(0), vertices.max(0)
        center = lower + (upper - lower) * 0.5
        scale = float(np.max(upper - lower))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid triangle-mesh coordinate scale")
        started = time.monotonic()
        points = wp.array(np.asarray((vertices - center) / scale, dtype=np.float32),
                          dtype=wp.vec3, device=self.device)
        indices = wp.array(triangles.reshape(-1).astype(np.int32), dtype=wp.int32, device=self.device)
        gpu_mesh = wp.Mesh(points=points, indices=indices)
        queries = wp.array(np.asarray((query - center) / scale, dtype=np.float32),
                           dtype=wp.vec3, device=self.device)
        gpu_distances = wp.empty(len(query), dtype=wp.float32, device=self.device)
        gpu_faces = wp.empty(len(query), dtype=wp.int32, device=self.device)
        wp.launch(closest_kernel, dim=len(query),
                  inputs=[gpu_mesh.id, queries, gpu_distances, gpu_faces], device=self.device)
        distance = gpu_distances.numpy().astype(np.float64) * scale
        closest = gpu_faces.numpy().astype(np.int64)
        # Warp omits nearly degenerate triangles. Query a conservative superset
        # of those faces with libigl and merge their exact distances back in.
        tri = vertices[triangles]
        edge0, edge1, edge2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], tri[:, 2] - tri[:, 1]
        max_edge_squared = np.maximum.reduce([(edge * edge).sum(1) for edge in (edge0, edge1, edge2)])
        cross = np.cross(edge0, edge1)
        slender = np.flatnonzero((cross * cross).sum(1) <= 1e-4 * max_edge_squared ** 2)
        if len(slender):
            squared, face, _ = igl.point_mesh_squared_distance(query, vertices, triangles[slender])
            repair = np.sqrt(np.maximum(squared, 0))
            chosen = (repair < distance) | (closest < 0)
            distance[chosen] = repair[chosen]
            closest[chosen] = slender[np.asarray(face, dtype=np.int64)[chosen]]
        # Check the first full mesh per worker and a fixed spread of queries on
        # every later mesh. Any mismatch uses exact libigl targets for that mesh.
        checked = np.arange(len(query)) if self.calls == 0 else np.linspace(0, len(query) - 1, 128, dtype=int)
        squared, reference_faces, _ = igl.point_mesh_squared_distance(query[checked], vertices, triangles)
        reference = np.sqrt(np.maximum(squared, 0))
        max_error = float(np.max(np.abs(distance[checked] - reference)) / scale)
        fallback = bool((closest < 0).any() or not np.isfinite(distance).all() or max_error > 1e-5)
        if fallback:
            if len(checked) != len(query):
                squared, reference_faces, _ = igl.point_mesh_squared_distance(query, vertices, triangles)
            distance = np.sqrt(np.maximum(squared, 0))
            closest = np.asarray(reference_faces, dtype=np.int64)
        assert np.isfinite(distance).all() and (distance >= 0).all()
        assert ((closest >= 0) & (closest < len(triangles))).all()
        self.calls += 1
        self.last = dict(backend="warp_cuda_with_igl_sliver_repair", warp_version=wp.__version__,
                         cpu_reference_queries=len(checked), max_reference_error_over_extent=max_error,
                         cpu_full_fallback=fallback, repaired_triangle_candidates=len(slender),
                         triangles=len(triangles), distance_seconds=time.monotonic() - started)
        return distance, closest


def main(args):
    plan = json.loads((ROOT / f"tasks_{args.worker:02d}.json").read_text())
    quota = plan["quota"]
    journal = ROOT / "records" / f"worker_{args.worker:02d}.jsonl"
    failure_log = ROOT / "records" / f"failures_{args.worker:02d}.jsonl"
    existing = []
    if journal.exists():
        for line in journal.read_text().splitlines():
            if line.strip():
                existing.append(json.loads(line))
    completed = {r["record"]["id"] for r in existing}
    failed = set()
    if failure_log.exists():
        failed = {json.loads(line)["id"] for line in failure_log.read_text().splitlines() if line.strip()}
    assert len(completed) == len(existing) <= quota
    generated = sum(not r["record"]["cached_original"] for r in existing)
    reused = len(existing) - generated
    started = time.monotonic()
    wp.config.kernel_cache_dir = str(ROOT / "cache" / f"warp_{args.worker:02d}")
    wp.init()
    device = wp.get_device(f"cuda:{args.device}")
    assert device.is_cuda
    queries = TriangleQueries(device, args.worker)
    preparation.closest_faces = queries

    def progress(phase, **extra):
        write_json(ROOT / f"progress_{args.worker:02d}.json", dict(
            phase=phase, worker=args.worker, gpu=str(device), gpu_name=device.name,
            completed=len(completed), target=quota, generated=generated, reused=reused,
            failures=len(failed), elapsed_seconds=time.monotonic() - started, **extra))

    progress("starting")
    with journal.open("a", buffering=1) as log, failure_log.open("a", buffering=1) as errors:
        for record in plan["records"]:
            if len(completed) >= quota:
                break
            if record["id"] in completed or record["id"] in failed:
                continue
            progress("preparing", current_id=record["id"])
            try:
                output = Path(record["path"])
                details = None
                if not record["cached_original"] and not output.exists():
                    output.parent.mkdir(parents=True, exist_ok=True)
                    details = preparation.prepare_mesh(
                        record["source_mesh"], output, points=16384, queries=32768,
                        seed=stable_seed(17, record["id"], 0), backend="igl")
                    details.update(queries.last)
                audit = inspect_record(record, require_targets=True)
                if "error" in audit:
                    raise ValueError(audit["error"])
                assert audit["queries"] == 32768
                item = dict(record=record, audit=audit, preparation=details)
                log.write(json.dumps(item, allow_nan=False) + "\n")
                completed.add(record["id"])
                if record["cached_original"]:
                    reused += 1
                else:
                    generated += 1
            except Exception as error:
                failed.add(record["id"])
                errors.write(json.dumps(dict(id=record["id"], path=record["path"],
                    error=f"{type(error).__name__}: {error}")) + "\n")
            progress("preparing")
    if len(completed) != quota:
        progress("failed", error="Candidate list exhausted before reaching the quota")
        raise RuntimeError("Insufficient valid candidates")
    progress("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--device", type=int, required=True)
    main(parser.parse_args())
