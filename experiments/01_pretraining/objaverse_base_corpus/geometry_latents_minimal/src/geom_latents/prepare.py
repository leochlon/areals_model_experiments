"""Analytical smoke data and optional triangle-mesh supervision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import SCHEMA
from .io import write_json


def save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Output must have the .npz suffix")
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def sphere_targets(queries: np.ndarray, radius: float):
    lengths = np.linalg.norm(queries, axis=1)
    normals = queries / np.maximum(lengths[:, None], 1e-12)
    normals[lengths < 1e-12] = [1, 0, 0]
    return np.abs(lengths - radius), normals


def box_targets(queries: np.ndarray, half: np.ndarray):
    difference = np.abs(queries) - half
    signed = np.linalg.norm(np.maximum(difference, 0), axis=1) + np.minimum(difference.max(axis=1), 0)
    axis = difference.argmax(axis=1)
    normals = np.zeros_like(queries)
    sign = np.where(queries[np.arange(len(queries)), axis] >= 0, 1, -1)
    normals[np.arange(len(queries)), axis] = sign
    return np.abs(signed), normals


def query_distribution(points, normals, count, rng, near_fraction=0.7, sigma=0.04):
    scale = float(np.sqrt(np.mean(np.sum((points - points.mean(0)) ** 2, axis=1))))
    near_count = round(count * near_fraction)
    indices = rng.integers(len(points), size=near_count)
    near = points[indices] + rng.normal(0, sigma * scale, (near_count, 1)) * normals[indices]
    near += rng.normal(0, sigma * scale * 0.15, near.shape)
    far = rng.uniform(points.min(0) - 0.2 * scale, points.max(0) + 0.2 * scale, (count - near_count, 3))
    queries = np.concatenate([near, far])
    rng.shuffle(queries)
    return queries


def make_demo(output: str | Path, *, train=12, val=4, test=4, points=512, queries=1024, seed=11):
    """Generate varied spheres/boxes with exact analytical distance targets.

    A family is one independently generated geometric instance. This dataset is
    for software verification; it is not a benchmark for vehicle understanding.
    """
    output = Path(output)
    if min(train, val, test) < 1 or points < 16 or queries < 8:
        raise ValueError("Each split needs a shape, with >=16 surface points and >=8 queries")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Demo output must be empty: {output}")
    rng = np.random.default_rng(seed)
    records = []
    for split, count in (("train", train), ("val", val), ("test", test)):
        for index in range(count):
            identity = f"{split}_{index:04d}"
            if index % 2 == 0:
                radius = rng.uniform(0.6, 1.4)
                normals = rng.normal(size=(points, 3))
                normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                surface = radius * normals
                area = 4 * np.pi * radius**2
                query = query_distribution(surface, normals, queries, rng)
                udf, query_normals = sphere_targets(query, radius)
                kind = "sphere"
            else:
                half = rng.uniform([0.6, 0.25, 0.25], [1.8, 0.8, 0.9])
                face_areas = 4 * np.array([half[1] * half[2], half[0] * half[2], half[0] * half[1]])
                area = 2 * face_areas.sum()
                axis = rng.choice(3, points, p=face_areas / face_areas.sum())
                sign = rng.choice([-1.0, 1.0], points)
                surface = rng.uniform(-half, half, (points, 3))
                surface[np.arange(points), axis] = half[axis] * sign
                normals = np.zeros_like(surface)
                normals[np.arange(points), axis] = sign
                query = query_distribution(surface, normals, queries, rng)
                udf, query_normals = box_targets(query, half)
                kind = "box"
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            rotation[:, 0] *= np.linalg.det(rotation)
            translation = rng.uniform(-0.4, 0.4, 3)
            save_npz(
                output / f"{identity}.npz",
                points=(surface @ rotation.T + translation).astype(np.float32),
                normals=(normals @ rotation.T).astype(np.float32),
                weights=np.full(points, area / points, dtype=np.float32),
                queries=(query @ rotation.T + translation).astype(np.float32),
                udf=udf.astype(np.float32),
                query_normals=(query_normals @ rotation.T).astype(np.float32),
            )
            records.append(dict(id=identity, family=identity, split=split, path=f"{identity}.npz", kind=kind))
    manifest = output / "manifest.json"
    write_json(
        manifest,
        dict(
            schema=SCHEMA,
            units="synthetic_length",
            records=records,
            target_source="analytical sphere/box UDF; nearest-face outward normals",
        ),
    )
    return manifest


DISTANCE_BACKENDS = ("auto", "trimesh", "igl")


def resolve_backend(backend: str) -> str:
    if backend not in DISTANCE_BACKENDS:
        raise ValueError(f"backend must be one of {DISTANCE_BACKENDS}")
    if backend == "auto":
        try:
            import igl  # noqa: F401

            return "igl"
        except ImportError:
            return "trimesh"
    return backend


def closest_faces(mesh, query: np.ndarray, *, backend="trimesh", query_chunk=2048):
    """Unsigned point-to-triangle distance and the closest face for each query.

    The igl backend (libigl AABB tree) is exact and much faster on large meshes;
    trimesh's candidate search misses the true face for a few near-surface
    queries per mesh (errors around 1e-5 of the shape size). Where two faces are
    exactly equidistant, which happens whenever the closest point lies on a
    shared edge, either backend may pick either face.
    """
    if query_chunk < 1:
        raise ValueError("query_chunk must be positive")
    backend = resolve_backend(backend)
    if backend == "igl":
        import igl

        squared, faces, _ = igl.point_mesh_squared_distance(
            np.ascontiguousarray(query, dtype=np.float64),
            np.ascontiguousarray(mesh.vertices, dtype=np.float64),
            np.ascontiguousarray(mesh.faces, dtype=np.int64),
        )
        return np.sqrt(np.maximum(np.asarray(squared, dtype=np.float64), 0)), np.asarray(faces, dtype=np.int64)
    import trimesh

    distances, faces = [], []
    try:
        for start in range(0, len(query), query_chunk):
            _, distance, closest_face = trimesh.proximity.closest_point(mesh, query[start : start + query_chunk])
            distances.append(distance)
            faces.append(closest_face)
    except ModuleNotFoundError as error:
        raise ImportError("Triangle proximity requires the mesh extra, including rtree") from error
    return np.concatenate(distances), np.concatenate(faces)


def prepare_mesh(
    input_path: str | Path,
    output_path: str | Path,
    *,
    points=16384,
    queries=32768,
    seed=17,
    query_chunk=2048,
    backend="trimesh",
) -> dict:
    """Area-sample a triangle mesh and compute point-to-triangle UDF targets.

    Orientation is preserved, not repaired. Open meshes are allowed for UDF;
    usable normal supervision still requires consistently oriented faces (or
    the unoriented normal loss). The report says whether the winding was even
    consistent, which watertightness alone does not tell; double-sided
    duplicates pass both checks.
    """
    try:
        import trimesh
    except ImportError as error:
        raise ImportError("Mesh preparation requires: pip install 'geometry-latents[mesh]'") from error
    if points < 16 or queries < 8 or query_chunk < 1:
        raise ValueError("Need >=16 points, >=8 queries and a positive query chunk")
    backend = resolve_backend(backend)
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    mesh = trimesh.load(str(input_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("Input must contain a nonempty triangle mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh has nonfinite vertices")
    positive = mesh.area_faces > 0
    if not positive.any():
        raise ValueError("Mesh has no positive-area faces")
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[positive], process=False)
    area = float(mesh.area)
    rng = np.random.default_rng(seed)
    face = rng.choice(len(mesh.faces), points, p=mesh.area_faces / area)
    uv = rng.random((points, 2))
    root = np.sqrt(uv[:, 0])
    barycentric = np.stack([1 - root, root * (1 - uv[:, 1]), root * uv[:, 1]], axis=-1)
    surface = (mesh.triangles[face] * barycentric[:, :, None]).sum(1)
    normals = np.asarray(mesh.face_normals[face])
    query = query_distribution(surface, normals, queries, rng)
    udf, closest = closest_faces(mesh, query, backend=backend, query_chunk=query_chunk)
    save_npz(
        output_path,
        points=surface.astype(np.float32),
        normals=normals.astype(np.float32),
        weights=np.full(points, area / points, dtype=np.float32),
        queries=query.astype(np.float32),
        udf=udf.astype(np.float32),
        query_normals=np.asarray(mesh.face_normals[closest]).astype(np.float32),
    )
    return dict(
        input=str(input_path),
        output=str(output_path),
        area=area,
        points=points,
        queries=queries,
        watertight=bool(mesh.is_watertight),
        winding_consistent=bool(mesh.is_winding_consistent),
        backend=backend,
        target_source="point-to-triangle unsigned distance",
    )


def _prepare_job(job: dict) -> dict:
    """One record, in a worker; failures come back as data so the pool survives them."""
    try:
        details = prepare_mesh(
            job["source"],
            job["output"],
            points=job["points"],
            queries=job["queries"],
            seed=job["seed"],
            backend=job["backend"],
        )
        return dict(id=job["id"], ok=True, details=details)
    except Exception as error:  # noqa: BLE001 - reported per record
        return dict(id=job["id"], ok=False, error=f"{type(error).__name__}: {error}")


def prepare_mesh_manifest(
    source: str | Path,
    output: str | Path,
    *,
    points=16384,
    queries=32768,
    seed=17,
    workers=1,
    resume=False,
    bad_records="raise",
    backend="trimesh",
):
    """Convert a geometry manifest whose record paths point to meshes, not NPZ.

    Records are independent (each is seeded from its id), so they may be
    converted by several worker processes and in any order. With resume=True a
    partly converted directory is continued: records whose file exists are kept.
    With bad_records="skip" unconvertible meshes are listed in failures.json and
    left out of the written manifest instead of stopping the conversion.
    """
    from .data import read_manifest, stable_seed
    from .parallel import map_jobs

    if bad_records not in ("raise", "skip"):
        raise ValueError("bad_records must be raise or skip")
    backend = resolve_backend(backend)
    records, metadata = read_manifest(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ledger = output / "prepare.records.json"
    expected = [[index, record["id"]] for index, record in enumerate(records)]
    if resume:
        if ledger.exists() and json.loads(ledger.read_text()) != expected:
            raise ValueError("The output directory was started from a different manifest")
    elif any(output.iterdir()):
        raise FileExistsError(f"Mesh output must be empty: {output}")
    write_json(ledger, expected)
    names = {record["id"]: f"shape_{index:07d}.npz" for index, record in enumerate(records)}
    jobs = [
        dict(
            id=record["id"],
            source=record["path"],
            output=str(output / names[record["id"]]),
            points=points,
            queries=queries,
            seed=stable_seed(seed, record["id"], 0),
            backend=backend,
        )
        for record in records
        if not (resume and (output / names[record["id"]]).exists())
    ]
    results = map_jobs(_prepare_job, jobs, workers, ordered=False)
    failures = {}
    for result in results:
        if result["ok"]:
            print(json.dumps(dict(id=result["id"], **result["details"])), flush=True)
        else:
            failures[result["id"]] = result["error"]
            print(json.dumps(dict(id=result["id"], error=result["error"])), flush=True)
    if failures and bad_records == "raise":
        raise RuntimeError(
            f"{len(failures)} mesh(es) failed, e.g. {next(iter(failures.items()))}; rerun with resume "
            "after fixing them, or convert with bad_records=skip"
        )
    converted = [
        dict(record, path=names[record["id"]], source_mesh=record["path"])
        for record in records
        if record["id"] not in failures
    ]
    if not converted:
        raise RuntimeError("No mesh could be converted")
    write_json(output / "failures.json", [dict(id=key, error=value) for key, value in failures.items()])
    manifest = output / "manifest.json"
    write_json(
        manifest,
        dict(
            schema=SCHEMA,
            units=metadata["units"],
            records=converted,
            target_source="point-to-triangle unsigned distance",
        ),
    )
    return manifest
