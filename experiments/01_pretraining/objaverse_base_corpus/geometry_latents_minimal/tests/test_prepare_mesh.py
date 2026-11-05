"""Mesh conversion: parallel, resumable, tolerant, and the same numbers as 0.2.0."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from geom_latents.data import read_manifest, stable_seed  # noqa: E402
from geom_latents.prepare import closest_faces, prepare_mesh, prepare_mesh_manifest  # noqa: E402


def write_meshes(folder: Path, *, with_bad=True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(2.0, 1.0, 1.0)).export(folder / "box.obj")
    trimesh.creation.icosphere(subdivisions=3).export(folder / "sphere.obj")
    records = [
        {"id": "box", "family": "box", "split": "train", "path": "box.obj"},
        {"id": "sphere", "family": "sphere", "split": "val", "path": "sphere.obj"},
    ]
    if with_bad:
        trimesh.Scene(trimesh.PointCloud(np.random.default_rng(0).random((50, 3)))).export(folder / "points.glb")
        records.insert(1, {"id": "points", "family": "points", "split": "train", "path": "points.glb"})
    manifest = folder / "meshes.json"
    manifest.write_text(json.dumps({"schema": "geometry_latents_v1", "units": "m", "records": records}))
    return manifest


def test_manifest_conversion_is_parallel_tolerant_and_resumable(tmp_path):
    manifest = write_meshes(tmp_path / "meshes")
    output = tmp_path / "prepared"
    with pytest.raises(RuntimeError, match="points"):
        prepare_mesh_manifest(manifest, output, points=64, queries=128, workers=2)
    # the good meshes were converted before the failure surfaced; continue without the bad one
    with pytest.raises(FileExistsError):
        prepare_mesh_manifest(manifest, output, points=64, queries=128)
    written = {path.name: path.stat().st_mtime_ns for path in output.glob("shape_*.npz")}
    assert written.keys() == {"shape_0000000.npz", "shape_0000002.npz"}
    result = prepare_mesh_manifest(manifest, output, points=64, queries=128, workers=2, resume=True, bad_records="skip")
    assert {path.name: path.stat().st_mtime_ns for path in output.glob("shape_*.npz")} == written  # not redone
    records, _ = read_manifest(result)
    assert [record["id"] for record in records] == ["box", "sphere"]
    failures = json.loads((output / "failures.json").read_text())
    assert [failure["id"] for failure in failures] == ["points"] and "triangle mesh" in failures[0]["error"]
    # each file is what a sequential single-mesh conversion with the record's seed gives, byte for byte
    for record in records:
        single = tmp_path / f"single_{record['id']}.npz"
        prepare_mesh(record["source_mesh"], single, points=64, queries=128, seed=stable_seed(17, record["id"], 0))
        with np.load(record["path"]) as a, np.load(single) as b:
            assert a.files == b.files and all(np.array_equal(a[key], b[key]) for key in a.files)
    # a directory started from one manifest refuses another
    other = write_meshes(tmp_path / "other", with_bad=False)
    with pytest.raises(ValueError, match="different manifest"):
        prepare_mesh_manifest(other, output, points=64, queries=128, resume=True)


def test_winding_report_distinguishes_flipped_faces(tmp_path):
    box = trimesh.creation.box()
    faces = box.faces.copy()
    faces[::2] = faces[::2][:, ::-1]
    trimesh.Trimesh(box.vertices, faces, process=False).export(tmp_path / "flipped.obj")
    details = prepare_mesh(tmp_path / "flipped.obj", tmp_path / "flipped.npz", points=64, queries=64)
    assert details["watertight"] and not details["winding_consistent"]


def test_igl_backend_matches_trimesh_distances(tmp_path):
    pytest.importorskip("igl", reason="libigl not installed")
    mesh = trimesh.creation.icosphere(subdivisions=4)
    rng = np.random.default_rng(3)
    query = rng.normal(size=(2000, 3)) * 1.2
    udf_trimesh, faces_trimesh = closest_faces(mesh, query, backend="trimesh", query_chunk=512)
    udf_igl, faces_igl = closest_faces(mesh, query, backend="igl")
    assert np.allclose(udf_trimesh, udf_igl, atol=1e-9)
    # Where the backends name different faces, both faces are exactly equidistant
    # (the closest point is on their shared edge or vertex), so neither is wrong;
    # 0.2.0's own choice among tied faces was equally arbitrary.
    ties = 0
    for i in np.flatnonzero(faces_trimesh != faces_igl):
        both = trimesh.triangles.closest_point(
            mesh.triangles[[faces_trimesh[i], faces_igl[i]]], np.repeat(query[i][None], 2, 0)
        )
        distances = np.linalg.norm(both - query[i], axis=1)
        assert abs(distances[0] - distances[1]) < 1e-9
        ties += 1
    assert ties < len(query)
