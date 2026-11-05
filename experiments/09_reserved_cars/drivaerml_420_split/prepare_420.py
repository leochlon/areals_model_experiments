"""Add 356 DrivAerML training cars using the original surface preparation recipe."""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import multiprocessing
import os
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@lru_cache(maxsize=1)
def legacy():
    pilot = Path("/workspace/home/drivaerml_transfer_pilot_20260925T121525Z")
    spec = importlib.util.spec_from_file_location("original_drivaerml_preparation", pilot / "prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, json.loads((pilot / "config.json").read_text())


def save_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(path)


def prepare_case(task):
    case, destination = task
    data = Path(destination)
    record_path = data / "data_records" / f"{case}.json"
    geometry_path = data / "geometry" / f"{case}.npz"
    field_path = data / "fields" / f"{case}.npz"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        assert record["case"] == case and record["split"] == "train"
        return record
    module, original = legacy()
    storage = module.Storage(original)
    raw = data / "raw_tmp" / f"{case}.fields.h5"
    started = time.monotonic()
    try:
        source = storage.fetch(original["source_prefix"] + case + "/fields.h5", raw)
        with h5py.File(raw, "r") as packed:
            points = packed["mesh.verts"][:].astype(np.float64)
            normals = packed["mesh.verts_normals"][:].astype(np.float64)
            raw_weights = packed["mass"][:].astype(np.float64).reshape(-1)
            pressure = packed["mesh.PressureCoeff"][:].astype(np.float32).reshape(-1)
            face_area = float(packed["mesh.areas"][:].astype(np.float64).sum())
        assert points.ndim == 2 and points.shape[1] == 3 and normals.shape == points.shape
        assert len(points) == len(raw_weights) == len(pressure)
        assert all(np.isfinite(v).all() for v in (points, normals, raw_weights, pressure))
        assert (raw_weights >= 0).all() and raw_weights.sum() > 0
        assert np.isclose(raw_weights.sum(), face_area, rtol=1e-4)
        weights = raw_weights.copy()
        lengths = np.linalg.norm(normals, axis=-1)
        invalid = (weights > 0) & (lengths < 1e-8)
        excluded = float(weights[invalid].sum() / weights.sum())
        assert excluded < 1e-5, f"Invalid-normal surface fraction too large: {case}: {excluded}"
        weights[invalid] = 0
        positive = weights > 0
        normals[positive] /= lengths[positive, None]
        cloud = module.canonical_cloud(points, normals, weights)
        geometry = module.prepare_cloud(cloud, original["geometry_latents"], original["encoder_points"],
                                        np.random.default_rng(original["sampling_seed"] + int(case.split("_")[-1])))
        # These are exactly the conversions in the old two-stage pilot/native
        # preparation. The one pass retains the full Cp field from the same H5.
        world = (geometry["points"].astype(np.float64) * geometry["scale"].astype(np.float64)
                 + geometry["center"].astype(np.float64)).astype(np.float32)
        sampled_normals = geometry["normals"].astype(np.float32)
        sampled_weights = geometry["point_weights"].astype(np.float32)
        assert len(world) == len(sampled_normals) == len(sampled_weights)
        assert np.isfinite(world).all() and np.isfinite(sampled_normals).all()
        assert (sampled_weights > 0).all()
        np.testing.assert_allclose(np.linalg.norm(sampled_normals, axis=-1), 1.0, atol=2e-5)
        save_npz(geometry_path, points=world, normals=sampled_normals, weights=sampled_weights)
        save_npz(field_path, points=points.astype(np.float32), values=pressure, weights=raw_weights.astype(np.float32))
        record = dict(case=case, split="train", source=source, reused=False,
                      geometry_points=len(world), field_vertices=len(points),
                      zero_normal_vertices_excluded=int(invalid.sum()), excluded_surface_area_fraction=excluded,
                      surface_area=float(weights.sum()), pressure_min=float(pressure.min()),
                      pressure_max=float(pressure.max()), seconds=time.monotonic() - started,
                      geometry_sampling_seed=original["sampling_seed"] + int(case.split("_")[-1]))
        write_json(record_path, record)
        return record
    finally:
        storage.close()
        raw.unlink(missing_ok=True)


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    data = Path(cfg["data_root"])
    for name in ("geometry", "fields", "data_records", "raw_tmp"):
        (data / name).mkdir(parents=True, exist_ok=True)
    old = Path(cfg["comparison_run"])
    pilot = Path(cfg["pilot_root"])
    original_manifest = json.loads((old / "manifest.json").read_text())
    split = json.loads((pilot / "manifest.json").read_text())
    _, original_config = legacy()
    split_root = Path(original_config["split_root"])
    pool = set((split_root / "train_design_ids.txt").read_text().split())
    train, val, test = set(split["train"]), set(split["validation"]), set(split["benchmark_test_excluded"])
    assert len(pool) == 436 and len(train) == 64 and len(val) == 16 and len(test) == 48
    assert train <= pool and val <= pool and not (train & val or pool & test)
    additional = sorted(pool - train - val, key=lambda case: int(case.split("_")[-1]))
    assert len(additional) == 356
    originals = original_manifest["records"]
    old_train = [r for r in originals if r["split"] == "train"]
    old_val = [r for r in originals if r["split"] == "val"]
    assert [r["id"] for r in old_val] == split["validation"]
    records = old_train + [dict(id=case, family=case, split="train", path=f"geometry/{case}.npz")
                           for case in additional] + old_val
    assert len(records) == len({r["id"] for r in records}) == 436
    assert {r["id"] for r in records if r["split"] == "train"} == pool - val
    manifest = {**original_manifest, "records": records}
    manifest_path = data / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        write_json(manifest_path, manifest)
    reused = []
    for record in originals:
        case = record["id"]
        metadata = json.loads((old / "data_records" / f"{case}.json").read_text())
        for subdir in ("geometry", "fields"):
            source = old / subdir / f"{case}.npz"
            link = data / subdir / source.name
            assert source.is_file()
            if link.is_symlink():
                assert link.resolve() == source.resolve()
            else:
                assert not link.exists()
                link.symlink_to(source)
        metadata = {**metadata, "reused": True}
        write_json(data / "data_records" / f"{case}.json", metadata)
        reused.append(metadata)
    write_json(data / "split.json", dict(train=[r["id"] for r in records if r["split"] == "train"],
        validation=split["validation"], benchmark_test_excluded=split["benchmark_test_excluded"],
        original_train=split["train"], additional_train=additional))
    started = time.monotonic()
    prepared = []
    def progress(phase):
        state = dict(phase=phase, utc=datetime.now(timezone.utc).isoformat(), reused=80, added_complete=len(prepared), added_total=356,
                     total_complete=80+len(prepared), total=436, workers=cfg["prepare_workers"],
                     seconds=round(time.monotonic()-started, 2))
        write_json(ROOT / "preparation_progress.json", state)
        print(json.dumps(state), flush=True)
    progress("preparing")
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["prepare_workers"],
                                               mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(prepare_case, (case, str(data))): case for case in additional}
        for future in concurrent.futures.as_completed(futures):
            try:
                prepared.append(future.result())
            except BaseException as error:
                write_json(ROOT / "preparation_failure.json", dict(case=futures[future], error=f"{type(error).__name__}: {error}"))
                raise
            if len(prepared) % 8 == 0 or len(prepared) == 356:
                progress("preparing")
    expected = {r["id"] + ".npz" for r in records}
    assert {e.name for e in os.scandir(data / "geometry") if e.name.endswith(".npz")} == expected
    assert {e.name for e in os.scandir(data / "fields") if e.name.endswith(".npz")} == expected
    assert [r for r in records if r["split"] == "val"] == old_val
    report = dict(train_cars=420, validation_cars=16, held_out_test_cars=48,
                  total_prepared=436, reused=80, newly_prepared=356,
                  manifest=str(manifest_path), source_bytes_read=sum(r["source"]["bytes"] for r in prepared),
                  maximum_excluded_surface_area_fraction=max(r["excluded_surface_area_fraction"] for r in prepared),
                  seconds=round(time.monotonic()-started,2), records=reused+prepared)
    write_json(data / "preparation.json", report)
    write_json(ROOT / "data_ready.json", {k:v for k,v in report.items() if k != "records"})
    progress("complete")


if __name__ == "__main__":
    main()
