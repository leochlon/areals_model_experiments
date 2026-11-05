"""Prepare a reproducible, area-sampled DrivAerML pilot from existing S3 data."""

import argparse
import concurrent.futures
import json
import time
from pathlib import Path

import h5py
import numpy as np

from geom_latents.geometry import canonical_cloud, prepare_cloud


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def make_manifest(cfg):
    source = Path(cfg["split_root"])
    train_path = source / "train_design_ids.txt"
    test_path = source / "test_design_ids.txt"
    pool = sorted(set(train_path.read_text().split()), key=lambda s: int(s.split("_")[-1]))
    test = set(test_path.read_text().split())
    assert not (set(pool) & test), "Original training and test cases overlap"
    order = np.random.default_rng(cfg["split_seed"]).permutation(len(pool))
    need = cfg["train_cases"] + cfg["validation_cases"]
    assert need <= len(pool)
    selected = [pool[i] for i in order[:need]]
    return {
        "train": selected[: cfg["train_cases"]],
        "validation": selected[cfg["train_cases"] :],
        "benchmark_test_excluded": sorted(test),
        "source_training_pool_size": len(pool),
        "selection_seed": cfg["split_seed"],
        "source_prefix": cfg["source_prefix"],
    }


def prepare_one(args):
    cfg, case, split = args
    root = Path(cfg["run_root"])
    output = root / "data" / (case + ".npz")
    record_path = root / "data" / (case + ".json")
    if output.exists() and record_path.exists():
        return json.loads(record_path.read_text())
    raw_path = root / "raw_tmp" / (case + ".fields.h5")
    scalar_path = root / "raw_tmp" / (case + ".scalars.h5")
    start = time.monotonic()
    storage = Storage(cfg)
    try:
        fields_source = storage.fetch(cfg["source_prefix"] + case + "/fields.h5", raw_path)
        scalar_source = storage.fetch(cfg["source_prefix"] + case + "/scalars.h5", scalar_path)
        with h5py.File(raw_path, "r") as data:
            points = data["mesh.verts"][:].astype(np.float64)
            normals = data["mesh.verts_normals"][:].astype(np.float64)
            weights = data["mass"][:].astype(np.float64).reshape(-1)
            pressure = data["mesh.PressureCoeff"][:].astype(np.float32).reshape(-1, 1)
            face_area = float(data["mesh.areas"][:].astype(np.float64).sum())
        with h5py.File(scalar_path, "r") as data:
            conditions = np.array([float(data["upstream_velocity"][()]) / 40,
                                   float(data["reference_area"][()]) / 2.17], dtype=np.float32)
        assert len(points) == len(normals) == len(weights) == len(pressure)
        assert all(np.isfinite(v).all() for v in (points, normals, weights, pressure, conditions))
        assert (weights >= 0).all() and weights.sum() > 0
        assert np.isclose(weights.sum(), face_area, rtol=1e-4), "Vertex and face area disagree"
        normal_lengths = np.linalg.norm(normals, axis=-1)
        invalid = (weights > 0) & (normal_lengths < 1e-8)
        excluded_fraction = float(weights[invalid].sum() / weights.sum())
        # A few source meshes have two opposite, coincident faces whose vertex normals cancel.
        # Their measured area fraction is below 6.1e-7. Exclude only that ambiguous measure,
        # identically for both arms; never invent an orientation or alter the source mesh.
        assert excluded_fraction < 1e-5, f"Too much surface has invalid normals: {case}"
        weights[invalid] = 0
        positive = weights > 0
        normals[positive] /= normal_lengths[positive, None]
        case_number = int(case.split("_")[-1])
        cloud = canonical_cloud(points, normals, weights)
        geometry = prepare_cloud(cloud, cfg["geometry_latents"], cfg["encoder_points"],
                                 np.random.default_rng(cfg["sampling_seed"] + case_number))
        rng = np.random.default_rng(cfg["sampling_seed"] + 100000 + case_number)
        chosen = rng.choice(len(points), size=cfg["field_points"], replace=True, p=weights / weights.sum())
        arrays = {
            "surface_points": points[chosen].astype(np.float32),
            "surface_normals": normals[chosen].astype(np.float32),
            "surface_weights": np.full((len(chosen), 1), weights.sum() / len(chosen), dtype=np.float32),
            "pressure_coeff": pressure[chosen],
            "source_vertex_indices": chosen.astype(np.int64),
            "conditions": conditions,
            **{"geometry_" + k: v for k, v in geometry.items()},
        }
        with open(output.with_suffix(".npz.tmp"), "wb") as stream:
            np.savez(stream, **arrays)
        output.with_suffix(".npz.tmp").replace(output)
        record = {
            "case": case, "split": split, "source_vertices": len(points),
            "field_points": len(chosen), "distinct_sample_vertices": len(np.unique(chosen)),
            "surface_area": float(weights.sum()), "pressure_min": float(pressure.min()),
            "zero_normal_vertices_excluded": int(invalid.sum()),
            "excluded_surface_area_fraction": excluded_fraction,
            "pressure_max": float(pressure.max()), "geometry_points": len(geometry["points"]),
            "geometry_latents": len(geometry["anchors"]), "sources": [fields_source, scalar_source],
            "prepared_bytes": output.stat().st_size,
            "seconds": time.monotonic() - start,
        }
        write_json(record_path, record)
        return record
    finally:
        storage.close()
        raw_path.unlink(missing_ok=True)
        scalar_path.unlink(missing_ok=True)


def prepare(config_path, limit=None):
    cfg = json.loads(Path(config_path).read_text())
    root = Path(cfg["run_root"])
    (root / "data").mkdir(exist_ok=True)
    (root / "raw_tmp").mkdir(exist_ok=True)
    manifest = make_manifest(cfg)
    existing = root / "manifest.json"
    if existing.exists():
        assert json.loads(existing.read_text()) == manifest, "manifest.json holds a different split"
    else:
        write_json(existing, manifest)
    jobs = [(cfg, case, split) for split in ("train", "validation") for case in manifest[split]]
    if limit is not None:
        jobs = jobs[:limit]
    started = time.monotonic()
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg["prepare_workers"]) as pool:
        futures = [pool.submit(prepare_one, args) for args in jobs]
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(json.dumps({"event": "prepared", "done": len(records), "total": len(jobs),
                              "case": record["case"], "seconds": round(time.monotonic() - started, 2)}), flush=True)
    audit = {"cases": len(records), "seconds": time.monotonic() - started,
             "prepared_bytes": sum(r["prepared_bytes"] for r in records),
             "source_bytes_read": sum(s["bytes"] for r in records for s in r["sources"]),
             "records": sorted(records, key=lambda r: r["case"])}
    write_json(root / "preparation.json", audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    prepare(args.config, args.limit)
