#!/usr/bin/env python3
"""Write a known-answer field benchmark: Whittle-Matern random fields on each surface of an existing benchmark.

Every geometry in the source manifest gets random fields built from its own Laplace-Beltrami operator (no
simulation, so the fields are label-free like the geometry). The output is a manifest and a field directory in the
format field_benchmark.py reads, so the usual stages run unchanged: observations, train-areal (pretrained,
--scratch, --freeze-trunk), predict. grf_eval.py then scores the predictions against the exact posterior mean.

  python make_grf_fields.py --manifest M --fields F --out grf \
      --group-regex '^(.*)_aoa' --draws-train 8 --draws-eval 4

Records that share one geometry (DoubleDelta has nine angles per wing) are grouped automatically by the content of
their cloud files; --group-regex instead takes the geometry from a record id. --subset-train N also writes
OUT/manifest_train{N}.json with only the first N training shapes (and every validation and test record), for the
low-data comparison; validation problems drawn from it are identical to the full manifest's. Validation and test shapes also get their eigenbasis
written to OUT/bases, which grf_eval.py needs. The Laplacian is built on --lap-points area-weighted samples of the
field grid (robust_laplacian when installed: pip install robust-laplacian) and its eigenfunctions are carried to
every field point by inverse-distance interpolation; the prior is defined on that interpolated basis, so the
posterior mean grf_eval.py computes is exact for the fields written here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from surface_fields import SurfaceGRF, eigenpairs, point_cloud_laplacian


def load_npz(path):
    with np.load(path, allow_pickle=False) as packed:
        return {key: packed[key] for key in packed.files}


def basis_on_grid(points, weights, lap_points, eigs, rng):
    """Laplace-Beltrami eigenpairs of the surface sampled by `points`, carried to every point."""
    n = len(points)
    if n > lap_points:
        index = np.sort(rng.choice(n, lap_points, replace=False, p=weights / weights.sum()))
        sub_weights = np.full(lap_points, weights.sum() / lap_points)
    else:
        index, sub_weights = np.arange(n), weights
    sub = points[index]
    L, M = point_cloud_laplacian(sub, sub_weights)
    lam, phi_sub = eigenpairs(L, M, eigs)
    if len(index) == n:
        return lam, phi_sub
    dist, nearest = cKDTree(sub).query(points, 3)
    w = 1.0 / np.maximum(dist, 1e-12)
    w /= w.sum(1, keepdims=True)
    return lam, np.einsum("nj,njk->nk", w, phi_sub[nearest])


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shape_basis(job):
    """One shape's eigenbasis on its field points (run in a worker process)."""
    path, lap_points, eigs, seed = job
    field = load_npz(path)
    points = field["points"].astype(np.float64)
    weights = field["weights"].astype(np.float64).reshape(-1)
    lam, phi = basis_on_grid(points, weights, lap_points, eigs, np.random.default_rng(seed))
    return lam, phi.astype(np.float32)


def rms_radius(points, weights):
    w = weights / weights.sum()
    centre = w @ points
    return float(np.sqrt(w @ ((points - centre) ** 2).sum(1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fields", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--group-regex", default=None,
                        help="regex whose first group is the geometry id (default: group identical cloud files)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--subset-train", type=int, nargs="*", default=[],
                        help="also write manifest_train{N}.json with only the first N training shapes")
    parser.add_argument("--draws-train", type=int, default=8)
    parser.add_argument("--draws-eval", type=int, default=4)
    parser.add_argument("--eigs", type=int, default=128)
    parser.add_argument("--lap-points", type=int, default=6000)
    parser.add_argument("--ells", type=float, nargs="+", default=[0.25, 0.5, 1.0],
                        help="length scales in units of each shape's RMS radius; drawn uniformly per field")
    parser.add_argument("--nu", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = json.loads(Path(args.manifest).read_text())
    base_dir = Path(args.manifest).resolve().parent
    out = Path(args.out)
    for sub in ("fields", "clouds", "bases"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.group_regex) if args.group_regex else None
    shapes, by_digest = {}, {}
    for record in source["records"]:
        if pattern:
            key = pattern.match(record["id"]).group(1)
        else:
            # records whose cloud files are byte-identical share one geometry; the first one names the shape
            digest = file_digest((base_dir / record["path"]).resolve())
            key = by_digest.setdefault(digest, record["id"])
        if key in shapes and shapes[key]["split"] != record["split"]:
            raise SystemExit(f"shape {key} appears in two splits ({shapes[key]['split']}, {record['split']})")
        shapes.setdefault(key, record)                 # the first record of each shape supplies its geometry
    print(f"{len(source['records'])} records, {len(shapes)} distinct shapes", flush=True)
    order = sorted(shapes.items())
    jobs = [(Path(args.fields) / f"{record['id']}.npz", args.lap_points, args.eigs, args.seed + i)
            for i, (_, record) in enumerate(order)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        bases = list(pool.map(shape_basis, jobs, chunksize=1))
    rng = np.random.default_rng(args.seed)
    records, summary = [], []
    for number, ((shape, record), (lam, phi)) in enumerate(zip(order, bases), 1):
        field = load_npz(Path(args.fields) / f"{record['id']}.npz")
        points = field["points"].astype(np.float64)
        weights = field["weights"].astype(np.float64).reshape(-1)
        phi = phi.astype(np.float64)
        grf = SurfaceGRF(lam, phi, weights, ells=tuple(args.ells), nu=args.nu, scale=rms_radius(points, weights))
        evaluation = record["split"] in ("val", "test")
        draws = args.draws_eval if evaluation else args.draws_train
        values, which = grf.sample(rng, draws)
        cloud_source = (base_dir / record["path"]).resolve()
        for d in range(draws):
            identity = f"{shape}__grf{d}"
            arrays = dict(points=field["points"], values=values[d].astype(np.float32), weights=field["weights"])
            if "normals" in field:
                arrays["normals"] = field["normals"]
            np.savez(out / "fields" / f"{identity}.npz", **arrays)
            # the manifest refuses two records with one source path, so each draw gets its own hard link
            target = out / "clouds" / f"{identity}{cloud_source.suffix}"
            if not target.exists():
                try:
                    os.link(cloud_source, target)
                except OSError:
                    shutil.copy2(cloud_source, target)
            records.append(dict(id=identity, family=shape, split=record["split"],
                                path=os.path.relpath(target, out)))
        if evaluation:
            np.savez(out / "bases" / f"{shape}.npz", lam=lam, phi=phi.astype(np.float32), mass=weights,
                     ells=np.array(args.ells), nu=args.nu, scale=grf.scale)
        summary.append(dict(shape=shape, split=record["split"], points=len(points), draws=draws,
                            lambda_max=float(lam[-1])))
        print(f"{number}/{len(shapes)} {shape} ({record['split']}): {len(points)} points, {draws} fields, "
              f"lambda_1..k = {lam[1]:.3g}..{lam[-1]:.3g}", flush=True)
    manifest = {k: v for k, v in source.items() if k != "records"}
    manifest["records"] = records
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    train_shapes = sorted({r["family"] for r in records if r["split"] == "train"})
    for count in args.subset_train:
        keep = set(train_shapes[:count])
        subset = dict(manifest, records=[r for r in records if r["split"] != "train" or r["family"] in keep])
        (out / f"manifest_train{count}.json").write_text(json.dumps(subset, indent=1))
        print(f"wrote manifest_train{count}.json: {len(keep)} training shapes")
    (out / "grf_config.json").write_text(json.dumps(dict(vars(args), shapes=summary), indent=1))
    print(f"wrote {len(records)} fields for {len(shapes)} shapes to {out}")


if __name__ == "__main__":
    main()
