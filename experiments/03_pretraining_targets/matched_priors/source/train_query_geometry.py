#!/usr/bin/env python3
"""Raw meshes -> nearest-point labels (d, g, nn^T, H_r) -> AREALS-style encoder + query decoder.

Geometry pretraining with off-surface Hessian targets, built for the Figure 2 ablation.
There are no charts, surface fits or orientation fixes: every label comes from
nearest-point queries on the raw triangle mesh, so open, non-manifold and
inconsistently wound meshes are usable as they are.

Targets at a query x (eta = d^2/2, pi = nearest-point map onto the mesh):
  d    = |x - pi(x)|                 unsigned distance (Huber loss, as in AREALS)
  g    = x - pi(x) = grad eta        offset from the foot point: sign-free, zero on the surface
  P    = l l^T                       normal line; l points to the foot (face normal on the surface)
  H_r  = ball average of Hess eta    smoothed Hessian at radii r (units of the RMS radius R)
H_r is estimated without bias from nearest-point queries alone (divergence theorem):
  H_r(x) = I - 3/(2 r m) sum_k nu_k [pi(x + r nu_k) - pi(x - r nu_k)]^T, symmetrised.
Near a smooth wall it is a smoothed nn^T + tW(I + tW)^{-1}. It is <= I everywhere and
large and negative near the medial axis, so labels are eigen-clipped at --h-floor.
It is unbiased for H_r, not for the unsmoothed Hessian: below the mesh edge length it
sees facets, and curvature is identifiable only where t*kappa is not small, so
evaluation reports curvature by offset band.

Figure 2 ablation: pretrain identical models on identical data with
  --targets d,P        (the AREALS targets)
  --targets d,P,H      (adds the off-surface Hessian)
  --targets d,g,P,H    (adds the foot offset)
then fine-tune each, and a model from scratch, on the same simulation budgets.
`export` writes the trunk; make_field_model() adds zero-initialised field input columns
and a field head, so fine-tuning starts exactly from the geometry model. The field
fine-tuning loop itself is not part of this script.

Commands:
  python train_query_geometry.py selftest
  python train_query_geometry.py prepare --manifest meshes.json --out corpus.json --workers 32
  python train_query_geometry.py train --data corpus.json --out run_dPH --targets d,P,H
  python train_query_geometry.py evaluate --data corpus.json --checkpoint run_dPH/best.pt --split test
  python train_query_geometry.py export --checkpoint run_dPH/best.pt --out trunk_dPH.pt
  python train_query_geometry.py query --data corpus.json --checkpoint run_dPH/best.pt --mesh 0 --points "0,0,1"
Manifest: JSON list of {"path": "a.glb", "split": "train|val|test", "group": "family"}.
About 3M parameters by default (AREALS-sized); about 29M with
  --width 512 --heads 8 --depth 6 --decoder-depth 2.
Cost: prepare makes about 2M nearest-point queries per mesh at the defaults (install
open3d), and shards take about 1 MB per mesh. Training reads one shard per example
unless --views > 1 draws several consecutive examples from the same mesh.

Hessian labels carry Monte Carlo noise; a second, independent draw on the evaluation
queries gives the floor. The architecture follows the AREALS layout but is not the
production AREALS code.
Dependencies: numpy, scipy, trimesh (with rtree), torch >= 2.0; open3d optional, much faster.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F

VERSION = "query-geometry-1"
SYM = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))  # packed order of a symmetric 3x3
FRO = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)                      # Frobenius weights of the packed entries
ISO = (1 / 3, 1 / 3, 1 / 3, 0.0, 0.0, 0.0)                # I/3, the isotropic projector


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def config_sha(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def pack(M):
    """[..., 3, 3] symmetric -> [..., 6]; numpy or torch."""
    xp = torch if isinstance(M, torch.Tensor) else np
    return xp.stack([M[..., i, j] for i, j in SYM], -1)


def unpack(v):
    """[..., 6] -> [..., 3, 3]; numpy or torch."""
    xp = torch if isinstance(v, torch.Tensor) else np
    a, b, c, d, e, f = (v[..., k] for k in range(6))
    return xp.stack((xp.stack((a, d, e), -1), xp.stack((d, b, f), -1), xp.stack((e, f, c), -1)), -2)


# ----------------------------------------------------------------------------- labels

def fibonacci_sphere(m):
    i = np.arange(m) + 0.5
    z = 1 - 2 * i / m
    s = np.sqrt(np.clip(1 - z * z, 0, None))
    a = np.pi * (1 + 5 ** 0.5) * i
    return np.stack((s * np.cos(a), s * np.sin(a), z), -1)


def random_orthogonal(n, rng):
    """n Haar-distributed orthogonal 3x3 matrices (reflections included; the sphere is symmetric)."""
    q, r = np.linalg.qr(rng.normal(size=(n, 3, 3)))
    return q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]


def random_rotation(rng):
    q = random_orthogonal(1, rng)[0]
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def random_directions(n, m, rng):
    """[n, m, 3]: a Fibonacci design under an independent Haar rotation per query.
    The random rotation makes the design average an unbiased estimate of the sphere average."""
    return np.einsum("qij,mj->qmi", random_orthogonal(n, rng), fibonacci_sphere(m))


def smoothed_hessian(x, r, nu, closest):
    """Unbiased estimate of the radius-r ball average of Hess(eta), eta = d^2/2, at x [Q, 3].
    Divergence theorem on grad eta = y - pi(y): the linear part integrates to I and the
    antithetic pairs cancel the constant part. Valid for any closed set. nu: [Q, m, 3]."""
    q, m, _ = nu.shape
    ys = np.concatenate((x[:, None] + r * nu, x[:, None] - r * nu), 1).reshape(-1, 3)
    feet = closest(ys)[0].reshape(q, 2 * m, 3)
    G = 3.0 / (2.0 * r * m) * np.einsum("qmi,qmj->qij", nu, feet[:, :m] - feet[:, m:])
    return np.eye(3) - 0.5 * (G + np.swapaxes(G, 1, 2))


def point_labels(x, closest, face_normals, d_min):
    """d, g = x - pi(x), and the normal line P (direction to the foot; face normal on the surface)."""
    feet, faces = closest(x)
    g = x - feet
    d = np.linalg.norm(g, axis=1)
    line = g / np.maximum(d, 1e-30)[:, None]
    on = d <= d_min
    line[on] = face_normals[faces[on]]
    return d, g, line[:, :, None] * line[:, None, :]


def clip_eigenvalues(H, floor):
    """Clip eigenvalues below `floor`; returns (clipped matrices, which ones were clipped)."""
    w, V = np.linalg.eigh(H)
    return np.einsum("...ij,...j,...kj->...ik", V, np.maximum(w, floor), V), (w < floor).any(-1)


class Closest:
    """Nearest points on a triangle soup: open3d (fast, float32) if available, else trimesh."""

    def __init__(self, vertices, faces, backend="auto", chunk=1 << 18):
        self.chunk, self.backend = chunk, backend
        if backend in ("auto", "open3d"):
            try:
                import open3d as o3d
                self.o3d = o3d
                self.scene = o3d.t.geometry.RaycastingScene()
                self.scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(vertices, np.float32)),
                                         o3d.core.Tensor(np.ascontiguousarray(faces, np.uint32)))
                self.backend = "open3d"
            except Exception:  # missing or incompatible open3d: fall back unless it was requested
                if backend == "open3d":
                    raise
                self.backend = "trimesh"
        if self.backend == "trimesh":
            import trimesh
            self.query = trimesh.proximity.ProximityQuery(trimesh.Trimesh(vertices, faces, process=False))
        elif self.backend != "open3d":
            raise ValueError(f"unknown nearest-point backend {backend!r}")

    def __call__(self, y):
        feet, faces = [], []
        for s in range(0, len(y), self.chunk):
            part = np.ascontiguousarray(y[s:s + self.chunk])
            if self.backend == "open3d":
                found = self.scene.compute_closest_points(self.o3d.core.Tensor(part.astype(np.float32)))
                feet.append(found["points"].numpy().astype(np.float64))
                faces.append(found["primitive_ids"].numpy().astype(np.int64))
            else:
                p, _, f = self.query.on_surface(part)
                feet.append(np.asarray(p, np.float64))
                faces.append(np.asarray(f, np.int64))
        return np.concatenate(feet), np.concatenate(faces)


def load_mesh(path):
    import trimesh
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("not a triangle mesh")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("non-finite vertices")
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) == 0 or not mesh.area > 0:
        raise ValueError("no non-degenerate faces")
    return mesh


def sample_surface(mesh, n, rng):
    areas = mesh.area_faces
    faces = rng.choice(len(areas), n, p=areas / areas.sum())
    b = rng.random((n, 2))
    flip = b.sum(1) > 1
    b[flip] = 1 - b[flip]
    tri = mesh.triangles[faces]
    return tri[:, 0] + b[:, :1] * (tri[:, 1] - tri[:, 0]) + b[:, 1:] * (tri[:, 2] - tri[:, 0]), faces


def normalise(mesh, rng, samples=200_000):
    """Centre on the area centroid and scale by the area-weighted RMS radius R (the paper's unit)."""
    points, _ = sample_surface(mesh, samples, rng)
    center = points.mean(0)
    R = float(np.sqrt(((points - center) ** 2).sum(1).mean()))
    if not R > 0:
        raise ValueError("zero-size mesh")
    mesh.vertices = (mesh.vertices - center) / R
    return center, R


def sample_queries(mesh, n, rng, near, on_surface, t_min, t_max, pad):
    """A `near` share at log-uniform offsets |t| in [t_min, t_max] on both sides (some exactly on
    the surface), the rest uniform in the padded bounding box; returned shuffled."""
    k = int(round(near * n))
    points, faces = sample_surface(mesh, k, rng)
    t = np.exp(rng.uniform(np.log(t_min), np.log(t_max), k)) * rng.choice((-1.0, 1.0), k)
    t[rng.random(k) < on_surface] = 0.0
    x_near = points + t[:, None] * mesh.face_normals[faces] + 0.1 * np.abs(t)[:, None] * rng.normal(size=(k, 3))
    lo, hi = mesh.bounds[0] - pad, mesh.bounds[1] + pad
    x = np.concatenate((x_near, rng.uniform(lo, hi, (n - k, 3))))
    near_flag = np.arange(n) < k
    order = rng.permutation(n)
    return x[order], near_flag[order]


# ----------------------------------------------------------------------------- prepare

def prepare_mesh(path, out, cfg, seed):
    started = time.time()
    rng = np.random.default_rng(seed)
    mesh = load_mesh(path)
    center, R = normalise(mesh, rng)
    closest = Closest(mesh.vertices, mesh.faces, cfg["backend"])
    surface, surface_faces = sample_surface(mesh, cfg["surface_points"], rng)
    x, near = sample_queries(mesh, cfg["queries"], rng, cfg["near"], cfg["on_surface"],
                             cfg["t_min"], cfg["t_max"], cfg["pad"])
    d, g, P = point_labels(x, closest, mesh.face_normals, cfg["d_min"])
    H, clipped = [], np.zeros(len(x), bool)
    for r in cfg["scales"]:
        nu = random_directions(len(x), cfg["directions"], rng)
        h, c = clip_eigenvalues(smoothed_hessian(x, r, nu, closest), cfg["h_floor"])
        H.append(h)
        clipped |= c
    nf = min(cfg["floor_queries"], len(x))  # an independent second draw gives the label-noise floor
    H2 = [clip_eigenvalues(smoothed_hessian(x[:nf], r, random_directions(nf, cfg["directions"], rng), closest),
                           cfg["h_floor"])[0] for r in cfg["scales"]]
    edge = float(mesh.edges_unique_length.mean())
    meta = dict(version=VERSION, config_sha=config_sha(cfg), seed=seed, source=str(path),
                source_sha256=sha256(path), center=center.tolist(), rms_radius=R,
                faces=int(len(mesh.faces)), vertices=int(len(mesh.vertices)), mean_edge=edge,
                scale_over_edge=[r / edge for r in cfg["scales"]], clipped_fraction=float(clipped.mean()),
                backend=closest.backend, seconds=round(time.time() - started, 2))
    tmp = Path(out).with_suffix(".tmp.npz")
    np.savez_compressed(tmp, surface=surface.astype(np.float32),
                        normals=mesh.face_normals[surface_faces].astype(np.float32),
                        x=x.astype(np.float32), near=near, d=d.astype(np.float32), g=g.astype(np.float32),
                        P=pack(P).astype(np.float16), H=pack(np.stack(H, 1)).astype(np.float16),
                        H2=pack(np.stack(H2, 1)).astype(np.float16), metadata=np.array(json.dumps(meta)))
    tmp.replace(out)
    return meta


def prepare_job(path, out, cfg, seed):
    """Never raises, so one unreadable mesh cannot stop a corpus run; reuses a finished shard."""
    try:
        if Path(out).exists():
            with np.load(out, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"]))
            if meta.get("config_sha") == config_sha(cfg) and meta.get("seed") == seed:
                return dict(ok=True, meta=meta)
        meta = prepare_mesh(path, out, cfg, seed)
        return dict(ok=True, meta=meta)
    except Exception as error:
        return dict(ok=False, error=f"{type(error).__name__}: {error}")


def prepare(args):
    manifest = Path(args.manifest).resolve()
    entries = json.loads(manifest.read_text())
    groups = {}
    for e in entries:
        if e.get("split") not in ("train", "val", "test") or "group" not in e or "path" not in e:
            raise ValueError("every manifest entry needs path, split (train/val/test) and group")
        if groups.setdefault(e["group"], e["split"]) != e["split"]:
            raise ValueError(f"group {e['group']!r} crosses splits")
    cfg = dict(backend=args.backend, surface_points=args.surface_points, queries=args.queries,
               scales=[float(r) for r in args.scales], directions=args.directions,
               floor_queries=args.floor_queries, h_floor=args.h_floor, d_min=args.d_min, near=args.near,
               on_surface=args.on_surface, t_min=args.t_min, t_max=args.t_max, pad=args.pad)
    out = Path(args.out).resolve()
    root = out.parent / (out.stem + "_shards")
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(str((manifest.parent / e["path"]).resolve()), str(root / f"mesh_{i:07d}.npz"), cfg, args.seed + i)
            for i, e in enumerate(entries)]
    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare_job, *job): i for i, job in enumerate(jobs)}
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if done % 100 == 0 or done == len(jobs):
                failed = sum(1 for r in results if r is not None and not r["ok"])
                print(f"{done}/{len(jobs)} meshes, {failed} failed", flush=True)
    ok = [(e, job, res) for e, job, res in zip(entries, jobs, results) if res["ok"]]
    failures = [dict(path=e["path"], split=e["split"], error=res["error"])
                for e, res in zip(entries, results) if not res["ok"]]
    seen = {s: {res["meta"]["source_sha256"] for e, _, res in ok if e["split"] == s} for s in ("train", "val")}
    shards = []
    for e, job, res in ok:  # exact duplicates may not leak from train into val/test, or from val into test
        digest = res["meta"]["source_sha256"]
        if (e["split"] != "train" and digest in seen["train"]) or (e["split"] == "test" and digest in seen["val"]):
            failures.append(dict(path=e["path"], split=e["split"], error="exact duplicate of a mesh in an earlier split"))
            continue
        meta = res["meta"]
        shards.append(dict(path=os.path.relpath(job[1], out.parent), split=e["split"], group=e["group"],
                           rms_radius=meta["rms_radius"], faces=meta["faces"],
                           mean_edge=meta["mean_edge"], clipped_fraction=meta["clipped_fraction"]))
    json_write(out, dict(version=VERSION, config=cfg, config_sha=config_sha(cfg), shards=shards, failures=failures))
    counts = {s: sum(x["split"] == s for x in shards) for s in ("train", "val", "test")}
    print(json.dumps(dict(prepared=len(shards), failed=len(failures), by_split=counts)))
    return 0


# ----------------------------------------------------------------------------- data

class Corpus:
    """Shards written by `prepare`, with an LRU cache per process."""

    def __init__(self, path, cache=32):
        self.path = Path(path).resolve()
        manifest = json.loads(self.path.read_text())
        if manifest.get("version") != VERSION:
            raise ValueError("corpus prepared by a different version of this script")
        self.config, self.shards = manifest["config"], manifest["shards"]
        self.scales = self.config["scales"]
        self.size = cache
        self.cache = OrderedDict()
        self.split = {s: [i for i, e in enumerate(self.shards) if e["split"] == s] for s in ("train", "val", "test")}

    def load(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        path = self.path.parent / self.shards[i]["path"]
        with np.load(path, allow_pickle=False) as z:
            item = {k: z[k] for k in z.files if k != "metadata"}
        self.cache[i] = item
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return item


def farthest_points(points, k, rng):
    k = min(k, len(points))
    chosen = np.empty(k, np.int64)
    chosen[0] = rng.integers(len(points))
    dist = ((points - points[chosen[0]]) ** 2).sum(1)
    for i in range(1, k):
        chosen[i] = int(dist.argmax())
        dist = np.minimum(dist, ((points - points[chosen[i]]) ** 2).sum(1))
    return chosen


def data_config(args, evaluation=False):
    return dict(input_points=args.input_points, queries=args.queries, tokens=args.tokens,
                mask_ratio=args.mask_ratio, mask_patches=args.mask_patches, whole_fraction=args.whole_fraction,
                rotate=not args.no_rotate and not evaluation, views=args.views)


class Examples(torch.utils.data.Dataset):
    """Example `index` is a pure function of (seed, index), so runs resume exactly and do not
    depend on the worker count. Training draws one mesh per block of `views` consecutive
    examples; evaluation walks the split in order and uses the stored evaluation queries
    (the first `floor_queries`, which carry a second Hessian draw)."""

    def __init__(self, corpus, split, cfg, seed, length, evaluation=False):
        self.corpus, self.ids, self.cfg, self.seed = corpus, corpus.split[split], cfg, seed
        self.length, self.evaluation = length, evaluation
        if not self.ids:
            raise ValueError(f"the {split} split is empty")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        cfg, index = self.cfg, int(index)
        rng = np.random.default_rng([self.seed, index])
        if self.evaluation:
            mesh = self.ids[index % len(self.ids)]
        else:
            pick = np.random.default_rng([self.seed, 1, index // cfg["views"]])
            mesh = self.ids[int(pick.integers(len(self.ids)))]
        item = self.corpus.load(mesh)
        n = cfg["input_points"]
        if 'surface_weights' in item:
            mass=item['surface_weights'].astype(np.float64)
            sel=rng.choice(len(item['surface']),n,replace=True,p=mass/mass.sum())
        else:
            sel = rng.choice(len(item["surface"]), n, replace=len(item["surface"]) < n)
        p, nrm = item["surface"][sel].astype(np.float64), item["normals"][sel].astype(np.float64)
        visible = np.ones(n, bool)
        whole = cfg["mask_ratio"] <= 0 or rng.random() < cfg["whole_fraction"]
        if not whole:  # hide about mask_ratio of the area as Voronoi patches around FPS seeds
            seeds = farthest_points(p, cfg["mask_patches"], rng)
            patch = cKDTree(p[seeds]).query(p)[1]
            share = np.bincount(patch, minlength=len(seeds)) / n
            hidden, total = [], 0.0
            for j in rng.permutation(len(seeds)):
                if total >= cfg["mask_ratio"]:
                    break
                hidden.append(j)
                total += share[j]
            visible = ~np.isin(patch, hidden)
        vis = np.flatnonzero(visible)
        anchors = np.resize(vis[farthest_points(p[vis], cfg["tokens"], rng)], cfg["tokens"])
        owner = cKDTree(p[anchors]).query(p[vis])[1]  # duplicate anchors get zero weight
        token_weights = np.bincount(owner, minlength=cfg["tokens"]).astype(np.float64)
        if self.evaluation:
            q = np.arange(len(item["H2"]))
        else:
            q = rng.choice(len(item["x"]), cfg["queries"], replace=len(item["x"]) < cfg["queries"])
        x, g = item["x"][q].astype(np.float64), item["g"][q].astype(np.float64)
        P, H = unpack(item["P"][q].astype(np.float64)), unpack(item["H"][q].astype(np.float64))
        hidden_query = ~visible[cKDTree(p).query(x - g)[1]]  # the foot falls in a hidden region
        if cfg["rotate"]:  # labels are equivariant, so rotation needs no relabelling
            Q = random_rotation(rng)
            p, nrm, x, g = p @ Q.T, nrm @ Q.T, x @ Q.T, g @ Q.T
            P = Q @ P @ Q.T
            H = np.einsum("ij,qsjk,lk->qsil", Q, H, Q)
        out = dict(points=p.astype(np.float32), normals=nrm.astype(np.float32),
                   weights=visible.astype(np.float32), anchors=anchors.astype(np.int64),
                   token_weights=token_weights.astype(np.float32), whole=np.float32(whole),
                   x=x.astype(np.float32), d=item["d"][q].astype(np.float32), g=g.astype(np.float32),
                   P=pack(P).astype(np.float32), H=pack(H).astype(np.float32),
                   near=item["near"][q], hidden=hidden_query, mesh=np.int64(mesh))
        if self.evaluation:
            out["H2"] = item["H2"].astype(np.float32)
        return out


def make_loader(dataset, start, stop, batch_size, workers, device):
    return torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(start, stop)), batch_size=batch_size,
                                       shuffle=False, num_workers=workers, pin_memory=device.type == "cuda",
                                       persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)


# ----------------------------------------------------------------------------- model

def fourier(x, bands):
    freqs = math.pi * 2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)
    a = (x[..., None] * freqs).flatten(-2)
    return torch.cat((x, a.sin(), a.cos()), -1)


def projector_features(n):
    """Sign-free normal features: the six independent entries of n n^T (off-diagonals scaled by sqrt 2)."""
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    s = math.sqrt(2.0)
    return torch.stack((n[..., 0] ** 2, n[..., 1] ** 2, n[..., 2] ** 2, s * n[..., 0] * n[..., 1],
                        s * n[..., 0] * n[..., 2], s * n[..., 1] * n[..., 2]), -1)


def log_bias(w):
    """Log area weight of each key as an additive attention bias; zero weight hides the key."""
    return torch.where(w > 0, w.clamp_min(1e-30).log(), torch.full_like(w, -math.inf))[:, None, None, :]


class Rotary3D(nn.Module):
    """3-D rotary position embedding: per axis, 2 * (head_dim // 6) channels rotate with that coordinate."""

    def __init__(self, head_dim, min_wavelength=0.02, max_wavelength=8.0):
        super().__init__()
        self.per = 2 * (head_dim // 6)
        if self.per == 0:
            raise ValueError("head dimension must be at least 6")
        wavelengths = torch.logspace(math.log10(max_wavelength), math.log10(min_wavelength), self.per // 2)
        self.register_buffer("freqs", 2 * math.pi / wavelengths, persistent=False)

    def forward(self, x, pos):  # x [B, heads, L, head_dim], pos [B, L, 3]
        half, parts = self.per // 2, []
        for axis in range(3):
            seg = x[..., axis * self.per:(axis + 1) * self.per]
            a = pos[:, None, :, axis, None].to(self.freqs.dtype) * self.freqs
            cos, sin = a.cos().to(x.dtype), a.sin().to(x.dtype)
            parts += [seg[..., :half] * cos - seg[..., half:] * sin, seg[..., :half] * sin + seg[..., half:] * cos]
        parts.append(x[..., 3 * self.per:])
        return torch.cat(parts, -1)


class Attention(nn.Module):
    """Pre-norm cross- or self-attention with log-area bias and 3-D rotary positions, then an MLP."""

    def __init__(self, width, heads, rotary):
        super().__init__()
        self.heads, self.head_dim, self.rotary = heads, width // heads, rotary
        self.qnorm, self.kvnorm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.q, self.kv, self.proj = nn.Linear(width, width), nn.Linear(width, 2 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(),
                                 nn.Linear(4 * width, width))

    def forward(self, q, q_pos, kv, kv_pos, bias):
        B, Lq, W = q.shape
        Lk = kv.shape[1]

        def split(t, length):
            return t.reshape(B, length, self.heads, self.head_dim).transpose(1, 2)

        query = self.rotary(split(self.q(self.qnorm(q)), Lq), q_pos)
        k, v = self.kv(self.kvnorm(kv)).chunk(2, -1)
        key = self.rotary(split(k, Lk), kv_pos)
        attended = F.scaled_dot_product_attention(query, key, split(v, Lk), attn_mask=bias.to(query.dtype))
        q = q + self.proj(attended.transpose(1, 2).reshape(B, Lq, W))
        return q + self.mlp(q)


class GeometryNet(nn.Module):
    """AREALS layout: an area-weighted read of the surface samples into anchor tokens, token
    self-attention, and a point-query decoder with heads d, g, P, H (and optionally a field)."""

    def __init__(self, scales=3, width=192, heads=4, depth=4, decoder_depth=2, bands=8, extra_inputs=0,
                 field_outputs=0):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.config = dict(scales=scales, width=width, heads=heads, depth=depth, decoder_depth=decoder_depth,
                           bands=bands, extra_inputs=extra_inputs, field_outputs=field_outputs)
        positional = 3 + 6 * bands
        self.embed = nn.Sequential(nn.Linear(positional + 6 + extra_inputs, width), nn.GELU(), nn.Linear(width, width))
        self.query_embed = nn.Sequential(nn.Linear(positional, width), nn.GELU(), nn.Linear(width, width))
        self.whole_flag = nn.Parameter(torch.zeros(width))  # tells the decoder the input is complete
        rotary = Rotary3D(width // heads)
        self.read = Attention(width, heads, rotary)
        self.mix = nn.ModuleList([Attention(width, heads, rotary) for _ in range(depth)])
        self.decoder = nn.ModuleList([Attention(width, heads, rotary) for _ in range(decoder_depth)])
        self.norm = nn.LayerNorm(width)
        sizes = dict(d=1, g=3, P=6, H=6 * scales)
        if field_outputs:
            sizes["field"] = field_outputs
        self.out = nn.ModuleDict({k: nn.Linear(width, v) for k, v in sizes.items()})
        with torch.no_grad():  # projector-valued heads start at I/3, the paper's isotropic initialisation
            for name, copies in (("P", 1), ("H", scales)):
                self.out[name].weight.zero_()
                self.out[name].bias.copy_(torch.tensor(ISO * copies))

    def encode(self, points, normals, weights, anchors, token_weights, extra=None):
        features = [fourier(points, self.config["bands"]), projector_features(normals)]
        if self.config["extra_inputs"]:
            if extra is None:
                extra = points.new_zeros(*points.shape[:2], self.config["extra_inputs"])
            features.append(extra)
        h = self.embed(torch.cat(features, -1))
        index = anchors[..., None]
        anchor_pos = torch.gather(points, 1, index.expand(-1, -1, 3))
        tokens = torch.gather(h, 1, index.expand(-1, -1, h.shape[-1]))
        tokens = self.read(tokens, anchor_pos, h, points, log_bias(weights))
        bias = log_bias(token_weights)
        for block in self.mix:
            tokens = block(tokens, anchor_pos, tokens, anchor_pos, bias)
        return tokens, anchor_pos, bias

    def decode(self, x, tokens, anchor_pos, bias, whole):
        h = self.query_embed(fourier(x, self.config["bands"])) + whole[:, None, None] * self.whole_flag
        for block in self.decoder:
            h = block(h, x, tokens, anchor_pos, bias)
        h = self.norm(h)
        out = dict(d=F.softplus(self.out["d"](h))[..., 0], g=self.out["g"](h), P=self.out["P"](h),
                   H=self.out["H"](h).unflatten(-1, (self.config["scales"], 6)))
        if "field" in self.out:
            out["field"] = self.out["field"](h)
        return out

    def forward(self, batch, extra=None):
        tokens, anchor_pos, bias = self.encode(batch["points"], batch["normals"], batch["weights"],
                                               batch["anchors"], batch["token_weights"], extra)
        return self.decode(batch["x"], tokens, anchor_pos, bias, batch["whole"])


def make_field_model(trunk_path, extra_inputs=2, field_outputs=1):
    """Geometry trunk plus zero-initialised field input columns (e.g. value * presence, presence)
    and a new field head. At initialisation the trunk computes exactly what the geometry model
    did, as in AREALS Section 3.3; fine-tune all of it on simulations."""
    state = torch.load(trunk_path, map_location="cpu", weights_only=True)
    cfg = dict(state["model_config"])
    cfg.update(extra_inputs=cfg["extra_inputs"] + extra_inputs, field_outputs=field_outputs)
    model = GeometryNet(**cfg)
    weights = dict(state["model"])
    first = weights["embed.0.weight"]
    weights["embed.0.weight"] = torch.cat((first, first.new_zeros(first.shape[0], extra_inputs)), 1)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or any(not k.startswith("out.field.") for k in missing):
        raise ValueError(f"trunk does not match: missing {missing}, unexpected {unexpected}")
    return model


# ----------------------------------------------------------------------------- training

TARGETS = ("d", "g", "P", "H")


def parse_targets(text):
    targets = tuple(t.strip() for t in text.split(",") if t.strip())
    if not targets or any(t not in TARGETS for t in targets) or len(set(targets)) != len(targets):
        raise ValueError(f"--targets must be a comma list drawn from {TARGETS}")
    return targets


def loss_terms(pred, batch, huber):
    """All four terms are always computed (and logged); --targets decides which are optimised."""
    fro = pred["P"].new_tensor(FRO)
    return dict(d=F.huber_loss(pred["d"], batch["d"], delta=huber),
                g=(pred["g"] - batch["g"]).square().sum(-1).mean(),
                P=((pred["P"] - batch["P"]).square() * fro).sum(-1).mean(),
                H=((pred["H"] - batch["H"]).square() * fro).sum(-1).mean())


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(name)


def autocast(device, enabled):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=bool(enabled) and device.type == "cuda")


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def as_float(pred):
    return {k: v.float() for k, v in pred.items()}


def make_optimizer(model, lr, weight_decay):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    plain = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([dict(params=decay, weight_decay=weight_decay), dict(params=plain, weight_decay=0.0)],
                             lr=lr, betas=(0.9, 0.95))


def learning_rate(step, args):
    """Linear warm-up, then cosine decay to final_lr * lr."""
    if step <= args.warmup:
        return args.lr * step / max(args.warmup, 1)
    progress = min(1.0, (step - args.warmup) / max(args.steps - args.warmup, 1))
    return args.lr * (args.final_lr + (1 - args.final_lr) * 0.5 * (1 + math.cos(math.pi * progress)))


def save_checkpoint(path, model, optimizer, step, best, config):
    payload = dict(version=VERSION, config=config, model_config=model.config, model=model.state_dict(),
                   optimizer=optimizer.state_dict(), step=step, best=best)
    tmp = Path(str(path) + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def restore(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("version") != VERSION:
        raise ValueError("checkpoint written by a different version of this script")
    model = GeometryNet(**state["model_config"]).to(device)
    model.load_state_dict(state["model"])
    return model, state


@torch.no_grad()
def validate(model, loader, device, huber, bf16):
    model.eval()
    sums, count = dict.fromkeys(TARGETS, 0.0), 0
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device, bf16):
            pred = model(batch)
        terms = loss_terms(as_float(pred), batch, huber)
        n = len(batch["x"])
        for k in TARGETS:
            sums[k] += float(terms[k]) * n
        count += n
    model.train()
    return {k: v / count for k, v in sums.items()}


def train(args):
    device = choose_device(args.device)
    corpus = Corpus(args.data, cache=args.cache)
    targets = parse_targets(args.targets)
    weights = dict(d=args.weight_d, g=args.weight_g, P=args.weight_P, H=args.weight_H)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    skip = ("func", "resume", "stop_after", "device", "workers", "cache", "threads", "command")
    config = {k: v for k, v in vars(args).items() if k not in skip}
    torch.manual_seed(args.seed)
    model = GeometryNet(len(corpus.scales), args.width, args.heads, args.depth, args.decoder_depth,
                        args.bands).to(device)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    start, best = 0, math.inf
    if args.resume:
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
        if state["config"] != config:
            raise ValueError("--resume needs the identical settings "
                             "(only --stop-after, --device, --workers and --cache may change)")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["step"], state["best"]
    elif (out / "last.pt").exists():
        raise ValueError("the output folder already holds a run; pass --resume or choose a fresh --out")
    parameters = sum(p.numel() for p in model.parameters())
    val_split = "val" if corpus.split["val"] else "train"
    json_write(out / "config.json", dict(config, parameters=parameters,
                                         validation_split=val_split, scales=corpus.scales))
    print(f"{parameters:,} parameters on {device}; targets {','.join(targets)}; "
          f"selection on the {val_split} split", flush=True)
    end = min(args.steps, args.stop_after) if args.stop_after else args.steps
    train_data = Examples(corpus, "train", data_config(args), args.seed, args.steps * args.batch_size)
    loader = make_loader(train_data, start * args.batch_size, end * args.batch_size, args.batch_size,
                         args.workers, device)
    val_data = Examples(corpus, val_split, data_config(args, evaluation=True), args.seed + 1,
                        min(args.val_examples, len(corpus.split[val_split])), evaluation=True)
    val_loader = make_loader(val_data, 0, len(val_data), args.batch_size, min(args.workers, 4), device)
    log = open(out / "train.jsonl", "a" if args.resume else "w", buffering=1)
    started, skipped = time.monotonic(), 0
    model.train()
    try:
        for step, batch in enumerate(loader, start + 1):
            batch = to_device(batch, device)
            lr = learning_rate(step, args)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with autocast(device, args.bf16):
                pred = model(batch)
            terms = loss_terms(as_float(pred), batch, args.huber)
            loss = sum(weights[k] * terms[k] for k in targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            finite = bool(torch.isfinite(norm))
            if finite:
                optimizer.step()
            else:
                skipped += 1
                if skipped > 10:
                    raise FloatingPointError("more than 10 non-finite gradients")
            log.write(json.dumps(dict(step=step, lr=lr, loss=float(loss), grad_norm=float(norm), applied=finite,
                                      **{k: float(v) for k, v in terms.items()})) + "\n")
            if step % args.validate_every == 0 or step == args.steps:
                val = validate(model, val_loader, device, args.huber, args.bf16)
                score = sum(weights[k] * val[k] for k in targets)  # each arm selects on its own objective
                if score < best:
                    best = score
                    save_checkpoint(out / "best.pt", model, optimizer, step, best, config)
                print(json.dumps(dict(step=step, validation=val, selection_score=score, split=val_split,
                                      minutes=round((time.monotonic() - started) / 60, 1))), flush=True)
            if step % args.save_every == 0 or step == end:
                save_checkpoint(out / "last.pt", model, optimizer, step, best, config)
    finally:
        log.close()
    if end < args.steps:
        print(f"paused at step {end}; rerun the same command with --resume")
    return 0


# ----------------------------------------------------------------------------- evaluation

BANDS = (0.002, 0.01, 0.05, 0.3)  # offset bands (units of R) for curvature: identifiable only at t*kappa ~ 1


def top_lines(packed):
    return np.linalg.eigh(unpack(packed))[1][..., :, -1]


def line_angles(a, b):
    """Degrees between the top eigenvectors (lines) of two packed symmetric matrices."""
    cos = np.abs((top_lines(a) * top_lines(b)).sum(-1))
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def offset_curvature(packed_H):
    """t*kappa in the two tangent directions from h = t k / (1 + t k); NaN where h >= 0.9.
    H <= I, so the normal eigenvalue is the largest and the two smallest are tangential."""
    w = np.linalg.eigvalsh(unpack(packed_H))[..., :2]
    return np.where(w < 0.9, w / (1.0 - np.minimum(w, 0.9)), np.nan)


def inputs_only_labels(batch, scales, directions, rng, d_min=1e-4):
    """The same labels computed from the visible input samples alone (nearest sample as the foot):
    what the inputs give with no learning. Near the sample spacing its H reflects Voronoi cells
    of the samples, not curvature."""
    out = dict(d=[], g=[], P=[], H=[])
    for b in range(len(batch["x"])):
        visible = batch["weights"][b] > 0
        points = batch["points"][b][visible].astype(np.float64)
        normals = batch["normals"][b][visible].astype(np.float64)
        tree = cKDTree(points)

        def closest(y, tree=tree, points=points):
            index = tree.query(y)[1]
            return points[index], index

        x = batch["x"][b].astype(np.float64)
        d, g, P = point_labels(x, closest, normals, d_min)
        H = [smoothed_hessian(x, r, random_directions(len(x), directions, rng), closest) for r in scales]
        for key, value in (("d", d), ("g", g), ("P", pack(P)), ("H", pack(np.stack(H, 1)))):
            out[key].append(value)
    return {k: np.stack(v) for k, v in out.items()}


def query_rows(pred, lab, scales):
    S = len(scales)
    return dict(d_err=(pred["d"] - lab["d"]).ravel(),
                angle=line_angles(pred["P"], lab["P"]).ravel(),
                foot=np.linalg.norm(pred["g"] - lab["g"], axis=-1).ravel(),
                h_sq=(((pred["H"] - lab["H"]) ** 2) * np.array(FRO)).sum(-1).reshape(-1, S),
                tk_err=np.abs(offset_curvature(pred["H"]) - offset_curvature(lab["H"])).reshape(-1, S, 2))


def finite(v):
    v = np.asarray(v, np.float64).ravel()
    return v[np.isfinite(v)]


def quantiles(v):
    v = np.abs(finite(v))
    if not len(v):
        return None
    return dict(n=int(len(v)), mean=float(v.mean()), p50=float(np.quantile(v, 0.5)),
                p95=float(np.quantile(v, 0.95)), p99=float(np.quantile(v, 0.99)), max=float(v.max()))


def root_mean(v, squared=False):
    v = finite(v)
    if not len(v):
        return None
    return float(np.sqrt(v.mean() if squared else (v ** 2).mean()))


def summarise(rows, masks, scales, floor_sq):
    groups = dict(near=masks["near"], near_visible=masks["near"] & ~masks["hidden"],
                  near_hidden=masks["near"] & masks["hidden"], on_surface=masks["on"], far=~masks["near"])
    report = {}
    for name, m in groups.items():
        if not m.any():
            continue
        entry = dict(queries=int(m.sum()), distance_rms=root_mean(rows["d_err"][m]),
                     distance_abs=quantiles(rows["d_err"][m]), normal_line_degrees=quantiles(rows["angle"][m]),
                     foot_offset=quantiles(rows["foot"][m]), hessian={})
        for s, r in enumerate(scales):
            mse, floor = finite(rows["h_sq"][m, s]), finite(floor_sq[m, s])
            h = dict(rms=root_mean(mse, squared=True), label_noise_floor=root_mean(floor, squared=True))
            if len(mse) and len(floor):
                h["excess_rms"] = float(np.sqrt(max(0.0, mse.mean() - floor.mean())))
            bands = {}
            for lo, hi in zip(BANDS[:-1], BANDS[1:]):
                inside = m & (masks["dist"] >= lo) & (masks["dist"] < hi)
                if inside.any():
                    err = rows["tk_err"][inside, s]
                    bands[f"{lo:g}-{hi:g}"] = dict(t_kappa_rms=root_mean(err),
                                                   defined_fraction=float(np.isfinite(err).mean()))
            h["t_kappa_by_offset"] = bands
            entry["hessian"][f"r={r:g}"] = h
        report[name] = entry
    return report


def evaluate(args):
    device = choose_device(args.device)
    model, state = restore(args.checkpoint, device)
    model.eval()
    corpus = Corpus(args.data, cache=args.cache)
    if len(corpus.scales) != model.config["scales"]:
        raise ValueError("the corpus Hessian scales do not match the model")
    cfg = data_config(argparse.Namespace(**state["config"]), evaluation=True)
    cfg.update(mask_ratio=args.mask_ratio, whole_fraction=0.0)
    ids = corpus.split[args.split]
    if not ids:
        raise ValueError(f"the {args.split} split is empty")
    count = len(ids) if args.examples <= 0 else min(args.examples, len(ids))
    loader = make_loader(Examples(corpus, args.split, cfg, args.seed, count, evaluation=True), 0, count,
                         args.batch_size, args.workers, device)
    rng = np.random.default_rng(args.seed)
    rows = dict(learned={}, inputs_only={})
    masks, floor = dict(near=[], hidden=[], on=[], dist=[]), []
    with torch.no_grad():
        for batch in loader:
            with autocast(device, args.bf16):
                pred = model(to_device(batch, device))
            lab = {k: v.numpy() for k, v in batch.items()}
            methods = dict(learned={k: v.float().cpu().numpy() for k, v in pred.items()})
            if not args.no_baseline:
                methods["inputs_only"] = inputs_only_labels(lab, corpus.scales, args.baseline_directions, rng)
            for name, values in methods.items():
                for k, v in query_rows(values, lab, corpus.scales).items():
                    rows[name].setdefault(k, []).append(v)
            masks["near"].append(lab["near"].ravel())
            masks["hidden"].append(lab["hidden"].ravel())
            masks["on"].append((lab["d"] <= 1e-4).ravel())
            masks["dist"].append(lab["d"].ravel())
            floor.append((((lab["H"] - lab["H2"]) ** 2) * np.array(FRO)).sum(-1).reshape(-1, len(corpus.scales)) / 2)
    masks = {k: np.concatenate(v) for k, v in masks.items()}
    floor = np.concatenate(floor)
    metrics = {name: summarise({k: np.concatenate(v) for k, v in r.items()}, masks, corpus.scales, floor)
               for name, r in rows.items() if r}
    group = "near_hidden" if args.mask_ratio > 0 else "near"
    learned, base = metrics["learned"].get(group, {}), metrics.get("inputs_only", {}).get(group, {})
    normal = (learned.get("normal_line_degrees") or {}).get("mean")
    gates = {}
    if args.gate_distance is not None:
        gates["distance_rms"] = learned.get("distance_rms") is not None and learned["distance_rms"] <= args.gate_distance
    if args.gate_normal is not None:
        gates["normal_line_mean"] = normal is not None and normal <= args.gate_normal
    if args.require_beat_baseline and base:
        base_normal = (base.get("normal_line_degrees") or {}).get("mean")
        gates["distance_beats_inputs_only"] = None not in (learned.get("distance_rms"), base.get("distance_rms")) \
            and learned["distance_rms"] < base["distance_rms"]
        gates["normal_beats_inputs_only"] = None not in (normal, base_normal) and normal < base_normal
    edges = np.array([corpus.shards[i]["mean_edge"] for i in ids[:count]])
    result = dict(version=VERSION, checkpoint=str(args.checkpoint),
                  checkpoint_step=state["step"], targets=state["config"]["targets"], split=args.split,
                  meshes=count, mask_ratio=args.mask_ratio, headline_group=group, scales=corpus.scales,
                  median_scale_over_edge=[float(np.median(r / edges)) for r in corpus.scales],
                  metrics=metrics, gates=gates, passed=all(gates.values()) if gates else None)
    json_write(args.out, result)
    brief = {name: dict(distance_rms=m.get(group, {}).get("distance_rms"),
                        normal_mean=(m.get(group, {}).get("normal_line_degrees") or {}).get("mean"),
                        hessian_rms={k: v["rms"] for k, v in m.get(group, {}).get("hessian", {}).items()})
             for name, m in metrics.items()}
    print(json.dumps(dict(group=group, meshes=count, summary=brief, gates=gates, out=args.out), indent=2))
    return 2 if gates and not all(gates.values()) else 0


# ----------------------------------------------------------------------------- export and query

def export(args):
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if state.get("version") != VERSION:
        raise ValueError("checkpoint written by a different version of this script")
    torch.save(dict(version=VERSION, model_config=state["model_config"], model=state["model"],
                    targets=state["config"]["targets"], step=state["step"]), args.out)
    print(json.dumps(dict(out=args.out, targets=state["config"]["targets"], step=state["step"])))
    return 0


def query(args):
    """Encode one prepared mesh from complete input and decode geometry at points in its own units."""
    device = choose_device(args.device)
    model, state = restore(args.checkpoint, device)
    model.eval()
    corpus = Corpus(args.data)
    if not 0 <= args.mesh < len(corpus.shards):
        raise ValueError("mesh index out of range")
    with np.load(corpus.path.parent / corpus.shards[args.mesh]["path"], allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"]))
    if args.points.endswith(".npy"):
        points = np.load(args.points).astype(np.float64).reshape(-1, 3)
    else:
        points = np.array([[float(c) for c in p.split(",")] for p in args.points.split(";")])
    center, R = np.asarray(meta["center"]), meta["rms_radius"]
    item = corpus.load(args.mesh)
    cfg = data_config(argparse.Namespace(**state["config"]), evaluation=True)
    rng = np.random.default_rng(args.seed)
    n = cfg["input_points"]
    sel = rng.choice(len(item["surface"]), n, replace=len(item["surface"]) < n)
    p = item["surface"][sel].astype(np.float64)
    anchors = np.resize(farthest_points(p, cfg["tokens"], rng), cfg["tokens"])
    owner = cKDTree(p[anchors]).query(p)[1]
    batch = dict(points=p[None].astype(np.float32), normals=item["normals"][sel][None].astype(np.float32),
                 weights=np.ones((1, n), np.float32), anchors=anchors[None].astype(np.int64),
                 token_weights=np.bincount(owner, minlength=cfg["tokens"])[None].astype(np.float32),
                 whole=np.ones(1, np.float32), x=((points - center) / R)[None].astype(np.float32))
    with torch.no_grad():
        pred = model({k: torch.as_tensor(v, device=device) for k, v in batch.items()})
    pred = {k: v.float().cpu().numpy()[0] for k, v in pred.items()}
    results = []
    for i, point in enumerate(points):
        H = unpack(pred["H"][i])
        results.append(dict(point=point.tolist(), distance=float(pred["d"][i] * R),
                            foot=(point - pred["g"][i] * R).tolist(), normal_line=top_lines(pred["P"][i]).tolist(),
                            hessian_eigenvalues={f"r={r:g}R": np.linalg.eigvalsh(H[s]).tolist()
                                                 for s, r in enumerate(corpus.scales)}))
    print(json.dumps(dict(targets=state["config"]["targets"], results=results), indent=2))
    return 0


# ----------------------------------------------------------------------------- self-test

def selftest(args):
    """Closed-form label identities, exact equivariance, unbiasedness and model invariances."""
    rng = np.random.default_rng(17)
    checks, limits = {}, {}
    zeros = lambda y: np.zeros(len(y), np.int64)
    sphere = lambda y: (y / np.linalg.norm(y, axis=1, keepdims=True), zeros(y))
    box = lambda y: (np.clip(y, -1.0, 1.0), zeros(y))
    planes = lambda y: (np.c_[y[:, 0], y[:, 1], np.where(y[:, 2] >= 0, 1.0, -1.0)], zeros(y))

    for t in (0.2, -0.2):  # unit sphere, both sides: tangent eigenvalues t / (1 + t), normal eigenvalue 1
        x = np.array([[1 + t, 0.0, 0.0]])
        w = np.linalg.eigvalsh(smoothed_hessian(x, 0.05, random_directions(1, 4096, rng), sphere))[0]
        checks[f"sphere_t{t:+.1f}"] = float(np.abs(w - [t / (1 + t), t / (1 + t), 1.0]).max())
        limits[f"sphere_t{t:+.1f}"] = 1e-2

    nu = random_directions(1, 4096, rng)  # cube: edge region -> normal plane (0, 1, 1); face -> line (0, 0, 1)
    edge = np.linalg.eigvalsh(smoothed_hessian(np.array([[1.1, 1.1, 0.0]]), 0.05, nu, box))[0]
    face = np.linalg.eigvalsh(smoothed_hessian(np.array([[1.1, 0.9, 0.0]]), 0.05, nu, box))[0]
    checks["cube_edge_normal_plane"] = float(np.abs(edge - [0.0, 1.0, 1.0]).max())
    checks["cube_face_line"] = float(np.abs(face - [0.0, 0.0, 1.0]).max())
    limits.update(cube_edge_normal_plane=1e-2, cube_face_line=1e-2)

    w = np.linalg.eigvalsh(smoothed_hessian(np.zeros((1, 3)), 0.1, random_directions(1, 4096, rng), planes))[0]
    checks["planes_medial_axis"] = float(max(abs(w[0] + 14.0) / 14.0, abs(w[1]), abs(w[2])))  # 1 - 1.5 a / r
    limits["planes_medial_axis"] = 1e-2

    Q = random_rotation(rng)  # rotating geometry, queries and directions together rotates H exactly
    turned = lambda y: (np.clip(y @ Q, -1.0, 1.0) @ Q.T, zeros(y))
    x = np.array([[1.1, 0.95, 0.3], [0.2, 1.3, -1.2], [0.5, 0.1, 1.04]])
    nu = random_directions(len(x), 64, rng)
    H0 = smoothed_hessian(x, 0.07, nu, box)
    H1 = smoothed_hessian(x @ Q.T, 0.07, nu @ Q.T, turned)
    checks["rotation_equivariance"] = float(np.abs(H1 - Q @ H0 @ Q.T).max())
    limits["rotation_equivariance"] = 1e-9

    x = np.array([[1.2, 0.0, 0.0]])  # small random designs average to the large-design value
    reference = smoothed_hessian(x, 0.05, random_directions(1, 16384, rng), sphere)[0]
    draws = np.stack([smoothed_hessian(x, 0.05, random_directions(1, 8, rng), sphere)[0] for _ in range(4000)])
    gap = np.abs(draws.mean(0) - reference)
    checks["unbiased_small_designs"] = bool((gap <= 4 * draws.std(0) / np.sqrt(len(draws)) + 2e-3).all())

    M = rng.normal(size=(5, 3, 3))
    M = M + np.swapaxes(M, 1, 2)
    checks["pack_roundtrip"] = float(np.abs(unpack(pack(M)) - M).max())
    checks["packed_frobenius"] = float(np.abs((pack(M) ** 2 * np.array(FRO)).sum(-1) - (M ** 2).sum((-1, -2))).max())
    limits.update(pack_roundtrip=1e-12, packed_frobenius=1e-9)

    try:  # reversing every face's winding must not change any label
        import trimesh
        mesh = trimesh.creation.icosphere(subdivisions=3)
        x = np.random.default_rng(3).normal(size=(64, 3)) * 0.8
        a = Closest(mesh.vertices, mesh.faces, args.backend)
        b = Closest(mesh.vertices, np.ascontiguousarray(mesh.faces[:, ::-1]), args.backend)
        la, lb = point_labels(x, a, mesh.face_normals, 1e-4), point_labels(x, b, -mesh.face_normals, 1e-4)
        nu = random_directions(len(x), 16, rng)
        checks["winding_flip_labels"] = float(max(np.abs(u - v).max() for u, v in zip(la, lb)))
        checks["winding_flip_hessian"] = float(np.abs(smoothed_hessian(x, 0.05, nu, a)
                                                      - smoothed_hessian(x, 0.05, nu, b)).max())
        limits.update(winding_flip_labels=1e-5, winding_flip_hessian=1e-4)  # float32 nearest points
    except ImportError:
        checks["winding_flip"] = "skipped: trimesh is not installed"

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    model = GeometryNet(scales=2, width=48, heads=2, depth=1, decoder_depth=1, bands=2).double()
    B, N, T, Qn = 2, 40, 8, 5
    batch = dict(points=torch.randn(B, N, 3, generator=gen, dtype=torch.float64),
                 normals=F.normalize(torch.randn(B, N, 3, generator=gen, dtype=torch.float64), dim=-1),
                 weights=torch.rand(B, N, generator=gen, dtype=torch.float64) + 0.5,
                 anchors=torch.stack([torch.randperm(N, generator=gen)[:T] for _ in range(B)]),
                 token_weights=torch.rand(B, T, generator=gen, dtype=torch.float64) + 0.5,
                 whole=torch.zeros(B, dtype=torch.float64),
                 x=torch.randn(B, Qn, 3, generator=gen, dtype=torch.float64))
    with torch.no_grad():
        base = model(batch)

        def gap(other):
            return float(max((base[k] - other[k]).abs().max() for k in ("d", "g", "P", "H")))

        perm = torch.randperm(N, generator=gen)
        inverse = torch.argsort(perm)
        checks["input_permutation"] = gap(model(dict(batch, points=batch["points"][:, perm],
                                                     normals=batch["normals"][:, perm],
                                                     weights=batch["weights"][:, perm],
                                                     anchors=inverse[batch["anchors"]])))
        signs = torch.where(torch.rand(B, N, 1, generator=gen) < 0.5, -1.0, 1.0).double()
        checks["normal_sign"] = gap(model(dict(batch, normals=batch["normals"] * signs)))
        used = set(batch["anchors"].flatten().tolist())
        j = next(i for i in range(N) if i not in used)
        w = batch["weights"].clone()
        w[:, j] /= 2  # one key split into two copies of half the area
        checks["area_weighted_duplication"] = gap(model(dict(
            batch, points=torch.cat((batch["points"], batch["points"][:, j:j + 1]), 1),
            normals=torch.cat((batch["normals"], batch["normals"][:, j:j + 1]), 1),
            weights=torch.cat((w, w[:, j:j + 1]), 1))))
        with tempfile.TemporaryDirectory() as folder:
            trunk = Path(folder) / "trunk.pt"
            torch.save(dict(version=VERSION, model_config=model.config, model=model.state_dict()), trunk)
            field = make_field_model(trunk).double()
            extra = torch.randn(B, N, 2, generator=gen, dtype=torch.float64)
            checks["field_columns_start_at_zero"] = gap(field(batch, extra=extra))
    limits.update(input_permutation=1e-9, normal_sign=1e-9, area_weighted_duplication=1e-9,
                  field_columns_start_at_zero=1e-9)
    loss = sum(v.square().mean() for v in model(batch).values())
    loss.backward()
    checks["gradients_finite"] = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())

    failed = [k for k, v in checks.items()
              if (isinstance(v, bool) and not v) or (k in limits and not v <= limits[k])]
    result = dict(passed=not failed, failed=failed, checks=checks, limits=limits, torch=torch.__version__)
    if args.out:
        json_write(args.out, result)
    print(json.dumps(result, indent=2))
    return 0 if not failed else 2


# ----------------------------------------------------------------------------- command line

def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 keeps the default); before the command")
    sub = p.add_subparsers(dest="command", required=True)
    backends = ("auto", "open3d", "trimesh")

    s = sub.add_parser("selftest", help="label identities, equivariance and model invariances")
    s.add_argument("--backend", default="auto", choices=backends)
    s.add_argument("--out")
    s.set_defaults(func=selftest)

    s = sub.add_parser("prepare", help="nearest-point labels for every mesh in a manifest")
    s.add_argument("--manifest", required=True)
    s.add_argument("--out", required=True, help="corpus .json; shards are written to <stem>_shards/")
    s.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    s.add_argument("--seed", type=int, default=17)
    s.add_argument("--backend", default="auto", choices=backends)
    s.add_argument("--surface-points", type=int, default=16384)
    s.add_argument("--queries", type=int, default=8192)
    s.add_argument("--scales", type=float, nargs="+", default=[0.02, 0.05, 0.15], help="Hessian radii, units of R")
    s.add_argument("--directions", type=int, default=32, help="antithetic direction pairs per Hessian label")
    s.add_argument("--floor-queries", type=int, default=2048, help="queries given a second, independent Hessian draw")
    s.add_argument("--h-floor", type=float, default=-8.0, help="eigenvalue clip near the medial axis")
    s.add_argument("--d-min", type=float, default=1e-4, help="below this distance P uses the face normal")
    s.add_argument("--near", type=float, default=0.7, help="share of queries near the surface")
    s.add_argument("--on-surface", type=float, default=0.2, help="share of near queries placed on the surface")
    s.add_argument("--t-min", type=float, default=0.002)
    s.add_argument("--t-max", type=float, default=0.3)
    s.add_argument("--pad", type=float, default=0.2)
    s.set_defaults(func=prepare)

    s = sub.add_parser("train", help="geometry pretraining")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--targets", default="d,P,H", help="comma list from d,g,P,H; one ablation arm per run")
    s.add_argument("--device", default="auto")
    s.add_argument("--seed", type=int, default=17)
    s.add_argument("--steps", type=int, default=100000)
    s.add_argument("--batch-size", type=int, default=64)
    s.add_argument("--workers", type=int, default=8)
    s.add_argument("--cache", type=int, default=32, help="shards cached per data worker")
    s.add_argument("--views", type=int, default=1, help="consecutive examples drawn from one mesh (less I/O)")
    s.add_argument("--input-points", type=int, default=4096)
    s.add_argument("--queries", type=int, default=1024)
    s.add_argument("--tokens", type=int, default=256)
    s.add_argument("--mask-ratio", type=float, default=0.6)
    s.add_argument("--mask-patches", type=int, default=32)
    s.add_argument("--whole-fraction", type=float, default=0.1)
    s.add_argument("--no-rotate", action="store_true")
    s.add_argument("--width", type=int, default=192)
    s.add_argument("--heads", type=int, default=4)
    s.add_argument("--depth", type=int, default=4)
    s.add_argument("--decoder-depth", type=int, default=2)
    s.add_argument("--bands", type=int, default=8)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--final-lr", type=float, default=0.01, help="final learning rate as a fraction of --lr")
    s.add_argument("--warmup", type=int, default=2000)
    s.add_argument("--weight-decay", type=float, default=0.05)
    s.add_argument("--clip", type=float, default=1.0)
    s.add_argument("--huber", type=float, default=0.05)
    for name in TARGETS:
        s.add_argument(f"--weight-{name}", type=float, default=1.0)
    s.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--validate-every", type=int, default=1000)
    s.add_argument("--val-examples", type=int, default=512)
    s.add_argument("--save-every", type=int, default=1000)
    s.add_argument("--resume", action="store_true")
    s.add_argument("--stop-after", type=int)
    s.set_defaults(func=train)

    s = sub.add_parser("evaluate", help="held-out metrics, label-noise floor and inputs-only baseline")
    s.add_argument("--data", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--split", default="test", choices=("train", "val", "test"))
    s.add_argument("--device", default="auto")
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--cache", type=int, default=32)
    s.add_argument("--examples", type=int, default=0, help="meshes to evaluate (0 = the whole split)")
    s.add_argument("--mask-ratio", type=float, default=0.0, help="0 = complete inputs, Proposition 1's setting")
    s.add_argument("--seed", type=int, default=101)
    s.add_argument("--no-baseline", action="store_true")
    s.add_argument("--baseline-directions", type=int, default=16)
    s.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    s.add_argument("--gate-distance", type=float, help="maximum distance RMS, units of R")
    s.add_argument("--gate-normal", type=float, help="maximum mean normal-line error, degrees")
    s.add_argument("--require-beat-baseline", action="store_true")
    s.add_argument("--out", default="evaluation.json")
    s.set_defaults(func=evaluate)

    s = sub.add_parser("export", help="write the trunk for field fine-tuning (see make_field_model)")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=export)

    s = sub.add_parser("query", help="decode geometry at given points for one prepared mesh")
    s.add_argument("--data", required=True)
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--mesh", type=int, required=True, help="index into the corpus shard list")
    s.add_argument("--points", required=True, help='"x,y,z;x,y,z" in the mesh\'s own units, or a .npy file')
    s.add_argument("--device", default="auto")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=query)
    return p


def main():
    args = parser().parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    positive = ("steps", "batch_size", "queries", "input_points", "tokens", "surface_points", "directions",
                "validate_every", "save_every", "val_examples", "views", "mask_patches", "floor_queries")
    for name in positive:
        if getattr(args, name, 1) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "workers", 0) < 0 or (args.command == "prepare" and args.workers < 1):
        raise ValueError("--workers must be at least 1 for prepare and non-negative otherwise")
    if args.command == "prepare" and (min(args.scales) <= 0 or not 0 < args.t_min < args.t_max):
        raise ValueError("scales must be positive and 0 < t-min < t-max")
    if hasattr(args, "mask_ratio") and not 0 <= args.mask_ratio < 1:
        raise ValueError("--mask-ratio must lie in [0, 1)")
    if args.command == "train":
        parse_targets(args.targets)
        if args.tokens > args.input_points * (1 - args.mask_ratio):
            raise ValueError("--tokens exceeds the visible input points; lower it or the mask ratio")
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
