#!/usr/bin/env python3
"""Mesh -> smooth local charts -> learned coefficients -> analytic metric/A.

Dependencies: Python 3.10+, numpy, scipy, trimesh, torch.
Example:
  python train_mesh_geometry.py selftest
  python train_mesh_geometry.py prepare --meshes car.glb --out car.npz
  python train_mesh_geometry.py check --data car.npz --out check.json
  python train_mesh_geometry.py train --data car.npz --out run --device cuda
  python train_mesh_geometry.py evaluate --data car.npz --checkpoint run/best.pt
  python train_mesh_geometry.py encode --data car.npz --checkpoint run/best.pt --out encoded.npz
  python train_mesh_geometry.py query --encoded encoded.npz --patch 0 --u 0 --v 0 --offset 0

Corpus: create a JSON list of {"path":"car.glb", "split":"train", "group":"family1"}
entries (paths relative to that JSON; split train/val/test; families stay together).
  python train_mesh_geometry.py prepare --manifest meshes.json --out corpus.json
  python train_mesh_geometry.py train --data corpus.json --out run --device cuda
A .json output streams one NPZ shard per mesh; a .npz output holds a compact
dataset in memory. --eval-patches controls a reproducible validation subset.
Run evaluate with --max-patches 0 for every patch in its selected split.
Encode a corpus to a directory; query one resulting per-mesh encoded NPZ.

Loss: mean ||A_pred-A_reference||_F^2 under reference-area quadrature, with
an augmented-Lagrangian position constraint RMS/mesh_diagonal <= 0.01.
Default optimizer: AdamW(1e-4, weight_decay=0), clip=1, 20,000 updates, seed=17.
Validate every 250 updates; select lowest A loss among position-feasible models.
Best model is tested on fresh queries with unchanged weights. Report thresholds:
position RMS <=0.01 mesh diagonal, normal-line mean <=5 degrees, relative A RMS
<=0.01, sampled relative A maximum <=0.05, no invalid sampled coordinate maps,
sampled mesh coverage >=95%, overlap position RMS <=0.01 mesh diagonal.
Mesh-fit quality is measured independently against new triangle samples; a
second fit tests reference-A stability. Domain/side information counts are saved.
Reference-plane origins need not lie on the fitted surface: constant and linear
polynomial coefficients are learned too. No frozen encoder or oracle readout.

Resume: rerun the identical train command with --resume. --stop-after N pauses
without changing the original --steps budget. Checkpoints contain optimizer,
dual multiplier, RNG states, model configuration and the data fingerprint.
--local-only permits patch diagnostics on incomplete covers, never a whole-mesh
pass. No seam samples means the seam check is unverified and does not pass.

This is a NEW, fully trainable patch encoder, not a loader for best_normal.pt.
Stage: complete-geometry reconstruction, not masked completion or CFD training.
Each chart is a fixed total-degree polynomial (one smooth polynomial element).
The network predicts its coefficients ONCE. Spatial derivatives differentiate
the polynomial basis, never attention or LayerNorm. No separate normal/A head.
Reference fits and learned reconstruction have separate validation reports.

Coordinate convention: mesh coordinates are divided by its bounding-box diagonal.
With patch radius r, s=(r*u,r*v), t=r*tau, and orthonormal frame E,
  p(s) = origin + r E [u,v,h(u,v)], x(s,t)=p(s)+t*n(s).
geometry() returns J = dx/d(s1,s2,t), g=J.T J, A=abs(det(J))*inverse(g).
Thus A uses the SAME physical-length chart for prediction and reference. To use
dimensionless (u,v,tau) instead, its 3-D coefficient is r*A. W=+Dn.

Distances to the union of these patches are NOT inferred from local offsets.
query checks chart support and local Jacobian, not global nearest-point uniqueness.
Object frames, origins, radii and chart domains are explicit encoded side data.
Reference curvature is curvature of a smooth fit, not of triangle edges.

Exit codes: 0 execution/qualification success, 2 measured qualification failure,
1 invalid input or execution failure. A completed training run is not a PASS.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

VERSION = "1.0"
DTYPE = torch.float64


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def powers(degree):
    return [(i, d - i) for d in range(degree + 1) for i in range(d + 1)]


def basis(uv, degree, du=0, dv=0):
    """Exact derivative of a fixed monomial basis; numpy or torch."""
    xp = torch if isinstance(uv, torch.Tensor) else np
    u, v = uv[..., 0], uv[..., 1]
    result = []
    for i, j in powers(degree):
        if i < du or j < dv:
            result.append(u * 0)
        else:
            factor = math.factorial(i) / math.factorial(i - du)
            factor *= math.factorial(j) / math.factorial(j - dv)
            result.append(factor * u ** (i - du) * v ** (j - dv))
    return xp.stack(result, axis=-1) if xp is np else xp.stack(result, dim=-1)


def geometry(c, uv, tau, degree, strict=True):
    """c [B,K], uv [B,Q,2], tau [B,Q]; all algebra supports backprop in c."""
    def deriv(i, j):
        return torch.einsum("bqk,bk->bq", basis(uv, degree, i, j), c)
    h, hu, hv = deriv(0, 0), deriv(1, 0), deriv(0, 1)
    huu, huv, hvv = deriv(2, 0), deriv(1, 1), deriv(0, 2)
    z, one = torch.zeros_like(h), torch.ones_like(h)
    q = torch.stack((-hu, -hv, one), -1)
    length = torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    n = q / length
    dq = torch.stack((torch.stack((-huu, -huv, z), -1),
                      torch.stack((-huv, -hvv, z), -1)), -1)
    dn = (dq - n[..., :, None] * (n[..., :, None] * dq).sum(-2, keepdim=True)) / length[..., None]
    tangent = torch.stack((torch.stack((one, z, hu), -1),
                           torch.stack((z, one, hv), -1)), -1)
    J = torch.cat((tangent + tau[..., None, None] * dn, n[..., :, None]), -1)
    determinant = torch.linalg.det(J)
    gs = tangent.transpose(-1, -2) @ tangent
    stretched = gs + tau[..., None, None] * (tangent.transpose(-1, -2) @ dn)
    # det(J)>0 alone would also accept crossing TWO focal points. Positive
    # principal stretches keep the connected normal-coordinate branch at t=0.
    stretches_positive = torch.linalg.eigvalsh((stretched + stretched.transpose(-1, -2)) * .5)[..., 0] > 1e-7
    valid = torch.isfinite(J).all(dim=(-2, -1)) & (determinant > 1e-7) & stretches_positive
    if strict and not bool(valid.all()):
        raise ValueError("Invalid predicted normal-coordinate chart; no singular metric is clamped into a pass.")
    # Invalid evaluations are explicitly counted and fail gates. The identity
    # substitute only prevents a diagnostic batch from failing inside inv().
    safe_J = torch.where(valid[..., None, None], J, torch.eye(3, dtype=J.dtype, device=J.device))
    g = safe_J.transpose(-1, -2) @ safe_J
    A = torch.linalg.det(safe_J).abs()[..., None, None] * torch.linalg.inv(g)
    p = torch.stack((uv[..., 0], uv[..., 1], h), -1)
    W = torch.linalg.solve(gs, tangent.transpose(-1, -2) @ dn)
    return dict(p=p, n=n, dn=dn, J=J, g=g, A=A, W=W,
                area=length[..., 0], valid=valid, det=determinant)


def tensor_geometry(c, uv, tau, degree):
    with torch.no_grad():
        result = geometry(torch.as_tensor(c[None], dtype=DTYPE),
                          torch.as_tensor(uv[None], dtype=DTYPE),
                          torch.as_tensor(tau[None], dtype=DTYPE), degree)
    return {k: v[0].cpu().numpy() for k, v in result.items()}


def fit_coefficients(points, degree):
    matrix = basis(points[:, :2], degree)
    # Column equilibration changes neither the least-squares objective nor the
    # polynomial. It avoids treating small chart coordinates as rank loss.
    scale = np.linalg.norm(matrix, axis=0)
    if np.any(scale < 1e-20):
        raise ValueError("Rank-deficient or ill-conditioned chart fit")
    c, _, rank, s = np.linalg.lstsq(matrix / scale, points[:, 2], rcond=1e-10)
    if rank != matrix.shape[1] or s[0] / s[-1] > 1e6:
        raise ValueError("Rank-deficient or ill-conditioned chart fit")
    return c / scale, float(s[0] / s[-1])


def offset_limit(c, degree, requested):
    """Conservative curvature bound on |u|,|v|<=1, not a global reach bound."""
    bounds = []
    for du, dv in ((2, 0), (1, 1), (0, 2)):
        bounds.append(sum(abs(a) * math.factorial(i) / math.factorial(i - du)
                          * math.factorial(j) / math.factorial(j - dv)
                          for a, (i, j) in zip(c, powers(degree)) if i >= du and j >= dv))
    H = math.sqrt(bounds[0] ** 2 + 2 * bounds[1] ** 2 + bounds[2] ** 2)
    return min(requested, .25 / max(H, 1e-12))


def triangle_samples(triangles, count, rng, projected=False):
    """Samples within a unit disk in the patch chart, without extrapolating."""
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = np.abs(cross[:, 2]) if projected else np.linalg.norm(cross, axis=1)
    if not np.isfinite(areas).all() or areas.sum() <= 1e-14:
        raise ValueError("Empty projected patch")
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-30)
    pieces, face_parts, total = [], [], 0
    for _ in range(200):
        size = max(64, 2 * (count - total))
        f = rng.choice(len(triangles), size=size, p=areas / areas.sum())
        b = rng.random((size, 2))
        outside = b.sum(1) > 1
        b[outside] = 1 - b[outside]
        pts = triangles[f, 0] + b[:, :1] * (triangles[f, 1] - triangles[f, 0]) + b[:, 1:] * (triangles[f, 2] - triangles[f, 0])
        keep = np.sum(pts[:, :2] ** 2, axis=1) < 1
        pieces.append(pts[keep]); face_parts.append(f[keep]); total += int(keep.sum())
        if total >= count:
            return np.concatenate(pieces)[:count], normals[np.concatenate(face_parts)[:count]]
    raise ValueError("Insufficient chart support in the unit disk")


def inside_domain(uv, triangles):
    """Membership in the projected triangle union intersected with unit disk."""
    a = triangles[:, 0, :2]
    e = triangles[:, 1, :2] - a
    f = triangles[:, 2, :2] - a
    det = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]
    good = np.abs(det) > 1e-14
    a, e, f, det = a[good], e[good], f[good], det[good]
    if not len(det):
        return np.zeros(len(uv), dtype=bool)
    q = uv[:, None] - a[None]
    b = (q[..., 0] * f[:, 1] - q[..., 1] * f[:, 0]) / det
    c = (e[:, 0] * q[..., 1] - e[:, 1] * q[..., 0]) / det
    return ((b >= -1e-10) & (c >= -1e-10) & (b + c <= 1 + 1e-10)).any(1) & ((uv ** 2).sum(1) <= 1 + 1e-10)


def frame_from_normal(n):
    n = n / np.linalg.norm(n)
    axis = np.eye(3)[np.argmin(np.abs(n))]
    e1 = np.cross(axis, n); e1 /= np.linalg.norm(e1)
    return np.stack((e1, np.cross(n, e1), n), axis=1)


def build_mesh_patches(path, args, rng):
    import trimesh
    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError(f"Not a triangle mesh: {path}")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"Nonfinite vertices: {path}")
    source_faces = len(mesh.faces)
    center = mesh.bounds.mean(0)
    scale = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    if scale <= 0:
        raise ValueError("Zero-size mesh")
    mesh.vertices = (mesh.vertices - center) / scale
    # GLB often duplicates vertices for material/normal seams. Rebuild geometry
    # connectivity before detecting sharp edges. Tolerance: 1e-12 mesh diagonal.
    mesh.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=12)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals(multibody=True)
    adjacency = [[] for _ in mesh.faces]
    cos_crease = math.cos(math.radians(args.crease_degrees))
    for a, b in mesh.face_adjacency:
        if float(mesh.face_normals[a] @ mesh.face_normals[b]) >= cos_crease:
            adjacency[a].append(int(b)); adjacency[b].append(int(a))
    areas = mesh.area_faces
    def surface_probes(count):
        faces = rng.choice(len(areas), count, p=areas / areas.sum())
        b = rng.random((count, 2)); b[b.sum(1) > 1] = 1 - b[b.sum(1) > 1]
        tri = mesh.triangles[faces]
        pts = tri[:, 0] + b[:, :1] * (tri[:, 1] - tri[:, 0]) + b[:, 1:] * (tri[:, 2] - tri[:, 0])
        return faces, pts
    # Place subsequent patches on uncovered area; allow multiple anchors on a
    # large triangle. Evaluation below uses an independent set of probes.
    design_faces, design_xyz = surface_probes(max(16384, 4 * args.coverage_points))
    covered = np.zeros(len(design_xyz), dtype=bool)
    eligible = np.ones(len(design_xyz), dtype=bool)
    accepted, failures = [], {}
    for attempt in range(args.patches * 8):
        remaining = np.flatnonzero(eligible & ~covered)
        if len(accepted) >= args.patches or not len(remaining) or covered.mean() >= args.target_coverage:
            break
        anchor = int(rng.choice(remaining))
        eligible[anchor] = False
        face = int(design_faces[anchor])
        origin = design_xyz[anchor]
        frame = frame_from_normal(mesh.face_normals[face])
        for shrink in range(4):
            radius = args.radius * (0.65 ** shrink)
            todo, seen, region = [face], {face}, []
            while todo and len(region) < args.max_faces:
                f = todo.pop(0)
                if np.min(np.linalg.norm(mesh.triangles[f] - origin, axis=1)) > radius * 1.8 and f != face:
                    continue
                if mesh.face_normals[f] @ frame[:, 2] < .5:
                    continue
                region.append(f)
                for other in adjacency[f]:
                    if other not in seen:
                        seen.add(other); todo.append(other)
            try:
                triangles = ((mesh.triangles[region] - origin) @ frame) / radius
                raw, _ = triangle_samples(triangles, args.fit_points, rng)
                c, condition = fit_coefficients(raw, args.degree)
                check, check_n = triangle_samples(triangles, args.fit_points, rng)
                ref = tensor_geometry(c, check[:, :2], np.zeros(len(check)), args.degree)
                fit_rms = float(np.sqrt(np.mean((ref["p"][:, 2] - check[:, 2]) ** 2)))
                angle = float(np.degrees(np.arccos(np.clip(np.abs((ref["n"] * check_n).sum(1)), 0, 1))).mean())
                if fit_rms > args.fit_tolerance or angle > args.fit_normal_degrees:
                    raise ValueError("Mesh fit exceeds position/normal tolerance")
                independent_c, _ = fit_coefficients(check, args.degree)
                limit = offset_limit(c, args.degree, args.offset)
                probes, _ = triangle_samples(triangles, 128, rng, projected=True)
                tau = rng.uniform(-limit, limit, len(probes))
                aa = tensor_geometry(c, probes[:, :2], tau, args.degree)["A"]
                bb = tensor_geometry(independent_c, probes[:, :2], tau, args.degree)["A"]
                eig, vec = np.linalg.eigh(aa)
                inverse_root = (vec * (1 / np.sqrt(eig))[:, None, :]) @ vec.transpose(0, 2, 1)
                err = inverse_root @ bb @ inverse_root - np.eye(3)
                stability = float(np.sqrt(np.mean(np.max(np.abs(np.linalg.eigvalsh(err)), axis=1) ** 2)))
                if stability > args.stability_tolerance:
                    raise ValueError("Independent mesh fits disagree on A")
                inp, inp_n = triangle_samples(triangles, args.input_points, rng)
                fit_coefficients(inp, args.degree)  # observable baseline must be identifiable
                accepted.append(dict(origin=origin, frame=frame, radius=radius, coeff=c,
                                     triangles=triangles, faces=np.asarray(region),
                                     inputs=np.concatenate((inp, inp_n), -1), offset=limit,
                                     fit_rms=fit_rms, fit_angle=angle, stability=stability,
                                     fit_condition=condition))
                local = (design_xyz - origin) @ frame / radius
                support = np.isin(design_faces, region) & ((local[:, :2] ** 2).sum(1) < 1)
                z = basis(local[support, :2], args.degree) @ c
                ids = np.flatnonzero(support)
                covered[ids[np.abs(z - local[support, 2]) <= 3 * args.fit_tolerance]] = True
                if len(accepted) % 32 == 0:
                    print(f"{Path(path).name}: {len(accepted)} patches, design coverage {covered.mean():.1%}", flush=True)
                break
            except ValueError as e:
                failures[str(e)] = failures.get(str(e), 0) + 1
    if not accepted:
        raise ValueError(f"No acceptable patches from {path}: {failures}")
    # Independent area-sampled mesh coverage. Being on a selected face alone
    # does not count: the point must lie inside the actual chart disk and fit.
    face_ids, xyz = surface_probes(args.coverage_points)
    coverage = np.zeros(len(xyz), dtype=bool)
    for patch in accepted:
        local = (xyz - patch["origin"]) @ patch["frame"] / patch["radius"]
        support = np.isin(face_ids, patch["faces"]) & ((local[:, :2] ** 2).sum(1) < 1)
        z = basis(local[support, :2], args.degree) @ patch["coeff"]
        ids = np.flatnonzero(support)
        coverage[ids[np.abs(z - local[support, 2]) <= 3 * args.fit_tolerance]] = True
    return accepted, dict(path=str(Path(path).resolve()), sha256=sha256(path),
                          center=center.tolist(), scale=scale, source_faces=source_faces,
                          faces=len(mesh.faces), vertices=len(mesh.vertices),
                          patches=len(accepted), requested_patches=args.patches,
                          sampled_coverage=float(coverage.mean()), coverage_points=len(xyz),
                          rejected_attempts=failures)


def prepare(args):
    rng = np.random.default_rng(args.seed)
    if args.manifest:
        entries = json.loads(Path(args.manifest).read_text())
        base = Path(args.manifest).resolve().parent
        for entry in entries:
            entry["path"] = str((base / entry["path"]).resolve())
    else:
        entries = [dict(path=p, split="train", group=sha256(p)) for p in args.meshes]
    groups, digests = {}, {}
    for e in entries:
        if e["split"] not in ("train", "val", "test"):
            raise ValueError("Manifest split must be train, val or test")
        if "group" not in e:
            raise ValueError("Manifest must specify a family/duplicate group for every mesh")
        digest = sha256(e["path"])
        for key, table in ((e["group"], groups), (digest, digests)):
            if key in table and table[key] != e["split"]:
                raise ValueError("A family group or duplicate file crosses dataset splits")
            table[key] = e["split"]
    if not any(e["split"] == "train" for e in entries):
        raise ValueError("No training geometry")
    # A JSON destination writes one independently loadable NPZ per mesh. This
    # keeps preparation and training memory bounded on a large mesh corpus.
    if Path(args.out).suffix == ".json":
        root = Path(args.out).resolve().parent / (Path(args.out).stem + "_shards")
        root.mkdir(parents=True, exist_ok=True)
        shards = []
        for i, e in enumerate(entries):
            child = copy.copy(args)
            child.manifest = None; child.meshes = [e["path"]]
            child.out = str(root / f"mesh_{i:07d}.npz")
            child.seed = args.seed + i
            prepare(child)
            with np.load(child.out, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata"]))
                count = len(z["coeff"])
            record = meta["meshes"][0]
            record.update(split=e["split"], group=e["group"])
            shards.append(dict(path=os.path.relpath(child.out, Path(args.out).resolve().parent),
                               patches=count, sha256=sha256(child.out), mesh=record))
        json_write(args.out, dict(version=VERSION, degree=args.degree, shards=shards))
        print(f"Saved {len(shards)} mesh shards: {args.out}")
        return 0
    if Path(args.out).suffix != ".npz":
        raise ValueError("Use --out cache.npz for a compact dataset or cache.json for a sharded corpus")
    all_patches, records, mesh_index, splits = [], [], [], []
    for i, e in enumerate(entries):
        patches, report = build_mesh_patches(e["path"], args, rng)
        report.update(split=e["split"], group=e["group"])
        print(json.dumps(report), flush=True)
        all_patches.extend(patches); records.append(report)
        mesh_index.extend([i] * len(patches)); splits.extend([("train", "val", "test").index(e["split"])] * len(patches))
    sizes = [len(p["triangles"]) for p in all_patches]
    offsets = np.concatenate(([0], np.cumsum(sizes)))
    meta = dict(version=VERSION, degree=args.degree, seed=args.seed, meshes=records,
                setup={k: v for k, v in vars(args).items() if k != "func"},
                scope="complete_geometry_local_charts", offset_is_global_distance=False,
                spatial_coordinate_units="mesh_diagonal_normalized_lengths_s1_s2_t")
    arrays = {key: np.asarray([p[key] for p in all_patches]) for key in
              ("origin", "frame", "radius", "coeff", "inputs", "offset", "fit_rms", "fit_angle", "stability", "fit_condition")}
    arrays.update(triangles=np.concatenate([p["triangles"] for p in all_patches]),
                  triangle_offsets=offsets, mesh_index=np.array(mesh_index), split=np.array(splits),
                  metadata=np.array(json.dumps(meta)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    json_write(str(args.out) + ".json", meta)
    print(f"Prepared {len(all_patches)} patches. Reference coverage is reported, not assumed.")


def load_data(path):
    if Path(path).suffix == ".json":
        return ShardedData(path)
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    data["meta"] = json.loads(str(data["metadata"]))
    if data["meta"]["version"] != VERSION:
        raise ValueError("Unsupported cache format")
    return data


class ShardedColumn:
    def __init__(self, parent, key):
        self.parent, self.key = parent, key

    def __len__(self):
        return int(self.parent.ends[-1])

    def __getitem__(self, index):
        extra = ()
        if isinstance(index, tuple):
            index, extra = index[0], index[1:]
        if isinstance(index, slice):
            index = np.arange(*index.indices(len(self)))
        scalar = np.isscalar(index)
        ids = np.atleast_1d(index).astype(np.int64)
        if (ids < 0).any() or (ids >= len(self)).any():
            raise IndexError("Global patch index out of bounds")
        parts = []
        for i in ids:
            shard, local = self.parent.locate(int(i))
            parts.append(self.parent.get_shard(shard)[self.key][local])
        result = parts[0] if scalar else np.stack(parts)
        return result[extra] if scalar and extra else result[(slice(None),) + extra] if extra else result


class ShardedData:
    def __init__(self, path):
        self.path = Path(path).resolve()
        manifest = json.loads(self.path.read_text())
        if manifest["version"] != VERSION or not manifest["shards"]:
            raise ValueError("Invalid sharded dataset")
        self.shards = manifest["shards"]
        counts = [s["patches"] for s in self.shards]
        self.ends = np.cumsum(counts)
        self.cache = OrderedDict()
        self.verified = set()
        self.meta = dict(version=VERSION, degree=manifest["degree"], meshes=[s["mesh"] for s in self.shards])
        self.mesh_index = np.repeat(np.arange(len(counts)), counts)
        self.split = np.repeat([("train", "val", "test").index(s["mesh"]["split"]) for s in self.shards], counts)

    def locate(self, i):
        shard = int(np.searchsorted(self.ends, i, side="right"))
        return shard, i - (int(self.ends[shard - 1]) if shard else 0)

    def get_shard(self, i):
        if i not in self.cache:
            entry = self.shards[i]
            path = self.path.parent / entry["path"]
            if i not in self.verified:
                if sha256(path) != entry["sha256"]:
                    raise ValueError(f"Shard changed after preparation: {path}")
                self.verified.add(i)
            self.cache[i] = load_data(path)
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(i)
        return self.cache[i]

    def __getitem__(self, key):
        if key == "meta":
            return self.meta
        if key == "split":
            return self.split
        if key == "mesh_index":
            return self.mesh_index
        return ShardedColumn(self, key)


def triangles_for(data, i):
    if isinstance(data, ShardedData):
        shard, local = data.locate(int(i))
        return triangles_for(data.get_shard(shard), local)
    a, b = data["triangle_offsets"][i:i + 2]
    return data["triangles"][a:b]


def fresh_queries(data, ids, count, rng):
    uv = np.stack([triangle_samples(triangles_for(data, i), count, rng, projected=True)[0][:, :2] for i in ids])
    # Always include the surface and both offset signs.
    tau = rng.uniform(.1, 1, (len(ids), count)) * data["offset"][ids, None]
    tau[:, 0::3] = 0
    tau[:, 1::3] *= -1
    return uv, tau


class AttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.width, self.heads = width, heads
        self.qnorm, self.vnorm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.q, self.kv, self.out = nn.Linear(width, width), nn.Linear(width, 2 * width), nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, q, v, weights=None):
        def split(x):
            return x.reshape(x.shape[0], x.shape[1], self.heads, self.width // self.heads).transpose(1, 2)
        query = split(self.q(self.qnorm(q)))
        k, value = self.kv(self.vnorm(v)).chunk(2, -1)
        scores = query @ split(k).transpose(-1, -2) / math.sqrt(self.width // self.heads)
        if weights is not None:
            scores = scores + weights.clamp_min(1e-30).log()[:, None, None, :]
        result = (scores.softmax(-1) @ split(value)).transpose(1, 2).reshape(q.shape)
        q = q + self.out(result)
        return q + self.ff(q)


class PatchEncoder(nn.Module):
    def __init__(self, degree=3, width=192, heads=4, depth=4, latents=4):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.config = dict(degree=degree, width=width, heads=heads, depth=depth, latents=latents)
        self.embed = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.Linear(width, width))
        self.tokens = nn.Parameter(torch.randn(1, latents, width) * .02)
        self.read = AttentionBlock(width, heads)
        self.blocks = nn.ModuleList([AttentionBlock(width, heads) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, len(powers(degree))))
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, weights=None):
        p, n = x[..., :3], x[..., 3:]
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        projector = torch.stack((n[..., 0] ** 2, n[..., 1] ** 2, n[..., 2] ** 2,
                                 math.sqrt(2) * n[..., 0] * n[..., 1],
                                 math.sqrt(2) * n[..., 0] * n[..., 2],
                                 math.sqrt(2) * n[..., 1] * n[..., 2]), -1)
        features = self.embed(torch.cat((p, projector), -1))
        h = self.read(self.tokens.expand(len(x), -1, -1), features, weights)
        for block in self.blocks:
            h = block(h, h)
        return self.head(h.mean(1))


def weighted_mean(x, weights):
    return (x * weights).sum() / weights.sum()


def batch_values(data, ids, uv, tau, device):
    return (torch.as_tensor(data["inputs"][ids], dtype=torch.float32, device=device),
            torch.as_tensor(data["coeff"][ids], dtype=DTYPE, device=device),
            torch.as_tensor(uv, dtype=DTYPE, device=device),
            torch.as_tensor(tau, dtype=DTYPE, device=device),
            torch.as_tensor(data["radius"][ids, None], dtype=DTYPE, device=device))


def objective(model, data, ids, uv, tau, multiplier, rho, position_tolerance, device):
    inputs, truth, uv, tau, radii = batch_values(data, ids, uv, tau, device)
    predicted = model(inputs).to(DTYPE)
    reference = geometry(truth, uv, tau, model.config["degree"])
    learned = geometry(predicted, uv, tau, model.config["degree"])
    w = reference["area"].detach()
    LA = weighted_mean((learned["A"] - reference["A"]).square().sum((-1, -2)), w)
    Lp = weighted_mean((learned["p"] - reference["p"]).square().sum(-1) * radii.square(), w)
    constraint = Lp / position_tolerance ** 2 - 1
    penalty = (torch.relu(multiplier + rho * constraint).square() - multiplier ** 2) / (2 * rho)
    return LA + penalty, dict(A=LA, p=Lp, constraint=constraint)


@torch.no_grad()
def predictions(model, data, ids, device, batch_size=16):
    model.eval()
    return np.concatenate([model(torch.as_tensor(data["inputs"][part], dtype=torch.float32, device=device)).double().cpu().numpy()
                           for part in (ids[i:i + batch_size] for i in range(0, len(ids), batch_size))])


def coefficient_metrics(data, ids, coeffs, queries, seed):
    rng = np.random.default_rng(seed)
    degree = data["meta"]["degree"]
    sums = dict(weight=0., A=0., relative=0., p=0., angle=0.)
    maximum, invalid, total = 0., 0, 0
    per_patch = []
    for i, predicted in zip(ids, coeffs):
        uv, tau = fresh_queries(data, [i], queries, rng)
        ref = tensor_geometry(data["coeff"][i], uv[0], tau[0], degree)
        with torch.no_grad():
            pred = geometry(torch.tensor(predicted[None], dtype=DTYPE), torch.tensor(uv, dtype=DTYPE),
                            torch.tensor(tau, dtype=DTYPE), degree, strict=False)
        pred = {k: v[0].numpy() for k, v in pred.items()}
        w = ref["area"]
        eig, vec = np.linalg.eigh(ref["A"])
        inverse_root = (vec * (1 / np.sqrt(eig))[:, None, :]) @ vec.transpose(0, 2, 1)
        error = inverse_root @ pred["A"] @ inverse_root - np.eye(3)
        relative = np.max(np.abs(np.linalg.eigvalsh(error)), axis=-1)
        angle = np.degrees(np.arccos(np.clip(np.abs((pred["n"] * ref["n"]).sum(1)), 0, 1)))
        p2 = np.sum((pred["p"] - ref["p"]) ** 2, axis=1) * data["radius"][i] ** 2
        a2 = np.sum((pred["A"] - ref["A"]) ** 2, axis=(1, 2))
        for name, values in (("A", a2), ("relative", relative ** 2), ("p", p2), ("angle", angle)):
            sums[name] += float(np.sum(values * w))
        sums["weight"] += float(w.sum())
        maximum = max(maximum, float(relative.max()))
        invalid += int((~pred["valid"]).sum()); total += len(w)
        per_patch.append(dict(patch=int(i), position_rms=float(np.sqrt(np.average(p2, weights=w))),
                              relative_A_rms=float(np.sqrt(np.average(relative ** 2, weights=w)))))
    return dict(A_mse=sums["A"] / sums["weight"],
                relative_A_rms=math.sqrt(sums["relative"] / sums["weight"]),
                relative_A_max=maximum, position_rms=math.sqrt(sums["p"] / sums["weight"]),
                normal_line_degrees=sums["angle"] / sums["weight"], invalid_maps=invalid,
                evaluated_points=total, patch_count=len(ids), per_patch=per_patch)


def seam_metrics(data, ids, coeffs, seed, samples=24):
    """Check overlapping fitted charts at shared mesh-projected chart locations.

    No invented global surface: report reference and learned discrepancies.
    The same mesh sample defines chart correspondence on both sides.
    """
    rng = np.random.default_rng(seed)
    degree = data["meta"]["degree"]
    ref_err, pred_err = [], []
    lookup = {int(i): c for i, c in zip(ids, coeffs)}
    for i in ids:
        local, _ = triangle_samples(triangles_for(data, i), samples, rng)
        world = data["origin"][i] + data["radius"][i] * local @ data["frame"][i].T
        near = ids[(data["mesh_index"][ids] == data["mesh_index"][i]) & (ids > i)]
        if len(near):
            dist = np.linalg.norm(data["origin"][near] - data["origin"][i], axis=1)
            near = near[dist < data["radius"][near] + data["radius"][i]]
        for j in near:
            other = (world - data["origin"][j]) @ data["frame"][j] / data["radius"][j]
            keep = inside_domain(other[:, :2], triangles_for(data, j))
            # Other-side sheets and charts across a crease are not seam partners.
            keep &= np.abs(other[:, 2] - basis(other[:, :2], degree) @ data["coeff"][j]) < .03
            if data["frame"][i, :, 2] @ data["frame"][j, :, 2] < math.cos(math.radians(30)):
                continue
            if not keep.any():
                continue
            pair_ref, pair_pred = [], []
            for k, points in ((i, local[keep]), (j, other[keep])):
                uv = points[:, :2]
                for cs, dest in ((data["coeff"][k], pair_ref), (lookup[int(k)], pair_pred)):
                    p = np.column_stack((uv, basis(uv, degree) @ cs))
                    dest.append(data["origin"][k] + data["radius"][k] * p @ data["frame"][k].T)
            ref_err.extend(np.sum((pair_ref[0] - pair_ref[1]) ** 2, axis=1))
            pred_err.extend(np.sum((pair_pred[0] - pair_pred[1]) ** 2, axis=1))
    return dict(samples=len(ref_err), reference_rms=math.sqrt(float(np.mean(ref_err))) if ref_err else None,
                learned_rms=math.sqrt(float(np.mean(pred_err))) if pred_err else None)


def gates(metrics, position_tolerance=.01, normal_tolerance=5., operator_rms=.01, operator_max=.05):
    return dict(position=metrics["position_rms"] <= position_tolerance,
                normals=metrics["normal_line_degrees"] <= normal_tolerance,
                coefficient_rms=metrics["relative_A_rms"] <= operator_rms,
                coefficient_max=metrics["relative_A_max"] <= operator_max,
                valid_maps=metrics["invalid_maps"] == 0)


def choose_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu")
    return torch.device(name)


def save_checkpoint(path, model, optimizer, step, multiplier, rng, config, data_hash, best):
    payload = dict(version=VERSION, config=config, model_config=model.config,
                   model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                   multiplier=multiplier, numpy_rng=json.dumps(rng.bit_generator.state),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                   data_sha256=data_hash, best=best)
    tmp = Path(str(path) + ".tmp"); torch.save(payload, tmp); tmp.replace(path)


def restore_model(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["version"] != VERSION:
        raise ValueError("Wrong checkpoint version")
    model = PatchEncoder(**state["model_config"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    return model, state


def train(args):
    device = choose_device(args.device)
    data = load_data(args.data); data_hash = sha256(args.data)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train_ids = np.flatnonzero(data["split"] == 0)
    if not len(train_ids):
        raise ValueError("No training patches")
    reference_coverage = min(m["sampled_coverage"] for m in data["meta"]["meshes"] if m["split"] == "train")
    if reference_coverage < args.coverage and not args.local_only:
        raise ValueError(f"Reference coverage {reference_coverage:.2%} is below {args.coverage:.2%}. Increase the patch cap or resolve rejected fits before whole-mesh training. --local-only explicitly permits a patch-only diagnostic.")
    val_ids = np.flatnonzero(data["split"] == 1)
    scope = "held_out_objects" if len(val_ids) else "fresh_queries_same_training_objects"
    if not len(val_ids):
        val_ids = train_ids.copy()
    if len(val_ids) > args.eval_patches:
        val_ids = np.sort(np.random.default_rng(args.seed + 100003).choice(val_ids, args.eval_patches, replace=False))
    mesh_ids, starts, counts = np.unique(data["mesh_index"][train_ids], return_index=True, return_counts=True)
    by_mesh = {int(m): train_ids[start:start + count] for m, start, count in zip(mesh_ids, starts, counts)}
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    config = {k: v for k, v in vars(args).items() if k not in ("func", "resume", "stop_after")}
    model = PatchEncoder(data["meta"]["degree"], args.width, args.heads, args.depth, args.latents).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    start, multiplier, best = 0, 0., float("inf")
    if args.resume:
        model, state = restore_model(out / "last.pt", device)
        if state["data_sha256"] != data_hash or state["config"] != config:
            raise ValueError("Resume data/config mismatch; preserve the original planned --steps and settings")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
        optimizer.load_state_dict(state["optimizer"])
        start, multiplier, best = state["step"], state["multiplier"], state["best"]
        rng.bit_generator.state = json.loads(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    elif (out / "last.pt").exists():
        raise ValueError("Output already contains a run; use --resume or a fresh --out")
    json_write(out / "config.json", dict(**config, parameters=sum(p.numel() for p in model.parameters()), validation_scope=scope))
    print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters on {device}; validation: {scope}", flush=True)
    end = min(args.steps, args.stop_after) if args.stop_after else args.steps
    log = open(out / "train.jsonl", "a" if args.resume else "w", buffering=1)
    started = time.monotonic()
    try:
        for step in range(start + 1, end + 1):
            model.train()
            # A few objects per batch keeps the streaming shard cache bounded.
            chosen_meshes = rng.choice(mesh_ids, min(args.objects_per_batch, len(mesh_ids), args.batch_size), replace=False)
            per_object = math.ceil(args.batch_size / len(chosen_meshes))
            ids = np.concatenate([rng.choice(by_mesh[int(m)], per_object, replace=True) for m in chosen_meshes])[:args.batch_size]
            uv, tau = fresh_queries(data, ids, args.queries, rng)
            optimizer.zero_grad(set_to_none=True)
            loss, terms = objective(model, data, ids, uv, tau, multiplier, args.rho, args.position_tolerance, device)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite objective")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            optimizer.step()
            # Dual ascent uses training samples only, following the specified
            # augmented-Lagrangian step; validation never updates parameters.
            multiplier = max(0., multiplier + args.rho * float(terms["constraint"].detach()))
            record = dict(step=step, loss=float(loss.detach()), gradient_norm=norm, multiplier=multiplier,
                          **{k: float(v.detach()) for k, v in terms.items()})
            log.write(json.dumps(record) + "\n")
            if step % args.validate_every == 0 or step == args.steps:
                coeff = predictions(model, data, val_ids, device)
                metrics = coefficient_metrics(data, val_ids, coeff, args.eval_queries, args.seed + 1000003)
                feasible = metrics["position_rms"] <= args.position_tolerance and metrics["invalid_maps"] == 0
                if feasible and metrics["A_mse"] < best:
                    best = metrics["A_mse"]
                    save_checkpoint(out / "best.pt", model, optimizer, step, multiplier, rng, config, data_hash, best)
                brief = {k: v for k, v in metrics.items() if k != "per_patch"}
                print(json.dumps(dict(step=step, validation=brief, elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
                json_write(out / "validation.json", dict(scope=scope, step=step, **metrics))
            if step % args.validate_every == 0 or step == end:
                save_checkpoint(out / "last.pt", model, optimizer, step, multiplier, rng, config, data_hash, best)
    finally:
        log.close()
    if end < args.steps:
        print(f"Paused at {end}; resume with identical settings plus --resume.")
        return 0
    chosen = out / "best.pt" if (out / "best.pt").exists() else out / "last.pt"
    eval_args = argparse.Namespace(data=args.data, checkpoint=str(chosen), device=args.device,
                                   queries=args.eval_queries, seed=args.seed + 2000003,
                                   split="test", out=str(out / "qualification.json"),
                                   position_tolerance=args.position_tolerance, coverage=args.coverage,
                                   max_patches=args.eval_patches)
    return evaluate(eval_args)


def evaluate(args):
    device = choose_device(args.device)
    data = load_data(args.data)
    model, state = restore_model(args.checkpoint, device)
    if model.config["degree"] != data["meta"]["degree"]:
        raise ValueError("Polynomial degree mismatch")
    if args.split == "all":
        ids = np.arange(len(data["coeff"]))
    else:
        ids = np.flatnonzero(data["split"] == ("train", "val", "test").index(args.split))
    actual_split = args.split
    if not len(ids):
        if args.split != "test":
            raise ValueError("Requested split is empty")
        ids = np.flatnonzero(data["split"] == 0)
        actual_split = "fresh_queries_same_training_objects"
    total_patches = len(ids)
    if args.max_patches and len(ids) > args.max_patches:
        ids = np.sort(np.random.default_rng(args.seed + 3).choice(ids, args.max_patches, replace=False))
    before = sha256(args.checkpoint)
    coeff = predictions(model, data, ids, device)
    metrics = coefficient_metrics(data, ids, coeff, args.queries, args.seed)
    analytic = np.stack([fit_coefficients(data["inputs"][i, :, :3], data["meta"]["degree"])[0] for i in ids])
    baseline = coefficient_metrics(data, ids, analytic, args.queries, args.seed)
    seams = seam_metrics(data, ids, coeff, args.seed + 1)
    checks = gates(metrics, args.position_tolerance)
    selected_meshes = sorted(set(map(int, data["mesh_index"][ids])))
    coverage = min(data["meta"]["meshes"][i]["sampled_coverage"] for i in selected_meshes)
    local_pass = all(checks.values())
    checks["sampled_mesh_coverage"] = coverage >= args.coverage
    checks["sampled_seams"] = seams["learned_rms"] is not None and seams["learned_rms"] <= args.position_tolerance
    result = dict(version=VERSION, checkpoint_sha256=before, checkpoint_step=state["step"],
                  data_sha256=sha256(args.data), evaluated_split=actual_split,
                  available_split_patches=total_patches, evaluated_patch_subset=len(ids) < total_patches,
                  seed=args.seed, learned=metrics, analytic_input_only_baseline=baseline,
                  seams=seams, minimum_sampled_mesh_coverage=coverage, gates=checks,
                  local_reconstruction_pass=local_pass, sampled_atlas_pass=all(checks.values()),
                  global_surface_certified=False, global_distance_certified=False, cfd_transfer_measured=False,
                  reference_fit_rms_max=float(data["fit_rms"][ids].max()),
                  reference_A_stability_max=float(data["stability"][ids].max()))
    if sha256(args.checkpoint) != before:
        raise AssertionError("Evaluation modified checkpoint")
    json_write(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("learned", "analytic_input_only_baseline")}, indent=2))
    return 0 if result["sampled_atlas_pass"] else 2


def encode(args):
    data = load_data(args.data); device = choose_device(args.device)
    if isinstance(data, ShardedData):
        destination = Path(args.out); destination.mkdir(parents=True, exist_ok=True)
        encoded = []
        for i, shard in enumerate(data.shards):
            child = copy.copy(args)
            child.data = str(data.path.parent / shard["path"])
            child.out = str(destination / f"mesh_{i:07d}.npz")
            encode(child)
            encoded.append(dict(path=Path(child.out).name, source_mesh=shard["mesh"]["path"]))
        json_write(destination / "index.json", dict(encoded=encoded))
        return 0
    model, state = restore_model(args.checkpoint, device)
    ids = np.arange(len(data["inputs"]))
    coeff = predictions(model, data, ids, device)
    # Deliberately omit targets, raw inputs and reference curvature from packet.
    packet = {k: data[k] for k in ("origin", "frame", "radius", "mesh_index", "triangle_offsets")}
    # Only the projected chart domain is needed, not mesh height or mesh normals.
    packet["domain_uv"] = data["triangles"][..., :2]
    packet["coeff"] = coeff
    packet["metadata"] = np.array(json.dumps(dict(version=VERSION, degree=model.config["degree"],
                                                  meshes=data["meta"]["meshes"], checkpoint_sha256=sha256(args.checkpoint))))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **packet)
    scalars = sum(v.size for k, v in packet.items() if k != "metadata")
    print(json.dumps(dict(encoded_patches=len(coeff), array_scalars_including_domains_and_frames=scalars,
                          additional_metadata_bytes=len(str(packet["metadata"]).encode()),
                          file_bytes=Path(args.out).stat().st_size, out=args.out)))
    return 0


def check_charts(args):
    data = load_data(args.data)
    ids = np.flatnonzero(data["split"] == 0)
    if args.max_patches and len(ids) > args.max_patches:
        ids = np.sort(np.random.default_rng(args.seed).choice(ids, args.max_patches, replace=False))
    rng = np.random.default_rng(args.seed + 7)
    c = []
    for i in ids:
        points, _ = triangle_samples(triangles_for(data, i), max(128, args.queries), rng, projected=True)
        points[:, 2] = basis(points[:, :2], data["meta"]["degree"]) @ data["coeff"][i]
        fitted, _ = fit_coefficients(points, data["meta"]["degree"])
        c.append(fitted)
    c = np.stack(c)
    oracle = coefficient_metrics(data, ids, c, args.queries, args.seed)
    fitted = np.stack([fit_coefficients(data["inputs"][i, :, :3], data["meta"]["degree"])[0] for i in ids])
    observable = coefficient_metrics(data, ids, fitted, args.queries, args.seed)
    seams = seam_metrics(data, ids, c, args.seed + 1)
    coverage = min(m["sampled_coverage"] for m in data["meta"]["meshes"] if m["split"] == "train")
    result = dict(reference_round_trip=oracle, input_only_analytic_fit=observable,
                  reference_seams=seams, minimum_sampled_training_coverage=coverage,
                  evaluated_patches=len(ids), neural_training_performed=False,
                  passed=all(gates(oracle).values()) and coverage >= args.coverage
                         and seams["reference_rms"] is not None and seams["reference_rms"] <= .01)
    json_write(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("reference_round_trip", "input_only_analytic_fit")}, indent=2))
    return 0 if result["passed"] else 2


def query(args):
    with np.load(args.encoded, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(str(data["metadata"])); i = args.patch
    if not 0 <= i < len(data["coeff"]):
        raise ValueError("Patch index out of range")
    uv = np.array([[args.u, args.v]], dtype=np.float64)
    a, b = data["triangle_offsets"][i:i + 2]
    domain = np.concatenate((data["domain_uv"][a:b], np.zeros((b - a, 3, 1))), -1)
    if not inside_domain(uv, domain)[0]:
        raise ValueError("Query is outside encoded chart support")
    r = data["radius"][i]
    mesh = meta["meshes"][int(data["mesh_index"][i])]
    # --offset is in the ORIGINAL mesh's physical units.
    t = args.offset / mesh["scale"]
    geo = tensor_geometry(data["coeff"][i], uv, np.array([t / r]), meta["degree"])
    p = data["origin"][i] + r * geo["p"][0] @ data["frame"][i].T
    n = geo["n"][0] @ data["frame"][i].T
    world_p = p * mesh["scale"] + np.asarray(mesh["center"])
    result = dict(p=world_p.tolist(), normal=n.tolist(), x=(world_p + args.offset * n).tolist(),
                  g=geo["g"][0].tolist(), A=geo["A"][0].tolist(),
                  principal_curvatures=np.sort(np.linalg.eigvals(geo["W"][0]).real / (r * mesh["scale"])).tolist(),
                  half_normal_offset_squared=.5 * args.offset ** 2,
                  global_nearest_point_verified=False)
    if args.out:
        json_write(args.out, result)
    print(json.dumps(result, indent=2))
    return 0


def selftest(args):
    """Independent geometry identities, finite differences, and invariances."""
    torch.manual_seed(17); rng = np.random.default_rng(17)
    degree = 3; k = len(powers(degree)); checks = {}
    uv = torch.tensor(rng.uniform(-.5, .5, (1, 19, 2)), dtype=DTYPE)
    tau = torch.tensor(rng.uniform(-.2, .2, (1, 19)), dtype=DTYPE)
    zero = torch.zeros((1, k), dtype=DTYPE)
    plane = geometry(zero, uv, tau, degree)
    checks["plane_A_identity"] = float((plane["A"] - torch.eye(3)).abs().max())
    c = torch.tensor(rng.normal(0, .04, (1, k)), dtype=DTYPE, requires_grad=True)
    geo = geometry(c, uv, tau, degree)
    step = 1e-5
    dn_fd, j_fd = [], []
    for axis in range(2):
        delta = torch.zeros_like(uv); delta[..., axis] = step
        plus, minus = geometry(c, uv + delta, tau, degree), geometry(c, uv - delta, tau, degree)
        dn_fd.append((plus["n"] - minus["n"]) / (2 * step))
        j_fd.append((plus["p"] + tau[..., None] * plus["n"] - minus["p"] - tau[..., None] * minus["n"]) / (2 * step))
    j_fd.append(geo["n"])
    checks["normal_spatial_fd"] = float((torch.stack(dn_fd, -1) - geo["dn"]).abs().max().detach())
    checks["coordinate_jacobian_fd"] = float((torch.stack(j_fd, -1) - geo["J"]).abs().max().detach())
    # Independent energy transformation: physical gradient = J^{-T} v.
    v = torch.tensor(rng.normal(size=(1, 19, 3)), dtype=DTYPE)
    physical = torch.linalg.solve(geo["J"].transpose(-1, -2), v[..., None])[..., 0]
    direct = torch.linalg.det(geo["J"]).abs() * physical.square().sum(-1)
    via_A = torch.einsum("bqi,bqij,bqj->bq", v, geo["A"], v)
    checks["physical_energy_identity"] = float((direct - via_A).abs().max().detach())
    # Sphere/cylinder jets at the north pole: h''=-1/R, W=+I/R.
    for name, axes in (("sphere", (0, 1)), ("cylinder", (0,))):
        cc = zero.clone(); radius = 2.
        for axis in axes:
            cc[0, powers(degree).index((2, 0) if axis == 0 else (0, 2))] = -1 / (2 * radius)
        tt = torch.tensor([[.3]], dtype=DTYPE)
        out = geometry(cc, torch.zeros((1, 1, 2), dtype=DTYPE), tt, degree)
        stretch = [1 + .3 / radius if axis in axes else 1 for axis in (0, 1)]
        expected = torch.diag(torch.tensor([stretch[1] / stretch[0], stretch[0] / stretch[1], stretch[0] * stretch[1]], dtype=DTYPE))
        checks[name + "_pole_A"] = float((out["A"][0, 0] - expected).abs().max())
    loss = geo["A"].square().sum() + geo["p"].square().sum()
    gradient = torch.autograd.grad(loss, c)[0]
    fd = []
    for a in range(k):
        delta = torch.zeros_like(c); delta[0, a] = step
        vals = []
        for cc in (c.detach() + delta, c.detach() - delta):
            g = geometry(cc, uv, tau, degree); vals.append(g["A"].square().sum() + g["p"].square().sum())
        fd.append((vals[0] - vals[1]) / (2 * step))
    checks["coefficient_gradient_fd"] = float((torch.stack(fd) - gradient[0]).abs().max())
    model = PatchEncoder(degree, width=24, heads=4, depth=1, latents=2).double()
    inp = torch.tensor(rng.normal(size=(2, 20, 6)), dtype=DTYPE)
    out = model(inp)
    shuffled = model(inp[:, rng.permutation(20)])
    flipped = inp.clone(); flipped[..., 3:] *= torch.tensor(rng.choice([-1, 1], size=(2, 20, 1)))
    checks["point_permutation"] = float((out - shuffled).abs().max().detach())
    checks["normal_sign"] = float((out - model(flipped)).abs().max().detach())
    duplicate = torch.repeat_interleave(inp, 2, dim=1)
    checks["uniform_density_duplication"] = float((out - model(duplicate)).abs().max().detach())
    duplicate_one = torch.cat((inp[:, :1], inp), 1)
    weights = torch.ones(duplicate_one.shape[:2], dtype=DTYPE); weights[:, :2] = .5
    checks["area_weighted_density_change"] = float((out - model(duplicate_one, weights)).abs().max().detach())
    # End-to-end parameter finite difference, including curvature and matrix inverse.
    def neural_loss():
        g = geometry(model(inp), uv.expand(2, -1, -1), tau.expand(2, -1), degree)
        return (g["A"] - geo["A"].detach()).square().mean() + (g["p"] - geo["p"].detach()).square().mean()
    L = neural_loss(); L.backward()
    checks["all_parameter_gradients_finite"] = all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    parameter = model.embed[0].weight; index = (0, 0); analytic = parameter.grad[index].item()
    with torch.no_grad():
        original = parameter[index].item(); parameter[index] = original + step
        pos = neural_loss().item(); parameter[index] = original - step
        neg = neural_loss().item(); parameter[index] = original
    checks["encoder_parameter_gradient_fd"] = abs(analytic - (pos - neg) / (2 * step))
    # Linear reference round trip and a deliberately corrupted coefficient.
    points = rng.uniform(-.7, .7, (200, 2)); truth = c.detach().numpy()[0]
    fit, _ = fit_coefficients(np.column_stack((points, basis(points, degree) @ truth)), degree)
    checks["analytic_fit_round_trip"] = float(np.max(np.abs(fit - truth)))
    corrupt = zero.clone(); corrupt[0, powers(degree).index((2, 0))] = 2.
    bad = geometry(corrupt, uv, tau * 0, degree)
    checks["corruption_detected"] = bool((bad["A"] - plane["A"]).abs().max() > .1)
    focal = zero.clone()
    for power in ((2, 0), (0, 2)):
        focal[0, powers(degree).index(power)] = -.25
    folded = geometry(focal, torch.zeros((1, 1, 2), dtype=DTYPE), torch.tensor([[-3.]], dtype=DTYPE), degree, strict=False)
    checks["two_focal_crossings_rejected"] = not bool(folded["valid"].all())
    rotation, _ = torch.linalg.qr(torch.tensor(rng.normal(size=(3, 3)), dtype=DTYPE))
    rotated = rotation @ geo["J"]
    rotated_g = rotated.transpose(-1, -2) @ rotated
    checks["rigid_frame_metric_invariance"] = float((rotated_g - geo["g"]).abs().max().detach())
    translated = zero.clone(); translated[0, 0] = .5
    shift = geometry(translated, uv, tau, degree)
    checks["position_constraint_detects_A_invisible_translation"] = bool(
        torch.allclose(shift["A"], plane["A"]) and (shift["p"] - plane["p"]).abs().max() > .4)
    numerical = [v for v in checks.values() if not isinstance(v, bool)]
    passed = max(numerical) < 1e-6 and all(v for v in checks.values() if isinstance(v, bool))
    result = dict(passed=passed, checks=checks, torch=str(torch.__version__), cuda_tested=False)
    if args.out:
        json_write(args.out, result)
    print(json.dumps(result, indent=2))
    return 0 if passed else 2


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", type=int, default=1, help="CPU threads; before subcommand")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("selftest"); s.add_argument("--out"); s.set_defaults(func=selftest)
    s = sub.add_parser("prepare")
    source = s.add_mutually_exclusive_group(required=True)
    source.add_argument("--meshes", nargs="+"); source.add_argument("--manifest")
    s.add_argument("--out", required=True); s.add_argument("--seed", type=int, default=17)
    s.add_argument("--patches", type=int, default=512); s.add_argument("--radius", type=float, default=.12)
    s.add_argument("--degree", type=int, choices=(2, 3, 4), default=3)
    s.add_argument("--input-points", type=int, default=128); s.add_argument("--fit-points", type=int, default=1024)
    s.add_argument("--max-faces", type=int, default=128); s.add_argument("--crease-degrees", type=float, default=35)
    s.add_argument("--fit-tolerance", type=float, default=.01, help="Height RMS / patch radius")
    s.add_argument("--fit-normal-degrees", type=float, default=10)
    s.add_argument("--stability-tolerance", type=float, default=.05, help="Independent fits' relative A RMS")
    s.add_argument("--offset", type=float, default=.1, help="Max |t|/patch radius before curvature bound")
    s.add_argument("--coverage-points", type=int, default=8192); s.set_defaults(func=prepare)
    s.add_argument("--target-coverage", type=float, default=.98, help="Design coverage goal, subject to --patches cap")
    s = sub.add_parser("train")
    s.add_argument("--data", required=True); s.add_argument("--out", required=True)
    s.add_argument("--device", default="auto"); s.add_argument("--steps", type=int, default=20000)
    s.add_argument("--seed", type=int, default=17); s.add_argument("--batch-size", type=int, default=16)
    s.add_argument("--objects-per-batch", type=int, default=4, help="At most 8 to bound shard-cache I/O")
    s.add_argument("--queries", type=int, default=64); s.add_argument("--eval-queries", type=int, default=256)
    s.add_argument("--eval-patches", type=int, default=512, help="Fixed validation subset and final evaluation cap")
    s.add_argument("--width", type=int, default=192); s.add_argument("--heads", type=int, default=4)
    s.add_argument("--depth", type=int, default=4); s.add_argument("--latents", type=int, default=4)
    s.add_argument("--lr", type=float, default=1e-4); s.add_argument("--rho", type=float, default=1.)
    s.add_argument("--position-tolerance", type=float, default=.01)
    s.add_argument("--coverage", type=float, default=.95)
    s.add_argument("--local-only", action="store_true", help="Permit training when the reference covers too little of the whole mesh")
    s.add_argument("--validate-every", type=int, default=250)
    s.add_argument("--resume", action="store_true"); s.add_argument("--stop-after", type=int)
    s.set_defaults(func=train)
    s = sub.add_parser("evaluate")
    s.add_argument("--data", required=True); s.add_argument("--checkpoint", required=True)
    s.add_argument("--device", default="auto"); s.add_argument("--queries", type=int, default=512)
    s.add_argument("--seed", type=int, default=10000019)
    s.add_argument("--max-patches", type=int, default=0, help="0 evaluates every patch in the selected split")
    s.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    s.add_argument("--out", default="qualification.json")
    s.add_argument("--position-tolerance", type=float, default=.01); s.add_argument("--coverage", type=float, default=.95)
    s.set_defaults(func=evaluate)
    s = sub.add_parser("check")
    s.add_argument("--data", required=True); s.add_argument("--out", default="check.json")
    s.add_argument("--seed", type=int, default=101); s.add_argument("--queries", type=int, default=128)
    s.add_argument("--max-patches", type=int, default=512); s.add_argument("--coverage", type=float, default=.95)
    s.set_defaults(func=check_charts)
    s = sub.add_parser("encode")
    s.add_argument("--data", required=True); s.add_argument("--checkpoint", required=True)
    s.add_argument("--device", default="auto"); s.add_argument("--out", required=True); s.set_defaults(func=encode)
    s = sub.add_parser("query")
    s.add_argument("--encoded", required=True); s.add_argument("--patch", type=int, required=True)
    s.add_argument("--u", type=float, required=True); s.add_argument("--v", type=float, required=True)
    s.add_argument("--offset", type=float, default=0); s.add_argument("--out"); s.set_defaults(func=query)
    return p


def main():
    args = parser().parse_args()
    torch.set_num_threads(args.threads)
    # Required for deterministic cuBLAS on CUDA; set before any CUDA operation.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    for name in ("steps", "batch_size", "objects_per_batch", "queries", "eval_queries", "eval_patches", "validate_every", "patches", "radius", "input_points", "fit_points", "max_faces", "rho", "lr", "position_tolerance", "coverage_points"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.command == "prepare" and args.offset <= 0:
        raise ValueError("prepare --offset must be positive; query --offset may be zero or negative")
    for name, value in vars(args).items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite")
    if hasattr(args, "objects_per_batch") and args.objects_per_batch > 8:
        raise ValueError("--objects-per-batch must be <=8")
    for name in ("coverage", "target_coverage"):
        if hasattr(args, name) and not 0 < getattr(args, name) <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must lie in (0,1]")
    return args.func(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
