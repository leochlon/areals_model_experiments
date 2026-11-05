"""Rigid symmetries of a visible cloud: reflections, rotations and translations between repeated parts.

Transformation-space voting (Mitra, Guibas and Pauly 2006): pairs of points with matching principal curvatures each
propose the rigid map taking one curvature frame to the other; dense clusters of proposals are candidates, kept where
they carry the cloud onto itself over whole regions (point-to-plane within half a spacing, normals within 12 degrees).
Coordinates are the packet's normalized frame; results are deterministic given the generator.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _line_angle(a, b):
    c = np.abs((_unit(a) * _unit(b)).sum(-1))
    return np.degrees(np.arccos(np.clip(c, 0, 1)))


def _spacing(points, sample=2000):
    sub = points[:: max(1, len(points) // sample)]
    return float(np.median(cKDTree(points).query(sub, k=2)[0][:, 1]))


def _apply(R, t, x):
    return x @ R.T + t


def curvature_frames(X, N, centers, k=24):
    """Principal curvatures k1 >= k2 and frames at X[centers] from a quadric fit to k neighbours, each oriented so
    that mean curvature is >= 0 (normals carry no sign)."""
    tree = cKDTree(X)
    _, nb = tree.query(X[centers], k=min(k, len(X)))
    p, n = X[centers], _unit(N[centers])
    helper = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    e1 = _unit(np.cross(n, helper))
    e2 = np.cross(n, e1)
    rel = X[nb] - p[:, None]
    x, y, z = (rel * e1[:, None]).sum(-1), (rel * e2[:, None]).sum(-1), (rel * n[:, None]).sum(-1)
    A = np.stack([x * x, x * y, y * y, x, y, np.ones_like(x)], -1)
    AtA = np.einsum("mki,mkj->mij", A, A) + 1e-9 * np.eye(6)
    coef = np.linalg.solve(AtA, np.einsum("mki,mk->mi", A, z)[..., None])[..., 0]
    shape = np.stack([np.stack([2 * coef[:, 0], coef[:, 1]], -1), np.stack([coef[:, 1], 2 * coef[:, 2]], -1)], -2)
    vals, vecs = np.linalg.eigh(shape)
    k1, k2 = vals[:, 1], vals[:, 0]
    d1 = vecs[:, 0, 1, None] * e1 + vecs[:, 1, 1, None] * e2
    d2 = vecs[:, 0, 0, None] * e1 + vecs[:, 1, 0, None] * e2
    flip = (k1 + k2) < 0
    n = np.where(flip[:, None], -n, n)
    k1, k2 = np.where(flip, -k2, k1), np.where(flip, -k1, k2)
    d1 = np.where(flip[:, None], d2, d1)
    return p, n, _unit(d1), k1, k2


def _encode(R, t):
    """A rigid map as the images of the origin and the three unit points (12 numbers, in shape units)."""
    return np.concatenate([t, (R + t[..., :, None]).swapaxes(-1, -2).reshape(*t.shape[:-1], 9)], -1)


def _decode(v, det_sign):
    t = v[:3]
    M = (v[3:].reshape(3, 3) - t).T
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, det_sign * np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt, t


def lands(R, t, W, NW, X, NX, treeX, h, plane_tol=0.5, reach=3.0, angle=12.0):
    """Does the map carry each point of W onto the surface sampled by X (point-to-plane within plane_tol spacings,
    nearest sample within reach spacings), with a matching normal?"""
    y = _apply(R, t, W)
    d, j = treeX.query(y)
    normal = np.abs(((y - X[j]) * NX[j]).sum(1))
    tangential = np.sqrt(np.maximum(d * d - normal * normal, 0))
    return (normal < plane_tol * h) & (tangential < reach * h) & (_line_angle(NW @ R.T, NX[j]) < angle)


def region_flags(flags, neighbours, share=0.75):
    """Hits whose neighbourhoods are hits too: a region, not scattered points."""
    return flags & (flags[neighbours].mean(1) >= share)


def vote_symmetries(X, N, rng, *, M=1500, k=24, pairs=15000, min_separation=0.15, bandwidth=0.25, min_votes=25,
                    min_support=0.08, max_maps=24, seeds=4000):
    """Candidate symmetries of the cloud X: [(R, t, support share)], inverses included, identity excluded."""
    if len(X) < 50:
        return []
    h = _spacing(X)
    treeX = cKDTree(X)
    centers = rng.choice(len(X), min(M, len(X)), replace=False)
    p, n, d1, k1, k2 = curvature_frames(X, N, centers, k)
    big = np.maximum(np.abs(k1), np.abs(k2))
    keep = (big > 0.5) & (np.abs(k1 - k2) > 0.2 * big)  # frames are defined: curved, not umbilic
    p, n, d1, k1, k2 = p[keep], n[keep], d1[keep], k1[keep], k2[keep]
    if len(p) < 10:
        return []
    signature = np.stack([np.arcsinh(k1), np.arcsinh(k2)], 1)
    pair = np.array(sorted(cKDTree(signature).query_pairs(0.15)), dtype=np.int64).reshape(-1, 2)
    if len(pair) == 0:
        return []
    pair = pair[np.linalg.norm(p[pair[:, 0]] - p[pair[:, 1]], axis=1) > min_separation]
    if len(pair) == 0:
        return []
    if len(pair) > pairs:
        pair = pair[rng.choice(len(pair), pairs, replace=False)]
    pair = np.concatenate([pair, pair[:, ::-1]])
    i, j = pair[:, 0], pair[:, 1]
    frames = np.stack([d1, np.cross(n, d1), n], -1)
    votes, dets = [], []
    for s in (1.0, -1.0):  # the sign of d1 is not defined
        R = (frames[j] * np.array([s, s, 1.0])) @ frames[i].swapaxes(1, 2)
        votes.append(_encode(R, p[j] - np.einsum("mab,mb->ma", R, p[i])))
        dets.append(np.ones(len(i)))
    a = _unit(p[i] - p[j])
    o = (a * (p[i] + p[j]) / 2).sum(1)
    Rf = np.eye(3) - 2 * a[:, :, None] * a[:, None, :]
    ok = (_line_angle(np.einsum("mab,mb->ma", Rf, n[i]), n[j]) < 15) & (
        _line_angle(np.einsum("mab,mb->ma", Rf, d1[i]), d1[j]) < 20)
    votes.append(_encode(Rf[ok], 2 * o[ok, None] * a[ok]))
    dets.append(-np.ones(int(ok.sum())))
    V, det = np.concatenate(votes), np.concatenate(dets)
    tree = cKDTree(V)
    seed = rng.choice(len(V), min(seeds, len(V)), replace=False)
    counts = tree.query_ball_point(V[seed], bandwidth, return_length=True)
    alive = np.ones(len(V), bool)
    found = []
    for s, count in zip(seed[np.argsort(-counts)], np.sort(counts)[::-1], strict=True):
        if len(found) >= 3 * max_maps or count < min_votes:
            break
        if not alive[s]:
            continue
        c = V[s]
        for _ in range(6):  # mean shift among live votes of the same handedness
            nb = np.array(tree.query_ball_point(c, bandwidth), dtype=np.int64)
            nb = nb[(det[nb] == det[s]) & alive[nb]]
            if len(nb) == 0:
                break
            c = V[nb].mean(0)
        members = np.array(tree.query_ball_point(c, bandwidth), dtype=np.int64)
        if (alive[members] & (det[members] == det[s])).sum() < min_votes:
            alive[s] = False
            continue
        alive[members] = False
        found.append(_decode(c, det[s]))
    neighbours = treeX.query(X, k=9)[1][:, 1:]
    maps = []
    identity = _encode(np.eye(3), np.zeros(3))
    for R, t in found:
        shift = np.linalg.norm((_encode(R, t) - identity).reshape(4, 3), axis=1).mean()
        if np.linalg.det(R) > 0 and shift < 0.1:
            continue
        share = float(region_flags(lands(R, t, X, N, X, N, treeX, h), neighbours).mean())
        if share >= min_support:
            maps.append((R, t, share))
    maps.sort(key=lambda m: -m[2])
    kept = []
    for R, t, share in maps:
        if all(np.linalg.norm(_encode(R, t) - _encode(R2, t2)) > bandwidth for R2, t2, _ in kept):
            kept.append((R, t, share))
        if len(kept) >= max_maps:
            break
    out = list(kept)
    for R, t, share in kept:
        if np.linalg.det(R) > 0:
            Ri, ti = R.T, -R.T @ t
            if all(np.linalg.norm(_encode(Ri, ti) - _encode(R2, t2)) > bandwidth for R2, t2, _ in out):
                out.append((Ri, ti, share))
    return sorted(out, key=lambda m: -m[2])


def symmetry_candidates(points, normals, surfels, surfel_normals, count, rng):
    """The packet's symmetry maps: the `count` best voted maps of the visible cloud (padded with invalid identities)
    and, per map and surfel, whether the map holds there as a region.

    Returns rotation [count, 3, 3], translation [count, 3], support [count], valid [count], local [count, S]."""
    points, normals = np.asarray(points, np.float64), _unit(np.asarray(normals, np.float64))
    surfels, surfel_normals = np.asarray(surfels, np.float64), _unit(np.asarray(surfel_normals, np.float64))
    maps = vote_symmetries(points, normals, rng)[:count]
    rotation = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    translation = np.zeros((count, 3))
    support = np.zeros(count)
    valid = np.zeros(count)
    local = np.zeros((count, len(surfels)))
    if maps:
        h = _spacing(points)
        tree = cKDTree(points)
        neighbours = cKDTree(surfels).query(surfels, k=min(9, len(surfels)))[1][:, 1:]
        for m, (R, t, share) in enumerate(maps):
            rotation[m], translation[m], support[m], valid[m] = R, t, share, 1.0
            local[m] = region_flags(lands(R, t, surfels, surfel_normals, points, normals, tree, h), neighbours)
    return (rotation.astype(np.float32), translation.astype(np.float32), support.astype(np.float32),
            valid.astype(np.float32), local.astype(np.float32))
