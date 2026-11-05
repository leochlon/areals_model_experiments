"""Surface clouds: canonical form, frame, masking, sampling and anchors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

try:  # compiled farthest-point loop; same indices as the numpy loop
    import numba as _numba
except ImportError:  # pragma: no cover
    _numba = None


@dataclass
class Cloud:
    points: np.ndarray
    normals: np.ndarray
    weights: np.ndarray

    def subset(self, indices) -> "Cloud":
        return Cloud(self.points[indices], self.normals[indices], self.weights[indices])


def random_rotation(mode: str, rng: np.random.Generator) -> np.ndarray:
    """Identity, uniform on SO(3), or one of the 24 axis-aligned rotations."""
    if mode == "none":
        return np.eye(3)
    if mode == "so3":
        w, x, y, z = (q := rng.normal(size=4)) / np.linalg.norm(q)
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    if mode == "octahedral":
        while True:
            rotation = np.eye(3)[rng.permutation(3)] * rng.choice([-1.0, 1.0], 3)[:, None]
            if np.linalg.det(rotation) > 0:
                return rotation
    raise ValueError(f"unknown rotation mode {mode!r}")


def rotate_cloud(cloud: Cloud, rotation: np.ndarray) -> Cloud:
    return Cloud(cloud.points @ rotation.T, cloud.normals @ rotation.T, cloud.weights)


def flip_normal_signs(cloud: Cloud, probability: float, rng: np.random.Generator) -> Cloud:
    if probability <= 0:
        return cloud
    flip = rng.random(len(cloud.points)) < probability
    return Cloud(cloud.points, np.where(flip[:, None], -cloud.normals, cloud.normals), cloud.weights)


def canonical_cloud(points, normals, weights, *, allow_single: bool = False) -> Cloud:
    """Sorted unique points; coincident samples merge, their area summed and normals area-averaged."""
    p = np.asarray(points, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 3 or n.shape != p.shape or len(w) != len(p):
        raise ValueError("expected points [N,3], normals [N,3] and weights [N]")
    if not all(np.isfinite(v).all() for v in (p, n, w)) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("geometry must be finite, with nonnegative weights and positive total area")
    keep = w > 0
    p, n, w = p[keep], n[keep], w[keep]
    lengths = np.linalg.norm(n, axis=1)
    if (lengths < 1e-8).any():
        raise ValueError("a surface sample has a zero normal")
    n = n / lengths[:, None]
    unique, inverse = np.unique(p, axis=0, return_inverse=True)
    total = np.bincount(inverse, weights=w, minlength=len(unique))
    summed = np.stack([np.bincount(inverse, weights=w * n[:, d], minlength=len(unique)) for d in range(3)], axis=1)
    magnitude = np.linalg.norm(summed, axis=1)
    if (magnitude < 1e-8 * total).any():
        raise ValueError("coincident samples have opposing normals")
    if len(unique) < 2 and not allow_single:
        raise ValueError("at least two distinct surface points are required")
    return Cloud(unique, summed / magnitude[:, None], total)


def frame(cloud: Cloud) -> tuple[np.ndarray, float, float]:
    """Area-weighted centroid, RMS radius about it, and total area."""
    area = float(cloud.weights.sum())
    center = (cloud.points * (cloud.weights / area)[:, None]).sum(axis=0)
    scale = float(np.sqrt(((cloud.points - center) ** 2).sum(axis=1) @ (cloud.weights / area)))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("degenerate shape: zero extent")
    return center, scale, area


def _farthest_numpy(points: np.ndarray, first: int, count: int) -> np.ndarray:
    indices = np.empty(count, dtype=np.int64)
    distance = np.full(len(points), np.inf)
    chosen = np.zeros(len(points), dtype=bool)
    for i in range(count):
        indices[i] = first
        chosen[first] = True
        distance = np.minimum(distance, ((points - points[first]) ** 2).sum(axis=1))
        distance[chosen] = -np.inf
        first = int(np.argmax(distance))
    return indices


if _numba is not None:

    @_numba.njit(cache=True)
    def _farthest_numba(points, first, count):  # pragma: no cover - compiled
        n = points.shape[0]
        indices = np.empty(count, dtype=np.int64)
        distance = np.full(n, np.inf)
        for i in range(count):
            indices[i] = first
            fx, fy, fz = points[first, 0], points[first, 1], points[first, 2]
            for j in range(n):
                if distance[j] != -np.inf:
                    dx, dy, dz = points[j, 0] - fx, points[j, 1] - fy, points[j, 2] - fz
                    value = dx * dx
                    value += dy * dy
                    value += dz * dz
                    if value < distance[j]:
                        distance[j] = value
            distance[first] = -np.inf
            best, best_value = 0, distance[0]
            for j in range(1, n):
                if distance[j] > best_value:
                    best, best_value = j, distance[j]
            first = best
        return indices

else:  # pragma: no cover
    _farthest_numba = None


def farthest_indices(points: np.ndarray, weights: np.ndarray, count: int) -> np.ndarray:
    """Deterministic farthest-point sampling, starting from the point farthest from the weighted centroid."""
    count = min(int(count), len(points))
    if count < 1:
        raise ValueError("cannot select an empty anchor set")
    center = (points * (weights / weights.sum())[:, None]).sum(axis=0)
    first = int(np.argmax(((points - center) ** 2).sum(axis=1)))
    if _farthest_numba is not None:
        return _farthest_numba(np.ascontiguousarray(points, dtype=np.float64), first, count)
    return _farthest_numpy(points, first, count)


def visible_mask(cloud: Cloud, fraction: float, patches: int, rng: np.random.Generator) -> np.ndarray:
    """Hide about `fraction` of the surface as whole patches (Voronoi cells of farthest-point centres)."""
    if fraction == 0:
        return np.ones(len(cloud.points), dtype=bool)
    centers = farthest_indices(cloud.points, cloud.weights, max(2, patches))
    assignment = cKDTree(cloud.points[centers]).query(cloud.points, workers=1)[1]
    count = min(len(centers) - 1, max(1, round(fraction * len(centers))))
    order = rng.permutation(len(centers))
    minimum = min(8, max(2, len(cloud.points) // 4))
    while count:
        visible = ~np.isin(assignment, order[:count])
        if visible.sum() >= minimum:
            return visible
        count -= 1
    return np.ones(len(cloud.points), dtype=bool)


def sample_measure(cloud: Cloud, maximum: int, rng: np.random.Generator) -> Cloud:
    """At most `maximum` points, drawn in proportion to area with replacement, so all carry equal weight."""
    if len(cloud.points) <= maximum:
        return cloud
    indices = rng.choice(len(cloud.points), maximum, replace=True, p=cloud.weights / cloud.weights.sum())
    weights = np.full(maximum, cloud.weights.sum() / maximum)
    return canonical_cloud(cloud.points[indices], cloud.normals[indices], weights, allow_single=True)


def prepare_cloud(cloud: Cloud, latent_count: int, maximum: int, rng: np.random.Generator,
                  surfel_count: int = 0) -> dict:
    """Frame, sample and anchor one visible cloud. Anchors are the first farthest-point picks; with surfel_count
    above latent_count the same sampling continues and the extra picks become surfels."""
    center, scale, area = frame(cloud)
    sampled = sample_measure(cloud, maximum, rng)
    points = (sampled.points - center) / scale
    surfels = None
    if surfel_count > latent_count:
        surfels = farthest_indices(points, sampled.weights, surfel_count)
        indices = surfels[: min(int(latent_count), len(surfels))]
    else:
        indices = farthest_indices(points, sampled.weights, latent_count)
    anchor_points = points[indices]
    owners = cKDTree(anchor_points).query(points, workers=1)[1]
    mass = np.bincount(owners, weights=sampled.weights, minlength=len(indices))
    arrays = dict(points=points, normals=sampled.normals, point_weights=sampled.weights / sampled.weights.sum(),
                  anchors=anchor_points, anchor_normals=sampled.normals[indices], anchor_weights=mass / mass.sum(),
                  center=center, scale=np.array([scale]), area=np.array([area]))
    if surfels is not None:
        arrays.update(surfels=points[surfels], surfel_normals=sampled.normals[surfels],
                      surfel_weights=np.full(len(surfels), 1.0 / len(surfels)))
    return {key: np.asarray(value, dtype=np.float32) for key, value in arrays.items()}


def sampled_surface_queries(cloud: Cloud, hidden: np.ndarray, count: int, near_fraction: float, near_sigma: float,
                            box_padding: float, rng: np.random.Generator):
    """Queries near the (hidden) surface and uniform in the padded box, with targets from the nearest sample."""
    _, scale, _ = frame(cloud)
    candidates = np.flatnonzero(hidden)
    if not len(candidates):
        candidates = np.arange(len(cloud.points))
    n_near = round(count * near_fraction)
    chosen = rng.choice(candidates, n_near, replace=True, p=cloud.weights[candidates] / cloud.weights[candidates].sum())
    near = cloud.points[chosen] + rng.normal(0, near_sigma * scale, (n_near, 1)) * cloud.normals[chosen]
    near += rng.normal(0, near_sigma * scale * 0.15, near.shape)
    low = cloud.points.min(axis=0) - box_padding * scale
    high = cloud.points.max(axis=0) + box_padding * scale
    queries = np.concatenate([near, rng.uniform(low, high, (count - n_near, 3))], axis=0)
    rng.shuffle(queries, axis=0)
    distances, nearest = cKDTree(cloud.points).query(queries, workers=1)
    return queries, distances, cloud.normals[nearest]
