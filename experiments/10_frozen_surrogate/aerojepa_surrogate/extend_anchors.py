"""Extend the existing 256 FPS anchors without changing the input point order."""
import numpy as np
from numba import njit
from scipy.spatial import cKDTree


@njit(cache=True, nogil=True)
def extend_fps(points, distance, initial_count, count):
    indices = np.empty(count, np.int64)
    indices[:initial_count] = np.arange(initial_count)
    for i in range(initial_count, count):
        selected = np.argmax(distance)
        indices[i] = selected
        for j in range(len(points)):
            if distance[j] >= 0:
                dx = points[j, 0] - points[selected, 0]
                dy = points[j, 1] - points[selected, 1]
                dz = points[j, 2] - points[selected, 2]
                value = dx * dx + dy * dy + dz * dz
                if value < distance[j]:
                    distance[j] = value
        distance[selected] = -np.inf
    return indices


def add_anchor_features(prepared, count):
    points = prepared["context_pos"].astype(np.float64)
    if not 256 < count <= len(points):
        raise ValueError(f"Cannot extend 256 anchors to {count} on {len(points)} points")
    distance = cKDTree(points[:256]).query(points, workers=1)[0] ** 2
    distance[:256] = -np.inf
    indices = extend_fps(points, distance, 256, count)
    owners = cKDTree(points[indices]).query(points, workers=1)[1]
    weights = prepared["context_feat"][:, 3]
    mass = np.bincount(owners, weights=weights, minlength=count)
    if (mass <= 0).any():
        raise ValueError("Empty anchor cell")
    extra = np.zeros((len(points), 2), np.float32)
    extra[indices, 0] = np.arange(1, count + 1)
    extra[indices, 1] = mass / mass.sum()
    prepared["context_feat"] = np.concatenate([prepared["context_feat"], extra], axis=1)
