"""Minimal orthographic mesh renderer for matplotlib (painter's algorithm).

Faces are projected with a fixed view, back faces culled, sorted far to
near and drawn as one PolyCollection with Lambert shading applied to the
face colours. Returns the 2D projection so annotations can be placed in
the same coordinates.
"""
import numpy as np
from matplotlib.collections import PolyCollection


def view_matrix(azim_deg, elev_deg):
    """Rotation taking world (x along car, y left, z up) to screen
    (u right, v up, w towards the viewer)."""
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    # camera direction (from target to camera)
    c = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, c)
    right /= np.linalg.norm(right)
    true_up = np.cross(c, right)
    return np.vstack([right, true_up, c])


class View:
    def __init__(self, azim, elev, light=(0.35, -0.45, 0.82)):
        self.M = view_matrix(azim, elev)
        L = np.asarray(light, float)
        self.light = L / np.linalg.norm(L)

    def project(self, P):
        Q = np.asarray(P) @ self.M.T
        return Q[:, :2], Q[:, 2]

    def faces(self, V, F, face_rgba, ambient=0.55, diffuse=0.5, cull=True,
              spec=0.0):
        """PolyCollection of shaded, depth-sorted faces."""
        Q = V @ self.M.T
        tri = Q[F]                                  # (nf, 3, 3)
        n_world = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        n_world /= np.linalg.norm(n_world, axis=1)[:, None] + 1e-15
        n_view = n_world @ self.M.T
        keep = n_view[:, 2] > 0 if cull else np.ones(len(F), bool)
        depth = tri[:, :, 2].mean(1)
        order = np.argsort(depth[keep])
        idx = np.where(keep)[0][order]
        lam = np.clip(n_world[idx] @ self.light, 0, 1)
        shade = ambient + diffuse * lam
        rgba = np.array(face_rgba)[idx].copy()
        rgba[:, :3] = np.clip(rgba[:, :3] * shade[:, None], 0, 1)
        if spec > 0:
            h = self.light + np.array([0, 0, 1.0]) @ self.M
            h /= np.linalg.norm(h)
            s = np.clip(n_world[idx] @ h, 0, 1) ** 40 * spec
            rgba[:, :3] = np.clip(rgba[:, :3] + s[:, None], 0, 1)
        pc = PolyCollection(tri[idx, :, :2], facecolors=rgba, edgecolors=rgba,
                            linewidths=0.15, antialiased=True)
        return pc, idx

    def visible(self, P, nrm):
        """Points whose normal faces the viewer."""
        return (np.asarray(nrm) @ self.M.T)[:, 2] > 0
