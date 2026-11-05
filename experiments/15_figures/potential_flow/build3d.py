"""3D data for the AREALS hero figure, on one car mesh.

Stages (each cached in hero3d.npz next to this file):
  mesh     closed car-like body whose centre-plane section is the fastback
           profile of the GIFs; sides rounded, cabin narrower (tumblehome)
  geom     normals, vertex areas, principal and mean curvature, RMS radius
  flow     non-lifting potential flow, U = (1, 0, 0), by the method of
           fundamental solutions; Cp at every vertex (sphere-validated)
  eta      exact unsigned distance on the y = 0 plane; the y = 0 section
  anchors  256 farthest-point anchors with normals and area weights
  mask     60% of the area hidden as 32 patches
  omega    ~30 perturbed shapes: field change vs Hausdorff distance in
           position only and in position + normal line

usage: python3 build3d.py STAGE [STAGE ...]   (or: all)
"""
import os
import sys
import time

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from matplotlib.path import Path
import trimesh
import igl
from skimage.measure import marching_cubes

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(os.path.dirname(HERE), "gifs"))
import aero2d as A2

OUT = os.path.join(HERE, "hero3d.npz")
U_INF = np.array([1.0, 0.0, 0.0])


def load():
    return dict(np.load(OUT, allow_pickle=False)) if os.path.exists(OUT) else {}


def save(d):
    np.savez_compressed(OUT, **d)


def smoothstep(z):
    z = np.clip(z, 0, 1)
    return z * z * (3 - 2 * z)


# ---------------------------------------------------------------- mesh
def build_mesh(h=0.02, hs=0.004, blur=0.05, r_edge=0.26):
    O = A2.Outline()                                   # same car as the GIFs
    P = O.point(np.linspace(0, O.length, 4000, endpoint=False))
    x0, x1, z0, z1 = -0.3, 4.85, -0.1, 1.75
    gx = np.arange(x0, x1, hs)
    gz = np.arange(z0, z1, hs)
    GX, GZ = np.meshgrid(gx, gz, indexing="ij")
    inside = Path(P).contains_points(np.column_stack([GX.ravel(), GZ.ravel()])).reshape(GX.shape)
    sdf = (ndimage.distance_transform_edt(inside) - ndimage.distance_transform_edt(~inside)) * hs
    sdf = ndimage.gaussian_filter(sdf, blur / hs)
    S2 = RegularGridInterpolator((gx, gz), sdf, bounds_error=False, fill_value=-1.0)
    X = np.arange(-0.2, 4.78, h)
    Y = np.arange(-1.02, 1.02 + h / 2, h)
    Z = np.arange(0.05, 1.62, h)
    XX, ZZ = np.meshgrid(X, Z, indexing="ij")
    s_xz = S2(np.column_stack([XX.ravel(), ZZ.ravel()])).reshape(XX.shape)
    w0 = 0.62 - 0.30 * smoothstep((Z - 0.92) / 0.5)           # tumblehome
    ay = np.abs(Y)
    over = ay[None, :] - w0[:, None]                           # (nz, ny)
    phi = np.where(over <= 0, 0.0,
                   np.where(over < r_edge,
                            r_edge - np.sqrt(np.maximum(r_edge ** 2 - over ** 2, 0.0)),
                            r_edge + 1000 * (over - r_edge)))
    F = s_xz[:, None, :] - phi.T[None, :, :]                   # (nx, ny, nz)
    F[[0, -1], :, :] = -1
    F[:, [0, -1], :] = -1
    F[:, :, [0, -1]] = -1
    V, Fc, _, _ = marching_cubes(F, level=0.0, spacing=(h, h, h))
    V += np.array([X[0], Y[0], Z[0]])
    m = trimesh.Trimesh(V, Fc, process=True)
    if m.volume < 0:
        m.invert()
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=30)
    import fast_simplification
    Vs, Fs = fast_simplification.simplify(m.vertices, m.faces, target_reduction=0.55)
    m = trimesh.Trimesh(Vs, Fs, process=True)
    trimesh.smoothing.filter_taubin(m, lamb=0.5, nu=-0.53, iterations=10)
    if m.volume < 0:
        m.invert()
    print(f"mesh: {len(m.vertices)} vertices, {len(m.faces)} faces, watertight={m.is_watertight}, "
          f"volume={m.volume:.3f}, area={m.area:.3f}")
    return m, P


def stage_mesh(d):
    m, P = build_mesh()
    d["V"], d["F"] = np.asarray(m.vertices), np.asarray(m.faces)
    d["profile"] = P
    # centre-plane section vs the 2D profile
    sec = m.section(plane_origin=[0, 0, 0], plane_normal=[0, 1, 0])
    pts = np.vstack([np.asarray(e) for e in sec.discrete]) if sec is not None else np.zeros((0, 3))
    dev = cKDTree(P).query(pts[:, [0, 2]])[0].max() if len(pts) else np.nan
    print(f"section vs profile: max deviation {dev:.4f}")
    d["section_dev"] = np.array(dev)


# ---------------------------------------------------------------- geometry
def stage_geom(d):
    m = trimesh.Trimesh(d["V"], d["F"], process=False)
    V, F = d["V"], d["F"]
    N = np.asarray(m.vertex_normals)
    area_v = np.zeros(len(V))
    np.add.at(area_v, F.ravel(), np.repeat(m.area_faces / 3, 3))
    pd1, pd2, pv1, pv2, *_ = igl.principal_curvature(V, F, radius=7)
    H = 0.5 * (pv1 + pv2)
    Adj = igl.adjacency_matrix(F)
    deg = np.asarray(Adj.sum(1)).ravel()
    # vertex normals of sliver triangles can swing; average over neighbours
    for _ in range(3):
        N = N + Adj @ N
        N /= np.linalg.norm(N, axis=1)[:, None]
    for _ in range(6):
        H = 0.5 * H + 0.5 * (Adj @ H) / deg
    # convex = positive with outward normals: the roof top must be convex
    top = np.argsort(V[:, 2])[-50:]
    if np.median(H[top]) < 0:
        H, pv1, pv2 = -H, -pv1, -pv2
    c = (V * area_v[:, None]).sum(0) / area_v.sum()
    R = np.sqrt((((V - c) ** 2).sum(1) * area_v).sum() / area_v.sum())
    d.update(N=N, area_v=area_v, k1=pv1, k2=pv2, H=H, R=np.array(R), centroid=c)
    print(f"R = {R:.3f}, H range {np.percentile(H, 1):.2f} .. {np.percentile(H, 99):.2f}")


# ---------------------------------------------------------------- flow (MFS)
def fps(P, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(P)))]
    dist = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(dist))
        idx.append(i)
        dist = np.minimum(dist, np.linalg.norm(P - P[i], axis=1))
    return np.array(idx)


def kernel(X, S):
    """Velocity of unit point sources at S, evaluated at X: (nx, ns, 3)."""
    r = X[:, None, :] - S[None, :, :]
    r3 = np.linalg.norm(r, axis=2) ** 3
    return r / (4 * np.pi * r3[..., None])


def mfs_solve(Xc, Nc, Src, lam=1e-12):
    A = np.einsum("ijk,ik->ij", kernel(Xc, Src), Nc)
    b = -Nc @ U_INF
    AtA = A.T @ A
    AtA[np.diag_indices_from(AtA)] += lam * np.trace(AtA) / len(AtA)
    sig = np.linalg.solve(AtA, A.T @ b)
    res = A @ sig - b
    return sig, np.abs(res).max()


def velocity(X, Src, sig, chunk=4000):
    out = np.empty((len(X), 3))
    for k in range(0, len(X), chunk):
        out[k:k + chunk] = U_INF + np.einsum("ijk,j->ik", kernel(X[k:k + chunk], Src), sig)
    return out


def surface_cp(V, N, Src, sig):
    u = velocity(V, Src, sig)
    ut = u - (u * N).sum(1)[:, None] * N
    return 1.0 - (ut ** 2).sum(1), np.abs((u * N).sum(1))


def mfs_setup(V, N, m_col=5200, n_src=2300, delta=0.10, seed=1):
    col = fps(V, m_col, seed)
    src = col[fps(V[col], n_src, seed + 1)]
    return col, src, V[src] - delta * N[src]


def sphere_check():
    s = trimesh.creation.icosphere(subdivisions=5)
    V, N = np.asarray(s.vertices), np.asarray(s.vertex_normals)
    col, src, S = mfs_setup(V, N, m_col=3000, n_src=1200, delta=0.12)
    sig, res = mfs_solve(V[col], N[col], S)
    cp, un = surface_cp(V, N, S, sig)
    th = np.arccos(np.clip(V[:, 0], -1, 1))
    exact = 1 - 2.25 * np.sin(th) ** 2
    err = np.abs(cp - exact).max()
    print(f"sphere: max |Cp - exact| = {err:.4f}, max |u.n| = {un.max():.2e}")
    return err


def stage_flow(d):
    d["sphere_err"] = np.array(sphere_check())
    V, N = d["V"], d["N"]
    t0 = time.time()
    col, src, S = mfs_setup(V, N)
    w = igl.winding_number(V, d["F"], S)
    print(f"sources inside: {np.mean(w > 0.5) * 100:.1f}%")
    sig, res = mfs_solve(V[col], N[col], S)
    cp, un = surface_cp(V, N, S, sig)
    print(f"car: Cp {cp.min():.2f} .. {cp.max():.2f}, max |u.n| at vertices {np.percentile(un, 99):.3f} (99th pct), "
          f"{time.time() - t0:.1f}s")
    d.update(Cp=cp, mfs_col=col, mfs_src=src, mfs_sig=sig, un=un)


# ---------------------------------------------------------------- eta + section
def stage_eta(d):
    V, F = d["V"], d["F"]
    gx = np.arange(-0.8, 5.3 + 1e-9, 0.01)
    gz = np.arange(-0.45, 2.25 + 1e-9, 0.01)
    GX, GZ = np.meshgrid(gx, gz)
    Q = np.column_stack([GX.ravel(), np.zeros(GX.size), GZ.ravel()])
    sq, fi, C = igl.point_mesh_squared_distance(Q, V, F)
    d.update(eta_x=gx, eta_z=gz, eta_d=np.sqrt(sq).reshape(GX.shape))
    m = trimesh.Trimesh(V, F, process=False)
    sec = m.section(plane_origin=[0, 0, 0], plane_normal=[0, 1, 0])
    loops = [np.asarray(e) for e in sec.discrete]
    loop = max(loops, key=len)[:, [0, 2]]
    if np.allclose(loop[0], loop[-1]):
        loop = loop[:-1]
    # orient clockwise (outward normal = left normal), as in the 2D outline
    area = 0.5 * np.sum(loop[:, 0] * np.roll(loop[:, 1], -1) - np.roll(loop[:, 0], -1) * loop[:, 1])
    if area > 0:
        loop = loop[::-1]
    O = A2.Outline(P=loop, smooth=0.02)
    s = np.linspace(0, O.length, 3000, endpoint=False)
    Pp, Tt, Nn, Kk = O.frame(s)
    # Cp on the section from the mesh (closest vertex values, barycentric)
    Q3 = np.column_stack([Pp[:, 0], np.zeros(len(Pp)), Pp[:, 1]])
    _, fi, C = igl.point_mesh_squared_distance(Q3, V, F)
    bc = igl.barycentric_coordinates(C, V[F[fi, 0]], V[F[fi, 1]], V[F[fi, 2]])
    cp_sec = (d["Cp"][F[fi]] * bc).sum(1) if "Cp" in d else np.zeros(len(Pp))
    d.update(sec_s=s, sec_p=Pp, sec_t=Tt, sec_n=Nn, sec_k=Kk, sec_cp=cp_sec)
    print(f"section: {len(loop)} points, length {O.length:.3f}, kappa {Kk.min():.2f} .. {Kk.max():.2f}")


# ---------------------------------------------------------------- anchors + mask
def sample(d, n, seed=3):
    m = trimesh.Trimesh(d["V"], d["F"], process=False)
    P, fi = trimesh.sample.sample_surface_even(m, n, seed=seed) if False else trimesh.sample.sample_surface(m, n, seed=seed)
    bc = trimesh.triangles.points_to_barycentric(m.triangles[fi], P)
    Nn = (d["N"][d["F"][fi]] * bc[..., None]).sum(1)
    Nn /= np.linalg.norm(Nn, axis=1)[:, None]
    return P, Nn, m.area / n


def stage_anchors(d):
    P, Nn, a = sample(d, 80000)
    idx = fps(P, 256, seed=5)
    own = cKDTree(P[idx]).query(P)[1]
    w = np.bincount(own, minlength=256) * a
    d.update(samp_p=P, samp_n=Nn, samp_area=np.array(a), samp_own=own,
             anc_p=P[idx], anc_n=Nn[idx], anc_w=w)
    print(f"anchors: weights {w.min():.4f} .. {w.max():.4f}, sum {w.sum():.3f}")


def stage_mask(d, frac=0.60, n_patch=32, seed=11):
    P = d["samp_p"]
    rng = np.random.default_rng(seed)
    cen = P[rng.choice(len(P), n_patch, replace=False)]
    dist = cKDTree(cen).query(P)[0]
    lo, hi = 0.0, 3.0
    for _ in range(50):
        r = 0.5 * (lo + hi)
        lo, hi = (r, hi) if np.mean(dist < r) < frac else (lo, r)
    dv, pid = cKDTree(cen).query(d["V"])
    d.update(mask_centres=cen, mask_r=np.array(r), hidden=(dv < r), patch_id=pid)
    print(f"mask: radius {r:.3f}, hidden area {np.mean(dist < r) * 100:.1f}%")


# ---------------------------------------------------------------- omega(r)
def embed(P, Nn, R):
    nn = np.einsum("ki,kj->kij", Nn, Nn).reshape(len(Nn), 9)
    return np.hstack([P / R, nn])


def stage_omega(d, n_cases=30, seed=21):
    V, F, N, R = d["V"], d["F"], d["N"], float(d["R"])
    col, src = d["mfs_col"], d["mfs_src"]
    av = d["area_v"]
    E0 = embed(V, N, R)
    T0, T0e = cKDTree(V / R), cKDTree(E0)
    rng = np.random.default_rng(seed)
    lams = np.array([3.0, 1.5, 0.8, 0.5, 0.3, 0.2])
    rows = []
    t0 = time.time()
    for i in range(n_cases):
        lam = lams[i % len(lams)]
        ks = rng.standard_normal((3, 3))
        ks = ks / np.linalg.norm(ks, axis=1)[:, None] * 2 * np.pi / lam
        ph = rng.uniform(0, 2 * np.pi, 3)
        arg = V @ ks.T + ph
        h = np.cos(arg).sum(1)
        amp = np.exp(rng.uniform(np.log(0.003), np.log(0.04))) * R
        scale = amp / np.abs(h).max()
        h *= scale
        gh = -(np.sin(arg)[:, :, None] * ks[None, :, :]).sum(1) * scale   # grad h
        gs = gh - (gh * N).sum(1)[:, None] * N                           # surface gradient
        Vp = V + h[:, None] * N
        # first-order normal of the displaced surface, n - grad_S h: smooth,
        # unlike vertex normals of sliver triangles after decimation
        Np = N - gs
        Np /= np.linalg.norm(Np, axis=1)[:, None]
        Sp = Vp[src] - 0.10 * Np[src]
        sig, res = mfs_solve(Vp[col], Np[col], Sp)
        cpp, _ = surface_cp(Vp, Np, Sp, sig)
        _, fi, C = igl.point_mesh_squared_distance(V, Vp, F)
        bc = igl.barycentric_coordinates(C, Vp[F[fi, 0]], Vp[F[fi, 1]], Vp[F[fi, 2]])
        cp_back = (cpp[F[fi]] * bc).sum(1)
        dcp = np.sqrt((av * (cp_back - d["Cp"]) ** 2).sum() / av.sum())
        d_pos = max(T0.query(Vp / R)[0].max(), cKDTree(Vp / R).query(V / R)[0].max())
        Ep = embed(Vp, Np, R)
        d_full = max(T0e.query(Ep)[0].max(), cKDTree(Ep).query(E0)[0].max())
        rows.append((dcp, d_pos, d_full, lam, amp / R))
        print(f"  case {i:2d}: lambda {lam:.1f}, amp {amp / R:.4f} R, dCp {dcp:.4f}, d_pos {d_pos:.4f}, "
              f"d_full {d_full:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    d["omega"] = np.array(rows)


STAGES = {"mesh": stage_mesh, "geom": stage_geom, "flow": stage_flow, "eta": stage_eta,
          "anchors": stage_anchors, "mask": stage_mask, "omega": stage_omega}

if __name__ == "__main__":
    todo = list(STAGES) if sys.argv[1:] == ["all"] else sys.argv[1:]
    d = load()
    for st in todo:
        t = time.time()
        print(f"== {st}", flush=True)
        STAGES[st](d)
        save(d)
        print(f"   ({time.time() - t:.1f}s)", flush=True)
