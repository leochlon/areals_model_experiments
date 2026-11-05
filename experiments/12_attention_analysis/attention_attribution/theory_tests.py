#!/usr/bin/env python3
"""Tests of the geometry-prior theory in the AREALS paper, run on arrays exported from trained checkpoints.

Each subcommand tests one claim of Section 2 / Appendix B, prints a report with pass criteria fixed in advance
(PASS / FAIL / SKIP when the design cannot reach significance) and can write the numbers to JSON with --out.
Data layout and the meaning of each criterion: README.md.

  prop1        Proposition 1: reconstruction error bounds what the tokens lose about the field. Estimates the
               Lipschitz constant L from pairs of simulated shapes, measures full-surface reconstruction error d and
               the error of a field probe on frozen tokens for each checkpoint, checks the bound 4 L^2 E d^2, and
               tests that field error tracks reconstruction error across checkpoints, across shapes and along a
               token-perturbation dose-response.
  prior        Level 1: conjugate Gaussian last layer on frozen features. Compares checkpoints by evidence and
               held-out predictive log-likelihood, and traces held-out error against the number of training shapes.
  datacurve    Level 1: the prior's advantage should shrink as simulations grow. Fits the pretrained-over-scratch
               gain against training-set size from full fine-tuning runs (CSV), paired by seed.
  condition    Level 2: explicit condition posterior from true-condition predictions over a condition grid;
               condition accuracy and calibration, amortisation gap of the one-pass model, and gap closure G.
  normals      Section 3.2: if the projector head outputs the posterior mean of n n^T, its top eigenvalue predicts
               its own error, E[sin^2] = 1 - lambda_max, and its trace is 1.
  equivalence  Section 2: at a single operating condition the sensor and shape-only models should be equivalent
               (two one-sided tests on per-case scores).
  attention    Section 2, local and long-range dependence: pretrained attention should be short-range and fine-tuning
               should add long-range mass. Distance profiles of query-to-token attention per checkpoint, against the
               area baseline (all logits zero), paired across checkpoints by shape.
  attribution  Section 2: do pretraining's gains sit where local geometry dominates (curved, attached regions) or
               where the field is set globally (extreme values over flat surface: vortex footprints, wakes)?
               Gains by curvature and field-extremity strata, or by your own region labels.
  selftest     Synthetic data with known answers for every analysis.

Requires numpy and scipy.
"""
import argparse
import csv
import glob
import itertools
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata, spearmanr
from scipy.stats import t as student_t

SQ2 = math.sqrt(2.0)
DEFAULT_ORDER = 'xx,yy,zz,xy,xz,yz'


# ======================================================================================================================
# Geometry helpers

def sym3(P, order=DEFAULT_ORDER):
    """[..., 6] in the given component order, or [..., 3, 3], to a symmetric [..., 3, 3]."""
    P = np.asarray(P, dtype=np.float64)
    if P.shape[-1] == 6:
        pos = {'xx': (0, 0), 'yy': (1, 1), 'zz': (2, 2), 'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
        out = np.zeros(P.shape[:-1] + (3, 3))
        for k, name in enumerate(order.split(',')):
            i, j = pos[name.strip()]
            out[..., i, j] = P[..., k]
            out[..., j, i] = P[..., k]
        return out
    if P.shape[-2:] == (3, 3):
        return 0.5 * (P + np.swapaxes(P, -1, -2))
    raise ValueError(f'projector array must end in 6 or (3, 3), got shape {P.shape}')


def proj_embed(P):
    """Symmetric [..., 3, 3] to R^6 with sqrt(2) on off-diagonals: Euclidean distance equals Frobenius distance."""
    return np.stack([P[..., 0, 0], P[..., 1, 1], P[..., 2, 2],
                     SQ2 * P[..., 0, 1], SQ2 * P[..., 0, 2], SQ2 * P[..., 1, 2]], axis=-1)


def unit(N):
    N = np.asarray(N, dtype=np.float64)
    return N / np.clip(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12, None)


def outer_normals(N):
    N = unit(N)
    return N[..., :, None] * N[..., None, :]


def top_eigvec(P):
    if len(P) == 0:
        return np.zeros((0, 3))
    _, V = np.linalg.eigh(P)
    return V[..., :, -1]


def rank1(P):
    """Projector onto the top eigenvector: the normal-line estimate of a (possibly uncertain) projector output."""
    return outer_normals(top_eigvec(P)) if len(P) else np.zeros((0, 3, 3))


def rms_radius(X, A):
    w = A / A.sum()
    c = (w[:, None] * X).sum(0)
    return float(np.sqrt((w * ((X - c) ** 2).sum(1)).sum()))


def embed(X, P, scale, lam):
    """Joint metric z = (x / scale, sqrt(lam) vec(P)); with lam = 1/2 the normal term of two unit projectors is
    |sin(angle)| between the normal lines."""
    return np.hstack([X / scale, math.sqrt(lam) * proj_embed(P)])


def surface_match(Xq, Pq, Xs, Ps, scale, lam, k=8, tau=0.35, tree=None, chunk=20000):
    """Match query points (normal-line projectors Pq) to a sampled surface (Xs, Ps) in the joint metric.

    Among each query's k nearest surface samples in the joint embedding, take the sample with the smallest joint
    distance when position is measured to the sample's tangent plane and the normal term uses the sample's own
    projector; then refine the normal term by interpolating projectors over that sample's same-face neighbours
    (projectors within tau, Frobenius) with tangential inverse-distance weights, and keep the smaller of the two.
    Matching on joint neighbours never pairs a point with the wrong face of a crease, and the interpolation removes
    the floor that point spacing puts on smooth surfaces. Returns (distance [Q], candidates [Q, k], weights [Q, k])."""
    kk = min(k, len(Xs))
    tree = tree if tree is not None else cKDTree(embed(Xs, Ps, scale, lam))
    Ns = top_eigvec(Ps)
    out_d = np.empty(len(Xq))
    out_c = np.empty((len(Xq), kk), dtype=np.int64)
    out_w = np.empty((len(Xq), kk))
    for s in range(0, len(Xq), chunk):
        xq, pq = Xq[s:s + chunk], Pq[s:s + chunk]
        _, cand = tree.query(embed(xq, pq, scale, lam), k=kk)
        cand = cand.reshape(len(xq), kk)
        Pc, Nc = Ps[cand], Ns[cand]
        off = xq[:, None, :] - Xs[cand]
        pos = np.abs((off * Nc).sum(-1))
        dj = np.sqrt((pos / scale) ** 2 + lam * np.linalg.norm(pq[:, None] - Pc, axis=(-2, -1)) ** 2)
        r = np.arange(len(xq))
        best = dj.argmin(1)
        same = np.linalg.norm(Pc - Pc[r, best][:, None], axis=(-2, -1)) < tau
        n0 = Nc[r, best]
        tang = off - (off * n0[:, None, :]).sum(-1, keepdims=True) * n0[:, None, :]
        w = np.where(same, 1.0 / np.clip(np.linalg.norm(tang, axis=-1), 1e-9, None), 0.0)
        w /= w.sum(1, keepdims=True)
        Pint = rank1((w[:, :, None, None] * Pc).sum(1))
        dref = np.sqrt((pos[r, best] / scale) ** 2 + lam * np.linalg.norm(pq - Pint, axis=(1, 2)) ** 2)
        out_d[s:s + chunk] = np.minimum(dj[r, best], dref)
        out_c[s:s + chunk] = cand
        out_w[s:s + chunk] = w
    return out_d, out_c, out_w


def surface_distance(X, P, A, Xr, Pr, scale, lam=0.5, empty_value=np.inf):
    """Distances between a simulated surface S = (points X, projectors P, area weights A) and a reconstruction
    (points Xr, projector outputs Pr). Forward (S to reconstruction): exact joint nearest neighbour, since the
    reconstruction has one point per evaluation point. Backward (reconstruction to S): surface_match. hausdorff
    estimates the Hausdorff distance of Proposition 1; chamfer_rms and normal_deg are robust diagnostics."""
    if len(Xr) == 0:
        return {'hausdorff': empty_value, 'forward_max': empty_value, 'backward_max': empty_value,
                'chamfer_rms': empty_value, 'normal_deg': 90.0, 'n_recon': 0}
    Pr1 = rank1(Pr)
    fa, j = cKDTree(embed(Xr, Pr1, scale, lam)).query(embed(X, P, scale, lam))
    gb, _, _ = surface_match(Xr, Pr1, X, P, scale, lam)
    w = A / A.sum()
    cosang = np.clip(np.abs((top_eigvec(Pr1[j]) * top_eigvec(P)).sum(1)), 0.0, 1.0)
    return {'hausdorff': float(max(fa.max(), gb.max())), 'forward_max': float(fa.max()),
            'backward_max': float(gb.max()),
            'chamfer_rms': float(np.sqrt(0.5 * (w * fa ** 2).sum() + 0.5 * (gb ** 2).mean())),
            'normal_deg': float((w * np.degrees(np.arccos(cosang))).sum()), 'n_recon': int(len(Xr))}


def empty_distance(X, scale, lam):
    """Distance assigned when the decoder rebuilds nothing: the joint-metric diameter of the shape."""
    ext = np.ptp(X, axis=0)
    return float(math.sqrt(float((ext ** 2).sum()) / scale ** 2 + 2.0 * lam))


def line_search(X, N, R, udf_fn, line_range=0.10, step=0.005, chunk=262144):
    """For each point, the point on its normal line within +/- line_range * R where udf_fn is smallest, refined
    between grid points assuming a unit-slope distance. udf_fn(Q [M, 3]) -> (udf [M] in world units, projector [M, 6]
    or [M, 3, 3]). Returns (Xhat [P, 3], udf_hat [P], Phat [P, ...]); udf_hat < eps * R means the point is rebuilt."""
    ts = np.arange(-line_range, line_range + 0.5 * step, step) * R
    T, h = len(ts), step * R
    Nn = unit(N)
    Xhat, uhat, Phat = np.empty_like(np.asarray(X, float)), np.empty(len(X)), None
    per = max(1, chunk // T)
    for s in range(0, len(X), per):
        x, n = X[s:s + per], Nn[s:s + per]
        u, PQ = udf_fn((x[:, None, :] + ts[None, :, None] * n[:, None, :]).reshape(-1, 3))
        u = np.asarray(u, float).reshape(len(x), T)
        PQ = np.asarray(PQ, float)
        PQ = PQ.reshape((len(x), T) + PQ.shape[1:])
        r = np.arange(len(x))
        j = u.argmin(1)
        jm, jp = np.clip(j - 1, 0, T - 1), np.clip(j + 1, 0, T - 1)
        inner = (j > 0) & (j < T - 1)
        delta = np.where(inner, np.clip(0.5 * (u[r, jm] - u[r, jp]), -0.5 * h, 0.5 * h), 0.0)
        Xhat[s:s + per] = x + (ts[j] + delta)[:, None] * n
        uhat[s:s + per] = np.maximum(u[r, j] - np.abs(delta), 0.0)
        Pj = PQ[r, j]
        if Phat is None:
            Phat = np.empty((len(X),) + Pj.shape[1:])
        Phat[s:s + per] = Pj
    return Xhat, uhat, Phat


def reconstruction_set(shape, rec, eps, sym_order=DEFAULT_ORDER):
    """Rebuilt surface: normal-line points with udf_hat < eps * R, plus box queries with udf < eps * R that lie farther
    than 3 eps R + point spacing from S (spurious surface). Box queries close to S are covered by the normal-line
    points and are left out, since near a crease their nearest-point normals legitimately differ from either face."""
    R = shape['R']
    on = rec['udf_hat'] < eps * R
    w = shape['A'] / shape['A'].sum()
    parts_x, parts_p = [rec['Xhat'][on]], [sym3(rec['Phat'][on], sym_order)]
    spurious = 0
    if 'Qb' in rec and len(rec['Qb']):
        cand = rec['udf_b'] < eps * R
        if cand.any():
            dq, _ = cKDTree(shape['X']).query(rec['Qb'][cand])
            far = dq > 3 * eps * R + shape['spacing']
            spurious = int(far.sum())
            parts_x.append(rec['Qb'][cand][far])
            parts_p.append(sym3(rec['PQb'][cand][far], sym_order))
    Xr = np.vstack(parts_x)
    Pr = np.concatenate(parts_p) if len(Xr) else np.zeros((0, 3, 3))
    return Xr, Pr, float(w[on].sum()), spurious


# ======================================================================================================================
# Statistics helpers

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float('nan')
    for v in (x[ok], y[ok]):
        if np.ptp(v) <= 1e-12 * max(float(np.abs(v).max()), 1e-300) + 1e-15:
            return float('nan')          # a constant (or numerically constant) series carries no rank information
    return float(spearmanr(x[ok], y[ok]).statistic)


def perm_pvalue_spearman(x, y, n_perm=20000, seed=0):
    """One-sided p-value for a positive Spearman correlation (Pearson correlation of ranks): exact enumeration of all
    permutations for up to 8 points, Monte Carlo with the (count + 1) / (n + 1) correction otherwise."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float('nan')
    rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean())
    ry = (ry - ry.mean()) / np.linalg.norm(ry - ry.mean())
    obs = float(rx @ ry)
    if len(x) <= 8:
        perms = np.array(list(itertools.permutations(range(len(x)))))
        return float(((ry[perms] @ rx) >= obs - 1e-12).mean())
    rng = np.random.default_rng(seed)
    perms = np.array([rng.permutation(len(x)) for _ in range(n_perm)])
    return float((((ry[perms] @ rx) >= obs - 1e-12).sum() + 1) / (n_perm + 1))


def boot_ci(stat_fn, units, n_boot=2000, seed=0, level=0.95):
    """Percentile bootstrap over resampled units (e.g. geometries). stat_fn(list_of_units) -> float. Warns when
    resamples are dropped because the statistic is undefined on them."""
    rng = np.random.default_rng(seed)
    units = list(units)
    vals = []
    for _ in range(n_boot):
        v = stat_fn([units[i] for i in rng.integers(0, len(units), len(units))])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 0.99 * n_boot:
        print(f'  warning: {n_boot - len(vals)} of {n_boot} bootstrap resamples gave an undefined statistic',
              file=sys.stderr)
    if not vals:
        return (float('nan'), float('nan'))
    a = (1 - level) / 2
    return (float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a)))


def ols(x, y):
    """Least-squares slope and intercept; NaN when x is constant (the slope is not identified)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.ptp(x) <= 1e-12 * max(float(np.abs(x).max()), 1e-300):
        return float('nan'), float('nan')
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), float(coef[1])


def area_weighted_nrmse(pred, target, area, mask=None):
    """Area-weighted RMSE (on mask, default all points) divided by the full-field area-weighted standard deviation."""
    w = area / area.sum()
    std = math.sqrt(float((w * (target - (w * target).sum()) ** 2).sum()))
    m = np.ones_like(target, bool) if mask is None else mask
    a = area[m]
    return math.sqrt(float((a * (pred[m] - target[m]) ** 2).sum() / a.sum())) / max(std, 1e-12)


def print_criteria(c):
    for k, v in c.items():
        print(f"  [{'SKIP' if v is None else ('PASS' if v else 'FAIL')}] {k}")


# ======================================================================================================================
# Conjugate last layer (Bayesian linear regression, MacKay evidence maximisation)

class ConjugateHead:
    """f = [phi, 1] w + noise, w ~ N(0, I / alpha), noise ~ N(0, 1 / beta). Fitted from sufficient statistics; null
    directions of the design are dropped (they cancel in the evidence) and the noise variance is floored at 1e-8 of
    the signal power, so features that fit the field exactly do not overflow."""

    def __init__(self, m):
        self.G = np.zeros((m + 1, m + 1))
        self.b = np.zeros(m + 1)
        self.ff = 0.0
        self.n = 0

    @staticmethod
    def _aug(phi):
        return np.hstack([phi, np.ones((len(phi), 1))])

    def add(self, phi, f):
        X = self._aug(phi)
        self.G += X.T @ X
        self.b += X.T @ f
        self.ff += float(f @ f)
        self.n += len(f)

    def fit(self, iters=1000, tol=1e-10):
        lam_all, U = np.linalg.eigh(self.G)
        lam_all = np.clip(lam_all, 0.0, None)
        keep = lam_all > 1e-12 * max(lam_all.max(), 1e-300)
        lam, Uk = lam_all[keep], U[:, keep]
        Ub = Uk.T @ self.b
        n = self.n
        power = max(self.ff / max(n, 1), 1e-300)
        beta_max = 1e8 / power
        alpha, beta = 1.0, 1.0 / power

        def resid2(mu):
            return max(self.ff - 2.0 * float(mu @ Ub) + float((lam * mu ** 2).sum()), 1e-300)
        for _ in range(iters):
            d = alpha + beta * lam
            mu = beta * Ub / d
            gamma = float((beta * lam / d).sum())
            a_new = float(np.clip(gamma / max(float(mu @ mu), 1e-300), 1e-12, 1e12))
            b_new = float(min(max(n - gamma, 1e-12) / resid2(mu), beta_max))
            done = abs(a_new - alpha) <= tol * alpha and abs(b_new - beta) <= tol * beta
            alpha, beta = a_new, b_new
            if done:
                break
        d = alpha + beta * lam
        mu = beta * Ub / d
        self.alpha, self.beta, self.w = alpha, beta, Uk @ mu
        self.U, self.d = U, alpha + beta * lam_all
        self.log_evidence = float(0.5 * len(lam) * math.log(alpha) + 0.5 * n * math.log(beta)
                                  - 0.5 * beta * resid2(mu) - 0.5 * alpha * float(mu @ mu)
                                  - 0.5 * float(np.log(d).sum()) - 0.5 * n * math.log(2 * math.pi))
        return self

    def predict(self, phi):
        X = self._aug(phi)
        proj = X @ self.U
        return X @ self.w, 1.0 / self.beta + (proj ** 2 / self.d).sum(1)


# ======================================================================================================================
# Data loading (layout in README.md)

def npz_files(path):
    files = sorted(glob.glob(os.path.join(path, '*.npz')))
    if not files:
        sys.exit(f'no .npz files in {path}')
    return files


def cond_keys(cond):
    return [tuple(np.round(r, 6)) for r in np.asarray(cond, float)]


_WARNED = set()


def read_cond(z, C, name):
    """Conditions as [C, k] rows: [C] -> [C, 1]; one condition given as [k] -> [1, k]. Missing -> 0..C-1, which
    assumes every shape lists its conditions in the same order."""
    if 'cond' in z.files:
        c = np.asarray(z['cond'], dtype=np.float64)
        if c.size % C:
            sys.exit(f'{name}: cond has {c.size} values for {C} field rows')
        c = c.reshape(C, -1)
    else:
        if 'cond' not in _WARNED:
            print('  warning: shape files without cond: conditions are matched by row index across shapes',
                  file=sys.stderr)
            _WARNED.add('cond')
        c = np.arange(C, dtype=np.float64).reshape(C, 1)
    if len(set(cond_keys(c))) != C:
        sys.exit(f'{name}: repeated condition values in one shape file')
    return c


def load_shapes(data, points=0, seed=0):
    rng = np.random.default_rng(seed)
    shapes = {}
    for f in npz_files(os.path.join(data, 'shapes')):
        name = os.path.splitext(os.path.basename(f))[0]
        z = np.load(f, allow_pickle=False)
        X, N, A = (z[k].astype(np.float64) for k in ('X', 'N', 'A'))
        F = np.atleast_2d(z['F']).astype(np.float64)
        R = float(z['R']) if 'R' in z.files else rms_radius(X, A)
        idx = np.arange(len(X))
        if points and len(X) > points:
            idx = np.sort(rng.choice(len(X), size=points, replace=False))
        Xs = X[idx]
        dnn, _ = cKDTree(Xs).query(Xs, k=2)
        shapes[name] = dict(X=Xs, N=N[idx], P=outer_normals(N[idx]), A=A[idx], F=F[:, idx],
                            cond=read_cond(z, F.shape[0], name), R=R, spacing=float(np.median(dnn[:, 1])),
                            split=str(z['split']) if 'split' in z.files else 'train', idx=idx)
    return shapes


def load_features(data, name, shapes):
    folder = os.path.join(data, 'ckpt', name)
    if not os.path.isdir(folder):
        sys.exit(f'missing checkpoint folder {folder}')
    out = {}
    for sid, s in shapes.items():
        f = os.path.join(folder, sid + '.npz')
        if os.path.exists(f):
            z = np.load(f, allow_pickle=False)
            if 'phi' in z.files:
                out[sid] = z['phi'][s['idx']].astype(np.float64)
    return out


def load_recon(data, name, sid, shape):
    z = np.load(os.path.join(data, 'ckpt', name, sid + '.npz'), allow_pickle=False)
    if 'Xhat' not in z.files:
        sys.exit(f'{name}/{sid}.npz has no Xhat/udf_hat/Phat: re-export with the current export_template.py')
    idx = shape['idx']
    rec = {'Xhat': z['Xhat'][idx].astype(np.float64), 'udf_hat': z['udf_hat'][idx].astype(np.float64),
           'Phat': z['Phat'][idx]}
    if 'Qb' in z.files:
        rec.update(Qb=z['Qb'].astype(np.float64), udf_b=z['udf_b'].astype(np.float64), PQb=z['PQb'])
    return rec


def field_stats(shapes, ids):
    vals = np.concatenate([shapes[s]['F'].ravel() for s in ids])
    return float(vals.mean()), float(vals.std() + 1e-12)


def split_ids(shapes, have=None):
    have = set(shapes) if have is None else set(have) & set(shapes)
    train = sorted(s for s in have if shapes[s]['split'] == 'train')
    test = sorted(s for s in have if shapes[s]['split'] != 'train')
    if not train or not test:
        sys.exit('need shapes with split "train" and shapes with split "test"')
    return train, test


# ======================================================================================================================
# Field probe on frozen features: the best predictor from tokens and condition, approximated by conjugate linear heads

class FieldProbe:
    """One conjugate head per condition when there are at most max_discrete training conditions and every test
    condition was seen in training; otherwise one head on [phi, 1] x [1, c, c^2] (quadratic terms only for condition
    dimensions with at least three training values), which includes condition main effects."""

    def __init__(self, shapes, feats, train_ids, test_ids, fmean, fstd, max_discrete=20):
        self.fmean, self.fstd = fmean, fstd
        train_keys = set(k for s in train_ids for k in cond_keys(shapes[s]['cond']))
        test_keys = set(k for s in test_ids for k in cond_keys(shapes[s]['cond']))
        self.discrete = len(train_keys) <= max_discrete and test_keys <= train_keys
        allc = np.vstack([shapes[s]['cond'] for s in train_ids])
        self.cm, self.cs = allc.mean(0), allc.std(0) + 1e-12
        self.quad = [len(np.unique(np.round(allc[:, d], 6))) >= 3 for d in range(allc.shape[1])]
        m = feats[train_ids[0]].shape[1]
        if self.discrete:
            self.heads = {}
            for s in train_ids:
                for j, k in enumerate(cond_keys(shapes[s]['cond'])):
                    self.heads.setdefault(k, ConjugateHead(m)).add(feats[s], self._z(shapes[s]['F'][j]))
            for h in self.heads.values():
                h.fit()
        else:
            self.head = ConjugateHead((m + 1) * (1 + allc.shape[1] + sum(self.quad)))
            for s in train_ids:
                for j, c in enumerate(shapes[s]['cond']):
                    self.head.add(self._basis(feats[s], c), self._z(shapes[s]['F'][j]))
            self.head.fit()

    def _z(self, f):
        return (f - self.fmean) / self.fstd

    def _basis(self, phi, c):
        ct = (np.asarray(c, float) - self.cm) / self.cs
        base = np.hstack([phi, np.ones((len(phi), 1))])
        return np.hstack([base] + [base * v for v in ct] + [base * v * v for v, q in zip(ct, self.quad) if q])

    def predict(self, phi, c):
        if self.discrete:
            return self.heads[tuple(np.round(np.asarray(c, float), 6))].predict(phi)
        return self.head.predict(self._basis(phi, c))

    def log_evidence_per_point(self):
        heads = list(self.heads.values()) if self.discrete else [self.head]
        return sum(h.log_evidence for h in heads) / sum(h.n for h in heads)

    def score(self, shape, phi):
        """Per shape: standardised area-weighted MSE, nRMSE and mean predictive log-likelihood, averaged over the
        shape's conditions."""
        mse, nr, ll = [], [], []
        w = shape['A'] / shape['A'].sum()
        for j, c in enumerate(shape['cond']):
            mean, var = self.predict(phi, c)
            fz = self._z(shape['F'][j])
            mse.append(float((w * (mean - fz) ** 2).sum()))
            nr.append(area_weighted_nrmse(mean * self.fstd + self.fmean, shape['F'][j], shape['A']))
            ll.append(float((-0.5 * (np.log(2 * np.pi * var) + (fz - mean) ** 2 / var)).mean()))
        return float(np.mean(mse)), float(np.mean(nr)), float(np.mean(ll))


# ======================================================================================================================
# Proposition 1

def estimate_lipschitz(shapes, scale, lam, fmean, fstd, k=3, coarse=256, seed=0):
    """Local Lipschitz estimates ||F(S,c) - F(S',c) o pi|| / d_H(S,S') over nearest-neighbour pairs of simulated
    shapes at shared conditions, in standardised field units; fields are carried across by same-face interpolation."""
    rng = np.random.default_rng(seed)
    ids = sorted(shapes)
    emb, trees = {}, {}
    for s in ids:
        sh = shapes[s]
        sub = rng.choice(len(sh['X']), size=min(coarse, len(sh['X'])), replace=False)
        emb[s] = embed(sh['X'][sub], sh['P'][sub], scale, lam)
        trees[s] = cKDTree(emb[s])
    n = len(ids)
    D = np.full((n, n), np.inf)
    for a in range(n):
        for b in range(a + 1, n):
            D[a, b] = D[b, a] = 0.5 * (trees[ids[b]].query(emb[ids[a]])[0].mean()
                                       + trees[ids[a]].query(emb[ids[b]])[0].mean())
    pairs = sorted(set((min(a, int(b)), max(a, int(b))) for a in range(n) for b in np.argsort(D[a])[:k]))
    ftree = {s: cKDTree(embed(shapes[s]['X'], shapes[s]['P'], scale, lam)) for s in ids}
    ratios, used = [], 0
    for a, b in pairs:
        sa, sb = ids[a], ids[b]
        A, B = shapes[sa], shapes[sb]
        ka, kb = cond_keys(A['cond']), cond_keys(B['cond'])
        common = [(i, kb.index(key)) for i, key in enumerate(ka) if key in kb]
        if not common:
            continue
        ia, ib = (list(t) for t in zip(*common))
        da, ca, wa = surface_match(A['X'], A['P'], B['X'], B['P'], scale, lam, tree=ftree[sb])
        db, cb, wb = surface_match(B['X'], B['P'], A['X'], A['P'], scale, lam, tree=ftree[sa])
        h = float(max(da.max(), db.max()))
        if h <= 0:
            continue
        Fa, Fb = (A['F'][ia] - fmean) / fstd, (B['F'][ib] - fmean) / fstd
        wA, wB = A['A'] / A['A'].sum(), B['A'] / B['A'].sum()
        diff_ab = np.sqrt((wA[None] * (Fa - (Fb[:, ca] * wa[None]).sum(2)) ** 2).sum(1))
        diff_ba = np.sqrt((wB[None] * (Fb - (Fa[:, cb] * wb[None]).sum(2)) ** 2).sum(1))
        ratios.extend((np.concatenate([diff_ab, diff_ba]) / h).tolist())
        used += 1
    if not ratios:
        sys.exit('no shape pairs share a condition: cannot estimate L')
    r = np.array(ratios)
    return {'pairs': used, 'ratios': int(len(r)), 'median': float(np.median(r)), 'q90': float(np.quantile(r, 0.9)),
            'q99': float(np.quantile(r, 0.99)), 'max': float(r.max())}


def prop1(data, ckpts, order=None, points=4096, lam=0.5, eps=0.01, k=3, max_shapes=400, n_boot=2000, seed=0,
          sym_order=DEFAULT_ORDER, within=None, verbose=True):
    ckpts = list(ckpts)
    for name in list(order or []) + list(within or []):
        if name not in ckpts:
            sys.exit(f'{name} is in --order/--within but not in --ckpts')
    shapes = load_shapes(data, points=points, seed=seed)
    scale = float(np.median([s['R'] for s in shapes.values()]))
    train_all, _ = split_ids(shapes)
    fmean, fstd = field_stats(shapes, train_all)
    sub = shapes
    if max_shapes and len(shapes) > max_shapes:
        keep = np.random.default_rng(seed).choice(sorted(shapes), size=max_shapes, replace=False)
        sub = {s: shapes[s] for s in keep}
    L = estimate_lipschitz(sub, scale, lam, fmean, fstd, k=k, seed=seed)
    res = {'scale': scale, 'lambda': lam, 'eps': eps, 'L': L, 'checkpoints': {}}
    per_shape = {}
    for name in ckpts:
        feats = load_features(data, name, shapes)
        train, test = split_ids(shapes, feats)
        probe = FieldProbe(shapes, feats, train, test, fmean, fstd)
        rows = {}
        for s in test:
            sh = shapes[s]
            Xr, Pr, cov, spur = reconstruction_set(sh, load_recon(data, name, s, sh), eps, sym_order)
            dist = surface_distance(sh['X'], sh['P'], sh['A'], Xr, Pr, scale, lam,
                                    empty_value=empty_distance(sh['X'], scale, lam))
            mse, nr, _ = probe.score(sh, feats[s])
            rows[s] = dict(dist, mse=mse, nrmse=nr, coverage=cov, spurious=spur)
        per_shape[name] = rows
        ids = sorted(rows)
        dH = np.array([rows[s]['hausdorff'] for s in ids])
        dC = np.array([rows[s]['chamfer_rms'] for s in ids])
        mse = np.array([rows[s]['mse'] for s in ids])
        bound99 = 4 * L['q99'] ** 2 * float(np.mean(dH ** 2))
        res['checkpoints'][name] = {
            'test_shapes': len(ids), 'coverage': float(np.mean([rows[s]['coverage'] for s in ids])),
            'spurious_points': int(sum(rows[s]['spurious'] for s in ids)),
            'hausdorff_mean': float(dH.mean()), 'chamfer_rms_mean': float(dC.mean()),
            'normal_deg_mean': float(np.mean([rows[s]['normal_deg'] for s in ids])),
            'probe_mse': float(mse.mean()), 'probe_nrmse': float(np.mean([rows[s]['nrmse'] for s in ids])),
            'bound_q99': bound99, 'bound_max': 4 * L['max'] ** 2 * float(np.mean(dH ** 2)),
            'ratio_to_bound_q99': float(mse.mean() / bound99) if bound99 > 0 else float('inf'),
            'within_spearman_chamfer': spearman(dC, mse), 'within_spearman_hausdorff': spearman(dH, mse)}
    common = sorted(set.intersection(*(set(per_shape[n]) for n in ckpts)))
    dm = [res['checkpoints'][n]['chamfer_rms_mean'] for n in ckpts]
    em = [res['checkpoints'][n]['probe_mse'] for n in ckpts]

    def across(units):
        return spearman([np.mean([per_shape[n][s]['chamfer_rms'] for s in units]) for n in ckpts],
                        [np.mean([per_shape[n][s]['mse'] for s in units]) for n in ckpts])
    if len(ckpts) >= 3:
        res['across'] = {'spearman': spearman(dm, em), 'p_one_sided': perm_pvalue_spearman(dm, em),
                         'stability_ci95_over_shapes': boot_ci(across, common, n_boot, seed)}
    within_names = list(within) if within else [n for n in ckpts if '_noise' not in n]

    def within_stat(units):
        vals = [spearman([per_shape[n][s]['chamfer_rms'] for s in units], [per_shape[n][s]['mse'] for s in units])
                for n in within_names]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float('nan')
    res['within'] = {'checkpoints': within_names, 'mean_spearman': within_stat(common),
                     'ci95': boot_ci(within_stat, common, n_boot, seed)}
    if order:
        idx = list(range(len(order)))
        od = [res['checkpoints'][n]['chamfer_rms_mean'] for n in order]
        oe = [res['checkpoints'][n]['probe_mse'] for n in order]
        res['dose_response'] = {'order': list(order), 'spearman_level_vs_distance': spearman(idx, od),
                                'spearman_level_vs_probe_mse': spearman(idx, oe),
                                'p_distance': perm_pvalue_spearman(idx, od), 'p_probe': perm_pvalue_spearman(idx, oe)}
    c = {'P1 bound holds with L at the 99th percentile (a FAIL is inconclusive)': all(
        v['probe_mse'] <= v['bound_q99'] for v in res['checkpoints'].values())}
    if len(ckpts) >= 5:
        a = res['across']
        c['P2 field error rises with reconstruction error across checkpoints (p <= 0.05)'] = bool(
            a['spearman'] > 0 and a['p_one_sided'] <= 0.05)
    else:
        c['P2 across checkpoints: needs at least 5 checkpoints'] = None
    w = res['within']
    c[f"P3 field error rises with reconstruction error across shapes ({', '.join(within_names)})"] = bool(
        w['mean_spearman'] > 0 and w['ci95'][0] > 0) if np.isfinite(w['mean_spearman']) else None
    if order and len(order) >= 4:
        d = res['dose_response']
        c['P4 perturbing tokens raises both errors (both p <= 0.05)'] = bool(
            d['p_distance'] <= 0.05 and d['p_probe'] <= 0.05)
    elif order:
        c['P4 dose-response: needs at least 4 levels'] = None
    res['criteria'] = c
    if verbose:
        print_prop1(res)
    return res


def print_prop1(res):
    L = res['L']
    print(f"scale (median RMS radius) {res['scale']:.4g}; lambda {res['lambda']}; rebuilt where udf < "
          f"{res['eps']} x R")
    print(f"Lipschitz estimates over {L['pairs']} shape pairs ({L['ratios']} ratios): median {L['median']:.3g}, "
          f"q90 {L['q90']:.3g}, q99 {L['q99']:.3g}, max {L['max']:.3g}")
    print(f"{'checkpoint':<24} {'cover':>6} {'d_H':>8} {'d_rms':>8} {'normal':>7} {'probe MSE':>10} {'nRMSE':>7} "
          f"{'bound q99':>10} {'ratio':>7} {'rho_shape':>9}")
    for n, v in res['checkpoints'].items():
        print(f"{n:<24} {v['coverage']:6.3f} {v['hausdorff_mean']:8.4f} {v['chamfer_rms_mean']:8.4f} "
              f"{v['normal_deg_mean']:7.2f} {v['probe_mse']:10.4f} {v['probe_nrmse']:7.4f} {v['bound_q99']:10.4f} "
              f"{v['ratio_to_bound_q99']:7.3f} {v['within_spearman_chamfer']:9.3f}")
    if 'across' in res:
        a = res['across']
        print(f"across checkpoints: rho {a['spearman']:.3f}, exact one-sided p {a['p_one_sided']:.3g} (stability "
              f"over shapes [{a['stability_ci95_over_shapes'][0]:.3f}, {a['stability_ci95_over_shapes'][1]:.3f}])")
    w = res['within']
    print(f"across shapes, mean rho over {', '.join(w['checkpoints'])}: {w['mean_spearman']:.3f}, 95% CI "
          f"[{w['ci95'][0]:.3f}, {w['ci95'][1]:.3f}]")
    if 'dose_response' in res:
        d = res['dose_response']
        print(f"dose-response: rho(level, d) {d['spearman_level_vs_distance']:.3f} (p {d['p_distance']:.3g}), "
              f"rho(level, probe MSE) {d['spearman_level_vs_probe_mse']:.3f} (p {d['p_probe']:.3g})")
    print_criteria(res['criteria'])


# ======================================================================================================================
# Level 1: evidence and data efficiency on frozen features

def prior(data, ckpts, reference=None, points=2048, ns=(), repeats=5, n_boot=2000, seed=0, verbose=True):
    shapes = load_shapes(data, points=points, seed=seed)
    feats = {n: load_features(data, n, shapes) for n in ckpts}
    common = set.intersection(*(set(f) for f in feats.values()))
    train, test = split_ids(shapes, common)
    fmean, fstd = field_stats(shapes, train)
    reference = reference or ckpts[-1]
    if reference not in ckpts:
        sys.exit('--reference must be one of --ckpts')
    res = {'reference': reference, 'train_shapes': len(train), 'test_shapes': len(test), 'checkpoints': {}}
    scores = {}
    for n in ckpts:
        probe = FieldProbe(shapes, feats[n], train, test, fmean, fstd)
        scores[n] = {s: probe.score(shapes[s], feats[n][s]) for s in test}
        res['checkpoints'][n] = {'log_evidence_per_point': probe.log_evidence_per_point(),
                                 'heldout_loglik': float(np.mean([v[2] for v in scores[n].values()])),
                                 'heldout_nrmse': float(np.mean([v[1] for v in scores[n].values()]))}
    for n in ckpts:
        if n == reference:
            continue
        dll = {s: scores[n][s][2] - scores[reference][s][2] for s in test}
        dnr = {s: scores[n][s][1] - scores[reference][s][1] for s in test}
        res['checkpoints'][n]['vs_reference'] = {
            'loglik_gain': float(np.mean(list(dll.values()))),
            'loglik_gain_ci95': boot_ci(lambda u: float(np.mean([dll[s] for s in u])), test, n_boot, seed),
            'nrmse_change': float(np.mean(list(dnr.values()))),
            'nrmse_change_ci95': boot_ci(lambda u: float(np.mean([dnr[s] for s in u])), test, n_boot, seed)}
    if ns:
        rng = np.random.default_rng(seed)
        curve = {n: defaultdict(list) for n in ckpts}
        for size in ns:
            if size > len(train):
                continue
            for _ in range(repeats):
                pick = sorted(rng.choice(train, size=size, replace=False))
                for n in ckpts:
                    probe = FieldProbe(shapes, feats[n], pick, test, fmean, fstd)
                    curve[n][size].append(float(np.mean([probe.score(shapes[s], feats[n][s])[1] for s in test])))
        res['curve'] = {n: {str(k): v for k, v in c.items()} for n, c in curve.items()}
        res['curve_trend'] = {}
        for n in ckpts:
            if n == reference:
                continue
            sizes = sorted(curve[n])
            gains = {sz: (np.array(curve[reference][sz]) - np.array(curve[n][sz])) / np.array(curve[reference][sz])
                     for sz in sizes}
            slope, _ = ols(np.log10(sizes), [gains[sz].mean() for sz in sizes])
            ci = boot_ci(lambda u: ols(np.log10(sizes), [gains[sz][u].mean() for sz in sizes])[0],
                         list(range(repeats)), n_boot, seed)
            res['curve_trend'][n] = {'relative_gain': {str(sz): float(gains[sz].mean()) for sz in sizes},
                                     'slope_per_decade': slope, 'slope_ci95': ci}
    res['criteria'] = {f'B1 {n} gives higher held-out log-likelihood than {reference}':
                       bool(v['vs_reference']['loglik_gain_ci95'][0] > 0)
                       for n, v in res['checkpoints'].items() if 'vs_reference' in v}
    if verbose:
        print(f"train shapes {len(train)}, test shapes {len(test)}, reference {reference}")
        print(f"{'checkpoint':<24} {'log ev/pt':>10} {'held-out ll':>12} {'nRMSE':>8} {'ll gain [95% CI]':>28}")
        for n, v in res['checkpoints'].items():
            g = v.get('vs_reference')
            gs = f"{g['loglik_gain']:.4f} [{g['loglik_gain_ci95'][0]:.4f}, {g['loglik_gain_ci95'][1]:.4f}]" if g \
                else '(reference)'
            print(f"{n:<24} {v['log_evidence_per_point']:10.4f} {v['heldout_loglik']:12.4f} "
                  f"{v['heldout_nrmse']:8.4f} {gs:>28}")
        for n, t in res.get('curve_trend', {}).items():
            gains = ', '.join(f'{k}: {v:+.3f}' for k, v in t['relative_gain'].items())
            print(f"frozen-feature curve, {n} vs {reference}: relative gain by training shapes {gains}; slope per "
                  f"decade {t['slope_per_decade']:+.3f} [{t['slope_ci95'][0]:+.3f}, {t['slope_ci95'][1]:+.3f}]")
        print_criteria(res['criteria'])
    return res


def datacurve(csv_path, pre='pretrained', scratch='scratch', n_boot=2000, seed=0, verbose=True):
    """Relative gain of pretrained over scratch against log10(training size). With seeds paired across the two
    initialisations at every size, the slope is computed per seed and given a t-interval; otherwise a bootstrap that
    resamples seeds within each cell is used."""
    rows = list(csv.DictReader(open(csv_path)))
    groups = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        groups[r.get('benchmark', 'all')][(int(float(r['n'])), r['init'])][str(r['seed'])] = float(r['nrmse'])
    res = {}
    for bench, g in groups.items():
        sizes = sorted(set(n for n, i in g if (n, pre) in g and (n, scratch) in g))
        if len(sizes) < 3:
            continue
        x = np.log10(sizes)
        seed_sets = [set(g[(n, pre)]) & set(g[(n, scratch)]) for n in sizes]
        paired = set.intersection(*seed_sets)
        mean_gain = [(np.mean(list(g[(n, scratch)].values())) - np.mean(list(g[(n, pre)].values())))
                     / np.mean(list(g[(n, scratch)].values())) for n in sizes]
        slope, _ = ols(x, mean_gain)
        if len(paired) >= 2:
            per_seed = [ols(x, [(g[(n, scratch)][s] - g[(n, pre)][s]) / g[(n, scratch)][s] for n in sizes])[0]
                        for s in sorted(paired)]
            m, sd, S = float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)), len(per_seed)
            half = float(student_t.ppf(0.975, S - 1) * sd / math.sqrt(S))
            ci, method = (m - half, m + half), f't-interval over {S} paired seeds'
            slope = m
        else:
            rng = np.random.default_rng(seed)
            boots = []
            for _ in range(n_boot):
                gm = []
                for n in sizes:
                    a = rng.choice(list(g[(n, scratch)].values()), size=len(g[(n, scratch)]))
                    b = rng.choice(list(g[(n, pre)].values()), size=len(g[(n, pre)]))
                    gm.append((a.mean() - b.mean()) / a.mean())
                boots.append(ols(x, gm)[0])
            ci, method = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))), 'unpaired bootstrap'
            print(f'  warning: {bench}: seeds are not paired across initialisations; using a bootstrap',
                  file=sys.stderr)
        res[bench] = {'sizes': sizes, 'relative_gain': [float(v) for v in mean_gain], 'slope_per_decade': slope,
                      'slope_ci95': ci, 'interval': method,
                      'criteria': {'D1 gain shrinks with more simulations (slope CI below 0)': bool(ci[1] < 0)}}
        if verbose:
            print(f"{bench}: relative gain of {pre} over {scratch} by training size: "
                  + ', '.join(f'{n}: {v:+.3f}' for n, v in zip(sizes, mean_gain)))
            print(f"  slope per decade {slope:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] ({method})")
            print_criteria(res[bench]['criteria'])
    return res


# ======================================================================================================================
# Level 2: explicit condition posterior

def condition_posterior(pred_c, y_obs, idx, log_prior, sigma2):
    resid = y_obs[None, :] - pred_c[:, idx]
    logp = log_prior - 0.5 * (resid ** 2).sum(axis=1) / sigma2
    logp -= logp.max()
    w = np.exp(logp)
    return w / w.sum()


def sensor_keys(data):
    return sorted((k for k in data if k.startswith('sensors_')), key=lambda k: int(k.split('_', 1)[1]))


def ece(pmax, hit, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    err = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pmax > a) & (pmax <= b) if a > 0 else (pmax >= a) & (pmax <= b)
        if m.any():
            err += m.sum() / len(pmax) * abs(pmax[m].mean() - hit[m].mean())
    return float(err)


def ece_null(pmax, hit, n_sim=2000, seed=0):
    """Observed ECE against its distribution under perfect calibration (outcomes drawn as Bernoulli(pmax))."""
    rng = np.random.default_rng(seed)
    obs = ece(pmax, hit)
    sims = np.array([ece(pmax, (rng.uniform(size=len(pmax)) < pmax).astype(float)) for _ in range(n_sim)])
    return obs, float(np.quantile(sims, 0.95)), float(((sims >= obs - 1e-12).sum() + 1) / (n_sim + 1))


def score_condition(data, sigma2, n_boot=0, seed=0):
    pred, target, area = data['pred'], data['target'], data['area']
    n, c, p = pred.shape
    prior_c = data['prior'] if 'prior' in data else np.full(c, 1.0 / c)
    log_prior = np.log(np.clip(prior_c, 1e-300, None))
    true_c = data['true_c'].astype(int) if 'true_c' in data else None
    group = data['group'] if 'group' in data else np.arange(n)
    out = {}
    for key in sensor_keys(data):
        K = key.split('_', 1)[1]
        s2 = sigma2[key] if isinstance(sigma2, dict) else sigma2
        sens = data[key].astype(int)
        rec = defaultdict(list)
        for i in range(n):
            hidden = np.ones(p, bool)
            hidden[sens[i]] = False
            w = condition_posterior(pred[i], target[i, sens[i]], sens[i], log_prior, s2)
            rec['post'].append(area_weighted_nrmse(w @ pred[i], target[i], area[i], hidden))
            rec['map'].append(area_weighted_nrmse(pred[i, int(w.argmax())], target[i], area[i], hidden))
            rec['pmax'].append(float(w.max()))
            rec['entropy'].append(float(-(w * np.log(np.clip(w, 1e-300, None))).sum()))
            if true_c is not None:
                t = int(true_c[i])
                rec['oracle'].append(area_weighted_nrmse(pred[i, t], target[i], area[i], hidden))
                rec['hit'].append(float(int(w.argmax()) == t))
                rec['nll'].append(float(-math.log(max(w[t], 1e-300))))
            if 'shape_only' in data:
                rec['shape'].append(area_weighted_nrmse(data['shape_only'][i], target[i], area[i], hidden))
            if f'amortised_{K}' in data:
                rec['amortised'].append(area_weighted_nrmse(data[f'amortised_{K}'][i], target[i], area[i], hidden))
        r = {k: float(np.mean(v)) for k, v in rec.items()}
        r['sigma2'] = float(s2)
        r['mean_normalised_mse'] = {k: float(np.mean(np.square(rec[k]))) for k in
                                    ('post', 'map', 'oracle', 'shape', 'amortised') if k in rec}
        if 'hit' in rec:
            obs, q95, pval = ece_null(np.array(rec['pmax']), np.array(rec['hit']), seed=seed)
            r.update(ece=obs, ece_null_q95=q95, ece_p=pval)
        if 'shape' in rec and 'oracle' in rec:
            m = r['mean_normalised_mse']
            for kind in ('post', 'amortised'):
                if kind in rec:
                    den, den2 = m['shape'] - m['oracle'], r['shape'] - r['oracle']
                    r[f'G_mse_{kind}'] = float((m['shape'] - m[kind]) / den) if den > 0 else float('nan')
                    r[f'G_nrmse_{kind}'] = float((r['shape'] - r[kind]) / den2) if den2 > 0 else float('nan')
        if 'amortised' in rec and n_boot:
            gap = np.array(rec['amortised']) - np.array(rec['post'])
            units = sorted(set(group.tolist()))
            by = {u: gap[group == u] for u in units}
            r['amortisation_gap'] = float(gap.mean())
            r['amortisation_gap_ci95'] = boot_ci(lambda us: float(np.mean(np.concatenate([by[u] for u in us]))),
                                                 units, n_boot, seed)
        out[K] = r
    return out


def load_npz_dict(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def choose_sigma2(val, verbose=True):
    """Per sensor count, the sensor-noise variance that minimises the validation negative log-likelihood of the true
    condition, searched over multiples of the residual variance of the true-condition model."""
    if 'true_c' not in val:
        sys.exit('the validation file needs true_c to choose sigma2')
    pred, target, true_c = val['pred'], val['target'], val['true_c'].astype(int)
    base = float(((target - pred[np.arange(len(true_c)), true_c]) ** 2).mean())
    mults = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    chosen = {}
    for key in sensor_keys(val):
        sub = {k: v for k, v in val.items() if not k.startswith('sensors_') or k == key}
        best = None
        for mult in mults:
            r = score_condition(sub, base * mult)[key.split('_', 1)[1]]
            if best is None or r['nll'] < best[1]:
                best = (base * mult, r['nll'])
        chosen[key] = best[0]
        if verbose:
            print(f'  {key}: sigma2 {best[0]:.4g} (validation NLL {best[1]:.3f})')
    return chosen


def condition(test_path, val_path=None, sigma2=None, gap_margin=0.01, n_boot=2000, seed=0, verbose=True):
    test = load_npz_dict(test_path)
    if sigma2 is None:
        if not val_path:
            sys.exit('give --val (preferred) or --sigma2')
        if verbose:
            print('choosing sigma2 per sensor count on the validation file')
        sigma2 = choose_sigma2(load_npz_dict(val_path), verbose)
        missing = [k for k in sensor_keys(test) if k not in sigma2]
        if missing:
            sys.exit(f'validation file lacks {missing}')
    res = {'by_sensor_count': score_condition(test, sigma2, n_boot, seed)}
    c = {}
    for K, r in res['by_sensor_count'].items():
        if 'ece_p' in r:
            c[f'C2 condition posterior not detectably miscalibrated at K={K} (ECE p > 0.05)'] = bool(r['ece_p'] > 0.05)
        if 'amortisation_gap_ci95' in r:
            lo, hi = r['amortisation_gap_ci95']
            c[f'C3 one-pass model within {gap_margin} nRMSE of the explicit posterior at K={K}'] = bool(
                lo > -gap_margin and hi < gap_margin)
    res['criteria'] = c
    if verbose:
        print(f"{'K':>5} {'posterior':>9} {'MAP':>7} {'true c':>7} {'shape':>7} {'one-pass':>8} {'c acc':>6} "
              f"{'ECE':>6} {'(null95)':>8} {'G post':>7} {'G 1-pass':>8} {'gap [95% CI]':>24}")
        for K, r in res['by_sensor_count'].items():
            gap = (f"{r['amortisation_gap']:+.4f} [{r['amortisation_gap_ci95'][0]:+.4f}, "
                   f"{r['amortisation_gap_ci95'][1]:+.4f}]") if 'amortisation_gap' in r else ''
            nan = float('nan')
            print(f"{K:>5} {r['post']:9.4f} {r['map']:7.4f} {r.get('oracle', nan):7.4f} {r.get('shape', nan):7.4f} "
                  f"{r.get('amortised', nan):8.4f} {r.get('hit', nan):6.2f} {r.get('ece', nan):6.3f} "
                  f"{r.get('ece_null_q95', nan):8.3f} {r.get('G_mse_post', nan):7.3f} "
                  f"{r.get('G_mse_amortised', nan):8.3f} {gap:>24}")
        print('  G is a diagnostic: with model error it can exceed 1 without any bug.')
        print_criteria(c)
    return res


# ======================================================================================================================
# Section 3.2: the normal-line head reports its own error

def risk_ece(pred, actual, bins=10):
    edges = np.unique(np.quantile(pred, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return float(abs(pred.mean() - actual.mean()))
    which = np.clip(np.searchsorted(edges, pred, side='right') - 1, 0, len(edges) - 2)
    err = 0.0
    for b in range(len(edges) - 1):
        m = which == b
        if m.any():
            err += m.sum() / len(pred) * abs(pred[m].mean() - actual[m].mean())
    return float(err)


def normals(folder, sym_order=DEFAULT_ORDER, n_boot=2000, seed=0, verbose=True):
    meshes = []
    for f in npz_files(folder):
        z = np.load(f, allow_pickle=False)
        P = sym3(z['Phat'], sym_order)
        w, V = np.linalg.eigh(P)
        wc = np.clip(w, 0.0, None)
        lmax = wc[..., -1] / np.clip(wc.sum(-1), 1e-12, None)
        n = unit(z['N'])
        meshes.append(dict(pred=1.0 - lmax, actual=1.0 - (V[..., :, -1] * n).sum(1) ** 2,
                           hidden=z['hidden'].astype(bool) if 'hidden' in z.files else np.zeros(len(n), bool),
                           trace=w.sum(-1), negative=w[..., 0] < -0.05))
    res = {'meshes': len(meshes)}
    for part in ('hidden', 'visible', 'all'):
        def sel(mesh):
            if part == 'all':
                return np.ones(len(mesh['pred']), bool)
            return mesh['hidden'] if part == 'hidden' else ~mesh['hidden']
        units = [i for i, mm in enumerate(meshes) if sel(mm).any()]
        if not units:
            continue

        def cat(us, key):
            return np.concatenate([meshes[i][key][sel(meshes[i])] for i in us])
        pred, actual = cat(units, 'pred'), cat(units, 'actual')
        slope, icpt = ols(pred, actual)
        res[part] = {
            'points': int(len(pred)), 'mean_predicted_risk': float(pred.mean()), 'mean_actual_sin2': float(actual.mean()),
            'calibration_slope': slope, 'calibration_intercept': icpt,
            'slope_ci95': boot_ci(lambda us: ols(cat(us, 'pred'), cat(us, 'actual'))[0], units, n_boot, seed),
            'spearman': spearman(pred, actual), 'ece': risk_ece(pred, actual),
            'trace_mean': float(cat(units, 'trace').mean()), 'trace_sd': float(cat(units, 'trace').std()),
            'negative_eigenvalue_fraction': float(cat(units, 'negative').mean()),
            'mean_normal_error_deg': float(np.degrees(np.arcsin(np.sqrt(np.clip(actual, 0, 1)))).mean())}
    part = 'hidden' if 'hidden' in res else 'all'
    h = res[part]
    res['criteria_scored_on'] = part
    res['criteria'] = {
        f'N1 calibration slope within [0.8, 1.25] ({part} points)':
            bool(np.isfinite(h['calibration_slope']) and 0.8 <= h['calibration_slope'] <= 1.25),
        f'N2 calibration intercept within 0.05 of zero ({part} points)':
            bool(np.isfinite(h['calibration_intercept']) and abs(h['calibration_intercept']) <= 0.05),
        f'N3 predicted risk ranks actual error, rho > 0.3 ({part} points)':
            bool(np.isfinite(h['spearman']) and h['spearman'] > 0.3),
        'N4 raw trace within 0.05 of 1 (all points)': bool(abs(res['all']['trace_mean'] - 1.0) <= 0.05)}
    if verbose:
        print(f"{res['meshes']} meshes; criteria use {part} points")
        for p in ('hidden', 'visible', 'all'):
            if p in res:
                r = res[p]
                print(f"{p:>8}: points {r['points']}, predicted risk {r['mean_predicted_risk']:.4f}, actual sin^2 "
                      f"{r['mean_actual_sin2']:.4f}, slope {r['calibration_slope']:.3f} [{r['slope_ci95'][0]:.3f}, "
                      f"{r['slope_ci95'][1]:.3f}], intercept {r['calibration_intercept']:+.4f}, rho "
                      f"{r['spearman']:.3f}, ECE {r['ece']:.4f}, trace {r['trace_mean']:.3f} +/- {r['trace_sd']:.3f}, "
                      f"normal error {r['mean_normal_error_deg']:.2f} deg")
        print_criteria(res['criteria'])
    return res


# ======================================================================================================================
# Section 2: single-condition equivalence

def equivalence(csv_path, a='measurements', b='geometry', case_col='car', group_col=None, delta=0.005,
                baseline_setting=None, n_boot=5000, seed=0, verbose=True):
    """Paired equivalence of method a and method b per setting (two one-sided tests via a 90% bootstrap CI).
    A method name also matches name_seed<N>, so seeds pool and are averaged per case. With baseline_setting, each
    setting of a is compared with baseline_setting of b. With group_col, the bootstrap resamples groups (e.g. physical
    geometries) instead of rows. Verdict: equivalent (CI inside +/- delta), different (CI outside it), or
    inconclusive."""
    rows = list(csv.DictReader(open(csv_path)))
    pat = {lab: re.compile(rf'^{re.escape(name)}(_seed\d+)?$') for lab, name in (('a', a), ('b', b))}
    vals = defaultdict(lambda: defaultdict(list))
    groups = {}
    for r in rows:
        for lab in ('a', 'b'):
            if pat[lab].match(r['method']):
                vals[(r['setting'], lab)][r[case_col]].append(float(r['nrmse']))
                groups[r[case_col]] = r[group_col] if group_col else r[case_col]
    res = {'a': a, 'b': b, 'baseline_setting': baseline_setting, 'delta': delta, 'group_col': group_col,
           'settings': {}}
    for s in sorted(set(s for s, _ in vals)):
        sb = baseline_setting or s
        if s == baseline_setting or (s, 'a') not in vals or (sb, 'b') not in vals:
            continue
        cases = sorted(set(vals[(s, 'a')]) & set(vals[(sb, 'b')]))
        diff = {cid: np.mean(vals[(s, 'a')][cid]) - np.mean(vals[(sb, 'b')][cid]) for cid in cases}
        by = defaultdict(list)
        for cid in cases:
            by[groups[cid]].append(diff[cid])
        units = sorted(by)
        mean = float(np.mean(list(diff.values())))
        lo, hi = boot_ci(lambda us: float(np.mean(np.concatenate([by[u] for u in us]))), units, n_boot, seed,
                         level=0.90)
        verdict = 'equivalent' if (lo > -delta and hi < delta) else ('different' if (lo > delta or hi < -delta)
                                                                     else 'inconclusive')
        res['settings'][s] = {'cases': len(cases), 'groups': len(units), 'mean_difference': mean, 'ci90': (lo, hi),
                              'verdict': verdict}
    vs = f"{b} at {baseline_setting}" if baseline_setting else b
    res['criteria'] = {f'E1 {a} at {s} equivalent to {vs} within {delta}':
                       {'equivalent': True, 'different': False, 'inconclusive': None}[v['verdict']]
                       for s, v in res['settings'].items()}
    if verbose:
        print(f'{a} minus {vs}, per-case nRMSE; margin {delta}; 90% bootstrap CI over '
              f"{group_col or case_col} (TOST at 5%)")
        for s, v in res['settings'].items():
            print(f"  {s:>16}: {v['cases']} cases in {v['groups']} groups, mean {v['mean_difference']:+.5f}, CI "
                  f"[{v['ci90'][0]:+.5f}, {v['ci90'][1]:+.5f}] -> {v['verdict']}")
    return res


# ======================================================================================================================
# Attention range and the attribution of pretraining's gains (Section 2, "Local and long-range dependence")

ATTN_EDGES = np.array([0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, np.inf])


def attention_summaries(Q, anchors, tw, A, R, edges=ATTN_EDGES):
    """Distance profile of query-to-token attention, with distances in units of R. For every layer, head and query:
    the attention-weighted mean distance to the token anchors and the attention mass in each distance bin. The baseline
    is the attention a query would pay with all logits zero: each token in proportion to its area weight, which is what
    the log-area bias alone gives. A is [L, H, M, T], [H, M, T] or [M, T], rows summing to one."""
    A = np.asarray(A, np.float64)
    while A.ndim < 4:
        A = A[None]
    rows = A.sum(-1)
    if np.abs(rows - 1).max() > 1e-3:
        print(f'  warning: attention rows sum to {rows.min():.4f}..{rows.max():.4f}; renormalising', file=sys.stderr)
        A = A / np.clip(rows[..., None], 1e-300, None)
    d = np.linalg.norm(np.asarray(Q, float)[:, None, :] - np.asarray(anchors, float)[None], axis=2) / float(R)
    nb = len(edges) - 1
    b = np.clip(np.searchsorted(edges, d, side='right') - 1, 0, nb - 1)
    p0 = np.asarray(tw, float) / float(np.sum(tw))
    bin_mass = np.zeros(A.shape[:3] + (nb,))
    base_mass = np.zeros((d.shape[0], nb))
    for k in range(nb):
        m = b == k
        bin_mass[..., k] = (A * m).sum(-1)
        base_mass[:, k] = m.astype(float) @ p0
    return dict(dist_mean=(A * d).sum(-1), bin_mass=bin_mass, base_mean=d @ p0, base_mass=base_mass,
                edges=np.asarray(edges, float))


def load_attention(folder, names):
    """{name: {shape_id: summary}} from DIR/<name>/<shape_id>.npz holding raw attention (Q, anchors, tw, A, R) or the
    summaries written by attention_summaries; plus each shape's bootstrap group (key 'group', default the shape)."""
    data, groups = {}, {}
    for name in names:
        files = npz_files(os.path.join(folder, name))
        if not files:
            sys.exit(f'no attention files for checkpoint {name!r} under {folder}')
        data[name] = {}
        for path in files:
            sid = os.path.splitext(os.path.basename(path))[0]
            z = np.load(path, allow_pickle=False)
            if 'A' in z.files:
                s = attention_summaries(z['Q'], z['anchors'], z['tw'], z['A'], float(z['R']))
            else:
                s = {k: np.asarray(z[k], float) for k in ('dist_mean', 'bin_mass', 'base_mean', 'base_mass', 'edges')}
            while s['dist_mean'].ndim < 3:
                s['dist_mean'] = s['dist_mean'][None]
            while s['bin_mass'].ndim < 4:
                s['bin_mass'] = s['bin_mass'][None]
            s['qw'] = np.asarray(z['qw'], float) if 'qw' in z.files else np.ones(len(s['base_mean']))
            data[name][sid] = s
            groups.setdefault(sid, str(z['group']) if 'group' in z.files else sid)
    return data, groups


def _edge_index(edges, rho):
    k = int(np.argmin(np.abs(edges - rho)))
    if abs(edges[k] - rho) > 1e-9:
        sys.exit(f'radius {rho} is not a bin edge; choose from {list(edges[1:-1])}')
    return k


def _attn_shape_stats(s, rhos, near=0.25):
    qw = s['qw'] / s['qw'].sum()
    edges, dm, bm, base = s['edges'], s['dist_mean'], s['bin_mass'], s['base_mass']
    D = np.einsum('lhm,m->lh', dm, qw)
    D0 = float(s['base_mean'] @ qw)
    out = {'D': float(D.mean()), 'D0': D0, 'ratio': float(D.mean() / D0), 'ratio_layer': D.mean(1) / D0}
    for rho in rhos:
        k = _edge_index(edges, rho)
        T = np.einsum('lhm,m->lh', bm[..., k:].sum(-1), qw)
        out[f'tail{rho:g}'] = float(T.mean())
        out[f'tail{rho:g}_layer'] = T.mean(1)
        out[f'tail{rho:g}_base'] = float(base[:, k:].sum(-1) @ qw)
    k = _edge_index(edges, near)
    out['near'] = float(np.einsum('lhm,m->lh', bm[..., :k].sum(-1), qw).mean())
    out['near_base'] = float(base[:, :k].sum(-1) @ qw)
    out['hist'] = np.einsum('lhmb,m->b', bm, qw) / (bm.shape[0] * bm.shape[1])
    out['hist_base'] = base.T @ qw
    return out


def sign_flip_pvalue(x, n_perm=20000, seed=0):
    """One-sided p-value for a positive mean of paired differences x (one per independent unit): exact enumeration of
    all sign flips for up to 16 units, Monte Carlo with the (count + 1) / (n + 1) correction otherwise."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan')
    obs = x.mean()
    if len(x) <= 16:
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(x))))
        return float(((signs * x).mean(1) >= obs - 1e-15).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(x)))
    return float((((signs * x).mean(1) >= obs - 1e-15).sum() + 1) / (n_perm + 1))


def t_interval(x, level=0.95):
    """Mean and Student-t interval of independent unit-level values (one per physical geometry)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float('nan'), (float('nan'), float('nan'))
    m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))
    q = float(student_t.ppf(0.5 + level / 2, len(x) - 1))
    return m, (m - q * se, m + q * se)


def attention(folder, pre='pretrained', post='finetuned', scratch=None, rho=0.5, rhos=(0.25, 0.5, 1.0), n_boot=2000,
              seed=0, verbose=True):
    """Pretrained attention should be short-range and fine-tuning should add long-range mass. Shapes are paired across
    checkpoints (same inputs, same queries). Each group of shapes (a physical geometry) contributes one value, its
    mean; intervals are Student-t over groups and the sign-flip test permutes groups. n_boot is unused."""
    names = [pre, post] + ([scratch] if scratch else [])
    data, groups = load_attention(folder, names)
    shapes = sorted(set.intersection(*(set(data[n]) for n in names)))
    if not shapes:
        sys.exit('no shape appears under every checkpoint')
    rhos = sorted(set(rhos) | {rho})
    st = {n: {sid: _attn_shape_stats(data[n][sid], rhos) for sid in shapes} for n in names}
    by = defaultdict(list)
    for sid in shapes:
        by[groups[sid]].append(sid)
    units = sorted(by)

    def group_values(fn):
        return [float(np.mean([fn(sid) for sid in by[u]])) for u in units]

    res = {'checkpoints': names, 'shapes': len(shapes), 'groups': len(units), 'rho': rho,
           'edges': list(data[pre][shapes[0]]['edges']), 'per_checkpoint': {}, 'contrasts': {}}
    for n in names:
        summ = {}
        for key in ['ratio', 'near'] + [f'tail{r:g}' for r in rhos]:
            m, ci = t_interval(group_values(lambda sid, n=n, key=key: st[n][sid][key]))
            summ[key] = {'mean': m, 'ci95': ci}
        for key in ['D', 'D0', 'near_base'] + [f'tail{r:g}_base' for r in rhos]:
            summ[key] = float(np.mean([st[n][sid][key] for sid in shapes]))
        for key in ['ratio_layer', f'tail{rho:g}_layer', 'hist', 'hist_base']:
            summ[key] = np.mean([st[n][sid][key] for sid in shapes], 0)
        res['per_checkpoint'][n] = summ

    def contrast(a, b):
        key = f'tail{rho:g}'
        gm = group_values(lambda sid: st[a][sid][key] - st[b][sid][key])
        m, (lo, hi) = t_interval(gm)
        layer = np.mean([st[a][sid][f'{key}_layer'] - st[b][sid][f'{key}_layer'] for sid in shapes], 0)
        return {'mean': m, 'ci95': (lo, hi), 'p_one_sided': sign_flip_pvalue(gm, seed=seed),
                'per_layer': layer, 'verdict': 'longer' if lo > 0 else ('shorter' if hi < 0 else 'inconclusive')}
    res['contrasts'][f'{post} - {pre}'] = contrast(post, pre)
    if scratch:
        res['contrasts'][f'{scratch} - {pre}'] = contrast(scratch, pre)
    enough = len(units) >= 5            # with fewer units no sign-flip test can reach p <= 0.05
    lo, hi = res['per_checkpoint'][pre]['ratio']['ci95']
    verdict = res['contrasts'][f'{post} - {pre}']['verdict']
    res['criteria'] = {
        f'A1 {pre} attention is shorter-range than the area baseline (mean-distance ratio, 95% CI below 1)':
            (True if hi < 1 else (False if lo > 1 else None)) if enough else None,
        f'A2 fine-tuning adds long-range attention (mass beyond {rho:g} R, {post} minus {pre}, 95% CI above 0)':
            {'longer': True, 'shorter': False, 'inconclusive': None}[verdict] if enough else None}
    if verbose:
        print(f'attention distances in units of R; {len(shapes)} shapes in {len(units)} groups; area baseline '
              f'in brackets')
        print(f"  {'checkpoint':<14}{'mean distance / baseline':>28}{'within 0.25R':>18}" +
              ''.join(f'{"beyond " + format(r, "g") + "R":>16}' for r in rhos))
        for n in names:
            s = res['per_checkpoint'][n]
            r = s['ratio']
            line = f"  {n:<14}{r['mean']:>10.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
            line += f"{s['near']['mean']:>11.3f} ({s['near_base']:.3f})"
            line += ''.join(f"{s[f'tail{q:g}']['mean']:>9.3f} ({s[f'tail{q:g}_base']:.3f})" for q in rhos)
            print(line)
            print(f"  {'':<14}per layer: ratio " + ' '.join(f'{v:.3f}' for v in s['ratio_layer']) +
                  f"; beyond {rho:g}R " + ' '.join(f'{v:.3f}' for v in s[f'tail{rho:g}_layer']))
        for k, c in res['contrasts'].items():
            print(f"  {k}: mass beyond {rho:g}R {c['mean']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}], "
                  f"one-sided p {c['p_one_sided']:.4f}, per layer " + ' '.join(f'{v:+.4f}' for v in c['per_layer'])
                  + f" -> {c['verdict']}")
        print_criteria(res['criteria'])
    return res


def tangent_frames(N):
    n = unit(N)
    helper = np.where(np.abs(n[:, :1]) < 0.9, np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 1.0, 0.0]]))
    t1 = unit(np.cross(n, helper))
    return n, t1, np.cross(n, t1)


def shape_operator_norm(X, N, radius, k=24, min_nb=6):
    """Frobenius norm of the shape operator, sqrt(k1^2 + k2^2), at resolution `radius`, without orientations. Uses the
    identity of Lemma 1: the derivative of the projector n n^T along a tangent u, applied to n, is W u. For each point,
    the neighbours within `radius` give u (tangent offset) and (P_j - P_i) n_i ~ W u, and a symmetric 2x2 W is fitted
    by least squares. NaN where fewer than min_nb neighbours fall inside the radius."""
    X = np.asarray(X, float)
    n, t1, t2 = tangent_frames(N)
    kk = min(k + 1, len(X))
    dist, idx = cKDTree(X).query(X, k=kk)
    dist, idx = dist[:, 1:], idx[:, 1:]
    dx = X[idx] - X[:, None, :]
    u1, u2 = (dx * t1[:, None]).sum(-1), (dx * t2[:, None]).sum(-1)
    nj = n[idx]
    g3 = nj * (nj * n[:, None]).sum(-1, keepdims=True) - n[:, None]          # (P_j - P_i) n_i, sign-free
    g1, g2 = (g3 * t1[:, None]).sum(-1), (g3 * t2[:, None]).sum(-1)
    w = (dist <= radius).astype(float)
    M = np.zeros((len(X), 3, 3))
    M[:, 0, 0] = (w * u1 * u1).sum(1)
    M[:, 0, 1] = M[:, 1, 0] = (w * u1 * u2).sum(1)
    M[:, 1, 1] = (w * (u1 * u1 + u2 * u2)).sum(1)
    M[:, 1, 2] = M[:, 2, 1] = (w * u1 * u2).sum(1)
    M[:, 2, 2] = (w * u2 * u2).sum(1)
    rhs = np.stack([(w * u1 * g1).sum(1), (w * (u2 * g1 + u1 * g2)).sum(1), (w * u2 * g2).sum(1)], 1)
    reg = 1e-12 * (np.trace(M, axis1=1, axis2=2) + 1e-30)
    sol = np.linalg.solve(M + reg[:, None, None] * np.eye(3), rhs[..., None])[..., 0]
    norm = np.sqrt(sol[:, 0] ** 2 + 2 * sol[:, 1] ** 2 + sol[:, 2] ** 2)
    norm[w.sum(1) < min_nb] = np.nan
    return norm


def weighted_quantile(x, w, qs):
    o = np.argsort(x)
    x, w = np.asarray(x, float)[o], np.asarray(w, float)[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, c, x)


def _attr_case(z, a, b, res_frac):
    X, N, A, f = (np.asarray(z[k], float) for k in ('X', 'N', 'A', 'f'))
    pa, pb = np.asarray(z[f'pred_{a}'], float), np.asarray(z[f'pred_{b}'], float)
    R = float(z['R']) if 'R' in z.files else rms_radius(X, A)
    scored = np.asarray(z['scored'], bool) if 'scored' in z.files else np.ones(len(X), bool)
    kappa = shape_operator_norm(X, N, res_frac * R) * R
    if np.isnan(kappa[scored]).mean() > 0.1:
        print(f'  warning: curvature undefined at {100 * np.isnan(kappa[scored]).mean():.0f}% of points (fewer than 6 '
              f'neighbours within {res_frac:g} R); raise --res or export more points', file=sys.stderr)
    w = A / A.sum()
    spread = float(z['spread']) if 'spread' in z.files else math.sqrt(float(w @ (f - w @ f) ** 2))
    ext = np.abs(f - weighted_quantile(f, A, [0.5])[0]) / max(spread, 1e-300)
    m = scored & np.isfinite(kappa) & np.isfinite(pa) & np.isfinite(pb) & np.isfinite(f)
    if m.sum() < 30 or spread <= 0:
        return None
    ci = np.digitize(kappa, weighted_quantile(kappa[m], A[m], [1 / 3, 2 / 3]))
    xi = np.digitize(ext, weighted_quantile(ext[m], A[m], [1 / 3, 2 / 3]))
    ea2, eb2 = ((pa - f) / spread) ** 2, ((pb - f) / spread) ** 2
    cells = np.zeros((3, 3, 3))                        # [curvature tercile, extremity tercile, (Sa, Sb, area)]
    for c in range(3):
        for x in range(3):
            mm = m & (ci == c) & (xi == x)
            cells[c, x] = (A[mm] @ ea2[mm], A[mm] @ eb2[mm], A[mm].sum())
    regions = {}
    if 'region' in z.files:
        reg = np.asarray(z['region'], int)
        for r in np.unique(reg[m]):
            mm = m & (reg == r)
            regions[int(r)] = np.array([A[mm] @ ea2[mm], A[mm] @ eb2[mm], A[mm].sum()])
    return {'cells': cells, 'regions': regions}


def attribution(folder, a='finetuned', b='scratch', res_frac=0.1, j_region=None, b_region=None, n_boot=2000, seed=0,
                verbose=True):
    """Where do the gains of model a over model b (pretrained over scratch) sit? Gain of a stratum within a physical
    geometry: one minus the ratio of its pooled, spread-normalised, area-weighted squared errors. Strata: terciles
    (per case, by area) of local curvature at resolution res_frac * R and of field extremity |f - median| / spread.
    Local stratum: most curved third with non-extreme values; global stratum: flattest third with the most extreme
    values. Gains and contrasts are averaged over geometries with Student-t intervals. n_boot is unused."""
    cases, groups = {}, {}
    for path in npz_files(folder):
        z = np.load(path, allow_pickle=False)
        cid = os.path.splitext(os.path.basename(path))[0]
        r = _attr_case(z, a, b, res_frac)
        if r is None:
            print(f'  warning: skipped {cid} (too few scored points or zero spread)', file=sys.stderr)
            continue
        cases[cid] = r
        groups[cid] = str(z['group']) if 'group' in z.files else cid
    if not cases:
        sys.exit(f'no usable cases under {folder}')
    by = defaultdict(list)
    for cid in cases:
        by[groups[cid]].append(cid)
    units = sorted(by)
    local = np.zeros((3, 3), bool)
    local[2, :2] = True
    glob_ = np.zeros((3, 3), bool)
    glob_[0, 2] = True

    def gain_of(t):
        return 1 - t[0] / t[1] if t[1] > 0 else float('nan')

    def group_gains(sel):
        return np.array([gain_of(sum(cases[c]['cells'][sel].sum(0) for c in by[u])) for u in units])

    def group_region_gains(r):
        return np.array([gain_of(sum(cases[c]['regions'].get(r, np.zeros(3)) for c in by[u])) for u in units])

    allc = np.ones((3, 3), bool)
    total = sum(cases[c]['cells'] for c in cases)
    share = total[..., 2] / total[..., 2].sum()
    res = {'a': a, 'b': b, 'cases': len(cases), 'groups': len(units), 'resolution': res_frac,
           'area_share': share, 'cell_gain': np.where(total[..., 1] > 0, 1 - total[..., 0] / np.where(
               total[..., 1] > 0, total[..., 1], 1), np.nan)}
    for name, sel in (('all', allc), ('local', local), ('global', glob_)):
        m, ci = t_interval(group_gains(sel))
        res[name] = {'gain': m, 'ci95': ci, 'area_share': float(share[sel].sum())}
    contrast = group_gains(local) - group_gains(glob_)
    diff, (lo, hi) = t_interval(contrast)
    verdict = 'local' if lo > 0 else ('global' if hi < 0 else 'inconclusive')
    res['local_minus_global'] = {'difference': diff, 'ci95': (lo, hi), 'verdict': verdict,
                                 'p_one_sided': sign_flip_pvalue(contrast, seed=seed)}
    labels = sorted({r for c in cases.values() for r in c['regions']})
    if labels:
        res['regions'] = {}
        for r in labels:
            m, ci = t_interval(group_region_gains(r))
            res['regions'][r] = {'gain': m, 'ci95': ci}
        if j_region is not None and b_region is not None:
            m, ci = t_interval(group_region_gains(j_region) - group_region_gains(b_region))
            res['region_contrast'] = {'difference': m, 'ci95': ci, 'local_region': j_region,
                                      'global_region': b_region}
    populated = res['local']['area_share'] >= 0.02 and res['global']['area_share'] >= 0.02
    enough = len(units) >= 5
    res['criteria'] = {
        'R1 local and global strata each hold at least 2% of the scored area': populated,
        f'R2 gains of {a} over {b} concentrate where local geometry dominates (local minus global, 95% CI above 0)':
            ({'local': True, 'global': False, 'inconclusive': None}[verdict] if populated and enough else None)}
    if verbose:
        print(f'gain of {a} over {b} = 1 - (pooled normalised MSE of {a}) / (that of {b}); {len(cases)} cases in '
              f'{len(units)} groups; curvature at {res_frac:g} R')
        for name in ('all', 'local', 'global'):
            g = res[name]
            print(f"  {name:<7} gain {g['gain']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}], "
                  f"{100 * g['area_share']:.1f}% of area")
        print(f"  local minus global {diff:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}")
        print('  pooled gain by curvature tercile (rows, flat to curved) and extremity tercile (columns, mild to '
              'extreme), with area shares:')
        for c in range(3):
            print('    ' + '  '.join(f"{res['cell_gain'][c, x]:+.3f} ({100 * share[c, x]:4.1f}%)" for x in range(3)))
        for r, g in res.get('regions', {}).items():
            print(f"  region {r}: gain {g['gain']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]")
        if 'region_contrast' in res:
            rc = res['region_contrast']
            print(f"  region {rc['local_region']} minus region {rc['global_region']}: {rc['difference']:+.3f} "
                  f"[{rc['ci95'][0]:+.3f}, {rc['ci95'][1]:+.3f}]")
        print_criteria(res['criteria'])
    return res


# ======================================================================================================================
# Synthetic self-tests

def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def cube_samples(n_per_face, rng, half=1.0):
    X, N = [], []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            uv = rng.uniform(-half, half, size=(n_per_face, 2))
            pts = np.zeros((n_per_face, 3))
            others = [d for d in range(3) if d != axis]
            pts[:, others[0]], pts[:, others[1]] = uv[:, 0], uv[:, 1]
            pts[:, axis] = sign * half
            nrm = np.zeros((n_per_face, 3))
            nrm[:, axis] = sign
            X.append(pts)
            N.append(nrm)
    return np.vstack(X), np.vstack(N)


def _check(name, ok, detail=''):
    print(f"  [{'ok' if ok else 'FAILED'}] {name}{(': ' + detail) if detail else ''}")
    if not ok:
        raise AssertionError(name)


def _write_sphere_dataset(tmp, rng, n_shapes=60, n_test=20, npts=400, conds=(0.0, 1.0), k_field=2.0):
    dirs = fibonacci_sphere(npts)
    os.makedirs(os.path.join(tmp, 'shapes'), exist_ok=True)
    radii = {}
    for i in range(n_shapes):
        r = rng.uniform(1.0, 2.0)
        sid = f's{i:03d}'
        radii[sid] = r
        F = np.stack([k_field * r + 0.5 * c + 0.3 * dirs[:, 0] for c in conds])
        np.savez(os.path.join(tmp, 'shapes', sid + '.npz'), X=dirs * r, N=dirs,
                 A=np.full(npts, 4 * np.pi * r * r / npts), F=F, cond=np.array(conds), R=r,
                 split='train' if i < n_shapes - n_test else 'test')
    return dirs, radii


def selftest():
    rng = np.random.default_rng(0)
    print('distances')
    g = np.stack(np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30)), -1).reshape(-1, 2)
    X = np.hstack([g, np.zeros((len(g), 1))])
    P = outer_normals(np.tile([0.0, 0.0, 1.0], (len(X), 1)))
    A = np.ones(len(X))
    d = surface_distance(X, P, A, X + [0, 0, 0.05], P, scale=1.0)
    _check('offset plane gives Hausdorff = offset', abs(d['hausdorff'] - 0.05) < 1e-9, f"{d['hausdorff']:.6f}")
    ang = np.radians(20)
    Pt = outer_normals(np.tile([np.sin(ang), 0.0, np.cos(ang)], (len(X), 1)))
    d = surface_distance(X, P, A, X, Pt, scale=1.0, lam=0.5)
    _check('tilted normals give |sin(angle)| with lambda = 1/2', abs(d['hausdorff'] - np.sin(ang)) < 1e-9,
           f"{d['hausdorff']:.6f} vs {np.sin(ang):.6f}")
    Xa, Na = cube_samples(800, rng)
    Xb, Nb = cube_samples(800, rng)
    Pa, Pb = outer_normals(Na), outer_normals(Nb)
    h = pair_distance_selftest(Xa, Pa, Xb, Pb)
    _check('two samplings of one cube are close (no crease artefact)', h < 0.06, f'{h:.4f}')
    d = surface_distance(Xa, Pa, np.ones(len(Xa)), Xa, Pa, scale=1.0)
    _check('perfect reconstruction of a cube has d = 0', d['hausdorff'] < 1e-9, f"{d['hausdorff']:.2e}")
    dirs = fibonacci_sphere(1500)
    prev = -1.0
    for off in (0.03, 0.04, 0.06):
        Xh, uh, Ph = line_search(dirs, dirs, 1.0, lambda Q: (np.abs(np.linalg.norm(Q, axis=1) - (1 + off)),
                                                              outer_normals(Q)))
        on = uh < 0.01
        d = surface_distance(dirs, outer_normals(dirs), np.ones(len(dirs)), Xh[on], Ph[on], scale=1.0)
        _check(f'normal-line search finds a surface offset by {off:.0%}', on.all() and abs(d['hausdorff'] - off) < 2e-3,
               f"d_H {d['hausdorff']:.4f}")
        _check('distance grows with the offset', d['hausdorff'] > prev)
        prev = d['hausdorff']

    tmp = tempfile.mkdtemp(prefix='theory_tests_')
    try:
        print('Proposition 1 (spheres, field = k * radius, known L)')
        dirs, radii = _write_sphere_dataset(tmp, rng)
        levels = {'ck_a': 0.005, 'ck_b': 0.01, 'ck_c': 0.03, 'ck_d': 0.08, 'ck_e': 0.2}
        for name, sig in levels.items():
            os.makedirs(os.path.join(tmp, 'ckpt', name))
            for sid, r in radii.items():
                rh = r + rng.normal(0, sig)
                phi = np.hstack([np.tile([rh, rh ** 2], (len(dirs), 1)), dirs]) \
                    + 1e-4 * rng.standard_normal((len(dirs), 5))
                Qb = rng.uniform(-3, 3, size=(500, 3))
                np.savez(os.path.join(tmp, 'ckpt', name, sid + '.npz'), phi=phi, Xhat=dirs * rh,
                         udf_hat=np.zeros(len(dirs)), Phat=outer_normals(dirs), Qb=Qb,
                         udf_b=np.abs(np.linalg.norm(Qb, axis=1) - rh), PQb=outer_normals(Qb))
        res = prop1(tmp, list(levels), order=list(levels), points=0, k=3, n_boot=300, verbose=False)
        shapes = load_shapes(tmp)
        fstd = field_stats(shapes, [s for s in shapes if shapes[s]['split'] == 'train'])[1]
        L_true = 2.0 * res['scale'] / fstd
        _check('median Lipschitz estimate equals k * scale / std', abs(res['L']['median'] / L_true - 1) < 0.02,
               f"{res['L']['median']:.4f} vs {L_true:.4f}")
        crit = res['criteria']
        _check('bound holds for every checkpoint', next(v for k, v in crit.items() if k.startswith('P1')))
        _check('across checkpoints passes with 5 checkpoints', next(v for k, v in crit.items() if k.startswith('P2')),
               f"rho {res['across']['spearman']:.2f}, p {res['across']['p_one_sided']:.3f}")
        _check('across shapes passes', next(v for k, v in crit.items() if k.startswith('P3')),
               f"{res['within']['mean_spearman']:.3f}")
        _check('dose-response passes with 5 levels', next(v for k, v in crit.items() if k.startswith('P4')))
        res3 = prop1(tmp, ['ck_a', 'ck_c', 'ck_e'], points=0, k=3, n_boot=100, verbose=False)
        _check('three checkpoints give SKIP for the across-checkpoint test', any(
            k.startswith('P2') and v is None for k, v in res3['criteria'].items()))

        print('probe with many conditions (continuous basis needs condition main effects)')
        tmp2 = os.path.join(tmp, 'many')
        _write_sphere_dataset(tmp2, rng, n_shapes=30, n_test=10, npts=200, conds=tuple(np.linspace(0, 1, 25)))
        shapes2 = load_shapes(tmp2)
        feats2 = {s: np.hstack([np.full((200, 1), shapes2[s]['R']), fibonacci_sphere(200)]) for s in shapes2}
        tr2, te2 = split_ids(shapes2)
        fm, fs = field_stats(shapes2, tr2)
        probe = FieldProbe(shapes2, feats2, tr2, te2, fm, fs)
        nr = float(np.mean([probe.score(shapes2[s], feats2[s])[1] for s in te2]))
        _check('continuous probe fits a condition main effect exactly', (not probe.discrete) and nr < 1e-3,
               f'nRMSE {nr:.2e}')

        print('condition labels')
        os.makedirs(os.path.join(tmp, 'labels', 'shapes'))
        np.savez(os.path.join(tmp, 'labels', 'shapes', 'a.npz'), X=dirs, N=dirs, A=np.ones(len(dirs)),
                 F=np.stack([dirs[:, 0], dirs[:, 1]]), split='train')
        np.savez(os.path.join(tmp, 'labels', 'shapes', 'b.npz'), X=dirs, N=dirs, A=np.ones(len(dirs)),
                 F=dirs[:, 0][None], cond=np.array([0.3, 5.0]), split='test')
        sh = load_shapes(os.path.join(tmp, 'labels'))
        _check('missing cond defaults to one label per row', sh['a']['cond'].ravel().tolist() == [0.0, 1.0])
        _check('a single condition given as [k] is one row', sh['b']['cond'].shape == (1, 2))

        print('Level 1 evidence (informative vs random features)')
        for name in ('feat_good', 'feat_rand'):
            os.makedirs(os.path.join(tmp, 'ckpt', name))
        for sid, r in radii.items():
            np.savez(os.path.join(tmp, 'ckpt', 'feat_good', sid + '.npz'),
                     phi=np.hstack([np.tile([r], (len(dirs), 1)), dirs, 0.1 * rng.standard_normal((len(dirs), 3))]))
            np.savez(os.path.join(tmp, 'ckpt', 'feat_rand', sid + '.npz'), phi=rng.standard_normal((len(dirs), 7)))
        res = prior(tmp, ['feat_good', 'feat_rand'], reference='feat_rand', points=0, ns=(5, 10, 20, 40), repeats=4,
                    n_boot=300, verbose=False)
        g = res['checkpoints']['feat_good']['vs_reference']
        _check('informative features give higher held-out log-likelihood', g['loglik_gain_ci95'][0] > 0,
               f"gain {g['loglik_gain']:.3f}")

        print('Level 1 data curve (paired seeds)')

        def write_curve(path, decay):
            with open(path, 'w', newline='') as fh:
                wr = csv.writer(fh)
                wr.writerow(['n', 'init', 'seed', 'nrmse'])
                for n in (25, 50, 100, 200, 400):
                    gain = 0.6 * n ** (-decay) if decay else 0.15
                    for s in range(5):
                        base_v = 0.25 + rng.normal(0, 0.002)
                        wr.writerow([n, 'scratch', s, base_v])
                        wr.writerow([n, 'pretrained', s, base_v * (1 - gain) + rng.normal(0, 0.001)])
        path = os.path.join(tmp, 'curve.csv')
        write_curve(path, 0.4)
        r = datacurve(path, verbose=False)['all']
        _check('shrinking gain detected', next(iter(r['criteria'].values())), f"slope {r['slope_per_decade']:+.3f}")
        false_pass = 0
        for _ in range(60):
            write_curve(path, 0.0)
            false_pass += next(iter(datacurve(path, verbose=False)['all']['criteria'].values()))
        _check('flat gain rarely mistaken for shrinking (false-pass rate <= 0.10 over 60 replicates)',
               false_pass / 60 <= 0.10, f'{false_pass}/60')

        print('Level 2 condition posterior')
        n, c, p = 120, 9, 400
        x = rng.uniform(-1, 1, size=(n, p))
        pred = np.stack([np.sin(3 * x) + 0.03 * j * x for j in range(c)], axis=1)
        true_c = rng.integers(0, c, size=n)
        target = pred[np.arange(n), true_c] + 0.05 * rng.standard_normal((n, p))
        base = {'pred': pred, 'area': rng.uniform(0.5, 1.5, (n, p)), 'true_c': true_c.astype(float),
                'shape_only': pred.mean(1), 'group': np.arange(n) // 3}
        for K in (5, 20, 100):
            base[f'sensors_{K}'] = np.stack([rng.choice(p, K, replace=False) for _ in range(n)])
        test, val = dict(base, target=target), dict(base, target=pred[np.arange(n), true_c]
                                                    + 0.05 * rng.standard_normal((n, p)))
        for K in (5, 20, 100):
            post = []
            for i in range(n):
                idx = test[f'sensors_{K}'][i]
                w = condition_posterior(pred[i], target[i, idx], idx, np.log(np.full(c, 1 / c)), 0.05 ** 2)
                post.append(w @ pred[i])
            test[f'amortised_{K}'] = np.array(post) + 0.002 * rng.standard_normal((n, p))
        np.savez(os.path.join(tmp, 'cond_test.npz'), **test)
        np.savez(os.path.join(tmp, 'cond_val.npz'), **val)
        res = condition(os.path.join(tmp, 'cond_test.npz'), os.path.join(tmp, 'cond_val.npz'), n_boot=300,
                        verbose=False)
        by = res['by_sensor_count']
        _check('condition identified at K=100', by['100']['hit'] > 0.9, f"{by['100']['hit']:.2f}")
        _check('posterior spread is genuine at K=5 (not a degenerate test)', by['5']['pmax'] < 0.95,
               f"mean pmax {by['5']['pmax']:.2f}")
        flags = {'true sigma2': 0, 'sigma2 / 16': 0}
        for _ in range(20):
            tc = rng.integers(0, c, size=n)
            tg = pred[np.arange(n), tc] + 0.05 * rng.standard_normal((n, p))
            d = {'pred': pred, 'target': tg, 'area': base['area'], 'true_c': tc.astype(float),
                 'sensors_20': base['sensors_20']}
            for label, s2 in (('true sigma2', 0.05 ** 2), ('sigma2 / 16', 0.05 ** 2 / 16)):
                flags[label] += score_condition(d, s2, seed=int(rng.integers(1 << 30)))['20']['ece_p'] <= 0.05
        _check('calibrated posterior rarely flagged (<= 4 of 20 replicates)', flags['true sigma2'] <= 4,
               f"{flags['true sigma2']}/20")
        _check('overconfident posterior usually flagged (>= 14 of 20 replicates)', flags['sigma2 / 16'] >= 14,
               f"{flags['sigma2 / 16']}/20")
        _check('amortisation gap near zero when the one-pass model equals the posterior', all(
            v for k, v in res['criteria'].items() if k.startswith('C3')))

        print('Section 3.2 normal-line self-reported error')
        heads = (('calibrated', lambda q: q), ('overconfident', lambda q: q ** (1 / 3)),
                 ('constant', lambda q: np.full_like(q, 0.5)))
        for label, conf in heads:
            folder = os.path.join(tmp, f'normals_{label}')
            os.makedirs(folder)
            for mesh in range(30):
                m = 400
                nvec = fibonacci_sphere(m)[rng.permutation(m)]
                hidden = rng.uniform(0, 1, m) < 0.6
                q = np.where(hidden, rng.uniform(0, 1, m), 1.0)
                rnd = unit(rng.standard_normal((m, 3)))
                truth = np.where((rng.uniform(0, 1, m) < q)[:, None], nvec, rnd)
                qq = np.where(hidden, conf(q), 1.0)
                Phat = qq[:, None, None] * outer_normals(nvec) + (1 - qq)[:, None, None] * np.eye(3) / 3
                np.savez(os.path.join(folder, f'm{mesh:02d}.npz'), N=truth, Phat=Phat, hidden=hidden)
            r = normals(folder, n_boot=200, verbose=False)
            crit = list(r['criteria'].values())
            if label == 'calibrated':
                _check('posterior-mean head passes N1-N4 on hidden points', all(crit),
                       f"slope {r['hidden']['calibration_slope']:.3f}")
            elif label == 'overconfident':
                _check('overconfident head fails N1', not crit[0], f"slope {r['hidden']['calibration_slope']:.3f}")
            else:
                _check('constant-risk head fails N3 (the mask alone cannot pass it)', not crit[2])

        print('Section 2 single-condition equivalence')
        path = os.path.join(tmp, 'cases.csv')
        with open(path, 'w', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(['method', 'car', 'family', 'setting', 'nrmse'])
            for cid in range(48):
                base_v = rng.uniform(0.2, 0.3)
                for s in range(3):
                    wr.writerow([f'geometry_seed{s}', cid, cid // 2, 'sensors-20', base_v])
                    wr.writerow([f'measurements_seed{s}', cid, cid // 2, 'sensors-20', base_v + rng.normal(0, 0.001)])
                    wr.writerow([f'geometry_seed{s}', cid, cid // 2, 'sensors-500', base_v])
                    wr.writerow([f'measurements_seed{s}', cid, cid // 2, 'sensors-500',
                                 base_v + 0.012 + rng.normal(0, 0.001)])
                wr.writerow(['geometry_large', cid, cid // 2, 'sensors-20', base_v + 1.0])
        r = equivalence(path, group_col='family', n_boot=1000, verbose=False)['settings']
        _check('equivalent methods pass and large-model rows are not pooled', r['sensors-20']['verdict'] == 'equivalent',
               f"mean {r['sensors-20']['mean_difference']:+.4f}")
        _check('offset methods are judged different', r['sensors-500']['verdict'] == 'different')
        _check('bootstrap resamples groups', r['sensors-20']['groups'] == 24)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _selftest_attention_attribution()
    print('all self-tests passed')


def _ellipsoid(n, axes):
    """Points, unit normals and area weights on an ellipsoid (mapped Fibonacci sphere)."""
    u = fibonacci_sphere(n)
    ax = np.asarray(axes, float)
    g = u / ax
    return u * ax, unit(g), float(np.prod(ax)) * np.linalg.norm(g, axis=1) * 4 * np.pi / n


def _synthetic_attention(folder, rng, n_shapes=16, M=120, T=48, width=0.12, post_mix=0.25, mode='tails'):
    """Pretrained attention is a Gaussian kernel in distance (times the area weights). 'tails': fine-tuning mixes in a
    share of area-uniform attention; 'reverse': the reverse; 'null': both checkpoints get small random mixes."""
    anchors0 = fibonacci_sphere(T)
    for i in range(n_shapes):
        sid = f'shape{i:02d}'
        r = rng.uniform(0.8, 1.2)
        anchors = anchors0 * r
        Q = fibonacci_sphere(400)[rng.choice(400, M, replace=False)] * r
        tw = rng.uniform(0.5, 1.5, T)
        d = np.linalg.norm(Q[:, None] - anchors[None], axis=2) / r
        local = tw * np.exp(-d ** 2 / (2 * width ** 2))
        local /= local.sum(1, keepdims=True)
        unif = np.tile(tw / tw.sum(), (M, 1))
        m_pre, m_post = rng.uniform(0, 0.04, 2) if mode == 'null' else (0.0, post_mix)
        A_pre, A_post = (1 - m_pre) * local + m_pre * unif, (1 - m_post) * local + m_post * unif
        if mode == 'reverse':
            A_pre, A_post = A_post, A_pre
        for name, Am in (('pretrained', A_pre), ('finetuned', A_post)):
            os.makedirs(os.path.join(folder, name), exist_ok=True)
            A4 = np.stack([np.stack([Am, Am])] * 2)                        # [layers, heads, queries, tokens]
            path = os.path.join(folder, name, sid + '.npz')
            if i % 2 == 0:
                np.savez(path, Q=Q, anchors=anchors, tw=tw, A=A4, R=r, group=f'g{i // 2}')
            else:
                np.savez(path, R=r, group=f'g{i // 2}', **attention_summaries(Q, anchors, tw, A4, r))


def _synthetic_attribution(folder, rng, n_cases=20, mode='local', npts=1200, res_frac=0.2):
    """Ellipsoids with a suction peak over the flat upper face. 'local': the better model halves its error on the most
    curved third; 'global': on flat points with extreme values; 'null': the same factor everywhere."""
    for i in range(n_cases):
        s = rng.uniform(0.8, 1.2)
        X, N, A = _ellipsoid(npts, (1.0 * s, 0.8 * s, 0.25 * s))
        R = rms_radius(X, A)
        f = -2.0 * np.exp(-(X[:, 0] ** 2 + X[:, 1] ** 2) / (0.3 * s * s)) * (X[:, 2] > 0) + 0.3 * X[:, 0] / s
        w = A / A.sum()
        spread = math.sqrt(float(w @ (f - w @ f) ** 2))
        kappa = shape_operator_norm(X, N, res_frac * R) * R
        ext = np.abs(f - weighted_quantile(f, A, [0.5])[0]) / spread
        ok = np.isfinite(kappa)
        kc = np.digitize(kappa, weighted_quantile(kappa[ok], A[ok], [1 / 3, 2 / 3]))
        xc = np.digitize(ext, weighted_quantile(ext[ok], A[ok], [1 / 3, 2 / 3]))
        eb = rng.normal(0, 0.2 * spread, npts)
        fac = np.ones(npts)
        if mode == 'local':
            fac[ok & (kc == 2)] = 0.5
        elif mode == 'global':
            fac[ok & (kc == 0) & (xc == 2)] = 0.5
        else:
            fac[:] = 0.8
        ea = eb * fac + (rng.normal(0, 0.02 * spread, npts) if mode == 'null' else 0.0)
        region = np.where(kc == 2, 0, np.where((kc == 0) & (xc == 2), 1, 2))
        np.savez(os.path.join(folder, f'case{i:02d}.npz'), X=X, N=N, A=A, f=f, pred_finetuned=f + ea,
                 pred_scratch=f + eb, R=R, group=f'g{i // 2}', region=region)


def _selftest_attention_attribution():
    rng = np.random.default_rng(7)
    print('shape operator from the projector (Lemma 1)')
    u = fibonacci_sphere(3000)
    k_sph = shape_operator_norm(2 * u, u, 0.3)
    _check('sphere of radius 2: |W| = sqrt(2)/2', abs(np.nanmedian(k_sph) / (math.sqrt(2) / 2) - 1) < 0.03,
           f'{np.nanmedian(k_sph):.4f}')
    flips = np.where(rng.uniform(size=len(u)) < 0.5, -1.0, 1.0)[:, None]
    _check('no orientation needed: random normal flips change nothing',
           np.nanmax(np.abs(shape_operator_norm(2 * u, u * flips, 0.3) - k_sph)) < 1e-9)
    th, zc = rng.uniform(0, 2 * np.pi, 4000), rng.uniform(-2, 2, 4000)
    k_cyl = shape_operator_norm(np.stack([2 * np.cos(th), 2 * np.sin(th), zc], 1),
                                np.stack([np.cos(th), np.sin(th), 0 * th], 1), 0.3)
    _check('cylinder of radius 2: |W| = 1/2', abs(np.nanmedian(k_cyl[np.abs(zc) < 1.6]) / 0.5 - 1) < 0.03,
           f'{np.nanmedian(k_cyl[np.abs(zc) < 1.6]):.4f}')
    xy = rng.uniform(-1, 1, (2000, 2))
    k_pl = shape_operator_norm(np.column_stack([xy, np.zeros(2000)]), np.tile([0.0, 0.0, 1.0], (2000, 1)), 0.15)
    _check('plane: |W| = 0', np.nanmax(k_pl) < 1e-9)

    print('attention range')
    tmp = tempfile.mkdtemp()
    try:
        _synthetic_attention(os.path.join(tmp, 'tails'), rng)
        d_raw, _ = load_attention(os.path.join(tmp, 'tails'), ['pretrained'])
        z = np.load(os.path.join(tmp, 'tails', 'pretrained', 'shape00.npz'))
        again = attention_summaries(z['Q'], z['anchors'], z['tw'], z['A'], float(z['R']))
        _check('raw attention and stored summaries give the same profile',
               np.allclose(again['bin_mass'], d_raw['pretrained']['shape00']['bin_mass']))
        r = attention(os.path.join(tmp, 'tails'), n_boot=500, verbose=False)
        c = list(r['criteria'].values())
        _check('short-range pretrained attention passes A1', c[0] is True,
               f"ratio {r['per_checkpoint']['pretrained']['ratio']['mean']:.3f}")
        _check('fine-tuning that adds long-range mass passes A2', c[1] is True,
               f"{r['contrasts']['finetuned - pretrained']['mean']:+.4f}")
        _synthetic_attention(os.path.join(tmp, 'rev'), rng, mode='reverse')
        r = attention(os.path.join(tmp, 'rev'), n_boot=500, verbose=False)
        _check('fine-tuning that removes long-range mass fails A2', list(r['criteria'].values())[1] is False)
        passes = 0
        for rep in range(20):
            folder = os.path.join(tmp, f'null{rep}')
            _synthetic_attention(folder, rng, mode='null')
            passes += list(attention(folder, n_boot=400, verbose=False)['criteria'].values())[1] is True
        _check('A2 rarely passes when fine-tuning changes nothing', passes <= 3, f'{passes} of 20 null replicates')

        print('attribution of the gains')
        for mode, want in (('local', True), ('global', False)):
            folder = os.path.join(tmp, f'attr_{mode}')
            os.makedirs(folder)
            _synthetic_attribution(folder, rng, mode=mode)
            r = attribution(folder, res_frac=0.2, j_region=0, b_region=1, n_boot=500, verbose=False)
            crit = list(r['criteria'].values())
            _check(f"gains placed on {'curved points' if mode == 'local' else 'flat extreme points'} give R2 "
                   f"{want}", crit[0] is True and crit[1] is want,
                   f"local {r['local']['gain']:+.3f}, global {r['global']['gain']:+.3f}")
            if mode == 'local':
                rc = r['region_contrast']
                _check('region labels reproduce the contrast', rc['ci95'][0] > 0, f"{rc['difference']:+.3f}")
        passes = 0
        for rep in range(30):
            folder = os.path.join(tmp, f'attr_null{rep}')
            os.makedirs(folder)
            _synthetic_attribution(folder, rng, n_cases=16, mode='null', npts=800, res_frac=0.25)
            passes += list(attribution(folder, res_frac=0.25, verbose=False)['criteria'].values())[1] is True
        _check('R2 rarely passes when the gain is uniform', passes <= 3, f'{passes} of 30 null replicates')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def pair_distance_selftest(Xa, Pa, Xb, Pb, scale=1.0, lam=0.5):
    da = surface_match(Xa, Pa, Xb, Pb, scale, lam)[0]
    db = surface_match(Xb, Pb, Xa, Pa, scale, lam)[0]
    return float(max(da.max(), db.max()))


# ======================================================================================================================
# Command line

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('prop1', help='Proposition 1')
    p.add_argument('--data', required=True)
    p.add_argument('--ckpts', nargs='+', required=True, help='checkpoint folder names under DATA/ckpt')
    p.add_argument('--order', nargs='*', help='checkpoints in increasing token perturbation (at least 4 levels)')
    p.add_argument('--within', nargs='*', help='checkpoints for the across-shapes test P3 (default: all whose name '
                                               'lacks "_noise"; leave out random initialisation)')
    p.add_argument('--points', type=int, default=4096, help='evaluation points per shape (0 = all)')
    p.add_argument('--lam', type=float, default=0.5, help='weight of the normal-line term in d')
    p.add_argument('--eps', type=float, default=0.01, help='rebuilt where predicted udf < eps * R')
    p.add_argument('--neighbours', type=int, default=3)
    p.add_argument('--max-shapes', type=int, default=400, help='shapes used to estimate L')
    p.add_argument('--sym-order', default=DEFAULT_ORDER)

    p = sub.add_parser('prior', help='Level 1 evidence and data efficiency on frozen features')
    p.add_argument('--data', required=True)
    p.add_argument('--ckpts', nargs='+', required=True)
    p.add_argument('--reference', help='checkpoint to compare against (default: the last one listed)')
    p.add_argument('--points', type=int, default=2048)
    p.add_argument('--ns', type=int, nargs='*', default=[], help='training-set sizes for the frozen-feature curve')
    p.add_argument('--repeats', type=int, default=5)

    p = sub.add_parser('datacurve', help='Level 1 data efficiency from full fine-tuning runs')
    p.add_argument('--csv', required=True)
    p.add_argument('--pretrained-label', default='pretrained')
    p.add_argument('--scratch-label', default='scratch')

    p = sub.add_parser('condition', help='Level 2 explicit condition posterior')
    p.add_argument('--test', required=True)
    p.add_argument('--val')
    p.add_argument('--sigma2', type=float, help='fixed sensor-noise variance for every sensor count')
    p.add_argument('--gap-margin', type=float, default=0.01)

    p = sub.add_parser('normals', help='normal-line head reports its own error')
    p.add_argument('--dir', required=True)
    p.add_argument('--sym-order', default=DEFAULT_ORDER)

    p = sub.add_parser('equivalence', help='single-condition equivalence of sensor and shape-only models')
    p.add_argument('--csv', required=True)
    p.add_argument('--a', default='measurements')
    p.add_argument('--b', default='geometry')
    p.add_argument('--case-col', default='car')
    p.add_argument('--group-col', help='bootstrap over this column (e.g. the physical geometry of each case)')
    p.add_argument('--delta', type=float, default=0.005)
    p.add_argument('--baseline-setting', help='compare every setting of --a with this setting of --b')

    p = sub.add_parser('attention', help='attention range before and after fine-tuning')
    p.add_argument('--dir', required=True, help='folder with one subfolder of shape files per checkpoint')
    p.add_argument('--pre', default='pretrained', help='checkpoint before fine-tuning')
    p.add_argument('--post', default='finetuned', help='the same model after fine-tuning')
    p.add_argument('--scratch', help='optional: the same architecture trained from random initialisation')
    p.add_argument('--rho', type=float, default=0.5, help='long-range threshold in units of R (a bin edge)')

    p = sub.add_parser('attribution', help='where pretraining gains: curved, attached regions or global features')
    p.add_argument('--dir', required=True, help='folder of per-case files')
    p.add_argument('--a', default='finetuned', help='prediction key pred_<a>: the pretrained model')
    p.add_argument('--b', default='scratch', help='prediction key pred_<b>: the model trained from scratch')
    p.add_argument('--res', type=float, default=0.1, help='curvature resolution as a fraction of R')
    p.add_argument('--local-region', type=int, help='region label where local geometry should dominate')
    p.add_argument('--global-region', type=int, help='region label where the field is set globally')

    sub.add_parser('selftest', help='synthetic checks with known answers')
    for name, sp in sub.choices.items():
        if name != 'selftest':
            sp.add_argument('--out', help='write results to this JSON file')
            sp.add_argument('--boot', type=int, default=2000, help='bootstrap resamples')
            sp.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.cmd == 'selftest':
        selftest()
        return
    if args.cmd == 'prop1':
        res = prop1(args.data, args.ckpts, args.order, args.points, args.lam, args.eps, args.neighbours,
                    args.max_shapes, args.boot, args.seed, args.sym_order, args.within)
    elif args.cmd == 'prior':
        res = prior(args.data, args.ckpts, args.reference, args.points, args.ns, args.repeats, args.boot, args.seed)
    elif args.cmd == 'datacurve':
        res = datacurve(args.csv, args.pretrained_label, args.scratch_label, args.boot, args.seed)
    elif args.cmd == 'condition':
        res = condition(args.test, args.val, args.sigma2, args.gap_margin, args.boot, args.seed)
    elif args.cmd == 'normals':
        res = normals(args.dir, args.sym_order, args.boot, args.seed)
    elif args.cmd == 'attention':
        res = attention(args.dir, args.pre, args.post, args.scratch, args.rho, n_boot=args.boot, seed=args.seed)
    elif args.cmd == 'attribution':
        res = attribution(args.dir, args.a, args.b, args.res, args.local_region, args.global_region, args.boot,
                          args.seed)
    else:
        res = equivalence(args.csv, args.a, args.b, args.case_col, args.group_col, args.delta, args.baseline_setting,
                          max(args.boot, 2000), args.seed)
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(res, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
