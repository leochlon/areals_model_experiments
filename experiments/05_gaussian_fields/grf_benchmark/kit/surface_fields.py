"""Random fields on surfaces with exact posterior means: a known-answer test of inverse solving.

A Whittle-Matern field on a surface S is f = sum_k xi_k sqrt(s_k) phi_k, where (lambda_k, phi_k) are the
Laplace-Beltrami eigenpairs of S (orthonormal in the surface measure), s_k = (2 nu / ell^2 + lambda_k)^(-nu - 1)
(surface dimension d = 2) and xi_k ~ N(0, 1). The spectrum is rescaled so the field has unit variance on average
over the surface. The covariance is intrinsic: it follows the surface, not the ambient space, and it is set by
the geometry alone, so these fields need no simulation.

Observations are y = f(V) + eps with eps ~ N(0, sigma^2). With a finite hyperprior over the length scale ell, the
Bayes estimator under squared error, E[f | y, S], is a closed-form mixture of kriging means, one per ell, weighted
by each ell's marginal likelihood. Everything is computed in the k-dimensional coefficient space, so a surface with
tens of thousands of points costs a k x k solve per length scale.

The excess risk of any estimator g over the Bayes estimator is E ||g - E[f|y,S]||^2 (the cross term vanishes), so
the "Bayes gap" E||g - E[f|y,S]||^2 / E||f - E[f|y,S]||^2 measures how far g is from the optimal inverse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree


# ------------------------------------------------------------------------------------------------- operators
def cotan_laplacian(vertices: np.ndarray, faces: np.ndarray) -> tuple[sp.csr_matrix, sp.dia_matrix]:
    """Cotangent stiffness L (positive semi-definite) and lumped mass M of a triangle mesh: L phi = lambda M phi."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = len(v)
    rows, cols, vals = [], [], []
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        i, j, k = f[:, a], f[:, b], f[:, c]
        e1, e2 = v[i] - v[k], v[j] - v[k]
        cross = np.linalg.norm(np.cross(e1, e2), axis=1)
        cot = (e1 * e2).sum(1) / np.maximum(cross, 1e-12)
        w = 0.5 * cot
        rows += [i, j, i, j]
        cols += [j, i, i, j]
        vals += [-w, -w, w, w]
    L = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    area = 0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    mass = np.bincount(f.ravel(), weights=np.repeat(area / 3.0, 3), minlength=n)
    return L, sp.diags(mass)


def point_cloud_laplacian(points: np.ndarray, weights: np.ndarray, neighbours: int = 30,
                          bandwidth_neighbour: int = 6) -> tuple[sp.csr_matrix, sp.dia_matrix]:
    """Laplacian of a sampled surface without faces (for example a field grid). Uses robust_laplacian (Sharp and
    Crane, 2020) when it is installed; otherwise an area-weighted heat-kernel graph on k nearest neighbours, scaled so
    that L phi = lambda M phi approximates the Laplace-Beltrami spectrum (within 2% of l(l+1) on the unit sphere)."""
    try:
        import robust_laplacian  # pip install robust-laplacian

        L, M = robust_laplacian.point_cloud_laplacian(np.asarray(points, dtype=np.float64))
        return sp.csr_matrix(L), sp.diags(np.asarray(M.diagonal()))
    except ImportError:
        pass
    x = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    dist, idx = cKDTree(x).query(x, neighbours + 1)
    h2 = np.median(dist[:, bandwidth_neighbour]) ** 2
    rows = np.repeat(np.arange(len(x)), neighbours)
    kernel = np.exp(-dist[:, 1:].ravel() ** 2 / h2)
    K = sp.coo_matrix((kernel, (rows, idx[:, 1:].ravel())), shape=(len(x), len(x))).tocsr()
    K = 0.5 * (K + K.T)
    K = sp.diags(w) @ K @ sp.diags(w)          # integrate against the surface measure
    L = sp.diags(np.asarray(K.sum(1)).ravel()) - K
    # for a Gaussian kernel exp(-|u|^2/h^2) on a 2-manifold, sum_j w_j K_ij (f_i - f_j) ~ (pi h^4 / 4) (-Delta f)
    return (L * (4.0 / (np.pi * h2 * h2))).tocsr(), sp.diags(w)


def eigenpairs(L: sp.spmatrix, M: sp.spmatrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The k smallest generalised eigenpairs L phi = lambda M phi, with phi orthonormal in M."""
    k = min(k, L.shape[0] - 2)
    lam, phi = eigsh(L.tocsc(), k=k, M=M.tocsc(), sigma=-1e-6, which="LM")
    order = np.argsort(lam)
    lam, phi = np.clip(lam[order], 0.0, None), phi[:, order]
    norms = np.sqrt(np.einsum("nk,n,nk->k", phi, np.asarray(M.diagonal()), phi))
    return lam, phi / norms


# ------------------------------------------------------------------------------------------------- the prior
@dataclass
class SurfaceGRF:
    """Whittle-Matern fields on one surface, with a uniform hyperprior over a few length scales."""

    lam: np.ndarray            # [k] Laplace-Beltrami eigenvalues
    phi: np.ndarray            # [N, k] eigenfunctions at the N surface points, orthonormal in `mass`
    mass: np.ndarray           # [N] area of each point
    ells: tuple = (0.25, 0.5, 1.0)   # length scales, in units of the surface's RMS radius
    nu: float = 1.5
    sigma: float = 0.05        # observation noise, in units of the field's standard deviation
    scale: float = 1.0         # the surface's RMS radius, so ell is relative to the shape's size

    def __post_init__(self):
        area = float(self.mass.sum())
        spectra = []
        for ell in self.ells:
            s = (2.0 * self.nu / (ell * self.scale) ** 2 + self.lam) ** (-self.nu - 1.0)
            spectra.append(s * area / s.sum())        # unit variance on average over the surface
        self.spectra = np.stack(spectra)               # [J, k]

    def sample(self, rng: np.random.Generator, count: int = 1) -> tuple[np.ndarray, np.ndarray]:
        which = rng.integers(0, len(self.ells), count)
        xi = rng.standard_normal((count, self.phi.shape[1]))
        coefficients = xi * np.sqrt(self.spectra[which])
        return coefficients @ self.phi.T, which

    def posterior_mean(self, observed: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """E[f | y] at every point, and the posterior weights of the length scales. `observed` indexes the points."""
        phi_v = self.phi[observed]                     # [n, k]
        s2 = self.sigma ** 2
        gram = phi_v.T @ phi_v / s2                    # [k, k]
        rhs = phi_v.T @ y / s2                          # [k]
        means, logml = [], []
        n = len(observed)
        for s in self.spectra:
            precision = np.diag(1.0 / s) + gram
            chol = np.linalg.cholesky(precision)
            m = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
            quad = (y @ y - y @ (phi_v @ m)) / s2
            logdet = n * np.log(s2) + np.log(s).sum() + 2.0 * np.log(np.diag(chol)).sum()
            logml.append(-0.5 * (quad + logdet + n * np.log(2 * np.pi)))
            means.append(self.phi @ m)
        logml = np.array(logml)
        weights = np.exp(logml - logml.max())
        weights /= weights.sum()
        return np.tensordot(weights, np.stack(means), axes=1), weights


# ------------------------------------------------------------------------------------------------- scoring
def bayes_gap(prediction: np.ndarray, truth: np.ndarray, posterior: np.ndarray, weights: np.ndarray,
              hidden: np.ndarray) -> dict:
    """Area-weighted squared errors over the hidden points: the estimator's risk, the Bayes risk and the excess."""
    w = weights[hidden] / weights[hidden].sum()
    risk = float(w @ (prediction[hidden] - truth[hidden]) ** 2)
    bayes = float(w @ (posterior[hidden] - truth[hidden]) ** 2)
    excess = float(w @ (prediction[hidden] - posterior[hidden]) ** 2)
    prior = float(w @ truth[hidden] ** 2)          # risk of the prior mean, which is zero for these fields
    return dict(risk=risk, bayes=bayes, excess=excess, prior=prior)


def self_test(seed: int = 0) -> None:
    """Checks on a sphere: the Laplacian's spectrum, unit variance, and that the posterior mean beats perturbations."""
    rng = np.random.default_rng(seed)
    from icosphere import icosphere  # local helper used by the pilot

    v, f = icosphere(3)
    L, M = cotan_laplacian(v, f)
    lam, phi = eigenpairs(L, M, 36)
    # sphere: eigenvalues l(l+1) with multiplicity 2l+1
    expected = np.concatenate([[l * (l + 1)] * (2 * l + 1) for l in range(6)])[:36]
    assert np.allclose(lam[:16], expected[:16], rtol=0.05, atol=0.05), (lam[:16], expected[:16])
    grf = SurfaceGRF(lam, phi, np.asarray(M.diagonal()), scale=1.0)
    fields, _ = grf.sample(rng, 4000)
    var = (fields ** 2).mean(0) @ grf.mass / grf.mass.sum()
    assert abs(var - 1.0) < 0.08, var
    # the posterior mean must have lower risk than a shrunk or inflated copy of itself
    risks = np.zeros(3)
    for t in range(200):
        f_true, _ = grf.sample(rng, 1)
        f_true = f_true[0]
        observed = rng.choice(len(v), 40, replace=False)
        y = f_true[observed] + grf.sigma * rng.standard_normal(40)
        post, _ = grf.posterior_mean(observed, y)
        hidden = np.ones(len(v), bool)
        hidden[observed] = False
        for i, c in enumerate((1.0, 0.8, 1.2)):
            risks[i] += bayes_gap(c * post, f_true, post, grf.mass, hidden)["risk"]
    assert risks[0] < risks[1] and risks[0] < risks[2], risks
    print("surface_fields self-test passed: sphere spectrum", np.round(lam[:9], 2), "| mean variance %.3f" % var,
          "| risk of posterior mean vs shrunk/inflated", np.round(risks / 200, 4))


if __name__ == "__main__":
    self_test()
