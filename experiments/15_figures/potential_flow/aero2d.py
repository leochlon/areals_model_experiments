"""2D car aerodynamics and geometry shared by the AREALS step GIFs.

The car is the fastback from the earlier explainer GIF: a polygon rounded
by Chaikin corner cutting, then fitted with a periodic cubic spline so that
tangents, normals and curvature are smooth functions of arc length.

Flow: constant-strength source panels with a ground plane by the method of
images (the solver ground sits below the drawn road, as before), free
stream (1, 0). Everything here is potential flow except `sep_field`, the
illustrative separation model used for the sensor step.
"""
import numpy as np
from matplotlib.path import Path
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

FRONT_TOP = [(0.00, 0.40), (0.00, 0.60), (0.07, 0.74), (0.55, 0.85),
             (1.45, 0.97), (2.12, 1.37), (2.38, 1.43), (2.90, 1.45)]
REAR = [(3.25, 1.41), (3.75, 1.21), (4.20, 1.01), (4.44, 0.92),
        (4.53, 0.78), (4.53, 0.40)]
UNDER = [(4.45, 0.26), (4.10, 0.20), (0.60, 0.20), (0.20, 0.24),
         (0.05, 0.32)]
WHEELS = [(0.98, 0.31, 0.31), (3.62, 0.31, 0.31)]    # (x, y, radius)
Y_GROUND = -0.5                                       # solver ground


def _densify(P, step=0.12):
    out = []
    for a, b in zip(P, np.roll(P, -1, axis=0)):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / step)))
        out.extend(a + (b - a) * k / n for k in range(n))
    return np.array(out)


def _chaikin(P, iterations=5):
    for _ in range(iterations):
        Q = np.roll(P, -1, axis=0)
        P = np.empty((2 * len(P), 2))
        P[0::2] = 0.75 * np.roll(Q, 1, axis=0) + 0.25 * Q
        P[1::2] = 0.25 * np.roll(Q, 1, axis=0) + 0.75 * Q
    return P


# --------------------------------------------------------------- outline
class Outline:
    """Smooth closed clockwise curve parameterised by arc length s.

    Outward normal = left normal of the clockwise tangent. Curvature kappa
    is positive where the body is convex (dn/ds = kappa t)."""

    def __init__(self, P=None, smooth=0.05, start=None):
        if P is None:
            P = _chaikin(_densify(np.array(FRONT_TOP + REAR + UNDER, float)))
        tck, _ = splprep([P[:, 0], P[:, 1]], s=smooth, per=True, k=3)
        u = np.linspace(0, 1, 20001)
        x, y = splev(u, tck)
        s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        self.tck, self.length = tck, s[-1]
        if start is None:                           # the nose
            i0 = int(np.argmin(x[:-1]))
        else:
            i0 = int(np.argmin(np.hypot(x[:-1] - start[0],
                                        y[:-1] - start[1])))
        self._u0 = u[i0]
        self._s_tab = (s - s[i0]) % self.length
        # monotone table from shifted arc length back to spline parameter
        order = np.argsort(self._s_tab[:-1])
        ss = self._s_tab[:-1][order]
        uu = np.unwrap(u[:-1][order] * 2 * np.pi) / (2 * np.pi)
        # pad one period on each side so interpolation never crosses the seam
        self._s_sorted = np.concatenate([[ss[-1] - self.length], ss,
                                         [ss[0] + self.length]])
        self._u_sorted = np.concatenate([[uu[-1] - 1.0], uu, [uu[0] + 1.0]])

    def _u(self, s):
        s = np.mod(np.asarray(s, float), self.length)
        return np.mod(np.interp(s, self._s_sorted, self._u_sorted), 1.0)

    def point(self, s):
        x, y = splev(self._u(s), self.tck)
        return np.column_stack([x, y])

    def frame(self, s):
        """Points, unit tangents, outward normals and curvature at s."""
        u = self._u(s)
        x, y = splev(u, self.tck)
        dx, dy = splev(u, self.tck, der=1)
        ddx, ddy = splev(u, self.tck, der=2)
        sp = np.hypot(dx, dy)
        t = np.column_stack([dx / sp, dy / sp])
        n = np.column_stack([-t[:, 1], t[:, 0]])
        kappa = -(dx * ddy - dy * ddx) / sp ** 3
        return np.column_stack([x, y]), t, n, kappa

    def nodes(self, n):
        s = np.linspace(0, self.length, n + 1)
        P = self.point(s[:-1])
        return np.vstack([P, P[:1]])


# ------------------------------------------------------ source panels
def _panel_velocity(px, py, x0, y0, x1, y1):
    """Velocity at points induced by unit-strength source panels (P, N)."""
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    tx, ty = dx / L, dy / L
    ex, ey = -ty, tx
    rx = px[:, None] - x0[None, :]
    ry = py[:, None] - y0[None, :]
    xl = rx * tx + ry * ty
    yl = rx * ex + ry * ey
    r1 = np.hypot(xl, yl)
    r2 = np.hypot(xl - L, yl)
    ul = np.log(np.maximum(r1, 1e-12) / np.maximum(r2, 1e-12)) / (2 * np.pi)
    vl = (np.arctan2(yl, xl - L) - np.arctan2(yl, xl)) / (2 * np.pi)
    return ul * tx + vl * ex, ul * ty + vl * ey


def _panel_potential(px, py, x0, y0, x1, y1):
    """Potential (1/2pi) int ln r ds of unit-strength source panels (P, N)."""
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    tx, ty = dx / L, dy / L
    ex, ey = -ty, tx
    rx = px[:, None] - x0[None, :]
    ry = py[:, None] - y0[None, :]
    x = rx * tx + ry * ty
    y = rx * ex + ry * ey
    r1 = np.maximum(np.hypot(x, y), 1e-14)
    r2 = np.maximum(np.hypot(x - L, y), 1e-14)
    with np.errstate(divide="ignore", invalid="ignore"):
        ang = np.where(np.abs(y) > 1e-14,
                       y * (np.arctan((L - x) / y) + np.arctan(x / y)), 0.0)
    return ((L - x) * np.log(r2) + x * np.log(r1) - L + ang) / (2 * np.pi)


class Body:
    """Source-panel potential flow around a closed clockwise polyline."""

    def __init__(self, nodes, y_ground=Y_GROUND):
        self.nodes = np.asarray(nodes, float)
        self.y_ground = y_ground
        x0, y0 = self.nodes[:-1, 0], self.nodes[:-1, 1]
        x1, y1 = self.nodes[1:, 0], self.nodes[1:, 1]
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        d = np.column_stack([x1 - x0, y1 - y0])
        self.L = np.hypot(d[:, 0], d[:, 1])
        self.t = d / self.L[:, None]
        self.n = np.column_stack([-self.t[:, 1], self.t[:, 0]])
        self.mid = 0.5 * (self.nodes[:-1] + self.nodes[1:])
        self.s = np.concatenate([[0], np.cumsum(self.L)])
        self.s_mid = 0.5 * (self.s[:-1] + self.s[1:])
        self.perimeter = self.s[-1]
        self._solve()

    def _images(self):
        g2 = 2 * self.y_ground
        return self.x0, g2 - self.y0, self.x1, g2 - self.y1

    def induced(self, px, py):
        u, v = _panel_velocity(px, py, self.x0, self.y0, self.x1, self.y1)
        if self.y_ground is None:
            return u, v
        ui, vi = _panel_velocity(px, py, *self._images())
        return u + ui, v + vi

    def _solve(self):
        eps = 1e-7
        px = self.mid[:, 0] + eps * self.n[:, 0]
        py = self.mid[:, 1] + eps * self.n[:, 1]
        u, v = self.induced(px, py)
        A = u * self.n[:, 0:1] + v * self.n[:, 1:2]
        self.sigma = np.linalg.solve(A, -self.n[:, 0])
        ut = 1.0 + u @ self.sigma
        vt = v @ self.sigma
        self.Vt = ut * self.t[:, 0] + vt * self.t[:, 1]
        self.cp = 1.0 - self.Vt ** 2

    def velocity(self, px, py):
        px, py = np.atleast_1d(px).astype(float), np.atleast_1d(py).astype(float)
        u, v = self.induced(px, py)
        return 1.0 + u @ self.sigma, v @ self.sigma

    def phi_pert(self, px, py):
        """Perturbation potential of the source sheet (plus image)."""
        px, py = np.atleast_1d(px).astype(float), np.atleast_1d(py).astype(float)
        G = _panel_potential(px, py, self.x0, self.y0, self.x1, self.y1)
        if self.y_ground is not None:
            G = G + _panel_potential(px, py, *self._images())
        return G @ self.sigma

    def inside(self, px, py):
        return Path(self.nodes).contains_points(np.column_stack([px, py]))


def outline_body(outline, n=320, h=None, y_ground=Y_GROUND):
    """Body on the outline, optionally displaced by h(s) along the normal."""
    s = np.linspace(0, outline.length, n + 1)[:-1]
    P, t, nrm, k = outline.frame(s)
    if h is not None:
        P = P + h(s)[:, None] * nrm
    return Body(np.vstack([P, P[:1]]), y_ground=y_ground)


# ------------------------------------------------ projection / distances
def project(points, nodes, chunk=4000):
    """Closest points on a closed polyline: distance, foot, segment, frac."""
    pts = np.atleast_2d(points).astype(float)
    a, b = nodes[:-1], nodes[1:]
    ab = b - a
    ab2 = (ab ** 2).sum(1)
    out_d = np.empty(len(pts))
    out_f = np.empty((len(pts), 2))
    out_i = np.empty(len(pts), int)
    out_t = np.empty(len(pts))
    for k in range(0, len(pts), chunk):
        p = pts[k:k + chunk]
        ap = p[:, None, :] - a[None, :, :]
        t = np.clip((ap * ab[None]).sum(2) / ab2[None], 0, 1)
        foot = a[None] + t[..., None] * ab[None]
        d2 = ((p[:, None, :] - foot) ** 2).sum(2)
        j = np.argmin(d2, axis=1)
        r = np.arange(len(p))
        out_d[k:k + chunk] = np.sqrt(d2[r, j])
        out_f[k:k + chunk] = foot[r, j]
        out_i[k:k + chunk] = j
        out_t[k:k + chunk] = t[r, j]
    return out_d, out_f, out_i, out_t


def map_back(values_mid, body_from, points):
    """Interpolate panel-midpoint values of body_from at the closest points
    of `points` on body_from (linear between neighbouring midpoints)."""
    d, foot, j, t = project(points, body_from.nodes)
    vm = np.asarray(values_mid)
    jn = (j + 1) % len(vm)
    jp = (j - 1) % len(vm)
    w = t - 0.5                       # signed offset from the midpoint
    return np.where(w < 0, vm[j] + w * (vm[j] - vm[jp]),
                    vm[j] + w * (vm[jn] - vm[j]))


def hausdorff(A, B):
    ta, tb = cKDTree(A), cKDTree(B)
    return max(tb.query(A)[0].max(), ta.query(B)[0].max())


def rms_radius(nodes):
    P = nodes[:-1]
    L = np.hypot(*np.diff(nodes, axis=0).T)
    c = (P * L[:, None]).sum(0) / L.sum()
    return float(np.sqrt((((P - c) ** 2).sum(1) * L).sum() / L.sum()))


def embed(P, n, R):
    """(x / R, n n^T) as 2 + 4 numbers per point for product-metric Hausdorff."""
    nn = np.einsum("ki,kj->kij", n, n).reshape(len(n), 4)
    return np.hstack([P / R, nn])


# ------------------------------------------ separation model (illustrative)
SEP_RANGE = (3.30, 4.42)
X_SEP_LOWER = 4.40
CP_B_REF = -0.27


def _smoothstep(z):
    z = np.clip(z, 0, 1)
    return z * z * (3 - 2 * z)


def sep_field(body, theta):
    """Surface pressure with an illustrative separated rear: theta = (q,
    cp_b): q in [0, 1] places the upper separation point in SEP_RANGE,
    cp_b is the base pressure behind it."""
    q, cp_b = theta
    mid, s = body.mid, body.s_mid
    i_top = int(np.argmax(mid[:, 1]))
    i_tail = int(np.argmax(mid[:, 0]))
    idx = np.arange(i_top, i_tail + 1)
    x_up = SEP_RANGE[0] + q * (SEP_RANGE[1] - SEP_RANGE[0])
    j_up = int(idx[np.searchsorted(mid[idx, 0], x_up).clip(0, len(idx) - 1)])
    lo = np.arange(i_tail, len(mid))
    order = np.argsort(-mid[lo, 0])
    xs = mid[lo, 0][order]
    k = np.searchsorted(-xs, -X_SEP_LOWER).clip(0, len(lo) - 1)
    j_lo = int(lo[order][k])
    s_up, s_lo = s[j_up], s[j_lo]
    kappa, lam_up, lam_lo = 0.9, 0.55, 0.7
    d_up = np.maximum(s_up - s, 0.0)
    d_lo = np.maximum(s - s_lo, 0.0)
    infl = kappa * np.exp(-d_up / lam_up) * np.exp(-d_lo / lam_lo)
    cp = body.cp + infl * (cp_b - CP_B_REF)
    blend = 0.16
    w = _smoothstep((s - s_up) / blend) * _smoothstep((s_lo - s) / blend)
    cp = (1 - w) * cp + w * cp_b
    floor, kk = -2.2, 5.0
    return floor + np.logaddexp(0.0, kk * (cp - floor)) / kk


def car():
    """The reference outline (arc length from the front stagnation point)
    and its body with 320 panels."""
    o = Outline()
    b = outline_body(o, 320)
    front = np.where(b.mid[:, 0] < 0.3)[0]
    i = front[np.argmax(b.cp[front])]
    o = Outline(start=b.mid[i])
    return o, outline_body(o, 320)
