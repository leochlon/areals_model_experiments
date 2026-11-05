"""Computed figures: Figure 2, Figure B.3, Figure G.1 and the Appendix B "Computed example" paragraph.

Rebuilt from the stored arrays of the potential-flow example in experiments/15_figures/potential_flow/
(copied with the code that wrote them):

  hero3d.npz   build3d.py stages mesh, geom, flow, eta, anchors and mask: the car mesh, the method-of-
               fundamental-solutions flow and its unit-sphere check, the centre-plane section with its
               curvature, the 80,000 area-uniform samples and the 256 farthest-point anchors
  omega.npy    build3d.stage_omega(d, n_cases=30): one row per perturbed shape,
               (field error dCp, distance in x/R, distance in (x/R, nn^T), wavelength, amplitude / R)
  samp_cp.npy  Cp interpolated to the 80,000 samples (barycentric, on the closest mesh face)

Figure B.3 and Figure G.1 have no stored number file: their numbers are computed when the figure is
drawn. This module repeats that computation exactly as hero_fig.py does it (panel_a for Figure B.3,
panel_c for Figure G.1, including the random draws with numpy's default_rng(0)), without plotting.

The schematic figures (B.1, B.2, E.1, I.1, K.1) and the pgfplots figures (1, D.1) carry no numbers
computed here. Their measured labels and plotted values are checked by the modules of the experiments
that measured them (see experiments/15_figures/README.md), except the "81.5% less drift" label of
Figures B.1 and K.1, which the second table checks against results/architecture_ablations/density_shift/.
"""
from __future__ import annotations

import numpy as np

from paper_tables._core import RESULTS, Table, fmt, load_json

ARRAYS = RESULTS.parent / "experiments" / "15_figures" / "potential_flow"


def _load():
    hero = dict(np.load(ARRAYS / "hero3d.npz"))
    omega = np.load(ARRAYS / "omega.npy")
    samp_cp = np.load(ARRAYS / "samp_cp.npy")
    return hero, omega, samp_cp


def _wall_stretch(d: dict) -> tuple[float, float]:
    """1 + t*kappa at t = 0.21 over the concave and convex parts round the windscreen (hero_fig.panel_a)."""
    S, Pp, K = d["sec_s"], d["sec_p"], d["sec_k"]
    xl = (0.95, 3.05)
    win = (Pp[:, 1] > 0.75) & (Pp[:, 0] > xl[0] - 0.2) & (Pp[:, 0] < xl[1] + 0.2)
    ids = np.where(win)[0]
    ids = ids[np.argsort(S[ids])]
    reg = ids[(Pp[ids, 0] > 1.25) & (Pp[ids, 0] < 2.7)]
    concave, convex = reg[np.argmin(K[reg])], reg[np.argmax(K[reg])]
    return 1 + 0.21 * K[concave], 1 + 0.21 * K[convex]


def _density_test(d: dict, v: np.ndarray):
    """Attention at a query above the roof under a front-half density shift (hero_fig.panel_c).

    Returns the surface integral (kernel average over all samples), the densities and E[density, draw, k]
    with k = 0 attention with the log-area (inverse-proposal) weight, k = 1 ordinary attention.
    """
    V, P = d["V"], d["samp_p"]
    q = np.array([2.2, 0.0, 1.75])
    kq = np.exp(-((P - q) ** 2).sum(1) / (2 * 0.6 ** 2))
    exact = (kq * v).sum() / kq.sum()
    x_mid = 0.5 * (V[:, 0].min() + V[:, 0].max())
    front = P[:, 0] < x_mid
    rhos = np.array([1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9])
    rng = np.random.default_rng(0)
    E = np.zeros((len(rhos), 32, 2))
    for a, rho in enumerate(rhos):
        p = np.where(front, rho, 1.0)
        p = p / p.sum()
        for b in range(32):
            idx = rng.choice(len(P), 4096, replace=True, p=p)
            w = 1.0 / p[idx]
            kk, vv = kq[idx], v[idx]
            E[a, b] = ((w * kk * vv).sum() / (w * kk).sum(), (kk * vv).sum() / kk.sum())
    return float(exact), rhos, E


def tables() -> list[Table]:
    d, omega, samp_cp = _load()
    dcp, d_pos, d_full, lam, amp = omega.T
    waves = np.unique(lam)[::-1]                      # longest to shortest
    per_wave = {w: lam == w for w in waves}
    counts = np.array([m.sum() for m in per_wave.values()])
    pos_max = np.array([np.max(dcp[m] / d_pos[m]) for m in per_wave.values()])
    full_max = np.array([np.max(dcp[m] / d_full[m]) for m in per_wave.values()])
    L = float(np.max(dcp / d_full))
    longest, shortest = waves[0], waves[-1]

    concave, convex = _wall_stretch(d)
    exact, rhos, E = _density_test(d, samp_cp)
    mean = E.mean(1)                                   # (density, k)
    dev = 100 * np.abs(mean - exact) / abs(exact)      # % deviation from the surface integral
    nine = int(np.argmax(rhos))

    t = Table(
        key="Figure numbers",
        title="Figures 2, B.3 and G.1 and the Appendix B computed example, rebuilt from the stored "
        "potential-flow arrays",
        columns=["Where printed", "Quantity", "Rebuilt", "Paper"],
    )

    def row(where: str, what: str, value: float, paper: float, decimals: int, shown: str | None = None):
        t.check(f"{where}: {what}", value, paper, decimals)
        t.rows.append([where, what, shown if shown is not None else fmt(value, decimals), fmt(paper, decimals)])

    # ---------------- Appendix B, computed example: mesh, flow, scale ----------------
    V = d["V"]
    row("App. B", "mesh vertices (thousands)", len(V) / 1000, 31.8, 1, f"{len(V) / 1000:.1f} ({len(V):,})")
    row("App. B", "car length (x extent)", np.ptp(V[:, 0]), 4.5, 1)
    row("App. B", "centre-plane section: largest deviation from the profile", float(d["section_dev"]), 0.016, 3)
    row("App. B", "MFS sources", len(d["mfs_src"]), 2300, 0)
    row("App. B", "MFS collocation points", len(d["mfs_col"]), 5200, 0)
    row("App. B", "unit sphere: largest Cp error", float(d["sphere_err"]), 0.037, 3)
    row("App. B", "R, area-weighted RMS radius", float(d["R"]), 1.51, 2)

    # ---------------- Appendix B / Figure G.1 sampling ----------------
    row("App. B; Fig. G.1 caption", "farthest-point anchors", len(d["anc_p"]), 256, 0)
    row("App. B", "area-uniform samples", len(d["samp_p"]), 80000, 0)

    # ---------------- Appendix B / Figure 2(c): the perturbed shapes ----------------
    row("App. B; Fig. 2 caption", "perturbed shapes, flow re-solved on each", len(omega), 30, 0)
    row("App. B", "shapes per wavelength (fewest)", counts.min(), 5, 0)
    row("App. B", "shapes per wavelength (most)", counts.max(), 5, 0)
    row("App. B", "shortest wavelength", shortest, 0.2, 1)
    row("App. B", "longest wavelength", longest, 3, 0)
    row("App. B", "smallest amplitude (units of R)", amp.min(), 0.003, 3)
    row("App. B", "largest amplitude (units of R)", amp.max(), 0.04, 2)
    row("App. B", "largest field error / distance with normal lines, all 30 shapes", L, 0.567, 3)
    row("App. B", f"largest field error / distance, position only, wavelength {longest:g}", pos_max[0], 1.76, 2)
    row("App. B", f"largest field error / distance, position only, wavelength {shortest:g}", pos_max[-1], 24.0, 1)
    row("Fig. 2 caption", "per unit distance, position only: longest waves", pos_max[0], 1.8, 1)
    row("Fig. 2 caption", "per unit distance, position only: shortest waves", pos_max[-1], 24, 0)
    row("Fig. 2 caption", "per unit distance with normal lines: lowest per-wavelength maximum", full_max.min(), 0.45, 2)
    row("Fig. 2 caption", "per unit distance with normal lines: highest per-wavelength maximum", full_max.max(), 0.57, 2)
    row("Fig. 2 caption", "omega(r) <= L r: slope L", L, 0.57, 2)

    # ---------------- Figure B.3: wall coordinates ----------------
    row("Fig. B.3 (panel and caption)", "1 + t kappa at t = 0.21, concave", concave, 0.81, 2)
    row("Fig. B.3 (panel and caption)", "1 + t kappa at t = 0.21, convex", convex, 1.26, 2)

    # ---------------- Figure G.1 / Appendix G: ninefold density shift ----------------
    row("App. G; Fig. G.1 caption", "ordinary attention: drift from the surface integral at 9x (%)",
        dev[nine, 1], 24, 0)
    row("App. G", "log-area attention: deviation from the surface integral at 9x (%)", dev[nine, 0], 0.7, 1)

    t.notes.append(
        "Sources: experiments/15_figures/potential_flow/{hero3d.npz, omega.npy, samp_cp.npy}. Ratios are dCp / d "
        "per perturbed shape (omega.npy columns 0-2); per-wavelength maxima are over the five shapes of each "
        "wavelength. Figure B.3 and Figure G.1 numbers repeat hero_fig.py's panel_a and panel_c computations.")
    t.notes.append(
        "Per-wavelength maxima, wavelength " + ", ".join(f"{w:g}" for w in waves) + ": position only "
        + ", ".join(f"{x:.2f}" for x in pos_max) + "; with normal lines "
        + ", ".join(f"{x:.3f}" for x in full_max) + ". The position-only maximum grows "
        + ("monotonically" if np.all(np.diff(pos_max) > 0) else "non-monotonically")
        + " as the waves shorten.")
    t.notes.append(
        f"Figure G.1: surface integral {exact:.4f}; at 9x, ordinary attention {mean[nine, 1]:.4f} and log-area "
        f"attention {mean[nine, 0]:.4f} (means of 32 draws). The log-area deviation at the intermediate densities "
        "is " + ", ".join(f"{r:g}x {x:.2f}%" for r, x in zip(rhos, dev[:, 0])) + "; the 0.7% in Appendix G is "
        "the value at the ninefold shift.")
    t.notes.append(
        "Settings of the computation, not results (set in build3d.py and hero_fig.py, so not checked here): "
        "t = 0.21 (Figure B.3); 4,096 samples, nine-fold density ratio, kernel length 0.6 and 32 draws "
        "(Figure G.1); free stream U = (1, 0, 0) (flow).")
    return [t, _schematic_labels()]


def _schematic_labels() -> Table:
    """The one measured label of Figures B.1 and K.1 that no other module checks at these locations.

    make_guarantees.py (Figure B.1) and make_longrange.py (Figure K.1, panel B) print "81.5% less drift", the
    density-shift result of Section 4.6 and Appendix G. The other measured labels of the schematics (52% and 3% in
    Figure B.2; 17% and 40% in Figure K.1) are checked per figure in attention_analysis.
    """
    src = "architecture_ablations/density_shift/independent_grouped.json"
    g = load_json(*src.split("/"))
    mode = "biased_fixed_anchors_frame"
    area = g["summaries"]["reference"][mode]["prediction_change"]["mean"]
    ordinary = g["summaries"]["ordinary_attention"][mode]["prediction_change"]["mean"]
    less = 100 * (1 - area / ordinary)
    t = Table(
        key="Figures B.1 and K.1",
        title="Measured label in the schematic figures: less drift with log-area attention under a ninefold "
        "density shift (%)",
        columns=["Where printed", "Rebuilt", "Paper", "Source (results/)"],
    )
    for where in ("Figure B.1 (sampling tile)", "Figure K.1(B)"):
        t.check(f"{where}: less drift with log-area attention (%)", less, 81.5, 1)
        t.rows.append([where, fmt(less, 1), fmt(81.5, 1), src])
    t.notes.append(f"1 - {area:.7f} / {ordinary:.7f} (log-area over ordinary attention drift, anchors and frame fixed); "
                   "the same value is checked at Section 4.6 in architecture_ablations.")
    return t
