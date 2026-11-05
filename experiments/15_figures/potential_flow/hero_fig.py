"""AREALS hero figure: steps 1-4 of the argument on one car mesh.

Top row: the mesh, one view per step. Bottom row: the centre-plane section
and two measurements. Every field is computed: non-lifting potential flow
on the mesh (method of fundamental solutions, sphere-validated), exact
distances, farthest-point anchors with area weights, and perturbed shapes
with the flow re-solved for the omega(r) panel.

usage: python3 hero_fig.py OUT_STEM   (reads hero3d.npz, samp_cp.npy, omega.npy)
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import TwoSlopeNorm, Normalize, to_rgba, LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from scipy.spatial import cKDTree

from render3d import View

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")
OTF_GLOBS = ["/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-*.otf",
             "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-*.otf",
             "/usr/local/texlive/*/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-*.otf",
             "/Library/TeX/Root/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-*.otf"]
if os.environ.get("TERMES_DIR"):                 # e.g. a MiKTeX or custom font folder
    OTF_GLOBS.insert(0, os.path.join(os.environ["TERMES_DIR"], "texgyretermes-*.otf"))


def otf_to_ttf(src, dst):
    """Convert a CFF OpenType font to TrueType outlines (fontTools' otf2ttf recipe).

    matplotlib embeds fonts for pdf.fonttype 42 as CIDFontType2, which must hold
    TrueType outlines; embedding the CFF .otf there gives a PDF that strict
    viewers and checkers reject, so the figure uses TrueType copies of Termes.
    """
    from fontTools.ttLib import TTFont, newTable
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    font = TTFont(src)
    order = font.getGlyphOrder()
    gs = font.getGlyphSet()
    glyphs = {}
    for name in order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyphs[name] = pen.glyph()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, glyphs
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    glyf.compile(font)
    hmtx = font["hmtx"]
    for name, g in glyf.glyphs.items():
        if hasattr(g, "xMin"):
            hmtx[name] = (hmtx[name][0], g.xMin)
    font["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for k in ("maxZones",):
        setattr(maxp, k, 1)
    for k in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
              "maxStackElements", "maxSizeOfInstructions"):
        setattr(maxp, k, 0)
    maxp.maxComponentElements = max(len(getattr(g, "components", []) or [])
                                    for g in glyf.glyphs.values())
    maxp.compile(font)
    post = font["post"]
    post.formatType, post.extraNames, post.mapping, post.glyphOrder = 2.0, [], {}, order
    font.sfntVersion = "\000\001\000\000"
    font.save(dst)


def termes_fonts():
    os.makedirs(FONT_DIR, exist_ok=True)
    ttfs = sorted(glob.glob(os.path.join(FONT_DIR, "texgyretermes-*.ttf")))
    if len(ttfs) < 4:
        otfs = sorted({f for g in OTF_GLOBS for f in glob.glob(g)})
        for f in otfs:
            dst = os.path.join(FONT_DIR, os.path.basename(f)[:-4] + ".ttf")
            if not os.path.exists(dst):
                otf_to_ttf(f, dst)
        ttfs = sorted(glob.glob(os.path.join(FONT_DIR, "texgyretermes-*.ttf")))
    if not ttfs:
        print("warning: TeX Gyre Termes not found; matplotlib will substitute a serif font")
        return
    fm = font_manager.fontManager
    # drop any system .otf copies so the TrueType ones are the only match
    fm.ttflist = [e for e in fm.ttflist if not (e.name == "TeX Gyre Termes"
                                                and e.fname.lower().endswith(".otf"))]
    for f in ttfs:
        fm.addfont(f)
    if hasattr(fm, "_findfont_cached"):
        fm._findfont_cached.cache_clear()


termes_fonts()
plt.rcParams.update({
    "font.family": "TeX Gyre Termes",
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#8A9099",
    "axes.labelcolor": "#2A2F38",
    "axes.labelsize": 6.5,
    "xtick.color": "#2A2F38", "ytick.color": "#2A2F38",
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "xtick.major.size": 2, "ytick.major.size": 2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "pdf.fonttype": 42,
})

C_SHAPE = "#E8740C"
C_SHAPE_D = "#9C4A06"
C_DATA = "#0E8C8E"
C_EST = "#7B2CBF"
C_TEXT = "#16181D"
C_SUB = "#555C68"
C_GREY = "#9AA1AC"
BODY = np.array(to_rgba("#DADDE2"))
CMAP_CP = plt.get_cmap("RdBu_r")
NORM_CP = TwoSlopeNorm(vcenter=0.0, vmin=-1.4, vmax=1.0)
CMAP_W = LinearSegmentedColormap.from_list("W", ["#E4E6EA", "#F4D7B8", "#E8740C", "#8A3F05"])
NORM_W = Normalize(0, 4.0)

W_IN, H_IN = 6.75, 2.98
VIEW = View(235, 15, light=(-0.2, -0.6, 0.78))

d = dict(np.load(os.path.join(HERE, "hero3d.npz")))
V, F, N = d["V"], d["F"], d["N"]
R = float(d["R"])
P2, _ = VIEW.project(V)

LEFT, GAP = 0.04, 0.14
CW = (W_IN - 2 * LEFT - 3 * GAP) / 4
TOP_Y, TOP_H = 1.88, 0.74           # mesh row
CAP_Y = 1.72                        # caption / colour-bar row
BOT_Y, BOT_H = 0.27, 1.32           # detail row


def col_x(k):
    return LEFT + k * (CW + GAP)


def inch_axes(fig, x, y, w, h):
    return fig.add_axes([x / W_IN, y / H_IN, w / W_IN, h / H_IN])


def mesh_axes(fig, k, extra_left=0.0):
    ax = inch_axes(fig, col_x(k), TOP_Y, CW, TOP_H)
    ax.set_aspect("equal")
    ax.axis("off")
    lo, hi = P2.min(0), P2.max(0)
    ax.set_xlim(lo[0] - 0.05 - extra_left, hi[0] + 0.05)
    ax.set_ylim(lo[1] - 0.05, hi[1] + 0.05)
    return ax


def draw_mesh(ax, face_rgba, **kw):
    pc, idx = VIEW.faces(V, F, face_rgba, **kw)
    pc.set_rasterized(True)
    ax.add_collection(pc)


def proj(P):
    return VIEW.project(np.atleast_2d(P))[0]


def facing(Nn, thresh):
    return (Nn @ VIEW.M.T)[:, 2] > thresh


def face_values(v):
    return v[F].mean(1)


def caption(fig, k, text, x_off=0.0, color=C_SUB, size=6.2):
    fig.text((col_x(k) + x_off) / W_IN, (CAP_Y + 0.01) / H_IN, text, fontsize=size,
             color=color, va="baseline")


def colorbar(fig, k, cmap, norm, ticks, width=0.5):
    cax = inch_axes(fig, col_x(k) + 0.04, CAP_Y + 0.06, width, 0.045)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                      orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(0.3)
    cax.tick_params(length=1.2, labelsize=5.5, pad=0.8)
    return cax


def section():
    return d["sec_s"], d["sec_p"], d["sec_t"], d["sec_n"], d["sec_k"], d["sec_cp"]


def upper_index(Pp, x):
    up = np.where(Pp[:, 1] > 0.75)[0]
    return up[np.argmin(np.abs(Pp[up, 0] - x))]


HEAD = [("a", "Physics", "derived, attached flow", "#0B4F3E"),
        ("b", r"Geometry from $\eta_S=\frac{1}{2}d_S^2$", "Lemma 1", "#0B4F3E"),
        ("c", "Anchors as a quadrature", "structural analogy", "#3C3489"),
        ("d", "Reconstruction bounds the loss", "Proposition 1", "#0B4F3E")]


def headers(fig):
    for k, (lab, title, tag, tcol) in enumerate(HEAD):
        x = col_x(k)
        fig.text(x / W_IN, (H_IN - 0.13) / H_IN, f"({lab})", fontsize=7.5,
                 fontweight="bold", color=C_TEXT, va="baseline")
        fig.text((x + 0.19) / W_IN, (H_IN - 0.13) / H_IN, title, fontsize=7.5,
                 fontweight="bold", color=C_TEXT, va="baseline")
        fig.text((x + 0.19) / W_IN, (H_IN - 0.255) / H_IN, tag, fontsize=6.3,
                 style="italic", color=tcol, va="baseline")


# ------------------------------------------------------------- (a) physics
def panel_a(fig):
    ax = mesh_axes(fig, 0, extra_left=0.0)
    draw_mesh(ax, CMAP_CP(NORM_CP(face_values(d["Cp"]))))
    dirn = proj([1.0, 0, 0])[0] - proj([0.0, 0, 0])[0]
    dirn /= np.linalg.norm(dirn)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    a0 = np.array([x0 + 0.06, y1 - 0.30])
    a1 = a0 + 0.62 * dirn
    ax.annotate("", xy=a1, xytext=a0, arrowprops=dict(arrowstyle="-|>", lw=0.8,
                color=C_TEXT, mutation_scale=6), annotation_clip=False)
    ax.text(a0[0], a0[1] + 0.16, r"$U_\infty$", fontsize=7, color=C_TEXT, va="bottom")
    S, Pp, T, Nn, K, cps = section()
    i = upper_index(Pp, 2.6)
    p3 = np.array([Pp[i, 0], 0.0, Pp[i, 1]])
    n3 = np.array([Nn[i, 0], 0.0, Nn[i, 1]])
    pp, xx = proj(p3)[0], proj(p3 + 0.42 * n3)[0]
    ax.plot([pp[0], xx[0]], [pp[1], xx[1]], color=C_TEXT, lw=0.7, zorder=8)
    ax.plot(*pp, "o", ms=3.0, mfc=C_SHAPE, mec="white", mew=0.5, zorder=9)
    ax.plot(*xx, "o", ms=2.4, color=C_TEXT, zorder=9)
    ax.text(xx[0] + 0.07, xx[1], r"$x=p+t\,n$", fontsize=6.2, va="center", color=C_TEXT)
    ax.text(pp[0] + 0.08, pp[1] - 0.03, "$p$", fontsize=6.5, va="top", color=C_SHAPE_D)
    colorbar(fig, 0, CMAP_CP, NORM_CP, [-1, 0, 1])
    caption(fig, 0, r"$C_p$, potential flow", x_off=0.6)

    # wall coordinates on the centre-plane section, around the windscreen
    ax2 = inch_axes(fig, col_x(0), BOT_Y, CW, BOT_H)
    ax2.set_aspect("equal")
    ax2.axis("off")
    xl, zl = (0.95, 3.05), (0.52, 2.26)
    ax2.set_xlim(*xl)
    ax2.set_ylim(*zl)
    win = (Pp[:, 1] > 0.75) & (Pp[:, 0] > xl[0] - 0.2) & (Pp[:, 0] < xl[1] + 0.2)
    ids = np.where(win)[0]
    ids = ids[np.argsort(S[ids])]
    seg = Pp[ids]
    ax2.add_collection(LineCollection(np.stack([seg[:-1], seg[1:]], 1),
                                      colors=CMAP_CP(NORM_CP(cps[ids][:-1])), linewidths=2.6,
                                      capstyle="round", zorder=5))
    s_up = S[ids]
    feet = ids[np.searchsorted(s_up, np.arange(s_up[0] + 0.02, s_up[-1], 0.1)).clip(0, len(ids) - 1)]
    tt = np.array([0.0, 0.42])
    lines = [Pp[j] + tt[:, None] * Nn[j] for j in feet]
    for to in (0.14, 0.28, 0.42):
        lines.append(Pp[ids] + to * Nn[ids])
    ax2.add_collection(LineCollection(lines, colors=C_SHAPE, linewidths=0.45, zorder=3))
    reg = ids[(Pp[ids, 0] > 1.25) & (Pp[ids, 0] < 2.7)]
    for j, lab, side in ((reg[np.argmin(K[reg])], "concave", -1),
                         (reg[np.argmax(K[reg])], "convex", 1)):
        sl = (S >= S[j] - 0.05) & (S <= S[j] + 0.05)
        Pc, Nc = Pp[sl], Nn[sl]
        cell = np.vstack([Pc + 0.14 * Nc, (Pc + 0.28 * Nc)[::-1]])
        ax2.add_patch(Polygon(cell, closed=True, fc=C_SHAPE, alpha=0.6, lw=0, zorder=4))
        mid = Pp[j] + 0.21 * Nn[j]
        tp = Pp[j] - (0.42 if side < 0 else 0.30) * Nn[j] + np.array([0.12 * side, 0.0])
        ax2.annotate(f"{lab}\n$1+t\\kappa={1 + 0.21 * K[j]:.2f}$", xy=mid, xytext=tp,
                     fontsize=5.9, ha="center", va="top", color=C_SHAPE_D, linespacing=1.0,
                     arrowprops=dict(arrowstyle="-", lw=0.4, color=C_SHAPE_D,
                                     shrinkA=0.5, shrinkB=0.5))
    ax2.text(xl[0] + 0.02, zl[1] - 0.02, r"$ds^2=(1+t\kappa)^2dp^2+dt^2$", fontsize=6.6,
             va="top", color=C_TEXT)
    ax2.text(xl[0] + 0.02, zl[1] - 0.19, "wall coordinates $(p,t)$ on the centre\nplane, round the windscreen",
             fontsize=6.0, color=C_SUB, va="top", linespacing=1.05)


# ------------------------------------------------------------- (b) geometry
def panel_b(fig):
    ax = mesh_axes(fig, 1)
    Wn = np.sqrt(d["k1"] ** 2 + d["k2"] ** 2)
    draw_mesh(ax, CMAP_W(NORM_W(face_values(Wn))))
    vis = np.where(facing(N, 0.4))[0]
    from build3d import fps
    rng = np.random.default_rng(2)
    pick = vis[rng.choice(len(vis), min(3000, len(vis)), replace=False)]
    sel = pick[fps(V[pick], 70, seed=3)]
    segs = [proj(np.vstack([V[j] - 0.11 * N[j], V[j] + 0.11 * N[j]])) for j in sel]
    ax.add_collection(LineCollection(segs, colors=C_EST, linewidths=0.6, zorder=5))
    colorbar(fig, 1, CMAP_W, NORM_W, [0, 2, 4], width=0.42)
    caption(fig, 1, r"$|W|$;  lines: $nn^{\!\!\top}$", x_off=0.52)

    ax2 = inch_axes(fig, col_x(1), BOT_Y + 0.42, CW, BOT_H - 0.42)
    ax2.set_aspect("equal")
    ax2.axis("off")
    gx, gz, D = d["eta_x"], d["eta_z"], d["eta_d"]
    xl = (-0.42, 4.92)
    zl = (-0.15, -0.15 + (BOT_H - 0.42) / CW * (xl[1] - xl[0]))
    ax2.set_xlim(*xl)
    ax2.set_ylim(*zl)
    band = (np.floor(D / 0.12) % 2).astype(bool)
    img = np.zeros(D.shape + (4,))
    img[..., :3] = np.array(to_rgba(C_SHAPE)[:3])
    img[..., 3] = 0.34 * np.exp(-D / 0.8) * np.where(band, 0.5, 1.0)
    im = ax2.imshow(img, extent=(gx[0], gx[-1], gz[0], gz[-1]), origin="lower",
                    interpolation="bilinear", zorder=0)
    im.set_rasterized(True)
    S, Pp, T, Nn, K, _ = section()
    ax2.plot(np.append(Pp[:, 0], Pp[0, 0]), np.append(Pp[:, 1], Pp[0, 1]), color=C_TEXT,
             lw=0.9, zorder=3)
    for x in (0.5, 1.0, 1.75, 2.45, 3.2, 3.9):
        j = upper_index(Pp, x)
        a, b = Pp[j] - 0.3 * Nn[j], Pp[j] + 0.3 * Nn[j]
        ax2.plot([a[0], b[0]], [a[1], b[1]], color=C_EST, lw=0.75, zorder=4,
                 solid_capstyle="round")
    up = np.where(Pp[:, 1] > 0.75)[0]
    reg = up[(Pp[up, 0] > 1.25) & (Pp[up, 0] < 2.7)]
    for j in (reg[np.argmax(K[reg])], reg[np.argmin(K[reg])]):
        r = 1 / abs(K[j])
        c = Pp[j] - Nn[j] / K[j]
        a0 = np.arctan2(*(Pp[j] - c)[::-1])
        span = min(1.2, 0.9 / r) if r > 0.5 else 1.4
        th = np.linspace(a0 - span, a0 + span, 80)
        ax2.plot(c[0] + r * np.cos(th), c[1] + r * np.sin(th), color=C_SHAPE_D, lw=0.7,
                 ls=(0, (3, 1.5)), zorder=5)
        ax2.plot(*Pp[j], "o", ms=2.6, mfc=C_SHAPE, mec="white", mew=0.4, zorder=6)
    ax2.text(xl[0] + 0.04, zl[1] - 0.03,
             r"value: $\eta_S=0$ on $S$" + "\n" + r"Hessian: $\nabla^2\eta_S=nn^{\!\!\top}\!$ (purple)"
             + "\n" + r"third derivative: $W$ (arcs)", fontsize=6.0, va="top", color=C_TEXT,
             linespacing=1.12)
    ax2.text(xl[1] - 0.03, zl[0] + 0.04, r"bands: $\eta_S$, centre plane", fontsize=6.0,
             color=C_SUB, ha="right", va="bottom")
    # the Hessian at p, read off two slices of eta through p (same p as in (a))
    import igl
    j = upper_index(Pp, 2.6)
    p3 = np.array([Pp[j, 0], 0.0, Pp[j, 1]])
    ax2.plot(*Pp[j], "o", ms=2.8, mfc=C_SHAPE, mec="white", mew=0.4, zorder=6)
    ax2.text(Pp[j, 0] + 0.1, Pp[j, 1] - 0.08, "$p$", fontsize=6.2, color=C_SHAPE_D, va="top")
    u = np.linspace(-0.3, 0.3, 121)
    for i, (vec, col, lab) in enumerate((
            (np.array([Nn[j, 0], 0.0, Nn[j, 1]]), C_EST, r"across $S$: $\frac{1}{2}t^2$"),
            (np.array([T[j, 0], 0.0, T[j, 1]]), C_GREY, r"along $S$: flat"))):
        Q = p3[None, :] + u[:, None] * vec[None, :]
        sq, _, _ = igl.point_mesh_squared_distance(Q, V, F)
        a3 = inch_axes(fig, col_x(1) + 0.1 + i * 0.76, BOT_Y + 0.04, 0.62, 0.24)
        for side in ("top", "right"):
            a3.spines[side].set_visible(False)
        a3.plot(u, 0.5 * sq, color=col, lw=1.0)
        a3.plot([0], [0], "o", ms=2.4, mfc=C_SHAPE, mec="white", mew=0.3)
        a3.set_xlim(-0.3, 0.3)
        a3.set_ylim(0, 0.048)
        a3.set_xticks([-0.3, 0, 0.3])
        a3.set_xticklabels(["−0.3", "0", "0.3"], fontsize=5)
        a3.set_yticks([0, 0.04])
        a3.set_yticklabels(["0", ".04"], fontsize=5)
        a3.tick_params(length=1.2, pad=0.8)
        a3.set_title(lab, fontsize=5.6, color=col if i == 0 else C_SUB, pad=1.5)


# ------------------------------------------------------------- (c) anchors
def panel_c(fig):
    ax = mesh_axes(fig, 2)
    ap, an, aw = d["anc_p"], d["anc_n"], d["anc_w"]
    owner = cKDTree(ap).query(V[F].mean(1))[1]
    tones = np.array([to_rgba(c) for c in ("#F6E2CD", "#E9C29B", "#F0D1B0", "#E2B48A")])
    draw_mesh(ax, tones[(owner * 7) % 4], ambient=0.62, diffuse=0.42)
    vis = facing(an, 0.2)
    th = np.linspace(0, 2 * np.pi, 18)
    depth = (ap @ VIEW.M.T)[:, 2]
    polys = []
    for j in np.argsort(depth):
        if not vis[j]:
            continue
        n = an[j]
        t1 = np.cross(n, [0, 0, 1.0])
        t1 = np.cross(n, [1.0, 0, 0]) if np.linalg.norm(t1) < 1e-3 else t1
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        r = 0.36 * np.sqrt(aw[j] / np.pi)
        circ = ap[j] + r * (np.cos(th)[:, None] * t1 + np.sin(th)[:, None] * t2) + 0.005 * n
        polys.append(proj(circ))
    ax.add_collection(PolyCollection(polys, facecolors=C_SHAPE_D, edgecolors="white",
                                     linewidths=0.25, zorder=6))
    caption(fig, 2, r"256 anchors and their cells;  disk area $\propto w_j$", x_off=0.0)

    P = d["samp_p"]
    v = np.load(os.path.join(HERE, "samp_cp.npy"))
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
    ax2 = inch_axes(fig, col_x(2) + 0.27, BOT_Y + 0.2, CW - 0.3, BOT_H - 0.33)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    ax2.axhline(exact, color=C_TEXT, lw=0.6, ls=(0, (4, 2)))
    for c, col in ((1, C_GREY), (0, C_EST)):
        m, s = E[:, :, c].mean(1), E[:, :, c].std(1)
        ax2.fill_between(rhos, m - s, m + s, color=col, alpha=0.25, lw=0)
        ax2.plot(rhos, m, color=col, lw=1.1)
    ax2.set_xlim(1, 9)
    ax2.set_xticks([1, 3, 5, 7, 9])
    ax2.set_xticklabels(["1×", "3×", "5×", "7×", "9×"])
    lo = min(E[:, :, 0].min(), exact) - 0.01
    hi = E[:, :, 1].max() + 0.01
    ax2.set_ylim(lo, hi)
    ax2.set_xlabel("sampling density, front half : rear half", labelpad=1)
    ax2.set_ylabel(r"attention at $q$, $C_p$", labelpad=1)
    ax2.text(5.0, E[4, :, 1].mean() + 0.012, "ordinary", fontsize=6.0, color=C_SUB,
             ha="center", va="bottom")
    ax2.text(5.0, E[4, :, 0].min() - 0.004, "with log-area bias", fontsize=6.0, color=C_EST,
             ha="center", va="top")
    right = rhos >= 6                    # label the dashed line at its right end,
    band_top = (E[:, :, 0].mean(1) + E[:, :, 0].std(1))[right].max()   # above the purple band
    ax2.text(8.9, band_top + 0.003, "surface integral", fontsize=5.8, color=C_TEXT,
             ha="right", va="bottom")
    fig.text((col_x(2) + 0.27) / W_IN, (BOT_Y + BOT_H - 0.04) / H_IN,
             r"4096 samples; query $q$ above the roof", fontsize=6.0, color=C_SUB, va="top")
    return float(exact), E


# ------------------------------------------------------------- (d) reconstruction
def panel_d(fig):
    ax = mesh_axes(fig, 3)
    hid = d["hidden"][F].sum(1) >= 2
    draw_mesh(ax, np.where(hid[:, None], np.array(to_rgba("#8FD3CB")), BODY))
    caption(fig, 3, "masked input: 60% hidden as 32 patches", x_off=0.0)
    ax2 = inch_axes(fig, col_x(3) + 0.27, BOT_Y + 0.2, CW - 0.3, BOT_H - 0.33)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    om_path = os.path.join(HERE, "omega.npy")
    out = None
    if os.path.exists(om_path):
        om = np.load(om_path)
        dcp, dpos, dfull = om[:, 0], om[:, 1], om[:, 2]
        Lsl = float(np.max(dcp / dfull))
        xm = 1.08 * dfull.max()
        ax2.plot([0, xm], [0, Lsl * xm], color=C_SHAPE, lw=0.7, ls=(0, (4, 2)), zorder=2)
        ax2.scatter(dpos, dcp, s=8, facecolors="white", edgecolors=C_SUB, linewidths=0.6,
                    zorder=3, clip_on=False)
        ax2.scatter(dfull, dcp, s=8, color=C_SHAPE, lw=0, zorder=4, clip_on=False)
        ax2.set_xlim(0, xm)
        top = 1.12 * dcp.max()
        ax2.set_ylim(0, top)
        yt = 0.96 * top
        ax2.text(yt / Lsl + 0.04 * xm, yt, r"$\omega(r)\leq L\,r$", fontsize=6.0,
                 color=C_SHAPE_D, ha="left", va="center")
        # direct labels instead of a legend: open circles hug the axis, filled ones spread out
        ax2.text(dpos.max() + 0.03 * xm, 0.74 * top, "position only", fontsize=5.8,
                 color=C_SUB, ha="left", va="center")
        ax2.text(0.34 * xm, 0.09 * top, "with normal lines", fontsize=5.8,
                 color=C_SHAPE_D, ha="left", va="center")
        out = (Lsl, om)
    ax2.set_xlabel(r"shape distance $d(S,\hat S)$", labelpad=1)
    ax2.set_ylabel(r"field error $\|\Delta C_p\|_S$", labelpad=1)
    fig.text((col_x(3) + 0.27) / W_IN, (BOT_Y + BOT_H - 0.04) / H_IN,
             "perturbed shapes, flow re-solved", fontsize=6.0, color=C_SUB, va="top")
    return out


def main(stem):
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor="white")
    headers(fig)
    panel_a(fig)
    panel_b(fig)
    ex, E = panel_c(fig)
    om = panel_d(fig)
    fig.savefig(stem + ".pdf", dpi=600)
    fig.savefig(stem + ".png", dpi=300)
    print("saved", stem)
    print("attention: exact %.3f, area-weighted at 9x %.3f, ordinary at 9x %.3f"
          % (ex, E[-1, :, 0].mean(), E[-1, :, 1].mean()))
    if om is not None:
        L, o = om
        print("omega: L = %.3f; max dCp/d_pos = %.1f; n = %d" % (L, np.max(o[:, 0] / o[:, 1]), len(o)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "areals_hero"))
