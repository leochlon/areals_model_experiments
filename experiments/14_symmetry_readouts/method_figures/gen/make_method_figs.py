"""Writes the TikZ method figures (Figures 2, 3 and 5) in the style of Figure 1.

Geometry (curves, normals, disks, vector fields) is computed here; labels are TikZ nodes, so every word is set in the
paper's font. Run from anywhere: python3 make_method_figs.py  -> ../masked_pretraining.tex, ../priors.tex, ../signfree.tex
"""
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def P(x, y):
    return f"({x:.3f},{y:.3f})"


def plot(pts):
    return "plot coordinates {" + " ".join(P(x, y) for x, y in pts) + "}"


HEAD = r"""\begin{tikzpicture}[x=1in, y=1in, font=\footnotesize,
    flow/.style={-{Stealth[length=4.2pt,width=3.4pt]}, line width=0.6pt, draw=arealfg},
    lab/.style={inner sep=1pt, font=\scriptsize, text=arealmuted},
    tlab/.style={inner sep=1pt, font=\scriptsize, text=arealteal},
    wlab/.style={inner sep=1pt, font=\scriptsize, text=arealwarn},
    vis/.style={draw=arealfg, line width=0.9pt},
    hid/.style={draw=arealmuted!80, line width=0.8pt, dash pattern=on 2.6pt off 1.8pt},
    disk/.style={draw=arealteal, line width=1.9pt, opacity=0.55, line cap=butt}]
"""
TAIL = "\\end{tikzpicture}%\n"


# ===================================================================================================== Figure 2
def masked_pretraining():
    rng = np.random.default_rng(4)
    n = 1440
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = 1 + 0.16 * np.sin(2 * th + 0.6) + 0.10 * np.cos(3 * th - 0.4) + 0.05 * np.sin(5 * th + 1.3)
    x, y = r * np.cos(th), r * np.sin(th)
    W, H, cx, cy = 2.30, 1.72, 1.28, 1.02
    x = cx + (x - (x.max() + x.min()) / 2) * W / (x.max() - x.min())
    y = cy + (y - (y.max() + y.min()) / 2) * H / (y.max() - y.min())
    dx, dy = np.gradient(x), np.gradient(y)
    dx[0], dy[0] = x[1] - x[-1], y[1] - y[-1]
    dx[-1], dy[-1] = x[0] - x[-2], y[0] - y[-2]
    t = np.hypot(dx, dy)
    nx, ny = dy / t, -dx / t                                  # outward normal of a counter-clockwise curve
    seg = np.hypot(np.diff(x, append=x[0]), np.diff(y, append=y[0]))
    s = np.concatenate([[0], np.cumsum(seg)[:-1]])
    hidden_arcs = [(1.70, 2.62), (4.80, 5.72)]
    hidden = np.zeros(n, bool)
    for a, b in hidden_arcs:
        hidden |= (th >= a) & (th <= b)

    def runs(mask):
        idx = np.flatnonzero(mask)
        out, start = [], idx[0]
        for i, j in zip(idx[:-1], idx[1:]):
            if j != i + 1:
                out.append((start, i)); start = j
        out.append((start, idx[-1]))
        if len(out) > 1 and out[0][0] == 0 and out[-1][1] == n - 1:   # join across the seam
            out[0] = (out[-1][0], out[0][1] + n); out.pop()
        return out

    def arc(i0, i1):
        ii = np.arange(i0, i1 + 2) % n
        return list(zip(x[ii], y[ii]))

    L = []
    # query points: 70% near the surface, the rest anywhere in the box
    qn = 46
    k = rng.integers(0, n, qn)
    off = rng.normal(0, 0.075, qn)
    near = np.stack([x[k] + off * nx[k], y[k] + off * ny[k]], 1)
    far = np.stack([rng.uniform(x.min() - 0.12, x.max() + 0.12, 20), rng.uniform(y.min() - 0.08, y.max() + 0.08, 20)], 1)
    for qx, qy in np.concatenate([near, far]):
        if cx - 0.66 < qx < cx + 0.70 and cy - 0.38 < qy < cy + 0.14:      # keep the token label clear
            continue
        L.append(f"\\fill[arealmuted!45] {P(qx, qy)} circle (0.011);")
    for i0, i1 in runs(hidden):
        L.append(f"\\draw[hid] {plot(arc(i0, i1))};")
    for i0, i1 in runs(~hidden):
        L.append(f"\\draw[vis] {plot(arc(i0, i1))};")
    # areal tokens on the visible surface, irregularly spaced; dot area tracks the arc each one represents
    toks = []
    for i0, i1 in runs(~hidden):
        ii = np.arange(i0, i1 + 1) % n
        sl = np.cumsum(np.r_[0, seg[ii][:-1]])
        length = sl[-1]
        m = max(2, int(round(length / 0.28)))
        u = (np.arange(m) + 0.5) / m
        u = np.clip(u + 0.33 / m * np.sin(2.7 * np.arange(m) + i0), 0.04, 0.96)
        u.sort()
        pos = u * length
        edges = np.r_[0, (pos[1:] + pos[:-1]) / 2, length]
        for p, w in zip(pos, np.diff(edges)):
            j = ii[np.searchsorted(sl, p).clip(0, len(ii) - 1)]
            toks.append((j, w))
    wmean = np.mean([w for _, w in toks])
    for j, w in toks:
        rad = float(np.clip(0.027 * np.sqrt(w / wmean), 0.019, 0.037))
        px, py = x[j], y[j]
        L.append(f"\\draw[arealteal, line width=0.6pt] {P(px, py)} -- {P(px + 0.11 * nx[j], py + 0.11 * ny[j])};")
        L.append(f"\\filldraw[fill=arealteal, draw=arealfg, line width=0.3pt] {P(px, py)} circle ({rad:.3f});")
    # one token labelled from inside the shape
    jl = min(toks, key=lambda tw: (x[tw[0]] - (cx + 0.15)) ** 2 + (y[tw[0]] - (cy - 0.85)) ** 2)[0]
    L.append(f"\\draw[arealteal!70, line width=0.4pt] {P(x[jl] - 0.02 * nx[jl], y[jl] - 0.02 * ny[jl])} -- {P(cx + 0.02, cy - 0.30)};")
    L.append(f"\\node[tlab, anchor=south, align=center] at {P(cx + 0.02, cy - 0.31)} {{areal token\\\\[-1pt]\\textcolor{{arealmuted}}{{position $p_j$, normal $n_j$,}}\\\\[-1pt]\\textcolor{{arealmuted}}{{area weight $w_j$}}}};")
    # queries x1 (over visible surface) and x2 (over a hidden patch): distance to the full surface, normal line at the foot
    for name, th0, dist, side in (("x_1", 3.05, -0.30, "west"), ("x_2", 5.30, 0.30, "south west")):
        j = int(np.argmin(np.abs(th - th0)))
        fx, fy = x[j], y[j]
        qx, qy = fx + dist * nx[j], fy + dist * ny[j]
        L.append(f"\\draw[arealfg, line width=0.5pt, densely dotted] {P(qx, qy)} -- {P(fx, fy)};")
        L.append(f"\\fill[arealfg] {P(qx, qy)} circle (0.030);")
        L.append(f"\\node[inner sep=2pt, anchor={side}, font=\\footnotesize] at {P(qx, qy)} {{${name}$}};")
        if name == "x_2":
            x2 = (qx, qy)
    jv = int(np.argmin(np.abs(th - 4.25)))
    L.append(f"\\node[lab, anchor=north] at {P(x[jv] + 0.05 * nx[jv], y[jv] + 0.05 * ny[jv])} {{visible}};")
    jh = int(np.argmin(np.abs(th - 2.15)))
    L.append(f"\\node[lab, anchor=south east] at {P(x[jh] + 0.06 * nx[jh], y[jh] + 0.06 * ny[jh])} {{hidden}};")

    # the network, as in Figure 1
    yc, ex, ew, tx, dxn, hx, hw = 1.34, 3.26, 0.48, 4.08, 4.43, 4.95, 0.40
    L.append(f"\\filldraw[fill=arealgrid!45, draw=arealfg, line width=0.5pt] {P(ex, yc - 0.40)} -- {P(ex, yc + 0.40)} -- {P(ex + ew, yc + 0.20)} -- {P(ex + ew, yc - 0.20)} -- cycle;")
    L.append(f"\\filldraw[fill=arealgrid!45, draw=arealfg, line width=0.5pt] {P(dxn, yc - 0.20)} -- {P(dxn, yc + 0.20)} -- {P(dxn + ew, yc + 0.40)} -- {P(dxn + ew, yc - 0.40)} -- cycle;")
    L.append(f"\\node[text=arealfg] at {P(ex + 0.22, yc)} {{Encoder}};")
    L.append(f"\\node[text=arealfg] at {P(dxn + 0.26, yc)} {{Decoder}};")
    wts = np.array([1.0, 0.7, 1.3, 0.9, 1.15, 0.8, 1.0, 1.35, 0.75, 1.1, 0.95, 0.65, 1.25, 0.85, 1.05, 0.9])
    for a in range(4):
        for b in range(4):
            rad = 0.0185 * np.sqrt(wts[4 * a + b])
            L.append(f"\\filldraw[fill=arealteal, draw=arealfg, line width=0.3pt] {P(tx - 0.1275 + 0.085 * a, yc - 0.1275 + 0.085 * b)} circle ({rad:.4f});")
    L.append(f"\\node[tlab] at {P(tx, yc + 0.30)} {{256 areal tokens}};")
    L.append(f"\\node[lab] at {P(tx, yc - 0.30)} {{dot area $\\propto w_j$}};")
    L.append(f"\\draw[flow] {P(ex + ew + 0.02, yc)} -- {P(tx - 0.165, yc)};")
    L.append(f"\\draw[flow] {P(tx + 0.165, yc)} -- {P(dxn - 0.02, yc)};")
    L.append(f"\\node[draw=arealfg, fill=white, line width=0.5pt, minimum width={hw}in, minimum height=0.17in, inner sep=0pt, font=\\scriptsize] at {P(hx + hw / 2, yc)} {{$d,\\ \\nnT$}};")
    L.append(f"\\node[lab, anchor=west, align=left] at {P(hx + hw + 0.06, yc)} {{unsigned distance\\\\[-1pt]and normal line}};")
    # visible samples in; answers at every query out, hidden patches included
    xr = x.max()
    jin = int(np.argmin((x - xr) ** 2 + (y - (yc + 0.05)) ** 2 + 10 * hidden))
    L.append(f"\\draw[flow] {P(x[jin] + 0.10, yc)} -- {P(ex - 0.02, yc)};")
    L.append(f"\\node[lab, anchor=south] at {P((x[jin] + 0.10 + ex) / 2, yc + 0.02)} {{visible samples}};")
    L.append(f"\\node[lab, anchor=north] at {P((x[jin] + 0.10 + ex) / 2, yc - 0.02)} {{$(x_i,\\,n_i,\\,w_i)$}};")
    yb = 0.10
    L.append(f"\\draw[flow, rounded corners=6pt] {P(hx + hw / 2, yc - 0.10)} -- {P(hx + hw / 2, yb)} -- {P(x2[0] + 0.30, yb)} to[out=180, in=-75] {P(x2[0] + 0.012, x2[1] - 0.045)};")
    L.append(f"\\node[lab, anchor=south] at {P((hx + hw / 2 + x2[0] + 0.30) / 2 + 0.25, yb + 0.01)} {{answers at any query $x$: the loss covers hidden patches too}};")
    return HEAD + "\n".join(L) + "\n" + TAIL


# ===================================================================================================== Figure 3
def bump(u):
    return np.exp(-u * u)


def priors():
    L = []
    arrow = "-{Stealth[length=3.8pt,width=3.1pt]}"
    # ------------------------------------------------------------------------------- (a) surfel readout and gate
    X = np.linspace(0.05, 3.20, 900)
    base = lambda v: 0.40 + 0.09 * np.sin(2 * np.pi * v / 1.55 + 0.5)
    h0, h1 = 1.28, 2.06
    hc, hw_ = (h0 + h1) / 2, (h1 - h0) / 2
    inside = lambda v: (v > h0) & (v < h1)
    true_y = lambda v: base(v) - 0.17 * bump((v - hc) / (0.55 * hw_)) * inside(v)
    head_y = lambda v: base(v) - 0.11 * bump((v - hc) / (0.62 * hw_)) * inside(v)
    for m in (X <= h0, X >= h1):
        L.append(f"\\draw[vis] {plot(zip(X[m], base(X[m])))};")
    mh = (X >= h0) & (X <= h1)
    L.append(f"\\draw[hid] {plot(zip(X[mh], true_y(X[mh])))};")
    L.append(f"\\draw[arealteal, line width=0.8pt, dash pattern=on 2.6pt off 1.8pt] {plot(zip(X[mh], head_y(X[mh])))};")
    anchors = [0.30, 0.82, 2.40, 2.92]
    extras = [0.12, 0.47, 0.64, 0.99, 1.16, 2.23, 2.57, 2.75, 3.09]
    rad = 0.075

    def tangent(v):
        e = 1e-3
        d = np.array([2 * e, base(v + e) - base(v - e)])
        return d / np.hypot(*d)

    for v in sorted(anchors + extras):
        tv = tangent(v)
        c = np.array([v, base(v)])
        L.append(f"\\draw[disk] {P(*(c - rad * tv))} -- {P(*(c + rad * tv))};")
    for v in extras:
        L.append(f"\\filldraw[fill=white, draw=arealteal, line width=0.5pt] {P(v, base(v))} circle (0.019);")
    for v in anchors:
        L.append(f"\\filldraw[fill=arealteal, draw=arealfg, line width=0.3pt] {P(v, base(v))} circle (0.026);")
    # x1: projects inside its nearest disk
    v1 = 0.64
    tv = tangent(v1); nv = np.array([-tv[1], tv[0]])
    foot = np.array([v1, base(v1)])
    q1 = foot + 0.42 * nv
    L.append(f"\\draw[{arrow}, arealfg, line width=0.55pt] {P(*q1)} -- {P(*(foot + 0.035 * nv))};")
    L.append(f"\\fill[arealfg] {P(*q1)} circle (0.028);")
    L.append(f"\\node[inner sep=2pt, anchor=west] at {P(*q1)} {{$x_1$}};")
    L.append(f"\\node[inner sep=1pt, anchor=east, font=\\scriptsize] at {P(*(foot + 0.21 * nv - np.array([0.03, 0])))} {{$d$}};")
    L.append(f"\\node[tlab, anchor=south, align=center] at {P(q1[0], q1[1] + 0.07)} {{inside its disk:\\\\[-1pt]$\\lambda\\approx1$, trust the disks}};")
    # x2: over the hole; its nearest disk is the last one before the hole, and it projects outside that disk
    q2 = np.array([1.56, 0.98])
    near = min(anchors + extras, key=lambda v: np.hypot(q2[0] - v, q2[1] - base(v)))
    tv = tangent(near); nv = np.array([-tv[1], tv[0]])
    c = np.array([near, base(near)])
    along = float(np.dot(q2 - c, tv))
    proj = c + along * tv
    edge = c + np.sign(along) * rad * tv
    L.append(f"\\draw[arealmuted, line width=0.45pt, dash pattern=on 1pt off 1.3pt] {P(*edge)} -- {P(*(proj + np.sign(along) * 0.08 * tv))};")
    L.append(f"\\draw[arealfg, line width=0.5pt, densely dotted] {P(*q2)} -- {P(*proj)};")
    L.append(f"\\draw[arealwarn, line width=0.7pt, |-|] {P(*(edge + 0.05 * nv))} -- {P(*(proj + 0.05 * nv))};")
    L.append(f"\\node[wlab, anchor=south] at {P(*((proj + edge) / 2 + 0.07 * nv))} {{rim}};")
    hm = (q2 + proj) / 2
    L.append(f"\\node[lab, anchor=west] at {P(hm[0] + 0.03, hm[1])} {{height}};")
    L.append(f"\\fill[arealfg] {P(*q2)} circle (0.028);")
    L.append(f"\\node[inner sep=2pt, anchor=west] at {P(*q2)} {{$x_2$}};")
    L.append(f"\\node[lab, anchor=south, align=center] at {P(q2[0], q2[1] + 0.07)} {{outside every disk:\\\\[-1pt]$\\lambda\\approx0$, trust the heads}};")
    L.append(f"\\node[lab, anchor=north east, align=right] at (3.22,0.88) {{$d=\\lambda\\,d_{{\\mathrm{{disk}}}}+(1-\\lambda)\\,d_{{\\mathrm{{head}}}}$\\\\[1pt]and likewise for $\\nnT$}};")
    ly = 0.05
    L.append(f"\\filldraw[fill=arealteal, draw=arealfg, line width=0.3pt] (0.10,{ly}) circle (0.024);")
    L.append(f"\\node[lab, anchor=west] at (0.15,{ly}) {{anchor}};")
    L.append(f"\\filldraw[fill=white, draw=arealteal, line width=0.5pt] (0.62,{ly}) circle (0.019);")
    L.append(f"\\node[lab, anchor=west] at (0.67,{ly}) {{extra surfel}};")
    L.append(f"\\draw[disk] (1.30,{ly}) -- (1.45,{ly});")
    L.append(f"\\node[lab, anchor=west] at (1.49,{ly}) {{disk}};")
    L.append(f"\\draw[arealteal, line width=0.8pt, dash pattern=on 2.6pt off 1.8pt] (1.85,{ly}) -- (2.05,{ly});")
    L.append(f"\\node[lab, anchor=west] at (2.09,{ly}) {{heads' completion}};")
    L.append(f"\\node[font=\\small, text=arealfg] at (1.62,-0.17) {{(a) Surfel readout and gate}};")

    # ------------------------------------------------------------------------------- (b) symmetry maps
    ox = 3.50
    Xb = np.linspace(0.05, 3.20, 1200)
    c1, c2, tshift = 0.95, 2.35, 1.40

    def prof(v, c):                     # one repeated part, steep on one side and gentle on the other
        u = v - c
        return 0.48 * np.where(u < 0, np.exp(-(u / 0.26) ** 2), np.exp(-(u / 0.14) ** 2))

    yb_ = lambda v: 0.28 + prof(v, c1) + prof(v, c2)
    g0, g1_ = c2 - 0.05, c2 + 0.30            # hidden patch on the right copy
    for m in (Xb <= g0, Xb >= g1_):
        L.append(f"\\draw[vis] {plot(zip(ox + Xb[m], yb_(Xb[m])))};")
    mh = (Xb >= g0) & (Xb <= g1_)
    L.append(f"\\draw[hid] {plot(zip(ox + Xb[mh], yb_(Xb[mh])))};")

    def tan_b(v):
        e = 1e-3
        d = np.array([2 * e, yb_(v + e) - yb_(v - e)])
        return d / np.hypot(*d)

    for v in [0.20, 0.45, 0.62, 0.74, 0.86, 1.00, 1.10, 1.40, 1.70, 1.92, 2.08, 2.20, 2.85, 3.08]:
        tvec = tan_b(v)
        cc = np.array([ox + v, yb_(v)])
        L.append(f"\\draw[disk] {P(*(cc - 0.06 * tvec))} -- {P(*(cc + 0.06 * tvec))};")
    vx = c2 + 0.05
    nb = np.array([-tan_b(vx)[1], tan_b(vx)[0]])
    qx = np.array([ox + vx, yb_(vx)]) + 0.15 * nb
    gx = qx - np.array([tshift, 0])
    L.append(f"\\draw[{arrow}, arealteal, line width=0.7pt] {P(*qx)} to[out=150, in=30] {P(*(gx + np.array([0.035, 0.03])))};")
    L.append(f"\\node[tlab, anchor=south east] at {P((qx[0] + gx[0]) / 2 - 0.08, max(qx[1], gx[1]) + 0.27)} {{$g_1$: shift by $t$}};")
    L.append(f"\\draw[{arrow}, arealteal, line width=0.6pt, dash pattern=on 2.4pt off 1.6pt] {P(*(gx + np.array([0.02, -0.035])))} to[out=-35, in=215] {P(*(qx + np.array([-0.035, -0.03])))};")
    L.append(f"\\node[tlab, anchor=north] at {P((qx[0] + gx[0]) / 2 - 0.30, min(qx[1], gx[1]) - 0.30)} {{readout carried back}};")
    L.append(f"\\filldraw[fill=white, draw=arealteal, line width=0.7pt] {P(*gx)} circle (0.030);")
    L.append(f"\\node[tlab, anchor=south east] at {P(gx[0] - 0.02, gx[1] + 0.03)} {{$g_1(x)$: on a disk}};")
    g2 = qx + np.array([0.10, 0.50])
    L.append(f"\\draw[{arrow}, arealwarn, line width=0.7pt] {P(*qx)} -- {P(*(g2 - np.array([0.035, 0.035])))};")
    L.append(f"\\draw[arealwarn, line width=0.8pt] {P(*(g2 + np.array([-0.035, -0.035])))} -- {P(*(g2 + np.array([0.035, 0.035])))} {P(*(g2 + np.array([-0.035, 0.035])))} -- {P(*(g2 + np.array([0.035, -0.035])))};")
    L.append(f"\\node[wlab, anchor=east, align=right] at {P(g2[0] - 0.07, g2[1])} {{$g_2(x)$: no disk,\\\\[-1pt]weight 0}};")
    L.append(f"\\fill[arealfg] {P(*qx)} circle (0.028);")
    L.append(f"\\node[inner sep=2pt, anchor=west] at {P(qx[0] + 0.01, qx[1] - 0.02)} {{$x$}};")
    L.append(f"\\node[lab, anchor=west] at {P(ox + g1_ + 0.05, yb_(g1_) + 0.17)} {{hidden}};")
    L.append(f"\\node[lab, anchor=north] at {P(ox + c1, 0.22)} {{visible copy}};")
    L.append(f"\\node[font=\\small, text=arealfg] at {P(ox + 1.62, -0.17)} {{(b) Symmetry maps}};")
    return HEAD + "\n".join(L) + "\n" + TAIL


# ===================================================================================================== Figure 5
def signfree():
    L = []
    for panel, ox in (("vec", 0.0), ("proj", 3.30)):
        ecx, ecy, ea, eb = 0.85, 0.92, 0.50, 0.34
        tip = ecx + ea + 0.62
        L.append(f"\\filldraw[fill=white, draw=arealfg, line width=0.9pt] {P(ox + ecx, ecy)} ellipse ({ea} and {eb});")
        L.append(f"\\draw[arealfg, line width=1.1pt] {P(ox + ecx + ea, ecy)} -- {P(ox + tip, ecy)};")
        if panel == "vec":
            L.append(f"\\fill[arealwarn!13] {P(ox + tip + 0.05, ecy - 0.11)} rectangle {P(ox + 3.02, ecy + 0.11)};")
        th = np.linspace(0, 2 * np.pi, 2000)
        ell = np.stack([ecx + ea * np.cos(th), ecy + eb * np.sin(th)], 1)
        for gx in np.arange(0.12, 3.05, 0.19):
            for gy in np.arange(0.12, 1.78, 0.19):
                p = np.array([gx, gy])
                if ((gx - ecx) / (ea + 0.07)) ** 2 + ((gy - ecy) / (eb + 0.07)) ** 2 < 1:
                    continue
                if abs(gy - ecy) < 0.06 and ecx + ea - 0.02 < gx < tip + 0.03:
                    continue
                if gx > tip:                         # beyond the free edge, where the fin would continue
                    if panel == "vec":
                        ang = (np.pi / 2) * np.tanh((gy - ecy) / 0.10)
                        d = np.array([np.cos(ang), np.sin(ang)])
                    else:
                        d = np.array([0.0, 1.0])
                else:
                    cand = [ell[np.argmin(((ell - p) ** 2).sum(1))]]
                    if ecx + ea <= gx <= tip:
                        cand.append(np.array([gx, ecy]))
                    q = min(cand, key=lambda c: ((c - p) ** 2).sum())
                    d = (p - q) / np.linalg.norm(p - q)
                h = 0.062
                a, b = p - h * d, p + h * d
                if panel == "vec":
                    warm = gx > tip and abs(np.degrees(np.arctan2(d[1], d[0]))) < 50
                    col = "arealwarn" if warm else "arealfg"
                    L.append(f"\\draw[-{{Stealth[length=2.6pt,width=2.2pt]}}, {col}, line width=0.5pt] {P(ox + a[0], a[1])} -- {P(ox + b[0], b[1])};")
                else:
                    L.append(f"\\draw[arealfg, line width=0.55pt] {P(ox + a[0], a[1])} -- {P(ox + b[0], b[1])};")
        L.append(f"\\node[lab, anchor=north, fill=white, inner sep=1.5pt] at {P(ox + tip + 0.02, ecy - 0.12)} {{free edge}};")
        title = "Unit-vector head: $n$" if panel == "vec" else "Projector head: $\\nnT$"
        L.append(f"\\node[font=\\footnotesize\\bfseries, text=arealfg] at {P(ox + 1.55, 1.93)} {{{title}}};")
        if panel == "vec":
            L.append(f"\\node[wlab] at {P(ox + 1.55, -0.06)} {{a wall of $90^\\circ$ errors where the fin would continue}};")
        else:
            L.append(f"\\node[tlab] at {P(ox + 1.55, -0.06)} {{$n$ and $-n$ give the same line: no wall}};")
    L.append(f"\\draw[arealgrid, line width=0.6pt] {P(3.15, 0.0)} -- {P(3.15, 1.85)};")
    return HEAD + "\n".join(L) + "\n" + TAIL


if __name__ == "__main__":
    for name, fn in (("masked_pretraining", masked_pretraining), ("priors", priors), ("signfree", signfree)):
        with open(os.path.join(OUT, f"{name}.tex"), "w") as fh:
            fh.write(f"% Generated by gen/make_method_figs.py; edit that script, not this file.\n{fn()}")
        print("wrote", name)
