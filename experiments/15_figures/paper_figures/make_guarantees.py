"""Writes figures/guarantees.tex: the Euler--Lagrange construction and what it guarantees.

Left: the construction (Appendix B, "The field view"): locality, the shift symmetry, rotations and Galilean boosts fix
the Lagrangian density as the pressure up to the equation of state; making its integral stationary gives the
Euler--Lagrange equation, the law the flow obeys; so a flow has three separate inputs.
Right: eight guarantees of that construction (Appendix B, "The field view" and "Locality, symmetry and the wall"),
each with a pictogram, what it says, and its counterpart in AREALS.

Usage: python3 make_guarantees.py OUT.tex
"""
import sys

TILES = [
    # column 1: the wall and its geometry
    ('local', 'Local geometry only',
     r'In wall coordinates the shape enters only through $g_S$ and $W$ at the nearest wall point, all read from $\eta_S$.',
     r'reconstruct distance and normal line; pretrained attention stays local (\cref{fig:locality}).'),
    ('wall', 'No flow through the wall, for free',
     r'Stationarity itself implies $\partial_n\varphi=0$ on the wall; it is never imposed.',
     r'the wall acts only through its geometry.'),
    ('orient', 'Blind to orientation',
     r'Flipping $(n,t,W)\to(-n,-t,-W)$ leaves the metric, hence the action, unchanged.',
     r'sign-free targets lose nothing; raw meshes need no consistent normals.'),
    ('sample', 'Blind to sampling',
     r'The action sees the wall through its metric and area, not its parametrisation or mesh.',
     r'area-weighted attention: 81.5\% less drift under ninefold resampling.'),
    # column 2: the solution and the condition
    ('global', 'Long range lives in the solution',
     "Solving the local equations together spreads each wall element's influence over the flow.",
     r'learned from simulations: fine-tuning widens attention (\cref{fig:attention}).'),
    ('conserve', 'Mass is conserved',
     r'Noether: the shift symmetry makes the Euler--Lagrange equation mass conservation.',
     r'simulations obey it; synthetic trajectories (GeoPT) do not.'),
    ('rotate', 'Only the relative stream matters',
     "Rotating body and stream together leaves the action unchanged: the condition is the stream's speed and direction.",
     r'what the sensors recover; rotations augment pretraining.'),
    ('mirror', 'Symmetry needs a symmetric condition',
     r'A mirror-symmetric body has a mirror-symmetric field only if the condition is symmetric too.',
     r'symmetry maps extend to fields only then.'),
]


def picto(kind, x, y):
    """A small pictogram centred at (x, y), about 0.85 cm across."""
    P = []
    if kind == 'local':
        P.append(f"\\fill[arealgrid] ({x-0.42:.2f},{y-0.3:.2f}) .. controls ({x-0.12:.2f},{y+0.02:.2f}) and ({x+0.12:.2f},{y+0.02:.2f}) .. ({x+0.42:.2f},{y-0.3:.2f}) -- cycle;")
        P.append(f"\\draw[arealfg, line width=0.9pt] ({x-0.42:.2f},{y-0.3:.2f}) .. controls ({x-0.12:.2f},{y+0.02:.2f}) and ({x+0.12:.2f},{y+0.02:.2f}) .. ({x+0.42:.2f},{y-0.3:.2f});")
        P.append(f"\\draw[arealteal, line width=0.7pt, fill=arealrec, fill opacity=0.3] ({x:.2f},{y-0.02:.2f}) circle (0.2);")
        P.append(f"\\fill[arealwarn] ({x:.2f},{y+0.09:.2f}) circle (1.1pt);")
    elif kind == 'global':
        P.append(f"\\fill[arealgrid] ({x:.2f},{y-0.2:.2f}) ellipse (0.42 and 0.14);")
        for k in range(8):
            import math
            t = 2 * math.pi * k / 8
            P.append(f"\\draw[arealfg, line width=0.35pt, opacity=0.7] ({x-0.12:.2f},{y+0.28:.2f}) -- ({x+0.42*math.cos(t):.3f},{y-0.2+0.14*math.sin(t):.3f});")
        P.append(f"\\draw[arealfg, line width=0.9pt] ({x:.2f},{y-0.2:.2f}) ellipse (0.42 and 0.14);")
        P.append(f"\\fill[arealwarn] ({x-0.12:.2f},{y+0.28:.2f}) circle (1.1pt);")
    elif kind == 'wall':
        P.append(f"\\fill[arealgrid] ({x-0.42:.2f},{y-0.36:.2f}) rectangle ({x+0.42:.2f},{y-0.22:.2f});")
        P.append(f"\\draw[arealfg, line width=0.9pt] ({x-0.42:.2f},{y-0.22:.2f}) -- ({x+0.42:.2f},{y-0.22:.2f});")
        for dy in (-0.06, 0.14, 0.32):
            P.append(f"\\draw[arealteal, line width=0.6pt, -{{Stealth[length=2.5pt]}}] ({x-0.38:.2f},{y+dy:.2f}) -- ({x+0.38:.2f},{y+dy:.2f});")
    elif kind == 'conserve':
        P.append(f"\\draw[arealfg, line width=0.7pt, fill=arealgrid] ({x-0.17:.2f},{y-0.17:.2f}) rectangle ({x+0.17:.2f},{y+0.17:.2f});")
        P.append(f"\\draw[arealteal, line width=0.9pt, -{{Stealth[length=3pt]}}] ({x-0.46:.2f},{y:.2f}) -- ({x-0.2:.2f},{y:.2f});")
        P.append(f"\\draw[arealteal, line width=0.9pt, -{{Stealth[length=3pt]}}] ({x+0.2:.2f},{y:.2f}) -- ({x+0.46:.2f},{y:.2f});")
    elif kind == 'sample':
        for yy, n, r in ((y + 0.14, 3, 1.4), (y - 0.2, 8, 0.85)):
            P.append(f"\\draw[arealfg, line width=0.8pt] ({x-0.42:.2f},{yy:.2f}) -- ({x+0.42:.2f},{yy:.2f});")
            for k in range(n):
                P.append(f"\\fill[arealteal] ({x-0.36+0.72*k/(n-1):.3f},{yy:.2f}) circle ({r}pt);")
    elif kind == 'orient':
        P.append(f"\\fill[arealgrid] ({x-0.42:.2f},{y-0.26:.2f}) rectangle ({x+0.42:.2f},{y-0.12:.2f});")
        P.append(f"\\draw[arealfg, line width=0.9pt] ({x-0.42:.2f},{y-0.12:.2f}) -- ({x+0.42:.2f},{y-0.12:.2f});")
        P.append(f"\\draw[arealteal, line width=0.8pt, {{Stealth[length=3pt]}}-{{Stealth[length=3pt]}}] ({x:.2f},{y-0.42:.2f}) -- ({x:.2f},{y+0.3:.2f});")
    elif kind == 'rotate':
        P.append(f"\\begin{{scope}}[rotate around={{-20:({x+0.1:.2f},{y-0.05:.2f})}}]")
        P.append(f"\\draw[arealfg, line width=0.8pt, fill=arealgrid] ({x+0.1:.2f},{y-0.05:.2f}) ellipse (0.22 and 0.09);")
        P.append(f"\\draw[arealteal, line width=0.8pt, -{{Stealth[length=3pt]}}] ({x-0.42:.2f},{y-0.05:.2f}) -- ({x-0.16:.2f},{y-0.05:.2f});")
        P.append("\\end{scope}")
        P.append(f"\\draw[arealmuted, line width=0.5pt, -{{Stealth[length=2.5pt]}}] ({x:.2f},{y:.2f}) ++(35:0.42) arc (35:320:0.42);")
    elif kind == 'mirror':
        P.append(f"\\draw[arealfg, line width=0.8pt, fill=arealgrid] ({x:.2f},{y-0.04:.2f}) ellipse (0.36 and 0.16);")
        P.append(f"\\draw[arealteal, line width=0.6pt, dash pattern=on 2pt off 1.5pt] ({x:.2f},{y-0.38:.2f}) -- ({x:.2f},{y+0.32:.2f});")
    return P


def main(path):
    L = [r"""% Generated by make_guarantees.py; edit that script, not this file.
\begin{tikzpicture}[x=1cm, y=1cm, font=\scriptsize, execute at begin node={\hyphenpenalty=10000},
    box/.style={draw=arealmuted, line width=0.5pt, rounded corners=2pt, fill=white, inner sep=3pt, align=left,
                text width=4.25cm, font=\scriptsize, text=arealfg},
    elbox/.style={box, draw=arealteal, line width=1.1pt, fill=arealteal!7, align=center},
    flow/.style={-{Stealth[length=4pt,width=3.4pt]}, line width=0.6pt, draw=arealfg},
    lab/.style={inner sep=1pt, font=\scriptsize, text=arealmuted}]"""]
    # ---------------------------------------------------------------- the construction
    L.append(r"""\node[box, anchor=north west] (A) at (0.05,6.85)
  {\textbf{Assume only}\\[1pt]
   \textbullet\ locality: the density at a point uses $\varphi$'s first derivatives there\\
   \textbullet\ shift symmetry: $\varphi\to\varphi+c$\\
   \textbullet\ rotations and Galilean boosts};
\node[box, anchor=north west] (B) at (0.05,4.7)
  {\textbf{The Lagrangian density is the pressure}\\[1pt]
   $p\big(\partial_\tau\varphi+\tfrac12|\nabla\varphi|^2\big)$, the only invariant; the gas's equation of state is the
   only free function};
\node[elbox, anchor=north west] (C) at (0.05,2.95)
  {\textbf{Euler--Lagrange equation}\\the law the flow obeys};
\node[box, anchor=north west] (D) at (0.05,1.75)
  {\textbf{So a flow has three separate inputs}\\[1pt]
   \textbullet\ the shape, through local geometry: \textcolor{arealteal}{meshes}\\
   \textbullet\ the condition, through the far stream: \textcolor{arealteal}{sensors}\\
   \textbullet\ how they combine, the solution: \textcolor{arealteal}{simulations}};
\draw[flow] (A) -- (B) node[lab, midway, right] {fix the density};
\draw[flow] (B) -- (C) node[lab, midway, right] {make $\iint p$ stationary};
\draw[flow] (C) -- (D);""")
    # ---------------------------------------------------------------- the guarantees
    cols = [5.0, 11.05]
    rows = [6.85, 5.05, 3.25, 1.45]
    L.append(r"\node[anchor=south west, font=\small\bfseries, text=arealfg] at (5.0,6.9) {What the Euler--Lagrange construction guarantees, and what \areals{} does with it};")
    L.append(r"\draw[flow, line width=0.9pt, draw=arealteal] (C.east) -- (C.east -| 4.85,0);")
    L.append(r"\draw[arealteal, line width=0.9pt] (4.85,-0.27) -- (4.85,6.8);")
    for k, (kind, title, body, areals) in enumerate(TILES):
        tx, ty = cols[k // 4], rows[k % 4]
        L.append(f"\\draw[arealgrid, line width=0.6pt, rounded corners=2pt] ({tx:.2f},{ty:.2f}) rectangle ({tx + 5.85:.2f},{ty - 1.72:.2f});")
        L += picto(kind, tx + 0.52, ty - 0.8)
        L.append(f"\\node[anchor=north west, align=left, text width=4.75cm, inner sep=2pt] at ({tx + 1.02:.2f},{ty - 0.04:.2f})\n"
                 f"  {{\\textbf{{{title}}}\\\\\\textcolor{{arealmuted}}{{{body}}}\\\\\\textcolor{{arealteal}}{{\\areals{{}}: {areals}}}}};")
    L.append(r"\end{tikzpicture}")
    open(path, 'w').write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])
