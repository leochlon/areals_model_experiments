# 15 Computed and schematic figures

This folder holds the code and data behind the paper's figures that are not plots of a training experiment. The main part is the potential-flow example on one car mesh. Every field in it is computed: the mesh, a non-lifting potential flow solved by the method of fundamental solutions (MFS), exact distances, farthest-point anchors with area weights, and 30 perturbed shapes with the flow re-solved on each. The example supports these parts of the paper:

- Figure 2, panels (a)–(c), including the field-error-to-distance ratios in the caption
- Figure B.3, the wall coordinates and the half squared distance
- Figure G.1, anchors as a quadrature, and the Appendix G sentence on the car of Figure 2 (24% drift against 0.7%)
- the "Computed example" paragraph of Appendix B

The folder also holds the generators of the schematic Figures B.1, B.2 and K.1, and the pgfplots sources of Figures 1 and D.1.

## Contents

| Path | Purpose |
|---|---|
| `potential_flow/build3d.py` | Builds the data in stages, each cached in `hero3d.npz`: `mesh`, `geom`, `flow`, `eta`, `anchors`, `mask`. `stage_omega` is the perturbed-shape study behind Figure 2(c) |
| `potential_flow/aero2d.py` | The two-dimensional fastback profile (Chaikin corner cutting and a periodic spline). It is extruded into the car mesh and also gives the centre-plane section frame |
| `potential_flow/render3d.py` | Orthographic mesh renderer used by `hero_fig.py` |
| `potential_flow/hero_fig.py` | Draws the figure from `hero3d.npz`, `samp_cp.npy` and `omega.npy`. It also computes the Figure B.3 and Figure G.1 numbers as it draws (see Notes). It imports `render3d.View` and `build3d.fps`, so it needs the same packages as `build3d.py` |
| `potential_flow/README.md` | The original recipe, including the inline step that writes `omega.npy` and `samp_cp.npy` |
| `potential_flow/hero3d.npz` | Stored stage outputs: mesh (31,780 vertices), normals, curvatures, R, MFS solution and Cp, unit-sphere check, centre-plane distance grid and section, 80,000 samples, 256 anchors and weights, 60% mask |
| `potential_flow/omega.npy` | 30 × 5 array, one row per perturbed shape: field error, distance in x/R, distance in (x/R, nnᵀ), wavelength, amplitude / R |
| `potential_flow/samp_cp.npy` | Cp at the 80,000 samples, used by the Figure G.1 attention test |
| `paper_figures/make_guarantees.py` | Writes the TikZ source of Figure B.1 |
| `paper_figures/make_locality.py` | Writes the TikZ source of Figure B.2 |
| `paper_figures/make_longrange.py` | Writes the TikZ source of Figure K.1 |
| `paper_figures/efficiency.tex` | pgfplots source of Figure 1, with its data tables embedded |
| `paper_figures/doubledelta.tex` | pgfplots source of Figure D.1, with its coordinates written into the file |
| `../../paper_tables/figures.py` | Rebuilds the numbers of Figures 2, B.3 and G.1 and of Appendix B from the stored arrays, and checks the 81.5% label of Figures B.1 and K.1 |

Figures E.1 and I.1 come from `experiments/14_symmetry_readouts/method_figures/gen/make_method_figs.py`, which is not repeated here.

## Run

Everything runs on a CPU, and no workspace setup is needed. Every script resolves its files relative to its own directory. No step uses `areal_tokens`, a `shared/field_benchmark` version or a checkpoint.

1. **Install the dependencies**, as given in the original recipe:
   ```
   pip install numpy scipy matplotlib trimesh libigl scikit-image fast-simplification
   ```
   `hero_fig.py` also uses `fonttools` to convert TeX Gyre Termes to TrueType when that font is installed (see Notes).

2. **Recompute the potential-flow data (optional).** The stages write `hero3d.npz`, `omega.npy` and `samp_cp.npy` next to the scripts and overwrite the stored copies, so work on a copy of the folder:
   ```
   cp -r experiments/15_figures/potential_flow potential_flow_rerun && cd potential_flow_rerun
   rm hero3d.npz omega.npy samp_cp.npy
   python3 build3d.py mesh geom flow eta anchors mask
   python3 - <<'PY'
   import numpy as np, build3d as B, igl
   d = B.load(); B.stage_omega(d, n_cases=30); np.save("omega.npy", d["omega"])
   V, F, Cp, P = d["V"], d["F"], d["Cp"], d["samp_p"]
   _, fi, C = igl.point_mesh_squared_distance(P, V, F)
   bc = igl.barycentric_coordinates(C, V[F[fi, 0]], V[F[fi, 1]], V[F[fi, 2]])
   np.save("samp_cp.npy", (Cp[F[fi]] * bc).sum(1))
   PY
   ```
   The seeds are fixed in `build3d.py`: farthest-point sampling seeds 1, 2 and 5, surface sampling seed 3, mask seed 11 and perturbation seed 21. On the stored arrays, `build3d.py`'s own code reproduces several stored outputs exactly: the anchors, their cells and weights, and the MFS collocation and source indices. Cp recomputed from the stored MFS solution agrees to within 2e-15. `stage_omega` reproduces the wavelength, amplitude and both distance columns of `omega.npy` to within 2e-16. The mesh stage depends on marching cubes, smoothing and decimation in `scikit-image`, `trimesh` and `fast-simplification`, so a fresh mesh can differ slightly between library versions.

3. **Draw the figure**:
   ```
   cd experiments/15_figures/potential_flow && python3 hero_fig.py areals_hero
   ```
   This writes `areals_hero.pdf` and `areals_hero.png`, a four-panel figure that combines Figures 2, B.3 and G.1. It also prints the Figure G.1 values (surface integral, and both attention variants at 9×) and the slope L of Figure 2(c).

4. **Write the schematic figures.** These scripts need only the Python standard library:
   ```
   cd experiments/15_figures/paper_figures
   python3 make_guarantees.py guarantees.tex
   python3 make_locality.py locality.tex
   python3 make_longrange.py longrange.tex
   ```
   The outputs are byte-identical to the paper's sources.

5. **Rebuild the figure numbers**, from the repository root:
   ```
   python -m paper_tables --only figures --check
   ```

### Building the TikZ and pgfplots sources

The `.tex` files are inputs to the paper (`\input{figures/...}`). They use these definitions from its preamble (`\gapClosed` is one of the paper's result macros):

```latex
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots,fillbetween}
\usepackage[capitalize,noabbrev]{cleveref}
\definecolor{arealteal}{HTML}{0A7A72}
\definecolor{arealwarn}{HTML}{B4541A}
\definecolor{arealmuted}{HTML}{5B676C}
\definecolor{arealfg}{HTML}{1B2326}
\definecolor{arealgrid}{HTML}{E3E7E8}
\definecolor{arealrec}{HTML}{8AC4BB}
\newcommand{\areals}{AREALS}
\newcommand{\gapClosed}{90.7\%}
```

`make_guarantees.py` writes cross-references (`\cref{fig:locality}`, `\cref{fig:attention}`) that resolve only inside the paper.

## Outputs → results

This experiment writes nothing to `results/`. The table module reads the stored arrays in `potential_flow/`, and one file of the density-shift study:

| File | Used for |
|---|---|
| `potential_flow/hero3d.npz` | Appendix B: 31.8k vertices, car length 4.5, section within 0.016, 2,300 sources and 5,200 collocation points, unit-sphere Cp error 0.037, R = 1.51, 256 anchors over 80,000 samples. Figure B.3: 1 + tκ = 0.81 and 1.26 at t = 0.21 (from `sec_p`, `sec_s`, `sec_k`) |
| `potential_flow/omega.npy` | Figure 2(c) and Appendix B: 30 shapes, five per wavelength from 0.2 to 3, amplitudes 0.003–0.04R. Ratios of field error to distance: 1.76 (wavelength 3) to 24.0 (wavelength 0.2) in position alone; 0.45–0.57 per wavelength with normal lines, with a largest value of 0.567 |
| `potential_flow/samp_cp.npy` (with `hero3d.npz`) | Figure G.1 and Appendix G: ordinary attention drifts by 24% at the ninefold shift, and log-area attention stays within 0.7% of the surface integral |
| `results/architecture_ablations/density_shift/independent_grouped.json` | Figures B.1 and K.1: the label "81.5% less drift" (1 − log-area drift / ordinary drift) |

## Compute

From the original recipe: about 8 minutes on 2 CPUs to recompute the data. Of this, about 1 minute is the stages in step 2 and about 6 minutes is the 30-shape study. Drawing the figure takes seconds, and so do the schematic generators. The table module runs in under a second.

## Notes

- **Final renderer not available.** The three figure files in the paper (Figures 2, B.3 and G.1) were drawn by a later, edited version of `hero_fig.py` that is not available. That version split the four-panel figure into three figures and changed labels and ticks. It plotted the same three arrays. Every number printed in those figures, their captions and Appendix B recomputes from the arrays with the panel logic of the `hero_fig.py` included here. `paper_tables/figures.py` repeats that logic without plotting.
- **Numbers computed while drawing.** No stored file holds the Figure B.3 and Figure G.1 numbers. `hero_fig.py` computes them as it draws: `panel_a` gives 1 + 0.21κ at the most concave and most convex section points round the windscreen, and `panel_c` runs the 32 random draws per density with `default_rng(0)`.
- **Scope of the 0.7%.** The 0.7% in Appendix G is the deviation at the ninefold shift (0.71%). At intermediate densities, the mean deviation of log-area attention over 32 draws reaches 1.22% at 2× and 1.32% at 8×. The table module lists all ten densities.
- **Settings, not results.** The following are settings in `build3d.py` and `hero_fig.py`, so the table module does not check them: t = 0.21 in Figure B.3; and 4,096 samples, the ninefold density ratio, kernel length 0.6 and 32 draws per density in Figure G.1.
- **Figures 1 and D.1.** Their coordinates are written directly into `efficiency.tex` and `doubledelta.tex`; the script that wrote the Figure 1 data tables is not included. The stored results that hold these coordinates are `results/simulation_efficiency/` (Figure 1) and `results/sensor_completion/nine_angle/` (Figure D.1: `test_comparison.json` for the AREALS and GABI curves, `references_test_grouped.json` for the geometry-only and true-angle curves and the G label). The modules `simulation_efficiency` and `sensor_completion` check the plotted values against them.
- **Measured values in the schematics.** Some labels in the schematics are measured values that were typed into the generators:
  - 52% of decoder attention within 0.25R against 3% (Figure B.2);
  - the 17% and 40% attention shares to which `make_longrange.py` fits its panel C widths (Figure K.1);
  - "81.5% less drift" (Figure B.1) and "81.5% less" (Figure K.1, panel B).
  
  The first two come from Appendix K and are checked in `attention_analysis`. The 81.5% is the density-shift result of Section 4.6 and Appendix G, stored in `results/architecture_ablations/density_shift/`; `architecture_ablations` checks it in Section 4.6, and `paper_tables/figures.py` checks it in Figures B.1 and K.1. Everything else drawn in the schematics is computed by the generators for illustration. Each script's docstring says what its line widths mean.
- **Files not included.** `build3d.py` also adds a sibling `gifs` folder to `sys.path`; it is not needed, because `aero2d.py` is in this folder. `potential_flow/README.md` mentions `areals_hero.pdf` and `caption.tex`, which are not included: the first is an output and the second a draft caption of the four-panel figure.
- **Fonts.** `hero_fig.py` looks for TeX Gyre Termes in common TeX locations, or in `TERMES_DIR`, and converts it to TrueType in `./fonts`. Without it, matplotlib substitutes a serif font. Only the appearance changes, not the numbers.
