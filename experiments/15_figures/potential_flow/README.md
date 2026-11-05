# AREALS hero figure

`areals_hero.pdf` is the figure (vector text, meshes rasterised at 600 dpi), sized for a full-width ICML figure (6.75 × 2.98 in). `caption.tex` holds a `figure*` block with a draft caption.

## Re-render (seconds)

```
python3 hero_fig.py areals_hero      # reads hero3d.npz, samp_cp.npy, omega.npy
```

Layout, colours, fonts (TeX Gyre Termes + STIX maths) and labels are all in `hero_fig.py`. Termes is taken from your TeX installation and converted once to TrueType in `./fonts` (set `TERMES_DIR` to the folder holding `texgyretermes-*.otf` if it is not found), because matplotlib can only embed TrueType outlines cleanly in a PDF; every font in `areals_hero.pdf` is an embedded TrueType subset, with no Type 3 fonts.

## Recompute the data (about 8 minutes on 2 CPUs)

```
pip install numpy scipy matplotlib trimesh libigl scikit-image fast-simplification
python3 build3d.py mesh geom flow eta anchors mask    # ~1 min, writes hero3d.npz
python3 - <<'PY'                                       # ~6 min: the omega(r) study
import numpy as np, build3d as B, igl
d = B.load(); B.stage_omega(d, n_cases=30); np.save("omega.npy", d["omega"])
V, F, Cp, P = d["V"], d["F"], d["Cp"], d["samp_p"]
_, fi, C = igl.point_mesh_squared_distance(P, V, F)
bc = igl.barycentric_coordinates(C, V[F[fi, 0]], V[F[fi, 1]], V[F[fi, 2]])
np.save("samp_cp.npy", (Cp[F[fi]] * bc).sum(1))
PY
```

## What is computed

- **Mesh**: the 2D fastback profile of `aero2d.py`, extruded with rounded edges and a narrower cabin via an implicit function, marching cubes, Taubin smoothing and decimation. Its centre-plane section stays within 0.016 of the profile.
- **Flow**: non-lifting potential flow, U = (1, 0, 0), by the method of fundamental solutions (2,300 sources, 5,200 collocation points). On a unit sphere the maximum Cp error is 0.037.
- **(b)**: exact unsigned distance (libigl) on the y = 0 plane; slices of η through p; principal curvatures by quadric fitting.
- **(c)**: 256 farthest-point anchors over 80k area-uniform samples, with Voronoi area weights. The attention test draws 4,096 samples with the front half up to 9× denser, Gaussian kernel ℓ = 0.6, and 32 draws per density.
- **(d)**: 30 perturbations h(x) = Σ cos(k·x + φ) along the normal, with wavelengths 0.2–3 and amplitudes 0.003–0.04 R. Normals of the perturbed surface are n − ∇_S h (first order). The field error is the area-weighted RMS of Cp after closest-point mapping. Distances are Hausdorff distances between vertex sets, in position only and in (x/R, nnᵀ), as in the paper's d of Proposition 1, with R = 1.51 the area-weighted r.m.s. radius of the car.
