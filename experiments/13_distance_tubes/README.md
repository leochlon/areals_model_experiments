# 13 Distance tubes: the distance target reproduces topology

The frozen production geometry checkpoint reads eleven shapes: the first eight test-split meshes in identifier order, a unit sphere, a torus with radii 1 and 0.35, and two spheres of radius 0.55 whose centres are 1.3 apart. Each shape is read either whole or with 60% of its area hidden as 32 patches. The decoder's unsigned distance on a 97³ grid is compared with the exact distance to the mesh triangles or the analytic surface. At nine radii from 0.025 to 0.4, each tube {d ≤ r} gives three quantities: the area estimate Vol/(2r), the Betti numbers of the closed-voxel union, and the 26-connected path lengths between six landmarks. Persistence diagrams of the sublevel filtrations are compared by bottleneck distance. For complete inputs the decoder returns the distance to the 2,048 disks kept with the tokens. For partial inputs it returns the learned distance, which is compared with disks built from the visible surface. No model is trained here.

Paper: Section 4.6 ("The distance target reproduces topology"); Appendix J (the protocol, complete inputs and partial inputs, including Tables J.1 and J.2); and the "Distance tubes" row of Table A.1.

## Contents

- `distance_scale/` (workspace `/workspace/runs/areals-distance-scale-evidence-20261026`): exact reference fields, complete-input fields and the tube metrics.
  - `protocol.json`, `cohort.json`: the checkpoint and corpus manifest, the analytic controls, seed 20261026, 16,384 surface samples, the 97³ grid, the nine radii and the probe counts; the cohort is the first eight test-split records by id.
  - `sample_fields.py`: encodes each complete shape and evaluates the exact distance (libigl point–mesh distance, or the closed form) and the decoder on the grid, on 65,536 independent probes and at six farthest-point landmarks. Writes `results/<shape>/{fields,prepared,independent_queries,mesh}.npz`, `metadata.json` and `results/provenance.json`.
  - `measure.py`: the tube library: area estimate, Betti numbers over F2 (GUDHI cubical complex), 26-connected landmark paths, plus an analytic-sphere validation. Run on its own, it writes `results/<shape>/metrics.json` for the decoder's original learned distance; both comparison scripts import it and check their reference metrics against that file.
  - `disk_comparison/disk_fields.py`: complete inputs, the distance to the 2,048 token disks (native surfel readout) on the grid, checked against a replay of the saved decoder output and an independent float64 disk formula. Writes `disk_comparison/results/<shape>/disk_field.npz` and `fields_summary.json`.
  - `disk_comparison/compare.py`: complete inputs, the tube metrics plus persistence diagrams, full and capped at r = 0.4, compared with the exact distance. Writes `summary.json`, `metrics.csv`, `persistence.csv` and `persistence.json`.
- `complete_masked/` (workspace `/workspace/runs/areals-complete-masked-20261026`): complete-input routing in the decoder and the 60%-hidden inputs.
  - `decoder.patch`, `code/areal_tokens/model.py`: `src/areal_tokens` with `decoder.patch` applied. A complete input (`complete = 1`, or no flag) returns the token-disk distance; a partial input (`complete = 0`) keeps the learned distance. Normals, weights and state-dict keys are unchanged.
  - `protocol.json`: the baseline study path, checkpoint, seed, mask (60% of the area as 32 patches) and evaluation grids, radii and persistence cap.
  - `run_fields.py`: hides 60% of each shape and encodes the visible part only, then evaluates the learned and visible-disk distances on the same grid, probes and 16,384 surface samples. Checks that the routing changes no partial-input output and that complete inputs return the native disks. Writes `results/<shape>/{fields,surface_queries,prepared,independent_queries}.npz` and `results/{cases,distance_metrics,regression,inference_provenance}.json`.
  - `compare.py`: the same tube and persistence metrics for partial inputs. Writes `results/{summary.json,metrics.csv,persistence.csv,persistence.json,all_metrics.json}`.
  - `test_complete_distance.py`: unit tests of the routing against the unpatched model, run together with the package's own tests.
  - `audit.py`: an independent recomputation of the areas, lost paths, 66 Betti numbers from Euler characteristics, the visible-only frame and disk centres, and the test results. Writes `results/audit.json` and `distance_bands.json`.

## Inputs

`scripts/setup_workspace.py` copies the two folders and provides `src/areal_tokens` at `distance_scale/vendor/areal_tokens` (put first on `sys.path` there) and at `complete_masked/baseline/areal_tokens` (the unpatched `model.py` for the parity checks of `run_fields.py` and `test_complete_distance.py`), the patched package at `complete_masked/code/areal_tokens`, and the six package tests from `tests/` (every `test_*.py` except `test_paper_tables.py`) at `complete_masked/original_tests`. No `shared/field_benchmark` version is used. Supply in addition:

- the production geometry checkpoint `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/field_finetune/geometry_epoch11.pt` and the corpus manifest `/workspace/runs/objaverse_training_pool/versions/20260927T140652Z/manifest.json` (`01_pretraining`);
- the eight source meshes at the `source_mesh` paths in `cohort.json`. These are absolute paths outside `/workspace`, which `sample_fields.py` loads as written. They are Objaverse-XL GLB files; `source_url` gives each model's page.

Every step uses the Python environment `/workspace/home/cfd_vector_icl_20261002T160236Z/.venv`; the recorded versions are torch 2.11.0+cu128, numpy 2.5.2, scipy 1.18.1, trimesh 5.1.0, libigl 2.6.3 and gudhi 3.13.0.

## Run

```bash
cd /workspace/runs/areals-distance-scale-evidence-20261026
python sample_fields.py                   # exact and decoder fields, complete inputs (one GPU)
python measure.py --workers 4             # reference tube metrics (CPU)
cd disk_comparison
python disk_fields.py                     # token-disk distance, complete inputs (one GPU)
python compare.py                         # Table J.1 and the complete-input bottleneck distances (CPU)

cd /workspace/runs/areals-complete-masked-20261026 && mkdir -p results
python run_fields.py                      # partial inputs (one GPU); reads the outputs of sample_fields.py
python compare.py                         # partial-input bottleneck distances (CPU)
PYTHONPATH=code python -m pytest -q original_tests test_complete_distance.py --junitxml=results/tests.xml
python audit.py

python -m paper_tables --only distance_tubes --check
```

`run_fields.py` imports `sample_fields.py` and `disk_comparison/disk_fields.py` from the `remote_baseline` path in its `protocol.json`, `/workspace/runs/areals-distance-scale-evidence-20261026`. The table module reads the copies listed below.

## Outputs → results

| File in `results/distance_tubes/` | Written by | Used for |
|---|---|---|
| `complete_inputs/summary.json` | `disk_comparison/compare.py` (`disk_comparison/results/summary.json`) | Table J.1. Field `disk`, grid 97, groups `control` (analytic) and `heldout` (meshes). Also the tube volume, connection and path claims for the meshes |
| `complete_inputs/fields_summary.json` | `disk_fields.py` (`disk_comparison/results/fields_summary.json`) | Largest over- and underestimates of the disk distance: 0.07 and 0.001 on the analytic shapes, 0.045 on the meshes |
| `complete_inputs/persistence.csv` | `disk_comparison/compare.py` (`disk_comparison/results/persistence.csv`) | Bottleneck distances 0.063, 0.051 and 0.101. Field `disk`, grid 97, `bottleneck_capped`, maximum over the eleven shapes. The 0.21 and 0.11 claims follow from these |
| `partial_inputs/surface_queries/<shape>.npz` | `run_fields.py` (`results/<shape>/surface_queries.npz`) | Table J.2, the hidden-area medians and the visible-area share. Mesh *k* is the *k*-th held-out identifier in sorted order |
| `partial_inputs/persistence.csv` | `complete_masked/compare.py` (`results/persistence.csv`) | Bottleneck distances: 0.053, 0.090 and 0.119 against 0.200, 0.110 and 0.190. Group `heldout`, grid 97, `bottleneck_capped`; field `full` is AREALS and field `disk` is the visible disks |
| `partial_inputs/summary.json` | `complete_masked/compare.py` (`results/summary.json`) | Partial-input tube maxima (not printed in the paper) |

The protocol numbers of Appendix J and the Table A.1 row are checked against these records in `results/distance_tubes_extra/`:

| File in `results/distance_tubes_extra/` | Source | Used for |
|---|---|---|
| `complete_inputs/metadata/<shape>.json` | `sample_fields.py` (`results/<shape>/metadata.json`) | Analytic dimensions (unit sphere, torus radii 1 and 0.35, two spheres of radius 0.55 with centres 1.3 apart), 0.6 grid padding, six landmarks per shape |
| `complete_inputs/protocol.json`, `complete_inputs/metrics_protocol.json` | recorded copies of `distance_scale/protocol.json` and of the persistence settings of `disk_comparison/compare.py` | At most 4,096 encoder points, 2,048 surfels, the seed, the persistence cap r = 0.4 |
| `partial_inputs/protocol.json`, `partial_inputs/cases.json` | recorded copy of `complete_masked/protocol.json`; `run_fields.py` (`results/cases.json`) | 60% hidden as 32 patches, the persistence cap, realised disk centres (2,048) and encoder points per masked shape |

The recorded protocol copies also carry the checkpoint and manifest hashes and design notes, which the repository versions no longer list.

## Compute

| Step | Hardware | Recorded time |
|---|---|---|
| `sample_fields.py` | one NVIDIA B200 | 5 min 58 s |
| `measure.py` | CPU | 26 s |
| `disk_fields.py` | one NVIDIA B200 | about 20 s of per-shape compute |
| `compare.py` (both) | CPU, 4 worker processes | not recorded |
| `run_fields.py` | one NVIDIA B200 | about 5.4 min of per-shape compute |
| tests | CPU | 57 s (59 tests passed) |

## Notes

- **Method details.**
  - Grid: an isotropic cube containing the normalised mesh bounding box with at least 0.6 RMS radii of padding on each side; samples are the centres of closed voxels. The coarse grid (49³) takes every second sample.
  - Distances are in units of the preprocessor's area-weighted point-cloud RMS radius. Distance errors are maxima over the grid and the independent probes, sampled rather than certified bounds.
  - Paths: six surface landmarks by farthest-point sampling, all 15 pairs, attached to their nearest grid centre and held fixed across radii and fields; Euclidean-weighted 26-neighbour shortest paths in the occupied voxels. A missing learned path where the reference path is finite counts as an infinite error; false connections are counted separately. A first version attached landmarks only to grid centres within half the smallest radius of the surface, which moved one landmark by 1.189 RMS radii; the reported metrics use the nearest-centre rule. Tubes that touch the domain boundary are flagged.
  - Persistence: ordinary H0, H1 and H2 over F2 on the same cubical grid, full and with the common cap min(d, 0.4), compared by GUDHI bottleneck distance with additive tolerance 10⁻⁶ (`TOL` and `CAP` in both `compare.py` scripts).
  - Partial inputs: one fixed mask per shape (`visible_mask` with `stable_seed(seed, id + ':mask', 0)`). The frame and the symmetry candidates come from the visible surface only, and the output distances are converted to the complete shape's RMS radius.
- **No aggregation script for Table J.2.** No script was saved for the Table J.2 coverage, the hidden-area medians or the visible-area share. `paper_tables/distance_tubes.py` recomputes them from the saved surface samples. Coverage is the area-weighted share of hidden (or visible) samples whose distance is strictly below 0.05 or 0.1 RMS radii, and the median is area-weighted. This recomputation reproduces all 44 cells of Table J.2.
- **The paper's prose bounds.** These are the stored maxima rounded up at the printed precision; lower bounds are rounded down. For example, "area within 1.4%" covers a stored maximum of 1.336%, which Table J.1 prints as 1.3. The module checks them that way.
- **Realised hidden share.** The mask requests 60% of the area. The realised hidden share is 53.8–65.7% per shape, with a mean of 60.1%.
- **Cohort check.** `sample_fields.py` checks that `cohort.json` equals the first eight test-split records of the manifest and that none of their families is in the training split. The `path` entries in `cohort.json` have been rewritten to workspace paths.
- **Large files.** Per-shape fields, diagrams and logs are not included because of their size. Of the per-shape arrays, only the surface samples behind Table J.2 are in `results/`.
- **Not included.** The first path-measurement version, and the timing benchmark, diagnostic coverage script, plotting and report scripts of the original study directory, which no paper number depends on.
- **Optional `dependencies/` directory.** `measure.py` and both `compare.py` scripts put `distance_scale/dependencies` first on `sys.path`, for a local install of GUDHI. It is optional when `gudhi` is installed in the environment.
- **Inputs not included.** The checkpoint, the manifest and the source meshes are not part of this repository.
