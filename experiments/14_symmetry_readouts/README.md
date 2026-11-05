# 14 Surfel and symmetry readouts

This experiment covers the surfel and symmetry readouts of Section 3.2 ("Surfel and symmetry readouts"), Appendix I, Table I.1, Figure I.1 and the "Symmetry" row of Table A.1. Both readouts are options of the `areal_tokens` package in `src/` (version 0.6.0). The surfel readout reads the tokens, with extra farthest-point surfels carried in the packet, as disks. A learned gate blends the disks with the learned heads (`"normal_prior": "surfels"`, `"distance_prior": "surfels"`, `surfels_per_token`, `input_condition`). The symmetry-map prior (`"symmetry_prior": "maps"`) finds rigid symmetries of the visible input by transformation-space voting. It carries the surfel readout at each query's image back to the query, and the gate weighs it against the direct readout and the heads. Table I.1 compares two package presets, `configs/surfels.json` ("Surfels, 8 per token") and `configs/symmetry.json` ("+ symmetry maps"). Each is trained on 170 meshes and evaluated on 41 held-out meshes, with one seed. Training and evaluation use the package command line. This folder holds only the generator of Figure I.1.

## Contents

- `method_figures/gen/make_method_figs.py`: writes the TikZ source of Figure I.1 (`priors.tex`: (a) surfel readout and gate, (b) symmetry maps). It also writes Figure E.1 (`masked_pretraining.tex`) and `signfree.tex`, which the paper does not use. Its docstring and section comments use other figure numbers (2, 3 and 5).
- Used from the package, not duplicated here:
  - `src/areal_tokens/symmetry.py`: voting (`vote_symmetries`, `symmetry_candidates`);
  - `src/areal_tokens/data.py`: per-sample symmetry maps in the loader;
  - `src/areal_tokens/model.py`: surfel readout, gate and symmetry-map blend; the gate starts nearly closed;
  - `src/areal_tokens/evaluate.py`, `src/areal_tokens/cli.py`;
  - `configs/surfels.json`, `configs/symmetry.json`;
  - `tests/test_symmetry_prior.py`;
  - `docs/areal_tokens.md`, sections "Normals" and "Symmetry".

## Run

Run the commands from the repository root after `pip install ".[mesh]"`. No `shared/field_benchmark` version and no upstream checkpoint is used: both arms train from random weights (seed 17 in both presets). No workspace link is needed.

1. **Unit tests.** Among other things, they check that a carried readout recovers a hidden repeated part and that the gate starts nearly closed:

   ```bash
   python -m pytest -q tests/test_symmetry_prior.py
   ```

2. **Data.** Write a mesh manifest `meshes.json` in the package format (see "Data" in `docs/areal_tokens.md`) with `train`, `val` and `test` records. `areal train` scores the `val` split after every epoch. Then prepare and check the meshes:

   ```bash
   areal prepare-mesh --manifest meshes.json --out prepared --points 16384 --queries 32768
   areal validate     --manifest prepared/manifest.json
   ```

3. **Train both arms.**

   ```bash
   areal train --manifest prepared/manifest.json --config configs/surfels.json  --out runs/surfels  --device cuda
   areal train --manifest prepared/manifest.json --config configs/symmetry.json --out runs/symmetry --device cuda
   ```

   The presets plan 100 epochs, but the paper's arms trained for 24 (`docs/areal_tokens.md`). There are two ways to get 24 epochs:
   - `--stop-after-epochs 24` stops at epoch 24 and keeps the 100-epoch learning-rate schedule;
   - a copy of the preset with `"epochs": 24` shortens the schedule itself.

   Which of the two the original runs used is not recorded. Each run writes `best.pt` (lowest validation distance nRMSE), `last.pt` and `metrics.jsonl`.

4. **Evaluate on the held-out split, with full input and with 60% of the surface hidden.**

   ```bash
   areal evaluate --checkpoint runs/surfels/best.pt  --manifest prepared/manifest.json --out runs/surfels/test.json  --masks 0 0.6 --device cuda
   areal evaluate --checkpoint runs/symmetry/best.pt --manifest prepared/manifest.json --out runs/symmetry/test.json --masks 0 0.6 --device cuda
   ```

   In each report, `settings` follows the order of `--masks`:
   - **Full** in Table I.1 is `settings[0].aggregate.normal_degrees`.
   - **All** is `settings[1].aggregate.normal_degrees`, pooled over query points.
   - `settings[1].normal_breakdown.by_region` gives the pooled hidden and visible normal errors.

   The package does not compute the remaining quantities of Table I.1 and Appendix I:
   - **Hidden**: hidden-region normal error averaged over meshes, and the count of meshes that improve (32 of 41);
   - **Repeated parts**: the error on hidden points that have a visible copy;
   - the gate's weight on the symmetry maps at those points, 61%. The model returns the per-query weight as `symmetry_weight` among its decoder outputs;
   - the maps-removed ablation;
   - the 63% share of hidden points that map onto visible input when symmetries are detected on the complete shapes.

5. **Figure I.1.** The script needs only numpy:

   ```bash
   python experiments/14_symmetry_readouts/method_figures/gen/make_method_figs.py
   ```

   It writes `priors.tex`, `masked_pretraining.tex` and `signfree.tex` into `method_figures/`. The `priors.tex` it writes is byte-identical to the paper's Figure I.1 source.

## Outputs → results

`results/` holds no file for this experiment. `python -m paper_tables --only symmetry_readouts` prints the layout of Table I.1 and lists the 22 printed numbers of this study, each with a check:
- the eight cells of Table I.1;
- the Section 3.2 and Appendix I figures (10.9, 6.9, 17.6, 15.7, 32 of 41, 61%, 63%);
- the protocol of the Table I.1 caption and the Table A.1 "Symmetry" row (170, 41, one seed, 60%, 8 per token).

No stored value backs these numbers, so every check reports a mismatch (rebuilt `nan`).

## Compute

No compute record is available for the 41-mesh runs. `docs/areal_tokens.md` gives relative costs:
- symmetry voting takes 0.2 to 0.5 s per sample in the data loader, so set `train.workers` (4 in both presets);
- a training step of the `symmetry` preset takes about 1.8 times as long as one of the `surfels` preset.

The figure script runs in seconds on a CPU.

## Notes

- **Not included.** The 41-mesh evaluation is not part of this supplement. The missing pieces are:
  - the mesh manifest (170 training and 41 held-out meshes);
  - training logs and checkpoints of both arms;
  - per-mesh outputs;
  - the analysis behind the Hidden and Repeated-parts columns, 32 of 41, 61%, the maps-removed ablation and 63%.

  Table I.1 and the Appendix I numbers therefore cannot be rebuilt from `results/`. The same values are stated in `docs/areal_tokens.md` ("Normals" and "Symmetry").
- **What the documentation states.**
  - For the `surfels` preset, `docs/areal_tokens.md` gives 17.4° as the pooled hidden-region error, the quantity `evaluate` reports. Table I.1's Hidden column (17.6°) is the mean over meshes.
  - Hidden points with a repeated part are 21% of all hidden points.
  - For the `symmetry` preset, the documentation says full-input accuracy is unchanged and gives no separate number.
- **Building Figure I.1.** The TikZ output uses colours and a macro defined in the paper's preamble:

  ```latex
  \usepackage{tikz}
  \usetikzlibrary{arrows.meta,calc}
  \definecolor{arealteal}{HTML}{0A7A72}
  \definecolor{arealwarn}{HTML}{B4541A}
  \definecolor{arealmuted}{HTML}{5B676C}
  \definecolor{arealfg}{HTML}{1B2326}
  \definecolor{arealgrid}{HTML}{E3E7E8}
  \newcommand{\nnT}{nn^{\top}}
  ```
