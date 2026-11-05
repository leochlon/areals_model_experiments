# 12 Attention range and field error by region

This experiment compares three checkpoints on the 144 DoubleDelta test cases: 16 wings, nine angles each, read with the whole surface and no sensors. The checkpoints are:

- **Pretrained.** The production geometry checkpoint, placed in the field architecture with its new input columns at zero.
- **Fine-tuned.** The AREALS field model fine-tuned from it.
- **From scratch.** The same transformer trained from scratch on the same 2,160 cases.

**Export.** One GPU export takes 2,048 area-sampled surface queries per case. For each query it records the distance profile of decoder cross-attention: both layers, all four heads, including the log-area bias. It records encoder token self-attention the same way. Distances are in units of the surface's area-weighted RMS radius R, and each profile is compared with attention in proportion to area. The export also stores the full-field predictions of the fine-tuned and from-scratch models.

**Attention analysis.** A CPU analysis turns the attention profiles into Table K.1.

**Error decomposition.** A second CPU analysis splits each model's squared error into two parts:

- its distance from the wing's nine-angle mean field;
- the angle variation about that mean, which no model without sensors or angle can predict.

It then reports the share of the from-scratch error that AREALS removes, over the whole surface and in curvature and pressure-extremity regions.

Paper: Section 4.5 (first paragraph); Appendix K (the protocol, "Fine-tuning adds long-range attention" and "Field error by region"); Table K.1; the measured shares in panel C of Figure K.1 (17% and 40% beyond 0.5R) and in panel C of Figure B.2 (52% against 3% within 0.25R); and the "Attention range and field error by region" row of Table A.1.

## Contents

- `attention_attribution/` (workspace `/workspace/runs/areals-attention-attribution-20261026`): the export and the attention analysis.
  - `run_real.py`: the export entry point. It sets the DoubleDelta manifest and field directory and exports with setting `none` (no sensors), 2,048 queries, seed 0, and predictions from `finetuned` and `scratch`.
  - `export_attention.py`: the exporter. Area-weighted query sampling with a per-case RNG, attention distance profiles, and per-case attribution arrays. Its docstring describes the layout of the exported files.
  - `native_hooks.py`: adapters to the field model. It loads the three checkpoints and checks that the two field models share their architecture and training settings and start from the pretrained checkpoint; captures the post-rotary queries and keys of every attention layer and recomputes attention with the log-area bias; checks the recomputed attention against the fused attention output and the predictions against `field_benchmark.py`; and writes `real/export_audit.json`.
  - `theory_tests.py`: the analysis toolkit. `attention` computes the Table K.1 statistics, with Student-t intervals over wings and exact sign-flip tests; `attribution` gives the per-angle region score; `selftest` runs known-answer checks. It also provides the curvature, quantile and test helpers that `recompute.py` uses.
  - `audit_export.py`: checks completeness, pairing and probabilities for every case, and that each wing's nine no-input predictions are identical. Writes `attention_per_case.csv`, `attention_per_wing.csv`, `attribution_per_wing.csv`, `attribution_coverage_per_case.csv` and `integrity_and_coverage.json`.
- `angle_decomposition/recompute.py` (workspace `/workspace/runs/areals-angle-decomposition-20261026`): the error decomposition for every wing, region and variant. Writes `results.json`, `summary.csv`, `per_wing_decomposition.csv`, `per_wing_contrasts.csv` and `checks.json`.

The model code is not duplicated here. `native_hooks.py` puts `native/` next to itself first on `sys.path`, so the setup places `shared/field_benchmark/native/field_benchmark.py` at `native/field_benchmark.py` and `src/areal_tokens` at `native/areal_tokens/` of the attention study. `recompute.py` imports `theory_tests.py` from its sibling directory `../areals-attention-attribution-20261026`.

## Inputs

`scripts/setup_workspace.py` places both folders. Every step uses the Python environment `/workspace/home/cfd_vector_icl_20261002T160236Z/.venv` with `torch`, `numpy` and `scipy`; the recorded export used torch 2.11.0+cu128, CUDA 12.8 and numpy 2.5.2 on one NVIDIA B200. The export reads:

| Input | Workspace path | Produced by |
|---|---|---|
| DoubleDelta manifest and fields | `/workspace/runs/areals_multicondition_wings_20261023/manifest.json`, `.../fields/` | `06_sensor_completion/single_seed_study` |
| Pretrained | `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/run/best_normal.pt` (production checkpoint, epoch 11) | `01_pretraining/geometry_pretraining` |
| Fine-tuned | `/workspace/runs/areals_multicondition_wings_20261023/bench/models/measurements/field_best.pt` (measurements role: 40 epochs, batch 8, seed 0) | `06_sensor_completion/single_seed_study` |
| From scratch | `/workspace/runs/areals_doubledelta_scratch_20261023/bench/models/areal_scratch/field_best.pt` (same settings, `--scratch`) | `06_sensor_completion/scratch_baseline` |

## Run

```bash
cd /workspace/runs/areals-attention-attribution-20261026
python theory_tests.py selftest                    # optional known-answer checks (CPU)
python export_attention.py --dummy --out demo_attn
python theory_tests.py attention --dir demo_attn/attention_decoder --scratch scratch

# Export (one GPU; real/ must be fresh)
python run_real.py --manifest /workspace/runs/areals_multicondition_wings_20261023/manifest.json \
  --fields /workspace/runs/areals_multicondition_wings_20261023/fields --out real \
  --ckpt pretrained=/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/run/best_normal.pt \
  --ckpt finetuned=/workspace/runs/areals_multicondition_wings_20261023/bench/models/measurements/field_best.pt \
  --ckpt scratch=/workspace/runs/areals_doubledelta_scratch_20261023/bench/models/areal_scratch/field_best.pt

# Attention analysis (CPU)
python theory_tests.py attention --dir real/attention_decoder --scratch scratch --out decoder_results.json
python theory_tests.py attention --dir real/attention_encoder --scratch scratch --out encoder_results.json
python theory_tests.py attribution --dir real/attribution --res 0.25 --out attribution_results.json
python audit_export.py

# Field error by region (CPU)
cd /workspace/runs/areals-angle-decomposition-20261026
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python recompute.py

python -m paper_tables --only attention_analysis --check
```

The export writes `real/attention_decoder/<checkpoint>/<case>.npz`, `real/attention_encoder/<checkpoint>/<case>.npz`, `real/attribution/<case>.npz` (geometry, area weights, the field and the two models' predictions) and `real/export_audit.json`. `recompute.py` asserts that predictions, geometry and area weights are identical across each wing's nine angles, and that its variant `original` reproduces the region statistics and cell gains of `attribution_results.json` to 10⁻¹⁰.

## Outputs → results

| File | Produced as | Used for |
|---|---|---|
| `results/attention_analysis/decoder_results.json` | `decoder_results.json` | Table K.1 decoder rows; the Section 4.5 and Appendix K decoder numbers: 52% and 3% within 0.25R; 17%, 40% and 72% beyond 0.5R; the increase 0.230 [0.220, 0.239] with p = 2^-16; distance ratio 0.25 → 0.41; last layer 12% → 46%; Figures K.1 and B.2 |
| `results/attention_analysis/encoder_results.json` | `encoder_results.json` | Table K.1 encoder rows; encoder 47% → 59%, increase +0.126 [0.123, 0.129], 86% from scratch |
| `results/attention_analysis/region_decomposition.json` | `angle_decomposition/results.json` | Appendix K "Field error by region": variant `nine_angle_mean_fixed_masks`, regions `all` (58.8%), `local` (55.6%) and `global` (61.4%), with positive-wing counts |
| `results/attention_analysis_extra/attention_per_wing.csv` | `attention_per_wing.csv` of `audit_export.py` | "On every wing" for the decoder and encoder increases |
| `results/attention_analysis_extra/export_audit.json` | `real/export_audit.json` | Table A.1 row: training cases, epochs and seed from each checkpoint's embedded metadata, the pretrained checkpoint's epoch, the 144 cases and their empty observation sets; the attention heads in the Appendix K protocol; the export's checks |

The recorded `export_audit.json` also lists file hashes that the current export no longer writes.

## Compute

The export ran on one B200 GPU; the analyses run on CPU (`recompute.py` with 4 BLAS threads). Run times were not recorded.

## Notes

- **What is stored.** Checkpoints, DoubleDelta fields and the exported arrays under `real/` are not included. The tables are rebuilt from the stored summaries.
- **Resumed export.** In the recorded run the export worker slowed sharply after 141 of the 144 cases; the last three cases were exported by a second worker with the same checkpoints and queries, and the two records were merged (block `resume` of `export_audit.json`).
- **Settings.** Long range means beyond 0.5R; curvature is the shape-operator norm at 0.25R. The attribution regions are per-case area-weighted terciles of curvature and pressure extremity, which are proxies for attached flow and vortex or wake regions, not labelled flow regions. `R1` in the coverage files means at least 2% of the scored surface area in both compared regions.
- **The 52% in the text.** Section 4.5, Appendix K and Figure B.2 quote Table K.1's 51.5% (stored 0.51491) at integer precision; the table module checks these text values against the printed table entry.
- **Region variant used.** The paper uses the variant `nine_angle_mean_fixed_masks` of `region_decomposition.json`. Pressure-extremity terciles are fixed on the nine-angle mean field, curvature terciles come from the shape-operator norm at 0.25R, and area weights are used. The other variants are diagnostics. In particular, `original` scores against each angle's own field, as `theory_tests.py attribution` does, so it includes angle variation that no model without sensors or angle can predict.
- **Numerical environment for `recompute.py`.** The recorded run used Python 3.12.14, numpy 2.5.2 and scipy 1.18.1 on Linux, with 4 BLAS threads. Tercile boundaries can tie, so the environment matters. That run reproduced the per-angle statistics of `attribution_results.json` to 10⁻¹⁰. A trial run on a different platform changed the per-angle whole-surface gain by about 2 × 10⁻⁶.
- **Figure K.1.** The figure is a schematic. The widths in panel C are fitted so that the share beyond 0.5R equals the measured decoder shares (17% and 40%). The figure script is `experiments/15_figures/paper_figures/make_longrange.py`.
- **Seeds.** Each field model was trained with one seed.
- **Toolkit documentation.** The docstring of `theory_tests.py` refers to a separate toolkit README, which is not included. The number of attention queries per case (2,048) is set in `run_real.py`; no stored result file records it.
