# 07 GABI baseline on DoubleDelta

GABI (geometric autoencoder priors with approximate Bayesian computation) is run with its authors' released code and scored on the same nine-angle DoubleDelta test cases and saved observation sets as AREALS, with the angle unreported. A single-seed run on GABI's straight-line nearest-neighbour graph supplies the exported training data, the training-set normalisation statistics and the hidden-value check. The reported comparison is a paired ablation trained from scratch for three seeds on each of two graphs: a straight-line 8-nearest-neighbour graph, and an along-mesh graph that measures nearness by shortest distance along the original mesh edges.

Paper: Table D.3 (training budgets; the AREALS column comes from the nine-angle AREALS study); Table D.5 (straight-line and along-mesh GABI, their difference, and along-mesh GABI minus AREALS, each with a 95% wing-bootstrap interval); the "GABI" paragraph of Appendix E (architecture, schedule, ABC settings; 12.7% lower nRMSE and 69% lower force error at 20 sensors); the GABI parts of Appendix H (hidden-value perturbation, six checks with the largest change below 8 × 10⁻⁶ of the field spread; the pilot parity and leak check; a measurement response of about 0.10 spreads at 20 sensors; the recomputation of 864 predictions, within 3.4 × 10⁻¹⁴). The along-mesh GABI numbers also appear in Table 2, Section 4.4, Figure D.1 and Table D.4; `06_sensor_completion` computes those comparisons from this experiment's scores.

## Contents

| Path | What it does |
|---|---|
| `gabi_single_seed/prepare_gabi.py` | `abc_check` (the shared-pool ABC against the authors' `abc_inference` on a synthetic graph), `export` (writes `bench/gabi/{train,val}.pkl`, `config.yml` and `export.json` through `gabi_adapter.py export`), `probe` (GPU memory at batch 100) |
| `gabi_single_seed/resume_train.py` | Continues the first training run from its lowest-validation checkpoint to 455 epochs with the authors' model, loss and normalisation; writes `bench/gabi/trained.json` |
| `gabi_graph_ablation/prepare_graphs.py` | Downloads the 15° surface meshes of the 272 grids from the public DoubleDelta dataset (fixed revision) and writes `graphs/<grid>.npz` with both edge sets (`euclidean`, `surface`), a per-grid record and `graphs/complete.json` |
| `gabi_graph_ablation/prepare.py` | Checks the exported pickles and graphs against the field files, writes `normalization.pt` (training-set statistics of the single-seed checkpoint; no weights), `config.json` and `protocol.json` |
| `gabi_graph_ablation/common.py`, `dataset.py` | Paths and split records; the graph dataset over the exported pickles with the chosen edge set, and the per-epoch example order |
| `gabi_graph_ablation/train.py` | One run: the authors' `GCNGeomAutoencoder` and `compute_loss`, AdamW (1e-3, weight decay 1e-4), exponential decay, batch 100, 455 epochs; keeps `best.pt` at the lowest validation reconstruction + MMD loss. `--pilot` trains 2 epochs |
| `gabi_graph_ablation/check_pilot.py` | Pilot checks: our shared-pool ABC against the authors' `abc_inference` on a real geometry, and a hidden-value scramble; writes `pilot/complete.json` |
| `gabi_graph_ablation/evaluate.py` | GABI-ABC on every saved observation set of a split: 50,000 prior latents decoded once per geometry and shared by its nine angles, 100 accepted, noise 0.01; scored with `field_benchmark.problem_metrics`. Writes `runs/<variant>_seed<seed>/<split>/scores.json` and the predictions |
| `gabi_graph_ablation/analyze.py` | Paired comparison with a 10,000-draw bootstrap over the 16 physical wings (rng 20261028); writes `results/{val,test}_comparison.json` |
| `gabi_graph_ablation/recompute_scores.py` | Recomputes the 20-sensor test nRMSE and force error of all six runs from the saved predictions |
| `gabi_graph_ablation/source/gabi_adapter.py` | Adapter between GABI and the field benchmark: `export`, `train`, `predict`, `check`, the shared-pool `abc_pool` and the k-NN `graph` |
| `gabi_graph_ablation/source/GABI/` | The authors' GABI code at commit `c67f98b1d0e1478f4f6691adbbeea56df6350206`, unmodified, with its MIT `LICENSE` |

## Inputs

`scripts/setup_workspace.py` places `gabi_single_seed/` at `/workspace/runs/gabi_doubledelta_20261023` and `gabi_graph_ablation/` at `/workspace/runs/gabi_graph_ablation_20261028`, with the `native` harness as `field_benchmark.py` in both (in `source/` for the ablation), and copies `gabi_adapter.py` and `GABI/` into the single-seed directory. Supply in addition:

- `src/areal_tokens` at `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/areal_tokens/src`, which `field_benchmark.py` and `gabi_adapter.py` import;
- the nine-angle DoubleDelta data of `06_sensor_completion` at `/workspace/runs/areals_multicondition_wings_20261023` (`manifest.json`, `fields/*.npz`);
- the saved observation sets: copy `bench/config.json` and `bench/observations/{val,test}.{json,npz}` from that study's `bench/` to `/workspace/runs/gabi_doubledelta_20261023/bench/`.

One Python environment (`/workspace/runs/gabi_doubledelta_20261023/.venv` in the recorded run) with torch, torch_geometric, numpy, scipy, PyYAML and python-box serves every step. The recorded run used torch 2.11.0+cu128, torch_geometric 2.8.1, numpy 2.5.3 and CUDA 12.8.

## Run

**Single-seed GABI** (one GPU):

```bash
cd /workspace/runs/gabi_doubledelta_20261023
export PYTHONPATH=/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/areal_tokens/src
python prepare_gabi.py abc_check
python prepare_gabi.py export
python prepare_gabi.py probe
python gabi_adapter.py train --out $PWD/bench/gabi --gabi-repo $PWD/GABI > train.log
python resume_train.py
python gabi_adapter.py check --bench $PWD/bench --out $PWD/bench/gabi --gabi-repo $PWD/GABI \
    --device cuda --samples 2000 --accept 20 > check.log
```

The export writes 2,160 training and 144 validation cases on the straight-line 8-nearest-neighbour graph. Training uses batch 100 and 455 epochs (455 × 22 = 10,010 updates): the export's default target of 10,000 updates (`--iterations 10000`, GABI's published schedule) rounded up to whole epochs. The recorded first run stopped after four epochs; `resume_train.py` reloads its lowest-validation checkpoint (epoch 3) with the normalisation statistics and continues to 455 epochs with a fresh AdamW at the scheduled learning rate, seed 0. The check (Appendix H) uses the defaults `--split val --cars 2 --per-car 3 --tolerance 1e-3`, six checks; in the recorded run it ran after a prediction step (`predict_grouped.py`) that is not included.

**Graph ablation** (one GPU per run). Needs the single-seed outputs `bench/gabi/{train,val}.pkl` and `bench/gabi/runs/GCN_dim_100_resume_20261023T195959Z/{config.yml,trained_models/best.pt}`.

```bash
cd /workspace/runs/gabi_graph_ablation_20261028
export PYTHONPATH=/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/areal_tokens/src
python prepare_graphs.py --workers 8        # CPU
python prepare.py
python train.py --variant euclidean --seed 0 --pilot     # 2 epochs, 44 updates
python train.py --variant surface --seed 0 --pilot
python check_pilot.py
for seed in 0 1 2; do
  for variant in euclidean surface; do python train.py --variant $variant --seed $seed; done
done
for split in val test; do
  for seed in 0 1 2; do
    for variant in euclidean surface; do python evaluate.py --variant $variant --seed $seed --split $split; done
  done
  python analyze.py --split $split
done
python recompute_scores.py > results/independent_verification.json
```

Both graphs of a seed start from the same initial weights and see the examples in the same order, because both are seeded by the training seed alone; the training seed is also the ABC seed. The recorded run evaluated the test split only after the validation comparison was written.

GABI − AREALS (Table D.5, last column) is computed by `06_sensor_completion/three_seeds/analyze.py` from `runs/surface_seed{0,1,2}/{val,test}/scores.json`, with a 10,000-draw wing bootstrap (rng 20261029).

`python -m paper_tables --only gabi_baseline --check` rebuilds Tables D.3 and D.5 and the Appendix E and H numbers.

## Outputs → results

| Result file | Written by | Used for |
|---|---|---|
| `results/sensor_completion/gabi/test_comparison.json` | `analyze.py --split test` | Table D.5: straight-line and along-mesh means, along-mesh s.d. (from the per-seed means), difference and interval; Appendix E: 12.7% and 69% |
| `results/sensor_completion/gabi/config.json` | `prepare.py` | Table D.3: GABI epochs and batch size; Appendix E: layers, width, latent dimension |
| `results/sensor_completion/gabi/export.json` | `prepare_gabi.py export` | Table D.3: GABI updates and exposures (2,160 cases × 455 epochs); 32,000 points, 8 neighbours |
| `results/sensor_completion/gabi/independent_verification.json` | `recompute_scores.py` | Appendix H: 864 predictions of six runs within 3.4 × 10⁻¹⁴; Appendix E: nine angles per pool |
| `results/sensor_completion/nine_angle/test_comparison.json` | `06_sensor_completion/three_seeds/analyze.py` | Table D.5: Measurements ± s.d. and GABI − AREALS with interval |
| `results/sensor_completion/nine_angle/references_test_grouped.json` | `06_sensor_completion/single_seed_study/summarize.py` | Table D.5: shape-only and true-angle references |
| `results/sensor_completion/nine_angle/seed{0,1,2}_training_complete.json` | `06_sensor_completion/three_seeds` | Table D.3: AREALS epochs, updates and seeds |
| `results/sensor_completion_extra/nine_angle/protocol.json` | `06_sensor_completion/three_seeds/prepare.py` | Table D.3: AREALS batch size, 2,160 training cases, 2,048 queries |
| `results/gabi_baseline_extra/gabi_graph_ablation/protocol.json` | `prepare.py` (the recorded copy also lists file hashes and notes) | Appendix E: 50,000 latents, 100 accepted, noise 0.01, three seeds; Table D.5 caption: 8 neighbours |
| `results/gabi_baseline_extra/gabi_graph_ablation/pilot_verification.json` | `check_pilot.py` (`pilot/complete.json`) | Appendix H: pilot ABC parity and hidden-value scramble |
| `results/gabi_baseline_extra/gabi_single_seed/check.log` | `gabi_adapter.py check` | Appendix H: 6/6 checks, largest change 7.64 × 10⁻⁶, response +0.10 at 20 sensors |

## Compute

- Graph-ablation runs: six runs, each on one NVIDIA B200 (12 CPU cores, 160 GB). The first epochs took about 50 s, so a 455-epoch run takes about 6.3 GPU-hours.
- Evaluation: one B200 per run and split, under 6 h.
- Graph building: CPU only, 8 workers.
- Single-seed run: the continuation to 455 epochs took 6 h 17 min on one B200; the prediction job that ran the check took 33 min 31 s.

## Notes

- **`prepare_gabi.py` version.** The export ran with an earlier revision of `prepare_gabi.py` that is not available; the file here is a later revision from the same run directory. `prepare.py` checks every exported field against the field files.
- **Single-seed run.** Its own test scores are not reported. The Appendix H hidden-value check and measurement response come from this model (straight-line graph), with a reduced ABC pool (2,000 prior samples, 20 accepted) on two validation wings. The three-seed pilot repeats the hidden-value check and the ABC parity check on both graphs.
- **Run-directory names.** GABI's training script names its run directory with a random suffix, and `resume_train.py` names its directory by start time. `resume_train.py` expects the first run at `bench/gabi/runs/GCN_dim_100_uid_xNGi` with a `train.log` holding four validation lines, and `prepare.py` reads `bench/gabi/runs/GCN_dim_100_resume_20261023T195959Z`. A rerun must provide these names, for example by symlink.
- **Not included:** trained checkpoints, prediction archives, graphs, normalisation statistics, per-run scores and logs, and the single-seed prediction and scoring scripts (`predict_grouped.py`, `score_grouped.py`), whose outputs the paper does not use.
- **Determinism.** CUDA scatter reductions are not bitwise deterministic, so retraining gives statistically comparable models, not identical numbers.
- **Data source.** The DoubleDelta meshes come from the public dataset `yirens/double-delta-aero` on the Hugging Face Hub at revision `9a1943ee5df9ff65cfcb7d0051fde43f42d4742c`.
