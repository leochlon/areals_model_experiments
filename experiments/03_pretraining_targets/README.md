# 03 Matched pretraining targets

Which geometry targets carry the transfer to pressure prediction? Three priors are pretrained from fresh weights on the same 1,668 Objaverse-XL meshes for 10,626 updates:

- AREALS, trained on the distance and normal line (d, nn⊤);
- an AREALS-sized query network that adds the off-surface Hessian;
- a patch encoder trained on the chart coefficients.

Each prior is fine-tuned on WindsorML pressure with 18 to 274 training cars and compared with its own architecture trained from scratch. Learning rates are selected on validation.

The folder also holds:

- the 385-mesh patch pilot;
- a check of the AREALS checkpoint saved at 6,500 updates;
- the coefficient prior: the same coefficients trained inside AREALS's own encoder and query decoder on 14,304 meshes. It is compared with scratch, and with the same network trained on d and nn⊤ alone, over three pretraining seeds.

Paper: Section 4.3, Table 1, Tables M.1–M.3, Appendix M ("Patch pilot", "Scope", "Coefficients inside AREALS", "Pretraining seeds"), and the Table A.1 rows "Matched pretraining targets", "Patch pilot" and "Coefficients inside AREALS".

## Contents

| Folder | Workspace directory | What it does |
|---|---|---|
| `cohort_and_windsor_data/` | `runs/objaverse_target_comparison_20261030` | Prepares the Objaverse-XL target packs (`prepare_corpus.py`) and the WindsorML patch packs (`prepare_windsor.py`), and indexes both (`source/study_control.py`) |
| `matched_priors/` | `runs/matched_priors_windsor_20261030` | The three-prior study. `source/` holds the matched cohort (`matched_control.py`), pretraining (`pretrain.py`), geometry scores (`evaluate_geometry.py`) and the fine-tuning driver (`field_study.py`: `tune`, `select`, `followup`, `score`, `report`). It also holds the scripts shared with the other folders. `source/native/areal_tokens/` holds the two package files that differ from `src/areal_tokens`: the sign-free nn⊤ normal input. `diagnostics/areals_interim_20261031/` fine-tunes from the 6,500-update checkpoint (validation only) |
| `patch_pilot/` | `runs/patch385_windsor_20261030` | The 385-mesh patch prior and its 274-car WindsorML comparison (`scripts/prepare_pilot.py`, `scripts/pressure_pilot.py`), run through the patch-pressure package `patch_pressure/` |
| `coefficient_prior/` | `runs/shared_query_controlled_20261031` | The coefficient prior. It holds the shared-query package `AREALS_shared_query/`, the 14,304-mesh cohort (`scripts/make_cohort.py`), the world-size migrations (`scripts32/`, `scripts8/`), the 90 WindsorML fits (`scripts_windsor_parallel/run.py`), scoring (`scripts32/pressure_task.py`), the report (`scripts32/report.py`) and the comparison with the patch prior (`checks/`) |
| `coefficient_replicates/` | `runs/curvature_controls_20261031` | The d, nn⊤-only priors (pretraining seeds 17 and 29) and the fine-tuning-seed-0 fits of both priors. These give the right half of Table M.3 and the pretraining-seed spread |

`scripts/setup_workspace.py` copies the shared scripts, the two variant package files and `AREALS_shared_query/` into every study directory that uses them.

## Inputs

- **Environment.** Every `env.sh` sets the Python environment and `PYTHONPATH`. Preparation also needs trimesh, pymeshlab, open3d, igl, numba and h5py (see `AREALS_shared_query/requirements.txt` and `patch_pressure/requirements.txt`). `cohort_and_windsor_data/env.sh` puts the geometry-processing packages of `/workspace/runs/areals_drivaerml_geometry_physics_20261030/vendor` and `/workspace/runs/objaverse_target_comparison_20261030/vendor` on `PYTHONPATH`; neither is included.
- **WindsorML.** Pressure fields and geometry caches come from `/workspace/runs/areals_all_simulation_efficiency_20261028/datasets/windsor`, prepared by `02_simulation_efficiency`.
- **Objaverse-XL.** `prepare_corpus.py objects` reads the pool manifest from `01_pretraining` and `/workspace/runs/areals_existing_data_expansion_20261025/larger_pretraining/geometry_manifest.json`.

No step uses `shared/field_benchmark` or the production checkpoint: every prior here is trained from fresh weights.

## Run

Run each step after the previous one has finished. Commands with `--task` run once per task index, so the ranges below can be spread over GPUs. Every block starts with `cd <workspace directory>; source env.sh`, and `torchrun` stands for `$PYTHON -m torch.distributed.run`.

1. **Target packs and WindsorML index** (`cohort_and_windsor_data/`):

   ```bash
   $PYTHON source/prepare_corpus.py objects                                      # writes protocol/objects.json
   for k in $(seq 0 511); do $PYTHON source/prepare_corpus.py pack --pack $k --workers 4 --out prepared; done
   for k in $(seq 0 42); do $PYTHON source/prepare_windsor.py --pack $k; done
   $PYTHON source/study_control.py windsor                                       # writes windsor/index.json
   ```

2. **Matched cohort and the three priors** (`matched_priors/`). The cohort contains every object that passed both preparations in the preparation logs: 1,668 training, 85 validation and 92 test meshes.

   ```bash
   $PYTHON source/matched_control.py snapshot
   $PYTHON source/matched_control.py aggregate
   for arm in patch areals hessian; do
     $PYTHON -m torch.distributed.run --standalone --nproc-per-node=4 source/pretrain.py --arm $arm \
         --out pretraining/$arm --steps 10626 --batch 64 --workers 4 --validation-objects 85
     $PYTHON source/evaluate_geometry.py --arm $arm                             # results/geometry_<arm>.json
   done
   ```

3. **WindsorML fine-tuning, selection and testing** (`matched_priors/`, one GPU per task):

   ```bash
   for i in $(seq 0 89); do $PYTHON source/field_study.py tune --task $i; done      # seed 0, three encoder rates
   $PYTHON source/field_study.py select                                              # field/selections.json
   for i in $(seq 0 59); do $PYTHON source/field_study.py followup --task $i; done  # seeds 1 and 2 at the selected rate
   for i in $(seq 0 89); do $PYTHON source/field_study.py score --task $i; done     # field/test/NNN.json
   $PYTHON source/field_study.py report                                              # results/comparison.json
   ```

4. **Checkpoint at 6,500 updates.** Copy `pretraining/areals/last.pt` from the time it held step 6,500 to `diagnostics/areals_interim_20261031/geometry_step_6500.pt`. Then run:

   ```bash
   for i in 0 1 2; do
     $PYTHON source/field_study.py tune --task $((30 + i)) --checkpoint diagnostics/areals_interim_20261031/geometry_step_6500.pt \
         --out diagnostics/areals_interim_20261031/pressure/lr$i --steps 8000 --validate-every 250
   done
   python diagnostics/areals_interim_20261031/summarize.py                         # validation_summary.json
   ```

5. **Patch pilot** (`patch_pilot/`). This step needs the 413-object pilot list `protocol/cohort.json` (385 training, 12 validation and 16 test meshes).

   ```bash
   for i in $(seq 0 412); do $PYTHON scripts/prepare_pilot.py one --index $i; done
   $PYTHON scripts/prepare_pilot.py consolidate
   $PYTHON scripts/prepare_pilot.py windsor                                        # windsor/cases.json
   $PYTHON -m torch.distributed.run --standalone --nproc-per-node=4 source/pretrain.py --arm patch \
       --out pretraining/patch --steps 10626 --batch 64 --workers 2 --validation-objects 12 --validate-every 250
   $PYTHON scripts/pressure_pilot.py export
   $PYTHON scripts/pressure_pilot.py register
   $PYTHON scripts/pressure_pilot.py real-check
   for i in $(seq 0 5); do $PYTHON scripts/pressure_pilot.py trial --index $i; done  # two initialisations x three rates
   $PYTHON scripts/pressure_pilot.py finish                                           # pressure/summary.json
   ```

6. **Coefficient prior** (`coefficient_prior/`). Seed 17 trained on 4 GPUs to update 500, on 32 GPUs (four nodes of eight) to update 4,000, and on 8 GPUs to the end, always at a global batch of 64. `scripts32/migrate.py` and `scripts8/migrate.py` convert the checkpoint between world sizes. On a single node, the same command with `--resume` continues at any world size, but the run is then not bitwise identical. `PRETRAIN` below stands for `AREALS_shared_query/pretrain.py --objects protocol/objects.json --index prepared/index.json --config AREALS_shared_query/native_config.json --mode operator --steps 10626 --batch-size 64 --validation-objects 128 --validate-every 500 --device cuda --amp fp32`.

   ```bash
   $PYTHON scripts/make_cohort.py                                       # 14,304 / 637 / 632 meshes
   torchrun --standalone --nproc-per-node=4 $PRETRAIN --workers 4 --out pretraining/operator   # stop once last.pt holds step 500
   $PYTHON scripts32/migrate.py
   torchrun --nnodes=4 --nproc-per-node=8 --rdzv-endpoint=<host:port> $PRETRAIN --workers 1 --out pretraining/operator32 --resume   # stop at step 4,000
   $PYTHON scripts8/migrate.py
   torchrun --standalone --nproc-per-node=8 $PRETRAIN --workers 4 --out pretraining/operator32 --resume
   for seed in 29 43; do
     torchrun --standalone --nproc-per-node=4 $PRETRAIN --workers 4 --threads 1 --seed $seed --lr 3e-4 --weight-decay 0.05 \
         --gradient-clip 1 --aux-weight 0.1 --patches 8 --patch-queries 64 --hessian-queries 256 --out pretraining/operator_seed$seed
   done
   for t in $(seq 0 14); do $PYTHON scripts_windsor_parallel/run.py prepare --mode final --task $t; done
   for t in $(seq 0 89); do $PYTHON scripts_windsor_parallel/run.py candidate --mode final --task $t; done   # one GPU each
   for t in $(seq 0 14); do $PYTHON scripts_windsor_parallel/run.py select --mode final --task $t; done
   for t in $(seq 0 14); do $PYTHON scripts32/pressure_task.py score --mode operator --task $t; done
   $PYTHON scripts32/report.py                                            # results/operator32_comparison.json
   $PYTHON checks/annotate_geometry_allocation.py                         # records the 4/32/8-GPU history in the report
   $PYTHON checks/compare_final_windsor.py                                # results/final_vs_previous_patch.json (needs step 3)
   ```

7. **d, nn⊤ alone, and fine-tuning seed 0 for both priors** (`coefficient_replicates/`). Rows 0 and 4 of `protocol/pretraining.json` are the d, nn⊤-only priors with seeds 17 and 29. The other rows are curvature-weighted priors that the paper does not report.

   ```bash
   $PYTHON scripts/copy_inputs.py                                         # cohort and WindsorML files from coefficient_prior/
   for row in 0 4; do $PYTHON scripts/pretrain_task.py --task $row; done   # four GPUs each
   for t in $(seq 0 9); do $PYTHON scripts/early_areals.py prepare --task $t; done
   for t in $(seq 0 29); do torchrun --standalone --nproc-per-node=4 scripts/early_areals.py fit --task $t; done
   for t in $(seq 0 9); do $PYTHON scripts/patch_query.py prepare --task $t; done
   for seed in 29 43; do for t in $(seq 0 14); do
     torchrun --standalone --nproc-per-node=4 scripts/patch_query.py fit --seed $seed --task $t
   done; done
   $PYTHON scripts/patch_query.py report                                  # results/patch_query_windsor.json
   ```

   The coefficient-prior checkpoints for seeds 29 and 43, and the seed-17 test results, are read from `coefficient_prior/` (paths in `protocol/patch_query_windsor.json`).

8. **Tables.** From the repository root, run `python -m paper_tables --only pretraining_targets --check`.

## Outputs → results

| Result file (under `results/`) | Written by | Used for |
|---|---|---|
| `pretraining_targets/matched/comparison.json` | `field_study.py report` | Table 1; Table M.2 test columns, gains and intervals; cars needed to reach 0.25 |
| `pretraining_targets/matched/test/000-089.json` | `field_study.py score` | AREALS-minus-Hessian differences and intervals; "31 of the 35 cars" |
| `pretraining_targets/matched/selections.json`, `test_lock.json` | `field_study.py select`; `test_lock.json` is the recorded list of the 90 selected runs, which `field_study.py` now derives from `selections.json` | Validation column (seed 0); rates chosen at the edge of the grid; the 90 tested runs |
| `pretraining_targets/matched/validation/*/*/n*/seed{1,2}-selected/best_validation.json` | `field_study.py followup` | Validation column (seeds 1 and 2); best-update share |
| `pretraining_targets_extra/matched_tuning_validation/*/*/n*/seed0-lr*/best_validation.json` | `field_study.py tune` | Best-update share over all 150 fits |
| `pretraining_targets_extra/matched_fit_configs/*/*/n*/*/config.json` | `field_study.py tune`, `followup` | Fine-tuning recipe (8,000 updates, 8 cars, 2,048 queries, rate grid, new-parameter rate, no weight decay, validation every 250); 274 training cars; nested budgets |
| `pretraining_targets/matched/geometry_{areals,hessian,patch}.json` | `evaluate_geometry.py` | Table M.1 geometry columns |
| `pretraining_targets/matched/pretraining_config_{arm}.json` | `pretrain.py` | Table M.1 parameters and peak rates; updates, batch, seed, GPUs |
| `pretraining_targets_extra/matched_cohort.json` | `matched_control.py snapshot` | Cohort counts (5,287 / 2,123 / 4,529; 1,668 / 85 / 92) |
| `pretraining_targets/checkpoint_sensitivity/validation_summary.json` | `diagnostics/areals_interim_20261031/summarize.py` | Checkpoint at 6,500 updates (0.2522) |
| `pretraining_targets/patch_pilot/summary.json` | `patch_pressure/tune_pressure.py` via `pressure_pilot.py finish` | Patch pilot 0.2176 vs 0.1917, selected rates |
| `pretraining_targets_extra/patch_pilot/export_receipt.json`, `pretraining_complete.json` | `pressure_pilot.py export`, `pretrain.py` | Pilot meshes, updates and seed |
| `pretraining_targets_extra/patch_pilot/windsor_cases.json`, `pressure_protocol.json` | `prepare_pilot.py windsor`, `pressure_pilot.py register` | Pilot split (274 training, 35 test cars) and rate grid |
| `pretraining_targets_extra/coefficient_pretraining/*/run.json` | `AREALS_shared_query/pretrain.py` | Coefficient-prior recipe (auxiliary weight 0.1, seeds 17/29/43, batch 64, FP32, no masking or rotation) and d, nn⊤-only seeds 17/29 |
| `pretraining_targets/coefficient_prior/operator32_comparison.json` | `scripts32/report.py`, then `checks/annotate_geometry_allocation.py` | Table M.3 left means; "18–38%"; cohort 14,304/637/632 |
| `pretraining_targets/coefficient_prior/per_car/n*/seed*/test_metrics.json` | `scripts32/pressure_task.py score` | Table M.3 left differences and intervals; seed-17 coefficient prior and seed-0 scratch |
| `pretraining_targets/coefficient_prior/final_vs_previous_patch.json` | `checks/compare_final_windsor.py` | "29–40% more accurate than the patch prior" |
| `pretraining_targets/coefficient_replicates/patch_query_windsor.json` | `scripts/patch_query.py report` | Table M.3 right "Coefficients"; pretraining-seed s.d. |
| `pretraining_targets/coefficient_replicates/per_car/{areals,patch_query}/seed*/n*/seed0/early_test_metrics.json` | `scripts/early_areals.py` (called by both `fit` commands) | Table M.3 right d, nn⊤ alone, differences and intervals; "25–41%"; "at most 0.007" |

## Compute

- **Matched priors:** four B200 GPUs per prior for 10,626 updates. Fine-tuning used one B200 per fit, up to 12 at a time. The median 8,000-update fit took 74 minutes for the AREALS arm and 37 for the patch arm, and the whole chain about 15 hours.
- **Patch pilot:** four B200 GPUs for pretraining, then six fine-tuning fits on one GPU each.
- **Coefficient prior:** seed 17 on 4, 32 and then 8 GPUs, as above; seeds 29 and 43 on four GPUs each. The 90 fits used one GPU each, up to 32 at a time.
- **d, nn⊤-only priors:** four GPUs each. Their fine-tuning-seed-0 fits, and those of the coefficient priors, used four GPUs each.

## Notes

- **Not included:**
  - the data manifests: the 65,491-object `protocol/objects.json`, `prepared/index.json`, `windsor/index.json`, the WindsorML case list `windsor_protocol/data_manifest.json` read by `prepare_windsor.py`, and the patch-pilot `protocol/cohort.json`;
  - HDF5 packs, WindsorML fields, checkpoints and logs.
- **The patch-pressure package.** `patch_pilot/patch_pressure/` is included as released. Its data audit identifies cars, meshes and CFD files by content hash to detect split leakage, and its tuning records tie each report to its data and checkpoint. For that reason it, and the two pilot scripts that feed it, still compute file hashes.
- **Statistics computed in `paper_tables/pretraining_targets.py`.** No saved script computed these numbers; the module computes them from the per-car result files:
  - the AREALS-minus-Hessian differences and intervals;
  - the Table M.3 differences and intervals;
  - the d, nn⊤-only means;
  - "31 of the 35 cars", "25–41%", "at most 0.007" and "73% of the 150 fits".

  The bootstrap uses a fresh `default_rng(2091)` per budget and 5,000 draws. The Table M.2 intervals are the stored ones, written by `field_study.py report` with one generator shared across all cells. The 13.5% patch-pilot figure is pretrained/scratch − 1 from `pressure/summary.json`.
- **Constants set only in code.** The Table M.1 weight decays (0.05, 0.05, 0), the 200-update warm-up and gradient clipping at 1 are fixed in `matched_priors/source/pretrain.py`. They are not written to any result file, so they are not rebuilt.
- **Hardware differs between the two halves of Table M.3.** The seed-17 coefficient-prior fits ran on one GPU each. The seed-29 and seed-43 fits and the d, nn⊤-only fits ran on four GPUs each.
