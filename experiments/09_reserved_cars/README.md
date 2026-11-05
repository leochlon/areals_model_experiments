# 09 Reserved-car completion: DrivAerML 420/16/48 split

Field completion on 48 DrivAerML cars that were never used for training or checkpoint selection. The production AREALS geometry prior and the same architecture from scratch are fine-tuned on 420 cars (three seeds each, matched examples, updates, learning rate and scalar-head initialisation), with checkpoints selected on 16 development cars. The split and all checkpoints are fixed before the test pressures are prepared, and each model is scored once on the 48 reserved cars, averaging the 25 observation settings equally within each car.

Paper: Table C.5 (row "DrivAerML, reserved"), the 20.2% statement of Section 4.2, the 420/16/48 split of Section 4.1, the "Reserved-car completion" paragraph of Appendix E, the Table A.1 row "Reserved-car completion" and the 48-car numbers of Appendix H (hidden-value perturbation, independent recomputation).

The same registry lists 48 checkpoints: the six completion models, 12 geometry-only online-encoder models, 21 coordinate-interface variants (Appendix F, `11_architecture_ablations`) and nine frozen-surrogate models (the 48-car replication of Appendix L, `10_frozen_surrogate`). This folder trains the completion, online-encoder and frozen-surrogate models and evaluates and scores all 48; experiments 10 and 11 read their 48-car numbers from this folder's `score.py` output.

## Contents

- `drivaerml_pilot_split/` (workspace `/workspace/home/drivaerml_transfer_pilot_20260925T121525Z`): `prepare_data.py` (`make_manifest`) and `config.json` define the split. The 48 reserved cars are the source `test_design_ids.txt`; the 16 development cars are drawn from the training pool with `split_seed` 20260925. The file also holds the geometry and pressure sampling recipe that the later preparation steps reuse.
- `drivaerml_420_split/` (workspace `/workspace/home/areal_drivaerml_420_epoch14_20260927T162928Z`): `prepare_420.py` and `config.json` add the remaining 356 training-pool cars and write the 420/16/48 `split.json` and the training manifest to the data root `/workspace/runs/areal_drivaerml_420_epoch14_20260927T162928Z`. `prepare_420.prepare_case` is also used for the 48 test cars.
- `model_training/source/` (study root `/workspace/runs/areals_architecture_controls_20261024`, shared with other experiments; see Notes):
  - `matched_seed_replication.py` trains the six completion models, AREALS (`pretrained`) and `scratch`, seeds 0–2. It sets the new scalar head with `torch.manual_seed(57300 + seed)`, so each pretrained/scratch pair starts from the same head, and passes the real downstream seed to the harness.
  - `frozen_replication.py` trains the nine frozen-surrogate models (AeroJEPA tutorial `native`, `native_normals` and frozen-AREALS `areal`, seeds 0–2) with the `run_field.py` driver of `10_frozen_surrogate`, and recomputes every validation score from the raw predictions.
  - The 12 online-encoder models are trained with `online_backbone.py` of `11_architecture_ablations` (AREALS and Point-MAE encoders behind a fresh query decoder, backbone learning rates 1e-3 and 1e-4, seeds 0–2).
- `holdout_evaluation/source/` (study root `/workspace/runs/areals_fresh_car_holdout_20261024`):
  - `registry.py`: checks the 420/16/48 split and its disjointness, loads the 48 checkpoints, checks their ancestry, and writes `models.json` and `protocol.json` (split ids and comparisons). It also defines the study paths and the `write` helper that the other scripts import.
  - `prepare.py`: prepares the 48 test cars with the original recipe (`prepare_420.prepare_case`, plus the frozen-surrogate inputs through `FieldDataset._prepare`), after checking exact array parity with the stored arrays of one development car. It also writes the test bench configurations `benches/{completion,geometry}/config.json`.
  - `setup_evaluation.py`: checks the split against the source split lists and tests for duplicate surface-coordinate sets, then writes the fixed test observations: the 25 settings plus the no-observation diagnostic for completion, and no observation for the geometry-only models.
  - `run_task.py --task N`: evaluates model N of `models.json` once. It runs `predict` and the hidden-value perturbation `check` (`--cars 4 --per-car 4`) for the completion and geometry-only models, writing their output to `logs/evaluate_NN.log`, and `frozen_eval.py` for the frozen-surrogate models.
  - `frozen_eval.py`: test inference of the frozen-surrogate models on 2,048 fixed queries per car, with four target-pressure perturbation checks per model (`--parity` replays one stored validation call instead).
  - `score.py`: recomputes every per-car score from the saved predictions, targets and area weights and compares it with the harness metric (below 1e-10; 1e-12 against `frozen_eval.py` for the frozen-surrogate models), counts the `PASS` lines of each evaluation log, and writes `results/complete/summary.json` and `per_problem.csv`. Intervals come from a 4,000-draw car bootstrap (rng 20261024) that keeps the seeds within each car.
  - `audit_complete.py`: an independent standard-library reduction of `per_problem.csv` and the evaluation logs (`results/complete/raw_csv_and_log_audit.json`; 264 passed and 0 failed checks).
  - `check_flagged_controls.py`: checks that the rotary modules of the `*_no_rotary` models are the identity, and recomputes the frozen AREALS − AeroJEPA-with-normals interval from the CSV (`results/complete/flagged_controls_check.json`).
- `native_field_preparation/` (copied over `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z`, the production-prior folder of `01_pretraining/geometry_pretraining`): what `prepare.py` imports to build the frozen-surrogate inputs, namely `field_utonia_compare_420/common.py` and its `config.json`, `field_prior_gate_check/compatibility.json`, and the eight modules of the `areals` 0.1.0 package that this import needs.

## Inputs

`scripts/setup_workspace.py` places the folders above, the `v1` harness at `/workspace/runs/field_benchmark_4gpu_20261023/field_benchmark.py` and `src/areal_tokens` at `/workspace/runs/areals_architecture_controls_20261024/vendor/areal_tokens/src`. Supply in addition `geom_latents` (`01_pretraining/objaverse_base_corpus/geometry_latents_minimal`) at `/workspace/home/objaverse_base_030_20260924T172527Z/geometry_latents_minimal/src`, the production prior checkpoint at `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/run/best_normal.pt` (below `$BASE`) and the raw-data reader (see Notes). Two Python environments are used: one with numpy, scipy, torch, h5py and numba for most steps (`/workspace/home/objaverse_base_030_20260924T172527Z/.venv`), and the AeroJEPA environment of `10_frozen_surrogate` (`/workspace/home/areal_aerojepa_20260928/.venv`, below `$AEROJEPA_PYTHON`) for the online-encoder and frozen-surrogate models and the evaluation.

## Run

**1. Split.** From the DrivAerML split lists `train_design_ids.txt` and `test_design_ids.txt` (directory `split_root` in `drivaerml_pilot_split/config.json`):

```bash
cd /workspace/home/drivaerml_transfer_pilot_20260925T121525Z && python prepare_data.py --config config.json
```

This writes `manifest.json` (64 training cars, 16 development cars, and `benchmark_test_excluded` with the 48 reserved cars) before any data are read; the per-car preparation that follows needs the raw-data reader.

**2. Training data for the 420 cars** (CPU, 24 workers): writes `split.json` (420/16/48), `manifest.json`, `geometry/` and `fields/` under the data root, reusing the 80 cars already prepared in `comparison_run` (see Notes).

```bash
cd /workspace/home/areal_drivaerml_420_epoch14_20260927T162928Z
PYTHONPATH=/workspace/home/objaverse_base_030_20260924T172527Z/geometry_latents_minimal/src python prepare_420.py
```

**3. Training bench.** `/workspace/runs/areals_architecture_controls_20261024/replication/bench` holds `config.json` and the observations of the 16 development cars (25 settings) used for checkpoint selection. The step that assembled it is not included (see Notes).

**4. Train the registered models** (one GPU per model), from `/workspace/runs/areals_architecture_controls_20261024`:

```bash
cd /workspace/runs/areals_architecture_controls_20261024
HARNESS=/workspace/runs/field_benchmark_4gpu_20261023/field_benchmark.py

# Completion models: 40 epochs, batch 8, learning rate 1e-3 for pretrained and new parameters
export PYTHONPATH=$PWD/vendor/areal_tokens/src:/workspace/home/objaverse_base_030_20260924T172527Z/geometry_latents_minimal/src
bench=$PWD/replication/bench
for seed in 0 1 2; do
  for arm in pretrained scratch; do
    scratch=; if [ $arm = scratch ]; then scratch=--scratch; fi
    name=${arm}_seed$seed
    python source/matched_seed_replication.py train-areal --bench $bench --checkpoint $BASE --out $bench/models/$name $scratch \
        --epochs 40 --batch-size 8 --lr 1e-3 --new-lr 1e-3 --queries 2048 --normalise dataset --floor .1 \
        --visible-weight .25 --weight-decay .05 --mix-unobserved .1 --mix-sensors .35 --seed $seed --amp bf16 --workers 4
    python $HARNESS predict --bench $bench --method areal --name $name --model $bench/models/$name/field_best.pt
    python $HARNESS check --bench $bench --method areal --name $name --model $bench/models/$name/field_best.pt
  done
done

# Online-encoder models: 40 epochs, batch 8, reader learning rate 1e-3, geometry only
export PYTHONPATH=$PWD/vendor/areal_tokens/src
bench=$PWD/architecture_transfer/bench
for lr in 1e-3 1e-4; do
  for seed in 0 1 2; do
    for arm in areals pointmae; do
      name=online_${arm}_lr${lr}_seed$seed
      online() { $AEROJEPA_PYTHON source/online_backbone.py --backbone $arm --audit-seed $seed "$@"; }
      online train-areal --bench $bench --checkpoint $BASE --out $bench/models/$name --epochs 40 --batch-size 8 \
          --lr $lr --new-lr 1e-3 --queries 2048 --normalise dataset --mix-unobserved 1 --mix-sensors 0 --seed $seed --workers 4
      online predict --bench $bench --method areal --name $name --model $bench/models/$name/field_best.pt
      online check --bench $bench --method areal --name $name --model $bench/models/$name/field_best.pt
    done
  done
done

# Frozen-surrogate models: 20 epochs, batch 8 on one GPU, 256 geometry tokens
export PYTHONPATH=/workspace/home/areal_objaverse_050_20260927T090929Z/areal_tokens/src:/workspace/home/areal_aerojepa_20260928:/workspace/home/areal_aerojepa_20260928/physicsnemo
for seed in 0 1 2; do
  for name in native areal native_normals; do
    arm=$name; normals=; if [ $name = native_normals ]; then arm=native; normals=--normals; fi
    $AEROJEPA_PYTHON source/frozen_replication.py --arm $arm --epochs 20 --batch-per-rank 8 --seed $seed \
        --geometry-tokens 256 --run-name ${name}_seed$seed $normals
  done
done
```

The online-encoder models need `architecture_transfer/bench` (`setup_transfer.py`), `source/online_backbone.py` and `source/transfer_architecture.py` of `11_architecture_ablations`, and the Point-MAE loader `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/field_pointmae_released_420/extract_features.py` from the same experiment. The 21 coordinate-interface variants are trained by `11_architecture_ablations` and written to `architecture_transfer/bench/models/<arm>_seed<s>/`. The frozen-surrogate models write `frozen_replication/runs/<name>_seed<s>/{run.json,final_live.pt}`; they import `run_field.py` and read `data/` and `data_ready.json` from `/workspace/home/areal_aerojepa_20260928` (`10_frozen_surrogate`), and their frozen geometry prior is the epoch-14 checkpoint described there.

**5–10. Register, prepare, evaluate and score**, from the holdout study root:

```bash
cd /workspace/runs/areals_fresh_car_holdout_20261024
CONTROLS=/workspace/runs/areals_architecture_controls_20261024
GEOM=/workspace/home/objaverse_base_030_20260924T172527Z/geometry_latents_minimal/src
PYTHONPATH=$CONTROLS/vendor/areal_tokens/src python source/registry.py      # the 48 models of step 4
PYTHONPATH=/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/field_prior_gate_check/areals/src:$CONTROLS/vendor/areal_tokens/src:$GEOM \
    python source/prepare.py                                                  # the 48 test cars (CPU, 10 cores)
PYTHONPATH=$CONTROLS/vendor/areal_tokens/src python source/setup_evaluation.py
for task in $(seq 0 47); do                                                   # one GPU per task
  path=$CONTROLS/vendor/areal_tokens/src:$GEOM
  if [ $task -ge 39 ]; then path=/workspace/home/areal_objaverse_050_20260927T090929Z/areal_tokens/src:/workspace/home/areal_aerojepa_20260928/physicsnemo; fi
  PYTHONPATH=$path $AEROJEPA_PYTHON source/run_task.py --task $task
done
PYTHONPATH=$CONTROLS/vendor/areal_tokens/src python source/score.py          # results/complete/; --primary scores the 12 primary models
python source/audit_complete.py
PYTHONPATH=$CONTROLS/vendor/areal_tokens/src python source/check_flagged_controls.py
```

`prepare.py` derives the test bench configurations from `replication/bench/config.json` and from `architecture_transfer/bench/config.json` of `11_architecture_ablations`. The completion models are tasks 2, 3, 6, 7, 10 and 11, and tasks 39–47 are the frozen-surrogate models. `check_flagged_controls.py` loads the `*_no_rotary` checkpoints and `source/position_control.py` and `source/transfer_architecture.py` of `11_architecture_ablations`.

`python -m paper_tables --only reserved_cars --check` rebuilds the tables from the stored results.

## Outputs → results

| File | Written by | Used for |
|---|---|---|
| `results/reserved_cars/summary.json` | `score.py` (`results/complete/summary.json`) | Table C.5 row 4; Section 4.2 (20.2%); Appendix E and Table A.1 (25 settings, three seeds, 48 cars); Appendix H (264 checks, recomputation within 3.4e-16). Also read by `10_frozen_surrogate` (`frozen`) and `11_architecture_ablations` (`architecture`) |
| `results/reserved_cars_extra/data_split.json` | `prepare_420.py` (`split.json` in the data root) | 420/16/48 (Table C.5 Train, Section 4.1, Appendix E, Table A.1) |
| `results/reserved_cars_extra/split_and_coordinate_audit.json` | `setup_evaluation.py` | split counts and duplicate-coordinate check (Appendix H, split identities) |
| `results/reserved_cars_extra/raw_csv_and_log_audit.json` | `audit_complete.py` | independent means; 264 passed and 0 failed hidden-value checks (Appendix H) |

`paper_tables/reserved_cars.py` rebuilds Table C.5 row 4, the split and protocol numbers of Sections 4.1 and 4.2, Appendix E and Table A.1, and the 48-car numbers of Appendix H, and computes the reduction from the stored means.

## Compute

- Training: one B200 GPU, 10 CPU cores and 80–96 GB per model, under 2 h each (six completion, 12 online-encoder and nine frozen-surrogate models).
- Evaluation: one B200, 6 CPU cores and 48 GB per model, under 1.5 h each; no training.
- CPU steps: test-car preparation (10 cores, under 40 min), observations and split checks (6 cores, under 30 min), scoring (4 cores, under 1 h), and the 420-car preparation (24 cores, under 20 min).

These are the limits the runs were given; wall-clock times were not recorded.

## Notes

- **Shared study root.** `model_training/` is one part of `/workspace/runs/areals_architecture_controls_20261024`. The online-encoder and architecture code of `11_architecture_ablations` and the held-out-angle code of `06_sensor_completion/heldout_angles` provide the rest; the setup script combines them in that one directory. `vendor/areal_tokens/src` there is `src/areal_tokens`.
- **Architecture seeds.** The 21 architecture variants vary the upstream pretraining seed and the new-head seed (0, 1, 2) but reuse downstream data-sampling seed 0, because the original wrapper consumed that flag; they are not three independent downstream runs. The online-encoder, completion and frozen-surrogate comparisons have real downstream seeds 0, 1 and 2.
- **Raw-data reader removed.** The original `prepare_data.py` had a `Storage` class that read the per-car DrivAerML files `<source_prefix><run>/fields.h5` and `scalars.h5` from an object-store copy of the dataset; it was specific to that infrastructure and is removed. `prepare_data.py`, `prepare_420.py` and `holdout_evaluation`'s `prepare.py` call `Storage(config)`, `.fetch(key, destination)` (returning a dict that includes `bytes`) and `.close()`. To rerun preparation, provide an equivalent class in `prepare_data.py`. `source_prefix` names the dataset snapshot that was read.
- The DrivAerML split lists are read from `/workspace/data/drivaerml_splits` (set in `drivaerml_pilot_split/config.json` and in `setup_evaluation.py`).
- `prepare_420.py` reuses the 64 training and 16 development cars of the step-1 `manifest.json` from `comparison_run` (`/workspace/home/areal_drivaerml_finetune_20260927T131234Z`, providing `manifest.json`, `data_records/`, `geometry/` and `fields/`). The script that converted those 80 cars from the per-car files of step 1 is not included; `prepare_420.prepare_case` performs the same conversions in one pass, and `prepare.py` checks exact parity with the stored arrays of one development car.
- The script that assembled `replication/bench` is not included. Its development observations are those that `field_benchmark.py observations --split val` writes with the default settings (3 patch counts × 6 hidden shares and 7 sensor counts = 25).
- `prepare.py` and `frozen_eval.py` import the AeroJEPA code and data of `10_frozen_surrogate` from `/workspace/home/areal_aerojepa_20260928`: `run_field.py`, `drivaerml.py`, `areal_context.py`, `physicsnemo/`, `data/` and `data_ready.json`.
- The frozen-surrogate training and evaluation (tasks 39–47) put `areal_tokens` 0.5.0 (`/workspace/home/areal_objaverse_050_20260927T090929Z/areal_tokens/src`) on `PYTHONPATH`, because `areal_context.py` of `10_frozen_surrogate` imports it. That version is not included.
- `frozen_replication.py --smoke` runs one training and one validation batch as a software check; the runs above do not use it.
- `areals` 0.1.0 is an earlier package; only the modules imported by `field_utonia_compare_420/common.py` are included, and `prepare.py` uses only `FieldDataset._prepare`. `common.experiment()` also reads the production-prior `config.json` from `01_pretraining/geometry_pretraining`.
- The hidden-value checks used the harness default tolerance (1e-5 of the field spread), because `run_task.py` passes no `--tolerance`; this is stricter than the 10⁻³ stated in Appendix H.
- Model checkpoints, predictions and evaluation logs are not included.
