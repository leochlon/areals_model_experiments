# 11 Architecture ablations: decoder transfer, coordinate interface, normal targets and sampling

These studies test the design choices of AREALS one at a time at a reduced budget. The controlled geometry models are trained for four epochs (2,048 updates) on 8,192 Objaverse objects, with 128 development and 128 test objects, three seeds and one change per arm. They are then fine-tuned to surface pressure on DrivAerML. A second study fine-tunes the same kind of checkpoint on 380 cars and scores 40 withheld cars. It compares three transfer pathways: the pretrained query decoder kept, the decoder reset, and a released Point-MAE encoder with a fresh decoder. The production checkpoint is used only for the finite-token and resampling tests.

Paper: Section 4.5 ("The transfer runs through the query decoder") and Table E.1; Section 4.6 ("Each design choice does its job": normal targets, density shift and coordinate interface); Appendix E ("Decoder transfer"); Appendix F ("Normal lines", "A coordinate-aware interface", "Isotropic projector initialisation"); Appendix G ("Density shift", Table G.1, "Attention splitting", "Averaging resampled views"); the 40-car decoder study of Appendix H (96 hidden-value checks, recomputation within 2.0 × 10⁻¹⁶); and the Table A.1 rows "Decoder transfer and head initialisation", "Coordinate interface", "Normal targets", "Density shift" and "Finite tokens".

## Contents

- `architecture_controls/source/` (study root `/workspace/runs/areals_architecture_controls_20261024`, shared with `06_sensor_completion/heldout_angles` and `09_reserved_cars/model_training`, whose scripts the setup places in the same `source/` folder; no file names overlap):
  - `architecture.py`: four-epoch geometry pretraining of the arms `reference` (log-area attention, surface anchors, rotary positions, projector normal head), `ordinary_attention`, `unanchored_latents` and `oriented_normal_target`. Every arm starts from the same common tensors. It reads `production_config.json` and `architecture_manifest.json` and writes `results/architecture/<arm>/seed<s>/{best.pt,test.json}`.
  - `position_control.py`: `anchored_no_rotary` and `unanchored_no_rotary`, the same arms with the 3-D rotary transform replaced by the identity.
  - `orientation_objective.py`: `unoriented_vector_target`, the vector head trained with 1 − |cos| instead of 1 − cos.
  - `check_arms.py` (optional): a forward and backward pass of every arm on two training objects, confirming that the arms share their common initial tensors (`results/arm_check.json`).
  - `audit_geometry.py`: reloads the 21 checkpoints, saves raw test predictions and checks them against each `test.json`.
  - `summarize_architecture.py`: recomputes normal angles and distance errors from the raw predictions (`geometry()`) and the density drift from the raw arrays (`density()`), with an object bootstrap of 4,000 draws (rng seed 20261024).
  - `setup_transfer.py`, `transfer_architecture.py`: geometry-only pressure fine-tuning of the 21 checkpoints (7 arms × 3 seeds, 40 epochs) on the development observations of `architecture_transfer/bench`. `transfer_architecture.py` is also the model loader for the density test and the decoder study.
  - `online_backbone.py`: the Point-MAE and AREALS encoders behind a fresh common query decoder (seed 91700 + seed) with a 384 → 192 projection for Point-MAE. The decoder study and `09_reserved_cars` import it.
  - `density_stress.py`: a ninefold density shift on 32 test meshes, 3 seeds × 2 attention types × 3 draws, with fixed or recomputed anchors (1,344 prediction records).
  - `finite_token.py`: Table G.1 and the attention-splitting fixture, on the production checkpoint. `finite_token_raw.py` reruns it and saves every trial's predictions (64 files); `verify_finite.py` recomputes the 448 cases from them.
  - `nuisance_raw.py`: averaging four resampled views of the production checkpoint on 32 meshes, with the full input and with 60% hidden (it calls `author_nuisance_audit.main()` with `--draws 4 --masks 0.0,0.6`); `verify_nuisance.py` recomputes the results from the raw predictions.
- `decoder_transfer/decoder_study/source/` (study root `/workspace/runs/areals_decoder_transfer_20261024`): the 40-car decoder study (Table E.1, Section 4.5), the I/3 head initialisation (Appendix F) and the 21-contrast interface analysis (Appendix F):
  - `setup_confirmation.py`: the 380 / 16 / 40 split (40 of the 420 training cars, in a fixed hash order, become the test cars), the 18 primary tasks (`tasks.json`), the four primary contrasts (`confirmation_protocol.json`) and the geometry-only observation files. It also defines the study paths and the shared helpers.
  - `repair.py`: the I/3 head (zero weights, bias 1/3, 1/3, 1/3, 0, 0, 0 for the projector head; unit-z bias for the vector head) on top of `architecture.main`. `--check` confirms that the new head changes no other tensor and no distance output, and gives the encoder no initial normal-loss gradient (`results/repair_check.json`).
  - `skip_unused_patches.py`: removes the unused patch preprocessing from `bench/config.json`, after checking on one car that the training examples are unchanged.
  - `setup_decoder_controls.py`: the six fresh-decoder models (Point-MAE or AREALS encoder, three seeds; `decoder_controls_tasks.json`, `decoder_controls_protocol.json`). Their scalar head starts as in the primary models (seed 57300 + seed).
  - `run_task.py --task N` (through `bootstrap.py`): one of the 24 field models with the runtime settings of `fast_runtime.py`. Tasks 0–17 are the primary models (`confirmation_task.py`), tasks 18–23 the fresh-decoder models (`extra_task.py`). Each task trains 40 epochs on 380 cars (batch 8, learning rate 1e-3, 2,048 queries, geometry only), predicts every surface point of the 40 test cars and runs the hidden-value check `--cars 4 --per-car 4`, whose output goes to `logs/task_NN.log`.
  - `fast_runtime.py`: 8,192 fixed development queries per car, BF16 training with 256-query chunks and FP32 evaluation with 2,048-query chunks. `check_runtime.py` (optional) checks the chunking numerically on one training car.
  - `score_confirmation.py`, `score_decoder_controls.py`: recompute each car's nRMSE from the saved predictions, count the `PASS` lines of the hidden-value checks, and compute the contrasts with 10,000 paired car-bootstrap draws (rng seed 20261024), Bonferroni-adjusted over the six contrasts (99.17%).
  - `check_decoder_initialization.py`: resetting the decoder leaves the 102 non-decoder tensors and the scalar head identical.
  - `reverse_initialization.py`: an A-B-A test on one fixed model and batch (CPU, FP32): original head, I/3 head, original head restored.
  - `contrast_native_interface.py`: all 21 pairwise contrasts of the seven controlled variants on the 48 reserved cars (10,000 draws, Bonferroni over 21).
- `decoder_transfer/pointmae_released/` (workspace `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/field_pointmae_released_420`): `extract_features.author_encoder()` loads the released Point-MAE encoder; `source.json` names the upstream repository revision and the checkpoint.
- `numerical_checks/evidence/recompute.py` (workspace `/workspace/local/areals_numerical_checks_20261024/evidence`): recomputes several paper numbers from per-case extracts placed beside it. Here it supplies the 81.5% drift reduction.

## Inputs

`scripts/setup_workspace.py` copies the folders above, places `src/areal_tokens` at `/workspace/runs/areals_architecture_controls_20261024/vendor/areal_tokens/src` and the `v1` field harness (`shared/field_benchmark/v1/field_benchmark.py`) at `/workspace/runs/field_benchmark_4gpu_20261023`. The steps use the Python environment `/workspace/home/objaverse_base_030_20260924T172527Z/.venv`. Supply in addition:

- the Objaverse corpus manifest `/workspace/runs/objaverse_training_pool/versions/20260927T140652Z/manifest.json` and the production checkpoint `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/run/best_normal.pt` with its `config.json` (`01_pretraining`);
- the prepared geometry files of the 8,448 selected objects in `restored_geometry/<id>.npz` of the controls study (see Notes);
- the DrivAerML 420/16-car manifest and fields at `/workspace/runs/areal_drivaerml_420_epoch14_20260927T162928Z` and the benchmark directory `/workspace/runs/field_benchmark_4gpu_20261023/bench`;
- for the Point-MAE models, the public Point-MAE repository at revision `7445a68068d707da6d959b66773bfef84c4e3e32` in `pointmae_released/pointmae_source/` (`models/Point_MAE.py`, `cfgs/pretrain.yaml`) and its released `pretrain.pth` in `pointmae_released/weights/`. `bootstrap.py` adds `timm` and `PyYAML` from a second environment, `/workspace/home/areal_aerojepa_20260928/.venv/lib/python3.12/site-packages`.

## Run

### A. Controlled geometry study (Appendix F, Section 4.6, Appendix G)

```bash
cd /workspace/runs/areals_architecture_controls_20261024
export PYTHONPATH=$PWD/vendor/areal_tokens/src
python source/prepare_audit.py            # architecture_manifest.json (8,192 / 128 / 128 objects) and production_config.json
python source/check_arms.py               # optional (one GPU)

python source/finite_token.py             # Table G.1 and attention splitting: results/finite_token/
for arm in reference ordinary_attention unanchored_latents oriented_normal_target; do
  for seed in 0 1 2; do python source/architecture.py --arm $arm --seed $seed; done
done
for arm in anchored_no_rotary unanchored_no_rotary; do
  for seed in 0 1 2; do python source/position_control.py --arm $arm --seed $seed; done
done
for seed in 0 1 2; do python source/orientation_objective.py --seed $seed; done

python source/audit_geometry.py           # raw re-evaluation of the 21 checkpoints
python source/density_stress.py
python source/summarize_architecture.py   # results/geometry_audit/grouped.json, results/density_stress/independent_grouped.json
python source/finite_token_raw.py && python source/verify_finite.py
python source/nuisance_raw.py && python source/verify_nuisance.py   # results/nuisance_raw/independent_grouped.json
```

`prepare_audit.py` belongs to `06_sensor_completion/heldout_angles` and also prepares that study, so it needs that experiment's inputs as well. It selects the objects by SHA-256 order of their ids within the original splits and points every record at `restored_geometry/<id>.npz`. The density and finite-token tests also read each record's `source_mesh`.

Pressure transfer of the 21 checkpoints (coordinate interface) on the 420-car DrivAerML training split:

```bash
python source/setup_transfer.py           # development observations of architecture_transfer/bench
bench=architecture_transfer/bench
for arm in reference ordinary_attention unanchored_latents oriented_normal_target unoriented_vector_target \
           anchored_no_rotary unanchored_no_rotary; do
  for seed in 0 1 2; do
    name=${arm}_seed$seed
    t() { python source/transfer_architecture.py --arm $arm --seed $seed "$@"; }
    t train-areal --bench $bench --checkpoint results/architecture/$arm/seed$seed/best.pt --out $bench/models/$name \
      --epochs 40 --batch-size 8 --lr 1e-3 --new-lr 1e-3 --queries 2048 --normalise dataset \
      --mix-unobserved 1 --mix-sensors 0 --seed $seed --workers 4
    t predict --bench $bench --method areal --name $name --model $bench/models/$name/field_best.pt
    t check --bench $bench --method areal --name $name --model $bench/models/$name/field_best.pt
  done
done
```

The wrapper's own `--seed` argument consumes both `--seed` flags, so every fit uses downstream seed 0; the three replicas differ in their geometry checkpoint and head initialisation, as in Appendix F. The 21 fits are scored on the 48 reserved cars by `09_reserved_cars`: its `run_task.py` tasks 18–38 call `transfer_architecture.py predict|check` on that study's bench, and its `score.py` writes `results/complete/per_problem.csv`, which step B's `contrast_native_interface.py` reads.

### B. Decoder study (Table E.1, Section 4.5, Appendices E, F and H)

This needs the step-A checkpoints `results/architecture/{reference,ordinary_attention,oriented_normal_target}/seed*/best.pt` of the controls study.

```bash
cd /workspace/runs/areals_decoder_transfer_20261024
export PYTHONPATH=/workspace/runs/areals_architecture_controls_20261024/vendor/areal_tokens/src
python source/repair.py --check           # optional (one GPU)
python source/setup_confirmation.py       # split, tasks.json, observation files, links architecture_manifest.json and geometry/
python source/skip_unused_patches.py
python source/setup_decoder_controls.py
for t in $(seq 0 23); do python source/bootstrap.py source/run_task.py --task $t; done    # one GPU each
python source/check_runtime.py            # optional, on the model of task 4
python source/score_confirmation.py       # results/confirmation/
python source/score_decoder_controls.py   # results/complete/{summary.json,per_car.csv}
python source/bootstrap.py source/check_decoder_initialization.py   # results/decoder_initialization_check.json
python source/reverse_initialization.py   # results/initialization_reversal.json (CPU)
python source/contrast_native_interface.py   # results/combined_interface_contrast.json
```

- Tasks 0–5 and 15–17 first train their I/3-head geometry checkpoint with `repair.py --arm <arm> --seed <s>` (`results/architecture/<arm>/seed<s>/`); a checkpoint whose `test.json` already exists is reused. Tasks 6–14 start from the step-A checkpoints.
- Tasks 18–23 start from the geometry checkpoints of tasks 0, 2 and 4, so run them last.
- `contrast_native_interface.py` reads `/workspace/runs/areals_fresh_car_holdout_20261024/results/complete/per_problem.csv` (`09_reserved_cars`).

### C. Tables

```bash
python -m paper_tables --only architecture_ablations --check
```

## Outputs → results

| File in `results/` | Written by (study path) | Used for |
|---|---|---|
| `architecture_ablations/decoder_transfer/summary.json` | `score_decoder_controls.py` (`results/complete/summary.json`) | Table E.1; Section 4.5; Appendix E intervals; Appendix F pressure 0.2564 → 0.2399; Appendix H (96 checks, 1.94 × 10⁻¹⁶) |
| `architecture_ablations/decoder_transfer/encoder_parameter_audit.json` | parameter count of both backbones (see Notes) | Table E.1 and Appendix E: 21.825M against 2.274M, 256 tokens, 4,096 points |
| `architecture_ablations/decoder_transfer/initialization_reversal.json` | `reverse_initialization.py` | Appendix F: 155.0 → 0 → 155.0, head gradient 2.17 |
| `architecture_ablations/decoder_transfer/geometry_repair_aggregation.json` | per-seed `test.json` rows of the original-head and I/3-head reference arms (see Notes) | Appendix F: 0.01167 → 0.00854, 25.34° → 24.87° |
| `architecture_ablations/normal_targets/grouped.json` | `summarize_architecture.py` (`results/geometry_audit/grouped.json`) | Appendix F and Section 4.6 normal lines |
| `architecture_ablations/density_shift/summary.json`, `independent_grouped.json` | `density_stress.py`; `summarize_architecture.py` | Appendix G and Section 4.6 density shift |
| `architecture_ablations/finite_tokens/summary.json`, `raw_summary.json` | `finite_token.py`; `finite_token_raw.py` (identical files) | Table G.1, attention splitting |
| `architecture_ablations/finite_tokens/independent_verification.json` | `verify_finite.py` | Appendix G: 448 cases, recomputation bounds |
| `architecture_ablations/averaging_views/independent_grouped.json` | `verify_nuisance.py` | Appendix G: 19.06 → 18.53, 24.72 → 24.26 |
| `reserved_cars/summary.json` | `score.py` of `09_reserved_cars` | Appendix F and Section 4.6: 0.2469 against 0.3898, 36.7% |
| `architecture_ablations_extra/coordinate_interface/combined_interface_contrast.json` | `contrast_native_interface.py` | Adjusted interval 35.0–38.1%, 21 contrasts, seven variants |
| `architecture_ablations_extra/decoder_transfer/per_car.csv` | `score_decoder_controls.py` | Rebuilding the six adjusted intervals |
| `architecture_ablations_extra/decoder_transfer/decoder_controls_protocol.json` | `setup_decoder_controls.py` | 10,000 draws, 99.17% level |
| `architecture_ablations_extra/decoder_transfer/decoder_initialization_check.json` | `check_decoder_initialization.py` | Appendix E: 102 identical non-decoder tensors |
| `architecture_ablations_extra/decoder_transfer/tasks/*.json`, `extra_tasks/*.json` | `run_task.py` (`results/tasks/`, `results/extra_tasks/`) | Appendix E: 8,192 fixed development queries per car for all 24 models |
| `architecture_ablations_extra/decoder_transfer/architecture/reference/seed*/test.json` | `repair.py` (`results/architecture/reference/seed*/test.json`) | Section 4.5 and Table E.1: four-epoch AREALS checkpoints; cross-check of the I/3-head rows (0.00854, 24.87°) |
| `architecture_ablations_extra/density_shift/density_fixed_rows.json` | the 576 fixed-anchor rows of `results/density_stress/rows.json` | Section 4.6: 81.5% (the density input of `recompute.py`) |

The recorded protocol and task files also carry fields that the current scripts no longer write (file hashes, job ids and design notes).

## Compute

Every GPU step ran on one B200. Recorded run times come from the saved `test.json`, `history.json` and result files.

| Step | Recorded time |
|---|---|
| `architecture.py`, `position_control.py`, `orientation_objective.py` (21 four-epoch runs) | 4.7–5.4 min per run |
| `repair.py` geometry pretraining (nine I/3-head runs) | 5.0–6.6 min per run |
| `run_task.py` field fits (24 models) | 40 epochs reached after 104–115 s (AREALS encoder) or 145–151 s (Point-MAE) |
| `reverse_initialization.py` (CPU) | 3.7 s |

The pressure transfer (21 fits), the raw re-evaluation, density, finite-token and resampling tests and the scoring (CPU) have no recorded times.

## Notes

- **What is stored.** Checkpoints, raw predictions, the restored geometry files and logs are not included. The tables are rebuilt from the stored summaries. All recomputations from raw predictions (normal lines, density drift, finite tokens, resampled views, decoder-study scores) were run when the results were produced, and their outputs are stored.
- **Geometry restoration.** The 8,448 selected objects were copied from cold storage into `restored_geometry/` by a site-specific staging script, which is not included. They are the prepared corpus files of the manifest; `prepare_audit.py` records each record's original path.
- **Decoder-study history.** In the original run the nine I/3-head geometry checkpoints came from a first pass whose field fits validated on the full surface of every development car. That was too slow; those fits were stopped, and every reported field model was trained with the `fast_runtime.py` settings, reusing the geometry checkpoints.
- **Writers not included.**
  - `encoder_parameter_audit.json` was written by a short command that was not saved. For each backbone it lists the parameter count, the number of parameter tensors and their name prefixes, the inputs, the input point cap and the token count.
  - `geometry_repair_aggregation.json` was aggregated from the per-seed `test.json` files of `results/architecture/reference/seed*` in both study directories, by a command that was not saved. Its original-head rows agree with the raw-prediction recomputation in `normal_targets/grouped.json` to 1.4 × 10⁻¹⁰. Its I/3-head rows equal the three stored `test.json` files of `repair.py`; the 24.87° is the mean of those three rows.
- **Hidden-value tolerance.** The checks of the 40-car decoder study used the harness default tolerance, 1e-5 of the field spread. Appendix H states 1e-3. Every check passed under the stricter tolerance.
- **Adjusted family.** The Bonferroni family is the six contrasts in `six_contrasts` of `summary.json`: the four primary contrasts plus the two decoder comparisons, involving seven model groups.
- **Not stored as numbers.** The ninefold density ratio (`density_stress.py`), the four resampled views (`nuisance_raw.py`), the 384 → 192 projection (`online_backbone.py`) and the single production checkpoint of the finite-token test are set in code. The table module does not check them.
- **Upper bounds.** The paper prints the recomputation maxima as upper bounds, rounded up. The stored values are 2.24 × 10⁻⁹ (output change), 4.34 × 10⁻¹⁹ (distance NMSE) and 1.94 × 10⁻¹⁶ (decoder transfer). The table module checks them as rounded-up bounds.
- **`recompute.py`.** Its density input is `results/architecture_ablations_extra/density_shift/density_fixed_rows.json`. Its other inputs, DoubleDelta and GABI per-problem scores and a 48-car extract, belong to other experiments. Its output file is not stored; the table module recomputes the 81.5% from the same rows.
- **Point-MAE.** `extract_features.py` executes the encoder classes of the released Point-MAE code unchanged and loads the released weights; neither is included.
