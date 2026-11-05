# 08 Expanded benchmarks: DoubleDelta 18–19°, BlendedNet++, DrivAerML official split, DrivAerNet++

The production AREALS geometry prior is fine-tuned as a sensor-conditioned field model and compared, with matched examples, updates, learning rate and seeds, against the same architecture trained from scratch, a shape-only reference and a true-condition reference. There are four benchmarks: DoubleDelta wings at angles beyond the training range (18–19°), BlendedNet++ aircraft at higher Mach numbers than in training, the official DrivAerML split, and DrivAerNet++.

Paper: Table 2 (rows "DoubleDelta, 18–19°" and "BlendedNet++"), Table C.1, Table C.5 (rows 1–3), the corresponding rows of Table A.1, the field-training protocol of Appendix C, Section 4.2 (18.0% with all 400 official DrivAerML cars, 2% with all 6,492 DrivAerNet++ cars) and Section 4.4 ("Beyond the training range", "Other unseen conditions", and the DrivAerML sensor model matching the shape-only model).

## Contents

`existing_data_expansion/` is the study root (workspace `/workspace/runs/areals_existing_data_expansion_20261025`); the scripts are in `source/`:

- `common.sh`: the environment (study root, base checkpoint, `PYTHONPATH` with `kit/` and `areal_tokens`).
- `util.py`: shared helpers.
- `protocol.py`: writes `protocol.json`, the field-training protocol and the splits.
- `prepare_doubledelta.py`, `export_native.py`, `data_checks.py`: dataset preparation and split checks (no physical geometry across splits, no held-out conditions in training, finite aligned arrays, training-only condition statistics).
- `drivaerml_official_splits.json`: the official DrivAerML split lists (`full_train` / `full_val` / `full_test` = 400 / 34 / 50).
- `field_study.py`: the harness with the dataset's input role. `setup` writes the fixed validation and test observation sets (5, 20, 100 and 500 sensors, and nothing observed; seed 0); `check-setup` confirms on a GPU that hidden targets cannot reach the inputs, that wrapping the checkpoint for field inputs keeps its weights, and that loss and gradients are finite. The other commands are those of `field_benchmark.py`.
- `check_field_run.py`: the per-fit completion check.
- `aggregate.py`: scores the predictions and writes `bench/scores/test_grouped.json` (equal weight per geometry, mean over seeds, paired geometry bootstrap with 5,000 draws).
- `field_scale.py`, `scale_score.py`: a 400-car DrivAerNet++ control with the full-data update budget. It is the `small` entry of `drivaernet_test_grouped.json` and is not reported in the paper.

## Inputs

`scripts/setup_workspace.py` places `existing_data_expansion/` at the study root and the `kit` harness at `kit/field_benchmark.py`. Supply in addition the production checkpoint `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/run/best_normal.pt` with `src/areal_tokens` at `.../areal_tokens/src`, and:

- for DoubleDelta, the prepared nine-angle data of `06_sensor_completion` at `/workspace/runs/areals_multicondition_wings_20261023`;
- for the other three datasets, the native data reader (see Notes).

One Python environment with numpy, scipy, torch and h5py serves every step; `common.sh` expects it at `/workspace/home/objaverse_base_030_20260924T172527Z/.venv`.

## Run

Every command runs from the study root with the environment of `common.sh`:

```bash
cd /workspace/runs/areals_existing_data_expansion_20261025
source source/common.sh
python source/protocol.py
```

Then, for each `dataset` in `doubledelta`, `blendednet`, `drivaerml` and `drivaernet`:

```bash
data=$STUDY_ROOT/datasets/$dataset
if [ $dataset = doubledelta ]; then                      # data preparation (CPU)
  python source/prepare_doubledelta.py                  # one grid per wing family; angles 11–16° / 17° / 18–19°
else
  python source/export_native.py --dataset $dataset --workers 8
fi
python source/data_checks.py --dataset $dataset
python source/field_study.py --dataset $dataset setup
python source/field_study.py --dataset $dataset check-setup     # optional (one GPU)

fit() { python source/field_study.py --dataset $dataset --role $role "$@"; }
for seed in 0 1 2; do                                    # 12 fits, one GPU each
  for arm in pretrained scratch shape conditioned; do
    role=measurements; extra=
    case $arm in scratch) extra=--scratch ;; shape) role=geometry ;; conditioned) role=conditioned ;; esac
    out=$data/models/${arm}_seed$seed
    fit train-areal --bench $data/bench --checkpoint $BASE_CHECKPOINT --out $out --epochs 40 --batch-size 8 \
        --lr 1e-3 --new-lr 1e-3 --queries 2048 --workers 4 --seed $seed --normalise dataset --mix-unobserved 0.1 \
        --mix-sensors 0.9 --select-cars 64 --eval-batch-size 4 --device cuda --amp bf16 $extra
    for split in val test; do
      final=; if [ $split = test ]; then final=--final; fi
      fit predict --bench $data/bench --method areal --name ${arm}_seed$seed --model $out/field_best.pt \
          --split $split --batch-size 4 --device cuda --amp bf16 $final
      fit check --bench $data/bench --method areal --name ${arm}_seed$seed --model $out/field_best.pt \
          --split $split --batch-size 4 --device cuda --amp bf16 --cars 3 --per-car 4 $final
    done
    python source/check_field_run.py --dataset $dataset --name $arm --seed $seed
  done
done
python source/aggregate.py --dataset $dataset            # scoring (CPU)
```

Each dataset ends with `datasets/<dataset>/bench/scores/test_grouped.json` and `datasets/<dataset>/complete.json`.

- **BlendedNet++ cohort.** The stored cohort was built with `python source/export_native.py --dataset blendednet --workers 4 --all-condition-bands`, followed by `data_checks.py`: each geometry family goes to its most populated Mach band. The recorded run first exported with the default rule and rebuilt the cohort this way before any observation sets or fits.
- **DoubleDelta evaluation.** The stored DoubleDelta scores were produced after rerunning the `predict`, `check` and `check_field_run.py` steps of one fit (pretrained, seed 1) from its saved checkpoint, without retraining, and then `aggregate.py`.
- **400-car control** (optional, after DrivAerNet++): for seeds 0–2, run `source/field_scale.py --dataset drivaernet` with the `train-areal`, `predict` and `check` arguments above and `--name small_seed$seed` (outputs in `models/small_seed$seed`), then `python source/field_scale.py --dataset drivaernet check-budget` and `python source/scale_score.py`.

`python -m paper_tables --only expanded_benchmarks --check` rebuilds the tables from the stored results.

## Outputs → results

| File | Written by | Used for |
|---|---|---|
| `results/expanded_benchmarks/doubledelta_test_grouped.json` | `aggregate.py --dataset doubledelta` | Table 2 row 2, Table C.1, Table C.5 row 1, Section 4.4 |
| `results/expanded_benchmarks/blendednet_test_grouped.json` | `aggregate.py --dataset blendednet` | Table 2 row 3, Table C.1, Table C.5 row 2, Section 4.4 |
| `results/expanded_benchmarks/drivaerml_test_grouped.json` | `aggregate.py --dataset drivaerml` | Table C.1, Table C.5 row 3, Sections 4.2 and 4.4 |
| `results/expanded_benchmarks/drivaernet_test_grouped.json` | `aggregate.py --dataset drivaernet` | Table C.1, Section 4.2 |
| `results/expanded_benchmarks/latest_completed_results.json` | a summary of the DoubleDelta, DrivAerML and DrivAerNet++ score files with `reduction_vs_scratch_percent` | cross-check of Table C.5 |
| `results/expanded_benchmarks/protocol.json` | `protocol.py` (the recorded copy also lists the checkpoint hash, resource notes and a list of checks) | Appendix C protocol |
| `results/expanded_benchmarks_extra/data_readiness.json` | `ready.json` and `data_checks.json` of each dataset | split sizes (Table C.5, Sections 4.1 and 4.2, Appendix C) |

`paper_tables/expanded_benchmarks.py` rebuilds every table and number above, including the four rows of Table A.1, and computes the reductions and gap shares from the score files.

## Compute

One GPU per field fit (8 CPU cores, 96 GB, under 24 h), 12 fits per dataset (3 more for the 400-car control); preparation and scoring run on CPU. Optimizer updates per fit: 6,840 on DoubleDelta, 2,000 on DrivAerML and 32,480 on DrivAerNet++ (40 epochs at batch 8). Wall-clock times were not recorded.

## Notes

- **Native data reader not included.** `export_native.py` reads the native HDF5 files of BlendedNet++, DrivAerML and DrivAerNet++ (`mesh.verts`, `mesh.verts_normals`, `mass`, `mesh.PressureCoeff`, and condition scalars from `scalars.h5`) from an object-store copy of the public releases, through a module `storage` providing `client()`, `objects(s3, bucket, prefix)` and `RangeFile(s3, bucket, key, size)`. That module is not included; `PREFIXES` in `export_native.py` names the dataset snapshots it read. To rerun preparation, provide an equivalent `source/storage.py`.
- **Geometry grouping.** `export_native.py` groups cases into physical families by source design and by a hash of the quantised, sorted native vertex set, and `data_checks.py` hashes the prepared geometry to confirm that no identical geometry crosses splits. These hashes find duplicates; they are not integrity checks.
- Code paths for an additional dataset that is not in the paper were removed from `export_native.py`, `data_checks.py`, `protocol.py` and `util.py`; behaviour for the four benchmarks is unchanged.
- The stored DoubleDelta and DrivAerML split records were written by an earlier revision of `data_checks.py`; the included revision wrote the BlendedNet++ and DrivAerNet++ records.
- The study used the `kit` harness: `check_field_run.py` and `04_masked_vs_unmasked` need the per-epoch `update` field that only this version writes to `history.json`.
- `$STUDY_ROOT/vendor` and `.../geometry_latents_minimal/src` are on `PYTHONPATH` in `common.sh` but are not imported.
- Table C.5 counts DoubleDelta training data in physical wing families (228 families × 6 angles = 1,368 cases; test: 16 wings × 2 angles = 32 cases).
- **BlendedNet++ split.** Mach bands are set at the 70th and 85th percentiles of the condition metadata before any field export; each geometry family goes to its most populated band, so validation and test aircraft are unseen geometries at higher Mach.
- On DrivAerML and DrivAerNet++ the true-condition arm receives a constant input (one inflow condition), so it is in effect a second shape-only fit.
- The BlendedNet++ reduction over scratch ranges from 14.5% at K = 5 to 27.0% at K = 500.
- `latest_completed_results.json` was compiled from the score files after the runs; no script here writes it. Logs, checkpoints, predictions and per-problem score tables are not included.
