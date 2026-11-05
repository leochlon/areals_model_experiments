# 02 Simulation efficiency

How many labelled CFD simulations do the pretrained model and the same architecture from scratch need to reach a fixed test pressure nRMSE? On WindsorML, DrivAerML and DrivAerNet++, both arms are fine-tuned on nested subsets of 1/16, 1/8, 1/4, 1/2 and all of the training split. Each fit runs 8,000 updates, with three paired seeds, for 30 fits per dataset. Each fit is scored once on the common test split, with no sensors and with 20 sensors.

Paper: Figure 1, Section 4.2, Appendix C ("Simulation efficiency"), the Table A.1 row "Simulation efficiency (Figure 1)", and the 50–74.5% savings quoted in the contributions and conclusion.

## Contents

- `drivaerml/`: DrivAerML with its official 400/34/50 split and budgets 25, 50, 100, 200 and 400. `protocol.json` holds the recipe. `source/study.py` has the stages `prepare`, `check`, `train` and `score`. `source/analyse.py` runs the paired bootstrap and finds the budget thresholds, tables and figures. `source/replay.py` rescores every stored prediction.
- `windsor_drivaernet/`: the same protocol on WindsorML (274/35/35 after removing repeated geometries) and DrivAerNet++ (6,492/778/848). `suite_protocol.json` holds the shared recipe. `datasets/windsor/cohort_amendment.json` holds the WindsorML deduplication rule and `public_sources/windsorml/splits_manifest.json` the official split. `source/prepare_data.py` exports the WindsorML native fields. `source/setup_dataset.py` writes each dataset's `protocol.json`, with budgets `ceil(T/16) … T`.

DrivAerML keeps one checkpoint per fit, selected by zero-sensor validation error and reused at 20 sensors. WindsorML and DrivAerNet++ keep a validation-selected checkpoint for each sensor setting (`best_s0.pt`, `best_s20.pt`). The zero-sensor results behind Figure 1 are selected the same way in both.

## Inputs

`scripts/setup_workspace.py` places `drivaerml/` at `/workspace/runs/areals_simulation_efficiency_20261028` and `windsor_drivaernet/` at `/workspace/runs/areals_all_simulation_efficiency_20261028`. Each gets the `kit` harness as `source/field_benchmark.py` and `src/areal_tokens` as `model_src/areal_tokens`. Supply in addition:

- the production checkpoint (`best_normal.pt` from `01_pretraining`) at `<study>/checkpoints/geometry.pt`;
- the DrivAerML and DrivAerNet++ native-field exports at `/workspace/runs/areals_existing_data_expansion_20261025/datasets/{drivaerml,drivaernet}`, written by `08_expanded_benchmarks/existing_data_expansion/source/export_native.py`;
- for the WindsorML export, the public WindsorML files and their object inventory at `/workspace/runs/areals_dataset_inventory_20261025/WindsorML_objects.json`.

One Python environment with torch, numpy, scipy, numba and matplotlib (and h5py for the WindsorML export) serves every step. The recorded runs used Python 3.12, torch 2.11 and numpy 2.5. Set `OMP_NUM_THREADS=1`.

## Run

**DrivAerML.** `prepare` expects the export's `fields/` and geometry files under `data/fields/` and `data/geometry/`, and its `manifest.json` and `split_protocol.json` under `evidence/`:

```bash
cd /workspace/runs/areals_simulation_efficiency_20261028
export STUDY_ROOT=$PWD PYTHONPATH=$PWD/source:$PWD/model_src
python source/study.py prepare                     # nested subsets, run matrix, geometry cache (CPU)
python source/study.py check                       # optional setup checks (one GPU)
for i in $(seq 0 29); do python source/study.py train --index $i; done    # one GPU per fit
for i in $(seq 0 29); do python source/study.py score --index $i; done
python source/analyse.py
python source/replay.py --gpu-index 0
```

**WindsorML and DrivAerNet++.** WindsorML is exported first; DrivAerNet++ reuses the existing export:

```bash
cd /workspace/runs/areals_all_simulation_efficiency_20261028
export SUITE_ROOT=$PWD PYTHONPATH=$PWD/source:$PWD/model_src
python source/prepare_data.py plan --dataset windsor
for rank in 0 1 2 3; do python source/prepare_data.py export --dataset windsor --rank $rank --shards 4 --workers 2; done
python source/prepare_data.py finish --dataset windsor     # applies datasets/windsor/cohort_amendment.json
for dataset in windsor drivaernet; do
  export STUDY_ROOT=$SUITE_ROOT/datasets/$dataset
  python source/setup_dataset.py $dataset
  python source/study.py prepare
  python source/study.py check
  for i in $(seq 0 29); do python source/study.py train --index $i; done
  for i in $(seq 0 29); do python source/study.py score --index $i; done
  python source/analyse.py
  python source/replay.py --gpu-index 0
done
```

Outputs go to `<study>/results/` for DrivAerML and `datasets/<dataset>/results/` for the others: `summary.json`, `thresholds.json`, `learning_curves.csv`, `per_case.csv`, figures and `replay.json`. `check` confirms that corrupting unobserved pressures leaves the encoder inputs unchanged, that both arms start from the same new head weights, and that a resumed data stream reproduces the same draws. `replay.py` recomputes every stored test score from the saved predictions to 1e-6, and reruns one selected checkpoint on the GPU to 1e-5.

`python -m paper_tables --only simulation_efficiency --check` rebuilds Figure 1 (as a table), the thresholds and savings, and the Appendix C numbers.

## Outputs → results

| File | Written by | Used for |
|---|---|---|
| `results/simulation_efficiency/drivaerml/summary.json` | DrivAerML `analyse.py` | Figure 1 (middle panel); gains of 25–27% and 17.5%; 0.181 against 0.219 at 400 cars; 6 of 30 fits still improving |
| `results/simulation_efficiency/drivaerml/thresholds.json` | DrivAerML `analyse.py` | 200 against 400 (50% fewer, 46.1% including validation); 100 against 200 at 0.3; at 0.2 only AREALS reaches the target; bootstrap fractions |
| `results/simulation_efficiency/drivaerml/replay.json` | DrivAerML `replay.py` | Replay of 3,000 scored rows |
| `results/simulation_efficiency/windsor_drivaernet/summary.json` | each dataset's `analyse.py` summary | Figure 1 (left and right panels); thresholds at 0.2, 0.25 and 0.3; savings including validation; gains; still-improving counts of 0 and 17 |
| `results/simulation_efficiency/windsor_drivaernet/fixed_accuracy.csv` | the nRMSE 0.25 endpoints of the same summaries | 35 against 137 (74.5%, 59.3%); 1,623 against 3,246 (50%, 40.3%) |
| `results/simulation_efficiency_extra/{windsor,drivaernet}/replay.json` | `replay.py` | Replay of 2,100 and 50,880 scored rows |

## Compute

Every fit ran on one NVIDIA B200. Summed fit time, excluding geometry pretraining, was 3.3 GPU-hours on WindsorML, 3.2 on DrivAerML and 17.7 on DrivAerNet++. DrivAerML fits used 8 CPUs and 48 GB. WindsorML and DrivAerNet++ fits used 8 CPUs and 160 GB, because the fields and model inputs are held in memory.

## Notes

- `prepare_data.py` imports `RangeFile` and `client` from a module `storage`, which is not included. That module read the public WindsorML HDF5 files (Hugging Face revision `8a6ca32a…`) from the original object store. A replacement must provide `client()` (an S3-style client) and `RangeFile(client, bucket, key, size)` (a seekable read-only file exposing `etag` and `transferred`). DrivAerML and DrivAerNet++ do not need it.
- `windsor_drivaernet/summary.json` and `fixed_accuracy.csv` were assembled by copying each dataset's `analyse.py` summary unchanged; that small script is not included. Neither is the step that wrote the Figure 1 data tables from the summaries. The table module reproduces all 90 plotted values.
- `prepare_data.py` and `setup_dataset.py` still contain code paths for suite datasets the paper does not report.
- Section 4.2 compares the 8,000-update budget with the 40-epoch schedule of Table C.5 (`08_expanded_benchmarks`). The 1,668-mesh prior that reaches 0.25 with 35 WindsorML cars is from `03_pretraining_targets` (Table M.2).
- WindsorML and DrivAerML are CC BY-SA 4.0; DrivAerNet++ is CC BY-NC 4.0. Results can differ in the last digits across GPUs and library versions.
