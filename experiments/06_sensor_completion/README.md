# 06 Sensor completion with the operating condition unreported

Can a few pressure sensors stand in for an unreported operating condition? On DoubleDelta wings at nine angles of attack, the AREALS sensor model is fine-tuned from the production geometry checkpoint and compared with three models: the same architecture trained from random initialisation, a shape-only reference and a true-angle reference. The experiment also covers explicit inference of the angle (Equation 2), angles held out of training, and the controls for these studies. GABI's own training and Tables D.3 and D.5 are in `07_gabi_baseline`.

Paper: the nine-angle row of Table 2; Tables D.1, D.2, D.4 and D.6; Figure D.1; the DoubleDelta numbers in the abstract and Sections 1, 4.4 and 5; Appendices D and E (from-scratch baseline, fine-tuning seeds, explicit inference, held-out-angle split); the nine-angle and held-out-angle parts of Appendix H; the Table A.1 rows "Unreported angle", "Held-out angles" and "Explicit condition inference".

## Contents

| Folder | Workspace | What it does |
|---|---|---|
| `single_seed_study/` | `/workspace/runs/areals_multicondition_wings_20261023` | Prepares DoubleDelta (2,160 / 144 / 144 cases; 228 / 16 / 16 wings), then trains and scores the seed-0 measurement model and the shape-only and true-angle references (`study.py`, `summarize.py`, `controls.py`) |
| `scratch_baseline/` | `/workspace/runs/areals_doubledelta_scratch_20261023` | The same measurement model trained from random initialisation, seed 0 (`compare.py`) |
| `three_seeds/` | `/workspace/runs/areals_nine_angle_seeds_20261029` | Fine-tuning seeds 1 and 2, the three-seed comparison with GABI, and an independent rescoring of all 6,912 AREALS predictions |
| `condition_inference/` | `/workspace/runs/areals_condition_inference_20261029` | Explicit condition inference with the true-condition model on the 18–19° split, FP32 re-evaluation of AREALS, and timing (Table D.1) |
| `heldout_angles/source/` | `/workspace/runs/areals_architecture_controls_20261024` (shared with other studies) | Training on odd angles and testing on even angles (Table D.6), with the hidden-value and measurement-response checks |

## Inputs

Every step uses `src/areal_tokens` and the `native` field harness (`shared/field_benchmark/native/field_benchmark.py`); `scripts/setup_workspace.py` places both where each study imports them. The fine-tuning steps also need the production geometry checkpoint; the commands below expect `BASE=/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/run/best_normal.pt`. Further inputs are listed with each step. One Python environment with torch, numpy and scipy serves every step; `single_seed_study/prepare.py` also needs h5py.

## Run

**1. Seed-0 models and references** (`single_seed_study/`). Data preparation is CPU only; training uses one GPU per role.

```bash
cd /workspace/runs/areals_multicondition_wings_20261023
export PYTHONPATH=/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/areal_tokens/src
python prepare.py          # per-family exact common surface grids: fields/, geometry/, receipts/
python finalize_data.py    # groups identical geometries, split (rng 20261023), manifest.json, study_protocol.json
for split in val test; do
  python study.py --role measurements observations --manifest $PWD/manifest.json --fields $PWD/fields --bench $PWD/bench \
      --split $split --patch-counts 64 --hidden .25 .6 .9 --sensors 5 20 100 500 --unobserved
done
python study.py --role measurements check-setup   # optional: data and initialisation checks, writes setup_checks.json
for role in geometry measurements conditioned; do
  python study.py --role $role train-areal --bench $PWD/bench --checkpoint $BASE --out $PWD/bench/models/$role \
      --epochs 40 --batch-size 8 --queries 2048 --lr 1e-4 --new-lr 1e-3 --mix-unobserved .1 --mix-sensors .45 \
      --workers 0 --select-cars 36 --eval-batch-size 8
  for split in val test; do
    final=; if [ $split = test ]; then final=--final; fi
    python study.py --role $role predict --bench $PWD/bench --split $split $final --method areal --name $role \
        --model $PWD/bench/models/$role/field_best.pt
  done
  python study.py --role $role check --bench $PWD/bench --split val --method areal --name $role \
      --model $PWD/bench/models/$role/field_best.pt
done
python study.py --role measurements score --bench $PWD/bench --split val --reference geometry
python study.py --role measurements score --bench $PWD/bench --split test --final --reference geometry
python summarize.py        # results/{val,test}_grouped.json; also runs ambiguity_bound.py
python controls.py         # measurement-swap controls (GPU)
```

The recorded run trained the three roles at the same time, one GPU each. `results/test_grouped.json` holds the references and the seed-0 gap closure G, with a 4,000-draw bootstrap over physical wings.

**2. From scratch** (`scratch_baseline/`, one GPU). The workspace gets copies of `study.py` and the native harness. Copy step 1's `bench/config.json`, `bench/observations/` and `bench/predictions/{geometry,measurements,conditioned}/` into `bench/` first (the recorded run did this by hand; `compare.py` also scores GABI predictions if they are present).

```bash
cd /workspace/runs/areals_doubledelta_scratch_20261023
python initialisation_check.py
python study.py --role measurements train-areal --bench $PWD/bench --checkpoint $BASE --out $PWD/bench/models/areal_scratch \
    --scratch --epochs 40 --batch-size 8 --queries 2048 --lr 1e-4 --new-lr 1e-3 --mix-unobserved .1 --mix-sensors .45 \
    --workers 0 --select-cars 36 --eval-batch-size 8 --seed 0 --amp bf16 --normalise dataset --floor .5 \
    --visible-weight .25 --weight-decay .05
for split in val test; do
  final=; if [ $split = test ]; then final=--final; fi
  python study.py --role measurements predict --bench $PWD/bench --split $split $final --method areal \
      --name areal_scratch --model $PWD/bench/models/areal_scratch/field_best.pt
done
python study.py --role measurements check --bench $PWD/bench --split val --method areal --name areal_scratch \
    --model $PWD/bench/models/areal_scratch/field_best.pt
python compare.py          # results/{val,test}_grouped.json and an independent rescoring of every prediction
```

**3. Three fine-tuning seeds** (`three_seeds/`). Needs step 1's outputs and the three-seed GABI study of `07_gabi_baseline` at `/workspace/runs/gabi_graph_ablation_20261028`.

```bash
cd /workspace/runs/areals_nine_angle_seeds_20261029
python prepare.py              # copies step 1's bench, seed-0 model and predictions; writes protocol.json
python run.py pilot            # replays seed 0 on two validation cases and runs the hidden-value check (GPU)
for seed in 1 2; do
  python run.py train --seed $seed       # 40 epochs, 10,800 updates (GPU)
  python run.py evaluate --seed $seed    # predictions on val and test, hidden-value check (GPU)
done
python analyze.py              # results/test_comparison.json, results/verification.json
```

**4. Explicit condition inference** (`condition_inference/`). Needs the 18–19° DoubleDelta and BlendedNet++ datasets and models of `08_expanded_benchmarks` at `/workspace/runs/areals_existing_data_expansion_20261025/datasets/{doubledelta,blendednet}/`, and steps 1 and 3 for the nine-angle entries.

```bash
cd /workspace/runs/areals_condition_inference_20261029
for dataset in nine_angle doubledelta blendednet; do python prepare.py --dataset $dataset; done   # CPU
python inference_pilot.py      # interface, synthetic-recovery and label-access checks (GPU)
for split in val test; do
  for seed in 0 1 2; do python condition_inference.py --dataset doubledelta --split $split --index $seed; done
done
python verify_predictions.py --dataset doubledelta
python analyse.py --partial-inference
for i in $(seq 0 8); do python matched_fp32.py --index $i; done     # FP32 re-evaluation of AREALS
python verify_matched_fp32.py --partial
for seed in 0 1 2; do python winner_times.py measure --index $seed; done
python winner_times.py summarise
```

The validation runs choose the mixture temperature per seed, domain and sensor count (`inference_selection/`); the test runs use it. Solver settings are in `inference_protocol.json`: a 0.25° angle grid for DoubleDelta, and for BlendedNet++ 256 Sobol starts, the best 16 refined with Adam (200 steps) and bounded L-BFGS (40 steps).

**5. Held-out angles** (`heldout_angles/`). Needs step 1's manifest, fields and benchmark directory.

```bash
cd /workspace/runs/areals_architecture_controls_20261024
export PYTHONPATH=$PWD/vendor/areal_tokens/src
python source/prepare_audit.py    # unseen_conditions/: odd-angle train/dev (1,200 / 80), even-angle test (64)
bench=$PWD/unseen_conditions/bench
for role in geometry measurements conditioned; do     # one GPU each
  python source/withheld_condition.py --role $role train-areal --bench $bench --checkpoint $BASE \
      --out $bench/models/$role --epochs 72 --batch-size 8 --queries 2048 --lr 1e-4 --new-lr 1e-3 \
      --mix-unobserved .1 --mix-sensors .45 --workers 0 --select-cars 20 --eval-batch-size 8
  for split in val test; do
    final=; if [ $split = test ]; then final=--final; fi
    python source/withheld_condition.py --role $role predict --bench $bench --split $split $final \
        --method areal --name $role --model $bench/models/$role/field_best.pt
  done
  python source/withheld_condition.py --role $role check --bench $bench --split val --method areal --name $role \
      --model $bench/models/$role/field_best.pt
done
python source/summarize_unseen.py        # scores, recomputation, unseen_conditions/results/{val,test}_grouped.json
python source/check_no_angle_search.py   # results/no_angle_search.json: one encoder call, no condition reads
python source/check_condition_geometry.py
```

**6. Tables.** `python -m paper_tables --only sensor_completion --check`

## Outputs → results

| File in `results/` | Written by | Used for |
|---|---|---|
| `sensor_completion/nine_angle/test_comparison.json` | `three_seeds/analyze.py` | AREALS and GABI three-seed means in Table 2, Table D.1 (nine-angle row), Table D.2 (Measurements), Table D.4, Figure D.1 and Section 4.4 |
| `sensor_completion/nine_angle/verification.json` | `three_seeds/analyze.py` | Appendix H: 6,912 predictions, within 4 × 10⁻¹⁴ |
| `sensor_completion/nine_angle/seed{0,1,2}_training_complete.json` | `three_seeds/prepare.py` (seed 0), `run.py train` | Appendix D, fine-tuning seeds |
| `sensor_completion_extra/nine_angle/protocol.json` | `three_seeds/prepare.py` (the recorded copy also lists file hashes and notes) | Appendices D and E: training arguments (batch 8, 36 development cases, learning rates 10⁻⁴ / 10⁻³); Table A.1: 2,160 training cases, 40 epochs |
| `sensor_completion/nine_angle/references_test_grouped.json` | `single_seed_study/summarize.py` | Table 2 Shape and Condition, Table D.2 references, G = 90.7% (84.4–96.9%) |
| `sensor_completion/nine_angle/scratch_test_grouped.json` | `scratch_baseline/compare.py` | Table 2 Scratch; Appendix D 0.2219 / 0.2177 |
| `sensor_completion/condition_inference/winning_comparison.json` | `condition_inference/winner_times.py summarise` | Table D.1, 18–19° rows at 20–500 sensors, with timings |
| `sensor_completion/condition_inference/matched_fp32_areals.json` | `condition_inference/verify_matched_fp32.py` | Table D.1, 18–19° at 5 sensors |
| `sensor_completion/heldout_angles/test_grouped.json` | `heldout_angles/source/summarize_unseen.py` | Table D.6; Section 4.4 (91.7%) |
| `sensor_completion/heldout_angles/control_audit.json` | see Notes | Appendix H: 36 hidden-value checks; 1,200 training cases |
| `sensor_completion_extra/heldout_angles/val_grouped.json` | `heldout_angles/source/summarize_unseen.py` | Appendix E: 80 development cases |
| `sensor_completion_extra/heldout_angles/no_angle_search.json` | `heldout_angles/source/check_no_angle_search.py` | Appendix H, measurement response |

The table module also reads three files stored for other experiments: `sensor_completion/gabi/test_per_problem.csv` (per-problem scores of the seed-0 references, for the true-angle force errors of 1.9% and 1.8%, the shape-only force error of 15.5% and the scored area at 500 sensors), `expanded_benchmarks/blendednet_test_grouped.json` (the BlendedNet++ cell of Table D.1) and `expanded_benchmarks/protocol.json` (the training set and test angles of the 18–19° true-condition model in Table A.1). Ratios, gap closures in nRMSE and gains over the from-scratch model are computed by the module.

## Compute

- Seed-0 study: 3 B200 GPUs (one per role), 32 CPU cores, 160 GB, under 6 h. Data preparation: CPU only, under 2 h.
- From scratch: one B200, under 6 h.
- Seeds 1 and 2: about 16 min each on one B200 (40 epochs, 10,800 updates at batch 8); evaluation about 2 min per seed, analysis 25 s on CPU.
- Condition inference: one B200 per run, under 12 h. The Table D.1 timings are medians on one B200 in FP32.
- Held-out angles: one B200 per role, under 2 h; scoring on CPU, under 40 min.

## Notes

- **Seeds and precision.** AREALS and GABI values are three-seed means. The from-scratch model, the two references and G = 90.7% are seed 0. In Table D.1 the nine-angle row comes from BF16 evaluation and the 18–19° rows from FP32 re-evaluation.
- **Filter in `winning_comparison.json`.** `winner_times.py` keeps only configurations in which the AREALS mean is below both explicit-inference estimates, and records this filter in the file. The unfiltered summary from `analyse.py` (`condition_inference_comparison.json`) also covers the nine-angle split and an extended search domain at 18–19°; it is not among the stored result files.
- **Nine-angle references for seeds 1 and 2.** Explicit inference for nine-angle seeds 1 and 2 reads `legacy_references/`, extra nine-angle reference seeds written by a script that is not included. The paper's nine-angle references are seed 0, and Table D.1 reports no nine-angle explicit inference.
- **`control_audit.json`** was written by a short inline command that is not included. It counts the `PASS` lines in each role's `check` log (12 per role: `--cars 3 --per-car 4`, tolerance 10⁻³) and copies the recomputation difference from `test_grouped.json`.
- **Data preparation.**
  - `single_seed_study/prepare.py` reads the DoubleDelta surface solutions from object storage through a helper module `prepare_automotive_delta` (`s3_client`) that is not included, and the condition metadata from `/workspace/runs/cfd-joint-gno-20261010/data/conditions-training.json`. Files already under `source_fields/` are not downloaded again. `finalize_data.py` imports `prepare.py`, so the helper module must be importable for it too.
  - `prepare.py` and `finalize_data.py` read only the `schema` field of `/workspace/runs/areal_drivaerml_420_epoch14_20260927T162928Z/manifest.json`.
  - Several DoubleDelta wing families share one physical geometry. `finalize_data.py` groups them by a hash of the sorted surface points and normals, so a geometry never crosses splits, and case IDs are hashes of the source IDs.
  - `heldout_angles/source/prepare_audit.py` also prepares inputs for other studies in the same workspace and reads their manifests (including the manifest above). The held-out-angle design: training and development on 11, 13, 15, 17 and 19°, test on 12, 14, 16 and 18°, 1,200 training cases, 72 epochs, batch 8, 10,800 updates.
- **Numbers quoted from tables and bounds.** Appendix D quotes the Table D.1 entries (0.2060, 0.1775, 0.1872) at three decimals; the table module checks these against the printed table. The held-out-angle recomputation agrees with the recorded scores to 1.1 × 10⁻¹⁶ (test) and 1.7 × 10⁻¹⁶ (development), inside the 10⁻¹² tolerance in `control_audit.json` and the bound stated in Appendix H.
- **Figure D.1** is drawn from coordinates entered by hand in its source; the table module checks them against the stored files.
