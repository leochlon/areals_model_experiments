# 05 Gaussian fields: where the posterior is exact

Gaussian random fields are drawn on 16 held-out DoubleDelta geometry families, eight fresh fields per family. On these fields the posterior mean given any set of observations is known exactly. A field model starts from a geometry trunk, either AREALS-pretrained or the same seed's random initialisation, which is then frozen or fine-tuned for 9,120 updates, with three seeds. Each model is scored by its *efficiency*: the share of the exact posterior mean's improvement over the prior that the model recovers, (R_prior − R_model) / (R_prior − R_Bayes). The score is averaged with equal weight over five observation settings: 5, 20, 100 and 500 sensors, and 32 patches with 60% of the area hidden.

Paper: Section 4.4 ("Where the posterior is exact"); Tables C.2 and C.3, including the Table C.2 caption contrast (frozen pretraining adds 25.7 points, 95% interval 21.6–29.5); the "Gaussian fields" protocol paragraph of Appendix C; the "Gaussian fields" row of Table A.1.

## Contents

| Path | What it does |
|---|---|
| `grf_benchmark/kit/make_grf_fields.py` | Whittle–Matérn random fields from each surface's Laplace–Beltrami eigenbasis. Writes `grf/manifest.json`, `grf/fields/` and the eigenbases `grf/bases/` |
| `grf_benchmark/kit/surface_fields.py` | Random-field library: Laplacians, eigenpairs, sampling and the exact posterior mean |
| `grf_benchmark/kit/grf_eval.py` | Scores predictions against the exact posterior mean (nugget σ = 0.001). Writes per-case prior, Bayes and model risks to `bench/scores/grf_{split}_rows.json` |
| `grf_benchmark/source/common.sh` | Environment of the benchmark steps (`TASK_ROOT`, `DD_ROOT`) |
| `grf_benchmark/source/check_data.py`, `validate_fields.py` | Checks on the DoubleDelta source data (260 distinct shapes, one split and family each, outward unit normals) and on the generated training and validation fields |
| `grf_benchmark/source/grf_entry.py` | `field_benchmark.py` (`train-areal`, `predict`, `check`) with caching of prepared inputs |
| `grf_benchmark/source/check_trained.py` | Weight checks after training: a frozen trunk must be bitwise unchanged, a fine-tuned trunk must change, and the head and the two new input columns must train |
| `field_adaptation/source/prepare_fields.py` | Fresh test fields, eight per held-out family (rng seed 91317), and the test observation sets |
| `field_adaptation/source/audit_field_data.py` | Field-data checks: disjoint splits (by family and by point set), exact regeneration of every fresh field, and agreement of the posterior reference with direct sensor-space Gaussian conditioning (to 1e-4) |
| `field_adaptation/source/check_field_run.py` | Per-fit checks: 9,120 updates, 40 validations, 1,824 training fields, base checkpoint and seed |
| `field_adaptation/source/aggregate.py` | Per-setting efficiency, the equal-weight mean of the five settings, and a 10,000-draw bootstrap that jointly resamples geometry families and seeds (rng seed 91318). Writes `SUMMARY.json` and `RESULTS.md` |
| `field_adaptation/partial_native_frozen/aggregate_native_subset.py` | `aggregate.py` restricted to the three frozen arms; it wrote the stored frozen-trunk summary |

## Inputs

`scripts/setup_workspace.py` places `grf_benchmark/` at `/workspace/runs/areals_inversion_20261024`, with the `kit` harness as `kit/field_benchmark.py`, and `field_adaptation/` at `/workspace/runs/areals_production_mechanism_20261025`. That second directory is shared with `04_masked_vs_unmasked/geometry_pretraining/`, whose `source/common.sh` (setting `STUDY_ROOT` and `TASK_ROOT`) and `source/study.py` the field-adaptation scripts use. Supply in addition:

- the DoubleDelta wings at `/workspace/runs/areals_multicondition_wings_20261023` (`manifest.json`, `fields/`);
- the geometry trunks from `04_masked_vs_unmasked`: `pretraining/<arm>_seed{17,18,19}/last.pt` and `initial.pt`, and for the decoder-reset arms the `derived/` checkpoints of `reset_decoder.py`.

Both `common.sh` scripts put `src/areal_tokens` (at `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/areal_tokens/src`) and the Python environment at `/workspace/home/objaverse_base_030_20260924T172527Z/.venv` on the paths.

## Run

**1. Training and validation fields** (CPU): 228 training families × 8 fields = 1,824, and 16 validation families × 4.

```bash
source /workspace/runs/areals_inversion_20261024/source/common.sh
cd $TASK_ROOT/kit
python $TASK_ROOT/source/check_data.py
python make_grf_fields.py --manifest $DD_ROOT/manifest.json --fields $DD_ROOT/fields --out $TASK_ROOT/grf \
    --draws-train 8 --draws-eval 4 --subset-train 30 --workers 16
python field_benchmark.py observations --manifest $TASK_ROOT/grf/manifest.json --fields $TASK_ROOT/grf/fields \
    --bench $TASK_ROOT/grf/bench --split val --patch-counts 32 --hidden .6 --sensors 5 20 100 500
python field_benchmark.py observations --manifest $TASK_ROOT/grf/manifest_train30.json --fields $TASK_ROOT/grf/fields \
    --bench $TASK_ROOT/grf/bench_train30 --split val --patch-counts 32 --hidden .6 --sensors 5 20 100 500
python $TASK_ROOT/source/validate_fields.py
```

**2. Fresh test fields and field-data checks** (CPU):

```bash
source /workspace/runs/areals_production_mechanism_20261025/source/common.sh
python $STUDY_ROOT/source/prepare_fields.py
python $STUDY_ROOT/source/audit_field_data.py
```

**3. Field adaptation** (one GPU per fit): 10 arms × 3 seeds. Seed `s` uses geometry seed `17 + s`. The first six arms freeze the trunk (only the two new input columns and the field head train, at learning rate 1e-3 without weight decay); the last four fine-tune every weight at 1e-3 with weight decay 0.05. Every fit trains for 40 epochs (9,120 updates) at batch 8 with 2,048 queries, validates every 228 updates on the 64 validation fields and keeps the best validation checkpoint.

```bash
source /workspace/runs/areals_production_mechanism_20261025/source/common.sh
arms=(masked_native_frozen unmasked_native_frozen masked_common_frame_frozen random_frozen
      masked_native_reset_decoder_frozen unmasked_native_reset_decoder_frozen
      masked_native_full unmasked_native_full masked_common_frame_full random_full)
trunk() {   # geometry trunk of arm $1 at geometry seed $2
  case $1 in
    0|6) echo pretraining/masked_native_seed$2/last.pt ;;
    1|7) echo pretraining/unmasked_native_seed$2/last.pt ;;
    2|8) echo pretraining/masked_common_frame_seed$2/last.pt ;;
    3|9) echo pretraining/masked_native_seed$2/initial.pt ;;
    4) echo derived/masked_native_reset_decoder_seed$2.pt ;;
    5) echo derived/unmasked_native_reset_decoder_seed$2.pt ;;
  esac
}
bench=$STUDY_ROOT/bench
for seed in 0 1 2; do
  for i in $(seq 0 9); do
    name=${arms[$i]}_seed$seed
    base=$STUDY_ROOT/$(trunk $i $((17 + seed)))
    out=$STUDY_ROOT/field_models/$name
    freeze=; if [ $i -lt 6 ]; then freeze=--freeze-trunk; fi
    python $TASK_ROOT/source/grf_entry.py train-areal --bench $bench --checkpoint $base --out $out \
        --epochs 40 --validation-every-updates 228 --batch-size 8 --queries 2048 --workers 0 --eval-batch-size 8 \
        --select-cars 64 --seed $seed --device cuda --normalise dataset --lr 1e-3 --new-lr 1e-3 \
        --mix-unobserved .1 --mix-sensors .35 --visible-weight .25 --weight-decay .05 $freeze
    python $TASK_ROOT/source/check_trained.py $out/field_best.pt $out/field_last.pt
    python $STUDY_ROOT/source/check_field_run.py $out $base $seed
    for split in val test; do
      final=; if [ $split = test ]; then final=--final; fi
      python $TASK_ROOT/source/grf_entry.py predict --bench $bench --split $split --method areal \
          --name $name --model $out/field_best.pt --device cuda $final
      python $TASK_ROOT/source/grf_entry.py check --bench $bench --split $split --method areal \
          --name $name --model $out/field_best.pt --device cuda --cars 128 --per-car 5 $final
    done
  done
done
```

The arms in Tables C.2 and C.3 are `masked_native_frozen` (AREALS, frozen), `random_frozen`, `masked_native_full` (AREALS, fine-tuned) and `random_full`. The six-arm readout adds `unmasked_native_frozen` and `unmasked_native_full`. The other four arms are not reported.

**4. Scoring** (CPU), in the same shell:

```bash
methods=$(for seed in 0 1 2; do for arm in "${arms[@]}"; do echo ${arm}_seed$seed; done; done)
for split in val test; do
  python $TASK_ROOT/kit/grf_eval.py --bench $bench --bases $STUDY_ROOT/bases --split $split --methods $methods --sigma .001
done
python $STUDY_ROOT/source/aggregate.py
```

The stored results come from subsets of the same fits. The nine fits of the three frozen arms with masked, unmasked and random trunks were scored on a bench under `partial_native_frozen/` that links `bench/config.json`, the observation files and the nine prediction folders, with `grf_eval.py --bench $STUDY_ROOT/partial_native_frozen/bench --bases $STUDY_ROOT/bases --split {val,test} --methods <9 fits> --sigma .001 --pairs`, followed by `python $STUDY_ROOT/partial_native_frozen/aggregate_native_subset.py`, which wrote `SUMMARY.json`. The same procedure on `partial_native_all/` covers the 18 fits of the six-arm readout. Because `grf_eval.py` scores each method independently and `aggregate.py` draws its bootstrap indices before looping over arms, the full scoring gives the same values for the four arms of Table C.2 and their contrast.

`python -m paper_tables --only gaussian_fields --check` rebuilds Tables C.2 and C.3 from the stored results.

## Outputs → results

| File | Produced as | Used for |
|---|---|---|
| `results/gaussian_fields/mechanism_native_readout.json` | `partial_native_all/readout.json`: `efficiency_percent` (100 × mean of the five test settings) and the equal-weight contrasts | Table C.2: all four cells, and the caption gain and interval; Section 4.4 |
| `results/gaussian_fields/frozen_trunk_summary.json` | `partial_native_frozen/SUMMARY.json`, from `aggregate_native_subset.py` | Table C.3: per-setting test efficiencies and the mean row. Also holds per-seed values, the validation split and the frozen contrasts |
| `results/gaussian_fields_extra/frozen_trunk_readout.json` | `partial_native_frozen/readout.json` | The field-adaptation budget (9,120 updates) |

## Compute

| Step | Resources |
|---|---|
| Training and validation fields | 32 CPUs, 96 GB, under 2 h |
| Fresh test fields | 8 CPUs, 48 GB, under 1 h |
| Field-data checks | 4 CPUs, 16 GB, under 20 min |
| Field adaptation | 1 B200 GPU, 8 CPUs and 48 GB per fit, under 3 h each |
| Scoring | 8 CPUs, 96 GB, under 3 h |

These are the limits the runs were given; actual run times were not recorded.

## Notes

- **Not included:** checkpoints, generated fields, eigenbases, predictions and per-case score rows. The tables are rebuilt from the stored summaries.
- **Readout commands not included.** The commands that assembled the two subset benches and wrote the two `readout.json` files are not included. Each readout's `efficiency_percent` is 100 × the mean of the five per-setting test efficiencies in the corresponding `SUMMARY.json`, and its `contrasts` are that file's equal-weight entries. The six-arm summary is `aggregate.summarise` with the six arms and the pairs among them. The table module confirms that the frozen arms and their contrast agree to every digit between the two stored files.
- **Laplacian backend.** `make_grf_fields.py` builds each surface Laplacian with the optional `robust_laplacian` package when it is installed, and otherwise with a k-nearest-neighbour heat-kernel graph (`surface_fields.point_cloud_laplacian`). The stored outputs do not record which backend built the bases, so regenerated fields may differ numerically from the original run.
- **Shape grouping.** `make_grf_fields.py` and `check_data.py` treat DoubleDelta records whose cloud files are byte-identical as one shape, by comparing file hashes; `grf_entry.py` caches prepared inputs by a hash of the point arrays. `audit_field_data.py` compares point sets and field values across splits by array hashes. These hashes find duplicates; they are not integrity checks.
- **Self-test helper.** The `surface_fields.py` self-test (`python surface_fields.py`) imports an `icosphere` helper that is not included. The benchmark does not use it.
- **The "six times" ratio in Section 4.4** compares the printed Table C.2 entries, 30.4 / 4.7 = 6.47; the unrounded means give 6.51.
