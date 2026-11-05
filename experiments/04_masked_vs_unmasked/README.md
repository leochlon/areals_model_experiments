# 04 Masked against unmasked geometry pretraining

Does masking during geometry pretraining make the prior transfer to pressure fields? The production geometry recipe is pretrained twice from the same initialisation: once with 60% of each surface hidden in patches (`masked_native`, the production setting) and once on complete surfaces (`unmasked_native`, `mask_fraction = 0`), for geometry seeds 17, 18 and 19. Each epoch-11 checkpoint (10,626 updates) is fine-tuned once on DoubleDelta (train 11–16°, test 18–19°) and once on the official DrivAerML split with the same field recipe. Hidden-region nRMSE is compared at 0, 5, 20, 100 and 500 sensors. Intervals jointly resample seeds and geometry families (5,000 draws); the 20-sensor endpoint was chosen before the fits and uses a 97.5% interval.

Paper: Section 4.3 ("Reconstruction, not masking, is what transfers"), Table C.4, and the Table A.1 row "Masked vs. unmasked".

## Contents

- `geometry_pretraining/source/`: pretraining of the arms.
  - `common.sh`: the environment.
  - `setup.py`: copies the production `config.json` and writes `geometry_manifest.json`.
  - `study.py`: the model, the optimiser and the arm settings.
  - `train_geometry.py`: 4-GPU training for 11 epochs (10,626 updates). Writes `initial.pt`, `epoch_NN.pt`, `best_normal.pt`, `last.pt`, `history.json` and `complete.json` to `pretraining/<arm>_seed<seed>/`.
  - `reset_decoder.py`: the decoder-reset checkpoints that two arms of `05_gaussian_fields` start from. This experiment does not use them.
- `mask_pressure_transfer/`: the transfer comparison.
  - `protocol.json`: the field recipe and endpoints.
  - `source/setup.py`: checks the six checkpoints, copies the two benchmarks and writes `inputs.json`.
  - `source/score.py`: test scoring and the joint bootstrap, written to `results.json`.

## Inputs

`scripts/setup_workspace.py` places `geometry_pretraining/` at `/workspace/runs/areals_production_mechanism_20261025` (shared with `05_gaussian_fields/field_adaptation/`) and `mask_pressure_transfer/` at `/workspace/runs/areals_mask_pressure_transfer_20261025`. Supply in addition:

- the production pretraining directory `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z` from `01_pretraining`, with `config.json`, `fixed_global_batches.py` and `src/areal_tokens` at `areal_tokens/src`;
- the 61,822-mesh corpus manifest `/workspace/runs/objaverse_training_pool/versions/20260927T140652Z/manifest.json` and its prepared mesh files (see the note on corpus files below);
- for the transfer, the expanded-benchmark study `/workspace/runs/areals_existing_data_expansion_20261025` from `08_expanded_benchmarks`: its `source/` scripts, the `kit` harness at `kit/field_benchmark.py`, and the prepared datasets `datasets/{doubledelta,drivaerml}/`, each with `manifest.json`, `data_checks.json`, `fields/`, `bench/config.json` and `bench/observations/`.

One Python environment with torch, numpy, scipy and numba (optional, faster geometry preprocessing) serves every step; `common.sh` expects it at `/workspace/home/objaverse_base_030_20260924T172527Z/.venv`.

## Run

**Geometry pretraining** (one node with 4 GPUs per run):

```bash
source /workspace/runs/areals_production_mechanism_20261025/source/common.sh
python $STUDY_ROOT/source/setup.py
for seed in 17 18 19; do
  for arm in masked_native unmasked_native; do
    python -m torch.distributed.run --standalone --nproc-per-node=4 $STUDY_ROOT/source/train_geometry.py --arm $arm --seed $seed
  done
done
python $STUDY_ROOT/source/reset_decoder.py    # only needed for 05_gaussian_fields
```

`train_geometry.py` also accepts `--arm masked_common_frame`, a diagnostic arm (masked inputs normalised in the frame of the complete object). `05_gaussian_fields` uses it; this experiment does not.

**Transfer fits** (one GPU per fit). `common.sh` of the expanded-benchmark study sets `STUDY_ROOT` to that study, whose `field_study.py` runs the fits:

```bash
source /workspace/runs/areals_existing_data_expansion_20261025/source/common.sh
MP=/workspace/runs/areals_mask_pressure_transfer_20261025
GEOMETRY=/workspace/runs/areals_production_mechanism_20261025/pretraining
python $MP/source/setup.py
for dataset in doubledelta drivaerml; do
  fit() { python $STUDY_ROOT/source/field_study.py --dataset $dataset --role measurements "$@"; }
  bench=$MP/datasets/$dataset/bench
  for seed in 0 1 2; do
    for arm in masked unmasked; do
      out=$MP/datasets/$dataset/models/${arm}_seed$seed
      fit train-areal --bench $bench --checkpoint $GEOMETRY/${arm}_native_seed$((17 + seed))/epoch_11.pt --out $out \
          --epochs 40 --batch-size 8 --lr 1e-3 --new-lr 1e-3 --queries 2048 --workers 4 --seed $seed \
          --normalise dataset --mix-unobserved 0.1 --mix-sensors 0.9 --select-cars 64 --eval-batch-size 4 \
          --device cuda --amp bf16
      for split in val test; do
        final=; if [ $split = test ]; then final=--final; fi
        fit predict --bench $bench --method areal --name ${arm}_seed$seed --model $out/field_best.pt \
            --split $split --batch-size 4 --device cuda --amp bf16 $final
        fit check --bench $bench --method areal --name ${arm}_seed$seed --model $out/field_best.pt \
            --split $split --batch-size 4 --device cuda --amp bf16 --cars 3 --per-car 4 $final
      done
    done
  done
done
python $MP/source/score.py
```

Field seed `s` uses geometry seed `17 + s` for both arms. `score.py` checks that the two arms of each seed ran the same number of updates, scores the test split with `field_benchmark score --final`, averages per seed and geometry family, and draws 5,000 joint seed-and-family bootstrap samples (rng 57300).

`python -m paper_tables --only masked_vs_unmasked --check` rebuilds Table C.4 and the protocol numbers.

## Outputs → results

| File in `results/` | Written by | Used for |
|---|---|---|
| `masked_vs_unmasked/results.json` | `score.py` (`results.json` in the transfer study) | Every cell of Table C.4; interval levels; test families; seeds |
| `masked_vs_unmasked/doubledelta_test_per_problem.csv`, `drivaerml_test_per_problem.csv` | `field_benchmark score` inside `score.py` (`datasets/<dataset>/bench/scores/test_per_problem.csv`) | Cross-check of the Table C.4 means and differences |
| `masked_vs_unmasked_extra/pins.json` | the recorded run's `setup.py`, which also stored file hashes; this `setup.py` writes the other fields to `inputs.json` | Geometry seeds, epoch and updates of the six checkpoints; split sizes |
| `masked_vs_unmasked_extra/protocol.json` | the recorded protocol; `mask_pressure_transfer/protocol.json` is a shortened copy | Bootstrap draws; DoubleDelta angle split |

## Compute

- Geometry pretraining: one run per arm and seed on 4 B200 GPUs (64 CPU cores, 192 GB), about 2 hours each including per-epoch validation.
- Transfer fits: 12 fits on one GPU each (8 CPU cores, 96 GB), about 7–9 minutes each.
- Scoring: CPU only (4 cores, 64 GB).

## Notes

- **Not included:** the geometry checkpoints, field weights, predictions and logs.
- **Arm pairing.** Within each seed, both arms share the initialisation, data order, field-head initialisation, update count and checkpoint-selection rule. The geometry checkpoint is epoch 11 for both arms; it is not selected per arm.
- **Batch padding.** `collate_fixed` zero-pads every batch to 4,096 points, 256 anchors and 2,048 surfels with zero weights, so all four ranks run tensors of the same size.
- **Field recipe.** The expanded-benchmark recipe (Appendix C, after Table C.4), with learning rate 10⁻³ for all parameters and the encoder fine-tuned.
- **Corpus files (not included).** `geometry_manifest.json` points each record either to the file listed in an earlier control study's manifest (`/workspace/runs/areals_mask_objective_control_20261025/full_manifest.json`) or to `restored_geometry/<id>.npz`, which a separate script copied from object storage. Neither that script nor the earlier manifest is included. The files are the production corpus records from `01_pretraining`.
- **Intervals are read, not recomputed.** The bootstrap resamples geometry families, which the per-problem tables in `results/` do not label, so the table module reads the intervals from `results.json`. It recomputes the arm means, per-seed means and differences from the per-problem tables; they agree to 1e-16.
