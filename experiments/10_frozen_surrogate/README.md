# 10 Frozen areal tokens as a surrogate's geometry input

Do areal tokens help a surrogate that was not built around them? The frozen AREALS geometry encoder replaces the context encoder of the public AeroJEPA tutorial surrogate (NVIDIA PhysicsNeMo). Everything else follows the tutorial configuration: 20 epochs (1,060 updates) on the 420 DrivAerML training cars, training-only normalisation and final weights. The score is the mean per-car relative L2 error on 2,048 fixed pressure queries of the 16 validation cars. The same driver also reads frozen Utonia and Point-MAE features, and AREALS at larger token and point budgets. The geometry prior is the earlier 61,822-mesh checkpoint selected at epoch 14, not the production epoch-11 checkpoint.

Paper: Table L.1, the Appendix L text, the "Frozen surrogate" paragraph of Appendix E, the "Frozen surrogate" row of Table A.1 and the 0.9% in Section 4.2. The 48-car, three-seed replication quoted in Appendix L reuses this folder's driver but is trained and scored in `09_reserved_cars`.

## Contents

- `aerojepa_surrogate/` (workspace `/workspace/home/areal_aerojepa_20260928`):
  - `run_field.py`: the training driver. It builds the tutorial model from PhysicsNeMo's `conf/model/aerojepa.yaml` and `conf/training/superwing.yaml`, swaps in a frozen context encoder for `--arm areal|utonia|pointmae`, trains with the tutorial's `_run_epoch` and scores the final weights (`final.json` → `final_live_rl2`). This is the final revision, which the 48-car replication also imports; the revisions behind each table row are in `table_l1_revisions/`.
  - `areal_context.py`: `ArealContextEncoder`, the frozen AREALS encoder with a trainable projection to AeroJEPA's 128-wide tokens.
  - `drivaerml.py`: `DrivAerMLDataset`, DrivAerML surface pressure in AeroJEPA's dataset interface, with fixed validation queries.
  - `extend_anchors.py`: extends the 256 farthest-point anchors to more tokens without reordering the input.
  - `feature_context.py`: frozen Utonia and Point-MAE features used as context tokens.
  - `extract_backbone_tokens.py`: Utonia and Point-MAE features at the 1,024 anchors. Point-MAE reads 1,024 points in 64 groups, and its features are interpolated to the anchors (the dagger in Table L.1).
  - `prepare_drivaerml.py`: writes `data/<car>.npz` and `data_ready.json` (420 training and 16 validation cars).
  - `prepare_dense_context.py`: denser geometry input; each size extends the previous one (16,384, then 32,768, then 65,536 points).
  - `prepare_dense_anchors.py`: farthest-point anchors on the dense input, 10,000 first and then 32,768.
  - `table_l1_revisions/`: the earlier revisions of `run_field.py`, `areal_context.py`, `drivaerml.py`, `extend_anchors.py`, `prepare_dense_context.py` and `prepare_dense_anchors.py` that produced the Table L.1 rows (mapping below).
- `earlier_prior/` (workspace `/workspace/home/areal_objaverse_scratch61822_15ep_20260927T152147Z`): training of the frozen geometry checkpoint.
  - `run_pipeline.py` runs `train_six.py` on six GPUs: 15 epochs on 61,822 meshes (fixed global batch 64, 14,490 updates), saving `run/best_normal.pt` at the epoch with the lowest validation normal error, epoch 14.
  - `config.json`, `scale_recipe.json` and `benchmark_protocol.json`: the recipe. The setup also places `fixed_global_batches.py` and `benchmark_support.py` from `01_pretraining/geometry_pretraining` here.

## Inputs

`scripts/setup_workspace.py` copies the two folders to their workspace directories. No `shared/field_benchmark` version is used. Supply in addition:

- **Python environment** at `/workspace/home/areal_aerojepa_20260928/.venv`: Python 3.12.14, `torch==2.11.0` and `torchvision==0.26.0` (CUDA 12.8 wheels), and `numpy scipy numba timm jaxtyping hydra-core warp-lang nvtx treelib requests GitPython fsspec s3fs h5py cftime tensordict zarr packaging importlib-metadata psutil einops termcolor tensorboard trimesh pytest`, as in the recorded install.
- **PhysicsNeMo**, cloned into the study directory at the commit that every `run.json` records:
  ```bash
  cd /workspace/home/areal_aerojepa_20260928
  git clone https://github.com/NVIDIA/physicsnemo.git physicsnemo
  git -C physicsnemo checkout 426f7552da4b4fa675e404e8a4f437e27681b668
  ```
- **`areal_tokens` 0.5.0** at `/workspace/home/areal_objaverse_050_20260927T090929Z/areal_tokens/src` (not included; see Notes).
- **Inputs from other experiments:**
  - `/workspace/runs/areal_drivaerml_420_epoch14_20260927T162928Z/manifest.json`: the 420/16 split from `09_reserved_cars/drivaerml_420_split/prepare_420.py`;
  - `/workspace/home/drivaerml_transfer_pilot_20260925T121525Z/` (`09_reserved_cars/drivaerml_pilot_split/`): its `manifest.json` (validation ids), `config.json` (source prefix) and `data/<car>.npz` (the 2,048 fixed validation queries);
  - `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/field_utonia_compare_420/`, from a separate 420-car frozen-feature comparison that is not included (`09_reserved_cars/native_field_preparation/` supplies only its `common.py` and `config.json`): `cache_ready.json` and `cache/<car>.npz` (the prepared surfaces, up to 4,096 points each with 256 anchors), and `utonia_source/`, `weights/utonia.pth` and `.venv` for the Utonia features;
  - `/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/field_pointmae_released_420/extract_features.py`: the released Point-MAE encoder loader (`11_architecture_ablations/decoder_transfer/pointmae_released/`).
- **Frozen prior** at `/workspace/home/areal_objaverse_scratch61822_15ep_20260927T152147Z/run/best_normal.pt` (step 1).

## Run

**1. Frozen prior** (optional; one node with six GPUs; the checkpoint is not included). Needs the 61,822-mesh pool at `/workspace/runs/objaverse_training_pool/versions/20260927T140652Z/` and the environment and `geometry_latents_minimal` under `/workspace/home/objaverse_base_030_20260924T172527Z/` from `01_pretraining`:

```bash
cd /workspace/home/areal_objaverse_scratch61822_15ep_20260927T152147Z
PYTHONPATH=/workspace/home/areal_objaverse_050_20260927T090929Z/areal_tokens/src:/workspace/home/objaverse_base_030_20260924T172527Z/geometry_latents_minimal/src \
    /workspace/home/objaverse_base_030_20260924T172527Z/.venv/bin/python run_pipeline.py
```

**2–7. Data and the Table L.1 runs** (each run on one node with two GPUs):

```bash
cd /workspace/home/areal_aerojepa_20260928
export PYTHONPATH=$PWD:$PWD/physicsnemo:$PWD/physicsnemo/examples/cfd/external_aerodynamics/aerojepa:/workspace/home/areal_objaverse_050_20260927T090929Z/areal_tokens/src
train() { .venv/bin/torchrun --standalone --nnodes=1 --nproc-per-node=2 run_field.py --epochs 20 --batch-per-rank 4 "$@"; }

.venv/bin/python -u prepare_drivaerml.py                 # 2. data
train --arm native                                        # 3. AeroJEPA tutorial (runs/native)
train --arm areal                                         #    AREALS 256 (runs/areal)
train --arm areal --geometry-tokens 1024                  # 4. AREALS 1,024 (runs/areal_1024)

FEATURES=/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/field_utonia_compare_420/.venv
for source in utonia pointmae; do                         # 5. Utonia and Point-MAE
  PYTHONPATH=/workspace/home/areal_objaverse_050_20260927T090929Z/areal_tokens/src \
  LD_LIBRARY_PATH=$FEATURES/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$FEATURES/lib/python3.12/site-packages/nvidia/cuda_runtime/lib \
      $FEATURES/bin/python -u extract_backbone_tokens.py --source $source
  train --arm $source --geometry-tokens 1024              #    runs/utonia_1024, runs/pointmae_1024
done

.venv/bin/python -u prepare_dense_context.py              # 6. dense input: 16,384 points
.venv/bin/python -u prepare_dense_context.py --points 32768 --extend-from $PWD/dense_context_16384
.venv/bin/python -u prepare_dense_context.py --points 65536 --extend-from $PWD/dense_context_32768
train --arm areal --geometry-tokens 1024 --dense-context $PWD/dense_context_65536          # runs/areal_1024_input65536

.venv/bin/python -u prepare_dense_anchors.py --tokens 10000 --workers 12                    # 7. dense anchors
.venv/bin/python -u prepare_dense_anchors.py --tokens 32768 --source $PWD/dense_context_65536 \
    --extend-from $PWD/dense_anchors_10000 --workers 28
train --arm areal --geometry-tokens 32768 --dense-context $PWD/dense_context_65536 \
    --dense-anchors $PWD/dense_anchors_32768                                                # runs/areal_32768_input65536
```

`prepare_drivaerml.py` downloads each car's DrivAerML `scalars.h5` and `prepare_dense_context.py` its `fields.h5` from an S3-compatible copy of the public release; set `S3_ENDPOINT_URL` and the usual AWS credential variables. `prepare_drivaerml.py` skips the download when `scalars/<car>.h5` already exists.

Each run writes `runs/<run>/{run.json, metrics.jsonl, latest.json, best_validation.json, final.json, final_live_validation.json}` and checkpoints. `python -m paper_tables --only frozen_surrogate --check` rebuilds Table L.1, the Appendix L text numbers and the Appendix E / Table A.1 numbers (56 checks).

**Exact revision per row.** The top-level files are the final revision. Each later revision adds arms or options (`--geometry-tokens`, `--dense-context`, `--dense-anchors`, `--areal-checkpoint`, `--run-name`), recorded metadata or a `final_live.pt` checkpoint, so the final revision accepts every command above. Each Table L.1 row was produced by the earlier revision listed here:

| Table L.1 row | Run | `run_field` | `areal_context` | `drivaerml` | Other files |
|---|---|---|---|---|---|
| AREALS 256; AeroJEPA tutorial | `areal`, `native` | `r0` | `r0` | `r0` | — |
| AREALS 1,024 | `areal_1024` | `r1` | `r1` | `r1` | `extend_anchors_r0` |
| Utonia; Point-MAE | `utonia_1024`, `pointmae_1024` | `r2` | not imported | `r1` | `feature_context.py`, `extract_backbone_tokens.py`, `extend_anchors_r0` |
| AREALS 1,024 / 65,536 | `areal_1024_input65536` | `r4` | `r2` | `r2` | `extend_anchors_r0`; first 16,384 points from `prepare_dense_context_r1`, the rest from `prepare_dense_context.py` |
| AREALS 32,768 / 65,536 | `areal_32768_input65536` | `r5` | `r3` | `drivaerml.py` | `extend_anchors.py`; first 10,000 anchors from `prepare_dense_anchors_r0`, the rest from `prepare_dense_anchors.py` |

To rerun a row with its exact code, copy the listed files over the top-level names (for example `cp table_l1_revisions/run_field_r0.py run_field.py`), run the row's command, and restore the top-level files afterwards.

## Outputs → results

| File | Produced by | Used for |
|---|---|---|
| `results/frozen_surrogate/<run>/final.json` | `run_field.py` → `runs/<run>/final.json` | Table L.1 relative L2 (`final_live_rl2`: final weights, mean over the 16 validation cars) |
| `results/frozen_surrogate_extra/<run>/run.json` | `run_field.py` → `runs/<run>/run.json` | Token and point counts, 20 epochs, 1,060 updates, 420/16 cars |
| `results/frozen_surrogate_extra/<run>/final_live_validation.json` | `run_field.py` | Per-car scores (16 cars × 2,048 queries) |
| `results/frozen_surrogate_extra/pointmae_1024/features_ready_metadata.json` | `extract_backbone_tokens.py` → `backbone_tokens/pointmae/features_ready.json` (without its per-car `records`) | Point-MAE row: 1,024 input points and 64 point groups (dagger) |
| `results/frozen_surrogate_extra/earlier_prior/{config.json, metrics.jsonl, training_result.json}` | `earlier_prior/train_six.py` | 256 latent tokens, selection at epoch 14, 61,822 training meshes |
| `results/frozen_surrogate_extra/drivaerml_preparation/config.json` | the DrivAerML surface-sampling configuration that `prepare_420.py` reads (`09_reserved_cars`) | `encoder_points` 4,096, the "≤4,096" points |
| `results/reserved_cars/summary.json` | the 48-car scorer of `09_reserved_cars` | 0.32885 against 0.33193, 0.9%, paired interval [−0.00571, −0.00034] |

## Compute

- **Table L.1 runs:** one node with two NVIDIA B200 GPUs (global batch 8), 16 CPUs (32 for the 32,768-anchor run) and 96 GB; twenty epochs took 257–405 s per run (`elapsed_seconds` in `final.json`).
- **Earlier prior:** one node with six NVIDIA B200 GPUs, 64 CPUs and 384 GB; fifteen epochs (14,490 updates) took about 1,405 s of training.

## Notes

- **`areal_tokens` 0.5.0 is not available.** These names come from it: `areal_tokens.io.load_model` in `areal_context.py`, `areal_tokens.geometry.canonical_cloud` in `prepare_dense_context.py`, `areal_tokens.data.stable_seed` in `extract_backbone_tokens.py`, and the training, data, loss and evaluation modules in `earlier_prior/`. The 0.6.0 package in `src/` provides all of these names, but the recorded runs used 0.5.0 and the two versions have not been compared.
- **Frozen checkpoint.** `run/best_normal.pt` is not included. `run_field.py` copies the frozen encoder's weights before training and records in `final.json` whether they are unchanged afterwards (`frozen_prior_unchanged`); the four AREALS runs report `true`. `table_l1_revisions/areal_context_r0.py` to `_r3.py` assert that the checkpoint is from epoch 14. In every run the frozen encoder is built inside `torch.random.fork_rng`, so the remaining modules start from the same initial weights as the tutorial run.
- **`earlier_prior/`.** The second phase of `run_pipeline.py` runs `finish_benchmark.py`, a test-split comparison with earlier runs that the paper does not report; it is not included, and the checkpoint is written in the first phase.
- **Third-party code and weights.** PhysicsNeMo (the AeroJEPA tutorial), Utonia and the released Point-MAE encoder and checkpoint are not included and keep their own licences.
- **Data, caches and checkpoints** are not included: the DrivAerML data, the prepared caches (`data/`, `dense_context_*`, `dense_anchors_*`, `backbone_tokens/`) and the run checkpoints (`best.pt`, `last.pt`, `final_live.pt`).
- **Unreported runs.** The recorded run also trained `areal_1024_input16384`, `areal_1024_input32768` and `areal_10000_input{16384,65536}` (with `--dense-context` set to the smaller inputs, and `--geometry-tokens 10000 --dense-anchors $PWD/dense_anchors_10000`). `areal_1024_input16384` also rounds to 0.3198; the paper's row is the 65,536-point run.
- **Point-MAE feature-cache metadata.** The Point-MAE row's 1,024 input points and 64 point groups are set in `extract_backbone_tokens.py` and recorded in `backbone_tokens/pointmae/features_ready.json`; the stored copy holds that file's metadata without its per-car `records`.
