# 01 Production geometry pretraining

Builds the Objaverse-XL training pool (Sketchfab subset: 61,822 training meshes, with 1,844 validation and 1,825 test meshes held out) and pretrains the production prior, the masked areal-token autoencoder in `src/areal_tokens` (0.6.0, 3.2M parameters). The paper's checkpoint is the epoch-11 `best_normal.pt` (10,626 updates). With 60% of each surface hidden, it reconstructs the validation shapes with distance nRMSE 0.098 and normal-line error 22.0°.

Paper: Section 3.2, Appendix E, Table A.1 ("Production"), Table N.1.

## Contents

| Folder | Workspace directory | What it does |
|---|---|---|
| `objaverse_base_corpus/` | `home/objaverse_base_030_20260924T172527Z` | 90/5/5 split of the Objaverse-XL metadata (`scripts/build_manifest.py`); mesh preparation with `geom_latents` 0.3.0 (`geometry_latents_minimal/`); the 35,493-mesh base snapshot (`pilots/prepared_35493_5ep/make_snapshot.py`) |
| `objaverse_corpus_expansion/` | `home/objaverse_add30000_20260927T140652Z` | 30,000 more training meshes: GPU preparation workers (NVIDIA Warp closest-triangle queries, checked against libigl), the coordinator that merges them and rejects exact copies of held-out geometry, and publication of the pool |
| `geometry_pretraining/` | `home/areal_objaverse_060_61822_15ep_20260928T121633Z` | Six-GPU training with a fixed global batch of 64 (split 11/11/11/11/10/10 across ranks); recipe in `config.json` and `scale_recipe.json` |

## Run

Run `python scripts/setup_workspace.py` from the repository root first. All steps use one Python environment with torch, numpy, scipy, trimesh, rtree, numba and libigl; step 3 also needs warp-lang. The recorded run used Python 3.12, torch 2.11 (CUDA 12.8) and numpy 2.5. Put the Objaverse-XL Sketchfab download, with its `metadata.csv`, at `/workspace/data/objaverse_xl`, and set `OMP_NUM_THREADS=1` for the preparation steps.

1. **Split and prepare the base corpus** (CPU, 24 workers):

   ```bash
   cd /workspace/home/objaverse_base_030_20260924T172527Z
   export PYTHONPATH=$PWD/geometry_latents_minimal/src
   python scripts/build_manifest.py --dataset /workspace/data/objaverse_xl --out manifests/meshes.json
   python -m geom_latents prepare-mesh --manifest manifests/meshes.json \
       --out /workspace/runs/objaverse_base_030_20260924T172527Z/prepared \
       --points 16384 --queries 32768 --backend igl --workers 24 --skip-bad --resume
   ```

2. **Snapshot the base corpus:**

   ```bash
   python pilots/prepared_35493_5ep/make_snapshot.py
   ```

   This takes the first 35,493 prepared files by manifest index and validates them. Two meshes with zero-normal surface samples are rejected. That leaves 35,491 meshes in `clean.json` (31,822 train, 1,844 val, 1,825 test), with the validation report in `audit.json`. The recorded run took the snapshot while preparation was still running.

3. **Expand the pool to 61,822 training meshes** (three GPUs, eight workers per GPU):

   ```bash
   cd /workspace/home/objaverse_add30000_20260927T140652Z
   export PYTHONPATH=/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/areal_tokens/src:/workspace/home/objaverse_base_030_20260924T172527Z/geometry_latents_minimal/src
   python qualify_gpu.py                          # prepares one new mesh on the GPU and writes qualification.json
   python run_gpu_group.py --first-workers 0 8    # workers 0-15 on two GPUs
   python run_group_and_finish.py                 # workers 16-23 on a third GPU, and finish_preparation.py
   python publish_pool.py                         # once results.json exists; CPU only
   ```

   Run each worker group on its own GPUs (for example with `CUDA_VISIBLE_DEVICES`); `run_gpu_group.py` numbers the GPUs it sees from 0. Each worker reads a task list `tasks_NN.json` of training-split candidates outside the snapshot and stops at 1,250 meshes. Of the 30,000 meshes added, 22,686 reuse files written in step 1 and 7,314 are prepared afresh. `publish_pool.py` writes the pool to `/workspace/runs/objaverse_training_pool/versions/20260927T140652Z/`.

4. **Train** (one node with six GPUs):

   ```bash
   cd /workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z
   export PYTHONPATH=$PWD/areal_tokens/src
   python -m torch.distributed.run --standalone --nproc_per_node=6 train_six.py
   ```

   Training starts from seed-17 random weights and validates on the 1,844 validation meshes after every epoch with a fixed 60% mask. It writes `run/metrics.jsonl`, `run/best.pt` (lowest validation nRMSE), `run/best_normal.pt` (lowest validation normal error), `run/last.pt`, `validation/epoch_NN.json` and `progress.json`. The schedule is a 15-epoch cosine (966 updates per epoch, 14,490 in total). The recorded run was stopped during epoch 12, after 10,875 updates.

The parameter count can be checked without data:

```bash
python -c "from areal_tokens.config import ExperimentConfig as C; from areal_tokens.model import GeometryAutoencoder as M; print(sum(p.numel() for p in M(C.load('results/pretraining/config.json').model).parameters()))"
```

It prints `3214927`.

## Outputs → results

| File | Produced by | Used for |
|---|---|---|
| `results/pretraining/metrics.jsonl` | `train_six.py` (`run/metrics.jsonl`) | Per-epoch validation. The epoch-11 row has nRMSE 0.0985, normal error 21.98° and 10,626 updates, and is the argmin of normal error |
| `results/pretraining/checkpoints.json` | Metadata read from the three checkpoints | Epoch, step, config, runtime (6 × B200, global batch 64) and parameter count of `best_normal.pt`, `best.pt` and `last.pt` |
| `results/pretraining/config.json` | `geometry_pretraining/config.json` | Recipe in Appendix E |
| `results/pretraining/progress.json` | `train_six.py` (`progress.json`) | Stop point: epoch 12, 10,875 of 14,490 planned updates |
| `results/pretraining_extra/pool_ready.json` | `publish_pool.py` | Pool splits 61,822 / 1,844 / 1,825 and the expansion counts |

`python -m paper_tables --only pretraining --check` rebuilds the two pretraining tables, for the run and for the recipe.

## Compute

- **Training:** one node with six NVIDIA B200 GPUs, 64 CPUs and 384 GB. The run took 3,841 s of wall-clock time to the stop at 10,875 updates.
- **Preparation:** the base corpus used 24 CPU workers. The expansion used 24 Warp workers over three GPUs (Warp 1.17.0). Preparation times were not recorded.

## Notes

- Not included: the script that wrote the per-worker task lists `tasks_NN.json`. The candidates are meshes already prepared in step 1, plus untouched source indices from 61,889 onwards (see `objaverse_corpus_expansion/config.json`).
- The recorded expansion and pool publication ran with `areal_tokens` 0.5.0, which is not included. These scripts use only `read_manifest`, `write_json`, `inspect_record` and `load_audit`, which 0.6.0 also provides.
- The epoch with the lowest validation nRMSE (epoch 7, `best.pt`) is not the paper's checkpoint. The paper selects by normal error, as `train_six.py` does when it writes `best_normal.pt`.
