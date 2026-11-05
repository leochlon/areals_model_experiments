#!/usr/bin/env bash
set -euo pipefail
export STUDY_ROOT=/workspace/runs/areals_production_mechanism_20261025
export TASK_ROOT=/workspace/runs/areals_inversion_20261024
export PRODUCTION_ROOT=/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z
export AREALS_SRC=$PRODUCTION_ROOT/areal_tokens/src
export PATH=/workspace/home/objaverse_base_030_20260924T172527Z/.venv/bin:$PATH
export PYTHONPATH=$STUDY_ROOT/source:$TASK_ROOT/kit:$AREALS_SRC:/workspace/home/objaverse_base_030_20260924T172527Z/geometry_latents_minimal/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NUMBA_CACHE_DIR=$STUDY_ROOT/cache/numba
export MPLCONFIGDIR=$STUDY_ROOT/cache/matplotlib
export CUDA_CACHE_PATH=$STUDY_ROOT/cache/cuda
export TMPDIR=/tmp/areals_prod_$$
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
