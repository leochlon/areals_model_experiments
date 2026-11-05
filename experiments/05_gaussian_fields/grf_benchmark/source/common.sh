#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/workspace/runs/areals_inversion_20261024
DD_ROOT=/workspace/runs/areals_multicondition_wings_20261023
export PATH=/workspace/home/objaverse_base_030_20260924T172527Z/.venv/bin:$PATH
export AREALS_SRC=/workspace/home/areal_objaverse_060_61822_15ep_20260928T121633Z/areal_tokens/src
export PYTHONPATH=$TASK_ROOT/kit:$AREALS_SRC:/workspace/home/objaverse_base_030_20260924T172527Z/geometry_latents_minimal/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export NUMBA_CACHE_DIR=$TASK_ROOT/cache/numba
export MPLCONFIGDIR=$TASK_ROOT/cache/matplotlib
export CUDA_CACHE_PATH=$TASK_ROOT/cache/cuda
export TMPDIR=/tmp/areals_inversion_$$
mkdir -p "$TMPDIR" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$CUDA_CACHE_PATH"
export TASK_ROOT DD_ROOT
