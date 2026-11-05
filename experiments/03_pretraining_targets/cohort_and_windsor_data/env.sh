export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export OPEN3D_DISABLE_WEB_VISUALIZER=true CUBLAS_WORKSPACE_CONFIG=:4096:8
COMPARISON_ROOT=/workspace/runs/objaverse_target_comparison_20261030
GEOMETRY_VENDOR=/workspace/runs/areals_drivaerml_geometry_physics_20261030/vendor
export PYTHONPATH="$COMPARISON_ROOT/source:$COMPARISON_ROOT/vendor:$GEOMETRY_VENDOR/poisson_packages:$GEOMETRY_VENDOR/packages"
export LD_LIBRARY_PATH="$COMPARISON_ROOT/vendor/system/usr/lib/x86_64-linux-gnu:$GEOMETRY_VENDOR/system/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
PYTHON=/workspace/home/objaverse_base_030_20260924T172527Z/.venv/bin/python
