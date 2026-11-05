# Reproducibility

## What can be checked without a GPU

Every table in the paper, and the numbers quoted in the text, are rebuilt from the stored result files in `results/`:

```bash
python -m paper_tables --check
```

- **Checked:** 1,740 printed numbers across Tables 1, 2, C.1–C.5, D.1–D.6, E.1, G.1, J.1, J.2, K.1, L.1 and M.1–M.3, Figures 1, 2, B.1–B.3, D.1, G.1 and K.1, and the statistics in Sections 3–4 and the appendices.
- **How they are rebuilt:** most numbers are read directly from the result files. Differences, ratios and bootstrap intervals are recomputed where they are not stored, for example Table M.3 and the hidden-area coverage of Table J.2. Each module's docstring says how.
- **Text that quotes a table:** where the text quotes a table entry at lower precision (for example 52% for Table K.1's 51.5), the check compares the text with the printed table entry.
- **Not rebuilt:** Table I.1 (surfel and symmetry readouts) is listed but not rebuilt, because its evaluation outputs are not included.

## What was used

- **Hardware:** all training and evaluation ran on NVIDIA B200 GPUs. The production prior was pretrained on one node with six B200s; the run took 3,841 s of wall-clock time for 10,875 updates, and the selected checkpoint is at update 10,626. Most field fits used one B200 each. Every experiment README lists the hardware and the recorded run times.
- **Recorded software:** Python 3.12, PyTorch 2.11 (CUDA 12.8), NumPy 2.5 and SciPy 1.18. Some experiments also need torch_geometric (GABI), gudhi (distance tubes), or trimesh, libigl, numba and warp-lang (mesh preparation), as their READMEs state.
- **Datasets:** see [`data/README.md`](data/README.md).

## Rerunning an experiment

Each study ran in place in a fixed directory under `/workspace`, and its scripts refer to those paths. To recreate that layout, run:

```bash
python scripts/setup_workspace.py --python-env /path/to/venv
```

This does three things:
- it copies each study folder, the field-training harness versions (`shared/field_benchmark/`) and the model package to the locations in `scripts/workspace_layout.json`;
- it merges the studies that share a directory;
- with `--python-env`, it links one Python environment at each environment path the experiments expect.

Each experiment README then gives the ordered steps and exact commands. It also lists the inputs that must be supplied:
- prepared datasets;
- the production prior checkpoint, from `experiments/01_pretraining`;
- the outputs of upstream experiments.

The scripts are those that produced the results, with paths generalised. The cluster job files were replaced by the plain commands in each README, and checks that compared file hashes or required earlier check files were removed; the training, evaluation and scoring code is otherwise as run. Hashes that select or seed data (split orders, random seeds, duplicate detection, cache keys) are kept. Some recorded result files still carry file hashes and job identifiers from the original runs.

## Not included

- **Trained weights:** the production prior, the matched and controlled priors, and the fine-tuned field models.
- **Datasets and their prepared forms:** field arrays, geometry caches and observation sets. The preparation code is included; reading the raw files from cloud storage is left to the user, because the original object-store helper was specific to the original infrastructure.
- **Raw outputs:** per-case predictions and run logs.
- **`areal_tokens` 0.5.0:** the earlier package that the frozen-surrogate scripts and part of the corpus expansion import.
- **A few one-off aggregation steps** that were not saved as scripts. Where their outputs feed a table, the table module recomputes them from stored inputs:
  - the Figure 1 data extraction;
  - the Table M.3 statistics;
  - the Table J.2 coverage.

  Each experiment README names its own gaps under Notes.
